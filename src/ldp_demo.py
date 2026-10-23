"""Local Differential Privacy (LDP) demonstration using k-ary Randomized Response (k-RR) for IA-2."""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

logger = logging.getLogger("household_energy_dp")


def compute_discrete_bin_thresholds(
    y_train: np.ndarray,
    num_bins: int = 3,
) -> np.ndarray:
    """Computes quantile bin edges strictly on the training partition.

    Args:
        y_train: Continuous energy consumption target from training set.
        num_bins: Number of discrete bins (e.g. 3 for Low, Medium, High).

    Returns:
        Array of internal threshold values (length = num_bins - 1).
    """
    percentiles = np.linspace(0, 100, num_bins + 1)[1:-1]
    thresholds = np.percentile(y_train, percentiles)
    return thresholds


def discretize_feature(
    values: np.ndarray,
    thresholds: np.ndarray,
) -> np.ndarray:
    """Discretizes continuous energy values into integer category indices [0, ..., k-1].

    Args:
        values: Continuous consumption values.
        thresholds: Pre-computed quantile thresholds from training data.

    Returns:
        1D integer array with category indices in range [0, len(thresholds)].
    """
    return np.digitize(values, thresholds)


class KRandomizedResponse:
    """k-ary Randomized Response (k-RR) mechanism for Local Differential Privacy.

    Guarantees epsilon-LDP for any individual report before it leaves the client/meter.

    Probabilities for domain of size k:
        P(Y_tilde = i | Y = i) = p = exp(eps) / (exp(eps) + k - 1)
        P(Y_tilde = j | Y = i) = q = 1 / (exp(eps) + k - 1) for j != i
    """

    def __init__(self, epsilon: float, k: int, random_state: int = 42) -> None:
        if epsilon <= 0:
            raise ValueError(f"Privacy budget epsilon must be strictly positive, got {epsilon}")
        if k < 2:
            raise ValueError(f"Domain size k must be at least 2, got {k}")

        self.epsilon = float(epsilon)
        self.k = int(k)
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)

        # Mechanism transition probabilities
        denom = np.exp(self.epsilon) + self.k - 1
        self.p = float(np.exp(self.epsilon) / denom)
        self.q = float(1.0 / denom)

    def perturb(self, true_categories: np.ndarray) -> np.ndarray:
        """Perturbs true discrete categories using the k-RR mechanism.

        Args:
            true_categories: Integer array with values in [0, k-1].

        Returns:
            Perturbed integer array with values in [0, k-1].
        """
        n = len(true_categories)
        perturbed = np.empty(n, dtype=int)

        draws = self.rng.random(size=n)
        keep_mask = draws < self.p

        perturbed[keep_mask] = true_categories[keep_mask]

        flip_indices = np.where(~keep_mask)[0]
        for idx in flip_indices:
            true_val = true_categories[idx]
            other_categories = [c for c in range(self.k) if c != true_val]
            perturbed[idx] = self.rng.choice(other_categories)

        return perturbed

    def estimate_proportions(self, perturbed_categories: np.ndarray) -> np.ndarray:
        """Computes the mathematically unbiased population frequency estimator:

        Formula:
            hat_pi_i = (tilde_pi_i - q) / (p - q)
        """
        n = len(perturbed_categories)
        counts = np.bincount(perturbed_categories, minlength=self.k)
        empirical_proportions = counts / n

        unbiased_estimates = (empirical_proportions - self.q) / (self.p - self.q)
        return unbiased_estimates


def run_ldp_demonstration(
    y_train: np.ndarray,
    y_test: np.ndarray,
    num_bins: int = 3,
    bin_labels: List[str] | None = None,
    epsilons: List[float] | None = None,
    n_repeats: int = 5,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, any], pd.DataFrame]:
    """Runs end-to-end Local Differential Privacy demonstration with multi-trial variance and downstream utility.

    Returns:
        Tuple of (summary results DataFrame, summary dictionary, all trials DataFrame).
    """
    if bin_labels is None:
        bin_labels = [f"Bin_{i}" for i in range(num_bins)]
    if epsilons is None:
        epsilons = [0.5, 1.0, 2.0, 4.0]

    thresholds = compute_discrete_bin_thresholds(y_train, num_bins=num_bins)
    true_train_categories = discretize_feature(y_train, thresholds)
    true_test_categories = discretize_feature(y_test, thresholds)

    true_counts = np.bincount(true_test_categories, minlength=num_bins)
    true_proportions = true_counts / len(true_test_categories)

    summary_records: List[Dict[str, any]] = []
    trial_records: List[Dict[str, any]] = []

    for eps in epsilons:
        tvds: List[float] = []
        maes: List[float] = []
        downstream_accs: List[float] = []
        downstream_f1s: List[float] = []

        last_p = 0.0
        last_q = 0.0
        last_est_props = None

        for t in range(n_repeats):
            seed = random_state + t * 31
            mech = KRandomizedResponse(epsilon=eps, k=num_bins, random_state=seed)
            perturbed_reports = mech.perturb(true_test_categories)
            est_props = mech.estimate_proportions(perturbed_reports)

            abs_err = np.abs(est_props - true_proportions)
            tvd = float(0.5 * np.sum(abs_err))
            mae = float(np.mean(abs_err))

            tvds.append(tvd)
            maes.append(mae)
            last_p = mech.p
            last_q = mech.q
            last_est_props = est_props

            # Downstream utility evaluation: peak demand classification accuracy
            # Can an analyst infer the binary peak state (top tertile) directly from perturbed reports?
            is_true_peak = (true_test_categories == (num_bins - 1)).astype(int)
            is_perturbed_peak = (perturbed_reports == (num_bins - 1)).astype(int)
            peak_acc = float(accuracy_score(is_true_peak, is_perturbed_peak))
            peak_f1 = float(f1_score(is_true_peak, is_perturbed_peak, zero_division=0))

            downstream_accs.append(peak_acc)
            downstream_f1s.append(peak_f1)

            trial_records.append({
                "epsilon_ldp": float(eps),
                "trial": t + 1,
                "seed": seed,
                "tvd": round(tvd, 6),
                "mae": round(mae, 6),
                "downstream_peak_accuracy": round(peak_acc, 4),
                "downstream_peak_f1": round(peak_f1, 4),
            })

        rec = {
            "epsilon_ldp": float(eps),
            "k_categories": num_bins,
            "p_true": round(last_p, 4),
            "q_other": round(last_q, 4),
            "mean_tvd": round(float(np.mean(tvds)), 5),
            "std_tvd": round(float(np.std(tvds)), 5),
            "mean_mae_proportions": round(float(np.mean(maes)), 5),
            "std_mae_proportions": round(float(np.std(maes)), 5),
            "mean_downstream_accuracy": round(float(np.mean(downstream_accs)), 4),
            "mean_downstream_f1": round(float(np.mean(downstream_f1s)), 4),
            "n_trials": n_repeats,
        }
        for i, label in enumerate(bin_labels):
            rec[f"true_prop_{label}"] = round(float(true_proportions[i]), 4)
            if last_est_props is not None:
                rec[f"est_prop_{label}"] = round(float(last_est_props[i]), 4)

        summary_records.append(rec)

    summary_df = pd.DataFrame(summary_records)
    trials_df = pd.DataFrame(trial_records)

    summary = {
        "num_bins": num_bins,
        "bin_labels": bin_labels,
        "thresholds": [round(float(t), 4) for t in thresholds],
        "test_sample_count": len(y_test),
        "tested_epsilons": epsilons,
        "n_repeats": n_repeats,
    }
    return summary_df, summary, trials_df
