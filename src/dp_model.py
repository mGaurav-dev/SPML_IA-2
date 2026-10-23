"""Differentially Private Regression using IBM Diffprivlib with stability and sensitivity investigation."""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
from diffprivlib.models import LinearRegression as DPLinearRegression

from .utils import calculate_regression_metrics, utility_loss_percentage

logger = logging.getLogger("household_energy_dp")


def compute_training_data_bounds(
    X_train: np.ndarray,
    y_train: np.ndarray,
    percentiles: Tuple[float, float] = (0.5, 99.5),
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[float, float]]:
    """Derives feature and target clipping bounds strictly from the training partition.

    Mathematical Assumption:
        Diffprivlib LinearRegression requires known compact bounds on feature domain X and
        target domain y to bound the global sensitivity of the Gram matrix X^T X and vector X^T y.
        To avoid test leakage, these bounds are computed solely on training observations.

    Args:
        X_train: Training feature matrix.
        y_train: Training target array.
        percentiles: Lower and upper percentiles for clipping extreme sensor outliers.

    Returns:
        Tuple of (bounds_X, bounds_y) formatted for Diffprivlib.
    """
    p_low, p_high = percentiles
    x_min = np.percentile(X_train, p_low, axis=0)
    x_max = np.percentile(X_train, p_high, axis=0)
    y_min = float(np.percentile(y_train, p_low))
    y_max = float(np.percentile(y_train, p_high))

    # Guard against identical bounds
    x_max = np.where(x_max <= x_min, x_min + 1.0, x_max)
    if y_max <= y_min:
        y_max = y_min + 1.0

    bounds_X = (x_min, x_max)
    bounds_y = (y_min, y_max)
    return bounds_X, bounds_y


class DifferentiallyPrivateRegressor:
    """Wrapper around IBM Diffprivlib's LinearRegression with timing and post-processing evaluation."""

    def __init__(
        self,
        epsilon: float,
        bounds_X: Tuple[np.ndarray, np.ndarray],
        bounds_y: Tuple[float, float],
        random_state: Optional[int] = None,
    ) -> None:
        self.epsilon = float(epsilon)
        self.bounds_X = bounds_X
        self.bounds_y = bounds_y
        self.random_state = random_state
        self.fit_time: float = 0.0
        self.predict_time: float = 0.0

        self.model = DPLinearRegression(
            epsilon=self.epsilon,
            bounds_X=self.bounds_X,
            bounds_y=self.bounds_y,
            random_state=self.random_state,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> DifferentiallyPrivateRegressor:
        start = time.perf_counter()
        self.model.fit(X, y)
        self.fit_time = time.perf_counter() - start
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        start = time.perf_counter()
        preds = self.model.predict(X)
        self.predict_time = time.perf_counter() - start
        return preds

    def predict_clipped(self, X: np.ndarray, y_min: float = 0.0, y_max: float = 3.5) -> np.ndarray:
        """Applies DP Post-Processing Theorem: clipping raw outputs preserves (eps, 0)-DP."""
        raw = self.predict(X)
        return np.clip(raw, y_min, y_max)


def evaluate_dp_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    epsilon: float,
    bounds_X: Tuple[np.ndarray, np.ndarray],
    bounds_y: Tuple[float, float],
    base_rmse: float,
    n_repeats: int = 5,
    base_seed: int = 42,
) -> Tuple[Dict[str, any], np.ndarray, List[Dict[str, any]]]:
    """Evaluates DP linear regression over multiple stochastic repetitions to measure variance.

    Returns:
        Tuple of (aggregated metrics dictionary, representative predictions array, per-trial list).
    """
    raw_maes: List[float] = []
    raw_rmses: List[float] = []
    raw_r2s: List[float] = []
    clipped_maes: List[float] = []
    clipped_rmses: List[float] = []
    clipped_r2s: List[float] = []
    fit_times: List[float] = []
    last_preds: Optional[np.ndarray] = None
    trials_detail: List[Dict[str, any]] = []

    y_max_bound = float(bounds_y[1])

    for i in range(n_repeats):
        seed = base_seed + i * 17
        dp_reg = DifferentiallyPrivateRegressor(
            epsilon=epsilon,
            bounds_X=bounds_X,
            bounds_y=bounds_y,
            random_state=seed,
        )
        dp_reg.fit(X_train, y_train)
        raw_p = dp_reg.predict(X_test)
        clipped_p = np.clip(raw_p, 0.0, y_max_bound)

        raw_m = calculate_regression_metrics(y_test, raw_p)
        clipped_m = calculate_regression_metrics(y_test, clipped_p)

        raw_maes.append(raw_m["mae"])
        raw_rmses.append(raw_m["rmse"])
        raw_r2s.append(raw_m["r2"])
        clipped_maes.append(clipped_m["mae"])
        clipped_rmses.append(clipped_m["rmse"])
        clipped_r2s.append(clipped_m["r2"])
        fit_times.append(dp_reg.fit_time)
        last_preds = raw_p

        trials_detail.append({
            "epsilon": epsilon,
            "trial": i + 1,
            "seed": seed,
            "raw_rmse": raw_m["rmse"],
            "raw_mae": raw_m["mae"],
            "raw_r2": raw_m["r2"],
            "clipped_rmse": clipped_m["rmse"],
            "clipped_mae": clipped_m["mae"],
            "clipped_r2": clipped_m["r2"],
            "fit_time_sec": round(dp_reg.fit_time, 4),
        })

    mean_raw_rmse = float(np.mean(raw_rmses))
    std_raw_rmse = float(np.std(raw_rmses))
    mean_raw_mae = float(np.mean(raw_maes))
    std_raw_mae = float(np.std(raw_maes))
    mean_raw_r2 = float(np.mean(raw_r2s))

    mean_clip_rmse = float(np.mean(clipped_rmses))
    std_clip_rmse = float(np.std(clipped_rmses))
    mean_clip_mae = float(np.mean(clipped_maes))
    mean_clip_r2 = float(np.mean(clipped_r2s))

    raw_utility_loss = utility_loss_percentage(base_rmse, mean_raw_rmse)
    clip_utility_loss = utility_loss_percentage(base_rmse, mean_clip_rmse)

    summary = {
        "epsilon": float(epsilon),
        "mean_mae": round(mean_raw_mae, 5),
        "std_mae": round(std_raw_mae, 5),
        "mean_rmse": round(mean_raw_rmse, 5),
        "std_rmse": round(std_raw_rmse, 5),
        "mean_r2": round(mean_raw_r2, 5),
        "utility_loss_pct": raw_utility_loss,
        "mean_clipped_rmse": round(mean_clip_rmse, 5),
        "std_clipped_rmse": round(std_clip_rmse, 5),
        "mean_clipped_mae": round(mean_clip_mae, 5),
        "mean_clipped_r2": round(mean_clip_r2, 5),
        "clipped_utility_loss_pct": clip_utility_loss,
        "n_repeats": n_repeats,
        "avg_fit_time_sec": round(float(np.mean(fit_times)), 4),
        "protected_unit": "Single 30-min record (x_i, y_i)",
        "formal_guarantee": "Record-level pure (epsilon, 0)-Differential Privacy",
    }

    return summary, last_preds, trials_detail


def run_privacy_utility_sweep(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    epsilons: List[float],
    base_rmse: float,
    n_repeats: int = 5,
    random_state: int = 42,
    percentiles: Tuple[float, float] = (0.5, 99.5),
) -> Tuple[List[Dict[str, any]], Dict[float, np.ndarray], List[Dict[str, any]]]:
    """Runs empirical privacy-utility sweep across multiple epsilon values.

    Returns:
        Tuple of (list of summary dicts, dict mapping epsilon to representative predictions, all trials list).
    """
    logger.info("Computing clipping bounds strictly on training data (percentiles=%s)...", percentiles)
    bounds_X, bounds_y = compute_training_data_bounds(X_train, y_train, percentiles=percentiles)

    logger.info("Bounds derived: bounds_y=(%.4f, %.4f)", bounds_y[0], bounds_y[1])

    summaries: List[Dict[str, any]] = []
    predictions_map: Dict[float, np.ndarray] = {}
    all_trials: List[Dict[str, any]] = []

    for eps in epsilons:
        logger.info("Evaluating DP Linear Regression at epsilon=%.3f (%d trials)...", eps, n_repeats)
        summary, preds, trials = evaluate_dp_regression(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            epsilon=eps,
            bounds_X=bounds_X,
            bounds_y=bounds_y,
            base_rmse=base_rmse,
            n_repeats=n_repeats,
            base_seed=random_state,
        )
        summaries.append(summary)
        all_trials.extend(trials)
        if preds is not None:
            predictions_map[eps] = preds

        logger.info(
            "eps=%.2f -> Raw RMSE=%.4f (Std=%.4f, dRMSE=%.1f%%) | Clipped RMSE=%.4f",
            eps,
            summary["mean_rmse"],
            summary["std_rmse"],
            summary["utility_loss_pct"],
            summary["mean_clipped_rmse"],
        )

    return summaries, predictions_map, all_trials
