"""Compressive Privacy (Module 5) demonstration: Privacy Funnel vs Naive PCA for IA-2."""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, roc_auc_score

from .utils import calculate_regression_metrics

logger = logging.getLogger("household_energy_dp")


def define_sensitive_attribute(
    y: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Defines binary sensitive attribute: high-consumption appliance/occupancy burst.

    Args:
        y: Target continuous consumption in kWh.
        threshold: Quantile threshold (e.g. 85th percentile derived strictly from train).

    Returns:
        1D binary integer array (1 for sensitive burst, 0 otherwise).
    """
    return (y >= threshold).astype(int)


class CompressivePrivacySubspaceProjector:
    """Implements linear Compressive Privacy via Orthogonal Subspace Projection (Privacy Funnel).

    Mathematical Foundation:
        Given input features X in R^d, sensitive attribute S in {0, 1}, and utility target Y:
        Standard PCA maximizes variance Tr(W^T Sigma_X W), concentrating high-energy burst signatures.
        In contrast, Compressive Privacy solves the Privacy Funnel:
            min I(Z; S)  subject to  I(Z; Y) >= R

        In the linear setting, we identify the sensitive discriminant vector:
            v_S = (X_train - mu_X)^T S_train / ||...||
        and project X onto the orthogonal null space:
            P_perp = I - v_S v_S^T
        Then we extract k principal components of the sanitized representation:
            Z = X P_perp W_k
    """

    def __init__(self, n_components: int = 4, random_state: int = 42) -> None:
        self.n_components = n_components
        self.random_state = random_state
        self.v_s: Optional[np.ndarray] = None
        self.pca: Optional[PCA] = None

    def fit(self, X: np.ndarray, S: np.ndarray) -> CompressivePrivacySubspaceProjector:
        """Fits sensitive direction and sanitized subspace strictly on training data."""
        # Mean-centered sensitive cross-covariance
        S_centered = S - np.mean(S)
        cross_cov = np.dot(X.T, S_centered)
        norm = np.linalg.norm(cross_cov)
        if norm > 1e-12:
            self.v_s = cross_cov / norm
        else:
            self.v_s = np.zeros(X.shape[1])
            self.v_s[0] = 1.0

        # Project training data onto orthogonal complement of sensitive direction
        # X_sanitized = X - X (v_s v_s^T)
        proj = np.outer(self.v_s, self.v_s)
        X_sanitized = X - np.dot(X, proj)

        # Fit PCA on sanitized representation to extract top k utility dimensions
        self.pca = PCA(n_components=self.n_components, random_state=self.random_state)
        self.pca.fit(X_sanitized)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Projects observations onto the privacy-preserving subspace."""
        if self.v_s is None or self.pca is None:
            raise RuntimeError("Projector must be fitted before transforming data.")
        proj = np.outer(self.v_s, self.v_s)
        X_sanitized = X - np.dot(X, proj)
        return self.pca.transform(X_sanitized)


def run_compressive_privacy_demonstration(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    sensitive_quantile: float = 0.85,
    compression_dimensions: List[int] | None = None,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, any]]:
    """Evaluates utility forecasting vs sensitive attribute leakage across representations.

    Compares:
        1. Full Uncompressed Features (Baseline)
        2. Standard PCA (Variance-maximizing, privacy-oblivious)
        3. Compressive Privacy Subspace Projection (Privacy Funnel formulation)

    Returns:
        Tuple of (comparison metrics DataFrame, summary metadata dict).
    """
    if compression_dimensions is None:
        compression_dimensions = [2, 4, 6]

    # Step 1: Derive sensitive burst threshold strictly on training data
    burst_threshold = float(np.percentile(y_train, sensitive_quantile * 100.0))
    S_train = define_sensitive_attribute(y_train, burst_threshold)
    S_test = define_sensitive_attribute(y_test, burst_threshold)

    logger.info(
        "Compressive Privacy: sensitive burst threshold=%.4f kWh (train prevalence=%.2f%%, test prevalence=%.2f%%)",
        burst_threshold,
        np.mean(S_train) * 100.0,
        np.mean(S_test) * 100.0,
    )

    records: List[Dict[str, any]] = []

    # 1. Full Uncompressed Features Benchmark
    ridge_raw = Ridge(alpha=1.0, random_state=random_state)
    ridge_raw.fit(X_train, y_train)
    preds_raw = ridge_raw.predict(X_test)
    raw_util = calculate_regression_metrics(y_test, preds_raw)

    # Adversary classifier trying to infer sensitive burst S from raw features X
    clf_raw = LogisticRegression(max_iter=1000, random_state=random_state)
    clf_raw.fit(X_train, S_train)
    s_probs_raw = clf_raw.predict_proba(X_test)[:, 1]
    s_preds_raw = clf_raw.predict(X_test)
    auc_raw = float(roc_auc_score(S_test, s_probs_raw))
    acc_raw = float(accuracy_score(S_test, s_preds_raw))

    records.append({
        "representation_method": "Raw Features (Uncompressed)",
        "dimension": X_train.shape[1],
        "utility_rmse": raw_util["rmse"],
        "utility_mae": raw_util["mae"],
        "utility_r2": raw_util["r2"],
        "adversary_sensitive_auc": round(auc_raw, 4),
        "adversary_sensitive_acc": round(acc_raw, 4),
        "privacy_mechanism": "None (Full sensitive exposure)",
        "syllabus_concept": "Uncompressed Data Baseline",
    })

    # 2. Evaluate across compression dimensions
    for k in compression_dimensions:
        # Standard PCA (Oblivious to S)
        pca = PCA(n_components=k, random_state=random_state)
        Z_train_pca = pca.fit_transform(X_train)
        Z_test_pca = pca.transform(X_test)

        ridge_pca = Ridge(alpha=1.0, random_state=random_state)
        ridge_pca.fit(Z_train_pca, y_train)
        preds_pca = ridge_pca.predict(Z_test_pca)
        util_pca = calculate_regression_metrics(y_test, preds_pca)

        clf_pca = LogisticRegression(max_iter=1000, random_state=random_state)
        clf_pca.fit(Z_train_pca, S_train)
        s_probs_pca = clf_pca.predict_proba(Z_test_pca)[:, 1]
        s_preds_pca = clf_pca.predict(Z_test_pca)
        auc_pca = float(roc_auc_score(S_test, s_probs_pca))
        acc_pca = float(accuracy_score(S_test, s_preds_pca))

        records.append({
            "representation_method": f"Standard PCA (k={k})",
            "dimension": k,
            "utility_rmse": util_pca["rmse"],
            "utility_mae": util_pca["mae"],
            "utility_r2": util_pca["r2"],
            "adversary_sensitive_auc": round(auc_pca, 4),
            "adversary_sensitive_acc": round(acc_pca, 4),
            "privacy_mechanism": "Variance Maximization (Oblivious to Sensitive Attribute)",
            "syllabus_concept": "Naive Dimensionality Reduction (Not Compressive Privacy)",
        })

        # Compressive Privacy Subspace Projection (Privacy Funnel)
        cp_proj = CompressivePrivacySubspaceProjector(n_components=k, random_state=random_state)
        cp_proj.fit(X_train, S_train)
        Z_train_cp = cp_proj.transform(X_train)
        Z_test_cp = cp_proj.transform(X_test)

        ridge_cp = Ridge(alpha=1.0, random_state=random_state)
        ridge_cp.fit(Z_train_cp, y_train)
        preds_cp = ridge_cp.predict(Z_test_cp)
        util_cp = calculate_regression_metrics(y_test, preds_cp)

        clf_cp = LogisticRegression(max_iter=1000, random_state=random_state)
        clf_cp.fit(Z_train_cp, S_train)
        s_probs_cp = clf_cp.predict_proba(Z_test_cp)[:, 1]
        s_preds_cp = clf_cp.predict(Z_test_cp)
        auc_cp = float(roc_auc_score(S_test, s_probs_cp))
        acc_cp = float(accuracy_score(S_test, s_preds_cp))

        records.append({
            "representation_method": f"Compressive Privacy Subspace (k={k})",
            "dimension": k,
            "utility_rmse": util_cp["rmse"],
            "utility_mae": util_cp["mae"],
            "utility_r2": util_cp["r2"],
            "adversary_sensitive_auc": round(auc_cp, 4),
            "adversary_sensitive_acc": round(acc_cp, 4),
            "privacy_mechanism": "Privacy Funnel Orthogonal Null-Space Projection (Pi_{S^perp})",
            "syllabus_concept": "Compressive Privacy for Machine Learning (Module 5)",
        })

    metrics_df = pd.DataFrame(records)

    summary = {
        "sensitive_quantile": sensitive_quantile,
        "burst_threshold_kwh": round(burst_threshold, 4),
        "train_prevalence": round(float(np.mean(S_train)), 4),
        "test_prevalence": round(float(np.mean(S_test)), 4),
        "compression_dimensions": compression_dimensions,
    }

    logger.info("Compressive Privacy evaluation completed across %d representation configurations.", len(records))
    return metrics_df, summary
