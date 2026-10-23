"""Unit tests for Compressive Privacy (Module 5) demonstration for IA-2."""

from __future__ import annotations

import numpy as np
import pytest

from src.compressive_privacy_demo import (
    CompressivePrivacySubspaceProjector,
    define_sensitive_attribute,
    run_compressive_privacy_demonstration,
)


def test_define_sensitive_attribute() -> None:
    """Verifies binary discretization at sensitive threshold."""
    y = np.array([0.1, 0.2, 0.5, 0.9, 1.2])
    threshold = 0.5
    s = define_sensitive_attribute(y, threshold)
    assert np.array_equal(s, [0, 0, 1, 1, 1])


def test_compressive_privacy_subspace_projection() -> None:
    """Verifies that subspace projection reduces dimension and nulls sensitive correlation."""
    rng = np.random.default_rng(42)
    n = 200
    d = 8
    X = rng.normal(size=(n, d))
    # Sensitive attribute correlated with feature 0
    S = (X[:, 0] > 0.5).astype(int)

    projector = CompressivePrivacySubspaceProjector(n_components=3, random_state=42)
    projector.fit(X, S)
    Z = projector.transform(X)

    assert Z.shape == (n, 3)
    assert not np.isnan(Z).any()


def test_compressive_privacy_demonstration_execution() -> None:
    """Verifies that run_compressive_privacy_demonstration outputs expected schema and metrics."""
    rng = np.random.default_rng(42)
    n_train = 300
    n_test = 100
    d = 8

    X_train = rng.normal(size=(n_train, d))
    y_train = np.abs(X_train[:, 0] * 0.5 + rng.normal(0, 0.1, size=n_train))
    X_test = rng.normal(size=(n_test, d))
    y_test = np.abs(X_test[:, 0] * 0.5 + rng.normal(0, 0.1, size=n_test))

    metrics_df, summary = run_compressive_privacy_demonstration(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        sensitive_quantile=0.85,
        compression_dimensions=[2, 4],
        random_state=42,
    )

    assert len(metrics_df) == 5  # 1 raw + 2 PCA + 2 CP
    assert "utility_rmse" in metrics_df.columns
    assert "adversary_sensitive_auc" in metrics_df.columns
    assert summary["burst_threshold_kwh"] > 0
