"""Unit tests for Differential Privacy model, bounds computation, and post-processing for IA-2."""

from __future__ import annotations

import numpy as np
import pytest

from src.config import DPConfig
from src.dp_model import DifferentiallyPrivateRegressor, compute_training_data_bounds


def test_dp_config_validation() -> None:
    """Verifies default DP configuration attributes and parameters."""
    dp_cfg = DPConfig()
    assert 1.0 in dp_cfg.epsilons
    assert dp_cfg.target_epsilon == 1.0
    assert dp_cfg.n_repeats >= 1
    assert "Single 30-minute" in dp_cfg.protected_unit
    assert "Bounded" in dp_cfg.adjacency_definition


def test_dp_bounds_computation_strictly_train() -> None:
    """Verifies that compute_training_data_bounds returns valid min/max tuples for X and y."""
    X_train = np.array([
        [1.0, 10.0],
        [2.0, 20.0],
        [3.0, 30.0],
        [4.0, 40.0],
    ])
    y_train = np.array([0.5, 1.5, 2.5, 3.5])

    bounds_X, bounds_y = compute_training_data_bounds(X_train, y_train, percentiles=(0.0, 100.0))

    assert len(bounds_X) == 2
    assert np.all(bounds_X[0] <= bounds_X[1])
    assert bounds_y[0] < bounds_y[1]
    assert bounds_y[0] == pytest.approx(0.5)
    assert bounds_y[1] == pytest.approx(3.5)


def test_dp_model_fit_and_predict() -> None:
    """Verifies that DifferentiallyPrivateRegressor fits and predicts valid numerical arrays."""
    X_train = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
    y_train = np.array([1.1, 2.1, 3.1, 4.1, 5.1])
    bounds_X = (np.array([1.0]), np.array([5.0]))
    bounds_y = (1.0, 6.0)

    dp_reg = DifferentiallyPrivateRegressor(epsilon=1.0, bounds_X=bounds_X, bounds_y=bounds_y, random_state=42)
    dp_reg.fit(X_train, y_train)

    X_test = np.array([[2.5], [3.5]])
    preds = dp_reg.predict(X_test)

    assert preds.shape == (2,)
    assert not np.isnan(preds).any()
    assert dp_reg.fit_time >= 0.0
    assert dp_reg.predict_time >= 0.0


def test_dp_post_processing_clipping() -> None:
    """Verifies that predict_clipped enforces compact bounding box [0, y_max]."""
    X_train = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
    y_train = np.array([1.1, 2.1, 3.1, 4.1, 5.1])
    bounds_X = (np.array([1.0]), np.array([5.0]))
    bounds_y = (1.0, 6.0)

    dp_reg = DifferentiallyPrivateRegressor(epsilon=1.0, bounds_X=bounds_X, bounds_y=bounds_y, random_state=42)
    dp_reg.fit(X_train, y_train)

    X_test = np.array([[100.0], [-100.0]])
    clipped_preds = dp_reg.predict_clipped(X_test, y_min=0.0, y_max=3.0)

    assert np.all(clipped_preds >= 0.0)
    assert np.all(clipped_preds <= 3.0)
