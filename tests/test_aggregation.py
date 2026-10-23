"""Unit tests for Privacy-Preserving Data Mining (PPDM) aggregation and differencing attacks for IA-2."""

from __future__ import annotations

import pandas as pd
import pytest

from src.aggregation_demo import compute_temporal_aggregations


def test_compute_temporal_aggregations_shapes() -> None:
    """Verifies that 30-min data resamples cleanly to 1-hour and 1-day temporal series."""
    # 96 half-hours = 2 full days = 48 hours
    timestamps = pd.date_range("2013-01-01 00:00:00", periods=96, freq="30min")
    df = pd.DataFrame({
        "timestamp": timestamps,
        "energy_kwh": [0.5] * 96,
    })

    hh_df, h_df, d_df, metrics_df, diff_df = compute_temporal_aggregations(df)

    assert len(hh_df) == 96
    assert len(h_df) == 48
    assert len(d_df) == 2
    assert len(metrics_df) == 3

    # Total energy must be conserved across resolutions
    total_hh = hh_df["energy_kwh"].sum()
    total_h = h_df["energy_kwh"].sum()
    total_d = d_df["energy_kwh"].sum()
    assert total_hh == pytest.approx(total_h)
    assert total_h == pytest.approx(total_d)


def test_differencing_attack_exact_reconstruction() -> None:
    """Verifies that differencing sliding range queries reconstructs private values with zero error."""
    timestamps = pd.date_range("2013-01-01 00:00:00", periods=150, freq="30min")
    values = [0.1 * (i % 10 + 1) for i in range(150)]
    df = pd.DataFrame({"timestamp": timestamps, "energy_kwh": values})

    _, _, _, _, diff_df = compute_temporal_aggregations(df)

    assert len(diff_df) > 0
    assert (diff_df["reconstruction_error"] < 1e-6).all()
    assert (diff_df["attack_success"]).all()
