"""Unit tests for artifact export and dataset path resolution for IA-2."""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import pytest

from src.config import ProjectPaths
from src.data_loader import resolve_dataset_path
from src.evaluation import export_experiment_artifacts


def test_missing_data_resolution_failure() -> None:
    """Verifies that resolve_dataset_path returns None when given a non-existent path."""
    resolved = resolve_dataset_path("non_existent_directory_or_file_12345.csv")
    assert resolved is None


def test_output_file_creation(tmp_path: Path) -> None:
    """Verifies that export_experiment_artifacts creates all required CSV and JSON output files."""
    paths = ProjectPaths(root_dir=tmp_path)
    data_quality = {"status": "ok", "rows": 100}
    baseline_metrics = [{"model": "Persistence", "rmse": 0.25}]
    dp_results = [{"epsilon": 1.0, "mean_rmse": 0.28, "std_rmse": 0.02, "utility_loss_pct": 12.0}]
    dp_trials = [{"epsilon": 1.0, "trial": 1, "raw_rmse": 0.28}]
    ldp_df = pd.DataFrame([{"epsilon_ldp": 1.0, "mae_proportion_error": 0.01}])
    ldp_trials = pd.DataFrame([{"epsilon_ldp": 1.0, "trial": 1, "tvd": 0.01}])
    agg_df = pd.DataFrame([{"resolution": "30min", "record_count": 100}])
    diff_df = pd.DataFrame([{"timestamp": "2013-01-01 00:00:00", "reconstruction_error": 0.0}])
    cp_df = pd.DataFrame([{"representation_method": "Raw", "utility_rmse": 0.18}])
    run_cfg = {"tested": True}
    env_meta = {"python_version": "3.11"}

    artifacts = export_experiment_artifacts(
        paths=paths,
        data_quality_report=data_quality,
        baseline_metrics=baseline_metrics,
        dp_results=dp_results,
        dp_trials_detail=dp_trials,
        ldp_df=ldp_df,
        ldp_trials_df=ldp_trials,
        aggregation_df=agg_df,
        differencing_attack_df=diff_df,
        compressive_privacy_df=cp_df,
        run_config=run_cfg,
        environment_metadata=env_meta,
    )

    for name, path in artifacts.items():
        assert path.exists(), f"Expected artifact {name} at {path} does not exist!"
        assert path.stat().st_size > 0, f"Artifact {name} at {path} is empty!"
