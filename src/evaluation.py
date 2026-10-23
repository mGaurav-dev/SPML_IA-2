"""Evaluation routines, figure generation, and structured artifact exporter for IA-2."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import ProjectPaths
from .utils import ensure_directory, save_json

logger = logging.getLogger("household_energy_dp")


def plot_prediction_comparison(
    y_test: np.ndarray,
    baseline_preds: Dict[str, np.ndarray],
    dp_preds: Dict[float, np.ndarray],
    output_path: Path,
    sample_window: int = 144,  # Exactly 3 days (144 half-hours)
) -> Path:
    """Plots actual versus predicted half-hourly electricity consumption over a test window."""
    ensure_directory(output_path.parent)
    window = min(sample_window, len(y_test))
    x_indices = np.arange(window)

    fig, ax = plt.subplots(figsize=(12, 5), dpi=300)

    # Ground truth
    ax.plot(x_indices, y_test[:window], label="Ground Truth (Actual)", color="black", linewidth=1.6, alpha=0.95)

    # Baselines
    if "Persistence" in baseline_preds:
        ax.plot(
            x_indices,
            baseline_preds["Persistence"][:window],
            label="Persistence (Lag-1)",
            color="#2ca02c",
            linestyle="--",
            alpha=0.75,
        )

    if "Ridge" in baseline_preds:
        ax.plot(
            x_indices,
            baseline_preds["Ridge"][:window],
            label="Ridge Baseline (Non-Private)",
            color="#1f77b4",
            linestyle="-.",
            alpha=0.85,
        )

    plotted_epsilons = [e for e in [5.0, 2.0, 1.0] if e in dp_preds]
    colors = {5.0: "#9467bd", 2.0: "#d62728", 1.0: "#e377c2"}

    y_max_plot = float(np.percentile(y_test[:window], 99.5)) * 1.5

    for eps in plotted_epsilons:
        preds_window = np.clip(dp_preds[eps][:window], 0.0, y_max_plot)
        ax.plot(
            x_indices,
            preds_window,
            label=f"DP Linear Regression (ε = {eps})",
            color=colors.get(eps, "#d62728"),
            linewidth=1.3,
            alpha=0.85,
        )

    ax.set_ylim(-0.05, y_max_plot)
    ax.set_title("Short-Term Household Electricity Forecasting: Actual vs Baselines vs DP Predictions", fontsize=12, fontweight="bold")
    ax.set_xlabel("Half-Hourly Time Intervals (3-day evaluation window)", fontsize=10)
    ax.set_ylabel("Electricity Consumption (kWh / 30 min)", fontsize=10)
    ax.legend(loc="upper right", frameon=True, fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    logger.info("Saved prediction comparison plot to: %s", output_path)
    return output_path


def plot_privacy_utility_tradeoff(
    dp_results: List[Dict[str, any]],
    base_ridge_rmse: float,
    output_path: Path,
) -> Path:
    """Plots empirical privacy-utility trade-off curves (Raw vs Clipped RMSE vs Epsilon) with error bars."""
    ensure_directory(output_path.parent)

    epsilons = [r["epsilon"] for r in dp_results]
    mean_raw_rmses = [r["mean_rmse"] for r in dp_results]
    std_raw_rmses = [r["std_rmse"] for r in dp_results]
    mean_clip_rmses = [r.get("mean_clipped_rmse", r["mean_rmse"]) for r in dp_results]
    std_clip_rmses = [r.get("std_clipped_rmse", r["std_rmse"]) for r in dp_results]

    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)

    # Raw DP error curve
    ax.errorbar(
        epsilons,
        mean_raw_rmses,
        yerr=std_raw_rmses,
        fmt="-o",
        color="#d62728",
        ecolor="#ff9896",
        elinewidth=2,
        capsize=4,
        label="Diffprivlib Raw Output (Mean ± Std)",
        linewidth=1.8,
    )

    # Post-processed clipped DP curve
    ax.errorbar(
        epsilons,
        mean_clip_rmses,
        yerr=std_clip_rmses,
        fmt="--s",
        color="#2ca02c",
        ecolor="#98df8a",
        elinewidth=1.8,
        capsize=4,
        label="DP Post-Processed [0, y_max] (Mean ± Std)",
        linewidth=1.8,
    )

    # Non-private reference line
    ax.axhline(
        base_ridge_rmse,
        color="#1f77b4",
        linestyle=":",
        linewidth=2.0,
        label=f"Non-Private Ridge Baseline (RMSE = {base_ridge_rmse:.4f})",
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Differential Privacy: Empirical Privacy–Utility Trade-off Curve", fontsize=12, fontweight="bold")
    ax.set_xlabel("Privacy Budget Epsilon (ε) [log scale]", fontsize=10)
    ax.set_ylabel("Root Mean Squared Error (RMSE in kWh) [log scale]", fontsize=10)
    ax.legend(loc="upper right", frameon=True, fontsize=9)
    ax.grid(True, which="both", linestyle=":", alpha=0.6)

    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    logger.info("Saved privacy-utility trade-off plot to: %s", output_path)
    return output_path


def plot_aggregation_comparison(
    half_hourly_df: pd.DataFrame,
    hourly_df: pd.DataFrame,
    daily_df: pd.DataFrame,
    output_path: Path,
    days_to_plot: int = 14,
) -> Path:
    """Visualizes multi-resolution temporal profiles to illustrate data minimization and smoothing."""
    ensure_directory(output_path.parent)

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), dpi=300, sharex=False)

    hh_slice = half_hourly_df.iloc[: days_to_plot * 48]
    h_slice = hourly_df.iloc[: days_to_plot * 24]
    d_slice = daily_df.iloc[:days_to_plot]

    # Subplot 1: 30-min native
    axes[0].plot(hh_slice["timestamp"], hh_slice["energy_kwh"], color="#1f77b4", linewidth=1.0)
    axes[0].set_title(f"Native 30-Minute Resolution (Full Temporal Detail, {len(hh_slice)} intervals)", fontsize=10, fontweight="bold")
    axes[0].set_ylabel("kWh / 30min", fontsize=9)
    axes[0].grid(True, linestyle=":", alpha=0.6)

    # Subplot 2: Hourly sum
    axes[1].plot(h_slice["timestamp"], h_slice["energy_kwh"], color="#ff7f0e", linewidth=1.2)
    axes[1].set_title(f"Hourly Aggregated Sum (Smoothed Micro-Events, {len(h_slice)} hours)", fontsize=10, fontweight="bold")
    axes[1].set_ylabel("kWh / hour", fontsize=9)
    axes[1].grid(True, linestyle=":", alpha=0.6)

    # Subplot 3: Daily sum
    axes[2].bar(d_slice["timestamp"], d_slice["energy_kwh"], color="#2ca02c", width=0.8, alpha=0.8)
    axes[2].set_title(f"Daily Aggregated Sum (Macroscopic Daily Totals, {len(d_slice)} days)", fontsize=10, fontweight="bold")
    axes[2].set_ylabel("kWh / day", fontsize=9)
    axes[2].grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    logger.info("Saved aggregation comparison plot to: %s", output_path)
    return output_path


def plot_compressive_privacy_tradeoff(
    compressive_df: pd.DataFrame,
    output_path: Path,
) -> Path:
    """Plots Utility (R2) vs Privacy Leakage (Adversary Sensitive Detection AUC) for Module 5."""
    ensure_directory(output_path.parent)

    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)

    pca_mask = compressive_df["representation_method"].str.contains("Standard PCA")
    cp_mask = compressive_df["representation_method"].str.contains("Compressive Privacy")
    raw_mask = compressive_df["representation_method"].str.contains("Raw")

    df_pca = compressive_df[pca_mask]
    df_cp = compressive_df[cp_mask]
    df_raw = compressive_df[raw_mask]

    # Raw point
    if not df_raw.empty:
        ax.scatter(
            df_raw["utility_r2"],
            df_raw["adversary_sensitive_auc"],
            color="black",
            s=120,
            marker="X",
            zorder=5,
            label="Raw Features (Full Leakage)",
        )

    # Standard PCA curve
    ax.plot(
        df_pca["utility_r2"],
        df_pca["adversary_sensitive_auc"],
        "-o",
        color="#d62728",
        linewidth=2,
        markersize=8,
        label="Standard PCA (Variance-Maximizing, High Leakage)",
    )
    for _, row in df_pca.iterrows():
        ax.annotate(f"k={row['dimension']}", (row["utility_r2"], row["adversary_sensitive_auc"]),
                    textcoords="offset points", xytext=(-5, 8), fontsize=8)

    # Compressive Privacy curve
    ax.plot(
        df_cp["utility_r2"],
        df_cp["adversary_sensitive_auc"],
        "-s",
        color="#1f77b4",
        linewidth=2,
        markersize=8,
        label="Compressive Privacy Subspace (Privacy Funnel)",
    )
    for _, row in df_cp.iterrows():
        ax.annotate(f"k={row['dimension']}", (row["utility_r2"], row["adversary_sensitive_auc"]),
                    textcoords="offset points", xytext=(-5, -12), fontsize=8)

    # Reference line for random guessing adversary
    ax.axhline(0.5, color="gray", linestyle=":", label="Random Guessing Adversary (AUC = 0.50)")

    ax.set_title("Module 5: Compressive Privacy Trade-off (Utility R² vs Sensitive Burst Leakage AUC)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Downstream Forecast Utility (R² Score) [Higher is Better]", fontsize=10)
    ax.set_ylabel("Adversary Sensitive Detection AUC [Lower is Better Privacy]", fontsize=10)
    ax.legend(loc="upper left", frameon=True, fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    logger.info("Saved compressive privacy trade-off plot to: %s", output_path)
    return output_path


def export_experiment_artifacts(
    paths: ProjectPaths,
    data_quality_report: Dict[str, any],
    baseline_metrics: List[Dict[str, any]],
    dp_results: List[Dict[str, any]],
    dp_trials_detail: List[Dict[str, any]],
    ldp_df: pd.DataFrame,
    ldp_trials_df: pd.DataFrame,
    aggregation_df: pd.DataFrame,
    differencing_attack_df: pd.DataFrame,
    compressive_privacy_df: pd.DataFrame,
    run_config: Dict[str, any],
    environment_metadata: Dict[str, any],
) -> Dict[str, Path]:
    """Persists all experiment results into standardized CSV and JSON artifact files in results/tables/."""
    tables_dir = paths.results_tables
    ensure_directory(tables_dir)

    artifacts: Dict[str, Path] = {}

    # 1. Data Quality Report
    dq_path = tables_dir / "data_quality_report.json"
    save_json(data_quality_report, dq_path)
    artifacts["data_quality_report"] = dq_path

    # 2. Baseline Metrics
    base_path = tables_dir / "baseline_metrics.csv"
    pd.DataFrame(baseline_metrics).to_csv(base_path, index=False)
    artifacts["baseline_metrics"] = base_path

    # 3. DP Privacy Summary
    dp_path = tables_dir / "privacy_comparison.csv"
    pd.DataFrame(dp_results).to_csv(dp_path, index=False)
    artifacts["privacy_comparison"] = dp_path

    # 4. DP Per-Trial Records
    dp_trials_path = tables_dir / "privacy_trials_detail.csv"
    pd.DataFrame(dp_trials_detail).to_csv(dp_trials_path, index=False)
    artifacts["privacy_trials_detail"] = dp_trials_path

    # 5. LDP Results Summary
    ldp_path = tables_dir / "ldp_results.csv"
    ldp_df.to_csv(ldp_path, index=False)
    artifacts["ldp_results"] = ldp_path

    # 6. LDP Per-Trial Records
    ldp_trials_path = tables_dir / "ldp_trials_detail.csv"
    ldp_trials_df.to_csv(ldp_trials_path, index=False)
    artifacts["ldp_trials_detail"] = ldp_trials_path

    # 7. Aggregation Summary
    agg_path = tables_dir / "aggregation_comparison.csv"
    aggregation_df.to_csv(agg_path, index=False)
    artifacts["aggregation_comparison"] = agg_path

    # 8. Aggregation Differencing Attack Demonstration
    diff_path = tables_dir / "aggregation_differencing_attack.csv"
    differencing_attack_df.to_csv(diff_path, index=False)
    artifacts["aggregation_differencing_attack"] = diff_path

    # 9. Compressive Privacy Metrics
    cp_path = tables_dir / "compressive_privacy_metrics.csv"
    compressive_privacy_df.to_csv(cp_path, index=False)
    artifacts["compressive_privacy_metrics"] = cp_path

    # 10. Run Configuration
    cfg_path = tables_dir / "run_config.json"
    save_json(run_config, cfg_path)
    artifacts["run_config"] = cfg_path

    # 11. Environment Metadata
    env_path = tables_dir / "environment_metadata.json"
    save_json(environment_metadata, env_path)
    artifacts["environment_metadata"] = env_path

    logger.info("Successfully exported %d structured artifacts to %s", len(artifacts), tables_dir)
    return artifacts
