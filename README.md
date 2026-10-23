# Evaluating Privacy–Utility Trade-offs in Machine Learning on Household Energy Consumption Data Using Differential Privacy

**Assessment:** Security and Privacy for Machine Learning (IA-2 Implementation and Technical Report)  
**Author:** Individual Course Candidate (Privacy-Preserving Machine Learning Researcher)  
**Target Work Directory:** `E:\SPML\IA2`  
**Reference Baseline:** `E:\SPML\IA1` (strictly read-only)  

---

## 1. Project Purpose and Technical Overview

This project implements, validates, and evaluates privacy-preserving machine learning techniques applied to high-frequency domestic smart meter electricity consumption data.

Covering all five course syllabus modules:
1. **Privacy Considerations in Machine Learning:** Comprehensive threat modelling for smart meter infrastructure, identifying sensitive behavioral side-channels (occupancy flatlines, appliance signatures, daily routines), and analyzing membership/attribute inference attacks.
2. **Differential Privacy for Machine Learning:** Formally grounded $(\epsilon, 0)$-DP linear regression via IBM Diffprivlib across privacy budgets $\epsilon \in [0.05, 10.0]$ under stochastic repetition ($N=5$ trials). Critical empirical investigation of low-$\epsilon$ numerical instability, explaining the Functional Mechanism's loss of Gram matrix positive semi-definiteness, alongside DP post-processing clipping mitigations.
3. **Local Differential Privacy (LDP):** Client-side $k$-ary Randomized Response ($k$-RR, $k=3$) mechanism on energy consumption tertiles with an analytically unbiased aggregate population frequency estimator and downstream peak-demand classification utility evaluation.
4. **Privacy-Preserving Data Mining (PPDM):** Multi-resolution temporal aggregation (30-min native vs 1-hour sum vs 1-day sum) and an empirical demonstration of differencing query reconstruction attacks, proving why heuristic aggregation does not satisfy differential privacy.
5. **Compressive Privacy for Machine Learning:** Controlled demonstration of the Information-Theoretic Privacy Funnel via Orthogonal Subspace Projection ($\Pi_{S^\perp}$), comparing utility forecasting against adversary sensitive appliance detection AUC, demonstrating why standard PCA fails to preserve privacy.

---

## 2. Directory Structure and Workspace Organization

The IA-2 workspace is strictly isolated under `E:\SPML\IA2`:

```
E:\SPML\IA2\
├── data/
│   ├── raw/
│   │   └── LCL-June2015v2_0.csv         <- London Smart Meter subset (50.7 MB, 1,000,000 rows)
│   └── processed/
│       └── processed_features_sample.csv <- Snapshot of first 1,000 processed feature intervals
├── report/
│   ├── IA1_AUDIT.md                     <- Comprehensive audit of IA-1 codebase and baseline results
│   ├── DATASET_PROVENANCE.md            <- Verified provenance, licensing, and schema documentation
│   ├── IA2_Technical_Report.md          <- Final 24-section comprehensive assessment technical report
│   ├── ia1_pre_execution_hashes.json    <- SHA-256 pre-execution hash manifest of all IA-1 files
│   └── ia1_post_execution_hashes.json   <- SHA-256 post-execution hash manifest verifying IA-1 untouched
├── results/
│   ├── figures/
│   │   ├── prediction_comparison.png    <- Ground truth vs baselines vs DP models (3-day window)
│   │   ├── privacy_utility_plot.png     <- Epsilon vs RMSE trade-off curve (Raw vs Clipped DP)
│   │   ├── aggregation_comparison.png   <- Multi-resolution temporal profile comparison
│   │   └── compressive_privacy_tradeoff.png <- Utility R² vs sensitive burst detection AUC
│   ├── tables/
│   │   ├── data_quality_report.json     <- Deduplication, missing value imputation, cadence report
│   │   ├── baseline_metrics.csv         <- Persistence and Ridge regression metrics
│   │   ├── privacy_comparison.csv       <- DP regression summary across epsilon budgets
│   │   ├── privacy_trials_detail.csv    <- Every stochastic trial recorded across all epsilons
│   │   ├── ldp_results.csv              <- LDP summary: TVD, proportion MAE, downstream accuracy/F1
│   │   ├── ldp_trials_detail.csv        <- Per-trial LDP stochastic evaluations
│   │   ├── aggregation_comparison.csv   <- Descriptive stats and peak-to-average ratios
│   │   ├── aggregation_differencing_attack.csv <- Empirical differencing reconstruction proof
│   │   ├── compressive_privacy_metrics.csv <- Module 5 representation comparison metrics
│   │   ├── run_config.json              <- Complete runtime configuration parameters
│   │   └── environment_metadata.json    <- System platform, Python, and pinned library metadata
│   └── logs/
│       └── run.log                      <- Full timestamped pipeline execution log
├── scripts/
│   ├── run_experiments.ps1              <- Automated PowerShell pipeline runner
│   └── run_tests.ps1                    <- Automated Pytest test suite runner
├── src/
│   ├── __init__.py                      <- Package marker
│   ├── config.py                        <- Dataclass configurations and paths
│   ├── data_loader.py                   <- Chunked loading and schema adapter
│   ├── preprocessing.py                 <- Time-series validation, lags, chronological split, scaling
│   ├── baseline_model.py                <- Persistence (unscaled lag-1) and Ridge baselines
│   ├── dp_model.py                      <- IBM Diffprivlib DP regression & clipping wrapper
│   ├── ldp_demo.py                      <- k-RR LDP mechanism & unbiased frequency estimator
│   ├── aggregation_demo.py              <- PPDM aggregation and differencing attack demo
│   ├── compressive_privacy_demo.py      <- Privacy Funnel orthogonal projection & adversary AUC
│   ├── evaluation.py                    <- Matplotlib figures (Agg backend) & artifact exporter
│   └── utils.py                         <- Seeding, metrics, and JSON utilities
├── tests/
│   ├── test_preprocessing.py            <- Schema, deduplication, imputation, household boundaries
│   ├── test_splits.py                   <- Chronological order, zero temporal lookahead leakage
│   ├── test_dp.py                       <- DP bounds, model execution, post-processing clipping
│   ├── test_ldp.py                      <- k-RR probability axioms, domain bounds, unbiasedness
│   ├── test_aggregation.py              <- Resampling conservation, differencing reconstruction
│   ├── test_compressive_privacy.py      <- Sensitive thresholding, subspace projection, adversary AUC
│   └── test_artifacts.py                <- Artifact creation and path resolution tests
├── main.py                              <- Unified CLI entry point
├── requirements.txt                     <- Pinned library dependencies
├── .gitignore                           <- Source control exclusions
└── README.md                            <- Setup and user manual (this file)
```

---

## 3. Prerequisites and Environment Setup

- **Operating System:** Windows 10/11 (x64)
- **Python:** Python 3.11 (tested on Python 3.11.4)
- **Editor:** Visual Studio Code (VS Code) with Python extension

### Windows PowerShell Setup

Open PowerShell in `E:\SPML\IA2`:

```powershell
# Set execution policy for current process if required
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

# Install/verify dependencies
pip install -r requirements.txt
```

---

## 4. Execution Instructions

### A. Run Complete Real-Data Experiment Pipeline
Executes the full pipeline on household `MAC000002` across all 5 syllabus modules:

```powershell
python main.py
```

Or using the automated PowerShell script:
```powershell
.\scripts\run_experiments.ps1
```

### B. Command-Line Arguments

| Flag | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--data-path` | string | Auto-detect | Explicit path to CSV file |
| `--household-id` | string | `MAC000002` | Target household identifier |
| `--subset-size` | int | `25000` | Maximum rows to load |
| `--smoke-test` | flag | `False` | Run synthetic smoke test (independent of raw data) |
| `--skip-plots` | flag | `False` | Skip rendering PNG figures |
| `--epsilons` | float list | `0.05 0.1 0.2 ... 10.0` | Epsilon budgets for central DP |
| `--n-repeats` | int | `5` | Stochastic trials per epsilon |

**Examples:**
```powershell
# Run with custom epsilon values and 10 trials
python main.py --epsilons 0.5 1.0 2.0 5.0 --n-repeats 10

# Run in quick synthetic smoke-test mode
python main.py --smoke-test
```

---

## 5. Running the Unit Test Suite

All 24 automated unit tests are executed using `pytest`:

```powershell
python -m pytest tests/ -v
```

Or via the PowerShell runner:
```powershell
.\scripts\run_tests.ps1
```

---

## 6. Key Scientific Findings and Bounds

1. **Baseline Accuracy:**
   - Operational Persistence Baseline: RMSE = 0.2029 kWh, $R^2 = 0.5116$.
   - Non-Private Ridge Regression ($\alpha=1.0$): RMSE = 0.1864 kWh, $R^2 = 0.5880$ (+8.15% utility improvement over persistence).
2. **Central Differential Privacy (Diffprivlib Functional Mechanism):**
   - High Epsilon ($\epsilon = 10.0$): Raw RMSE = $0.1876 \pm 0.0006$ kWh ($\Delta\text{RMSE} = +0.64\%$), tight utility preservation.
   - Moderate Epsilon ($\epsilon = 2.0$): Raw RMSE = $0.2920 \pm 0.1906$ kWh ($\Delta\text{RMSE} = +56.66\%$).
   - Low Epsilon ($\epsilon \le 1.0$): Severe numerical instability due to Gram matrix noise exceeding sample size, inducing negative eigenvalues. BFGS diverges on the unbounded-below quadratic objective.
   - **Post-Processing Mitigation:** Applying physical domain clipping $\hat{y}_{\text{clipped}} \in [0, y_{\max}]$ strictly preserves $(\epsilon, 0)$-DP and caps RMSE to $< 0.94$ kWh even at $\epsilon = 0.05$.
3. **Local Differential Privacy ($k$-RR, $k=3$):**
   - Total Variation Distance (TVD) decreases monotonically from $0.03401$ ($\epsilon = 0.5$) to $0.00332$ ($\epsilon = 4.0$).
   - Downstream peak classification accuracy recovers from 59.46% ($\epsilon = 0.5$) to 97.33% ($\epsilon = 4.0$).
4. **PPDM Aggregation and Differencing Attack:**
   - Aggregation reduces peak-to-average ratio (11.857 $\to$ 9.101 $\to$ 3.254), smoothing appliance spikes.
   - However, differencing sliding window queries reconstructs private half-hourly readings with exact zero error ($0.00000000$ kWh error), proving that deterministic aggregation is **not** a differential privacy mechanism.
5. **Compressive Privacy vs Naive PCA:**
   - Raw features leak sensitive appliance bursts with adversary detection AUC = 0.9016.
   - Standard PCA preserves high sensitive leakage (AUC = 0.8956 to 0.9015) because principal components capture maximum variance, which aligns directly with peak appliance activations.
   - Compressive Privacy Subspace Projection ($\Pi_{S^\perp}$) suppresses adversary sensitive AUC to 0.4411 – 0.5174 (equivalent to random guessing), formally demonstrating the Privacy Funnel principle.

---

## 7. Strict Filesystem Safety and Isolation Verification

As required by assessment safety rules:
- `E:\SPML\IA1` was maintained strictly read-only throughout all audits, implementations, and evaluations.
- Pre-execution SHA-256 hashes of all 38 files in `E:\SPML\IA1` were saved in `report/ia1_pre_execution_hashes.json`.
- Post-execution SHA-256 verification confirmed that zero bytes, files, or timestamps in `E:\SPML\IA1` were altered.
- All generated files, caches, figures, tables, and reports are strictly contained within `E:\SPML\IA2`.
