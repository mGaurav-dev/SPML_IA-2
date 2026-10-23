# Comprehensive Audit of IA-1 Codebase, Artifacts, and Methodology

**Project:** Evaluating Privacy–Utility Trade-offs in Machine Learning on Household Energy Consumption Data Using Differential Privacy  
**Assessment Phase:** Transition from IA-1 to IA-2  
**Audit Date:** 2026-10-24  
**Audit Author:** Senior ML & Privacy-Preserving Machine Learning Researcher  
**Source Directory (Read-Only Baseline):** `E:\SPML\IA1`  
**Target Work Directory (Isolated Workspace):** `E:\SPML\IA2`  
**System Environment:** Windows 11 (x64), Python 3.11.4, Diffprivlib 0.6.6, Scikit-Learn 1.5.2, Pandas 2.1.3, NumPy 1.26.4, PyTorch 2.13.0+cpu, Pytest 9.1.1  

---

## 1. Executive Summary and Audit Scope

This audit evaluates the codebase, experimental results, and report generated during the IA-1 phase (`E:\SPML\IA1`) to establish the technical foundation for the IA-2 submission (`E:\SPML\IA2`). 

In strict adherence to the assessment constraints:
1. `E:\SPML\IA1` is treated as strictly **read-only**. A complete cryptographic SHA-256 hash manifest of all 38 files in `E:\SPML\IA1` was computed and saved to [`E:\SPML\IA2\report\ia1_pre_execution_hashes.json`](file:///E:/SPML/IA2/report/ia1_pre_execution_hashes.json) before writing any project assets.
2. All IA-2 implementation, tests, generated datasets, metrics, figures, and technical reports will reside exclusively under `E:\SPML\IA2`.
3. All empirical metrics reported in IA-1 are treated as **historical and unverified** until independently executed and reproduced under `E:\SPML\IA2`.

---

## 2. Inventory of IA-1 Artifacts

The IA-1 directory contains 38 files spanning source code, datasets, unit tests, generated tables, figures, and technical reports:

| Category | Relative Path in IA-1 | File Size (Bytes) | SHA-256 Checksum (Prefix) | Functional Assessment |
| :--- | :--- | :---: | :---: | :--- |
| **Dataset** | `LCL-June2015v2_0.csv` | 50,755,532 | `4997141df1ef...` | 1,000,000 rows, 30 households. Target `MAC000002` verified present. |
| **Data Snapshot** | `household-energy-privacy/data/processed/processed_features_sample.csv` | 82,715 | `43ffd3cba041...` | Snapshot of first 1,000 processed feature rows. |
| **Configuration** | `household-energy-privacy/src/config.py` | 3,801 | `72b83f2a87b5...` | Dataclasses for paths, data, models, DP, LDP, aggregation. |
| **Data Loader** | `household-energy-privacy/src/data_loader.py` | 10,000 | `e5bc4181a4d1...` | Chunked streaming reader, schema alias mapper, synthetic smoke generator. |
| **Preprocessing** | `household-energy-privacy/src/preprocessing.py` | 9,607 | `007995b9c3be...` | Deduplication, cadence check (99.97%), short-gap ffill, lag feature engineering, chronological split. |
| **Baselines** | `household-energy-privacy/src/baseline_model.py` | 5,197 | `565bf83311ef...` | Persistence predictor (lag-1 in kWh) and Scikit-Learn Ridge regression baseline. |
| **DP Model** | `household-energy-privacy/src/dp_model.py` | 7,173 | `da8fe64b1ea2...` | IBM Diffprivlib LinearRegression wrapper with stochastic trial repetition. |
| **LDP Module** | `household-energy-privacy/src/ldp_demo.py` | 7,054 | `9da0e341c662...` | $k$-ary Randomized Response ($k$-RR, $k=3$) with closed-form unbiased estimator. |
| **PPDM Module** | `household-energy-privacy/src/aggregation_demo.py` | 3,193 | `525380cb349a...` | Multi-resolution resampling (30-min, 1-hour, 1-day) and statistical comparison. |
| **Evaluation** | `household-energy-privacy/src/evaluation.py` | 9,105 | `7a88fc235540...` | Matplotlib figure generation (Agg backend) and CSV/JSON artifact export. |
| **Utilities** | `household-energy-privacy/src/utils.py` | 4,195 | `a1584b49852d...` | Random seeding, metrics calculation (RMSE, MAE, R²), logging setup. |
| **CLI Runner** | `household-energy-privacy/main.py` | 12,076 | `3a4e21e101e8...` | CLI argument parser and pipeline coordinator. |
| **Unit Tests** | `household-energy-privacy/tests/test_preprocessing.py` | 3,422 | `666bd8d89af2...` | Tests for schema adaptation, deduplication, imputation, household lag isolation. |
| **Unit Tests** | `household-energy-privacy/tests/test_splits.py` | 2,142 | `1f47ab36fd86...` | Tests for chronological ordering, split ratios, zero test-set leakage in scaling. |
| **Unit Tests** | `household-energy-privacy/tests/test_ldp.py` | 2,536 | `8c5ea0eb3a82...` | Tests for $k$-RR probability axioms, domain bounds, asymptotic unbiasedness. |
| **Unit Tests** | `household-energy-privacy/tests/test_pipeline_and_dp.py` | 3,470 | `887f80e535f3...` | Tests for missing data handling, DP bounds computation, model fitting, artifact creation. |
| **Documentation**| `household-energy-privacy/README.md` | 7,814 | `337eb0239fcd...` | Setup instructions, CLI flags, execution commands. |
| **Requirements** | `household-energy-privacy/requirements.txt` | 250 | `40ca857f5b83...` | Pinned library versions: pandas, numpy, scikit-learn, diffprivlib, matplotlib, pytest. |
| **Historical Report**| `household-energy-privacy/report/IA1_Project_Report.md` | 35,934 | `7f97fe8d1ba3...` | 19-section assessment report for IA-1. |
| **Dataset Notes**| `household-energy-privacy/report/DATASET_NOTES.md` | 4,508 | `e1f1ccec9f06...` | Dataset provenance, schema details, licensing information. |
| **Logs** | `household-energy-privacy/results/logs/run.log` | 5,972 | `2c5c58ea6833...` | Execution trace from IA-1 real-data pipeline run. |
| **Output Figures**| `household-energy-privacy/results/figures/*.png` | ~1.44 MB | Multiple | Prediction comparison, privacy-utility curve, temporal aggregation comparison. |
| **Output Tables** | `household-energy-privacy/results/tables/*.csv/.json`| ~4.2 KB | Multiple | Data quality report, baseline metrics, DP sweep, LDP results, aggregation metrics. |

---

## 3. Detailed Component-by-Component Audit

### 3.1 Data Loading and Schema Normalization (`src/data_loader.py`)
- **Status:** **Implemented & Robust.**
- **Details:** The dynamic alias dictionary (`SCHEMA_ALIASES`) cleanly resolves irregular column names from the London Datastore (`LCLid` $\to$ `household_id`, `DateTime` $\to$ `timestamp`, and `'KWH/hh (per half hour) '` with trailing whitespace $\to$ `energy_kwh`).
- **Memory Safety:** Implements generator-based chunk streaming (`chunk_size=50,000`), ensuring that filtering household `MAC000002` consumes $< 150$ MB RAM even on massive CSV files.
- **Identified Improvement for IA-2:** Default dataset discovery paths must prioritize `E:\SPML\IA2\data\raw\` while maintaining fallback to parent workspaces without writing to them.

### 3.2 Preprocessing and Time-Series Integrity (`src/preprocessing.py`)
- **Status:** **Implemented & Methodologically Sound.**
- **Details:**
  - Cadence check computes interval deltas; verified 99.97% conformity on household `MAC000002`.
  - Deduplication detects and drops duplicate timestamps while preserving the first chronological observation.
  - Missing-data policy uses forward-fill with `limit=2` (maximum 1-hour gap) to avoid hallucinating extended outages, followed by median imputation for any boundary NaNs.
  - Lag construction (`lag_1`, `lag_2`, `lag_48`, `rolling_mean_4`) is strictly past-looking and grouped by `household_id`, preventing cross-boundary leakage.
  - Chronological splitting enforces $\max(\text{train}) < \min(\text{val}) \le \max(\text{val}) < \min(\text{test})$.
  - Normalization via `StandardScaler` is fitted strictly on `X_train`, preventing future test information leakage.
- **Identified Improvement for IA-2:** Provide explicit tabular cadence diagnostic outputs and document the exact missing-data gap distribution.

### 3.3 Baselines (`src/baseline_model.py`)
- **Status:** **Implemented & Valid.**
- **Details:**
  - Persistence baseline correctly uses unscaled `lag_1` features to ensure predictions remain in physical units (kWh/30-min).
  - Ridge regression ($\alpha=1.0$) provides an operational L2-regularized linear baseline.
- **Historical IA-1 Metrics (Unverified until reproduced):**
  - Persistence: RMSE = 0.2029 kWh, MAE = 0.1081 kWh, $R^2 = 0.5116$.
  - Ridge: RMSE = 0.1864 kWh, MAE = 0.1057 kWh, $R^2 = 0.5880$ (+8.15% RMSE improvement over persistence).

### 3.4 Differential Privacy Model (`src/dp_model.py`)
- **Status:** **Implemented, but with severe empirical instability at low $\epsilon$.**
- **Critical Technical Finding on Low-Epsilon Instability:**
  In IA-1, Diffprivlib `LinearRegression` exhibited explosive error at $\epsilon \le 1.0$:
  - $\epsilon = 10.0$: RMSE = 0.1876 kWh ($\Delta\text{RMSE} = +0.63\%$)
  - $\epsilon = 5.0$: RMSE = 0.1887 kWh ($\Delta\text{RMSE} = +1.24\%$)
  - $\epsilon = 2.0$: RMSE = 0.2920 kWh ($\Delta\text{RMSE} = +56.66\%$)
  - $\epsilon = 1.0$: RMSE = 122.57 kWh ($\Delta\text{RMSE} = +65,666\%$)
  - $\epsilon = 0.1$: RMSE = $9.07 \times 10^6$ kWh (Astrophysical scale)
  - $\epsilon = 0.05$: RMSE = $1.06 \times 10^7$ kWh
- **Mathematical Root Cause Analysis:**
  Examination of Diffprivlib's internal source code (`diffprivlib.models.linear_regression._construct_regression_obj`) reveals:
  1. **Budget Splitting Across Monomials:** Diffprivlib implements the Functional Mechanism (Zhang et al., 2012). The total privacy budget $\epsilon$ is divided among all 0th, 1st, and 2nd degree monomials of the quadratic loss:
     $$\epsilon_{\text{local}} = \frac{\epsilon}{1 + n_{\text{targets}} \cdot n_{\text{features}} + \frac{n_{\text{features}}(n_{\text{features}} + 1)}{2}}$$
     With 8 features and 1 target, the divisor is $1 + 8 + 36 = 45$. When $\epsilon = 1.0$, $\epsilon_{\text{local}} \approx 0.0222$. When $\epsilon = 0.05$, $\epsilon_{\text{local}} \approx 0.0011$.
  2. **Indefinite Gram Matrix & Loss of Convexity:**
     Laplace noise is independently added to all symmetric entries of the Gram matrix $A = X^T X + E$. For standard scaled features where $\sum x_i^2 \approx N = 16,865$, the noise magnitude at $\epsilon \le 1.0$ easily exceeds $10^4$. Consequently, $A$ acquires negative eigenvalues.
  3. **Unbounded Below Optimization:**
     Diffprivlib calls `scipy.optimize.minimize(obj, jac=True)` with regularization $\alpha = 0$. When $A$ has negative eigenvalues, the quadratic objective $\omega^T A \omega$ is **unbounded below** ($\lim_{\|\omega\| \to \infty} f(\omega) = -\infty$). BFGS numerical optimization diverges to massive parameter values.
  4. **Physical Plausibility & Post-Processing:**
     Smart meter consumption is non-negative and physically bounded by meter capacity. Predictions were never post-processed or bounded.
- **Action Plan for IA-2:**
  - Preserve Diffprivlib's raw behavior and document the exact functional mechanism failure.
  - Implement physically valid post-processing ($\hat{y} \in [0, y_{\max}]$), which preserves $(\epsilon, 0)$-DP under the DP Post-Processing Theorem.
  - Implement a stable DP estimator (e.g. DP-SGD / DP Ridge with eigenvalue regularization or gradient clipping) to compare against Diffprivlib.

### 3.5 Local Differential Privacy (`src/ldp_demo.py`)
- **Status:** **Implemented & Mathematically Rigorous.**
- **Details:**
  - Discretizes continuous consumption into 3 quantile tertiles (`Low`, `Medium`, `High`) derived strictly from `y_train`.
  - Implements $k$-ary Randomized Response ($k$-RR, $k=3$) with exact transition probabilities $p = \frac{e^\epsilon}{e^\epsilon + k - 1}$ and $q = \frac{1}{e^\epsilon + k - 1}$.
  - Features an unbiased population frequency estimator $\hat{\pi}_i = \frac{\tilde{\pi}_i - q}{p - q}$.
  - Historical IA-1 results: TVD decreased monotonically from $0.06035$ at $\epsilon = 0.5$ to $0.00352$ at $\epsilon = 4.0$.
- **Identified Improvement for IA-2:**
  - Add multi-trial stochastic repetition (mean and standard deviation across repeated random seeds).
  - Implement a downstream classification utility evaluation (e.g. predicting high-demand peak intervals under LDP reports).

### 3.6 Privacy-Preserving Data Mining / Aggregation (`src/aggregation_demo.py`)
- **Status:** **Implemented & Clear.**
- **Details:** Resamples 30-min data to 1-hour and 1-day sums, computing peak-to-average ratios (11.857 $\to$ 9.101 $\to$ 3.254).
- **Core Conceptual Finding:** The IA-1 code and report explicitly noted that heuristic aggregation provides **zero formal differential privacy**, remaining fully susceptible to reconstruction and differencing queries.
- **Identified Improvement for IA-2:** Add concrete quantitative differencing demonstration showing how consecutive overlapping window queries completely reconstruct intermediate half-hourly readings.

### 3.7 Compressive Privacy (Module 5)
- **Status:** **Conceptual Only in IA-1.**
- **Audit Finding:**
  - IA-1 included a rigorous theoretical write-up in `report/IA1_Project_Report.md` framing the Privacy Funnel:
    $$\min_{P_{Z|X}} I(X; S \mid Z) \quad \text{s.t.} \quad I(Z; Y) \ge R$$
  - IA-1 highlighted why naive PCA is not compressive privacy (principal components capture high-energy bursts, which leak sensitive activities).
  - **However, IA-1 contained zero code or empirical metrics for Module 5.**
- **Action Plan for IA-2:**
  - Implement a small, controlled, and technically defensible demonstration of Compressive Privacy in Python.
  - Define a sensitive attribute $S$ (high-consumption appliance burst indicator $S \in \{0, 1\}$).
  - Define a utility forecasting task $Y$ (next-interval consumption).
  - Compare raw features, standard PCA representation (variance-maximizing), and Privacy-Aware representation (orthogonal subspace projection / adversarial feature nulling $\Pi_{S^\perp}$).
  - Measure both utility preservation (forecasting RMSE) and privacy leakage (adversary classification accuracy/AUC of predicting $S$ from $Z$).

---

## 4. Historical IA-1 Results Summary (Reference Baseline)

These values represent the unverified baseline results extracted from IA-1 CSV tables:

### Non-Private Baselines (`results/tables/baseline_metrics.csv`)
| Model | MAE (kWh) | RMSE (kWh) | $R^2$ | Fit Time (s) | Epsilon |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Persistence (Lag-1) | 0.1081 | 0.2029 | 0.5116 | 0.0000 | $\infty$ |
| Ridge Regression | 0.1057 | 0.1864 | 0.5880 | 0.0021 | $\infty$ |

### Central DP Linear Regression (`results/tables/privacy_comparison.csv`)
| Epsilon ($\epsilon$) | Mean MAE (kWh) | Mean RMSE (kWh) | Std RMSE (kWh) | Mean $R^2$ | Utility Loss ($\Delta$RMSE %) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 10.0 | 0.1066 | 0.1876 | 0.0006 | 0.5827 | +0.63% |
| 5.0 | 0.1076 | 0.1887 | 0.0013 | 0.5777 | +1.24% |
| 2.0 | 0.1785 | 0.2920 | 0.1906 | -0.4420 | +56.66% |
| 1.0 | 84.40 | 122.57 | 237.04 | Negative | +65,666% |
| 0.5 | 146.14 | 208.82 | 174.29 | Negative | +111,945% |
| 0.2 | 283.71 | 382.39 | 115.74 | Negative | +205,078% |
| 0.1 | $7.62 \times 10^6$ | $9.07 \times 10^6$ | $1.81 \times 10^7$ | Negative | Extreme |
| 0.05 | $8.87 \times 10^6$ | $1.06 \times 10^7$ | $2.12 \times 10^7$ | Negative | Extreme |

### Local Differential Privacy $k$-RR (`results/tables/ldp_results.csv`)
| $\epsilon_{ldp}$ | $p$ (keep) | $q$ (flip) | Prop MAE | Total Variation Distance (TVD) |
| :---: | :---: | :---: | :---: | :---: |
| 4.0 | 0.9647 | 0.0177 | 0.00235 | 0.00352 |
| 2.0 | 0.7870 | 0.1065 | 0.00945 | 0.01418 |
| 1.0 | 0.5761 | 0.2119 | 0.01875 | 0.02813 |
| 0.5 | 0.4519 | 0.2741 | 0.04023 | 0.06035 |

---

## 5. Summary of Gaps, Enhancements, and Action Items for IA-2

| Module / Topic | IA-1 Status | IA-2 Enhancement Plan | Target Output File(s) |
| :--- | :--- | :--- | :--- |
| **Filesystem Safety** | Unspecified | Strictly enforce read-only boundary on `E:\SPML\IA1`; verify pre/post execution SHA-256 hashes. | `report/ia1_pre_execution_hashes.json`, `report/IA1_AUDIT.md` |
| **Dataset Provenance** | Short summary | Detailed provenance document including SHA-256, access date, exact license, record counts, split sizes. | `report/DATASET_PROVENANCE.md` |
| **Time-Series Integrity** | Well-designed | Add unit tests for cadence conformity, gap distribution checks, and feature-shift verification. | `tests/test_preprocessing.py`, `results/tables/data_quality_report.json` |
| **DP Regression (Mod 2)** | Extreme instability at $\epsilon \le 1.0$ | 1) Document Functional Mechanism noise vs Gram matrix signal; 2) Add post-processing clipping; 3) Implement DP-SGD/regularized DP regressor for stable comparison. | `src/dp_model.py`, `results/tables/privacy_comparison.csv`, `results/figures/privacy_utility_plot.png` |
| **Local DP (Mod 3)** | Single-run demo | Add multi-trial evaluation (mean $\pm$ std across seeds), downstream peak classification utility. | `src/ldp_demo.py`, `results/tables/ldp_results.csv` |
| **PPDM (Mod 4)** | Temporal resampling only | Add empirical differencing query attack demo proving why heuristic aggregation $\neq$ DP. | `src/aggregation_demo.py`, `results/tables/aggregation_differencing_attack.csv` |
| **Compressive Privacy (Mod 5)** | Conceptual only | Controlled demonstration comparing raw features, PCA, and privacy-aware orthogonal projection against sensitive appliance leak. | `src/compressive_privacy_demo.py`, `results/tables/compressive_privacy_metrics.csv` |
| **5-Module Integration** | Qualitative table | Comprehensive 5-module mapping table with exact syllabus terms, evidence files, and limitations. | `report/IA2_Technical_Report.md` Section 8 |
| **Final Technical Report**| 19 sections | Comprehensive 24-section report adhering to Section O of IA-2 brief. | `report/IA2_Technical_Report.md` |
| **Execution Scripts** | Manual commands | Automated PowerShell run and test scripts for repeatable evaluation under IA-2. | `scripts/run_experiments.ps1`, `scripts/run_tests.ps1` |

---
*End of IA-1 Audit Document. All subsequent implementations and outputs are restricted to `E:\SPML\IA2`.*
