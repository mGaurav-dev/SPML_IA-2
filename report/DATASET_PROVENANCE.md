# Dataset Provenance and Data Governance Record

**Project:** Evaluating Privacy–Utility Trade-offs in Machine Learning on Household Energy Consumption Data Using Differential Privacy  
**Assessment:** Security and Privacy for Machine Learning (IA-2)  
**Document Status:** Verified Empirical Record  
**Location in Repository:** `report/DATASET_PROVENANCE.md`  

---

## 1. Source and Publisher Identification

- **Dataset Name:** SmartMeter Energy Consumption Data in London Households
- **Publishing Organization:** Greater London Authority (GLA) / London Datastore
- **Originating Programme:** Low Carbon London (LCL) project led by UK Power Networks
- **Official Portal URL:** [https://data.london.gov.uk/dataset/smartmeter-energy-consumption-data-in-london-households-vqm0d](https://data.london.gov.uk/dataset/smartmeter-energy-consumption-data-in-london-households-vqm0d)
- **Portal Access / Verification Date:** 2026-10-23 (verified during IA-1), audited 2026-10-24 for IA-2.
- **Licence Terms:** Creative Commons Attribution (CC BY 4.0) and UK Open Government Licence (OGL v3) compatible. Public open data.
- **Access Restrictions:** Open access without paywall, registration token, or NDAs.

---

## 2. File-Level Provenance and Integrity

| Parameter | Monolithic Portal Archive | Examined Split File |
| :--- | :--- | :--- |
| **Filename** | `smartmeter-energy-consumption-data-in-london-households.zip` | `LCL-June2015v2_0.csv` |
| **Local File Path** | Not stored locally (avoids ~10 GB download) | `E:\SPML\IA2\data\raw\LCL-June2015v2_0.csv` |
| **Exact File Size** | ~764.54 MB (ZIP) / ~10.2 GB (Uncompressed) | **50,755,532 bytes** (~50.7 MB) |
| **Cryptographic Hash (SHA-256)** | `Unknown` (Portal monolithic zip not downloaded) | `4997141DF1EFB62488305936C3D2AEE395C456F475542BFBDCD5A2145479BD61` |
| **Total Record Count** | ~167,000,000 half-hourly rows | **1,000,000 rows** |
| **Household Coverage** | 5,567 households across Greater London | **30 unique households** (`MAC000002` through `MAC000034`) |
| **Temporal Span** | November 2011 – February 2014 | October 2012 – February 2014 |

---

## 3. Schema and Field-Level Specifications

The raw CSV file contains 4 fields:

```csv
LCLid,stdorToU,DateTime,KWH/hh (per half hour) 
```

| Raw Column Header | Canonical Standard Name | Inferred Dtype | Physical Meaning & Units | Handling & Normalization Policy |
| :--- | :--- | :--- | :--- | :--- |
| `LCLid` | `household_id` | `string` / `category` | Anonymized unique domestic meter ID (e.g., `MAC000002`) | Filtered on target household; stripped of whitespace. |
| `stdorToU` | `tariff_type` | `category` | Tariff regime: `Std` (Standard flat rate) or `ToU` (Dynamic Time-of-Use) | Documented; not used as continuous predictive feature. |
| `DateTime` | `timestamp` | `datetime64[ns]` | Interval start timestamp (Format: `YYYY-MM-DD HH:MM:SS.0000000`) | Parsed via `pd.to_datetime`; checked for cadence conformity. |
| `KWH/hh (per half hour) ` | `energy_kwh` | `float64` | Energy consumed during the 30-minute interval (kWh) | Header has trailing whitespace; missing readings marked `'Null'` coerced to IEEE `NaN`. |

---

## 4. Cohort Selection, Filtering, and Record Flow

For both IA-1 and IA-2, household `MAC000002` serves as the primary target subject, offering a contiguous 505-day load trace:

```
[Raw CSV File: 1,000,000 rows across 30 households]
        │
        ▼ (Stream filter: LCLid == "MAC000002")
[Target Household Raw Records: 24,158 rows]
        │
        ▼ (Timestamp validation & NaT purge: 0 invalid timestamps dropped)
[Valid Timestamp Records: 24,158 rows]
        │
        ▼ (Deduplication on [household_id, timestamp]: 17 duplicate timestamps removed)
[Cleaned Unique Timestamp Records: 24,141 rows]
        │
        ▼ (Cadence conformity: 99.97% 30-min intervals; 1 initial null energy reading ffilled)
[Regular Cadence Series: 24,141 intervals]
        │
        ▼ (Past-looking lag construction: lag_1, lag_2, lag_48, rolling_mean_4; initial 48 rows dropped)
[Usable Feature Matrix: 24,093 intervals]
        │
        ├────────────────────────┬────────────────────────┐
        ▼ (70% Train)            ▼ (15% Validation)       ▼ (15% Chronological Test)
[16,865 intervals]       [3,614 intervals]        [3,614 intervals]
(2012-10-13 to 2013-09-30) (2013-09-30 to 2013-12-14) (2013-12-14 to 2014-02-28)
```

### Numerical Pipeline Reconciliation:
1. **Initial Raw Rows for `MAC000002`:** 24,158
2. **Duplicate Timestamps Removed:** 17 (kept first chronological occurrence)
3. **Invalid Timestamps Dropped:** 0
4. **Missing Energy Values Imputed via Forward-Fill ($\le 2$ intervals):** 1
5. **Residual Missing Values (Median Imputed):** 0
6. **Cleaned Rows Retained:** 24,141
7. **Cadence Conformity Ratio:** 99.97% (48 readings per day)
8. **Rows Dropped for Lag Initialization (first 48 half-hours = 24 hours):** 48
9. **Final Feature Sample Count ($N$):** 24,093
10. **Chronological Split Allocation:**
    - **Train ($N_{\text{train}}$):** 16,865 rows (70.00%)
    - **Validation ($N_{\text{val}}$):** 3,614 rows (15.00%)
    - **Test ($N_{\text{test}}$):** 3,614 rows (15.00%)
    - **Leakage Verification:** $\max(\text{train}) = \text{2013-09-30 09:30:00} < \min(\text{val}) = \text{2013-09-30 10:00:00} \le \max(\text{val}) = \text{2013-12-14 16:30:00} < \min(\text{test}) = \text{2013-12-14 17:00:00}$.

---

## 5. Statistical Profile of Target Household `MAC000002`

| Metric | Raw Training ($y_{\text{train}}$) | Unseen Test ($y_{\text{test}}$) | Full Cleaned Series |
| :--- | :---: | :---: | :---: |
| **Observation Count** | 16,865 | 3,614 | 24,141 |
| **Minimum Consumption** | 0.000 kWh | 0.000 kWh | 0.000 kWh |
| **25th Percentile ($Q_1$)** | 0.093 kWh | 0.106 kWh | 0.096 kWh |
| **Median ($Q_2$)** | 0.158 kWh | 0.158 kWh | 0.158 kWh |
| **75th Percentile ($Q_3$)** | 0.320 kWh | 0.354 kWh | 0.326 kWh |
| **99.5th Percentile (Upper Bound)**| 1.341 kWh | 1.482 kWh | 1.365 kWh |
| **Maximum Consumption** | 2.994 kWh | 2.502 kWh | 2.994 kWh |
| **Mean Consumption ($\mu$)** | 0.2458 kWh | 0.2764 kWh | 0.2525 kWh |
| **Standard Deviation ($\sigma$)** | 0.2413 kWh | 0.2663 kWh | 0.2471 kWh |
| **Peak-to-Average Ratio** | 12.18 | 9.05 | 11.86 |

---

## 6. Privacy Risks, Sensitivity, and Threat Classification

1. **High-Frequency Behavioral Signatures:**
   At 30-minute intervals, individual appliance cycles produce distinct rectangular and triangular pulses (e.g. 3 kW electric kettle, 2 kW oven, 1.5 kW space heater). Non-Intrusive Load Monitoring (NILM) algorithms can disaggregate appliance activations with high precision.
2. **Occupancy and Physical Burglary Risk:**
   Extended low-variance baseline consumption ($< 0.05$ kWh) indicates that residents are asleep or absent from the property (vacation, workday absence), introducing severe physical security liabilities.
3. **Record-Level vs. Household-Level Differential Privacy:**
   - Protecting individual 30-minute intervals bounds the probability ratio of any single appliance burst event.
   - However, household time series exhibit strong autocorrelation. Under the group privacy property, protecting $k = 48$ consecutive intervals (1 calendar day) degrades the privacy budget by $\epsilon_{\text{group}} = k \cdot \epsilon = 48\epsilon$.
   - The boundary of our formal mathematical guarantee is therefore strictly **record-level DP**, not household-level trajectory privacy.

---

## 7. Ethical Governance and Data Use Compliance

1. **Licensing & Copyright:** The data is used in full compliance with the Creative Commons Attribution (CC BY 4.0) licence.
2. **Personal Data & Anonymization:** Meter IDs are pseudonymous identifiers (`MAC000002`). No customer names, street addresses, postal codes, telephone numbers, or financial billing records are present in the dataset.
3. **Redistribution Policy:** Large raw CSV datasets are excluded from Git version control via `.gitignore`. The dataset provenance and acquisition instructions allow any third-party researcher to download the identical dataset directly from the official portal.

---

## 8. Data Placement Instructions for Reproducibility

To reproduce these experiments from scratch:
1. Visit the London Datastore portal: [London SmartMeter Energy Consumption](https://data.london.gov.uk/dataset/smartmeter-energy-consumption-data-in-london-households-vqm0d).
2. Download split archive file `LCL-June2015v2_0.csv`.
3. Verify that the SHA-256 hash matches `4997141DF1EFB62488305936C3D2AEE395C456F475542BFBDCD5A2145479BD61`.
4. Place the file at `E:\SPML\IA2\data\raw\LCL-June2015v2_0.csv`.
5. Run `python main.py` from within `E:\SPML\IA2`.
