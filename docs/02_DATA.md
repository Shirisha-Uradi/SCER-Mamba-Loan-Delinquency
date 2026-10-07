# 2. Data

## 2.1 Source

**Fannie Mae Single-Family Loan Performance Data** – a public dataset of fixed-rate, fully amortising 30-year mortgages acquired by Fannie Mae, with monthly performance records.

* Download: <https://capitalmarkets.fanniemae.com/credit-risk-transfer/single-family-credit-risk-transfer/fannie-mae-single-family-loan-performance-data> (free registration).
* Format: one pipe-delimited (`|`) file per acquisition quarter, **no header**, 108 columns; dates as `MMYYYY`. Each row = one loan in one month (static fields repeated on every row).
* **The data is not stored in this repository.** Place the raw files under `Data/` locally (see `.gitignore`).

## 2.2 Quarters used

| Quarter | Role | Notes |
|---|---|---|
| **2023 Q1** | Main development set | 210,810 unique loans. Performance observed up to the latest release. Used for all J18/J24 experiments. |
| **2015 Q1** | History-length study | Long histories available (up to 48+ months). **Records after February 2020 are dropped** so COVID-19 forbearance does not distort labels. |

## 2.3 Descriptive statistics (2023 Q1)

From `results/early_vscode/data_analysis_summary.txt` (script `preprocessing/2023Q1/data_analysis.py`):

| Statistic | Value |
|---|---|
| Unique loans | 210,810 |
| Static fields | 32 |
| Median / mean credit score | 763 / 754.9 |
| Median / mean DTI | 39% / 37.5% |
| Median / mean LTV | 80% / 75.3% |

EDA plots are in `figures/eda/` (credit score, DTI, LTV, interest rate, loan amount, loan purpose, occupancy, property type, delinquency status, 60+ delinquency trend, risky vs non-risky, feature importance).

## 2.4 Features

### Static features (12, one vector per loan)

| # | Feature | Meaning |
|---|---|---|
| 1 | borrower_credit_score_at_origination | FICO of main borrower |
| 2 | co_borrower_credit_score_at_origination | FICO of co-borrower (NaN if none) |
| 3 | dti | Debt-to-income ratio (%) |
| 4 | original_ltv | Loan-to-value at origination (%) |
| 5 | original_cltv | Combined LTV (%) |
| 6 | original_upb | Original unpaid principal balance |
| 7 | original_interest_rate | Note rate (%) |
| 8 | number_of_borrowers | 1, 2, … |
| 9 | mortgage_insurance_percentage | MI coverage (%) |
| 10 | has_co_borrower | 1 if number_of_borrowers > 1 |
| 11 | co_borrower_score_missing | Missing-value indicator |
| 12 | mortgage_insurance_missing | Missing-value indicator |

### Categorical features (Experiment B only; 70 one-hot columns)

`loan_purpose`, `occupancy`, `property_type`, `channel`, `first_time_home_buyer`, `property_state` (missing → "Missing" category).

### Temporal features (13 per month)

| Col | Feature | Meaning |
|---|---|---|
| 0 | current_interest_rate | Rate in that month |
| 1 | current_actual_upb | Outstanding balance |
| 2 | delinquency_status_numeric | Months past due (0, 1, 2, …) |
| 3 | remaining_months_to_maturity | Months left |
| 4 | is_30plus_delinquent | 1 if ≥ 1 month late |
| 5 | is_60plus_delinquent | 1 if ≥ 2 months late |
| 6 | is_90plus_delinquent | 1 if ≥ 3 months late |
| 7 | delinquency_change | Status this month − last month |
| 8 | delinquency_worsening | 1 if status increased |
| 9 | cumulative_max_delinquency | Worst status so far |
| 10 | consecutive_30plus_months | Current run of late months |
| 11 | recent_3m_30plus_count | Late months in last 3 |
| 12 | monthly_upb_change | Balance change vs last month |

Columns 4–6 (delinquency flags) are fed directly into the risk-event attention module.
Columns 7–12 are the "enhanced" features added in step F1/J5; they are computed only from the current and past months (no look-ahead).

## 2.5 Target

```
y = 1  if the loan reaches delinquency_status_numeric ≥ 2 (60+ days past due)
       in ANY of the 6 months after the observation month
y = 0  otherwise
```

Exclusions:
* loans already 60+ days late at the observation month (we want early warning, not detection of existing default);
* loans without 6 full months of follow-up (label unknown – includes loans that prepaid);
* each loan contributes **one** sample (no overlapping windows of the same loan across splits).

## 2.6 Cohort sizes

### 2023 Q1

| Cohort | History | Train (risky / total) | Val (risky / total) | Test (risky / total) |
|---|---|---|---|---|
| J24 | 24 months | 1,198 / 114,296 | 257 / 24,492 | 257 / 24,492 |

Risky rate ≈ **1.05%** in every split (stratified 70 / 15 / 15 split, seed 42).

### 2015 Q1 (pre-COVID)

| History | Loans | Risky | Risky % |
|---|---|---|---|
| 12 months | 365,200 | 1,041 | 0.29% |
| 24 months | 313,494 | 1,232 | 0.39% |
| 36 months | 276,990 | 1,130 | 0.41% |
| 48 months | 249,424 | 1,172 | 0.47% |

Labels were independently re-computed and checked against the cohort files: **0 mismatches**.
