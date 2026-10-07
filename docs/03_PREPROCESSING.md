# 3. Preprocessing Pipeline

All preprocessing scripts are in `preprocessing/`. They were run locally (VS Code, Python 3.11) because the raw files are several GB.

## 3.1 2023 Q1 pipeline (`preprocessing/2023Q1/`)

| Order | Script | What it does | Output |
|---|---|---|---|
| 1 | `data_analysis.py` | EDA: distributions, delinquency trend, quick feature importance | `figures/eda/*.png`, `data_analysis_summary.txt` |
| 2 | `step2_data_cleaning.py` | Reads raw pipe file in chunks, assigns column names, splits into **static** (one row per loan) and **temporal** (loan × month) tables, converts dates, numeric types | static & temporal CSV/parquet |
| 3 | `step2_temporal_cleaning.py` | Delinquency status → numeric (`XX`/blank → NaN), drops duplicate loan-months, sorts by loan and month | cleaned temporal table |
| 4 | `step2_final_validation.py` | Missing-value report, sanity checks | `results/early_vscode/step2/*_missing_values.csv` |
| 5 | `step3_target_metric.py` | Builds the 6-month-ahead 60+ DPD target | target table |
| 6 | `step3_target_validation.py` | Re-checks the target logic on samples | console report |
| 7 | `step4_prepare_model_data.py`, `step4_split_normalise.py` | Static-only model matrix; stratified split; scaling fitted on train | static model data |
| 8 | `step5_build_sequences.py` | First 12 months of each loan → 12 × 7 sequence (features 0–6) + label | sequence arrays |
| 9 | `step5_split_normalise_sequences.py` | Split + impute + scale sequences (train statistics only) | `.npy` arrays |
| 10 | `stepF0_check_raw_sequences.py`, `stepF1_enhanced_temporal_features.py`, `stepF2_split_normalise_enhanced.py` | Adds the 6 enhanced temporal features (→ 13 per month) for the 12-month data | enhanced arrays |
| 11 | `stepJ0`–`stepJ3` | Checks: available temporal columns, history length per loan, how many loans reach age 18/24, target columns | console reports |
| 12 | `stepJ4_build_18_24_sequences.py` | Builds 18- and 24-month cohorts (observation at loan age 18 / 24) with forward 6-month labels | J18 / J24 raw sequences |
| 13 | `stepJ5_enhance_18_24_sequences.py` | Adds the enhanced temporal features | 13-feature sequences |
| 14 | `stepJ6_split_normalise_18_24.py` | Stratified 70/15/15 split (seed 42), median imputation and z-score scaling **fitted on train only** | `X_sequence_*.npy`, `X_static_*.npy`, `y_*.npy` |
| 15 | `stepJ7_zip_18_24.py` | Zips the J18/J24 folders for upload to Colab / Google Drive | `sequential_enhanced_model_ready_age24.zip` |

## 3.2 2015 Q1 pipeline (`preprocessing/2015Q1/`)

| Order | Script | What it does |
|---|---|---|
| 1 | `step0_split_raw_2015Q1.py` | Streams the raw 2015Q1 file, splits into static and temporal CSVs, **drops all monthly records after February 2020** (COVID cut-off). |
| 2 | `stepL1_build_2015Q1_cohorts.py` | For each history length N ∈ {12, 24, 36, 48}: takes loans observed at age N, builds the N × 13 sequence, the 12 static features, 70 one-hot categorical features and the forward 6-month label; applies exclusions; stratified split; train-only imputation/scaling; verifies labels (0 mismatches); writes `Data/fm2015_age{N}.zip` with `X_seq_*.npy`, `X_static_*.npy`, `X_cat_*.npy`, `y_*.npy`. |

## 3.3 Leakage safeguards

* Features use only months **≤ observation month**; labels use only months **> observation month**.
* Loans already 60+ DPD at the observation month are removed.
* One sample per loan, so the same loan never appears in two splits.
* Imputation medians, scaling means/SDs and one-hot categories are learned on the **training split only**.
* Decision thresholds are chosen on **validation**, then applied once to **test**.
* Balancing (resampling) is applied to the **training split only**; validation and test keep the real class ratio.
* 2015 Q1 data are cut at Feb 2020 so the 6-month label window never includes COVID forbearance months.
