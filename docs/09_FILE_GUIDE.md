# 9. File Guide

Every code file in the repository and what it does.


## `preprocessing/2023Q1/`

| File | Purpose |
|---|---|
| `data_analysis.py` | Exploratory data analysis of 2023 Q1: distributions (credit score, DTI, LTV, rate, amount, purpose, occupancy, property type), delinquency trend, quick feature importance |
| `step2_data_cleaning.py` | Clean static and temporal tables: types, dates, missing-value report |
| `step2_final_validation.py` | Sanity checks of the cleaned tables |
| `step2_temporal_cleaning.py` | Temporal cleaning: numeric delinquency status, duplicates, sorting |
| `step3_target_metric.py` | Build the 6-month-ahead 60+ DPD target |
| `step3_target_validation.py` | Re-check target logic on sample loans |
| `step4_prepare_model_data.py` | Static model dataset at loan age 12 (for LR / RF baselines) |
| `step4_split_normalise.py` | Stratified split + train-only scaling for static models |
| `step5_build_sequences.py` | Build 12-month × 7-feature sequences + static vectors + labels |
| `step5_split_normalise_sequences.py` | Split, impute and scale sequences (train statistics only) |
| `stepF0_check_raw_sequences.py` | Inspect raw sequence arrays |
| `stepF1_enhanced_temporal_features.py` | Add 6 enhanced temporal features (delinquency dynamics, UPB change) |
| `stepF2_split_normalise_enhanced.py` | Split and scale the enhanced 12-month sequences |
| `stepJ0_check_temporal_columns.py` | List available temporal columns |
| `stepJ1_check_history_length.py` | Distribution of observed history length per loan |
| `stepJ2_check_prediction_cohorts.py` | How many loans can be observed at age 18 / 24 with 6-month follow-up |
| `stepJ3_check_target_columns.py` | Check target-related columns |
| `stepJ4_build_18_24_sequences.py` | Build J18 and J24 cohorts (sequences, static, forward labels, exclusions) |
| `stepJ5_enhance_18_24_sequences.py` | Add enhanced temporal features to J18 / J24 (→ 13 features) |
| `stepJ6_split_normalise_18_24.py` | Stratified 70/15/15 split, train-only median imputation and scaling |
| `stepJ7_zip_18_24.py` | Zip J18 / J24 folders for Colab / Drive |

## `preprocessing/2015Q1/`

| File | Purpose |
|---|---|
| `step0_split_raw_2015Q1.py` | STEP 0 - SPLIT RAW 2015Q1 FILE INTO STATIC + TEMPORAL CSVs |
| `stepL1_build_2015Q1_cohorts.py` | STEP L1 - BUILD 2015Q1 COHORTS: 12, 24, 36, 48 MONTHS |

## `src/early_experiments/`

| File | Purpose |
|---|---|
| `step10_final_results.py` | Phase-1 final results table (hard-coded from test runs) |
| `step10b_final_comparison_graph.py` | Bar chart of Phase-1 model comparison |
| `step10c_reproducibility_graph.py` | Experiment D 3-seed reproducibility plot |
| `step10d_final_summary.py` | Writes Phase-1 text summary |
| `step11_updated_results.py` | Updated Phase-1 table incl. sampler / focal / Exp. C–E |
| `step11b_updated_results_graph.py` | Plot of the updated Phase-1 table |
| `step12_final_results_v2.py` | J18 / J24 held-out and validation results tables |
| `step13_final_comparison_graph_v2.py` | J18 vs J24 comparison chart |
| `step14_key_metrics_graph.py` | J18 / J24 key-metric chart |
| `step15_project_summary_v2.txt` | Text summary of Phase 2 incl. Experiment K |
| `step16_update_experiment_K_results.py` | Adds Experiment K 3-seed results to the tables |
| `step17_experiment_K_graph.py` | Experiment K validation comparison chart |
| `step4_baseline_logistic.py` | Logistic Regression baseline (static features) |
| `step4_random_forest.py` | Random Forest baseline + feature importance |
| `step4_rf_threshold.py` | Threshold search for Random Forest |
| `step4_threshold_optimisation.py` | Threshold search for Logistic Regression |
| `step5_lstm_baseline.py` | LSTM baseline on 12-month sequences (train + validation) |
| `step6_attention_check.py` | Checks attention weights of SCER pre-Mamba on validation |
| `step6_scer_pre_mamba.py` | SCER pre-Mamba model (static conditioning + risk attention + gated fusion) |
| `step7a_lstm_test.py` | LSTM test-set evaluation |
| `step7b_scer_test.py` | SCER pre-Mamba test-set evaluation |
| `step7c_final_comparison.py` | LSTM vs SCER comparison table |
| `step8a_no_static_conditioning.py` | Ablation: remove static conditioning |
| `step8b_no_risk_attention.py` | Ablation: remove risk-event attention |
| `step8c_no_gated_fusion.py` | Ablation: remove adaptive gated fusion |
| `step8d_ablation_comparison.py` | Ablation summary table |
| `step9_simplified_scer.py` | Simplified SCER (lessons from ablation) |

## `src/J24_final/`

| File | Purpose |
|---|---|
| `01_recovery_and_balancing.py` | All-in-one: reload J24 data, define MultiPoolSCERMamba + FocalLoss, run balancing study |
| `02_balancing_experiment.py` | BALANCING EXPERIMENT - EXPERIMENT K (MultiPoolSCERMamba) ON J24 |
| `03_show_balancing_counts.py` | SHOW DATA BEFORE AND AFTER BALANCING (J24) |
| `04_final_comparison.py` | FINAL COMPARISON - J24 HELD-OUT TEST SET |
| `05_improvement_round.py` | IMPROVEMENT ROUND - FAIR TUNING ON J24 |
| `06_tuning_with_drive.py` | ALL-IN-ONE: GOOGLE DRIVE SETUP + J24 RECOVERY + SCER-MAMBA TUNING |
| `07_final_balanced_model.py` | Self-contained: final S3 SCER-Mamba trained on oversampled (1:1) data, 3 seeds, vs focal |
| `08_balance_ratio_test_optional.py` | BALANCING-RATIO TEST FOR THE TUNED SCER-MAMBA (J24) |
| `09_final_figures_and_attention.py` | FINAL STEP - FINAL MODEL CHOICE, TABLE, FIGURES, ATTENTION |

## `src/2015Q1_study/`

| File | Purpose |
|---|---|
| `01_history_length_study_colab.py` | HISTORY-LENGTH STUDY (2015Q1): SCER-MAMBA vs LSTM vs XGBOOST |
| `02_history_length_study_kaggle.py` | HISTORY-LENGTH STUDY (2015Q1) - KAGGLE VERSION |
| `03_significance_test.py` | SIGNIFICANCE TEST: SCER-MAMBA vs LSTM (2015Q1, all lengths) |
| `04_experiment_B_categorical.py` | EXPERIMENT B: + CATEGORICAL FEATURES (2015Q1, 48 months) |
| `05_final_figures_2015Q1.py` | FINAL FIGURES - 2015Q1 STUDY (history length, significance, |
| `06_experiment_C_multitask.py` | EXPERIMENT C: MULTI-TASK LEARNING (2015Q1, 48 months + categorical) |

## Other folders

| Path | Contents |
|---|---|
| `notebooks/RadarSeq_colab.ipynb` | Original Colab notebook (114 cells): SCER-Mamba v2, sampler / focal tests, Experiments C, D, E, K |
| `results/2023Q1/` | Final J24 table, balancing comparison, per-seed focal and oversampled results, label counts |
| `results/tuning/search_results.csv` | S1–S8 architecture search (validation) |
| `results/2015Q1/` | History-length means, significance, Experiment B, Experiment C (partial) |
| `results/early_vscode/` | All CSV / TXT outputs of the local Phase 1–2 scripts |
| `figures/main/` | Final figures 1–11 |
| `figures/eda/` | Exploratory data analysis plots |
| `figures/early/` | Phase 1–2 comparison plots |
| `models/` | Final SCER-Mamba S3 weights, seeds 42 / 123 / 2026 (2023 Q1, J24) |
| `docs/SCER-Mamba_Results_Summary_with_Figures.docx` | 6-page Word summary with all figures |
| `docs/SCER_Mamba_project_summary.xlsx` | Spreadsheet summary of data, methods and results |
