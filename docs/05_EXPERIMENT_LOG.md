# 5. Experiment Log

Every experiment in the order it was run, with the script, setting and outcome. Numbers are **test-set** results unless marked "val".

---

## Phase 1 – Local pipeline on 2023 Q1, 12-month history (VS Code)

Test set for this phase: 228 risky / 27,029 not risky.

| # | Experiment | Script | PR-AUC | F1 | Outcome |
|---|---|---|---|---|---|
| 1.1 | Logistic Regression (static, age 12) | `src/early_experiments/step4_baseline_logistic.py`, `step4_threshold_optimisation.py` | – | – | First baseline; threshold tables in `results/early_vscode/models/` |
| 1.2 | Random Forest (static) | `step4_random_forest.py`, `step4_rf_threshold.py` | – | – | Feature importance: credit score, DTI, LTV, rate dominate |
| 1.3 | LSTM, 12 × 7 sequence | `step5_lstm_baseline.py`, `step7a_lstm_test.py` | **0.189** | 0.340 | Strong sequence baseline |
| 1.4 | SCER pre-Mamba (GRU-style encoder + gates + attention) | `step6_scer_pre_mamba.py`, `step7b_scer_test.py` | 0.177 | 0.290 | Below LSTM |
| 1.5 | SCER-Mamba v2 | (Colab, `notebooks/RadarSeq_colab.ipynb`) | 0.169 | 0.297 | Below LSTM |
| 1.6 | Ablation: no static conditioning | `step8a_no_static_conditioning.py` | 0.181 | 0.320 | Gate not helpful → removed later |
| 1.7 | Ablation: no risk-event attention | `step8b_no_risk_attention.py` | 0.174 | 0.290 | Attention helps → kept |
| 1.8 | Ablation: no adaptive gated fusion | `step8c_no_gated_fusion.py` | 0.181 | 0.323 | Gate not helpful → removed later |
| 1.9 | Simplified SCER | `step9_simplified_scer.py` | – | – | Led to Experiment D |
| 1.10 | Sampler + focal / sampler only / focal only | notebook | 0.165 / 0.160 / **0.179** | 0.312 / 0.281 / 0.321 | Focal loss alone best |
| 1.11 | Exp. C (no gate), D (no static cond.), E (no cond. + no gate) | notebook | 0.173 / 0.182 / 0.178 | 0.324 / 0.326 / 0.311 | D best |
| 1.12 | Exp. D, 3 seeds | `step10*`, `step11*` | 0.182 ± 0.018 | 0.310 ± 0.006 | Reproducible, still < LSTM |

**Lesson:** with only 12 months of history and 7 features, there is little sequential signal. → add features and lengthen history.

## Phase 2 – Enhanced features + longer history (J18 / J24)

| # | Experiment | Script | Result |
|---|---|---|---|
| 2.1 | 6 enhanced temporal features (13 total) | `preprocessing/2023Q1/stepF1_*`, `stepJ5_*` | Features for delinquency dynamics |
| 2.2 | J18 cohort, SCER-Mamba vs LSTM | `step12_final_results_v2.py` | PR-AUC 0.291 vs 0.308 |
| 2.3 | J24 cohort, SCER-Mamba vs LSTM | same | PR-AUC 0.326 vs 0.368 |
| 2.4 | **Experiment K**: multi-pool SCER-Mamba (attention + mean + last) | `step16_*`, `step17_*` | 3-seed val PR-AUC 0.309 ± 0.010 |

**Lesson:** longer history roughly doubles PR-AUC (0.18 → 0.33). Multi-pooling stabilises SCER-Mamba.

## Phase 3 – Colab experiments on J24 (2023 Q1, 24 months)

Test set: 257 risky / 24,235 not risky.

| # | Experiment | Script | Result |
|---|---|---|---|
| 3.1 | Recovery of J24 data + Exp. K models after Colab reset | `src/J24_final/01_recovery_and_balancing.py` | – |
| 3.2 | **Balancing study** (seed 42): focal, under, over, hybrid | `02_balancing_experiment.py` | PR-AUC 0.335 / 0.283 / 0.325 / 0.314 – **focal best** |
| 3.3 | Final comparison: LSTM ×3, Exp. K ×3, ensembles, rule, LR, XGBoost | `04_final_comparison.py` | LSTM 0.364, K 0.343, XGB 0.345, LR 0.327, rule 0.187 |
| 3.4 | Architecture search S1–S8 (val) | `05_improvement_round.py`, `06_tuning_with_drive.py` | S3 (1 layer) best, val 0.3255 |
| 3.5 | Final SCER-Mamba S3, focal, 3 seeds | `06_tuning_with_drive.py` | **0.354 ± 0.002**, F1 0.441 |
| 3.6 | Final SCER-Mamba S3, oversampling 1:1, 3 seeds | `07_final_balanced_model.py` | 0.324 ± 0.007, F1 0.436 – balancing does not help |
| 3.7 | Figures 1–6 + attention ratio | `09_final_figures_and_attention.py` | attention ratio 3.69× |

## Phase 4 – 2015 Q1 history-length study (pre-COVID)

| # | Experiment | Script | Result |
|---|---|---|---|
| 4.1 | Raw split + COVID cut-off | `preprocessing/2015Q1/step0_split_raw_2015Q1.py` | static + temporal CSVs |
| 4.2 | Build 12/24/36/48-month cohorts | `preprocessing/2015Q1/stepL1_build_2015Q1_cohorts.py` | 4 cohorts, labels verified (0 mismatches) |
| 4.3 | SCER-Mamba vs LSTM vs XGBoost per history length, 3 seeds | `src/2015Q1_study/01_history_length_study_colab.py` (Kaggle version `02_...`) | see table in [07_RESULTS.md](07_RESULTS.md) |
| 4.4 | Paired bootstrap significance | `03_significance_test.py` | p = 0.12, 0.14, 0.46, 0.53 – no significant difference |
| 4.5 | **Experiment B**: + 70 categorical one-hot features (48 m) | `04_experiment_B_categorical.py` | SCER 0.317 vs LSTM 0.312 vs XGB 0.299; p = 0.757 |
| 4.6 | Figures 7–11 | `05_final_figures_2015Q1.py` | attention ratio 3.83× |
| 4.7 | **Experiment C**: multi-task (aux. next-month 30+ target), 48 m + categorical | `06_experiment_C_multitask.py` | **In progress**: SCER seed 42 PR-AUC 0.318, F1 0.389, ROC 0.934. Seed 123 stopped at epoch ~11 (Colab GPU limit). Seed 2026 and LSTM not yet run. Script resumes automatically. |

## Practical issues met (and fixes)

| Issue | Fix |
|---|---|
| Colab runtime resets lost trained models | All outputs written to Google Drive; tuning results pre-filled so finished configs are skipped |
| Colab GPU quota | Kaggle version of the history-length script; resumable checkpoints |
| Interrupted training left half-written `.pt` files | Experiment C writes `*.partial` then `os.replace()` to the final name |
| `mamba-ssm` CUDA build too slow on Colab | Used HuggingFace `MambaModel` pure-PyTorch path (same maths, slower) |
| VS Code project zip contained `.venv` (GBs) | Virtual environment and data excluded from the repo (`.gitignore`) |
