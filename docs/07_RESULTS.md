# 7. Results

All numbers are on the **held-out test set** with the real class ratio. Threshold = F1-maximising value on validation. "Mean ± SD" = over seeds 42, 123, 2026. Source CSVs are in `results/`.

---

## 7.1 Main comparison – 2023 Q1, 24-month history (J24)

Test set: 257 risky / 24,235 not risky (1.05%). Source: `results/2023Q1/FINAL_comparison_table.csv`.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| LSTM (3 seeds) | 0.987 | 0.405 ± 0.013 | 0.511 ± 0.020 | 0.452 ± 0.011 | 0.921 ± 0.001 | **0.364 ± 0.004** |
| **SCER-Mamba S3 + focal (3 seeds)** | 0.986 ± 0.001 | 0.382 ± 0.020 | 0.521 ± 0.010 | 0.441 ± 0.010 | 0.917 ± 0.002 | 0.354 ± 0.002 |
| SCER-Mamba Exp. K (untuned, 3 seeds) | – | 0.387 ± 0.010 | 0.520 ± 0.019 | 0.444 ± 0.004 | 0.917 ± 0.002 | 0.343 ± 0.012 |
| XGBoost | – | 0.357 | 0.545 | 0.431 | 0.914 | 0.345 |
| SCER-Mamba S3 + oversampling 1:1 (3 seeds) | 0.986 ± 0.001 | 0.388 ± 0.019 | 0.502 ± 0.047 | 0.436 ± 0.015 | 0.900 ± 0.005 | 0.324 ± 0.007 |
| Logistic Regression | – | 0.396 | 0.424 | 0.410 | 0.917 | 0.327 |
| Rule: currently 30+ days late | – | 0.416 | 0.436 | 0.426 | 0.715 | 0.187 |

### Per-seed – final SCER-Mamba S3 + focal (`results/2023Q1/final_balanced/focal_3seeds_test.csv`)

| Seed | Val PR-AUC | Threshold | Precision | Recall | F1 | ROC-AUC | PR-AUC | TP | FP | FN |
|---|---|---|---|---|---|---|---|---|---|---|
| 42 | 0.326 | 0.389 | 0.402 | 0.514 | 0.451 | 0.916 | 0.354 | 132 | 196 | 125 |
| 123 | 0.324 | 0.373 | 0.362 | 0.533 | 0.431 | 0.915 | 0.352 | 137 | 241 | 120 |
| 2026 | 0.320 | 0.421 | 0.382 | 0.518 | 0.440 | 0.918 | 0.356 | 133 | 215 | 124 |

Interpretation of seed 42: of 328 loans flagged, 132 really became 60+ days late (precision 40%); the model caught 132 of 257 risky loans (recall 51%). For comparison, a random selection of 328 loans would contain ~3–4 risky ones.

### Per-seed – SCER-Mamba S3 + oversampling 1:1 (`balanced_3seeds_test.csv`)

| Seed | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| 42 | 0.406 | 0.502 | 0.449 | 0.896 | 0.329 |
| 123 | 0.390 | 0.455 | 0.420 | 0.900 | 0.316 |
| 2026 | 0.368 | 0.549 | 0.441 | 0.905 | 0.327 |
| 3-seed ensemble | 0.408 | 0.459 | 0.432 | 0.905 | 0.329 |

Note the thresholds for the oversampled model are ~0.97: resampling makes the model's probabilities very over-confident on the real data (poor calibration), which is one reason ranking quality (PR-AUC) drops.

## 7.2 Class-balancing study (J24, seed 42)

| Method | Train risky | Train not risky | F1 | PR-AUC |
|---|---|---|---|---|
| **Focal loss, real ratio** | 1,198 | 113,098 | **0.448** | **0.335** |
| Oversampling 1:1 | 113,098 | 113,098 | 0.441 | 0.325 |
| Hybrid 20k / 20k | 20,000 | 20,000 | 0.424 | 0.314 |
| Undersampling 1:1 | 1,198 | 1,198 | 0.421 | 0.283 |

Validation and test (all methods): 257 risky / 24,235 not risky – **not balanced** on purpose.

Why undersampling is worst: it throws away 99% of the non-risky loans (111,900 of 113,098), so the model never sees most normal repayment patterns. Oversampling keeps all data but repeats the same 1,198 risky loans ~94 times each, which encourages memorisation.

## 7.3 Architecture search (J24, validation PR-AUC, seed 42)

| S1 base | S2 hidden 128 | **S3 1 layer** | S4 3 layers | S5 lr 1e-3 | S6 strong reg. | S7 state 32 | S8 bidirectional |
|---|---|---|---|---|---|---|---|
| 0.3007 | 0.3153 | **0.3255** | 0.3255 | 0.3108 | 0.3033 | 0.3090 | 0.3129 |

## 7.4 Early-stage results (2023 Q1, 12-month history)

Test: 228 risky / 27,029 not risky. Source: `results/early_vscode/final_results/updated_complete_model_results.csv`.

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| LSTM | 0.316 | 0.368 | 0.340 | 0.856 | 0.189 |
| Exp. D (no static conditioning), 3-seed mean | 0.294 | 0.327 | 0.310 | 0.856 | 0.182 |
| Exp. D single | 0.304 | 0.351 | 0.326 | 0.861 | 0.182 |
| Focal loss only | 0.313 | 0.329 | 0.321 | 0.858 | 0.179 |
| SCER pre-Mamba | 0.253 | 0.338 | 0.290 | 0.856 | 0.177 |
| SCER-Mamba v2 | 0.296 | 0.298 | 0.297 | 0.853 | 0.169 |
| Sampler + focal | 0.296 | 0.329 | 0.312 | 0.848 | 0.165 |
| Sampler only | 0.302 | 0.263 | 0.281 | 0.849 | 0.160 |

### Ablation (12-month data) – `results/early_vscode/ablation/ablation_comparison.csv`

| Variant | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|
| Full SCER | 0.289 | 0.855 | 0.177 |
| − static conditioning | 0.320 | 0.857 | 0.181 |
| − risk-event attention | 0.290 | 0.854 | 0.174 |
| − adaptive gated fusion | 0.323 | 0.857 | 0.181 |

### J18 vs J24 (single seed) – `results/early_vscode/final_results_v2/j18_j24_heldout_test_results.csv`

| Cohort | Model | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| J18 | SCER-Mamba | 0.350 | 0.405 | 0.375 | 0.886 | 0.291 |
| J18 | LSTM | 0.396 | 0.368 | 0.381 | 0.890 | 0.308 |
| J24 | SCER-Mamba | 0.365 | 0.541 | 0.436 | 0.917 | 0.326 |
| J24 | LSTM | 0.392 | 0.564 | 0.463 | 0.920 | 0.368 |

## 7.5 History-length study – 2015 Q1 (pre-COVID), mean of 3 seeds

Source: `results/2015Q1/table_history_length_means.csv`.

| History | Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|---|
| 12 m | LSTM | 0.995 | 0.286 | 0.395 | 0.329 | 0.910 | **0.216** |
| 12 m | SCER-Mamba | 0.996 | 0.292 | 0.359 | 0.322 | 0.910 | 0.203 |
| 12 m | XGBoost | 0.995 | 0.260 | 0.455 | 0.331 | 0.911 | 0.204 |
| 24 m | LSTM | 0.995 | 0.373 | 0.375 | 0.373 | 0.934 | **0.317** |
| 24 m | SCER-Mamba | 0.995 | 0.376 | 0.382 | 0.376 | 0.927 | 0.287 |
| 24 m | XGBoost | 0.995 | 0.336 | 0.400 | 0.365 | 0.935 | 0.303 |
| 36 m | LSTM | 0.995 | 0.395 | 0.371 | 0.382 | 0.936 | **0.332** |
| 36 m | SCER-Mamba | 0.994 | 0.313 | 0.465 | 0.371 | 0.936 | 0.320 |
| 36 m | XGBoost | 0.995 | 0.346 | 0.377 | 0.361 | 0.938 | 0.297 |
| 48 m | LSTM | 0.992 | 0.310 | 0.540 | 0.392 | 0.943 | **0.309** |
| 48 m | SCER-Mamba | 0.992 | 0.317 | 0.511 | 0.388 | **0.946** | 0.308 |
| 48 m | XGBoost | 0.992 | 0.306 | 0.506 | 0.381 | 0.942 | 0.281 |

### Significance (paired bootstrap, 2,000 resamples, 3-seed ensembles)

| History | SCER − LSTM PR-AUC | p-value | Significant? |
|---|---|---|---|
| 12 m | −0.013 | 0.12 | No |
| 24 m | −0.030 | 0.14 | No |
| 36 m | −0.012 | 0.46 | No |
| 48 m | −0.001 | 0.53 | No |
| 48 m + categorical | **+0.005** (95% CI −0.027 to +0.035) | 0.757 | No |

(Differences in the first four rows use 3-seed means; the bootstrap itself is run on the ensembles.)

## 7.6 Experiment B – 48 months + 70 categorical features

| Model | Precision | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|
| **SCER-Mamba** | **0.353** | 0.385 | 0.941 | **0.317** |
| LSTM | 0.323 | **0.395** | **0.946** | 0.312 |
| XGBoost | – | – | – | 0.299 |

Ensemble PR-AUC: SCER 0.3247 vs LSTM 0.3198, p = 0.757. Categorical features help SCER-Mamba (+0.009 PR-AUC vs 48 m base) more than LSTM (+0.003).

## 7.7 Experiment C – multi-task (in progress)

| Model | Seed | Status | Test PR-AUC | F1 | ROC-AUC |
|---|---|---|---|---|---|
| SCER-Mamba MT | 42 | done | 0.318 | 0.389 | 0.934 |
| SCER-Mamba MT | 123 | stopped epoch ~11 (val 0.271) | – | – | – |
| SCER-Mamba MT | 2026 | not run | – | – | – |
| LSTM MT | all | not run | – | – | – |

Seed 42 alone is roughly equal to Experiment B; no conclusion yet.

## 7.8 Interpretability

| Data | Mean attention on 30+ late months ÷ on-time months |
|---|---|
| 2023 Q1, 24 m | **3.69×** |
| 2015 Q1, 48 m | **3.83×** |

---

## Figures

| File | Caption |
|---|---|
| `figures/main/fig1_model_comparison.png` | Test metrics of all models, 2023 Q1 J24. |
| `figures/main/fig2_pr_curves.png` | Test precision–recall curves of the final SCER-Mamba 3-seed ensembles (focal: AP 0.358; oversampled: AP 0.329) vs the 30+-late rule (0.187) and random (0.010). |
| `figures/main/fig3_confusion_matrix.png` | Confusion matrix of the final SCER-Mamba focal 3-seed ensemble at the validation-chosen threshold: TP 147, FP 273, FN 110, TN 23,962 (precision 0.35, recall 0.57). |
| `figures/main/fig4_balancing_comparison.png` | F1 and PR-AUC for focal / over / hybrid / under sampling. |
| `figures/main/fig5_attention_by_month.png` | Average attention on late vs on-time months by position in the 24-month window. |
| `figures/main/fig6_attention_examples.png` | Attention profiles of individual risky test loans with their delinquency history. |
| `figures/main/fig7_history_length_all_metrics.png` | All metrics vs history length (12–48 months), 2015 Q1. |
| `figures/main/fig8_significance_scer_vs_lstm.png` | Bootstrap PR-AUC difference (SCER − LSTM) with 95% CI per history length. |
| `figures/main/fig9_experiment_B_categorical.png` | Experiment B: effect of categorical features at 48 months. |
| `figures/main/fig10_attention_by_month_48m.png` | Attention on late vs on-time months, 48-month window. |
| `figures/main/fig11_attention_examples_48m.png` | Example attention profiles, 48-month window. |
| `figures/eda/*` | Exploratory data analysis of 2023 Q1. |
| `figures/early/*` | Early-stage comparisons (12-month data, J18/J24, Experiment K). |
