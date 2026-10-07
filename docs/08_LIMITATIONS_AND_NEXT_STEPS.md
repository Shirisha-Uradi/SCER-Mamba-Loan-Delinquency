# 8. Limitations and Next Steps

This page answers the two open questions raised in review:
**(1) find a better solution for balancing the data, and (2) find the best model.**

---

## 8.1 Why the results look weak

| Observation | Explanation |
|---|---|
| F1 ≈ 0.44, PR-AUC ≈ 0.35 look low | With 1.05% positives a random model scores PR-AUC 0.0105; the models are **~34× better than random**. F1 of 0.40–0.50 is typical for 1%-prevalence credit-event prediction. Accuracy is 98.6% but meaningless here. |
| SCER-Mamba does not beat the LSTM | Only ~1,200 risky loans for training and ~170–260 in each test set. Differences under ~0.03 PR-AUC are inside noise (bootstrap p > 0.1 everywhere). A bigger model cannot learn much more from 1,200 positives. |
| Balancing did not help | Random resampling does not add information: oversampling repeats the same 1,198 risky loans; undersampling discards 99% of normal loans. Both hurt probability calibration. Focal loss already re-weights hard examples. |
| Strong rule baseline | Most future 60+ delinquencies are already 30+ late now. The models' extra value is in the **early** cases (still current) – which are inherently hard. |

## 8.2 Limitations

1. **Few positive examples** per cohort (~1,000–1,200).
2. **One quarter at a time**; no out-of-time test (train on older vintages, test on a newer one).
3. **Mamba ran in the slow pure-PyTorch path**, which limited hyper-parameter search and number of seeds.
4. **One sample per loan** at a fixed loan age – ignores most of each loan's life.
5. No macro-economic features (unemployment, house-price index, interest-rate environment).
6. Hyper-parameters were tuned for SCER-Mamba but **not for the LSTM** (the LSTM still wins on PR-AUC, which makes the comparison conservative but also shows the LSTM is a strong baseline).
7. Experiment C (multi-task) is incomplete.

## 8.3 Plan A – better balancing (data side)

Ordered by expected gain / effort.

| # | Idea | Why it should help | Effort |
|---|---|---|---|
| A1 | **More real positives: pool several quarters** (e.g. 2014Q1–2016Q4, pre-COVID) | The real fix for imbalance is more minority examples, not copies. 12 quarters ≈ 12–15k risky loans instead of ~1.2k. | Medium (preprocessing scripts already parameterised by quarter) |
| A2 | **Sliding-window sampling for risky loans** | Use several observation months per loan (e.g. ages 24, 27, 30 …). Each risky loan gives several *different* real sequences before its default – natural oversampling. Split by loan ID to avoid leakage. | Medium |
| A3 | **Class-balanced / tuned focal loss** (α, γ grid; Cui et al. 2019 "effective number of samples") | Cheap tuning of the loss that already works best | Low |
| A4 | **Moderate resampling ratios** (e.g. 1:4, 1:10 instead of 1:1) | 1:1 is the extreme; mild ratios often beat both 1:1 and none | Low (script `src/J24_final/08_balance_ratio_test_optional.py` is ready) |
| A5 | **Sequence-aware augmentation** for risky loans: jitter of balances/rates, small time shifts, masking of non-risk months | Creates *plausible* new histories, unlike SMOTE | Medium |
| A6 | **Probability calibration** (Platt / isotonic on validation) after resampling | Fixes the over-confidence (~0.97 thresholds) caused by oversampling | Low |
| A7 | Generative oversampling (TimeGAN / VAE on risky sequences) | Research-level; risk of unrealistic sequences | High |

## 8.4 Plan B – finding the best model (model side)

| # | Idea | Why |
|---|---|---|
| B1 | **Fair tuning of all models** (Optuna, same budget for LSTM, GRU, TCN, Transformer, SCER-Mamba, XGBoost) | Current comparison tuned only SCER-Mamba |
| B2 | **Add GRU, Temporal CNN and a small Transformer** baselines | A standard benchmark table |
| B3 | **Hybrid SCER-Mamba + XGBoost** (stack / average probabilities) | Trees and sequence models make different errors; ensembles usually add +0.01–0.03 PR-AUC |
| B4 | **Install `mamba-ssm` CUDA kernels** (or run on a local GPU) | 5–10× faster Mamba → more seeds and tuning |
| B5 | **Finish Experiment C** (multi-task) | One seed already matches Experiment B |
| B6 | **Macro features** (state unemployment, FHFA house-price index) added per month | Delinquency is strongly driven by local economy |
| B7 | **Out-of-time evaluation**: train 2014–2015, test 2016 | Realistic deployment test; reviewers expect it |
| B8 | Report **cost-based metrics** (recall at 1% / 5% of loans flagged, expected loss saved) | Closer to how a lender would use the model |

## 8.5 Recommended order for the next round

1. A1 + A2 (more real positives) – biggest expected impact on both balancing and model comparison.
2. A3 + A4 + A6 (quick balancing variants) on the bigger data.
3. B1 + B2 + B4 (fair tuned benchmark with faster Mamba).
4. B3 (ensemble) and B5 (multi-task) as the "proposed method" candidates.
5. B7 out-of-time test for the final table, with bootstrap significance as already implemented.

## 8.6 What can already be claimed

* A complete, leakage-safe, reproducible pipeline for sequential mortgage-delinquency prediction.
* SCER-Mamba is **statistically on par** with an LSTM, better than XGBoost on long histories, more stable across seeds, and **interpretable** (attention focuses 3.7–3.8× on late-payment months).
* Focal loss on the real distribution is a better imbalance strategy than random over/under/hybrid sampling for this task.
* History length matters more than architecture: going from 12 to 36 months improves PR-AUC by ~0.12, while model choice changes it by ~0.01–0.03.
