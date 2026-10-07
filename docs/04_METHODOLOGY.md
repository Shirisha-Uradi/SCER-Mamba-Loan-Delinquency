# 4. Methodology

## 4.1 Models compared

| Model | Input | Notes |
|---|---|---|
| **Rule baseline** | last month | Flag the loan if it is currently 30+ days late. |
| **Logistic Regression** | hand-crafted tabular features | Static (12) + last month + last-3-month mean + full-window mean + full-window max of the 13 temporal features; `class_weight="balanced"`, standardised. |
| **Random Forest** (early stage only) | static + age-12 snapshot | Threshold tuned on validation. |
| **XGBoost** | same tabular features as LR | 1,000 trees max, lr 0.03, depth 5, subsample 0.8, colsample 0.8, early stopping (50) on validation PR-AUC. |
| **LSTM** | sequence + static | 2-layer LSTM (hidden 64, dropout 0.2) on the sequence; static MLP 12→32→64; concat → 128→64→1. Same loss, optimiser, early stopping and seeds as SCER-Mamba. |
| **SCER-Mamba** | sequence + static | Proposed model, below. |

## 4.2 SCER-Mamba architecture

**S**tatic-**C**onditioned, **E**vent-aware, **R**isk-attentive Mamba.

### Final version (configuration "S3", used for all final results)

1. **Static encoder:** `Linear(12→32) → ReLU → Dropout(0.2) → Linear(32→64) → ReLU` → static embedding *s* (64).
2. **Temporal projection:** `Linear(13→64)` applied to each month.
3. **Mamba encoder:** HuggingFace `transformers.MambaModel`, 1 layer, hidden 64, state size 16, expand 2, conv kernel 4. Mamba is a selective state-space model: like an RNN it reads the sequence in order and keeps a hidden state, but its state update parameters depend on the input, so it can decide month-by-month what to remember or forget. It is linear in sequence length and parallelisable.
4. **LayerNorm** over the Mamba outputs *h₁…h_T*.
5. **Risk-event attention:** for each month *t*, score `a_t = w·h_t + MLP(flags_t)` where `flags_t` = columns 4–6 (30+/60+/90+ late indicators). Softmax over time → weights α_t; attention vector `Σ α_t h_t`. The flag branch lets the model put more weight on months with payment problems, which makes the attention interpretable.
6. **Multi-pooling:** attention vector, mean of h_t, last state h_T.
7. **Fusion:** concatenate [attention, mean, last, *s*] = 256 dims.
8. **Classifier:** `256→128→ReLU→Dropout→32→ReLU→1` → logit → sigmoid probability.

### Components tried and removed (ablation-driven)

| Component | What it did | Why removed |
|---|---|---|
| Static conditioning gate (FiLM-style) | Static embedding scaled/shifted the temporal projection | Removing it **improved** PR-AUC (0.177 → 0.181, early 12-month data) |
| Adaptive gated fusion | Learned gate mixing static and temporal vectors | Removing it **improved** PR-AUC (0.177 → 0.181) |
| Risk-event attention | — | **Kept**: removing it lowered PR-AUC (0.177 → 0.174) and lost interpretability |

### Architecture search on J24 (validation PR-AUC, seed 42)

| Config | Change vs base | Val PR-AUC |
|---|---|---|
| S1 base | 2 layers, hidden 64, dropout 0.2, lr 3e-4 | 0.3007 |
| S2 | hidden 128 | 0.3153 |
| **S3** | **1 Mamba layer** | **0.3255** |
| S4 | 3 Mamba layers | 0.3255 |
| S5 | lr 1e-3 | 0.3108 |
| S6 | dropout 0.3, wd 1e-3 | 0.3033 |
| S7 | state size 32 | 0.3090 |
| S8 | bidirectional Mamba | 0.3129 |

S3 and S4 tie; S3 was chosen because it is smaller and faster (Occam's razor).

## 4.3 Loss functions

* **Focal loss** (Lin et al., 2017), γ = 2: `FL = −(1−p_t)^γ log(p_t)`. Down-weights easy, confidently-correct examples (the vast majority of non-risky loans) so training focuses on hard cases. Used on the real class ratio.
* **BCE with logits**: used with resampled (balanced) training sets, because applying focal loss on top of resampling would correct the imbalance twice.
* **Multi-task (Experiment C):** `L = FL(main) + λ · BCE(aux)`, λ = 0.5, where the auxiliary target is, for every month t in the window, "is the loan 30+ days late in month t+1?" (computed only inside the observed window, so no leakage into the 6-month label window).

## 4.4 Class-imbalance strategies

All applied to the **training split only**. Validation and test always keep the real ~1% ratio, so reported metrics reflect real-world prevalence.

| Strategy | How | Train risky / not risky (J24) |
|---|---|---|
| Real ratio + focal loss (default) | no resampling | 1,198 / 113,098 |
| Class weights | `class_weight="balanced"` (used for the Logistic Regression baseline) | 1,198 / 113,098 |
| Weighted random sampler (Phase 1) | sample each batch ~50/50 | real counts, re-drawn per epoch |
| Random undersampling | keep all risky, randomly keep equal number of non-risky | 1,198 / 1,198 |
| Random oversampling | keep all, repeat risky (with replacement) to 1:1 | 113,098 / 113,098 |
| Hybrid | oversample risky to 20,000 + undersample non-risky to 20,000 | 20,000 / 20,000 |

Code: `make_training_indices()` in `src/J24_final/02_balancing_experiment.py`; counts printed by `src/J24_final/03_show_balancing_counts.py`.

**Why SMOTE was not used:** SMOTE interpolates between feature vectors. For monthly sequences it creates impossible histories (e.g. "0.4 of a missed payment", balances that do not amortise), and interpolated binary delinquency flags break the risk-event attention.

## 4.5 Training protocol

| Setting | Value |
|---|---|
| Optimiser | AdamW, lr 3e-4, weight decay 1e-4 |
| Batch size | 256 |
| Gradient clipping | 1.0 |
| Max epochs / early stopping | 20 / patience 4 on validation PR-AUC (best checkpoint restored) |
| Seeds | 42, 123, 2026 (Python, NumPy, PyTorch, DataLoader generator) |
| Hardware | Google Colab T4 GPU; Mamba ran in the pure-PyTorch fallback (no `mamba-ssm` CUDA kernels), so it is slower than an LSTM in this setup |

## 4.6 Evaluation protocol

* **Primary metric: PR-AUC (average precision).** With ~1% positives, PR-AUC measures how well risky loans are ranked at the top; a random model scores ≈ the positive rate (0.01).
* **Threshold-based metrics** (precision, recall, F1, accuracy, confusion matrix): threshold = the value that maximises F1 on **validation**, then frozen and applied to **test**.
* **ROC-AUC** reported for completeness (it looks high for every model on imbalanced data, so it is not used to choose models).
* **Accuracy** is reported only because it was requested; it is ~0.99 for every model including "predict all non-risky", so it is not informative here.
* **Seeds:** mean ± SD over 3 seeds, plus a **3-seed ensemble** (average of predicted probabilities).
* **Significance:** paired bootstrap on the test set (2,000 resamples of test loans, same resample for both models), two-sided p-value for the PR-AUC difference between the two 3-seed ensembles.

## 4.7 Interpretability

For each test loan we extract the attention weights α_t and compare the average weight on months where the loan was 30+ days late to months where it was current. Ratio > 1 means the model focuses on risk events. Results: **3.69×** (2023 Q1, J24) and **3.83×** (2015 Q1, 48 months). Plots: `figures/main/fig5`, `fig6`, `fig10`, `fig11`.
