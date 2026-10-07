# 1. Project Overview

## 1.1 Motivation

Mortgage lenders and servicers want to know **which currently-performing loans are likely to become seriously delinquent soon**, so they can contact the borrower, offer forbearance or a modification, and reduce losses. Traditional credit-risk models use only origination data (credit score, debt-to-income, loan-to-value). But a loan's **monthly repayment behaviour after origination** carries strong signals: a missed payment, a growing balance, repeated short delays.

This project treats each loan's repayment history as a **time series** and asks whether a modern state-space sequence model (**Mamba**), combined with the loan's static profile and an attention mechanism focused on risk events, can predict near-term serious delinquency better than classic baselines.

## 1.2 Research questions

1. **RQ1 – Performance:** Does SCER-Mamba predict 60+ day delinquency within 6 months better than Logistic Regression, XGBoost, a rule baseline and an LSTM?
2. **RQ2 – Imbalance:** Only ~1% of loans become risky. Which way of handling this works best: focal loss, class weights, weighted sampling, random undersampling, random oversampling or a hybrid?
3. **RQ3 – History length:** How much repayment history (12, 24, 36, 48 months) is needed, and does a longer history favour the sequence models?
4. **RQ4 – Interpretability:** Does the model's attention focus on economically meaningful months (months when the borrower was late)?
5. **RQ5 – Extra information:** Do categorical loan attributes (purpose, occupancy, property type, channel, first-time buyer, state) and an auxiliary multi-task target help?

## 1.3 Contributions

* A leakage-safe pipeline that turns raw Fannie Mae performance files into fixed-length, one-sample-per-loan sequences with 6-month forward labels, for several history lengths.
* **SCER-Mamba**: a Mamba sequence encoder conditioned on static loan features, with a risk-event attention head and multi-pooling (attention + mean + last state).
* A systematic study of six class-imbalance strategies on the same model, same split and same seeds.
* A history-length study on a pre-COVID cohort (2015 Q1, cut at February 2020) with paired-bootstrap significance tests.
* Ablations (static conditioning, risk attention, gated fusion), architecture search (8 configurations) and an interpretability analysis of attention weights.

## 1.4 Answers so far (short)

| RQ | Answer |
|---|---|
| RQ1 | SCER-Mamba ≈ LSTM (no significant difference, p > 0.1 everywhere); both beat Logistic Regression and the rule; SCER-Mamba beats XGBoost on 36- and 48-month histories. |
| RQ2 | **Focal loss on the real ratio** is best. Oversampling is close; hybrid and undersampling are worse. |
| RQ3 | Performance rises strongly from 12 → 24 → 36 months, then flattens at 48 months. |
| RQ4 | Yes: 3.7–3.8× more attention on 30+-day-late months than on on-time months. |
| RQ5 | Categorical features give a small gain for SCER-Mamba (0.308 → 0.317 PR-AUC, 48 m); multi-task learning is still running (1 of 3 seeds finished). |

## 1.5 Project timeline (high level)

1. **Local pipeline (VS Code, Aug–Sep 2026):** data analysis, cleaning, targets, static baselines (Logistic Regression, Random Forest), 12-month sequences, LSTM, SCER pre-Mamba, SCER-Mamba v2, ablations.
2. **Enhanced features & longer history (late Sep 2026):** 7 extra temporal features, 18- and 24-month cohorts (J18, J24), Experiment K (multi-pool SCER-Mamba).
3. **Colab experiments on J24 (Oct 2026):** balancing study, final comparison with 3 seeds, architecture tuning (S1–S8), final balanced model, figures and attention analysis.
4. **2015 Q1 study (Oct 2026):** raw split, 12/24/36/48-month cohorts, history-length study, significance test, Experiment B (categorical), Experiment C (multi-task, in progress).

See [05_EXPERIMENT_LOG.md](05_EXPERIMENT_LOG.md) for every step.
