# 6. How to Run

## 6.1 Environment

* Python 3.10–3.11
* `pip install -r requirements.txt`
* GPU recommended for the deep models (Google Colab T4 was used). CPU works for preprocessing and XGBoost.

## 6.2 Get the data

1. Register and download from Fannie Mae Data Dynamics: *Single-Family Loan Performance Data* → acquisition quarters **2023Q1** and **2015Q1**.
2. Unzip into a local `Data/` folder:
   ```
   Data/2023Q1.csv
   Data/2015Q1.csv
   ```
   (`Data/` is in `.gitignore` – never commit it.)

## 6.3 2023 Q1 – build the 24-month dataset (local)

The 2023 Q1 cleaning step expects the raw file already split into `Data/2023Q1_CLEANED_LoanLevel(static data).csv` and `Data/2023Q1_CLEANED_Temporal.csv` (the original 2023 split script is not part of the local project folder). The simplest way to reproduce it is `preprocessing/2015Q1/step0_split_raw_2015Q1.py` with `QUARTER = "2023Q1"` and `COVID_CUTOFF` set to a date after the last reporting month (e.g. `"2100-01-01"`).

Then run from the repository root, in this order:

```bash
python preprocessing/2023Q1/data_analysis.py              # optional EDA
python preprocessing/2023Q1/step2_data_cleaning.py
python preprocessing/2023Q1/step2_temporal_cleaning.py
python preprocessing/2023Q1/step2_final_validation.py
python preprocessing/2023Q1/step3_target_metric.py
python preprocessing/2023Q1/step3_target_validation.py
python preprocessing/2023Q1/stepJ4_build_18_24_sequences.py
python preprocessing/2023Q1/stepJ5_enhance_18_24_sequences.py
python preprocessing/2023Q1/stepJ6_split_normalise_18_24.py
python preprocessing/2023Q1/stepJ7_zip_18_24.py           # → sequential_enhanced_model_ready_age24.zip
```

(The step4/step5/stepF scripts rebuild the earlier 12-month datasets used in Phase 1.)

## 6.4 2023 Q1 – experiments (Google Colab)

1. Upload `sequential_enhanced_model_ready_age24.zip` to Google Drive folder `MyDrive/SCER_Mamba/` (scripts save outputs there too).
2. Open a Colab notebook with **GPU** runtime; mount Drive.
3. Run, each as one cell, in order:

| Order | Script | Produces |
|---|---|---|
| 1 | `src/J24_final/01_recovery_and_balancing.py` | Loads J24, defines models/losses, data loaders |
| 2 | `src/J24_final/03_show_balancing_counts.py` | Label counts before/after balancing |
| 3 | `src/J24_final/02_balancing_experiment.py` | Balancing comparison table |
| 4 | `src/J24_final/04_final_comparison.py` | LSTM / SCER / XGB / LR / rule table |
| 5 | `src/J24_final/06_tuning_with_drive.py` | S1–S8 search + final S3 3-seed results (saved to Drive, resumable) |
| 6 | `src/J24_final/07_final_balanced_model.py` | S3 with oversampling, 3 seeds (self-contained) |
| 7 | `src/J24_final/09_final_figures_and_attention.py` | Figures 1–6, `FINAL_comparison_table.csv`, attention ratio |

`05_improvement_round.py` is the earlier non-Drive version of step 5; `08_balance_ratio_test_optional.py` tests milder oversampling ratios (1:4 and 1:10) against focal loss and 1:1 and has not been run yet.

## 6.5 2015 Q1 – history-length study

Local:
```bash
python preprocessing/2015Q1/step0_split_raw_2015Q1.py
python preprocessing/2015Q1/stepL1_build_2015Q1_cohorts.py    # → Data/fm2015_age{12,24,36,48}.zip
```
Upload the four zips to `MyDrive/`. Then in Colab (GPU):

| Order | Script | Notes |
|---|---|---|
| 1 | `src/2015Q1_study/01_history_length_study_colab.py` | Set `AGES_TO_RUN`; results saved per age to Drive |
| 2 | `src/2015Q1_study/03_significance_test.py` | Uses saved models; no retraining |
| 3 | `src/2015Q1_study/04_experiment_B_categorical.py` | 48 m + categorical |
| 4 | `src/2015Q1_study/05_final_figures_2015Q1.py` | Figures 7–11 |
| 5 | `src/2015Q1_study/06_experiment_C_multitask.py` | Resumable; re-run the cell after a disconnect |

Kaggle alternative: `02_history_length_study_kaggle.py` (no Drive; attach the zips as a Kaggle dataset).

## 6.6 Using the saved final models

`models/S3_layers1_seed{42,123,2026}.pt` are PyTorch `state_dict`s of the final SCER-Mamba (S3) trained on 2023 Q1 J24. Build the model class from `src/J24_final/06_tuning_with_drive.py` (`TunableSCERMamba` with the S3 config: 1 layer, hidden 64, dropout 0.2) and call `model.load_state_dict(torch.load(path))`. Inputs must be preprocessed with the same J24 train statistics.

## 6.7 Original notebook

`notebooks/RadarSeq_colab.ipynb` is the original 114-cell Colab notebook (SCER-Mamba v2, sampler/focal experiments, Experiments C/D/E/K). It is kept for traceability; the scripts in `src/` are the cleaned, final versions.

## 6.8 Path note for the local scripts

The local scripts (`preprocessing/2023Q1/`, `src/early_experiments/`) use relative paths such as `Data/...` and `output/...`. Run them **from the repository root** after creating those two folders (`mkdir Data output`); the scripts create their own sub-folders.
