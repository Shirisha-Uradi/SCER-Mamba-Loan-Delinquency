# Saved models

| File | Model | Data | Test PR-AUC |
|---|---|---|---|
| `S3_layers1_seed42.pt` | SCER-Mamba S3 (1 Mamba layer, hidden 64), focal loss | 2023 Q1, 24-month history (J24) | 0.354 |
| `S3_layers1_seed123.pt` | same, seed 123 | same | 0.352 |
| `S3_layers1_seed2026.pt` | same, seed 2026 | same | 0.356 |

These are PyTorch `state_dict` files. Load with the `TunableSCERMamba` class in `src/J24_final/06_tuning_with_drive.py` using the S3 configuration. Input data must be preprocessed with the J24 pipeline (same train-set imputation and scaling statistics).
