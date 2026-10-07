import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ============================================================
# STEP 4.2 - TRAIN / VALIDATION / TEST SPLIT + NORMALISATION
# ============================================================

input_file = "Data/step4_baseline_age12_dataset.csv"

output_folder = "Data/model_ready"
os.makedirs(output_folder, exist_ok=True)

print("========================================")
print("STEP 4.2 - SPLIT AND NORMALISE")
print("========================================")


# ============================================================
# 1. LOAD MODELLING DATA
# ============================================================

df = pd.read_csv(
    input_file,
    low_memory=False,
    dtype={
        "loan_identifier": "string"
    }
)

print("\nDataset loaded.")
print("Rows:", len(df))
print("Columns:", df.shape[1])


# ============================================================
# 2. CHECK TARGET
# ============================================================

print("\nTarget distribution:")
print(
    df["target_risk_6m"]
    .value_counts()
    .sort_index()
)

print("\nTarget percentages:")
print(
    df["target_risk_6m"]
    .value_counts(normalize=True)
    .sort_index()
    * 100
)


# ============================================================
# 3. TRAIN / VALIDATION / TEST SPLIT
#
# 70% TRAIN
# 15% VALIDATION
# 15% TEST
#
# Stratification preserves the rare risky class.
# ============================================================

train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    random_state=42,
    stratify=df["target_risk_6m"]
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=42,
    stratify=temp_df["target_risk_6m"]
)

print("\n========================================")
print("SPLIT SIZES")
print("========================================")

print(
    "Training:",
    len(train_df)
)

print(
    "Validation:",
    len(val_df)
)

print(
    "Test:",
    len(test_df)
)


# ============================================================
# 4. VERIFY THERE IS NO LOAN OVERLAP
# ============================================================

train_ids = set(
    train_df["loan_identifier"]
)

val_ids = set(
    val_df["loan_identifier"]
)

test_ids = set(
    test_df["loan_identifier"]
)

train_val_overlap = len(
    train_ids.intersection(val_ids)
)

train_test_overlap = len(
    train_ids.intersection(test_ids)
)

val_test_overlap = len(
    val_ids.intersection(test_ids)
)

print("\nLoan overlap checks:")

print(
    "Train vs Validation:",
    train_val_overlap
)

print(
    "Train vs Test:",
    train_test_overlap
)

print(
    "Validation vs Test:",
    val_test_overlap
)


# ============================================================
# 5. DEFINE NUMERICAL FEATURES
# ============================================================

continuous_features = [
    "borrower_credit_score_at_origination",
    "co_borrower_credit_score_at_origination",
    "dti",
    "original_ltv",
    "original_cltv",
    "original_upb",
    "original_interest_rate",
    "number_of_borrowers",
    "mortgage_insurance_percentage",
    "loan_age",
    "current_interest_rate",
    "current_actual_upb",
    "remaining_months_to_maturity"
]


# Binary / already encoded variables
binary_features = [
    "has_co_borrower",
    "co_borrower_score_missing",
    "mortgage_insurance_missing",
    "delinquency_status_numeric"
]


# ============================================================
# 6. CONVERT FEATURES TO NUMERIC
# ============================================================

all_model_features = (
    continuous_features +
    binary_features
)

for column in all_model_features:

    train_df[column] = pd.to_numeric(
        train_df[column],
        errors="coerce"
    )

    val_df[column] = pd.to_numeric(
        val_df[column],
        errors="coerce"
    )

    test_df[column] = pd.to_numeric(
        test_df[column],
        errors="coerce"
    )


# ============================================================
# 7. TRAIN-ONLY MEDIAN IMPUTATION
#
# Important:
# Medians are calculated ONLY from training data.
# ============================================================

training_medians = {}

print("\n========================================")
print("MISSING VALUE IMPUTATION")
print("========================================")

for column in continuous_features:

    median_value = train_df[column].median()

    training_medians[column] = median_value

    train_missing = train_df[column].isna().sum()
    val_missing = val_df[column].isna().sum()
    test_missing = test_df[column].isna().sum()

    if (
        train_missing > 0
        or val_missing > 0
        or test_missing > 0
    ):

        print(
            column,
            "median =",
            median_value
        )

        print(
            "  Missing before filling:",
            "Train =", train_missing,
            "Validation =", val_missing,
            "Test =", test_missing
        )

    train_df[column] = (
        train_df[column]
        .fillna(median_value)
    )

    val_df[column] = (
        val_df[column]
        .fillna(median_value)
    )

    test_df[column] = (
        test_df[column]
        .fillna(median_value)
    )


# ============================================================
# 8. NORMALISATION
#
# StandardScaler is FIT ONLY on training data.
# ============================================================

scaler = StandardScaler()

train_df[continuous_features] = (
    scaler.fit_transform(
        train_df[continuous_features]
    )
)

val_df[continuous_features] = (
    scaler.transform(
        val_df[continuous_features]
    )
)

test_df[continuous_features] = (
    scaler.transform(
        test_df[continuous_features]
    )
)

print(
    "\nContinuous numerical features normalised."
)


# ============================================================
# 9. FINAL MODEL FEATURES
# ============================================================

final_features = (
    continuous_features +
    binary_features
)

print("\nNumber of model features:")
print(len(final_features))

print("\nModel features:")

for feature in final_features:
    print("-", feature)


# ============================================================
# 10. FINAL MISSING VALUE CHECK
# ============================================================

print("\n========================================")
print("FINAL MISSING VALUE CHECK")
print("========================================")

print(
    "Training missing:",
    train_df[final_features]
    .isna()
    .sum()
    .sum()
)

print(
    "Validation missing:",
    val_df[final_features]
    .isna()
    .sum()
    .sum()
)

print(
    "Test missing:",
    test_df[final_features]
    .isna()
    .sum()
    .sum()
)


# ============================================================
# 11. TARGET DISTRIBUTION IN EACH SPLIT
# ============================================================

print("\n========================================")
print("TARGET DISTRIBUTIONS")
print("========================================")

print("\nTRAIN:")
print(
    train_df["target_risk_6m"]
    .value_counts()
    .sort_index()
)

print("\nVALIDATION:")
print(
    val_df["target_risk_6m"]
    .value_counts()
    .sort_index()
)

print("\nTEST:")
print(
    test_df["target_risk_6m"]
    .value_counts()
    .sort_index()
)


# ============================================================
# 12. SAVE DATASETS
# ============================================================

train_file = (
    f"{output_folder}/train_normalised.csv"
)

val_file = (
    f"{output_folder}/validation_normalised.csv"
)

test_file = (
    f"{output_folder}/test_normalised.csv"
)

train_df.to_csv(
    train_file,
    index=False
)

val_df.to_csv(
    val_file,
    index=False
)

test_df.to_csv(
    test_file,
    index=False
)


# ============================================================
# 13. SAVE SCALER AND MEDIANS
# ============================================================

joblib.dump(
    scaler,
    f"{output_folder}/standard_scaler.pkl"
)

joblib.dump(
    training_medians,
    f"{output_folder}/training_medians.pkl"
)

joblib.dump(
    final_features,
    f"{output_folder}/model_features.pkl"
)


print("\n========================================")
print("STEP 4.2 COMPLETED")
print("========================================")

print("Training rows:", len(train_df))
print("Validation rows:", len(val_df))
print("Test rows:", len(test_df))

print("\nFiles saved in:")
print(output_folder)