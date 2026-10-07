import pandas as pd
import os

# ============================================================
# STEP 4.1 - PREPARE BASELINE MODELLING DATASET
# Prediction point: Loan Age = 12 months
# Target: 60+ delinquency in next 6 months
# ============================================================

static_file = "Data/cleaned/2023Q1_STATIC_CLEANED_STEP2.csv"
temporal_file = "Data/cleaned/2023Q1_TEMPORAL_CLEANED_STEP2.csv"
target_file = "Data/2023Q1_TARGET_6M_Risk_STEP3.csv"

output_file = "Data/step4_baseline_age12_dataset.csv"

print("========================================")
print("STEP 4.1 - PREPARING MODEL DATA")
print("========================================")


# ============================================================
# 1. GET LOAN-AGE-12 TEMPORAL OBSERVATIONS
# ============================================================

age12_parts = []

temporal_columns = [
    "loan_identifier",
    "monthly_reporting_period",
    "loan_age",
    "current_interest_rate",
    "current_actual_upb",
    "remaining_months_to_maturity",
    "delinquency_status_numeric"
]

chunk_number = 0

for chunk in pd.read_csv(
    temporal_file,
    usecols=temporal_columns,
    chunksize=500000,
    low_memory=False,
    dtype={
        "loan_identifier": "string",
        "monthly_reporting_period": "string"
    }
):

    chunk_number += 1
    print(f"Checking temporal chunk {chunk_number}...")

    chunk["loan_identifier"] = (
        chunk["loan_identifier"]
        .astype("string")
        .str.strip()
        .str.zfill(12)
    )

    chunk["loan_age"] = pd.to_numeric(
        chunk["loan_age"],
        errors="coerce"
    )

    age12 = chunk[
        chunk["loan_age"] == 12
    ].copy()

    if len(age12) > 0:
        age12_parts.append(age12)


if not age12_parts:
    raise ValueError(
        "No loan-age-12 observations found in temporal data."
    )


age12_df = pd.concat(
    age12_parts,
    ignore_index=True
)

del age12_parts

print("\nAge-12 observations:", len(age12_df))


# ============================================================
# 2. STANDARDISE AGE-12 IDENTIFIERS AND DATES
# ============================================================

age12_df["loan_identifier"] = (
    age12_df["loan_identifier"]
    .astype("string")
    .str.strip()
    .str.zfill(12)
)

age12_df["monthly_reporting_period"] = pd.to_datetime(
    age12_df["monthly_reporting_period"],
    errors="coerce"
)

print(
    "Missing Age-12 reporting dates:",
    age12_df["monthly_reporting_period"].isna().sum()
)


# ============================================================
# 3. MATCH AGE-12 OBSERVATIONS WITH TARGET
# ============================================================

matched_target_parts = []

target_chunk_number = 0
total_matches = 0

for target_chunk in pd.read_csv(
    target_file,
    chunksize=500000,
    low_memory=False,
    dtype={
        "loan_identifier": "string",
        "observation_month": "string"
    }
):

    target_chunk_number += 1

    print(
        f"\nMatching target chunk {target_chunk_number}..."
    )

    # Standardise target loan identifiers
    target_chunk["loan_identifier"] = (
        target_chunk["loan_identifier"]
        .astype("string")
        .str.strip()
        .str.zfill(12)
    )

    # Standardise target dates
    target_chunk["observation_month"] = pd.to_datetime(
        target_chunk["observation_month"],
        errors="coerce"
    )

    matched = age12_df.merge(
        target_chunk[
            [
                "loan_identifier",
                "observation_month",
                "target_risk_6m"
            ]
        ],
        left_on=[
            "loan_identifier",
            "monthly_reporting_period"
        ],
        right_on=[
            "loan_identifier",
            "observation_month"
        ],
        how="inner"
    )

    print(
        "Matches in this chunk:",
        len(matched)
    )

    total_matches += len(matched)

    if len(matched) > 0:
        matched_target_parts.append(matched)


print(
    "\nTotal target matches:",
    total_matches
)


if not matched_target_parts:
    print("\nDEBUG INFORMATION")
    print("-----------------")

    print(
        "Example Age-12 loan IDs:"
    )
    print(
        age12_df["loan_identifier"]
        .head()
        .tolist()
    )

    print(
        "\nExample Age-12 dates:"
    )
    print(
        age12_df["monthly_reporting_period"]
        .head()
        .tolist()
    )

    raise ValueError(
        "No target matches found. "
        "Loan ID or observation-month formats do not match."
    )


model_df = pd.concat(
    matched_target_parts,
    ignore_index=True
)

del matched_target_parts

print(
    "\nAge-12 observations with valid target:",
    len(model_df)
)


# ============================================================
# 4. LOAD CLEANED STATIC FEATURES
# ============================================================

static_features = [
    "loan_identifier",
    "borrower_credit_score_at_origination",
    "co_borrower_credit_score_at_origination",
    "dti",
    "original_ltv",
    "original_cltv",
    "original_upb",
    "original_interest_rate",
    "number_of_borrowers",
    "mortgage_insurance_percentage",
    "has_co_borrower",
    "co_borrower_score_missing",
    "mortgage_insurance_missing"
]

static_df = pd.read_csv(
    static_file,
    usecols=static_features,
    low_memory=False,
    dtype={
        "loan_identifier": "string"
    }
)

static_df["loan_identifier"] = (
    static_df["loan_identifier"]
    .astype("string")
    .str.strip()
    .str.zfill(12)
)

print(
    "\nStatic rows loaded:",
    len(static_df)
)


# ============================================================
# 5. MERGE STATIC + TEMPORAL + TARGET
# ============================================================

model_df = model_df.merge(
    static_df,
    on="loan_identifier",
    how="left",
    validate="many_to_one"
)

print(
    "Rows after static merge:",
    len(model_df)
)


# ============================================================
# 6. REMOVE DUPLICATES
# ============================================================

model_df = model_df.drop_duplicates(
    subset=[
        "loan_identifier",
        "monthly_reporting_period"
    ]
)

print(
    "Rows after duplicate removal:",
    len(model_df)
)


# ============================================================
# 7. CHECK MISSING STATIC MATCHES
# ============================================================

missing_static = (
    model_df[
        "borrower_credit_score_at_origination"
    ]
    .isna()
    .sum()
)

print(
    "\nRows missing static borrower data:",
    missing_static
)


# ============================================================
# 8. FINAL DATASET VALIDATION
# ============================================================

print("\n========================================")
print("MODEL DATASET VALIDATION")
print("========================================")

print(
    "Rows:",
    len(model_df)
)

print(
    "Columns:",
    model_df.shape[1]
)

print(
    "Unique loans:",
    model_df["loan_identifier"].nunique()
)

print(
    "Duplicate loan-month observations:",
    model_df.duplicated(
        subset=[
            "loan_identifier",
            "monthly_reporting_period"
        ]
    ).sum()
)

print(
    "Missing target values:",
    model_df["target_risk_6m"].isna().sum()
)


# ============================================================
# 9. TARGET DISTRIBUTION
# ============================================================

print("\nTarget distribution:")

target_counts = (
    model_df["target_risk_6m"]
    .value_counts()
    .sort_index()
)

print(target_counts)


print("\nTarget percentages:")

target_percentages = (
    model_df["target_risk_6m"]
    .value_counts(normalize=True)
    .sort_index()
    * 100
)

print(target_percentages)


# ============================================================
# 10. CHECK CURRENT DELINQUENCY
# ============================================================

print(
    "\nCurrent delinquency distribution at prediction point:"
)

print(
    model_df["delinquency_status_numeric"]
    .value_counts()
    .sort_index()
)


already_60plus = (
    model_df["delinquency_status_numeric"] >= 2
).sum()

print(
    "\nAlready 60+ delinquent at prediction point:",
    already_60plus
)


# ============================================================
# 11. SAVE FINAL BASELINE DATASET
# ============================================================

model_df.to_csv(
    output_file,
    index=False
)

print("\n========================================")
print("STEP 4.1 COMPLETED")
print("========================================")

print(
    "Final modelling rows:",
    len(model_df)
)

print(
    "Final modelling columns:",
    model_df.shape[1]
)

print(
    "Output file:",
    output_file
)