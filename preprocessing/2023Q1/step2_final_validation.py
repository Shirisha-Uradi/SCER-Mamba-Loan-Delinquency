import pandas as pd

static_file = "Data/cleaned/2023Q1_STATIC_CLEANED_STEP2.csv"
temporal_file = "Data/cleaned/2023Q1_TEMPORAL_CLEANED_STEP2.csv"

print("========================================")
print("STEP 2 FINAL VALIDATION")
print("========================================")

# -----------------------------
# STATIC DATA VALIDATION
# -----------------------------

static_df = pd.read_csv(
    static_file,
    low_memory=False
)

print("\nSTATIC DATA")
print("Rows:", len(static_df))
print("Columns:", static_df.shape[1])

print(
    "Duplicate rows:",
    static_df.duplicated().sum()
)

print(
    "Duplicate loan IDs:",
    static_df["loan_identifier"].duplicated().sum()
)

# -----------------------------
# TEMPORAL DATA VALIDATION
# -----------------------------

temporal_rows = 0
duplicate_rows = 0
missing_loan_ids = 0
missing_months = 0
missing_delinquency = 0

for chunk in pd.read_csv(
    temporal_file,
    chunksize=500000,
    low_memory=False,
    dtype={
        "loan_identifier": "string",
        "monthly_reporting_period": "string"
    }
):

    temporal_rows += len(chunk)

    duplicate_rows += (
        chunk.duplicated().sum()
    )

    missing_loan_ids += (
        chunk["loan_identifier"]
        .isna()
        .sum()
    )

    missing_months += (
        chunk["monthly_reporting_period"]
        .isna()
        .sum()
    )

    missing_delinquency += (
        chunk["delinquency_status_numeric"]
        .isna()
        .sum()
    )


print("\nTEMPORAL DATA")

print("Rows:", temporal_rows)

print(
    "Duplicate rows:",
    duplicate_rows
)

print(
    "Missing loan IDs:",
    missing_loan_ids
)

print(
    "Missing reporting months:",
    missing_months
)

print(
    "Missing numeric delinquency status:",
    missing_delinquency
)

print("\n========================================")
print("VALIDATION COMPLETED")
print("========================================")