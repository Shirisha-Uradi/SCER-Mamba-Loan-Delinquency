import pandas as pd
import os

# ============================================================
# STEP 2 - TEMPORAL DATA CLEANING
# ============================================================

temporal_file = "Data/2023Q1_CLEANED_Temporal.csv"
output_temporal = "Data/cleaned/2023Q1_TEMPORAL_CLEANED_STEP2.csv"

os.makedirs("Data/cleaned", exist_ok=True)

print("========================================")
print("STARTING TEMPORAL DATA CLEANING")
print("========================================")

# Delete previous incomplete output if it exists
if os.path.exists(output_temporal):
    os.remove(output_temporal)

first_chunk = True
chunk_number = 0
total_clean_rows = 0

for chunk in pd.read_csv(
    temporal_file,
    chunksize=500000,
    low_memory=False,
    dtype={
        "loan_identifier": "string",
        "monthly_reporting_period": "string",
        "current_loan_delinquency_status": "string"
    }
):

    chunk_number += 1

    print(f"Cleaning chunk {chunk_number}...")

    # 1. Standardise loan ID
    chunk["loan_identifier"] = (
        chunk["loan_identifier"]
        .astype("string")
        .str.strip()
        .str.zfill(12)
    )

    # 2. Clean reporting month
    chunk["monthly_reporting_period"] = (
        chunk["monthly_reporting_period"]
        .astype("string")
        .str.strip()
    )

    # 3. Clean text columns
    text_columns = chunk.select_dtypes(
        include=["object", "string"]
    ).columns

    for column in text_columns:
        chunk[column] = (
            chunk[column]
            .astype("string")
            .str.strip()
        )

    # 4. Convert important columns to numeric
    numeric_columns = [
        "loan_age",
        "remaining_months_to_legal_maturity",
        "remaining_months_to_maturity",
        "current_interest_rate",
        "current_actual_upb",
        "total_principal_current"
    ]

    for column in numeric_columns:
        if column in chunk.columns:
            chunk[column] = pd.to_numeric(
                chunk[column],
                errors="coerce"
            )

    # 5. Numeric delinquency status
    chunk["delinquency_status_numeric"] = pd.to_numeric(
        chunk["current_loan_delinquency_status"],
        errors="coerce"
    )

    # 6. Delinquency indicators
    chunk["is_current"] = (
        chunk["delinquency_status_numeric"] == 0
    ).astype(int)

    chunk["is_30plus_delinquent"] = (
        chunk["delinquency_status_numeric"] >= 1
    ).astype(int)

    chunk["is_60plus_delinquent"] = (
        chunk["delinquency_status_numeric"] >= 2
    ).astype(int)

    chunk["is_90plus_delinquent"] = (
        chunk["delinquency_status_numeric"] >= 3
    ).astype(int)

    # 7. Standardise modification flag
    if "modification_flag" in chunk.columns:
        chunk["modification_flag"] = (
            chunk["modification_flag"]
            .astype("string")
            .str.strip()
            .str.upper()
        )

    # 8. Remove exact duplicate rows
    chunk = chunk.drop_duplicates()

    # 9. Sort
    chunk = chunk.sort_values(
        ["loan_identifier", "monthly_reporting_period"]
    )

    # 10. Save
    chunk.to_csv(
        output_temporal,
        mode="w" if first_chunk else "a",
        header=first_chunk,
        index=False
    )

    first_chunk = False
    total_clean_rows += len(chunk)


print("\n========================================")
print("TEMPORAL CLEANING COMPLETED")
print("========================================")

print("Total cleaned rows:", total_clean_rows)
print("Output file:", output_temporal)