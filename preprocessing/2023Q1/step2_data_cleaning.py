import pandas as pd
import os

# ============================================================
# STEP 2 - DATA CLEANING AND NORMALISATION
# ============================================================

os.makedirs("Data/cleaned", exist_ok=True)
os.makedirs("output/step2", exist_ok=True)

static_file = "Data/2023Q1_CLEANED_LoanLevel(static data).csv"
temporal_file = "Data/2023Q1_CLEANED_Temporal.csv"

print("Starting Step 2: Data Cleaning and Normalisation")

# ============================================================
# 1. LOAD STATIC DATA
# ============================================================

static_df = pd.read_csv(
    static_file,
    low_memory=False
)

print("\nSTATIC DATA")
print("Rows:", static_df.shape[0])
print("Columns:", static_df.shape[1])

# ============================================================
# 2. CHECK DUPLICATES
# ============================================================

duplicate_rows = static_df.duplicated().sum()

print("\nDuplicate rows:", duplicate_rows)

duplicate_loans = static_df["loan_identifier"].duplicated().sum()

print("Duplicate loan identifiers:", duplicate_loans)

# ============================================================
# 3. CHECK MISSING VALUES
# ============================================================

missing = static_df.isnull().sum()

missing_report = pd.DataFrame({
    "column": missing.index,
    "missing_count": missing.values
})

missing_report["missing_percentage"] = (
    missing_report["missing_count"] /
    len(static_df)
) * 100

missing_report = missing_report.sort_values(
    "missing_percentage",
    ascending=False
)

print("\nTop columns with missing values:")
print(missing_report.head(20))

missing_report.to_csv(
    "output/step2/static_missing_values.csv",
    index=False
)

print("\nMissing-value report saved.")
# ============================================================
# 4. INVESTIGATE MISSING AND INVALID VALUES
# ============================================================

print("\n========================================")
print("CHECKING DATA QUALITY")
print("========================================")

# ------------------------------------------------------------
# Check data types
# ------------------------------------------------------------

print("\nColumn Data Types:")
print(static_df.dtypes)


# ------------------------------------------------------------
# Co-borrower credit score investigation
# ------------------------------------------------------------

print("\nCo-borrower missing values by number of borrowers:")

co_missing = static_df[
    static_df["co_borrower_credit_score_at_origination"].isna()
]

print(
    co_missing["number_of_borrowers"]
    .value_counts(dropna=False)
    .sort_index()
)


# ------------------------------------------------------------
# Mortgage insurance missing values
# ------------------------------------------------------------

print("\nMortgage Insurance missing values by LTV range:")

mi_missing = static_df[
    static_df["mortgage_insurance_percentage"].isna()
].copy()

mi_missing["ltv_range"] = pd.cut(
    mi_missing["original_ltv"],
    bins=[0, 80, 90, 100, float("inf")],
    labels=[
        "0-80",
        "81-90",
        "91-100",
        "100+"
    ],
    include_lowest=True
)

print(mi_missing["ltv_range"].value_counts().sort_index())


# ------------------------------------------------------------
# Check important numerical ranges
# ------------------------------------------------------------

numeric_columns = [
    "borrower_credit_score_at_origination",
    "co_borrower_credit_score_at_origination",
    "dti",
    "original_ltv",
    "original_cltv",
    "original_upb",
    "original_interest_rate",
    "number_of_borrowers",
    "mortgage_insurance_percentage"
]

print("\nNumerical Value Ranges:")

for column in numeric_columns:

    values = pd.to_numeric(
        static_df[column],
        errors="coerce"
    )

    print(
        f"{column}: "
        f"min={values.min()}, "
        f"max={values.max()}"
    )


# ------------------------------------------------------------
# Check categorical values
# ------------------------------------------------------------

categorical_columns = [
    "channel",
    "first_time_home_buyer",
    "loan_purpose",
    "property_type",
    "occupancy"
]

print("\nCategorical Values:")

for column in categorical_columns:

    if column in static_df.columns:

        print(f"\n{column}:")
        print(
            static_df[column]
            .value_counts(dropna=False)
        )


print("\nData-quality investigation completed.")
# ============================================================
# 5. CLEAN STATIC DATA
# ============================================================

print("\n========================================")
print("CLEANING STATIC DATA")
print("========================================")

clean_static = static_df.copy()


# ------------------------------------------------------------
# 1. Standardise Loan Identifier
# ------------------------------------------------------------

clean_static["loan_identifier"] = (
    clean_static["loan_identifier"]
    .astype("Int64")
    .astype("string")
    .str.zfill(12)
)

print("\nLoan identifiers standardised.")


# ------------------------------------------------------------
# 2. Remove unnecessary whitespace from text columns
# ------------------------------------------------------------

text_columns = clean_static.select_dtypes(
    include=["object", "string"]
).columns

for column in text_columns:
    clean_static[column] = (
        clean_static[column]
        .astype("string")
        .str.strip()
    )

print("Text columns standardised.")


# ------------------------------------------------------------
# 3. Convert date columns
# ------------------------------------------------------------

date_columns = [
    "first_reporting_period",
    "origination_date",
    "first_payment_date"
]

for column in date_columns:
    clean_static[column] = pd.to_datetime(
        clean_static[column],
        errors="coerce"
    )

print("Date columns converted.")


# ------------------------------------------------------------
# 4. Create missing-value indicators
# ------------------------------------------------------------

clean_static["has_co_borrower"] = (
    clean_static["number_of_borrowers"] > 1
).astype(int)

clean_static["co_borrower_score_missing"] = (
    clean_static[
        "co_borrower_credit_score_at_origination"
    ].isna()
).astype(int)

clean_static["mortgage_insurance_missing"] = (
    clean_static[
        "mortgage_insurance_percentage"
    ].isna()
).astype(int)

print("Missing-value indicators created.")


# ------------------------------------------------------------
# 5. Impute small amount of missing borrower credit score
# ------------------------------------------------------------

credit_median = clean_static[
    "borrower_credit_score_at_origination"
].median()

clean_static[
    "borrower_credit_score_at_origination"
] = clean_static[
    "borrower_credit_score_at_origination"
].fillna(credit_median)

print(
    "Missing borrower credit scores filled with median:",
    credit_median
)


# ------------------------------------------------------------
# 6. Impute missing DTI
# ------------------------------------------------------------

dti_median = clean_static["dti"].median()

clean_static["dti"] = (
    clean_static["dti"]
    .fillna(dti_median)
)

print(
    "Missing DTI values filled with median:",
    dti_median
)


# ------------------------------------------------------------
# 7. Seller name missing values
# ------------------------------------------------------------

clean_static["seller_name"] = (
    clean_static["seller_name"]
    .fillna("Unknown")
)

print("Missing seller names replaced with 'Unknown'.")


# ------------------------------------------------------------
# IMPORTANT:
# Do NOT fill co-borrower credit score with zero.
#
# Most missing values mean there is no co-borrower.
# The has_co_borrower feature records this information.
# Model-specific imputation will be done later.
# ------------------------------------------------------------


# ------------------------------------------------------------
# 8. Final duplicate check
# ------------------------------------------------------------

clean_static = clean_static.drop_duplicates()

print(
    "\nDuplicates after cleaning:",
    clean_static.duplicated().sum()
)


# ------------------------------------------------------------
# 9. Final missing-value report
# ------------------------------------------------------------

final_missing = clean_static.isnull().sum()

final_missing = final_missing[
    final_missing > 0
].sort_values(ascending=False)

print("\nRemaining Missing Values:")
print(final_missing)


# ------------------------------------------------------------
# 10. Save cleaned static dataset
# ------------------------------------------------------------

clean_static.to_csv(
    "Data/cleaned/2023Q1_STATIC_CLEANED_STEP2.csv",
    index=False
)

print("\n========================================")
print("STATIC DATA CLEANING COMPLETED")
print("========================================")

print("Final Rows:", clean_static.shape[0])
print("Final Columns:", clean_static.shape[1])

print(
    "\nSaved to:",
    "Data/cleaned/2023Q1_STATIC_CLEANED_STEP2.csv"
)
# ============================================================
# 6. TEMPORAL DATA QUALITY CHECK
# ============================================================

print("\n========================================")
print("CHECKING TEMPORAL DATA")
print("========================================")

temporal_file = "Data/2023Q1_CLEANED_Temporal.csv"

temporal_rows = 0
duplicate_rows = 0

missing_totals = {}

min_values = {
    "loan_age": None,
    "current_interest_rate": None,
    "current_actual_upb": None
}

max_values = {
    "loan_age": None,
    "current_interest_rate": None,
    "current_actual_upb": None
}

delinquency_counts = {}

chunk_number = 0


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

    print(f"Processing temporal chunk {chunk_number}...")

    temporal_rows += len(chunk)

    # --------------------------------------------------------
    # Duplicate rows
    # --------------------------------------------------------

    duplicate_rows += chunk.duplicated().sum()


    # --------------------------------------------------------
    # Missing values
    # --------------------------------------------------------

    chunk_missing = chunk.isnull().sum()

    for column, count in chunk_missing.items():

        missing_totals[column] = (
            missing_totals.get(column, 0) + count
        )


    # --------------------------------------------------------
    # Convert important numerical fields
    # --------------------------------------------------------

    numeric_columns = [
        "loan_age",
        "current_interest_rate",
        "current_actual_upb"
    ]

    for column in numeric_columns:

        if column in chunk.columns:

            values = pd.to_numeric(
                chunk[column],
                errors="coerce"
            )

            current_min = values.min()
            current_max = values.max()

            if pd.notna(current_min):

                if min_values[column] is None:
                    min_values[column] = current_min

                else:
                    min_values[column] = min(
                        min_values[column],
                        current_min
                    )

            if pd.notna(current_max):

                if max_values[column] is None:
                    max_values[column] = current_max

                else:
                    max_values[column] = max(
                        max_values[column],
                        current_max
                    )


    # --------------------------------------------------------
    # Delinquency status distribution
    # --------------------------------------------------------

    if "current_loan_delinquency_status" in chunk.columns:

        counts = (
            chunk["current_loan_delinquency_status"]
            .value_counts(dropna=False)
        )

        for status, count in counts.items():

            delinquency_counts[status] = (
                delinquency_counts.get(status, 0)
                + count
            )


# ============================================================
# RESULTS
# ============================================================

print("\n========================================")
print("TEMPORAL DATA QUALITY RESULTS")
print("========================================")

print("\nTotal temporal rows:")
print(temporal_rows)

print("\nDuplicate rows:")
print(duplicate_rows)


print("\nMissing Values:")

missing_report_temporal = pd.Series(
    missing_totals
).sort_values(ascending=False)

print(
    missing_report_temporal[
        missing_report_temporal > 0
    ]
)


print("\nNumerical Ranges:")

for column in min_values:

    print(
        column,
        "min =",
        min_values[column],
        "max =",
        max_values[column]
    )


print("\nDelinquency Status Counts:")

delinquency_series = pd.Series(
    delinquency_counts
).sort_index()

print(delinquency_series)


# Save missing-value report

missing_report_temporal.to_csv(
    "output/step2/temporal_missing_values.csv",
    header=["missing_count"]
)

print(
    "\nTemporal data-quality report saved successfully."
)