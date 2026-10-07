import pandas as pd
import os

# ============================================================
# STEP 3 - CREATE 6-MONTH LOAN RISK TARGET
# ============================================================

input_file = "Data/cleaned/2023Q1_TEMPORAL_CLEANED_STEP2.csv"
output_file = "Data/2023Q1_TARGET_6M_Risk_STEP3.csv"

print("========================================")
print("STEP 3 - TARGET METRIC CREATION")
print("========================================")

# ------------------------------------------------------------
# 1. Load only required columns
# ------------------------------------------------------------

use_columns = [
    "loan_identifier",
    "monthly_reporting_period",
    "delinquency_status_numeric"
]

chunks = []

for chunk_number, chunk in enumerate(
    pd.read_csv(
        input_file,
        usecols=use_columns,
        chunksize=500000,
        low_memory=False
    ),
    start=1
):

    print(f"Loading chunk {chunk_number}...")

    chunks.append(chunk)

df = pd.concat(
    chunks,
    ignore_index=True
)

del chunks

print("\nRows loaded:", len(df))


# ------------------------------------------------------------
# 2. Convert reporting month to date
# ------------------------------------------------------------

df["monthly_reporting_period"] = pd.to_datetime(
    df["monthly_reporting_period"],
    errors="coerce"
)

df["delinquency_status_numeric"] = pd.to_numeric(
    df["delinquency_status_numeric"],
    errors="coerce"
)


# ------------------------------------------------------------
# 3. Sort loan history correctly
# ------------------------------------------------------------

df = df.sort_values(
    [
        "loan_identifier",
        "monthly_reporting_period"
    ]
).reset_index(drop=True)

print("Loan histories sorted.")


# ------------------------------------------------------------
# 4. Identify months where loan is 60+ delinquent
# ------------------------------------------------------------

df["risk_event_month"] = (
    df["monthly_reporting_period"]
    .where(
        df["delinquency_status_numeric"] >= 2
    )
)


# ------------------------------------------------------------
# 5. Find next future 60+ delinquency event
# ------------------------------------------------------------

shifted_risk = (
    df.groupby(
        "loan_identifier",
        sort=False
    )["risk_event_month"]
    .shift(-1)
)

df["next_risk_month"] = (
    shifted_risk
    .groupby(
        df["loan_identifier"],
        sort=False
    )
    .bfill()
)


# ------------------------------------------------------------
# 6. Find final available month for every loan
# ------------------------------------------------------------

df["last_available_month"] = (
    df.groupby(
        "loan_identifier",
        sort=False
    )["monthly_reporting_period"]
    .transform("max")
)


# ------------------------------------------------------------
# 7. Define end of 6-month prediction window
# ------------------------------------------------------------

df["prediction_window_end"] = (
    df["monthly_reporting_period"]
    + pd.DateOffset(months=6)
)


# ------------------------------------------------------------
# 8. Keep observations with full 6-month follow-up
# ------------------------------------------------------------

eligible = (
    df["last_available_month"]
    >= df["prediction_window_end"]
)


# ------------------------------------------------------------
# 9. Do not use loans already 60+ delinquent
# ------------------------------------------------------------

not_already_risky = (
    df["delinquency_status_numeric"] < 2
)


# ------------------------------------------------------------
# 10. Create target
# ------------------------------------------------------------

df["target_risk_6m"] = (
    (
        df["next_risk_month"].notna()
    )
    &
    (
        df["next_risk_month"]
        <= df["prediction_window_end"]
    )
).astype(int)


# ------------------------------------------------------------
# 11. Keep valid prediction observations
# ------------------------------------------------------------

target_df = df[
    eligible &
    not_already_risky
].copy()


# ------------------------------------------------------------
# 12. Final target file
# ------------------------------------------------------------

target_df = target_df[
    [
        "loan_identifier",
        "monthly_reporting_period",
        "delinquency_status_numeric",
        "target_risk_6m"
    ]
]

target_df = target_df.rename(
    columns={
        "monthly_reporting_period":
        "observation_month"
    }
)


# ------------------------------------------------------------
# 13. Target distribution
# ------------------------------------------------------------

print("\n========================================")
print("TARGET DISTRIBUTION")
print("========================================")

counts = (
    target_df["target_risk_6m"]
    .value_counts()
    .sort_index()
)

print(counts)

print("\nPercentages:")

print(
    target_df["target_risk_6m"]
    .value_counts(normalize=True)
    .sort_index()
    * 100
)


# ------------------------------------------------------------
# 14. Save target
# ------------------------------------------------------------

target_df.to_csv(
    output_file,
    index=False
)

print("\n========================================")
print("STEP 3 TARGET CREATION COMPLETED")
print("========================================")

print(
    "Target observations:",
    len(target_df)
)

print(
    "Risky observations:",
    int(target_df["target_risk_6m"].sum())
)

print(
    "Non-risky observations:",
    int(
        (target_df["target_risk_6m"] == 0).sum()
    )
)

print(
    "Output file:",
    output_file
)