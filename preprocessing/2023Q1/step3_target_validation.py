import pandas as pd

target_file = "Data/2023Q1_TARGET_6M_Risk_STEP3.csv"

print("========================================")
print("STEP 3 TARGET VALIDATION")
print("========================================")

df = pd.read_csv(
    target_file,
    low_memory=False,
    dtype={
        "loan_identifier": "string"
    }
)

print("\nRows:", len(df))
print("Columns:", df.shape[1])

print(
    "\nDuplicate rows:",
    df.duplicated().sum()
)

print(
    "Missing Loan IDs:",
    df["loan_identifier"].isna().sum()
)

print(
    "Missing observation months:",
    df["observation_month"].isna().sum()
)

print(
    "Missing target values:",
    df["target_risk_6m"].isna().sum()
)

print("\nTarget values:")
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

print(
    "\nUnique Loans:",
    df["loan_identifier"].nunique()
)

# Check that target only contains 0 and 1
valid_targets = set(
    df["target_risk_6m"]
    .dropna()
    .unique()
)

print(
    "\nTarget values are only 0/1:",
    valid_targets.issubset({0, 1})
)

# Check observations already 60+ delinquent
already_risky = (
    df["delinquency_status_numeric"] >= 2
).sum()

print(
    "Observations already 60+ delinquent:",
    already_risky
)

print("\n========================================")
print("TARGET VALIDATION COMPLETED")
print("========================================")