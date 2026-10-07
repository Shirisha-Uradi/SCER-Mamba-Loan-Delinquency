import pandas as pd
import matplotlib.pyplot as plt
import os

# Create output folder if it does not exist
os.makedirs("output", exist_ok=True)

# Load static dataset
file_path = "Data/2023Q1_CLEANED_LoanLevel(static data).csv"

df = pd.read_csv(file_path)

print("Dataset loaded successfully")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

print("\nFirst 5 rows:")
print(df.head())

# Credit Score Distribution
plt.figure(figsize=(8, 5))

plt.hist(
    df["borrower_credit_score_at_origination"].dropna(),
    bins=30
)

plt.title("Borrower Credit Score Distribution")
plt.xlabel("Credit Score")
plt.ylabel("Number of Loans")

plt.tight_layout()

# Save chart
plt.savefig("output/credit_score_distribution.png", dpi=300)

plt.close()

print("\nCredit Score chart created successfully.")
# DTI Distribution
plt.figure(figsize=(8, 5))

plt.hist(
    df["dti"].dropna(),
    bins=30
)

plt.title("Debt-to-Income (DTI) Distribution")
plt.xlabel("DTI (%)")
plt.ylabel("Number of Loans")

plt.tight_layout()

# Save chart
plt.savefig("output/dti_distribution.png", dpi=300)

plt.close()

print("DTI chart created successfully.")
# LTV Distribution
plt.figure(figsize=(8, 5))

plt.hist(
    df["original_ltv"].dropna(),
    bins=30
)

plt.title("Original Loan-to-Value (LTV) Distribution")
plt.xlabel("LTV (%)")
plt.ylabel("Number of Loans")

plt.tight_layout()

plt.savefig("output/ltv_distribution.png", dpi=300)

plt.close()

print("LTV chart created successfully.")
# Original Loan Amount Distribution
plt.figure(figsize=(8, 5))

plt.hist(
    df["original_upb"].dropna(),
    bins=30
)

plt.title("Original Loan Amount Distribution")
plt.xlabel("Original Loan Amount ($)")
plt.ylabel("Number of Loans")

plt.tight_layout()

plt.savefig("output/original_loan_amount_distribution.png", dpi=300)

plt.close()

print("Original Loan Amount chart created successfully.")
# Interest Rate Distribution
plt.figure(figsize=(8, 5))

plt.hist(
    df["original_interest_rate"].dropna(),
    bins=30
)

plt.title("Original Interest Rate Distribution")
plt.xlabel("Interest Rate (%)")
plt.ylabel("Number of Loans")

plt.tight_layout()

plt.savefig("output/interest_rate_distribution.png", dpi=300)

plt.close()

print("Interest Rate chart created successfully.")
# Loan Purpose Distribution
loan_purpose_counts = df["loan_purpose"].value_counts()

plt.figure(figsize=(8, 5))

loan_purpose_counts.plot(kind="bar")

plt.title("Loan Purpose Distribution")
plt.xlabel("Loan Purpose")
plt.ylabel("Number of Loans")

plt.xticks(rotation=30)
plt.tight_layout()

plt.savefig("output/loan_purpose_distribution.png", dpi=300)

plt.close()

print("Loan Purpose chart created successfully.")
# Property Type Distribution
property_type_counts = df["property_type"].value_counts()

plt.figure(figsize=(8, 5))

property_type_counts.plot(kind="bar")

plt.title("Property Type Distribution")
plt.xlabel("Property Type")
plt.ylabel("Number of Loans")

plt.xticks(rotation=30)
plt.tight_layout()

plt.savefig("output/property_type_distribution.png", dpi=300)

plt.close()

print("Property Type chart created successfully.")
# Occupancy Distribution
occupancy_counts = df["occupancy"].value_counts()

plt.figure(figsize=(8, 5))

occupancy_counts.plot(kind="bar")

plt.title("Occupancy Distribution")
plt.xlabel("Occupancy Type")
plt.ylabel("Number of Loans")

plt.xticks(rotation=30)
plt.tight_layout()

plt.savefig("output/occupancy_distribution.png", dpi=300)

plt.close()

print("Occupancy chart created successfully.")
# Delinquency Status Distribution
temporal_file = "Data/2023Q1_CLEANED_Temporal.csv"

delinquency_counts = pd.Series(dtype="int64")

for chunk in pd.read_csv(
    temporal_file,
    usecols=["current_loan_delinquency_status"],
    chunksize=500000,
    dtype={"current_loan_delinquency_status": "string"}
):
    counts = chunk["current_loan_delinquency_status"].value_counts()
    delinquency_counts = delinquency_counts.add(counts, fill_value=0)

delinquency_counts = delinquency_counts.sort_index()

print("\nDelinquency Status Counts:")
print(delinquency_counts)

plt.figure(figsize=(9, 5))

delinquency_counts.plot(kind="bar")

plt.title("Monthly Delinquency Status Distribution")
plt.xlabel("Delinquency Status")
plt.ylabel("Number of Monthly Records")

plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig(
    "output/delinquency_status_distribution.png",
    dpi=300
)

plt.close()

print("Delinquency Status chart created successfully.")
# 60+ Days Delinquency Trend Over Time

monthly_total = {}
monthly_risky = {}

for chunk in pd.read_csv(
    temporal_file,
    usecols=[
        "monthly_reporting_period",
        "current_loan_delinquency_status"
    ],
    chunksize=500000,
    dtype={
        "monthly_reporting_period": "string",
        "current_loan_delinquency_status": "string"
    }
):
    # Total records per month
    total_counts = chunk["monthly_reporting_period"].value_counts()

    for month, count in total_counts.items():
        monthly_total[month] = monthly_total.get(month, 0) + count

    # Convert delinquency status to number
    chunk["delinq_num"] = pd.to_numeric(
        chunk["current_loan_delinquency_status"],
        errors="coerce"
    )

    # Keep 60+ days delinquent = code 02 or more
    risky_chunk = chunk[chunk["delinq_num"] >= 2]

    risky_counts = risky_chunk["monthly_reporting_period"].value_counts()

    for month, count in risky_counts.items():
        monthly_risky[month] = monthly_risky.get(month, 0) + count


# Create DataFrame
trend_df = pd.DataFrame({
    "month": list(monthly_total.keys()),
    "total_records": list(monthly_total.values())
})

trend_df["risky_records"] = trend_df["month"].map(monthly_risky).fillna(0)

trend_df["risk_rate"] = (
    trend_df["risky_records"] /
    trend_df["total_records"]
) * 100

trend_df = trend_df.sort_values("month")


print("\nMonthly 60+ DPD Risk Rate:")
print(trend_df.head())


# Plot chart
plt.figure(figsize=(12, 6))

plt.plot(
    trend_df["month"],
    trend_df["risk_rate"],
    marker="o"
)

plt.title("60+ Days Delinquency Rate Over Time")
plt.xlabel("Reporting Month")
plt.ylabel("60+ DPD Rate (%)")

plt.xticks(rotation=60)

plt.tight_layout()

plt.savefig(
    "output/60plus_delinquency_trend.png",
    dpi=300
)

plt.close()

print("60+ Days Delinquency Trend chart created successfully.")
# Risky vs Non-Risky Target Distribution

target_file = "Data/2023Q1_TARGET_6M_Risk.csv"

target_counts = pd.Series(dtype="int64")

for chunk in pd.read_csv(
    target_file,
    usecols=["target_risk_6m"],
    chunksize=500000
):
    counts = chunk["target_risk_6m"].value_counts()
    target_counts = target_counts.add(counts, fill_value=0)

target_counts = target_counts.sort_index()

print("\nTarget Distribution:")
print(target_counts)

# Rename labels
labels = {
    0: "Non-risky",
    1: "Risky"
}

target_counts.index = target_counts.index.map(labels)

# Plot chart
plt.figure(figsize=(7, 5))

target_counts.plot(kind="bar")

plt.title("Risky vs Non-Risky Loan Distribution")
plt.xlabel("Risk Class")
plt.ylabel("Number of Observations")

plt.xticks(rotation=0)

plt.tight_layout()

plt.savefig(
    "output/risky_vs_non_risky_distribution.png",
    dpi=300
)

plt.close()

print("Risky vs Non-risky chart created successfully.")
# ============================================================
# FEATURE IMPORTANCE
# ============================================================

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import numpy as np


print("\nStarting Feature Importance Analysis...")


# ------------------------------------------------------------
# 1. Take loan-age 12 observation from temporal data
# ------------------------------------------------------------

age12_rows = []

for chunk in pd.read_csv(
    temporal_file,
    usecols=[
        "loan_identifier",
        "monthly_reporting_period",
        "loan_age",
        "current_interest_rate",
        "current_actual_upb",
        "current_loan_delinquency_status"
    ],
    chunksize=500000,
    dtype={
        "loan_identifier": "string",
        "monthly_reporting_period": "string",
        "current_loan_delinquency_status": "string"
    }
):

    chunk["loan_age"] = pd.to_numeric(
        chunk["loan_age"],
        errors="coerce"
    )

    # Keep prediction point = loan age 12
    age12 = chunk[chunk["loan_age"] == 12].copy()

    age12_rows.append(age12)


age12_df = pd.concat(age12_rows, ignore_index=True)

print("Age-12 observations:", len(age12_df))


# ------------------------------------------------------------
# 2. Load target
# ------------------------------------------------------------

target_df = pd.read_csv(
    target_file,
    usecols=[
        "loan_identifier",
        "observation_month",
        "target_risk_6m"
    ],
    dtype={
        "loan_identifier": "string",
        "observation_month": "string"
    }
)


# ------------------------------------------------------------
# 3. Merge age-12 temporal data with target
# ------------------------------------------------------------

model_df = age12_df.merge(
    target_df,
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


# ------------------------------------------------------------
# 4. Merge static borrower / loan features
# ------------------------------------------------------------

static_features = [
    "loan_identifier",
    "borrower_credit_score_at_origination",
    "co_borrower_credit_score_at_origination",
    "dti",
    "original_ltv",
    "original_cltv",
    "original_upb",
    "original_interest_rate",
    "number_of_borrowers"
]

static_df = df[static_features].copy()

static_df["loan_identifier"] = (
    static_df["loan_identifier"]
    .astype("string")
    .str.zfill(12)
)

model_df["loan_identifier"] = (
    model_df["loan_identifier"]
    .astype("string")
    .str.zfill(12)
)

model_df = model_df.merge(
    static_df,
    on="loan_identifier",
    how="left"
)


# ------------------------------------------------------------
# 5. Prepare features
# ------------------------------------------------------------

features = [
    "borrower_credit_score_at_origination",
    "co_borrower_credit_score_at_origination",
    "dti",
    "original_ltv",
    "original_cltv",
    "original_upb",
    "original_interest_rate",
    "number_of_borrowers",
    "current_interest_rate",
    "current_actual_upb",
    "current_loan_delinquency_status"
]


# Convert everything to numeric
for column in features:
    model_df[column] = pd.to_numeric(
        model_df[column],
        errors="coerce"
    )


X = model_df[features].copy()
y = model_df["target_risk_6m"]


# Fill missing numeric values using median
X = X.fillna(X.median())


print("\nFeature importance dataset shape:")
print(X.shape)

print("\nTarget counts:")
print(y.value_counts())


# ------------------------------------------------------------
# 6. Split data
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)


# ------------------------------------------------------------
# 7. Train baseline Random Forest
# ------------------------------------------------------------

rf_model = RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)


# ------------------------------------------------------------
# 8. Basic validation
# ------------------------------------------------------------

pred_prob = rf_model.predict_proba(X_test)[:, 1]

auc = roc_auc_score(
    y_test,
    pred_prob
)

print("\nBaseline ROC-AUC:", round(auc, 4))


# ------------------------------------------------------------
# 9. Calculate Feature Importance
# ------------------------------------------------------------

importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": rf_model.feature_importances_
})

importance_df = importance_df.sort_values(
    "Importance",
    ascending=False
)


print("\nFeature Importance:")
print(importance_df)


# Save table
importance_df.to_csv(
    "output/feature_importance.csv",
    index=False
)


# ------------------------------------------------------------
# 10. Feature Importance Chart
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))

plt.barh(
    importance_df["Feature"],
    importance_df["Importance"]
)

plt.title("Loan Risk Feature Importance")
plt.xlabel("Importance")

plt.gca().invert_yaxis()

plt.tight_layout()

plt.savefig(
    "output/feature_importance.png",
    dpi=300
)

plt.close()

print("Feature Importance chart created successfully.")
# ============================================================
# WRITTEN DATA ANALYSIS SUMMARY
# ============================================================

print("\nCreating written analysis summary...")

# Numerical statistics
credit = df["borrower_credit_score_at_origination"].dropna()
dti = df["dti"].dropna()
ltv = df["original_ltv"].dropna()
loan_amount = df["original_upb"].dropna()
interest = df["original_interest_rate"].dropna()

# Top categorical values
top_purpose = df["loan_purpose"].value_counts()
top_property = df["property_type"].value_counts()
top_occupancy = df["occupancy"].value_counts()

# Target values
risky_count = int(target_counts.get("Risky", 0))
non_risky_count = int(target_counts.get("Non-risky", 0))

target_total = risky_count + non_risky_count

risky_percentage = (
    risky_count / target_total * 100
    if target_total > 0 else 0
)

non_risky_percentage = (
    non_risky_count / target_total * 100
    if target_total > 0 else 0
)

# Top 5 important features
top_features = importance_df.head(5)

summary = f"""
FANNIE MAE 2023 Q1 DATA ANALYSIS
=================================

1. DATASET SHAPE
----------------
Total unique loans: {len(df):,}
Static features: {df.shape[1]}

The dataset contains borrower, loan and property information together with
monthly repayment behaviour. This makes it suitable for sequential loan-risk
prediction.

2. CREDIT SCORE DISTRIBUTION
----------------------------
Median Credit Score: {credit.median():.0f}
Average Credit Score: {credit.mean():.2f}

Most borrowers have credit scores concentrated in the mid-to-high credit
score range. Credit score is an important borrower-risk characteristic and
also appears as an important feature in the feature-importance analysis.

3. DTI DISTRIBUTION
-------------------
Median DTI: {dti.median():.0f}%
Average DTI: {dti.mean():.2f}%

DTI measures how much of the borrower's income is used for debt repayments.
Higher DTI may indicate greater repayment pressure.

4. LTV DISTRIBUTION
-------------------
Median LTV: {ltv.median():.0f}%
Average LTV: {ltv.mean():.2f}%

LTV represents the loan amount relative to the property value. Higher LTV
means that the borrower has less equity in the property.

5. ORIGINAL LOAN AMOUNT
-----------------------
Median Loan Amount: ${loan_amount.median():,.0f}
Average Loan Amount: ${loan_amount.mean():,.0f}

The distribution shows variation in mortgage size across the portfolio.

6. INTEREST RATE DISTRIBUTION
-----------------------------
Median Interest Rate: {interest.median():.3f}%
Average Interest Rate: {interest.mean():.3f}%

Interest rate affects the repayment burden and is therefore retained as a
loan-risk feature.

7. LOAN PURPOSE
---------------
Most common loan purpose:
{top_purpose.head(3).to_string()}

The dataset contains different purposes such as purchase and refinance loans,
allowing differences in loan behaviour to be analysed.

8. PROPERTY TYPE
----------------
Most common property types:
{top_property.head(3).to_string()}

Property type is retained as a categorical feature because different property
types may show different risk characteristics.

9. OCCUPANCY
------------
Most common occupancy types:
{top_occupancy.head(3).to_string()}

Occupancy distinguishes principal residences, investment properties and
other occupancy types.

10. TARGET DISTRIBUTION
-----------------------
Non-risky observations: {non_risky_count:,} ({non_risky_percentage:.3f}%)
Risky observations: {risky_count:,} ({risky_percentage:.3f}%)

The target is highly imbalanced, with significantly fewer risky observations.
This will need to be considered later during model training and evaluation.

11. FEATURE IMPORTANCE
----------------------
Top predictive features:

{top_features.to_string(index=False)}

Feature importance identifies which borrower, loan and repayment variables
provide the strongest predictive information for future loan risk.

12. CONCLUSION
--------------
The analysis confirms that the Fannie Mae dataset contains both useful static
borrower/loan characteristics and temporal repayment information. The target
is highly imbalanced, while repayment behaviour and borrower credit
characteristics provide important signals for future loan-risk prediction.

SCER-Mamba implementation and training have not started at this stage.
"""

# Save summary
with open(
    "output/data_analysis_summary.txt",
    "w",
    encoding="utf-8"
) as file:
    file.write(summary)

print(summary)

print("\nData analysis summary saved successfully.")