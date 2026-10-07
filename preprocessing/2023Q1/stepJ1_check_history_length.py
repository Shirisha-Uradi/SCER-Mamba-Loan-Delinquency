import pandas as pd

path = r"Data/cleaned/2023Q1_TEMPORAL_CLEANED_STEP2.csv"

usecols = [
    "loan_identifier",
    "monthly_reporting_period",
    "loan_age"
]

print("Loading temporal history...")

df = pd.read_csv(
    path,
    usecols=usecols
)

print("\nRows:", len(df))
print("Unique loans:", df["loan_identifier"].nunique())

print("\nLoan age range:")
print("Minimum:", df["loan_age"].min())
print("Maximum:", df["loan_age"].max())

counts = df.groupby(
    "loan_identifier"
).size()

print("\nMonthly records per loan:")
print(counts.describe())

print("\nLoans with at least 12 months:")
print((counts >= 12).sum())

print("\nLoans with at least 18 months:")
print((counts >= 18).sum())

print("\nLoans with at least 24 months:")
print((counts >= 24).sum())