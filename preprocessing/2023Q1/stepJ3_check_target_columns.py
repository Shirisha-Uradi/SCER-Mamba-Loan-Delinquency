import pandas as pd

path = r"Data/2023Q1_TARGET_6M_Risk_STEP3.csv"

df = pd.read_csv(
    path,
    nrows=5
)

print("TARGET FILE SHAPE SAMPLE:")
print(df.shape)

print("\nTARGET COLUMNS:")
for i, col in enumerate(df.columns):
    print(i, col)

print("\nFIRST 5 ROWS:")
print(df.head())