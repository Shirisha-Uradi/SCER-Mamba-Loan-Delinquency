import pandas as pd

path = r"Data/cleaned/2023Q1_TEMPORAL_CLEANED_STEP2.csv"

df = pd.read_csv(path, nrows=5)

print("TEMPORAL COLUMNS:")
for i, col in enumerate(df.columns):
    print(i, col)