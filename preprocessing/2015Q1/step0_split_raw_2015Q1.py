# ============================================================
# STEP 0 - SPLIT RAW 2015Q1 FILE INTO STATIC + TEMPORAL CSVs
# ------------------------------------------------------------
# Input : Data/2015Q1.csv  (raw Fannie Mae file, pipe-delimited,
#                           no header, one row per loan per month)
# Output: Data/2015Q1_CLEANED_LoanLevel(static data).csv
#         Data/2015Q1_CLEANED_Temporal.csv
#
# Same column names as your 2023Q1 files, so the rest of your
# pipeline (step2, step3, J-scripts) works unchanged.
#
# Also keeps the extra categorical columns (loan purpose,
# occupancy, property type, state, channel, first-time buyer)
# for the later "B" experiment.
#
# PRE-COVID CUTOFF: monthly records after Feb 2020 are dropped,
# so no labels or histories are affected by COVID forbearance.
#
# Run from the project folder (.Loan Prediction):
#     python step0_split_raw_2015Q1.py
# ============================================================

import os
import time
import pandas as pd

QUARTER = "2015Q1"
RAW_FILE = f"Data/{QUARTER}.csv"
STATIC_OUT = f"Data/{QUARTER}_CLEANED_LoanLevel(static data).csv"
TEMPORAL_OUT = f"Data/{QUARTER}_CLEANED_Temporal.csv"

COVID_CUTOFF = "2020-02-01"   # keep monthly records up to and including Feb 2020
CHUNK_SIZE = 1_000_000

# ------------------------------------------------------------
# Column positions in the raw Fannie Mae file (0-based)
# ------------------------------------------------------------
COLUMNS = {
    1: "loan_identifier",
    2: "monthly_reporting_period",
    3: "channel",
    4: "seller_name",
    5: "servicer_name",
    7: "original_interest_rate",
    8: "current_interest_rate",
    9: "original_upb",
    11: "current_actual_upb",
    12: "original_loan_term",
    13: "origination_date",
    14: "first_payment_date",
    15: "loan_age",
    16: "remaining_months_to_legal_maturity",
    17: "remaining_months_to_maturity",
    18: "maturity_date",
    19: "original_ltv",
    20: "original_cltv",
    21: "number_of_borrowers",
    22: "dti",
    23: "borrower_credit_score_at_origination",
    24: "co_borrower_credit_score_at_origination",
    25: "first_time_home_buyer",
    26: "loan_purpose",
    27: "property_type",
    28: "number_of_units",
    29: "occupancy",
    30: "property_state",
    31: "msa",
    32: "zip_code_short",
    33: "mortgage_insurance_percentage",
    34: "amortization_type",
    39: "current_loan_delinquency_status",
    41: "modification_flag",
    43: "zero_balance_code",
    44: "zero_balance_effective_date",
    48: "total_principal_current",
}

STATIC_COLS = [
    "loan_identifier", "channel", "seller_name", "original_interest_rate",
    "original_upb", "original_loan_term", "origination_date", "first_payment_date",
    "first_reporting_period", "maturity_date", "original_ltv", "original_cltv",
    "number_of_borrowers", "dti", "borrower_credit_score_at_origination",
    "co_borrower_credit_score_at_origination", "first_time_home_buyer",
    "loan_purpose", "property_type", "number_of_units", "occupancy",
    "property_state", "msa", "zip_code_short", "mortgage_insurance_percentage",
    "amortization_type",
]

TEMPORAL_COLS = [
    "loan_identifier", "monthly_reporting_period", "servicer_name",
    "current_interest_rate", "current_actual_upb", "loan_age",
    "remaining_months_to_legal_maturity", "remaining_months_to_maturity",
    "current_loan_delinquency_status", "modification_flag",
    "zero_balance_code", "zero_balance_effective_date", "total_principal_current",
]

NUMERIC_STATIC = [
    "original_interest_rate", "original_upb", "original_loan_term", "original_ltv",
    "original_cltv", "number_of_borrowers", "dti",
    "borrower_credit_score_at_origination", "co_borrower_credit_score_at_origination",
    "number_of_units", "mortgage_insurance_percentage",
]


def mmyyyy_to_date(series):
    """Raw dates are MMYYYY (e.g. 012015) -> 2015-01-01."""
    s = series.astype("string").str.strip().str.zfill(6)
    return pd.to_datetime(s, format="%m%Y", errors="coerce")


# ------------------------------------------------------------
# Main loop
# ------------------------------------------------------------
if not os.path.exists(RAW_FILE):
    raise FileNotFoundError(f"{RAW_FILE} not found - run this from the project folder")

for path in [STATIC_OUT, TEMPORAL_OUT]:
    if os.path.exists(path):
        os.remove(path)

seen_loans = set()
first_static, first_temporal = True, True
rows_in, rows_kept, rows_after_cutoff = 0, 0, 0
cutoff = pd.Timestamp(COVID_CUTOFF)
start = time.time()

reader = pd.read_csv(
    RAW_FILE, sep="|", header=None, usecols=list(COLUMNS.keys()),
    dtype=str, chunksize=CHUNK_SIZE, low_memory=False
)

for i, chunk in enumerate(reader, start=1):
    chunk = chunk.rename(columns=COLUMNS)
    rows_in += len(chunk)

    chunk["loan_identifier"] = chunk["loan_identifier"].str.strip().str.zfill(12)
    for col in ["monthly_reporting_period", "origination_date",
                "first_payment_date", "maturity_date", "zero_balance_effective_date"]:
        chunk[col] = mmyyyy_to_date(chunk[col])

    # ---------------- STATIC: first record of each new loan ----------------
    first_rows = chunk.drop_duplicates("loan_identifier", keep="first")
    first_rows = first_rows[~first_rows["loan_identifier"].isin(seen_loans)].copy()
    if len(first_rows):
        seen_loans.update(first_rows["loan_identifier"])
        first_rows["first_reporting_period"] = first_rows["monthly_reporting_period"]
        for col in NUMERIC_STATIC:
            first_rows[col] = pd.to_numeric(first_rows[col], errors="coerce")
        first_rows[STATIC_COLS].to_csv(
            STATIC_OUT, mode="w" if first_static else "a",
            header=first_static, index=False)
        first_static = False

    # ---------------- TEMPORAL: monthly rows up to Feb 2020 ----------------
    before = len(chunk)
    chunk = chunk[chunk["monthly_reporting_period"] <= cutoff]
    rows_after_cutoff += before - len(chunk)
    rows_kept += len(chunk)

    chunk[TEMPORAL_COLS].to_csv(
        TEMPORAL_OUT, mode="w" if first_temporal else "a",
        header=first_temporal, index=False)
    first_temporal = False

    print(f"Chunk {i:3d} | rows read {rows_in:>12,} | loans {len(seen_loans):>9,} "
          f"| {(time.time() - start) / 60:5.1f} min")

print("\n" + "=" * 60)
print("STEP 0 COMPLETE")
print("=" * 60)
print(f"Raw rows read               : {rows_in:,}")
print(f"Monthly rows kept (<= Feb 2020): {rows_kept:,}")
print(f"Rows dropped (after Feb 2020) : {rows_after_cutoff:,}")
print(f"Unique loans                : {len(seen_loans):,}")
print(f"Static file  : {STATIC_OUT}")
print(f"Temporal file: {TEMPORAL_OUT}")
print(f"Time taken   : {(time.time() - start) / 60:.1f} minutes")
