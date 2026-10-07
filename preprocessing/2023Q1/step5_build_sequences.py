import pandas as pd
import numpy as np
import os
import pickle

# ============================================================
# STEP 5.1 - BUILD 12-MONTH SEQUENTIAL DATASET
# ============================================================

temporal_file = "Data/cleaned/2023Q1_TEMPORAL_CLEANED_STEP2.csv"
static_file = "Data/cleaned/2023Q1_STATIC_CLEANED_STEP2.csv"

# Same cohort used for baseline models
baseline_file = "Data/step4_baseline_age12_dataset.csv"

output_folder = "Data/sequential_raw"
os.makedirs(output_folder, exist_ok=True)

print("========================================")
print("STEP 5.1 - BUILD 12-MONTH SEQUENCES")
print("========================================")


# ============================================================
# 1. LOAD ELIGIBLE LOANS + TARGET
# ============================================================

baseline_df = pd.read_csv(
    baseline_file,
    low_memory=False,
    dtype={
        "loan_identifier": "string"
    }
)

baseline_df["loan_identifier"] = (
    baseline_df["loan_identifier"]
    .astype("string")
    .str.strip()
    .str.zfill(12)
)

baseline_df = baseline_df.drop_duplicates(
    subset=["loan_identifier"]
)

eligible_ids = set(
    baseline_df["loan_identifier"]
)

target_map = (
    baseline_df
    .set_index("loan_identifier")
    ["target_risk_6m"]
)

print(
    "\nEligible baseline loans:",
    len(eligible_ids)
)


# ============================================================
# 2. TEMPORAL FEATURES
# ============================================================

temporal_features = [
    "current_interest_rate",
    "current_actual_upb",
    "delinquency_status_numeric",
    "remaining_months_to_maturity",
    "is_30plus_delinquent",
    "is_60plus_delinquent",
    "is_90plus_delinquent"
]

temporal_columns = [
    "loan_identifier",
    "monthly_reporting_period",
    "loan_age"
] + temporal_features


# ============================================================
# 3. LOAD MONTHS 1-12 FOR ELIGIBLE LOANS
# ============================================================

temporal_parts = []

for chunk_number, chunk in enumerate(
    pd.read_csv(
        temporal_file,
        usecols=temporal_columns,
        chunksize=500000,
        low_memory=False,
        dtype={
            "loan_identifier": "string"
        }
    ),
    start=1
):

    print(
        f"Processing temporal chunk {chunk_number}..."
    )

    chunk["loan_identifier"] = (
        chunk["loan_identifier"]
        .astype("string")
        .str.strip()
        .str.zfill(12)
    )

    chunk["loan_age"] = pd.to_numeric(
        chunk["loan_age"],
        errors="coerce"
    )

    # Keep only baseline cohort
    chunk = chunk[
        chunk["loan_identifier"].isin(
            eligible_ids
        )
    ]

    # Keep first 12 months only
    chunk = chunk[
        (chunk["loan_age"] >= 1) &
        (chunk["loan_age"] <= 12)
    ].copy()

    if len(chunk) > 0:
        temporal_parts.append(chunk)


if not temporal_parts:
    raise ValueError(
        "No temporal observations found for eligible loans."
    )


temporal_df = pd.concat(
    temporal_parts,
    ignore_index=True
)

del temporal_parts


print(
    "\nTemporal rows from loan age 1-12:",
    len(temporal_df)
)


# ============================================================
# 4. CONVERT TEMPORAL FEATURES TO NUMERIC
# ============================================================

for column in temporal_features:

    temporal_df[column] = pd.to_numeric(
        temporal_df[column],
        errors="coerce"
    )


# ============================================================
# 5. CHECK DUPLICATE LOAN-AGE RECORDS
# ============================================================

duplicate_loan_months = temporal_df.duplicated(
    subset=[
        "loan_identifier",
        "loan_age"
    ]
).sum()

print(
    "Duplicate loan-age observations:",
    duplicate_loan_months
)

if duplicate_loan_months > 0:

    temporal_df = temporal_df.drop_duplicates(
        subset=[
            "loan_identifier",
            "loan_age"
        ],
        keep="last"
    )


# ============================================================
# 6. FIND COMPLETE 12-MONTH HISTORIES
# ============================================================

month_counts = (
    temporal_df
    .groupby("loan_identifier")
    ["loan_age"]
    .nunique()
)

complete_ids = month_counts[
    month_counts == 12
].index

print(
    "Loans with complete 12-month history:",
    len(complete_ids)
)


# ============================================================
# 7. KEEP COMPLETE SEQUENCES ONLY
# ============================================================

temporal_df = temporal_df[
    temporal_df["loan_identifier"]
    .isin(complete_ids)
].copy()

temporal_df = temporal_df.sort_values(
    [
        "loan_identifier",
        "loan_age"
    ]
)


# ============================================================
# 8. BUILD SEQUENCE ARRAY
#
# Shape:
# number_of_loans x 12 months x temporal_features
# ============================================================

sequence_list = []
loan_ids = []

for loan_id, group in temporal_df.groupby(
    "loan_identifier",
    sort=True
):

    group = group.sort_values(
        "loan_age"
    )

    if group["loan_age"].nunique() != 12:
        continue

    sequence = group[
        temporal_features
    ].to_numpy(
        dtype=np.float32
    )

    if sequence.shape != (
        12,
        len(temporal_features)
    ):
        continue

    sequence_list.append(
        sequence
    )

    loan_ids.append(
        loan_id
    )


if not sequence_list:
    raise ValueError(
        "No complete sequences could be constructed."
    )


X_sequence_raw = np.stack(
    sequence_list
)

loan_ids = np.array(
    loan_ids
)


print("\nRaw sequence shape:")
print(X_sequence_raw.shape)


# ============================================================
# 9. CREATE TARGET ARRAY
# ============================================================

y = np.array(
    [
        int(target_map.loc[loan_id])
        for loan_id in loan_ids
    ],
    dtype=np.int64
)


print("\nTarget shape:")
print(y.shape)


unique_values, counts = np.unique(
    y,
    return_counts=True
)

print("\nTarget distribution:")

for value, count in zip(
    unique_values,
    counts
):

    percentage = (
        count /
        len(y)
    ) * 100

    print(
        f"{value}: {count} "
        f"({percentage:.4f}%)"
    )


# ============================================================
# 10. LOAD STATIC FEATURES
# ============================================================

static_features = [
    "borrower_credit_score_at_origination",
    "co_borrower_credit_score_at_origination",
    "dti",
    "original_ltv",
    "original_cltv",
    "original_upb",
    "original_interest_rate",
    "number_of_borrowers",
    "mortgage_insurance_percentage",
    "has_co_borrower",
    "co_borrower_score_missing",
    "mortgage_insurance_missing"
]


static_df = pd.read_csv(
    static_file,
    usecols=[
        "loan_identifier"
    ] + static_features,
    low_memory=False,
    dtype={
        "loan_identifier": "string"
    }
)


static_df["loan_identifier"] = (
    static_df["loan_identifier"]
    .astype("string")
    .str.strip()
    .str.zfill(12)
)


static_df = static_df.drop_duplicates(
    subset=["loan_identifier"]
)

static_df = static_df.set_index(
    "loan_identifier"
)


# ============================================================
# 11. BUILD RAW STATIC ARRAY
#
# Important:
# Do NOT impute or normalise here.
# That will happen after train/validation/test split.
# ============================================================

static_rows = []

missing_static_ids = 0

for loan_id in loan_ids:

    if loan_id not in static_df.index:

        missing_static_ids += 1

        static_rows.append(
            np.full(
                len(static_features),
                np.nan,
                dtype=np.float32
            )
        )

        continue


    row = static_df.loc[
        loan_id,
        static_features
    ]

    row = pd.to_numeric(
        row,
        errors="coerce"
    )

    static_rows.append(
        row.to_numpy(
            dtype=np.float32
        )
    )


X_static_raw = np.stack(
    static_rows
)


print("\nRaw static shape:")
print(X_static_raw.shape)

print(
    "Loans missing complete static row:",
    missing_static_ids
)


# ============================================================
# 12. MISSING VALUE COUNTS
# ============================================================

sequence_missing = int(
    np.isnan(
        X_sequence_raw
    ).sum()
)

static_missing = int(
    np.isnan(
        X_static_raw
    ).sum()
)

print("\nRaw missing values:")

print(
    "Temporal sequence missing:",
    sequence_missing
)

print(
    "Static feature missing:",
    static_missing
)


# ============================================================
# 13. FINAL ALIGNMENT CHECKS
# ============================================================

assert len(
    X_sequence_raw
) == len(
    X_static_raw
) == len(
    y
) == len(
    loan_ids
)

print("\nAlignment check: PASSED")

print(
    "Final number of sequential samples:",
    len(y)
)


# ============================================================
# 14. SAVE RAW DATA
# ============================================================

np.save(
    f"{output_folder}/X_sequence_raw.npy",
    X_sequence_raw
)

np.save(
    f"{output_folder}/X_static_raw.npy",
    X_static_raw
)

np.save(
    f"{output_folder}/y.npy",
    y
)

np.save(
    f"{output_folder}/loan_ids.npy",
    loan_ids
)


with open(
    f"{output_folder}/temporal_features.pkl",
    "wb"
) as file:

    pickle.dump(
        temporal_features,
        file
    )


with open(
    f"{output_folder}/static_features.pkl",
    "wb"
) as file:

    pickle.dump(
        static_features,
        file
    )


# ============================================================
# 15. COMPLETE
# ============================================================

print("\n========================================")
print("STEP 5.1 COMPLETED")
print("========================================")

print(
    "Sequential samples:",
    len(y)
)

print(
    "Sequence shape:",
    X_sequence_raw.shape
)

print(
    "Static shape:",
    X_static_raw.shape
)

print(
    "Target shape:",
    y.shape
)

print(
    "Files saved in:",
    output_folder
)