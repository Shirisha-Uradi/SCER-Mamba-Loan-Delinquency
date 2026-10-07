import os
import pickle
import numpy as np
import pandas as pd

# ============================================================
# PATHS
# ============================================================

TEMPORAL_PATH = r"Data/cleaned/2023Q1_TEMPORAL_CLEANED_STEP2.csv"
STATIC_PATH = r"Data/cleaned/2023Q1_STATIC_CLEANED_STEP2.csv"
TARGET_PATH = r"Data/2023Q1_TARGET_6M_Risk_STEP3.csv"

OUTPUT_18 = r"Data/sequential_raw_age18"
OUTPUT_24 = r"Data/sequential_raw_age24"

os.makedirs(OUTPUT_18, exist_ok=True)
os.makedirs(OUTPUT_24, exist_ok=True)

# ============================================================
# FEATURES
# ============================================================

TEMPORAL_FEATURES = [
    "current_interest_rate",
    "current_actual_upb",
    "delinquency_status_numeric",
    "remaining_months_to_maturity",
    "is_30plus_delinquent",
    "is_60plus_delinquent",
    "is_90plus_delinquent",
]

STATIC_FEATURES = [
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
    "mortgage_insurance_missing",
]


def normalise_id(series):

    x = series.astype("string").str.strip()

    # Handles values such as 12345.0 if present
    x = x.str.replace(
        r"\.0$",
        "",
        regex=True
    )

    # Make IDs consistent when some files lost leading zeros
    x = x.str.lstrip("0")

    x = x.mask(
        x == "",
        "0"
    )

    return x


# ============================================================
# 1. LOAD STATIC DATA
# ============================================================

print("\n======================================")
print("LOADING STATIC DATA")
print("======================================")

static = pd.read_csv(
    STATIC_PATH,
    usecols=["loan_identifier"] + STATIC_FEATURES
)

static["loan_identifier"] = normalise_id(
    static["loan_identifier"]
)

static = static.drop_duplicates(
    subset=["loan_identifier"],
    keep="first"
)

print("Static loans:", len(static))


# ============================================================
# 2. LOAD ONLY TEMPORAL AGES 1-24
# ============================================================

print("\n======================================")
print("LOADING TEMPORAL HISTORY AGES 1-24")
print("======================================")

required_temporal_columns = [
    "loan_identifier",
    "monthly_reporting_period",
    "loan_age",
] + TEMPORAL_FEATURES

temporal_parts = []

for chunk_number, chunk in enumerate(
    pd.read_csv(
        TEMPORAL_PATH,
        usecols=required_temporal_columns,
        chunksize=500_000
    ),
    start=1
):

    chunk["loan_age"] = pd.to_numeric(
        chunk["loan_age"],
        errors="coerce"
    )

    # Keep only ages required for J18/J24
    chunk = chunk[
        (chunk["loan_age"] >= 1)
        &
        (chunk["loan_age"] <= 24)
        &
        (chunk["loan_age"] % 1 == 0)
    ].copy()

    if len(chunk) == 0:
        continue

    chunk["loan_age"] = chunk[
        "loan_age"
    ].astype(np.int16)

    chunk["loan_identifier"] = normalise_id(
        chunk["loan_identifier"]
    )

    chunk["monthly_reporting_period"] = pd.to_datetime(
        chunk["monthly_reporting_period"],
        errors="coerce"
    )

    for col in TEMPORAL_FEATURES:

        chunk[col] = pd.to_numeric(
            chunk[col],
            errors="coerce"
        ).astype("float32")

    temporal_parts.append(chunk)

    print(
        f"Processed temporal chunk {chunk_number}"
    )


temporal = pd.concat(
    temporal_parts,
    ignore_index=True
)

del temporal_parts

print("\nRows ages 1-24:", len(temporal))
print(
    "Unique loans:",
    temporal["loan_identifier"].nunique()
)

# Remove duplicate loan-age records if any
duplicates = temporal.duplicated(
    subset=[
        "loan_identifier",
        "loan_age"
    ]
).sum()

print("Duplicate loan-age rows:", duplicates)

temporal = temporal.sort_values(
    [
        "loan_identifier",
        "loan_age",
        "monthly_reporting_period"
    ]
)

temporal = temporal.drop_duplicates(
    subset=[
        "loan_identifier",
        "loan_age"
    ],
    keep="last"
)


# ============================================================
# 3. FIND OBSERVATION ROWS AT AGE 18 AND 24
# ============================================================

observation_frames = []

for age in [18, 24]:

    obs = temporal[
        temporal["loan_age"] == age
    ][
        [
            "loan_identifier",
            "monthly_reporting_period",
            "is_60plus_delinquent"
        ]
    ].copy()

    # Prediction is only for loans not already 60+ DPD
    obs["is_60plus_delinquent"] = pd.to_numeric(
        obs["is_60plus_delinquent"],
        errors="coerce"
    )

    obs = obs[
        obs["is_60plus_delinquent"].eq(0)
    ].copy()

    obs["observation_age"] = age

    observation_frames.append(obs)

    print(
        f"\nAge {age} observation candidates:",
        len(obs)
    )


observation_keys = pd.concat(
    observation_frames,
    ignore_index=True
)

del observation_frames


# ============================================================
# 4. LOAD TARGET LABELS
#    Only labels matching age-18/24 observation months are kept
# ============================================================

print("\n======================================")
print("MATCHING 6-MONTH TARGET LABELS")
print("======================================")

target_matches = []

target_usecols = [
    "loan_identifier",
    "observation_month",
    "target_risk_6m"
]

for chunk_number, target_chunk in enumerate(
    pd.read_csv(
        TARGET_PATH,
        usecols=target_usecols,
        chunksize=500_000
    ),
    start=1
):

    target_chunk["loan_identifier"] = normalise_id(
        target_chunk["loan_identifier"]
    )

    target_chunk["observation_month"] = pd.to_datetime(
        target_chunk["observation_month"],
        errors="coerce"
    )

    target_chunk = target_chunk.rename(
        columns={
            "observation_month":
            "monthly_reporting_period"
        }
    )

    matched = target_chunk.merge(
        observation_keys[
            [
                "loan_identifier",
                "monthly_reporting_period",
                "observation_age"
            ]
        ],
        on=[
            "loan_identifier",
            "monthly_reporting_period"
        ],
        how="inner"
    )

    if len(matched) > 0:
        target_matches.append(matched)

    print(
        f"Processed target chunk {chunk_number}"
    )


labels = pd.concat(
    target_matches,
    ignore_index=True
)

del target_matches

labels["target_risk_6m"] = pd.to_numeric(
    labels["target_risk_6m"],
    errors="coerce"
)

labels = labels.dropna(
    subset=["target_risk_6m"]
)

labels["target_risk_6m"] = labels[
    "target_risk_6m"
].astype(np.int8)

labels = labels.drop_duplicates(
    subset=[
        "loan_identifier",
        "observation_age"
    ],
    keep="last"
)

print("\nMatched labels:", len(labels))


# ============================================================
# 5. BUILD EACH SEQUENCE DATASET
# ============================================================

def build_sequence_dataset(
    observation_age,
    output_folder
):

    print("\n")
    print("=" * 60)
    print(
        f"BUILDING AGE-{observation_age} DATASET"
    )
    print("=" * 60)

    # Labels for this observation age
    samples = labels[
        labels["observation_age"]
        == observation_age
    ].copy()

    print(
        "Target-eligible loans:",
        len(samples)
    )

    # History is age 1 through observation age only.
    # No months after the prediction point enter the model.
    history = temporal[
        (temporal["loan_age"] >= 1)
        &
        (
            temporal["loan_age"]
            <= observation_age
        )
    ].copy()

    target_ids = set(
        samples["loan_identifier"]
    )

    history = history[
        history["loan_identifier"].isin(
            target_ids
        )
    ]

    # Require every age from 1...observation_age
    age_counts = history.groupby(
        "loan_identifier"
    )["loan_age"].nunique()

    valid_history_ids = set(
        age_counts[
            age_counts == observation_age
        ].index
    )

    print(
        "Loans with complete history:",
        len(valid_history_ids)
    )

    # Require static features
    static_ids = set(
        static["loan_identifier"]
    )

    final_ids = (
        valid_history_ids
        &
        static_ids
    )

    samples = samples[
        samples["loan_identifier"].isin(
            final_ids
        )
    ].copy()

    # One sample per loan
    samples = samples.drop_duplicates(
        subset=["loan_identifier"],
        keep="last"
    )

    # Sort history so reshape is safe
    history = history[
        history["loan_identifier"].isin(
            set(samples["loan_identifier"])
        )
    ].sort_values(
        [
            "loan_identifier",
            "loan_age"
        ]
    )

    # Final safety check
    final_counts = history.groupby(
        "loan_identifier"
    ).size()

    complete_ids = final_counts[
        final_counts == observation_age
    ].index

    history = history[
        history["loan_identifier"].isin(
            complete_ids
        )
    ]

    # Order of loans in history
    ordered_ids = history[
        "loan_identifier"
    ].drop_duplicates().to_numpy()

    samples = (
        samples
        .set_index("loan_identifier")
        .loc[ordered_ids]
        .reset_index()
    )

    static_indexed = static.set_index(
        "loan_identifier"
    )

    static_aligned = static_indexed.loc[
        ordered_ids,
        STATIC_FEATURES
    ]

    # ========================================================
    # CREATE ARRAYS
    # ========================================================

    X_sequence = history[
        TEMPORAL_FEATURES
    ].to_numpy(
        dtype=np.float32
    ).reshape(
        len(ordered_ids),
        observation_age,
        len(TEMPORAL_FEATURES)
    )

    X_static = static_aligned.to_numpy(
        dtype=np.float32
    )

    y = samples[
        "target_risk_6m"
    ].to_numpy(
        dtype=np.int8
    )

    loan_ids = np.asarray(
        ordered_ids
    )

    # ========================================================
    # CHECK ALIGNMENT
    # ========================================================

    assert len(X_sequence) == len(X_static)
    assert len(X_sequence) == len(y)
    assert len(X_sequence) == len(loan_ids)

    assert X_sequence.shape[1] == observation_age
    assert X_sequence.shape[2] == 7

    # ========================================================
    # SAVE
    # ========================================================

    np.save(
        os.path.join(
            output_folder,
            "loan_ids.npy"
        ),
        loan_ids
    )

    np.save(
        os.path.join(
            output_folder,
            "X_sequence_raw.npy"
        ),
        X_sequence
    )

    np.save(
        os.path.join(
            output_folder,
            "X_static_raw.npy"
        ),
        X_static
    )

    np.save(
        os.path.join(
            output_folder,
            "y.npy"
        ),
        y
    )

    with open(
        os.path.join(
            output_folder,
            "temporal_features.pkl"
        ),
        "wb"
    ) as f:

        pickle.dump(
            TEMPORAL_FEATURES,
            f
        )

    with open(
        os.path.join(
            output_folder,
            "static_features.pkl"
        ),
        "wb"
    ) as f:

        pickle.dump(
            STATIC_FEATURES,
            f
        )

    # Save observation information too
    samples[
        [
            "loan_identifier",
            "monthly_reporting_period",
            "target_risk_6m"
        ]
    ].to_csv(
        os.path.join(
            output_folder,
            "sample_metadata.csv"
        ),
        index=False
    )

    risky = int(y.sum())
    non_risky = int(
        len(y) - risky
    )

    print("\nFINAL DATASET")
    print(
        "Samples:",
        len(y)
    )

    print(
        "Sequence shape:",
        X_sequence.shape
    )

    print(
        "Static shape:",
        X_static.shape
    )

    print(
        "Risky:",
        risky
    )

    print(
        "Non-risky:",
        non_risky
    )

    print(
        "Risk rate:",
        round(
            risky / len(y) * 100,
            4
        ),
        "%"
    )

    print(
        "Saved to:",
        output_folder
    )


# ============================================================
# 6. BUILD BOTH
# ============================================================

build_sequence_dataset(
    18,
    OUTPUT_18
)

build_sequence_dataset(
    24,
    OUTPUT_24
)

print("\n======================================")
print("J18 + J24 RAW SEQUENCES COMPLETE")
print("======================================")