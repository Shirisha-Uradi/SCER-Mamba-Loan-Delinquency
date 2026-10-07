import numpy as np
import os
import pickle
import joblib

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


# ============================================================
# STEP 5.2 - SPLIT + NORMALISE SEQUENTIAL DATA
# ============================================================

input_folder = "Data/sequential_raw"
output_folder = "Data/sequential_model_ready"

os.makedirs(
    output_folder,
    exist_ok=True
)

print("========================================")
print("STEP 5.2 - SPLIT + NORMALISE SEQUENCES")
print("========================================")


# ============================================================
# 1. LOAD RAW ARRAYS
# ============================================================

X_sequence = np.load(
    f"{input_folder}/X_sequence_raw.npy"
)

X_static = np.load(
    f"{input_folder}/X_static_raw.npy"
)

y = np.load(
    f"{input_folder}/y.npy"
)

loan_ids = np.load(
    f"{input_folder}/loan_ids.npy"
)


with open(
    f"{input_folder}/temporal_features.pkl",
    "rb"
) as file:

    temporal_features = pickle.load(
        file
    )


with open(
    f"{input_folder}/static_features.pkl",
    "rb"
) as file:

    static_features = pickle.load(
        file
    )


print("\nLoaded shapes:")

print(
    "Sequence:",
    X_sequence.shape
)

print(
    "Static:",
    X_static.shape
)

print(
    "Target:",
    y.shape
)

print(
    "Loan IDs:",
    loan_ids.shape
)


# ============================================================
# 2. CREATE SAMPLE INDICES
# ============================================================

indices = np.arange(
    len(y)
)


# ============================================================
# 3. TRAIN / VALIDATION / TEST SPLIT
#
# 70% Training
# 15% Validation
# 15% Test
#
# Stratified by target.
# ============================================================

train_idx, temp_idx = train_test_split(
    indices,
    test_size=0.30,
    random_state=42,
    stratify=y
)

val_idx, test_idx = train_test_split(
    temp_idx,
    test_size=0.50,
    random_state=42,
    stratify=y[temp_idx]
)


# ============================================================
# 4. CREATE SPLITS
# ============================================================

X_seq_train = X_sequence[
    train_idx
]

X_seq_val = X_sequence[
    val_idx
]

X_seq_test = X_sequence[
    test_idx
]


X_static_train = X_static[
    train_idx
]

X_static_val = X_static[
    val_idx
]

X_static_test = X_static[
    test_idx
]


y_train = y[
    train_idx
]

y_val = y[
    val_idx
]

y_test = y[
    test_idx
]


loan_ids_train = loan_ids[
    train_idx
]

loan_ids_val = loan_ids[
    val_idx
]

loan_ids_test = loan_ids[
    test_idx
]


print("\n========================================")
print("SPLIT SIZES")
print("========================================")

print(
    "Training:",
    len(y_train)
)

print(
    "Validation:",
    len(y_val)
)

print(
    "Test:",
    len(y_test)
)


# ============================================================
# 5. VERIFY NO LOAN OVERLAP
# ============================================================

train_ids = set(
    loan_ids_train.tolist()
)

val_ids = set(
    loan_ids_val.tolist()
)

test_ids = set(
    loan_ids_test.tolist()
)


print("\nLoan overlap checks:")

print(
    "Train vs Validation:",
    len(
        train_ids.intersection(
            val_ids
        )
    )
)

print(
    "Train vs Test:",
    len(
        train_ids.intersection(
            test_ids
        )
    )
)

print(
    "Validation vs Test:",
    len(
        val_ids.intersection(
            test_ids
        )
    )
)


# ============================================================
# 6. TEMPORAL FEATURE GROUPS
# ============================================================

temporal_continuous_indices = [
    temporal_features.index(
        "current_interest_rate"
    ),
    temporal_features.index(
        "current_actual_upb"
    ),
    temporal_features.index(
        "delinquency_status_numeric"
    ),
    temporal_features.index(
        "remaining_months_to_maturity"
    )
]


temporal_binary_indices = [
    temporal_features.index(
        "is_30plus_delinquent"
    ),
    temporal_features.index(
        "is_60plus_delinquent"
    ),
    temporal_features.index(
        "is_90plus_delinquent"
    )
]


# ============================================================
# 7. TEMPORAL IMPUTATION
#
# Fit ONLY on training observations.
# ============================================================

n_train = X_seq_train.shape[0]
n_val = X_seq_val.shape[0]
n_test = X_seq_test.shape[0]

sequence_length = X_seq_train.shape[1]


train_continuous = X_seq_train[
    :,
    :,
    temporal_continuous_indices
].reshape(
    -1,
    len(
        temporal_continuous_indices
    )
)


val_continuous = X_seq_val[
    :,
    :,
    temporal_continuous_indices
].reshape(
    -1,
    len(
        temporal_continuous_indices
    )
)


test_continuous = X_seq_test[
    :,
    :,
    temporal_continuous_indices
].reshape(
    -1,
    len(
        temporal_continuous_indices
    )
)


temporal_imputer = SimpleImputer(
    strategy="median"
)

train_continuous = (
    temporal_imputer.fit_transform(
        train_continuous
    )
)

val_continuous = (
    temporal_imputer.transform(
        val_continuous
    )
)

test_continuous = (
    temporal_imputer.transform(
        test_continuous
    )
)


# ============================================================
# 8. TEMPORAL NORMALISATION
#
# StandardScaler fitted ONLY on training data.
# ============================================================

temporal_scaler = StandardScaler()

train_continuous = (
    temporal_scaler.fit_transform(
        train_continuous
    )
)

val_continuous = (
    temporal_scaler.transform(
        val_continuous
    )
)

test_continuous = (
    temporal_scaler.transform(
        test_continuous
    )
)


# Restore sequence dimensions
train_continuous = train_continuous.reshape(
    n_train,
    sequence_length,
    len(
        temporal_continuous_indices
    )
)

val_continuous = val_continuous.reshape(
    n_val,
    sequence_length,
    len(
        temporal_continuous_indices
    )
)

test_continuous = test_continuous.reshape(
    n_test,
    sequence_length,
    len(
        temporal_continuous_indices
    )
)


# Copy arrays before replacement
X_seq_train_ready = (
    X_seq_train.copy()
)

X_seq_val_ready = (
    X_seq_val.copy()
)

X_seq_test_ready = (
    X_seq_test.copy()
)


# Insert scaled continuous values
X_seq_train_ready[
    :,
    :,
    temporal_continuous_indices
] = train_continuous

X_seq_val_ready[
    :,
    :,
    temporal_continuous_indices
] = val_continuous

X_seq_test_ready[
    :,
    :,
    temporal_continuous_indices
] = test_continuous


# Binary temporal features:
# replace missing values with zero

for index in temporal_binary_indices:

    X_seq_train_ready[
        :,
        :,
        index
    ] = np.nan_to_num(
        X_seq_train_ready[
            :,
            :,
            index
        ],
        nan=0.0
    )

    X_seq_val_ready[
        :,
        :,
        index
    ] = np.nan_to_num(
        X_seq_val_ready[
            :,
            :,
            index
        ],
        nan=0.0
    )

    X_seq_test_ready[
        :,
        :,
        index
    ] = np.nan_to_num(
        X_seq_test_ready[
            :,
            :,
            index
        ],
        nan=0.0
    )


# ============================================================
# 9. STATIC FEATURE GROUPS
# ============================================================

static_binary_names = [
    "has_co_borrower",
    "co_borrower_score_missing",
    "mortgage_insurance_missing"
]


static_binary_indices = [
    static_features.index(
        feature
    )
    for feature in static_binary_names
]


static_continuous_indices = [
    index
    for index in range(
        len(static_features)
    )
    if index not in static_binary_indices
]


# ============================================================
# 10. STATIC IMPUTATION
#
# Fit ONLY on training data.
# ============================================================

static_imputer = SimpleImputer(
    strategy="median"
)


train_static_continuous = (
    static_imputer.fit_transform(
        X_static_train[
            :,
            static_continuous_indices
        ]
    )
)


val_static_continuous = (
    static_imputer.transform(
        X_static_val[
            :,
            static_continuous_indices
        ]
    )
)


test_static_continuous = (
    static_imputer.transform(
        X_static_test[
            :,
            static_continuous_indices
        ]
    )
)


# ============================================================
# 11. STATIC NORMALISATION
# ============================================================

static_scaler = StandardScaler()


train_static_continuous = (
    static_scaler.fit_transform(
        train_static_continuous
    )
)


val_static_continuous = (
    static_scaler.transform(
        val_static_continuous
    )
)


test_static_continuous = (
    static_scaler.transform(
        test_static_continuous
    )
)


X_static_train_ready = (
    X_static_train.copy()
)

X_static_val_ready = (
    X_static_val.copy()
)

X_static_test_ready = (
    X_static_test.copy()
)


X_static_train_ready[
    :,
    static_continuous_indices
] = train_static_continuous

X_static_val_ready[
    :,
    static_continuous_indices
] = val_static_continuous

X_static_test_ready[
    :,
    static_continuous_indices
] = test_static_continuous


# Binary static features:
# fill missing values with zero

for index in static_binary_indices:

    X_static_train_ready[
        :,
        index
    ] = np.nan_to_num(
        X_static_train_ready[
            :,
            index
        ],
        nan=0.0
    )

    X_static_val_ready[
        :,
        index
    ] = np.nan_to_num(
        X_static_val_ready[
            :,
            index
        ],
        nan=0.0
    )

    X_static_test_ready[
        :,
        index
    ] = np.nan_to_num(
        X_static_test_ready[
            :,
            index
        ],
        nan=0.0
    )


# ============================================================
# 12. FINAL MISSING-VALUE CHECK
# ============================================================

print("\n========================================")
print("FINAL MISSING VALUE CHECK")
print("========================================")

print(
    "Train sequence missing:",
    np.isnan(
        X_seq_train_ready
    ).sum()
)

print(
    "Validation sequence missing:",
    np.isnan(
        X_seq_val_ready
    ).sum()
)

print(
    "Test sequence missing:",
    np.isnan(
        X_seq_test_ready
    ).sum()
)


print(
    "Train static missing:",
    np.isnan(
        X_static_train_ready
    ).sum()
)

print(
    "Validation static missing:",
    np.isnan(
        X_static_val_ready
    ).sum()
)

print(
    "Test static missing:",
    np.isnan(
        X_static_test_ready
    ).sum()
)


# ============================================================
# 13. TARGET DISTRIBUTIONS
# ============================================================

def print_target_distribution(
    name,
    target
):

    unique, counts = np.unique(
        target,
        return_counts=True
    )

    print(
        f"\n{name}:"
    )

    for value, count in zip(
        unique,
        counts
    ):

        percentage = (
            count /
            len(target)
        ) * 100

        print(
            f"{value}: "
            f"{count} "
            f"({percentage:.4f}%)"
        )


print("\n========================================")
print("TARGET DISTRIBUTIONS")
print("========================================")


print_target_distribution(
    "TRAIN",
    y_train
)

print_target_distribution(
    "VALIDATION",
    y_val
)

print_target_distribution(
    "TEST",
    y_test
)


# ============================================================
# 14. SAVE MODEL-READY ARRAYS
# ============================================================

np.save(
    f"{output_folder}/X_sequence_train.npy",
    X_seq_train_ready
)

np.save(
    f"{output_folder}/X_sequence_val.npy",
    X_seq_val_ready
)

np.save(
    f"{output_folder}/X_sequence_test.npy",
    X_seq_test_ready
)


np.save(
    f"{output_folder}/X_static_train.npy",
    X_static_train_ready
)

np.save(
    f"{output_folder}/X_static_val.npy",
    X_static_val_ready
)

np.save(
    f"{output_folder}/X_static_test.npy",
    X_static_test_ready
)


np.save(
    f"{output_folder}/y_train.npy",
    y_train
)

np.save(
    f"{output_folder}/y_val.npy",
    y_val
)

np.save(
    f"{output_folder}/y_test.npy",
    y_test
)


np.save(
    f"{output_folder}/loan_ids_train.npy",
    loan_ids_train
)

np.save(
    f"{output_folder}/loan_ids_val.npy",
    loan_ids_val
)

np.save(
    f"{output_folder}/loan_ids_test.npy",
    loan_ids_test
)


# ============================================================
# 15. SAVE PREPROCESSING OBJECTS
# ============================================================

joblib.dump(
    temporal_imputer,
    f"{output_folder}/temporal_imputer.pkl"
)

joblib.dump(
    temporal_scaler,
    f"{output_folder}/temporal_scaler.pkl"
)

joblib.dump(
    static_imputer,
    f"{output_folder}/static_imputer.pkl"
)

joblib.dump(
    static_scaler,
    f"{output_folder}/static_scaler.pkl"
)


# ============================================================
# 16. COMPLETE
# ============================================================

print("\n========================================")
print("STEP 5.2 COMPLETED")
print("========================================")

print(
    "Training sequence shape:",
    X_seq_train_ready.shape
)

print(
    "Validation sequence shape:",
    X_seq_val_ready.shape
)

print(
    "Test sequence shape:",
    X_seq_test_ready.shape
)

print(
    "Training static shape:",
    X_static_train_ready.shape
)

print(
    "Files saved in:",
    output_folder
)