import os
import pickle
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


RAW_DIR = "Data/sequential_enhanced_raw"
OLD_READY_DIR = "Data/sequential_model_ready"
OUTPUT_DIR = "Data/sequential_enhanced_model_ready"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# LOAD ENHANCED RAW DATA
# ============================================================

X_seq = np.load(
    f"{RAW_DIR}/X_sequence_raw_enhanced.npy"
)

X_static = np.load(
    f"{RAW_DIR}/X_static_raw.npy"
)

y = np.load(
    f"{RAW_DIR}/y.npy"
)

loan_ids = np.load(
    f"{RAW_DIR}/loan_ids.npy"
)

with open(
    f"{RAW_DIR}/temporal_features_enhanced.pkl",
    "rb"
) as f:
    temporal_features = pickle.load(f)

with open(
    f"{RAW_DIR}/static_features.pkl",
    "rb"
) as f:
    static_features = pickle.load(f)


print("Enhanced sequence:", X_seq.shape)
print("Static:", X_static.shape)
print("Target:", y.shape)


# ============================================================
# LOAD ORIGINAL SPLIT IDS
# This keeps Experiment F directly comparable
# ============================================================

train_ids = np.load(
    f"{OLD_READY_DIR}/loan_ids_train.npy"
)

val_ids = np.load(
    f"{OLD_READY_DIR}/loan_ids_val.npy"
)

test_ids = np.load(
    f"{OLD_READY_DIR}/loan_ids_test.npy"
)


# ============================================================
# MAP LOAN IDS TO ENHANCED DATA
# ============================================================

id_to_index = {
    str(loan_id): i
    for i, loan_id in enumerate(loan_ids)
}


def get_indices(ids):

    indices = []

    missing = []

    for loan_id in ids:

        key = str(loan_id)

        if key in id_to_index:
            indices.append(
                id_to_index[key]
            )
        else:
            missing.append(key)

    return np.asarray(indices), missing


train_idx, missing_train = get_indices(train_ids)
val_idx, missing_val = get_indices(val_ids)
test_idx, missing_test = get_indices(test_ids)


print("\nMissing IDs:")
print("Train:", len(missing_train))
print("Val:", len(missing_val))
print("Test:", len(missing_test))


if (
    len(missing_train) > 0
    or len(missing_val) > 0
    or len(missing_test) > 0
):
    raise ValueError(
        "Some original split loan IDs were not found."
    )


# ============================================================
# CREATE SAME SPLITS
# ============================================================

X_seq_train = X_seq[train_idx]
X_seq_val = X_seq[val_idx]
X_seq_test = X_seq[test_idx]

X_static_train = X_static[train_idx]
X_static_val = X_static[val_idx]
X_static_test = X_static[test_idx]

y_train = y[train_idx]
y_val = y[val_idx]
y_test = y[test_idx]


print("\nBefore normalisation:")

print("Train sequence:", X_seq_train.shape)
print("Val sequence:", X_seq_val.shape)
print("Test sequence:", X_seq_test.shape)

print("Risky train:", int(y_train.sum()))
print("Risky val:", int(y_val.sum()))
print("Risky test:", int(y_test.sum()))


# ============================================================
# TEMPORAL IMPUTATION
# Fit ONLY on training set
# ============================================================

n_train, seq_len, n_features = X_seq_train.shape

train_2d = X_seq_train.reshape(
    -1,
    n_features
)

val_2d = X_seq_val.reshape(
    -1,
    n_features
)

test_2d = X_seq_test.reshape(
    -1,
    n_features
)


temporal_imputer = SimpleImputer(
    strategy="median"
)

train_2d = temporal_imputer.fit_transform(
    train_2d
)

val_2d = temporal_imputer.transform(
    val_2d
)

test_2d = temporal_imputer.transform(
    test_2d
)


# ============================================================
# TEMPORAL NORMALISATION
# Fit ONLY on training set
# ============================================================

temporal_scaler = StandardScaler()

train_2d = temporal_scaler.fit_transform(
    train_2d
)

val_2d = temporal_scaler.transform(
    val_2d
)

test_2d = temporal_scaler.transform(
    test_2d
)


X_seq_train = train_2d.reshape(
    X_seq_train.shape
).astype(np.float32)

X_seq_val = val_2d.reshape(
    X_seq_val.shape
).astype(np.float32)

X_seq_test = test_2d.reshape(
    X_seq_test.shape
).astype(np.float32)


# ============================================================
# STATIC IMPUTATION + NORMALISATION
# ============================================================

static_imputer = SimpleImputer(
    strategy="median"
)

X_static_train = static_imputer.fit_transform(
    X_static_train
)

X_static_val = static_imputer.transform(
    X_static_val
)

X_static_test = static_imputer.transform(
    X_static_test
)


static_scaler = StandardScaler()

X_static_train = static_scaler.fit_transform(
    X_static_train
).astype(np.float32)

X_static_val = static_scaler.transform(
    X_static_val
).astype(np.float32)

X_static_test = static_scaler.transform(
    X_static_test
).astype(np.float32)


# ============================================================
# SAVE
# ============================================================

np.save(
    f"{OUTPUT_DIR}/X_sequence_train.npy",
    X_seq_train
)

np.save(
    f"{OUTPUT_DIR}/X_sequence_val.npy",
    X_seq_val
)

np.save(
    f"{OUTPUT_DIR}/X_sequence_test.npy",
    X_seq_test
)

np.save(
    f"{OUTPUT_DIR}/X_static_train.npy",
    X_static_train
)

np.save(
    f"{OUTPUT_DIR}/X_static_val.npy",
    X_static_val
)

np.save(
    f"{OUTPUT_DIR}/X_static_test.npy",
    X_static_test
)

np.save(
    f"{OUTPUT_DIR}/y_train.npy",
    y_train
)

np.save(
    f"{OUTPUT_DIR}/y_val.npy",
    y_val
)

np.save(
    f"{OUTPUT_DIR}/y_test.npy",
    y_test
)

np.save(
    f"{OUTPUT_DIR}/loan_ids_train.npy",
    train_ids
)

np.save(
    f"{OUTPUT_DIR}/loan_ids_val.npy",
    val_ids
)

np.save(
    f"{OUTPUT_DIR}/loan_ids_test.npy",
    test_ids
)


with open(
    f"{OUTPUT_DIR}/temporal_imputer.pkl",
    "wb"
) as f:
    pickle.dump(
        temporal_imputer,
        f
    )

with open(
    f"{OUTPUT_DIR}/temporal_scaler.pkl",
    "wb"
) as f:
    pickle.dump(
        temporal_scaler,
        f
    )

with open(
    f"{OUTPUT_DIR}/static_imputer.pkl",
    "wb"
) as f:
    pickle.dump(
        static_imputer,
        f
    )

with open(
    f"{OUTPUT_DIR}/static_scaler.pkl",
    "wb"
) as f:
    pickle.dump(
        static_scaler,
        f
    )

with open(
    f"{OUTPUT_DIR}/temporal_features.pkl",
    "wb"
) as f:
    pickle.dump(
        temporal_features,
        f
    )

with open(
    f"{OUTPUT_DIR}/static_features.pkl",
    "wb"
) as f:
    pickle.dump(
        static_features,
        f
    )


print("\n======================================")
print("EXPERIMENT F DATA READY")
print("======================================")

print("Train:", X_seq_train.shape)
print("Validation:", X_seq_val.shape)
print("Test:", X_seq_test.shape)

print("Temporal features:", len(temporal_features))
print("Static features:", len(static_features))

print("\nSaved to:")
print(OUTPUT_DIR)