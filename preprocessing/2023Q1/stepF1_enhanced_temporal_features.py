import os
import pickle
import numpy as np

INPUT_DIR = "Data/sequential_raw"
OUTPUT_DIR = "Data/sequential_enhanced_raw"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# LOAD RAW DATA
# ============================================================

X = np.load(
    f"{INPUT_DIR}/X_sequence_raw.npy"
).astype(np.float32)

X_static = np.load(
    f"{INPUT_DIR}/X_static_raw.npy"
)

y = np.load(
    f"{INPUT_DIR}/y.npy"
)

loan_ids = np.load(
    f"{INPUT_DIR}/loan_ids.npy"
)

with open(
    f"{INPUT_DIR}/temporal_features.pkl",
    "rb"
) as f:
    old_features = pickle.load(f)

with open(
    f"{INPUT_DIR}/static_features.pkl",
    "rb"
) as f:
    static_features = pickle.load(f)

print("Original sequence shape:", X.shape)

# ============================================================
# ORIGINAL FEATURE INDEXES
# ============================================================

UPB_IDX = 1
DELINQ_IDX = 2
DPD30_IDX = 4

delinq = X[:, :, DELINQ_IDX]
upb = X[:, :, UPB_IDX]
dpd30 = X[:, :, DPD30_IDX]

n_samples, seq_len = delinq.shape

# ============================================================
# 1. DELINQUENCY CHANGE
# ============================================================

delinq_change = np.zeros_like(delinq)

delinq_change[:, 1:] = (
    delinq[:, 1:] -
    delinq[:, :-1]
)

# ============================================================
# 2. DELINQUENCY WORSENING
# ============================================================

delinq_worsening = (
    delinq_change > 0
).astype(np.float32)

# ============================================================
# 3. CUMULATIVE MAXIMUM DELINQUENCY
# ============================================================

max_delinq = np.maximum.accumulate(
    delinq,
    axis=1
).astype(np.float32)

# ============================================================
# 4. CONSECUTIVE 30+ DELINQUENCY STREAK
# ============================================================

delinq_streak = np.zeros_like(
    dpd30,
    dtype=np.float32
)

for t in range(seq_len):

    if t == 0:

        delinq_streak[:, t] = (
            dpd30[:, t] > 0
        ).astype(np.float32)

    else:

        current_bad = (
            dpd30[:, t] > 0
        )

        delinq_streak[:, t] = np.where(
            current_bad,
            delinq_streak[:, t - 1] + 1,
            0
        )

# ============================================================
# 5. ROLLING 3-MONTH 30+ DELINQUENCY COUNT
# ============================================================

recent_3m_delinquency = np.zeros_like(
    dpd30,
    dtype=np.float32
)

for t in range(seq_len):

    start = max(0, t - 2)

    recent_3m_delinquency[:, t] = (
        dpd30[:, start:t + 1].sum(axis=1)
    )

# ============================================================
# 6. MONTHLY UPB CHANGE
# ============================================================

upb_change = np.zeros_like(
    upb,
    dtype=np.float32
)

previous_upb = upb[:, :-1]

upb_change[:, 1:] = (
    upb[:, 1:] - previous_upb
) / (
    np.abs(previous_upb) + 1e-6
)

# Avoid extreme values
upb_change = np.clip(
    upb_change,
    -5,
    5
)

# ============================================================
# STACK NEW FEATURES
# ============================================================

new_features = np.stack(
    [
        delinq_change,
        delinq_worsening,
        max_delinq,
        delinq_streak,
        recent_3m_delinquency,
        upb_change
    ],
    axis=2
)

X_enhanced = np.concatenate(
    [
        X,
        new_features
    ],
    axis=2
).astype(np.float32)

enhanced_features = list(old_features) + [
    "delinquency_change",
    "delinquency_worsening",
    "cumulative_max_delinquency",
    "consecutive_30plus_months",
    "recent_3m_30plus_count",
    "monthly_upb_change"
]

# ============================================================
# SAVE
# ============================================================

np.save(
    f"{OUTPUT_DIR}/X_sequence_raw_enhanced.npy",
    X_enhanced
)

np.save(
    f"{OUTPUT_DIR}/X_static_raw.npy",
    X_static
)

np.save(
    f"{OUTPUT_DIR}/y.npy",
    y
)

np.save(
    f"{OUTPUT_DIR}/loan_ids.npy",
    loan_ids
)

with open(
    f"{OUTPUT_DIR}/temporal_features_enhanced.pkl",
    "wb"
) as f:
    pickle.dump(
        enhanced_features,
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

# ============================================================
# CHECK
# ============================================================

print("\nEnhanced sequence shape:", X_enhanced.shape)

print("\nEnhanced temporal features:")

for i, feature in enumerate(enhanced_features):
    print(i, feature)

print("\nRisky samples:", int(y.sum()))
print("Non-risky samples:", int((y == 0).sum()))

print("\nSaved to:")
print(OUTPUT_DIR)