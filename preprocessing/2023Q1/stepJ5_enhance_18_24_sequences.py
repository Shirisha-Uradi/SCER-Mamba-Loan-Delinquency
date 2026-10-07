import os
import pickle
import numpy as np


# ============================================================
# FEATURE ENGINEERING FUNCTION
# ============================================================

def enhance_sequences(input_folder, output_folder):

    print("\n======================================")
    print("PROCESSING:", input_folder)
    print("======================================")

    os.makedirs(
        output_folder,
        exist_ok=True
    )

    X = np.load(
        os.path.join(
            input_folder,
            "X_sequence_raw.npy"
        )
    )

    X_static = np.load(
        os.path.join(
            input_folder,
            "X_static_raw.npy"
        )
    )

    y = np.load(
        os.path.join(
            input_folder,
            "y.npy"
        )
    )

    loan_ids = np.load(
        os.path.join(
            input_folder,
            "loan_ids.npy"
        ),
        allow_pickle=True
    )

    print("Original sequence shape:", X.shape)

    n_samples = X.shape[0]
    seq_len = X.shape[1]

    # ========================================================
    # ORIGINAL FEATURE POSITIONS
    # ========================================================

    delinquency = X[:, :, 2]

    is_30plus = X[:, :, 4]

    current_upb = X[:, :, 1]

    # ========================================================
    # 1. DELINQUENCY CHANGE
    # ========================================================

    delinquency_change = np.zeros(
        (n_samples, seq_len),
        dtype=np.float32
    )

    delinquency_change[:, 1:] = (
        delinquency[:, 1:]
        -
        delinquency[:, :-1]
    )

    # ========================================================
    # 2. DELINQUENCY WORSENING
    # ========================================================

    delinquency_worsening = (
        delinquency_change > 0
    ).astype(np.float32)

    # ========================================================
    # 3. CUMULATIVE MAX DELINQUENCY
    # ========================================================

    cumulative_max_delinquency = (
        np.maximum.accumulate(
            delinquency,
            axis=1
        )
    ).astype(np.float32)

    # ========================================================
    # 4. CONSECUTIVE 30+ MONTHS
    # ========================================================

    consecutive_30plus = np.zeros(
        (n_samples, seq_len),
        dtype=np.float32
    )

    for t in range(seq_len):

        if t == 0:

            consecutive_30plus[:, t] = (
                is_30plus[:, t] > 0
            )

        else:

            consecutive_30plus[:, t] = (
                (is_30plus[:, t] > 0)
                *
                (
                    consecutive_30plus[:, t - 1]
                    + 1
                )
            )

    # ========================================================
    # 5. RECENT 3-MONTH 30+ COUNT
    # ========================================================

    recent_3m_count = np.zeros(
        (n_samples, seq_len),
        dtype=np.float32
    )

    for t in range(seq_len):

        start = max(
            0,
            t - 2
        )

        recent_3m_count[:, t] = (
            is_30plus[
                :,
                start:t + 1
            ].sum(axis=1)
        )

    # ========================================================
    # 6. MONTHLY UPB CHANGE
    # ========================================================

    monthly_upb_change = np.zeros(
        (n_samples, seq_len),
        dtype=np.float32
    )

    monthly_upb_change[:, 1:] = (
        current_upb[:, 1:]
        -
        current_upb[:, :-1]
    )

    # ========================================================
    # STACK NEW FEATURES
    # ========================================================

    new_features = np.stack(
        [
            delinquency_change,
            delinquency_worsening,
            cumulative_max_delinquency,
            consecutive_30plus,
            recent_3m_count,
            monthly_upb_change,
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

    # ========================================================
    # SAVE
    # ========================================================

    np.save(
        os.path.join(
            output_folder,
            "X_sequence_raw.npy"
        ),
        X_enhanced
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

    np.save(
        os.path.join(
            output_folder,
            "loan_ids.npy"
        ),
        loan_ids
    )

    temporal_features = [
        "current_interest_rate",
        "current_actual_upb",
        "delinquency_status_numeric",
        "remaining_months_to_maturity",
        "is_30plus_delinquent",
        "is_60plus_delinquent",
        "is_90plus_delinquent",
        "delinquency_change",
        "delinquency_worsening",
        "cumulative_max_delinquency",
        "consecutive_30plus_months",
        "recent_3m_30plus_count",
        "monthly_upb_change",
    ]

    with open(
        os.path.join(
            output_folder,
            "temporal_features.pkl"
        ),
        "wb"
    ) as f:

        pickle.dump(
            temporal_features,
            f
        )

    print("\nEnhanced sequence shape:", X_enhanced.shape)
    print("Static shape:", X_static.shape)
    print("Targets:", y.shape)
    print("Risky:", int(y.sum()))
    print("Saved to:", output_folder)


# ============================================================
# AGE 18
# ============================================================

enhance_sequences(
    r"Data/sequential_raw_age18",
    r"Data/sequential_enhanced_raw_age18"
)

# ============================================================
# AGE 24
# ============================================================

enhance_sequences(
    r"Data/sequential_raw_age24",
    r"Data/sequential_enhanced_raw_age24"
)

print("\n======================================")
print("J18 + J24 ENHANCED FEATURES COMPLETE")
print("======================================")