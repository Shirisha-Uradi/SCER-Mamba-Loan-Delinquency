import os
import gc
import pickle
import numpy as np

from sklearn.model_selection import train_test_split


SEED = 42


def split_and_normalise(input_folder, output_folder):

    print("\n" + "=" * 60)
    print("PROCESSING:", input_folder)
    print("=" * 60)

    os.makedirs(output_folder, exist_ok=True)

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    X_seq = np.load(
        os.path.join(input_folder, "X_sequence_raw.npy")
    ).astype(np.float32)

    X_static = np.load(
        os.path.join(input_folder, "X_static_raw.npy")
    ).astype(np.float32)

    y = np.load(
        os.path.join(input_folder, "y.npy")
    )

    loan_ids = np.load(
        os.path.join(input_folder, "loan_ids.npy"),
        allow_pickle=True
    )

    print("Sequence:", X_seq.shape)
    print("Static  :", X_static.shape)
    print("Targets :", y.shape)

    assert len(X_seq) == len(X_static) == len(y) == len(loan_ids)
    assert X_seq.shape[2] == 13
    assert X_static.shape[1] == 12

    # --------------------------------------------------------
    # STRATIFIED 70 / 15 / 15 SPLIT
    # --------------------------------------------------------

    indices = np.arange(len(y))

    train_idx, temp_idx = train_test_split(
        indices,
        test_size=0.30,
        random_state=SEED,
        stratify=y
    )

    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.50,
        random_state=SEED,
        stratify=y[temp_idx]
    )

    X_seq_train = X_seq[train_idx].copy()
    X_seq_val = X_seq[val_idx].copy()
    X_seq_test = X_seq[test_idx].copy()

    X_static_train = X_static[train_idx].copy()
    X_static_val = X_static[val_idx].copy()
    X_static_test = X_static[test_idx].copy()

    y_train = y[train_idx]
    y_val = y[val_idx]
    y_test = y[test_idx]

    ids_train = loan_ids[train_idx]
    ids_val = loan_ids[val_idx]
    ids_test = loan_ids[test_idx]

    # Free original large arrays
    del X_seq
    del X_static
    gc.collect()

    # --------------------------------------------------------
    # TEMPORAL IMPUTATION — TRAINING DATA ONLY
    # --------------------------------------------------------

    X_seq_train[~np.isfinite(X_seq_train)] = np.nan
    X_seq_val[~np.isfinite(X_seq_val)] = np.nan
    X_seq_test[~np.isfinite(X_seq_test)] = np.nan

    temporal_median = np.nanmedian(
        X_seq_train,
        axis=(0, 1)
    ).astype(np.float32)

    temporal_median = np.where(
        np.isfinite(temporal_median),
        temporal_median,
        0.0
    ).astype(np.float32)

    for j in range(X_seq_train.shape[2]):

        for arr in [
            X_seq_train,
            X_seq_val,
            X_seq_test
        ]:

            feature = arr[:, :, j]

            missing = ~np.isfinite(feature)

            if missing.any():
                feature[missing] = temporal_median[j]

    # --------------------------------------------------------
    # TEMPORAL NORMALIZATION — FIT TRAIN ONLY
    # --------------------------------------------------------

    temporal_mean = X_seq_train.mean(
        axis=(0, 1)
    ).astype(np.float32)

    temporal_std = X_seq_train.std(
        axis=(0, 1)
    ).astype(np.float32)

    temporal_std[
        temporal_std < 1e-8
    ] = 1.0

    for j in range(X_seq_train.shape[2]):

        X_seq_train[:, :, j] = (
            X_seq_train[:, :, j]
            - temporal_mean[j]
        ) / temporal_std[j]

        X_seq_val[:, :, j] = (
            X_seq_val[:, :, j]
            - temporal_mean[j]
        ) / temporal_std[j]

        X_seq_test[:, :, j] = (
            X_seq_test[:, :, j]
            - temporal_mean[j]
        ) / temporal_std[j]

    # --------------------------------------------------------
    # STATIC IMPUTATION — TRAINING DATA ONLY
    # --------------------------------------------------------

    X_static_train[
        ~np.isfinite(X_static_train)
    ] = np.nan

    X_static_val[
        ~np.isfinite(X_static_val)
    ] = np.nan

    X_static_test[
        ~np.isfinite(X_static_test)
    ] = np.nan

    static_median = np.nanmedian(
        X_static_train,
        axis=0
    ).astype(np.float32)

    static_median = np.where(
        np.isfinite(static_median),
        static_median,
        0.0
    ).astype(np.float32)

    for j in range(X_static_train.shape[1]):

        for arr in [
            X_static_train,
            X_static_val,
            X_static_test
        ]:

            missing = ~np.isfinite(
                arr[:, j]
            )

            if missing.any():
                arr[missing, j] = static_median[j]

    # --------------------------------------------------------
    # STATIC NORMALIZATION — FIT TRAIN ONLY
    # --------------------------------------------------------

    static_mean = X_static_train.mean(
        axis=0
    ).astype(np.float32)

    static_std = X_static_train.std(
        axis=0
    ).astype(np.float32)

    static_std[
        static_std < 1e-8
    ] = 1.0

    X_static_train = (
        X_static_train
        - static_mean
    ) / static_std

    X_static_val = (
        X_static_val
        - static_mean
    ) / static_std

    X_static_test = (
        X_static_test
        - static_mean
    ) / static_std

    # --------------------------------------------------------
    # SAFETY CHECKS
    # --------------------------------------------------------

    assert np.isfinite(
        X_seq_train
    ).all()

    assert np.isfinite(
        X_seq_val
    ).all()

    assert np.isfinite(
        X_seq_test
    ).all()

    assert np.isfinite(
        X_static_train
    ).all()

    assert np.isfinite(
        X_static_val
    ).all()

    assert np.isfinite(
        X_static_test
    ).all()

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    arrays = {
        "X_sequence_train.npy": X_seq_train,
        "X_sequence_val.npy": X_seq_val,
        "X_sequence_test.npy": X_seq_test,

        "X_static_train.npy": X_static_train,
        "X_static_val.npy": X_static_val,
        "X_static_test.npy": X_static_test,

        "y_train.npy": y_train,
        "y_val.npy": y_val,
        "y_test.npy": y_test,

        "loan_ids_train.npy": ids_train,
        "loan_ids_val.npy": ids_val,
        "loan_ids_test.npy": ids_test,
    }

    for filename, array in arrays.items():

        np.save(
            os.path.join(
                output_folder,
                filename
            ),
            array
        )

    # Save normalization statistics
    np.savez(
        os.path.join(
            output_folder,
            "normalisation_stats.npz"
        ),
        temporal_median=temporal_median,
        temporal_mean=temporal_mean,
        temporal_std=temporal_std,
        static_median=static_median,
        static_mean=static_mean,
        static_std=static_std
    )

    # Copy feature names
    for filename in [
        "temporal_features.pkl",
        "static_features.pkl"
    ]:

        source = os.path.join(
            input_folder,
            filename
        )

        if os.path.exists(source):

            with open(source, "rb") as f:
                data = pickle.load(f)

            with open(
                os.path.join(
                    output_folder,
                    filename
                ),
                "wb"
            ) as f:
                pickle.dump(data, f)

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print("\nTRAIN:", X_seq_train.shape)
    print(
        "Risky train:",
        int(y_train.sum())
    )

    print("\nVAL:", X_seq_val.shape)
    print(
        "Risky val:",
        int(y_val.sum())
    )

    print("\nTEST:", X_seq_test.shape)
    print(
        "Risky test:",
        int(y_test.sum())
    )

    print(
        "\nSaved:",
        output_folder
    )

    # Clean memory before next dataset
    del (
        X_seq_train,
        X_seq_val,
        X_seq_test,
        X_static_train,
        X_static_val,
        X_static_test
    )

    gc.collect()


# ============================================================
# J18
# ============================================================

split_and_normalise(
    r"Data/sequential_enhanced_raw_age18",
    r"Data/sequential_enhanced_model_ready_age18"
)

# ============================================================
# J24
# ============================================================

split_and_normalise(
    r"Data/sequential_enhanced_raw_age24",
    r"Data/sequential_enhanced_model_ready_age24"
)

print("\n======================================")
print("J18 + J24 SPLIT/NORMALISATION COMPLETE")
print("======================================")