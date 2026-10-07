import numpy as np
import pickle

X = np.load("Data/sequential_raw/X_sequence_raw.npy")

with open(
    "Data/sequential_raw/temporal_features.pkl",
    "rb"
) as f:
    temporal_features = pickle.load(f)

with open(
    "Data/sequential_raw/static_features.pkl",
    "rb"
) as f:
    static_features = pickle.load(f)

print("Sequence shape:", X.shape)

print("\nTemporal features:")
for i, feature in enumerate(temporal_features):
    print(i, feature)

print("\nStatic features:")
for i, feature in enumerate(static_features):
    print(i, feature)