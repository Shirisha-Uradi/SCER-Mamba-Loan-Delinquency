# ============================================================
# HISTORY-LENGTH STUDY (2015Q1): SCER-MAMBA vs LSTM vs XGBOOST
# ------------------------------------------------------------
# Question: does SCER-Mamba's advantage grow with longer
# repayment histories (12 -> 24 -> 36 -> 48 months)?
#
# Before running:
#   1. Runtime -> Change runtime type -> T4 GPU
#   2. !pip install -U transformers
#   3. Upload fm2015_age12/24/36/48.zip to My Drive/SCER_Mamba/
#
# - Same settings for every history length
# - Focal loss, real class ratio, threshold chosen on validation
# - 3 seeds per model, test set used once per trained model
# - RESUMABLE: every finished run is saved to Google Drive.
#   If Colab resets, just run this cell again.
# - Set AGES_TO_RUN to run one history length at a time
#   (useful if the GPU time limit is short).
# ============================================================

import os
import gc
import time
import random
import zipfile
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from transformers import MambaConfig, MambaModel
from sklearn.metrics import (
    average_precision_score, roc_auc_score, precision_recall_curve,
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)
from google.colab import drive

AGES_TO_RUN = [48]     # change to [12], [24] or [36] in later sessions
SEEDS = [42, 123, 2026]
MODELS = ["SCER-Mamba", "LSTM"]

drive.mount("/content/drive")
DRIVE_DIR = "/content/drive/MyDrive/SCER_Mamba"
STUDY_DIR = f"{DRIVE_DIR}/history_length_2015Q1"
os.makedirs(STUDY_DIR, exist_ok=True)
RESULTS_CSV = f"{STUDY_DIR}/results_per_run.csv"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
if device.type != "cuda":
    raise RuntimeError("No GPU. Runtime -> Change runtime type -> T4 GPU, then run again.")


# ------------------------------------------------------------
# Data
# ------------------------------------------------------------
def load_cohort(age):
    zip_path = f"{DRIVE_DIR}/fm2015_age{age}.zip"
    folder = f"/content/fm2015_age{age}"
    if not os.path.exists(folder):
        if not os.path.exists(zip_path):
            raise FileNotFoundError(f"Upload fm2015_age{age}.zip to My Drive/SCER_Mamba/")
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(folder)
    data = {}
    for split in ["train", "val", "test"]:
        data[split] = {
            "seq": np.load(f"{folder}/X_sequence_{split}.npy").astype(np.float32),
            "static": np.load(f"{folder}/X_static_{split}.npy").astype(np.float32),
            "y": np.load(f"{folder}/y_{split}.npy").astype(int),
        }
    return data


def make_loader(d, shuffle, seed=42, batch_size=256):
    ds = TensorDataset(torch.from_numpy(d["seq"]), torch.from_numpy(d["static"]),
                       torch.tensor(d["y"], dtype=torch.float32))
    gen = torch.Generator().manual_seed(seed) if shuffle else None
    return DataLoader(ds, batch_size=batch_size if shuffle else 2048, shuffle=shuffle,
                      generator=gen, num_workers=2, pin_memory=True)


# ------------------------------------------------------------
# Loss and models (same as your final J24 models)
# ------------------------------------------------------------
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p = torch.sigmoid(logits)
        pt = p * targets + (1 - p) * (1 - targets)
        return (((1 - pt) ** self.gamma) * bce).mean()


class SCERMamba(nn.Module):
    """Final SCER-Mamba (S3 config: 1 Mamba layer, hidden 64)."""

    def __init__(self, temporal_features=13, static_features=12, hidden_size=64,
                 num_layers=1, dropout=0.20, state_size=16):
        super().__init__()
        self.static_encoder = nn.Sequential(
            nn.Linear(static_features, 32), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(32, hidden_size), nn.ReLU())
        self.temporal_projection = nn.Linear(temporal_features, hidden_size)
        self.mamba = MambaModel(MambaConfig(
            vocab_size=2, hidden_size=hidden_size, state_size=state_size,
            num_hidden_layers=num_layers, expand=2, conv_kernel=4, use_cache=False))
        self.output_norm = nn.LayerNorm(hidden_size)
        self.temporal_attention = nn.Linear(hidden_size, 1)
        self.risk_event_attention = nn.Sequential(nn.Linear(3, 16), nn.ReLU(), nn.Linear(16, 1))
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 4, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 1))

    def forward(self, sequence, static):
        static_emb = self.static_encoder(static)
        out = self.mamba(inputs_embeds=self.temporal_projection(sequence),
                         use_cache=False).last_hidden_state
        out = self.output_norm(out)
        scores = (self.temporal_attention(out).squeeze(-1)
                  + self.risk_event_attention(sequence[:, :, 4:7]).squeeze(-1))
        att = torch.softmax(scores, dim=1)
        pooled = torch.sum(out * att.unsqueeze(-1), dim=1)
        combined = torch.cat([pooled, out.mean(dim=1), out[:, -1, :], static_emb], dim=1)
        return self.classifier(combined).squeeze(-1), att


class LSTMBaseline(nn.Module):
    """Same LSTM as your J24 comparison."""

    def __init__(self, temporal_features=13, static_features=12, hidden_size=64,
                 num_layers=2, dropout=0.20):
        super().__init__()
        self.lstm = nn.LSTM(temporal_features, hidden_size, num_layers,
                            batch_first=True, dropout=dropout)
        self.static_encoder = nn.Sequential(
            nn.Linear(static_features, 32), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(32, hidden_size), nn.ReLU())
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 1))

    def forward(self, sequence, static):
        _, (h, _) = self.lstm(sequence)
        return self.classifier(torch.cat([h[-1], self.static_encoder(static)], dim=1)).squeeze(-1)


def build(model_name):
    return (SCERMamba() if model_name == "SCER-Mamba" else LSTMBaseline()).to(device)


# ------------------------------------------------------------
# Training / evaluation
# ------------------------------------------------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def predict(model, loader):
    model.eval()
    out = []
    with torch.no_grad():
        for seq, stat, _ in loader:
            logits = model(seq.to(device), stat.to(device))
            if isinstance(logits, tuple):
                logits = logits[0]
            out.append(torch.sigmoid(logits).float().cpu().numpy())
    return np.concatenate(out)


def evaluate(y_val, val_probs, y_test, test_probs):
    p, r, t = precision_recall_curve(y_val, val_probs)
    f1s = 2 * p[:-1] * r[:-1] / (p[:-1] + r[:-1] + 1e-10)
    thr = float(t[int(np.argmax(f1s))])
    pred = (test_probs >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    pr_auc = average_precision_score(y_test, test_probs)
    return {"Val_PR_AUC": average_precision_score(y_val, val_probs), "Threshold": thr,
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred, zero_division=0),
            "Recall": recall_score(y_test, pred, zero_division=0),
            "F1": f1_score(y_test, pred, zero_division=0),
            "ROC_AUC": roc_auc_score(y_test, test_probs),
            "PR_AUC": pr_auc, "Lift": pr_auc / y_test.mean(),
            "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn)}


def train_model(model_name, age, seed, data, loaders, max_epochs=20, patience=4):
    path = f"{STUDY_DIR}/{model_name}_age{age}_seed{seed}.pt"
    model = build(model_name)
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=device))
        return model
    set_seed(seed)
    model = build(model_name)
    train_loader = make_loader(data["train"], shuffle=True, seed=seed)
    criterion = FocalLoss(gamma=2.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    best, bad = -1, 0
    for epoch in range(max_epochs):
        t0 = time.time()
        model.train()
        for seq, stat, tgt in train_loader:
            seq, stat, tgt = seq.to(device), stat.to(device), tgt.to(device)
            optimizer.zero_grad()
            logits = model(seq, stat)
            if isinstance(logits, tuple):
                logits = logits[0]
            loss = criterion(logits, tgt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        pr = average_precision_score(data["val"]["y"], predict(model, loaders["val"]))
        print(f"[{model_name} age{age} seed{seed}] Epoch {epoch + 1:02d} | "
              f"Val PR-AUC {pr:.4f} | {(time.time() - t0) / 60:.1f} min")
        if pr > best:
            best, bad = pr, 0
            torch.save(model.state_dict(), path)
        else:
            bad += 1
            if bad >= patience:
                print(f"[{model_name} age{age} seed{seed}] Early stopping")
                break
    model.load_state_dict(torch.load(path, map_location=device))
    return model


def tabular(d):
    s = d["seq"]
    return np.hstack([d["static"], s[:, -1, :], s[:, -3:, :].mean(1), s.mean(1), s.max(1)])


def save_row(row):
    df = pd.read_csv(RESULTS_CSV) if os.path.exists(RESULTS_CSV) else pd.DataFrame()
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(RESULTS_CSV, index=False)


def done(model_name, age, seed):
    if not os.path.exists(RESULTS_CSV):
        return False
    df = pd.read_csv(RESULTS_CSV)
    return ((df["Model"] == model_name) & (df["Age"] == age) & (df["Seed"] == seed)).any()


# ------------------------------------------------------------
# Main loop
# ------------------------------------------------------------
for age in AGES_TO_RUN:
    print("\n" + "=" * 70 + f"\nHISTORY LENGTH: {age} MONTHS\n" + "=" * 70)
    data = load_cohort(age)
    print({s: (data[s]["seq"].shape, int(data[s]["y"].sum())) for s in data})
    loaders = {s: make_loader(data[s], shuffle=False) for s in ["val", "test"]}

    # XGBoost baseline (once per age)
    if not done("XGBoost", age, 0):
        try:
            import xgboost as xgb
        except ImportError:
            os.system("pip install -q xgboost")
            import xgboost as xgb
        clf = xgb.XGBClassifier(n_estimators=1000, learning_rate=0.03, max_depth=5,
                                subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                                eval_metric="aucpr", early_stopping_rounds=50,
                                tree_method="hist", device="cuda", random_state=42)
        clf.fit(tabular(data["train"]), data["train"]["y"],
                eval_set=[(tabular(data["val"]), data["val"]["y"])], verbose=False)
        res = evaluate(data["val"]["y"], clf.predict_proba(tabular(data["val"]))[:, 1],
                       data["test"]["y"], clf.predict_proba(tabular(data["test"]))[:, 1])
        save_row({"Model": "XGBoost", "Age": age, "Seed": 0, **res})
        print(f"XGBoost age{age}: PR-AUC {res['PR_AUC']:.4f} | F1 {res['F1']:.4f}")

    for model_name in MODELS:
        for seed in SEEDS:
            if done(model_name, age, seed):
                print(f"Skipping {model_name} age{age} seed{seed} (already done)")
                continue
            model = train_model(model_name, age, seed, data, loaders)
            res = evaluate(data["val"]["y"], predict(model, loaders["val"]),
                           data["test"]["y"], predict(model, loaders["test"]))
            save_row({"Model": model_name, "Age": age, "Seed": seed, **res})
            print(f"==> {model_name} age{age} seed{seed}: PR-AUC {res['PR_AUC']:.4f} "
                  f"| F1 {res['F1']:.4f} | ROC {res['ROC_AUC']:.4f}")
            del model
            torch.cuda.empty_cache()

    del data, loaders
    gc.collect()


# ------------------------------------------------------------
# Summary table and chart
# ------------------------------------------------------------
import matplotlib.pyplot as plt

res = pd.read_csv(RESULTS_CSV)
cols = ["Precision", "Recall", "F1", "ROC_AUC", "PR_AUC", "Lift"]
summary = res.groupby(["Model", "Age"])[cols].agg(["mean", "std"]).round(4)
print("\n" + "=" * 70 + "\nHISTORY-LENGTH STUDY - TEST SET (mean, std over seeds)\n" + "=" * 70)
print(summary.to_string())
summary.to_csv(f"{STUDY_DIR}/summary_by_model_and_age.csv")

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for model_name, g in res.groupby("Model"):
    m = g.groupby("Age")[["PR_AUC", "F1"]].agg(["mean", "std"])
    for ax, metric in zip(axes, ["PR_AUC", "F1"]):
        ax.errorbar(m.index, m[(metric, "mean")], yerr=m[(metric, "std")].fillna(0),
                    marker="o", capsize=4, label=model_name)
for ax, metric in zip(axes, ["PR-AUC", "F1"]):
    ax.set_xlabel("Months of repayment history")
    ax.set_ylabel(f"{metric} (test set)")
    ax.set_xticks([12, 24, 36, 48])
    ax.set_title(f"{metric} vs history length (2015Q1)")
    ax.legend()
plt.tight_layout()
plt.savefig(f"{STUDY_DIR}/history_length_study.png", dpi=150)
plt.show()
print(f"\nSaved in {STUDY_DIR}/")
