# ============================================================
# EXPERIMENT B: + CATEGORICAL FEATURES (2015Q1, 48 months)
# SCER-Mamba vs LSTM vs XGBoost, same settings and seeds
# ============================================================
import os, gc, time, random, zipfile, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from transformers import MambaConfig, MambaModel
from sklearn.metrics import (average_precision_score, roc_auc_score, precision_recall_curve,
                             accuracy_score, precision_score, recall_score, f1_score, confusion_matrix)
from google.colab import drive

drive.mount("/content/drive")
DRIVE_DIR = "/content/drive/MyDrive/SCER_Mamba"
STUDY_DIR = f"{DRIVE_DIR}/history_length_2015Q1"
B_DIR = f"{DRIVE_DIR}/experiment_B_categorical"
os.makedirs(B_DIR, exist_ok=True)
B_CSV = f"{B_DIR}/results_per_run.csv"
AGE, SEEDS = 48, [42, 123, 2026]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
assert device.type == "cuda", "No GPU - Runtime -> Change runtime type -> T4 GPU"

# ---------------- data: static + one-hot categorical ----------------
folder = f"/content/fm2015_age{AGE}"
if not os.path.exists(folder):
    with zipfile.ZipFile(f"{DRIVE_DIR}/fm2015_age{AGE}.zip") as z:
        z.extractall(folder)
data = {}
for s in ["train", "val", "test"]:
    data[s] = {"seq": np.load(f"{folder}/X_sequence_{s}.npy").astype(np.float32),
               "static": np.hstack([np.load(f"{folder}/X_static_{s}.npy"),
                                    np.load(f"{folder}/X_cat_{s}.npy")]).astype(np.float32),
               "y": np.load(f"{folder}/y_{s}.npy").astype(int)}
N_STATIC = data["train"]["static"].shape[1]
print(f"Static inputs: 12 numeric + {N_STATIC - 12} categorical = {N_STATIC}")


def make_loader(d, shuffle, seed=42):
    ds = TensorDataset(torch.from_numpy(d["seq"]), torch.from_numpy(d["static"]),
                       torch.tensor(d["y"], dtype=torch.float32))
    return DataLoader(ds, batch_size=256 if shuffle else 2048, shuffle=shuffle,
                      generator=torch.Generator().manual_seed(seed) if shuffle else None,
                      num_workers=2, pin_memory=True)


loaders = {s: make_loader(data[s], False) for s in ["val", "test"]}


# ---------------- models (same as before, wider static input) ----------------
class FocalLoss(nn.Module):
    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p = torch.sigmoid(logits)
        return (((1 - (p * targets + (1 - p) * (1 - targets))) ** 2.0) * bce).mean()


class SCERMamba(nn.Module):
    def __init__(self, static_features, temporal_features=13, hidden_size=64, dropout=0.20):
        super().__init__()
        self.static_encoder = nn.Sequential(nn.Linear(static_features, 32), nn.ReLU(), nn.Dropout(dropout),
                                            nn.Linear(32, hidden_size), nn.ReLU())
        self.temporal_projection = nn.Linear(temporal_features, hidden_size)
        self.mamba = MambaModel(MambaConfig(vocab_size=2, hidden_size=hidden_size, state_size=16,
                                            num_hidden_layers=1, expand=2, conv_kernel=4, use_cache=False))
        self.output_norm = nn.LayerNorm(hidden_size)
        self.temporal_attention = nn.Linear(hidden_size, 1)
        self.risk_event_attention = nn.Sequential(nn.Linear(3, 16), nn.ReLU(), nn.Linear(16, 1))
        self.classifier = nn.Sequential(nn.Linear(hidden_size * 4, 128), nn.ReLU(), nn.Dropout(dropout),
                                        nn.Linear(128, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 1))

    def forward(self, sequence, static):
        s = self.static_encoder(static)
        out = self.output_norm(self.mamba(inputs_embeds=self.temporal_projection(sequence),
                                          use_cache=False).last_hidden_state)
        att = torch.softmax(self.temporal_attention(out).squeeze(-1)
                            + self.risk_event_attention(sequence[:, :, 4:7]).squeeze(-1), dim=1)
        pooled = torch.sum(out * att.unsqueeze(-1), dim=1)
        return self.classifier(torch.cat([pooled, out.mean(1), out[:, -1, :], s], 1)).squeeze(-1), att


class LSTMBaseline(nn.Module):
    def __init__(self, static_features, temporal_features=13, hidden_size=64, dropout=0.20):
        super().__init__()
        self.lstm = nn.LSTM(temporal_features, hidden_size, 2, batch_first=True, dropout=dropout)
        self.static_encoder = nn.Sequential(nn.Linear(static_features, 32), nn.ReLU(), nn.Dropout(dropout),
                                            nn.Linear(32, hidden_size), nn.ReLU())
        self.classifier = nn.Sequential(nn.Linear(hidden_size * 2, 64), nn.ReLU(), nn.Dropout(dropout),
                                        nn.Linear(64, 1))

    def forward(self, sequence, static):
        _, (h, _) = self.lstm(sequence)
        return self.classifier(torch.cat([h[-1], self.static_encoder(static)], 1)).squeeze(-1)


def predict(model, loader):
    model.eval()
    out = []
    with torch.no_grad():
        for seq, sta, _ in loader:
            lg = model(seq.to(device), sta.to(device))
            out.append(torch.sigmoid(lg[0] if isinstance(lg, tuple) else lg).float().cpu().numpy())
    return np.concatenate(out)


def evaluate(vp, tp):
    yv, yt = data["val"]["y"], data["test"]["y"]
    p, r, t = precision_recall_curve(yv, vp)
    thr = float(t[int(np.argmax(2 * p[:-1] * r[:-1] / (p[:-1] + r[:-1] + 1e-10)))])
    pred = (tp >= thr).astype(int)
    return {"Accuracy": accuracy_score(yt, pred), "Precision": precision_score(yt, pred, zero_division=0),
            "Recall": recall_score(yt, pred, zero_division=0), "F1": f1_score(yt, pred, zero_division=0),
            "ROC_AUC": roc_auc_score(yt, tp), "PR_AUC": average_precision_score(yt, tp)}


def done(name, seed):
    return os.path.exists(B_CSV) and ((pd.read_csv(B_CSV)[["Model", "Seed"]]
                                       == [name, seed]).all(axis=1)).any()


def save_row(row):
    df = pd.read_csv(B_CSV) if os.path.exists(B_CSV) else pd.DataFrame()
    pd.concat([df, pd.DataFrame([row])], ignore_index=True).to_csv(B_CSV, index=False)


def train(name, seed):
    path = f"{B_DIR}/{name}_cat_age{AGE}_seed{seed}.pt"
    build = lambda: (SCERMamba(N_STATIC) if name == "SCER-Mamba" else LSTMBaseline(N_STATIC)).to(device)
    model = build()
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=device))
        return model
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    model = build()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    crit, best, bad = FocalLoss(), -1, 0
    for epoch in range(20):
        t0 = time.time()
        model.train()
        for seq, sta, tgt in make_loader(data["train"], True, seed):
            seq, sta, tgt = seq.to(device), sta.to(device), tgt.to(device)
            opt.zero_grad()
            lg = model(seq, sta)
            loss = crit(lg[0] if isinstance(lg, tuple) else lg, tgt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        pr = average_precision_score(data["val"]["y"], predict(model, loaders["val"]))
        print(f"[{name}+cat seed{seed}] Epoch {epoch + 1:02d} | Val PR-AUC {pr:.4f} | {(time.time() - t0) / 60:.1f} min")
        if pr > best:
            best, bad = pr, 0
            torch.save(model.state_dict(), path)
        else:
            bad += 1
            if bad >= 4:
                break
    model.load_state_dict(torch.load(path, map_location=device))
    return model


# ---------------- XGBoost + categorical ----------------
if not done("XGBoost", 0):
    import xgboost as xgb
    tab = lambda d: np.hstack([d["static"], d["seq"][:, -1], d["seq"][:, -3:].mean(1),
                               d["seq"].mean(1), d["seq"].max(1)])
    clf = xgb.XGBClassifier(n_estimators=1000, learning_rate=0.03, max_depth=5, subsample=0.8,
                            colsample_bytree=0.8, min_child_weight=5, eval_metric="aucpr",
                            early_stopping_rounds=50, tree_method="hist", device="cuda", random_state=42)
    clf.fit(tab(data["train"]), data["train"]["y"], eval_set=[(tab(data["val"]), data["val"]["y"])], verbose=False)
    res = evaluate(clf.predict_proba(tab(data["val"]))[:, 1], clf.predict_proba(tab(data["test"]))[:, 1])
    save_row({"Model": "XGBoost", "Seed": 0, **res})
    print(f"XGBoost+cat: PR-AUC {res['PR_AUC']:.4f} | F1 {res['F1']:.4f}")

# ---------------- SCER-Mamba and LSTM, 3 seeds ----------------
probs = {"SCER-Mamba": [], "LSTM": []}
for name in ["SCER-Mamba", "LSTM"]:
    for seed in SEEDS:
        model = train(name, seed)
        vp, tp = predict(model, loaders["val"]), predict(model, loaders["test"])
        probs[name].append(tp)
        if not done(name, seed):
            res = evaluate(vp, tp)
            save_row({"Model": name, "Seed": seed, **res})
            print(f"==> {name}+cat seed{seed}: PR-AUC {res['PR_AUC']:.4f} | F1 {res['F1']:.4f} | ROC {res['ROC_AUC']:.4f}")
        del model
        torch.cuda.empty_cache()

# ---------------- comparison: without vs with categorical ----------------
cols = ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC", "PR_AUC"]
base = pd.read_csv(f"{STUDY_DIR}/results_per_run.csv")
base = base[base["Age"] == AGE].assign(Features="Numeric only")
withcat = pd.read_csv(B_CSV).assign(Features="+ Categorical")
both = pd.concat([base, withcat], ignore_index=True)
summary = both.groupby(["Model", "Features"])[cols].agg(["mean", "std"]).round(4)
print("\n" + "=" * 80 + f"\nEXPERIMENT B - {AGE} MONTHS, TEST SET (mean, std over seeds)\n" + "=" * 80)
print(summary.to_string())
summary.to_csv(f"{B_DIR}/summary_with_without_categorical.csv")

# ---------------- significance: SCER-Mamba+cat vs LSTM+cat ----------------
y = data["test"]["y"]
ps, pl = np.mean(probs["SCER-Mamba"], 0), np.mean(probs["LSTM"], 0)
rng, diffs = np.random.default_rng(42), []
for _ in range(2000):
    i = rng.integers(0, len(y), len(y))
    diffs.append(average_precision_score(y[i], ps[i]) - average_precision_score(y[i], pl[i]))
diffs = np.array(diffs)
p = min(1.0, 2 * min((diffs <= 0).mean(), (diffs >= 0).mean()))
print(f"\nSIGNIFICANCE (+cat, 3-seed ensembles): SCER-Mamba PR-AUC {average_precision_score(y, ps):.4f} "
      f"vs LSTM {average_precision_score(y, pl):.4f} | diff {average_precision_score(y, ps) - average_precision_score(y, pl):+.4f} "
      f"| 95% CI [{np.percentile(diffs, 2.5):+.4f}, {np.percentile(diffs, 97.5):+.4f}] | p = {p:.3f}")
print(f"\nSaved in {B_DIR}/")
