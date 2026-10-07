# ============================================================
# EXPERIMENT C: MULTI-TASK LEARNING (2015Q1, 48 months + categorical)
# ============================================================
import os, time, random, zipfile, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from transformers import MambaConfig, MambaModel
from sklearn.metrics import (average_precision_score, roc_auc_score, precision_recall_curve,
                             accuracy_score, precision_score, recall_score, f1_score)
from google.colab import drive

drive.mount("/content/drive")
DRIVE_DIR = "/content/drive/MyDrive/SCER_Mamba"
B_DIR = f"{DRIVE_DIR}/experiment_B_categorical"
C_DIR = f"{DRIVE_DIR}/experiment_C_multitask"
os.makedirs(C_DIR, exist_ok=True)
C_CSV = f"{C_DIR}/results_per_run.csv"
AGE, SEEDS, LAMBDA = 48, [42, 123, 2026], 0.5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
assert device.type == "cuda", "No GPU - Runtime -> Change runtime type -> T4 GPU"

folder = f"/content/fm2015_age{AGE}"
if not os.path.exists(folder):
    with zipfile.ZipFile(f"{DRIVE_DIR}/fm2015_age{AGE}.zip") as z:
        z.extractall(folder)
data = {}
for s in ["train", "val", "test"]:
    seq = np.load(f"{folder}/X_sequence_{s}.npy").astype(np.float32)
    data[s] = {"seq": seq,
               "static": np.hstack([np.load(f"{folder}/X_static_{s}.npy"),
                                    np.load(f"{folder}/X_cat_{s}.npy")]).astype(np.float32),
               "y": np.load(f"{folder}/y_{s}.npy").astype(int),
               "aux": (seq[:, 1:, 4] > 0).astype(np.float32)}
N_STATIC = data["train"]["static"].shape[1]
print(f"Static inputs: {N_STATIC} | extra-task targets per loan: {data['train']['aux'].shape[1]}")

eval_loaders = {s: DataLoader(TensorDataset(torch.from_numpy(data[s]["seq"]), torch.from_numpy(data[s]["static"])),
                              batch_size=2048) for s in ["val", "test"]}
train_ds = TensorDataset(torch.from_numpy(data["train"]["seq"]), torch.from_numpy(data["train"]["static"]),
                         torch.tensor(data["train"]["y"], dtype=torch.float32), torch.from_numpy(data["train"]["aux"]))


class SCERMamba(nn.Module):
    def __init__(self, static_features, multitask, temporal_features=13, hidden_size=64, dropout=0.20):
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
        self.aux_head = nn.Linear(hidden_size, 1) if multitask else None

    def forward(self, sequence, static):
        s = self.static_encoder(static)
        out = self.output_norm(self.mamba(inputs_embeds=self.temporal_projection(sequence),
                                          use_cache=False).last_hidden_state)
        att = torch.softmax(self.temporal_attention(out).squeeze(-1)
                            + self.risk_event_attention(sequence[:, :, 4:7]).squeeze(-1), dim=1)
        pooled = torch.sum(out * att.unsqueeze(-1), dim=1)
        main = self.classifier(torch.cat([pooled, out.mean(1), out[:, -1, :], s], 1)).squeeze(-1)
        aux = self.aux_head(out[:, :-1, :]).squeeze(-1) if self.aux_head is not None else None
        return main, aux


class LSTMModel(nn.Module):
    def __init__(self, static_features, multitask, temporal_features=13, hidden_size=64, dropout=0.20):
        super().__init__()
        self.lstm = nn.LSTM(temporal_features, hidden_size, 2, batch_first=True, dropout=dropout)
        self.static_encoder = nn.Sequential(nn.Linear(static_features, 32), nn.ReLU(), nn.Dropout(dropout),
                                            nn.Linear(32, hidden_size), nn.ReLU())
        self.classifier = nn.Sequential(nn.Linear(hidden_size * 2, 64), nn.ReLU(), nn.Dropout(dropout),
                                        nn.Linear(64, 1))
        self.aux_head = nn.Linear(hidden_size, 1) if multitask else None

    def forward(self, sequence, static):
        out, (h, _) = self.lstm(sequence)
        main = self.classifier(torch.cat([h[-1], self.static_encoder(static)], 1)).squeeze(-1)
        aux = self.aux_head(out[:, :-1, :]).squeeze(-1) if self.aux_head is not None else None
        return main, aux


def build(name, multitask=True):
    cls = SCERMamba if name == "SCER-Mamba" else LSTMModel
    return cls(N_STATIC, multitask).to(device)


def focal(logits, t):
    bce = F.binary_cross_entropy_with_logits(logits, t, reduction="none")
    p = torch.sigmoid(logits)
    return (((1 - (p * t + (1 - p) * (1 - t))) ** 2.0) * bce).mean()


def predict(model, loader):
    model.eval()
    out = []
    with torch.no_grad():
        for seq, sta in loader:
            out.append(torch.sigmoid(model(seq.to(device), sta.to(device))[0]).float().cpu().numpy())
    return np.concatenate(out)


def evaluate(vp, tp):
    yv, yt = data["val"]["y"], data["test"]["y"]
    p, r, t = precision_recall_curve(yv, vp)
    thr = float(t[int(np.argmax(2 * p[:-1] * r[:-1] / (p[:-1] + r[:-1] + 1e-10)))])
    pred = (tp >= thr).astype(int)
    return {"Accuracy": accuracy_score(yt, pred), "Precision": precision_score(yt, pred, zero_division=0),
            "Recall": recall_score(yt, pred, zero_division=0), "F1": f1_score(yt, pred, zero_division=0),
            "ROC_AUC": roc_auc_score(yt, tp), "PR_AUC": average_precision_score(yt, tp),
            "Val_PR_AUC": average_precision_score(yv, vp)}


def train(name, seed, max_epochs=20, patience=4):
    final = f"{C_DIR}/{name}_MT_seed{seed}.pt"
    tmp = final + ".partial"
    model = build(name)
    if os.path.exists(final):
        model.load_state_dict(torch.load(final, map_location=device))
        return model
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    model = build(name)
    loader = DataLoader(train_ds, batch_size=256, shuffle=True,
                        generator=torch.Generator().manual_seed(seed), num_workers=2, pin_memory=True)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    best, bad = -1, 0
    for epoch in range(max_epochs):
        t0 = time.time()
        model.train()
        for seq, sta, y, aux in loader:
            seq, sta, y, aux = seq.to(device), sta.to(device), y.to(device), aux.to(device)
            opt.zero_grad()
            main, aux_logits = model(seq, sta)
            loss = focal(main, y) + LAMBDA * F.binary_cross_entropy_with_logits(aux_logits, aux)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        pr = average_precision_score(data["val"]["y"], predict(model, eval_loaders["val"]))
        print(f"[{name}+MT seed{seed}] Epoch {epoch + 1:02d} | Val PR-AUC {pr:.4f} | {(time.time() - t0) / 60:.1f} min")
        if pr > best:
            best, bad = pr, 0
            torch.save(model.state_dict(), tmp)
        else:
            bad += 1
            if bad >= patience:
                break
    os.replace(tmp, final)
    model.load_state_dict(torch.load(final, map_location=device))
    return model


def done(name, seed):
    return os.path.exists(C_CSV) and ((pd.read_csv(C_CSV)[["Model", "Seed"]] == [name, seed]).all(axis=1)).any()


probs = {"SCER-Mamba": [], "LSTM": []}
for name in ["SCER-Mamba", "LSTM"]:
    for seed in SEEDS:
        model = train(name, seed)
        vp, tp = predict(model, eval_loaders["val"]), predict(model, eval_loaders["test"])
        probs[name].append(tp)
        if not done(name, seed):
            res = evaluate(vp, tp)
            pd.DataFrame([{"Model": name, "Seed": seed, **res}]).to_csv(
                C_CSV, mode="a", header=not os.path.exists(C_CSV), index=False)
            print(f"==> {name}+MT seed{seed}: PR-AUC {res['PR_AUC']:.4f} | F1 {res['F1']:.4f} | ROC {res['ROC_AUC']:.4f}")
        del model
        torch.cuda.empty_cache()

cols = ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC", "PR_AUC"]
mt = pd.read_csv(C_CSV).assign(Version="+ Multi-task")
b = pd.read_csv(f"{B_DIR}/results_per_run.csv").assign(Version="Experiment B (no multi-task)")
summary = pd.concat([mt, b], ignore_index=True).groupby(["Model", "Version"])[cols].agg(["mean", "std"]).round(4)
print("\n" + "=" * 80 + "\nEXPERIMENT C - 48 MONTHS + CATEGORICAL, TEST SET (mean, std over seeds)\n" + "=" * 80)
print(summary.to_string())
summary.to_csv(f"{C_DIR}/summary_multitask.csv")

y = data["test"]["y"]
ps, pl_mt = np.mean(probs["SCER-Mamba"], 0), np.mean(probs["LSTM"], 0)
pl_b = []
for seed in SEEDS:
    m = LSTMModel(N_STATIC, multitask=False).to(device)
    m.load_state_dict(torch.load(f"{B_DIR}/LSTM_cat_age48_seed{seed}.pt", map_location=device))
    pl_b.append(predict(m, eval_loaders["test"]))
pl_b = np.mean(pl_b, 0)


def boot(a, b, n=2000):
    rng, d = np.random.default_rng(42), []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        d.append(average_precision_score(y[i], a[i]) - average_precision_score(y[i], b[i]))
    d = np.array(d)
    p = min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))
    diff = average_precision_score(y, a) - average_precision_score(y, b)
    return f"diff {diff:+.4f} | 95% CI [{np.percentile(d, 2.5):+.4f}, {np.percentile(d, 97.5):+.4f}] | p = {p:.3f}"


print(f"\nSCER-Mamba+MT PR-AUC (3-seed ensemble): {average_precision_score(y, ps):.4f}")
print(f"LSTM+MT       PR-AUC (3-seed ensemble): {average_precision_score(y, pl_mt):.4f}")
print(f"LSTM (Exp B)  PR-AUC (3-seed ensemble): {average_precision_score(y, pl_b):.4f}")
print("SIGNIFICANCE SCER-Mamba+MT vs LSTM+MT     :", boot(ps, pl_mt))
print("SIGNIFICANCE SCER-Mamba+MT vs LSTM (Exp B):", boot(ps, pl_b))
print(f"\nSaved in {C_DIR}/")
