# ============================================================
# FINAL FIGURES - 2015Q1 STUDY (history length, significance,
# experiment B, attention at 48 months). Saved to Google Drive.
# ============================================================
import os, zipfile, numpy as np, pandas as pd, torch, torch.nn as nn, matplotlib.pyplot as plt
from transformers import MambaConfig, MambaModel
from sklearn.metrics import average_precision_score
from torch.utils.data import TensorDataset, DataLoader
from google.colab import drive

drive.mount("/content/drive")
DRIVE_DIR = "/content/drive/MyDrive/SCER_Mamba"
STUDY_DIR = f"{DRIVE_DIR}/history_length_2015Q1"
B_DIR = f"{DRIVE_DIR}/experiment_B_categorical"
FIG_DIR = f"{DRIVE_DIR}/figures_2015Q1"
os.makedirs(FIG_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
plt.rcParams.update({"figure.dpi": 130, "font.size": 10})
COLORS = {"SCER-Mamba": "#1f77b4", "LSTM": "#ff7f0e", "XGBoost": "#2ca02c"}
METRICS = ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC", "PR_AUC"]

# ---------------- Fig 7: history length, all metrics ----------------
res = pd.read_csv(f"{STUDY_DIR}/results_per_run.csv")
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, m in zip(axes.flat, METRICS):
    for model, g in res.groupby("Model"):
        s = g.groupby("Age")[m].agg(["mean", "std"])
        ax.errorbar(s.index, s["mean"], yerr=s["std"].fillna(0), marker="o", capsize=4,
                    label=model, color=COLORS[model])
    ax.set_title(m.replace("_", "-"))
    ax.set_xticks([12, 24, 36, 48])
    ax.set_xlabel("Months of repayment history")
    ax.grid(alpha=0.3)
axes[0, 0].legend()
fig.suptitle("History-length study - 2015Q1 test set (mean ± SD over 3 seeds)")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig7_history_length_all_metrics.png")
plt.show()
table = res.groupby(["Age", "Model"])[METRICS].mean().round(4)
table.to_csv(f"{FIG_DIR}/table_history_length_means.csv")
print(table.to_string())

# ---------------- Fig 8: significance (SCER-Mamba - LSTM) ----------------
sig = pd.read_csv(f"{STUDY_DIR}/significance_test.csv")
sig["lo"] = sig["95% CI"].str.extract(r"\[([-+0-9.]+),")[0].astype(float)
sig["hi"] = sig["95% CI"].str.extract(r",\s*([-+0-9.]+)\]")[0].astype(float)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, m in zip(axes, ["PR-AUC", "ROC-AUC"]):
    d = sig[sig["Metric"] == m]
    diff = d["Diff (SCER-LSTM)"]
    ax.errorbar(d["Age"], diff, yerr=[diff - d["lo"], d["hi"] - diff], fmt="o", capsize=6, color="#1f77b4")
    ax.axhline(0, color="grey", linestyle="--")
    ax.set_xticks([12, 24, 36, 48])
    ax.set_xlabel("Months of repayment history")
    ax.set_ylabel(f"{m}: SCER-Mamba minus LSTM")
    ax.set_title(f"{m} difference with 95% CI (all p > 0.05)")
    ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig8_significance_scer_vs_lstm.png")
plt.show()

# ---------------- Fig 9: experiment B ----------------
b = pd.read_csv(f"{B_DIR}/results_per_run.csv").assign(Features="+ Categorical")
both = pd.concat([res[res["Age"] == 48].assign(Features="Numeric only"), b], ignore_index=True)
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for ax, m in zip(axes, ["PR_AUC", "Precision", "F1"]):
    s = both.groupby(["Model", "Features"])[m].mean().unstack()[["Numeric only", "+ Categorical"]]
    s.plot.bar(ax=ax, rot=0, color=["#9ecae1", "#08519c"])
    for c in ax.containers:
        ax.bar_label(c, fmt="%.3f", fontsize=8)
    ax.set_title(f"{m.replace('_', '-')} - 48 months")
    ax.set_xlabel("")
    ax.set_ylim(0, s.values.max() * 1.2)
plt.suptitle("Experiment B: effect of categorical borrower features")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig9_experiment_B_categorical.png")
plt.show()


# ---------------- models for experiment B significance + attention ----------------
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


folder = "/content/fm2015_age48"
if not os.path.exists(folder):
    with zipfile.ZipFile(f"{DRIVE_DIR}/fm2015_age48.zip") as z:
        z.extractall(folder)
seq = np.load(f"{folder}/X_sequence_test.npy").astype(np.float32)
sta = np.hstack([np.load(f"{folder}/X_static_test.npy"),
                 np.load(f"{folder}/X_cat_test.npy")]).astype(np.float32)
y = np.load(f"{folder}/y_test.npy").astype(int)
loader = DataLoader(TensorDataset(torch.from_numpy(seq), torch.from_numpy(sta)), batch_size=2048)


def run(model):
    model.eval()
    probs, atts = [], []
    with torch.no_grad():
        for s_, t_ in loader:
            out = model(s_.to(device), t_.to(device))
            if isinstance(out, tuple):
                atts.append(out[1].float().cpu().numpy())
                out = out[0]
            probs.append(torch.sigmoid(out).float().cpu().numpy())
    return np.concatenate(probs), (np.concatenate(atts) if atts else None)


# ---------------- Experiment B significance ----------------
ens = {}
for name in ["SCER-Mamba", "LSTM"]:
    ps = []
    for seed in [42, 123, 2026]:
        model = (SCERMamba(sta.shape[1]) if name == "SCER-Mamba" else LSTMBaseline(sta.shape[1])).to(device)
        model.load_state_dict(torch.load(f"{B_DIR}/{name}_cat_age48_seed{seed}.pt", map_location=device))
        ps.append(run(model)[0])
        del model
        torch.cuda.empty_cache()
    ens[name] = np.mean(ps, axis=0)
rng, diffs = np.random.default_rng(42), []
for _ in range(2000):
    i = rng.integers(0, len(y), len(y))
    diffs.append(average_precision_score(y[i], ens["SCER-Mamba"][i]) - average_precision_score(y[i], ens["LSTM"][i]))
diffs = np.array(diffs)
p = min(1.0, 2 * min((diffs <= 0).mean(), (diffs >= 0).mean()))
pa, pb = average_precision_score(y, ens["SCER-Mamba"]), average_precision_score(y, ens["LSTM"])
line = (f"EXPERIMENT B SIGNIFICANCE (48 months, + categorical, 3-seed ensembles): SCER-Mamba PR-AUC {pa:.4f} "
        f"vs LSTM {pb:.4f} | diff {pa - pb:+.4f} | 95% CI [{np.percentile(diffs, 2.5):+.4f}, "
        f"{np.percentile(diffs, 97.5):+.4f}] | p = {p:.3f}")
print("\n" + line)
open(f"{FIG_DIR}/experiment_B_significance.txt", "w").write(line)

# ---------------- Fig 10-11: attention at 48 months ----------------
model = SCERMamba(sta.shape[1]).to(device)
model.load_state_dict(torch.load(f"{B_DIR}/SCER-Mamba_cat_age48_seed42.pt", map_location=device))
prob, att = run(model)
late = (seq[:, :, 4] > 0).astype(int)
months = np.arange(1, att.shape[1] + 1)

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(months, att[y == 1].mean(0), marker="o", label="Risky loans")
ax.plot(months, att[y == 0].mean(0), marker="o", label="Not-risky loans")
ax.set_xlabel("Month of repayment history")
ax.set_ylabel("Average attention weight")
ax.set_title("SCER-Mamba attention across the 48-month history (2015Q1 test set)")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig10_attention_by_month_48m.png")
plt.show()

w_late, w_ok = att[late == 1].mean(), att[late == 0].mean()
print(f"\nAttention on months with a 30+ day late payment : {w_late:.4f}")
print(f"Attention on months without a late payment       : {w_ok:.4f}")
print(f"Ratio: {w_late / w_ok:.2f}x more attention on late-payment months")

idx = np.where(y == 1)[0]
idx = idx[np.argsort(-prob[idx])][:8]
fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
axes[0].imshow(att[idx], aspect="auto", cmap="Reds")
axes[0].set_title("Attention weights (darker = more focus)")
axes[1].imshow(late[idx], aspect="auto", cmap="Greys")
axes[1].set_title("Months 30+ days late (black = late)")
for a in axes:
    a.set_yticks(range(len(idx)))
    a.set_yticklabels([f"Loan {i + 1} (p={prob[k]:.2f})" for i, k in enumerate(idx)], fontsize=8)
axes[1].set_xticks(range(0, len(months), 3))
axes[1].set_xticklabels(months[::3], fontsize=8)
axes[1].set_xlabel("Month of repayment history")
plt.suptitle("SCER-Mamba attention vs late payments - top risky loans (48 months)")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig11_attention_examples_48m.png")
plt.show()

print(f"\nAll figures saved in {FIG_DIR}/")
