# ============================================================
# SIGNIFICANCE TEST: SCER-MAMBA vs LSTM (2015Q1, all lengths)
# Paired bootstrap on the test set, 3-seed ensembles, no retraining
# ============================================================
import os, zipfile, numpy as np, pandas as pd, torch, torch.nn as nn
from transformers import MambaConfig, MambaModel
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import TensorDataset, DataLoader
from google.colab import drive

drive.mount("/content/drive")
DRIVE_DIR = "/content/drive/MyDrive/SCER_Mamba"
STUDY_DIR = f"{DRIVE_DIR}/history_length_2015Q1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AGES, SEEDS, N_BOOT = [12, 24, 36, 48], [42, 123, 2026], 2000


class SCERMamba(nn.Module):
    def __init__(self, temporal_features=13, static_features=12, hidden_size=64,
                 num_layers=1, dropout=0.20, state_size=16):
        super().__init__()
        self.static_encoder = nn.Sequential(nn.Linear(static_features, 32), nn.ReLU(), nn.Dropout(dropout),
                                            nn.Linear(32, hidden_size), nn.ReLU())
        self.temporal_projection = nn.Linear(temporal_features, hidden_size)
        self.mamba = MambaModel(MambaConfig(vocab_size=2, hidden_size=hidden_size, state_size=state_size,
                                            num_hidden_layers=num_layers, expand=2, conv_kernel=4, use_cache=False))
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
    def __init__(self, temporal_features=13, static_features=12, hidden_size=64, num_layers=2, dropout=0.20):
        super().__init__()
        self.lstm = nn.LSTM(temporal_features, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.static_encoder = nn.Sequential(nn.Linear(static_features, 32), nn.ReLU(), nn.Dropout(dropout),
                                            nn.Linear(32, hidden_size), nn.ReLU())
        self.classifier = nn.Sequential(nn.Linear(hidden_size * 2, 64), nn.ReLU(), nn.Dropout(dropout),
                                        nn.Linear(64, 1))

    def forward(self, sequence, static):
        _, (h, _) = self.lstm(sequence)
        return self.classifier(torch.cat([h[-1], self.static_encoder(static)], 1)).squeeze(-1)


def load_test(age):
    folder = f"/content/fm2015_age{age}"
    if not os.path.exists(folder):
        with zipfile.ZipFile(f"{DRIVE_DIR}/fm2015_age{age}.zip") as z:
            z.extractall(folder)
    seq = np.load(f"{folder}/X_sequence_test.npy").astype(np.float32)
    sta = np.load(f"{folder}/X_static_test.npy").astype(np.float32)
    y = np.load(f"{folder}/y_test.npy").astype(int)
    return DataLoader(TensorDataset(torch.from_numpy(seq), torch.from_numpy(sta)), batch_size=2048), y


def ensemble_probs(name, age, loader):
    probs = []
    for seed in SEEDS:
        model = (SCERMamba() if name == "SCER-Mamba" else LSTMBaseline()).to(device)
        model.load_state_dict(torch.load(f"{STUDY_DIR}/{name}_age{age}_seed{seed}.pt", map_location=device))
        model.eval()
        out = []
        with torch.no_grad():
            for seq, sta in loader:
                logits = model(seq.to(device), sta.to(device))
                logits = logits[0] if isinstance(logits, tuple) else logits
                out.append(torch.sigmoid(logits).float().cpu().numpy())
        probs.append(np.concatenate(out))
        del model
        torch.cuda.empty_cache()
    return np.mean(probs, axis=0)


rows = []
rng = np.random.default_rng(42)
for age in AGES:
    print(f"Age {age}: predicting...")
    loader, y = load_test(age)
    p_scer, p_lstm = ensemble_probs("SCER-Mamba", age, loader), ensemble_probs("LSTM", age, loader)
    d_pr, d_roc = [], []
    for _ in range(N_BOOT):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].sum() == 0:
            continue
        d_pr.append(average_precision_score(y[idx], p_scer[idx]) - average_precision_score(y[idx], p_lstm[idx]))
        d_roc.append(roc_auc_score(y[idx], p_scer[idx]) - roc_auc_score(y[idx], p_lstm[idx]))
    for metric, diffs, fn in [("PR-AUC", np.array(d_pr), average_precision_score),
                              ("ROC-AUC", np.array(d_roc), roc_auc_score)]:
        p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
        rows.append({"Age": age, "Metric": metric,
                     "SCER-Mamba": round(fn(y, p_scer), 4), "LSTM": round(fn(y, p_lstm), 4),
                     "Diff (SCER-LSTM)": round(fn(y, p_scer) - fn(y, p_lstm), 4),
                     "95% CI": f"[{np.percentile(diffs, 2.5):+.4f}, {np.percentile(diffs, 97.5):+.4f}]",
                     "p-value": round(min(p, 1.0), 4),
                     "Significant (p<0.05)": "Yes" if p < 0.05 else "No"})

result = pd.DataFrame(rows)
print("\n" + "=" * 90 + "\nSIGNIFICANCE TEST: SCER-MAMBA vs LSTM (3-seed ensembles, 2015Q1 test set)\n" + "=" * 90)
print(result.to_string(index=False))
result.to_csv(f"{STUDY_DIR}/significance_test.csv", index=False)
print(f"\nSaved: {STUDY_DIR}/significance_test.csv")
