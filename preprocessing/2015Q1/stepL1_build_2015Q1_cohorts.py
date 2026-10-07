# ============================================================
# STEP L1 - BUILD 2015Q1 COHORTS: 12, 24, 36, 48 MONTHS
# ------------------------------------------------------------
# Combines your existing step2 (temporal flags), step3 (target),
# J4 (sequences), J5 (enhanced features), J6 (split + normalise)
# and J7 (zip) into one script, using EXACTLY the same rules.
#
# Input : Data/2015Q1_CLEANED_LoanLevel(static data).csv
#         Data/2015Q1_CLEANED_Temporal.csv      (from step 0)
# Output: Data/fm2015_age12/ ... Data/fm2015_age48/ + .zip files
#
# Each folder has the same file names as your J24 data
# (X_sequence_train.npy, X_static_train.npy, y_train.npy, ...)
# plus X_cat_*.npy (one-hot categorical features for experiment B).
#
# Run from the project folder:
#     python stepL1_build_2015Q1_cohorts.py
# ============================================================

import os
import gc
import time
import pickle
import shutil
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

QUARTER = "2015Q1"
STATIC_FILE = f"Data/{QUARTER}_CLEANED_LoanLevel(static data).csv"
TEMPORAL_FILE = f"Data/{QUARTER}_CLEANED_Temporal.csv"
OUT_ROOT = "Data"
AGES = [12, 24, 36, 48]
HORIZON = 6          # months ahead for the label
SEED = 42

TEMPORAL_FEATURES = [
    "current_interest_rate", "current_actual_upb", "delinquency_status_numeric",
    "remaining_months_to_maturity", "is_30plus_delinquent",
    "is_60plus_delinquent", "is_90plus_delinquent",
]
ENHANCED_FEATURES = [
    "delinquency_change", "delinquency_worsening", "cumulative_max_delinquency",
    "consecutive_30plus_months", "recent_3m_30plus_count", "monthly_upb_change",
]
STATIC_FEATURES = [
    "borrower_credit_score_at_origination", "co_borrower_credit_score_at_origination",
    "dti", "original_ltv", "original_cltv", "original_upb", "original_interest_rate",
    "number_of_borrowers", "mortgage_insurance_percentage",
    "has_co_borrower", "co_borrower_score_missing", "mortgage_insurance_missing",
]
CATEGORICAL_FEATURES = [
    "loan_purpose", "occupancy", "property_type", "channel",
    "first_time_home_buyer", "property_state",
]

start = time.time()


def elapsed():
    return f"{(time.time() - start) / 60:5.1f} min"


# ============================================================
# 1. STATIC DATA (same derived columns as your step2)
# ============================================================
print("=" * 60 + "\n1. LOADING STATIC DATA\n" + "=" * 60)
static = pd.read_csv(STATIC_FILE, low_memory=False,
                     dtype={"loan_identifier": "int64"})
static = static.drop_duplicates("loan_identifier", keep="first")

for col in STATIC_FEATURES[:9]:
    static[col] = pd.to_numeric(static[col], errors="coerce")

static["has_co_borrower"] = (static["number_of_borrowers"] > 1).astype(int)
static["co_borrower_score_missing"] = static["co_borrower_credit_score_at_origination"].isna().astype(int)
static["mortgage_insurance_missing"] = static["mortgage_insurance_percentage"].isna().astype(int)

# Categorical one-hot (categories come from the data itself, no labels used)
cat = static[["loan_identifier"] + CATEGORICAL_FEATURES].copy()
for col in CATEGORICAL_FEATURES:
    cat[col] = cat[col].astype("string").str.strip().fillna("Missing")
cat_onehot = pd.get_dummies(cat[CATEGORICAL_FEATURES], prefix=CATEGORICAL_FEATURES,
                            dtype=np.float32)
CAT_NAMES = list(cat_onehot.columns)
cat_onehot.index = cat["loan_identifier"].to_numpy()
static = static.set_index("loan_identifier")
print(f"Static loans: {len(static):,} | categorical one-hot columns: {len(CAT_NAMES)} | {elapsed()}")


# ============================================================
# 2. TEMPORAL DATA + FLAGS (same as your step2_temporal_cleaning)
# ============================================================
print("\n" + "=" * 60 + "\n2. LOADING TEMPORAL DATA\n" + "=" * 60)
usecols = ["loan_identifier", "monthly_reporting_period", "loan_age",
           "current_interest_rate", "current_actual_upb",
           "remaining_months_to_maturity", "current_loan_delinquency_status"]
parts = []
for i, ch in enumerate(pd.read_csv(TEMPORAL_FILE, usecols=usecols, chunksize=2_000_000,
                                   dtype={"loan_identifier": "int64",
                                          "current_loan_delinquency_status": "string"}),
                       start=1):
    period = pd.to_datetime(ch["monthly_reporting_period"], errors="coerce")
    out = pd.DataFrame({
        "loan_identifier": ch["loan_identifier"].to_numpy(),
        "month_index": (period.dt.year * 12 + period.dt.month).astype("float64").to_numpy(),
        "loan_age": pd.to_numeric(ch["loan_age"], errors="coerce").astype("float32"),
        "current_interest_rate": pd.to_numeric(ch["current_interest_rate"], errors="coerce").astype("float32"),
        "current_actual_upb": pd.to_numeric(ch["current_actual_upb"], errors="coerce").astype("float32"),
        "remaining_months_to_maturity": pd.to_numeric(ch["remaining_months_to_maturity"], errors="coerce").astype("float32"),
        "delinquency_status_numeric": pd.to_numeric(ch["current_loan_delinquency_status"], errors="coerce").astype("float32"),
    })
    parts.append(out)
    print(f"  chunk {i:2d} loaded | {elapsed()}")
df = pd.concat(parts, ignore_index=True)
del parts
gc.collect()

d = df["delinquency_status_numeric"]
df["is_30plus_delinquent"] = (d >= 1).astype(np.float32)
df["is_60plus_delinquent"] = (d >= 2).astype(np.float32)
df["is_90plus_delinquent"] = (d >= 3).astype(np.float32)

df = df.dropna(subset=["month_index"])
df = df.sort_values(["loan_identifier", "month_index"]).drop_duplicates(
    ["loan_identifier", "month_index"], keep="last").reset_index(drop=True)
print(f"Monthly rows: {len(df):,} | loans: {df['loan_identifier'].nunique():,} | {elapsed()}")


# ============================================================
# 3. TARGET (same logic as your step3_target_metric.py)
#    target = first 60+ delinquency in the 6 months after t
#    eligible = loan observed until at least t + 6 months
#    not already risky = delinquency < 60 days at t
# ============================================================
print("\n" + "=" * 60 + "\n3. CREATING 6-MONTH TARGET\n" + "=" * 60)
g = df.groupby("loan_identifier", sort=False)
risk_month = df["month_index"].where(df["delinquency_status_numeric"] >= 2)
next_risk = risk_month.groupby(df["loan_identifier"], sort=False).shift(-1)
next_risk = next_risk.groupby(df["loan_identifier"], sort=False).bfill()
last_month = g["month_index"].transform("max")
window_end = df["month_index"] + HORIZON

df["target_risk_6m"] = (next_risk.notna() & (next_risk <= window_end)).astype(np.int8)
df["eligible"] = ((last_month >= window_end) & (df["delinquency_status_numeric"] < 2)).astype(np.int8)
del risk_month, next_risk, last_month, window_end, g
gc.collect()
print(f"Target created | {elapsed()}")


# ============================================================
# 4. ENHANCED FEATURES (same as your stepJ5)
# ============================================================
def enhance(X):
    n, L, _ = X.shape
    delinq, upb, is30 = X[:, :, 2], X[:, :, 1], X[:, :, 4]
    change = np.zeros((n, L), np.float32)
    change[:, 1:] = delinq[:, 1:] - delinq[:, :-1]
    worsening = (change > 0).astype(np.float32)
    cum_max = np.maximum.accumulate(delinq, axis=1).astype(np.float32)
    consec = np.zeros((n, L), np.float32)
    recent3 = np.zeros((n, L), np.float32)
    for t in range(L):
        consec[:, t] = (is30[:, t] > 0) * ((consec[:, t - 1] + 1) if t else 1)
        recent3[:, t] = is30[:, max(0, t - 2):t + 1].sum(axis=1)
    upb_change = np.zeros((n, L), np.float32)
    upb_change[:, 1:] = upb[:, 1:] - upb[:, :-1]
    new = np.stack([change, worsening, cum_max, consec, recent3, upb_change], axis=2)
    return np.concatenate([X, new], axis=2).astype(np.float32)


# ============================================================
# 5. SPLIT + NORMALISE (same as your stepJ6)
# ============================================================
def impute_and_scale(train, others, axes):
    train[~np.isfinite(train)] = np.nan
    for o in others:
        o[~np.isfinite(o)] = np.nan
    med = np.nanmedian(train, axis=axes).astype(np.float32)
    med = np.where(np.isfinite(med), med, 0.0).astype(np.float32)
    for arr in [train] + others:
        mask = np.isnan(arr)
        arr[mask] = np.broadcast_to(med, arr.shape)[mask]
    mean = train.mean(axis=axes).astype(np.float32)
    std = train.std(axis=axes).astype(np.float32)
    std[std < 1e-8] = 1.0
    for arr in [train] + others:
        arr -= mean
        arr /= std
    return med, mean, std


# ============================================================
# 6. BUILD EACH COHORT
# ============================================================
summary = []
for age in AGES:
    print("\n" + "=" * 60 + f"\n6. BUILDING AGE-{age} COHORT\n" + "=" * 60)

    # Observation rows: loan age == age, eligible, not already 60+
    obs = df[(df["loan_age"] == age) & (df["eligible"] == 1)][["loan_identifier", "target_risk_6m"]]
    obs = obs.drop_duplicates("loan_identifier", keep="last")
    obs = obs[obs["loan_identifier"].isin(static.index)]

    # History: loan ages 1..age, complete (no missing months)
    hist = df[(df["loan_age"] >= 1) & (df["loan_age"] <= age)
              & df["loan_identifier"].isin(set(obs["loan_identifier"]))]
    hist = hist.sort_values(["loan_identifier", "loan_age"]).drop_duplicates(
        ["loan_identifier", "loan_age"], keep="last")
    counts = hist.groupby("loan_identifier")["loan_age"].nunique()
    complete = counts[counts == age].index
    hist = hist[hist["loan_identifier"].isin(set(complete))]
    ids = hist["loan_identifier"].drop_duplicates().to_numpy()

    obs = obs.set_index("loan_identifier").loc[ids]
    y = obs["target_risk_6m"].to_numpy(np.int8)
    X_seq = hist[TEMPORAL_FEATURES].to_numpy(np.float32).reshape(len(ids), age, len(TEMPORAL_FEATURES))
    X_seq = enhance(X_seq)
    X_static = static.loc[ids, STATIC_FEATURES].to_numpy(np.float32)
    X_cat = cat_onehot.loc[ids, CAT_NAMES].to_numpy(np.float32)
    del hist, obs
    gc.collect()

    print(f"Loans: {len(ids):,} | risky: {int(y.sum()):,} ({100 * y.mean():.2f}%) "
          f"| sequence {X_seq.shape} | {elapsed()}")

    # Stratified 70 / 15 / 15 split, seed 42 (same as J6)
    idx = np.arange(len(y))
    tr, tmp = train_test_split(idx, test_size=0.30, random_state=SEED, stratify=y)
    va, te = train_test_split(tmp, test_size=0.50, random_state=SEED, stratify=y[tmp])

    seq = {k: X_seq[v].copy() for k, v in [("train", tr), ("val", va), ("test", te)]}
    sta = {k: X_static[v].copy() for k, v in [("train", tr), ("val", va), ("test", te)]}
    del X_seq, X_static
    gc.collect()

    t_stats = impute_and_scale(seq["train"], [seq["val"], seq["test"]], axes=(0, 1))
    s_stats = impute_and_scale(sta["train"], [sta["val"], sta["test"]], axes=0)

    folder = os.path.join(OUT_ROOT, f"fm2015_age{age}")
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)
    for split, ix in [("train", tr), ("val", va), ("test", te)]:
        np.save(f"{folder}/X_sequence_{split}.npy", seq[split])
        np.save(f"{folder}/X_static_{split}.npy", sta[split])
        np.save(f"{folder}/X_cat_{split}.npy", X_cat[ix])
        np.save(f"{folder}/y_{split}.npy", y[ix])
        np.save(f"{folder}/loan_ids_{split}.npy", ids[ix])
    with open(f"{folder}/feature_info.pkl", "wb") as f:
        pickle.dump({"temporal_features": TEMPORAL_FEATURES + ENHANCED_FEATURES,
                     "static_features": STATIC_FEATURES,
                     "categorical_features": CAT_NAMES,
                     "temporal_median_mean_std": t_stats,
                     "static_median_mean_std": s_stats,
                     "quarter": QUARTER, "observation_age": age,
                     "horizon_months": HORIZON, "covid_cutoff": "2020-02"}, f)

    shutil.make_archive(folder, "zip", folder)
    size_mb = os.path.getsize(folder + ".zip") / 1024 ** 2

    row = {"Cohort": f"age{age}", "Loans": len(ids), "Risky": int(y.sum()),
           "Risky %": round(100 * y.mean(), 2),
           "Train": len(tr), "Train risky": int(y[tr].sum()),
           "Val": len(va), "Val risky": int(y[va].sum()),
           "Test": len(te), "Test risky": int(y[te].sum()),
           "Zip MB": round(size_mb, 1)}
    summary.append(row)
    print(f"Saved {folder}.zip ({size_mb:.0f} MB) | {elapsed()}")
    del seq, sta, X_cat, y, ids
    gc.collect()


# ============================================================
# 7. SUMMARY
# ============================================================
summary_df = pd.DataFrame(summary)
summary_df.to_csv(os.path.join(OUT_ROOT, "fm2015_cohort_summary.csv"), index=False)
print("\n" + "=" * 60 + "\nALL COHORTS COMPLETE\n" + "=" * 60)
print(summary_df.to_string(index=False))
print(f"\nTotal time: {elapsed()}")
print("Upload the fm2015_age*.zip files from the Data folder to Google Drive "
      "(My Drive/SCER_Mamba/).")
