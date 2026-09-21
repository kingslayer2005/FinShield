"""
data/scripts/preprocess.py
Loads raw CSVs (real Kaggle or synthetic), merges, computes behavioural
features on the FULL time-sorted dataset BEFORE splitting, does a
chronological split, fits encoders on TRAIN only, and saves parquets.

Run from repo root:
    python data/scripts/preprocess.py          # real data
    SMOKE=1 python data/scripts/preprocess.py  # synthetic smoke data

Leakage protections applied (see Phase-1 corrections):
  1. Chronological split — no random shuffle
  2. Behavioural features computed on full dataset BEFORE split so val/test
     rows can use their card's real history from earlier periods, but
     never same-time or later transactions (the function is strictly-past)
  3. Label encoders fitted on TRAIN only — unseen categories map to __unseen__
  4. NaN kept for XGBoost (it handles missingness natively).
     Median imputation + scaling applied only to AE/LSTM inputs (saved
     separately as *_scaled.parquet)
  5. Inner-validation set carved from last 10% of TRAIN period for
     XGBoost / neural-net early stopping
"""

import pandas as pd
import numpy as np
import pickle
import os
import sys
import json

# ---------------------------------------------------------------------------
# Allow imports from repo root
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from utils.config import cfg, is_smoke, get_processed_dir, get_reports_dir
from features.behavioral import compute_behavioral_features
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ---------------------------------------------------------------------------
# Resolve paths from config
# ---------------------------------------------------------------------------
RAW_DIR = cfg["data"]["raw_dir"]           # real or synthetic depending on SMOKE
OUT_DIR = get_processed_dir()              # data/processed or data/synthetic/processed
TRAIN_FRAC = cfg["split"]["train_frac"]    # 0.70
VAL_FRAC = cfg["split"]["val_frac"]        # 0.15
INNER_VAL = cfg["split"]["inner_val_frac"] # 0.10 — last 10% of train for early stop
MISS_THRESH = cfg["preprocessing"]["missing_threshold"]  # 0.50
CARD_KEY = cfg["features"]["card_key"]     # ["card1", "addr1"]
WINDOWS = cfg["features"]["rolling_windows_sec"]  # [3600, 86400]

os.makedirs(OUT_DIR, exist_ok=True)

# ===== STEP 1: Load and merge raw data ======================================
print("=" * 60)
print("FinShield Preprocessing — leak-free pipeline")
if is_smoke:
    print("  MODE: SMOKE (synthetic data)")
print("=" * 60)

print("\n[1/8] Loading raw CSVs...")
txn_path = os.path.join(RAW_DIR, "train_transaction.csv")
ident_path = os.path.join(RAW_DIR, "train_identity.csv")

# Check that raw files exist — this is a hard blocker
if not os.path.exists(txn_path):
    print(f"ERROR: {txn_path} not found. Run make_synthetic_data.py or download Kaggle data.")
    sys.exit(1)

txn = pd.read_csv(txn_path)
ident = pd.read_csv(ident_path)
print(f"   Transactions : {txn.shape}")
print(f"   Identity     : {ident.shape}")

# Left join: not every transaction has identity info
df = txn.merge(ident, on="TransactionID", how="left")
print(f"   Merged       : {df.shape}")
print(f"   Fraud rate   : {df['isFraud'].mean() * 100:.2f}%")

# Free memory — the raw DataFrames are large
del txn, ident

# ===== STEP 2: Sort by time and compute behavioural features =================
# CORRECTION 2: Compute behavioural features on FULL dataset BEFORE splitting.
# A val/test row uses that card's earlier transactions (real history, not leakage);
# it never uses same-time or later transactions (enforced by the function).
print("\n[2/8] Computing behavioural features on full time-sorted dataset...")
print("   This uses only each card's PAST transactions — no future leakage.")

# Sort by time first — required for behavioural features
df = df.sort_values("TransactionDT").reset_index(drop=True)

df = compute_behavioral_features(df, CARD_KEY, WINDOWS)
print(f"   Done: {df.shape}")

# ===== STEP 3: Chronological split on TransactionDT =========================
print("\n[3/8] Chronological split (train 70% / val 15% / test 15%)...")

n = len(df)
train_end = int(n * TRAIN_FRAC)                       # first 70%
val_end = int(n * (TRAIN_FRAC + VAL_FRAC))             # next 15%

# Slice into three contiguous time-ordered chunks
train_df = df.iloc[:train_end].copy()
val_df = df.iloc[train_end:val_end].copy()
test_df = df.iloc[val_end:].copy()

print(f"   Train : {len(train_df):,} rows  "
      f"(DT {train_df['TransactionDT'].min()}-{train_df['TransactionDT'].max()})")
print(f"   Val   : {len(val_df):,} rows  "
      f"(DT {val_df['TransactionDT'].min()}-{val_df['TransactionDT'].max()})")
print(f"   Test  : {len(test_df):,} rows  "
      f"(DT {test_df['TransactionDT'].min()}-{test_df['TransactionDT'].max()})")

# Sanity check: no overlap in time
assert train_df["TransactionDT"].max() <= val_df["TransactionDT"].min(), \
    "Train/val time overlap detected!"
assert val_df["TransactionDT"].max() <= test_df["TransactionDT"].min(), \
    "Val/test time overlap detected!"

# CORRECTION 5: Carve out last 10% of TRAIN period as inner-validation for
# XGBoost / neural-net early stopping. This avoids using the main validation
# set three times.
inner_boundary = int(len(train_df) * (1 - INNER_VAL))
train_inner_df = train_df.iloc[:inner_boundary].copy()      # 90% of train
inner_val_df = train_df.iloc[inner_boundary:].copy()         # last 10% of train

print(f"\n   Inner train: {len(train_inner_df):,} rows (for model fitting)")
print(f"   Inner val:   {len(inner_val_df):,} rows (for early stopping)")

# ===== STEP 4: Fit preprocessing on TRAIN only ==============================
print("\n[4/8] Fitting preprocessing on TRAIN split only...")

# 4a. Identify columns with > 50% missing IN TRAIN
missing_frac = train_df.isnull().mean()  # fraction missing per column
drop_cols = missing_frac[missing_frac > MISS_THRESH].index.tolist()
# Always keep TransactionID, TransactionDT, isFraud regardless of missingness
protected = {"TransactionID", "TransactionDT", "isFraud"}
drop_cols = [c for c in drop_cols if c not in protected]
print(f"   Dropping {len(drop_cols)} columns with >{MISS_THRESH*100:.0f}% missing in train")

# Apply the drop to all splits
for split in [train_df, val_df, test_df, train_inner_df, inner_val_df]:
    split.drop(columns=[c for c in drop_cols if c in split.columns],
               inplace=True, errors="ignore")

# 4b. Identify numeric and categorical columns (after dropping)
num_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = train_df.select_dtypes(include=["object"]).columns.tolist()

# Remove ID/target from the feature lists
for meta_col in ["TransactionID", "TransactionDT", "isFraud"]:
    if meta_col in num_cols:
        num_cols.remove(meta_col)

print(f"   Numeric features: {len(num_cols)}")
print(f"   Categorical features: {len(cat_cols)}")

# 4c. Compute medians from TRAIN only — used ONLY for AE/LSTM inputs
train_medians = train_df[num_cols].median()
print(f"   Computed medians for {len(num_cols)} numeric columns (train only)")

# 4d. CORRECTION 3: Fit label encoders on TRAIN only; unseen categories
# in val/test map to "__unseen__" code instead of crashing.
label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    # Fill NaN as "unknown" string, then fit with __unseen__ reserved
    train_vals = train_df[col].fillna("unknown").astype(str)
    all_classes = list(train_vals.unique()) + ["__unseen__"]
    le.fit(all_classes)
    label_encoders[col] = le
print(f"   Fitted label encoders for {len(cat_cols)} categoricals (with __unseen__)")

# ===== STEP 5: Encode categoricals (all splits) =============================
print("\n[5/8] Encoding categoricals (unseen → __unseen__)...")


def encode_categoricals(split_df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Label-encode categorical columns using train-fitted encoders.

    CORRECTION 3: Unseen categories (not in train) map to __unseen__ code
    instead of raising an error.

    CORRECTION 4: Numeric NaN is KEPT for XGBoost (it handles missing natively).
    Imputation is done separately for AE/LSTM only.

    Args:
        split_df: One of train_df, val_df, test_df.
        name: "train"/"val"/"test" for logging.

    Returns:
        DataFrame with categoricals label-encoded, numeric NaN preserved.
    """
    out = split_df.copy()

    for col in cat_cols:
        if col not in out.columns:
            continue
        # Fill NaN with "unknown", cast to string
        out[col] = out[col].fillna("unknown").astype(str)
        le = label_encoders[col]
        known = set(le.classes_)
        # Map any category not seen in train to "__unseen__"
        out[col] = out[col].apply(lambda x: x if x in known else "__unseen__")
        out[col] = le.transform(out[col])

    print(f"   {name}: encoded → {out.shape}")
    return out


train_df = encode_categoricals(train_df, "train")
val_df = encode_categoricals(val_df, "val")
test_df = encode_categoricals(test_df, "test")
train_inner_df = encode_categoricals(train_inner_df, "train_inner")
inner_val_df = encode_categoricals(inner_val_df, "inner_val")

# ===== STEP 6: Simple engineered features ====================================
print("\n[6/8] Adding simple engineered features...")


def add_simple_features(split_df: pd.DataFrame) -> pd.DataFrame:
    """Add time-of-day, day-of-week, high-value flag, and log-amount.

    These features are computed from the row itself — no leakage risk.
    """
    out = split_df.copy()
    # Hour of day (TransactionDT is seconds from reference)
    out["Transaction_Hour"] = (out["TransactionDT"] / 3600 % 24).astype(int)
    # Day of week
    out["Transaction_Day"] = (out["TransactionDT"] / (3600 * 24) % 7).astype(int)
    # Flag for high-value transactions (above $500)
    out["Is_High_Value"] = (out["TransactionAmt"] > 500).astype(int)
    # Log-transformed amount (log1p handles zero-amount edge case)
    out["TransactionAmt_Log"] = np.log1p(out["TransactionAmt"])
    return out


train_df = add_simple_features(train_df)
val_df = add_simple_features(val_df)
test_df = add_simple_features(test_df)
train_inner_df = add_simple_features(train_inner_df)
inner_val_df = add_simple_features(inner_val_df)
print("   Added: Transaction_Hour, Transaction_Day, Is_High_Value, TransactionAmt_Log")

# ===== STEP 7: Prepare scaled versions for AE/LSTM ===========================
# CORRECTION 4: NaN kept for XGBoost. Median imputation + StandardScaler
# applied ONLY to AE/LSTM inputs, saved as separate *_scaled.parquet files.
print("\n[7/8] Preparing scaled data for autoencoder / LSTM...")

# Feature columns: everything except ID, time, and label
META_COLS = ["TransactionID", "TransactionDT", "isFraud"]
feature_cols = [c for c in train_df.columns if c not in META_COLS]

# Fit scaler on train (after imputing NaN with train medians)
scaler = StandardScaler()
train_imputed = train_df[feature_cols].fillna(train_medians)
scaler.fit(train_imputed)

# Update num_cols to include new engineered features
num_cols_final = [c for c in feature_cols if c not in cat_cols]


def make_scaled_df(split_df: pd.DataFrame) -> pd.DataFrame:
    """Impute NaN with train medians and apply StandardScaler.

    Used only for AE and LSTM inputs — XGBoost gets raw NaN.
    """
    feats = split_df[feature_cols].fillna(train_medians)
    scaled = pd.DataFrame(
        scaler.transform(feats),
        columns=feature_cols,
        index=split_df.index,
    )
    # Keep meta columns for joining
    for mc in META_COLS:
        if mc in split_df.columns:
            scaled[mc] = split_df[mc].values
    return scaled


train_scaled = make_scaled_df(train_df)
val_scaled = make_scaled_df(val_df)
test_scaled = make_scaled_df(test_df)
train_inner_scaled = make_scaled_df(train_inner_df)
inner_val_scaled = make_scaled_df(inner_val_df)
print(f"   Scaled data shape: {train_scaled.shape}")

# ===== STEP 8: Save everything ===============================================
print("\n[8/8] Saving outputs...")

# Save split DataFrames as parquet (NaN preserved for XGBoost)
train_df.to_parquet(os.path.join(OUT_DIR, "train.parquet"), index=False)
val_df.to_parquet(os.path.join(OUT_DIR, "val.parquet"), index=False)
test_df.to_parquet(os.path.join(OUT_DIR, "test.parquet"), index=False)
train_inner_df.to_parquet(os.path.join(OUT_DIR, "train_inner.parquet"), index=False)
inner_val_df.to_parquet(os.path.join(OUT_DIR, "inner_val.parquet"), index=False)

# Save scaled versions for AE/LSTM (NaN imputed, scaled)
train_scaled.to_parquet(os.path.join(OUT_DIR, "train_scaled.parquet"), index=False)
val_scaled.to_parquet(os.path.join(OUT_DIR, "val_scaled.parquet"), index=False)
test_scaled.to_parquet(os.path.join(OUT_DIR, "test_scaled.parquet"), index=False)
train_inner_scaled.to_parquet(os.path.join(OUT_DIR, "train_inner_scaled.parquet"), index=False)
inner_val_scaled.to_parquet(os.path.join(OUT_DIR, "inner_val_scaled.parquet"), index=False)

print(f"   Saved train.parquet ({len(train_df):,} rows)")
print(f"   Saved val.parquet   ({len(val_df):,} rows)")
print(f"   Saved test.parquet  ({len(test_df):,} rows)")
print(f"   Saved train_inner + inner_val parquets for early stopping")
print(f"   Saved *_scaled.parquet files for AE/LSTM")

# Save the fitted preprocessor so other scripts and tests can verify it
preprocessor = {
    "drop_cols": drop_cols,              # columns dropped due to missingness
    "num_cols": num_cols,                # original numeric feature column names
    "cat_cols": cat_cols,                # categorical column names
    "feature_cols": feature_cols,        # all feature columns (final)
    "train_medians": train_medians,      # median values for imputation
    "label_encoders": label_encoders,    # fitted LabelEncoder per cat column
    "scaler": scaler,                    # fitted StandardScaler for AE/LSTM
}
with open(os.path.join(OUT_DIR, "preprocessor.pkl"), "wb") as f:
    pickle.dump(preprocessor, f)
print("   Saved preprocessor.pkl (fitted on train only)")

# Save split boundaries for reproducibility and test verification
split_info = {
    "train_dt_min": int(train_df["TransactionDT"].min()),
    "train_dt_max": int(train_df["TransactionDT"].max()),
    "val_dt_min": int(val_df["TransactionDT"].min()),
    "val_dt_max": int(val_df["TransactionDT"].max()),
    "test_dt_min": int(test_df["TransactionDT"].min()),
    "test_dt_max": int(test_df["TransactionDT"].max()),
    "train_rows": len(train_df),
    "val_rows": len(val_df),
    "test_rows": len(test_df),
    "inner_train_rows": len(train_inner_df),
    "inner_val_rows": len(inner_val_df),
    "n_features": len(feature_cols),
    "is_smoke": is_smoke,
}
with open(os.path.join(OUT_DIR, "split_info.pkl"), "wb") as f:
    pickle.dump(split_info, f)

# Also save as JSON for human readability
with open(os.path.join(OUT_DIR, "split_info.json"), "w") as f:
    json.dump(split_info, f, indent=2)
print("   Saved split_info.pkl + split_info.json")

# Print final summary
print(f"\n{'=' * 60}")
print("Preprocessing complete!")
print(f"   Train fraud rate : {train_df['isFraud'].mean()*100:.2f}%")
print(f"   Val fraud rate   : {val_df['isFraud'].mean()*100:.2f}%")
print(f"   Test fraud rate  : {test_df['isFraud'].mean()*100:.2f}%")
print(f"   Feature columns  : {len(feature_cols)}")
print(f"   Output directory : {OUT_DIR}")
print(f"{'=' * 60}")