"""
models/train_xgboost.py
Trains XGBoost on the chronological train split using inner-validation
(last 10% of TRAIN period) for early stopping.  Compares two strategies:
  A. scale_pos_weight only
  B. SMOTE + scale_pos_weight

Keeps whichever has higher validation PR-AUC.  Logs to MLflow with
local-artifact fallback if the server is unreachable.

Run from repo root:
    python models/train_xgboost.py
    SMOKE=1 python models/train_xgboost.py
"""

import pandas as pd
import numpy as np
import pickle
import os
import sys
import json
import warnings

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Allow imports from repo root
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.config import cfg, is_smoke, get_processed_dir, get_mlflow_uri, get_reports_dir

import mlflow
import mlflow.sklearn
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score
from imblearn.over_sampling import SMOTE

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PROCESSED = get_processed_dir()
XGB_CFG = cfg["xgboost"]
SMOTE_CFG = cfg["smote"]

# ---------------------------------------------------------------------------
# Setup MLflow with server / fallback
# ---------------------------------------------------------------------------
mlflow_uri = get_mlflow_uri()
mlflow.set_tracking_uri(mlflow_uri)
mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

# ---------------------------------------------------------------------------
# Load data: use inner train/val for early stopping
# ---------------------------------------------------------------------------
print("=" * 60)
print("FinShield XGBoost Training")
if is_smoke:
    print("  MODE: SMOKE (synthetic data, tiny settings)")
print("=" * 60)

print("\n[1/5] Loading data...")
# CORRECTION 5: Use inner train (90% of train) and inner val (last 10%)
# for early stopping — never touch the main validation set
train_inner = pd.read_parquet(os.path.join(PROCESSED, "train_inner.parquet"))
inner_val = pd.read_parquet(os.path.join(PROCESSED, "inner_val.parquet"))
# Full train for final refit after picking strategy
full_train = pd.read_parquet(os.path.join(PROCESSED, "train.parquet"))

# Define feature columns: everything except ID, time, and label
META_COLS = ["TransactionID", "TransactionDT", "isFraud"]
feature_cols = [c for c in train_inner.columns if c not in META_COLS]

# Separate features and labels (NaN preserved — XGBoost handles it natively)
X_inner = train_inner[feature_cols].astype(np.float32)
y_inner = train_inner["isFraud"]
X_ival = inner_val[feature_cols].astype(np.float32)
y_ival = inner_val["isFraud"]

print(f"   Inner train: {len(X_inner):,} rows, {len(feature_cols)} features")
print(f"   Inner val:   {len(X_ival):,} rows")
print(f"   Inner train fraud rate: {y_inner.mean()*100:.2f}%")

# ---------------------------------------------------------------------------
# Compute scale_pos_weight from inner train class balance
# ---------------------------------------------------------------------------
n_legit = (y_inner == 0).sum()
n_fraud = (y_inner == 1).sum()
auto_spw = n_legit / max(n_fraud, 1)  # avoid division by zero
print(f"\n   scale_pos_weight: {auto_spw:.1f} ({n_legit:,} legit / {n_fraud:,} fraud)")

# ---------------------------------------------------------------------------
# Base XGBoost params from config
# ---------------------------------------------------------------------------
base_params = {
    "n_estimators": XGB_CFG["n_estimators"],
    "max_depth": XGB_CFG["max_depth"],
    "learning_rate": XGB_CFG["learning_rate"],
    "subsample": XGB_CFG["subsample"],
    "colsample_bytree": XGB_CFG["colsample_bytree"],
    "eval_metric": XGB_CFG["eval_metric"],
    "random_state": XGB_CFG["random_state"],
    "n_jobs": XGB_CFG["n_jobs"],
    "early_stopping_rounds": XGB_CFG["early_stopping_rounds"],
}

# ---------------------------------------------------------------------------
# Strategy A: scale_pos_weight only (no SMOTE)
# ---------------------------------------------------------------------------
print("\n[2/5] Training Strategy A: scale_pos_weight only...")
params_a = {**base_params, "scale_pos_weight": auto_spw}

with mlflow.start_run(run_name="XGBoost_spw_only"):
    model_a = xgb.XGBClassifier(**params_a)
    model_a.fit(
        X_inner, y_inner,
        eval_set=[(X_ival, y_ival)],  # inner validation, NOT main val
        verbose=50 if not is_smoke else 10,
    )

    val_probs_a = model_a.predict_proba(X_ival)[:, 1]
    pr_auc_a = average_precision_score(y_ival, val_probs_a)
    roc_auc_a = roc_auc_score(y_ival, val_probs_a)

    mlflow.log_params(params_a)
    mlflow.log_metric("inner_val_pr_auc", pr_auc_a)
    mlflow.log_metric("inner_val_roc_auc", roc_auc_a)
    mlflow.log_param("strategy", "scale_pos_weight_only")

print(f"   Strategy A — Inner-val PR-AUC: {pr_auc_a:.4f}, ROC-AUC: {roc_auc_a:.4f}")

# ---------------------------------------------------------------------------
# Strategy B: SMOTE + scale_pos_weight
# ---------------------------------------------------------------------------
print("\n[3/5] Training Strategy B: SMOTE + scale_pos_weight...")
print("   Applying SMOTE...")

# SMOTE requires no NaN — impute with 0 for SMOTE only (XGBoost NaN feature
# still handled natively in the final model which is refitted without SMOTE)
X_inner_filled = X_inner.fillna(0)

smote = SMOTE(
    random_state=XGB_CFG["random_state"],
    sampling_strategy=SMOTE_CFG["sampling_strategy"],
)
X_smote, y_smote = smote.fit_resample(X_inner_filled, y_inner)
print(f"   After SMOTE: {len(X_smote):,} rows")

params_b = {**base_params, "scale_pos_weight": 10}

with mlflow.start_run(run_name="XGBoost_smote"):
    model_b = xgb.XGBClassifier(**params_b)
    model_b.fit(
        X_smote, y_smote,
        eval_set=[(X_ival, y_ival)],
        verbose=50 if not is_smoke else 10,
    )

    val_probs_b = model_b.predict_proba(X_ival)[:, 1]
    pr_auc_b = average_precision_score(y_ival, val_probs_b)
    roc_auc_b = roc_auc_score(y_ival, val_probs_b)

    mlflow.log_params(params_b)
    mlflow.log_metric("inner_val_pr_auc", pr_auc_b)
    mlflow.log_metric("inner_val_roc_auc", roc_auc_b)
    mlflow.log_param("strategy", "smote_plus_spw")

print(f"   Strategy B — Inner-val PR-AUC: {pr_auc_b:.4f}, ROC-AUC: {roc_auc_b:.4f}")

# ---------------------------------------------------------------------------
# Pick the winner based on inner-validation PR-AUC
# ---------------------------------------------------------------------------
print("\n[4/5] Comparing strategies...")
if pr_auc_a >= pr_auc_b:
    winner_name = "scale_pos_weight_only"
    winner_params = params_a
    winner_pr_auc = pr_auc_a
    winner_roc_auc = roc_auc_a
    print(f"   → Winner: Strategy A (scale_pos_weight only)")
else:
    winner_name = "smote_plus_spw"
    winner_params = params_b
    winner_pr_auc = pr_auc_b
    winner_roc_auc = roc_auc_b
    print(f"   → Winner: Strategy B (SMOTE + spw=10)")

# ---------------------------------------------------------------------------
# Refit winner on FULL train set (inner train + inner val combined)
# ---------------------------------------------------------------------------
print("\n[5/5] Refitting winner on full train set and saving...")

X_full = full_train[feature_cols].astype(np.float32)
y_full = full_train["isFraud"]

# Refit without early stopping (we already know the best n_estimators)
refit_params = {k: v for k, v in winner_params.items()
                if k != "early_stopping_rounds"}
# Use the number of rounds from the early-stopped model
if winner_name == "scale_pos_weight_only":
    refit_params["n_estimators"] = model_a.best_iteration + 1
else:
    refit_params["n_estimators"] = model_b.best_iteration + 1

winner_model = xgb.XGBClassifier(**refit_params)
winner_model.fit(X_full, y_full, verbose=0)

# Save model and feature columns
os.makedirs("models/saved", exist_ok=True)
with open("models/saved/xgboost_model.pkl", "wb") as f:
    pickle.dump(winner_model, f)
with open("models/saved/feature_cols.pkl", "wb") as f:
    pickle.dump(feature_cols, f)

# Save comparison for documentation
comparison = {
    "strategy_a": {"name": "scale_pos_weight_only",
                   "inner_val_pr_auc": float(pr_auc_a),
                   "inner_val_roc_auc": float(roc_auc_a)},
    "strategy_b": {"name": "smote_plus_spw",
                   "inner_val_pr_auc": float(pr_auc_b),
                   "inner_val_roc_auc": float(roc_auc_b)},
    "winner": winner_name,
    "best_n_estimators": int(refit_params["n_estimators"]),
}
with open("models/saved/xgb_comparison.json", "w") as f:
    json.dump(comparison, f, indent=2)

# Log winner to MLflow with model artifact
with mlflow.start_run(run_name="XGBoost_winner"):
    mlflow.log_params(refit_params)
    mlflow.log_param("strategy", winner_name)
    mlflow.log_metric("inner_val_pr_auc", winner_pr_auc)
    mlflow.log_metric("inner_val_roc_auc", winner_roc_auc)
    mlflow.sklearn.log_model(winner_model, "xgboost_model")

print(f"\n{'=' * 60}")
print(f"XGBoost training complete!")
print(f"   Winner       : {winner_name}")
print(f"   Inner-val PR-AUC : {winner_pr_auc:.4f}")
print(f"   Inner-val ROC-AUC: {winner_roc_auc:.4f}")
print(f"   Best rounds  : {refit_params['n_estimators']}")
print(f"   Saved to     : models/saved/xgboost_model.pkl")
print(f"{'=' * 60}")
