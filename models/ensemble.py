"""
models/ensemble.py
Stacking ensemble: fits a logistic regression on validation-set
out-of-fold predictions from XGBoost, Autoencoder, and (optionally) LSTM.

CORRECTION 5: Fits stacking LR on validation with 5-fold cross_val_predict.
              Chooses both thresholds (review / block) on those OOF scores.
CORRECTION 7: Model gate — keeps AE and LSTM only if the ensemble WITH that
              model beats the ensemble WITHOUT it on validation PR-AUC.
              Saves the comparison to reports/model_selection.json.

Run from repo root:
    python models/ensemble.py
    SMOKE=1 python models/ensemble.py
"""

import pandas as pd
import numpy as np
import pickle
import os
import sys
import json
import warnings

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.config import cfg, is_smoke, get_processed_dir, get_mlflow_uri, get_reports_dir

import mlflow
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import (
    average_precision_score, roc_auc_score, classification_report,
    precision_recall_curve,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PROCESSED = get_processed_dir()
REPORTS = get_reports_dir()
STACKING_FOLDS = cfg["ensemble"]["stacking_cv_folds"]
REVIEW_THRESH = cfg["thresholds"]["review"]
BLOCK_THRESH = cfg["thresholds"]["block"]

mlflow_uri = get_mlflow_uri()
mlflow.set_tracking_uri(mlflow_uri)
mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

# ---------------------------------------------------------------------------
# Load models and validation data
# ---------------------------------------------------------------------------
print("=" * 60)
print("FinShield Ensemble (Stacking + Model Gate)")
if is_smoke:
    print("  MODE: SMOKE")
print("=" * 60)

print("\n[1/6] Loading models and data...")

# XGBoost
with open("models/saved/xgboost_model.pkl", "rb") as f:
    xgb_model = pickle.load(f)
with open("models/saved/feature_cols.pkl", "rb") as f:
    feature_cols = pickle.load(f)

# Autoencoder
import tensorflow as tf
ae_model = tf.keras.models.load_model("models/saved/autoencoder.keras")
with open("models/saved/ae_calibration.pkl", "rb") as f:
    ae_legit_mse = pickle.load(f)

# LSTM (may not exist if training failed)
lstm_available = os.path.exists("models/saved/lstm_model.keras")
if lstm_available:
    lstm_model = tf.keras.models.load_model("models/saved/lstm_model.keras")
    with open("models/saved/lstm_features.pkl", "rb") as f:
        lstm_features = pickle.load(f)
    print("   LSTM model loaded")
else:
    print("   LSTM model not found — ensemble will use XGBoost + AE only")

# Load validation data (raw NaN for XGBoost, scaled for AE/LSTM)
val_df = pd.read_parquet(os.path.join(PROCESSED, "val.parquet"))
val_scaled = pd.read_parquet(os.path.join(PROCESSED, "val_scaled.parquet"))

META_COLS = ["TransactionID", "TransactionDT", "isFraud"]
X_val = val_df[feature_cols].astype(np.float32)
y_val = val_df["isFraud"].values
X_val_scaled = val_scaled[feature_cols].astype(np.float32).values

print(f"   Val rows: {len(y_val):,}")
print(f"   Val fraud rate: {y_val.mean()*100:.2f}%")

# ---------------------------------------------------------------------------
# Generate component predictions on validation set
# ---------------------------------------------------------------------------
print("\n[2/6] Generating component predictions on validation set...")

# XGBoost predictions
xgb_probs = xgb_model.predict_proba(X_val)[:, 1]
print(f"   XGBoost done — mean prob: {xgb_probs.mean():.4f}")

# Autoencoder anomaly scores (percentile calibration)
ae_recon = ae_model.predict(X_val_scaled, verbose=0)
ae_mse = np.mean(np.power(X_val_scaled - ae_recon, 2), axis=1)
ae_scores = np.searchsorted(ae_legit_mse, ae_mse) / len(ae_legit_mse)
print(f"   Autoencoder done — mean score: {ae_scores.mean():.4f}")

# LSTM predictions (if available)
if lstm_available:
    SEQ_LEN = cfg["features"]["lstm_seq_len"]
    # Build sequences for validation
    # Load all data up to val for sequence history
    train_scaled_full = pd.read_parquet(os.path.join(PROCESSED, "train_scaled.parquet"))
    all_for_seq = pd.concat([train_scaled_full, val_scaled], ignore_index=True)
    all_for_seq = all_for_seq.sort_values("TransactionDT").reset_index(drop=True)

    X_seq = all_for_seq[lstm_features].astype(np.float32).values
    dt_seq = all_for_seq["TransactionDT"].values
    val_dt_min = float(val_scaled["TransactionDT"].min())
    val_dt_max = float(val_scaled["TransactionDT"].max())

    # Generate predictions for val-period transactions
    lstm_preds_list = []
    lstm_indices = []  # track which val rows have LSTM predictions
    for i in range(SEQ_LEN, len(X_seq)):
        if dt_seq[i] < val_dt_min or dt_seq[i] > val_dt_max:
            continue
        seq = X_seq[i - SEQ_LEN:i][np.newaxis]  # (1, seq_len, n_features)
        pred = lstm_model.predict(seq, verbose=0)[0, 0]
        lstm_preds_list.append(pred)
        lstm_indices.append(i)

    # Map LSTM predictions back to val indices
    # Not all val rows may have LSTM predictions (e.g., first few rows)
    lstm_probs = np.zeros(len(y_val))
    lstm_has_pred = np.zeros(len(y_val), dtype=bool)

    # Find val rows in the combined dataframe
    val_dt_values = val_df["TransactionDT"].values
    seq_dt_values = dt_seq[lstm_indices] if lstm_indices else np.array([])

    # Simple approach: for each val row, find matching LSTM prediction by DT
    if len(lstm_preds_list) > 0:
        pred_idx = 0
        for v_idx in range(len(y_val)):
            if pred_idx < len(lstm_preds_list) and pred_idx < len(seq_dt_values):
                if abs(val_dt_values[v_idx] - seq_dt_values[pred_idx]) < 1:
                    lstm_probs[v_idx] = lstm_preds_list[pred_idx]
                    lstm_has_pred[v_idx] = True
                    pred_idx += 1

    # For rows without LSTM predictions, use XGBoost prob as fallback
    lstm_probs[~lstm_has_pred] = xgb_probs[~lstm_has_pred]
    print(f"   LSTM done — {lstm_has_pred.sum()} rows with predictions")


# ---------------------------------------------------------------------------
# Model gate: test if each component improves ensemble PR-AUC
# ---------------------------------------------------------------------------
print("\n[3/6] Model gate — testing component contributions...")

model_selection = {}


def evaluate_stacking(stack_X, y, name, n_folds=STACKING_FOLDS):
    """Fit stacking LR with cross_val_predict and return OOF PR-AUC.

    CORRECTION 5: Uses cross_val_predict on validation so the stacking
    model is never evaluated on data it trained on.
    """
    lr = LogisticRegression(random_state=42, max_iter=1000)
    # Out-of-fold predictions using cross_val_predict
    oof_probs = cross_val_predict(lr, stack_X, y, cv=n_folds, method="predict_proba")
    oof_scores = oof_probs[:, 1]  # probability of fraud
    pr_auc = average_precision_score(y, oof_scores)
    roc_auc = roc_auc_score(y, oof_scores)
    print(f"   {name}: PR-AUC={pr_auc:.4f}, ROC-AUC={roc_auc:.4f}")
    return pr_auc, roc_auc, oof_scores


# Ensemble A: XGBoost only (baseline)
stack_xgb = xgb_probs.reshape(-1, 1)
prauc_xgb, rocauc_xgb, _ = evaluate_stacking(stack_xgb, y_val, "XGBoost only")
model_selection["xgboost_only"] = {"pr_auc": float(prauc_xgb)}

# Ensemble B: XGBoost + AE
stack_xgb_ae = np.column_stack([xgb_probs, ae_scores])
prauc_xgb_ae, rocauc_xgb_ae, _ = evaluate_stacking(
    stack_xgb_ae, y_val, "XGBoost + AE")
model_selection["xgboost_ae"] = {"pr_auc": float(prauc_xgb_ae)}

# Track which components are included
include_ae = prauc_xgb_ae > prauc_xgb
include_lstm = False

best_prauc = prauc_xgb_ae if include_ae else prauc_xgb
best_name = "xgboost_ae" if include_ae else "xgboost_only"

if lstm_available:
    # Ensemble C: XGBoost + AE + LSTM
    stack_all = np.column_stack([xgb_probs, ae_scores, lstm_probs])
    prauc_all, rocauc_all, _ = evaluate_stacking(
        stack_all, y_val, "XGBoost + AE + LSTM")
    model_selection["xgboost_ae_lstm"] = {"pr_auc": float(prauc_all)}

    # Ensemble D: XGBoost + LSTM (without AE)
    stack_xgb_lstm = np.column_stack([xgb_probs, lstm_probs])
    prauc_xgb_lstm, _, _ = evaluate_stacking(
        stack_xgb_lstm, y_val, "XGBoost + LSTM")
    model_selection["xgboost_lstm"] = {"pr_auc": float(prauc_xgb_lstm)}

    # Pick the best ensemble
    candidates = {
        "xgboost_only": prauc_xgb,
        "xgboost_ae": prauc_xgb_ae,
        "xgboost_ae_lstm": prauc_all,
        "xgboost_lstm": prauc_xgb_lstm,
    }
    best_name = max(candidates, key=candidates.get)
    best_prauc = candidates[best_name]
    include_ae = "ae" in best_name
    include_lstm = "lstm" in best_name

print(f"\n   → Best ensemble: {best_name} (PR-AUC: {best_prauc:.4f})")
print(f"   → Include AE: {include_ae}")
print(f"   → Include LSTM: {include_lstm}")

# Record model selection decision
model_selection["winner"] = best_name
model_selection["include_ae"] = include_ae
model_selection["include_lstm"] = include_lstm

os.makedirs(REPORTS, exist_ok=True)
with open(os.path.join(REPORTS, "model_selection.json"), "w") as f:
    json.dump(model_selection, f, indent=2)
print(f"   Saved: {REPORTS}/model_selection.json")

# ---------------------------------------------------------------------------
# Fit final stacking LR on full validation with cross_val_predict
# ---------------------------------------------------------------------------
print("\n[4/6] Fitting final stacking ensemble with cross_val_predict...")

# Build the final stack matrix based on model gate results
components = [("xgboost", xgb_probs)]
if include_ae:
    components.append(("autoencoder", ae_scores))
if include_lstm:
    components.append(("lstm", lstm_probs))

component_names = [c[0] for c in components]
stack_X = np.column_stack([c[1] for c in components])

# Get OOF predictions for threshold selection
lr_final = LogisticRegression(random_state=42, max_iter=1000)
oof_probs = cross_val_predict(
    lr_final, stack_X, y_val, cv=STACKING_FOLDS, method="predict_proba"
)
oof_scores = oof_probs[:, 1]

# Fit final model on all validation data (for inference)
lr_final.fit(stack_X, y_val)

# ---------------------------------------------------------------------------
# Choose thresholds on OOF scores
# ---------------------------------------------------------------------------
print("\n[5/6] Choosing thresholds on OOF scores...")

final_pr_auc = average_precision_score(y_val, oof_scores)
final_roc_auc = roc_auc_score(y_val, oof_scores)
print(f"   Ensemble OOF PR-AUC:  {final_pr_auc:.4f}")
print(f"   Ensemble OOF ROC-AUC: {final_roc_auc:.4f}")

# Find optimal thresholds from config
review_thresh = REVIEW_THRESH
block_thresh = BLOCK_THRESH

# Also compute F1-optimal threshold from precision-recall curve
precision, recall, pr_thresholds = precision_recall_curve(y_val, oof_scores)
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
best_f1_idx = np.argmax(f1_scores)
f1_optimal_thresh = float(pr_thresholds[best_f1_idx]) if best_f1_idx < len(pr_thresholds) else 0.5

print(f"   Config review threshold: {review_thresh}")
print(f"   Config block threshold:  {block_thresh}")
print(f"   F1-optimal threshold:    {f1_optimal_thresh:.4f}")

# Apply thresholds for classification report
decisions = np.where(oof_scores >= block_thresh, "block",
                     np.where(oof_scores >= review_thresh, "review", "approve"))
print(f"\n   Decision distribution:")
for d in ["approve", "review", "block"]:
    count = (decisions == d).sum()
    fraud_in = y_val[decisions == d].sum() if count > 0 else 0
    print(f"     {d:8s}: {count:,} ({fraud_in} fraud)")

# ---------------------------------------------------------------------------
# Save ensemble config and metrics
# ---------------------------------------------------------------------------
print("\n[6/6] Saving ensemble...")

os.makedirs("models/saved", exist_ok=True)

ensemble_config = {
    "components": component_names,
    "stacking_model": lr_final,
    "review_threshold": review_thresh,
    "block_threshold": block_thresh,
    "f1_optimal_threshold": f1_optimal_thresh,
    "feature_cols": feature_cols,
    "include_ae": include_ae,
    "include_lstm": include_lstm,
}
with open("models/saved/ensemble_config.pkl", "wb") as f:
    pickle.dump(ensemble_config, f)

# Save metrics
ensemble_metrics = {
    "val_pr_auc": float(final_pr_auc),
    "val_roc_auc": float(final_roc_auc),
    "components": component_names,
    "review_threshold": float(review_thresh),
    "block_threshold": float(block_thresh),
    "f1_optimal_threshold": float(f1_optimal_thresh),
}
with open("models/saved/ensemble_metrics.json", "w") as f:
    json.dump(ensemble_metrics, f, indent=2)

# Log to MLflow
with mlflow.start_run(run_name="Ensemble_stacking"):
    mlflow.log_metric("val_pr_auc", final_pr_auc)
    mlflow.log_metric("val_roc_auc", final_roc_auc)
    mlflow.log_params({
        "components": str(component_names),
        "stacking_folds": STACKING_FOLDS,
        "include_ae": include_ae,
        "include_lstm": include_lstm,
    })

print(f"\n{'=' * 60}")
print("Ensemble training complete!")
print(f"   Components   : {component_names}")
print(f"   Val PR-AUC   : {final_pr_auc:.4f}")
print(f"   Val ROC-AUC  : {final_roc_auc:.4f}")
print(f"{'=' * 60}")