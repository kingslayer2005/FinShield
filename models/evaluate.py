"""
models/evaluate.py
Final evaluation on the TEST set.  Computes PR-AUC, ROC-AUC, precision,
recall, F1 for every component model and the ensemble.  Saves:
  - reports/metrics.json (or reports_smoke/metrics.json)
  - reports/figures/ (confusion matrix, PR curve, ROC curve)

Run from repo root:
    python models/evaluate.py
    SMOKE=1 python models/evaluate.py
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

from utils.config import cfg, is_smoke, get_processed_dir, get_reports_dir

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving plots
import matplotlib.pyplot as plt
from sklearn.metrics import (
    average_precision_score, roc_auc_score, precision_recall_curve,
    roc_curve, confusion_matrix, classification_report,
    f1_score, precision_score, recall_score,
)
import tensorflow as tf

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PROCESSED = get_processed_dir()
REPORTS = get_reports_dir()
FIGURES = os.path.join(REPORTS, "figures")
os.makedirs(FIGURES, exist_ok=True)

# ---------------------------------------------------------------------------
# Load models
# ---------------------------------------------------------------------------
print("=" * 60)
print("FinShield Final Evaluation on TEST Set")
if is_smoke:
    print("  MODE: SMOKE (synthetic data — results are NOT real)")
print("=" * 60)

print("\n[1/4] Loading models...")

with open("models/saved/xgboost_model.pkl", "rb") as f:
    xgb_model = pickle.load(f)
with open("models/saved/feature_cols.pkl", "rb") as f:
    feature_cols = pickle.load(f)
with open("models/saved/ensemble_config.pkl", "rb") as f:
    ensemble_cfg = pickle.load(f)

ae_model = tf.keras.models.load_model("models/saved/autoencoder.keras")
with open("models/saved/ae_calibration.pkl", "rb") as f:
    ae_legit_mse = pickle.load(f)

include_ae = ensemble_cfg["include_ae"]
include_lstm = ensemble_cfg["include_lstm"]

lstm_model = None
lstm_features = None
if include_lstm and os.path.exists("models/saved/lstm_model.keras"):
    lstm_model = tf.keras.models.load_model("models/saved/lstm_model.keras")
    with open("models/saved/lstm_features.pkl", "rb") as f:
        lstm_features = pickle.load(f)

stacking_lr = ensemble_cfg["stacking_model"]
review_thresh = ensemble_cfg["review_threshold"]
block_thresh = ensemble_cfg["block_threshold"]

# ---------------------------------------------------------------------------
# Load test data
# ---------------------------------------------------------------------------
print("\n[2/4] Loading test data...")
test_df = pd.read_parquet(os.path.join(PROCESSED, "test.parquet"))
test_scaled = pd.read_parquet(os.path.join(PROCESSED, "test_scaled.parquet"))

META_COLS = ["TransactionID", "TransactionDT", "isFraud"]
X_test = test_df[feature_cols].astype(np.float32)
y_test = test_df["isFraud"].values
X_test_scaled = test_scaled[feature_cols].astype(np.float32).values

print(f"   Test rows: {len(y_test):,}")
print(f"   Test fraud rate: {y_test.mean()*100:.2f}%")

# ---------------------------------------------------------------------------
# Generate predictions from each component
# ---------------------------------------------------------------------------
print("\n[3/4] Generating predictions...")

results = {}

# XGBoost
xgb_probs = xgb_model.predict_proba(X_test)[:, 1]
xgb_prauc = average_precision_score(y_test, xgb_probs)
xgb_rocauc = roc_auc_score(y_test, xgb_probs)
results["xgboost"] = {
    "test_pr_auc": float(xgb_prauc),
    "test_roc_auc": float(xgb_rocauc),
}
print(f"   XGBoost — PR-AUC: {xgb_prauc:.4f}, ROC-AUC: {xgb_rocauc:.4f}")

# Autoencoder
ae_recon = ae_model.predict(X_test_scaled, verbose=0)
ae_mse = np.mean(np.power(X_test_scaled - ae_recon, 2), axis=1)
ae_scores = np.searchsorted(ae_legit_mse, ae_mse) / len(ae_legit_mse)
ae_prauc = average_precision_score(y_test, ae_scores)
ae_rocauc = roc_auc_score(y_test, ae_scores)
results["autoencoder"] = {
    "test_pr_auc": float(ae_prauc),
    "test_roc_auc": float(ae_rocauc),
    "included_in_ensemble": include_ae,
}
print(f"   Autoencoder — PR-AUC: {ae_prauc:.4f}, ROC-AUC: {ae_rocauc:.4f}"
      f" {'(included)' if include_ae else '(excluded by gate)'}")

# LSTM
if lstm_model is not None:
    SEQ_LEN = cfg["features"]["lstm_seq_len"]
    train_scaled_full = pd.read_parquet(os.path.join(PROCESSED, "train_scaled.parquet"))
    all_for_seq = pd.concat([train_scaled_full, test_scaled], ignore_index=True)
    all_for_seq = all_for_seq.sort_values("TransactionDT").reset_index(drop=True)

    X_seq = all_for_seq[lstm_features].astype(np.float32).values
    dt_seq = all_for_seq["TransactionDT"].values
    test_dt_min = float(test_scaled["TransactionDT"].min())
    test_dt_max = float(test_scaled["TransactionDT"].max())

    lstm_preds = np.zeros(len(y_test))
    lstm_has_pred = np.zeros(len(y_test), dtype=bool)
    test_dt = test_df["TransactionDT"].values

    pred_list = []
    pred_dts = []
    for i in range(SEQ_LEN, len(X_seq)):
        if dt_seq[i] < test_dt_min or dt_seq[i] > test_dt_max:
            continue
        seq = X_seq[i - SEQ_LEN:i][np.newaxis]
        pred = lstm_model.predict(seq, verbose=0)[0, 0]
        pred_list.append(pred)
        pred_dts.append(dt_seq[i])

    # Map predictions back
    if pred_list:
        pred_idx = 0
        for t_idx in range(len(y_test)):
            if pred_idx < len(pred_list):
                if abs(test_dt[t_idx] - pred_dts[pred_idx]) < 1:
                    lstm_preds[t_idx] = pred_list[pred_idx]
                    lstm_has_pred[t_idx] = True
                    pred_idx += 1
        lstm_preds[~lstm_has_pred] = xgb_probs[~lstm_has_pred]

    lstm_prauc = average_precision_score(y_test, lstm_preds)
    lstm_rocauc = roc_auc_score(y_test, lstm_preds)
    results["lstm"] = {
        "test_pr_auc": float(lstm_prauc),
        "test_roc_auc": float(lstm_rocauc),
        "included_in_ensemble": include_lstm,
        "coverage": float(lstm_has_pred.mean()),
    }
    print(f"   LSTM — PR-AUC: {lstm_prauc:.4f}, ROC-AUC: {lstm_rocauc:.4f}"
          f" {'(included)' if include_lstm else '(excluded by gate)'}")
else:
    lstm_preds = xgb_probs  # fallback
    results["lstm"] = {"included_in_ensemble": False, "reason": "not trained or not found"}

# Ensemble
stack_cols = [xgb_probs]
if include_ae:
    stack_cols.append(ae_scores)
if include_lstm and lstm_model is not None:
    stack_cols.append(lstm_preds)

stack_X = np.column_stack(stack_cols)
ensemble_probs = stacking_lr.predict_proba(stack_X)[:, 1]

ens_prauc = average_precision_score(y_test, ensemble_probs)
ens_rocauc = roc_auc_score(y_test, ensemble_probs)

# Apply thresholds
y_pred_binary = (ensemble_probs >= review_thresh).astype(int)
ens_f1 = f1_score(y_test, y_pred_binary)
ens_precision = precision_score(y_test, y_pred_binary, zero_division=0)
ens_recall = recall_score(y_test, y_pred_binary)

decisions = np.where(ensemble_probs >= block_thresh, "block",
                     np.where(ensemble_probs >= review_thresh, "review", "approve"))

results["ensemble"] = {
    "test_pr_auc": float(ens_prauc),
    "test_roc_auc": float(ens_rocauc),
    "test_f1": float(ens_f1),
    "test_precision": float(ens_precision),
    "test_recall": float(ens_recall),
    "review_threshold": float(review_thresh),
    "block_threshold": float(block_thresh),
    "components": ensemble_cfg["components"],
    "approve_count": int((decisions == "approve").sum()),
    "review_count": int((decisions == "review").sum()),
    "block_count": int((decisions == "block").sum()),
}
print(f"   Ensemble — PR-AUC: {ens_prauc:.4f}, ROC-AUC: {ens_rocauc:.4f}, "
      f"F1: {ens_f1:.4f}")

# ---------------------------------------------------------------------------
# Save metrics and generate figures
# ---------------------------------------------------------------------------
print("\n[4/4] Saving metrics and figures...")

# Add metadata
results["metadata"] = {
    "is_smoke": is_smoke,
    "test_rows": int(len(y_test)),
    "test_fraud_count": int(y_test.sum()),
    "test_fraud_rate": float(y_test.mean()),
}

metrics_path = os.path.join(REPORTS, "metrics.json")
with open(metrics_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"   Saved: {metrics_path}")

# PR Curve
plt.figure(figsize=(8, 6))
for name, probs in [("XGBoost", xgb_probs), ("Ensemble", ensemble_probs)]:
    prec, rec, _ = precision_recall_curve(y_test, probs)
    auc = average_precision_score(y_test, probs)
    plt.plot(rec, prec, label=f"{name} (PR-AUC={auc:.4f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve" + (" [SMOKE]" if is_smoke else ""))
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES, "pr_curve.png"), dpi=150)
plt.close()

# ROC Curve
plt.figure(figsize=(8, 6))
for name, probs in [("XGBoost", xgb_probs), ("Ensemble", ensemble_probs)]:
    fpr, tpr, _ = roc_curve(y_test, probs)
    auc = roc_auc_score(y_test, probs)
    plt.plot(fpr, tpr, label=f"{name} (ROC-AUC={auc:.4f})")
plt.plot([0, 1], [0, 1], "k--", alpha=0.3)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve" + (" [SMOKE]" if is_smoke else ""))
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES, "roc_curve.png"), dpi=150)
plt.close()

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_binary)
plt.figure(figsize=(6, 5))
plt.imshow(cm, interpolation="nearest", cmap="Blues")
plt.title("Confusion Matrix" + (" [SMOKE]" if is_smoke else ""))
plt.colorbar()
for i in range(2):
    for j in range(2):
        plt.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                 color="white" if cm[i, j] > cm.max() / 2 else "black")
plt.xticks([0, 1], ["Legit", "Fraud"])
plt.yticks([0, 1], ["Legit", "Fraud"])
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES, "confusion_matrix.png"), dpi=150)
plt.close()

print(f"   Saved figures to {FIGURES}/")

# Print summary table
print(f"\n{'=' * 60}")
print("TEST SET RESULTS" + (" [SMOKE — synthetic data]" if is_smoke else ""))
print(f"{'=' * 60}")
print(f"{'Model':<15} {'PR-AUC':>10} {'ROC-AUC':>10}")
print(f"{'-'*15} {'-'*10} {'-'*10}")
for model_name in ["xgboost", "autoencoder", "lstm", "ensemble"]:
    if model_name in results and "test_pr_auc" in results[model_name]:
        pr = results[model_name]["test_pr_auc"]
        roc = results[model_name]["test_roc_auc"]
        suffix = ""
        if model_name in ["autoencoder", "lstm"]:
            incl = results[model_name].get("included_in_ensemble", False)
            suffix = " ✓" if incl else " ✗"
        print(f"{model_name:<15} {pr:>10.4f} {roc:>10.4f}{suffix}")
print(f"{'=' * 60}")
