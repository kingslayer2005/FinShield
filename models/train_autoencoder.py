"""
models/train_autoencoder.py
Trains an autoencoder on LEGIT transactions from train-inner (scaled).
Uses inner-validation for early stopping.  Calibrates reconstruction
error into a [0,1] anomaly score using percentile rank against
validation legit MSE distribution.

CORRECTION 4: Uses pre-imputed + scaled data (*_scaled.parquet).
CORRECTION 5: Early stopping on inner_val, not main validation set.

Run from repo root:
    python models/train_autoencoder.py
    SMOKE=1 python models/train_autoencoder.py
"""

import pandas as pd
import numpy as np
import pickle
import os
import sys
import warnings

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # suppress TF info/warning logs

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.config import cfg, is_smoke, get_processed_dir, get_mlflow_uri

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import average_precision_score, roc_auc_score
import mlflow

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PROCESSED = get_processed_dir()
AE_CFG = cfg["autoencoder"]

mlflow_uri = get_mlflow_uri()
mlflow.set_tracking_uri(mlflow_uri)
mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

# ---------------------------------------------------------------------------
# Load scaled data (NaN already imputed + StandardScaled)
# ---------------------------------------------------------------------------
print("=" * 60)
print("FinShield Autoencoder Training")
if is_smoke:
    print("  MODE: SMOKE")
print("=" * 60)

print("\n[1/6] Loading scaled data...")
train_inner = pd.read_parquet(os.path.join(PROCESSED, "train_inner_scaled.parquet"))
inner_val = pd.read_parquet(os.path.join(PROCESSED, "inner_val_scaled.parquet"))
val_scaled = pd.read_parquet(os.path.join(PROCESSED, "val_scaled.parquet"))

# Feature columns: everything except ID, time, label
META_COLS = ["TransactionID", "TransactionDT", "isFraud"]
feature_cols = [c for c in train_inner.columns if c not in META_COLS]

X_train_all = train_inner[feature_cols].astype(np.float32).values
y_train = train_inner["isFraud"].values
X_ival = inner_val[feature_cols].astype(np.float32).values
y_ival = inner_val["isFraud"].values
X_val = val_scaled[feature_cols].astype(np.float32).values
y_val = val_scaled["isFraud"].values if "isFraud" in val_scaled.columns else None

# Autoencoder trains ONLY on legit transactions (learns "normal" patterns)
X_legit = X_train_all[y_train == 0]
print(f"   Train legit rows: {len(X_legit):,}")
print(f"   Inner val rows:   {len(X_ival):,}")

# Also prepare inner-val legit for AE validation loss
X_ival_legit = X_ival[y_ival == 0]
print(f"   Inner val legit:  {len(X_ival_legit):,}")

# ---------------------------------------------------------------------------
# Build autoencoder architecture
# ---------------------------------------------------------------------------
print("\n[2/6] Building autoencoder...")
input_dim = len(feature_cols)
layers = AE_CFG["layer_sizes"]    # e.g. [256, 128, 64, 32] or [32, 16] in smoke
drop_rate = AE_CFG["dropout"]

# Encoder: progressively compress the input
inputs = Input(shape=(input_dim,), name="encoder_input")
x = inputs
for i, units in enumerate(layers):
    x = Dense(units, activation="relu", name=f"encoder_{i}")(x)
    if i < len(layers) - 1:  # no dropout on bottleneck
        x = Dropout(drop_rate, name=f"encoder_drop_{i}")(x)

# Decoder: mirror the encoder to reconstruct the input
for i, units in enumerate(reversed(layers[:-1])):
    x = Dense(units, activation="relu", name=f"decoder_{i}")(x)
    x = Dropout(drop_rate, name=f"decoder_drop_{i}")(x)

# Output layer: same dimension as input, linear activation for reconstruction
outputs = Dense(input_dim, activation="linear", name="decoder_output")(x)

autoencoder = Model(inputs, outputs, name="fraud_autoencoder")
autoencoder.compile(optimizer="adam", loss="mse")
autoencoder.summary()

# ---------------------------------------------------------------------------
# Train on legit data only, early stopping on inner-val legit
# ---------------------------------------------------------------------------
print("\n[3/6] Training autoencoder on legit transactions...")

callbacks = [
    EarlyStopping(
        patience=AE_CFG["patience"],
        restore_best_weights=True,
        verbose=1,
        monitor="val_loss",
    ),
]

with mlflow.start_run(run_name="Autoencoder_v2"):
    history = autoencoder.fit(
        X_legit, X_legit,               # input = target (reconstruction)
        epochs=AE_CFG["epochs"],
        batch_size=AE_CFG["batch_size"],
        validation_data=(X_ival_legit, X_ival_legit),  # inner-val for early stop
        callbacks=callbacks,
        verbose=1,
    )

    # -------------------------------------------------------------------
    # Calibrate scores on main validation set
    # -------------------------------------------------------------------
    print("\n[4/6] Calibrating anomaly scores...")

    # Compute MSE per row on val set
    val_recon = autoencoder.predict(X_val, verbose=0)
    val_mse = np.mean(np.power(X_val - val_recon, 2), axis=1)

    # Reference distribution: MSE of validation legit rows
    val_legit_mse = np.sort(val_mse[y_val == 0])

    def mse_to_percentile(mse_values: np.ndarray,
                          reference_dist: np.ndarray) -> np.ndarray:
        """Convert MSE to percentile rank against reference distribution.

        Higher = more anomalous. Returns values in [0, 1].
        """
        return np.searchsorted(reference_dist, mse_values) / len(reference_dist)

    val_scores = mse_to_percentile(val_mse, val_legit_mse)

    # Evaluate calibrated scores
    val_pr_auc = average_precision_score(y_val, val_scores)
    val_roc_auc = roc_auc_score(y_val, val_scores)

    print(f"   Val PR-AUC  (calibrated): {val_pr_auc:.4f}")
    print(f"   Val ROC-AUC (calibrated): {val_roc_auc:.4f}")

    mlflow.log_metric("val_pr_auc", val_pr_auc)
    mlflow.log_metric("val_roc_auc", val_roc_auc)
    mlflow.log_params({
        "layer_sizes": str(AE_CFG["layer_sizes"]),
        "dropout": AE_CFG["dropout"],
        "epochs_trained": len(history.history["loss"]),
    })

    # -------------------------------------------------------------------
    # Save model and calibration data
    # -------------------------------------------------------------------
    print("\n[5/6] Saving model and calibration...")

    os.makedirs("models/saved", exist_ok=True)
    autoencoder.save("models/saved/autoencoder.keras")
    print("   Saved autoencoder.keras")

    # Save the sorted legit MSE distribution for percentile calibration
    with open("models/saved/ae_calibration.pkl", "wb") as f:
        pickle.dump(val_legit_mse, f)
    print("   Saved ae_calibration.pkl (val legit MSE distribution)")

    # Save AE val metrics for model selection
    ae_metrics = {
        "val_pr_auc": float(val_pr_auc),
        "val_roc_auc": float(val_roc_auc),
        "epochs_trained": len(history.history["loss"]),
    }
    with open("models/saved/ae_metrics.json", "w") as f:
        import json
        json.dump(ae_metrics, f, indent=2)

    print(f"\n{'=' * 60}")
    print("Autoencoder training complete!")
    print(f"   Val PR-AUC : {val_pr_auc:.4f}")
    print(f"   Val ROC-AUC: {val_roc_auc:.4f}")
    print(f"{'=' * 60}")
