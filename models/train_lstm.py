"""
models/train_lstm.py
Trains an LSTM on transaction sequences using only the top N XGBoost
features (default 30) to limit RAM.  Uses tf.data generators so
sequences never need to all fit in memory at once.

CORRECTION 6: Sequences assigned to train/val/test by the timestamp of the
transaction being predicted, not by card.  History may come from earlier periods.
CORRECTION 5: Early stopping uses inner-validation, not main val.
CORRECTION 4: Uses scaled data (NaN imputed + StandardScaled).

Run from repo root:
    python models/train_lstm.py
    SMOKE=1 python models/train_lstm.py
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

from utils.config import cfg, is_smoke, get_processed_dir, get_mlflow_uri

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import average_precision_score, roc_auc_score
import mlflow

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PROCESSED = get_processed_dir()
LSTM_CFG = cfg["lstm"]
SEQ_LEN = cfg["features"]["lstm_seq_len"]  # max past transactions for sequence
TOP_N = LSTM_CFG.get("top_features", 30)   # use top N XGBoost features

mlflow_uri = get_mlflow_uri()
mlflow.set_tracking_uri(mlflow_uri)
mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

# ---------------------------------------------------------------------------
# Load XGBoost feature importances to select top N features
# ---------------------------------------------------------------------------
print("=" * 60)
print("FinShield LSTM Training")
if is_smoke:
    print("  MODE: SMOKE")
print("=" * 60)

print("\n[1/7] Loading XGBoost feature importances...")
with open("models/saved/xgboost_model.pkl", "rb") as f:
    xgb_model = pickle.load(f)
with open("models/saved/feature_cols.pkl", "rb") as f:
    all_feature_cols = pickle.load(f)

# Get feature importances and pick top N
importances = xgb_model.feature_importances_
importance_pairs = sorted(zip(all_feature_cols, importances),
                          key=lambda x: x[1], reverse=True)
lstm_features = [feat for feat, _ in importance_pairs[:TOP_N]]
print(f"   Using top {len(lstm_features)} features by XGBoost importance")
print(f"   Top 5: {[f for f, _ in importance_pairs[:5]]}")

# Save LSTM feature list for inference
with open("models/saved/lstm_features.pkl", "wb") as f:
    pickle.dump(lstm_features, f)

# ---------------------------------------------------------------------------
# Load full scaled dataset sorted by time
# ---------------------------------------------------------------------------
print("\n[2/7] Loading scaled data...")
# Load all scaled splits and concatenate to build sequences across boundaries
train_scaled = pd.read_parquet(os.path.join(PROCESSED, "train_inner_scaled.parquet"))
ival_scaled = pd.read_parquet(os.path.join(PROCESSED, "inner_val_scaled.parquet"))
val_scaled = pd.read_parquet(os.path.join(PROCESSED, "val_scaled.parquet"))

# Sort each by time (should already be sorted, but be safe)
train_scaled = train_scaled.sort_values("TransactionDT").reset_index(drop=True)
ival_scaled = ival_scaled.sort_values("TransactionDT").reset_index(drop=True)

# Get time boundaries for split assignment
train_dt_max = train_scaled["TransactionDT"].max()
ival_dt_min = ival_scaled["TransactionDT"].min()

# Combine all data sorted by time for sequence building
all_data = pd.concat([train_scaled, ival_scaled], ignore_index=True)
all_data = all_data.sort_values("TransactionDT").reset_index(drop=True)

X_all = all_data[lstm_features].astype(np.float32).values
y_all = all_data["isFraud"].values
dt_all = all_data["TransactionDT"].values

print(f"   Combined rows: {len(all_data):,}")
print(f"   LSTM features: {len(lstm_features)}")

# ---------------------------------------------------------------------------
# Build sequences with a generator (memory-efficient)
# ---------------------------------------------------------------------------
print("\n[3/7] Building sequences with tf.data generator...")


def make_sequences_split(X: np.ndarray, y: np.ndarray, dt: np.ndarray,
                         dt_min: float, dt_max: float, seq_len: int):
    """Build LSTM sequences for rows whose prediction timestamp is in [dt_min, dt_max].

    CORRECTION 6: Each sequence is assigned to a split by the timestamp of the
    transaction being predicted (the last row).  The history (earlier rows in
    the sequence) may come from previous time periods — this is real history.

    Uses a generator to avoid materialising all sequences in RAM.

    Args:
        X: Feature array for all rows, sorted by time.
        y: Label array for all rows.
        dt: TransactionDT array for all rows.
        dt_min: Minimum DT for this split (inclusive).
        dt_max: Maximum DT for this split (inclusive).
        seq_len: Number of past transactions in each sequence.

    Yields:
        (sequence, label) tuples where sequence is (seq_len, n_features).
    """
    for i in range(seq_len, len(X)):
        # The transaction being predicted is at index i
        if dt[i] < dt_min or dt[i] > dt_max:
            continue  # skip rows outside this split's time range

        # Build sequence from the preceding seq_len rows
        seq = X[i - seq_len:i]  # shape: (seq_len, n_features)
        label = y[i]
        yield seq, label


def generator_to_dataset(X, y, dt, dt_min, dt_max, seq_len, n_features, batch_size):
    """Wrap the sequence generator as a tf.data.Dataset for efficient training.

    This avoids loading all sequences into RAM at once.
    """
    def gen():
        yield from make_sequences_split(X, y, dt, dt_min, dt_max, seq_len)

    ds = tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec(shape=(seq_len, n_features), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int32),
        ),
    )
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


n_features = len(lstm_features)
batch_size = LSTM_CFG["batch_size"]

# Training sequences: prediction timestamp in inner-train range
train_ds = generator_to_dataset(
    X_all, y_all, dt_all,
    dt_min=float(dt_all[SEQ_LEN]),  # first valid sequence
    dt_max=float(train_dt_max),
    seq_len=SEQ_LEN,
    n_features=n_features,
    batch_size=batch_size,
)

# Inner-val sequences: prediction timestamp in inner-val range
ival_ds = generator_to_dataset(
    X_all, y_all, dt_all,
    dt_min=float(ival_dt_min),
    dt_max=float(dt_all[-1]),
    seq_len=SEQ_LEN,
    n_features=n_features,
    batch_size=batch_size,
)

# Count sequences for reporting
n_train_seq = sum(1 for _ in make_sequences_split(
    X_all, y_all, dt_all, float(dt_all[SEQ_LEN]), float(train_dt_max), SEQ_LEN))
n_ival_seq = sum(1 for _ in make_sequences_split(
    X_all, y_all, dt_all, float(ival_dt_min), float(dt_all[-1]), SEQ_LEN))
print(f"   Train sequences: {n_train_seq:,}")
print(f"   Inner-val sequences: {n_ival_seq:,}")

# ---------------------------------------------------------------------------
# Build LSTM model
# ---------------------------------------------------------------------------
print("\n[4/7] Building LSTM architecture...")
hidden_units = LSTM_CFG["hidden_units"]
dense_units = LSTM_CFG["dense_units"]
dropout = LSTM_CFG["dropout"]

model = Sequential(name="fraud_lstm")
for i, units in enumerate(hidden_units):
    # return_sequences=True for all but the last LSTM layer
    return_seq = i < len(hidden_units) - 1
    if i == 0:
        model.add(LSTM(units, input_shape=(SEQ_LEN, n_features),
                        return_sequences=return_seq, name=f"lstm_{i}"))
    else:
        model.add(LSTM(units, return_sequences=return_seq, name=f"lstm_{i}"))
    model.add(BatchNormalization(name=f"bn_{i}"))
    model.add(Dropout(dropout, name=f"drop_{i}"))

model.add(Dense(dense_units, activation="relu", name="dense_1"))
model.add(Dense(1, activation="sigmoid", name="output"))

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=[tf.keras.metrics.AUC(name="auc")],
)
model.summary()

# ---------------------------------------------------------------------------
# Train LSTM
# ---------------------------------------------------------------------------
print("\n[5/7] Training LSTM...")

callbacks = [
    EarlyStopping(
        patience=LSTM_CFG["patience"],
        restore_best_weights=True,
        verbose=1,
        monitor="val_loss",
    ),
]

class_weight_fraud = LSTM_CFG["class_weight_fraud"]

# Recreate datasets (generators are exhausted after counting)
train_ds = generator_to_dataset(
    X_all, y_all, dt_all,
    dt_min=float(dt_all[SEQ_LEN]),
    dt_max=float(train_dt_max),
    seq_len=SEQ_LEN, n_features=n_features, batch_size=batch_size,
)
ival_ds = generator_to_dataset(
    X_all, y_all, dt_all,
    dt_min=float(ival_dt_min),
    dt_max=float(dt_all[-1]),
    seq_len=SEQ_LEN, n_features=n_features, batch_size=batch_size,
)

with mlflow.start_run(run_name="LSTM_v2"):
    history = model.fit(
        train_ds,
        epochs=LSTM_CFG["epochs"],
        validation_data=ival_ds,
        callbacks=callbacks,
        class_weight={0: 1, 1: class_weight_fraud},
        verbose=1,
    )

    # -------------------------------------------------------------------
    # Evaluate on main validation set
    # -------------------------------------------------------------------
    print("\n[6/7] Evaluating on validation set...")

    # Build val sequences from combined data including val period
    val_full = pd.concat([train_scaled, ival_scaled, val_scaled], ignore_index=True)
    val_full = val_full.sort_values("TransactionDT").reset_index(drop=True)
    X_vf = val_full[lstm_features].astype(np.float32).values
    y_vf = val_full["isFraud"].values
    dt_vf = val_full["TransactionDT"].values

    val_dt_min = float(val_scaled["TransactionDT"].min())
    val_dt_max = float(val_scaled["TransactionDT"].max())

    # Collect predictions for val period
    val_preds = []
    val_labels = []
    for seq, label in make_sequences_split(
            X_vf, y_vf, dt_vf, val_dt_min, val_dt_max, SEQ_LEN):
        val_preds.append(model.predict(seq[np.newaxis], verbose=0)[0, 0])
        val_labels.append(label)

    val_preds = np.array(val_preds)
    val_labels = np.array(val_labels)

    if len(val_labels) > 0 and val_labels.sum() > 0:
        val_pr_auc = average_precision_score(val_labels, val_preds)
        val_roc_auc = roc_auc_score(val_labels, val_preds)
    else:
        val_pr_auc = 0.0
        val_roc_auc = 0.0

    print(f"   Val PR-AUC:  {val_pr_auc:.4f}")
    print(f"   Val ROC-AUC: {val_roc_auc:.4f}")
    print(f"   Val sequences evaluated: {len(val_labels)}")

    mlflow.log_metric("val_pr_auc", val_pr_auc)
    mlflow.log_metric("val_roc_auc", val_roc_auc)
    mlflow.log_params({
        "hidden_units": str(hidden_units),
        "dense_units": dense_units,
        "dropout": dropout,
        "seq_len": SEQ_LEN,
        "top_features": TOP_N,
        "epochs_trained": len(history.history["loss"]),
    })

    # -------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------
    print("\n[7/7] Saving LSTM model...")
    os.makedirs("models/saved", exist_ok=True)
    model.save("models/saved/lstm_model.keras")

    lstm_metrics = {
        "val_pr_auc": float(val_pr_auc),
        "val_roc_auc": float(val_roc_auc),
        "epochs_trained": len(history.history["loss"]),
        "n_features": n_features,
        "seq_len": SEQ_LEN,
    }
    with open("models/saved/lstm_metrics.json", "w") as f:
        json.dump(lstm_metrics, f, indent=2)

    print(f"\n{'=' * 60}")
    print("LSTM training complete!")
    print(f"   Val PR-AUC : {val_pr_auc:.4f}")
    print(f"   Val ROC-AUC: {val_roc_auc:.4f}")
    print(f"{'=' * 60}")
