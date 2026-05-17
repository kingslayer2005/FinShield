# models/lstm_model.py
import pandas as pd
import numpy as np
import pickle
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler
import mlflow
import warnings
warnings.filterwarnings("ignore")

print("FinShield LSTM Training Started...")


print("\nLoading data...")
df = pd.read_csv("data/processed/finshield_processed.csv")

DROP_COLS    = ["TransactionID", "TransactionDT", "isFraud"]
TARGET       = "isFraud"
feature_cols = [c for c in df.columns if c not in DROP_COLS]

X = df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
y = df[TARGET]


print("Scaling features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# Group by card1 (user) and create sequences of 5 transactions
# LSTM learns: "given last 5 transactions, is this one fraud?"
print("Creating transaction sequences...")
print("   This takes 2-3 minutes...")

SEQ_LEN = 5
X_seq, y_seq = [], []

# Use sample for speed (20K sequences)
df_sample = df.sample(n=50000, random_state=42).reset_index(drop=True)
X_sample  = X_scaled[df_sample.index]
y_sample  = y.iloc[df_sample.index].values

for i in range(SEQ_LEN, len(X_sample)):
    X_seq.append(X_sample[i-SEQ_LEN:i])
    y_seq.append(y_sample[i])

X_seq = np.array(X_seq)
y_seq = np.array(y_seq)

print(f"   Sequences shape : {X_seq.shape}")
print(f"   Fraud rate      : {y_seq.mean()*100:.2f}%")


split = int(0.8 * len(X_seq))
X_train, X_test = X_seq[:split], X_seq[split:]
y_train, y_test = y_seq[:split], y_seq[split:]


print("\nBuilding LSTM architecture...")
input_shape = (SEQ_LEN, X.shape[1])

model = Sequential([
    LSTM(64, input_shape=input_shape, return_sequences=True),
    BatchNormalization(),
    Dropout(0.3),
    LSTM(32, return_sequences=False),
    BatchNormalization(),
    Dropout(0.3),
    Dense(16, activation="relu"),
    Dense(1,  activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["auc"]
)
model.summary()


print("\nTraining LSTM...")

callbacks = [
    EarlyStopping(patience=3, restore_best_weights=True, verbose=1),
    ModelCheckpoint("models/saved/lstm_model.keras",
                    save_best_only=True, verbose=0)
]

mlflow.set_experiment("FinShield-Fraud-Detection")
with mlflow.start_run(run_name="LSTM_v1"):

    history = model.fit(
        X_train, y_train,
        epochs          = 15,
        batch_size      = 256,
        validation_split= 0.1,
        callbacks       = callbacks,
        class_weight    = {0: 1, 1: 10},
        verbose         = 1
    )


    print("\nEvaluating LSTM...")
    y_pred_prob = model.predict(X_test, verbose=0).flatten()
    y_pred      = (y_pred_prob > 0.5).astype(int)

    auc = roc_auc_score(y_test, y_pred_prob)

    print(f"\n{'='*50}")
    print(f"  LSTM AUC-ROC          : {auc:.4f}")
    print(f"{'='*50}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred,
                                target_names=["Legit", "Fraud"]))

    mlflow.log_metric("auc_roc", auc)

    with open("models/saved/lstm_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    print(f"\nLSTM saved to models/saved/lstm_model.keras")
    print(f"Logged to MLflow — http://localhost:5000")