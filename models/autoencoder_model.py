# models/autoencoder_model.py
import pandas as pd
import numpy as np
import pickle
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report
import mlflow
import warnings
warnings.filterwarnings("ignore")

print("FinShield Autoencoder Training Started...")


print("\nLoading data...")
df = pd.read_csv("data/processed/finshield_processed.csv")

DROP_COLS  = ["TransactionID", "TransactionDT", "isFraud"]
TARGET     = "isFraud"
feature_cols = [c for c in df.columns if c not in DROP_COLS]

X = df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
y = df[TARGET]


# Autoencoder learns "normal" — anything it can't reconstruct = fraud
print("Training only on LEGIT transactions...")
X_legit = X[y == 0]
X_fraud  = X[y == 1]
print(f"   Legit samples : {len(X_legit):,}")
print(f"   Fraud samples : {len(X_fraud):,}")


scaler  = StandardScaler()
X_legit_scaled = scaler.fit_transform(X_legit)
X_all_scaled   = scaler.transform(X)

# Save scaler
os.makedirs("models/saved", exist_ok=True)
with open("models/saved/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("\nBuilding Autoencoder architecture...")
input_dim = X.shape[1]

inputs   = Input(shape=(input_dim,))

encoded  = Dense(256, activation="relu")(inputs)
encoded  = Dropout(0.2)(encoded)
encoded  = Dense(128, activation="relu")(encoded)
encoded  = Dropout(0.2)(encoded)
encoded  = Dense(64,  activation="relu")(encoded)
bottleneck = Dense(32, activation="relu")(encoded)

# Decoder
decoded  = Dense(64,  activation="relu")(bottleneck)
decoded  = Dropout(0.2)(decoded)
decoded  = Dense(128, activation="relu")(decoded)
decoded  = Dropout(0.2)(decoded)
decoded  = Dense(256, activation="relu")(decoded)
outputs  = Dense(input_dim, activation="linear")(decoded)

autoencoder = Model(inputs, outputs)
autoencoder.compile(optimizer="adam", loss="mse")
autoencoder.summary()

print("\nTraining Autoencoder on legit transactions...")

os.makedirs("models/saved", exist_ok=True)
callbacks = [
    EarlyStopping(patience=3, restore_best_weights=True, verbose=1),
    ModelCheckpoint("models/saved/autoencoder.keras",
                    save_best_only=True, verbose=0)
]

mlflow.set_experiment("FinShield-Fraud-Detection")
with mlflow.start_run(run_name="Autoencoder_v1"):

    history = autoencoder.fit(
        X_legit_scaled, X_legit_scaled,
        epochs          = 20,
        batch_size      = 512,
        validation_split= 0.1,
        callbacks       = callbacks,
        verbose         = 1
    )


    print("\nComputing reconstruction errors...")
    X_pred     = autoencoder.predict(X_all_scaled, verbose=0)
    mse_errors = np.mean(np.power(X_all_scaled - X_pred, 2), axis=1)

    # Find best threshold using percentile
    threshold  = np.percentile(mse_errors[y == 0], 95)
    y_pred_ae  = (mse_errors > threshold).astype(int)

    auc = roc_auc_score(y, mse_errors)
    print(f"\n{'='*50}")
    print(f"  Autoencoder AUC-ROC   : {auc:.4f}")
    print(f"  Threshold (95th pct)  : {threshold:.6f}")
    print(f"{'='*50}")
    print(f"\nClassification Report:")
    print(classification_report(y, y_pred_ae,
                                target_names=["Legit", "Fraud"]))

    mlflow.log_metric("auc_roc",   auc)
    mlflow.log_metric("threshold", threshold)

    with open("models/saved/ae_threshold.pkl", "wb") as f:
        pickle.dump(threshold, f)

    print(f"\nAutoencoder saved to models/saved/autoencoder.keras")
    print(f"Logged to MLflow — http://localhost:5000")