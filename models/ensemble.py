# models/ensemble.py
import pandas as pd
import numpy as np
import pickle
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.keras.models import load_model
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import mlflow
import warnings
warnings.filterwarnings("ignore")

print("🚀 FinShield Ensemble Model Starting...")

print("\nLoading all 3 models...")

with open("models/saved/xgboost_model.pkl", "rb") as f:
    xgb_model = pickle.load(f)

autoencoder = load_model("models/saved/autoencoder.keras")

lstm_model  = load_model("models/saved/lstm_model.keras")

with open("models/saved/scaler.pkl", "rb") as f:
    ae_scaler = pickle.load(f)

with open("models/saved/lstm_scaler.pkl", "rb") as f:
    lstm_scaler = pickle.load(f)

with open("models/saved/ae_threshold.pkl", "rb") as f:
    ae_threshold = pickle.load(f)

with open("models/saved/feature_cols.pkl", "rb") as f:
    feature_cols = pickle.load(f)

print("XGBoost loaded")
print("Autoencoder loaded")
print("LSTM loaded")


print("\nLoading test data...")
df = pd.read_csv("data/processed/finshield_sample.csv")

TARGET = "isFraud"
X = df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
y = df[TARGET]

print(f"   Sample size : {len(df):,}")
print(f"   Fraud rate  : {y.mean()*100:.2f}%")


print("\nGetting predictions from each model...")


xgb_probs = xgb_model.predict_proba(X)[:, 1]
print(f"   XGBoost done  — mean fraud prob: {xgb_probs.mean():.4f}")


X_scaled_ae   = ae_scaler.transform(X)
X_pred_ae     = autoencoder.predict(X_scaled_ae, verbose=0)
mse_errors    = np.mean(np.power(X_scaled_ae - X_pred_ae, 2), axis=1)
ae_probs      = mse_errors / (mse_errors.max() + 1e-10)
print(f"   Autoencoder done — mean fraud prob: {ae_probs.mean():.4f}")


X_scaled_lstm = lstm_scaler.transform(X)
SEQ_LEN       = 5
X_lstm_seq    = np.array([
    X_scaled_lstm[max(0, i-SEQ_LEN):i+1].mean(axis=0)
    for i in range(len(X_scaled_lstm))
])
X_lstm_seq    = X_lstm_seq.reshape(len(X_lstm_seq), 1, X.shape[1])


X_lstm_padded = np.zeros((len(X), SEQ_LEN, X.shape[1]))
for i in range(len(X)):
    X_lstm_padded[i] = np.tile(X_scaled_lstm[i], (SEQ_LEN, 1))

lstm_probs    = lstm_model.predict(X_lstm_padded, verbose=0).flatten()
print(f"   LSTM done     — mean fraud prob: {lstm_probs.mean():.4f}")

# XGBoost gets highest weight (best AUC)
# Autoencoder medium weight (catches novel fraud)
# LSTM lowest weight (weakest on this data)
print("\nComputing weighted ensemble...")

W_XGB = 0.70
W_AE  = 0.20
W_LSTM= 0.10

ensemble_probs = (W_XGB * xgb_probs +
                  W_AE  * ae_probs   +
                  W_LSTM* lstm_probs)

# Best threshold via ROC curve
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y, ensemble_probs)
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]

y_pred_ensemble = (ensemble_probs > optimal_threshold).astype(int)


print("\nEnsemble Results:")
auc = roc_auc_score(y, ensemble_probs)

print(f"\n{'='*55}")
print(f"  Model          AUC      Weight")
print(f"  {'─'*40}")
print(f"  XGBoost        0.9259   {W_XGB}")
print(f"  Autoencoder    0.6797   {W_AE}")
print(f"  LSTM           0.5267   {W_LSTM}")
print(f"  {'─'*40}")
print(f" ENSEMBLE    {auc:.4f}   Combined")
print(f"{'='*55}")
print(f"\nClassification Report (Ensemble):")
print(classification_report(y, y_pred_ensemble,
                            target_names=["Legit", "Fraud"]))


ensemble_config = {
    "weights"           : {"xgb": W_XGB, "ae": W_AE, "lstm": W_LSTM},
    "optimal_threshold" : optimal_threshold,
    "feature_cols"      : feature_cols,
    "ae_threshold"      : ae_threshold
}

with open("models/saved/ensemble_config.pkl", "wb") as f:
    pickle.dump(ensemble_config, f)


mlflow.set_experiment("FinShield-Fraud-Detection")
with mlflow.start_run(run_name="Ensemble_v1"):
    mlflow.log_metric("ensemble_auc",      auc)
    mlflow.log_metric("xgb_weight",        W_XGB)
    mlflow.log_metric("ae_weight",         W_AE)
    mlflow.log_metric("lstm_weight",       W_LSTM)
    mlflow.log_metric("optimal_threshold", optimal_threshold)

print(f"\nEnsemble config saved to models/saved/ensemble_config.pkl")
print(f"Logged to MLflow!")
print(f"\nAll 3 models combined into FinShield Ensemble!")