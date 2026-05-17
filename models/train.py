# models/train.py
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, roc_auc_score,
                             confusion_matrix, average_precision_score)
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")

print("FinShield Model Training Started...")


print("\nLoading processed data...")
df = pd.read_csv("data/processed/finshield_processed.csv")
print(f"   Shape: {df.shape}")

print("\nPreparing features...")

DROP_COLS = ["TransactionID", "TransactionDT", "isFraud"]
TARGET    = "isFraud"
feature_cols = [c for c in df.columns if c not in DROP_COLS]
X = df[feature_cols]
y = df[TARGET]

X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

print(f"   Features : {X.shape[1]}")
print(f"   Samples  : {X.shape[0]:,}")
print(f"   Fraud %  : {y.mean()*100:.2f}%")


print("\nSplitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"   Train : {X_train.shape[0]:,} rows")
print(f"   Test  : {X_test.shape[0]:,} rows")


print("\nApplying SMOTE to balance classes...")
print("   This takes 2-3 minutes on 590K rows, please wait...")
smote = SMOTE(random_state=42, sampling_strategy=0.1)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
print(f"   After SMOTE - Fraud: {y_train_bal.sum():,} | Legit: {(y_train_bal==0).sum():,}")


print("\nTraining XGBoost model...")

mlflow.set_experiment("FinShield-Fraud-Detection")

with mlflow.start_run(run_name="XGBoost_v1"):

    params = {
        "n_estimators"     : 300,
        "max_depth"        : 6,
        "learning_rate"    : 0.05,
        "subsample"        : 0.8,
        "colsample_bytree" : 0.8,
        "scale_pos_weight" : 10,
        "use_label_encoder": False,
        "eval_metric"      : "auc",
        "random_state"     : 42,
        "n_jobs"           : -1,
    }

    model = xgb.XGBClassifier(**params)
    model.fit(
        X_train_bal, y_train_bal,
        eval_set=[(X_test, y_test)],
        verbose=100
    )


    
    print("\nEvaluating model...")
    y_pred      = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]

    auc      = roc_auc_score(y_test, y_pred_prob)
    avg_prec = average_precision_score(y_test, y_pred_prob)

    print(f"\n{'='*50}")
    print(f"  AUC-ROC Score         : {auc:.4f}")
    print(f"  Average Precision     : {avg_prec:.4f}")
    print(f"{'='*50}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Legit", "Fraud"]))

    
    mlflow.log_params(params)
    mlflow.log_metric("auc_roc",       auc)
    mlflow.log_metric("avg_precision", avg_prec)
    mlflow.sklearn.log_model(model, "xgboost_fraud_model")

    print(f"\nModel logged to MLflow!")
    print(f"   Open http://localhost:5000 to see experiment")


os.makedirs("models/saved", exist_ok=True)
with open("models/saved/xgboost_model.pkl", "wb") as f:
    pickle.dump(model, f)

# Save feature column names (needed for inference later)
with open("models/saved/feature_cols.pkl", "wb") as f:
    pickle.dump(feature_cols, f)

print(f"\nModel saved to models/saved/xgboost_model.pkl")
print(f"Training complete!")