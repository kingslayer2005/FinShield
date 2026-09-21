"""
airflow/dags/retrain_dag.py
Airflow DAG for FinShield model retraining.

Workflow:
  1. Treat the next time window as newly labelled data
  2. Retrain XGBoost on the expanded training set
  3. Evaluate on a later held-out window
  4. Register in MLflow
  5. Promote to "champion" only if PR-AUC beats the current champion

Also runnable manually:
    python airflow/dags/retrain_dag.py
"""

import os
import sys
import json
import pickle
import logging
from datetime import datetime, timedelta

# Airflow imports (gracefully handle missing airflow for manual runs)
try:
    from airflow import DAG
    from airflow.operators.python import PythonOperator
    HAS_AIRFLOW = True
except ImportError:
    HAS_AIRFLOW = False

# Allow imports from repo root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("finshield.retrain")


def retrain_and_evaluate(**kwargs):
    """Full retraining pipeline as a single task.

    Steps:
      1. Load current champion model and its PR-AUC
      2. Retrain XGBoost on expanded training data
      3. Evaluate on held-out window
      4. Compare PR-AUC: promote only if challenger wins
      5. Log the decision to MLflow and reports/
    """
    import numpy as np
    import pandas as pd
    import xgboost as xgb
    from sklearn.metrics import average_precision_score
    import mlflow
    import mlflow.sklearn
    import yaml

    # Load config
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    is_smoke = os.environ.get("SMOKE", "0") == "1" or cfg.get("smoke", False)

    if is_smoke:
        for key, val in cfg.get("xgboost_smoke", {}).items():
            cfg["xgboost"][key] = val
        processed = os.path.join(cfg["data"]["synthetic_dir"], "processed")
        reports = "reports_smoke"
    else:
        processed = cfg["data"]["processed_dir"]
        reports = "reports"

    os.makedirs(reports, exist_ok=True)

    # Setup MLflow
    try:
        import urllib.request
        urllib.request.urlopen(cfg["mlflow"]["tracking_uri"], timeout=3)
        mlflow_uri = cfg["mlflow"]["tracking_uri"]
    except Exception:
        mlflow_uri = cfg["mlflow"]["fallback_uri"]
        logger.warning("MLflow server unreachable, using local: %s", mlflow_uri)

    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

    # Load data
    logger.info("Loading training data...")
    train_df = pd.read_parquet(os.path.join(processed, "train.parquet"))
    val_df = pd.read_parquet(os.path.join(processed, "val.parquet"))
    test_df = pd.read_parquet(os.path.join(processed, "test.parquet"))

    META_COLS = ["TransactionID", "TransactionDT", "isFraud"]
    feature_cols = [c for c in train_df.columns if c not in META_COLS]

    # Combine train + val as expanded training for retraining
    # Evaluate on test (simulating "later held-out window")
    expanded_train = pd.concat([train_df, val_df], ignore_index=True)
    X_train = expanded_train[feature_cols].astype(np.float32)
    y_train = expanded_train["isFraud"]
    X_test = test_df[feature_cols].astype(np.float32)
    y_test = test_df["isFraud"]

    logger.info("Expanded train: %d rows, Test: %d rows", len(X_train), len(X_test))

    # Load current champion metrics
    champion_prauc = 0.0
    metrics_path = os.path.join(reports, "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            old_metrics = json.load(f)
        champion_prauc = old_metrics.get("ensemble", {}).get("test_pr_auc", 0.0)
    logger.info("Current champion PR-AUC: %.4f", champion_prauc)

    # Train challenger
    xgb_cfg = cfg["xgboost"]
    params = {
        "n_estimators": xgb_cfg["n_estimators"],
        "max_depth": xgb_cfg["max_depth"],
        "learning_rate": xgb_cfg["learning_rate"],
        "subsample": xgb_cfg["subsample"],
        "colsample_bytree": xgb_cfg["colsample_bytree"],
        "eval_metric": xgb_cfg["eval_metric"],
        "random_state": xgb_cfg["random_state"],
        "n_jobs": xgb_cfg["n_jobs"],
    }

    # Compute scale_pos_weight
    n_legit = (y_train == 0).sum()
    n_fraud = max((y_train == 1).sum(), 1)
    params["scale_pos_weight"] = n_legit / n_fraud

    logger.info("Training challenger XGBoost...")
    challenger = xgb.XGBClassifier(**params)
    challenger.fit(X_train, y_train, verbose=0)

    # Evaluate challenger on test
    challenger_probs = challenger.predict_proba(X_test)[:, 1]
    challenger_prauc = average_precision_score(y_test, challenger_probs)
    logger.info("Challenger PR-AUC: %.4f", challenger_prauc)

    # Decision: promote or not
    promoted = challenger_prauc > champion_prauc
    decision = {
        "timestamp": datetime.now().isoformat(),
        "champion_pr_auc": float(champion_prauc),
        "challenger_pr_auc": float(challenger_prauc),
        "promoted": promoted,
        "is_smoke": is_smoke,
    }

    # Log to MLflow
    with mlflow.start_run(run_name="Retrain_challenger"):
        mlflow.log_params(params)
        mlflow.log_metric("test_pr_auc", challenger_prauc)
        mlflow.log_param("promoted", promoted)
        if promoted:
            mlflow.sklearn.log_model(challenger, "xgboost_model")

    # Save decision
    retrain_log_path = os.path.join(reports, "retrain_log.json")
    with open(retrain_log_path, "w") as f:
        json.dump(decision, f, indent=2)

    if promoted:
        logger.info("PROMOTED — challenger beats champion (%.4f > %.4f)",
                     challenger_prauc, champion_prauc)
        # Save new champion
        with open("models/saved/xgboost_model.pkl", "wb") as f:
            pickle.dump(challenger, f)
        logger.info("New champion saved to models/saved/xgboost_model.pkl")
    else:
        logger.info("NOT PROMOTED — champion still wins (%.4f >= %.4f)",
                     champion_prauc, challenger_prauc)

    return decision


# ---------------------------------------------------------------------------
# Airflow DAG definition
# ---------------------------------------------------------------------------
if HAS_AIRFLOW:
    default_args = {
        "owner": "finshield",
        "depends_on_past": False,
        "email_on_failure": False,
        "retries": 1,
        "retry_delay": timedelta(minutes=5),
    }

    dag = DAG(
        "finshield_retrain",
        default_args=default_args,
        description="Retrain FinShield fraud model and promote if better",
        schedule_interval=timedelta(days=7),  # weekly
        start_date=datetime(2024, 1, 1),
        catchup=False,
        tags=["finshield", "ml", "retraining"],
    )

    retrain_task = PythonOperator(
        task_id="retrain_and_evaluate",
        python_callable=retrain_and_evaluate,
        dag=dag,
    )


# ---------------------------------------------------------------------------
# Manual execution
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logger.info("Running retraining pipeline manually...")
    result = retrain_and_evaluate()
    logger.info("Result: %s", json.dumps(result, indent=2))
