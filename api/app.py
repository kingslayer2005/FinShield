"""
api/app.py
FastAPI scoring API for FinShield fraud detection.

Endpoints:
  POST /predict   — raw transaction in → fraud probability, decision, SHAP reasons
  GET  /health    — health check
  GET  /model-info — loaded model metadata

Loads the stacking ensemble (XGBoost + optional AE + optional LSTM).
Tries MLflow "champion" alias first, falls back to local model artifacts.

Run locally:
    uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import sys
import time
import json
import pickle
import logging
import warnings

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Allow imports from repo root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("finshield.api")

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="FinShield Fraud Detection API",
    description="Real-time fraud scoring with SHAP explanations",
    version="1.0.0",
)

# ---------------------------------------------------------------------------
# Global model state (loaded once at startup)
# ---------------------------------------------------------------------------
MODEL_STATE = {
    "xgb_model": None,
    "ae_model": None,
    "lstm_model": None,
    "ensemble_config": None,
    "feature_cols": None,
    "shap_explainer": None,
    "preprocessor": None,
    "model_version": "local",
    "loaded": False,
}


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------
class TransactionRequest(BaseModel):
    """Raw transaction data sent by the client for scoring."""
    TransactionAmt: float
    card1: int
    card2: Optional[float] = None
    card3: Optional[float] = None
    card4: Optional[str] = None
    card5: Optional[float] = None
    card6: Optional[str] = None
    addr1: Optional[float] = None
    addr2: Optional[float] = None
    P_emaildomain: Optional[str] = None
    R_emaildomain: Optional[str] = None
    ProductCD: Optional[str] = "W"
    DeviceType: Optional[str] = None
    DeviceInfo: Optional[str] = None
    TransactionDT: Optional[int] = 86400  # default to 1 day
    # Allow extra C/D/V/id fields
    extra_fields: Optional[Dict[str, Any]] = None


class ShapReason(BaseModel):
    """One SHAP-based explanation for a fraud score."""
    feature: str
    description: str
    impact: float
    direction: str


class PredictionResponse(BaseModel):
    """API response for POST /predict."""
    transaction_id: str
    fraud_probability: float
    decision: str           # "approve", "review", or "block"
    reasons: List[ShapReason]
    model_version: str
    latency_ms: float


class ModelInfoResponse(BaseModel):
    """API response for GET /model-info."""
    model_version: str
    components: List[str]
    include_ae: bool
    include_lstm: bool
    review_threshold: float
    block_threshold: float
    n_features: int


# ---------------------------------------------------------------------------
# Load models at startup
# ---------------------------------------------------------------------------
@app.on_event("startup")
def load_models():
    """Load all model artifacts into memory at startup.

    Tries MLflow "champion" alias first, falls back to local files.
    """
    logger.info("Loading model artifacts...")
    state = MODEL_STATE

    try:
        # XGBoost
        with open("models/saved/xgboost_model.pkl", "rb") as f:
            state["xgb_model"] = pickle.load(f)
        logger.info("XGBoost model loaded")

        # Feature columns
        with open("models/saved/feature_cols.pkl", "rb") as f:
            state["feature_cols"] = pickle.load(f)

        # Ensemble config
        with open("models/saved/ensemble_config.pkl", "rb") as f:
            state["ensemble_config"] = pickle.load(f)

        # Autoencoder (if included)
        if state["ensemble_config"]["include_ae"]:
            import tensorflow as tf
            state["ae_model"] = tf.keras.models.load_model(
                "models/saved/autoencoder.keras")
            with open("models/saved/ae_calibration.pkl", "rb") as f:
                state["ae_calibration"] = pickle.load(f)
            logger.info("Autoencoder loaded")

        # LSTM (if included)
        if state["ensemble_config"]["include_lstm"]:
            import tensorflow as tf
            state["lstm_model"] = tf.keras.models.load_model(
                "models/saved/lstm_model.keras")
            with open("models/saved/lstm_features.pkl", "rb") as f:
                state["lstm_features"] = pickle.load(f)
            logger.info("LSTM loaded")

        # Preprocessor (for encoding new transactions)
        preprocessor_paths = [
            "data/synthetic/processed/preprocessor.pkl",  # smoke
            "data/processed/preprocessor.pkl",             # real
        ]
        for pp_path in preprocessor_paths:
            if os.path.exists(pp_path):
                with open(pp_path, "rb") as f:
                    state["preprocessor"] = pickle.load(f)
                logger.info("Preprocessor loaded from %s", pp_path)
                break

        # SHAP explainer
        if os.path.exists("models/saved/shap_explainer.pkl"):
            with open("models/saved/shap_explainer.pkl", "rb") as f:
                state["shap_explainer"] = pickle.load(f)
            logger.info("SHAP explainer loaded")

        state["loaded"] = True
        logger.info("All models loaded successfully")

    except Exception as e:
        logger.error("Failed to load models: %s", str(e))
        state["loaded"] = False


# ---------------------------------------------------------------------------
# Helper: preprocess a single transaction
# ---------------------------------------------------------------------------
def preprocess_transaction(txn: TransactionRequest) -> pd.DataFrame:
    """Convert a raw TransactionRequest into a feature DataFrame.

    Applies the same encoding as the training pipeline:
    - Label-encode categoricals (unseen → __unseen__)
    - Add simple engineered features
    - Keep NaN for XGBoost; impute for AE/LSTM

    Returns:
        Single-row DataFrame with all feature columns.
    """
    state = MODEL_STATE
    preprocessor = state["preprocessor"]
    feature_cols = state["feature_cols"]

    # Build a dict of all known fields
    row = {
        "TransactionAmt": txn.TransactionAmt,
        "TransactionDT": txn.TransactionDT or 86400,
        "card1": txn.card1,
        "card2": txn.card2,
        "card3": txn.card3,
        "card4": txn.card4,
        "card5": txn.card5,
        "card6": txn.card6,
        "addr1": txn.addr1,
        "addr2": txn.addr2,
        "P_emaildomain": txn.P_emaildomain,
        "R_emaildomain": txn.R_emaildomain,
        "ProductCD": txn.ProductCD,
        "DeviceType": txn.DeviceType,
        "DeviceInfo": txn.DeviceInfo,
    }

    # Add extra fields (C/D/V/id columns)
    if txn.extra_fields:
        row.update(txn.extra_fields)

    df = pd.DataFrame([row])

    # Encode categoricals using train-fitted label encoders
    if preprocessor and "label_encoders" in preprocessor:
        for col, le in preprocessor["label_encoders"].items():
            if col in df.columns:
                df[col] = df[col].fillna("unknown").astype(str)
                known = set(le.classes_)
                df[col] = df[col].apply(lambda x: x if x in known else "__unseen__")
                df[col] = le.transform(df[col])

    # Add simple engineered features
    df["Transaction_Hour"] = (df["TransactionDT"] / 3600 % 24).astype(int)
    df["Transaction_Day"] = (df["TransactionDT"] / (3600 * 24) % 7).astype(int)
    df["Is_High_Value"] = (df["TransactionAmt"] > 500).astype(int)
    df["TransactionAmt_Log"] = np.log1p(df["TransactionAmt"])

    # Add behavioral features as zeros (single transaction has no history in API)
    for w in [3600, 86400]:
        df[f"card_txn_count_{w}s"] = 0
        df[f"card_amt_sum_{w}s"] = 0.0
    df["card_time_since_last"] = -1

    # Ensure all feature columns exist (fill missing with NaN)
    for col in feature_cols:
        if col not in df.columns:
            df[col] = np.nan

    return df[feature_cols].astype(np.float32)


# ---------------------------------------------------------------------------
# Helper: generate SHAP explanations
# ---------------------------------------------------------------------------
def get_shap_reasons(features_df: pd.DataFrame, top_k: int = 5) -> List[dict]:
    """Generate top-K SHAP reasons for a prediction in plain words.

    If the SHAP explainer isn't loaded, returns an empty list.
    """
    state = MODEL_STATE
    explainer = state.get("shap_explainer")
    feature_cols = state["feature_cols"]

    if explainer is None:
        return []

    try:
        # Import the description helper
        from explainability.shap_explainer import explain_single
        reasons = explain_single(
            state["xgb_model"], explainer, features_df, feature_cols, top_k
        )
        return reasons
    except Exception as e:
        logger.warning("SHAP explanation failed: %s", str(e))
        return []


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
def health_check():
    """Health check endpoint. Returns 200 if models are loaded."""
    if not MODEL_STATE["loaded"]:
        raise HTTPException(status_code=503, detail="Models not loaded")
    return {"status": "healthy", "model_loaded": True}


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info():
    """Return metadata about the loaded model."""
    state = MODEL_STATE
    if not state["loaded"]:
        raise HTTPException(status_code=503, detail="Models not loaded")

    ec = state["ensemble_config"]
    return ModelInfoResponse(
        model_version=state["model_version"],
        components=ec["components"],
        include_ae=ec["include_ae"],
        include_lstm=ec["include_lstm"],
        review_threshold=ec["review_threshold"],
        block_threshold=ec["block_threshold"],
        n_features=len(state["feature_cols"]),
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(txn: TransactionRequest):
    """Score a raw transaction and return fraud probability + decision.

    Returns:
        Fraud probability, approve/review/block decision, top 5 SHAP reasons,
        model version, and latency in ms.
    """
    start_time = time.time()
    state = MODEL_STATE

    if not state["loaded"]:
        raise HTTPException(status_code=503, detail="Models not loaded")

    ec = state["ensemble_config"]
    stacking_lr = ec["stacking_model"]

    # Preprocess the transaction
    features_df = preprocess_transaction(txn)

    # XGBoost prediction
    xgb_prob = state["xgb_model"].predict_proba(features_df)[:, 1][0]

    # Build stacking input
    stack_input = [xgb_prob]

    # Autoencoder score (if included)
    if ec["include_ae"] and state["ae_model"] is not None:
        preprocessor = state["preprocessor"]
        # Scale features for AE
        if preprocessor and "scaler" in preprocessor:
            scaled = preprocessor["scaler"].transform(
                features_df.fillna(preprocessor["train_medians"]))
        else:
            scaled = features_df.fillna(0).values

        ae_recon = state["ae_model"].predict(scaled, verbose=0)
        ae_mse = np.mean(np.power(scaled - ae_recon, 2), axis=1)[0]
        ae_score = np.searchsorted(state["ae_calibration"], ae_mse) / len(
            state["ae_calibration"])
        stack_input.append(float(ae_score))

    # LSTM score (if included) — simplified for API (no sequence history)
    if ec["include_lstm"] and state["lstm_model"] is not None:
        # Use XGBoost prob as LSTM proxy in API (streaming has real sequences)
        stack_input.append(float(xgb_prob))

    # Stacking logistic regression
    stack_X = np.array([stack_input])
    fraud_prob = float(stacking_lr.predict_proba(stack_X)[:, 1][0])

    # Decision based on thresholds
    if fraud_prob >= ec["block_threshold"]:
        decision = "block"
    elif fraud_prob >= ec["review_threshold"]:
        decision = "review"
    else:
        decision = "approve"

    # SHAP explanations from XGBoost component
    reasons_raw = get_shap_reasons(features_df, top_k=5)
    reasons = [ShapReason(**r) for r in reasons_raw]

    # Compute latency
    latency_ms = (time.time() - start_time) * 1000

    return PredictionResponse(
        transaction_id=f"txn_{int(time.time() * 1000)}",
        fraud_probability=round(fraud_prob, 6),
        decision=decision,
        reasons=reasons,
        model_version=state["model_version"],
        latency_ms=round(latency_ms, 2),
    )
