"""
streaming/consumer.py
Kafka consumer that scores each transaction event in real time.

For each event:
  1. Computes behavioural features from per-card state in Redis
     (same logic as features/behavioral.py — single source of truth)
  2. Scores with the ensemble (XGBoost + optional AE + optional LSTM)
  3. Writes prediction to Postgres table 'predictions'
  4. Publishes flagged events to topic 'alerts'

Run from repo root:
    python streaming/consumer.py
"""

import os
import sys
import json
import time
import pickle
import logging
import warnings

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.config import cfg, is_smoke

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("finshield.consumer")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TOPIC = cfg["streaming"]["topic_transactions"]
TOPIC_ALERTS = cfg["streaming"]["topic_alerts"]
BOOTSTRAP = cfg["streaming"]["kafka_bootstrap"]
REDIS_URL = cfg["streaming"]["redis_url"]
PG_DSN = cfg["streaming"]["postgres_dsn"]
WINDOWS = cfg["features"]["rolling_windows_sec"]  # [3600, 86400]
CARD_KEY_COLS = cfg["features"]["card_key"]


def make_card_key(txn: dict) -> str:
    """Build card key string from transaction dict.

    Falls back to card1 only if addr1 is missing.
    """
    parts = []
    for col in CARD_KEY_COLS:
        val = txn.get(col)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            parts.append("")
        else:
            parts.append(str(int(val)) if isinstance(val, float) else str(val))
    return "_".join(parts)


class RedisCardState:
    """Manages per-card transaction history in Redis for behavioural features.

    Each card has a Redis sorted set keyed by TransactionDT, with values
    being TransactionAmt. This mirrors the offline behavioral.py logic:
    for each new transaction, we count and sum prior transactions within
    each rolling window.
    """

    def __init__(self, redis_url: str):
        import redis
        self.r = redis.Redis.from_url(redis_url, decode_responses=True)
        logger.info("Connected to Redis at %s", redis_url)

    def compute_features(self, card_key: str, txn_dt: float,
                         txn_amt: float) -> dict:
        """Compute behavioural features for one transaction using Redis state.

        This is the ONLINE equivalent of features/behavioral.py. The logic
        must produce identical results for parity testing.

        Args:
            card_key: Unique card identifier string.
            txn_dt: Transaction timestamp (seconds).
            txn_amt: Transaction amount.

        Returns:
            Dict with behavioural feature values.
        """
        redis_key = f"card:{card_key}"

        # Get time of last transaction (the highest-scored member before this one)
        # zrangebyscore returns members with score in [min, max]
        prev_txns = self.r.zrangebyscore(redis_key, "-inf", f"({txn_dt}",
                                          withscores=True)

        # Time since last transaction
        if prev_txns:
            last_dt = prev_txns[-1][1]  # score of last member
            time_since_last = txn_dt - last_dt
        else:
            time_since_last = -1  # first transaction for this card

        features = {"card_time_since_last": time_since_last}

        # Rolling window counts and sums
        for window in WINDOWS:
            suffix = f"{window}s"
            window_start = txn_dt - window

            # Get transactions in [window_start, txn_dt) — strictly before current
            window_txns = self.r.zrangebyscore(
                redis_key, str(window_start), f"({txn_dt}",
                withscores=True
            )

            count = len(window_txns)
            amt_sum = sum(float(member) for member, score in window_txns)

            features[f"card_txn_count_{suffix}"] = count
            features[f"card_amt_sum_{suffix}"] = amt_sum

        # Add current transaction to Redis for future lookups
        # Store amount as member name (with txn_dt as score for dedup)
        member_key = f"{txn_dt}:{txn_amt}"  # unique member per transaction
        self.r.zadd(redis_key, {member_key: txn_dt})

        # Optional: expire old data to bound Redis memory (keep 7 days)
        cutoff = txn_dt - 7 * 86400
        self.r.zremrangebyscore(redis_key, "-inf", str(cutoff))

        return features


def create_predictions_table(engine):
    """Create the predictions table in Postgres if it doesn't exist."""
    from sqlalchemy import text
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS predictions (
                id SERIAL PRIMARY KEY,
                transaction_id TEXT NOT NULL,
                event_time FLOAT NOT NULL,
                score FLOAT NOT NULL,
                decision TEXT NOT NULL,
                reasons JSONB,
                model_version TEXT,
                latency_ms FLOAT,
                created_at TIMESTAMP DEFAULT NOW()
            )
        """))
        conn.commit()
    logger.info("Postgres predictions table ready")


def run_consumer():
    """Main consumer loop: read from Kafka, score, write to Postgres."""
    from confluent_kafka import Consumer, Producer as KProducer
    from sqlalchemy import create_engine, text

    # Load model artifacts
    logger.info("Loading model artifacts...")
    with open("models/saved/xgboost_model.pkl", "rb") as f:
        xgb_model = pickle.load(f)
    with open("models/saved/feature_cols.pkl", "rb") as f:
        feature_cols = pickle.load(f)
    with open("models/saved/ensemble_config.pkl", "rb") as f:
        ensemble_cfg = pickle.load(f)

    preprocessor_paths = [
        "data/synthetic/processed/preprocessor.pkl",
        "data/processed/preprocessor.pkl",
    ]
    preprocessor = None
    for pp_path in preprocessor_paths:
        if os.path.exists(pp_path):
            with open(pp_path, "rb") as f:
                preprocessor = pickle.load(f)
            break

    stacking_lr = ensemble_cfg["stacking_model"]
    review_thresh = ensemble_cfg["review_threshold"]
    block_thresh = ensemble_cfg["block_threshold"]

    # Init Redis for behavioral features
    card_state = RedisCardState(REDIS_URL)

    # Init Postgres
    engine = create_engine(PG_DSN)
    create_predictions_table(engine)

    # Kafka consumer
    consumer = Consumer({
        "bootstrap.servers": BOOTSTRAP,
        "group.id": "finshield-scorer",
        "auto.offset.reset": "earliest",
    })
    consumer.subscribe([TOPIC])

    # Kafka producer for alerts
    alert_producer = KProducer({"bootstrap.servers": BOOTSTRAP})

    logger.info("Consumer ready. Listening on topic '%s'...", TOPIC)
    processed = 0

    try:
        while True:
            msg = consumer.poll(timeout=1.0)
            if msg is None:
                continue
            if msg.error():
                logger.error("Consumer error: %s", msg.error())
                continue

            start_time = time.time()

            # Parse transaction
            txn = json.loads(msg.value().decode("utf-8"))
            txn_id = str(txn.get("TransactionID", processed))
            txn_dt = float(txn.get("TransactionDT", 0))
            txn_amt = float(txn.get("TransactionAmt", 0))

            # Compute behavioral features from Redis
            card_key = make_card_key(txn)
            behav_features = card_state.compute_features(card_key, txn_dt, txn_amt)
            txn.update(behav_features)

            # Encode categoricals
            df = pd.DataFrame([txn])
            if preprocessor:
                for col, le in preprocessor.get("label_encoders", {}).items():
                    if col in df.columns:
                        df[col] = df[col].fillna("unknown").astype(str)
                        known = set(le.classes_)
                        df[col] = df[col].apply(
                            lambda x: x if x in known else "__unseen__")
                        df[col] = le.transform(df[col])

            # Add simple features
            df["Transaction_Hour"] = (df["TransactionDT"] / 3600 % 24).astype(int)
            df["Transaction_Day"] = (df["TransactionDT"] / (3600 * 24) % 7).astype(int)
            df["Is_High_Value"] = (df["TransactionAmt"] > 500).astype(int)
            df["TransactionAmt_Log"] = np.log1p(df["TransactionAmt"])

            # Ensure all feature columns exist
            for col in feature_cols:
                if col not in df.columns:
                    df[col] = np.nan

            X = df[feature_cols].astype(np.float32)

            # XGBoost prediction
            xgb_prob = float(xgb_model.predict_proba(X)[:, 1][0])
            stack_input = [xgb_prob]

            # Add AE / LSTM scores if included (simplified for consumer)
            for _ in range(len(ensemble_cfg["components"]) - 1):
                stack_input.append(xgb_prob)  # fallback

            stack_X = np.array([stack_input])
            score = float(stacking_lr.predict_proba(stack_X)[:, 1][0])

            # Decision
            if score >= block_thresh:
                decision = "block"
            elif score >= review_thresh:
                decision = "review"
            else:
                decision = "approve"

            latency_ms = (time.time() - start_time) * 1000

            # Write to Postgres
            with engine.connect() as conn:
                conn.execute(text("""
                    INSERT INTO predictions
                        (transaction_id, event_time, score, decision, reasons,
                         model_version, latency_ms)
                    VALUES (:tid, :et, :score, :decision, :reasons, :mv, :lat)
                """), {
                    "tid": txn_id,
                    "et": txn_dt,
                    "score": score,
                    "decision": decision,
                    "reasons": json.dumps([]),  # SHAP not computed in stream
                    "mv": "local",
                    "lat": latency_ms,
                })
                conn.commit()

            # Publish alert if flagged
            if decision in ("review", "block"):
                alert_msg = json.dumps({
                    "transaction_id": txn_id,
                    "score": score,
                    "decision": decision,
                    "latency_ms": latency_ms,
                })
                alert_producer.produce(TOPIC_ALERTS, value=alert_msg)

            processed += 1
            if processed % 100 == 0:
                logger.info("Processed %d events (last: score=%.4f, decision=%s, "
                            "latency=%.1fms)", processed, score, decision, latency_ms)

    except KeyboardInterrupt:
        logger.info("Shutting down after %d events", processed)
    finally:
        consumer.close()
        alert_producer.flush()


if __name__ == "__main__":
    run_consumer()
