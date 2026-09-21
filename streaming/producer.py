"""
streaming/producer.py
Kafka producer that replays test-set transactions in time order at a
configurable rate.  Each message is a JSON-serialised transaction row.

Run from repo root:
    python streaming/producer.py                  # real data
    SMOKE=1 python streaming/producer.py          # synthetic data
    SMOKE=1 REPLAY_RATE=50 python streaming/producer.py  # custom rate

Environment variables:
    REPLAY_RATE  — transactions per second (default from config.yaml)
    MAX_EVENTS   — stop after this many events (default: all)
"""

import os
import sys
import json
import time
import logging

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.config import cfg, is_smoke, get_processed_dir

import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("finshield.producer")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PROCESSED = get_processed_dir()
TOPIC = cfg["streaming"]["topic_transactions"]
BOOTSTRAP = cfg["streaming"]["kafka_bootstrap"]
RATE = int(os.environ.get("REPLAY_RATE", cfg["streaming"]["replay_rate"]))
MAX_EVENTS = int(os.environ.get("MAX_EVENTS", 0))  # 0 = all


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if pd.isna(obj):
            return None
        return super().default(obj)


def run_producer():
    """Replay test-set transactions to Kafka at a configurable rate."""
    from confluent_kafka import Producer

    logger.info("Loading test data from %s...", PROCESSED)
    test_df = pd.read_parquet(os.path.join(PROCESSED, "test.parquet"))
    test_df = test_df.sort_values("TransactionDT").reset_index(drop=True)

    if MAX_EVENTS > 0:
        test_df = test_df.head(MAX_EVENTS)

    logger.info("Replaying %d transactions to topic '%s' at %d/sec",
                len(test_df), TOPIC, RATE)

    # Kafka producer config
    producer = Producer({
        "bootstrap.servers": BOOTSTRAP,
        "linger.ms": 10,  # batch small messages
    })

    # Delivery callback for error tracking
    delivery_errors = [0]

    def delivery_callback(err, msg):
        if err:
            delivery_errors[0] += 1
            logger.error("Delivery failed: %s", err)

    sent = 0
    interval = 1.0 / RATE  # seconds between messages

    for idx, row in test_df.iterrows():
        # Serialise the row as JSON
        msg = json.dumps(row.to_dict(), cls=NumpyEncoder)

        # Send to Kafka
        producer.produce(
            TOPIC,
            key=str(int(row.get("TransactionID", idx))),
            value=msg,
            callback=delivery_callback,
        )

        sent += 1
        if sent % 1000 == 0:
            producer.flush()
            logger.info("  Sent %d / %d transactions", sent, len(test_df))

        # Rate limiting
        time.sleep(interval)

    # Final flush
    producer.flush()
    logger.info("Done! Sent %d transactions, %d delivery errors",
                sent, delivery_errors[0])


if __name__ == "__main__":
    run_producer()
