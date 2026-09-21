"""
monitoring/drift.py
Evidently drift report: compares recent streamed transactions against a
reference sample from training data. Saves HTML report to reports/drift/
and a summary row to Postgres table 'drift_reports'.

Run from repo root:
    python monitoring/drift.py
    SMOKE=1 python monitoring/drift.py
"""

import os
import sys
import json
import logging
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.config import cfg, is_smoke, get_processed_dir, get_reports_dir

import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("finshield.drift")


def run_drift_check():
    """Run Evidently drift detection and save results."""
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset

    PROCESSED = get_processed_dir()
    REPORTS = get_reports_dir()
    DRIFT_DIR = os.path.join(REPORTS, "drift")
    os.makedirs(DRIFT_DIR, exist_ok=True)

    REF_SAMPLE = cfg["monitoring"]["drift_reference_sample"]
    META_COLS = ["TransactionID", "TransactionDT", "isFraud"]

    # Load reference data (sample from training set)
    logger.info("Loading reference data...")
    train_df = pd.read_parquet(os.path.join(PROCESSED, "train.parquet"))
    feature_cols = [c for c in train_df.columns if c not in META_COLS]

    ref_sample = train_df[feature_cols].sample(
        n=min(REF_SAMPLE, len(train_df)), random_state=42
    )
    logger.info("Reference sample: %d rows", len(ref_sample))

    # Load current data (test set as proxy for "recent streamed")
    logger.info("Loading current data...")
    test_df = pd.read_parquet(os.path.join(PROCESSED, "test.parquet"))
    current_sample = test_df[feature_cols].sample(
        n=min(REF_SAMPLE, len(test_df)), random_state=42
    )
    logger.info("Current sample: %d rows", len(current_sample))

    # Run Evidently drift report
    logger.info("Computing drift report...")
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref_sample, current_data=current_sample)

    # Save HTML report
    html_path = os.path.join(DRIFT_DIR, "drift_report.html")
    report.save_html(html_path)
    logger.info("Saved HTML report: %s", html_path)

    # Extract summary
    result = report.as_dict()
    metrics = result.get("metrics", [{}])
    drift_result = metrics[0].get("result", {}) if metrics else {}

    drift_detected = drift_result.get("dataset_drift", False)
    drift_share = drift_result.get("share_of_drifted_columns", 0.0)
    n_drifted = drift_result.get("number_of_drifted_columns", 0)
    n_columns = drift_result.get("number_of_columns", 0)

    summary = {
        "drift_detected": bool(drift_detected),
        "drift_share": float(drift_share),
        "n_drifted_columns": int(n_drifted),
        "n_columns": int(n_columns),
        "is_smoke": is_smoke,
    }

    # Save summary JSON
    summary_path = os.path.join(DRIFT_DIR, "drift_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Saved summary: %s", summary_path)

    # Try to write to Postgres
    try:
        from sqlalchemy import create_engine, text
        engine = create_engine(cfg["streaming"]["postgres_dsn"])
        with engine.connect() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS drift_reports (
                    id SERIAL PRIMARY KEY,
                    drift_detected BOOLEAN,
                    drift_share FLOAT,
                    n_drifted_columns INT,
                    n_columns INT,
                    created_at TIMESTAMP DEFAULT NOW()
                )
            """))
            conn.execute(text("""
                INSERT INTO drift_reports
                    (drift_detected, drift_share, n_drifted_columns, n_columns)
                VALUES (:dd, :ds, :ndc, :nc)
            """), {
                "dd": drift_detected,
                "ds": drift_share,
                "ndc": n_drifted,
                "nc": n_columns,
            })
            conn.commit()
        logger.info("Wrote drift summary to Postgres")
    except Exception as e:
        logger.warning("Could not write to Postgres: %s", e)

    logger.info("Drift detected: %s (%.1f%% columns drifted)",
                drift_detected, drift_share * 100)
    return summary


if __name__ == "__main__":
    run_drift_check()
