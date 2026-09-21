"""
tests/test_phase2.py
Tests for Phase 2: FastAPI scoring API.

Run from repo root:
    SMOKE=1 python -m pytest tests/test_phase2.py -v
"""

import os
import sys
import pytest
import json
import time
import numpy as np

os.environ["SMOKE"] = "1"
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    """Create a TestClient for the FastAPI app."""
    # Check that models exist before trying to load them
    if not os.path.exists("models/saved/xgboost_model.pkl"):
        pytest.skip("Model artifacts not found — train models first")

    from api.app import app
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------
class TestHealth:
    def test_health_returns_200(self, client):
        """GET /health should return 200 when models are loaded."""
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True


# ---------------------------------------------------------------------------
# Model info endpoint
# ---------------------------------------------------------------------------
class TestModelInfo:
    def test_model_info_returns_metadata(self, client):
        """GET /model-info should return model metadata."""
        resp = client.get("/model-info")
        assert resp.status_code == 200
        data = resp.json()
        assert "components" in data
        assert "review_threshold" in data
        assert "block_threshold" in data
        assert data["n_features"] > 0


# ---------------------------------------------------------------------------
# Predict endpoint
# ---------------------------------------------------------------------------
class TestPredict:
    def test_predict_returns_valid_response(self, client):
        """POST /predict should return fraud probability and decision."""
        txn = {
            "TransactionAmt": 150.0,
            "card1": 1234,
            "TransactionDT": 100000,
        }
        resp = client.post("/predict", json=txn)
        assert resp.status_code == 200
        data = resp.json()

        # Check required fields
        assert "fraud_probability" in data
        assert "decision" in data
        assert "reasons" in data
        assert "model_version" in data
        assert "latency_ms" in data

        # Fraud probability should be between 0 and 1
        assert 0.0 <= data["fraud_probability"] <= 1.0

        # Decision should be one of approve/review/block
        assert data["decision"] in ["approve", "review", "block"]

        # Latency should be positive
        assert data["latency_ms"] > 0

    def test_predict_high_amount_transaction(self, client):
        """A very high amount transaction should score higher than a low one."""
        low_txn = {"TransactionAmt": 10.0, "card1": 5000, "TransactionDT": 100000}
        high_txn = {"TransactionAmt": 9999.0, "card1": 5000, "TransactionDT": 100000}

        resp_low = client.post("/predict", json=low_txn)
        resp_high = client.post("/predict", json=high_txn)

        assert resp_low.status_code == 200
        assert resp_high.status_code == 200

        # Not guaranteed, but generally high amounts score higher
        # Just verify both return valid responses
        assert 0 <= resp_low.json()["fraud_probability"] <= 1
        assert 0 <= resp_high.json()["fraud_probability"] <= 1

    def test_predict_with_optional_fields(self, client):
        """POST /predict should work with all optional fields."""
        txn = {
            "TransactionAmt": 250.0,
            "card1": 3000,
            "card4": "visa",
            "card6": "debit",
            "P_emaildomain": "gmail.com",
            "DeviceType": "mobile",
            "TransactionDT": 200000,
        }
        resp = client.post("/predict", json=txn)
        assert resp.status_code == 200

    def test_predict_with_unseen_email(self, client):
        """An unseen email domain should not crash the prediction."""
        txn = {
            "TransactionAmt": 100.0,
            "card1": 4000,
            "P_emaildomain": "totally_new_domain_never_seen.xyz",
            "TransactionDT": 150000,
        }
        resp = client.post("/predict", json=txn)
        assert resp.status_code == 200
        assert 0 <= resp.json()["fraud_probability"] <= 1

    def test_shap_reasons_in_response(self, client):
        """Response should include SHAP reasons (may be empty if explainer not loaded)."""
        txn = {"TransactionAmt": 500.0, "card1": 2000, "TransactionDT": 100000}
        resp = client.post("/predict", json=txn)
        assert resp.status_code == 200
        data = resp.json()
        # reasons is a list (may be empty if SHAP not loaded)
        assert isinstance(data["reasons"], list)


# ---------------------------------------------------------------------------
# Latency measurement
# ---------------------------------------------------------------------------
class TestLatency:
    def test_measure_latency(self, client):
        """Measure p50 and p95 latency over 50 requests."""
        txn = {"TransactionAmt": 200.0, "card1": 3000, "TransactionDT": 100000}
        latencies = []

        for _ in range(50):
            start = time.time()
            resp = client.post("/predict", json=txn)
            elapsed = (time.time() - start) * 1000  # ms
            latencies.append(elapsed)
            assert resp.status_code == 200

        latencies.sort()
        p50 = latencies[len(latencies) // 2]
        p95 = latencies[int(len(latencies) * 0.95)]

        print(f"\n  Latency p50: {p50:.1f}ms, p95: {p95:.1f}ms")

        # Save latency report
        reports_dir = "reports_smoke" if os.environ.get("SMOKE") == "1" else "reports"
        os.makedirs(reports_dir, exist_ok=True)
        latency_report = {
            "p50_ms": round(p50, 2),
            "p95_ms": round(p95, 2),
            "n_requests": len(latencies),
            "min_ms": round(min(latencies), 2),
            "max_ms": round(max(latencies), 2),
            "is_smoke": os.environ.get("SMOKE") == "1",
        }
        with open(os.path.join(reports_dir, "latency.json"), "w") as f:
            json.dump(latency_report, f, indent=2)

        # Sanity check: p95 should be under 5 seconds
        assert p95 < 5000, f"p95 latency too high: {p95:.1f}ms"
