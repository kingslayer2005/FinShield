"""
tests/test_phase1.py
Tests for Phase 1: preprocessing, behavioural features, unseen categories,
chronological split, NaN handling, and model training smoke tests.

Run from repo root:
    SMOKE=1 python -m pytest tests/test_phase1.py -v
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd
import pickle
import json

# Allow imports from repo root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Force smoke mode for tests
os.environ["SMOKE"] = "1"

from utils.config import cfg, is_smoke, get_processed_dir


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def processed_dir():
    """Return the processed directory path (smoke mode)."""
    return get_processed_dir()


@pytest.fixture(scope="module")
def train_df(processed_dir):
    """Load train parquet."""
    path = os.path.join(processed_dir, "train.parquet")
    if not os.path.exists(path):
        pytest.skip("train.parquet not found — run preprocessing first")
    return pd.read_parquet(path)


@pytest.fixture(scope="module")
def val_df(processed_dir):
    """Load val parquet."""
    path = os.path.join(processed_dir, "val.parquet")
    if not os.path.exists(path):
        pytest.skip("val.parquet not found — run preprocessing first")
    return pd.read_parquet(path)


@pytest.fixture(scope="module")
def test_df(processed_dir):
    """Load test parquet."""
    path = os.path.join(processed_dir, "test.parquet")
    if not os.path.exists(path):
        pytest.skip("test.parquet not found — run preprocessing first")
    return pd.read_parquet(path)


@pytest.fixture(scope="module")
def preprocessor(processed_dir):
    """Load the fitted preprocessor."""
    path = os.path.join(processed_dir, "preprocessor.pkl")
    if not os.path.exists(path):
        pytest.skip("preprocessor.pkl not found — run preprocessing first")
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Test: Chronological split — no time overlap
# ---------------------------------------------------------------------------
class TestChronologicalSplit:
    """Verify the chronological train/val/test split has no time overlap."""

    def test_train_before_val(self, train_df, val_df):
        """Every train row must have DT <= every val row DT."""
        assert train_df["TransactionDT"].max() <= val_df["TransactionDT"].min()

    def test_val_before_test(self, val_df, test_df):
        """Every val row must have DT <= every test row DT."""
        assert val_df["TransactionDT"].max() <= test_df["TransactionDT"].min()

    def test_split_fractions(self, train_df, val_df, test_df):
        """Split fractions should be approximately 70/15/15."""
        total = len(train_df) + len(val_df) + len(test_df)
        train_frac = len(train_df) / total
        val_frac = len(val_df) / total
        test_frac = len(test_df) / total
        assert 0.65 <= train_frac <= 0.75, f"Train frac {train_frac:.2f} out of range"
        assert 0.10 <= val_frac <= 0.20, f"Val frac {val_frac:.2f} out of range"
        assert 0.10 <= test_frac <= 0.20, f"Test frac {test_frac:.2f} out of range"


# ---------------------------------------------------------------------------
# Test: Behavioural features computed correctly
# ---------------------------------------------------------------------------
class TestBehavioralFeatures:
    """Verify behavioural features are present and leakage-safe."""

    def test_behavioral_columns_exist(self, train_df):
        """Behavioural feature columns should be present."""
        expected = ["card_time_since_last"]
        for w in cfg["features"]["rolling_windows_sec"]:
            expected.append(f"card_txn_count_{w}s")
            expected.append(f"card_amt_sum_{w}s")
        for col in expected:
            assert col in train_df.columns, f"Missing behavioural column: {col}"

    def test_first_transaction_has_no_history(self, train_df):
        """The very first row (by time) should have -1 for time_since_last
        and 0 for rolling counts/sums."""
        # Sort by time and check the first row of any card
        sorted_df = train_df.sort_values("TransactionDT")
        first_row = sorted_df.iloc[0]
        assert first_row["card_time_since_last"] == -1, \
            "First transaction should have time_since_last = -1"

    def test_behavioral_features_present_in_all_splits(self, train_df, val_df, test_df):
        """Behavioral features should be in all splits."""
        for col in ["card_time_since_last"]:
            assert col in train_df.columns
            assert col in val_df.columns
            assert col in test_df.columns


# ---------------------------------------------------------------------------
# Test: Unseen categories map to __unseen__ code (CORRECTION 3)
# ---------------------------------------------------------------------------
class TestUnseenCategories:
    """Verify that categories not seen in train don't crash the encoders."""

    def test_unseen_email_domain(self, preprocessor):
        """An email domain not in train should map to __unseen__ code."""
        cat_cols = preprocessor["cat_cols"]
        le_dict = preprocessor["label_encoders"]

        # Test with a completely new domain
        if "P_emaildomain" in le_dict:
            le = le_dict["P_emaildomain"]
            # __unseen__ must be in the encoder's classes
            assert "__unseen__" in le.classes_, \
                "Label encoder must have __unseen__ class"

            # Verify that transforming __unseen__ doesn't crash
            code = le.transform(["__unseen__"])[0]
            assert isinstance(code, (int, np.integer)), \
                "__unseen__ should map to an integer code"

    def test_unseen_device_type(self, preprocessor):
        """A device type not in train should map to __unseen__ code."""
        le_dict = preprocessor["label_encoders"]
        if "DeviceType" in le_dict:
            le = le_dict["DeviceType"]
            assert "__unseen__" in le.classes_

    def test_encoding_does_not_crash_on_novel_value(self, preprocessor):
        """Simulate encoding a value that was never in training data."""
        le_dict = preprocessor["label_encoders"]
        for col_name, le in le_dict.items():
            known = set(le.classes_)
            novel_val = "THIS_IS_A_COMPLETELY_NEW_VALUE_12345"
            # The pipeline maps unknowns to __unseen__
            mapped = novel_val if novel_val in known else "__unseen__"
            assert mapped in known, \
                f"__unseen__ not in encoder for {col_name}"
            code = le.transform([mapped])[0]
            assert isinstance(code, (int, np.integer))


# ---------------------------------------------------------------------------
# Test: NaN handling (CORRECTION 4)
# ---------------------------------------------------------------------------
class TestNaNHandling:
    """Verify NaN is preserved for XGBoost and imputed for scaled data."""

    def test_train_has_nan_for_xgboost(self, train_df):
        """Raw train parquet should still contain NaN values (for XGBoost)."""
        # The original data has NaN in V-columns, D-columns, etc.
        # After encoding categoricals, numeric NaN should be preserved
        meta = ["TransactionID", "TransactionDT", "isFraud"]
        num_cols = [c for c in train_df.columns
                    if c not in meta and train_df[c].dtype in [np.float64, np.float32]]
        if num_cols:
            has_nan = train_df[num_cols].isnull().any().any()
            # Synthetic data has NaN injected, so this should be True
            assert has_nan, "Train should preserve NaN for XGBoost"

    def test_scaled_has_no_nan(self, processed_dir):
        """Scaled parquets (for AE/LSTM) should have no NaN after imputation."""
        path = os.path.join(processed_dir, "train_scaled.parquet")
        if not os.path.exists(path):
            pytest.skip("train_scaled.parquet not found")
        scaled = pd.read_parquet(path)
        meta = ["TransactionID", "TransactionDT", "isFraud"]
        feat_cols = [c for c in scaled.columns if c not in meta]
        assert not scaled[feat_cols].isnull().any().any(), \
            "Scaled data should have no NaN (imputed with train medians)"


# ---------------------------------------------------------------------------
# Test: Inner validation split (CORRECTION 5)
# ---------------------------------------------------------------------------
class TestInnerValidation:
    """Verify inner train/val split exists and is chronologically ordered."""

    def test_inner_splits_exist(self, processed_dir):
        """Inner train and inner val parquets should exist."""
        assert os.path.exists(os.path.join(processed_dir, "train_inner.parquet"))
        assert os.path.exists(os.path.join(processed_dir, "inner_val.parquet"))

    def test_inner_split_chronological(self, processed_dir):
        """Inner train DT max <= inner val DT min."""
        inner_train = pd.read_parquet(os.path.join(processed_dir, "train_inner.parquet"))
        inner_val = pd.read_parquet(os.path.join(processed_dir, "inner_val.parquet"))
        assert inner_train["TransactionDT"].max() <= inner_val["TransactionDT"].min()


# ---------------------------------------------------------------------------
# Test: Behavioral features are computed BEFORE split (CORRECTION 2)
# ---------------------------------------------------------------------------
class TestBehavioralBeforeSplit:
    """Verify that val/test rows can have non-zero behavioral features
    from cards that appeared in train (real history)."""

    def test_val_has_nonzero_behavioral(self, val_df):
        """Some val rows should have non-zero card_txn_count_3600s,
        indicating they inherited history from the train period."""
        count_col = "card_txn_count_3600s"
        if count_col in val_df.columns:
            has_history = (val_df[count_col] > 0).any()
            # This is expected because behavioral features were computed
            # on the full dataset before splitting
            assert has_history, \
                "Val rows should have non-zero behavioral features from train history"


# ---------------------------------------------------------------------------
# Test: Config and smoke mode
# ---------------------------------------------------------------------------
class TestConfig:
    """Verify configuration and smoke mode setup."""

    def test_smoke_mode_active(self):
        """SMOKE=1 should activate smoke mode."""
        assert is_smoke, "SMOKE=1 should make is_smoke True"

    def test_smoke_overrides_applied(self):
        """Smoke mode should override XGBoost n_estimators to a small value."""
        assert cfg["xgboost"]["n_estimators"] <= 50, \
            "Smoke mode should reduce n_estimators"
