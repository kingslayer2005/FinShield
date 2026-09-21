"""
explainability/shap_explainer.py
Generates SHAP explanations for the XGBoost component.
Provides both global feature importance and per-transaction explanations
in plain words (used by the API for "top 5 reasons").

Run from repo root:
    python explainability/shap_explainer.py
    SMOKE=1 python explainability/shap_explainer.py
"""

import pandas as pd
import numpy as np
import pickle
import os
import sys
import json
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.config import cfg, is_smoke, get_processed_dir, get_reports_dir

import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Feature name → plain-English mapping for API responses
# ---------------------------------------------------------------------------
FEATURE_DESCRIPTIONS = {
    "TransactionAmt": "Transaction amount",
    "TransactionAmt_Log": "Log of transaction amount",
    "Transaction_Hour": "Hour of the day",
    "Transaction_Day": "Day of the week",
    "Is_High_Value": "High-value transaction flag",
    "card1": "Card identifier",
    "card2": "Card attribute 2",
    "card3": "Card attribute 3",
    "card5": "Card attribute 5",
    "addr1": "Billing address",
    "addr2": "Billing country",
    "P_emaildomain": "Purchaser email domain",
    "R_emaildomain": "Recipient email domain",
    "DeviceType": "Device type",
    "DeviceInfo": "Device info",
    "ProductCD": "Product code",
    "card_time_since_last": "Time since last transaction",
}

# Add rolling window features
for w in [3600, 86400]:
    label = "1 hour" if w == 3600 else "24 hours"
    FEATURE_DESCRIPTIONS[f"card_txn_count_{w}s"] = f"Transaction count in last {label}"
    FEATURE_DESCRIPTIONS[f"card_amt_sum_{w}s"] = f"Total amount in last {label}"


def get_feature_description(feat_name: str) -> str:
    """Convert a feature name to a plain-English description.

    Falls back to the raw feature name with underscores replaced by spaces.
    """
    if feat_name in FEATURE_DESCRIPTIONS:
        return FEATURE_DESCRIPTIONS[feat_name]
    # Auto-generate: "C1" → "Feature C1", "V12" → "Feature V12"
    if feat_name[0] in ("C", "D", "V") and feat_name[1:].isdigit():
        return f"Vesta feature {feat_name}"
    if feat_name.startswith("id_"):
        return f"Identity feature {feat_name}"
    return feat_name.replace("_", " ").title()


def explain_single(xgb_model, explainer, features_row: pd.DataFrame,
                    feature_cols: list, top_k: int = 5) -> list:
    """Generate top-K SHAP reasons for a single transaction in plain words.

    Args:
        xgb_model: Trained XGBoost model.
        explainer: SHAP TreeExplainer.
        features_row: Single-row DataFrame of features.
        feature_cols: List of feature column names.
        top_k: Number of top reasons to return.

    Returns:
        List of dicts with 'feature', 'description', 'impact', 'direction'.
    """
    shap_values = explainer.shap_values(features_row)
    # For binary classification, shap_values may be a list [class0, class1]
    if isinstance(shap_values, list):
        sv = shap_values[1]  # class 1 (fraud) SHAP values
    else:
        sv = shap_values

    # Get the first (and only) row
    sv_row = sv[0] if sv.ndim > 1 else sv

    # Pair feature names with SHAP values
    pairs = list(zip(feature_cols, sv_row))
    # Sort by absolute SHAP value (most impactful first)
    pairs.sort(key=lambda x: abs(x[1]), reverse=True)

    reasons = []
    for feat, shap_val in pairs[:top_k]:
        direction = "increases" if shap_val > 0 else "decreases"
        reasons.append({
            "feature": feat,
            "description": get_feature_description(feat),
            "impact": round(float(abs(shap_val)), 4),
            "direction": f"{direction} fraud risk",
        })
    return reasons


def main():
    """Generate global SHAP importance plot and save explainer."""
    PROCESSED = get_processed_dir()
    REPORTS = get_reports_dir()
    FIGURES = os.path.join(REPORTS, "figures")
    os.makedirs(FIGURES, exist_ok=True)

    print("=" * 60)
    print("FinShield SHAP Explainability")
    if is_smoke:
        print("  MODE: SMOKE")
    print("=" * 60)

    # Load model and data
    print("\n[1/3] Loading model and data...")
    with open("models/saved/xgboost_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("models/saved/feature_cols.pkl", "rb") as f:
        feature_cols = pickle.load(f)

    test_df = pd.read_parquet(os.path.join(PROCESSED, "test.parquet"))
    X_test = test_df[feature_cols].astype(np.float32)

    # Use a sample for SHAP (it's slow on large data)
    sample_size = min(500, len(X_test))
    X_sample = X_test.sample(n=sample_size, random_state=42)

    # Build explainer
    print("\n[2/3] Computing SHAP values...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # Global importance plot
    print("\n[3/3] Generating plots...")
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample,
                      plot_type="bar", max_display=15, show=False)
    plt.title("FinShield — Top 15 Fraud Indicators (SHAP)"
              + (" [SMOKE]" if is_smoke else ""))
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES, "shap_global.png"), dpi=150,
                bbox_inches="tight")
    plt.close()
    print(f"   Saved: {FIGURES}/shap_global.png")

    # Save explainer for API use
    with open("models/saved/shap_explainer.pkl", "wb") as f:
        pickle.dump(explainer, f)
    print("   Saved: models/saved/shap_explainer.pkl")

    # Example explanations
    print("\nExample explanation:")
    sample_row = X_sample.iloc[[0]]
    reasons = explain_single(model, explainer, sample_row, feature_cols)
    for r in reasons:
        print(f"   {r['description']}: {r['direction']} (impact: {r['impact']})")

    print(f"\n{'=' * 60}")
    print("SHAP explainability complete!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()