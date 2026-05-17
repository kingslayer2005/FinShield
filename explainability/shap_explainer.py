
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings("ignore")

print("FinShield SHAP Explainability Starting...")


with open("models/saved/xgboost_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/saved/feature_cols.pkl", "rb") as f:
    feature_cols = pickle.load(f)

df = pd.read_csv("data/processed/finshield_sample.csv")
X  = df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
y  = df["isFraud"]

# Use small sample for SHAP (it's slow on large data)
X_sample = X.sample(n=500, random_state=42)


print("\nComputing SHAP values (takes 1-2 mins)...")
explainer   = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_sample)

print("Generating global importance plot...")
os.makedirs("docs", exist_ok=True)

plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_sample,
                  plot_type="bar",
                  max_display=15,
                  show=False)
plt.title("FinShield — Top 15 Fraud Indicators (SHAP)", fontsize=13)
plt.tight_layout()
plt.savefig("docs/shap_global.png", dpi=150, bbox_inches="tight")
plt.close()
print("   Saved: docs/shap_global.png")


print("\nExplaining individual transactions...")

def explain_transaction(idx, X_data, shap_vals, threshold=0.5):
    """Generate human-readable explanation for one transaction."""
    pred_prob = model.predict_proba(X_data.iloc[[idx]])[:, 1][0]
    is_fraud  = pred_prob > threshold

    # Get top contributing features
    feature_shap = list(zip(feature_cols, shap_vals[idx]))
    feature_shap.sort(key=lambda x: abs(x[1]), reverse=True)
    top_features = feature_shap[:5]

    # Calculate contribution percentages
    total = sum(abs(s) for _, s in top_features) + 1e-10
    contributions = [(f, s, abs(s)/total*100) for f, s in top_features]

    print(f"\n{'='*55}")
    print(f"  Transaction #{idx}")
    print(f"  Fraud Probability : {pred_prob:.2%}")
    print(f"  Decision          : {'🚨 FRAUD' if is_fraud else '✅ LEGIT'}")
    print(f"  {'─'*45}")
    print(f"  Top Reasons:")
    for feat, shap_val, pct in contributions:
        direction = "↑ increases" if shap_val > 0 else "↓ decreases"
        print(f"    {feat[:30]:<30} {direction} risk ({pct:.1f}%)")
    print(f"{'='*55}")

    return {
        "transaction_idx" : idx,
        "fraud_probability": round(pred_prob, 4),
        "is_fraud"        : bool(is_fraud),
        "risk_level"      : "HIGH" if pred_prob > 0.8 else
                            "MEDIUM" if pred_prob > 0.5 else "LOW",
        "top_reasons"     : [
            {"feature": f, "contribution_pct": round(p, 1),
             "direction": "risk_increase" if s > 0 else "risk_decrease"}
            for f, s, p in contributions
        ]
    }

# Explain 3 transactions — 2 fraud, 1 legit
fraud_idx = df[df["isFraud"] == 1].index[:2].tolist()
legit_idx = df[df["isFraud"] == 0].index[:1].tolist()

results = []
for idx in fraud_idx + legit_idx:
    sample_pos = X_sample.index.get_loc(idx) if idx in X_sample.index else 0
    result = explain_transaction(sample_pos, X_sample,
                                 shap_values)
    results.append(result)


with open("models/saved/shap_explainer.pkl", "wb") as f:
    pickle.dump(explainer, f)

print(f"\nSHAP explainer saved to models/saved/shap_explainer.pkl")
print(f"Global importance chart saved to docs/shap_global.png")
print(f"\nExample API response with explanation:")

import json
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.float32, np.float64, np.integer)):
            return float(obj)
        return super().default(obj)

print(json.dumps(results[0], indent=2, cls=NumpyEncoder))