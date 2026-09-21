"""
demo/app.py
Streamlit demo for FinShield fraud detection.
Deployable on Hugging Face Spaces with no Kafka dependency.

Features:
  - Pick or edit a test transaction
  - See fraud score, decision, and SHAP waterfall plot
  - View metrics page from reports/metrics.json

Run locally:
    streamlit run demo/app.py
    SMOKE=1 streamlit run demo/app.py
"""

import os
import sys
import json
import pickle
import warnings

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="FinShield — Fraud Detection",
    page_icon="🛡️",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Custom CSS for premium look
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    }
    .metric-card {
        background: rgba(255,255,255,0.05);
        border-radius: 12px;
        padding: 20px;
        border: 1px solid rgba(255,255,255,0.1);
        backdrop-filter: blur(10px);
    }
    .decision-approve { color: #00e676; font-size: 28px; font-weight: bold; }
    .decision-review  { color: #ffc107; font-size: 28px; font-weight: bold; }
    .decision-block   { color: #ff1744; font-size: 28px; font-weight: bold; }
    h1 { color: #e0e0e0 !important; }
    h2, h3 { color: #b0b0b0 !important; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Load models (cached)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_models():
    """Load all model artifacts for the demo."""
    state = {}

    try:
        with open("models/saved/xgboost_model.pkl", "rb") as f:
            state["xgb_model"] = pickle.load(f)
        with open("models/saved/feature_cols.pkl", "rb") as f:
            state["feature_cols"] = pickle.load(f)
        with open("models/saved/ensemble_config.pkl", "rb") as f:
            state["ensemble_config"] = pickle.load(f)

        # Try to load preprocessor
        for pp_path in ["data/synthetic/processed/preprocessor.pkl",
                        "data/processed/preprocessor.pkl"]:
            if os.path.exists(pp_path):
                with open(pp_path, "rb") as f:
                    state["preprocessor"] = pickle.load(f)
                break

        # SHAP explainer
        if os.path.exists("models/saved/shap_explainer.pkl"):
            with open("models/saved/shap_explainer.pkl", "rb") as f:
                state["shap_explainer"] = pickle.load(f)

        state["loaded"] = True
    except Exception as e:
        state["loaded"] = False
        state["error"] = str(e)

    return state


@st.cache_data
def load_test_data():
    """Load some test transactions for the demo picker."""
    for path in ["data/synthetic/processed/test.parquet",
                 "data/processed/test.parquet"]:
        if os.path.exists(path):
            df = pd.read_parquet(path)
            return df.head(100)  # show first 100 for picking
    return None


@st.cache_data
def load_metrics():
    """Load metrics.json (smoke or real)."""
    for path in ["reports_smoke/metrics.json", "reports/metrics.json"]:
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
    return None


# ---------------------------------------------------------------------------
# Prediction helper
# ---------------------------------------------------------------------------
def predict_transaction(state, txn_dict):
    """Score a single transaction and return results."""
    feature_cols = state["feature_cols"]
    ensemble_cfg = state["ensemble_config"]
    preprocessor = state.get("preprocessor")

    df = pd.DataFrame([txn_dict])

    # Encode categoricals
    if preprocessor:
        for col, le in preprocessor.get("label_encoders", {}).items():
            if col in df.columns:
                df[col] = df[col].fillna("unknown").astype(str)
                known = set(le.classes_)
                df[col] = df[col].apply(lambda x: x if x in known else "__unseen__")
                df[col] = le.transform(df[col])

    # Add engineered features
    if "TransactionDT" in df.columns:
        df["Transaction_Hour"] = (df["TransactionDT"] / 3600 % 24).astype(int)
        df["Transaction_Day"] = (df["TransactionDT"] / (3600 * 24) % 7).astype(int)
    if "TransactionAmt" in df.columns:
        df["Is_High_Value"] = (df["TransactionAmt"] > 500).astype(int)
        df["TransactionAmt_Log"] = np.log1p(df["TransactionAmt"])

    # Add zero behavioral features
    for w in [3600, 86400]:
        df[f"card_txn_count_{w}s"] = 0
        df[f"card_amt_sum_{w}s"] = 0.0
    df["card_time_since_last"] = -1

    # Ensure all feature columns exist
    for col in feature_cols:
        if col not in df.columns:
            df[col] = np.nan

    X = df[feature_cols].astype(np.float32)

    # XGBoost prediction
    xgb_prob = float(state["xgb_model"].predict_proba(X)[:, 1][0])

    # Stacking ensemble
    stacking_lr = ensemble_cfg["stacking_model"]
    n_components = len(ensemble_cfg["components"])
    stack_input = [xgb_prob] * n_components
    stack_X = np.array([stack_input])
    fraud_prob = float(stacking_lr.predict_proba(stack_X)[:, 1][0])

    # Decision
    if fraud_prob >= ensemble_cfg["block_threshold"]:
        decision = "block"
    elif fraud_prob >= ensemble_cfg["review_threshold"]:
        decision = "review"
    else:
        decision = "approve"

    # SHAP values
    shap_values = None
    if "shap_explainer" in state:
        try:
            sv = state["shap_explainer"].shap_values(X)
            if isinstance(sv, list):
                sv = sv[1]
            shap_values = sv[0]
        except Exception:
            pass

    return {
        "fraud_probability": fraud_prob,
        "xgb_probability": xgb_prob,
        "decision": decision,
        "shap_values": shap_values,
        "feature_cols": feature_cols,
        "feature_values": X.iloc[0].values,
    }


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------
def main():
    st.title("🛡️ FinShield — Real-Time Fraud Detection")
    st.markdown("*AI-powered transaction scoring with explainable decisions*")

    # Sidebar navigation
    page = st.sidebar.radio("Navigate", ["🔍 Score Transaction", "📊 Model Metrics"])

    state = load_models()

    if not state.get("loaded"):
        st.error(f"Models not loaded: {state.get('error', 'Unknown error')}")
        st.info("Run the training pipeline first: `SMOKE=1 python models/train_xgboost.py`")
        return

    if page == "🔍 Score Transaction":
        score_page(state)
    else:
        metrics_page()


def score_page(state):
    """Transaction scoring page with SHAP waterfall."""
    st.header("Score a Transaction")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Transaction Details")

        # Load test data for picker
        test_data = load_test_data()

        if test_data is not None:
            idx = st.selectbox("Pick a test transaction",
                              range(min(20, len(test_data))),
                              format_func=lambda i: f"Txn #{i+1} (${test_data.iloc[i]['TransactionAmt']:.2f})")
            default_row = test_data.iloc[idx]
        else:
            default_row = None

        # Editable fields
        amt = st.number_input("Amount ($)", min_value=0.01, max_value=50000.0,
                             value=float(default_row["TransactionAmt"]) if default_row is not None else 150.0)
        card1 = st.number_input("Card ID", min_value=1000, max_value=99999,
                               value=int(default_row["card1"]) if default_row is not None else 5000)
        email = st.selectbox("Email Domain",
                            ["gmail.com", "yahoo.com", "hotmail.com", "outlook.com",
                             "protonmail.com", "newmail.com", "(none)"])
        device = st.selectbox("Device Type", ["desktop", "mobile", "smartwatch", "(none)"])
        product = st.selectbox("Product Code", ["W", "H", "C", "S", "R"])

        # Build transaction dict
        txn = {
            "TransactionAmt": amt,
            "card1": card1,
            "TransactionDT": int(default_row["TransactionDT"]) if default_row is not None else 100000,
            "ProductCD": product,
            "P_emaildomain": email if email != "(none)" else None,
            "DeviceType": device if device != "(none)" else None,
        }

        # Add other columns from default row if available
        if default_row is not None:
            for col in default_row.index:
                if col not in txn and col not in ["isFraud"]:
                    txn[col] = default_row[col]

        scored = st.button("🔍 Score Transaction", type="primary")

    with col2:
        if scored:
            result = predict_transaction(state, txn)
            prob = result["fraud_probability"]
            decision = result["decision"]

            st.subheader("Result")

            # Decision banner
            decision_class = f"decision-{decision}"
            decision_emoji = {"approve": "✅", "review": "⚠️", "block": "🚨"}
            st.markdown(
                f'<div class="metric-card"><span class="{decision_class}">'
                f'{decision_emoji.get(decision, "")} {decision.upper()}</span></div>',
                unsafe_allow_html=True,
            )

            # Metrics
            mcol1, mcol2 = st.columns(2)
            mcol1.metric("Fraud Probability", f"{prob:.1%}")
            mcol2.metric("XGBoost Score", f"{result['xgb_probability']:.1%}")

            # SHAP waterfall plot
            if result["shap_values"] is not None:
                st.subheader("Why This Decision (SHAP)")

                sv = result["shap_values"]
                fc = result["feature_cols"]
                fv = result["feature_values"]

                # Get top 10 by absolute SHAP value
                top_idx = np.argsort(np.abs(sv))[-10:][::-1]

                fig = go.Figure(go.Waterfall(
                    name="SHAP",
                    orientation="h",
                    y=[fc[i][:25] for i in top_idx],
                    x=[sv[i] for i in top_idx],
                    connector={"line": {"color": "rgba(63, 63, 63, 0.3)"}},
                    decreasing={"marker": {"color": "#00e676"}},
                    increasing={"marker": {"color": "#ff1744"}},
                ))
                fig.update_layout(
                    title="Top 10 Feature Contributions",
                    template="plotly_dark",
                    height=400,
                    margin=dict(l=150),
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("SHAP explainer not loaded — run `python explainability/shap_explainer.py`")


def metrics_page():
    """Model metrics page from reports/metrics.json."""
    st.header("📊 Model Metrics")

    metrics = load_metrics()

    if metrics is None:
        st.warning("No metrics found. Run evaluation first.")
        st.code("SMOKE=1 python models/evaluate.py", language="bash")
        return

    is_smoke = metrics.get("metadata", {}).get("is_smoke", False)
    if is_smoke:
        st.warning("⚠️ These are SMOKE metrics from synthetic data. Not real results.")

    # Results table
    st.subheader("Test Set Results")
    rows = []
    for model_name in ["xgboost", "autoencoder", "lstm", "ensemble"]:
        if model_name in metrics and "test_pr_auc" in metrics[model_name]:
            m = metrics[model_name]
            rows.append({
                "Model": model_name.title(),
                "PR-AUC": f"{m['test_pr_auc']:.4f}",
                "ROC-AUC": f"{m['test_roc_auc']:.4f}",
                "In Ensemble": "✓" if m.get("included_in_ensemble", model_name in ["xgboost", "ensemble"]) else "✗",
            })
    if rows:
        st.table(pd.DataFrame(rows))

    # Ensemble details
    if "ensemble" in metrics:
        ens = metrics["ensemble"]
        st.subheader("Ensemble Details")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("PR-AUC", f"{ens.get('test_pr_auc', 0):.4f}")
        col2.metric("F1 Score", f"{ens.get('test_f1', 0):.4f}")
        col3.metric("Precision", f"{ens.get('test_precision', 0):.4f}")
        col4.metric("Recall", f"{ens.get('test_recall', 0):.4f}")

        # Decision distribution
        st.subheader("Decision Distribution")
        decisions = {
            "Approve": ens.get("approve_count", 0),
            "Review": ens.get("review_count", 0),
            "Block": ens.get("block_count", 0),
        }
        fig = go.Figure(go.Bar(
            x=list(decisions.keys()),
            y=list(decisions.values()),
            marker_color=["#00e676", "#ffc107", "#ff1744"],
        ))
        fig.update_layout(
            template="plotly_dark",
            height=300,
            yaxis_title="Count",
        )
        st.plotly_chart(fig, use_container_width=True)

    # Show figures if they exist
    for fig_dir in ["reports_smoke/figures", "reports/figures"]:
        if os.path.isdir(fig_dir):
            st.subheader("Evaluation Plots")
            for fname in ["pr_curve.png", "roc_curve.png", "confusion_matrix.png"]:
                fpath = os.path.join(fig_dir, fname)
                if os.path.exists(fpath):
                    st.image(fpath, caption=fname.replace("_", " ").replace(".png", "").title())
            break


if __name__ == "__main__":
    main()
