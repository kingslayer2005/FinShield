"""
scripts/generate_readme_table.py
Reads reports/metrics.json and generates the results table + resume bullets
for the README.  If no real metrics exist, outputs a placeholder.

Run from repo root:
    python scripts/generate_readme_table.py
"""

import os
import json
import sys


def generate_table(metrics_path: str = "reports/metrics.json") -> str:
    """Generate a markdown results table from metrics.json.

    Returns:
        Markdown string for the results section.
    """
    if not os.path.exists(metrics_path):
        return (
            "## Results\n\n"
            "> **Results pending full-data run.**\n"
            "> Run `make train` or `.\\run_all.ps1` with real data to generate metrics.\n"
        )

    with open(metrics_path) as f:
        metrics = json.load(f)

    is_smoke = metrics.get("metadata", {}).get("is_smoke", False)
    if is_smoke:
        return (
            "## Results\n\n"
            "> **Results pending full-data run.**\n"
            "> Current metrics are from synthetic smoke data and are not shown.\n"
        )

    # Build table
    lines = [
        "## Results\n",
        "| Model | PR-AUC | ROC-AUC | In Ensemble |",
        "|-------|--------|---------|-------------|",
    ]

    for model_name in ["xgboost", "autoencoder", "lstm", "ensemble"]:
        if model_name in metrics and "test_pr_auc" in metrics[model_name]:
            m = metrics[model_name]
            pr = m["test_pr_auc"]
            roc = m["test_roc_auc"]
            incl = m.get("included_in_ensemble", model_name in ["xgboost", "ensemble"])
            mark = "✓" if incl else "✗"
            lines.append(f"| {model_name.title()} | {pr:.4f} | {roc:.4f} | {mark} |")

    lines.append("")

    # Ensemble details
    ens = metrics.get("ensemble", {})
    if ens:
        lines.append(f"**Ensemble F1:** {ens.get('test_f1', 0):.4f} | "
                      f"**Precision:** {ens.get('test_precision', 0):.4f} | "
                      f"**Recall:** {ens.get('test_recall', 0):.4f}")
        lines.append(f"\nTest set: {metrics['metadata']['test_rows']:,} transactions "
                      f"({metrics['metadata']['test_fraud_rate']*100:.2f}% fraud)")

    return "\n".join(lines)


def generate_resume_bullets(metrics_path: str = "reports/metrics.json") -> str:
    """Generate 3 resume bullets grounded in real metrics.

    Returns:
        Markdown bullet list, or placeholder if no real metrics.
    """
    if not os.path.exists(metrics_path):
        return ""

    with open(metrics_path) as f:
        metrics = json.load(f)

    if metrics.get("metadata", {}).get("is_smoke", False):
        return ""

    ens = metrics.get("ensemble", {})
    xgb = metrics.get("xgboost", {})
    meta = metrics.get("metadata", {})

    prauc = ens.get("test_pr_auc", 0)
    f1 = ens.get("test_f1", 0)
    n_test = meta.get("test_rows", 0)
    components = ens.get("components", [])

    bullets = [
        f"- Built a stacking ensemble ({'+'.join(c.title() for c in components)}) "
        f"achieving **{prauc:.4f} PR-AUC** on {n_test:,} held-out test transactions "
        f"from the IEEE-CIS fraud detection dataset",
        f"- Designed a real-time scoring API with Kafka streaming, Redis-based "
        f"behavioral features, and SHAP explanations delivering **{f1:.4f} F1** "
        f"with approve/review/block decisions",
        f"- Implemented MLOps pipeline with Airflow retraining, Evidently drift "
        f"monitoring, Grafana dashboards, and champion/challenger model promotion",
    ]
    return "\n".join(bullets)


if __name__ == "__main__":
    table = generate_table()
    bullets = generate_resume_bullets()
    print(table)
    if bullets:
        print("\n### Resume Bullets\n")
        print(bullets)
