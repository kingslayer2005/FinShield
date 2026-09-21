"""
utils/config.py
Central configuration loader. Reads config.yaml, applies smoke-mode
overrides when SMOKE=1 is set, and provides a single `cfg` dict
everywhere in the codebase.

Usage:
    from utils.config import cfg, is_smoke, get_reports_dir
"""

import os
import yaml

# ---------------------------------------------------------------------------
# Load base config from config.yaml at the repo root
# ---------------------------------------------------------------------------
_CONFIG_PATH = os.path.join(
    os.path.dirname(__file__), "..", "config.yaml"  # utils/ -> repo root
)
with open(_CONFIG_PATH, "r") as _f:
    cfg = yaml.safe_load(_f)

# ---------------------------------------------------------------------------
# Smoke mode: activated by SMOKE=1 env var OR config.yaml smoke: true
# ---------------------------------------------------------------------------
is_smoke = os.environ.get("SMOKE", "0") == "1" or cfg.get("smoke", False)

if is_smoke:
    # Override XGBoost settings with tiny values for fast iteration
    for key, val in cfg.get("xgboost_smoke", {}).items():
        cfg["xgboost"][key] = val

    # Override Autoencoder settings
    for key, val in cfg.get("autoencoder_smoke", {}).items():
        cfg["autoencoder"][key] = val

    # Override LSTM settings
    for key, val in cfg.get("lstm_smoke", {}).items():
        cfg["lstm"][key] = val

    # Point data at synthetic directory instead of real raw data
    cfg["data"]["raw_dir"] = cfg["data"]["synthetic_dir"]

    # Print a clear banner so nobody confuses smoke with real
    print("=" * 60)
    print("  ⚠️  SMOKE MODE — synthetic data, tiny models")
    print("  Results are NOT real. Do not quote them.")
    print("=" * 60)


def get_reports_dir() -> str:
    """Return 'reports_smoke' in smoke mode, 'reports' otherwise.

    This ensures smoke outputs never overwrite real metric files.
    """
    d = "reports_smoke" if is_smoke else "reports"
    os.makedirs(d, exist_ok=True)  # create if it doesn't exist
    return d


def get_processed_dir() -> str:
    """Return the processed data directory, suffixed for smoke mode."""
    if is_smoke:
        d = os.path.join(cfg["data"]["synthetic_dir"], "processed")
    else:
        d = cfg["data"]["processed_dir"]
    os.makedirs(d, exist_ok=True)
    return d


def get_mlflow_uri() -> str:
    """Try the MLflow tracking server; fall back to local file store.

    Returns the URI string to pass to mlflow.set_tracking_uri().
    Logs a warning if the server is unreachable.
    """
    import logging
    logger = logging.getLogger("finshield")

    server_uri = cfg["mlflow"]["tracking_uri"]
    fallback_uri = cfg["mlflow"]["fallback_uri"]

    # In smoke mode, always use local file store (no Docker dependency)
    if is_smoke:
        logger.info("Smoke mode: using local MLflow file store '%s'", fallback_uri)
        return fallback_uri

    # Try to reach the MLflow server
    try:
        import urllib.request
        urllib.request.urlopen(server_uri, timeout=3)
        return server_uri
    except Exception:
        logger.warning(
            "MLflow server at %s unreachable — falling back to local '%s'",
            server_uri, fallback_uri,
        )
        return fallback_uri
