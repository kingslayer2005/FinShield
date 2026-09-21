# FinShield Makefile
# Works on Linux/Mac. Windows users: use run_all.ps1 or the commands below.

.PHONY: setup train smoke test up down demo lint clean

# --- Environment setup ---
setup:
	uv venv --python 3.11 .venv
	uv pip install -r requirements.txt

# --- Full pipeline (real data) ---
train:
	python data/scripts/preprocess.py
	python models/train_xgboost.py
	python models/train_autoencoder.py
	python models/train_lstm.py || echo "LSTM failed — ensemble will use XGBoost + AE"
	python models/ensemble.py
	python models/evaluate.py
	python explainability/shap_explainer.py
	python scripts/generate_readme_table.py

# --- Smoke test (synthetic data, tiny models) ---
smoke:
	SMOKE=1 python tests/fixtures/make_synthetic_data.py
	SMOKE=1 python data/scripts/preprocess.py
	SMOKE=1 python models/train_xgboost.py
	SMOKE=1 python models/train_autoencoder.py
	SMOKE=1 python models/train_lstm.py || echo "LSTM failed"
	SMOKE=1 python models/ensemble.py
	SMOKE=1 python models/evaluate.py
	SMOKE=1 python explainability/shap_explainer.py
	SMOKE=1 python -m pytest tests/ -v --tb=short

# --- Tests ---
test:
	SMOKE=1 python -m pytest tests/ -v --tb=short

# --- Docker services ---
up:
	docker compose up -d

down:
	docker compose down

# --- Demo ---
demo:
	streamlit run demo/app.py

# --- Linting ---
lint:
	ruff check . --fix

# --- Clean ---
clean:
	rm -rf models/saved/ data/processed/ data/synthetic/ reports_smoke/ mlruns/ __pycache__/
