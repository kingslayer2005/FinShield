<# 
  run_all.ps1
  One-command full pipeline for FinShield.
  Requires: data/raw/train_transaction.csv and data/raw/train_identity.csv

  Usage:
    .\run_all.ps1              # full pipeline on real data
    $env:SMOKE="1"; .\run_all.ps1  # smoke test on synthetic data

  This script:
    1. Generates synthetic data (if SMOKE=1)
    2. Runs preprocessing
    3. Trains XGBoost, Autoencoder, LSTM
    4. Runs ensemble model gate + stacking
    5. Evaluates on test set → reports/metrics.json
    6. Generates SHAP explanations
    7. Runs drift check
    8. Runs all pytest tests
    9. Regenerates README results table
#>

$ErrorActionPreference = "Stop"

# Activate venv if it exists
if (Test-Path ".venv\Scripts\Activate.ps1") {
    . .venv\Scripts\Activate.ps1
}

$is_smoke = $env:SMOKE -eq "1"
$prefix = if ($is_smoke) { "[SMOKE] " } else { "" }

Write-Host "============================================================"
Write-Host "  ${prefix}FinShield Full Pipeline"
Write-Host "============================================================"

# Step 0: Generate synthetic data (smoke mode only)
if ($is_smoke) {
    Write-Host "`n>>> Generating synthetic data..."
    python tests/fixtures/make_synthetic_data.py
    if ($LASTEXITCODE -ne 0) { Write-Error "Synthetic data generation failed"; exit 1 }
}

# Step 1: Check for raw data (real mode only)
if (-not $is_smoke) {
    if (-not (Test-Path "data/raw/train_transaction.csv")) {
        Write-Error "data/raw/train_transaction.csv not found! Download Kaggle data first."
        exit 1
    }
}

# Step 2: Preprocess
Write-Host "`n>>> Preprocessing..."
python data/scripts/preprocess.py
if ($LASTEXITCODE -ne 0) { Write-Error "Preprocessing failed"; exit 1 }

# Step 3: Train XGBoost
Write-Host "`n>>> Training XGBoost..."
python models/train_xgboost.py
if ($LASTEXITCODE -ne 0) { Write-Error "XGBoost training failed"; exit 1 }

# Step 4: Train Autoencoder
Write-Host "`n>>> Training Autoencoder..."
python models/train_autoencoder.py
if ($LASTEXITCODE -ne 0) { Write-Error "Autoencoder training failed"; exit 1 }

# Step 5: Train LSTM
Write-Host "`n>>> Training LSTM..."
python models/train_lstm.py
if ($LASTEXITCODE -ne 0) { 
    Write-Warning "LSTM training failed — ensemble will use XGBoost + AE only"
}

# Step 6: Ensemble (model gate + stacking)
Write-Host "`n>>> Building ensemble..."
python models/ensemble.py
if ($LASTEXITCODE -ne 0) { Write-Error "Ensemble failed"; exit 1 }

# Step 7: Evaluate on test set
Write-Host "`n>>> Evaluating on test set..."
python models/evaluate.py
if ($LASTEXITCODE -ne 0) { Write-Error "Evaluation failed"; exit 1 }

# Step 8: SHAP explanations
Write-Host "`n>>> Computing SHAP explanations..."
python explainability/shap_explainer.py
if ($LASTEXITCODE -ne 0) { Write-Warning "SHAP failed — non-critical" }

# Step 9: Drift check (may fail without Postgres)
Write-Host "`n>>> Running drift check..."
python monitoring/drift.py
if ($LASTEXITCODE -ne 0) { Write-Warning "Drift check failed — needs Postgres" }

# Step 10: Run tests
Write-Host "`n>>> Running tests..."
python -m pytest tests/ -v --tb=short
if ($LASTEXITCODE -ne 0) { Write-Warning "Some tests failed" }

# Step 11: Generate README table
Write-Host "`n>>> Generating README results table..."
python scripts/generate_readme_table.py

Write-Host "`n============================================================"
Write-Host "  ${prefix}Pipeline complete!"
Write-Host "============================================================"

if (-not $is_smoke) {
    $metrics_path = "reports/metrics.json"
} else {
    $metrics_path = "reports_smoke/metrics.json"
}

if (Test-Path $metrics_path) {
    Write-Host "`nMetrics saved to: $metrics_path"
    Get-Content $metrics_path | ConvertFrom-Json | ConvertTo-Json -Depth 3
}
