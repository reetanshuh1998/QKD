#!/bin/bash
# QKD Eavesdropper Detection — Full Pipeline
# Usage: bash run_pipeline.sh [--iters N] [--pulses N]
#
# Default: 1000 sessions/class, 10000 pulses/session
# Recommended for paper: --iters 5000 --pulses 10000

set -e

ITERS=${1:-1000}
PULSES=${2:-10000}

echo "=== QKD ML Pipeline ==="
echo "Sessions per class: $ITERS | Pulses per session: $PULSES"

cd "$(dirname "$0")"

# Activate virtual environment if present
if [ -d "venv" ]; then
    source venv/bin/activate
fi

echo ""
echo "[1/8] Generating decoy-state BB84 dataset..."
cd src
python3 generate_qkd_dataset.py --iters "$ITERS" --pulses "$PULSES"

echo ""
echo "[2/8] Feature engineering (label encoding)..."
python3 feature_engineering.py

echo ""
echo "[3/8] Training autoencoder + hybrid XGBoost..."
python3 model_training.py

echo ""
echo "[4/8] Hyperparameter tuning (RandomizedSearchCV)..."
python3 hyperparameter_tuning.py

echo ""
echo "[5/8] Baseline comparison (standalone XGB, DNN)..."
python3 baseline_comparison.py

echo ""
echo "[6/8] Adversarial evasion test..."
python3 adversarial_attack.py

echo ""
echo "[7/8] Mechanistic signature plots..."
python3 mechanistic_signatures.py

echo ""
echo "[8/8] SHAP explainability analysis..."
python3 shap_analysis.py

cd ..
echo ""
echo "=== Pipeline complete. All outputs in data/ and models/ ==="
