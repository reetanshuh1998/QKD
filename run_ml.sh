#!/bin/bash
echo "Waiting for generate_qkd_dataset.py to finish..."
pid=$(pgrep -f generate_qkd_dataset.py)
if [ -n "$pid" ]; then
    echo "Found PID $pid, waiting..."
    tail --pid=$pid -f /dev/null
fi
echo "Generation complete. Starting ML pipeline..."
cd /home/reet/QKD/src
source ../venv/bin/activate

echo "--- Running Feature Engineering ---"
python3 feature_engineering.py

echo "--- Running Model Training ---"
python3 model_training.py

echo "--- Running Hyperparameter Tuning ---"
python3 hyperparameter_tuning.py

# Final verification
cp ../models/plots/*.png /home/reet/.gemini/antigravity/brain/3a198833-1122-4649-aafd-25803c686c5a/
echo "ML Pipeline Complete!"
