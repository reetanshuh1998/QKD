#!/bin/bash
echo "Activating virtual environment..."
cd /home/reet/QKD
source venv/bin/activate

echo "Installing Qiskit dependencies..."
pip install qiskit qiskit-aer seaborn pandas numpy tqdm

echo "Checking package installation..."
python3 -c "import qiskit; import qiskit_aer; print('Qiskit and Aer installed successfully.')"

echo "Running custom BB84 dataset generation..."
cd src
python3 generate_qkd_dataset.py

echo "Generation Script Finished."
