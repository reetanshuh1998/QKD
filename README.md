# QKD Eavesdropper Detection: Hybrid Autoencoder–XGBoost Architecture

## Overview

This repository implements a machine learning pipeline for detecting eavesdropping attacks in Quantum Key Distribution (QKD) systems using the decoy-state BB84 protocol with Weak Coherent Pulse (WCP) sources.

The system consists of:
1. A **Monte Carlo BB84 simulator** generating physically motivated session-level features
2. A **hybrid detection architecture** combining a deep autoencoder (anomaly feature extraction) with XGBoost (classification)
3. An **adversarial resilience evaluation** demonstrating robustness against gradient-based evasion

## Dataset Generation (Decoy-State BB84 Simulator)

The simulator (`src/generate_qkd_dataset.py`) produces synthetic QKD session data without relying on circuit simulators. Each session generates 10,000 pulses and records 30 physical observables.

### Physics Modeled
- **Poisson photon statistics**: Signal ($\mu = 0.5$), decoy ($\nu = 0.1$), and vacuum intensities with probabilities 0.80/0.15/0.05
- **Fiber channel loss**: Standard telecom model at 0.2 dB/km, distances 5–50 km
- **Detector model**: Lumped efficiency $\eta_{\text{det}} = 0.15$, dark count probability $10^{-6}$ per gate, intrinsic misalignment error 0.015

### Feature Vector (30 dimensions)
| Category | Features |
|----------|----------|
| Session statistics | `Pulses_Sent`, `Total_Clicks`, `Sifted_Bits` |
| QKD monitoring | `Base_Mismatch_Rate`, `QBER_Total`, `QBER_Z`, `QBER_X`, `Sifted_Bit_Entropy` |
| Decoy-state gains | `Q_mu`, `Q_nu`, `Q_0` |
| Per-intensity QBER | `E_mu`, `E_nu`, `E_0` |
| Basis-conditional (signal) | `Q_mu_Z`, `Q_mu_X`, `E_mu_Z`, `E_mu_X` |
| Raw counts | `Clicks_mu`, `Clicks_nu`, `Clicks_0`, `DetMatch_mu`, `DetMatch_nu`, `DetMatch_0` |
| Hardware monitors | `Rx_Power_Mean`, `Rx_Power_Std`, `Timing_Mean_us`, `Timing_Std_us`, `Double_Click_Rate`, `Monitor_Alarm_Rate` |

`Distance_km` is stored for analysis but excluded from ML input features.

### Attack Classes (8 total)
| Class | Observable Signature |
|-------|---------------------|
| `normal` | Baseline WCP exchange |
| `mitm_attack` | Elevated QBER; timing jitter increase |
| `pns_attack` | Single-photon yield suppression; distorted $Q_\nu/Q_\mu$ ratio |
| `trojan_horse_attack` | Elevated Rx optical power; increased alarm rate |
| `wavelength_dependent_trojan_attack` | Basis-dependent efficiency asymmetry |
| `rng_attack` | Reduced sifted-bit entropy; basis bias |
| `detector_blinding_attack` | Very high Rx power; QBER → 0; double-click rate → 0 |
| `combined_attack` | Multi-feature distortion (PNS + MITM + Trojan elements) |

## ML Pipeline

### Architecture
- **Autoencoder** (`30 → 64 → 32 → 16 → 32 → 64 → 30`): Trained only on normal sessions. Learns the latent manifold of legitimate BB84 statistics. Reconstruction MSE serves as anomaly score.
- **Hybrid XGBoost**: Classifies the concatenation of original features (30) + latent vector (16) + MSE (1) = 47-dimensional input
- **DNN baseline**: 128–64–32–16 dense network with LeakyReLU, batch normalization, and dropout

### Data Leakage Prevention
StandardScaler is fit exclusively on the training split (80%) and applied to the test split (20%). The autoencoder is trained only on the normal-class subset of the training data.

## Running the Pipeline

```bash
# Install dependencies
pip install -r requirements.txt

# Full pipeline (default: 1000 sessions/class)
bash run_pipeline.sh

# Paper-quality run (5000 sessions/class)
bash run_pipeline.sh 5000 10000
```

### Pipeline Steps
1. `generate_qkd_dataset.py` — Monte Carlo simulation
2. `feature_engineering.py` — Label encoding
3. `model_training.py` — Autoencoder + hybrid XGBoost training
4. `hyperparameter_tuning.py` — RandomizedSearchCV + evaluation plots
5. `baseline_comparison.py` — Standalone XGBoost and DNN comparison
6. `adversarial_attack.py` — White-box gradient evasion test
7. `mechanistic_signatures.py` — Physical observable plots
8. `shap_analysis.py` — SHAP explainability analysis

## Output Plots

All plots are saved to `models/plots/`:

| Plot | Description |
|------|-------------|
| `attack_signatures_across_observables.png` | 4-panel scatter/KDE of physical observables by attack class |
| `hybrid_feature_importance.png` | XGBoost feature importance on the 47-dim hybrid input |
| `roc_curves.png` | One-vs-rest ROC curves for all 8 classes |
| `confusion_matrix.png` | Test-set confusion matrix |
| `architecture_comparison.png` | Accuracy comparison across three architectures |
| `adversarial_evasion.png` | Detection rate before/after gradient evasion |
| `paper_figures/shap_summary_bar.png` | SHAP global feature importance |

## Reproducibility

All results are reproducible with fixed random seed (`seed=42`, `random_state=42` throughout). Tested with Python 3.12, TensorFlow 2.x, XGBoost 2.x, scikit-learn 1.x.

## License

MIT License — see [LICENSE](LICENSE).
