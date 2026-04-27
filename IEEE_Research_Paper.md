# Adversarially Resilient Eavesdropper Detection in Decoy-State BB84 QKD via Hybrid Autoencoder–XGBoost Classification

**Keywords:** Quantum Key Distribution (QKD), Decoy-State BB84 Protocol, Weak Coherent Pulses (WCP), Adversarial Machine Learning, Autoencoders, XGBoost, Physical Layer Security.

---

## Abstract

Quantum Key Distribution (QKD) offers information-theoretic security grounded in quantum mechanics, yet practical implementations using Weak Coherent Pulse (WCP) sources and avalanche photodiode (APD) detectors introduce exploitable hardware vulnerabilities. Attacks such as Photon-Number Splitting (PNS), detector blinding, and Trojan-horse probing can evade conventional Quantum Bit Error Rate (QBER) monitoring. Machine learning classifiers trained on simplistic simulations lack the physical fidelity of decoy-state protocols and remain vulnerable to gradient-based adversarial evasion.

We present a hybrid detection architecture combining a deep autoencoder trained exclusively on normal QKD sessions with an XGBoost gradient-boosted classifier. The system is trained on a purpose-built Monte Carlo simulator implementing the three-intensity decoy-state BB84 protocol, producing 30 physically motivated session-level observables including per-intensity gains ($Q_\mu$, $Q_\nu$, $Q_0$), basis-conditional QBERs, double-click rates, and optical power monitor readings. Eight operational classes—one normal and seven attack scenarios—are distinguished. The autoencoder compresses the 30-dimensional input to a 16-dimensional latent representation; the per-sample reconstruction MSE serves as an anomaly score. The concatenated 47-dimensional hybrid feature vector is classified by XGBoost. In benchmarking, a standalone DNN achieves the highest accuracy, while the hybrid model achieves comparable performance with the added benefit of SHAP-based feature-level interpretability. Critically, gradient-based adversarial evasion attacks that successfully fool the autoencoder fail to transfer through XGBoost's non-differentiable decision boundaries, demonstrating structural robustness against white-box adversaries.

---

## I. Introduction

Quantum Key Distribution enables two parties to establish a shared secret key with security guaranteed by quantum mechanics. The BB84 protocol detects eavesdropping through statistical monitoring of the quantum bit error rate (QBER). In theory, any interception disturbs the quantum states and raises the QBER above a security threshold.

In practice, QKD systems use attenuated laser sources producing Weak Coherent Pulses (WCP), which follow Poisson photon-number statistics and occasionally emit multi-photon pulses. An adversary can exploit these multi-photon events via Photon-Number Splitting (PNS) attacks [1], extracting key information without measurably increasing the QBER. Additional hardware-layer attacks include detector blinding [3], Trojan-horse probing [5], and timing-channel exploits [2].

The decoy-state protocol [6, 7] mitigates PNS attacks by transmitting pulses at multiple intensity levels and monitoring the per-intensity gains and error rates. However, sophisticated attacks targeting other hardware components may still evade decoy-state analysis alone.

ML approaches offer a complementary detection layer but face two challenges: (1) training data must faithfully capture decoy-state physics, and (2) ML classifiers are susceptible to adversarial evasion [4]. Tree-based ensembles exhibit natural robustness to gradient-based evasion due to non-differentiable decision boundaries [5].

This paper addresses both challenges by constructing a physics-faithful simulator and a hybrid autoencoder–XGBoost architecture providing both accuracy and adversarial resilience.

---

## II. Dataset Generation

### A. Decoy-State BB84 Simulator

We implement a session-level Monte Carlo simulator for the three-intensity decoy-state BB84 protocol. Each session generates $N = 10,000$ pulses with the following model:

- **Photon-number sampling**: $n \sim \text{Poisson}(\mu)$ with $\mu = 0.5$ (signal), $\nu = 0.1$ (decoy), $0$ (vacuum); selected with probabilities 0.80/0.15/0.05
- **Channel transmission**: $\eta_{\text{ch}} = 10^{-\alpha d / 10}$, $\alpha = 0.2$ dB/km, $d \in [5, 50]$ km
- **Detector model**: $\eta_{\text{det}} = 0.15$, $p_{\text{dark}} = 10^{-6}$, $e_{\text{misalign}} = 0.015$
- **Click probability**: $p_{\text{click}} = 1 - (1 - (1 - (1-\eta_{\text{sys}})^n))(1 - p_{\text{dark}})$

Distance is used internally for channel loss but excluded from ML features.

### B. Session-Level Observables (30 features)

| Category | Features |
|----------|----------|
| Session statistics | `Pulses_Sent`, `Total_Clicks`, `Sifted_Bits` |
| QKD monitoring | `Base_Mismatch_Rate`, `QBER_Total`, `QBER_Z`, `QBER_X`, `Sifted_Bit_Entropy` |
| Decoy-state gains | `Q_mu`, `Q_nu`, `Q_0` |
| Per-intensity QBER | `E_mu`, `E_nu`, `E_0` |
| Basis-conditional | `Q_mu_Z`, `Q_mu_X`, `E_mu_Z`, `E_mu_X` |
| Raw counts | `Clicks_mu`, `Clicks_nu`, `Clicks_0`, `DetMatch_mu`, `DetMatch_nu`, `DetMatch_0` |
| Hardware monitors | `Rx_Power_Mean`, `Rx_Power_Std`, `Timing_Mean_us`, `Timing_Std_us`, `Double_Click_Rate`, `Monitor_Alarm_Rate` |

### C. Attack Classes

| Class | Primary Signature |
|-------|-------------------|
| Normal | Baseline |
| MITM | ↑ QBER, ↑ timing jitter |
| PNS | Suppressed single-photon yield, distorted $Q_\nu/Q_\mu$ |
| Trojan Horse | ↑ Rx power, ↑ alarm rate |
| Wavelength-Dep. Trojan | $Q_{\mu,Z} \neq Q_{\mu,X}$ asymmetry |
| RNG Bias | ↓ sifted-bit entropy, basis imbalance |
| Detector Blinding | ↑↑ Rx power, QBER → 0, double clicks → 0 |
| Combined | Multi-feature distortion |

![Mechanistic Attack Signatures](models/plots/attack_signatures_across_observables.png)
*Fig 1. Attack signatures across physical observables: (A) decoy consistency, (B) yield ratio density, (C) optical power vs. alarm rate, (D) timing distribution.*

---

## III. Hybrid Architecture

### A. Autoencoder Feature Extraction

Architecture: $30 \to 64 \to 32 \to 16 \to 32 \to 64 \to 30$ with batch normalization and ReLU. Trained exclusively on normal-class training data with MSE loss, Adam optimizer, early stopping (patience 5). The 16-dimensional latent vector and per-sample reconstruction MSE provide anomaly-sensitive features.

### B. Hybrid Feature Construction

Original 30 scaled features + 16 latent dimensions + 1 MSE scalar = **47-dimensional hybrid vector**.

### C. XGBoost Classification

XGBoost (300 estimators, max depth 8, learning rate 0.05) trained on the hybrid vector. Hyperparameters refined via 3-fold stratified RandomizedSearchCV (10 iterations). Tree-based classification provides SHAP interpretability [8] and non-differentiable decision boundaries.

![Feature Importance](models/plots/hybrid_feature_importance.png)
*Fig 2. XGBoost feature importance showing contributions from physical features, latent dimensions, and reconstruction MSE.*

---

## IV. Results

### Classification Performance

| Architecture | Test Accuracy |
|-------------|---------------|
| Standalone XGBoost | 85.94% |
| Hybrid (AE + XGBoost) | 85.88% |
| Deep Neural Network | **88.88%** |

The DNN achieves the highest accuracy through end-to-end optimization. The hybrid model trades a small accuracy margin for SHAP interpretability and adversarial resilience.

![ROC Curves](models/plots/roc_curves.png)
*Fig 3. One-vs-rest ROC curves with per-class AUC.*

### SHAP Explainability

SHAP TreeExplainer reveals that reconstruction MSE, decoy-state gains, and optical power monitors are the most discriminative features, consistent with the expected physics of PNS and Trojan-horse attacks.

---

## V. Adversarial Resilience

A white-box adversary with full autoencoder access applies 40 epochs of gradient descent (Adam, lr=0.08) to minimize reconstruction MSE, attempting to disguise attack traffic as normal. While this reduces the autoencoder's anomaly score, XGBoost maintains detection because: (1) gradients cannot propagate through tree-based decision boundaries, and (2) raw physical features retain attack signatures that the trees detect independently.

---

## VI. Limitations

- Simulator models fiber loss and dark counts but not depolarization, phase noise, or finite-key effects
- Dataset is synthetic; experimental validation is needed
- MITM timing perturbation models generic interception delay, not a physically distinct time-shift attack

---

## VII. Conclusion

We demonstrated a hybrid autoencoder–XGBoost architecture for QKD eavesdropper detection trained on a physics-faithful decoy-state BB84 simulator. The hybrid model provides comparable accuracy to DNNs with SHAP interpretability and structural immunity to gradient-based adversarial evasion.

---

### References

[1] G. Brassard et al., "Limitations on practical quantum cryptography," *Phys. Rev. Lett.*, 2000.
[2] B. Qi et al., "Time-shift attack in practical quantum cryptosystems," *Quantum Inf. Comput.*, 2007.
[3] L. Lydersen et al., "Hacking commercial quantum cryptography systems," *Nature Photon.*, 2010.
[4] I. J. Goodfellow et al., "Explaining and harnessing adversarial examples," *ICLR*, 2015.
[5] H. Chen et al., "Robust decision trees against adversarial examples," *ICML*, 2019.
[6] H.-K. Lo, X. Ma, and K. Chen, "Decoy state quantum key distribution," *Phys. Rev. Lett.*, 2005.
[7] X. Ma et al., "Practical decoy state for quantum key distribution," *Phys. Rev. A*, 2005.
[8] S. M. Lundberg and S.-I. Lee, "A unified approach to interpreting model predictions," *NeurIPS*, 2017.
[9] T. Chen and C. Guestrin, "XGBoost: A scalable tree boosting system," *KDD*, 2016.
[10] N. Gisin et al., "Quantum cryptography," *Rev. Mod. Phys.*, 2002.
