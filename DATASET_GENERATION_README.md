# QKD Custom Dataset Generation - Technical Documentation

This repository employs a physically grounded, Decoy-state Monte Carlo Quantum Key Distribution (QKD) simulator to build a robust 8-class machine learning dataset. To meet high-tier journal standards, this simulator explicitly implements Decoy-state BB84 methodologies using Weak Coherent Pulses (WCP), bridging the gap between abstract quantum protocols and physical optical hardware measurements.

---

## 1. Simulator Working Steps (Instruction Manual)

The simulator generates vectorized arrays corresponding to physical QKD transmission sessions. To execute the dataset generator, run:
```bash
python3 src/generate_qkd_dataset.py
```

### Step-by-Step Mechanics:
1. **Pulse Preparation & Poisson Statistics:** Alice prepares photons across millions of pulses using Weak Coherent Pulses (WCP). The simulator natively maps Poisson distributions: $P(n) = \frac{\lambda^n e^{-\lambda}}{n!}$ for Signal ($\mu=0.5$), Decoy ($\nu=0.1$), and Vacuum ($\omega=0.0$) states.
2. **Channel Attenuation:** Mimics realistic optical fiber metrics. Distance is sampled between 10-50km with a standard 0.2 dB/km logarithmic loss limit ($\eta$).
3. **Hardware Sensitivities:** Incorporates true physical thresholds like Bob's global setup efficiency ($\eta_d$) and intrinsic detector dark count probabilities ($p_d$).
4. **Bob's Measurements & Yield Extractions:** Based on survival likelihood equations $Y_{\lambda} = 1 - e^{-\eta_{sys} \lambda} + p_d$, Bob measures and sifts the resulting metrics.
5. **Physical Feature Extraction:** Distinct from logical bit matrices, our generator outputs variables that real experimentalists track:
    * `Yield_Signal` & `Yield_Decoy`: Detection rate for signal vs decoy pulses natively.
    * `QBER_Signal_X` / `QBER_Signal_Z`: The error rates split across localized measurement arrays.
    * `Monitor_Intensity_Mean`: Emulates an active optical incoming-light monitor protecting against bright-light injections.
    * `Double_Click_Rate`: Maps true APD (Avalanche Photodiode) cross-talk/thermal anomalies.
    * `Sifted_Bit_Bias` & `Bob_Basis_Bias`: Extracts inherent statistical deviations from a pristine 50/50 protocol choice.

---

## 2. Simulated Quantum Attacks & Theoretical Grounding

The simulator evaluates the baseline natural protocol (`normal`) alongside 7 documented implementation attacks, directly mapping them to these continuous observables. 

### 1. Intercept-Resend (IR) Attack (`mitm_attack`)
* **Mechanism:** Eve intercepts states, measures them arbitrarily, and collapses the photonic array before regenerating and resending it to Bob. 
* **Statistical Signature:** Fundamentally drives the Quantum Bit Error Rate (QBER) to ~25% as the basis selection mismatch completely fractures mathematical correlation. It also induces significant `Timing_Jitter_Mean` spikes simulating physical intercept delays.
* *Reference: Gisin, N., et al. (2002). Quantum cryptography. Reviews of Modern Physics, 74(1), 145.*

### 2. Photon-Number Splitting (PNS) Attack (`pns_attack`)
* **Mechanism:** Exploits WCP architectures natively. Eve replaces the lossy channel with a lossless equivalent. She categorically blocks transmission of $n=1$ photon states, but flawlessly splits and forwards states where $n \ge 2$. 
* **Statistical Signature:** Since the Decoy state ($\nu=0.1$) statistically holds almost zero multi-photon elements compared to the Signal state ($\mu=0.5$), Eve's suppression violently destroys the proportional fraction of `Yield_Decoy` compared to `Yield_Signal`.
* *Reference: Brassard, G., et al. (2000). Limitations on practical quantum cryptography. Physical Review Letters, 85(6), 1330.*

### 3. Detector Blinding Attack (`detector_blinding_attack`)
* **Mechanism:** Eve targets active APDs, flooding the sensor with continuous-wave (CW) intense optical light, forcing detectors entirely out of quantum-threshold gating into a classical linear saturation array. 
* **Statistical Signature:** Eve completely assumes control over Bob's hardware, meaning Bob only clicks when orchestrated. Thus, the generic `Double_Click_Rate` drops strictly to zero, but the `Monitor_Intensity_Mean` explodes as excess voltage is detected. Because Eve perfectly forces identical states, QBER becomes artificially 0%. 
* *Reference: Lydersen, L., et al. (2010). Hacking commercial quantum cryptography systems by tailored bright illumination. Nature Photonics, 4(10), 686-689.*

### 4. Trojan-Horse Attack (`trojan_horse_attack`)
* **Mechanism:** Eve injects a secondary intense optical probe directly up Alice's transmitter line, analyzing the back-scattered reflection mathematically mapped to internal hardware modulations.
* **Statistical Signature:** Produces extremely severe anomalies within the `Monitor_Intensity_Mean` feature corresponding to incoming probe power thresholds. 
* *Reference: Gisin, N., et al. (2006). Trojan-horse attacks on quantum-key-distribution systems. Physical Review A, 73(2), 022320.*

### 5. Wavelength-Dependent Trojan Attack (`wavelength_dependent_trojan_attack`)
* **Mechanism:** Eve utilizes physical beam splitter sensitivities. By altering the specific wavelength spectrum of the incoming photons, she breaks the 50/50 reflection-transmission ratio at Bob's end natively.
* **Statistical Signature:** Bob's foundational probabilistic matrix is destroyed natively, leading directly to sweeping variances inside `Bob_Basis_Bias` (fracturing it to an 80/20 probability constraint).
* *Reference: Li, H. W., et al. (2011). Attacking a practical QKD system with wavelength-dependent beam splitter... Physical Review A, 84(6), 062308.*

### 6. Laser Seeding / RNG Bias Attack (`rng_attack`)
* **Mechanism:** Eve exploits external hardware elements natively connected to Alice's active Random Number Generator, deliberately altering stochastic seed logic to create predictable transmission geometries. 
* **Statistical Signature:** Measured definitively as the final distribution constraint in the key generation crashes. The `Sifted_Bit_Bias` radically departs from 0.50.

### 7. Sophisticated Multi-Variate Intrusion (`combined_attack`)
* **Mechanism:** Models highly integrated adversarial systems running multiple parallel attack trajectories to scatter baseline ML profiling. 
* **Statistical Signature:** Blends identical features from PNS depletion algorithms natively alongside brute-force Trojan measurement intensities and IR jitters.
