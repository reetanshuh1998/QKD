# QKD Custom Dataset Generation - Technical Documentation

This repository employs a custom, mathematically determinable Monte Carlo Quantum Key Distribution (QKD) simulator to build a robust 8-class machine learning dataset. Since simulating quantum circuits via full density matrices for thousands of iterations is computationally prohibitive, this simulator translates the strict physical laws of quantum state collapse and probability theory into highly optimized, vectorized numerical arrays.

---

## 1. Simulator Working Steps (Instruction Manual)

The simulator uses NumPy to process thousands of identical photon exchanges synchronously. To execute the dataset generator, run:
```bash
python3 src/generate_qkd_dataset.py
```

### Step-by-Step Mechanics:
1. **Initialization (`Length` constraint):** Alice arbitrarily selects a random key length (between 150-400 photons) and generates strict `alice_bits` (0 or 1) and `alice_bases` (Z or X basis). 
2. **Channel Transmission Configuration:** A base depolarizing noise coefficient `p_noise` is established, representing natural optical fiber interference (0% to 5%).
3. **Eve's Interception Phase:** If an active attack is triggered, Eve calculates her own randomized basis arrays and intercepts the transmission. When Eve's basis fails to match Alice's origin basis, Eve forces a destructive "state collapse".
4. **Bob's Measurement Phase:** Bob receives the remaining quantum states. Any measurements Bob conducts in a mismatched basis result in a randomized 50-50 flip outcome.
5. **Optical Noise Injection:** The depolarizing noise profile artificially flips a deterministic percentage of Bob's finalized bits, adhering to pristine physical degradation standards (strictly constrained to 1-3% operational QBER to mimic high-grade commercial optical fiber links). 
6. **Photon Intensity Mapping:** To comprehensively simulate advanced hardware intrusions (such as Avalanche Photodiode blinding or secondary probe pulses), the algorithm uniquely tracks the `Photon_Pulse_Intensity` variance reaching Bob.
7. **Temporal Delay Tracking:** Tracking the `Transmission_Delay_Jitter`, mimicking the physical transmission latency across the channel. This mathematically identifies Time-Shift vulnerabilities where Eavesdropping interception natively forces computation and execution hardware delays (e.g. *Qi et al. 2007*).
8. **Sifted Key Extraction:** The Sifted Key is compiled strictly from indices where Alice and Bob's bases perfectly aligned. The code subsequently extracts physical identifiers: Quantum Bit Error Rate (QBER), Mismatch Rate, Photon Intensity, Delay Jitter, and Shannon Entropy. 

---

## 2. Simulated Quantum Attacks & Theoretical Grounding

The simulator evaluates the baseline natural protocol (`normal`) alongside 7 documented physical attacks by manipulating the mathematical boundaries of the simulation.

### 1. Intercept-Resend (IR) Attack (`mitm_attack`)
* **Mechanism:** Eve intercepts the quantum states traversing the channel, measures them randomly, and generates new photon states to forward to Bob based on her readings. 
* **Statistical Signature:** Eve chooses the incorrect basis 50% of the time, transmitting a collapsed state to Bob. When Bob measures it, he will be incorrect 50% of the time. This deterministically injects exactly a $0.50 \times 0.50 = 25\%$ artificial spike into the QBER.
* *Reference: Gisin, N., Ribordy, G., Tittel, W., & Zbinden, H. (2002). Quantum cryptography. Reviews of Modern Physics, 74(1), 145.*

### 2. Photon-Number Splitting (PNS) Attack (`pns_attack`)
* **Mechanism:** In practical BB84 systems, weak coherent pulses often accidentally emit 2 or 3 photons simultaneously. Eve deterministically steals extra photons from multi-photon pulses while violently blocking any single-photon bursts. 
* **Statistical Signature:** Eve gains complete key knowledge without interacting with Bob's measurements, meaning her attack operates flawlessly at a 0% QBER footprint. However, because Eve arbitrarily blocks single photons, the natural Sifted Key throughput drops severely (simulated mathematically via a 60% attenuation deletion matrix). 
* *Reference: Brassard, G., Lütkenhaus, N., Mor, T., & Sanders, B. C. (2000). Limitations on practical quantum cryptography. Physical Review Letters, 85(6), 1330.*

### 3. Detector Blinding Attack (`detector_blinding_attack`)
* **Mechanism:** Eve exploits hardware vulnerabilities in Bob's avalanche photodiodes (APDs). By injecting constant, highly-intense bright light, Eve forces the detectors out of quantum mapping and into classical linear saturation. 
* **Statistical Signature:** Eve takes total control over Bob's hardware, meaning Bob only registers the exact states Eve transmits. Our model mimics this by forcing `bob_bases = alice_bases` and overriding Bob's state arrays, resulting in a perfectly flawless anomaly: 0% Base Mismatch and 0% QBER. 
* *Reference: Lydersen, L., Wiechers, C., Wittmann, C., Elser, D., Skaar, J., & Makarov, V. (2010). Hacking commercial quantum cryptography systems by tailored bright illumination. Nature Photonics, 4(10), 686-689.*

### 4. Trojan-Horse Attack (`trojan_horse_attack`)
* **Mechanism:** Eve actively probes Alice’s transmission hardware (e.g., phase modulators) by inserting a secondary bright pulse and evaluating the resulting back-reflection to map the hardware's internal alignment.
* **Statistical Signature:** While Trojan attacks evade high QBER detection thresholds, the excess light and back-scatter physically inflate the channel's inherent background noise. We translated this mathematically by injecting violent uniform baseline variances (`+4-10%`) to the measured `Noise_Interference_Level`.
* *Reference: Gisin, N., Fasel, S., Kraus, B., Zbinden, H., & Ribordy, G. (2006). Trojan-horse attacks on quantum-key-distribution systems. Physical Review A, 73(2), 022320.*

### 5. Wavelength-Dependent Trojan Attack (`wavelength_dependent_trojan_attack`)
* **Mechanism:** A highly specific sub-variant where Eve deliberately exploits the wavelength-dependent properties of Bob’s 50/50 beam splitters. By shifting the wavelength, Eve structurally manipulates the routing logic of the photons. 
* **Statistical Signature:** Bob's beam splitter probability matrix structurally crashes. Rather than choosing between X and Z bases equally (50/50), Bob is forced into an imbalanced probabilistic outcome (e.g., 80/20 bias). This skyrockets the `Base_Mismatch_Rate` array in the simulation. 
* *Reference: Li, H. W., Shen, S., et al. (2011). Attacking a practical quantum-key-distribution system with wavelength-dependent beam splitter and multiplexer. Physical Review A, 84(6), 062308.*

### 6. Laser Seeding / RNG Bias Attack (`rng_attack`)
* **Mechanism:** Eve injects phase-locked light directly to seed Alice's laser source or targets Alice’s Random Number Generator (RNG), forcing her "random" base and bit generation arrays into highly predictable, periodic bounds. 
* **Statistical Signature:** The intrinsic probability array for generating binary 1s and 0s structurally collapses from `[0.50, 0.50]` to `[0.85, 0.15]`. As a mathematical consequence, the overall Shannon Sequence Entropy plummets drastically.

### 7. Combined Attack (`combined_attack`)
* **Mechanism:** Evaluates a complex physical adversary executing dual-point manipulations.
* **Statistical Signature:** Merges the severe throughput degradation filters of a successful PNS Attack alongside the destructive state-collapsing algorithms symptomatic of a classical Intercept-Resend attack. 
