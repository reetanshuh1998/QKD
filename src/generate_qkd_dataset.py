import numpy as np
import pandas as pd
import math
import os
from tqdm import tqdm

def calculate_shannon_entropy(array):
    if len(array) == 0: return 0.0
    p1 = np.sum(array) / len(array)
    p0 = 1.0 - p1
    ent = 0.0
    if p0 > 0: ent -= p0 * math.log2(p0)
    if p1 > 0: ent -= p1 * math.log2(p1)
    return ent

def generate_dataset(num_iters_per_class=1000):
    print(f"Generating full 8-class QKD dataset mathematically ({num_iters_per_class} iter/class)...")
    data = []
    np.random.seed(42)
    
    attack_types = ['normal', 'mitm_attack', 'pns_attack', 'trojan_horse_attack', 
                    'wavelength_dependent_trojan_attack', 'rng_attack', 
                    'detector_blinding_attack', 'combined_attack']
                    
    for attack in attack_types:
        for _ in tqdm(range(num_iters_per_class), desc=f"Simulating {attack}"):
            p_noise = np.random.uniform(0.01, 0.03) # Pristine optical fiber baseline
            length = np.random.randint(150, 400)
            
            # Base generation
            alice_p = [0.5, 0.5]
            if attack == 'rng_attack':
                alice_p = [0.85, 0.15] # biased RNG for bits
            alice_bits = np.random.choice([0, 1], p=alice_p, size=length)
            
            alice_bases = np.random.randint(2, size=length) 
            
            bob_base_p = [0.5, 0.5]
            if attack == 'wavelength_dependent_trojan_attack':
                bob_base_p = [0.8, 0.2] # Eve forces Bob's beam splitter bias
            bob_bases = np.random.choice([0, 1], p=bob_base_p, size=length)
            
            eve_bases = np.random.randint(2, size=length)
            
            photon_pulse_intensity = np.random.normal(1.0, 0.05)
            transmission_delay_jitter = np.random.normal(0.5, 0.1) # Standard optical transit delay (microseconds)
            
            if attack == 'detector_blinding_attack':
                # Eve completely forces Bob to read what she wants, matching Alice perfectly
                bob_bases = alice_bases.copy()
                photon_pulse_intensity = np.random.normal(5.0, 0.1)
                transmission_delay_jitter = np.random.normal(1.2, 0.2)
            elif attack == 'trojan_horse_attack':
                photon_pulse_intensity = np.random.normal(2.5, 0.2)
                transmission_delay_jitter = np.random.normal(1.8, 0.3)
            elif attack in ['pns_attack', 'combined_attack']:
                photon_pulse_intensity = np.random.normal(0.6, 0.05)
                
            if attack in ['mitm_attack', 'combined_attack']:
                # The hardware time to intercept, measure, and resend adds massive latency
                transmission_delay_jitter += np.random.normal(3.5, 0.6)
            
            base_mismatch_rate = np.sum(alice_bases != bob_bases) / length
            
            current_bits = alice_bits.copy()
            current_bases = alice_bases.copy()
            
            # 2. Transmission & Interception (MITM or Combined)
            if attack in ['mitm_attack', 'combined_attack']:
                eve_wrong_basis = (eve_bases != alice_bases)
                eve_random_flips = np.random.randint(2, size=length)
                current_bits = np.where(eve_wrong_basis, eve_random_flips, current_bits)
                current_bases = eve_bases
                
            # 3. Bob Measures
            if attack == 'detector_blinding_attack':
                # Eve forces perfect readings bypassing bases
                bob_bits = current_bits.copy()
            else:
                bob_wrong_basis = (bob_bases != current_bases)
                bob_random_flips = np.random.randint(2, size=length)
                bob_bits = np.where(bob_wrong_basis, bob_random_flips, current_bits)
            
            # 4. Channel Depolarizing Noise
            noise_inj = p_noise
            if attack == 'trojan_horse_attack':
                noise_inj += np.random.uniform(0.04, 0.10) # Heavy optical noise interference
            noise_flips = (np.random.rand(length) < (noise_inj / 2)).astype(int)
            bob_bits = (bob_bits + noise_flips) % 2
            
            if attack == 'detector_blinding_attack':
                bob_bits = current_bits.copy() # Zero noise effect due to extremely bright Eve pulse
            
            # 5. Sifting Process
            sifted_indices = (alice_bases == bob_bases)
            sifted_alice = alice_bits[sifted_indices]
            sifted_bob = bob_bits[sifted_indices]
            
            # PNS Attack attenuation (Delete interceptable single photons, ~60% loss)
            if attack in ['pns_attack', 'combined_attack']:
                retention_mask = np.random.rand(len(sifted_alice)) > 0.60
                sifted_alice = sifted_alice[retention_mask]
                sifted_bob = sifted_bob[retention_mask]
            
            # 6. Extraction
            sifted_len = len(sifted_alice)
            qber = np.sum(sifted_alice != sifted_bob) / sifted_len if sifted_len > 0 else 0.0
            entropy = calculate_shannon_entropy(sifted_bob)
            
            data.append({
                'Initial_Key_Length': length,
                'Sifted_Key_Length': sifted_len,
                'Base_Mismatch_Rate': base_mismatch_rate,
                'Noise_Interference_Level': p_noise,
                'Photon_Pulse_Intensity': photon_pulse_intensity,
                'Transmission_Delay_Jitter': transmission_delay_jitter,
                'Measurement_Entropy': entropy,
                'QBER': qber,
                'Label': attack
            })
            
    df = pd.DataFrame(data)
    os.makedirs('../data/raw', exist_ok=True)
    out_path = '../data/raw/custom_qkd_dataset.csv'
    df.to_csv(out_path, index=False)
    print(f"\nDataset compiled at {out_path} with {len(df)} rows.")
    print(df['Label'].value_counts())

if __name__ == '__main__':
    generate_dataset(10000)
