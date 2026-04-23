#!/usr/bin/env python
# coding: utf-8

# In[1]:


import qiskit


# In[3]:


# Step 1: Import All Necessary Packages
import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error

# Step 2: Core Logic of the BB84 Protocol

def encode_qubit(qc, bit, basis):
    """
    Encodes a classical bit (0 or 1) onto a qubit in the specified basis ('Z' or 'X').
    """
    if bit == 1:
        qc.x(0)
    if basis == 'X':
        qc.h(0)

def measure_qubit(qc, basis):
    """
    Measures a qubit in the specified basis ('Z' or 'X').
    """
    if basis == 'X':
        qc.h(0)
    qc.measure(0, 0)

class BB84_Protocol:
    """
    Implements the full BB84 protocol.
    """
    def __init__(self, key_length, backend=None):
        self.key_length = key_length
        self.alice_bits = np.random.randint(2, size=key_length)
        self.alice_bases = np.random.choice(['Z', 'X'], size=key_length)
        self.bob_bases = np.random.choice(['Z', 'X'], size=key_length)
        self.eve_bases = np.random.choice(['Z', 'X'], size=key_length)
        self.bob_results = []
        self.sifted_key_alice = []
        self.sifted_key_bob = []

        if backend is None:
            self.backend = AerSimulator()
        else:
            self.backend = backend

    def _run_circuit(self, qc, shots=1):
        """Helper function to run a quantum circuit."""
        t_qc = transpile(qc, self.backend)
        result = self.backend.run(t_qc, shots=shots).result()
        counts = result.get_counts(qc)
        return int(list(counts.keys())[0])

    def simulate_exchange(self, eavesdropper_present=False):
        """Simulates the qubit exchange between Alice and Bob."""
        self.bob_results = []
        for i in range(self.key_length):
            qc = QuantumCircuit(1, 1)
            encode_qubit(qc, self.alice_bits[i], self.alice_bases[i])

            if eavesdropper_present:
                eve_measured_bit = self._measure_and_resend(qc, self.eve_bases[i])
                qc_bob = QuantumCircuit(1, 1)
                encode_qubit(qc_bob, eve_measured_bit, self.eve_bases[i])
                measure_qubit(qc_bob, self.bob_bases[i])
                self.bob_results.append(self._run_circuit(qc_bob))
            else:
                measure_qubit(qc, self.bob_bases[i])
                self.bob_results.append(self._run_circuit(qc))

    def _measure_and_resend(self, qc, eve_basis):
        """Simulates Eve's intercept-resend attack."""
        qc_eve = qc.copy()
        measure_qubit(qc_eve, eve_basis)
        return self._run_circuit(qc_eve)

    def sift_keys(self):
        """Creates the sifted key by comparing bases."""
        self.sifted_key_alice = []
        self.sifted_key_bob = []
        for i in range(self.key_length):
            if self.alice_bases[i] == self.bob_bases[i]:
                self.sifted_key_alice.append(self.alice_bits[i])
                self.sifted_key_bob.append(self.bob_results[i])

    def calculate_qber(self):
        """Calculates the Quantum Bit Error Rate (QBER)."""
        if not self.sifted_key_alice or not self.sifted_key_bob:
            return 0.0, [], []

        mismatches = 0
        for i in range(len(self.sifted_key_alice)):
            if self.sifted_key_alice[i] != self.sifted_key_bob[i]:
                mismatches += 1

        qber = mismatches / len(self.sifted_key_alice) if self.sifted_key_alice else 0.0

        return qber, self.sifted_key_alice, self.sifted_key_bob 

# Step 3: Visualization and Runner Functions

def plot_key_comparison(key1, key2, title):
    """Creates a plot to visually compare two keys."""
    if not key1 or not key2 or len(key1) != len(key2):
        print(f"Cannot plot keys for '{title}'. Keys are empty or have different lengths.")
        return

    match = np.array(key1) == np.array(key2)

    fig, ax = plt.subplots(figsize=(10, 1.5))
    ax.imshow([match], cmap='viridis', aspect='auto', interpolation='nearest')
    ax.set_title(title)
    ax.set_yticks([])
    ax.set_xlabel("Sifted Key Bits")
    fig.text(0.5, 0.05, 'Green = Match, Yellow = Mismatch', ha='center', va='bottom', fontsize=9)
    plt.tight_layout(pad=2.0)
    plt.show()

def run_simulation(key_length, eavesdropper, backend, backend_name):
    """Runs a single simulation and prints its results."""
    print("-" * 50)
    print(f"Running Simulation: Key Length={key_length}, Eavesdropper={eavesdropper}, Backend={backend_name}")

    protocol = BB84_Protocol(key_length=key_length, backend=backend)
    protocol.simulate_exchange(eavesdropper_present=eavesdropper)
    protocol.sift_keys()
    qber, final_key_alice, final_key_bob = protocol.calculate_qber()

    print(f"Sifted key length: {len(protocol.sifted_key_alice)}")
    print(f"Quantum Bit Error Rate (QBER): {qber:.2%}")
    if qber > 0.1: # Set a security threshold
        print("High error rate detected! Key has been discarded.")
    else:
        print("Low error rate. The key is considered secure.")
        print(f"Final secure key length: {len(final_key_alice)}")

    plot_key_comparison(final_key_alice, final_key_bob, f"Key Comparison: {backend_name}, Eve: {eavesdropper}")
    print("-" * 50 + "\n")

# Step 4: Run All Scenarios
KEY_LENGTH = 200
ideal_backend = AerSimulator()

print("=============== BASIC SIMULATIONS ===============")
run_simulation(KEY_LENGTH, eavesdropper=False, backend=ideal_backend, backend_name="Ideal_Simulator")
run_simulation(KEY_LENGTH, eavesdropper=True, backend=ideal_backend, backend_name="Ideal_Simulator")

# Step 5: Advanced Analysis - Impact of Noise (EXTRA)

def analyze_noise_impact():
    """
    Analyzes and plots the effect of varying noise levels on QBER.
    """
    print("\n=============== ADVANCED ANALYSIS: NOISE IMPACT ===============")
    noise_levels = np.linspace(0, 0.1, 11) # Noise from 0% to 10%
    qber_results = []

    for p_error in noise_levels:
        noise_model = NoiseModel()
        error = depolarizing_error(p_error, 1)
        noise_model.add_all_qubit_quantum_error(error, ['x', 'h'])
        noisy_backend = AerSimulator(noise_model=noise_model)


        protocol = BB84_Protocol(key_length=KEY_LENGTH, backend=noisy_backend)
        protocol.simulate_exchange(eavesdropper_present=False) # Without Eve
        protocol.sift_keys()
        qber, _, _ = protocol.calculate_qber()
        qber_results.append(qber)
        print(f"Noise Level: {p_error:.1%}, Measured QBER: {qber:.2%}")

    # Plot the results
    plt.figure(figsize=(8, 5))
    plt.plot(noise_levels * 100, qber_results, 'o-', label='Simulated QBER')
    plt.plot(noise_levels * 100, noise_levels * 0.5, '--', label='Theoretical QBER (p/2)', color='red')
    plt.title('Impact of Depolarizing Noise on BB84 QBER')
    plt.xlabel('Gate Error Probability (%)')
    plt.ylabel('Quantum Bit Error Rate (QBER)')
    plt.legend()
    plt.grid(True)
    plt.show()

# Run the advanced analysis
analyze_noise_impact()


