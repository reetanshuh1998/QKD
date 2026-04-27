#!/usr/bin/env python3
"""
Decoy-state BB84 (WCP) session-level dataset generator for QKD attack detection.

Design goals (for publication-quality applied ML + quantum security work):
- Event-based simulation (no circuit simulation) using standard WCP Poisson model.
- Decoy-state observables (gains and QBER by intensity, and basis-conditional stats).
- Attack classes implemented as parameterized perturbations tied to measurable observables.
- Distance (5–50 km metro) is simulated internally via standard fiber-loss model and
  stored for analysis, but should be EXCLUDED from ML input features.

Output: CSV with one row per session, label in {8 classes}.

Recommended run (40,000 rows total):
- 8 classes × 5,000 sessions/class, N=10,000 pulses/session
"""

from __future__ import annotations

import math
import os
import argparse
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm


# -----------------------------
# Utilities
# -----------------------------

def db_to_linear(loss_db: float) -> float:
    """Convert optical loss in dB to linear transmittance."""
    return 10 ** (-loss_db / 10.0)


def fiber_transmittance(alpha_db_per_km: float, distance_km: float) -> float:
    """Standard telecom fiber attenuation model."""
    return db_to_linear(alpha_db_per_km * distance_km)


def shannon_entropy_binary(bits: np.ndarray) -> float:
    """Shannon entropy of a binary array (0/1)."""
    if bits.size == 0:
        return 0.0
    p1 = float(np.mean(bits))
    p0 = 1.0 - p1
    ent = 0.0
    if p0 > 0:
        ent -= p0 * math.log2(p0)
    if p1 > 0:
        ent -= p1 * math.log2(p1)
    return float(ent)


def safe_rate(numer: int, denom: int) -> float:
    return float(numer) / float(denom) if denom > 0 else 0.0


# -----------------------------
# Configuration
# -----------------------------

@dataclass(frozen=True)
class SessionConfig:
    # Session size (pulses)
    N: int = 10_000

    # Metro distance range for internal simulation (not ML feature)
    distance_km_min: float = 5.0
    distance_km_max: float = 50.0
    alpha_db_per_km: float = 0.2  # typical ~0.2 dB/km @1550nm

    # Decoy-state settings (typical starting values; tune if needed)
    mu: float = 0.5   # signal mean photon number
    nu: float = 0.1   # decoy mean photon number
    p_mu: float = 0.80
    p_nu: float = 0.15
    p_0: float = 0.05

    # Device / channel parameters (lumped model)
    eta_det: float = 0.15        # detector efficiency (lumped)
    p_dark: float = 1e-6         # dark count prob per gate
    e_misalign: float = 0.015    # intrinsic misalignment error rate on matched bases

    # Monitor proxy baselines (dimensionless/scaled)
    rx_power_mean: float = 1.0
    rx_power_std: float = 0.03
    timing_mean_us: float = 0.50
    timing_std_us: float = 0.05

    # RNG attack knobs
    # If rng_attack: bias Alice basis toward Z with probability pZ>0.5,
    # and bias intensity selection.
    rng_pZ: float = 0.75
    rng_p_mu: float = 0.90
    rng_p_nu: float = 0.08
    rng_p_0: float = 0.02


ATTACK_TYPES = [
    "normal",
    "mitm_attack",
    "pns_attack",
    "trojan_horse_attack",
    "wavelength_dependent_trojan_attack",
    "rng_attack",
    "detector_blinding_attack",
    "combined_attack",
]


# -----------------------------
# Core session simulation
# -----------------------------

def simulate_session(cfg: SessionConfig, rng: np.random.Generator, label: str) -> Dict[str, float]:
    """
    Simulate one decoy-state BB84 session and return session-level observables.

    Notes:
    - We compute "gains" Q_* as click probability (clicks / sent) for each intensity setting.
    - We compute QBER E_* on the detected AND basis-matched subset for that intensity
      (a common operational statistic; aligns with monitoring + parameter estimation logic).
    """

    # ---- Internal distance / transmittance ----
    distance_km = float(rng.uniform(cfg.distance_km_min, cfg.distance_km_max))
    eta_ch = fiber_transmittance(cfg.alpha_db_per_km, distance_km)
    eta_sys = eta_ch * cfg.eta_det  # lump channel + detector efficiency

    # ---- Alice RNG: bits and bases ----
    alice_bits = rng.integers(0, 2, size=cfg.N, dtype=np.int8)

    # Basis: 0=Z, 1=X
    if label == "rng_attack":
        pZ = cfg.rng_pZ
    else:
        pZ = 0.5
    alice_basis = (rng.random(cfg.N) > pZ).astype(np.int8)  # True->X(1), False->Z(0)

    # ---- Intensity choices: mu, nu, 0 ----
    if label == "rng_attack":
        p_mu, p_nu, p_0 = cfg.rng_p_mu, cfg.rng_p_nu, cfg.rng_p_0
    else:
        p_mu, p_nu, p_0 = cfg.p_mu, cfg.p_nu, cfg.p_0

    which = rng.choice(3, size=cfg.N, p=[p_mu, p_nu, p_0])
    intensity = np.empty(cfg.N, dtype=np.float32)
    intensity[which == 0] = cfg.mu
    intensity[which == 1] = cfg.nu
    intensity[which == 2] = 0.0

    # ---- Photon number (WCP) ----
    n = rng.poisson(intensity).astype(np.int16)

    # ---- Bob basis choice (wavelength-dependent attack bias) ----
    if label == "wavelength_dependent_trojan_attack":
        pZ_bob = 0.75
    else:
        pZ_bob = 0.5
    bob_basis = (rng.random(cfg.N) > pZ_bob).astype(np.int8)

    bases_match = (alice_basis == bob_basis)

    # ---- Baseline click probability model ----
    # p_click_photon = 1 - (1-eta_sys)^n
    p_click_photon = 1.0 - np.power(1.0 - eta_sys, n, dtype=np.float64)
    p_click = 1.0 - (1.0 - p_click_photon) * (1.0 - cfg.p_dark)

    # ---- Attack-driven monitor proxies & knobs ----
    rx_mean = cfg.rx_power_mean
    rx_std = cfg.rx_power_std
    t_mean = cfg.timing_mean_us
    t_std = cfg.timing_std_us

    # Proxy "alarm rate" target (we'll approximate)
    alarm_target = 0.0

    # Trojan-horse: inject extra incoming light + basis leakage (abstract)
    basis_leak_prob = 0.5  # P(Eve basis = Alice basis)
    if label == "trojan_horse_attack":
        rx_mean, rx_std = 2.2, 0.25
        alarm_target = 0.20
        basis_leak_prob = 0.70

    # Detector blinding: high power + click control ability
    blinding_control_prob = 0.0
    if label == "detector_blinding_attack":
        rx_mean, rx_std = 5.0, 0.40
        alarm_target = 0.45
        blinding_control_prob = 0.85
        t_mean, t_std = 0.70, 0.08

    # MITM: timing overhead
    if label == "mitm_attack":
        t_mean += 1.50
        t_std += 0.20

    # Combined: combine PNS + partial intercept-resend + trojan-ish leakage
    if label == "combined_attack":
        t_mean += 0.80
        t_std += 0.15
        rx_mean, rx_std = 1.8, 0.20
        alarm_target = 0.12
        basis_leak_prob = 0.65

    # Wavelength-dependent: basis-dependent efficiency mismatch (proxy)
    if label == "wavelength_dependent_trojan_attack":
        eta_sys_Z = eta_sys * 1.15
        eta_sys_X = max(eta_sys * 0.85, 0.0)
        eta_eff = np.where(bob_basis == 0, eta_sys_Z, eta_sys_X)
        p_click_photon = 1.0 - np.power(1.0 - eta_eff, n, dtype=np.float64)
        p_click = 1.0 - (1.0 - p_click_photon) * (1.0 - cfg.p_dark)

    # PNS: distort gains by selectively blocking singles and boosting multis
    if label in ("pns_attack", "combined_attack"):
        block_single_prob = 0.35 if label == "pns_attack" else 0.25
        boost_multi = 1.20 if label == "pns_attack" else 1.10
        is_single = (n == 1)
        is_multi = (n >= 2)
        p_click = np.where(is_single, p_click * (1.0 - block_single_prob), p_click)
        p_click = np.where(is_multi, np.clip(p_click * boost_multi, 0.0, 1.0), p_click)

    # ---- Double-click model ----
    # Double clicks occur when both detectors fire on the same pulse.
    # Driven by multi-photon events splitting across detectors + independent dark counts.
    # p_double ≈ p_click * p_dark + (for n≥2) contribution from photon splitting.
    p_double_base = p_click * cfg.p_dark + np.where(n >= 2, 0.5 * np.power(eta_sys, 2) * (n * (n - 1)).astype(np.float64) / 2.0, 0.0)
    p_double_base = np.clip(p_double_base, 0.0, 1.0)

    if label == "detector_blinding_attack":
        # Eve controls which detector fires → double clicks suppressed to near zero
        p_double_base *= (1.0 - blinding_control_prob)

    double_clicks = (rng.random(cfg.N) < p_double_base)

    # ---- Realize clicks ----
    clicked = (rng.random(cfg.N) < p_click)

    # ---- Bob bit outcomes (baseline) ----
    # If bases mismatch -> random
    bob_bits = rng.integers(0, 2, size=cfg.N, dtype=np.int8)
    # If bases match -> Alice bit with misalignment flips
    flip = (rng.random(cfg.N) < cfg.e_misalign).astype(np.int8)
    bob_bits = np.where(bases_match, (alice_bits ^ flip), bob_bits)

    # ---- MITM intercept-resend disturbance ----
    if label in ("mitm_attack", "combined_attack"):
        f_ir = 0.60 if label == "mitm_attack" else 0.35
        attacked = (rng.random(cfg.N) < f_ir)

        # Eve basis, with possible leakage (combined/trojan style)
        eve_basis = rng.integers(0, 2, size=cfg.N, dtype=np.int8)
        if basis_leak_prob > 0.5:
            leak_mask = (rng.random(cfg.N) < basis_leak_prob)
            eve_basis = np.where(leak_mask, alice_basis, eve_basis)

        # Eve measurement: correct if basis matches else random
        eve_bit = rng.integers(0, 2, size=cfg.N, dtype=np.int8)
        eve_bit = np.where(eve_basis == alice_basis, alice_bits, eve_bit)

        # Bob receives resent state encoded in eve_basis with eve_bit
        bob_from_eve = rng.integers(0, 2, size=cfg.N, dtype=np.int8)
        flip2 = (rng.random(cfg.N) < cfg.e_misalign).astype(np.int8)
        bob_from_eve = np.where(bob_basis == eve_basis, (eve_bit ^ flip2), bob_from_eve)

        bob_bits = np.where(attacked, bob_from_eve, bob_bits)

    # ---- Detector blinding: partial click/outcome control ----
    if label == "detector_blinding_attack":
        controlled = (rng.random(cfg.N) < blinding_control_prob) & clicked
        # Keep QBER low on matched bases but not "magically perfect" for all
        bob_bits = np.where(controlled & bases_match, alice_bits, bob_bits)

    # ---- Sifting mask ----
    sift_mask = clicked & bases_match
    sifted_alice = alice_bits[sift_mask]
    sifted_bob = bob_bits[sift_mask]
    sifted_basis = alice_basis[sift_mask]  # same as bob basis on sifted
    sifted_intensity = intensity[sift_mask]

    # ---- Gains/QBER per intensity ----
    def gain_qber_for_intensity(val: float) -> Tuple[float, float, int, int]:
        sent = int(np.sum(intensity == val))
        if sent == 0:
            return 0.0, 0.0, 0, 0
        clicks = int(np.sum(clicked & (intensity == val)))
        # error rate on detected+basis-matched subset for that intensity
        det_match = clicked & bases_match & (intensity == val)
        det_match_n = int(np.sum(det_match))
        errors = int(np.sum((alice_bits != bob_bits) & det_match))
        Q = safe_rate(clicks, sent)
        E = safe_rate(errors, det_match_n)
        return Q, E, clicks, det_match_n

    Q_mu, E_mu, clicks_mu, detmatch_mu = gain_qber_for_intensity(cfg.mu)
    Q_nu, E_nu, clicks_nu, detmatch_nu = gain_qber_for_intensity(cfg.nu)
    Q_0, E_0, clicks_0, detmatch_0 = gain_qber_for_intensity(0.0)

    # ---- Basis-conditional on mu (click rates + qber) ----
    def mu_basis_stats(basis: int) -> Tuple[float, float]:
        idx_sent = (intensity == cfg.mu) & (bob_basis == basis)
        sent = int(np.sum(idx_sent))
        if sent == 0:
            return 0.0, 0.0
        clicks = int(np.sum(clicked & idx_sent))
        det_match = clicked & bases_match & idx_sent
        det_match_n = int(np.sum(det_match))
        errors = int(np.sum((alice_bits != bob_bits) & det_match))
        return safe_rate(clicks, sent), safe_rate(errors, det_match_n)

    Q_mu_Z, E_mu_Z = mu_basis_stats(0)
    Q_mu_X, E_mu_X = mu_basis_stats(1)

    # ---- Overall sifted stats ----
    sifted_len = int(sifted_alice.size)
    total_clicks = int(np.sum(clicked))
    base_mismatch_rate = float(np.mean(alice_basis != bob_basis))

    qber_total = float(np.mean(sifted_alice != sifted_bob)) if sifted_len > 0 else 0.0
    entropy_sifted = shannon_entropy_binary(sifted_bob) if sifted_len > 0 else 0.0
    qber_Z = float(np.mean((sifted_alice != sifted_bob)[sifted_basis == 0])) if np.any(sifted_basis == 0) else 0.0
    qber_X = float(np.mean((sifted_alice != sifted_bob)[sifted_basis == 1])) if np.any(sifted_basis == 1) else 0.0

    # ---- Monitor proxies (session-level) ----
    rx_samples = rng.normal(rx_mean, rx_std, size=cfg.N)
    t_samples = rng.normal(t_mean, t_std, size=cfg.N)

    # Approximate alarm rate by thresholding at the (1 - alarm_target) quantile
    if alarm_target > 0.0:
        thr = float(np.quantile(rx_samples, 1.0 - alarm_target))
        alarm_rate = float(np.mean(rx_samples > thr))
    else:
        alarm_rate = 0.0

    # ---- Build feature dict (decoy-state schema) ----
    # IMPORTANT: Distance_km is included for analysis ONLY; drop it from ML features.
    return {
        # Analysis-only (DO NOT feed to ML)
        "Distance_km": distance_km,

        # Session scale / observed totals
        "Pulses_Sent": float(cfg.N),
        "Total_Clicks": float(total_clicks),
        "Sifted_Bits": float(sifted_len),

        # BB84 monitoring stats
        "Base_Mismatch_Rate": float(base_mismatch_rate),
        "QBER_Total": float(qber_total),
        "QBER_Z": float(qber_Z),
        "QBER_X": float(qber_X),
        "Sifted_Bit_Entropy": float(entropy_sifted),

        # Decoy-state gains (click probabilities)
        "Q_mu": float(Q_mu),
        "Q_nu": float(Q_nu),
        "Q_0": float(Q_0),

        # Decoy-state QBER on detected+basis-matched subset for each intensity
        "E_mu": float(E_mu),
        "E_nu": float(E_nu),
        "E_0": float(E_0),

        # Basis-conditional (mu) stats
        "Q_mu_Z": float(Q_mu_Z),
        "Q_mu_X": float(Q_mu_X),
        "E_mu_Z": float(E_mu_Z),
        "E_mu_X": float(E_mu_X),

        # Helpful internal counts as features (still realistic monitoring stats)
        "Clicks_mu": float(clicks_mu),
        "Clicks_nu": float(clicks_nu),
        "Clicks_0": float(clicks_0),
        "DetMatch_mu": float(detmatch_mu),
        "DetMatch_nu": float(detmatch_nu),
        "DetMatch_0": float(detmatch_0),

        # Monitor proxies (implementation-layer observables)
        "Rx_Power_Mean": float(np.mean(rx_samples)),
        "Rx_Power_Std": float(np.std(rx_samples)),
        "Timing_Mean_us": float(np.mean(t_samples)),
        "Timing_Std_us": float(np.std(t_samples)),
        "Double_Click_Rate": float(np.sum(double_clicks)) / float(cfg.N),
        "Monitor_Alarm_Rate": float(alarm_rate),

        "Label": label,
    }

# -----------------------------
# Dataset generation entrypoint
# -----------------------------

def generate_dataset(
    out_path: str = "../data/raw/custom_qkd_dataset.csv",
    sessions_per_class: int = 1000,
    N: int = 10_000,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cfg = SessionConfig(N=N)

    rows = []
    for label in ATTACK_TYPES:
        for _ in tqdm(range(sessions_per_class), desc=f"Simulating {label}"):
            rows.append(simulate_session(cfg, rng, label))

    df = pd.DataFrame(rows)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--iters', type=int, default=1000, help='Iters per class')
    parser.add_argument('--pulses', type=int, default=10_000, help='Pulses per iter')
    args = parser.parse_args()
    
    df = generate_dataset(
        out_path=os.path.join(os.path.dirname(__file__), "../data/raw/custom_qkd_dataset.csv"),
        sessions_per_class=args.iters,
        N=args.pulses,
        seed=42,
    )
    print("Saved:", df.shape)
    print(df["Label"].value_counts())
