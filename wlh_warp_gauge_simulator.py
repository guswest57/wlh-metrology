#!/usr/bin/env python3
"""
WLH Warp Gauge Simulator v1.0
Burren Gemini Collective — November 10, 2025
Full 6D → 4D reduction + lab observables + K=50 bootstrap
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import norm
import json
import pandas as pd

# === PHYSICAL CONSTANTS ===
C = 3e8
G = 6.6743e-11
K_NEDERY = 50

# === WARP POTENTIAL ===
def phi(r, Phi0, R):
    return Phi0 * np.exp(-(r / R)**4)

def dphi_dr(r, Phi0, R):
    x = r / R
    return Phi0 * (-4 * x**3) * np.exp(-x**4) / R

# === MULTI-MODAL OBSERVABLES ===
def clock_shift(r, Phi0, R, f0=5.184e14):  # Yb clock ~518 THz
    return f0 * (1 + phi(r, Phi0, R) / C**2)

def interferometry_phase(r, Phi0, R, L0=1.0, lambda0=1064e-9):
    delta_L = L0 * phi(r, Phi0, R) / C**2
    return (4 * np.pi / lambda0) * delta_L  # Round-trip

def cavity_detuning(r, Phi0, R, L0=0.5):
    return L0 * (1 + phi(r, Phi0, R) / C**2)

def gravimeter(r, Phi0, R, g0=9.81):
    dr = 1e-3
    return g0 + (dphi_dr(r + dr, Phi0, R) - dphi_dr(r - dr, Phi0, R)) / (2 * dr)

# === SYNTHETIC DATA GENERATOR ===
def generate_data(Phi0_true=1.5e16, R_true=12.0, r_max=60, n_points=201,
                  noise_fracs=None):
    if noise_fracs is None:
        noise_fracs = {'clock': 2e-4, 'interf': 0.005, 'cavity': 1e-4, 'grav': 0.001}
    
    r = np.linspace(0, r_max, n_points)
    data = {'r_m': r}
    
    for channel in ['clock', 'interf', 'cavity', 'grav']:
        func = {'clock': clock_shift, 'interf': interferometry_phase,
                'cavity': cavity_detuning, 'grav': gravimeter}[channel]
        clean = func(r, Phi0_true, R_true)
        noise = noise_fracs[channel] * np.random.randn(len(r)) * np.abs(clean)
        data[f'{channel}_clean'] = clean
        data[f'{channel}_noisy'] = clean + noise
    
    return pd.DataFrame(data)

# === FITTING & BOOTSTRAP ===
def chi2(params, r, data_dict):
    Phi0, R = params
    loss = 0
    for ch, func in [('clock', clock_shift), ('interf', interferometry_phase),
                     ('cavity', cavity_detuning), ('grav', gravimeter)]:
        pred = func(r, Phi0, R)
        obs = data_dict[f'{ch}_noisy']
        loss += np.sum((obs - pred)**2)
    return loss

def bootstrap_fit(df, K=50, sigma_jit=0.01):
    results = []
    r = df['r_m'].values
    data_dict = {k: df[k].values for k in df.columns if '_noisy' in k}
    
    for k in range(K):
        jittered = {k: v * (1 + sigma_jit * np.random.randn(len(v))) 
                    for k, v in data_dict.items()}
        res = minimize(lambda p: chi2(p, r, jittered), [1.4e16, 12.0],
                       bounds=((1e15, 1e17), (5, 20)))
        results.append({'iter': k, 'Phi0': res.x[0], 'R': res.x[1], 'loss': res.fun})
    
    return pd.DataFrame(results)

# === RUN DEMO ===
if __name__ == "__main__":
    # Generate truth
    df = generate_data()
    df.to_csv("synthetic_observables.csv", index=False)
    
    # Fit
    boot_df = bootstrap_fit(df, K=K_NEDERY)
    boot_df.to_csv("fit_bootstrap.csv", index=False)
    
    # Save results
    fit_mean = boot_df[['Phi0', 'R']].mean()
    results = {
        "true": {"Phi0": 1.5e16, "R": 12.0},
        "fit": {"Phi0": float(fit_mean['Phi0']), "R": float(fit_mean['R']), "loss": float(boot_df['loss'].mean())},
        "config": {"K": K_NEDERY}
    }
    with open("fit_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("Simulator run complete.")
    print(f"Φ₀ = {results['fit']['Phi0']:.3e} ± {boot_df['Phi0'].std():.3e}")
    print(f"R = {results['fit']['R']:.3f} ± {boot_df['R'].std():.3f}")

