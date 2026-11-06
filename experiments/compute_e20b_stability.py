"""
E20b stability control (rev 126): cos(mu_hat_half1, mu_hat_half2) for all four cells.
Random item-split into two disjoint halves (RNG=42), per-cell gallery.
Expected ~1.0 if mu_hat is a stable property of model × dataset.
"""
import numpy as np, json
from pathlib import Path

CACHE_DIR = Path("/tmp/e12_feature_cache")
RESULTS_DIR = Path("/data/papers/low_dimensional_reid/results")
N_Q_MKT = 3368
N_Q_MSMT = 11655
RNG = np.random.default_rng(42)

CELLS = {
    "A": ("market_test.npz",            N_Q_MKT),
    "B": ("msmt17_test.npz",            N_Q_MSMT),
    "C": ("bot_msmt_seed42_market.npz", N_Q_MKT),
    "D": ("bot_msmt_seed42_msmt.npz",   N_Q_MSMT),
}

results = {}
for cell, (fname, n_q) in CELLS.items():
    d = np.load(CACHE_DIR / fname)
    g_emb = d["emb"][n_q:].astype(np.float64)
    N = len(g_emb)

    # Random item-split into two halves (no ID awareness — tests item-level stability)
    idx = RNG.permutation(N)
    half1 = g_emb[idx[:N//2]]
    half2 = g_emb[idx[N//2:]]

    mu1 = half1.mean(0); mu1 /= np.linalg.norm(mu1)
    mu2 = half2.mean(0); mu2 /= np.linalg.norm(mu2)
    cos_halves = float(np.dot(mu1, mu2))

    # Also full-gallery mu_hat for reference
    mu_full = g_emb.mean(0); mu_full /= np.linalg.norm(mu_full)
    cos_full_h1 = float(np.dot(mu_full, mu1))
    cos_full_h2 = float(np.dot(mu_full, mu2))

    print(f"Cell {cell}: N={N}  cos(h1,h2)={cos_halves:.6f}  cos(full,h1)={cos_full_h1:.6f}  cos(full,h2)={cos_full_h2:.6f}", flush=True)
    results[cell] = {
        "N": N,
        "cos_half1_half2": cos_halves,
        "cos_full_half1": cos_full_h1,
        "cos_full_half2": cos_full_h2,
    }

out_path = RESULTS_DIR / "e20b_stability_control.json"
with open(out_path, "w") as f:
    json.dump({"protocol": "random item-split gallery halves (RNG=42); cos(mu_hat_half1, mu_hat_half2)", "cells": results}, f, indent=2)
print(f"\nSaved → {out_path}", flush=True)
