"""
E20a — Confirm the cone directly.
Protocol: gallery embeddings from e12_feature_cache, L2-normalised.
Convention: weighted scatter decomposition as in compute_e18.py (size-weighted class means).
Predicate: gallery-only statistics; mu computed over gallery only (not query+gallery).
Seed: RNG=42 for pair sampling.

Outputs all six measurements requested in rev 120 (with rev 121 promotions):
  1. Decomposition convention stated; tr(S_B)+tr(S_W)=tr(S_T)=1-||mu||^2 verified numerically.
  2. ||mu|| and tr(S_T) measured directly.
  3. Angular spread: percentiles 10/25/50/75/90 of arccos(x.mu_hat); PROMOTED: std(x.mu_hat).
  4. Mean inter-sample cosine vs ||mu||^2 (DEMOTED: one-line identity check).
  5. cos(mu_hat_A, mu_hat_B) and cos(mu_hat_D, mu_hat_C) — model property vs data.
  6. cos(mu_hat, v1) per cell — does cone coincide with leading uncentered eigenvector?
"""
import numpy as np, json
from pathlib import Path

CACHE_DIR = Path("/tmp/e12_feature_cache")
RESULTS_DIR = Path("/data/papers/low_dimensional_reid/results")
N_Q_MKT = 3368
N_Q_MSMT = 11655
RNG = np.random.default_rng(42)
N_PAIR_SAMPLE = 50000  # for mean inter-sample cosine

def load_gallery(fname, n_query):
    d = np.load(CACHE_DIR / fname)
    g_emb = d["emb"][n_query:].astype(np.float64)
    g_pids = d["pids"][n_query:]
    g_cids = d["cids"][n_query:]
    return g_emb, g_pids, g_cids

def analyze_cell(label, g_emb, g_pids, g_cids, n_sample_pairs=N_PAIR_SAMPLE, rng=RNG):
    print(f"\n=== Cell {label}: N={len(g_emb)}, D={g_emb.shape[1]} ===", flush=True)
    N, D = g_emb.shape

    # 1. Global mean and ||mu||
    mu = g_emb.mean(0)  # (D,)
    mu_norm = np.linalg.norm(mu)
    mu_hat = mu / mu_norm
    mu_sq = float(mu_norm**2)
    print(f"  ||mu|| = {mu_norm:.6f}  ||mu||^2 = {mu_sq:.6f}", flush=True)

    # tr(S_T) directly: (1/N) sum ||x_i - mu||^2
    diff = g_emb - mu  # (N, D)
    tr_ST_direct = float(np.mean(np.sum(diff**2, axis=1)))
    tr_ST_formula = 1.0 - mu_sq  # for unit-norm features
    print(f"  tr(S_T) direct = {tr_ST_direct:.6f}", flush=True)
    print(f"  1 - ||mu||^2   = {tr_ST_formula:.6f}", flush=True)
    print(f"  difference     = {abs(tr_ST_direct - tr_ST_formula):.2e}", flush=True)

    # Check unit norm assumption
    norms = np.linalg.norm(g_emb, axis=1)
    print(f"  feature norms: mean={norms.mean():.6f} std={norms.std():.6f}", flush=True)

    # tr(S_B) and tr(S_W) re-derived (matching compute_e18.py)
    unique_ids = np.unique(g_pids)
    trSB, trSW = 0.0, 0.0
    for uid in unique_ids:
        mask = g_pids == uid
        X_id = g_emb[mask]
        n_i = len(X_id)
        mu_i = X_id.mean(0)
        trSB += (n_i / N) * float(np.sum((mu_i - mu)**2))
        diff_w = X_id - mu_i
        id_scatter = float(np.mean(np.sum(diff_w**2, axis=1)))
        trSW += (n_i / N) * id_scatter

    tr_sum = trSB + trSW
    print(f"  tr(S_B)+tr(S_W) = {tr_sum:.6f} vs tr(S_T)={tr_ST_direct:.6f}  diff={abs(tr_sum-tr_ST_direct):.2e}", flush=True)

    # 3. Angular spread: arccos(x.mu_hat) — angle from mean direction
    proj = (g_emb @ mu_hat).astype(np.float64)  # cos(angle from mu_hat), shape (N,)
    # Clip for arccos safety
    proj_clipped = np.clip(proj / norms, -1.0, 1.0)
    angles_deg = np.degrees(np.arccos(proj_clipped))  # in degrees
    pctiles = np.percentile(angles_deg, [10, 25, 50, 75, 90])
    proj_std = float(proj.std())  # PROMOTED: std of projection x.mu_hat
    print(f"  angular spread (arccos(x.mu_hat)) percentiles:", flush=True)
    for p, v in zip([10,25,50,75,90], pctiles):
        print(f"    p{p:02d} = {v:.2f}°", flush=True)
    print(f"  std(x.mu_hat) = {proj_std:.6f}  [PROMOTED: spread of mu.delta_i]", flush=True)

    # 4. Mean inter-sample cosine vs ||mu||^2 (identity check, DEMOTED)
    idx_i = rng.integers(0, N, size=n_sample_pairs)
    idx_j = rng.integers(0, N, size=n_sample_pairs)
    same = idx_i == idx_j
    idx_j[same] = (idx_j[same] + 1) % N
    cos_pairs = float(np.mean(np.sum(g_emb[idx_i] * g_emb[idx_j], axis=1)))
    print(f"  mean inter-sample cosine = {cos_pairs:.6f} vs ||mu||^2 = {mu_sq:.6f}  diff={abs(cos_pairs-mu_sq):.4f}", flush=True)

    # 6. cos(mu_hat, v1) — alignment of mean with leading uncentered eigenvector
    # v1 = leading right singular vector of g_emb (= leading eigenvector of (1/N) X^T X)
    # We only need the top-1, so use power iteration or just full SVD on a sample
    # For efficiency: leading vector of X.T @ X via Lanczos approximation using top-1 SVD
    print(f"  Computing top-1 uncentered SVD...", flush=True)
    # Use randomized SVD (power iteration): much faster than full SVD for top-1
    # Simple implementation: X.T @ X is D×D = 2048×2048, manageable
    # But N can be 82132 x 2048: X^T X = g_emb.T @ g_emb, shape (2048, 2048)
    gram = g_emb.T @ g_emb  # (D, D)
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    v1 = eigenvectors[:, -1]  # leading eigenvector (largest eigenvalue)
    # Ensure consistent sign
    if np.dot(v1, mu_hat) < 0:
        v1 = -v1
    cos_mu_v1 = float(np.dot(mu_hat, v1))
    print(f"  cos(mu_hat, v1) = {cos_mu_v1:.6f}", flush=True)

    return {
        "N": int(N),
        "D": int(D),
        "mu_norm": float(mu_norm),
        "mu_sq": mu_sq,
        "tr_ST_direct": tr_ST_direct,
        "tr_ST_formula_1_minus_musq": tr_ST_formula,
        "tr_ST_direct_minus_formula": float(tr_ST_direct - tr_ST_formula),
        "tr_SB": trSB,
        "tr_SW": trSW,
        "tr_SB_plus_SW": tr_sum,
        "tr_SB_plus_SW_minus_tr_ST": float(tr_sum - tr_ST_direct),
        "angular_spread_deg": {
            "p10": float(pctiles[0]), "p25": float(pctiles[1]),
            "p50": float(pctiles[2]), "p75": float(pctiles[3]),
            "p90": float(pctiles[4])
        },
        "std_x_dot_mu_hat": proj_std,
        "mean_inter_sample_cosine": cos_pairs,
        "mu_sq_vs_mean_cosine_diff": float(abs(cos_pairs - mu_sq)),
        "cos_mu_hat_v1": cos_mu_v1,
        "mu_hat": mu_hat.tolist(),  # store for cross-model cosine
    }


print("Loading galleries...", flush=True)
A_emb, A_pids, A_cids = load_gallery("market_test.npz",          N_Q_MKT)
B_emb, B_pids, B_cids = load_gallery("msmt17_test.npz",          N_Q_MSMT)
C_emb, C_pids, C_cids = load_gallery("bot_msmt_seed42_market.npz", N_Q_MKT)
D_emb, D_pids, D_cids = load_gallery("bot_msmt_seed42_msmt.npz",  N_Q_MSMT)

results = {}
for label, emb, pids, cids in [("A", A_emb, A_pids, A_cids),
                                 ("B", B_emb, B_pids, B_cids),
                                 ("C", C_emb, C_pids, C_cids),
                                 ("D", D_emb, D_pids, D_cids)]:
    results[label] = analyze_cell(label, emb, pids, cids)

# 5. Cross-model mean direction cosines
mu_hat_A = np.array(results["A"]["mu_hat"])
mu_hat_B = np.array(results["B"]["mu_hat"])
mu_hat_C = np.array(results["C"]["mu_hat"])
mu_hat_D = np.array(results["D"]["mu_hat"])

cos_A_B = float(np.dot(mu_hat_A, mu_hat_B))  # BoT-Market: in-domain vs cross-domain mean
cos_D_C = float(np.dot(mu_hat_D, mu_hat_C))  # BoT-MSMT17: in-domain vs cross-domain mean
cos_A_C = float(np.dot(mu_hat_A, mu_hat_C))  # Market gallery: BoT-Market vs BoT-MSMT17
cos_D_B = float(np.dot(mu_hat_D, mu_hat_B))  # MSMT gallery: BoT-MSMT17 vs BoT-Market
cos_A_D = float(np.dot(mu_hat_A, mu_hat_D))  # cross-model, different data
cos_B_C = float(np.dot(mu_hat_B, mu_hat_C))  # both cross-domain cells

print(f"\n=== CROSS-MODEL MEAN DIRECTIONS ===", flush=True)
print(f"  cos(mu_hat_A, mu_hat_B) = {cos_A_B:.6f}  [BoT-Market: Market vs MSMT17]", flush=True)
print(f"  cos(mu_hat_D, mu_hat_C) = {cos_D_C:.6f}  [BoT-MSMT17: MSMT17 vs Market]", flush=True)
print(f"  cos(mu_hat_A, mu_hat_C) = {cos_A_C:.6f}  [Market gallery: BoT-Mkt vs BoT-MSMT]", flush=True)
print(f"  cos(mu_hat_D, mu_hat_B) = {cos_D_B:.6f}  [MSMT gallery: BoT-MSMT vs BoT-Mkt]", flush=True)
print(f"  cos(mu_hat_B, mu_hat_C) = {cos_B_C:.6f}  [cross-domain cells: both on unfamiliar data]", flush=True)

cross_model = {
    "cos_mu_hat_A_B": cos_A_B,
    "cos_mu_hat_D_C": cos_D_C,
    "cos_mu_hat_A_C": cos_A_C,
    "cos_mu_hat_D_B": cos_D_B,
    "cos_mu_hat_B_C": cos_B_C,
    "note": "A/B = BoT-Market on Market vs MSMT17; D/C = BoT-MSMT17 on MSMT17 vs Market"
}

# Strip mu_hat vectors from per-cell output (large, not needed in JSON)
for cell in results:
    del results[cell]["mu_hat"]

out = {
    "protocol": "gallery-only; mu computed over gallery; L2-normalised features assumed (verified below); weighted scatter decomposition as in compute_e18.py (tr(S_B)=sum_i(n_i/N)||mu_i-mu||^2, tr(S_W)=(1/N)sum_ij||x_ij-mu_i||^2)",
    "cache": "e12_feature_cache: market_test.npz (A), msmt17_test.npz (B), bot_msmt_seed42_market.npz (C), bot_msmt_seed42_msmt.npz (D)",
    "seed": "RNG=42",
    "cells": results,
    "cross_model_mean_directions": cross_model
}

out_path = RESULTS_DIR / "e20a_cone_direct.json"
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)
print(f"\nSaved → {out_path}", flush=True)
