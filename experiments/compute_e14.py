#!/usr/bin/env python3
"""
E14 — Oracle direction reweighting/removal ceiling.

Given E13's per-direction (w_i, f_tgt) classification using target labels,
how much can mAP improve if we reweight or remove Harmful directions?

Two interventions on the source-basis feature representation:
  Removal:    project out top-K Harmful directions, re-normalise
  Reweighting: scale each direction by f_i^alpha (sweep alpha)

Leakage guard: split MSMT17 test identities in half.
  Half-A: classify Harmful (estimate f_tgt)
  Half-B: evaluate mAP

Controls (must beat both):
  1. Equal-weights (alpha=0 / PCA whitening — all coords scaled by lambda_i^-0.5)
  2. Permuted-weights (same weight distribution, randomly shuffled across directions)
  3. Random-subspace removal at matched rank (for removal arm)

Curves throughout; no single operating points.
"""

import json
import random
from pathlib import Path

import numpy as np
from scipy.stats import skew

CACHE_DIR = Path("/tmp/e12_feature_cache")
E13_PATH  = Path("/data/papers/low_dimensional_reid/results/e13_results.json")
OUT_PATH  = Path("/data/papers/low_dimensional_reid/results/e14_results.json")
RNG_SEED  = 42


# ---------------------------------------------------------------------------
# mAP computation (standard ReID: query vs gallery, exclude same-cam same-id)
# ---------------------------------------------------------------------------

def compute_map(q_emb, q_pids, q_cids, g_emb, g_pids, g_cids):
    """Standard ReID mAP. Returns scalar mAP."""
    sims = q_emb @ g_emb.T   # (Nq, Ng)
    aps = []
    for i in range(len(q_pids)):
        sim  = sims[i]
        good = (g_pids == q_pids[i]) & (g_cids != q_cids[i])
        junk = (g_pids == q_pids[i]) & (g_cids == q_cids[i])
        junk |= (g_pids <= 0)
        order = np.argsort(-sim)
        good_sorted = good[order]
        junk_sorted = junk[order]
        # remove junk
        keep = ~junk_sorted
        good_keep = good_sorted[keep]
        if good_keep.sum() == 0:
            continue
        n_good = good_keep.sum()
        hits = np.where(good_keep)[0] + 1   # 1-based positions
        precisions = np.arange(1, n_good + 1) / hits
        aps.append(precisions.mean())
    return float(np.mean(aps)) if aps else 0.0


# ---------------------------------------------------------------------------
# Feature transform helpers
# ---------------------------------------------------------------------------

def project_out(emb, directions):
    """Remove listed directions (columns of U_sub) from emb."""
    if directions.shape[1] == 0:
        return emb
    proj = emb @ directions @ directions.T
    return emb - proj


def reweight(emb, U_src, weights):
    """Scale each source-basis coordinate by weights[i], return in ambient space.
    emb: (N, D), U_src: (D, K), weights: (K,) — one per direction.
    """
    coords = emb @ U_src           # (N, K)
    coords_w = coords * weights    # (N, K)
    residual = emb - coords @ U_src.T
    return residual + coords_w @ U_src.T


def l2_norm(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-10
    return X / norms


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    rng = np.random.default_rng(RNG_SEED)

    # ---- Load features -----------------------------------------------------
    print("Loading features...")
    d_src = np.load(CACHE_DIR / "market_test.npz")
    X_src = d_src["emb"].astype(np.float32)
    pids_src = d_src["pids"]

    d_tgt = np.load(CACHE_DIR / "msmt17_test.npz")
    X_tgt_all = d_tgt["emb"].astype(np.float32)
    pids_tgt_all = d_tgt["pids"]
    cids_tgt_all = d_tgt["cids"]

    print(f"  MSMT17: {X_tgt_all.shape}, {len(np.unique(pids_tgt_all))} ids")

    # ---- Leakage split: split target IDs in half ---------------------------
    unique_ids = np.unique(pids_tgt_all[pids_tgt_all > 0])
    rng.shuffle(unique_ids)
    half = len(unique_ids) // 2
    ids_A = set(unique_ids[:half])   # classify on A
    ids_B = set(unique_ids[half:])   # evaluate on B

    mask_A = np.array([p in ids_A for p in pids_tgt_all])
    mask_B = np.array([p in ids_B for p in pids_tgt_all])

    X_A = X_tgt_all[mask_A]; pids_A = pids_tgt_all[mask_A]; cids_A = cids_tgt_all[mask_A]
    X_B = X_tgt_all[mask_B]; pids_B = pids_tgt_all[mask_B]; cids_B = cids_tgt_all[mask_B]

    print(f"  Half-A (classify): {X_A.shape}, {len(ids_A)} ids")
    print(f"  Half-B (evaluate): {X_B.shape}, {len(ids_B)} ids")

    # Use first 1/3 of B as gallery, rest as query (standard split by cam coverage)
    # Actually split by camera: use all as gallery, sample queries
    # Standard: all B images as gallery+query, exclude same-cam same-id
    # For speed: subsample queries if large
    MAX_QUERIES = 3000
    q_idx = rng.choice(len(X_B), min(MAX_QUERIES, len(X_B)), replace=False)
    X_q = X_B[q_idx]; pids_q = pids_B[q_idx]; cids_q = cids_B[q_idx]
    X_g = X_B; pids_g = pids_B; cids_g = cids_B

    # ---- Source PCA --------------------------------------------------------
    print("Computing source PCA...")
    N_DIRS = 128
    Xsc = X_src - X_src.mean(0)
    _, s_src, Vt_src = np.linalg.svd(Xsc / np.sqrt(len(Xsc) - 1), full_matrices=False)
    U_src = Vt_src[:N_DIRS].T   # (D, N_DIRS)
    lam_src = s_src[:N_DIRS] ** 2

    # ---- Estimate f_tgt on half-A ------------------------------------------
    print("Estimating f_tgt on half-A...")
    Xac = X_A - X_A.mean(0)
    sb_A = np.zeros(X_A.shape[1], dtype=np.float64)
    sw_A = np.zeros(X_A.shape[1], dtype=np.float64)
    N_A = len(X_A)
    mu_A = X_A.mean(0)
    for uid in np.unique(pids_A):
        Xc = X_A[pids_A == uid]
        mu_c = Xc.mean(0)
        sb_A += (pids_A == uid).sum() / N_A * (mu_c - mu_A) ** 2
        sw_A += np.mean((Xc - mu_c) ** 2, axis=0) * (pids_A == uid).sum() / N_A

    f_tgt_A = np.array([
        float((U_src[:, i] ** 2 @ sb_A) / (U_src[:, i] ** 2 @ sw_A + 1e-10))
        for i in range(N_DIRS)
    ])

    w_A = np.array([
        float(np.mean((Xac @ U_src[:, i]) ** 2))
        for i in range(N_DIRS)
    ])
    w_A /= w_A.sum() + 1e-10

    # Classify: Harmful = high w, low f_tgt (median splits)
    w_med = np.median(w_A)
    f_med = np.median(f_tgt_A)
    is_harmful = (w_A >= w_med) & (f_tgt_A < f_med)
    harmful_dirs = np.where(is_harmful)[0]
    print(f"  Harmful directions on half-A: {is_harmful.sum()}")

    # ---- Baseline mAP ------------------------------------------------------
    print("\nComputing baseline mAP (half-B)...")
    map_baseline = compute_map(X_q, pids_q, cids_q, X_g, pids_g, cids_g)
    print(f"  Baseline mAP: {map_baseline:.4f}")

    # ---- Control 1: PCA whitening (equal weights, alpha=0 limit) -----------
    # Scale each direction by 1/sqrt(lambda_i) — makes all coords unit variance
    print("Control 1: PCA whitening (equal source weights)...")
    white_weights = 1.0 / (np.sqrt(lam_src) + 1e-10)
    white_weights /= white_weights.mean()   # normalise scale

    def apply_reweight_eval(weights):
        Xq_r = l2_norm(reweight(X_q.astype(np.float64), U_src, weights).astype(np.float32))
        Xg_r = l2_norm(reweight(X_g.astype(np.float64), U_src, weights).astype(np.float32))
        return compute_map(Xq_r, pids_q, cids_q, Xg_r, pids_g, cids_g)

    map_white = apply_reweight_eval(white_weights)
    print(f"  PCA whitening mAP: {map_white:.4f}")

    # ---- Control 2: permuted weights (shuffle f_tgt_A across directions) ---
    print("Control 2: permuted weights...")
    map_perm_list = []
    for _ in range(5):
        perm = rng.permutation(N_DIRS)
        perm_weights = f_tgt_A[perm]
        perm_weights = perm_weights / (perm_weights.mean() + 1e-10)
        map_perm_list.append(apply_reweight_eval(perm_weights))
    map_perm = float(np.mean(map_perm_list))
    print(f"  Permuted weights mAP: {map_perm:.4f} (mean of 5)")

    # ---- Reweighting: scale by f_tgt_A^alpha, sweep alpha ------------------
    print("\nReweighting sweep (f_i^alpha)...")
    alphas = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
    reweight_results = []
    for alpha in alphas:
        weights = f_tgt_A ** alpha
        weights = weights / (weights.mean() + 1e-10)
        m = apply_reweight_eval(weights)
        reweight_results.append({"alpha": alpha, "map": m})
        print(f"  alpha={alpha:.2f}: mAP={m:.4f}")

    # ---- Removal sweep: project out top-K Harmful, sweep K ----------------
    print("\nRemoval sweep (project out top-K Harmful directions)...")
    # Sort Harmful by w_A (loudest first = most disruptive)
    harmful_sorted = harmful_dirs[np.argsort(-w_A[harmful_dirs])]
    removal_results = []
    for k in [1, 3, 5, 10, 15, 20, 27]:
        if k > len(harmful_sorted):
            break
        dirs_to_remove = U_src[:, harmful_sorted[:k]]   # (D, k)
        Xq_r = l2_norm(project_out(X_q.astype(np.float64), dirs_to_remove).astype(np.float32))
        Xg_r = l2_norm(project_out(X_g.astype(np.float64), dirs_to_remove).astype(np.float32))
        m = compute_map(Xq_r, pids_q, cids_q, Xg_r, pids_g, cids_g)
        removal_results.append({"k_removed": k, "map": m})
        print(f"  Remove top-{k:2d} Harmful: mAP={m:.4f}")

    # Control 3: random subspace removal at matched rank
    print("\nControl 3: random subspace removal (matched rank)...")
    rand_removal_results = []
    for k in [5, 10, 20]:
        maps_rand = []
        for _ in range(10):
            rand_idx = rng.choice(N_DIRS, k, replace=False)
            dirs_rand = U_src[:, rand_idx]
            Xq_r = l2_norm(project_out(X_q.astype(np.float64), dirs_rand).astype(np.float32))
            Xg_r = l2_norm(project_out(X_g.astype(np.float64), dirs_rand).astype(np.float32))
            maps_rand.append(compute_map(Xq_r, pids_q, cids_q, Xg_r, pids_g, cids_g))
        rand_removal_results.append({
            "k_removed": k,
            "map_mean": float(np.mean(maps_rand)),
            "map_std":  float(np.std(maps_rand)),
        })
        print(f"  Random removal k={k:2d}: mAP={np.mean(maps_rand):.4f} ± {np.std(maps_rand):.4f}")

    # ---- Save --------------------------------------------------------------
    results = {
        "baseline_map": map_baseline,
        "control_whitening_map": map_white,
        "control_permuted_map": map_perm,
        "n_harmful_dirs": int(is_harmful.sum()),
        "n_query": len(X_q),
        "n_gallery": len(X_g),
        "reweighting_sweep": reweight_results,
        "removal_sweep": removal_results,
        "random_removal_control": rand_removal_results,
        "best_reweight": max(reweight_results, key=lambda x: x["map"]),
        "best_removal": max(removal_results, key=lambda x: x["map"]),
        "note": (
            "Oracle ceiling: f_tgt estimated on half-A target IDs, evaluated on half-B. "
            "Beats baseline + both controls = real ceiling effect."
        ),
    }

    best_rw = results["best_reweight"]
    best_rm = results["best_removal"]
    print(f"\n=== E14 Summary ===")
    print(f"  Baseline:           {map_baseline:.4f}")
    print(f"  PCA whitening:      {map_white:.4f}  (control 1)")
    print(f"  Permuted weights:   {map_perm:.4f}  (control 2)")
    print(f"  Best reweighting:   {best_rw['map']:.4f}  (alpha={best_rw['alpha']})")
    print(f"  Best removal:       {best_rm['map']:.4f}  (k={best_rm['k_removed']})")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {OUT_PATH}")


if __name__ == "__main__":
    main()
