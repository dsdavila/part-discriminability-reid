#!/usr/bin/env python3
"""
E14 (MSMT17→Market direction): oracle direction reweighting/removal ceiling.

Source: BoT-MSMT17 (seeds 42 + 123), evaluated on Market test set.
Cross-domain mAP: ~27% — well above floor, meaningful signal.

Same protocol as E14 (Market->MSMT17):
  - Split Market IDs in half (leakage guard)
  - Classify Harmful on half-A (using seed-averaged f_tgt)
  - Evaluate mAP on half-B: removal sweep, reweighting sweep
  - Controls: PCA whitening, permuted weights, random subspace removal
"""

import json
from pathlib import Path

import numpy as np
from reid_tk.datasets import get_dataset

CACHE_DIR = Path("/tmp/e12_feature_cache")
OUT_PATH  = Path("/data/papers/low_dimensional_reid/results/e14_msmt_market_results.json")
SEEDS     = [42, 123]
N_DIRS    = 128
RNG_SEED  = 42


# ---------------------------------------------------------------------------
# Helpers (copied from compute_e14.py)
# ---------------------------------------------------------------------------

def compute_map(q_emb, q_pids, q_cids, g_emb, g_pids, g_cids):
    sims = q_emb @ g_emb.T
    aps = []
    for i in range(len(q_pids)):
        good = (g_pids == q_pids[i]) & (g_cids != q_cids[i])
        junk = ((g_pids == q_pids[i]) & (g_cids == q_cids[i])) | (g_pids <= 0)
        order = np.argsort(-sims[i])
        good_s = good[order][~junk[order]]
        if good_s.sum() == 0: continue
        hits = np.where(good_s)[0] + 1
        aps.append((np.arange(1, good_s.sum()+1) / hits).mean())
    return float(np.mean(aps)) if aps else 0.0


def project_out(emb, directions):
    return emb - emb @ directions @ directions.T


def reweight(emb, U, weights):
    coords = emb @ U
    residual = emb - coords @ U.T
    return residual + (coords * weights) @ U.T


def l2norm(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-10)


def scatter_diag(X, labels):
    mu = X.mean(0); N = len(X)
    sb = np.zeros(X.shape[1], dtype=np.float64)
    sw = np.zeros(X.shape[1], dtype=np.float64)
    for u in np.unique(labels):
        Xc = X[labels == u]; mu_c = Xc.mean(0); n = len(Xc)
        sb += n/N * (mu_c - mu)**2
        sw += np.mean((Xc - mu_c)**2, axis=0) * n/N
    return sb, sw


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    rng = np.random.default_rng(RNG_SEED)

    # ---- Load Market target features (averaged across seeds) ---------------
    print("Loading Market features (avg across seeds)...")
    embs, pids_ref, cids_ref = [], None, None
    for seed in SEEDS:
        d = np.load(CACHE_DIR / f"bot_msmt_seed{seed}_market.npz")
        embs.append(d["emb"].astype(np.float32))
        if pids_ref is None:
            pids_ref = d["pids"]
            cids_ref = d["cids"]
    X_tgt = np.mean(embs, axis=0)
    X_tgt = l2norm(X_tgt)   # re-normalise after averaging
    pids_tgt, cids_tgt = pids_ref, cids_ref
    print(f"  Market target: {X_tgt.shape}, {len(np.unique(pids_tgt[pids_tgt>0]))} ids")

    # ---- Source PCA (from MSMT17 features, seed-averaged) ------------------
    print("Computing source PCA (seed-averaged)...")
    U_seeds, lam_seeds = [], []
    for seed in SEEDS:
        d = np.load(CACHE_DIR / f"bot_msmt_seed{seed}_msmt.npz")
        Xsc = d["emb"].astype(np.float32); Xsc -= Xsc.mean(0)
        _, s, Vt = np.linalg.svd(Xsc / np.sqrt(len(Xsc)-1), full_matrices=False)
        U_seeds.append(Vt[:N_DIRS].T)
        lam_seeds.append(s[:N_DIRS]**2)
    # Use seed 42 eigenvectors as canonical basis (close enough; seed variance is tiny)
    U_src = U_seeds[0]
    lam_src = np.mean(lam_seeds, axis=0)

    # ---- Leakage split on Market IDs ---------------------------------------
    unique_ids = np.unique(pids_tgt[pids_tgt > 0])
    rng.shuffle(unique_ids)
    half = len(unique_ids) // 2
    ids_A, ids_B = set(unique_ids[:half]), set(unique_ids[half:])

    mask_A = np.array([p in ids_A for p in pids_tgt])
    mask_B = np.array([p in ids_B for p in pids_tgt])
    X_A, pids_A, cids_A = X_tgt[mask_A], pids_tgt[mask_A], cids_tgt[mask_A]
    X_B, pids_B, cids_B = X_tgt[mask_B], pids_tgt[mask_B], cids_tgt[mask_B]
    print(f"  Half-A: {X_A.shape}, {len(ids_A)} ids  |  Half-B: {X_B.shape}, {len(ids_B)} ids")

    MAX_QUERIES = 2000
    q_idx = rng.choice(len(X_B), min(MAX_QUERIES, len(X_B)), replace=False)
    X_q, pids_q, cids_q = X_B[q_idx], pids_B[q_idx], cids_B[q_idx]
    X_g, pids_g, cids_g = X_B, pids_B, cids_B

    # ---- Estimate f_tgt on half-A ------------------------------------------
    print("Estimating f_tgt on half-A...")
    sb_A, sw_A = scatter_diag(X_A, pids_A)
    f_tgt_A = np.array([(U_src[:, i]**2 @ sb_A) / (U_src[:, i]**2 @ sw_A + 1e-10) for i in range(N_DIRS)])
    Xac = X_A - X_A.mean(0)
    w_A = np.array([float(np.mean((Xac @ U_src[:, i])**2)) for i in range(N_DIRS)])
    w_A /= w_A.sum() + 1e-10

    is_harmful = (w_A >= np.median(w_A)) & (f_tgt_A < np.median(f_tgt_A))
    harmful_dirs = np.where(is_harmful)[0]
    print(f"  Harmful directions: {is_harmful.sum()}")

    # ---- Baseline ----------------------------------------------------------
    print("\nBaseline mAP...")
    map_baseline = compute_map(X_q, pids_q, cids_q, X_g, pids_g, cids_g)
    print(f"  Baseline: {map_baseline:.4f}")

    # ---- Controls ----------------------------------------------------------
    white_w = 1.0 / (np.sqrt(lam_src) + 1e-10); white_w /= white_w.mean()

    def eval_rw(weights):
        Xq_r = l2norm(reweight(X_q.astype(np.float64), U_src, weights).astype(np.float32))
        Xg_r = l2norm(reweight(X_g.astype(np.float64), U_src, weights).astype(np.float32))
        return compute_map(Xq_r, pids_q, cids_q, Xg_r, pids_g, cids_g)

    map_white = eval_rw(white_w)
    print(f"  PCA whitening: {map_white:.4f}")

    map_perm = float(np.mean([
        eval_rw(f_tgt_A[rng.permutation(N_DIRS)] / (f_tgt_A.mean() + 1e-10))
        for _ in range(5)
    ]))
    print(f"  Permuted weights: {map_perm:.4f}")

    # ---- Reweighting sweep -------------------------------------------------
    print("\nReweighting sweep...")
    reweight_results = []
    for alpha in [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]:
        w = f_tgt_A**alpha; w /= w.mean() + 1e-10
        m = eval_rw(w)
        reweight_results.append({"alpha": alpha, "map": m})
        print(f"  alpha={alpha:.2f}: {m:.4f}")

    # ---- Removal sweep -----------------------------------------------------
    print("\nRemoval sweep...")
    harmful_sorted = harmful_dirs[np.argsort(-w_A[harmful_dirs])]
    removal_results = []
    for k in [1, 3, 5, 10, 15, 20, len(harmful_dirs)]:
        if k > len(harmful_sorted): break
        dirs_rm = U_src[:, harmful_sorted[:k]]
        Xq_r = l2norm(project_out(X_q.astype(np.float64), dirs_rm).astype(np.float32))
        Xg_r = l2norm(project_out(X_g.astype(np.float64), dirs_rm).astype(np.float32))
        m = compute_map(Xq_r, pids_q, cids_q, Xg_r, pids_g, cids_g)
        removal_results.append({"k_removed": k, "map": m})
        print(f"  Remove k={k:2d}: {m:.4f}")

    # ---- Random removal controls -------------------------------------------
    print("\nRandom removal control...")
    rand_results = []
    for k in [5, 10, 20]:
        maps = [
            compute_map(
                l2norm(project_out(X_q.astype(np.float64), U_src[:, rng.choice(N_DIRS, k, replace=False)]).astype(np.float32)),
                pids_q, cids_q,
                l2norm(project_out(X_g.astype(np.float64), U_src[:, rng.choice(N_DIRS, k, replace=False)]).astype(np.float32)),
                pids_g, cids_g
            ) for _ in range(10)
        ]
        rand_results.append({"k_removed": k, "map_mean": float(np.mean(maps)), "map_std": float(np.std(maps))})
        print(f"  Random k={k}: {np.mean(maps):.4f} ± {np.std(maps):.4f}")

    best_rw = max(reweight_results, key=lambda x: x["map"])
    best_rm = max(removal_results,  key=lambda x: x["map"])

    print(f"\n=== E14 MSMT→Market Summary ===")
    print(f"  Baseline:          {map_baseline:.4f}")
    print(f"  PCA whitening:     {map_white:.4f}")
    print(f"  Permuted weights:  {map_perm:.4f}")
    print(f"  Best reweighting:  {best_rw['map']:.4f}  (alpha={best_rw['alpha']})")
    print(f"  Best removal:      {best_rm['map']:.4f}  (k={best_rm['k_removed']})")

    results = {
        "direction": "MSMT17->Market",
        "baseline_map": map_baseline,
        "control_whitening_map": map_white,
        "control_permuted_map": map_perm,
        "n_harmful_dirs": int(is_harmful.sum()),
        "reweighting_sweep": reweight_results,
        "removal_sweep": removal_results,
        "random_removal_control": rand_results,
        "best_reweight": best_rw,
        "best_removal": best_rm,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {OUT_PATH}")


if __name__ == "__main__":
    main()
