#!/usr/bin/env python3
"""
E14 (fixed) — Oracle ceiling with proper k/alpha selection on half-A.

Fix: select best k (removal) and best alpha (reweighting) on half-A,
evaluate only on half-B. Bootstrap CIs on best arms.
Both directions: Market->MSMT17 and MSMT17->Market.
"""

import json
from pathlib import Path
import numpy as np

CACHE_DIR = Path("/tmp/e12_feature_cache")
OUT_PATH  = Path("/data/papers/low_dimensional_reid/results/e14_fixed_results.json")
N_DIRS    = 128
RNG_SEED  = 42
N_BOOT    = 500


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
    return np.array(aps)   # return per-query APs for bootstrapping


def map_from_aps(aps): return float(np.mean(aps)) if len(aps) else 0.0


def bootstrap_ci(aps, n=N_BOOT, rng=None):
    if rng is None: rng = np.random.default_rng(42)
    means = [np.mean(rng.choice(aps, len(aps), replace=True)) for _ in range(n)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def project_out(emb, dirs):
    return emb - emb @ dirs @ dirs.T

def reweight(emb, U, w):
    c = emb @ U; return (emb - c @ U.T) + (c * w) @ U.T

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


def run_direction(label, src_cache_pattern, tgt_cache, seeds, rng):
    print(f"\n{'='*60}\n{label}\n{'='*60}")

    # Load target features
    if isinstance(tgt_cache, list):
        embs = [np.load(CACHE_DIR / f.format(s))["emb"].astype(np.float32) for s, f in zip(seeds, tgt_cache)]
        d0   = np.load(CACHE_DIR / tgt_cache[0].format(seeds[0]))
        X_tgt = l2norm(np.mean(embs, axis=0))
        pids_tgt, cids_tgt = d0["pids"], d0["cids"]
    else:
        d = np.load(CACHE_DIR / tgt_cache)
        X_tgt = d["emb"].astype(np.float32)
        pids_tgt, cids_tgt = d["pids"], d["cids"]

    # Source PCA (use seed 42)
    d_src = np.load(CACHE_DIR / src_cache_pattern.format(seeds[0]))
    Xsc = d_src["emb"].astype(np.float32); pids_src = d_src["pids"]
    Xsc -= Xsc.mean(0)
    _, s, Vt = np.linalg.svd(Xsc / np.sqrt(len(Xsc)-1), full_matrices=False)
    U_src = Vt[:N_DIRS].T
    lam_src = s[:N_DIRS]**2

    # Leakage split
    unique_ids = np.unique(pids_tgt[pids_tgt > 0])
    rng.shuffle(unique_ids)
    half = len(unique_ids) // 2
    ids_A, ids_B = set(unique_ids[:half]), set(unique_ids[half:])
    mask_A = np.array([p in ids_A for p in pids_tgt])
    mask_B = np.array([p in ids_B for p in pids_tgt])
    X_A, pids_A, cids_A = X_tgt[mask_A], pids_tgt[mask_A], cids_tgt[mask_A]
    X_B, pids_B, cids_B = X_tgt[mask_B], pids_tgt[mask_B], cids_tgt[mask_B]
    print(f"Half-A: {len(ids_A)} ids  Half-B: {len(ids_B)} ids")

    # Classify Harmful on half-A
    sb_A, sw_A = scatter_diag(X_A, pids_A)
    f_tgt_A = np.array([(U_src[:,i]**2 @ sb_A)/(U_src[:,i]**2 @ sw_A + 1e-10) for i in range(N_DIRS)])
    Xac = X_A - X_A.mean(0)
    w_A = np.array([float(np.mean((Xac @ U_src[:,i])**2)) for i in range(N_DIRS)])
    w_A /= w_A.sum() + 1e-10
    is_harmful = (w_A >= np.median(w_A)) & (f_tgt_A < np.median(f_tgt_A))
    harmful_sorted = np.where(is_harmful)[0][np.argsort(-w_A[is_harmful])]
    print(f"Harmful dirs: {is_harmful.sum()}")

    # Helper: eval on a given half
    MAX_Q = 2000
    def eval_half(X_half, pids_half, cids_half, transform_fn):
        q_idx = rng.choice(len(X_half), min(MAX_Q, len(X_half)), replace=False)
        Xq = transform_fn(X_half[q_idx].astype(np.float64)).astype(np.float32)
        Xg = transform_fn(X_half.astype(np.float64)).astype(np.float32)
        return compute_map(Xq, pids_half[q_idx], cids_half[q_idx], Xg, pids_half, cids_half)

    identity = lambda x: l2norm(x.astype(np.float32))

    # Baseline on B
    aps_base_B = eval_half(X_B, pids_B, cids_B, identity)
    map_base_B = map_from_aps(aps_base_B)
    ci_base = bootstrap_ci(aps_base_B, rng=rng)
    print(f"Baseline half-B: {map_base_B:.4f}  95% CI [{ci_base[0]:.4f}, {ci_base[1]:.4f}]")

    # --- SELECT k ON HALF-A ---
    alphas = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
    ks     = [k for k in [1, 3, 5, 10, 15, 20, is_harmful.sum()] if k <= is_harmful.sum()]

    print("\nSelecting k (removal) on half-A...")
    map_A_per_k = {}
    for k in ks:
        dirs_rm = U_src[:, harmful_sorted[:k]]
        aps_A = eval_half(X_A, pids_A, cids_A,
                          lambda x, d=dirs_rm: l2norm(project_out(x, d).astype(np.float32)))
        map_A_per_k[k] = map_from_aps(aps_A)
        print(f"  k={k:2d}: half-A mAP={map_A_per_k[k]:.4f}")
    best_k = max(map_A_per_k, key=map_A_per_k.get)
    print(f"  → selected k={best_k}")

    print("\nSelecting alpha (reweighting) on half-A...")
    map_A_per_alpha = {}
    for alpha in alphas:
        w = f_tgt_A**alpha; w /= w.mean() + 1e-10
        aps_A = eval_half(X_A, pids_A, cids_A,
                          lambda x, ww=w: l2norm(reweight(x, U_src, ww).astype(np.float32)))
        map_A_per_alpha[alpha] = map_from_aps(aps_A)
        print(f"  alpha={alpha:.2f}: half-A mAP={map_A_per_alpha[alpha]:.4f}")
    best_alpha = max(map_A_per_alpha, key=map_A_per_alpha.get)
    print(f"  → selected alpha={best_alpha}")

    # --- EVALUATE SELECTED ARMS ON HALF-B ---
    dirs_rm_best = U_src[:, harmful_sorted[:best_k]]
    aps_rm_B = eval_half(X_B, pids_B, cids_B,
                         lambda x, d=dirs_rm_best: l2norm(project_out(x, d).astype(np.float32)))
    map_rm_B = map_from_aps(aps_rm_B)
    ci_rm = bootstrap_ci(aps_rm_B, rng=rng)

    w_best = f_tgt_A**best_alpha; w_best /= w_best.mean() + 1e-10
    aps_rw_B = eval_half(X_B, pids_B, cids_B,
                         lambda x, ww=w_best: l2norm(reweight(x, U_src, ww).astype(np.float32)))
    map_rw_B = map_from_aps(aps_rw_B)
    ci_rw = bootstrap_ci(aps_rw_B, rng=rng)

    # --- RANDOM REMOVAL CONTROL (matched k) on half-B ---
    rand_maps = [map_from_aps(eval_half(X_B, pids_B, cids_B,
                     lambda x, idx=rng.choice(N_DIRS, best_k, replace=False):
                         l2norm(project_out(x, U_src[:, idx]).astype(np.float32))))
                 for _ in range(20)]
    map_rand_mean = float(np.mean(rand_maps))
    map_rand_std  = float(np.std(rand_maps))

    print(f"\n=== {label} ===")
    print(f"  Baseline half-B:       {map_base_B:.4f}  CI [{ci_base[0]:.4f}, {ci_base[1]:.4f}]")
    print(f"  Removal (k={best_k}, half-B):  {map_rm_B:.4f}  CI [{ci_rm[0]:.4f}, {ci_rm[1]:.4f}]")
    print(f"  Reweight (a={best_alpha}, half-B): {map_rw_B:.4f}  CI [{ci_rw[0]:.4f}, {ci_rw[1]:.4f}]")
    print(f"  Random removal k={best_k}:    {map_rand_mean:.4f}  ±{map_rand_std:.4f}")

    return {
        "direction": label,
        "baseline": {"map": map_base_B, "ci": list(ci_base)},
        "removal":  {"k": best_k, "map": map_rm_B, "ci": list(ci_rm)},
        "reweight": {"alpha": best_alpha, "map": map_rw_B, "ci": list(ci_rw)},
        "random_removal": {"k": best_k, "map_mean": map_rand_mean, "map_std": map_rand_std},
        "ceiling_above_noise": {
            "removal":  ci_rm[0] > ci_base[1],
            "reweight": ci_rw[0] > ci_base[1],
        },
    }


def main():
    rng = np.random.default_rng(RNG_SEED)

    results = []

    # Direction 1: Market -> MSMT17 (BoT-Market)
    results.append(run_direction(
        label="Market->MSMT17",
        src_cache_pattern="market_test.npz",   # source features
        tgt_cache="msmt17_test.npz",           # target features
        seeds=[42],
        rng=rng,
    ))

    # Direction 2: MSMT17 -> Market (BoT-MSMT17, avg seeds)
    results.append(run_direction(
        label="MSMT17->Market",
        src_cache_pattern="bot_msmt_seed{}_msmt.npz",
        tgt_cache=["bot_msmt_seed{}_market.npz".format(s) for s in [42, 123]],
        seeds=[42, 123],
        rng=rng,
    ))

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {OUT_PATH}")


if __name__ == "__main__":
    main()
