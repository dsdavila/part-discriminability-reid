#!/usr/bin/env python3
"""
E14 paired bootstrap — correct test for ceiling vs baseline.

Paired test: same query set for both baseline and arm within each resample.
Computes Δ = arm_AP[i] - baseline_AP[i] per query, bootstraps CI on mean(Δ).
Runs all four cells: {Market->MSMT17, MSMT17->Market} × {removal, reweight}.
"""

import json
from pathlib import Path
import numpy as np

CACHE_DIR = Path("/tmp/e12_feature_cache")
OUT_PATH  = Path("/data/papers/low_dimensional_reid/results/e14_paired_results.json")
N_DIRS    = 128
RNG_SEED  = 42
N_BOOT    = 2000
MAX_Q     = 2000


def compute_aps(q_emb, q_pids, q_cids, g_emb, g_pids, g_cids):
    """Return per-query AP array (same length as q_emb)."""
    sims = q_emb @ g_emb.T
    aps = []
    for i in range(len(q_pids)):
        good = (g_pids == q_pids[i]) & (g_cids != q_cids[i])
        junk = ((g_pids == q_pids[i]) & (g_cids == q_cids[i])) | (g_pids <= 0)
        order = np.argsort(-sims[i])
        good_s = good[order][~junk[order]]
        if good_s.sum() == 0:
            aps.append(np.nan)
            continue
        hits = np.where(good_s)[0] + 1
        aps.append(float((np.arange(1, good_s.sum()+1) / hits).mean()))
    return np.array(aps)


def paired_bootstrap_ci(aps_base, aps_arm, n=N_BOOT, rng=None):
    """Bootstrap CI on mean(aps_arm - aps_base), paired by query."""
    if rng is None:
        rng = np.random.default_rng(42)
    valid = ~(np.isnan(aps_base) | np.isnan(aps_arm))
    delta = aps_arm[valid] - aps_base[valid]
    means = [np.mean(rng.choice(delta, len(delta), replace=True)) for _ in range(n)]
    lo, hi = float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))
    return float(np.nanmean(delta)), lo, hi


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

    # Source PCA
    d_src = np.load(CACHE_DIR / src_cache_pattern.format(seeds[0]))
    Xsc = d_src["emb"].astype(np.float32)
    Xsc -= Xsc.mean(0)
    _, s, Vt = np.linalg.svd(Xsc / np.sqrt(len(Xsc)-1), full_matrices=False)
    U_src = Vt[:N_DIRS].T

    # Leakage split (same RNG sequence as E14 fixed)
    unique_ids = np.unique(pids_tgt[pids_tgt > 0])
    rng.shuffle(unique_ids)
    half = len(unique_ids) // 2
    ids_A, ids_B = set(unique_ids[:half]), set(unique_ids[half:])
    mask_A = np.array([p in ids_A for p in pids_tgt])
    mask_B = np.array([p in ids_B for p in pids_tgt])
    X_A, pids_A, cids_A = X_tgt[mask_A], pids_tgt[mask_A], cids_tgt[mask_A]
    X_B, pids_B, cids_B = X_tgt[mask_B], pids_tgt[mask_B], cids_tgt[mask_B]

    # Classify harmful on half-A
    sb_A, sw_A = scatter_diag(X_A, pids_A)
    f_tgt_A = np.array([(U_src[:,i]**2 @ sb_A)/(U_src[:,i]**2 @ sw_A + 1e-10) for i in range(N_DIRS)])
    Xac = X_A - X_A.mean(0)
    w_A = np.array([float(np.mean((Xac @ U_src[:,i])**2)) for i in range(N_DIRS)])
    w_A /= w_A.sum() + 1e-10
    is_harmful = (w_A >= np.median(w_A)) & (f_tgt_A < np.median(f_tgt_A))
    harmful_sorted = np.where(is_harmful)[0][np.argsort(-w_A[is_harmful])]

    # Select best k and alpha on half-A (same logic as E14 fixed)
    alphas = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
    ks     = [k for k in [1, 3, 5, 10, 15, 20, is_harmful.sum()] if k <= is_harmful.sum()]

    def map_halfA(transform_fn):
        q_idx = rng.choice(len(X_A), min(MAX_Q, len(X_A)), replace=False)
        Xq = transform_fn(X_A[q_idx].astype(np.float64)).astype(np.float32)
        Xg = transform_fn(X_A.astype(np.float64)).astype(np.float32)
        aps = compute_aps(Xq, pids_A[q_idx], cids_A[q_idx], Xg, pids_A, cids_A)
        return float(np.nanmean(aps))

    identity = lambda x: l2norm(x.astype(np.float32))

    map_A_per_k = {k: map_halfA(lambda x, d=U_src[:,harmful_sorted[:k]]: l2norm(project_out(x, d).astype(np.float32))) for k in ks}
    best_k = max(map_A_per_k, key=map_A_per_k.get)

    map_A_per_alpha = {a: map_halfA(lambda x, ww=f_tgt_A**a / (np.mean(f_tgt_A**a)+1e-10): l2norm(reweight(x, U_src, ww).astype(np.float32))) for a in alphas}
    best_alpha = max(map_A_per_alpha, key=map_A_per_alpha.get)
    print(f"Selected: k={best_k}, alpha={best_alpha}")

    # Fixed query set for paired evaluation on half-B
    q_idx_B = rng.choice(len(X_B), min(MAX_Q, len(X_B)), replace=False)
    Xq_B = X_B[q_idx_B].astype(np.float64)
    Xg_B = X_B.astype(np.float64)
    pids_q, cids_q = pids_B[q_idx_B], cids_B[q_idx_B]

    def eval_transform(transform_fn):
        Xq = transform_fn(Xq_B).astype(np.float32)
        Xg = transform_fn(Xg_B).astype(np.float32)
        return compute_aps(Xq, pids_q, cids_q, Xg, pids_B, cids_B)

    aps_base = eval_transform(identity)

    dirs_rm = U_src[:, harmful_sorted[:best_k]]
    aps_rm = eval_transform(lambda x, d=dirs_rm: l2norm(project_out(x, d).astype(np.float32)))

    w_best = f_tgt_A**best_alpha; w_best /= w_best.mean() + 1e-10
    aps_rw = eval_transform(lambda x, ww=w_best: l2norm(reweight(x, U_src, ww).astype(np.float32)))

    map_base = float(np.nanmean(aps_base))
    map_rm   = float(np.nanmean(aps_rm))
    map_rw   = float(np.nanmean(aps_rw))

    delta_rm, lo_rm, hi_rm = paired_bootstrap_ci(aps_base, aps_rm, rng=rng)
    delta_rw, lo_rw, hi_rw = paired_bootstrap_ci(aps_base, aps_rw, rng=rng)

    sig_rm = lo_rm > 0 or hi_rm < 0
    sig_rw = lo_rw > 0 or hi_rw < 0

    print(f"\n=== {label} ===")
    print(f"  Baseline mAP:              {map_base:.4f}")
    print(f"  Removal  k={best_k}: mAP={map_rm:.4f}  Δ={delta_rm:+.4f}  95%CI [{lo_rm:+.4f},{hi_rm:+.4f}]  sig={sig_rm}")
    print(f"  Reweight a={best_alpha}: mAP={map_rw:.4f}  Δ={delta_rw:+.4f}  95%CI [{lo_rw:+.4f},{hi_rw:+.4f}]  sig={sig_rw}")

    return {
        "direction": label,
        "baseline_map": map_base,
        "removal": {"k": best_k, "map": map_rm, "delta": delta_rm,
                    "ci": [lo_rm, hi_rm], "significant": sig_rm},
        "reweight": {"alpha": best_alpha, "map": map_rw, "delta": delta_rw,
                     "ci": [lo_rw, hi_rw], "significant": sig_rw},
    }


def main():
    rng = np.random.default_rng(RNG_SEED)
    results = []

    results.append(run_direction(
        label="Market->MSMT17",
        src_cache_pattern="market_test.npz",
        tgt_cache="msmt17_test.npz",
        seeds=[42],
        rng=rng,
    ))

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
