#!/usr/bin/env python3
"""
E14 full-query — Market→MSMT17 with all half-B query images (no MAX_Q cap).

Identical to compute_e14_paired.py for Market→MSMT17 direction, except:
- MAX_Q removed: all half-B images used as queries (not sampled)
- Sims matrix computed in batches of BATCH_Q to avoid OOM
- Reports full-query baseline mAP and arm improvement

Used to drop the "(n_query=2000)" footnote from Result 5.
"""

import json
from pathlib import Path
import numpy as np

CACHE_DIR = Path("/tmp/e12_feature_cache")
OUT_PATH  = Path("/data/papers/low_dimensional_reid/results/e14_fullquery_results.json")
N_DIRS    = 128
RNG_SEED  = 42
N_BOOT    = 2000
BATCH_Q   = 2000   # batch size for AP computation (controls peak RAM: BATCH_Q × |gallery| × 4B)


def compute_aps_batched(q_emb, q_pids, q_cids, g_emb, g_pids, g_cids, batch=BATCH_Q):
    """Compute per-query AP in batches to avoid OOM. Returns (N_q,) float array."""
    aps = np.full(len(q_pids), np.nan)
    for start in range(0, len(q_pids), batch):
        end = min(start + batch, len(q_pids))
        sims = q_emb[start:end] @ g_emb.T
        for bi, qi in enumerate(range(start, end)):
            qid, qcam = q_pids[qi], q_cids[qi]
            order = np.argsort(-sims[bi])
            junk = ((g_pids == qid) & (g_cids == qcam)) | (g_pids <= 0)
            order = order[~junk[order]]
            good = g_pids[order] == qid
            if not good.any():
                continue
            pos = np.where(good)[0]
            aps[qi] = float((np.arange(1, good.sum()+1) / (pos+1)).mean())
    return aps


def paired_bootstrap_ci(aps_base, aps_arm, n=N_BOOT, rng=None):
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


def main():
    rng = np.random.default_rng(RNG_SEED)

    print("Loading Market source features (for PCA)...")
    d_src = np.load(CACHE_DIR / "market_test.npz")
    Xsc = d_src["emb"].astype(np.float32)
    Xsc -= Xsc.mean(0)
    _, _, Vt = np.linalg.svd(Xsc / np.sqrt(len(Xsc)-1), full_matrices=False)
    U_src = Vt[:N_DIRS].T
    print(f"  Source PCA: U_src shape {U_src.shape}")

    print("Loading MSMT17 target features (combined query+gallery)...")
    d_tgt = np.load(CACHE_DIR / "msmt17_test.npz")
    X_tgt_raw = d_tgt["emb"].astype(np.float32)
    pids_tgt   = d_tgt["pids"]
    cids_tgt   = d_tgt["cids"]
    print(f"  MSMT17 test: {X_tgt_raw.shape[0]} images, "
          f"{len(np.unique(pids_tgt[pids_tgt > 0]))} valid IDs")

    # Half-A / half-B split (same RNG sequence as E14)
    unique_ids = np.unique(pids_tgt[pids_tgt > 0])
    rng.shuffle(unique_ids)
    half = len(unique_ids) // 2
    ids_A, ids_B = set(unique_ids[:half]), set(unique_ids[half:])
    mask_A = np.array([p in ids_A for p in pids_tgt])
    mask_B = np.array([p in ids_B for p in pids_tgt])
    X_A, pids_A, cids_A = X_tgt_raw[mask_A], pids_tgt[mask_A], cids_tgt[mask_A]
    X_B, pids_B, cids_B = X_tgt_raw[mask_B], pids_tgt[mask_B], cids_tgt[mask_B]
    print(f"  Half-A: {len(X_A)} imgs, {len(ids_A)} IDs")
    print(f"  Half-B: {len(X_B)} imgs, {len(ids_B)} IDs")

    # Tune: classify harmful on half-A
    sb_A, sw_A = scatter_diag(X_A, pids_A)
    f_tgt_A = np.array([(U_src[:,i]**2 @ sb_A) / (U_src[:,i]**2 @ sw_A + 1e-10)
                         for i in range(N_DIRS)])
    Xac = X_A - X_A.mean(0)
    w_A = np.array([float(np.mean((Xac @ U_src[:,i])**2)) for i in range(N_DIRS)])
    w_A /= w_A.sum() + 1e-10
    is_harmful = (w_A >= np.median(w_A)) & (f_tgt_A < np.median(f_tgt_A))
    harmful_sorted = np.where(is_harmful)[0][np.argsort(-w_A[is_harmful])]

    # Select best k (MAX_Q=2000 for tuning, same as E14)
    alphas = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
    ks = [k for k in [1, 3, 5, 10, 15, 20, is_harmful.sum()] if k <= is_harmful.sum()]

    def map_halfA(transform_fn):
        q_idx = rng.choice(len(X_A), min(2000, len(X_A)), replace=False)
        Xq = transform_fn(X_A[q_idx].astype(np.float64)).astype(np.float32)
        Xg = transform_fn(X_A.astype(np.float64)).astype(np.float32)
        aps = compute_aps_batched(Xq, pids_A[q_idx], cids_A[q_idx], Xg, pids_A, cids_A)
        return float(np.nanmean(aps))

    identity = lambda x: l2norm(x.astype(np.float32))

    map_A_per_k = {k: map_halfA(lambda x, d=U_src[:,harmful_sorted[:k]]:
                   l2norm(project_out(x, d).astype(np.float32))) for k in ks}
    best_k = max(map_A_per_k, key=map_A_per_k.get)

    map_A_per_alpha = {a: map_halfA(lambda x, ww=f_tgt_A**a/(np.mean(f_tgt_A**a)+1e-10):
                       l2norm(reweight(x, U_src, ww).astype(np.float32))) for a in alphas}
    best_alpha = max(map_A_per_alpha, key=map_A_per_alpha.get)
    print(f"Selected: k={best_k}, alpha={best_alpha}")

    # Full-query evaluation on ALL half-B images (no MAX_Q cap)
    print(f"Evaluating on ALL {len(X_B)} half-B images as queries (batch={BATCH_Q})...")
    Xg_B = X_B.astype(np.float64)

    def eval_transform(transform_fn, label=''):
        Xq = transform_fn(X_B.astype(np.float64)).astype(np.float32)
        Xg = transform_fn(Xg_B).astype(np.float32)
        print(f"  Computing APs ({label}, {len(Xq)} queries × {len(Xg)} gallery)...")
        return compute_aps_batched(Xq, pids_B, cids_B, Xg, pids_B, cids_B)

    aps_base = eval_transform(identity, 'baseline')
    dirs_rm   = U_src[:, harmful_sorted[:best_k]]
    aps_rm    = eval_transform(
        lambda x, d=dirs_rm: l2norm(project_out(x, d).astype(np.float32)), 'removal')
    w_best = f_tgt_A**best_alpha; w_best /= w_best.mean() + 1e-10
    aps_rw = eval_transform(
        lambda x, ww=w_best: l2norm(reweight(x, U_src, ww).astype(np.float32)), 'reweight')

    map_base = float(np.nanmean(aps_base))
    map_rm   = float(np.nanmean(aps_rm))
    map_rw   = float(np.nanmean(aps_rw))

    delta_rm, lo_rm, hi_rm = paired_bootstrap_ci(aps_base, aps_rm, rng=rng)
    delta_rw, lo_rw, hi_rw = paired_bootstrap_ci(aps_base, aps_rw, rng=rng)

    sig_rm = lo_rm > 0 or hi_rm < 0
    sig_rw = lo_rw > 0 or hi_rw < 0

    print(f"\n=== Market->MSMT17 FULL-QUERY ===")
    print(f"  n_queries:  {len(aps_base)} (all half-B images, no MAX_Q cap)")
    print(f"  Baseline:   {map_base*100:.2f}%")
    print(f"  Removal k={best_k}: {map_rm*100:.2f}%  Δ={delta_rm*100:+.3f}pp  "
          f"95%CI [{lo_rm*100:+.3f},{hi_rm*100:+.3f}]pp  sig={sig_rm}")
    print(f"  Reweight a={best_alpha}: {map_rw*100:.2f}%  Δ={delta_rw*100:+.3f}pp  "
          f"95%CI [{lo_rw*100:+.3f},{hi_rw*100:+.3f}]pp  sig={sig_rw}")

    result = {
        "direction": "Market->MSMT17",
        "n_queries": int(len(aps_base)),
        "n_gallery": int(len(X_B)),
        "n_half_B_ids": len(ids_B),
        "note": "Full-query (all half-B images, no MAX_Q cap). Compare to e14_paired_results.json baseline_map=3.26% with n_q=2000.",
        "baseline_map": round(map_base, 6),
        "removal": {"k": int(best_k), "map": round(map_rm, 6), "delta": round(delta_rm, 6),
                    "ci": [round(lo_rm, 6), round(hi_rm, 6)], "significant": bool(sig_rm)},
        "reweight": {"alpha": float(best_alpha), "map": round(map_rw, 6),
                     "delta": round(delta_rw, 6),
                     "ci": [round(lo_rw, 6), round(hi_rw, 6)], "significant": bool(sig_rw)},
        "e14_paired_comparison": {
            "paired_baseline_map": 0.032613,
            "paired_n_queries": 2000,
            "paired_rm_delta": 0.0020,
            "paired_rm_significant": True
        }
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved → {OUT_PATH}")


if __name__ == "__main__":
    main()
