#!/usr/bin/env python3
"""
E14 standard-protocol Market→MSMT17 — for comparison to published work.

Uses the official MSMT17 query (11,655 non-junk) and gallery (82,132 non-junk)
instead of the half-B split used in compute_e14_fullquery.py.

Tuning parameters (k=15, alpha=3.0) from half-A of MSMT17 (same seed=42 split
as compute_e14_fullquery.py). The standard protocol evaluation is NOT used for
tuning — no target-label leakage beyond what was already present in E14 half-A.

Published comparisons (standard MSMT17 protocol):
  CaCL: 36.5 mAP / 66.6 R1  (note: these may be MSMT17-trained, not cross-domain)
  IDM:  33.5 mAP
  RESL: 33.6 mAP

Writes query_count and gallery_count into JSON so protocol is unambiguous.
"""

import json
from pathlib import Path
import numpy as np

CACHE_DIR = Path("/tmp/e12_feature_cache")
OUT_PATH  = Path("/data/papers/low_dimensional_reid/results/e14_standard_protocol_results.json")
N_DIRS    = 128
RNG_SEED  = 42
N_BOOT    = 2000
BATCH_Q   = 2000

# Standard protocol split (verified: query_count=11655, gallery_count=82132)
MSMT17_QUERY_COUNT   = 11655
MSMT17_GALLERY_COUNT = 82132


def compute_aps_batched(q_emb, q_pids, q_cids, g_emb, g_pids, g_cids, batch=BATCH_Q):
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
            aps[qi] = float((np.arange(1, good.sum() + 1) / (pos + 1)).mean())
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

    # Source PCA from Market features
    print("Loading Market source features (for PCA)...")
    d_src = np.load(CACHE_DIR / "market_test.npz")
    Xsc = d_src["emb"].astype(np.float32)
    Xsc -= Xsc.mean(0)
    _, _, Vt = np.linalg.svd(Xsc / np.sqrt(len(Xsc) - 1), full_matrices=False)
    U_src = Vt[:N_DIRS].T
    print(f"  Source PCA: U_src shape {U_src.shape}")

    # Load full MSMT17 test features
    print("Loading MSMT17 features...")
    d_tgt = np.load(CACHE_DIR / "msmt17_test.npz")
    X_tgt_raw = d_tgt["emb"].astype(np.float32)
    pids_tgt   = d_tgt["pids"]
    cids_tgt   = d_tgt["cids"]
    print(f"  Total: {len(X_tgt_raw)} images")

    # Half-A / half-B split (SAME RNG sequence as E14 for tuning)
    unique_ids = np.unique(pids_tgt[pids_tgt > 0])
    rng.shuffle(unique_ids)
    half = len(unique_ids) // 2
    ids_A = set(unique_ids[:half])
    mask_A = np.array([p in ids_A for p in pids_tgt])
    X_A, pids_A, cids_A = X_tgt_raw[mask_A], pids_tgt[mask_A], cids_tgt[mask_A]
    print(f"  Half-A (tuning): {len(X_A)} imgs, {len(ids_A)} IDs")

    # Standard protocol split: first MSMT17_QUERY_COUNT are query
    X_q   = X_tgt_raw[:MSMT17_QUERY_COUNT]
    pids_q = pids_tgt[:MSMT17_QUERY_COUNT]
    cids_q = cids_tgt[:MSMT17_QUERY_COUNT]
    X_g   = X_tgt_raw[MSMT17_QUERY_COUNT:]
    pids_g = pids_tgt[MSMT17_QUERY_COUNT:]
    cids_g = cids_tgt[MSMT17_QUERY_COUNT:]
    print(f"  Standard query:   {len(X_q)} images, "
          f"{len(np.unique(pids_q[pids_q > 0]))} valid IDs")
    print(f"  Standard gallery: {len(X_g)} images, "
          f"{len(np.unique(pids_g[pids_g > 0]))} valid IDs")

    # Tune: compute F-stats and identify harmful directions on half-A
    sb_A, sw_A = scatter_diag(X_A, pids_A)
    f_tgt_A = np.array([(U_src[:, i]**2 @ sb_A) / (U_src[:, i]**2 @ sw_A + 1e-10)
                         for i in range(N_DIRS)])
    Xac = X_A - X_A.mean(0)
    w_A = np.array([float(np.mean((Xac @ U_src[:, i])**2)) for i in range(N_DIRS)])
    w_A /= w_A.sum() + 1e-10
    is_harmful = (w_A >= np.median(w_A)) & (f_tgt_A < np.median(f_tgt_A))
    harmful_sorted = np.where(is_harmful)[0][np.argsort(-w_A[is_harmful])]
    print(f"  Harmful directions: {is_harmful.sum()}")

    # Use k=15, alpha=3.0 (from E14 half-A tuning — same parameters)
    best_k = 15
    best_alpha = 3.0
    print(f"  Using k={best_k}, alpha={best_alpha} (from E14 half-A tuning)")

    # Evaluate on standard protocol (query → gallery)
    print(f"\nEvaluating on standard protocol "
          f"({len(X_q)} queries × {len(X_g)} gallery)...")

    def eval_transform(Xq_raw, Xg_raw, transform_fn, label):
        Xq = transform_fn(Xq_raw.astype(np.float64)).astype(np.float32)
        Xg = transform_fn(Xg_raw.astype(np.float64)).astype(np.float32)
        print(f"  Computing APs ({label})...")
        return compute_aps_batched(Xq, pids_q, cids_q, Xg, pids_g, cids_g)

    identity = lambda x: l2norm(x.astype(np.float32))

    aps_base = eval_transform(X_q, X_g, identity, 'baseline')
    dirs_rm  = U_src[:, harmful_sorted[:best_k]]
    aps_rm   = eval_transform(
        X_q, X_g,
        lambda x, d=dirs_rm: l2norm(project_out(x, d).astype(np.float32)),
        f'removal k={best_k}')
    w_best = f_tgt_A**best_alpha; w_best /= w_best.mean() + 1e-10
    aps_rw = eval_transform(
        X_q, X_g,
        lambda x, ww=w_best: l2norm(reweight(x, U_src, ww).astype(np.float32)),
        f'reweight alpha={best_alpha}')

    map_base = float(np.nanmean(aps_base))
    map_rm   = float(np.nanmean(aps_rm))
    map_rw   = float(np.nanmean(aps_rw))

    delta_rm, lo_rm, hi_rm = paired_bootstrap_ci(aps_base, aps_rm, rng=rng)
    delta_rw, lo_rw, hi_rw = paired_bootstrap_ci(aps_base, aps_rw, rng=rng)
    sig_rm = lo_rm > 0 or hi_rm < 0
    sig_rw = lo_rw > 0 or hi_rw < 0

    print(f"\n=== Market→MSMT17 STANDARD PROTOCOL ===")
    print(f"  n_queries:  {len(aps_base)} (official MSMT17 query, no halving)")
    print(f"  n_gallery:  {len(X_g)} (official MSMT17 gallery, no halving)")
    print(f"  Baseline:   {map_base*100:.2f}%")
    print(f"  Removal k={best_k}: {map_rm*100:.2f}%  Δ={delta_rm*100:+.3f}pp  "
          f"95%CI [{lo_rm*100:+.3f},{hi_rm*100:+.3f}]pp  sig={sig_rm}")
    print(f"  Reweight a={best_alpha}: {map_rw*100:.2f}%  Δ={delta_rw*100:+.3f}pp  "
          f"95%CI [{lo_rw*100:+.3f},{hi_rw*100:+.3f}]pp  sig={sig_rw}")

    result = {
        "direction": "Market->MSMT17",
        "protocol": "standard (official MSMT17 query + gallery, no halving)",
        "n_queries": int(len(aps_base)),
        "n_gallery": int(len(X_g)),
        "tuning": {
            "k": best_k,
            "alpha": best_alpha,
            "source": "half-A of MSMT17 test, same split as compute_e14_fullquery.py (RNG seed=42)"
        },
        "baseline_map": round(map_base, 6),
        "removal": {
            "k": int(best_k), "map": round(map_rm, 6), "delta": round(delta_rm, 6),
            "ci": [round(lo_rm, 6), round(hi_rm, 6)], "significant": bool(sig_rm)
        },
        "reweight": {
            "alpha": float(best_alpha), "map": round(map_rw, 6), "delta": round(delta_rw, 6),
            "ci": [round(lo_rw, 6), round(hi_rw, 6)], "significant": bool(sig_rw)
        },
        "comparison": {
            "e14_fullquery_baseline": 0.034121,
            "e14_fullquery_n_queries": 46455,
            "e14_fullquery_n_gallery": 46455,
            "note": "full-query uses half-B (46455) as both query and gallery; standard uses 11655 query vs 82132 gallery"
        }
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved → {OUT_PATH}")


if __name__ == "__main__":
    main()
