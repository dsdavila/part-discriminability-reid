#!/usr/bin/env python3
"""
Cell B top-k bootstrap CIs and k-stability check.

The E17 rank sweep found Cell B improves from 2.56% → 4.03% at k=32.
This script:
  1. Computes per-ID mAP for Cell B under top-k projection for all k in E17 sweep
  2. Runs ID-level bootstrap (N=10000 resamples) to get CIs for the delta vs k=0
  3. Reports the full k-stability curve to see if 4.03% is a plateau or a peak

Uses the normalized cached features from /tmp/e12_feature_cache/.
Top-k projection: project out the top-k eigendirections of the gallery covariance.

Output: /data/papers/low_dimensional_reid/results/cellb_bootstrap_results.json
"""

import json
import sys
from pathlib import Path

import numpy as np

CACHE_DIR = Path("/tmp/e12_feature_cache")
RESULTS_DIR = Path("/data/papers/low_dimensional_reid/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

KS = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256]
N_BOOTSTRAP = 10000
RNG_SEED = 42


def project_out(X, V):
    """Remove subspace spanned by columns of V from rows of X."""
    X64 = X.astype(np.float64)
    return (X64 - X64 @ V @ V.T).astype(np.float32)


def per_id_map(q_emb, q_pids, q_cids, g_emb, g_pids, g_cids, batch=512):
    """Return per-ID mean AP: dict {pid: mean_ap_for_queries_of_that_pid}."""
    n_q = len(q_pids)
    id_aps = {}  # pid -> list of AP values

    for start in range(0, n_q, batch):
        end = min(start + batch, n_q)
        qb = q_emb[start:end].astype(np.float32)
        sim = qb @ g_emb.T
        order = np.argsort(-sim, axis=1)
        for qi, idx in enumerate(order):
            qpid = q_pids[start + qi]; qcid = q_cids[start + qi]
            g_sorted_pids = g_pids[idx]; g_sorted_cids = g_cids[idx]
            mask_valid = ~((g_sorted_pids == qpid) & (g_sorted_cids == qcid))
            mask_pos = (g_sorted_pids == qpid) & mask_valid
            if mask_pos.sum() == 0:
                continue
            valid_idx = np.where(mask_valid)[0]
            pos_in_valid = np.where(mask_pos[valid_idx])[0]
            n_pos = len(pos_in_valid)
            prec = (np.arange(1, n_pos + 1)) / (pos_in_valid + 1)
            ap = float(prec.mean())
            if qpid not in id_aps:
                id_aps[qpid] = []
            id_aps[qpid].append(ap)

    # Mean AP per ID
    return {pid: float(np.mean(aps)) for pid, aps in id_aps.items()}


def id_level_bootstrap(id_aps_k0, id_aps_k, n_resamples=10000, rng_seed=42):
    """
    ID-level bootstrap CI for delta (mean_ap(k) - mean_ap(k0)).
    id_aps: dict {pid -> mean_ap}
    """
    pids = sorted(id_aps_k0.keys())
    v0 = np.array([id_aps_k0[p] for p in pids])
    vk = np.array([id_aps_k.get(p, 0.0) for p in pids])
    delta_obs = float(vk.mean() - v0.mean())

    rng = np.random.default_rng(rng_seed)
    n = len(pids)
    deltas_boot = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        deltas_boot[i] = vk[idx].mean() - v0[idx].mean()

    ci_lo = float(np.percentile(deltas_boot, 2.5))
    ci_hi = float(np.percentile(deltas_boot, 97.5))
    significant = ci_lo > 0 or ci_hi < 0

    return {
        "delta_pp": delta_obs * 100,
        "ci_lo_pp": ci_lo * 100,
        "ci_hi_pp": ci_hi * 100,
        "significant": significant,
        "n_ids": n,
    }


def main():
    rng = np.random.default_rng(RNG_SEED)

    # Load Cell B features: BoT-Market on MSMT17
    print("Loading Cell B features (BoT-Market on MSMT17)...", flush=True)
    c_msmt = np.load(CACHE_DIR / "msmt17_test.npz")
    msmt_emb = c_msmt["emb"]
    msmt_pids = c_msmt["pids"]
    msmt_cids = c_msmt["cids"]

    n_q = 11655
    q_emb = msmt_emb[:n_q].copy()
    q_pids = msmt_pids[:n_q]
    q_cids = msmt_cids[:n_q]
    g_emb = msmt_emb[n_q:].copy()
    g_pids = msmt_pids[n_q:]
    g_cids = msmt_cids[n_q:]

    n_g = len(g_pids)
    D = g_emb.shape[1]
    print(f"  query={n_q}, gallery={n_g}, D={D}", flush=True)
    print(f"  n_unique_ids_query={len(set(q_pids.tolist()))}", flush=True)

    # Eigendecompose gallery covariance (centered, D×D)
    print("Eigendecomposing gallery covariance...", flush=True)
    g_mean = g_emb.mean(axis=0).astype(np.float64)
    g_c = g_emb.astype(np.float64) - g_mean
    cov = g_c.T @ g_c / (n_g - 1)
    eigvals_cov, V = np.linalg.eigh(cov)
    V = V[:, ::-1].copy()  # descending eigenvalue order

    # Compute per-ID mAP for each k
    print("\nComputing per-ID mAP for each k...", flush=True)
    per_id_maps = {}  # k -> {pid: mean_ap}

    for k in KS:
        print(f"  k={k}...", end=" ", flush=True)
        if k == 0:
            q_abl, g_abl = q_emb, g_emb
        else:
            k_clip = min(k, D - 1)
            vecs = V[:, :k_clip]
            q_abl = project_out(q_emb, vecs)
            g_abl = project_out(g_emb, vecs)

        id_aps = per_id_map(q_abl, q_pids, q_cids, g_abl, g_pids, g_cids)
        overall_map = float(np.mean(list(id_aps.values()))) * 100
        per_id_maps[k] = id_aps
        print(f"mAP={overall_map:.3f}% n_ids={len(id_aps)}", flush=True)

    # Bootstrap CIs for each k vs k=0
    print("\nBootstrap CIs (ID-level, N=10000)...", flush=True)
    boot_results = {}
    id_aps_k0 = per_id_maps[0]

    for k in KS:
        if k == 0:
            boot_results[k] = {
                "map_pct": float(np.mean(list(per_id_maps[k].values()))) * 100,
                "delta_pp": 0.0,
                "ci_lo_pp": 0.0,
                "ci_hi_pp": 0.0,
                "significant": False,
                "n_ids": len(per_id_maps[k]),
            }
        else:
            ci = id_level_bootstrap(id_aps_k0, per_id_maps[k], n_resamples=N_BOOTSTRAP, rng_seed=k)
            map_k = float(np.mean(list(per_id_maps[k].values()))) * 100
            boot_results[k] = {
                "map_pct": map_k,
                **ci,
            }
            print(f"  k={k:3d}: mAP={map_k:.3f}%  delta={ci['delta_pp']:+.3f}pp  "
                  f"CI=[{ci['ci_lo_pp']:+.3f}, {ci['ci_hi_pp']:+.3f}]  sig={ci['significant']}", flush=True)

    # Print k-stability summary
    print("\n" + "="*60, flush=True)
    print("K-STABILITY CURVE (Cell B top-k projection out)", flush=True)
    print("="*60, flush=True)
    print(f"{'k':>5} {'mAP%':>8} {'delta':>8} {'CI_lo':>8} {'CI_hi':>8} {'sig':>6}", flush=True)
    for k in KS:
        r = boot_results[k]
        print(f"{k:5d} {r['map_pct']:8.3f} {r['delta_pp']:+8.3f} "
              f"{r['ci_lo_pp']:+8.3f} {r['ci_hi_pp']:+8.3f} {str(r['significant']):>6}", flush=True)

    # Peak analysis
    maps = [boot_results[k]["map_pct"] for k in KS]
    peak_k = KS[int(np.argmax(maps))]
    peak_map = max(maps)
    print(f"\nPeak: k={peak_k}, mAP={peak_map:.3f}%", flush=True)
    print(f"k=0 baseline: {boot_results[0]['map_pct']:.3f}%", flush=True)
    print(f"Peak vs baseline: {peak_map - boot_results[0]['map_pct']:+.3f}pp", flush=True)

    # Check if it's a plateau or peak: count k values within 0.1pp of peak
    near_peak = [k for k in KS if boot_results[k]["map_pct"] >= peak_map - 0.1]
    print(f"k values within 0.1pp of peak: {near_peak}", flush=True)

    results = {
        "description": "Cell B (BoT-Market→MSMT17) top-k projection-out sweep with ID-level bootstrap CIs",
        "n_bootstrap": N_BOOTSTRAP,
        "k_stability_curve": [
            {"k": k, **boot_results[k]} for k in KS
        ],
        "peak_k": peak_k,
        "peak_map_pct": peak_map,
        "baseline_map_pct": boot_results[0]["map_pct"],
        "peak_delta_pp": peak_map - boot_results[0]["map_pct"],
        "near_peak_ks": near_peak,
    }

    out_path = RESULTS_DIR / "cellb_bootstrap_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {out_path}", flush=True)


if __name__ == "__main__":
    main()
