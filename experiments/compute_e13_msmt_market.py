#!/usr/bin/env python3
"""
E13 (MSMT17→Market direction): per-direction transfer analysis.

Source: BoT-MSMT17 (seeds 42 + 123), MSMT17 test features
Target: Market test features (same models)

Averages classification across seeds for robustness.
Same metrics as E13 (Market→MSMT17):
  w_i    = u_i' Cov_tgt u_i           (target metric weight)
  f_tgt  = target Fisher ratio         (Market identity separability)
  f_src  = source Fisher ratio         (MSMT17 discriminability)
  lam_i  = source eigenvalue

Four classes (median splits): Shared / Harmful / Latent / Dormant
Bias test: Harmful-class rate vs source discriminability rank.
"""

import json
from pathlib import Path

import numpy as np

CACHE_DIR = Path("/tmp/e12_feature_cache")
OUT_PATH  = Path("/data/papers/low_dimensional_reid/results/e13_msmt_market_results.json")
N_DIRS    = 128
SEEDS     = [42, 123]


def scatter_diag(X, labels):
    mu = X.mean(0); N = len(X)
    sb = np.zeros(X.shape[1], dtype=np.float64)
    sw = np.zeros(X.shape[1], dtype=np.float64)
    for u in np.unique(labels):
        Xc = X[labels == u]; mu_c = Xc.mean(0); n = len(Xc)
        sb += n / N * (mu_c - mu) ** 2
        sw += np.mean((Xc - mu_c) ** 2, axis=0) * n / N
    return sb, sw


def run_one_seed(seed):
    print(f"\n  --- seed {seed} ---")
    X_src = np.load(CACHE_DIR / f"bot_msmt_seed{seed}_msmt.npz")["emb"].astype(np.float32)
    d_tgt = np.load(CACHE_DIR / f"bot_msmt_seed{seed}_market.npz")
    X_tgt, pids_tgt = d_tgt["emb"].astype(np.float32), d_tgt["pids"]
    d_src_full = np.load(CACHE_DIR / f"bot_msmt_seed{seed}_msmt.npz")
    pids_src = d_src_full["pids"]

    # Source PCA
    Xsc = X_src - X_src.mean(0)
    _, s, Vt = np.linalg.svd(Xsc / np.sqrt(len(Xsc) - 1), full_matrices=False)
    U = Vt[:N_DIRS].T
    lam = s[:N_DIRS] ** 2

    # Source Fisher ratios
    sb_src, sw_src = scatter_diag(X_src, pids_src)
    f_src = np.array([(U[:, i]**2 @ sb_src) / (U[:, i]**2 @ sw_src + 1e-10) for i in range(N_DIRS)])

    # Target metric weight + Fisher
    Xtc = X_tgt - X_tgt.mean(0)
    cov_tgt_diag = np.array([float(np.mean((Xtc @ U[:, i])**2)) for i in range(N_DIRS)])
    w = cov_tgt_diag / (cov_tgt_diag.sum() + 1e-10)

    sb_tgt, sw_tgt = scatter_diag(X_tgt, pids_tgt)
    f_tgt = np.array([(U[:, i]**2 @ sb_tgt) / (U[:, i]**2 @ sw_tgt + 1e-10) for i in range(N_DIRS)])

    return U, lam, f_src, w, f_tgt


def main():
    print("Loading and processing both seeds...")
    results_per_seed = [run_one_seed(s) for s in SEEDS]

    # Average w and f_tgt across seeds for classification
    w_avg    = np.mean([r[3] for r in results_per_seed], axis=0)
    f_tgt_avg = np.mean([r[4] for r in results_per_seed], axis=0)
    f_src_avg = np.mean([r[2] for r in results_per_seed], axis=0)
    lam_avg   = np.mean([r[1] for r in results_per_seed], axis=0)

    # Classify
    w_med  = np.median(w_avg)
    f_med  = np.median(f_tgt_avg)
    high_w = w_avg >= w_med
    high_f = f_tgt_avg >= f_med
    cls = np.where(high_w & high_f,   "Shared",
          np.where(high_w & ~high_f,  "Harmful",
          np.where(~high_w & high_f,  "Latent",
                                       "Dormant")))

    counts = {c: int((cls == c).sum()) for c in ["Shared","Harmful","Latent","Dormant"]}
    print(f"\nFour-class counts (N={N_DIRS}):")
    for c, n in counts.items():
        print(f"  {c:<8}: {n:3d}  ({100*n/N_DIRS:.0f}%)")

    # Bias test
    rank_order = np.argsort(-f_src_avg)
    is_harmful = cls == "Harmful"
    base_rate  = is_harmful.mean()
    cum_harmful = np.cumsum(is_harmful[rank_order]) / (np.arange(N_DIRS) + 1)

    rng = np.random.default_rng(42)
    boot = rng.choice([0,1], size=(2000, N_DIRS), p=[1-base_rate, base_rate]).mean(axis=1)
    ci_lo, ci_hi = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))

    cutoffs = [10, 20, 50, 100, N_DIRS]
    bias_summary = []
    print(f"\nBias test (base rate={base_rate:.3f}, CI [{ci_lo:.3f},{ci_hi:.3f}]):")
    print(f"  {'Top-k':<8} {'Harmful%':>10} {'Above base?':>12}")
    for k in [c for c in cutoffs if c <= N_DIRS]:
        rate_k = float(cum_harmful[k-1])
        above  = rate_k > ci_hi
        bias_summary.append({"top_k": k, "harmful_rate": rate_k, "base_rate": float(base_rate), "above_base": above})
        print(f"  {k:<8} {rate_k:>10.3f} {'YES' if above else 'no':>12}")

    # Source-weight breakdown
    lam_frac = lam_avg / lam_avg.sum()
    sort_src_w = np.argsort(-lam_frac)
    print(f"\nTop-k by source eigenvalue:")
    for k in [10, 20, 50]:
        sub = cls[sort_src_w[:k]]
        ct  = {c: int((sub==c).sum()) for c in ["Shared","Harmful","Latent","Dormant"]}
        print(f"  Top-{k:3d}: " + "  ".join(f"{c}={ct[c]} ({100*ct[c]/k:.0f}%)" for c in ["Shared","Harmful","Latent","Dormant"]))

    results = {
        "direction": "MSMT17->Market",
        "seeds": SEEDS,
        "n_dirs": N_DIRS,
        "class_counts": counts,
        "base_rate_harmful": float(base_rate),
        "base_rate_ci": [ci_lo, ci_hi],
        "bias_summary": bias_summary,
        "bias_conclusion": ("above_base_rate" if any(b["above_base"] for b in bias_summary[:3]) else "at_or_below_base_rate"),
        "per_direction": [
            {"rank_src": int(i), "lam_frac": float(lam_frac[i]),
             "f_src": float(f_src_avg[i]), "w": float(w_avg[i]),
             "f_tgt": float(f_tgt_avg[i]), "class": str(cls[i])}
            for i in range(N_DIRS)
        ],
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {OUT_PATH}")


if __name__ == "__main__":
    main()
