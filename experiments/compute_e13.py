#!/usr/bin/env python3
"""
E13 — Per-direction transfer analysis.

For each source PCA eigenvector u_i (BoT-Market basis), compute:

  w_i  = u_i' Cov_tgt u_i               (target metric weight: how much cosine
                                          metric listens to this direction)
  f_i  = (u_i' S_B_tgt u_i)             (target Fisher ratio: identity
        / (u_i' S_W_tgt u_i)             separability in target)
  f_i_src = source Fisher ratio          (source discriminability)
  lam_i   = source eigenvalue            (source variance / total)

Four-class classification (median splits on w_i and f_i):
  Shared:  high w_i, high f_i  — transferable signal, listened to
  Harmful: high w_i, low  f_i  — distractor, listened to
  Latent:  low  w_i, high f_i  — transferable but ignored
  Dormant: low  w_i, low  f_i  — neither useful nor harmful

Bias test: sort source directions by f_i_src (descending), compute
cumulative Harmful-class rate vs base rate with bootstrap CIs.
Above base rate → model preferentially latches onto distractors.

Uses cached BoT-Market features:
  /tmp/e12_feature_cache/market_test.npz   (source)
  /tmp/e12_feature_cache/msmt17_test.npz   (target)
"""

import json
from pathlib import Path

import numpy as np
from scipy.stats import bootstrap as scipy_bootstrap

CACHE_DIR = Path("/tmp/e12_feature_cache")
OUT_PATH  = Path("/data/papers/low_dimensional_reid/results/e13_results.json")
N_DIRS    = 128   # source PCA directions to analyse


# ---------------------------------------------------------------------------
# Load cached features
# ---------------------------------------------------------------------------

def load(name):
    d = np.load(CACHE_DIR / name)
    return d["emb"].astype(np.float32), d["pids"], d["cids"]


# ---------------------------------------------------------------------------
# Scatter matrices (fast, vectorised)
# ---------------------------------------------------------------------------

def scatter_between_within(X, labels):
    """Compute S_B and S_W (mean-removed), returns diagonal approximations via
    per-direction variance for efficiency with large N."""
    mu = X.mean(0)
    unique, counts = np.unique(labels, return_counts=True)
    N = len(X)
    sb = np.zeros(X.shape[1], dtype=np.float64)
    sw = np.zeros(X.shape[1], dtype=np.float64)
    for u, c in zip(unique, counts):
        Xc = X[labels == u]
        mu_c = Xc.mean(0)
        sb += c / N * (mu_c - mu) ** 2
        sw += np.mean((Xc - mu_c) ** 2, axis=0) * c / N
    return sb, sw   # (D,) — per-feature between/within variance


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading features...")
    X_src, pids_src, _ = load("market_test.npz")
    X_tgt, pids_tgt, _ = load("msmt17_test.npz")
    print(f"  source: {X_src.shape}  ({len(np.unique(pids_src))} ids)")
    print(f"  target: {X_tgt.shape}  ({len(np.unique(pids_tgt))} ids)")

    # ---- Source PCA --------------------------------------------------------
    print(f"\nComputing source PCA ({N_DIRS} directions)...")
    Xsc = X_src - X_src.mean(0)
    # economy SVD — U: (N_src, k), S: (k,), Vt: (k, D)
    U, s, Vt = np.linalg.svd(Xsc / np.sqrt(len(Xsc) - 1), full_matrices=False)
    U_src = Vt[:N_DIRS].T   # (D, N_DIRS) — source eigenvectors as columns
    lam   = s[:N_DIRS] ** 2 # (N_DIRS,) — source eigenvalues (variance)
    lam_frac = lam / lam.sum()  # fractional

    # ---- Source Fisher ratios (per direction) ------------------------------
    print("Computing source scatter (Fisher ratios)...")
    sb_src_full, sw_src_full = scatter_between_within(X_src, pids_src)
    # project scatter diagonals onto eigenvectors
    f_src = np.array([
        float((U_src[:, i] ** 2 @ sb_src_full) /
              (U_src[:, i] ** 2 @ sw_src_full + 1e-10))
        for i in range(N_DIRS)
    ])

    # ---- Target scatter (project features onto source basis) ---------------
    print("Computing target scatter in source basis...")
    Xtc = X_tgt - X_tgt.mean(0)
    Cov_tgt_diag = np.array([
        float(np.mean((Xtc @ U_src[:, i]) ** 2))
        for i in range(N_DIRS)
    ])
    w = Cov_tgt_diag / (Cov_tgt_diag.sum() + 1e-10)   # normalised metric weight

    # Target Fisher ratios
    sb_tgt_full, sw_tgt_full = scatter_between_within(X_tgt, pids_tgt)
    f_tgt = np.array([
        float((U_src[:, i] ** 2 @ sb_tgt_full) /
              (U_src[:, i] ** 2 @ sw_tgt_full + 1e-10))
        for i in range(N_DIRS)
    ])

    # ---- Four-class classification -----------------------------------------
    w_med   = np.median(w)
    f_med   = np.median(f_tgt)
    high_w  = w    >= w_med
    high_f  = f_tgt >= f_med

    cls = np.where(high_w  & high_f,  "Shared",
          np.where(high_w  & ~high_f, "Harmful",
          np.where(~high_w & high_f,  "Latent",
                                      "Dormant")))

    counts = {c: int((cls == c).sum()) for c in ["Shared","Harmful","Latent","Dormant"]}
    print(f"\nFour-class counts (N={N_DIRS}):")
    for c, n in counts.items():
        print(f"  {c:<8}: {n:3d}  ({100*n/N_DIRS:.0f}%)")

    # ---- Bias test ---------------------------------------------------------
    # Sort directions by source discriminability (f_src descending)
    rank_order = np.argsort(-f_src)   # index 0 = most source-discriminative
    is_harmful = (cls == "Harmful")
    base_rate  = is_harmful.mean()

    # Cumulative harmful-class rate as we step through source rank
    cum_harmful = np.cumsum(is_harmful[rank_order]) / (np.arange(N_DIRS) + 1)

    # Bootstrap 95% CI on base rate
    rng = np.random.default_rng(42)
    boot_samples = rng.choice([0, 1], size=(2000, N_DIRS),
                               p=[1 - base_rate, base_rate])
    boot_means = boot_samples.mean(axis=1)
    ci_lo, ci_hi = float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))

    # Summary stats at representative cutoffs
    cutoffs = [10, 20, 50, 100, N_DIRS]
    bias_summary = []
    for k in [c for c in cutoffs if c <= N_DIRS]:
        rate_k = float(cum_harmful[k - 1])
        bias_summary.append({"top_k": k, "harmful_rate": rate_k,
                              "base_rate": float(base_rate),
                              "above_base": rate_k > ci_hi})

    print(f"\nBias test (base rate = {base_rate:.3f}, 95% CI [{ci_lo:.3f}, {ci_hi:.3f}]):")
    print(f"  {'Top-k':<8} {'Harmful rate':>14} {'Above base?':>12}")
    for b in bias_summary:
        flag = "YES" if b["above_base"] else "no"
        print(f"  {b['top_k']:<8} {b['harmful_rate']:>14.3f} {flag:>12}")

    # ---- Save --------------------------------------------------------------
    results = {
        "n_dirs": N_DIRS,
        "w_median": float(w_med),
        "f_tgt_median": float(f_med),
        "class_counts": counts,
        "base_rate_harmful": float(base_rate),
        "base_rate_ci": [ci_lo, ci_hi],
        "bias_summary": bias_summary,
        "bias_conclusion": (
            "above_base_rate" if any(b["above_base"] for b in bias_summary[:3]) else
            "at_or_below_base_rate"
        ),
        "per_direction": [
            {
                "rank_src": int(i),
                "lam_frac": float(lam_frac[i]),
                "f_src": float(f_src[i]),
                "w": float(w[i]),
                "f_tgt": float(f_tgt[i]),
                "class": str(cls[i]),
            }
            for i in range(N_DIRS)
        ],
        "note": (
            "bias_test: sort source dirs by f_src (descending). "
            "Harmful = high w + low f_tgt (median splits). "
            "above_base_rate in top-k = model preferentially encodes distractors."
        ),
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {OUT_PATH}")


if __name__ == "__main__":
    main()
