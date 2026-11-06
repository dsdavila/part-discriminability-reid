#!/usr/bin/env python3
"""
E14 ID-level bootstrap — corrects pseudo-replication in image-level CI.

The image-level bootstrap in compute_e14_fullquery.py and
compute_e14_standard_protocol.py treats 46,455 images (or 11,655 images)
as independent draws. They are not: 30+ images per identity are correlated.
The independent unit is the identity. This script resamples IDs.

Runs both evaluations:
  (A) Full half-B: 1,530 IDs  (replaces e14_fullquery_results.json CI fields)
  (B) Standard protocol: 3,059 IDs  (replaces e14_standard_protocol_results.json CI fields)

Method: for each bootstrap resample, draw n_ids IDs with replacement, take all
per-query APs for those IDs, compute mean delta. This gives a CI whose width is
proportional to 1/sqrt(n_ids) rather than 1/sqrt(n_images).

Also adds explicit junk_exclusion fields to standard protocol JSON.
"""

import json
from pathlib import Path
import numpy as np

CACHE_DIR = Path("/tmp/e12_feature_cache")
OUT_HALFB = Path("/data/papers/low_dimensional_reid/results/e14_fullquery_results.json")
OUT_STD   = Path("/data/papers/low_dimensional_reid/results/e14_standard_protocol_results.json")
N_DIRS    = 128
RNG_SEED  = 42
N_BOOT    = 2000
BATCH_Q   = 2000

# Standard protocol counts (verified)
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
            # Junk exclusion: (i) same-ID-same-cam (includes query image itself),
            #                 (ii) invalid pids (g_pids <= 0)
            junk = ((g_pids == qid) & (g_cids == qcam)) | (g_pids <= 0)
            order = order[~junk[order]]
            good = g_pids[order] == qid
            if not good.any():
                continue
            pos = np.where(good)[0]
            aps[qi] = float((np.arange(1, good.sum() + 1) / (pos + 1)).mean())
    return aps


def id_level_bootstrap_ci(aps_base, aps_arm, pids, n_boot=N_BOOT, rng=None):
    """Bootstrap CI by resampling IDs (the independent unit), not images."""
    if rng is None:
        rng = np.random.default_rng(42)
    valid = ~(np.isnan(aps_base) | np.isnan(aps_arm))
    delta_per_img = aps_arm - aps_base
    pids_v = pids[valid]
    delta_v = delta_per_img[valid]

    # Group deltas by identity
    delta_by_id = {}
    for pid, d in zip(pids_v.tolist(), delta_v.tolist()):
        delta_by_id.setdefault(pid, []).append(d)

    unique_ids = np.array(list(delta_by_id.keys()))
    n_ids = len(unique_ids)
    point = float(np.mean(delta_v))

    boot_means = []
    for _ in range(n_boot):
        sampled = unique_ids[rng.integers(0, n_ids, n_ids)]
        deltas = np.concatenate([delta_by_id[id_] for id_ in sampled])
        boot_means.append(float(np.mean(deltas)))

    lo = float(np.percentile(boot_means, 2.5))
    hi = float(np.percentile(boot_means, 97.5))
    sig = lo > 0 or hi < 0
    return point, lo, hi, sig, n_ids


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


def setup_transforms(rng_shared):
    """Load caches, run half-A tuning, return transform components."""
    d_src = np.load(CACHE_DIR / "market_test.npz")
    Xsc = d_src["emb"].astype(np.float32)
    Xsc -= Xsc.mean(0)
    _, _, Vt = np.linalg.svd(Xsc / np.sqrt(len(Xsc) - 1), full_matrices=False)
    U_src = Vt[:N_DIRS].T

    d_tgt = np.load(CACHE_DIR / "msmt17_test.npz")
    X_tgt_raw = d_tgt["emb"].astype(np.float32)
    pids_tgt   = d_tgt["pids"]
    cids_tgt   = d_tgt["cids"]

    # Half-A/B split (same RNG state as E14: must be called FIRST on the shared rng)
    unique_ids = np.unique(pids_tgt[pids_tgt > 0])
    rng_shared.shuffle(unique_ids)
    half = len(unique_ids) // 2
    ids_A, ids_B = set(unique_ids[:half]), set(unique_ids[half:])
    mask_A = np.array([p in ids_A for p in pids_tgt])
    mask_B = np.array([p in ids_B for p in pids_tgt])
    X_A, pids_A, cids_A = X_tgt_raw[mask_A], pids_tgt[mask_A], cids_tgt[mask_A]
    X_B, pids_B, cids_B = X_tgt_raw[mask_B], pids_tgt[mask_B], cids_tgt[mask_B]

    # F-stats on half-A
    sb_A, sw_A = scatter_diag(X_A, pids_A)
    f_tgt_A = np.array([(U_src[:, i]**2 @ sb_A) / (U_src[:, i]**2 @ sw_A + 1e-10)
                         for i in range(N_DIRS)])
    Xac = X_A - X_A.mean(0)
    w_A = np.array([float(np.mean((Xac @ U_src[:, i])**2)) for i in range(N_DIRS)])
    w_A /= w_A.sum() + 1e-10
    is_harmful = (w_A >= np.median(w_A)) & (f_tgt_A < np.median(f_tgt_A))
    harmful_sorted = np.where(is_harmful)[0][np.argsort(-w_A[is_harmful])]

    best_k = 15
    best_alpha = 3.0
    dirs_rm = U_src[:, harmful_sorted[:best_k]]
    w_best = f_tgt_A**best_alpha; w_best /= w_best.mean() + 1e-10

    return (U_src, dirs_rm, w_best, best_k, best_alpha,
            X_B, pids_B, cids_B, X_tgt_raw, pids_tgt, cids_tgt)


def run_eval(X_q, pids_q, cids_q, X_g, pids_g, cids_g, dirs_rm, U_src, w_best,
             best_k, best_alpha, label, n_q_ids):
    identity  = lambda x: l2norm(x.astype(np.float32))
    rm_fn     = lambda x: l2norm(project_out(x, dirs_rm).astype(np.float32))
    rw_fn     = lambda x: l2norm(reweight(x, U_src, w_best).astype(np.float32))

    print(f"  Computing baseline APs ({len(X_q)} queries × {len(X_g)} gallery)...")
    aps_base = compute_aps_batched(identity(X_q.astype(np.float64)), pids_q, cids_q,
                                    identity(X_g.astype(np.float64)), pids_g, cids_g)
    print(f"  Computing removal APs (k={best_k})...")
    aps_rm   = compute_aps_batched(rm_fn(X_q.astype(np.float64)), pids_q, cids_q,
                                    rm_fn(X_g.astype(np.float64)), pids_g, cids_g)
    print(f"  Computing reweight APs (alpha={best_alpha})...")
    aps_rw   = compute_aps_batched(rw_fn(X_q.astype(np.float64)), pids_q, cids_q,
                                    rw_fn(X_g.astype(np.float64)), pids_g, cids_g)

    return aps_base, aps_rm, aps_rw


def main():
    rng = np.random.default_rng(RNG_SEED)

    print("Setting up transforms (source PCA + half-A tuning)...")
    (U_src, dirs_rm, w_best, best_k, best_alpha,
     X_B, pids_B, cids_B, X_tgt_raw, pids_tgt, cids_tgt) = setup_transforms(rng)

    rng_boot = np.random.default_rng(RNG_SEED + 1)  # separate RNG for bootstrap

    # ─── (A) Full half-B ────────────────────────────────────────────────────
    print(f"\n=== (A) Full half-B: {len(X_B)} queries × {len(X_B)} gallery ===")
    aps_b_base, aps_b_rm, aps_b_rw = run_eval(
        X_B, pids_B, cids_B, X_B, pids_B, cids_B,
        dirs_rm, U_src, w_best, best_k, best_alpha,
        "half-B", len(np.unique(pids_B)))

    map_b_base = float(np.nanmean(aps_b_base))
    map_b_rm   = float(np.nanmean(aps_b_rm))
    map_b_rw   = float(np.nanmean(aps_b_rw))

    print(f"  Running ID-level bootstrap (n_ids={len(np.unique(pids_B))})...")
    d_rm_b, lo_rm_b, hi_rm_b, sig_rm_b, n_ids_b = id_level_bootstrap_ci(
        aps_b_base, aps_b_rm, pids_B, rng=rng_boot)
    d_rw_b, lo_rw_b, hi_rw_b, sig_rw_b, _ = id_level_bootstrap_ci(
        aps_b_base, aps_b_rw, pids_B, rng=rng_boot)

    print(f"\n  Baseline: {map_b_base*100:.2f}%")
    print(f"  Removal k={best_k}: Δ={d_rm_b*100:+.3f}pp  "
          f"ID-CI [{lo_rm_b*100:+.3f},{hi_rm_b*100:+.3f}]pp  sig={sig_rm_b}  (n_ids={n_ids_b})")
    print(f"  Reweight a={best_alpha}: Δ={d_rw_b*100:+.3f}pp  "
          f"ID-CI [{lo_rw_b*100:+.3f},{hi_rw_b*100:+.3f}]pp  sig={sig_rw_b}")

    # Update e14_fullquery_results.json with ID-level CIs
    with open(OUT_HALFB) as f:
        out_halfb = json.load(f)
    out_halfb['removal']['ci_id_level'] = [round(lo_rm_b, 6), round(hi_rm_b, 6)]
    out_halfb['removal']['significant_id_level'] = bool(sig_rm_b)
    out_halfb['removal']['n_ids'] = int(n_ids_b)
    out_halfb['removal']['ci_note'] = (
        f"ci is image-level (pseudo-replicated, {len(X_B)} images / {n_ids_b} IDs); "
        "ci_id_level resamples IDs — the independent unit")
    out_halfb['reweight']['ci_id_level'] = [round(lo_rw_b, 6), round(hi_rw_b, 6)]
    out_halfb['reweight']['significant_id_level'] = bool(sig_rw_b)
    out_halfb['reweight']['ci_note'] = out_halfb['removal']['ci_note']
    out_halfb['junk_exclusion'] = {
        "self_excluded": True,
        "same_id_same_cam": "excluded (standard mAP protocol)",
        "invalid_pids": "excluded (g_pids <= 0)",
        "note": ("same-ID-same-cam rule excludes the query image itself since "
                 "it appears in the gallery with the same pid and cam")
    }
    with open(OUT_HALFB, 'w') as f:
        json.dump(out_halfb, f, indent=2)
    print(f"  Updated → {OUT_HALFB}")

    # ─── (B) Standard protocol ──────────────────────────────────────────────
    X_q = X_tgt_raw[:MSMT17_QUERY_COUNT]
    pids_q = pids_tgt[:MSMT17_QUERY_COUNT]
    cids_q = cids_tgt[:MSMT17_QUERY_COUNT]
    X_g = X_tgt_raw[MSMT17_QUERY_COUNT:]
    pids_g = pids_tgt[MSMT17_QUERY_COUNT:]
    cids_g = cids_tgt[MSMT17_QUERY_COUNT:]

    n_std_ids = len(np.unique(pids_q[pids_q > 0]))
    print(f"\n=== (B) Standard protocol: {len(X_q)} queries × {len(X_g)} gallery ===")
    print(f"  Standard query IDs: {n_std_ids}")
    aps_s_base, aps_s_rm, aps_s_rw = run_eval(
        X_q, pids_q, cids_q, X_g, pids_g, cids_g,
        dirs_rm, U_src, w_best, best_k, best_alpha,
        "standard", n_std_ids)

    map_s_base = float(np.nanmean(aps_s_base))
    map_s_rm   = float(np.nanmean(aps_s_rm))
    map_s_rw   = float(np.nanmean(aps_s_rw))

    print(f"  Running ID-level bootstrap (n_ids={n_std_ids})...")
    d_rm_s, lo_rm_s, hi_rm_s, sig_rm_s, n_ids_s = id_level_bootstrap_ci(
        aps_s_base, aps_s_rm, pids_q, rng=rng_boot)
    d_rw_s, lo_rw_s, hi_rw_s, sig_rw_s, _ = id_level_bootstrap_ci(
        aps_s_base, aps_s_rw, pids_q, rng=rng_boot)

    print(f"\n  Baseline: {map_s_base*100:.2f}%")
    print(f"  Removal k={best_k}: Δ={d_rm_s*100:+.3f}pp  "
          f"ID-CI [{lo_rm_s*100:+.3f},{hi_rm_s*100:+.3f}]pp  sig={sig_rm_s}  (n_ids={n_ids_s})")
    print(f"  Reweight a={best_alpha}: Δ={d_rw_s*100:+.3f}pp  "
          f"ID-CI [{lo_rw_s*100:+.3f},{hi_rw_s*100:+.3f}]pp  sig={sig_rw_s}")

    # Update e14_standard_protocol_results.json with ID-level CIs
    with open(OUT_STD) as f:
        out_std = json.load(f)
    out_std['removal']['map'] = round(map_s_rm, 6)
    out_std['removal']['delta'] = round(d_rm_s, 6)
    out_std['removal']['ci_id_level'] = [round(lo_rm_s, 6), round(hi_rm_s, 6)]
    out_std['removal']['significant_id_level'] = bool(sig_rm_s)
    out_std['removal']['n_ids'] = int(n_ids_s)
    out_std['removal']['ci_note'] = (
        f"image-level ci: [{out_std['removal']['ci'][0]:.6f}, {out_std['removal']['ci'][1]:.6f}]; "
        "ci_id_level resamples IDs — the independent unit")
    out_std['reweight']['map'] = round(map_s_rw, 6)
    out_std['reweight']['delta'] = round(d_rw_s, 6)
    out_std['reweight']['ci_id_level'] = [round(lo_rw_s, 6), round(hi_rw_s, 6)]
    out_std['reweight']['significant_id_level'] = bool(sig_rw_s)
    out_std['reweight']['ci_note'] = out_std['removal']['ci_note']
    out_std['junk_exclusion'] = {
        "self_excluded": True,
        "same_id_same_cam": "excluded (standard mAP protocol)",
        "invalid_pids": "excluded (g_pids <= 0)",
        "note": ("standard protocol: query and gallery are disjoint (different subdirs); "
                 "same-ID-same-cam exclusion removes images of query ID from same camera in gallery")
    }
    with open(OUT_STD, 'w') as f:
        json.dump(out_std, f, indent=2)
    print(f"  Updated → {OUT_STD}")

    print("\n=== SUMMARY ===")
    print(f"\nFull half-B (n_images={len(X_B)}, n_ids={n_ids_b}):")
    print(f"  baseline: {map_b_base*100:.2f}%")
    print(f"  removal k={best_k}: +{d_rm_b*100:.3f}pp  ID-CI [{lo_rm_b*100:+.3f},{hi_rm_b*100:+.3f}]  sig={sig_rm_b}")
    print(f"  reweight a={best_alpha}: +{d_rw_b*100:.3f}pp  ID-CI [{lo_rw_b*100:+.3f},{hi_rw_b*100:+.3f}]  sig={sig_rw_b}")

    print(f"\nStandard protocol (n_images={len(X_q)}, n_ids={n_ids_s}):")
    print(f"  baseline: {map_s_base*100:.2f}%")
    print(f"  removal k={best_k}: +{d_rm_s*100:.3f}pp  ID-CI [{lo_rm_s*100:+.3f},{hi_rm_s*100:+.3f}]  sig={sig_rm_s}")
    print(f"  reweight a={best_alpha}: +{d_rw_s*100:.3f}pp  ID-CI [{lo_rw_s*100:+.3f},{hi_rw_s*100:+.3f}]  sig={sig_rw_s}")


if __name__ == "__main__":
    main()
