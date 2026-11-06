#!/usr/bin/env python3
"""
E18 — Per-ID subspace geometry.

Computes identity-level geometry measures on the gallery of each 2×2 cell.
Phase 1 (post first): Fisher decomposition + within/between-camera scatter.
Phase 2:              Per-ID overlap ratio vs hub score + correlation.

Three confounds handled explicitly:
  1. 6 cameras (Market) vs 15 (MSMT17) — decompose W_i into B_c + W_c
  2. ~26 images/ID in 2048 dims — NO per-ID effective rank; use scalar summaries
  3. Unseen identity ≠ unseen domain — compare in-domain (A, D) vs cross-domain (B, C) cells

Fisher decomposition (per cell, on gallery):
  tr(S_B) = Σ_i (n_i/N) ||μ_i - μ||²       (between-ID variance)
  tr(S_W) = (1/N) Σ_ij ||x_ij - μ_i||²     (within-ID variance)
  F = tr(S_B) / tr(S_W)                      (Fisher separability ratio)

Camera decomposition (within each ID i):
  B_c(i) = Σ_c (n_ic/n_i) ||μ_ic - μ_i||²  (between-camera scatter for ID i)
  W_c(i) = (1/n_i) Σ_cj ||x_icj - μ_ic||²  (within-camera scatter for ID i)
  Decomposition: tr(S_W) ≈ mean_n(B_c) + mean_n(W_c)

Hub score: N-count (how often gallery images appear in k-NN of other gallery images)

Output: /data/papers/low_dimensional_reid/results/e18_per_id_geometry.json
"""

import json
import sys
from pathlib import Path

import numpy as np

CACHE_DIR = Path("/tmp/e12_feature_cache")
RESULTS_DIR = Path("/data/papers/low_dimensional_reid/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

N_Q_MKT = 3368
N_Q_MSMT = 11655
KNN_K = 10
N_HUB_SAMPLE = 5000  # max queries to use for hub score estimation


def fisher_decomposition(emb, pids, cids, label=""):
    """
    Compute Fisher ratio and camera decomposition on features.

    Returns dict with scalar summaries.
    """
    N, D = emb.shape
    unique_ids = np.unique(pids)
    n_ids = len(unique_ids)
    print(f"  {label}: N={N}, n_ids={n_ids}, D={D}", flush=True)

    global_mean = emb.mean(0)

    # Accumulators
    trSB = 0.0    # between-ID variance
    trSW = 0.0    # total within-ID variance
    trBc = 0.0    # between-camera component of within-ID variance
    trWc = 0.0    # within-camera component of within-ID variance
    n_ids_with_multiimg = 0
    n_ids_with_multicam = 0
    n_ids_with_multimg_per_cam = 0

    per_id_B = []    # per-ID between-camera scatter
    per_id_W = []    # per-ID within-camera scatter
    per_id_total = []  # per-ID total within scatter
    per_id_n = []
    per_id_ncams = []

    for uid in unique_ids:
        mask = pids == uid
        X_id = emb[mask]          # (n_i, D)
        C_id = cids[mask]         # (n_i,) camera IDs for this ID
        n_i = len(X_id)
        mu_i = X_id.mean(0)

        # Between-ID contribution
        trSB += (n_i / N) * float(np.sum((mu_i - global_mean) ** 2))

        # Within-ID total
        diff_total = X_id - mu_i
        id_scatter = float(np.mean(np.sum(diff_total ** 2, axis=1)))  # mean ||x - μ_i||²
        trSW += (n_i / N) * id_scatter

        if n_i > 1:
            n_ids_with_multiimg += 1

        # Camera decomposition
        unique_cams = np.unique(C_id)
        n_cams = len(unique_cams)
        per_id_ncams.append(n_cams)

        B_c_i = 0.0
        W_c_i = 0.0
        has_multiimg_cam = False

        for cam in unique_cams:
            mask_c = C_id == cam
            X_ic = X_id[mask_c]
            n_ic = len(X_ic)
            mu_ic = X_ic.mean(0)
            # Between-camera: distance of camera centroid from ID centroid
            B_c_i += (n_ic / n_i) * float(np.sum((mu_ic - mu_i) ** 2))
            # Within-camera: variance within this camera
            if n_ic > 1:
                diff_c = X_ic - mu_ic
                W_c_i += (n_ic / n_i) * float(np.mean(np.sum(diff_c ** 2, axis=1)))
                has_multiimg_cam = True

        if n_cams > 1:
            n_ids_with_multicam += 1
        if has_multiimg_cam:
            n_ids_with_multimg_per_cam += 1

        trBc += (n_i / N) * B_c_i
        trWc += (n_i / N) * W_c_i

        per_id_B.append(B_c_i)
        per_id_W.append(W_c_i)
        per_id_total.append(id_scatter)
        per_id_n.append(n_i)

    fisher_ratio = float(trSB / max(trSW, 1e-10))
    cam_decomp_check = trBc + trWc  # should ≈ trSW (exactly if all cameras have ≥1 image)

    per_id_n = np.array(per_id_n)
    per_id_ncams = np.array(per_id_ncams)
    per_id_B = np.array(per_id_B)
    per_id_W = np.array(per_id_W)

    print(f"    tr(S_B)={trSB:.4f}  tr(S_W)={trSW:.4f}  F={fisher_ratio:.4f}", flush=True)
    print(f"    B_c={trBc:.4f}  W_c={trWc:.4f}  B_c+W_c={cam_decomp_check:.4f}", flush=True)
    print(f"    B_c/(B_c+W_c)={trBc/max(cam_decomp_check,1e-10):.3f}  W_c/(B_c+W_c)={trWc/max(cam_decomp_check,1e-10):.3f}", flush=True)
    print(f"    cameras/ID: mean={per_id_ncams.mean():.2f}  max={per_id_ncams.max()}", flush=True)
    print(f"    IDs with >1 image: {n_ids_with_multiimg}  with >1 cam: {n_ids_with_multicam}  with >1 img/cam: {n_ids_with_multimg_per_cam}", flush=True)

    return {
        "n_samples": int(N),
        "n_ids": int(n_ids),
        "D": int(D),
        "tr_S_B": float(trSB),
        "tr_S_W": float(trSW),
        "fisher_ratio": float(fisher_ratio),
        "tr_between_cam": float(trBc),
        "tr_within_cam": float(trWc),
        "cam_decomp_fraction_between": float(trBc / max(cam_decomp_check, 1e-10)),
        "cam_decomp_fraction_within": float(trWc / max(cam_decomp_check, 1e-10)),
        "mean_cameras_per_id": float(per_id_ncams.mean()),
        "median_cameras_per_id": float(np.median(per_id_ncams)),
        "mean_images_per_id": float(per_id_n.mean()),
        "n_ids_with_multicam": int(n_ids_with_multicam),
        "n_ids_with_multiimg_per_cam": int(n_ids_with_multimg_per_cam),
        "per_id_between_cam_scatter_mean": float(per_id_B.mean()),
        "per_id_within_cam_scatter_mean": float(per_id_W.mean()),
    }


def hub_score_and_overlap(emb, pids, label="", k=KNN_K, n_sample=N_HUB_SAMPLE, rng_seed=42):
    """
    Compute per-ID hub score and overlap ratio.

    hub_score(ID i) = mean N_count of ID i's images in k-NN of sampled gallery images
    overlap_ratio(ID i) = within-ID angular radius / nearest other-ID centroid distance

    Pearson correlation between hub_score and overlap_ratio.
    """
    print(f"  Hub score + overlap [{label}]...", flush=True)
    rng = np.random.default_rng(rng_seed)
    N = len(pids)
    unique_ids = np.unique(pids)
    n_ids = len(unique_ids)

    # Compute per-ID centroids
    id_to_idx = {uid: i for i, uid in enumerate(unique_ids)}
    centroids = np.zeros((n_ids, emb.shape[1]), dtype=np.float64)
    counts = np.zeros(n_ids)
    for i, (x, pid) in enumerate(zip(emb, pids)):
        idx = id_to_idx[pid]
        centroids[idx] += x
        counts[idx] += 1
    centroids /= counts[:, np.newaxis]
    centroids_f = centroids.astype(np.float32)

    # Hub score: N-count per gallery image, aggregated by ID
    n_q = min(n_sample, N)
    sampled_idx = rng.choice(N, n_q, replace=False)
    sampled_emb = emb[sampled_idx].astype(np.float32)

    # Compute similarity to all gallery images in batches
    n_count = np.zeros(N)
    batch = 512
    for start in range(0, n_q, batch):
        end = min(start + batch, n_q)
        qb = sampled_emb[start:end]
        sim = qb @ emb.T   # (batch, N)
        # Zero out self-similarity (exact match)
        for qi in range(end - start):
            sim[qi, sampled_idx[start + qi]] = -1e9
        knn = np.argpartition(sim, -k, axis=1)[:, -k:]
        for row in knn:
            n_count[row] += 1

    # Aggregate N-count to ID level
    id_ncount = np.zeros(n_ids)
    for img_idx, pid in enumerate(pids):
        id_ncount[id_to_idx[pid]] += n_count[img_idx]
    # Normalize by ID size (per-image average hub score)
    id_ncount_normalized = id_ncount / np.maximum(counts, 1)

    # Overlap ratio per ID
    overlap_ratios = np.zeros(n_ids)
    # Pairwise distances between centroids (cosine: 1 - sim)
    cent_sim = centroids_f @ centroids_f.T   # (n_ids, n_ids)
    np.fill_diagonal(cent_sim, 1.0)           # self = 1.0 (max sim)
    nearest_other_sim = np.max(cent_sim - np.eye(n_ids) * 2.0, axis=1)  # exclude self
    nearest_other_dist = 1.0 - nearest_other_sim  # cosine distance

    for id_idx, uid in enumerate(unique_ids):
        mask = pids == uid
        X_id = emb[mask].astype(np.float32)
        mu = centroids_f[id_idx]
        # Angular radius = max cosine distance from centroid to any image
        sims = X_id @ mu
        norms_x = np.linalg.norm(X_id, axis=1)
        mu_norm = np.linalg.norm(mu)
        cos_sims = sims / (norms_x * mu_norm + 1e-8)
        angular_radius = float(1.0 - cos_sims.min())   # max cos distance = angular radius
        dist_to_nearest = float(nearest_other_dist[id_idx])
        overlap_ratios[id_idx] = angular_radius / max(dist_to_nearest, 1e-8)

    # Pearson correlation
    valid = np.isfinite(overlap_ratios) & np.isfinite(id_ncount_normalized)
    if valid.sum() > 10:
        corr = float(np.corrcoef(overlap_ratios[valid], id_ncount_normalized[valid])[0, 1])
    else:
        corr = float("nan")

    print(f"    overlap_ratio: mean={overlap_ratios.mean():.4f} median={np.median(overlap_ratios):.4f}", flush=True)
    print(f"    hub_score: mean={id_ncount_normalized.mean():.4f} median={np.median(id_ncount_normalized):.4f}", flush=True)
    print(f"    Pearson corr(overlap, hub): {corr:.4f}", flush=True)

    return {
        "mean_overlap_ratio": float(overlap_ratios.mean()),
        "median_overlap_ratio": float(np.median(overlap_ratios)),
        "std_overlap_ratio": float(overlap_ratios.std()),
        "mean_hub_score_normalized": float(id_ncount_normalized.mean()),
        "median_hub_score_normalized": float(np.median(id_ncount_normalized)),
        "pearson_corr_overlap_hub": corr,
        "n_ids": int(n_ids),
        "n_sampled_queries_for_hub": int(n_q),
        "knn_k": k,
        # Store percentiles for scatter plot
        "overlap_ratio_p25": float(np.percentile(overlap_ratios, 25)),
        "overlap_ratio_p75": float(np.percentile(overlap_ratios, 75)),
        "hub_score_p25": float(np.percentile(id_ncount_normalized, 25)),
        "hub_score_p75": float(np.percentile(id_ncount_normalized, 75)),
    }


def main():
    print("E18: Per-ID subspace geometry", flush=True)

    # Load cached features
    c_mkt = np.load(CACHE_DIR / "market_test.npz")
    mkt_emb, mkt_pids, mkt_cids = c_mkt["emb"], c_mkt["pids"], c_mkt["cids"]

    c_msmt = np.load(CACHE_DIR / "msmt17_test.npz")
    msmt_emb, msmt_pids, msmt_cids = c_msmt["emb"], c_msmt["pids"], c_msmt["cids"]

    c_bmkt = np.load(CACHE_DIR / "bot_msmt_seed42_market.npz")
    bmkt_emb = c_bmkt["emb"]

    c_bmsmt = np.load(CACHE_DIR / "bot_msmt_seed42_msmt.npz")
    bmsmt_emb = c_bmsmt["emb"]

    # Gallery splits
    mkt_g_emb = mkt_emb[N_Q_MKT:]; mkt_g_pids = mkt_pids[N_Q_MKT:]; mkt_g_cids = mkt_cids[N_Q_MKT:]
    msmt_g_emb = msmt_emb[N_Q_MSMT:]; msmt_g_pids = msmt_pids[N_Q_MSMT:]; msmt_g_cids = msmt_cids[N_Q_MSMT:]
    bmkt_g_emb = bmkt_emb[N_Q_MKT:]; bmsmt_g_emb = bmsmt_emb[N_Q_MSMT:]

    results = {}

    # =========================================================================
    # PHASE 1: Fisher decomposition (post first)
    # =========================================================================
    print("\n" + "="*60, flush=True)
    print("PHASE 1: FISHER DECOMPOSITION", flush=True)
    print("="*60, flush=True)

    print("\nCell A — BoT-Market gallery on Market:", flush=True)
    results["A"] = {"fisher": fisher_decomposition(mkt_g_emb, mkt_g_pids, mkt_g_cids, "Cell A gallery")}

    print("\nCell B — BoT-Market gallery on MSMT17:", flush=True)
    results["B"] = {"fisher": fisher_decomposition(msmt_g_emb, msmt_g_pids, msmt_g_cids, "Cell B gallery")}

    print("\nCell C — BoT-MSMT17 gallery on Market:", flush=True)
    results["C"] = {"fisher": fisher_decomposition(bmkt_g_emb, mkt_g_pids, mkt_g_cids, "Cell C gallery")}

    print("\nCell D — BoT-MSMT17 gallery on MSMT17:", flush=True)
    results["D"] = {"fisher": fisher_decomposition(bmsmt_g_emb, msmt_g_pids, msmt_g_cids, "Cell D gallery")}

    # Print comparative summary
    print("\n" + "="*60, flush=True)
    print("FISHER DECOMPOSITION SUMMARY", flush=True)
    print("="*60, flush=True)
    print(f"{'Cell':>5} {'tr(S_B)':>10} {'tr(S_W)':>10} {'F':>8} {'B_c/(B+W)':>10} {'n_cams/ID':>10}", flush=True)
    for cell in ["A", "B", "C", "D"]:
        r = results[cell]["fisher"]
        print(f"  {cell:>3}  {r['tr_S_B']:>10.4f} {r['tr_S_W']:>10.4f} {r['fisher_ratio']:>8.4f} "
              f"{r['cam_decomp_fraction_between']:>10.3f} {r['mean_cameras_per_id']:>10.2f}", flush=True)

    # =========================================================================
    # PHASE 2: Hub score + overlap ratio
    # =========================================================================
    print("\n" + "="*60, flush=True)
    print("PHASE 2: HUB SCORE + OVERLAP RATIO", flush=True)
    print("="*60, flush=True)

    print("\nCell A gallery:", flush=True)
    results["A"]["overlap_hub"] = hub_score_and_overlap(mkt_g_emb, mkt_g_pids, "Cell A")

    print("\nCell B gallery:", flush=True)
    results["B"]["overlap_hub"] = hub_score_and_overlap(msmt_g_emb, msmt_g_pids, "Cell B")

    print("\nCell C gallery:", flush=True)
    results["C"]["overlap_hub"] = hub_score_and_overlap(bmkt_g_emb, mkt_g_pids, "Cell C")

    print("\nCell D gallery:", flush=True)
    results["D"]["overlap_hub"] = hub_score_and_overlap(bmsmt_g_emb, msmt_g_pids, "Cell D")

    # Summary
    print("\n" + "="*60, flush=True)
    print("HUB + OVERLAP SUMMARY", flush=True)
    print("="*60, flush=True)
    print(f"{'Cell':>5} {'overlap_mean':>14} {'hub_mean':>12} {'corr(ov,hub)':>14}", flush=True)
    for cell in ["A", "B", "C", "D"]:
        r = results[cell]["overlap_hub"]
        print(f"  {cell:>3}  {r['mean_overlap_ratio']:>14.4f} {r['mean_hub_score_normalized']:>12.4f} "
              f"{r['pearson_corr_overlap_hub']:>14.4f}", flush=True)

    # Save
    out_path = RESULTS_DIR / "e18_per_id_geometry.json"
    with open(out_path, "w") as f:
        json.dump({"description": "E18 per-ID subspace geometry (Fisher + camera decomp + overlap/hub)",
                   "confound_notes": {
                       "cameras": "Market=6 cams, MSMT17=15 cams; cam_decomp_fraction_between accounts for this",
                       "sample_size": "~26 imgs/ID in D=2048; no per-ID eff_rank; use scalar tr(S_W)/tr(S_B)",
                       "unseen_id_vs_domain": "Compare in-domain cells (A,D) vs cross-domain (B,C) with same measures"
                   },
                   "cells": results}, f, indent=2)
    print(f"\nSaved → {out_path}", flush=True)


if __name__ == "__main__":
    main()
