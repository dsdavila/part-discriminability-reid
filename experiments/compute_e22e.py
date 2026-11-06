"""
E22e — Per-ID nearest-training-centroid angle vs per-ID AP.

Join of two already-computed measures at the per-identity level:
  1. nearest_train_angle: arccos of max cosine similarity from test ID centroid to
     any training ID centroid (same model/train set pair as E20c-v2)
  2. per_id_ap: per-ID mean AP under canonical predicate (junk = same-pid & same-cam)
  3. overlap_ratio: incumbent predictor from E20b — angular_radius / nearest_centroid_dist
  4. confounders: per-ID gallery image count, per-ID camera count

Reports per cell (paired within eval set, never pooled):
  Market:  A (BoT-Market/Market) and C (BoT-MSMT17/Market)
  MSMT17:  D (BoT-MSMT17/MSMT17) and B (BoT-Market/MSMT17)

For each cell:
  a. Spearman and Pearson: nearest_train_angle vs per-ID AP (with bootstrap CI, n_boot=1000)
  b. Spearman and Pearson: overlap_ratio vs per-ID AP (same bootstrap)
  c. Partial Spearman: nearest_train_angle vs AP, controlling for overlap (and reverse)
  d. Confound controls: correlation of gallery_count and cam_count with both predictors and AP
  e. Partial Spearman: nearest_train_angle vs AP, controlling for gallery_count + cam_count

Prototype pairs (from E20c-v2 / within-pair rule):
  A/B: BoT-Market model,  training prototypes = Market-train  (bot_market_market_train.npz)
  C/D: BoT-MSMT17 model, training prototypes = MSMT17-train  (bot_msmt_msmt_train.npz)

Pre-registered prediction (claude-chat rev 141):
  Spearman negative in all four cells; |rho| in [0.25, 0.50];
  stronger than overlap in cross-domain cells (B and C).
  Falsifier: |rho| < 0.15 in any cell → four-cell exponential is an n=4 artifact.

Cache: /tmp/e12_feature_cache/, /tmp/e18_train_cache/
Seed: RNG=42, n_boot=1000.
"""
import sys
sys.path.insert(0, "/workspace/reid-tk")
import numpy as np
import json
from pathlib import Path
from scipy.stats import spearmanr, pearsonr, rankdata

CACHE_DIR  = Path("/tmp/e12_feature_cache")
TRAIN_DIR  = Path("/tmp/e18_train_cache")
RESULTS    = Path("/data/papers/low_dimensional_reid/results")
N_Q_MKT    = 3368
N_Q_MSMT   = 11655
RNG        = np.random.default_rng(42)
N_BOOT     = 1000


# ── Helpers reused from E20b / E20c-v2 ────────────────────────────────────────

def load_cell(fname, n_q):
    d = np.load(CACHE_DIR / fname)
    emb = d["emb"].astype(np.float32)
    pids, cids = d["pids"], d["cids"]
    return (emb[:n_q], pids[:n_q], cids[:n_q],
            emb[n_q:],  pids[n_q:], cids[n_q:])

def load_train(fname):
    d = np.load(TRAIN_DIR / fname)
    return d["emb"].astype(np.float32), d["pids"]

def unit_centroids(emb, pids):
    unique_ids = np.unique(pids)
    n_ids, D = len(unique_ids), emb.shape[1]
    id2i = {uid: i for i, uid in enumerate(unique_ids)}
    c = np.zeros((n_ids, D), dtype=np.float64)
    cnt = np.zeros(n_ids)
    for x, pid in zip(emb, pids):
        i = id2i[pid]; c[i] += x.astype(np.float64); cnt[i] += 1
    c /= cnt[:, None]
    return (c / np.maximum(np.linalg.norm(c, axis=1, keepdims=True), 1e-9)).astype(np.float32), unique_ids, id2i

def per_id_nearest_train_angle(test_emb, test_pids, train_emb, train_pids):
    """arccos(max cos sim from each test centroid to any train centroid). Returns dict uid→angle_deg."""
    uc_test, ids_test, _ = unit_centroids(test_emb, test_pids)
    uc_train, _, _       = unit_centroids(train_emb, train_pids)
    cos_block = uc_test.astype(np.float64) @ uc_train.astype(np.float64).T  # (n_test, n_train)
    max_cos = np.clip(cos_block.max(axis=1), -1.0, 1.0)
    angles  = np.degrees(np.arccos(max_cos))
    return dict(zip(ids_test.tolist(), angles.tolist()))

def per_id_overlap_ratio(g_emb, g_pids):
    """From E20b: angular_radius / nearest_centroid_dist per gallery ID."""
    uc, ids, id2i = unit_centroids(g_emb, g_pids)
    n_ids = len(ids)
    sim = (uc.astype(np.float64) @ uc.astype(np.float64).T).astype(np.float32)
    np.fill_diagonal(sim, -2.0)
    nearest_sim  = sim.max(axis=1)
    nearest_dist = np.clip(1.0 - nearest_sim, 0, 2)
    ang_radii = np.zeros(n_ids)
    for idx, uid in enumerate(ids):
        mask = g_pids == uid
        X = g_emb[mask].astype(np.float64)
        u = uc[idx].astype(np.float64)
        cos_sims = (X @ u) / np.maximum(np.linalg.norm(X, axis=1) * np.linalg.norm(u), 1e-8)
        ang_radii[idx] = float(1.0 - np.clip(cos_sims, -1, 1).min())
    overlap = ang_radii / np.maximum(nearest_dist, 1e-8)
    return dict(zip(ids.tolist(), overlap.tolist()))

def per_query_ap(q_emb, q_pids, q_cids, g_emb, g_pids, g_cids, batch=512):
    """Returns per-query AP array and q_pids."""
    n_q = len(q_pids)
    aps = []
    for s in range(0, n_q, batch):
        e = min(s + batch, n_q)
        sim = q_emb[s:e].astype(np.float32) @ g_emb.T
        order = np.argsort(-sim, axis=1)
        for qi in range(e - s):
            qp, qc = q_pids[s+qi], q_cids[s+qi]
            gp, gc = g_pids[order[qi]], g_cids[order[qi]]
            valid = ~((gp == qp) & (gc == qc))
            pos   = (gp == qp) & valid
            if pos.sum() == 0:
                aps.append(0.0); continue
            vi  = np.where(valid)[0]
            pi  = np.where(pos[vi])[0]
            prec = np.arange(1, len(pi)+1) / (pi + 1)
            aps.append(float(prec.mean()))
        if s % 5000 == 0:
            print(f"    AP rows {s}/{n_q}", flush=True)
    return np.array(aps), q_pids

def agg_per_id_ap(aps, pids):
    ids = np.unique(pids)
    return {uid: float(aps[pids == uid].mean()) for uid in ids}

def per_id_gallery_stats(g_pids, g_cids):
    """Per-ID gallery image count and unique camera count."""
    ids = np.unique(g_pids)
    n_imgs = {uid: int((g_pids == uid).sum()) for uid in ids}
    n_cams = {uid: int(len(np.unique(g_cids[g_pids == uid]))) for uid in ids}
    return n_imgs, n_cams

# ── Statistics ────────────────────────────────────────────────────────────────

def boot_spearman(x, y, n_boot=N_BOOT, rng=RNG):
    n = len(x)
    rhos = []
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        rhos.append(float(spearmanr(x[idx], y[idx]).statistic))
    return np.array(rhos)

def partial_spearman(x, y, z):
    """Partial Spearman of x vs y, controlling for z (rank-based)."""
    rx = rankdata(x).astype(float)
    ry = rankdata(y).astype(float)
    rz = rankdata(z).astype(float)
    def ols_resid(a, b):
        b_ = b - b.mean(); a_ = a - a.mean()
        slope = (a_ * b_).sum() / ((b_ ** 2).sum() + 1e-12)
        return a - slope * b - (a.mean() - slope * b.mean())
    rx_r = ols_resid(rx, rz)
    ry_r = ols_resid(ry, rz)
    return float(pearsonr(rx_r, ry_r)[0])

def partial_spearman_2z(x, y, z1, z2):
    """Partial Spearman of x vs y, controlling for z1 and z2."""
    rx = rankdata(x).astype(float)
    ry = rankdata(y).astype(float)
    rz1 = rankdata(z1).astype(float)
    rz2 = rankdata(z2).astype(float)
    Z = np.column_stack([rz1 - rz1.mean(), rz2 - rz2.mean()])
    def ols_resid_multi(a, Z):
        a_ = a - a.mean()
        beta = np.linalg.lstsq(Z, a_, rcond=None)[0]
        return a_ - Z @ beta
    rx_r = ols_resid_multi(rx, Z)
    ry_r = ols_resid_multi(ry, Z)
    return float(pearsonr(rx_r, ry_r)[0])

def summarise(x, y, label_x, label_y, rng):
    n = len(x)
    rho_sp = float(spearmanr(x, y).statistic)
    rho_pe = float(pearsonr(x, y)[0])
    boots  = boot_spearman(x, y, rng=rng)
    ci_lo, ci_hi = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
    print(f"    {label_x} vs {label_y}: Spearman={rho_sp:.4f} [CI {ci_lo:.4f},{ci_hi:.4f}]  Pearson={rho_pe:.4f}  n={n}", flush=True)
    return {"spearman": rho_sp, "pearson": rho_pe,
            "boot_ci_95": [ci_lo, ci_hi], "n": n}


# ── Main ──────────────────────────────────────────────────────────────────────

print("Loading training embeddings...", flush=True)
mkt_tr_emb, mkt_tr_pids = load_train("bot_market_market_train.npz")
mst_tr_emb, mst_tr_pids = load_train("bot_msmt_msmt_train.npz")
print(f"  Market-train: {len(mkt_tr_pids)} imgs  MSMT17-train: {len(mst_tr_pids)} imgs", flush=True)

# Each cell: (cache_fname, n_q, train_emb, train_pids)
CELL_DEFS = {
    "A": ("market_test.npz",            N_Q_MKT,  mkt_tr_emb, mkt_tr_pids),
    "C": ("bot_msmt_seed42_market.npz", N_Q_MKT,  mst_tr_emb, mst_tr_pids),
    "D": ("bot_msmt_seed42_msmt.npz",   N_Q_MSMT, mst_tr_emb, mst_tr_pids),
    "B": ("msmt17_test.npz",            N_Q_MSMT, mkt_tr_emb, mkt_tr_pids),
}

results = {}

for cell, (fname, n_q, tr_emb, tr_pids) in CELL_DEFS.items():
    print(f"\n=== Cell {cell} ===", flush=True)
    q_emb, q_pids, q_cids, g_emb, g_pids, g_cids = load_cell(fname, n_q)

    # Per-ID nearest-train angle
    print("  nearest-train angle...", flush=True)
    nt_angle = per_id_nearest_train_angle(g_emb, g_pids, tr_emb, tr_pids)

    # Per-ID overlap ratio
    print("  overlap ratio...", flush=True)
    ov_ratio = per_id_overlap_ratio(g_emb, g_pids)

    # Per-ID AP
    print("  per-ID AP...", flush=True)
    aps, _   = per_query_ap(q_emb, q_pids, q_cids, g_emb, g_pids, g_cids)
    id_ap    = agg_per_id_ap(aps, q_pids)
    mAP      = float(np.mean(aps) * 100)
    print(f"  mAP={mAP:.4f}%", flush=True)

    # Gallery confounders
    n_imgs, n_cams = per_id_gallery_stats(g_pids, g_cids)

    # Join on IDs present in all three measures (query IDs ∩ gallery IDs)
    common_ids = sorted(set(nt_angle) & set(ov_ratio) & set(id_ap))
    n = len(common_ids)
    print(f"  n_common_ids={n}", flush=True)

    angle_arr  = np.array([nt_angle[i]  for i in common_ids])
    overlap_arr = np.array([ov_ratio[i] for i in common_ids])
    ap_arr      = np.array([id_ap[i]    for i in common_ids])
    n_img_arr   = np.array([n_imgs.get(i, 0) for i in common_ids], dtype=float)
    n_cam_arr   = np.array([n_cams.get(i, 0) for i in common_ids], dtype=float)

    print("  a. nearest-train angle vs AP:", flush=True)
    r_angle_ap = summarise(angle_arr, ap_arr, "NearestTrainAngle", "AP", RNG)

    print("  b. overlap ratio vs AP:", flush=True)
    r_overlap_ap = summarise(overlap_arr, ap_arr, "OverlapRatio", "AP", RNG)

    # Partial correlations
    part_angle_ctrl_overlap = partial_spearman(angle_arr, ap_arr, overlap_arr)
    part_overlap_ctrl_angle = partial_spearman(overlap_arr, ap_arr, angle_arr)
    print(f"  c. partial(NearestTrainAngle, AP | overlap)    = {part_angle_ctrl_overlap:.4f}", flush=True)
    print(f"     partial(overlap, AP | NearestTrainAngle)    = {part_overlap_ctrl_angle:.4f}", flush=True)

    # Confound controls
    r_cnt_angle  = summarise(n_img_arr, angle_arr,  "n_gallery_imgs", "NearestTrainAngle", RNG)
    r_cnt_ap     = summarise(n_img_arr, ap_arr,     "n_gallery_imgs", "AP", RNG)
    r_cam_angle  = summarise(n_cam_arr, angle_arr,  "n_cameras", "NearestTrainAngle", RNG)
    r_cam_ap     = summarise(n_cam_arr, ap_arr,     "n_cameras", "AP", RNG)
    part_angle_ctrl_cnt_cam = partial_spearman_2z(angle_arr, ap_arr, n_img_arr, n_cam_arr)
    print(f"  e. partial(NearestTrainAngle, AP | n_imgs, n_cams) = {part_angle_ctrl_cnt_cam:.4f}", flush=True)

    results[cell] = {
        "mAP": mAP,
        "n_common_ids": n,
        "nearest_train_angle_mean": float(angle_arr.mean()),
        "nearest_train_angle_std":  float(angle_arr.std()),
        "overlap_ratio_mean":  float(overlap_arr.mean()),
        "nearest_train_angle_vs_AP": r_angle_ap,
        "overlap_ratio_vs_AP": r_overlap_ap,
        "partial_angle_vs_AP_ctrl_overlap": float(part_angle_ctrl_overlap),
        "partial_overlap_vs_AP_ctrl_angle": float(part_overlap_ctrl_angle),
        "confound_n_gallery_imgs_vs_angle": r_cnt_angle,
        "confound_n_gallery_imgs_vs_AP":    r_cnt_ap,
        "confound_n_cameras_vs_angle": r_cam_angle,
        "confound_n_cameras_vs_AP":    r_cam_ap,
        "partial_angle_vs_AP_ctrl_count_and_cams": float(part_angle_ctrl_cnt_cam),
    }

# Save
out = {
    "protocol": (
        "Per-ID join: nearest_train_angle=arccos(max_cos_to_train_centroid) (same model/train pair as E20c-v2); "
        "overlap_ratio=angular_radius/nearest_centroid_dist (E20b definition); "
        "per_id_ap=mean AP over queries of that identity (canonical predicate: junk=same-pid&same-cam). "
        "Bootstrap CI (95%) over identities, n_boot=1000, RNG=42. "
        "Partial Spearman: rank all variables, OLS-residualise ranks on control(s), Pearson of residuals. "
        "Confounds: per-ID gallery image count and per-ID camera count. "
        "Paired within eval set only (Market: A,C; MSMT17: D,B)."
    ),
    "prototype_pairs": {
        "A_and_B": "BoT-Market model vs Market-train prototypes (bot_market_market_train.npz)",
        "C_and_D": "BoT-MSMT17 model vs MSMT17-train prototypes (bot_msmt_msmt_train.npz)",
    },
    "cache": "e12_feature_cache + e18_train_cache",
    "seed": "RNG=42",
    "n_boot": N_BOOT,
    "cells": results,
}
out_path = RESULTS / "e22e_angle_vs_ap.json"
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)
print(f"\nSaved → {out_path}", flush=True)
