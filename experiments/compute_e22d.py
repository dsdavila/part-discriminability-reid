"""
E22d — Label-free colour prediction.

For each gallery image: resize to 256×128, split into vertical thirds
(exact boundaries below), extract mean RGB and mean HSV hue/saturation
per region; ridge-regress from 2048-d embedding to each colour statistic.

Region boundaries (pixel rows, inclusive, on 256×128 image):
  head:        rows   0–85   (fraction [0.000, 0.336))
  upper_body:  rows  86–170  (fraction [0.336, 0.668))
  lower_body:  rows 171–255  (fraction [0.668, 1.000])

Colour statistics per region (15 total per image):
  mean_R, mean_G, mean_B, mean_H, mean_S
  (H in [0,360], S in [0,1]; R/G/B in [0,1])

Train/test split: by identity, 80% train IDs / 20% test IDs (sorted, no shuffle).
Ridge regression: RidgeCV(alphas=[0.1, 1, 10, 100, 1000], cv=5).
Also: R² after projecting out μ̂ (per-cell gallery mean direction) from embeddings.

Per cell: A (BoT-Market / Market), C (BoT-MSMT / Market),
          D (BoT-MSMT / MSMT), B (BoT-Market / MSMT).

Cache: /tmp/e12_feature_cache/
Image source: dataset API (same order as cache).
"""
import sys
sys.path.insert(0, "/workspace/reid-tk")

import numpy as np
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
from sklearn.linear_model import RidgeCV
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from reid_tk.datasets import get_dataset

CACHE_DIR = Path("/tmp/e12_feature_cache")
RESULTS   = Path("/data/papers/low_dimensional_reid/results")
N_Q_MKT   = 3368
N_Q_MSMT  = 11655
IMG_H, IMG_W = 256, 128

# Region boundaries (pixel rows, inclusive)
REGIONS = {
    "head":        (0,   85),   # rows 0–85   (0.000–0.336)
    "upper_body":  (86,  170),  # rows 86–170 (0.336–0.668)
    "lower_body":  (171, 255),  # rows 171–255 (0.668–1.000)
}
COLOUR_STATS = [f"{reg}_{ch}" for reg in REGIONS for ch in ["R","G","B","H","S"]]


def rgb_to_hs(img_np):
    """
    Convert (H, W, 3) float32 RGB [0,1] array to H (degrees, [0,360]) and S ([0,1]).
    Returns (H_arr, S_arr) each of shape (H, W).
    """
    r, g, b = img_np[:,:,0], img_np[:,:,1], img_np[:,:,2]
    cmax = np.maximum(r, np.maximum(g, b))
    cmin = np.minimum(r, np.minimum(g, b))
    delta = cmax - cmin
    s = np.where(cmax > 1e-8, delta / cmax, 0.0)

    h = np.zeros_like(r)
    eps = 1e-8
    mask = delta > eps
    mr = mask & (cmax == r)
    mg = mask & (cmax == g)
    mb = mask & (cmax == b)
    if mr.any():
        h[mr] = 60.0 * (((g[mr] - b[mr]) / delta[mr]) % 6)
    if mg.any():
        h[mg] = 60.0 * ((b[mg] - r[mg]) / delta[mg] + 2.0)
    if mb.any():
        h[mb] = 60.0 * ((r[mb] - g[mb]) / delta[mb] + 4.0)
    return h, s


def colour_features_from_path(path):
    """Load image, resize to 256×128, return 15-d colour feature vector."""
    img = Image.open(path).convert("RGB").resize((IMG_W, IMG_H), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0   # (256, 128, 3)
    h_arr, s_arr = rgb_to_hs(arr)
    feats = []
    for (r0, r1) in REGIONS.values():
        patch = arr[r0:r1+1]        # shape (n_rows, 128, 3)
        ph    = h_arr[r0:r1+1]
        ps    = s_arr[r0:r1+1]
        feats.append(patch[:,:,0].mean())   # R
        feats.append(patch[:,:,1].mean())   # G
        feats.append(patch[:,:,2].mean())   # B
        feats.append(ph.mean())             # H degrees
        feats.append(ps.mean())             # S
    return np.array(feats, dtype=np.float32)


def load_colour_features(image_paths, n_workers=16):
    """Parallel colour feature extraction. Returns (N, 15) array in order."""
    results = [None] * len(image_paths)
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futs = {pool.submit(colour_features_from_path, p): i
                for i, p in enumerate(image_paths)}
        done = 0
        for fut in futs:
            results[futs[fut]] = fut.result()
            done += 1
            if done % 5000 == 0:
                print(f"    loaded {done}/{len(image_paths)}", flush=True)
    return np.stack(results, axis=0)


def load_gallery_cache(fname, n_q):
    d = np.load(CACHE_DIR / fname)
    emb  = d["emb"][n_q:].astype(np.float32)
    pids = d["pids"][n_q:]
    return emb, pids


def project_out_mean(emb):
    """Return embeddings with gallery mean direction projected out."""
    mu = emb.mean(axis=0).astype(np.float64)
    mu /= np.linalg.norm(mu) + 1e-12
    mu = mu.astype(np.float32)
    return emb - (emb @ mu[:, None]) * mu[None, :]


def id_split(pids, frac_train=0.8):
    """80/20 train/test split by sorted unique IDs. Returns bool masks."""
    unique = np.unique(pids)
    n_train = int(len(unique) * frac_train)
    train_ids = set(unique[:n_train])
    train_mask = np.array([p in train_ids for p in pids], dtype=bool)
    return train_mask, ~train_mask


def ridge_r2(X_train, Y_train, X_test, Y_test, alphas=(0.1, 1, 10, 100, 1000)):
    """
    Fit RidgeCV per target column, return held-out R² per column.
    """
    r2s = []
    for k in range(Y_train.shape[1]):
        y_tr = Y_train[:, k]
        y_te = Y_test[:, k]
        clf = RidgeCV(alphas=alphas, cv=5)
        clf.fit(X_train, y_tr)
        ss_res = np.sum((y_te - clf.predict(X_test)) ** 2)
        ss_tot = np.sum((y_te - y_te.mean()) ** 2) + 1e-12
        r2s.append(float(1.0 - ss_res / ss_tot))
    return r2s


def analyse_cell(label, emb, pids, colour_feat, stat_names):
    """Run ridge R² analysis for one cell, with and without μ̂ projection."""
    train_mask, test_mask = id_split(pids)
    X_tr, Y_tr = emb[train_mask], colour_feat[train_mask]
    X_te, Y_te = emb[test_mask],  colour_feat[test_mask]

    print(f"  Cell {label}: n_train={train_mask.sum()}  n_test={test_mask.sum()}", flush=True)

    r2_raw = ridge_r2(X_tr, Y_tr, X_te, Y_te)

    # Project out μ̂ (computed on train set only to avoid leakage)
    mu_tr = X_tr.mean(axis=0).astype(np.float64)
    mu_tr /= np.linalg.norm(mu_tr) + 1e-12
    mu_tr = mu_tr.astype(np.float32)
    X_tr_proj = X_tr - (X_tr @ mu_tr[:, None]) * mu_tr[None, :]
    X_te_proj = X_te - (X_te @ mu_tr[:, None]) * mu_tr[None, :]
    r2_proj = ridge_r2(X_tr_proj, Y_tr, X_te_proj, Y_te)

    for name, v_raw, v_proj in zip(stat_names, r2_raw, r2_proj):
        print(f"    {name}: R²={v_raw:.4f}  R²_proj={v_proj:.4f}", flush=True)

    return {
        "n_train": int(train_mask.sum()),
        "n_test":  int(test_mask.sum()),
        "n_train_ids": int(len(np.unique(pids[train_mask]))),
        "n_test_ids":  int(len(np.unique(pids[test_mask]))),
        "r2": {n: v for n, v in zip(stat_names, r2_raw)},
        "r2_proj_mu_hat": {n: v for n, v in zip(stat_names, r2_proj)},
    }


# ── Market gallery ─────────────────────────────────────────────────────────────
print("Building Market dataset...", flush=True)
mkt = get_dataset("market1501", Path("/data/datasets/reid/Market-1501-v15.09.15"))
mkt_gallery_samples = [s for s in mkt.gallery if not s.is_junk]
mkt_image_paths = [s.image_path for s in mkt_gallery_samples]
print(f"  Market gallery: {len(mkt_gallery_samples)} images", flush=True)

print("Extracting Market colour features...", flush=True)
mkt_colour = load_colour_features(mkt_image_paths)
print(f"  colour matrix: {mkt_colour.shape}", flush=True)

# Load Market gallery embeddings
print("Loading Market gallery embeddings...", flush=True)
A_emb, A_pids = load_gallery_cache("market_test.npz",            N_Q_MKT)
C_emb, C_pids = load_gallery_cache("bot_msmt_seed42_market.npz", N_Q_MKT)
assert len(A_pids) == len(mkt_gallery_samples), f"Market size mismatch: cache={len(A_pids)}, images={len(mkt_gallery_samples)}"
assert np.array_equal(A_pids, C_pids), "Market: A/C pid mismatch"

# ── MSMT17 gallery ─────────────────────────────────────────────────────────────
print("\nBuilding MSMT17 dataset...", flush=True)
mst = get_dataset("msmt17", Path("/data/datasets/reid/MSMT17_V1"))
mst_gallery_samples = [s for s in mst.gallery if not s.is_junk]
mst_image_paths = [s.image_path for s in mst_gallery_samples]
print(f"  MSMT17 gallery: {len(mst_gallery_samples)} images", flush=True)

print("Extracting MSMT17 colour features...", flush=True)
mst_colour = load_colour_features(mst_image_paths)
print(f"  colour matrix: {mst_colour.shape}", flush=True)

print("Loading MSMT17 gallery embeddings...", flush=True)
B_emb, B_pids = load_gallery_cache("msmt17_test.npz",            N_Q_MSMT)
D_emb, D_pids = load_gallery_cache("bot_msmt_seed42_msmt.npz",   N_Q_MSMT)
assert len(B_pids) == len(mst_gallery_samples), f"MSMT17 size mismatch: cache={len(B_pids)}, images={len(mst_gallery_samples)}"
assert np.array_equal(B_pids, D_pids), "MSMT17: B/D pid mismatch"

# ── Ridge regression per cell ──────────────────────────────────────────────────
print("\n=== Ridge regression: colour prediction ===", flush=True)
results = {}

print("\n-- Market eval set --", flush=True)
results["A"] = analyse_cell("A", A_emb, A_pids, mkt_colour, COLOUR_STATS)
results["C"] = analyse_cell("C", C_emb, C_pids, mkt_colour, COLOUR_STATS)

print("\n-- MSMT17 eval set --", flush=True)
results["D"] = analyse_cell("D", D_emb, D_pids, mst_colour, COLOUR_STATS)
results["B"] = analyse_cell("B", B_emb, B_pids, mst_colour, COLOUR_STATS)

# ── Save ──────────────────────────────────────────────────────────────────────
out = {
    "protocol": (
        "Vertical thirds of 256×128 resized gallery images: "
        "head rows 0–85 (fractions [0.000,0.336)), "
        "upper_body rows 86–170 ([0.336,0.668)), "
        "lower_body rows 171–255 ([0.668,1.000]). "
        "Colour stats: mean R,G,B ([0,1]) and mean H (degrees,[0,360]), S ([0,1]) per region. "
        "RidgeCV(alphas=[0.1,1,10,100,1000], cv=5) from 2048-d embedding to each stat. "
        "Train/test split by sorted ID, 80/20. "
        "r2_proj_mu_hat: same after projecting out train-set gallery mean direction (μ̂ from train IDs only)."
    ),
    "colour_stats": COLOUR_STATS,
    "region_boundaries": {
        k: {"rows": list(v), "fractions": [round(v[0]/256, 3), round((v[1]+1)/256, 3)]}
        for k, v in REGIONS.items()
    },
    "cache": "e12_feature_cache",
    "image_order": "dataset.gallery filtered by not is_junk, same order as cache",
    "cells": results,
}
out_path = RESULTS / "e22d_colour_pred_results.json"
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)
print(f"\nSaved → {out_path}", flush=True)
