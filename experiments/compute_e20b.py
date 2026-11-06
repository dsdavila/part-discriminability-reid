"""
E20b — Three Spearman correlations per eval set (never pooled).
Protocol: canonical predicate junk=(g_pid==q_pid)&(g_cam==q_cam) for per-ID AP.
Feature space: gallery-only for hub/overlap/centroid; query+gallery split for AP.
Hub score: per-gallery-item k-occurrence in sampled gallery k-NN, k=10, N_SAMPLE=5000.
Seed: RNG=42. Cache: e12_feature_cache.

Correlations (Spearman rho, per eval set = Market and MSMT17 separately):
  C1: per-ID overlap ratio vs per-ID AP
  C2: per-item hub score vs cos(x, mu_hat) [load-bearing]; also vs z-scored projection
  C3: per-ID AP vs cos(mu_hat_ID, mu_hat_global)

Eval sets:
  Market:  Cell A (BoT-Market/Market)  and  Cell C (BoT-MSMT17/Market)
  MSMT17:  Cell B (BoT-Market/MSMT17) and  Cell D (BoT-MSMT17/MSMT17)
"""
import numpy as np, json
from pathlib import Path
from scipy.stats import spearmanr

CACHE_DIR = Path("/tmp/e12_feature_cache")
RESULTS_DIR = Path("/data/papers/low_dimensional_reid/results")
N_Q_MKT = 3368
N_Q_MSMT = 11655
KNN_K = 10
N_HUB_SAMPLE = 5000
RNG = np.random.default_rng(42)

def load_cell(fname, n_query):
    d = np.load(CACHE_DIR / fname)
    emb = d["emb"].astype(np.float32)
    pids, cids = d["pids"], d["cids"]
    q_emb, q_pids, q_cids = emb[:n_query], pids[:n_query], cids[:n_query]
    g_emb, g_pids, g_cids = emb[n_query:], pids[n_query:], cids[n_query:]
    return q_emb, q_pids, q_cids, g_emb, g_pids, g_cids

def per_query_ap_canonical(q_emb, q_pids, q_cids, g_emb, g_pids, g_cids, batch=256):
    """Returns (per_query_ap, query_pids) — canonical predicate."""
    n_q = len(q_pids)
    aps = []
    for start in range(0, n_q, batch):
        end = min(start + batch, n_q)
        sim = q_emb[start:end].astype(np.float32) @ g_emb.T
        order = np.argsort(-sim, axis=1)
        for qi in range(end - start):
            qpid, qcid = q_pids[start+qi], q_cids[start+qi]
            gp, gc = g_pids[order[qi]], g_cids[order[qi]]
            mask_valid = ~((gp == qpid) & (gc == qcid))
            mask_pos = (gp == qpid) & mask_valid
            if mask_pos.sum() == 0:
                aps.append(0.0); continue
            valid_idx = np.where(mask_valid)[0]
            pos_in_valid = np.where(mask_pos[valid_idx])[0]
            prec = np.arange(1, len(pos_in_valid)+1) / (pos_in_valid+1)
            aps.append(float(prec.mean()))
        if start % 2000 == 0:
            print(f"    AP batch {start}/{n_q}", flush=True)
    return np.array(aps), q_pids

def per_id_ap(aps, pids):
    """Aggregate per-query APs to per-ID mean APs."""
    unique_ids = np.unique(pids)
    id_ap = {}
    for uid in unique_ids:
        mask = pids == uid
        id_ap[uid] = float(np.mean(aps[mask]))
    return id_ap

def hub_score(g_emb, g_pids, n_sample=N_HUB_SAMPLE, k=KNN_K, rng=RNG):
    """Per-item hub score: k-occurrence in k-NN of sampled gallery items."""
    N = len(g_pids)
    n_q = min(n_sample, N)
    idx_q = rng.choice(N, n_q, replace=False)
    qemb = g_emb[idx_q]
    n_count = np.zeros(N, dtype=np.float32)
    batch = 256
    for start in range(0, n_q, batch):
        end = min(start + batch, n_q)
        sim = qemb[start:end] @ g_emb.T
        for qi in range(end - start):
            sim[qi, idx_q[start+qi]] = -1e9  # mask self
        knn = np.argpartition(sim, -k, axis=1)[:, -k:]
        for row in knn:
            n_count[row] += 1
    return n_count / n_q  # normalise to rate

def per_id_overlap_ratio(g_emb, g_pids, rng=RNG):
    """Per-ID overlap ratio: angular radius / nearest-other-centroid cosine distance."""
    unique_ids = np.unique(g_pids)
    n_ids = len(unique_ids)
    id_to_idx = {uid: i for i, uid in enumerate(unique_ids)}

    # Centroids
    centroids = np.zeros((n_ids, g_emb.shape[1]), dtype=np.float64)
    counts = np.zeros(n_ids)
    for x, pid in zip(g_emb, g_pids):
        idx = id_to_idx[pid]
        centroids[idx] += x.astype(np.float64)
        counts[idx] += 1
    centroids /= counts[:, None]
    cent_norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    unit_cents = (centroids / cent_norms).astype(np.float32)

    # Nearest other centroid distances
    sim_matrix = unit_cents @ unit_cents.T  # (n_ids, n_ids)
    np.fill_diagonal(sim_matrix, -2.0)
    nearest_sim = sim_matrix.max(axis=1)
    nearest_dist = np.clip(1.0 - nearest_sim, 0, 2)

    # Angular radii (max cosine distance from centroid to any image)
    angular_radii = np.zeros(n_ids)
    for id_idx, uid in enumerate(unique_ids):
        mask = g_pids == uid
        X = g_emb[mask].astype(np.float64)
        u = unit_cents[id_idx].astype(np.float64)
        cos_sims = (X @ u) / (np.linalg.norm(X, axis=1) * np.linalg.norm(u) + 1e-8)
        cos_sims = np.clip(cos_sims, -1, 1)
        angular_radii[id_idx] = float(1.0 - cos_sims.min())  # max cosine distance

    overlap = angular_radii / np.maximum(nearest_dist, 1e-8)
    return dict(zip(unique_ids.tolist(), overlap.tolist()))

def analyze_cell(label, q_emb, q_pids, q_cids, g_emb, g_pids, g_cids, mu_hat):
    print(f"\n=== Cell {label} ===", flush=True)
    mu_hat = np.array(mu_hat, dtype=np.float32)

    # Per-query AP -> per-ID AP
    print("  Computing per-query AP (canonical)...", flush=True)
    aps, _ = per_query_ap_canonical(q_emb, q_pids, q_cids, g_emb, g_pids, g_cids)
    id_ap_map = per_id_ap(aps, q_pids)
    print(f"  mAP={np.mean(aps)*100:.4f}%  n_query_ids={len(id_ap_map)}", flush=True)

    # Per-ID overlap ratio (gallery-based)
    print("  Computing per-ID overlap ratio...", flush=True)
    id_overlap_map = per_id_overlap_ratio(g_emb, g_pids)

    # Hub score (gallery-internal)
    print("  Computing hub scores...", flush=True)
    hub = hub_score(g_emb, g_pids)

    # cos(x, mu_hat) per gallery item
    g_emb_f64 = g_emb.astype(np.float64)
    g_norms = np.linalg.norm(g_emb_f64, axis=1)
    cos_x_mu = (g_emb_f64 @ mu_hat.astype(np.float64)) / np.maximum(g_norms, 1e-8)
    cos_x_mu_zscore = (cos_x_mu - cos_x_mu.mean()) / (cos_x_mu.std() + 1e-10)

    # Per-ID centroid direction vs global mean
    unique_ids_g = np.unique(g_pids)
    centroids_g = np.zeros((len(unique_ids_g), g_emb.shape[1]), dtype=np.float64)
    counts_g = np.zeros(len(unique_ids_g))
    id_to_idx_g = {uid: i for i, uid in enumerate(unique_ids_g)}
    for x, pid in zip(g_emb_f64, g_pids):
        centroids_g[id_to_idx_g[pid]] += x
        counts_g[id_to_idx_g[pid]] += 1
    centroids_g /= counts_g[:, None]
    mu_global_g = g_emb_f64.mean(0)
    mu_global_g /= np.linalg.norm(mu_global_g)
    cent_norms_g = np.linalg.norm(centroids_g, axis=1)
    cos_cent_mu = (centroids_g @ mu_global_g) / np.maximum(cent_norms_g, 1e-8)
    id_cos_cent_mu = dict(zip(unique_ids_g.tolist(), cos_cent_mu.tolist()))

    # ---- Correlations ----
    # C1: per-ID overlap ratio vs per-ID AP
    # Match on IDs present in both query and gallery
    common_ids = sorted(set(id_ap_map) & set(id_overlap_map))
    c1_ap = np.array([id_ap_map[i] for i in common_ids])
    c1_ov = np.array([id_overlap_map[i] for i in common_ids])
    rho_c1 = spearmanr(c1_ov, c1_ap).statistic
    print(f"  C1 overlap vs AP: rho={rho_c1:.4f}  n_ids={len(common_ids)}", flush=True)

    # C2: per-item hub score vs cos(x, mu_hat)
    rho_c2_raw = spearmanr(hub, cos_x_mu).statistic
    rho_c2_z   = spearmanr(hub, cos_x_mu_zscore).statistic
    print(f"  C2 hub vs cos(x,mu_hat): rho={rho_c2_raw:.4f}", flush=True)
    print(f"  C2 hub vs z-score(x.mu_hat): rho={rho_c2_z:.4f}", flush=True)

    # C3: per-ID AP vs cos(mu_hat_ID, mu_hat_global) — match on query IDs present in gallery
    common_ids_c3 = sorted(set(id_ap_map) & set(id_cos_cent_mu))
    c3_ap = np.array([id_ap_map[i] for i in common_ids_c3])
    c3_cos = np.array([id_cos_cent_mu[i] for i in common_ids_c3])
    rho_c3 = spearmanr(c3_cos, c3_ap).statistic
    print(f"  C3 AP vs cos(cent_ID, mu_global): rho={rho_c3:.4f}  n_ids={len(common_ids_c3)}", flush=True)

    return {
        "mAP": float(np.mean(aps) * 100),
        "n_query_ids": len(id_ap_map),
        "n_gallery_ids": len(unique_ids_g),
        "n_common_c1": len(common_ids),
        "C1_overlap_vs_AP": {
            "spearman_rho": float(rho_c1),
            "n_ids": len(common_ids),
            "overlap_mean": float(c1_ov.mean()),
            "ap_mean": float(c1_ap.mean()),
        },
        "C2_hub_vs_cos_x_mu": {
            "spearman_rho_raw_projection": float(rho_c2_raw),
            "spearman_rho_zscore_projection": float(rho_c2_z),
            "hub_mean": float(hub.mean()),
            "cos_x_mu_mean": float(cos_x_mu.mean()),
            "n_items": len(hub),
        },
        "C3_AP_vs_cos_cent_ID_mu_global": {
            "spearman_rho": float(rho_c3),
            "n_ids": len(common_ids_c3),
        },
    }


# Load mu_hat from E20a
with open(RESULTS_DIR / "e20a_cone_direct.json") as f:
    e20a = json.load(f)

# Reconstruct mu_hat vectors from E20a (we need to recompute since we stripped them)
# Just recompute them from gallery embeddings
def get_mu_hat(fname, n_query):
    d = np.load(CACHE_DIR / fname)
    g_emb = d["emb"][n_query:].astype(np.float64)
    mu = g_emb.mean(0)
    return (mu / np.linalg.norm(mu)).astype(np.float32)

print("Loading cells and computing mu_hat...", flush=True)
mu_hat_A = get_mu_hat("market_test.npz", N_Q_MKT)
mu_hat_B = get_mu_hat("msmt17_test.npz", N_Q_MSMT)
mu_hat_C = get_mu_hat("bot_msmt_seed42_market.npz", N_Q_MKT)
mu_hat_D = get_mu_hat("bot_msmt_seed42_msmt.npz", N_Q_MSMT)

results = {}

print("\n--- Market eval set ---", flush=True)
q_emb, q_pids, q_cids, g_emb, g_pids, g_cids = load_cell("market_test.npz", N_Q_MKT)
results["A"] = analyze_cell("A", q_emb, q_pids, q_cids, g_emb, g_pids, g_cids, mu_hat_A)

q_emb, q_pids, q_cids, g_emb, g_pids, g_cids = load_cell("bot_msmt_seed42_market.npz", N_Q_MKT)
results["C"] = analyze_cell("C", q_emb, q_pids, q_cids, g_emb, g_pids, g_cids, mu_hat_C)

print("\n--- MSMT17 eval set ---", flush=True)
q_emb, q_pids, q_cids, g_emb, g_pids, g_cids = load_cell("msmt17_test.npz", N_Q_MSMT)
results["B"] = analyze_cell("B", q_emb, q_pids, q_cids, g_emb, g_pids, g_cids, mu_hat_B)

q_emb, q_pids, q_cids, g_emb, g_pids, g_cids = load_cell("bot_msmt_seed42_msmt.npz", N_Q_MSMT)
results["D"] = analyze_cell("D", q_emb, q_pids, q_cids, g_emb, g_pids, g_cids, mu_hat_D)

# Ordering check (monotonicity): mAP ordering expected A > D > C > B
print("\n=== MONOTONICITY CHECK ===", flush=True)
for cell in ["A","D","C","B"]:
    print(f"  Cell {cell}: mAP={results[cell]['mAP']:.4f}%  C1_rho={results[cell]['C1_overlap_vs_AP']['spearman_rho']:.4f}  C2_rho={results[cell]['C2_hub_vs_cos_x_mu']['spearman_rho_raw_projection']:.4f}  C3_rho={results[cell]['C3_AP_vs_cos_cent_ID_mu_global']['spearman_rho']:.4f}", flush=True)

out = {
    "protocol": "canonical junk=(g_pid==q_pid)&(g_cam==q_cam); hub=k-occurrence in gallery k-NN (k=10, N_sample=5000); per eval set (never pooled)",
    "cache": "e12_feature_cache",
    "seed": "RNG=42",
    "cells": results,
    "eval_set_structure": {
        "Market": ["A", "C"],
        "MSMT17": ["B", "D"]
    }
}
out_path = RESULTS_DIR / "e20b_correlations.json"
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)
print(f"\nSaved → {out_path}", flush=True)
