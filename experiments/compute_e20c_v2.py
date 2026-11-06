"""
E20c-v2: Raw centroid-to-centroid angle distributions.
Drop caps/thresholds entirely. Report theta_ij = arccos(u_i . u_j) in degrees.

For each cell within-pair (same prototype source as E20c):
  A/B pair: Market-train under BoT-Market  (bot_market_market_train.npz, 751 IDs)
  C/D pair: MSMT17-train under BoT-MSMT17  (bot_msmt_msmt_train.npz, 1041 IDs)

Measurements:
  1. Test-test: pairwise theta_ij distribution among test IDs (per cell)
  2. Test-train (within-pair): theta_ij distribution for all (test_i, train_j) pairs
  3. Per-test-ID nearest-training-centroid angle: min_j arccos(u_i . u_j)
  4. Random null: generated from actual gallery sizes in 2048-d

Cache: e12_feature_cache + e18_train_cache. Seed: RNG=42.
"""
import numpy as np, json
from pathlib import Path

CACHE_DIR  = Path("/tmp/e12_feature_cache")
TRAIN_DIR  = Path("/tmp/e18_train_cache")
RESULTS    = Path("/data/papers/low_dimensional_reid/results")
N_Q_MKT    = 3368
N_Q_MSMT   = 11655
RNG        = np.random.default_rng(42)
NULL_REPS  = 3  # average three null draws for stability
PCTILES    = [1, 5, 10, 25, 50, 75, 90, 95, 99]

def load_gallery(fname, n_q):
    d = np.load(CACHE_DIR / fname)
    g_emb = d["emb"][n_q:].astype(np.float32)
    g_pids = d["pids"][n_q:]
    return g_emb, g_pids

def load_train(fname):
    d = np.load(TRAIN_DIR / fname)
    return d["emb"].astype(np.float32), d["pids"]

def unit_centroids(emb, pids):
    unique_ids = np.unique(pids)
    n_ids = len(unique_ids)
    D = emb.shape[1]
    id_to_idx = {uid: i for i, uid in enumerate(unique_ids)}
    cents = np.zeros((n_ids, D), dtype=np.float64)
    counts = np.zeros(n_ids)
    for x, pid in zip(emb, pids):
        idx = id_to_idx[pid]
        cents[idx] += x.astype(np.float64)
        counts[idx] += 1
    cents /= counts[:, None]
    norms = np.linalg.norm(cents, axis=1, keepdims=True)
    return (cents / np.maximum(norms, 1e-9)).astype(np.float32), unique_ids

def pairwise_angles_deg(ucA, ucB, self_compare=False):
    """All pairwise arccos(u_i . u_j) in degrees.
    If self_compare, exclude diagonal (upper-tri only for efficiency)."""
    if self_compare:
        n = len(ucA)
        i_idx, j_idx = np.triu_indices(n, k=1)
        cos_ij = np.sum(ucA[i_idx].astype(np.float64) * ucB[j_idx].astype(np.float64), axis=1)
        cos_ij = np.clip(cos_ij, -1.0, 1.0)
        return np.degrees(np.arccos(cos_ij))
    else:
        # All n_A x n_B pairs — may be large; chunk if needed
        n_A = len(ucA)
        n_B = len(ucB)
        angles_flat = []
        chunk = 200
        for start in range(0, n_A, chunk):
            end = min(start + chunk, n_A)
            cos_block = ucA[start:end].astype(np.float64) @ ucB.astype(np.float64).T
            cos_block = np.clip(cos_block, -1.0, 1.0)
            angles_flat.append(np.degrees(np.arccos(cos_block)).ravel())
        return np.concatenate(angles_flat)

def angle_stats(angles, label=""):
    pcts = np.percentile(angles, PCTILES)
    stats = {
        "n_pairs": int(len(angles)),
        "mean": float(angles.mean()),
        "std": float(angles.std()),
        "min": float(angles.min()),
        "max": float(angles.max()),
        "percentiles": {f"p{p}": float(v) for p, v in zip(PCTILES, pcts)},
    }
    if label:
        print(f"  {label}: n={len(angles)}  mean={stats['mean']:.2f}°  std={stats['std']:.2f}°  "
              f"p10={stats['percentiles']['p10']:.2f}°  p50={stats['percentiles']['p50']:.2f}°  "
              f"p90={stats['percentiles']['p90']:.2f}°", flush=True)
    return stats

def random_null(n_A, n_B, D=2048, reps=NULL_REPS, rng=RNG):
    """Generated random null: n_A x n_B pairwise angles between uniform unit vectors in D dims."""
    all_means = []
    all_p50 = []
    all_p10 = []
    all_p90 = []
    for _ in range(reps):
        vA = rng.standard_normal((n_A, D)).astype(np.float64)
        vA /= np.linalg.norm(vA, axis=1, keepdims=True)
        vB = rng.standard_normal((n_B, D)).astype(np.float64)
        vB /= np.linalg.norm(vB, axis=1, keepdims=True)
        # Sample up to 50k pairs to save time
        n_pairs = min(n_A * n_B, 50000)
        i_idx = rng.integers(0, n_A, n_pairs)
        j_idx = rng.integers(0, n_B, n_pairs)
        cos_ij = np.sum(vA[i_idx] * vB[j_idx], axis=1)
        cos_ij = np.clip(cos_ij, -1.0, 1.0)
        theta = np.degrees(np.arccos(cos_ij))
        all_means.append(float(theta.mean()))
        all_p50.append(float(np.percentile(theta, 50)))
        all_p10.append(float(np.percentile(theta, 10)))
        all_p90.append(float(np.percentile(theta, 90)))
    return {
        "n_A": n_A, "n_B": n_B, "D": D, "reps": reps,
        "mean": float(np.mean(all_means)),
        "p10": float(np.mean(all_p10)),
        "p50": float(np.mean(all_p50)),
        "p90": float(np.mean(all_p90)),
    }

# ---- Load ----
print("Loading gallery embeddings...", flush=True)
A_emb, A_pids = load_gallery("market_test.npz",            N_Q_MKT)
B_emb, B_pids = load_gallery("msmt17_test.npz",            N_Q_MSMT)
C_emb, C_pids = load_gallery("bot_msmt_seed42_market.npz", N_Q_MKT)
D_emb, D_pids = load_gallery("bot_msmt_seed42_msmt.npz",   N_Q_MSMT)

print("Loading training embeddings...", flush=True)
mkt_tr_emb, mkt_tr_pids = load_train("bot_market_market_train.npz")
msmt_tr_emb, msmt_tr_pids = load_train("bot_msmt_msmt_train.npz")

# ---- Compute unit centroids ----
print("Computing unit centroids...", flush=True)
uc_A, ids_A = unit_centroids(A_emb, A_pids)
uc_B, ids_B = unit_centroids(B_emb, B_pids)
uc_C, ids_C = unit_centroids(C_emb, C_pids)
uc_D, ids_D = unit_centroids(D_emb, D_pids)
uc_mkt_tr, _ = unit_centroids(mkt_tr_emb, mkt_tr_pids)
uc_msmt_tr, _ = unit_centroids(msmt_tr_emb, msmt_tr_pids)

for label, uc in [("A", uc_A), ("B", uc_B), ("C", uc_C), ("D", uc_D),
                   ("mkt_train", uc_mkt_tr), ("msmt_train", uc_msmt_tr)]:
    print(f"  {label}: {len(uc)} centroids", flush=True)

results = {}

# ---- Test-test pairwise angles ----
print("\n=== Test-test pairwise centroid angles ===", flush=True)
for cell, uc in [("A", uc_A), ("B", uc_B), ("C", uc_C), ("D", uc_D)]:
    angles = pairwise_angles_deg(uc, uc, self_compare=True)
    results.setdefault("test_test", {})[cell] = angle_stats(angles, f"Cell {cell} test-test")

# ---- Test-train pairwise angles (within-pair) ----
print("\n=== Test-train pairwise centroid angles (within-pair) ===", flush=True)
within_pair = [
    ("A", uc_A, uc_mkt_tr, "Market-train/BoT-Market"),
    ("B", uc_B, uc_mkt_tr, "Market-train/BoT-Market"),
    ("C", uc_C, uc_msmt_tr, "MSMT-train/BoT-MSMT17"),
    ("D", uc_D, uc_msmt_tr, "MSMT-train/BoT-MSMT17"),
]
for cell, uc_test, uc_train, proto in within_pair:
    angles = pairwise_angles_deg(uc_test, uc_train)
    results.setdefault("test_train", {})[cell] = {
        **angle_stats(angles, f"Cell {cell} test-train ({proto})"),
        "prototype_source": proto,
    }

# ---- Per-test-ID nearest-training-centroid angle ----
print("\n=== Per-test-ID nearest training centroid (min angle) ===", flush=True)
for cell, uc_test, uc_train, proto in within_pair:
    # For each test ID, find minimum angle to any training centroid
    cos_block = uc_test.astype(np.float64) @ uc_train.astype(np.float64).T  # (n_test, n_train)
    cos_block = np.clip(cos_block, -1.0, 1.0)
    max_cos = cos_block.max(axis=1)  # nearest = max cosine = min angle
    nearest_angles = np.degrees(np.arccos(max_cos))
    s = angle_stats(nearest_angles, f"Cell {cell} nearest-train-centroid")
    results.setdefault("nearest_train_centroid", {})[cell] = {**s, "prototype_source": proto}

# ---- Random null ----
print("\n=== Random null (generated) ===", flush=True)
D = 2048
for cell, uc_test, uc_train, _ in within_pair:
    null = random_null(len(uc_test), len(uc_train), D=D)
    print(f"  Cell {cell} null ({len(uc_test)}x{len(uc_train)}): "
          f"mean={null['mean']:.2f}°  p10={null['p10']:.2f}°  p50={null['p50']:.2f}°  p90={null['p90']:.2f}°",
          flush=True)
    results.setdefault("random_null_test_train", {})[cell] = null

# Test-test null
for cell, uc in [("A", uc_A), ("B", uc_B), ("C", uc_C), ("D", uc_D)]:
    null = random_null(len(uc), len(uc), D=D)
    print(f"  Cell {cell} test-test null: mean={null['mean']:.2f}°  p50={null['p50']:.2f}°", flush=True)
    results.setdefault("random_null_test_test", {})[cell] = null

out = {
    "protocol": "raw centroid-to-centroid angles theta_ij=arccos(u_i.u_j); no caps/thresholds; within-pair prototype sources",
    "prototype_source": {
        "A_and_B": "Market-train under BoT-Market (bot_market_market_train.npz, 751 IDs)",
        "C_and_D": "MSMT17-train under BoT-MSMT17 (bot_msmt_msmt_train.npz, 1041 IDs)",
    },
    "cache": "e12_feature_cache + e18_train_cache",
    "seed": "RNG=42",
    "percentiles_reported": PCTILES,
    **results,
}
out_path = RESULTS / "e20c_v2_centroid_angles.json"
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)
print(f"\nSaved → {out_path}", flush=True)
