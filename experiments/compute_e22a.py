"""
E22a — Cross-model representational similarity analysis (RSA).

For each eval set, build per-ID centroid cosine similarity matrix under both models
that see that gallery, then compute Spearman rho between upper triangles.

  Market gallery (750 test IDs):  S_A (BoT-Market) vs S_C (BoT-MSMT17)
  MSMT17 gallery (3059 test IDs): S_D (BoT-MSMT17) vs S_B (BoT-Market)

Null: permute rows/cols of one model's similarity matrix (200 reps, RNG=42).
      This preserves marginal distributional properties while destroying
      inter-model correspondence. (Random rotation of L2-normalised features
      preserves all cosines, so permutation is the meaningful null here.)

Per-image level: subsample 2000 gallery images per eval set (RNG=42),
      compute pairwise cosine matrix, report Spearman.

Cache: /tmp/e12_feature_cache/
Seed: RNG=42
N_Q_MKT=3368, N_Q_MSMT=11655
"""
import numpy as np
import json
from pathlib import Path
from scipy.stats import spearmanr

CACHE_DIR = Path("/tmp/e12_feature_cache")
RESULTS   = Path("/data/papers/low_dimensional_reid/results")
N_Q_MKT   = 3368
N_Q_MSMT  = 11655
RNG       = np.random.default_rng(42)
N_NULL    = 200
N_IMG_SUB = 2000   # per-image subsample size

def load_gallery(fname, n_q):
    d = np.load(CACHE_DIR / fname)
    g_emb  = d["emb"][n_q:].astype(np.float32)
    g_pids = d["pids"][n_q:]
    return g_emb, g_pids

def unit_centroids(emb, pids):
    unique_ids = np.unique(pids)
    n_ids, D = len(unique_ids), emb.shape[1]
    id_to_idx = {uid: i for i, uid in enumerate(unique_ids)}
    cents  = np.zeros((n_ids, D), dtype=np.float64)
    counts = np.zeros(n_ids)
    for x, pid in zip(emb, pids):
        idx = id_to_idx[pid]
        cents[idx]  += x.astype(np.float64)
        counts[idx] += 1
    cents /= counts[:, None]
    norms = np.linalg.norm(cents, axis=1, keepdims=True)
    return (cents / np.maximum(norms, 1e-9)).astype(np.float32), unique_ids

def cosine_sim_matrix(cents):
    """Symmetric cosine similarity matrix for unit vectors."""
    S = (cents.astype(np.float64) @ cents.astype(np.float64).T).astype(np.float32)
    np.clip(S, -1.0, 1.0, out=S)
    return S

def upper_tri(S):
    n = S.shape[0]
    idx = np.triu_indices(n, k=1)
    return S[idx]

def permutation_null(S_A, S_C_ut, n_null, rng):
    """Permute rows/cols of S_A, compute Spearman with S_C_ut. n_null reps."""
    n = S_A.shape[0]
    nulls = []
    for _ in range(n_null):
        perm = rng.permutation(n)
        S_A_perm = S_A[np.ix_(perm, perm)]
        S_A_perm_ut = upper_tri(S_A_perm)
        nulls.append(float(spearmanr(S_A_perm_ut, S_C_ut).statistic))
    return np.array(nulls)

def rsa_pair(label_A, S_A, label_C, S_C, gallery_label, rng, n_null=N_NULL):
    n = S_A.shape[0]
    assert S_C.shape[0] == n
    S_A_ut = upper_tri(S_A)
    S_C_ut = upper_tri(S_C)
    rho = float(spearmanr(S_A_ut, S_C_ut).statistic)
    print(f"  {gallery_label}: Spearman({label_A}, {label_C}) = {rho:.4f}  n_ids={n}  n_pairs={len(S_A_ut)}", flush=True)

    print(f"  {gallery_label}: computing permutation null ({n_null} reps)...", flush=True)
    null_dist = permutation_null(S_A, S_C_ut, n_null, rng)
    null_mean = float(null_dist.mean())
    null_std  = float(null_dist.std())
    null_p95  = float(np.percentile(null_dist, 95))
    z_score   = (rho - null_mean) / (null_std + 1e-10)
    print(f"  {gallery_label}: null mean={null_mean:.4f} std={null_std:.4f} p95={null_p95:.4f}  z={z_score:.1f}", flush=True)

    return {
        "label_A": label_A, "label_C": label_C,
        "n_ids": n, "n_pairs": int(len(S_A_ut)),
        "spearman_rho": rho,
        "null_mean": null_mean, "null_std": null_std, "null_p95": null_p95,
        "z_score_vs_null": float(z_score),
        "null_n_reps": n_null,
        "null_note": "permutation of rows/cols of S_A; preserves marginal distribution, destroys inter-model correspondence",
    }

def per_image_rsa(emb_A, pids_A, emb_C, pids_C, label_A, label_C, gallery_label, rng, n_sub=N_IMG_SUB):
    """Subsample n_sub images from gallery, compute pairwise cosine Spearman."""
    assert np.array_equal(pids_A, pids_C), "Identity order must match"
    N = len(pids_A)
    n_sub = min(n_sub, N)
    idx = rng.choice(N, n_sub, replace=False)
    eA = emb_A[idx].astype(np.float64)
    eC = emb_C[idx].astype(np.float64)
    # Cosine similarity matrices
    SA = np.clip(eA @ eA.T, -1.0, 1.0).astype(np.float32)
    SC = np.clip(eC @ eC.T, -1.0, 1.0).astype(np.float32)
    SA_ut = upper_tri(SA)
    SC_ut = upper_tri(SC)
    rho = float(spearmanr(SA_ut, SC_ut).statistic)
    print(f"  {gallery_label} per-image (n={n_sub}): Spearman = {rho:.4f}", flush=True)
    return {
        "label_A": label_A, "label_C": label_C,
        "n_subsample": n_sub, "n_pairs": int(len(SA_ut)),
        "spearman_rho": rho,
        "subsample_seed": 42,
    }

# ── Load all four gallery embeddings ──────────────────────────────────────────
print("Loading gallery embeddings...", flush=True)
A_emb, A_pids = load_gallery("market_test.npz",            N_Q_MKT)
B_emb, B_pids = load_gallery("msmt17_test.npz",            N_Q_MSMT)
C_emb, C_pids = load_gallery("bot_msmt_seed42_market.npz", N_Q_MKT)
D_emb, D_pids = load_gallery("bot_msmt_seed42_msmt.npz",   N_Q_MSMT)
print(f"  A: {len(A_pids)} imgs  B: {len(B_pids)} imgs  C: {len(C_pids)} imgs  D: {len(D_pids)} imgs", flush=True)

# Verify identity alignment within each eval set
assert np.array_equal(A_pids, C_pids), "Market gallery IDs must match across models"
assert np.array_equal(B_pids, D_pids), "MSMT17 gallery IDs must match across models"

# ── Compute per-ID unit centroids ─────────────────────────────────────────────
print("\nComputing unit centroids...", flush=True)
uc_A, ids_A = unit_centroids(A_emb, A_pids)
uc_B, ids_B = unit_centroids(B_emb, B_pids)
uc_C, ids_C = unit_centroids(C_emb, C_pids)
uc_D, ids_D = unit_centroids(D_emb, D_pids)
print(f"  A: {len(uc_A)} centroids  B: {len(uc_B)}  C: {len(uc_C)}  D: {len(uc_D)}", flush=True)

# Verify centroid count alignment
assert len(uc_A) == len(uc_C) and np.array_equal(ids_A, ids_C)
assert len(uc_B) == len(uc_D) and np.array_equal(ids_B, ids_D)

# ── Build cosine similarity matrices ──────────────────────────────────────────
print("\nBuilding cosine similarity matrices...", flush=True)
S_A = cosine_sim_matrix(uc_A); print("  S_A done", flush=True)
S_C = cosine_sim_matrix(uc_C); print("  S_C done", flush=True)
S_B = cosine_sim_matrix(uc_B); print("  S_B done", flush=True)
S_D = cosine_sim_matrix(uc_D); print("  S_D done", flush=True)

results = {}

# ── Market eval RSA: S_A vs S_C ───────────────────────────────────────────────
print("\n=== Market eval: S_A (BoT-Market) vs S_C (BoT-MSMT17) ===", flush=True)
results["market_centroid"] = rsa_pair("A (BoT-Market)", S_A, "C (BoT-MSMT17)", S_C,
                                       "Market gallery", RNG)

# ── MSMT17 eval RSA: S_D vs S_B ───────────────────────────────────────────────
print("\n=== MSMT17 eval: S_D (BoT-MSMT17) vs S_B (BoT-Market) ===", flush=True)
results["msmt17_centroid"] = rsa_pair("D (BoT-MSMT17)", S_D, "B (BoT-Market)", S_B,
                                       "MSMT17 gallery", RNG)

# ── Per-image RSA (subsample) ──────────────────────────────────────────────────
print("\n=== Per-image RSA (subsample) ===", flush=True)
results["market_per_image"] = per_image_rsa(
    A_emb, A_pids, C_emb, C_pids,
    "A (BoT-Market)", "C (BoT-MSMT17)", "Market gallery", RNG)
results["msmt17_per_image"] = per_image_rsa(
    D_emb, D_pids, B_emb, B_pids,
    "D (BoT-MSMT17)", "B (BoT-Market)", "MSMT17 gallery", RNG)

# ── Save ──────────────────────────────────────────────────────────────────────
out = {
    "protocol": (
        "RSA: Spearman rho between off-diagonal entries of per-ID centroid cosine "
        "similarity matrices, within eval set (never across). "
        "Null: permutation of rows/cols of one model's centroid matrix (200 reps, RNG=42). "
        "Per-image: pairwise cosine matrix on 2000-image subsample (RNG=42)."
    ),
    "cache": "e12_feature_cache",
    "seed": "RNG=42",
    "n_null_reps": N_NULL,
    "n_img_subsample": N_IMG_SUB,
    **results,
}
out_path = RESULTS / "e22a_rsa_results.json"
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)
print(f"\nSaved → {out_path}", flush=True)
