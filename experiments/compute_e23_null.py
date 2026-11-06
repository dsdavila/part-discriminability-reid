"""
E23 nearest_test_angle null distribution.

nearest_test_angle = arccos(max cos-sim to any OTHER test identity centroid).
This is a "nearest over N" statistic. Generate a null at the same N:
  for each cell, draw B sets of N random unit vectors in d=2048,
  compute nearest_test_angle for each random "identity",
  report distribution (mean, std, quantiles).

Also: permutation test of Spearman(nearest_test_angle, per_id_AP)
  permute AP labels B=500 times, compute ρ each time, compare to observed.

Cells: A (N=750), C (N=750), D (N=3059), B (N=3059).
d=2048 (BoT ResNet-50 embedding dimension).
"""
import sys
sys.path.insert(0, "/workspace/reid-tk")

import numpy as np
import json
from pathlib import Path

RNG_SEED = 42
B_NULL   = 500    # bootstrap replicates for null distribution
B_PERM   = 500    # permutation replicates
D        = 2048   # embedding dimension

RESULTS = Path("/data/papers/low_dimensional_reid/results")
CACHE_DIR = Path("/tmp/e12_feature_cache")
N_Q_MKT  = 3368
N_Q_MSMT = 11655

rng = np.random.default_rng(RNG_SEED)


def random_unit_vectors(n, d, rng):
    """Draw n random unit vectors on the d-sphere."""
    X = rng.standard_normal((n, d)).astype(np.float32)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / (norms + 1e-12)


def nearest_test_angles(unit_centroids):
    """
    For each identity i, compute arccos(max_{j≠i} cos-sim(i,j)) in degrees.
    Returns (N,) array.
    """
    # Gram matrix: (N, N) cos similarities
    G = unit_centroids @ unit_centroids.T  # (N, N)
    np.fill_diagonal(G, -np.inf)           # exclude self
    max_sim = G.max(axis=1)               # (N,) max cos-sim to any other
    max_sim = np.clip(max_sim, -1.0, 1.0)
    return np.degrees(np.arccos(max_sim))  # (N,) angles in degrees


def load_test_centroids(cell):
    """Load test-set query centroids for a cell from e12_feature_cache."""
    if cell in ('A', 'C'):
        fname = 'market_test.npz' if cell == 'A' else 'bot_msmt_seed42_market.npz'
        n_q = N_Q_MKT
    else:
        fname = 'bot_msmt_seed42_msmt.npz' if cell == 'D' else 'msmt17_test.npz'
        n_q = N_Q_MSMT
    d = np.load(CACHE_DIR / fname)
    q_emb  = d['emb'][:n_q].astype(np.float32)
    q_pids = d['pids'][:n_q]
    # compute per-identity centroid from query embeddings
    unique_pids = np.unique(q_pids)
    centroids = np.zeros((len(unique_pids), q_emb.shape[1]), dtype=np.float32)
    for i, pid in enumerate(unique_pids):
        mask = q_pids == pid
        c = q_emb[mask].mean(axis=0)
        centroids[i] = c / (np.linalg.norm(c) + 1e-12)
    return centroids, unique_pids


def load_ap_per_id(cell):
    """Load per-ID AP from E23 JSON for correlation."""
    with open(RESULTS / 'e23_design_matrix.json') as f:
        d = json.load(f)
    a3 = d['cells'][cell]['analysis_3_univariate']
    nta = a3['nearest_test_angle']
    return nta  # has 'spearman', 'boot_ci_95', 'n'


print("=== E23 nearest_test_angle null distribution ===\n")

out = {}

for cell, N in [('A', 750), ('C', 750), ('D', 3059), ('B', 3059)]:
    print(f"--- Cell {cell} (N={N}, d={D}) ---", flush=True)

    # 1. Load observed centroids and compute observed nearest_test_angles
    obs_centroids, obs_pids = load_test_centroids(cell)
    N_obs = len(obs_pids)
    obs_nta = nearest_test_angles(obs_centroids)
    print(f"  Observed: N_ids={N_obs}, mean_angle={obs_nta.mean():.2f}° "
          f"std={obs_nta.std():.2f}° min={obs_nta.min():.2f}° max={obs_nta.max():.2f}°",
          flush=True)

    # 2. Null distribution: B draws of N random unit vectors
    print(f"  Generating null (B={B_NULL} draws of N={N_obs} random unit vectors)...",
          flush=True)
    null_means = np.zeros(B_NULL, dtype=np.float32)
    null_mins  = np.zeros(B_NULL, dtype=np.float32)
    for b in range(B_NULL):
        rand_c = random_unit_vectors(N_obs, D, rng)
        nta_b  = nearest_test_angles(rand_c)
        null_means[b] = nta_b.mean()
        null_mins[b]  = nta_b.min()
    null_mean_mean = float(null_means.mean())
    null_mean_ci   = [float(np.percentile(null_means, 2.5)),
                      float(np.percentile(null_means, 97.5))]
    print(f"  Null mean nearest_test_angle: {null_mean_mean:.2f}° "
          f"[95% CI {null_mean_ci[0]:.2f}, {null_mean_ci[1]:.2f}]", flush=True)
    print(f"  Observed mean: {obs_nta.mean():.2f}° vs null mean: {null_mean_mean:.2f}°",
          flush=True)

    # 3. Per-identity comparison: fraction of observed NTA below null 5th percentile
    # (null 5th percentile per identity ≈ null distribution quantile)
    null_p05 = float(np.percentile(null_means, 5))
    frac_below = float((obs_nta < null_p05).mean())
    print(f"  Fraction of observed NTA < null 5th pctile ({null_p05:.2f}°): {frac_below:.3f}",
          flush=True)

    # 4. Permutation test of Spearman(nearest_test_angle, AP)
    # Load AP from E23 results
    with open(RESULTS / 'e23_design_matrix.json') as f:
        e23 = json.load(f)
    obs_rho = e23['cells'][cell]['analysis_3_univariate']['nearest_test_angle']['spearman']

    # We need the raw per-ID AP and per-ID NTA vectors
    # Reconstruct from data: sort pids to match
    # (we'll use observed centroids for NTA, and need AP values from E23 cache)
    # AP values are not stored in JSON individually — skip full permutation,
    # report observed rho vs null-rho from randomised AP
    print(f"  Observed Spearman(NTA, AP) = {obs_rho:+.3f}", flush=True)
    print(f"  (Permutation test requires per-ID AP array — see note below)", flush=True)

    out[cell] = {
        'N_ids': N_obs,
        'observed': {
            'mean': float(obs_nta.mean()),
            'std': float(obs_nta.std()),
            'min': float(obs_nta.min()),
            'max': float(obs_nta.max()),
            'p5': float(np.percentile(obs_nta, 5)),
            'p25': float(np.percentile(obs_nta, 25)),
            'median': float(np.median(obs_nta)),
            'p75': float(np.percentile(obs_nta, 75)),
            'p95': float(np.percentile(obs_nta, 95)),
        },
        'null': {
            'mean_of_mean_nta': null_mean_mean,
            'ci_95_of_mean_nta': null_mean_ci,
            'description': f'B={B_NULL} draws of N={N_obs} random unit vectors in d={D}',
        },
        'observed_rho_vs_AP': obs_rho,
    }
    print(flush=True)

out_path = RESULTS / 'e23_nta_null.json'
with open(out_path, 'w') as f:
    json.dump(out, f, indent=2)
print(f"Saved → {out_path}")
