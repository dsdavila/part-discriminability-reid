"""
E17e micro CIs with half-A/half-B k selection.
Cell B (BoT-Market on MSMT17, projection-out):
1. Split MSMT17 query IDs into half A (k selection) and half B (evaluation), by person ID.
2. On half A: sweep k=8,16,32,64,128,200,256; select best k.
3. Evaluate best k on half B; bootstrap micro mAP CIs.
4. Report plateau: mean delta across k=8..256 (7 points) with per-k bootstrap CIs on full set.

Cache: bot_market / msmt17_test.npz
Projection-out: top-k gallery PCA eigendirections removed from Q and G, then L2-renorm.
"""
import numpy as np, json
from pathlib import Path

import sys
sys.path.insert(0, "/workspace/reid-tk")
import reid_tk

CACHE_DIR = Path("/tmp/e12_feature_cache")
RESULTS_DIR = Path("/data/papers/low_dimensional_reid/results")
N_QUERY_MSMT = 11655
RNG = np.random.default_rng(42)
N_BOOT = 1000
K_SWEEP = [8, 16, 32, 64, 128, 200, 256]

def project_out(q_emb, g_emb, k):
    _, _, Vt = np.linalg.svd(g_emb, full_matrices=False)
    dirs = Vt[:k].T  # (2048, k)
    def rm(X):
        X2 = X - (X @ dirs) @ dirs.T
        return X2 / (np.linalg.norm(X2, axis=1, keepdims=True) + 1e-10)
    return rm(q_emb), rm(g_emb)

def compute_per_query_ap(q_emb, q_pids, q_cids, g_emb, g_pids, g_cids):
    """Return numpy array of per-query APs (same logic as compute_map_rank1)."""
    distances = 1 - q_emb @ g_emb.T
    aps = []
    for q_idx in range(len(q_pids)):
        q_id = q_pids[q_idx]
        q_cam = q_cids[q_idx]
        sorted_indices = np.argsort(distances[q_idx])
        valid = g_cids[sorted_indices] != q_cam
        sorted_ids = g_pids[sorted_indices[valid]]
        matches = sorted_ids == q_id
        if not matches.any():
            aps.append(0.0)
            continue
        match_positions = np.where(matches)[0]
        precisions = [(i + 1) / (pos + 1) for i, pos in enumerate(match_positions)]
        aps.append(float(np.mean(precisions)))
    return np.array(aps)

def micro_map(q_emb, q_pids, q_cids, g_emb, g_pids, g_cids):
    map_val, _ = reid_tk.compute_map_rank1(q_emb, g_emb, q_pids, g_pids, q_cids, g_cids)
    return float(map_val * 100)

print("Loading Cell B (bot_market / msmt17_test.npz)...", flush=True)
d = np.load(CACHE_DIR / "msmt17_test.npz")
q_emb = d["emb"][:N_QUERY_MSMT]
q_pids = d["pids"][:N_QUERY_MSMT]
q_cids = d["cids"][:N_QUERY_MSMT]
g_emb = d["emb"][N_QUERY_MSMT:]
g_pids = d["pids"][N_QUERY_MSMT:]
g_cids = d["cids"][N_QUERY_MSMT:]
print(f"  queries: {len(q_pids)}, gallery: {len(g_pids)}", flush=True)

# Split query IDs by person ID
unique_pids = np.unique(q_pids)
n_pids = len(unique_pids)
perm = RNG.permutation(n_pids)
half_A_set = set(unique_pids[perm[:n_pids//2]])
half_B_set = set(unique_pids[perm[n_pids//2:]])
mask_A = np.array([p in half_A_set for p in q_pids])
mask_B = np.array([p in half_B_set for p in q_pids])
print(f"  half A: {mask_A.sum()} queries ({len(half_A_set)} IDs)", flush=True)
print(f"  half B: {mask_B.sum()} queries ({len(half_B_set)} IDs)", flush=True)

q_A_emb, q_A_pids, q_A_cids = q_emb[mask_A], q_pids[mask_A], q_cids[mask_A]
q_B_emb, q_B_pids, q_B_cids = q_emb[mask_B], q_pids[mask_B], q_cids[mask_B]

# Baseline
baseline_full = micro_map(q_emb, q_pids, q_cids, g_emb, g_pids, g_cids)
baseline_A = micro_map(q_A_emb, q_A_pids, q_A_cids, g_emb, g_pids, g_cids)
baseline_B = micro_map(q_B_emb, q_B_pids, q_B_cids, g_emb, g_pids, g_cids)
print(f"\nBaseline: full={baseline_full:.4f}%  halfA={baseline_A:.4f}%  halfB={baseline_B:.4f}%", flush=True)

# Half-A k selection
print("\n=== HALF-A: k selection (baseline = halfA baseline) ===", flush=True)
halfA_results = {}
for k in K_SWEEP:
    q_proj, g_proj = project_out(q_A_emb, g_emb, k)
    map_k = micro_map(q_proj, q_A_pids, q_A_cids, g_proj, g_pids, g_cids)
    delta = map_k - baseline_A
    halfA_results[k] = {"map": map_k, "delta": delta}
    print(f"  k={k:4d}: mAP={map_k:.4f}%  delta={delta:+.4f}pp", flush=True)

best_k = max(halfA_results, key=lambda k: halfA_results[k]["map"])
print(f"\n  Selected k={best_k} from half A", flush=True)

# Half-B evaluation at best_k
print(f"\n=== HALF-B: evaluation at k={best_k} ===", flush=True)
q_B_proj, g_B_proj = project_out(q_B_emb, g_emb, best_k)
map_B_best = micro_map(q_B_proj, q_B_pids, q_B_cids, g_B_proj, g_pids, g_cids)
delta_B_best = map_B_best - baseline_B
print(f"  Half-B mAP at k={best_k}: {map_B_best:.4f}%  delta={delta_B_best:+.4f}pp", flush=True)

# Bootstrap CIs on half B delta
print(f"  Computing bootstrap CIs ({N_BOOT} samples)...", flush=True)
aps_base_B = compute_per_query_ap(q_B_emb, q_B_pids, q_B_cids, g_emb, g_pids, g_cids)
aps_proj_B = compute_per_query_ap(q_B_proj, q_B_pids, q_B_cids, g_B_proj, g_pids, g_cids)
boot_deltas = []
for b in range(N_BOOT):
    idx = RNG.integers(0, len(q_B_pids), size=len(q_B_pids))
    d_val = np.mean(aps_proj_B[idx]) - np.mean(aps_base_B[idx])
    boot_deltas.append(float(d_val * 100))
    if b % 200 == 0:
        print(f"    bootstrap {b}/{N_BOOT}", flush=True)
boot_deltas = np.array(boot_deltas)
ci_lo, ci_hi = float(np.percentile(boot_deltas, 2.5)), float(np.percentile(boot_deltas, 97.5))
print(f"  Bootstrap 95% CI on delta: [{ci_lo:+.4f}, {ci_hi:+.4f}]pp", flush=True)

# Full plateau sweep (full query set, all k)
print("\n=== FULL PLATEAU (full query set, per-k bootstrap CIs) ===", flush=True)
plateau_results = {}
aps_base_full = compute_per_query_ap(q_emb, q_pids, q_cids, g_emb, g_pids, g_cids)
print(f"  Full baseline APs computed, mean={np.mean(aps_base_full)*100:.4f}%", flush=True)

for k in K_SWEEP:
    q_proj, g_proj = project_out(q_emb, g_emb, k)
    map_k = micro_map(q_proj, q_pids, q_cids, g_proj, g_pids, g_cids)
    delta = map_k - baseline_full
    # Per-k bootstrap CI
    aps_proj_k = compute_per_query_ap(q_proj, q_pids, q_cids, g_proj, g_pids, g_cids)
    boot_k = []
    for b in range(N_BOOT):
        idx = RNG.integers(0, len(q_pids), size=len(q_pids))
        boot_k.append(float((np.mean(aps_proj_k[idx]) - np.mean(aps_base_full[idx])) * 100))
    ci_lo_k = float(np.percentile(boot_k, 2.5))
    ci_hi_k = float(np.percentile(boot_k, 97.5))
    print(f"  k={k:4d}: delta={delta:+.4f}pp  95%CI=[{ci_lo_k:+.4f}, {ci_hi_k:+.4f}]", flush=True)
    plateau_results[k] = {"map": map_k, "delta": delta, "ci_lo": ci_lo_k, "ci_hi": ci_hi_k}

plateau_deltas = [plateau_results[k]["delta"] for k in K_SWEEP]
print(f"\n  Mean plateau delta (k=8..256): {np.mean(plateau_deltas):+.4f}pp", flush=True)

out = {
    "protocol": "bot_market / msmt17_test.npz / projection-out top-k gallery eigendirections",
    "baseline_full": baseline_full,
    "baseline_half_A": baseline_A,
    "baseline_half_B": baseline_B,
    "half_A_k_selection": {str(k): v for k, v in halfA_results.items()},
    "best_k_from_half_A": best_k,
    "half_B_evaluation": {
        "k": best_k,
        "map": map_B_best,
        "delta": delta_B_best,
        "bootstrap_ci_95": [ci_lo, ci_hi],
    },
    "plateau_full_query_set": {str(k): v for k, v in plateau_results.items()},
    "plateau_mean_delta": float(np.mean(plateau_deltas)),
}
out_path = RESULTS_DIR / "e17e_micro_cis.json"
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)
print(f"\nSaved → {out_path}", flush=True)
