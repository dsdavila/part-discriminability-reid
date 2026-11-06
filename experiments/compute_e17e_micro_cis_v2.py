"""
E17e micro CIs with half-A/half-B k selection — CANONICAL PROTOCOL.
Canonical predicate: junk = (g_pid == q_pid) AND (g_cam == q_cam)
(same as compute_2x2_map.py and compute_projout_control.py)
Cache: bot_market / msmt17_test.npz
"""
import numpy as np, json
from pathlib import Path

CACHE_DIR = Path("/tmp/e12_feature_cache")
RESULTS_DIR = Path("/data/papers/low_dimensional_reid/results")
N_QUERY_MSMT = 11655
RNG = np.random.default_rng(42)
N_BOOT = 1000
K_SWEEP = [8, 16, 32, 64, 128, 200, 256]

def project_out(q_emb, g_emb, k):
    _, _, Vt = np.linalg.svd(g_emb.astype(np.float64), full_matrices=False)
    V_k = Vt[:k].T  # (2048, k)
    def rm(X):
        X64 = X.astype(np.float64)
        X2 = X64 - (X64 @ V_k) @ V_k.T
        return (X2 / np.maximum(np.linalg.norm(X2, axis=1, keepdims=True), 1e-10)).astype(np.float32)
    return rm(q_emb), rm(g_emb)

def compute_per_query_ap_canonical(q_emb, q_pids, q_cids, g_emb, g_pids, g_cids, batch=512):
    """Canonical predicate: exclude same-ID AND same-cam gallery entries only."""
    n_q = len(q_pids)
    aps = []
    for start in range(0, n_q, batch):
        end = min(start + batch, n_q)
        sim = q_emb[start:end].astype(np.float32) @ g_emb.T
        order = np.argsort(-sim, axis=1)
        for qi, idx in enumerate(order):
            qpid = q_pids[start + qi]
            qcid = q_cids[start + qi]
            gp = g_pids[idx]
            gc = g_cids[idx]
            mask_valid = ~((gp == qpid) & (gc == qcid))
            mask_pos = (gp == qpid) & mask_valid
            if mask_pos.sum() == 0:
                aps.append(0.0)
                continue
            valid_idx = np.where(mask_valid)[0]
            pos_in_valid = np.where(mask_pos[valid_idx])[0]
            n_pos = len(pos_in_valid)
            prec = np.arange(1, n_pos + 1) / (pos_in_valid + 1)
            aps.append(float(prec.mean()))
    return np.array(aps)

def micro_map_from_aps(aps):
    return float(np.mean(aps) * 100)

print("Loading Cell B (bot_market / msmt17_test.npz)...", flush=True)
d = np.load(CACHE_DIR / "msmt17_test.npz")
q_emb = d["emb"][:N_QUERY_MSMT].astype(np.float32)
q_pids = d["pids"][:N_QUERY_MSMT]
q_cids = d["cids"][:N_QUERY_MSMT]
g_emb = d["emb"][N_QUERY_MSMT:].astype(np.float32)
g_pids = d["pids"][N_QUERY_MSMT:]
g_cids = d["cids"][N_QUERY_MSMT:]
print(f"  queries: {len(q_pids)}, gallery: {len(g_pids)}", flush=True)

# Split query IDs by person ID (half A for k selection, half B for evaluation)
unique_pids = np.unique(q_pids)
n_pids = len(unique_pids)
perm = RNG.permutation(n_pids)
half_A_set = set(unique_pids[perm[:n_pids//2]])
half_B_set = set(unique_pids[perm[n_pids//2:]])
mask_A = np.array([p in half_A_set for p in q_pids])
mask_B = np.array([p in half_B_set for p in q_pids])
q_A_emb, q_A_pids, q_A_cids = q_emb[mask_A], q_pids[mask_A], q_cids[mask_A]
q_B_emb, q_B_pids, q_B_cids = q_emb[mask_B], q_pids[mask_B], q_cids[mask_B]
print(f"  half A: {mask_A.sum()} queries ({len(half_A_set)} IDs)", flush=True)
print(f"  half B: {mask_B.sum()} queries ({len(half_B_set)} IDs)", flush=True)

# Baseline per-query APs (canonical protocol)
print("\nComputing baselines...", flush=True)
aps_base_full = compute_per_query_ap_canonical(q_emb, q_pids, q_cids, g_emb, g_pids, g_cids)
aps_base_A    = compute_per_query_ap_canonical(q_A_emb, q_A_pids, q_A_cids, g_emb, g_pids, g_cids)
aps_base_B    = compute_per_query_ap_canonical(q_B_emb, q_B_pids, q_B_cids, g_emb, g_pids, g_cids)
baseline_full = micro_map_from_aps(aps_base_full)
baseline_A    = micro_map_from_aps(aps_base_A)
baseline_B    = micro_map_from_aps(aps_base_B)
print(f"  full={baseline_full:.4f}%  halfA={baseline_A:.4f}%  halfB={baseline_B:.4f}%", flush=True)

# Half-A k selection sweep
print("\n=== HALF-A: k selection ===", flush=True)
halfA_results = {}
for k in K_SWEEP:
    q_proj, g_proj = project_out(q_A_emb, g_emb, k)
    aps_k = compute_per_query_ap_canonical(q_proj, q_A_pids, q_A_cids, g_proj, g_pids, g_cids)
    map_k = micro_map_from_aps(aps_k)
    delta = map_k - baseline_A
    halfA_results[k] = {"map": map_k, "delta": delta}
    print(f"  k={k:4d}: mAP={map_k:.4f}%  delta={delta:+.4f}pp", flush=True)

best_k = max(halfA_results, key=lambda k: halfA_results[k]["map"])
print(f"\n  Selected k={best_k}", flush=True)

# Half-B evaluation at best_k with bootstrap CIs
print(f"\n=== HALF-B: evaluation at k={best_k} ===", flush=True)
q_B_proj, g_B_proj = project_out(q_B_emb, g_emb, best_k)
aps_proj_B = compute_per_query_ap_canonical(q_B_proj, q_B_pids, q_B_cids, g_B_proj, g_pids, g_cids)
map_B_best = micro_map_from_aps(aps_proj_B)
delta_B_best = map_B_best - baseline_B
print(f"  mAP={map_B_best:.4f}%  delta={delta_B_best:+.4f}pp", flush=True)

print(f"  Bootstrapping {N_BOOT} samples...", flush=True)
boot_deltas = []
n_B = len(q_B_pids)
for b in range(N_BOOT):
    idx = RNG.integers(0, n_B, size=n_B)
    d_val = (np.mean(aps_proj_B[idx]) - np.mean(aps_base_B[idx])) * 100
    boot_deltas.append(float(d_val))
    if b % 200 == 0:
        print(f"    boot {b}/{N_BOOT}", flush=True)
ci_lo, ci_hi = float(np.percentile(boot_deltas, 2.5)), float(np.percentile(boot_deltas, 97.5))
print(f"  95% CI: [{ci_lo:+.4f}, {ci_hi:+.4f}]pp", flush=True)

# Full plateau with per-k bootstrap CIs
print("\n=== FULL PLATEAU (all 11655 queries) ===", flush=True)
plateau_results = {}
for k in K_SWEEP:
    q_proj, g_proj = project_out(q_emb, g_emb, k)
    aps_k = compute_per_query_ap_canonical(q_proj, q_pids, q_cids, g_proj, g_pids, g_cids)
    map_k = micro_map_from_aps(aps_k)
    delta = map_k - baseline_full
    boot_k = []
    n_full = len(q_pids)
    for b in range(N_BOOT):
        idx = RNG.integers(0, n_full, size=n_full)
        boot_k.append(float((np.mean(aps_k[idx]) - np.mean(aps_base_full[idx])) * 100))
    ci_lo_k, ci_hi_k = float(np.percentile(boot_k, 2.5)), float(np.percentile(boot_k, 97.5))
    print(f"  k={k:4d}: delta={delta:+.4f}pp  95%CI=[{ci_lo_k:+.4f}, {ci_hi_k:+.4f}]", flush=True)
    plateau_results[k] = {"map": map_k, "delta": delta, "ci_lo": ci_lo_k, "ci_hi": ci_hi_k}

plateau_deltas = [plateau_results[k]["delta"] for k in K_SWEEP]
print(f"\n  Plateau mean delta: {np.mean(plateau_deltas):+.4f}pp", flush=True)

out = {
    "protocol": "canonical: junk=(g_pid==q_pid)&(g_cam==q_cam); same as compute_2x2_map.py and compute_projout_control.py",
    "cache": "bot_market / msmt17_test.npz",
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
    "plateau_min_delta": float(min(plateau_deltas)),
    "plateau_max_delta": float(max(plateau_deltas)),
}
out_path = RESULTS_DIR / "e17e_micro_cis_canonical.json"
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)
print(f"\nSaved → {out_path}", flush=True)
