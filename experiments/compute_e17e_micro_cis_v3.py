"""
E17e micro CIs v3 — canonical protocol + PCA project_out matching compute_projout_control.py.
Protocol: junk=(g_pid==q_pid)&(g_cam==q_cam) [same as compute_2x2_map.py]
PCA: mean-center gallery before SVD [same as compute_projout_control.py line 85-87]
"""
import numpy as np, json
from pathlib import Path

CACHE_DIR = Path("/tmp/e12_feature_cache")
RESULTS_DIR = Path("/data/papers/low_dimensional_reid/results")
N_QUERY_MSMT = 11655
RNG = np.random.default_rng(42)
N_BOOT = 1000
K_SWEEP = [8, 16, 32, 64, 128, 200, 256]

def project_out_pca(q_emb, g_emb, k):
    """Mean-center gallery, take top-k PCA directions, project out. Matches compute_projout_control.py."""
    Xc = g_emb.astype(np.float64) - g_emb.mean(0)
    _, _, Vt = np.linalg.svd(Xc / np.sqrt(len(g_emb) - 1), full_matrices=False)
    V_k = Vt[:k].T  # (2048, k)
    def rm(X):
        X64 = X.astype(np.float64)
        residual = X64 - (X64 @ V_k) @ V_k.T
        return (residual / np.maximum(np.linalg.norm(residual, axis=1, keepdims=True), 1e-10)).astype(np.float32)
    return rm(q_emb), rm(g_emb)

def compute_per_query_ap_canonical(q_emb, q_pids, q_cids, g_emb, g_pids, g_cids, batch=512):
    """Canonical predicate: junk = (g_pid==q_pid) & (g_cam==q_cam). Same as compute_2x2_map.py."""
    n_q = len(q_pids)
    aps = []
    for start in range(0, n_q, batch):
        end = min(start + batch, n_q)
        sim = q_emb[start:end].astype(np.float32) @ g_emb.T
        order = np.argsort(-sim, axis=1)
        for qi, idx in enumerate(order):
            qpid, qcid = q_pids[start + qi], q_cids[start + qi]
            gp, gc = g_pids[idx], g_cids[idx]
            mask_valid = ~((gp == qpid) & (gc == qcid))
            mask_pos = (gp == qpid) & mask_valid
            if mask_pos.sum() == 0:
                aps.append(0.0)
                continue
            valid_idx = np.where(mask_valid)[0]
            pos_in_valid = np.where(mask_pos[valid_idx])[0]
            prec = np.arange(1, len(pos_in_valid) + 1) / (pos_in_valid + 1)
            aps.append(float(prec.mean()))
    return np.array(aps)

def micro_map(aps):
    return float(np.mean(aps) * 100)

print("Loading Cell B (bot_market / msmt17_test.npz)...", flush=True)
d = np.load(CACHE_DIR / "msmt17_test.npz")
q_emb = d["emb"][:N_QUERY_MSMT].astype(np.float32)
q_pids, q_cids = d["pids"][:N_QUERY_MSMT], d["cids"][:N_QUERY_MSMT]
g_emb = d["emb"][N_QUERY_MSMT:].astype(np.float32)
g_pids, g_cids = d["pids"][N_QUERY_MSMT:], d["cids"][N_QUERY_MSMT:]
print(f"  queries: {len(q_pids)}, gallery: {len(g_pids)}", flush=True)

# Split by person ID
unique_pids = np.unique(q_pids)
perm = RNG.permutation(len(unique_pids))
half_A_set = set(unique_pids[perm[:len(unique_pids)//2]])
half_B_set = set(unique_pids[perm[len(unique_pids)//2:]])
mask_A = np.array([p in half_A_set for p in q_pids])
mask_B = np.array([p in half_B_set for p in q_pids])
q_A_emb, q_A_pids, q_A_cids = q_emb[mask_A], q_pids[mask_A], q_cids[mask_A]
q_B_emb, q_B_pids, q_B_cids = q_emb[mask_B], q_pids[mask_B], q_cids[mask_B]
print(f"  half A: {mask_A.sum()} queries ({len(half_A_set)} IDs)", flush=True)
print(f"  half B: {mask_B.sum()} queries ({len(half_B_set)} IDs)", flush=True)

print("\nComputing baselines...", flush=True)
aps_base_full = compute_per_query_ap_canonical(q_emb, q_pids, q_cids, g_emb, g_pids, g_cids)
aps_base_A    = compute_per_query_ap_canonical(q_A_emb, q_A_pids, q_A_cids, g_emb, g_pids, g_cids)
aps_base_B    = compute_per_query_ap_canonical(q_B_emb, q_B_pids, q_B_cids, g_emb, g_pids, g_cids)
baseline_full, baseline_A, baseline_B = micro_map(aps_base_full), micro_map(aps_base_A), micro_map(aps_base_B)
print(f"  full={baseline_full:.4f}%  halfA={baseline_A:.4f}%  halfB={baseline_B:.4f}%", flush=True)
print(f"  (canonical should be 2.561%: {'OK' if abs(baseline_full - 2.561) < 0.01 else 'MISMATCH'})", flush=True)

print("\n=== HALF-A: k selection (mean-centered PCA) ===", flush=True)
halfA_results = {}
for k in K_SWEEP:
    q_proj, g_proj = project_out_pca(q_A_emb, g_emb, k)
    aps_k = compute_per_query_ap_canonical(q_proj, q_A_pids, q_A_cids, g_proj, g_pids, g_cids)
    map_k, delta = micro_map(aps_k), micro_map(aps_k) - baseline_A
    halfA_results[k] = {"map": map_k, "delta": delta}
    print(f"  k={k:4d}: mAP={map_k:.4f}%  delta={delta:+.4f}pp", flush=True)

best_k = max(halfA_results, key=lambda k: halfA_results[k]["map"])
print(f"\n  Selected k={best_k}", flush=True)

print(f"\n=== HALF-B: evaluation at k={best_k} ===", flush=True)
q_B_proj, g_B_proj = project_out_pca(q_B_emb, g_emb, best_k)
aps_proj_B = compute_per_query_ap_canonical(q_B_proj, q_B_pids, q_B_cids, g_B_proj, g_pids, g_cids)
map_B_best = micro_map(aps_proj_B)
delta_B_best = map_B_best - baseline_B
print(f"  mAP={map_B_best:.4f}%  delta={delta_B_best:+.4f}pp", flush=True)

print(f"  Bootstrapping {N_BOOT} samples...", flush=True)
boot_deltas = []
for b in range(N_BOOT):
    idx = RNG.integers(0, len(q_B_pids), size=len(q_B_pids))
    boot_deltas.append(float((np.mean(aps_proj_B[idx]) - np.mean(aps_base_B[idx])) * 100))
    if b % 200 == 0:
        print(f"    boot {b}/{N_BOOT}", flush=True)
ci_lo = float(np.percentile(boot_deltas, 2.5))
ci_hi = float(np.percentile(boot_deltas, 97.5))
print(f"  95% CI: [{ci_lo:+.4f}, {ci_hi:+.4f}]pp", flush=True)

print("\n=== FULL PLATEAU ===", flush=True)
plateau_results = {}
for k in K_SWEEP:
    q_proj, g_proj = project_out_pca(q_emb, g_emb, k)
    aps_k = compute_per_query_ap_canonical(q_proj, q_pids, q_cids, g_proj, g_pids, g_cids)
    map_k, delta = micro_map(aps_k), micro_map(aps_k) - baseline_full
    boot_k = []
    for b in range(N_BOOT):
        idx = RNG.integers(0, len(q_pids), size=len(q_pids))
        boot_k.append(float((np.mean(aps_k[idx]) - np.mean(aps_base_full[idx])) * 100))
    ci_lo_k, ci_hi_k = float(np.percentile(boot_k, 2.5)), float(np.percentile(boot_k, 97.5))
    print(f"  k={k:4d}: delta={delta:+.4f}pp  95%CI=[{ci_lo_k:+.4f}, {ci_hi_k:+.4f}]", flush=True)
    plateau_results[k] = {"map": map_k, "delta": delta, "ci_lo": ci_lo_k, "ci_hi": ci_hi_k}

plateau_deltas = [plateau_results[k]["delta"] for k in K_SWEEP]
print(f"\n  Plateau mean delta: {np.mean(plateau_deltas):+.4f}pp", flush=True)

out = {
    "protocol": "canonical junk=(g_pid==q_pid)&(g_cam==q_cam); PCA mean-centered [matches compute_projout_control.py]",
    "cache": "bot_market / msmt17_test.npz",
    "baseline_full": baseline_full, "baseline_half_A": baseline_A, "baseline_half_B": baseline_B,
    "half_A_k_selection": {str(k): v for k, v in halfA_results.items()},
    "best_k_from_half_A": best_k,
    "half_B_evaluation": {"k": best_k, "map": map_B_best, "delta": delta_B_best, "bootstrap_ci_95": [ci_lo, ci_hi]},
    "plateau_full_query_set": {str(k): v for k, v in plateau_results.items()},
    "plateau_mean_delta": float(np.mean(plateau_deltas)),
    "plateau_min_delta": float(min(plateau_deltas)),
    "plateau_max_delta": float(max(plateau_deltas)),
}
with open(RESULTS_DIR / "e17e_micro_cis_final.json", "w") as f:
    json.dump(out, f, indent=2)
print(f"\nSaved → {RESULTS_DIR / 'e17e_micro_cis_final.json'}", flush=True)
