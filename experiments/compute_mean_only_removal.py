"""
Mean-only removal decomposition for E17e.
Project out the gallery mean direction alone: u = mu/|mu|, then X' = X - (X@u)u, L2-renorm.
Compare with:
  - v3 (mean-centered PCA plateau): +1.4256pp
  - v2 (raw SVD plateau): +1.800pp
  - gap: +0.374pp
"""
import numpy as np, json
from pathlib import Path

CACHE_DIR = Path("/tmp/e12_feature_cache")
RESULTS_DIR = Path("/data/papers/low_dimensional_reid/results")
N_QUERY_MSMT = 11655
RNG = np.random.default_rng(42)
N_BOOT = 1000

def compute_per_query_ap_canonical(q_emb, q_pids, q_cids, g_emb, g_pids, g_cids, batch=512):
    n_q = len(q_pids)
    aps = []
    for start in range(0, n_q, batch):
        end = min(start + batch, n_q)
        sim = q_emb[start:end].astype(np.float32) @ g_emb.T
        order = np.argsort(-sim, axis=1)
        for qi, idx in enumerate(order):
            qpid, qcid = q_pids[start+qi], q_cids[start+qi]
            gp, gc = g_pids[idx], g_cids[idx]
            mask_valid = ~((gp == qpid) & (gc == qcid))
            mask_pos = (gp == qpid) & mask_valid
            if mask_pos.sum() == 0:
                aps.append(0.0)
                continue
            valid_idx = np.where(mask_valid)[0]
            pos_in_valid = np.where(mask_pos[valid_idx])[0]
            prec = np.arange(1, len(pos_in_valid)+1) / (pos_in_valid+1)
            aps.append(float(prec.mean()))
    return np.array(aps)

def micro_map(aps):
    return float(np.mean(aps) * 100)

def project_out_mean(q_emb, g_emb):
    """Project out the mean direction of the gallery: u = mu/|mu|."""
    mu = g_emb.astype(np.float64).mean(0)
    u = mu / np.linalg.norm(mu)  # unit mean direction
    def rm(X):
        X64 = X.astype(np.float64)
        residual = X64 - np.outer(X64 @ u, u)
        norms = np.linalg.norm(residual, axis=1, keepdims=True)
        return (residual / np.maximum(norms, 1e-10)).astype(np.float32)
    return rm(q_emb), rm(g_emb), u

print("Loading Cell B (bot_market / msmt17_test.npz)...", flush=True)
d = np.load(CACHE_DIR / "msmt17_test.npz")
q_emb = d["emb"][:N_QUERY_MSMT].astype(np.float32)
q_pids, q_cids = d["pids"][:N_QUERY_MSMT], d["cids"][:N_QUERY_MSMT]
g_emb = d["emb"][N_QUERY_MSMT:].astype(np.float32)
g_pids, g_cids = d["pids"][N_QUERY_MSMT:], d["cids"][N_QUERY_MSMT:]
print(f"  queries: {len(q_pids)}, gallery: {len(g_pids)}", flush=True)

print("\nComputing baseline...", flush=True)
aps_base = compute_per_query_ap_canonical(q_emb, q_pids, q_cids, g_emb, g_pids, g_cids)
baseline = micro_map(aps_base)
print(f"  baseline={baseline:.4f}% (canonical should be 2.561%: {'OK' if abs(baseline-2.561)<0.01 else 'MISMATCH'})", flush=True)

print("\nComputing mean-only removal...", flush=True)
q_proj, g_proj, u = project_out_mean(q_emb, g_emb)
aps_mean = compute_per_query_ap_canonical(q_proj, q_pids, q_cids, g_proj, g_pids, g_cids)
map_mean = micro_map(aps_mean)
delta_mean = map_mean - baseline
print(f"  mAP={map_mean:.4f}%  delta={delta_mean:+.4f}pp", flush=True)

# Bootstrap CI
print(f"  Bootstrapping {N_BOOT} samples...", flush=True)
boot_deltas = []
for b in range(N_BOOT):
    idx = RNG.integers(0, len(q_pids), size=len(q_pids))
    boot_deltas.append(float((np.mean(aps_mean[idx]) - np.mean(aps_base[idx])) * 100))
    if b % 200 == 0:
        print(f"    boot {b}/{N_BOOT}", flush=True)
ci_lo = float(np.percentile(boot_deltas, 2.5))
ci_hi = float(np.percentile(boot_deltas, 97.5))
print(f"  95% CI: [{ci_lo:+.4f}, {ci_hi:+.4f}]pp", flush=True)

# Decomposition summary
v3_plateau = 1.4256
v2_plateau = 1.800
gap = v2_plateau - v3_plateau
print(f"\n=== DECOMPOSITION ===", flush=True)
print(f"  raw SVD (v2) plateau:       +{v2_plateau:.4f}pp", flush=True)
print(f"  mean-centered PCA (v3):     +{v3_plateau:.4f}pp", flush=True)
print(f"  gap (v2 - v3):              +{gap:.4f}pp", flush=True)
print(f"  mean-only removal delta:    {delta_mean:+.4f}pp", flush=True)
print(f"  mean explains gap? {delta_mean:.3f}/{gap:.3f} = {delta_mean/gap*100:.0f}%", flush=True)

out = {
    "protocol": "canonical junk=(g_pid==q_pid)&(g_cam==q_cam); mean-only removal: u=mu/|mu|, X'=X-(X@u)u, L2-renorm",
    "cache": "bot_market / msmt17_test.npz",
    "baseline_full": baseline,
    "mean_only_removal": {
        "map": map_mean,
        "delta": delta_mean,
        "bootstrap_ci_95": [ci_lo, ci_hi]
    },
    "decomposition": {
        "raw_svd_v2_plateau": v2_plateau,
        "mean_centered_pca_v3_plateau": v3_plateau,
        "gap_v2_minus_v3": gap,
        "mean_only_delta": delta_mean,
        "mean_explains_gap_fraction": delta_mean / gap if gap > 0 else None
    }
}
with open(RESULTS_DIR / "e17e_mean_only_decomposition.json", "w") as f:
    import json
    json.dump(out, f, indent=2)
print(f"\nSaved → {RESULTS_DIR / 'e17e_mean_only_decomposition.json'}", flush=True)
