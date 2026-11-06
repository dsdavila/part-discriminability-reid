"""
E17e in-domain control: projection-out mAP for all 4 cells at k=[8,16,32,64,128].
No bootstrapping — just mAP at each k to get the dissociation table.
"""
import numpy as np
import sys
import json
from pathlib import Path

CACHE_DIR = Path("/tmp/e12_feature_cache")
RESULTS_DIR = Path("/data/papers/low_dimensional_reid/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LOG = sys.stdout

KS = [8, 16, 32, 64, 128]

# Cache stores combined query+gallery (query first).
# Market split: 3368 query + 13115 gallery = 16483 total
# MSMT17 split: 11655 query + 82132 gallery = 93787 total
N_QUERY_MARKET = 3368
N_QUERY_MSMT   = 11655

CELLS = {
    "A": ("market_test.npz",         "market", "BoT-Market on Market"),
    "B": ("msmt17_test.npz",         "msmt",   "BoT-Market on MSMT17"),
    "C": ("bot_msmt_seed42_market.npz", "market", "BoT-MSMT on Market"),
    "D": ("bot_msmt_seed42_msmt.npz",   "msmt",   "BoT-MSMT on MSMT17"),
}


def load_cell(fname, dataset):
    """Load combined npz and split into query/gallery."""
    d = np.load(CACHE_DIR / fname)
    emb = d["emb"]
    pids = d["pids"]
    cids = d["cids"]
    n_q = N_QUERY_MARKET if dataset == "market" else N_QUERY_MSMT
    q_emb, q_pids, q_cids = emb[:n_q], pids[:n_q], cids[:n_q]
    g_emb, g_pids, g_cids = emb[n_q:], pids[n_q:], cids[n_q:]
    return q_emb, q_pids, q_cids, g_emb, g_pids, g_cids


def retrieval_map(q_emb, q_pids, q_cids, g_emb, g_pids, g_cids, batch=512):
    n_q = len(q_pids)
    aps = []
    for start in range(0, n_q, batch):
        end = min(start + batch, n_q)
        sim = q_emb[start:end].astype(np.float32) @ g_emb.T
        order = np.argsort(-sim, axis=1)
        for qi, idx in enumerate(order):
            qpid = q_pids[start + qi]; qcid = q_cids[start + qi]
            gp = g_pids[idx]; gc = g_cids[idx]
            mask_valid = ~((gp == qpid) & (gc == qcid))
            mask_pos = (gp == qpid) & mask_valid
            if mask_pos.sum() == 0:
                continue
            valid_idx = np.where(mask_valid)[0]
            pos_in_valid = np.where(mask_pos[valid_idx])[0]
            n_pos = len(pos_in_valid)
            prec = np.arange(1, n_pos + 1) / (pos_in_valid + 1)
            aps.append(prec.mean())
    return float(np.mean(aps) * 100) if aps else 0.0


def project_out(X, V_k):
    """Remove top-k eigendirections from X, L2-renormalize."""
    X64 = X.astype(np.float64)
    proj_onto = X64 @ V_k @ V_k.T
    residual = X64 - proj_onto
    norms = np.linalg.norm(residual, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    return (residual / norms).astype(np.float32)


results = {}
for cell, (fname, dataset, label) in CELLS.items():
    q_emb, q_pids, q_cids, g_emb, g_pids, g_cids = load_cell(fname, dataset)
    print(f"\n=== Cell {cell} — {label} ({len(q_pids)} queries x {len(g_pids)} gallery) ===", flush=True)

    # Baseline
    map_base = retrieval_map(q_emb, q_pids, q_cids, g_emb, g_pids, g_cids)
    print(f"  baseline: mAP={map_base:.3f}%", flush=True)

    # Covariance eigenvectors from gallery
    Xc = g_emb - g_emb.mean(0)
    _, _, Vt = np.linalg.svd(Xc / np.sqrt(len(g_emb) - 1), full_matrices=False)
    V = Vt.T  # columns = eigenvectors

    cell_results = {"baseline": map_base, "ks": {}}
    for k in KS:
        V_k = V[:, :k]
        q_proj = project_out(q_emb, V_k)
        g_proj = project_out(g_emb, V_k)
        map_k = retrieval_map(q_proj, q_pids, q_cids, g_proj, g_pids, g_cids)
        delta = map_k - map_base
        print(f"  k={k:3d}: mAP={map_k:.3f}%  delta={delta:+.3f}pp", flush=True)
        cell_results["ks"][str(k)] = {"map": map_k, "delta": delta}

    results[cell] = cell_results

print("\n\n========================================")
print("DISSOCIATION TABLE (projection-out mAP)")
print("========================================")
print(f"{'':8s} {'base':>8s}", end="")
for k in KS:
    print(f"  k={k:3d}", end="")
print()
for cell in "ABCD":
    r = results[cell]
    print(f"  Cell {cell}  {r['baseline']:>7.3f}%", end="")
    for k in KS:
        print(f"  {r['ks'][str(k)]['map']:>6.3f}%", end="")
    print()

print("\nDeltas:")
print(f"{'':8s} {'':8s}", end="")
for k in KS:
    print(f"  k={k:3d}", end="")
print()
for cell in "ABCD":
    r = results[cell]
    print(f"  Cell {cell}  {'':>8s}", end="")
    for k in KS:
        print(f"  {r['ks'][str(k)]['delta']:>+6.3f}", end="")
    print()

out = RESULTS_DIR / "projout_control_results.json"
with open(out, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved → {out}")
