"""
Corrected affine interpolation for cells B and C using source-domain prototypes.
- Cell B: BoT-Market model, Market-train prototypes (751 IDs), MSMT17 test IDs
- Cell C: BoT-MSMT17 model, MSMT17-train prototypes (1041 IDs), Market test IDs
A and D from Phase 2b were already correct (same-model, same-domain).
"""
import numpy as np, json
from pathlib import Path
from scipy.optimize import minimize

CACHE_DIR = Path("/tmp/e12_feature_cache")
TRAIN_CACHE_DIR = Path("/tmp/e18_train_cache")
RESULTS_DIR = Path("/data/papers/low_dimensional_reid/results")
N_QUERY_MARKET, N_QUERY_MSMT = 3368, 11655
K_INTERP = 20

def load_gallery(fname, dataset):
    d = np.load(CACHE_DIR / fname)
    n_q = N_QUERY_MARKET if dataset == "market" else N_QUERY_MSMT
    return d["emb"][n_q:], d["pids"][n_q:], d["cids"][n_q:]

def load_train_cache(path):
    d = np.load(path)
    return d["emb"], d["pids"]

def compute_centroids(emb, pids):
    unique_ids = np.unique(pids)
    centroids = []
    for pid in unique_ids:
        c = emb[pids == pid].mean(0)
        c /= np.linalg.norm(c) + 1e-10
        centroids.append(c)
    return np.array(centroids)

def interpolation_error(tgt_centroids, src_centroids, k=20, label=""):
    sim = tgt_centroids @ src_centroids.T
    errors = []
    for i in range(len(tgt_centroids)):
        idx = np.argsort(-sim[i])[:k]
        S = src_centroids[idx].T  # (2048, k)
        t = tgt_centroids[i]
        def obj(w):
            approx = S @ w
            approx = approx / (np.linalg.norm(approx) + 1e-10)
            return float(1 - np.dot(approx, t))
        w0 = np.ones(k) / k
        res = minimize(obj, w0, method='SLSQP',
                       bounds=[(0, 1)] * k,
                       constraints={'type': 'eq', 'fun': lambda w: w.sum() - 1},
                       options={'maxiter': 100, 'ftol': 1e-6})
        errors.append(float(res.fun))
        if i % 200 == 0:
            print(f"    {label} {i}/{len(tgt_centroids)}", flush=True)
    return {
        "mean_cos_error": float(np.mean(errors)),
        "median_cos_error": float(np.median(errors)),
        "p90_cos_error": float(np.percentile(errors, 90)),
        "n_tgt": len(tgt_centroids),
        "n_src": len(src_centroids),
    }

print("Loading test gallery features...", flush=True)
B_emb, B_pids, _ = load_gallery("msmt17_test.npz", "msmt")
C_emb, C_pids, _ = load_gallery("bot_msmt_seed42_market.npz", "market")

print("Loading train features...", flush=True)
mkt_mkt_emb, mkt_mkt_pids = load_train_cache(TRAIN_CACHE_DIR / "bot_market_market_train.npz")
msmt_msmt_emb, msmt_msmt_pids = load_train_cache(TRAIN_CACHE_DIR / "bot_msmt_msmt_train.npz")

print("Computing centroids...", flush=True)
mkt_mkt_src = compute_centroids(mkt_mkt_emb, mkt_mkt_pids)
msmt_msmt_src = compute_centroids(msmt_msmt_emb, msmt_msmt_pids)
B_tgt = compute_centroids(B_emb, B_pids)
C_tgt = compute_centroids(C_emb, C_pids)

print(f"\nPrototype counts: Market-train={len(mkt_mkt_src)}, MSMT17-train={len(msmt_msmt_src)}", flush=True)
print(f"Test ID counts: B={len(B_tgt)}, C={len(C_tgt)}\n", flush=True)

results = {}

print("Cell B CORRECTED (BoT-Market / Market-train prototypes / MSMT17 test):", flush=True)
results["B"] = interpolation_error(B_tgt, mkt_mkt_src, k=K_INTERP, label="B")
print(f"  mean_cos_error={results['B']['mean_cos_error']:.4f}  median={results['B']['median_cos_error']:.4f}  p90={results['B']['p90_cos_error']:.4f}", flush=True)

print("\nCell C CORRECTED (BoT-MSMT17 / MSMT17-train prototypes / Market test):", flush=True)
results["C"] = interpolation_error(C_tgt, msmt_msmt_src, k=K_INTERP, label="C")
print(f"  mean_cos_error={results['C']['mean_cos_error']:.4f}  median={results['C']['median_cos_error']:.4f}  p90={results['C']['p90_cos_error']:.4f}", flush=True)

# Load original A and D results
with open(RESULTS_DIR / "e18_phase2_results.json") as f:
    orig = json.load(f)

print("\n=== SUMMARY TABLE (A, D from Phase 2b; B, C corrected) ===", flush=True)
print(f"{'Cell':<6} {'mean_cos_err':<15} {'median':<10} {'p90':<10} {'note'}")
for cell, mean_e, med_e, p90_e, note in [
    ("A", orig["interpolation"]["A"]["mean_cos_error"],
         orig["interpolation"]["A"]["median_cos_error"],
         orig["interpolation"]["A"]["p90_cos_error"], "in-domain control"),
    ("B", results["B"]["mean_cos_error"], results["B"]["median_cos_error"], results["B"]["p90_cos_error"], "CORRECTED"),
    ("C", results["C"]["mean_cos_error"], results["C"]["median_cos_error"], results["C"]["p90_cos_error"], "CORRECTED"),
    ("D", orig["interpolation"]["D"]["mean_cos_error"],
         orig["interpolation"]["D"]["median_cos_error"],
         orig["interpolation"]["D"]["p90_cos_error"], "in-domain control"),
]:
    print(f"  {cell:<4} {mean_e:<15.4f} {med_e:<10.4f} {p90_e:<10.4f} {note}")

out = RESULTS_DIR / "e18_interpolation_corrected.json"
with open(out, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved → {out}", flush=True)
