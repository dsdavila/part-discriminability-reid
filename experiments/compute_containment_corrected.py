"""
Corrected containment: use SOURCE-domain train prototypes for cross-domain cells.
- Cell A: BoT-Market model, Market-train proto -> Market test IDs (in-domain control) [unchanged]
- Cell B: BoT-Market model, Market-train proto -> MSMT17 test IDs (CORRECTED: was MSMT17-train)
- Cell C: BoT-MSMT17 model, MSMT17-train proto -> Market test IDs (CORRECTED: was Market-train)
- Cell D: BoT-MSMT17 model, MSMT17-train proto -> MSMT17 test IDs (in-domain control) [unchanged]

Also includes count-matched check: subsample larger prototype set to match smaller (751 Market vs 1041 MSMT17).
Reports alongside tr(S_B) from Fisher results.
"""
import numpy as np, json
from pathlib import Path

CACHE_DIR = Path("/tmp/e12_feature_cache")
TRAIN_CACHE_DIR = Path("/tmp/e18_train_cache")
RESULTS_DIR = Path("/data/papers/low_dimensional_reid/results")
N_QUERY_MARKET, N_QUERY_MSMT = 3368, 11655
RNG = np.random.default_rng(42)

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

def containment_stats(tgt_centroids, src_centroids, label=""):
    """Max cosine sim of each tgt centroid to nearest src centroid."""
    sim = tgt_centroids @ src_centroids.T  # (n_tgt, n_src)
    max_sims = sim.max(axis=1)
    r = {
        "n_tgt": len(tgt_centroids),
        "n_src": len(src_centroids),
        "mean_max_sim": float(max_sims.mean()),
        "median_max_sim": float(np.median(max_sims)),
        "frac_gt_0.3": float((max_sims > 0.3).mean()),
        "frac_gt_0.5": float((max_sims > 0.5).mean()),
        "frac_gt_0.7": float((max_sims > 0.7).mean()),
        "frac_gt_0.9": float((max_sims > 0.9).mean()),
    }
    print(f"  {label}: n_tgt={len(tgt_centroids)}, n_src={len(src_centroids)}")
    print(f"    mean_max_sim={r['mean_max_sim']:.4f}  median={r['median_max_sim']:.4f}")
    print(f"    frac>0.5={r['frac_gt_0.5']:.3f}  frac>0.7={r['frac_gt_0.7']:.3f}")
    return r

print("Loading test gallery features...", flush=True)
A_emb, A_pids, _ = load_gallery("market_test.npz", "market")
B_emb, B_pids, _ = load_gallery("msmt17_test.npz", "msmt")
C_emb, C_pids, _ = load_gallery("bot_msmt_seed42_market.npz", "market")
D_emb, D_pids, _ = load_gallery("bot_msmt_seed42_msmt.npz", "msmt")

print("Loading train features...", flush=True)
mkt_mkt_emb, mkt_mkt_pids = load_train_cache(TRAIN_CACHE_DIR / "bot_market_market_train.npz")
msmt_msmt_emb, msmt_msmt_pids = load_train_cache(TRAIN_CACHE_DIR / "bot_msmt_msmt_train.npz")

print("Computing centroids...", flush=True)
mkt_mkt_src = compute_centroids(mkt_mkt_emb, mkt_mkt_pids)   # 751 IDs, BoT-Market on Market-train
msmt_msmt_src = compute_centroids(msmt_msmt_emb, msmt_msmt_pids)  # 1041 IDs, BoT-MSMT on MSMT-train

A_tgt = compute_centroids(A_emb, A_pids)  # Market test (750 IDs) under BoT-Market
B_tgt = compute_centroids(B_emb, B_pids)  # MSMT17 test (3059 IDs) under BoT-Market
C_tgt = compute_centroids(C_emb, C_pids)  # Market test (750 IDs) under BoT-MSMT17
D_tgt = compute_centroids(D_emb, D_pids)  # MSMT17 test (3059 IDs) under BoT-MSMT17

print(f"\nPrototype counts: Market-train={len(mkt_mkt_src)}, MSMT17-train={len(msmt_msmt_src)}", flush=True)

# Count-matched: subsample MSMT17-train to 751 IDs (= Market-train count)
n_match = len(mkt_mkt_src)
idx_sub = RNG.choice(len(msmt_msmt_src), size=n_match, replace=False)
msmt_msmt_src_sub = msmt_msmt_src[idx_sub]

print("\n=== CORRECTED CONTAINMENT (source-domain prototypes) ===\n", flush=True)
results = {}

print("Cell A (BoT-Market / Market-train prototypes / Market test — in-domain control):", flush=True)
results["A"] = containment_stats(A_tgt, mkt_mkt_src, "A")

print("\nCell B CORRECTED (BoT-Market / Market-train prototypes / MSMT17 test):", flush=True)
results["B"] = containment_stats(B_tgt, mkt_mkt_src, "B")

print("\nCell C CORRECTED (BoT-MSMT17 / MSMT17-train prototypes / Market test):", flush=True)
results["C"] = containment_stats(C_tgt, msmt_msmt_src, "C")

print("\nCell D (BoT-MSMT17 / MSMT17-train prototypes / MSMT17 test — in-domain control):", flush=True)
results["D"] = containment_stats(D_tgt, msmt_msmt_src, "D")

print("\n=== COUNT-MATCHED CHECK (subsample MSMT17-train to 751 IDs) ===\n", flush=True)
results["B_count_matched"] = containment_stats(B_tgt, mkt_mkt_src, "B (already 751, no change needed)")
results["C_count_matched"] = containment_stats(C_tgt, msmt_msmt_src_sub, "C count-matched (751 MSMT17 proto)")
results["D_count_matched"] = containment_stats(D_tgt, msmt_msmt_src_sub, "D count-matched (751 MSMT17 proto)")

# tr(S_B) from Fisher results (from previous Fisher computation)
# These values from claude-chat rev 106: A=0.627, B=0.241, C=0.391, D=0.448
tr_sb = {"A": 0.627, "B": 0.241, "C": 0.391, "D": 0.448}

print("\n=== SUMMARY TABLE ===\n", flush=True)
print(f"{'Cell':<8} {'tr(S_B)':<10} {'mean_max_sim':<15} {'frac>0.7':<10}")
for cell in ["A", "B", "C", "D"]:
    r = results[cell]
    print(f"  {cell:<6} {tr_sb[cell]:<10.3f} {r['mean_max_sim']:<15.4f} {r['frac_gt_0.7']:<10.3f}")

out = RESULTS_DIR / "e18_containment_corrected.json"
with open(out, "w") as f:
    json.dump({"results": results, "tr_sb": tr_sb}, f, indent=2)
print(f"\nSaved → {out}", flush=True)
