"""
Corrected mixing-weight entropy: use SOURCE-domain train prototypes for cross-domain cells.
- Cell A: BoT-Market / Market-train proto / Market test (in-domain, same as before)
- Cell B: BoT-Market / Market-train proto / MSMT17 test (CORRECTED: was MSMT17-train)
- Cell C: BoT-MSMT17 / MSMT17-train proto / Market test (CORRECTED: was Market-train)
- Cell D: BoT-MSMT17 / MSMT17-train proto / MSMT17 test (in-domain, same as before)
"""
import numpy as np, json
from pathlib import Path
from scipy.optimize import minimize

CACHE_DIR = Path("/tmp/e12_feature_cache")
TRAIN_CACHE_DIR = Path("/tmp/e18_train_cache")
RESULTS_DIR = Path("/data/papers/low_dimensional_reid/results")
N_QUERY_MARKET, N_QUERY_MSMT = 3368, 11655
K = 20

def load_gallery(fname, dataset):
    d = np.load(CACHE_DIR / fname)
    n_q = N_QUERY_MARKET if dataset == "market" else N_QUERY_MSMT
    return d["emb"][n_q:], d["pids"][n_q:]

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

def mixing_entropy_stats(src_centroids, tgt_centroids, label=""):
    sim = tgt_centroids @ src_centroids.T
    entropies = []
    for i in range(len(tgt_centroids)):
        idx = np.argsort(-sim[i])[:K]
        S = src_centroids[idx].T
        t = tgt_centroids[i]
        def obj(w):
            approx = S @ w
            approx = approx / (np.linalg.norm(approx) + 1e-10)
            return float(1 - np.dot(approx, t))
        w0 = np.ones(K) / K
        res = minimize(obj, w0, method='SLSQP',
                       bounds=[(0, 1)] * K,
                       constraints={'type': 'eq', 'fun': lambda w: w.sum() - 1},
                       options={'maxiter': 100, 'ftol': 1e-6})
        w = np.maximum(res.x, 0)
        w /= w.sum() + 1e-300
        H = -np.sum(w * np.log(w + 1e-300))
        H_max = np.log(K)
        entropies.append(H / H_max)
        if i % 200 == 0:
            print(f"  {label} {i}/{len(tgt_centroids)}", flush=True)
    return {
        "mean_norm_entropy": float(np.mean(entropies)),
        "median_norm_entropy": float(np.median(entropies)),
        "p10_norm_entropy": float(np.percentile(entropies, 10)),
        "n_tgt": len(tgt_centroids),
        "n_src": len(src_centroids),
        "proto_cache": "source-domain",
    }

print("Loading gallery features...", flush=True)
A_emb, A_pids = load_gallery("market_test.npz", "market")
B_emb, B_pids = load_gallery("msmt17_test.npz", "msmt")
C_emb, C_pids = load_gallery("bot_msmt_seed42_market.npz", "market")
D_emb, D_pids = load_gallery("bot_msmt_seed42_msmt.npz", "msmt")

print("Loading train features...", flush=True)
mkt_mkt_emb, mkt_mkt_pids = load_train_cache(TRAIN_CACHE_DIR / "bot_market_market_train.npz")
msmt_msmt_emb, msmt_msmt_pids = load_train_cache(TRAIN_CACHE_DIR / "bot_msmt_msmt_train.npz")

print("Computing centroids...", flush=True)
mkt_mkt_src  = compute_centroids(mkt_mkt_emb, mkt_mkt_pids)    # 751 IDs, Market-train/BoT-Market
msmt_msmt_src = compute_centroids(msmt_msmt_emb, msmt_msmt_pids)  # 1041 IDs, MSMT17-train/BoT-MSMT17
A_tgt = compute_centroids(A_emb, A_pids)
B_tgt = compute_centroids(B_emb, B_pids)
C_tgt = compute_centroids(C_emb, C_pids)
D_tgt = compute_centroids(D_emb, D_pids)
print(f"  src: mkt_mkt={len(mkt_mkt_src)}, msmt_msmt={len(msmt_msmt_src)}", flush=True)
print(f"  tgt: A={len(A_tgt)}, B={len(B_tgt)}, C={len(C_tgt)}, D={len(D_tgt)}", flush=True)

results = {}
for label, src_c, tgt_c, desc in [
    ("A", mkt_mkt_src,   A_tgt, "BoT-Mkt/Market-train proto/Market test (in-domain)"),
    ("B", mkt_mkt_src,   B_tgt, "BoT-Mkt/Market-train proto/MSMT17 test (cross-dom, CORRECTED)"),
    ("C", msmt_msmt_src, C_tgt, "BoT-MSMT/MSMT17-train proto/Market test (cross-dom, CORRECTED)"),
    ("D", msmt_msmt_src, D_tgt, "BoT-MSMT/MSMT17-train proto/MSMT17 test (in-domain)"),
]:
    print(f"\n=== Cell {label}: {desc} ===", flush=True)
    r = mixing_entropy_stats(src_c, tgt_c, label=label)
    r["desc"] = desc
    results[label] = r
    print(f"  mean_norm_H={r['mean_norm_entropy']:.4f}  median={r['median_norm_entropy']:.4f}  p10={r['p10_norm_entropy']:.4f}", flush=True)

out = RESULTS_DIR / "e18_mixing_entropy_corrected.json"
with open(out, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved → {out}", flush=True)
