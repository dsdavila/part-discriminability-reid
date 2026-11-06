"""
Per-identity rank-1 error rate: fraction of an identity's gallery items whose
top-1 gallery neighbour (same-pid-same-cam excluded) is a different identity.

Report histogram per cell and class counts at thresholds 0.25, 0.50, 0.75.
"""
import json, time
import numpy as np
from pathlib import Path

CACHE_DIR = Path('/tmp/e12_feature_cache')
RESULTS   = Path('/data/papers/low_dimensional_reid/results')

CELLS = {
    "A'": {'file': 'market_test_noREA.npz',    'n_q': 3368},
    'C':  {'file': 'bot_msmt_seed42_market.npz','n_q': 3368},
    'D':  {'file': 'bot_msmt_seed42_msmt.npz', 'n_q': 11659},
    "B'": {'file': 'msmt17_test_noREA.npz',    'n_q': 11659},
}


def load_gallery(cell):
    cfg = CELLS[cell]
    d = np.load(CACHE_DIR / cfg['file']); n_q = cfg['n_q']
    return (d['emb'][n_q:].astype(np.float32), d['pids'][n_q:], d['cids'][n_q:])


def top1_neighbour_pid(G, g_pids, g_cids, chunk=2048):
    """Return array of top-1 neighbour pid (excluding same-pid-same-cam) per gallery item."""
    N = len(G)
    out = np.full(N, -1, dtype=np.int64)
    for start in range(0, N, chunk):
        end = min(N, start + chunk)
        sims = G[start:end] @ G.T
        for local, gi in enumerate(range(start, end)):
            exc = (g_pids == g_pids[gi]) & (g_cids == g_cids[gi])
            sims[local, exc] = -2.0
            top = int(np.argmax(sims[local]))
            out[gi] = g_pids[top]
    return out


def main():
    out = {}
    THRESHOLDS = [0.25, 0.50, 0.75]
    for cell in CELLS:
        print(f"\n=== Cell {cell} ===", flush=True)
        G, g_pids, g_cids = load_gallery(cell)
        print(f"  gallery N={len(G)}  n_ids={len(np.unique(g_pids))}", flush=True)
        t0 = time.time()
        top1 = top1_neighbour_pid(G, g_pids, g_cids)
        print(f"  top-1 computed in {(time.time()-t0)/60:.1f}m", flush=True)
        wrong = (top1 != g_pids)
        unique_ids = np.unique(g_pids)
        per_id_rate = []
        for uid in unique_ids:
            mask = (g_pids == uid)
            per_id_rate.append(float(wrong[mask].mean()))
        per_id_rate = np.array(per_id_rate)
        # Histogram
        hist, edges = np.histogram(per_id_rate, bins=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.001])
        hist_frac = (hist / len(per_id_rate)).tolist()
        print(f"  rank-1 error rate: min={per_id_rate.min():.3f}  max={per_id_rate.max():.3f}  median={np.median(per_id_rate):.3f}  mean={per_id_rate.mean():.3f}")
        print(f"  histogram (10 bins, 0-1):")
        for i in range(10):
            print(f"    [{edges[i]:.1f}, {edges[i+1]:.1f}): {hist[i]:5d} ids ({hist_frac[i]*100:5.1f}%)")
        confused_counts = {}
        for thr in THRESHOLDS:
            n_confused = int((per_id_rate > thr).sum())
            confused_counts[str(thr)] = {'n_confused': n_confused,
                                          'frac': float(n_confused / len(per_id_rate))}
            print(f"  threshold > {thr}:  n_confused = {n_confused} / {len(per_id_rate)} = {n_confused/len(per_id_rate)*100:.1f}%")
        out[cell] = {
            'n_ids': int(len(unique_ids)),
            'mean_rate': float(per_id_rate.mean()),
            'median_rate': float(np.median(per_id_rate)),
            'min_rate': float(per_id_rate.min()),
            'max_rate': float(per_id_rate.max()),
            'hist_counts': hist.tolist(),
            'hist_fracs': hist_frac,
            'confused_at_thresholds': confused_counts,
        }
    out_path = RESULTS / 'rank1_error_hist.json'
    with open(out_path, 'w') as f: json.dump(out, f, indent=2)
    print(f"\nSaved → {out_path}")


if __name__ == '__main__':
    main()
