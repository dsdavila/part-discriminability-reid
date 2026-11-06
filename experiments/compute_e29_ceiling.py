"""
E29 ceiling redo — per-identity ceiling, not ratio-of-means.

For each identity i:
  shared_i   = |D_set_i ∩ B_set_i|
  ceiling_i  = min(|D_set_i|, |B_set_i|)      [max attainable intersection]
  fraction_i = shared_i / ceiling_i if ceiling_i > 0 else NaN
  jaccard_i  = |D ∩ B| / |D ∪ B|
  jaccard_ceiling_i = 1 (always; attained when sets are equal and same size)

Aggregate: mean/median of fraction_i, Jaccard; bootstrap n=1000 over identities.
Pairs: MSMT D vs B, Market A vs C.
"""
import json, time
import numpy as np
from pathlib import Path

CACHE_DIR = Path('/tmp/e12_feature_cache')
RESULTS   = Path('/data/papers/low_dimensional_reid/results')
K_LIST    = [10, 50]
N_BOOT    = 1000
RNG       = np.random.default_rng(42)

CELLS = {
    'A': {'file': 'market_test.npz',           'n_q': 3368},
    'C': {'file': 'bot_msmt_seed42_market.npz','n_q': 3368},
    'D': {'file': 'bot_msmt_seed42_msmt.npz',  'n_q': 11659},
    'B': {'file': 'msmt17_test.npz',           'n_q': 11659},
}
PAIRS = [('MSMT',  'D', 'B'),
         ('Market','A', 'C')]


def load_gallery(cell):
    d = np.load(CACHE_DIR / CELLS[cell]['file'])
    n_q = CELLS[cell]['n_q']
    return (d['emb'][n_q:].astype(np.float32),
            d['pids'][n_q:], d['cids'][n_q:])


def topk_neighbours(G, g_pids, g_cids, k_max, chunk=2048):
    N = len(G)
    out = np.full((N, k_max), -1, dtype=np.int64)
    for start in range(0, N, chunk):
        end = min(N, start + chunk)
        sims = G[start:end] @ G.T
        for local, gi in enumerate(range(start, end)):
            exc = (g_pids == g_pids[gi]) & (g_cids == g_cids[gi])
            sims[local, exc] = -2.0
            top_idx = np.argpartition(-sims[local], k_max)[:k_max]
            order   = np.argsort(-sims[local, top_idx])
            out[gi] = top_idx[order]
    return out


def per_identity_sets(topk_idx, g_pids, k):
    """Return list of {'pid': int, 'set': set-of-intruder-pids} across identities."""
    unique_ids = np.unique(g_pids)
    nbr_pids_k = g_pids[topk_idx[:, :k]]      # (N, k)
    im_mask    = (nbr_pids_k != g_pids[:, None])
    out = {}
    for uid in unique_ids:
        mask = (g_pids == uid)
        intruders = nbr_pids_k[mask][im_mask[mask]]
        out[int(uid)] = set(int(x) for x in intruders)
    return out


def analyse_pair(eval_name, cell_m1, cell_m2):
    print(f"\n=== Pair {eval_name}: {cell_m1} vs {cell_m2} ===", flush=True)
    G1, pids1, cids1 = load_gallery(cell_m1)
    G2, pids2, cids2 = load_gallery(cell_m2)
    assert np.array_equal(pids1, pids2) and np.array_equal(cids1, cids2)
    g_pids = pids1; g_cids = cids1

    k_max = max(K_LIST)
    t0 = time.time()
    print(f"  top-{k_max} under {cell_m1}...", flush=True)
    top_m1 = topk_neighbours(G1, g_pids, g_cids, k_max)
    print(f"  top-{k_max} under {cell_m2}... t={(time.time()-t0)/60:.1f}m", flush=True)
    top_m2 = topk_neighbours(G2, g_pids, g_cids, k_max)
    print(f"  done. t={(time.time()-t0)/60:.1f}m", flush=True)

    out = {'eval': eval_name, 'm1': cell_m1, 'm2': cell_m2, 'per_k': {}}

    for k in K_LIST:
        sets_m1 = per_identity_sets(top_m1, g_pids, k)
        sets_m2 = per_identity_sets(top_m2, g_pids, k)

        # Per-identity arrays
        frac_arr    = []     # shared / min(|m1|, |m2|)
        jacc_arr    = []     # |∩| / |∪|
        shared_arr  = []     # |∩|
        size_m1_arr = []
        size_m2_arr = []
        for uid in sets_m1:
            a, b = sets_m1[uid], sets_m2[uid]
            if not a or not b: continue
            inter = len(a & b)
            uni   = len(a | b)
            ceil  = min(len(a), len(b))
            frac_arr.append(inter / ceil)
            jacc_arr.append(inter / uni)
            shared_arr.append(inter)
            size_m1_arr.append(len(a))
            size_m2_arr.append(len(b))

        frac_arr    = np.array(frac_arr, dtype=np.float64)
        jacc_arr    = np.array(jacc_arr, dtype=np.float64)
        shared_arr  = np.array(shared_arr, dtype=np.float64)
        size_m1_arr = np.array(size_m1_arr, dtype=np.float64)
        size_m2_arr = np.array(size_m2_arr, dtype=np.float64)

        def boot_stat(x, fn, n_boot=N_BOOT):
            rng = np.random.default_rng(42)
            n = len(x)
            vals = []
            for _ in range(n_boot):
                idx = rng.integers(0, n, size=n)
                vals.append(fn(x[idx]))
            return float(fn(x)), [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]

        frac_mean, frac_ci = boot_stat(frac_arr, np.mean)
        frac_med, frac_ci_med = boot_stat(frac_arr, np.median)
        jacc_mean, jacc_ci = boot_stat(jacc_arr, np.mean)
        jacc_med, _ = boot_stat(jacc_arr, np.median)

        row = {
            'n_matched_ids': int(len(frac_arr)),
            'per_identity_frac_of_ceiling': {
                'mean': frac_mean, 'mean_ci95': frac_ci,
                'median': frac_med, 'median_ci95': frac_ci_med,
                'quartiles': [float(np.percentile(frac_arr, 25)),
                              float(np.median(frac_arr)),
                              float(np.percentile(frac_arr, 75))],
            },
            'per_identity_jaccard': {
                'mean': jacc_mean, 'mean_ci95': jacc_ci,
                'median': jacc_med,
            },
            'set_sizes': {
                'mean_size_m1': float(size_m1_arr.mean()),
                'mean_size_m2': float(size_m2_arr.mean()),
                'median_size_m1': float(np.median(size_m1_arr)),
                'median_size_m2': float(np.median(size_m2_arr)),
            },
            'mean_shared_intruders': float(shared_arr.mean()),
        }
        print(f"  k={k}  frac_of_ceiling: mean={frac_mean:.4f} CI={frac_ci}  "
              f"median={frac_med:.4f}  Jaccard mean={jacc_mean:.4f}  "
              f"|m1|={size_m1_arr.mean():.2f}  |m2|={size_m2_arr.mean():.2f}  "
              f"shared={shared_arr.mean():.3f}", flush=True)
        out['per_k'][str(k)] = row
    return out


def main():
    results = {}
    for eval_name, m1, m2 in PAIRS:
        results[eval_name] = analyse_pair(eval_name, m1, m2)
    out_path = RESULTS / 'e29_ceiling_per_identity.json'
    with open(out_path, 'w') as f: json.dump(results, f, indent=2)
    print(f"\nSaved → {out_path}", flush=True)


if __name__ == '__main__':
    main()
