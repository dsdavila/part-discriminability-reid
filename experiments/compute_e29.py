"""
E29 — Confuser-set overlap: separate (a) model failure from (b) label degeneracy.

For each test identity i on eval set S:
  - Under model M1, find its top-k intruder identity set across its gallery items
    (canonical junk predicate: same-pid-same-cam; intruders = distinct other IDs)
  - Same under model M2
  - Jaccard(set_M1(i), set_M2(i))
  - Null: random intruder set of the same size as M2's drawn uniformly from
    non-same identities, Jaccard against M1's set, averaged over N_PERM draws.

Pairs:
  MSMT:   D (in-domain) vs B (Market→MSMT, OOD)      — the central comparison
  Market: A (in-domain) vs C (MSMT→Market, OOD)      — control where degeneracy is lower
"""
import json, time
import numpy as np
from pathlib import Path
from collections import defaultdict

CACHE_DIR = Path('/tmp/e12_feature_cache')
RESULTS   = Path('/data/papers/low_dimensional_reid/results')
K_LIST    = [10, 50]
N_PERM    = 100
RNG       = np.random.default_rng(42)

# Per cell, we only use the gallery portion for neighbourhood analysis
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
    """Returns topk_idx (N, k_max) of gallery neighbours, excluding same-pid-same-cam."""
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


def per_identity_intruder_set(topk_idx, g_pids, k):
    """For each identity, return the set of distinct intruder PIDs across its items."""
    unique_ids = np.unique(g_pids)
    nbr_pids_k = g_pids[topk_idx[:, :k]]      # (N, k)
    item_pids  = g_pids[:, None]
    intruder_mask = (nbr_pids_k != item_pids)  # (N, k)
    per_id_set = {}
    for uid in unique_ids:
        mask = (g_pids == uid)
        nbr_slice = nbr_pids_k[mask]
        im_slice  = intruder_mask[mask]
        intruders = nbr_slice[im_slice]
        per_id_set[int(uid)] = set(int(x) for x in intruders)
    return per_id_set


def jaccard(a, b):
    if not a and not b: return 1.0
    u = len(a | b)
    if u == 0: return 0.0
    return len(a & b) / u


def analyse_pair(eval_name, cell_m1, cell_m2):
    print(f"\n=== Pair: {eval_name}  {cell_m1} vs {cell_m2} ===", flush=True)
    G1, pids1, cids1 = load_gallery(cell_m1)
    G2, pids2, cids2 = load_gallery(cell_m2)
    assert np.array_equal(pids1, pids2), "gallery item order must match"
    assert np.array_equal(cids1, cids2)
    g_pids = pids1; g_cids = cids1

    k_max = max(K_LIST)
    t0 = time.time()
    print(f"  top-{k_max} under {cell_m1}... ", flush=True)
    top_m1 = topk_neighbours(G1, g_pids, g_cids, k_max)
    print(f"  top-{k_max} under {cell_m2}... t={(time.time()-t0)/60:.1f}m", flush=True)
    top_m2 = topk_neighbours(G2, g_pids, g_cids, k_max)
    print(f"  done. t={(time.time()-t0)/60:.1f}m", flush=True)

    all_eval_ids = np.unique(g_pids)
    out = {'eval': eval_name, 'm1': cell_m1, 'm2': cell_m2, 'n_ids': int(len(all_eval_ids)),
           'per_k': {}}

    rng = np.random.default_rng(42)
    for k in K_LIST:
        set_m1 = per_identity_intruder_set(top_m1, g_pids, k)
        set_m2 = per_identity_intruder_set(top_m2, g_pids, k)

        jaccards = []
        null_jaccards = []
        shared_frac = []   # |m1 ∩ m2| / |m2|  — fraction of OOD intruders shared with in-domain
        for uid in set_m1:
            a, b = set_m1[uid], set_m2[uid]
            if not b:
                continue
            jaccards.append(jaccard(a, b))
            shared_frac.append(len(a & b) / len(b))

            # Permutation null: random intruder set of size |b|, drawn from
            # all other eval identities (excluding uid itself)
            candidates = np.array([int(x) for x in all_eval_ids if int(x) != uid])
            vals = []
            m = min(len(b), len(candidates))
            for _ in range(N_PERM):
                sample = set(int(x) for x in rng.choice(candidates, m, replace=False))
                vals.append(jaccard(a, sample))
            null_jaccards.append(float(np.mean(vals)))

        jaccards = np.array(jaccards)
        null_jaccards = np.array(null_jaccards)
        shared_frac = np.array(shared_frac)

        row = {
            'n_matched_ids': int(len(jaccards)),
            'jaccard': {
                'mean': float(jaccards.mean()),
                'median': float(np.median(jaccards)),
                'quartiles': [float(np.percentile(jaccards,25)),
                              float(np.median(jaccards)),
                              float(np.percentile(jaccards,75))],
            },
            'null_jaccard': {
                'mean': float(null_jaccards.mean()),
                'median': float(np.median(null_jaccards)),
            },
            'jaccard_minus_null': {
                'mean': float((jaccards - null_jaccards).mean()),
                'median': float(np.median(jaccards - null_jaccards)),
            },
            'shared_frac_of_m2': {
                'mean': float(shared_frac.mean()),
                'median': float(np.median(shared_frac)),
            },
            'ratio_jaccard_to_null': float(jaccards.mean() / max(null_jaccards.mean(), 1e-12)),
        }
        print(f"  k={k}  mean Jaccard = {row['jaccard']['mean']:.4f}  "
              f"null = {row['null_jaccard']['mean']:.4f}  "
              f"ratio = {row['ratio_jaccard_to_null']:.1f}×  "
              f"shared_frac_of_{cell_m2} = {row['shared_frac_of_m2']['mean']:.4f}", flush=True)
        out['per_k'][str(k)] = row

    return out


def main():
    results = {}
    for eval_name, m1, m2 in PAIRS:
        results[eval_name] = analyse_pair(eval_name, m1, m2)
    out_path = RESULTS / 'e29_confuser_overlap.json'
    with open(out_path, 'w') as f: json.dump(results, f, indent=2)
    print(f"\nSaved → {out_path}", flush=True)


if __name__ == '__main__':
    main()
