"""
Reciprocity-vs-mAP sweep across all (j, k) projections of all four cells.

For each (j, k) in the band-pass grid, computes:
  - projected gallery features
  - top-50 nearest neighbours (excluding same-pid-same-cam)
  - k=50 chance-corrected reciprocity
  - edge density
  - (for pairs A'/C and D/B'): per-identity ceiling-corrected overlap

Output: /data/papers/low_dimensional_reid/results/bandpass_recip_sweep.json

Needs the mAP sweep results to merge into the final scatter plot — reads from
/data/papers/low_dimensional_reid/results/bandpass_sweep_noREA_results.json.
"""
import json, time
import numpy as np
from pathlib import Path

CACHE_DIR = Path('/tmp/e12_feature_cache')
RESULTS   = Path('/data/papers/low_dimensional_reid/results')
MAP_PATH  = RESULTS / 'bandpass_sweep_noREA_results.json'
K_RECIP   = 50
N_REWIRE  = 20
RNG       = np.random.default_rng(42)

CELLS = {
    "A'": {'file': 'market_test_noREA.npz',    'n_q': 3368,  'eval': 'market'},
    'C':  {'file': 'bot_msmt_seed42_market.npz','n_q': 3368,  'eval': 'market'},
    'D':  {'file': 'bot_msmt_seed42_msmt.npz', 'n_q': 11659, 'eval': 'msmt'},
    "B'": {'file': 'msmt17_test_noREA.npz',    'n_q': 11659, 'eval': 'msmt'},
}
J_GRID = [0, 2, 4, 8, 16, 32]
K_GRID_RAW = [32, 64, 128, 256, 'all']
PAIRS_FOR_OVERLAP = [('MSMT', 'D', "B'"), ('Market', "A'", 'C')]


def load_cell(cell):
    cfg = CELLS[cell]
    d = np.load(CACHE_DIR / cfg['file']); n_q = cfg['n_q']
    emb = d['emb'].astype(np.float32)
    return (emb[n_q:], d['pids'][n_q:], d['cids'][n_q:])


def l2(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(n, 1e-12)


def pca_basis(G):
    mu = G.mean(axis=0, keepdims=True)
    Gc = (G - mu).astype(np.float64)
    _, _, Vt = np.linalg.svd(Gc, full_matrices=False)
    return mu.astype(np.float32), Vt.T.astype(np.float32)


def project(g, mu, V, j, k_or_all):
    D = V.shape[1]
    end = D if k_or_all == 'all' else min(D, j + k_or_all)
    if end <= j: return None
    Vk = V[:, j:end]
    return l2((g - mu) @ Vk)


def topk_neighbours(G, g_pids, g_cids, k_max, chunk=2048):
    N = len(G)
    out = np.full((N, k_max), -1, dtype=np.int64)
    for start in range(0, N, chunk):
        end = min(N, start + chunk)
        sims = G[start:end] @ G.T
        for local, gi in enumerate(range(start, end)):
            exc = (g_pids == g_pids[gi]) & (g_cids == g_cids[gi])
            sims[local, exc] = -2.0
            top = np.argpartition(-sims[local], k_max)[:k_max]
            out[gi] = top[np.argsort(-sims[local, top])]
    return out


def build_graph(topk_idx, g_pids, k):
    unique_ids = np.unique(g_pids)
    nbr = g_pids[topk_idx[:, :k]]
    im = (nbr != g_pids[:, None])
    adj = {int(u): set() for u in unique_ids}
    for i, u in enumerate(g_pids):
        for jj in nbr[i][im[i]]:
            adj[int(u)].add(int(jj))
    return adj


def reciprocity_of(adj):
    edges = set()
    for i, outs in adj.items():
        for j in outs: edges.add((i, j))
    if not edges: return 0.0, 0
    recip = sum(1 for e in edges if (e[1], e[0]) in edges)
    return recip / len(edges), len(edges)


def config_rewire(adj, rng, factor=15):
    edges = [(i, j) for i, outs in adj.items() for j in outs]
    n = len(edges)
    adj_sets = {k: set(v) for k, v in adj.items()}
    for _ in range(factor * n):
        if n < 2: break
        a, b = rng.integers(0, n), rng.integers(0, n)
        if a == b: continue
        i1, j1 = edges[a]; i2, j2 = edges[b]
        if j2 in adj_sets[i1] or j1 in adj_sets[i2]: continue
        if j2 == i1 or j1 == i2: continue
        adj_sets[i1].remove(j1); adj_sets[i1].add(j2)
        adj_sets[i2].remove(j2); adj_sets[i2].add(j1)
        edges[a] = (i1, j2); edges[b] = (i2, j1)
    return adj_sets


def per_id_sets(topk_idx, g_pids, k):
    uniq = np.unique(g_pids)
    nbr = g_pids[topk_idx[:, :k]]
    im = (nbr != g_pids[:, None])
    out = {}
    for uid in uniq:
        mask = g_pids == uid
        intruders = nbr[mask][im[mask]]
        out[int(uid)] = set(int(x) for x in intruders)
    return out


def ceiling_corrected_overlap(sets1, sets2):
    vals = []
    for uid in sets1:
        a, b = sets1[uid], sets2[uid]
        if not a or not b: continue
        vals.append(len(a & b) / min(len(a), len(b)))
    return float(np.mean(vals)) if vals else 0.0


def main():
    # Load mAP sweep for merging
    maps = json.load(open(MAP_PATH)) if MAP_PATH.exists() else {}
    print(f"Loaded mAP sweep: {len(maps)} cells, {sum(len(v) for v in maps.values())} configs", flush=True)

    # Pre-compute PCA bases
    bases = {}
    cell_data = {}
    for cell in CELLS:
        print(f"\nLoading {cell}, PCA basis...", flush=True)
        G, g_pids, g_cids = load_cell(cell)
        t0 = time.time()
        mu, V = pca_basis(G)
        print(f"  N={len(G)} D={G.shape[1]}  PCA t={(time.time()-t0):.1f}s", flush=True)
        bases[cell] = (mu, V)
        cell_data[cell] = (G, g_pids, g_cids)

    # Sweep per-cell: topk + reciprocity per (j, k)
    results = {'per_cell': {}, 'per_pair_overlap': {}}
    topk_cache = {}
    for cell in CELLS:
        G, g_pids, g_cids = cell_data[cell]
        mu, V = bases[cell]
        n_ids = len(np.unique(g_pids))
        max_edges = n_ids * (n_ids - 1)
        cell_out = {}
        print(f"\n=== Cell {cell} ===", flush=True)
        for j in J_GRID:
            for k_raw in K_GRID_RAW:
                proj_g = project(G, mu, V, j, k_raw)
                if proj_g is None: continue
                key = f'j={j}_k={k_raw}'
                t0 = time.time()
                topk = topk_neighbours(proj_g, g_pids, g_cids, K_RECIP)
                adj = build_graph(topk, g_pids, K_RECIP)
                meas, n_edges = reciprocity_of(adj)
                rng = np.random.default_rng(42)
                null_vals = [reciprocity_of(config_rewire(adj, rng))[0] for _ in range(N_REWIRE)]
                null_mean = float(np.mean(null_vals))
                cc = (meas - null_mean) / max(1 - null_mean, 1e-12)
                density = n_edges / max(max_edges, 1)
                # Get mAP from the merged results
                map_val = maps.get(cell, {}).get(key, {}).get('mAP_pct', None)
                cell_out[key] = {
                    'j': j, 'k_band': k_raw, 'k_effective': proj_g.shape[1],
                    'mAP_pct': map_val,
                    'n_edges': int(n_edges), 'density': density,
                    'reciprocity_measured': meas, 'reciprocity_null_mean': null_mean,
                    'reciprocity_chance_corrected': cc,
                }
                print(f"  {key:<16}  mAP={map_val if map_val is None else f'{map_val:.2f}':>7}  "
                      f"edges={n_edges:>7}  density={density:.4f}  "
                      f"recip={meas:.4f}  cc={cc:.4f}  t={time.time()-t0:.1f}s", flush=True)
                topk_cache[(cell, key)] = topk
        results['per_cell'][cell] = cell_out

    # Pair-wise ceiling-corrected overlap per (j, k): need both cells' topk at same (j, k)
    print("\n=== Pair overlaps ===", flush=True)
    for pair_name, c1, c2 in PAIRS_FOR_OVERLAP:
        pair_out = {}
        for j in J_GRID:
            for k_raw in K_GRID_RAW:
                key = f'j={j}_k={k_raw}'
                t1 = topk_cache.get((c1, key)); t2 = topk_cache.get((c2, key))
                if t1 is None or t2 is None: continue
                _, g_pids, _ = cell_data[c1]
                for k in [10, 50]:
                    sets1 = per_id_sets(t1, g_pids, k)
                    sets2 = per_id_sets(t2, g_pids, k)
                    cco = ceiling_corrected_overlap(sets1, sets2)
                    pair_out.setdefault(key, {})[f'ceiling_corrected_overlap_k{k}'] = cco
        results['per_pair_overlap'][pair_name] = pair_out

    out_path = RESULTS / 'bandpass_recip_sweep.json'
    with open(out_path, 'w') as f: json.dump(results, f, indent=2)
    print(f"\nSaved → {out_path}", flush=True)


if __name__ == '__main__':
    main()
