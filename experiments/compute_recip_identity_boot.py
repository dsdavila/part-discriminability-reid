"""
Identity-level block bootstrap for reciprocity CIs.

For each cell at k ∈ {10, 50}:
  - Build the identity-level confusion graph (same as E30-B).
  - Compute null via config_rewire (deterministic from original graph, so bootstrap
    only accounts for identity-set uncertainty; null is treated as fixed).
  - Resample identities with replacement; include each sampled identity's full
    out-edge set and in-edge set (edges where the sampled identity is i OR j).
  - Compute reciprocity on the induced subgraph (edges whose BOTH endpoints are
    in the sampled identity set).
  - Chance-correct using the fixed null.
"""
import json, time
import numpy as np
from pathlib import Path

CACHE_DIR = Path('/tmp/e12_feature_cache')
RESULTS   = Path('/data/papers/low_dimensional_reid/results')
K_LIST = [10, 50]
N_BOOT = 500
N_REWIRE = 20

CELLS = {
    "A'": {'file': 'market_test_noREA.npz',    'n_q': 3368},
    'C':  {'file': 'bot_msmt_seed42_market.npz','n_q': 3368},
    'D':  {'file': 'bot_msmt_seed42_msmt.npz', 'n_q': 11659},
    "B'": {'file': 'msmt17_test_noREA.npz',    'n_q': 11659},
}


def load_gallery(cell):
    cfg = CELLS[cell]
    d = np.load(CACHE_DIR / cfg['file']); n_q = cfg['n_q']
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


def reciprocity_edges(edges):
    es = set(edges)
    if not es: return 0.0, 0
    recip = sum(1 for (i, j) in es if (j, i) in es)
    return recip / len(es), len(es)


def config_rewire_edges(edges, rng, factor=15):
    edges = list(edges)
    n = len(edges)
    adj_sets = {}
    for i, j in edges:
        adj_sets.setdefault(i, set()).add(j)
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
    return edges


def main():
    out = {}
    for cell in CELLS:
        print(f"\n=== Cell {cell} ===", flush=True)
        G, g_pids, g_cids = load_gallery(cell)
        print(f"  top-{max(K_LIST)}...", flush=True)
        t0 = time.time()
        topk = topk_neighbours(G, g_pids, g_cids, max(K_LIST))
        print(f"    t={(time.time()-t0)/60:.1f}m", flush=True)
        cell_out = {}
        for k in K_LIST:
            adj = build_graph(topk, g_pids, k)
            edges = [(i, j) for i, outs in adj.items() for j in outs]
            n_edges = len(edges)
            meas, _ = reciprocity_edges(edges)
            # Null (fixed)
            null_vals = []
            rng_null = np.random.default_rng(42)
            for _ in range(N_REWIRE):
                re = config_rewire_edges(edges, rng_null)
                null_vals.append(reciprocity_edges(re)[0])
            null_mean = float(np.mean(null_vals))
            cc_point = (meas - null_mean) / max(1 - null_mean, 1e-12)
            # Identity-level bootstrap
            unique_ids = np.array(sorted(adj.keys()))
            # Pre-group edges by endpoint identity for fast induced subgraph
            edges_by_id = {int(u): [] for u in unique_ids}
            for e in edges:
                edges_by_id[e[0]].append(e)
                if e[1] != e[0]: edges_by_id[e[1]].append(e)
            rng = np.random.default_rng(42)
            cc_boots, meas_boots = [], []
            n_id = len(unique_ids)
            for _ in range(N_BOOT):
                sampled = set(int(x) for x in rng.choice(unique_ids, n_id, replace=True))
                sub_edges = [e for e in edges if e[0] in sampled and e[1] in sampled]
                m_b, _ = reciprocity_edges(sub_edges)
                meas_boots.append(m_b)
                cc_boots.append((m_b - null_mean) / max(1 - null_mean, 1e-12))
            cc_ci = (float(np.percentile(cc_boots, 2.5)), float(np.percentile(cc_boots, 97.5)))
            meas_ci = (float(np.percentile(meas_boots, 2.5)), float(np.percentile(meas_boots, 97.5)))
            cell_out[str(k)] = {
                'n_edges': n_edges,
                'measured': meas, 'measured_ci95_identity_boot': meas_ci,
                'null_mean': null_mean,
                'chance_corrected': cc_point,
                'chance_corrected_ci95_identity_boot': cc_ci,
            }
            print(f"  k={k}  edges={n_edges}  meas={meas:.4f}  null={null_mean:.4f}  "
                  f"cc={cc_point:.4f} CI_ID=[{cc_ci[0]:.4f},{cc_ci[1]:.4f}]", flush=True)
        out[cell] = cell_out
    out_path = RESULTS / 'reciprocity_identity_boot.json'
    with open(out_path, 'w') as f: json.dump(out, f, indent=2)
    print(f"\nSaved → {out_path}")


if __name__ == '__main__':
    main()
