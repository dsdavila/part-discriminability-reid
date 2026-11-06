"""
Bootstrap CIs for the three asymmetry constructs, no-REA cells:
  (a) ceiling-corrected per-ID confuser overlap (E29/E30-A ID-side equivalence)
  (b) Jaccard vs random-draw null
  (c) reciprocity vs degree-preserving null
"""
import json, time
import numpy as np
from pathlib import Path

CACHE_DIR = Path('/tmp/e12_feature_cache')
RESULTS   = Path('/data/papers/low_dimensional_reid/results')
K_LIST = [10, 50]
N_BOOT = 1000
N_PERM = 50
N_REWIRE = 20
RNG = np.random.default_rng(42)

CELLS = {
    "A'": {'file': 'market_test_noREA.npz',    'n_q': 3368},
    'C':  {'file': 'bot_msmt_seed42_market.npz','n_q': 3368},
    'D':  {'file': 'bot_msmt_seed42_msmt.npz', 'n_q': 11659},
    "B'": {'file': 'msmt17_test_noREA.npz',    'n_q': 11659},
}
PAIRS = [('MSMT', 'D', "B'"), ('Market', "A'", 'C')]


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


def sets_per_id(topk_idx, g_pids, k):
    unique_ids = np.unique(g_pids)
    nbr = g_pids[topk_idx[:, :k]]
    im = (nbr != g_pids[:, None])
    out = {}
    for uid in unique_ids:
        mask = (g_pids == uid)
        intruders = nbr[mask][im[mask]]
        out[int(uid)] = set(int(x) for x in intruders)
    return out


# ─── Construct (a): ceiling-corrected per-ID overlap ─────────────────────────

def per_id_ceiling_values(sets1, sets2):
    vals = []
    for uid in sets1:
        a, b = sets1[uid], sets2[uid]
        if not a or not b: continue
        ceil = min(len(a), len(b))
        vals.append(len(a & b) / ceil)
    return np.array(vals)


# ─── Construct (b): Jaccard vs random-draw null ──────────────────────────────

def jaccard_vals_and_null(sets1, sets2, all_ids, n_perm=N_PERM, rng=None):
    rng = rng or np.random.default_rng(42)
    j, nj = [], []
    all_ids_arr = np.array([int(x) for x in all_ids])
    for uid in sets1:
        a, b = sets1[uid], sets2[uid]
        if not a or not b: continue
        inter = len(a & b); uni = len(a | b)
        j.append(inter / uni)
        cands = all_ids_arr[all_ids_arr != uid]
        m = min(len(b), len(cands))
        null_js = []
        for _ in range(n_perm):
            s = set(int(x) for x in rng.choice(cands, m, replace=False))
            null_js.append(len(a & s) / len(a | s) if (a | s) else 0.0)
        nj.append(float(np.mean(null_js)))
    return np.array(j), np.array(nj)


# ─── Construct (c): reciprocity vs degree-preserving null ────────────────────

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
    if not edges: return 0.0
    recip = sum(1 for e in edges if (e[1], e[0]) in edges)
    return recip / len(edges)


def config_rewire(adj, rng, factor=20):
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


def bootstrap_scalar(values, n_boot=N_BOOT, seed=42, stat=np.mean):
    rng = np.random.default_rng(seed)
    n = len(values)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots.append(float(stat(values[idx])))
    return float(stat(values)), (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))


def main():
    results = {'pairs': {}}
    pre_topks = {}
    for name, c1, c2 in PAIRS:
        print(f"\n=== Pair {name}: {c1} vs {c2} ===", flush=True)
        G1, pids1, cids1 = load_gallery(c1)
        G2, pids2, cids2 = load_gallery(c2)
        g_pids = pids1; g_cids = cids1
        k_max = max(K_LIST)
        print(f"  topk...", flush=True)
        t0 = time.time()
        if c1 not in pre_topks:
            pre_topks[c1] = topk_neighbours(G1, g_pids, g_cids, k_max)
        if c2 not in pre_topks:
            pre_topks[c2] = topk_neighbours(G2, g_pids, g_cids, k_max)
        top1 = pre_topks[c1]; top2 = pre_topks[c2]
        print(f"    t={(time.time()-t0)/60:.1f}m", flush=True)
        unique_ids = np.unique(g_pids)
        pair_out = {'per_k': {}}
        for k in K_LIST:
            sets1 = sets_per_id(top1, g_pids, k)
            sets2 = sets_per_id(top2, g_pids, k)
            # (a) ceiling-corrected per-ID overlap
            frac_vals = per_id_ceiling_values(sets1, sets2)
            frac_mean, frac_ci = bootstrap_scalar(frac_vals)
            # (b) Jaccard vs null
            j_vals, nj_vals = jaccard_vals_and_null(sets1, sets2, unique_ids)
            def rtn(idx):
                return float(j_vals[idx].mean() / max(nj_vals[idx].mean(), 1e-12))
            # bootstrap over identity indices for ratio-to-null
            rng = np.random.default_rng(42)
            boots_rtn = []
            n = len(j_vals)
            for _ in range(N_BOOT):
                idx = rng.integers(0, n, size=n)
                boots_rtn.append(rtn(idx))
            rtn_point = float(j_vals.mean() / max(nj_vals.mean(), 1e-12))
            rtn_ci = (float(np.percentile(boots_rtn, 2.5)), float(np.percentile(boots_rtn, 97.5)))
            pair_out['per_k'][str(k)] = {
                'n_matched_ids': int(len(frac_vals)),
                'ceiling_corrected_overlap': {
                    'mean': frac_mean, 'ci95': frac_ci,
                },
                'jaccard_ratio_to_null': {
                    'mean': rtn_point, 'ci95': rtn_ci,
                    'jaccard_mean': float(j_vals.mean()),
                    'null_mean': float(nj_vals.mean()),
                },
            }
            print(f"  k={k}  ceiling-corrected overlap = {frac_mean:.4f} [{frac_ci[0]:.4f},{frac_ci[1]:.4f}]  "
                  f"Jaccard ratio-to-null = {rtn_point:.2f} [{rtn_ci[0]:.2f},{rtn_ci[1]:.2f}]", flush=True)
        results['pairs'][name] = pair_out

    # Reciprocity per cell: compute point + null rewires + bootstrap over edges
    results['reciprocity'] = {}
    for cell in ["A'", 'C', 'D', "B'"]:
        print(f"\n--- Reciprocity: {cell} ---", flush=True)
        G, g_pids, g_cids = load_gallery(cell)
        if cell not in pre_topks:
            pre_topks[cell] = topk_neighbours(G, g_pids, g_cids, max(K_LIST))
        topk = pre_topks[cell]
        cell_out = {}
        for k in K_LIST:
            adj = build_graph(topk, g_pids, k)
            meas = reciprocity_of(adj)
            # Null rewires
            rng = np.random.default_rng(42)
            null_vals = [reciprocity_of(config_rewire(adj, rng)) for _ in range(N_REWIRE)]
            null_mean = float(np.mean(null_vals))
            null_std  = float(np.std(null_vals))
            cc = (meas - null_mean) / max(1 - null_mean, 1e-12)
            # Bootstrap CI on measured reciprocity over edges (per-node resample would be more correct;
            # using edge-bootstrap as a quick sanity check)
            edges = [(i, j) for i, outs in adj.items() for j in outs]
            n = len(edges)
            edges_set = set(edges)
            rng = np.random.default_rng(42)
            cc_boots = []
            for _ in range(N_BOOT):
                idx = rng.integers(0, n, size=n)
                recip_c = sum(1 for ei in idx if (edges[ei][1], edges[ei][0]) in edges_set)
                m_b = recip_c / n
                cc_boots.append((m_b - null_mean) / max(1 - null_mean, 1e-12))
            cc_ci = (float(np.percentile(cc_boots, 2.5)), float(np.percentile(cc_boots, 97.5)))
            cell_out[str(k)] = {
                'n_edges': int(n),
                'measured': meas,
                'null_mean': null_mean, 'null_std': null_std,
                'ratio_to_null': meas / max(null_mean, 1e-12),
                'chance_corrected': cc,
                'chance_corrected_ci95': cc_ci,
            }
            print(f"  k={k}  edges={n}  measured={meas:.4f}  null={null_mean:.4f}  "
                  f"chance-corr={cc:.4f} [{cc_ci[0]:.4f},{cc_ci[1]:.4f}]", flush=True)
        results['reciprocity'][cell] = cell_out

    out_path = RESULTS / 'noREA_CIs.json'
    with open(out_path, 'w') as f: json.dump(results, f, indent=2)
    print(f"\nSaved → {out_path}", flush=True)


if __name__ == '__main__':
    main()
