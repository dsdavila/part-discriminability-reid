"""
Re-run E29, E30 Part A, E30 Part B, E30 symmetric split and E29 per-ID ceiling
with no-REA Market checkpoint for cells A' and B'. Cells C and D unchanged
(already no-REA MSMT seed 42).
"""
import json, time, importlib.util
import numpy as np
from pathlib import Path
from collections import defaultdict
from scipy.stats import spearmanr

CACHE_DIR = Path('/tmp/e12_feature_cache')
RESULTS   = Path('/data/papers/low_dimensional_reid/results')
K_LIST    = [10, 50]
N_PERM    = 100
N_BOOT    = 1000
N_SWAPS_FACTOR = 20
N_NULL_DRAWS = 20

# NOTE: A' and B' now point at the no-REA Market caches.
CELLS = {
    "A'": {'file': 'market_test_noREA.npz',    'n_q': 3368,  'eval': 'market'},
    'C':  {'file': 'bot_msmt_seed42_market.npz','n_q': 3368,  'eval': 'market'},
    'D':  {'file': 'bot_msmt_seed42_msmt.npz', 'n_q': 11659, 'eval': 'msmt'},
    "B'": {'file': 'msmt17_test_noREA.npz',    'n_q': 11659, 'eval': 'msmt'},
}
PAIRS      = [('MSMT', 'D', "B'"), ('Market', "A'", 'C')]
ID_PAIRS   = [('MSMT', 'D', "B'"), ('Market', "A'", 'C')]


def load_cell(name):
    cfg = CELLS[name]
    d = np.load(CACHE_DIR / cfg['file'])
    n_q = cfg['n_q']
    emb = d['emb'].astype(np.float32)
    return (emb[:n_q], d['pids'][:n_q], d['cids'][:n_q],
            emb[n_q:], d['pids'][n_q:], d['cids'][n_q:])


# ─── Common utilities ────────────────────────────────────────────────────────

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
            order = np.argsort(-sims[local, top_idx])
            out[gi] = top_idx[order]
    return out


def per_identity_sets(topk_idx, g_pids, k):
    unique_ids = np.unique(g_pids)
    nbr_pids_k = g_pids[topk_idx[:, :k]]
    im_mask = (nbr_pids_k != g_pids[:, None])
    out = {}
    for uid in unique_ids:
        mask = (g_pids == uid)
        intruders = nbr_pids_k[mask][im_mask[mask]]
        out[int(uid)] = set(int(x) for x in intruders)
    return out


def per_query_dap(q_emb, q_pids, q_cids, g_emb, g_pids, g_cids):
    sims = q_emb @ g_emb.T
    results = []
    for i in range(len(q_emb)):
        qp, qc = q_pids[i], q_cids[i]
        valid = ~((g_pids == qp) & (g_cids == qc))
        rel = (g_pids == qp) & valid
        if rel.sum() == 0: continue
        order = np.argsort(-sims[i][valid])
        rel_v = rel[valid][order]
        if rel_v.sum() == 0: continue
        valid_g_pids = g_pids[valid][order]
        pos_mask = rel_v
        m = int(pos_mask.sum())
        pos_ranks = np.where(pos_mask)[0] + 1
        j_arr = np.arange(1, m+1)
        ap = float((j_arr / pos_ranks).mean())
        term = (1.0 / pos_ranks) / m
        csfa = np.cumsum(term[::-1])[::-1]
        csfa_ext = np.concatenate([csfa, [0.0]])
        intr_pos = np.where(~pos_mask)[0] + 1
        first_j_idx = np.searchsorted(pos_ranks, intr_pos, side='right')
        contrib = csfa_ext[first_j_idx]
        intr_pids = valid_g_pids[~pos_mask]
        dap_by_intr = defaultdict(float)
        for cid, c in zip(intr_pids, contrib):
            dap_by_intr[int(cid)] += float(c)
        results.append((i, int(qp), ap, dict(dap_by_intr)))
    return results


# ─── E29 overlap + per-identity ceiling ──────────────────────────────────────

def e29_pair(eval_name, c1, c2):
    print(f"\n[E29] {eval_name}: {c1} vs {c2}", flush=True)
    G1, pids1, cids1, _, _, _ = (None,)*6
    _, _, _, G1, pids1, cids1 = load_cell(c1)
    _, _, _, G2, pids2, cids2 = load_cell(c2)
    assert np.array_equal(pids1, pids2) and np.array_equal(cids1, cids2)
    g_pids = pids1; g_cids = cids1
    k_max = max(K_LIST)
    t0 = time.time()
    print(f"  top-{k_max} under {c1} and {c2}...", flush=True)
    top1 = topk_neighbours(G1, g_pids, g_cids, k_max)
    top2 = topk_neighbours(G2, g_pids, g_cids, k_max)
    print(f"  done. t={(time.time()-t0)/60:.1f}m", flush=True)

    unique_ids = np.unique(g_pids)
    rng = np.random.default_rng(42)
    out = {'eval': eval_name, 'm1': c1, 'm2': c2, 'per_k': {}}
    for k in K_LIST:
        set1 = per_identity_sets(top1, g_pids, k)
        set2 = per_identity_sets(top2, g_pids, k)
        jacc, null_j, shared_frac, frac_ceiling = [], [], [], []
        for uid in set1:
            a, b = set1[uid], set2[uid]
            if not a or not b: continue
            inter = len(a & b)
            uni = len(a | b)
            jacc.append(inter / uni)
            shared_frac.append(inter / max(len(b),1))
            frac_ceiling.append(inter / max(min(len(a), len(b)),1))
            # Permutation null
            cands = np.array([int(x) for x in unique_ids if int(x) != uid])
            vals = []
            m = min(len(b), len(cands))
            for _ in range(N_PERM):
                s = set(int(x) for x in rng.choice(cands, m, replace=False))
                vals.append(len(a & s) / len(a | s) if (a | s) else 0.0)
            null_j.append(float(np.mean(vals)))
        jacc = np.array(jacc); null_j = np.array(null_j)
        shared_frac = np.array(shared_frac); frac_ceiling = np.array(frac_ceiling)
        row = {
            'n_matched_ids': int(len(jacc)),
            'jaccard_mean': float(jacc.mean()),
            'null_jaccard_mean': float(null_j.mean()),
            'ratio_to_null': float(jacc.mean() / max(null_j.mean(), 1e-12)),
            'shared_frac_of_m2_mean': float(shared_frac.mean()),
            'per_identity_frac_of_ceiling_mean': float(frac_ceiling.mean()),
            'per_identity_frac_of_ceiling_median': float(np.median(frac_ceiling)),
        }
        print(f"  k={k} jacc={row['jaccard_mean']:.4f} null={row['null_jaccard_mean']:.4f} "
              f"ratio={row['ratio_to_null']:.1f}× shared/m2={row['shared_frac_of_m2_mean']:.4f} "
              f"per-ID_ceiling={row['per_identity_frac_of_ceiling_mean']:.4f}", flush=True)
        out['per_k'][str(k)] = row
    return out


# ─── E30 Part A ──────────────────────────────────────────────────────────────

def e30_partA(eval_name, c_id, c_ood):
    """Partition OOD cell's ΔAP into shared (with ID) vs OOD-only."""
    print(f"\n[E30-A OOD] {eval_name}: {c_id}(ID) vs {c_ood}(OOD)", flush=True)
    _, _, _, G_id, g_pids_id, g_cids_id = load_cell(c_id)
    q_emb, q_pids, q_cids, G_ood, g_pids_ood, g_cids_ood = load_cell(c_ood)
    assert np.array_equal(g_pids_id, g_pids_ood)
    g_pids = g_pids_id; g_cids = g_cids_id
    print(f"  topk under {c_id}...", flush=True)
    top_id = topk_neighbours(G_id, g_pids, g_cids, max(K_LIST))
    print(f"  per-query ΔAP under {c_ood}...", flush=True)
    attr = per_query_dap(q_emb, q_pids, q_cids, G_ood, g_pids, g_cids)
    out = {}
    for k in K_LIST:
        set_id = per_identity_sets(top_id, g_pids, k)
        total_shared = total_ood_only = total_dap = 0.0
        n_shared_items = n_ood_only_items = 0
        shared_vals, ood_vals = [], []
        for (i, qp, ap, dap_dict) in attr:
            shared_set = set_id.get(qp, set())
            total_dap += (1 - ap)
            for cid, val in dap_dict.items():
                if cid in shared_set:
                    total_shared += val; shared_vals.append(val); n_shared_items += 1
                else:
                    total_ood_only += val; ood_vals.append(val); n_ood_only_items += 1
        row = {
            'total_sigma_dap': total_dap,
            'share_dap_shared': total_shared / max(total_dap, 1e-12),
            'share_dap_OODonly': total_ood_only / max(total_dap, 1e-12),
            'mean_dap_per_shared_item': float(np.mean(shared_vals)) if shared_vals else 0.0,
            'mean_dap_per_OODonly_item': float(np.mean(ood_vals)) if ood_vals else 0.0,
            'n_shared_items': n_shared_items, 'n_OODonly_items': n_ood_only_items,
        }
        row['sum_sanity'] = row['share_dap_shared'] + row['share_dap_OODonly']
        print(f"  k={k} shared={row['share_dap_shared']:.4f} OOD-only={row['share_dap_OODonly']:.4f} "
              f"sum={row['sum_sanity']:.4f}", flush=True)
        out[str(k)] = row
    return out


def e30_partA_symmetric(eval_name, c_id, c_ood):
    """Partition ID cell's ΔAP into shared (with OOD) vs ID-only."""
    print(f"\n[E30-A ID] {eval_name}: {c_id}(ID) shared-set from {c_ood}(OOD)", flush=True)
    q_emb, q_pids, q_cids, G_id, g_pids_id, g_cids_id = load_cell(c_id)
    _, _, _, G_ood, g_pids_ood, g_cids_ood = load_cell(c_ood)
    g_pids = g_pids_id; g_cids = g_cids_id
    print(f"  topk under {c_ood} (for classification)...", flush=True)
    top_ood = topk_neighbours(G_ood, g_pids, g_cids, max(K_LIST))
    print(f"  per-query ΔAP under {c_id}...", flush=True)
    attr = per_query_dap(q_emb, q_pids, q_cids, G_id, g_pids, g_cids)
    out = {}
    for k in K_LIST:
        set_ood = per_identity_sets(top_ood, g_pids, k)
        total_shared = total_id_only = total_dap = 0.0
        for (i, qp, ap, dap_dict) in attr:
            shared_set = set_ood.get(qp, set())
            total_dap += (1 - ap)
            for cid, val in dap_dict.items():
                if cid in shared_set: total_shared += val
                else:                 total_id_only += val
        row = {
            'total_sigma_dap': total_dap,
            'share_dap_shared': total_shared / max(total_dap, 1e-12),
            'share_dap_IDonly': total_id_only / max(total_dap, 1e-12),
        }
        row['sum_sanity'] = row['share_dap_shared'] + row['share_dap_IDonly']
        print(f"  k={k} shared={row['share_dap_shared']:.4f} ID-only={row['share_dap_IDonly']:.4f} "
              f"sum={row['sum_sanity']:.4f}", flush=True)
        out[str(k)] = row
    return out


# ─── E30 Part B — reciprocity ─────────────────────────────────────────────────

def build_graph(topk_idx, g_pids, k):
    unique_ids = np.unique(g_pids)
    nbr = g_pids[topk_idx[:, :k]]
    im = (nbr != g_pids[:, None])
    adj = {int(u): set() for u in unique_ids}
    for i, u in enumerate(g_pids):
        for j in nbr[i][im[i]]:
            adj[int(u)].add(int(j))
    return adj


def reciprocity(adj):
    edges = set()
    for i, outs in adj.items():
        for j in outs: edges.add((i, j))
    if not edges: return 0.0, 0
    recip = sum(1 for e in edges if (e[1], e[0]) in edges)
    return recip / len(edges), len(edges)


def config_rewire(adj, rng, factor=N_SWAPS_FACTOR):
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


def partB_cell(cell):
    print(f"\n[E30-B] {cell}", flush=True)
    _, _, _, G, g_pids, g_cids = load_cell(cell)
    print(f"  top-{max(K_LIST)}...", flush=True)
    topk = topk_neighbours(G, g_pids, g_cids, max(K_LIST))
    out = {}
    for k in K_LIST:
        adj = build_graph(topk, g_pids, k)
        meas, n_edges = reciprocity(adj)
        rng = np.random.default_rng(42)
        null_vals = []
        for _ in range(N_NULL_DRAWS):
            null_vals.append(reciprocity(config_rewire(adj, rng))[0])
        null_mean = float(np.mean(null_vals))
        chance_corr = (meas - null_mean) / max(1 - null_mean, 1e-12)
        row = {'n_edges': int(n_edges), 'measured': meas, 'null_mean': null_mean,
               'null_std': float(np.std(null_vals)),
               'ratio_to_null': meas / max(null_mean, 1e-12),
               'chance_corrected': chance_corr}
        print(f"  k={k} edges={n_edges} measured={meas:.4f} null={null_mean:.4f} "
              f"ratio={row['ratio_to_null']:.2f}× chance-corr={chance_corr:.4f}", flush=True)
        out[str(k)] = row
    return out


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    results = {'cells_used': {k: v['file'] for k, v in CELLS.items()}}

    results['e29'] = {}
    for name, m1, m2 in PAIRS:
        results['e29'][name] = e29_pair(name, m1, m2)

    results['e30_partA_OODside'] = {}
    results['e30_partA_IDside'] = {}
    for name, c_id, c_ood in ID_PAIRS:
        results['e30_partA_OODside'][name] = e30_partA(name, c_id, c_ood)
        results['e30_partA_IDside'][name]  = e30_partA_symmetric(name, c_id, c_ood)

    results['e30_partB'] = {}
    for cell in ["A'", 'C', 'D', "B'"]:
        results['e30_partB'][cell] = partB_cell(cell)

    out = RESULTS / 'noREA_rerun_results.json'
    with open(out, 'w') as f: json.dump(results, f, indent=2)
    print(f"\nSaved → {out}", flush=True)


if __name__ == '__main__':
    main()
