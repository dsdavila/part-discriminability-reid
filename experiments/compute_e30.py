"""
E30 — ΔAP-weighted decomposition + confusion reciprocity.

Part A: Per identity, partition its total ΔAP between D-shared and B-only
        confuser identities. Use the E25 attribution: for an intruder item
        at rank p, its contribution to ΔAP for a positive at rank r_j > p is
        (1/m) * j / (r_j * (r_j - 1)). Sum per confuser identity across the
        identity's queries.

Part B: Build directed confusion graph (identity-level) per cell. Edge i→j
        when identity j appears in identity i's top-k gallery neighbourhood
        (aggregated across i's items). Measure reciprocity and compare to a
        degree-preserving configuration-model null.
"""
import json, time
import numpy as np
from pathlib import Path
from collections import defaultdict

CACHE_DIR = Path('/tmp/e12_feature_cache')
RESULTS   = Path('/data/papers/low_dimensional_reid/results')
K_LIST    = [10, 50]
N_BOOT    = 500
N_SWAPS_FACTOR = 20       # number of edge swaps = 20 * |E|
N_NULL_DRAWS = 20
RNG       = np.random.default_rng(42)

CELLS = {
    'A': {'file': 'market_test.npz',           'n_q': 3368},
    'C': {'file': 'bot_msmt_seed42_market.npz','n_q': 3368},
    'D': {'file': 'bot_msmt_seed42_msmt.npz',  'n_q': 11659},
    'B': {'file': 'msmt17_test.npz',           'n_q': 11659},
}
PAIRS = [('MSMT',   'D', 'B'),
         ('Market', 'A', 'C')]


def load_cell(fname, n_q):
    d = np.load(CACHE_DIR / fname)
    emb = d['emb'].astype(np.float32)
    pids = d['pids']; cids = d['cids']
    return (emb[:n_q], pids[:n_q], cids[:n_q],
            emb[n_q:], pids[n_q:], cids[n_q:])


# ── Part A: ΔAP-weighted decomposition ───────────────────────────────────────

def per_query_dap_by_intruder_id(q_emb, q_pids, q_cids, g_emb, g_pids, g_cids):
    """Yields (q_idx, q_pid, actual_ap, dap_by_intruder_id dict)."""
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
        pos_ranks = np.where(pos_mask)[0] + 1   # 1-indexed
        j_arr    = np.arange(1, m+1)
        ap       = float((j_arr / pos_ranks).mean())

        # Precompute per-position cost term for the intruder's rank:
        # For each positive j at rank r_j, contribution of each intruder at rank p<r_j
        # is (1/m) * j / (r_j * (r_j - 1)).
        # Equivalently: for each intruder at rank p, its ΔAP contribution is
        # (1/m) * sum_{j: r_j > p} j / (r_j * (r_j - 1)).
        if m == 0 or pos_ranks[-1] <= 1:
            results.append((i, int(qp), ap, {}))
            continue
        # Additive attribution: contrib(p) = (1/m) * sum_{i: r_i > p} 1/r_i
        # This sums exactly to ΔAP = 1 - AP (verified analytically).
        term = (1.0 / pos_ranks) / m   # per-positive term
        cumsum_from_above = np.cumsum(term[::-1])[::-1]
        csfa_ext = np.concatenate([cumsum_from_above, [0.0]])
        intr_pos = np.where(~pos_mask)[0] + 1
        first_j_idx = np.searchsorted(pos_ranks, intr_pos, side='right')
        contrib = csfa_ext[first_j_idx]

        # Group by intruder identity
        intr_pids = valid_g_pids[~pos_mask]
        dap_by_intr = defaultdict(float)
        for cid, c in zip(intr_pids, contrib):
            dap_by_intr[int(cid)] += float(c)
        results.append((i, int(qp), ap, dict(dap_by_intr)))
    return results


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
    unique_ids = np.unique(g_pids)
    nbr_pids_k = g_pids[topk_idx[:, :k]]
    im_mask    = (nbr_pids_k != g_pids[:, None])
    out = {}
    for uid in unique_ids:
        mask = (g_pids == uid)
        intruders = nbr_pids_k[mask][im_mask[mask]]
        out[int(uid)] = set(int(x) for x in intruders)
    return out


def partA_decompose(eval_name, cell_m1, cell_m2):
    print(f"\n--- Part A: {eval_name}  {cell_m1} (ID) vs {cell_m2} (OOD) ---", flush=True)
    d1 = np.load(CACHE_DIR / CELLS[cell_m1]['file'])
    d2 = np.load(CACHE_DIR / CELLS[cell_m2]['file'])
    n_q = CELLS[cell_m2]['n_q']

    # Gallery (shared across models in terms of PID/CID — only embeddings differ)
    g1_pids = d1['pids'][n_q:]; g1_cids = d1['cids'][n_q:]
    g2_pids = d2['pids'][n_q:]; g2_cids = d2['cids'][n_q:]
    assert np.array_equal(g1_pids, g2_pids) and np.array_equal(g1_cids, g2_cids)
    g_pids = g2_pids; g_cids = g2_cids

    # Build D-intruder sets per identity (gallery top-k under M1)
    print(f"  computing top-{max(K_LIST)} intruder sets under {cell_m1}...", flush=True)
    G1 = d1['emb'][n_q:].astype(np.float32)
    top_m1 = topk_neighbours(G1, g_pids, g_cids, max(K_LIST))

    # Compute per-query ΔAP attribution under M2 (OOD model, the one with the gap)
    print(f"  computing per-query ΔAP under {cell_m2}...", flush=True)
    q_emb = d2['emb'][:n_q].astype(np.float32); q_pids = d2['pids'][:n_q]; q_cids = d2['cids'][:n_q]
    G2 = d2['emb'][n_q:].astype(np.float32)
    attr = per_query_dap_by_intruder_id(q_emb, q_pids, q_cids, G2, g_pids, g_cids)

    results_by_k = {}
    for k in K_LIST:
        set_m1 = per_identity_sets(top_m1, g_pids, k)

        total_shared = 0.0
        total_bonly  = 0.0
        total_dap    = 0.0
        n_counted    = 0
        n_shared_items = 0
        n_bonly_items  = 0
        shared_vals = []
        bonly_vals  = []
        for (i, qp, ap, dap_dict) in attr:
            shared_set = set_m1.get(qp, set())
            dap_q = 1.0 - ap
            total_dap += dap_q
            share_s = 0.0
            share_b = 0.0
            for cid, val in dap_dict.items():
                if cid in shared_set:
                    share_s += val
                    n_shared_items += 1
                    shared_vals.append(val)
                else:
                    share_b += val
                    n_bonly_items += 1
                    bonly_vals.append(val)
            total_shared += share_s
            total_bonly  += share_b
            n_counted    += 1

        shared_vals = np.array(shared_vals); bonly_vals = np.array(bonly_vals)
        row = {
            'n_queries': n_counted,
            'total_sigma_dap': total_dap,
            'share_dap_D_shared': total_shared / max(total_dap, 1e-12),
            'share_dap_B_only':   total_bonly  / max(total_dap, 1e-12),
            'mean_dap_per_shared_intruder_item': float(shared_vals.mean()) if len(shared_vals) else 0.0,
            'mean_dap_per_bonly_intruder_item':  float(bonly_vals.mean())  if len(bonly_vals) else 0.0,
            'n_shared_intruder_items': int(n_shared_items),
            'n_bonly_intruder_items':  int(n_bonly_items),
        }
        row['sum_shares_sanity'] = row['share_dap_D_shared'] + row['share_dap_B_only']
        print(f"  k={k} share Σ ΔAP: shared={row['share_dap_D_shared']:.4f}  "
              f"B-only={row['share_dap_B_only']:.4f}  (sum={row['sum_shares_sanity']:.4f}, "
              f"should ≈1.0)  "
              f"mean/intr shared={row['mean_dap_per_shared_intruder_item']:.6f}  "
              f"B-only={row['mean_dap_per_bonly_intruder_item']:.6f}  "
              f"items: shared={n_shared_items} B-only={n_bonly_items}", flush=True)
        results_by_k[str(k)] = row
    return results_by_k


# ── Part B: Reciprocity ──────────────────────────────────────────────────────

def build_identity_graph(topk_idx, g_pids, k):
    """Return edge set: dict pid_i -> set of pid_j appearing in i's top-k gallery items' neighbourhoods (minus same-pid)."""
    unique_ids = np.unique(g_pids)
    nbr_pids_k = g_pids[topk_idx[:, :k]]
    im_mask    = (nbr_pids_k != g_pids[:, None])
    adj = {int(uid): set() for uid in unique_ids}
    for i, uid in enumerate(g_pids):
        row = nbr_pids_k[i]
        row_mask = im_mask[i]
        for j in row[row_mask]:
            adj[int(uid)].add(int(j))
    return adj


def reciprocity(adj):
    edges = set()
    for i, outs in adj.items():
        for j in outs: edges.add((i, j))
    total = len(edges)
    if total == 0: return 0.0, 0
    recip = sum(1 for (i,j) in edges if (j,i) in edges)
    return recip / total, total


def configuration_rewire(adj, rng, n_swap_factor=N_SWAPS_FACTOR):
    """In-place edge rewiring preserving in- and out-degrees. Returns new adj."""
    edges = [(i, j) for i, outs in adj.items() for j in outs]
    n_edges = len(edges)
    n_swap = n_swap_factor * n_edges
    adj_sets = {k: set(v) for k, v in adj.items()}   # out-neighbours
    for _ in range(n_swap):
        if n_edges < 2: break
        a = rng.integers(0, n_edges)
        b = rng.integers(0, n_edges)
        if a == b: continue
        i1, j1 = edges[a]
        i2, j2 = edges[b]
        # Rewire (i1→j1), (i2→j2) → (i1→j2), (i2→j1), only if valid
        if j2 in adj_sets[i1] or j1 in adj_sets[i2]: continue
        if j2 == i1 or j1 == i2: continue
        adj_sets[i1].remove(j1); adj_sets[i1].add(j2)
        adj_sets[i2].remove(j2); adj_sets[i2].add(j1)
        edges[a] = (i1, j2); edges[b] = (i2, j1)
    return adj_sets


def partB_reciprocity_cell(cell):
    print(f"\n--- Part B: {cell} ---", flush=True)
    d = np.load(CACHE_DIR / CELLS[cell]['file'])
    n_q = CELLS[cell]['n_q']
    g_emb  = d['emb'][n_q:].astype(np.float32)
    g_pids = d['pids'][n_q:]; g_cids = d['cids'][n_q:]
    k_max = max(K_LIST)
    print(f"  top-{k_max}...", flush=True)
    topk = topk_neighbours(g_emb, g_pids, g_cids, k_max)

    out = {}
    for k in K_LIST:
        adj = build_identity_graph(topk, g_pids, k)
        meas, n_edges = reciprocity(adj)
        # Null via configuration-model rewiring
        rng = np.random.default_rng(42)
        null_vals = []
        for _ in range(N_NULL_DRAWS):
            adj_null = configuration_rewire(adj, rng)
            r_null, _ = reciprocity(adj_null)
            null_vals.append(r_null)
        null_mean = float(np.mean(null_vals))
        null_std  = float(np.std(null_vals))
        ratio = meas / max(null_mean, 1e-12)
        row = {'n_edges': int(n_edges),
               'reciprocity_measured': meas,
               'reciprocity_null_mean': null_mean,
               'reciprocity_null_std':  null_std,
               'ratio_to_null': ratio}
        print(f"  k={k}  edges={n_edges}  measured={meas:.4f}  null={null_mean:.4f}±{null_std:.4f}  ratio={ratio:.2f}×", flush=True)
        out[str(k)] = row
    return out


def main():
    partA = {}
    for name, m1, m2 in PAIRS:
        partA[name] = partA_decompose(name, m1, m2)

    partB = {}
    for cell in ['A', 'C', 'D', 'B']:
        partB[cell] = partB_reciprocity_cell(cell)

    out = {'partA': partA, 'partB': partB}
    out_path = RESULTS / 'e30_damage_and_reciprocity.json'
    with open(out_path, 'w') as f: json.dump(out, f, indent=2)
    print(f"\nSaved → {out_path}", flush=True)


if __name__ == '__main__':
    main()
