"""
E28 Part 2 — Neighbourhood composition (geometry side).

For each test identity, for each of its gallery items, take the k nearest gallery
neighbours at k=10 and k=50, excluding same-pid-same-cam (canonical junk).

Per identity:
  - purity:                fraction of neighbours that are same identity
  - n_distinct_intruders:  count of distinct other identities among neighbours
                           (normalised by intruder slots available)
  - intruder_top1_share:   fraction of intruder slots held by most frequent intruder
  - intruder_gini:         Gini of intruder identity counts

Per cell: median + IQR of each. Per-identity Spearman vs AP with bootstrap CIs.
Pair within eval set only (A vs C on Market, D vs B on MSMT).
"""
import json, time
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr
from collections import defaultdict

CACHE_DIR = Path('/tmp/e12_feature_cache')
RESULTS   = Path('/data/papers/low_dimensional_reid/results')
N_BOOT    = 500
RNG       = np.random.default_rng(42)

CELLS = {
    'A': {'file': 'market_test.npz',           'n_q': 3368},
    'C': {'file': 'bot_msmt_seed42_market.npz','n_q': 3368},
    'D': {'file': 'bot_msmt_seed42_msmt.npz',  'n_q': 11659},
    'B': {'file': 'msmt17_test.npz',           'n_q': 11659},
}
K_LIST = [10, 50]


def load_cell(fname, n_q):
    d = np.load(CACHE_DIR / fname)
    emb = d['emb'].astype(np.float32)
    pids = d['pids']; cids = d['cids']
    return (emb[:n_q], pids[:n_q], cids[:n_q],
            emb[n_q:], pids[n_q:], cids[n_q:])


def per_query_ap(q_emb, q_pids, q_cids, g_emb, g_pids, g_cids):
    sims = q_emb @ g_emb.T
    aps, qpids_out = [], []
    for i in range(len(q_emb)):
        qp, qc = q_pids[i], q_cids[i]
        valid = ~((g_pids == qp) & (g_cids == qc))
        rel = (g_pids == qp) & valid
        if rel.sum() == 0: continue
        order = np.argsort(-sims[i][valid])
        rel_v = rel[valid][order]
        if rel_v.sum() == 0: continue
        pos = np.where(rel_v)[0]
        precisions = (np.arange(len(pos)) + 1) / (pos + 1)
        aps.append(float(precisions.mean()))
        qpids_out.append(int(qp))
    return np.array(aps, dtype=np.float64), np.array(qpids_out, dtype=np.int64)


def gini(x):
    x = np.sort(np.asarray(x, dtype=np.float64))
    n = len(x)
    if n == 0 or x.sum() == 0: return 0.0
    cum = np.cumsum(x)
    return float((n + 1 - 2 * cum.sum() / cum[-1]) / n)


def topk_mask_chunked(G, k_max, exclude_mask_fn, chunk=2048):
    """
    For each row g[i] in G (as query), return the indices of its top-k neighbours
    in G, with per-row exclusion. Returns indices array (N, k_max).
    exclude_mask_fn(i) returns a bool array of length N (True = exclude).
    """
    N, D = G.shape
    out = np.full((N, k_max), -1, dtype=np.int64)
    for start in range(0, N, chunk):
        end = min(N, start + chunk)
        sims = G[start:end] @ G.T   # (chunk, N)
        for local, gi in enumerate(range(start, end)):
            exc = exclude_mask_fn(gi)
            sims[local, exc] = -2.0
            # Partial sort for top-k_max
            top_idx = np.argpartition(-sims[local], k_max)[:k_max]
            order   = np.argsort(-sims[local, top_idx])
            out[gi] = top_idx[order]
    return out


def per_id_neighbourhood_stats(G, g_pids, g_cids, k_values):
    """For each gallery identity, aggregate purity/distinct_intruders/top1/gini across k."""
    N = len(G)
    k_max = max(k_values)
    unique_ids = np.unique(g_pids)
    print(f"    pre-computing top-{k_max} for N={N} ...", flush=True)
    t0 = time.time()
    def exc(i): return (g_pids == g_pids[i]) & (g_cids == g_cids[i])
    topk_idx = topk_mask_chunked(G, k_max, exc)
    print(f"      done in {(time.time()-t0)/60:.1f}m", flush=True)
    nbr_pids = g_pids[topk_idx]      # (N, k_max)
    same_mask_full = (nbr_pids == g_pids[:, None])

    results = {}
    for k in k_values:
        nbr_k = nbr_pids[:, :k]
        same_k = same_mask_full[:, :k]
        per_item_purity = same_k.mean(axis=1)
        agg = {}
        for uid in unique_ids:
            mask_id = (g_pids == uid)
            # Purity: mean fraction of same-id neighbours across the identity's items
            purity = float(per_item_purity[mask_id].mean())
            # Collect intruder pids (non-same-ID neighbours) for all items of this ID
            nbr_slice = nbr_k[mask_id]
            same_slice = same_k[mask_id]
            intruders = nbr_slice[~same_slice]
            intruder_slots = int((~same_slice).sum())
            if intruder_slots == 0:
                agg[int(uid)] = {'purity': purity, 'n_distinct_intruders_norm': 0.0,
                                 'intruder_top1_share': 0.0, 'intruder_gini': 0.0,
                                 'intruder_slots': 0}
                continue
            uniq_int, counts = np.unique(intruders, return_counts=True)
            n_distinct = len(uniq_int)
            n_distinct_norm = n_distinct / intruder_slots
            top1_share = float(counts.max() / intruder_slots)
            agg[int(uid)] = {'purity': purity,
                             'n_distinct_intruders_norm': n_distinct_norm,
                             'intruder_top1_share': top1_share,
                             'intruder_gini': gini(counts),
                             'intruder_slots': intruder_slots}
        results[k] = agg
    return results


def main():
    out = {}
    for cell, cfg in CELLS.items():
        print(f"\n=== Cell {cell} ===", flush=True)
        q_emb, q_pids, q_cids, g_emb, g_pids, g_cids = load_cell(cfg['file'], cfg['n_q'])
        ap_arr, qpid = per_query_ap(q_emb, q_pids, q_cids, g_emb, g_pids, g_cids)
        id_to_aps = defaultdict(list)
        for p, a in zip(qpid, ap_arr): id_to_aps[int(p)].append(a)
        per_id_ap = {pid: float(np.mean(v)) for pid,v in id_to_aps.items()}

        print(f"  gallery N={len(g_emb)}", flush=True)
        stats_by_k = per_id_neighbourhood_stats(g_emb, g_pids, g_cids, K_LIST)

        cell_out = {'n_gallery': int(len(g_emb)), 'per_k': {}}
        unique_ids = np.unique(g_pids)
        for k, agg in stats_by_k.items():
            # Build arrays aligned
            matched = [int(u) for u in unique_ids if int(u) in per_id_ap]
            pur   = np.array([agg[u]['purity']                   for u in matched])
            nnd   = np.array([agg[u]['n_distinct_intruders_norm'] for u in matched])
            top1  = np.array([agg[u]['intruder_top1_share']       for u in matched])
            gi_   = np.array([agg[u]['intruder_gini']             for u in matched])
            ap    = np.array([per_id_ap[u]                        for u in matched])

            def quartiles(x):
                return [float(np.percentile(x, 25)), float(np.median(x)), float(np.percentile(x, 75))]
            def rho_ci(x, y, n_boot=N_BOOT, seed=42):
                rng = np.random.default_rng(seed)
                point = float(spearmanr(x, y).statistic)
                boot = []
                n = len(x)
                for _ in range(n_boot):
                    idx = rng.integers(0, n, size=n)
                    try: boot.append(float(spearmanr(x[idx], y[idx]).statistic))
                    except Exception: pass
                return point, [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]

            per_k = {
                'quartiles': {
                    'purity': quartiles(pur),
                    'n_distinct_intruders_norm': quartiles(nnd),
                    'intruder_top1_share': quartiles(top1),
                    'intruder_gini': quartiles(gi_),
                },
                'spearman_vs_ap': {},
            }
            for name, arr in [('purity', pur), ('n_distinct_intruders_norm', nnd),
                              ('intruder_top1_share', top1), ('intruder_gini', gi_)]:
                r, ci = rho_ci(arr, ap)
                per_k['spearman_vs_ap'][name] = {'rho': r, 'ci95': ci}
                print(f"  k={k} {name:<28} med={np.median(arr):.4f}  ρ(AP)={r:+.3f}  CI=[{ci[0]:+.3f},{ci[1]:+.3f}]", flush=True)
            cell_out['per_k'][str(k)] = per_k
        out[cell] = cell_out

    out_path = RESULTS / 'e28_part2_neighbourhood_composition.json'
    with open(out_path, 'w') as f: json.dump(out, f, indent=2)
    print(f"\nSaved → {out_path}", flush=True)


if __name__ == '__main__':
    main()
