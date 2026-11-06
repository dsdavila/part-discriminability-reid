"""
E28 Part 1 — Attribution concentration (AP side).

For each cell (A, B, C, D) separately — ALL queries, no AP band, no subsampling:
  - Share of ΣΔAP carried by top 1%, 5%, 10% of offenders ranked by ΔAP
  - Gini coefficient of ΔAP across offenders
  - Distinct-identity count of top offenders (normalised by # of query identities)

ΔAP := (1 - AP) per query. Offenders = queries with ΔAP > 0 (i.e., AP < 1).
Junk predicate: same-pid & same-cam, canonical.
"""
import json
import numpy as np
from pathlib import Path

CACHE_DIR = Path('/tmp/e12_feature_cache')
RESULTS   = Path('/data/papers/low_dimensional_reid/results')
RESULTS.mkdir(parents=True, exist_ok=True)

CELLS = {
    'A': {'file': 'market_test.npz',           'n_q': 3368,   'eval': 'market'},
    'C': {'file': 'bot_msmt_seed42_market.npz','n_q': 3368,   'eval': 'market'},
    'D': {'file': 'bot_msmt_seed42_msmt.npz',  'n_q': 11659,  'eval': 'msmt'},
    'B': {'file': 'msmt17_test.npz',           'n_q': 11659,  'eval': 'msmt'},
}


def load_cell(fname, n_q):
    d = np.load(CACHE_DIR / fname)
    emb = d['emb'].astype(np.float32)
    pids = d['pids']; cids = d['cids']
    return (emb[:n_q], pids[:n_q], cids[:n_q],
            emb[n_q:], pids[n_q:], cids[n_q:])


def per_query_ap(q_emb, q_pids, q_cids, g_emb, g_pids, g_cids):
    """Return (ap_arr, q_pid_arr) for every query with at least one positive."""
    sims = q_emb @ g_emb.T
    aps, pids_out = [], []
    for i in range(len(q_emb)):
        qp, qc = q_pids[i], q_cids[i]
        # Canonical junk predicate: same-pid & same-cam
        valid = ~((g_pids == qp) & (g_cids == qc))
        rel = (g_pids == qp) & valid
        if rel.sum() == 0:
            continue
        order = np.argsort(-sims[i][valid])
        rel_v = rel[valid][order]
        if rel_v.sum() == 0:
            continue
        n_rel = rel_v.sum()
        pos = np.where(rel_v)[0]
        precisions = (np.arange(len(pos)) + 1) / (pos + 1)
        aps.append(float(precisions.mean()))
        pids_out.append(int(qp))
    return np.array(aps, dtype=np.float64), np.array(pids_out, dtype=np.int64)


def gini(x):
    x = np.sort(np.asarray(x, dtype=np.float64))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return 0.0
    cum = np.cumsum(x)
    return float((n + 1 - 2 * cum.sum() / cum[-1]) / n)


def top_share(deltas, pct):
    """Share of total ΣΔAP carried by top pct% of offenders by ΔAP."""
    offenders = deltas[deltas > 0]
    if len(offenders) == 0:
        return 0.0, 0
    k = max(1, int(np.ceil(len(offenders) * pct / 100)))
    top_vals = np.sort(offenders)[-k:]
    return float(top_vals.sum() / offenders.sum()), int(k)


def top_distinct_ids(deltas, pids, pct):
    """# distinct identities among top pct% of offenders."""
    mask = deltas > 0
    off_dap = deltas[mask]
    off_pid = pids[mask]
    if len(off_dap) == 0:
        return 0
    k = max(1, int(np.ceil(len(off_dap) * pct / 100)))
    order = np.argsort(-off_dap)[:k]  # top-k by ΔAP desc
    return int(len(np.unique(off_pid[order])))


def main():
    out = {}
    for cell, cfg in CELLS.items():
        print(f"\n=== Cell {cell} ===")
        q_emb, q_pids, q_cids, g_emb, g_pids, g_cids = load_cell(cfg['file'], cfg['n_q'])
        n_eval_ids = len(np.unique(q_pids))
        print(f"  queries={len(q_emb)}, gallery={len(g_emb)}, n_query_ids={n_eval_ids}")

        ap, pid_valid = per_query_ap(q_emb, q_pids, q_cids, g_emb, g_pids, g_cids)
        mean_ap = float(ap.mean()) * 100
        print(f"  valid queries={len(ap)}  mean AP={mean_ap:.2f}%")

        deltas = 1.0 - ap
        row = {
            'cell': cell,
            'eval_set': cfg['eval'],
            'n_queries_valid': int(len(ap)),
            'n_query_ids': int(n_eval_ids),
            'mean_ap_pct': mean_ap,
            'total_sigma_dap': float(deltas.sum()),
            'gini_dap_offenders': gini(deltas[deltas > 0]),
            'gini_dap_all_queries': gini(deltas),
        }
        for pct in [1.0, 5.0, 10.0]:
            share, k_offenders = top_share(deltas, pct)
            n_distinct = top_distinct_ids(deltas, pid_valid, pct)
            row[f'top{int(pct)}pct_share'] = share
            row[f'top{int(pct)}pct_n_offenders'] = k_offenders
            row[f'top{int(pct)}pct_n_distinct_ids'] = n_distinct
            row[f'top{int(pct)}pct_distinct_frac_of_eval_ids'] = n_distinct / n_eval_ids
            print(f"  top{int(pct)}%: {k_offenders} queries carry {share*100:.1f}% of ΣΔAP, "
                  f"across {n_distinct}/{n_eval_ids} IDs ({100*n_distinct/n_eval_ids:.1f}% of eval set)")
        print(f"  Gini(ΔAP|offenders)={row['gini_dap_offenders']:.4f}  "
              f"Gini(ΔAP|all)={row['gini_dap_all_queries']:.4f}")
        out[cell] = row

    out_path = RESULTS / 'e28_part1_attribution_concentration.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved → {out_path}")

    # Rename super_additivity → attribution_coverage in e25_attribution_results.json
    p25 = RESULTS / 'e25_attribution_results.json'
    if p25.exists():
        with open(p25) as f:
            d25 = json.load(f)
        changed = False
        for cell in d25:
            if 'super_additivity' in d25[cell] and 'attribution_coverage' not in d25[cell]:
                d25[cell]['attribution_coverage'] = d25[cell].pop('super_additivity')
                changed = True
        if changed:
            with open(p25, 'w') as f:
                json.dump(d25, f, indent=2)
            print(f"Renamed super_additivity → attribution_coverage in {p25.name}")


if __name__ == '__main__':
    main()
