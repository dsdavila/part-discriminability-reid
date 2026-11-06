"""
E28 Part 1 re-run on no-REA cells. Attribution concentration.
Same spec as compute_e28_part1.py but with A'→market_test_noREA, B'→msmt17_test_noREA.
Ceiling-normalised Gini (= raw / (1 − mean ΔAP)) kept.
"""
import json
import numpy as np
from pathlib import Path

CACHE_DIR = Path('/tmp/e12_feature_cache')
RESULTS   = Path('/data/papers/low_dimensional_reid/results')

CELLS = {
    "A'": {'file': 'market_test_noREA.npz',   'n_q': 3368,  'eval': 'market'},
    'C':  {'file': 'bot_msmt_seed42_market.npz','n_q': 3368,  'eval': 'market'},
    'D':  {'file': 'bot_msmt_seed42_msmt.npz',  'n_q': 11659, 'eval': 'msmt'},
    "B'": {'file': 'msmt17_test_noREA.npz',    'n_q': 11659, 'eval': 'msmt'},
}


def load_cell(cell):
    cfg = CELLS[cell]
    d = np.load(CACHE_DIR / cfg['file']); n_q = cfg['n_q']
    emb = d['emb'].astype(np.float32)
    return (emb[:n_q], d['pids'][:n_q], d['cids'][:n_q],
            emb[n_q:], d['pids'][n_q:], d['cids'][n_q:], cfg['eval'])


def per_query_ap(q_emb, q_pids, q_cids, g_emb, g_pids, g_cids):
    sims = q_emb @ g_emb.T
    aps, pids_out = [], []
    for i in range(len(q_emb)):
        qp, qc = q_pids[i], q_cids[i]
        valid = ~((g_pids == qp) & (g_cids == qc))
        rel = (g_pids == qp) & valid
        if rel.sum() == 0: continue
        order = np.argsort(-sims[i][valid])
        rel_v = rel[valid][order]
        if rel_v.sum() == 0: continue
        pos = np.where(rel_v)[0]
        aps.append(float(((np.arange(len(pos))+1) / (pos+1)).mean()))
        pids_out.append(int(qp))
    return np.array(aps), np.array(pids_out)


def gini(x):
    x = np.sort(np.asarray(x, dtype=np.float64))
    n = len(x)
    if n == 0 or x.sum() == 0: return 0.0
    cum = np.cumsum(x)
    return float((n + 1 - 2 * cum.sum() / cum[-1]) / n)


def top_share(deltas, pct):
    offenders = deltas[deltas > 0]
    if len(offenders) == 0: return 0.0, 0
    k = max(1, int(np.ceil(len(offenders) * pct / 100)))
    return float(np.sort(offenders)[-k:].sum() / offenders.sum()), int(k)


def top_distinct_ids(deltas, pids, pct):
    mask = deltas > 0
    off_dap = deltas[mask]; off_pid = pids[mask]
    if len(off_dap) == 0: return 0
    k = max(1, int(np.ceil(len(off_dap) * pct / 100)))
    order = np.argsort(-off_dap)[:k]
    return int(len(np.unique(off_pid[order])))


def main():
    out = {}
    for cell in CELLS:
        print(f"\n=== Cell {cell} ===")
        q_emb, q_pids, q_cids, g_emb, g_pids, g_cids, eval_set = load_cell(cell)
        ap, pid_v = per_query_ap(q_emb, q_pids, q_cids, g_emb, g_pids, g_cids)
        mean_ap = float(ap.mean()) * 100
        print(f"  valid Q={len(ap)}  mean AP={mean_ap:.2f}%")
        deltas = 1.0 - ap
        mean_dap = float(deltas.mean())
        max_gini = 1.0 - mean_dap   # Note: max Gini for ΔAP ∈ [0,1] with mean m is (1 − m)
        row = {
            'cell': cell, 'eval_set': eval_set,
            'n_queries_valid': int(len(ap)),
            'n_query_ids': int(len(np.unique(q_pids))),
            'mean_ap_pct': mean_ap,
            'mean_dap': mean_dap,
            'max_gini_ceiling': max_gini,
            'gini_dap_all': gini(deltas),
            'gini_dap_offenders': gini(deltas[deltas > 0]),
            'gini_dap_all_normalised': gini(deltas) / max(max_gini, 1e-12),
        }
        for pct in [1.0, 5.0, 10.0]:
            share, k_off = top_share(deltas, pct)
            n_dist = top_distinct_ids(deltas, pid_v, pct)
            row[f'top{int(pct)}pct_share'] = share
            row[f'top{int(pct)}pct_n_offenders'] = k_off
            row[f'top{int(pct)}pct_n_distinct_ids'] = n_dist
            n_eval = row['n_query_ids']
            row[f'top{int(pct)}pct_distinct_frac'] = n_dist / n_eval
            print(f"  top{int(pct)}%: {k_off} Q → {share*100:.1f}% ΣΔAP; {n_dist}/{n_eval} IDs")
        print(f"  Gini(ΔAP|all) = {row['gini_dap_all']:.4f}  / max={max_gini:.4f}  "
              f"= **{row['gini_dap_all_normalised']:.4f}** normalised")
        out[cell] = row

    out_path = RESULTS / 'e28_part1_noREA.json'
    with open(out_path, 'w') as f: json.dump(out, f, indent=2)
    print(f"\nSaved → {out_path}")


if __name__ == '__main__':
    main()
