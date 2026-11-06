"""
Bootstrap CIs on raw Gini and ceiling-normalised Gini, per cell.
Resample identities (not queries), use their queries' ΔAP distribution.
"""
import json
import numpy as np
from pathlib import Path
from collections import defaultdict

CACHE_DIR = Path('/tmp/e12_feature_cache')
RESULTS   = Path('/data/papers/low_dimensional_reid/results')
N_BOOT = 1000

CELLS = {
    "A'": {'file': 'market_test_noREA.npz',    'n_q': 3368},
    'C':  {'file': 'bot_msmt_seed42_market.npz','n_q': 3368},
    'D':  {'file': 'bot_msmt_seed42_msmt.npz', 'n_q': 11659},
    "B'": {'file': 'msmt17_test_noREA.npz',    'n_q': 11659},
}


def load_cell(cell):
    cfg = CELLS[cell]
    d = np.load(CACHE_DIR / cfg['file']); n_q = cfg['n_q']
    emb = d['emb'].astype(np.float32)
    return (emb[:n_q], d['pids'][:n_q], d['cids'][:n_q],
            emb[n_q:], d['pids'][n_q:], d['cids'][n_q:])


def per_query_ap(q_emb, q_pids, q_cids, g_emb, g_pids, g_cids):
    sims = q_emb @ g_emb.T
    aps, qpid = [], []
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
        qpid.append(int(qp))
    return np.array(aps), np.array(qpid)


def gini(x):
    x = np.sort(np.asarray(x, dtype=np.float64))
    n = len(x)
    if n == 0 or x.sum() == 0: return 0.0
    cum = np.cumsum(x)
    return float((n + 1 - 2 * cum.sum() / cum[-1]) / n)


def main():
    out = {}
    for cell in CELLS:
        print(f"\n=== Cell {cell} ===", flush=True)
        q_emb, q_pids, q_cids, g_emb, g_pids, g_cids = load_cell(cell)
        ap, qpid = per_query_ap(q_emb, q_pids, q_cids, g_emb, g_pids, g_cids)
        deltas = 1.0 - ap
        # Group by identity
        id_to_dap = defaultdict(list)
        for p, d in zip(qpid, deltas):
            id_to_dap[int(p)].append(d)
        unique_ids = np.array(sorted(id_to_dap.keys()))
        # Point estimate
        raw_gini = gini(deltas)
        mean_dap = float(deltas.mean())
        ceiling = 1.0 - mean_dap
        norm_gini = raw_gini / max(ceiling, 1e-12)
        # Bootstrap over identities
        rng = np.random.default_rng(42)
        raw_boots, norm_boots = [], []
        n = len(unique_ids)
        for _ in range(N_BOOT):
            sampled = rng.choice(unique_ids, n, replace=True)
            all_dap = np.concatenate([id_to_dap[p] for p in sampled])
            g_raw = gini(all_dap)
            md = float(all_dap.mean())
            ceil = 1.0 - md
            raw_boots.append(g_raw)
            norm_boots.append(g_raw / max(ceil, 1e-12))
        raw_ci = (float(np.percentile(raw_boots, 2.5)), float(np.percentile(raw_boots, 97.5)))
        norm_ci = (float(np.percentile(norm_boots, 2.5)), float(np.percentile(norm_boots, 97.5)))
        out[cell] = {
            'mean_ap': float(ap.mean()),
            'mean_dap': mean_dap,
            'ceiling': ceiling,
            'raw_gini': raw_gini, 'raw_gini_ci95': raw_ci,
            'normalised_gini': norm_gini, 'normalised_gini_ci95': norm_ci,
        }
        print(f"  raw Gini = {raw_gini:.4f} CI={raw_ci}")
        print(f"  normalised Gini = {norm_gini:.4f} CI={norm_ci}")
    out_path = RESULTS / 'e28_part1_gini_CIs.json'
    with open(out_path, 'w') as f: json.dump(out, f, indent=2)
    print(f"\nSaved → {out_path}")


if __name__ == '__main__':
    main()
