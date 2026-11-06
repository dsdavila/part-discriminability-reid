"""
E28 Part 3 REDO — β decomposition on a floor-free outcome.

Outcomes:
  y_logrank = log(rank of first correct match)   (lower is better)
  y_mrr     = 1 / rank of first correct match     (higher is better)

Predictors (same as Part 3): within_id_spread_p75, centroid_norm.
Per identity: aggregate over queries of that ID (mean of log(rank)).
Bootstrap n=1000, seed=42.
"""
import json, time
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr
from collections import defaultdict

CACHE_DIR = Path('/tmp/e12_feature_cache')
RESULTS   = Path('/data/papers/low_dimensional_reid/results')
N_BOOT    = 1000

CELLS = {
    'A': {'file': 'market_test.npz',           'n_q': 3368},
    'C': {'file': 'bot_msmt_seed42_market.npz','n_q': 3368},
    'D': {'file': 'bot_msmt_seed42_msmt.npz',  'n_q': 11659},
    'B': {'file': 'msmt17_test.npz',           'n_q': 11659},
}


def load_cell(fname, n_q):
    d = np.load(CACHE_DIR / fname)
    emb = d['emb'].astype(np.float32)
    pids = d['pids']; cids = d['cids']
    return (emb[:n_q], pids[:n_q], cids[:n_q],
            emb[n_q:], pids[n_q:], cids[n_q:])


def per_query_rank_first(q_emb, q_pids, q_cids, g_emb, g_pids, g_cids):
    """Returns (rank_first_arr, qpid_arr). Rank is 1-indexed within valid gallery."""
    sims = q_emb @ g_emb.T
    ranks, qpids_out = [], []
    for i in range(len(q_emb)):
        qp, qc = q_pids[i], q_cids[i]
        valid = ~((g_pids == qp) & (g_cids == qc))
        rel = (g_pids == qp) & valid
        if rel.sum() == 0: continue
        order = np.argsort(-sims[i][valid])
        rel_v = rel[valid][order]
        if rel_v.sum() == 0: continue
        first = int(np.argmax(rel_v)) + 1   # 1-indexed
        ranks.append(first)
        qpids_out.append(int(qp))
    return np.array(ranks, dtype=np.int64), np.array(qpids_out, dtype=np.int64)


def per_id_predictors(g_emb, g_pids):
    unique_ids = np.unique(g_pids)
    spread75 = np.zeros(len(unique_ids))
    cnorm    = np.zeros(len(unique_ids))
    for i, uid in enumerate(unique_ids):
        E = g_emb[g_pids == uid]
        raw_mean = E.mean(axis=0)
        cnorm[i] = float(np.linalg.norm(raw_mean))
        unit_mean = raw_mean / max(cnorm[i], 1e-12)
        cos_vals = np.clip(E @ unit_mean, -1.0, 1.0)
        angles = np.degrees(np.arccos(cos_vals))
        spread75[i] = float(np.percentile(angles, 75))
    return unique_ids, spread75, cnorm


def beta_decomp(x, y, n_boot=N_BOOT, rng=None):
    rng = rng or np.random.default_rng(42)
    def fit(xa, ya):
        rho = float(spearmanr(xa, ya).statistic)
        X = np.vstack([xa, np.ones(len(xa))]).T
        coef, *_ = np.linalg.lstsq(X, ya, rcond=None)
        beta = float(coef[0])
        sdx, sdy = float(np.std(xa, ddof=1)), float(np.std(ya, ddof=1))
        cvx = sdx / max(abs(np.mean(xa)), 1e-12)
        return {'rho': rho, 'beta': beta, 'sd_x': sdx, 'sd_y': sdy,
                'mean_x': float(np.mean(xa)), 'mean_y': float(np.mean(ya)),
                'cv_x': cvx, 'beta_sdx_over_sdy': float(beta * sdx / max(sdy, 1e-12))}
    xa = np.asarray(x, dtype=np.float64); ya = np.asarray(y, dtype=np.float64)
    point = fit(xa, ya)
    boot = {k: [] for k in point}
    n = len(x)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        try:
            b = fit(xa[idx], ya[idx])
            for k, v in b.items(): boot[k].append(v)
        except Exception: pass
    ci = {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k,v in boot.items() if v}
    return point, ci


def main():
    out = {}
    for cell, cfg in CELLS.items():
        print(f"\n=== Cell {cell} ===", flush=True)
        q_emb, q_pids, q_cids, g_emb, g_pids, g_cids = load_cell(cfg['file'], cfg['n_q'])
        ranks, qpid = per_query_rank_first(q_emb, q_pids, q_cids, g_emb, g_pids, g_cids)
        print(f"  queries with valid pos: {len(ranks)}  median rank_first = {np.median(ranks):.0f}")

        # Per-identity aggregates
        id_to_logr = defaultdict(list)
        id_to_mrr  = defaultdict(list)
        for p, r in zip(qpid, ranks):
            id_to_logr[int(p)].append(np.log(r))
            id_to_mrr[int(p)].append(1.0 / r)
        per_id_logr = {pid: float(np.mean(v)) for pid,v in id_to_logr.items()}
        per_id_mrr  = {pid: float(np.mean(v)) for pid,v in id_to_mrr.items()}

        uids, spread75, cnorm = per_id_predictors(g_emb, g_pids)
        matched = [i for i, u in enumerate(uids) if int(u) in per_id_logr]
        spread75 = spread75[matched]
        cnorm    = cnorm[matched]
        y_logr   = np.array([per_id_logr[int(uids[i])] for i in matched])
        y_mrr    = np.array([per_id_mrr[int(uids[i])]  for i in matched])

        print(f"  matched IDs: {len(y_logr)}  mean log(rank_first) = {y_logr.mean():.3f}  "
              f"mean MRR = {y_mrr.mean():.4f}")

        rng = np.random.default_rng(42)
        results_cell = {'n_matched_ids': int(len(y_logr))}

        for pname, x in [('within_id_spread_p75', spread75), ('centroid_norm', cnorm)]:
            results_cell[pname] = {}
            for yname, y in [('log_rank_first', y_logr), ('mrr', y_mrr)]:
                point, ci = beta_decomp(x, y, rng=np.random.default_rng(abs(hash((cell, pname, yname))) % (2**32)))
                results_cell[pname][yname] = {'point': point, 'ci95': ci}
                print(f"  {pname:<22} vs {yname:<16}: "
                      f"ρ={point['rho']:+.3f} β={point['beta']:+.5f} "
                      f"SD(x)={point['sd_x']:.3f} CV(x)={point['cv_x']:.3f} "
                      f"SD(y)={point['sd_y']:.4f} β·SDx/SDy={point['beta_sdx_over_sdy']:+.3f}")

        out[cell] = results_cell

    out_path = RESULTS / 'e28_part3_redo_logrank.json'
    with open(out_path, 'w') as f: json.dump(out, f, indent=2)
    print(f"\nSaved → {out_path}", flush=True)


if __name__ == '__main__':
    main()
