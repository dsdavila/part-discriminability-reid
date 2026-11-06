"""
E28 Part 3 — β decomposition audit of E23 governance switch.

For each cell (A, B, C, D), compute per-identity AP + two E23 predictors:
  - within_id_spread_p75: 75th pctile of arccos(gallery_image, id_centroid) [deg]
  - centroid_norm: L2 norm of raw gallery mean before unit-normalising

Then: ρ(x,y), β = slope(y ~ x) (OLS unstandardised), SD(x), SD(y), CV(x) = SD(x)/mean(x),
and arithmetic check ρ ≈ β · SD(x) / SD(y). Bootstrap CIs over identities (n_boot=1000, seed=42).

Pairing within eval set only (Market: A vs C; MSMT: D vs B).
"""
import json
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr

CACHE_DIR = Path('/tmp/e12_feature_cache')
RESULTS   = Path('/data/papers/low_dimensional_reid/results')
RNG       = np.random.default_rng(42)
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


def per_id_predictors(g_emb, g_pids):
    """Return (unique_ids, within_id_spread_p75[deg], centroid_norm, mean_per_id_cos)."""
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
    """Return dict with ρ, β, SD(x), SD(y), CV(x) = SD(x)/mean(x), bootstrap CIs."""
    rng = rng or np.random.default_rng(42)
    def fit(xa, ya):
        rho = float(spearmanr(xa, ya).statistic)
        X = np.vstack([xa, np.ones(len(xa))]).T
        coef, *_ = np.linalg.lstsq(X, ya, rcond=None)
        beta = float(coef[0])
        sdx, sdy = float(np.std(xa, ddof=1)), float(np.std(ya, ddof=1))
        cvx = sdx / max(abs(np.mean(xa)), 1e-12)
        rho_check = float(beta * sdx / max(sdy, 1e-12))
        # Pearson correlation (direct):
        pearson = float(np.corrcoef(xa, ya)[0,1])
        return {'rho': rho, 'pearson': pearson, 'beta': beta,
                'sd_x': sdx, 'sd_y': sdy, 'mean_x': float(np.mean(xa)),
                'cv_x': cvx, 'beta_sdx_over_sdy': rho_check}
    point = fit(np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64))
    boot = {k: [] for k in point}
    n = len(x)
    xa = np.asarray(x, dtype=np.float64); ya = np.asarray(y, dtype=np.float64)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        try:
            b = fit(xa[idx], ya[idx])
            for k, v in b.items():
                boot[k].append(v)
        except Exception:
            continue
    ci = {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k,v in boot.items() if v}
    return point, ci


def main():
    out = {}
    for cell, cfg in CELLS.items():
        print(f"\n=== Cell {cell} ===")
        q_emb, q_pids, q_cids, g_emb, g_pids, g_cids = load_cell(cfg['file'], cfg['n_q'])

        ap_arr, qpid = per_query_ap(q_emb, q_pids, q_cids, g_emb, g_pids, g_cids)
        # Aggregate per-ID AP = mean over queries of that identity
        from collections import defaultdict
        id_to_aps = defaultdict(list)
        for p, a in zip(qpid, ap_arr): id_to_aps[int(p)].append(a)
        per_id_ap = {pid: float(np.mean(v)) for pid,v in id_to_aps.items()}

        uids, spread75, cnorm = per_id_predictors(g_emb, g_pids)
        print(f"  gallery IDs: {len(uids)}  query IDs with AP: {len(per_id_ap)}")

        # Only identities that appear in both the gallery predictors and the query AP map
        matched = [i for i, u in enumerate(uids) if int(u) in per_id_ap]
        spread75 = spread75[matched]
        cnorm    = cnorm[matched]
        y        = np.array([per_id_ap[int(uids[i])] for i in matched])
        print(f"  matched identities: {len(y)}  mean AP = {y.mean()*100:.2f}%")

        rng_s = np.random.default_rng(42)
        rng_c = np.random.default_rng(43)
        p_sp, ci_sp = beta_decomp(spread75, y, rng=rng_s)
        p_cn, ci_cn = beta_decomp(cnorm, y, rng=rng_c)

        out[cell] = {
            'n_matched_ids': int(len(y)),
            'mean_ap': float(y.mean()),
            'within_id_spread_p75': {'point': p_sp, 'ci95': ci_sp},
            'centroid_norm':        {'point': p_cn, 'ci95': ci_cn},
        }
        print(f"  within_id_spread_p75:  ρ={p_sp['rho']:+.3f}  β={p_sp['beta']:+.6f}  "
              f"SD(x)={p_sp['sd_x']:.3f}  CV(x)={p_sp['cv_x']:.3f}  SD(y)={p_sp['sd_y']:.3f}  "
              f"β·SDx/SDy={p_sp['beta_sdx_over_sdy']:+.3f}")
        print(f"  centroid_norm:         ρ={p_cn['rho']:+.3f}  β={p_cn['beta']:+.6f}  "
              f"SD(x)={p_cn['sd_x']:.3f}  CV(x)={p_cn['cv_x']:.3f}  SD(y)={p_cn['sd_y']:.3f}  "
              f"β·SDx/SDy={p_cn['beta_sdx_over_sdy']:+.3f}")

    out_path = RESULTS / 'e28_part3_beta_decomposition.json'
    with open(out_path, 'w') as f: json.dump(out, f, indent=2)
    print(f"\nSaved → {out_path}")


if __name__ == '__main__':
    main()
