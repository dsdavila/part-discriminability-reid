"""
E41 bootstrap v2 — fixes rev-52 defects in compute_e41_bootstrap.py.

Fixes:
  1. Store per-resample arrays per (bin, arm) so paired ratio CIs are computable.
  2. Compute paired ratios (ood/noise, ood/randproj) directly for CMC@1 and medR.
  3. Fix alignment bug — cmc1 and medR append together (medR = nan if no
     positives in that resample), so pairing by index remains valid.
  4. Market: switch binning variable from in-domain rank-1 error to in-domain
     median positive rank per identity. 519/750 Market ids have rank-1
     error = 0.0 exactly → np.quantile edges collapse; median positive
     rank is continuous so it binnes cleanly at equal-count quantile edges.
"""
import json, time
import numpy as np
from pathlib import Path
from collections import Counter

CACHE_DIR = Path('/tmp/e12_feature_cache')
RESULTS   = Path('/data/papers/low_dimensional_reid/results')
N_BOOT = 500

CELLS = {
    'D':  ('bot_msmt_seed42_msmt.npz',  11659),
    "B'": ('msmt17_test_noREA.npz',     11659),
    "A'": ('market_test_noREA.npz',      3368),
    'C':  ('bot_msmt_seed42_market.npz', 3368),
}
SIGMA_D = 0.0625
SIGMA_A = 0.062
K_RANDPROJ = 17


def load_cell(name):
    fn, n_q = CELLS[name]
    d = np.load(CACHE_DIR / fn); emb = d['emb'].astype(np.float32)
    return (emb[n_q:], d['pids'][n_q:], d['cids'][n_q:])


def l2(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(n, 1e-12)


def noise_degrade(G, sigma, seed=42):
    rng = np.random.default_rng(seed)
    g_eps = rng.normal(0, sigma, G.shape).astype(np.float32)
    return l2(G + g_eps)


def randproj_degrade(G, k_target, seed=42):
    D = G.shape[1]
    rng = np.random.default_rng(seed)
    R = rng.standard_normal((D, k_target)).astype(np.float32)
    Qmat, _ = np.linalg.qr(R)
    return l2(G @ Qmat)


def compute_per_query(G, g_pids, g_cids, chunk=128):
    N = len(G)
    cmc1 = np.zeros(N, dtype=bool)
    pos_ranks_per_q = [None] * N
    for start in range(0, N, chunk):
        end = min(N, start + chunk)
        sims = G[start:end] @ G.T
        for local in range(end - start):
            gi = start + local
            qpid, qcid = g_pids[gi], g_cids[gi]
            exc = (g_pids == qpid) & (g_cids == qcid)
            sims[local, exc] = -2.0
            order = np.argsort(-sims[local])
            pos_mask_order = g_pids[order] == qpid
            if pos_mask_order.any():
                pos_ranks = (np.where(pos_mask_order)[0] + 1).tolist()
                pos_ranks_per_q[gi] = pos_ranks
                cmc1[gi] = pos_ranks[0] == 1
            else:
                pos_ranks_per_q[gi] = []
    return cmc1, pos_ranks_per_q


def per_id_rank1_err(cmc1_id, g_pids):
    out = {}
    for pid in np.unique(g_pids):
        m = g_pids == pid
        out[int(pid)] = float(1.0 - cmc1_id[m].mean())
    return out


def per_id_median_pos_rank(pos_ranks_per_q, g_pids):
    out = {}
    for pid in np.unique(g_pids):
        q = np.where(g_pids == pid)[0]
        ar = []
        for qi in q:
            ar.extend(pos_ranks_per_q[qi])
        out[int(pid)] = float(np.median(ar)) if ar else float('inf')
    return out


def quantile_bins(per_id_val, n_bins):
    """Equal-count quantile bins; stable argsort handles ties."""
    pids = np.array(sorted(per_id_val.keys()))
    vals = np.array([per_id_val[int(p)] for p in pids])
    order = np.argsort(vals, kind='stable')
    ranks = np.empty_like(order)
    ranks[order] = np.arange(len(vals))
    edges = np.linspace(0, len(vals), n_bins + 1).astype(int)
    bins = np.zeros(len(vals), dtype=int)
    for b in range(n_bins):
        bins[(ranks >= edges[b]) & (ranks < edges[b + 1])] = b
    bins[ranks == len(vals) - 1] = n_bins - 1
    # record value edges for reporting
    sorted_vals = vals[order]
    value_edges = [float(sorted_vals[min(edges[i], len(vals) - 1)]) for i in range(n_bins + 1)]
    return {int(p): int(b) for p, b in zip(pids, bins)}, value_edges


def bootstrap_all_arms(arm_data, g_pids, bin_map, n_bins, n_boot=N_BOOT):
    bin_to_pids = {b: [] for b in range(n_bins)}
    for pid, b in bin_map.items():
        bin_to_pids[b].append(pid)
    pid_to_qi = {}
    for qi, pid in enumerate(g_pids):
        pid_to_qi.setdefault(int(pid), []).append(qi)

    result = {b: {arm: {'cmc1': [], 'medR': []} for arm in arm_data}
              for b in range(n_bins)}

    for b in range(n_bins):
        pids = bin_to_pids[b]
        if not pids:
            continue
        rng = np.random.default_rng(42 + b)
        for r in range(n_boot):
            resampled = rng.choice(pids, size=len(pids), replace=True)
            q_idx = []
            for pid in resampled:
                q_idx.extend(pid_to_qi.get(int(pid), []))
            q_idx_arr = np.array(q_idx) if q_idx else np.array([], dtype=np.int64)
            for arm, (cmc1, pos_ranks_per_q) in arm_data.items():
                if len(q_idx_arr) == 0:
                    result[b][arm]['cmc1'].append(float('nan'))
                    result[b][arm]['medR'].append(float('nan'))
                    continue
                cmc1_val = float(cmc1[q_idx_arr].mean())
                all_ranks = []
                for qi in q_idx:
                    all_ranks.extend(pos_ranks_per_q[qi])
                medR_val = float(np.median(all_ranks)) if all_ranks else float('nan')
                result[b][arm]['cmc1'].append(cmc1_val)
                result[b][arm]['medR'].append(medR_val)
    return result, bin_to_pids


def summarize(result, bin_to_pids, n_bins):
    out = {}
    for b in range(n_bins):
        pids = bin_to_pids[b]
        if not pids or not result.get(b):
            out[str(b)] = None; continue
        arm_result = result[b]
        arms = list(arm_result.keys())
        bin_out = {'n_ids': len(pids), 'arms': {}, 'ratios': {}}
        for arm in arms:
            c = np.array(arm_result[arm]['cmc1'])
            m = np.array(arm_result[arm]['medR'])
            bin_out['arms'][arm] = {
                'cmc1_samples': arm_result[arm]['cmc1'],
                'medR_samples': arm_result[arm]['medR'],
                'cmc1_mean': float(np.nanmean(c)),
                'cmc1_lo': float(np.nanquantile(c, 0.025)),
                'cmc1_hi': float(np.nanquantile(c, 0.975)),
                'medR_median': float(np.nanmedian(m)),
                'medR_lo': float(np.nanquantile(m, 0.025)),
                'medR_hi': float(np.nanquantile(m, 0.975)),
            }
        # paired ratios (B'/noise, B'/randproj) — index-matched resamples
        for denom in ('noise', 'randproj'):
            if 'ood' in arms and denom in arms:
                c_num = np.array(arm_result['ood']['cmc1'])
                c_den = np.array(arm_result[denom]['cmc1'])
                m_num = np.array(arm_result['ood']['medR'])
                m_den = np.array(arm_result[denom]['medR'])
                c_ratio = c_num / np.where(c_den > 0, c_den, np.nan)
                m_ratio = m_num / np.where(m_den > 0, m_den, np.nan)
                bin_out['ratios'][f'ood_vs_{denom}'] = {
                    'cmc1_samples': c_ratio.tolist(),
                    'medR_samples': m_ratio.tolist(),
                    'cmc1_median': float(np.nanmedian(c_ratio)),
                    'cmc1_lo': float(np.nanquantile(c_ratio, 0.025)),
                    'cmc1_hi': float(np.nanquantile(c_ratio, 0.975)),
                    'medR_median': float(np.nanmedian(m_ratio)),
                    'medR_lo': float(np.nanquantile(m_ratio, 0.025)),
                    'medR_hi': float(np.nanquantile(m_ratio, 0.975)),
                }
        out[str(b)] = bin_out
    return out


def run(name, id_cell, ood_cell, sigma, k_rp, n_bins, bin_by='rank1_err'):
    print(f"\n########## {name}: id={id_cell}/{ood_cell}, n_bins={n_bins}, "
          f"bin_by={bin_by}, n_boot={N_BOOT} ##########", flush=True)
    G_id, g_pids, g_cids = load_cell(id_cell)
    G_ood, _, _ = load_cell(ood_cell)
    print(f"  N={len(G_id)} n_ids={len(np.unique(g_pids))}", flush=True)
    G_noise = noise_degrade(G_id, sigma)
    G_rp = randproj_degrade(G_id, k_rp)
    arms = {'id': G_id, 'ood': G_ood, 'noise': G_noise, 'randproj': G_rp}

    print("  per-query metrics all arms...", flush=True)
    arm_data = {}
    for arm_name, G_ in arms.items():
        t0 = time.time()
        cmc1, pr = compute_per_query(G_, g_pids, g_cids)
        arm_data[arm_name] = (cmc1, pr)
        print(f"    [{arm_name}] t={(time.time() - t0) / 60:.1f}m", flush=True)

    # bin
    if bin_by == 'rank1_err':
        per_id_val = per_id_rank1_err(arm_data['id'][0], g_pids)
    elif bin_by == 'median_pos_rank':
        per_id_val = per_id_median_pos_rank(arm_data['id'][1], g_pids)
    else:
        raise ValueError(f"unknown bin_by: {bin_by}")
    bin_map, edges = quantile_bins(per_id_val, n_bins)
    counts = dict(sorted(Counter(bin_map.values()).items()))
    print(f"  bin counts: {counts}", flush=True)
    print(f"  edges ({bin_by}): {[f'{e:.3f}' for e in edges]}", flush=True)

    t0 = time.time()
    print(f"  bootstrap {n_bins} bins x {N_BOOT} resamples x 4 arms...", flush=True)
    result, bin_to_pids = bootstrap_all_arms(arm_data, g_pids, bin_map, n_bins)
    print(f"    t={(time.time() - t0) / 60:.1f}m", flush=True)

    summary = summarize(result, bin_to_pids, n_bins)
    return {'bin_by': bin_by, 'edges': edges, 'bin_counts': counts,
            'bootstrap': summary}


def print_report(name, res):
    print(f"\n========== {name} ({res['bin_by']}) ratio CIs (paired, {N_BOOT} resamples) ==========")
    print(f"  bin  n_ids |  B'/noise CMC@1        |  B'/noise medR         |  B'/randproj CMC@1     |  B'/randproj medR")
    boot = res['bootstrap']
    for b in sorted(boot.keys(), key=int):
        r = boot[b]
        if r is None:
            continue
        rn = r['ratios'].get('ood_vs_noise', {})
        rp = r['ratios'].get('ood_vs_randproj', {})
        if not rn or not rp:
            continue
        print(f"  {b:>3}  {r['n_ids']:>5} | "
              f"{rn['cmc1_median']:.3f} [{rn['cmc1_lo']:.3f}, {rn['cmc1_hi']:.3f}] | "
              f"{rn['medR_median']:.3f} [{rn['medR_lo']:.3f}, {rn['medR_hi']:.3f}] | "
              f"{rp['cmc1_median']:.3f} [{rp['cmc1_lo']:.3f}, {rp['cmc1_hi']:.3f}] | "
              f"{rp['medR_median']:.3f} [{rp['medR_lo']:.3f}, {rp['medR_hi']:.3f}]")


def main():
    out = {}
    out['MSMT_rank1err_10'] = run('MSMT (10 deciles, rank-1 err)',
                                   'D', "B'", SIGMA_D, K_RANDPROJ, 10,
                                   bin_by='rank1_err')
    out['Market_medR_4'] = run('Market (4 bins, median positive rank)',
                               "A'", 'C', SIGMA_A, K_RANDPROJ, 4,
                               bin_by='median_pos_rank')
    print_report('MSMT rank-1 err 10-decile', out['MSMT_rank1err_10'])
    print_report('Market medR 4-bin', out['Market_medR_4'])

    out_path = RESULTS / 'e41_bootstrap_v2.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved → {out_path}")


if __name__ == '__main__':
    main()
