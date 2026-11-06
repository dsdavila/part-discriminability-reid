"""
E41 follow-up: per-decile bootstrap CIs over identities, and a Market 4-bin
rebin to replace the collapsed 10-bin result.

Reuses the partition and metric pipeline from compute_e41.py. 500 bootstrap
resamples of the identities within each decile (keeping that decile's n_ids).
Reports 95% CI on D/B'/noise/randproj CMC@1 and median positive rank per
decile. Market also computed with 4 quartiles to replace the bin-collapsed
10-bin result.
"""
import json, time
import numpy as np
from pathlib import Path
from collections import Counter

CACHE_DIR = Path('/tmp/e12_feature_cache')
RESULTS   = Path('/data/papers/low_dimensional_reid/results')
N_BOOT = 500
RNG = np.random.default_rng(42)
MARKET_N_BINS = 4

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
    """Compute per-query: CMC{1,5,10}, list of positive ranks."""
    N = len(G)
    cmc1 = np.zeros(N, dtype=bool)
    cmc5 = np.zeros(N, dtype=bool)
    cmc10 = np.zeros(N, dtype=bool)
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
                cmc5[gi] = pos_ranks[0] <= 5
                cmc10[gi] = pos_ranks[0] <= 10
            else:
                pos_ranks_per_q[gi] = []
    return cmc1, cmc5, cmc10, pos_ranks_per_q


def per_id_rank1_err(cmc1_arm_id, g_pids):
    out = {}
    for pid in np.unique(g_pids):
        mask = g_pids == pid
        out[int(pid)] = float(1.0 - cmc1_arm_id[mask].mean())
    return out


def quantile_bins(per_id_err, n_bins):
    pids = np.array(sorted(per_id_err.keys()))
    errs = np.array([per_id_err[int(p)] for p in pids])
    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.quantile(errs, qs)
    bins = np.clip(np.digitize(errs, edges[1:-1], right=False), 0, n_bins - 1)
    return {int(p): int(b) for p, b in zip(pids, bins)}, edges.tolist()


def bootstrap_metrics_per_decile(cmc1, pos_ranks_per_q, g_pids,
                                 bin_map, n_bins, n_boot=N_BOOT):
    """For each bin, bootstrap-resample identities (with replacement), compute
    CMC@1 and median positive rank across the queries of the resampled ids."""
    # precompute: for each bin, list of pids
    bin_to_pids = {b: [] for b in range(n_bins)}
    for pid, b in bin_map.items():
        bin_to_pids[b].append(pid)
    # per-pid: query indices and positive ranks flattened
    pid_to_qi = {}
    for qi, pid in enumerate(g_pids):
        pid_to_qi.setdefault(int(pid), []).append(qi)
    rng = np.random.default_rng(42)
    boot = {b: {'cmc1': [], 'medR': []} for b in range(n_bins)}
    for b in range(n_bins):
        pids = bin_to_pids[b]
        if len(pids) == 0:
            continue
        for _ in range(n_boot):
            resampled = rng.choice(pids, size=len(pids), replace=True)
            # collect queries
            q_idx = []
            for pid in resampled:
                q_idx.extend(pid_to_qi.get(int(pid), []))
            if not q_idx:
                continue
            q_idx = np.array(q_idx, dtype=np.int64)
            cmc1_b = float(cmc1[q_idx].mean())
            all_ranks = []
            for qi in q_idx:
                all_ranks.extend(pos_ranks_per_q[qi])
            if all_ranks:
                boot[b]['medR'].append(float(np.median(all_ranks)))
            boot[b]['cmc1'].append(cmc1_b)
    # summarize
    summary = {}
    for b in range(n_bins):
        if not boot[b]['cmc1']:
            summary[b] = None; continue
        c = np.array(boot[b]['cmc1']); m = np.array(boot[b]['medR'])
        summary[b] = {
            'n_ids': len(bin_to_pids[b]),
            'cmc1_mean': float(c.mean()), 'cmc1_lo': float(np.quantile(c, 0.025)), 'cmc1_hi': float(np.quantile(c, 0.975)),
            'medR_mean': float(m.mean()) if len(m) else None,
            'medR_lo': float(np.quantile(m, 0.025)) if len(m) else None,
            'medR_hi': float(np.quantile(m, 0.975)) if len(m) else None,
        }
    return summary


def run(name, id_cell, ood_cell, sigma, k_rp, n_bins):
    print(f"\n########## {name}: {id_cell}/{ood_cell}, n_bins={n_bins}, n_boot={N_BOOT} ##########", flush=True)
    G_id, g_pids, g_cids = load_cell(id_cell)
    G_ood, _, _ = load_cell(ood_cell)
    print(f"  N={len(G_id)} n_ids={len(np.unique(g_pids))}", flush=True)

    G_noise = noise_degrade(G_id, sigma)
    G_rp = randproj_degrade(G_id, k_rp)
    arms = {'id': G_id, 'ood': G_ood, 'noise': G_noise, 'randproj': G_rp}

    t0 = time.time()
    print("  per-query metrics on id arm...", flush=True)
    cmc1_id, _, _, _ = compute_per_query(G_id, g_pids, g_cids)
    print(f"    t={(time.time() - t0) / 60:.1f}m", flush=True)
    per_id_err = per_id_rank1_err(cmc1_id, g_pids)
    bin_map, edges = quantile_bins(per_id_err, n_bins)
    print(f"  bin counts: {dict(sorted(Counter(bin_map.values()).items()))}", flush=True)
    print(f"  edges: {[f'{e:.4f}' for e in edges]}", flush=True)

    out = {'edges': edges, 'bin_counts': dict(sorted(Counter(bin_map.values()).items())),
           'arms': {}}
    for arm_name, G_ in arms.items():
        t0 = time.time()
        print(f"  [{arm_name}] per-query...", flush=True)
        cmc1, _, _, pr = compute_per_query(G_, g_pids, g_cids)
        print(f"    per-q t={(time.time() - t0) / 60:.1f}m", flush=True)
        t0 = time.time()
        summary = bootstrap_metrics_per_decile(cmc1, pr, g_pids, bin_map, n_bins)
        print(f"    bootstrap t={(time.time() - t0) / 60:.1f}m", flush=True)
        out['arms'][arm_name] = summary
    return out


def print_report(name, res):
    print(f"\n========== {name} bootstrap (95% CI from {N_BOOT} resamples) ==========")
    arms = res['arms']
    arm_order = ['id', 'ood', 'noise', 'randproj']
    for b in sorted(res['bin_counts'].keys()):
        row = [arms[a][b] for a in arm_order]
        if any(r is None for r in row):
            continue
        n = res['bin_counts'][b]
        print(f"\n  decile {b} (n={n})")
        for a, r in zip(arm_order, row):
            print(f"    {a:<10}  CMC@1 {r['cmc1_mean']*100:5.2f}% [{r['cmc1_lo']*100:5.2f}, {r['cmc1_hi']*100:5.2f}]   "
                  f"medR {r['medR_mean']:>8.1f} [{r['medR_lo']:>8.1f}, {r['medR_hi']:>8.1f}]")


def main():
    out = {}
    out['MSMT_10bin'] = run('MSMT (10 deciles, bootstrap)', 'D', "B'", SIGMA_D, K_RANDPROJ, 10)
    out['Market_4bin'] = run('Market (4 quartiles, bootstrap)', "A'", 'C', SIGMA_A, K_RANDPROJ, MARKET_N_BINS)
    print_report('MSMT 10-decile', out['MSMT_10bin'])
    print_report('Market 4-bin', out['Market_4bin'])

    out_path = RESULTS / 'e41_bootstrap.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved → {out_path}")


if __name__ == '__main__':
    main()
