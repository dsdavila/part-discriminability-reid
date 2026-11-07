"""
Hubness audit (Channel 2 rev 133). Per-arm k-occurrence distribution at k=10
+ per-arm skewness + 3x3 double stratifications on neighbourhood homophily
against difficulty and against within-identity spread.

Hubness is a candidate common cause of both difficulty and within-identity
spread (Radovanovic et al.). Standard measure: skewness of the k-occurrence
distribution at k=10. MSMT only (where the paper's claims live).

Phases:
  1. Compute per-item k=10 neighbours per arm (id/ood/noise/randproj),
     same-cam-same-pid excluded.
  2. Per-item k-occurrence = how many times this item appears in others'
     top-10. Per-identity k-occurrence = sum over its items.
  3. Per-arm skewness of the k-occurrence distribution.
  4. Save per-identity k-occurrence per arm + per-arm skewness.

Follow-up scripts will do the 3x3 double stratifications against the
in-domain (id arm) k-occurrence tercile, using the same bootstrap
machinery as compute_e42_doublestrat_boot_v2.
"""
import json, time
import numpy as np
from pathlib import Path
from scipy import stats

CACHE_DIR = Path('/tmp/e12_feature_cache')
RESULTS   = Path('/data/papers/low_dimensional_reid/results')
K_OCC = 10

CELLS = {
    'D':  ('bot_msmt_seed42_msmt.npz',  11659),
    "B'": ('msmt17_test_noREA.npz',     11659),
    "A'": ('market_test_noREA.npz',      3368),
    'C':  ('bot_msmt_seed42_market.npz', 3368),
}
SIGMA_MSMT = 0.0625
SIGMA_MARKET = 0.062
K_RANDPROJ = 17


def load_cell(name):
    fn, n_q = CELLS[name]
    d = np.load(CACHE_DIR / fn); emb = d['emb'].astype(np.float32)
    return emb[n_q:], d['pids'][n_q:], d['cids'][n_q:]


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


def compute_k_occurrence(G, g_pids, g_cids, k=K_OCC, chunk=128):
    """For each item i, top-k nearest other items (same-cam-same-pid excluded).
    Then k-occurrence[j] = how many queries have j in their top-k.
    """
    N = len(G)
    occurrence = np.zeros(N, dtype=np.int64)
    for start in range(0, N, chunk):
        end = min(N, start + chunk)
        sims = G[start:end] @ G.T
        for local in range(end - start):
            gi = start + local
            exc = (g_pids == g_pids[gi]) & (g_cids == g_cids[gi])
            sims[local, exc] = -2.0
            idx = np.argpartition(-sims[local], k)[:k]
            order = idx[np.argsort(-sims[local, idx])]
            # Increment occurrence of each neighbour
            occurrence[order] += 1
    return occurrence


def run_dataset(name, id_cell, ood_cell, sigma):
    print(f"\n########## {name}: id={id_cell}/ood={ood_cell} ##########", flush=True)
    G_id, g_pids, g_cids = load_cell(id_cell)
    G_ood, _, _ = load_cell(ood_cell)
    N = len(G_id)
    print(f"  N={N}  n_ids={len(np.unique(g_pids))}", flush=True)

    G_noise = noise_degrade(G_id, sigma)
    G_rp = randproj_degrade(G_id, K_RANDPROJ)
    arms = {'id': G_id, 'ood': G_ood, 'noise': G_noise, 'randproj': G_rp}

    per_item_occ = {}
    per_id_occ = {}
    per_arm_summary = {}
    for arm_name, G in arms.items():
        t0 = time.time()
        print(f"  [{arm_name}] computing k=10 occurrence...", flush=True)
        occ = compute_k_occurrence(G, g_pids, g_cids, k=K_OCC)
        per_item_occ[arm_name] = occ
        print(f"    t={(time.time() - t0) / 60:.1f}m", flush=True)

        # Per-identity: sum of its items' occurrence (and mean for alternative)
        pid_occ_sum = {}
        pid_occ_mean = {}
        for pid in np.unique(g_pids):
            mask = g_pids == pid
            pid_occ_sum[int(pid)] = int(occ[mask].sum())
            pid_occ_mean[int(pid)] = float(occ[mask].mean())
        per_id_occ[arm_name] = {'sum': pid_occ_sum, 'mean': pid_occ_mean}

        # Per-arm distributional summary
        # Skewness is the headline statistic for hubness (Radovanovic et al.)
        skew_items = float(stats.skew(occ))
        skew_id_mean = float(stats.skew(list(pid_occ_mean.values())))
        skew_id_sum = float(stats.skew(list(pid_occ_sum.values())))
        q = np.quantile(occ, [0.5, 0.9, 0.99])
        per_arm_summary[arm_name] = {
            'mean_occ_per_item': float(occ.mean()),
            'max_occ_per_item': int(occ.max()),
            'skewness_items': skew_items,
            'skewness_pid_mean': skew_id_mean,
            'skewness_pid_sum': skew_id_sum,
            'frac_items_with_zero_occ': float((occ == 0).mean()),
            'q50_q90_q99_items': [float(x) for x in q],
        }

    print(f"\n========== {name} k-occurrence summary (k=10) ==========")
    print(f"  {'arm':<10} {'skew_items':>10} {'skew_pid_mean':>14} {'mean_occ':>10} "
          f"{'max_occ':>8} {'frac_zero':>10}")
    for arm in arms:
        s = per_arm_summary[arm]
        print(f"  {arm:<10} {s['skewness_items']:>10.3f} {s['skewness_pid_mean']:>14.3f} "
              f"{s['mean_occ_per_item']:>10.3f} {s['max_occ_per_item']:>8d} "
              f"{s['frac_items_with_zero_occ']:>10.3f}")

    return {
        'k_occ': K_OCC,
        'N_items': int(N),
        'n_ids': int(len(np.unique(g_pids))),
        'per_arm_summary': per_arm_summary,
        'per_id_occ': {arm: per_id_occ[arm] for arm in arms},
        'per_item_occ': {arm: per_item_occ[arm].tolist() for arm in arms},
    }


def main():
    out = {}
    out['MSMT'] = run_dataset('MSMT', 'D', "B'", SIGMA_MSMT)
    out['Market'] = run_dataset('Market', "A'", 'C', SIGMA_MARKET)
    out_path = RESULTS / 'hubness_koccurrence.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved → {out_path}")


if __name__ == '__main__':
    main()
