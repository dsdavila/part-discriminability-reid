"""
E37 Δspread addendum (Plan rev 46 outstanding items).

Two things:
  1. Per-identity bootstrap CIs on real-transfer Δspread levels, by class.
  2. Random-subspace route for Δspread, at matched-mAP k_target (MSMT k=17,
     Market k=17 from noise_curve_v2), per class.

Spread per identity i = mean over i's gallery items of arccos(cos(x_j, μ_i))
with μ_i L2-normed gallery centroid.

Partition classes from rank-1 error (R-conservation / E35 convention).

Compares three corruption routes at matched-mAP:
  - real transfer (B' / C)
  - isotropic noise (σ as in E32)
  - random subspace projection (k_target as in noise_curve_v2)
"""
import json, time
import numpy as np
from pathlib import Path
from collections import Counter

CACHE_DIR = Path('/tmp/e12_feature_cache')
RESULTS   = Path('/data/papers/low_dimensional_reid/results')

CELLS = {
    'D':  ('bot_msmt_seed42_msmt.npz',  11659),
    "B'": ('msmt17_test_noREA.npz',     11659),
    "A'": ('market_test_noREA.npz',      3368),
    'C':  ('bot_msmt_seed42_market.npz', 3368),
}
SIGMA_D = 0.0625
SIGMA_A = 0.062
K_RANDPROJ_D = 17   # noise_curve_v2 matched-mAP for MSMT
K_RANDPROJ_A = 17   # noise_curve_v2 matched-mAP for Market
N_BOOT = 2000

CLASSES = ['EASY', 'ID-ONLY', 'SHARED', 'OOD-ONLY']


def load_cell(name):
    fn, n_q = CELLS[name]
    d = np.load(CACHE_DIR / fn); emb = d['emb'].astype(np.float32)
    return (emb[:n_q], d['pids'][:n_q], d['cids'][:n_q],
            emb[n_q:], d['pids'][n_q:], d['cids'][n_q:])


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
    Q, _ = np.linalg.qr(R)
    return l2(G @ Q)


def per_identity_spread(G, g_pids):
    """Return {pid: spread_radians} with spread = mean arccos(x, μ) over items."""
    out = {}
    for uid in np.unique(g_pids):
        mask = g_pids == uid
        xi = G[mask]
        mu = xi.mean(axis=0)
        mu /= max(np.linalg.norm(mu), 1e-12)
        cos_x = np.clip(xi @ mu, -1.0, 1.0)
        out[int(uid)] = float(np.arccos(cos_x).mean())
    return out


def top1_wrong_rate(G, g_pids, g_cids, chunk=512):
    N = len(G); t1 = np.full(N, -1, dtype=np.int64)
    for start in range(0, N, chunk):
        end = min(N, start + chunk)
        sims = G[start:end] @ G.T
        for local, gi in enumerate(range(start, end)):
            exc = (g_pids == g_pids[gi]) & (g_cids == g_cids[gi])
            sims[local, exc] = -2.0
            t1[gi] = int(np.argmax(sims[local]))
    pred = g_pids[t1]; wrong = pred != g_pids
    return {int(uid): float(wrong[g_pids == uid].mean()) for uid in np.unique(g_pids)}


def classify(rates_id, rates_ood, thr=0.5):
    classes = {}
    for uid in rates_id:
        ri = rates_id.get(uid, 0.0); ro = rates_ood.get(uid, 0.0)
        if ri <= thr and ro <= thr: classes[uid] = 'EASY'
        elif ri > thr and ro > thr: classes[uid] = 'SHARED'
        elif ri > thr: classes[uid] = 'ID-ONLY'
        else: classes[uid] = 'OOD-ONLY'
    return classes


def bootstrap_mean(vals, n_boot=N_BOOT, seed=42):
    rng = np.random.default_rng(seed)
    v = np.asarray(vals)
    n = len(v)
    if n == 0: return {'mean': float('nan'), 'CI95': [float('nan'), float('nan')]}
    means = np.zeros(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        means[b] = v[idx].mean()
    return {'mean': float(v.mean()),
            'CI95': [float(np.percentile(means, 2.5)),
                     float(np.percentile(means, 97.5))]}


def deg(r):
    return np.degrees(r)


def run_dataset(name, id_cell, ood_cell, sigma, k_rp):
    print(f"\n########## {name}: {id_cell}/{ood_cell}, σ={sigma}, randproj k={k_rp} ##########", flush=True)
    _, _, _, G_id, g_pids, g_cids = load_cell(id_cell)
    _, _, _, G_ood, _, _ = load_cell(ood_cell)
    print(f"  N={len(G_id)}  n_ids={len(np.unique(g_pids))}", flush=True)

    # Partition from rank-1 under id/ood
    print("  rank-1 under id...", flush=True); t0 = time.time()
    r_id = top1_wrong_rate(G_id, g_pids, g_cids)
    print(f"    t={(time.time()-t0)/60:.1f}m", flush=True)
    print("  rank-1 under ood...", flush=True); t0 = time.time()
    r_ood = top1_wrong_rate(G_ood, g_pids, g_cids)
    print(f"    t={(time.time()-t0)/60:.1f}m", flush=True)
    classes = classify(r_id, r_ood)
    print(f"  partition: {dict(Counter(classes.values()))}", flush=True)

    # Spreads under 4 arms
    print("  spread under id...", flush=True)
    s_id = per_identity_spread(G_id, g_pids)
    print("  spread under ood...", flush=True)
    s_ood = per_identity_spread(G_ood, g_pids)
    print(f"  degrading {id_cell} σ={sigma}...", flush=True)
    G_n = noise_degrade(G_id, sigma)
    s_noise = per_identity_spread(G_n, g_pids)
    print(f"  randproj {id_cell} k={k_rp}...", flush=True)
    G_rp = randproj_degrade(G_id, k_rp)
    s_rp = per_identity_spread(G_rp, g_pids)

    # Δspread per identity relative to id
    out_rows = {}
    pids = sorted(s_id.keys())
    for arm_name, s in [('ood', s_ood), ('noise', s_noise), ('randproj', s_rp)]:
        out_rows[arm_name] = {}
        # all classes pooled
        d_all = np.array([s[p] - s_id[p] for p in pids])  # radians
        d_all_deg = deg(d_all)
        out_rows[arm_name]['POOLED'] = bootstrap_mean(d_all_deg.tolist())
        # per class
        for c in CLASSES:
            pids_c = [p for p in pids if classes.get(p) == c]
            if len(pids_c) == 0:
                out_rows[arm_name][c] = {'mean': float('nan'),
                                          'CI95': [float('nan'), float('nan')],
                                          'n': 0}
                continue
            d_c = np.array([s[p] - s_id[p] for p in pids_c])
            d_c_deg = deg(d_c)
            r = bootstrap_mean(d_c_deg.tolist())
            r['n'] = len(pids_c)
            out_rows[arm_name][c] = r

    # Levels: absolute spread per arm per class
    levels = {}
    for arm_name, s in [('id', s_id), ('ood', s_ood), ('noise', s_noise), ('randproj', s_rp)]:
        levels[arm_name] = {}
        for c in CLASSES + ['POOLED']:
            if c == 'POOLED':
                vals = [s[p] for p in pids]
            else:
                vals = [s[p] for p in pids if classes.get(p) == c]
            if not vals:
                levels[arm_name][c] = {'mean': float('nan'),
                                        'CI95': [float('nan'), float('nan')],
                                        'n': 0}
                continue
            vals_deg = deg(np.array(vals))
            r = bootstrap_mean(vals_deg.tolist())
            r['n'] = len(vals)
            levels[arm_name][c] = r

    return {'dataset': name, 'id_cell': id_cell, 'ood_cell': ood_cell,
            'sigma': sigma, 'k_randproj': k_rp,
            'delta_spread_deg': out_rows,
            'spread_level_deg': levels,
            'partition_counts': dict(Counter(classes.values()))}


def print_arm_report(name, res):
    print(f"\n========== {name} — spread levels and Δspread (deg) ==========")
    print(f"\nPartition: {res['partition_counts']}", flush=True)

    print("\n  Absolute spread per class per arm (mean ± 95% CI, degrees):")
    for c in CLASSES + ['POOLED']:
        lv = res['spread_level_deg']
        line = f"  {c:<10}"
        for arm in ('id', 'ood', 'noise', 'randproj'):
            r = lv[arm][c]
            n = r.get('n', 0)
            line += f"  [{arm}] {r['mean']:6.2f} [{r['CI95'][0]:6.2f},{r['CI95'][1]:6.2f}] n={n:>4}"
        print(line)

    print("\n  Δspread (per-arm minus id), mean ± 95% CI (degrees):")
    for c in CLASSES + ['POOLED']:
        ds = res['delta_spread_deg']
        line = f"  {c:<10}"
        for arm in ('ood', 'noise', 'randproj'):
            r = ds[arm][c]
            n = r.get('n', 0)
            line += f"  [{arm}] {r['mean']:+6.2f} [{r['CI95'][0]:+6.2f},{r['CI95'][1]:+6.2f}] n={n:>4}"
        print(line)


def main():
    out = {}
    out['MSMT'] = run_dataset('MSMT', 'D', "B'", SIGMA_D, K_RANDPROJ_D)
    out['Market'] = run_dataset('Market', "A'", 'C', SIGMA_A, K_RANDPROJ_A)
    print_arm_report('MSMT', out['MSMT'])
    print_arm_report('Market', out['Market'])

    out_path = RESULTS / 'e37_dspread_cis.json'
    with open(out_path, 'w') as f: json.dump(out, f, indent=2)
    print(f"\nSaved → {out_path}")


if __name__ == '__main__':
    main()
