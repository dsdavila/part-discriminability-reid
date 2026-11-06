"""
Noise-null Δg: compare class-conditional geometric change under real transfer
(D vs B') against the same change under structure-free degradation (D vs
noise-degraded-D at matched mAP).

If OOD-ONLY identities' Δspread is +4° under real transfer AND +4° under noise,
the column is pure selection. If it is much smaller under noise, the real Δg
has signal beyond selection.

Also computes per-identity bootstrap CIs on class-conditional Δg means.
"""
import json, time
import numpy as np
from pathlib import Path
from collections import defaultdict

CACHE_DIR = Path('/tmp/e12_feature_cache')
RESULTS   = Path('/data/papers/low_dimensional_reid/results')
# σ values that hit B'=5.4% and C=26.66% from noise_curve_v2
MSMT_SIGMA_TO_B = 0.0625
MKT_SIGMA_TO_C  = 0.0620
N_BOOT = 500
K_HUB = 10


def load_cell(cell):
    files = {
        "A'": ('market_test_noREA.npz', 3368),
        'C':  ('bot_msmt_seed42_market.npz', 3368),
        'D':  ('bot_msmt_seed42_msmt.npz', 11659),
        "B'": ('msmt17_test_noREA.npz', 11659),
    }
    fn, n_q = files[cell]
    d = np.load(CACHE_DIR / fn); emb = d['emb'].astype(np.float32)
    return (emb[:n_q], d['pids'][:n_q], d['cids'][:n_q],
            emb[n_q:], d['pids'][n_q:], d['cids'][n_q:])


def l2(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(n, 1e-12)


def noise_degrade(G, Q, sigma, seed=42):
    rng = np.random.default_rng(seed)
    g_eps = rng.normal(0, sigma, G.shape).astype(np.float32)
    q_eps = rng.normal(0, sigma, Q.shape).astype(np.float32)
    return l2(G + g_eps), l2(Q + q_eps)


def top1_wrong_rate(G, g_pids, g_cids, chunk=2048):
    N = len(G); t1 = np.full(N, -1, dtype=np.int64)
    for start in range(0, N, chunk):
        end = min(N, start + chunk)
        sims = G[start:end] @ G.T
        for local, gi in enumerate(range(start, end)):
            exc = (g_pids == g_pids[gi]) & (g_cids == g_cids[gi])
            sims[local, exc] = -2.0
            t1[gi] = g_pids[int(np.argmax(sims[local]))]
    wrong = (t1 != g_pids)
    return {int(uid): float(wrong[g_pids == uid].mean()) for uid in np.unique(g_pids)}


def topk_neighbours(G, g_pids, g_cids, k_max, chunk=2048):
    N = len(G); out = np.full((N, k_max), -1, dtype=np.int64)
    for start in range(0, N, chunk):
        end = min(N, start + chunk)
        sims = G[start:end] @ G.T
        for local, gi in enumerate(range(start, end)):
            exc = (g_pids == g_pids[gi]) & (g_cids == g_cids[gi])
            sims[local, exc] = -2.0
            top = np.argpartition(-sims[local], k_max)[:k_max]
            out[gi] = top[np.argsort(-sims[local, top])]
    return out


def per_id_geometry_plus_hub(G, g_pids, g_cids):
    uniq = np.unique(g_pids)
    out = {}
    for uid in uniq:
        E = G[g_pids == uid]
        if len(E) < 2:
            out[int(uid)] = {'spread75': 0.0, 'cnorm': 0.0, 'n_items': int(len(E))}
            continue
        raw_mean = E.mean(axis=0)
        cnorm = float(np.linalg.norm(raw_mean))
        unit_mean = raw_mean / max(cnorm, 1e-12)
        cos_vals = np.clip(E @ unit_mean, -1.0, 1.0)
        angles = np.degrees(np.arccos(cos_vals))
        sp75 = float(np.percentile(angles, 75))
        out[int(uid)] = {'spread75': sp75, 'cnorm': cnorm, 'n_items': int(len(E))}
    # hubness
    topk = topk_neighbours(G, g_pids, g_cids, K_HUB)
    nbr = g_pids[topk]
    counts = defaultdict(int)
    for i in range(len(nbr)):
        for jj in nbr[i]:
            counts[int(jj)] += 1
    for uid in uniq:
        out[int(uid)]['hubness'] = float(counts.get(int(uid), 0) / max((g_pids == uid).sum(), 1))
    return out


def classify(rates_id, rates_ood, thr=0.5):
    classes = {}
    for uid in rates_id:
        r_id = rates_id.get(uid, 0.0); r_ood = rates_ood.get(uid, 0.0)
        if r_id <= thr and r_ood <= thr: classes[uid] = 'EASY'
        elif r_id > thr and r_ood > thr: classes[uid] = 'SHARED'
        elif r_id > thr: classes[uid] = 'ID-ONLY'
        else: classes[uid] = 'OOD-ONLY'
    return classes


def bootstrap_mean_ci(vals, n_boot=N_BOOT, seed=42):
    if len(vals) == 0: return 0.0, (0.0, 0.0)
    rng = np.random.default_rng(seed)
    arr = np.asarray(vals)
    boots = [arr[rng.integers(0, len(arr), len(arr))].mean() for _ in range(n_boot)]
    return float(arr.mean()), (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))


def analyse_pair(name, c_id, c_ood, sigma=None):
    """
    If sigma is None, this is the real transfer case (c_ood is the OOD cell).
    If sigma is set, c_ood is ignored and we degrade c_id's features with Gaussian noise.
    """
    print(f"\n=== {name} ===", flush=True)
    q_id, qp_id, qc_id, G_id, g_pids, g_cids = load_cell(c_id)
    if sigma is None:
        _, _, _, G_ood, _, _ = load_cell(c_ood)
        ood_label = c_ood
    else:
        G_ood, _ = noise_degrade(G_id, q_id, sigma)   # degraded gallery
        ood_label = f'noise-{c_id}-σ={sigma}'

    print(f"  rank-1 under {c_id}...", flush=True)
    rates_id = top1_wrong_rate(G_id, g_pids, g_cids)
    print(f"  rank-1 under {ood_label}...", flush=True)
    rates_ood = top1_wrong_rate(G_ood, g_pids, g_cids)
    classes = classify(rates_id, rates_ood)
    from collections import Counter
    cnts = Counter(classes.values())
    print(f"  class counts: {dict(cnts)}", flush=True)

    print(f"  per-ID geometry under {c_id}...", flush=True)
    geo_id = per_id_geometry_plus_hub(G_id, g_pids, g_cids)
    print(f"  per-ID geometry under {ood_label}...", flush=True)
    geo_ood = per_id_geometry_plus_hub(G_ood, g_pids, g_cids)

    # Δg per identity, per class
    class_delta = defaultdict(lambda: defaultdict(list))
    for uid, cls in classes.items():
        if uid not in geo_id or uid not in geo_ood: continue
        class_delta[cls]['spread75'].append(geo_ood[uid]['spread75'] - geo_id[uid]['spread75'])
        class_delta[cls]['cnorm'].append(geo_ood[uid]['cnorm'] - geo_id[uid]['cnorm'])
        class_delta[cls]['hubness'].append(geo_ood[uid]['hubness'] - geo_id[uid]['hubness'])

    agg = {}
    print(f"  Class-conditional Δg (OOD-{ood_label} − ID-{c_id}):", flush=True)
    print(f"  {'class':<10} {'n':>5}  {'mean Δspread (CI95)':<28}  {'mean Δcnorm (CI95)':<28}  {'mean Δhub (CI95)':<28}", flush=True)
    for cls in ['EASY', 'ID-ONLY', 'SHARED', 'OOD-ONLY']:
        if cls not in class_delta:
            print(f"  {cls:<10}    0  --")
            continue
        n = len(class_delta[cls]['spread75'])
        msp, ci_sp = bootstrap_mean_ci(class_delta[cls]['spread75'])
        mcn, ci_cn = bootstrap_mean_ci(class_delta[cls]['cnorm'])
        mhb, ci_hb = bootstrap_mean_ci(class_delta[cls]['hubness'])
        print(f"  {cls:<10} {n:>5}  "
              f"{msp:+.3f} [{ci_sp[0]:+.3f},{ci_sp[1]:+.3f}]   "
              f"{mcn:+.4f} [{ci_cn[0]:+.4f},{ci_cn[1]:+.4f}]   "
              f"{mhb:+.3f} [{ci_hb[0]:+.3f},{ci_hb[1]:+.3f}]", flush=True)
        agg[cls] = {'n': int(n),
                    'mean_dspread75': msp, 'ci95_dspread75': list(ci_sp),
                    'mean_dcnorm': mcn, 'ci95_dcnorm': list(ci_cn),
                    'mean_dhubness': mhb, 'ci95_dhubness': list(ci_hb)}
    return {'id_cell': c_id, 'ood_cell': ood_label,
            'class_counts': dict(cnts), 'class_delta_g': agg}


def main():
    results = {}
    # MSMT real
    results['MSMT_real']   = analyse_pair('MSMT real (D vs B\\\')', 'D', "B'", sigma=None)
    # MSMT noise (D vs noise-D at matched B' mAP)
    results['MSMT_noise']  = analyse_pair("MSMT noise (D vs noise-D at σ=%.4f, matched 5.4%% mAP)" % MSMT_SIGMA_TO_B,
                                           'D', "B'", sigma=MSMT_SIGMA_TO_B)
    # Market real
    results['Market_real'] = analyse_pair("Market real (A' vs C)", "A'", 'C', sigma=None)
    # Market noise
    results['Market_noise']= analyse_pair("Market noise (A' vs noise-A' at σ=%.4f)" % MKT_SIGMA_TO_C,
                                           "A'", 'C', sigma=MKT_SIGMA_TO_C)

    out_path = RESULTS / 'delta_g_noise_null.json'
    with open(out_path, 'w') as f: json.dump(results, f, indent=2)
    print(f"\nSaved → {out_path}")


if __name__ == '__main__':
    main()
