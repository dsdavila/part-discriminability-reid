"""
Mean distance to own-arm centroid, with identity-clustered CIs (Ch2 rev 175).

Per (dataset, arm): compute each gallery item's Euclidean distance to its
own arm's centroid, average over items. Id-bootstrap to get CI on the mean.

Pre-registered reading (rev 175): transfer contracts the cloud toward its
own centroid on both datasets; both nulls expand it. 4-for-4 with ood
contraction roughly 10x the nulls' expansion. Point estimates were in
e43_centrality_r_run0.json.

Load per-arm embeddings, apply same noise/randproj construction as in
Run 0.
"""
import json, time
import numpy as np
from pathlib import Path
from scipy import stats
import torch

CACHE_DIR = Path('/tmp/e12_feature_cache')
RESULTS   = Path('/data/papers/low_dimensional_reid/results')
N_BOOT = 500
_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CELLS = {
    'D':  ('bot_msmt_seed42_msmt.npz',  11659),
    "B'": ('msmt17_test_noREA.npz',     11659),
    "A'": ('market_test_noREA.npz',      3368),
    'C':  ('bot_msmt_seed42_market.npz', 3368),
}
SIGMA_MSMT = 0.0625
SIGMA_MARKET = 0.062
K_RANDPROJ = 17

DATASETS = {
    'MSMT':   {'id': 'D', 'ood': "B'", 'sigma': SIGMA_MSMT},
    'Market': {'id': "A'", 'ood': 'C', 'sigma': SIGMA_MARKET},
}


def load_gallery(cell):
    fn, n_q = CELLS[cell]
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


def mean_dist_to_centroid(G_np):
    c = G_np.mean(axis=0, keepdims=True)
    d = np.linalg.norm(G_np - c, axis=1)
    return d


def id_boot_mean(dist, pids, n_boot, rng):
    unique = np.unique(pids)
    pid_to_idx = {int(p): np.where(pids == p)[0] for p in unique}
    boots = []
    for b in range(n_boot):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([pid_to_idx[int(p)] for p in sampled])
        boots.append(float(dist[idx].mean()))
    return boots


def main():
    out = {}
    for ds_name, cfg in DATASETS.items():
        print(f"\n### {ds_name} ###", flush=True)
        G_id_raw, g_pids, g_cids = load_gallery(cfg['id'])
        G_ood_raw, _, _ = load_gallery(cfg['ood'])
        G_id = l2(G_id_raw); G_ood = l2(G_ood_raw)
        G_noise = noise_degrade(G_id, cfg['sigma'])
        G_rp = randproj_degrade(G_id, K_RANDPROJ)
        arms = {'id': G_id, 'ood': G_ood, 'noise': G_noise, 'randproj': G_rp}

        ds_out = {}
        for arm_name, G in arms.items():
            d = mean_dist_to_centroid(G)
            point = float(d.mean())
            rng = np.random.default_rng(42 + hash(arm_name) % 10000)
            boots = id_boot_mean(d, g_pids, N_BOOT, rng)
            arr = np.array(boots)
            ds_out[arm_name] = {
                'point': point,
                'median': float(np.median(arr)),
                'lo': float(np.quantile(arr, 0.025)),
                'hi': float(np.quantile(arr, 0.975)),
                'boot': boots,
            }
            print(f"  {arm_name:<10} mean={point:.4f}  CI=[{ds_out[arm_name]['lo']:.4f}, {ds_out[arm_name]['hi']:.4f}]", flush=True)

        # Paired differences on identity resamples (shared variance removed)
        unique = np.unique(g_pids)
        pid_to_idx = {int(p): np.where(g_pids == p)[0] for p in unique}
        rng_p = np.random.default_rng(1001)
        d_arrays = {arm: mean_dist_to_centroid(G) for arm, G in arms.items()}
        paired = {}
        contrasts = [('id', 'ood'), ('id', 'noise'), ('id', 'randproj')]
        boot_mats = {c: [] for c in contrasts}
        for b in range(N_BOOT):
            sampled = rng_p.choice(unique, size=len(unique), replace=True)
            idx = np.concatenate([pid_to_idx[int(p)] for p in sampled])
            for a, b2 in contrasts:
                boot_mats[(a, b2)].append(float(d_arrays[a][idx].mean()
                                                - d_arrays[b2][idx].mean()))
        for (a, b2), vs in boot_mats.items():
            arr = np.array(vs)
            paired[f'{a}_minus_{b2}'] = {
                'median': float(np.median(arr)),
                'lo': float(np.quantile(arr, 0.025)),
                'hi': float(np.quantile(arr, 0.975)),
            }
            sig = '*' if paired[f'{a}_minus_{b2}']['lo'] * paired[f'{a}_minus_{b2}']['hi'] > 0 else ' '
            print(f"  paired {a} - {b2}{sig}: {paired[f'{a}_minus_{b2}']['median']:+.4f} [{paired[f'{a}_minus_{b2}']['lo']:+.4f}, {paired[f'{a}_minus_{b2}']['hi']:+.4f}]", flush=True)
        ds_out['paired'] = paired
        out[ds_name] = ds_out

    out_path = RESULTS / 'e43_mean_global_dist_bootstrap.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == '__main__':
    main()
