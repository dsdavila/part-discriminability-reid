"""
Run 0 redo: identity-clustered bootstrap CIs on |r_global| (Ch2 rev 162).

Rulings 19 + 4 applied: every |r| gets an interval, resampling identities
not items. Local-medoid measure withdrawn per rev 162 (construction + data).

Report Spearman r (signed), its 95% identity-bootstrap CI, per arm per
dataset. 500 resamples.
"""
import json, time
import numpy as np
from pathlib import Path
from scipy import stats

CACHE_DIR = Path('/tmp/e12_feature_cache')
RESULTS   = Path('/data/papers/low_dimensional_reid/results')
HUBNESS   = RESULTS / 'hubness_koccurrence.json'
N_BOOT = 500

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
    'MSMT': {'id': 'D', 'ood': "B'", 'sigma': SIGMA_MSMT},
    'Market': {'id': "A'", 'ood': 'C', 'sigma': SIGMA_MARKET},
}


def load_gallery(name):
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


def dist_to_global_centroid(G):
    c = G.mean(axis=0, keepdims=True)
    return np.linalg.norm(G - c, axis=1)


def id_cluster_bootstrap_r(kocc, dist, pids, n_boot, rng):
    """Resample identities with replacement; aggregate all items of the
    sampled identities; compute Spearman on the aggregated (kocc, dist)."""
    unique_pids = np.unique(pids)
    pid_to_items = {int(p): np.where(pids == p)[0] for p in unique_pids}
    boots = []
    for b in range(n_boot):
        sampled = rng.choice(unique_pids, size=len(unique_pids), replace=True)
        item_idx = np.concatenate([pid_to_items[int(p)] for p in sampled])
        r, _ = stats.spearmanr(kocc[item_idx], dist[item_idx])
        boots.append(float(r))
    arr = np.array(boots)
    return (float(np.median(arr)),
            float(np.quantile(arr, 0.025)),
            float(np.quantile(arr, 0.975)),
            boots)


def main():
    print("Loading hubness JSON...", flush=True)
    hub = json.load(open(HUBNESS))

    out = {}
    for ds_name, cfg in DATASETS.items():
        print(f"\n########## {ds_name} ##########", flush=True)
        G_id, g_pids, g_cids = load_gallery(cfg['id'])
        G_ood, _, _ = load_gallery(cfg['ood'])
        # Defensive: ensure unit-norm on id/ood. Caches checked unit-norm
        # but the script should not depend on an upstream invariant.
        G_id = l2(G_id); G_ood = l2(G_ood)
        G_noise = noise_degrade(G_id, cfg['sigma'])
        G_rp = randproj_degrade(G_id, K_RANDPROJ)
        arms = {'id': G_id, 'ood': G_ood, 'noise': G_noise, 'randproj': G_rp}
        print(f"  Ng={len(G_id)}  n_ids={len(np.unique(g_pids))}", flush=True)

        ds_out = {'per_arm': {}}
        for arm_name, G in arms.items():
            d_global = dist_to_global_centroid(G)
            kocc = np.array(hub[ds_name]['per_item_occ'][arm_name])
            r_point, _ = stats.spearmanr(kocc, d_global)
            rng = np.random.default_rng(42 + hash(arm_name) % 10000)
            t0 = time.time()
            med, lo, hi, boots = id_cluster_bootstrap_r(
                kocc, d_global, g_pids, N_BOOT, rng)
            ds_out['per_arm'][arm_name] = {
                'r_spearman_point': float(r_point),
                'r_spearman_boot_median': med,
                'r_spearman_boot_lo': lo,
                'r_spearman_boot_hi': hi,
                'boot': boots,
            }
            print(f"  [{arm_name}] r_spearman={r_point:+.4f}  "
                  f"id-boot: median={med:+.4f} [{lo:+.4f}, {hi:+.4f}]  "
                  f"t={(time.time()-t0):.1f}s", flush=True)
        out[ds_name] = ds_out

    print("\n========== Identity-clustered |r_global| CIs ==========")
    print(f"  {'dataset':<8} {'arm':<10} {'r_point':>10} {'id-boot CI':<35}")
    for ds in DATASETS:
        for arm in ('id', 'ood', 'noise', 'randproj'):
            s = out[ds]['per_arm'][arm]
            ci = f"[{s['r_spearman_boot_lo']:+.3f}, {s['r_spearman_boot_hi']:+.3f}]"
            print(f"  {ds:<8} {arm:<10} {s['r_spearman_point']:+10.3f}  {ci:<35}")

    # Paired differences (ood - noise, ood - randproj, ood - id)
    print("\n========== Paired differences (identity bootstrap) ==========")
    out['paired'] = {}
    for ds in DATASETS:
        out['paired'][ds] = {}
        print(f"\n[{ds}]")
        ood_boots = np.array(out[ds]['per_arm']['ood']['boot'])
        for other in ('id', 'noise', 'randproj'):
            other_boots = np.array(out[ds]['per_arm'][other]['boot'])
            diff = ood_boots - other_boots
            lo = float(np.quantile(diff, 0.025))
            hi = float(np.quantile(diff, 0.975))
            med = float(np.median(diff))
            sig = "*" if lo * hi > 0 else " "
            out['paired'][ds][f'ood_minus_{other}'] = {
                'median': med, 'lo': lo, 'hi': hi,
            }
            print(f"  ood - {other:<10} {med:+.4f} [{lo:+.4f}, {hi:+.4f}] {sig}")

    out_path = RESULTS / 'e43_centrality_r_run0_boot.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == '__main__':
    main()
