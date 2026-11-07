"""
Run 0: centrality pre-test (Ch2 rev 158).

For each gallery item i, compute:
  - k-occurrence(i) at k=10 (reuse hubness_koccurrence.json)
  - distance to GLOBAL dataset centroid (standard survey convention,
    Feldbauer & Flexer 2019: "C = distance to data-set centroid")
  - distance to LOCAL 10-NN medoid (alternative "local medoid"
    interpretation of the same phrase)

Report Spearman |r| between each centrality measure and k-occurrence,
per arm, both datasets. Match survey baseline: |r| ~0.38 untreated.

Ruling 25 companion check: centrality measures are NOT algebraically
entangled with k-occurrence. Centrality is a geometric distance in
embedding space; k-occurrence is a graph-structural count. They can
be arbitrarily correlated or uncorrelated depending on the arrangement.
No tautology.
"""
import json, time
import numpy as np
from pathlib import Path
from scipy import stats

CACHE_DIR = Path('/tmp/e12_feature_cache')
RESULTS   = Path('/data/papers/low_dimensional_reid/results')
HUBNESS   = RESULTS / 'hubness_koccurrence.json'

CELLS = {
    'D':  ('bot_msmt_seed42_msmt.npz',  11659),
    "B'": ('msmt17_test_noREA.npz',     11659),
    "A'": ('market_test_noREA.npz',      3368),
    'C':  ('bot_msmt_seed42_market.npz', 3368),
}
SIGMA_MSMT = 0.0625
SIGMA_MARKET = 0.062
K_RANDPROJ = 17
K_LOCAL = 10

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


def distance_to_global_centroid(G):
    """Euclidean distance to the dataset centroid (not l2-normalized to sphere;
    use raw centroid in the L2-normalised embedding space)."""
    c = G.mean(axis=0, keepdims=True)
    d = np.linalg.norm(G - c, axis=1)
    return d


def distance_to_local_medoid(G, k=K_LOCAL, chunk=128):
    """For each item i, local k-NN medoid = the neighbour of i whose average
    distance to i's other k-1 neighbours is smallest (classical medoid of the
    neighbour set). Return distance from i to that medoid.

    For speed: use cosine (sim) to find neighbours, then within each neighbour
    set compute pairwise distances (k x k) and pick the medoid.
    """
    N = len(G)
    out = np.zeros(N, dtype=np.float32)
    for start in range(0, N, chunk):
        end = min(N, start + chunk)
        sims = G[start:end] @ G.T
        for local in range(end - start):
            gi = start + local
            sims[local, gi] = -2.0  # exclude self
            idx = np.argpartition(-sims[local], k)[:k]
            nb_feat = G[idx]  # (k, D)
            # pairwise distances within neighbours
            diffs = nb_feat[:, None, :] - nb_feat[None, :, :]
            pw = np.linalg.norm(diffs, axis=-1)  # (k, k)
            avg_within = pw.mean(axis=1)
            medoid_local_idx = idx[int(np.argmin(avg_within))]
            out[gi] = float(np.linalg.norm(G[gi] - G[medoid_local_idx]))
    return out


def main():
    print("Loading hubness JSON (kocc per arm, per item)...", flush=True)
    hub = json.load(open(HUBNESS))

    out = {}
    for ds_name, cfg in DATASETS.items():
        print(f"\n########## {ds_name} ##########", flush=True)
        G_id, g_pids, g_cids = load_gallery(cfg['id'])
        G_ood, _, _ = load_gallery(cfg['ood'])
        G_noise = noise_degrade(G_id, cfg['sigma'])
        G_rp = randproj_degrade(G_id, K_RANDPROJ)
        arms = {'id': G_id, 'ood': G_ood, 'noise': G_noise, 'randproj': G_rp}

        kocc = {arm: np.array(hub[ds_name]['per_item_occ'][arm])
                for arm in arms}

        ds_out = {'per_arm': {}}
        for arm_name, G in arms.items():
            t0 = time.time()
            # Global centroid distance
            d_global = distance_to_global_centroid(G)
            # Local medoid distance
            d_local = distance_to_local_medoid(G, k=K_LOCAL)
            k = kocc[arm_name]
            # Spearman and Pearson
            r_global_s, _ = stats.spearmanr(k, d_global)
            r_global_p, _ = stats.pearsonr(k, d_global)
            r_local_s, _ = stats.spearmanr(k, d_local)
            r_local_p, _ = stats.pearsonr(k, d_local)
            # Note: hubs are typically CLOSE to centroid in the literature,
            # so |r| is positive (small distance = high k-occurrence) means
            # the sign would be negative. We report raw r (sign-aware) and |r|.
            ds_out['per_arm'][arm_name] = {
                'r_spearman_vs_global_centroid_dist': float(r_global_s),
                'r_pearson_vs_global_centroid_dist': float(r_global_p),
                'r_spearman_vs_local_medoid_dist': float(r_local_s),
                'r_pearson_vs_local_medoid_dist': float(r_local_p),
                'n_items': int(len(G)),
                'mean_global_dist': float(d_global.mean()),
                'mean_local_dist': float(d_local.mean()),
            }
            print(f"  [{arm_name}] r_spearman: global={r_global_s:+.3f} "
                  f"|global|={abs(r_global_s):.3f}  "
                  f"local={r_local_s:+.3f} |local|={abs(r_local_s):.3f}  "
                  f"t={(time.time()-t0):.1f}s", flush=True)

        out[ds_name] = ds_out

    print("\n========== Centrality |r| summary ==========")
    print("Survey baseline: |r| ~0.38 untreated, <0.2 after correction.")
    print()
    print(f"  {'dataset':<8} {'arm':<10} {'|r_global|':>10} {'|r_local|':>10}")
    for ds in DATASETS:
        for arm in ('id', 'ood', 'noise', 'randproj'):
            s = out[ds]['per_arm'][arm]
            rg = abs(s['r_spearman_vs_global_centroid_dist'])
            rl = abs(s['r_spearman_vs_local_medoid_dist'])
            print(f"  {ds:<8} {arm:<10} {rg:>10.3f} {rl:>10.3f}")

    out_path = RESULTS / 'e43_centrality_r_run0.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == '__main__':
    main()
