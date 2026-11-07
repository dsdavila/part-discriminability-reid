"""
Size-matched MSMT hubness (Ch2 rev 138). Subsample MSMT gallery to
Market's n=13,115 items, 3 draws, compute k=10 skewness per arm.

If MSMT's skewness falls toward Market's at matched size, the
cross-dataset skewness gap is sample-size driven. If it holds, the
gap is real at matched size.

Fast: 4 arms x 13,115 items x k=10 = ~20s per draw.
"""
import json, time
import numpy as np
from pathlib import Path
from scipy import stats

CACHE_DIR = Path('/tmp/e12_feature_cache')
RESULTS   = Path('/data/papers/low_dimensional_reid/results')
K_OCC = 10
N_SUBSAMPLE = 13115
N_DRAWS = 3

CELLS = {
    'D':  ('bot_msmt_seed42_msmt.npz',  11659),
    "B'": ('msmt17_test_noREA.npz',     11659),
}
SIGMA_MSMT = 0.0625
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


def compute_k_occurrence(G, g_pids, g_cids, k=K_OCC, chunk=256):
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
            occurrence[idx] += 1
    return occurrence


def main():
    print("Loading MSMT full gallery...", flush=True)
    G_id_full, pids_full, cids_full = load_cell('D')
    G_ood_full, _, _ = load_cell("B'")
    N_full = len(G_id_full)
    print(f"  full N={N_full}, subsampling to {N_SUBSAMPLE}, {N_DRAWS} draws", flush=True)

    draws_results = []
    for draw in range(N_DRAWS):
        rng = np.random.default_rng(42 + draw * 1001)
        idx = rng.choice(N_full, size=N_SUBSAMPLE, replace=False)
        G_id = G_id_full[idx]; g_pids = pids_full[idx]; g_cids = cids_full[idx]
        G_ood = G_ood_full[idx]
        G_noise = noise_degrade(G_id, SIGMA_MSMT, seed=42 + draw)
        G_rp = randproj_degrade(G_id, K_RANDPROJ, seed=42 + draw)
        arms = {'id': G_id, 'ood': G_ood, 'noise': G_noise, 'randproj': G_rp}

        draw_rec = {'n_items': N_SUBSAMPLE, 'n_ids': int(len(np.unique(g_pids))),
                    'per_arm': {}}
        for arm_name, G in arms.items():
            t0 = time.time()
            occ = compute_k_occurrence(G, g_pids, g_cids, k=K_OCC)
            draw_rec['per_arm'][arm_name] = {
                'skewness_items': float(stats.skew(occ)),
                'max_occ': int(occ.max()),
                'frac_zero': float((occ == 0).mean()),
                'mean_occ': float(occ.mean()),
            }
            print(f"  draw {draw+1} [{arm_name}]: skew={stats.skew(occ):.3f} "
                  f"max={occ.max()} fz={float((occ==0).mean()):.3f} "
                  f"t={time.time()-t0:.1f}s", flush=True)
        draws_results.append(draw_rec)

    print("\n========== Size-matched MSMT hubness (n=13,115, 3 draws) ==========")
    print(f"  Compare to full MSMT (n=82,128): id 1.175, ood 2.375, noise 1.119, randproj 0.956")
    print(f"  Compare to Market (n=13,115): id 0.844, ood 1.697, noise 0.945, randproj 0.868")
    print()
    for arm in ('id', 'ood', 'noise', 'randproj'):
        skews = [d['per_arm'][arm]['skewness_items'] for d in draws_results]
        print(f"  {arm:<10} mean={np.mean(skews):.3f} std={np.std(skews):.3f} "
              f"draws={[f'{s:.3f}' for s in skews]}")

    out = {'n_subsample': N_SUBSAMPLE, 'n_draws': N_DRAWS,
           'draws': draws_results,
           'full_msmt_ref': {'id': 1.175, 'ood': 2.375, 'noise': 1.119, 'randproj': 0.956},
           'market_ref': {'id': 0.844, 'ood': 1.697, 'noise': 0.945, 'randproj': 0.868}}
    out_path = RESULTS / 'hubness_size_matched.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == '__main__':
    main()
