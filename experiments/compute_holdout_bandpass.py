"""
Held-out (j, k) selection for the band-pass sweep.

Protocol: for each cell, split queries by identity into two halves.
  - For each (j, k) in the grid: project gallery, project queries, compute
    mAP on Q_train and Q_test separately.
  - Report best-of-grid (j, k) chosen on Q_train, test mAP under that (j, k).
  - Also report best-of-grid on the full set, for the biased upper bound.
"""
import json, time
import numpy as np
from pathlib import Path

CACHE_DIR = Path('/tmp/e12_feature_cache')
RESULTS   = Path('/data/papers/low_dimensional_reid/results')

CELLS = {
    "A'": {'file': 'market_test_noREA.npz',    'n_q': 3368,  'eval': 'market'},
    'C':  {'file': 'bot_msmt_seed42_market.npz','n_q': 3368,  'eval': 'market'},
    'D':  {'file': 'bot_msmt_seed42_msmt.npz', 'n_q': 11659, 'eval': 'msmt'},
    "B'": {'file': 'msmt17_test_noREA.npz',    'n_q': 11659, 'eval': 'msmt'},
}
J_GRID = [0, 2, 4, 8, 16, 32]
K_GRID_RAW = [32, 64, 128, 256, 'all']


def load_cell(cell):
    cfg = CELLS[cell]
    d = np.load(CACHE_DIR / cfg['file']); n_q = cfg['n_q']
    emb = d['emb'].astype(np.float32)
    return (emb[:n_q], d['pids'][:n_q], d['cids'][:n_q],
            emb[n_q:], d['pids'][n_q:], d['cids'][n_q:])


def l2(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(n, 1e-12)


def pca_basis(G):
    mu = G.mean(axis=0, keepdims=True)
    Gc = (G - mu).astype(np.float64)
    _, _, Vt = np.linalg.svd(Gc, full_matrices=False)
    return mu.astype(np.float32), Vt.T.astype(np.float32)


def project(X, mu, V, j, k_or_all):
    D = V.shape[1]
    end = D if k_or_all == 'all' else min(D, j + k_or_all)
    if end <= j: return None
    Vk = V[:, j:end]
    return l2((X - mu) @ Vk)


def mAP_rank(q, q_pids, q_cids, g, g_pids, g_cids):
    sims = q @ g.T
    aps = []
    for i in range(len(q)):
        qp, qc = q_pids[i], q_cids[i]
        valid = ~((g_pids == qp) & (g_cids == qc))
        rel = (g_pids == qp) & valid
        if rel.sum() == 0: continue
        order = np.argsort(-sims[i][valid])
        rel_v = rel[valid][order]
        if rel_v.sum() == 0: continue
        pos = np.where(rel_v)[0]
        aps.append(float(((np.arange(len(pos))+1) / (pos+1)).mean()))
    return float(np.mean(aps)) * 100


def main():
    out = {}
    for cell in CELLS:
        print(f"\n=== Cell {cell} ===", flush=True)
        q_emb, q_pids, q_cids, g_emb, g_pids, g_cids = load_cell(cell)
        q_emb = q_emb.astype(np.float32); g_emb = g_emb.astype(np.float32)

        # Split queries by identity so no leakage
        rng = np.random.default_rng(42)
        uniq_qids = np.unique(q_pids)
        rng.shuffle(uniq_qids)
        n_half = len(uniq_qids) // 2
        train_pids = set(int(p) for p in uniq_qids[:n_half])
        test_pids  = set(int(p) for p in uniq_qids[n_half:])
        mask_train = np.array([int(p) in train_pids for p in q_pids])
        mask_test  = np.array([int(p) in test_pids  for p in q_pids])

        # PCA basis from the full gallery (unchanged)
        mu, V = pca_basis(g_emb)

        # Baseline mAP (raw, canonical) on train, test, full
        print(f"  computing baselines...", flush=True)
        raw_full = mAP_rank(l2(q_emb), q_pids, q_cids, l2(g_emb), g_pids, g_cids)
        raw_train = mAP_rank(l2(q_emb[mask_train]), q_pids[mask_train], q_cids[mask_train],
                             l2(g_emb), g_pids, g_cids)
        raw_test  = mAP_rank(l2(q_emb[mask_test]),  q_pids[mask_test],  q_cids[mask_test],
                             l2(g_emb), g_pids, g_cids)
        print(f"  raw: full={raw_full:.2f} train={raw_train:.2f} test={raw_test:.2f}")

        # Sweep (j, k)
        grid = {}
        for j in J_GRID:
            for k_raw in K_GRID_RAW:
                proj_q = project(q_emb, mu, V, j, k_raw)
                proj_g = project(g_emb, mu, V, j, k_raw)
                if proj_q is None: continue
                m_train = mAP_rank(proj_q[mask_train], q_pids[mask_train], q_cids[mask_train],
                                   proj_g, g_pids, g_cids)
                m_test  = mAP_rank(proj_q[mask_test],  q_pids[mask_test],  q_cids[mask_test],
                                   proj_g, g_pids, g_cids)
                m_full  = mAP_rank(proj_q, q_pids, q_cids, proj_g, g_pids, g_cids)
                grid[(j, k_raw)] = {'train': m_train, 'test': m_test, 'full': m_full}

        # Choose best on train, report test
        best_key_train = max(grid, key=lambda k: grid[k]['train'])
        best_key_full  = max(grid, key=lambda k: grid[k]['full'])
        baseline_key   = (0, 'all')

        row = {
            'raw_canonical_full': raw_full,
            'raw_canonical_train': raw_train,
            'raw_canonical_test': raw_test,
            'mean_centred_full': grid[baseline_key]['full'],
            'mean_centred_train': grid[baseline_key]['train'],
            'mean_centred_test': grid[baseline_key]['test'],
            'best_on_full': {
                'jk': list(best_key_full),
                'mAP_full': grid[best_key_full]['full'],
            },
            'best_on_train_tested_on_test': {
                'jk_selected_on_train': list(best_key_train),
                'train_mAP': grid[best_key_train]['train'],
                'test_mAP': grid[best_key_train]['test'],
                'gain_from_raw_test': grid[best_key_train]['test'] - raw_test,
                'gain_from_centred_test': grid[best_key_train]['test'] - grid[baseline_key]['test'],
            },
            'grid': {f'j={j}_k={k}': v for (j, k), v in grid.items()},
        }
        print(f"  best on full: j={best_key_full[0]} k={best_key_full[1]} → full mAP={grid[best_key_full]['full']:.2f}")
        print(f"  held-out: best on train j={best_key_train[0]} k={best_key_train[1]} → train={grid[best_key_train]['train']:.2f}, test={grid[best_key_train]['test']:.2f}")
        print(f"  gain(test from raw)={row['best_on_train_tested_on_test']['gain_from_raw_test']:+.2f} pp")
        print(f"  gain(test from mean-centred)={row['best_on_train_tested_on_test']['gain_from_centred_test']:+.2f} pp")
        out[cell] = row

    out_path = RESULTS / 'bandpass_holdout.json'
    with open(out_path, 'w') as f: json.dump(out, f, indent=2)
    print(f"\nSaved → {out_path}")


if __name__ == '__main__':
    main()
