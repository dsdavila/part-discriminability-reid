"""
Band-pass subspace sweep (first intervention experiment).

For each cell X ∈ {A, C, D, B}:
  - Mean-center gallery embeddings; compute PCA basis V (eigvecs of X^T X).
  - For each (j, k) in grid:
      - project out top-j directions
      - keep next k directions (bands [j+1 .. j+k])
      - L2-normalize the resulting features
      - compute mAP on canonical protocol
  - Record mAP(j, k) per cell.

Then for the best (j, k) on B:
  - Re-run E29 confuser overlap (D vs B on projected features)
  - Re-run Part B reciprocity on projected B features

Grid: j ∈ {0, 2, 4, 8, 16, 32}; k ∈ {32, 64, 128, 256, 'all'} where 'all' = feature_dim - j.
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


def load_cell(fname, n_q):
    d = np.load(CACHE_DIR / fname)
    emb = d['emb'].astype(np.float32)
    return (emb[:n_q], d['pids'][:n_q], d['cids'][:n_q],
            emb[n_q:], d['pids'][n_q:], d['cids'][n_q:])


def l2(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(n, 1e-12)


def compute_pca_basis(G):
    """Returns (mu, V) where V is (D, D) orthonormal, columns = eigvecs by descending variance."""
    mu = G.mean(axis=0, keepdims=True)
    Gc = (G - mu).astype(np.float64)
    # SVD gives V_T = eigenvectors of X^T X
    _, _, Vt = np.linalg.svd(Gc, full_matrices=False)
    return mu.astype(np.float32), Vt.T.astype(np.float32)  # V: (D, R)


def project(q, g, mu, V, j, k_or_all):
    """Project q, g into band [j+1 .. j+k], L2-norm."""
    D = V.shape[1]  # number of components available
    if k_or_all == 'all':
        end = D
    else:
        end = min(D, j + k_or_all)
    if end <= j:
        return None, None
    Vk = V[:, j:end]  # (D, k_eff)
    qc = (q - mu) @ Vk
    gc = (g - mu) @ Vk
    return l2(qc), l2(gc)


def mAP_rank(q, q_pids, q_cids, g, g_pids, g_cids):
    """Fast mAP on canonical junk predicate (same-pid same-cam excluded)."""
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
        precisions = (np.arange(len(pos)) + 1) / (pos + 1)
        aps.append(float(precisions.mean()))
    return float(np.mean(aps)) * 100


def main():
    results = {}
    for cell, cfg in CELLS.items():
        print(f"\n=== Cell {cell} ({cfg['eval']}) ===", flush=True)
        q_emb, q_pids, q_cids, g_emb, g_pids, g_cids = load_cell(cfg['file'], cfg['n_q'])
        q_emb = q_emb.astype(np.float32); g_emb = g_emb.astype(np.float32)
        print(f"  PCA basis from gallery (N={len(g_emb)}, D={g_emb.shape[1]})...", flush=True)
        t0 = time.time()
        mu, V = compute_pca_basis(g_emb)
        print(f"    done in {time.time()-t0:.1f}s", flush=True)

        cell_results = {}
        for j in J_GRID:
            for k_raw in K_GRID_RAW:
                proj_q, proj_g = project(q_emb, g_emb, mu, V, j, k_raw)
                if proj_q is None: continue
                map_val = mAP_rank(proj_q, q_pids, q_cids, proj_g, g_pids, g_cids)
                key = f'j={j}_k={k_raw}'
                cell_results[key] = {'j': j, 'k': k_raw,
                                     'k_effective': proj_q.shape[1],
                                     'mAP_pct': map_val}
                print(f"  {key:<16}  dim={proj_q.shape[1]:4d}  mAP={map_val:.2f}%", flush=True)
        results[cell] = cell_results

    out_path = RESULTS / 'bandpass_sweep_noREA_results.json'
    with open(out_path, 'w') as f: json.dump(results, f, indent=2)
    print(f"\nSaved → {out_path}", flush=True)

    # Baseline mAPs for reference
    print("\n=== Baselines ===", flush=True)
    for cell, byKey in results.items():
        base = byKey.get('j=0_k=all', {}).get('mAP_pct', None)
        print(f"  {cell}: baseline mAP (j=0,k=all) = {base:.2f}%", flush=True)

    # Find best (j, k) on B'
    best_B = max(results["B'"].items(), key=lambda kv: kv[1]['mAP_pct'])
    print(f"\nBest (j,k) on B': {best_B[0]}  mAP={best_B[1]['mAP_pct']:.2f}%", flush=True)


if __name__ == '__main__':
    main()
