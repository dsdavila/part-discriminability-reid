"""
3x3 principal-angle double-strat on IDENTITY CENTROIDS (Channel 2 rev 113).

Item PCA with identity matching did not fix the column effect: compactness
mechanically determines effective rank at fixed sample size, so compact
cells sit at lower effective rank and thus wider principal angles. Centroid
PCA averages within-identity spread out before forming the covariance - one
vector per identity, so effective rank is bounded by n_ids regardless of
how compact each identity is.

Spec per rev 113:
  - 117 identity centroids per cell (grid min), 5 draws
  - k in {10, 25, 50} (all well below 117)
  - Both arms
  - Mean + spread per effect
"""
import json, time
import numpy as np
from pathlib import Path

try:
    import torch
    _DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
except ImportError:
    _DEVICE = None

CACHE_DIR = Path('/tmp/e12_feature_cache')
RESULTS   = Path('/data/papers/low_dimensional_reid/results')
N_DRAWS = 5
K_VALUES = [10, 25, 50]
IDS_PER_CELL = 117

CELLS = {
    'D':  ('bot_msmt_seed42_msmt.npz',  11659),
    "B'": ('msmt17_test_noREA.npz',     11659),
}


def load_cell(name):
    fn, n_q = CELLS[name]
    d = np.load(CACHE_DIR / fn); emb = d['emb'].astype(np.float32)
    return emb[n_q:], d['pids'][n_q:], d['cids'][n_q:]


def pca_basis_cpu(X, k):
    X = X.astype(np.float32)
    mean = X.mean(axis=0, keepdims=True)
    Xc = X - mean
    XtX = Xc.T @ Xc
    eigvals, eigvecs = np.linalg.eigh(XtX)
    eigvecs = eigvecs[:, ::-1]
    n_items, d = X.shape
    k_eff = min(k, n_items - 1, d)
    return eigvecs[:, :k_eff].astype(np.float32)


def principal_angle_median(A, B):
    Aq, _ = np.linalg.qr(A); Bq, _ = np.linalg.qr(B)
    M = Aq.T @ Bq
    s = np.clip(np.linalg.svd(M, compute_uv=False), -1.0, 1.0)
    return float(np.degrees(np.median(np.arccos(s))))


def per_id_spread_p75(G, g_pids):
    out = {}
    for pid in np.unique(g_pids):
        X = G[g_pids == pid]
        centroid = X.mean(axis=0)
        cu = centroid / max(np.linalg.norm(centroid), 1e-12)
        cos_x = np.clip(X @ cu, -1.0, 1.0)
        out[int(pid)] = float(np.percentile(np.arccos(cos_x), 75))
    return out


def per_id_rank1_err(G, g_pids, g_cids, chunk=256):
    N = len(G); wrong = np.zeros(N, dtype=bool)
    for start in range(0, N, chunk):
        end = min(N, start + chunk)
        sims = G[start:end] @ G.T
        for local, gi in enumerate(range(start, end)):
            exc = (g_pids == g_pids[gi]) & (g_cids == g_cids[gi])
            sims[local, exc] = -2.0
            wrong[gi] = g_pids[int(np.argmax(sims[local]))] != g_pids[gi]
    out = {}
    for pid in np.unique(g_pids):
        out[int(pid)] = float(wrong[g_pids == pid].mean())
    return out


def tercile_assign(values_by_pid):
    pids = np.array(sorted(values_by_pid.keys()))
    vals = np.array([values_by_pid[int(p)] for p in pids])
    order = np.argsort(vals, kind='stable')
    ranks = np.empty_like(order); ranks[order] = np.arange(len(vals))
    edges = np.linspace(0, len(vals), 4).astype(int)
    bins = np.zeros(len(vals), dtype=int)
    for b in range(3):
        bins[(ranks >= edges[b]) & (ranks < edges[b + 1])] = b
    bins[ranks == len(vals) - 1] = 2
    return {int(p): int(b) for p, b in zip(pids, bins)}


def compute_centroids(G, g_pids, pids_to_use):
    """One L2-normalised centroid per identity from G items."""
    centroids = []
    for pid in pids_to_use:
        X = G[g_pids == pid]
        c = X.mean(axis=0)
        c = c / max(np.linalg.norm(c), 1e-12)
        centroids.append(c)
    return np.array(centroids, dtype=np.float32)


def main():
    print("Loading MSMT...", flush=True)
    G_id, g_pids, g_cids = load_cell('D')
    G_ood, _, _ = load_cell("B'")

    spread = per_id_spread_p75(G_id, g_pids)
    diff = per_id_rank1_err(G_id, g_pids, g_cids)
    spread_bin = tercile_assign(spread)
    diff_bin = tercile_assign(diff)
    pid_to_cell = {int(p): spread_bin[int(p)] * 3 + diff_bin[int(p)]
                    for p in spread.keys()}
    cell_pids = {c: [] for c in range(9)}
    for pid, c in pid_to_cell.items():
        cell_pids[c].append(int(pid))
    print(f"  cell id counts: { {c: len(v) for c, v in cell_pids.items()} }", flush=True)

    all_results = {k: {arm: {'row_pairs': {}, 'col_pairs': {}}
                       for arm in ('id', 'ood')}
                   for k in K_VALUES}

    for draw in range(N_DRAWS):
        rng = np.random.default_rng(42 + draw * 7)
        sampled_pids = {c: list(rng.choice(cell_pids[c], size=IDS_PER_CELL, replace=False))
                        for c in range(9)}
        for arm, G in [('id', G_id), ('ood', G_ood)]:
            for k_val in K_VALUES:
                bases = {}
                for c in range(9):
                    cents = compute_centroids(G, g_pids, sampled_pids[c])
                    basis = pca_basis_cpu(cents, k_val)
                    bases[c] = basis
                for s in range(3):
                    for d1, d2 in [(0, 1), (0, 2), (1, 2)]:
                        c1 = s * 3 + d1; c2 = s * 3 + d2
                        ang = principal_angle_median(bases[c1], bases[c2])
                        key = f's{s}_d{d1}d{d2}'
                        all_results[k_val][arm]['row_pairs'].setdefault(key, []).append(ang)
                for d in range(3):
                    for s1, s2 in [(0, 1), (0, 2), (1, 2)]:
                        c1 = s1 * 3 + d; c2 = s2 * 3 + d
                        ang = principal_angle_median(bases[c1], bases[c2])
                        key = f'd{d}_s{s1}s{s2}'
                        all_results[k_val][arm]['col_pairs'].setdefault(key, []).append(ang)
        print(f"  draw {draw + 1}/{N_DRAWS} done", flush=True)

    def summarize(xs):
        a = np.array(xs)
        return {'mean': float(a.mean()), 'std': float(a.std()),
                'min': float(a.min()), 'max': float(a.max()), 'draws': xs}

    out = {'ids_per_cell': IDS_PER_CELL, 'k_values': K_VALUES, 'n_draws': N_DRAWS,
           'per_k': {}}
    for k_val in K_VALUES:
        out['per_k'][str(k_val)] = {'per_arm': {}}
        for arm in ('id', 'ood'):
            out['per_k'][str(k_val)]['per_arm'][arm] = {
                'row_pairs': {k: summarize(v)
                               for k, v in all_results[k_val][arm]['row_pairs'].items()},
                'col_pairs': {k: summarize(v)
                               for k, v in all_results[k_val][arm]['col_pairs'].items()},
            }

    print("\n========== 3x3 CENTROID-PCA principal-angle (MSMT) ==========")
    print(f"  117 identity centroids per cell, {N_DRAWS} draws")
    for k_val in K_VALUES:
        print(f"\n  k={k_val}")
        for arm in ('id', 'ood'):
            print(f"    [{arm}] row (fixed spread, d0-d2):")
            for s in range(3):
                info = out['per_k'][str(k_val)]['per_arm'][arm]['row_pairs'].get(f's{s}_d0d2')
                if info:
                    print(f"      s={s}: {info['mean']:.2f}° ± {info['std']:.2f}°")
            print(f"    [{arm}] col (fixed difficulty, s0-s2):")
            for d in range(3):
                info = out['per_k'][str(k_val)]['per_arm'][arm]['col_pairs'].get(f'd{d}_s0s2')
                if info:
                    print(f"      d={d}: {info['mean']:.2f}° ± {info['std']:.2f}°")

    out_path = RESULTS / 'e40_doublestrat_pa_centroids.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved → {out_path}")


if __name__ == '__main__':
    main()
