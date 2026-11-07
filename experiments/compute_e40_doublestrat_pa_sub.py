"""
3x3 principal-angle double-strat with SUBSAMPLING to common minimum
(Channel 2 rev 108, ruling 13).

9 cells (3 spread terciles x 3 difficulty terciles); cell items range
2,446-15,113 (6.2x). Noisier subspaces sit at larger angles — manufactures
spurious effects. Fix: subsample every cell to 2,446 items per draw,
5 draws, report mean and spread across draws.
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
FIXED_K   = 200
N_SUBSAMPLE_DRAWS = 5

CELLS = {
    'D':  ('bot_msmt_seed42_msmt.npz',  11659),
    "B'": ('msmt17_test_noREA.npz',     11659),
}


def load_cell(name):
    fn, n_q = CELLS[name]
    d = np.load(CACHE_DIR / fn); emb = d['emb'].astype(np.float32)
    return emb[n_q:], d['pids'][n_q:], d['cids'][n_q:]


def pca_basis_gpu(X_np, k):
    X = torch.from_numpy(X_np).to(_DEVICE)
    mean = X.mean(dim=0, keepdim=True)
    Xc = X - mean
    XtX = (Xc.T @ Xc).cpu().numpy()
    eigvals, eigvecs = np.linalg.eigh(XtX)
    eigvals = eigvals[::-1]; eigvecs = eigvecs[:, ::-1]
    n_items, d = X_np.shape
    k_eff = min(k, n_items - 1, d)
    return eigvecs[:, :k_eff].astype(np.float32), k_eff


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
        centroid_u = centroid / max(np.linalg.norm(centroid), 1e-12)
        cos_x = np.clip(X @ centroid_u, -1.0, 1.0)
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
    g_cell = np.array([pid_to_cell.get(int(p), -1) for p in g_pids], dtype=np.int8)
    cell_items = {c: int((g_cell == c).sum()) for c in range(9)}
    common_min = min(cell_items.values())
    print(f"  cell_items: {cell_items}", flush=True)
    print(f"  subsampling to common min: {common_min} items per cell", flush=True)

    # Cell indices per cell
    cell_indices = {c: np.where(g_cell == c)[0] for c in range(9)}

    # 5 subsample draws
    all_angles = {arm: {'row_pairs': {}, 'col_pairs': {}} for arm in ('id', 'ood')}

    for draw in range(N_SUBSAMPLE_DRAWS):
        rng = np.random.default_rng(42 + draw)
        subs = {}
        for c in range(9):
            idx = cell_indices[c]
            sub = rng.choice(idx, size=common_min, replace=False)
            subs[c] = sub

        for arm, G in [('id', G_id), ('ood', G_ood)]:
            bases = {}
            for c in range(9):
                X = G[subs[c]]
                basis, k_eff = pca_basis_gpu(X, FIXED_K)
                bases[c] = basis
            # row pairs (fixed spread, span d0-d2)
            for s in range(3):
                for d1, d2 in [(0, 1), (0, 2), (1, 2)]:
                    c1 = s * 3 + d1; c2 = s * 3 + d2
                    ang = principal_angle_median(bases[c1], bases[c2])
                    key = (s, d1, d2)
                    all_angles[arm]['row_pairs'].setdefault(key, []).append(ang)
            # col pairs (fixed difficulty, span s0-s2)
            for d in range(3):
                for s1, s2 in [(0, 1), (0, 2), (1, 2)]:
                    c1 = s1 * 3 + d; c2 = s2 * 3 + d
                    ang = principal_angle_median(bases[c1], bases[c2])
                    key = (d, s1, s2)
                    all_angles[arm]['col_pairs'].setdefault(key, []).append(ang)
        print(f"  draw {draw + 1}/{N_SUBSAMPLE_DRAWS} done", flush=True)

    # Summarize
    def summarize(xs):
        a = np.array(xs)
        return {'mean': float(a.mean()), 'std': float(a.std()),
                'min': float(a.min()), 'max': float(a.max())}

    out = {'cell_items': cell_items, 'common_min': common_min,
           'n_draws': N_SUBSAMPLE_DRAWS, 'per_arm': {}}
    for arm in ('id', 'ood'):
        out['per_arm'][arm] = {'row_pairs': {}, 'col_pairs': {}}
        for key, vals in all_angles[arm]['row_pairs'].items():
            s, d1, d2 = key
            out['per_arm'][arm]['row_pairs'][f'{s}_{d1}_{d2}'] = {
                's': s, 'd1': d1, 'd2': d2, 'draws': vals, **summarize(vals)}
        for key, vals in all_angles[arm]['col_pairs'].items():
            d, s1, s2 = key
            out['per_arm'][arm]['col_pairs'][f'{d}_{s1}_{s2}'] = {
                'd': d, 's1': s1, 's2': s2, 'draws': vals, **summarize(vals)}

    print(f"\n========== 3x3 principal-angle (subsampled to {common_min} items per cell) ==========")
    for arm in ('id', 'ood'):
        print(f"\n  [{arm}] row pairs (fixed spread, d1-d2, mean ± std over {N_SUBSAMPLE_DRAWS} draws):")
        for key, info in out['per_arm'][arm]['row_pairs'].items():
            print(f"    s={info['s']}  d{info['d1']}-d{info['d2']}  "
                  f"{info['mean']:.2f}° ± {info['std']:.2f}°  "
                  f"[{info['min']:.2f}, {info['max']:.2f}]")
        print(f"  [{arm}] col pairs (fixed difficulty, s1-s2, mean ± std):")
        for key, info in out['per_arm'][arm]['col_pairs'].items():
            print(f"    d={info['d']}  s{info['s1']}-s{info['s2']}  "
                  f"{info['mean']:.2f}° ± {info['std']:.2f}°  "
                  f"[{info['min']:.2f}, {info['max']:.2f}]")

    out_path = RESULTS / 'e40_doublestrat_pa_sub.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved → {out_path}")


if __name__ == '__main__':
    main()
