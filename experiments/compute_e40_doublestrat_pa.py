"""
3x3 double stratification on PRINCIPAL ANGLES (Channel 2 rev 105).

The neighbourhood 3x3 (compute_e42_doublestrat.py) showed difficulty
has incremental content and compactness does not. The marginal spread
principal-angle result (rev 85 covariate null) suggested the opposite.
Rev 105 points out those cannot both be right; and claude-chat withdrew
the leg-merge until the 3x3 is run on the principal-angle statistic.

This script does that.

9 cells (3 spread terciles x 3 difficulty terciles). Per-cell PCA basis
at k_eff = min(200, n_items-1, d). For each 'fixed-spread row' and each
'fixed-difficulty column', compute median principal angles between the
adjacent cell pairs in both arms. Report cell counts, k_eff per cell,
and all pair-angle values.
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

    print("Computing per-id spread_p75 and rank-1 err...", flush=True)
    spread = per_id_spread_p75(G_id, g_pids)
    diff = per_id_rank1_err(G_id, g_pids, g_cids)
    spread_bin = tercile_assign(spread)
    diff_bin = tercile_assign(diff)
    pid_to_cell = {int(p): spread_bin[int(p)] * 3 + diff_bin[int(p)]
                    for p in spread.keys()}

    # Cell counts
    g_cell = np.array([pid_to_cell.get(int(p), -1) for p in g_pids], dtype=np.int8)
    cell_ids = {c: sum(1 for v in pid_to_cell.values() if v == c)
                for c in range(9)}
    cell_items = {c: int((g_cell == c).sum()) for c in range(9)}
    print(f"  cell_ids (s*3+d): {cell_ids}", flush=True)
    print(f"  cell_items: {cell_items}", flush=True)

    # Per-cell PCA basis per arm
    out = {'cell_ids': cell_ids, 'cell_items': cell_items,
           'per_arm': {}}
    for arm, G in [('id', G_id), ('ood', G_ood)]:
        print(f"  [{arm}] computing per-cell PCA...", flush=True)
        bases = {}
        k_effs = {}
        for c in range(9):
            idx = (g_cell == c)
            if idx.sum() < 2:
                bases[c] = None; k_effs[c] = 0; continue
            t0 = time.time()
            basis, k_eff = pca_basis_gpu(G[idx], FIXED_K)
            bases[c] = basis; k_effs[c] = int(k_eff)
        print(f"    k_effs: {k_effs}", flush=True)

        # Row pairs: fixed spread, vary difficulty
        # Columns pairs: fixed difficulty, vary spread
        row_pairs = []  # (s, d1, d2, cell1, cell2)
        for s in range(3):
            for d1, d2 in [(0, 1), (0, 2), (1, 2)]:
                row_pairs.append((s, d1, d2, s * 3 + d1, s * 3 + d2))
        col_pairs = []
        for d in range(3):
            for s1, s2 in [(0, 1), (0, 2), (1, 2)]:
                col_pairs.append((d, s1, s2, s1 * 3 + d, s2 * 3 + d))

        row_results = []
        for s, d1, d2, c1, c2 in row_pairs:
            if bases[c1] is None or bases[c2] is None:
                row_results.append({'s': s, 'd1': d1, 'd2': d2, 'median_deg': None}); continue
            common_k = min(k_effs[c1], k_effs[c2])
            A = bases[c1][:, :common_k]; B = bases[c2][:, :common_k]
            ang = principal_angle_median(A, B)
            row_results.append({'s': s, 'd1': d1, 'd2': d2, 'c1': c1, 'c2': c2,
                                'common_k': common_k, 'median_deg': ang})
        col_results = []
        for d, s1, s2, c1, c2 in col_pairs:
            if bases[c1] is None or bases[c2] is None:
                col_results.append({'d': d, 's1': s1, 's2': s2, 'median_deg': None}); continue
            common_k = min(k_effs[c1], k_effs[c2])
            A = bases[c1][:, :common_k]; B = bases[c2][:, :common_k]
            ang = principal_angle_median(A, B)
            col_results.append({'d': d, 's1': s1, 's2': s2, 'c1': c1, 'c2': c2,
                                'common_k': common_k, 'median_deg': ang})
        out['per_arm'][arm] = {'k_effs': k_effs, 'row_pairs': row_results,
                                'col_pairs': col_results}

    # Headline: row effects (fixed spread, adjacent-difficulty median principal angle)
    print("\n========== 3x3 principal-angle double-strat (MSMT) ==========")
    for arm in ('id', 'ood'):
        print(f"\n  [{arm}] row pairs (fixed spread, d1→d2, median principal angle):")
        rp = out['per_arm'][arm]['row_pairs']
        for r in rp:
            if r['median_deg'] is None: continue
            print(f"    s={r['s']}  d{r['d1']}-d{r['d2']}  common_k={r['common_k']:>3}  "
                  f"median {r['median_deg']:.2f}°")
        print(f"  [{arm}] col pairs (fixed difficulty, s1→s2, median principal angle):")
        cp = out['per_arm'][arm]['col_pairs']
        for r in cp:
            if r['median_deg'] is None: continue
            print(f"    d={r['d']}  s{r['s1']}-s{r['s2']}  common_k={r['common_k']:>3}  "
                  f"median {r['median_deg']:.2f}°")

    out_path = RESULTS / 'e40_doublestrat_pa.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved → {out_path}")


if __name__ == '__main__':
    main()
