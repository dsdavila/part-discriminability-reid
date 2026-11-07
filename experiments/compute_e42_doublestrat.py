"""
3x3 double stratification: within-identity spread_p75 x in-domain-difficulty
rank-1 error, both terciles. Per Channel 2 rev 100: the marginal spread
comparison cannot separate the two correlated predictors (rho=-0.566).
A fixed-spread-row sweep across difficulty columns, and fixed-difficulty-
column sweep across spread rows, identifies which (if either) is doing the
work.

Cells: 3x3 = 9. ~340 identities / ~9,100 items each with 3059 total ids.
Metric: for each cell q = (s_q, d_q), compute floor-corrected neighbourhood
enrichment on the 9-cell partition, aggregate under common item weights,
and measure:
  - across-row (fixed s_q, vary d_q): does homophily track difficulty at
    fixed compactness?
  - across-column (fixed d_q, vary s_q): does homophily track compactness
    at fixed difficulty?

Depth sweep r=1/5/10/20. Same v4 machinery (per-cell floor, common item
weights, aggregate-then-ratio). id and ood arms.
"""
import json, time
import numpy as np
from pathlib import Path

CACHE_DIR = Path('/tmp/e12_feature_cache')
RESULTS   = Path('/data/papers/low_dimensional_reid/results')
N_BOOT = 200
K_MAX = 20
DEPTH_RANKS = [1, 5, 10, 20]
FLOOR_SIM = 2000

CELLS = {
    'D':  ('bot_msmt_seed42_msmt.npz',  11659),
    "B'": ('msmt17_test_noREA.npz',     11659),
}
SIGMA_D = 0.0625
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
    """Assign each pid to tercile 0/1/2 by stable-argsort rank."""
    pids = np.array(sorted(values_by_pid.keys()))
    vals = np.array([values_by_pid[int(p)] for p in pids])
    order = np.argsort(vals, kind='stable')
    ranks = np.empty_like(order)
    ranks[order] = np.arange(len(vals))
    edges = np.linspace(0, len(vals), 4).astype(int)
    bins = np.zeros(len(vals), dtype=int)
    for b in range(3):
        bins[(ranks >= edges[b]) & (ranks < edges[b+1])] = b
    bins[ranks == len(vals) - 1] = 2
    return {int(p): int(b) for p, b in zip(pids, bins)}


def compute_neighbour_cells(G, g_pids, g_cids, pid_to_cell, n_cells,
                             k_max=K_MAX, chunk=128):
    """For each query, return (k_max,) int array:
       -1 if rank-r neighbour same-id, else 0..n_cells-1 for 9-cell label.
    """
    N = len(G)
    g_cell = np.array([pid_to_cell.get(int(p), -1) for p in g_pids], dtype=np.int8)
    neighbour = np.full((N, k_max), -2, dtype=np.int8)
    for start in range(0, N, chunk):
        end = min(N, start + chunk)
        sims = G[start:end] @ G.T
        for local in range(end - start):
            gi = start + local
            qpid, qcid = g_pids[gi], g_cids[gi]
            exc = (g_pids == qpid) & (g_cids == qcid)
            sims[local, exc] = -2.0
            idx = np.argpartition(-sims[local], k_max)[:k_max]
            order = idx[np.argsort(-sims[local, idx])]
            for r, nb in enumerate(order):
                if g_pids[nb] == qpid:
                    neighbour[gi, r] = -1
                else:
                    neighbour[gi, r] = g_cell[nb]
    return neighbour, g_cell


def multinomial_floor(N, base_rates, n_sim=FLOOR_SIM, seed=42):
    if N == 0: return 0.0
    rng = np.random.default_rng(seed)
    draws = rng.multinomial(N, base_rates, size=n_sim)
    shares = draws / N
    enrich = shares / base_rates
    return float(np.mean(np.abs(enrich - 1), axis=1).mean())


def main():
    print("Loading MSMT D + B'...", flush=True)
    G_id, g_pids, g_cids = load_cell('D')
    G_ood, _, _ = load_cell("B'")

    print("Computing per-id spread_p75 and rank-1 error...", flush=True)
    spread = per_id_spread_p75(G_id, g_pids)
    diff = per_id_rank1_err(G_id, g_pids, g_cids)

    spread_bin = tercile_assign(spread)
    diff_bin = tercile_assign(diff)
    pid_to_cell = {int(p): spread_bin[int(p)] * 3 + diff_bin[int(p)]
                    for p in spread.keys()}
    # cell index = s*3 + d (s and d in 0..2, cell in 0..8)
    n_cells = 9
    cell_counts_ids = {c: sum(1 for v in pid_to_cell.values() if v == c)
                       for c in range(n_cells)}
    cell_item_counts = {}
    for c in range(n_cells):
        pids = [p for p, v in pid_to_cell.items() if v == c]
        cell_item_counts[c] = sum(int((g_pids == p).sum()) for p in pids)
    print(f"  cell_ids (s*3+d): {cell_counts_ids}", flush=True)
    print(f"  cell_items (s*3+d): {cell_item_counts}", flush=True)

    total_items = sum(cell_item_counts.values())
    base_rates = np.array([cell_item_counts[c] / total_items for c in range(n_cells)])

    G_noise = noise_degrade(G_id, SIGMA_D)
    G_rp = randproj_degrade(G_id, K_RANDPROJ)
    arms = {'id': G_id, 'ood': G_ood, 'noise': G_noise, 'randproj': G_rp}
    arm_nb = {}
    for arm_name, G in arms.items():
        t0 = time.time()
        print(f"  [{arm_name}] top-{K_MAX} neighbours...", flush=True)
        nb, _ = compute_neighbour_cells(G, g_pids, g_cids, pid_to_cell, n_cells)
        arm_nb[arm_name] = nb
        print(f"    t={(time.time() - t0) / 60:.1f}m", flush=True)

    nb_mapped = {arm: a.copy() for arm, a in arm_nb.items()}
    for arm in arms:
        nb_mapped[arm][nb_mapped[arm] == -1] = n_cells
        nb_mapped[arm] = nb_mapped[arm].astype(np.int16)

    g_cell = np.array([pid_to_cell.get(int(p), -1) for p in g_pids], dtype=np.int8)

    # Precompute floors per (arm, cell, rank)
    print("  precomputing cell floors...", flush=True)
    floor_cache = {arm: {r: {} for r in DEPTH_RANKS} for arm in arms}
    for arm in arms:
        nb = nb_mapped[arm]
        for r_pos in DEPTH_RANKS:
            r_idx = r_pos - 1
            for c in range(n_cells):
                rows = (g_cell == c)
                sub = nb[rows, r_idx]
                row = np.bincount(sub, minlength=n_cells + 1)
                intruder_count = int(row[:n_cells].sum())
                if intruder_count == 0:
                    floor_cache[arm][r_pos][c] = (0, 0.0); continue
                fl = multinomial_floor(intruder_count, base_rates,
                                       n_sim=FLOOR_SIM, seed=42 + c * 11 + r_pos)
                floor_cache[arm][r_pos][c] = (intruder_count, fl)

    # Per-cell observed enrichment mean |e-1| at r=1 (and all depths)
    print("  per-cell observed enrichment (point estimates)...", flush=True)
    per_cell = {arm: {r: {} for r in DEPTH_RANKS} for arm in arms}
    for arm in arms:
        nb = nb_mapped[arm]
        for r_pos in DEPTH_RANKS:
            r_idx = r_pos - 1
            for c in range(n_cells):
                rows = (g_cell == c)
                sub = nb[rows, r_idx]
                row = np.bincount(sub, minlength=n_cells + 1).astype(np.int64)
                intruder_count = int(row[:n_cells].sum())
                if intruder_count == 0:
                    per_cell[arm][r_pos][c] = None; continue
                row_intruder = row[:n_cells]
                shares = row_intruder / intruder_count
                enrich = shares / base_rates
                mad = float(np.mean(np.abs(enrich - 1)))
                _, fl = floor_cache[arm][r_pos][c]
                per_cell[arm][r_pos][c] = {
                    'intruder_count': intruder_count,
                    'raw': mad,
                    'floor': fl,
                    'excess': mad - fl,
                    'own_share': float(shares[c]),
                    'own_enrichment': float(enrich[c]),
                }

    # Headline: 3x3 grid of (own-cell enrichment - floor) for id and ood at r=1
    print("\n========== 3x3 cells: 'excess' mean |e-1| at r=1, id arm ==========")
    print("  rows = spread tercile (0=low/compact, 2=high/diffuse)")
    print("  cols = difficulty tercile (0=low err, 2=high err)")
    print("  [id arm]")
    print("          d=0    d=1    d=2")
    for s in range(3):
        row_str = f"  s={s}:"
        for d in range(3):
            c = s * 3 + d
            cell = per_cell['id'][1].get(c)
            if cell is None:
                row_str += "   --  "
            else:
                row_str += f"  {cell['excess']:+.3f}"
        print(row_str)
    print("  [ood arm]")
    print("          d=0    d=1    d=2")
    for s in range(3):
        row_str = f"  s={s}:"
        for d in range(3):
            c = s * 3 + d
            cell = per_cell['ood'][1].get(c)
            row_str += f"  {cell['excess']:+.3f}" if cell else "   --  "
        print(row_str)

    out = {
        'cell_counts_ids': cell_counts_ids,
        'cell_items': cell_item_counts,
        'per_cell_point': {
            arm: {str(r): {str(c): per_cell[arm][r][c] for c in range(n_cells)
                            if per_cell[arm][r][c] is not None}
                   for r in DEPTH_RANKS}
            for arm in arms},
    }
    out_path = RESULTS / 'e42_doublestrat.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved → {out_path}")


if __name__ == '__main__':
    main()
