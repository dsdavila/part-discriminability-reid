"""
Own-class-rate 3x3 grids: difficulty x k-occurrence AND spread x k-occurrence.
Shares neighbour computation across both grids (same identities, same arms).
This is the pre-registered-sign test per Ch2 rev 138/141.

Metric per (identity, arm, rank r):
  rate_r = (same-id count in top-r) / r
Per-cell aggregate = weighted mean rate (identity-bootstrapped).

Row effect at fixed row-s (difficulty or spread tercile):
  high-kocc minus low-kocc at fixed row. Negative at r=1 for id arm
  means hubs crowd out own-class neighbours (claude-chat's natural reading).
"""
import json, time
import numpy as np
from pathlib import Path

CACHE_DIR = Path('/tmp/e12_feature_cache')
RESULTS   = Path('/data/papers/low_dimensional_reid/results')
HUBNESS   = RESULTS / 'hubness_koccurrence.json'
N_BOOT = 500
K_MAX = 20
DEPTH_RANKS = [1, 5, 10, 20]

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


def compute_same_id_mask(G, g_pids, g_cids, k_max=K_MAX, chunk=128):
    """For each item i, return a (N, k_max) boolean: 1 if top-r neighbour is same-id, else 0.
    Same-cam-same-pid excluded."""
    N = len(G)
    is_same = np.zeros((N, k_max), dtype=np.int8)
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
                is_same[gi, r] = 1 if g_pids[nb] == qpid else 0
    return is_same


def run_one_grid(arms_same, g_pids, pid_to_cell, cell_pids, grid_name,
                  row_axis_name, col_axis_name):
    """
    Bootstrap own-class rate on a given partition.
    arms_same: {arm: (N, k_max) int8 — 1 if same-id neighbour at that rank}
    """
    pid_to_qi = {}
    for qi, pid in enumerate(g_pids):
        pid_to_qi.setdefault(int(pid), []).append(qi)

    # Per-id per-arm cumulative same-id counts at each rank cutoff:
    #   same_cum[pid][arm][r] = sum over items of same-id matches in top-r
    #   item_cnt[pid] = number of items
    per_id = {}
    for arm in arms_same:
        for pid in pid_to_qi:
            if pid not in per_id: per_id[pid] = {}
            per_id[pid][arm] = {}
    for arm, arr in arms_same.items():
        for pid, qi_list in pid_to_qi.items():
            per_id[pid][arm]['same_in_r'] = {}
            qi_arr = np.array(qi_list)
            subs = arr[qi_arr]  # (n_items, k_max)
            for r_pos in DEPTH_RANKS:
                per_id[pid][arm]['same_in_r'][r_pos] = int(subs[:, :r_pos].sum())
            per_id[pid]['n_items'] = int(len(qi_list))

    # Bootstrap: resample identities within each cell with replacement
    rng = np.random.default_rng(42)
    boot = {arm: {r: {'cells': {c: [] for c in range(9)}} for r in DEPTH_RANKS}
            for arm in arms_same}
    row_eff = {arm: {r: {s: [] for s in range(3)} for r in DEPTH_RANKS}
               for arm in arms_same}
    col_eff = {arm: {r: {d: [] for d in range(3)} for r in DEPTH_RANKS}
               for arm in arms_same}

    t0 = time.time()
    for b in range(N_BOOT):
        resampled = {c: rng.choice(cell_pids[c], size=len(cell_pids[c]), replace=True)
                     for c in range(9)}
        for arm in arms_same:
            for r_pos in DEPTH_RANKS:
                cell_rate = {}
                for c in range(9):
                    same_sum = 0; total = 0
                    for pid in resampled[c]:
                        pid = int(pid)
                        same_sum += per_id[pid][arm]['same_in_r'][r_pos]
                        total += per_id[pid]['n_items'] * r_pos
                    cell_rate[c] = same_sum / total if total > 0 else 0.0
                for s in range(3):
                    row_eff[arm][r_pos][s].append(cell_rate[s*3+2] - cell_rate[s*3+0])
                for d in range(3):
                    col_eff[arm][r_pos][d].append(cell_rate[2*3+d] - cell_rate[0*3+d])
                # Also store cell rates themselves for raw output
                for c in range(9):
                    boot[arm][r_pos]['cells'][c].append(cell_rate[c])
        if (b + 1) % 100 == 0:
            print(f"    [{grid_name}] boot {b+1}/{N_BOOT} t={(time.time()-t0)/60:.1f}m", flush=True)

    def ci(xs):
        a = np.array(xs)
        return {'mean': float(a.mean()), 'median': float(np.median(a)),
                'lo': float(np.quantile(a, 0.025)), 'hi': float(np.quantile(a, 0.975))}

    out = {'row_axis': row_axis_name, 'col_axis': col_axis_name,
           'cell_id_counts': {c: len(cell_pids[c]) for c in range(9)},
           'per_arm': {}}
    for arm in arms_same:
        out['per_arm'][arm] = {}
        for r_pos in DEPTH_RANKS:
            out['per_arm'][arm][str(r_pos)] = {
                'cell_rates': {str(c): {**ci(boot[arm][r_pos]['cells'][c]),
                                         'raw': boot[arm][r_pos]['cells'][c]}
                                for c in range(9)},
                'row_effects': {str(s): {**ci(row_eff[arm][r_pos][s]),
                                          'raw': row_eff[arm][r_pos][s]}
                                 for s in range(3)},
                'col_effects': {str(d): {**ci(col_eff[arm][r_pos][d]),
                                          'raw': col_eff[arm][r_pos][d]}
                                 for d in range(3)},
            }

    print(f"\n========== {grid_name}: Own-class rate 3x3 ==========")
    for arm in ('id', 'ood'):
        print(f"\n[{arm}]")
        for r_pos in DEPTH_RANKS:
            print(f"  r={r_pos}:")
            for s in range(3):
                c = ci(row_eff[arm][r_pos][s])
                sig = "*" if c['lo'] * c['hi'] > 0 else " "
                print(f"    row s={s} (high-kocc rate - low-kocc rate){sig}: {c['median']:+.4f} [{c['lo']:+.4f}, {c['hi']:+.4f}]")
            for d in range(3):
                c = ci(col_eff[arm][r_pos][d])
                sig = "*" if c['lo'] * c['hi'] > 0 else " "
                print(f"    col d={d} (row_eff hi-lo){sig}: {c['median']:+.4f} [{c['lo']:+.4f}, {c['hi']:+.4f}]")
    return out


def main():
    print("Loading MSMT...", flush=True)
    G_id, g_pids, g_cids = load_cell('D')
    G_ood, _, _ = load_cell("B'")

    print("Loading hubness JSON...", flush=True)
    hub = json.load(open(HUBNESS))
    kocc_mean_id = {int(k): v for k, v in hub['MSMT']['per_id_occ']['id']['mean'].items()}
    kocc_bin = tercile_assign(kocc_mean_id)

    print("Computing per-id difficulty (rank-1 err)...", flush=True)
    diff = per_id_rank1_err(G_id, g_pids, g_cids)
    diff_bin = tercile_assign(diff)
    diff_cell = {int(p): diff_bin[int(p)] * 3 + kocc_bin[int(p)]
                  for p in diff.keys()}
    diff_cell_pids = {c: [] for c in range(9)}
    for pid, c in diff_cell.items():
        diff_cell_pids[c].append(int(pid))
    print(f"  diff-x-kocc cells: {[len(diff_cell_pids[c]) for c in range(9)]}", flush=True)

    print("Computing per-id spread_p75...", flush=True)
    spread = per_id_spread_p75(G_id, g_pids)
    spread_bin = tercile_assign(spread)
    spread_cell = {int(p): spread_bin[int(p)] * 3 + kocc_bin[int(p)]
                    for p in spread.keys()}
    spread_cell_pids = {c: [] for c in range(9)}
    for pid, c in spread_cell.items():
        spread_cell_pids[c].append(int(pid))
    print(f"  spread-x-kocc cells: {[len(spread_cell_pids[c]) for c in range(9)]}", flush=True)

    # Build the four arms, compute same-id masks once per arm (shared across both grids)
    G_noise = noise_degrade(G_id, SIGMA_D)
    G_rp = randproj_degrade(G_id, K_RANDPROJ)
    arms = {'id': G_id, 'ood': G_ood, 'noise': G_noise, 'randproj': G_rp}
    arms_same = {}
    for arm_name, G in arms.items():
        t0 = time.time()
        print(f"  [{arm_name}] top-{K_MAX} same-id mask...", flush=True)
        arms_same[arm_name] = compute_same_id_mask(G, g_pids, g_cids)
        print(f"    t={(time.time()-t0)/60:.1f}m", flush=True)

    out_diff = run_one_grid(arms_same, g_pids, diff_cell, diff_cell_pids,
                             'diff_x_kocc',
                             'difficulty_rank1err_tercile',
                             'kocc_mean_id_arm_tercile')
    out_spread = run_one_grid(arms_same, g_pids, spread_cell, spread_cell_pids,
                               'spread_x_kocc',
                               'spread_p75_tercile',
                               'kocc_mean_id_arm_tercile')

    out = {'diff_x_kocc': out_diff, 'spread_x_kocc': out_spread,
           'n_boot': N_BOOT, 'metric': 'own_class_rate_in_top_r'}
    out_path = RESULTS / 'e42_ownclass_grids.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == '__main__':
    main()
