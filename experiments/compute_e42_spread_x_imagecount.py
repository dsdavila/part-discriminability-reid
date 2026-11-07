"""
3x3 spread_p75 x image-count double stratification (Ch2 rev 115 open threat
to Leg 2). Does the compactness effect survive at fixed image count?

Spread (within_id_spread_p75) and image count (items per identity) are
correlated candidate variables for the "compactness" compactness-like signal.
Ruling 12: marginal comparison identifies nothing; 3x3 double stratification
required.

Per identity bootstrap on row and column effects. Per-draw arrays retained
(ruling 18). Same v2 machinery as compute_e42_doublestrat_boot_v2.py.
"""
import json, time
import numpy as np
from pathlib import Path

CACHE_DIR = Path('/tmp/e12_feature_cache')
RESULTS   = Path('/data/papers/low_dimensional_reid/results')
N_BOOT = 500
K_MAX = 20
DEPTH_RANKS = [1, 5, 10, 20]
FLOOR_SIM = 1000

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


def compute_neighbour_cells(G, g_pids, g_cids, pid_to_cell, n_cells=9,
                              k_max=K_MAX, chunk=128):
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
    print("Loading MSMT...", flush=True)
    G_id, g_pids, g_cids = load_cell('D')
    G_ood, _, _ = load_cell("B'")

    # Partitions: spread tercile (row) x image-count tercile (col)
    spread = per_id_spread_p75(G_id, g_pids)
    items_by_pid = {int(p): int((g_pids == p).sum()) for p in np.unique(g_pids)}
    spread_bin = tercile_assign(spread)
    items_bin = tercile_assign(items_by_pid)
    pid_to_cell = {int(p): spread_bin[int(p)] * 3 + items_bin[int(p)]
                    for p in spread.keys()}
    cell_pids = {c: [] for c in range(9)}
    for pid, c in pid_to_cell.items():
        cell_pids[c].append(int(pid))
    cell_id_counts = {c: len(v) for c, v in cell_pids.items()}
    g_cell = np.array([pid_to_cell.get(int(p), -1) for p in g_pids], dtype=np.int8)
    cell_item_counts = {c: int((g_cell == c).sum()) for c in range(9)}
    print(f"  cell id counts (row=spread tercile, col=items tercile):", flush=True)
    for s in range(3):
        print(f"    s={s}: {[cell_id_counts[s*3+d] for d in range(3)]}", flush=True)
    print(f"  cell item counts:", flush=True)
    for s in range(3):
        print(f"    s={s}: {[cell_item_counts[s*3+d] for d in range(3)]}", flush=True)

    total_items = sum(cell_item_counts.values())
    base_rates = np.array([cell_item_counts[c] / total_items for c in range(9)])

    G_noise = noise_degrade(G_id, SIGMA_D)
    G_rp = randproj_degrade(G_id, K_RANDPROJ)
    arms = {'id': G_id, 'ood': G_ood, 'noise': G_noise, 'randproj': G_rp}
    arm_nb = {}
    for arm_name, G in arms.items():
        t0 = time.time()
        print(f"  [{arm_name}] top-{K_MAX} neighbours...", flush=True)
        nb, _ = compute_neighbour_cells(G, g_pids, g_cids, pid_to_cell)
        arm_nb[arm_name] = nb
        print(f"    t={(time.time() - t0) / 60:.1f}m", flush=True)

    nb_mapped = {arm: a.copy() for arm, a in arm_nb.items()}
    for arm in arms:
        nb_mapped[arm][nb_mapped[arm] == -1] = 9
        nb_mapped[arm] = nb_mapped[arm].astype(np.int16)

    pid_to_qi = {}
    for qi, pid in enumerate(g_pids):
        pid_to_qi.setdefault(int(pid), []).append(qi)

    print("  building per-identity records...", flush=True)
    per_id_records = {arm: {} for arm in arms}
    for arm in arms:
        nb = nb_mapped[arm]
        for c in range(9):
            for pid in cell_pids[c]:
                q_idx = pid_to_qi[int(pid)]
                rec = {'cell': c, 'n_items': len(q_idx), 'bincounts': {}}
                for r_pos in DEPTH_RANKS:
                    r_idx = r_pos - 1
                    rec['bincounts'][r_pos] = np.bincount(nb[q_idx, r_idx], minlength=10)
                per_id_records[arm][int(pid)] = rec

    print("  bootstrap on row/col effects...", flush=True)
    rng = np.random.default_rng(42)
    boot_effects = {arm: {r: {'row': {s: [] for s in range(3)},
                                'col': {d: [] for d in range(3)}}
                           for r in DEPTH_RANKS}
                     for arm in arms}

    t_total = time.time()
    for b in range(N_BOOT):
        resampled = {c: rng.choice(cell_pids[c], size=len(cell_pids[c]), replace=True)
                     for c in range(9)}
        for arm in arms:
            for r_pos in DEPTH_RANKS:
                cell_agg = {}
                for c in range(9):
                    agg = np.zeros(10, dtype=np.int64)
                    for pid in resampled[c]:
                        agg += per_id_records[arm][int(pid)]['bincounts'][r_pos]
                    cell_agg[c] = agg
                cell_excess = {}
                for c in range(9):
                    intruder = cell_agg[c][:9]
                    N = int(intruder.sum())
                    if N == 0:
                        cell_excess[c] = 0.0; continue
                    shares = intruder / N
                    enrich = shares / base_rates
                    mad = float(np.mean(np.abs(enrich - 1)))
                    floor = multinomial_floor(N, base_rates, n_sim=FLOOR_SIM,
                                              seed=42 + c * 11 + r_pos * 3 + b)
                    cell_excess[c] = mad - floor
                for s in range(3):
                    row_eff = cell_excess[s * 3 + 2] - cell_excess[s * 3 + 0]
                    boot_effects[arm][r_pos]['row'][s].append(row_eff)
                for d in range(3):
                    col_eff = cell_excess[2 * 3 + d] - cell_excess[0 * 3 + d]
                    boot_effects[arm][r_pos]['col'][d].append(col_eff)
        if (b + 1) % 50 == 0:
            print(f"    boot {b + 1}/{N_BOOT}: t={(time.time() - t_total) / 60:.1f}m", flush=True)

    def ci(xs):
        a = np.array(xs)
        return {'mean': float(a.mean()), 'median': float(np.median(a)),
                'lo': float(np.quantile(a, 0.025)), 'hi': float(np.quantile(a, 0.975))}

    out = {'row_axis': 'spread_p75_tercile',
           'col_axis': 'items_per_id_tercile',
           'cell_id_counts': cell_id_counts,
           'cell_item_counts': cell_item_counts,
           'per_arm': {}}
    for arm in arms:
        out['per_arm'][arm] = {}
        for r_pos in DEPTH_RANKS:
            out['per_arm'][arm][str(r_pos)] = {
                'row_effects': {str(s): {**ci(boot_effects[arm][r_pos]['row'][s]),
                                          'raw': boot_effects[arm][r_pos]['row'][s]}
                                 for s in range(3)},
                'col_effects': {str(d): {**ci(boot_effects[arm][r_pos]['col'][d]),
                                          'raw': boot_effects[arm][r_pos]['col'][d]}
                                 for d in range(3)},
            }

    print("\n========== Spread x Image-count 3x3 bootstrap ==========")
    for arm in ('id', 'ood'):
        print(f"\n[{arm}]")
        for r_pos in DEPTH_RANKS:
            print(f"  r={r_pos}:")
            for s in range(3):
                c = ci(boot_effects[arm][r_pos]['row'][s])
                print(f"    row s={s} (high-low img): {c['median']:+.4f} [{c['lo']:+.4f}, {c['hi']:+.4f}]")
            for d in range(3):
                c = ci(boot_effects[arm][r_pos]['col'][d])
                print(f"    col d={d} (diffuse-compact): {c['median']:+.4f} [{c['lo']:+.4f}, {c['hi']:+.4f}]")

    # Real-minus-noise paired CIs
    print("\n  real-minus-noise paired CIs (col d, diffuse-compact):")
    for arm_null in ('noise',):
        for arm_real in ('id', 'ood'):
            print(f"  {arm_real} - {arm_null}:")
            for r_pos in DEPTH_RANKS:
                for d in range(3):
                    real = np.array(boot_effects[arm_real][r_pos]['col'][d])
                    null = np.array(boot_effects[arm_null][r_pos]['col'][d])
                    diff = real - null
                    print(f"    r={r_pos} col d={d}: {np.median(diff):+.4f} "
                          f"[{np.quantile(diff, 0.025):+.4f}, {np.quantile(diff, 0.975):+.4f}]")

    out_path = RESULTS / 'e42_spread_x_imagecount.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved → {out_path}")


if __name__ == '__main__':
    main()
