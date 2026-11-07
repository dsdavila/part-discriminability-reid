"""
E42 (c) + (a) v3: adds the multinomial floor correction and the four-depth
retention table per claude-chat Channel 2 rev 80 + 82.

Headline changes from v2:
  - Per-arm, per-decile, per-rank multinomial FLOOR: expected mean |e-1| under
    the null "same intruder count drawn proportionally to base rates." Floor is
    simulated (10k draws) once from the observed intruder count and base rates.
  - Signal excess = observed - floor  (how much structure is kept)
  - Ratio to floor = observed / floor  (significance of structure's presence)
  - Retention vs id = ood_excess / id_excess, with paired CI.
  - DEPTH SWEEP at r = 1, 5, 10, 20 as the primary display.
  - Identity-bootstrap CIs on the floor-corrected signal, not the raw.
"""
import json, time
import numpy as np
from pathlib import Path

CACHE_DIR = Path('/tmp/e12_feature_cache')
RESULTS   = Path('/data/papers/low_dimensional_reid/results')
PARTITION_PATH = RESULTS / 'e41_partition_frozen.json'
N_BOOT = 500
K_MAX = 20
DEPTH_RANKS = [1, 5, 10, 20]
FLOOR_SIM = 2000

CELLS = {
    'D':  ('bot_msmt_seed42_msmt.npz',  11659),
    "B'": ('msmt17_test_noREA.npz',     11659),
    "A'": ('market_test_noREA.npz',      3368),
    'C':  ('bot_msmt_seed42_market.npz', 3368),
}
SIGMA_D = 0.0625
SIGMA_A = 0.062
K_RANDPROJ = 17


def load_cell(name):
    fn, n_q = CELLS[name]
    d = np.load(CACHE_DIR / fn); emb = d['emb'].astype(np.float32)
    return (emb[n_q:], d['pids'][n_q:], d['cids'][n_q:])


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


def compute_neighbour_deciles(G, g_pids, g_cids, pid_to_bin, n_bins,
                              k_max=K_MAX, chunk=128):
    N = len(G)
    g_decile = np.array([pid_to_bin.get(int(p), -1) for p in g_pids], dtype=np.int8)
    neighbour_dec = np.full((N, k_max), -2, dtype=np.int8)
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
                    neighbour_dec[gi, r] = -1
                else:
                    neighbour_dec[gi, r] = g_decile[nb]
    return neighbour_dec, g_decile


def observed_mean_abs_dev(counts_at_rank, n_bins, base_rates):
    """From a (n_bins,) vector of intruder counts at one rank for one query-decile
    (one row of counts[d_q, :n_bins, r]), compute mean |enrichment - 1| over
    neighbour deciles. intruder_total = sum of counts.
    """
    total = counts_at_rank.sum()
    if total == 0:
        return 0.0
    shares = counts_at_rank / total
    enrich = shares / base_rates
    return float(np.mean(np.abs(enrich - 1)))


def multinomial_floor(N, base_rates, n_sim=FLOOR_SIM, seed=42):
    """E[mean |e - 1|] under Multinomial(N, p=base_rates)."""
    rng = np.random.default_rng(seed)
    if N == 0: return 0.0
    draws = rng.multinomial(N, base_rates, size=n_sim)
    shares = draws / N
    enrich = shares / base_rates
    return float(np.mean(np.abs(enrich - 1), axis=1).mean())


def main():
    print("Loading frozen partition...", flush=True)
    with open(PARTITION_PATH) as f:
        part = json.load(f)

    out = {}
    for dataset, (id_cell, ood_cell, sigma, k_rp) in [
        ('MSMT', ('D', "B'", SIGMA_D, K_RANDPROJ)),
        ('Market', ("A'", 'C', SIGMA_A, K_RANDPROJ)),
    ]:
        print(f"\n########## {dataset}: {id_cell}/{ood_cell} ##########", flush=True)
        G_id, g_pids, g_cids = load_cell(id_cell)
        G_ood, _, _ = load_cell(ood_cell)
        n_bins = part[dataset]['n_bins']
        pid_to_bin = {int(k): int(v) for k, v in part[dataset]['assignment'].items()}
        g_decile = np.array([pid_to_bin.get(int(p), -1) for p in g_pids], dtype=np.int8)
        bin_item_counts = {b: int((g_decile == b).sum()) for b in range(n_bins)}
        total_items = sum(bin_item_counts.values())
        base_rates = np.array([bin_item_counts[b] / total_items for b in range(n_bins)])
        print(f"  n_bins={n_bins}  base_rates={base_rates.round(4).tolist()}", flush=True)

        G_noise = noise_degrade(G_id, sigma)
        G_rp = randproj_degrade(G_id, k_rp)
        arms = {'id': G_id, 'ood': G_ood, 'noise': G_noise, 'randproj': G_rp}
        arm_nb = {}
        for arm_name, G_ in arms.items():
            t0 = time.time()
            print(f"  [{arm_name}] top-{K_MAX} neighbours...", flush=True)
            nb_dec, _ = compute_neighbour_deciles(G_, g_pids, g_cids, pid_to_bin, n_bins)
            arm_nb[arm_name] = nb_dec
            print(f"    t={(time.time() - t0) / 60:.1f}m", flush=True)

        # pre-map (-1 -> n_bins) for bincount
        nb_mapped = {arm: a.copy() for arm, a in arm_nb.items()}
        for arm in arms: nb_mapped[arm][nb_mapped[arm] == -1] = n_bins
        for arm in arms: nb_mapped[arm] = nb_mapped[arm].astype(np.int16)

        # Precompute per-decile pid lists and pid-to-queries map
        unique_pids_sorted = np.array(sorted(pid_to_bin.keys()))
        decile_to_pids = {d: [int(p) for p in unique_pids_sorted if pid_to_bin[int(p)] == d]
                          for d in range(n_bins)}
        pid_to_qi = {}
        for qi, pid in enumerate(g_pids):
            pid_to_qi.setdefault(int(pid), []).append(qi)

        print("  precomputing static floors per (arm, decile, rank)...", flush=True)
        # Pre-cache floor by (arm, decile, rank) using the full-sample intruder count
        floor_cache = {arm: {r: {} for r in DEPTH_RANKS} for arm in arms}
        for arm in arms:
            nb = nb_mapped[arm]
            for r_pos in DEPTH_RANKS:
                r_idx = r_pos - 1
                for d in range(n_bins):
                    rows = (g_decile == d)
                    sub = nb[rows, r_idx]
                    row = np.bincount(sub, minlength=n_bins + 1)
                    intruder_count = int(row[:n_bins].sum())
                    if intruder_count == 0:
                        floor_cache[arm][r_pos][d] = (0, 0.0); continue
                    fl = multinomial_floor(intruder_count, base_rates,
                                           n_sim=FLOOR_SIM, seed=42 + d * 7 + r_pos)
                    floor_cache[arm][r_pos][d] = (intruder_count, fl)

        print("  bootstrap at depths r = 1, 5, 10, 20...", flush=True)
        t_total = time.time()
        rng = np.random.default_rng(42)

        # For each arm, store per-depth per-bootstrap:
        #   raw mean |e-1|  (aggregate over all query deciles)
        #   floor (mean of per-decile floors, weighted by intruder count)
        #   signal excess = raw - floor
        boot_raw = {arm: {r: [] for r in DEPTH_RANKS} for arm in arms}
        boot_floor = {arm: {r: [] for r in DEPTH_RANKS} for arm in arms}
        boot_excess = {arm: {r: [] for r in DEPTH_RANKS} for arm in arms}
        boot_retention = {r: [] for r in DEPTH_RANKS}  # ood_excess / id_excess

        # Also store per-decile self-affinity (own-decile enrichment) at each depth
        # for the headline table.
        boot_self_aff = {arm: {r: {d: [] for d in range(n_bins)}
                                 for r in DEPTH_RANKS} for arm in arms}

        for b in range(N_BOOT):
            resampled_q_idx = {}
            for d in range(n_bins):
                pids = decile_to_pids[d]
                if not pids: continue
                sampled = rng.choice(pids, size=len(pids), replace=True)
                q_idx_list = []
                for pid in sampled:
                    q_idx_list.extend(pid_to_qi.get(int(pid), []))
                resampled_q_idx[d] = np.array(q_idx_list, dtype=np.int64) if q_idx_list else np.array([], dtype=np.int64)

            per_arm_excess_r = {r: {} for r in DEPTH_RANKS}
            for arm_name in arms:
                for r_pos in DEPTH_RANKS:
                    r_idx = r_pos - 1
                    # aggregate mean |e-1| across all query deciles, weighted by intruder count
                    all_pairs_sum = 0.0
                    all_pairs_count = 0
                    floor_sum = 0.0
                    floor_count = 0
                    for d in range(n_bins):
                        q_idx = resampled_q_idx.get(d)
                        if q_idx is None or len(q_idx) == 0: continue
                        sub = nb_mapped[arm_name][q_idx, r_idx]
                        row = np.bincount(sub, minlength=n_bins + 1).astype(np.int64)
                        intruder_count = int(row[:n_bins].sum())
                        if intruder_count == 0: continue
                        row_intruder = row[:n_bins]
                        shares = row_intruder / intruder_count
                        enrich = shares / base_rates
                        mad = float(np.mean(np.abs(enrich - 1)))
                        all_pairs_sum += mad * intruder_count
                        all_pairs_count += intruder_count
                        # Use cached floor (full-sample intruder count approximation)
                        _, fl = floor_cache[arm_name][r_pos][d]
                        floor_sum += fl * intruder_count
                        floor_count += intruder_count
                        # per-decile self-affinity (own-decile enrichment)
                        if r_pos == 1:  # only at r=1 to save
                            e_d = enrich[d]
                            boot_self_aff[arm_name][r_pos][d].append(float(e_d))

                    raw = all_pairs_sum / max(all_pairs_count, 1)
                    floor = floor_sum / max(floor_count, 1)
                    excess = raw - floor
                    boot_raw[arm_name][r_pos].append(raw)
                    boot_floor[arm_name][r_pos].append(floor)
                    boot_excess[arm_name][r_pos].append(excess)
                    per_arm_excess_r[r_pos][arm_name] = excess

            # paired retention = ood_excess / id_excess per depth
            for r_pos in DEPTH_RANKS:
                id_e = per_arm_excess_r[r_pos].get('id', 0)
                ood_e = per_arm_excess_r[r_pos].get('ood', 0)
                if id_e > 1e-9:
                    boot_retention[r_pos].append(ood_e / id_e)
                else:
                    boot_retention[r_pos].append(float('nan'))

            if (b + 1) % 100 == 0:
                print(f"    boot {b+1}/{N_BOOT}: t={(time.time() - t_total) / 60:.1f}m", flush=True)

        # Aggregate static floor per (arm, rank) from the cache
        static_floor = {arm: {r: None for r in DEPTH_RANKS} for arm in arms}
        for arm in arms:
            for r_pos in DEPTH_RANKS:
                weights, floors = [], []
                for d in range(n_bins):
                    ic, fl = floor_cache[arm][r_pos][d]
                    if ic > 0:
                        weights.append(ic); floors.append(fl)
                static_floor[arm][r_pos] = (float(np.average(floors, weights=weights))
                                             if weights else None)

        # Summarise
        summary = {'n_bins': n_bins, 'bin_item_counts': bin_item_counts,
                   'base_rates': base_rates.tolist(),
                   'static_floor': static_floor,
                   'depth_sweep': {}}
        for r_pos in DEPTH_RANKS:
            summary['depth_sweep'][str(r_pos)] = {'per_arm': {}}
            for arm in arms:
                raw_arr = np.array(boot_raw[arm][r_pos])
                floor_arr = np.array(boot_floor[arm][r_pos])
                excess_arr = np.array(boot_excess[arm][r_pos])
                summary['depth_sweep'][str(r_pos)]['per_arm'][arm] = {
                    'raw_mean': float(raw_arr.mean()),
                    'raw_lo': float(np.quantile(raw_arr, 0.025)),
                    'raw_hi': float(np.quantile(raw_arr, 0.975)),
                    'floor_mean': float(floor_arr.mean()),
                    'excess_mean': float(excess_arr.mean()),
                    'excess_lo': float(np.quantile(excess_arr, 0.025)),
                    'excess_hi': float(np.quantile(excess_arr, 0.975)),
                    'ratio_to_floor_mean': float((raw_arr / np.maximum(floor_arr, 1e-9)).mean()),
                }
            ret_arr = np.array(boot_retention[r_pos])
            ret_arr = ret_arr[~np.isnan(ret_arr)]
            if len(ret_arr):
                summary['depth_sweep'][str(r_pos)]['retention_ood_vs_id'] = {
                    'median': float(np.median(ret_arr)),
                    'lo': float(np.quantile(ret_arr, 0.025)),
                    'hi': float(np.quantile(ret_arr, 0.975)),
                }

        # Per-decile self-affinity r=1 (already computed in loop)
        summary['self_aff_r1_per_decile'] = {}
        for arm in arms:
            summary['self_aff_r1_per_decile'][arm] = {}
            for d in range(n_bins):
                a = boot_self_aff[arm][1][d]
                if a:
                    summary['self_aff_r1_per_decile'][arm][str(d)] = {
                        'median': float(np.median(a)),
                        'lo': float(np.quantile(a, 0.025)),
                        'hi': float(np.quantile(a, 0.975)),
                    }
        out[dataset] = summary
        print(f"  {dataset} done. t={(time.time() - t_total) / 60:.1f}m", flush=True)

    # Headline print
    for dataset in ('MSMT', 'Market'):
        print(f"\n========== {dataset} — floor-corrected retention ==========")
        print(f"  {'arm':<10}  {'r=1':>18}  {'r=5':>18}  {'r=10':>18}  {'r=20':>18}")
        res = out[dataset]
        for arm in ('id', 'ood', 'noise', 'randproj'):
            vals = []
            for r in DEPTH_RANKS:
                e = res['depth_sweep'][str(r)]['per_arm'][arm]['excess_mean']
                raw = res['depth_sweep'][str(r)]['per_arm'][arm]['raw_mean']
                flr = res['depth_sweep'][str(r)]['per_arm'][arm]['floor_mean']
                vals.append(f"{e:.4f}(f={flr:.3f})")
            print(f"  {arm:<10}  {vals[0]:>18}  {vals[1]:>18}  {vals[2]:>18}  {vals[3]:>18}")
        print(f"  retention ood/id by depth:")
        for r in DEPTH_RANKS:
            ret = res['depth_sweep'][str(r)].get('retention_ood_vs_id')
            if ret:
                print(f"    r={r:>2}: {ret['median']:.3f} [{ret['lo']:.3f}, {ret['hi']:.3f}]")

    out_path = RESULTS / 'e42_ca_v3_floor.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved → {out_path}")


if __name__ == '__main__':
    main()
