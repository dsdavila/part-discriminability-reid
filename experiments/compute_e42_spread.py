"""
E42 (c) + (a) re-stratified by within_id_spread_p75 terciles instead of
in-domain-difficulty deciles (Channel 2 rev 98).

Question: is transfer's ~92% neighbourhood-structure retention a property
of the difficulty partition, or of compactness?

Uses the same v4 machinery: multinomial floor per (arm, decile, rank),
common item-count weighting, aggregate-then-ratio, depth sweep at r=1/5/10/20,
identity-bootstrap CIs on the floor-corrected retention.

Partition: 3 item-matched terciles of within_id_spread_p75 (same construction
as spread_p75_asc in compute_e40_spread_centroid_null.py), target item counts
7,160/14,797/60,162 to match the real EASY/SHARED/OOD-ONLY sizes.
"""
import json, time
import numpy as np
from pathlib import Path

CACHE_DIR = Path('/tmp/e12_feature_cache')
RESULTS   = Path('/data/papers/low_dimensional_reid/results')
N_BOOT = 500
K_MAX = 20
DEPTH_RANKS = [1, 5, 10, 20]
FLOOR_SIM = 2000
TARGET_ITEMS = {'EASY': 7160, 'SHARED': 14797, 'OOD-ONLY': 60162}

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
        angles = np.arccos(cos_x)
        out[int(pid)] = float(np.percentile(angles, 75))
    return out


def build_item_matched_partition(g_pids, covariate_by_pid, target_items):
    unique_pids = sorted(covariate_by_pid.keys())
    sorted_pids = sorted(unique_pids, key=lambda p: covariate_by_pid[p])
    pid_items = {int(p): int((g_pids == p).sum()) for p in unique_pids}
    class_map = {}; cursor = 0; actual_items = {}
    classes = list(target_items.keys())
    for cls in classes:
        target = target_items[cls]
        acc = 0
        while cursor < len(sorted_pids) and acc < target:
            pid = sorted_pids[cursor]
            acc += pid_items[int(pid)]
            class_map[int(pid)] = cls
            cursor += 1
        actual_items[cls] = acc
    return class_map, actual_items


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

    print("Computing per-id spread_p75...", flush=True)
    spread_p75 = per_id_spread_p75(G_id, g_pids)

    # Partition by spread_p75 terciles item-matched to real partition's item counts.
    # Classes: 'EASY' (low spread = tight centroid), 'SHARED' (middle), 'OOD-ONLY' (high)
    class_map, actual_items = build_item_matched_partition(
        g_pids, spread_p75, TARGET_ITEMS)
    classes = list(TARGET_ITEMS.keys())
    n_bins = len(classes)
    pid_to_bin = {p: classes.index(v) for p, v in class_map.items()}
    id_counts = {c: sum(1 for v in class_map.values() if v == c) for c in classes}
    print(f"  partition: id_counts={id_counts}  item_counts={actual_items}", flush=True)
    bin_item_counts = {i: actual_items[c] for i, c in enumerate(classes)}
    total_items = sum(bin_item_counts.values())
    base_rates = np.array([bin_item_counts[b] / total_items for b in range(n_bins)])

    # Four arms
    G_noise = noise_degrade(G_id, SIGMA_D)
    G_rp = randproj_degrade(G_id, K_RANDPROJ)
    arms = {'id': G_id, 'ood': G_ood, 'noise': G_noise, 'randproj': G_rp}

    # Neighbour deciles per query per arm
    arm_nb = {}
    for arm_name, G in arms.items():
        t0 = time.time()
        print(f"  [{arm_name}] top-{K_MAX} neighbours...", flush=True)
        nb_dec, _ = compute_neighbour_deciles(G, g_pids, g_cids, pid_to_bin, n_bins)
        arm_nb[arm_name] = nb_dec
        print(f"    t={(time.time() - t0) / 60:.1f}m", flush=True)

    nb_mapped = {arm: a.copy() for arm, a in arm_nb.items()}
    for arm in arms:
        nb_mapped[arm][nb_mapped[arm] == -1] = n_bins
        nb_mapped[arm] = nb_mapped[arm].astype(np.int16)

    g_decile = np.array([pid_to_bin.get(int(p), -1) for p in g_pids], dtype=np.int8)

    # Precompute static floors per (arm, decile, rank)
    print("  precomputing floors...", flush=True)
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

    # Identity-bootstrap: resample identities within each decile, recompute
    # aggregate excess under common item weights, then ood/id ratio.
    print("  identity bootstrap...", flush=True)
    rng = np.random.default_rng(42)
    unique_pids_sorted = np.array(sorted(pid_to_bin.keys()))
    decile_to_pids = {d: [p for p in unique_pids_sorted if pid_to_bin[int(p)] == d]
                      for d in range(n_bins)}
    pid_to_qi = {}
    for qi, pid in enumerate(g_pids):
        pid_to_qi.setdefault(int(pid), []).append(qi)

    boot_excess = {arm: {r: [] for r in DEPTH_RANKS} for arm in arms}
    boot_retention = {r: [] for r in DEPTH_RANKS}

    t_total = time.time()
    for b in range(N_BOOT):
        resampled_q_idx = {}
        for d in range(n_bins):
            pids = decile_to_pids[d]
            sampled = rng.choice(pids, size=len(pids), replace=True)
            q_idx_list = []
            for pid in sampled:
                q_idx_list.extend(pid_to_qi.get(int(pid), []))
            resampled_q_idx[d] = np.array(q_idx_list, dtype=np.int64) if q_idx_list else np.array([], dtype=np.int64)
        per_arm_excess_r = {r: {} for r in DEPTH_RANKS}
        for arm_name in arms:
            for r_pos in DEPTH_RANKS:
                r_idx = r_pos - 1
                weighted_excess_sum = 0.0; weight_sum = 0
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
                    _, fl = floor_cache[arm_name][r_pos][d]
                    excess_d = mad - fl
                    w = bin_item_counts[d]
                    weighted_excess_sum += excess_d * w
                    weight_sum += w
                excess = weighted_excess_sum / max(weight_sum, 1)
                boot_excess[arm_name][r_pos].append(excess)
                per_arm_excess_r[r_pos][arm_name] = excess
        for r_pos in DEPTH_RANKS:
            id_e = per_arm_excess_r[r_pos].get('id', 0)
            ood_e = per_arm_excess_r[r_pos].get('ood', 0)
            if id_e > 1e-9:
                boot_retention[r_pos].append(ood_e / id_e)
            else:
                boot_retention[r_pos].append(float('nan'))
        if (b + 1) % 100 == 0:
            print(f"    boot {b+1}/{N_BOOT}: t={(time.time() - t_total) / 60:.1f}m", flush=True)

    summary = {'id_counts': id_counts, 'item_counts': actual_items,
               'depth_sweep': {}}
    for r_pos in DEPTH_RANKS:
        summary['depth_sweep'][str(r_pos)] = {'per_arm': {}}
        for arm in arms:
            arr = np.array(boot_excess[arm][r_pos])
            summary['depth_sweep'][str(r_pos)]['per_arm'][arm] = {
                'excess_mean': float(arr.mean()),
                'excess_lo': float(np.quantile(arr, 0.025)),
                'excess_hi': float(np.quantile(arr, 0.975)),
            }
        ret_arr = np.array(boot_retention[r_pos])
        ret_arr = ret_arr[~np.isnan(ret_arr)]
        summary['depth_sweep'][str(r_pos)]['retention_ood_vs_id'] = {
            'median': float(np.median(ret_arr)),
            'lo': float(np.quantile(ret_arr, 0.025)),
            'hi': float(np.quantile(ret_arr, 0.975)),
        }

    print("\n========== MSMT, SPREAD partition — floor-corrected retention ==========")
    for arm in arms:
        print(f"  {arm:<10}", end='  ')
        for r in DEPTH_RANKS:
            e = summary['depth_sweep'][str(r)]['per_arm'][arm]['excess_mean']
            print(f"r={r}: {e:.4f}", end='  ')
        print()
    print("  retention ood/id by depth:")
    for r in DEPTH_RANKS:
        ret = summary['depth_sweep'][str(r)]['retention_ood_vs_id']
        print(f"    r={r:>2}: {ret['median']:.3f} [{ret['lo']:.3f}, {ret['hi']:.3f}]")

    out_path = RESULTS / 'e42_spread_partition.json'
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved → {out_path}")


if __name__ == '__main__':
    main()
