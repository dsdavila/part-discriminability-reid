"""
Spread and centroid-norm comparator nulls for Leg 3 covariate analysis
(Channel 2 rev 94). Each comparator gets its own size-matched null per
the general rule.

Primary: within_id_spread_p75 (Leg 2's variable; if Leg 3 beats this it
survives as independent, if not Leg 3 nests under Leg 2).
Secondary: centroid_norm.

For each comparator:
  1. Compute per-identity value from D-cell gallery embeddings.
  2. Partition identities into three classes (EASY/SHARED/OOD-ONLY) by
     ascending tercile with item targets 7,160/14,797/60,162.
  3. Compute principal angles (common-k) on id and ood arms.
  4. Build n=100 size-matched random null at the ACHIEVED item counts.
  5. Report gaps.
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
N_PERM    = 100
CLASSES   = ['EASY', 'SHARED', 'OOD-ONLY']
TARGET_ITEMS = {'EASY': 7160, 'SHARED': 14797, 'OOD-ONLY': 60162}


def load_cell(name):
    cells = {'D': ('bot_msmt_seed42_msmt.npz', 11659),
             "B'": ('msmt17_test_noREA.npz', 11659)}
    fn, n_q = cells[name]
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


def compute_class_pair_medians(G, g_pids, class_map, k=FIXED_K):
    bases, k_effs = {}, {}
    for c in CLASSES:
        idx = np.array([class_map.get(int(p)) == c for p in g_pids])
        if idx.sum() < 2:
            bases[c] = None; k_effs[c] = 0; continue
        basis, k_eff = pca_basis_gpu(G[idx], k)
        bases[c] = basis; k_effs[c] = int(k_eff)
    out = {}
    for c1, c2 in [('EASY', 'OOD-ONLY'), ('EASY', 'SHARED'), ('OOD-ONLY', 'SHARED')]:
        if bases[c1] is None or bases[c2] is None:
            out[f'{c1}|{c2}'] = None; continue
        common_k = min(k_effs[c1], k_effs[c2])
        A = bases[c1][:, :common_k]; B = bases[c2][:, :common_k]
        out[f'{c1}|{c2}'] = principal_angle_median(A, B)
    return out, k_effs


def build_item_matched_partition(g_pids, covariate_by_pid, target_items):
    unique_pids = sorted(covariate_by_pid.keys())
    sorted_pids = sorted(unique_pids, key=lambda p: covariate_by_pid[p])
    pid_items = {int(p): int((g_pids == p).sum()) for p in unique_pids}
    class_map = {}; cursor = 0; actual_items = {}
    for cls in CLASSES:
        target = target_items[cls]
        acc = 0
        while cursor < len(sorted_pids) and acc < target:
            pid = sorted_pids[cursor]
            acc += pid_items[int(pid)]
            class_map[int(pid)] = cls
            cursor += 1
        actual_items[cls] = acc
    return class_map, actual_items


def build_random_partition(g_pids, target_items, seed):
    rng = np.random.default_rng(seed)
    unique_pids = np.unique(g_pids).tolist()
    rng.shuffle(unique_pids)
    pid_items = {int(p): int((g_pids == p).sum()) for p in unique_pids}
    class_map = {}; cursor = 0; actual = {}
    for cls, target in target_items.items():
        acc = 0
        while cursor < len(unique_pids) and acc < target:
            pid = unique_pids[cursor]
            acc += pid_items[int(pid)]
            class_map[int(pid)] = cls
            cursor += 1
        actual[cls] = acc
    return class_map


def per_id_features(G, g_pids):
    """Compute within_id_spread_p75 and centroid_norm per identity."""
    unique_pids = np.unique(g_pids)
    spread_p75 = {}
    centroid_norm = {}
    for pid in unique_pids:
        X = G[g_pids == pid]
        centroid = X.mean(axis=0)
        centroid_norm[int(pid)] = float(np.linalg.norm(centroid))
        centroid_u = centroid / max(np.linalg.norm(centroid), 1e-12)
        cos_x = np.clip(X @ centroid_u, -1.0, 1.0)
        angles = np.arccos(cos_x)
        spread_p75[int(pid)] = float(np.percentile(angles, 75))
    return spread_p75, centroid_norm


def main():
    print("Loading MSMT D + B'...", flush=True)
    G_id, g_pids, g_cids = load_cell('D')
    G_ood, _, _ = load_cell("B'")

    print("Computing per-identity spread_p75 and centroid_norm from D...", flush=True)
    spread_p75, centroid_norm = per_id_features(G_id, g_pids)
    print(f"  spread_p75: min={min(spread_p75.values()):.3f} "
          f"max={max(spread_p75.values()):.3f}", flush=True)
    print(f"  centroid_norm: min={min(centroid_norm.values()):.3f} "
          f"max={max(centroid_norm.values()):.3f}", flush=True)

    comparators = {
        'spread_p75_asc': spread_p75,   # low spread = easy; if correlation
                                         # with difficulty, low->OOD or ...
        'centroid_norm_desc': {p: -v for p, v in centroid_norm.items()},
                                         # high centroid norm = easy (positive
                                         # rho with AP); flip sign so ASC order
                                         # puts easy first, matching difficulty
                                         # partition order
    }

    out = {}
    for name, covariate in comparators.items():
        print(f"\n########## {name} ##########", flush=True)
        class_map, actual = build_item_matched_partition(
            g_pids, covariate, TARGET_ITEMS)
        id_counts = {c: sum(1 for v in class_map.values() if v == c) for c in CLASSES}
        print(f"  id_counts={id_counts}  item_counts={actual}", flush=True)

        # real statistic
        real_stats = {}
        for arm, G in [('id', G_id), ('ood', G_ood)]:
            stats, k_effs = compute_class_pair_medians(G, g_pids, class_map)
            real_stats[arm] = {'medians': stats, 'k_effs': k_effs}
            print(f"  real [{arm}] medians={stats}", flush=True)

        # size-matched null at ACHIEVED item counts
        print(f"  null: {N_PERM} draws at achieved items {actual}...", flush=True)
        perm_stats = {arm: {p: [] for p in real_stats[arm]['medians']} for arm in ('id', 'ood')}
        t0 = time.time()
        for seed in range(N_PERM):
            perm_class_map = build_random_partition(g_pids, actual, seed=seed)
            for arm, G in [('id', G_id), ('ood', G_ood)]:
                stats, _ = compute_class_pair_medians(G, g_pids, perm_class_map)
                for pair, v in stats.items():
                    if v is not None:
                        perm_stats[arm][pair].append(v)
            if (seed + 1) % 20 == 0:
                print(f"    {seed+1}/{N_PERM} t={(time.time() - t0)/60:.1f}m", flush=True)

        # Compare
        comparison = {'arms': {}, 'actual_items': actual, 'id_counts': id_counts}
        for arm in ('id', 'ood'):
            comparison['arms'][arm] = {}
            for pair, real_val in real_stats[arm]['medians'].items():
                if real_val is None: continue
                nulls = np.array(perm_stats[arm][pair])
                pct = float((nulls < real_val).mean())
                comparison['arms'][arm][pair] = {
                    'real_deg': real_val,
                    'null_mean_deg': float(nulls.mean()),
                    'null_sd_deg': float(nulls.std()),
                    'null_min': float(nulls.min()),
                    'null_max': float(nulls.max()),
                    'gap': real_val - float(nulls.mean()),
                    'pct_below': pct,
                    'nulls': perm_stats[arm][pair],
                }
        out[name] = comparison

    print("\n========== Headline (real - null_mean), MSMT ==========")
    print("  Reference (difficulty): id +4.70 / +6.75 / +0.65")
    for name, res in out.items():
        print(f"\n  [{name}]")
        for arm in ('id', 'ood'):
            line = f"    {arm}: "
            for pair in ('EASY|OOD-ONLY', 'EASY|SHARED', 'OOD-ONLY|SHARED'):
                r = res['arms'][arm].get(pair)
                if r:
                    line += f"{pair.split('|')[0][:4]}|{pair.split('|')[1][:4]}=" + \
                            f"{r['gap']:+.2f} ({r['pct_below']*100:.0f}%)  "
            print(line)

    out_path = RESULTS / 'e40_spread_centroid_null.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved → {out_path}")


if __name__ == '__main__':
    main()
