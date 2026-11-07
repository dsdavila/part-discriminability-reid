"""
E40 covariate permutation null (Channel 2 rev 85).

Question: is the in-domain-difficulty partition SPECIAL in its spatial
distinctness, or would any real covariate (camera ID, images per identity,
etc.) produce similar principal-angle gaps over a random-partition null?

Approach: build alternative partitions from known covariates with the same
class sizes as EASY/SHARED/OOD-ONLY (228/692/2138 identities, 7,160/14,797/
60,162 items). Compute median principal angles between class pairs on the
real features. Compare each covariate's statistic to the SAME item-matched
random-partition null used in E40 v2+common-k (that null depends only on
class sizes, not on how the real classes are defined).

MSMT only (Leg 3 is MSMT-clear, Market-mixed).

Covariates:
  (a) Camera-modal: each identity assigned to its most-frequent camera id;
      cameras grouped into 3 bins by cumulative item count.
  (b) Items-per-identity: each identity binned by its item count (ascending).

Both use common-k = min(k_target, real_k_eff, null_k_eff) per class pair,
matching compute_e40_perm_v2.py.
"""
import json, time
import numpy as np
from pathlib import Path
from collections import Counter

try:
    import torch
    _HAVE_TORCH = True
    _DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
except ImportError:
    _HAVE_TORCH = False
    _DEVICE = None

CACHE_DIR = Path('/tmp/e12_feature_cache')
RESULTS   = Path('/data/papers/low_dimensional_reid/results')
PARTITION_PATH = RESULTS / 'e41_partition_frozen.json'
PERM_NULL_PATH = RESULTS / 'e40_permutation_null_v2.json'
FIXED_K   = 200
CLASSES   = ['EASY', 'SHARED', 'OOD-ONLY']

CELLS = {
    'D':  ('bot_msmt_seed42_msmt.npz',  11659),
    "B'": ('msmt17_test_noREA.npz',     11659),
}


def load_cell(name):
    fn, n_q = CELLS[name]
    d = np.load(CACHE_DIR / fn); emb = d['emb'].astype(np.float32)
    return (emb[n_q:], d['pids'][n_q:], d['cids'][n_q:])


def pca_basis_gpu(X_np, k, device=_DEVICE):
    X = torch.from_numpy(X_np).to(device)
    mean = X.mean(dim=0, keepdim=True)
    Xc = X - mean
    XtX = (Xc.T @ Xc).cpu().numpy()
    eigvals, eigvecs = np.linalg.eigh(XtX)
    eigvals = eigvals[::-1]; eigvecs = eigvecs[:, ::-1]
    n_items, d = X_np.shape
    k_eff = min(k, n_items - 1, d)
    return eigvecs[:, :k_eff].astype(np.float32), k_eff


def principal_angle_median(A, B):
    Aq, _ = np.linalg.qr(A)
    Bq, _ = np.linalg.qr(B)
    M = Aq.T @ Bq
    s = np.linalg.svd(M, compute_uv=False)
    s = np.clip(s, -1.0, 1.0)
    return float(np.degrees(np.median(np.arccos(s))))


def compute_class_pair_medians(G, g_pids, class_map, classes=CLASSES, k=FIXED_K):
    bases, k_effs = {}, {}
    for c in classes:
        idx = np.array([class_map.get(int(p)) == c for p in g_pids])
        if idx.sum() < 2:
            bases[c] = None; k_effs[c] = 0; continue
        basis, k_eff = pca_basis_gpu(G[idx], k)
        bases[c] = basis; k_effs[c] = int(k_eff)
    out = {}; pair_ks = {}
    pairs = [('EASY', 'OOD-ONLY'), ('EASY', 'SHARED'), ('OOD-ONLY', 'SHARED')]
    for c1, c2 in pairs:
        if bases[c1] is None or bases[c2] is None:
            out[f'{c1}|{c2}'] = None; pair_ks[f'{c1}|{c2}'] = 0; continue
        common_k = min(k_effs[c1], k_effs[c2])
        A = bases[c1][:, :common_k]; B = bases[c2][:, :common_k]
        out[f'{c1}|{c2}'] = principal_angle_median(A, B)
        pair_ks[f'{c1}|{c2}'] = int(common_k)
    return out, k_effs, pair_ks


def build_covariate_partition_by_items(g_pids, covariate_vals_by_pid,
                                        target_item_counts, ascending=True):
    """Sort identities by covariate value, greedy assign to classes with target
    item counts {EASY: n1, SHARED: n2, OOD-ONLY: n3}. Accept first overshoot.
    """
    unique_pids = sorted(covariate_vals_by_pid.keys())
    sorted_pids = sorted(unique_pids,
                         key=lambda p: covariate_vals_by_pid[p],
                         reverse=(not ascending))
    pid_items = {int(p): int((g_pids == p).sum()) for p in unique_pids}

    class_map = {}
    cursor = 0
    actual_items = {}
    class_order = ['EASY', 'SHARED', 'OOD-ONLY']
    for cls in class_order:
        target = target_item_counts[cls]
        acc = 0; start = cursor
        while cursor < len(sorted_pids) and acc < target:
            pid = sorted_pids[cursor]
            acc += pid_items[int(pid)]
            class_map[int(pid)] = cls
            cursor += 1
        actual_items[cls] = acc
    return class_map, actual_items


def main():
    # Load the frozen partition for target class sizes
    print("Loading frozen partition and existing null distribution...", flush=True)
    with open(PARTITION_PATH) as f:
        part = json.load(f)
    with open(PERM_NULL_PATH) as f:
        null_v2 = json.load(f)

    # Target item counts for EASY/SHARED/OOD-ONLY (from the frozen real partition)
    # Note: the frozen partition uses 10 decile bins, not 3 behavioral classes.
    # Need to load the 3-class real partition item counts from null_v2.
    real_item_counts = null_v2['MSMT']['real_items']
    print(f"  Target item counts (real EASY/SHARED/OOD-ONLY): {real_item_counts}")

    # Load MSMT features
    G_id, g_pids, g_cids = load_cell('D')
    G_ood, _, _ = load_cell("B'")
    unique_pids = np.unique(g_pids)
    print(f"  MSMT: N={len(g_pids)}, n_ids={len(unique_pids)}", flush=True)

    # Compute covariates per identity
    print("Computing identity-level covariates...", flush=True)
    # (a) Modal camera per identity
    modal_cam_by_pid = {}
    for pid in unique_pids:
        cams = g_cids[g_pids == pid]
        modal_cam_by_pid[int(pid)] = int(Counter(cams).most_common(1)[0][0])
    # (b) Items per identity
    items_by_pid = {int(p): int((g_pids == p).sum()) for p in unique_pids}

    # Build covariate partitions
    print("\nBuilding covariate partitions (item-matched)...", flush=True)
    partitions = {}
    for name, (cov, ascending) in [
        ('camera_modal_asc', (modal_cam_by_pid, True)),
        ('items_per_id_asc', (items_by_pid, True)),
        ('items_per_id_desc', (items_by_pid, False)),
    ]:
        class_map, actual = build_covariate_partition_by_items(
            g_pids, cov, real_item_counts, ascending=ascending)
        id_counts = {c: sum(1 for v in class_map.values() if v == c) for c in CLASSES}
        print(f"  [{name}] id_counts={id_counts}  item_counts={actual}", flush=True)
        partitions[name] = {'class_map': class_map, 'id_counts': id_counts,
                            'item_counts': actual}

    # Per-camera variant: pick 3 individual cameras whose item counts come
    # closest to (EASY, SHARED, OOD-ONLY) targets. Report achieved items +
    # shortfall. Identities with modal camera in those 3 only are assigned.
    print("\nPer-camera variant (3 individual cameras closest to targets)...", flush=True)
    cam_item_counts = Counter(int(c) for c in g_cids)
    sorted_cams = sorted(cam_item_counts.items(), key=lambda kv: kv[1])
    # Camera closest to 7160 target (EASY)
    def closest_cam(target, used):
        best = None; best_diff = float('inf')
        for cam, cnt in cam_item_counts.items():
            if cam in used: continue
            diff = abs(cnt - target)
            if diff < best_diff:
                best_diff = diff; best = cam
        return best
    cam_assignments = {}
    for cls, target in [('EASY', real_item_counts['EASY']),
                        ('SHARED', real_item_counts['SHARED']),
                        ('OOD-ONLY', real_item_counts['OOD-ONLY'])]:
        cam = closest_cam(target, set(cam_assignments.values()))
        cam_assignments[cls] = cam
        print(f"  {cls} target={target} -> camera {cam} ({cam_item_counts[cam]} items, "
              f"{'shortfall' if cam_item_counts[cam] < target else 'overshoot'}"
              f" {cam_item_counts[cam] - target})", flush=True)
    # Assign identities by modal camera
    class_map_pc = {}
    cam_to_cls = {v: k for k, v in cam_assignments.items()}
    for pid in unique_pids:
        modal = modal_cam_by_pid[int(pid)]
        if modal in cam_to_cls:
            class_map_pc[int(pid)] = cam_to_cls[modal]
    id_counts_pc = {c: sum(1 for v in class_map_pc.values() if v == c) for c in CLASSES}
    item_counts_pc = {}
    for c in CLASSES:
        pids_in_c = [p for p, v in class_map_pc.items() if v == c]
        item_counts_pc[c] = int(sum((g_pids == p).sum() for p in pids_in_c))
    print(f"  [camera_per_camera] id_counts={id_counts_pc}  item_counts={item_counts_pc}", flush=True)
    partitions['camera_per_camera'] = {
        'class_map': class_map_pc,
        'id_counts': id_counts_pc,
        'item_counts': item_counts_pc,
        'camera_assignments': cam_assignments,
    }

    # Compute principal angles for each covariate partition, both arms
    print("\nComputing principal angles for each covariate partition...", flush=True)
    results = {}
    for name, p in partitions.items():
        results[name] = {'arms': {}, 'id_counts': p['id_counts'],
                         'item_counts': p['item_counts']}
        for arm, G in [('id', G_id), ('ood', G_ood)]:
            t0 = time.time()
            stats, k_effs, pair_ks = compute_class_pair_medians(G, g_pids, p['class_map'])
            print(f"  [{name}][{arm}] t={(time.time()-t0)/60:.1f}m  "
                  f"medians={stats}  k_effs={k_effs}", flush=True)
            results[name]['arms'][arm] = {
                'class_pair_medians': stats,
                'class_k_effs': k_effs,
                'pair_common_k': pair_ks,
            }

    # Compare each to the random-partition null distribution from v2
    print("\nComparing to random-partition null (from v2)...", flush=True)
    comparison = {}
    for name, res in results.items():
        comparison[name] = {'arms': {}}
        for arm in ('id', 'ood'):
            comparison[name]['arms'][arm] = {}
            for pair_key, real_val in res['arms'][arm]['class_pair_medians'].items():
                if real_val is None: continue
                # null distribution for this (arm, pair) is in null_v2['MSMT']['per_arm'][arm][pair_key]['nulls']
                null_data = null_v2['MSMT']['per_arm'][arm].get(pair_key)
                if null_data is None: continue
                nulls = np.array(null_data['nulls'])
                pct_below = float((nulls < real_val).mean())
                gap = real_val - float(nulls.mean())
                comparison[name]['arms'][arm][pair_key] = {
                    'real_deg': real_val,
                    'null_mean_deg': float(nulls.mean()),
                    'null_min': float(nulls.min()),
                    'null_max': float(nulls.max()),
                    'gap_real_minus_null_mean': gap,
                    'pct_below_real': pct_below,
                }

    out = {
        'dataset': 'MSMT',
        'real_item_counts_target': real_item_counts,
        'covariate_partitions': results,
        'comparison_to_random_null': comparison,
    }

    # Print headline comparison
    print("\n========== Covariate real-minus-null-mean gaps, MSMT (degrees) ==========")
    print("  Reference (difficulty-based from compute_e40_perm_v2 at common-k) at id arm:")
    print("    EASY|OOD-ONLY +4.70   EASY|SHARED +6.75   OOD-ONLY|SHARED +0.65")
    for name in partitions:
        print(f"\n  [{name}]")
        for arm in ('id', 'ood'):
            cmp = comparison[name]['arms'][arm]
            line = f"    {arm}: "
            for pair in ('EASY|OOD-ONLY', 'EASY|SHARED', 'OOD-ONLY|SHARED'):
                if pair in cmp:
                    gap = cmp[pair]['gap_real_minus_null_mean']
                    pct = cmp[pair]['pct_below_real'] * 100
                    line += f"{pair}={gap:+.2f} ({pct:.0f}%)  "
            print(line)

    out_path = RESULTS / 'e40_covariate_null.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved → {out_path}")


if __name__ == '__main__':
    main()
