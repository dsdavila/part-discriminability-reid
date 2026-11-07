"""
Per-camera-matched random null for E40 covariate comparison.

Runs the same item-matched random permutation as compute_e40_perm_v2.py, but
with target item counts matching the per-camera partition (4,681 / 16,420 /
9,889) instead of the real EASY/SHARED/OOD-ONLY sizes. This gives the clean
apples-to-apples comparison for per-camera: is per-camera MORE distinct than
random partitions of the SAME item sizes?

MSMT only. n=100 draws, id + ood arms, common-k per class pair.
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

TARGET_ITEMS = {'EASY': 4681, 'SHARED': 16420, 'OOD-ONLY': 9889}

PER_CAMERA_REAL = {
    'id': {
        'EASY|OOD-ONLY': 18.515,
        'EASY|SHARED': 17.664,
        'OOD-ONLY|SHARED': 12.041,
    },
    'ood': {
        'EASY|OOD-ONLY': 15.252,
        'EASY|SHARED': 14.419,
        'OOD-ONLY|SHARED': 9.970,
    },
}


def load_cell(name):
    cells = {'D':  ('bot_msmt_seed42_msmt.npz', 11659),
             "B'": ('msmt17_test_noREA.npz',    11659)}
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
    Aq, _ = np.linalg.qr(A)
    Bq, _ = np.linalg.qr(B)
    M = Aq.T @ Bq
    s = np.linalg.svd(M, compute_uv=False)
    s = np.clip(s, -1.0, 1.0)
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


def build_item_matched_pseudo_classes(g_pids, class_item_targets, seed):
    rng = np.random.default_rng(seed)
    unique_pids = np.unique(g_pids).tolist()
    rng.shuffle(unique_pids)
    pid_items = {int(p): int((g_pids == p).sum()) for p in unique_pids}
    class_map = {}
    cursor = 0
    class_actual_items = {}
    for cls, target in class_item_targets.items():
        acc = 0
        while cursor < len(unique_pids) and acc < target:
            pid = unique_pids[cursor]
            acc += pid_items[int(pid)]
            class_map[int(pid)] = cls
            cursor += 1
        class_actual_items[cls] = acc
    return class_map, class_actual_items


def main():
    print(f"Loading MSMT, target items {TARGET_ITEMS}...", flush=True)
    G_id, g_pids, _ = load_cell('D')
    G_ood, _, _ = load_cell("B'")
    print(f"  N={len(g_pids)}, n_ids={len(np.unique(g_pids))}", flush=True)

    # Run n=100 permutations at the per-camera target sizes
    perm_stats = {'id': {k: [] for k in PER_CAMERA_REAL['id']},
                  'ood': {k: [] for k in PER_CAMERA_REAL['ood']}}
    t0 = time.time()
    for seed in range(N_PERM):
        perm_classes, _ = build_item_matched_pseudo_classes(
            g_pids, TARGET_ITEMS, seed=seed)
        stats_id, _ = compute_class_pair_medians(G_id, g_pids, perm_classes)
        stats_ood, _ = compute_class_pair_medians(G_ood, g_pids, perm_classes)
        for pair, val in stats_id.items():
            if val is not None: perm_stats['id'][pair].append(val)
        for pair, val in stats_ood.items():
            if val is not None: perm_stats['ood'][pair].append(val)
        if (seed + 1) % 10 == 0:
            print(f"  draw {seed + 1}/{N_PERM}: t={(time.time() - t0) / 60:.1f}m", flush=True)

    # Compare per-camera real to this null
    out = {'target_items': TARGET_ITEMS, 'per_arm': {}}
    for arm in ('id', 'ood'):
        out['per_arm'][arm] = {}
        for pair_key, real_val in PER_CAMERA_REAL[arm].items():
            nulls = np.array(perm_stats[arm][pair_key])
            pct_below = float((nulls < real_val).mean())
            out['per_arm'][arm][pair_key] = {
                'real_deg': real_val,
                'n_draws': len(nulls),
                'null_mean_deg': float(nulls.mean()),
                'null_std_deg': float(nulls.std()),
                'null_min': float(nulls.min()),
                'null_max': float(nulls.max()),
                'gap': real_val - float(nulls.mean()),
                'pct_below_real': pct_below,
                'nulls': perm_stats[arm][pair_key],
            }

    print(f"\n========== Per-camera real vs null matched to per-camera sizes ==========")
    for arm in ('id', 'ood'):
        print(f"\n[{arm}]")
        for pair, r in out['per_arm'][arm].items():
            print(f"  {pair:<20} real={r['real_deg']:.2f}  null mean={r['null_mean_deg']:.2f} "
                  f"sd={r['null_std_deg']:.2f} range=[{r['null_min']:.2f}, {r['null_max']:.2f}]  "
                  f"gap={r['gap']:+.2f}  pct_below={r['pct_below_real']*100:.0f}%")

    out_path = RESULTS / 'e40_percamera_null.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved → {out_path}")


if __name__ == '__main__':
    main()
