"""
E40 permutation null v2 (Channel 2 rev 56/62/64 spec).

Fixes two defects in the original compute_e40_perm.py:
  1. CPU XtX on 60k x 2048 is 1-4 min per draw per arm; moves to torch GPU
     matmul (~40ms) so n=100 is feasible.
  2. Original builds pseudo-classes matched on IDENTITY count, but per-class
     PCA runs on ITEMS. EASY/OOD-ONLY/SHARED carry 31.4/28.1/21.4 items per
     identity on MSMT, so a size-228 identity draw yields ~6,100 items against
     EASY's 7,160 — the real class gets ~17% more data than its null, biasing
     subspace estimates in the direction of the claim. Fix: draw identities
     until cumulative item count reaches the real class's item count (accept
     the first overshoot). Reports items-per-class real vs null side by side.

Also:
  - id (D) and ood (B' or C) arms only. noise and randproj permutation nulls
    were degenerate and not load-bearing for the claim.
  - n=100 draws, seed=range(100). Original 5 are the first 5 (preserved).
  - Reports the REAL statistic's PERCENTILE in the null distribution, not its
    difference from the null mean.
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
CLASSES   = ['EASY', 'SHARED', 'OOD-ONLY']  # drop ID-ONLY (n=1)
FIXED_K   = 200
N_PERMUTATIONS = 100

CELLS = {
    'D':  ('bot_msmt_seed42_msmt.npz',  11659),
    "B'": ('msmt17_test_noREA.npz',     11659),
    "A'": ('market_test_noREA.npz',      3368),
    'C':  ('bot_msmt_seed42_market.npz', 3368),
}


def load_cell(name):
    fn, n_q = CELLS[name]
    d = np.load(CACHE_DIR / fn); emb = d['emb'].astype(np.float32)
    return (emb[n_q:], d['pids'][n_q:], d['cids'][n_q:])


def top1_wrong_rate(G, g_pids, g_cids, chunk=512):
    N = len(G); t1 = np.full(N, -1, dtype=np.int64)
    for start in range(0, N, chunk):
        end = min(N, start + chunk)
        sims = G[start:end] @ G.T
        for local, gi in enumerate(range(start, end)):
            exc = (g_pids == g_pids[gi]) & (g_cids == g_cids[gi])
            sims[local, exc] = -2.0
            t1[gi] = int(np.argmax(sims[local]))
    pred = g_pids[t1]; wrong = pred != g_pids
    return {int(uid): float(wrong[g_pids == uid].mean()) for uid in np.unique(g_pids)}


def classify(rates_id, rates_ood, thr=0.5):
    out = {}
    for uid in rates_id:
        ri = rates_id.get(uid, 0.0); ro = rates_ood.get(uid, 0.0)
        if ri <= thr and ro <= thr: out[uid] = 'EASY'
        elif ri > thr and ro > thr: out[uid] = 'SHARED'
        elif ri > thr: out[uid] = 'ID-ONLY'
        else: out[uid] = 'OOD-ONLY'
    return out


def pca_basis_fixed_k_gpu(X_np, k, device=_DEVICE):
    """Top-k eigenvectors of X_centered.T @ X_centered via torch (GPU if avail).
    Caps k at n_items - 1 to avoid null-space directions in rank-starved classes.
    """
    X = torch.from_numpy(X_np).to(device)
    mean = X.mean(dim=0, keepdim=True)
    Xc = X - mean
    XtX = Xc.T @ Xc  # (D, D)
    XtX_cpu = XtX.cpu().numpy()  # eigh is faster on CPU for 2048x2048
    eigvals, eigvecs = np.linalg.eigh(XtX_cpu)
    eigvals = eigvals[::-1]; eigvecs = eigvecs[:, ::-1]
    n_items, d = X_np.shape
    k_eff = min(k, n_items - 1, d)
    return eigvecs[:, :k_eff].astype(np.float32), k_eff


def pca_basis_fixed_k_cpu(X, k):
    if X.shape[0] < 2: return None, 0
    X = X.astype(np.float32)
    mean = X.mean(axis=0, keepdims=True)
    Xc = X - mean
    XtX = Xc.T @ Xc
    eigvals, eigvecs = np.linalg.eigh(XtX)
    eigvals = eigvals[::-1]; eigvecs = eigvecs[:, ::-1]
    n_items, d = X.shape
    k_eff = min(k, n_items - 1, d)
    return eigvecs[:, :k_eff].astype(np.float32), k_eff


def principal_angles(A, B):
    Aq, _ = np.linalg.qr(A)
    Bq, _ = np.linalg.qr(B)
    M = Aq.T @ Bq
    s = np.linalg.svd(M, compute_uv=False)
    s = np.clip(s, -1.0, 1.0)
    return np.arccos(s)


def principal_angle_median(A, B):
    """Just the median principal angle (deg), what the null is tested against."""
    ang = principal_angles(A, B)
    return float(np.degrees(np.median(ang)))


def per_class_items(G, g_pids, class_map, cls):
    idx = np.array([class_map.get(int(p)) == cls for p in g_pids])
    return G[idx], idx.sum()


def build_item_matched_pseudo_classes(g_pids, class_item_targets, seed):
    """Draw identities one at a time (without replacement from all ids),
    accumulate until items reach each class's target, accept first overshoot.

    Draws classes in fixed order of class_item_targets dict to make output
    reproducible for seed.
    """
    rng = np.random.default_rng(seed)
    unique_pids = np.unique(g_pids).tolist()
    rng.shuffle(unique_pids)
    pid_items = {int(p): int((g_pids == p).sum()) for p in unique_pids}

    class_map = {}
    cursor = 0
    class_actual_items = {}
    for cls, target in class_item_targets.items():
        acc = 0
        start = cursor
        while cursor < len(unique_pids) and acc < target:
            pid = unique_pids[cursor]
            acc += pid_items[int(pid)]
            class_map[int(pid)] = cls
            cursor += 1
        class_actual_items[cls] = acc
        if cursor >= len(unique_pids) and acc < target:
            # ran out of identities
            break
    return class_map, class_actual_items


def compute_class_pair_medians(G, g_pids, class_map, classes=CLASSES, k=FIXED_K, use_gpu=True):
    """For each class in classes, build PCA basis capped at min(k, n_items-1, d).
    Returns median principal angle per pair and per-class k_eff.
    """
    bases = {}
    k_effs = {}
    for c in classes:
        idx = np.array([class_map.get(int(p)) == c for p in g_pids])
        if idx.sum() < 2:
            bases[c] = None; k_effs[c] = 0; continue
        X = G[idx]
        if use_gpu and _HAVE_TORCH:
            basis, k_eff = pca_basis_fixed_k_gpu(X, k)
        else:
            basis, k_eff = pca_basis_fixed_k_cpu(X, k)
        bases[c] = basis
        k_effs[c] = int(k_eff)
    out = {}
    pair_keys = [('EASY', 'OOD-ONLY'), ('EASY', 'SHARED'), ('OOD-ONLY', 'SHARED')]
    for c1, c2 in pair_keys:
        if bases[c1] is None or bases[c2] is None:
            out[f'{c1}|{c2}'] = None; continue
        out[f'{c1}|{c2}'] = principal_angle_median(bases[c1], bases[c2])
    return out, k_effs


def run_dataset(name, id_cell, ood_cell, n_perm=N_PERMUTATIONS, k=FIXED_K):
    print(f"\n########## {name}: id={id_cell}/ood={ood_cell}, "
          f"n_perm={n_perm}, k={k}, device={_DEVICE} ##########", flush=True)
    G_id, g_pids, g_cids = load_cell(id_cell)
    G_ood, _, _ = load_cell(ood_cell)
    print(f"  N={len(G_id)} n_ids={len(np.unique(g_pids))}", flush=True)

    t0 = time.time()
    rates_i = top1_wrong_rate(G_id, g_pids, g_cids)
    rates_o = top1_wrong_rate(G_ood, g_pids, g_cids)
    print(f"  rank-1 passes: t={(time.time() - t0) / 60:.1f}m", flush=True)
    real_classes = classify(rates_i, rates_o)
    real_counts = {c: sum(1 for v in real_classes.values() if v == c) for c in CLASSES}
    # item counts per class for matching
    class_item_counts = {c: int(sum((g_pids == pid).sum() for pid, v in real_classes.items() if v == c))
                         for c in CLASSES}
    print(f"  partition (identities): {real_counts}", flush=True)
    print(f"  partition (items): {class_item_counts}", flush=True)

    # Real statistic per arm
    arms = {'id': G_id, 'ood': G_ood}
    real_stats = {}
    real_k_effs = {}
    for arm_name, G_ in arms.items():
        t0 = time.time()
        real_stats[arm_name], real_k_effs[arm_name] = compute_class_pair_medians(G_, g_pids, real_classes, use_gpu=True)
        print(f"  real [{arm_name}] class-pair medians: t={(time.time() - t0) / 60:.1f}m "
              f"=> {real_stats[arm_name]}", flush=True)
        print(f"    k_eff per class: {real_k_effs[arm_name]}", flush=True)

    # Permutation null
    perm_stats = {arm: {k: [] for k in real_stats['id']} for arm in arms}
    perm_k_effs = {arm: [] for arm in arms}
    perm_items = []
    t_total = time.time()
    for seed in range(n_perm):
        t0 = time.time()
        perm_classes, class_actual_items = build_item_matched_pseudo_classes(
            g_pids, class_item_counts, seed=seed)
        perm_items.append(class_actual_items)
        for arm_name, G_ in arms.items():
            stats, k_effs = compute_class_pair_medians(G_, g_pids, perm_classes, use_gpu=True)
            perm_k_effs[arm_name].append(k_effs)
            for pair_key, val in stats.items():
                if val is not None:
                    perm_stats[arm_name][pair_key].append(val)
        if (seed + 1) % 10 == 0 or seed < 5:
            el = time.time() - t_total
            print(f"  draw {seed + 1}/{n_perm}: avg {el / (seed + 1):.1f}s/draw, "
                  f"total {el / 60:.1f}m", flush=True)

    # Summarize: real stat's percentile in null distribution
    summary = {'real': real_stats, 'perm_items_mean':
               {c: float(np.mean([p.get(c, 0) for p in perm_items])) for c in CLASSES},
               'real_items': class_item_counts,
               'real_k_effs': real_k_effs,
               'perm_k_eff_mean': {arm: {c: float(np.mean([pk.get(c, 0) for pk in perm_k_effs[arm]]))
                                         for c in CLASSES} for arm in arms},
               'partition_counts': real_counts, 'per_arm': {}}
    for arm in arms:
        summary['per_arm'][arm] = {}
        for pair_key, nulls in perm_stats[arm].items():
            real = real_stats[arm].get(pair_key)
            if real is None or not nulls:
                summary['per_arm'][arm][pair_key] = None; continue
            nulls_arr = np.array(nulls)
            pct_below = float((nulls_arr < real).mean())
            pct_above = float((nulls_arr > real).mean())
            summary['per_arm'][arm][pair_key] = {
                'real_deg': real,
                'n_draws': len(nulls),
                'null_mean_deg': float(nulls_arr.mean()),
                'null_median_deg': float(np.median(nulls_arr)),
                'null_std_deg': float(nulls_arr.std()),
                'null_min': float(nulls_arr.min()),
                'null_max': float(nulls_arr.max()),
                'pct_below_real': pct_below,
                'pct_above_real': pct_above,
                'nulls': nulls,
            }
    return summary


def print_report(name, res):
    print(f"\n========== {name} — real stat percentile in null distribution ==========")
    print(f"  items per class: real {res['real_items']} vs null-mean {res['perm_items_mean']}")
    for arm in ('id', 'ood'):
        print(f"\n  [{arm}]")
        for pair_key, r in res['per_arm'][arm].items():
            if r is None: continue
            pct = r['pct_below_real'] * 100
            print(f"    {pair_key:<20} real={r['real_deg']:.3f}° "
                  f"| null mean={r['null_mean_deg']:.3f}° median={r['null_median_deg']:.3f}° "
                  f"range=[{r['null_min']:.3f}, {r['null_max']:.3f}]  n={r['n_draws']}  "
                  f"pct_below={pct:.1f}%")


def main():
    import sys
    market_only = '--market-only' in sys.argv
    out = {}
    if not market_only:
        out['MSMT'] = run_dataset('MSMT', 'D', "B'")
        print_report('MSMT', out['MSMT'])
    out['Market'] = run_dataset('Market', "A'", 'C')
    print_report('Market', out['Market'])
    out_path = RESULTS / ('e40_permutation_null_v2_market_kfix.json' if market_only
                          else 'e40_permutation_null_v2.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved → {out_path}")


if __name__ == '__main__':
    main()
