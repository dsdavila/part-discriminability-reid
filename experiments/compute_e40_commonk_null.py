"""
e40 partition-pair angles under COMMON-k + structure-matched null (rev 364).

Per claude-chat rev 364: pair k_eff must match across arms to remove the
outcome-dependent parameter (ood k_eff smaller in every pair). For each
pair, use k = min(k_id_arm, k_ood_arm). Also sweep k over admissible range
to test robustness.

Structure-matched null: shuffle identity->class membership keeping n per
class constant. Recompute per-class items PCA at the same k, compute
pair principal angles, 500 draws per arm. Report observed vs null mean.

For each pair:
  - Common-k pair angle per arm (observed)
  - Null: for each of 500 draws, shuffle labels within arm, recompute
  - Report observed-vs-null, excess, p
  - Cross-arm comparison: observed_ood - observed_id, vs null-of-diffs

For the baseline: mean-principal-angle of two random k-dim subspaces in
a d-dim space is arccos(sqrt(k/d)). Also quote the structure-matched
null mean (which absorbs anisotropy).

Sweep k:
  Market SHARED pairs: 3..21 (admissible range for n=22 SHARED)
  Market EASY|OOD-ONLY: just at k=81 (common of 107, 81)
  MSMT SHARED pairs: common k of (k_id_cent, k_ood_cent)
  MSMT EASY|OOD-ONLY: just at k=80 (common of 109, 80)
"""
import json, time
import numpy as np
from pathlib import Path
from collections import Counter

CACHE_DIR = Path('/tmp/e12_feature_cache')
RESULTS   = Path('/data/papers/low_dimensional_reid/results')
VAR_THRESHOLD = 0.9
N_NULL = 500
RNG_SEED = 42

CELLS = {
    'MSMT':   {'id': ('bot_msmt_seed42_msmt.npz',  11659),
               'ood': ('msmt17_test_noREA.npz',     11659)},
    'Market': {'id': ('market_test_noREA.npz',     3368),
               'ood': ('bot_msmt_seed42_market.npz', 3368)},
}

CLASSES = ['EASY', 'ID-ONLY', 'SHARED', 'OOD-ONLY']


def l2(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(n, 1e-12)


def load_cell(fname, n_q):
    d = np.load(CACHE_DIR / fname)
    emb = d['emb'].astype(np.float32); pids = d['pids']; cids = d['cids']
    return l2(emb[n_q:]), pids[n_q:], cids[n_q:]


def top1_wrong_rate_chunk(G, g_pids, g_cids, chunk=512):
    n = len(g_pids)
    wrong = np.zeros(n, dtype=bool)
    for i0 in range(0, n, chunk):
        i1 = min(i0 + chunk, n)
        sims = G[i0:i1] @ G.T
        for j, i in enumerate(range(i0, i1)):
            exc = (g_pids == g_pids[i]) & (g_cids == g_cids[i])
            sims[j, exc] = -2.0
            order = np.argmax(sims[j])
            wrong[i] = g_pids[order] != g_pids[i]
    per_id = {}
    for p in np.unique(g_pids):
        mask = g_pids == p
        per_id[int(p)] = float(wrong[mask].mean())
    return per_id


def classify(rates_id, rates_ood, thr=0.5):
    classes = {}
    for uid in rates_id:
        ri = rates_id[uid]; ro = rates_ood.get(uid, 1.0)
        if ri <= thr and ro <= thr: classes[uid] = 'EASY'
        elif ri > thr and ro > thr: classes[uid] = 'SHARED'
        elif ri > thr: classes[uid] = 'ID-ONLY'
        else: classes[uid] = 'OOD-ONLY'
    return classes


def pca_basis(X, k):
    mu = X.mean(0); Xc = X - mu
    n, d = Xc.shape
    if d <= n:
        C = Xc.T @ Xc / max(n - 1, 1)
        w, v = np.linalg.eigh(C.astype(np.float64))
        order = np.argsort(-w)
        return v[:, order[:k]].astype(np.float32), w[order[:k]].astype(np.float32)
    G = Xc @ Xc.T / max(n - 1, 1)
    w, u = np.linalg.eigh(G.astype(np.float64))
    order = np.argsort(-w)
    w_sel = w[order[:k]]; u_sel = u[:, order[:k]]
    v = (Xc.T @ u_sel) / np.sqrt(np.maximum(w_sel, 1e-12) * (n - 1))
    v = v / np.maximum(np.linalg.norm(v, axis=0, keepdims=True), 1e-12)
    return v.astype(np.float32), w_sel.astype(np.float32)


def k_for_variance(cum_var, threshold=VAR_THRESHOLD):
    idxs = np.where(cum_var >= threshold)[0]
    return int(idxs[0]) + 1 if len(idxs) else len(cum_var)


def principal_angles(A, B):
    s = np.linalg.svd(A.T @ B, compute_uv=False)
    s = np.clip(s, -1.0, 1.0)
    return np.degrees(np.arccos(s))


def per_id_centroids(G, pids):
    uids = np.unique(pids)
    return uids, np.stack([G[pids == p].mean(0) for p in uids])


def classify_arm_rates(G, gp, gc, name):
    print(f"  {name} rank-1-wrong rates...", flush=True)
    t0 = time.time()
    rates = top1_wrong_rate_chunk(G, gp, gc)
    print(f"    t={(time.time()-t0)/60:.1f}m", flush=True)
    return rates


def pair_subspaces(G, gp, classes, c1, c2, k):
    """Return items PCA basis of two classes at rank k each."""
    mask1 = np.array([classes.get(int(p)) == c1 for p in gp])
    mask2 = np.array([classes.get(int(p)) == c2 for p in gp])
    X1 = G[mask1]; X2 = G[mask2]
    if len(X1) == 0 or len(X2) == 0:
        return None, None
    k1 = min(k, X1.shape[0] - 1, X1.shape[1])
    k2 = min(k, X2.shape[0] - 1, X2.shape[1])
    k_use = min(k1, k2)
    if k_use < 1: return None, None
    b1, _ = pca_basis(X1, k_use)
    b2, _ = pca_basis(X2, k_use)
    return b1, b2


def pair_mean_angle(G, gp, classes, c1, c2, k):
    b1, b2 = pair_subspaces(G, gp, classes, c1, c2, k)
    if b1 is None: return None
    angs = principal_angles(b1, b2)
    return float(angs.mean())


def compute_k_cent_for_classes(G, gp, classes):
    """Return dict class -> k_centroids_90pct for each class."""
    out = {}
    for c in CLASSES:
        mask = np.array([classes.get(int(p)) == c for p in gp])
        if mask.sum() == 0:
            out[c] = 0; continue
        X_c = G[mask]
        p_c = gp[mask]
        _, mu_c = per_id_centroids(X_c, p_c)
        if len(mu_c) < 2:
            out[c] = 0; continue
        _, eigvals = pca_basis(mu_c, k=min(mu_c.shape[1], mu_c.shape[0]))
        cum = np.cumsum(eigvals) / eigvals.sum() if eigvals.sum() > 0 else np.array([0.0])
        out[c] = k_for_variance(cum)
    return out


def run_pair_null(G, gp, classes, c1, c2, k, n_null, rng_seed):
    """Structure-matched null: shuffle id->class labels at matched sizes,
    recompute pair angle at the same k. Returns null array."""
    uids = sorted(classes.keys())
    class_labels = np.array([classes[u] for u in uids])
    # Build id -> items map
    id_to_items = {u: np.where(gp == u)[0] for u in uids}
    rng = np.random.default_rng(rng_seed)
    null_angles = []
    for di in range(n_null):
        perm = rng.permutation(len(class_labels))
        shuffled = class_labels[perm]
        fake_classes = {uids[i]: shuffled[i] for i in range(len(uids))}
        a = pair_mean_angle(G, gp, fake_classes, c1, c2, k)
        if a is not None:
            null_angles.append(a)
    return np.array(null_angles)


def process_dataset(name, cell_cfg):
    print(f"\n============ {name} ============", flush=True)
    G_id, gp_id, gc_id = load_cell(*cell_cfg['id'])
    G_ood, gp_ood, gc_ood = load_cell(*cell_cfg['ood'])
    # classify
    rates_id = classify_arm_rates(G_id, gp_id, gc_id, 'id')
    rates_ood = classify_arm_rates(G_ood, gp_ood, gc_ood, 'ood')
    classes = classify(rates_id, rates_ood)
    print(f"  partition counts: {Counter(classes.values())}", flush=True)
    # compute per-class k_centroids on both arms
    k_cent_id = compute_k_cent_for_classes(G_id, gp_id, classes)
    k_cent_ood = compute_k_cent_for_classes(G_ood, gp_ood, classes)
    print(f"  k_centroids id : {k_cent_id}", flush=True)
    print(f"  k_centroids ood: {k_cent_ood}", flush=True)

    pair_keys = [('EASY', 'OOD-ONLY'), ('EASY', 'SHARED'), ('OOD-ONLY', 'SHARED')]
    results = {}
    for c1, c2 in pair_keys:
        k_id = k_cent_id.get(c1, 0)
        k_ood = k_cent_ood.get(c1, 0)
        k_id2 = k_cent_id.get(c2, 0)
        k_ood2 = k_cent_ood.get(c2, 0)
        # Common-k across arms AND across the two classes (most conservative)
        k_common = min(k_id, k_ood, k_id2, k_ood2)
        if k_common < 2:
            print(f"  {c1}|{c2}: k_common={k_common} too small, skip", flush=True)
            continue
        print(f"\n  --- {c1}|{c2} at k_common = {k_common} ---", flush=True)
        # Observed
        obs_id = pair_mean_angle(G_id, gp_id, classes, c1, c2, k_common)
        obs_ood = pair_mean_angle(G_ood, gp_ood, classes, c1, c2, k_common)
        print(f"    observed: id={obs_id:.3f}  ood={obs_ood:.3f}  diff(ood-id)={obs_ood - obs_id:+.3f}", flush=True)
        # Null per arm
        print(f"    null id ({N_NULL} draws)...", flush=True)
        null_id = run_pair_null(G_id, gp_id, classes, c1, c2, k_common, N_NULL, RNG_SEED)
        print(f"    null ood ({N_NULL} draws)...", flush=True)
        null_ood = run_pair_null(G_ood, gp_ood, classes, c1, c2, k_common, N_NULL, RNG_SEED + 1)
        # Cross-arm comparison via pair draws
        rng_pair = np.random.default_rng(2026)
        n_p = 50000
        idx_i = rng_pair.integers(0, len(null_id), n_p)
        idx_j = rng_pair.integers(0, len(null_ood), n_p)
        null_diff = null_ood[idx_j] - null_id[idx_i]
        obs_diff = obs_ood - obs_id

        # Random-subspace baseline for 2048D at k=k_common
        d = 2048
        uniform_baseline = float(np.degrees(np.arccos(np.sqrt(k_common / d))))

        results[f'{c1}|{c2}'] = {
            'k_common': int(k_common),
            'obs_id': obs_id, 'obs_ood': obs_ood, 'obs_diff_ood_id': obs_diff,
            'null_id': {
                'mean': float(null_id.mean()), 'sd': float(null_id.std(ddof=1)),
                'range': [float(null_id.min()), float(null_id.max())],
                'excess_obs': float(obs_id - null_id.mean()),
                'z': float((obs_id - null_id.mean()) / null_id.std(ddof=1)),
                'p_below': float((null_id < obs_id).mean()),
            },
            'null_ood': {
                'mean': float(null_ood.mean()), 'sd': float(null_ood.std(ddof=1)),
                'range': [float(null_ood.min()), float(null_ood.max())],
                'excess_obs': float(obs_ood - null_ood.mean()),
                'z': float((obs_ood - null_ood.mean()) / null_ood.std(ddof=1)),
                'p_below': float((null_ood < obs_ood).mean()),
            },
            'null_diff': {
                'mean': float(null_diff.mean()), 'sd': float(null_diff.std(ddof=1)),
                'range': [float(null_diff.min()), float(null_diff.max())],
                'excess_obs': float(obs_diff - null_diff.mean()),
                'z': float((obs_diff - null_diff.mean()) / null_diff.std(ddof=1)),
                'p_below': float((null_diff < obs_diff).mean()),
                'p_above': float((null_diff > obs_diff).mean()),
            },
            'uniform_random_baseline_deg': uniform_baseline,
        }
        print(f"    structure-matched null id : mean={null_id.mean():.3f}  sd={null_id.std(ddof=1):.3f}  obs excess={obs_id - null_id.mean():+.3f}  z={(obs_id - null_id.mean())/null_id.std(ddof=1):+.2f}", flush=True)
        print(f"    structure-matched null ood: mean={null_ood.mean():.3f}  sd={null_ood.std(ddof=1):.3f}  obs excess={obs_ood - null_ood.mean():+.3f}  z={(obs_ood - null_ood.mean())/null_ood.std(ddof=1):+.2f}", flush=True)
        print(f"    diff(ood-id) null: mean={null_diff.mean():+.3f}  sd={null_diff.std(ddof=1):.3f}  obs_diff={obs_diff:+.3f}  excess={obs_diff - null_diff.mean():+.3f}  z={(obs_diff - null_diff.mean())/null_diff.std(ddof=1):+.2f}  p_below={(null_diff < obs_diff).mean():.4f}  p_above={(null_diff > obs_diff).mean():.4f}", flush=True)
        print(f"    uniform-random baseline for 2048D at k={k_common}: {uniform_baseline:.2f} deg", flush=True)
    return results


def main():
    out = {}
    for name, cfg in CELLS.items():
        out[name] = process_dataset(name, cfg)
    out_path = RESULTS / 'e40_commonk_null.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {out_path}")

    print("\n\n========== e40 common-k pair-angle structure-matched null ==========")
    for dataset in out:
        print(f"\n  {dataset}:")
        for pair, r in out[dataset].items():
            print(f"    {pair:<22} k={r['k_common']}  id obs={r['obs_id']:.2f}  null={r['null_id']['mean']:.2f}  excess={r['null_id']['excess_obs']:+.2f}  z_id={r['null_id']['z']:+.2f}")
            print(f"    {'':22} ood obs={r['obs_ood']:.2f}  null={r['null_ood']['mean']:.2f}  excess={r['null_ood']['excess_obs']:+.2f}  z_ood={r['null_ood']['z']:+.2f}")
            print(f"    {'':22} diff(ood-id)={r['obs_diff_ood_id']:+.2f}  null_diff={r['null_diff']['mean']:+.2f}  excess_diff={r['null_diff']['excess_obs']:+.2f}  z_diff={r['null_diff']['z']:+.2f}  p_below={r['null_diff']['p_below']:.4f}")


if __name__ == '__main__':
    main()
