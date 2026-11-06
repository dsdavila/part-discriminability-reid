"""
E40 (claude-chat, 2026-11-06T19:48Z, Plan rev 53) — Subspace measurements.

Daniel: everything held is angular; which DIRECTIONS each class occupies is
unmeasured. E40 is the subspace layer.

Parts:
  (a) per-class principal subspace by PCA over identity centroids AND over items.
      k = smallest reaching 90% variance, k reported.
  (b) FULL spectrum of principal angles between class subspaces (EASY↔OOD-ONLY,
      EASY↔SHARED, OOD-ONLY↔SHARED), not just first or mean. Sensitivity at
      0.5× / 1× / 2× k of each class.
  (c) cos between global gallery mean (unit vector) and its projection onto
      each class's subspace (= ||V_c V_c^T g||, i.e. sqrt(sum of (v_i·g)^2
      over class subspace basis vectors v_i).
  (d) All four arms (D, B', noise, randproj), both datasets (MSMT, Market),
      identity counts included.

BINDING (Channel 2 rev 18): principal angles BETWEEN CLASSES WITHIN ONE ARM
are legitimate and comparable across arms. Do NOT compute principal angles
between a D subspace and a B' subspace. Do not describe any subspace as
"rotated" or "moved".
"""
import json, time
import numpy as np
from pathlib import Path
from collections import Counter

CACHE_DIR = Path('/tmp/e12_feature_cache')
RESULTS   = Path('/data/papers/low_dimensional_reid/results')
CLASSES = ['EASY', 'ID-ONLY', 'SHARED', 'OOD-ONLY']
# Subspace PCA threshold
VAR_THRESHOLD = 0.90
SENSITIVITY_FACTORS = [0.5, 1.0, 2.0]

CELLS = {
    'D':  ('bot_msmt_seed42_msmt.npz',  11659),
    "B'": ('msmt17_test_noREA.npz',     11659),
    "A'": ('market_test_noREA.npz',      3368),
    'C':  ('bot_msmt_seed42_market.npz', 3368),
}
SIGMA_D = 0.0625
SIGMA_A = 0.062
K_RANDPROJ_D = 17
K_RANDPROJ_A = 17


def load_cell(name):
    fn, n_q = CELLS[name]
    d = np.load(CACHE_DIR / fn); emb = d['emb'].astype(np.float32)
    return (emb[:n_q], d['pids'][:n_q], d['cids'][:n_q],
            emb[n_q:], d['pids'][n_q:], d['cids'][n_q:])


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
    classes = {}
    for uid in rates_id:
        ri = rates_id.get(uid, 0.0); ro = rates_ood.get(uid, 0.0)
        if ri <= thr and ro <= thr: classes[uid] = 'EASY'
        elif ri > thr and ro > thr: classes[uid] = 'SHARED'
        elif ri > thr: classes[uid] = 'ID-ONLY'
        else: classes[uid] = 'OOD-ONLY'
    return classes


def per_id_centroids(G, pids):
    ids = np.unique(pids)
    mu = np.zeros((len(ids), G.shape[1]), np.float32)
    for k, pid in enumerate(ids):
        mask = pids == pid
        xi = G[mask]
        mu_k = xi.mean(axis=0)
        mu_k /= max(np.linalg.norm(mu_k), 1e-12)
        mu[k] = mu_k
    return ids, mu


def pca_basis(X, k):
    """Return top-k right singular vectors (columns), i.e. basis of principal
    subspace. X is (n, d). Centering NOT applied (we want principal directions
    not centered data) — but since features are L2-normed, mean is near zero.
    Actually SVD of X directly returns subspace. Return basis as (d, k)."""
    X = X.astype(np.float32)
    # Centering: subtract per-feature mean so PCA is over covariance
    mean = X.mean(axis=0, keepdims=True)
    Xc = X - mean
    # SVD: Xc = U S V^T, columns of V are principal directions
    # Use np.linalg.svd(Xc, full_matrices=False) — but if n >= d (~2048)
    # we want only top-k. np.linalg.svd on an 82128x2048 matrix is slow but
    # tractable (~a few sec). For speed, use eigendecomposition of X^T X.
    XtX = Xc.T @ Xc  # (d, d)
    # eigendecomp
    eigvals, eigvecs = np.linalg.eigh(XtX)
    # ascending — flip
    eigvals = eigvals[::-1]; eigvecs = eigvecs[:, ::-1]
    # ensure non-negative
    eigvals = np.maximum(eigvals, 0.0)
    total = eigvals.sum()
    if total <= 0:
        return eigvecs[:, :k], np.zeros(k)
    cum_var = np.cumsum(eigvals) / total
    k = min(k, eigvecs.shape[1])
    basis = eigvecs[:, :k].astype(np.float32)
    return basis, eigvals[:k], cum_var


def k_for_variance(cum_var, threshold=VAR_THRESHOLD):
    """Smallest k such that cum_var[k-1] >= threshold."""
    idx = np.searchsorted(cum_var, threshold)
    return int(idx) + 1


def principal_angles(A, B):
    """Return principal angles (rad) between subspaces with bases A (d, k1) and
    B (d, k2). Compute SVD of A^T B; singular values are cosines of angles."""
    # Ensure orthonormal columns
    Aq, _ = np.linalg.qr(A)
    Bq, _ = np.linalg.qr(B)
    M = Aq.T @ Bq
    s = np.linalg.svd(M, compute_uv=False)
    s = np.clip(s, -1.0, 1.0)
    # singular values in descending order; angles ascending
    angles = np.arccos(s)
    return angles


def global_mean_vs_subspace(G, basis):
    """Return cos between global gallery mean direction and its projection
    onto subspace spanned by columns of `basis`. For unit g, cos equals
    norm of V^T g."""
    gm = G.mean(axis=0)
    gm = gm / max(np.linalg.norm(gm), 1e-12)
    proj_coords = basis.T @ gm  # (k,)
    norm = float(np.linalg.norm(proj_coords))
    return norm, gm


def run_arm(G, gp, gc, classes, arm_name):
    print(f"\n  --- arm={arm_name} ---", flush=True)
    results = {'arm': arm_name}
    # (a) per-class principal subspace
    per_cls = {}
    for c in CLASSES:
        idxs_class = np.array([classes.get(int(p)) == c for p in gp])
        n_items = int(idxs_class.sum())
        if n_items == 0:
            per_cls[c] = None
            continue
        X_items = G[idxs_class]
        ids_class = np.unique(gp[idxs_class])
        n_ids = len(ids_class)
        # centroids
        _, mu_c = per_id_centroids(X_items, gp[idxs_class])
        # PCA over items (choose k from cum var threshold)
        _, eigvals_items, cum_items = pca_basis(X_items, k=min(X_items.shape[1], X_items.shape[0]))
        k_items = k_for_variance(cum_items)
        basis_items, _, _ = pca_basis(X_items, k=k_items)
        # PCA over centroids
        if n_ids >= 2:
            _, eigvals_ids, cum_ids = pca_basis(mu_c, k=min(mu_c.shape[1], mu_c.shape[0]))
            k_centroids = k_for_variance(cum_ids)
            basis_centroids, _, _ = pca_basis(mu_c, k=k_centroids)
        else:
            k_centroids = 0
            basis_centroids = None
        per_cls[c] = {
            'n_ids': int(n_ids), 'n_items': int(n_items),
            'k_items_90pct': k_items,
            'k_centroids_90pct': k_centroids,
            'basis_items': basis_items,
            'basis_centroids': basis_centroids,
        }
    # (c) cos(global mean, projection onto class subspace) — on items basis
    for c in CLASSES:
        if per_cls[c] is None: continue
        norm_items, _ = global_mean_vs_subspace(G, per_cls[c]['basis_items'])
        per_cls[c]['cos_global_in_items_subspace'] = norm_items
        if per_cls[c]['basis_centroids'] is not None:
            norm_cent, _ = global_mean_vs_subspace(G, per_cls[c]['basis_centroids'])
            per_cls[c]['cos_global_in_centroids_subspace'] = norm_cent

    # (b) principal angles between class pairs — items subspace, three k-levels
    pair_keys = [('EASY', 'OOD-ONLY'), ('EASY', 'SHARED'), ('OOD-ONLY', 'SHARED')]
    pair_results = {}
    for c1, c2 in pair_keys:
        if per_cls[c1] is None or per_cls[c2] is None: continue
        k1 = per_cls[c1]['k_items_90pct']
        k2 = per_cls[c2]['k_items_90pct']
        per_level = {}
        for factor in SENSITIVITY_FACTORS:
            k1f = max(1, int(k1 * factor))
            k2f = max(1, int(k2 * factor))
            # rebuild bases at scaled k (reuse items PCA)
            X1 = G[np.array([classes.get(int(p)) == c1 for p in gp])]
            X2 = G[np.array([classes.get(int(p)) == c2 for p in gp])]
            b1, _, _ = pca_basis(X1, k=min(k1f, X1.shape[1]))
            b2, _, _ = pca_basis(X2, k=min(k2f, X2.shape[1]))
            angles = principal_angles(b1, b2)
            per_level[f"{factor}x"] = {
                'k1': int(b1.shape[1]), 'k2': int(b2.shape[1]),
                'angles_deg': [float(np.degrees(a)) for a in angles],
                'angles_mean_deg': float(np.degrees(angles).mean()),
                'angles_median_deg': float(np.median(np.degrees(angles))),
                'first_angle_deg': float(np.degrees(angles[0])),
                'last_angle_deg': float(np.degrees(angles[-1])),
            }
        pair_results[f"{c1}|{c2}"] = per_level

    # Strip bases from output JSON (large)
    out_per_cls = {}
    for c in CLASSES:
        if per_cls[c] is None:
            out_per_cls[c] = None; continue
        out_per_cls[c] = {k: v for k, v in per_cls[c].items()
                          if not k.startswith('basis_')}
    results['per_class'] = out_per_cls
    results['principal_angles'] = pair_results
    return results


def run_dataset(name, id_cell, ood_cell, sigma, k_rp):
    print(f"\n########## {name}: {id_cell}/{ood_cell}, σ={sigma}, randproj k={k_rp} ##########", flush=True)
    _, _, _, G_id, g_pids, g_cids = load_cell(id_cell)
    _, _, _, G_ood, _, _ = load_cell(ood_cell)
    print(f"  N={len(G_id)}  n_ids={len(np.unique(g_pids))}", flush=True)

    print("  rank-1 under id...", flush=True); t0 = time.time()
    rates_i = top1_wrong_rate(G_id, g_pids, g_cids)
    print(f"    t={(time.time()-t0)/60:.1f}m", flush=True); t0 = time.time()
    print("  rank-1 under ood...", flush=True)
    rates_o = top1_wrong_rate(G_ood, g_pids, g_cids)
    print(f"    t={(time.time()-t0)/60:.1f}m", flush=True)
    classes = classify(rates_i, rates_o)
    print(f"  partition: {dict(Counter(classes.values()))}", flush=True)

    G_n = noise_degrade(G_id, sigma)
    G_rp = randproj_degrade(G_id, k_rp)

    arms = {
        'id': G_id, 'ood': G_ood, 'noise': G_n, 'randproj': G_rp,
    }
    out = {'dataset': name, 'id_cell': id_cell, 'ood_cell': ood_cell,
           'partition_counts': dict(Counter(classes.values())),
           'arms': {}}
    for arm_name, G_ in arms.items():
        out['arms'][arm_name] = run_arm(G_, g_pids, g_cids, classes, arm_name)
    return out


def print_report(name, res):
    print(f"\n========== {name} — (a) per-class subspace dimensionality (k for 90% var) ==========")
    print(f"  {'class':<10} {'n_ids':>6} {'n_items':>8}  "
          + "  ".join([f"{'k_it_'+arm:>8}" for arm in ('id','ood','noise','randproj')]) + "  "
          + "  ".join([f"{'k_cn_'+arm:>8}" for arm in ('id','ood','noise','randproj')]))
    for c in CLASSES:
        pc_id = res['arms']['id']['per_class'][c]
        if pc_id is None:
            print(f"  {c:<10} 0"); continue
        row = f"  {c:<10} {pc_id['n_ids']:>6d} {pc_id['n_items']:>8d}  "
        for arm in ('id', 'ood', 'noise', 'randproj'):
            pc = res['arms'][arm]['per_class'][c]
            row += f"{pc['k_items_90pct']:>8d}  "
        for arm in ('id', 'ood', 'noise', 'randproj'):
            pc = res['arms'][arm]['per_class'][c]
            row += f"{pc['k_centroids_90pct']:>8d}  "
        print(row)

    print(f"\n========== {name} — (c) cos(global mean, projection onto class subspace) ==========")
    for c in CLASSES:
        pc_any = res['arms']['id']['per_class'][c]
        if pc_any is None: continue
        row = f"  {c:<10}"
        for arm in ('id', 'ood', 'noise', 'randproj'):
            pc = res['arms'][arm]['per_class'][c]
            row += f"  [{arm}] items={pc.get('cos_global_in_items_subspace', float('nan')):.4f}"
            if 'cos_global_in_centroids_subspace' in pc:
                row += f" centroids={pc['cos_global_in_centroids_subspace']:.4f}"
        print(row)

    print(f"\n========== {name} — (b) principal angles summary (k=1× from items) ==========")
    print(f"  {'pair':<22} {'arm':<10}  {'first (deg)':>12} {'median (deg)':>13} {'last (deg)':>11}")
    for pair in ('EASY|OOD-ONLY', 'EASY|SHARED', 'OOD-ONLY|SHARED'):
        for arm in ('id', 'ood', 'noise', 'randproj'):
            pa = res['arms'][arm]['principal_angles']
            if pair not in pa: continue
            r = pa[pair]['1.0x']
            print(f"  {pair:<22} {arm:<10}  {r['first_angle_deg']:>11.2f} "
                  f"{r['angles_median_deg']:>12.2f} {r['last_angle_deg']:>10.2f}")


def main():
    out = {}
    out['MSMT'] = run_dataset('MSMT', 'D', "B'", SIGMA_D, K_RANDPROJ_D)
    out['Market'] = run_dataset('Market', "A'", 'C', SIGMA_A, K_RANDPROJ_A)
    print_report('MSMT', out['MSMT'])
    print_report('Market', out['Market'])

    out_path = RESULTS / 'e40_subspaces.json'
    with open(out_path, 'w') as f: json.dump(out, f, indent=2)
    print(f"\nSaved → {out_path}")


if __name__ == '__main__':
    main()
