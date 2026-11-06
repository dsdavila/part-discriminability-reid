"""
E40 with permutation null + fixed k=200 (replies to Channel 2 rev 21).

Gap 1: corruption nulls (noise/randproj) are degenerate on principal angles.
        Need a STRUCTURE null: randomly partition identities into pseudo-classes
        of exactly the matched sizes (228/692/2138 MSMT), 5 draws, recompute
        (a)(b)(c). Report real classes against the pseudo-class spread.

Gap 2: E40(b) compares subspaces of different dimension across arms. Use
        FIXED k=200 everywhere (above each class's n_ids but below ambient 2048).
"""
import json, time
import numpy as np
from pathlib import Path
from collections import Counter

CACHE_DIR = Path('/tmp/e12_feature_cache')
RESULTS   = Path('/data/papers/low_dimensional_reid/results')
CLASSES = ['EASY', 'ID-ONLY', 'SHARED', 'OOD-ONLY']
FIXED_K = 200
N_PERMUTATIONS = 5

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


def pca_basis_fixed_k(X, k):
    """Top-k principal directions of X (n, d). Returns (d, k) basis."""
    if X.shape[0] < 2:
        return None
    X = X.astype(np.float32)
    mean = X.mean(axis=0, keepdims=True)
    Xc = X - mean
    # Use covariance eigendecomp
    XtX = Xc.T @ Xc
    eigvals, eigvecs = np.linalg.eigh(XtX)
    eigvals = eigvals[::-1]; eigvecs = eigvecs[:, ::-1]
    k_eff = min(k, eigvecs.shape[1])
    return eigvecs[:, :k_eff].astype(np.float32)


def principal_angles(A, B):
    Aq, _ = np.linalg.qr(A)
    Bq, _ = np.linalg.qr(B)
    M = Aq.T @ Bq
    s = np.linalg.svd(M, compute_uv=False)
    s = np.clip(s, -1.0, 1.0)
    return np.arccos(s)


def cos_global_in_subspace(G, basis):
    """||V^T g|| where g is unit global mean direction."""
    gm = G.mean(axis=0); gm /= max(np.linalg.norm(gm), 1e-12)
    coords = basis.T @ gm
    return float(np.linalg.norm(coords))


def compute_arm(G, g_pids, class_map, fixed_k=FIXED_K):
    """For a given arm, compute per-class subspace at fixed_k, principal angles
    between class pairs, and cos(global, projection onto subspace)."""
    bases = {}
    cos_global = {}
    for c in CLASSES:
        idx = np.array([class_map.get(int(p)) == c for p in g_pids])
        if idx.sum() < 2:
            bases[c] = None; cos_global[c] = float('nan'); continue
        X = G[idx]
        b = pca_basis_fixed_k(X, fixed_k)
        bases[c] = b
        cos_global[c] = cos_global_in_subspace(G, b)
    # principal angles between class pairs
    pair_keys = [('EASY', 'OOD-ONLY'), ('EASY', 'SHARED'), ('OOD-ONLY', 'SHARED')]
    pa = {}
    for c1, c2 in pair_keys:
        if bases[c1] is None or bases[c2] is None: continue
        ang = principal_angles(bases[c1], bases[c2])
        pa[f"{c1}|{c2}"] = {
            'first_deg': float(np.degrees(ang[0])),
            'median_deg': float(np.degrees(np.median(ang))),
            'last_deg': float(np.degrees(ang[-1])),
            'mean_deg': float(np.degrees(ang.mean())),
        }
    return {'cos_global': cos_global, 'principal_angles': pa}


def build_permutation_classes(g_pids, class_sizes, seed):
    """Random permutation of identity-to-pseudo-class with sizes matched."""
    rng = np.random.default_rng(seed)
    unique_pids = np.unique(g_pids)
    shuffled = rng.permutation(unique_pids)
    class_map = {}
    idx = 0
    for c, n in class_sizes.items():
        for p in shuffled[idx:idx+n]:
            class_map[int(p)] = c
        idx += n
    # leftover: nothing (sizes must sum to total)
    return class_map


def run_dataset(name, id_cell, ood_cell, sigma, k_rp):
    print(f"\n########## {name}: {id_cell}/{ood_cell}, k_fixed={FIXED_K}, perms={N_PERMUTATIONS} ##########", flush=True)
    _, _, _, G_id, g_pids, g_cids = load_cell(id_cell)
    _, _, _, G_ood, _, _ = load_cell(ood_cell)
    print(f"  N={len(G_id)}  n_ids={len(np.unique(g_pids))}", flush=True)

    t0 = time.time()
    print("  rank-1 under id...", flush=True)
    rates_i = top1_wrong_rate(G_id, g_pids, g_cids)
    print(f"    t={(time.time()-t0)/60:.1f}m", flush=True); t0 = time.time()
    print("  rank-1 under ood...", flush=True)
    rates_o = top1_wrong_rate(G_ood, g_pids, g_cids)
    print(f"    t={(time.time()-t0)/60:.1f}m", flush=True)
    real_classes = classify(rates_i, rates_o)
    real_counts = {c: sum(1 for v in real_classes.values() if v == c) for c in CLASSES}
    print(f"  partition: {real_counts}", flush=True)

    G_n = noise_degrade(G_id, sigma)
    G_rp = randproj_degrade(G_id, k_rp)
    arms = {'id': G_id, 'ood': G_ood, 'noise': G_n, 'randproj': G_rp}

    out = {'dataset': name, 'id_cell': id_cell, 'ood_cell': ood_cell,
           'partition_counts': real_counts,
           'real': {}, 'permutation': {}}

    print("\n  REAL CLASSES:", flush=True)
    for arm, G_ in arms.items():
        t0 = time.time()
        out['real'][arm] = compute_arm(G_, g_pids, real_classes)
        print(f"    [{arm}] done. t={(time.time()-t0)/60:.1f}m", flush=True)

    # Build class_sizes for permutations (skip ID-ONLY since n=1)
    class_sizes = {c: real_counts[c] for c in CLASSES if real_counts[c] > 1}
    print(f"\n  PERMUTATION NULL — {N_PERMUTATIONS} draws, sizes={class_sizes}", flush=True)
    for seed in range(N_PERMUTATIONS):
        print(f"\n  draw {seed}...", flush=True)
        perm_classes = build_permutation_classes(g_pids, class_sizes, seed=seed)
        out['permutation'][str(seed)] = {}
        for arm, G_ in arms.items():
            t0 = time.time()
            out['permutation'][str(seed)][arm] = compute_arm(G_, g_pids, perm_classes)
            print(f"    [{arm}] done. t={(time.time()-t0)/60:.1f}m", flush=True)

    return out


def summarize_dataset(name, res):
    print(f"\n========== {name} — summary ==========")
    pair_keys = ['EASY|OOD-ONLY', 'EASY|SHARED', 'OOD-ONLY|SHARED']
    # median principal angles
    print(f"\nMedian principal angles between class pairs (at k={FIXED_K}):")
    for arm in ('id', 'ood', 'noise', 'randproj'):
        print(f"\n  [{arm}]")
        for pair in pair_keys:
            r = res['real'][arm]['principal_angles'].get(pair)
            if r is None: continue
            # perm null
            perm_medians = []
            for s in res['permutation']:
                d = res['permutation'][s][arm]['principal_angles'].get(pair)
                if d is not None:
                    perm_medians.append(d['median_deg'])
            if perm_medians:
                pm = np.array(perm_medians)
                print(f"    {pair:<22} real={r['median_deg']:>7.3f}° | "
                      f"perm null mean={pm.mean():>7.3f}°  [{pm.min():.3f}, {pm.max():.3f}]  "
                      f"real−perm={r['median_deg'] - pm.mean():>+7.3f}°")
            else:
                print(f"    {pair:<22} real={r['median_deg']:>7.3f}° | (no perm)")

    print(f"\ncos(global mean, projection onto class subspace) at k={FIXED_K}:")
    for arm in ('id', 'ood', 'noise', 'randproj'):
        print(f"\n  [{arm}]")
        for c in CLASSES:
            r = res['real'][arm]['cos_global'].get(c, float('nan'))
            if r != r: continue
            perm_vals = [res['permutation'][s][arm]['cos_global'].get(c, float('nan'))
                         for s in res['permutation']]
            perm_vals = [v for v in perm_vals if v == v]
            if perm_vals:
                pv = np.array(perm_vals)
                print(f"    {c:<10} real={r:.4f} | perm mean={pv.mean():.4f} "
                      f"[{pv.min():.4f}, {pv.max():.4f}]  real−perm={r - pv.mean():+.4f}")
            else:
                print(f"    {c:<10} real={r:.4f} | (no perm)")


def main():
    out = {}
    out['MSMT'] = run_dataset('MSMT', 'D', "B'", SIGMA_D, K_RANDPROJ_D)
    out['Market'] = run_dataset('Market', "A'", 'C', SIGMA_A, K_RANDPROJ_A)
    summarize_dataset('MSMT', out['MSMT'])
    summarize_dataset('Market', out['Market'])

    out_path = RESULTS / 'e40_permutation_null.json'
    with open(out_path, 'w') as f: json.dump(out, f, indent=2)
    print(f"\nSaved → {out_path}")


if __name__ == '__main__':
    main()
