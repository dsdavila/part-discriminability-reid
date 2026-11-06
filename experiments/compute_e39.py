"""
E39 (claude-chat, 2026-11-06T18:27Z, Plan rev 51).

Daniel's question: do OOD-ONLY identities stay interleaved with EASY in a set
that contracts uniformly, or are they displaced into a region of their own?

BINDING: D and B' come from different networks → no common basis. Frame-
invariant quantities only (angles inside one model's space, ratios, orderings).
No cross-model coordinate differences.

Four parts:
(a) Per-class-pair margin split into centroid-angle (numerator) and spread-sum
    (denominator) separately, D vs B' vs noise vs randproj.
(b) Class-level geometry over ALL identities (not just confused pairs):
    - within-class dispersion of identity centroids
    - between-class centroid angles
    - between/within ratios
    - mean cos(id centroid, global gallery mean) per class + spread
    - identity counts
(c) Gallery-mean subtraction per class: recompute mAP after subtracting the
    global gallery mean from all gallery items and queries. Per-class ΔmAP.
(d) Noise arm (and randproj where relevant) alongside every quantity.

Pre-registered:
  Uniform contraction → comparable shrink factors everywhere, (c) helps all alike.
  Group displacement → OOD-ONLY angle to EASY holds/grows, dispersion collapses,
    (c) recovers OOD-ONLY specifically.
  Interleaving → EASY↔OOD-ONLY centroid angle shrinks toward within-class dispersion.
"""
import json, time
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict

import sys; sys.path.insert(0, '/workspace/reid-tk')
from torchreid.reid.metrics import evaluate_rank

CACHE_DIR = Path('/tmp/e12_feature_cache')
RESULTS   = Path('/data/papers/low_dimensional_reid/results')

CLASSES = ['EASY', 'ID-ONLY', 'SHARED', 'OOD-ONLY']

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


def noise_degrade(G, Q, sigma, seed=42):
    rng = np.random.default_rng(seed)
    g_eps = rng.normal(0, sigma, G.shape).astype(np.float32)
    q_eps = rng.normal(0, sigma, Q.shape).astype(np.float32)
    return l2(G + g_eps), l2(Q + q_eps)


def randproj_degrade(G, Q, k_target, seed=42):
    D = G.shape[1]
    rng = np.random.default_rng(seed)
    R = rng.standard_normal((D, k_target)).astype(np.float32)
    Qmat, _ = np.linalg.qr(R)
    return l2(G @ Qmat), l2(Q @ Qmat)


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


def per_id_centroid_spread(X, pids):
    ids = np.unique(pids)
    mu = np.zeros((len(ids), X.shape[1]), np.float32)
    spread = np.zeros(len(ids), np.float32)
    for k, pid in enumerate(ids):
        mask = pids == pid
        xi = X[mask]
        mu_k = xi.mean(axis=0)
        mu_k /= max(np.linalg.norm(mu_k), 1e-12)
        mu[k] = mu_k
        cos_x = np.clip(xi @ mu_k, -1.0, 1.0)
        spread[k] = np.arccos(cos_x).mean()
    return ids, mu, spread


def class_geometry(mu, ids, classes, gallery_mean_dir):
    """Class-level geometry over ALL identities.
    Returns per-class:
      - n_ids
      - within-class dispersion: mean angle from class-mean-of-centroids
      - cos to global mean: mean, std (angle in rad)
    Also returns between-class centroid angles (between class means of centroids).
    """
    per_class = {}
    class_means = {}
    for c in CLASSES:
        idxs = [k for k, p in enumerate(ids) if classes.get(int(p)) == c]
        if len(idxs) == 0:
            per_class[c] = None
            continue
        mu_c = mu[idxs]  # (nc, D)
        class_mean = mu_c.mean(axis=0)
        class_mean /= max(np.linalg.norm(class_mean), 1e-12)
        class_means[c] = class_mean
        # within-class dispersion: angle of each id centroid to class mean
        cos_vals = np.clip(mu_c @ class_mean, -1.0, 1.0)
        within_angles = np.arccos(cos_vals)
        # cos to global gallery mean direction
        cos_global = np.clip(mu_c @ gallery_mean_dir, -1.0, 1.0)
        per_class[c] = {
            'n_ids': len(idxs),
            'within_dispersion_rad_mean': float(within_angles.mean()),
            'within_dispersion_rad_std': float(within_angles.std()),
            'cos_to_global_mean_mean': float(cos_global.mean()),
            'cos_to_global_mean_std': float(cos_global.std()),
        }
    # between-class centroid angles
    between = {}
    for c1 in CLASSES:
        for c2 in CLASSES:
            if c1 >= c2: continue
            if c1 not in class_means or c2 not in class_means: continue
            m1, m2 = class_means[c1], class_means[c2]
            cos12 = float(np.clip(m1 @ m2, -1.0, 1.0))
            between[f"{c1}|{c2}"] = {
                'angle_rad': float(np.arccos(cos12)),
                'cos': cos12,
            }
    # between/within ratios: for each class c1, min between-angle to any other class / within-dispersion of c1
    ratios = {}
    for c1 in CLASSES:
        if per_class[c1] is None: continue
        w = per_class[c1]['within_dispersion_rad_mean']
        if w <= 1e-6:
            ratios[c1] = None
            continue
        min_between = None
        for c2 in CLASSES:
            if c1 == c2: continue
            key = f"{min(c1, c2)}|{max(c1, c2)}"
            if key in between:
                a = between[key]['angle_rad']
                if min_between is None or a < min_between:
                    min_between = a
        ratios[c1] = float(min_between / w) if min_between is not None else None
    return {'per_class': per_class,
            'between_class_angle_rad': between,
            'min_between_over_within_ratio': ratios}


def map_r1(q_emb, q_pids, q_cids, g_emb, g_pids, g_cids):
    distmat = -(q_emb @ g_emb.T)
    cmc, mAP = evaluate_rank(distmat, q_pids, g_pids, q_cids, g_cids, max_rank=10)
    return float(mAP * 100), float(cmc[0] * 100)


def per_class_map(q_emb, q_pids, q_cids, g_emb, g_pids, g_cids, classes):
    """Compute per-query AP, then average by query class."""
    sims = q_emb @ g_emb.T
    aps = np.full(len(q_pids), np.nan, np.float32)
    for i in range(len(q_pids)):
        qp, qc = q_pids[i], q_cids[i]
        valid = ~((g_pids == qp) & (g_cids == qc))
        order = np.argsort(-sims[i][valid])
        rel = (g_pids == qp)[valid][order]
        m = int(rel.sum())
        if m == 0: continue
        pos_ranks = np.where(rel)[0] + 1
        j = np.arange(1, m + 1)
        aps[i] = float((j / pos_ranks).mean())
    q_class = np.array([classes.get(int(p), 'UNK') for p in q_pids])
    per_cls = {}
    for c in CLASSES:
        m = (q_class == c) & ~np.isnan(aps)
        n = int(m.sum())
        per_cls[c] = {
            'n_queries': n,
            'mAP_pct': float(aps[m].mean() * 100) if n else 0.0,
        }
    return {'pooled_mAP_pct': float(np.nanmean(aps) * 100), 'per_class': per_cls}


def gallery_mean_subtract(G, Q):
    gm = G.mean(axis=0)
    Gc = G - gm
    Qc = Q - gm
    return l2(Gc), l2(Qc)


def run_dataset(name, id_cell, ood_cell, sigma, k_rp):
    print(f"\n########## {name}: {id_cell}/{ood_cell}, σ={sigma}, randproj k={k_rp} ##########", flush=True)
    q_i, qp_i, qc_i, G_i, gp, gc = load_cell(id_cell)
    q_o, qp_o, qc_o, G_o, _, _ = load_cell(ood_cell)
    print(f"  N={len(G_i)}  n_ids={len(np.unique(gp))}", flush=True)

    # Partition
    print("  rank-1 under id...", flush=True); t0 = time.time()
    rates_i = top1_wrong_rate(G_i, gp, gc)
    print(f"    t={(time.time()-t0)/60:.1f}m", flush=True)
    print("  rank-1 under ood...", flush=True); t0 = time.time()
    rates_o = top1_wrong_rate(G_o, gp, gc)
    print(f"    t={(time.time()-t0)/60:.1f}m", flush=True)
    classes = classify(rates_i, rates_o)
    print(f"  partition: {dict(Counter(classes.values()))}", flush=True)

    # Four arms
    G_n, Q_n = noise_degrade(G_i, q_i, sigma)
    G_rp, Q_rp = randproj_degrade(G_i, q_i, k_rp)

    arms = {
        'id':      (G_i, q_i, qp_i, qc_i),
        'ood':     (G_o, q_o, qp_o, qc_o),
        'noise':   (G_n, Q_n, qp_i, qc_i),
        'randproj': (G_rp, Q_rp, qp_i, qc_i),
    }

    results = {'classes': dict(classes), 'partition_counts': dict(Counter(classes.values()))}
    for arm, (G_, Q_, qp_, qc_) in arms.items():
        print(f"\n  --- arm={arm} ---", flush=True)
        # centroids + spread
        ids_arr, mu_arr, spread_arr = per_id_centroid_spread(G_, gp)
        # global gallery mean direction
        gm = G_.mean(axis=0); gm /= max(np.linalg.norm(gm), 1e-12)
        # (b)
        geom = class_geometry(mu_arr, ids_arr, classes, gm)
        # (c): per-class mAP with and without gallery-mean subtraction
        print(f"    per-class mAP (raw)...", flush=True); t0 = time.time()
        raw_map = per_class_map(Q_, qp_, qc_, G_, gp, gc, classes)
        print(f"      pooled {raw_map['pooled_mAP_pct']:.2f}%  t={(time.time()-t0)/60:.1f}m", flush=True)
        Gc, Qc = gallery_mean_subtract(G_, Q_)
        print(f"    per-class mAP (gallery-mean-subtracted)...", flush=True); t0 = time.time()
        sub_map = per_class_map(Qc, qp_, qc_, Gc, gp, gc, classes)
        print(f"      pooled {sub_map['pooled_mAP_pct']:.2f}%  t={(time.time()-t0)/60:.1f}m", flush=True)
        per_cls_diff = {}
        for c in CLASSES:
            r = raw_map['per_class'][c]['mAP_pct']
            s = sub_map['per_class'][c]['mAP_pct']
            per_cls_diff[c] = {
                'n': raw_map['per_class'][c]['n_queries'],
                'mAP_raw': r, 'mAP_sub': s, 'delta_mAP': s - r,
            }
        pooled_diff = sub_map['pooled_mAP_pct'] - raw_map['pooled_mAP_pct']
        # (a): per-class-pair pooled margin decomposition — use centroid angle
        #   between class-means AND within-class spread-sums.
        class_mean_angle = geom['between_class_angle_rad']
        results[arm] = {
            'geom': geom,
            'mAP_raw_pct': raw_map['pooled_mAP_pct'],
            'mAP_sub_pct': sub_map['pooled_mAP_pct'],
            'pooled_gain_from_submean': pooled_diff,
            'per_class_gain_from_submean_pp': per_cls_diff,
        }
    return results


def print_report(name, res):
    print(f"\n========== {name} — class-level geometry (b) ==========")
    for arm in ('id', 'ood', 'noise', 'randproj'):
        print(f"\n  [{arm}]")
        pc = res[arm]['geom']['per_class']
        print(f"  {'class':<10} {'n':>5} {'within':>12} {'cos_global':>12}")
        for c in CLASSES:
            if pc[c] is None:
                print(f"  {c:<10} 0"); continue
            print(f"  {c:<10} {pc[c]['n_ids']:>5} "
                  f"{np.degrees(pc[c]['within_dispersion_rad_mean']):>7.2f}°±{np.degrees(pc[c]['within_dispersion_rad_std']):.2f} "
                  f"{pc[c]['cos_to_global_mean_mean']:>+6.3f}±{pc[c]['cos_to_global_mean_std']:.3f}")
        print(f"\n  between-class angles (deg):")
        bw = res[arm]['geom']['between_class_angle_rad']
        for k, v in bw.items():
            print(f"    {k:<30} {np.degrees(v['angle_rad']):>6.2f}°  (cos={v['cos']:+.3f})")
        print(f"\n  min-between-angle / within-class-dispersion ratios:")
        for c, r in res[arm]['geom']['min_between_over_within_ratio'].items():
            print(f"    {c:<10} {r if r is not None else float('nan'):.3f}×")

    print(f"\n========== {name} — gallery-mean subtraction (c) ==========")
    print(f"  {'arm':<12} {'pooled raw':>11} {'pooled sub':>11} {'pooled Δ':>9} | "
          f"{'EASY Δ':>8} {'ID-ONLY Δ':>10} {'SHARED Δ':>9} {'OOD-ONLY Δ':>11}")
    for arm in ('id', 'ood', 'noise', 'randproj'):
        d = res[arm]
        row = (f"  {arm:<12} {d['mAP_raw_pct']:>10.2f}% {d['mAP_sub_pct']:>10.2f}% "
               f"{d['pooled_gain_from_submean']:>+8.2f}  | ")
        for c in CLASSES:
            row += f"{d['per_class_gain_from_submean_pp'][c]['delta_mAP']:>+8.2f} "
        print(row)


def main():
    out = {}
    out['MSMT'] = run_dataset('MSMT', 'D', "B'", SIGMA_D, K_RANDPROJ_D)
    out['Market'] = run_dataset('Market', "A'", 'C', SIGMA_A, K_RANDPROJ_A)
    print_report('MSMT', out['MSMT'])
    print_report('Market', out['Market'])

    out_path = RESULTS / 'e39_geometry.json'
    # drop classes dict (large) to keep JSON manageable
    for ds in out:
        if 'classes' in out[ds]:
            del out[ds]['classes']
    with open(out_path, 'w') as f: json.dump(out, f, indent=2)
    print(f"\nSaved → {out_path}")


if __name__ == '__main__':
    main()
