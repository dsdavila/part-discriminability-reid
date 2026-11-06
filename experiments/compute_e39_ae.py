"""
E39 parts (a) and (e), added at Plan rev 52 / Channel 2 rev 12.

(a) Per-class-pair margin split into centroid angle and spread-sum separately,
    D vs B' vs noise vs randproj. Pooled across pairs: 18.3° gap lost against
    1.5° blur gained.
(e) Full CMC per class (ranks 1..20) and median rank of ALL true matches per
    class, D vs B' vs noise vs randproj.

Uses confuser pairs for (a) via E38 code path; per-query AP and rank-of-matches
for (e).
"""
import json, time
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter

import sys; sys.path.insert(0, '/workspace/reid-tk')

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
CMC_MAX = 20


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


def top1_and_topk(G, g_pids, g_cids, k_max, chunk=512):
    N = len(G)
    top1 = np.full(N, -1, dtype=np.int64)
    topk = np.full((N, k_max), -1, dtype=np.int64)
    for start in range(0, N, chunk):
        end = min(N, start + chunk)
        sims = G[start:end] @ G.T
        for local, gi in enumerate(range(start, end)):
            exc = (g_pids == g_pids[gi]) & (g_cids == g_cids[gi])
            sims[local, exc] = -2.0
            top1[gi] = int(np.argmax(sims[local]))
            top_idx = np.argpartition(-sims[local], k_max)[:k_max]
            topk[gi] = top_idx[np.argsort(-sims[local, top_idx])]
    return top1, topk


def confusers_from_topk(topk, g_pids, k=10):
    conf = defaultdict(set)
    for i in range(len(g_pids)):
        qid = int(g_pids[i])
        for j in topk[i, :k]:
            j_pid = int(g_pids[j])
            if j_pid != qid:
                conf[qid].add(j_pid)
    return conf


def pair_angle_sum_by_class(conf, ids, mu, spread, classes):
    """Per-class-pair: list of ang (centroid angle), spread_sum (si+sj),
    and derived margin = ang/spread_sum."""
    id_to_idx = {int(p): k for k, p in enumerate(ids)}
    edges = set()
    for i, js in conf.items():
        for j in js:
            edges.add((i, int(j)))
    pairs = set()
    for (i, j) in edges:
        pairs.add((min(i, j), max(i, j)))
    buckets = defaultdict(lambda: {'ang': [], 'spread_sum': [], 'margin': [],
                                    'n': 0, 'n_recip': 0, 'n_one': 0})
    for (i, j) in pairs:
        if i not in id_to_idx or j not in id_to_idx:
            continue
        ci = classes.get(i, 'UNK'); cj = classes.get(j, 'UNK')
        key = tuple(sorted([ci, cj]))
        ki, kj = id_to_idx[i], id_to_idx[j]
        cos_ij = float(np.clip(mu[ki] @ mu[kj], -1.0, 1.0))
        ang = float(np.arccos(cos_ij))
        si, sj = float(spread[ki]), float(spread[kj])
        s_sum = si + sj
        marg = ang / max(s_sum, 1e-12)
        buckets[key]['ang'].append(ang)
        buckets[key]['spread_sum'].append(s_sum)
        buckets[key]['margin'].append(marg)
        buckets[key]['n'] += 1
        if (i, j) in edges and (j, i) in edges:
            buckets[key]['n_recip'] += 1
        else:
            buckets[key]['n_one'] += 1
    out = {}
    for key, b in buckets.items():
        if b['n'] == 0: continue
        out[f"{key[0]}|{key[1]}"] = {
            'n_pairs': b['n'],
            'ang_mean_rad': float(np.mean(b['ang'])),
            'ang_median_rad': float(np.median(b['ang'])),
            'spread_sum_mean_rad': float(np.mean(b['spread_sum'])),
            'spread_sum_median_rad': float(np.median(b['spread_sum'])),
            'margin_mean': float(np.mean(b['margin'])),
            'margin_median': float(np.median(b['margin'])),
        }
    # pooled
    all_ang = []; all_sum = []; all_marg = []
    for b in buckets.values():
        all_ang.extend(b['ang']); all_sum.extend(b['spread_sum']); all_marg.extend(b['margin'])
    out['POOLED'] = {
        'n_pairs': len(all_ang),
        'ang_mean_rad': float(np.mean(all_ang)),
        'spread_sum_mean_rad': float(np.mean(all_sum)),
        'margin_mean': float(np.mean(all_marg)),
    }
    return out


def per_class_cmc_and_median_rank(q_emb, q_pids, q_cids, g_emb, g_pids, g_cids,
                                   classes, max_rank=CMC_MAX):
    """For each query, find ranks of all true positives in the gallery.
    Return per-class CMC @ ranks 1..max_rank, and per-class median rank of
    all true matches (pooled across queries of that class)."""
    sims = q_emb @ g_emb.T
    # structures
    hits_at = {c: np.zeros(max_rank, dtype=np.int64) for c in CLASSES}
    n_queries_with_pos = {c: 0 for c in CLASSES}
    pos_ranks_per_class = {c: [] for c in CLASSES}
    for i in range(len(q_pids)):
        qp, qc = q_pids[i], q_cids[i]
        q_cls = classes.get(int(qp), 'UNK')
        if q_cls not in hits_at: continue
        valid = ~((g_pids == qp) & (g_cids == qc))
        order = np.argsort(-sims[i][valid])
        rel = (g_pids == qp)[valid][order]
        if rel.sum() == 0: continue
        n_queries_with_pos[q_cls] += 1
        first_hit = int(np.argmax(rel))  # rank of first positive (0-indexed)
        if first_hit < max_rank:
            hits_at[q_cls][first_hit:] += 1
        # collect ranks of ALL positives
        pos_idx = np.where(rel)[0] + 1  # 1-indexed
        pos_ranks_per_class[q_cls].extend(pos_idx.tolist())
    # normalize CMC
    cmc = {}
    median_rank_all = {}
    median_rank_first = {}
    for c in CLASSES:
        n = n_queries_with_pos[c]
        cmc[c] = (hits_at[c] / n).tolist() if n else [0.0] * max_rank
        ranks = pos_ranks_per_class[c]
        median_rank_all[c] = float(np.median(ranks)) if ranks else float('nan')
        # first-hit median per query
        # recompute cheaply
    return {'cmc_percent': {c: [100*v for v in cmc[c]] for c in CLASSES},
            'median_rank_all_positives': median_rank_all,
            'n_queries_with_pos': n_queries_with_pos}


def run_dataset(name, id_cell, ood_cell, sigma, k_rp):
    print(f"\n########## {name}: {id_cell}/{ood_cell}, σ={sigma}, randproj k={k_rp} ##########", flush=True)
    q_i, qp_i, qc_i, G_i, gp, gc = load_cell(id_cell)
    q_o, qp_o, qc_o, G_o, _, _ = load_cell(ood_cell)
    print(f"  N={len(G_i)}  n_ids={len(np.unique(gp))}", flush=True)

    # Partition
    t0 = time.time()
    print("  rank-1 under id...", flush=True)
    rates_i = top1_wrong_rate(G_i, gp, gc)
    print(f"    t={(time.time()-t0)/60:.1f}m", flush=True); t0 = time.time()
    print("  rank-1 under ood...", flush=True)
    rates_o = top1_wrong_rate(G_o, gp, gc)
    print(f"    t={(time.time()-t0)/60:.1f}m", flush=True)
    classes = classify(rates_i, rates_o)
    print(f"  partition: {dict(Counter(classes.values()))}", flush=True)

    G_n, Q_n = noise_degrade(G_i, q_i, sigma)
    G_rp, Q_rp = randproj_degrade(G_i, q_i, k_rp)

    arms_gallery = {'id': (G_i, q_i), 'ood': (G_o, q_o),
                    'noise': (G_n, Q_n), 'randproj': (G_rp, Q_rp)}

    result = {}
    # (a) per-class-pair margin decomposition
    print("\n  (a) per-class-pair margin decomposition...", flush=True)
    for arm, (G_, Q_) in arms_gallery.items():
        print(f"    [{arm}] top-10 + per-id centroids+spread...", flush=True)
        t0 = time.time()
        _, topk_arm = top1_and_topk(G_, gp, gc, 10)
        ids_arr, mu_arm, sp_arm = per_id_centroid_spread(G_, gp)
        conf = confusers_from_topk(topk_arm, gp, k=10)
        pair_tbl = pair_angle_sum_by_class(conf, ids_arr, mu_arm, sp_arm, classes)
        print(f"      t={(time.time()-t0)/60:.1f}m  n_pair_buckets={len(pair_tbl)-1}", flush=True)
        result.setdefault('part_a', {})[arm] = pair_tbl

    # (e) per-class CMC and median rank
    print("\n  (e) per-class CMC and median rank of all matches...", flush=True)
    arms_qg = {
        'id':      (q_i, qp_i, qc_i, G_i, gp, gc),
        'ood':     (q_o, qp_o, qc_o, G_o, gp, gc),
        'noise':   (Q_n, qp_i, qc_i, G_n, gp, gc),
        'randproj': (Q_rp, qp_i, qc_i, G_rp, gp, gc),
    }
    for arm, args in arms_qg.items():
        print(f"    [{arm}]...", flush=True); t0 = time.time()
        d = per_class_cmc_and_median_rank(*args, classes, max_rank=CMC_MAX)
        print(f"      t={(time.time()-t0)/60:.1f}m", flush=True)
        result.setdefault('part_e', {})[arm] = d

    return result


def print_report(name, res):
    print(f"\n========== {name} — (a) per-class-pair margin decomposition ==========")
    for arm in ('id', 'ood', 'noise', 'randproj'):
        print(f"\n  [{arm}]")
        pt = res['part_a'][arm]
        print(f"  {'class pair':<25}  {'n_pairs':>8} {'ang_mean (°)':>13} "
              f"{'spread_sum (°)':>14} {'margin_mean':>12}")
        for key in sorted(pt.keys(), key=lambda k: (-pt[k]['n_pairs'] if 'n_pairs' in pt[k] else 0)):
            row = pt[key]
            print(f"  {key:<25}  {row['n_pairs']:>8d} "
                  f"{np.degrees(row['ang_mean_rad']):>12.2f}° "
                  f"{np.degrees(row['spread_sum_mean_rad']):>13.2f}° "
                  f"{row['margin_mean']:>12.4f}")

    print(f"\n========== {name} — (e) per-class CMC@1/@5/@10 and median rank ==========")
    print(f"  {'q_class':<10}  {'arm':<10}  {'CMC@1':>8} {'CMC@5':>8} {'CMC@10':>8} "
          f"{'median rank (all matches)':>26} {'n_queries':>10}")
    for arm in ('id', 'ood', 'noise', 'randproj'):
        for c in CLASSES:
            d = res['part_e'][arm]
            cmc = d['cmc_percent'][c]
            mr = d['median_rank_all_positives'][c]
            n = d['n_queries_with_pos'][c]
            print(f"  {c:<10}  {arm:<10}  {cmc[0]:>7.2f}% {cmc[4]:>7.2f}% {cmc[9]:>7.2f}% "
                  f"{mr:>26.1f} {n:>10d}")


def main():
    out = {}
    out['MSMT'] = run_dataset('MSMT', 'D', "B'", SIGMA_D, K_RANDPROJ_D)
    out['Market'] = run_dataset('Market', "A'", 'C', SIGMA_A, K_RANDPROJ_A)
    print_report('MSMT', out['MSMT'])
    print_report('Market', out['Market'])

    out_path = RESULTS / 'e39_parts_ae.json'
    with open(out_path, 'w') as f: json.dump(out, f, indent=2)
    print(f"\nSaved → {out_path}")


if __name__ == '__main__':
    main()
