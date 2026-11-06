"""
E38 (claude-chat, Plan rev 49). Class-resolved read-outs over existing caches.

(a) All four classes' column shares at k=1, 5, 10 (closing the E35(a) gap
    where I only reported SHARED in the body of R-E35).
(b) H3 broken out by class pair, with pair counts. For each (q_class, n_class)
    pair of confused identities, report pair count and per-pair stats
    (margin, imbalance, angle).
(c) Per-rank slot share 1–10 and mean appearance rank per class.

MSMT and Market, 3 arms (id, ood, noise) at matched mAP σ.
"""
import json, time
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter

CACHE_DIR = Path('/tmp/e12_feature_cache')
RESULTS   = Path('/data/papers/low_dimensional_reid/results')
K_MAX = 10
K_LEVELS = [1, 5, 10]
CLASSES = ['EASY', 'ID-ONLY', 'SHARED', 'OOD-ONLY']

CELLS = {
    'D':  ('bot_msmt_seed42_msmt.npz',  11659),
    "B'": ('msmt17_test_noREA.npz',     11659),
    "A'": ('market_test_noREA.npz',      3368),
    'C':  ('bot_msmt_seed42_market.npz', 3368),
}
SIGMA_D = 0.0625
SIGMA_A = 0.062


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


def wrong_rate(top1_idx, g_pids):
    pred = g_pids[top1_idx]; wrong = pred != g_pids
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


# --- (a) column shares at k=1/5/10 for all 4 classes
def col_shares(topk, g_pids, classes, k):
    nbr_pids = g_pids[topk[:, :k]].ravel()
    nbr_cls = np.array([classes.get(int(p), 'UNK') for p in nbr_pids])
    N = len(nbr_cls)
    out = {}
    for c in CLASSES:
        out[c] = 100.0 * (nbr_cls == c).sum() / N
    # proportional null (gallery share)
    prop = {c: 100.0 * sum(1 for p in g_pids if classes.get(int(p)) == c) / len(g_pids)
            for c in CLASSES}
    return out, prop


# --- (c) per-rank slot share per class + mean appearance rank
def rank_breakdown(topk, g_pids, classes, k_max=10):
    """For each query class, for each rank 1..10, what fraction of the slot
    at that rank is each neighbour class? Also mean appearance rank of each
    neighbour class within each query class's top-k list (first appearance)."""
    per_q = {c: {'rank_slot_share': {r+1: Counter() for r in range(k_max)},
                 'first_appearance_ranks': defaultdict(list),
                 'n_queries': 0} for c in CLASSES}
    for i in range(len(g_pids)):
        qid = int(g_pids[i]); q_cls = classes.get(qid, 'UNK')
        if q_cls not in per_q: continue
        per_q[q_cls]['n_queries'] += 1
        seen_cls = set()
        for r in range(k_max):
            j = int(topk[i, r])
            n_pid = int(g_pids[j]); n_cls = classes.get(n_pid, 'UNK')
            per_q[q_cls]['rank_slot_share'][r+1][n_cls] += 1
            if n_cls not in seen_cls:
                per_q[q_cls]['first_appearance_ranks'][n_cls].append(r+1)
                seen_cls.add(n_cls)
    # normalize
    out = {}
    for q in CLASSES:
        rs = {}
        for r in range(1, k_max+1):
            cnt = per_q[q]['rank_slot_share'][r]
            tot = sum(cnt.values())
            rs[r] = {c: 100.0 * cnt.get(c, 0) / tot if tot else 0.0 for c in CLASSES}
        far = {}
        for c in CLASSES:
            ranks = per_q[q]['first_appearance_ranks'].get(c, [])
            far[c] = {
                'mean_rank': float(np.mean(ranks)) if ranks else float('nan'),
                'n_with_appearance': len(ranks),
            }
        out[q] = {'n_queries': per_q[q]['n_queries'],
                  'rank_slot_share_pct': rs,
                  'first_appearance_rank': far}
    return out


# --- (b) H3 broken by class pair
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


def topk_confuser_pairs(sim_topk, g_pids, g_cids, G, k=10):
    """From full gallery, get confuser edges per gallery item as query.
    Returns dict id -> set(id)."""
    conf = defaultdict(set)
    for i in range(len(g_pids)):
        qid = int(g_pids[i])
        for j in sim_topk[i, :k]:
            j_pid = int(g_pids[j])
            if j_pid != qid:
                conf[qid].add(j_pid)
    return conf


def pair_stats_by_class_pair(conf, ids, mu, spread, classes):
    """Build pairs, compute per-pair (ang, margin, imbalance), group by
    (min_class, max_class) ordered pair of the two identities' classes."""
    id_to_idx = {int(p): k for k, p in enumerate(ids)}
    # all undirected pairs i<j where at least one direction has edge
    edges_dir = set()
    for i, js in conf.items():
        for j in js:
            edges_dir.add((i, int(j)))
    pairs = set()
    for (i, j) in edges_dir:
        pairs.add((min(i, j), max(i, j)))
    recip = {(i, j) for (i, j) in pairs if (i, j) in edges_dir and (j, i) in edges_dir}
    # by class pair
    buckets = defaultdict(lambda: {'ang': [], 'margin': [], 'imbalance': [],
                                   'n_reciprocal': 0, 'n_one_dir': 0})
    for (i, j) in pairs:
        if i not in id_to_idx or j not in id_to_idx:
            continue
        ci = classes.get(i, 'UNK'); cj = classes.get(j, 'UNK')
        key = tuple(sorted([ci, cj]))
        ki, kj = id_to_idx[i], id_to_idx[j]
        cos_ij = float(np.clip(mu[ki] @ mu[kj], -1.0, 1.0))
        ang = float(np.arccos(cos_ij))
        si, sj = float(spread[ki]), float(spread[kj])
        marg = ang / max(si + sj, 1e-12)
        imb = max(si, sj) / max(min(si, sj), 1e-12)
        buckets[key]['ang'].append(ang)
        buckets[key]['margin'].append(marg)
        buckets[key]['imbalance'].append(imb)
        if (i, j) in recip:
            buckets[key]['n_reciprocal'] += 1
        else:
            buckets[key]['n_one_dir'] += 1
    # summarize
    out = {}
    for key, b in buckets.items():
        n = len(b['ang'])
        if n == 0: continue
        out[f"{key[0]}|{key[1]}"] = {
            'n_pairs': n,
            'n_reciprocal': b['n_reciprocal'],
            'n_one_directional': b['n_one_dir'],
            'ang_mean': float(np.mean(b['ang'])),
            'ang_median': float(np.median(b['ang'])),
            'margin_mean': float(np.mean(b['margin'])),
            'margin_median': float(np.median(b['margin'])),
            'imbalance_mean': float(np.mean(b['imbalance'])),
            'imbalance_median': float(np.median(b['imbalance'])),
        }
    return out


def run_dataset(name, id_cell, ood_cell, sigma):
    print(f"\n########## {name}: {id_cell} vs {ood_cell}, σ={sigma} ##########", flush=True)
    _, _, _, G_id, g_pids, g_cids = load_cell(id_cell)
    _, _, _, G_ood, _, _ = load_cell(ood_cell)
    print(f"  N={len(G_id)}  n_ids={len(np.unique(g_pids))}", flush=True)

    G_noise = noise_degrade(G_id, sigma)

    t0 = time.time()
    print("  top-K id...", flush=True)
    top1_id, topk_id = top1_and_topk(G_id, g_pids, g_cids, 10)
    print(f"    t={(time.time()-t0)/60:.1f}m", flush=True); t0 = time.time()
    print("  top-K ood...", flush=True)
    top1_ood, topk_ood = top1_and_topk(G_ood, g_pids, g_cids, 10)
    print(f"    t={(time.time()-t0)/60:.1f}m", flush=True); t0 = time.time()
    print("  top-K noise...", flush=True)
    top1_noise, topk_noise = top1_and_topk(G_noise, g_pids, g_cids, 10)
    print(f"    t={(time.time()-t0)/60:.1f}m", flush=True)

    classes = classify(wrong_rate(top1_id, g_pids), wrong_rate(top1_ood, g_pids))
    print(f"  partition: {dict(Counter(classes.values()))}", flush=True)

    # (a) column shares all classes
    col = {}
    for k in K_LEVELS:
        col[str(k)] = {}
        for arm, topk in [('id', topk_id), ('ood', topk_ood), ('noise', topk_noise)]:
            shares, prop = col_shares(topk, g_pids, classes, k)
            col[str(k)][arm] = {'shares_pct': shares, 'proportional_null_pct': prop}

    # (c) per-rank slot share and first-appearance rank
    rank = {}
    for arm, topk in [('id', topk_id), ('ood', topk_ood), ('noise', topk_noise)]:
        rank[arm] = rank_breakdown(topk, g_pids, classes, K_MAX)

    # (b) H3 per class pair
    print("  per-id centroid+spread (id)...", flush=True)
    ids_arr, mu_id, sp_id = per_id_centroid_spread(G_id, g_pids)
    print("  per-id centroid+spread (ood)...", flush=True)
    _, mu_ood, sp_ood = per_id_centroid_spread(G_ood, g_pids)
    print("  per-id centroid+spread (noise)...", flush=True)
    _, mu_noise, sp_noise = per_id_centroid_spread(G_noise, g_pids)

    conf_id = topk_confuser_pairs(topk_id, g_pids, g_cids, G_id, k=10)
    conf_ood = topk_confuser_pairs(topk_ood, g_pids, g_cids, G_ood, k=10)
    conf_noise = topk_confuser_pairs(topk_noise, g_pids, g_cids, G_noise, k=10)

    pair_tables = {
        'id': pair_stats_by_class_pair(conf_id, ids_arr, mu_id, sp_id, classes),
        'ood': pair_stats_by_class_pair(conf_ood, ids_arr, mu_ood, sp_ood, classes),
        'noise': pair_stats_by_class_pair(conf_noise, ids_arr, mu_noise, sp_noise, classes),
    }

    return {
        'dataset': name, 'id_cell': id_cell, 'ood_cell': ood_cell, 'sigma': sigma,
        'partition': dict(Counter(classes.values())),
        'col_shares_by_k': col,       # (a)
        'pair_stats_by_class_pair': pair_tables,  # (b)
        'rank_breakdown': rank,       # (c)
    }


def main():
    out = {}
    out['MSMT'] = run_dataset('MSMT', 'D', "B'", SIGMA_D)
    out['Market'] = run_dataset('Market', "A'", 'C', SIGMA_A)

    # pretty-print (a) and (c) headlines
    for ds in ('MSMT', 'Market'):
        res = out[ds]
        print(f"\n========== {ds} column shares (a) ==========")
        for k in K_LEVELS:
            print(f"\n  --- k={k} ---")
            for arm in ('id', 'ood', 'noise'):
                d = res['col_shares_by_k'][str(k)][arm]
                print(f"    [{arm}]  shares: " + "  ".join(
                    f"{c}={d['shares_pct'][c]:5.2f}%" for c in CLASSES))
                print(f"           null:   " + "  ".join(
                    f"{c}={d['proportional_null_pct'][c]:5.2f}%" for c in CLASSES))
                print(f"           Δvs null: " + "  ".join(
                    f"{c}={(d['shares_pct'][c]-d['proportional_null_pct'][c]):+5.2f}" for c in CLASSES))

        # (c) first-appearance ranks
        print(f"\n========== {ds} first-appearance rank per (q_class, n_class) ==========")
        for arm in ('id', 'ood', 'noise'):
            rb = res['rank_breakdown'][arm]
            print(f"\n  [{arm}] mean first appearance rank of neighbour class")
            print(f"  {'q\\\\n':<10}  " + "  ".join(f"{c:<10}" for c in CLASSES))
            for q in CLASSES:
                row = f"  {q:<10}  "
                for c in CLASSES:
                    far = rb[q]['first_appearance_rank'].get(c, {})
                    mr = far.get('mean_rank', float('nan'))
                    row += f"{mr if mr==mr else 0:>6.2f}      "
                print(row)

        # (b) pair stats summary
        print(f"\n========== {ds} H3 pair stats by class pair (k=10) ==========")
        for arm in ('id', 'ood', 'noise'):
            pst = res['pair_stats_by_class_pair'][arm]
            print(f"\n  [{arm}]")
            print(f"  {'class pair':<25}  {'n_pairs':>8} {'n_recip':>8} {'n_onedir':>9} "
                  f"{'ang_mean':>10} {'marg_mean':>10} {'imb_mean':>10}")
            for key, row in sorted(pst.items(), key=lambda kv: -kv[1]['n_pairs']):
                print(f"  {key:<25}  {row['n_pairs']:>8d} {row['n_reciprocal']:>8d} "
                      f"{row['n_one_directional']:>9d} "
                      f"{row['ang_mean']:>10.4f} {row['margin_mean']:>10.4f} "
                      f"{row['imbalance_mean']:>10.4f}")

    # Save
    out_path = RESULTS / 'e38_class_resolved.json'
    with open(out_path, 'w') as f: json.dump(out, f, indent=2)
    print(f"\nSaved → {out_path}")


if __name__ == '__main__':
    main()
