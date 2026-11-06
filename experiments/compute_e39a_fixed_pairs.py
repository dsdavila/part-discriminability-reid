"""
E39(a) fixed pair set (replies to Channel 2 rev 21, gap 3).

Confuser sets in R-E39(a) differ 4× across arms (D 82K / B' 236K / noise 333K /
randproj 269K) and each is selected on its own arm's errors — selection-on-outcome.
Also: randproj has B'-like mAP (5.18 vs 5.40) but D-like margin (0.7577 vs 0.7715),
which undercuts margin-on-confused-pairs as the retrieval-governing quantity.

Fix: compute margin over a FIXED pair set in all four arms:
  (1) D's confuser pairs evaluated everywhere.
  (2) A random sample of identity pairs evaluated everywhere.

Separates \"more pairs confused\" from \"worse margin per pair\".
"""
import json, time
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter

CACHE_DIR = Path('/tmp/e12_feature_cache')
RESULTS   = Path('/data/papers/low_dimensional_reid/results')
CLASSES = ['EASY', 'ID-ONLY', 'SHARED', 'OOD-ONLY']
N_RANDOM_PAIRS = 200_000

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


def top1_wrong_rate_and_topk(G, g_pids, g_cids, k_max=10, chunk=512):
    N = len(G)
    t1 = np.full(N, -1, dtype=np.int64)
    topk = np.full((N, k_max), -1, dtype=np.int64)
    for start in range(0, N, chunk):
        end = min(N, start + chunk)
        sims = G[start:end] @ G.T
        for local, gi in enumerate(range(start, end)):
            exc = (g_pids == g_pids[gi]) & (g_cids == g_cids[gi])
            sims[local, exc] = -2.0
            t1[gi] = int(np.argmax(sims[local]))
            idx = np.argpartition(-sims[local], k_max)[:k_max]
            topk[gi] = idx[np.argsort(-sims[local, idx])]
    pred = g_pids[t1]; wrong = pred != g_pids
    rates = {int(uid): float(wrong[g_pids == uid].mean()) for uid in np.unique(g_pids)}
    return rates, topk


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


def confusers_from_topk(topk, g_pids, k=10):
    """Return set of undirected pairs (i,j) with i<j that are confusers."""
    edges = set()
    for i in range(len(g_pids)):
        qid = int(g_pids[i])
        for j in topk[i, :k]:
            j_pid = int(g_pids[j])
            if j_pid != qid:
                edges.add((min(qid, j_pid), max(qid, j_pid)))
    return edges


def sample_random_pairs(ids, n_pairs, seed=42):
    """Random undirected pairs of identities (i, j), i != j."""
    rng = np.random.default_rng(seed)
    n = len(ids)
    pairs = set()
    tries = 0
    while len(pairs) < n_pairs and tries < n_pairs * 10:
        idx1 = rng.integers(0, n, n_pairs)
        idx2 = rng.integers(0, n, n_pairs)
        for a, b in zip(idx1, idx2):
            if a == b: continue
            pa, pb = int(ids[a]), int(ids[b])
            pairs.add((min(pa, pb), max(pa, pb)))
            if len(pairs) >= n_pairs: break
        tries += 1
    return pairs


def stats_over_pairs(pair_set, ids, mu, spread, classes):
    """Per-pair (ang, spread_sum, margin). Return pooled summary and by-class-pair."""
    id_to_idx = {int(p): k for k, p in enumerate(ids)}
    per_pair = []
    by_cls_pair = defaultdict(lambda: {'ang': [], 'ssum': [], 'marg': []})
    for (i, j) in pair_set:
        if i not in id_to_idx or j not in id_to_idx:
            continue
        ki, kj = id_to_idx[i], id_to_idx[j]
        cos_ij = float(np.clip(mu[ki] @ mu[kj], -1.0, 1.0))
        ang = float(np.arccos(cos_ij))
        si, sj = float(spread[ki]), float(spread[kj])
        s_sum = si + sj
        marg = ang / max(s_sum, 1e-12)
        per_pair.append((ang, s_sum, marg))
        ci = classes.get(i, 'UNK'); cj = classes.get(j, 'UNK')
        key = tuple(sorted([ci, cj]))
        by_cls_pair[key]['ang'].append(ang)
        by_cls_pair[key]['ssum'].append(s_sum)
        by_cls_pair[key]['marg'].append(marg)
    angs = np.array([p[0] for p in per_pair])
    ssums = np.array([p[1] for p in per_pair])
    margs = np.array([p[2] for p in per_pair])
    out = {
        'n_pairs': len(per_pair),
        'ang_mean_rad': float(angs.mean()) if len(angs) else float('nan'),
        'spread_sum_mean_rad': float(ssums.mean()) if len(ssums) else float('nan'),
        'margin_mean': float(margs.mean()) if len(margs) else float('nan'),
        'by_class_pair': {},
    }
    for key, vals in by_cls_pair.items():
        if not vals['ang']: continue
        out['by_class_pair'][f"{key[0]}|{key[1]}"] = {
            'n_pairs': len(vals['ang']),
            'ang_mean_rad': float(np.mean(vals['ang'])),
            'spread_sum_mean_rad': float(np.mean(vals['ssum'])),
            'margin_mean': float(np.mean(vals['marg'])),
        }
    return out


def run_dataset(name, id_cell, ood_cell, sigma, k_rp):
    print(f"\n########## {name}: {id_cell}/{ood_cell} ##########", flush=True)
    _, _, _, G_id, g_pids, g_cids = load_cell(id_cell)
    _, _, _, G_ood, _, _ = load_cell(ood_cell)
    print(f"  N={len(G_id)}  n_ids={len(np.unique(g_pids))}", flush=True)

    t0 = time.time()
    print("  top-10 under id for partition + fixed pair set...", flush=True)
    rates_i, topk_id = top1_wrong_rate_and_topk(G_id, g_pids, g_cids, 10)
    print(f"    t={(time.time()-t0)/60:.1f}m", flush=True); t0 = time.time()
    print("  top-10 under ood for partition...", flush=True)
    rates_o, _ = top1_wrong_rate_and_topk(G_ood, g_pids, g_cids, 10)
    print(f"    t={(time.time()-t0)/60:.1f}m", flush=True)
    classes = classify(rates_i, rates_o)
    print(f"  partition: {dict(Counter(classes.values()))}", flush=True)

    # Fixed pair set 1: D's confusers
    conf_D = confusers_from_topk(topk_id, g_pids, k=10)
    # Fixed pair set 2: random pairs
    all_ids = np.unique(g_pids)
    rand_pairs = sample_random_pairs(all_ids, N_RANDOM_PAIRS, seed=42)
    print(f"  pair sets: D-confusers n={len(conf_D)}  random n={len(rand_pairs)}", flush=True)

    # Four arms
    G_n = noise_degrade(G_id, sigma)
    G_rp = randproj_degrade(G_id, k_rp)
    arms = {'id': G_id, 'ood': G_ood, 'noise': G_n, 'randproj': G_rp}

    out = {'dataset': name,
           'pair_set_sizes': {'D_confusers': len(conf_D), 'random_pairs': len(rand_pairs)},
           'results': {}}
    for arm_name, G_ in arms.items():
        print(f"\n  [{arm_name}] computing per-id centroids+spreads...", flush=True); t0 = time.time()
        ids, mu, spread = per_id_centroid_spread(G_, g_pids)
        print(f"    t={(time.time()-t0)/60:.1f}m", flush=True)
        print(f"    stats over D-confusers...", flush=True); t0 = time.time()
        s1 = stats_over_pairs(conf_D, ids, mu, spread, classes)
        print(f"      t={(time.time()-t0)/60:.1f}m  n_pairs={s1['n_pairs']}", flush=True)
        print(f"    stats over random pairs...", flush=True); t0 = time.time()
        s2 = stats_over_pairs(rand_pairs, ids, mu, spread, classes)
        print(f"      t={(time.time()-t0)/60:.1f}m  n_pairs={s2['n_pairs']}", flush=True)
        out['results'][arm_name] = {
            'on_D_confusers': s1,
            'on_random_pairs': s2,
        }
    return out


def print_report(name, res):
    print(f"\n========== {name} — margin over fixed pair sets ==========")
    print(f"\n  Pair set 1: D's confusers (n={res['pair_set_sizes']['D_confusers']})")
    print(f"  {'arm':<10} {'ang (°)':>10} {'spread_sum (°)':>16} {'margin':>10}")
    for arm in ('id', 'ood', 'noise', 'randproj'):
        d = res['results'][arm]['on_D_confusers']
        print(f"  {arm:<10} {np.degrees(d['ang_mean_rad']):>9.2f}° "
              f"{np.degrees(d['spread_sum_mean_rad']):>15.2f}° "
              f"{d['margin_mean']:>10.4f}")
    print(f"\n  Pair set 2: random identity pairs (n={res['pair_set_sizes']['random_pairs']})")
    print(f"  {'arm':<10} {'ang (°)':>10} {'spread_sum (°)':>16} {'margin':>10}")
    for arm in ('id', 'ood', 'noise', 'randproj'):
        d = res['results'][arm]['on_random_pairs']
        print(f"  {arm:<10} {np.degrees(d['ang_mean_rad']):>9.2f}° "
              f"{np.degrees(d['spread_sum_mean_rad']):>15.2f}° "
              f"{d['margin_mean']:>10.4f}")


def main():
    out = {}
    out['MSMT'] = run_dataset('MSMT', 'D', "B'", SIGMA_D, K_RANDPROJ_D)
    out['Market'] = run_dataset('Market', "A'", 'C', SIGMA_A, K_RANDPROJ_A)
    print_report('MSMT', out['MSMT'])
    print_report('Market', out['Market'])

    out_path = RESULTS / 'e39a_fixed_pairs.json'
    with open(out_path, 'w') as f: json.dump(out, f, indent=2)
    print(f"\nSaved → {out_path}")


if __name__ == '__main__':
    main()
