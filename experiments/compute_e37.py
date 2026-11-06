"""
E37 (Daniel/claude-chat, 2026-11-06T16:00Z, Plan rev 46).

Decompose every class's top-k neighbour slots three ways:
  (i)   same identity  (same pid as query)
  (ii)  different identity, same class (per the partition)
  (iii) other class

Report counts and shares per class, D vs B', at k=1, 5, 10, MSMT and Market,
each against matched-accuracy noise null. Slot totals conserve — show sum.

Queries = gallery items with same-pid-same-cam excluded (R-conservation
convention). Partition = rank-1 error > 0.5 in-domain / OOD.

Pre-registered reading: if same-identity collapse + proportional refill
accounts for OOD-ONLY pattern, finding 2 is just a restatement of the
accuracy drop.
"""
import json, time
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict

CACHE_DIR = Path('/tmp/e12_feature_cache')
RESULTS   = Path('/data/papers/low_dimensional_reid/results')
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


def classify(rates_id, rates_ood, thr=0.5):
    classes = {}
    for uid in rates_id:
        ri = rates_id.get(uid, 0.0); ro = rates_ood.get(uid, 0.0)
        if ri <= thr and ro <= thr: classes[uid] = 'EASY'
        elif ri > thr and ro > thr: classes[uid] = 'SHARED'
        elif ri > thr: classes[uid] = 'ID-ONLY'
        else: classes[uid] = 'OOD-ONLY'
    return classes


def wrong_rate(top1_idx, g_pids):
    pred = g_pids[top1_idx]; wrong = pred != g_pids
    return {int(uid): float(wrong[g_pids == uid].mean()) for uid in np.unique(g_pids)}


def decompose_slots(topk, g_pids, classes, k, items_per_class):
    """For each query class, count slots as (same_id, same_cls_diff_id, other_cls).
    Returns per-q-class dict with counts and shares.
    """
    nbr_pids = g_pids[topk[:, :k]]  # (N, k)
    q_pids = g_pids
    out = {c: {'same_id': 0, 'same_cls_diff_id': 0, 'other_cls': 0, 'total': 0}
           for c in CLASSES}
    for i in range(len(q_pids)):
        qp = int(q_pids[i])
        q_cls = classes.get(qp, 'UNK')
        if q_cls not in out: continue
        for j in nbr_pids[i]:
            jj = int(j)
            n_cls = classes.get(jj, 'UNK')
            if jj == qp:
                out[q_cls]['same_id'] += 1
            elif n_cls == q_cls:
                out[q_cls]['same_cls_diff_id'] += 1
            else:
                out[q_cls]['other_cls'] += 1
            out[q_cls]['total'] += 1
    # shares
    for c in CLASSES:
        t = out[c]['total']
        out[c]['share_same_id'] = (100.0 * out[c]['same_id'] / t if t else 0.0)
        out[c]['share_same_cls_diff_id'] = (100.0 * out[c]['same_cls_diff_id'] / t if t else 0.0)
        out[c]['share_other_cls'] = (100.0 * out[c]['other_cls'] / t if t else 0.0)
    return out


def run_dataset(name, id_cell, ood_cell, sigma):
    print(f"\n########## {name}: {id_cell} vs {ood_cell}, noise σ={sigma} ##########",
          flush=True)
    _, _, _, G_id, g_pids, g_cids = load_cell(id_cell)
    _, _, _, G_ood, _, _ = load_cell(ood_cell)
    print(f"  N={len(G_id)}  n_ids={len(np.unique(g_pids))}", flush=True)

    print(f"  degrading {id_cell} σ={sigma}...", flush=True)
    G_noise = noise_degrade(G_id, sigma)

    t0 = time.time()
    print("  top-K under id...", flush=True)
    top1_id, topk_id = top1_and_topk(G_id, g_pids, g_cids, 10)
    print(f"    t={(time.time()-t0)/60:.1f}m", flush=True)
    t0 = time.time()
    print("  top-K under ood...", flush=True)
    top1_ood, topk_ood = top1_and_topk(G_ood, g_pids, g_cids, 10)
    print(f"    t={(time.time()-t0)/60:.1f}m", flush=True)
    t0 = time.time()
    print("  top-K under noise...", flush=True)
    top1_noise, topk_noise = top1_and_topk(G_noise, g_pids, g_cids, 10)
    print(f"    t={(time.time()-t0)/60:.1f}m", flush=True)

    rates_id = wrong_rate(top1_id, g_pids)
    rates_ood = wrong_rate(top1_ood, g_pids)
    classes = classify(rates_id, rates_ood)
    print(f"  partition: {dict(Counter(classes.values()))}", flush=True)
    items_per_class = {c: int(sum(classes.get(int(p)) == c for p in g_pids)) for c in CLASSES}

    result = {
        'dataset': name, 'id_cell': id_cell, 'ood_cell': ood_cell, 'sigma': sigma,
        'items_per_class': items_per_class,
        'k_tables': {},
    }
    for k in K_LEVELS:
        result['k_tables'][str(k)] = {
            'id': decompose_slots(topk_id, g_pids, classes, k, items_per_class),
            'ood': decompose_slots(topk_ood, g_pids, classes, k, items_per_class),
            'noise': decompose_slots(topk_noise, g_pids, classes, k, items_per_class),
        }
    return result


def print_result(res):
    print(f"\n========== {res['dataset']}: {res['id_cell']} vs {res['ood_cell']} ==========")
    for k in K_LEVELS:
        print(f"\n  --- k={k} ---", flush=True)
        for arm_lbl, arm in (('id', res['id_cell']),
                              ('ood', res['ood_cell']),
                              ('noise', f"noise-{res['id_cell']}")):
            d = res['k_tables'][str(k)][arm_lbl]
            print(f"\n    [{arm}]", flush=True)
            print(f"      {'class':<10} {'total':>7} {'same_id':>10} {'same_cls':>10} "
                  f"{'other':>10} | {'%same_id':>9} {'%same_cls':>10} {'%other':>9}")
            for c in CLASSES:
                r = d[c]
                print(f"      {c:<10} {r['total']:>7d} {r['same_id']:>10d} "
                      f"{r['same_cls_diff_id']:>10d} {r['other_cls']:>10d} | "
                      f"{r['share_same_id']:>8.2f}% {r['share_same_cls_diff_id']:>9.2f}% "
                      f"{r['share_other_cls']:>8.2f}%")
            # sum check
            total = sum(d[c]['total'] for c in CLASSES)
            print(f"      TOTAL slots = {total} "
                  f"(expect {sum(res['items_per_class'][c]*k for c in CLASSES)})")


def main():
    out = {}
    out['MSMT'] = run_dataset('MSMT', 'D', "B'", SIGMA_D)
    out['Market'] = run_dataset('Market', "A'", 'C', SIGMA_A)
    for ds in ('MSMT', 'Market'):
        print_result(out[ds])

    out_path = RESULTS / 'e37_neighbour_decomp.json'
    with open(out_path, 'w') as f: json.dump(out, f, indent=2)
    print(f"\nSaved → {out_path}")


if __name__ == '__main__':
    main()
