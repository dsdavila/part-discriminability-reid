"""
Market-side replication of R-conservation-Donly.

Partition by A'-rank-1-error threshold 0.5:
  A'-hard: A'-rate > 0.5 (expected ~23 identities)
  A'-easy: A'-rate <= 0.5

Compute slot-flux matrix under A', C, and noise-degraded-A' at σ=0.062 (matches
C's 26.66% mAP). Report excess slots routed to A'-hard under C vs noise-A'.
"""
import json, time
import numpy as np
from pathlib import Path
from collections import defaultdict

CACHE_DIR = Path('/tmp/e12_feature_cache')
RESULTS   = Path('/data/papers/low_dimensional_reid/results')
K = 10
SIGMA_TO_C = 0.062


def load_cell(cell):
    files = {
        "A'": ('market_test_noREA.npz', 3368),
        'C':  ('bot_msmt_seed42_market.npz', 3368),
    }
    fn, n_q = files[cell]
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


def top1_wrong_rate(G, g_pids, g_cids, chunk=2048):
    N = len(G); t1 = np.full(N, -1, dtype=np.int64)
    for start in range(0, N, chunk):
        end = min(N, start + chunk)
        sims = G[start:end] @ G.T
        for local, gi in enumerate(range(start, end)):
            exc = (g_pids == g_pids[gi]) & (g_cids == g_cids[gi])
            sims[local, exc] = -2.0
            t1[gi] = g_pids[int(np.argmax(sims[local]))]
    wrong = (t1 != g_pids)
    return {int(uid): float(wrong[g_pids == uid].mean()) for uid in np.unique(g_pids)}


def topk_neighbours(G, g_pids, g_cids, k_max, chunk=2048):
    N = len(G); out = np.full((N, k_max), -1, dtype=np.int64)
    for start in range(0, N, chunk):
        end = min(N, start + chunk)
        sims = G[start:end] @ G.T
        for local, gi in enumerate(range(start, end)):
            exc = (g_pids == g_pids[gi]) & (g_cids == g_cids[gi])
            sims[local, exc] = -2.0
            top = np.argpartition(-sims[local], k_max)[:k_max]
            out[gi] = top[np.argsort(-sims[local, top])]
    return out


def slot_flux(topk_idx, g_pids, classes, k=K):
    nbr = g_pids[topk_idx[:, :k]]
    flux = defaultdict(int)
    for i in range(len(nbr)):
        q_cls = classes.get(int(g_pids[i]), 'UNK')
        for jj in nbr[i]:
            n_cls = classes.get(int(jj), 'UNK')
            flux[(q_cls, n_cls)] += 1
    return dict(flux)


def main():
    print("Loading A' and C...", flush=True)
    q_A, _, _, G_A, g_pids, g_cids = load_cell("A'")
    _, _, _, G_C, _, _ = load_cell('C')
    print(f"  N={len(G_A)}  n_ids={len(np.unique(g_pids))}", flush=True)

    print("Rank-1 under A' (for classification)...", flush=True)
    rates_A = top1_wrong_rate(G_A, g_pids, g_cids)
    classes = {int(uid): ('A-hard' if r > 0.5 else 'A-easy') for uid, r in rates_A.items()}
    from collections import Counter
    cnts = Counter(classes.values())
    print(f"  classes: {dict(cnts)}", flush=True)
    items_per_class = {c: int(sum(classes.get(int(g_pids[i])) == c for i in range(len(g_pids))))
                       for c in ['A-easy', 'A-hard']}
    print(f"  items per class: {items_per_class}", flush=True)

    print(f"Noise-degrade A' at σ={SIGMA_TO_C}...", flush=True)
    G_A_noise, _ = noise_degrade(G_A, q_A, SIGMA_TO_C)

    print("Top-10 under A'...", flush=True)
    top_A = topk_neighbours(G_A, g_pids, g_cids, K)
    print("Top-10 under C...", flush=True)
    top_C = topk_neighbours(G_C, g_pids, g_cids, K)
    print("Top-10 under noise-A'...", flush=True)
    top_N = topk_neighbours(G_A_noise, g_pids, g_cids, K)

    flux_A = slot_flux(top_A, g_pids, classes)
    flux_C = slot_flux(top_C, g_pids, classes)
    flux_N = slot_flux(top_N, g_pids, classes)

    print(f"\n  {'(q,n)':<22} {'A slots':>12} {'C slots':>12} {'noise slots':>13} {'ΔC−A':>10} {'ΔN−A':>10} {'C−N':>10}")
    out = {'classes': dict(cnts), 'items_per_class': items_per_class, 'rows': []}
    for q in ['A-easy', 'A-hard']:
        for n in ['A-easy', 'A-hard']:
            dA = flux_A.get((q, n), 0); dC = flux_C.get((q, n), 0); dN = flux_N.get((q, n), 0)
            print(f"  {(q,n)!s:<22} {dA:>12} {dC:>12} {dN:>13} {dC-dA:>+10} {dN-dA:>+10} {dC-dN:>+10}")
            out['rows'].append({'q_class': q, 'n_class': n,
                                'slots_A': dA, 'slots_C': dC, 'slots_noise': dN,
                                'delta_C_minus_A': dC-dA, 'delta_noise_minus_A': dN-dA,
                                'C_minus_noise': dC-dN})

    N_total = sum(flux_A.values())
    print(f"\n  Column totals (slots received as neighbour):")
    print(f"  {'class':<10} {'A':>12} {'C':>12} {'noise':>13} {'C share':>10} {'null share':>12}")
    col_tot = {}
    for n_cls in ['A-easy', 'A-hard']:
        a_tot = sum(flux_A.get((q, n_cls), 0) for q in ['A-easy', 'A-hard'])
        c_tot = sum(flux_C.get((q, n_cls), 0) for q in ['A-easy', 'A-hard'])
        n_tot = sum(flux_N.get((q, n_cls), 0) for q in ['A-easy', 'A-hard'])
        null_share = items_per_class[n_cls] / sum(items_per_class.values())
        print(f"  {n_cls:<10} {a_tot:>12} {c_tot:>12} {n_tot:>13} {c_tot/N_total:>10.3f} {null_share:>12.3f}")
        col_tot[n_cls] = {'A': a_tot, 'C': c_tot, 'noise': n_tot,
                          'C_share': c_tot/N_total, 'null_share': null_share}
    out['col_totals'] = col_tot

    a_hard_C = col_tot['A-hard']['C']
    a_hard_A = col_tot['A-hard']['A']
    a_hard_N = col_tot['A-hard']['noise']
    excess = a_hard_C - a_hard_N
    pct_above_noise = 100 * excess / max(a_hard_N, 1)
    print(f"\n  A-hard slots A→C diff: {a_hard_C - a_hard_A:+}")
    print(f"  A-hard slots A→noise diff: {a_hard_N - a_hard_A:+}")
    print(f"  Excess (C over noise): {excess:+} ({pct_above_noise:+.1f}% above noise prediction)")
    print(f"  C-share A-hard: {a_hard_C/N_total*100:.2f}%  noise: {a_hard_N/N_total*100:.2f}%  null: {col_tot['A-hard']['null_share']*100:.2f}%")

    out_path = RESULTS / 'conservation_Donly_market.json'
    with open(out_path, 'w') as f: json.dump(out, f, indent=2)
    print(f"\nSaved → {out_path}")


if __name__ == '__main__':
    main()
