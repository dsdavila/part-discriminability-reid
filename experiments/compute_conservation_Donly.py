"""
Conservation reconciliation with D-ONLY partition.

Classes defined using ONLY D's rank-1 error rate (threshold 0.5):
  D-hard: D-rate > 0.5
  D-easy: D-rate <= 0.5

Compute slot-flux matrix under D, B', and noise-degraded-D (at σ=0.0625 matching
B's 5.4% mAP). Ask: does B' route slots toward D-hard identities more than
matched-accuracy noise does?
"""
import json, time
import numpy as np
from pathlib import Path
from collections import defaultdict

CACHE_DIR = Path('/tmp/e12_feature_cache')
RESULTS   = Path('/data/papers/low_dimensional_reid/results')
K = 10
SIGMA_TO_B = 0.0625   # matches B' mAP from noise_curve_v2


def load_cell(cell):
    files = {
        'D':  ('bot_msmt_seed42_msmt.npz', 11659),
        "B'": ('msmt17_test_noREA.npz', 11659),
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


def slot_flux_D_only(topk_idx, g_pids, classes_D_only, k=K):
    nbr = g_pids[topk_idx[:, :k]]
    flux = defaultdict(int)
    for i in range(len(nbr)):
        q_cls = classes_D_only.get(int(g_pids[i]), 'UNK')
        for jj in nbr[i]:
            n_cls = classes_D_only.get(int(jj), 'UNK')
            flux[(q_cls, n_cls)] += 1
    return dict(flux)


def main():
    print("Loading D and B' galleries...", flush=True)
    q_D, _, _, G_D, g_pids, g_cids = load_cell('D')
    _, _, _, G_B, _, _ = load_cell("B'")
    print(f"  N={len(G_D)}  n_ids={len(np.unique(g_pids))}", flush=True)

    print("Rank-1 rate under D (for classification)...", flush=True)
    rates_D = top1_wrong_rate(G_D, g_pids, g_cids)
    classes_Donly = {int(uid): ('D-hard' if r > 0.5 else 'D-easy') for uid, r in rates_D.items()}
    from collections import Counter
    cnts = Counter(classes_Donly.values())
    print(f"  classes: {dict(cnts)}", flush=True)
    items_per_class = {c: int(sum(classes_Donly.get(int(g_pids[i])) == c for i in range(len(g_pids))))
                       for c in ['D-easy', 'D-hard']}
    print(f"  items per class: {items_per_class}", flush=True)

    # Noise-degraded D at σ=0.0625 (match B' mAP)
    print(f"Noise-degrade D at σ={SIGMA_TO_B}...", flush=True)
    G_D_noise, _ = noise_degrade(G_D, q_D, SIGMA_TO_B)

    # Top-10 under each model
    print("Top-10 under D...", flush=True)
    top_D = topk_neighbours(G_D, g_pids, g_cids, K)
    print("Top-10 under B'...", flush=True)
    top_B = topk_neighbours(G_B, g_pids, g_cids, K)
    print("Top-10 under noise-D...", flush=True)
    top_N = topk_neighbours(G_D_noise, g_pids, g_cids, K)

    flux_D = slot_flux_D_only(top_D, g_pids, classes_Donly)
    flux_B = slot_flux_D_only(top_B, g_pids, classes_Donly)
    flux_N = slot_flux_D_only(top_N, g_pids, classes_Donly)

    print(f"\n  {'(q,n)':<22} {'D slots':>12} {'B slots':>12} {'noise slots':>13} {'ΔB−D':>10} {'ΔN−D':>10} {'B−N':>10}")
    out = {'classes': dict(cnts), 'items_per_class': items_per_class, 'rows': []}
    for q in ['D-easy', 'D-hard']:
        for n in ['D-easy', 'D-hard']:
            dD = flux_D.get((q, n), 0)
            dB = flux_B.get((q, n), 0)
            dN = flux_N.get((q, n), 0)
            print(f"  {(q,n)!s:<22} {dD:>12} {dB:>12} {dN:>13} {dB-dD:>+10} {dN-dD:>+10} {dB-dN:>+10}")
            out['rows'].append({'q_class': q, 'n_class': n,
                                'slots_D': dD, 'slots_B': dB, 'slots_noise': dN,
                                'delta_B_minus_D': dB-dD,
                                'delta_noise_minus_D': dN-dD,
                                'B_minus_noise': dB-dN})

    # Column totals
    print(f"\n  Column totals (slots received by class as neighbour):")
    print(f"  {'class':<10} {'D':>12} {'B':>12} {'noise':>13} {'B share':>10} {'null share':>12}")
    N_total = sum(flux_D.values())
    col_tot = {}
    for n_cls in ['D-easy', 'D-hard']:
        d_tot = sum(flux_D.get((q, n_cls), 0) for q in ['D-easy', 'D-hard'])
        b_tot = sum(flux_B.get((q, n_cls), 0) for q in ['D-easy', 'D-hard'])
        n_tot = sum(flux_N.get((q, n_cls), 0) for q in ['D-easy', 'D-hard'])
        null_share = items_per_class[n_cls] / sum(items_per_class.values())
        print(f"  {n_cls:<10} {d_tot:>12} {b_tot:>12} {n_tot:>13} {b_tot/N_total:>10.3f} {null_share:>12.3f}")
        col_tot[n_cls] = {'D': d_tot, 'B': b_tot, 'noise': n_tot,
                          'B_share': b_tot/N_total, 'null_share': null_share}
    out['col_totals'] = col_tot

    # Key quantity: how much does B' route slots to D-hard vs noise does?
    d_hard_B = col_tot['D-hard']['B']
    d_hard_D = col_tot['D-hard']['D']
    d_hard_N = col_tot['D-hard']['noise']
    print(f"\n  D-hard slots D→B diff: {d_hard_B - d_hard_D:+}")
    print(f"  D-hard slots D→noise diff: {d_hard_N - d_hard_D:+}")
    print(f"  Excess (B over noise): {d_hard_B - d_hard_N:+}")
    print(f"  B-share D-hard: {d_hard_B/N_total*100:.2f}%  vs noise-share: {d_hard_N/N_total*100:.2f}%  vs null: {col_tot['D-hard']['null_share']*100:.2f}%")

    out_path = RESULTS / 'conservation_Donly.json'
    with open(out_path, 'w') as f: json.dump(out, f, indent=2)
    print(f"\nSaved → {out_path}")


if __name__ == '__main__':
    main()
