"""
Conservation reconciliation: track where SHARED's extra k-occurrence slots go.

Under D and under B', build a slot-flux matrix: for each (query-identity-class,
neighbour-identity-class) pair, count how many top-k slots fall into it. Compare.

Classes are defined per MSMT by rank-1 error rate threshold 0.5 (same as E31).
"""
import json, time
import numpy as np
from pathlib import Path
from collections import defaultdict

CACHE_DIR = Path('/tmp/e12_feature_cache')
RESULTS   = Path('/data/papers/low_dimensional_reid/results')
K = 10


def load_cell(cell):
    files = {
        'D':  ('bot_msmt_seed42_msmt.npz', 11659),
        "B'": ('msmt17_test_noREA.npz', 11659),
        "A'": ('market_test_noREA.npz',  3368),
        'C':  ('bot_msmt_seed42_market.npz', 3368),
    }
    fn, n_q = files[cell]
    d = np.load(CACHE_DIR / fn); emb = d['emb'].astype(np.float32)
    return (emb[:n_q], d['pids'][:n_q], d['cids'][:n_q],
            emb[n_q:], d['pids'][n_q:], d['cids'][n_q:])


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


def classify(rates_id, rates_ood, thr=0.5):
    classes = {}
    for uid in rates_id:
        r_id = rates_id.get(uid, 0.0); r_ood = rates_ood.get(uid, 0.0)
        if r_id <= thr and r_ood <= thr: classes[uid] = 'EASY'
        elif r_id > thr and r_ood > thr: classes[uid] = 'SHARED'
        elif r_id > thr: classes[uid] = 'ID-ONLY'
        else: classes[uid] = 'OOD-ONLY'
    return classes


def slot_flux(topk_idx, g_pids, classes, k=K):
    """Returns {(q_class, n_class): total top-k slots} and totals per class."""
    nbr = g_pids[topk_idx[:, :k]]
    flux = defaultdict(int)
    for i in range(len(nbr)):
        q_cls = classes.get(int(g_pids[i]), 'UNK')
        for jj in nbr[i]:
            n_cls = classes.get(int(jj), 'UNK')
            flux[(q_cls, n_cls)] += 1
    return dict(flux)


def main():
    print("Loading D and B' galleries...", flush=True)
    _, _, _, G_D, g_pids, g_cids = load_cell('D')
    _, _, _, G_B, _, _ = load_cell("B'")
    print(f"  N={len(G_D)}  n_ids={len(np.unique(g_pids))}", flush=True)

    print("Rank-1 rates under D...", flush=True)
    rates_D = top1_wrong_rate(G_D, g_pids, g_cids)
    print("Rank-1 rates under B'...", flush=True)
    rates_B = top1_wrong_rate(G_B, g_pids, g_cids)
    classes = classify(rates_D, rates_B)
    from collections import Counter
    cnts = Counter(classes.values())
    print(f"  classes: {dict(cnts)}", flush=True)

    print("Top-10 under D...", flush=True)
    top_D = topk_neighbours(G_D, g_pids, g_cids, K)
    print("Top-10 under B'...", flush=True)
    top_B = topk_neighbours(G_B, g_pids, g_cids, K)

    print("\nSlot-flux matrix (query class → neighbour class, top-10 slots):")
    classes_list = ['EASY', 'ID-ONLY', 'SHARED', 'OOD-ONLY']

    flux_D = slot_flux(top_D, g_pids, classes)
    flux_B = slot_flux(top_B, g_pids, classes)
    # Print normalized by query-class slot total
    N_items_per_class = {c: sum((classes[uid] == c) for uid in classes) * sum(
        (classes[int(g_pids[i])] == c) for i in range(0)
    ) for c in classes_list}
    # Actually items per class
    items_per_class = {c: int(sum(classes.get(int(g_pids[i])) == c for i in range(len(g_pids)))) for c in classes_list}
    print(f"  items per class: {items_per_class}", flush=True)
    total_slots_per_q_class = {c: items_per_class[c] * K for c in classes_list}

    rows = []
    print(f"\n  {'(q,n)':<22} {'D slots':>12} {'B slots':>12} {'Δ slots':>12} {'D share (%)':>13} {'B share (%)':>13}")
    for q in classes_list:
        for n in classes_list:
            dD = flux_D.get((q, n), 0); dB = flux_B.get((q, n), 0)
            total_q = total_slots_per_q_class[q]
            d_share = (100 * dD / total_q) if total_q else 0.0
            b_share = (100 * dB / total_q) if total_q else 0.0
            print(f"  {(q,n)!s:<22} {dD:>12} {dB:>12} {dB-dD:>+12} {d_share:>12.2f} {b_share:>12.2f}")
            rows.append({'q_class': q, 'n_class': n,
                         'slots_D': dD, 'slots_B': dB, 'delta_slots': dB-dD,
                         'share_D_of_q_slots_pct': d_share, 'share_B_of_q_slots_pct': b_share})

    # Column sums = how many slots SHARED/EASY/OOD-ONLY/ID-ONLY receive as neighbours total
    print(f"\n  Column totals (total slots received as neighbour):")
    print(f"  {'class':<10} {'D total':>12} {'B total':>12} {'Δ':>12}")
    col_totals = {}
    for n_cls in classes_list:
        d_tot = sum(flux_D.get((q, n_cls), 0) for q in classes_list)
        b_tot = sum(flux_B.get((q, n_cls), 0) for q in classes_list)
        col_totals[n_cls] = {'D': d_tot, 'B': b_tot, 'delta': b_tot - d_tot}
        print(f"  {n_cls:<10} {d_tot:>12} {b_tot:>12} {b_tot-d_tot:>+12}")

    out = {
        'threshold': 0.5, 'n_ids': int(len(classes)),
        'items_per_class': items_per_class,
        'flux_rows': rows,
        'col_totals': col_totals,
    }
    out_path = RESULTS / 'conservation.json'
    with open(out_path, 'w') as f: json.dump(out, f, indent=2)
    print(f"\nSaved → {out_path}")


if __name__ == '__main__':
    main()
