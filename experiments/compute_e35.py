"""
E35 (Daniel/claude-chat, 2026-11-06T14:10Z, Plan rev 43).

Two parts:
  (a) Recompute R-conservation flux at k=1 and k=5 (and k=10 as reference) on
      MSMT D vs B' and Market A' vs C, with matched-accuracy noise null at each k.
  (b) Condition on queries whose rank-1 is wrong and report which class the
      intruding item belongs to; D vs B' (MSMT) and A' vs C (Market), vs
      proportional null and matched-accuracy noise null.

Queries = gallery items as self-queries (same-pid same-cam excluded), matching
the R-conservation convention used for the k=10 table.

Partitions are the ones already in use:
  rank-1 error > 0.5 in-domain  ⇒ ID-ONLY  (if OOD ≤ 0.5)
  rank-1 error > 0.5 OOD        ⇒ OOD-ONLY (if ID ≤ 0.5)
  both > 0.5                    ⇒ SHARED
  both ≤ 0.5                    ⇒ EASY
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
# Matched-mAP σ (E32)
SIGMA_MSMT = 0.0625
SIGMA_MARKET = 0.062


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


def l2(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(n, 1e-12)


def noise_degrade(G, sigma, seed=42):
    rng = np.random.default_rng(seed)
    g_eps = rng.normal(0, sigma, G.shape).astype(np.float32)
    return l2(G + g_eps)


def top1_and_topk_from_sims(G, g_pids, g_cids, k_max, chunk=512):
    N = len(G)
    top1 = np.full(N, -1, dtype=np.int64)
    topk = np.full((N, k_max), -1, dtype=np.int64)
    for start in range(0, N, chunk):
        end = min(N, start + chunk)
        sims = G[start:end] @ G.T   # (chunk, N)
        for local, gi in enumerate(range(start, end)):
            exc = (g_pids == g_pids[gi]) & (g_cids == g_cids[gi])
            sims[local, exc] = -2.0
            top1[gi] = int(np.argmax(sims[local]))
            top_idx = np.argpartition(-sims[local], k_max)[:k_max]
            topk[gi] = top_idx[np.argsort(-sims[local, top_idx])]
    return top1, topk


def wrong_rate(top1_idx, g_pids):
    pred = g_pids[top1_idx]
    wrong = (pred != g_pids)
    rates = {}
    for uid in np.unique(g_pids):
        mask = g_pids == uid
        rates[int(uid)] = float(wrong[mask].mean())
    return rates


def classify(rates_id, rates_ood, thr=0.5):
    classes = {}
    for uid in rates_id:
        r_id = rates_id.get(uid, 0.0); r_ood = rates_ood.get(uid, 0.0)
        if r_id <= thr and r_ood <= thr: classes[uid] = 'EASY'
        elif r_id > thr and r_ood > thr: classes[uid] = 'SHARED'
        elif r_id > thr: classes[uid] = 'ID-ONLY'
        else: classes[uid] = 'OOD-ONLY'
    return classes


def flux_at_k(topk, g_pids, classes, k):
    """Return (flux dict {(q_cls, n_cls): count}, items_per_class, col_totals_per_n_cls)."""
    nbr = g_pids[topk[:, :k]]
    flux = defaultdict(int)
    for i in range(len(nbr)):
        q_cls = classes.get(int(g_pids[i]), 'UNK')
        for jj in nbr[i]:
            n_cls = classes.get(int(jj), 'UNK')
            flux[(q_cls, n_cls)] += 1
    items_per_cls = {c: int(sum(classes.get(int(p)) == c for p in g_pids)) for c in CLASSES}
    col_totals = {}
    for n_cls in CLASSES:
        col_totals[n_cls] = sum(flux.get((q, n_cls), 0) for q in CLASSES)
    return dict(flux), items_per_cls, col_totals


def table_from_flux(flux, items_per_cls, col_totals, k):
    """Build a rows list and diagonal shares."""
    rows = []
    for q in CLASSES:
        for n in CLASSES:
            slots = flux.get((q, n), 0)
            q_tot = items_per_cls[q] * k
            share_pct = (100.0 * slots / q_tot) if q_tot else 0.0
            rows.append({'q_class': q, 'n_class': n,
                         'slots': int(slots),
                         'share_of_q_slots_pct': float(share_pct)})
    N = sum(items_per_cls.values()) * k
    col_share = {n: (100.0 * col_totals[n] / N if N else 0.0) for n in CLASSES}
    item_share = {c: (100.0 * items_per_cls[c] / sum(items_per_cls.values())
                      if sum(items_per_cls.values()) else 0.0) for c in CLASSES}
    diag_share = {c: (100.0 * flux.get((c, c), 0) / (items_per_cls[c] * k)
                      if items_per_cls[c] else 0.0) for c in CLASSES}
    return {'rows': rows, 'col_totals': col_totals,
            'col_share_pct': col_share,
            'proportional_null_pct': item_share,
            'diag_share_pct': diag_share,
            'items_per_class': items_per_cls, 'k': int(k), 'N': int(N)}


def intruder_composition(top1_idx, g_pids, classes):
    """For queries where rank-1 is wrong, what class does the intruder belong to?"""
    pred = g_pids[top1_idx]
    wrong = pred != g_pids
    # per query class, count intruder class distribution
    per_q_cls = {c: Counter() for c in CLASSES}
    totals = Counter()
    for i in np.where(wrong)[0]:
        q_cls = classes.get(int(g_pids[i]), 'UNK')
        n_cls = classes.get(int(pred[i]), 'UNK')
        if q_cls in per_q_cls:
            per_q_cls[q_cls][n_cls] += 1
            totals[n_cls] += 1
    n_wrong = int(wrong.sum())
    # pool
    pooled_share = {c: (100.0 * totals[c] / n_wrong if n_wrong else 0.0) for c in CLASSES}
    per_q_share = {}
    for q in CLASSES:
        tot = sum(per_q_cls[q].values())
        per_q_share[q] = {c: (100.0 * per_q_cls[q][c] / tot if tot else 0.0) for c in CLASSES}
    return {'n_wrong': n_wrong, 'pooled_intruder_share_pct': pooled_share,
            'per_query_class_intruder_share_pct': per_q_share,
            'per_query_class_counts': {q: dict(c) for q, c in per_q_cls.items()}}


def run_dataset(name, id_cell, ood_cell, sigma):
    print(f"\n########## {name}: {id_cell} vs {ood_cell}, noise σ={sigma} ##########", flush=True)
    _, _, _, G_id, g_pids, g_cids = load_cell(id_cell)
    _, _, _, G_ood, _, _ = load_cell(ood_cell)
    print(f"  N={len(G_id)}  n_ids={len(np.unique(g_pids))}", flush=True)

    print(f"  degrading {id_cell} with σ={sigma}...", flush=True)
    G_noise = noise_degrade(G_id, sigma)

    print(f"  top-{K_MAX} (and top-1) under {id_cell}...", flush=True)
    t0 = time.time()
    top1_id, topk_id = top1_and_topk_from_sims(G_id, g_pids, g_cids, K_MAX)
    print(f"    t={(time.time()-t0)/60:.1f}m", flush=True)

    print(f"  top-{K_MAX} under {ood_cell}...", flush=True)
    t0 = time.time()
    top1_ood, topk_ood = top1_and_topk_from_sims(G_ood, g_pids, g_cids, K_MAX)
    print(f"    t={(time.time()-t0)/60:.1f}m", flush=True)

    print(f"  top-{K_MAX} under noise-{id_cell}...", flush=True)
    t0 = time.time()
    top1_noise, topk_noise = top1_and_topk_from_sims(G_noise, g_pids, g_cids, K_MAX)
    print(f"    t={(time.time()-t0)/60:.1f}m", flush=True)

    rates_id = wrong_rate(top1_id, g_pids)
    rates_ood = wrong_rate(top1_ood, g_pids)
    classes = classify(rates_id, rates_ood)
    cls_cnts = Counter(classes.values())
    print(f"  partition: {dict(cls_cnts)}", flush=True)

    # Per-k flux tables
    k_tables = {}
    for k in K_LEVELS:
        flux_id, items, col_id = flux_at_k(topk_id, g_pids, classes, k)
        flux_ood, _, col_ood = flux_at_k(topk_ood, g_pids, classes, k)
        flux_noise, _, col_noise = flux_at_k(topk_noise, g_pids, classes, k)

        k_tables[str(k)] = {
            'id': table_from_flux(flux_id, items, col_id, k),
            'ood': table_from_flux(flux_ood, items, col_ood, k),
            'noise': table_from_flux(flux_noise, items, col_noise, k),
        }

    # Part (b) intruder composition at rank-1
    print(f"  intruder composition at rank-1...", flush=True)
    intr = {
        'id': intruder_composition(top1_id, g_pids, classes),
        'ood': intruder_composition(top1_ood, g_pids, classes),
        'noise': intruder_composition(top1_noise, g_pids, classes),
    }

    # Items-per-class breakdown — also report at item vs identity level
    ident_per_class = Counter(classes.values())
    items_per_cls = k_tables['10']['id']['items_per_class']
    ident_counts = {c: int(ident_per_class.get(c, 0)) for c in CLASSES}

    return {
        'name': name, 'id_cell': id_cell, 'ood_cell': ood_cell, 'sigma': sigma,
        'items_per_class': items_per_cls,
        'identities_per_class': ident_counts,
        'k_tables': k_tables,
        'intruder': intr,
    }


def print_k_tables(name, res):
    print(f"\n========== {name} — k-flux tables ==========", flush=True)
    for k in K_LEVELS:
        t = res['k_tables'][str(k)]
        print(f"\n  --- k={k} ---", flush=True)
        for arm_lbl, arm in [('id', res['id_cell']),
                              ('ood', res['ood_cell']),
                              ('noise', f"noise-{res['id_cell']}")]:
            d = t[arm_lbl]
            col_share = d['col_share_pct']
            prop = d['proportional_null_pct']
            diag = d['diag_share_pct']
            print(f"\n    [{arm}] col share (%) vs proportional null (%):", flush=True)
            for c in CLASSES:
                print(f"      {c:<10} share={col_share[c]:6.2f}  null={prop[c]:6.2f}  Δ={col_share[c]-prop[c]:+6.2f}", flush=True)
            print(f"    diag share (%) (q=c → n=c within each class's k-slots):", flush=True)
            for c in CLASSES:
                print(f"      {c:<10} {diag[c]:6.2f}", flush=True)


def print_intruder(name, res):
    print(f"\n========== {name} — rank-1 intruder composition ==========", flush=True)
    for arm_lbl, arm in [('id', res['id_cell']),
                          ('ood', res['ood_cell']),
                          ('noise', f"noise-{res['id_cell']}")]:
        d = res['intruder'][arm_lbl]
        print(f"\n  [{arm}] n_wrong_queries={d['n_wrong']}", flush=True)
        print(f"  pooled intruder class share (%) vs items-per-class (proportional null):", flush=True)
        prop = res['k_tables']['10']['id']['proportional_null_pct']
        for c in CLASSES:
            s = d['pooled_intruder_share_pct'][c]
            print(f"    {c:<10} share={s:6.2f}  null={prop[c]:6.2f}  Δ={s-prop[c]:+6.2f}", flush=True)


def main():
    out = {}
    out['MSMT'] = run_dataset('MSMT', 'D', "B'", SIGMA_MSMT)
    out['Market'] = run_dataset('Market', "A'", 'C', SIGMA_MARKET)

    print_k_tables('MSMT', out['MSMT'])
    print_intruder('MSMT', out['MSMT'])
    print_k_tables('Market', out['Market'])
    print_intruder('Market', out['Market'])

    # ID-ONLY counts in both units
    print("\n========== ID-ONLY counts (both units) ==========")
    for dataset in ('MSMT', 'Market'):
        r = out[dataset]
        items = r['items_per_class']['ID-ONLY']
        ids = r['identities_per_class']['ID-ONLY']
        print(f"  {dataset}: identities={ids}  items={items}", flush=True)

    out_path = RESULTS / 'e35_class_flux_kvar.json'
    with open(out_path, 'w') as f: json.dump(out, f, indent=2)
    print(f"\nSaved → {out_path}")


if __name__ == '__main__':
    main()
