"""
E36 (Daniel/claude-chat, 2026-11-06T15:47Z, Plan rev 45).

Per-class ΔAP decomposition of the mAP gap on MSMT (D → B') and Market (A' → C),
with matched-accuracy noise null. Read-out over existing caches — no new runs.

Columns per identity class (EASY, SHARED, ID-ONLY, OOD-ONLY):
  Q_c (query count), Q_c/Q (query share), mean AP in-domain, mean AP transferred,
  mean ΔAP, contribution to gap in mAP points (= Q_c/Q × mean ΔAP), and the
  matched-accuracy noise null arm with the same columns.

Sum check: mAP_id − mAP_ood = Σ_c (Q_c/Q) × mean ΔAP_c (holds by construction).

Partition classes come from rank-1 error rate on gallery-as-query (the R-conservation
convention also used by E35).
"""
import json, time
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter

CACHE_DIR = Path('/tmp/e12_feature_cache')
RESULTS   = Path('/data/papers/low_dimensional_reid/results')
CLASSES = ['EASY', 'ID-ONLY', 'SHARED', 'OOD-ONLY']

CELLS = {
    'D':   {'file': 'bot_msmt_seed42_msmt.npz',  'n_q': 11659},  # MSMT in-domain
    "B'":  {'file': 'msmt17_test_noREA.npz',     'n_q': 11659},  # MSMT cross
    "A'":  {'file': 'market_test_noREA.npz',     'n_q':  3368},  # Market in-domain
    'C':   {'file': 'bot_msmt_seed42_market.npz','n_q':  3368},  # Market cross
}
# Matched-mAP noise σ for in-domain arms (from E32)
SIGMA_D = 0.0625
SIGMA_A = 0.062


def load_cell(name):
    cfg = CELLS[name]
    d = np.load(CACHE_DIR / cfg['file'])
    emb = d['emb'].astype(np.float32)
    nq = cfg['n_q']
    return (emb[:nq], d['pids'][:nq], d['cids'][:nq],
            emb[nq:], d['pids'][nq:], d['cids'][nq:])


def l2(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(n, 1e-12)


def noise_degrade(G, Q, sigma, seed=42):
    rng = np.random.default_rng(seed)
    g_eps = rng.normal(0, sigma, G.shape).astype(np.float32)
    q_eps = rng.normal(0, sigma, Q.shape).astype(np.float32)
    return l2(G + g_eps), l2(Q + q_eps)


def per_query_ap(q_emb, q_pids, q_cids, g_emb, g_pids, g_cids):
    """Return per-query AP array (nan for queries with no positives)."""
    sims = q_emb @ g_emb.T  # (nq, ng)
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
    return aps


def compute_partition_for_gallery(G_id, G_ood, g_pids, g_cids, thr=0.5, chunk=512):
    """Rank-1 error rate per identity, gallery-as-query, same-pid-same-cam excluded.
    Classify identities into EASY/SHARED/ID-ONLY/OOD-ONLY."""
    def rates(G):
        N = len(G)
        top1 = np.full(N, -1, dtype=np.int64)
        for start in range(0, N, chunk):
            end = min(N, start + chunk)
            sims = G[start:end] @ G.T
            for local, gi in enumerate(range(start, end)):
                exc = (g_pids == g_pids[gi]) & (g_cids == g_cids[gi])
                sims[local, exc] = -2.0
                top1[gi] = int(np.argmax(sims[local]))
        pred = g_pids[top1]
        wrong = pred != g_pids
        rates = {}
        for uid in np.unique(g_pids):
            mask = g_pids == uid
            rates[int(uid)] = float(wrong[mask].mean())
        return rates
    r_id = rates(G_id)
    r_ood = rates(G_ood)
    classes = {}
    for uid in r_id:
        ri, ro = r_id.get(uid, 0.0), r_ood.get(uid, 0.0)
        if ri <= thr and ro <= thr: classes[uid] = 'EASY'
        elif ri > thr and ro > thr: classes[uid] = 'SHARED'
        elif ri > thr: classes[uid] = 'ID-ONLY'
        else: classes[uid] = 'OOD-ONLY'
    return classes


def decompose(dataset, id_cell, ood_cell, sigma):
    print(f"\n########## {dataset}: {id_cell} → {ood_cell} (noise σ={sigma}) ##########",
          flush=True)
    q_i, qp_i, qc_i, G_i, gp, gc = load_cell(id_cell)
    q_o, qp_o, qc_o, G_o, _, _ = load_cell(ood_cell)
    # Sanity: query pids/cids match between the two cells
    assert np.array_equal(qp_i, qp_o), f"{dataset}: query pids differ between cells"
    assert np.array_equal(qc_i, qc_o), f"{dataset}: query cids differ between cells"

    print(f"  N_q={len(q_i)}  N_g={len(G_i)}  n_g_ids={len(np.unique(gp))}",
          flush=True)

    # Partition from gallery-as-query rank-1
    print("  computing partition (rank-1 on gallery)...", flush=True)
    t0 = time.time()
    classes = compute_partition_for_gallery(G_i, G_o, gp, gc)
    print(f"    t={(time.time()-t0)/60:.1f}m  classes: {dict(Counter(classes.values()))}",
          flush=True)

    # Noise arm
    print(f"  noise-degrade {id_cell} at σ={sigma}...", flush=True)
    G_n, Q_n = noise_degrade(G_i, q_i, sigma)

    # Per-query AP for all three
    print("  AP in-domain...", flush=True); t0 = time.time()
    ap_i = per_query_ap(q_i, qp_i, qc_i, G_i, gp, gc)
    print(f"    t={(time.time()-t0)/60:.1f}m  mAP={np.nanmean(ap_i)*100:.2f}%", flush=True)
    print("  AP cross...", flush=True); t0 = time.time()
    ap_o = per_query_ap(q_o, qp_o, qc_o, G_o, gp, gc)
    print(f"    t={(time.time()-t0)/60:.1f}m  mAP={np.nanmean(ap_o)*100:.2f}%", flush=True)
    print("  AP noise...", flush=True); t0 = time.time()
    ap_n = per_query_ap(Q_n, qp_i, qc_i, G_n, gp, gc)
    print(f"    t={(time.time()-t0)/60:.1f}m  mAP={np.nanmean(ap_n)*100:.2f}%", flush=True)

    # Group by class via query pid
    q_class = np.array([classes.get(int(p), 'UNK') for p in qp_i])
    out_rows = []
    Q_total = int(np.sum(~np.isnan(ap_i)))
    mAP_i = float(np.nanmean(ap_i) * 100)
    mAP_o = float(np.nanmean(ap_o) * 100)
    mAP_n = float(np.nanmean(ap_n) * 100)

    total_contrib_real = 0.0
    total_contrib_noise = 0.0
    for c in CLASSES:
        mask = (q_class == c) & ~np.isnan(ap_i) & ~np.isnan(ap_o) & ~np.isnan(ap_n)
        Qc = int(mask.sum())
        share = Qc / Q_total if Q_total else 0.0
        ap_i_m = float(ap_i[mask].mean() * 100) if Qc else 0.0
        ap_o_m = float(ap_o[mask].mean() * 100) if Qc else 0.0
        ap_n_m = float(ap_n[mask].mean() * 100) if Qc else 0.0
        delta_real = ap_i_m - ap_o_m
        delta_noise = ap_i_m - ap_n_m
        contrib_real = share * delta_real
        contrib_noise = share * delta_noise
        total_contrib_real += contrib_real
        total_contrib_noise += contrib_noise
        out_rows.append({
            'class': c,
            'Q_c': Qc, 'share': share,
            'mAP_id': ap_i_m, 'mAP_ood': ap_o_m, 'mAP_noise': ap_n_m,
            'delta_real': delta_real, 'delta_noise': delta_noise,
            'contrib_real_pts': contrib_real,
            'contrib_noise_pts': contrib_noise,
        })

    gap_real = mAP_i - mAP_o
    gap_noise = mAP_i - mAP_n
    return {
        'dataset': dataset,
        'id_cell': id_cell, 'ood_cell': ood_cell, 'sigma': sigma,
        'Q_total': Q_total,
        'mAP_id_pct': mAP_i, 'mAP_ood_pct': mAP_o, 'mAP_noise_pct': mAP_n,
        'gap_real_pts': gap_real, 'gap_noise_pts': gap_noise,
        'sum_contrib_real_pts': total_contrib_real,
        'sum_contrib_noise_pts': total_contrib_noise,
        'rows': out_rows,
    }


def print_table(d):
    print(f"\n=== {d['dataset']}: {d['id_cell']} → {d['ood_cell']} ===")
    print(f"  mAP_id={d['mAP_id_pct']:.2f}  mAP_ood={d['mAP_ood_pct']:.2f}  "
          f"mAP_noise={d['mAP_noise_pct']:.2f}")
    print(f"  gap_real={d['gap_real_pts']:.2f}pt  gap_noise={d['gap_noise_pts']:.2f}pt",
          flush=True)
    print(f"  sum_check real={d['sum_contrib_real_pts']:.2f}  "
          f"noise={d['sum_contrib_noise_pts']:.2f}")
    print(f"\n  {'class':<10} {'Q_c':>6} {'share':>7} "
          f"{'mAP_id':>8} {'mAP_ood':>9} {'mAP_noise':>10} "
          f"{'ΔAP_real':>10} {'ΔAP_noise':>11} "
          f"{'contrib_real':>14} {'contrib_noise':>15}")
    for r in d['rows']:
        print(f"  {r['class']:<10} {r['Q_c']:>6d} {r['share']*100:>6.2f}% "
              f"{r['mAP_id']:>8.2f} {r['mAP_ood']:>9.2f} {r['mAP_noise']:>10.2f} "
              f"{r['delta_real']:>+10.2f} {r['delta_noise']:>+11.2f} "
              f"{r['contrib_real_pts']:>+13.2f}pt {r['contrib_noise_pts']:>+13.2f}pt")


def main():
    out = {}
    out['MSMT'] = decompose('MSMT', 'D', "B'", SIGMA_D)
    out['Market'] = decompose('Market', "A'", 'C', SIGMA_A)
    for ds in ('MSMT', 'Market'):
        print_table(out[ds])

    out_path = RESULTS / 'e36_class_ap_decomp.json'
    with open(out_path, 'w') as f: json.dump(out, f, indent=2)
    print(f"\nSaved → {out_path}")


if __name__ == '__main__':
    main()
