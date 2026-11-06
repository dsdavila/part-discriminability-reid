"""
E31 with rank-1-error-rate partition (threshold 0.5):
  1. Classify each gallery identity per eval set into EASY/SHARED/ID-ONLY/OOD-ONLY
  2. Loop-closing: per-class ΔAP under B' band-pass (j=8, k=all)
  3. Per-ID geometry: spread75, cnorm, eff_rank, cos(μ_i, v_1..v_8), hubness N_k
  4. Report Δg(i) = g(OOD) − g(ID) per identity, aggregated by class
  5. Continuous regression Δg ~ ΔAP + ID_AP, no thresholding
"""
import json, time
import numpy as np
from pathlib import Path
from collections import defaultdict

CACHE_DIR = Path('/tmp/e12_feature_cache')
RESULTS   = Path('/data/papers/low_dimensional_reid/results')
K_HUB     = 10
THR       = 0.5
J_BEST, K_BEST = 8, 'all'

CELL_FILES = {
    "A'": ('market_test_noREA.npz', 3368),
    'C':  ('bot_msmt_seed42_market.npz', 3368),
    'D':  ('bot_msmt_seed42_msmt.npz', 11659),
    "B'": ('msmt17_test_noREA.npz', 11659),
}
EVAL_PAIRS = [('MSMT', 'D', "B'"), ('Market', "A'", 'C')]


def load_cell(cell):
    fn, n_q = CELL_FILES[cell]
    d = np.load(CACHE_DIR / fn); emb = d['emb'].astype(np.float32)
    return (emb[:n_q], d['pids'][:n_q], d['cids'][:n_q],
            emb[n_q:], d['pids'][n_q:], d['cids'][n_q:])


def l2(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(n, 1e-12)


def pca_basis(G):
    mu = G.mean(axis=0, keepdims=True)
    Gc = (G - mu).astype(np.float64)
    _, _, Vt = np.linalg.svd(Gc, full_matrices=False)
    return mu.astype(np.float32), Vt.T.astype(np.float32)


def project(X, mu, V, j, k_or_all):
    D = V.shape[1]
    end = D if k_or_all == 'all' else min(D, j + k_or_all)
    Vk = V[:, j:end]
    return l2((X - mu) @ Vk)


def top1_neighbour_pid(G, g_pids, g_cids, chunk=2048):
    N = len(G); out = np.full(N, -1, dtype=np.int64)
    for start in range(0, N, chunk):
        end = min(N, start + chunk)
        sims = G[start:end] @ G.T
        for local, gi in enumerate(range(start, end)):
            exc = (g_pids == g_pids[gi]) & (g_cids == g_cids[gi])
            sims[local, exc] = -2.0
            out[gi] = g_pids[int(np.argmax(sims[local]))]
    return out


def rank1_err_per_id(G, g_pids, g_cids):
    t1 = top1_neighbour_pid(G, g_pids, g_cids)
    wrong = (t1 != g_pids)
    rates = {}
    for uid in np.unique(g_pids):
        mask = (g_pids == uid)
        rates[int(uid)] = float(wrong[mask].mean())
    return rates


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


def per_id_geometry(G, g_pids, V_top8=None):
    """Returns dict of {pid -> {spread75, cnorm, eff_rank}} plus hubness via k-occurrence."""
    uniq = np.unique(g_pids)
    out = {}
    for uid in uniq:
        E = G[g_pids == uid]
        if len(E) < 2:
            out[int(uid)] = {'spread75': 0.0, 'cnorm': 0.0, 'eff_rank': 1.0, 'n_items': int(len(E))}
            if V_top8 is not None:
                out[int(uid)]['cos_mu_v18'] = [0.0]*8
            continue
        raw_mean = E.mean(axis=0)
        cnorm = float(np.linalg.norm(raw_mean))
        unit_mean = raw_mean / max(cnorm, 1e-12)
        # Spread75
        cos_vals = np.clip(E @ unit_mean, -1.0, 1.0)
        angles = np.degrees(np.arccos(cos_vals))
        sp75 = float(np.percentile(angles, 75))
        # Effective rank of within-identity scatter
        Ec = E - E.mean(0)
        _, s, _ = np.linalg.svd(Ec, full_matrices=False)
        if s.sum() > 1e-12:
            p = s**2 / (s**2).sum()
            p = p[p > 1e-12]
            er = float(np.exp(-np.sum(p * np.log(p))))
        else:
            er = 1.0
        row = {'spread75': sp75, 'cnorm': cnorm, 'eff_rank': er, 'n_items': int(len(E))}
        if V_top8 is not None:
            row['cos_mu_v18'] = [float(abs(unit_mean @ V_top8[:, i])) for i in range(V_top8.shape[1])]
        out[int(uid)] = row
    return out


def per_id_hubness(topk_idx, g_pids, k=K_HUB):
    """k-occurrence count N_k per identity: how many items have me in their top-k."""
    uniq = np.unique(g_pids)
    nbr = g_pids[topk_idx[:, :k]]
    counts = defaultdict(int)
    for i in range(len(nbr)):
        for jj in nbr[i]:
            counts[int(jj)] += 1
    out = {int(uid): float(counts.get(int(uid), 0) / max((g_pids == uid).sum(), 1)) for uid in uniq}
    return out


def per_query_ap(q_emb, q_pids, q_cids, g_emb, g_pids, g_cids):
    sims = q_emb @ g_emb.T
    out = []
    for i in range(len(q_emb)):
        qp, qc = q_pids[i], q_cids[i]
        valid = ~((g_pids == qp) & (g_cids == qc))
        rel = (g_pids == qp) & valid
        if rel.sum() == 0: out.append((int(qp), None)); continue
        order = np.argsort(-sims[i][valid])
        rel_v = rel[valid][order]
        if rel_v.sum() == 0: out.append((int(qp), None)); continue
        pos = np.where(rel_v)[0]
        out.append((int(qp), float(((np.arange(len(pos))+1) / (pos+1)).mean())))
    return out


def classify(rates_id, rates_ood, thr=THR):
    """Classify identity i into EASY/SHARED/ID-ONLY/OOD-ONLY."""
    classes = {}
    for uid in rates_id:
        r_id = rates_id.get(uid, 0.0); r_ood = rates_ood.get(uid, 0.0)
        if r_id <= thr and r_ood <= thr: classes[uid] = 'EASY'
        elif r_id > thr and r_ood > thr: classes[uid] = 'SHARED'
        elif r_id > thr: classes[uid] = 'ID-ONLY'
        else: classes[uid] = 'OOD-ONLY'
    return classes


def main():
    results = {'threshold': THR}

    # Load all four cells' galleries + their rank-1 error rates
    print("Loading cells and computing rank-1 error rates...", flush=True)
    cell_data = {}
    rank1_rates = {}
    for cell in CELL_FILES:
        q, qp, qc, g, gp, gc = load_cell(cell)
        cell_data[cell] = {'q': q, 'qp': qp, 'qc': qc, 'g': g, 'gp': gp, 'gc': gc}
        t0 = time.time()
        rank1_rates[cell] = rank1_err_per_id(g, gp, gc)
        print(f"  {cell}: rank-1 rates t={(time.time()-t0)/60:.1f}m", flush=True)

    # For each eval pair: classify, compute per-ID geometry under both models, loop-closing
    for eval_name, c_id, c_ood in EVAL_PAIRS:
        print(f"\n=== {eval_name}: {c_id} vs {c_ood} ===", flush=True)
        rates_id = rank1_rates[c_id]; rates_ood = rank1_rates[c_ood]
        classes = classify(rates_id, rates_ood)
        from collections import Counter
        print(f"  class counts: {dict(Counter(classes.values()))}", flush=True)

        # PCA top-8 under each model for cos alignment
        g_id = cell_data[c_id]['g']; g_ood = cell_data[c_ood]['g']
        g_pids = cell_data[c_id]['gp']; g_cids = cell_data[c_id]['gc']
        assert np.array_equal(g_pids, cell_data[c_ood]['gp'])

        print(f"  PCA under {c_id}...", flush=True)
        mu_id, V_id = pca_basis(g_id)
        V_id8 = V_id[:, :8]
        print(f"  PCA under {c_ood}...", flush=True)
        mu_ood, V_ood = pca_basis(g_ood)
        V_ood8 = V_ood[:, :8]

        print(f"  per-ID geometry under {c_id}...", flush=True)
        geo_id = per_id_geometry(g_id, g_pids, V_id8)
        print(f"  per-ID geometry under {c_ood}...", flush=True)
        geo_ood = per_id_geometry(g_ood, g_pids, V_ood8)

        print(f"  topk+hubness under both...", flush=True)
        top_id = topk_neighbours(g_id, g_pids, g_cids, K_HUB)
        top_ood = topk_neighbours(g_ood, g_pids, g_cids, K_HUB)
        hub_id = per_id_hubness(top_id, g_pids, K_HUB)
        hub_ood = per_id_hubness(top_ood, g_pids, K_HUB)
        for uid in geo_id:
            geo_id[uid]['hubness'] = hub_id.get(uid, 0.0)
            geo_ood[uid]['hubness'] = hub_ood.get(uid, 0.0)

        # Aggregate Δg by class
        class_delta = defaultdict(lambda: defaultdict(list))
        for uid, cls in classes.items():
            if uid not in geo_id or uid not in geo_ood: continue
            class_delta[cls]['spread75'].append(geo_ood[uid]['spread75'] - geo_id[uid]['spread75'])
            class_delta[cls]['cnorm'].append(geo_ood[uid]['cnorm'] - geo_id[uid]['cnorm'])
            class_delta[cls]['eff_rank'].append(geo_ood[uid]['eff_rank'] - geo_id[uid]['eff_rank'])
            class_delta[cls]['hubness'].append(geo_ood[uid]['hubness'] - geo_id[uid]['hubness'])
            for d in range(8):
                class_delta[cls][f'cos_v{d+1}'].append(geo_ood[uid]['cos_mu_v18'][d] - geo_id[uid]['cos_mu_v18'][d])

        print(f"\n  Class-conditional Δg (OOD − ID):")
        header = ['class', 'n', 'Δspread75', 'Δcnorm', 'Δeff_rank', 'Δhubness']
        print(f"  {'class':<10} {'n':>5} {'Δspread':>9} {'Δcnorm':>9} {'Δeff_rank':>10} {'Δhubness':>10}")
        agg = {}
        for cls in ['EASY', 'ID-ONLY', 'SHARED', 'OOD-ONLY']:
            if cls not in class_delta:
                print(f"  {cls:<10}    0 {'--':>9} {'--':>9} {'--':>10} {'--':>10}")
                continue
            n = len(class_delta[cls]['spread75'])
            msp = np.mean(class_delta[cls]['spread75'])
            mcn = np.mean(class_delta[cls]['cnorm'])
            mer = np.mean(class_delta[cls]['eff_rank'])
            mhb = np.mean(class_delta[cls]['hubness'])
            print(f"  {cls:<10} {n:>5} {msp:>+9.3f} {mcn:>+9.4f} {mer:>+10.4f} {mhb:>+10.3f}")
            agg[cls] = {'n': int(n),
                        'mean_dspread75': float(msp),
                        'mean_dcnorm': float(mcn),
                        'mean_deff_rank': float(mer),
                        'mean_dhubness': float(mhb),
                        'mean_dcos_v18': [float(np.mean(class_delta[cls][f'cos_v{d+1}'])) for d in range(8)]}

        results[eval_name] = {
            'id_cell': c_id, 'ood_cell': c_ood,
            'class_counts': dict(Counter(classes.values())),
            'class_delta_g': agg,
        }

        # Loop-closing specifically for MSMT with band-pass projection of B'
        if eval_name == 'MSMT':
            print(f"\n  LOOP-CLOSING: B' at j={J_BEST}, k={K_BEST}...", flush=True)
            q_B = cell_data[c_ood]['q']; qp_B = cell_data[c_ood]['qp']; qc_B = cell_data[c_ood]['qc']
            g_B = cell_data[c_ood]['g']
            # PCA basis on B' gallery (fresh, not from above)
            mu_B, V_B = pca_basis(g_B)
            proj_q = project(q_B, mu_B, V_B, J_BEST, K_BEST)
            proj_g = project(g_B, mu_B, V_B, J_BEST, K_BEST)
            aps_base = per_query_ap(q_B, qp_B, qc_B, g_B, g_pids, g_cids)
            aps_proj = per_query_ap(proj_q, qp_B, qc_B, proj_g, g_pids, g_cids)
            id_base = defaultdict(list); id_proj = defaultdict(list)
            for (p1, a1), (p2, a2) in zip(aps_base, aps_proj):
                if a1 is None or a2 is None: continue
                id_base[p1].append(a1); id_proj[p1].append(a2)
            cls_ap = defaultdict(lambda: {'base': [], 'proj': [], 'delta': []})
            for uid, cls in classes.items():
                if uid not in id_base: continue
                ap_b = float(np.mean(id_base[uid])); ap_p = float(np.mean(id_proj[uid]))
                cls_ap[cls]['base'].append(ap_b); cls_ap[cls]['proj'].append(ap_p)
                cls_ap[cls]['delta'].append(ap_p - ap_b)
            print(f"  {'class':<10} {'n':>5} {'base AP':>8} {'proj AP':>8} {'ΔAP':>8}  share")
            loop_out = {}
            total_dap = sum(sum(v['delta']) for v in cls_ap.values())
            for cls in ['EASY', 'ID-ONLY', 'SHARED', 'OOD-ONLY']:
                if cls not in cls_ap:
                    print(f"  {cls:<10}    0  --  --  --  --")
                    continue
                n = len(cls_ap[cls]['base'])
                b = np.mean(cls_ap[cls]['base'])
                p = np.mean(cls_ap[cls]['proj'])
                d = np.mean(cls_ap[cls]['delta'])
                s = sum(cls_ap[cls]['delta']) / max(total_dap, 1e-12)
                print(f"  {cls:<10} {n:>5} {b*100:>7.2f}% {p*100:>7.2f}% {d*100:>+7.3f}%  {s*100:>5.1f}%")
                loop_out[cls] = {'n': int(n),
                                 'mean_ap_base_pct': float(b*100),
                                 'mean_ap_proj_pct': float(p*100),
                                 'mean_dap_pct': float(d*100),
                                 'share_total_dap_pct': float(s*100)}
            results['MSMT']['loop_closing'] = loop_out

    out_path = RESULTS / 'e31_rank1_partition.json'
    with open(out_path, 'w') as f: json.dump(results, f, indent=2)
    print(f"\nSaved → {out_path}")


if __name__ == '__main__':
    main()
