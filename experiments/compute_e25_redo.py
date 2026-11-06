"""
E25 redo: AP-stratified attribution budget.
Fix: AP histograms first, then matched-AP-band analysis.
B sets the ceiling. Quintile-share view + variance partitioning inside band.
C vs B composition within matched AP band.
"""
import sys; sys.path.insert(0, '/workspace/reid-tk')
import numpy as np
import json
from pathlib import Path
from scipy import stats

OUT   = Path('/data/papers/low_dimensional_reid/results')
CACHE = Path('/tmp/e12_feature_cache')

FILES = {
    'A': ('market_test.npz',            3368,  13115),
    'B': ('msmt17_test.npz',            11659, 82128),
    'C': ('bot_msmt_seed42_market.npz', 3368,  13115),
    'D': ('bot_msmt_seed42_msmt.npz',   11659, 82128),
}

RNG = np.random.default_rng(42)


def l2norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(n, 1e-12)


def compute_ap(q_emb, q_pid, q_cid, g_emb, g_pids, g_cids):
    """Compute AP for one query (same-cam excluded)."""
    sims = q_emb @ g_emb.T
    # mask same-cam same-id (junk)
    junk = (g_pids == q_pid) & (g_cids == q_cid)
    # relevant = same id, diff cam
    rel = (g_pids == q_pid) & (~junk)
    n_rel = rel.sum()
    if n_rel == 0:
        return None
    # rank non-junk items
    valid = ~junk
    s = sims[valid]
    r = rel[valid]
    order = np.argsort(-s)
    r_sorted = r[order]
    # AP
    cum = np.cumsum(r_sorted)
    ranks = np.arange(1, len(r_sorted) + 1)
    ap = float((cum[r_sorted] / ranks[r_sorted]).mean())
    return ap


def vectorized_dap(q_emb, g_emb, q_pid, q_cid, g_pids, g_cids, mu_hat=None, sorted_pair_angles=None):
    """
    Compute per-query: AP, offender tags and delta-AP contributions.
    Returns: ap (float), per_offender dict with tags and dap values (or None if no relevant).
    Tags:
      overlap: mean cos(offender, nearest correct)
      hub: off_deg / mean_deg (neighbourhood out-degree ratio)
      cone: cos(offender, mu_hat)
      far_tail: mean centroid-angle of outranked correct items
      outlier_q: angle(query, own centroid in gallery)
    """
    sims = q_emb @ g_emb.T           # (G,)
    junk = (g_pids == q_pid) & (g_cids == q_cid)
    rel  = (g_pids == q_pid) & (~junk)
    n_rel = rel.sum()
    if n_rel == 0:
        return None, None

    valid = ~junk
    s_v = sims[valid]; r_v = rel[valid]; g_idx = np.where(valid)[0]
    order = np.argsort(-s_v)
    s_ord = s_v[order]; r_ord = r_v[order]; g_idx_ord = g_idx[order]

    G = len(s_ord)
    # AP
    cum = np.cumsum(r_ord)
    ranks_all = np.arange(1, G+1)
    ap = float((cum[r_ord] / ranks_all[r_ord]).mean())

    # --- Offenders: non-relevant items ranked above at least one relevant ---
    # For each non-rel at rank k, it's an offender if any rel item has rank > k.
    r_positions = np.where(r_ord)[0]   # 0-indexed positions of relevant items
    if len(r_positions) == 0:
        return ap, None
    last_rel = r_positions[-1]
    offender_mask = (~r_ord) & (np.arange(G) < last_rel)
    offender_pos  = np.where(offender_mask)[0]   # 0-indexed rank positions
    if len(offender_pos) == 0:
        return ap, None

    # --- delta-AP (vectorized) ---
    j_arr = np.arange(1, G+1, dtype=np.float32)
    r_arr = ranks_all.astype(np.float32)
    contrib = np.where(r_arr > 1, j_arr / (r_arr * (r_arr - 1)), 0.0)
    cum_contrib = np.cumsum(contrib[::-1])[::-1]
    idxs = np.searchsorted(r_arr, offender_pos + 1 + 1, 'right')  # +1: 0-indexed→1-indexed; +1: inserted after
    # Actually: for offender at 0-indexed rank k (rank k+1), ΔAP = sum of contrib[j] for all j with rank_j > k+1
    # i.e., contrib for ranks > offender_rank = contrib at positions > k
    idxs_correct = offender_pos + 1   # first index to sum from (0-indexed in contrib)
    valid_off = idxs_correct < G
    dap_arr = np.zeros(len(offender_pos), dtype=np.float32)
    dap_arr[valid_off] = cum_contrib[idxs_correct[valid_off]] / G

    # --- Tags ---
    off_g_idx = g_idx_ord[offender_pos]   # original gallery indices of offenders
    off_emb   = g_emb[off_g_idx]          # (n_off, D)

    # 1. overlap: for each offender, cos to nearest correct item (in gallery)
    rel_g_idx = g_idx_ord[r_positions]    # original gallery indices of relevant items
    rel_emb   = g_emb[rel_g_idx]          # (n_rel, D)
    cross_sim  = off_emb @ rel_emb.T      # (n_off, n_rel)
    overlap    = cross_sim.max(axis=1)    # (n_off,)

    # 2. hub: neighbourhood out-degree ratio (approximated on 2000-sample gallery subset)
    eps = 0.9  # cos similarity threshold
    gal_sub_idx = RNG.choice(len(g_emb), min(2000, len(g_emb)), replace=False)
    gal_sub = g_emb[gal_sub_idx]
    all_sim = off_emb @ gal_sub.T          # (n_off, sub)
    deg     = (all_sim >= eps).sum(axis=1).astype(np.float32)
    all_sim_all = gal_sub @ gal_sub.T      # (sub, sub)
    mean_deg = float((all_sim_all >= eps).sum() / len(gal_sub))
    hub = deg / (mean_deg + 1e-12)

    # 3. cone: cos(offender, mu_hat)
    if mu_hat is not None:
        cone = off_emb @ mu_hat
    else:
        cone = np.zeros(len(off_g_idx), dtype=np.float32)

    # 4. far_tail: mean centroid-angle of outranked correct items
    # For offender at 0-indexed rank k, outranked correct items have rank > k → positions > k in r_positions
    # For each offender, find the relevant items ranked below it
    # Relevant centroid angle: angle between relevant item and its class centroid
    # (Here, centroid angle = 1 - cos(relevant_emb, class_centroid))
    # We approximate class centroid as mean of relevant items in gallery
    if n_rel >= 1:
        class_centroid = rel_emb.mean(0)
        class_centroid /= (np.linalg.norm(class_centroid) + 1e-12)
        c_angles = 1.0 - (rel_emb @ class_centroid)   # (n_rel,)  centroid-angle per correct item
        # For each offender at position k, outranked correct items = those at positions > k
        # r_positions is 0-indexed rank of correct items
        # offender at 0-indexed rank opos → outranked correct = r_positions[r_positions > opos]
        ft_list = []
        for opos in offender_pos:
            mask_out = r_positions > opos
            if mask_out.sum() > 0:
                ft_list.append(float(c_angles[mask_out].mean()))
            else:
                ft_list.append(float(c_angles.mean()))
        far_tail = np.array(ft_list, dtype=np.float32)
    else:
        far_tail = np.zeros(len(offender_pos), dtype=np.float32)

    # 5. outlier_q: angle between query and class centroid (single value per query, broadcast to all offenders)
    if n_rel >= 1:
        q_centroid_cos = float(q_emb @ class_centroid)
        outlier_q = np.full(len(offender_pos), 1.0 - q_centroid_cos, dtype=np.float32)
    else:
        outlier_q = np.zeros(len(offender_pos), dtype=np.float32)

    tags = {
        'overlap':   overlap.astype(np.float32),
        'hub':       hub.astype(np.float32),
        'cone':      cone.astype(np.float32),
        'far_tail':  far_tail,
        'outlier_q': outlier_q,
    }
    return ap, {'dap': dap_arr, 'tags': tags, 'n_off': len(offender_pos)}


def run_cell(cell_name, fn, nq, ng, max_queries=5000):
    print(f"\n=== Cell {cell_name} ===", flush=True)
    d = np.load(CACHE / fn)
    emb  = d['emb'].astype(np.float32)
    pids = d['pids']; cids = d['cids']
    q_emb  = l2norm(emb[:nq]);      g_emb  = l2norm(emb[nq:nq+ng])
    q_pids = pids[:nq];             g_pids = pids[nq:nq+ng]
    q_cids = cids[:nq];             g_cids = cids[nq:nq+ng]

    # Gallery mean direction (canonical cos(mu,v1) convention)
    mu_gal = g_emb.mean(0); mu_hat = mu_gal / (np.linalg.norm(mu_gal) + 1e-12)

    # --- Step 1: compute AP for all queries (chunked to control memory) ---
    print("  Computing per-query AP...", flush=True)
    all_ap = []
    all_qi = []
    CHUNK = 256
    for qi_start in range(0, nq, CHUNK):
        qi_end = min(qi_start + CHUNK, nq)
        sim_chunk = q_emb[qi_start:qi_end] @ g_emb.T  # (chunk, G)
        for i, qi in enumerate(range(qi_start, qi_end)):
            sims = sim_chunk[i]
            junk = (g_pids == q_pids[qi]) & (g_cids == q_cids[qi])
            rel  = (g_pids == q_pids[qi]) & (~junk)
            if rel.sum() == 0:
                continue
            valid = ~junk
            s_v = sims[valid]; r_v = rel[valid]
            order = np.argsort(-s_v)
            r_ord = r_v[order]
            cum = np.cumsum(r_ord)
            ranks = np.arange(1, len(r_ord)+1)
            ap_val = float((cum[r_ord] / ranks[r_ord]).mean())
            all_ap.append(ap_val)
            all_qi.append(qi)
    all_ap = np.array(all_ap, dtype=np.float32)
    print(f"  Valid queries: {len(all_ap)}  mean AP={all_ap.mean()*100:.2f}%", flush=True)

    # AP histogram (10 bins 0..1)
    hist, bin_edges = np.histogram(all_ap, bins=10, range=(0, 1))
    ap_hist = {'counts': hist.tolist(), 'bin_edges': bin_edges.tolist()}

    return {
        'cell': cell_name, 'n_queries': int(len(all_ap)),
        'mean_ap_pct': float(all_ap.mean() * 100),
        'ap_hist': ap_hist,
        'ap_p10': float(np.percentile(all_ap, 10)),
        'ap_p25': float(np.percentile(all_ap, 25)),
        'ap_p50': float(np.percentile(all_ap, 50)),
        'ap_p75': float(np.percentile(all_ap, 75)),
        'ap_p90': float(np.percentile(all_ap, 90)),
        '_all_ap': all_ap,
        '_all_qi': all_qi,
        '_emb': (q_emb, g_emb, q_pids, g_pids, q_cids, g_cids, mu_hat),
    }


def run_attribution(cell_res, ap_band, max_queries_in_band=2000):
    """Run attribution on queries with AP in ap_band = (lo, hi)."""
    cell_name = cell_res['cell']
    all_ap = cell_res['_all_ap']
    q_emb, g_emb, q_pids, g_pids, q_cids, g_cids, mu_hat = cell_res['_emb']
    nq = len(q_emb)

    lo, hi = ap_band
    all_ap  = cell_res['_all_ap']
    all_qi  = cell_res['_all_qi']
    # Use pre-computed AP values
    mask = (all_ap >= lo) & (all_ap < hi)
    band_qi_all = [all_qi[i] for i in np.where(mask)[0]]
    band_ap_all = all_ap[mask]

    if len(band_qi_all) == 0:
        return {'cell': cell_name, 'band': ap_band, 'n_in_band': 0}

    # Subsample if too many
    if len(band_qi_all) > max_queries_in_band:
        idx = RNG.choice(len(band_qi_all), max_queries_in_band, replace=False)
        band_qi = [band_qi_all[i] for i in sorted(idx)]
        band_ap = np.array([band_ap_all[i] for i in sorted(idx)], dtype=np.float32)
    else:
        band_qi = band_qi_all
        band_ap = np.array(band_ap_all, dtype=np.float32)

    print(f"  Band {lo:.2f}-{hi:.2f}: {len(band_qi)} queries, mean AP={band_ap.mean()*100:.2f}%", flush=True)

    # Attribution for each query in band
    all_dap = []; all_tags = {k: [] for k in ['overlap','hub','cone','far_tail','outlier_q']}
    per_query_sigma = []; per_query_one_minus_ap = []
    n_offenders = 0
    for ii, qi in enumerate(band_qi):
        ap_val, off_data = vectorized_dap(
            q_emb[qi], g_emb, q_pids[qi], q_cids[qi], g_pids, g_cids, mu_hat=mu_hat)
        if off_data is None:
            per_query_sigma.append(0.0)
        else:
            dap = off_data['dap']; tags = off_data['tags']
            all_dap.extend(dap.tolist())
            for k in all_tags:
                all_tags[k].extend(tags[k].tolist())
            n_offenders += len(dap)
            per_query_sigma.append(float(dap.sum()))
        per_query_one_minus_ap.append(float(1.0 - band_ap[ii]))

    if n_offenders == 0:
        return {'cell': cell_name, 'band': ap_band, 'n_in_band': len(band_qi), 'n_offenders': 0}

    all_dap = np.array(all_dap, dtype=np.float32)
    tag_arrs = {k: np.array(v, dtype=np.float32) for k, v in all_tags.items()}

    # Variance partitioning (unique R² via semi-partial)
    X = np.stack([tag_arrs[k] for k in ['overlap','hub','cone','far_tail','outlier_q']], axis=1)
    y = all_dap

    # Subsample for regression if too large
    if len(y) > 100000:
        idx = RNG.choice(len(y), 100000, replace=False)
        X_r = X[idx]; y_r = y[idx]
    else:
        X_r = X; y_r = y

    tag_names = ['overlap','hub','cone','far_tail','outlier_q']
    # Full R² from multiple regression
    from numpy.linalg import lstsq
    Xb = np.hstack([X_r, np.ones((len(X_r),1))])
    coeffs, _, _, _ = lstsq(Xb, y_r, rcond=None)
    y_hat = Xb @ coeffs
    ss_res = float(((y_r - y_hat)**2).sum())
    ss_tot = float(((y_r - y_r.mean())**2).sum()) + 1e-30
    r2_full = 1.0 - ss_res / ss_tot

    # Unique R²: drop one predictor at a time
    unique_r2 = {}
    for drop_i, tag in enumerate(tag_names):
        keep = [j for j in range(5) if j != drop_i]
        Xr = np.hstack([X_r[:, keep], np.ones((len(X_r),1))])
        c, _, _, _ = lstsq(Xr, y_r, rcond=None)
        yh = Xr @ c
        ss_r = float(((y_r - yh)**2).sum())
        unique_r2[tag] = max(0.0, (ss_r - ss_res) / ss_tot)

    # Quintile shares: fraction of total ΣΔAP in each quintile of each tag
    quintile_shares = {}
    total_dap = float(all_dap.sum()) + 1e-30
    for k in tag_names:
        t = tag_arrs[k]
        qs = np.percentile(t, [0, 20, 40, 60, 80, 100])
        shares = []
        for qi in range(5):
            mask = (t >= qs[qi]) & (t < qs[qi+1]) if qi < 4 else (t >= qs[qi])
            shares.append(float(all_dap[mask].sum() / total_dap))
        quintile_shares[k] = shares

    # Super-additivity
    total_ap_deficit = float((1.0 - band_ap).sum())
    super_add = float(all_dap.sum()) / (total_ap_deficit + 1e-30)

    # ΣΔAP vs (1-AP) correlation (already collected in attribution loop)
    corr_sigma_omap = float(stats.pearsonr(per_query_sigma, per_query_one_minus_ap)[0]) if len(per_query_sigma) > 2 else 0.0

    return {
        'cell': cell_name,
        'band': list(ap_band),
        'n_in_band': len(band_qi),
        'n_offenders': n_offenders,
        'mean_ap_pct': float(band_ap.mean() * 100),
        'super_additivity': super_add,
        'formula_super_add': 'sum_q_sum_g_prime(deltaAP) / sum_q(1-AP)',
        'r2_full': r2_full,
        'unique_r2': unique_r2,
        'quintile_shares': quintile_shares,
        'corr_sigma_dap_vs_1_minus_ap': corr_sigma_omap,
    }


if __name__ == '__main__':
    print("=== E25 redo: AP-stratified attribution ===", flush=True)

    # Step 1: AP histograms for all cells
    cell_results = {}
    for cell, (fn, nq, ng) in FILES.items():
        cell_results[cell] = run_cell(cell, fn, nq, ng)

    # Step 2: Determine AP band from B (ceiling)
    # B's AP distribution: get a band with reasonable n_queries in B
    b_ap = cell_results['B']['_all_ap']
    print(f"\nB AP distribution:")
    for pct in [10, 25, 50, 75, 90]:
        print(f"  p{pct}: {np.percentile(b_ap, pct)*100:.2f}%")

    # Find a band where B has >= 100 queries
    # B has very low AP, so the band should be in the low-AP region
    b_p75 = float(np.percentile(b_ap, 75))
    b_p90 = float(np.percentile(b_ap, 90))
    b_p95 = float(np.percentile(b_ap, 95))

    # Check how many B queries fall in various bands
    for band_hi in [0.05, 0.10, 0.15, 0.20]:
        n_b = int((b_ap < band_hi).sum())
        for cell in ['A','C','D']:
            ap_arr = cell_results[cell]['_all_ap']
            n_cell = int((ap_arr < band_hi).sum())
            print(f"  Band [0, {band_hi:.2f}): B={n_b}  {cell}={n_cell}")

    # Choose band: [0, 0.20) which should have:
    # - Many B queries (almost all, since B mAP=2.56%)
    # - Some A/C/D queries (the worst-performing ones)
    AP_BAND = (0.0, 0.20)
    print(f"\nChosen band: {AP_BAND}", flush=True)

    # Step 3: Attribution inside band for all cells
    print("\nRunning attribution inside AP band...", flush=True)
    band_results = {}
    for cell in ['A', 'B', 'C', 'D']:
        print(f"\n  Cell {cell}:", flush=True)
        res = run_attribution(cell_results[cell], AP_BAND)
        band_results[cell] = res

    # Step 4: Summary
    summary = {
        'ap_band': list(AP_BAND),
        'ap_histograms': {
            cell: {
                'mean_ap_pct': cell_results[cell]['mean_ap_pct'],
                'ap_hist': cell_results[cell]['ap_hist'],
                'percentiles': {
                    'p10': cell_results[cell]['ap_p10'],
                    'p25': cell_results[cell]['ap_p25'],
                    'p50': cell_results[cell]['ap_p50'],
                    'p75': cell_results[cell]['ap_p75'],
                    'p90': cell_results[cell]['ap_p90'],
                }
            }
            for cell in ['A','B','C','D']
        },
        'band_attribution': {cell: {k:v for k,v in band_results[cell].items() if k not in ('_all_ap','_emb')}
                             for cell in ['A','B','C','D']},
    }

    out = OUT / 'e25_redo_ap_stratified.json'
    with open(out, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved → {out}", flush=True)
