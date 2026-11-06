"""
E25: AP-loss attribution budget.
Per spec from claude-chat rev 159.

All four cells: A (BoT-Mkt→Mkt), B (BoT-Mkt→MSMT), C (BoT-MSMT→Mkt), D (BoT-MSMT→MSMT)
Junk predicate: same pid + same cam
Market gallery: canonical 13115 (PID != -1, from cache rows 3368:)
MSMT17 gallery: cache rows 11659: (N=82128, gallery size stated)
μ̂: gallery-only, L2-normalized mean (canonical convention)
Hub: k=10, deterministic k-occurrence
"""
import sys; sys.path.insert(0, '/workspace/reid-tk')
import numpy as np
import json
from pathlib import Path
from collections import defaultdict

OUT = Path('/data/papers/low_dimensional_reid/results')
CACHE = Path('/tmp/e12_feature_cache')

FILES = {
    'A': ('market_test.npz',            3368),
    'B': ('msmt17_test.npz',            11659),
    'C': ('bot_msmt_seed42_market.npz', 3368),
    'D': ('bot_msmt_seed42_msmt.npz',   11659),
}

RNG = np.random.default_rng(42)
REG_SAMPLE = 200_000   # max (query, item) pairs for regression per cell
BOOT_N     = 500       # bootstrap replicates over queries

def l2(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(norms, 1e-12)

def load_cell(cell):
    fname, nq = FILES[cell]
    d = np.load(CACHE / fname)
    emb  = l2(d['emb'].astype(np.float32))
    pids = d['pids']
    cids = d['cids']
    return emb[:nq], pids[:nq], cids[:nq], emb[nq:], pids[nq:], cids[nq:]

def compute_centroids(emb, pids):
    """L2-normalised per-identity centroid from provided embeddings."""
    unique = np.unique(pids)
    cents = {}
    for pid in unique:
        mask = pids == pid
        c = emb[mask].mean(0)
        c = c / (np.linalg.norm(c) + 1e-12)
        cents[pid] = c
    return cents   # dict pid -> (2048,) unit vector

def all_pairs_angles(cents_dict):
    """All off-diagonal centroid-pair angles, sorted ascending."""
    pids = sorted(cents_dict.keys())
    C = np.stack([cents_dict[p] for p in pids])   # (n_ids, d)
    G = np.clip(C @ C.T, -1.0, 1.0)
    n = len(pids)
    rows, cols = np.tril_indices(n, k=-1)
    angles = np.degrees(np.arccos(G[rows, cols])).astype(np.float32)
    return np.sort(angles), pids, C

def compute_hub_scores(q_emb, g_emb, g_pids, g_cids, q_pids, q_cids, k=10, batch=512):
    """k-occurrence: how many queries has each gallery item appeared in top-k for."""
    N_Q, N_G = len(q_emb), len(g_emb)
    counts = np.zeros(N_G, dtype=np.int32)
    for start in range(0, N_Q, batch):
        end = min(start + batch, N_Q)
        sims = q_emb[start:end] @ g_emb.T   # (batch, N_G)
        for qi in range(end - start):
            gq_i = start + qi
            junk = (g_pids == q_pids[gq_i]) & (g_cids == q_cids[gq_i])
            sim_row = sims[qi].copy()
            sim_row[junk] = -np.inf
            top_k = np.argpartition(sim_row, -k)[-k:]
            counts[top_k] += 1
    return counts   # (N_G,)

def delta_ap_and_tags(q_emb, q_pids, q_cids,
                      g_emb, g_pids, g_cids,
                      cents_gal, mu_hat, hub_scores,
                      all_pair_angles_sorted,
                      batch=128):
    """
    Returns per-cell aggregate results:
      - total_ap_deficit: sum over queries of (1 - AP/100 if normalized else 1 - AP)
      - total_sigma_dap: sum over all (query, offender) pairs of ΔAP(g')
      - quintile_counts: dict tag -> 5-elem array of cumulative ΣΔAP per quintile
      - sample_rows: (N_sample, 7) array: [overlap_pct, hub_pct, cone_pct, far_tail_pct, outlier_q_pct, dap, query_idx]
      - G_distribution: array of G values per query
    """
    N_Q = len(q_emb)
    N_G = len(g_emb)
    D = g_emb.shape[1]

    # Precompute percentile lookups
    # overlap: all_pair_angles_sorted already sorted
    # hub: sort indices
    hub_sorted = np.sort(hub_scores.astype(np.float32))
    # cone: cos(g, mu_hat) for all gallery items
    cone_scores = (g_emb @ mu_hat).astype(np.float32)   # (N_G,)
    cone_sorted = np.sort(cone_scores)
    # far-tail correct: arccos(g, u_pid) — computed per query
    # outlier query: arccos(q, u_q_pid) — computed per query

    # Build centroid matrix indexed by pid
    pid_list = sorted(cents_gal.keys())
    pid_idx  = {pid: i for i, pid in enumerate(pid_list)}
    C_mat    = np.stack([cents_gal[p] for p in pid_list]).astype(np.float32)   # (n_ids, d)

    def pctile(val, sorted_arr):
        return float(np.searchsorted(sorted_arr, val, side='left')) / len(sorted_arr)

    def pctile_batch(vals, sorted_arr):
        return np.searchsorted(sorted_arr, vals, side='left').astype(np.float32) / len(sorted_arr)

    TAGS  = ['overlap', 'hub', 'cone', 'far_tail', 'outlier_q']
    quint_bins = {t: np.zeros(5) for t in TAGS}
    G_list     = []
    ap_list    = []
    sigma_dap_list = []

    # Reservoir sampling for regression
    sample_list = []   # list of (overlap, hub, cone, far_tail, outlier_q, dap, q_idx)
    reservoir_n = 0

    for start in range(0, N_Q, batch):
        end = min(start + batch, N_Q)
        sims = (q_emb[start:end] @ g_emb.T).astype(np.float32)   # (batch, N_G)

        for bi, qi in enumerate(range(start, end)):
            q_pid  = q_pids[qi]
            q_cid  = q_cids[qi]

            sim = sims[bi].copy()
            junk_mask    = (g_pids == q_pid) & (g_cids == q_cid)
            correct_mask = (g_pids == q_pid) & ~junk_mask
            wrong_mask   = (g_pids != q_pid)

            # Sort gallery excluding junk
            valid_mask = ~junk_mask
            valid_idx  = np.where(valid_mask)[0]
            sim_valid  = sim[valid_idx]
            sorted_pos = np.argsort(-sim_valid)    # descending
            ranked_idx = valid_idx[sorted_pos]     # gallery indices in rank order
            ranked_pid = g_pids[ranked_idx]

            G = int(correct_mask.sum())
            if G == 0:
                continue
            G_list.append(G)

            # Ranks of correct items (1-indexed)
            is_correct_ranked = (ranked_pid == q_pid)
            correct_positions = np.where(is_correct_ranked)[0]   # 0-indexed positions in ranked list
            r = correct_positions + 1   # 1-indexed ranks
            r_G = int(r[-1])

            # AP
            j_arr = np.arange(1, G + 1)
            ap = float(np.sum(j_arr / r)) / G
            ap_list.append(ap)

            # Identify offending items (wrong identity, ranked above r_G)
            # In ranked list: positions 0..r_G-2 that are wrong identity
            wrong_above = np.where((ranked_pid[:r_G] != q_pid))[0]  # 0-indexed in ranked list
            if len(wrong_above) == 0:
                sigma_dap_list.append(0.0)
                continue

            # For each offending item, compute ΔAP
            # S(g') = {j : rank of g' < r_j} i.e. correct items outranked by g'
            # rank of g' in full valid list = pos + 1
            offender_ranks = wrong_above + 1   # 1-indexed
            # correct items outranked by g' at rank og': S(g') = {j : r_j > og'}
            # For each offender (rank og'), sum over j in S(g') of j*(1/(r_j-1) - 1/r_j)

            sigma_dap = 0.0
            # Precompute j/r*(r-1) = j / r_j*(r_j-1) for each correct item
            # ΔAP contribution from correct item j outranked by g': j*(1/(r_j-1) - 1/r_j) / G
            j_contrib = j_arr / (r * (r - 1) + 1e-12) / G   # (G,) avoid /0 at r=1
            # for r_j=1: 1/(r_j-1) → inf; but a correct item at r=1 can't be outranked → fine

            # Outlier query tag (per query)
            if q_pid in pid_idx:
                u_q_pid = C_mat[pid_idx[q_pid]]
                q_emb_unit = q_emb[qi] / (np.linalg.norm(q_emb[qi]) + 1e-12)
                cos_q_cent = float(np.clip(q_emb_unit @ u_q_pid, -1.0, 1.0))
                angle_q_cent = float(np.degrees(np.arccos(np.clip(cos_q_cent, -1, 1))))
            else:
                angle_q_cent = 90.0
            # Percentile of outlier query in all queries → we'll compute after per-cell
            # For now store raw angle; convert to pctile after collecting all

            for og_pos, og_rank in zip(wrong_above, offender_ranks):
                gal_g_idx = int(ranked_idx[og_pos])

                # S(g'): correct items with r_j > og_rank
                S_mask = r > og_rank
                if not S_mask.any():
                    continue

                # ΔAP
                dap = float(np.sum(j_contrib[S_mask]))
                sigma_dap += dap

                # --- tags ---
                # overlap: centroid-pair angle between q_pid and g'_pid
                g_pid = int(g_pids[gal_g_idx])
                if q_pid in pid_idx and g_pid in pid_idx:
                    u_q = C_mat[pid_idx[q_pid]]
                    u_g = C_mat[pid_idx[g_pid]]
                    pair_angle = float(np.degrees(np.arccos(np.clip(float(u_q @ u_g), -1.0, 1.0))))
                    overlap_pct = float(np.searchsorted(all_pair_angles_sorted, pair_angle)) / len(all_pair_angles_sorted)
                else:
                    overlap_pct = 0.5

                # hub
                hub_val = float(hub_scores[gal_g_idx])
                hub_pct = float(np.searchsorted(hub_sorted, hub_val)) / len(hub_sorted)

                # cone-aligned
                cone_val = float(cone_scores[gal_g_idx])
                cone_pct = float(np.searchsorted(cone_sorted, cone_val)) / len(cone_sorted)

                # far-tail correct: mean arccos angle of outranked correct items to identity centroid
                correct_in_S = correct_positions[S_mask]   # 0-indexed positions in ranked list
                correct_g_idxs = ranked_idx[correct_in_S]
                if q_pid in pid_idx and len(correct_g_idxs) > 0:
                    u_q_cent = C_mat[pid_idx[q_pid]]
                    g_correct_emb = g_emb[correct_g_idxs]  # (n_S, d)
                    cos_gc = np.clip(g_correct_emb @ u_q_cent, -1.0, 1.0)
                    angles_gc = np.degrees(np.arccos(cos_gc))
                    mean_angle_gc = float(angles_gc.mean())
                    # Percentile in within-cell far-tail distribution — approximate using gallery-all
                    # We'll compute a global far-tail distribution per cell after
                    far_tail_raw = mean_angle_gc
                else:
                    far_tail_raw = 45.0

                # Outlier query tag: raw angle (percentile computed per-cell after)
                outlier_q_raw = angle_q_cent

                # Accumulate quintile bins (raw values; convert to quintile after)
                sample_list.append((overlap_pct, hub_pct, cone_pct, far_tail_raw, outlier_q_raw, dap, qi))

            sigma_dap_list.append(sigma_dap)

        if (start // batch) % 10 == 0:
            print(f"  Processed {min(end, N_Q)}/{N_Q} queries, collected {len(sample_list)} items",
                  flush=True)

    print(f"  Done. Total (query,item) pairs: {len(sample_list)}", flush=True)

    # Convert far_tail and outlier_q to percentiles (cell-wide distribution)
    sample_arr = np.array(sample_list, dtype=np.float32) if sample_list else np.zeros((0, 7), dtype=np.float32)

    if len(sample_arr) > 0:
        far_vals = sample_arr[:, 3]
        out_vals = sample_arr[:, 4]
        far_sorted = np.sort(far_vals)
        out_sorted = np.sort(out_vals)
        sample_arr[:, 3] = np.searchsorted(far_sorted, far_vals).astype(np.float32) / len(far_sorted)
        sample_arr[:, 4] = np.searchsorted(out_sorted, out_vals).astype(np.float32) / len(out_sorted)

        # Compute quintile bins
        total_sigma = float(sample_arr[:, 5].sum())
        quint_edges = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        for ti, tag in enumerate(TAGS):
            pcts = sample_arr[:, ti]
            for qi_bin in range(5):
                lo, hi = quint_edges[qi_bin], quint_edges[qi_bin + 1]
                in_bin = (pcts >= lo) & (pcts < hi)
                if qi_bin == 4: in_bin = (pcts >= lo) & (pcts <= hi)
                quint_bins[tag][qi_bin] = float(sample_arr[in_bin, 5].sum())
            # Normalise to share
            total = quint_bins[tag].sum()
            if total > 0:
                quint_bins[tag] /= total

    # Downsample for regression if needed
    if len(sample_arr) > REG_SAMPLE:
        idx = RNG.choice(len(sample_arr), REG_SAMPLE, replace=False)
        sample_arr = sample_arr[idx]

    return {
        'G_distribution': G_list,
        'ap_per_query': ap_list,
        'sigma_dap_per_query': sigma_dap_list,
        'quint_bins': {t: v.tolist() for t, v in quint_bins.items()},
        'sample_arr': sample_arr,   # (N_sample, 7): [overlap,hub,cone,far_tail,outlier_q,dap,q_idx]
        'total_sigma_dap': float(sum(sigma_dap_list)),
        'total_ap_deficit': float(sum(1.0 - ap for ap in ap_list)),
    }

def variance_partition(sample_arr):
    """Variance partitioning: unique and shared R^2 for each tag, holding out by query."""
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler

    if len(sample_arr) < 100:
        return None

    TAGS = ['overlap', 'hub', 'cone', 'far_tail', 'outlier_q']

    X = sample_arr[:, :5].astype(np.float64)   # 5 tags
    # Add pairwise products
    pairs = [(i, j) for i in range(5) for j in range(i+1, 5)]
    X_inter = np.column_stack([X[:, i] * X[:, j] for i, j in pairs])
    X_full = np.column_stack([X, X_inter])   # 5 + 10 = 15 features
    y = sample_arr[:, 5].astype(np.float64)  # dap

    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X_full)

    alphas = [0.01, 0.1, 1.0, 10.0, 100.0]
    model = RidgeCV(alphas=alphas, cv=5)
    model.fit(X_sc, y)
    y_pred = model.predict(X_sc)
    ss_tot = np.var(y) * len(y)
    ss_res = np.sum((y - y_pred)**2)
    r2_full = float(1.0 - ss_res / (ss_tot + 1e-12))

    # Unique contribution: R2_full - R2_full_without_tag
    unique_r2 = {}
    for ti, tag in enumerate(TAGS):
        keep = [j for j in range(15) if j != ti and (j < 5 or (pairs[j-5][0] != ti and pairs[j-5][1] != ti))]
        X_sub = X_sc[:, keep]
        m_sub = RidgeCV(alphas=alphas, cv=5)
        m_sub.fit(X_sub, y)
        yp_sub = m_sub.predict(X_sub)
        r2_sub = float(1.0 - np.sum((y - yp_sub)**2) / (ss_tot + 1e-12))
        unique_r2[tag] = float(r2_full - r2_sub)

    residual = 1.0 - r2_full

    return {'r2_full': r2_full, 'unique_r2': unique_r2, 'residual': residual,
            'n_obs': int(len(y))}

# ─── MAIN ──────────────────────────────────────────────────────────────────
print("=== E25: AP-loss attribution budget ===\n")

results = {}

for cell in ['A', 'C', 'D', 'B']:
    print(f"\n{'='*60}\nCell {cell}\n{'='*60}", flush=True)

    q_emb, q_pids, q_cids, g_emb, g_pids, g_cids = load_cell(cell)
    N_Q, N_G = len(q_emb), len(g_emb)
    print(f"  Q={N_Q}, G={N_G}", flush=True)

    # Compute centroids from gallery
    print("  Computing identity centroids...", flush=True)
    cents_gal = compute_centroids(g_emb, g_pids)
    n_gal_ids = len(cents_gal)

    # μ̂ from gallery (canonical)
    mu = g_emb.mean(0)
    mu_hat = (mu / (np.linalg.norm(mu) + 1e-12)).astype(np.float32)

    # All-pairs centroid angles
    print("  Computing all-pairs centroid angles...", flush=True)
    pair_angles_sorted, pid_list, C_mat = all_pairs_angles(cents_gal)
    print(f"  n_id_pairs={len(pair_angles_sorted):,}", flush=True)

    # Hub scores
    print("  Computing hub scores (k=10)...", flush=True)
    hub_scores = compute_hub_scores(q_emb, g_emb, g_pids, g_cids, q_pids, q_cids, k=10)
    print(f"  Hub: mean={hub_scores.mean():.1f}, max={hub_scores.max()}", flush=True)

    # Main attribution loop
    print("  Running attribution loop...", flush=True)
    res = delta_ap_and_tags(q_emb, q_pids, q_cids, g_emb, g_pids, g_cids,
                            cents_gal, mu_hat, hub_scores, pair_angles_sorted)

    ap_arr  = np.array(res['ap_per_query'])
    sig_arr = np.array(res['sigma_dap_per_query'])
    G_arr   = np.array(res['G_distribution'])

    mean_ap = float(ap_arr.mean() * 100) if len(ap_arr) else 0.0
    total_deficit = float(res['total_ap_deficit'])
    total_sigma   = float(res['total_sigma_dap'])

    print(f"\n  mAP (micro avg): {mean_ap:.2f}%", flush=True)
    print(f"  Total AP deficit Σ(1-AP): {total_deficit:.4f}", flush=True)
    print(f"  Total ΣΔAP:               {total_sigma:.4f}", flush=True)
    print(f"  Super-additivity ratio:   {total_sigma/total_deficit:.3f}x", flush=True)

    print("\n  Quintile shares of ΣΔAP per tag (Q1→Q5 = low→high percentile):")
    for tag, vals in res['quint_bins'].items():
        print(f"    {tag:12s}: {[f'{v:.3f}' for v in vals]}")

    # Variance partitioning
    print("\n  Running variance partitioning...", flush=True)
    vp = variance_partition(res['sample_arr'])
    if vp:
        print(f"  R²_full={vp['r2_full']:.4f}, residual={vp['residual']:.4f}")
        print(f"  Unique R² per tag: {vp['unique_r2']}")

    # Bootstrap CI on mean_ap and total_sigma
    if len(ap_arr) > 10:
        boot_ap    = [float(np.mean(RNG.choice(ap_arr,   len(ap_arr),   replace=True))) for _ in range(BOOT_N)]
        boot_sigma = [float(np.mean(RNG.choice(sig_arr,  len(sig_arr),  replace=True))) for _ in range(BOOT_N)]
        ci_ap    = [float(np.percentile(boot_ap,    2.5)), float(np.percentile(boot_ap,    97.5))]
        ci_sigma = [float(np.percentile(boot_sigma, 2.5)), float(np.percentile(boot_sigma, 97.5))]
    else:
        ci_ap = ci_sigma = [None, None]

    results[cell] = {
        'N_Q': N_Q, 'N_G': N_G, 'n_gal_ids': n_gal_ids,
        'mean_ap_pct': mean_ap,
        'total_ap_deficit': total_deficit,
        'total_sigma_dap':  total_sigma,
        'super_additivity': float(total_sigma / total_deficit) if total_deficit > 0 else None,
        'quint_bins': res['quint_bins'],
        'variance_partition': vp,
        'G_stats': {
            'mean': float(G_arr.mean()),
            'median': float(np.median(G_arr)),
            'p10': float(np.percentile(G_arr, 10)),
            'p90': float(np.percentile(G_arr, 90)),
        } if len(G_arr) else {},
        'boot_ci_mean_ap': ci_ap,
        'boot_ci_mean_sigma': ci_sigma,
    }

out_path = OUT / 'e25_attribution_results.json'
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2, default=lambda x: x.tolist() if hasattr(x, 'tolist') else x)
print(f"\nSaved → {out_path}")
