"""
C2 rebuild as eigen-analysis of M = C_neg - C_pos (rev 327/328).

Per arm:
  Positives: same-pid gallery pairs (standard protocol junk excluded).
  Rank-matched negatives: for each query, non-same-pid gallery items at
    ranks 1-20 (standard protocol junk excluded). Equal selection
    pressure across arms, rule independent of positive positions.
  Random negatives: K=100 uniform-sampled non-pos gallery items per
    query. Unconditioned reference.
  Hard negatives: diagnostic only; gallery items ranked above first true
    positive. Report per-query contributing counts, no cross-cell
    comparison.

Build C_pos, C_rank, C_rand, C_hard (2048 x 2048 each) via outer-product
accumulation of (q - item) differences.

M_rank = C_rank - C_pos. Spectrum of M_rank:
  - trace(M)
  - sd(eigenvalues)
  - neg mass share: sum|lam_neg| / sum|lam|
  - cumulative |lam| curve (top-200)
  - compare to gallery feature cov spectrum (cumulative variance curve)

Principal angles between top-10 eigenspaces of M_rank_id and M_rank_ood
per model pair.
"""
import json
import numpy as np
from pathlib import Path

CACHE_DIR = Path('/tmp/e12_feature_cache')
RESULTS   = Path('/data/papers/low_dimensional_reid/results')
RNG_SEED = 42
K_RAND_NEG = 100
RANK_BAND = 20   # ranks 1..20 among non-same-pid
TOP_EIG = 10     # for principal angle comparison

CELLS = {
    'A_prime': {'file': 'market_test_noREA.npz',     'n_q': 3368,  'label': "Market model → Market (id)"},
    'B_prime': {'file': 'msmt17_test_noREA.npz',     'n_q': 11659, 'label': "Market model → MSMT (ood)"},
    'C':       {'file': 'bot_msmt_seed42_market.npz','n_q': 3368,  'label': "MSMT model → Market (ood)"},
    'D':       {'file': 'bot_msmt_seed42_msmt.npz', 'n_q': 11659, 'label': "MSMT model → MSMT (id)"},
}


def l2(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(n, 1e-12)


def load_cell(info):
    d = np.load(CACHE_DIR / info['file'])
    emb = d['emb'].astype(np.float32); pids = d['pids']; cids = d['cids']
    n_q = info['n_q']
    q_emb = l2(emb[:n_q]); q_pids = pids[:n_q]; q_cids = cids[:n_q]
    g_emb = l2(emb[n_q:]); g_pids = pids[n_q:]; g_cids = cids[n_q:]
    return q_emb, q_pids, q_cids, g_emb, g_pids, g_cids


def accumulate_covariances(q_emb, q_pids, q_cids, g_emb, g_pids, g_cids, rng):
    """Returns dict of {'pos': (C, n), 'rank': ..., 'rand': ..., 'hard': ...}.
    Also tracks per-query hard-neg counts.
    C accumulates sum (q - item)(q - item)^T; later divide by n.
    """
    d = q_emb.shape[1]
    sums = {k: np.zeros((d, d), dtype=np.float64) for k in ['pos', 'rank', 'rand', 'hard']}
    counts = {k: 0 for k in ['pos', 'rank', 'rand', 'hard']}
    hard_per_query = []

    for i in range(len(q_pids)):
        if i % 2000 == 0:
            print(f"    query {i}/{len(q_pids)}  (counts: pos={counts['pos']} rank={counts['rank']} rand={counts['rand']} hard={counts['hard']})", flush=True)
        qp = q_pids[i]; qc = q_cids[i]
        q = q_emb[i]
        junk = (g_pids == qp) & (g_cids == qc)
        pos_mask = (g_pids == qp) & ~junk

        # Positives
        if pos_mask.any():
            pos_items = g_emb[pos_mask]
            diffs = q - pos_items   # (n_pos_i, d)
            sums['pos'] += diffs.T @ diffs
            counts['pos'] += len(pos_items)

        # Need to rank non-same-pid items for both rank-matched and hard
        sims = q @ g_emb.T
        sims_masked = sims.copy()
        sims_masked[junk] = -np.inf
        order_full = np.argsort(-sims_masked)
        # Build is_pos sequence along order
        is_pos = g_pids[order_full] == qp

        # Rank-matched negatives: top RANK_BAND non-same-pid, by sim
        neg_order_mask = ~is_pos & (sims_masked[order_full] > -np.inf)
        neg_order_idx = order_full[neg_order_mask]  # ordered indices of non-pos, non-junk
        rank_neg_idx = neg_order_idx[:RANK_BAND]
        if len(rank_neg_idx) > 0:
            rn_items = g_emb[rank_neg_idx]
            diffs = q - rn_items
            sums['rank'] += diffs.T @ diffs
            counts['rank'] += len(rank_neg_idx)

        # Random negatives
        neg_mask = (g_pids != qp) & ~junk
        all_neg_idx = np.where(neg_mask)[0]
        if len(all_neg_idx) > 0:
            K = min(K_RAND_NEG, len(all_neg_idx))
            sel = rng.choice(len(all_neg_idx), size=K, replace=False)
            rand_items = g_emb[all_neg_idx[sel]]
            diffs = q - rand_items
            sums['rand'] += diffs.T @ diffs
            counts['rand'] += K

        # Hard negatives: gallery items ranked above first true positive
        if is_pos.any():
            first_pos = int(np.argmax(is_pos))
            hn = first_pos  # number of hard negatives for this query
            hard_per_query.append(hn)
            if hn > 0:
                hard_idx = order_full[:first_pos]
                hard_items = g_emb[hard_idx]
                diffs = q - hard_items
                sums['hard'] += diffs.T @ diffs
                counts['hard'] += hn
        else:
            hard_per_query.append(0)

    # Convert to covariances
    covs = {}
    for k in ['pos', 'rank', 'rand', 'hard']:
        n = counts[k]
        if n > 0:
            covs[k] = sums[k] / n
        else:
            covs[k] = None
    return covs, counts, np.array(hard_per_query)


def spectrum_of_M(M):
    """Eigendecomposition of symmetric M. Return eigs sorted by |lam|, and
    the eigenvectors in that order. Also provide signed eigenvalues."""
    w, V = np.linalg.eigh(M.astype(np.float64))
    # Sort by |lam| descending
    order = np.argsort(-np.abs(w))
    w = w[order]; V = V[:, order]
    return w, V


def principal_angles_eigenspaces(V1, V2, K):
    """Principal angles between top-K columns of V1 and V2."""
    B1 = V1[:, :K]; B2 = V2[:, :K]
    s = np.linalg.svd(B1.T @ B2, compute_uv=False)
    s = np.clip(s, -1.0, 1.0)
    return np.degrees(np.arccos(s))


def process_cell(tag, info):
    print(f"\n### Cell {tag} ({info['label']}) ###", flush=True)
    q_emb, q_pids, q_cids, g_emb, g_pids, g_cids = load_cell(info)
    rng = np.random.default_rng(RNG_SEED)
    print(f"  computing covariances...", flush=True)
    covs, counts, hard_per_query = accumulate_covariances(
        q_emb, q_pids, q_cids, g_emb, g_pids, g_cids, rng)
    print(f"  final counts: pos={counts['pos']}  rank={counts['rank']}  rand={counts['rand']}  hard={counts['hard']}", flush=True)
    n_queries_with_hard = int((hard_per_query > 0).sum())
    print(f"  queries contributing to hard: {n_queries_with_hard}/{len(hard_per_query)}  mean hard-per-query={hard_per_query.mean():.2f}  median={int(np.median(hard_per_query))}  max={int(hard_per_query.max())}", flush=True)

    # Build M matrices
    results = {
        'cell': tag, 'label': info['label'],
        'counts': counts,
        'n_queries_with_hard': n_queries_with_hard,
        'hard_per_query_stats': {
            'mean': float(hard_per_query.mean()),
            'median': int(np.median(hard_per_query)),
            'max': int(hard_per_query.max()),
            'frac_nonzero': float((hard_per_query > 0).mean()),
        },
    }

    # Also gallery feature covariance for variance-share baseline
    mu_g = g_emb.mean(0)
    Xc = g_emb - mu_g
    C_feat = (Xc.T @ Xc) / (len(g_emb) - 1)
    eigs_feat, _ = spectrum_of_M(C_feat)
    # Cumulative variance share from eigs_feat (sorted by |lam|, and they're all positive)
    cum_var = np.cumsum(np.abs(eigs_feat)) / np.abs(eigs_feat).sum()
    results['gallery_var_cum_top_200'] = cum_var[:200].tolist()
    results['gallery_var_total'] = float(np.abs(eigs_feat).sum())

    for neg_name in ['rank', 'rand', 'hard']:
        if covs[neg_name] is None:
            continue
        M = covs[neg_name] - covs['pos']
        eigs, V = spectrum_of_M(M)
        tr = float(eigs.sum())
        sd_eig = float(eigs.std(ddof=1))
        neg_mass = float(np.sum(np.abs(eigs[eigs < 0])) / np.sum(np.abs(eigs)))
        pos_mass = float(np.sum(np.abs(eigs[eigs > 0])) / np.sum(np.abs(eigs)))
        # Cumulative |lam| share, top-200 by |lam|
        cum = np.cumsum(np.abs(eigs)) / np.abs(eigs).sum()
        print(f"  M_{neg_name}: trace={tr:+.5f}  sd_eig={sd_eig:.5f}  neg_mass_share={neg_mass:.4f}  top-1 |lam|={abs(eigs[0]):.5f}", flush=True)
        results[f'M_{neg_name}'] = {
            'trace': tr, 'sd_eig': sd_eig,
            'neg_mass_share': neg_mass, 'pos_mass_share': pos_mass,
            'eigs_signed_top_200': eigs[:200].tolist(),
            'cumulative_abs_lam_top_200': cum[:200].tolist(),
            'top_K_eigvecs': V[:, :TOP_EIG].tolist(),  # for principal angles later
        }
    return results


def main():
    out = {}
    for tag, info in CELLS.items():
        out[tag] = process_cell(tag, info)

    # Principal angles between top-K eigenspaces of M_rank id vs ood per model
    print("\n\n=== Principal angles M_rank_id vs M_rank_ood (top-K) ===")
    pa_results = {}
    for label, id_tag, ood_tag in [('Market_model', 'A_prime', 'B_prime'), ('MSMT_model', 'D', 'C')]:
        if 'M_rank' not in out[id_tag] or 'M_rank' not in out[ood_tag]:
            continue
        V_id = np.array(out[id_tag]['M_rank']['top_K_eigvecs'])
        V_ood = np.array(out[ood_tag]['M_rank']['top_K_eigvecs'])
        angles = principal_angles_eigenspaces(V_id, V_ood, TOP_EIG)
        pa_results[label] = {
            'all_angles_deg': angles.tolist(),
            'mean_deg': float(angles.mean()),
            'median_deg': float(np.median(angles)),
        }
        print(f"  {label} M_rank: top-{TOP_EIG} principal angles = {np.round(angles, 2).tolist()}")
        print(f"    mean = {angles.mean():.2f}  median = {np.median(angles):.2f}", flush=True)

    out['principal_angles_M_rank_top_K'] = pa_results

    # Trim very large eigvec arrays from output
    for tag in CELLS:
        for key in list(out[tag].keys()):
            if key.startswith('M_') and isinstance(out[tag][key], dict) and 'top_K_eigvecs' in out[tag][key]:
                out[tag][key].pop('top_K_eigvecs')  # too big for JSON
    out_path = RESULTS / 'e43_C2_eigen_rebuild.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {out_path}")

    # Summary
    print("\n\n========= C2 EIGEN REBUILD SUMMARY =========")
    print("\nM_rank (rank-matched negatives, ranks 1-20 non-same-pid):")
    print(f"{'cell':9s}  {'trace':>9s}  {'sd_eig':>9s}  {'neg_mass':>9s}  {'cum|lam| top-10':>17s}  {'top-50':>8s}  {'top-200':>8s}")
    for tag in CELLS:
        r = out[tag]['M_rank']
        cum = r['cumulative_abs_lam_top_200']
        print(f"{tag:9s}  {r['trace']:+9.5f}  {r['sd_eig']:9.5f}  {r['neg_mass_share']:9.4f}  {cum[9]:17.4f}  {cum[49]:8.4f}  {cum[199]:8.4f}")
    print("\nM_rand (random negatives):")
    for tag in CELLS:
        r = out[tag]['M_rand']
        cum = r['cumulative_abs_lam_top_200']
        print(f"{tag:9s}  trace={r['trace']:+.5f}  sd_eig={r['sd_eig']:.5f}  neg_mass={r['neg_mass_share']:.4f}  cum|lam|top10={cum[9]:.4f}  top200={cum[199]:.4f}")

    print(f"\nPrincipal angles M_rank top-{TOP_EIG}:")
    for label, pa in pa_results.items():
        print(f"  {label}: mean={pa['mean_deg']:.2f}  median={pa['median_deg']:.2f}  angles={[f'{a:.1f}' for a in pa['all_angles_deg']]}")


if __name__ == '__main__':
    main()
