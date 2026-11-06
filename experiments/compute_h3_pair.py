"""
H3 pair-level — spread imbalance for reciprocal vs one-directional confused pairs.

Spec (Channel rev 227):
  For every confused pair (i,j) in B' and in D:
    - centroid angle    ang_ij  = arccos(<μ_i, μ_j>) (μ L2-normed)
    - pair margin       marg_ij = ang_ij / (spread_i + spread_j)
    - spread imbalance  imb_ij  = max(s_i, s_j) / min(s_i, s_j)
  Split pairs by reciprocal vs one-directional.
  Primary: within-cell reciprocal-vs-one-directional comparison (selection-
  consistent); secondary: B' vs D levels (selected-on, not load-bearing).

Noise null attached: same pair statistics on noise-degraded-D at matched mAP
(σ=0.0625, from E32).

Spread per identity i: mean angle of i's items to μ_i on the unit sphere,
  spread_i = mean_j arccos(<x_ij, μ_i>).
Confused pair (i,j): j appears in top-10 gallery items of some i-query, with
  pid(gallery_item) = j ≠ i. This is the "confuser edge" at k=10 we've used.
Reciprocal pair: i→j AND j→i both appear as confuser edges.
One-directional: exactly one direction.
"""
import json, time
import numpy as np
from pathlib import Path
from collections import defaultdict

CACHE_DIR = Path('/tmp/e12_feature_cache')
RESULTS   = Path('/data/papers/low_dimensional_reid/results')
K_CONF    = 10
NOISE_SIGMA_D = 0.0625  # matched mAP for MSMT


def load_cell(cell):
    files = {
        "A'": ('market_test_noREA.npz', 3368),
        "B'": ('msmt17_test_noREA.npz', 11659),
        'D':  ('bot_msmt_seed42_msmt.npz', 11659),
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


def per_id_centroid_spread(X, pids):
    """Return μ (L2-normed) and spread (mean arccos to μ) per unique pid."""
    ids = np.unique(pids)
    mu = np.zeros((len(ids), X.shape[1]), np.float32)
    spread = np.zeros(len(ids), np.float32)
    for k, pid in enumerate(ids):
        mask = pids == pid
        xi = X[mask]
        mu_k = xi.mean(axis=0)
        mu_k /= max(np.linalg.norm(mu_k), 1e-12)
        mu[k] = mu_k
        cos_x = np.clip(xi @ mu_k, -1.0, 1.0)
        spread[k] = np.arccos(cos_x).mean()
    return ids, mu, spread


def topk_confusers(Q, qp, qc, G, gp, gc, k=10):
    """Return dict id_i -> Counter(id_j -> #hits) of top-k confuser pids."""
    # same-id/same-cam filter applied per query
    sim = Q @ G.T  # (nQ, nG)
    confusers = defaultdict(lambda: defaultdict(int))
    for qi in range(len(qp)):
        qid, qcid = qp[qi], qc[qi]
        valid = ~((gp == qid) & (gc == qcid))
        s = sim[qi].copy()
        s[~valid] = -np.inf
        top_idx = np.argpartition(-s, k)[:k]
        top_idx = top_idx[np.argsort(-s[top_idx])]
        for gi in top_idx:
            j = gp[gi]
            if j == qid:
                continue
            confusers[qid][j] += 1
    return {i: dict(d) for i, d in confusers.items()}


def extract_pairs(confusers, min_hits=1):
    """Return reciprocal and one-directional pair sets.
    i→j edge: j appears in i's top-k with >= min_hits.
    """
    edges = set()
    for i, cnts in confusers.items():
        for j, c in cnts.items():
            if c >= min_hits:
                edges.add((int(i), int(j)))
    pairs = set()
    for (i, j) in edges:
        pairs.add((min(i, j), max(i, j)))
    recip = {(i, j) for (i, j) in pairs if (i, j) in edges and (j, i) in edges}
    one_dir = pairs - recip
    return edges, recip, one_dir


def pair_stats(pairs, id_to_idx, mu, spread):
    """Compute per-pair (ang, margin, imbalance)."""
    out = []
    for (i, j) in pairs:
        if i not in id_to_idx or j not in id_to_idx:
            continue
        ki, kj = id_to_idx[i], id_to_idx[j]
        cos_ij = float(np.clip(mu[ki] @ mu[kj], -1.0, 1.0))
        ang = float(np.arccos(cos_ij))
        si, sj = float(spread[ki]), float(spread[kj])
        marg = ang / max(si + sj, 1e-12)
        imb = max(si, sj) / max(min(si, sj), 1e-12)
        out.append({'i': i, 'j': j, 'ang': ang, 'margin': marg, 'imbalance': imb})
    return out


def summarize(label, stats):
    if not stats:
        return {'label': label, 'n': 0}
    imb = np.array([s['imbalance'] for s in stats])
    marg = np.array([s['margin'] for s in stats])
    ang = np.array([s['ang'] for s in stats])
    return {'label': label, 'n': len(stats),
            'imbalance_mean': float(imb.mean()),
            'imbalance_median': float(np.median(imb)),
            'imbalance_p25': float(np.percentile(imb, 25)),
            'imbalance_p75': float(np.percentile(imb, 75)),
            'margin_mean': float(marg.mean()),
            'margin_median': float(np.median(marg)),
            'ang_mean': float(ang.mean()),
            'ang_median': float(np.median(ang))}


def bootstrap_diff(recip_vals, one_vals, n_boot=2000, seed=42):
    """Bootstrap CI on mean(one) - mean(recip)."""
    rng = np.random.default_rng(seed)
    nr, no = len(recip_vals), len(one_vals)
    rvals = np.asarray(recip_vals)
    ovals = np.asarray(one_vals)
    diffs = np.zeros(n_boot)
    for b in range(n_boot):
        ri = rng.integers(0, nr, nr)
        oi = rng.integers(0, no, no)
        diffs[b] = ovals[oi].mean() - rvals[ri].mean()
    return {'mean': float(ovals.mean() - rvals.mean()),
            'CI95': [float(np.percentile(diffs, 2.5)),
                     float(np.percentile(diffs, 97.5))]}


def run_cell(cell_name, Q, qp, qc, G, gp, gc):
    print(f"\n=== {cell_name}: pair statistics ===", flush=True)
    t0 = time.time()
    print(f"  centroids+spread...", flush=True)
    # Use gallery to build id features (quieter than mixing)
    ids, mu, spread = per_id_centroid_spread(G, gp)
    id_to_idx = {int(p): k for k, p in enumerate(ids)}
    print(f"    n_ids={len(ids)}  t={(time.time()-t0)/60:.1f}m", flush=True)

    t1 = time.time()
    print(f"  top-{K_CONF} confusers...", flush=True)
    conf = topk_confusers(Q, qp, qc, G, gp, gc, k=K_CONF)
    print(f"    n_query_ids={len(conf)}  t={(time.time()-t1)/60:.1f}m", flush=True)

    edges, recip, one_dir = extract_pairs(conf)
    print(f"  edges: {len(edges)}  reciprocal pairs: {len(recip)}  "
          f"one-directional: {len(one_dir)}", flush=True)

    recip_stats = pair_stats(recip, id_to_idx, mu, spread)
    one_stats = pair_stats(one_dir, id_to_idx, mu, spread)

    s_r = summarize(f"{cell_name}:reciprocal", recip_stats)
    s_o = summarize(f"{cell_name}:one_directional", one_stats)

    diff_tests = {}
    if recip_stats and one_stats:
        for key in ('imbalance', 'margin', 'ang'):
            rvals = [s[key] for s in recip_stats]
            ovals = [s[key] for s in one_stats]
            diff_tests[key] = bootstrap_diff(rvals, ovals)

    return {'cell': cell_name,
            'n_edges': len(edges),
            'n_reciprocal': len(recip),
            'n_one_directional': len(one_dir),
            'recip_summary': s_r,
            'one_dir_summary': s_o,
            'one_minus_recip_diffs': diff_tests}


def main():
    results = {'params': {'k': K_CONF, 'noise_sigma_D': NOISE_SIGMA_D}}

    print("\n########## B' (Market → MSMT, real transfer) ##########", flush=True)
    q, qp, qc, G, gp, gc = load_cell("B'")
    results["B_prime"] = run_cell("B'", q, qp, qc, G, gp, gc)
    del q, qp, qc, G, gp, gc

    print("\n########## D (MSMT → MSMT, in-domain) ##########", flush=True)
    q, qp, qc, G, gp, gc = load_cell('D')
    results['D'] = run_cell('D', q, qp, qc, G, gp, gc)

    print("\n########## noise-D at σ=0.0625 (matched to B' mAP) ##########", flush=True)
    Gn, Qn = noise_degrade(G, q, NOISE_SIGMA_D)
    results['noise_D'] = run_cell('noise-D', Qn, qp, qc, Gn, gp, gc)
    del q, qp, qc, G, gp, gc, Gn, Qn

    out = RESULTS / 'h3_pair_level.json'
    with open(out, 'w') as f: json.dump(results, f, indent=2)
    print(f"\nSaved → {out}")

    # Short printed summary
    print("\n=== Within-cell (one-directional minus reciprocal) ===")
    print(f"{'cell':<12} {'n_recip':>8} {'n_onedir':>10} {'Δimb mean':>12} {'Δimb 95CI':>24} {'Δmargin mean':>14}")
    for cell_key in ("B_prime", 'D', 'noise_D'):
        c = results[cell_key]
        d = c['one_minus_recip_diffs']
        if not d:
            print(f"{cell_key:<12} (insufficient data)")
            continue
        imb_d = d['imbalance']
        mg_d = d['margin']
        print(f"{cell_key:<12} {c['n_reciprocal']:>8} {c['n_one_directional']:>10} "
              f"{imb_d['mean']:>+12.4f} [{imb_d['CI95'][0]:+.4f},{imb_d['CI95'][1]:+.4f}] "
              f"{mg_d['mean']:>+14.4f}")


if __name__ == '__main__':
    main()
