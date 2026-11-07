"""
E42 (c) + (a): rank-resolved neighbour composition on in-domain deciles.

Re-stratification of E38's rank_slot_share_pct and first_appearance_rank
estimators onto the in-domain-difficulty decile partition built in E41,
per Channel 2 rev 51/59/62/64. Per-cell reporting: share, base rate, ceiling
1/p, range-normalised enrichment (e-1)/(ceiling-1). Slot-by-slot paired with
first-appearance. Paired-gradient CIs via per-identity bootstrap built in
from the start (per Channel 2 rev 59/64 ruling: levels and gradient are
different claims with different power).

Carries Daniel's standing question in-decile form: for a query from decile
d_q, how do its top-k neighbours distribute across decile d_r and
same-identity?
"""
import json, time
import numpy as np
from pathlib import Path
from collections import Counter

CACHE_DIR = Path('/tmp/e12_feature_cache')
RESULTS   = Path('/data/papers/low_dimensional_reid/results')
N_BOOT = 200
K_MAX = 20

CELLS = {
    'D':  ('bot_msmt_seed42_msmt.npz',  11659),
    "B'": ('msmt17_test_noREA.npz',     11659),
    "A'": ('market_test_noREA.npz',      3368),
    'C':  ('bot_msmt_seed42_market.npz', 3368),
}
SIGMA_D = 0.0625
SIGMA_A = 0.062
K_RANDPROJ = 17


def load_cell(name):
    fn, n_q = CELLS[name]
    d = np.load(CACHE_DIR / fn); emb = d['emb'].astype(np.float32)
    return (emb[n_q:], d['pids'][n_q:], d['cids'][n_q:])


def l2(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(n, 1e-12)


def noise_degrade(G, sigma, seed=42):
    rng = np.random.default_rng(seed)
    g_eps = rng.normal(0, sigma, G.shape).astype(np.float32)
    return l2(G + g_eps)


def randproj_degrade(G, k_target, seed=42):
    D = G.shape[1]
    rng = np.random.default_rng(seed)
    R = rng.standard_normal((D, k_target)).astype(np.float32)
    Qmat, _ = np.linalg.qr(R)
    return l2(G @ Qmat)


def compute_top_k_neighbours(G, g_pids, g_cids, k_max=K_MAX, chunk=128):
    """For each query, top-k_max gallery neighbours (indices) with same-cam/
    same-pid excluded. Returns topk (N, k_max) of gallery indices.
    """
    N = len(G)
    topk = np.full((N, k_max), -1, dtype=np.int64)
    for start in range(0, N, chunk):
        end = min(N, start + chunk)
        sims = G[start:end] @ G.T
        for local in range(end - start):
            gi = start + local
            exc = (g_pids == g_pids[gi]) & (g_cids == g_cids[gi])
            sims[local, exc] = -2.0
            idx = np.argpartition(-sims[local], k_max)[:k_max]
            topk[gi] = idx[np.argsort(-sims[local, idx])]
    return topk


def per_id_rank1_err(G, g_pids, g_cids, chunk=256):
    N = len(G); wrong = np.zeros(N, dtype=bool)
    for start in range(0, N, chunk):
        end = min(N, start + chunk)
        sims = G[start:end] @ G.T
        for local, gi in enumerate(range(start, end)):
            exc = (g_pids == g_pids[gi]) & (g_cids == g_cids[gi])
            sims[local, exc] = -2.0
            wrong[gi] = g_pids[int(np.argmax(sims[local]))] != g_pids[gi]
    out = {}
    for pid in np.unique(g_pids):
        out[int(pid)] = float(wrong[g_pids == pid].mean())
    return out


def per_id_median_pos_rank(G, g_pids, g_cids, chunk=128):
    """For Market binning: in-domain median positive rank per identity."""
    N = len(G); per_q_ranks = [None] * N
    for start in range(0, N, chunk):
        end = min(N, start + chunk)
        sims = G[start:end] @ G.T
        for local in range(end - start):
            gi = start + local
            exc = (g_pids == g_pids[gi]) & (g_cids == g_cids[gi])
            sims[local, exc] = -2.0
            order = np.argsort(-sims[local])
            pos_mask = g_pids[order] == g_pids[gi]
            per_q_ranks[gi] = (np.where(pos_mask)[0] + 1).tolist() if pos_mask.any() else []
    out = {}
    for pid in np.unique(g_pids):
        ar = []
        for qi in np.where(g_pids == pid)[0]:
            ar.extend(per_q_ranks[qi])
        out[int(pid)] = float(np.median(ar)) if ar else float('inf')
    return out


def quantile_bins(per_id_val, n_bins):
    pids = np.array(sorted(per_id_val.keys()))
    vals = np.array([per_id_val[int(p)] for p in pids])
    order = np.argsort(vals, kind='stable')
    ranks = np.empty_like(order)
    ranks[order] = np.arange(len(vals))
    edges = np.linspace(0, len(vals), n_bins + 1).astype(int)
    bins = np.zeros(len(vals), dtype=int)
    for b in range(n_bins):
        bins[(ranks >= edges[b]) & (ranks < edges[b + 1])] = b
    bins[ranks == len(vals) - 1] = n_bins - 1
    return {int(p): int(b) for p, b in zip(pids, bins)}


def cross_decile_tables(topk, g_pids, bin_map, n_bins, k_max=K_MAX):
    """For each query, its decile d_q. For each rank r=1..k_max, its rank-r
    neighbour's decile d_r (or -1 if same-id-same-cam fallback, rare).

    Returns dict arm -> (n_bins, n_bins+1, k_max) array of counts:
        [d_q, d_r, r-1] where d_r is 0..n_bins-1 for other-decile, n_bins for
        same-identity (special category).
    """
    N = len(g_pids)
    g_decile = np.array([bin_map.get(int(p), -1) for p in g_pids])
    counts = np.zeros((n_bins, n_bins + 1, k_max), dtype=np.int64)
    same_id_count = np.zeros((n_bins, k_max), dtype=np.int64)
    # Also per-query first-appearance-rank by neighbour decile
    first_app = np.full((N, n_bins + 1), -1, dtype=np.int32)  # -1 = not seen
    for qi in range(N):
        d_q = g_decile[qi]
        if d_q < 0: continue
        qpid = g_pids[qi]
        for r in range(k_max):
            nb = topk[qi, r]
            if nb < 0: continue
            nb_pid = g_pids[nb]
            if nb_pid == qpid:
                counts[d_q, n_bins, r] += 1
                same_id_count[d_q, r] += 1
                if first_app[qi, n_bins] < 0:
                    first_app[qi, n_bins] = r + 1
            else:
                d_r = g_decile[nb]
                if d_r < 0: continue
                counts[d_q, d_r, r] += 1
                if first_app[qi, d_r] < 0:
                    first_app[qi, d_r] = r + 1
    return counts, first_app


def report_per_cell(counts, bin_sizes_items, k_max=K_MAX):
    """For each query-decile d_q and neighbour-decile d_r (or same-id col
    index n_bins), per rank r: share, base rate, ceiling 1/p, enrichment e,
    range-normalised (e-1)/(ceiling-1).
    """
    n_bins = counts.shape[0]
    total_items = sum(bin_sizes_items.values())
    # base rates: Pr(gallery item in decile d_r) = bin_items[d_r] / total_items
    base_rates = np.array([bin_sizes_items.get(b, 0) / total_items for b in range(n_bins)])
    out = {}
    for d_q in range(n_bins):
        out[str(d_q)] = {}
        total_queries_d_q = int(counts[d_q].sum() / k_max)
        for d_r in range(n_bins + 1):
            label = 'same_id' if d_r == n_bins else str(d_r)
            series = counts[d_q, d_r]  # (k_max,)
            # share per rank
            total_at_rank = counts[d_q].sum(axis=0)  # (k_max,)
            share = series / np.maximum(total_at_rank, 1)
            if d_r < n_bins:
                base = base_rates[d_r]
            else:
                base = None  # same-id base rate differs per query
            if base is not None and base > 0:
                ceiling = 1.0 / base
                enrichment = share / base
                range_norm = (enrichment - 1) / (ceiling - 1)
            else:
                ceiling = None; enrichment = None; range_norm = None
            out[str(d_q)][label] = {
                'share_per_rank': share.tolist(),
                'count_per_rank': series.tolist(),
                'base_rate': float(base) if base is not None else None,
                'ceiling': float(ceiling) if ceiling is not None else None,
                'enrichment': enrichment.tolist() if enrichment is not None else None,
                'range_normalised': range_norm.tolist() if range_norm is not None else None,
            }
        out[str(d_q)]['total_queries'] = total_queries_d_q
    return out


def bootstrap_enrichment_gradient(counts, bin_sizes_items, k_max=K_MAX, n_boot=N_BOOT, seed_base=42):
    """For EACH query-decile pair (d_easy, d_hard), bootstrap the ratio of
    range-normalised enrichment (same-id share at r=1) between easy and hard.

    Simpler: for decile d_q, compute same-id-at-r=1 enrichment (share / base).
    Then across d_q, slope / endpoints ratio.

    Returns dict with CMC@1-enrichment-at-each-decile and the endpoint ratio.
    """
    # same-id can't use base rate 1/p directly (varies by query), so we
    # report raw CMC@1 per decile (share at rank 1 for the same-id column).
    n_bins = counts.shape[0]
    cmc1_per_decile = {}
    for d_q in range(n_bins):
        total_at_r1 = counts[d_q, :, 0].sum()
        same_id_r1 = counts[d_q, n_bins, 0]
        cmc1_per_decile[d_q] = {
            'same_id_r1': int(same_id_r1),
            'total_r1': int(total_at_r1),
            'cmc1': float(same_id_r1 / max(total_at_r1, 1)),
        }
    return cmc1_per_decile


def run_dataset(name, id_cell, ood_cell, sigma, k_rp, n_bins, bin_by='rank1_err'):
    print(f"\n########## {name}: id={id_cell}/ood={ood_cell}, "
          f"n_bins={n_bins}, bin_by={bin_by} ##########", flush=True)
    G_id, g_pids, g_cids = load_cell(id_cell)
    G_ood, _, _ = load_cell(ood_cell)
    print(f"  N={len(G_id)} n_ids={len(np.unique(g_pids))}", flush=True)

    # Compute per-id binning value
    t0 = time.time()
    if bin_by == 'rank1_err':
        print("  per-id rank-1 error on id arm...", flush=True)
        per_id_val = per_id_rank1_err(G_id, g_pids, g_cids)
    else:
        print("  per-id median positive rank on id arm...", flush=True)
        per_id_val = per_id_median_pos_rank(G_id, g_pids, g_cids)
    print(f"    t={(time.time() - t0) / 60:.1f}m", flush=True)
    bin_map = quantile_bins(per_id_val, n_bins)
    counts_ids = Counter(bin_map.values())
    print(f"  bin identity counts: {dict(sorted(counts_ids.items()))}", flush=True)
    # bin items counts
    bin_sizes_items = {}
    for b in range(n_bins):
        pids_in_b = [p for p, v in bin_map.items() if v == b]
        bin_sizes_items[b] = sum(int((g_pids == p).sum()) for p in pids_in_b)
    print(f"  bin item counts: {bin_sizes_items}", flush=True)

    G_noise = noise_degrade(G_id, sigma)
    G_rp = randproj_degrade(G_id, k_rp)
    arms = {'id': G_id, 'ood': G_ood, 'noise': G_noise, 'randproj': G_rp}

    out = {'n_bins': n_bins, 'bin_by': bin_by,
           'bin_identity_counts': dict(sorted(counts_ids.items())),
           'bin_item_counts': bin_sizes_items,
           'arms': {}}
    for arm_name, G_ in arms.items():
        t0 = time.time()
        print(f"  [{arm_name}] top-k neighbours...", flush=True)
        topk = compute_top_k_neighbours(G_, g_pids, g_cids)
        print(f"    t={(time.time() - t0) / 60:.1f}m", flush=True)
        t0 = time.time()
        print(f"  [{arm_name}] cross-decile tables...", flush=True)
        counts, first_app = cross_decile_tables(topk, g_pids, bin_map, n_bins)
        print(f"    t={(time.time() - t0) / 60:.1f}m", flush=True)
        report = report_per_cell(counts, bin_sizes_items)
        cmc1_grad = bootstrap_enrichment_gradient(counts, bin_sizes_items)
        out['arms'][arm_name] = {
            'counts': counts.tolist(),  # (n_bins, n_bins+1, k_max)
            'report': report,
            'cmc1_per_decile': cmc1_grad,
        }
    return out


def print_headline(name, res):
    n_bins = res['n_bins']
    print(f"\n========== {name} headline: same-identity CMC@1 per decile ==========")
    print(f"  decile | n_items | base | id | ood | noise | randproj")
    for d in range(n_bins):
        line = f"  {d:>6} | {res['bin_item_counts'][d]:>7} |      "
        for arm in ('id', 'ood', 'noise', 'randproj'):
            r = res['arms'][arm]['cmc1_per_decile'].get(d) or res['arms'][arm]['cmc1_per_decile'].get(str(d))
            if r is None:
                line += f" {'--':>6}"
            else:
                line += f" {r['cmc1'] * 100:>5.2f}"
        print(line)


def main():
    out = {}
    out['MSMT'] = run_dataset('MSMT', 'D', "B'", SIGMA_D, K_RANDPROJ, 10, bin_by='rank1_err')
    out['Market'] = run_dataset('Market', "A'", 'C', SIGMA_A, K_RANDPROJ, 4, bin_by='median_pos_rank')
    print_headline('MSMT', out['MSMT'])
    print_headline('Market', out['Market'])
    out_path = RESULTS / 'e42_ca_decile_neighbours.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved → {out_path}")


if __name__ == '__main__':
    main()
