"""
Freeze the in-domain-difficulty partition used by E41 / bootstrap / E42(c)
per claude-chat Channel 2 rev 74/76.

Rule, canonical going forward:
  - MSMT: bin identities by in-domain (D cell) rank-1 error, equal-count
    deciles via stable argsort (numpy.argsort kind='stable'), 10 bins.
  - Market: bin identities by in-domain (A' cell) median positive rank,
    equal-count quartiles via stable argsort, 4 bins.
  - Bin indices 0..n_bins-1, ordered easy->hard (ascending difficulty).
  - Edge values reported as per-bin cut points for reference, not for
    digitize-based re-derivation. The authoritative assignment is the
    {pid: bin} map in this file.

Output: results/e41_partition_frozen.json, loaded by all downstream
decile-stratified experiments. Any script that needs the partition
reads it rather than re-deriving it.
"""
import json
import numpy as np
from pathlib import Path

CACHE_DIR = Path('/tmp/e12_feature_cache')
RESULTS   = Path('/data/papers/low_dimensional_reid/results')

CELLS = {
    'D':  ('bot_msmt_seed42_msmt.npz',  11659),
    "A'": ('market_test_noREA.npz',      3368),
}
MSMT_N_BINS = 10
MARKET_N_BINS = 4


def load_cell_gallery(name):
    fn, n_q = CELLS[name]
    d = np.load(CACHE_DIR / fn); emb = d['emb'].astype(np.float32)
    return emb[n_q:], d['pids'][n_q:], d['cids'][n_q:]


def per_id_rank1_err(G, g_pids, g_cids, chunk=256):
    """Per-identity rank-1 error, using same-cam/same-pid exclusion on queries
    drawn from the gallery. Sum over the identity's queries, divide by count.
    """
    N = len(G)
    wrong = np.zeros(N, dtype=np.int8)
    for start in range(0, N, chunk):
        end = min(N, start + chunk)
        sims = G[start:end] @ G.T
        for local, gi in enumerate(range(start, end)):
            exc = (g_pids == g_pids[gi]) & (g_cids == g_cids[gi])
            sims[local, exc] = -2.0
            top1 = int(np.argmax(sims[local]))
            wrong[gi] = int(g_pids[top1] != g_pids[gi])
    out = {}
    for pid in np.unique(g_pids):
        m = g_pids == pid
        out[int(pid)] = float(wrong[m].mean())
    return out


def per_id_median_pos_rank(G, g_pids, g_cids, chunk=128):
    """Per-identity median rank of all positives across the identity's queries.
    """
    N = len(G)
    per_q_ranks = [None] * N
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


def equal_count_bins(per_id_val, n_bins):
    """CANONICAL: stable-argsort rank-based equal-count bins. Returns
    {pid: bin} where bin ranges 0..n_bins-1, bin edges (value cut points),
    and per-bin counts.
    """
    pids = np.array(sorted(per_id_val.keys()))
    vals = np.array([per_id_val[int(p)] for p in pids])
    order = np.argsort(vals, kind='stable')
    ranks = np.empty_like(order)
    ranks[order] = np.arange(len(vals))
    n = len(vals)
    edges_idx = np.linspace(0, n, n_bins + 1).astype(int)
    bins = np.zeros(n, dtype=int)
    for b in range(n_bins):
        bins[(ranks >= edges_idx[b]) & (ranks < edges_idx[b + 1])] = b
    bins[ranks == n - 1] = n_bins - 1
    sorted_vals = vals[order]
    value_edges = [float(sorted_vals[min(edges_idx[i], n - 1)]) for i in range(n_bins + 1)]
    counts = {b: int((bins == b).sum()) for b in range(n_bins)}
    return (
        {int(p): int(b) for p, b in zip(pids, bins)},
        value_edges,
        counts,
    )


def main():
    out = {
        'rule': ('equal-count bins via stable argsort of per-identity '
                 'in-domain difficulty variable, ordered easy->hard'),
        'tie_breaking': 'numpy.argsort(kind="stable"), pid-ascending within ties',
        'MSMT': {}, 'Market': {},
    }

    # MSMT: 10 deciles by in-domain (D) rank-1 error
    print("MSMT: computing per-id rank-1 error on D...", flush=True)
    G, pids, cids = load_cell_gallery('D')
    per_id_val = per_id_rank1_err(G, pids, cids)
    assign, edges, counts = equal_count_bins(per_id_val, MSMT_N_BINS)
    out['MSMT'] = {
        'cell': 'D',
        'variable': 'in-domain rank-1 error per identity',
        'n_bins': MSMT_N_BINS,
        'edges_value': edges,
        'bin_counts': counts,
        'assignment': assign,
        'n_ids': len(assign),
    }
    print(f"  MSMT bin counts: {counts}", flush=True)

    # Market: 4 quartiles by in-domain (A') median positive rank
    print("Market: computing per-id median positive rank on A'...", flush=True)
    G, pids, cids = load_cell_gallery("A'")
    per_id_val = per_id_median_pos_rank(G, pids, cids)
    assign, edges, counts = equal_count_bins(per_id_val, MARKET_N_BINS)
    out['Market'] = {
        'cell': "A'",
        'variable': 'in-domain median positive rank per identity',
        'n_bins': MARKET_N_BINS,
        'edges_value': edges,
        'bin_counts': counts,
        'assignment': assign,
        'n_ids': len(assign),
    }
    print(f"  Market bin counts: {counts}", flush=True)

    out_path = RESULTS / 'e41_partition_frozen.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved → {out_path}")


if __name__ == '__main__':
    main()
