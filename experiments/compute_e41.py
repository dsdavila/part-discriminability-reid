"""
E41: in-domain-only partition (Channel 2 rev 33 queue / Plan rev 58 audit).

Rebuild the partition from in-domain rank-1 error ALONE (no OOD arm in the
definition), bin by deciles with counts, re-run CMC@1/@5/@10 and median rank
of all positives per bin, four arms (id/ood/noise/randproj), MSMT and Market.

Plus a HELD-OUT version: bins fit on a random half of each identity's items,
evaluated on the other half.
"""
import json, time
import numpy as np
from pathlib import Path
from collections import Counter

CACHE_DIR = Path('/tmp/e12_feature_cache')
RESULTS   = Path('/data/papers/low_dimensional_reid/results')
N_BINS = 10
HELDOUT_SEED = 42

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


def compute_rank1_per_identity(G, g_pids, g_cids, chunk=256):
    """Per-identity mean rank-1 error, using all its items as queries."""
    N = len(G)
    q_wrong = np.zeros(N, dtype=np.int8)
    for start in range(0, N, chunk):
        end = min(N, start + chunk)
        sims = G[start:end] @ G.T
        for local, gi in enumerate(range(start, end)):
            exc = (g_pids == g_pids[gi]) & (g_cids == g_cids[gi])
            sims[local, exc] = -2.0
            top1 = int(np.argmax(sims[local]))
            q_wrong[gi] = int(g_pids[top1] != g_pids[gi])
    per_id_err = {}
    for pid in np.unique(g_pids):
        mask = g_pids == pid
        per_id_err[int(pid)] = float(q_wrong[mask].mean())
    return per_id_err, q_wrong


def compute_cmc_and_pos_ranks(G, g_pids, g_cids, q_idx, chunk=128):
    """For queries at q_idx (indices into G), compute per-query CMC{1,5,10}
    and ranks of ALL positives (1-indexed, same-cam/same-pid excluded)."""
    N = len(G)
    nq = len(q_idx)
    cmc1 = np.zeros(nq, dtype=bool)
    cmc5 = np.zeros(nq, dtype=bool)
    cmc10 = np.zeros(nq, dtype=bool)
    all_pos_ranks = []
    q_pids_out = g_pids[q_idx]
    for start in range(0, nq, chunk):
        end = min(nq, start + chunk)
        rows = q_idx[start:end]
        sims = G[rows] @ G.T
        for local, gi in enumerate(rows):
            qpid, qcid = g_pids[gi], g_cids[gi]
            exc = (g_pids == qpid) & (g_cids == qcid)
            sims[local, exc] = -2.0
            order = np.argsort(-sims[local])
            pos_mask_order = g_pids[order] == qpid
            if pos_mask_order.any():
                pos_ranks = np.where(pos_mask_order)[0] + 1
                all_pos_ranks.extend(int(r) for r in pos_ranks.tolist())
                cmc1[start + local] = pos_ranks[0] == 1
                cmc5[start + local] = pos_ranks[0] <= 5
                cmc10[start + local] = pos_ranks[0] <= 10
    return cmc1, cmc5, cmc10, all_pos_ranks, q_pids_out


def heldout_split(g_pids, seed=HELDOUT_SEED):
    rng = np.random.default_rng(seed)
    A = np.zeros(len(g_pids), dtype=bool)
    B = np.zeros(len(g_pids), dtype=bool)
    for pid in np.unique(g_pids):
        idx = np.where(g_pids == pid)[0]
        perm = rng.permutation(idx)
        half = len(perm) // 2
        A[perm[:half]] = True
        B[perm[half:]] = True
    return A, B


def compute_rank1_per_identity_from_subset(G, g_pids, g_cids, q_mask, chunk=256):
    """Per-identity rank-1 error using only queries in q_mask."""
    N = len(G)
    q_idx = np.where(q_mask)[0]
    q_wrong = {}
    for start in range(0, len(q_idx), chunk):
        end = min(len(q_idx), start + chunk)
        rows = q_idx[start:end]
        sims = G[rows] @ G.T
        for local, gi in enumerate(rows):
            exc = (g_pids == g_pids[gi]) & (g_cids == g_cids[gi])
            sims[local, exc] = -2.0
            top1 = int(np.argmax(sims[local]))
            q_wrong[gi] = int(g_pids[top1] != g_pids[gi])
    per_id_err = {}
    for pid in np.unique(g_pids):
        mask = (g_pids == pid) & q_mask
        if mask.sum() == 0:
            continue
        rates = [q_wrong[i] for i in np.where(mask)[0] if i in q_wrong]
        if rates:
            per_id_err[int(pid)] = float(np.mean(rates))
    return per_id_err


def bin_by_deciles(per_id_err, n_bins=N_BINS):
    """Return pid -> bin 0..n_bins-1 by quantile of in-domain error."""
    pids = np.array(sorted(per_id_err.keys()))
    errs = np.array([per_id_err[int(p)] for p in pids])
    # quantile edges
    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.quantile(errs, qs)
    # bin assignment (0..n_bins-1)
    bins = np.clip(np.digitize(errs, edges[1:-1], right=False), 0, n_bins - 1)
    return {int(p): int(b) for p, b in zip(pids, bins)}, edges.tolist()


def summarize_bin_metrics(cmc1, cmc5, cmc10, pos_ranks_list, q_pids, bin_map):
    """Group queries by bin (via query pid -> bin), report metrics per bin.
    pos_ranks_list is a flat list; we need to track which query each came from.
    Instead, recompute: we pass pos_ranks as per-query lists."""
    # bin for each query
    q_bins = np.array([bin_map.get(int(p), -1) for p in q_pids])
    results = {}
    for b in range(N_BINS):
        mask = q_bins == b
        if mask.sum() == 0:
            results[b] = None
            continue
        results[b] = {
            'n_queries': int(mask.sum()),
            'cmc1': float(cmc1[mask].mean()),
            'cmc5': float(cmc5[mask].mean()),
            'cmc10': float(cmc10[mask].mean()),
        }
    return results


def compute_metrics_with_bins(G, g_pids, g_cids, q_idx, bin_map, chunk=128):
    """Combined: for queries at q_idx, compute per-query CMC{1,5,10} and
    per-query median positive rank, grouped by bin from bin_map."""
    nq = len(q_idx)
    per_q_cmc1 = np.zeros(nq, dtype=bool)
    per_q_cmc5 = np.zeros(nq, dtype=bool)
    per_q_cmc10 = np.zeros(nq, dtype=bool)
    per_q_pos_ranks = [None] * nq
    for start in range(0, nq, chunk):
        end = min(nq, start + chunk)
        rows = q_idx[start:end]
        sims = G[rows] @ G.T
        for local, gi in enumerate(rows):
            qpid, qcid = g_pids[gi], g_cids[gi]
            exc = (g_pids == qpid) & (g_cids == qcid)
            sims[local, exc] = -2.0
            order = np.argsort(-sims[local])
            pos_mask_order = g_pids[order] == qpid
            if pos_mask_order.any():
                pos_ranks = (np.where(pos_mask_order)[0] + 1).tolist()
                per_q_pos_ranks[start + local] = pos_ranks
                per_q_cmc1[start + local] = pos_ranks[0] == 1
                per_q_cmc5[start + local] = pos_ranks[0] <= 5
                per_q_cmc10[start + local] = pos_ranks[0] <= 10
            else:
                per_q_pos_ranks[start + local] = []
    q_pids_out = g_pids[q_idx]
    q_bins = np.array([bin_map.get(int(p), -1) for p in q_pids_out])
    bins_res = {}
    for b in range(N_BINS):
        mask = q_bins == b
        if mask.sum() == 0:
            bins_res[b] = None
            continue
        all_ranks = []
        for i in np.where(mask)[0]:
            all_ranks.extend(per_q_pos_ranks[i])
        bins_res[b] = {
            'n_queries': int(mask.sum()),
            'n_positives': len(all_ranks),
            'cmc1': float(per_q_cmc1[mask].mean()),
            'cmc5': float(per_q_cmc5[mask].mean()),
            'cmc10': float(per_q_cmc10[mask].mean()),
            'median_pos_rank': float(np.median(all_ranks)) if all_ranks else None,
            'mean_pos_rank': float(np.mean(all_ranks)) if all_ranks else None,
        }
    return bins_res


def run_dataset(name, id_cell, ood_cell, sigma, k_rp):
    print(f"\n########## {name}: id={id_cell}/ood={ood_cell}, "
          f"σ={sigma}, k_rp={k_rp} ##########", flush=True)
    G_id, g_pids, g_cids = load_cell(id_cell)
    G_ood, _, _ = load_cell(ood_cell)
    print(f"  N={len(G_id)}  n_ids={len(np.unique(g_pids))}", flush=True)

    G_noise = noise_degrade(G_id, sigma)
    G_rp = randproj_degrade(G_id, k_rp)
    arms = {'id': G_id, 'ood': G_ood, 'noise': G_noise, 'randproj': G_rp}

    # ========== PART 1: full-items bin fit, evaluated on all queries ==========
    print("\n  --- PART 1: full-items partition ---", flush=True)
    t0 = time.time()
    print("  fitting per-identity rank-1 error (id arm, all items)...", flush=True)
    per_id_err, _ = compute_rank1_per_identity(G_id, g_pids, g_cids)
    print(f"    t={(time.time() - t0) / 60:.1f}m", flush=True)
    bin_map, edges = bin_by_deciles(per_id_err)
    bin_counts = Counter(bin_map.values())
    print(f"  deciles: {dict(sorted(bin_counts.items()))}", flush=True)
    print(f"  edges: {[f'{e:.4f}' for e in edges]}", flush=True)
    full_q_idx = np.arange(len(G_id))

    part1 = {'bin_edges': edges,
             'bin_counts': {int(b): int(c) for b, c in bin_counts.items()},
             'arms': {}}
    for arm_name, G_ in arms.items():
        t0 = time.time()
        print(f"  [{arm_name}] metrics per bin...", flush=True)
        bres = compute_metrics_with_bins(G_, g_pids, g_cids, full_q_idx, bin_map)
        print(f"    t={(time.time() - t0) / 60:.1f}m", flush=True)
        part1['arms'][arm_name] = bres

    # ========== PART 2: held-out (bins from half-A, eval on half-B) ==========
    print("\n  --- PART 2: held-out partition ---", flush=True)
    A_mask, B_mask = heldout_split(g_pids, seed=HELDOUT_SEED)
    print(f"  held-out split: A={A_mask.sum()} B={B_mask.sum()}", flush=True)
    t0 = time.time()
    print("  fitting per-identity rank-1 error (id arm, half-A queries)...", flush=True)
    per_id_err_h = compute_rank1_per_identity_from_subset(
        G_id, g_pids, g_cids, A_mask)
    print(f"    t={(time.time() - t0) / 60:.1f}m", flush=True)
    bin_map_h, edges_h = bin_by_deciles(per_id_err_h)
    bin_counts_h = Counter(bin_map_h.values())
    print(f"  held-out deciles: {dict(sorted(bin_counts_h.items()))}", flush=True)
    B_q_idx = np.where(B_mask)[0]

    part2 = {'bin_edges': edges_h,
             'bin_counts': {int(b): int(c) for b, c in bin_counts_h.items()},
             'arms': {}}
    for arm_name, G_ in arms.items():
        t0 = time.time()
        print(f"  [{arm_name}] metrics on half-B per held-out bin...", flush=True)
        bres = compute_metrics_with_bins(G_, g_pids, g_cids, B_q_idx, bin_map_h)
        print(f"    t={(time.time() - t0) / 60:.1f}m", flush=True)
        part2['arms'][arm_name] = bres

    return {'full': part1, 'heldout': part2}


def print_report(name, res, key='full'):
    print(f"\n========== {name} ({key}) — CMC@1 and median rank of all positives per decile ==========")
    arms = res[key]['arms']
    print(f"  decile n_ids   {'D cmc1':>7} {'B′ cmc1':>8} {'noise cmc1':>10} {'rp cmc1':>8} | "
          f"{'D medR':>8} {'B′ medR':>8} {'noise medR':>10} {'rp medR':>8}")
    arm_order = ['id', 'ood', 'noise', 'randproj']
    for b in range(N_BINS):
        rows = [arms[a][b] for a in arm_order]
        if any(r is None for r in rows):
            continue
        n_ids = res[key]['bin_counts'].get(b, 0)
        line = f"  {b:>6} {n_ids:>5}"
        for r in rows:
            line += f" {r['cmc1'] * 100:>6.2f}"
        line += " |"
        for r in rows:
            mr = r['median_pos_rank']
            line += f" {mr:>8.1f}" if mr is not None else f" {'--':>8}"
        print(line)


def main():
    out = {}
    out['MSMT'] = run_dataset('MSMT', 'D', "B'", SIGMA_D, K_RANDPROJ)
    out['Market'] = run_dataset('Market', "A'", 'C', SIGMA_A, K_RANDPROJ)
    print_report('MSMT', out['MSMT'], 'full')
    print_report('MSMT', out['MSMT'], 'heldout')
    print_report('Market', out['Market'], 'full')
    print_report('Market', out['Market'], 'heldout')

    out_path = RESULTS / 'e41_indomain_partition.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved → {out_path}")


if __name__ == '__main__':
    main()
