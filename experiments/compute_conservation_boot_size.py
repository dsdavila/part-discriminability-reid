"""
Two follow-ups to R-conservation-Donly:
  (1) Per-identity bootstrap CI on the Market excess (C-N routing to A-hard)
  (2) Size-matched MSMT subsample: shrink D-hard to Market's 1.4% item share
      and check whether the 18.2% excess falls toward 11.2%.
"""
import json, time
import numpy as np
from pathlib import Path
from collections import defaultdict

CACHE_DIR = Path('/tmp/e12_feature_cache')
RESULTS   = Path('/data/papers/low_dimensional_reid/results')
K = 10
MSMT_SIGMA = 0.0625
MKT_SIGMA  = 0.062
N_BOOT     = 500


def load_cell(cell):
    files = {
        "A'": ('market_test_noREA.npz', 3368),
        'C':  ('bot_msmt_seed42_market.npz', 3368),
        'D':  ('bot_msmt_seed42_msmt.npz', 11659),
        "B'": ('msmt17_test_noREA.npz', 11659),
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


def top1_wrong_rate(G, g_pids, g_cids, chunk=2048):
    N = len(G); t1 = np.full(N, -1, dtype=np.int64)
    for start in range(0, N, chunk):
        end = min(N, start + chunk)
        sims = G[start:end] @ G.T
        for local, gi in enumerate(range(start, end)):
            exc = (g_pids == g_pids[gi]) & (g_cids == g_cids[gi])
            sims[local, exc] = -2.0
            t1[gi] = g_pids[int(np.argmax(sims[local]))]
    wrong = (t1 != g_pids)
    return {int(uid): float(wrong[g_pids == uid].mean()) for uid in np.unique(g_pids)}


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


def per_item_hits_to_hardclass(top10_idx, g_pids, hard_pid_set):
    """For each item i, count how many of its top-10 neighbours have pid ∈ hard_pid_set."""
    nbr = g_pids[top10_idx[:, :K]]
    is_hard = np.isin(nbr, list(hard_pid_set))
    return is_hard.sum(axis=1)   # (N,) counts of hard neighbours in top-10


def item_to_pid_class(g_pids, hard_pid_set):
    return np.array(['hard' if int(p) in hard_pid_set else 'easy' for p in g_pids])


# ─── Part (1): Market per-identity bootstrap of excess ──────────────────────

def part1_market_bootstrap():
    print("\n=== Part 1: Market per-identity bootstrap of excess routing ===", flush=True)
    q_A, _, _, G_A, g_pids, g_cids = load_cell("A'")
    _, _, _, G_C, _, _ = load_cell('C')
    rates_A = top1_wrong_rate(G_A, g_pids, g_cids)
    hard_pids = set(int(uid) for uid, r in rates_A.items() if r > 0.5)
    item_class = item_to_pid_class(g_pids, hard_pids)
    print(f"  A-hard identities: {len(hard_pids)} (items: {(item_class == 'hard').sum()})", flush=True)

    print("  topk under A'...", flush=True)
    top_A = topk_neighbours(G_A, g_pids, g_cids, K)
    print("  topk under C...", flush=True)
    top_C = topk_neighbours(G_C, g_pids, g_cids, K)
    print(f"  noise-A' at σ={MKT_SIGMA}...", flush=True)
    G_A_noise, _ = noise_degrade(G_A, q_A, MKT_SIGMA)
    print("  topk under noise-A'...", flush=True)
    top_N = topk_neighbours(G_A_noise, g_pids, g_cids, K)

    # Per-item: slots to A-hard under each condition
    hits_A = per_item_hits_to_hardclass(top_A, g_pids, hard_pids)
    hits_C = per_item_hits_to_hardclass(top_C, g_pids, hard_pids)
    hits_N = per_item_hits_to_hardclass(top_N, g_pids, hard_pids)

    # Aggregate to per-identity for A-easy identities
    easy_pids = sorted([int(uid) for uid in np.unique(g_pids) if int(uid) not in hard_pids])
    per_id_C = {}
    per_id_N = {}
    for uid in easy_pids:
        mask = (g_pids == uid)
        per_id_C[uid] = float(hits_C[mask].sum())  # total A-hard slots in uid's items' top-10 under C
        per_id_N[uid] = float(hits_N[mask].sum())  # under noise

    diffs = np.array([per_id_C[uid] - per_id_N[uid] for uid in easy_pids])
    print(f"  per-A'-easy-identity (C − noise) hits to A-hard:")
    print(f"    n={len(easy_pids)}  mean={diffs.mean():.4f}  SD={diffs.std():.4f}  median={np.median(diffs):.4f}")
    total_C = sum(per_id_C.values()); total_N = sum(per_id_N.values())
    pct_excess = 100 * (total_C - total_N) / max(total_N, 1)
    print(f"  total excess slots (C − noise): {total_C - total_N:+.0f}, pct above noise: {pct_excess:+.1f}%")

    # Bootstrap over easy identities
    rng = np.random.default_rng(42)
    boots_pct = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, len(easy_pids), size=len(easy_pids))
        sampled_pids = [easy_pids[i] for i in idx]
        tC = sum(per_id_C[p] for p in sampled_pids)
        tN = sum(per_id_N[p] for p in sampled_pids)
        boots_pct.append(100 * (tC - tN) / max(tN, 1))
    ci = (float(np.percentile(boots_pct, 2.5)), float(np.percentile(boots_pct, 97.5)))
    print(f"  bootstrap CI95 on pct-excess: [{ci[0]:+.2f}%, {ci[1]:+.2f}%]", flush=True)
    return {'n_easy': len(easy_pids),
            'total_excess_slots': total_C - total_N,
            'pct_excess_point': pct_excess,
            'pct_excess_ci95': list(ci),
            'per_identity_mean': float(diffs.mean()),
            'per_identity_sd': float(diffs.std())}


# ─── Part (2): MSMT size-matched subsample ──────────────────────────────────

def part2_msmt_size_matched():
    print("\n=== Part 2: MSMT with D-hard subsampled to Market-size (1.4% of items) ===", flush=True)
    q_D, _, _, G_D, g_pids, g_cids = load_cell('D')
    _, _, _, G_B, _, _ = load_cell("B'")
    rates_D = top1_wrong_rate(G_D, g_pids, g_cids)
    hard_all = sorted([int(uid) for uid, r in rates_D.items() if r > 0.5])
    print(f"  MSMT D-hard full: {len(hard_all)} identities", flush=True)

    print("  topk under D...", flush=True)
    top_D = topk_neighbours(G_D, g_pids, g_cids, K)
    print("  topk under B'...", flush=True)
    top_B = topk_neighbours(G_B, g_pids, g_cids, K)
    G_D_noise, _ = noise_degrade(G_D, q_D, MSMT_SIGMA)
    print("  topk under noise-D...", flush=True)
    top_N = topk_neighbours(G_D_noise, g_pids, g_cids, K)

    results_per_K = []
    # Target fractions matching ratios like Market's. Also include full for comparison.
    rng = np.random.default_rng(42)
    for n_hard in [23, 50, 100, 200, 400, 693]:
        trials = []
        for trial in range(10 if n_hard < 693 else 1):
            if n_hard >= len(hard_all):
                hard_sub = set(hard_all)
            else:
                hard_sub = set(rng.choice(hard_all, n_hard, replace=False).tolist())
            # Items in subsample
            n_items_hard = int(sum(int(p) in hard_sub for p in g_pids))
            # Compute flux: for each condition, count slots to hard-sub
            hits_D = per_item_hits_to_hardclass(top_D, g_pids, hard_sub).sum()
            hits_B = per_item_hits_to_hardclass(top_B, g_pids, hard_sub).sum()
            hits_N = per_item_hits_to_hardclass(top_N, g_pids, hard_sub).sum()
            pct_excess = 100 * (hits_B - hits_N) / max(hits_N, 1)
            trials.append({'n_hard': int(n_hard), 'n_items_hard': int(n_items_hard),
                            'hits_D': int(hits_D), 'hits_B': int(hits_B), 'hits_N': int(hits_N),
                            'pct_excess': float(pct_excess)})
        pct_means = [t['pct_excess'] for t in trials]
        avg_items = np.mean([t['n_items_hard'] for t in trials])
        print(f"  n_hard={n_hard:>3} (~{avg_items:.0f} items, {avg_items/82128*100:.2f}% of gallery): "
              f"pct_excess = {np.mean(pct_means):+.2f}% ± {np.std(pct_means):.2f}  (over {len(trials)} trials)", flush=True)
        results_per_K.append({'n_hard': n_hard,
                              'mean_pct_excess': float(np.mean(pct_means)),
                              'sd_pct_excess': float(np.std(pct_means)),
                              'avg_n_items': float(avg_items),
                              'trials': trials})
    return results_per_K


def main():
    out = {}
    out['market_bootstrap'] = part1_market_bootstrap()
    out['msmt_size_matched'] = part2_msmt_size_matched()
    out_path = RESULTS / 'conservation_followups.json'
    with open(out_path, 'w') as f: json.dump(out, f, indent=2)
    print(f"\nSaved → {out_path}")


if __name__ == '__main__':
    main()
