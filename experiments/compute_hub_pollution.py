"""
Hub-pollution test: do OOD-ONLY identities' B' confuser sets contain SHARED
identities at above the rate expected from SHARED's gallery share?

Also:
- noise-null Δg comparison for Market and MSMT (D vs noise-degraded-D)
- Per-class bootstrap CIs on Δg

Priority 1: hub-pollution test (cheapest and most mechanistic).
"""
import json, time
import numpy as np
from pathlib import Path
from collections import defaultdict

CACHE_DIR = Path('/tmp/e12_feature_cache')
RESULTS   = Path('/data/papers/low_dimensional_reid/results')

CELL_FILES = {
    "A'": ('market_test_noREA.npz', 3368),
    'C':  ('bot_msmt_seed42_market.npz', 3368),
    'D':  ('bot_msmt_seed42_msmt.npz', 11659),
    "B'": ('msmt17_test_noREA.npz', 11659),
}


def load_cell(cell):
    fn, n_q = CELL_FILES[cell]
    d = np.load(CACHE_DIR / fn); emb = d['emb'].astype(np.float32)
    return (emb[:n_q], d['pids'][:n_q], d['cids'][:n_q],
            emb[n_q:], d['pids'][n_q:], d['cids'][n_q:])


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
    rates = {}
    for uid in np.unique(g_pids):
        rates[int(uid)] = float(wrong[g_pids == uid].mean())
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


def per_identity_sets(topk_idx, g_pids, k=10):
    uniq = np.unique(g_pids)
    nbr = g_pids[topk_idx[:, :k]]
    im = (nbr != g_pids[:, None])
    out = {}
    for uid in uniq:
        mask = g_pids == uid
        intruders = nbr[mask][im[mask]]
        out[int(uid)] = set(int(x) for x in intruders)
    return out


def classify(rates_id, rates_ood, thr=0.5):
    classes = {}
    for uid in rates_id:
        r_id = rates_id.get(uid, 0.0); r_ood = rates_ood.get(uid, 0.0)
        if r_id <= thr and r_ood <= thr: classes[uid] = 'EASY'
        elif r_id > thr and r_ood > thr: classes[uid] = 'SHARED'
        elif r_id > thr: classes[uid] = 'ID-ONLY'
        else: classes[uid] = 'OOD-ONLY'
    return classes


def main():
    out = {}
    for eval_name, c_id, c_ood in [('MSMT', 'D', "B'"), ('Market', "A'", 'C')]:
        print(f"\n=== {eval_name}: {c_id} vs {c_ood} ===", flush=True)
        _, _, _, G_id, g_pids, g_cids = load_cell(c_id)
        _, _, _, G_ood, _, _ = load_cell(c_ood)
        print(f"  rank-1 rates (id)...", flush=True)
        rates_id = top1_wrong_rate(G_id, g_pids, g_cids)
        print(f"  rank-1 rates (ood)...", flush=True)
        rates_ood = top1_wrong_rate(G_ood, g_pids, g_cids)
        classes = classify(rates_id, rates_ood)
        from collections import Counter
        cnts = Counter(classes.values())
        print(f"  classes: {dict(cnts)}", flush=True)

        shared_ids = set(uid for uid, c in classes.items() if c == 'SHARED')
        ood_ids   = set(uid for uid, c in classes.items() if c == 'OOD-ONLY')
        easy_ids  = set(uid for uid, c in classes.items() if c == 'EASY')
        all_ids   = set(classes.keys())
        shared_fraction_global = len(shared_ids) / len(all_ids)

        print(f"  top-10 neighbours under {c_ood}...", flush=True)
        topk_ood = topk_neighbours(G_ood, g_pids, g_cids, 10)
        sets_ood = per_identity_sets(topk_ood, g_pids, 10)

        # Expected under uniform: for each identity, fraction of its confuser set that
        # falls in SHARED should ≈ |SHARED|/(N_ids - 1).
        # Measured: compute per-OOD-ONLY-identity: |confuser_set ∩ SHARED| / |confuser_set|
        def share_in_class(target_set, confuser_set):
            if not confuser_set: return None
            return len(confuser_set & target_set) / len(confuser_set)

        # For OOD-ONLY identities: share of their B'-confusers that are SHARED
        shares_ood_in_shared = [share_in_class(shared_ids, sets_ood[uid]) for uid in ood_ids
                                 if share_in_class(shared_ids, sets_ood[uid]) is not None]
        # Baseline: share for EASY identities' confusers (should also hit SHARED at chance?)
        shares_easy_in_shared = [share_in_class(shared_ids, sets_ood[uid]) for uid in easy_ids
                                  if share_in_class(shared_ids, sets_ood[uid]) is not None]
        # Baseline: share for SHARED identities' confusers (self-confusion)
        shares_shared_in_shared = [share_in_class(shared_ids, sets_ood[uid]) for uid in shared_ids
                                    if share_in_class(shared_ids, sets_ood[uid]) is not None]

        # Null: uniform over all non-same identities
        null_share = shared_fraction_global   # fraction of all identities that are SHARED

        def summarize(name, vals):
            if not vals:
                print(f"  {name:<30}: n=0")
                return {}
            arr = np.array(vals)
            print(f"  {name:<30}: n={len(arr)}  mean={arr.mean():.4f}  median={np.median(arr):.4f}  vs null {null_share:.4f}")
            return {'n': int(len(arr)), 'mean': float(arr.mean()), 'median': float(np.median(arr))}

        row = {
            'eval': eval_name, 'id_cell': c_id, 'ood_cell': c_ood,
            'class_counts': dict(cnts),
            'null_share_SHARED': float(null_share),
            'OOD-ONLY_confusers_share_in_SHARED': summarize('OOD-ONLY → fraction in SHARED', shares_ood_in_shared),
            'EASY_confusers_share_in_SHARED': summarize('EASY → fraction in SHARED', shares_easy_in_shared),
            'SHARED_confusers_share_in_SHARED': summarize('SHARED → fraction in SHARED', shares_shared_in_shared),
        }
        # Also compute ratio to null
        if row['OOD-ONLY_confusers_share_in_SHARED']:
            row['OOD-ONLY_ratio_to_null'] = row['OOD-ONLY_confusers_share_in_SHARED']['mean'] / null_share
            print(f"  → OOD-ONLY share / null = {row['OOD-ONLY_ratio_to_null']:.3f}× (null = SHARED's share of gallery)")
        out[eval_name] = row

    out_path = RESULTS / 'hub_pollution.json'
    with open(out_path, 'w') as f: json.dump(out, f, indent=2)
    print(f"\nSaved → {out_path}")


if __name__ == '__main__':
    main()
