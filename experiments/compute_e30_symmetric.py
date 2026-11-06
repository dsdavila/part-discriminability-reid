"""
E30 — symmetric ΔAP split: D-only vs shared on MSMT, A-only vs shared on Market.

For each in-domain cell X (D or A), partition X's total ΔAP between confuser
identities that are shared with its OOD counterpart (Y) and X-only confusers.

Uses the additive 1/r attribution from compute_e30.py.
Recoverable fraction = (OOD B-only share − ID X-only share) ≤ (OOD_ΔAP − ID_ΔAP)/OOD_ΔAP
"""
import json, time
import numpy as np
from pathlib import Path
from collections import defaultdict

CACHE_DIR = Path('/tmp/e12_feature_cache')
RESULTS   = Path('/data/papers/low_dimensional_reid/results')
K_LIST    = [10, 50]

CELLS = {
    'A': {'file': 'market_test.npz',           'n_q': 3368},
    'C': {'file': 'bot_msmt_seed42_market.npz','n_q': 3368},
    'D': {'file': 'bot_msmt_seed42_msmt.npz',  'n_q': 11659},
    'B': {'file': 'msmt17_test.npz',           'n_q': 11659},
}

# For each in-domain pass: evaluate ID cell's ΔAP, label confuser as "shared" if also in OOD cell's intruder set.
# Pair (ID_cell, OOD_cell_for_shared_classification):
ID_PAIRS = [('MSMT', 'D', 'B'),   # Compute D's ΔAP split, "shared" = also in B's intruder set
            ('Market', 'A', 'C')]


def load_cell(fname, n_q):
    d = np.load(CACHE_DIR / fname)
    emb = d['emb'].astype(np.float32)
    return (emb[:n_q], d['pids'][:n_q], d['cids'][:n_q],
            emb[n_q:], d['pids'][n_q:], d['cids'][n_q:])


def per_query_dap_by_intruder_id(q_emb, q_pids, q_cids, g_emb, g_pids, g_cids):
    sims = q_emb @ g_emb.T
    results = []
    for i in range(len(q_emb)):
        qp, qc = q_pids[i], q_cids[i]
        valid = ~((g_pids == qp) & (g_cids == qc))
        rel = (g_pids == qp) & valid
        if rel.sum() == 0: continue
        order = np.argsort(-sims[i][valid])
        rel_v = rel[valid][order]
        if rel_v.sum() == 0: continue
        valid_g_pids = g_pids[valid][order]
        pos_mask = rel_v
        m = int(pos_mask.sum())
        pos_ranks = np.where(pos_mask)[0] + 1
        j_arr = np.arange(1, m+1)
        ap = float((j_arr / pos_ranks).mean())
        if m == 0:
            results.append((i, int(qp), ap, {})); continue
        term = (1.0 / pos_ranks) / m
        csfa = np.cumsum(term[::-1])[::-1]
        csfa_ext = np.concatenate([csfa, [0.0]])
        intr_pos = np.where(~pos_mask)[0] + 1
        first_j_idx = np.searchsorted(pos_ranks, intr_pos, side='right')
        contrib = csfa_ext[first_j_idx]
        intr_pids = valid_g_pids[~pos_mask]
        dap_by_intr = defaultdict(float)
        for cid, c in zip(intr_pids, contrib):
            dap_by_intr[int(cid)] += float(c)
        results.append((i, int(qp), ap, dict(dap_by_intr)))
    return results


def topk_neighbours(G, g_pids, g_cids, k_max, chunk=2048):
    N = len(G)
    out = np.full((N, k_max), -1, dtype=np.int64)
    for start in range(0, N, chunk):
        end = min(N, start + chunk)
        sims = G[start:end] @ G.T
        for local, gi in enumerate(range(start, end)):
            exc = (g_pids == g_pids[gi]) & (g_cids == g_cids[gi])
            sims[local, exc] = -2.0
            top_idx = np.argpartition(-sims[local], k_max)[:k_max]
            order = np.argsort(-sims[local, top_idx])
            out[gi] = top_idx[order]
    return out


def per_identity_sets(topk_idx, g_pids, k):
    unique_ids = np.unique(g_pids)
    nbr_pids_k = g_pids[topk_idx[:, :k]]
    im_mask = (nbr_pids_k != g_pids[:, None])
    out = {}
    for uid in unique_ids:
        mask = (g_pids == uid)
        intruders = nbr_pids_k[mask][im_mask[mask]]
        out[int(uid)] = set(int(x) for x in intruders)
    return out


def analyse_id(eval_name, cell_id, cell_ood):
    print(f"\n--- {eval_name}: {cell_id} (ID) with shared-set from {cell_ood} (OOD) ---", flush=True)
    d_id  = np.load(CACHE_DIR / CELLS[cell_id]['file'])
    d_ood = np.load(CACHE_DIR / CELLS[cell_ood]['file'])
    n_q = CELLS[cell_id]['n_q']
    g_pids = d_ood['pids'][n_q:]; g_cids = d_ood['cids'][n_q:]

    print(f"  topk under {cell_ood} (OOD) for shared-set classification...", flush=True)
    G_ood = d_ood['emb'][n_q:].astype(np.float32)
    top_ood = topk_neighbours(G_ood, g_pids, g_cids, max(K_LIST))

    print(f"  per-query ΔAP under {cell_id} (ID)...", flush=True)
    q_emb = d_id['emb'][:n_q].astype(np.float32)
    q_pids = d_id['pids'][:n_q]; q_cids = d_id['cids'][:n_q]
    G_id = d_id['emb'][n_q:].astype(np.float32)
    attr = per_query_dap_by_intruder_id(q_emb, q_pids, q_cids, G_id, g_pids, g_cids)

    out = {}
    for k in K_LIST:
        set_ood = per_identity_sets(top_ood, g_pids, k)
        total_shared = total_idonly = total_dap = 0.0
        for (i, qp, ap, dap_dict) in attr:
            shared_set = set_ood.get(qp, set())
            total_dap += (1 - ap)
            for cid, val in dap_dict.items():
                if cid in shared_set: total_shared += val
                else: total_idonly += val
        row = {
            'total_sigma_dap': total_dap,
            'share_dap_shared_with_OOD': total_shared / max(total_dap, 1e-12),
            'share_dap_IDonly':          total_idonly / max(total_dap, 1e-12),
        }
        row['sum_sanity'] = row['share_dap_shared_with_OOD'] + row['share_dap_IDonly']
        print(f"  k={k}  shared={row['share_dap_shared_with_OOD']:.4f}  "
              f"{cell_id}-only={row['share_dap_IDonly']:.4f}  "
              f"sum={row['sum_sanity']:.4f}", flush=True)
        out[str(k)] = row
    return out


def main():
    results = {}
    for name, c_id, c_ood in ID_PAIRS:
        results[name] = {'id_cell': c_id, 'ood_cell': c_ood,
                         'splits': analyse_id(name, c_id, c_ood)}
    out_path = RESULTS / 'e30_symmetric_split.json'
    with open(out_path, 'w') as f: json.dump(results, f, indent=2)
    print(f"\nSaved → {out_path}", flush=True)


if __name__ == '__main__':
    main()
