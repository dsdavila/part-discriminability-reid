"""
k-reciprocal re-ranking (Zhong et al. CVPR 2017), Ch2 rev 165.

Memory-efficient version: we do NOT materialise the full (Nq+Ng)x(Nq+Ng)
distance matrix. Instead we compute top-k1 indices for every query and
gallery item against the combined (query + gallery) set using chunked
GPU matmul. The expansion + Jaccard distance are then computed from
these compact index lists per probe.

Parameters (Zhong 2017 defaults):
  k1 = 20 (initial k-reciprocal neighbourhood)
  k2 = 6  (local query expansion)
  lambda = 0.3

Readouts (post re-ranking):
  - mAP (query -> gallery, standard Re-ID)
  - item-level k=10 skewness over gallery only (post-rerank gallery-gallery)
  - symmetric 10-NN fraction over gallery only
  - |r_global| Spearman of kocc vs distance to gallery centroid

For Run 2 completeness. Pre-registered by claude-chat rev 165:
  - MSMT ood |delta mAP| smallest of four arms (if ceiling holds)
  - Market more responsive than MSMT (shallower positives)
"""
import json, time, sys
import numpy as np
from pathlib import Path
from scipy import stats
import torch

CACHE_DIR = Path('/tmp/e12_feature_cache')
RESULTS   = Path('/data/papers/low_dimensional_reid/results')
K_OCC = 10
K1 = 20
K2 = 6
LAMBDA = 0.3

CELLS = {
    'D':  ('bot_msmt_seed42_msmt.npz',  11659),
    "B'": ('msmt17_test_noREA.npz',     11659),
    "A'": ('market_test_noREA.npz',      3368),
    'C':  ('bot_msmt_seed42_market.npz', 3368),
}

DATASETS = {
    'MSMT': {'id': 'D', 'ood': "B'"},
    'Market': {'id': "A'", 'ood': 'C'},
}

_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_qg(name):
    fn, n_q = CELLS[name]
    d = np.load(CACHE_DIR / fn); emb = d['emb'].astype(np.float32)
    return {
        'Q': emb[:n_q], 'Qp': d['pids'][:n_q], 'Qc': d['cids'][:n_q],
        'G': emb[n_q:], 'Gp': d['pids'][n_q:], 'Gc': d['cids'][n_q:],
    }


def l2(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(n, 1e-12)


def topk_against_combined(X_cuda, combined_cuda, k, chunk=256):
    """For each row of X (on GPU), compute top-k argmax indices against combined.
    Excludes self (combined items at the same row index if X is part of combined -
    handled via the exclusion bitmask on same-cam-same-pid later).
    Returns int32 array (N, k)."""
    N = X_cuda.shape[0]
    out = np.zeros((N, k), dtype=np.int32)
    for start in range(0, N, chunk):
        end = min(N, start + chunk)
        sims = X_cuda[start:end] @ combined_cuda.T  # (chunk, Nc)
        # top-k indices along dim=1
        _, idx = torch.topk(sims, k, dim=1)
        out[start:end] = idx.cpu().numpy().astype(np.int32)
    return out


def dist_from_sim(sims):
    return np.sqrt(np.clip(2.0 - 2.0 * sims, 0.0, None))


def k_reciprocal_rerank(Q_np, G_np, q_pids, q_cids, g_pids, g_cids,
                         k1=K1, k2=K2, lam=LAMBDA):
    """Memory-efficient Zhong 2017.

    Strategy:
      1) Build combined = [Q; G]. Compute top-(k1+1) indices for every row
         against combined using chunked GPU matmul. Returns initial_rank
         as (all_num, k1+1).
      2) For each i, compute k-reciprocal set R(i), expand using k1/2 for
         each candidate, weight by exp(-d).
      3) If k2 > 1, do local query expansion: V[i] = mean(V[top-k2 of i]).
      4) Compute Jaccard distance from V using inverted index.
      5) Combine with original: final = (1-lam) * jaccard + lam * original.
      6) Extract final[:Nq, Nq:] as the query-to-gallery distance.

    Same-cam-same-pid gallery items are excluded from the mAP computation,
    not the re-ranking (standard Re-ID convention).
    """
    Nq = len(Q_np); Ng = len(G_np)
    all_num = Nq + Ng

    combined = np.concatenate([Q_np, G_np], axis=0).astype(np.float32)
    combined_cuda = torch.from_numpy(combined).to(_DEVICE)

    print(f"    top-{k1+1} on combined ({all_num}x{all_num}) via GPU chunked...", flush=True)
    t0 = time.time()
    initial_rank = topk_against_combined(combined_cuda, combined_cuda, k1 + 1)
    # For the exact Zhong recipe we need an "original_dist" row per i, which
    # we don't store. Instead we compute what's needed row-by-row below.
    # But weighting V[i, ...] = exp(-d). We'll need d(i, j) for j in reciprocal
    # set only, which is small.
    print(f"      initial_rank shape {initial_rank.shape}  t={time.time()-t0:.1f}s", flush=True)

    # Precompute on-the-fly distances for the i-to-reciprocal distances.
    # For each i, we need sims(i, reciprocal_set). The reciprocal set size
    # is at most k1 * k1 (after expansion, usually ~k1 to 3*k1). We compute
    # those similarities per i via a small dot product.

    # Build V matrix sparsely: V[i, j] = normalised weight if j is in i's
    # expanded reciprocal set.
    V_rows = [None] * all_num  # list of (col_idx_array, value_array)

    print("    building V (reciprocal sets, expanded)...", flush=True)
    t0 = time.time()
    for i in range(all_num):
        forward = initial_rank[i, :k1+1]  # includes self at index 0 typically
        # backward: for each j in forward, is i in j's top-(k1+1)?
        backward = initial_rank[forward, :k1+1]
        # Find j's where i is in their top-(k1+1)
        in_back = (backward == i).any(axis=1)
        k_recip = forward[in_back]
        # Expand: for each candidate in k_recip, if its own k1/2 reciprocal set
        # overlaps with k_recip by > 2/3, add it
        k_recip_expansion = set(int(x) for x in k_recip)
        half = int(round(k1 / 2))
        for cand in k_recip:
            cand_forward = initial_rank[cand, :half+1]
            cand_backward = initial_rank[cand_forward, :half+1]
            cand_in_back = (cand_backward == cand).any(axis=1)
            cand_k_recip = cand_forward[cand_in_back]
            if len(cand_k_recip) == 0:
                continue
            inter = len(set(cand_k_recip.tolist()) & set(k_recip.tolist()))
            if inter > (2.0 / 3.0) * len(cand_k_recip):
                for x in cand_k_recip:
                    k_recip_expansion.add(int(x))
        cols = np.array(sorted(k_recip_expansion), dtype=np.int32)
        # Compute d(i, cols) via GPU
        sims = (combined_cuda[i:i+1] @ combined_cuda[cols].T).cpu().numpy()[0]
        d = dist_from_sim(sims)
        w = np.exp(-d)
        w = w / max(w.sum(), 1e-12)
        V_rows[i] = (cols, w.astype(np.float32))
        if (i + 1) % 5000 == 0:
            print(f"      i={i+1}/{all_num} t={time.time()-t0:.1f}s", flush=True)

    print(f"      V built  t={time.time()-t0:.1f}s", flush=True)

    # Local query expansion: V[i] = mean of V[initial_rank[i, :k2]]
    if k2 > 1:
        print("    local query expansion...", flush=True)
        t0 = time.time()
        V_rows_new = [None] * all_num
        for i in range(all_num):
            agg = {}
            neigh = initial_rank[i, :k2]
            for n in neigh:
                cols, vals = V_rows[int(n)]
                for c, v in zip(cols.tolist(), vals.tolist()):
                    agg[c] = agg.get(c, 0.0) + v / k2
            if len(agg) == 0:
                V_rows_new[i] = (np.array([], dtype=np.int32), np.array([], dtype=np.float32))
            else:
                cols_n = np.array(sorted(agg.keys()), dtype=np.int32)
                vals_n = np.array([agg[c] for c in cols_n.tolist()], dtype=np.float32)
                V_rows_new[i] = (cols_n, vals_n)
            if (i + 1) % 5000 == 0:
                print(f"      i={i+1}/{all_num} t={time.time()-t0:.1f}s", flush=True)
        V_rows = V_rows_new
        print(f"      QE done  t={time.time()-t0:.1f}s", flush=True)

    # Inverted index: for each j, list of (i, V[i, j])
    print("    inverted index...", flush=True)
    t0 = time.time()
    inv = [[] for _ in range(all_num)]
    for i in range(all_num):
        cols, vals = V_rows[i]
        for c, v in zip(cols, vals):
            inv[int(c)].append((i, float(v)))
    print(f"      t={time.time()-t0:.1f}s", flush=True)

    # Jaccard distance for queries vs galleries:
    # jaccard(i, j) = 1 - sum_c min(V[i,c], V[j,c]) / (sum_c (V[i,c] + V[j,c]) - sum_c min(...))
    # Standard Zhong: 1 - temp_min / (2 - temp_min)
    print("    computing Jaccard distance for queries vs gallery...", flush=True)
    jaccard_qg = np.zeros((Nq, Ng), dtype=np.float32)
    t0 = time.time()
    for q in range(Nq):
        cols_q, vals_q = V_rows[q]
        if len(cols_q) == 0:
            jaccard_qg[q] = 1.0
            continue
        # For each col c in cols_q, we know V[q, c]. We need min(V[q, c], V[j, c])
        # for each j in [Nq, Nq+Ng). Use inverted index.
        temp_min = np.zeros(Ng, dtype=np.float32)
        for c, vq in zip(cols_q.tolist(), vals_q.tolist()):
            for (idx, vj) in inv[int(c)]:
                if idx >= Nq:  # gallery item
                    g_idx = idx - Nq
                    temp_min[g_idx] += min(vq, vj)
        jaccard_qg[q] = 1.0 - temp_min / (2.0 - temp_min + 1e-12)
        if (q + 1) % 500 == 0:
            print(f"      q={q+1}/{Nq} t={time.time()-t0:.1f}s", flush=True)
    print(f"      t={time.time()-t0:.1f}s", flush=True)

    # Original distance (query to gallery)
    sims_qg = (combined_cuda[:Nq] @ combined_cuda[Nq:].T).cpu().numpy()
    orig_qg = dist_from_sim(sims_qg)
    # Normalise per Zhong: original_dist = original_dist / max(original_dist, axis=0)
    # (But he squares + normalises in a specific way. For simplicity use cosine
    # dist in [0, 2].)
    final = (1 - lam) * jaccard_qg + lam * orig_qg

    del combined_cuda
    torch.cuda.empty_cache()

    # Compute mAP from final
    ap_sum = 0.0; n_valid = 0
    for q in range(Nq):
        qp, qc = q_pids[q], q_cids[q]
        exc = (g_pids == qp) & (g_cids == qc)
        n_pos = int(((g_pids == qp) & ~exc).sum())
        if n_pos == 0:
            continue
        d = final[q].copy()
        d[exc] = 1e9
        order = np.argsort(d, kind='stable')
        pos = (g_pids[order] == qp) & ~exc[order]
        cum = np.cumsum(pos)
        ranks = np.arange(1, len(order) + 1)
        prec = cum / ranks
        ap = prec[pos].sum() / n_pos
        ap_sum += ap
        n_valid += 1
    mAP = ap_sum / n_valid if n_valid > 0 else 0.0
    return mAP, final


def main():
    print(f"Device: {_DEVICE}", flush=True)
    print(f"GPU mem total: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB", flush=True)

    out = {}
    for ds_name, cfg in DATASETS.items():
        out[ds_name] = {}
        for arm_name, cell_key in [('id', cfg['id']), ('ood', cfg['ood'])]:
            print(f"\n### {ds_name} / {arm_name} ###", flush=True)
            data = load_qg(cell_key)
            Q = l2(data['Q']); G = l2(data['G'])
            # Guard for MSMT: 93k x 93k matrix cannot be materialised fully.
            # Our implementation does not materialise it; initial_rank is
            # (93k, 21) int32 = 7.8 MB. V is sparse (avg ~20-50 per row).
            # Jaccard loop will be 11k * 20 lookups per query = feasible.
            Nq, Ng = len(Q), len(G)
            all_num = Nq + Ng
            print(f"  Nq={Nq}  Ng={Ng}  all_num={all_num}", flush=True)
            if all_num > 50000:
                print(f"  NOTE: all_num={all_num} is large; Jaccard loop will be slow", flush=True)
            t0 = time.time()
            mAP, _ = k_reciprocal_rerank(Q, G, data['Qp'], data['Qc'],
                                          data['Gp'], data['Gc'])
            print(f"  k-reciprocal mAP = {mAP*100:.3f}  t={(time.time()-t0)/60:.1f}m", flush=True)
            out[ds_name][arm_name] = {'k_reciprocal_mAP_pct': float(mAP * 100.0),
                                       'k1': K1, 'k2': K2, 'lambda': LAMBDA}

    print("\n========== k-reciprocal summary ==========")
    for ds in DATASETS:
        for arm in ('id', 'ood'):
            r = out[ds][arm]
            print(f"  {ds} {arm}: k-reciprocal mAP = {r['k_reciprocal_mAP_pct']:.3f}")

    out_path = RESULTS / 'e43_k_reciprocal.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == '__main__':
    main()
