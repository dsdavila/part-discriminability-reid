"""
E31 loop-closing test: does the band-pass gain concentrate on OOD-ONLY identities?

Protocol (MSMT eval):
  1. Build per-identity confuser sets at k=10 under D and under B'.
  2. Classify each gallery identity into EASY/SHARED/ID-ONLY/OOD-ONLY using the
     chance-corrected SHARED rule (|∩_i| > |D_i|·|B'_i|/N).
  3. Compute per-query AP under B' baseline (undegraded).
  4. Compute per-query AP under B' projected (j=8, k=all band-pass).
  5. Report per-class mean AP in both conditions and the mean ΔAP recovery.
"""
import json, time
import numpy as np
from pathlib import Path

CACHE_DIR = Path('/tmp/e12_feature_cache')
RESULTS   = Path('/data/papers/low_dimensional_reid/results')
K = 10
J_BEST, K_BAND_BEST = 8, 'all'   # B's best (j, k) from the band-pass sweep


def load_cell(cell_file, n_q):
    d = np.load(CACHE_DIR / cell_file)
    emb = d['emb'].astype(np.float32)
    return (emb[:n_q], d['pids'][:n_q], d['cids'][:n_q],
            emb[n_q:], d['pids'][n_q:], d['cids'][n_q:])


def l2(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(n, 1e-12)


def topk_neighbours(G, g_pids, g_cids, k_max, chunk=2048):
    N = len(G)
    out = np.full((N, k_max), -1, dtype=np.int64)
    for start in range(0, N, chunk):
        end = min(N, start + chunk)
        sims = G[start:end] @ G.T
        for local, gi in enumerate(range(start, end)):
            exc = (g_pids == g_pids[gi]) & (g_cids == g_cids[gi])
            sims[local, exc] = -2.0
            top = np.argpartition(-sims[local], k_max)[:k_max]
            out[gi] = top[np.argsort(-sims[local, top])]
    return out


def per_identity_sets(topk_idx, g_pids, k):
    uniq = np.unique(g_pids)
    nbr = g_pids[topk_idx[:, :k]]
    im = (nbr != g_pids[:, None])
    out = {}
    for uid in uniq:
        mask = g_pids == uid
        intruders = nbr[mask][im[mask]]
        out[int(uid)] = set(int(x) for x in intruders)
    return out


def per_query_ap(q_emb, q_pids, q_cids, g_emb, g_pids, g_cids):
    sims = q_emb @ g_emb.T
    out = []
    for i in range(len(q_emb)):
        qp, qc = q_pids[i], q_cids[i]
        valid = ~((g_pids == qp) & (g_cids == qc))
        rel = (g_pids == qp) & valid
        if rel.sum() == 0:
            out.append((int(qp), None)); continue
        order = np.argsort(-sims[i][valid])
        rel_v = rel[valid][order]
        if rel_v.sum() == 0:
            out.append((int(qp), None)); continue
        pos = np.where(rel_v)[0]
        ap = float(((np.arange(len(pos))+1) / (pos+1)).mean())
        out.append((int(qp), ap))
    return out


def pca_basis(G):
    mu = G.mean(axis=0, keepdims=True)
    Gc = (G - mu).astype(np.float64)
    _, _, Vt = np.linalg.svd(Gc, full_matrices=False)
    return mu.astype(np.float32), Vt.T.astype(np.float32)


def project(X, mu, V, j, k_or_all):
    D = V.shape[1]
    end = D if k_or_all == 'all' else min(D, j + k_or_all)
    if end <= j: return None
    Vk = V[:, j:end]
    return l2((X - mu) @ Vk)


def main():
    print("Loading D and B'...", flush=True)
    q_emb_B, q_pids_B, q_cids_B, G_B, g_pids_B, g_cids_B = load_cell('msmt17_test_noREA.npz', 11659)
    _, _, _, G_D, g_pids_D, g_cids_D = load_cell('bot_msmt_seed42_msmt.npz', 11659)
    assert np.array_equal(g_pids_B, g_pids_D)
    g_pids = g_pids_B; g_cids = g_cids_B
    N_ids = int(len(np.unique(g_pids)))
    print(f"  N_ids = {N_ids}", flush=True)

    print("Topk under D (for classification)...", flush=True)
    t0 = time.time()
    top_D = topk_neighbours(G_D, g_pids, g_cids, K)
    print(f"  t={(time.time()-t0)/60:.1f}m", flush=True)
    print("Topk under B' (for classification)...", flush=True)
    t0 = time.time()
    top_B = topk_neighbours(G_B, g_pids, g_cids, K)
    print(f"  t={(time.time()-t0)/60:.1f}m", flush=True)

    sets_D = per_identity_sets(top_D, g_pids, K)
    sets_B = per_identity_sets(top_B, g_pids, K)

    # Classify
    classes = {}
    for uid in sets_D:
        a, b = sets_D[uid], sets_B[uid]
        if not a and not b:
            cls = 'EASY'
        else:
            inter = len(a & b)
            chance = len(a) * len(b) / N_ids
            has_shared = inter > chance
            in_D = len(a) > 0; in_B = len(b) > 0
            if in_D and in_B and has_shared:
                cls = 'SHARED'
            elif in_D and in_B and not has_shared:
                cls = 'SHARED_atchance'   # both confused but not above chance
            elif in_D and not in_B:
                cls = 'ID-ONLY'
            elif in_B and not in_D:
                cls = 'OOD-ONLY'
            else:
                cls = 'EASY'
        classes[uid] = cls
    # Counts
    from collections import Counter
    counts = Counter(classes.values())
    print(f"  class counts: {dict(counts)}", flush=True)

    # PCA basis + project B' features (both query and gallery)
    print("PCA basis on B' gallery...", flush=True)
    mu, V = pca_basis(G_B)
    print("Projecting at j=8, k=all...", flush=True)
    proj_q = project(q_emb_B, mu, V, J_BEST, K_BAND_BEST)
    proj_g = project(G_B, mu, V, J_BEST, K_BAND_BEST)

    # Per-query AP in both conditions
    print("Per-query AP: B' baseline...", flush=True)
    aps_base = per_query_ap(q_emb_B, q_pids_B, q_cids_B, G_B, g_pids, g_cids)
    print("Per-query AP: B' projected...", flush=True)
    aps_proj = per_query_ap(proj_q, q_pids_B, q_cids_B, proj_g, g_pids, g_cids)

    # Aggregate by class (per-identity mean AP)
    from collections import defaultdict
    id_ap_base = defaultdict(list)
    id_ap_proj = defaultdict(list)
    for (p1, a1), (p2, a2) in zip(aps_base, aps_proj):
        if a1 is None or a2 is None: continue
        id_ap_base[p1].append(a1); id_ap_proj[p1].append(a2)
    rows = []
    cls_ap = defaultdict(lambda: {'base': [], 'proj': [], 'delta': []})
    for uid, cls in classes.items():
        if uid not in id_ap_base: continue
        ap_b = float(np.mean(id_ap_base[uid]))
        ap_p = float(np.mean(id_ap_proj[uid]))
        cls_ap[cls]['base'].append(ap_b)
        cls_ap[cls]['proj'].append(ap_p)
        cls_ap[cls]['delta'].append(ap_p - ap_b)
    print(f"\n{'class':<18} {'n':>5} {'mean AP base':>12} {'mean AP proj':>12} {'mean ΔAP':>10}")
    out = {}
    for cls, vals in sorted(cls_ap.items()):
        base = np.mean(vals['base']) if vals['base'] else 0
        proj = np.mean(vals['proj']) if vals['proj'] else 0
        d = np.mean(vals['delta']) if vals['delta'] else 0
        n = len(vals['delta'])
        print(f"{cls:<18} {n:>5} {base*100:>12.2f} {proj*100:>12.2f} {d*100:>10.3f}")
        out[cls] = {'n': int(n),
                    'mean_ap_base_pct': float(base * 100),
                    'mean_ap_proj_pct': float(proj * 100),
                    'mean_dap_pct': float(d * 100),
                    'total_dap_contrib_pct': float(sum(vals['delta']) * 100)}

    # Total ΔAP across all queries
    all_delta = [(a2 - a1) for (p1, a1), (p2, a2) in zip(aps_base, aps_proj)
                 if a1 is not None and a2 is not None]
    print(f"\nTotal mean ΔAP across all queries: {np.mean(all_delta)*100:.3f} pp")
    print(f"Total sum ΔAP (equivalent to mean × n): {sum(all_delta)*100:.2f} pp")

    out['class_counts'] = dict(counts)
    out['total_mean_dap_pct'] = float(np.mean(all_delta) * 100)
    out['total_sum_dap'] = float(sum(all_delta) * 100)
    out['share_of_dap_by_class'] = {
        c: float(sum(cls_ap[c]['delta']) / sum(all_delta)) if sum(all_delta) != 0 else 0
        for c in cls_ap
    }
    with open(RESULTS / 'e31_loopclose.json', 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved → {RESULTS / 'e31_loopclose.json'}")


if __name__ == '__main__':
    main()
