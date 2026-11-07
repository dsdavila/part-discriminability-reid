"""
Paired identity-bootstrap mAP CIs for the P15 dominance table
(Ch2 rev 191 queue). Four (dataset, arm) cells x three mAP series:
  - baseline (no correction)
  - NICDM_k50 (rescaling)
  - k-reciprocal blended (set-expansion)

Per-query AP arrays are computed for each series; id-clustered bootstrap
on query identities yields marginal CIs on mAP and paired CIs on the
two correction-vs-baseline differences and the NICDM-vs-krec difference.
"""
import json, time
import numpy as np
from pathlib import Path
import torch

CACHE_DIR = Path('/tmp/e12_feature_cache')
RESULTS   = Path('/data/papers/low_dimensional_reid/results')
N_BOOT = 500
K_NICDM = 50
K1_KREC = 20
K2_KREC = 6
LAMBDA_KREC = 0.3
_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CELLS = {
    'D':  ('bot_msmt_seed42_msmt.npz',  11659),
    "B'": ('msmt17_test_noREA.npz',     11659),
    "A'": ('market_test_noREA.npz',      3368),
    'C':  ('bot_msmt_seed42_market.npz', 3368),
}

TARGETS = [
    ('MSMT_id',    'D'),
    ('MSMT_ood',   "B'"),
    ('Market_id',  "A'"),
    ('Market_ood', 'C'),
]


def load_qg(cell):
    fn, n_q = CELLS[cell]
    d = np.load(CACHE_DIR / fn); emb = d['emb'].astype(np.float32)
    return {
        'Q': emb[:n_q], 'Qp': d['pids'][:n_q], 'Qc': d['cids'][:n_q],
        'G': emb[n_q:], 'Gp': d['pids'][n_q:], 'Gc': d['cids'][n_q:],
    }


def l2(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(n, 1e-12)


def dist_from_sim(s):
    return np.sqrt(np.clip(2.0 - 2.0 * s, 0.0, None))


def per_query_ap(d_qg, q_pids, q_cids, g_pids, g_cids):
    """Per-query AP under the standard Re-ID convention."""
    Nq = d_qg.shape[0]
    ap = np.full(Nq, np.nan, dtype=np.float32)
    for q in range(Nq):
        qp, qc = q_pids[q], q_cids[q]
        exc = (g_pids == qp) & (g_cids == qc)
        n_pos = int(((g_pids == qp) & ~exc).sum())
        if n_pos == 0:
            continue
        score = -d_qg[q].copy()
        score[exc] = -1e9
        order = np.argsort(-score, kind='stable')
        pos = (g_pids[order] == qp) & ~exc[order]
        cum = np.cumsum(pos)
        ranks = np.arange(1, len(order) + 1)
        prec = cum / ranks
        ap[q] = prec[pos].sum() / n_pos
    return ap


def nicdm_apply(sims_qg, q_pids, q_cids, G_cuda, g_pids, g_cids, k_corr):
    """Apply NICDM correction and return per-query AP array."""
    Ng = G_cuda.shape[0]
    chunk = 256
    # r(i) for every gallery item
    r_g = np.zeros(Ng, dtype=np.float32)
    for start in range(0, Ng, chunk):
        end = min(Ng, start + chunk)
        sims = (G_cuda[start:end] @ G_cuda.T).cpu().numpy()
        for local in range(end - start):
            i = start + local
            row = sims[local].copy()
            exc = (g_pids == g_pids[i]) & (g_cids == g_cids[i])
            row[exc] = -2.0
            idx = np.argpartition(-row, k_corr)[:k_corr]
            d = dist_from_sim(row[idx])
            r_g[i] = float(d.mean())
    # r(q) for every query
    Nq = sims_qg.shape[0]
    r_q = np.zeros(Nq, dtype=np.float32)
    for q in range(Nq):
        row = sims_qg[q].copy()
        exc = (g_pids == q_pids[q]) & (g_cids == q_cids[q])
        row[exc] = -2.0
        idx = np.argpartition(-row, k_corr)[:k_corr]
        d = dist_from_sim(row[idx])
        r_q[q] = float(d.mean())
    d_qg = dist_from_sim(sims_qg)
    rq = np.sqrt(r_q + 1e-12)[:, None]
    rg = np.sqrt(r_g + 1e-12)[None, :]
    d_prime = d_qg / (rq * rg)
    return per_query_ap(d_prime, q_pids, q_cids, g_pids, g_cids)


def topk_combined(X_cuda, k, chunk=256):
    N = X_cuda.shape[0]
    out = np.zeros((N, k), dtype=np.int32)
    for start in range(0, N, chunk):
        end = min(N, start + chunk)
        sims = X_cuda[start:end] @ X_cuda.T
        _, idx = torch.topk(sims, k, dim=1)
        out[start:end] = idx.cpu().numpy().astype(np.int32)
    return out


def k_reciprocal_apply(Q_np, G_np, Q_pids, Q_cids, G_pids, G_cids,
                        k1=K1_KREC, k2=K2_KREC, lam=LAMBDA_KREC):
    """Apply Zhong k-reciprocal and return per-query AP under final blend.
    Memory-efficient: top-k1 lists only, sparse V, inverted index.
    """
    Nq = len(Q_np); Ng = len(G_np)
    all_num = Nq + Ng
    combined = np.concatenate([Q_np, G_np], axis=0).astype(np.float32)
    C_cuda = torch.from_numpy(combined).to(_DEVICE)
    initial_rank = topk_combined(C_cuda, k1 + 1)
    V_rows = [None] * all_num
    for i in range(all_num):
        forward = initial_rank[i, :k1+1]
        backward = initial_rank[forward, :k1+1]
        in_back = (backward == i).any(axis=1)
        k_recip = forward[in_back]
        expansion = set(int(x) for x in k_recip)
        half = int(round(k1 / 2))
        for cand in k_recip:
            cf = initial_rank[cand, :half+1]
            cb = initial_rank[cf, :half+1]
            cib = (cb == cand).any(axis=1)
            ckr = cf[cib]
            if len(ckr) == 0: continue
            inter = len(set(ckr.tolist()) & set(k_recip.tolist()))
            if inter > (2.0/3.0) * len(ckr):
                for x in ckr: expansion.add(int(x))
        cols = np.array(sorted(expansion), dtype=np.int32)
        sims = (C_cuda[i:i+1] @ C_cuda[cols].T).cpu().numpy()[0]
        w = np.exp(-dist_from_sim(sims))
        w = w / max(w.sum(), 1e-12)
        V_rows[i] = (cols, w.astype(np.float32))
    V_new = [None] * all_num
    for i in range(all_num):
        agg = {}
        for n in initial_rank[i, :k2]:
            cols, vals = V_rows[int(n)]
            for c, v in zip(cols.tolist(), vals.tolist()):
                agg[c] = agg.get(c, 0.0) + v / k2
        if agg:
            cols_n = np.array(sorted(agg.keys()), dtype=np.int32)
            vals_n = np.array([agg[c] for c in cols_n.tolist()], dtype=np.float32)
            V_new[i] = (cols_n, vals_n)
        else:
            V_new[i] = (np.array([], dtype=np.int32), np.array([], dtype=np.float32))
    V_rows = V_new
    inv = [[] for _ in range(all_num)]
    for i in range(all_num):
        cols, vals = V_rows[i]
        for c, v in zip(cols, vals):
            inv[int(c)].append((i, float(v)))
    # Jaccard query-to-gallery
    jaccard_qg = np.zeros((Nq, Ng), dtype=np.float32)
    for q in range(Nq):
        cols_q, vals_q = V_rows[q]
        temp_min = np.zeros(Ng, dtype=np.float32)
        if len(cols_q) > 0:
            for c, vq in zip(cols_q.tolist(), vals_q.tolist()):
                for (idx, vj) in inv[int(c)]:
                    if idx >= Nq:
                        temp_min[idx - Nq] += min(vq, vj)
        jaccard_qg[q] = 1.0 - temp_min / (2.0 - temp_min + 1e-12)
    sims_qg = (C_cuda[:Nq] @ C_cuda[Nq:].T).cpu().numpy()
    orig_qg = dist_from_sim(sims_qg)
    final = (1 - lam) * jaccard_qg + lam * orig_qg
    del C_cuda
    torch.cuda.empty_cache()
    return per_query_ap(final, Q_pids, Q_cids, G_pids, G_cids)


def _ci_mean(ap_subset):
    m = ~np.isnan(ap_subset)
    return float(ap_subset[m].mean()) if m.any() else 0.0


def id_boot_pairs(ap_arrays, q_pids, n_boot, rng):
    """Paired identity-clustered bootstrap. ap_arrays is a dict name->ap (Nq,).
    Returns marginals + pairwise diffs."""
    unique = np.unique(q_pids)
    pid_to_idx = {int(p): np.where(q_pids == p)[0] for p in unique}
    names = list(ap_arrays.keys())
    marginals = {n: [] for n in names}
    diffs = {}
    for a in names:
        for b in names:
            if a != b:
                diffs[(a, b)] = []
    for _ in range(n_boot):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([pid_to_idx[int(p)] for p in sampled])
        vals = {n: _ci_mean(ap_arrays[n][idx]) for n in names}
        for n in names:
            marginals[n].append(vals[n])
        for (a, b) in diffs:
            diffs[(a, b)].append(vals[a] - vals[b])
    def _summ(arr):
        a = np.array(arr)
        return {'median': float(np.median(a)),
                'lo': float(np.quantile(a, 0.025)),
                'hi': float(np.quantile(a, 0.975))}
    out = {'marginals': {n: _summ(marginals[n]) for n in names}}
    for (a, b), vs in diffs.items():
        out[f'{a}_minus_{b}'] = _summ(vs)
    return out


def main():
    print(f"Device: {_DEVICE}", flush=True)
    out = {}
    for tag, cell in TARGETS:
        print(f"\n### {tag} ###", flush=True)
        data = load_qg(cell)
        Q = l2(data['Q']); G = l2(data['G'])
        Qp = data['Qp']; Qc = data['Qc']; Gp = data['Gp']; Gc = data['Gc']
        print(f"  Nq={len(Q)}  Ng={len(G)}", flush=True)

        G_cuda = torch.from_numpy(G).to(_DEVICE)
        Q_cuda = torch.from_numpy(Q).to(_DEVICE)

        # Baseline per-query AP
        t0 = time.time()
        sims_qg = (Q_cuda @ G_cuda.T).cpu().numpy()
        d_base = dist_from_sim(sims_qg)
        ap_base = per_query_ap(d_base, Qp, Qc, Gp, Gc)
        print(f"  baseline AP done t={time.time()-t0:.1f}s  mAP={_ci_mean(ap_base)*100:.3f}", flush=True)

        # NICDM_k50 per-query AP
        t0 = time.time()
        ap_nicdm = nicdm_apply(sims_qg, Qp, Qc, G_cuda, Gp, Gc, K_NICDM)
        print(f"  NICDM_k50 AP done t={time.time()-t0:.1f}s  mAP={_ci_mean(ap_nicdm)*100:.3f}", flush=True)

        del Q_cuda, G_cuda
        torch.cuda.empty_cache()

        # k-reciprocal per-query AP
        t0 = time.time()
        ap_krec = k_reciprocal_apply(Q, G, Qp, Qc, Gp, Gc)
        print(f"  k-reciprocal AP done t={time.time()-t0:.1f}s  mAP={_ci_mean(ap_krec)*100:.3f}", flush=True)

        # Bootstrap
        rng = np.random.default_rng(42)
        t0 = time.time()
        res = id_boot_pairs({'baseline': ap_base, 'NICDM_k50': ap_nicdm,
                              'k_reciprocal': ap_krec}, Qp, N_BOOT, rng)
        print(f"  bootstrap done t={time.time()-t0:.1f}s", flush=True)

        for n, s in res['marginals'].items():
            print(f"    {n:<16} mAP% = {s['median']*100:.3f} [{s['lo']*100:.3f}, {s['hi']*100:.3f}]", flush=True)
        for key in ['NICDM_k50_minus_baseline', 'k_reciprocal_minus_baseline', 'k_reciprocal_minus_NICDM_k50']:
            s = res[key]
            print(f"    delta {key} pp = {s['median']*100:+.3f} [{s['lo']*100:+.3f}, {s['hi']*100:+.3f}]", flush=True)

        out[tag] = res

    out_path = RESULTS / 'e43_paired_map_bootstrap.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == '__main__':
    main()
