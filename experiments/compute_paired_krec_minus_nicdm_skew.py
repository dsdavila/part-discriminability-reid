"""
Paired identity-clustered bootstrap of (k-rec post-skew - NICDM_k50 post-skew)
per arm (Ch2 rev 194).

Shared identity resample removes shared variance algebraically rather than
after the fact. Provides the direct interval on the dominance claim's skew
axis across all four (dataset, arm) combinations.

Loads per-item kocc arrays from the npz cache (saved by
compute_paired_skew_bootstrap.py) when available; otherwise recomputes.
"""
import json, time
import numpy as np
from pathlib import Path
from scipy import stats
import torch

CACHE_DIR = Path('/tmp/e12_feature_cache')
RESULTS   = Path('/data/papers/low_dimensional_reid/results')
KOCC_CACHE = RESULTS / 'kocc_arrays_cache'
N_BOOT = 500
K_OCC = 10
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


def load_gallery(cell):
    fn, n_q = CELLS[cell]
    d = np.load(CACHE_DIR / fn); emb = d['emb'].astype(np.float32)
    return emb[n_q:], d['pids'][n_q:], d['cids'][n_q:]


def l2(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(n, 1e-12)


def dist_from_sim(s):
    return np.sqrt(np.clip(2.0 - 2.0 * s, 0.0, None))


def compute_nicdm_kocc(G_cuda, g_pids, g_cids, k_corr, chunk=256):
    Ng = G_cuda.shape[0]
    r = np.zeros(Ng, dtype=np.float32)
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
            r[i] = float(d.mean())
    inv_r = 1.0 / np.sqrt(r + 1e-12)
    kocc = np.zeros(Ng, dtype=np.int64)
    for start in range(0, Ng, chunk):
        end = min(Ng, start + chunk)
        sims = (G_cuda[start:end] @ G_cuda.T).cpu().numpy()
        for local in range(end - start):
            i = start + local
            row = sims[local].copy()
            exc = (g_pids == g_pids[i]) & (g_cids == g_cids[i])
            d = dist_from_sim(row)
            d_prime = d * inv_r
            d_prime[exc] = 1e9
            idx = np.argpartition(d_prime, K_OCC)[:K_OCC]
            kocc[idx] += 1
    return kocc


def topk_gallery(G_cuda, k, chunk=256):
    N = G_cuda.shape[0]
    out = np.zeros((N, k), dtype=np.int32)
    for start in range(0, N, chunk):
        end = min(N, start + chunk)
        sims = G_cuda[start:end] @ G_cuda.T
        _, idx = torch.topk(sims, k, dim=1)
        out[start:end] = idx.cpu().numpy().astype(np.int32)
    return out


def compute_krec_blended_kocc(G_cuda, g_pids, g_cids, k1=K1_KREC, k2=K2_KREC,
                                 lam=LAMBDA_KREC):
    Ng = G_cuda.shape[0]
    initial_rank = topk_gallery(G_cuda, k1 + 1)
    V_rows = [None] * Ng
    for i in range(Ng):
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
        sims = (G_cuda[i:i+1] @ G_cuda[cols].T).cpu().numpy()[0]
        w = np.exp(-dist_from_sim(sims))
        w = w / max(w.sum(), 1e-12)
        V_rows[i] = (cols, w.astype(np.float32))
    V_new = [None] * Ng
    for i in range(Ng):
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
    inv = [[] for _ in range(Ng)]
    for i in range(Ng):
        cols, vals = V_rows[i]
        for c, v in zip(cols, vals):
            inv[int(c)].append((i, float(v)))
    kocc = np.zeros(Ng, dtype=np.int64)
    for i in range(Ng):
        cols_q, vals_q = V_rows[i]
        temp_min = np.zeros(Ng, dtype=np.float32)
        if len(cols_q) > 0:
            for c, vq in zip(cols_q.tolist(), vals_q.tolist()):
                for (idx, vj) in inv[int(c)]:
                    temp_min[idx] += min(vq, vj)
        jac = 1.0 - temp_min / (2.0 - temp_min + 1e-12)
        sims_row = (G_cuda[i:i+1] @ G_cuda.T).cpu().numpy()[0]
        d_orig = dist_from_sim(sims_row)
        blend = lam * d_orig + (1.0 - lam) * jac
        exc = (g_pids == g_pids[i]) & (g_cids == g_cids[i])
        blend[exc] = 1e9
        idx_top = np.argpartition(blend, K_OCC)[:K_OCC]
        kocc[idx_top] += 1
    return kocc


def get_or_compute(label, fn, *args, **kwargs):
    cache = KOCC_CACHE / f'{label}.npz'
    if cache.exists():
        print(f"  [cached] {label}", flush=True)
        return np.load(cache)['kocc']
    t0 = time.time()
    kocc = fn(*args, **kwargs)
    np.savez(cache, kocc=kocc)
    print(f"  [computed] {label} t={time.time()-t0:.1f}s", flush=True)
    return kocc


def main():
    print(f"Device: {_DEVICE}", flush=True)
    out = {}
    for tag, cell in TARGETS:
        print(f"\n### {tag} ###", flush=True)
        G_np, g_pids, g_cids = load_gallery(cell)
        G = l2(G_np)
        G_cuda = torch.from_numpy(G).to(_DEVICE)

        # Match the naming convention from compute_paired_skew_bootstrap.py so
        # cached arrays are reused across scripts.
        ds, arm = tag.rsplit('_', 1)
        nicdm_label = f'{ds}_{arm}_nicdm_k50' if arm == 'ood' else f'{ds}_{tag.split("_")[1]}_nicdm_k50_{arm}'
        krec_label = f'{ds}_{arm}_krec_blended' if arm == 'ood' else f'{ds}_{tag.split("_")[1]}_krec_blended_{arm}'
        # Simpler: unique labels per (dataset, arm)
        nicdm_label = f'{tag}_nicdm_k50'
        krec_label = f'{tag}_krec_blended'

        kocc_nicdm = get_or_compute(nicdm_label,
                                     compute_nicdm_kocc, G_cuda, g_pids, g_cids, K_NICDM)
        kocc_krec = get_or_compute(krec_label,
                                    compute_krec_blended_kocc, G_cuda, g_pids, g_cids)

        del G_cuda
        torch.cuda.empty_cache()

        # Paired bootstrap: resample identities once, compute skew of both
        # kocc arrays on the same resample, diff is k-rec minus NICDM.
        rng = np.random.default_rng(42)
        unique = np.unique(g_pids)
        pid_to_idx = {int(p): np.where(g_pids == p)[0] for p in unique}
        diffs = []
        krec_marg = []
        nicdm_marg = []
        for _ in range(N_BOOT):
            sampled = rng.choice(unique, size=len(unique), replace=True)
            idx = np.concatenate([pid_to_idx[int(p)] for p in sampled])
            s_krec = float(stats.skew(kocc_krec[idx].astype(np.float32)))
            s_nicdm = float(stats.skew(kocc_nicdm[idx].astype(np.float32)))
            diffs.append(s_krec - s_nicdm)
            krec_marg.append(s_krec)
            nicdm_marg.append(s_nicdm)
        arr = np.array(diffs)
        krec_arr = np.array(krec_marg)
        nicdm_arr = np.array(nicdm_marg)
        out[tag] = {
            'krec_post_skew_point': float(stats.skew(kocc_krec.astype(np.float32))),
            'nicdm_post_skew_point': float(stats.skew(kocc_nicdm.astype(np.float32))),
            'diff_point': float(stats.skew(kocc_krec.astype(np.float32))
                                 - stats.skew(kocc_nicdm.astype(np.float32))),
            'krec_marginal': {'median': float(np.median(krec_arr)),
                                'lo': float(np.quantile(krec_arr, 0.025)),
                                'hi': float(np.quantile(krec_arr, 0.975))},
            'nicdm_marginal': {'median': float(np.median(nicdm_arr)),
                                 'lo': float(np.quantile(nicdm_arr, 0.025)),
                                 'hi': float(np.quantile(nicdm_arr, 0.975))},
            'paired_krec_minus_nicdm': {'median': float(np.median(arr)),
                                          'lo': float(np.quantile(arr, 0.025)),
                                          'hi': float(np.quantile(arr, 0.975))},
        }
        print(f"  nicdm post-skew: {out[tag]['nicdm_post_skew_point']:.4f}  CI_marginal=[{out[tag]['nicdm_marginal']['lo']:.4f}, {out[tag]['nicdm_marginal']['hi']:.4f}]")
        print(f"  krec  post-skew: {out[tag]['krec_post_skew_point']:.4f}  CI_marginal=[{out[tag]['krec_marginal']['lo']:.4f}, {out[tag]['krec_marginal']['hi']:.4f}]")
        print(f"  **paired k-rec - NICDM**: {out[tag]['diff_point']:+.4f}  CI=[{out[tag]['paired_krec_minus_nicdm']['lo']:+.4f}, {out[tag]['paired_krec_minus_nicdm']['hi']:+.4f}]")

    out_path = RESULTS / 'e43_paired_krec_minus_nicdm_skew.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == '__main__':
    main()
