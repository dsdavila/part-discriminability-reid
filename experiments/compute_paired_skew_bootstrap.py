"""
Paired id-minus-ood skewness bootstrap (Ch2 rev 183).

Same gallery, same identity set, two arms (id and ood) under two
corrections (NICDM_k50 on ood arm, k-reciprocal blended on ood arm).
Resample identities once per draw; compute both arms' k=10
skewness on the resampled items; interval the difference.

Corrections applied:
  - id arm: baseline (no correction)
  - ood arm: NICDM_k50 (rescaling) OR k-reciprocal blended (set-expansion)

Loads if present, else recomputes the per-item kocc arrays and
caches them to npz for reuse.
"""
import json, time
import numpy as np
from pathlib import Path
from scipy import stats
import torch

CACHE_DIR = Path('/tmp/e12_feature_cache')
RESULTS   = Path('/data/papers/low_dimensional_reid/results')
KOCC_CACHE = RESULTS / 'kocc_arrays_cache'
KOCC_CACHE.mkdir(exist_ok=True)
K_OCC = 10
N_BOOT = 500
_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

K_NICDM = 50
K1_KREC = 20
K2_KREC = 6
LAMBDA_KREC = 0.3

CELLS = {
    'D':  ('bot_msmt_seed42_msmt.npz',  11659),
    "B'": ('msmt17_test_noREA.npz',     11659),
    "A'": ('market_test_noREA.npz',      3368),
    'C':  ('bot_msmt_seed42_market.npz', 3368),
}

DATASETS = {
    'MSMT':   {'id_cell': 'D',  'ood_cell': "B'"},
    'Market': {'id_cell': "A'", 'ood_cell': 'C'},
}


def load_gallery(cell):
    fn, n_q = CELLS[cell]
    d = np.load(CACHE_DIR / fn); emb = d['emb'].astype(np.float32)
    return emb[n_q:], d['pids'][n_q:], d['cids'][n_q:]


def l2(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(n, 1e-12)


def dist_from_sim(s):
    return np.sqrt(np.clip(2.0 - 2.0 * s, 0.0, None))


def compute_baseline_kocc(G_np, g_pids, g_cids, chunk=256):
    """Baseline (no correction) k=10 occurrence over gallery."""
    G_cuda = torch.from_numpy(G_np).to(_DEVICE)
    Ng = len(G_np)
    kocc = np.zeros(Ng, dtype=np.int64)
    for start in range(0, Ng, chunk):
        end = min(Ng, start + chunk)
        sims = (G_cuda[start:end] @ G_cuda.T).cpu().numpy()
        for local in range(end - start):
            i = start + local
            row = sims[local].copy()
            exc = (g_pids == g_pids[i]) & (g_cids == g_cids[i])
            row[exc] = -2.0
            idx = np.argpartition(-row, K_OCC)[:K_OCC]
            kocc[idx] += 1
    del G_cuda
    torch.cuda.empty_cache()
    return kocc


def compute_nicdm_kocc(G_np, g_pids, g_cids, k_corr=K_NICDM, chunk=256):
    """NICDM-corrected k=10 occurrence over gallery."""
    G_cuda = torch.from_numpy(G_np).to(_DEVICE)
    Ng = len(G_np)
    # r(i) = mean distance to k-NN (excluding same-cam-same-pid)
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
    del G_cuda
    torch.cuda.empty_cache()
    return kocc


def compute_krec_blended_kocc(G_np, g_pids, g_cids, k1=K1_KREC, k2=K2_KREC,
                                 lam=LAMBDA_KREC, chunk=256):
    """k-reciprocal blended top-10 kocc over gallery."""
    G_cuda = torch.from_numpy(G_np).to(_DEVICE)
    Ng = len(G_np)
    # top-k1+1 initial rank
    initial_rank = np.zeros((Ng, k1 + 1), dtype=np.int32)
    for start in range(0, Ng, chunk):
        end = min(Ng, start + chunk)
        sims = G_cuda[start:end] @ G_cuda.T
        _, idx = torch.topk(sims, k1 + 1, dim=1)
        initial_rank[start:end] = idx.cpu().numpy().astype(np.int32)
    # Build V
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
    # Local QE
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
    # Inverted index
    inv = [[] for _ in range(Ng)]
    for i in range(Ng):
        cols, vals = V_rows[i]
        for c, v in zip(cols, vals):
            inv[int(c)].append((i, float(v)))
    # Blended top-10
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
    del G_cuda
    torch.cuda.empty_cache()
    return kocc


def paired_skew_bootstrap(kocc_id, kocc_ood, pids, n_boot, rng):
    """Paired identity-clustered bootstrap: for each draw, resample identities,
    aggregate items, compute (skew(id[idx]) - skew(ood[idx])), interval it.

    Reports the paired difference and both marginal series.
    """
    unique = np.unique(pids)
    pid_to_idx = {int(p): np.where(pids == p)[0] for p in unique}
    diffs = []
    id_boots = []
    ood_boots = []
    for b in range(n_boot):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([pid_to_idx[int(p)] for p in sampled])
        s_id = float(stats.skew(kocc_id[idx].astype(np.float32)))
        s_ood = float(stats.skew(kocc_ood[idx].astype(np.float32)))
        id_boots.append(s_id)
        ood_boots.append(s_ood)
        diffs.append(s_id - s_ood)
    return {
        'id_skew_point': float(stats.skew(kocc_id.astype(np.float32))),
        'ood_skew_point': float(stats.skew(kocc_ood.astype(np.float32))),
        'id_minus_ood_point': float(stats.skew(kocc_id.astype(np.float32))
                                      - stats.skew(kocc_ood.astype(np.float32))),
        'id_marginal_ci': {'lo': float(np.quantile(id_boots, 0.025)),
                             'hi': float(np.quantile(id_boots, 0.975))},
        'ood_marginal_ci': {'lo': float(np.quantile(ood_boots, 0.025)),
                              'hi': float(np.quantile(ood_boots, 0.975))},
        'id_minus_ood_ci': {'median': float(np.median(diffs)),
                              'lo': float(np.quantile(diffs, 0.025)),
                              'hi': float(np.quantile(diffs, 0.975))},
    }


def get_kocc(label, computation_fn, *args, **kwargs):
    """Cache to npz to avoid recomputation on reruns."""
    cache = KOCC_CACHE / f'{label}.npz'
    if cache.exists():
        print(f"  [cached] {label}", flush=True)
        return np.load(cache)['kocc']
    t0 = time.time()
    kocc = computation_fn(*args, **kwargs)
    np.savez(cache, kocc=kocc)
    print(f"  [computed] {label} t={time.time()-t0:.1f}s saved to {cache.name}", flush=True)
    return kocc


def main():
    print(f"Device: {_DEVICE}", flush=True)
    out = {}
    for ds_name, cfg in DATASETS.items():
        print(f"\n### {ds_name} ###", flush=True)
        G_id_np, g_pids, g_cids = load_gallery(cfg['id_cell'])
        G_ood_np, _, _ = load_gallery(cfg['ood_cell'])
        G_id = l2(G_id_np); G_ood = l2(G_ood_np)
        print(f"  Ng={len(G_id)}  n_ids={len(np.unique(g_pids))}", flush=True)

        kocc_id_base = get_kocc(f'{ds_name}_id_baseline',
                                 compute_baseline_kocc, G_id, g_pids, g_cids)
        kocc_ood_nicdm = get_kocc(f'{ds_name}_ood_nicdm_k50',
                                    compute_nicdm_kocc, G_ood, g_pids, g_cids)
        kocc_ood_krec = get_kocc(f'{ds_name}_ood_krec_blended',
                                   compute_krec_blended_kocc, G_ood, g_pids, g_cids)

        rng = np.random.default_rng(42)
        print(f"  paired bootstraps (n={N_BOOT})...", flush=True)
        t0 = time.time()
        r_nicdm = paired_skew_bootstrap(kocc_id_base, kocc_ood_nicdm, g_pids,
                                          N_BOOT, rng)
        r_krec = paired_skew_bootstrap(kocc_id_base, kocc_ood_krec, g_pids,
                                         N_BOOT, np.random.default_rng(42))
        print(f"    t={time.time()-t0:.1f}s", flush=True)

        out[ds_name] = {'vs_NICDM_k50': r_nicdm, 'vs_k_reciprocal_blended': r_krec}

        print(f"\n  === {ds_name} id_baseline - ood_NICDM_k50 ===")
        print(f"    id   point={r_nicdm['id_skew_point']:.4f}  CI_marginal=[{r_nicdm['id_marginal_ci']['lo']:.4f}, {r_nicdm['id_marginal_ci']['hi']:.4f}]")
        print(f"    ood  point={r_nicdm['ood_skew_point']:.4f}  CI_marginal=[{r_nicdm['ood_marginal_ci']['lo']:.4f}, {r_nicdm['ood_marginal_ci']['hi']:.4f}]")
        print(f"    diff point={r_nicdm['id_minus_ood_point']:+.4f}  CI_paired=[{r_nicdm['id_minus_ood_ci']['lo']:+.4f}, {r_nicdm['id_minus_ood_ci']['hi']:+.4f}]")

        print(f"\n  === {ds_name} id_baseline - ood_k_reciprocal_blended ===")
        print(f"    id   point={r_krec['id_skew_point']:.4f}  CI_marginal=[{r_krec['id_marginal_ci']['lo']:.4f}, {r_krec['id_marginal_ci']['hi']:.4f}]")
        print(f"    ood  point={r_krec['ood_skew_point']:.4f}  CI_marginal=[{r_krec['ood_marginal_ci']['lo']:.4f}, {r_krec['ood_marginal_ci']['hi']:.4f}]")
        print(f"    diff point={r_krec['id_minus_ood_point']:+.4f}  CI_paired=[{r_krec['id_minus_ood_ci']['lo']:+.4f}, {r_krec['id_minus_ood_ci']['hi']:+.4f}]")

    out_path = RESULTS / 'e43_paired_skew_bootstrap.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == '__main__':
    main()
