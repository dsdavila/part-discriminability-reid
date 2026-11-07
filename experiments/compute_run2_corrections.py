"""
Run 2: hubness-reduction corrections (Ch2 rev 162 spec).

Three methods: NICDM, Local Scaling (LS), k-reciprocal re-ranking.
Correction k in {10, 20, 50}.

Four readouts per (dataset, arm, method, k_corr) — in the standing
report order (ruling: graph shape first, mAP second):
  1. item-level k=10 occurrence skewness (gallery self-neighbourhood)
  2. symmetric 10-NN fraction (fraction of pairs i,j with i in topk(j) AND j in topk(i))
  3. mAP (query -> gallery, standard Re-ID eval)
  4. |r_global| = Spearman of (k-occurrence, distance to global centroid)
     — identity-clustered bootstrap CIs retained (ruling 19, 4)

Report absolute AND normalised ((id - ood)/id) values together (ruling 26).

Pre-registration (rev 162):
- MSMT ood mAP delta near-null (< ~2 pp); > 2 pp refutes the ceiling argument.
- id mAP delta expected small (ordering from literature: methods help more where skew is higher).
- No absolute threshold claims (ruling 27).

k-reciprocal implementation: Zhong et al. 2017; Jaccard on expanded reciprocal sets.
"""
import json, time, sys
import numpy as np
from pathlib import Path
from scipy import stats
from sym_10nn import symmetric_knn_fraction

try:
    import torch
    _HAS_TORCH = True
    _DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
except ImportError:
    _HAS_TORCH = False
    _DEVICE = None

CACHE_DIR = Path('/tmp/e12_feature_cache')
RESULTS   = Path('/data/papers/low_dimensional_reid/results')
K_OCC = 10
K_CORR_VALUES = [10, 20, 50]
N_BOOT_R = 200

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


def sim_to_dist(sim):
    """For L2-normalised vectors, d^2 = 2 - 2*sim, so d = sqrt(max(2 - 2*sim, 0))."""
    return np.sqrt(np.clip(2.0 - 2.0 * sim, 0.0, None))


def chunked_gallery_rows(G_cuda, chunk=256):
    """Yield (start, end, sims_chunk) where sims_chunk = G[start:end] @ G.T on GPU."""
    Ng = G_cuda.shape[0]
    for start in range(0, Ng, chunk):
        end = min(Ng, start + chunk)
        sims = (G_cuda[start:end] @ G_cuda.T).cpu().numpy()
        yield start, end, sims


def compute_r_nicdm(G_cuda, g_pids, g_cids, k_corr, chunk=256):
    """r(i) = mean distance from i to its k-NN (excluding same-cam-same-pid, and self).
    Uses cosine-derived distance d = sqrt(2-2sim).
    """
    Ng = G_cuda.shape[0]
    r = np.zeros(Ng, dtype=np.float32)
    for start, end, sims in chunked_gallery_rows(G_cuda, chunk):
        for local in range(end - start):
            i = start + local
            row = sims[local].copy()
            # exclude same-cam-same-pid + self
            exc = (g_pids == g_pids[i]) & (g_cids == g_cids[i])
            row[exc] = -2.0
            idx = np.argpartition(-row, k_corr)[:k_corr]
            d = sim_to_dist(row[idx])
            r[i] = float(d.mean())
    return r


def compute_r_ls(G_cuda, g_pids, g_cids, k_corr, chunk=256):
    """sigma(i) = distance from i to its k-th nearest neighbour."""
    Ng = G_cuda.shape[0]
    r = np.zeros(Ng, dtype=np.float32)
    for start, end, sims in chunked_gallery_rows(G_cuda, chunk):
        for local in range(end - start):
            i = start + local
            row = sims[local].copy()
            exc = (g_pids == g_pids[i]) & (g_cids == g_cids[i])
            row[exc] = -2.0
            idx = np.argpartition(-row, k_corr)[:k_corr]
            d = sim_to_dist(row[idx])
            # k-th neighbour = largest distance among the k nearest
            r[i] = float(d.max())
    return r


def compute_r_q_nicdm(Q_cuda, G_cuda, q_pids, q_cids, g_pids, g_cids, k_corr):
    """r(q) for each query, mean distance to k-NN gallery items (same exclusions)."""
    sims = (Q_cuda @ G_cuda.T).cpu().numpy()
    Nq = sims.shape[0]
    r = np.zeros(Nq, dtype=np.float32)
    for q in range(Nq):
        row = sims[q].copy()
        exc = (g_pids == q_pids[q]) & (g_cids == q_cids[q])
        row[exc] = -2.0
        idx = np.argpartition(-row, k_corr)[:k_corr]
        d = sim_to_dist(row[idx])
        r[q] = float(d.mean())
    return r, sims


def compute_r_q_ls(Q_cuda, G_cuda, q_pids, q_cids, g_pids, g_cids, k_corr):
    sims = (Q_cuda @ G_cuda.T).cpu().numpy()
    Nq = sims.shape[0]
    r = np.zeros(Nq, dtype=np.float32)
    for q in range(Nq):
        row = sims[q].copy()
        exc = (g_pids == q_pids[q]) & (g_cids == q_cids[q])
        row[exc] = -2.0
        idx = np.argpartition(-row, k_corr)[:k_corr]
        d = sim_to_dist(row[idx])
        r[q] = float(d.max())
    return r, sims


def eval_post_correction_gallery(G_cuda, g_pids, g_cids, r_g, method,
                                   chunk=256, k_readout=K_OCC):
    """For each gallery item i, compute corrected top-k_readout in gallery.
    Returns:
      kocc (Ng,) - item occurrence count
      top_nb (Ng, k_readout) - int array of corrected neighbour indices
    Correction:
      NICDM: d'(i,j) = d(i,j) / sqrt(r(i) r(j)) => rank by d' ascending
      LS:    d'(i,j) = d(i,j)^2 / (r(i) r(j))  => rank by d' ascending
    Both are monotone in d(i,j) for fixed i, so the ordering is
    determined by d(i,j) / r(j)  (NICDM) or d(i,j)^2 / r(j) (LS, same ordering)
    => effective per-candidate penalty = d(i,j) / r(j) (ordering equivalent).
    """
    Ng = G_cuda.shape[0]
    kocc = np.zeros(Ng, dtype=np.int64)
    top_nb = np.zeros((Ng, k_readout), dtype=np.int32)
    inv_r = 1.0 / np.sqrt(r_g + 1e-12)  # for scaling candidates
    for start, end, sims in chunked_gallery_rows(G_cuda, chunk):
        for local in range(end - start):
            i = start + local
            row = sims[local].copy()
            exc = (g_pids == g_pids[i]) & (g_cids == g_cids[i])
            d = sim_to_dist(row)
            # corrected per-candidate score: ascending d' => descending negative d'
            if method == 'NICDM':
                d_prime = d * inv_r  # = d(i,j) / sqrt(r(j)) (constant 1/sqrt(r(i)) factored)
            elif method == 'LS':
                d_prime = (d * d) * inv_r  # = d^2 / sqrt(r(j)) (constant 1/sqrt(r(i)) factored)
            else:
                raise ValueError(method)
            d_prime[exc] = 1e9  # exclude
            idx = np.argpartition(d_prime, k_readout)[:k_readout]
            order = idx[np.argsort(d_prime[idx])]
            top_nb[i] = order
            kocc[order] += 1
    return kocc, top_nb


def dist_to_centroid(G_np):
    c = G_np.mean(axis=0, keepdims=True)
    return np.linalg.norm(G_np - c, axis=1)


def r_global_with_boot(kocc, dist, pids, n_boot, rng):
    r_point, _ = stats.spearmanr(kocc, dist)
    pid_to_items = {int(p): np.where(pids == p)[0] for p in np.unique(pids)}
    unique = list(pid_to_items.keys())
    boots = []
    for b in range(n_boot):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([pid_to_items[int(p)] for p in sampled])
        r, _ = stats.spearmanr(kocc[idx], dist[idx])
        boots.append(float(r))
    arr = np.array(boots)
    return {
        'r_point': float(r_point),
        'r_boot_median': float(np.median(arr)),
        'r_boot_lo': float(np.quantile(arr, 0.025)),
        'r_boot_hi': float(np.quantile(arr, 0.975)),
    }


def mAP_corrected(sims_qg, q_pids, q_cids, g_pids, g_cids, r_q, r_g, method):
    """Compute mAP using corrected scoring d'(q, g) (same correction formula).
    Returns mAP (fraction, not pct)."""
    Nq = sims_qg.shape[0]
    d_qg_base = sim_to_dist(sims_qg)
    if method == 'NICDM':
        # d'(q, g) = d(q, g) / sqrt(r(q) r(g))
        rq = np.sqrt(r_q + 1e-12)[:, None]
        rg = np.sqrt(r_g + 1e-12)[None, :]
        d_prime = d_qg_base / (rq * rg)
    elif method == 'LS':
        rq = np.sqrt(r_q + 1e-12)[:, None]
        rg = np.sqrt(r_g + 1e-12)[None, :]
        d_prime = (d_qg_base * d_qg_base) / (rq * rg)
    elif method == 'baseline':
        d_prime = d_qg_base
    else:
        raise ValueError(method)
    # Score = -d_prime (higher better). Sort descending.
    ap_sum = 0.0; n_valid = 0
    for q in range(Nq):
        qp, qc = q_pids[q], q_cids[q]
        exc = (g_pids == qp) & (g_cids == qc)
        n_pos = int(((g_pids == qp) & ~exc).sum())
        if n_pos == 0:
            continue
        score = -d_prime[q].copy()
        score[exc] = -1e9
        order = np.argsort(-score, kind='stable')
        pos = (g_pids[order] == qp) & ~exc[order]
        cum = np.cumsum(pos)
        ranks = np.arange(1, len(order) + 1)
        prec = cum / ranks
        ap = prec[pos].sum() / n_pos
        ap_sum += ap
        n_valid += 1
    return ap_sum / n_valid if n_valid > 0 else 0.0


def process_one(dataset, arm_name, cell):
    print(f"\n### {dataset} / {arm_name} ###", flush=True)
    data = load_qg(cell)
    Q = l2(data['Q']); G = l2(data['G'])
    Qp = data['Qp']; Qc = data['Qc']
    Gp = data['Gp']; Gc = data['Gc']
    Ng = len(G); Nq = len(Q)
    print(f"  Ng={Ng}  Nq={Nq}", flush=True)

    # GPU tensors
    Q_cuda = torch.from_numpy(Q).to(_DEVICE)
    G_cuda = torch.from_numpy(G).to(_DEVICE)

    results = {'baseline': {}}
    rng = np.random.default_rng(42)

    # --- Baseline ---
    t0 = time.time()
    # Baseline gallery kocc & top-10 (no correction)
    kocc_base = np.zeros(Ng, dtype=np.int64)
    top_nb_base = np.zeros((Ng, K_OCC), dtype=np.int32)
    for start, end, sims in chunked_gallery_rows(G_cuda):
        for local in range(end - start):
            i = start + local
            row = sims[local].copy()
            exc = (Gp == Gp[i]) & (Gc == Gc[i])
            row[exc] = -2.0
            idx = np.argpartition(-row, K_OCC)[:K_OCC]
            order = idx[np.argsort(-row[idx])]
            top_nb_base[i] = order
            kocc_base[order] += 1
    skew_base = float(stats.skew(kocc_base))
    sym_base = symmetric_knn_fraction(top_nb_base, k=K_OCC)
    d_global = dist_to_centroid(G)
    r_info_base = r_global_with_boot(kocc_base, d_global, Gp, N_BOOT_R,
                                      np.random.default_rng(42))
    sims_qg = (Q_cuda @ G_cuda.T).cpu().numpy()
    mAP_base = mAP_corrected(sims_qg, Qp, Qc, Gp, Gc, None, None, 'baseline')
    results['baseline'] = {
        'skew_items': skew_base,
        'sym_10nn_frac': float(sym_base),
        'r_global': r_info_base,
        'mAP_pct': float(mAP_base * 100.0),
    }
    print(f"  baseline: skew={skew_base:.3f}  sym10NN={sym_base:.3f}  "
          f"|r|={abs(r_info_base['r_point']):.3f}  mAP={mAP_base*100:.3f}  "
          f"t={time.time()-t0:.1f}s", flush=True)

    # --- NICDM + LS, each k ---
    for method in ['NICDM', 'LS']:
        for k_corr in K_CORR_VALUES:
            t0 = time.time()
            if method == 'NICDM':
                r_g = compute_r_nicdm(G_cuda, Gp, Gc, k_corr)
                r_q, sims_qg_fresh = compute_r_q_nicdm(Q_cuda, G_cuda, Qp, Qc,
                                                        Gp, Gc, k_corr)
            else:
                r_g = compute_r_ls(G_cuda, Gp, Gc, k_corr)
                r_q, sims_qg_fresh = compute_r_q_ls(Q_cuda, G_cuda, Qp, Qc,
                                                     Gp, Gc, k_corr)
            kocc, top_nb = eval_post_correction_gallery(
                G_cuda, Gp, Gc, r_g, method
            )
            skew = float(stats.skew(kocc))
            sym = symmetric_knn_fraction(top_nb, k=K_OCC)
            r_info = r_global_with_boot(kocc, d_global, Gp, N_BOOT_R,
                                         np.random.default_rng(42 + k_corr))
            mAP = mAP_corrected(sims_qg_fresh, Qp, Qc, Gp, Gc, r_q, r_g, method)
            key = f'{method}_k{k_corr}'
            results[key] = {
                'skew_items': skew,
                'sym_10nn_frac': float(sym),
                'r_global': r_info,
                'mAP_pct': float(mAP * 100.0),
            }
            print(f"  {key}: skew={skew:.3f} (Δ {skew-skew_base:+.3f})  "
                  f"sym10NN={sym:.3f} (Δ {sym-sym_base:+.3f})  "
                  f"|r|={abs(r_info['r_point']):.3f} (Δ {abs(r_info['r_point'])-abs(r_info_base['r_point']):+.3f})  "
                  f"mAP={mAP*100:.3f} (Δ {(mAP - mAP_base)*100:+.3f} pp)  "
                  f"t={time.time()-t0:.1f}s", flush=True)

    # Free GPU
    del Q_cuda, G_cuda
    if _HAS_TORCH:
        torch.cuda.empty_cache()

    return results


def main():
    if not _HAS_TORCH:
        print("ERROR: torch not available"); sys.exit(1)
    print(f"Using device: {_DEVICE}", flush=True)

    out = {}
    for ds_name, cfg in DATASETS.items():
        out[ds_name] = {}
        for arm, cell in [('id', cfg['id']), ('ood', cfg['ood'])]:
            out[ds_name][arm] = process_one(ds_name, arm, cell)

    # Report tables
    print("\n========== RUN 2 SUMMARY ==========")
    print("Order per rev 162: skew, sym-10NN first; mAP second; absolute + normalised together.")
    for ds in out:
        print(f"\n=== {ds} ===")
        print(f"  {'variant':<20} {'skew':>8} {'sym10NN':>8} {'|r_global|':>12} {'mAP':>8}")
        for arm in ['id', 'ood']:
            print(f"  [{arm}]")
            for key in ['baseline'] + [f'{m}_k{k}' for m in ['NICDM', 'LS']
                                         for k in K_CORR_VALUES]:
                r = out[ds][arm][key]
                print(f"    {key:<18} {r['skew_items']:>8.3f} {r['sym_10nn_frac']:>8.3f} "
                      f"{abs(r['r_global']['r_point']):>12.3f} {r['mAP_pct']:>8.3f}")

    # id-ood gap at each config
    print("\n=== Per-dataset (id - ood) mAP gap, absolute + normalised ===")
    for ds in out:
        print(f"\n[{ds}]")
        for key in ['baseline'] + [f'{m}_k{k}' for m in ['NICDM', 'LS']
                                     for k in K_CORR_VALUES]:
            id_m = out[ds]['id'][key]['mAP_pct']
            ood_m = out[ds]['ood'][key]['mAP_pct']
            gap = id_m - ood_m
            frac = gap / id_m if id_m > 0 else 0.0
            print(f"  {key:<18} id={id_m:>7.3f}  ood={ood_m:>7.3f}  "
                  f"gap={gap:>7.3f}  gap/id={frac:.4f}")

    out_path = RESULTS / 'e43_run2_corrections.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {out_path}")
    print("\nNOTE: k-reciprocal not yet implemented; running NICDM + LS only.")


if __name__ == '__main__':
    main()
