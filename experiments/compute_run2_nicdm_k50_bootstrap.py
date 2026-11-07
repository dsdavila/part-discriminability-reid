"""
Identity-bootstrap CIs on the P15 exhibit cells: NICDM_k50 on
MSMT ood and Market ood (Ch2 rev 168).

Readouts with id-bootstrap intervals (ruling 19, 4):
  - item-level k=10 skewness (gallery-resample over identities)
  - mAP (query-resample over identities)
  - |r_global| Spearman kocc vs dist-to-global-centroid

Skews and |r|: 500 id-bootstraps over gallery pids.
mAP: 500 id-bootstraps over query pids.

Also computes id-arm baseline deltas for paired contrasts if useful.
Reports signed point estimates with 95% id-clustered CIs per ruling 28.
"""
import json, time
import numpy as np
from pathlib import Path
from scipy import stats
import torch

CACHE_DIR = Path('/tmp/e12_feature_cache')
RESULTS   = Path('/data/papers/low_dimensional_reid/results')
K_OCC = 10
K_CORR = 50
N_BOOT = 500
_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CELLS = {
    'D':  ('bot_msmt_seed42_msmt.npz',  11659),
    "B'": ('msmt17_test_noREA.npz',     11659),
    "A'": ('market_test_noREA.npz',      3368),
    'C':  ('bot_msmt_seed42_market.npz', 3368),
}

TARGETS = {
    'MSMT_ood': ("B'", 'MSMT', 'ood'),
    'Market_ood': ('C', 'Market', 'ood'),
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
    return np.sqrt(np.clip(2.0 - 2.0 * sim, 0.0, None))


def compute_r_nicdm_gallery(G_cuda, g_pids, g_cids, k_corr, chunk=256):
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
            d = sim_to_dist(row[idx])
            r[i] = float(d.mean())
    return r


def compute_r_nicdm_query(Q_cuda, G_cuda, q_pids, q_cids, g_pids, g_cids, k_corr):
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


def nicdm_gallery_kocc(G_cuda, g_pids, g_cids, r_g, chunk=256):
    Ng = G_cuda.shape[0]
    kocc = np.zeros(Ng, dtype=np.int64)
    inv_r = 1.0 / np.sqrt(r_g + 1e-12)
    for start in range(0, Ng, chunk):
        end = min(Ng, start + chunk)
        sims = (G_cuda[start:end] @ G_cuda.T).cpu().numpy()
        for local in range(end - start):
            i = start + local
            row = sims[local].copy()
            exc = (g_pids == g_pids[i]) & (g_cids == g_cids[i])
            d = sim_to_dist(row)
            d_prime = d * inv_r
            d_prime[exc] = 1e9
            idx = np.argpartition(d_prime, K_OCC)[:K_OCC]
            kocc[idx] += 1
    return kocc


def nicdm_per_query_ap(sims_qg, q_pids, q_cids, g_pids, g_cids, r_q, r_g):
    """Returns per-query AP array (Nq,) and the n_valid count."""
    Nq = sims_qg.shape[0]
    d_qg = sim_to_dist(sims_qg)
    rq = np.sqrt(r_q + 1e-12)[:, None]
    rg = np.sqrt(r_g + 1e-12)[None, :]
    d_prime = d_qg / (rq * rg)
    ap = np.full(Nq, np.nan, dtype=np.float32)
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
        ap[q] = prec[pos].sum() / n_pos
    return ap


def dist_to_global_centroid_np(G_np):
    c = G_np.mean(axis=0, keepdims=True)
    return np.linalg.norm(G_np - c, axis=1)


def id_bootstrap_1d(values, pids, n_boot, rng, agg_fn):
    """Resample identities with replacement; aggregate values of all items of sampled ids
    via agg_fn (takes a flat np array, returns scalar). Returns point + CI."""
    unique = np.unique(pids)
    pid_to_idx = {int(p): np.where(pids == p)[0] for p in unique}
    boots = []
    for b in range(n_boot):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([pid_to_idx[int(p)] for p in sampled])
        boots.append(float(agg_fn(values[idx])))
    arr = np.array(boots)
    return {
        'point': float(agg_fn(values)),
        'median': float(np.median(arr)),
        'lo': float(np.quantile(arr, 0.025)),
        'hi': float(np.quantile(arr, 0.975)),
        'boot': boots,
    }


def id_bootstrap_spearman(x, y, pids, n_boot, rng):
    unique = np.unique(pids)
    pid_to_idx = {int(p): np.where(pids == p)[0] for p in unique}
    point, _ = stats.spearmanr(x, y)
    boots = []
    for b in range(n_boot):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([pid_to_idx[int(p)] for p in sampled])
        r, _ = stats.spearmanr(x[idx], y[idx])
        boots.append(float(r))
    arr = np.array(boots)
    return {
        'point': float(point),
        'median': float(np.median(arr)),
        'lo': float(np.quantile(arr, 0.025)),
        'hi': float(np.quantile(arr, 0.975)),
        'boot': boots,
    }


def id_bootstrap_ap(ap, q_pids, n_boot, rng):
    """mAP via resample of query identities. NaN APs (no positives) skipped."""
    unique = np.unique(q_pids)
    pid_to_idx = {int(p): np.where(q_pids == p)[0] for p in unique}
    def _mean_valid(a):
        m = ~np.isnan(a)
        return float(a[m].mean()) if m.any() else 0.0
    point = _mean_valid(ap)
    boots = []
    for b in range(n_boot):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([pid_to_idx[int(p)] for p in sampled])
        boots.append(_mean_valid(ap[idx]))
    arr = np.array(boots)
    return {
        'point': float(point),
        'median': float(np.median(arr)),
        'lo': float(np.quantile(arr, 0.025)),
        'hi': float(np.quantile(arr, 0.975)),
        'boot': boots,
    }


def main():
    print(f"Device: {_DEVICE}", flush=True)
    out = {}
    for tag, (cell, ds_name, arm) in TARGETS.items():
        print(f"\n### {tag} (cell={cell}) ###", flush=True)
        data = load_qg(cell)
        Q = l2(data['Q']); G = l2(data['G'])
        Qp = data['Qp']; Qc = data['Qc']
        Gp = data['Gp']; Gc = data['Gc']
        print(f"  Nq={len(Q)}  Ng={len(G)}", flush=True)

        Q_cuda = torch.from_numpy(Q).to(_DEVICE)
        G_cuda = torch.from_numpy(G).to(_DEVICE)

        # Baseline per-query AP (no correction)
        t0 = time.time()
        sims_qg_base = (Q_cuda @ G_cuda.T).cpu().numpy()
        d_base = sim_to_dist(sims_qg_base)
        ap_base = np.full(len(Q), np.nan, dtype=np.float32)
        for q in range(len(Q)):
            qp, qc = Qp[q], Qc[q]
            exc = (Gp == qp) & (Gc == qc)
            n_pos = int(((Gp == qp) & ~exc).sum())
            if n_pos == 0:
                continue
            score = -d_base[q].copy()
            score[exc] = -1e9
            order = np.argsort(-score, kind='stable')
            pos = (Gp[order] == qp) & ~exc[order]
            cum = np.cumsum(pos)
            ranks = np.arange(1, len(order) + 1)
            prec = cum / ranks
            ap_base[q] = prec[pos].sum() / n_pos
        print(f"  baseline per-query AP computed t={time.time()-t0:.1f}s", flush=True)

        # Baseline kocc on gallery
        t0 = time.time()
        kocc_base = np.zeros(len(G), dtype=np.int64)
        for start in range(0, len(G), 256):
            end = min(len(G), start + 256)
            sims = (G_cuda[start:end] @ G_cuda.T).cpu().numpy()
            for local in range(end - start):
                i = start + local
                row = sims[local].copy()
                exc = (Gp == Gp[i]) & (Gc == Gc[i])
                row[exc] = -2.0
                idx = np.argpartition(-row, K_OCC)[:K_OCC]
                kocc_base[idx] += 1
        d_centroid = dist_to_global_centroid_np(G)
        print(f"  baseline kocc + centroid dist computed t={time.time()-t0:.1f}s", flush=True)

        # NICDM_k50 correction
        t0 = time.time()
        r_g = compute_r_nicdm_gallery(G_cuda, Gp, Gc, K_CORR)
        r_q, sims_qg_fresh = compute_r_nicdm_query(Q_cuda, G_cuda, Qp, Qc, Gp, Gc, K_CORR)
        kocc_post = nicdm_gallery_kocc(G_cuda, Gp, Gc, r_g)
        ap_post = nicdm_per_query_ap(sims_qg_fresh, Qp, Qc, Gp, Gc, r_q, r_g)
        print(f"  NICDM_k50 correction + readouts t={time.time()-t0:.1f}s", flush=True)

        # Id-bootstrap intervals (rulings 19, 4)
        print(f"  running id-bootstrap (n={N_BOOT})...", flush=True)
        t0 = time.time()
        rng = np.random.default_rng(42)

        # Skewness (on gallery)
        skew_base = id_bootstrap_1d(kocc_base.astype(np.float32), Gp, N_BOOT, rng,
                                     lambda x: stats.skew(x))
        skew_post = id_bootstrap_1d(kocc_post.astype(np.float32), Gp, N_BOOT, rng,
                                     lambda x: stats.skew(x))

        # |r_global| (signed r, per ruling 28)
        r_base = id_bootstrap_spearman(kocc_base.astype(np.float32),
                                        d_centroid, Gp, N_BOOT, rng)
        r_post = id_bootstrap_spearman(kocc_post.astype(np.float32),
                                        d_centroid, Gp, N_BOOT, rng)

        # mAP (on queries)
        map_base = id_bootstrap_ap(ap_base, Qp, N_BOOT, rng)
        map_post = id_bootstrap_ap(ap_post, Qp, N_BOOT, rng)

        # Paired: post - base per bootstrap pair (requires matched resamples)
        # For precision, re-run paired bootstrap (same resample for both)
        # Skew:
        unique_g = np.unique(Gp)
        pid_to_idx_g = {int(p): np.where(Gp == p)[0] for p in unique_g}
        rng_paired = np.random.default_rng(1001)
        pairs_skew = []
        pairs_r = []
        for b in range(N_BOOT):
            sampled = rng_paired.choice(unique_g, size=len(unique_g), replace=True)
            idx = np.concatenate([pid_to_idx_g[int(p)] for p in sampled])
            pairs_skew.append(stats.skew(kocc_post[idx].astype(np.float32)) -
                               stats.skew(kocc_base[idx].astype(np.float32)))
            r1, _ = stats.spearmanr(kocc_post[idx].astype(np.float32), d_centroid[idx])
            r0, _ = stats.spearmanr(kocc_base[idx].astype(np.float32), d_centroid[idx])
            pairs_r.append(r1 - r0)
        pairs_skew = np.array(pairs_skew)
        pairs_r = np.array(pairs_r)

        unique_q = np.unique(Qp)
        pid_to_idx_q = {int(p): np.where(Qp == p)[0] for p in unique_q}
        rng_paired2 = np.random.default_rng(2002)
        pairs_map = []
        for b in range(N_BOOT):
            sampled = rng_paired2.choice(unique_q, size=len(unique_q), replace=True)
            idx = np.concatenate([pid_to_idx_q[int(p)] for p in sampled])
            def _m(a):
                m = ~np.isnan(a); return float(a[m].mean()) if m.any() else 0.0
            pairs_map.append(_m(ap_post[idx]) - _m(ap_base[idx]))
        pairs_map = np.array(pairs_map)

        def _ci(a):
            return {'median': float(np.median(a)),
                    'lo': float(np.quantile(a, 0.025)),
                    'hi': float(np.quantile(a, 0.975))}

        out[tag] = {
            'baseline': {
                'skew': skew_base, 'r_global_signed': r_base, 'mAP_fraction': map_base,
            },
            'NICDM_k50': {
                'skew': skew_post, 'r_global_signed': r_post, 'mAP_fraction': map_post,
            },
            'paired_delta_post_minus_base': {
                'skew': _ci(pairs_skew),
                'r_global_signed': _ci(pairs_r),
                'mAP_fraction': _ci(pairs_map),
            },
        }
        print(f"  all CIs t={time.time()-t0:.1f}s", flush=True)
        print(f"    skew baseline {skew_base['point']:.3f} [{skew_base['lo']:.3f}, {skew_base['hi']:.3f}]", flush=True)
        print(f"    skew post     {skew_post['point']:.3f} [{skew_post['lo']:.3f}, {skew_post['hi']:.3f}]", flush=True)
        print(f"    skew delta    {_ci(pairs_skew)['median']:+.3f} [{_ci(pairs_skew)['lo']:+.3f}, {_ci(pairs_skew)['hi']:+.3f}]", flush=True)
        print(f"    r_sign base   {r_base['point']:+.3f} [{r_base['lo']:+.3f}, {r_base['hi']:+.3f}]", flush=True)
        print(f"    r_sign post   {r_post['point']:+.3f} [{r_post['lo']:+.3f}, {r_post['hi']:+.3f}]", flush=True)
        print(f"    r delta       {_ci(pairs_r)['median']:+.3f} [{_ci(pairs_r)['lo']:+.3f}, {_ci(pairs_r)['hi']:+.3f}]", flush=True)
        print(f"    mAP base      {map_base['point']*100:.3f}% [{map_base['lo']*100:.3f}, {map_base['hi']*100:.3f}]", flush=True)
        print(f"    mAP post      {map_post['point']*100:.3f}% [{map_post['lo']*100:.3f}, {map_post['hi']*100:.3f}]", flush=True)
        print(f"    mAP delta pp  {_ci(pairs_map)['median']*100:+.3f} [{_ci(pairs_map)['lo']*100:+.3f}, {_ci(pairs_map)['hi']*100:+.3f}]", flush=True)

        del Q_cuda, G_cuda
        torch.cuda.empty_cache()

    out_path = RESULTS / 'e43_run2_nicdm_k50_bootstrap.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == '__main__':
    main()
