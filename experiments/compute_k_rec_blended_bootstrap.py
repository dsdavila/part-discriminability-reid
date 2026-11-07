"""
Identity-clustered bootstrap CIs on k-reciprocal BLENDED gallery-shape
readouts (Ch2 rev 179). Load-bearing comparison: k-rec post-skew vs
id-baseline skew on MSMT ood and Market ood.

Reruns k-reciprocal with blended distance (lambda=0.3 on original +
(1-lambda)=0.7 on Jaccard), saves per-item kocc and top-10 neighbours,
then id-bootstrap for:
  - skew (per ruling 19)
  - sym-10NN (per ruling 19 parity, rev 171)
  - signed r_global (per rulings 19 + 28)

All four arms; id-bootstrap conditioned on the realised gallery graph.
"""
import json, time
import numpy as np
from pathlib import Path
from scipy import stats
import torch
from sym_10nn import symmetric_knn_fraction, per_item_reciprocity

CACHE_DIR = Path('/tmp/e12_feature_cache')
RESULTS   = Path('/data/papers/low_dimensional_reid/results')
K_OCC = 10
K1 = 20
K2 = 6
LAMBDA = 0.3
N_BOOT = 500
_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CELLS = {
    'MSMT_id':   ('bot_msmt_seed42_msmt.npz',  11659),
    'MSMT_ood':  ('msmt17_test_noREA.npz',     11659),
    'Market_id': ('market_test_noREA.npz',      3368),
    'Market_ood':('bot_msmt_seed42_market.npz', 3368),
}


def load_gallery(fname, n_q):
    d = np.load(CACHE_DIR / fname); emb = d['emb'].astype(np.float32)
    return emb[n_q:], d['pids'][n_q:], d['cids'][n_q:]


def l2(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(n, 1e-12)


def dist_from_sim(s):
    return np.sqrt(np.clip(2.0 - 2.0 * s, 0.0, None))


def topk_gallery_only(G_cuda, k, chunk=256):
    N = G_cuda.shape[0]
    out = np.zeros((N, k), dtype=np.int32)
    for start in range(0, N, chunk):
        end = min(N, start + chunk)
        sims = G_cuda[start:end] @ G_cuda.T
        _, idx = torch.topk(sims, k, dim=1)
        out[start:end] = idx.cpu().numpy().astype(np.int32)
    return out


def k_reciprocal_blended(G_np, g_pids, g_cids, k1=K1, k2=K2, lam=LAMBDA):
    Ng = len(G_np)
    G_cuda = torch.from_numpy(G_np).to(_DEVICE)

    print(f"    top-{k1+1}...", flush=True); t0 = time.time()
    initial_rank = topk_gallery_only(G_cuda, k1 + 1)
    print(f"      t={time.time()-t0:.1f}s", flush=True)

    print("    V matrix...", flush=True); t0 = time.time()
    V_rows = [None] * Ng
    for i in range(Ng):
        forward = initial_rank[i, :k1+1]
        backward = initial_rank[forward, :k1+1]
        in_back = (backward == i).any(axis=1)
        k_recip = forward[in_back]
        expansion = set(int(x) for x in k_recip)
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
                    expansion.add(int(x))
        cols = np.array(sorted(expansion), dtype=np.int32)
        sims = (G_cuda[i:i+1] @ G_cuda[cols].T).cpu().numpy()[0]
        w = np.exp(-dist_from_sim(sims))
        w = w / max(w.sum(), 1e-12)
        V_rows[i] = (cols, w.astype(np.float32))
    print(f"      t={time.time()-t0:.1f}s", flush=True)

    if k2 > 1:
        print("    QE...", flush=True); t0 = time.time()
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
        print(f"      t={time.time()-t0:.1f}s", flush=True)

    inv = [[] for _ in range(Ng)]
    for i in range(Ng):
        cols, vals = V_rows[i]
        for c, v in zip(cols, vals):
            inv[int(c)].append((i, float(v)))

    print(f"    blended top-10 (lambda={lam})...", flush=True); t0 = time.time()
    kocc = np.zeros(Ng, dtype=np.int64)
    top_nb = np.zeros((Ng, K_OCC), dtype=np.int32)
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
        order = idx_top[np.argsort(blend[idx_top])]
        top_nb[i] = order
        kocc[order] += 1
    print(f"      t={time.time()-t0:.1f}s", flush=True)

    c = G_np.mean(axis=0, keepdims=True)
    d_centroid = np.linalg.norm(G_np - c, axis=1)

    del G_cuda
    torch.cuda.empty_cache()
    return kocc, top_nb, d_centroid


def id_boot_skew(kocc, pids, n_boot, rng):
    unique = np.unique(pids)
    pid_to_idx = {int(p): np.where(pids == p)[0] for p in unique}
    boots = []
    for b in range(n_boot):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([pid_to_idx[int(p)] for p in sampled])
        boots.append(float(stats.skew(kocc[idx].astype(np.float32))))
    return boots


def id_boot_r(kocc, d, pids, n_boot, rng):
    unique = np.unique(pids)
    pid_to_idx = {int(p): np.where(pids == p)[0] for p in unique}
    boots = []
    for b in range(n_boot):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([pid_to_idx[int(p)] for p in sampled])
        r, _ = stats.spearmanr(kocc[idx].astype(np.float32), d[idx])
        boots.append(float(r))
    return boots


def id_boot_sym10nn(top_nb, pids, n_boot, rng, k=K_OCC):
    """Bootstrap sym-10NN per Ch2 rev 183: full-gallery neighbour graph
    stays fixed; resample identities; AVERAGE the per-item reciprocity
    indicator over the resampled items. No neighbour filtering.

    per_item[i] = fraction of i's own top-k neighbours whose list
    reciprocates i (computed on the FULL gallery graph, not a subset).
    """
    per_item = per_item_reciprocity(top_nb, k=k)
    unique = np.unique(pids)
    pid_to_idx = {int(p): np.where(pids == p)[0] for p in unique}
    boots = []
    for b in range(n_boot):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([pid_to_idx[int(p)] for p in sampled])
        boots.append(float(per_item[idx].mean()))
    return boots


def _ci(arr):
    a = np.array(arr)
    return {'point': float(a.mean()),
            'median': float(np.median(a)),
            'lo': float(np.quantile(a, 0.025)),
            'hi': float(np.quantile(a, 0.975))}


def main():
    print(f"Device: {_DEVICE}", flush=True)
    out = {}
    for tag, (fname, n_q) in CELLS.items():
        print(f"\n### {tag} ###", flush=True)
        G, g_pids, g_cids = load_gallery(fname, n_q)
        G = l2(G)
        print(f"  Ng={len(G)}  n_ids={len(np.unique(g_pids))}", flush=True)
        t_full = time.time()
        kocc, top_nb, d_c = k_reciprocal_blended(G, g_pids, g_cids)
        skew_point = float(stats.skew(kocc.astype(np.float32)))
        r_point, _ = stats.spearmanr(kocc.astype(np.float32), d_c)
        # Full-sample sym-10NN via the shared, unit-tested implementation
        sym_point = symmetric_knn_fraction(top_nb, k=K_OCC)
        print(f"  point: skew={skew_point:.4f}  sym10NN={sym_point:.4f}  r={r_point:+.4f}", flush=True)

        print(f"  id-boot n={N_BOOT}...", flush=True); t0 = time.time()
        rng = np.random.default_rng(42)
        skew_boots = id_boot_skew(kocc, g_pids, N_BOOT, rng)
        r_boots = id_boot_r(kocc, d_c, g_pids, N_BOOT, rng)
        print(f"    skew+r done t={time.time()-t0:.1f}s", flush=True)
        sym_boots = id_boot_sym10nn(top_nb, g_pids, N_BOOT, rng)
        print(f"    sym-10NN done t={time.time()-t0:.1f}s", flush=True)

        out[tag] = {
            'skew': _ci(skew_boots) | {'point_est': skew_point},
            'r_global_signed': _ci(r_boots) | {'point_est': r_point},
            'sym_10nn_frac': _ci(sym_boots) | {'point_est': sym_point},
            'kocc_summary': {
                'mean': float(kocc.mean()),
                'max': int(kocc.max()),
                'frac_zero': float((kocc == 0).mean()),
            },
            'readout_space': 'blended (0.3 original + 0.7 Jaccard)',
            'n_boot': N_BOOT,
        }
        print(f"  skew   point={skew_point:.4f}  CI=[{out[tag]['skew']['lo']:.4f}, {out[tag]['skew']['hi']:.4f}]", flush=True)
        print(f"  r_sign point={r_point:+.4f}  CI=[{out[tag]['r_global_signed']['lo']:+.4f}, {out[tag]['r_global_signed']['hi']:+.4f}]", flush=True)
        print(f"  sym    point={sym_point:.4f}  CI=[{out[tag]['sym_10nn_frac']['lo']:.4f}, {out[tag]['sym_10nn_frac']['hi']:.4f}]", flush=True)
        print(f"  total t={(time.time()-t_full)/60:.1f}m", flush=True)

    out_path = RESULTS / 'e43_k_reciprocal_blended_bootstrap.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == '__main__':
    main()
