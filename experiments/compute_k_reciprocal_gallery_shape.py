"""
k-reciprocal gallery-side graph-shape readouts (Ch2 rev 171).

Applies k-reciprocal (Zhong 2017) to the GALLERY only, computes top-10
neighbours in the corrected Jaccard distance, and reads:
  - item k=10 skewness
  - symmetric 10-NN fraction
  - signed r_global = Spearman(kocc, dist_to_centroid)

Per dataset per arm (id, ood). This is the dissociation test for P15:
  - If skew / sym-10NN land near NICDM_k50's floor while mAP recovered
    4x more, P15 strengthens (same graph shape, different mAP).
  - If skew drops substantially below NICDM_k50 tracking the larger
    mAP gain, P15 is withdrawn.
"""
import json, time
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
    'MSMT_id':   ('bot_msmt_seed42_msmt.npz',  11659),
    'MSMT_ood':  ('msmt17_test_noREA.npz',     11659),
    'Market_id': ('market_test_noREA.npz',      3368),
    'Market_ood':('bot_msmt_seed42_market.npz', 3368),
}

_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_gallery(fname, n_q):
    d = np.load(CACHE_DIR / fname); emb = d['emb'].astype(np.float32)
    return emb[n_q:], d['pids'][n_q:], d['cids'][n_q:]


def l2(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(n, 1e-12)


def dist_from_sim(sim):
    return np.sqrt(np.clip(2.0 - 2.0 * sim, 0.0, None))


def topk_gallery_only(G_cuda, k, chunk=256):
    """Top-k argmax indices for each gallery item against all gallery items.
    Does NOT exclude self; caller handles."""
    N = G_cuda.shape[0]
    out = np.zeros((N, k), dtype=np.int32)
    for start in range(0, N, chunk):
        end = min(N, start + chunk)
        sims = G_cuda[start:end] @ G_cuda.T  # (chunk, N)
        _, idx = torch.topk(sims, k, dim=1)
        out[start:end] = idx.cpu().numpy().astype(np.int32)
    return out


def k_reciprocal_gallery_shape(G_np, g_pids, g_cids, k1=K1, k2=K2):
    """Compute Jaccard-corrected top-10 neighbours per gallery item.
    Returns:
      kocc (Ng,) int64
      top_nb (Ng, K_OCC) int32
      d_centroid (Ng,) float
    """
    Ng = len(G_np)
    G_cuda = torch.from_numpy(G_np).to(_DEVICE)

    print(f"    top-{k1+1} on gallery ({Ng}x{Ng}) via GPU chunked...", flush=True)
    t0 = time.time()
    initial_rank = topk_gallery_only(G_cuda, k1 + 1)
    print(f"      t={time.time()-t0:.1f}s", flush=True)

    print("    building V (reciprocal sets, expanded)...", flush=True)
    t0 = time.time()
    V_rows = [None] * Ng
    for i in range(Ng):
        forward = initial_rank[i, :k1+1]
        backward = initial_rank[forward, :k1+1]
        in_back = (backward == i).any(axis=1)
        k_recip = forward[in_back]
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
        sims = (G_cuda[i:i+1] @ G_cuda[cols].T).cpu().numpy()[0]
        d = dist_from_sim(sims)
        w = np.exp(-d)
        w = w / max(w.sum(), 1e-12)
        V_rows[i] = (cols, w.astype(np.float32))
        if (i + 1) % 20000 == 0:
            print(f"      i={i+1}/{Ng} t={time.time()-t0:.1f}s", flush=True)
    print(f"      V built  t={time.time()-t0:.1f}s", flush=True)

    # Local QE
    if k2 > 1:
        print("    local QE...", flush=True)
        t0 = time.time()
        V_new = [None] * Ng
        for i in range(Ng):
            agg = {}
            neigh = initial_rank[i, :k2]
            for n in neigh:
                cols, vals = V_rows[int(n)]
                for c, v in zip(cols.tolist(), vals.tolist()):
                    agg[c] = agg.get(c, 0.0) + v / k2
            if len(agg) == 0:
                V_new[i] = (np.array([], dtype=np.int32), np.array([], dtype=np.float32))
            else:
                cols_n = np.array(sorted(agg.keys()), dtype=np.int32)
                vals_n = np.array([agg[c] for c in cols_n.tolist()], dtype=np.float32)
                V_new[i] = (cols_n, vals_n)
            if (i + 1) % 20000 == 0:
                print(f"      i={i+1}/{Ng} t={time.time()-t0:.1f}s", flush=True)
        V_rows = V_new
        print(f"      QE t={time.time()-t0:.1f}s", flush=True)

    # Inverted index
    print("    inverted index...", flush=True)
    inv = [[] for _ in range(Ng)]
    for i in range(Ng):
        cols, vals = V_rows[i]
        for c, v in zip(cols, vals):
            inv[int(c)].append((i, float(v)))

    # For each gallery item i, compute Jaccard distance to all other gallery
    # items via inverted index, take top-10 (minimum Jaccard distance),
    # excluding same-cam-same-pid.
    print("    computing top-10 per gallery item in Jaccard space...", flush=True)
    t0 = time.time()
    kocc = np.zeros(Ng, dtype=np.int64)
    top_nb = np.zeros((Ng, K_OCC), dtype=np.int32)
    for i in range(Ng):
        cols_q, vals_q = V_rows[i]
        if len(cols_q) == 0:
            # fallback to raw order
            temp_min = np.zeros(Ng, dtype=np.float32)
        else:
            temp_min = np.zeros(Ng, dtype=np.float32)
            for c, vq in zip(cols_q.tolist(), vals_q.tolist()):
                for (idx, vj) in inv[int(c)]:
                    temp_min[idx] += min(vq, vj)
        jac = 1.0 - temp_min / (2.0 - temp_min + 1e-12)
        # same-cam-same-pid exclusion + self
        exc = (g_pids == g_pids[i]) & (g_cids == g_cids[i])
        jac_masked = jac.copy()
        jac_masked[exc] = 1e9
        idx_top = np.argpartition(jac_masked, K_OCC)[:K_OCC]
        order = idx_top[np.argsort(jac_masked[idx_top])]
        top_nb[i] = order
        kocc[order] += 1
        if (i + 1) % 20000 == 0:
            print(f"      i={i+1}/{Ng} t={time.time()-t0:.1f}s", flush=True)
    print(f"      top-10 t={time.time()-t0:.1f}s", flush=True)

    # distance to global centroid (unchanged by correction)
    c = G_np.mean(axis=0, keepdims=True)
    d_centroid = np.linalg.norm(G_np - c, axis=1)

    del G_cuda
    torch.cuda.empty_cache()
    return kocc, top_nb, d_centroid


def symmetric_10nn_fraction(top_nb, k=K_OCC):
    Ng = top_nb.shape[0]
    total = Ng * k
    sets = [set(int(x) for x in top_nb[j]) for j in range(Ng)]
    sym = 0
    for i in range(Ng):
        for r in range(k):
            j = int(top_nb[i, r])
            if i in sets[j]:
                sym += 1
    return sym / total


def main():
    print(f"Device: {_DEVICE}", flush=True)
    out = {}
    for tag, (fname, n_q) in CELLS.items():
        print(f"\n### {tag} ###", flush=True)
        G, g_pids, g_cids = load_gallery(fname, n_q)
        G = l2(G)
        print(f"  Ng={len(G)}  n_ids={len(np.unique(g_pids))}", flush=True)
        t0 = time.time()
        kocc, top_nb, d_c = k_reciprocal_gallery_shape(G, g_pids, g_cids)
        skew_val = float(stats.skew(kocc.astype(np.float32)))
        sym = symmetric_10nn_fraction(top_nb)
        r_val, _ = stats.spearmanr(kocc.astype(np.float32), d_c)
        print(f"  skew_items    = {skew_val:.3f}", flush=True)
        print(f"  sym_10nn_frac = {sym:.3f}", flush=True)
        print(f"  signed r      = {r_val:+.3f}", flush=True)
        print(f"  total t={(time.time()-t0)/60:.1f}m", flush=True)
        out[tag] = {
            'skew_items': skew_val,
            'sym_10nn_frac': float(sym),
            'r_global_signed': float(r_val),
            'kocc_summary': {
                'mean': float(kocc.mean()),
                'max': int(kocc.max()),
                'min': int(kocc.min()),
                'frac_zero': float((kocc == 0).mean()),
            },
            'k1': K1, 'k2': K2,
        }

    print("\n========== k-reciprocal gallery-side graph-shape summary ==========")
    for tag in CELLS:
        s = out[tag]
        print(f"  {tag:<12} skew={s['skew_items']:.3f}  sym10NN={s['sym_10nn_frac']:.3f}  "
              f"r={s['r_global_signed']:+.3f}")

    out_path = RESULTS / 'e43_k_reciprocal_gallery_shape.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == '__main__':
    main()
