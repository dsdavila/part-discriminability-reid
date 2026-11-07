"""
Identity-clustered bootstrap CIs on the IN-DOMAIN baseline cells
(MSMT id / Market id, no correction). For the P15 CI-vs-CI statement.

Readouts:
  - item k=10 skewness on gallery (bootstrap over gallery identities)
  - signed r_global = Spearman(kocc, dist_to_centroid) on gallery items
Both: 500 identity-clustered bootstrap draws.
"""
import json, time
import numpy as np
from pathlib import Path
from scipy import stats
import torch

CACHE_DIR = Path('/tmp/e12_feature_cache')
RESULTS   = Path('/data/papers/low_dimensional_reid/results')
K_OCC = 10
N_BOOT = 500
_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CELLS = {
    'MSMT_id':   ('bot_msmt_seed42_msmt.npz',  11659),
    'Market_id': ('market_test_noREA.npz',      3368),
}


def load_gallery(fname, n_q):
    d = np.load(CACHE_DIR / fname); emb = d['emb'].astype(np.float32)
    return emb[n_q:], d['pids'][n_q:], d['cids'][n_q:]


def l2(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(n, 1e-12)


def compute_kocc_gallery(G_cuda, g_pids, g_cids, chunk=256):
    Ng = G_cuda.shape[0]
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
    return kocc


def dist_to_centroid(G):
    c = G.mean(axis=0, keepdims=True)
    return np.linalg.norm(G - c, axis=1)


def main():
    print(f"Device: {_DEVICE}", flush=True)
    out = {}
    for tag, (fname, n_q) in CELLS.items():
        print(f"\n### {tag} ###", flush=True)
        G, g_pids, g_cids = load_gallery(fname, n_q)
        G = l2(G)
        G_cuda = torch.from_numpy(G).to(_DEVICE)
        print(f"  Ng={len(G)}, n_ids={len(np.unique(g_pids))}", flush=True)

        t0 = time.time()
        kocc = compute_kocc_gallery(G_cuda, g_pids, g_cids)
        d_c = dist_to_centroid(G)
        print(f"  kocc + dist computed t={time.time()-t0:.1f}s", flush=True)

        unique = np.unique(g_pids)
        pid_to_idx = {int(p): np.where(g_pids == p)[0] for p in unique}
        rng = np.random.default_rng(42)
        skew_boots = []
        r_boots = []
        for b in range(N_BOOT):
            sampled = rng.choice(unique, size=len(unique), replace=True)
            idx = np.concatenate([pid_to_idx[int(p)] for p in sampled])
            skew_boots.append(float(stats.skew(kocc[idx].astype(np.float32))))
            r, _ = stats.spearmanr(kocc[idx].astype(np.float32), d_c[idx])
            r_boots.append(float(r))
        skew_arr = np.array(skew_boots)
        r_arr = np.array(r_boots)

        skew_point = float(stats.skew(kocc.astype(np.float32)))
        r_point, _ = stats.spearmanr(kocc.astype(np.float32), d_c)
        out[tag] = {
            'skew': {'point': skew_point,
                      'lo': float(np.quantile(skew_arr, 0.025)),
                      'hi': float(np.quantile(skew_arr, 0.975)),
                      'median': float(np.median(skew_arr))},
            'r_global_signed': {'point': float(r_point),
                                 'lo': float(np.quantile(r_arr, 0.025)),
                                 'hi': float(np.quantile(r_arr, 0.975)),
                                 'median': float(np.median(r_arr))},
        }
        print(f"  skew   point={skew_point:.3f}  CI=[{out[tag]['skew']['lo']:.3f}, {out[tag]['skew']['hi']:.3f}]", flush=True)
        print(f"  r_sign point={r_point:+.3f}  CI=[{out[tag]['r_global_signed']['lo']:+.3f}, {out[tag]['r_global_signed']['hi']:+.3f}]", flush=True)
        del G_cuda
        torch.cuda.empty_cache()

    out_path = RESULTS / 'e43_id_baseline_bootstrap.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == '__main__':
    main()
