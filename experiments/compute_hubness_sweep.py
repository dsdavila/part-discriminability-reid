"""
Hubness gallery-size sweep (Ch2 rev 143/145). Shared ladder on both
datasets: n in {6000, 13115} on both, plus MSMT native (n=82128).
3 draws per rung. All four arms. Report per-arm skewness + multiplier
(ood/id) at each rung.

Pre-registered sign (rev 145): smaller gallery -> larger multiplier
on Market as on MSMT.
"""
import json, time
import numpy as np
from pathlib import Path
from scipy import stats

CACHE_DIR = Path('/tmp/e12_feature_cache')
RESULTS   = Path('/data/papers/low_dimensional_reid/results')
K_OCC = 10
N_DRAWS = 3

CELLS = {
    'D':  ('bot_msmt_seed42_msmt.npz',  11659),
    "B'": ('msmt17_test_noREA.npz',     11659),
    "A'": ('market_test_noREA.npz',      3368),
    'C':  ('bot_msmt_seed42_market.npz', 3368),
}
SIGMA_MSMT = 0.0625
SIGMA_MARKET = 0.062
K_RANDPROJ = 17

RUNGS = [
    ('MSMT', 'D', "B'", SIGMA_MSMT, [6000, 13115, None]),  # None = native
    ('Market', "A'", 'C', SIGMA_MARKET, [6000, 13115]),    # Market native = 13115 ~ same as rung
]


def load_cell(name):
    fn, n_q = CELLS[name]
    d = np.load(CACHE_DIR / fn); emb = d['emb'].astype(np.float32)
    return emb[n_q:], d['pids'][n_q:], d['cids'][n_q:]


def l2(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(n, 1e-12)


def noise_degrade(G, sigma, seed=42):
    rng = np.random.default_rng(seed)
    g_eps = rng.normal(0, sigma, G.shape).astype(np.float32)
    return l2(G + g_eps)


def randproj_degrade(G, k_target, seed=42):
    D = G.shape[1]
    rng = np.random.default_rng(seed)
    R = rng.standard_normal((D, k_target)).astype(np.float32)
    Qmat, _ = np.linalg.qr(R)
    return l2(G @ Qmat)


def compute_k_occurrence(G, g_pids, g_cids, k=K_OCC, chunk=256):
    N = len(G)
    occurrence = np.zeros(N, dtype=np.int64)
    for start in range(0, N, chunk):
        end = min(N, start + chunk)
        sims = G[start:end] @ G.T
        for local in range(end - start):
            gi = start + local
            exc = (g_pids == g_pids[gi]) & (g_cids == g_cids[gi])
            sims[local, exc] = -2.0
            idx = np.argpartition(-sims[local], k)[:k]
            occurrence[idx] += 1
    return occurrence


def run_rung(label, G_id_full, pids_full, cids_full, G_ood_full, sigma, n_target, draw_idx):
    rng = np.random.default_rng(42 + draw_idx * 1001 + hash(label) % 10000)
    N_full = len(G_id_full)
    if n_target is None or n_target >= N_full:
        idx = np.arange(N_full)
        n_use = N_full
    else:
        idx = rng.choice(N_full, size=n_target, replace=False)
        n_use = n_target
    G_id = G_id_full[idx]; g_pids = pids_full[idx]; g_cids = cids_full[idx]
    G_ood = G_ood_full[idx]
    G_noise = noise_degrade(G_id, sigma, seed=42 + draw_idx)
    G_rp = randproj_degrade(G_id, K_RANDPROJ, seed=42 + draw_idx)
    arms = {'id': G_id, 'ood': G_ood, 'noise': G_noise, 'randproj': G_rp}
    out = {'n_items': int(n_use), 'n_ids': int(len(np.unique(g_pids))),
           'per_arm': {}}
    for arm_name, G in arms.items():
        occ = compute_k_occurrence(G, g_pids, g_cids, k=K_OCC)
        out['per_arm'][arm_name] = {
            'skewness_items': float(stats.skew(occ)),
            'max_occ': int(occ.max()),
            'frac_zero': float((occ == 0).mean()),
            'mean_occ': float(occ.mean()),
        }
    out['multiplier'] = out['per_arm']['ood']['skewness_items'] / out['per_arm']['id']['skewness_items']
    return out


def main():
    all_out = {}
    for dataset, id_cell, ood_cell, sigma, rungs in RUNGS:
        print(f"\n########## {dataset}: id={id_cell} ood={ood_cell} ##########", flush=True)
        G_id_full, pids_full, cids_full = load_cell(id_cell)
        G_ood_full, _, _ = load_cell(ood_cell)
        N_native = len(G_id_full)
        print(f"  native n={N_native}", flush=True)
        all_out[dataset] = {'native_n': N_native, 'rungs': {}}
        for n_target in rungs:
            rung_label = f"n={n_target or N_native}"
            if n_target is not None and n_target > N_native:
                print(f"  SKIP {rung_label} (exceeds native)", flush=True)
                continue
            draws = []
            t0 = time.time()
            for draw in range(N_DRAWS):
                r = run_rung(f"{dataset}-{n_target}", G_id_full, pids_full, cids_full,
                              G_ood_full, sigma, n_target, draw)
                draws.append(r)
                print(f"  {rung_label} draw {draw+1}: "
                      f"id={r['per_arm']['id']['skewness_items']:.3f} "
                      f"ood={r['per_arm']['ood']['skewness_items']:.3f} "
                      f"mult={r['multiplier']:.3f} "
                      f"t={time.time()-t0:.1f}s", flush=True)
            # Summary across draws
            summ = {}
            for arm in ('id', 'ood', 'noise', 'randproj'):
                skews = [d['per_arm'][arm]['skewness_items'] for d in draws]
                summ[arm] = {'mean': float(np.mean(skews)),
                              'std': float(np.std(skews)),
                              'draws': skews}
            mults = [d['multiplier'] for d in draws]
            summ['multiplier'] = {'mean': float(np.mean(mults)),
                                   'std': float(np.std(mults)),
                                   'draws': mults}
            all_out[dataset]['rungs'][rung_label] = {
                'n_items': draws[0]['n_items'],
                'n_ids_range': [min(d['n_ids'] for d in draws),
                                 max(d['n_ids'] for d in draws)],
                'per_arm_summary': summ,
                'draws': draws,
            }

    print("\n========== Hubness sweep summary ==========")
    for dataset, info in all_out.items():
        print(f"\n[{dataset}] native n={info['native_n']}")
        print(f"  {'rung':<15} {'n':>7} {'id':>8} {'ood':>8} {'mult':>8} {'noise':>8} {'rp':>8}")
        for rung, r in info['rungs'].items():
            s = r['per_arm_summary']
            print(f"  {rung:<15} {r['n_items']:>7} "
                  f"{s['id']['mean']:>8.3f} {s['ood']['mean']:>8.3f} "
                  f"{s['multiplier']['mean']:>8.3f} "
                  f"{s['noise']['mean']:>8.3f} {s['randproj']['mean']:>8.3f}")

    out_path = RESULTS / 'hubness_sweep.json'
    with open(out_path, 'w') as f:
        json.dump(all_out, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == '__main__':
    main()
