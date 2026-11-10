"""
Matched-rank containment with structure-matched null (rev 366).

For each of 4 comparisons, build span at COMMON rank r=5 and measure
mean capture of OOD cam offsets. Structure-matched null: shuffle cam
labels in the SPAN-side dataset only, rebuild (b) B_cam at same r,
measure capture of unchanged probe. N=200 draws.

Report:
  obs capture
  null mean, sd
  excess = obs - null mean
  x-chance = obs / (r/d)  where d=2048
  x-null = obs / null_mean
"""
import json
import numpy as np
from pathlib import Path

CACHE_DIR = Path('/tmp/e12_feature_cache')
TRAIN_CACHE = Path('/tmp/e18_train_cache')
RESULTS   = Path('/data/papers/low_dimensional_reid/results')
R = 5
N_NULL = 200
RNG_SEED = 42
D_EMB = 2048

COMPARISONS = {
    'Market_train_vs_Bprime': {
        'span_src': 'train', 'span_file': 'bot_market_market_train.npz', 'span_n_q': None,
        'probe_file': 'msmt17_test_noREA.npz', 'probe_n_q': 11659,
        'label': 'Market model: TRAIN span vs B′ (MSMT) cam offsets',
    },
    'MSMT_train_vs_C': {
        'span_src': 'train', 'span_file': 'bot_msmt_msmt_train.npz', 'span_n_q': None,
        'probe_file': 'bot_msmt_seed42_market.npz', 'probe_n_q': 3368,
        'label': 'MSMT model: TRAIN span vs C (Market) cam offsets',
    },
    'Market_test_vs_Bprime': {
        'span_src': 'test', 'span_file': 'market_test_noREA.npz', 'span_n_q': 3368,
        'probe_file': 'msmt17_test_noREA.npz', 'probe_n_q': 11659,
        'label': 'Market model: TEST (A′) span vs B′ cam offsets',
    },
    'MSMT_test_vs_C': {
        'span_src': 'test', 'span_file': 'bot_msmt_seed42_msmt.npz', 'span_n_q': 11659,
        'probe_file': 'bot_msmt_seed42_market.npz', 'probe_n_q': 3368,
        'label': 'MSMT model: TEST (D) span vs C cam offsets',
    },
}


def l2(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(n, 1e-12)


def camera_offsets_b(X, pids, cids):
    unique_pids = np.unique(pids)
    uc = np.unique(cids)
    per_cam = {c: [] for c in uc}
    for p in unique_pids:
        mask_p = (pids == p)
        items_p = X[mask_p]
        if len(items_p) < 2:
            continue
        mu_i = items_p.mean(0)
        cams_p = cids[mask_p]
        for c in np.unique(cams_p):
            mu_ic = items_p[cams_p == c].mean(0)
            per_cam[c].append(mu_ic - mu_i)
    nu_c = []
    for c in uc:
        if per_cam[c]:
            nu_c.append(np.mean(np.stack(per_cam[c]), axis=0))
        else:
            nu_c.append(np.zeros(X.shape[1], dtype=X.dtype))
    nu_c = np.stack(nu_c)
    return nu_c - nu_c.mean(0)


def top_r_basis(mat, r):
    U, s, Vt = np.linalg.svd(mat, full_matrices=False)
    return Vt[:min(r, Vt.shape[0])].T


def capture_mean(B, nu_probe):
    """Mean fraction of each probe row's energy captured by span B."""
    caps = []
    for nu in nu_probe:
        if np.linalg.norm(nu) < 1e-12:
            continue
        coef = B.T @ nu
        caps.append((coef ** 2).sum() / (nu ** 2).sum())
    return float(np.mean(caps))


def load_items(info):
    d = np.load((TRAIN_CACHE if info['span_src'] == 'train' else CACHE_DIR) / info['span_file'])
    emb = d['emb'].astype(np.float32); pids = d['pids']; cids = d['cids']
    if info['span_n_q'] is not None:
        # test cache: take gallery portion
        emb = emb[info['span_n_q']:]; pids = pids[info['span_n_q']:]; cids = cids[info['span_n_q']:]
    return l2(emb), pids, cids


def load_probe(info):
    d = np.load(CACHE_DIR / info['probe_file'])
    emb = d['emb'].astype(np.float32); pids = d['pids']; cids = d['cids']
    n_q = info['probe_n_q']
    return l2(emb[n_q:]), pids[n_q:], cids[n_q:]


def process(name, info):
    print(f"\n### {name} — {info['label']} ###", flush=True)
    X_s, p_s, c_s = load_items(info)
    X_p, p_p, c_p = load_probe(info)
    print(f"  span: n={len(X_s)}  n_cams={len(np.unique(c_s))}  probe: n={len(X_p)}  n_cams={len(np.unique(c_p))}", flush=True)
    # Observed span
    nu_span = camera_offsets_b(X_s, p_s, c_s)
    B_span = top_r_basis(nu_span, R)
    # Probe (unchanged)
    nu_probe = camera_offsets_b(X_p, p_p, c_p)
    obs = capture_mean(B_span, nu_probe)
    # Random chance
    chance = R / D_EMB
    x_chance = obs / chance
    print(f"  obs capture at r={R}: {obs:.4f}  chance={chance:.4f}  x-chance={x_chance:.2f}", flush=True)

    # Structure-matched null: shuffle cam labels in span-side (keep per-cam
    # counts fixed), rebuild (b) B_cam, measure capture
    print(f"  structure-matched null ({N_NULL} draws)...", flush=True)
    rng = np.random.default_rng(RNG_SEED)
    cam_sizes = [int((c_s == c).sum()) for c in np.unique(c_s)]
    null_caps = []
    n = len(X_s)
    for di in range(N_NULL):
        perm = rng.permutation(n)
        fake_cids = np.empty(n, dtype=c_s.dtype); idx = 0
        for k, nk in enumerate(cam_sizes):
            fake_cids[perm[idx:idx+nk]] = sorted(np.unique(c_s))[k]
            idx += nk
        nu_fake = camera_offsets_b(X_s, p_s, fake_cids)
        B_fake = top_r_basis(nu_fake, R)
        null_caps.append(capture_mean(B_fake, nu_probe))
        if di < 2 or (di + 1) % 50 == 0:
            print(f"    draw {di+1}/{N_NULL}: cap={null_caps[-1]:.4f}", flush=True)
    null_caps = np.array(null_caps)
    excess = obs - null_caps.mean()
    z = (obs - null_caps.mean()) / null_caps.std(ddof=1)
    x_null = obs / null_caps.mean() if null_caps.mean() > 0 else float('inf')
    print(f"  null: mean={null_caps.mean():.4f}  sd={null_caps.std(ddof=1):.5f}  range=[{null_caps.min():.4f}, {null_caps.max():.4f}]  excess={excess:+.4f}  z={z:+.2f}  x-null={x_null:.2f}", flush=True)

    return {
        'name': name, 'label': info['label'], 'r': R,
        'obs_capture': obs, 'chance': chance, 'x_chance': x_chance,
        'null_mean': float(null_caps.mean()), 'null_sd': float(null_caps.std(ddof=1)),
        'null_range': [float(null_caps.min()), float(null_caps.max())],
        'excess': float(excess), 'z': float(z), 'x_null': float(x_null),
    }


def main():
    out = {}
    for name, info in COMPARISONS.items():
        out[name] = process(name, info)
    out_path = RESULTS / 'e43_containment_matched_r5.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {out_path}")

    print("\n\n========= Containment matched-rank r=5 summary =========")
    print(f"{'comparison':<30s}  {'obs':>7s}  {'chance':>7s}  {'x-chance':>9s}  {'null mean':>10s}  {'excess':>8s}  {'x-null':>7s}  {'z':>6s}")
    for n in COMPARISONS:
        r = out[n]
        print(f"{n:<30s}  {r['obs_capture']:7.4f}  {r['chance']:7.4f}  {r['x_chance']:9.2f}  {r['null_mean']:10.4f}  {r['excess']:+8.4f}  {r['x_null']:7.2f}  {r['z']:+6.2f}")


if __name__ == '__main__':
    main()
