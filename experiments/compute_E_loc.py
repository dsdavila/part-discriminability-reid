"""
E-loc: where does the OOD cloud sit relative to the in-domain cloud,
in each model's own embedding space (rev 378).

For each model:
  In-domain cloud = model's training-set embeddings
  OOD cloud = model's output on OOD test gallery

Three measurements:
1. Centroid separation in RMS-radius units (||mu_id - mu_ood|| / rms_radius).
2. Blob-level nearest neighbours: for each OOD blob centroid, find the
   nearest centroid in the pooled set of all in-domain blobs + other OOD
   blobs. Fraction whose NN is another OOD blob vs an in-domain blob.
3. Linear probe margin (in-domain items vs OOD items) with within-domain
   control (two random halves of same domain, matched size).

Caches:
  Market model id (train): bot_market_market_train.npz
  Market model ood:        msmt17_test_noREA.npz (gallery portion, n_q=11659)
  MSMT model id (train):   bot_msmt_msmt_train.npz
  MSMT model ood:          bot_msmt_seed42_market.npz (gallery, n_q=3368)
"""
import json
import numpy as np
from pathlib import Path

CACHE_DIR = Path('/tmp/e12_feature_cache')
TRAIN_CACHE = Path('/tmp/e18_train_cache')
RESULTS   = Path('/data/papers/low_dimensional_reid/results')
RNG_SEED = 42

MODELS = {
    'Market_model': {
        'id_file': 'bot_market_market_train.npz', 'id_src': 'train',
        'ood_file': 'msmt17_test_noREA.npz', 'ood_src': 'test', 'ood_n_q': 11659,
        'label': 'Market model: id=Market-train, ood=MSMT-test (B′)',
    },
    'MSMT_model': {
        'id_file': 'bot_msmt_msmt_train.npz', 'id_src': 'train',
        'ood_file': 'bot_msmt_seed42_market.npz', 'ood_src': 'test', 'ood_n_q': 3368,
        'label': 'MSMT model: id=MSMT-train, ood=Market-test (C)',
    },
}


def l2(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(n, 1e-12)


def load_items(info_key, info, src_key='ood'):
    if src_key == 'id':
        d = np.load(TRAIN_CACHE / info['id_file'])
        return l2(d['emb'].astype(np.float32)), d['pids']
    else:
        d = np.load(CACHE_DIR / info['ood_file'])
        n_q = info['ood_n_q']
        return l2(d['emb'].astype(np.float32)[n_q:]), d['pids'][n_q:]


def identity_centroids(X, pids):
    uids = np.unique(pids)
    return uids, np.stack([X[pids == p].mean(0) for p in uids])


def rms_radius(X, mu):
    """RMS distance from centroid."""
    diffs = X - mu
    return float(np.sqrt((diffs ** 2).sum(axis=1).mean()))


def linear_probe_margin(X_a, X_b, rng, max_items=5000):
    """Fit linear SVM-like probe: w . x minimises hinge-ish loss.
    Return min margin and mean |w.x| separation.
    Simplification: use centroid-difference as weight direction.
    """
    # Subsample for speed
    n_a = len(X_a); n_b = len(X_b)
    if n_a > max_items:
        X_a = X_a[rng.choice(n_a, max_items, replace=False)]
    if n_b > max_items:
        X_b = X_b[rng.choice(n_b, max_items, replace=False)]
    # Centroid direction
    mu_a = X_a.mean(0); mu_b = X_b.mean(0)
    w = mu_a - mu_b
    w = w / np.maximum(np.linalg.norm(w), 1e-12)
    proj_a = X_a @ w
    proj_b = X_b @ w
    # Decision boundary at midpoint between projected means
    mid = (proj_a.mean() + proj_b.mean()) / 2
    # Margin stats
    sign = np.sign(proj_a.mean() - proj_b.mean())
    margins_a = sign * (proj_a - mid)
    margins_b = -sign * (proj_b - mid)
    acc = ((margins_a > 0).sum() + (margins_b > 0).sum()) / (len(margins_a) + len(margins_b))
    mean_margin = (margins_a.mean() + margins_b.mean()) / 2
    min_margin = float(min(margins_a.min(), margins_b.min()))
    mean_abs_separation = float(proj_a.mean() - proj_b.mean()) * sign
    return {
        'accuracy': float(acc),
        'mean_margin': float(mean_margin),
        'min_margin': min_margin,
        'mean_separation': float(mean_abs_separation),
        'n_a_used': int(len(X_a)), 'n_b_used': int(len(X_b)),
    }


def process_model(name, info):
    print(f"\n### {name}: {info['label']} ###", flush=True)
    X_id, p_id = load_items('id', info, 'id')
    X_ood, p_ood = load_items('ood', info, 'ood')
    print(f"  id items: n={len(X_id)}  ids={len(np.unique(p_id))}", flush=True)
    print(f"  ood items: n={len(X_ood)}  ids={len(np.unique(p_ood))}", flush=True)

    # 1. Centroid separation / RMS radius
    mu_id = X_id.mean(0)
    mu_ood = X_ood.mean(0)
    r_id = rms_radius(X_id, mu_id)
    r_ood = rms_radius(X_ood, mu_ood)
    sep = float(np.linalg.norm(mu_id - mu_ood))
    sep_in_r_id = sep / max(r_id, 1e-12)
    sep_in_r_ood = sep / max(r_ood, 1e-12)
    sep_in_rms_avg = sep / max((r_id + r_ood) / 2, 1e-12)
    print(f"  centroid sep={sep:.4f}  r_id={r_id:.4f}  r_ood={r_ood:.4f}", flush=True)
    print(f"  sep in units of: r_id={sep_in_r_id:.4f}  r_ood={sep_in_r_ood:.4f}  avg={sep_in_rms_avg:.4f}", flush=True)

    # 2. Blob-level nearest neighbours
    uids_id, cen_id = identity_centroids(X_id, p_id)
    uids_ood, cen_ood = identity_centroids(X_ood, p_ood)
    # For each OOD blob, nearest in pooled set
    all_cen = np.concatenate([cen_id, cen_ood], axis=0)
    all_lab = np.array(['id'] * len(cen_id) + ['ood'] * len(cen_ood))
    # For each ood blob, find nearest OTHER blob (exclude self)
    frac_ood_nn = 0
    n_ood = len(cen_ood)
    for i in range(n_ood):
        q = cen_ood[i]
        # distances to all centroids
        d2 = ((all_cen - q) ** 2).sum(axis=1)
        d2[len(cen_id) + i] = np.inf  # mask self
        nearest = d2.argmin()
        if all_lab[nearest] == 'ood':
            frac_ood_nn += 1
    frac_ood_nn = frac_ood_nn / n_ood
    print(f"  fraction of OOD blob centroids whose NN is another OOD blob: {frac_ood_nn:.4f}  (chance = {n_ood / (len(cen_id) + n_ood - 1):.4f})", flush=True)

    # 3. Linear probe margin
    rng = np.random.default_rng(RNG_SEED)
    probe = linear_probe_margin(X_id, X_ood, rng)
    print(f"  linear probe (id vs ood): accuracy={probe['accuracy']:.4f}  mean_margin={probe['mean_margin']:.4f}  min_margin={probe['min_margin']:.4f}  separation={probe['mean_separation']:.4f}", flush=True)
    # Within-domain control: split in-domain into two random halves
    n_id = len(X_id)
    perm = rng.permutation(n_id)
    half = n_id // 2
    X_a = X_id[perm[:half]]
    X_b = X_id[perm[half:half*2]]
    probe_ctrl = linear_probe_margin(X_a, X_b, rng)
    print(f"  linear probe control (id half1 vs id half2): accuracy={probe_ctrl['accuracy']:.4f}  mean_margin={probe_ctrl['mean_margin']:.4f}  separation={probe_ctrl['mean_separation']:.4f}", flush=True)

    return {
        'name': name, 'label': info['label'],
        'n_id_items': int(len(X_id)), 'n_ood_items': int(len(X_ood)),
        'n_id_blobs': int(len(uids_id)), 'n_ood_blobs': int(len(uids_ood)),
        'centroid_separation_raw': sep,
        'rms_radius_id': r_id, 'rms_radius_ood': r_ood,
        'sep_in_r_id': sep_in_r_id, 'sep_in_r_ood': sep_in_r_ood,
        'sep_in_rms_avg': sep_in_rms_avg,
        'frac_ood_blob_nn_is_ood': float(frac_ood_nn),
        'chance_ood_nn_fraction': float(n_ood / (len(cen_id) + n_ood - 1)),
        'linear_probe': probe,
        'within_domain_control': probe_ctrl,
    }


def main():
    out = {}
    for name, info in MODELS.items():
        out[name] = process_model(name, info)
    out_path = RESULTS / 'e43_E_loc.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {out_path}")

    print("\n\n========= E-loc summary =========")
    for name in MODELS:
        r = out[name]
        print(f"\n  {name}: {r['label']}")
        print(f"    centroid separation: {r['centroid_separation_raw']:.4f}")
        print(f"      = {r['sep_in_r_id']:.3f} x r_id = {r['sep_in_r_ood']:.3f} x r_ood = {r['sep_in_rms_avg']:.3f} x rms_avg")
        print(f"    fraction of OOD blob NN that is OOD: {r['frac_ood_blob_nn_is_ood']:.4f}  (chance={r['chance_ood_nn_fraction']:.4f})")
        print(f"    linear probe id-vs-ood: accuracy={r['linear_probe']['accuracy']:.4f}  margin={r['linear_probe']['mean_margin']:.4f}  separation={r['linear_probe']['mean_separation']:.4f}")
        print(f"    within-domain control:   accuracy={r['within_domain_control']['accuracy']:.4f}  margin={r['within_domain_control']['mean_margin']:.4f}  separation={r['within_domain_control']['mean_separation']:.4f}")


if __name__ == '__main__':
    main()
