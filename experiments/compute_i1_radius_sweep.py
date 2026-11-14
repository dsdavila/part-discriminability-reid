"""
I-1 radius sweep (rev 529 §2, promoted after rev 531's retraction).

My rev 528a inference was that any angular clamp destroys within-camera
radial identity info. But I-1 as specced (rev 508 §0) is a TAIL CLAMP at
radius r, identity inside the ball. At a generous radius it touches only
the tail.

Sweep r over {p50, p75, p90, p95, p99} of the training-distance
distribution. Recompute same-camera d' / AUC on clamped features per
arm. Cost-vs-radius curve. If d' barely moves at p95 and only collapses
at p50, the generous-radius clamp is nearly free.

Reference center: training-set mean per model (Market-trained -> Market
training mean, MSMT-trained -> MSMT training mean). Radius thresholds
from the training-distance distribution (same training set, same model).
Pre-BNNeck layer4 GAP (2048-d), same point as I-5.
"""
import json
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
from collections import defaultdict
sys.path.insert(0, '/workspace/reid-tk')
from reid_tk.datasets import get_dataset

RESULTS = Path('/data/papers/low_dimensional_reid/results')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
RNG_SEED = 42
MAX_IDS = 400
MIN_IMGS = 2
MAX_IMGS_PER_ID = 6
PCTILES = [50, 75, 90, 95, 99]
TOP_DECILE = True  # headline is top-decile cosine subset

CKPT_MARKET = Path('/data/papers/low_dimensional_reid/pretrained/rea_ablation/bot_resnet50_market1501_seed42_final.pth')
CKPT_MSMT   = Path('/data/papers/low_dimensional_reid/pretrained/bot_resnet50_msmt17_seed42_final.pth')

TRANSFORM = T.Compose([T.Resize((256, 128)), T.ToTensor(),
                       T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

ARMS = {
    'A_prime': {'ckpt': 'market', 'test_set': 'market'},
    'B_prime': {'ckpt': 'market', 'test_set': 'msmt'},
    'C':       {'ckpt': 'msmt',   'test_set': 'market'},
    'D':       {'ckpt': 'msmt',   'test_set': 'msmt'},
}


class BoTSpatial(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet50(weights=None)
        resnet.layer4[0].conv2.stride = (1, 1)
        resnet.layer4[0].downsample[0].stride = (1, 1)
        self.stem_conv = resnet.conv1; self.stem_bn = resnet.bn1
        self.stem_relu = resnet.relu; self.stem_pool = resnet.maxpool
        self.layer1 = resnet.layer1; self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3; self.layer4 = resnet.layer4

    def forward(self, x):
        x = self.stem_conv(x); x = self.stem_bn(x)
        x = self.stem_relu(x); x = self.stem_pool(x)
        x = self.layer1(x); x = self.layer2(x)
        x = self.layer3(x); x = self.layer4(x)
        return x.mean(dim=(2, 3))


def load_into(model, ckpt_path):
    sd_raw = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = {}
    MAP = {'0.0.': 'stem_conv.', '0.1.': 'stem_bn.',
           '0.4.': 'layer1.', '0.5.': 'layer2.',
           '0.6.': 'layer3.', '0.7.': 'layer4.'}
    for k, v in sd_raw.items():
        if 'num_batches_tracked' in k:
            continue
        for prefix, target in MAP.items():
            if k.startswith(prefix):
                sd[target + k[len(prefix):]] = v
                break
    model.load_state_dict(sd, strict=False)
    return model


class DSimg(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        s = self.samples[i]
        return TRANSFORM(s.load_image())


class DStest(Dataset):
    def __init__(self, samples):
        self.samples = [s for s in samples if not s.is_junk]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        s = self.samples[i]
        return TRANSFORM(s.load_image()), int(s.person_id), int(s.camera_id)


@torch.no_grad()
def extract_train(model, samples, batch=64):
    ds = DSimg(samples)
    print(f"    extracting {len(ds)} training items...", flush=True)
    ldr = DataLoader(ds, batch_size=batch, num_workers=0, pin_memory=False)
    feats = []
    for imgs in ldr:
        out = model(imgs.to(device, non_blocking=True))
        feats.append(out.cpu().numpy())
    return np.vstack(feats)


@torch.no_grad()
def extract_test(model, samples, batch=64):
    ds = DStest(samples)
    print(f"    extracting {len(ds)} test items...", flush=True)
    ldr = DataLoader(ds, batch_size=batch, num_workers=0, pin_memory=False)
    feats, labels, cams = [], [], []
    for imgs, lbls, cm in ldr:
        out = model(imgs.to(device, non_blocking=True))
        feats.append(out.cpu().numpy())
        labels.append(lbls.numpy())
        cams.append(cm.numpy())
    return np.vstack(feats), np.concatenate(labels), np.concatenate(cams)


def sample_pool(X, y, cams, rng, max_ids=MAX_IDS, max_per_id=MAX_IMGS_PER_ID):
    id_to_idx = defaultdict(list)
    for i, lbl in enumerate(y):
        id_to_idx[int(lbl)].append(i)
    ids = [k for k, v in id_to_idx.items()
           if len(v) >= MIN_IMGS and len(set(cams[i] for i in v)) >= 2]
    if len(ids) > max_ids:
        ids = list(rng.choice(ids, size=max_ids, replace=False))
    chosen, chosen_ids, chosen_cams = [], [], []
    for k in ids:
        idx = id_to_idx[k]
        if len(idx) > max_per_id:
            idx = list(rng.choice(idx, size=max_per_id, replace=False))
        for i in idx:
            chosen.append(i)
            chosen_ids.append(k)
            chosen_cams.append(cams[i])
    return np.array(chosen), np.array(chosen_ids), np.array(chosen_cams)


def apply_tail_clamp(X, mu, r):
    """Clamp features beyond radius r back to the ball of radius r around mu."""
    delta = X - mu
    d = np.linalg.norm(delta, axis=1, keepdims=True)
    scale = np.minimum(1.0, r / (d + 1e-12))
    return mu + delta * scale


def compute_pair_stats(X, ids, cams):
    n = X.shape[0]
    norms = np.linalg.norm(X, axis=1)
    dots = X @ X.T
    cos = dots / (norms[:, None] * norms[None, :] + 1e-12)
    d2 = (norms[:, None] ** 2) + (norms[None, :] ** 2) - 2 * dots
    d2 = np.clip(d2, 0, None)
    eucl = np.sqrt(d2)
    same_id = ids[:, None] == ids[None, :]
    same_cam = cams[:, None] == cams[None, :]
    iu = np.triu_indices(n, k=1)
    return cos[iu], eucl[iu], same_id[iu], same_cam[iu]


def dprime(same_vals, diff_vals):
    if len(same_vals) < 2 or len(diff_vals) < 2:
        return float('nan')
    ms, md = float(same_vals.mean()), float(diff_vals.mean())
    vs, vd = float(same_vals.var(ddof=1)), float(diff_vals.var(ddof=1))
    denom = (0.5 * (vs + vd)) ** 0.5
    if denom < 1e-12:
        return float('nan')
    return (md - ms) / denom


def pair_auc(same_vals, diff_vals, max_pairs=200000, seed=0):
    rng = np.random.default_rng(seed)
    n_same = len(same_vals)
    n_diff = len(diff_vals)
    if n_same == 0 or n_diff == 0:
        return float('nan')
    m = min(max_pairs, n_same * n_diff)
    si = rng.integers(0, n_same, size=m)
    di = rng.integers(0, n_diff, size=m)
    wins = (same_vals[si] < diff_vals[di]).sum()
    ties = (same_vals[si] == diff_vals[di]).sum()
    return float((wins + 0.5 * ties) / m)


def measure_same_cam(X, ids, cams, label):
    cos, eucl, same_id, same_cam = compute_pair_stats(X, ids, cams)
    top_decile_thr = float(np.quantile(cos, 0.90))
    thr_mask = cos >= top_decile_thr
    mask = same_cam & thr_mask
    e_sub = eucl[mask]
    s_sub = same_id[mask]
    same_vals = e_sub[s_sub]
    diff_vals = e_sub[~s_sub]
    dp = dprime(same_vals, diff_vals)
    auc = pair_auc(same_vals, diff_vals)
    n = int(mask.sum())
    n_s = int(s_sub.sum())
    return {
        'label': label,
        'top_decile_cos_threshold': top_decile_thr,
        'n_pairs': n,
        'n_same': n_s,
        'dprime_same_cam': dp,
        'auc_same_cam': auc,
    }


def main():
    print(f"device: {device}", flush=True)
    market = get_dataset('market1501', sample=False)
    msmt = get_dataset('msmt17', sample=False)
    rng = np.random.default_rng(RNG_SEED)
    print("Loading models...", flush=True)
    m_mkt = BoTSpatial().to(device).eval()
    load_into(m_mkt, CKPT_MARKET)
    m_msmt = BoTSpatial().to(device).eval()
    load_into(m_msmt, CKPT_MSMT)
    models_map = {'market': m_mkt, 'msmt': m_msmt}

    # Training features and per-model reference mean + distance quantiles
    ref = {}
    for key, model, ds_name, ds in [('market', m_mkt, 'market1501', market),
                                     ('msmt', m_msmt, 'msmt17', msmt)]:
        print(f"\n### training ref for {ds_name} ###", flush=True)
        train_samples = ds.train
        X_tr = extract_train(model, train_samples)
        mu = X_tr.mean(axis=0)
        d_tr = np.linalg.norm(X_tr - mu, axis=1)
        qvals = {f"p{p}": float(np.quantile(d_tr, p / 100.0)) for p in PCTILES}
        ref[key] = {'mu': mu, 'qvals': qvals, 'n_train': len(X_tr),
                    'mean_dist': float(d_tr.mean()), 'median_dist': float(np.median(d_tr))}
        print(f"  n_train={len(X_tr)}  mean_dist={d_tr.mean():.3f}  median_dist={np.median(d_tr):.3f}",
              flush=True)
        print(f"  quantiles: " + "  ".join(f"{k}={v:.3f}" for k, v in qvals.items()),
              flush=True)
        del X_tr

    # Per-arm test features + clamped variants
    all_out = {}
    for arm, info in ARMS.items():
        print(f"\n### arm {arm} ###", flush=True)
        model = models_map[info['ckpt']]
        test_ds = market if info['test_set'] == 'market' else msmt
        n_q = sum(1 for s in test_ds.query if not s.is_junk)
        gallery_samples = [s for s in (test_ds.query + test_ds.gallery)[n_q:]
                           if not s.is_junk]
        X, y, cams = extract_test(model, gallery_samples)
        chosen, pool_ids, pool_cams = sample_pool(X, y, cams, rng)
        X_pool = X[chosen]
        print(f"  pool: {len(X_pool)} feats, {len(set(pool_ids.tolist()))} multi-cam ids", flush=True)

        mu = ref[info['ckpt']]['mu']
        qvals = ref[info['ckpt']]['qvals']

        arm_out = {'ref_model': info['ckpt'], 'quantiles': qvals, 'radii': {}}

        # Baseline (no clamp)
        base = measure_same_cam(X_pool, pool_ids, pool_cams, 'baseline')
        arm_out['baseline'] = base
        print(f"  baseline: same-cam d'={base['dprime_same_cam']:.3f}  "
              f"AUC={base['auc_same_cam']:.3f}  n={base['n_pairs']}  "
              f"thr={base['top_decile_cos_threshold']:.4f}", flush=True)

        # Clamp at each radius
        for p in PCTILES:
            r = qvals[f"p{p}"]
            X_clamped = apply_tail_clamp(X_pool, mu, r)
            d_new = np.linalg.norm(X_clamped - mu, axis=1)
            frac_clamped = float((np.linalg.norm(X_pool - mu, axis=1) > r).mean())
            res = measure_same_cam(X_clamped, pool_ids, pool_cams, f"p{p}")
            res['radius_r'] = r
            res['fraction_clamped'] = frac_clamped
            res['dprime_delta'] = res['dprime_same_cam'] - base['dprime_same_cam']
            res['auc_delta'] = res['auc_same_cam'] - base['auc_same_cam']
            arm_out['radii'][f"p{p}"] = res
            print(f"  r=p{p:>2d}={r:.3f}  frac_clamped={frac_clamped:.3f}  "
                  f"d'={res['dprime_same_cam']:.3f} (Δ{res['dprime_delta']:+.3f})  "
                  f"AUC={res['auc_same_cam']:.3f} (Δ{res['auc_delta']:+.3f})", flush=True)

        all_out[arm] = arm_out

    all_out['_reference'] = {
        'market': {'quantiles': ref['market']['qvals'],
                   'n_train': ref['market']['n_train'],
                   'mean_dist': ref['market']['mean_dist'],
                   'median_dist': ref['market']['median_dist']},
        'msmt': {'quantiles': ref['msmt']['qvals'],
                 'n_train': ref['msmt']['n_train'],
                 'mean_dist': ref['msmt']['mean_dist'],
                 'median_dist': ref['msmt']['median_dist']},
    }

    outp = RESULTS / 'e44_i1_radius_sweep.json'
    with open(outp, 'w') as f:
        json.dump(all_out, f, indent=2)
    print(f"\nSaved {outp}")


if __name__ == '__main__':
    main()
