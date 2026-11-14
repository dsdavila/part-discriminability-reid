"""
Within-camera radial-excess preview (rev 526 §2).

Does pre-normalisation Euclidean distance separate same-id from diff-id
*among same-camera pairs*? If yes, radial carries within-camera identity
info and I-1 has an incremental benefit over per-camera centring. If no,
I-1 is per-camera centring in disguise.

Cheap settlement before spending on the full (per-cam | clamp | both)
ablation.
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
COS_THRESHOLDS = [0.50, 0.70, 0.80, 0.90]

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


class DS(Dataset):
    def __init__(self, samples):
        self.samples = [s for s in samples if not s.is_junk]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        s = self.samples[i]
        return TRANSFORM(s.load_image()), int(s.person_id), int(s.camera_id)


@torch.no_grad()
def extract(model, samples, batch=64):
    ds = DS(samples)
    print(f"    extracting {len(ds)} items...", flush=True)
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
    chosen = []
    chosen_ids = []
    chosen_cams = []
    for k in ids:
        idx = id_to_idx[k]
        if len(idx) > max_per_id:
            idx = list(rng.choice(idx, size=max_per_id, replace=False))
        for i in idx:
            chosen.append(i)
            chosen_ids.append(k)
            chosen_cams.append(cams[i])
    return np.array(chosen), np.array(chosen_ids), np.array(chosen_cams)


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


def pair_auc(same_vals, diff_vals, max_pairs=200000):
    rng = np.random.default_rng(0)
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


def summarize(cos, eucl, same_id, mask, thr, label):
    thr_mask = cos >= thr
    both = mask & thr_mask
    if not both.any():
        return None
    e_sub = eucl[both]
    s_sub = same_id[both]
    same_vals = e_sub[s_sub]
    diff_vals = e_sub[~s_sub]
    return {
        'subset': label,
        'cos_threshold': float(thr),
        'n_pairs': int(both.sum()),
        'n_same': int(s_sub.sum()),
        'n_diff': int((~s_sub).sum()),
        'dprime': dprime(same_vals, diff_vals),
        'pair_auc': pair_auc(same_vals, diff_vals),
    }


def run_arm(name, ckpt_key, test_set_name, models_map, market, msmt, rng):
    print(f"\n### {name} ###", flush=True)
    model = models_map[ckpt_key]
    test_ds = market if test_set_name == 'market' else msmt
    n_q = sum(1 for s in test_ds.query if not s.is_junk)
    gallery_samples = [s for s in (test_ds.query + test_ds.gallery)[n_q:]
                       if not s.is_junk]
    feats, labels, cams = extract(model, gallery_samples)
    chosen, chosen_ids, chosen_cams = sample_pool(feats, labels, cams, rng)
    X = feats[chosen]
    ids = chosen_ids
    cams_pool = chosen_cams
    print(f"    pool: {len(X)} feats, {len(set(ids.tolist()))} multi-cam ids, "
          f"{len(set(cams_pool.tolist()))} cameras", flush=True)
    cos, eucl, same_id, same_cam = compute_pair_stats(X, ids, cams_pool)
    cross_cam = ~same_cam

    n_same_cam_same_id = int((same_cam & same_id).sum())
    n_same_cam_diff_id = int((same_cam & ~same_id).sum())
    n_cross_cam_same_id = int((cross_cam & same_id).sum())
    n_cross_cam_diff_id = int((cross_cam & ~same_id).sum())
    print(f"    pair counts: "
          f"same_cam_same_id={n_same_cam_same_id}  "
          f"same_cam_diff_id={n_same_cam_diff_id}  "
          f"cross_cam_same_id={n_cross_cam_same_id}  "
          f"cross_cam_diff_id={n_cross_cam_diff_id}", flush=True)

    top_decile_thr = float(np.quantile(cos, 0.90))
    print(f"    top-decile cos threshold: {top_decile_thr:.4f}", flush=True)

    out = {
        'n_same_cam_same_id': n_same_cam_same_id,
        'n_same_cam_diff_id': n_same_cam_diff_id,
        'n_cross_cam_same_id': n_cross_cam_same_id,
        'n_cross_cam_diff_id': n_cross_cam_diff_id,
        'top_decile_cos_threshold': top_decile_thr,
        'same_camera': {},
        'cross_camera': {},
    }

    thr_list = sorted(set(COS_THRESHOLDS + [top_decile_thr]))
    for thr in thr_list:
        sc = summarize(cos, eucl, same_id, same_cam, thr, 'same_cam')
        cc = summarize(cos, eucl, same_id, cross_cam, thr, 'cross_cam')
        if sc:
            out['same_camera'][f"{thr:.4f}"] = sc
        if cc:
            out['cross_camera'][f"{thr:.4f}"] = cc
        sc_dp = sc['dprime'] if sc else float('nan')
        sc_auc = sc['pair_auc'] if sc else float('nan')
        sc_n = sc['n_pairs'] if sc else 0
        sc_s = sc['n_same'] if sc else 0
        cc_dp = cc['dprime'] if cc else float('nan')
        cc_auc = cc['pair_auc'] if cc else float('nan')
        print(f"    thr={thr:.4f}  SAME-CAM d'={sc_dp:.3f} AUC={sc_auc:.3f} n={sc_n} n_same={sc_s}  |  "
              f"CROSS-CAM d'={cc_dp:.3f} AUC={cc_auc:.3f}", flush=True)
    return out


def main():
    print(f"device: {device}", flush=True)
    market = get_dataset('market1501', sample=False)
    msmt = get_dataset('msmt17', sample=False)
    rng = np.random.default_rng(RNG_SEED)
    print("Loading models...", flush=True)
    models_map = {}
    for key, ckpt in [('market', CKPT_MARKET), ('msmt', CKPT_MSMT)]:
        m = BoTSpatial().to(device).eval()
        load_into(m, ckpt)
        models_map[key] = m

    all_out = {}
    for arm, info in ARMS.items():
        all_out[arm] = run_arm(arm, info['ckpt'], info['test_set'],
                               models_map, market, msmt, rng)

    outp = RESULTS / 'e44_i5_radial_withincam.json'
    with open(outp, 'w') as f:
        json.dump(all_out, f, indent=2)
    print(f"\nSaved {outp}")


if __name__ == '__main__':
    main()
