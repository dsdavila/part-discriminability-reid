"""
BNNeck survival test (rev 529 §3).

Does within-camera radial identity signal (I-5's d'=1.92 on B', 3.67 on
A') reach the post-BNNeck retrieval space, or does BN+L2 destroy it?

BoT's BNNeck is a BatchNorm1d(2048) + L2 normalisation. Cosine on the
post-BN L2-normed vectors is what retrieval actually scores on.

Pre-BN scale -> post-BN direction coupling: BN subtracts the running
mean, so changing a vector's pre-BN scale rotates its post-BN direction.
That coupling is what could preserve the radial signal in a form that
post-BN cosine can see.

For each arm, at top-decile cosine (defined on pre-BN direction to match
I-5), compute d'/AUC of SAME-CAMERA identity discrimination under two
distance definitions:
  (a) pre-BN Euclidean (I-5 baseline)
  (b) post-BN cosine distance (1 - cos_post_bn)

If (b) >> 0, the signal survives into the retrieval space. If (b) ~ 0,
clamping pre-BN radial costs nothing at retrieval and the I-1 cost
argument is empirically zero.
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


class BoTBackbone(nn.Module):
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


def load_backbone_and_bn(ckpt_path):
    sd_raw = torch.load(ckpt_path, map_location=device, weights_only=False)
    backbone = BoTBackbone().to(device).eval()
    MAP = {'0.0.': 'stem_conv.', '0.1.': 'stem_bn.',
           '0.4.': 'layer1.', '0.5.': 'layer2.',
           '0.6.': 'layer3.', '0.7.': 'layer4.'}
    sd = {}
    bn_sd = {}
    for k, v in sd_raw.items():
        if 'num_batches_tracked' in k:
            continue
        if k.startswith('3.'):
            bn_sd[k[2:]] = v
            continue
        for prefix, target in MAP.items():
            if k.startswith(prefix):
                sd[target + k[len(prefix):]] = v
                break
    backbone.load_state_dict(sd, strict=False)

    bn = nn.BatchNorm1d(2048, affine=True, track_running_stats=True).to(device).eval()
    bn.weight.data = bn_sd['weight'].to(device)
    bn.bias.data = bn_sd['bias'].to(device)
    bn.running_mean.data = bn_sd['running_mean'].to(device)
    bn.running_var.data = bn_sd['running_var'].to(device)
    return backbone, bn


class DStest(Dataset):
    def __init__(self, samples):
        self.samples = [s for s in samples if not s.is_junk]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        s = self.samples[i]
        return TRANSFORM(s.load_image()), int(s.person_id), int(s.camera_id)


@torch.no_grad()
def extract_both(backbone, bn, samples, batch=64):
    ds = DStest(samples)
    print(f"    extracting {len(ds)} test items...", flush=True)
    ldr = DataLoader(ds, batch_size=batch, num_workers=0, pin_memory=False)
    pre, post_bn, labels, cams = [], [], [], []
    for imgs, lbls, cm in ldr:
        f_pre = backbone(imgs.to(device, non_blocking=True))
        f_bn = bn(f_pre)
        pre.append(f_pre.cpu().numpy())
        post_bn.append(f_bn.cpu().numpy())
        labels.append(lbls.numpy())
        cams.append(cm.numpy())
    return (np.vstack(pre), np.vstack(post_bn),
            np.concatenate(labels), np.concatenate(cams))


def sample_pool(y, cams, rng, max_ids=MAX_IDS, max_per_id=MAX_IMGS_PER_ID):
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


def pair_stats(X, ids, cams):
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


def measure(dist_vals, same_id, same_cam, thr_mask, subset_mask, label):
    mask = subset_mask & thr_mask
    d_sub = dist_vals[mask]
    s_sub = same_id[mask]
    same_vals = d_sub[s_sub]
    diff_vals = d_sub[~s_sub]
    return {
        'label': label,
        'n_pairs': int(mask.sum()),
        'n_same': int(s_sub.sum()),
        'n_diff': int((~s_sub).sum()),
        'dprime': dprime(same_vals, diff_vals),
        'auc': pair_auc(same_vals, diff_vals),
        'mean_same': float(same_vals.mean()) if len(same_vals) else float('nan'),
        'mean_diff': float(diff_vals.mean()) if len(diff_vals) else float('nan'),
    }


def main():
    print(f"device: {device}", flush=True)
    market = get_dataset('market1501', sample=False)
    msmt = get_dataset('msmt17', sample=False)
    rng = np.random.default_rng(RNG_SEED)
    print("Loading backbones + BNNeck...", flush=True)
    m_mkt_bb, m_mkt_bn = load_backbone_and_bn(CKPT_MARKET)
    m_msmt_bb, m_msmt_bn = load_backbone_and_bn(CKPT_MSMT)
    models_map = {'market': (m_mkt_bb, m_mkt_bn), 'msmt': (m_msmt_bb, m_msmt_bn)}

    all_out = {}
    for arm, info in ARMS.items():
        print(f"\n### arm {arm} ###", flush=True)
        backbone, bn = models_map[info['ckpt']]
        test_ds = market if info['test_set'] == 'market' else msmt
        n_q = sum(1 for s in test_ds.query if not s.is_junk)
        gallery_samples = [s for s in (test_ds.query + test_ds.gallery)[n_q:]
                           if not s.is_junk]
        X_pre, X_post, y, cams = extract_both(backbone, bn, gallery_samples)
        chosen, pool_ids, pool_cams = sample_pool(y, cams, rng)
        X_pre_p = X_pre[chosen]
        X_post_p = X_post[chosen]
        print(f"  pool: {len(X_pre_p)} feats, {len(set(pool_ids.tolist()))} multi-cam ids",
              flush=True)

        # Pre-BN stats
        cos_pre, eucl_pre, same_id, same_cam = pair_stats(X_pre_p, pool_ids, pool_cams)
        # Post-BN stats (use cos post-BN as distance via 1-cos)
        cos_post, eucl_post_bn, _, _ = pair_stats(X_post_p, pool_ids, pool_cams)
        cos_post_dist = 1.0 - cos_post

        thr_pre = float(np.quantile(cos_pre, 0.90))
        thr_post = float(np.quantile(cos_post, 0.90))

        thr_pre_mask = cos_pre >= thr_pre
        thr_post_mask = cos_post >= thr_post

        arm_out = {
            'top_decile_cos_pre': thr_pre,
            'top_decile_cos_post': thr_post,
            'same_cam_at_thr_pre_pre_eucl': measure(eucl_pre, same_id, same_cam,
                                                    thr_pre_mask, same_cam,
                                                    'same_cam_at_thr_pre_pre_eucl'),
            'same_cam_at_thr_pre_post_cos': measure(cos_post_dist, same_id, same_cam,
                                                    thr_pre_mask, same_cam,
                                                    'same_cam_at_thr_pre_post_cos'),
            'same_cam_at_thr_post_post_cos': measure(cos_post_dist, same_id, same_cam,
                                                     thr_post_mask, same_cam,
                                                     'same_cam_at_thr_post_post_cos'),
            'same_cam_at_thr_post_pre_eucl': measure(eucl_pre, same_id, same_cam,
                                                     thr_post_mask, same_cam,
                                                     'same_cam_at_thr_post_pre_eucl'),
            'cross_cam_at_thr_post_post_cos': measure(cos_post_dist, same_id, ~same_cam,
                                                      thr_post_mask, ~same_cam,
                                                      'cross_cam_at_thr_post_post_cos'),
        }

        a = arm_out['same_cam_at_thr_pre_pre_eucl']
        b = arm_out['same_cam_at_thr_pre_post_cos']
        c = arm_out['same_cam_at_thr_post_post_cos']
        d = arm_out['same_cam_at_thr_post_pre_eucl']
        e = arm_out['cross_cam_at_thr_post_post_cos']
        print(f"  thr_pre={thr_pre:.4f}  thr_post={thr_post:.4f}", flush=True)
        print(f"  [thr_pre]  same-cam PRE-BN eucl  d'={a['dprime']:.3f}  AUC={a['auc']:.3f}  "
              f"n={a['n_pairs']}", flush=True)
        print(f"  [thr_pre]  same-cam POST-BN cos  d'={b['dprime']:.3f}  AUC={b['auc']:.3f}  "
              f"n={b['n_pairs']}", flush=True)
        print(f"  [thr_post] same-cam POST-BN cos  d'={c['dprime']:.3f}  AUC={c['auc']:.3f}  "
              f"n={c['n_pairs']}", flush=True)
        print(f"  [thr_post] same-cam PRE-BN eucl  d'={d['dprime']:.3f}  AUC={d['auc']:.3f}  "
              f"n={d['n_pairs']}", flush=True)
        print(f"  [thr_post] cross-cam POST-BN cos d'={e['dprime']:.3f}  AUC={e['auc']:.3f}  "
              f"n={e['n_pairs']}", flush=True)

        all_out[arm] = arm_out

    outp = RESULTS / 'e44_i5_bnneck_survival.json'
    with open(outp, 'w') as f:
        json.dump(all_out, f, indent=2)
    print(f"\nSaved {outp}")


if __name__ == '__main__':
    main()
