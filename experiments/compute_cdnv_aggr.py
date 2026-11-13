"""
CDNV aggregation fix (rev 504 §1): bring num/den split onto the SAME aggregation
as the headline CDNV (mean-of-ratios). Previously num_mean, den_mean lived in the
ratio-of-means family while the headline was mean-of-ratios — 2.1-2.4x Jensen gap.

Reports per stage per arm:
  - mean_num, mean_den (ratio-of-means family), mean_ratio (headline)
  - median_num, median_den, median_ratio (robust)
  - Hard-quartile (bottom 25% of per-pair den): mean and median of (num, den, ratio)
  - Easy-quartile (top 25% of per-pair den): same
  - Per-pair full distribution saved for layer4 only (for plotting if needed)

If attribution holds on median AND on the hard-pair subset AND on the full
population, §C is solid. If it flips on hard pairs, that is itself the finding.
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
MAX_IDS = 200
MIN_IMGS = 2

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
STAGES = ['stem', 'layer1', 'layer2', 'layer3', 'layer4']


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
        x = self.stem_conv(x); x = self.stem_bn(x); x = self.stem_relu(x); x = self.stem_pool(x)
        stem = x; l1 = self.layer1(x); l2 = self.layer2(l1); l3 = self.layer3(l2); l4 = self.layer4(l3)
        out = {}
        for name, feat in [('stem', stem), ('layer1', l1), ('layer2', l2), ('layer3', l3), ('layer4', l4)]:
            out[name] = feat.mean(dim=(2, 3))
        return out


def load_into(model, ckpt_path):
    sd_raw = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = {}
    MAP = {'0.0.': 'stem_conv.', '0.1.': 'stem_bn.',
           '0.4.': 'layer1.', '0.5.': 'layer2.',
           '0.6.': 'layer3.', '0.7.': 'layer4.'}
    for k, v in sd_raw.items():
        if 'num_batches_tracked' in k: continue
        for prefix, target in MAP.items():
            if k.startswith(prefix):
                sd[target + k[len(prefix):]] = v; break
    model.load_state_dict(sd, strict=False)
    return model


class DS(Dataset):
    def __init__(self, samples):
        self.samples = [s for s in samples if not s.is_junk]
    def __len__(self): return len(self.samples)
    def __getitem__(self, i):
        s = self.samples[i]
        return TRANSFORM(s.load_image()), int(s.person_id)


@torch.no_grad()
def extract_gap(model, samples, batch=64):
    ds = DS(samples)
    print(f"    extracting {len(ds)} items...", flush=True)
    ldr = DataLoader(ds, batch_size=batch, num_workers=0, pin_memory=False)
    stages = {s: [] for s in STAGES}
    labels = []
    for imgs, lbls in ldr:
        out = model(imgs.to(device, non_blocking=True))
        for s in STAGES:
            stages[s].append(out[s].cpu().numpy())
        labels.append(lbls.numpy())
    return {s: np.vstack(v) for s, v in stages.items()}, np.concatenate(labels)


def cdnv_full(X, y, rng, max_ids=200, save_pairs=False):
    id_to_idx = defaultdict(list)
    for i, lbl in enumerate(y):
        id_to_idx[int(lbl)].append(i)
    ids = [k for k, v in id_to_idx.items() if len(v) >= MIN_IMGS]
    if len(ids) > max_ids:
        ids = list(rng.choice(ids, size=max_ids, replace=False))
    means = {}
    within_var = {}
    for k in ids:
        Xk = X[id_to_idx[k]]
        mu = Xk.mean(axis=0)
        means[k] = mu
        within_var[k] = float(((Xk - mu) ** 2).sum(axis=1).mean())
    mu_arr = np.array([means[k] for k in ids])
    wv_arr = np.array([within_var[k] for k in ids])
    K = len(ids)
    if K < 2:
        return dict(K=K)
    # Collect per-pair arrays
    nums = []; dens = []
    for i in range(K):
        diff = mu_arr - mu_arr[i:i + 1]
        d2 = (diff * diff).sum(axis=1)
        for j in range(i + 1, K):
            num = wv_arr[i] + wv_arr[j]
            den = 2.0 * d2[j]
            if den > 1e-12:
                nums.append(num); dens.append(den)
    nums = np.array(nums); dens = np.array(dens)
    ratios = nums / dens

    def summarize(mask=None):
        n = nums[mask] if mask is not None else nums
        d = dens[mask] if mask is not None else dens
        r = ratios[mask] if mask is not None else ratios
        return {
            'count': int(len(n)),
            'mean_num': float(n.mean()), 'median_num': float(np.median(n)),
            'mean_den': float(d.mean()), 'median_den': float(np.median(d)),
            'mean_ratio': float(r.mean()), 'median_ratio': float(np.median(r)),
        }

    # Hard quartile: lowest 25% of denominators (closest-mean pairs)
    den_q25 = np.quantile(dens, 0.25)
    den_q75 = np.quantile(dens, 0.75)
    hard = dens <= den_q25
    easy = dens >= den_q75

    out = {
        'K': K,
        'all': summarize(None),
        'hard_q25_by_den': summarize(hard),
        'easy_q75_by_den': summarize(easy),
        'den_q25_threshold': float(den_q25),
        'den_q75_threshold': float(den_q75),
    }
    if save_pairs:
        out['pair_num'] = nums.tolist()
        out['pair_den'] = dens.tolist()
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
        print(f"\n### {arm} ###", flush=True)
        model = models_map[info['ckpt']]
        test_ds = market if info['test_set'] == 'market' else msmt
        n_q = sum(1 for s in test_ds.query if not s.is_junk)
        test_g = [s for s in (test_ds.query + test_ds.gallery)[n_q:] if not s.is_junk]
        feats, labels = extract_gap(model, test_g)
        arm_out = {}
        for stage in STAGES:
            save = (stage == 'layer4')
            r = cdnv_full(feats[stage], labels, rng, max_ids=MAX_IDS, save_pairs=save)
            arm_out[stage] = r
            a = r['all']; h = r['hard_q25_by_den']; e = r['easy_q75_by_den']
            print(f"  {stage:<8s}: all mean_num={a['mean_num']:.4f} mean_den={a['mean_den']:.4f} "
                  f"mean_r={a['mean_ratio']:.4f} med_r={a['median_ratio']:.4f}  "
                  f"HARD med_r={h['median_ratio']:.4f} med_num={h['median_num']:.4f} med_den={h['median_den']:.4f}  "
                  f"EASY med_r={e['median_ratio']:.4f}", flush=True)
        all_out[arm] = arm_out

    with open(RESULTS / 'e44_cdnv_aggr.json', 'w') as f:
        json.dump(all_out, f, indent=2)
    print(f"\nSaved {RESULTS / 'e44_cdnv_aggr.json'}")


if __name__ == '__main__':
    main()
