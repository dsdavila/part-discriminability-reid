"""
Per-channel survival of the trained-over-null gain (Daniel via claude-chat,
Channel 4 rev 660 §4).

"We know from our experiments that the OOD samples use SOME of the pathways
from the training set, because they are ranking better than the totally
untrained null network. Where do the wet noodles align and where do they
fall off the rails?"

Per channel, per layer, compute per-channel CDNV (identity discriminability
restricted to a single channel -- the existing CDNV formula from
compute_cdnv.py/compute_cdnv_imagenet.py, decomposed per-dimension instead
of traced over all dimensions) under three conditions:

  null:       ImageNet-init backbone (BoTSpatial_IN, existing class) on OOD data
  ceiling:    trained backbone on IN-DOMAIN data
  OOD:        trained backbone on OOD data (the thing being located)

Then, per channel per layer (lower CDNV = more discriminable, so this is a
*distance* in the opposite sense of discriminability -- survival is framed
on discriminability, so we invert: discriminability = 1/CDNV, or more
robustly use (1/CDNV) since CDNV can be noisy/unstable near zero denominator):

    survival = (disc_OOD - disc_null) / (disc_ceiling - disc_null)

Channels near 1 are pathways the OOD sample still uses as well as the
in-domain ceiling; channels near 0 are where it has fallen back to the
untrained null. Report the distribution per layer, not just the mean
(per Daniel's explicit ask) -- if bimodal, that bimodality is the finding.

This is a channel-level (not spatial-cell-level) statistic, so it cannot
degenerate into a person detector the way the cell-residual line did.

Writes /data/papers/low_dimensional_reid/results/e44_per_channel_survival.json
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
CKPT_MSMT = Path('/data/papers/low_dimensional_reid/pretrained/bot_resnet50_msmt17_seed42_final.pth')

TRANSFORM = T.Compose([T.Resize((256, 128)), T.ToTensor(),
                       T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
STAGES = ['layer1', 'layer2', 'layer3', 'layer4']


class BoTSpatial(nn.Module):
    """Trained backbone, GAP per stage. Same class as compute_cdnv.py."""
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
        l1 = self.layer1(x); l2 = self.layer2(l1); l3 = self.layer3(l2); l4 = self.layer4(l3)
        out = {}
        for name, feat in [('layer1', l1), ('layer2', l2), ('layer3', l3), ('layer4', l4)]:
            out[name] = feat.mean(dim=(2, 3))
        return out


class BoTSpatial_IN(nn.Module):
    """ImageNet-init backbone, same architecture. Same class as
    compute_cdnv_imagenet.py -- the 'null' condition."""
    def __init__(self):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        resnet.layer4[0].conv2.stride = (1, 1)
        resnet.layer4[0].downsample[0].stride = (1, 1)
        self.stem_conv = resnet.conv1; self.stem_bn = resnet.bn1
        self.stem_relu = resnet.relu; self.stem_pool = resnet.maxpool
        self.layer1 = resnet.layer1; self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3; self.layer4 = resnet.layer4

    def forward(self, x):
        x = self.stem_conv(x); x = self.stem_bn(x); x = self.stem_relu(x); x = self.stem_pool(x)
        l1 = self.layer1(x); l2 = self.layer2(l1); l3 = self.layer3(l2); l4 = self.layer4(l3)
        out = {}
        for name, feat in [('layer1', l1), ('layer2', l2), ('layer3', l3), ('layer4', l4)]:
            out[name] = feat.mean(dim=(2, 3))
        return out


def load_trained(ckpt_path):
    sd_raw = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = BoTSpatial().to(device).eval()
    MAP = {'0.0.': 'stem_conv.', '0.1.': 'stem_bn.',
           '0.4.': 'layer1.', '0.5.': 'layer2.',
           '0.6.': 'layer3.', '0.7.': 'layer4.'}
    sd = {}
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


def per_channel_cdnv(X, y, rng, max_ids=MAX_IDS):
    """X: (N, D). y: (N,). Returns per-channel CDNV ratio: (D,) array, the
    SAME formula as compute_cdnv.py's scalar CDNV, decomposed per dimension
    instead of traced/summed over all dimensions.

    num_c (per identity k) = per-channel within-identity variance at channel c
    den_c (per pair i,j)   = per-channel squared mean-difference at channel c
    ratio_c = mean over all identity pairs of (num_i,c + num_j,c) / (2 * den_ij,c)
    """
    id_to_idx = defaultdict(list)
    for i, lbl in enumerate(y):
        id_to_idx[int(lbl)].append(i)
    ids = [k for k, v in id_to_idx.items() if len(v) >= MIN_IMGS]
    if len(ids) > max_ids:
        ids = list(rng.choice(ids, size=max_ids, replace=False))
    D = X.shape[1]
    K = len(ids)
    if K < 2:
        return np.full(D, np.nan), K

    means = np.zeros((K, D), dtype=np.float64)
    within_var = np.zeros((K, D), dtype=np.float64)  # per-channel variance, NOT traced
    for ki, k in enumerate(ids):
        Xk = X[id_to_idx[k]]
        mu = Xk.mean(axis=0)
        means[ki] = mu
        within_var[ki] = ((Xk - mu) ** 2).mean(axis=0)  # (D,) -- per-channel, mean over images

    ratio_sum = np.zeros(D, dtype=np.float64)
    count = np.zeros(D, dtype=np.int64)
    for i in range(K):
        diff2 = (means - means[i:i + 1]) ** 2  # (K, D)
        for j in range(i + 1, K):
            num_c = within_var[i] + within_var[j]  # (D,)
            den_c = 2.0 * diff2[j]  # (D,)
            valid = den_c > 1e-12
            ratio_c = np.where(valid, num_c / np.where(valid, den_c, 1.0), np.nan)
            ratio_sum += np.nan_to_num(ratio_c, nan=0.0)
            count += valid.astype(np.int64)

    mean_ratio = np.where(count > 0, ratio_sum / np.maximum(count, 1), np.nan)
    return mean_ratio, K


def main():
    print(f"device: {device}", flush=True)
    market = get_dataset('market1501', sample=False)
    msmt = get_dataset('msmt17', sample=False)
    rng = np.random.default_rng(RNG_SEED)

    print("Loading models...", flush=True)
    trained_market = load_trained(CKPT_MARKET)
    trained_msmt = load_trained(CKPT_MSMT)
    imagenet_init = BoTSpatial_IN().to(device).eval()

    def gallery_for(ds):
        n_q = sum(1 for s in ds.query if not s.is_junk)
        return [s for s in (ds.query + ds.gallery)[n_q:] if not s.is_junk]

    market_gallery = gallery_for(market)
    msmt_gallery = gallery_for(msmt)

    print("\nExtracting: trained-Market on Market-gallery (A' ceiling)...", flush=True)
    feat_A, lbl_A = extract_gap(trained_market, market_gallery)
    print("\nExtracting: trained-Market on MSMT-gallery (B' OOD)...", flush=True)
    feat_B, lbl_B = extract_gap(trained_market, msmt_gallery)
    print("\nExtracting: trained-MSMT on MSMT-gallery (D ceiling)...", flush=True)
    feat_D, lbl_D = extract_gap(trained_msmt, msmt_gallery)
    print("\nExtracting: trained-MSMT on Market-gallery (C OOD)...", flush=True)
    feat_C, lbl_C = extract_gap(trained_msmt, market_gallery)
    print("\nExtracting: ImageNet-init on MSMT-gallery (null for B')...", flush=True)
    feat_null_msmt, lbl_null_msmt = extract_gap(imagenet_init, msmt_gallery)
    print("\nExtracting: ImageNet-init on Market-gallery (null for C)...", flush=True)
    feat_null_market, lbl_null_market = extract_gap(imagenet_init, market_gallery)

    out = {'rev': 671, 'max_ids': MAX_IDS, 'min_imgs': MIN_IMGS, 'per_arm': {}}

    for arm_name, feat_ceiling, lbl_ceiling, feat_ood, lbl_ood, feat_null, lbl_null in [
        ('B_prime', feat_A, lbl_A, feat_B, lbl_B, feat_null_msmt, lbl_null_msmt),
        ('C',       feat_D, lbl_D, feat_C, lbl_C, feat_null_market, lbl_null_market),
    ]:
        print(f"\n=== arm {arm_name} ===", flush=True)
        arm_out = {}
        for stage in STAGES:
            cdnv_ceiling, K_ceil = per_channel_cdnv(feat_ceiling[stage], lbl_ceiling, rng)
            cdnv_ood, K_ood = per_channel_cdnv(feat_ood[stage], lbl_ood, rng)
            cdnv_null, K_null = per_channel_cdnv(feat_null[stage], lbl_null, rng)

            # discriminability = 1/CDNV (higher = more discriminable); survival on
            # discriminability, as specified ("rank better than the untrained null").
            disc_ceiling = 1.0 / np.maximum(cdnv_ceiling, 1e-8)
            disc_ood = 1.0 / np.maximum(cdnv_ood, 1e-8)
            disc_null = 1.0 / np.maximum(cdnv_null, 1e-8)

            denom = disc_ceiling - disc_null
            survival = np.where(np.abs(denom) > 1e-8, (disc_ood - disc_null) / denom, np.nan)
            valid_survival = survival[~np.isnan(survival)]

            # Denominator-quality filter (claude-chat, rev 670 §3): survival is only
            # a meaningful ratio where the ceiling genuinely separates from the null.
            # Threshold set relative to the layer's own ceiling scale (not to the
            # denom distribution itself, to avoid circularity): keep channels where
            # (disc_ceiling - disc_null) exceeds 10% of the median disc_ceiling.
            valid_mask = ~np.isnan(survival)
            median_disc_ceiling = float(np.nanmedian(disc_ceiling[valid_mask])) if valid_mask.any() else float('nan')
            denom_threshold = 0.1 * median_disc_ceiling
            keep_mask = valid_mask & (denom > denom_threshold)
            kept_survival = survival[keep_mask]
            frac_kept = float(keep_mask.sum() / len(survival))

            print(f"  {stage}: K(ceil/ood/null)={K_ceil}/{K_ood}/{K_null}  "
                  f"n_channels_valid={len(valid_survival)}/{len(survival)}  "
                  f"survival: mean={np.nanmean(survival):.3f}  median={np.nanmedian(survival):.3f}  "
                  f"q25={np.nanpercentile(survival,25):.3f}  q75={np.nanpercentile(survival,75):.3f}  "
                  f"frac<0={float((valid_survival<0).mean()):.3f}  frac>1={float((valid_survival>1).mean()):.3f}",
                  flush=True)
            print(f"           FILTERED (denom>0.1*median_ceiling={denom_threshold:.4g}): "
                  f"kept {keep_mask.sum()}/{len(survival)} ({100*frac_kept:.1f}%)  "
                  f"survival: mean={np.nanmean(kept_survival) if len(kept_survival) else float('nan'):.3f}  "
                  f"median={np.nanmedian(kept_survival) if len(kept_survival) else float('nan'):.3f}",
                  flush=True)

            arm_out[stage] = {
                'K_ceiling': K_ceil, 'K_ood': K_ood, 'K_null': K_null,
                'n_channels': int(len(survival)), 'n_channels_valid': int(len(valid_survival)),
                'survival_mean': float(np.nanmean(survival)),
                'survival_median': float(np.nanmedian(survival)),
                'survival_q25': float(np.nanpercentile(survival, 25)),
                'survival_q75': float(np.nanpercentile(survival, 75)),
                'survival_std': float(np.nanstd(survival)),
                'frac_below_0': float((valid_survival < 0).mean()) if len(valid_survival) else float('nan'),
                'frac_above_1': float((valid_survival > 1).mean()) if len(valid_survival) else float('nan'),
                'survival_histogram': {
                    f'{lo:.1f}_{lo+0.5:.1f}': int(((valid_survival >= lo) & (valid_survival < lo + 0.5)).sum())
                    for lo in np.arange(-2.0, 3.0, 0.5)
                },
                'survival_values': survival.tolist(),
                'disc_null': disc_null.tolist(),
                'disc_ceiling': disc_ceiling.tolist(),
                'disc_ood': disc_ood.tolist(),
                'denom_threshold': denom_threshold,
                'frac_kept_filtered': frac_kept,
                'n_kept_filtered': int(keep_mask.sum()),
                'survival_mean_filtered': float(np.nanmean(kept_survival)) if len(kept_survival) else float('nan'),
                'survival_median_filtered': float(np.nanmedian(kept_survival)) if len(kept_survival) else float('nan'),
                'survival_q25_filtered': float(np.nanpercentile(kept_survival, 25)) if len(kept_survival) else float('nan'),
                'survival_q75_filtered': float(np.nanpercentile(kept_survival, 75)) if len(kept_survival) else float('nan'),
                'frac_below_0_filtered': float((kept_survival < 0).mean()) if len(kept_survival) else float('nan'),
                'frac_above_1_filtered': float((kept_survival > 1).mean()) if len(kept_survival) else float('nan'),
            }
        out['per_arm'][arm_name] = arm_out

    outp = RESULTS / 'e44_per_channel_survival_filtered.json'
    outp.write_text(json.dumps(out, indent=2))
    print(f"\nsaved {outp}", flush=True)


if __name__ == '__main__':
    main()
