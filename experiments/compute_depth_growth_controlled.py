"""
§1 magnitude confound control for §2 depth-growth (rev 499).

For each (arm, stage):
  - Report ||x|| and ||x - mu_train|| for own and foreign (per-cell patches).
  - Recompute OOS rate with RELATIVE residual: ||(x-mu) - P(x-mu)|| / ||x-mu||, thresholded.
  - Compute participation ratio of the foreign feature distribution per stage (effective dim).

Decides whether B'/C mirror-opposite pattern survives magnitude correction.
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
sys.path.insert(0, '/workspace/reid-tk')
from reid_tk.datasets import get_dataset

RESULTS = Path('/data/papers/low_dimensional_reid/results')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
RNG_SEED = 42
SUBSAMPLE = 400
PCA_K = 50
PATCH_FIT_N = 20000

CKPT_MARKET = Path('/data/papers/low_dimensional_reid/pretrained/rea_ablation/bot_resnet50_market1501_seed42_final.pth')
CKPT_MSMT   = Path('/data/papers/low_dimensional_reid/pretrained/bot_resnet50_msmt17_seed42_final.pth')

TRANSFORM = T.Compose([T.Resize((256, 128)), T.ToTensor(),
                       T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

ARMS = {
    'A_prime': {'ckpt': 'market', 'train_set': 'market', 'test_set': 'market', 'dataset_test': 'market1501'},
    'B_prime': {'ckpt': 'market', 'train_set': 'market', 'test_set': 'msmt',   'dataset_test': 'msmt17'},
    'C':       {'ckpt': 'msmt',   'train_set': 'msmt',   'test_set': 'market', 'dataset_test': 'market1501'},
    'D':       {'ckpt': 'msmt',   'train_set': 'msmt',   'test_set': 'msmt',   'dataset_test': 'msmt17'},
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
        return {'stem': stem, 'layer1': l1, 'layer2': l2, 'layer3': l3, 'layer4': l4}


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
def extract_spatial(model, samples, batch=32):
    ds = DS(samples)
    print(f"    extracting {len(ds)} items...", flush=True)
    ldr = DataLoader(ds, batch_size=batch, num_workers=0, pin_memory=False)
    stages = {s: [] for s in STAGES}
    for imgs, _ in ldr:
        out = model(imgs.to(device, non_blocking=True))
        for s in STAGES:
            stages[s].append(out[s].cpu().numpy())
    return {s: np.vstack(v) for s, v in stages.items()}


def extract_patches(feat_map, pad=1):
    N, C, H, W = feat_map.shape
    padded = np.zeros((N, C, H + 2 * pad, W + 2 * pad), dtype=np.float32)
    padded[:, :, pad:pad + H, pad:pad + W] = feat_map
    patches = np.zeros((N, 9 * C, H, W), dtype=np.float32)
    for dy in range(3):
        for dx in range(3):
            patches[:, (dy * 3 + dx) * C:(dy * 3 + dx + 1) * C, :, :] = padded[:, :, dy:dy + H, dx:dx + W]
    return patches.transpose(0, 2, 3, 1).reshape(N * H * W, 9 * C), H, W


def pca_ops(train_patches, probe_patches, k):
    """Return: train residuals, probe residuals, train centred norms, probe centred norms."""
    mu = train_patches.mean(axis=0)
    train_c = train_patches - mu
    probe_c = probe_patches - mu
    _, _, Vt = np.linalg.svd(train_c, full_matrices=False)
    Vk = Vt[:k]
    tr_res = np.linalg.norm(train_c - (train_c @ Vk.T) @ Vk, axis=1)
    pr_res = np.linalg.norm(probe_c - (probe_c @ Vk.T) @ Vk, axis=1)
    tr_cn = np.linalg.norm(train_c, axis=1)
    pr_cn = np.linalg.norm(probe_c, axis=1)
    return tr_res, pr_res, tr_cn, pr_cn, mu


def participation_ratio(X):
    """PR = (sum lambda_i)^2 / sum lambda_i^2 where lambda_i are eigenvalues of cov(X)."""
    Xc = X - X.mean(axis=0)
    # SVD gives sqrt of eigenvalues (up to scale)
    N, D = Xc.shape
    if N > D:
        cov = (Xc.T @ Xc) / max(N - 1, 1)
        lam = np.linalg.eigvalsh(cov)
    else:
        gram = (Xc @ Xc.T) / max(N - 1, 1)
        lam = np.linalg.eigvalsh(gram)
    lam = np.clip(lam, 0, None)
    return float(lam.sum() ** 2 / max((lam ** 2).sum(), 1e-12))


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
        train_ds = market.train if info['train_set'] == 'market' else msmt.train
        test_ds = market if info['test_set'] == 'market' else msmt
        n_q = sum(1 for s in test_ds.query if not s.is_junk)
        test_g = [s for s in (test_ds.query + test_ds.gallery)[n_q:] if not s.is_junk]
        tr_sub = [train_ds[i] for i in rng.choice(len(train_ds), size=min(SUBSAMPLE, len(train_ds)), replace=False)]
        g_sub = [test_g[i] for i in rng.choice(len(test_g), size=min(SUBSAMPLE, len(test_g)), replace=False)]
        tr_feats = extract_spatial(model, tr_sub)
        g_feats = extract_spatial(model, g_sub)

        arm_out = {}
        for stage in STAGES:
            X_tr = tr_feats[stage]; X_g = g_feats[stage]
            tr_patches, H, W = extract_patches(X_tr)
            g_patches, _, _ = extract_patches(X_g)
            fit_idx = rng.choice(len(tr_patches), size=min(PATCH_FIT_N, len(tr_patches)), replace=False)
            tr_fit = tr_patches[fit_idx]
            k = min(PCA_K, tr_fit.shape[1] // 2)
            tr_res, g_res, tr_cn, g_cn, mu = pca_ops(tr_fit, g_patches, k)
            # Absolute threshold from tr_res p95
            thr_abs = float(np.quantile(tr_res, 0.95))
            oos_abs = float((g_res > thr_abs).mean())
            # Relative residual
            tr_rel = tr_res / np.maximum(tr_cn, 1e-12)
            g_rel = g_res / np.maximum(g_cn, 1e-12)
            thr_rel = float(np.quantile(tr_rel, 0.95))
            oos_rel = float((g_rel > thr_rel).mean())
            # Participation ratio (on patches of this stage; subsample for cost)
            pr_tr = participation_ratio(tr_fit)
            pr_g = participation_ratio(g_patches[rng.choice(len(g_patches), size=min(PATCH_FIT_N, len(g_patches)), replace=False)])

            arm_out[stage] = {
                'HxW': [int(H), int(W)],
                'tr_patch_norm_mean': float(tr_cn.mean()),
                'g_patch_norm_mean': float(g_cn.mean()),
                'tr_patch_norm_ratio_g_over_tr': float(g_cn.mean() / max(tr_cn.mean(), 1e-12)),
                'oos_rate_absolute': oos_abs,
                'oos_rate_relative': oos_rel,
                'participation_ratio_train': pr_tr,
                'participation_ratio_ood': pr_g,
                'pr_ratio_ood_over_train': pr_g / max(pr_tr, 1e-12),
            }
            print(f"  {stage:<8s}: ||x-mu||_tr={tr_cn.mean():.3f}  ||x-mu||_g={g_cn.mean():.3f}  g/tr={g_cn.mean() / max(tr_cn.mean(), 1e-12):.3f}  "
                  f"OOS_abs={oos_abs:.4f}  OOS_rel={oos_rel:.4f}  PR_tr={pr_tr:.2f}  PR_g={pr_g:.2f}  PR_g/PR_tr={pr_g / max(pr_tr, 1e-12):.3f}", flush=True)
        all_out[arm] = arm_out

    with open(RESULTS / 'e44_depth_growth_controlled.json', 'w') as f:
        json.dump(all_out, f, indent=2)
    print(f"\nSaved {RESULTS / 'e44_depth_growth_controlled.json'}")


if __name__ == '__main__':
    main()
