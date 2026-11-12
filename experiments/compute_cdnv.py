"""
CDNV + VCI per stage, per cell, by gallery (rev 502 §F #1, #2).

Per (arm, stage):
  1. Extract GAP features of OOD gallery images.
  2. Group by identity. Keep identities with >= 2 images.
  3. Compute per-identity within-variance (trace of covariance) and mean.
  4. CDNV: V(Q1,Q2) = [Var(Q1) + Var(Q2)] / (2 ||mu(Q1) - mu(Q2)||^2) for all pairs.
     Report mean numerator, mean denominator, mean ratio separately per rev 500 §2.
  5. VCI = 1 - Tr[Sigma_T_dagger Sigma_B] / rank(Sigma_B).

Expected: foreign cells show lower denominator (between-identity separation collapse)
with roughly stable numerator, per Daniel's v9 restatement.
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
MAX_IDS = 200       # cap identities per arm
MIN_IMGS = 2        # min images per identity

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


def cdnv(X, y, rng, max_ids=200):
    """
    X: (N, D). y: (N,).
    Returns: mean_num (avg within-identity trace-cov), mean_den (avg between-mean-dist^2),
             mean_ratio (CDNV), and VCI.
    """
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
        return dict(num=float('nan'), den=float('nan'), ratio=float('nan'),
                    vci=float('nan'), K=K)
    num_sum = 0.0; den_sum = 0.0; ratio_sum = 0.0; count = 0
    for i in range(K):
        diff = mu_arr - mu_arr[i:i + 1]
        d2 = (diff * diff).sum(axis=1)
        for j in range(i + 1, K):
            num = wv_arr[i] + wv_arr[j]
            den = 2.0 * d2[j]
            if den > 1e-12:
                num_sum += num; den_sum += den; ratio_sum += num / den; count += 1
    mean_num = num_sum / max(count, 1)
    mean_den = den_sum / max(count, 1)
    mean_ratio = ratio_sum / max(count, 1)
    grand_mean = X.mean(axis=0)
    Xc = X - grand_mean
    Sigma_T = (Xc.T @ Xc) / max(len(X) - 1, 1)
    counts = np.array([len(id_to_idx[k]) for k in ids])
    B_rows = np.array([means[k] - grand_mean for k in ids])
    Sigma_B = (B_rows.T @ (B_rows * counts[:, None])) / max(counts.sum() - 1, 1)
    try:
        Sigma_T_pinv = np.linalg.pinv(Sigma_T, rcond=1e-8)
        tr = float(np.trace(Sigma_T_pinv @ Sigma_B))
        rank_B = int(np.linalg.matrix_rank(Sigma_B, tol=1e-6))
        vci = 1.0 - tr / max(rank_B, 1)
    except Exception:
        vci = float('nan'); rank_B = -1
    return dict(num=mean_num, den=mean_den, ratio=mean_ratio, vci=vci,
                K=K, rank_B=rank_B)


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
            r = cdnv(feats[stage], labels, rng, max_ids=MAX_IDS)
            arm_out[stage] = r
            print(f"  {stage:<8s}: K={r['K']:3d}  num={r['num']:.4f}  den={r['den']:.4f}  "
                  f"CDNV={r['ratio']:.4f}  VCI={r['vci']:.4f}", flush=True)
        all_out[arm] = arm_out

    with open(RESULTS / 'e44_cdnv_vci.json', 'w') as f:
        json.dump(all_out, f, indent=2)
    print(f"\nSaved {RESULTS / 'e44_cdnv_vci.json'}")


if __name__ == '__main__':
    main()
