"""
Build no-REA Market feature caches in e12 format for cells A' and B'.

Uses the same data pipeline as compute_bot_msmt_2x2.py:
  reid_tk.datasets.get_dataset(..., sample=False)
  market.query + market.gallery  (is_junk filtered in DataLoader)

Checkpoint: /data/papers/low_dimensional_reid/pretrained/rea_ablation/bot_resnet50_market1501_seed42_final.pth
"""
import sys; sys.path.insert(0, '/workspace/reid-tk')
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import torchvision.models as models, torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
from pathlib import Path

from reid_tk.datasets import get_dataset

CACHE_DIR = Path('/tmp/e12_feature_cache')
CKPT = Path('/data/papers/low_dimensional_reid/pretrained/rea_ablation/bot_resnet50_market1501_seed42_final.pth')
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def build_bot():
    resnet = models.resnet50(weights=None)
    resnet.layer4[0].conv2.stride = (1, 1)
    resnet.layer4[0].downsample[0].stride = (1, 1)
    base = nn.Sequential(*list(resnet.children())[:-2])
    bn = nn.BatchNorm1d(2048); bn.bias.requires_grad_(False)
    return nn.Sequential(base, nn.AdaptiveAvgPool2d(1), nn.Flatten(), bn)


def load_noREA_ckpt(model):
    sd_raw = torch.load(CKPT, map_location=device, weights_only=False)
    # Drop any 'num_batches_tracked', keep 0.* and 3.* (matches Sequential structure)
    sd = {}
    for k, v in sd_raw.items():
        if 'num_batches_tracked' in k: continue
        # The old flat format uses "0." for base and "3." for bottleneck (bn).
        # Our Sequential is: 0: base, 1: AvgPool, 2: Flatten, 3: BN.
        # Keys 0.X and 3.X should map 1-to-1. Classifier keys (if any) are ignored.
        if k.startswith('classifier'): continue
        sd[k] = v
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"  missing keys: {len(missing)}  unexpected: {len(unexpected)}")
    if missing: print(f"    first missing: {missing[:3]}")
    if unexpected: print(f"    first unexpected: {unexpected[:3]}")
    return model


TRANSFORM = T.Compose([T.Resize((256, 128)), T.ToTensor(),
                       T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])


class DS(Dataset):
    def __init__(self, samples):
        self.samples = [s for s in samples if not s.is_junk]
    def __len__(self): return len(self.samples)
    def __getitem__(self, i):
        s = self.samples[i]
        return TRANSFORM(s.load_image()), s.person_id, s.camera_id


@torch.no_grad()
def extract(model, samples, out_path, batch=128):
    n = len([s for s in samples if not s.is_junk])
    print(f"  extracting {n} samples → {out_path.name}")
    ds = DS(samples); ldr = DataLoader(ds, batch_size=batch, num_workers=0, pin_memory=False)
    embs, pids, cids = [], [], []
    for imgs, ps, cs in ldr:
        f = model(imgs.to(device))
        embs.append(F.normalize(f, p=2, dim=1).cpu().numpy())
        pids.extend(ps.numpy()); cids.extend(cs.numpy())
    emb = np.vstack(embs)
    np.savez(out_path, emb=emb, pids=np.array(pids), cids=np.array(cids))
    print(f"    saved {emb.shape}")


def main():
    print("Building no-REA BoT-Market model...")
    model = build_bot().to(device).eval()
    load_noREA_ckpt(model)

    print("\nLoading datasets...")
    market = get_dataset("market1501", sample=False)
    msmt   = get_dataset("msmt17",     sample=False)
    market_samples = market.query + market.gallery
    msmt_samples   = msmt.query   + msmt.gallery

    out_a = CACHE_DIR / 'market_test_noREA.npz'
    out_b = CACHE_DIR / 'msmt17_test_noREA.npz'

    if not out_a.exists():
        extract(model, market_samples, out_a)
    else:
        print(f"  already exists: {out_a.name}")

    if not out_b.exists():
        extract(model, msmt_samples, out_b)
    else:
        print(f"  already exists: {out_b.name}")

    # Verify
    for p in [out_a, out_b]:
        d = np.load(p)
        print(f"  {p.name}: emb={d['emb'].shape} pids={d['pids'].shape}")

    # Sanity-check: compare pid/cid ordering with existing with-REA caches
    for name, orig, new in [('Market', CACHE_DIR/'market_test.npz', out_a),
                            ('MSMT',   CACHE_DIR/'msmt17_test.npz', out_b)]:
        a = np.load(orig); b = np.load(new)
        same_order = np.array_equal(a['pids'], b['pids']) and np.array_equal(a['cids'], b['cids'])
        print(f"  {name}: pid/cid ordering matches existing cache = {same_order}")


if __name__ == '__main__':
    main()
