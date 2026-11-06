#!/usr/bin/env python3
"""
E8a — IBN-ResNet50 Market→MSMT17 transfer mAP (persisting the channel-only result).

Re-evaluates bot_ibn_resnet50_market1501_final.pth using the same protocol as
the plain-BoT entries in e8_results.json: Market test mAP (in-domain) and
MSMT17 test mAP (cross-domain zero-shot).
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, '/workspace/reid-tk')

CKPT     = Path("/data/papers/low_dimensional_reid/pretrained/bot_ibn_resnet50_market1501_final.pth")
OUT_PATH = Path("/data/papers/low_dimensional_reid/results/e8a_ibn_results.json")
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"

TF = T.Compose([
    T.Resize((256, 128)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class _DS(Dataset):
    def __init__(self, samples):
        self.s = [s for s in samples if not getattr(s, 'is_junk', False)]
    def __len__(self): return len(self.s)
    def __getitem__(self, i):
        s = self.s[i]
        return TF(s.load_image()), s.person_id, s.camera_id


def build_ibn_extractor(ckpt_path, device):
    import torchreid
    full = torchreid.models.build_model(name='resnet50_ibn_a', num_classes=1, pretrained=False)
    full.layer4[0].conv2.stride = (1, 1)
    full.layer4[0].downsample[0].stride = (1, 1)
    base = nn.Sequential(
        full.conv1, full.bn1, full.relu, full.maxpool,
        full.layer1, full.layer2, full.layer3, full.layer4,
    )
    bn = nn.BatchNorm1d(2048)
    bn.bias.requires_grad_(False)
    extractor = nn.Sequential(base, nn.AdaptiveAvgPool2d(1), nn.Flatten(), bn)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    extractor.load_state_dict(state)
    return extractor.to(device).eval()


@torch.no_grad()
def extract(model, samples, device):
    loader = DataLoader(_DS(samples), batch_size=256, num_workers=0)
    embs, pids, cids = [], [], []
    for imgs, ids, cs in loader:
        e = F.normalize(model(imgs.to(device)), dim=1)
        embs.append(e.cpu().numpy())
        pids.extend(ids.tolist())
        cids.extend(cs.tolist())
    return np.vstack(embs), np.array(pids), np.array(cids)


def retrieval_map(q_emb, g_emb, q_ids, g_ids, q_cams, g_cams):
    dist = -(q_emb @ g_emb.T)
    aps, r1_cnt = [], 0
    for qi in range(len(q_ids)):
        qid, qcam = q_ids[qi], q_cams[qi]
        order = np.argsort(dist[qi])
        junk = (g_ids == -1) | ((g_ids == qid) & (g_cams == qcam))
        order = order[~junk[order]]
        matches = g_ids[order] == qid
        if not matches.any():
            continue
        if matches[0]:
            r1_cnt += 1
        pos = np.where(matches)[0]
        aps.append(float(np.mean([(i + 1) / (p + 1) for i, p in enumerate(pos)])))
    return float(np.mean(aps)) if aps else 0.0, r1_cnt / max(len(q_ids), 1)


def main():
    print(f"Device: {DEVICE}")
    from reid_tk.datasets import get_dataset
    ds_mkt  = get_dataset('market1501', root=Path('/data/datasets/reid/Market-1501-v15.09.15'))
    ds_msmt = get_dataset('msmt17',     root=Path('/data/datasets/reid/MSMT17_V1'))

    print("Building IBN-ResNet50 extractor...")
    model = build_ibn_extractor(CKPT, DEVICE)

    print("Extracting Market test features...")
    q_emb, q_ids, q_cams = extract(model, ds_mkt.query,   DEVICE)
    g_emb, g_ids, g_cams = extract(model, ds_mkt.gallery, DEVICE)
    mkt_map, mkt_r1 = retrieval_map(q_emb, g_emb, q_ids, g_ids, q_cams, g_cams)
    print(f"  Market  mAP={mkt_map*100:.2f}%  R1={mkt_r1*100:.2f}%")

    print("Extracting MSMT17 test features...")
    qm_emb, qm_ids, qm_cams = extract(model, ds_msmt.query,   DEVICE)
    gm_emb, gm_ids, gm_cams = extract(model, ds_msmt.gallery, DEVICE)
    msmt_map, msmt_r1 = retrieval_map(qm_emb, gm_emb, qm_ids, gm_ids, qm_cams, gm_cams)
    print(f"  MSMT17  mAP={msmt_map*100:.2f}%  R1={msmt_r1*100:.2f}%")

    result = {
        "model": "IBN-ResNet50 (BoT, Market-trained)",
        "checkpoint": str(CKPT),
        "protocol": "E8 protocol — Market test query/gallery; MSMT17 test query/gallery",
        "market_map": round(mkt_map, 6),
        "market_r1":  round(mkt_r1, 6),
        "msmt17_map": round(msmt_map, 6),
        "msmt17_r1":  round(msmt_r1, 6),
        "transfer_gap": round(mkt_map - msmt_map, 6),
        "ratio_r1_over_map_msmt17": round(msmt_r1 / msmt_map, 3) if msmt_map > 0 else None,
        "note": ("Re-evaluation to persist Channel-only E8a result. "
                 "2.56% (plain BoT bot_final) and 6.63% (IBN) both use this protocol.")
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved → {OUT_PATH}")


if __name__ == "__main__":
    main()
