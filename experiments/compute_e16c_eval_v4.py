#!/usr/bin/env python3
"""
E16c eval harness — option (a): split over MSMT17 test IDs, BoT-MSMT17 model.

Rungs (from e16c_split_v4.json):
  R1: in-split IDs × seen cameras    (in-domain reference for BoT-MSMT17)
  R2: held-out IDs × seen cameras    (open-set: novel IDs, same cam subset)
  R3: held-out IDs × held cameras    (camera-restricted: novel IDs, held cam subset)
  R4: Market-1501                     (cross-dataset)

NOTE: BoT-MSMT17 saw all 15 cameras in training. R2→R3 tests camera-composition
(gallery restricted to a cam subset), NOT genuinely unseen cameras.
Do NOT interpret R2→R3 as an unseen-camera effect.

Image-count matching: each rung subsampled to R3's gallery image count
(the bottleneck). This was NOT done in the (c) run and was flagged as a caveat.

Query protocol: one random image per (ID, camera) pair as query; all images
in rung as gallery; same-camera gallery excluded from relevant set (standard mAP).
"""

import json
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from collections import defaultdict
import sys

sys.path.insert(0, '/workspace/reid-tk')

CKPT         = Path("/data/papers/low_dimensional_reid/pretrained/bot_resnet50_msmt17_seed42_final.pth")
SPLIT_V4     = Path("/data/papers/low_dimensional_reid/results/e16c_split_v4.json")
OUT_PATH     = Path("/data/papers/low_dimensional_reid/results/e16c_eval_results_v4.json")
RNG_SEED     = 42
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

TRANSFORM = T.Compose([
    T.Resize((256, 128)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class _DS(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform
    def __len__(self): return len(self.samples)
    def __getitem__(self, i):
        path, pid, cid = self.samples[i]
        img = Image.open(path).convert('RGB')
        return self.transform(img), pid, cid


def load_bot(ckpt, device):
    resnet = models.resnet50(weights=None)
    resnet.layer4[0].conv2.stride = (1, 1)
    resnet.layer4[0].downsample[0].stride = (1, 1)
    base = nn.Sequential(*list(resnet.children())[:-2])
    bn = nn.BatchNorm1d(2048)
    bn.bias.requires_grad_(False)
    model = nn.Sequential(base, nn.AdaptiveAvgPool2d(1), nn.Flatten(), bn)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=False))
    return model.to(device).eval()


@torch.no_grad()
def extract_features(model, samples, device, batch_size=256):
    ds = _DS(samples, TRANSFORM)
    loader = DataLoader(ds, batch_size=batch_size, num_workers=0, pin_memory=False)
    embs, pids, cids = [], [], []
    for imgs, p, c in loader:
        out = model(imgs.to(device))
        out = out / out.norm(dim=1, keepdim=True)
        embs.append(out.cpu().numpy())
        pids.extend(p.tolist())
        cids.extend(c.tolist())
    return np.concatenate(embs, 0).astype(np.float32), np.array(pids), np.array(cids)


def compute_map(q_emb, q_pids, q_cids, g_emb, g_pids, g_cids):
    sims = q_emb @ g_emb.T
    aps = []
    for i in range(len(q_pids)):
        good = (g_pids == q_pids[i]) & (g_cids != q_cids[i])
        junk = (g_pids == q_pids[i]) & (g_cids == q_cids[i])
        if good.sum() == 0:
            continue
        order = np.argsort(-sims[i])
        good_s = good[order]
        junk_s = junk[order]
        valid_mask = ~junk_s
        good_v = good_s[valid_mask]
        hits = np.where(good_v)[0] + 1
        aps.append(float((np.arange(1, good_v.sum() + 1) / hits).mean()))
    return float(np.mean(aps)) if aps else 0.0, len(aps)


def eval_rung(emb_all, pids_all, cids_all, rung_pids_set, rung_cids_set,
              rng, n_imgs_target=None):
    """Filter to rung, optionally subsample to n_imgs_target images, eval mAP."""
    mask = np.array([
        (p in rung_pids_set) and (c in rung_cids_set)
        for p, c in zip(pids_all.tolist(), cids_all.tolist())
    ])
    emb = emb_all[mask]
    pids = pids_all[mask]
    cids = cids_all[mask]
    n_imgs_before = int(mask.sum())

    # Subsample to image-count target if specified
    if n_imgs_target is not None and len(emb) > n_imgs_target:
        idx = rng.choice(len(emb), n_imgs_target, replace=False)
        idx.sort()
        emb = emb[idx]; pids = pids[idx]; cids = cids[idx]

    # One query per (pid, cid) pair
    pc_groups = defaultdict(list)
    for i, (p, c) in enumerate(zip(pids.tolist(), cids.tolist())):
        pc_groups[(p, c)].append(i)

    q_idxs = np.array([rng.choice(v) for v in pc_groups.values()])
    q_emb  = emb[q_idxs]
    q_pids = pids[q_idxs]
    q_cids = cids[q_idxs]

    map_val, n_q = compute_map(q_emb, q_pids, q_cids, emb, pids, cids)
    return {
        "n_ids": len(set(pids.tolist())),
        "n_imgs_before_match": n_imgs_before,
        "n_imgs": len(emb),
        "n_queries": n_q,
        "map": round(map_val, 6)
    }


def eval_market_rung(mkt_emb, mkt_pids, mkt_cids, n_imgs_target, rng):
    """Subsample Market to n_imgs_target images and evaluate mAP."""
    if len(mkt_emb) > n_imgs_target:
        idx = rng.choice(len(mkt_emb), n_imgs_target, replace=False)
        idx.sort()
        emb   = mkt_emb[idx]; pids = mkt_pids[idx]; cids = mkt_cids[idx]
    else:
        emb, pids, cids = mkt_emb, mkt_pids, mkt_cids

    pc_groups = defaultdict(list)
    for i, (p, c) in enumerate(zip(pids.tolist(), cids.tolist())):
        pc_groups[(p, c)].append(i)

    q_idxs = np.array([rng.choice(v) for v in pc_groups.values()])
    q_emb  = emb[q_idxs]
    q_pids = pids[q_idxs]
    q_cids = cids[q_idxs]

    map_val, n_q = compute_map(q_emb, q_pids, q_cids, emb, pids, cids)
    return {
        "n_ids": len(np.unique(pids)),
        "n_imgs_before_match": len(mkt_emb),
        "n_imgs": len(emb),
        "n_queries": n_q,
        "map": round(map_val, 6)
    }


def main():
    print(f"Device: {DEVICE}")

    split = json.load(open(SPLIT_V4))
    in_split_ids  = set(split['id_split']['in_split_ids'])
    held_out_ids  = set(split['id_split']['held_out_ids'])
    draws         = split['balanced_draws']['draws']

    print(f"Split: {len(in_split_ids)} in-split, {len(held_out_ids)} held-out MSMT17 test IDs")

    print("Loading BoT-MSMT17 model...")
    model = load_bot(CKPT, DEVICE)

    print("Loading MSMT17 gallery (test images)...")
    from reid_tk.datasets import get_dataset
    ds = get_dataset('msmt17', root=Path('/data/datasets/reid/MSMT17_V1'))
    gallery_samples = [(str(s.image_path), s.person_id, s.camera_id)
                       for s in ds.gallery if not s.is_junk]
    print(f"  {len(gallery_samples)} gallery images, "
          f"{len(set(p for _,p,_ in gallery_samples))} IDs, "
          f"{len(set(c for _,_,c in gallery_samples))} cams")

    print("Extracting MSMT17 gallery features (one pass)...")
    gallery_emb, gallery_pids, gallery_cids = extract_features(model, gallery_samples, DEVICE)
    print(f"  Done: {gallery_emb.shape}")

    print("Extracting Market test features with BoT-MSMT17 (R4)...")
    ds_mkt = get_dataset('market1501', root=Path('/data/datasets/reid/Market-1501-v15.09.15'))
    mkt_samples = [(str(s.image_path), s.person_id, s.camera_id)
                   for s in ds_mkt.query + ds_mkt.gallery
                   if not s.is_junk]
    mkt_emb, mkt_pids, mkt_cids = extract_features(model, mkt_samples, DEVICE)
    print(f"  Market: {len(mkt_pids)} imgs, {len(np.unique(mkt_pids))} IDs (extracted with BoT-MSMT17)")

    rng = np.random.default_rng(RNG_SEED)
    draw_results = []

    cam_ids = sorted(set(gallery_cids.tolist()))

    for draw in draws:
        draw_idx  = draw['draw']
        held_cams = set(draw['held_cameras'])
        seen_cams = set(cam_ids) - held_cams
        r3_n_imgs = draw['n_imgs_r3']  # image-count bottleneck
        r3_n_ids  = draw['n_ids_r3']
        print(f"\n--- Draw {draw_idx}: hold={sorted(held_cams)}  "
              f"R3 target={r3_n_ids}IDs/{r3_n_imgs}imgs ---")

        dr = {"draw": draw_idx, "held_cameras": sorted(held_cams),
              "r3_bottleneck_imgs": r3_n_imgs, "r3_bottleneck_ids": r3_n_ids}

        for tag, pid_set, cam_set in [
            ("r1",  in_split_ids,  seen_cams),
            ("r2",  held_out_ids,  seen_cams),
            ("r3",  held_out_ids,  held_cams),
        ]:
            # R3 is the reference (not subsampled); R1, R2 subsampled to R3 image count
            target = None if tag == "r3" else r3_n_imgs
            res = eval_rung(gallery_emb, gallery_pids, gallery_cids,
                            pid_set, cam_set, rng, n_imgs_target=target)
            dr[tag] = res
            print(f"  {tag:4s}: {res['n_ids']:5d}IDs/{res['n_imgs']:6d}imgs  "
                  f"mAP={res['map']:.4f}")

        # R4: Market subsampled to R3 image count
        r4_res = eval_market_rung(mkt_emb, mkt_pids, mkt_cids, r3_n_imgs, rng)
        dr["r4_market"] = r4_res
        print(f"  r4:   {r4_res['n_ids']:5d}IDs/{r4_res['n_imgs']:6d}imgs  "
              f"mAP={r4_res['map']:.4f}")

        draw_results.append(dr)

    print("\n=== Summary ===")
    print(f"{'Draw':>5} {'R1':>8} {'R2':>8} {'R3':>8} {'R4_Mkt':>8}")
    for dr in draw_results:
        print(f"{dr['draw']:>5} "
              f"{dr['r1']['map']:>8.4f} "
              f"{dr['r2']['map']:>8.4f} "
              f"{dr['r3']['map']:>8.4f} "
              f"{dr['r4_market']['map']:>8.4f}")

    avg = lambda key: np.mean([dr[key]['map'] for dr in draw_results])
    print(f"\nMeans: R1={avg('r1'):.4f}  R2={avg('r2'):.4f}  "
          f"R3={avg('r3'):.4f}  R4={avg('r4_market'):.4f}")

    out = {
        "model": "BoT-MSMT17 ResNet-50 seed42",
        "checkpoint": str(CKPT),
        "split_file": str(SPLIT_V4),
        "option": "(a): split over MSMT17 test IDs (disjoint from BoT-MSMT17 training)",
        "matching": "R1/R2/R4 subsampled to R3 GALLERY IMAGE COUNT (bottleneck)",
        "protocol": ("one query per (ID,cam) pair; same-cam gallery excluded; "
                     "MSMT17 gallery images for R1-R3; Market-1501 test for R4"),
        "ladder_semantics": split["ladder_semantics"],
        "draw_results": draw_results,
        "means": {
            "r1": round(float(avg('r1')), 6),
            "r2": round(float(avg('r2')), 6),
            "r3": round(float(avg('r3')), 6),
            "r4_market": round(float(avg('r4_market')), 6),
        },
        "caveats": [
            ("R2→R3 tests camera composition (gallery restricted), NOT genuinely unseen cameras. "
             "BoT-MSMT17 saw all 15 cameras in training."),
            "Image-count matching applied (unlike the (c) run with ID-count matching only).",
        ]
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved → {OUT_PATH}")


if __name__ == "__main__":
    main()
