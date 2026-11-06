#!/usr/bin/env python3
"""
E16c eval harness — within-dataset shift ladder.

Evaluates BoT-Market (ResNet-50, Market-trained) across four rungs per draw,
using e16c_split_v3.json as the split input:

  R1: in-split IDs × seen cameras    (memorization ceiling)
  R2: held-out IDs × seen cameras    (open-set cost alone)
  R3: held-out IDs × held cameras    (camera + ID shift, within MSMT17)
  R4: Market-1501                     (full cross-dataset, subsampled to R3 IDs)

Within-draw matching: R1, R2, R4 subsampled to R3 bottleneck ID count.
R1 reported both unmatched and matched.

Query/gallery protocol (train images): one random image per (ID, camera) pair
as query; all images in rung as gallery; same-camera gallery entries excluded
from the relevant set (standard CMC/mAP protocol).
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

CKPT         = Path("/data/papers/low_dimensional_reid/pretrained/bot_market1501_final.pth")
SPLIT_V3     = Path("/data/papers/low_dimensional_reid/results/e16c_split_v3.json")
MARKET_CACHE = Path("/tmp/e12_feature_cache/market_test.npz")
OUT_PATH     = Path("/data/papers/low_dimensional_reid/results/e16c_eval_results.json")
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
    """Standard ReID mAP: exclude same-cam same-ID; junk = same-cam same-ID or pid<=0."""
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
    return (float(np.mean(aps)) if aps else 0.0), len(aps)


def eval_rung(emb_all, pids_all, cids_all, rung_pids_set, rung_cids_set, rng):
    """Evaluate one rung: filter to rung, pick one query per (pid,cid), eval mAP."""
    mask = np.array([
        (p in rung_pids_set) and (c in rung_cids_set)
        for p, c in zip(pids_all.tolist(), cids_all.tolist())
    ])
    emb = emb_all[mask]
    pids = pids_all[mask]
    cids = cids_all[mask]

    # One query per (pid, cid) pair
    pc_groups = defaultdict(list)
    for i, (p, c) in enumerate(zip(pids.tolist(), cids.tolist())):
        pc_groups[(p, c)].append(i)

    q_idxs = np.array([rng.choice(v) for v in pc_groups.values()])
    q_emb  = emb[q_idxs]
    q_pids = pids[q_idxs]
    q_cids = cids[q_idxs]

    map_val, n_q = compute_map(q_emb, q_pids, q_cids, emb, pids, cids)
    n_ids = len(set(pids.tolist()))
    n_imgs = int(mask.sum())
    return {"n_ids": n_ids, "n_imgs": n_imgs, "n_queries": n_q, "map": round(map_val, 6)}


def subsample_pids(all_pids, n_keep, rng):
    """Return a randomly chosen subset of n_keep unique pids."""
    unique = sorted(all_pids)
    if len(unique) <= n_keep:
        return set(unique)
    chosen = rng.choice(unique, n_keep, replace=False).tolist()
    return set(chosen)


def eval_market_rung(mkt_emb, mkt_pids, mkt_cids, n_ids, rng):
    """Subsample Market to n_ids and evaluate mAP."""
    unique = np.unique(mkt_pids)
    if len(unique) > n_ids:
        chosen = set(rng.choice(unique, n_ids, replace=False).tolist())
        mask = np.array([p in chosen for p in mkt_pids.tolist()])
        emb   = mkt_emb[mask]
        pids  = mkt_pids[mask]
        cids  = mkt_cids[mask]
    else:
        emb, pids, cids = mkt_emb, mkt_pids, mkt_cids

    # One query per (pid, cid)
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
        "n_imgs": len(pids),
        "n_queries": n_q,
        "map": round(map_val, 6)
    }


def main():
    print(f"Device: {DEVICE}")

    split = json.load(open(SPLIT_V3))
    in_split_ids  = set(split['id_split']['in_split_ids'])
    held_out_ids  = set(split['id_split']['held_out_ids'])
    draws         = split['balanced_draws']['draws']

    print("Loading BoT-Market model...")
    model = load_bot(CKPT, DEVICE)

    print("Loading MSMT17 train set...")
    from reid_tk.datasets import get_dataset
    ds = get_dataset('msmt17', root=Path('/data/datasets/reid/MSMT17_V1'))
    all_train = [(s.image_path, s.person_id, s.camera_id)
                 for s in ds.train if s.person_id > 0]
    print(f"  {len(all_train)} train images, "
          f"{len(set(p for _,p,_ in all_train))} IDs, "
          f"{len(set(c for _,_,c in all_train))} cams")

    print("Extracting MSMT17 train features (one pass)...")
    train_emb, train_pids, train_cids = extract_features(model, all_train, DEVICE)
    print(f"  Done: {train_emb.shape}")

    print("Loading Market test features (R4)...")
    mkt      = np.load(MARKET_CACHE)
    mkt_emb  = mkt['emb'].astype(np.float32)
    mkt_pids = mkt['pids']
    mkt_cids = mkt['cids']
    # Confirm / enforce L2 norm
    mkt_emb  = mkt_emb / (np.linalg.norm(mkt_emb, axis=1, keepdims=True) + 1e-10)
    print(f"  Market: {len(mkt_pids)} imgs, {len(np.unique(mkt_pids))} IDs")

    rng = np.random.default_rng(RNG_SEED)
    draw_results = []

    for draw in draws:
        draw_idx  = draw['draw']
        held_cams = set(draw['held_cameras'])
        seen_cams = set(draw['seen_cameras'])
        r3_n      = draw['n_ids_r3']
        print(f"\n--- Draw {draw_idx}: hold={sorted(held_cams)}  R3 target={r3_n} IDs ---")

        # Build ID sets for each rung
        def ids_for_rung(id_set, cam_set):
            return set(
                p for p, pid, cid in all_train
                if pid in id_set and cid in cam_set
            )

        r1_ids = {pid for _, pid, cid in all_train
                  if pid in in_split_ids and cid in seen_cams}
        r2_ids = {pid for _, pid, cid in all_train
                  if pid in held_out_ids and cid in seen_cams}
        r3_ids = {pid for _, pid, cid in all_train
                  if pid in held_out_ids and cid in held_cams}

        # Subsample R1, R2 to R3 bottleneck
        r1_matched_ids = subsample_pids(sorted(r1_ids), r3_n, rng)
        r2_matched_ids = subsample_pids(sorted(r2_ids), r3_n, rng)

        print(f"  R1 unmatched: {len(r1_ids)} IDs")
        print(f"  R1 matched:   {len(r1_matched_ids)} IDs")
        print(f"  R2 unmatched: {len(r2_ids)} IDs")
        print(f"  R2 matched:   {len(r2_matched_ids)} IDs")
        print(f"  R3:           {len(r3_ids)} IDs  (bottleneck)")

        dr = {"draw": draw_idx, "held_cameras": sorted(held_cams), "r3_target_ids": r3_n}

        # Evaluate each rung
        for tag, pid_set, cam_set in [
            ("r1_unmatched", r1_ids,         seen_cams),
            ("r1_matched",   r1_matched_ids,  seen_cams),
            ("r2_unmatched", r2_ids,          seen_cams),
            ("r2_matched",   r2_matched_ids,  seen_cams),
            ("r3",           r3_ids,          held_cams),
        ]:
            res = eval_rung(train_emb, train_pids, train_cids, pid_set, cam_set, rng)
            print(f"  {tag:16s}: {res['n_ids']:4d} IDs  mAP={res['map']:.4f}")
            dr[tag] = res

        # R4: Market subsampled to R3 bottleneck
        r4_res = eval_market_rung(mkt_emb, mkt_pids, mkt_cids, r3_n, rng)
        print(f"  {'r4_market':16s}: {r4_res['n_ids']:4d} IDs  mAP={r4_res['map']:.4f}")
        dr["r4_market"] = r4_res

        draw_results.append(dr)

    print("\n=== Summary (matched rungs) ===")
    print(f"{'Draw':>5} {'R1_match':>10} {'R2_match':>10} {'R3':>10} {'R4_Market':>10}")
    for dr in draw_results:
        print(f"{dr['draw']:>5} "
              f"{dr['r1_matched']['map']:>10.4f} "
              f"{dr['r2_matched']['map']:>10.4f} "
              f"{dr['r3']['map']:>10.4f} "
              f"{dr['r4_market']['map']:>10.4f}")

    # Averages
    avg = lambda key, sub=None: np.mean([
        dr[key]['map'] if sub is None else dr[key][sub]['map']
        for dr in draw_results
    ])
    print(f"\nMean across 6 draws:")
    print(f"  R1 (unmatched): {avg('r1_unmatched'):.4f}")
    print(f"  R1 (matched):   {avg('r1_matched'):.4f}")
    print(f"  R2 (matched):   {avg('r2_matched'):.4f}")
    print(f"  R3:             {avg('r3'):.4f}")
    print(f"  R4 (Market):    {avg('r4_market'):.4f}")

    out = {
        "model": "BoT-Market ResNet-50",
        "checkpoint": str(CKPT),
        "split_file": str(SPLIT_V3),
        "matching": "R1/R2/R4 subsampled to R3 bottleneck ID count",
        "protocol": ("one query per (ID,cam) pair; same-cam gallery excluded; "
                     "MSMT17 train images for R1-R3; Market-1501 test for R4"),
        "draw_results": draw_results,
        "means": {
            "r1_unmatched": round(float(avg('r1_unmatched')), 6),
            "r1_matched":   round(float(avg('r1_matched')), 6),
            "r2_matched":   round(float(avg('r2_matched')), 6),
            "r3":           round(float(avg('r3')), 6),
            "r4_market":    round(float(avg('r4_market')), 6),
        }
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved → {OUT_PATH}")


if __name__ == "__main__":
    main()
