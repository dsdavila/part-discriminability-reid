#!/usr/bin/env python3
"""
E16c shift ladder — v2 split definition.

- Stratified 70/30 ID split by camera count (seed=42)
- 5 random 5-camera holdout draws from the 15 MSMT17 cameras
- Reports indoor/outdoor composition per draw
- Reports matched ID count and image count for all 4 rungs
- Rung 4 (Market) gallery subsampled to match rung 3 ID count

MSMT17 camera labels (from dataset paper):
  Outdoor: cams 1-12
  Indoor:  cams 13-15
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict

OUT_PATH = Path("/data/papers/low_dimensional_reid/results/e16c_split_v2.json")
RNG_SEED = 42
HELD_OUT_FRAC = 0.30
N_DRAWS = 5
CAMS_PER_HOLD = 5

# MSMT17 cameras: 1-12 outdoor, 13-15 indoor (from dataset paper)
OUTDOOR_CAMS = set(range(1, 13))
INDOOR_CAMS  = set(range(13, 16))


def main():
    from reid_tk.datasets import get_dataset
    ds = get_dataset("msmt17", root=Path("/data/datasets/reid/MSMT17_V1"))

    # Build identity metadata from training set
    id_to_cams = defaultdict(set)
    id_cam_imgs = defaultdict(lambda: defaultdict(int))
    for s in ds.train:
        pid, cid = s.person_id, s.camera_id
        if pid >= 0:
            id_to_cams[pid].add(cid)
            id_cam_imgs[pid][cid] += 1

    all_ids = np.array(sorted(id_to_cams.keys()))
    cam_counts = np.array([len(id_to_cams[pid]) for pid in all_ids])

    # Stratified 70/30 ID split by camera count
    rng = np.random.default_rng(RNG_SEED)
    in_split_ids = []
    held_out_ids = []
    for n_cams in sorted(set(cam_counts)):
        stratum = all_ids[cam_counts == n_cams].copy()
        rng.shuffle(stratum)
        n_held = max(1, round(len(stratum) * HELD_OUT_FRAC))
        held_out_ids.extend(stratum[:n_held].tolist())
        in_split_ids.extend(stratum[n_held:].tolist())
    in_split_ids = set(in_split_ids)
    held_out_ids = set(held_out_ids)

    print(f"ID split: {len(in_split_ids)} in-split, {len(held_out_ids)} held-out")

    all_cams = set(range(1, 16))

    # 5 random 5-camera holdout draws
    draw_rng = np.random.default_rng(RNG_SEED + 1)
    draws = []

    for draw_idx in range(N_DRAWS):
        held_cams = set(draw_rng.choice(sorted(all_cams), CAMS_PER_HOLD, replace=False).tolist())
        seen_cams = all_cams - held_cams
        n_indoor_held  = len(held_cams & INDOOR_CAMS)
        n_outdoor_held = len(held_cams & OUTDOOR_CAMS)

        # Rung 1: in-split IDs × seen cameras (same-camera exclusion: query cam ≠ gallery cam)
        r1_ids = [pid for pid in in_split_ids
                  if any(c in seen_cams for c in id_to_cams[pid])]
        r1_imgs = sum(id_cam_imgs[pid][c]
                      for pid in r1_ids
                      for c in id_to_cams[pid] if c in seen_cams)

        # Rung 2: held-out IDs × seen cameras
        r2_ids = [pid for pid in held_out_ids
                  if any(c in seen_cams for c in id_to_cams[pid])]
        r2_imgs = sum(id_cam_imgs[pid][c]
                      for pid in r2_ids
                      for c in id_to_cams[pid] if c in seen_cams)

        # Rung 3: held-out IDs × held cameras
        r3_ids = [pid for pid in held_out_ids
                  if any(c in held_cams for c in id_to_cams[pid])]
        r3_imgs = sum(id_cam_imgs[pid][c]
                      for pid in r3_ids
                      for c in id_to_cams[pid] if c in held_cams)

        # Matched counts: use rung 3 as the bottleneck
        n_matched = len(r3_ids)

        draw_result = {
            "draw": draw_idx + 1,
            "held_cameras": sorted(held_cams),
            "seen_cameras": sorted(seen_cams),
            "held_indoor": n_indoor_held,
            "held_outdoor": n_outdoor_held,
            "rungs": {
                "rung1": {"n_ids": len(r1_ids), "n_images": r1_imgs},
                "rung2": {"n_ids": len(r2_ids), "n_images": r2_imgs},
                "rung3": {"n_ids": len(r3_ids), "n_images": r3_imgs},
            },
            "bottleneck_ids": n_matched,
        }
        draws.append(draw_result)

        print(f"Draw {draw_idx+1}: held={sorted(held_cams)} "
              f"(in:{n_indoor_held}/out:{n_outdoor_held})  "
              f"R1:{len(r1_ids)}/{r1_imgs}  R2:{len(r2_ids)}/{r2_imgs}  "
              f"R3:{len(r3_ids)}/{r3_imgs}  bottleneck={n_matched}")

    # Rung 4: Market gallery info (we'll report that gallery should be
    # subsampled to match rung 3's bottleneck ID count per draw)
    try:
        ds_mkt = get_dataset("market1501", root=Path("/data/datasets/reid/market1501"))
        mkt_gallery_pids = np.array([s.person_id for s in ds_mkt.gallery if s.person_id > 0])
        mkt_n_ids = len(np.unique(mkt_gallery_pids))
        mkt_n_imgs = len(mkt_gallery_pids)
        print(f"\nMarket gallery: {mkt_n_ids} IDs, {mkt_n_imgs} images")
        print(f"Rung 3 bottleneck range across draws: "
              f"{min(d['bottleneck_ids'] for d in draws)}–"
              f"{max(d['bottleneck_ids'] for d in draws)} IDs")
        print(f"Market gallery must be subsampled to match (e.g., randomly sample "
              f"{min(d['bottleneck_ids'] for d in draws)} IDs × their images)")
        rung4_info = {"n_ids_full": mkt_n_ids, "n_images_full": mkt_n_imgs,
                      "subsample_target_range": [min(d['bottleneck_ids'] for d in draws),
                                                  max(d['bottleneck_ids'] for d in draws)]}
    except Exception as e:
        print(f"Market load error: {e}")
        rung4_info = {"error": str(e)}

    results = {
        "id_split": {
            "rng_seed": RNG_SEED,
            "held_out_frac": HELD_OUT_FRAC,
            "n_in_split": len(in_split_ids),
            "n_held_out": len(held_out_ids),
            "in_split_ids": sorted(in_split_ids),
            "held_out_ids": sorted(held_out_ids),
        },
        "camera_holdout_draws": draws,
        "rung4_market": rung4_info,
        "camera_labels": {
            "outdoor": sorted(OUTDOOR_CAMS),
            "indoor": sorted(INDOOR_CAMS),
        }
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {OUT_PATH}")


if __name__ == "__main__":
    main()
