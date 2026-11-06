#!/usr/bin/env python3
"""
E16c shift ladder — define and document the stratified ID split.

Split MSMT17 training IDs into in-split (70%) and held-out (30%),
stratified by camera count. Reports the four rungs of the shift ladder:
  Rung 1: in-split IDs, in-split cameras (normal ReID)
  Rung 2: held-out IDs, in-split cameras only
  Rung 3: held-out IDs, held-out cameras (cameras not covered by in-split IDs)
  Rung 4: Market-1501 (cross-dataset)

Requirements:
  - Stratify ID split by camera count
  - Report surviving ID and image counts for each rung
  - Fix the split by seed for reproducibility
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict

OUT_PATH = Path("/data/papers/low_dimensional_reid/results/e16c_split.json")
RNG_SEED = 42
HELD_OUT_FRAC = 0.30


def main():
    from reid_tk.datasets import get_dataset
    ds = get_dataset("msmt17", root=Path("/data/datasets/reid/MSMT17_V1"))

    # Build identity metadata
    id_to_cams = defaultdict(set)
    id_cam_imgs = defaultdict(lambda: defaultdict(int))
    for s in ds.train:
        pid, cid = s.person_id, s.camera_id
        if pid >= 0:
            id_to_cams[pid].add(cid)
            id_cam_imgs[pid][cid] += 1

    all_ids = np.array(sorted(id_to_cams.keys()))
    cam_counts = np.array([len(id_to_cams[pid]) for pid in all_ids])
    print(f"Total IDs: {len(all_ids)}, cam count distribution:")
    for n in sorted(set(cam_counts)):
        print(f"  n_cams={n}: {(cam_counts==n).sum()} IDs")

    # Stratified split by camera count
    rng = np.random.default_rng(RNG_SEED)
    in_split_ids = set()
    held_out_ids = set()

    for n_cams in sorted(set(cam_counts)):
        stratum = all_ids[cam_counts == n_cams]
        rng.shuffle(stratum)
        n_held = max(1, round(len(stratum) * HELD_OUT_FRAC))
        held_out_ids.update(stratum[:n_held].tolist())
        in_split_ids.update(stratum[n_held:].tolist())

    print(f"\nSplit: {len(in_split_ids)} in-split IDs, {len(held_out_ids)} held-out IDs")

    # Camera sets
    cams_in_split  = set().union(*[id_to_cams[pid] for pid in in_split_ids])
    cams_held_out  = set().union(*[id_to_cams[pid] for pid in held_out_ids])
    cams_only_held = cams_held_out - cams_in_split  # cameras exclusive to held-out IDs
    cams_shared    = cams_held_out & cams_in_split

    print(f"In-split cameras: {sorted(cams_in_split)}")
    print(f"Held-out cameras (all): {sorted(cams_held_out)}")
    print(f"Cameras exclusive to held-out IDs: {sorted(cams_only_held)}")
    print(f"Shared cameras: {sorted(cams_shared)}")

    # Rung 1: in-split IDs, their own cameras
    rung1_ids = list(in_split_ids)
    rung1_imgs = sum(sum(id_cam_imgs[pid].values()) for pid in rung1_ids)
    print(f"\nRung 1 (in-split IDs, all cameras): {len(rung1_ids)} IDs, {rung1_imgs} images")

    # Rung 2: held-out IDs, in-split cameras only
    rung2_ids = []
    rung2_imgs = 0
    for pid in held_out_ids:
        imgs = sum(id_cam_imgs[pid][c] for c in id_to_cams[pid] if c in cams_in_split)
        if imgs > 0:
            rung2_ids.append(pid)
            rung2_imgs += imgs
    print(f"Rung 2 (held-out IDs, in-split cameras): {len(rung2_ids)} IDs, {rung2_imgs} images")

    # Rung 3: held-out IDs, exclusive cameras only
    if cams_only_held:
        rung3_ids = []
        rung3_imgs = 0
        for pid in held_out_ids:
            imgs = sum(id_cam_imgs[pid][c] for c in id_to_cams[pid] if c in cams_only_held)
            if imgs > 0:
                rung3_ids.append(pid)
                rung3_imgs += imgs
        print(f"Rung 3 (held-out IDs, exclusive cameras): {len(rung3_ids)} IDs, {rung3_imgs} images")
    else:
        rung3_ids, rung3_imgs = [], 0
        print("Rung 3: NO cameras exclusive to held-out IDs — within-MSMT17 camera holdout not feasible")

    # Rung 4: Market-1501 (just report counts from standard split)
    try:
        ds_mkt = get_dataset("market1501", root=Path("/data/datasets/reid/market1501"))
        mkt_query_ids = set(s.person_id for s in ds_mkt.query if s.person_id > 0)
        mkt_gallery_ids = set(s.person_id for s in ds_mkt.gallery if s.person_id > 0)
        mkt_gallery_imgs = sum(1 for s in ds_mkt.gallery if s.person_id > 0)
        print(f"Rung 4 (Market-1501): {len(mkt_query_ids)} query IDs, {len(mkt_gallery_ids)} gallery IDs, {mkt_gallery_imgs} gallery imgs")
    except Exception as e:
        print(f"Rung 4: Market-1501 load error: {e}")
        mkt_query_ids, mkt_gallery_ids = set(), set()

    results = {
        "rng_seed": RNG_SEED,
        "held_out_frac": HELD_OUT_FRAC,
        "in_split_ids": sorted(in_split_ids),
        "held_out_ids": sorted(held_out_ids),
        "cameras_in_split": sorted(cams_in_split),
        "cameras_held_out": sorted(cams_held_out),
        "cameras_exclusive_to_held_out": sorted(cams_only_held),
        "rungs": {
            "rung1_in_split": {"n_ids": len(rung1_ids), "n_images": rung1_imgs,
                               "description": "in-split IDs, in-split cameras"},
            "rung2_held_out_same_cameras": {"n_ids": len(rung2_ids), "n_images": rung2_imgs,
                                            "description": "held-out IDs, in-split cameras only"},
            "rung3_held_out_new_cameras": {"n_ids": len(rung3_ids), "n_images": rung3_imgs,
                                           "description": "held-out IDs, cameras exclusive to held-out IDs"},
            "rung4_market": {"n_ids": len(mkt_query_ids), "description": "Market-1501 cross-dataset transfer"},
        },
        "feasibility_note": (
            "Rung 3 requires cameras exclusive to held-out IDs. "
            + ("NOT FEASIBLE: all cameras are shared." if not cams_only_held
               else f"Feasible: {len(cams_only_held)} exclusive cameras found.")
        )
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {OUT_PATH}")


if __name__ == "__main__":
    main()
