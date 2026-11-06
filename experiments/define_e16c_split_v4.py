#!/usr/bin/env python3
"""
E16c split v4 — redefines over MSMT17 TEST IDs (option a).

Rationale: BoT-MSMT17 trained on 1,041 train IDs, so all train-set IDs are
seen. Splitting over the 3,059 test IDs gives genuinely novel identities for
all four rungs. BoT-MSMT17 has seen all 15 cameras during training, so R2/R3
are camera-composition rungs (gallery restricted to specific camera subsets),
NOT unseen-camera rungs. This is documented in the split file.

Split design:
  - 2,000 in-split test IDs, 1,059 held-out test IDs (seed=42)
  - Balanced 6-draw camera hold-out: each camera held exactly twice, no
    camera pair co-held more than once (same design as v3)
  - Camera split is over MSMT17's 15 cameras (cams 1-15)

Rung labels under option (a):
  R1: in-split IDs × seen cameras       (in-domain: known IDs, known cams)
  R2: held-out IDs × seen cameras       (open-set: novel IDs, same cam subset)
  R3: held-out IDs × held cameras       (camera-restricted: novel IDs, held-back cam subset)
  R4: Market-1501                        (cross-dataset)

NOTE: BoT-MSMT17 has NOT seen these test IDs during training (disjoint), but
HAS seen all 15 cameras. So:
  R1 → R2: identity shift only (genuine open-set step)
  R2 → R3: camera composition only (weaker shift than "unseen cameras")
  R3 → R4: full domain shift
"""

import json
import sys
from pathlib import Path
from itertools import combinations
from collections import Counter
import random
import numpy as np

sys.path.insert(0, '/workspace/reid-tk')

OUT_PATH = Path("/data/papers/low_dimensional_reid/results/e16c_split_v4.json")
SEED = 42

# Camera hold-out design (reuse v3's balanced draws — same 15-camera structure)
BALANCED_DRAWS_V3 = [
    {"draw": 1, "held_cameras": [1, 7, 10, 11, 12]},
    {"draw": 2, "held_cameras": [4, 8, 12, 13, 14]},
    {"draw": 3, "held_cameras": [2, 6, 11, 14, 15]},
    {"draw": 4, "held_cameras": [1, 3, 5, 13, 15]},
    {"draw": 5, "held_cameras": [2, 3, 4, 7, 9]},
    {"draw": 6, "held_cameras": [5, 6, 8, 9, 10]},
]


def main():
    from reid_tk.datasets import get_dataset
    ds = get_dataset('msmt17', root=Path('/data/datasets/reid/MSMT17_V1'))

    # MSMT17 uses independent 0-based ID namespaces for train and test sets:
    # train IDs 0–1040 and test IDs 0–3059 refer to DIFFERENT people in
    # different image directories (train/ vs test/). Integer overlap is expected
    # and correct. Verify by checking that no image paths are shared.
    train_paths = {str(s.image_path) for s in ds.train}
    test_paths  = {str(s.image_path) for s in ds.query} | {str(s.image_path) for s in ds.gallery}
    path_overlap = train_paths & test_paths
    if path_overlap:
        print(f"ERROR: {len(path_overlap)} image paths shared between train and test!")
        sys.exit(1)
    else:
        print(f"Disjointness verified: 0 shared image paths (train and test are different people)")
        print(f"  (Note: integer IDs overlap 0–1040, but train/ and test/ dirs are disjoint people)")

    train_ids  = {s.person_id for s in ds.train}
    query_ids  = {s.person_id for s in ds.query}
    gallery_ids = {s.person_id for s in ds.gallery}
    test_ids   = query_ids | gallery_ids
    print(f"Train IDs: {len(train_ids)} (0-indexed namespace)")
    print(f"Test IDs (query∪gallery): {len(test_ids)} (separate 0-indexed namespace)")
    print(f"Test query IDs: {len(query_ids)}")
    print(f"Test gallery IDs: {len(gallery_ids)}")

    # Get test images (gallery images, standard eval uses query vs gallery)
    # For the ladder we use gallery images since they have multi-camera coverage
    # Use is_junk to exclude distractors (person_id = -1); person_id=0 is valid.
    test_samples = [(s.image_path, s.person_id, s.camera_id) for s in ds.gallery if not s.is_junk]
    test_pids = sorted({s.person_id for s in ds.gallery if not s.is_junk})
    print(f"\nGallery: {len(test_samples)} images, {len(test_pids)} IDs")

    # Camera coverage in gallery
    cam_ids = sorted({s.camera_id for s in ds.gallery if s.person_id > 0})
    print(f"Gallery cameras: {cam_ids}")

    # Split test IDs: 2000 in-split, rest held-out
    rng = np.random.default_rng(SEED)
    shuffled = list(test_pids)
    rng.shuffle(shuffled)
    N_IN = 2000
    in_split_ids = sorted(shuffled[:N_IN])
    held_out_ids = sorted(shuffled[N_IN:])
    print(f"\nID split: {len(in_split_ids)} in-split, {len(held_out_ids)} held-out (seed={SEED})")

    # Per-draw rung stats
    in_split_set = set(in_split_ids)
    held_out_set = set(held_out_ids)

    draws_out = []
    for d in BALANCED_DRAWS_V3:
        draw_idx = d['draw']
        held_cams = set(d['held_cameras'])
        seen_cams = set(cam_ids) - held_cams

        r1_imgs = [(p, pid, cid) for p, pid, cid in test_samples
                   if pid in in_split_set and cid in seen_cams]
        r2_imgs = [(p, pid, cid) for p, pid, cid in test_samples
                   if pid in held_out_set and cid in seen_cams]
        r3_imgs = [(p, pid, cid) for p, pid, cid in test_samples
                   if pid in held_out_set and cid in held_cams]

        r1_ids_present = len({pid for _, pid, _ in r1_imgs})
        r2_ids_present = len({pid for _, pid, _ in r2_imgs})
        r3_ids_present = len({pid for _, pid, _ in r3_imgs})

        print(f"Draw {draw_idx}: hold={sorted(held_cams)} "
              f"R1={r1_ids_present}IDs/{len(r1_imgs)}imgs  "
              f"R2={r2_ids_present}IDs/{len(r2_imgs)}imgs  "
              f"R3={r3_ids_present}IDs/{len(r3_imgs)}imgs")

        draws_out.append({
            "draw": draw_idx,
            "held_cameras": sorted(held_cams),
            "seen_cameras": sorted(seen_cams),
            "n_ids_r1": r1_ids_present,
            "n_imgs_r1": len(r1_imgs),
            "n_ids_r2": r2_ids_present,
            "n_imgs_r2": len(r2_imgs),
            "n_ids_r3": r3_ids_present,
            "n_imgs_r3": len(r3_imgs),
        })

    split_doc = {
        "version": "v4",
        "description": (
            "E16c split option (a): split over MSMT17 test IDs. "
            "BoT-MSMT17 has NOT seen these IDs (disjointness verified). "
            "BoT-MSMT17 HAS seen all 15 cameras during training, so R2/R3 "
            "are camera-COMPOSITION rungs (gallery restricted), not unseen-camera rungs."
        ),
        "ladder_semantics": {
            "R1": "in-split IDs × seen cameras — in-domain (novel IDs w.r.t. eval, known cams)",
            "R2": "held-out IDs × seen cameras — open-set: novel IDs, same cam subset",
            "R3": "held-out IDs × held cameras — camera-restricted: novel IDs, held cam subset",
            "R4": "Market-1501 — full cross-dataset",
            "WARNING": ("BoT-MSMT17 saw all 15 cameras in training. "
                        "R2→R3 tests gallery camera composition, NOT genuinely unseen cameras. "
                        "Do NOT describe R2/R3 as an unseen-camera measurement.")
        },
        "id_split": {
            "n_train_ids": len(train_ids),
            "n_test_ids": len(test_ids),
            "n_in_split": len(in_split_ids),
            "n_held_out": len(held_out_ids),
            "in_split_ids": in_split_ids,
            "held_out_ids": held_out_ids,
            "seed": SEED,
            "source": "ds.gallery (MSMT17 test set, person_id>0)",
            "disjoint_from_train": True
        },
        "camera_labels": {
            "all_cams": cam_ids,
            "n_cams": len(cam_ids),
            "note": "Camera IDs as returned by reid_tk MSMT17 dataset"
        },
        "balanced_draws": {
            "n_draws": len(draws_out),
            "design": "each camera held exactly twice across 6 draws, no pair co-held twice",
            "draws": draws_out,
        },
        "rung4_market": {
            "dataset": "Market-1501",
            "note": "Standard test query/gallery; evaluated with same extractor"
        },
        "provenance": {
            "supersedes": "e16c_split_v3.json (was over train IDs — design conflict with BoT-MSMT17)",
            "script": "define_e16c_split_v4.py",
            "option": "(a): no new training runs needed"
        }
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, 'w') as f:
        json.dump(split_doc, f, indent=2)
    print(f"\nSaved → {OUT_PATH}")


if __name__ == "__main__":
    main()
