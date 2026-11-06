#!/usr/bin/env python3
"""
E3b follow-up: skeleton-attention enrichment ratio.

Claude-chat (rev 36) asked: is off-skeleton attention due to skeleton area being
large (chance level) or genuine background preference?

Compute per image, using soft Gaussian masks:
  on_skel_attn_frac  = sum(attn * mask) / sum(attn)
  on_skel_area_frac  = sum(mask) / N_PATCHES
  enrichment         = on_skel_attn_frac / on_skel_area_frac

enrichment > 1 → skeleton-biased
enrichment < 1 → background-biased
enrichment ≈ 1 → indifferent

Also reports the median on-skeleton area fraction (answers "how much of the grid
is covered?") and caches (attn, top5_positions) for follow-up spatial-consistency check.
"""

import argparse
import json
import logging
import pickle
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm

# Reuse model + geometry from attention_e3b
import sys
sys.path.insert(0, str(Path(__file__).parent))
from attention_e3b import load_clip_encoder, kpt_to_patch_gaussians, N_PATCHES, IMG_H, IMG_W

from reid_tk.datasets import get_dataset
from reid_tk.pacm.descriptors import BODY_KPT_INDICES

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path,
                        default=Path("/data/papers/low_dimensional_reid/datasets/"
                                     "MSMT17_clipreid_12x12sie_ViT-B-16_60.pth"))
    parser.add_argument("--dataset", default="market1501")
    parser.add_argument("--cache", type=Path, default=Path("/tmp/pacm_cache_vitpose"))
    parser.add_argument("--conf", type=float, default=0.3)
    parser.add_argument("--sigma", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output", type=Path,
                        default=Path("/workspace/part-discriminability-reid/results/e3b_enrichment.json"))
    parser.add_argument("--attn-cache", type=Path,
                        default=Path("/tmp/e3b_attn_cache.pkl"),
                        help="Cache attention maps for follow-up checks")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Device: {device}")

    log.info("Loading CLIP-ReID image encoder...")
    enc = load_clip_encoder(args.ckpt, device)

    dataset = get_dataset(args.dataset, sample=False)
    kpt_tag = f"{args.dataset}_vitpose_conf{args.conf}"
    kpt_cache = args.cache / "keypoints" / kpt_tag / "gallery.pkl"
    log.info(f"Loading keypoints from {kpt_cache}")
    with open(kpt_cache, "rb") as f:
        kpts_list = pickle.load(f)

    gallery = [(s, kpt) for s, kpt in zip(dataset.gallery, kpts_list) if s.person_id > 0]
    log.info(f"Test-split gallery images: {len(gallery)}")

    transform = T.Compose([
        T.Resize((IMG_H, IMG_W)),
        T.ToTensor(),
        T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711]),
    ])

    enrichments = []
    on_skel_area_fracs = []
    on_skel_attn_fracs = []
    top5_positions_all = []   # (N_images, 5) patch indices — for spatial consistency check
    patch_norms_all = []      # (N_images, N_patches) — for norm-outlier check
    n_no_kpts = 0

    log.info("Extracting attention maps + computing enrichment...")
    for bs in tqdm(range(0, len(gallery), args.batch_size)):
        be = min(bs + args.batch_size, len(gallery))
        batch_samples = gallery[bs:be]

        imgs_t = torch.stack([
            transform(Image.open(s.image_path).convert("RGB"))
            for s, _ in batch_samples
        ]).to(device)

        with torch.no_grad():
            features = enc(imgs_t)  # also populates last_block.last_attn
            # patch token norms from final features (after projection, before BN)
            # enc returns the pooled BN feature, but we want pre-pool patch norms
            # Use the patch tokens stored in last_block's input instead.
            # Actually we'll use last_block.patch_tokens if available, else skip norm check.

        attn = enc.last_block.last_attn.cpu().numpy()  # (B, N_patches)

        for b_i, (s, kpts) in enumerate(batch_samples):
            a = attn[b_i]  # (210,)
            a_sum = a.sum()
            if a_sum < 1e-10:
                continue

            # Top-5 patch positions (for spatial consistency check later)
            top5_idx = np.argsort(a)[-5:]
            top5_positions_all.append(top5_idx)

            if kpts is None:
                n_no_kpts += 1
                continue
            mask = kpt_to_patch_gaussians(kpts, args.conf, sigma_patches=args.sigma)
            if mask is None:
                n_no_kpts += 1
                continue

            on_attn = float((a * mask).sum() / a_sum)
            on_area = float(mask.sum() / N_PATCHES)
            if on_area < 1e-6:
                n_no_kpts += 1
                continue

            enrichment = on_attn / on_area

            on_skel_attn_fracs.append(on_attn)
            on_skel_area_fracs.append(on_area)
            enrichments.append(enrichment)

    def stats(arr):
        arr = np.array(arr)
        return {
            "median": round(float(np.median(arr)), 4),
            "q25":    round(float(np.percentile(arr, 25)), 4),
            "q75":    round(float(np.percentile(arr, 75)), 4),
            "mean":   round(float(np.mean(arr)), 4),
            "n":      len(arr),
        }

    results = {
        "model": "CLIP-ReID ViT-B/16 (MSMT17-trained, cross-domain Market-1501)",
        "sigma_patches": args.sigma,
        "n_images_with_keypoints": len(enrichments),
        "n_no_keypoints": n_no_kpts,
        "on_skel_area_frac": stats(on_skel_area_fracs),
        "on_skel_attn_frac": stats(on_skel_attn_fracs),
        "enrichment": stats(enrichments),
        "note": (
            "enrichment = on_skel_attn_frac / on_skel_area_frac. "
            ">1 = skeleton-biased, <1 = background-biased, ~1 = chance. "
            "Soft Gaussian masks (sigma=2 patch units), no binary threshold."
        ),
    }

    log.info("\n=== E3b Enrichment Results ===")
    log.info(f"on-skeleton area frac: median={results['on_skel_area_frac']['median']:.3f}  "
             f"IQR=[{results['on_skel_area_frac']['q25']:.3f}, {results['on_skel_area_frac']['q75']:.3f}]")
    log.info(f"on-skeleton attn frac: median={results['on_skel_attn_frac']['median']:.3f}  "
             f"IQR=[{results['on_skel_attn_frac']['q25']:.3f}, {results['on_skel_attn_frac']['q75']:.3f}]")
    log.info(f"ENRICHMENT:            median={results['enrichment']['median']:.3f}  "
             f"IQR=[{results['enrichment']['q25']:.3f}, {results['enrichment']['q75']:.3f}]")
    log.info(f"(>1 = skeleton-biased, <1 = background-biased, ~1 = chance)")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"Saved → {args.output}")

    # Cache top-5 positions for spatial consistency check
    if top5_positions_all:
        with open(args.attn_cache, "wb") as f:
            pickle.dump({"top5_positions": np.array(top5_positions_all)}, f)
        log.info(f"Attention cache → {args.attn_cache}")


if __name__ == "__main__":
    main()
