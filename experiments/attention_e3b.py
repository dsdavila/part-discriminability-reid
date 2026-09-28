#!/usr/bin/env python3
"""
E3b — Per-image attention concentration analysis (CLIP-ReID ViT-B/16).

For each image in the Market-1501 gallery (test-split), extracts last-layer
CLS→patch attention from the CLIP-ReID image encoder and computes:

  top5_frac   fraction of attention mass in the top-5 tokens
  off_skel    fraction of attention mass outside all keypoint Gaussians (at 2σ)

Reports median and IQR over the gallery.  Images with no valid keypoints
contribute to top5_frac but are excluded from off_skel.

Architecture notes:
  CLIP-ReID ViT-B/16 trained on MSMT17 with stride=12 patch embedding.
  Input: 256 × 128 (H × W), patch_size=16, stride=12.
  Patch grid: floor((256-16)/12)+1 = 21 rows × floor((128-16)/12)+1 = 10 cols
  Tokens: 210 patches + 1 CLS = 211.  Positional embedding: (211, 768).

Usage:
  python scripts/attention_e3b.py \\
      --ckpt /data/papers/low_dimensional_reid/datasets/MSMT17_clipreid_12x12sie_ViT-B-16_60.pth \\
      --cache /tmp/pacm_cache_vitpose \\
      --output /tmp/e3b_results.json
"""

import argparse
import json
import logging
import math
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm

from reid_tk.datasets import get_dataset
from reid_tk.pacm.descriptors import BODY_KPT_INDICES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Patch-embedding geometry for this specific checkpoint
IMG_H, IMG_W = 256, 128
PATCH_SIZE = 16
STRIDE = 12
GRID_H = (IMG_H - PATCH_SIZE) // STRIDE + 1   # 21
GRID_W = (IMG_W - PATCH_SIZE) // STRIDE + 1   # 10
N_PATCHES = GRID_H * GRID_W                   # 210
N_TOKENS = N_PATCHES + 1                      # 211  (+ CLS)
EMBED_DIM = 768
NUM_HEADS = 12
NUM_LAYERS = 12


# ── CLIP ViT attention extractor ───────────────────────────────────────────────

class CLIPAttentionBlock(nn.Module):
    """Single CLIP transformer block that optionally saves CLS attention weights."""

    def __init__(self, d_model: int, n_heads: int, save_attn: bool = False):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.save_attn = save_attn

        self.ln_1 = nn.LayerNorm(d_model)
        self.attn_in = nn.Linear(d_model, 3 * d_model, bias=True)
        self.attn_out = nn.Linear(d_model, d_model, bias=True)
        self.ln_2 = nn.LayerNorm(d_model)
        self.mlp_c_fc = nn.Linear(d_model, 4 * d_model)
        self.mlp_c_proj = nn.Linear(4 * d_model, d_model)

        self.last_attn: torch.Tensor | None = None  # (B, N_patches)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        h = self.ln_1(x)

        qkv = self.attn_in(h)  # (B, N, 3D)
        q, k, v = qkv.split(D, dim=-1)

        # Reshape to (B*H, N, d_head)
        H, dh = self.n_heads, self.d_head
        q = q.reshape(B, N, H, dh).permute(0, 2, 1, 3).reshape(B * H, N, dh)
        k = k.reshape(B, N, H, dh).permute(0, 2, 1, 3).reshape(B * H, N, dh)
        v = v.reshape(B, N, H, dh).permute(0, 2, 1, 3).reshape(B * H, N, dh)

        scale = math.sqrt(dh)
        attn = torch.bmm(q, k.transpose(1, 2)) / scale  # (B*H, N, N)
        attn = attn.softmax(dim=-1)

        if self.save_attn:
            # CLS→patch attention, head-averaged: (B, N_patches)
            # attn shape: (B*H, N, N) → (B, H, N, N)
            attn_bh = attn.reshape(B, H, N, N)
            cls_attn = attn_bh[:, :, 0, 1:].mean(dim=1)  # (B, N_patches)
            self.last_attn = cls_attn.detach()

        out = torch.bmm(attn, v)  # (B*H, N, dh)
        out = out.reshape(B, H, N, dh).permute(0, 2, 1, 3).reshape(B, N, D)
        out = self.attn_out(out)
        x = x + out

        # MLP
        h2 = self.ln_2(x)
        h2 = self.mlp_c_proj(F.gelu(self.mlp_c_fc(h2)))
        x = x + h2
        return x


class CLIPImageEncoder(nn.Module):
    """Minimal CLIP ViT-B/16 image encoder for attention extraction.

    Loads weights from a CLIP-ReID checkpoint under the 'image_encoder.*' prefix.
    """

    def __init__(self, n_patches: int, embed_dim: int, n_heads: int, n_layers: int):
        super().__init__()
        self.conv1 = nn.Conv2d(3, embed_dim, kernel_size=PATCH_SIZE, stride=STRIDE, bias=False)
        self.class_embedding = nn.Parameter(torch.zeros(embed_dim))
        self.positional_embedding = nn.Parameter(torch.zeros(n_patches + 1, embed_dim))
        self.ln_pre = nn.LayerNorm(embed_dim)

        self.blocks = nn.ModuleList([
            CLIPAttentionBlock(embed_dim, n_heads, save_attn=(i == n_layers - 1))
            for i in range(n_layers)
        ])
        self.ln_post = nn.LayerNorm(embed_dim)

    @property
    def last_block(self) -> CLIPAttentionBlock:
        return self.blocks[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = self.conv1(x)           # (B, D, H_p, W_p)
        x = x.flatten(2).transpose(1, 2)  # (B, N_patches, D)

        cls = self.class_embedding.unsqueeze(0).unsqueeze(0).expand(B, 1, -1)
        x = torch.cat([cls, x], dim=1)  # (B, N+1, D)
        x = x + self.positional_embedding
        x = self.ln_pre(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.ln_post(x[:, 0, :])  # CLS token output
        return x


def load_clip_encoder(ckpt_path: Path, device: str) -> CLIPImageEncoder:
    """Load CLIP-ReID image encoder weights into CLIPImageEncoder."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    enc = CLIPImageEncoder(N_PATCHES, EMBED_DIM, NUM_HEADS, NUM_LAYERS)

    # Map checkpoint keys to our module:
    #   image_encoder.conv1.weight            → conv1.weight
    #   image_encoder.class_embedding         → class_embedding
    #   image_encoder.positional_embedding    → positional_embedding
    #   image_encoder.ln_pre.{weight,bias}    → ln_pre.{weight,bias}
    #   image_encoder.transformer.resblocks.N.attn.in_proj_weight → blocks.N.attn_in.weight
    #   image_encoder.transformer.resblocks.N.attn.in_proj_bias   → blocks.N.attn_in.bias
    #   image_encoder.transformer.resblocks.N.attn.out_proj.{weight,bias} → blocks.N.attn_out.{weight,bias}
    #   image_encoder.transformer.resblocks.N.ln_1.{weight,bias}  → blocks.N.ln_1.{weight,bias}
    #   image_encoder.transformer.resblocks.N.ln_2.{weight,bias}  → blocks.N.ln_2.{weight,bias}
    #   image_encoder.transformer.resblocks.N.mlp.c_fc.{weight,bias}   → blocks.N.mlp_c_fc.{weight,bias}
    #   image_encoder.transformer.resblocks.N.mlp.c_proj.{weight,bias} → blocks.N.mlp_c_proj.{weight,bias}

    state = {}
    for k, v in ckpt.items():
        if not k.startswith("image_encoder."):
            continue
        rest = k[len("image_encoder."):]

        if rest == "conv1.weight":
            state["conv1.weight"] = v
        elif rest == "class_embedding":
            state["class_embedding"] = v
        elif rest == "positional_embedding":
            state["positional_embedding"] = v
        elif rest.startswith("ln_pre."):
            state[rest] = v
        elif rest.startswith("transformer.resblocks."):
            parts = rest.split(".")
            n = parts[2]
            subkey = ".".join(parts[3:])
            prefix = f"blocks.{n}."
            if subkey == "attn.in_proj_weight":
                state[prefix + "attn_in.weight"] = v
            elif subkey == "attn.in_proj_bias":
                state[prefix + "attn_in.bias"] = v
            elif subkey == "attn.out_proj.weight":
                state[prefix + "attn_out.weight"] = v
            elif subkey == "attn.out_proj.bias":
                state[prefix + "attn_out.bias"] = v
            elif subkey in ("ln_1.weight", "ln_1.bias", "ln_2.weight", "ln_2.bias"):
                state[prefix + subkey] = v
            elif subkey == "mlp.c_fc.weight":
                state[prefix + "mlp_c_fc.weight"] = v
            elif subkey == "mlp.c_fc.bias":
                state[prefix + "mlp_c_fc.bias"] = v
            elif subkey == "mlp.c_proj.weight":
                state[prefix + "mlp_c_proj.weight"] = v
            elif subkey == "mlp.c_proj.bias":
                state[prefix + "mlp_c_proj.bias"] = v

    missing, unexpected = enc.load_state_dict(state, strict=False)
    if missing:
        log.warning(f"Missing keys: {missing[:5]}")
    if unexpected:
        log.warning(f"Unexpected keys: {unexpected[:5]}")
    enc.eval()
    return enc.to(device)


# ── Keypoint → patch-token mapping ────────────────────────────────────────────

def kpt_to_patch_gaussians(
    keypoints: np.ndarray,
    conf_threshold: float,
    sigma_patches: float = 2.0,
) -> np.ndarray | None:
    """Build a Gaussian on-skeleton mask over the 210-token patch grid.

    For each valid keypoint, place a Gaussian centred at the corresponding
    patch token. The mask is the sum of all Gaussians (clipped to [0,1]).
    Returns (N_PATCHES,) float32 mask, or None if no valid keypoints.
    """
    valid_kpts = []
    for idx in BODY_KPT_INDICES:
        if idx >= len(keypoints):
            continue
        x, y, conf = keypoints[idx]
        if conf < conf_threshold:
            continue
        # Map pixel → patch-grid coordinates
        # patch token at row r, col c covers pixels [r*STRIDE, r*STRIDE+PATCH_SIZE)
        # centre of patch (r,c) is at pixel (r*STRIDE + PATCH_SIZE/2, c*STRIDE + PATCH_SIZE/2)
        # Find nearest patch centre
        pc = (float(x) - PATCH_SIZE / 2) / STRIDE
        pr = (float(y) - PATCH_SIZE / 2) / STRIDE
        valid_kpts.append((pr, pc))

    if not valid_kpts:
        return None

    # Build Gaussian mask (H_grid, W_grid)
    rows = np.arange(GRID_H, dtype=np.float32)
    cols = np.arange(GRID_W, dtype=np.float32)
    RR, CC = np.meshgrid(rows, cols, indexing="ij")  # (H_p, W_p)

    mask = np.zeros((GRID_H, GRID_W), dtype=np.float32)
    for pr, pc in valid_kpts:
        g = np.exp(-((RR - pr) ** 2 + (CC - pc) ** 2) / (2 * sigma_patches ** 2))
        mask += g

    mask = np.clip(mask, 0.0, 1.0).reshape(N_PATCHES)
    return mask


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="E3b attention concentration analysis")
    parser.add_argument("--ckpt", type=Path,
                        default=Path("/data/papers/low_dimensional_reid/datasets/"
                                     "MSMT17_clipreid_12x12sie_ViT-B-16_60.pth"))
    parser.add_argument("--dataset", default="market1501")
    parser.add_argument("--cache", type=Path, default=Path("/tmp/pacm_cache_vitpose"))
    parser.add_argument("--conf", type=float, default=0.3)
    parser.add_argument("--sigma", type=float, default=2.0,
                        help="Gaussian sigma in patch units for skeleton mask (2σ criterion)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Device: {device}")

    # Load encoder
    log.info("Loading CLIP-ReID image encoder...")
    enc = load_clip_encoder(args.ckpt, device)

    # Load dataset + keypoints
    dataset = get_dataset(args.dataset, sample=False)
    kpt_tag = f"{args.dataset}_vitpose_conf{args.conf}"
    kpt_cache = args.cache / "keypoints" / kpt_tag / "gallery.pkl"
    log.info(f"Loading keypoints from {kpt_cache}")
    with open(kpt_cache, "rb") as f:
        kpts_list = pickle.load(f)

    # Filter to test-split (person_id > 0)
    gallery = [(s, kpt) for s, kpt in zip(dataset.gallery, kpts_list) if s.person_id > 0]
    log.info(f"Test-split gallery images: {len(gallery)}")

    transform = T.Compose([
        T.Resize((IMG_H, IMG_W)),
        T.ToTensor(),
        T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711]),  # CLIP normalisation
    ])

    top5_fracs = []
    off_skel_fracs = []
    n_no_kpts = 0

    log.info("Extracting attention maps...")
    for bs in tqdm(range(0, len(gallery), args.batch_size)):
        be = min(bs + args.batch_size, len(gallery))
        batch_samples = gallery[bs:be]

        imgs_t = torch.stack([
            transform(Image.open(s.image_path).convert("RGB"))
            for s, _ in batch_samples
        ]).to(device)

        with torch.no_grad():
            enc(imgs_t)  # forward pass to populate last_block.last_attn

        attn = enc.last_block.last_attn.cpu().numpy()  # (B, N_patches)

        for b_i, (s, kpts) in enumerate(batch_samples):
            a = attn[b_i]  # (210,)
            a_sum = a.sum()
            if a_sum < 1e-10:
                continue

            # top-5 fraction
            top5 = float(np.sort(a)[-5:].sum() / a_sum)
            top5_fracs.append(top5)

            # off-skeleton fraction (only when keypoints available)
            if kpts is None:
                n_no_kpts += 1
                continue
            skel_mask = kpt_to_patch_gaussians(kpts, args.conf, sigma_patches=args.sigma)
            if skel_mask is None:
                n_no_kpts += 1
                continue
            on_skel = float((a * skel_mask).sum() / a_sum)
            off_skel_fracs.append(1.0 - on_skel)

    top5_arr = np.array(top5_fracs)
    off_arr = np.array(off_skel_fracs)

    def iqr_stats(arr: np.ndarray) -> dict:
        return {
            "median": round(float(np.median(arr)), 4),
            "q25":    round(float(np.percentile(arr, 25)), 4),
            "q75":    round(float(np.percentile(arr, 75)), 4),
            "iqr":    round(float(np.percentile(arr, 75) - np.percentile(arr, 25)), 4),
            "n":      len(arr),
        }

    results = {
        "model": "CLIP-ReID ViT-B/16 (MSMT17-trained, cross-domain on Market-1501)",
        "sigma_patches": args.sigma,
        "top5_frac": iqr_stats(top5_arr),
        "off_skeleton_frac": iqr_stats(off_arr),
        "n_no_keypoints": n_no_kpts,
    }

    log.info("\n=== E3b Results ===")
    log.info(f"top-5 fraction:      median={results['top5_frac']['median']:.3f}  "
             f"IQR=[{results['top5_frac']['q25']:.3f}, {results['top5_frac']['q75']:.3f}]  "
             f"n={results['top5_frac']['n']}")
    log.info(f"off-skeleton frac:   median={results['off_skeleton_frac']['median']:.3f}  "
             f"IQR=[{results['off_skeleton_frac']['q25']:.3f}, {results['off_skeleton_frac']['q75']:.3f}]  "
             f"n={results['off_skeleton_frac']['n']}")
    log.info(f"images without keypoints (excluded from off-skel): {n_no_kpts}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        log.info(f"Saved → {args.output}")
    else:
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
