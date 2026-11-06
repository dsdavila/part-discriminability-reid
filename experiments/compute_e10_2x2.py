#!/usr/bin/env python3
"""
E10 2×2 model × dataset geometry experiment.

Fills in the second row of:
  rows: Market-trained BoT ResNet-50 | MSMT17-trained CLIP-ReID ViT-B/16
  cols: Market test features         | MSMT17 test features

Computes M1 (effective rank), M4 (hubness skew, size-matched), M5 (id/cam R²).

Architecture note: ResNet-50 vs ViT-B/16 are different, so flip direction
is directional evidence, not proof. In-domain numbers reported per model
to allow within-model comparison.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as T
from scipy.stats import skew
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "reid-tk"))
from reid_tk.datasets import get_dataset

# ---------------------------------------------------------------------------
# CLIP ViT-B/16 (stride-12) visual encoder
# ---------------------------------------------------------------------------

class _QuickGELU(nn.Module):
    def forward(self, x): return x * torch.sigmoid(1.702 * x)


class _ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int):
        super().__init__()
        self.attn   = nn.MultiheadAttention(d_model, n_head, batch_first=False)
        self.ln_1   = nn.LayerNorm(d_model)
        self.mlp    = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            _QuickGELU(),
            nn.Linear(d_model * 4, d_model),
        )
        self.ln_2   = nn.LayerNorm(d_model)

    def forward(self, x):
        y = self.ln_1(x)
        y, _ = self.attn(y, y, y, need_weights=False)
        x = x + y
        x = x + self.mlp(self.ln_2(x))
        return x


class _VisionTransformer(nn.Module):
    """CLIP ViT-B/16 with configurable conv stride."""

    def __init__(self, input_resolution, patch_size, stride, width, layers, heads):
        super().__init__()
        self.input_resolution = input_resolution
        self.conv1 = nn.Conv2d(3, width, kernel_size=patch_size, stride=stride, bias=False)
        scale = width ** -0.5
        self.class_embedding    = nn.Parameter(scale * torch.randn(width))
        h = (input_resolution[0] - patch_size) // stride + 1
        w = (input_resolution[1] - patch_size) // stride + 1
        n_patches = h * w
        self.positional_embedding = nn.Parameter(scale * torch.randn(1 + n_patches, width))
        self.ln_pre   = nn.LayerNorm(width)
        self.transformer = nn.Sequential(
            *[_ResidualAttentionBlock(width, heads) for _ in range(layers)]
        )
        self.ln_post = nn.LayerNorm(width)

    def forward(self, x):
        x = self.conv1(x)                                   # (B, C, H, W)
        B, C, H, W = x.shape
        x = x.reshape(B, C, -1).permute(0, 2, 1)           # (B, N, C)
        cls = self.class_embedding.unsqueeze(0).unsqueeze(0).expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)                     # (B, 1+N, C)
        x = x + self.positional_embedding
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)                             # (N+1, B, C) for attn
        x = self.transformer(x)
        x = x.permute(1, 0, 2)                             # (B, N+1, C)
        return self.ln_post(x[:, 0, :])                    # CLS token, (B, C)


def _load_clip_reid_extractor(checkpoint: Path, device: str) -> nn.Module:
    """Load CLIP-ReID image encoder + BN bottleneck. Returns 768-dim features."""
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)

    # Build ViT-B/16 stride-12 for 256×128 input
    vit = _VisionTransformer(
        input_resolution=(256, 128),
        patch_size=16, stride=12,
        width=768, layers=12, heads=12,
    )

    # Load image encoder weights (strip prefix, remap MLP keys, remap attn)
    ie_state = {}
    for k, v in ckpt.items():
        if not k.startswith("image_encoder."):
            continue
        k2 = k[len("image_encoder."):]
        # remap transformer.resblocks.N.* → transformer.N.*
        if k2.startswith("transformer.resblocks."):
            parts = k2.split(".")
            idx = parts[2]
            rest = ".".join(parts[3:])
            # remap MLP: mlp.c_fc → mlp.0, mlp.c_proj → mlp.2
            rest = rest.replace("mlp.c_fc", "mlp.0").replace("mlp.c_proj", "mlp.2")
            # remap attn: attn.in_proj_weight / in_proj_bias stay the same
            k2 = f"transformer.{idx}.{rest}"
        # drop proj (512-dim projection — we use raw 768-dim CLS)
        if k2 == "proj":
            continue
        ie_state[k2] = v

    missing, unexpected = vit.load_state_dict(ie_state, strict=False)
    if missing:
        print(f"  [CLIP ViT] missing keys: {missing[:5]}")
    if unexpected:
        print(f"  [CLIP ViT] unexpected keys: {unexpected[:5]}")

    # BN bottleneck (768-dim)
    bn = nn.BatchNorm1d(768)
    bn.bias.requires_grad_(False)
    bn_state = {k.replace("bottleneck.", ""): v
                for k, v in ckpt.items() if k.startswith("bottleneck.")}
    bn.load_state_dict(bn_state)

    class _Extractor(nn.Module):
        def __init__(self, vit, bn):
            super().__init__()
            self.vit = vit
            self.bn  = bn

        def forward(self, x):
            return self.bn(self.vit(x))

    extractor = _Extractor(vit, bn)
    return extractor.to(device).eval()


# Standard CLIP normalization (used by CLIP-ReID)
_CLIP_NORM = T.Normalize(
    mean=[0.48145466, 0.4578275, 0.40821073],
    std=[0.26862954, 0.26130258, 0.27577711],
)
_IMAGENET_NORM = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

_CLIP_TRANSFORM = T.Compose([T.Resize((256, 128)), T.ToTensor(), _CLIP_NORM])
_BOT_TRANSFORM  = T.Compose([T.Resize((256, 128)), T.ToTensor(), _IMAGENET_NORM])


# ---------------------------------------------------------------------------
# BoT ResNet-50 extractor (reused from compute_e12c)
# ---------------------------------------------------------------------------

def _load_bot_extractor(checkpoint: Path, device: str) -> nn.Module:
    resnet = models.resnet50(weights=None)
    resnet.layer4[0].conv2.stride = (1, 1)
    resnet.layer4[0].downsample[0].stride = (1, 1)
    base = nn.Sequential(*list(resnet.children())[:-2])
    bn = nn.BatchNorm1d(2048)
    bn.bias.requires_grad_(False)
    model = nn.Sequential(base, nn.AdaptiveAvgPool2d(1), nn.Flatten(), bn)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=False))
    return model.to(device).eval()


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

class _InferenceDS(Dataset):
    def __init__(self, samples, transform):
        self.samples   = [s for s in samples if not s.is_junk]
        self.transform = transform

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return self.transform(s.load_image()), s.person_id, s.camera_id


@torch.no_grad()
def extract_cached(model, samples, device, transform, cache_path: Path, batch_size=128):
    if cache_path.exists():
        print(f"  Loading cached from {cache_path}")
        d = np.load(cache_path)
        return d["emb"], d["pids"], d["cids"]
    n = len([s for s in samples if not s.is_junk])
    print(f"  Extracting {n} samples → {cache_path.name}")
    ds = _InferenceDS(samples, transform)
    loader = DataLoader(ds, batch_size=batch_size, num_workers=0, pin_memory=False)
    embs, pids, cids = [], [], []
    for imgs, ids, cams in loader:
        f = model(imgs.to(device))
        f = F.normalize(f, p=2, dim=1)
        embs.append(f.cpu().numpy())
        pids.extend(ids.numpy())
        cids.extend(cams.numpy())
    emb, pids, cids = np.vstack(embs), np.array(pids), np.array(cids)
    np.savez(cache_path, emb=emb, pids=pids, cids=cids)
    return emb, pids, cids


# ---------------------------------------------------------------------------
# Geometry (same as compute_e12c)
# ---------------------------------------------------------------------------

def effective_rank(X):
    Xc = X - X.mean(0)
    _, s, _ = np.linalg.svd(Xc / np.sqrt(len(X) - 1), full_matrices=False)
    lam = np.maximum(s ** 2, 0); lam = lam / lam.sum(); lam = lam[lam > 0]
    return float(np.exp(-np.sum(lam * np.log(lam))))


def hubness_skew(X, k=10, rng=None, n=5000):
    if rng is None: rng = np.random.default_rng(42)
    idx = rng.choice(len(X), min(n, len(X)), replace=False)
    Xs = X[idx]; sim = Xs @ Xs.T
    np.fill_diagonal(sim, -2.0)
    knn = np.argpartition(sim, -k, axis=1)[:, -k:]
    return float(skew(np.bincount(knn.ravel(), minlength=len(Xs))))


def variance_r2(X, ids, cids):
    mu = X.mean(0); total = float(np.mean(np.sum((X - mu) ** 2, axis=1))); N = len(X)
    def br2(labels):
        return sum(
            (labels == u).sum() / N * float(np.sum((X[labels == u].mean(0) - mu) ** 2))
            for u in np.unique(labels)
        ) / (total + 1e-8)
    return br2(ids), br2(cids)


def geometry(label, X, ids, cids, rng, ref_n_ids):
    er = effective_rank(X)
    # size-match hubness
    unique = np.unique(ids)
    if len(unique) > ref_n_ids:
        chosen = rng.choice(unique, ref_n_ids, replace=False)
        Xh = X[np.isin(ids, chosen)]
    else:
        Xh = X
    hub = hubness_skew(Xh, rng=rng)
    idr, cr = variance_r2(X, ids, cids)
    return {
        "label": label, "n": len(X),
        "n_ids": int(len(np.unique(ids))),
        "n_cams": int(len(np.unique(cids))),
        "eff_rank": er, "hubness_matched": hub,
        "identity_r2": idr, "camera_r2": cr,
        "ratio": idr / (cr + 1e-8),
    }


def print_row(g):
    print(f"  {g['label']:<40} er={g['eff_rank']:6.1f}  hub={g['hubness_matched']:5.3f}  "
          f"id={g['identity_r2']:.4f}  cam={g['camera_r2']:.4f}  ratio={g['ratio']:6.1f}×")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bot-ckpt", type=Path,
        default=Path("/data/papers/low_dimensional_reid/pretrained/bot_market1501_final.pth"))
    parser.add_argument("--clip-ckpt", type=Path,
        default=Path("/data/papers/low_dimensional_reid/datasets/MSMT17_clipreid_12x12sie_ViT-B-16_60.pth"))
    parser.add_argument("--cache-dir", type=Path, default=Path("/tmp/e12_feature_cache"))
    parser.add_argument("--out", type=Path,
        default=Path("/data/papers/low_dimensional_reid/results/e10_2x2_results.json"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    print("Loading datasets...")
    market = get_dataset("market1501", sample=False)
    msmt   = get_dataset("msmt17",     sample=False)
    src_samples = market.query + market.gallery
    tgt_samples = msmt.query   + msmt.gallery

    # -----------------------------------------------------------------------
    # Row 1: Market-trained BoT (reuse cached features)
    # -----------------------------------------------------------------------
    print("\n--- Row 1: Market-trained BoT ResNet-50 ---")
    bot = _load_bot_extractor(args.bot_ckpt, args.device)

    bot_src_emb, bot_src_ids, bot_src_cids = extract_cached(
        bot, src_samples, args.device, _BOT_TRANSFORM,
        args.cache_dir / "market_test.npz")
    bot_tgt_emb, bot_tgt_ids, bot_tgt_cids = extract_cached(
        bot, tgt_samples, args.device, _BOT_TRANSFORM,
        args.cache_dir / "msmt17_test.npz")

    ref_n_ids = len(np.unique(bot_src_ids))  # 750 for Market

    geo_bot_src = geometry("BoT / Market",  bot_src_emb, bot_src_ids, bot_src_cids, rng, ref_n_ids)
    geo_bot_tgt = geometry("BoT / MSMT17",  bot_tgt_emb, bot_tgt_ids, bot_tgt_cids, rng, ref_n_ids)

    del bot
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # -----------------------------------------------------------------------
    # Row 2: MSMT17-trained CLIP-ReID ViT-B/16
    # -----------------------------------------------------------------------
    print("\n--- Row 2: MSMT17-trained CLIP-ReID ViT-B/16 ---")
    clip_model = _load_clip_reid_extractor(args.clip_ckpt, args.device)

    clip_src_emb, clip_src_ids, clip_src_cids = extract_cached(
        clip_model, src_samples, args.device, _CLIP_TRANSFORM,
        args.cache_dir / "clipreid_market_test.npz")
    clip_tgt_emb, clip_tgt_ids, clip_tgt_cids = extract_cached(
        clip_model, tgt_samples, args.device, _CLIP_TRANSFORM,
        args.cache_dir / "clipreid_msmt17_test.npz")

    geo_clip_src = geometry("CLIP-ReID / Market",  clip_src_emb, clip_src_ids, clip_src_cids, rng, ref_n_ids)
    geo_clip_tgt = geometry("CLIP-ReID / MSMT17",  clip_tgt_emb, clip_tgt_ids, clip_tgt_cids, rng, ref_n_ids)

    # -----------------------------------------------------------------------
    # Print 2×2
    # -----------------------------------------------------------------------
    print("\n" + "=" * 90)
    print(f"{'Condition':<40} {'Eff.rank':>8} {'Hubness':>8} {'Id R²':>8} {'Cam R²':>8} {'ratio':>8}")
    print("=" * 90)
    for g in [geo_bot_src, geo_bot_tgt, geo_clip_src, geo_clip_tgt]:
        print_row(g)

    print("\n2×2 summary (eff. rank | hubness | id/cam ratio):")
    print(f"{'':30} Market             MSMT17")
    print(f"  BoT (Market-trained)    "
          f"er={geo_bot_src['eff_rank']:.0f} hub={geo_bot_src['hubness_matched']:.2f} ratio={geo_bot_src['ratio']:.0f}× | "
          f"er={geo_bot_tgt['eff_rank']:.0f} hub={geo_bot_tgt['hubness_matched']:.2f} ratio={geo_bot_tgt['ratio']:.0f}×")
    print(f"  CLIP-ReID (MSMT-trained)"
          f"er={geo_clip_src['eff_rank']:.0f} hub={geo_clip_src['hubness_matched']:.2f} ratio={geo_clip_src['ratio']:.0f}× | "
          f"er={geo_clip_tgt['eff_rank']:.0f} hub={geo_clip_tgt['hubness_matched']:.2f} ratio={geo_clip_tgt['ratio']:.0f}×")

    flip_rank = geo_clip_src["eff_rank"] < geo_clip_tgt["eff_rank"]
    flip_hub  = geo_clip_src["hubness_matched"] > geo_clip_tgt["hubness_matched"]
    print(f"\nFlip check (CLIP-ReID: Market < MSMT17 on rank={not flip_rank}, hub={not flip_hub}):")
    print(f"  Rank flips: {flip_rank}  Hub flips: {flip_hub}")

    # -----------------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------------
    results = {
        "grid": [geo_bot_src, geo_bot_tgt, geo_clip_src, geo_clip_tgt],
        "flip_rank": bool(flip_rank),
        "flip_hub":  bool(flip_hub),
        "note": (
            "flip_rank=True means CLIP-ReID (MSMT17-trained) shows lower eff. rank on Market than MSMT17 "
            "— collapse tracks out-of-domain direction. Architecture confound: ResNet-50 vs ViT-B/16."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
