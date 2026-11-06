#!/usr/bin/env python3
"""
CLIP-ReID ViT-B/16 effective rank collapse experiment.

Replicates E12 protocol (M1 spectral entropy effective rank, M4 hubness)
for CLIP-ReID ViT-B/16 trained on MSMT17:
  - Source: MSMT17 test gallery
  - Transfer: Market-1501 test gallery

Uses CLIPImageEncoder from attention_e3b.py. Applies image_encoder.proj
(768→512) to match the representation used for retrieval.
"""

import sys
import json
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torchvision.transforms as T

sys.path.insert(0, str(Path(__file__).parent))
from attention_e3b import load_clip_encoder, CLIPImageEncoder

CKPT = Path("/data/papers/low_dimensional_reid/datasets/MSMT17_clipreid_12x12sie_ViT-B-16_60.pth")
OUT_PATH = Path("/data/papers/low_dimensional_reid/results/clipreid_collapse_results.json")
BATCH = 64
IMG_H, IMG_W = 256, 128  # patch_size=16, stride=12 → 21×10=210 patches


class GalleryDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = [(s.image_path, s.person_id, s.camera_id) for s in samples]
        self.transform = transform

    def __len__(self): return len(self.samples)

    def __getitem__(self, i):
        path, pid, cid = self.samples[i]
        img = Image.open(path).convert("RGB")
        return self.transform(img), pid, cid


def get_transform():
    return T.Compose([
        T.Resize((IMG_H, IMG_W), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711]),
    ])


def extract_embeddings(model, proj, loader, device):
    model.eval()
    embs, pids, cids = [], [], []
    with torch.no_grad():
        for imgs, pid_batch, cid_batch in loader:
            imgs = imgs.to(device)
            # CLS token output (768D)
            out = model(imgs)          # (B, 768)
            # Apply projection to get 512D retrieval embedding
            out = out @ proj           # (B, 512): proj is (768, 512)
            out = F.normalize(out, dim=1)
            embs.append(out.cpu().numpy())
            pids.extend(pid_batch.tolist())
            cids.extend(cid_batch.tolist())
    return np.concatenate(embs, 0), np.array(pids), np.array(cids)


def effective_rank(X):
    """Spectral entropy effective rank: exp(-sum p_i log p_i), p_i = lam_i / sum(lam)."""
    Xc = X - X.mean(0)
    _, s, _ = np.linalg.svd(Xc / np.sqrt(len(Xc) - 1), full_matrices=False)
    lam = s**2
    p = lam / (lam.sum() + 1e-12)
    p = p[p > 0]
    return float(np.exp(-np.sum(p * np.log(p))))


def hubness_skew(X, k=10, n_samples=5000):
    rng = np.random.default_rng(42)
    idx = rng.choice(len(X), min(n_samples, len(X)), replace=False)
    Xs = X[idx]
    sims = Xs @ X.T
    np.fill_diagonal(sims, -1e9)
    top_k = np.argsort(-sims, axis=1)[:, :k]
    k_occ = np.bincount(top_k.ravel(), minlength=len(X))
    return float(k_occ.std() / k_occ.mean()) if k_occ.mean() > 0 else 0.0


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load encoder
    print("Loading CLIP-ReID image encoder...")
    encoder = load_clip_encoder(CKPT, device)
    encoder.eval()

    # Load projection matrix
    ck = torch.load(CKPT, map_location=device)
    proj = ck["image_encoder.proj"].float()  # (768, 512)

    transform = get_transform()

    from reid_tk.datasets import get_dataset
    results = {}

    for ds_name, ds_root, split_attr in [
        ("msmt17", Path("/data/datasets/reid/MSMT17_V1"), "gallery"),
        ("market1501", Path("/data/datasets/reid/Market-1501-v15.09.15"), "gallery"),
    ]:
        print(f"\nProcessing {ds_name} ({split_attr})...")
        try:
            ds = get_dataset(ds_name, root=ds_root)
            samples = getattr(ds, split_attr)
            dataset = GalleryDataset(samples, transform)
            loader = DataLoader(dataset, batch_size=BATCH, num_workers=0,
                                pin_memory=False, shuffle=False)
            embs, pids, cids = extract_embeddings(encoder, proj, loader, device)
            er = effective_rank(embs)
            hub = hubness_skew(embs)
            n_ids = len(np.unique(pids[pids > 0]))
            print(f"  {ds_name}: n={len(embs)}, ids={n_ids}, eff_rank={er:.1f}, hub={hub:.3f}")
            results[ds_name] = {"n": len(embs), "n_ids": n_ids,
                                "eff_rank": er, "hubness_skew": hub}
        except Exception as e:
            print(f"  ERROR: {e}")
            results[ds_name] = {"error": str(e)}

    # Rank retention
    if "msmt17" in results and "market1501" in results and \
       "eff_rank" in results["msmt17"] and "eff_rank" in results["market1501"]:
        er_src = results["msmt17"]["eff_rank"]    # in-domain (MSMT17-trained model on MSMT17)
        er_tgt = results["market1501"]["eff_rank"] # transfer (MSMT17-trained model on Market)
        # Compare against BoT-Market's in-domain (261.2) for the retention formula
        bot_market_er = 261.2
        retention = er_tgt / bot_market_er
        results["retention_vs_bot_market"] = retention
        results["raw_ratio"] = er_tgt / er_src
        print(f"\n=== CLIP-ReID Collapse ===")
        print(f"  MSMT17 (in-domain) eff_rank:  {er_src:.1f}")
        print(f"  Market (transfer)  eff_rank:  {er_tgt:.1f}")
        print(f"  Raw ratio (Market/MSMT17):     {er_tgt/er_src:.3f}")
        print(f"  Retention vs BoT-Market in-domain: {retention:.3f}")
        print(f"  BoT-MSMT17 had: MSMT17={359.9:.1f}, Market={207.2:.1f}, ratio={207.2/359.9:.3f}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {OUT_PATH}")


if __name__ == "__main__":
    main()
