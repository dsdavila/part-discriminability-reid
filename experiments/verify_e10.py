#!/usr/bin/env python3
"""
E10 manipulation check:
  (a) Verify styled MSMT17 images' channel stats match Market targets.
  (b) Retrieval mAP on styled MSMT17 vs baseline 2.56%.
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "reid-tk"))
from reid_tk.datasets import get_dataset
from reid_tk.metrics import compute_map_rank1

CACHE = Path("/tmp/e12_feature_cache")


# ---------------------------------------------------------------------------
# (a) Channel stat verification on a sample of styled images
# ---------------------------------------------------------------------------

class _StyleTransferTransform:
    def __init__(self, src_mean, src_std, tgt_mean, tgt_std):
        self.src_mean = torch.tensor(src_mean, dtype=torch.float32).view(3, 1, 1)
        self.src_std  = torch.tensor(src_std,  dtype=torch.float32).view(3, 1, 1)
        self.tgt_mean = torch.tensor(tgt_mean, dtype=torch.float32).view(3, 1, 1)
        self.tgt_std  = torch.tensor(tgt_std,  dtype=torch.float32).view(3, 1, 1)

    def __call__(self, x):
        x = (x - self.src_mean) / (self.src_std + 1e-8) * self.tgt_std + self.tgt_mean
        return x.clamp(0.0, 1.0)


class _SampleDS(Dataset):
    def __init__(self, samples, transform):
        self.samples = [s for s in samples if not s.is_junk][:500]
        self.transform = transform

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        img = T.Compose([T.Resize((256, 128)), T.ToTensor()])(self.samples[idx].load_image())
        return self.transform(img)


def check_channel_stats(samples, style_tf, market_mean, market_std):
    ds = _SampleDS(samples, style_tf)
    loader = DataLoader(ds, batch_size=128, num_workers=0)
    ch_sum = torch.zeros(3)
    ch_sq  = torch.zeros(3)
    n = 0
    for batch in loader:
        ch_sum += batch.sum(dim=(0, 2, 3))
        ch_sq  += (batch ** 2).sum(dim=(0, 2, 3))
        n += batch.shape[0] * batch.shape[2] * batch.shape[3]
    mean = (ch_sum / n).numpy()
    std  = torch.sqrt(ch_sq / n - (ch_sum / n) ** 2).numpy()
    print(f"  Styled MSMT17 channel mean: {mean.round(4)}  (target: {market_mean.round(4)})")
    print(f"  Styled MSMT17 channel std:  {std.round(4)}   (target: {market_std.round(4)})")
    max_mean_err = np.abs(mean - market_mean).max()
    max_std_err  = np.abs(std  - market_std).max()
    print(f"  Max mean error: {max_mean_err:.4f}  Max std error: {max_std_err:.4f}")
    ok = max_mean_err < 0.02 and max_std_err < 0.02
    print(f"  Check: {'PASS' if ok else 'FAIL (residual > 0.02)'}")
    return ok


# ---------------------------------------------------------------------------
# (b) mAP on styled MSMT17 features
# ---------------------------------------------------------------------------

def check_map(msmt):
    n_query = len([s for s in msmt.query if not s.is_junk])
    n_gallery = len([s for s in msmt.gallery if not s.is_junk])

    # Baseline
    base = np.load(CACHE / "msmt17_test.npz")
    emb_b, pids_b, cids_b = base["emb"], base["pids"], base["cids"]
    q_emb_b, q_ids_b, q_cids_b = emb_b[:n_query], pids_b[:n_query], cids_b[:n_query]
    g_emb_b, g_ids_b, g_cids_b = emb_b[n_query:], pids_b[n_query:], cids_b[n_query:]
    map_b, r1_b = compute_map_rank1(q_emb_b, g_emb_b, q_ids_b, g_ids_b, q_cids_b, g_cids_b)

    # Styled
    styled = np.load(CACHE / "msmt17_styled_test.npz")
    emb_s, pids_s, cids_s = styled["emb"], styled["pids"], styled["cids"]
    q_emb_s, q_ids_s, q_cids_s = emb_s[:n_query], pids_s[:n_query], cids_s[:n_query]
    g_emb_s, g_ids_s, g_cids_s = emb_s[n_query:], pids_s[n_query:], cids_s[n_query:]
    map_s, r1_s = compute_map_rank1(q_emb_s, g_emb_s, q_ids_s, g_ids_s, q_cids_s, g_cids_s)

    print(f"\n  Baseline mAP: {map_b*100:.2f}%  R1: {r1_b*100:.2f}%")
    print(f"  Styled   mAP: {map_s*100:.2f}%  R1: {r1_s*100:.2f}%")
    print(f"  Δ mAP: {(map_s-map_b)*100:+.2f}pp")
    return map_b, map_s


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading channel stats...")
    market_stats = np.load(CACHE / "market_channel_stats.npz")
    msmt_stats   = np.load(CACHE / "msmt17_channel_stats.npz")
    market_mean, market_std = market_stats["mean"], market_stats["std"]
    msmt_mean,   msmt_std   = msmt_stats["mean"],   msmt_stats["std"]

    style_tf = _StyleTransferTransform(msmt_mean, msmt_std, market_mean, market_std)

    print("\n(a) Channel stat verification (500-image sample)...")
    msmt = get_dataset("msmt17", sample=False)
    check_channel_stats(msmt.query + msmt.gallery, style_tf, market_mean, market_std)

    print("\n(b) Retrieval mAP comparison...")
    check_map(msmt)

    print("\nDone.")


if __name__ == "__main__":
    main()
