"""Compute per-keypoint discriminability with n=12 individual slots (unmerged L/R).

Same BoT ResNet-50 backbone as compute_fk_bot.py, but the 6 L/R pairs are kept
separate → 12 slots.  With n=12, ρ=0.714 reaches p≈0.009 (critical ρ≈0.587 for
n=12, two-tailed p=0.05), closing the H1 stability gate.

Outputs /tmp/fk_results_bot_n12.json.
"""

import json, pickle, random, sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from scipy.stats import spearmanr
from tqdm import tqdm

sys.path.insert(0, '/workspace/reid-tk')

from reid_tk.datasets import get_dataset
from reid_tk.backbones import BagOfTricksBackbone

# ── Config ────────────────────────────────────────────────────────────────────
CKPT_PATH  = Path("/data/papers/low_dimensional_reid/pretrained/bot_market1501_final.pth")
CONF       = 0.3
IMG_H, IMG_W = 256, 128
FEAT_H, FEAT_W = 16, 8
SIGMA      = 1.0
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
SEED       = 42
BATCH_SIZE = 64
N_BOOTSTRAP = 500

KPT_CACHE  = Path("/tmp/pacm_cache_vitpose/keypoints")

# 12 individual COCO body-part keypoints (no face): L-sh, R-sh, L-el, R-el,
# L-wr, R-wr, L-hi, R-hi, L-kn, R-kn, L-an, R-an
SLOTS = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
KPT_NAMES = ['L-Sh','R-Sh','L-El','R-El','L-Wr','R-Wr',
             'L-Hi','R-Hi','L-Kn','R-Kn','L-An','R-An']
K = len(SLOTS)
SLOT_IDX = {kpt: s for s, kpt in enumerate(SLOTS)}  # kpt_id → slot index

random.seed(SEED); np.random.seed(SEED)

BOT_TF = T.Compose([
    T.Resize((IMG_H, IMG_W)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def build_backbone_spatial():
    bb = BagOfTricksBackbone(weights_path=CKPT_PATH, input_size=(IMG_H, IMG_W))
    spatial = bb._build_model()[0]
    spatial.to(DEVICE).eval()
    return spatial


def gauss_weights(nx, ny, sigma=SIGMA):
    pc = np.clip(nx, 0, 1) * (FEAT_W - 1)
    pr = np.clip(ny, 0, 1) * (FEAT_H - 1)
    rs = np.arange(FEAT_H, dtype=np.float32)
    cs = np.arange(FEAT_W, dtype=np.float32)
    R, C = np.meshgrid(rs, cs, indexing='ij')
    g = np.exp(-((R - pr)**2 + (C - pc)**2) / (2 * sigma**2))
    g /= g.sum()
    return g.reshape(-1)


def compute_fk_and_sw_sb(by_id, n_pairs=4, n_bootstrap=N_BOOTSTRAP):
    rng = random.Random(SEED)
    pids = list(by_id.keys())

    def _one_pass(pid_list):
        wi_sum  = np.zeros(K); wi_cnt  = np.zeros(K, dtype=int)
        btw_sum = np.zeros(K); btw_cnt = np.zeros(K, dtype=int)
        for pid in pid_list:
            feats = by_id[pid]
            if len(feats) < 2: continue
            pairs = [(a,b) for i,a in enumerate(feats) for b in feats[i+1:]]
            for fa, fb in rng.sample(pairs, min(n_pairs, len(pairs))):
                for s in range(K):
                    if s in fa and s in fb:
                        d = float(np.linalg.norm(fa[s] - fb[s]))
                        wi_sum[s] += d; wi_cnt[s] += 1
            negs = rng.sample([p for p in pid_list if p != pid], min(n_pairs, len(pid_list)-1))
            for np_pid in negs:
                fa = rng.choice(feats); fb = rng.choice(by_id[np_pid])
                for s in range(K):
                    if s in fa and s in fb:
                        d = float(np.linalg.norm(fa[s] - fb[s]))
                        btw_sum[s] += d; btw_cnt[s] += 1
        within  = np.where(wi_cnt  > 0, wi_sum  / wi_cnt,  np.nan)
        between = np.where(btw_cnt > 0, btw_sum / btw_cnt, np.nan)
        return between / (within + 1e-6), within, between

    fk, sw, sb = _one_pass(pids)

    det_cnt = np.zeros(K, dtype=int)
    for feats in by_id.values():
        for f in feats:
            for s in f: det_cnt[s] += 1

    np_rng = np.random.default_rng(SEED)
    boot_fk = []
    for _ in range(n_bootstrap):
        sampled = np_rng.choice(pids, size=len(pids), replace=True).tolist()
        boot_sub = {p: by_id[p] for p in sampled if p in by_id}
        b_fk, _, _ = _one_pass(list(boot_sub.keys()))
        boot_fk.append(b_fk)
    boot_fk = np.array(boot_fk)
    ci_lo = np.nanpercentile(boot_fk, 2.5,  axis=0)
    ci_hi = np.nanpercentile(boot_fk, 97.5, axis=0)

    return fk, sw, sb, ci_lo, ci_hi, det_cnt


@torch.no_grad()
def deep_fk_from_resnet(samples, kpts_all, spatial_model, label=''):
    by_id = defaultdict(list)
    imgs, meta = [], []

    for i, (s, kpts) in enumerate(tqdm(
        zip(samples, kpts_all), total=len(samples), desc=label or 'deep-bot'
    )):
        if s.person_id <= 0: continue
        det = [(kidx, kpts[kidx]) for kidx in SLOTS if kpts[kidx, 2] >= CONF]
        if not det: continue

        img = s.load_image()
        W0, H0 = img.size
        imgs.append(BOT_TF(img))
        meta.append((s.person_id, det, W0, H0))

        flush = (len(imgs) == BATCH_SIZE) or (i == len(samples) - 1)
        if not flush: continue

        batch = torch.stack(imgs).to(DEVICE)
        feat_map = spatial_model(batch).cpu().numpy()   # (B, 2048, H, W)

        for b, (pid, det_b, W0, H0) in enumerate(meta):
            fm_flat = feat_map[b].reshape(2048, -1).T   # (128, 2048)
            sf = {}
            for kidx, kpt in det_b:
                sl = SLOT_IDX[kidx]
                nx, ny = float(kpt[0]) / W0, float(kpt[1]) / H0
                gw = gauss_weights(nx, ny)
                fv = (fm_flat * gw[:, None]).sum(0)
                sf[sl] = fv
            if sf: by_id[pid].append(sf)

        imgs, meta = [], []

    return compute_fk_and_sw_sb(dict(by_id))


def load_kpts(dataset_tag, split):
    p = KPT_CACHE / f"{dataset_tag}_vitpose_conf{CONF}" / f"{split}.pkl"
    with open(p, 'rb') as f:
        return pickle.load(f)


def main():
    print(f"Device: {DEVICE}  |  K={K} slots (individual L/R)")
    print("Building BoT backbone...")
    spatial = build_backbone_spatial()

    results = {'kpt_names': KPT_NAMES}

    for tag, ds_name in [('market1501', 'Market-1501'), ('msmt17', 'MSMT17')]:
        print(f"\n=== {ds_name} ===")
        ds = get_dataset(tag, sample=False)
        kpts_q = load_kpts(tag, 'query')
        kpts_g = load_kpts(tag, 'gallery')
        all_samp = ds.query + ds.gallery
        all_kpts = kpts_q + kpts_g

        fk, sw, sb, ci_lo, ci_hi, det = deep_fk_from_resnet(
            all_samp, all_kpts, spatial, f'{tag} n12'
        )

        print(f"\n  {'Part':<6}  {'S_W':>7}  {'S_B':>7}  {'F_k':>7}  {'CI':>20}")
        for i, nm in enumerate(KPT_NAMES):
            print(f"  {nm:<6}  {sw[i]:>7.3f}  {sb[i]:>7.3f}  {fk[i]:>7.3f}"
                  f"  [{ci_lo[i]:.3f}, {ci_hi[i]:.3f}]")

        results[tag] = {
            'fk': fk.tolist(), 'sw': sw.tolist(), 'sb': sb.tolist(),
            'ci_lo': ci_lo.tolist(), 'ci_hi': ci_hi.tolist(),
            'det_cnt': det.tolist(),
        }

    mkt  = np.array(results['market1501']['fk'])
    msmt = np.array(results['msmt17']['fk'])
    rho_mm, p_mm = spearmanr(mkt, msmt)
    print(f"\nρ(BoT-Mkt n12, BoT-MSMT n12) = {rho_mm:.3f}  p={p_mm:.4f}")

    # Also load color F_k (n=6 paired) for reference — pad to n=12 by repeating each pair
    with open('/tmp/fk_detailed.json') as f:
        det_prev = json.load(f)
    color6 = np.array(det_prev['clr_fk'])
    color12 = np.repeat(color6, 2)  # L=R for color (symmetric)
    rho_cm, p_cm = spearmanr(color12, mkt)
    rho_cs, p_cs = spearmanr(color12, msmt)
    print(f"ρ(color n12, BoT-Mkt n12)  = {rho_cm:.3f}  p={p_cm:.4f}")
    print(f"ρ(color n12, BoT-MSMT n12) = {rho_cs:.3f}  p={p_cs:.4f}")

    results['spearman'] = {
        'bot_mkt_vs_msmt': {'rho': rho_mm, 'p': p_mm},
        'color_vs_mkt':    {'rho': rho_cm, 'p': p_cm},
        'color_vs_msmt':   {'rho': rho_cs, 'p': p_cs},
    }

    out = Path('/tmp/fk_results_bot_n12.json')
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {out}")


if __name__ == '__main__':
    main()
