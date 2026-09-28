"""Compute per-keypoint discriminability with the Market-trained BoT ResNet-50.

Replaces CLIP-ViT with BagOfTricksBackbone (ResNet-50, last_stride=1, BNNeck).
For each body-part slot, Gaussian-weighted pooling over the 16×8 spatial feature
map gives a 2048-dim feature vector; F_k is computed identically to compute_fk.py.

Datasets:
  Market-1501  → in-domain for BoT  (E2 per spec)
  MSMT17       → cross-domain for BoT  (E4 per spec)

Outputs /tmp/fk_results_bot.json.
"""

import json, pickle, random, sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from tqdm import tqdm

# ── Add reid-tk to path if needed ────────────────────────────────────────────
sys.path.insert(0, '/workspace/reid-tk')

from reid_tk.datasets import get_dataset
from reid_tk.backbones import BagOfTricksBackbone
from reid_tk.pacm.descriptors import BODY_KPT_INDICES

# ── Config ────────────────────────────────────────────────────────────────────
CKPT_PATH  = Path("/data/papers/low_dimensional_reid/pretrained/bot_market1501_final.pth")
CONF       = 0.3
IMG_H, IMG_W = 256, 128       # BoT training size
FEAT_H, FEAT_W = 16, 8       # spatial feature map after ResNet-50 (last_stride=1)
SIGMA      = 1.0
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
SEED       = 42
BATCH_SIZE = 64
N_BOOTSTRAP = 500

KPT_CACHE  = Path("/tmp/pacm_cache_vitpose/keypoints")

BODY_PAIRS = [(5,6), (7,8), (9,10), (11,12), (13,14), (15,16)]
KPT_NAMES  = ['Sh', 'El', 'Wr', 'Hi', 'Kn', 'An']
K = len(BODY_PAIRS)

random.seed(SEED); np.random.seed(SEED)

BOT_TF = T.Compose([
    T.Resize((IMG_H, IMG_W)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ── Build model — return just the backbone (spatial feature map) ──────────────

def build_backbone_spatial():
    """Return BoT backbone that outputs (B, 2048, FEAT_H, FEAT_W)."""
    bb = BagOfTricksBackbone(weights_path=CKPT_PATH, input_size=(IMG_H, IMG_W))
    full_model = bb._build_model()   # Sequential: [backbone, gap, flatten, bnneck]
    # Extract just the spatial backbone (module 0)
    spatial = full_model[0]
    spatial.to(DEVICE).eval()
    return spatial


# ── Gaussian weights over the spatial feature map ────────────────────────────

def gauss_weights(nx, ny, sigma=SIGMA):
    """Return (FEAT_H * FEAT_W,) Gaussian weight map centred at normalised (nx, ny)."""
    pc = np.clip(nx, 0, 1) * (FEAT_W - 1)
    pr = np.clip(ny, 0, 1) * (FEAT_H - 1)
    rs = np.arange(FEAT_H, dtype=np.float32)
    cs = np.arange(FEAT_W, dtype=np.float32)
    R, C = np.meshgrid(rs, cs, indexing='ij')
    g = np.exp(-((R - pr)**2 + (C - pc)**2) / (2 * sigma**2))
    g /= g.sum()
    return g.reshape(-1)   # (FEAT_H*FEAT_W,)


# ── F_k computation ───────────────────────────────────────────────────────────

def compute_fk_and_sw_sb(by_id, n_pairs=4, n_bootstrap=N_BOOTSTRAP):
    """
    Returns:
        fk      (K,)  F_k = S_B / S_W
        sw      (K,)  mean within-ID distance
        sb      (K,)  mean between-ID distance
        ci_lo   (K,)  2.5th percentile of bootstrap F_k
        ci_hi   (K,)  97.5th percentile of bootstrap F_k
        det_cnt (K,)  detection count
    """
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

    # Detection counts
    det_cnt = np.zeros(K, dtype=int)
    for feats in by_id.values():
        for f in feats:
            for s in f: det_cnt[s] += 1

    # Bootstrap CIs (resample identities)
    boot_fk = []
    np_rng = np.random.default_rng(SEED)
    for _ in range(n_bootstrap):
        sampled = np_rng.choice(pids, size=len(pids), replace=True).tolist()
        boot_sub = {p: by_id[p] for p in sampled if p in by_id}
        b_fk, _, _ = _one_pass(list(boot_sub.keys()))
        boot_fk.append(b_fk)
    boot_fk = np.array(boot_fk)
    ci_lo = np.nanpercentile(boot_fk, 2.5, axis=0)
    ci_hi = np.nanpercentile(boot_fk, 97.5, axis=0)

    return fk, sw, sb, ci_lo, ci_hi, det_cnt


# ── Deep F_k from ResNet-50 spatial features ─────────────────────────────────

@torch.no_grad()
def deep_fk_from_resnet(samples, kpts_all, spatial_model, label=''):
    by_id = defaultdict(list)
    imgs, meta = [], []

    for i, (s, kpts) in enumerate(tqdm(
        zip(samples, kpts_all), total=len(samples), desc=label or 'deep-bot'
    )):
        if s.person_id <= 0: continue
        det = [(kidx, kpts[kidx]) for kidx in range(17) if kpts[kidx, 2] >= CONF]
        if not det: continue

        img = s.load_image()
        W0, H0 = img.size
        imgs.append(BOT_TF(img))
        meta.append((s.person_id, det, W0, H0))

        flush = (len(imgs) == BATCH_SIZE) or (i == len(samples) - 1)
        if not flush: continue

        batch = torch.stack(imgs).to(DEVICE)
        feat_map = spatial_model(batch)                   # (B, 2048, FEAT_H, FEAT_W)
        feat_map = feat_map.cpu().numpy()                 # (B, 2048, H, W)

        for b, (pid, det_b, W0, H0) in enumerate(meta):
            fm = feat_map[b]                              # (2048, FEAT_H, FEAT_W)
            fm_flat = fm.reshape(2048, -1).T              # (FEAT_H*FEAT_W, 2048)
            sf = {}
            for kidx, kpt in det_b:
                nx, ny = float(kpt[0]) / W0, float(kpt[1]) / H0
                gw = gauss_weights(nx, ny)                # (FEAT_H*FEAT_W,)
                fv = (fm_flat * gw[:, None]).sum(0)       # (2048,)
                for sl, (l, r) in enumerate(BODY_PAIRS):
                    if kidx in (l, r):
                        sf[sl] = (sf[sl] + fv) / 2 if sl in sf else fv
            if sf: by_id[pid].append(sf)

        imgs, meta = [], []

    return compute_fk_and_sw_sb(dict(by_id))


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_kpts(dataset_tag, split):
    p = KPT_CACHE / f"{dataset_tag}_vitpose_conf{CONF}" / f"{split}.pkl"
    with open(p, 'rb') as f:
        return pickle.load(f)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"Device: {DEVICE}")
    print("Building BoT backbone (spatial feature extractor)...")
    spatial = build_backbone_spatial()
    print(f"  Feature map size: {FEAT_H}×{FEAT_W} = {FEAT_H*FEAT_W} patches  (2048-dim each)")

    results = {}

    # ── Market-1501 (in-domain) ───────────────────────────────────────────────
    print("\n=== Market-1501 (in-domain for BoT) ===")
    ds = get_dataset('market1501', sample=False)
    kpts_q = load_kpts('market1501', 'query')
    kpts_g = load_kpts('market1501', 'gallery')
    all_samp = ds.query + ds.gallery
    all_kpts = kpts_q + kpts_g

    mkt_fk, mkt_sw, mkt_sb, mkt_ci_lo, mkt_ci_hi, mkt_det = deep_fk_from_resnet(
        all_samp, all_kpts, spatial, 'market deep-bot'
    )

    print("\nMarket-1501 deep F_k (BoT, in-domain):")
    print(f"  {'Part':<6}  {'S_W':>7}  {'S_B':>7}  {'F_k':>7}  {'CI_lo':>7}  {'CI_hi':>7}")
    for i, nm in enumerate(KPT_NAMES):
        print(f"  {nm:<6}  {mkt_sw[i]:>7.3f}  {mkt_sb[i]:>7.3f}  {mkt_fk[i]:>7.3f}  "
              f"{mkt_ci_lo[i]:>7.3f}  {mkt_ci_hi[i]:>7.3f}")

    results['market'] = {
        'fk': mkt_fk.tolist(), 'sw': mkt_sw.tolist(), 'sb': mkt_sb.tolist(),
        'ci_lo': mkt_ci_lo.tolist(), 'ci_hi': mkt_ci_hi.tolist(),
        'det_cnt': mkt_det.tolist(),
    }

    # ── MSMT17 (cross-domain) ─────────────────────────────────────────────────
    print("\n=== MSMT17 (cross-domain for BoT) ===")
    ds2 = get_dataset('msmt17', sample=False)

    # Check if MSMT17 keypoints are cached
    kpts_q2_path = KPT_CACHE / f"msmt17_vitpose_conf{CONF}/query.pkl"
    if kpts_q2_path.exists():
        kpts_q2 = load_kpts('msmt17', 'query')
        kpts_g2 = load_kpts('msmt17', 'gallery')
    else:
        print("  MSMT17 keypoints not cached — running ViTPose...")
        from reid_tk.pacm import PoseEstimator
        pose = PoseEstimator(backend='vitpose')
        all_paths = [str(s.image_path) for s in ds2.query + ds2.gallery]
        all_kpts2 = pose.batch(all_paths, conf_threshold=CONF, batch_size=32)
        kpts_q2 = all_kpts2[:len(ds2.query)]
        kpts_g2 = all_kpts2[len(ds2.query):]
        for split, kpts_s, samp_s in [('query', kpts_q2, ds2.query), ('gallery', kpts_g2, ds2.gallery)]:
            p = KPT_CACHE / f"msmt17_vitpose_conf{CONF}/{split}.pkl"
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, 'wb') as f:
                pickle.dump(kpts_s, f)

    msmt_samp = ds2.query + ds2.gallery
    msmt_kpts = kpts_q2 + kpts_g2

    msmt_fk, msmt_sw, msmt_sb, msmt_ci_lo, msmt_ci_hi, msmt_det = deep_fk_from_resnet(
        msmt_samp, msmt_kpts, spatial, 'msmt deep-bot'
    )

    print("\nMSMT17 deep F_k (BoT, cross-domain):")
    print(f"  {'Part':<6}  {'S_W':>7}  {'S_B':>7}  {'F_k':>7}  {'CI_lo':>7}  {'CI_hi':>7}")
    for i, nm in enumerate(KPT_NAMES):
        print(f"  {nm:<6}  {msmt_sw[i]:>7.3f}  {msmt_sb[i]:>7.3f}  {msmt_fk[i]:>7.3f}  "
              f"{msmt_ci_lo[i]:>7.3f}  {msmt_ci_hi[i]:>7.3f}")

    results['msmt17'] = {
        'fk': msmt_fk.tolist(), 'sw': msmt_sw.tolist(), 'sb': msmt_sb.tolist(),
        'ci_lo': msmt_ci_lo.tolist(), 'ci_hi': msmt_ci_hi.tolist(),
        'det_cnt': msmt_det.tolist(),
    }

    # ── Spearman correlations ─────────────────────────────────────────────────
    from scipy.stats import spearmanr
    print("\nSpearman ρ (n=6):")
    with open('/tmp/fk_detailed.json') as f:
        prev = json.load(f)
    color_fk = np.array(prev['clr_fk'])

    pairs = [
        ('color F_k (Mkt)', color_fk),
        ('BoT F_k (Mkt)',   mkt_fk),
        ('BoT F_k (Msmt)',  msmt_fk),
    ]
    print(f"  {'':22} " + "  ".join(f"{n:>20}" for n, _ in pairs))
    for ni, (na, va) in enumerate(pairs):
        row = f"  {na:<22} "
        for nj, (nb, vb) in enumerate(pairs):
            mask = np.isfinite(va) & np.isfinite(vb)
            rho, _ = spearmanr(va[mask], vb[mask]) if mask.sum() >= 3 else (float('nan'), None)
            row += f"  {rho:>20.3f}"
        print(row)

    results['kpt_names'] = KPT_NAMES
    out_path = Path('/tmp/fk_results_bot.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {out_path}")


if __name__ == '__main__':
    main()
