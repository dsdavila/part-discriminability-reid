"""E2b: L2-normalize per-slot features before computing S_B / S_W.

Loads the SAME BoT spatial feature maps as compute_fk_bot.py but normalises each
slot's feature vector to unit length before distance computation.  This removes the
per-slot feature-norm artifact so S_W and S_B are comparable across parts.

Questions answered:
  1. Does 1/S_W (reliability) rank differently from F_k (discriminability) after
     normalization? (raw 1/S_W gave Wr>El>Hi>Kn>An>Sh, inverse of F_k — possibly
     a scale artifact)
  2. Does S_B still vary ~1.8× (wardrobe diversity) or does normalization collapse it?

On L2-normalized unit vectors the distance is in [0, √2].

Outputs /tmp/fk_results_bot_e2b.json.
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

# 6 paired COCO body-part slots (tied L/R → one slot each)
BODY_PAIRS = [(5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]
KPT_NAMES  = ['Sh', 'El', 'Wr', 'Hi', 'Kn', 'An']
K = len(BODY_PAIRS)

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


def l2_normalize(v):
    n = np.linalg.norm(v)
    return v / (n + 1e-12)


def compute_stats(by_id, n_pairs=4, n_bootstrap=N_BOOTSTRAP):
    """Returns fk, sw, sb, ci_lo, ci_hi, det_cnt — identical to compute_fk_bot.py
    except features are expected to be already L2-normalized."""
    rng = random.Random(SEED)
    pids = list(by_id.keys())

    def _one_pass(pid_list):
        wi_sum  = np.zeros(K); wi_cnt  = np.zeros(K, dtype=int)
        btw_sum = np.zeros(K); btw_cnt = np.zeros(K, dtype=int)
        for pid in pid_list:
            feats = by_id[pid]
            if len(feats) < 2: continue
            pairs = [(a, b) for i, a in enumerate(feats) for b in feats[i+1:]]
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
    for _ in range(N_BOOTSTRAP):
        sampled = np_rng.choice(pids, size=len(pids), replace=True).tolist()
        boot_sub = {p: by_id[p] for p in sampled if p in by_id}
        b_fk, _, _ = _one_pass(list(boot_sub.keys()))
        boot_fk.append(b_fk)
    boot_fk = np.array(boot_fk)
    ci_lo = np.nanpercentile(boot_fk, 2.5,  axis=0)
    ci_hi = np.nanpercentile(boot_fk, 97.5, axis=0)

    return fk, sw, sb, ci_lo, ci_hi, det_cnt


@torch.no_grad()
def deep_fk_normalized(samples, kpts_all, spatial_model, label=''):
    """Same pipeline as compute_fk_bot.py but L2-normalizes each slot feature."""
    by_id = defaultdict(list)
    imgs, meta = [], []

    for i, (s, kpts) in enumerate(tqdm(
        zip(samples, kpts_all), total=len(samples), desc=label or 'e2b'
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
        feat_map = spatial_model(batch).cpu().numpy()   # (B, 2048, H, W)

        for b, (pid, det_b, W0, H0) in enumerate(meta):
            fm_flat = feat_map[b].reshape(2048, -1).T   # (128, 2048)
            sf = {}
            for kidx, kpt in det_b:
                nx, ny = float(kpt[0]) / W0, float(kpt[1]) / H0
                gw = gauss_weights(nx, ny)
                fv = (fm_flat * gw[:, None]).sum(0)     # (2048,) raw
                fv = l2_normalize(fv)                   # unit sphere — KEY CHANGE
                for sl, (l, r) in enumerate(BODY_PAIRS):
                    if kidx in (l, r):
                        sf[sl] = (sf[sl] + fv) / 2 if sl in sf else fv
            # Re-normalize after averaging L+R (average of unit vectors ≠ unit vector)
            sf = {s: l2_normalize(v) for s, v in sf.items()}
            if sf: by_id[pid].append(sf)

        imgs, meta = [], []

    return compute_stats(dict(by_id))


def load_kpts(dataset_tag, split):
    p = KPT_CACHE / f"{dataset_tag}_vitpose_conf{CONF}" / f"{split}.pkl"
    with open(p, 'rb') as f:
        return pickle.load(f)


def main():
    print(f"Device: {DEVICE}  |  E2b: L2-normalized features")
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

        fk, sw, sb, ci_lo, ci_hi, det = deep_fk_normalized(
            all_samp, all_kpts, spatial, f'{tag} e2b'
        )

        print(f"\n  {'Part':<6}  {'S_W':>7}  {'S_B':>7}  {'F_k':>7}  {'1/S_W':>7}  CI")
        sw_rank  = np.argsort(sw)          # ascending → low S_W = high reliability
        fk_rank  = np.argsort(fk)[::-1]   # descending → high F_k first
        for i, nm in enumerate(KPT_NAMES):
            print(f"  {nm:<6}  {sw[i]:>7.4f}  {sb[i]:>7.4f}  {fk[i]:>7.4f}  "
                  f"{1/sw[i]:>7.4f}  [{ci_lo[i]:.4f}, {ci_hi[i]:.4f}]")

        print(f"\n  F_k ranking:   {' > '.join(KPT_NAMES[i] for i in fk_rank)}")
        print(f"  1/S_W ranking: {' > '.join(KPT_NAMES[i] for i in sw_rank)}")

        results[tag] = {
            'fk': fk.tolist(), 'sw': sw.tolist(), 'sb': sb.tolist(),
            'ci_lo': ci_lo.tolist(), 'ci_hi': ci_hi.tolist(),
            'det_cnt': det.tolist(),
        }

    # Compare rankings
    mkt  = np.array(results['market1501']['fk'])
    mkt_sw = np.array(results['market1501']['sw'])
    msmt = np.array(results['msmt17']['fk'])
    msmt_sw = np.array(results['msmt17']['sw'])

    rho_fk_rel, p_fk_rel = spearmanr(mkt, 1.0 / mkt_sw)
    rho_fk_msmt, p_fk_msmt = spearmanr(mkt, msmt)
    rho_sw_msmt, p_sw_msmt = spearmanr(mkt_sw, msmt_sw)

    print(f"\n== Ranking comparisons (n=6, normalized features) ==")
    print(f"  ρ(F_k vs 1/S_W, Market)           = {rho_fk_rel:.3f}  p={p_fk_rel:.4f}")
    print(f"  ρ(F_k Market vs F_k MSMT)         = {rho_fk_msmt:.3f}  p={p_fk_msmt:.4f}")
    print(f"  ρ(S_W Market vs S_W MSMT)         = {rho_sw_msmt:.3f}  p={p_sw_msmt:.4f}")

    # Load unnormalized results for comparison
    with open('/tmp/fk_results_bot.json') as f:
        raw = json.load(f)
    raw_fk  = np.array(raw['market']['fk'])
    raw_sw  = np.array(raw['market']['sw'])
    rho_raw_norm_fk, _ = spearmanr(raw_fk, mkt)
    rho_raw_norm_sw, _ = spearmanr(raw_sw, mkt_sw)
    print(f"\n== Comparison against unnormalized (raw) features ==")
    print(f"  ρ(raw F_k, norm F_k)  = {rho_raw_norm_fk:.3f}")
    print(f"  ρ(raw S_W, norm S_W)  = {rho_raw_norm_sw:.3f}")

    results['spearman'] = {
        'fk_vs_reliability_mkt': {'rho': rho_fk_rel, 'p': p_fk_rel},
        'fk_mkt_vs_msmt':        {'rho': rho_fk_msmt, 'p': p_fk_msmt},
        'sw_mkt_vs_msmt':        {'rho': rho_sw_msmt, 'p': p_sw_msmt},
    }

    out = Path('/tmp/fk_results_bot_e2b.json')
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {out}")


if __name__ == '__main__':
    main()
