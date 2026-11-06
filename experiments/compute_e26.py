"""
E26: layer-wise geometry localisation.
Hook layer1 (base[4]), layer2 (base[5]), layer3 (base[6]), layer4 (base[7]).
GAP + L2-normalise at each layer. Compute ||mu||, eff_rank(S_B), eff_rank(S_W), Fisher F.
"""
import sys; sys.path.insert(0, '/workspace/reid-tk')
import numpy as np
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as T
from torch.utils.data import DataLoader

OUT   = Path('/data/papers/low_dimensional_reid/results')
PTWD  = Path('/data/papers/low_dimensional_reid/pretrained')
MKT_ROOT  = Path('/data/datasets/reid/Market-1501-v15.09.15')
MSMT_ROOT = Path('/data/datasets/reid/MSMT17_V1')
BATCH = 64
NUM_WORKERS = 0
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}", flush=True)

from scripts.train_bag_of_tricks import BagOfTricksModel
from reid_tk.datasets import get_dataset

MODEL_SPECS = [
    ('BoT_Market', PTWD / 'bot_market1501_final.pth',                     751),
    ('BoT_MSMT17', PTWD / 'bot_resnet50_msmt17_seed42_final.pth',        1041),
]

# base[4]=layer1, base[5]=layer2, base[6]=layer3, base[7]=layer4
LAYER_MAP  = {'layer1': 4, 'layer2': 5, 'layer3': 6, 'layer4': 7}
LAYER_DIMS = {'layer1': 256, 'layer2': 512, 'layer3': 1024, 'layer4': 2048}
LAYER_NAMES = ['layer1', 'layer2', 'layer3', 'layer4']

TRANSFORM = T.Compose([
    T.Resize((256, 128)), T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

class GalDS(torch.utils.data.Dataset):
    def __init__(self, samps):
        self.s = samps
    def __len__(self): return len(self.s)
    def __getitem__(self, i):
        img = Image.open(self.s[i].image_path).convert('RGB')
        return TRANSFORM(img), self.s[i].person_id

def load_model(ckpt_path, num_classes):
    model = BagOfTricksModel(num_classes=num_classes, last_stride=1, backbone='resnet50').to(device)
    raw = torch.load(ckpt_path, map_location=device)
    # Remap old key scheme: "0.*" → "base.*", "3.*" → "bottleneck.*"
    if '0.0.weight' in raw:
        new_sd = {}
        for k, v in raw.items():
            if k.startswith('0.'):
                new_sd['base.' + k[2:]] = v
            elif k.startswith('3.'):
                new_sd['bottleneck.' + k[2:]] = v
            else:
                new_sd[k] = v
        raw = new_sd
    model.load_state_dict(raw, strict=False)
    model.eval()
    return model

def extract_layer_feats(model, samps):
    loader = DataLoader(GalDS(samps), batch_size=BATCH, num_workers=NUM_WORKERS, pin_memory=True)
    gap = nn.AdaptiveAvgPool2d(1)
    captured = {}
    feats_all = {ln: [] for ln in LAYER_NAMES}
    pids_all  = []
    hooks = []

    def make_hook(ln):
        def h(mod, inp, out): captured[ln] = out.detach()
        return h

    for ln in LAYER_NAMES:
        hooks.append(model.base[LAYER_MAP[ln]].register_forward_hook(make_hook(ln)))

    model.eval()
    with torch.no_grad():
        for imgs, pids in tqdm(loader, leave=False):
            model.base(imgs.to(device))   # just the backbone forward
            for ln in LAYER_NAMES:
                v = gap(captured[ln]).squeeze(-1).squeeze(-1)
                v = F.normalize(v, dim=1)
                feats_all[ln].append(v.cpu().float().numpy())
            pids_all.extend(pids.numpy().tolist())

    for h in hooks: h.remove()
    return {ln: np.concatenate(feats_all[ln], 0) for ln in LAYER_NAMES}, np.array(pids_all)

def eff_rank_fn(X):
    Xc = X - X.mean(0, keepdims=True)
    if Xc.shape[0] < 2: return 1.0
    _, s, _ = np.linalg.svd(Xc, full_matrices=False)
    p = s**2; p /= p.sum() + 1e-12; p = p[p > 1e-12]
    return float(np.exp(-np.sum(p * np.log(p))))

def compute_metrics(feats, pids):
    D = feats.shape[1]
    unique_p = np.unique(pids)
    cents = np.vstack([feats[pids==p].mean(0) for p in unique_p])
    norms = np.linalg.norm(cents, axis=1, keepdims=True)
    cents /= np.maximum(norms, 1e-12)
    sw_list = []
    for i, p in enumerate(unique_p):
        sw_list.append(feats[pids==p] - cents[i])
    sw = np.vstack(sw_list)
    grand = cents.mean(0, keepdims=True)
    tr_sb = float(np.sum((cents - grand)**2))
    tr_sw = float(np.sum(sw**2))
    mu_norm = float(np.linalg.norm(feats.mean(0)))
    er_sb = eff_rank_fn(cents)
    er_sw = eff_rank_fn(sw)
    return {
        'mu_norm':          mu_norm,
        'eff_rank_SB':      er_sb,
        'eff_rank_SW':      er_sw,
        'eff_rank_SB_frac': er_sb / D,
        'eff_rank_SW_frac': er_sw / D,
        'fisher_F':         float(tr_sb / (tr_sw + 1e-12)),
        'dim': D, 'n_ids': int(len(unique_p)), 'n_samples': int(len(pids)),
    }

results = {}
gallery_specs = [
    ('Market', 'market1501', MKT_ROOT),
    ('MSMT17', 'msmt17',     MSMT_ROOT),
]

for model_name, ckpt_path, num_classes in MODEL_SPECS:
    print(f"\n{'='*60}\nModel: {model_name}", flush=True)
    if not ckpt_path.exists():
        print("  ERROR: checkpoint not found"); continue
    model = load_model(ckpt_path, num_classes)

    for gal_name, ds_name, ds_root in gallery_specs:
        print(f"\n  Gallery: {gal_name}", flush=True)
        try:
            ds = get_dataset(ds_name, ds_root)
        except Exception as e:
            print(f"  ERROR: {e}"); continue
        gallery = [s for s in ds.gallery if s.person_id != -1]
        print(f"  {len(gallery)} images", flush=True)
        layer_feats, pids = extract_layer_feats(model, gallery)
        key = f"{model_name}_x_{gal_name}"
        layer_metrics = {}
        for ln in LAYER_NAMES:
            m = compute_metrics(layer_feats[ln], pids)
            layer_metrics[ln] = m
            D = LAYER_DIMS[ln]
            print(f"    {ln} (D={D:4d}): ||mu||={m['mu_norm']:.4f}  "
                  f"eff_rank(SB)={m['eff_rank_SB']:.1f}/{D} ({m['eff_rank_SB_frac']:.3f})  "
                  f"eff_rank(SW)={m['eff_rank_SW']:.1f}/{D} ({m['eff_rank_SW_frac']:.3f})  "
                  f"Fisher={m['fisher_F']:.4f}", flush=True)
        results[key] = {'model': model_name, 'gallery': gal_name, 'layers': layer_metrics}

    del model; torch.cuda.empty_cache()

out_path = OUT / 'e26_layerwise_geometry.json'
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved → {out_path}")
