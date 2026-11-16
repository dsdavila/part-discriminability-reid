import json
import numpy as np
from scipy.stats import spearmanr

RESULTS = '/data/papers/low_dimensional_reid/results'
LAYER_NAMES = ['layer1', 'layer2', 'layer3', 'layer4']
POP_ORDER = ['A_prime', 'B_prime', 'D', 'C']
MODEL_FILES = {
    'A_prime': 'e44_e_align_BoT-Market.json', 'B_prime': 'e44_e_align_BoT-Market.json',
    'D': 'e44_e_align_BoT-MSMT.json', 'C': 'e44_e_align_BoT-MSMT.json',
}
PCTILES = [75, 90, 95]

for pop in POP_ORDER:
    d = json.load(open(f'{RESULTS}/{MODEL_FILES[pop]}'))
    p = d['populations'][pop]
    aps = np.array(p['per_query']['AP'])
    valid = np.array(p['per_query']['valid'])
    z_by_layer = np.stack([np.array(p['per_layer_raw']['meanCell'][L]) for L in LAYER_NAMES], axis=1)  # (N, 4)
    z_valid = z_by_layer[valid]
    aps_valid = aps[valid]
    N = z_valid.shape[0]

    print(f"\n=== {pop} (n_valid={N}) ===")
    argmax_layer = z_valid.argmax(axis=1)  # 0-indexed
    print(f"  argmax-layer distribution: " +
          ", ".join(f"L{i+1}={int((argmax_layer==i).sum())}" for i in range(4)))

    for pct in PCTILES:
        # threshold is the pct-th percentile of ALL this population's z-values pooled
        # across layers and samples (consistent cross-layer threshold, as z already
        # LOO-training-normalised per layer)
        thresh = np.percentile(z_valid.flatten(), pct)
        onset = np.full(N, -1, dtype=int)  # -1 = never crosses
        for i in range(N):
            crossed = np.where(z_valid[i] > thresh)[0]
            if len(crossed) > 0:
                onset[i] = crossed[0]
        never = (onset == -1).sum()
        dist_str = ", ".join(f"L{i+1}={int((onset==i).sum())}" for i in range(4))
        print(f"  [pct={pct}] thresh_z={thresh:.3f}  onset dist: {dist_str}  never-crosses={never}")

        valid_onset = onset >= 0
        if valid_onset.sum() > 5:
            rho, pval = spearmanr(onset[valid_onset], aps_valid[valid_onset])
            print(f"    Spearman(onset_layer, AP): rho={rho:+.4f}  p={pval:.3e}  n={valid_onset.sum()}")

        # onset vs argmax gap (cascade quantified), at this threshold
        gap = argmax_layer[valid_onset] - onset[valid_onset]
        if valid_onset.sum() > 0:
            print(f"    onset vs argmax gap (argmax - onset), mean={gap.mean():.3f}  "
                  f"frac(argmax>onset)={float((gap>0).mean()):.3f}  frac(argmax==onset)={float((gap==0).mean()):.3f}")
