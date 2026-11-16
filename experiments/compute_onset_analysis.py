"""
Onset analysis, with censoring check (Daniel via claude-chat, Channel 4
rev 655 section 4 / rev 677 item 14).

onset = lowest layer where z-score first crosses a swept percentile
threshold (distinct from argmax = layer with maximum z-score). A large
fraction of samples never cross at the higher percentiles -- the
Spearman(onset, AP) already excludes these non-crossers (filtered via
valid_onset = onset >= 0 before spearmanr), which is itself a censoring
choice. This adds the sensitivity check: what does rho look like if
non-crossers are NOT excluded but instead assigned onset=4 (treating
"never crossed within the observed depth" as equivalent to needing the
full depth or more -- the conservative convention), and does the
counterintuitive negative-rho direction (D, C at pct=75) survive it.

Writes /data/papers/low_dimensional_reid/results/e44_onset_analysis.json
"""
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
NEVER_CROSSES_VALUE = 4  # 0-indexed "beyond layer4", for the inclusive sensitivity check

out = {'rev': 677, 'pctiles': PCTILES, 'never_crosses_assigned_value': NEVER_CROSSES_VALUE,
       'populations': {}}

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

    pop_out = {'n_valid': N,
               'argmax_distribution': {f'L{i+1}': int((argmax_layer == i).sum()) for i in range(4)},
               'by_threshold': {}}

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
        never = int((onset == -1).sum())
        never_frac = never / N
        dist_str = ", ".join(f"L{i+1}={int((onset==i).sum())}" for i in range(4))
        print(f"  [pct={pct}] thresh_z={thresh:.3f}  onset dist: {dist_str}  "
              f"never-crosses={never} ({100*never_frac:.1f}%)", flush=True)

        valid_onset = onset >= 0
        rho_excl, p_excl, n_excl = float('nan'), float('nan'), 0
        if valid_onset.sum() > 5:
            rho_excl, p_excl = spearmanr(onset[valid_onset], aps_valid[valid_onset])
            n_excl = int(valid_onset.sum())
            print(f"    EXCLUDING non-crossers: rho={rho_excl:+.4f}  p={p_excl:.3e}  n={n_excl}", flush=True)

        # Censoring sensitivity check: assign non-crossers onset=NEVER_CROSSES_VALUE,
        # include everyone, recompute.
        onset_incl = onset.copy()
        onset_incl[onset_incl == -1] = NEVER_CROSSES_VALUE
        rho_incl, p_incl = spearmanr(onset_incl, aps_valid)
        print(f"    INCLUDING non-crossers (assigned L{NEVER_CROSSES_VALUE+1}): "
              f"rho={rho_incl:+.4f}  p={p_incl:.3e}  n={N}", flush=True)
        same_direction = (rho_excl == rho_excl) and (np.sign(rho_excl) == np.sign(rho_incl))
        print(f"    same sign (excl vs incl)? {same_direction}", flush=True)

        gap = argmax_layer[valid_onset] - onset[valid_onset]
        gap_mean = float(gap.mean()) if valid_onset.sum() > 0 else float('nan')
        frac_argmax_gt = float((gap > 0).mean()) if valid_onset.sum() > 0 else float('nan')
        frac_argmax_eq = float((gap == 0).mean()) if valid_onset.sum() > 0 else float('nan')
        if valid_onset.sum() > 0:
            print(f"    onset vs argmax gap (argmax - onset), mean={gap_mean:.3f}  "
                  f"frac(argmax>onset)={frac_argmax_gt:.3f}  frac(argmax==onset)={frac_argmax_eq:.3f}", flush=True)

        pop_out['by_threshold'][str(pct)] = {
            'thresh_z': float(thresh),
            'onset_distribution': {f'L{i+1}': int((onset == i).sum()) for i in range(4)},
            'never_crosses': never, 'never_crosses_fraction': never_frac,
            'spearman_excluding_noncrossers': {'rho': float(rho_excl), 'p': float(p_excl), 'n': n_excl},
            'spearman_including_noncrossers': {'rho': float(rho_incl), 'p': float(p_incl), 'n': N},
            'same_sign_excl_vs_incl': bool(same_direction) if same_direction == same_direction else None,
            'onset_vs_argmax_gap_mean': gap_mean,
            'frac_argmax_gt_onset': frac_argmax_gt, 'frac_argmax_eq_onset': frac_argmax_eq,
        }

    out['populations'][pop] = pop_out

outp = f'{RESULTS}/e44_onset_analysis.json'
with open(outp, 'w') as f:
    json.dump(out, f, indent=2)
print(f"\nsaved {outp}", flush=True)
