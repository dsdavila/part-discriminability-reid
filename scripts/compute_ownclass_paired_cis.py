"""
Paired real-minus-noise CIs on own-class rate row effects (Ch2 rev 141/145
deferred item, ruling 20 applied).

Loads per-draw arrays from e42_ownclass_grids.json and computes paired
differences real_arm - noise at each (rank, tercile). Reports median + CI.
"""
import json
import numpy as np
from pathlib import Path

RESULTS = Path('/data/papers/low_dimensional_reid/results')
IN  = RESULTS / 'e42_ownclass_grids.json'
OUT = RESULTS / 'e42_ownclass_paired_cis.json'

d = json.load(open(IN))
grids = ['diff_x_kocc', 'spread_x_kocc']
ranks = ['1', '5', '10', '20']

out = {'metric': 'paired_real_minus_noise_on_ownclass_rate_effects'}
for grid in grids:
    out[grid] = {}
    for arm in ('id', 'ood', 'randproj'):
        out[grid][arm] = {}
        for r in ranks:
            out[grid][arm][r] = {'row': {}, 'col': {}}
            for eff_kind in ('row_effects', 'col_effects'):
                key = 'row' if eff_kind == 'row_effects' else 'col'
                for s in range(3):
                    real = np.array(d[grid]['per_arm'][arm][r][eff_kind][str(s)]['raw'])
                    null = np.array(d[grid]['per_arm']['noise'][r][eff_kind][str(s)]['raw'])
                    diff = real - null
                    out[grid][arm][r][key][str(s)] = {
                        'median': float(np.median(diff)),
                        'lo': float(np.quantile(diff, 0.025)),
                        'hi': float(np.quantile(diff, 0.975)),
                        'mean': float(diff.mean()),
                    }

print("\n========== Paired real-minus-noise CIs on own-class RATE row effects ==========")
print("Ruling 20: these decide whether row effects are real or null-replicable.\n")
for grid in grids:
    print(f"\n=== {grid} ===")
    for arm in ('id', 'ood'):
        print(f"\n[{arm} - noise]")
        for r in ranks:
            print(f"  r={r}:")
            for s in range(3):
                c = out[grid][arm][r]['row'][str(s)]
                sig = "*" if c['lo'] * c['hi'] > 0 else " "
                print(f"    row s={s}{sig}: {c['median']:+.4f} [{c['lo']:+.4f}, {c['hi']:+.4f}]")

with open(OUT, 'w') as f:
    json.dump(out, f, indent=2)
print(f"\nSaved {OUT}")
