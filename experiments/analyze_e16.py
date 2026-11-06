"""E16 analysis: Spearman correlations, LOO predictor fits, C1 vs D3."""
import json, itertools
import numpy as np
from scipy import stats

with open('/data/papers/low_dimensional_reid/results/e16_training_results.json') as f:
    data = json.load(f)

arms = ['D1','D2','D3','D4','D5','D6','C1','C2']
mAP = np.array([data[a]['market_map'] for a in arms])
R1  = np.array([data[a]['market_r1'] for a in arms])

def g(a, k): return data[a]['geometry_market'][k]

geo = {
    'mu_norm':        np.array([g(a,'mu_norm')        for a in arms]),
    'std_cone':       np.array([g(a,'std_cone')        for a in arms]),
    'eff_rank_SB':    np.array([g(a,'eff_rank_SB')     for a in arms]),
    'eff_rank_SW':    np.array([g(a,'eff_rank_SW')     for a in arms]),
    'tr_SB':          np.array([g(a,'tr_SB')           for a in arms]),
    'tr_SW':          np.array([g(a,'tr_SW')           for a in arms]),
    'fisher_F':       np.array([g(a,'fisher_F')        for a in arms]),
    'med_pair_angle': np.array([g(a,'med_pair_angle')  for a in arms]),
    'cap_radius_deg': np.array([g(a,'cap_radius_deg')  for a in arms]),
    'retvar_32':      np.array([g(a,'retained_var')['32']  for a in arms]),
    'retvar_64':      np.array([g(a,'retained_var')['64']  for a in arms]),
    'retvar_128':     np.array([g(a,'retained_var')['128'] for a in arms]),
    'retvar_256':     np.array([g(a,'retained_var')['256'] for a in arms]),
}
n_cams = np.array([len(data[a]['cameras']) for a in arms])
n_ids  = np.array([data[a]['n_ids']        for a in arms])
geo['n_cams'] = n_cams
geo['n_ids']  = n_ids

# ── Spearman correlations with mAP ──────────────────────────────────────────
print("=== Spearman ρ vs mAP (n=8) ===")
print(f"{'Predictor':<20} {'rho':>7} {'p':>8}")
results_sp = {}
for name, x in sorted(geo.items(), key=lambda kv: -abs(stats.spearmanr(mAP, kv[1]).statistic)):
    r, p = stats.spearmanr(mAP, x)
    results_sp[name] = (r, p)
    flag = '*' if p < 0.05 else (' ?' if p < 0.10 else '')
    print(f"  {name:<20} {r:+.3f}   {p:.3f} {flag}")

# ── Predictor fit: all pairs of hold-out arms ────────────────────────────────
print("\n=== LOO-2 predictor fits (leave out each pair of arms) ===")
print("Using Fisher_F as candidate predictor (highest ρ from geometry)")

best_pred = 'fisher_F'
x_all = geo[best_pred]

all_pairs = list(itertools.combinations(range(8), 2))
errs = []
for ho_idx in all_pairs:
    tr_idx = [i for i in range(8) if i not in ho_idx]
    x_tr = x_all[tr_idx].reshape(-1,1)
    y_tr = mAP[tr_idx]
    # OLS with intercept
    Xb = np.hstack([x_tr, np.ones((len(x_tr),1))])
    coeffs, _, _, _ = np.linalg.lstsq(Xb, y_tr, rcond=None)
    for i in ho_idx:
        y_hat = coeffs[0]*x_all[i] + coeffs[1]
        errs.append(abs(y_hat - mAP[i]))

print(f"  {best_pred}: MAE on held-out arm = {np.mean(errs):.2f}% ({np.min(errs):.2f}–{np.max(errs):.2f}%)")

# Try n_cams as predictor
x_all2 = n_cams.astype(float)
errs2 = []
for ho_idx in all_pairs:
    tr_idx = [i for i in range(8) if i not in ho_idx]
    x_tr = x_all2[tr_idx].reshape(-1,1)
    y_tr = mAP[tr_idx]
    Xb = np.hstack([x_tr, np.ones((len(x_tr),1))])
    coeffs, _, _, _ = np.linalg.lstsq(Xb, y_tr, rcond=None)
    for i in ho_idx:
        y_hat = coeffs[0]*x_all2[i] + coeffs[1]
        errs2.append(abs(y_hat - mAP[i]))
print(f"  n_cams:   MAE on held-out arm = {np.mean(errs2):.2f}% ({np.min(errs2):.2f}–{np.max(errs2):.2f}%)")

# ── C1 vs D3 comparison ──────────────────────────────────────────────────────
print("\n=== C1 vs D3 (both 6 cameras, different subsets) ===")
for name in ['market_map','market_r1']:
    v3 = data['D3'][name]
    vc = data['C1'][name]
    print(f"  {name}: D3={v3:.2f}%  C1={vc:.2f}%  diff={v3-vc:+.2f}%")
print("  Geometry differences (D3 - C1):")
for k in ['mu_norm','std_cone','eff_rank_SB','eff_rank_SW','fisher_F','med_pair_angle','cap_radius_deg']:
    v3 = g('D3',k)
    vc = g('C1',k)
    print(f"    {k:<20} D3={v3:.4f}  C1={vc:.4f}  diff={v3-vc:+.4f}")

# ── Save results ─────────────────────────────────────────────────────────────
out = {
    'spearman': {k: {'rho': float(v[0]), 'p': float(v[1])} for k,v in results_sp.items()},
    'loo2_mae': {
        best_pred: float(np.mean(errs)),
        'n_cams': float(np.mean(errs2)),
    },
    'c1_vs_d3': {
        'D3_map': float(data['D3']['market_map']),
        'C1_map': float(data['C1']['market_map']),
        'D3_r1':  float(data['D3']['market_r1']),
        'C1_r1':  float(data['C1']['market_r1']),
    }
}
with open('/data/papers/low_dimensional_reid/results/e16_analysis.json','w') as f:
    json.dump(out, f, indent=2)
print("\nSaved → /data/papers/low_dimensional_reid/results/e16_analysis.json")
