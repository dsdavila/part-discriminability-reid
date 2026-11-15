"""
Ruling-51 identity-clustered paired bootstrap for E-align's M1-vs-M2 claim.

The F-test in the original M1/M2/M3 analysis treated each query as an
independent observation, which understates variance when identities carry
multiple queries. This resamples identities (within each of the four
populations separately, preserving all of a drawn identity's queries),
refits M1 (logit(AP) ~ alignment) and M2 (logit(AP) ~ alignment +
population) on the SAME resampled draw each replicate, and reports the
interval on the R^2 difference -- the paired form Ruling 51 requires.

Spec: Channel 4 rev 638 §2 caveat / rev 639 "next per the agreed order".

Writes /data/papers/low_dimensional_reid/results/e44_e_align_m1m2_paired_bootstrap.json
"""
import json
from pathlib import Path
import numpy as np

RESULTS = Path('/data/papers/low_dimensional_reid/results')
RNG_SEED = 42
N_BOOT = 2000
POP_ORDER = ['A_prime', 'B_prime', 'D', 'C']
MODEL_FILES = {
    'A_prime': 'e44_e_align_BoT-Market.json', 'B_prime': 'e44_e_align_BoT-Market.json',
    'D': 'e44_e_align_BoT-MSMT.json', 'C': 'e44_e_align_BoT-MSMT.json',
}
ALIGNMENT_VARIABLE = 'meanCell_maxLayer'  # strongest overall performer, Channel 4 rev 638 §1


def logit_clip(p, eps=1e-3):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def fit_r2(X_design, y):
    beta, *_ = np.linalg.lstsq(X_design, y, rcond=None)
    yhat = X_design @ beta
    ss_res = ((y - yhat) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return beta, r2, ss_res


def build_design(pops_arr, X_arr, include_population=True):
    n = len(pops_arr)
    cols = [np.ones(n), X_arr]
    if include_population:
        for p in POP_ORDER[1:]:
            cols.append((pops_arr == p).astype(float))
    return np.column_stack(cols)


def main():
    pop_data = {}
    for pop in POP_ORDER:
        d = json.load(open(RESULTS / MODEL_FILES[pop]))
        p = d['populations'][pop]
        aps = np.array(p['per_query']['AP'])
        valid = np.array(p['per_query']['valid'])
        align = np.array(p['alignment'][ALIGNMENT_VARIABLE])
        pids = np.array(p['per_query']['pid'])
        v_aps, v_align, v_pids = aps[valid], align[valid], pids[valid]
        unique_pids = np.unique(v_pids)
        id_to_idx = {int(pid): np.where(v_pids == pid)[0] for pid in unique_pids}
        pop_data[pop] = {'aps': v_aps, 'align': v_align, 'unique_pids': unique_pids,
                          'id_to_idx': id_to_idx}
        print(f"{pop}: n_valid={len(v_aps)}  n_identities={len(unique_pids)}", flush=True)

    rng = np.random.default_rng(RNG_SEED + 900)
    r2_m1 = np.zeros(N_BOOT)
    r2_m2 = np.zeros(N_BOOT)
    diff = np.zeros(N_BOOT)

    for b in range(N_BOOT):
        pops_list, align_list, ap_list = [], [], []
        for pop in POP_ORDER:
            pd = pop_data[pop]
            sampled = rng.choice(pd['unique_pids'], size=len(pd['unique_pids']), replace=True)
            idx = np.concatenate([pd['id_to_idx'][int(pid)] for pid in sampled])
            pops_list.append(np.full(len(idx), pop))
            align_list.append(pd['align'][idx])
            ap_list.append(pd['aps'][idx])
        pops_b = np.concatenate(pops_list)
        align_b = np.concatenate(align_list)
        logit_b = logit_clip(np.concatenate(ap_list))

        _, r2_1, _ = fit_r2(build_design(pops_b, align_b, include_population=False), logit_b)
        _, r2_2, _ = fit_r2(build_design(pops_b, align_b, include_population=True), logit_b)
        r2_m1[b] = r2_1
        r2_m2[b] = r2_2
        diff[b] = r2_2 - r2_1

    out = {
        'rev': 639, 'n_boot': N_BOOT, 'rng_seed': RNG_SEED + 900,
        'alignment_variable': ALIGNMENT_VARIABLE,
        'M1_R2': {'mean': float(r2_m1.mean()), 'lo95': float(np.quantile(r2_m1, 0.025)),
                  'hi95': float(np.quantile(r2_m1, 0.975))},
        'M2_R2': {'mean': float(r2_m2.mean()), 'lo95': float(np.quantile(r2_m2, 0.025)),
                  'hi95': float(np.quantile(r2_m2, 0.975))},
        'M2_minus_M1_R2': {'mean': float(diff.mean()), 'lo95': float(np.quantile(diff, 0.025)),
                           'hi95': float(np.quantile(diff, 0.975))},
    }
    outp = RESULTS / 'e44_e_align_m1m2_paired_bootstrap.json'
    outp.write_text(json.dumps(out, indent=2))
    print(f"\nM1 R2: mean={r2_m1.mean():.4f}  95% CI [{np.quantile(r2_m1,0.025):.4f}, {np.quantile(r2_m1,0.975):.4f}]")
    print(f"M2 R2: mean={r2_m2.mean():.4f}  95% CI [{np.quantile(r2_m2,0.025):.4f}, {np.quantile(r2_m2,0.975):.4f}]")
    print(f"M2-M1 diff: mean={diff.mean():.4f}  95% CI [{np.quantile(diff,0.025):.4f}, {np.quantile(diff,0.975):.4f}]")
    print(f"saved {outp}")


if __name__ == '__main__':
    main()
