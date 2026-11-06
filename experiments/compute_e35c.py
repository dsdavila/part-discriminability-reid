"""
E35 part (c) — self-preference table: share(q→n) / availability(n), at k=1/5/10,
on MSMT (D vs B' vs noise-D) and Market (A' vs C vs noise-A'), against matched-
accuracy noise null.

Reproduces claude-chat's hand arithmetic (Channel rev 241):
  EASY 9.51× → 6.17×; SHARED 2.62× → 1.91×; OOD-ONLY 1.21× → 1.04× (D→B' k=10 MSMT)

Also reports off-diagonal self-preferences so we can read:
  - OOD-ONLY ← SHARED: 0.50× → 1.09× (D→B')
  - EASY ← SHARED:    0.12× → 0.35× (D→B')

All three arms side by side per cell / k.
"""
import json
from pathlib import Path

RESULTS = Path('/data/papers/low_dimensional_reid/results')
CLASSES = ['EASY', 'ID-ONLY', 'SHARED', 'OOD-ONLY']


def self_preference_table(table):
    """Given a per-arm k-table with 'rows' and 'proportional_null_pct',
    return dict {(q, n): self_pref} where self_pref = share(q→n) / avail(n).
    """
    avail = table['proportional_null_pct']  # {class: pct of items}
    rows = table['rows']
    out = {}
    for row in rows:
        q, n = row['q_class'], row['n_class']
        share = row['share_of_q_slots_pct']
        a = avail.get(n, 0.0)
        pref = (share / a) if a > 0 else float('nan')
        out[(q, n)] = pref
    return out


def format_table(d, title):
    print(f"\n{title}")
    header = f"{'q\\n':<11}  " + "  ".join(f"{c:<10}" for c in CLASSES)
    print(header)
    for q in CLASSES:
        row = f"{q:<11}  " + "  ".join(f"{d.get((q, c), float('nan')):>7.2f}×   " for c in CLASSES)
        print(row)


def print_arm_triple(k_table_dict, k):
    print(f"\n========== k={k} ==========")
    for arm in ('id', 'ood', 'noise'):
        tbl = self_preference_table(k_table_dict[arm])
        format_table(tbl, f"[{arm}] self-preference (= column-share / availability)")


def main():
    data = json.load(open(RESULTS / 'e35_class_flux_kvar.json'))
    out = {}
    for ds_name in ('MSMT', 'Market'):
        print(f"\n######### {ds_name} #########")
        ds = data[ds_name]
        out[ds_name] = {}
        for k in ('1', '5', '10'):
            print(f"\n~~~~~~~~ k={k} ~~~~~~~~")
            k_arms = ds['k_tables'][k]
            per_k = {}
            for arm in ('id', 'ood', 'noise'):
                tbl = self_preference_table(k_arms[arm])
                per_k[arm] = {f"{q}|{n}": tbl[(q, n)] for (q, n) in tbl}
                format_table(tbl, f"[{arm}] k={k}")
            # diagonal-only summary
            print(f"\n  Diagonal (self): id | ood | noise")
            for c in CLASSES:
                i = per_k['id'][f"{c}|{c}"]
                o = per_k['ood'][f"{c}|{c}"]
                n = per_k['noise'][f"{c}|{c}"]
                real_vs_noise = o - n
                print(f"    {c:<10} {i:7.2f}× → {o:7.2f}× (noise {n:7.2f}×)  real−noise={real_vs_noise:+6.2f}")
            # key off-diagonals called out in rev 241
            print(f"  Off-diagonal on SHARED: id | ood | noise")
            for q in ('EASY', 'OOD-ONLY'):
                i = per_k['id'][f"{q}|SHARED"]
                o = per_k['ood'][f"{q}|SHARED"]
                n = per_k['noise'][f"{q}|SHARED"]
                print(f"    {q}←SHARED  {i:7.2f}× → {o:7.2f}× (noise {n:7.2f}×)  real−noise={o-n:+6.2f}")
            out[ds_name][k] = per_k

    out_path = RESULTS / 'e35c_self_preference.json'
    with open(out_path, 'w') as f: json.dump(out, f, indent=2)
    print(f"\nSaved → {out_path}")


if __name__ == '__main__':
    main()
