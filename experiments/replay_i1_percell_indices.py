"""
Replay the RNG traces of the three I-1 per-cell scripts to recover the
query and gallery subsample indices for each (arm, script) combination.

Writes /data/papers/low_dimensional_reid/results/e44_i1_percell_indices.json
with the structure:
    {script_name: {arm_name: {'q_idx': [...], 'g_idx': [...]}}}

Moves the three sidecar'd artefacts from approximating Ruling 43 to
satisfying it. See Channel 4 revs 616/618.

CRITICAL: this must track every rng call the real scripts make on the
shared rng before each dataset-subsample draw. That is:

posthoc_v2 and posthoc_v3 (simple trace):
    rng = default_rng(42)
    rng.choice(len(market.train), 3000)              # train pool
    rng.choice(len(non_junk_q_market), 500)          # A_prime q_idx
    rng.choice(len(non_junk_g_market), 5000)         # A_prime g_idx
    rng.choice(len(non_junk_q_msmt), 500)            # B_prime q_idx
    rng.choice(len(non_junk_g_msmt), 5000)           # B_prime g_idx

preview (v4-batched) adds for each arm after its extract and before the
next arm's extract:
    for f in [0.01, 0.05, 0.10, 0.25]:
        select_random_cells(N_q=500, f): N_q rng.choice(128, size=k) calls
        select_random_cells(N_g=5000, f): N_g rng.choice(128, size=k) calls
        # three conditions:
        # top_near: no rng
        # top_rand: apply_replacement with mode='random_patch' on q (500 imgs × k cells of rng.integers(0,3000)) + on g (5000 × k)
        # rand_near: no rng
where k = max(1, round(f * 128)).
"""
import json
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, '/workspace/reid-tk')
from reid_tk.datasets import get_dataset

RNG_SEED = 42
N_QUERY = 500
N_GALLERY = 5000
N_TRAIN_POOL = 3000
N_CELLS = 128  # 16 × 8
FRACS = [0.01, 0.05, 0.10, 0.25]
N_TRAIN_PATCHES = N_TRAIN_POOL  # train_patches_by_cell bucket size


def draw_arm_indices(rng, non_junk_q, non_junk_g):
    """Replay extract_arm's two rng.choice calls."""
    q_idx = rng.choice(len(non_junk_q), size=min(N_QUERY, len(non_junk_q)), replace=False)
    g_idx = rng.choice(len(non_junk_g), size=min(N_GALLERY, len(non_junk_g)), replace=False)
    return q_idx, g_idx


def consume_preview_inner_loop(rng, n_q, n_g):
    """Replay a preview arm's inner loop rng consumption exactly.

    Per f: N_q + N_g rng.choice calls (select_random_cells) +
    (N_q + N_g) * k rng.integers calls (apply_replacement top_rand).
    """
    for f in FRACS:
        k = max(1, int(round(f * N_CELLS)))
        # select_random_cells on q
        for _ in range(n_q):
            rng.choice(N_CELLS, size=k, replace=False)
        # select_random_cells on g
        for _ in range(n_g):
            rng.choice(N_CELLS, size=k, replace=False)
        # top_near: no rng
        # top_rand: apply_replacement with mode='random_patch' on q and g
        rng.integers(0, N_TRAIN_PATCHES, size=n_q * k)  # q
        rng.integers(0, N_TRAIN_PATCHES, size=n_g * k)  # g
        # rand_near: no rng


def replay_simple(market_q, market_g, msmt_q, msmt_g):
    """posthoc_v2 and posthoc_v3 share this trace."""
    rng = np.random.default_rng(RNG_SEED)
    # train pool (market.train = 12,936 > 3,000)
    rng.choice(12936, size=N_TRAIN_POOL, replace=False)
    # A_prime
    aq, ag = draw_arm_indices(rng, market_q, market_g)
    # B_prime
    bq, bg = draw_arm_indices(rng, msmt_q, msmt_g)
    return {'A_prime': {'q_idx': aq.tolist(), 'g_idx': ag.tolist()},
            'B_prime': {'q_idx': bq.tolist(), 'g_idx': bg.tolist()}}


def replay_preview(market_q, market_g, msmt_q, msmt_g):
    """preview has inner-loop rng consumption between arms."""
    rng = np.random.default_rng(RNG_SEED)
    # train pool
    rng.choice(12936, size=N_TRAIN_POOL, replace=False)
    # A_prime extract
    aq, ag = draw_arm_indices(rng, market_q, market_g)
    # A_prime inner loop
    consume_preview_inner_loop(rng, n_q=N_QUERY, n_g=N_GALLERY)
    # B_prime extract
    bq, bg = draw_arm_indices(rng, msmt_q, msmt_g)
    # B_prime inner loop would run next in the real script but we don't
    # need to consume it for recovery.
    return {'A_prime': {'q_idx': aq.tolist(), 'g_idx': ag.tolist()},
            'B_prime': {'q_idx': bq.tolist(), 'g_idx': bg.tolist()}}


def main():
    print('Loading datasets...', flush=True)
    market = get_dataset('market1501', sample=False)
    msmt = get_dataset('msmt17', sample=False)
    market_q = [s for s in market.query if not s.is_junk]
    market_g = [s for s in market.gallery if not s.is_junk]
    msmt_q = [s for s in msmt.query if not s.is_junk]
    msmt_g = [s for s in msmt.gallery if not s.is_junk]
    print(f'  market: q={len(market_q)} g={len(market_g)} train={len(market.train)}')
    print(f'  msmt:   q={len(msmt_q)} g={len(msmt_g)} train={len(msmt.train)}')

    print('\nReplaying posthoc_v2...', flush=True)
    posthoc_v2 = replay_simple(market_q, market_g, msmt_q, msmt_g)
    print(f"  A_prime q[:5]={posthoc_v2['A_prime']['q_idx'][:5]}")
    print(f"  B_prime q[:5]={posthoc_v2['B_prime']['q_idx'][:5]}")

    print('\nReplaying posthoc_v3 (same trace as v2)...', flush=True)
    posthoc_v3 = replay_simple(market_q, market_g, msmt_q, msmt_g)
    assert posthoc_v3 == posthoc_v2, "v2 and v3 share trace — should match exactly"
    print('  confirmed identical to v2')

    print('\nReplaying preview (v4-batched)...', flush=True)
    preview = replay_preview(market_q, market_g, msmt_q, msmt_g)
    print(f"  A_prime q[:5]={preview['A_prime']['q_idx'][:5]}")
    print(f"  B_prime q[:5]={preview['B_prime']['q_idx'][:5]}")

    # Sanity: A_prime should match between simple and preview (same draws before A_prime extract)
    assert preview['A_prime'] == posthoc_v2['A_prime'], "A_prime indices should match preview vs posthoc"
    print('  A_prime indices match preview vs posthoc (expected: same draws before A_prime)')
    # B_prime should DIFFER (preview has 308k rng calls in between)
    assert preview['B_prime'] != posthoc_v2['B_prime'], "B_prime should differ between preview and posthoc"
    print('  B_prime indices DIFFER preview vs posthoc (expected: inner-loop rng consumption)')

    out = {
        'replay_rev': 619,
        'rng_seed': RNG_SEED,
        'params': {'N_QUERY': N_QUERY, 'N_GALLERY': N_GALLERY,
                   'N_TRAIN_POOL': N_TRAIN_POOL, 'N_CELLS': N_CELLS,
                   'FRACS': FRACS},
        'datasets': {
            'market': {'non_junk_query': len(market_q),
                       'non_junk_gallery': len(market_g),
                       'train': len(market.train)},
            'msmt':   {'non_junk_query': len(msmt_q),
                       'non_junk_gallery': len(msmt_g),
                       'train': len(msmt.train)},
        },
        'scripts': {
            'compute_i1_percell_preview.py':      {'commit': '62bb359', 'arms': preview},
            'compute_i1_percell_posthoc_v2.py':   {'commit': '015a887', 'arms': posthoc_v2},
            'compute_i1_percell_posthoc_v3.py':   {'commit': 'c39e644', 'arms': posthoc_v3},
        },
        'note': 'Indices into the non-junk filtered query/gallery lists of each dataset. '
                'Ordering within non_junk is the dataset loader default (see reid_tk.datasets). '
                'Recovers Ruling 43 satisfaction for all three sidecar\'d artefacts.',
    }
    outp = Path('/data/papers/low_dimensional_reid/results/e44_i1_percell_indices.json')
    outp.write_text(json.dumps(out, indent=2))
    print(f'\nSaved {outp}', flush=True)


if __name__ == '__main__':
    main()
