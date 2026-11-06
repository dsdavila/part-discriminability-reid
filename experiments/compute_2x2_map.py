#!/usr/bin/env python3
"""
mAP 2×2 on one protocol: BoT-{Market,MSMT17} × {Market,MSMT17} eval.

All four cells use the same standard retrieval protocol:
  - Official query vs official gallery (from compute_bot_msmt_2x2.py caches)
  - Same-cam gallery excluded from relevant set (standard mAP)
  - No transforms — plain L2-normalized embeddings

Cache split: compute_bot_msmt_2x2.py stores [query_non_junk, gallery_non_junk]
in that order, with the following known boundaries:
  market_test.npz: [:3368] query, [3368:] gallery (13115 non-junk gallery)
  msmt17_test.npz: [:11655] query, [11655:] gallery (82132 non-junk gallery)

These are verified against dataset split counts before running.
"""

import json
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, '/workspace/reid-tk')
from reid_tk.datasets import get_dataset

CACHE_DIR = Path("/tmp/e12_feature_cache")
OUT_PATH  = Path("/data/papers/low_dimensional_reid/results/2x2_map_results.json")


def retrieval_map(q_emb, q_pids, q_cids, g_emb, g_pids, g_cids):
    """Standard mAP with same-cam exclusion; no junk IDs in our non-junk caches."""
    sims = q_emb @ g_emb.T
    aps, r1_hits = [], 0
    for i in range(len(q_pids)):
        qid, qcam = q_pids[i], q_cids[i]
        junk = (g_pids == qid) & (g_cids == qcam)
        order = np.argsort(-sims[i])
        order = order[~junk[order]]
        good = g_pids[order] == qid
        if not good.any():
            continue
        if good[0]:
            r1_hits += 1
        pos = np.where(good)[0]
        aps.append(float(np.mean((np.arange(1, good.sum() + 1) / (pos + 1)))))
    return float(np.mean(aps)) if aps else 0.0, r1_hits / len(q_pids)


def load_split(path, n_query):
    d = np.load(path)
    emb, pids, cids = d['emb'].astype(np.float32), d['pids'], d['cids']
    return (emb[:n_query], pids[:n_query], cids[:n_query],
            emb[n_query:], pids[n_query:], cids[n_query:])


def main():
    # Verify split boundaries against live dataset
    print("Verifying cache split boundaries...")
    mkt  = get_dataset('market1501', sample=False)
    msmt = get_dataset('msmt17',     sample=False)
    n_mkt_q  = sum(1 for s in mkt.query  if not s.is_junk)
    n_msmt_q = sum(1 for s in msmt.query if not s.is_junk)
    print(f"  Market query (non-junk): {n_mkt_q}")
    print(f"  MSMT17 query (non-junk): {n_msmt_q}")
    assert n_mkt_q  == 3368,  f"Unexpected Market query count: {n_mkt_q}"
    assert n_msmt_q == 11655, f"Unexpected MSMT17 query count: {n_msmt_q}"

    # Load all four cache slices
    print("Loading caches...")
    bm_mkt_q, bm_mkt_qpids, bm_mkt_qcids, \
    bm_mkt_g, bm_mkt_gpids, bm_mkt_gcids = load_split(
        CACHE_DIR / "market_test.npz", n_mkt_q)

    bm_msmt_q, bm_msmt_qpids, bm_msmt_qcids, \
    bm_msmt_g, bm_msmt_gpids, bm_msmt_gcids = load_split(
        CACHE_DIR / "msmt17_test.npz", n_msmt_q)

    bmsmt_mkt_q, bmsmt_mkt_qpids, bmsmt_mkt_qcids, \
    bmsmt_mkt_g, bmsmt_mkt_gpids, bmsmt_mkt_gcids = load_split(
        CACHE_DIR / "bot_msmt_seed42_market.npz", n_mkt_q)

    bmsmt_msmt_q, bmsmt_msmt_qpids, bmsmt_msmt_qcids, \
    bmsmt_msmt_g, bmsmt_msmt_gpids, bmsmt_msmt_gcids = load_split(
        CACHE_DIR / "bot_msmt_seed42_msmt.npz", n_msmt_q)

    cells = {}

    # Cell A: BoT-Market → Market
    print("Cell A: BoT-Market → Market ...")
    cells['A'] = dict(zip(
        ['map', 'r1'],
        retrieval_map(bm_mkt_q, bm_mkt_qpids, bm_mkt_qcids,
                      bm_mkt_g, bm_mkt_gpids, bm_mkt_gcids)
    ))
    cells['A'].update({'train': 'BoT-Market', 'eval': 'Market',
                       'n_query': n_mkt_q, 'n_gallery': len(bm_mkt_gpids)})
    print(f"  mAP={cells['A']['map']*100:.2f}%  R1={cells['A']['r1']*100:.2f}%")

    # Cell B: BoT-Market → MSMT17
    print("Cell B: BoT-Market → MSMT17 ...")
    cells['B'] = dict(zip(
        ['map', 'r1'],
        retrieval_map(bm_msmt_q, bm_msmt_qpids, bm_msmt_qcids,
                      bm_msmt_g, bm_msmt_gpids, bm_msmt_gcids)
    ))
    cells['B'].update({'train': 'BoT-Market', 'eval': 'MSMT17',
                       'n_query': n_msmt_q, 'n_gallery': len(bm_msmt_gpids)})
    print(f"  mAP={cells['B']['map']*100:.2f}%  R1={cells['B']['r1']*100:.2f}%")

    # Cell C: BoT-MSMT17 → Market
    print("Cell C: BoT-MSMT17 (seed42) → Market ...")
    cells['C'] = dict(zip(
        ['map', 'r1'],
        retrieval_map(bmsmt_mkt_q, bmsmt_mkt_qpids, bmsmt_mkt_qcids,
                      bmsmt_mkt_g, bmsmt_mkt_gpids, bmsmt_mkt_gcids)
    ))
    cells['C'].update({'train': 'BoT-MSMT17 (seed42)', 'eval': 'Market',
                       'n_query': n_mkt_q, 'n_gallery': len(bmsmt_mkt_gpids)})
    print(f"  mAP={cells['C']['map']*100:.2f}%  R1={cells['C']['r1']*100:.2f}%")

    # Cell D: BoT-MSMT17 → MSMT17
    print("Cell D: BoT-MSMT17 (seed42) → MSMT17 ...")
    cells['D'] = dict(zip(
        ['map', 'r1'],
        retrieval_map(bmsmt_msmt_q, bmsmt_msmt_qpids, bmsmt_msmt_qcids,
                      bmsmt_msmt_g, bmsmt_msmt_gpids, bmsmt_msmt_gcids)
    ))
    cells['D'].update({'train': 'BoT-MSMT17 (seed42)', 'eval': 'MSMT17',
                       'n_query': n_msmt_q, 'n_gallery': len(bmsmt_msmt_gpids)})
    print(f"  mAP={cells['D']['map']*100:.2f}%  R1={cells['D']['r1']*100:.2f}%")

    A, B, C, D = [cells[k]['map']*100 for k in 'ABCD']

    # 2×2 ANOVA decomposition
    main_eval   = ((A + C) - (B + D)) / 2   # Market vs MSMT17 eval
    main_train  = ((A + B) - (C + D)) / 2   # BoT-Market vs BoT-MSMT17
    interaction = ((A - B) - (C - D)) / 2   # domain alignment (same as (A+D)/2 - (B+C)/2)

    print("\n=== mAP 2×2 ===")
    print(f"{'':30}  Market eval     MSMT17 eval")
    print(f"  BoT-Market          {A:8.2f}%       {B:8.2f}%")
    print(f"  BoT-MSMT17 (s42)    {C:8.2f}%       {D:8.2f}%")
    print(f"\nANOVA decomposition:")
    print(f"  MAIN effect, eval dataset (Market vs MSMT17): {main_eval:+.2f}pp")
    print(f"  MAIN effect, training domain:                 {main_train:+.2f}pp")
    print(f"  INTERACTION (domain alignment):               {interaction:+.2f}pp")
    print(f"\n  Sanity check: A+B+C+D = {A+B+C+D:.2f}, mean = {(A+B+C+D)/4:.2f}")

    # Find-2 drop analysis (using correct commensurable cells)
    drop_BM = A - B   # BoT-Market: in-domain drop going cross-domain
    drop_BMS = D - C  # BoT-MSMT17: in-domain drop going cross-domain (D-C uses MSMT17 in-domain)
    print(f"\n  Finding 2 (domain-alignment asymmetry):")
    print(f"    BoT-Market in→out: {A:.2f}% → {B:.2f}%  drop={drop_BM:.2f}pp  retention={B/A:.3f}")
    print(f"    BoT-MSMT17 in→out: {D:.2f}% → {C:.2f}%  drop={drop_BMS:.2f}pp  retention={C/D:.3f}")

    result = {
        "protocol": "standard query/gallery split, same-cam exclusion, no transforms",
        "cache_split": {
            "market_query_count": n_mkt_q,
            "msmt17_query_count": n_msmt_q,
            "source": "compute_bot_msmt_2x2.py cache order: [query_non_junk, gallery_non_junk]"
        },
        "cells": {k: {"map": round(v['map'], 6), "r1": round(v['r1'], 6),
                      "map_pct": round(v['map']*100, 4),
                      "train": v['train'], "eval": v['eval'],
                      "n_query": v['n_query'], "n_gallery": v['n_gallery']}
                  for k, v in cells.items()},
        "anova": {
            "main_eval_dataset_pp": round(main_eval, 4),
            "main_training_domain_pp": round(main_train, 4),
            "interaction_domain_alignment_pp": round(interaction, 4),
            "note": ("Interaction = (A-B-C+D)/2 = (A+D)/2-(B+C)/2. "
                     "All cells on same protocol; commensurable.")
        },
        "finding2": {
            "BoT_Market_drop_pp": round(drop_BM, 4),
            "BoT_Market_retention": round(B/A, 4),
            "BoT_MSMT17_drop_pp": round(drop_BMS, 4),
            "BoT_MSMT17_retention": round(C/D, 4),
        }
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved → {OUT_PATH}")


if __name__ == "__main__":
    main()
