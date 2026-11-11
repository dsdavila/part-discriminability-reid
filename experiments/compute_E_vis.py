"""
E-vis: stratified contact sheet of OOD retrieval failures (rev 381).

For each OOD arm (B' = Market model on MSMT test; C = MSMT model on
Market test):
  - Rank all queries by first-true-positive rank bucket:
    {1-10, 11-100, 101-1000, >1000} (mAP protocol: same-pid-same-cam excluded)
  - Sample 8 queries per bucket (seeded)
  - For each sample: show query + top-20 retrieved (mAP protocol) +
    first-TP appended if outside top-20 + top-5 excluded same-pid-same-cam
  - Per-chip annotation: pid, cid, rank, cosine, same-cam/diff-cam to query,
    true/wrong marker
  - Rank-distribution histogram per arm

Output: HTML page per arm with images referenced by relative path.
Images served from a staging dir (symlinks or copies).
"""
import json
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, '/workspace/reid-tk')
from reid_tk.datasets import get_dataset

CACHE_DIR = Path('/tmp/e12_feature_cache')
RESULTS   = Path('/data/papers/low_dimensional_reid/results')
OUT_DIR   = Path('/data/papers/low_dimensional_reid/evis')
OUT_DIR.mkdir(parents=True, exist_ok=True)

RNG_SEED = 42
BUCKETS = [(1, 10), (11, 100), (101, 1000), (1001, 10**9)]
N_PER_BUCKET = 8
TOP_K_RETRIEVED = 20
TOP_K_EXCLUDED = 5

ARMS = {
    'Bprime': {'cache_file': 'msmt17_test_noREA.npz', 'n_q': 11659,
               'dataset_name': 'msmt17', 'label': "B′: Market model → MSMT test"},
    'C':      {'cache_file': 'bot_msmt_seed42_market.npz', 'n_q': 3368,
               'dataset_name': 'market1501', 'label': "C: MSMT model → Market test"},
}


def l2(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(n, 1e-12)


def load_arm(info):
    d = np.load(CACHE_DIR / info['cache_file'])
    emb = d['emb'].astype(np.float32); pids = d['pids']; cids = d['cids']
    n_q = info['n_q']
    q_emb = l2(emb[:n_q]); q_pids = pids[:n_q]; q_cids = cids[:n_q]
    g_emb = l2(emb[n_q:]); g_pids = pids[n_q:]; g_cids = cids[n_q:]
    return q_emb, q_pids, q_cids, g_emb, g_pids, g_cids


def rank_queries(q_emb, q_pids, q_cids, g_emb, g_pids, g_cids):
    """For each query, return (first_true_positive_rank, ap). Both in
    mAP protocol (same-pid-same-cam excluded).
    Rank = 0 means first true positive is rank-1 (0-indexed below first TP).
    Return rank1 (1-indexed) for interpretability. If no true positive, rank=inf.
    """
    n_q = len(q_pids)
    first_tp_rank = np.full(n_q, np.iinfo(np.int32).max // 2, dtype=np.int64)
    aps = np.full(n_q, 0.0, dtype=np.float64)
    for i in range(n_q):
        if i % 2000 == 0:
            print(f"    query {i}/{n_q}", flush=True)
        qp = q_pids[i]; qc = q_cids[i]
        sims = q_emb[i] @ g_emb.T
        junk = (g_pids == qp) & (g_cids == qc)
        sims_masked = sims.copy()
        sims_masked[junk] = -np.inf
        order = np.argsort(-sims_masked)
        # Remove junk from ranked list
        keep = ~junk[order]
        order_clean = order[keep]
        is_pos = g_pids[order_clean] == qp
        if is_pos.any():
            first_tp_rank[i] = int(np.argmax(is_pos)) + 1  # 1-indexed
            pos_positions = np.where(is_pos)[0]
            precisions = [(j + 1) / (p + 1) for j, p in enumerate(pos_positions)]
            aps[i] = float(np.mean(precisions))
    return first_tp_rank, aps


def sample_per_bucket(first_tp_rank, buckets, n_per, rng_seed):
    """Return dict bucket_name -> list of query indices sampled."""
    rng = np.random.default_rng(rng_seed)
    out = {}
    for lo, hi in buckets:
        name = f"{lo}-{hi}" if hi < 10**9 else f">{lo-1}"
        mask = (first_tp_rank >= lo) & (first_tp_rank <= hi)
        candidates = np.where(mask)[0]
        if len(candidates) == 0:
            out[name] = []
            continue
        k = min(n_per, len(candidates))
        sel = rng.choice(candidates, size=k, replace=False)
        out[name] = sorted(sel.tolist())
    return out


def build_dataset_map(dataset_name):
    """Return the ReIDDataset with non-junk-only query + gallery (matches
    the feature cache indexing: cache was built from non-junk items)."""
    print(f"  loading {dataset_name} dataset for image paths (filtering junk)...", flush=True)
    d = get_dataset(dataset_name, sample=False)
    # Filter junk from query and gallery, preserving order
    d_query_nojunk = [s for s in d.query if not s.is_junk]
    d_gallery_nojunk = [s for s in d.gallery if not s.is_junk]
    print(f"    query: {len(d.query)} -> {len(d_query_nojunk)} non-junk", flush=True)
    print(f"    gallery: {len(d.gallery)} -> {len(d_gallery_nojunk)} non-junk", flush=True)
    # Return a lightweight wrapper
    class _DS:
        pass
    ds = _DS()
    ds.query = d_query_nojunk
    ds.gallery = d_gallery_nojunk
    return ds


def get_image_path(sample):
    return str(sample.image_path)


def collect_row(q_idx, q_pid, q_cid, q_ap, q_tp_rank, q_emb, q_pids, q_cids,
                g_emb, g_pids, g_cids, dataset):
    """For a query, return dict with query info and lists of retrieved +
    excluded items, each with (sample, rank_mAP, sim, is_pos, same_cam).
    """
    qp = q_pid; qc = q_cid
    sims = q_emb[q_idx] @ g_emb.T
    junk = (g_pids == qp) & (g_cids == qc)
    order_full = np.argsort(-sims)
    # mAP-protocol ranking (junk excluded)
    keep = ~junk[order_full]
    order_clean = order_full[keep]
    # Top-K retrieved under mAP protocol
    top_k = order_clean[:TOP_K_RETRIEVED]
    retrieved = []
    for rank_mAP, g_i in enumerate(top_k, start=1):
        g_i = int(g_i)
        is_pos = bool(g_pids[g_i] == qp)
        same_cam = bool(g_cids[g_i] == qc)
        retrieved.append({
            'gallery_idx': g_i,
            'rank_mAP': rank_mAP,
            'sim': float(sims[g_i]),
            'pid': int(g_pids[g_i]), 'cid': int(g_cids[g_i]),
            'is_pos': is_pos,
            'same_cam_as_query': same_cam,
            'image_path': get_image_path(dataset.gallery[g_i]),
        })
    # First TP outside top-K if any
    is_pos_in_clean = g_pids[order_clean] == qp
    outside_tp = None
    if is_pos_in_clean.any():
        tp_positions = np.where(is_pos_in_clean)[0] + 1  # 1-indexed
        if tp_positions[0] > TOP_K_RETRIEVED:
            g_i = int(order_clean[tp_positions[0] - 1])
            outside_tp = {
                'gallery_idx': g_i, 'rank_mAP': int(tp_positions[0]),
                'sim': float(sims[g_i]),
                'pid': int(g_pids[g_i]), 'cid': int(g_cids[g_i]),
                'is_pos': True,
                'same_cam_as_query': bool(g_cids[g_i] == qc),
                'image_path': get_image_path(dataset.gallery[g_i]),
            }
    # Top-K excluded (same-pid-same-cam) by raw sim
    junk_idx = np.where(junk)[0]
    if len(junk_idx) > 0:
        junk_sims = sims[junk_idx]
        junk_order = junk_idx[np.argsort(-junk_sims)]
        top_k_excluded = junk_order[:TOP_K_EXCLUDED]
    else:
        top_k_excluded = np.array([], dtype=np.int64)
    excluded = []
    for rank_raw, g_i in enumerate(top_k_excluded, start=1):
        g_i = int(g_i)
        excluded.append({
            'gallery_idx': g_i,
            'raw_rank_within_junk': rank_raw,
            'sim': float(sims[g_i]),
            'pid': int(g_pids[g_i]), 'cid': int(g_cids[g_i]),
            'is_pos': True,  # same pid as query
            'same_cam_as_query': True,
            'image_path': get_image_path(dataset.gallery[g_i]),
        })
    return {
        'query_idx': int(q_idx), 'q_pid': int(qp), 'q_cid': int(qc),
        'q_ap': float(q_ap), 'q_tp_rank': int(q_tp_rank),
        'q_image_path': get_image_path(dataset.query[q_idx]),
        'retrieved': retrieved,
        'outside_top20_tp': outside_tp,
        'excluded_top5_same_pid_same_cam': excluded,
    }


def render_html(arm_name, info, per_bucket, rows, rank_hist, chip_w=80, chip_h=160):
    """Write HTML report. Images referenced by absolute path (file://)."""
    html = []
    html.append(f"<!DOCTYPE html><html><head><meta charset='utf-8'>")
    html.append(f"<title>E-vis {arm_name}</title>")
    html.append("<style>")
    html.append("body { font-family: monospace; background: #fafafa; }")
    html.append("h1,h2,h3 { margin: 0.3em 0; }")
    html.append(".bucket { margin: 10px 0; padding: 6px; background: #fff; border: 1px solid #ccc; }")
    html.append(".row { display: flex; align-items: flex-start; margin: 4px 0; padding: 4px; border-bottom: 1px solid #eee; overflow-x: auto; }")
    html.append(".chip { margin-right: 4px; text-align: center; font-size: 10px; min-width: " + str(chip_w) + "px; }")
    html.append(f".chip img {{ width: {chip_w}px; height: {chip_h}px; object-fit: cover; display: block; }}")
    html.append(".query { border: 3px solid blue; }")
    html.append(".pos { border: 3px solid green; }")
    html.append(".neg { border: 3px solid red; }")
    html.append(".outside { border: 3px dashed orange; }")
    html.append(".excluded { border: 3px dotted gray; }")
    html.append(".same-cam { background: #ffe; }")
    html.append(".header { font-weight: bold; color: #444; width: 180px; padding: 2px 8px; font-size: 11px; }")
    html.append(".annot { line-height: 1.1; }")
    html.append("</style></head><body>")
    html.append(f"<h1>E-vis — {info['label']}</h1>")
    html.append("<p><b>Legend:</b> blue=query, green=correct person, red=wrong person, orange-dashed=first true positive outside top-20, gray-dotted=excluded by mAP protocol (same-pid-same-cam). Yellow background = same camera as query.</p>")

    # Rank-distribution histogram
    html.append("<h2>Rank-distribution histogram (all queries, first TP rank)</h2>")
    html.append("<table style='border-collapse: collapse;'>")
    html.append("<tr><th>Bucket</th><th>Count</th><th>%</th><th>Visualisation</th></tr>")
    tot = sum(v for v in rank_hist.values())
    for bname, cnt in rank_hist.items():
        pct = cnt / max(tot, 1) * 100
        bar = "█" * int(pct / 2)
        html.append(f"<tr><td style='padding: 2px 8px;'>{bname}</td><td style='padding: 2px 8px;'>{cnt}</td><td style='padding: 2px 8px;'>{pct:.1f}%</td><td style='font-family: monospace;'>{bar}</td></tr>")
    html.append("</table>")
    html.append(f"<p>Total queries with any true positive: {tot}. 8 queries sampled per bucket (seed {RNG_SEED}).</p>")

    for bucket_name in per_bucket:
        if not per_bucket[bucket_name]:
            continue
        html.append(f"<div class='bucket'><h2>Bucket: first-TP rank in {bucket_name}</h2>")
        for q_idx in per_bucket[bucket_name]:
            row = rows[q_idx]
            same_cam_count = sum(1 for r in row['retrieved'] if r['same_cam_as_query'])
            html.append("<div class='row'>")
            html.append(f"<div class='header annot'>q_pid={row['q_pid']}<br>q_cid={row['q_cid']}<br>first-TP rank={row['q_tp_rank']}<br>AP={row['q_ap']:.4f}<br>top-20 same-cam: {same_cam_count}</div>")
            # Query chip
            html.append(f"<div class='chip'><img src='file://{row['q_image_path']}' class='query'><div class='annot'>QUERY<br>pid={row['q_pid']} c={row['q_cid']}</div></div>")
            # Top-K retrieved
            for item in row['retrieved']:
                cls = 'pos' if item['is_pos'] else 'neg'
                bg = ' same-cam' if item['same_cam_as_query'] else ''
                html.append(f"<div class='chip{bg}'><img src='file://{item['image_path']}' class='{cls}'><div class='annot'>#{item['rank_mAP']}<br>pid={item['pid']} c={item['cid']}<br>sim={item['sim']:.3f}<br>{'SAME-CAM' if item['same_cam_as_query'] else ''}</div></div>")
            # First-TP outside top-20
            if row['outside_top20_tp'] is not None:
                item = row['outside_top20_tp']
                bg = ' same-cam' if item['same_cam_as_query'] else ''
                html.append(f"<div class='chip{bg}'><img src='file://{item['image_path']}' class='outside'><div class='annot'>first TP<br>rank {item['rank_mAP']}<br>pid={item['pid']} c={item['cid']}<br>sim={item['sim']:.3f}</div></div>")
            # Excluded
            if row['excluded_top5_same_pid_same_cam']:
                html.append("<div class='chip annot' style='min-width: 60px;'>|<br>EXCLUDED<br>by protocol<br>(same pid,<br>same cam):<br>coarse-feature<br>intactness test</div>")
                for item in row['excluded_top5_same_pid_same_cam']:
                    html.append(f"<div class='chip same-cam'><img src='file://{item['image_path']}' class='excluded'><div class='annot'>exc #{item['raw_rank_within_junk']}<br>pid={item['pid']} c={item['cid']}<br>sim={item['sim']:.3f}</div></div>")
            html.append("</div>")
        html.append("</div>")
    html.append("</body></html>")
    out_path = OUT_DIR / f'evis_{arm_name}.html'
    out_path.write_text('\n'.join(html))
    print(f"  Saved {out_path}", flush=True)
    return out_path


def main():
    out = {}
    for arm_name, info in ARMS.items():
        print(f"\n### E-vis {arm_name}: {info['label']} ###", flush=True)
        q_emb, q_pids, q_cids, g_emb, g_pids, g_cids = load_arm(info)
        print(f"  n_q={len(q_pids)}  n_g={len(g_pids)}", flush=True)
        print(f"  ranking queries (mAP protocol)...", flush=True)
        first_tp, aps = rank_queries(q_emb, q_pids, q_cids, g_emb, g_pids, g_cids)
        valid = first_tp < 10**8
        print(f"  queries with at least one TP: {valid.sum()}/{len(first_tp)}", flush=True)
        # Rank histogram
        rank_hist = {}
        for lo, hi in BUCKETS:
            name = f"{lo}-{hi}" if hi < 10**9 else f">{lo-1}"
            rank_hist[name] = int(((first_tp >= lo) & (first_tp <= hi) & valid).sum())
        for name, cnt in rank_hist.items():
            print(f"    {name}: {cnt}", flush=True)
        # Sample per bucket
        per_bucket = sample_per_bucket(first_tp, BUCKETS, N_PER_BUCKET, RNG_SEED)
        for name, qs in per_bucket.items():
            print(f"    sampled bucket {name}: {len(qs)} queries, indices {qs[:3]}...", flush=True)
        # Load dataset for image paths
        dataset = build_dataset_map(info['dataset_name'])
        # Confirm length matches
        if len(dataset.query) != len(q_pids) or len(dataset.gallery) != len(g_pids):
            print(f"  WARNING: dataset len mismatch. query {len(dataset.query)} vs cache {len(q_pids)}; gallery {len(dataset.gallery)} vs cache {len(g_pids)}", flush=True)
        # Build rows for sampled queries
        rows = {}
        for bname, qs in per_bucket.items():
            for q_idx in qs:
                rows[q_idx] = collect_row(q_idx, q_pids[q_idx], q_cids[q_idx],
                                          aps[q_idx], first_tp[q_idx],
                                          q_emb, q_pids, q_cids,
                                          g_emb, g_pids, g_cids, dataset)
        # Render HTML
        out_path = render_html(arm_name, info, per_bucket, rows, rank_hist)
        out[arm_name] = {
            'label': info['label'], 'n_q': int(len(q_pids)),
            'rank_histogram': rank_hist,
            'sampled_queries_per_bucket': per_bucket,
            'html_path': str(out_path),
        }
    # Save manifest
    manifest_path = RESULTS / 'e43_E_vis_manifest.json'
    with open(manifest_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved manifest {manifest_path}")


if __name__ == '__main__':
    main()
