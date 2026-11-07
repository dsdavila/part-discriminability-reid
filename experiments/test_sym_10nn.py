"""
Unit test for symmetric-10NN fraction (Ch2 rev 183).

Function has had two degenerate-output bugs. Needs a known-answer test
plus two invariants:
  (i)  Fully symmetric k-NN graph -> output exactly 1.0.
  (ii) Function must not return a constant across arms differing in
       reciprocity.

Brute-force reference: for each (i, j) with j in topk(i), count 1 if
also i in topk(j). Average over all N * k ordered pairs.
"""
import numpy as np


def sym_10nn_brute(top_nb, k):
    """Brute-force reference. Returns the fraction of ordered pairs
    (i, j=top_nb[i, r]) with i also in top_nb[j]. Average over N * k pairs.
    """
    N = top_nb.shape[0]
    sym = 0
    for i in range(N):
        for r in range(k):
            j = int(top_nb[i, r])
            # Is i in top_nb[j]?
            if i in top_nb[j].tolist():
                sym += 1
    return sym / (N * k)


def sym_10nn_fast(top_nb, k):
    """Candidate implementation using set representation per row."""
    N = top_nb.shape[0]
    topk_sets = [set(int(x) for x in top_nb[j]) for j in range(N)]
    sym = 0
    for i in range(N):
        for r in range(k):
            j = int(top_nb[i, r])
            if i in topk_sets[j]:
                sym += 1
    return sym / (N * k)


def make_random_topk(N, k, rng):
    """Random top-k lists (not symmetric in general)."""
    top_nb = np.zeros((N, k), dtype=np.int32)
    for i in range(N):
        pool = list(range(N))
        pool.remove(i)
        rng.shuffle(pool)
        top_nb[i] = np.array(pool[:k], dtype=np.int32)
    return top_nb


def make_symmetric_topk(N, k, rng):
    """Construct a fully symmetric k-NN graph by: for each i, pick a
    random k-set S_i; set top_nb[i] = S_i; then force each j in S_i to
    include i in S_j, pruning to k.

    Simpler: build a random k-regular graph via pairing method, then
    use the k neighbours of each node as its top-k.
    """
    # Each node is paired with the same k partners; we construct it
    # greedy: fix top_nb for each i to be a chosen set, with the
    # constraint that i appears in top_nb[j] iff j appears in top_nb[i].
    # A simple realisation: structured cycles so each item's top-k is
    # (i+1) mod N, (i+2) mod N, ..., (i+k/2) mod N and (i-1), (i-2), ...
    # This is a circulant k-NN graph and is fully symmetric.
    assert k % 2 == 0, "Use even k for symmetric circulant"
    top_nb = np.zeros((N, k), dtype=np.int32)
    half = k // 2
    for i in range(N):
        offsets = list(range(1, half + 1)) + [-o for o in range(1, half + 1)]
        top_nb[i] = np.array([(i + o) % N for o in offsets], dtype=np.int32)
    return top_nb


def make_asymmetric_topk(N, k, rng):
    """All items point to the same k central items (hub-like). None of
    the central items has others in its own top-k except by chance.
    Highly asymmetric; reciprocity should be very low."""
    hubs = list(range(k))
    top_nb = np.zeros((N, k), dtype=np.int32)
    for i in range(N):
        if i in hubs:
            # Pick k random non-self items
            pool = [x for x in range(N) if x != i]
            rng.shuffle(pool)
            top_nb[i] = np.array(pool[:k], dtype=np.int32)
        else:
            top_nb[i] = np.array(hubs, dtype=np.int32)
    return top_nb


def main():
    rng = np.random.default_rng(42)
    N, k = 50, 10

    # Test 1: known-answer small random gallery
    print("=== Test 1: small random gallery, brute-force vs fast ===")
    for seed in [0, 1, 2]:
        r = np.random.default_rng(seed)
        top_nb = make_random_topk(N, k, r)
        bf = sym_10nn_brute(top_nb, k)
        fa = sym_10nn_fast(top_nb, k)
        print(f"  seed={seed}: brute={bf:.6f}  fast={fa:.6f}  diff={bf-fa:+.2e}")
        assert abs(bf - fa) < 1e-9, f"FAIL: brute {bf} vs fast {fa}"
    print("  PASS")

    # Test 2: fully symmetric graph
    print("\n=== Test 2: fully symmetric circulant k-NN graph ===")
    top_nb = make_symmetric_topk(100, 10, rng)
    bf = sym_10nn_brute(top_nb, k)
    fa = sym_10nn_fast(top_nb, k)
    print(f"  brute={bf:.6f}  fast={fa:.6f}")
    assert abs(bf - 1.0) < 1e-9, f"FAIL: symmetric should give 1.0, brute={bf}"
    assert abs(fa - 1.0) < 1e-9, f"FAIL: symmetric should give 1.0, fast={fa}"
    print("  PASS")

    # Test 3: hub-based asymmetric graph (reciprocity very low)
    print("\n=== Test 3: hub-based asymmetric graph (low reciprocity) ===")
    top_nb = make_asymmetric_topk(100, 10, rng)
    bf = sym_10nn_brute(top_nb, k)
    fa = sym_10nn_fast(top_nb, k)
    print(f"  brute={bf:.6f}  fast={fa:.6f}")
    assert abs(bf - fa) < 1e-9
    assert bf < 0.3, f"FAIL: hub graph should have low reciprocity, got {bf}"
    print("  PASS (low reciprocity {:.3f})".format(bf))

    # Test 4: cross-arm non-constancy
    print("\n=== Test 4: function must vary across arms with different reciprocity ===")
    r1 = np.random.default_rng(100)
    r2 = np.random.default_rng(200)
    arm_a = make_random_topk(80, k, r1)
    arm_b = make_symmetric_topk(80, k, rng)
    a_val = sym_10nn_fast(arm_a, k)
    b_val = sym_10nn_fast(arm_b, k)
    print(f"  random arm: {a_val:.6f}  symmetric arm: {b_val:.6f}")
    assert a_val < 0.5, f"Random arm should have moderate reciprocity, got {a_val}"
    assert abs(b_val - 1.0) < 1e-9
    assert abs(a_val - b_val) > 0.1, "Function returns suspiciously constant across arms"
    print("  PASS")

    print("\n=== ALL TESTS PASS ===")


if __name__ == '__main__':
    main()
