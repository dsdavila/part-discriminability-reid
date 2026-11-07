"""
Unit test for symmetric-10NN (Ch2 rev 183/186).

Validates the SHARED implementation in sym_10nn.py that production
scripts import. Prior version validated a function that existed only
in this test file; that was theatre (rev 186). Now the test exercises
the actual production code.

Tests:
  (1) Brute-force reference vs symmetric_knn_fraction on small random
      galleries, three seeds.
  (2) Fully symmetric circulant k-NN graph -> output exactly 1.0.
  (3) Hub-based asymmetric graph -> low reciprocity.
  (4) Non-constancy across arms differing in reciprocity.
  (5) per_item_reciprocity mean equals symmetric_knn_fraction on full set.
"""
import numpy as np
from sym_10nn import symmetric_knn_fraction, per_item_reciprocity


def sym_10nn_brute(top_nb, k):
    """Brute-force reference."""
    N = top_nb.shape[0]
    sym = 0
    for i in range(N):
        for r in range(k):
            j = int(top_nb[i, r])
            if i in top_nb[j, :k].tolist():
                sym += 1
    return sym / (N * k)


def make_random_topk(N, k, rng):
    top_nb = np.zeros((N, k), dtype=np.int32)
    for i in range(N):
        pool = list(range(N))
        pool.remove(i)
        rng.shuffle(pool)
        top_nb[i] = np.array(pool[:k], dtype=np.int32)
    return top_nb


def make_symmetric_topk(N, k, rng):
    """Circulant k-NN graph. Fully symmetric."""
    assert k % 2 == 0
    top_nb = np.zeros((N, k), dtype=np.int32)
    half = k // 2
    for i in range(N):
        offsets = list(range(1, half + 1)) + [-o for o in range(1, half + 1)]
        top_nb[i] = np.array([(i + o) % N for o in offsets], dtype=np.int32)
    return top_nb


def make_asymmetric_topk(N, k, rng):
    hubs = list(range(k))
    top_nb = np.zeros((N, k), dtype=np.int32)
    for i in range(N):
        if i in hubs:
            pool = [x for x in range(N) if x != i]
            rng.shuffle(pool)
            top_nb[i] = np.array(pool[:k], dtype=np.int32)
        else:
            top_nb[i] = np.array(hubs, dtype=np.int32)
    return top_nb


def main():
    rng = np.random.default_rng(42)
    N, k = 50, 10

    print("=== Test 1: brute-force vs symmetric_knn_fraction ===")
    for seed in [0, 1, 2]:
        r = np.random.default_rng(seed)
        top_nb = make_random_topk(N, k, r)
        bf = sym_10nn_brute(top_nb, k)
        fa = symmetric_knn_fraction(top_nb, k)
        print(f"  seed={seed}: brute={bf:.6f}  prod={fa:.6f}  diff={bf-fa:+.2e}")
        assert abs(bf - fa) < 1e-9
    print("  PASS")

    print("\n=== Test 2: symmetric circulant -> exactly 1.0 ===")
    top_nb = make_symmetric_topk(100, 10, rng)
    fa = symmetric_knn_fraction(top_nb, k)
    print(f"  prod={fa:.6f}")
    assert abs(fa - 1.0) < 1e-9
    print("  PASS")

    print("\n=== Test 3: hub-based asymmetric -> low reciprocity ===")
    top_nb = make_asymmetric_topk(100, 10, rng)
    fa = symmetric_knn_fraction(top_nb, k)
    print(f"  prod={fa:.6f}")
    assert fa < 0.3
    print(f"  PASS (low reciprocity {fa:.3f})")

    print("\n=== Test 4: non-constancy across arms ===")
    r1 = np.random.default_rng(100)
    arm_a = make_random_topk(80, k, r1)
    arm_b = make_symmetric_topk(80, k, rng)
    a_val = symmetric_knn_fraction(arm_a, k)
    b_val = symmetric_knn_fraction(arm_b, k)
    print(f"  random arm: {a_val:.6f}  symmetric arm: {b_val:.6f}")
    assert abs(a_val - b_val) > 0.1
    print("  PASS")

    print("\n=== Test 5: per_item_reciprocity mean == symmetric_knn_fraction ===")
    for seed in [0, 10, 20]:
        r = np.random.default_rng(seed)
        top_nb = make_random_topk(60, k, r)
        full = symmetric_knn_fraction(top_nb, k)
        per_item = per_item_reciprocity(top_nb, k)
        mean_per_item = float(per_item.mean())
        print(f"  seed={seed}: full={full:.6f}  mean_per_item={mean_per_item:.6f}  "
              f"diff={full-mean_per_item:+.2e}")
        assert abs(full - mean_per_item) < 1e-9
    print("  PASS")

    print("\n=== ALL TESTS PASS ===")


if __name__ == '__main__':
    main()
