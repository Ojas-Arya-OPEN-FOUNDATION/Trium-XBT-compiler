"""
Regression test for ternary GEMM operand layout and the matmul fast path.

Two traps are pinned down here:

1. `gemm_ternary` wants A row-major but B TRANSPOSED and then
   WORD-PLANE-MAJOR (`bp[w*n + j]`). The two orderings coincide exactly
   when k <= 64 (kw == 1), so a wrongly-packed multi-word B returns
   status 0 and WRONG numbers. Only shapes with k > 64 and n > 1 can
   catch it, so every case below is run at both k <= 64 and k > 64.

2. `matmul` used to be the unvectorized reference regardless of input.
   It now routes to `gemm_ternary` when every element is a trit. The
   routing must be value-preserving, and must NOT fire for non-ternary
   data (which the ternary kernel cannot represent).

Usage: python3 test/gemm_layout.py <absolute path to libtrium_sdk.so>
"""

import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'ffi', 'python'))

import trium

failures = []
checks = 0


def check(ok, label):
    global checks
    checks += 1
    if not ok:
        failures.append(label)
        print('  FAIL ' + label)
    else:
        print('  ok   ' + label)


def reference_gemm(a, b, m, k, n):
    """Plain schoolbook C[m,n] = A[m,k] * B[k,n]."""
    return [sum(a[i * k + t] * b[t * n + j] for t in range(k))
            for i in range(m) for j in range(n)]


def naive_row_major_b(b, k, n):
    """The WRONG layout: bp[j*kw + w]. Correct only when kw == 1."""
    kw = (k + 63) // 64
    bp = [0] * (n * kw)
    bn = [0] * (n * kw)
    for j in range(n):
        for t in range(k):
            v = b[t * n + j]
            if v > 0:
                bp[j * kw + t // 64] |= 1 << (t % 64)
            elif v < 0:
                bn[j * kw + t // 64] |= 1 << (t % 64)
    return bp, bn


def main():
    if len(sys.argv) < 2:
        print('usage: gemm_layout.py <absolute path to libtrium_sdk.so>')
        return 2
    ctx = trium.Context(library=os.path.abspath(sys.argv[1]))
    print('gemm layout and matmul fast path')
    print('================================')
    print('  sdk api version: %s' % hex(ctx.lib._api_version))

    have_gemm = ctx.lib._api_version >= 0x10003
    if not have_gemm:
        print('  (gemm_ternary unavailable: only the reference path is checked)')

    random.seed(20261105)

    # Shapes deliberately straddling the kw == 1 / kw > 1 boundary. n > 1 is
    # required as well: with n == 1 the two B layouts are the same array.
    shapes = [(1, 1, 1), (2, 3, 4), (3, 64, 5), (4, 65, 3), (3, 128, 4),
              (2, 130, 3), (5, 200, 6), (2, 63, 4), (2, 64, 4)]

    print('\n  pack_matrix_b is correct at every k (the k>64 trap)')
    for (m, k, n) in shapes:
        a = [random.choice((-1, 0, 1)) for _ in range(m * k)]
        b = [random.choice((-1, 0, 1)) for _ in range(k * n)]
        want = reference_gemm(a, b, m, k, n)
        ap, an = trium.Context.pack_matrix_a(a, m, k)
        bp, bn = trium.Context.pack_matrix_b(b, k, n)
        if have_gemm:
            got = ctx.gemm_ternary(ap, an, bp, bn, m, n, k)
            check(got == want, 'pack helpers correct at m=%d k=%d n=%d' % (m, k, n))
        else:
            check(True, 'pack helpers built at m=%d k=%d n=%d' % (m, k, n))

    if have_gemm:
        print('\n  the wrong B layout really is wrong for k>64')
        # Negative control: proves these shapes have teeth. If the naive
        # packing agreed everywhere, this whole test would be vacuous.
        teeth = 0
        for (m, k, n) in shapes:
            if k <= 64:
                continue
            a = [random.choice((-1, 0, 1)) for _ in range(m * k)]
            b = [random.choice((-1, 0, 1)) for _ in range(k * n)]
            want = reference_gemm(a, b, m, k, n)
            ap, an = trium.Context.pack_matrix_a(a, m, k)
            nbp, nbn = naive_row_major_b(b, k, n)
            if ctx.gemm_ternary(ap, an, nbp, nbn, m, n, k) != want:
                teeth += 1
        check(teeth > 0, 'naive row-major B is detected as wrong (negative control)')

    print('\n  matmul agrees with the reference on ternary input')
    for (m, k, n) in shapes:
        a = [random.choice((-1, 0, 1)) for _ in range(m * k)]
        b = [random.choice((-1, 0, 1)) for _ in range(k * n)]
        want = reference_gemm(a, b, m, k, n)
        fast = ctx.matmul(a, b, m, k, n)
        slow = ctx.matmul(a, b, m, k, n, force_reference=True)
        check(fast == want, 'matmul fast path m=%d k=%d n=%d' % (m, k, n))
        check(slow == want, 'matmul reference m=%d k=%d n=%d' % (m, k, n))
        check(fast == slow, 'fast path matches reference m=%d k=%d n=%d' % (m, k, n))

    print('\n  non-ternary input must not take the ternary path')
    a = [2, 3, 4, 5]
    b = [6, 7, 8, 9]
    check(ctx.matmul(a, b, 2, 2, 2) == reference_gemm(a, b, 2, 2, 2),
          'matmul on ordinary int64 still correct')
    check(trium.Context.is_ternary([2, 3]) is False, 'is_ternary rejects 2')
    check(trium.Context.is_ternary([-1, 0, 1]) is True, 'is_ternary accepts trits')
    check(trium.Context.is_ternary([-1, 0, 1, 5]) is False,
          'is_ternary rejects a single out-of-range element')

    print('\n  all-zero and all-negative corners')
    for (m, k, n) in [(2, 128, 3), (3, 200, 4)]:
        zeros_a = [0] * (m * k)
        zeros_b = [0] * (k * n)
        check(ctx.matmul(zeros_a, zeros_b, m, k, n) == [0] * (m * n),
              'zero matrices m=%d k=%d n=%d' % (m, k, n))
        negs_a = [-1] * (m * k)
        negs_b = [1] * (k * n)
        check(ctx.matmul(negs_a, negs_b, m, k, n) == reference_gemm(negs_a, negs_b, m, k, n),
              'all-negative A m=%d k=%d n=%d' % (m, k, n))

    print('\n%d checks, %d failure(s)' % (checks, len(failures)))
    if failures:
        print('GEMM LAYOUT: FAILED')
        return 1
    print('GEMM LAYOUT: PASSED')
    return 0


if __name__ == '__main__':
    sys.exit(main())
