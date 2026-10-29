#!/usr/bin/env python3
"""Independent Python-bigint / rational oracle for the C++ integer core."""
from fractions import Fraction
import random
import subprocess
import sys

lo, hi = -(2**63), 2**63-1
bound = (3**39-1)//2
edges = [lo, lo+1, -bound, -bound+1, -7, -3, -2, -1, 0, 1, 2, 3, 7, bound-1, bound, hi-1, hi]
samples = [(a,b) for a in edges for b in edges]
rng = random.Random(0x334D415448)
samples += [(rng.randint(lo,hi), rng.randint(lo,hi)) for _ in range(3000)]
expected = []
for a,b in samples:
    if not b:
        expected.append('domain')
        continue
    q,r = divmod(a,abs(b))
    if b < 0:
        q = -q
    ratio = Fraction(a,b)
    magnitude = abs(ratio)
    rounded = (2*magnitude.numerator + magnitude.denominator)//(2*magnitude.denominator)
    if ratio < 0:
        rounded = -rounded
    if not (lo <= q <= hi and lo <= rounded <= hi):
        expected.append('overflow')
    else:
        assert a == b*q+r and 0 <= r < abs(b)
        assert abs(ratio-rounded) <= Fraction(1,2)
        expected.append(f'{q} {r} {rounded}')
result = subprocess.run([sys.argv[1], '--oracle'], input=''.join(f'{a} {b}\n' for a,b in samples),
                        text=True, capture_output=True, timeout=30)
assert result.returncode == 0, result.stderr
actual = result.stdout.splitlines()
assert len(actual) == len(expected), (len(actual),len(expected))
for operands, got, wanted in zip(samples,actual,expected):
    assert got == wanted, (operands,got,wanted)
print(f'PASS: {len(samples)} full-int64 Euclidean division and rational-rounding cases')
