#!/usr/bin/env python3
"""Executable obligations for docs/MATHEMATICAL_CONTRACT.md.
These prove the stated models, not arbitrary compiler correctness.
Missing Z3 is a CTest skip, not proof success. Unknown/timeout is failure.
"""
import sys
try:
    import z3 as z
except ImportError:
    print('SKIP: install the pinned verification requirements')
    sys.exit(77)


def prove(name, assumptions, counterexample):
    solver = z.Solver()
    solver.set(timeout=20000)
    solver.add(*assumptions, counterexample)
    result = solver.check()
    if result != z.unsat:
        raise AssertionError(f'{name}: {result}; {solver.model() if result == z.sat else solver.reason_unknown()}')
    print(f'PROVED: {name}')


# Full scalar domain: widening add/sub/mul into signed i128 cannot overflow.
H = (3**39 - 1)//2
a, b = z.Ints('a b')
domain = [a >= -H, a <= H, b >= -H, b <= H]
for name, value in [('add', a+b), ('sub', a-b), ('mul', a*b)]:
    prove(f'i128 intermediate bounds for {name}', domain,
          z.Or(value < -(2**127), value >= 2**127))
    # The result is accepted if and only if it is in the declared range.
    accepted = z.And(value >= -H, value <= H)
    prove(f'checked {name} range predicate', domain,
          accepted != (z.Abs(value) <= H))

# Executable bitvector model of the widened LLVM lowering, exhaustively at
# reduced width. Full-width integer intermediate bounds are established above.
a8, b8 = z.BitVecs('a8 b8', 8)
small_domain = [a8 >= -13, a8 <= 13, b8 >= -13, b8 <= 13]
for name, small, wide in [('add', a8+b8, z.SignExt(8,a8)+z.SignExt(8,b8)),
                           ('sub', a8-b8, z.SignExt(8,a8)-z.SignExt(8,b8)),
                           ('mul', a8*b8, z.SignExt(8,a8)*z.SignExt(8,b8))]:
    prove(f'accepted {name} truncation is exact (3-trit model)',
          small_domain + [wide >= -13, wide <= 13],
          z.SignExt(8, z.Extract(7,0,wide)) != wide)

# Lazy three-way selection partitions all integer conditions exactly once.
c = z.Int('condition')
p, zero, n = c > 0, c == 0, c < 0
prove('selection is exhaustive and mutually exclusive', [],
      z.Sum([z.If(t,1,0) for t in (p,zero,n)]) != 1)

# Quantization: nearest integer reconstruction with scale s has bounded error;
# no distribution-independent zero-bias claim is made.
x, scale = z.Reals('x scale')
q = z.Int('q')
prove('nearest quantization error <= scale/2',
      [scale > 0, x/scale >= q-z.RealVal('1/2'), x/scale <= q+z.RealVal('1/2')],
      z.Abs(x-q*scale) > scale/2)

# CSR accumulation: <= UINT32_MAX signed 64-bit terms always fit signed i128.
assert (2**32-1)*2**63 < 2**127
print('PROVED by integer bound: CSR i128 accumulation capacity')
print('All Phase 1 mathematical model obligations passed')

# BTFP exponent intermediates are wide before normalization. These are model
# obligations, not a proof that every C++ expression has been extracted correctly.
ea, eb, ec = z.Ints('ea eb ec')
exponents = [v >= -(2**31) for v in (ea,eb,ec)] + [v < 2**31 for v in (ea,eb,ec)]
for label, expression in [('product', ea+eb), ('alignment', ea-eb),
                          ('fma alignment', ea+eb-ec), ('normalization', ea+eb+128)]:
    prove(f'BTFP int64 exponent capacity: {label}', exponents,
          z.Or(expression < -(2**63), expression >= 2**63))
# Exact host-integer inequalities validate the conservative shift guards for
# every instantiable precision, including W=40 (not just the common aliases).
for width in range(1,41):
    m = (3**width-1)//2
    assert m*m + m < 2**127
    assert m*3**(78-width) + m*m < 2**127
    assert m*m*3**max(78-2*width,0) + m < 2**127
    assert m*3**(78-width) + m < 2**127
assert 3**81//2 < 2**128
print('PROVED by integer bounds: guarded BTFP accumulator capacities, W=1..40')
