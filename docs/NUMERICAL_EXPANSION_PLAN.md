# Numerical expansion and SDK 1.2 plan

Date: 2026-10-30. User direction: extend floating-point operations, advanced numerical routines, older low-level helpers and SDK together. README/license grants remain unchanged. This is a focused next mathematical slice, not all advanced numerical analysis or hardware qualification.

## Implementation order and contracts

1. Extend finite BTFP widths 1..40 with reciprocal, cube root, hypot and integer power -16..16. Compute from exact integer data with one final nearest/ties-away rounding; enforce exponent bounds only on the final result. Define x^0=1, including 0^0; zero to a negative power is a domain error. Cube root accepts negative inputs. Hypot avoids intermediate square overflow and handles enormous exponent separation analytically.
2. Add a reusable header-level numerical module: exact sum, exact dot, norm2, and polynomial evaluation (ascending coefficients). These accumulate exact represented inputs/products and round once, unlike repeatedly rounded reductions/Horner. Bound arrays at 4096 terms, active term exponent window at 256 trits after stripping coefficient powers of three (before cancellation), and polynomial degree at 16. Reject a resource limit rather than silently discard terms. Ignore zero terms when measuring the window. Fixed 2048-bit scratch; no external arbitrary-precision dependency or binary floating intermediates.
3. Harden legacy low-level CORDIC helpers: signed zero/nonfinite circular input, safe atan2 scaling near binary64 maxima and subnormals, and explicit reduced-domain hyperbolic handling. Keep approximate-vs-exact contracts distinct. Exercise old power-of-three rounding helpers at signed minima/maxima as nonregressions.
4. Add additive ABI-1 / API-1.2 C functions, C++ and Python wrappers, metadata and limits. Preserve 24-byte FP records and transactional/alias-safe outputs. Keep existing SDK backends/caches intact. Do not expose approximate functions as certified operations. Rust changes require an available compiler before runtime qualification; no unverified certification.
5. Verify using an independent Python Fraction/root oracle, exhaustive small formats, all-width deterministic random/boundary cases, cancellation/permutation and adversarial exponent tests, raw C failure/alias checks, C++ consumers, Z3 admission/midpoint/buffer models, full Release and ASan/UBSan regressions, and installed packaging where tools are available.

## Gates and non-goals

- Count a new exact operation as qualified only after exact mantissa/exponent agreement, not a tolerance check.
- Array count/window/power failures must be explicit resource errors with unchanged output. Exponent overflow remains a range failure; invalid records/domain inputs remain invalid.
- Legacy helpers receive targeted robustness tests, not universal transcendental correctness claims. No GPU BTFP execution, floating GEMM, general linear solvers, autograd, unbounded polynomials or hardware synthesis is included.
- Current build/tool binaries may need restoration after workspace snapshots. Record actual tests/skips, versions and platform. Physical GPU testing stays isolated and user-run.
