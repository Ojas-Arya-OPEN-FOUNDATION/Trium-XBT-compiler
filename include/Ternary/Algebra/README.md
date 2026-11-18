# Trium-XBT Algebraic Virtual Memory & Circulant Weighing Engine

Isolated, header-only C++20 modules for computing beyond physical RAM limits **without compression** and **without OS swap**, using Circulant Weighing Matrices (`CW(n, k)`), Goethals–Seidel skew-Hadamard blocks, Kronecker lifting, reversible symplectic stepping, and bounded-RAM out-of-core streaming.

## Files

- [`CirculantWeighing.hpp`](CirculantWeighing.hpp) (`namespace bt::algebra`):
  - `CirculantTernaryOperator` — `O(n)` single-seed-row operators (`cw_120_36()`, `cw_even_weight4(m)`, `from_supports(...)`) with zero-multiplier `apply`, `apply_transpose`, and `apply_exact_inverse`.
  - `GoethalsSeidelSkewHadamard` — `order_212()` and `order_220()` orthogonal skew-Hadamard operators stored in 32 bytes.
  - `KroneckerTernaryOperator` — factorized `W_0 ⊗ ... ⊗ W_{d-1}` operators using `O(sum n_i)` seed storage.
  - `GeneralAlgebraicOperator` — arbitrary linear operators decomposed as `Circulant + LowRank(U*V^T) + SparseTriplet`.
- [`AlgebraicVirtualMemory.hpp`](AlgebraicVirtualMemory.hpp) (`namespace bt::vmem`):
  - `PhysicalMemoryMonitor` — live RSS, peak RSS, major page faults, and `ru_nswap` sampling.
  - `ZeroSwapStreamStore<T>` — bounded-RAM POSIX `pread`/`pwrite` out-of-core slab store (`ru_nswap == 0`).
  - `ReversibleSymplecticTape` — `O(1)` history-RAM forward/backward trajectory stepper.
  - `VirtualStencil3D<Scalar>` — matrix-free 3D 7-point Laplacian over `ZeroSwapStreamStore`.

## Quick Usage

```cpp
#include "include/Ternary/Algebra/CirculantWeighing.hpp"
#include "include/Ternary/Algebra/AlgebraicVirtualMemory.hpp"

// 1. Exact orthogonal CW(120, 36) transform (264 B seed, zero multipliers)
auto cw = bt::algebra::CirculantTernaryOperator::cw_120_36();
std::vector<int64_t> x(120, 1), y(120), x_rec(120);
cw.apply(x, y);
cw.apply_exact_inverse(y, x_rec); // x_rec == x

// 2. Kronecker lifting: 120^3 = 1,728,000 dim (~11.94 TB FP32 virtual) in 816 B seed
bt::algebra::KroneckerTernaryOperator kron({cw, cw, cw});

// 3. Zero-swap out-of-core 3D Laplacian under a 2 MiB slab budget
bt::vmem::ZeroSwapStreamStore<double> grid_in(128 * 128 * 128, 2 * 1024 * 1024);
bt::vmem::ZeroSwapStreamStore<double> grid_out(128 * 128 * 128, 2 * 1024 * 1024);
bt::vmem::VirtualStencil3D<double> lap(128, 128, 128);
lap.apply_laplacian_3d(grid_in, grid_out);
```

## Build, Test & Benchmark

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target unit_algebraic_memory algebraic_memory_bench -j
ctest --test-dir build -R "Layer1_Unit_Algebraic_Memory|Algebraic_Memory_Bench_Check|Algebraic_Memory_Z3" --output-on-failure
./build/algebraic_memory_bench
python3 verify_z3_algebraic_memory.py
```

## Extending

- **Add a new `CW(n, k)`**: call `CirculantTernaryOperator::from_supports(n, pos, neg, k)` and validate with `.verify_orthogonality()`.
- **Adjust out-of-core RAM budget**: pass `slab_bytes` to `ZeroSwapStreamStore<T>(count, slab_bytes)`.
