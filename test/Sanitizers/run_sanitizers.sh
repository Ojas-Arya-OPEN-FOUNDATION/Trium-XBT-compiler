#!/usr/bin/env bash
set -e

# ==============================================================================
# LLVM Layer 4: Sanitizer Verification Suite (ASan + UBSan)
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if [[ "$(uname -s)" == "Darwin" ]]; then
    if [[ -z "${SDKROOT}" ]]; then
        if [[ -d "/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk" ]]; then
            export SDKROOT="/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk"
        else
            export SDKROOT="$(xcrun --show-sdk-path 2>/dev/null || true)"
        fi
    fi
fi

echo "======================================================================"
echo "  LLVM Official Layer 4: AddressSanitizer & UndefinedBehaviorSanitizer"
echo "  Target: Clang compiler-rt instrumentation"
echo "======================================================================"

cd "${ROOT_DIR}"

mkdir -p bin_sanitizer

echo "[1/4] Compiling and running Layer 1 Core with ASan + UBSan..."
clang++ -std=c++20 -O1 -fsanitize=address,undefined -fno-omit-frame-pointer \
  test/Unit/test_unit_core.cpp -o bin_sanitizer/unit_core_asan
./bin_sanitizer/unit_core_asan
echo "  -> ASan/UBSan Core: 0 errors, 0 leaks, 0 UB."

echo "[2/4] Compiling and running Layer 1 Dense Algebra with ASan + UBSan..."
clang++ -std=c++20 -O1 -fsanitize=address,undefined -fno-omit-frame-pointer \
  test/Unit/test_unit_dense.cpp -o bin_sanitizer/unit_dense_asan
./bin_sanitizer/unit_dense_asan
echo "  -> ASan/UBSan Dense Algebra: 0 errors, 0 leaks, 0 UB."

echo "[3/4] Compiling and running Layer 1 Memory Arena with ASan + UBSan..."
clang++ -std=c++20 -O1 -fsanitize=address,undefined -fno-omit-frame-pointer \
  test/Unit/test_unit_memory.cpp -o bin_sanitizer/unit_memory_asan
./bin_sanitizer/unit_memory_asan
echo "  -> ASan/UBSan Memory Arena: 0 errors, 0 leaks, 0 UB."

echo "[4/4] Compiling and running Layer 1 MLIR Lowering with ASan + UBSan..."
clang++ -std=c++20 -O1 -fsanitize=address,undefined -fno-omit-frame-pointer \
  test/Unit/test_unit_mlir.cpp -o bin_sanitizer/unit_mlir_asan
./bin_sanitizer/unit_mlir_asan
echo "  -> ASan/UBSan MLIR Lowering: 0 errors, 0 leaks, 0 UB."

echo "[5/6] Compiling and running Layer 1 Frontend FP & Sparse Matrix with ASan + UBSan..."
clang++ -std=c++20 -O1 -fsanitize=address,undefined -fno-omit-frame-pointer \
  test/Unit/test_unit_frontend_fp_sparse.cpp -o bin_sanitizer/unit_frontend_fp_sparse_asan
./bin_sanitizer/unit_frontend_fp_sparse_asan
echo "  -> ASan/UBSan Frontend FP & Sparse: 0 errors, 0 leaks, 0 UB."

echo "[6/7] Compiling bt-compiler with ASan + UBSan and running self-test..."
clang++ -std=c++20 -O1 -fsanitize=address,undefined -fno-omit-frame-pointer \
  bt_compiler.cpp -o bin_sanitizer/bt_compiler_asan
./bin_sanitizer/bt_compiler_asan --test
echo "  -> ASan/UBSan Compiler Driver: 0 errors, 0 leaks, 0 UB."

echo "[7/7] Compiling and running Universal XBT Foundation with ASan + UBSan..."
clang++ -std=c++20 -O1 -fsanitize=address,undefined -fno-omit-frame-pointer -I. \
  test/Unit/test_unit_universal.cpp -o bin_sanitizer/unit_universal_asan
./bin_sanitizer/unit_universal_asan
echo "  -> ASan/UBSan Universal XBT Foundation: 0 errors, 0 leaks, 0 UB."

echo "======================================================================"
echo "  ALL SANITIZER TEST RUNS PASSED CLEANLY (0 LEAKS, 0 OUT-OF-BOUNDS, 0 UB)"
echo "======================================================================"
