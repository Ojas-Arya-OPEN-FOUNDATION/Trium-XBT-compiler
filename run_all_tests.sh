#!/usr/bin/env bash
set -e

# ==============================================================================
# Master Test Harness: check-all (Emulates 'ninja check-all' across all 7 layers)
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SDKROOT=/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk

echo "======================================================================"
echo "  ANTIGRAVITY BALANCED TERNARY COMPILER - LLVM 7-LAYER TEST SUITE"
echo "  Emulating LLVM 'ninja check-all' + compiler-rt + lit + FileCheck"
echo "======================================================================"

# Step 0: Ensure bt-compiler and bt-opt are built
echo ""
echo ">>> [BUILD] Building optimized native bt-compiler..."
clang++ -std=c++20 -O3 "${SCRIPT_DIR}/bt_compiler.cpp" -o "${SCRIPT_DIR}/bt-compiler"
echo "    -> bt-compiler built successfully."
echo ">>> [BUILD] Building optimized native bt-opt MLIR tool..."
clang++ -std=c++20 -O3 \
    "${SCRIPT_DIR}/lib/Ternary/TernaryDialect.cpp" \
    "${SCRIPT_DIR}/lib/Ternary/TernaryPasses.cpp" \
    "${SCRIPT_DIR}/iree/TernaryIREEPlugin.cpp" \
    "${SCRIPT_DIR}/tools/bt-opt/bt-opt.cpp" \
    -o "${SCRIPT_DIR}/bt-opt"
echo "    -> bt-opt built successfully."
echo ">>> [BUILD] Building shared C-ABI runtime library libbt_runtime.dylib..."
clang++ -std=c++20 -O3 -shared -fPIC "${SCRIPT_DIR}/lib/Ternary/ternary_c_abi.cpp" -I"${SCRIPT_DIR}" -o "${SCRIPT_DIR}/libbt_runtime.dylib"
echo "    -> libbt_runtime.dylib built successfully."

# Layer 1: Unit Tests
echo ""
echo ">>> [LAYER 1/7] Running Unit Tests (test/Unit)..."
clang++ -std=c++20 -O3 "${SCRIPT_DIR}/test/Unit/test_unit_core.cpp" -o "${SCRIPT_DIR}/test/Unit/unit_core"
"${SCRIPT_DIR}/test/Unit/unit_core"
clang++ -std=c++20 -O3 "${SCRIPT_DIR}/test/Unit/test_unit_dense.cpp" -o "${SCRIPT_DIR}/test/Unit/unit_dense"
"${SCRIPT_DIR}/test/Unit/unit_dense"
clang++ -std=c++20 -O3 "${SCRIPT_DIR}/test/Unit/test_unit_memory.cpp" -o "${SCRIPT_DIR}/test/Unit/unit_memory"
"${SCRIPT_DIR}/test/Unit/unit_memory"
clang++ -std=c++20 -O3 "${SCRIPT_DIR}/test/Unit/test_unit_ast.cpp" -o "${SCRIPT_DIR}/test/Unit/unit_ast"
"${SCRIPT_DIR}/test/Unit/unit_ast"
clang++ -std=c++20 -O3 "${SCRIPT_DIR}/test/Unit/test_unit_mlir.cpp" -o "${SCRIPT_DIR}/test/Unit/unit_mlir"
"${SCRIPT_DIR}/test/Unit/unit_mlir"
clang++ -std=c++20 -O3 "${SCRIPT_DIR}/test/Unit/test_unit_fp_sparse.cpp" -o "${SCRIPT_DIR}/test/Unit/unit_fp_sparse"
"${SCRIPT_DIR}/test/Unit/unit_fp_sparse"
clang++ -std=c++20 -O3 "${SCRIPT_DIR}/test/Unit/test_unit_frontend_fp_sparse.cpp" -o "${SCRIPT_DIR}/test/Unit/unit_frontend_fp_sparse"
"${SCRIPT_DIR}/test/Unit/unit_frontend_fp_sparse"
clang++ -std=c++20 -O3 "${SCRIPT_DIR}/test/Unit/test_unit_libbt.cpp" -o "${SCRIPT_DIR}/test/Unit/unit_libbt"
"${SCRIPT_DIR}/test/Unit/unit_libbt"
clang++ -std=c++20 -O3 \
    "${SCRIPT_DIR}/lib/Ternary/TernaryDialect.cpp" \
    "${SCRIPT_DIR}/lib/Ternary/TernaryPasses.cpp" \
    "${SCRIPT_DIR}/iree/TernaryIREEPlugin.cpp" \
    "${SCRIPT_DIR}/test/Unit/test_unit_bt_opt.cpp" \
    -o "${SCRIPT_DIR}/test/Unit/unit_bt_opt"
"${SCRIPT_DIR}/test/Unit/unit_bt_opt"
clang++ -std=c++20 -O3 "${SCRIPT_DIR}/test/Unit/test_unit_c_abi_ffi.cpp" -I"${SCRIPT_DIR}" -L"${SCRIPT_DIR}" -lbt_runtime -Wl,-rpath,"${SCRIPT_DIR}" -o "${SCRIPT_DIR}/test/Unit/unit_c_abi_ffi"
"${SCRIPT_DIR}/test/Unit/unit_c_abi_ffi"
python3 "${SCRIPT_DIR}/ffi/python/bt_ffi.py"
clang++ -std=c++20 -O3 "${SCRIPT_DIR}/test/Unit/test_unit_bigint_symplectic.cpp" -I"${SCRIPT_DIR}" -o "${SCRIPT_DIR}/test/Unit/unit_bigint_symplectic"
"${SCRIPT_DIR}/test/Unit/unit_bigint_symplectic"
clang++ -std=c++20 -O3 "${SCRIPT_DIR}/test/Unit/test_unit_xbt_hw.cpp" -I"${SCRIPT_DIR}" -o "${SCRIPT_DIR}/test/Unit/unit_xbt_hw"
"${SCRIPT_DIR}/test/Unit/unit_xbt_hw"
clang++ -std=c++20 -O3 "${SCRIPT_DIR}/test/Unit/test_unit_phase3.cpp" -I"${SCRIPT_DIR}" -o "${SCRIPT_DIR}/test/Unit/unit_phase3"
"${SCRIPT_DIR}/test/Unit/unit_phase3"



# Layer 2: Regression Tests
echo ""
echo ">>> [LAYER 2/7] Running Regression Tests (test/Regression with lit + FileCheck)..."
python3 "${SCRIPT_DIR}/test/lit_runner.py" "${SCRIPT_DIR}/test/Regression"

# Layer 3: Whole-Program Tests
echo ""
echo ">>> [LAYER 3/7] Running Whole-Program Tests (test/WholeProgram with lit + FileCheck)..."
python3 "${SCRIPT_DIR}/test/lit_runner.py" "${SCRIPT_DIR}/test/WholeProgram"

# Layer 4: Sanitizer Tests
echo ""
echo ">>> [LAYER 4/7] Running Compiler-RT Sanitizer Tests (ASan + UBSan)..."
"${SCRIPT_DIR}/test/Sanitizers/run_sanitizers.sh"

# Layer 5: Target CodeGen Tests
echo ""
echo ">>> [LAYER 5/7] Running Target CodeGen Tests (test/CodeGen with lit + FileCheck)..."
python3 "${SCRIPT_DIR}/test/lit_runner.py" "${SCRIPT_DIR}/test/CodeGen"

# Layer 6: MLIR Dialect Tests
echo ""
echo ">>> [LAYER 6/7] Running MLIR Dialect Tests (test/Dialect with lit + FileCheck)..."
python3 "${SCRIPT_DIR}/test/lit_runner.py" "${SCRIPT_DIR}/test/Dialect"

# Layer 7: Sema & Diagnostic Tests
echo ""
echo ">>> [LAYER 7/7] Running Sema & Diagnostic Tests (test/Sema with lit + FileCheck)..."
python3 "${SCRIPT_DIR}/test/lit_runner.py" "${SCRIPT_DIR}/test/Sema"

echo ""
echo "======================================================================"
echo "  ALL 7 LLVM TESTING LAYERS PASSED WITH 100% SUCCESS!"
echo "  1. Unit tests: PASSED"
echo "  2. Regression tests: PASSED"
echo "  3. Whole-program tests: PASSED"
echo "  4. Sanitizer tests (ASan+UBSan): PASSED"
echo "  5. Target CodeGen tests (ARM64/x86/Metal/Vulkan/LLVM): PASSED"
echo "  6. MLIR Dialect tests: PASSED"
echo "  7. Clang Sema & Diagnostic tests: PASSED"
echo "======================================================================"
