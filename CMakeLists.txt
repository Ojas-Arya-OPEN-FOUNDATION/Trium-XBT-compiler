# ==============================================================================
# Trium-XBT Foundational Compiler & MLIR Dialect Build Infrastructure
# ==============================================================================
# Supports both:
# 1. Out-of-tree LLVM/MLIR monorepo builds via find_package(MLIR / LLVM)
# 2. Standalone C++20 build target with zero dependencies for reproducible builds
# ==============================================================================

cmake_minimum_required(VERSION 3.20)
project(TriumXBTCompiler
    VERSION 2.0.0
    DESCRIPTION "Trium-XBT Foundational Compiler & MLIR Dialect Infrastructure"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
find_package(Threads REQUIRED)
link_libraries(Threads::Threads ${CMAKE_DL_LIBS})

# macOS ARM64 Apple Silicon Optimization & SDK Configuration
if(APPLE)
    if(NOT DEFINED ENV{SDKROOT} AND EXISTS "/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk")
        set(ENV{SDKROOT} "/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk")
    endif()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Build Options
option(BT_ENABLE_MLIR "Build with upstream LLVM/MLIR monorepo if found" ON)
option(BT_BUILD_TESTS "Build complete 7-layer test suite" ON)
option(BT_ENABLE_ASAN "Enable AddressSanitizer and UndefinedBehaviorSanitizer" OFF)

if(BT_ENABLE_ASAN)
    add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer)
    add_link_options(-fsanitize=address,undefined)
endif()

# ==============================================================================
# Out-of-tree LLVM / MLIR Configuration
# ==============================================================================
set(BT_HAS_MLIR FALSE)

if(BT_ENABLE_MLIR)
    find_package(MLIR CONFIG QUIET)
    find_package(LLVM CONFIG QUIET)

    if(MLIR_FOUND AND LLVM_FOUND)
        set(BT_HAS_MLIR TRUE)
        message(STATUS "Found MLIR ${MLIR_PACKAGE_VERSION} and LLVM ${LLVM_PACKAGE_VERSION}")
        message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
        
        list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
        list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
        
        include(TableGen)
        include(AddLLVM)
        include(AddMLIR)
        include(HandleLLVMOptions)

        include_directories(${LLVM_INCLUDE_DIRS})
        include_directories(${MLIR_INCLUDE_DIRS})
        link_directories(${LLVM_BUILD_LIBRARY_DIR})
        add_definitions(${LLVM_DEFINITIONS} -DBT_HAS_MLIR=1)

        # TableGen Dialect & Op Target Generation
        set(LLVM_TARGET_DEFINITIONS include/Ternary/TernaryOps.td)
        mlir_tablegen(TernaryOpsDialect.h.inc -gen-dialect-decls)
        mlir_tablegen(TernaryOpsDialect.cpp.inc -gen-dialect-defs)
        mlir_tablegen(TernaryOps.h.inc -gen-op-decls)
        mlir_tablegen(TernaryOps.cpp.inc -gen-op-defs)
        add_public_tablegen_target(TernaryOpsIncGen)
    else()
        message(STATUS "Upstream LLVM/MLIR monorepo not detected. Activating standalone C++20 fallback mode.")
    endif()
endif()

# In the standalone fallback the bt dialect and its passes are still built --
# they are the C++20 implementation in ternary_mlir_lowering.hpp, not stubs.
# Mark that so the frontend can lower through the dialect either way.
if(BT_ENABLE_MLIR AND NOT BT_HAS_MLIR)
    add_definitions(-DBT_HAS_STANDALONE_MLIR=1)
endif()

# ==============================================================================
# Include Paths
# ==============================================================================
include_directories(
    "${CMAKE_CURRENT_SOURCE_DIR}/."
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/iree"
)

# ==============================================================================
# Target: TernaryDialect Library
# ==============================================================================
add_library(TernaryDialect STATIC
    lib/Ternary/TernaryDialect.cpp
    lib/Ternary/TernaryPasses.cpp
)

target_include_directories(TernaryDialect PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

if(BT_HAS_MLIR)
    add_dependencies(TernaryDialect TernaryOpsIncGen)
    target_link_libraries(TernaryDialect PUBLIC
        MLIRIR
        MLIRSupport
        MLIRPass
        MLIRLinalgDialect
        MLIRVectorDialect
    )
endif()

# ==============================================================================
# Target: TernaryIREEPlugin Library
# ==============================================================================
add_library(TernaryIREEPlugin STATIC
    iree/TernaryIREEPlugin.cpp
)

target_include_directories(TernaryIREEPlugin PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/iree
)

target_link_libraries(TernaryIREEPlugin PUBLIC
    TernaryDialect
)

# ==============================================================================
# Target: ternary_clibi (Shared Runtime Library for C ABI / FFI)
# ==============================================================================
add_library(ternary_clibi SHARED
    lib/Ternary/ternary_c_abi.cpp
)

target_include_directories(ternary_clibi PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

set_target_properties(ternary_clibi PROPERTIES
    OUTPUT_NAME bt_runtime
    BUILD_RPATH "${CMAKE_CURRENT_BINARY_DIR}"
    INSTALL_RPATH "@loader_path"
)

# ==============================================================================
# Target: bt-compiler (Native Balanced Ternary End-to-End Compiler)
# ==============================================================================
add_executable(bt-compiler
    bt_compiler.cpp
)

target_include_directories(bt-compiler PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# ==============================================================================
# Target: bt-opt (MLIR Modular Optimizer Driver for 'bt' Dialect)
# ==============================================================================
add_executable(bt-opt
    tools/bt-opt/bt-opt.cpp
)

target_include_directories(bt-opt PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/iree
)

target_link_libraries(bt-opt PRIVATE
    TernaryDialect
    TernaryIREEPlugin
)

# ==============================================================================
# Testing: 7-Layer Test Suite & CTest Integration
# ==============================================================================
if(BT_BUILD_TESTS)
    enable_testing()
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    # Test predicates must survive Release/NDEBUG builds as well as Debug.
    if(MSVC)
        add_compile_options(/UNDEBUG)
    else()
        add_compile_options(-UNDEBUG)
    endif()

    # Unit Test Binaries
    add_executable(unit_core test/Unit/test_unit_core.cpp)
    target_include_directories(unit_core PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

    # Ternary GEMM micro-kernel: ISA-dispatched, register-blocked, packed storage.
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag("-march=native" BT_HAS_MARCH_NATIVE)
    add_executable(unit_microkernel test/Unit/test_unit_microkernel.cpp)
    target_include_directories(unit_microkernel PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/include)
    if(BT_HAS_MARCH_NATIVE)
        target_compile_options(unit_microkernel PRIVATE -march=native)
    endif()

    # Honest benchmark runner. `--check` verifies every kernel bit-exact
    # against a reference and times nothing it has not verified, so CI can run
    # it in seconds. The full sweep (256/512/1024) is opt-in and is the only
    # place performance claims should come from.
    add_executable(ternary_bench tools/ternary_bench.cpp tools/bench_harness.hpp)
    target_include_directories(ternary_bench PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/tools)
    if(BT_HAS_MARCH_NATIVE)
        target_compile_options(ternary_bench PRIVATE -march=native)
    endif()

    # Tile / ISA survey. Measures the (ISA x tile) grid with the two varied
    # INDEPENDENTLY - comparing a scalar 4x4 against an AVX-512 8x4 changes two
    # variables at once and cannot justify either choice. Emits CSV so results
    # from several machines can be compared. Not a CTest target: it is a
    # measurement instrument, and its numbers are not portable. See
    # docs/TILE_DEFAULT.md before changing any dispatch default.
    add_executable(tile_survey tools/tile_survey.cpp)
    target_include_directories(tile_survey PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/include)
    if(BT_HAS_MARCH_NATIVE)
        target_compile_options(tile_survey PRIVATE -march=native)
    endif()

    add_executable(unit_dense test/Unit/test_unit_dense.cpp)
    target_include_directories(unit_dense PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

    add_executable(unit_carry_free test/Unit/test_unit_carry_free.cpp)
    target_include_directories(unit_carry_free PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    add_test(NAME unit_carry_free COMMAND unit_carry_free)

    add_executable(unit_memory test/Unit/test_unit_memory.cpp)
    target_include_directories(unit_memory PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

    add_executable(unit_ast test/Unit/test_unit_ast.cpp)
    target_include_directories(unit_ast PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

    add_executable(unit_mlir test/Unit/test_unit_mlir.cpp)
    target_include_directories(unit_mlir PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

    add_executable(unit_fp_sparse test/Unit/test_unit_fp_sparse.cpp)
    target_include_directories(unit_fp_sparse PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

    add_executable(unit_libbt test/Unit/test_unit_libbt.cpp)
    target_include_directories(unit_libbt PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

    add_executable(unit_bt_opt test/Unit/test_unit_bt_opt.cpp)
    target_include_directories(unit_bt_opt PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/iree
    )
    target_compile_definitions(unit_bt_opt PRIVATE
        BT_UNIT_PROJECT_ROOT="${CMAKE_CURRENT_SOURCE_DIR}"
    )
    target_link_libraries(unit_bt_opt PRIVATE
        TernaryDialect
        TernaryIREEPlugin
    )

    add_executable(unit_frontend_fp_sparse test/Unit/test_unit_frontend_fp_sparse.cpp)
    target_include_directories(unit_frontend_fp_sparse PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

    add_executable(unit_c_abi_ffi test/Unit/test_unit_c_abi_ffi.cpp)
    target_include_directories(unit_c_abi_ffi PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(unit_c_abi_ffi PRIVATE ternary_clibi)
    set_target_properties(unit_c_abi_ffi PROPERTIES
        BUILD_RPATH "${CMAKE_CURRENT_BINARY_DIR}"
    )

    add_executable(unit_bigint_symplectic test/Unit/test_unit_bigint_symplectic.cpp)
    target_include_directories(unit_bigint_symplectic PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

    add_executable(unit_xbt_hw test/Unit/test_unit_xbt_hw.cpp)
    target_include_directories(unit_xbt_hw PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

    # Conformance: runs the Z3-verified vectors against the Xbt hardware model.
    # The vectors are generated by verification/gen_vectors.py from the
    # mathematical specification, so this target is the mechanical link between
    # "the proofs" and "the hardware model".
    #
    # The generated table is ~4 MB, so it is a build artifact (gitignored) and
    # is regenerated here rather than checked in.
    set(BT_XBT_VECTORS ${CMAKE_CURRENT_SOURCE_DIR}/verification/vectors/xbt_conformance.hpp)
    add_custom_command(
        OUTPUT ${BT_XBT_VECTORS}
        COMMAND ${Python3_EXECUTABLE} -m verification.gen_vectors
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/verification/gen_vectors.py
                ${CMAKE_CURRENT_SOURCE_DIR}/verification/xbt_isa.py
                ${CMAKE_CURRENT_SOURCE_DIR}/verification/z3core.py
        COMMENT "Generating Xbt conformance vectors from the verified specification"
        VERBATIM
    )
    add_custom_target(xbt_conformance_vectors DEPENDS ${BT_XBT_VECTORS})
    add_executable(unit_xbt_conformance test/Unit/test_xbt_conformance.cpp)
    add_dependencies(unit_xbt_conformance xbt_conformance_vectors)
    target_include_directories(unit_xbt_conformance PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

    # Instruction selection: core IR -> Xbt machine code -> execution on the
    # coprocessor model.
    add_executable(unit_xbt_selector test/Unit/test_xbt_selector.cpp)
    target_include_directories(unit_xbt_selector PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

    add_executable(unit_scalar_mul test/Unit/test_unit_scalar_mul.cpp)
    target_include_directories(unit_scalar_mul PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/include)
    add_test(NAME Layer1_Unit_Scalar_Mul COMMAND unit_scalar_mul)

    add_executable(unit_tensor test/Unit/test_unit_tensor.cpp)
    target_include_directories(unit_tensor PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/include)
    add_test(NAME Layer1_Unit_Tensor COMMAND unit_tensor)

    add_executable(unit_translation_validation test/Unit/test_translation_validation.cpp)
    target_include_directories(unit_translation_validation PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/include)
    add_test(NAME Translation_Validation COMMAND unit_translation_validation)

    # Retargeting seam: target descriptions + backend registry.
    add_executable(unit_backend_registry test/Unit/test_backend_registry.cpp)
    target_include_directories(unit_backend_registry PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

    # ---- target descriptions: targets/*.yaml -> generated C++ ---------------
    # The generated header is COMMITTED, so building never needs Python.
    # `gen-targets` regenerates it; `Check_Target_Descriptions` asserts the
    # committed copy matches the YAML, which is what keeps it honest.
    find_package(Python3 COMPONENTS Interpreter QUIET)
    if(Python3_Interpreter_FOUND)
        add_custom_target(gen-targets
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tools/gen_targets.py
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            COMMENT "Regenerating GeneratedTargets.hpp from targets/*.yaml"
            VERBATIM)
        add_test(NAME Check_Target_Descriptions
            COMMAND ${Python3_EXECUTABLE}
                    ${CMAKE_CURRENT_SOURCE_DIR}/tools/gen_targets.py --check
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
        set_tests_properties(Check_Target_Descriptions PROPERTIES
            LABELS "codegen")
    endif()

    add_executable(unit_generated_targets test/Unit/test_generated_targets.cpp)
    target_include_directories(unit_generated_targets PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    add_test(NAME unit_generated_targets COMMAND unit_generated_targets)

    # Frontend -> bt MLIR dialect bridge. Only meaningful when the dialect is
    # compiled in (BT_ENABLE_MLIR), since it links the dialect registry.
    if(BT_ENABLE_MLIR)
        add_executable(unit_bt_dialect_bridge test/Unit/test_bt_dialect_bridge.cpp)
        target_include_directories(unit_bt_dialect_bridge PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
        target_link_libraries(unit_bt_dialect_bridge PRIVATE TernaryDialect)
    endif()

    # Regression coverage for the 2026-11-02 deep audit. The audit found that
    # allocator sizes wrapped, pool exceptions escaped, and numerical edge cases
    # returned NaN/aborted; these targets keep those repairs from regressing.
    add_executable(unit_audit_regressions test/Unit/test_unit_audit_regressions.cpp)
    target_include_directories(unit_audit_regressions PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    find_package(Threads REQUIRED)
    target_link_libraries(unit_audit_regressions PRIVATE Threads::Threads)

    add_library(bt_hijack_fixture SHARED test/Unit/legacy_ffi_hijack.cpp)
    set_target_properties(bt_hijack_fixture PROPERTIES
        OUTPUT_NAME bt_runtime
        PREFIX "lib"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/ffi-hijack-fixture")

    # CTest Registration: Layer 1 Unit Tests
    add_test(NAME Layer1_Unit_Core COMMAND unit_core)
    add_test(NAME Layer1_Unit_MicroKernel COMMAND unit_microkernel)
    add_test(NAME Bench_Smoke_Check COMMAND ternary_bench --check)
    add_test(NAME Layer1_Unit_Dense COMMAND unit_dense)
    add_test(NAME Layer1_Unit_Memory COMMAND unit_memory)
    add_test(NAME Layer1_Unit_AST COMMAND unit_ast)
    add_test(NAME Layer1_Unit_MLIR COMMAND unit_mlir)
    add_test(NAME Layer1_Unit_FP_Sparse COMMAND unit_fp_sparse)
    add_test(NAME Layer1_Unit_LibBT COMMAND unit_libbt)
    add_test(NAME Layer1_Unit_BT_Opt COMMAND unit_bt_opt)
    add_test(NAME Layer1_Unit_Frontend_FP_Sparse
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test/require_tool.py clang $<TARGET_FILE:unit_frontend_fp_sparse>)
    set_tests_properties(Layer1_Unit_Frontend_FP_Sparse PROPERTIES SKIP_RETURN_CODE 77)
    add_test(NAME Layer1_Unit_C_ABI_FFI COMMAND unit_c_abi_ffi)
    add_test(NAME Layer1_Unit_BigInt_Symplectic COMMAND unit_bigint_symplectic)
    add_test(NAME Layer1_Unit_XBT_HW COMMAND unit_xbt_hw)
    add_test(NAME Layer1_Unit_XBT_Conformance COMMAND unit_xbt_conformance)
    add_test(NAME Layer1_Unit_XBT_Selector COMMAND unit_xbt_selector)
    add_test(NAME Layer1_Unit_Backend_Registry COMMAND unit_backend_registry)
    if(BT_ENABLE_MLIR)
        add_test(NAME Layer1_Unit_BT_Dialect_Bridge COMMAND unit_bt_dialect_bridge)
    endif()
    add_test(NAME Layer1_Unit_Audit_Regressions COMMAND unit_audit_regressions)

    # CTest Registration: Layers 2-7
    if(TARGET ternary_clibi)
        add_test(NAME Legacy_FFI_Hardening
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test/Unit/legacy_ffi_loader.py
                    $<TARGET_FILE:ternary_clibi> $<TARGET_FILE:bt_hijack_fixture>
                    ${CMAKE_CURRENT_SOURCE_DIR}/ffi/python/bt_ffi.py)
        set_tests_properties(Legacy_FFI_Hardening PROPERTIES TIMEOUT 120)
    endif()
    add_test(NAME Compiler_Audit_Regressions
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test/audit_regressions.py $<TARGET_FILE:bt-compiler>)
    set_tests_properties(Compiler_Audit_Regressions PROPERTIES TIMEOUT 180)
    add_test(NAME Layer2_Regression
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test/lit_runner.py ${CMAKE_CURRENT_SOURCE_DIR}/test/Regression
    )
    add_test(NAME Layer3_WholeProgram
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test/lit_runner.py ${CMAKE_CURRENT_SOURCE_DIR}/test/WholeProgram
    )
    # Sanitizers are an explicit separate build. They must not silently rebuild
    # source-tree artifacts with a different hardcoded compiler during ctest.
    if(BT_ENABLE_ASAN)
        message(STATUS "ASan/UBSan instrumentation enabled for all test targets")
    endif()
    add_test(NAME Layer5_CodeGen
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test/lit_runner.py ${CMAKE_CURRENT_SOURCE_DIR}/test/CodeGen
    )
    add_test(NAME Layer6_Dialect
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test/lit_runner.py ${CMAKE_CURRENT_SOURCE_DIR}/test/Dialect
    )
    add_test(NAME Layer6_Frontend_Dialect_Pipeline
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test/frontend_dialect_pipeline.py
    )
    add_test(NAME Layer7_Sema
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test/lit_runner.py ${CMAKE_CURRENT_SOURCE_DIR}/test/Sema
    )

    add_executable(unit_math_core test/Unit/test_unit_math_core.cpp)
    add_test(NAME Core_Math_IR COMMAND unit_math_core)
    add_test(NAME Core_Math_Reference COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/test/math_reference.py $<TARGET_FILE:unit_math_core>)
    add_executable(unit_core_lowering test/Unit/test_unit_core_lowering.cpp)
    add_test(NAME Core_Lowering_Optimizer COMMAND unit_core_lowering)
    add_test(NAME Core_Profile_Contract COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test/profile_contract.py)
    add_test(NAME Core_Backend_Regression COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/test/core_backend_regression.py $<TARGET_FILE:bt-compiler>)
    add_test(NAME Core_Optimized_Differential COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/test/phase1_differential.py $<TARGET_FILE:bt-compiler>)
    add_test(NAME Core_Optimized_Programs COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/test/core_program_differential.py $<TARGET_FILE:bt-compiler>)
    set_tests_properties(Core_Optimized_Differential Core_Optimized_Programs PROPERTIES ENVIRONMENT "BT_TEST_CORE_OPT=1" SKIP_RETURN_CODE 77)
    add_test(NAME CUDA_Driver_Unit COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test/cuda_driver_unit.py)
    # --gpu-map compiles a named function from .bt source into a real GLSL 450
    # compute kernel. These check the emitted kernel is SEMANTICALLY the user's
    # program, not merely well-formed: the kernel is transliterated back to C++,
    # run, and compared against the JIT result of the same source.
    # Kernel contract: the seam between the GLSL backend and any device
    # runtime. Runs with no GPU and no Vulkan SDK. Its negative controls must
    # keep passing - a contract check that cannot fail proves nothing.
    add_executable(unit_kernel_contract test/Unit/test_unit_kernel_contract.cpp)
    target_include_directories(unit_kernel_contract PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    add_test(NAME Core_Kernel_Contract COMMAND unit_kernel_contract)

    add_test(NAME GPU_Map_SignSelect COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/test/gpu_map_test.py $<TARGET_FILE:bt-compiler>
        ${CMAKE_CURRENT_SOURCE_DIR}/examples/07_gpu_kernel_demo.bt transform --inputs=-2,-1,0,1,2)
    add_test(NAME GPU_Map_LoopAndCall COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/test/gpu_map_test.py $<TARGET_FILE:bt-compiler>
        ${CMAKE_CURRENT_SOURCE_DIR}/examples/07_gpu_kernel_demo.bt accumulate --inputs=0,1,4)

    # Point the gpu-map tests at a shader compiler if one was installed.
    find_program(GLSLANG_VALIDATOR_EXECUTABLE glslangValidator
        HINTS ${CMAKE_CURRENT_SOURCE_DIR}/tools/.toolchains/bin)
    if(GLSLANG_VALIDATOR_EXECUTABLE)
        set_tests_properties(GPU_Map_SignSelect GPU_Map_LoopAndCall PROPERTIES
            ENVIRONMENT "GLSLANG_VALIDATOR=${GLSLANG_VALIDATOR_EXECUTABLE}")
    endif()
    add_test(NAME CUDA_Offline_Validation COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/test/cuda_map_test.py $<TARGET_FILE:bt-compiler>)
    add_test(NAME CUDA_Device_Qualification COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/test/cuda_map_test.py $<TARGET_FILE:bt-compiler> --device)
    set_tests_properties(Core_Backend_Regression CUDA_Offline_Validation CUDA_Device_Qualification PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 180)
    add_executable(unit_ir_program test/Unit/test_unit_ir_program.cpp)
    add_test(NAME Core_Program_IR COMMAND unit_ir_program)
    add_executable(unit_fp_foundation test/Unit/test_unit_fp_foundation.cpp)
    add_test(NAME FP_Foundation_Unit COMMAND unit_fp_foundation)
    add_executable(unit_numeric_oracle test/Unit/test_unit_numeric_oracle.cpp)
    add_test(NAME Core_Numeric_Reference COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/test/numeric_reference.py $<TARGET_FILE:unit_numeric_oracle>)
    add_test(NAME Core_Rocq COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test/check_rocq.py)
    set_tests_properties(Core_Rocq PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 1320)
    add_executable(unit_phase1 test/Unit/test_unit_phase1.cpp)
    target_link_libraries(unit_phase1 PRIVATE ternary_clibi)
    add_test(NAME Phase1_Safety COMMAND unit_phase1)
    add_executable(unit_phase3 test/Unit/test_unit_phase3.cpp)
    add_executable(unit_universal test/Unit/test_unit_universal.cpp)
    add_test(NAME Layer1_Unit_Phase3 COMMAND unit_phase3)
    add_test(NAME Layer1_Unit_Universal COMMAND unit_universal)
    add_test(NAME Phase1_LLVM_Differential
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test/phase1_differential.py $<TARGET_FILE:bt-compiler>)
    add_test(NAME Core_Program_Differential COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/test/core_program_differential.py $<TARGET_FILE:bt-compiler>)
    set_tests_properties(Core_Program_Differential PROPERTIES SKIP_RETURN_CODE 77)
    add_test(NAME Phase1_Z3
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/verify_z3_phase1.py)
    set_tests_properties(Phase1_LLVM_Differential Phase1_Z3 PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 120)
    add_test(NAME Runtime_ABI_Client COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/test/runtime_abi_regression.py $<TARGET_FILE:ternary_clibi>)
    add_test(NAME Phase1_Python_FFI
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/ffi/python/bt_ffi.py)
    set_tests_properties(Phase1_Python_FFI PROPERTIES
        ENVIRONMENT "BT_RUNTIME_LIBRARY=$<TARGET_FILE:ternary_clibi>")
    # ctypes loads an instrumented DSO into an otherwise unsanitized Python host.
    # GNU ASan must be preloaded before Python, not discovered during dlopen.
    if(BT_ENABLE_ASAN AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND UNIX AND NOT APPLE)
        execute_process(COMMAND ${CMAKE_CXX_COMPILER} -print-file-name=libasan.so
            OUTPUT_VARIABLE BT_ASAN_RUNTIME OUTPUT_STRIP_TRAILING_WHITESPACE)
        if(NOT EXISTS "${BT_ASAN_RUNTIME}")
            message(FATAL_ERROR "Cannot locate GNU ASan runtime for Python FFI test")
        endif()
        # Python does not link libstdc++ at startup. Preload it after ASan so
        # ASan can resolve __cxa_throw before a dlopened runtime throws.
        execute_process(COMMAND ${CMAKE_CXX_COMPILER} -print-file-name=libstdc++.so
            OUTPUT_VARIABLE BT_CXX_ABI_RUNTIME OUTPUT_STRIP_TRAILING_WHITESPACE)
        if(NOT EXISTS "${BT_CXX_ABI_RUNTIME}")
            message(FATAL_ERROR "Cannot locate C++ exception runtime for sanitized Python FFI tests")
        endif()
        set_property(TEST Phase1_Python_FFI Runtime_ABI_Client Legacy_FFI_Hardening APPEND PROPERTY ENVIRONMENT
            "LD_PRELOAD=${BT_ASAN_RUNTIME}:${BT_CXX_ABI_RUNTIME}" "ASAN_OPTIONS=detect_leaks=0")
    endif()
    set_tests_properties(Layer2_Regression Layer3_WholeProgram Layer5_CodeGen Layer6_Dialect Layer6_Frontend_Dialect_Pipeline Layer7_Sema
        PROPERTIES ENVIRONMENT "BT_COMPILER=$<TARGET_FILE:bt-compiler>;BT_OPT=$<TARGET_FILE:bt-opt>" SKIP_RETURN_CODE 77)

    # A clean 'check-all' must build every executable used by registered tests.
    #
    # The explicit list below was maintained by hand and went stale: targets
    # added later (unit_microkernel, ternary_bench, unit_kernel_contract,
    # unit_sdk_gemm_layout, unit_sdk_host) were missing, so on a clean build
    # ctest ran before its executables existed and reported "Not Run" rather
    # than a real pass or fail. Depending on `all` removes the class of bug -
    # a new test target can never again be silently left out.
    add_custom_target(check-all
        COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
        DEPENDS bt-compiler bt-opt unit_core unit_dense unit_memory unit_ast
            unit_mlir unit_fp_sparse unit_libbt unit_bt_opt unit_frontend_fp_sparse
            unit_c_abi_ffi unit_bigint_symplectic unit_xbt_hw unit_phase1
            unit_phase3 unit_universal unit_math_core unit_numeric_oracle unit_fp_foundation unit_ir_program unit_core_lowering
            unit_audit_regressions bt_hijack_fixture
            unit_microkernel ternary_bench unit_kernel_contract
            unit_xbt_conformance unit_xbt_selector unit_backend_registry
    )
endif()

# Versioned embedding SDK. GPU tools/headers are optional and are not fetched
# implicitly. The runtime resolves Vulkan on demand; CPU use needs no loader.
option(BT_BUILD_SDK "Build the in-process compiler/runtime SDK" ON)
option(BT_SDK_VULKAN "Build optional Vulkan compute runtime" OFF)
if(BT_BUILD_SDK)
    add_library(TriumMath INTERFACE)
    set_target_properties(TriumMath PROPERTIES EXPORT_NAME Math)
    target_compile_features(TriumMath INTERFACE cxx_std_20)
    target_include_directories(TriumMath INTERFACE "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>" $<INSTALL_INTERFACE:include>)
    add_library(TriumXBT::Math ALIAS TriumMath)
    add_library(trium_sdk SHARED lib/SDK/sdk.cpp lib/SDK/VulkanRuntime.cpp lib/SDK/NativeRuntime.cpp lib/SDK/NumericalRuntime.cpp)
    if(BT_HAS_MARCH_NATIVE)
        target_compile_options(trium_sdk PRIVATE -march=native)
    endif()
    target_include_directories(trium_sdk PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/." PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>" $<INSTALL_INTERFACE:include>)
    target_link_libraries(trium_sdk PRIVATE Threads::Threads ${CMAKE_DL_LIBS})
    set_target_properties(trium_sdk PROPERTIES EXPORT_NAME SDK VERSION 1.2.0 SOVERSION 1 CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON)
    add_library(TriumXBT::SDK ALIAS trium_sdk)
    if(BT_SDK_VULKAN)
        find_path(BT_VULKAN_INCLUDE_DIR vulkan/vulkan.h REQUIRED
            HINTS "${CMAKE_CURRENT_SOURCE_DIR}/tools/.toolchains/include")
        target_include_directories(trium_sdk PRIVATE ${BT_VULKAN_INCLUDE_DIR})
        target_compile_definitions(trium_sdk PRIVATE BT_SDK_HAS_VULKAN=1)
    endif()
    include(GNUInstallDirs)
    include(CMakePackageConfigHelpers)
    install(TARGETS trium_sdk TriumMath EXPORT TriumXBTTargets LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
    install(DIRECTORY include/Ternary/Math include/Ternary/SDK DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/Ternary)
    install(FILES ternary_wide.hpp ternary_fp.hpp ternary_core.hpp DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/Ternary/Detail)
    install(FILES LICENSE DESTINATION ${CMAKE_INSTALL_DATADIR}/TriumXBT)
    install(EXPORT TriumXBTTargets NAMESPACE TriumXBT:: DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/TriumXBT)
    configure_package_config_file(cmake/TriumXBTConfig.cmake.in ${CMAKE_CURRENT_BINARY_DIR}/TriumXBTConfig.cmake INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/TriumXBT)
    write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/TriumXBTConfigVersion.cmake VERSION 1.2.0 COMPATIBILITY SameMajorVersion)
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/TriumXBTConfig.cmake ${CMAKE_CURRENT_BINARY_DIR}/TriumXBTConfigVersion.cmake DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/TriumXBT)
    if(BT_BUILD_TESTS)
        add_executable(unit_sdk_host test/SDK/host.cpp)
        target_link_libraries(unit_sdk_host PRIVATE trium_sdk)
        add_test(NAME SDK_Host_Ownership COMMAND unit_sdk_host)
        # Pins the operand layout of bt_sdk_gemm_ternary. A is row-major and B
        # is word-plane-major; the difference is invisible for k <= 64, so this
        # deliberately covers k > 64 with n > 1, and asserts the row-major B
        # layout gives a DIFFERENT answer (otherwise the test is vacuous).
        add_executable(unit_sdk_gemm_layout test/SDK/gemm_layout.cpp)
        target_link_libraries(unit_sdk_gemm_layout PRIVATE trium_sdk)
        add_test(NAME SDK_Gemm_Layout COMMAND unit_sdk_gemm_layout)
        # Checks the Rust FFI binding against the real C ABI without needing a
        # Rust compiler: symbol existence, arity vs sdk.h, #[repr(C)] layouts.
        add_test(NAME SDK_Rust_FFI_ABI COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/test/rust_ffi_abi_check.py)
        set_tests_properties(SDK_Rust_FFI_ABI PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 300)
        # If cargo IS available, actually compile and test the Rust crate.
        # tools/install_toolchains.sh puts one in tools/.toolchains/bin, which
        # is searched explicitly because it is not on PATH by default.
        find_program(CARGO_EXECUTABLE cargo
            HINTS ${CMAKE_CURRENT_SOURCE_DIR}/tools/.toolchains/bin)
        if(CARGO_EXECUTABLE)
            add_test(NAME SDK_Rust_Build COMMAND
                ${CMAKE_CURRENT_SOURCE_DIR}/tools/run_rust_tests.sh ${CARGO_EXECUTABLE})
            set_tests_properties(SDK_Rust_Build PROPERTIES TIMEOUT 900)
        else()
            message(STATUS "cargo not found - the Rust crate will be ABI-checked only. "
                           "Run tools/install_toolchains.sh to get a toolchain.")
        endif()
        add_test(NAME SDK_Z3_Models COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/verify_z3_sdk_contracts.py)
        set_tests_properties(SDK_Z3_Models PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 120)
        add_test(NAME SDK_Contracts COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test/SDK/contracts.py $<TARGET_FILE:trium_sdk>)
        add_test(NAME SDK_Native COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test/SDK/native.py $<TARGET_FILE:trium_sdk>)
        set_tests_properties(SDK_Native PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 240)
        add_executable(unit_numerical test/SDK/numerical.cpp)
        target_link_libraries(unit_numerical PRIVATE trium_sdk TriumMath)
        add_test(NAME Numerical_CPP COMMAND unit_numerical)
        add_executable(benchmark_wide_math test/benchmark_wide_math.cpp)
        target_link_libraries(benchmark_wide_math PRIVATE trium_sdk TriumMath)
        add_executable(benchmark_arithmetic test/benchmark_arithmetic.cpp)
        target_link_libraries(benchmark_arithmetic PRIVATE trium_sdk TriumMath)
        add_test(NAME Numerical_Constants COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tools/verify_cordic_constants.py)
        add_test(NAME Numerical_Z3_Models COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/verify_z3_numerical.py)
        set_tests_properties(Numerical_Z3_Models PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 120)
        add_test(NAME Numerical_Exact_Reference COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test/numerical_reference.py $<TARGET_FILE:trium_sdk>)
        set_tests_properties(Numerical_Exact_Reference PROPERTIES TIMEOUT 240)
        add_executable(unit_sdk_math test/SDK/math.cpp)
        target_link_libraries(unit_sdk_math PRIVATE TriumMath)
        add_test(NAME SDK_Math_Interchange COMMAND unit_sdk_math)
        add_test(NAME FP_Foundation_Reference COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test/floating_foundation.py $<TARGET_FILE:trium_sdk>)
        set_tests_properties(FP_Foundation_Reference PROPERTIES TIMEOUT 240)
        add_test(NAME SDK_Integration COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test/SDK/integration.py $<TARGET_FILE:trium_sdk>)
        add_test(NAME SDK_Gemm_Layout_Python COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test/gemm_layout.py $<TARGET_FILE:trium_sdk>)
        set_tests_properties(SDK_Gemm_Layout_Python PROPERTIES TIMEOUT 240)
    endif()
endif()
if(BT_BUILD_TESTS AND BT_BUILD_SDK)
    add_test(NAME SDK_Vulkan_Offline COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test/SDK/vulkan.py $<TARGET_FILE:trium_sdk>)
    add_test(NAME SDK_Vulkan_Execution COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test/SDK/vulkan.py $<TARGET_FILE:trium_sdk> --execute)
    set_tests_properties(SDK_Vulkan_Offline SDK_Vulkan_Execution PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 240)
endif()
if(BT_BUILD_TESTS AND BT_BUILD_SDK)
    if(TARGET check-all)
        add_dependencies(check-all trium_sdk unit_sdk_math unit_sdk_host unit_numerical)
    endif()
    if(BT_ENABLE_ASAN AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        set_property(TEST SDK_Contracts SDK_Native Numerical_Exact_Reference FP_Foundation_Reference SDK_Integration SDK_Gemm_Layout_Python SDK_Vulkan_Offline SDK_Vulkan_Execution APPEND PROPERTY ENVIRONMENT
            "LD_PRELOAD=${BT_ASAN_RUNTIME}:${BT_CXX_ABI_RUNTIME}" "ASAN_OPTIONS=detect_leaks=0")
    endif()
endif()
if(BT_BUILD_TESTS AND BT_BUILD_SDK)
    add_test(NAME SDK_Relocated_Consumers COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test/SDK/package.py ${CMAKE_CURRENT_BINARY_DIR})
    if(BT_ENABLE_ASAN AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        set_property(TEST SDK_Relocated_Consumers APPEND PROPERTY ENVIRONMENT "LD_PRELOAD=${BT_ASAN_RUNTIME}:${BT_CXX_ABI_RUNTIME}" "ASAN_OPTIONS=detect_leaks=0")
    endif()
endif()
