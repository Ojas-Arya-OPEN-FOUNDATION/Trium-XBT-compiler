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

    add_executable(unit_dense test/Unit/test_unit_dense.cpp)
    target_include_directories(unit_dense PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

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

    # CTest Registration: Layer 1 Unit Tests
    add_test(NAME Layer1_Unit_Core COMMAND unit_core)
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

    # CTest Registration: Layers 2-7
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
    add_test(NAME CUDA_Offline_Validation COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/test/cuda_map_test.py $<TARGET_FILE:bt-compiler>)
    add_test(NAME CUDA_Device_Qualification COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/test/cuda_map_test.py $<TARGET_FILE:bt-compiler> --device)
    set_tests_properties(Core_Backend_Regression CUDA_Offline_Validation CUDA_Device_Qualification PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 180)
    add_executable(unit_ir_program test/Unit/test_unit_ir_program.cpp)
    add_test(NAME Core_Program_IR COMMAND unit_ir_program)
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
        set_property(TEST Phase1_Python_FFI Runtime_ABI_Client APPEND PROPERTY ENVIRONMENT
            "LD_PRELOAD=${BT_ASAN_RUNTIME}:${BT_CXX_ABI_RUNTIME}" "ASAN_OPTIONS=detect_leaks=0")
    endif()
    set_tests_properties(Layer2_Regression Layer3_WholeProgram Layer5_CodeGen Layer6_Dialect Layer7_Sema
        PROPERTIES ENVIRONMENT "BT_COMPILER=$<TARGET_FILE:bt-compiler>;BT_OPT=$<TARGET_FILE:bt-opt>" SKIP_RETURN_CODE 77)

    # A clean 'check-all' builds every executable used by registered tests.
    add_custom_target(check-all
        COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
        DEPENDS bt-compiler bt-opt unit_core unit_dense unit_memory unit_ast
            unit_mlir unit_fp_sparse unit_libbt unit_bt_opt unit_frontend_fp_sparse
            unit_c_abi_ffi unit_bigint_symplectic unit_xbt_hw unit_phase1
            unit_phase3 unit_universal unit_math_core unit_numeric_oracle unit_ir_program unit_core_lowering
    )
endif()
