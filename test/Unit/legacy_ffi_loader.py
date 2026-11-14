#!/usr/bin/env python3
"""Legacy Python FFI hardening checks (findings S-01 and S-02).

S-01: importing ffi/python/bt_ffi.py must not load a shared library from the
      current working directory.
S-02: mismatched buffer lengths must raise ValueError instead of reading past a
      ctypes buffer.

The import runs in a child process because the dynamic loader resolves
LD_LIBRARY_PATH once at process start.
"""
import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

runtime = Path(sys.argv[1]).resolve()
hijack = Path(sys.argv[2]).resolve()
module_path = Path(sys.argv[3]).resolve()


def child_checks(work: Path, marker: Path) -> None:
    spec = importlib.util.spec_from_file_location("bt_ffi_audit", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert not marker.exists(), "S-01: libbt_runtime.so from the working directory was loaded"

    def expect_value_error(call, label):
        try:
            call()
        except ValueError:
            return
        raise AssertionError(f"S-02: {label} did not raise ValueError")

    expect_value_error(lambda: module.xbt_rmsnorm([1.0] * 4, weight=[1.0]),
                       "mismatched RMSNorm weight")
    expect_value_error(lambda: module.csr_spmv(2, 4, [0, 1, 0], [1], [1], [1, 1, 1, 1]),
                       "row_ptr that does not end at nnz")
    expect_value_error(lambda: module.csr_spmv(2, 4, [0, 1, 1], [1], [1, 0], [1, 1, 1, 1]),
                       "wrong signs buffer length")
    expect_value_error(lambda: module.csr_spmv(2, 4, [0, 1, 1], [1], [1], [1, 1, 1]),
                       "short input vector")
    expect_value_error(lambda: module.xbt_sgemm([1.0], [1.0], 1, 2, 1), "short GEMM input")

    # An ordinary call still works.
    assert len(module.xbt_rmsnorm([3.0, 3.0], None, 1e-5)) == 2


if os.environ.get("BT_FFI_AUDIT_CHILD") == "1":
    child_checks(Path.cwd(), Path(os.environ["BT_HIJACK_MARKER"]))
    print("PASS: legacy FFI loader and length validation (S-01, S-02)")
    raise SystemExit(0)

with tempfile.TemporaryDirectory(prefix="bt-legacy-ffi-") as directory:
    work = Path(directory)
    marker = work / "hijack-marker"
    shutil.copy(hijack, work / "libbt_runtime.so")
    shutil.copy(hijack, work / "libbt_runtime.dylib")

    environment = dict(os.environ)
    environment.pop("BT_RUNTIME_LIBRARY", None)
    environment["BT_FFI_AUDIT_CHILD"] = "1"
    environment["BT_HIJACK_MARKER"] = str(marker)
    # The real runtime is discoverable through the loader search path only; the
    # working directory (which holds the hijack fixture) must not be consulted.
    for var in ("LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH"):
        existing = environment.get(var, "")
        environment[var] = f"{runtime.parent}{os.pathsep}{existing}" if existing else str(runtime.parent)

    result = subprocess.run([sys.executable, str(Path(__file__).resolve()), str(runtime), str(hijack),
                             str(module_path)], cwd=work, env=environment,
                            capture_output=True, text=True, timeout=120)
    assert not marker.exists(), "S-01: shared library from the working directory was loaded"
    if result.returncode != 0:
        sys.stdout.write(result.stdout)
        sys.stderr.write(result.stderr)
        raise SystemExit(result.returncode)
    sys.stdout.write(result.stdout)
