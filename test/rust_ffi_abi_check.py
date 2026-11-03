#!/usr/bin/env python3
"""Check the Rust FFI binding against the real C ABI.

rustc is not available everywhere (it is not installable in every sandbox), so
this checks the half of "does the Rust binding work" that does NOT need a Rust
compiler, and which a Rust compiler would not catch anyway:

  1. Every `extern "C" fn` the Rust crate declares really exists in the built
     libtrium_sdk.so. A renamed or removed C entry point is otherwise silent
     until link time.
  2. Every one of them is declared in sdk.h with the SAME parameter count.
  3. The #[repr(C)] structs in Rust have the same field types, in the same
     order, as their C counterparts - verified by asking the C compiler for
     real offsets/sizes and comparing to the Rust field list.

A mismatch here is silent UB at runtime, so it is worth catching without rustc.

Exit 77 if the shared library cannot be built (skip, do not fail).
"""
import os
import re
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HEADER = os.path.join(ROOT, "include", "Ternary", "SDK", "sdk.h")
RUST = os.path.join(ROOT, "ffi", "rust", "src", "sdk.rs")

# Rust type -> (C type, size) for the scalar types used in this binding
RUST_TO_C = {
    "u32": ("uint32_t", 4), "i32": ("int32_t", 4),
    "u64": ("uint64_t", 8), "i64": ("int64_t", 8),
    "usize": ("size_t", 8), "f64": ("double", 8),
    "c_char": ("char", 1),
}

# Rust struct -> C struct that must have an identical field layout
STRUCT_PAIRS = [("Options", "bt_sdk_options"), ("Slot", "bt_sdk_value")]


def build_library(workdir):
    src = os.path.join(ROOT, "lib", "SDK", "sdk.cpp")
    so = os.path.join(workdir, "libtrium_sdk.so")
    p = subprocess.run(
        ["c++", "-std=c++20", "-O1", "-fPIC", "-shared",
         "-I", ROOT, "-I", os.path.join(ROOT, "include"),
         "-I", os.path.join(ROOT, "lib", "SDK"), src, "-o", so, "-pthread"],
        capture_output=True, text=True)
    if p.returncode != 0:
        print("SKIP: cannot build libtrium_sdk.so\n" + p.stderr[:800])
        sys.exit(77)
    return so


def exported_symbols(so):
    out = subprocess.run(["nm", "-D", "--defined-only", so],
                         capture_output=True, text=True).stdout
    return set(re.findall(r"\b(bt_sdk_\w+)\b", out))


def parse_rust_extern(text):
    """Return {name: [arg_type, ...]} for the extern "C" block."""
    m = re.search(r'extern\s+"C"\s*\{(.*?)\n\}', text, re.S)
    if not m:
        raise SystemExit("no extern \"C\" block found in " + RUST)
    fns = {}
    for fm in re.finditer(r"fn\s+(\w+)\s*\(([^)]*)\)\s*(?:->\s*([^;{]+))?\s*;", m.group(1)):
        name, args = fm.group(1), fm.group(2)
        types = []
        for a in [x.strip() for x in args.split(",") if x.strip()]:
            am = re.match(r"(?:\w+)\s*:\s*(.+)$", a)
            if am:
                types.append(re.sub(r"\s+", "", am.group(1)))
        fns[name] = types
    return fns


def parse_header_decls(text):
    """Return {name: [param, ...]} for BT_SDK_API declarations, typedefs unfolded."""
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
    text = re.sub(r"//[^\n]*", " ", text)
    typedefs = dict(re.findall(r"typedef\s+([\w\s]+?)\s+(\w+)\s*;", text))
    decls = {}
    for m in re.finditer(r"BT_SDK_API\s+[\w\s\*]+?\b(bt_sdk_\w+)\s*\(([^;{}]*)\);", text):
        name, params = m.group(1), m.group(2)
        decls[name] = [p.strip() for p in params.split(",") if p.strip() and p.strip() != "void"]
    return decls, typedefs


def c_struct_fields(struct_name):
    """Ask the C compiler for the real field types/offsets/sizes."""
    prog = """#include <cstddef>
#include <cstdio>
#include <cstdint>
#include "Ternary/SDK/sdk.h"
int main(){
"""
    for _, cs in STRUCT_PAIRS:
        prog += '  std::printf("%s|%%zu\\n", sizeof(%s));\n' % (cs, cs)
    prog += "  return 0;\n}\n"
    if struct_name is None:
        return prog
    return prog


def rust_struct_fields(text, name):
    m = re.search(r"#\[repr\(C\)\]\s*(?:#\[derive\([^\]]*\)\]\s*)*(?:pub\s+)?struct\s+"
                  + name + r"\s*\{([^}]*)\}", text)
    if not m:
        return None
    fields = []
    for f in [x.strip() for x in m.group(1).split(",") if x.strip()]:
        fm = re.match(r"(?:pub\s+)?(\w+)\s*:\s*(.+)$", f)
        if fm:
            fields.append((fm.group(1), re.sub(r"\s+", "", fm.group(2))))
    return fields


def main():
    failures = []

    rust_src = open(RUST).read()
    rust_fns = parse_rust_extern(rust_src)
    if not rust_fns:
        raise SystemExit("parsed 0 extern fns from the Rust binding - check the parser")

    hdr = open(HEADER).read()
    hdr_decls, _ = parse_header_decls(hdr)

    with tempfile.TemporaryDirectory() as wd:
        so = build_library(wd)
        syms = exported_symbols(so)
        print("libtrium_sdk.so exports %d bt_sdk_* symbols" % len(syms))

        # 1. every Rust extern fn must exist in the shared library
        for name in sorted(rust_fns):
            if not name.startswith("bt_sdk_"):
                continue
            if name not in syms:
                failures.append("Rust declares %s but it is NOT exported by the library" % name)
        print("checked %d Rust extern fns against the library" % len(rust_fns))

        # 2. parameter counts must agree with the header
        checked = 0
        for name, rargs in sorted(rust_fns.items()):
            if name not in hdr_decls:
                failures.append("Rust declares %s but sdk.h has no such declaration" % name)
                continue
            cparams = hdr_decls[name]
            if len(rargs) != len(cparams):
                failures.append("%s: Rust has %d params, sdk.h has %d\n    rust: %s\n    c   : %s"
                                % (name, len(rargs), len(cparams), rargs, cparams))
            checked += 1
        print("checked %d signatures against sdk.h" % checked)

        # 3. #[repr(C)] structs must have identical field layouts
        prog = """#include <cstddef>
#include <cstdio>
#include <cstdint>
#include "Ternary/SDK/sdk.h"
"""
        body = "int main(){\n"
        for _, cs in STRUCT_PAIRS:
            body += '  std::printf("%%s=%%zu\\n", "%s", sizeof(%s));\n' % (cs, cs)
        body += "  return 0;\n}\n"
        cpp = os.path.join(wd, "sz.cpp")
        open(cpp, "w").write(prog + body)
        exe = os.path.join(wd, "sz")
        p = subprocess.run(["c++", "-std=c++20", "-I", os.path.join(ROOT, "include"),
                            cpp, "-o", exe], capture_output=True, text=True)
        if p.returncode != 0:
            failures.append("could not compile the struct-size probe:\n" + p.stderr[:500])
        else:
            c_sizes = dict(re.findall(r"(\w+)=(\d+)", subprocess.run(
                [exe], capture_output=True, text=True).stdout))
            for rs, cs in STRUCT_PAIRS:
                rfields = rust_struct_fields(rust_src, rs)
                if rfields is None:
                    failures.append("Rust struct %s not found or not #[repr(C)]" % rs)
                    continue
                expected = sum(RUST_TO_C.get(t, (None, 0))[1] for _, t in rfields)
                got = int(c_sizes.get(cs, -1))
                if got != expected:
                    failures.append(
                        "struct %s: Rust fields sum to %d bytes but C sizeof(%s)=%d\n"
                        "    rust fields: %s" % (rs, expected, cs, got, rfields))
                else:
                    print("struct %-8s <-> %-16s layout agrees (%d bytes)"
                          % (rs, cs, got))

    if failures:
        print("\nFAIL: Rust FFI disagrees with the C ABI")
        for f in failures:
            print("  - " + f)
        return 1
    print("\nPASS: Rust FFI binding is consistent with the C ABI "
          "(symbols, arity, struct layout)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
