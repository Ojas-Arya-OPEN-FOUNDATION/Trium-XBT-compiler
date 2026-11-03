#!/usr/bin/env python3
"""End-to-end check that --gpu-map emits a SEMANTICALLY CORRECT kernel.

A kernel that is well-formed GLSL but computes the wrong thing is worse than
no kernel at all, because it looks like it worked. This closes that gap:

  1. Build ground truth: strip the host print(...) calls out of the .bt source,
     append one print per test input, and run that on the CPU (JIT). Same IR,
     different backend.
  2. Compile the named function to GLSL with --gpu-map.
  3. Transliterate the emitted GLSL back to C++ (the emitted arithmetic is
     plain C-like int64_t), compile it, and run it on the same inputs.
  4. Compare. Any mismatch is a real bug in the GLSL backend.

This does NOT validate GLSL syntax (that needs glslangValidator). It validates
the part that a shader compiler cannot: that the emitted kernel is the user's
program.

Usage:
    gpu_map_test.py <bt-compiler> <source.bt> <function> --inputs=a,b,c
"""
import argparse
import os
import re
import subprocess
import sys
import tempfile

PRELUDE = """#include <cstdint>
#include <cstdio>
#include <initializer_list>
typedef long long int64_t_glsl;
"""

DRIVER = """
int main() {
    for (long long v : INPUTS) std::printf("%lld\\n", bt_ENTRY(v));
    return 0;
}
"""


def run(cmd):
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise SystemExit("command failed: %s\n%s" % (" ".join(cmd), p.stderr.strip()))
    return p.stdout


def cpu_truth(compiler, source, function, inputs, workdir):
    """Ground truth = the CPU running the SAME function on the SAME inputs."""
    src = open(source).read()
    body = "\n".join(
        ln for ln in src.splitlines()
        if not (ln.strip().startswith("print(") and ln.rstrip().endswith(");"))
    )
    probe = os.path.join(workdir, "probe.bt")
    open(probe, "w").write(
        body + "\n\n" + "".join("print(%s(%d));\n" % (function, i) for i in inputs))
    out = run([compiler, "run", probe])
    vals = [int(m.group(1)) for m in re.finditer(r"^\s*(-?\d+)\s+\[trits:", out, re.M)]
    if len(vals) != len(inputs):
        raise SystemExit("expected %d CPU results, parsed %d; output was:\n%s"
                         % (len(inputs), len(vals), out))
    return vals


def emit_glsl(compiler, source, function, out_path):
    run([compiler, source, "--gpu-map=" + function, "-o", out_path])
    return open(out_path).read()


def glsl_to_cpp(glsl, entry, inputs):
    """Strip GLSL-only syntax; the emitted function bodies are valid C++."""
    lines, keep, depth = [], True, 0
    for ln in glsl.splitlines():
        s = ln.strip()
        if s.startswith("#") or s.startswith("layout("):
            continue
        if s.startswith("void main()"):
            keep = False
        if keep:
            lines.append(ln)
        else:
            depth += ln.count("{") - ln.count("}")
            if depth <= 0:
                keep = True
    body = "\n".join(lines)
    driver = DRIVER.replace("INPUTS", "{" + ",".join(str(i) for i in inputs) + "}")
    driver = driver.replace("bt_ENTRY", "bt_" + re.sub(r"\W", "_", entry))
    return PRELUDE + body + driver


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("compiler")
    ap.add_argument("source")
    ap.add_argument("function")
    ap.add_argument("--inputs", required=True, help="comma-separated integers")
    a = ap.parse_args()

    inputs = [int(x) for x in a.inputs.split(",") if x.strip()]
    with tempfile.TemporaryDirectory() as wd:
        truth = cpu_truth(a.compiler, a.source, a.function, inputs, wd)
        comp = os.path.join(wd, "kernel.comp")
        glsl = emit_glsl(a.compiler, a.source, a.function, comp)
        cpp = os.path.join(wd, "kernel.cpp")
        open(cpp, "w").write(glsl_to_cpp(glsl, a.function, inputs))
        exe = os.path.join(wd, "kernel")
        p = subprocess.run(["c++", "-std=c++17", "-O2", cpp, "-o", exe],
                           capture_output=True, text=True)
        if p.returncode != 0:
            print("FAIL: transliterated kernel did not compile:\n" + p.stderr)
            print("--- emitted GLSL ---\n" + glsl)
            return 1
        got = [int(x) for x in run([exe]).split()]

    if got != truth:
        print("FAIL: GLSL kernel disagrees with the CPU")
        for i, (g, t) in enumerate(zip(got, truth)):
            if g != t:
                print("  input %d: kernel=%d cpu=%d" % (inputs[i], g, t))
        return 1

    print("PASS: --gpu-map=%s matches the CPU on %d inputs" % (a.function, len(truth)))
    print("  inputs : %s" % inputs)
    print("  results: %s" % truth)
    return 0


if __name__ == "__main__":
    sys.exit(main())
