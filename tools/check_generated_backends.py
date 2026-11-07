#!/usr/bin/env python3
"""
Verify the generated backend packages (docs/SCALING_PLAN.md 6.2 / 6.4).

The claim being tested is the Track B exit criterion:

    A target can be added by writing a description file with ZERO compiler
    edits; the generator produces a skeleton that COMPILES and FAILS its
    conformance suite.

So the expected outcome is deliberately a FAILURE -- a generated backend that
passed its own suite would mean the generator emitted a stub and called it
done. What must hold:

  1. every package compiles cleanly
  2. every conformance suite runs and reports outstanding obligations
  3. the target-description section PASSES (facts loaded from the YAML)
  4. the lowering section PASSES (it reports unsupported rather than
     emitting a silent stub)
  5. the number of obligations equals the number of declared native ops

Run:  python3 tools/check_generated_backends.py
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

PASS = 0
FAIL = 0


def report(name, ok, detail=""):
    global PASS, FAIL
    if ok:
        PASS += 1
        print(f"    [PASS] {name}{('  ' + detail) if detail else ''}")
    else:
        FAIL += 1
        print(f"    [FAIL] {name}  {detail}")


def main() -> int:
    print("=" * 78)
    print("  Generated backend packages (Track B)")
    print("=" * 78)

    # Import the generators so we use the same code the build does.
    import gen_backend
    import gen_targets

    targets = gen_targets.load_targets()
    print(f"\n  {len(targets)} targets: "
          f"{', '.join(t['name'] for t in targets)}\n")

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        for t in targets:
            gen_backend.generate(t, out)

        for t in targets:
            name = t["name"]
            d = out / name
            print(f"  {name}")

            srcs = [str(d / f"{name}_backend.cpp"),
                    str(d / f"test_{name}_conformance.cpp")]
            exe = d / f"conf_{name}"

            # ---- 1. compiles -------------------------------------------
            cp = subprocess.run(
                ["c++", "-std=c++20", f"-I{ROOT}", f"-I{ROOT}/include",
                 "-o", str(exe)] + srcs,
                capture_output=True, text=True)
            if cp.returncode != 0:
                report("compiles", False, cp.stderr.strip().splitlines()[-1]
                       if cp.stderr else "compile failed")
                continue
            report("compiles", True)

            # ---- 2. runs and reports obligations -----------------------
            rp = subprocess.run([str(exe)], capture_output=True, text=True)
            if rp.returncode == 0:
                report("suite fails as designed", False,
                       "it PASSED -- the generator emitted a stub and called "
                       "it done")
                continue
            report("suite fails as designed", True,
                   f"exit {rp.returncode}")

            out_text = rp.stdout

            # ---- 3. target facts load from the YAML --------------------
            # No [TODO] under the "target description" or "registration"
            # headings: those must pass outright.
            def section(text, heading):
                lines, on = [], False
                for ln in text.splitlines():
                    if ln.strip() == heading:
                        on = True
                        continue
                    if on and ln.startswith("  ") and not ln.startswith("    "):
                        break
                    if on:
                        lines.append(ln)
                return "\n".join(lines)

            desc = section(out_text, "  target description")
            reg = section(out_text, "  registration")
            report("target facts load from the YAML",
                   "[TODO]" not in desc, "description section is clean")
            report("backend self-registers",
                   "[TODO]" not in reg, "registration section is clean")

            # ---- 4. lowering refuses rather than stubbing --------------
            low = section(out_text, "  lowering")
            report("lowering reports unsupported, never a silent stub",
                   "[TODO]" not in low)

            # ---- 5. obligations match declared native ops --------------
            declared = t["native_ops"] or ["add", "negate", "dot"]
            n_todo = out_text.count("[TODO]")
            report("obligations match the declared native ops",
                   n_todo == len(declared),
                   f"{n_todo} outstanding, {len(declared)} ops declared")

    print("\n" + "=" * 78)
    print(f"  {PASS} passed, {FAIL} failed")
    print("=" * 78)
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
