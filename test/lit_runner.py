#!/usr/bin/env python3
"""
LLVM-Style lit + FileCheck Test Runner for Balanced Ternary Compiler
Supports:
  RUN: <command>
  CHECK: <pattern>
  CHECK-NEXT: <pattern>
  CHECK-LABEL: <pattern>
  CHECK-NOT: <pattern>

Verification semantics (deliberate, documented subset of upstream FileCheck):
  * Patterns are plain substrings matched against whole output lines.
  * Checks are executed in file order (mandatory ordering, no fuzzy jumps).
  * CHECK-LABEL anchors subsequent checks after its match in the output and
    re-anchors scoping for each new label.
  * CHECK-NOT applies only to the region between the preceding check's match
    and the match of the following check (or to EOF when it is last).
  * Check directives are scoped per RUN: they belong to the most recent RUN
    line *above them that pipes into FileCheck*. Multiple piped RUN commands
    therefore each verify only their own CHECK group, and a piped RUN without
    any following CHECK directives is a failure (an empty FileCheck check
    list never verifies anything).
  * Non-piped RUN commands are still executed and MUST exit 0 (a command that
    fails for any reason fails the test - there is no substring-based bypass).
"""

import sys
import os
import re
import shlex
import subprocess
import tempfile
import argparse

FILECHECK_SPLIT_RE = re.compile(r'\s*\|\s*%?FileCheck\b')
# NB: a token directly preceded by '%' (printf %%-escape) is left untouched.
VAR_SUB_RE = re.compile(r'(?<!%)%(bt-compiler|bt-opt|t[0-9]?|s)(?![A-Za-z0-9_])')


def substitute_vars(cmd, test_file, tmpdir, root_dir):
    """Replaces lit-style variables (%s, %t, %t0..%t9, %bt-compiler, %bt-opt).

    Longer tokens are resolved first and a word-boundary guard keeps %t1
    distinct from %t. Paths are shell-quoted for whitespace safety.
    """
    def path_for(var):
        if var == 's':
            return shlex.quote(os.path.abspath(test_file))
        if var == 'bt-compiler':
            return shlex.quote(os.path.join(root_dir, 'bt-compiler'))
        if var == 'bt-opt':
            return shlex.quote(os.path.join(root_dir, 'bt-opt'))
        # %t / %tN -> per-run temp file
        return shlex.quote(os.path.join(tmpdir, f'out_{var}.tmp' if var != 't' else 'out.tmp'))
    return VAR_SUB_RE.sub(lambda m: path_for(m.group(1)), cmd)


def parse_checks_and_runs(content):
    """Splits the test file into RUN commands with their associated CHECK lists.

    Each check directive attaches to the most recent RUN above it that pipes
    into FileCheck. Returns (runs, error) where runs is a list of dicts:
      {line, cmd_raw, is_file_check, checks: [(kind, pattern, src_line)]}
    """
    runs = []
    current_filecheck_run = None

    for idx, line in enumerate(content.splitlines(), start=1):
        line_clean = line.strip()
        run_match = re.search(r'(?:;|//|#)\s*RUN:\s*(.+)$', line_clean)
        if run_match:
            cmd = run_match.group(1).strip()
            is_file_check = bool(FILECHECK_SPLIT_RE.search(cmd))
            new_run = {
                'line': idx,
                'cmd_raw': cmd,
                'is_file_check': is_file_check,
                'checks': [],
            }
            runs.append(new_run)
            current_filecheck_run = new_run if is_file_check else current_filecheck_run
            continue

        kind = None
        for directive in ('CHECK-LABEL', 'CHECK-NOT', 'CHECK-NEXT', 'CHECK'):
            m = re.search(r'(?:;|//|#)\s*' + re.escape(directive) + r':\s*(.+)$', line_clean)
            if m:
                pattern = m.group(1).strip()
                if current_filecheck_run is None:
                    return None, (
                        f"{directive} at file line {idx} appears before any RUN command "
                        f"that pipes into FileCheck"
                    )
                current_filecheck_run['checks'].append((directive, pattern, idx))
                kind = directive
                break
    return runs, None


def _find_sub(lines, pattern, start, end=None):
    stop = len(lines) if end is None else end
    for i in range(start, stop):
        if pattern in lines[i]:
            return i
    return None


def run_filecheck(output_text, checks, run_label):
    """Verifies ordered CHECK/NEXT/LABEL/NOT semantics against the output."""
    lines = output_text.splitlines()
    total = len(lines)
    cursor = 0

    i = 0
    while i < len(checks):
        kind, pattern, src_line = checks[i]

        if kind in ('CHECK', 'CHECK-LABEL'):
            pos = _find_sub(lines, pattern, cursor)
            if pos is None:
                tail = ' | '.join(lines[max(0, total - 3):])[:500]
                return False, (
                    f"{run_label}: {kind} '{pattern}' (file line {src_line}) not found after output "
                    f"line {cursor}. Output tail: {tail}"
                )
            cursor = pos + 1
            i += 1

        elif kind == 'CHECK-NEXT':
            if cursor < total and pattern in lines[cursor]:
                cursor += 1
                i += 1
            else:
                actual = lines[cursor] if cursor < total else '<EOF>'
                return False, (
                    f"{run_label}: CHECK-NEXT '{pattern}' (file line {src_line}) failed. "
                    f"Got output line {cursor}: '{actual}'"
                )

        else:  # CHECK-NOT (possibly a contiguous run of NOT directives)
            j = i
            while j < len(checks) and checks[j][0] == 'CHECK-NOT':
                j += 1
            # Region end = line index where the NEXT non-NOT check will match
            # (exclusive); region extends to EOF when no following check exists.
            region_end = total
            if j < len(checks):
                nk, npat, _ = checks[j]
                if nk in ('CHECK', 'CHECK-LABEL'):
                    pos = _find_sub(lines, npat, cursor)
                    if pos is not None:
                        region_end = pos
                elif nk == 'CHECK-NEXT':
                    region_end = min(cursor + 1, total)
            for k in range(i, j):
                _, notpat, notline = checks[k]
                if _find_sub(lines, notpat, cursor, region_end) is not None:
                    hit = _find_sub(lines, notpat, cursor, region_end)
                    return False, (
                        f"{run_label}: CHECK-NOT '{notpat}' (file line {notline}) unexpectedly found "
                        f"at output line {hit + 1}: \"{lines[hit]}\" (region lines {cursor + 1}..{region_end})"
                    )
            i = j

    return True, 'PASSED'


def run_test_file(test_file, root_dir):
    with open(test_file, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()

    if not re.search(r'(?:;|//|#)\s*RUN:', content):
        return True, 'SKIPPED (no RUN lines)'

    runs, parse_error = parse_checks_and_runs(content)
    if parse_error:
        return False, parse_error
    assert runs is not None

    rel = os.path.relpath(test_file, root_dir)

    with tempfile.TemporaryDirectory(prefix='bt_lit_') as tmpdir:
        for r in runs:
            run_label = f"{rel}: RUN line {r['line']}"
            cmd = substitute_vars(r['cmd_raw'], test_file, tmpdir, root_dir)

            if r['is_file_check']:
                exec_cmd = FILECHECK_SPLIT_RE.split(cmd, maxsplit=1)[0].strip()
                try:
                    res = subprocess.run(
                        exec_cmd,
                        shell=True,
                        cwd=root_dir,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        timeout=30,
                    )
                except subprocess.TimeoutExpired:
                    return False, f'{run_label}: timeout executing: {exec_cmd}'
                except Exception as e:  # noqa: BLE001 - report and fail test
                    return False, f'{run_label}: execution failed: {e}'

                # Shell-pipeline semantics: the pipeline status is that of the
                # LAST element (FileCheck), so a diagnostic-verification RUN may
                # legitimately have a nonzero producer status. The CHECK result
                # decides; only spawn/timeout failures abort above.
                if not r['checks']:
                    return False, (
                        f'{run_label}: RUN pipes into FileCheck but no CHECK directives '
                        f'follow it in the file (empty check list verifies nothing)'
                    )

                ok, msg = run_filecheck(res.stdout, r['checks'], run_label)
                if not ok:
                    return False, f"{msg}\n--- command: {exec_cmd} ---\n--- output (first 800 chars) ---\n{res.stdout[:800]}"
                if res.returncode != 0:
                    # Producer failed but its output satisfied every check.
                    # Surface the status so flaky producers don't hide silently.
                    print(f"    [note] {run_label}: producer exited {res.returncode} (output matched all checks)")
            else:
                try:
                    res = subprocess.run(
                        cmd,
                        shell=True,
                        cwd=root_dir,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        timeout=30,
                    )
                except subprocess.TimeoutExpired:
                    return False, f'{run_label}: timeout executing: {cmd}'
                except Exception as e:  # noqa: BLE001 - report and fail test
                    return False, f'{run_label}: execution failed: {e}'
                if res.returncode != 0:
                    return False, (
                        f'{run_label}: command exited with {res.returncode}: {cmd}\n'
                        f'Output:\n{res.stdout[:800]}'
                    )

    return True, 'PASSED'


def main():
    parser = argparse.ArgumentParser(description="LLVM Lit Test Runner for Balanced Ternary Compiler")
    parser.add_argument("paths", nargs="+", help="Test files or directories to scan")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    root_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

    test_files = []
    for p in args.paths:
        abs_p = os.path.abspath(p)
        if os.path.isdir(abs_p):
            for r, dirs, files in os.walk(abs_p):
                for f in sorted(files):
                    if f.endswith(('.bt', '.test', '.mlir', '.ll')):
                        test_files.append(os.path.join(r, f))
        elif os.path.isfile(abs_p):
            test_files.append(abs_p)

    print("=" * 70)
    print(f"  LLVM lit Test Runner: Scanning {len(test_files)} test suite files")
    print("=" * 70)

    passed = 0
    failed = 0
    skipped = 0

    for tf in test_files:
        rel_path = os.path.relpath(tf, root_dir)
        if args.verbose:
            print(f"  [RUNNING] {rel_path}")
        ok, msg = run_test_file(tf, root_dir)
        if "SKIPPED" in msg:
            skipped += 1
            print(f"  [SKIPPED] {rel_path}")
        elif ok:
            passed += 1
            print(f"  \033[32m[PASS]\033[0m    {rel_path}")
        else:
            failed += 1
            print(f"  \033[31m[FAIL]\033[0m    {rel_path}")
            print(f"    -> {msg}")

    print("=" * 70)
    print(f"  Test Results: {passed} PASSED, {failed} FAILED, {skipped} SKIPPED")
    print("=" * 70)

    sys.exit(1 if failed > 0 else 0)


if __name__ == '__main__':
    main()
