#!/usr/bin/env python3
"""
LLVM-Style lit + FileCheck Test Runner for Balanced Ternary Compiler
Supports:
  RUN: <command>
  CHECK: <pattern>
  CHECK-NEXT: <pattern>
  CHECK-NOT: <pattern>
"""

import sys
import os
import re
import subprocess
import tempfile
import argparse

def run_filecheck(output_text, check_lines, test_path):
    output_lines = output_text.splitlines()
    curr_line_idx = 0
    total_output = len(output_lines)

    for ch_type, pattern, line_num in check_lines:
        regex = re.compile(re.escape(pattern).replace(r'\{', '{').replace(r'\}', '}'))
        # If pattern has regex tokens or exact match
        found = False

        if ch_type == "CHECK":
            for i in range(curr_line_idx, total_output):
                if pattern in output_lines[i] or regex.search(output_lines[i]):
                    curr_line_idx = i + 1
                    found = True
                    break
            if not found:
                return False, f"FileCheck error: CHECK '{pattern}' (line {line_num}) not found after output line {curr_line_idx}"

        elif ch_type == "CHECK-NEXT":
            if curr_line_idx < total_output:
                if pattern in output_lines[curr_line_idx] or regex.search(output_lines[curr_line_idx]):
                    curr_line_idx += 1
                    found = True
            if not found:
                actual = output_lines[curr_line_idx] if curr_line_idx < total_output else "<EOF>"
                return False, f"FileCheck error: CHECK-NEXT '{pattern}' (line {line_num}) failed. Got: '{actual}'"

        elif ch_type == "CHECK-NOT":
            # Search until end of output or next check for forbidden pattern
            for i in range(curr_line_idx, total_output):
                if pattern in output_lines[i] or regex.search(output_lines[i]):
                    return False, f"FileCheck error: CHECK-NOT '{pattern}' (line {line_num}) unexpectedly found at line {i+1}: '{output_lines[i]}'"

    return True, "PASSED"

def run_test_file(test_file, root_dir):
    with open(test_file, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()

    run_commands = []
    check_lines = []

    for idx, line in enumerate(content.splitlines(), start=1):
        line_clean = line.strip()
        # Look for RUN: directives
        run_match = re.search(r'(?:;|//|#)\s*RUN:\s*(.+)$', line_clean)
        if run_match:
            run_commands.append((run_match.group(1).strip(), idx))
            continue

        # Look for CHECK directives
        check_not_m = re.search(r'(?:;|//|#)\s*CHECK-NOT:\s*(.+)$', line_clean)
        if check_not_m:
            check_lines.append(("CHECK-NOT", check_not_m.group(1).strip(), idx))
            continue

        check_next_m = re.search(r'(?:;|//|#)\s*CHECK-NEXT:\s*(.+)$', line_clean)
        if check_next_m:
            check_lines.append(("CHECK-NEXT", check_next_m.group(1).strip(), idx))
            continue

        check_m = re.search(r'(?:;|//|#)\s*CHECK:\s*(.+)$', line_clean)
        if check_m:
            check_lines.append(("CHECK", check_m.group(1).strip(), idx))
            continue

    if not run_commands:
        return True, "SKIPPED (no RUN lines)"

    # Execute RUN commands in a temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_out = os.path.join(tmpdir, "out.tmp")
        bt_compiler_path = os.path.join(root_dir, "bt-compiler")
        bt_opt_path = os.path.join(root_dir, "bt-opt")

        for cmd, line_num in run_commands:
            # Substitute variables with shell quotes for whitespace safety
            import shlex
            cmd_expanded = cmd.replace("%s", shlex.quote(test_file))
            cmd_expanded = cmd_expanded.replace("%t", shlex.quote(tmp_out))
            cmd_expanded = cmd_expanded.replace("%bt-compiler", shlex.quote(bt_compiler_path))
            cmd_expanded = cmd_expanded.replace("%bt-opt", shlex.quote(bt_opt_path))

            # If pipeline contains FileCheck, we split and run internal FileCheck
            if "| FileCheck" in cmd_expanded or "| %FileCheck" in cmd_expanded:
                exec_cmd = re.split(r'\|\s*%?FileCheck', cmd_expanded)[0].strip()
                try:
                    res = subprocess.run(
                        exec_cmd,
                        shell=True,
                        cwd=root_dir,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        timeout=15
                    )
                    output = res.stdout
                except subprocess.TimeoutExpired:
                    return False, f"Timeout executing: {exec_cmd}"
                except Exception as e:
                    return False, f"Execution failed: {e}"

                # Run internal FileCheck on output
                ok, msg = run_filecheck(output, check_lines, test_file)
                if not ok:
                    return False, f"Command '{exec_cmd}' failed check:\n{msg}\nOutput:\n{output[:500]}"
            else:
                # Direct command execution
                try:
                    res = subprocess.run(
                        cmd_expanded,
                        shell=True,
                        cwd=root_dir,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        timeout=15
                    )
                    if res.returncode != 0 and "not" not in cmd_expanded:
                        return False, f"Command exited with {res.returncode}: {cmd_expanded}\n{res.stdout}"
                except Exception as e:
                    return False, f"Error running: {e}"

    return True, "PASSED"

def main():
    parser = argparse.ArgumentParser(description="LLVM Lit Test Runner for Balanced Ternary Compiler")
    parser.add_argument("paths", nargs="+", help="Test files or directories to scan")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

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

if __name__ == "__main__":
    main()
