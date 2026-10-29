#!/usr/bin/env python3
"""Check proof sources with native Rocq, or an explicitly selected jsCoq kernel.
A jsCoq CLI error may exit 0: inspect diagnostics AND a terminal proof marker.
jsCoq compatibility changes the Stdlib import prefix only, never proof bodies.
"""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

parser = argparse.ArgumentParser()
parser.add_argument('--jscoq', default=os.environ.get('BT_JSCOQ_CLI'))
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
files = [('BalancedTernaryFoundations.v', 'verlet_kdk_invertible'),
         ('CoreMath.v', 'TriumCore.core_math_complete'),
         ('ImplementationBridge.v', 'ImplementationBridge.implementation_bridge_complete'),
         ('CoreLowering.v', 'CoreLowering.core_lowering_complete')]
rocq = shutil.which('rocq')
coqc = shutil.which('coqc')
if args.jscoq:
    cli = Path(args.jscoq).resolve()
    package = json.loads((cli.parent.parent / 'package.json').read_text())
    if package['name'] != 'jscoq' or package['version'] != '0.17.1':
        raise RuntimeError('This compatibility path is pinned to jscoq 0.17.1')
    print('ENGINE: jsCoq 0.17.1 (Coq kernel; NOT native Rocq 9 validation)', flush=True)
elif rocq or coqc:
    print('ENGINE: native ' + (rocq or coqc), flush=True)
else:
    print('SKIP: install Rocq or explicitly set BT_JSCOQ_CLI to jscoq 0.17.1 CLI')
    sys.exit(77)

with tempfile.TemporaryDirectory(prefix='trium-proofs-') as directory:
    for filename, theorem in files:
        source = (root / 'proofs/rocq' / filename).read_text()
        if re.search(r'\b(Admitted|admit|Axiom|Parameter)\b', source):
            raise AssertionError(f'{filename}: unchecked assumptions are not allowed')
        # Print Assumptions checks transitive assumptions of all declared results,
        # not just the terminal marker. Checking the final source declaration alone
        # would miss an axiom dependency in an earlier theorem.
        names, scopes = [], []
        for match in re.finditer(r'^\s*(Module|Section|End|Theorem|Lemma|Example)\s+(\w+)', source, re.M):
            kind, name = match.groups()
            if kind in ('Module', 'Section'):
                scopes.append((name, kind == 'Module'))
            elif kind == 'End':
                assert scopes and scopes[-1][0] == name
                scopes.pop()
            else:
                names.append('.'.join([n for n, module in scopes if module] + [name]))
        source += '\n' + '\n'.join('Print Assumptions ' + name + '.' for name in names)
        source += '\nCheck ' + theorem + '.\n'
        if args.jscoq:
            source = source.replace('From Stdlib ', 'From Coq ')
        path = Path(directory) / filename
        path.write_text(source)
        command = ([shutil.which('node') or 'node', str(cli), 'run', '-l', str(path)]
                   if args.jscoq else ([rocq, 'compile', str(path)] if rocq else [coqc, str(path)]))
        result = subprocess.run(command, text=True, capture_output=True, timeout=300)
        output = result.stdout + result.stderr
        if result.returncode or re.search(r'\[(?:Error|Exception)\]|^Error:|^Anomaly:', output, re.M):
            print(output)
            raise AssertionError(f'{filename}: proof checker failed')
        if args.jscoq and (theorem not in output or output.count('Closed under the global context') < len(names)):
            print(output)
            raise AssertionError(f'{filename}: missing completed proof/assumption markers')
        if 'Axioms:' in output:
            print(output)
            raise AssertionError(f'{filename}: unexpected assumptions')
        print(f'PASS {filename}: proof bodies checked; all {len(names)} results have no axioms', flush=True)
