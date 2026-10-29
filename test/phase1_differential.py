#!/usr/bin/env python3
"""Compile actual emitted LLVM independently and compare to Python + interpreter.
This intentionally does not pretend to test the Clang/dlopen JIT wrapper.
"""
import math
import os
from pathlib import Path
import random
import re
import shutil
import subprocess
import sys
import tempfile
try:
    from llvmlite import binding as llvm
except ImportError:
    print('SKIP: llvmlite is required for independent LLVM execution')
    sys.exit(77)
compiler = str(Path(sys.argv[1]).resolve())
linker = shutil.which('c++')
if not linker or sys.platform != 'linux':
    print('SKIP: independent execution adapter currently requires Linux and a C++ linker')
    sys.exit(77)
llvm.initialize_native_target()
llvm.initialize_native_asmprinter()
tm = llvm.Target.from_default_triple().create_target_machine(reloc='pic')


def run(args):
    if os.environ.get('BT_TEST_CORE_OPT')=='1' and args[0]==compiler: args=[*args,'--core-opt']
    return subprocess.run(args, capture_output=True, text=True, timeout=30)


def emit(src, work):
    source = work/'program.bt'
    source.write_text(src)
    output = work/'program.ll'
    result = run([compiler, str(source), '--emit-llvm', '-o', str(output)])
    assert result.returncode == 0, result.stderr
    mod = llvm.parse_assembly(output.read_text())
    mod.verify()
    mod.triple = llvm.get_default_triple()
    mod.data_layout = str(tm.target_data)
    obj, shared = work/'program.o', work/'program.so'
    obj.write_bytes(tm.emit_object(mod))
    linked = run([linker, '-shared', str(obj), '-lm', '-o', str(shared)])
    assert linked.returncode == 0, linked.stderr
    return source, shared


def native(shared):
    # Isolated process: a regression cannot crash the test controller.
    return run([sys.executable, '-c',
                f'import ctypes; l=ctypes.CDLL({str(shared)!r}); raise SystemExit(l.bt_main())'])


rng = random.Random(0x584254)
expressions, expected = [], []
for _ in range(100):
    a,b,c = [rng.randint(-10000,10000) for _ in range(3)]
    expressions.extend([f'print(({a})+({b})*({c}));', f'print(({a})<=>({b}));'])
    expected.extend([a+b*c, (a>b)-(a<b)])
# Digit-wise logic oracle is independent of the C++ bitplane formulas.
def trit_logic(a,b,operation):
    result, place = 0,1
    while a or b:
        qa,ra = divmod(a,3)
        qb,rb = divmod(b,3)
        if ra == 2: qa,ra = qa+1,-1
        if rb == 2: qb,rb = qb+1,-1
        result += operation(ra,rb)*place
        a,b,place = qa,qb,place*3
    return result
for _ in range(100):
    a,b = [rng.randint(-100000,100000) for _ in range(2)]
    for operator, function in [('&', min), ('|', max), ('^', lambda x,y:x*y)]:
        expressions.append(f'print(({a}) {operator} ({b}));')
        expected.append(trit_logic(a,b,function))
expressions.extend(['print(2026277576509488133 + 0);', 'print(-2026277576509488133 + 1);'])
expected.extend([2026277576509488133,-2026277576509488132])
expressions.extend([
    'print(f(7)); func f(x: int) -> int { if (x > 0) { return x*6; } else { return 0; } }',
    'var outer = 5; if (1) { var outer = 9; } print(outer);',
    'var i = 0; while (i < 5) { i = i+1; } print(i);',
    'print(select(0, 100, select(-1, 1, 2, 3), 200));',
    'func side(x: int) -> int { print(x); return x; } print(select(-1, side(10), side(20), side(30)));',
    'func promoted(x: float) -> float { return x; } print(promoted(1));',
    'func ret_float() -> float { return 2; } print(ret_float());',
    'print(select(0.25, 1.5, 2, 3.5));',
    'func effect() { return print(7); } select(1, effect(), effect(), effect());',
    'var assigned:float = 0; assigned = 2; print(assigned * 2026277576509488133);',
    'var mixed = select(1, 2026277576509488133, 0.0, 0.0); print(mixed + mixed);',
    'var notnum = sqrt(-1.0); print(notnum != notnum); print(notnum == notnum);',
    'print(select(notnum, 1, 2, 3)); print(notnum <=> 1.0); if (notnum) { print(99); } else { print(5); }',
    'print(planar_dot64(7, 8, 5, 8)); print(btha_carry_p(7, 8, 5, 8)); print(btha_sum_p(7, 8, 5, 8));',

    'print(fma(1.5, 2.0, 0.25)); print(sqrt(16.0)); print(abs(-3.5));',
])
expected.extend([42,5,5,3,30,30,1,2,1.5,7,
                 2*float(2026277576509488133), 2*float(2026277576509488133), 1,-1,2,0,5,3,5,10,
                 3.25,4,3.5])

with tempfile.TemporaryDirectory(prefix='bt-differential-') as temp:
    work = Path(temp)
    source, shared = emit('\n'.join(expressions), work)
    compiled, interpreted = native(shared), run([compiler, 'run', str(source)])
    core = run([compiler, '--core-run', str(source)])
    assert core.returncode == 0, core.stderr
    assert compiled.returncode == interpreted.returncode == 0, (compiled.stderr, interpreted.stderr)
    def number(token):
        return float(token) if any(c in token for c in '.eE') else int(token)
    nvalues = [number(line.strip()) for line in compiled.stdout.splitlines() if line.strip()]
    ivalues = [number(v) for v in re.findall(r'^(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?) \[', interpreted.stdout, re.M)]
    cvalues = [number(v) for v in re.findall(r'^(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?) \[', core.stdout, re.M)]
    assert len(cvalues)==len(expected),(len(cvalues),len(expected))
    for got,want in zip(cvalues,expected):
        assert got==want if isinstance(want,int) else math.isclose(got,want,rel_tol=1e-12,abs_tol=1e-12),(got,want)
    assert len(nvalues) == len(ivalues) == len(expected), (len(nvalues),len(ivalues),len(expected))
    for index,(actual,reference,oracle) in enumerate(zip(nvalues,ivalues,expected)):
        if isinstance(oracle,int):
            assert actual == reference == oracle, (index,actual,reference,oracle)
        else:
            assert math.isclose(actual,oracle,rel_tol=1e-12,abs_tol=1e-6), (index,actual,oracle)
            assert math.isclose(reference,oracle,rel_tol=1e-12,abs_tol=1e-12), (index,reference,oracle)

    for bad in [
        'print(missing);', 'print(missing_function(1));',
        'func f(x: int) -> int { return x; } print(f());',
        'func f(x: int) -> int { return x; } print(f(1.5));',
        'func f() -> int { return 1.5; }',
        'if (1) { var x = 1; } print(x);',
        'print(1e999);', 'print(1e-999);',
        'var bad:trit = 42;', 'var w = ' + '+' + '0'*49 + '+; print(w+1);',
    ]:
        source.write_text(bad)
        for args in [[compiler,'run',str(source)], [compiler,'--core-run',str(source)], [compiler,str(source),'--emit-llvm']]:
            result = run(args)
            assert result.returncode != 0 and ('Error' in result.stderr or 'Failure' in result.stderr), result

    for overflow in [
        'print(2026277576509488133 + 1);',
        'print(-2026277576509488133 - 1);',
        'print(2026277576509488133 * 2);',
        'print(btha_sum_p(1152921504606846976, 0, 1152921504606846975, 0));',
        'func f() -> int { return 2026277576509488133 + 1; } print(f());',
    ]:
        source, shared = emit(overflow,work)
        assert native(shared).returncode == 1
        assert run([compiler,'run',str(source)]).returncode == 1
        assert run([compiler,'--core-run',str(source)]).returncode == 1
    source, shared = emit('return 7;',work)
    assert native(shared).returncode == run([compiler,'run',str(source)]).returncode == 7

    # Actual driver invocation must not evaluate output-path shell substitutions.
    marker = work/'injected'
    strange = str(work/f'out$(touch {marker}).s')
    run([compiler,str(source),'--target=x86_64','-S','-o',strange])
    assert not marker.exists()
    for options in [['--target=riscv64-xbt','-S'],['--target=riscv64-xbt','-c'],['--emit-metal'],['--emit-vulkan'],['--not-a-real-option']]:
        result = run([compiler,str(source),*options])
        assert result.returncode != 0, options
print(f'PASS: {len(expected)} native/AST/core-IR/Python oracle values, negative cases, overflow propagation and driver safety')
