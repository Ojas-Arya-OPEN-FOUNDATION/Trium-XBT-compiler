#!/usr/bin/env python3
"""Run a generated CUDA map, or explicitly emit PTX without claiming execution."""
import argparse
from pathlib import Path
import sys
from cuda_map import Driver,DeviceUnavailable,ptx_from_llvm
p=argparse.ArgumentParser()
p.add_argument('module',type=Path);p.add_argument('inputs',type=int,nargs='*')
p.add_argument('--device',type=int,default=0);p.add_argument('--arch')
p.add_argument('--emit-ptx',type=Path,help='offline PTX only; do not execute a device')
a=p.parse_args()
try:
    text=a.module.read_text()
    if a.emit_ptx:
        a.emit_ptx.write_text(ptx_from_llvm(text,a.arch or 'sm_50'))
        print(f'PTX written to {a.emit_ptx}; device execution was not performed')
    else:
        driver=Driver(a.device);ptx=ptx_from_llvm(text,a.arch or driver.architecture())
        print(driver.execute(ptx,a.inputs))
except DeviceUnavailable as e:
    print(f'UNAVAILABLE: {e}',file=sys.stderr);raise SystemExit(77)
except Exception as e:
    print(f'ERROR: {e}',file=sys.stderr);raise SystemExit(1)
