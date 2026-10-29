#!/usr/bin/env python3
"""Support-manifest completeness gate, not a substitute for semantic proofs."""
from pathlib import Path
import json,re
root=Path(__file__).resolve().parents[1]
s=json.loads((root/'specs/core-profile-v0.1.json').read_text())
source=(root/'include/Ternary/IR/Core.hpp').read_text()
names={n.strip() for n in re.search(r'enum class Op \{(.*?)\};',source,re.S)[1].replace('\n','').split(',')}
manifest={o['operation'] for o in s['operations']}
assert names==manifest,(names-manifest,manifest-names)
assert len(manifest)==len(s['operations'])
h=(3**s['scalar_integer_width']-1)//2
assert s['scalar_integer_min']==-h and s['scalar_integer_max']==h
for o in s['operations']:
    assert o['semantics'] and isinstance(o['llvm_cpu'],bool)
    assert not o['cuda_map'] or o['llvm_cpu']
assert s['runtime_abi_major']==1
print(f'PASS: complete operation/profile manifest ({len(names)} operations)')
