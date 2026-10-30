#!/usr/bin/env python3
"""User-run reference/native/Vulkan correctness and benchmark evidence, never simulated GPU timings.
Run on each laptop/cloud device. Software ICDs are labeled; --require-hardware
rejects them. JSON is also written for unavailable hardware/tool failures.
"""
from pathlib import Path
import argparse,datetime,hashlib,json,os,platform,statistics,sys,time,subprocess
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'ffi/python'))
import trium
p=argparse.ArgumentParser()
p.add_argument('--library',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
p.add_argument('--backend',choices=['cpu','native','vulkan','both','all'],default='both');p.add_argument('--count',type=int,default=10000)
p.add_argument('--input',choices=['list','numpy'],default='list',help='NumPy avoids Python element-by-element marshalling')
p.add_argument('--repeats',type=int,default=7);p.add_argument('--device',type=int,help='omit to test every enumerated device')
p.add_argument('--representations',choices=['portable32','native64','auto'],nargs='+',default=['portable32','native64'])
p.add_argument('--groups',type=int,nargs='+',default=[32,64,128]);p.add_argument('--require-hardware',action='store_true')
a=p.parse_args()
if not 0<a.count<=1000000 or not 3<=a.repeats<=100 or any(g not in (32,64,128) for g in a.groups):p.error('invalid count/repeats/workgroup')
library=a.library.resolve();values=[i%201-100 for i in range(a.count)]
inputs=values
if a.input=='numpy':
 import numpy as np
 inputs=np.asarray(values,dtype=np.int64)
workloads={
 'signed_map':('return select(x,x*3+1,0,x*3-1);',lambda x:3*x+(1 if x>0 else -1 if x<0 else 0)),
 'checked_polynomial':('return (x*x)+(2*x)+1;',lambda x:x*x+2*x+1),
 'ternary_weight_dot':('return planar_dot64(x,0,85,0);',lambda x:(x&85).bit_count())}
report={'schema':1,'date_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'platform':platform.platform(),
 'library_sha256':hashlib.sha256(library.read_bytes()).hexdigest(),'count':a.count,'repeats':a.repeats,'input':a.input,'devices':[], 'runs':[],
 'qualification':'physical devices require user-run verification; software devices remain separately labeled',
 'timing_notes':'cold_start includes backend compilation/setup. Repeated samples reuse module-owned native/Vulkan resources; growth reallocates buffers. Host total includes lookup, staging, dispatch and readback. Vulkan execute_ms is queue-submit-to-fence wall time; CPU/native execute_ms is host computation. total_ms is SDK-only; binding_total_ms also includes Python input/output marshalling. GPU timestamp bands include synchronization; -1 means unavailable. No GPU/CPU crossover speedup is inferred.'}
exit_code=0
try:
 with trium.Context(library=str(library)) as initial:
  if a.backend not in ('cpu','native'):
   try:report['devices']=initial.devices()
   except trium.Error as e:
    report['device_discovery']={'status':'unavailable' if e.code==4 else 'error','code':e.code,'message':str(e)}
    exit_code=77 if e.code==4 else 1
  selections=[]
  if a.backend in ('cpu','both','all'):selections.append(('cpu',0,64,None,'portable32'))
  if a.backend in ('native','all'):selections.append(('native',0,64,None,'portable32'))
  if a.backend not in ('cpu','native'):
   for device in report['devices']:
    if a.device is not None and device['index']!=a.device:continue
    if a.require_hardware and device['software']:continue
    for representation in a.representations:
     if representation=='native64' and not device.get('shader_int64',False):
      report.setdefault('unsupported_variants',[]).append({'device':device['index'],'representation':'native64','reason':'shaderInt64 unavailable'});continue
     selections.extend(('vulkan',device['index'],g,device,representation) for g in a.groups)
   if not any(s[0]=='vulkan' for s in selections):
    exit_code=77;report['no_matching_devices']='No selected compatible hardware/software device; hardware-only excludes CPU ICDs'
  for backend,index,group,device,representation in selections:
   with trium.Context(library=str(library),device=index,workgroup=group,representation=representation,operation_budget=100000000) as context:
    for name,(body,reference) in workloads.items():
     source='func transform(x:int)->int{'+body+'}'
     entry={'workload':name,'backend':backend,'device':device,'workgroup':group,'requested_representation':representation,'representation':'pending' if backend=='vulkan' else ('native-llvm-int39' if backend=='native' else 'core-interpreter-int39'),
       'source':source,'source_sha256':hashlib.sha256(source.encode()).hexdigest(),'samples':[]}
     try:
      expected=[reference(x) for x in values]
      start=time.perf_counter();module=context.compile(source);entry['source_compile_ms']=(time.perf_counter()-start)*1000
      with module:
       entry['cache_before']=context.cache_info
       runner=module.map if a.input=='list' else module.map_array
       start=time.perf_counter();result,cold=runner('transform',inputs,backend=backend);cold['binding_total_ms']=(time.perf_counter()-start)*1000
       assert list(result)==expected,'cold reference mismatch'
       if device:assert cold['device_uuid']==device['device_uuid'],'cold device attribution mismatch'
       entry['cold_start']=cold
       for _ in range(a.repeats):
        start=time.perf_counter();result,metrics=runner('transform',inputs,backend=backend);metrics['binding_total_ms']=(time.perf_counter()-start)*1000
        assert list(result)==expected,'reference mismatch'
        if device:assert metrics['device_uuid']==device['device_uuid'],'device enumeration changed: refusing incorrect benchmark attribution'
        entry['samples'].append(metrics)
      entry['status']='passed';entry['representation']=('native64-checked-multiply' if metrics['representation']==1 else 'uint32-pair-int39') if backend=='vulkan' else ('native-llvm-int39' if backend=='native' else 'core-interpreter-int39');entry['cache_after']=context.cache_info;entry['medians_ms']={field:statistics.median(s[field] for s in entry['samples']) for field in entry['samples'][0] if field.endswith('_ms')}
     except Exception as e:
      entry['status']='failed';entry['error']=str(e);exit_code=1
     report['runs'].append(entry)
except Exception as e:report['fatal_error']=str(e);exit_code=1
finally:
 report['exit_code']=exit_code;a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n')
 print(f'Evidence: {a.output}; exit={exit_code}; runs={len(report["runs"])}')
sys.exit(exit_code)
