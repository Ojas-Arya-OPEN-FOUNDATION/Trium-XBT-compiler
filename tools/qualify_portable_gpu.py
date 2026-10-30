#!/usr/bin/env python3
"""Isolated user-run device correctness + benchmark suite; JSON on every outcome.
Physical GPU checks are never inferred from software Vulkan. No network access.
"""
from pathlib import Path
import argparse,datetime,json,os,subprocess,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'ffi/python'))
import trium
p=argparse.ArgumentParser();p.add_argument('--library',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
p.add_argument('--allow-software',action='store_true');p.add_argument('--count',type=int,default=10000);p.add_argument('--repeats',type=int,default=7)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
report={'schema':1,'date_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'device_results':[],
 'physical_execution':'not_performed','policy':'physical devices only unless --allow-software; software success does not qualify physical GPUs'}
code=0
try:
 with trium.Context(library=str(a.library.resolve())) as context:devices=context.devices()
 selected=[d for d in devices if a.allow_software or not d['software']]
 report['enumerated_devices']=devices
 if not selected:code=77;report['status']='no_matching_device'
 for device in selected:
  index=device['index'];record={'device':device};report['device_results'].append(record)
  cmd=[sys.executable,str(ROOT/'test/SDK/vulkan.py'),str(a.library.resolve()),'--execute','--device-index',str(index)]
  test=subprocess.run(cmd,capture_output=True,text=True,timeout=300)
  log=a.output/f'device-{index}-correctness.log';log.write_text(test.stdout+test.stderr)
  record['correctness']={'exit_code':test.returncode,'log':log.name}
  if test.returncode:code=1;continue
  bench=a.output/f'device-{index}-benchmark.json'
  cmd=[sys.executable,str(ROOT/'tools/benchmark_portable_gpu.py'),'--library',str(a.library.resolve()),'--output',str(bench),'--backend','vulkan','--device',str(index),'--count',str(a.count),'--repeats',str(a.repeats)]
  if not a.allow_software:cmd+=['--require-hardware']
  benchmark=subprocess.run(cmd,capture_output=True,text=True,timeout=1800)
  (a.output/f'device-{index}-benchmark.log').write_text(benchmark.stdout+benchmark.stderr)
  record['benchmark']={'exit_code':benchmark.returncode,'json':bench.name}
  if benchmark.returncode:code=1
  if not device['software']:report['physical_execution']='performed_on_enumerated_non_cpu_device; see exact device metadata'
 report.setdefault('status','passed' if code==0 else 'failed')
except trium.Error as e:code=77 if e.code==4 else 1;report.update(status='unavailable' if code==77 else 'failed',error=str(e),sdk_status=e.code)
except Exception as e:code=1;report.update(status='failed',error=str(e))
report['exit_code']=code;(a.output/'qualification.json').write_text(json.dumps(report,indent=2)+'\n')
print(f'Qualification: {a.output}/qualification.json; exit={code}');sys.exit(code)
