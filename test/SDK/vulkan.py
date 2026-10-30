#!/usr/bin/env python3
"""Program-dependent shader validation; optional real Vulkan API/device execution.
Software Vulkan is explicitly labeled, never physical GPU qualification.
"""
from pathlib import Path
import os,sys,subprocess,tempfile,random,shutil,struct
os.environ['BT_SDK_LIBRARY']=str(Path(sys.argv[1]).resolve());sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'ffi/python'))
import trium
from concurrent.futures import ThreadPoolExecutor
H=(3**39-1)//2
def capabilities(path):
 data=path.read_bytes();words=struct.unpack('<'+'I'*(len(data)//4),data);index=5;caps=set()
 while index<len(words):
  count=words[index]>>16;opcode=words[index]&65535;assert count>0 and index+count<=len(words)
  if opcode==17:caps.add(words[index+1])
  index+=count
 return caps
def ternary(a,b,op):
 result=0;place=1
 for _ in range(39):
  qa,ra=divmod(a,3);qb,rb=divmod(b,3)
  if ra==2:qa+=1;ra=-1
  if rb==2:qb+=1;rb=-1
  result+=(min(ra,rb) if op==0 else max(ra,rb) if op==1 else ra*rb)*place
  place*=3;a=qa;b=qb
 return result
cases=[('return select(x,x*3+1,0,x*3-1);',lambda x:3*x+(1 if x>0 else -1 if x<0 else 0)),
       ('return abs(x);',abs),('return x<=>7;',lambda x:(x>7)-(x<7)),
       ('return planar_dot64(x,0,7,0);',lambda x:(x&7).bit_count()),
       ('var y=x+2;if(y>0){return y*2;}else{return -y;}',lambda x:2*(x+2) if x+2>0 else -(x+2)),
       ('return select(1,x,2026277576509488133+1,0);',lambda x:x)]
if not all(shutil.which(os.environ.get(k,v)) for k,v in [('BT_GLSLANG','glslangValidator'),('BT_SPIRV_VAL','spirv-val')]):
 print('SKIP: glslang/SPIRV-Tools required');raise SystemExit(77)
cases.extend([('return x & 13;',lambda x:ternary(x,13,0)),('return x | -7;',lambda x:ternary(x,-7,1)),('return x ^ 13;',lambda x:ternary(x,13,2)),('return btha_sum_p(x,0,1152921504606846975,0);',lambda x:x^1152921504606846975)])
execute='--execute' in sys.argv
index=int(sys.argv[sys.argv.index('--device-index')+1]) if '--device-index' in sys.argv else 0
with trium.Context(device=index) as ctx:
 if execute:
  try:devices=ctx.devices()
  except trium.Error as e:
   if e.code==4:print('SKIP Vulkan execution unavailable:',e);raise SystemExit(77)
   raise
  print('Vulkan devices:',devices)
  if not devices:raise SystemExit(77)
 for number,(body,reference) in enumerate(cases):
  with ctx.compile('func transform(x:int)->int{'+body+'}') as mod:
   shader=mod.shader('transform');assert 'int64_t' not in shader
   with tempfile.TemporaryDirectory() as d:
    source=Path(d)/'program.comp';output=Path(d)/'program.spv';source.write_text(shader)
    subprocess.run([os.environ.get('BT_GLSLANG','glslangValidator'),'-V','--target-env','vulkan1.1','-o',str(output),str(source)],check=True,capture_output=True)
    subprocess.run([os.environ.get('BT_SPIRV_VAL','spirv-val'),'--target-env','vulkan1.1',str(output)],check=True,capture_output=True)
    assert 11 not in capabilities(output),'portable SPIR-V unexpectedly requires Int64'
   rng=random.Random(9981);values=[-101,-2,-1,0,1,2,101]+[rng.randint(-100000,100000) for _ in range(100)]
   expected=[reference(x) for x in values];assert mod.map('transform',values)[0]==expected
   if execute:assert mod.map('transform',values,backend='vulkan')[0]==expected
 if execute:
  # Includes abs products that wrap int64 back into the int39 domain if the
  # magnitude guard is wrong. Native signed multiplication must not be used.
  for factor in (2,9,-9,17):
   with ctx.compile(f'func transform(x:int)->int{{return x*({factor});}}') as mod:
    try:mod.map('transform',[1,H],backend='vulkan')
    except trium.Error as e:assert e.code==2,e
    else:raise AssertionError('GPU multiplication overflow accepted')
 # Native64 SPIR-V is always validated offline; physical execution is gated.
 for representation in ('native64','auto'):
  with trium.Context(device=index,representation=representation) as variant:
   with variant.compile('func transform(x:int)->int{return x*9;}') as mod:
    shader=mod.shader('transform')
    with tempfile.TemporaryDirectory() as d:
     path=Path(d)/'variant.comp';binary=Path(d)/'variant.spv';path.write_text(shader)
     subprocess.run([os.environ.get('BT_GLSLANG','glslangValidator'),'-V','--target-env','vulkan1.1','-o',str(binary),str(path)],check=True,capture_output=True)
     subprocess.run([os.environ.get('BT_SPIRV_VAL','spirv-val'),'--target-env','vulkan1.1',str(binary)],check=True,capture_output=True)
     assert (11 in capabilities(binary))==(representation=='native64')
    if execute:
     supported=devices[index]['shader_int64'] or representation=='auto'
     if not supported:
      try:mod.map('transform',[1],backend='vulkan')
      except trium.Error as e:assert e.code==4,e
      else:raise AssertionError('unsupported native64 feature accepted')
     else:
      result,metrics=mod.map('transform',[-1,0,1],backend='vulkan');assert result==[-9,0,9]
      assert metrics['representation']==int(devices[index]['shader_int64'])
      try:mod.map('transform',[H],backend='vulkan')
      except trium.Error as e:assert e.code==2,e
      else:raise AssertionError('variant overflow accepted')
 if execute:
  with ctx.compile('func cachemap(x:int)->int{return x*9;}') as cached:
   builds=ctx.cache_info['vulkan_builds']
   for number,n in enumerate((4,1,33,0,129,7,129)):
    values=list(range(n));out,metrics=cached.map('cachemap',values,backend='vulkan');assert out==[9*x for x in values]
    assert metrics['compile_ms']>0 if number==0 else metrics['compile_ms']==0
   try:cached.map('cachemap',[1,H],backend='vulkan')
   except trium.Error as e:assert e.code==2
   else:raise AssertionError('cached overflow accepted')
   assert cached.map('cachemap',[2],backend='vulkan')[0]==[18]
   def dispatch(i):assert cached.map('cachemap',[i,-i],backend='vulkan')[0]==[i*9,-i*9]
   with ThreadPoolExecutor(max_workers=8) as pool:list(pool.map(dispatch,range(24)))
   assert ctx.cache_info['vulkan_builds']==builds+1
   ctx.clear_cache();dispatch(2);assert ctx.cache_info['vulkan_builds']==builds+2
   print('PASS cached Vulkan pipeline/buffer growth/shrink, status reset, concurrency and explicit invalidation')
 print('PASS uint32-only program-dependent SPIR-V and reference cases'+(' / real Vulkan API execution (device type reported above)' if execute else ' / emission only'))
