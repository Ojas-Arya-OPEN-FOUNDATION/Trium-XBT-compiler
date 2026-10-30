#!/usr/bin/env python3
from pathlib import Path
import tempfile,subprocess,sys,os
build=Path(sys.argv[1]).resolve();root=Path(__file__).resolve().parents[2]
cmake_args=['cmake','-S',str(root/'examples/sdk')]
if sys.platform=='darwin':
    sdk = os.environ.get('SDKROOT')
    if Path('/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk').exists():
        cmake_args.append('-DCMAKE_OSX_SYSROOT=/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk')
    elif sdk and Path(sdk).exists():
        cmake_args.append(f'-DCMAKE_OSX_SYSROOT={sdk}')
with tempfile.TemporaryDirectory(prefix='trium-relocated-') as d:
 directory=Path(d);prefix=directory/'original';relocated=directory/'moved';consumer=directory/'consumer'
 subprocess.run(['cmake','--install',str(build),'--prefix',str(prefix)],check=True,capture_output=True);prefix.rename(relocated)
 subprocess.run(cmake_args+['-B',str(consumer),'-DCMAKE_PREFIX_PATH='+str(relocated)],check=True,capture_output=True)
 subprocess.run(['cmake','--build',str(consumer),'-j2'],check=True,capture_output=True)
 for exe in ['sdk_c','sdk_cpp']:subprocess.run([str(consumer/exe)],check=True)
 print('PASS installed relocated C/C++ SDK+Math consumers without checkout include paths')
