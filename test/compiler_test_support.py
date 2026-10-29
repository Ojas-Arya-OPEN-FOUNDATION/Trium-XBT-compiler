"""Independent host execution of the actual LLVM emitted by bt-compiler."""
from pathlib import Path
import shutil
import subprocess
import sys
import time

class Harness:
    def __init__(self,compiler,optimize=False,flags=()):
        try:
            from llvmlite import binding as llvm
        except ImportError:
            print('SKIP: llvmlite required');raise SystemExit(77)
        self.llvm=llvm
        self.compiler=str(Path(compiler).resolve())
        self.linker=shutil.which('c++')
        if not self.linker or sys.platform!='linux':
            print('SKIP: independent host adapter requires Linux and C++');raise SystemExit(77)
        llvm.initialize_native_target();llvm.initialize_native_asmprinter()
        self.tm=llvm.Target.from_default_triple().create_target_machine(reloc='pic',opt=3 if optimize else 0)
        self.optimize=optimize
        self.flags=list(flags)
        self.metrics={}
    @staticmethod
    def run(args): return subprocess.run(args,capture_output=True,text=True,timeout=60)
    def emit(self,text,work):
        start=time.perf_counter();source=work/'program.bt';source.write_text(text)
        ir=work/'program.ll'
        p=self.run([self.compiler,str(source),'--emit-llvm','-o',str(ir),*self.flags])
        assert p.returncode==0,p.stderr
        self.metrics['emit_ms']=(time.perf_counter()-start)*1000
        start=time.perf_counter();mod=self.llvm.parse_assembly(ir.read_text());mod.verify()
        mod.triple=self.llvm.get_default_triple();mod.data_layout=str(self.tm.target_data)
        if self.optimize:
            pb=self.llvm.create_pass_builder(self.tm,self.llvm.PipelineTuningOptions(speed_level=3))
            pm=pb.getModulePassManager();pm.run(mod,pb)
        obj=work/'program.o';obj.write_bytes(self.tm.emit_object(mod))
        self.metrics['lower_ms']=(time.perf_counter()-start)*1000
        start=time.perf_counter();shared=work/'program.so'
        p=self.run([self.linker,'-shared',str(obj),'-lm','-o',str(shared)])
        assert p.returncode==0,p.stderr
        self.metrics['link_ms']=(time.perf_counter()-start)*1000
        return source,shared
    def native(self,shared):
        return self.run([sys.executable,'-c',f'import ctypes;l=ctypes.CDLL({str(shared)!r});raise SystemExit(l.bt_main())'])
