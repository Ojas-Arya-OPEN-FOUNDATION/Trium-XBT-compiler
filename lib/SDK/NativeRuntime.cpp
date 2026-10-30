#include "NativeRuntime.hpp"
#include "VulkanRuntime.hpp"
#include "Ternary/IR/LLVM.hpp"
#include "bt_process.hpp"
#include <cstdlib>
#include <fstream>
#include <filesystem>
#include <sstream>
#include <dlfcn.h>
namespace bt::sdk {
namespace {
size_t expression(const ir::Expr& e){
    if(e.op==ir::Op::Call || e.op==ir::Op::Print || e.op==ir::Op::Packed)
        throw Unavailable("native-v1 rejects calls/IO/packed values; choose explicit reference backend");
    size_t work=1;for(auto& child:e.operands)work+=expression(*child);return work;
}
size_t statements(const std::vector<ir::Statement>& body){
    size_t work=0;for(auto& s:body){
        if(s.kind==ir::StatementKind::While)throw Unavailable("native-v1 rejects loops; choose explicit reference backend");
        ++work;if(s.value)work+=expression(*s.value);
        work+=statements(s.body)+statements(s.alternative)+statements(s.negative);
    }return work;
}
std::string type(ir::Type t){return t==ir::Type::Void?"void":t==ir::Type::Float64?"double":"i64";}
}
size_t native_work(const ir::Function& f){return statements(f.body);}
std::string native_compiler(){
    auto valid=[](const std::filesystem::path& p){std::error_code ec;return std::filesystem::is_regular_file(p,ec) && access(p.c_str(),X_OK)==0;};
    if(const char* selected=std::getenv("BT_SDK_CLANG")){
        std::filesystem::path p(selected);return p.is_absolute() && valid(p)?p.string():std::string{};
    }
    if(const char* path=std::getenv("PATH")){
        std::istringstream paths(path);std::string item;
        while(std::getline(paths,item,':')){if(item.empty())continue;auto p=std::filesystem::path(item)/"clang";if(p.is_absolute() && valid(p))return p.string();}
    }return {};
}
struct NativeProgram::State {
    host::TemporaryDirectory directory;
    void* library=nullptr;
    using Entry=int32_t(*)(const uint64_t*,uint64_t*);
    Entry entry=nullptr;
    ~State(){if(library)dlclose(library);}
};
NativeProgram::NativeProgram(const ir::Function& f,const std::string& compiler):state(std::make_unique<State>()){
    native_work(f);
    if(compiler.empty())throw Unavailable("native compiler unavailable; set BT_SDK_CLANG to a trusted absolute Clang path");
    ir::Program p;p.functions.push_back(f);
    std::string text=ir::LLVMEmitter{}.generate(p,"","",true);
    // Uniform ABI: separate 64-bit payload slots, never libffi, varargs, or
    // guessed C function casts. The frontend/core verifier owns each slot type.
    std::ostringstream wrapper;
    wrapper<<"\ndefine i32 @bt_sdk_native_entry(ptr %arguments, ptr %output) {\nentry:\n"
             "  store i32 0, ptr @bt.core.error\n  store i32 0, ptr @bt.core.depth\n";
    for(size_t i=0;i<f.parameters.size();++i){
        wrapper<<"  %p"<<i<<" = getelementptr i64, ptr %arguments, i64 "<<i<<"\n"
               <<"  %a"<<i<<" = load i64, ptr %p"<<i<<", align 8\n";
        if(f.parameters[i].type==ir::Type::Float64)wrapper<<"  %f"<<i<<" = bitcast i64 %a"<<i<<" to double\n";
    }
    wrapper<<"  "<<(f.result==ir::Type::Void?"":"%value = ")<<"call "<<type(f.result)<<" @bt.core.fn.0(";
    for(size_t i=0;i<f.parameters.size();++i)wrapper<<(i?", ":"")<<type(f.parameters[i].type)<<" %"<<(f.parameters[i].type==ir::Type::Float64?"f":"a")<<i;
    wrapper<<")\n";
    if(f.result==ir::Type::Float64)wrapper<<"  %bits = bitcast double %value to i64\n";
    wrapper<<"  store i64 "<<(f.result==ir::Type::Void?"0":f.result==ir::Type::Float64?"%bits":"%value")<<", ptr %output, align 8\n"
             "  %status = load i32, ptr @bt.core.error\n  ret i32 %status\n}\n";
    text+=wrapper.str();auto source=state->directory.file("module.ll"),image=state->directory.file("module.so"),log=state->directory.file("compiler.log");
    {std::ofstream stream(source);stream<<text;if(!stream)throw std::runtime_error("cannot write native compiler input");}
    int status;
    try{
        std::vector<std::string> compile_args = {compiler,"-O3","-Wno-override-module","-shared","-fPIC"};
#if defined(__APPLE__)
        const char* sdk = std::getenv("SDKROOT");
        if (std::filesystem::exists("/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk")) {
            compile_args.push_back("-isysroot");
            compile_args.push_back("/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk");
        } else if (sdk) {
            compile_args.push_back("-isysroot");
            compile_args.push_back(sdk);
        }
#endif
        compile_args.insert(compile_args.end(), {source,"-o",image,"-lm"});
        status=host::run_process(compile_args,log,60000);
    }
    catch(const host::ProcessTimeout&){throw;}
    catch(const std::exception& e){throw Unavailable(e.what());}
    if(status){std::ifstream stream(log);std::string diagnostic(16384,'\0');stream.read(diagnostic.data(),diagnostic.size());diagnostic.resize(stream.gcount());throw Unavailable("native compiler failed: "+diagnostic);}
    state->library=dlopen(image.c_str(),RTLD_NOW|RTLD_LOCAL);
    if(!state->library)throw Unavailable(std::string("cannot load native module: ")+dlerror());
    state->entry=reinterpret_cast<State::Entry>(dlsym(state->library,"bt_sdk_native_entry"));
    if(!state->entry)throw std::runtime_error("native wrapper ABI symbol missing");
}
NativeProgram::~NativeProgram()=default;
uint64_t NativeProgram::invoke(const uint64_t* arguments)const{
    uint64_t result=0;if(state->entry(arguments,&result))throw std::overflow_error("native checked arithmetic failure");return result;
}
}
