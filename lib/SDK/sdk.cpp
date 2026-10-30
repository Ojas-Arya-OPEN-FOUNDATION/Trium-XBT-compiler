#include "Ternary/SDK/sdk.h"
#include "Ternary/Math/Float.hpp"
#include "Ternary/Kernel/Vulkan.hpp"
#include "Ternary/IR/Execute.hpp"
#include "ternary_ir_lowering.hpp"
#include <mutex>
#include <atomic>
#include <chrono>
#include <cstring>
#include <unordered_map>
#include <functional>
#include "VulkanRuntime.hpp"
#include "NativeRuntime.hpp"
#include "bt_process.hpp"
#include <list>
namespace {
using Clock=std::chrono::steady_clock;
struct Module;
struct SourceEntry {std::string source;std::shared_ptr<const bt::ir::Program> program;std::weak_ptr<Module> module;};
struct Context {
    uint64_t maximum=1000000,budget=10000000;std::atomic<bool> closed=false;
    uint32_t device=0,group=64,representation=0;
    const std::string compiler=bt::sdk::native_compiler();
    std::mutex sources_mutex;std::list<SourceEntry> sources;size_t source_bytes=0;
    std::atomic<uint64_t> epoch{1},source_hits{0},source_misses{0},native_hits{0},native_builds{0},vulkan_hits{0},vulkan_builds{0};
};
struct Module {
    std::shared_ptr<Context> context;bt::ir::Program program;
    std::unordered_map<std::string,size_t> exports;std::vector<bool> pure;std::string metadata;
    std::mutex cache_mutex;uint64_t epoch=0;
    std::unordered_map<std::string,std::shared_ptr<bt::sdk::NativeProgram>> native;
    std::unordered_map<std::string,std::shared_ptr<bt::sdk::VulkanProgram>> vulkan;
};
struct Entry {std::shared_ptr<Context> context;std::shared_ptr<Module> module;};
std::mutex mutex;uint64_t next_id=1;std::unordered_map<uint64_t,Entry> registry;
struct Resource:std::runtime_error{using std::runtime_error::runtime_error;};
struct Closed:std::runtime_error{using std::runtime_error::runtime_error;};
void message(char* out,size_t n,const char* text) noexcept {if(out && n){auto len=std::min(n-1,std::strlen(text));std::memcpy(out,text,len);out[len]=0;}}
template<class F> bt_sdk_status boundary(char* out,size_t n,F fn) noexcept {
    message(out,n,"");try{fn();return BT_SDK_OK;}
    catch(const Resource& e){message(out,n,e.what());return BT_SDK_RESOURCE;}
    catch(const bt::ir::ResourceLimit& e){message(out,n,e.what());return BT_SDK_RESOURCE;}
    catch(const bt::host::ProcessTimeout& e){message(out,n,e.what());return BT_SDK_RESOURCE;}
    catch(const bt::frontend::ParseError& e){message(out,n,e.what());return BT_SDK_INVALID;}
    catch(const bt::frontend::TypeError& e){message(out,n,e.what());return BT_SDK_INVALID;}
    catch(const Closed& e){message(out,n,e.what());return BT_SDK_CLOSED;}
    catch(const bt::sdk::Unavailable& e){message(out,n,e.what());return BT_SDK_UNAVAILABLE;}
    catch(const bt::sdk::DeviceError& e){message(out,n,e.what());return BT_SDK_DEVICE;}
    catch(const std::overflow_error& e){message(out,n,e.what());return BT_SDK_OVERFLOW;}
    catch(const std::domain_error& e){message(out,n,e.what());return BT_SDK_INVALID;}
    catch(const std::invalid_argument& e){message(out,n,e.what());return BT_SDK_INVALID;}
    catch(const std::bad_alloc&){message(out,n,"allocation failed");return BT_SDK_OOM;}
    catch(const std::exception& e){message(out,n,e.what());return BT_SDK_INTERNAL;}
    catch(...){message(out,n,"unknown SDK failure");return BT_SDK_INTERNAL;}
}
Entry get(uint64_t id) {std::lock_guard lock(mutex);auto i=registry.find(id);if(i==registry.end())throw std::invalid_argument("unknown/released SDK handle");return i->second;}
uint64_t add(Entry e){std::lock_guard lock(mutex);if(!next_id)throw std::overflow_error("handle ID exhausted");auto id=next_id++;registry.emplace(id,std::move(e));return id;}
std::shared_ptr<Module> module(uint64_t id){auto e=get(id);if(!e.module)throw std::invalid_argument("expected module handle");if(e.context->closed)throw Closed("SDK context closed");return e.module;}
void required(bool ok,const char* why){if(!ok)throw std::invalid_argument(why);}
void limited(bool ok,const char* why){if(!ok)throw Resource(why);}
void copy(const std::string& s,char* out,size_t n){required(out && n>s.size(),"output text buffer too small");std::memcpy(out,s.c_str(),s.size()+1);}
std::vector<bool> pure_exports(const Module& m){
    // Reverse call-graph propagation is linear in nodes+edges, including cycles.
    // Repeating transitive DFS for every export would be cubic on long chains.
    const auto n=m.program.functions.size();std::vector<bool> pure(n,true);
    std::vector<std::vector<size_t>> callers(n);std::vector<size_t> pending;
    for(size_t i=0;i<n;++i){
        std::function<void(const bt::ir::Expr&)> expr=[&](const auto& e){
            if(e.op==bt::ir::Op::Print && pure[i]){pure[i]=false;pending.push_back(i);}
            if(e.op==bt::ir::Op::Call)callers[m.exports.at(e.name)].push_back(i);
            for(auto& child:e.operands)expr(*child);
        };
        std::function<void(const std::vector<bt::ir::Statement>&)> body=[&](const auto& statements){for(auto& s:statements){if(s.value)expr(*s.value);body(s.body);body(s.alternative);body(s.negative);}};
        body(m.program.functions[i].body);
    }
    for(size_t next=0;next<pending.size();++next)for(auto caller:callers[pending[next]])if(pure[caller]){pure[caller]=false;pending.push_back(caller);}
    return pure;
}
void namecheck(const char* name){required(name && std::strlen(name)<=1024,"invalid export name");}
double elapsed(Clock::time_point start){return std::chrono::duration<double,std::milli>(Clock::now()-start).count();}
const bt::ir::Function& export_function(const Module& m,const std::string& name){
    auto f=m.exports.find(name);required(f!=m.exports.end(),"unknown export");return m.program.functions[f->second];
}
std::shared_ptr<bt::sdk::NativeProgram> native_image(const std::shared_ptr<Module>& m,const bt::ir::Function& f,bt_sdk_metrics& times){
    std::lock_guard lock(m->cache_mutex);auto epoch=m->context->epoch.load();
    if(m->epoch!=epoch){m->native.clear();m->vulkan.clear();m->epoch=epoch;}
    if(auto it=m->native.find(f.name);it!=m->native.end()){++m->context->native_hits;return it->second;}
    auto begin=Clock::now();auto image=std::make_shared<bt::sdk::NativeProgram>(f,m->context->compiler);times.compile_ms=elapsed(begin);
    if(m->native.size()>=16)m->native.erase(m->native.begin());m->native.emplace(f.name,image);++m->context->native_builds;return image;
}
std::shared_ptr<bt::sdk::VulkanProgram> vulkan_image(const std::shared_ptr<Module>& m,const std::string& name,bt_sdk_metrics& times){
    std::lock_guard lock(m->cache_mutex);auto c=m->context;auto epoch=c->epoch.load();
    if(m->epoch!=epoch){m->native.clear();m->vulkan.clear();m->epoch=epoch;}
    if(auto it=m->vulkan.find(name);it!=m->vulkan.end()){++c->vulkan_hits;return it->second;}
    auto portable=bt::kernel::VulkanEmitter{}.generate(m->program,name,c->group);
    auto native=c->representation?bt::kernel::VulkanEmitter{}.generate(m->program,name,c->group,false,true):std::string{};
    auto image=std::make_shared<bt::sdk::VulkanProgram>(portable,native,c->group,c->device,c->representation);times.compile_ms=image->compilation_ms();
    if(m->vulkan.size()>=4)m->vulkan.erase(m->vulkan.begin());m->vulkan.emplace(name,image);++c->vulkan_builds;return image;
}
std::string json(const std::string& value){
    const char* digits="0123456789abcdef";std::string r="\"";
    for(unsigned char c:value){if(c=='"'||c=='\\'){r+='\\';r+=c;}else if(c<32 || c>=127){r+="\\u00";r+=digits[c>>4];r+=digits[c&15];}else r+=c;}return r+'"';
}
void text_query(const std::string& text,char* output,size_t* bytes){required(bytes,"null text length");auto available=*bytes;*bytes=text.size()+1;if(output)copy(text,output,available);}
std::shared_ptr<Context> context_handle(uint64_t id){auto e=get(id);required(!e.module,"expected context handle");if(e.context->closed)throw Closed("SDK context closed");return e.context;}
std::string type_name(bt::ir::Type t){return t==bt::ir::Type::Void?"void":t==bt::ir::Type::Float64?"binary64":t==bt::ir::Type::Trit?"trit":"int39";}
std::string describe(const Module& m){
    std::string text="{\"schema\":1,\"source_float\":\"binary64\",\"native_profile\":\"native-v1-loopfree\",\"functions\":[";
    for(size_t i=0;i<m.program.functions.size();++i){const auto& f=m.program.functions[i];if(i)text+=",";
        bool pure=m.pure[i],native=true,map=true;size_t work=0;std::string reason;
        try{work=bt::sdk::native_work(f);}catch(const bt::sdk::Unavailable& e){native=false;reason=e.what();}
        try{if(f.parameters.size()!=1 || f.parameters[0].type!=bt::ir::Type::Int39 || f.result!=bt::ir::Type::Int39)map=false;else bt::kernel::statements(f.body);}catch(const std::invalid_argument&){map=false;}
        text+="{\"name\":"+json(f.name)+",\"result\":"+json(type_name(f.result))+",\"parameters\":[";
        for(size_t j=0;j<f.parameters.size();++j){if(j)text+=",";text+="{\"index\":"+std::to_string(j)+",\"type\":"+json(type_name(f.parameters[j].type))+"}";}
        text+="],\"reference\":"+std::string(pure?"true":"false")+",\"native\":"+(native?"true":"false")+",\"map\":"+(map?"true":"false")+",\"native_work\":"+std::to_string(work)+",\"native_rejection\":"+json(reason)+"}";
    }return text+"]}";
}
void initialize(Module& m){if(!m.metadata.empty())return;
    for(size_t i=0;i<m.program.functions.size();++i)m.exports.emplace(m.program.functions[i].name,i);
    m.pure=pure_exports(m);m.metadata=describe(m);
}
template<int W=1,class F> void float_width(uint32_t width,F&& fn){
    if(width==W)fn.template operator()<W>();
    else if constexpr(W<40)float_width<W+1>(width,std::forward<F>(fn));
    else throw std::invalid_argument("BTFP width must be 1..40");
}
template<int W> bt::BTFP<W> read_float(const bt_sdk_float_value* v){
    required(v && v->width==W && !v->reserved && !v->reserved2,"invalid BTFP record/profile");
    const auto lo=bt::BTFP<W>::min_norm_mantissa(),hi=bt::BTFP<W>::max_mantissa();
    required(v->mantissa==0?v->exponent==0:
        ((v->mantissa>=lo && v->mantissa<=hi)||(v->mantissa<=-lo && v->mantissa>=-hi)),"noncanonical BTFP record");
    return bt::BTFP<W>(v->mantissa,v->exponent);
}
template<int W> bt_sdk_float_value write_float(bt::BTFP<W> v){return {W,0,v.mantissa(),v.exponent(),0};}

}
extern "C" {
uint32_t bt_sdk_abi_version(){return 1;}
uint32_t bt_sdk_api_version(){return 0x00010001;}
bt_sdk_status bt_sdk_info(char* output,size_t* bytes,char* err,size_t cap){return boundary(err,cap,[&]{
    std::string text="{\"schema\":1,\"abi_major\":1,\"api_major\":1,\"api_minor\":1,\"finite_float_profile\":1,\"native_profile\":\"native-v1-loopfree\",\"native_compiler_candidate\":"+json(bt::sdk::native_compiler())+",\"vulkan_built\":";
#if defined(BT_SDK_HAS_VULKAN)
    text+="true";
#else
    text+="false";
#endif
    text+=",\"sizes\":{\"options\":"+std::to_string(sizeof(bt_sdk_options))+",\"metrics\":"+std::to_string(sizeof(bt_sdk_metrics))+",\"value\":"+std::to_string(sizeof(bt_sdk_value))+",\"float_value\":"+std::to_string(sizeof(bt_sdk_float_value))+"},\"source_cache_entries\":8,\"source_cache_bytes\":16777216,\"native_cache_entries_per_module\":16,\"vulkan_cache_entries_per_module\":4}";
    text_query(text,output,bytes);
});}
bt_sdk_status bt_sdk_module_info(bt_sdk_handle id,char* output,size_t* bytes,char* err,size_t cap){return boundary(err,cap,[&]{text_query(module(id)->metadata,output,bytes);});}
bt_sdk_status bt_sdk_cache_info(bt_sdk_handle id,char* output,size_t* bytes,char* err,size_t cap){return boundary(err,cap,[&]{
    auto c=context_handle(id);std::lock_guard lock(c->sources_mutex);
    auto text="{\"schema\":1,\"generation\":"+std::to_string(c->epoch.load())+",\"source_entries\":"+std::to_string(c->sources.size())+",\"source_bytes\":"+std::to_string(c->source_bytes)+",\"source_hits\":"+std::to_string(c->source_hits.load())+",\"source_misses\":"+std::to_string(c->source_misses.load())+",\"native_hits\":"+std::to_string(c->native_hits.load())+",\"native_builds\":"+std::to_string(c->native_builds.load())+",\"vulkan_hits\":"+std::to_string(c->vulkan_hits.load())+",\"vulkan_builds\":"+std::to_string(c->vulkan_builds.load())+",\"native_compiler_captured\":"+json(c->compiler)+"}";
    text_query(text,output,bytes);
});}
bt_sdk_status bt_sdk_cache_clear(bt_sdk_handle id,char* err,size_t cap){return boundary(err,cap,[&]{auto c=context_handle(id);std::lock_guard lock(c->sources_mutex);limited(c->epoch.load()!=UINT64_MAX,"cache generation exhausted");++c->epoch;c->sources.clear();c->source_bytes=0;});}

bt_sdk_status bt_sdk_context_create(const bt_sdk_options* options,bt_sdk_handle* output,char* err,size_t cap){return boundary(err,cap,[&]{
    required(output,"null context output");*output=0;auto c=std::make_shared<Context>();
    if(options){required(options->struct_size==sizeof(*options) && options->abi_major==1,"SDK options ABI/size mismatch");required(options->max_elements>0 && options->max_elements<=1000000 && options->operation_budget>0 && options->operation_budget<=100000000,"invalid SDK resource limits");c->maximum=options->max_elements;c->budget=options->operation_budget;required(options->representation<=2 && options->reserved==0,"invalid representation/reserved options");c->representation=options->representation;c->device=options->device_index;c->group=options->workgroup?options->workgroup:64;required(c->group==32||c->group==64||c->group==128,"invalid workgroup");}
    *output=add({c,{}});
});}
bt_sdk_status bt_sdk_release(bt_sdk_handle id,char* err,size_t cap){return boundary(err,cap,[&]{
    Entry retired;
    {std::lock_guard lock(mutex);auto i=registry.find(id);required(i!=registry.end(),"unknown/released SDK handle");if(!i->second.module)i->second.context->closed=true;retired=std::move(i->second);registry.erase(i);}
    // dlclose/device-idle/filesystem cleanup must not hold the global handle
    // registry lock: a slow driver must not block unrelated CPU contexts.
});}
bt_sdk_status bt_sdk_compile(bt_sdk_handle context,const char* source,size_t bytes,bt_sdk_handle* output,char* err,size_t cap){return boundary(err,cap,[&]{
    required(output,"null module output");*output=0;auto e=get(context);required(!e.module,"expected context handle");if(e.context->closed)throw Closed("SDK context closed");
    required(source && bytes,"empty/null source");limited(bytes<=8*1024*1024,"source byte limit exceeded");
    std::string text(source,bytes);required(text.find('\0')==text.npos,"NUL in source");
    std::lock_guard source_lock(e.context->sources_mutex);
    std::shared_ptr<Module> m;
    auto& cache=e.context->sources;
    auto cached=std::find_if(cache.begin(),cache.end(),[&](const auto& item){return item.source==text;});
    if(cached!=cache.end()){
        ++e.context->source_hits;m=cached->module.lock();
        if(!m){m=std::make_shared<Module>();m->context=e.context;m->program=*cached->program;cached->module=m;}
        cache.splice(cache.begin(),cache,cached);
    }else{
        ++e.context->source_misses;
        bt::frontend::Lexer lexer(text);for(auto& token:lexer.tokenize_all())required(token.type!=bt::frontend::TokenType::KwImport,"SDK imports disabled; provide a self-contained module");
        bt::frontend::Parser parser(text);auto ast=parser.parse_program();m=std::make_shared<Module>();m->context=e.context;m->program=bt::ir::Lowering{}.lower(*ast);
        for(auto& f:m->program.functions)required(f.name.size()<=1024,"SDK export name limit");
        required(m->program.body.empty(),"SDK module rejects top-level execution; use function exports");
        while(!cache.empty() && (cache.size()>=8 || e.context->source_bytes+bytes>16*1024*1024)){e.context->source_bytes-=cache.back().source.size();cache.pop_back();}
        cache.push_front({std::move(text),std::make_shared<const bt::ir::Program>(m->program),m});e.context->source_bytes+=bytes;
    }
    // The function-scope sources_mutex also guards initialization: another cache
    // hit cannot observe a Module with partially populated exports/metadata.
    initialize(*m);if(e.context->closed)throw Closed("context closed during compilation");*output=add({e.context,m});
});}
bt_sdk_status bt_sdk_exports(bt_sdk_handle id,char* output,size_t bytes,char* err,size_t cap){return boundary(err,cap,[&]{auto m=module(id);std::string text;for(auto& f:m->program.functions)text+=f.name+"\n";copy(text,output,bytes);});}
bt_sdk_status bt_sdk_shader(bt_sdk_handle id,const char* name,uint32_t group,char* output,size_t* bytes,char* err,size_t cap){return boundary(err,cap,[&]{
    required(bytes,"null shader length");namecheck(name);auto m=module(id);auto shader=bt::kernel::VulkanEmitter{}.generate(m->program,name,group,false,m->context->representation==1);auto available=*bytes;*bytes=shader.size()+1;if(output)copy(shader,output,available);
});}
bt_sdk_status bt_sdk_invoke(bt_sdk_handle id,const char* name,const bt_sdk_value* args,uint32_t count,bt_sdk_value* output,char* err,size_t cap){return bt_sdk_invoke_ex(id,name,BT_SDK_CPU,args,count,output,nullptr,err,cap);}
bt_sdk_status bt_sdk_invoke_ex(bt_sdk_handle id,const char* name,uint32_t backend,const bt_sdk_value* args,uint32_t count,bt_sdk_value* output,bt_sdk_metrics* metrics,char* err,size_t cap){return boundary(err,cap,[&]{
    auto start=Clock::now();auto m=module(id);namecheck(name);required(output && count<=1000 && (!count||args),"invalid scalar invocation storage");
    required(backend<=BT_SDK_NATIVE,"unknown backend");if(backend==BT_SDK_VULKAN)throw bt::sdk::Unavailable("Vulkan supports batch map, not scalar invoke");
    if(metrics)required(metrics->struct_size==sizeof(*metrics),"metrics struct size mismatch");
    const auto& f=export_function(*m,name);required(m->pure[m->exports.at(name)],"SDK export IO forbidden");required(count==f.parameters.size(),"export argument count mismatch");
    std::vector<bt::ir::Value> values;std::vector<uint64_t> payload;
    for(uint32_t i=0;i<count;++i){required(args[i].reserved==0,"nonzero value reserved field");
        if(f.parameters[i].type==bt::ir::Type::Float64){required(args[i].kind==2,"expected binary64 argument");values.emplace_back(args[i].floating);payload.push_back(std::bit_cast<uint64_t>(args[i].floating));}
        else {required(args[i].kind==1,"expected integral argument");bt::math::ScalarInteger::check(args[i].integer);if(f.parameters[i].type==bt::ir::Type::Trit)required(args[i].integer>=-1 && args[i].integer<=1,"invalid trit argument");values.emplace_back(args[i].integer);payload.push_back(std::bit_cast<uint64_t>(args[i].integer));}
    }
    bt_sdk_metrics times{};times.struct_size=sizeof(times);times.backend=backend;times.kernel_ms=times.gpu_upload_ms=times.gpu_download_ms=-1;bt_sdk_value value{};
    if(backend==BT_SDK_CPU){auto begin=Clock::now();std::ostringstream sink;auto result=bt::ir::Executor(m->program,sink,m->context->budget).invoke(name,values);
        if(std::holds_alternative<int64_t>(result)){value.kind=1;value.integer=std::get<int64_t>(result);}else if(std::holds_alternative<double>(result)){value.kind=2;value.floating=std::get<double>(result);}times.execute_ms=elapsed(begin);
    }else{
        limited(bt::sdk::native_work(f)<=m->context->budget,"native static work budget exceeded");
        auto image=native_image(m,f,times);auto begin=Clock::now();auto result=image->invoke(payload.data());times.execute_ms=elapsed(begin);
        if(f.result==bt::ir::Type::Float64){value.kind=2;value.floating=std::bit_cast<double>(result);}else if(f.result!=bt::ir::Type::Void){value.kind=1;value.integer=std::bit_cast<int64_t>(result);}
    }
    times.total_ms=elapsed(start);*output=value;if(metrics)*metrics=times;
});}
bt_sdk_status bt_sdk_map(bt_sdk_handle id,const char* name,uint32_t backend,const int64_t* input,uint64_t count,int64_t* output,bt_sdk_metrics* metrics,char* err,size_t cap){return boundary(err,cap,[&]{
    auto start=Clock::now();auto m=module(id);namecheck(name);required(backend<=BT_SDK_NATIVE,"unknown backend; no implicit fallback");limited(count<=m->context->maximum,"batch limit exceeded");required(!count || (input&&output),"null map buffer");
    if(metrics)required(metrics->struct_size==sizeof(*metrics),"metrics struct size mismatch");
    const auto& f=export_function(*m,name);required(f.parameters.size()==1 && f.parameters[0].type==bt::ir::Type::Int39 && f.result==bt::ir::Type::Int39,"map-v1 signature must be (int39)->int39");auto work=bt::kernel::statements(f.body);limited(!count || count<=100000000/std::max(size_t(1),work),"map aggregate work limit exceeded");std::vector<int64_t> values;if(count)values.assign(input,input+count);
    for(auto x:values)bt::math::ScalarInteger::check(x);
    bt_sdk_metrics times{};times.struct_size=sizeof(times);times.backend=backend;times.workgroup=m->context->group;times.kernel_ms=times.gpu_upload_ms=times.gpu_download_ms=-1;
    std::vector<int64_t> result;
    if(backend==BT_SDK_CPU){std::ostringstream sink;bt::ir::Executor executor(m->program,sink,m->context->budget);result.reserve(count);auto begin=Clock::now();for(auto x:values)result.push_back(std::get<int64_t>(executor.invoke(name,{x})));times.execute_ms=std::chrono::duration<double,std::milli>(Clock::now()-begin).count();}
    else if(backend==BT_SDK_NATIVE){limited(!count || count<=m->context->budget/std::max(size_t(1),work),"native aggregate work budget exceeded");auto image=native_image(m,f,times);result.reserve(count);auto begin=Clock::now();for(auto x:values){auto bits=std::bit_cast<uint64_t>(x);result.push_back(std::bit_cast<int64_t>(image->invoke(&bits)));}times.execute_ms=elapsed(begin);}
    else {auto image=vulkan_image(m,name,times);result=image->execute(values,times);}
    if(!result.empty())std::copy(result.begin(),result.end(),output);times.total_ms=std::chrono::duration<double,std::milli>(Clock::now()-start).count();if(metrics)*metrics=times;
});}
bt_sdk_status bt_sdk_devices(char* output,size_t bytes,char* err,size_t cap){return boundary(err,cap,[&]{copy(bt::sdk::devices(),output,bytes);});}
bt_sdk_status bt_sdk_dual_encode(int64_t n,uint32_t width,uint64_t* p,uint64_t* negative,char* err,size_t cap){return boundary(err,cap,[&]{required(p&&negative,"null rail output");auto w=bt::math::encode_integer(n,width);*p=w.positive;*negative=w.negative;});}
bt_sdk_status bt_sdk_dual_decode(uint64_t p,uint64_t n,uint32_t width,int64_t* output,char* err,size_t cap){return boundary(err,cap,[&]{required(output,"null scalar output");*output=bt::math::decode_integer({p,n},width);});}
bt_sdk_status bt_sdk_matmul(const int64_t* a,uint64_t na,const int64_t* b,uint64_t nb,int64_t* output,uint64_t no,uint64_t m,uint64_t k,uint64_t n,char* err,size_t cap){return boundary(err,cap,[&]{required(na<=1000000&&nb<=1000000&&no<=1000000&&m<=1000000&&k<=1000000&&n<=1000000,"matrix storage/dimension limit");required((!na||a)&&(!nb||b)&&(!no||output),"null matrix storage");bt::math::matmul(std::span(a,size_t(na)),std::span(b,size_t(nb)),std::span(output,size_t(no)),m,k,n);});}
bt_sdk_status bt_sdk_float_from_ratio(uint32_t width,int64_t n,int64_t d,int64_t exponent,bt_sdk_float_value* output,char* err,size_t cap){return boundary(err,cap,[&]{
    required(output,"null BTFP output");float_width(width,[&]<int W>(){auto r=bt::BTFP<W>::from_ratio(n,d,exponent);*output=write_float(r);});
});}
bt_sdk_status bt_sdk_float_from_double(uint32_t width,double value,bt_sdk_float_value* output,char* err,size_t cap){return boundary(err,cap,[&]{
    required(output,"null BTFP output");float_width(width,[&]<int W>(){*output=write_float(bt::BTFP<W>::from_double(value));});
});}
bt_sdk_status bt_sdk_float_to_double(const bt_sdk_float_value* input,double* output,char* err,size_t cap){return boundary(err,cap,[&]{
    required(input && output,"null BTFP storage");float_width(input->width,[&]<int W>(){*output=read_float<W>(input).to_double();});
});}
bt_sdk_status bt_sdk_float_apply(uint32_t op,const bt_sdk_float_value* a,const bt_sdk_float_value* b,const bt_sdk_float_value* c,bt_sdk_float_value* output,char* err,size_t cap){return boundary(err,cap,[&]{
    required(a && output && op>=1 && op<=6,"invalid BTFP operation/storage");
    float_width(a->width,[&]<int W>(){auto x=read_float<W>(a);bt::BTFP<W> r;
        if(op==BT_SDK_FP_SQRT)r=x.sqrt();
        else {auto y=read_float<W>(b);switch(op){
            case BT_SDK_FP_ADD:r=x+y;break;case BT_SDK_FP_SUB:r=x-y;break;
            case BT_SDK_FP_MUL:r=x*y;break;case BT_SDK_FP_DIV:r=x/y;break;
            case BT_SDK_FP_FMA:r=bt::BTFP<W>::fma(x,y,read_float<W>(c));break;
        }}*output=write_float(r);
    });
});}

}
