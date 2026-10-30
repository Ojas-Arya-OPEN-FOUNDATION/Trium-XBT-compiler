#ifndef TRIUM_SDK_HPP
#define TRIUM_SDK_HPP
#include "sdk.h"
#include <string>
#include <vector>
#include <stdexcept>
#include <utility>
#include <memory>
#include <atomic>
#include <variant>
namespace bt::sdk {
struct Error:std::runtime_error{int code;Error(int c,const char* m):std::runtime_error(m),code(c){}};
inline void check(int status,const char* message){if(status)throw Error(status,message);}
class TernaryFloat {
    bt_sdk_float_value value{};
    explicit TernaryFloat(bt_sdk_float_value v):value(v){}
public:
    static TernaryFloat from_ratio(uint32_t width,int64_t n,int64_t d=1,int64_t exponent=0){bt_sdk_float_value v;char err[4096];check(bt_sdk_float_from_ratio(width,n,d,exponent,&v,err,sizeof err),err);return TernaryFloat(v);}
    static TernaryFloat from_double(uint32_t width,double x){bt_sdk_float_value v;char err[4096];check(bt_sdk_float_from_double(width,x,&v,err,sizeof err),err);return TernaryFloat(v);}
    const bt_sdk_float_value& record()const{return value;}
    double to_double()const{double out;char err[4096];check(bt_sdk_float_to_double(&value,&out,err,sizeof err),err);return out;}
    TernaryFloat apply(uint32_t op,const TernaryFloat* b=nullptr,const TernaryFloat* c=nullptr)const{bt_sdk_float_value out;char err[4096];check(bt_sdk_float_apply(op,&value,b?&b->value:nullptr,c?&c->value:nullptr,&out,err,sizeof err),err);return TernaryFloat(out);}
};
// Owning RAII contexts are retained by modules. Explicit close invalidates later
// calls; already accepted C calls retain their own resources. Moving/destroying
// a wrapper concurrently with a method call is ordinary invalid C++ object use.
template<class F> std::string query(F fn){
    size_t size=0;char err[4096];check(fn(nullptr,&size,err,sizeof err),err);
    for(unsigned attempt=0;attempt<4;++attempt){
        if(!size || size>16*1024*1024)throw Error(BT_SDK_RESOURCE,"metadata capacity exceeded");
        std::vector<char> data(size);auto capacity=size;auto status=fn(data.data(),&size,err,sizeof err);
        if(status==BT_SDK_INVALID && size>capacity)continue;check(status,err);return std::string(data.data());
    }throw Error(BT_SDK_RESOURCE,"metadata changed repeatedly; retry");
}
inline std::string info(){return query([](char* p,size_t* n,char* e,size_t c){return bt_sdk_info(p,n,e,c);});}
struct ContextState {
    std::atomic<bt_sdk_handle> id{0};
    ~ContextState(){if(auto h=id.exchange(0))bt_sdk_release(h,nullptr,0);}
};
using Scalar=std::variant<std::monostate,int64_t,double>;
class Module {
    std::atomic<bt_sdk_handle> id{0};std::shared_ptr<ContextState> parent;
public:
    explicit Module(bt_sdk_handle h,std::shared_ptr<ContextState> owner={}):id(h),parent(std::move(owner)){}
    Module(const Module&)=delete;Module& operator=(const Module&)=delete;
    Module(Module&& m) noexcept:id(m.id.exchange(0)),parent(std::move(m.parent)){}
    Module& operator=(Module&& m)noexcept{if(this!=&m){if(auto h=id.exchange(m.id.exchange(0)))bt_sdk_release(h,nullptr,0);parent=std::move(m.parent);}return *this;}
    ~Module(){if(auto h=id.exchange(0))bt_sdk_release(h,nullptr,0);}
    void close(){if(auto h=id.exchange(0)){char err[4096];check(bt_sdk_release(h,err,sizeof err),err);}}
    std::string info()const{return query([&](char* p,size_t* n,char* e,size_t c){return bt_sdk_module_info(id.load(),p,n,e,c);});}
    std::vector<int64_t> map(const std::string& name,const std::vector<int64_t>& input,uint32_t backend=BT_SDK_CPU)const {
        if(name.find('\0')!=std::string::npos)throw Error(BT_SDK_INVALID,"NUL in export name");
        std::vector<int64_t> output(input.size());char err[4096];check(bt_sdk_map(id.load(),name.c_str(),backend,input.data(),input.size(),output.data(),nullptr,err,sizeof err),err);return output;
    }
    Scalar invoke(const std::string& name,const std::vector<Scalar>& args,uint32_t backend=BT_SDK_CPU)const{
        if(name.find('\0')!=std::string::npos || args.size()>1000)throw Error(BT_SDK_INVALID,"invalid scalar arguments/name");
        std::vector<bt_sdk_value> values;for(auto& a:args){bt_sdk_value v{};
            if(auto n=std::get_if<int64_t>(&a)){v.kind=1;v.integer=*n;}else if(auto n=std::get_if<double>(&a)){v.kind=2;v.floating=*n;}else throw Error(BT_SDK_INVALID,"void input argument");values.push_back(v);}
        bt_sdk_value out{};char err[4096];check(bt_sdk_invoke_ex(id.load(),name.c_str(),backend,values.data(),uint32_t(values.size()),&out,nullptr,err,sizeof err),err);
        if(out.kind==1)return out.integer;if(out.kind==2)return out.floating;return std::monostate{};
    }
};
class Context {
    std::shared_ptr<ContextState> state=std::make_shared<ContextState>();
public:
    explicit Context(const bt_sdk_options* options=nullptr){bt_sdk_handle h=0;char err[4096];check(bt_sdk_context_create(options,&h,err,sizeof err),err);state->id=h;}
    Context(const Context&)=delete;Context& operator=(const Context&)=delete;
    Context(Context&&)=default;Context& operator=(Context&&)=default;
    void close(){if(state)if(auto h=state->id.exchange(0)){char err[4096];check(bt_sdk_release(h,err,sizeof err),err);}}
    void clear_cache()const{char err[4096];check(bt_sdk_cache_clear((state?state->id.load():0),err,sizeof err),err);}
    std::string cache_info()const{return query([&](char* p,size_t* n,char* e,size_t c){return bt_sdk_cache_info((state?state->id.load():0),p,n,e,c);});}
    Module compile(const std::string& source)const {bt_sdk_handle m;char err[4096];check(bt_sdk_compile((state?state->id.load():0),source.data(),source.size(),&m,err,sizeof err),err);return Module(m,state);}
};
}
#endif
