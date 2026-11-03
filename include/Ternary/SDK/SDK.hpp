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
    TernaryFloat pow_int(int32_t power)const{bt_sdk_float_value out;char err[4096];check(bt_sdk_float_powi(&value,power,&out,err,sizeof err),err);return TernaryFloat(out);}
    static TernaryFloat reduce(uint32_t width,uint32_t operation,const std::vector<TernaryFloat>& a,const std::vector<TernaryFloat>& b={}){
        if(a.size()>4096 || b.size()>4096)throw Error(BT_SDK_RESOURCE,"BTFP reduction count exceeds 4096");
        if(operation==BT_SDK_FP_DOT && a.size()!=b.size())throw Error(BT_SDK_INVALID,"dot shape mismatch");
        std::vector<bt_sdk_float_value> av,bv;for(auto& v:a)av.push_back(v.value);for(auto& v:b)bv.push_back(v.value);
        bt_sdk_float_value out;char err[4096];check(bt_sdk_float_reduce(width,operation,av.data(),bv.data(),av.size(),&out,err,sizeof err),err);return TernaryFloat(out);
    }
    TernaryFloat polynomial(const std::vector<TernaryFloat>& c)const{
        if(c.size()>17)throw Error(BT_SDK_RESOURCE,"polynomial degree exceeds 16");
        std::vector<bt_sdk_float_value> cv;for(auto& v:c)cv.push_back(v.value);bt_sdk_float_value out;char err[4096];
        check(bt_sdk_float_polynomial(&value,cv.data(),cv.size(),&out,err,sizeof err),err);return TernaryFloat(out);
    }
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

/** Fast ternary GEMM: C[m,n] = A[m,k] * B[k,n], exact int64 result.
 *
 * Operands are dual-rail word arrays. A is row-major with ceil(k/64) words
 * per row; B is supplied transposed, as n rows of ceil(k/64) words. Weights
 * are 1.6 bits/trit in packed form and 2 bits/trit dual-rail, against 32 for
 * FP32 - so for large weights this is a memory-bandwidth win as much as a
 * compute one.
 *
 * CPU-only, no handle required. Backed by the register-blocked micro-kernel
 * (bt::blas::ternary_gemm), measured at ~213 G MAC/s at n=1024 on an
 * AVX-512 VPOPCNTDQ machine against ~62 for the previous autotuner.
 */
inline void gemm_ternary(const std::vector<uint64_t>& Ap,
                         const std::vector<uint64_t>& An,
                         const std::vector<uint64_t>& Bp,
                         const std::vector<uint64_t>& Bn,
                         std::vector<int64_t>& C,
                         uint64_t m, uint64_t n, uint64_t k) {
    C.assign(size_t(m) * n, 0);
    char err[4096];
    // Qualified: this helper is appended after `namespace bt::sdk` closes.
    bt::sdk::check(bt_sdk_gemm_ternary(Ap.data(), An.data(), Bp.data(), Bn.data(),
                                       C.data(), m, n, k, err, sizeof err), err);
}

/** True if the loaded library exports the fast GEMM (API >= 1.3). An older
 *  libtrium_sdk without it must fail clearly, not silently fall back. */
inline bool has_gemm_ternary() { return bt_sdk_api_version() >= 0x00010003; }
