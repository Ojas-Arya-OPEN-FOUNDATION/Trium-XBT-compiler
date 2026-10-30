#ifndef TRIUM_SDK_H
#define TRIUM_SDK_H
#include <stddef.h>
#include <stdint.h>
#if defined(_WIN32)
#define BT_SDK_API __declspec(dllexport)
#else
#define BT_SDK_API __attribute__((visibility("default")))
#endif
#ifdef __cplusplus
extern "C" {
#endif
/* SDK ABI v1. Handles are process-local IDs, never serialized pointers. All
 * calls synchronous. Raw arrays require valid caller-owned storage. Concurrent
 * calls on immutable modules supported; release invalidates ID, not live calls.
 * Context close rejects subsequent module operations; release remains legal. */
typedef uint64_t bt_sdk_handle;
typedef int32_t bt_sdk_status;
enum { BT_SDK_OK=0,BT_SDK_INVALID=1,BT_SDK_OVERFLOW=2,BT_SDK_INTERNAL=3,
       BT_SDK_UNAVAILABLE=4,BT_SDK_CLOSED=5,BT_SDK_RESOURCE=6,BT_SDK_DEVICE=7,BT_SDK_OOM=8 };
enum { BT_SDK_CPU=0,BT_SDK_VULKAN=1,BT_SDK_NATIVE=2 };
enum { BT_SDK_WORD32=0,BT_SDK_NATIVE64=1,BT_SDK_AUTO=2 };
typedef struct {uint32_t struct_size,abi_major;uint64_t max_elements,operation_budget;uint32_t device_index,workgroup,representation,reserved;} bt_sdk_options;
typedef struct {uint32_t struct_size,backend;double compile_ms,upload_ms,execute_ms,download_ms,total_ms,kernel_ms,gpu_upload_ms,gpu_download_ms;uint32_t workgroup,representation;uint8_t device_uuid[16];} bt_sdk_metrics;
/* Finite BTFP profile 1: normalized mantissa * 3^exponent, width 1..40,
 * int32 exponent, canonical zero. All records are values, never handles.
 * Nearest ternary result, ties away; binary64 export uses IEEE ties-to-even.
 * Overflow/underflow reject, no NaN/Inf/subnormal BTFP encodings. */
typedef struct {uint32_t width,reserved;int64_t mantissa;int32_t exponent;uint32_t reserved2;} bt_sdk_float_value;
enum {BT_SDK_FP_ADD=1,BT_SDK_FP_SUB=2,BT_SDK_FP_MUL=3,BT_SDK_FP_DIV=4,BT_SDK_FP_SQRT=5,BT_SDK_FP_FMA=6};
BT_SDK_API bt_sdk_status bt_sdk_float_from_ratio(uint32_t,int64_t,int64_t,int64_t,bt_sdk_float_value*,char*,size_t);
BT_SDK_API bt_sdk_status bt_sdk_float_from_double(uint32_t,double,bt_sdk_float_value*,char*,size_t);
BT_SDK_API bt_sdk_status bt_sdk_float_to_double(const bt_sdk_float_value*,double*,char*,size_t);
BT_SDK_API bt_sdk_status bt_sdk_float_apply(uint32_t,const bt_sdk_float_value*,const bt_sdk_float_value*,const bt_sdk_float_value*,bt_sdk_float_value*,char*,size_t);
BT_SDK_API uint32_t bt_sdk_abi_version(void);
/* Additive API 1.1; ABI-1 structs/layouts remain unchanged. Candidate compiler
 * discovery is NOT a guarantee that compilation/device qualification succeeds.
 * JSON APIs use length query then copy, including NUL in *bytes. */
BT_SDK_API uint32_t bt_sdk_api_version(void);
BT_SDK_API bt_sdk_status bt_sdk_info(char*,size_t*,char*,size_t);
BT_SDK_API bt_sdk_status bt_sdk_module_info(bt_sdk_handle,char*,size_t*,char*,size_t);
BT_SDK_API bt_sdk_status bt_sdk_cache_info(bt_sdk_handle,char*,size_t*,char*,size_t);
BT_SDK_API bt_sdk_status bt_sdk_cache_clear(bt_sdk_handle,char*,size_t);

/* Message is copied to caller-owned storage; includes NUL when capacity>0.
 * No diagnostic callbacks, allocator sharing or borrowed TLS strings. */
BT_SDK_API bt_sdk_status bt_sdk_context_create(const bt_sdk_options*,bt_sdk_handle*,char*,size_t);
BT_SDK_API bt_sdk_status bt_sdk_release(bt_sdk_handle,char*,size_t);
BT_SDK_API bt_sdk_status bt_sdk_compile(bt_sdk_handle,const char*,size_t,bt_sdk_handle*,char*,size_t);
BT_SDK_API bt_sdk_status bt_sdk_exports(bt_sdk_handle,char*,size_t,char*,size_t);
/* Scalar invocation: kind 1=int64, 2=binary64, 0=void output. Reserved=0.
 * Inputs are exact types (no implicit float/int conversion at the ABI). */
typedef struct {uint32_t kind,reserved;int64_t integer;double floating;} bt_sdk_value;
/* native-v1: pure loop-free scalar functions, no calls/IO; explicit unavailable
 * otherwise. Static worst-case work admission; no silent reference fallback. */
BT_SDK_API bt_sdk_status bt_sdk_invoke_ex(bt_sdk_handle,const char*,uint32_t,const bt_sdk_value*,uint32_t,bt_sdk_value*,bt_sdk_metrics*,char*,size_t);
BT_SDK_API bt_sdk_status bt_sdk_invoke(bt_sdk_handle,const char*,const bt_sdk_value*,uint32_t,bt_sdk_value*,char*,size_t);
BT_SDK_API bt_sdk_status bt_sdk_map(bt_sdk_handle,const char*,uint32_t,const int64_t*,uint64_t,int64_t*,bt_sdk_metrics*,char*,size_t);
/* Length-query then copy. If output=NULL, *bytes receives size including NUL. */
BT_SDK_API bt_sdk_status bt_sdk_shader(bt_sdk_handle,const char*,uint32_t,char*,size_t*,char*,size_t);
BT_SDK_API bt_sdk_status bt_sdk_devices(char*,size_t,char*,size_t);
BT_SDK_API bt_sdk_status bt_sdk_dual_encode(int64_t,uint32_t,uint64_t*,uint64_t*,char*,size_t);
BT_SDK_API bt_sdk_status bt_sdk_dual_decode(uint64_t,uint64_t,uint32_t,int64_t*,char*,size_t);
BT_SDK_API bt_sdk_status bt_sdk_matmul(const int64_t*,uint64_t,const int64_t*,uint64_t,int64_t*,uint64_t,uint64_t,uint64_t,uint64_t,char*,size_t);
#ifdef __cplusplus
}
#endif
#endif
