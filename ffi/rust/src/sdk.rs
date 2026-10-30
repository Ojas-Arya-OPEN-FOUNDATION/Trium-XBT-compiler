//! Synchronous SDK v1. The raw ABI is private; owned modules retain their context.
//! Linux linking requires libtrium_sdk and libbt_runtime from the same build.
use std::{ffi::{c_char,CString},sync::Arc};
#[derive(Debug)]
pub struct Error {pub code:i32,pub message:String}
impl std::fmt::Display for Error {fn fmt(&self,f:&mut std::fmt::Formatter<'_>)->std::fmt::Result{write!(f,"SDK error {}: {}",self.code,self.message)}}
impl std::error::Error for Error {}
#[repr(C)]
struct Options {size:u32,abi:u32,maximum:u64,budget:u64,device:u32,group:u32,representation:u32,reserved:u32}
#[link(name="trium_sdk")]
extern "C" {
 fn bt_sdk_abi_version()->u32;
 fn bt_sdk_api_version()->u32;
 fn bt_sdk_info(out:*mut c_char,bytes:*mut usize,error:*mut c_char,capacity:usize)->i32;
 fn bt_sdk_module_info(id:u64,out:*mut c_char,bytes:*mut usize,error:*mut c_char,capacity:usize)->i32;
 fn bt_sdk_cache_info(id:u64,out:*mut c_char,bytes:*mut usize,error:*mut c_char,capacity:usize)->i32;
 fn bt_sdk_cache_clear(id:u64,error:*mut c_char,capacity:usize)->i32;
 fn bt_sdk_invoke_ex(id:u64,name:*const c_char,backend:u32,args:*const Slot,count:u32,out:*mut Slot,metrics:*mut std::ffi::c_void,error:*mut c_char,capacity:usize)->i32;

 fn bt_sdk_context_create(options:*const Options,out:*mut u64,error:*mut c_char,capacity:usize)->i32;
 fn bt_sdk_release(handle:u64,error:*mut c_char,capacity:usize)->i32;
 fn bt_sdk_compile(context:u64,source:*const u8,bytes:usize,out:*mut u64,error:*mut c_char,capacity:usize)->i32;
 fn bt_sdk_map(module:u64,name:*const c_char,backend:u32,input:*const i64,count:u64,output:*mut i64,metrics:*mut std::ffi::c_void,error:*mut c_char,capacity:usize)->i32;
}
fn checked(f:impl FnOnce(*mut c_char,usize)->i32)->Result<(),Error>{
 let mut text=[0u8;4096];let code=f(text.as_mut_ptr().cast(),text.len());
 if code==0 {Ok(())}else{let end=text.iter().position(|x|*x==0).unwrap_or(text.len());Err(Error{code,message:String::from_utf8_lossy(&text[..end]).into_owned()})}
}
fn query(f:impl Fn(*mut c_char,*mut usize,*mut c_char,usize)->i32)->Result<String,Error>{
 let mut size=0;checked(|e,n|f(std::ptr::null_mut(),&mut size,e,n))?;
 for _ in 0..4 {
  if size==0||size>16*1024*1024{return Err(Error{code:6,message:"metadata capacity".into()});}
  let mut data=vec![0u8;size];let capacity=size;
  match checked(|e,n|f(data.as_mut_ptr().cast(),&mut size,e,n)){
   Ok(())=>{let end=data.iter().position(|x|*x==0).unwrap_or(data.len());return String::from_utf8(data[..end].to_vec()).map_err(|_|Error{code:3,message:"invalid SDK metadata encoding".into()});},
   Err(error)=>{if error.code!=1||size<=capacity{return Err(error);}}
  }
 }Err(Error{code:6,message:"metadata changed repeatedly".into()})
}
#[derive(Debug,Clone,Copy,PartialEq)]
pub enum Scalar {Void,Integer(i64),Float(f64)}
#[repr(C)]
#[derive(Default)]
struct Slot {kind:u32,reserved:u32,integer:i64,floating:f64}
pub fn info()->Result<String,Error>{query(|p,s,e,n|unsafe{bt_sdk_info(p,s,e,n)})}
struct Inner(u64);
impl Drop for Inner {fn drop(&mut self){unsafe{bt_sdk_release(self.0,std::ptr::null_mut(),0);}}}
#[derive(Clone)]
pub struct Context(Arc<Inner>);
impl Context {
 pub fn new()->Result<Self,Error>{
  if unsafe{bt_sdk_abi_version()}!=1 || unsafe{bt_sdk_api_version()}<0x10001{return Err(Error{code:1,message:"SDK ABI mismatch".into()});}
  let mut handle=0;checked(|e,n|unsafe{bt_sdk_context_create(std::ptr::null(),&mut handle,e,n)})?;Ok(Self(Arc::new(Inner(handle))))
 }
 pub fn cache_info(&self)->Result<String,Error>{query(|p,s,e,n|unsafe{bt_sdk_cache_info(self.0.0,p,s,e,n)})}
 pub fn clear_cache(&self)->Result<(),Error>{checked(|e,n|unsafe{bt_sdk_cache_clear(self.0.0,e,n)})}
 pub fn compile(&self,source:&str)->Result<Module,Error>{let mut handle=0;checked(|e,n|unsafe{bt_sdk_compile(self.0.0,source.as_ptr(),source.len(),&mut handle,e,n)})?;Ok(Module{handle,_context:self.0.clone()})}
}
#[derive(Debug,Clone,Copy,PartialEq,Eq)]
pub enum Backend {Cpu,Vulkan,Native}
pub struct Module {handle:u64,_context:Arc<Inner>}
impl Module {
 pub fn info(&self)->Result<String,Error>{query(|p,s,e,n|unsafe{bt_sdk_module_info(self.handle,p,s,e,n)})}
 pub fn invoke(&self,name:&str,args:&[Scalar],backend:Backend)->Result<Scalar,Error>{
  if args.len()>1000{return Err(Error{code:1,message:"argument limit".into()});}
  let name=CString::new(name).map_err(|_|Error{code:1,message:"NUL in name".into()})?;
  let mut slots=Vec::new();for a in args{slots.push(match a{Scalar::Integer(x)=>Slot{kind:1,integer:*x,..Slot::default()},Scalar::Float(x)=>Slot{kind:2,floating:*x,..Slot::default()},Scalar::Void=>return Err(Error{code:1,message:"void input".into()})});}
  let mut out=Slot::default();let selected=match backend{Backend::Cpu=>0,Backend::Vulkan=>1,Backend::Native=>2};
  checked(|e,n|unsafe{bt_sdk_invoke_ex(self.handle,name.as_ptr(),selected,slots.as_ptr(),slots.len()as u32,&mut out,std::ptr::null_mut(),e,n)})?;
  Ok(match out.kind{1=>Scalar::Integer(out.integer),2=>Scalar::Float(out.floating),_=>Scalar::Void})
 }
 pub fn map(&self,name:&str,input:&[i64],backend:Backend)->Result<Vec<i64>,Error>{
  let name=CString::new(name).map_err(|_|Error{code:1,message:"NUL in function name".into()})?;
  if input.len()>1000000{return Err(Error{code:6,message:"batch limit".into()});}
  let mut output=vec![0;input.len()];let backend=match backend{Backend::Cpu=>0,Backend::Vulkan=>1,Backend::Native=>2};
  checked(|e,n|unsafe{bt_sdk_map(self.handle,name.as_ptr(),backend,input.as_ptr(),input.len()as u64,output.as_mut_ptr(),std::ptr::null_mut(),e,n)})?;Ok(output)
 }
}
impl Drop for Module {fn drop(&mut self){unsafe{bt_sdk_release(self.handle,std::ptr::null_mut(),0);}}}
#[cfg(test)]mod tests {
 use super::*;
 #[test]fn owned_cpu_module(){let module={let context=Context::new().unwrap();context.compile("func transform(x:int)->int{return x*3;}").unwrap()};assert_eq!(module.map("transform",&[-2,0,2],Backend::Cpu).unwrap(),vec![-6,0,6]);}
 #[test]fn concurrent_module(){let context=Context::new().unwrap();let module=Arc::new(context.compile("func transform(x:int)->int{return x+1;}").unwrap());let workers:Vec<_>=(0..8).map(|i|{let m=module.clone();std::thread::spawn(move||assert_eq!(m.map("transform",&[i],Backend::Cpu).unwrap(),vec![i+1]))}).collect();for w in workers{w.join().unwrap();}}
 #[test]fn metadata_and_scalar(){let c=Context::new().unwrap();let m=c.compile("func f(x:float)->float{return fma(x,2.0,0.5);}").unwrap();assert!(m.info().unwrap().contains("binary64"));assert_eq!(m.invoke("f",&[Scalar::Float(1.25)],Backend::Cpu).unwrap(),Scalar::Float(3.0));c.clear_cache().unwrap();assert!(c.cache_info().unwrap().contains("generation"));}

}
