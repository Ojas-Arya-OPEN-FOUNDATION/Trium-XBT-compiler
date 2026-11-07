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
 fn bt_sdk_gemm_ternary(ap:*const u64,an:*const u64,bp:*const u64,bn:*const u64,output:*mut i64,m:u64,n:u64,k:u64,error:*mut c_char,capacity:usize)->i32;
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
/// Encode ternary matrices into the dual-rail layout `gemm_ternary` expects.
///
/// `a`  is `m` rows of `k` trits, row-major.
/// `bt` is `n` rows of `k` trits, i.e. B already TRANSPOSED (row j is B's
///      column j).
///
/// The two operands are stored differently at the C ABI - A row-major, B
/// word-plane-major - and the difference is invisible for k <= 64, so getting
/// it wrong by hand yields plausible wrong numbers. Use this and the layout is
/// correct by construction. Returns (ap, an, bp, bn).
pub fn encode_gemm(a: &[&[i8]], bt: &[&[i8]], k: usize)
    -> Result<(Vec<u64>, Vec<u64>, Vec<u64>, Vec<u64>), Error>
{
    if k == 0 { return Err(Error{code:1,message:"k must be positive".into()}); }
    let kw = (k + 63) / 64;
    for row in a.iter().chain(bt.iter()) {
        if row.len() != k {
            return Err(Error{code:1,message:"every row must have exactly k trits".into()});
        }
    }
    let mut ap = vec![0u64; a.len()*kw];
    let mut an = vec![0u64; a.len()*kw];
    for (i, row) in a.iter().enumerate() {
        for t in 0..k {
            match row[t] {
                1  => ap[i*kw + t/64] |= 1u64 << (t % 64),
                -1 => an[i*kw + t/64] |= 1u64 << (t % 64),
                _  => {}
            }
        }
    }
    // B: word-plane-major. Word w of B^T row j lives at index w*n + j.
    let n = bt.len();
    let mut bp = vec![0u64; n*kw];
    let mut bn = vec![0u64; n*kw];
    for (j, row) in bt.iter().enumerate() {
        for t in 0..k {
            match row[t] {
                1  => bp[(t/64)*n + j] |= 1u64 << (t % 64),
                -1 => bn[(t/64)*n + j] |= 1u64 << (t % 64),
                _  => {}
            }
        }
    }
    Ok((ap, an, bp, bn))
}

/// Pack A[m,k] -- `m` rows of `k` trits, row-major -- into dual-rail words.
///
/// Word w of row i is `ap[i*kw + w]`, with kw = ceil(k/64).
pub fn pack_matrix_a(a:&[&[i8]])->Result<(Vec<u64>,Vec<u64>),Error>{
    if a.is_empty(){return Err(Error{code:1,message:"a must have at least one row".into()});}
    let k=a[0].len();
    if k==0{return Err(Error{code:1,message:"k must be positive".into()});}
    for row in a.iter(){if row.len()!=k{return Err(Error{code:1,message:"every row of a must have exactly k trits".into()});}}
    let kw=(k+63)/64;
    let mut ap=vec![0u64;a.len()*kw];
    let mut an=vec![0u64;a.len()*kw];
    for (i,row) in a.iter().enumerate(){
        for t in 0..k{
            match row[t]{
                1  => ap[i*kw + t/64] |= 1u64 << (t%64),
                -1 => an[i*kw + t/64] |= 1u64 << (t%64),
                _  => {}
            }
        }
    }
    Ok((ap,an))
}

/// Pack B[k,n] -- `k` rows of `n` trits, row-major -- into dual-rail words.
///
/// The result is B TRANSPOSED and then WORD-PLANE-MAJOR: word w of B^T row j
/// (that is, column j of B) is `bp[w*n + j]`. This is NOT row-major, and the
/// two orderings coincide only when k <= 64 -- so a hand-packed multi-word B
/// that looks right will silently return wrong values. Prefer this helper, or
/// `encode_gemm` to pack both operands at once.
pub fn pack_matrix_b(b:&[&[i8]])->Result<(Vec<u64>,Vec<u64>),Error>{
    if b.is_empty(){return Err(Error{code:1,message:"b must have at least one row".into()});}
    let n=b[0].len();
    let k=b.len();
    if n==0||k==0{return Err(Error{code:1,message:"b dimensions must be positive".into()});}
    for row in b.iter(){if row.len()!=n{return Err(Error{code:1,message:"every row of b must have exactly n trits".into()});}}
    let kw=(k+63)/64;
    let mut bp=vec![0u64;n*kw];
    let mut bn=vec![0u64;n*kw];
    for (t,row) in b.iter().enumerate(){
        for j in 0..n{
            match row[j]{
                1  => bp[(t/64)*n + j] |= 1u64 << (t%64),
                -1 => bn[(t/64)*n + j] |= 1u64 << (t%64),
                _  => {}
            }
        }
    }
    Ok((bp,bn))
}

/// Exact ternary GEMM: C[m,n] = A[m,k] * B[k,n].
///
/// `ap`/`an` are A's P/N rail words: m rows of ceil(k/64) words, ROW-MAJOR,
/// so word w of row i is `ap[i*kw + w]`.
///
/// `bp`/`bn` are B TRANSPOSED and then WORD-PLANE-MAJOR: word w of B^T row j
/// is `bp[w*n + j]`. This is NOT row-major. The two orderings coincide only
/// when k <= 64, so a hand-built multi-word B that looks right will silently
/// return wrong values. Prefer `encode_gemm`, which lays both out correctly.
///
/// Arithmetic is exact int64 - every partial product lies in {-1,0,+1}, so
/// there is no rounding at any K that fits. Requires SDK API >= 1.3.
pub fn gemm_ternary(ap:&[u64],an:&[u64],bp:&[u64],bn:&[u64],m:u64,n:u64,k:u64)->Result<Vec<i64>,Error>{
 if m==0||n==0||k==0 {return Err(Error{code:1,message:"empty gemm dimension".into()});}
 let kw=((k+63)/64) as usize;
 if ap.len()<(m as usize)*kw||an.len()<(m as usize)*kw||bp.len()<(n as usize)*kw||bn.len()<(n as usize)*kw {
  return Err(Error{code:1,message:"rail storage too small for m,n,k".into()});}
 let mut out=vec![0i64;(m as usize)*(n as usize)];
 checked(|e,c| unsafe{bt_sdk_gemm_ternary(ap.as_ptr(),an.as_ptr(),bp.as_ptr(),bn.as_ptr(),out.as_mut_ptr(),m,n,k,e,c)})?;
 Ok(out)
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
/// True when every element is a trit in `{-1, 0, +1}`.
///
/// This is the routing predicate for [`matmul`]: the ternary GEMM is exact
/// only because every partial product lies in `{-1,0,+1}`, so deciding the
/// route on anything looser would be a correctness bug, not just a slowdown.
pub fn is_ternary(values:&[i8])->bool {
 values.iter().all(|v| (-1..=1).contains(v))
}

/// `C[m,n] = A[m,k] * B[k,n]`, auto-routing to the ternary GEMM.
///
/// Mirrors the Python `matmul`. When every element of both operands is a
/// trit and the library exposes `gemm_ternary`, this packs and routes;
/// otherwise it falls back to the reference kernel. Both paths must agree
/// bit-for-bit, which `force_reference` exists to let you check.
pub fn matmul(a:&[i8],b:&[i8],m:usize,k:usize,n:usize,force_reference:bool)
 ->Result<Vec<i64>,Error>{
 if m*k!=a.len()||k*n!=b.len() {
  return Err(Error{code:1,message:"matrix shapes/storage mismatch".into()});
 }
 if m==0||n==0||k==0 {return Ok(Vec::new());}
 if !force_reference && is_ternary(a) && is_ternary(b) {
  let mut ar:Vec<&[i8]>=Vec::with_capacity(m);
  for i in 0..m {ar.push(&a[i*k..(i+1)*k]);}
  let mut br:Vec<&[i8]>=Vec::with_capacity(k);
  for i in 0..k {br.push(&b[i*n..(i+1)*n]);}
  let (ap,an)=pack_matrix_a(&ar)?;
  let (bp,bn)=pack_matrix_b(&br)?;
  return gemm_ternary(&ap,&an,&bp,&bn,m as u64,n as u64,k as u64);
 }
 // Reference path: dense int64 accumulation, exact for any i8 operand.
 let mut out=vec![0i64;m*n];
 for i in 0..m {for j in 0..n {
  let mut acc=0i64;
  for t in 0..k {acc+=a[i*k+t] as i64*b[t*n+j] as i64;}
  out[i*n+j]=acc;
 }}
 Ok(out)
}

#[cfg(test)]mod tests {
 use super::*;
 #[test]fn owned_cpu_module(){let module={let context=Context::new().unwrap();context.compile("func transform(x:int)->int{return x*3;}").unwrap()};assert_eq!(module.map("transform",&[-2,0,2],Backend::Cpu).unwrap(),vec![-6,0,6]);}
 #[test]fn concurrent_module(){let context=Context::new().unwrap();let module=Arc::new(context.compile("func transform(x:int)->int{return x+1;}").unwrap());let workers:Vec<_>=(0..8).map(|i|{let m=module.clone();std::thread::spawn(move||assert_eq!(m.map("transform",&[i],Backend::Cpu).unwrap(),vec![i+1]))}).collect();for w in workers{w.join().unwrap();}}
 #[test]fn metadata_and_scalar(){let c=Context::new().unwrap();let m=c.compile("func f(x:float)->float{return fma(x,2.0,0.5);}").unwrap();assert!(m.info().unwrap().contains("binary64"));assert_eq!(m.invoke("f",&[Scalar::Float(1.25)],Backend::Cpu).unwrap(),Scalar::Float(3.0));c.clear_cache().unwrap();assert!(c.cache_info().unwrap().contains("generation"));}

 // A = [[1,-1],[0,1]], B = [[1,0],[-1,1]] -> B^T = [[1,-1],[0,1]].
 // C = [[1*1+(-1)(-1), 1*0+(-1)(1)], [0*1+1*(-1), 0*0+1*1]] = [[2,-1],[-1,1]].
 #[test]fn gemm_matches_hand_computed(){
  let a:  [&[i8];2]=[&[1,-1],&[0,1]];
  let bt: [&[i8];2]=[&[1,-1],&[0,1]];
  let (ap,an,bp,bn)=encode_gemm(&a,&bt,2).unwrap();
  let c=gemm_ternary(&ap,&an,&bp,&bn,2,2,2).unwrap();
  assert_eq!(c,vec![2,-1,-1,1]);
 }

 // A value that does not fit the rails must be rejected, not silently wrapped.
 #[test]fn gemm_rejects_bad_inputs(){
  let a:  [&[i8];2]=[&[1,-1],&[0,1]];
  let bt: [&[i8];2]=[&[1,-1],&[0,1]];
  let (ap,an,bp,bn)=encode_gemm(&a,&bt,2).unwrap();
  let short=vec![0u64;1];
  assert!(gemm_ternary(&short,&an,&bp,&bn,2,2,2).is_err());
  assert!(gemm_ternary(&ap,&an,&short,&bn,2,2,2).is_err());
  assert!(gemm_ternary(&ap,&an,&bp,&bn,0,2,2).is_err());
  // ragged rows must be rejected by the encoder
  let ragged:[&[i8];2]=[&[1,-1],&[1]];
  assert!(encode_gemm(&ragged,&bt,2).is_err());
 }

 // K wider than one 64-bit word AND n > 1: the only combination that
 // distinguishes the two B orderings. Regression guard for the layout.
 #[test]fn gemm_multiword_k_and_multi_n(){
  let k=70usize;
  let a1:Vec<i8>=(0..k).map(|i| if i%3==0 {1} else if i%3==1 {-1} else {0}).collect();
  let a2:Vec<i8>=(0..k).map(|i| if i%2==0 {-1} else {1}).collect();
  let b1:Vec<i8>=(0..k).map(|i| if i%5==0 {1} else {0}).collect();
  let b2:Vec<i8>=(0..k).map(|i| if i%4==0 {-1} else {1}).collect();
  let a:[&[i8];2]=[&a1[..],&a2[..]];
  let bt:[&[i8];2]=[&b1[..],&b2[..]];
  let (ap,an,bp,bn)=encode_gemm(&a,&bt,k).unwrap();
  let c=gemm_ternary(&ap,&an,&bp,&bn,2,2,k as u64).unwrap();
  for i in 0..2 { for j in 0..2 {
   let expect:i64=(0..k).map(|t| a[i][t] as i64 * bt[j][t] as i64).sum();
   assert_eq!(c[i*2+j],expect,"C[{}][{}]",i,j);
  }}
 }

 // Larger random case against a naive O(n^3) reference.
 #[test]fn gemm_matches_naive_reference(){
  let (m,n,k)=(5usize,4usize,100usize);
  let mut seed=12345u64;
  let mut next=||{seed=seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
   ((seed>>33)%3) as i8 - 1};
  let a:Vec<Vec<i8>>=(0..m).map(|_| (0..k).map(|_| next()).collect()).collect();
  let bt:Vec<Vec<i8>>=(0..n).map(|_| (0..k).map(|_| next()).collect()).collect();
  let aref:Vec<&[i8]>=a.iter().map(|r| &r[..]).collect();
  let bref:Vec<&[i8]>=bt.iter().map(|r| &r[..]).collect();
  let (ap,an,bp,bn)=encode_gemm(&aref,&bref,k).unwrap();
  let c=gemm_ternary(&ap,&an,&bp,&bn,m as u64,n as u64,k as u64).unwrap();
  for i in 0..m { for j in 0..n {
   let expect:i64=(0..k).map(|t| a[i][t] as i64 * bt[j][t] as i64).sum();
   assert_eq!(c[i*n+j],expect,"C[{}][{}]",i,j);
  }}
 }

 // pack_matrix_a / pack_matrix_b must agree with encode_gemm, and must be
 // correct at k > 64 -- the regime where B's word-plane-major ordering stops
 // coinciding with row-major. B is given ROW-MAJOR here (k rows of n), which
 // is the shape a caller actually holds; encode_gemm wants B^T.
 #[test]fn pack_helpers_match_encode_gemm(){
  for &k in &[1usize,7,63,64,65,70,128,130,200] {
   let (m,n)=(3usize,4usize);
   let mut seed=k as u64*7919+13;
   let mut next=||{seed=seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    ((seed>>33)%3) as i8 - 1};
   let a:Vec<Vec<i8>>=(0..m).map(|_| (0..k).map(|_| next()).collect()).collect();
   // b is B row-major: k rows of n.
   let b:Vec<Vec<i8>>=(0..k).map(|_| (0..n).map(|_| next()).collect()).collect();
   let aref:Vec<&[i8]>=a.iter().map(|r| &r[..]).collect();
   let bref:Vec<&[i8]>=b.iter().map(|r| &r[..]).collect();

   let (ap,an)=pack_matrix_a(&aref).unwrap();
   let (bp,bn)=pack_matrix_b(&bref).unwrap();

   // Cross-check against encode_gemm, which needs B^T.
   let btransposed:Vec<Vec<i8>>=(0..n).map(|j| (0..k).map(|t| b[t][j]).collect()).collect();
   let btref:Vec<&[i8]>=btransposed.iter().map(|r| &r[..]).collect();
   let (eap,ean,ebp,ebn)=encode_gemm(&aref,&btref,k).unwrap();
   assert_eq!(ap,eap,"A positive rail at k={}",k);
   assert_eq!(an,ean,"A negative rail at k={}",k);
   assert_eq!(bp,ebp,"B positive rail at k={}",k);
   assert_eq!(bn,ebn,"B negative rail at k={}",k);

   // And against a naive O(n^3) reference over the row-major B.
   let c=gemm_ternary(&ap,&an,&bp,&bn,m as u64,n as u64,k as u64).unwrap();
   for i in 0..m { for j in 0..n {
    let expect:i64=(0..k).map(|t| a[i][t] as i64 * b[t][j] as i64).sum();
    assert_eq!(c[i*n+j],expect,"C[{}][{}] at k={}",i,j,k);
   }}
  }
 }

 #[test]fn pack_helpers_reject_ragged_input(){
  let a:Vec<Vec<i8>>=vec![vec![1,-1],vec![1]];
  let aref:Vec<&[i8]>=a.iter().map(|r| &r[..]).collect();
  assert!(pack_matrix_a(&aref).is_err());
  let b:Vec<Vec<i8>>=vec![vec![1,0],vec![1]];
  let bref:Vec<&[i8]>=b.iter().map(|r| &r[..]).collect();
  assert!(pack_matrix_b(&bref).is_err());
  assert!(pack_matrix_a(&[]).is_err());
  assert!(pack_matrix_b(&[]).is_err());
 }

 #[test]
 fn parity_is_ternary_and_matmul_agree_with_reference(){
  // The whole point of the routing predicate: the ternary GEMM and the
  // reference kernel must agree EXACTLY, because both are exact integer
  // computations. Any discrepancy is a bug in one of them.
  let (m, k, n) = (3usize, 4usize, 2usize);
  let a:[i8;12]=[1,0,-1,1, 0,1,1,-1, -1,-1,0,1];
  let b:[i8;8] =[1,-1, 0,1, 1,0, -1,1];
  let fast=matmul(&a,&b,m,k,n,false).expect("routed matmul");
  let refd=matmul(&a,&b,m,k,n,true).expect("reference matmul");
  assert_eq!(fast,refd,"ternary GEMM and reference must agree exactly");
 }
 #[test]
 fn parity_is_ternary(){
  assert!(is_ternary(&[-1,0,1,1,-1]));
  assert!(is_ternary(&[]),"empty is vacuously ternary");
  assert!(!is_ternary(&[2,0,1]),"2 is not a trit");
  assert!(!is_ternary(&[-2]),"-2 is not a trit");
 }
 #[test]
 fn parity_matmul_rejects_shape_mismatch(){
  let a:[i8;6]=[1,0,-1,1,0,1];
  let b:[i8;8]=[1,0,-1,1,0,1,0,1];
  // a is 2x3, b claims k=3 n=... which does not fit its length
  assert!(matmul(&a,&b,2,3,3,false).is_err(),"shape mismatch must be refused");
 }
}
