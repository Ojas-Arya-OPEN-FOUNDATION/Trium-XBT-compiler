#include "bt_process.hpp"
#include "Ternary/SDK/SDK.hpp"
#include <cassert>
#include <fstream>
#include <iostream>
#include <sys/stat.h>
int main(){
 bt::host::TemporaryDirectory directory;auto log=directory.file("tool.log");
 assert(bt::host::run_process({"/bin/echo","literal;$(not-a-shell)"},log,1000)==0);
 std::ifstream stream(log);std::string line;std::getline(stream,line);assert(line=="literal;$(not-a-shell)");
 struct stat metadata{};assert(!stat(log.c_str(),&metadata) && (metadata.st_mode&0777)==0600);
 auto start=std::chrono::steady_clock::now();bool timeout=false;
 try{bt::host::run_process({"/bin/sleep","5"},directory.file("timeout.log"),30);}catch(const bt::host::ProcessTimeout&){timeout=true;}
 assert(timeout && std::chrono::steady_clock::now()-start<std::chrono::seconds(2));
 auto retained=[](){bt::sdk::Context c;return c.compile("func f(x:int)->int{return x+1;}");}();
 assert(std::get<int64_t>(retained.invoke("f",{int64_t(1)}))==2);
 auto moved=std::move(retained);retained.close();assert(moved.map("f",{1})==std::vector<int64_t>{2});
 if(bt::sdk::info().find("\"native_compiler_candidate\":\"\"")==std::string::npos){
   assert(moved.map("f",{1},BT_SDK_NATIVE)==std::vector<int64_t>{2});
   bt::sdk::Context c;auto real=c.compile("func g(x:float)->float{return fma(x,2.0,0.5);}");
   assert(std::get<double>(real.invoke("g",{1.25},BT_SDK_NATIVE))==3.0);
 }
 bt::sdk::Context context;auto module=context.compile("func f(x:int)->int{return x+1;}");
 assert(module.info().find("int39")!=std::string::npos);context.clear_cache();context.close();
 bool closed=false;try{module.map("f",{1});}catch(const bt::sdk::Error& e){closed=e.code==BT_SDK_CLOSED;}assert(closed);
 bt::sdk::Context first;auto second=std::move(first);first.close();
 std::cout<<"PASS C++ retained/moved/closed ownership and bounded private shell-free compiler process\n";
}
