#ifndef TRIUM_SDK_NATIVE_RUNTIME_HPP
#define TRIUM_SDK_NATIVE_RUNTIME_HPP
#include "Ternary/IR/Program.hpp"
#include <memory>
#include <string>
#include <cstdint>
namespace bt::sdk {
// Native-v1 deliberately excludes loops/calls/IO. Static admission bounds avoid
// removing the interpreter's termination/resource controls in native execution.
size_t native_work(const ir::Function& function);
std::string native_compiler(); // trusted absolute candidate; empty if absent
class NativeProgram {
    struct State;
    std::unique_ptr<State> state;
public:
    NativeProgram(const ir::Function&,const std::string& compiler);
    ~NativeProgram();
    NativeProgram(const NativeProgram&)=delete;
    NativeProgram& operator=(const NativeProgram&)=delete;
    uint64_t invoke(const uint64_t* arguments) const;
};
}
#endif
