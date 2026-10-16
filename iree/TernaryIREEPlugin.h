/**
 * @file TernaryIREEPlugin.h
 * @brief IREE Compiler Plugin Interface for the Balanced Ternary (bt) Dialect
 * @author Antigravity Prime Research & Engineering Engine
 * 
 * Bridges 'bt' dialect operations into IREE Flow and Stream dialects,
 * enabling direct dispatch compilation to Vulkan (SPIR-V), Metal (MSL),
 * CUDA (PTX), and CPU backends.
 */

#ifndef TERNARY_IREE_PLUGIN_H
#define TERNARY_IREE_PLUGIN_H

#include "../include/Ternary/TernaryDialect.h"
#include <string>
#include <vector>
#include <memory>
#include <functional>

namespace mlir::iree_compiler {

enum class TargetBackendKind {
    VulkanSPIRV,
    MetalMSL,
    CUDAPTX,
    CPULLVM
};

struct TargetBackendConfig {
    TargetBackendKind kind;
    std::string targetName;
    std::string deviceArch;
    unsigned simdWidth;
    bool supportsBitplanePopcount;
};

class TargetBackend {
public:
    virtual ~TargetBackend() = default;
    virtual TargetBackendKind getKind() const = 0;
    virtual std::string getName() const = 0;
    virtual std::string emitShaderKernel(const std::string& opName, size_t M, size_t N, size_t K) const = 0;
};

class VulkanSPIRVBackend : public TargetBackend {
public:
    TargetBackendKind getKind() const override { return TargetBackendKind::VulkanSPIRV; }
    std::string getName() const override { return "vulkan-spirv"; }
    std::string emitShaderKernel(const std::string& opName, size_t M, size_t N, size_t K) const override;
};

class MetalMSLBackend : public TargetBackend {
public:
    TargetBackendKind getKind() const override { return TargetBackendKind::MetalMSL; }
    std::string getName() const override { return "metal-msl"; }
    std::string emitShaderKernel(const std::string& opName, size_t M, size_t N, size_t K) const override;
};

class CUDAPTXBackend : public TargetBackend {
public:
    TargetBackendKind getKind() const override { return TargetBackendKind::CUDAPTX; }
    std::string getName() const override { return "cuda-nvptx"; }
    std::string emitShaderKernel(const std::string& opName, size_t M, size_t N, size_t K) const override;
};

class CPULLVMBackend : public TargetBackend {
public:
    TargetBackendKind getKind() const override { return TargetBackendKind::CPULLVM; }
    std::string getName() const override { return "llvm-cpu"; }
    std::string emitShaderKernel(const std::string& opName, size_t M, size_t N, size_t K) const override;
};

/**
 * @class TernaryIREEPlugin
 * @brief Main plugin class implementing IREE Compiler Plugin interface.
 */
class TernaryIREEPlugin {
public:
    TernaryIREEPlugin();

    static constexpr const char* kPluginId = "ternary_iree";
    static constexpr const char* kVersion = "1.0.0";
    static constexpr const char* kDescription = "IREE Compiler Plugin for Balanced Ternary Dialect (bt)";

    const char* getPluginId() const { return kPluginId; }
    const char* getVersion() const { return kVersion; }
    const char* getDescription() const { return kDescription; }

    void registerDialects(::mlir::DialectRegistry& registry);
    void registerPasses();

    std::vector<std::string> getSupportedTargetBackends() const;
    std::shared_ptr<TargetBackend> getBackend(TargetBackendKind kind) const;

    // Flow & Stream pipeline bridging
    std::string buildFlowDispatchModule(const std::string& kernelName, size_t M, size_t N, size_t K) const;
    std::string buildStreamDispatchModule(const std::string& kernelName, size_t M, size_t N, size_t K) const;

private:
    std::vector<std::shared_ptr<TargetBackend>> backends;
};

// C-compatible IREE plugin registration entrypoint
extern "C" {
    int iree_compiler_register_plugin_ternary();
}

} // namespace mlir::iree_compiler

#endif // TERNARY_IREE_PLUGIN_H
