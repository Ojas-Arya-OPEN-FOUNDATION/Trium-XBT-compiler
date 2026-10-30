#ifndef TRIUM_VULKAN_RUNTIME_HPP
#define TRIUM_VULKAN_RUNTIME_HPP
#include <cstdint>
#include <string>
#include <stdexcept>
#include <vector>
#include <memory>
#include "Ternary/SDK/sdk.h"
namespace bt::sdk {
struct Unavailable:std::runtime_error{using std::runtime_error::runtime_error;};
struct DeviceError:std::runtime_error{using std::runtime_error::runtime_error;};
class VulkanProgram {
    struct State;std::unique_ptr<State> state;
public:
    VulkanProgram(const std::string&,const std::string&,unsigned,unsigned,unsigned);
    ~VulkanProgram();
    VulkanProgram(const VulkanProgram&)=delete;
    VulkanProgram& operator=(const VulkanProgram&)=delete;
    double compilation_ms()const;
    std::vector<int64_t> execute(const std::vector<int64_t>&,bt_sdk_metrics&);
};
std::string devices();
std::vector<int64_t> vulkan_map(const std::string& shader,const std::string& native_shader,const std::vector<int64_t>& input,unsigned group,unsigned device,unsigned representation,bt_sdk_metrics& metrics);
}
#endif
