#include "VulkanRuntime.hpp"
#include <chrono>
#include <algorithm>
#include <cstring>
#include <fstream>
#include <filesystem>
#include <cstdlib>
#include <mutex>
#include "bt_process.hpp"
#if defined(BT_SDK_HAS_VULKAN)
#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>
#include <dlfcn.h>
#include <array>
#include <bit>
#include <sstream>
namespace bt::sdk {
namespace {
using Clock=std::chrono::steady_clock;
double ms(Clock::time_point a){return std::chrono::duration<double,std::milli>(Clock::now()-a).count();}
void check(VkResult r,const char* op){if(r!=VK_SUCCESS)throw DeviceError(std::string(op)+" failed: "+std::to_string(r));}
#define FUNCTIONS(X) X(DestroyInstance) X(EnumeratePhysicalDevices) X(GetPhysicalDeviceProperties) X(GetPhysicalDeviceProperties2) X(GetPhysicalDeviceFeatures) X(GetPhysicalDeviceMemoryProperties) X(GetPhysicalDeviceQueueFamilyProperties) X(CreateDevice) X(DestroyDevice) X(GetDeviceQueue) X(CreateBuffer) X(DestroyBuffer) X(GetBufferMemoryRequirements) X(AllocateMemory) X(FreeMemory) X(BindBufferMemory) X(MapMemory) X(UnmapMemory) X(FlushMappedMemoryRanges) X(InvalidateMappedMemoryRanges) X(CreateShaderModule) X(DestroyShaderModule) X(CreateDescriptorSetLayout) X(DestroyDescriptorSetLayout) X(CreatePipelineLayout) X(DestroyPipelineLayout) X(CreateComputePipelines) X(DestroyPipeline) X(CreateDescriptorPool) X(DestroyDescriptorPool) X(AllocateDescriptorSets) X(UpdateDescriptorSets) X(CreateCommandPool) X(DestroyCommandPool) X(AllocateCommandBuffers) X(BeginCommandBuffer) X(EndCommandBuffer) X(CmdBindPipeline) X(CmdBindDescriptorSets) X(CmdPushConstants) X(CmdDispatch) X(CmdCopyBuffer) X(CmdFillBuffer) X(CmdPipelineBarrier) X(CreateFence) X(DestroyFence) X(QueueSubmit) X(WaitForFences) X(DeviceWaitIdle) X(ResetCommandPool) X(ResetFences) X(CreateQueryPool) X(DestroyQueryPool) X(CmdResetQueryPool) X(CmdWriteTimestamp) X(GetQueryPoolResults)
struct API {
    void* library=nullptr;VkInstance instance=VK_NULL_HANDLE;PFN_vkGetInstanceProcAddr get=nullptr;
#define DECL(N) PFN_vk##N N=nullptr;
    FUNCTIONS(DECL)
#undef DECL
    API(){
        const char* path=std::getenv("BT_SDK_VULKAN_LIBRARY");library=dlopen(path?path:"libvulkan.so.1",RTLD_NOW|RTLD_LOCAL);
        if(!library)throw Unavailable("Vulkan loader unavailable; set BT_SDK_VULKAN_LIBRARY to a trusted loader/ICD");
        try {
            get=reinterpret_cast<PFN_vkGetInstanceProcAddr>(dlsym(library,"vkGetInstanceProcAddr"));if(!get)throw Unavailable("missing vkGetInstanceProcAddr");
            auto create=reinterpret_cast<PFN_vkCreateInstance>(get(VK_NULL_HANDLE,"vkCreateInstance"));if(!create)throw Unavailable("missing vkCreateInstance");
            VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};app.pApplicationName="Trium SDK";app.apiVersion=VK_API_VERSION_1_1;
            VkInstanceCreateInfo info{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};info.pApplicationInfo=&app;
            // Portability enumeration is enabled only when advertised.
            auto extensions=reinterpret_cast<PFN_vkEnumerateInstanceExtensionProperties>(get(VK_NULL_HANDLE,"vkEnumerateInstanceExtensionProperties"));
            const char* portability="VK_KHR_portability_enumeration";
            if(extensions){uint32_t count=0;check(extensions(nullptr,&count,nullptr),"instance extension count");std::vector<VkExtensionProperties> ext(count);check(extensions(nullptr,&count,ext.data()),"instance extensions");for(auto& x:ext)if(!std::strcmp(x.extensionName,portability)){info.flags|=VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;info.enabledExtensionCount=1;info.ppEnabledExtensionNames=&portability;}}
            auto r=create(&info,nullptr,&instance);if(r!=VK_SUCCESS)throw Unavailable("cannot create Vulkan 1.1 instance: "+std::to_string(r));
#define LOAD(N) N=reinterpret_cast<PFN_vk##N>(get(instance,"vk" #N));if(!N)throw Unavailable("missing Vulkan command " #N);
            FUNCTIONS(LOAD)
#undef LOAD
        } catch(...){if(instance){auto destroy=reinterpret_cast<PFN_vkDestroyInstance>(get(instance,"vkDestroyInstance"));if(destroy)destroy(instance,nullptr);}dlclose(library);throw;}
    }
    ~API(){if(instance)DestroyInstance(instance,nullptr);if(library)dlclose(library);}
    std::vector<VkPhysicalDevice> physical(){uint32_t n=0;check(EnumeratePhysicalDevices(instance,&n,nullptr),"device count");std::vector<VkPhysicalDevice> p(n);if(n)check(EnumeratePhysicalDevices(instance,&n,p.data()),"devices");p.resize(n);return p;}
};
struct Buffer {VkBuffer buffer=VK_NULL_HANDLE;VkDeviceMemory memory=VK_NULL_HANDLE;void* mapped=nullptr;bool coherent=false;};
struct Session {
    API api;VkDevice device=VK_NULL_HANDLE;VkPhysicalDevice physical=VK_NULL_HANDLE;VkQueue queue;uint32_t family=0,timestamp_bits=0;
    VkPhysicalDeviceProperties properties{};VkPhysicalDeviceMemoryProperties memory{};
    std::array<Buffer,6> buffers{};VkShaderModule shader=VK_NULL_HANDLE;VkDescriptorSetLayout layout=VK_NULL_HANDLE;VkPipelineLayout pipeline_layout=VK_NULL_HANDLE;VkPipeline pipeline=VK_NULL_HANDLE;VkDescriptorPool descriptors=VK_NULL_HANDLE;VkCommandPool commands=VK_NULL_HANDLE;VkFence fence=VK_NULL_HANDLE;VkQueryPool queries=VK_NULL_HANDLE;
    bool submitted=false,completed=false,native64=false;
    explicit Session(unsigned index,unsigned requested){auto devices=api.physical();if(index>=devices.size())throw Unavailable("requested Vulkan device does not exist");physical=devices[index];api.GetPhysicalDeviceProperties(physical,&properties);api.GetPhysicalDeviceMemoryProperties(physical,&memory);
        if(properties.apiVersion<VK_API_VERSION_1_1)throw Unavailable("device lacks Vulkan 1.1");
        uint32_t count=0;api.GetPhysicalDeviceQueueFamilyProperties(physical,&count,nullptr);std::vector<VkQueueFamilyProperties> qs(count);api.GetPhysicalDeviceQueueFamilyProperties(physical,&count,qs.data());family=count;
        for(uint32_t i=0;i<count;++i)if(qs[i].queueCount && (qs[i].queueFlags&VK_QUEUE_COMPUTE_BIT)){family=i;timestamp_bits=qs[i].timestampValidBits;break;}
        if(family==count)throw Unavailable("no Vulkan compute queue");
        float priority=1;VkDeviceQueueCreateInfo q{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};q.queueFamilyIndex=family;q.queueCount=1;q.pQueuePriorities=&priority;
        VkDeviceCreateInfo d{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};d.queueCreateInfoCount=1;d.pQueueCreateInfos=&q;
        VkPhysicalDeviceFeatures available{},enabled{};api.GetPhysicalDeviceFeatures(physical,&available);
        if(requested==1 && !available.shaderInt64)throw Unavailable("device does not support shaderInt64");
        native64=requested==1 || (requested==2 && available.shaderInt64);enabled.shaderInt64=native64;d.pEnabledFeatures=&enabled;
        auto enumext=reinterpret_cast<PFN_vkEnumerateDeviceExtensionProperties>(api.get(api.instance,"vkEnumerateDeviceExtensionProperties"));const char* subset="VK_KHR_portability_subset";
        if(enumext){uint32_t size=0;check(enumext(physical,nullptr,&size,nullptr),"device extensions count");std::vector<VkExtensionProperties> ext(size);check(enumext(physical,nullptr,&size,ext.data()),"device extensions");for(auto& e:ext)if(!std::strcmp(e.extensionName,subset)){d.enabledExtensionCount=1;d.ppEnabledExtensionNames=&subset;}}
        check(api.CreateDevice(physical,&d,nullptr,&device),"create device");api.GetDeviceQueue(device,family,0,&queue);
    }
    void clear_dispatch()noexcept{
        if(submitted&&!completed)api.DeviceWaitIdle(device);
        if(queries)api.DestroyQueryPool(device,queries,nullptr);if(fence)api.DestroyFence(device,fence,nullptr);if(commands)api.DestroyCommandPool(device,commands,nullptr);if(descriptors)api.DestroyDescriptorPool(device,descriptors,nullptr);
        for(auto& b:buffers){if(b.mapped)api.UnmapMemory(device,b.memory);if(b.buffer)api.DestroyBuffer(device,b.buffer,nullptr);if(b.memory)api.FreeMemory(device,b.memory,nullptr);}
        queries=VK_NULL_HANDLE;fence=VK_NULL_HANDLE;commands=VK_NULL_HANDLE;descriptors=VK_NULL_HANDLE;buffers={};submitted=completed=false;
    }
    ~Session(){if(!device)return;clear_dispatch();
        if(pipeline)api.DestroyPipeline(device,pipeline,nullptr);if(pipeline_layout)api.DestroyPipelineLayout(device,pipeline_layout,nullptr);if(layout)api.DestroyDescriptorSetLayout(device,layout,nullptr);if(shader)api.DestroyShaderModule(device,shader,nullptr);
        api.DestroyDevice(device,nullptr);
    }
    void allocate(unsigned index,VkDeviceSize size,VkBufferUsageFlags usage,bool host){auto& b=buffers[index];VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};info.size=size;info.usage=usage;info.sharingMode=VK_SHARING_MODE_EXCLUSIVE;check(api.CreateBuffer(device,&info,nullptr,&b.buffer),"create buffer");VkMemoryRequirements req;api.GetBufferMemoryRequirements(device,b.buffer,&req);
        int selected=-1;for(uint32_t i=0;i<memory.memoryTypeCount;++i){auto flags=memory.memoryTypes[i].propertyFlags;if((req.memoryTypeBits&(1u<<i)) && (!host || (flags&VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT))){if(selected<0)selected=i;if(flags&(host?VK_MEMORY_PROPERTY_HOST_COHERENT_BIT:VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)){selected=i;break;}}}
        if(selected<0)throw Unavailable("no compatible Vulkan memory type");b.coherent=memory.memoryTypes[selected].propertyFlags&VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        VkMemoryAllocateInfo alloc{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};alloc.allocationSize=req.size;alloc.memoryTypeIndex=selected;check(api.AllocateMemory(device,&alloc,nullptr,&b.memory),"allocate memory");check(api.BindBufferMemory(device,b.buffer,b.memory,0),"bind memory");if(host)check(api.MapMemory(device,b.memory,0,VK_WHOLE_SIZE,0,&b.mapped),"map memory");
    }
    void visibility(unsigned index,bool flush){auto& b=buffers[index];if(b.coherent)return;VkMappedMemoryRange range{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};range.memory=b.memory;range.size=VK_WHOLE_SIZE;check(flush?api.FlushMappedMemoryRanges(device,1,&range):api.InvalidateMappedMemoryRanges(device,1,&range),"mapped memory visibility");}
};
VkPhysicalDeviceIDProperties identity(API& api,VkPhysicalDevice physical){VkPhysicalDeviceIDProperties id{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES};VkPhysicalDeviceProperties2 p{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};p.pNext=&id;api.GetPhysicalDeviceProperties2(physical,&p);return id;}
std::string uuid(const uint8_t* data){const char* hex="0123456789abcdef";std::string result;for(unsigned i=0;i<16;++i){result+=hex[data[i]>>4];result+=hex[data[i]&15];}return result;}
std::string escaped(const char* p){std::string s;for(;*p;++p){unsigned char c=*p;if(c=='"'||c=='\\')s+='\\';if(c>=32)s+=c;else s+='?';}return s;}
std::vector<uint32_t> compile(const std::string& source){host::TemporaryDirectory dir;auto in=dir.file("kernel.comp"),out=dir.file("kernel.spv");{std::ofstream f(in);f<<source;if(!f)throw DeviceError("cannot write shader source");}
    const char* glslang=std::getenv("BT_GLSLANG");const char* validator=std::getenv("BT_SPIRV_VAL");
    auto run=[&](const std::vector<std::string>& args){auto log=dir.file("diagnostic.log");int status;
        try{status=host::run_process(args,log,60000);}catch(const host::ProcessTimeout&){throw;}catch(const std::exception& e){throw Unavailable(e.what());}
        if(status){std::ifstream f(log);std::string detail(16384,'\0');f.read(detail.data(),detail.size());detail.resize(f.gcount());throw DeviceError("shader tool failure: "+detail);}};
    run({glslang?glslang:"glslangValidator","-V","--target-env","vulkan1.1","-o",out,in});
    run({validator?validator:"spirv-val","--target-env","vulkan1.1",out});
    std::ifstream f(out,std::ios::binary|std::ios::ate);auto size=f.tellg();if(size<20 || size>16*1024*1024 || size%4)throw DeviceError("invalid SPIR-V size");std::vector<uint32_t> words(size_t(size)/4);f.seekg(0);f.read(reinterpret_cast<char*>(words.data()),size);if(!f || words[0]!=0x07230203)throw DeviceError("invalid SPIR-V");return words;
}
}
std::string devices(){API api;auto list=api.physical();std::string text="[";for(size_t i=0;i<list.size();++i){VkPhysicalDeviceProperties p;api.GetPhysicalDeviceProperties(list[i],&p);VkPhysicalDeviceFeatures features{};api.GetPhysicalDeviceFeatures(list[i],&features);auto id=identity(api,list[i]);if(i)text+=",";text+="{\"index\":"+std::to_string(i)+",\"name\":\""+escaped(p.deviceName)+"\",\"vendor_id\":"+std::to_string(p.vendorID)+",\"device_uuid\":\""+uuid(id.deviceUUID)+"\""+",\"device_id\":"+std::to_string(p.deviceID)+",\"driver_version\":"+std::to_string(p.driverVersion)+",\"api_version\":"+std::to_string(p.apiVersion)+",\"shader_int64\":"+(features.shaderInt64?"true":"false")+",\"device_type\":"+std::to_string(p.deviceType)+",\"software\":"+(p.deviceType==VK_PHYSICAL_DEVICE_TYPE_CPU?"true":"false")+"}";}return text+"]";}
struct VulkanProgram::State {
    std::mutex mutex;Session session;unsigned group;size_t capacity=0;
    VkDescriptorSet set=VK_NULL_HANDLE;VkCommandBuffer cmd=VK_NULL_HANDLE;
    bool poisoned=false;double compile_ms=0;
    State(const std::string& source,const std::string& native_source,unsigned g,unsigned index,unsigned requested):session(index,requested),group(g){
        auto& s=session;auto begin=Clock::now();auto code=compile(s.native64?native_source:source);compile_ms=ms(begin);
    VkShaderModuleCreateInfo shader{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};shader.codeSize=code.size()*4;shader.pCode=code.data();check(s.api.CreateShaderModule(s.device,&shader,nullptr,&s.shader),"shader module");
    std::array<VkDescriptorSetLayoutBinding,3> bindings{};for(unsigned i=0;i<3;++i){bindings[i].binding=i;bindings[i].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;bindings[i].descriptorCount=1;bindings[i].stageFlags=VK_SHADER_STAGE_COMPUTE_BIT;}
    VkDescriptorSetLayoutCreateInfo layout{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};layout.bindingCount=3;layout.pBindings=bindings.data();check(s.api.CreateDescriptorSetLayout(s.device,&layout,nullptr,&s.layout),"descriptor layout");
    VkPushConstantRange push{VK_SHADER_STAGE_COMPUTE_BIT,0,4};VkPipelineLayoutCreateInfo pi{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};pi.setLayoutCount=1;pi.pSetLayouts=&s.layout;pi.pushConstantRangeCount=1;pi.pPushConstantRanges=&push;check(s.api.CreatePipelineLayout(s.device,&pi,nullptr,&s.pipeline_layout),"pipeline layout");
    VkComputePipelineCreateInfo pipeline{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};pipeline.stage.sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;pipeline.stage.stage=VK_SHADER_STAGE_COMPUTE_BIT;pipeline.stage.module=s.shader;pipeline.stage.pName="main";pipeline.layout=s.pipeline_layout;check(s.api.CreateComputePipelines(s.device,VK_NULL_HANDLE,1,&pipeline,nullptr,&s.pipeline),"compute pipeline");
    }
    void reserve(size_t requested){
        if(requested<=capacity)return;
        session.clear_dispatch();capacity=0;auto& s=session;
        try{allocate(requested);capacity=requested;}catch(...){s.clear_dispatch();throw;}
    }
    void allocate(size_t capacity){auto& s=session;
    VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};check(s.api.CreateFence(s.device,&fi,nullptr,&s.fence),"fence");
    auto size=capacity*8;for(unsigned i=0;i<3;++i){auto bytes=i==2?size/2:size;s.allocate(i,bytes,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_SRC_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT,false);s.allocate(i+3,bytes,VK_BUFFER_USAGE_TRANSFER_SRC_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT,true);}
    VkDescriptorPoolSize poolsize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,3};VkDescriptorPoolCreateInfo pool{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};pool.maxSets=1;pool.poolSizeCount=1;pool.pPoolSizes=&poolsize;check(s.api.CreateDescriptorPool(s.device,&pool,nullptr,&s.descriptors),"descriptor pool");
    VkDescriptorSetAllocateInfo allocate{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};allocate.descriptorPool=s.descriptors;allocate.descriptorSetCount=1;allocate.pSetLayouts=&s.layout;check(s.api.AllocateDescriptorSets(s.device,&allocate,&set),"descriptor sets");
    std::array<VkDescriptorBufferInfo,3> infos{};std::array<VkWriteDescriptorSet,3>writes{};for(unsigned i=0;i<3;++i){infos[i]={s.buffers[i].buffer,0,i==2?size/2:size};writes[i].sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;writes[i].dstSet=set;writes[i].dstBinding=i;writes[i].descriptorCount=1;writes[i].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;writes[i].pBufferInfo=&infos[i];}s.api.UpdateDescriptorSets(s.device,3,writes.data(),0,nullptr);
    VkCommandPoolCreateInfo cp{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};cp.queueFamilyIndex=s.family;check(s.api.CreateCommandPool(s.device,&cp,nullptr,&s.commands),"command pool");VkCommandBufferAllocateInfo ca{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};ca.commandPool=s.commands;ca.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY;ca.commandBufferCount=1;check(s.api.AllocateCommandBuffers(s.device,&ca,&cmd),"command buffer");
    if(s.timestamp_bits){VkQueryPoolCreateInfo q{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};q.queryType=VK_QUERY_TYPE_TIMESTAMP;q.queryCount=4;check(s.api.CreateQueryPool(s.device,&q,nullptr,&s.queries),"timestamp pool");}
    }
    std::vector<int64_t> run(const std::vector<int64_t>& input,bt_sdk_metrics& times){
        static_assert(std::endian::native==std::endian::little,"Vulkan word ABI currently qualifies little-endian hosts");
        auto& s=session;auto id=identity(s.api,s.physical);std::copy(id.deviceUUID,id.deviceUUID+16,times.device_uuid);times.representation=s.native64?1:0;
        if(input.empty())return {};
        if(group>s.properties.limits.maxComputeWorkGroupInvocations || group>s.properties.limits.maxComputeWorkGroupSize[0] || (input.size()+group-1)/group>s.properties.limits.maxComputeWorkGroupCount[0] || input.size()*8>s.properties.limits.maxStorageBufferRange)throw Unavailable("kernel exceeds device limits");
        reserve(input.size());check(s.api.ResetCommandPool(s.device,s.commands,0),"reset command pool");check(s.api.ResetFences(s.device,1,&s.fence),"reset fence");s.submitted=s.completed=false;
        auto size=input.size()*8;auto begin=Clock::now();std::memcpy(s.buffers[3].mapped,input.data(),size);s.visibility(3,true);times.upload_ms=ms(begin);
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};check(s.api.BeginCommandBuffer(cmd,&bi),"begin commands");if(s.queries){s.api.CmdResetQueryPool(cmd,s.queries,0,4);s.api.CmdWriteTimestamp(cmd,VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,s.queries,0);}
    VkBufferCopy region{0,0,size};s.api.CmdCopyBuffer(cmd,s.buffers[3].buffer,s.buffers[0].buffer,1,&region);s.api.CmdFillBuffer(cmd,s.buffers[1].buffer,0,size,0x5a5a5a5a);s.api.CmdFillBuffer(cmd,s.buffers[2].buffer,0,size/2,0);
    if(s.queries)s.api.CmdWriteTimestamp(cmd,VK_PIPELINE_STAGE_TRANSFER_BIT,s.queries,1);
    VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};barrier.srcAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT;barrier.dstAccessMask=VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT;s.api.CmdPipelineBarrier(cmd,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,0,1,&barrier,0,nullptr,0,nullptr);
    s.api.CmdBindPipeline(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,s.pipeline);s.api.CmdBindDescriptorSets(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,s.pipeline_layout,0,1,&set,0,nullptr);uint32_t count=input.size();s.api.CmdPushConstants(cmd,s.pipeline_layout,VK_SHADER_STAGE_COMPUTE_BIT,0,4,&count);s.api.CmdDispatch(cmd,(count+group-1)/group,1,1);
    if(s.queries)s.api.CmdWriteTimestamp(cmd,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,s.queries,2);
    barrier.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT;barrier.dstAccessMask=VK_ACCESS_TRANSFER_READ_BIT;s.api.CmdPipelineBarrier(cmd,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT,0,1,&barrier,0,nullptr,0,nullptr);
    s.api.CmdCopyBuffer(cmd,s.buffers[1].buffer,s.buffers[4].buffer,1,&region);region.size=size/2;s.api.CmdCopyBuffer(cmd,s.buffers[2].buffer,s.buffers[5].buffer,1,&region);
    if(s.queries)s.api.CmdWriteTimestamp(cmd,VK_PIPELINE_STAGE_TRANSFER_BIT,s.queries,3);
    barrier.srcAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT;barrier.dstAccessMask=VK_ACCESS_HOST_READ_BIT;s.api.CmdPipelineBarrier(cmd,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_HOST_BIT,0,1,&barrier,0,nullptr,0,nullptr);check(s.api.EndCommandBuffer(cmd),"end commands");
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};submit.commandBufferCount=1;submit.pCommandBuffers=&cmd;
    begin=Clock::now();check(s.api.QueueSubmit(s.queue,1,&submit,s.fence),"queue submit");s.submitted=true;check(s.api.WaitForFences(s.device,1,&s.fence,VK_TRUE,UINT64_MAX),"fence wait");s.completed=true;times.execute_ms=ms(begin);
    if(s.queries){uint64_t ticks[4];check(s.api.GetQueryPoolResults(s.device,s.queries,0,4,sizeof(ticks),ticks,8,VK_QUERY_RESULT_64_BIT),"timestamp results");uint64_t mask=s.timestamp_bits==64?UINT64_MAX:(uint64_t(1)<<s.timestamp_bits)-1;auto elapsed=[&](unsigned a,unsigned b){return double((ticks[b]-ticks[a])&mask)*s.properties.limits.timestampPeriod/1e6;};times.gpu_upload_ms=elapsed(0,1);times.kernel_ms=elapsed(1,2);times.gpu_download_ms=elapsed(2,3);}
    begin=Clock::now();s.visibility(5,false);auto status=static_cast<const uint32_t*>(s.buffers[5].mapped);for(size_t i=0;i<input.size();++i)if(status[i])throw std::overflow_error("Vulkan lane arithmetic failure at index "+std::to_string(i));s.visibility(4,false);std::vector<int64_t> result(input.size());std::memcpy(result.data(),s.buffers[4].mapped,size);times.download_ms=ms(begin);return result;
}
};
VulkanProgram::VulkanProgram(const std::string& source,const std::string& native_source,unsigned group,unsigned index,unsigned requested):state(std::make_unique<State>(source,native_source,group,index,requested)){}
VulkanProgram::~VulkanProgram()=default;
double VulkanProgram::compilation_ms()const{return state->compile_ms;}
std::vector<int64_t> VulkanProgram::execute(const std::vector<int64_t>& input,bt_sdk_metrics& times){
    std::lock_guard lock(state->mutex);if(state->poisoned)throw DeviceError("cached Vulkan device faulted; clear cache before retrying");
    try{return state->run(input,times);}catch(const DeviceError&){state->poisoned=true;throw;}catch(...){if(state->session.submitted&&!state->session.completed)state->poisoned=true;throw;}
}
std::vector<int64_t> vulkan_map(const std::string& source,const std::string& native_source,const std::vector<int64_t>& input,unsigned group,unsigned index,unsigned requested,bt_sdk_metrics& times){
    VulkanProgram program(source,native_source,group,index,requested);times.compile_ms=program.compilation_ms();return program.execute(input,times);
}

} // namespace bt::sdk
#else
namespace bt::sdk {
struct VulkanProgram::State {};
VulkanProgram::VulkanProgram(const std::string&,const std::string&,unsigned,unsigned,unsigned){throw Unavailable("SDK built without Vulkan headers/backend");}
VulkanProgram::~VulkanProgram()=default;
double VulkanProgram::compilation_ms()const{return 0;}
std::vector<int64_t> VulkanProgram::execute(const std::vector<int64_t>&,bt_sdk_metrics&){throw Unavailable("SDK built without Vulkan headers/backend");}
std::string devices(){throw Unavailable("SDK built without Vulkan headers/backend");}
std::vector<int64_t> vulkan_map(const std::string&,const std::string&,const std::vector<int64_t>&,unsigned,unsigned,unsigned,bt_sdk_metrics&){throw Unavailable("SDK built without Vulkan headers/backend");}
}
#endif
