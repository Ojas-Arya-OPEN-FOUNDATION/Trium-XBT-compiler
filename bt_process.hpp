#ifndef BT_PROCESS_HPP
#define BT_PROCESS_HPP

// Host tooling boundary. No argument is interpreted by a command shell.
// POSIX hosts are supported; other hosts must supply an equivalent process API.
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <spawn.h>
#include <dirent.h>
#include <fcntl.h>
#include <stdexcept>
#include <string>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>
#include <chrono>
#include <thread>
#include <signal.h>
#include <climits>

extern char **environ;
namespace bt::host {
struct ProcessTimeout:std::runtime_error{using std::runtime_error::runtime_error;};

// strerror_r has two incompatible signatures (GNU returns char*, XSI returns int).
inline std::string safe_strerror(int error) {
    char buffer[256] = {};
#if defined(__GLIBC__) && defined(_GNU_SOURCE)
    return std::string(strerror_r(error, buffer, sizeof buffer)); // GNU variant
#else
    const int result = strerror_r(error, buffer, sizeof buffer);  // XSI variant
    return result == 0 ? std::string(buffer) : std::string("unknown error");
#endif
}

// File descriptors that the child must not inherit. posix_spawn does not close
// unrelated descriptors by itself, so a spawned compiler used to inherit every
// readable/writable descriptor the parent had open.
inline std::vector<int> inherited_descriptors() {
    std::vector<int> descriptors;
#if defined(__linux__)
    DIR* directory = opendir("/proc/self/fd");
    if (!directory) return descriptors;
    const int directory_fd = dirfd(directory);
    while (dirent* entry = readdir(directory)) {
        char* end = nullptr;
        const long value = std::strtol(entry->d_name, &end, 10);
        if (end && *end == '\0' && value > STDERR_FILENO && value != directory_fd && value <= INT_MAX)
            descriptors.push_back(static_cast<int>(value));
    }
    closedir(directory);
#endif
    return descriptors;
}

// Toolchain flags shared by the compiler driver and the SDK native JIT, so host
// cross-compilation and native JIT compilation resolve the same system SDK.
inline std::vector<std::string> platform_toolchain_flags() {
    std::vector<std::string> flags;
#if defined(__APPLE__)
    const char* sdk = std::getenv("SDKROOT");
    if (sdk && *sdk) {
        flags.push_back("-isysroot");
        flags.push_back(sdk);
    } else if (std::filesystem::exists("/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk")) {
        flags.push_back("-isysroot");
        flags.push_back("/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk");
    }
#endif
    return flags;
}
inline int run_process(const std::vector<std::string>& args,const std::string& capture="",unsigned timeout_ms=0) {
    if (args.empty() || args.front().empty()) throw std::invalid_argument("empty process command");
    std::vector<char*> argv;
    for (const auto& a : args) {
        if (a.find('\0') != std::string::npos) throw std::invalid_argument("NUL in process argument");
        argv.push_back(const_cast<char*>(a.c_str()));
    }
    argv.push_back(nullptr);
    pid_t pid;
    posix_spawn_file_actions_t actions;posix_spawn_file_actions_t* selected=nullptr;
    {
        if(!capture.empty() && capture.find('\0')!=std::string::npos)
            throw std::invalid_argument("NUL in capture path");
        int error=posix_spawn_file_actions_init(&actions);
        if(error)throw std::runtime_error("cannot initialize tool process actions");
        selected=&actions;
        // Close every unrelated inherited descriptor in the child.
        for(int descriptor:inherited_descriptors()){
            if(descriptor==STDOUT_FILENO||descriptor==STDERR_FILENO)continue;
            if((error=posix_spawn_file_actions_addclose(&actions,descriptor)))break;
        }
        if(!error && !capture.empty()){
            error=posix_spawn_file_actions_addopen(&actions,1,capture.c_str(),O_CREAT|O_WRONLY|O_TRUNC,0600);
            if(!error)error=posix_spawn_file_actions_adddup2(&actions,1,2);
        }
        if(error){posix_spawn_file_actions_destroy(&actions);throw std::runtime_error("cannot configure tool process descriptors");}
    }
    posix_spawnattr_t attributes;posix_spawnattr_t* attr=nullptr;
    {
        short flags=0;
#ifdef POSIX_SPAWN_CLOEXEC_DEFAULT
        flags|=POSIX_SPAWN_CLOEXEC_DEFAULT;
#endif
        if(timeout_ms)flags|=POSIX_SPAWN_SETPGROUP;
        if(flags){
            int error=posix_spawnattr_init(&attributes);
            if(!error){attr=&attributes;error=posix_spawnattr_setflags(attr,flags);}
            if(!error && (flags&POSIX_SPAWN_SETPGROUP))error=posix_spawnattr_setpgroup(attr,0);
            if(error){if(attr)posix_spawnattr_destroy(attr);if(selected)posix_spawn_file_actions_destroy(&actions);throw std::runtime_error("cannot configure tool process attributes");}
        }
    }
    int err = posix_spawnp(&pid, argv[0], selected, attr, argv.data(), environ);
    if(attr)posix_spawnattr_destroy(attr);
    if(selected)posix_spawn_file_actions_destroy(&actions);
    if (err) throw std::runtime_error("cannot execute '" + args[0] + "': " + safe_strerror(err));
    int status;
    const auto deadline=std::chrono::steady_clock::now()+std::chrono::milliseconds(timeout_ms);
    for(;;){
        auto result=waitpid(pid,&status,timeout_ms?WNOHANG:0);
        if(result==pid)break;
        if(result<0){if(errno==EINTR)continue;throw std::runtime_error("waitpid failed");}
        if(std::chrono::steady_clock::now()>=deadline){
            kill(-pid,SIGKILL);while(waitpid(pid,&status,0)<0 && errno==EINTR){}
            throw ProcessTimeout("external compiler/tool deadline exceeded");
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    if (WIFEXITED(status)) return WEXITSTATUS(status);
    return WIFSIGNALED(status) ? 128 + WTERMSIG(status) : 1;
}

class TemporaryDirectory {
    std::filesystem::path path_;
public:
    TemporaryDirectory() {
        auto base = std::filesystem::temp_directory_path() / "bt-private-XXXXXX";
        std::string pattern = base.string();
        char* made = mkdtemp(pattern.data()); // atomic, mode 0700
        if (!made) throw std::runtime_error("cannot create private temporary directory");
        path_ = made;
    }
    ~TemporaryDirectory() { std::error_code ec; std::filesystem::remove_all(path_, ec); }
    TemporaryDirectory(const TemporaryDirectory&) = delete;
    TemporaryDirectory& operator=(const TemporaryDirectory&) = delete;
    std::string file(const std::string& name) const { return (path_ / name).string(); }
};
inline std::vector<std::string> clang_command() {
    std::vector<std::string> command = {"clang", "-O3", "-Wno-override-module"};
    for (auto& flag : platform_toolchain_flags()) command.push_back(flag);
    return command;
}
} // namespace bt::host
#endif
