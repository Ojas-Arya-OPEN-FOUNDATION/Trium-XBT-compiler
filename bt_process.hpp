#ifndef BT_PROCESS_HPP
#define BT_PROCESS_HPP

// Host tooling boundary. No argument is interpreted by a command shell.
// POSIX hosts are supported; other hosts must supply an equivalent process API.
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <spawn.h>
#include <stdexcept>
#include <string>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

extern char **environ;
namespace bt::host {
inline int run_process(const std::vector<std::string>& args) {
    if (args.empty() || args.front().empty()) throw std::invalid_argument("empty process command");
    std::vector<char*> argv;
    for (const auto& a : args) {
        if (a.find('\0') != std::string::npos) throw std::invalid_argument("NUL in process argument");
        argv.push_back(const_cast<char*>(a.c_str()));
    }
    argv.push_back(nullptr);
    pid_t pid;
    int err = posix_spawnp(&pid, argv[0], nullptr, nullptr, argv.data(), environ);
    if (err) throw std::runtime_error("cannot execute '" + args[0] + "': " + std::strerror(err));
    int status;
    while (waitpid(pid, &status, 0) == -1) {
        if (errno != EINTR) throw std::runtime_error("waitpid failed");
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
    std::vector<std::string> cmd = {"clang", "-O3", "-Wno-override-module"};
#if defined(__APPLE__)
    const char* sdk = std::getenv("SDKROOT");
    if (sdk) {
        cmd.push_back("-isysroot");
        cmd.push_back(sdk);
    } else if (std::filesystem::exists("/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk")) {
        cmd.push_back("-isysroot");
        cmd.push_back("/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk");
    }
#endif
    return cmd;
}
} // namespace bt::host
#endif
