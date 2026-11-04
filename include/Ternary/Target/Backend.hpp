#ifndef TERNARY_TARGET_BACKEND_HPP
#define TERNARY_TARGET_BACKEND_HPP

/**
 * @file include/Ternary/Target/Backend.hpp
 * @brief Backend interface and registry.
 *
 * Before this file, adding a target meant editing a closed enum and roughly
 * nine separate dispatch regions of bt_compiler.cpp, with no compile-time
 * guarantee that every site had been found. There was no interface, no
 * registry and no virtual dispatch anywhere in the tree.
 *
 * Now a backend is a class that registers itself. The driver asks the registry
 * for a backend by name; it never knows the list. A new target is one new
 * .cpp file plus one registration line -- or, with the static registrar below,
 * just the file.
 *
 * The existing hardcoded path in bt_compiler.cpp still runs for the targets it
 * handles, so this is additive: the seam exists and new backends use it
 * without a risky rewrite of working code.
 */

#include "TargetDescription.hpp"
#include "../IR/Program.hpp"

#include <algorithm>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace bt::target {

/**
 * @brief What a backend produces, and whether it succeeded.
 *
 * `unsupported` is populated rather than thrown so a caller can report every
 * construct a backend could not lower in one pass, instead of discovering them
 * one run at a time.
 */
struct BackendResult {
    std::string output;                    // emitted code / assembly / IR
    std::vector<std::string> unsupported;  // constructs that could not be lowered
    bool ok{false};

    bool complete() const { return ok && unsupported.empty(); }
};

/**
 * @brief The contract every backend implements.
 */
class Backend {
public:
    virtual ~Backend() = default;

    /** Canonical backend name, e.g. "xbt". */
    virtual std::string name() const = 0;

    /** What this backend targets. */
    virtual const TargetDescription& target() const = 0;

    /**
     * @brief Names this backend answers to, including aliases.
     * Matched case-insensitively.
     */
    virtual std::vector<std::string> matches() const {
        std::vector<std::string> m{name(), target().name, target().triple};
        for (const auto& a : target().aliases) m.push_back(a);
        return m;
    }

    /**
     * @brief Lower a whole program.
     *
     * Implementations must never emit code they do not believe is correct.
     * A construct that cannot be lowered goes into `unsupported`; the caller
     * decides whether that is fatal.
     */
    virtual BackendResult lower(const ir::Program& program) const = 0;

    /** Short human-readable description for `--list-targets`. */
    virtual std::string summary() const {
        return name() + " (" + target().name + ")";
    }
};

/**
 * @brief Registry of all backends.
 *
 * Backends register statically, so a translation unit that defines one is
 * enough to make it available.
 */
class BackendRegistry {
public:
    static BackendRegistry& instance() {
        static BackendRegistry r;
        return r;
    }

    void register_backend(std::unique_ptr<Backend> b) {
        if (!b) throw std::invalid_argument("register_backend: null backend");
        // Later registration wins, so a target can be overridden in tests.
        for (const auto& m : b->matches()) {
            std::string k = normalise(m);
            if (!k.empty()) index_[k] = b.get();
        }
        backends_.push_back(std::move(b));
    }

    /** Look up by name, triple or alias. Returns nullptr if unknown. */
    Backend* find(const std::string& key) const {
        auto it = index_.find(normalise(key));
        return it == index_.end() ? nullptr : it->second;
    }

    std::vector<Backend*> all() const {
        std::vector<Backend*> v;
        for (const auto& b : backends_) v.push_back(b.get());
        return v;
    }

    std::vector<std::string> names() const {
        std::vector<std::string> v;
        for (const auto& b : backends_) v.push_back(b->name());
        std::sort(v.begin(), v.end());
        return v;
    }

    bool empty() const { return backends_.empty(); }

private:
    BackendRegistry() = default;

    static std::string normalise(const std::string& s) {
        std::string o;
        o.reserve(s.size());
        for (char c : s) o.push_back(static_cast<char>(std::tolower(
            static_cast<unsigned char>(c))));
        while (!o.empty() && (o.back() == ' ' || o.back() == '\t')) o.pop_back();
        return o;
    }

    std::vector<std::unique_ptr<Backend>> backends_;
    std::map<std::string, Backend*> index_;
};

/**
 * @brief Registers a backend at static initialisation time.
 *
 * Usage, in the backend's .cpp file:
 *     static bt::target::StaticBackend<MyBackend> registration;
 */
template <class T>
struct StaticBackend {
    StaticBackend() {
        BackendRegistry::instance().register_backend(std::make_unique<T>());
    }
};

} // namespace bt::target

#endif // TERNARY_TARGET_BACKEND_HPP
