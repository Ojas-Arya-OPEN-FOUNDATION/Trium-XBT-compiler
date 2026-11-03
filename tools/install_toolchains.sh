#!/usr/bin/env bash
# Install the optional toolchains this project can use but does not require.
#
# Everything here is OPTIONAL - the build and the core tests work without any
# of it. What each one unlocks:
#
#   rustc/cargo  compiles and tests ffi/rust (otherwise only ABI-checked)
#   glslang      validates emitted GLSL and compiles it to SPIR-V
#   clang        enables -S / -c (assembly and object emission)
#   vulkan       compiles lib/SDK/VulkanRuntime.cpp, which is otherwise
#                behind BT_SDK_HAS_VULKAN and compiles nowhere
#
# Everything is fetched from npm or GitHub, so it works in sandboxes that
# block the usual distribution sites (static.rust-lang.org, deb.debian.org).
# No root required; installs under tools/.toolchains by default.
#
# Usage:  ./tools/install_toolchains.sh [install_dir]
#         RUST_VERSION=1.88.0 ./tools/install_toolchains.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${1:-$ROOT/tools/.toolchains}"
RUST_VERSION="${RUST_VERSION:-1.88.0}"
VULKAN_HEADERS_VERSION="${VULKAN_HEADERS_VERSION:-v1.3.283}"

mkdir -p "$DEST/bin" "$DEST/lib" "$DEST/include"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

have() { command -v "$1" >/dev/null 2>&1; }
info() { printf '\033[1m==> %s\033[0m\n' "$*"; }

if ! have npm; then
    echo "npm is required for the rustc/cargo, glslang and clang installs." >&2
    echo "Vulkan headers need curl only; run with npm to get everything." >&2
fi

# ---------------------------------------------------------------------------
# 1. Rust: rustc + cargo + std, assembled into a normal toolchain layout.
# ---------------------------------------------------------------------------
if [ -x "$DEST/bin/rustc" ] && [ -x "$DEST/bin/cargo" ]; then
    info "rust: already installed at $DEST"
else
    info "rust: installing rustc/cargo/$RUST_VERSION from npm"
    mkdir -p "$WORK/rust"
    ( cd "$WORK/rust" && npm install --no-audit --no-fund --silent \
        "@rustbin/rustc-$RUST_VERSION-x86_64-unknown-linux-gnu" \
        "@rustbin/cargo-$RUST_VERSION-x86_64-unknown-linux-gnu" \
        "@rustbin/rust-std-$RUST_VERSION-x86_64-unknown-linux-gnu" )

    NM="$WORK/rust/node_modules"
    R="$NM/@rustbin/rustc-$RUST_VERSION-x86_64-unknown-linux-gnu/rustc"
    C="$NM/@rustbin/cargo-$RUST_VERSION-x86_64-unknown-linux-gnu/cargo"
    S="$NM/@rustbin/rust-std-$RUST_VERSION-x86_64-unknown-linux-gnu/rust-std-x86_64-unknown-linux-gnu"

    mkdir -p "$DEST/lib/rustlib"
    cp -r "$R"/bin/. "$DEST/bin/"
    cp -r "$R"/lib/.  "$DEST/lib/"
    [ -d "$R/libexec" ] && cp -r "$R"/libexec/. "$DEST/libexec/" || true
    cp "$C/bin/cargo" "$DEST/bin/"
    cp -r "$S/lib/rustlib/." "$DEST/lib/rustlib/"
    chmod +x "$DEST/bin"/* 2>/dev/null || true
fi

# ---------------------------------------------------------------------------
# 2. glslangValidator: validates emitted GLSL, compiles it to SPIR-V.
# ---------------------------------------------------------------------------
if [ -x "$DEST/bin/glslangValidator" ]; then
    info "glslang: already installed"
else
    info "glslang: installing prebuilt binary from npm"
    mkdir -p "$WORK/glsl"
    ( cd "$WORK/glsl" && npm install --no-audit --no-fund --silent \
        glslang-validator-prebuilt-predownloaded )
    GV="$WORK/glsl/node_modules/glslang-validator-prebuilt-predownloaded/bin/glslangValidator.linux"
    if [ -f "$GV" ]; then
        cp "$GV" "$DEST/bin/glslangValidator"
        chmod +x "$DEST/bin/glslangValidator"
    else
        echo "warning: no linux glslangValidator binary in that package" >&2
    fi
fi

# ---------------------------------------------------------------------------
# 3. clang: enables -S / -c.
# ---------------------------------------------------------------------------
if [ -x "$DEST/bin/clang" ]; then
    info "clang: already installed"
else
    info "clang: installing prebuilt LLVM/clang from npm"
    mkdir -p "$WORK/llvm"
    ( cd "$WORK/llvm" && npm install --no-audit --no-fund --silent llvm-runtime )
    CL="$(find "$WORK/llvm/node_modules" -type f -path "*llvm-runtime-clang-linux-x64*/bin/clang" 2>/dev/null | head -1)"
    if [ -n "$CL" ]; then
        # Copy the whole toolchain dir so clang finds its builtin headers.
        TD="$(dirname "$(dirname "$CL")")"
        cp -r "$TD"/. "$DEST/llvm-clang/"
        ln -sf "$DEST/llvm-clang/bin/clang"  "$DEST/bin/clang"
        ln -sf "$DEST/llvm-clang/bin/clang++" "$DEST/bin/clang++" 2>/dev/null || true
    else
        echo "warning: no linux clang binary in that package" >&2
    fi
fi

# ---------------------------------------------------------------------------
# 4. Vulkan headers: lets the Vulkan SDK runtime compile.
# ---------------------------------------------------------------------------
if [ -f "$DEST/include/vulkan/vulkan.h" ]; then
    info "vulkan-headers: already installed"
else
    info "vulkan-headers: fetching $VULKAN_HEADERS_VERSION from GitHub"
    curl -sSL -o "$WORK/vkh.tar.gz" \
        "https://codeload.github.com/KhronosGroup/Vulkan-Headers/tar.gz/refs/tags/$VULKAN_HEADERS_VERSION"
    tar xzf "$WORK/vkh.tar.gz" -C "$WORK"
    mkdir -p "$DEST/include"
    cp -r "$WORK"/Vulkan-Headers-*/include/vulkan "$DEST/include/"
fi

# ---------------------------------------------------------------------------
info "installed toolchains"
export PATH="$DEST/bin:$PATH"
export LD_LIBRARY_PATH="$DEST/lib:${LD_LIBRARY_PATH:-}"
printf '  rustc     : %s\n' "$(rustc --version 2>/dev/null || echo MISSING)"
printf '  cargo     : %s\n' "$(cargo --version 2>/dev/null || echo MISSING)"
printf '  glslang   : %s\n' "$("$DEST/bin/glslangValidator" --version 2>/dev/null | head -1 || echo MISSING)"
printf '  clang     : %s\n' "$(clang --version 2>/dev/null | head -1 || echo MISSING)"
printf '  vulkan.h  : %s\n' "$([ -f "$DEST/include/vulkan/vulkan.h" ] && echo present || echo MISSING)"

cat <<EOF

Add these to your environment:

    export PATH="$DEST/bin:\$PATH"
    export LD_LIBRARY_PATH="$DEST/lib:\${LD_LIBRARY_PATH:-}"

And to ./configure or cmake as:

    cmake -B build -DBT_SDK_VULKAN=ON \\
        -DCMAKE_PREFIX_PATH="$DEST" \\
        -DCMAKE_C_COMPILER="$DEST/bin/clang"

To compile the Rust FFI crate:
    cd ffi/rust && LD_LIBRARY_PATH=<build>:\$LD_LIBRARY_PATH cargo test
EOF
