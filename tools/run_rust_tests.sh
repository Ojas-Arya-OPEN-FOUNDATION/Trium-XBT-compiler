#!/usr/bin/env bash
# Compile and run the Rust FFI crate's tests against the built SDK.
#
# The crate links against libtrium_sdk and libbt_runtime by name, so cargo
# needs both a linker search path (-L) and a runtime path (LD_LIBRARY_PATH).
# This script finds the built libraries and sets both up.
#
# Usage:  tools/run_rust_tests.sh [cargo]
#
# Exit 77 if no cargo is available, so CTest reports SKIP rather than FAIL on
# machines without a Rust toolchain.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CARGO="${1:-cargo}"

if ! command -v "$CARGO" >/dev/null 2>&1; then
    echo "SKIP: no cargo on PATH (run tools/install_toolchains.sh)"
    exit 77
fi

# Locate the built SDK libraries: anywhere under build/ by default.
SEARCH_DIRS=()
while IFS= read -r f; do
    [ -n "$f" ] && SEARCH_DIRS+=("$(dirname "$f")")
done < <(
    find "${BUILD_DIR:-$ROOT/build}" "$ROOT" \( -type f -o -type l \) \
         \( -name 'libtrium_sdk.so' -o -name 'libtrium_sdk.dylib' -o -name 'libtrium_sdk.so.*' \) \
         -not -path '*/.git/*' -not -path '*/target/*' 2>/dev/null)

if [ "${#SEARCH_DIRS[@]}" -eq 0 ]; then
    echo "SKIP: libtrium_sdk not built yet - build the sdk target first"
    exit 77
fi

LIBDIR="${SEARCH_DIRS[0]}"
RUSTLIB=""
for d in "${SEARCH_DIRS[@]}"; do
    if [ -e "$d/libbt_runtime.so" ] || [ -e "$d/libbt_runtime.dylib" ]; then
        RUSTLIB="$d"
        break
    fi
done

if [ -z "$RUSTLIB" ]; then
    echo "SKIP: libbt_runtime not built - the crate links bt_* symbols from it"
    exit 77
fi

# rustc is a wrapper that dlopens librustc_driver from <toolchain>/lib. If the
# caller's environment does not already have that on the library path, cargo
# fails with the very unhelpful "could not execute process `rustc -vV`
# (never executed) / No such file or directory". Derive it from where cargo
# actually lives so this works regardless of how CTest invokes us.
CARGO_BIN_DIR="$(cd "$(dirname "$(command -v "$CARGO")")" && pwd)"
TOOLCHAIN_ROOT="$(dirname "$CARGO_BIN_DIR")"
EXTRA_LIBS=""
for cand in "$TOOLCHAIN_ROOT/lib" "$HOME/.rust/lib"; do
    [ -d "$cand" ] && EXTRA_LIBS="${EXTRA_LIBS:+$EXTRA_LIBS:}$cand"
done

echo "==> rust: $($CARGO --version)"
echo "==> sdk libs: $LIBDIR"
echo "==> runtime libs: $RUSTLIB"
echo "==> rustc driver libs: ${EXTRA_LIBS:-none}"

export LD_LIBRARY_PATH="$LIBDIR:$RUSTLIB${EXTRA_LIBS:+:$EXTRA_LIBS}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export DYLD_LIBRARY_PATH="$LIBDIR:$RUSTLIB${EXTRA_LIBS:+:$EXTRA_LIBS}${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
export BT_LIB_DIR="$LIBDIR"
export BT_RUNTIME_DIR="$RUSTLIB"

cd "$ROOT/ffi/rust"
exec "$CARGO" test --offline
