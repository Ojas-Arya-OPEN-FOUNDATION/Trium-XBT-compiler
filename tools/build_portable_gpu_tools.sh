#!/usr/bin/env bash
# Optional offline shader toolchain. No GPU execution/hardware claim.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
prefix="${1:-$root/build/portable-tools}"
tag=vulkan-sdk-1.4.341.0
mkdir -p "$prefix"
for repo in Vulkan-Headers SPIRV-Headers SPIRV-Tools glslang; do
 if [ ! -d "$prefix/$repo/.git" ]; then git clone --depth 1 --branch "$tag" "https://github.com/KhronosGroup/$repo.git" "$prefix/$repo"; fi
 expected="$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["repositories"][sys.argv[2]])' "$root/requirements/portable_gpu_tools.json" "$repo")"
 test "$(git -C "$prefix/$repo" rev-parse HEAD)" = "$expected"
done
cmake -S "$prefix/glslang" -B "$prefix/glslang-build" -DCMAKE_BUILD_TYPE=Release -DENABLE_OPT=OFF -DENABLE_GLSLANG_BINARIES=ON -DBUILD_TESTING=OFF
cmake --build "$prefix/glslang-build" -j2
cmake -S "$prefix/SPIRV-Tools" -B "$prefix/spirv-build" -DCMAKE_BUILD_TYPE=Release -DSPIRV-Headers_SOURCE_DIR="$prefix/SPIRV-Headers" -DSPIRV_SKIP_TESTS=ON
cmake --build "$prefix/spirv-build" --target spirv-val -j2
printf '\nSet BT_GLSLANG=%s/glslang-build/StandAlone/glslang\nSet BT_SPIRV_VAL=%s/spirv-build/tools/spirv-val\n' "$prefix" "$prefix"
