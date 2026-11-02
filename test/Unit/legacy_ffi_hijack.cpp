/* Fixture for the legacy FFI loader regression test.
 *
 * When loaded, this shared object records that its constructor ran. The test
 * copies it into a temporary working directory under the name libbt_runtime.so
 * and asserts that `import bt_ffi` never loads it.
 */
#include <cstdio>
#include <cstdlib>

namespace { __attribute__((constructor)) void bt_hijack_fixture_constructor() {
    const char* marker = std::getenv("BT_HIJACK_MARKER");
    if (!marker) return;
    std::FILE* file = std::fopen(marker, "w");
    if (!file) return;
    std::fputs("loaded\n", file);
    std::fclose(file);
} }
