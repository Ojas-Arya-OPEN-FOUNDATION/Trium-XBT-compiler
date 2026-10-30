# License boundaries for SDK development

Status: 2026-10-30. This is a repository/package boundary note, **not a new
license grant or legal opinion**. No license text or existing Cargo license
metadata was changed by the R1/R2 or SDK 1.1 implementation.
The SDK 1.1 native shared objects and emitted runtime helpers do not acquire an
independent permissive license merely because they are dynamically loaded.
Compiler/tool dependencies retain their upstream terms; no third-party source
was integrated or relicensed by this execution milestone.

## Current source and packaging

| Component | Current declaration / boundary |
|---|---|
| Repository | Root `LICENSE` contains GNU Affero General Public License v3 text. No separate permissive grant for the new SDK/math/runtime was introduced. |
| Native compiler, shared SDK and math headers | Developed inside that repository. There is no newly added linking/runtime exception; do not describe `libtrium_sdk` as MIT just because a wrapper has MIT metadata. |
| Rust crate | `ffi/rust/Cargo.toml` declares MIT. That existing declaration and the root AGPL text need maintainer clarification of exact file/component scope before publishing a newly expanded crate. |
| Python binding | New package metadata does not declare an independent permissive license. The tested wheel contains Python wrappers, not the native SDK library. Separately installing/linking the library does not itself settle licensing obligations. |
| Generated artifacts | No new generated-code exception is declared. User-authored program content and compiler-supplied runtime/shader helper content are distinct issues; do not assume every generated helper-containing artifact has a permissive redistribution grant. |
| Optional third-party GPU tools/runtimes | Khronos tools, Vulkan headers/loader, SwiftShader and their dependencies retain their own upstream licenses/notices. Their downloaded sources/builds are not vendored in this patch or silently relicensed under this project's text. |

This does not prevent continued open-source development as approved. It does
mean an SDK publication should have an explicit, consistent answer to what is
licensed under which terms. AGPL obligations matter for distribution and some
network use; applications with proprietary or mixed-license integration need
review against the actual license and linkage/distribution model.

## Before a public SDK distribution

1. Inventory ownership and license declarations per component/file, including
   older bindings. Confirm whether the Rust declaration intentionally grants
   rights independently of the native implementation.
2. Have the appropriate rights holders decide whether to retain the existing
   model, clarify component scopes, or explicitly grant an additional SDK/runtime/
   generated-helper exception. Such changes require authorization, not an agent
   editing a metadata string.
3. Make package metadata, shipped license/notice files and documentation agree.
   Preserve upstream notices for any redistributed dependencies/binaries.
4. Document which native runtime a Python/Rust package loads and which licenses
   apply to the complete delivered combination.

No claim is made here that MIT wrapper metadata overrides the root license, that
all generated programs must share one license, or that ordinary dependency use
changes upstream ownership. Those determinations require the component-specific
review above.
