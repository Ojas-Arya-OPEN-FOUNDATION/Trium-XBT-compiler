#!/usr/bin/env python3
"""
Compile the declarative target descriptions into C++.

    targets/*.yaml  --(validate against schema/target.schema.json)-->
        include/Ternary/Target/GeneratedTargets.hpp

Design decisions (settled with the user):

  * YAML is the authoring format. It has comments, anchors and multi-line
    strings, all of which matter when a vendor is writing a machine
    description by hand.
  * A JSON Schema is the contract. It is checked before generation, so a typo
    is a build failure with a precise path, not a silently-wrong target.
  * TableGen is deliberately NOT used here. It would add an `mlir-tblgen`
    dependency to the non-MLIR standalone build -- which is the common case in
    this repository. TableGen stays where it already is and works well:
    describing MLIR dialect operations.

  * The generated C++ IS COMMITTED. The build must not require Python. A CI
    job regenerates and asserts the diff is empty, which is what actually
    keeps the committed copy honest.

Usage:
    tools/gen_targets.py                       # generate
    tools/gen_targets.py --check               # regenerate, fail if different
    tools/gen_targets.py --print <name>        # dump one target's C++ summary
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import re
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    sys.exit("gen_targets: PyYAML is required (pip install pyyaml)")

try:
    from jsonschema import Draft202012Validator
except ImportError:
    sys.exit("gen_targets: jsonschema is required (pip install jsonschema)")


ROOT = Path(__file__).resolve().parent.parent
TARGETS_DIR = ROOT / "targets"
SCHEMA_PATH = ROOT / "schema" / "target.schema.json"
OUT_HEADER = ROOT / "include" / "Ternary" / "Target" / "GeneratedTargets.hpp"

LAYOUT_ENUM = {
    "dual-rail": "Layout::DualRail",
    "base-243": "Layout::Base243",
    "tcsr": "Layout::TCSR",
    "btfp": "Layout::BTFP",
}

FAMILY_ENUM = {
    "cpu": "Family::Cpu",
    "gpu": "Family::Gpu",
    "accelerator": "Family::Accelerator",
    "simulator": "Family::Simulator",
}


# ---------------------------------------------------------------------------
# Defaults (must mirror the schema's `default` values)
# ---------------------------------------------------------------------------

def apply_defaults(t: dict) -> dict:
    d = t.get("datapath") or {}
    s = t.get("storage") or {}
    r = t.get("registers") or {}
    m = t.get("memory") or {}
    return {
        "name": t["name"],
        "triple": t.get("triple", ""),
        "aliases": t.get("aliases", []),
        "family": t.get("family", "accelerator"),
        "datapath": {
            "native_trit_width": d.get("native_trit_width", 64),
            "max_immediate_trits": d.get("max_immediate_trits", 5),
            "has_word_add": d.get("has_word_add", True),
            "has_lane_mul": d.get("has_lane_mul", True),
            "has_scalar_mul": d.get("has_scalar_mul", False),
            "has_popcount_dot": d.get("has_popcount_dot", True),
        },
        "storage": {
            "native_layouts": s.get("native_layouts", ["dual-rail"]),
            "preferred_layout": s.get("preferred_layout", "dual-rail"),
        },
        "registers": {
            "ternary": r.get("ternary", 32),
            "scalar": r.get("scalar", 32),
            "zero_hardwired": r.get("zero_hardwired", True),
        },
        "native_ops": t.get("native_ops", []),
        "memory": {
            "cache_line_bytes": m.get("cache_line_bytes", 64),
            "scratchpad_bytes": m.get("scratchpad_bytes", 0),
            "global_bytes": m.get("global_bytes", 0),
            "explicit_movement": m.get("explicit_movement", False),
        },
        "costs": t.get("costs", {}),
        "default_cost": t.get("default_cost", 1),
        "simulator": t.get("simulator", None),
    }


def load_targets() -> list[dict]:
    """Load, validate and default every target file. Exits on any problem."""
    if not SCHEMA_PATH.exists():
        sys.exit(f"gen_targets: schema not found at {SCHEMA_PATH}")
    schema = json.loads(SCHEMA_PATH.read_text())
    validator = Draft202012Validator(schema)

    files = sorted(TARGETS_DIR.glob("*.yaml"))
    if not files:
        sys.exit(f"gen_targets: no target files in {TARGETS_DIR}")

    targets, errors, seen = [], [], {}
    for path in files:
        raw = yaml.safe_load(path.read_text())
        if raw is None:
            errors.append(f"{path.name}: empty file")
            continue

        for err in sorted(validator.iter_errors(raw), key=lambda e: list(e.path)):
            where = "/".join(str(p) for p in err.path) or "<root>"
            errors.append(f"{path.name}: {where}: {err.message}")

        if errors:
            continue

        t = apply_defaults(raw)

        # Cross-file constraints the schema cannot express alone.
        if t["name"] in seen:
            errors.append(f"{path.name}: duplicate target name "
                          f"{t['name']!r} (also in {seen[t['name']]})")
        seen[t["name"]] = path.name

        if t["storage"]["preferred_layout"] not in t["storage"]["native_layouts"]:
            errors.append(
                f"{path.name}: preferred_layout "
                f"{t['storage']['preferred_layout']!r} is not in native_layouts")

        for alias in t["aliases"]:
            if alias in seen:
                errors.append(f"{path.name}: alias {alias!r} collides with "
                              f"the target or alias in {seen[alias]}")
            seen[alias] = path.name

        targets.append((path.name, t))

    if errors:
        print("gen_targets: target descriptions are invalid:", file=sys.stderr)
        for e in errors:
            print(f"  {e}", file=sys.stderr)
        sys.exit(1)

    targets.sort(key=lambda kv: kv[1]["name"])
    return [t for _, t in targets]


# ---------------------------------------------------------------------------
# C++ emission
# ---------------------------------------------------------------------------

def cxx_str(s: str) -> str:
    out = s.replace("\\", "\\\\").replace('"', '\\"')
    return '"' + out + '"'


def cxx_list(items) -> str:
    return "{" + ", ".join(items) + "}" if items else "{}"


def emit(targets: list[dict], schema_hash: str) -> str:
    L = []
    w = L.append

    w("// ==============================================================")
    w("//  GeneratedTargets.hpp -- GENERATED FILE, DO NOT EDIT.")
    w("//")
    w("//  Source of truth:  targets/*.yaml")
    w("//  Contract:         schema/target.schema.json")
    w("//  Regenerate with:  tools/gen_targets.py")
    w("//")
    w(f"//  Generated {datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}"
      f"  schema sha256:{schema_hash[:12]}")
    w("//  This file is committed so the build never requires Python. A CI")
    w("//  job regenerates and fails if the result differs.")
    w("// ==============================================================")
    w("")
    w("#ifndef TERNARY_TARGET_GENERATED_TARGETS_HPP")
    w("#define TERNARY_TARGET_GENERATED_TARGETS_HPP")
    w("")
    w('#include "TargetDescription.hpp"')
    w("")
    w("#include <cstddef>")
    w("#include <cstdint>")
    w("#include <string>")
    w("#include <vector>")
    w("")
    w("namespace bt::target {")
    w("")
    w("/** Broad machine class. Determines which backend picks the target up. */")
    w("enum class Family { Cpu, Gpu, Accelerator, Simulator };")
    w("")
    w("inline const char* family_name(Family f) {")
    w("    switch (f) {")
    for k, v in FAMILY_ENUM.items():
        w(f'    case {v}: return "{k}";')
    w("    }")
    w('    return "unknown";')
    w("}")
    w("")
    w("/**")
    w(" * @brief A hardware target, exactly as described in its YAML file.")
    w(" *")
    w(" * Superset of TargetDescription: adds the memory hierarchy, the cost")
    w(" * model and the simulator hook that Track B and P3 need. Convert to the")
    w(" * legacy view with to_target_description().")
    w(" */")
    w("struct MachineDescription {")
    w("    // identity")
    w("    const char* name;")
    w("    const char* triple;")
    w("    const char* const* aliases;  size_t alias_count;")
    w("    Family family;")
    w("")
    w("    // ternary datapath")
    w("    unsigned native_trit_width;")
    w("    unsigned max_immediate_trits;")
    w("    bool has_word_add, has_lane_mul, has_scalar_mul, has_popcount_dot;")
    w("")
    w("    // storage")
    w("    const Layout* native_layouts;  size_t layout_count;")
    w("    Layout preferred_layout;")
    w("")
    w("    // registers")
    w("    unsigned ternary_registers, scalar_registers;")
    w("    bool zero_register_hardwired;")
    w("")
    w("    // operation coverage (empty == assume everything)")
    w("    const char* const* native_ops;  size_t op_count;")
    w("")
    w("    // memory hierarchy (Track A implicit vs Track B explicit)")
    w("    uint64_t cache_line_bytes, scratchpad_bytes, global_bytes;")
    w("    bool explicit_movement;")
    w("")
    w("    // cost model for the instruction selector")
    w("    const char* const* cost_keys;  const unsigned* cost_values;")
    w("    size_t cost_count;")
    w("    unsigned default_cost;")
    w("")
    w("    // Track B simulator hook; nullptr == real hardware")
    w("    const char* simulator_command;")
    w("    const char* const* simulator_args;  size_t simulator_arg_count;")
    w("")
    w("    bool has_op(const std::string& op) const {")
    w("        if (op_count == 0) return true;   // empty == assume everything")
    w("        for (size_t i = 0; i < op_count; ++i)")
    w("            if (op == native_ops[i]) return true;")
    w("        return false;")
    w("    }")
    w("")
    w("    unsigned cost_of(const std::string& op) const {")
    w("        for (size_t i = 0; i < cost_count; ++i)")
    w("            if (op == cost_keys[i]) return cost_values[i];")
    w("        return default_cost;")
    w("    }")
    w("")
    w("    bool supports_layout(Layout l) const {")
    w("        for (size_t i = 0; i < layout_count; ++i)")
    w("            if (native_layouts[i] == l) return true;")
    w("        return false;")
    w("    }")
    w("")
    w("    bool has_simulator() const { return simulator_command != nullptr; }")
    w("};")
    w("")

    # --- per-target constant tables -------------------------------------
    for t in targets:
        sym = t["name"].replace("-", "_").replace(".", "_")
        al = t["aliases"]
        if al:
            w(f"static const char* const kAliases_{sym}[] = "
              + cxx_list(cxx_str(a) for a in al) + ";")
        lays = t["storage"]["native_layouts"]
        w(f"static const Layout kLayouts_{sym}[] = "
          + cxx_list(LAYOUT_ENUM[x] for x in lays) + ";")
        ops = t["native_ops"]
        if ops:
            w(f"static const char* const kOps_{sym}[] = "
              + cxx_list(cxx_str(o) for o in ops) + ";")
        costs = sorted(t["costs"].items())
        if costs:
            w(f"static const char* const kCostKeys_{sym}[] = "
              + cxx_list(cxx_str(k) for k, _ in costs) + ";")
            w(f"static const unsigned kCostVals_{sym}[] = "
              + cxx_list(str(int(v)) for _, v in costs) + ";")
        sim = t["simulator"]
        if sim:
            args = sim.get("args", [])
            w(f'static const char* const kSimCmd_{sym}[] = '
              + '{' + cxx_str(sim["command"]) + '};')
            if args:
                w(f"static const char* const kSimArgs_{sym}[] = "
                  + cxx_list(cxx_str(a) for a in args) + ";")
        w("")

    # --- the table -------------------------------------------------------
    w("/** Every target compiled into the compiler, sorted by name. */")
    w("inline const MachineDescription* machines(size_t* count_out = nullptr) {")
    w(f"    static const MachineDescription table[] = {{")
    for t in targets:
        sym = t["name"].replace("-", "_").replace(".", "_")
        d, s, r, m = t["datapath"], t["storage"], t["registers"], t["memory"]
        al, ops, costs, sim = t["aliases"], t["native_ops"], sorted(
            t["costs"].items()), t["simulator"]
        w("        {")
        w(f"            {cxx_str(t['name'])}, {cxx_str(t['triple'])},")
        w(f"            {(('kAliases_' + sym) if al else 'nullptr')}, "
          f"{len(al)},")
        w(f"            {FAMILY_ENUM[t['family']]},")
        w(f"            {d['native_trit_width']}, "
          f"{d['max_immediate_trits']},")
        w(f"            {'true' if d['has_word_add'] else 'false'}, "
          f"{'true' if d['has_lane_mul'] else 'false'}, "
          f"{'true' if d['has_scalar_mul'] else 'false'}, "
          f"{'true' if d['has_popcount_dot'] else 'false'},")
        w(f"            kLayouts_{sym}, {len(s['native_layouts'])}, "
          f"{LAYOUT_ENUM[s['preferred_layout']]},")
        w(f"            {r['ternary']}, {r['scalar']}, "
          f"{'true' if r['zero_hardwired'] else 'false'},")
        w(f"            {(('kOps_' + sym) if ops else 'nullptr')}, "
          f"{len(ops)},")
        w(f"            {m['cache_line_bytes']}ULL, "
          f"{m['scratchpad_bytes']}ULL, {m['global_bytes']}ULL, "
          f"{'true' if m['explicit_movement'] else 'false'},")
        w(f"            {(('kCostKeys_' + sym) if costs else 'nullptr')}, "
          f"{(('kCostVals_' + sym) if costs else 'nullptr')}, "
          f"{len(costs)}, {t['default_cost']},")
        if sim:
            args = sim.get("args", [])
            w(f"            kSimCmd_{sym}[0], "
              f"{(('kSimArgs_' + sym) if args else 'nullptr')}, {len(args)},")
        else:
            w("            nullptr, nullptr, 0,")
        w("        },")
    w("    };")
    w("    if (count_out) *count_out = sizeof(table) / sizeof(table[0]);")
    w("    return table;")
    w("}")
    w("")

    w("/** Number of compiled-in targets. */")
    w("inline size_t machine_count() { size_t n = 0; machines(&n); return n; }")
    w("")

    w("/** Look up a target by canonical name or alias; nullptr if unknown. */")
    w("inline const MachineDescription* find_machine(const std::string& name) {")
    w("    size_t n = 0; const MachineDescription* t = machines(&n);")
    w("    for (size_t i = 0; i < n; ++i) {")
    w("        if (name == t[i].name) return &t[i];")
    w("        for (size_t j = 0; j < t[i].alias_count; ++j)")
    w("            if (name == t[i].aliases[j]) return &t[i];")
    w("    }")
    w("    return nullptr;")
    w("}")
    w("")

    w("/** All names a target may be selected by. */")
    w("inline std::vector<std::string> machine_names() {")
    w("    std::vector<std::string> out;")
    w("    size_t n = 0; const MachineDescription* t = machines(&n);")
    w("    for (size_t i = 0; i < n; ++i) {")
    w("        out.push_back(t[i].name);")
    w("        for (size_t j = 0; j < t[i].alias_count; ++j)")
    w("            out.push_back(t[i].aliases[j]);")
    w("    }")
    w("    return out;")
    w("}")
    w("")

    w("/**")
    w(" * Project onto the legacy TargetDescription. Fields the older struct")
    w(" * does not model (memory, costs, simulator) are dropped, so this is")
    w(" * lossy -- prefer MachineDescription in new code.")
    w(" */")
    w("inline TargetDescription to_target_description(const MachineDescription& m) {")
    w("    TargetDescription t;")
    w("    t.name = m.name;")
    w("    t.triple = m.triple ? m.triple : \"\";")
    w("    for (size_t i = 0; i < m.alias_count; ++i)")
    w("        t.aliases.push_back(m.aliases[i]);")
    w("    t.native_trit_width = m.native_trit_width;")
    w("    t.max_immediate_trits = m.max_immediate_trits;")
    w("    t.has_word_add = m.has_word_add;")
    w("    t.has_lane_mul = m.has_lane_mul;")
    w("    t.has_scalar_mul = m.has_scalar_mul;")
    w("    t.has_popcount_dot = m.has_popcount_dot;")
    w("    for (size_t i = 0; i < m.layout_count; ++i)")
    w("        t.native_layouts.insert(m.native_layouts[i]);")
    w("    t.preferred_layout = m.preferred_layout;")
    w("    t.ternary_registers = m.ternary_registers;")
    w("    t.scalar_registers = m.scalar_registers;")
    w("    t.zero_register_hardwired = m.zero_register_hardwired;")
    w("    for (size_t i = 0; i < m.op_count; ++i)")
    w("        t.native_ops.insert(m.native_ops[i]);")
    w("    return t;")
    w("}")
    w("")
    w("} // namespace bt::target")
    w("")
    w("#endif // TERNARY_TARGET_GENERATED_TARGETS_HPP")
    return "\n".join(L) + "\n"


def strip_volatile(text: str) -> str:
    """Remove the parts of the output that legitimately change every run.

    The generation timestamp is provenance, not content. Without stripping it,
    `--check` would report 'stale' on every single run and the CI job that is
    supposed to keep the committed copy honest would be worthless. The schema
    hash is deliberately NOT stripped: a schema change must count as stale.
    """
    return re.sub(r"(//  Generated )\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ",
                  r"\1<date>", text)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true",
                    help="fail if the committed output differs from a fresh run")
    ap.add_argument("--print", metavar="NAME",
                    help="print a human-readable summary of one target")
    args = ap.parse_args()

    targets = load_targets()
    schema_hash = hashlib.sha256(SCHEMA_PATH.read_bytes()).hexdigest()

    if args.print:
        t = next((x for x in targets
                  if x["name"] == args.print or args.print in x["aliases"]), None)
        if not t:
            print(f"gen_targets: no target named {args.print!r}", file=sys.stderr)
            return 1
        print(json.dumps(t, indent=2, sort_keys=True))
        return 0

    text = emit(targets, schema_hash)

    if args.check:
        if not OUT_HEADER.exists():
            print(f"gen_targets: {OUT_HEADER.relative_to(ROOT)} is missing; "
                  "run tools/gen_targets.py", file=sys.stderr)
            return 1
        if strip_volatile(OUT_HEADER.read_text()) != strip_volatile(text):
            print(f"gen_targets: {OUT_HEADER.relative_to(ROOT)} is STALE.",
                  file=sys.stderr)
            print("  targets/*.yaml changed but the generated C++ did not.",
                  file=sys.stderr)
            print("  Fix: run tools/gen_targets.py and commit the result.",
                  file=sys.stderr)
            return 1
        print(f"gen_targets: {OUT_HEADER.relative_to(ROOT)} is up to date "
              f"({len(targets)} targets)")
        return 0

    OUT_HEADER.parent.mkdir(parents=True, exist_ok=True)
    OUT_HEADER.write_text(text)
    print(f"gen_targets: wrote {OUT_HEADER.relative_to(ROOT)} "
          f"({len(targets)} targets, {text.count(chr(10))} lines)")
    for t in targets:
        print(f"  - {t['name']:<16} {t['family']:<12} "
              f"{t['datapath']['native_trit_width']:>4} trits"
              f"{'  [simulated]' if t['simulator'] else ''}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
