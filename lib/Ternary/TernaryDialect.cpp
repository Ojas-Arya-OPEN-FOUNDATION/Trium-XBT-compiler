/**
 * @file TernaryDialect.cpp
 * @brief MLIR Dialect implementation for the Balanced Ternary (bt) Dialect
 * @author Antigravity Prime Research & Engineering Engine
 */

#include "../../include/Ternary/TernaryDialect.h"
#include <iostream>
#include <unordered_set>
#include <cctype>

namespace mlir::bt {

static bool g_ternary_dialect_registered = false;
static std::unordered_set<std::string> g_registered_ops;
static std::unordered_set<std::string> g_registered_types;

void TernaryDialect::initialize() {
    // 1. Register Dialect Types
    g_registered_types.insert("!bt.trit");
    g_registered_types.insert("!bt.dualrail");
    g_registered_types.insert("!bt.csa_pair");
    g_registered_types.insert("!bt.tensor");

    // 2. Register Dialect Operations
    g_registered_ops.insert(PlanarDotOp::kOpName);
    g_registered_ops.insert(MatmulOp::kOpName);
    g_registered_ops.insert("bt.gemm");
    g_registered_ops.insert(BTHAOp::kOpName);
    g_registered_ops.insert("bt.btfa");
    g_registered_ops.insert(Base243UnpackOp::kOpName);
    g_registered_ops.insert("bt.unpack_base243");
    g_registered_ops.insert(CSAReduceOp::kOpName);
    g_registered_ops.insert(CSAAccumOp::kOpName);
    g_registered_ops.insert(Select3Op::kOpName);
    g_registered_ops.insert("bt.pack_base243");

    g_ternary_dialect_registered = true;
}

std::pair<uint64_t, uint64_t> CSAReduceOp::evalCSAReduce(uint64_t sp, uint64_t sn, uint64_t cp, uint64_t cn, int8_t* carry_out) {
    ::bt::TritWord64 curr_S(sp, sn);
    ::bt::TritWord64 curr_C(cp, cn);
    int8_t accumulated_cout = 0;
    size_t iters = 0;
    while ((curr_C.P != 0 || curr_C.N != 0) && iters < 64) {
        // Track MSB carry bit 63 before shifting left
        if ((curr_C.P >> 63) & 1ULL) {
            accumulated_cout += 1;
        } else if ((curr_C.N >> 63) & 1ULL) {
            accumulated_cout -= 1;
        }
        uint64_t sh_P = curr_C.P << 1;
        uint64_t sh_N = curr_C.N << 1;
        auto ha = ::bt::BTHA(curr_S, ::bt::TritWord64(sh_P, sh_N));
        curr_S = ha.Sum;
        curr_C = ha.Carry;
        iters++;
    }
    if (carry_out) {
        *carry_out = accumulated_cout;
    }
    return {curr_S.P, curr_S.N};
}

std::pair<uint64_t, uint64_t> CSAReduceOp::execute(uint64_t sp, uint64_t sn, uint64_t cp, uint64_t cn, int8_t* carry_out) {
    return evalCSAReduce(sp, sn, cp, cn, carry_out);
}

std::pair<uint64_t, uint64_t> evalCSAReduce(uint64_t sp, uint64_t sn, uint64_t cp, uint64_t cn, int8_t* carry_out) {
    return CSAReduceOp::evalCSAReduce(sp, sn, cp, cn, carry_out);
}

std::pair<uint64_t, uint64_t> TernaryDialect::evalCSAReduce(uint64_t sp, uint64_t sn, uint64_t cp, uint64_t cn, int8_t* carry_out) {
    return CSAReduceOp::evalCSAReduce(sp, sn, cp, cn, carry_out);
}

bool TernaryDialect::verifyCSAReduce(uint64_t sp, uint64_t sn, uint64_t cp, uint64_t cn, uint64_t& rp, uint64_t& rn) {
    if (!verifyDualRailDisjointness(sp, sn) || !verifyDualRailDisjointness(cp, cn)) return false;
    auto res = CSAReduceOp::execute(sp, sn, cp, cn);
    rp = res.first;
    rn = res.second;
    return verifyDualRailDisjointness(rp, rn);
}

void registerTernaryDialect(::mlir::DialectRegistry &registry) {
    registry.insert("bt");
    TernaryDialect dialect;
    dialect.initialize();
}

void registerTernaryDialect() {
    TernaryDialect dialect;
    dialect.initialize();
}

bool isTernaryDialectRegistered() {
    return g_ternary_dialect_registered;
}

// ==============================================================================
// Standalone MLIR text loader (no upstream MLIR parser in fallback builds)
// ==============================================================================

namespace {

// Replaces comment text ("//" to end of line) with spaces so that all structural
// scanning operates on comment-free text while original offsets are preserved.
std::string blankOutComments(const std::string& text) {
    std::string out = text;
    bool inString = false;
    for (size_t i = 0; i < out.size(); ++i) {
        char c = out[i];
        if (c == '"') {
            inString = !inString;
        } else if (!inString && c == '/' && i + 1 < out.size() && out[i + 1] == '/') {
            while (i < out.size() && out[i] != '\n') {
                if (out[i] != '\n' && out[i] != '\t') out[i] = ' ';
                ++i;
            }
            --i; // compensate for the loop increment; newline is preserved
        }
    }
    return out;
}

size_t lineOfOffset(const std::string& text, size_t offset) {
    return 1 + static_cast<size_t>(std::count(text.begin(), text.begin() + static_cast<long>(offset), '\n'));
}

void skipWs(const std::string& s, size_t& i) {
    while (i < s.size() && std::isspace(static_cast<unsigned char>(s[i]))) ++i;
}

std::string trim(const std::string& s) {
    size_t b = 0, e = s.size();
    while (b < e && std::isspace(static_cast<unsigned char>(s[b]))) ++b;
    while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1]))) --e;
    return s.substr(b, e - b);
}

bool readIdentifier(const std::string& s, size_t& i, std::string& id) {
    if (i >= s.size() || !(std::isalpha(static_cast<unsigned char>(s[i])) || s[i] == '_')) return false;
    size_t start = i;
    while (i < s.size() && (std::isalnum(static_cast<unsigned char>(s[i])) || s[i] == '_' || s[i] == '$')) ++i;
    id = s.substr(start, i - start);
    return true;
}

// Reads one fully-qualified MLIR type token (e.g. "i64", "!bt.dualrail<64>",
// "tensor<2x3xi64>") from s at i.
std::string readTypeToken(const std::string& s, size_t& i) {
    skipWs(s, i);
    size_t start = i;
    if (i < s.size() && s[i] == '!') {
        ++i; // dialect types: !bt.trit etc.
        std::string id;
        if (!readIdentifier(s, i, id)) return "";
        while (i < s.size() && (s[i] == '.' || std::isalnum(static_cast<unsigned char>(s[i])) || s[i] == '_')) ++i;
        // optional parameter list, e.g. !bt.dualrail<64> / !bt.tensor<2x2x64>
        if (i < s.size() && s[i] == '<') {
            int depth = 1;
            ++i;
            while (i < s.size() && depth > 0) {
                if (s[i] == '<') ++depth;
                else if (s[i] == '>') --depth;
                ++i;
            }
            if (depth != 0) return "";
        }
    } else if (i < s.size() && (std::isalpha(static_cast<unsigned char>(s[i])))) {
        std::string id;
        if (!readIdentifier(s, i, id)) return "";
        if (i < s.size() && s[i] == '<') {
            int depth = 1;
            ++i;
            while (i < s.size() && depth > 0) {
                if (s[i] == '<') ++depth;
                else if (s[i] == '>') --depth;
                ++i;
            }
            if (depth != 0) return "";
        }
    } else {
        return "";
    }
    return s.substr(start, i - start);
}

// Parses a type token into the standalone MLIRType representation. Only types
// whose MLIRType::to_string() round-trips are accepted; anything else is an
// honest parse error rather than silent corruption.
bool parseMLIRTypeToken(const std::string& tok, ::bt::mlir::MLIRType& out) {
    std::string t = trim(tok);
    auto extractAngle = [](const std::string& s, uint64_t& value) -> bool {
        size_t lt = s.find('<');
        if (lt == std::string::npos || s.back() != '>') return false;
        std::string inner = s.substr(lt + 1, s.size() - lt - 2);
        value = std::strtoull(inner.c_str(), nullptr, 10);
        return true;
    };

    if (t == "!bt.trit") { out = ::bt::mlir::MLIRType::getTrit(); return true; }
    if (t == "!bt.dualrail") { out = ::bt::mlir::MLIRType::getDualRail(64); return true; }
    if (t.rfind("!bt.dualrail<", 0) == 0) { uint64_t w; if (!extractAngle(t, w)) return false; out = ::bt::mlir::MLIRType::getDualRail(static_cast<unsigned>(w)); return true; }
    if (t.rfind("!bt.csa_pair<", 0) == 0) { uint64_t w; if (!extractAngle(t, w)) return false; out = ::bt::mlir::MLIRType::getCSAPair(static_cast<unsigned>(w)); return true; }
    if (t.rfind("!bt.tensor<", 0) == 0) {
        size_t lt = t.find('<');
        if (t.back() != '>') return false;
        std::string inner = t.substr(lt + 1, t.size() - lt - 2);
        std::vector<int64_t> dims;
        uint64_t width = 0;
        // Format: <d0xd1x...xW> — every dimension then the trailing bit width.
        size_t pos = 0;
        while (pos < inner.size()) {
            size_t x = inner.find('x', pos);
            std::string num = inner.substr(pos, (x == std::string::npos ? inner.size() : x) - pos);
            uint64_t v = std::strtoull(num.c_str(), nullptr, 10);
            if (x == std::string::npos) { width = v; break; }
            dims.push_back(static_cast<int64_t>(v));
            pos = x + 1;
        }
        out = ::bt::mlir::MLIRType::getTernaryTensor(dims, static_cast<unsigned>(width));
        return true;
    }
    if (t == "index") { out = ::bt::mlir::MLIRType::getIndex(); return true; }

    auto parseBracketed = [&](const std::string& prefix, bool isMaybeVector) -> bool {
        if (t.rfind(prefix, 0) != 0) return false;
        size_t lt = prefix.size() - 1; // position of '<'
        if (t.back() != '>') return false;
        std::string inner = t.substr(lt + 1, t.size() - lt - 2);
        // Split trailing element type from leading dims "2x3xi64" -> dims {2,3} elem "i64"
        std::vector<int64_t> dims;
        size_t pos = 0;
        std::string elem;
        while (true) {
            size_t x = inner.find('x', pos);
            if (x == std::string::npos) { elem = inner.substr(pos); break; }
            std::string num = inner.substr(pos, x - pos);
            bool digitsOnly = !num.empty() && std::all_of(num.begin(), num.end(),
                [](unsigned char c){ return std::isdigit(c) != 0; });
            if (!digitsOnly) { elem = inner.substr(pos); break; }
            dims.push_back(static_cast<int64_t>(std::strtoll(num.c_str(), nullptr, 10)));
            pos = x + 1;
        }
        ::bt::mlir::MLIRType elemType;
        if (!parseMLIRTypeToken(elem, elemType)) return false;
        if (isMaybeVector) {
            if (dims.size() != 1) return false;
            out = ::bt::mlir::MLIRType::getVector(static_cast<unsigned>(dims[0]), elemType);
        } else {
            out = ::bt::mlir::MLIRType::getTensor(dims, elemType);
        }
        return true;
    };
    if (parseBracketed("tensor<", false)) return true;
    if (parseBracketed("vector<", true)) return true;

    if (t.size() > 1 && t[0] == 'i' && std::all_of(t.begin() + 1, t.end(),
        [](unsigned char c){ return std::isdigit(c) != 0; })) {
        out = ::bt::mlir::MLIRType();
        out.kind = ::bt::mlir::TypeKind::Integer;
        out.bitwidth = static_cast<unsigned>(std::strtoul(t.c_str() + 1, nullptr, 10));
        return true;
    }
    return false;
}

} // anonymous namespace

bool isKnownBtOpName(const std::string& opName) {
    return g_registered_ops.count(opName) != 0;
}

bool isKnownBtTypeName(const std::string& typeName) {
    return g_registered_types.count("!" + typeName) != 0;
}

bool loadMLIRModuleFromString(::bt::mlir::MLIRModule& mod, const std::string& text, std::string& error) {
    // The op/type registries may be cold if the caller never registered the
    // dialect; initialize idempotently so validation is always meaningful.
    if (!isTernaryDialectRegistered()) {
        TernaryDialect dialect;
        dialect.initialize();
    }

    const std::string stripped = blankOutComments(text);

    // 1. Structural validation: balanced braces over comment-free text.
    {
        int depth = 0;
        bool inString = false;
        for (size_t i = 0; i < stripped.size(); ++i) {
            char c = stripped[i];
            if (c == '"') inString = !inString;
            if (inString) continue;
            if (c == '{') ++depth;
            else if (c == '}') {
                --depth;
                if (depth < 0) {
                    error = "line " + std::to_string(lineOfOffset(text, i)) + ": unbalanced '}'";
                    return false;
                }
            }
        }
        if (depth != 0) {
            error = "unbalanced braces: missing " + std::to_string(depth) + " closing brace(s)";
            return false;
        }
    }

    // Module symbol name (only honored when the input declares `module @name`).
    const std::string moduleName = [&]() -> std::string {
        size_t p = stripped.find("module");
        while (p != std::string::npos) {
            bool boundary = (p == 0 || std::isspace(static_cast<unsigned char>(stripped[p - 1])) ||
                             stripped[p - 1] == '{' || stripped[p - 1] == '\n');
            if (boundary) {
                size_t q = p + 6;
                skipWs(stripped, q);
                if (q < stripped.size() && stripped[q] == '@') {
                    size_t start = ++q;
                    while (q < stripped.size() &&
                           (std::isalnum(static_cast<unsigned char>(stripped[q])) ||
                            stripped[q] == '_' || stripped[q] == '.')) ++q;
                    if (q > start) return stripped.substr(start, q - start);
                }
                p = stripped.find("module", p + 6);
            } else {
                p = stripped.find("module", p + 6);
            }
        }
        return "";
    }();
    if (!moduleName.empty()) mod.name = moduleName;

    // 2. Validate every bt.* operation and !bt.* type reference against the
    //    registered dialect tables (unknown mnemonics are a hard error).
    {
        bool inString = false;
        for (size_t i = 0; i < stripped.size(); ++i) {
            char c = stripped[i];
            if (c == '"') { inString = !inString; continue; }
            if (inString) continue;

            if (c == '!' && stripped.compare(i + 1, 3, "bt.") == 0) {
                size_t j = i + 4;
                while (j < stripped.size() &&
                       (std::isalnum(static_cast<unsigned char>(stripped[j])) || stripped[j] == '_')) ++j;
                std::string name = stripped.substr(i + 1, j - (i + 1));
                if (!isKnownBtTypeName(name)) {
                    error = "line " + std::to_string(lineOfOffset(text, i)) + ": unknown bt dialect type '!" + name + "'";
                    return false;
                }
                i = j;
                continue;
            }

            bool boundaryBefore = (i == 0) ||
                !(std::isalnum(static_cast<unsigned char>(stripped[i - 1])) ||
                  stripped[i - 1] == '_' || stripped[i - 1] == '!' ||
                  stripped[i - 1] == '@' || stripped[i - 1] == '"' || stripped[i - 1] == '%');
            if (boundaryBefore && stripped.compare(i, 3, "bt.") == 0) {
                size_t j = i + 3;
                if (j < stripped.size() && (std::isalpha(static_cast<unsigned char>(stripped[j])) ||
                                            stripped[j] == '_')) {
                    while (j < stripped.size() &&
                           (std::isalnum(static_cast<unsigned char>(stripped[j])) || stripped[j] == '_')) ++j;
                    std::string name = stripped.substr(i, j - i);
                    if (!isKnownBtOpName(name)) {
                        error = "line " + std::to_string(lineOfOffset(text, i)) + ": unknown bt dialect operation '" + name + "'";
                        return false;
                    }
                }
            }
        }
    }

    // 3. Capture top-level #map / attribute definitions verbatim (depth 0 lines).
    {
        int depth = 0;
        bool inString = false;
        size_t lineStart = 0;
        for (size_t i = 0; i <= stripped.size(); ++i) {
            if (i == stripped.size() || stripped[i] == '\n') {
                if (depth == 0 && !inString) {
                    std::string line = trim(stripped.substr(lineStart, i - lineStart));
                    if (!line.empty() && line[0] == '#') {
                        bool present = false;
                        for (const auto& m : mod.affineMaps) if (m == line) { present = true; break; }
                        if (!present) mod.affineMaps.push_back(line);
                    }
                }
                lineStart = i + 1;
                continue;
            }
            char c = stripped[i];
            if (c == '"') inString = !inString;
            if (!inString) {
                if (c == '{') ++depth;
                else if (c == '}') --depth;
            }
        }
    }

    // 4. Parse func.func records.
    size_t scanPos = 0;
    while ((scanPos = stripped.find("func.func", scanPos)) != std::string::npos) {
        bool boundary = (scanPos == 0) ||
            !(std::isalnum(static_cast<unsigned char>(stripped[scanPos - 1])) || stripped[scanPos - 1] == '_');
        if (!boundary) { ++scanPos; continue; }

        size_t p = scanPos + 9;
        skipWs(stripped, p);
        // optional visibility keyword
        if (stripped.compare(p, 6, "public") == 0) {
            p += 6;
            skipWs(stripped, p);
        } else if (stripped.compare(p, 7, "private") == 0) {
            p += 7;
            skipWs(stripped, p);
        }
        if (p >= stripped.size() || stripped[p] != '@') {
            error = "line " + std::to_string(lineOfOffset(text, scanPos)) + ": func.func missing '@symbol'";
            return false;
        }
        ++p;
        std::string fname;
        if (!readIdentifier(stripped, p, fname)) {
            error = "line " + std::to_string(lineOfOffset(text, scanPos)) + ": malformed func.func symbol name";
            return false;
        }

        // Arguments
        std::vector<::bt::mlir::MLIRValue> args;
        skipWs(stripped, p);
        if (p < stripped.size() && stripped[p] == '(') {
            size_t q = p + 1;
            int depth = 0;
            bool inString = false;
            while (q < stripped.size()) {
                char c = stripped[q];
                if (c == '"') inString = !inString;
                if (!inString) {
                    if (c == '(' || c == '<') ++depth;
                    else if (c == ')') { if (depth == 0) break; --depth; }
                    else if (c == '>') --depth;
                }
                ++q;
            }
            if (q >= stripped.size()) {
                error = "line " + std::to_string(lineOfOffset(text, p)) + ": unterminated argument list";
                return false;
            }
            std::string argText = stripped.substr(p + 1, q - (p + 1));
            std::vector<std::string> parts;
            {
                std::string cur;
                depth = 0;
                bool argStr = false;
                for (char c : argText) {
                    if (c == '"') argStr = !argStr;
                    if (!argStr) {
                        if (c == '<' || c == '(') ++depth;
                        else if (c == '>' || c == ')') --depth;
                    }
                    if (c == ',' && depth == 0 && !argStr) { parts.push_back(cur); cur.clear(); }
                    else cur.push_back(c);
                }
                if (!trim(cur).empty()) parts.push_back(cur);
            }
            for (const auto& part : parts) {
                std::string piece = trim(part);
                if (piece.empty()) {
                    error = "line " + std::to_string(lineOfOffset(text, p)) + ": empty argument in func.func @" + fname;
                    return false;
                }
                size_t r = 0;
                if (piece[r] != '%') {
                    error = "line " + std::to_string(lineOfOffset(text, p)) + ": argument missing %value in func.func @" + fname + " ('" + piece + "')";
                    return false;
                }
                ++r;
                std::string vname;
                if (!readIdentifier(piece, r, vname)) {
                    error = "line " + std::to_string(lineOfOffset(text, p)) + ": bad SSA name in func.func @" + fname;
                    return false;
                }
                skipWs(piece, r);
                if (r >= piece.size() || piece[r] != ':') {
                    error = "line " + std::to_string(lineOfOffset(text, p)) + ": argument missing ':' in func.func @" + fname;
                    return false;
                }
                ++r;
                std::string typeTok = readTypeToken(piece, r);
                std::string leftover = trim(piece.substr(r));
                if (typeTok.empty() || !leftover.empty()) {
                    error = "line " + std::to_string(lineOfOffset(text, p)) + ": unsupported argument type in func.func @" + fname + " ('" + (leftover.empty() ? typeTok : leftover) + "')";
                    return false;
                }
                ::bt::mlir::MLIRType type;
                if (!parseMLIRTypeToken(typeTok, type)) {
                    error = "line " + std::to_string(lineOfOffset(text, p)) + ": unsupported MLIR type '" + typeTok + "' (standalone loader supports iN/index/tensor/vector/!bt.* types)";
                    return false;
                }
                args.push_back(::bt::mlir::MLIRValue("%" + vname, type));
            }
            p = q + 1;
        }

        // Optional return types
        std::vector<::bt::mlir::MLIRType> retTypes;
        skipWs(stripped, p);
        if (stripped.compare(p, 2, "->") == 0) {
            p += 2;
            skipWs(stripped, p);
            std::string retBlob;
            bool single = false;
            if (p < stripped.size() && stripped[p] == '(') {
                int depth = 0;
                bool inString = false;
                size_t q = p;
                while (q < stripped.size()) {
                    char c = stripped[q];
                    if (c == '"') inString = !inString;
                    if (!inString) {
                        if (c == '(') ++depth;
                        else if (c == ')') { --depth; if (depth == 0) break; }
                    }
                    ++q;
                }
                if (q >= stripped.size()) {
                    error = "line " + std::to_string(lineOfOffset(text, p)) + ": unterminated return type list";
                    return false;
                }
                retBlob = stripped.substr(p + 1, q - (p + 1));
                p = q + 1;
            } else {
                retBlob = readTypeToken(stripped, p);
                single = true;
                if (retBlob.empty()) {
                    error = "line " + std::to_string(lineOfOffset(text, p)) + ": malformed return type";
                    return false;
                }
            }

            std::vector<std::string> parts;
            if (single) {
                parts.push_back(retBlob);
            } else {
                std::string cur;
                int depth = 0;
                bool inStr = false;
                for (char c : retBlob) {
                    if (c == '"') inStr = !inStr;
                    if (!inStr) {
                        if (c == '<' || c == '(') ++depth;
                        else if (c == '>' || c == ')') --depth;
                    }
                    if (c == ',' && depth == 0 && !inStr) { parts.push_back(cur); cur.clear(); }
                    else cur.push_back(c);
                }
                if (!trim(cur).empty()) parts.push_back(cur);
            }
            for (const auto& part : parts) {
                std::string tok = trim(part);
                ::bt::mlir::MLIRType type;
                if (!parseMLIRTypeToken(tok, type)) {
                    error = "line " + std::to_string(lineOfOffset(text, p)) + ": unsupported MLIR return type '" + tok + "'";
                    return false;
                }
                retTypes.push_back(type);
            }
        }

        // Function body (or bare external declaration)
        skipWs(stripped, p);
        size_t bodyEnd = std::string::npos;
        if (p < stripped.size() && stripped[p] == '{') {
            int depth = 1;
            bool inString = false;
            size_t q = p + 1;
            while (q < stripped.size() && depth > 0) {
                char c = stripped[q];
                if (c == '"') inString = !inString;
                if (!inString) {
                    if (c == '{') ++depth;
                    else if (c == '}') --depth;
                }
                if (depth > 0) ++q;
            }
            if (depth != 0) {
                error = "line " + std::to_string(lineOfOffset(text, p)) + ": unterminated function body for @" + fname;
                return false;
            }
            bodyEnd = q;
        }

        for (const auto& existing : mod.functions) {
            if (existing.name == fname) {
                error = "line " + std::to_string(lineOfOffset(text, scanPos)) + ": duplicate function '@" + fname + "' in module";
                return false;
            }
        }

        if (bodyEnd == std::string::npos) {
            // External declaration: preserve the raw declaration text verbatim.
            size_t nl = stripped.find('\n', scanPos);
            std::string decl = trim(text.substr(scanPos, (nl == std::string::npos ? text.size() : nl) - scanPos));
            mod.affineMaps.push_back(decl);
        } else {
            ::bt::mlir::MLIRFunction fn;
            fn.name = fname;
            fn.arguments = args;
            fn.returnTypes = retTypes;
            // Raw body from the *original* text keeps input comments intact,
            // while all structural decisions above used comment-free text.
            std::string body = trim(text.substr(p + 1, bodyEnd - (p + 1)));
            if (!body.empty()) fn.rawCustomBody = body + "\n";
            mod.functions.push_back(fn);
            scanPos = bodyEnd + 1; // skip nested func.func occurrences in this body
            continue;
        }
        scanPos += 9;
    }

    return true;
}

} // namespace mlir::bt
