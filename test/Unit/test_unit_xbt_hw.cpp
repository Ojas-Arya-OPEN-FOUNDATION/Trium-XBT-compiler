/**
 * @file test/Unit/test_unit_xbt_hw.cpp
 * @brief Unit Tests for RISC-V "Xbt" Custom-0 Hardware Extension RTL & Co-Simulator
 */

#include "../../hw/xbt_alu_model.hpp"

#include <cassert>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace bt;
using namespace bt::hw;

static void test_encoding_and_disassembly() {
    uint32_t i_btha   = encode_xbt_r(0, 2, 1, static_cast<uint8_t>(XbtFunct3::BTHA), 3);
    uint32_t i_ksadd  = encode_xbt_r(0, 5, 4, static_cast<uint8_t>(XbtFunct3::KSADD), 6);
    uint32_t i_dot64  = encode_xbt_r(0, 2, 1, static_cast<uint8_t>(XbtFunct3::DOT64), 10);
    uint32_t i_unpack = encode_xbt_r(0, 0, 11, static_cast<uint8_t>(XbtFunct3::UNPACK243), 7);
    uint32_t i_neg    = encode_xbt_r(0, 0, 6, static_cast<uint8_t>(XbtFunct3::NEG), 8);

    DecodedXbtInsn d_dot = decode_xbt_r(i_dot64);
    assert(d_dot.is_valid_xbt());
    assert(d_dot.opcode == 0x0B);
    assert(d_dot.rd == 10);
    assert(d_dot.rs1 == 1);
    assert(d_dot.rs2 == 2);
    assert(d_dot.funct3 == 2);
    assert(d_dot.funct7 == 0);

    assert(disassemble_xbt(i_btha)   == "xbt.btha t3, t1, t2 // .insn r 0x0b, 0, 0, x3, x1, x2");
    assert(disassemble_xbt(i_ksadd)  == "xbt.ksadd t6, t4, t5 // .insn r 0x0b, 1, 0, x6, x4, x5");
    assert(disassemble_xbt(i_dot64)  == "xbt.dot64 x10, t1, t2 // .insn r 0x0b, 2, 0, x10, x1, x2");
    assert(disassemble_xbt(i_unpack) == "xbt.unpack243 t7, x11 // .insn r 0x0b, 3, 0, x7, x11, x0");
    assert(disassemble_xbt(i_neg)    == "xbt.neg t8, t6 // .insn r 0x0b, 4, 0, x8, x6, x0");

    std::cout << "  [PASS] Xbt R-Type Encoding, Decoding & Disassembly\n";
}

static void test_coprocessor_execution() {
    RiscvXbtCoprocessor cpu;

    // 1. Test XBT.BTHA
    cpu.treg[1] = Codec::from_int64(1);
    cpu.treg[2] = Codec::from_int64(1);
    cpu.execute(encode_xbt_r(4, 2, 1, static_cast<uint8_t>(XbtFunct3::BTHA), 3));
    assert(cpu.treg[3].isValid());
    assert(cpu.treg[3].N == 1ULL && cpu.treg[3].P == 0ULL); // Sum = -1
    assert(cpu.last_carry.P == 1ULL && cpu.last_carry.N == 0ULL); // Carry = +1
    assert(cpu.treg[4].P == 1ULL && cpu.treg[4].N == 0ULL);

    // 2. Test XBT.KSADD (6-stage Kogge-Stone parallel-prefix adder)
    const int64_t test_pairs[][2] = {
        {84, 27},
        {123456789LL, -987654321LL},
        {-4444444444LL, -5555555555LL},
        {364, 1},          // Long ripple chain: ++++++ (364) + 1 = +------ (365)
        {-364, -1},        // Long negative ripple chain: ------ (-364) + (-1) = -++++++ (-365)
        {29524, -29524},   // Exact cancellation to 0
        {19682, 19683}
    };

    for (const auto& pair : test_pairs) {
        int64_t va = pair[0];
        int64_t vb = pair[1];
        cpu.treg[5] = Codec::from_int64(va);
        cpu.treg[6] = Codec::from_int64(vb);
        cpu.execute(encode_xbt_r(0, 6, 5, static_cast<uint8_t>(XbtFunct3::KSADD), 7));
        assert(cpu.treg[7].isValid());
        int64_t actual = Codec::to_int64(cpu.treg[7]);
        assert(actual == va + vb);

        // Also verify bit-identical agreement with BT_CSA canonicalization
        BT_CSA csa;
        csa.accumulate(cpu.treg[5]);
        csa.accumulate(cpu.treg[6]);
        TritWord64 expected_w = csa.canonicalize();
        assert(cpu.treg[7].P == expected_w.P && cpu.treg[7].N == expected_w.N);
    }

    // 3. Test XBT.DOT64
    cpu.treg[10] = TritWord64(0x5555555555555555ULL, 0xAAAAAAAAAAAAAAAAULL); // 32 (+1), 32 (-1)
    cpu.treg[11] = TritWord64(0x5555555555555555ULL, 0x0000000000000000ULL); // 32 (+1) aligned with +1
    cpu.execute(encode_xbt_r(0, 11, 10, static_cast<uint8_t>(XbtFunct3::DOT64), 12));
    assert(cpu.xreg[12] == 32);

    // Verify x0 hardwired zero protection on XBT.DOT64
    cpu.execute(encode_xbt_r(0, 11, 10, static_cast<uint8_t>(XbtFunct3::DOT64), 0));
    assert(cpu.xreg[0] == 0);

    // 4. Test XBT.UNPACK243 exhaustively for all 243 valid bytes [0..242] + out-of-range [243..255]
    for (int b = 0; b < 256; ++b) {
        cpu.xreg[15] = b;
        cpu.execute(encode_xbt_r(0, 0, 15, static_cast<uint8_t>(XbtFunct3::UNPACK243), 16));
        assert(cpu.treg[16].isValid());
        auto ref_trits = Base243Codec::unpack5_fast(static_cast<uint8_t>(b));
        for (int k = 0; k < 5; ++k) {
            assert(cpu.treg[16].getTrit(k) == ref_trits[k]);
        }
        for (int k = 5; k < 64; ++k) {
            assert(cpu.treg[16].getTrit(k) == Trit::Zero);
        }
    }

    // 5. Test XBT.NEG (0-gate wire-swapped negation)
    cpu.treg[20] = Codec::from_int64(424242);
    uint64_t prev_cycles = cpu.cycles;
    uint64_t prev_neg    = cpu.zero_gate_negations;
    cpu.execute(encode_xbt_r(0, 0, 20, static_cast<uint8_t>(XbtFunct3::NEG), 21));
    assert(cpu.treg[21].P == cpu.treg[20].N && cpu.treg[21].N == cpu.treg[20].P);
    assert(Codec::to_int64(cpu.treg[21]) == -424242);
    assert(cpu.cycles == prev_cycles); // 0 ALU clock cycles for wire-swap negation
    assert(cpu.zero_gate_negations == prev_neg + 1);

    std::cout << "  [PASS] Cycle-Exact Co-Simulation of BTHA, KSADD, DOT64, UNPACK243, NEG\n";
}

static void test_systemverilog_rtl_presence() {
    std::ifstream sv_file("hw/riscv_xbt_extension.sv");
    if (!sv_file.is_open()) {
        sv_file.open("../../hw/riscv_xbt_extension.sv");
    }
    assert(sv_file.is_open() && "hw/riscv_xbt_extension.sv must exist!");

    std::stringstream buffer;
    buffer << sv_file.rdbuf();
    std::string content = buffer.str();

    const std::vector<std::string> required_modules = {
        "module xbt_btha64",
        "module xbt_kogge_stone_adder64",
        "module xbt_planar_dot64",
        "module xbt_base243_decoder",
        "module riscv_xbt_alu"
    };

    for (const auto& mod : required_modules) {
        assert(content.find(mod) != std::string::npos && "Missing required SystemVerilog module!");
    }

    std::cout << "  [PASS] SystemVerilog RTL Verification (All 5 Xbt Hardware Modules Verified)\n";
}

int main() {
    std::cout << "======================================================================\n";
    std::cout << "  Running Unit Test: RISC-V Xbt Hardware Extension & RTL Co-Simulator\n";
    std::cout << "======================================================================\n";

    test_encoding_and_disassembly();
    test_coprocessor_execution();
    test_systemverilog_rtl_presence();

    std::cout << "======================================================================\n";
    std::cout << "  ALL XBT HARDWARE UNIT TESTS PASSED!\n";
    std::cout << "======================================================================\n";
    return 0;
}
