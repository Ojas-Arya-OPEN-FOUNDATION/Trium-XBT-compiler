// =============================================================================
// File: hw/tb_riscv_xbt_alu.sv
// Description: Self-checking IEEE 1800-2017 SystemVerilog testbench for
//              riscv_xbt_alu (v1 + v2 ISA) and xbt_kogge_stone_adder64_v2.
//
// Simulator Compatibility:
//   - Verilator : verilator --binary -Wall -Wno-DECLFILENAME hw/riscv_xbt_extension.sv hw/tb_riscv_xbt_alu.sv && ./obj_dir/Vtb_riscv_xbt_alu
//   - Verilator : verilator --lint-only -Wall -Wno-DECLFILENAME hw/riscv_xbt_extension.sv hw/tb_riscv_xbt_alu.sv
//   - Icarus    : iverilog -g2012 -o tb_xbt hw/riscv_xbt_extension.sv hw/tb_riscv_xbt_alu.sv && vvp tb_xbt
// =============================================================================

`timescale 1ns/1ps
`default_nettype none

module tb_riscv_xbt_alu;

    localparam logic [6:0] OPCODE_XBT = 7'b0001011;

    // DUT 1: Top-level riscv_xbt_alu
    logic [6:0]         opcode;
    logic [2:0]         funct3;
    logic [6:0]         funct7;
    logic [63:0]        rs1_p, rs1_n;
    logic [63:0]        rs2_p, rs2_n;
    logic [63:0]        rd_p_in, rd_n_in;
    logic [63:0]        rs1_scalar;
    logic [11:0]        imm12;
    logic [63:0]        rd_p, rd_n;
    logic [63:0]        rd_aux_p, rd_aux_n;
    logic signed [63:0] rd_scalar;
    logic               valid_out;

    riscv_xbt_alu dut_alu (
        .opcode     (opcode),
        .funct3     (funct3),
        .funct7     (funct7),
        .rs1_p      (rs1_p),
        .rs1_n      (rs1_n),
        .rs2_p      (rs2_p),
        .rs2_n      (rs2_n),
        .rd_p_in    (rd_p_in),
        .rd_n_in    (rd_n_in),
        .rs1_scalar (rs1_scalar),
        .imm12      (imm12),
        .rd_p       (rd_p),
        .rd_n       (rd_n),
        .rd_aux_p   (rd_aux_p),
        .rd_aux_n   (rd_aux_n),
        .rd_scalar  (rd_scalar),
        .valid_out  (valid_out)
    );

    // DUT 2: xbt_kogge_stone_adder64_v2 vs reference xbt_kogge_stone_adder64
    logic [63:0] ks_a_p, ks_a_n, ks_b_p, ks_b_n;
    logic        ks_cin_p, ks_cin_n;
    logic [63:0] ks1_sum_p, ks1_sum_n;
    logic        ks1_cout_p, ks1_cout_n;
    logic [63:0] ks2_sum_p, ks2_sum_n;
    logic        ks2_cout_p, ks2_cout_n;

    xbt_kogge_stone_adder64 dut_ks1 (
        .a_p    (ks_a_p),
        .a_n    (ks_a_n),
        .b_p    (ks_b_p),
        .b_n    (ks_b_n),
        .sum_p  (ks1_sum_p),
        .sum_n  (ks1_sum_n),
        .cout_p (ks1_cout_p),
        .cout_n (ks1_cout_n)
    );

    xbt_kogge_stone_adder64_v2 dut_ks2 (
        .a_p    (ks_a_p),
        .a_n    (ks_a_n),
        .b_p    (ks_b_p),
        .b_n    (ks_b_n),
        .cin_p  (ks_cin_p),
        .cin_n  (ks_cin_n),
        .sum_p  (ks2_sum_p),
        .sum_n  (ks2_sum_n),
        .cout_p (ks2_cout_p),
        .cout_n (ks2_cout_n)
    );

    int pass_count;
    int fail_count;
    logic [63:0] saved_p, saved_n;

    task automatic clear_inputs();
        begin
            opcode     = OPCODE_XBT;
            funct3     = 3'b000;
            funct7     = 7'h00;
            rs1_p      = 64'd0;
            rs1_n      = 64'd0;
            rs2_p      = 64'd0;
            rs2_n      = 64'd0;
            rd_p_in    = 64'd0;
            rd_n_in    = 64'd0;
            rs1_scalar = 64'd0;
            imm12      = 12'd0;
        end
    endtask

    task automatic check_alu_rails(
        input string       name,
        input logic [63:0] exp_p,
        input logic [63:0] exp_n,
        input logic [63:0] exp_aux_p,
        input logic [63:0] exp_aux_n,
        input logic        exp_valid
    );
        begin
            #1;
            if ((rd_p & rd_n) != 64'd0) begin
                $display("[FAIL] %s: invalid dual-rail overlap rd_p=%h rd_n=%h", name, rd_p, rd_n);
                fail_count++;
            end else if (rd_p !== exp_p || rd_n !== exp_n ||
                         rd_aux_p !== exp_aux_p || rd_aux_n !== exp_aux_n ||
                         valid_out !== exp_valid) begin
                $display("[FAIL] %s: got rd=(%h,%h) aux=(%h,%h) valid=%b; exp rd=(%h,%h) aux=(%h,%h) valid=%b",
                         name, rd_p, rd_n, rd_aux_p, rd_aux_n, valid_out,
                         exp_p, exp_n, exp_aux_p, exp_aux_n, exp_valid);
                fail_count++;
            end else begin
                pass_count++;
            end
        end
    endtask

    task automatic check_alu_scalar(
        input string              name,
        input logic signed [63:0] exp_scalar,
        input logic               exp_valid
    );
        begin
            #1;
            if (rd_scalar !== exp_scalar || valid_out !== exp_valid) begin
                $display("[FAIL] %s: got scalar=%0d valid=%b; exp scalar=%0d valid=%b",
                         name, rd_scalar, valid_out, exp_scalar, exp_valid);
                fail_count++;
            end else begin
                pass_count++;
            end
        end
    endtask

    initial begin
        pass_count = 0;
        fail_count = 0;
        ks_a_p = 64'd0; ks_a_n = 64'd0;
        ks_b_p = 64'd0; ks_b_n = 64'd0;
        ks_cin_p = 1'b0; ks_cin_n = 1'b0;

        $display("=== Running tb_riscv_xbt_alu (v1 + v2 + KSADD_v2) ===");

        // ---------------------------------------------------------------------
        // 1. v1: XBT.BTHA (funct3 = 3'b000): (+1) + (+1) => sum=-1, carry=+1
        // ---------------------------------------------------------------------
        clear_inputs();
        funct3 = 3'b000;
        rs1_p  = 64'h0000_0000_0000_0001;
        rs2_p  = 64'h0000_0000_0000_0001;
        check_alu_rails("v1_BTHA_pos_pos", 64'd0, 64'd1, 64'd1, 64'd0, 1'b1);

        // (-1) + (-1) => sum=+1, carry=-1
        clear_inputs();
        funct3 = 3'b000;
        rs1_n  = 64'h0000_0000_0000_0001;
        rs2_n  = 64'h0000_0000_0000_0001;
        check_alu_rails("v1_BTHA_neg_neg", 64'd1, 64'd0, 64'd0, 64'd1, 1'b1);

        // ---------------------------------------------------------------------
        // 2. v1: XBT.KSADD (funct3 = 3'b001): 364 (++++++) + 1 = 365 (+------)
        // ---------------------------------------------------------------------
        clear_inputs();
        funct3 = 3'b001;
        rs1_p  = 64'h0000_0000_0000_003F; // +364
        rs2_p  = 64'h0000_0000_0000_0001; // +1
        check_alu_rails("v1_KSADD_ripple_pos", 64'h40, 64'h3F, 64'd0, 64'd0, 1'b1);

        // Negative ripple: -364 (------) + (-1) = -365 (-++++++)
        clear_inputs();
        funct3 = 3'b001;
        rs1_n  = 64'h0000_0000_0000_003F;
        rs2_n  = 64'h0000_0000_0000_0001;
        check_alu_rails("v1_KSADD_ripple_neg", 64'h3F, 64'h40, 64'd0, 64'd0, 1'b1);

        // ---------------------------------------------------------------------
        // 3. v1: XBT.DOT64 (funct3 = 3'b010)
        // ---------------------------------------------------------------------
        clear_inputs();
        funct3 = 3'b010;
        rs1_p  = 64'h5555_5555_5555_5555; // 32 (+1)
        rs1_n  = 64'hAAAA_AAAA_AAAA_AAAA; // 32 (-1)
        rs2_p  = 64'h5555_5555_5555_5555; // 32 (+1) aligned with +1
        check_alu_scalar("v1_DOT64_pos32", 64'sd32, 1'b1);

        clear_inputs();
        funct3 = 3'b010;
        rs1_p  = 64'h0000_0000_0000_00FF; // 8 (+1)
        rs2_n  = 64'h0000_0000_0000_00FF; // 8 (-1) -> dot = -8
        check_alu_scalar("v1_DOT64_neg8", -64'sd8, 1'b1);

        // ---------------------------------------------------------------------
        // 4. v1: XBT.UNPACK243 (funct3 = 3'b011)
        //    byte 0 -> -121 (-----), byte 121 -> 0 (00000), byte 242 -> +121 (+++++)
        //    byte 243 -> invalid -> 0
        // ---------------------------------------------------------------------
        clear_inputs();
        funct3     = 3'b011;
        rs1_scalar = 64'd0;
        check_alu_rails("v1_UNPACK243_0", 64'd0, 64'h1F, 64'd0, 64'd0, 1'b1);

        clear_inputs();
        funct3     = 3'b011;
        rs1_scalar = 64'd121;
        check_alu_rails("v1_UNPACK243_121", 64'd0, 64'd0, 64'd0, 64'd0, 1'b1);

        clear_inputs();
        funct3     = 3'b011;
        rs1_scalar = 64'd242;
        check_alu_rails("v1_UNPACK243_242", 64'h1F, 64'd0, 64'd0, 64'd0, 1'b1);

        clear_inputs();
        funct3     = 3'b011;
        rs1_scalar = 64'd243;
        check_alu_rails("v1_UNPACK243_invalid243", 64'd0, 64'd0, 64'd0, 64'd0, 1'b1);

        // ---------------------------------------------------------------------
        // 5. v1: XBT.NEG (funct3 = 3'b100)
        // ---------------------------------------------------------------------
        clear_inputs();
        funct3 = 3'b100;
        rs1_p  = 64'h0F0F_00FF_1234_0000;
        rs1_n  = 64'hF0F0_FF00_0000_5678;
        check_alu_rails("v1_NEG", 64'hF0F0_FF00_0000_5678, 64'h0F0F_00FF_1234_0000, 64'd0, 64'd0, 1'b1);

        // ---------------------------------------------------------------------
        // 6. v1: XBT.CSAMAC (funct3 = 3'b101): rd_in + (rs1 * rs2), distinct rd != rs1
        //    rs1 = -1, rs2 = -1 => prod = +1; rd_in = 0 => sum = +1, carry = 0
        //    (The old bug using rs1=-1 as acc returned sum=0, carry=0!)
        // ---------------------------------------------------------------------
        clear_inputs();
        funct3  = 3'b101;
        rs1_n   = 64'd1;
        rs2_n   = 64'd1;
        rd_p_in = 64'd0;
        rd_n_in = 64'd0;
        check_alu_rails("v1_CSAMAC_distinct_rd0", 64'd1, 64'd0, 64'd0, 64'd0, 1'b1);

        // rs1 = -1, rs2 = -1 => prod = +1; rd_in = +1 => sum = -1, carry = +1
        clear_inputs();
        funct3  = 3'b101;
        rs1_n   = 64'd1;
        rs2_n   = 64'd1;
        rd_p_in = 64'd1;
        rd_n_in = 64'd0;
        check_alu_rails("v1_CSAMAC_distinct_rd_pos1", 64'd0, 64'd1, 64'd1, 64'd0, 1'b1);

        // ---------------------------------------------------------------------
        // 7. v2: GROUP2.BTFA (funct3=3'b110, funct7=7'h00): (+1)+(+1)+(+1) => sum=0, carry=+1
        // ---------------------------------------------------------------------
        clear_inputs();
        funct3  = 3'b110;
        funct7  = 7'h00;
        rs1_p   = 64'd1;
        rs2_p   = 64'd1;
        rd_p_in = 64'd1;
        check_alu_rails("v2_BTFA_three_pos", 64'd0, 64'd0, 64'd1, 64'd0, 1'b1);

        // (-1)+(-1)+(-1) => sum=0, carry=-1
        clear_inputs();
        funct3  = 3'b110;
        funct7  = 7'h00;
        rs1_n   = 64'd1;
        rs2_n   = 64'd1;
        rd_n_in = 64'd1;
        check_alu_rails("v2_BTFA_three_neg", 64'd0, 64'd0, 64'd0, 64'd1, 1'b1);

        // ---------------------------------------------------------------------
        // 8. v2: GROUP2.MUL (funct3=3'b110, funct7=7'h01): per-lane AOI22
        // ---------------------------------------------------------------------
        clear_inputs();
        funct3 = 3'b110;
        funct7 = 7'h01;
        rs1_p  = 64'b0011; // lanes 0,1 = +1
        rs1_n  = 64'b1100; // lanes 2,3 = -1
        rs2_p  = 64'b0101; // lanes 0,2 = +1
        rs2_n  = 64'b1010; // lanes 1,3 = -1
        // Expected: lane0=+1, lane1=-1, lane2=-1, lane3=+1 => P=1001, N=0110
        check_alu_rails("v2_MUL_lane", 64'b1001, 64'b0110, 64'd0, 64'd0, 1'b1);

        // ---------------------------------------------------------------------
        // 9. v2: GROUP2.PACK243 (funct3=3'b110, funct7=7'h02) + UNPACK243 round-trip
        // ---------------------------------------------------------------------
        for (int b = 0; b < 243; b++) begin
            clear_inputs();
            funct3     = 3'b011;
            rs1_scalar = 64'(b);
            #1;
            saved_p = rd_p;
            saved_n = rd_n;

            clear_inputs();
            funct3 = 3'b110;
            funct7 = 7'h02;
            rs1_p  = saved_p;
            rs1_n  = saved_n;
            #1;
            if (rd_scalar !== $signed({32'd0, b}) || valid_out !== 1'b1) begin
                $display("[FAIL] v2_PACK243_roundtrip byte=%0d got=%0d valid=%b", b, rd_scalar, valid_out);
                fail_count++;
            end else begin
                pass_count++;
            end
        end

        // ---------------------------------------------------------------------
        // 10. v2: GROUP2.SIGN (funct3=3'b110, funct7=7'h03)
        // ---------------------------------------------------------------------
        clear_inputs();
        funct3 = 3'b110;
        funct7 = 7'h03;
        check_alu_rails("v2_SIGN_zero", 64'd0, 64'd0, 64'd0, 64'd0, 1'b1);

        clear_inputs();
        funct3 = 3'b110;
        funct7 = 7'h03;
        rs1_p  = 64'h0000_0000_0000_0100; // MSB at bit 8 (+1) dominates lower -1s
        rs1_n  = 64'h0000_0000_0000_00FF;
        check_alu_rails("v2_SIGN_pos", 64'd1, 64'd0, 64'd0, 64'd0, 1'b1);

        clear_inputs();
        funct3 = 3'b110;
        funct7 = 7'h03;
        rs1_p  = 64'h0000_0000_0000_00FF;
        rs1_n  = 64'h8000_0000_0000_0000; // MSB at bit 63 (-1)
        check_alu_rails("v2_SIGN_neg", 64'd0, 64'd1, 64'd0, 64'd0, 1'b1);

        // ---------------------------------------------------------------------
        // 11. v2: GROUP2.CMOVZ / CMOVP / CMOVN (funct7 = 7'h04 / 7'h05 / 7'h06)
        // ---------------------------------------------------------------------
        // CMOVZ taken (rs1 == 0)
        clear_inputs();
        funct3  = 3'b110;
        funct7  = 7'h04;
        rs2_p   = 64'h0000_0000_0000_00AA;
        rd_p_in = 64'h0000_0000_0000_0055;
        check_alu_rails("v2_CMOVZ_taken", 64'hAA, 64'd0, 64'd0, 64'd0, 1'b1);

        // CMOVZ not taken (rs1 > 0) -> preserves rd_in
        clear_inputs();
        funct3  = 3'b110;
        funct7  = 7'h04;
        rs1_p   = 64'd1;
        rs2_p   = 64'hAA;
        rd_p_in = 64'h55;
        check_alu_rails("v2_CMOVZ_nottaken", 64'h55, 64'd0, 64'd0, 64'd0, 1'b1);

        // CMOVP taken (rs1 > 0)
        clear_inputs();
        funct3  = 3'b110;
        funct7  = 7'h05;
        rs1_p   = 64'h10;
        rs2_n   = 64'h0F;
        rd_p_in = 64'h55;
        check_alu_rails("v2_CMOVP_taken", 64'd0, 64'h0F, 64'd0, 64'd0, 1'b1);

        // CMOVN taken (rs1 < 0)
        clear_inputs();
        funct3  = 3'b110;
        funct7  = 7'h06;
        rs1_n   = 64'h10;
        rs2_p   = 64'h0F;
        rd_p_in = 64'h55;
        check_alu_rails("v2_CMOVN_taken", 64'h0F, 64'd0, 64'd0, 64'd0, 1'b1);

        // CMOVN not taken (rs1 == 0)
        clear_inputs();
        funct3  = 3'b110;
        funct7  = 7'h06;
        rs2_p   = 64'h0F;
        rd_n_in = 64'h33;
        check_alu_rails("v2_CMOVN_nottaken", 64'd0, 64'h33, 64'd0, 64'd0, 1'b1);

        // ---------------------------------------------------------------------
        // 12. v2: GROUP2.SHL3 (funct3=3'b110, funct7=7'h07)
        // ---------------------------------------------------------------------
        clear_inputs();
        funct3 = 3'b110;
        funct7 = 7'h07;
        rs1_p  = 64'h0000_0000_0000_0005;
        rs1_n  = 64'h0000_0000_0000_0002;
        imm12  = {7'h07, 5'd4}; // shamt = 4
        check_alu_rails("v2_SHL3_shift4", 64'h50, 64'h20, 64'd0, 64'd0, 1'b1);

        // SHL3 overflow: bit 62 set, shift by 2 -> must refuse (valid_out == 0)
        clear_inputs();
        funct3 = 3'b110;
        funct7 = 7'h07;
        rs1_p  = 64'h4000_0000_0000_0000;
        imm12  = {7'h07, 5'd2};
        check_alu_rails("v2_SHL3_overflow", 64'd0, 64'd0, 64'd0, 64'd0, 1'b0);

        // ---------------------------------------------------------------------
        // 13. v2: XBT.LI5 (funct3 = 3'b111)
        //     chunk=2 (shift=10), byte=242 (+++++) => rd_p = 0x1F << 10 = 0x7C00
        // ---------------------------------------------------------------------
        clear_inputs();
        funct3 = 3'b111;
        imm12  = {1'b0, 3'd2, 8'd242};
        check_alu_rails("v2_LI5_chunk2_max", 64'h0000_0000_0000_7C00, 64'd0, 64'd0, 64'd0, 1'b1);

        // chunk=7 (shift=35), byte=0 (-----) => rd_n = 0x1F << 35
        clear_inputs();
        funct3 = 3'b111;
        imm12  = {1'b0, 3'd7, 8'd0};
        check_alu_rails("v2_LI5_chunk7_min", 64'd0, (64'h1F << 35), 64'd0, 64'd0, 1'b1);

        // invalid byte=250 -> (0, 0)
        clear_inputs();
        funct3 = 3'b111;
        imm12  = {1'b0, 3'd3, 8'd250};
        check_alu_rails("v2_LI5_invalid_byte", 64'd0, 64'd0, 64'd0, 64'd0, 1'b1);

        // ---------------------------------------------------------------------
        // 14. xbt_kogge_stone_adder64_v2 vs xbt_kogge_stone_adder64 (cin=0)
        //     and explicit cin=+1 / cin=-1 ripple tests
        // ---------------------------------------------------------------------
        ks_cin_p = 1'b0; ks_cin_n = 1'b0;

        // Long negative ripple where old fp_n bug (l_fm_n vs l_fp_n) manifested
        ks_a_p = 64'h0000_0000_0000_0000; ks_a_n = 64'hFFFF_FFFF_FFFF_FFFF;
        ks_b_p = 64'h0000_0000_0000_0000; ks_b_n = 64'h0000_0000_0000_0001;
        #1;
        if (ks2_sum_p !== ks1_sum_p || ks2_sum_n !== ks1_sum_n ||
            ks2_cout_p !== ks1_cout_p || ks2_cout_n !== ks1_cout_n) begin
            $display("[FAIL] KSADD_v2_neg_ripple: v2=(%h,%h,%b,%b) != v1=(%h,%h,%b,%b)",
                     ks2_sum_p, ks2_sum_n, ks2_cout_p, ks2_cout_n,
                     ks1_sum_p, ks1_sum_n, ks1_cout_p, ks1_cout_n);
            fail_count++;
        end else begin
            pass_count++;
        end

        // Mixed alternating trits across all 64 lanes
        ks_a_p = 64'h5555_0F0F_AAAA_3333; ks_a_n = 64'h2222_F0F0_1111_4444;
        ks_b_p = 64'h1111_3333_5555_00FF; ks_b_n = 64'h4444_4444_2222_FF00;
        #1;
        if (ks2_sum_p !== ks1_sum_p || ks2_sum_n !== ks1_sum_n ||
            ks2_cout_p !== ks1_cout_p || ks2_cout_n !== ks1_cout_n) begin
            $display("[FAIL] KSADD_v2_mixed: v2=(%h,%h,%b,%b) != v1=(%h,%h,%b,%b)",
                     ks2_sum_p, ks2_sum_n, ks2_cout_p, ks2_cout_n,
                     ks1_sum_p, ks1_sum_n, ks1_cout_p, ks1_cout_n);
            fail_count++;
        end else begin
            pass_count++;
        end

        // KSADD_v2 with cin_p = 1: all-+1 (64 trits) + 0 + cin(+1) => all--1, cout_p=1
        ks_a_p = 64'hFFFF_FFFF_FFFF_FFFF; ks_a_n = 64'd0;
        ks_b_p = 64'd0;                   ks_b_n = 64'd0;
        ks_cin_p = 1'b1; ks_cin_n = 1'b0;
        #1;
        if (ks2_sum_p !== 64'd0 || ks2_sum_n !== 64'hFFFF_FFFF_FFFF_FFFF ||
            ks2_cout_p !== 1'b1 || ks2_cout_n !== 1'b0) begin
            $display("[FAIL] KSADD_v2_cin_pos: got (%h,%h,%b,%b)",
                     ks2_sum_p, ks2_sum_n, ks2_cout_p, ks2_cout_n);
            fail_count++;
        end else begin
            pass_count++;
        end

        // KSADD_v2 with cin_n = 1: all--1 (64 trits) + 0 + cin(-1) => all-+1, cout_n=1
        ks_a_p = 64'd0; ks_a_n = 64'hFFFF_FFFF_FFFF_FFFF;
        ks_b_p = 64'd0; ks_b_n = 64'd0;
        ks_cin_p = 1'b0; ks_cin_n = 1'b1;
        #1;
        if (ks2_sum_p !== 64'hFFFF_FFFF_FFFF_FFFF || ks2_sum_n !== 64'd0 ||
            ks2_cout_p !== 1'b0 || ks2_cout_n !== 1'b1) begin
            $display("[FAIL] KSADD_v2_cin_neg: got (%h,%h,%b,%b)",
                     ks2_sum_p, ks2_sum_n, ks2_cout_p, ks2_cout_n);
            fail_count++;
        end else begin
            pass_count++;
        end

        // ---------------------------------------------------------------------
        // Summary
        // ---------------------------------------------------------------------
        $display("=== tb_riscv_xbt_alu summary: %0d passed, %0d failed ===", pass_count, fail_count);
        if (fail_count != 0) begin
            $fatal(1, "tb_riscv_xbt_alu FAILED with %0d mismatches", fail_count);
        end
        $finish;
    end

endmodule

`default_nettype wire
