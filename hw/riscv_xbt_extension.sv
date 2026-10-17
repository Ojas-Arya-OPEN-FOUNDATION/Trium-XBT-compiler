// =============================================================================
// File: hw/riscv_xbt_extension.sv
// Description: Synthesizable IEEE 1800-2017 SystemVerilog RTL for the RISC-V
//              "Xbt" (Balanced Ternary) Custom-0 Hardware Extension (0x0B).
//
// Modules:
//   1. xbt_btha64              - 64-Trit Dual-Rail Balanced Ternary Half-Adder
//   2. xbt_kogge_stone_adder64 - 6-Stage Parallel-Prefix Kogge-Stone Ternary CLA
//   3. xbt_planar_dot64        - 64-Trit Dual-Rail Dot Product Unit (Popcount Tree)
//   4. xbt_base243_decoder     - Division-Free Base-243 Hardware Unpacker
//   5. riscv_xbt_alu           - Top-Level RISC-V RoCC / Custom-0 Execution Unit
// =============================================================================

`default_nettype none

// -----------------------------------------------------------------------------
// 1. 64-Trit Dual-Rail Balanced Ternary Half-Adder (BTHA)
//    Combinational 6-gate network per trit lane.
//    Invariant: (a_p & a_n) == 0, (b_p & b_n) == 0 -> (sum_p & sum_n) == 0
// -----------------------------------------------------------------------------
module xbt_btha64 (
    input  logic [63:0] a_p,
    input  logic [63:0] a_n,
    input  logic [63:0] b_p,
    input  logic [63:0] b_n,
    output logic [63:0] sum_p,
    output logic [63:0] sum_n,
    output logic [63:0] carry_p,
    output logic [63:0] carry_n
);
    always_comb begin
        carry_p = a_p & b_p;
        carry_n = a_n & b_n;
        sum_p   = ((a_p ^ b_p) & ~(a_n | b_n)) | carry_n;
        sum_n   = ((a_n ^ b_n) & ~(a_p | b_p)) | carry_p;
    end
endmodule

// -----------------------------------------------------------------------------
// 2. 64-Trit 6-Stage Parallel-Prefix Kogge-Stone Balanced Ternary Adder
//    Computes exact carry-lookahead across 64 trits in O(log2 64) = 6 stages
//    of pure combinational logic using monotone 3-state transfer functions.
// -----------------------------------------------------------------------------
module xbt_kogge_stone_adder64 (
    input  logic [63:0] a_p,
    input  logic [63:0] a_n,
    input  logic [63:0] b_p,
    input  logic [63:0] b_n,
    output logic [63:0] sum_p,
    output logic [63:0] sum_n,
    output logic        cout_p,
    output logic        cout_n
);
    // Stage 0: Initial per-trit BTHA and transfer function F^{(0)}_i(c_in)
    logic [63:0] s1_p, s1_n, c1_p, c1_n;

    xbt_btha64 u_btha_init (
        .a_p     (a_p),
        .a_n     (a_n),
        .b_p     (b_p),
        .b_n     (b_n),
        .sum_p   (s1_p),
        .sum_n   (s1_n),
        .carry_p (c1_p),
        .carry_n (c1_n)
    );

    // Transfer function rails across 7 levels (0..6):
    // yp = F(+1), y0 = F(0), ym = F(-1)
    logic [63:0] yp_p [0:6];
    logic [63:0] yp_n [0:6];
    logic [63:0] y0_p [0:6];
    logic [63:0] y0_n [0:6];
    logic [63:0] ym_p [0:6];
    logic [63:0] ym_n [0:6];

    always_comb begin
        // Level 0 initialization
        yp_p[0] = (a_p | b_p) & ~(a_n | b_n);
        yp_n[0] = 64'd0;
        y0_p[0] = c1_p;
        y0_n[0] = c1_n;
        ym_p[0] = 64'd0;
        ym_n[0] = (a_n | b_n) & ~(a_p | b_p);
    end

    // 6 Kogge-Stone parallel-prefix doubling stages: d = 1, 2, 4, 8, 16, 32
    genvar s;
    generate
        for (s = 0; s < 6; s = s + 1) begin : gen_ks_stage
            localparam int D = 1 << s;
            localparam logic [63:0] LOW_MASK = (64'd1 << D) - 64'd1;

            logic [63:0] lp_p, lp_n, l0_p, l0_n, lm_p, lm_n;
            logic [63:0] zp, z0, zm;

            always_comb begin
                // Shift lower prefix state by D; fill low D lanes with identity map I
                lp_p = (yp_p[s] << D) | LOW_MASK;
                lp_n = (yp_n[s] << D);
                l0_p = (y0_p[s] << D);
                l0_n = (y0_n[s] << D);
                lm_p = (ym_p[s] << D);
                lm_n = (ym_n[s] << D) | LOW_MASK;

                zp = ~(lp_p | lp_n);
                z0 = ~(l0_p | l0_n);
                zm = ~(lm_p | lm_n);

                // Compose H o L for all 64 lanes in parallel
                yp_p[s+1] = (lp_p & yp_p[s]) | (zp & y0_p[s]) | (lp_n & ym_p[s]);
                yp_n[s+1] = (lp_p & yp_n[s]) | (zp & y0_n[s]) | (lp_n & ym_n[s]);

                y0_p[s+1] = (l0_p & yp_p[s]) | (z0 & y0_p[s]) | (l0_n & ym_p[s]);
                y0_n[s+1] = (l0_p & yp_n[s]) | (z0 & y0_n[s]) | (l0_n & ym_n[s]);

                ym_p[s+1] = (lm_p & yp_p[s]) | (zm & y0_p[s]) | (lm_n & ym_p[s]);
                ym_n[s+1] = (lm_p & yp_n[s]) | (zm & y0_n[s]) | (lm_n & ym_n[s]);
            end
        end
    endgenerate

    // Final Sum resolution: sum_i = BTHA(s1_i, cin_i).Sum
    logic [63:0] cin_p, cin_n;
    always_comb begin
        cin_p  = y0_p[6] << 1;
        cin_n  = y0_n[6] << 1;
        cout_p = y0_p[6][63];
        cout_n = y0_n[6][63];

        sum_p  = ((s1_p ^ cin_p) & ~(s1_n | cin_n)) | (s1_n & cin_n);
        sum_n  = ((s1_n ^ cin_n) & ~(s1_p | cin_p)) | (s1_p & cin_p);
    end
endmodule

// -----------------------------------------------------------------------------
// 3. 64-Trit Dual-Rail Planar Dot Product Unit
//    Computes prod_p = (a_p & b_p) | (a_n & b_n),
//             prod_n = (a_p & b_n) | (a_n & b_p),
//    followed by parallel Wallace/popcount tree reduction in 1 clock cycle.
// -----------------------------------------------------------------------------
module xbt_planar_dot64 (
    input  logic [63:0]        a_p,
    input  logic [63:0]        a_n,
    input  logic [63:0]        b_p,
    input  logic [63:0]        b_n,
    output logic signed [63:0] dot_out
);
    logic [63:0] prod_p, prod_n;
    logic [6:0]  cnt_p, cnt_n;

    always_comb begin
        prod_p = (a_p & b_p) | (a_n & b_n);
        prod_n = (a_p & b_n) | (a_n & b_p);

        cnt_p = 7'd0;
        cnt_n = 7'd0;
        for (int i = 0; i < 64; i++) begin
            cnt_p = cnt_p + {6'd0, prod_p[i]};
            cnt_n = cnt_n + {6'd0, prod_n[i]};
        end

        dot_out = 64'(signed'({1'b0, cnt_p})) - 64'(signed'({1'b0, cnt_n}));
    end
endmodule

// -----------------------------------------------------------------------------
// 4. Division-Free Base-243 Hardware Unpacker
//    Unpacks an 8-bit Base-243 byte [0..242] into 5 Dual-Rail trits in 1 cycle
//    using constant reciprocal multipliers (171, 57, 19, 203).
// -----------------------------------------------------------------------------
module xbt_base243_decoder (
    input  logic [7:0]  packed_byte,
    output logic [63:0] out_p,
    output logic [63:0] out_n
);
    logic [31:0] b32;
    logic [31:0] q1, q2, q3, q4;
    logic [31:0] d [0:4];

    always_comb begin
        out_p = 64'd0;
        out_n = 64'd0;
        b32   = {24'd0, packed_byte};

        q1 = (b32 * 32'd171) >> 9;
        q2 = (b32 * 32'd57)  >> 9;
        q3 = (b32 * 32'd19)  >> 9;
        q4 = (b32 * 32'd203) >> 14;

        d[0] = b32 - (q1 * 32'd3);
        d[1] = q1  - (q2 * 32'd3);
        d[2] = q2  - (q3 * 32'd3);
        d[3] = q3  - (q4 * 32'd3);
        d[4] = q4;

        if (packed_byte < 8'd243) begin
            for (int k = 0; k < 5; k++) begin
                if (d[k] == 32'd2) begin
                    out_p[k] = 1'b1;
                end else if (d[k] == 32'd0) begin
                    out_n[k] = 1'b1;
                end
            end
        end
    end
endmodule

// -----------------------------------------------------------------------------
// 5. Top-Level RISC-V "Xbt" Custom-0 (0x0B) Execution Unit (riscv_xbt_alu)
//    Opcode: 7'b0001011 (0x0B)
//    funct3 dispatch:
//      3'b000 (XBT.BTHA)      : 1-cycle 64-trit BTHA step
//      3'b001 (XBT.KSADD)     : 1-cycle 64-trit Kogge-Stone addition
//      3'b010 (XBT.DOT64)     : 1-cycle 64-trit Planar Dot Product -> rd_scalar
//      3'b011 (XBT.UNPACK243) : 1-cycle Base-243 Byte -> Dual-Rail unpack
//      3'b100 (XBT.NEG)       : 0-gate wire-swapped negation
// -----------------------------------------------------------------------------
module riscv_xbt_alu (
    input  logic [6:0]         opcode,
    input  logic [2:0]         funct3,
    input  logic [6:0]         funct7,
    input  logic [63:0]        rs1_p,
    input  logic [63:0]        rs1_n,
    input  logic [63:0]        rs2_p,
    input  logic [63:0]        rs2_n,
    input  logic [63:0]        rs1_scalar,
    output logic [63:0]        rd_p,
    output logic [63:0]        rd_n,
    output logic [63:0]        rd_aux_p,
    output logic [63:0]        rd_aux_n,
    output logic signed [63:0] rd_scalar,
    output logic               valid_out
);
    localparam logic [6:0] OPCODE_CUSTOM0 = 7'b0001011;

    logic [63:0] btha_sum_p, btha_sum_n, btha_carry_p, btha_carry_n;
    logic [63:0] ks_sum_p, ks_sum_n;
    logic        ks_cout_p, ks_cout_n;
    logic signed [63:0] dot64_res;
    logic [63:0] unpack_p, unpack_n;

    xbt_btha64 u_btha (
        .a_p     (rs1_p),
        .a_n     (rs1_n),
        .b_p     (rs2_p),
        .b_n     (rs2_n),
        .sum_p   (btha_sum_p),
        .sum_n   (btha_sum_n),
        .carry_p (btha_carry_p),
        .carry_n (btha_carry_n)
    );

    xbt_kogge_stone_adder64 u_ksadd (
        .a_p    (rs1_p),
        .a_n    (rs1_n),
        .b_p    (rs2_p),
        .b_n    (rs2_n),
        .sum_p  (ks_sum_p),
        .sum_n  (ks_sum_n),
        .cout_p (ks_cout_p),
        .cout_n (ks_cout_n)
    );

    xbt_planar_dot64 u_dot64 (
        .a_p     (rs1_p),
        .a_n     (rs1_n),
        .b_p     (rs2_p),
        .b_n     (rs2_n),
        .dot_out (dot64_res)
    );

    xbt_base243_decoder u_unpack243 (
        .packed_byte (rs1_scalar[7:0]),
        .out_p       (unpack_p),
        .out_n       (unpack_n)
    );

    always_comb begin
        rd_p      = 64'd0;
        rd_n      = 64'd0;
        rd_aux_p  = 64'd0;
        rd_aux_n  = 64'd0;
        rd_scalar = 64'sd0;
        valid_out = (opcode == OPCODE_CUSTOM0) && (funct7 == 7'd0);

        if (opcode == OPCODE_CUSTOM0) begin
            unique case (funct3)
                3'b000: begin // XBT.BTHA
                    rd_p     = btha_sum_p;
                    rd_n     = btha_sum_n;
                    rd_aux_p = btha_carry_p;
                    rd_aux_n = btha_carry_n;
                end
                3'b001: begin // XBT.KSADD
                    rd_p     = ks_sum_p;
                    rd_n     = ks_sum_n;
                    rd_aux_p = {63'd0, ks_cout_p};
                    rd_aux_n = {63'd0, ks_cout_n};
                end
                3'b010: begin // XBT.DOT64
                    rd_scalar = dot64_res;
                end
                3'b011: begin // XBT.UNPACK243
                    rd_p = unpack_p;
                    rd_n = unpack_n;
                end
                3'b100: begin // XBT.NEG (0-gate wire swap)
                    rd_p = rs1_n;
                    rd_n = rs1_p;
                end
                default: begin
                    valid_out = 1'b0;
                end
            endcase
        end
    end
endmodule

`default_nettype wire
