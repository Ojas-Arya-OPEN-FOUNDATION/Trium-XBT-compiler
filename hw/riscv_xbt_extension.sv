// =============================================================================
// File: hw/riscv_xbt_extension.sv
// Description: Synthesizable IEEE 1800-2017 SystemVerilog RTL for the RISC-V
//              "Xbt" (Balanced Ternary) Custom-0 Hardware Extension (0x0B).
//
// Modules:
//   1. xbt_btha64                 - 64-Trit Dual-Rail Balanced Ternary Half-Adder
//   2. xbt_kogge_stone_adder64    - 6-Stage Parallel-Prefix Kogge-Stone Ternary CLA
//   2b. xbt_kogge_stone_adder64_v2- Upgraded Monotone Lattice Kogge-Stone Adder
//   3. xbt_planar_dot64           - 64-Trit Dual-Rail Dot Product Unit (Popcount Tree)
//   3b. xbt_aoi22_planar_dot64    - CMOS Compound AOI22 Planar Dot Product Unit
//   3c. xbt_regfile32x128         - 32x128-bit Dual-Rail Ternary Register File
//   3d. xbt_systolic_btfa_unit    - Pipelined Systolic BTFA Compressor Unit
//   4. xbt_base243_decoder        - Division-Free Base-243 Hardware Unpacker
//   5. riscv_xbt_alu              - Top-Level RISC-V RoCC / Custom-0 Execution Unit (v1 + v2)
// =============================================================================

`timescale 1ns/1ps
`default_nettype none
/* verilator lint_off MULTITOP */

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

    // Level 0 initialization
    assign yp_p[0] = (a_p | b_p) & ~(a_n | b_n);
    assign yp_n[0] = 64'd0;
    assign y0_p[0] = c1_p;
    assign y0_n[0] = c1_n;
    assign ym_p[0] = 64'd0;
    assign ym_n[0] = (a_n | b_n) & ~(a_p | b_p);

    // 6 Kogge-Stone parallel-prefix doubling stages: d = 1, 2, 4, 8, 16, 32
    genvar s;
    generate
        for (s = 0; s < 6; s = s + 1) begin : gen_ks_stage
            localparam int D = 1 << s;
            localparam logic [63:0] LOW_MASK = (64'd1 << D) - 64'd1;

            wire [63:0] lp_p = (yp_p[s] << D) | LOW_MASK;
            wire [63:0] lp_n = (yp_n[s] << D);
            wire [63:0] l0_p = (y0_p[s] << D);
            wire [63:0] l0_n = (y0_n[s] << D);
            wire [63:0] lm_p = (ym_p[s] << D);
            wire [63:0] lm_n = (ym_n[s] << D) | LOW_MASK;

            wire [63:0] zp = ~(lp_p | lp_n);
            wire [63:0] z0 = ~(l0_p | l0_n);
            wire [63:0] zm = ~(lm_p | lm_n);

            // Compose H o L for all 64 lanes in parallel
            assign yp_p[s+1] = (lp_p & yp_p[s]) | (zp & y0_p[s]) | (lp_n & ym_p[s]);
            assign yp_n[s+1] = (lp_p & yp_n[s]) | (zp & y0_n[s]) | (lp_n & ym_n[s]);

            assign y0_p[s+1] = (l0_p & yp_p[s]) | (z0 & y0_p[s]) | (l0_n & ym_p[s]);
            assign y0_n[s+1] = (l0_p & yp_n[s]) | (z0 & y0_n[s]) | (l0_n & ym_n[s]);

            assign ym_p[s+1] = (lm_p & yp_p[s]) | (zm & y0_p[s]) | (lm_n & ym_p[s]);
            assign ym_n[s+1] = (lm_p & yp_n[s]) | (zm & y0_n[s]) | (lm_n & ym_n[s]);
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
// 2b. Upgraded 64-Trit Kogge-Stone Parallel-Prefix Adder (v2)
//     35% fewer gates via Monotone Lattice Absorption + Hardware Carry-In (cin_p, cin_n)
// -----------------------------------------------------------------------------
module xbt_kogge_stone_adder64_v2 (
    input  logic [63:0] a_p,
    input  logic [63:0] a_n,
    input  logic [63:0] b_p,
    input  logic [63:0] b_n,
    input  logic        cin_p,
    input  logic        cin_n,
    output logic [63:0] sum_p,
    output logic [63:0] sum_n,
    output logic        cout_p,
    output logic        cout_n
);
    logic [63:0] d_p, d_n, c1_p, c1_n;
    xbt_btha64 u_btha_init (
        .a_p(a_p), .a_n(a_n), .b_p(b_p), .b_n(b_n),
        .sum_p(d_p), .sum_n(d_n), .carry_p(c1_p), .carry_n(c1_n)
    );

    logic [63:0] fm_p [0:5], fm_n [0:5];
    logic [63:0] f0_p [0:5], f0_n [0:5];
    logic [63:0] fp_p [0:5], fp_n [0:5];
    logic [63:0] s1_p, s1_n;

    // Fold cin into bit 0 of Stage 0
    logic bit0_sp, bit0_sn, bit0_cp, bit0_cn;
    always_comb begin
        bit0_cp = d_p[0] & cin_p;
        bit0_cn = d_n[0] & cin_n;
        bit0_sp = ((d_p[0] ^ cin_p) & ~(d_n[0] | cin_n)) | bit0_cn;
        bit0_sn = ((d_n[0] ^ cin_n) & ~(d_p[0] | cin_p)) | bit0_cp;

        s1_p = {d_p[63:1], bit0_sp};
        s1_n = {d_n[63:1], bit0_sn};
    end

    assign f0_p[0] = {c1_p[63:1], (c1_p[0] | bit0_cp) & ~(c1_n[0] | bit0_cn)};
    assign f0_n[0] = {c1_n[63:1], (c1_n[0] | bit0_cn) & ~(c1_p[0] | bit0_cp)};
    assign fp_p[0] = (a_p | b_p) & ~(a_n | b_n);
    assign fp_n[0] = 64'd0;
    assign fm_p[0] = 64'd0;
    assign fm_n[0] = (a_n | b_n) & ~(a_p | b_p);

    // Stages 0..4 (shifts 1, 2, 4, 8, 16): Monotone Lattice Absorption (No LOW_MASK needed!)
    genvar s;
    generate
        for (s = 0; s < 5; s = s + 1) begin : gen_ks_fast
            localparam int SHIFT = 1 << s;
            wire [63:0] l_fm_p = fm_p[s] << SHIFT;
            wire [63:0] l_fm_n = fm_n[s] << SHIFT;
            wire [63:0] l_f0_p = f0_p[s] << SHIFT;
            wire [63:0] l_f0_n = f0_n[s] << SHIFT;
            wire [63:0] l_fp_p = fp_p[s] << SHIFT;
            wire [63:0] l_fp_n = fp_n[s] << SHIFT;

            assign fm_p[s+1] = fm_p[s] | (f0_p[s] & ~l_fm_n) | (fp_p[s] & l_fm_p);
            assign fm_n[s+1] = fp_n[s] | (f0_n[s] & ~l_fm_p) | (fm_n[s] & l_fm_n);
            assign f0_p[s+1] = fm_p[s] | (f0_p[s] & ~l_f0_n) | (fp_p[s] & l_f0_p);
            assign f0_n[s+1] = fp_n[s] | (f0_n[s] & ~l_f0_p) | (fm_n[s] & l_f0_n);
            assign fp_p[s+1] = fm_p[s] | (f0_p[s] & ~l_fp_n) | (fp_p[s] & l_fp_p);
            assign fp_n[s+1] = fp_n[s] | (f0_n[s] & ~l_fp_p) | (fm_n[s] & l_fp_n);
        end
    endgenerate

    // Stage 5 (shift = 32): Only compute cout rails
    logic [63:0] final_cout_p, final_cout_n, c_in_vec_p, c_in_vec_n;
    always_comb begin
        final_cout_p = fm_p[5] | (f0_p[5] & ~(f0_n[5] << 32)) | (fp_p[5] & (f0_p[5] << 32));
        final_cout_n = fp_n[5] | (f0_n[5] & ~(f0_p[5] << 32)) | (fm_n[5] & (f0_n[5] << 32));
        cout_p = final_cout_p[63];
        cout_n = final_cout_n[63];

        c_in_vec_p = final_cout_p << 1;
        c_in_vec_n = final_cout_n << 1;
        sum_p = ((s1_p ^ c_in_vec_p) & ~(s1_n | c_in_vec_n)) | (s1_n & c_in_vec_n);
        sum_n = ((s1_n ^ c_in_vec_n) & ~(s1_p | c_in_vec_p)) | (s1_p & c_in_vec_p);
    end
endmodule

// -----------------------------------------------------------------------------

// -----------------------------------------------------------------------------
// 2c. 64-Trit Area-Optimized Brent-Kung Parallel-Prefix Adder (Sigma-7 Lattice)
//     Uses the exact same associative monotone carry-transfer semigroup Sigma-7
//     as xbt_kogge_stone_adder64_v2, but with a Brent-Kung up-sweep / down-sweep
//     tree: 120 prefix compose nodes instead of Kogge-Stone's 321 (-62.6% nodes).
// -----------------------------------------------------------------------------
module xbt_brent_kung_adder64 (
    input  logic [63:0] a_p,
    input  logic [63:0] a_n,
    input  logic [63:0] b_p,
    input  logic [63:0] b_n,
    input  logic        cin_p,
    input  logic        cin_n,
    output logic [63:0] sum_p,
    output logic [63:0] sum_n,
    output logic        cout_p,
    output logic        cout_n
);
    logic [63:0] d_p, d_n, c1_p, c1_n;
    xbt_btha64 u_btha_bk (
        .a_p(a_p), .a_n(a_n), .b_p(b_p), .b_n(b_n),
        .sum_p(d_p), .sum_n(d_n), .carry_p(c1_p), .carry_n(c1_n)
    );

    // 12 pipeline-free combinational prefix layers:
    //   Layer 0     : Initial Sigma-7 transfer functions (bit 0 includes cin)
    //   Layers 1..6 : Brent-Kung up-sweep reduction (63 compose nodes)
    //   Layers 7..11: Brent-Kung down-sweep distribution (57 compose nodes)
    //   Total       : 120 compose nodes vs. Kogge-Stone's 321 (-62.6%)
    logic [63:0] fm_p [0:11], fm_n [0:11];
    logic [63:0] f0_p [0:11], f0_n [0:11];
    logic [63:0] fp_p [0:11], fp_n [0:11];
    logic [63:0] s1_p, s1_n;

    logic bit0_sp, bit0_sn, bit0_cp, bit0_cn;
    always_comb begin
        bit0_cp = d_p[0] & cin_p;
        bit0_cn = d_n[0] & cin_n;
        bit0_sp = ((d_p[0] ^ cin_p) & ~(d_n[0] | cin_n)) | bit0_cn;
        bit0_sn = ((d_n[0] ^ cin_n) & ~(d_p[0] | cin_p)) | bit0_cp;

        s1_p = {d_p[63:1], bit0_sp};
        s1_n = {d_n[63:1], bit0_sn};
    end

    assign f0_p[0] = {c1_p[63:1], (c1_p[0] | bit0_cp) & ~(c1_n[0] | bit0_cn)};
    assign f0_n[0] = {c1_n[63:1], (c1_n[0] | bit0_cn) & ~(c1_p[0] | bit0_cp)};
    assign fp_p[0] = (a_p | b_p) & ~(a_n | b_n);
    assign fp_n[0] = 64'd0;
    assign fm_p[0] = 64'd0;
    assign fm_n[0] = (a_n | b_n) & ~(a_p | b_p);

    genvar k, i;
    generate
        // Up-sweep reduction tree (k = 0..5 -> layers 1..6)
        for (k = 0; k < 6; k = k + 1) begin : gen_bk_up
            localparam int STEP = 1 << (k + 1);
            localparam int HALF = 1 << k;
            for (i = 0; i < 64; i = i + 1) begin : gen_up_bit
                if (((i + 1) % STEP) == 0) begin : gen_compose
                    assign fm_p[k+1][i] = fm_p[k][i] | (f0_p[k][i] & ~fm_n[k][i-HALF]) | (fp_p[k][i] & fm_p[k][i-HALF]);
                    assign fm_n[k+1][i] = fp_n[k][i] | (f0_n[k][i] & ~fm_p[k][i-HALF]) | (fm_n[k][i] & fm_n[k][i-HALF]);
                    assign f0_p[k+1][i] = fm_p[k][i] | (f0_p[k][i] & ~f0_n[k][i-HALF]) | (fp_p[k][i] & f0_p[k][i-HALF]);
                    assign f0_n[k+1][i] = fp_n[k][i] | (f0_n[k][i] & ~f0_p[k][i-HALF]) | (fm_n[k][i] & f0_n[k][i-HALF]);
                    assign fp_p[k+1][i] = fm_p[k][i] | (f0_p[k][i] & ~fp_n[k][i-HALF]) | (fp_p[k][i] & fp_p[k][i-HALF]);
                    assign fp_n[k+1][i] = fp_n[k][i] | (f0_n[k][i] & ~fp_p[k][i-HALF]) | (fm_n[k][i] & fp_n[k][i-HALF]);
                end else begin : gen_pass
                    assign fm_p[k+1][i] = fm_p[k][i];
                    assign fm_n[k+1][i] = fm_n[k][i];
                    assign f0_p[k+1][i] = f0_p[k][i];
                    assign f0_n[k+1][i] = f0_n[k][i];
                    assign fp_p[k+1][i] = fp_p[k][i];
                    assign fp_n[k+1][i] = fp_n[k][i];
                end
            end
        end

        // Down-sweep distribution tree (d = 0..4, k = 4..0 -> layers 7..11)
        for (k = 0; k < 5; k = k + 1) begin : gen_bk_dn
            localparam int LEVEL = 4 - k;
            localparam int STEP  = 1 << (LEVEL + 1);
            localparam int HALF  = 1 << LEVEL;
            localparam int SRC   = 6 + k;
            localparam int DST   = 7 + k;
            for (i = 0; i < 64; i = i + 1) begin : gen_dn_bit
                if ((i >= (STEP + HALF - 1)) && (((i + 1 - HALF) % STEP) == 0)) begin : gen_compose
                    assign fm_p[DST][i] = fm_p[SRC][i] | (f0_p[SRC][i] & ~fm_n[SRC][i-HALF]) | (fp_p[SRC][i] & fm_p[SRC][i-HALF]);
                    assign fm_n[DST][i] = fp_n[SRC][i] | (f0_n[SRC][i] & ~fm_p[SRC][i-HALF]) | (fm_n[SRC][i] & fm_n[SRC][i-HALF]);
                    assign f0_p[DST][i] = fm_p[SRC][i] | (f0_p[SRC][i] & ~f0_n[SRC][i-HALF]) | (fp_p[SRC][i] & f0_p[SRC][i-HALF]);
                    assign f0_n[DST][i] = fp_n[SRC][i] | (f0_n[SRC][i] & ~f0_p[SRC][i-HALF]) | (fm_n[SRC][i] & f0_n[SRC][i-HALF]);
                    assign fp_p[DST][i] = fm_p[SRC][i] | (f0_p[SRC][i] & ~fp_n[SRC][i-HALF]) | (fp_p[SRC][i] & fp_p[SRC][i-HALF]);
                    assign fp_n[DST][i] = fp_n[SRC][i] | (f0_n[SRC][i] & ~fp_p[SRC][i-HALF]) | (fm_n[SRC][i] & fp_n[SRC][i-HALF]);
                end else begin : gen_pass
                    assign fm_p[DST][i] = fm_p[SRC][i];
                    assign fm_n[DST][i] = fm_n[SRC][i];
                    assign f0_p[DST][i] = f0_p[SRC][i];
                    assign f0_n[DST][i] = f0_n[SRC][i];
                    assign fp_p[DST][i] = fp_p[SRC][i];
                    assign fp_n[DST][i] = fp_n[SRC][i];
                end
            end
        end
    endgenerate

    logic [63:0] c_in_vec_p, c_in_vec_n;
    always_comb begin
        cout_p = f0_p[11][63];
        cout_n = f0_n[11][63];

        c_in_vec_p = f0_p[11] << 1;
        c_in_vec_n = f0_n[11] << 1;
        sum_p = ((s1_p ^ c_in_vec_p) & ~(s1_n | c_in_vec_n)) | (s1_n & c_in_vec_n);
        sum_n = ((s1_n ^ c_in_vec_n) & ~(s1_p | c_in_vec_p)) | (s1_p & c_in_vec_p);
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

        dot_out = $signed({57'd0, cnt_p}) - $signed({57'd0, cnt_n});
    end
endmodule

// -----------------------------------------------------------------------------
// 3b. 64-Trit Dual-Rail CMOS Compound AOI22 Planar Dot Product Unit
//     Computes product rails using CMOS compound AOI22 gates:
//       ~P_prod = AOI22(A_P, B_P, A_N, B_N) = ~((A_P & B_P) | (A_N & B_N))
//       ~N_prod = AOI22(A_P, B_N, A_N, B_P) = ~((A_P & B_N) | (A_N & B_P))
//     followed by parallel popcount tree reduction in 1 clock cycle.
// -----------------------------------------------------------------------------
module xbt_aoi22_planar_dot64 (
    input  logic [63:0]        a_p,
    input  logic [63:0]        a_n,
    input  logic [63:0]        b_p,
    input  logic [63:0]        b_n,
    output logic [63:0]        prod_p_bar,
    output logic [63:0]        prod_n_bar,
    output logic [63:0]        prod_p,
    output logic [63:0]        prod_n,
    output logic signed [63:0] dot_out
);
    logic [6:0] cnt_p, cnt_n;

    always_comb begin
        // Compound AOI22 gates
        prod_p_bar = ~((a_p & b_p) | (a_n & b_n));
        prod_n_bar = ~((a_p & b_n) | (a_n & b_p));

        // Active-high dual rails
        prod_p = ~prod_p_bar;
        prod_n = ~prod_n_bar;

        cnt_p = 7'd0;
        cnt_n = 7'd0;
        for (int i = 0; i < 64; i++) begin
            cnt_p = cnt_p + {6'd0, prod_p[i]};
            cnt_n = cnt_n + {6'd0, prod_n[i]};
        end

        dot_out = $signed({57'd0, cnt_p}) - $signed({57'd0, cnt_n});
    end
endmodule

// -----------------------------------------------------------------------------
// 3c. 32 x 128-bit Dual-Rail Ternary Register File (tr0 .. tr31)
//     Each register stores 64 trits as dual rails (64 P bits + 64 N bits = 128 bits).
//     Features 0-gate wire-swap negation bypass on read ports.
//     Register tr0 is hardwired to 0 (P=0, N=0).
// -----------------------------------------------------------------------------
module xbt_regfile32x128 (
    input  logic        clk,
    input  logic        rst_n,
    // Read Port 1
    input  logic [4:0]  rs1_addr,
    input  logic        rs1_neg,      // Wire-swap negation bypass control
    output logic [63:0] rs1_p,
    output logic [63:0] rs1_n,
    // Read Port 2
    input  logic [4:0]  rs2_addr,
    input  logic        rs2_neg,      // Wire-swap negation bypass control
    output logic [63:0] rs2_p,
    output logic [63:0] rs2_n,
    // Write Port
    input  logic        we,
    input  logic [4:0]  rd_addr,
    input  logic [63:0] rd_p,
    input  logic [63:0] rd_n
);
    logic [63:0] mem_p [0:31];
    logic [63:0] mem_n [0:31];
    logic [63:0] raw1_p, raw1_n;
    logic [63:0] raw2_p, raw2_n;

    // Asynchronous read ports with 0-gate wire-swap negation bypass
    always_comb begin
        raw1_p = (rs1_addr == 5'd0) ? 64'd0 : mem_p[rs1_addr];
        raw1_n = (rs1_addr == 5'd0) ? 64'd0 : mem_n[rs1_addr];

        raw2_p = (rs2_addr == 5'd0) ? 64'd0 : mem_p[rs2_addr];
        raw2_n = (rs2_addr == 5'd0) ? 64'd0 : mem_n[rs2_addr];

        rs1_p = rs1_neg ? raw1_n : raw1_p;
        rs1_n = rs1_neg ? raw1_p : raw1_n;

        rs2_p = rs2_neg ? raw2_n : raw2_p;
        rs2_n = rs2_neg ? raw2_p : raw2_n;
    end

    // Synchronous write port
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int r = 0; r < 32; r++) begin
                mem_p[r] <= 64'd0;
                mem_n[r] <= 64'd0;
            end
        end else if (we && (rd_addr != 5'd0)) begin
            mem_p[rd_addr] <= rd_p;
            mem_n[rd_addr] <= rd_n;
        end
    end
endmodule

// -----------------------------------------------------------------------------
// 3d. Pipelined Systolic BTFA Compressor Unit
//     Streaming matrix operation unit with pipelined BTFA compressor stages.
//     Each stage compresses 3 dual-rail inputs (A, B, C) into Sum and Carry,
//     streaming intermediate matrix accumulators with zero carry stalls.
// -----------------------------------------------------------------------------
module xbt_systolic_btfa_unit #(
    parameter int STAGES = 2
) (
    input  logic        clk,
    input  logic        rst_n,
    input  logic        valid_in,
    input  logic [63:0] a_p,
    input  logic [63:0] a_n,
    input  logic [63:0] b_p,
    input  logic [63:0] b_n,
    input  logic [63:0] c_p,
    input  logic [63:0] c_n,
    output logic [63:0] sum_p,
    output logic [63:0] sum_n,
    output logic [63:0] carry_p,
    output logic [63:0] carry_n,
    output logic        valid_out
);
    // Stage 1 combinational BTFA
    logic [63:0] s1_p, s1_n, c1_p, c1_n;
    logic [63:0] s2_p, s2_n, c2_p, c2_n;
    logic [63:0] btfa_sum_p, btfa_sum_n;
    logic [63:0] btfa_carry_p, btfa_carry_n;

    xbt_btha64 u_btha1 (
        .a_p     (a_p),
        .a_n     (a_n),
        .b_p     (b_p),
        .b_n     (b_n),
        .sum_p   (s1_p),
        .sum_n   (s1_n),
        .carry_p (c1_p),
        .carry_n (c1_n)
    );

    xbt_btha64 u_btha2 (
        .a_p     (s1_p),
        .a_n     (s1_n),
        .b_p     (c_p),
        .b_n     (c_n),
        .sum_p   (s2_p),
        .sum_n   (s2_n),
        .carry_p (c2_p),
        .carry_n (c2_n)
    );

    always_comb begin
        btfa_sum_p   = s2_p;
        btfa_sum_n   = s2_n;
        btfa_carry_p = (c1_p | c2_p) & ~(c1_n | c2_n);
        btfa_carry_n = (c1_n | c2_n) & ~(c1_p | c2_p);
    end

    // Pipeline registers
    logic [63:0] pipe_sum_p   [0:STAGES-1];
    logic [63:0] pipe_sum_n   [0:STAGES-1];
    logic [63:0] pipe_carry_p [0:STAGES-1];
    logic [63:0] pipe_carry_n [0:STAGES-1];
    logic        pipe_valid   [0:STAGES-1];

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int s = 0; s < STAGES; s++) begin
                pipe_sum_p[s]   <= 64'd0;
                pipe_sum_n[s]   <= 64'd0;
                pipe_carry_p[s] <= 64'd0;
                pipe_carry_n[s] <= 64'd0;
                pipe_valid[s]   <= 1'b0;
            end
        end else begin
            pipe_sum_p[0]   <= btfa_sum_p;
            pipe_sum_n[0]   <= btfa_sum_n;
            pipe_carry_p[0] <= btfa_carry_p;
            pipe_carry_n[0] <= btfa_carry_n;
            pipe_valid[0]   <= valid_in;

            for (int s = 1; s < STAGES; s++) begin
                pipe_sum_p[s]   <= pipe_sum_p[s-1];
                pipe_sum_n[s]   <= pipe_sum_n[s-1];
                pipe_carry_p[s] <= pipe_carry_p[s-1];
                pipe_carry_n[s] <= pipe_carry_n[s-1];
                pipe_valid[s]   <= pipe_valid[s-1];
            end
        end
    end

    assign sum_p     = pipe_sum_p[STAGES-1];
    assign sum_n     = pipe_sum_n[STAGES-1];
    assign carry_p   = pipe_carry_p[STAGES-1];
    assign carry_n   = pipe_carry_n[STAGES-1];
    assign valid_out = pipe_valid[STAGES-1];
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
//    funct3 dispatch (v1 + v2):
//      3'b000 (XBT.BTHA)      : 1-cycle 64-trit BTHA step
//      3'b001 (XBT.KSADD)     : 1-cycle 64-trit Kogge-Stone addition
//      3'b010 (XBT.DOT64)     : 1-cycle 64-trit Planar Dot Product -> rd_scalar
//      3'b011 (XBT.UNPACK243) : 1-cycle Base-243 Byte -> Dual-Rail unpack
//      3'b100 (XBT.NEG)       : 0-gate wire-swapped negation
//      3'b101 (XBT.CSAMAC)    : 1-cycle Fused Carry-Save MAC: rd_in + (rs1 * rs2)
//      3'b110 (XBT.GROUP2)    : v2 R-type extended group (dispatched by funct7):
//        7'h00 (BTFA)         : rd = sum(rs1, rs2, rd_in); carry -> rd_aux
//        7'h01 (MUL)          : per-lane AOI22 product rd[i] = rs1[i] * rs2[i]
//        7'h02 (PACK243)      : rd_scalar = base-243 byte from rs1[4:0]
//        7'h03 (SIGN)         : rd = sign(rs1) in {-1, 0, +1} (priority encoder)
//        7'h04 (CMOVZ)        : if sign(rs1) == 0 then rd = rs2 else rd_in
//        7'h05 (CMOVP)        : if sign(rs1) >  0 then rd = rs2 else rd_in
//        7'h06 (CMOVN)        : if sign(rs1) <  0 then rd = rs2 else rd_in
//        7'h07 (SHL3)         : rd = rs1 << imm12[4:0] (valid_out=0 on overflow)
//      3'b111 (XBT.LI5)       : I-type 5-trit constant at chunk offset 5*imm12[10:8]
// -----------------------------------------------------------------------------
module riscv_xbt_alu #(
    parameter bit USE_BRENT_KUNG = 1'b0
) (
    input  logic [6:0]         opcode,
    input  logic [2:0]         funct3,
    input  logic [6:0]         funct7,
    input  logic [63:0]        rs1_p,
    input  logic [63:0]        rs1_n,
    input  logic [63:0]        rs2_p,
    input  logic [63:0]        rs2_n,
    input  logic [63:0]        rd_p_in,
    input  logic [63:0]        rd_n_in,
    input  logic [63:0]        rs1_scalar,
    input  logic [11:0]        imm12,
    output logic [63:0]        rd_p,
    output logic [63:0]        rd_n,
    output logic [63:0]        rd_aux_p,
    output logic [63:0]        rd_aux_n,
    output logic signed [63:0] rd_scalar,
    output logic               valid_out
);
    localparam logic [6:0] OPCODE_CUSTOM0 = 7'b0001011;

    logic [63:0] btha_sum_p, btha_sum_n, btha_carry_p, btha_carry_n;
    logic [63:0] btfa_sum_p, btfa_sum_n, btfa_c2_p, btfa_c2_n;
    logic [63:0] btfa_carry_p, btfa_carry_n;
    logic [63:0] ks_sum_p, ks_sum_n;
    logic        ks_cout_p, ks_cout_n;
    logic signed [63:0] dot64_res;
    logic [7:0]  unpack_byte_in;
    logic [63:0] unpack_p, unpack_n;

    // Shared combinational helpers for v1/v2 ops
    logic [63:0] prod_p, prod_n;
    logic        sign_p, sign_n;
    logic signed [63:0] pack243_val;
    logic [4:0]  shl3_shamt;
    logic [63:0] shl3_keep;
    logic        shl3_ovf;
    logic [5:0]  li5_shamt;
    wire         _unused_bits = &{1'b0, rs1_scalar[63:8], imm12[11], 1'b0};

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

    xbt_btha64 u_btfa_stage2 (
        .a_p     (btha_sum_p),
        .a_n     (btha_sum_n),
        .b_p     (rd_p_in),
        .b_n     (rd_n_in),
        .sum_p   (btfa_sum_p),
        .sum_n   (btfa_sum_n),
        .carry_p (btfa_c2_p),
        .carry_n (btfa_c2_n)
    );

    generate
        if (USE_BRENT_KUNG) begin : gen_adder_bk
            xbt_brent_kung_adder64 u_ksadd (
                .a_p    (rs1_p),
                .a_n    (rs1_n),
                .b_p    (rs2_p),
                .b_n    (rs2_n),
                .cin_p  (1'b0),
                .cin_n  (1'b0),
                .sum_p  (ks_sum_p),
                .sum_n  (ks_sum_n),
                .cout_p (ks_cout_p),
                .cout_n (ks_cout_n)
            );
        end else begin : gen_adder_ks
            xbt_kogge_stone_adder64_v2 u_ksadd (
                .a_p    (rs1_p),
                .a_n    (rs1_n),
                .b_p    (rs2_p),
                .b_n    (rs2_n),
                .cin_p  (1'b0),
                .cin_n  (1'b0),
                .sum_p  (ks_sum_p),
                .sum_n  (ks_sum_n),
                .cout_p (ks_cout_p),
                .cout_n (ks_cout_n)
            );
        end
    endgenerate

    xbt_planar_dot64 u_dot64 (
        .a_p     (rs1_p),
        .a_n     (rs1_n),
        .b_p     (rs2_p),
        .b_n     (rs2_n),
        .dot_out (dot64_res)
    );

    assign unpack_byte_in = (funct3 == 3'b111) ? imm12[7:0] : rs1_scalar[7:0];

    xbt_base243_decoder u_unpack243 (
        .packed_byte (unpack_byte_in),
        .out_p       (unpack_p),
        .out_n       (unpack_n)
    );

    always_comb begin
        // BTFA carry rail combination
        btfa_carry_p = (btha_carry_p | btfa_c2_p) & ~(btha_carry_n | btfa_c2_n);
        btfa_carry_n = (btha_carry_n | btfa_c2_n) & ~(btha_carry_p | btfa_c2_p);

        // Per-lane AOI22 product (shared by CSAMAC and GROUP2.MUL)
        prod_p = (rs1_p & rs2_p) | (rs1_n & rs2_n);
        prod_n = (rs1_p & rs2_n) | (rs1_n & rs2_p);

        // Priority encoder for sign(rs1) (ascending scan: MSB non-zero trit wins)
        sign_p = 1'b0;
        sign_n = 1'b0;
        for (int i = 0; i < 64; i++) begin
            if (rs1_p[i]) begin
                sign_p = 1'b1;
                sign_n = 1'b0;
            end else if (rs1_n[i]) begin
                sign_p = 1'b0;
                sign_n = 1'b1;
            end
        end

        // Base-243 pack of rs1[4:0]: 121 + sum_{k=0..4} trit(k) * 3^k
        pack243_val = 64'sd121
            + (rs1_p[0] ? 64'sd1  : (rs1_n[0] ? -64'sd1  : 64'sd0))
            + (rs1_p[1] ? 64'sd3  : (rs1_n[1] ? -64'sd3  : 64'sd0))
            + (rs1_p[2] ? 64'sd9  : (rs1_n[2] ? -64'sd9  : 64'sd0))
            + (rs1_p[3] ? 64'sd27 : (rs1_n[3] ? -64'sd27 : 64'sd0))
            + (rs1_p[4] ? 64'sd81 : (rs1_n[4] ? -64'sd81 : 64'sd0));

        // SHL3 shift amount (from rs2 field = imm12[4:0]) & overflow check
        shl3_shamt = imm12[4:0];
        shl3_keep  = 64'hFFFF_FFFF_FFFF_FFFF >> shl3_shamt;
        shl3_ovf   = (shl3_shamt != 5'd0) &&
                     (((rs1_p & ~shl3_keep) != 64'd0) || ((rs1_n & ~shl3_keep) != 64'd0));

        // LI5 trit shift = 5 * imm12[10:8] (0..35)
        li5_shamt  = {1'b0, imm12[10:8], 2'b00} + {3'b000, imm12[10:8]};

        // Output dispatch
        rd_p      = 64'd0;
        rd_n      = 64'd0;
        rd_aux_p  = 64'd0;
        rd_aux_n  = 64'd0;
        rd_scalar = 64'sd0;
        valid_out = (opcode == OPCODE_CUSTOM0);

        if (opcode == OPCODE_CUSTOM0) begin
            case (funct3)
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
                3'b101: begin // XBT.CSAMAC: rd_in + (rs1 * rs2)
                    rd_p     = ((prod_p ^ rd_p_in) & ~(prod_n | rd_n_in)) | (prod_n & rd_n_in);
                    rd_n     = ((prod_n ^ rd_n_in) & ~(prod_p | rd_p_in)) | (prod_p & rd_p_in);
                    rd_aux_p = prod_p & rd_p_in;
                    rd_aux_n = prod_n & rd_n_in;
                end
                3'b110: begin // XBT.GROUP2 (v2 extended operations)
                    case (funct7)
                        7'h00: begin // XBT.BTFA: sum(rs1, rs2, rd_in)
                            rd_p     = btfa_sum_p;
                            rd_n     = btfa_sum_n;
                            rd_aux_p = btfa_carry_p;
                            rd_aux_n = btfa_carry_n;
                        end
                        7'h01: begin // XBT.MUL: per-lane AOI22 product
                            rd_p = prod_p;
                            rd_n = prod_n;
                        end
                        7'h02: begin // XBT.PACK243: rs1[4:0] -> base-243 byte in rd_scalar
                            rd_scalar = pack243_val;
                        end
                        7'h03: begin // XBT.SIGN: priority encoder -> {-1, 0, +1}
                            rd_p = {63'd0, sign_p};
                            rd_n = {63'd0, sign_n};
                        end
                        7'h04: begin // XBT.CMOVZ: if sign(rs1) == 0 then rs2 else rd_in
                            if (!sign_p && !sign_n) begin
                                rd_p = rs2_p;
                                rd_n = rs2_n;
                            end else begin
                                rd_p = rd_p_in;
                                rd_n = rd_n_in;
                            end
                        end
                        7'h05: begin // XBT.CMOVP: if sign(rs1) > 0 then rs2 else rd_in
                            if (sign_p) begin
                                rd_p = rs2_p;
                                rd_n = rs2_n;
                            end else begin
                                rd_p = rd_p_in;
                                rd_n = rd_n_in;
                            end
                        end
                        7'h06: begin // XBT.CMOVN: if sign(rs1) < 0 then rs2 else rd_in
                            if (sign_n) begin
                                rd_p = rs2_p;
                                rd_n = rs2_n;
                            end else begin
                                rd_p = rd_p_in;
                                rd_n = rd_n_in;
                            end
                        end
                        7'h07: begin // XBT.SHL3: shift left by imm12[4:0] trits; refuse overflow
                            if (shl3_ovf) begin
                                valid_out = 1'b0;
                                rd_p      = 64'd0;
                                rd_n      = 64'd0;
                            end else begin
                                rd_p = rs1_p << shl3_shamt;
                                rd_n = rs1_n << shl3_shamt;
                            end
                        end
                        default: begin
                            valid_out = 1'b0;
                        end
                    endcase
                end
                3'b111: begin // XBT.LI5: (imm12[7:0] - 121) * 3^(5 * imm12[10:8])
                    rd_p = unpack_p << li5_shamt;
                    rd_n = unpack_n << li5_shamt;
                end
                default: begin
                    valid_out = 1'b0;
                end
            endcase
        end
    end
endmodule

`default_nettype wire
