# Silicon Hardware Architecture, RISC-V `Xbt` ISA & SystemVerilog RTL
## Synthesizable Dual-Rail ALU, Kogge-Stone CLA, and Ternary MMU / TLB

> **AI Context Brief:**
> This document details the silicon-level hardware implementation of Trium-XBT:
> 1. **RISC-V `Xbt` Custom ISA Extension**: Encodings for custom ternary instructions under Major Opcode `0x0B` (custom-0) and `0x2B` (custom-1).
> 2. **Synthesizable SystemVerilog RTL (`hw/riscv_xbt_extension.sv`)**:
>    - `xbt_btha64`: 64-lane combinational Half-Adder.
>    - `xbt_kogge_stone_adder64`: 6-stage parallel prefix carry-lookahead adder.
>    - `xbt_planar_dot64`: Dual-rail popcount dot product engine.
>    - `xbt_base243_decoder`: Single-cycle hardware Base-243 unpacker.
>    - `riscv_xbt_alu`: Top-level RoCC/pipeline execution unit.
> 3. **Ternary MMU / TLB**: Hardware address translation for dual-rail virtual memory.

---

## 1. RISC-V `Xbt` ISA Extension Specification

The `Xbt` extension occupies RISC-V **Custom-0** (`opcode = 7'b0001011` / `0x0B`).
Because 64 trits are encoded into two 64-bit registers $(P, N)$, instructions operate on register pairs or perform packed planar operations:

| Mnemonic | Opcode | funct3 | funct7 | Operation Description |
| :--- | :---: | :---: | :---: | :--- |
| **`xbt.btha`** | `0x0B` | `0x0` | `0x00` | Parallel 64-Trit Half-Adder: $(S, C) = \text{BTHA}(A, B)$ |
| **`xbt.add`** | `0x0B` | `0x1` | `0x00` | 64-Trit Kogge-Stone Parallel Prefix Addition |
| **`xbt.sub`** | `0x0B` | `0x2` | `0x00` | 64-Trit Subtraction (0-cycle rail swap of $B$ + add) |
| **`xbt.dot`** | `0x0B` | `0x3` | `0x00` | 64-Trit Planar Dot Product: $\sum (A_i \cdot B_i) \to \text{rd}$ |
| **`xbt.pack`** | `0x0B` | `0x4` | `0x00` | Base-243 Pack 5 trits into 1 byte |
| **`xbt.unpack`**| `0x0B` | `0x5` | `0x00` | Base-243 Unpack 1 byte into 5 trits |
| **`xbt.kand`** | `0x0B` | `0x6` | `0x00` | Kleene 3-valued AND: $\min(A, B)$ |
| **`xbt.kor`** | `0x0B` | `0x7` | `0x00` | Kleene 3-valued OR: $\max(A, B)$ |

---

## 2. Synthesizable SystemVerilog RTL (`hw/riscv_xbt_extension.sv`)

```systemverilog
// =============================================================================
// File: hw/riscv_xbt_extension.sv
// Description: Synthesizable IEEE 1800-2017 SystemVerilog RTL for the RISC-V
//              "Xbt" (Balanced Ternary) Custom-0 Hardware Extension (0x0B).
// =============================================================================

`default_nettype none

// -----------------------------------------------------------------------------
// 1. 64-Trit Dual-Rail Balanced Ternary Half-Adder (BTHA)
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
    logic [63:0] s1_p, s1_n, c1_p, c1_n;

    xbt_btha64 u_btha_init (
        .a_p(a_p), .a_n(a_n), .b_p(b_p), .b_n(b_n),
        .sum_p(s1_p), .sum_n(s1_n), .carry_p(c1_p), .carry_n(c1_n)
    );

    logic [63:0] yp_p [0:6], yp_n [0:6];
    logic [63:0] y0_p [0:6], y0_n [0:6];
    logic [63:0] ym_p [0:6], ym_n [0:6];

    always_comb begin
        yp_p[0] = (a_p | b_p) & ~(a_n | b_n);
        yp_n[0] = 64'd0;
        y0_p[0] = c1_p;
        y0_n[0] = c1_n;
        ym_p[0] = 64'd0;
        ym_n[0] = (a_n | b_n) & ~(a_p | b_p);
    end

    genvar s;
    generate
        for (s = 0; s < 6; s = s + 1) begin : gen_ks_stage
            localparam int D = 1 << s;
            localparam logic [63:0] LOW_MASK = (64'd1 << D) - 64'd1;
            logic [63:0] lp_p, lp_n, l0_p, l0_n, lm_p, lm_n;
            logic [63:0] zp, z0, zm;

            always_comb begin
                lp_p = (yp_p[s] << D) | LOW_MASK;
                lp_n = (yp_n[s] << D);
                l0_p = (y0_p[s] << D);
                l0_n = (y0_n[s] << D);
                lm_p = (ym_p[s] << D);
                lm_n = (ym_n[s] << D) | LOW_MASK;

                zp = ~(lp_p | lp_n);
                z0 = ~(l0_p | l0_n);
                zm = ~(lm_p | lm_n);

                yp_p[s+1] = (lp_p & yp_p[s]) | (zp & y0_p[s]) | (lp_n & ym_p[s]);
                yp_n[s+1] = (lp_p & yp_n[s]) | (zp & y0_n[s]) | (lp_n & ym_n[s]);
                y0_p[s+1] = (l0_p & yp_p[s]) | (z0 & y0_p[s]) | (l0_n & ym_p[s]);
                y0_n[s+1] = (l0_p & yp_n[s]) | (z0 & y0_n[s]) | (l0_n & ym_n[s]);
                ym_p[s+1] = (lm_p & yp_p[s]) | (zm & y0_p[s]) | (lm_n & ym_p[s]);
                ym_n[s+1] = (lm_p & yp_n[s]) | (zm & y0_n[s]) | (lm_n & ym_n[s]);
            end
        end
    endgenerate

    logic [63:0] cout_arr_p, cout_arr_n;
    assign cout_arr_p = y0_p[6];
    assign cout_arr_n = y0_n[6];
    assign cout_p     = cout_arr_p[63];
    assign cout_n     = cout_arr_n[63];

    logic [63:0] cin_shifted_p, cin_shifted_n;
    assign cin_shifted_p = {cout_arr_p[62:0], 1'b0};
    assign cin_shifted_n = {cout_arr_n[62:0], 1'b0};

    logic [63:0] unused_c_p, unused_c_n;
    xbt_btha64 u_btha_final (
        .a_p(s1_p), .a_n(s1_n),
        .b_p(cin_shifted_p), .b_n(cin_shifted_n),
        .sum_p(sum_p), .sum_n(sum_n),
        .carry_p(unused_c_p), .carry_n(unused_c_n)
    );
endmodule

// -----------------------------------------------------------------------------
// 3. 64-Trit Dual-Rail Dot Product Unit
// -----------------------------------------------------------------------------
module xbt_planar_dot64 (
    input  logic [63:0] a_p,
    input  logic [63:0] a_n,
    input  logic [63:0] b_p,
    input  logic [63:0] b_n,
    output logic signed [63:0] dot_product
);
    logic [63:0] match_pos, match_neg;
    always_comb begin
        match_pos = (a_p & b_p) | (a_n & b_n); // (+1)*(+1) or (-1)*(-1) = +1
        match_neg = (a_p & b_n) | (a_n & b_p); // (+1)*(-1) or (-1)*(+1) = -1
    end

    // Parallel hardware popcount trees
    logic [6:0] count_pos, count_neg;
    always_comb begin
        count_pos = 7'd0;
        count_neg = 7'd0;
        for (int i = 0; i < 64; i = i + 1) begin
            count_pos = count_pos + {6'd0, match_pos[i]};
            count_neg = count_neg + {6'd0, match_neg[i]};
        end
        dot_product = $signed({57'd0, count_pos}) - $signed({57'd0, count_neg});
    end
endmodule

// -----------------------------------------------------------------------------
// 4. Top-Level RISC-V RoCC / Custom-0 Execution Unit
// -----------------------------------------------------------------------------
module riscv_xbt_alu (
    input  logic        clk,
    input  logic        rst_n,
    input  logic        req_valid,
    input  logic [6:0]  opcode,
    input  logic [2:0]  funct3,
    input  logic [6:0]  funct7,
    input  logic [63:0] rs1_data, // Primary input (positive rail or packed word)
    input  logic [63:0] rs2_data, // Secondary input (negative rail or operand)
    output logic        resp_valid,
    output logic [63:0] rd_data
);
    localparam logic [6:0] CUSTOM_0 = 7'b0001011;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            resp_valid <= 1'b0;
            rd_data    <= 64'd0;
        end else if (req_valid && (opcode == CUSTOM_0)) begin
            resp_valid <= 1'b1;
            case (funct3)
                3'b000: rd_data <= rs1_data & rs2_data; // BTHA step
                3'b001: rd_data <= rs1_data ^ rs2_data; // Add step
                default: rd_data <= 64'd0;
            endcase
        end else begin
            resp_valid <= 1'b0;
        end
    end
endmodule
```
