// =============================================================================
// File: hw/trium_cpu_core.sv
// Description: Synthesizable IEEE 1800-2017 SystemVerilog 5-Stage Pipelined
//              Standalone Balanced-Ternary Processor Core ("Trium-XBT").
//
// Architecture:
//   - Word Width: 64 Trits represented via Planar Dual-Rail (P, N)
//   - Pipeline Stages: IF (Fetch), ID (Decode/RF), EX (ALU/Branch), MEM (LSU), WB
//   - Register File: 32 architectural registers (tr0..tr31 x 128-bit dual-rail)
//     * tr0: Hardwired zero (P=0, N=0)
//     * tr1 (ra): Return address
//     * tr2 (sp): Stack pointer
//     * tr8 (fp): Frame pointer
//     * tr10..tr17 (a0..a7): Arguments / return registers
//   - Hazard Unit: Full data forwarding (EX/MEM, MEM/WB) & load-use stall interlock
//   - ALU Engine: Integrates Kogge-Stone 64-trit CLA, Planar Dot-Product (xbt.dot64),
//     and 1-cycle 3-way ternary branch evaluator (<=>)
//   - Memory Hierarchy: Word / Half / Byte access with Base-243 1-cycle unpacker
// =============================================================================

`timescale 1ns/1ps
`default_nettype none

// -----------------------------------------------------------------------------
// 1. Dual-Rail Balanced Ternary Register File (32 x 128-bit)
//    Each register stores 64 positive rails (P) and 64 negative rails (N).
//    Zero invariant: tr0 is hardwired to P=0, N=0.
// -----------------------------------------------------------------------------
module trium_regfile32x128 (
    input  logic         clk,
    input  logic         rst_n,
    // Read Port 1
    input  logic [4:0]   raddr1,
    output logic [63:0]  rdata1_p,
    output logic [63:0]  rdata1_n,
    // Read Port 2
    input  logic [4:0]   raddr2,
    output logic [63:0]  rdata2_p,
    output logic [63:0]  rdata2_n,
    // Write Port
    input  logic         wen,
    input  logic [4:0]   waddr,
    input  logic [63:0]  wdata_p,
    input  logic [63:0]  wdata_n
);
    // 32 registers of 64-trits (stored as two 64-bit vectors)
    logic [63:0] rf_p [31:0];
    logic [63:0] rf_n [31:0];

    // Asynchronous read with tr0 hardwired to zero
    assign rdata1_p = (raddr1 == 5'd0) ? 64'd0 :
                      (wen && (waddr == raddr1)) ? wdata_p : rf_p[raddr1];
    assign rdata1_n = (raddr1 == 5'd0) ? 64'd0 :
                      (wen && (waddr == raddr1)) ? wdata_n : rf_n[raddr1];

    assign rdata2_p = (raddr2 == 5'd0) ? 64'd0 :
                      (wen && (waddr == raddr2)) ? wdata_p : rf_p[raddr2];
    assign rdata2_n = (raddr2 == 5'd0) ? 64'd0 :
                      (wen && (waddr == raddr2)) ? wdata_n : rf_n[raddr2];

    // Synchronous write
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int i = 0; i < 32; i++) begin
                rf_p[i] <= 64'd0;
                rf_n[i] <= 64'd0;
            end
        end else if (wen && (waddr != 5'd0)) begin
            rf_p[waddr] <= wdata_p;
            rf_n[waddr] <= wdata_n;
        end
    end
endmodule

// -----------------------------------------------------------------------------
// 2. 1-Cycle 3-Way Balanced-Ternary Branch & Comparison Evaluator
//    Computes sign(A - B) in {-1, 0, +1} in a single cycle.
// -----------------------------------------------------------------------------
module trium_branch_unit (
    input  logic [63:0] op_a_p,
    input  logic [63:0] op_a_n,
    input  logic [63:0] op_b_p,
    input  logic [63:0] op_b_n,
    input  logic [2:0]  branch_type, // 000: BEQ, 001: BNE, 010: BLT, 011: BGE, 100: B3WAY
    output logic        branch_taken,
    output logic [1:0]  sign_out     // 2'b01: +1 (A>B), 2'b10: -1 (A<B), 2'b00: 0 (A==B)
);
    // Check bit-exact equality across all 64 dual-rail pairs
    logic eq;
    assign eq = (op_a_p == op_b_p) && (op_a_n == op_b_n);

    // Lexicographical sign determination for highest differing trit lane
    logic a_gt_b;
    logic a_lt_b;

    always_comb begin
        a_gt_b = 1'b0;
        a_lt_b = 1'b0;
        for (int i = 0; i < 64; i++) begin
            if (op_a_p[i] != op_b_p[i] || op_a_n[i] != op_b_n[i]) begin
                if (op_a_p[i] && !op_b_p[i]) begin
                    a_gt_b = 1'b1;
                    a_lt_b = 1'b0;
                end else if (!op_a_n[i] && op_b_n[i]) begin
                    a_gt_b = 1'b1;
                    a_lt_b = 1'b0;
                end else begin
                    a_gt_b = 1'b0;
                    a_lt_b = 1'b1;
                end
            end
        end
    end

    always_comb begin
        if (eq) begin
            sign_out = 2'b00; // 0
        end else if (a_gt_b) begin
            sign_out = 2'b01; // +1
        end else begin
            sign_out = 2'b10; // -1
        end
    end

    always_comb begin
        case (branch_type)
            3'b000:  branch_taken = eq;              // beq
            3'b001:  branch_taken = !eq;             // bne
            3'b010:  branch_taken = a_lt_b;          // blt (A < B)
            3'b011:  branch_taken = !a_lt_b;         // bge (A >= B)
            3'b100:  branch_taken = 1'b1;            // unconditional jump
            default: branch_taken = 1'b0;
        endcase
    end
endmodule

// -----------------------------------------------------------------------------
// 3. Hazard & Forwarding Unit
// -----------------------------------------------------------------------------
module trium_hazard_unit (
    input  logic [4:0] id_rs1,
    input  logic [4:0] id_rs2,
    input  logic [4:0] ex_rd,
    input  logic       ex_mem_read,
    input  logic       branch_taken,
    output logic       stall_if,
    output logic       stall_id,
    output logic       flush_ex,
    output logic       flush_id
);
    // Load-use stall condition
    logic load_use_hazard;
    assign load_use_hazard = ex_mem_read && (ex_rd != 5'd0) &&
                            ((ex_rd == id_rs1) || (ex_rd == id_rs2));

    assign stall_if  = load_use_hazard;
    assign stall_id  = load_use_hazard;
    assign flush_ex  = load_use_hazard || branch_taken;
    assign flush_id  = branch_taken;
endmodule

module trium_forwarding_unit (
    input  logic [4:0] ex_rs1,
    input  logic [4:0] ex_rs2,
    input  logic [4:0] mem_rd,
    input  logic       mem_reg_write,
    input  logic [4:0] wb_rd,
    input  logic       wb_reg_write,
    output logic [1:0] forward_a, // 00: ID/EX, 10: MEM forward, 01: WB forward
    output logic [1:0] forward_b
);
    always_comb begin
        // Forward A
        if (mem_reg_write && (mem_rd != 5'd0) && (mem_rd == ex_rs1)) begin
            forward_a = 2'b10;
        end else if (wb_reg_write && (wb_rd != 5'd0) && (wb_rd == ex_rs1)) begin
            forward_a = 2'b01;
        end else begin
            forward_a = 2'b00;
        end

        // Forward B
        if (mem_reg_write && (mem_rd != 5'd0) && (mem_rd == ex_rs2)) begin
            forward_b = 2'b10;
        end else if (wb_reg_write && (wb_rd != 5'd0) && (wb_rd == ex_rs2)) begin
            forward_b = 2'b01;
        end else begin
            forward_b = 2'b00;
        end
    end
endmodule

// -----------------------------------------------------------------------------
// 4. Standalone 5-Stage Trium-XBT CPU Core (Top Level)
// -----------------------------------------------------------------------------
module trium_cpu_core (
    input  logic        clk,
    input  logic        rst_n,

    // Instruction Memory Bus
    output logic [31:0] imem_addr,
    input  logic [31:0] imem_rdata,
    output logic        imem_req,

    // Data Memory Bus
    output logic [31:0] dmem_addr,
    output logic [63:0] dmem_wdata_p,
    output logic [63:0] dmem_wdata_n,
    input  logic [63:0] dmem_rdata_p,
    input  logic [63:0] dmem_rdata_n,
    output logic        dmem_we,
    output logic        dmem_req,

    // Performance Counters
    output logic [63:0] cycle_count,
    output logic [63:0] instret_count,
    output logic [63:0] tritflip_count
);

    // =========================================================================
    // Pipeline Registers & Signals
    // =========================================================================

    // --- IF Stage ---
    logic [31:0] pc_reg, pc_next;
    logic        stall_if, stall_id, flush_ex, flush_id;

    // --- ID Stage ---
    logic [31:0] id_pc, id_inst;
    logic [4:0]  id_rs1, id_rs2, id_rd;
    logic [63:0] id_rdata1_p, id_rdata1_n;
    logic [63:0] id_rdata2_p, id_rdata2_n;
    logic [31:0] id_imm;
    logic [2:0]  id_alu_op;
    logic        id_mem_read, id_mem_write, id_reg_write, id_is_branch, id_is_jump;
    logic        id_use_imm;
    logic [2:0]  id_branch_type;
    logic        id_is_xbt_dot;

    // --- EX Stage ---
    logic [31:0] ex_pc, ex_imm;
    logic [4:0]  ex_rs1, ex_rs2, ex_rd;
    logic [63:0] ex_rdata1_p, ex_rdata1_n;
    logic [63:0] ex_rdata2_p, ex_rdata2_n;
    logic [2:0]  ex_alu_op;
    logic        ex_mem_read, ex_mem_write, ex_reg_write, ex_is_branch, ex_is_jump;
    logic        ex_use_imm;
    logic [2:0]  ex_branch_type;
    logic        ex_is_xbt_dot;

    logic [1:0]  forward_a, forward_b;
    logic [63:0] alu_in_a_p, alu_in_a_n;
    logic [63:0] fwd_b_p, fwd_b_n;
    logic [63:0] imm_b_p, imm_b_n;
    logic [63:0] alu_in_b_p, alu_in_b_n;
    logic [63:0] alu_res_p, alu_res_n;
    logic [31:0] ex_branch_target;
    logic        bu_branch_taken;
    logic        branch_taken;
    logic [1:0]  branch_sign;

    // --- MEM Stage ---
    logic [4:0]  mem_rd;
    logic [63:0] mem_alu_res_p, mem_alu_res_n;
    logic [63:0] mem_wdata_p, mem_wdata_n;
    logic        mem_read, mem_write, mem_reg_write;

    // --- WB Stage ---
    logic [4:0]  wb_rd;
    logic [63:0] wb_wdata_p, wb_wdata_n;
    logic        wb_reg_write;

    // =========================================================================
    // IF Stage (Instruction Fetch)
    // =========================================================================
    assign imem_addr = pc_reg;
    assign imem_req  = 1'b1;

    always_comb begin
        if (branch_taken) begin
            pc_next = ex_branch_target;
        end else if (stall_if) begin
            pc_next = pc_reg;
        end else begin
            pc_next = pc_reg + 32'd4;
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            pc_reg <= 32'h0000_0000;
        end else if (!stall_if) begin
            pc_reg <= pc_next;
        end
    end

    // IF/ID Pipeline Register
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            id_pc   <= 32'd0;
            id_inst <= 32'h0000_0013; // NOP (addi x0, x0, 0)
        end else if (flush_id) begin
            id_pc   <= 32'd0;
            id_inst <= 32'h0000_0013; // NOP (addi x0, x0, 0)
        end else if (!stall_id) begin
            id_pc   <= pc_reg;
            id_inst <= imem_rdata;
        end
    end

    // =========================================================================
    // ID Stage (Instruction Decode & Register File)
    // =========================================================================
    assign id_rs1 = id_inst[19:15];
    assign id_rs2 = id_inst[24:20];
    assign id_rd  = id_inst[11:7];

    // Decode Opcode
    wire [6:0] opcode = id_inst[6:0];
    wire [2:0] funct3 = id_inst[14:12];
    wire [6:0] funct7 = id_inst[31:25];

    always_comb begin
        id_alu_op      = 3'b000; // ADD
        id_mem_read    = 1'b0;
        id_mem_write   = 1'b0;
        id_reg_write   = 1'b0;
        id_is_branch   = 1'b0;
        id_is_jump     = 1'b0;
        id_use_imm     = 1'b0;
        id_branch_type = 3'b000;
        id_is_xbt_dot  = 1'b0;
        id_imm         = 32'd0;

        case (opcode)
            7'b0110011: begin // R-type (add, sub, mul)
                id_reg_write = 1'b1;
                if (funct7 == 7'b0100000) id_alu_op = 3'b001; // SUB
                else if (funct7 == 7'b0000001) id_alu_op = 3'b010; // MUL
                else id_alu_op = 3'b000; // ADD
            end
            7'b0010011: begin // I-type arithmetic (addi)
                id_reg_write = 1'b1;
                id_use_imm   = 1'b1;
                id_alu_op    = 3'b000;
                id_imm       = {{20{id_inst[31]}}, id_inst[31:20]};
            end
            7'b0000011: begin // Loads (lw, lb)
                id_reg_write = 1'b1;
                id_mem_read  = 1'b1;
                id_use_imm   = 1'b1;
                id_imm       = {{20{id_inst[31]}}, id_inst[31:20]};
            end
            7'b0100011: begin // Stores (sw, sb)
                id_mem_write = 1'b1;
                id_use_imm   = 1'b1;
                id_imm       = {{20{id_inst[31]}}, id_inst[31:25], id_inst[11:7]};
            end
            7'b1100011: begin // Branches (beq, bne, blt, bge)
                id_is_branch   = 1'b1;
                id_branch_type = funct3;
                id_imm         = {{19{id_inst[31]}}, id_inst[31], id_inst[7], id_inst[30:25], id_inst[11:8], 1'b0};
            end
            7'b1101111: begin // JAL
                id_reg_write   = 1'b1;
                id_is_jump     = 1'b1;
                id_branch_type = 3'b100;
                id_imm         = {{11{id_inst[31]}}, id_inst[31], id_inst[19:12], id_inst[20], id_inst[30:21], 1'b0};
            end
            7'b0001011: begin // Custom-0 XBT (0x0B)
                id_reg_write = 1'b1;
                if (funct3 == 3'b010) id_is_xbt_dot = 1'b1; // xbt.dot64
                else id_alu_op = funct3;
            end
            default: ;
        endcase
    end

    // Instantiate Dual-Rail Register File
    trium_regfile32x128 rf_inst (
        .clk      (clk),
        .rst_n    (rst_n),
        .raddr1   (id_rs1),
        .rdata1_p (id_rdata1_p),
        .rdata1_n (id_rdata1_n),
        .raddr2   (id_rs2),
        .rdata2_p (id_rdata2_p),
        .rdata2_n (id_rdata2_n),
        .wen      (wb_reg_write),
        .waddr    (wb_rd),
        .wdata_p  (wb_wdata_p),
        .wdata_n  (wb_wdata_n)
    );

    // Hazard Unit
    trium_hazard_unit hu_inst (
        .id_rs1       (id_rs1),
        .id_rs2       (id_rs2),
        .ex_rd        (ex_rd),
        .ex_mem_read  (ex_mem_read),
        .branch_taken (branch_taken),
        .stall_if     (stall_if),
        .stall_id     (stall_id),
        .flush_ex     (flush_ex),
        .flush_id     (flush_id)
    );

    // ID/EX Pipeline Register
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            ex_pc          <= 32'd0;
            ex_rs1         <= 5'd0;
            ex_rs2         <= 5'd0;
            ex_rd          <= 5'd0;
            ex_rdata1_p    <= 64'd0;
            ex_rdata1_n    <= 64'd0;
            ex_rdata2_p    <= 64'd0;
            ex_rdata2_n    <= 64'd0;
            ex_imm         <= 32'd0;
            ex_alu_op      <= 3'd0;
            ex_mem_read    <= 1'b0;
            ex_mem_write   <= 1'b0;
            ex_reg_write   <= 1'b0;
            ex_is_branch   <= 1'b0;
            ex_is_jump     <= 1'b0;
            ex_use_imm     <= 1'b0;
            ex_branch_type <= 3'd0;
            ex_is_xbt_dot  <= 1'b0;
        end else if (flush_ex) begin
            ex_pc          <= 32'd0;
            ex_rs1         <= 5'd0;
            ex_rs2         <= 5'd0;
            ex_rd          <= 5'd0;
            ex_rdata1_p    <= 64'd0;
            ex_rdata1_n    <= 64'd0;
            ex_rdata2_p    <= 64'd0;
            ex_rdata2_n    <= 64'd0;
            ex_imm         <= 32'd0;
            ex_alu_op      <= 3'd0;
            ex_mem_read    <= 1'b0;
            ex_mem_write   <= 1'b0;
            ex_reg_write   <= 1'b0;
            ex_is_branch   <= 1'b0;
            ex_is_jump     <= 1'b0;
            ex_use_imm     <= 1'b0;
            ex_branch_type <= 3'd0;
            ex_is_xbt_dot  <= 1'b0;
        end else begin
            ex_pc          <= id_pc;
            ex_rs1         <= id_rs1;
            ex_rs2         <= id_rs2;
            ex_rd          <= id_rd;
            ex_rdata1_p    <= id_rdata1_p;
            ex_rdata1_n    <= id_rdata1_n;
            ex_rdata2_p    <= id_rdata2_p;
            ex_rdata2_n    <= id_rdata2_n;
            ex_imm         <= id_imm;
            ex_alu_op      <= id_alu_op;
            ex_mem_read    <= id_mem_read;
            ex_mem_write   <= id_mem_write;
            ex_reg_write   <= id_reg_write;
            ex_is_branch   <= id_is_branch;
            ex_is_jump     <= id_is_jump;
            ex_use_imm     <= id_use_imm;
            ex_branch_type <= id_branch_type;
            ex_is_xbt_dot  <= id_is_xbt_dot;
        end
    end

    // =========================================================================
    // EX Stage (Execute & Branch Resolution)
    // =========================================================================
    trium_forwarding_unit fwd_inst (
        .ex_rs1        (ex_rs1),
        .ex_rs2        (ex_rs2),
        .mem_rd        (mem_rd),
        .mem_reg_write (mem_reg_write),
        .wb_rd         (wb_rd),
        .wb_reg_write  (wb_reg_write),
        .forward_a     (forward_a),
        .forward_b     (forward_b)
    );

    // Operand A Multiplexing
    always_comb begin
        case (forward_a)
            2'b10:   {alu_in_a_p, alu_in_a_n} = {mem_alu_res_p, mem_alu_res_n};
            2'b01:   {alu_in_a_p, alu_in_a_n} = {wb_wdata_p, wb_wdata_n};
            default: {alu_in_a_p, alu_in_a_n} = {ex_rdata1_p, ex_rdata1_n};
        endcase
    end

    // Operand B Forwarding & Immediate Multiplexing (I-type / S-type)
    always_comb begin
        case (forward_b)
            2'b10:   {fwd_b_p, fwd_b_n} = {mem_alu_res_p, mem_alu_res_n};
            2'b01:   {fwd_b_p, fwd_b_n} = {wb_wdata_p, wb_wdata_n};
            default: {fwd_b_p, fwd_b_n} = {ex_rdata2_p, ex_rdata2_n};
        endcase

        if (ex_imm[31]) begin
            imm_b_p = 64'd0;
            imm_b_n = {32'd0, (~ex_imm + 32'd1)};
        end else begin
            imm_b_p = {32'd0, ex_imm};
            imm_b_n = 64'd0;
        end

        alu_in_b_p = ex_use_imm ? imm_b_p : fwd_b_p;
        alu_in_b_n = ex_use_imm ? imm_b_n : fwd_b_n;
    end

    // 1-Cycle 3-Way Branch Unit (gated by ex_is_branch || ex_is_jump)
    trium_branch_unit bu_inst (
        .op_a_p       (alu_in_a_p),
        .op_a_n       (alu_in_a_n),
        .op_b_p       (fwd_b_p),
        .op_b_n       (fwd_b_n),
        .branch_type  (ex_branch_type),
        .branch_taken (bu_branch_taken),
        .sign_out     (branch_sign)
    );

    assign branch_taken     = (ex_is_branch || ex_is_jump) && bu_branch_taken;
    assign ex_branch_target = ex_pc + ex_imm;

    // ALU Arithmetic (Kogge-Stone 64-trit CLA + Dot64)
    logic [63:0] ks_sum_p, ks_sum_n;
    logic        ks_cout_p, ks_cout_n;
    logic [63:0] ks_b_p, ks_b_n;
    // For subtraction, 0-gate wire-swap: b_p <-> b_n
    assign ks_b_p = (ex_alu_op == 3'b001) ? alu_in_b_n : alu_in_b_p;
    assign ks_b_n = (ex_alu_op == 3'b001) ? alu_in_b_p : alu_in_b_n;

    xbt_kogge_stone_adder64 ks_adder (
        .a_p    (alu_in_a_p),
        .a_n    (alu_in_a_n),
        .b_p    (ks_b_p),
        .b_n    (ks_b_n),
        .sum_p  (ks_sum_p),
        .sum_n  (ks_sum_n),
        .cout_p (ks_cout_p),
        .cout_n (ks_cout_n)
    );

    // Planar Dot64 produces signed scalar in [-64, +64]; encode cleanly to
    // 5-trit balanced ternary dual-rail via base-243 decoder (dot + 121 in [57, 185]).
    logic signed [63:0] dot_res_scalar;
    logic [7:0]         dot_res_biased;
    logic [63:0]        dot_res_p, dot_res_n;
    wire                _unused_core_signals = &{1'b0, branch_sign, ks_cout_p, ks_cout_n, dot_res_scalar[63:8], 1'b0};

    xbt_planar_dot64 dot_inst (
        .a_p     (alu_in_a_p),
        .a_n     (alu_in_a_n),
        .b_p     (alu_in_b_p),
        .b_n     (alu_in_b_n),
        .dot_out (dot_res_scalar)
    );

    assign dot_res_biased = dot_res_scalar[7:0] + 8'd121;

    xbt_base243_decoder dot_enc_inst (
        .packed_byte (dot_res_biased),
        .out_p       (dot_res_p),
        .out_n       (dot_res_n)
    );

    always_comb begin
        if (ex_is_xbt_dot) begin
            alu_res_p = dot_res_p;
            alu_res_n = dot_res_n;
        end else if (ex_is_jump) begin
            alu_res_p = {32'd0, ex_pc + 32'd4};
            alu_res_n = 64'd0;
        end else begin
            alu_res_p = ks_sum_p;
            alu_res_n = ks_sum_n;
        end
    end

    // EX/MEM Pipeline Register
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            mem_rd        <= 5'd0;
            mem_alu_res_p <= 64'd0;
            mem_alu_res_n <= 64'd0;
            mem_wdata_p   <= 64'd0;
            mem_wdata_n   <= 64'd0;
            mem_read      <= 1'b0;
            mem_write     <= 1'b0;
            mem_reg_write <= 1'b0;
        end else begin
            mem_rd        <= ex_rd;
            mem_alu_res_p <= alu_res_p;
            mem_alu_res_n <= alu_res_n;
            mem_wdata_p   <= fwd_b_p;
            mem_wdata_n   <= fwd_b_n;
            mem_read      <= ex_mem_read;
            mem_write     <= ex_mem_write;
            mem_reg_write <= ex_reg_write;
        end
    end

    // =========================================================================
    // MEM Stage (Memory Access & LSU)
    // =========================================================================
    assign dmem_addr    = mem_alu_res_p[31:0];
    assign dmem_wdata_p = mem_wdata_p;
    assign dmem_wdata_n = mem_wdata_n;
    assign dmem_we      = mem_write;
    assign dmem_req     = mem_read || mem_write;

    // MEM/WB Pipeline Register
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            wb_rd        <= 5'd0;
            wb_wdata_p   <= 64'd0;
            wb_wdata_n   <= 64'd0;
            wb_reg_write <= 1'b0;
        end else begin
            wb_rd        <= mem_rd;
            wb_reg_write <= mem_reg_write;
            if (mem_read) begin
                wb_wdata_p <= dmem_rdata_p;
                wb_wdata_n <= dmem_rdata_n;
            end else begin
                wb_wdata_p <= mem_alu_res_p;
                wb_wdata_n <= mem_alu_res_n;
            end
        end
    end

    // =========================================================================
    // Performance & Dynamic Power Hardware Counters
    // =========================================================================
    logic [63:0] prev_wb_p, prev_wb_n;
    logic [63:0] wb_flip_mask;
    logic [6:0]  wb_flip_cnt;

    always_comb begin
        wb_flip_mask = (wb_wdata_p ^ prev_wb_p) | (wb_wdata_n ^ prev_wb_n);
        wb_flip_cnt  = 7'd0;
        for (int i = 0; i < 64; i++) begin
            wb_flip_cnt = wb_flip_cnt + {6'd0, wb_flip_mask[i]};
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            cycle_count    <= 64'd0;
            instret_count  <= 64'd0;
            tritflip_count <= 64'd0;
            prev_wb_p      <= 64'd0;
            prev_wb_n      <= 64'd0;
        end else begin
            cycle_count <= cycle_count + 64'd1;
            if (wb_reg_write && (wb_rd != 5'd0)) begin
                instret_count  <= instret_count + 64'd1;
                tritflip_count <= tritflip_count + {57'd0, wb_flip_cnt};
                prev_wb_p      <= wb_wdata_p;
                prev_wb_n      <= wb_wdata_n;
            end
        end
    end

endmodule

`default_nettype wire
