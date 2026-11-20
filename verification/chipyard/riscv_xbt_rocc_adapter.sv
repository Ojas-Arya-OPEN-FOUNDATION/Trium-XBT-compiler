// =============================================================================
// File: verification/chipyard/riscv_xbt_rocc_adapter.sv
// Description: Standard RISC-V Rocket Custom Coprocessor (RoCC) SystemVerilog
//              Adapter for Chipyard SoC Integration.
//
// Complies with the UC Berkeley Chipyard / Rocket-Chip RoCC interface:
//   - Non-blocking decoupled command channel (rocc_cmd_valid, rocc_cmd_ready)
//   - Decoupled response channel (rocc_resp_valid, rocc_resp_ready)
//   - Direct pipeline link to riscv_xbt_alu and xbt_regfile32x128
// =============================================================================

`default_nettype none

module riscv_xbt_rocc_adapter (
    input  logic        clk,
    input  logic        rst_n,

    // RoCC Command Interface (from Rocket Core)
    input  logic        rocc_cmd_valid,
    output logic        rocc_cmd_ready,
    input  logic [6:0]  rocc_cmd_inst_opcode,
    input  logic [4:0]  rocc_cmd_inst_rd,
    input  logic [4:0]  rocc_cmd_inst_rs1,
    input  logic [4:0]  rocc_cmd_inst_rs2,
    input  logic [2:0]  rocc_cmd_inst_funct3,
    input  logic [6:0]  rocc_cmd_inst_funct7,
    input  logic        rocc_cmd_inst_xd,
    input  logic        rocc_cmd_inst_xs1,
    input  logic        rocc_cmd_inst_xs2,
    input  logic [63:0] rocc_cmd_rs1_data,
    input  logic [63:0] rocc_cmd_rs2_data,

    // RoCC Response Interface (to Rocket Core)
    output logic        rocc_resp_valid,
    input  logic        rocc_resp_ready,
    output logic [4:0]  rocc_resp_rd,
    output logic [63:0] rocc_resp_data,

    // Status Signals
    output logic        rocc_busy,
    output logic        rocc_interrupt
);

    // Wires to internal Xbt ALU
    logic [63:0] alu_rd_p, alu_rd_n, alu_rd_aux_p, alu_rd_aux_n;
    logic signed [63:0] alu_rd_scalar;
    logic alu_valid;

    // Register File wires
    logic [63:0] rf_rs1_p, rf_rs1_n, rf_rs2_p, rf_rs2_n;
    logic rf_we;

    assign rf_we = rocc_cmd_valid && (rocc_cmd_inst_rd != 5'd0);

    // Instantiate 32x128 Dual-Rail Register File
    xbt_regfile32x128 u_regfile (
        .clk      (clk),
        .rst_n    (rst_n),
        .rs1_addr (rocc_cmd_inst_rs1),
        .rs1_neg  (1'b0),
        .rs1_p    (rf_rs1_p),
        .rs1_n    (rf_rs1_n),
        .rs2_addr (rocc_cmd_inst_rs2),
        .rs2_neg  (1'b0),
        .rs2_p    (rf_rs2_p),
        .rs2_n    (rf_rs2_n),
        .we       (rf_we),
        .rd_addr  (rocc_cmd_inst_rd),
        .rd_p     (alu_rd_p),
        .rd_n     (alu_rd_n)
    );

    // Instantiate Top-Level Xbt Execution Unit
    riscv_xbt_alu u_alu (
        .opcode     (rocc_cmd_inst_opcode),
        .funct3     (rocc_cmd_inst_funct3),
        .funct7     (rocc_cmd_inst_funct7),
        .rs1_p      (rf_rs1_p),
        .rs1_n      (rf_rs1_n),
        .rs2_p      (rf_rs2_p),
        .rs2_n      (rf_rs2_n),
        .rs1_scalar (rocc_cmd_rs1_data),
        .rd_p       (alu_rd_p),
        .rd_n       (alu_rd_n),
        .rd_aux_p   (alu_rd_aux_p),
        .rd_aux_n   (alu_rd_aux_n),
        .rd_scalar  (alu_rd_scalar),
        .valid_out  (alu_valid)
    );

    // Handshake logic
    always_comb begin
        rocc_cmd_ready  = rocc_resp_ready || !rocc_cmd_inst_xd;
        rocc_resp_valid = rocc_cmd_valid && rocc_cmd_inst_xd && alu_valid;
        rocc_resp_rd    = rocc_cmd_inst_rd;
        rocc_resp_data  = (rocc_cmd_inst_funct3 == 3'b010) ? 64'(signed'(alu_rd_scalar)) : alu_rd_p;
        rocc_busy       = rocc_cmd_valid;
        rocc_interrupt  = 1'b0;
    end

endmodule

`default_nettype wire
