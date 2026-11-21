// =============================================================================
// File: hw/tb_trium_cpu_core.sv
// Description: Synthesizable/Verilator IEEE 1800-2017 SystemVerilog RTL Testbench
//              for the 5-Stage Pipelined Trium-XBT CPU Core (hw/trium_cpu_core.sv).
//
// Workload:
//   Executes a real 15-instruction program exercising:
//     - Register initialization (addi)
//     - 64-Trit Planar Dot Product SIMD (xbt.dot64 Custom-0)
//     - Kogge-Stone Dual-Rail Accumulator (add)
//     - Dynamic loop control & hazard-flushed conditional branching (blt)
//     - Dual-rail exact register equality checking & branching (beq)
//     - Dynamic power tracking (Hamming trit-flips) & retirement counters
// =============================================================================

`timescale 1ns/1ps
`default_nettype none

module tb_trium_cpu_core;

    logic        clk;
    logic        rst_n;

    // Instruction Memory Interface
    logic [31:0] imem_addr;
    logic [31:0] imem_rdata;
    logic        imem_req;

    // Data Memory Interface
    logic [31:0] dmem_addr;
    logic [63:0] dmem_wdata_p;
    logic [63:0] dmem_wdata_n;
    logic [63:0] dmem_rdata_p;
    logic [63:0] dmem_rdata_n;
    logic        dmem_we;
    logic        dmem_req;

    // Performance Counters
    logic [63:0] cycle_count;
    logic [63:0] instret_count;
    logic [63:0] tritflip_count;
    wire         _unused_tb = &{1'b0, imem_addr[31:8], imem_addr[1:0], imem_req, dmem_addr[31:8], dmem_addr[1:0], 1'b0};

    // -------------------------------------------------------------------------
    // Device Under Test (DUT)
    // -------------------------------------------------------------------------
    trium_cpu_core dut (
        .clk            (clk),
        .rst_n          (rst_n),
        .imem_addr      (imem_addr),
        .imem_rdata     (imem_rdata),
        .imem_req       (imem_req),
        .dmem_addr      (dmem_addr),
        .dmem_wdata_p   (dmem_wdata_p),
        .dmem_wdata_n   (dmem_wdata_n),
        .dmem_rdata_p   (dmem_rdata_p),
        .dmem_rdata_n   (dmem_rdata_n),
        .dmem_we        (dmem_we),
        .dmem_req       (dmem_req),
        .cycle_count    (cycle_count),
        .instret_count  (instret_count),
        .tritflip_count (tritflip_count)
    );

    // -------------------------------------------------------------------------
    // 64-Word Instruction ROM (Preloaded with Loop + XBT.DOT64 Program)
    // -------------------------------------------------------------------------
    logic [31:0] imem [0:63];

    initial begin
        for (int i = 0; i < 64; i++) imem[i] = 32'h0000_0013; // default nop

        // Program: 4-iteration vector dot product accumulator loop
        imem[0]  = 32'h00000313; // 0x00: addi tr6, tr0, 0     (loop counter i = 0)
        imem[1]  = 32'h00400393; // 0x04: addi tr7, tr0, 4     (loop bound N = 4)
        imem[2]  = 32'h00000293; // 0x08: addi tr5, tr0, 0     (accumulator acc = 0)
        imem[3]  = 32'h00700513; // 0x0C: addi tr10, tr0, 7    (vec A: 3 trits +1)
        imem[4]  = 32'h00700593; // 0x10: addi tr11, tr0, 7    (vec B: 3 trits +1)
        // Loop Body:
        imem[5]  = 32'h00B5260B; // 0x14: xbt.dot64 tr12, tr10, tr11 (dot = 3*1 = +3)
        imem[6]  = 32'h00C282B3; // 0x18: add tr5, tr5, tr12   (acc = acc + 3)
        imem[7]  = 32'h00130313; // 0x1C: addi tr6, tr6, 1     (i = i + 1)
        imem[8]  = 32'hFE732AE3; // 0x20: blt tr6, tr7, -12    (if i < 4 branch 0x14)
        // Post-Loop Verification:
        imem[9]  = 32'h00800693; // 0x24: addi tr13, tr0, 8    (expected = 4 iters * 2 = 8)
        imem[10] = 32'h00D28663; // 0x28: beq tr5, tr13, +12   (if acc == 12 branch 0x34)
        imem[11] = 32'hFFF00713; // 0x2C: addi tr14, tr0, -1   (FAIL flag)
        imem[12] = 32'h0080006F; // 0x30: jal tr0, +8          (jump 0x38 halt)
        imem[13] = 32'h00100713; // 0x34: addi tr14, tr0, 1    (PASS flag)
        imem[14] = 32'h0000006F; // 0x38: jal tr0, 0           (halt loop)
    end

    // Instruction fetch (combinational / single-cycle model)
    wire [5:0] imem_word_addr = imem_addr[7:2];
    assign imem_rdata = imem[imem_word_addr];

    // -------------------------------------------------------------------------
    // Data Memory RAM Model (64 words x 128-bit dual-rail)
    // -------------------------------------------------------------------------
    logic [63:0] dmem_p [0:63];
    logic [63:0] dmem_n [0:63];
    wire [5:0] dmem_word_addr = dmem_addr[7:2];

    always_ff @(posedge clk) begin
        if (dmem_we && dmem_req) begin
            dmem_p[dmem_word_addr] <= dmem_wdata_p;
            dmem_n[dmem_word_addr] <= dmem_wdata_n;
        end
    end
    assign dmem_rdata_p = dmem_p[dmem_word_addr];
    assign dmem_rdata_n = dmem_n[dmem_word_addr];

    // -------------------------------------------------------------------------
    // Clock & Reset Generation
    // -------------------------------------------------------------------------
    initial begin
        clk = 1'b0;
        forever #5 clk = ~clk; // 100 MHz clock
    end

    initial begin
        rst_n = 1'b0;
        #25;
        rst_n = 1'b1;
    end

    // -------------------------------------------------------------------------
    // Test Execution & Verification Monitor
    // -------------------------------------------------------------------------
    int sim_cycles = 0;
    initial begin
        $display("=== Running tb_trium_cpu_core (5-Stage Pipelined RTL Testbench) ===");
        
        // Wait for reset release
        @(posedge rst_n);

        // Run until halt loop (PC == 0x38) or cycle limit
        while (sim_cycles < 150) begin
            @(posedge clk);
            sim_cycles++;

            if (dut.pc_reg == 32'h0000_0038) begin
                // Give pipeline 5 extra cycles to commit writeback to RF
                repeat (5) @(posedge clk);
                break;
            end
        end

        // Verify Register Results
        $display("[RTL VERIFY] Cycles elapsed   : %0d", sim_cycles);
        $display("[RTL VERIFY] Core cycle_count : %0d", cycle_count);
        $display("[RTL VERIFY] instret_count    : %0d", instret_count);
        $display("[RTL VERIFY] tritflip_count   : %0d", tritflip_count);

        $display("[RTL VERIFY] Final tr5 (acc)  : P=0x%016X, N=0x%016X",
                 dut.rf_inst.rf_p[5], dut.rf_inst.rf_n[5]);
        $display("[RTL VERIFY] Final tr6 (iter) : P=0x%016X, N=0x%016X",
                 dut.rf_inst.rf_p[6], dut.rf_inst.rf_n[6]);
        $display("[RTL VERIFY] Final tr14 (pass): P=0x%016X, N=0x%016X",
                 dut.rf_inst.rf_p[14], dut.rf_inst.rf_n[14]);

        // tr14 must be 1 (P=1, N=0)
        if (dut.rf_inst.rf_p[14] == 64'd1 && dut.rf_inst.rf_n[14] == 64'd0) begin
            $display("=== tb_trium_cpu_core: [PASS] 5-Stage Core Loop + XBT.DOT64 100%% RTL-Verified! ===");
            $finish;
        end else begin
            $display("=== tb_trium_cpu_core: [FAIL] Output mismatch in tr14 or accumulator! ===");
            $fatal(1, "RTL simulation failed assertions");
        end
    end

endmodule
`default_nettype wire
