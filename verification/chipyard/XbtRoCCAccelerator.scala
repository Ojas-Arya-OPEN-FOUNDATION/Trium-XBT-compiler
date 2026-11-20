package freechips.rocketchip.subsystem

import chisel3._
import chisel3.util._
import org.chipsalliance.cde.config._
import freechips.rocketchip.tile._
import freechips.rocketchip.diplomacy._
import freechips.rocketchip.rocket._

/**
 * XbtRoCCAccelerator -- Berkeley Chipyard RoCC Coprocessor Wrapper
 * for the Trium-XBT Balanced Ternary Hardware Extension.
 *
 * Plugs directly into Rocket Core, BOOM, or any Chipyard Tile
 * using the custom-0 opcode (0x0B).
 *
 * Implements non-blocking RoCC command queue, 64-trit dual-rail registers,
 * and passes memory requests through the TileLink/L1 D$ cache.
 */
class XbtRoCCAccelerator(opcodes: OpcodeSet)(implicit p: Parameters) extends LazyRoCC(opcodes) {
  override lazy val module = new XbtRoCCAcceleratorModuleImp(this)
}

class XbtRoCCAcceleratorModuleImp(outer: XbtRoCCAccelerator)(implicit p: Parameters)
    extends LazyRoCCModuleImp(outer) {

  val cmd = Queue(io.cmd)
  val funct = cmd.bits.inst.funct
  val funct3 = cmd.bits.inst.funct(2, 0)
  val funct7 = cmd.bits.inst.funct(6, 0)

  // BlackBox wrapper importing hw/riscv_xbt_extension.sv
  val alu = Module(new riscv_xbt_alu_blackbox)

  alu.io.opcode := cmd.bits.inst.opcode(6, 0)
  alu.io.funct3 := funct3
  alu.io.funct7 := funct7
  alu.io.rs1_scalar := cmd.bits.rs1
  alu.io.rs2_scalar := cmd.bits.rs2

  // Register File Connections (32 x 128-bit dual-rail registers)
  val regfile = Module(new xbt_regfile_blackbox)
  regfile.io.clk := clock
  regfile.io.rst_n := ~reset.asBool
  regfile.io.rs1_addr := cmd.bits.inst.rs1
  regfile.io.rs2_addr := cmd.bits.inst.rs2
  regfile.io.rd_addr := cmd.bits.inst.rd

  alu.io.rs1_p := regfile.io.rs1_p
  alu.io.rs1_n := regfile.io.rs1_n
  alu.io.rs2_p := regfile.io.rs2_p
  alu.io.rs2_n := regfile.io.rs2_n

  // Writeback handling
  val do_wb = cmd.valid && cmd.bits.inst.xd
  regfile.io.we := do_wb && (cmd.bits.inst.rd =/= 0.U)
  regfile.io.rd_p := alu.io.rd_p
  regfile.io.rd_n := alu.io.rd_n

  // RoCC Response to Rocket Core
  io.resp.valid := cmd.valid && cmd.bits.inst.xd
  io.resp.bits.rd := cmd.bits.inst.rd
  io.resp.bits.data := alu.io.rd_scalar

  cmd.ready := io.resp.ready || !cmd.bits.inst.xd
  io.busy := cmd.valid
  io.interrupt := false.B
}

class riscv_xbt_alu_blackbox extends BlackBox with HasBlackBoxResource {
  val io = IO(new Bundle {
    val opcode = Input(UInt(7.W))
    val funct3 = Input(UInt(3.W))
    val funct7 = Input(UInt(7.W))
    val rs1_p = Input(UInt(64.W))
    val rs1_n = Input(UInt(64.W))
    val rs2_p = Input(UInt(64.W))
    val rs2_n = Input(UInt(64.W))
    val rs1_scalar = Input(UInt(64.W))
    val rs2_scalar = Input(UInt(64.W))
    val rd_p = Output(UInt(64.W))
    val rd_n = Output(UInt(64.W))
    val rd_aux_p = Output(UInt(64.W))
    val rd_aux_n = Output(UInt(64.W))
    val rd_scalar = Output(SInt(64.W))
    val valid_out = Output(Bool())
  })
  addResource("/hw/riscv_xbt_extension.sv")
}

class xbt_regfile_blackbox extends BlackBox with HasBlackBoxResource {
  val io = IO(new Bundle {
    val clk = Input(Clock())
    val rst_n = Input(Bool())
    val rs1_addr = Input(UInt(5.W))
    val rs1_p = Output(UInt(64.W))
    val rs1_n = Output(UInt(64.W))
    val rs2_addr = Input(UInt(5.W))
    val rs2_p = Output(UInt(64.W))
    val rs2_n = Output(UInt(64.W))
    val we = Input(Bool())
    val rd_addr = Input(UInt(5.W))
    val rd_p = Input(UInt(64.W))
    val rd_n = Input(UInt(64.W))
  })
  addResource("/hw/riscv_xbt_extension.sv")
}
