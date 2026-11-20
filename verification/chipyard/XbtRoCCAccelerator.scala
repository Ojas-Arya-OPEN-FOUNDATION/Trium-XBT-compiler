package freechips.rocketchip.subsystem

import chisel3._
import chisel3.util._
import org.chipsalliance.cde.config._
import freechips.rocketchip.tile._
import freechips.rocketchip.diplomacy._
import freechips.rocketchip.rocket._

/**
 * XbtRoCCAccelerator -- Berkeley Chipyard RoCC Coprocessor Wrapper
 * for the Trium-XBT Balanced Ternary Hardware Extension (v1 + v2).
 *
 * Plugs directly into Rocket Core, BOOM, or any Chipyard Tile
 * using the custom-0 opcode (0x0B).
 */
class XbtRoCCAccelerator(opcodes: OpcodeSet)(implicit p: Parameters) extends LazyRoCC(opcodes) {
  override lazy val module = new XbtRoCCAcceleratorModuleImp(this)
}

class XbtRoCCAcceleratorModuleImp(outer: XbtRoCCAccelerator)(implicit p: Parameters)
    extends LazyRoCCModuleImp(outer) {

  val cmd = Queue(io.cmd)
  val funct7 = cmd.bits.inst.funct(6, 0)
  val funct3 = Cat(cmd.bits.inst.xd, cmd.bits.inst.xs1, cmd.bits.inst.xs2)
  val imm12  = Cat(funct7, cmd.bits.inst.rs2)

  // BlackBox wrapper importing hw/riscv_xbt_extension.sv (module riscv_xbt_alu)
  val alu = Module(new riscv_xbt_alu_blackbox)

  // Register File Connections (32 x 128-bit dual-rail registers, module xbt_regfile32x128)
  val regfile = Module(new xbt_regfile_blackbox)
  regfile.io.clk      := clock
  regfile.io.rst_n    := ~reset.asBool
  regfile.io.rs1_addr := cmd.bits.inst.rs1
  regfile.io.rs1_neg  := false.B
  regfile.io.rs2_addr := cmd.bits.inst.rs2
  regfile.io.rs2_neg  := false.B
  regfile.io.rd_addr  := cmd.bits.inst.rd

  // Shadow read bank for tr[rd] (accumulator / conditional-move source for CSAMAC, BTFA, CMOV*)
  val rdShadowP = RegInit(VecInit(Seq.fill(32)(0.U(64.W))))
  val rdShadowN = RegInit(VecInit(Seq.fill(32)(0.U(64.W))))
  val rdInP     = Mux(cmd.bits.inst.rd === 0.U, 0.U(64.W), rdShadowP(cmd.bits.inst.rd))
  val rdInN     = Mux(cmd.bits.inst.rd === 0.U, 0.U(64.W), rdShadowN(cmd.bits.inst.rd))

  alu.io.opcode     := cmd.bits.inst.opcode(6, 0)
  alu.io.funct3     := funct3
  alu.io.funct7     := funct7
  alu.io.rs1_p      := regfile.io.rs1_p
  alu.io.rs1_n      := regfile.io.rs1_n
  alu.io.rs2_p      := regfile.io.rs2_p
  alu.io.rs2_n      := regfile.io.rs2_n
  alu.io.rd_p_in    := rdInP
  alu.io.rd_n_in    := rdInN
  alu.io.rs1_scalar := cmd.bits.rs1
  alu.io.imm12      := imm12

  // Scalar-result instructions (DOT64 and GROUP2.PACK243) write x[rd] only and must not clobber tr[rd]
  val isScalarResult = (funct3 === 2.U) || (funct3 === 6.U && funct7 === 2.U)

  // Writeback handling
  val doRfWrite = cmd.fire && alu.io.valid_out && !isScalarResult && (cmd.bits.inst.rd =/= 0.U)
  regfile.io.we   := doRfWrite
  regfile.io.rd_p := alu.io.rd_p
  regfile.io.rd_n := alu.io.rd_n

  when(doRfWrite) {
    rdShadowP(cmd.bits.inst.rd) := alu.io.rd_p
    rdShadowN(cmd.bits.inst.rd) := alu.io.rd_n
  }

  // RoCC Response to Rocket Core
  io.resp.valid     := cmd.valid && cmd.bits.inst.xd && alu.io.valid_out
  io.resp.bits.rd   := cmd.bits.inst.rd
  io.resp.bits.data := Mux(isScalarResult, alu.io.rd_scalar.asUInt, alu.io.rd_p)

  cmd.ready    := io.resp.ready || !cmd.bits.inst.xd
  io.busy      := cmd.valid
  io.interrupt := false.B
}

class riscv_xbt_alu_blackbox extends BlackBox with HasBlackBoxResource {
  override def desiredName = "riscv_xbt_alu"

  val io = IO(new Bundle {
    val opcode     = Input(UInt(7.W))
    val funct3     = Input(UInt(3.W))
    val funct7     = Input(UInt(7.W))
    val rs1_p      = Input(UInt(64.W))
    val rs1_n      = Input(UInt(64.W))
    val rs2_p      = Input(UInt(64.W))
    val rs2_n      = Input(UInt(64.W))
    val rd_p_in    = Input(UInt(64.W))
    val rd_n_in    = Input(UInt(64.W))
    val rs1_scalar = Input(UInt(64.W))
    val imm12      = Input(UInt(12.W))
    val rd_p       = Output(UInt(64.W))
    val rd_n       = Output(UInt(64.W))
    val rd_aux_p   = Output(UInt(64.W))
    val rd_aux_n   = Output(UInt(64.W))
    val rd_scalar  = Output(SInt(64.W))
    val valid_out  = Output(Bool())
  })
  addResource("/hw/riscv_xbt_extension.sv")
}

class xbt_regfile_blackbox extends BlackBox with HasBlackBoxResource {
  override def desiredName = "xbt_regfile32x128"

  val io = IO(new Bundle {
    val clk      = Input(Clock())
    val rst_n    = Input(Bool())
    val rs1_addr = Input(UInt(5.W))
    val rs1_neg  = Input(Bool())
    val rs1_p    = Output(UInt(64.W))
    val rs1_n    = Output(UInt(64.W))
    val rs2_addr = Input(UInt(5.W))
    val rs2_neg  = Input(Bool())
    val rs2_p    = Output(UInt(64.W))
    val rs2_n    = Output(UInt(64.W))
    val we       = Input(Bool())
    val rd_addr  = Input(UInt(5.W))
    val rd_p     = Input(UInt(64.W))
    val rd_n     = Input(UInt(64.W))
  })
  addResource("/hw/riscv_xbt_extension.sv")
}

/**
 * Chipyard Config Mixin to attach XbtRoCCAccelerator on custom-0 (0x0B).
 */
class WithXbtRoCCAccelerator extends Config((site, here, up) => {
  case BuildRoCC => up(BuildRoCC) ++ Seq(
    (p: Parameters) => {
      val xbt = LazyModule(new XbtRoCCAccelerator(OpcodeSet.custom0)(p))
      xbt
    }
  )
})
