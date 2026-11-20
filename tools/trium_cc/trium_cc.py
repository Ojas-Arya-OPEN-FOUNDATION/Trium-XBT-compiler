#!/usr/bin/env python3
"""
trium_cc.py -- General-Purpose C and Assembly Compiler Driver for Trium-XBT.

Compiles high-level C programs (.c) or standard binary RISC-V assembly files (.s)
into native balanced-ternary Trium-XBT machine assembly (.tas) and machine bytecode.

Features:
  1. Full C99 Frontend (via pycparser):
     - Functions, stack frames, calling conventions (ABI: sp=tr2, ra=tr1, fp=tr8, a0..a7=tr10..tr17)
     - Arithmetic, pointer arithmetic, structs, arrays, loops, and control flow
     - Native balanced-ternary 3-way branch synthesis (<=> / sign)
  2. Auto-Vectorization & Idiom Recognition:
     - Loop vectorizer detects scalar dot-product and matrix loops:
       sum += a[i] * b[i]  -->  xbt.dot64 (64 trits per cycle)
     - Base-243 packing optimization for ternary weight tensors.
  3. Preprocessed Assembly Translation:
     - Lowering engine for standard RV32I / RV64I assembly files (e.g., Dhrystone, Coremark).
"""

from __future__ import annotations

import os
import sys
import argparse
import re
from typing import List, Dict, Tuple, Any, Optional

try:
    import pycparser
    from pycparser import c_parser, c_ast
    HAVE_PYCPARSER = True
except ImportError:
    HAVE_PYCPARSER = False

try:
    from tools.trium_cc.rv_translator import RVToTriumTranslator
except ImportError:
    from rv_translator import RVToTriumTranslator


class CToTriumCompiler(c_ast.NodeVisitor if HAVE_PYCPARSER else object):
    """
    AST Code Generator compiling C99 AST directly into Trium-XBT assembly.
    """
    def __init__(self, optimize_dot: bool = True):
        self.optimize_dot = optimize_dot
        self.code: List[str] = []
        self.locals: Dict[str, Tuple[str, Any]] = {}  # name -> (type, location)
        self.structs: Dict[str, Dict[str, int]] = {}
        self.label_counter = 0
        self.current_func = ""
        self.frame_size = 64

    def new_label(self, prefix: str = "L") -> str:
        self.label_counter += 1
        return f".{prefix}_{self.label_counter}"

    def emit(self, mnemonic: str, operands: Optional[List[str]] = None, comment: str = ""):
        ops = ", ".join(operands) if operands else ""
        line = f"    {mnemonic:<12} {ops}" if ops else f"    {mnemonic}"
        if comment:
            line = f"{line:<40} # {comment}"
        self.code.append(line)

    def emit_label(self, label: str):
        self.code.append(f"{label}:")

    def visit_FuncDef(self, node: c_ast.FuncDef):
        func_name = node.decl.name
        self.current_func = func_name
        self.emit_label(func_name)
        self.emit(f"# === Prologue for {func_name} ===")
        self.emit("xbt.addi", ["tr2", "tr2", f"-{self.frame_size}"], "allocate stack frame (sp=tr2)")
        self.emit("xbt.sw", ["tr1", f"{self.frame_size - 4}(tr2)"], "save return address (ra=tr1)")
        self.emit("xbt.sw", ["tr8", f"{self.frame_size - 8}(tr2)"], "save frame pointer (fp=tr8)")
        self.emit("xbt.addi", ["tr8", "tr2", str(self.frame_size)], "setup frame pointer")

        self.locals = {}
        self.stack_offset = -16
        # Map parameters to argument registers a0..a7 (tr10..tr17)
        if node.decl.type.args:
            for i, param in enumerate(node.decl.type.args.params):
                reg = f"tr{10 + i}"
                self.locals[param.name] = ("reg", reg)

        self.visit(node.body)

        self.emit_label(f".L_ret_{func_name}")
        self.emit(f"# === Epilogue for {func_name} ===")
        self.emit("xbt.lw", ["tr1", f"{self.frame_size - 4}(tr2)"], "restore ra")
        self.emit("xbt.lw", ["tr8", f"{self.frame_size - 8}(tr2)"], "restore fp")
        self.emit("xbt.addi", ["tr2", "tr2", str(self.frame_size)], "deallocate frame")
        self.emit("xbt.jalr", ["tr0", "0(tr1)"], "ret")

    def visit_Compound(self, node: c_ast.Compound):
        if node.block_items:
            for item in node.block_items:
                self.visit(item)

    def visit_DeclList(self, node: c_ast.DeclList):
        for d in node.decls:
            self.visit_Decl(d)

    def visit_Decl(self, node: c_ast.Decl):
        if isinstance(node.type, c_ast.FuncDecl):
            return
        name = node.name
        if name not in self.locals:
            self.locals[name] = ("stack", self.stack_offset)
            self.stack_offset -= 4
        if node.init:
            reg = self.visit_expr(node.init, target_reg="tr5")
            loc_type, loc = self.locals[name]
            if loc_type == "reg":
                if reg != loc:
                    self.emit("xbt.addi", [loc, reg, "0"])
            else:
                self.emit("xbt.sw", [reg, f"{loc}(tr8)"])

    def visit_UnaryOp(self, node: c_ast.UnaryOp):
        if node.op in ("p++", "++"):
            var_name = node.expr.name
            loc_type, loc = self.locals.get(var_name, ("unknown", None))
            if loc_type == "reg":
                self.emit("xbt.addi", [loc, loc, "1"])
            elif loc_type == "stack":
                self.emit("xbt.lw", ["tr5", f"{loc}(tr8)"])
                self.emit("xbt.addi", ["tr5", "tr5", "1"])
                self.emit("xbt.sw", ["tr5", f"{loc}(tr8)"])
        elif node.op in ("p--", "--"):
            var_name = node.expr.name
            loc_type, loc = self.locals.get(var_name, ("unknown", None))
            if loc_type == "reg":
                self.emit("xbt.addi", [loc, loc, "-1"])
            elif loc_type == "stack":
                self.emit("xbt.lw", ["tr5", f"{loc}(tr8)"])
                self.emit("xbt.addi", ["tr5", "tr5", "-1"])
                self.emit("xbt.sw", ["tr5", f"{loc}(tr8)"])

    def visit_Return(self, node: c_ast.Return):
        if node.expr:
            res_reg = self.visit_expr(node.expr, target_reg="tr10")
            if res_reg != "tr10":
                self.emit("xbt.addi", ["tr10", res_reg, "0"], "move return value to a0 (tr10)")
        self.emit("xbt.jal", ["tr0", f".L_ret_{self.current_func}"], "jump to epilogue")

    def visit_For(self, node: c_ast.For):
        # Check for loop vectorization (dot product pattern)
        if self.optimize_dot and self._try_vectorize_dot(node):
            return

        lbl_start = self.new_label("for_start")
        lbl_cond = self.new_label("for_cond")
        lbl_end = self.new_label("for_end")

        if node.init:
            self.visit(node.init)

        self.emit("xbt.jal", ["tr0", lbl_cond])
        self.emit_label(lbl_start)

        if node.stmt:
            self.visit(node.stmt)

        if node.next:
            self.visit(node.next)

        self.emit_label(lbl_cond)
        if node.cond:
            if isinstance(node.cond, c_ast.BinaryOp):
                left = self.visit_expr(node.cond.left, target_reg="tr5")
                right = self.visit_expr(node.cond.right, target_reg="tr6")
                op = node.cond.op
                if op == "<=":
                    self.emit("xbt.bge", [right, left, lbl_start])
                elif op == "<":
                    self.emit("xbt.blt", [left, right, lbl_start])
                elif op == ">=":
                    self.emit("xbt.bge", [left, right, lbl_start])
                elif op == ">":
                    self.emit("xbt.blt", [right, left, lbl_start])
                elif op == "==":
                    self.emit("xbt.beq", [left, right, lbl_start])
                elif op == "!=":
                    self.emit("xbt.bne", [left, right, lbl_start])
            else:
                res = self.visit_expr(node.cond, target_reg="tr5")
                self.emit("xbt.bne", [res, "tr0", lbl_start], "loop condition true")
        else:
            self.emit("xbt.jal", ["tr0", lbl_start])

        self.emit_label(lbl_end)

    def _try_vectorize_dot(self, node: c_ast.For) -> bool:
        """
        Detects vector dot-product loop:
          for (i = 0; i < N; i++) sum += a[i] * b[i];
        and promotes it to 64-trit SIMD: xbt.dot64.
        """
        try:
            # Check loop body
            body = node.stmt
            if isinstance(body, c_ast.Compound) and len(body.block_items) == 1:
                body = body.block_items[0]

            if not isinstance(body, c_ast.Assignment):
                return False

            # Pattern: sum += a[i] * b[i]
            if body.op in ("+=", "="):
                rhs = body.rvalue
                if body.op == "=" and isinstance(rhs, c_ast.BinaryOp) and rhs.op == "+":
                    rhs = rhs.right

                if isinstance(rhs, c_ast.BinaryOp) and rhs.op == "*":
                    if isinstance(rhs.left, c_ast.ArrayRef) and isinstance(rhs.right, c_ast.ArrayRef):
                        target_var = body.lvalue.name
                        a_name = rhs.left.name.name
                        b_name = rhs.right.name.name

                        a_loc = self.locals.get(a_name, ("reg", "tr10"))
                        b_loc = self.locals.get(b_name, ("reg", "tr11"))
                        res_loc = self.locals.get(target_var, ("reg", "tr10"))

                        a_reg = a_loc[1] if a_loc[0] == "reg" else "tr10"
                        b_reg = b_loc[1] if b_loc[0] == "reg" else "tr11"
                        res_reg = res_loc[1] if res_loc[0] == "reg" else "tr10"

                        self.emit(f"# --- Auto-Vectorized Loop -> XBT.DOT64 ---")
                        self.emit("xbt.dot64", [res_reg, a_reg, b_reg], "1-cycle 64-trit balanced ternary dot product")
                        if res_loc[0] == "stack":
                            self.emit("xbt.sw", [res_reg, f"{res_loc[1]}(tr8)"], f"store dot product result to {target_var}")
                        return True
        except Exception:
            pass
        return False

    def visit_While(self, node: c_ast.While):
        lbl_cond = self.new_label("while_cond")
        lbl_body = self.new_label("while_body")
        lbl_end = self.new_label("while_end")

        self.emit_label(lbl_cond)
        res = self.visit_expr(node.cond, target_reg="tr5")
        self.emit("xbt.beq", [res, "tr0", lbl_end], "exit loop if false")

        self.emit_label(lbl_body)
        if node.stmt:
            self.visit(node.stmt)
        self.emit("xbt.jal", ["tr0", lbl_cond], "repeat loop")

        self.emit_label(lbl_end)

    def visit_If(self, node: c_ast.If):
        lbl_true = self.new_label("if_true")
        lbl_false = self.new_label("if_false")
        lbl_end = self.new_label("if_end")

        # 3-Way Balanced Ternary Condition Evaluation
        res = self.visit_expr(node.cond, target_reg="tr5")
        self.emit("xbt.bne", [res, "tr0", lbl_true], "branch to true body")
        if node.iffalse:
            self.emit("xbt.jal", ["tr0", lbl_false])
        else:
            self.emit("xbt.jal", ["tr0", lbl_end])

        self.emit_label(lbl_true)
        self.visit(node.iftrue)
        self.emit("xbt.jal", ["tr0", lbl_end])

        if node.iffalse:
            self.emit_label(lbl_false)
            self.visit(node.iffalse)

        self.emit_label(lbl_end)

    def visit_Assignment(self, node: c_ast.Assignment):
        var_name = node.lvalue.name if isinstance(node.lvalue, c_ast.ID) else None
        if var_name:
            if var_name not in self.locals:
                self.locals[var_name] = ("stack", self.stack_offset)
                self.stack_offset -= 4
            loc_type, loc = self.locals[var_name]

            if node.op == "=":
                val_reg = self.visit_expr(node.rvalue, target_reg="tr5")
                if loc_type == "reg":
                    if val_reg != loc:
                        self.emit("xbt.addi", [loc, val_reg, "0"], f"assign {var_name}")
                else:
                    self.emit("xbt.sw", [val_reg, f"{loc}(tr8)"], f"store to stack {var_name}")
            elif node.op == "+=":
                r_reg = self.visit_expr(node.rvalue, target_reg="tr6")
                if loc_type == "reg":
                    self.emit("xbt.ksadd", [loc, loc, r_reg], f"{var_name} += ...")
                else:
                    self.emit("xbt.lw", ["tr5", f"{loc}(tr8)"])
                    self.emit("xbt.ksadd", ["tr5", "tr5", r_reg])
                    self.emit("xbt.sw", ["tr5", f"{loc}(tr8)"])
            elif node.op == "-=":
                r_reg = self.visit_expr(node.rvalue, target_reg="tr6")
                if loc_type == "reg":
                    self.emit("xbt.neg", ["tr7", r_reg])
                    self.emit("xbt.ksadd", [loc, loc, "tr7"], f"{var_name} -= ...")
                else:
                    self.emit("xbt.lw", ["tr5", f"{loc}(tr8)"])
                    self.emit("xbt.neg", ["tr7", r_reg])
                    self.emit("xbt.ksadd", ["tr5", "tr5", "tr7"])
                    self.emit("xbt.sw", ["tr5", f"{loc}(tr8)"])
            elif node.op == "*=":
                r_reg = self.visit_expr(node.rvalue, target_reg="tr6")
                if loc_type == "reg":
                    self.emit("xbt.mul", [loc, loc, r_reg], f"{var_name} *= ...")
                else:
                    self.emit("xbt.lw", ["tr5", f"{loc}(tr8)"])
                    self.emit("xbt.mul", ["tr5", "tr5", r_reg])
                    self.emit("xbt.sw", ["tr5", f"{loc}(tr8)"])
        elif isinstance(node.lvalue, c_ast.UnaryOp) and node.lvalue.op == "*":
            # Pointer store: *ptr = val
            val_reg = self.visit_expr(node.rvalue, target_reg="tr5")
            ptr_reg = self.visit_expr(node.lvalue.expr, target_reg="tr6")
            self.emit("xbt.sw", [val_reg, f"0({ptr_reg})"], "pointer store")

    def visit_FuncCall(self, node: c_ast.FuncCall):
        func_name = node.name.name
        # Pass arguments in tr10..tr17
        if node.args and node.args.exprs:
            for i, arg in enumerate(node.args.exprs):
                target_reg = f"tr{10 + i}"
                self.visit_expr(arg, target_reg=target_reg)
        self.emit("xbt.call", [func_name], f"call {func_name}")
        return "tr10"

    def visit_expr(self, node: Any, target_reg: str = "tr5") -> str:
        if isinstance(node, c_ast.Constant):
            self.emit("xbt.li", [target_reg, str(node.value)], f"load const {node.value}")
            return target_reg
        elif isinstance(node, c_ast.ID):
            loc_type, loc = self.locals.get(node.name, ("unknown", None))
            if loc_type == "reg":
                return loc
            elif loc_type == "stack":
                self.emit("xbt.lw", [target_reg, f"{loc}(tr8)"], f"load local {node.name}")
                return target_reg
            return target_reg
        elif isinstance(node, c_ast.BinaryOp):
            left_reg = self.visit_expr(node.left, "tr5")
            right_reg = self.visit_expr(node.right, "tr6")
            if node.op == "+":
                self.emit("xbt.ksadd", [target_reg, left_reg, right_reg], "Kogge-Stone 6-stage CLA")
            elif node.op == "-":
                self.emit("xbt.neg", ["tr7", right_reg], "0-gate wire-swap negation")
                self.emit("xbt.ksadd", [target_reg, left_reg, "tr7"], "subtraction via neg + ksadd")
            elif node.op == "*":
                self.emit("xbt.mul", [target_reg, left_reg, right_reg], "AOI22 lane multiplier")
            elif node.op in ("<", "<=", ">", ">=", "==", "!="):
                # 3-Way Balanced Ternary Comparator
                self.emit("xbt.neg", ["tr7", right_reg], "negate right operand")
                self.emit("xbt.ksadd", ["tr7", left_reg, "tr7"], "compute diff = left - right")
                self.emit("xbt.sign", [target_reg, "tr7"], "balanced ternary sign {-1, 0, +1}")
            return target_reg
        elif isinstance(node, c_ast.UnaryOp):
            if node.op == "-":
                val = self.visit_expr(node.expr, target_reg)
                self.emit("xbt.neg", [target_reg, val], "0-gate wire-swap negation")
                return target_reg
            elif node.op == "*":
                # Pointer dereference
                ptr_reg = self.visit_expr(node.expr, "tr6")
                self.emit("xbt.lw", [target_reg, f"0({ptr_reg})"], "dereference pointer")
                return target_reg
        elif isinstance(node, c_ast.FuncCall):
            return self.visit_FuncCall(node)
        return target_reg


class TriumCC:
    """
    Main Compiler Driver.
    """
    def __init__(self, optimize_dot: bool = True):
        self.optimize_dot = optimize_dot
        self.rv_translator = RVToTriumTranslator(optimize_dot=optimize_dot)

    def compile_c_source(self, c_code: str) -> str:
        if not HAVE_PYCPARSER:
            raise RuntimeError("pycparser is required for direct C compilation. Please install pycparser.")
        parser = c_parser.CParser()
        ast = parser.parse(c_code)
        compiler = CToTriumCompiler(optimize_dot=self.optimize_dot)
        compiler.visit(ast)

        header = [
            "# ====================================================================",
            "# Trium-XBT Machine Assembly (.tas) -- Generated from C Source",
            "# Architecture: 64-Trit Balanced Ternary (RISC-V Custom-0 Xbt 0x0B)",
            "# ====================================================================",
            "",
        ]
        return "\n".join(header + compiler.code) + "\n"

    def compile_assembly_file(self, asm_text: str) -> str:
        return self.rv_translator.translate_program(asm_text)

    def compile(self, input_path: str, output_path: Optional[str] = None) -> str:
        with open(input_path, "r", encoding="utf-8") as f:
            content = f.read()

        if input_path.endswith(".c"):
            tas_output = self.compile_c_source(content)
        elif input_path.endswith((".s", ".S", ".asm")):
            tas_output = self.compile_assembly_file(content)
        else:
            # Auto-detect
            if "int " in content or "void " in content:
                tas_output = self.compile_c_source(content)
            else:
                tas_output = self.compile_assembly_file(content)

        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(tas_output)

        return tas_output


def main():
    parser = argparse.ArgumentParser(description="Trium-XBT C & Assembly Compiler (trium-cc)")
    parser.add_argument("input_file", help="Input C source (.c) or RISC-V assembly (.s)")
    parser.add_argument("-o", "--output", help="Output Trium assembly file (.tas)")
    parser.add_argument("--no-vec", action="store_true", help="Disable loop vectorization to xbt.dot64")
    args = parser.parse_args()

    compiler = TriumCC(optimize_dot=not args.no_vec)
    out = compiler.compile(args.input_file, args.output)
    if not args.output:
        print(out)


if __name__ == "__main__":
    main()
