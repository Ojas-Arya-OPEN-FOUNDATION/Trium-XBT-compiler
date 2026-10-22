//===-- TernaryDialect.td - Balanced Ternary Dialect Definition -*- tablegen -*-===//
//
// Part of the Trium-XBT Foundational Compiler Infrastructure.
// Grounded in Lean 4 formal proofs and Z3 SMT bit-level verification.
//
//===----------------------------------------------------------------------===//
//
// This file defines the `bt` (Balanced Ternary) MLIR dialect and its core type
// system, modeling dual-rail bitplanes, redundant carry-save accumulators,
// and high-dimensional balanced ternary tensors.
//
//===----------------------------------------------------------------------===//

#ifndef BT_DIALECT_TD
#define BT_DIALECT_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"

//===----------------------------------------------------------------------===//
// Balanced Ternary Dialect Definition
//===----------------------------------------------------------------------===//

def BT_Dialect : Dialect {
  let name = "bt";
  let summary = "Balanced Ternary Dialect for Hardware-Agnostic Ternary Computing";
  let description = [{
    The `bt` (Balanced Ternary) dialect introduces first-class language and
    intermediate representation primitives for radix-3 balanced computation
    over the symmetric residue digit set $\{-1, 0, +1\}$.

    Key Architectural Principles:
    1. **Dual-Rail Bitplane Representation**: Every ternary word of width $W$
       is encoded into two parallel binary bitplanes: Positive rail ($P$) and
       Negative rail ($N$), satisfying the hardware invariant $P \land N = 0$.
    2. **Sign-Free 0-Cycle Inversion**: Negation corresponds to swapping the
       $P$ and $N$ registers, taking zero execution cycles via SSA renaming.
    3. **Carry-Free Carry-Save Accumulation (BT-CSA)**: Multi-operand additions
       and matrix inner products maintain redundant $(Sum, Carry)$ pairs in
       registers, eliminating carry ripple stalls in computational loops.
    4. **Division-Free Base-243 Memory Compaction**: 5 trits are compressed into
       1 byte (saving ~70% bandwidth) and unpacked in 1 cycle via magic fixed-point
       reciprocal multipliers without runtime integer division.
  }];

  let cppNamespace = "::mlir::bt";
  let useDefaultTypePrinterParser = 1;
}

//===----------------------------------------------------------------------===//
// Base Class for BT Types
//===----------------------------------------------------------------------===//

class BT_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<BT_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

//===----------------------------------------------------------------------===//
// 1. Primitive Trit Type: !bt.trit
//===----------------------------------------------------------------------===//

def BT_TritType : BT_Type<"Trit", "trit"> {
  let summary = "A single balanced ternary digit in {-1, 0, +1}";
  let description = [{
    The `!bt.trit` type represents a single scalar balanced ternary digit:
    - Neg (-1, often written as `-` or `T`)
    - Zero ( 0, written as `0`)
    - Pos (+1, written as `+` or `1`)

    In binary hardware lowering, a trit occupies 2 physical bits in dual-rail
    format ($p, n \in \{0, 1\}$ with $p \land n = 0$).
  }];
}

//===----------------------------------------------------------------------===//
// 2. Dual-Rail Bitplane Register Type: !bt.dualrail<width>
//===----------------------------------------------------------------------===//

def BT_DualRailType : BT_Type<"DualRail", "dualrail"> {
  let summary = "Bit-sliced dual-rail register representing width trits";
  let description = [{
    The `!bt.dualrail<width>` type represents a register or vector containing
    `width` balanced trits partitioned into two parallel bitplanes:
    - Positive plane `P`: bit $k$ is 1 iff the $k$-th trit is $+1$.
    - Negative plane `N`: bit $k$ is 1 iff the $k$-th trit is $-1$.

    Hardware invariant:
    $$\forall k \in [0, \text{width}-1]: P[k] \land N[k] = 0$$

    Common widths include 32 (SIMD single-lane), 64 (standard register),
    128 (NEON/SSE), 256 (AVX2), and 512 (AVX-512).
  }];

  let parameters = (ins "unsigned":$width);
  let assemblyFormat = "`<` $width `>`";

  let builders = [
    TypeBuilderWithInferredContext<(ins "unsigned":$width), [{
      return $_get($_ctxt, width);
    }]>
  ];
}

//===----------------------------------------------------------------------===//
// 3. Redundant Carry-Save Accumulator Type: !bt.csa_pair<width>
//===----------------------------------------------------------------------===//

def BT_CSAPairType : BT_Type<"CSAPair", "csa_pair"> {
  let summary = "Redundant Carry-Save Accumulator (Sum, Carry) pair";
  let description = [{
    The `!bt.csa_pair<width>` type encapsulates redundant carry-save accumulator
    state comprising:
    - `S`: Sum dual-rail vector of width `width` (!bt.dualrail<width>)
    - `C`: Carry dual-rail vector of width `width` (!bt.dualrail<width>)

    Maintaining $(S, C)$ in registers enables $O(1)$ constant-time accumulation
    latency across arbitrary loop depths without carry propagation stalls.
  }];

  let parameters = (ins "unsigned":$width);
  let assemblyFormat = "`<` $width `>`";

  let builders = [
    TypeBuilderWithInferredContext<(ins "unsigned":$width), [{
      return $_get($_ctxt, width);
    }]>
  ];
}

//===----------------------------------------------------------------------===//
// 4. Balanced Ternary Tensor Type: !bt.tensor<shape, width>
//===----------------------------------------------------------------------===//

def BT_TensorType : BT_Type<"TernaryTensor", "tensor"> {
  let summary = "Multi-dimensional tensor of dual-rail or Base-243 packed trits";
  let description = [{
    The `!bt.tensor<shape, width>` type represents an N-dimensional tensor of
    balanced ternary elements where each element contains `width` trits in
    dual-rail format (or Base-243 packed storage).

    Parameters:
    - `shape`: Dimension list defining tensor dimensions (e.g. 1024x1024).
    - `width`: Bitplane width per unit element (e.g. 64 trits per register block).
  }];

  let parameters = (ins
    ArrayRefParameter<"int64_t">:$shape,
    "unsigned":$width
  );

  let assemblyFormat = "`<` custom<DimensionList>($shape) `,` $width `>`";
}

//===----------------------------------------------------------------------===//
// Type Predicates and Constraints
//===----------------------------------------------------------------------===//

def BT_AnyDualRail : Type<CPred<"$_self.isa<::mlir::bt::DualRailType>()">,
                          "any balanced ternary dualrail type">;

def BT_AnyCSAPair  : Type<CPred<"$_self.isa<::mlir::bt::CSAPairType>()">,
                          "any balanced ternary csa_pair type">;

def BT_AnyTensor   : Type<CPred<"$_self.isa<::mlir::bt::TernaryTensorType>()">,
                          "any balanced ternary tensor type">;

#endif // BT_DIALECT_TD
