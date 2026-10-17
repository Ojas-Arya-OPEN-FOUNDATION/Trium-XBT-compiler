// Structural parse-error tests: bt-opt must reject malformed input with a
// nonzero exit status instead of silently running on canned data. Each RUN
// builds the bad input with printf into the runner sandbox (%t) and requires
// the tool invocation to fail (shell '!' negation).
//
// RUN: printf 'module @bad_unknown_op {\n  func.func @f() {\n    %%x = bt.nosuchop %%a : i64\n    return\n  }\n}\n' > %t && ! %bt-opt %t

// RUN: printf 'module @bad_braces {\n  func.func @f() {\n' > %t && ! %bt-opt %t

// RUN: ! %bt-opt --convert-all < /dev/null

// RUN: ! %bt-opt --emit-llvm-ir < /dev/null
