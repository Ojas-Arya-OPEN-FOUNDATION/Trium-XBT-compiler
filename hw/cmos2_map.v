// CMOS-2 (NAND, NOR, NOT) Gate Mapping Library
module \$_AND_ (input A, input B, output Y);
  wire n;
  \$_NAND_ _g (.A(A), .B(B), .Y(n));
  \$_NOT_ _i (.A(n), .Y(Y));
endmodule

module \$_OR_ (input A, input B, output Y);
  wire n;
  \$_NOR_ _g (.A(A), .B(B), .Y(n));
  \$_NOT_ _i (.A(n), .Y(Y));
endmodule

module \$_XOR_ (input A, input B, output Y);
  wire na, nb, nab;
  \$_NAND_ _g1 (.A(A), .B(B), .Y(nab));
  \$_NAND_ _g2 (.A(A), .B(nab), .Y(na));
  \$_NAND_ _g3 (.A(nab), .B(B), .Y(nb));
  \$_NAND_ _g4 (.A(na), .B(nb), .Y(Y));
endmodule
