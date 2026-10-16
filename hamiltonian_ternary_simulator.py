"""
Hamiltonian Symplectic Arithmetic & Reversible Phase-Space Simulator
Author: Antigravity Prime Research Engine

Demonstrating:
  1. Conjugate Phase-Space Representation (Q = Sum, P = Carry)
  2. Zero-Ripple Intermediate Accumulation: O(1) step latency
  3. Symplectic Reversibility: Reconstructing any past state without storing intermediate memory!
  4. Final Carry Assimilation: H(Q, P) = Q + 3*P
"""

import random

class HamiltonianTernaryAccumulator:
    """
    Maintains accumulation in conjugate phase space (Q, P):
      Q: Sum array (analogous to generalized position)
      P: Carry array (analogous to generalized momentum)
    Hamiltonian Invariant: H(Q, P) = sum(Q[i] * 3^i) + 3 * sum(P[i] * 3^i)
    """
    def __init__(self, width: int = 16):
        self.width = width
        self.Q = [0] * width  # Position (Sums)
        self.P = [0] * width  # Momentum (Carries)

    def energy(self) -> int:
        """Returns the conserved Hamiltonian energy (the total numerical value)."""
        val_Q = sum(t * (3 ** i) for i, t in enumerate(self.Q))
        val_P = sum(t * (3 ** i) for i, t in enumerate(self.P))
        return val_Q + 3 * val_P

    def forward_leapfrog_step(self, X: list[int]):
        """
        Discrete Symplectic Leapfrog Step:
          Step 1 (Momentum / Carry update): P' = P + carry(Q, X)
          Step 2 (Position / Sum update):     Q' = sum(Q, X)
        NO CARRY PROPAGATION! Pure local O(1) operations.
        """
        # 1. Update Momentum (Carries)
        new_P = list(self.P)
        for i in range(min(len(X), self.width)):
            s = self.Q[i] + X[i]
            if s >= 2:
                new_P[i] += 1
            elif s <= -2:
                new_P[i] -= 1

        # 2. Update Position (Sums)
        new_Q = list(self.Q)
        for i in range(min(len(X), self.width)):
            s = self.Q[i] + X[i]
            if s == 2:
                new_Q[i] = -1
            elif s == -2:
                new_Q[i] = 1
            elif s == 1:
                new_Q[i] = 1
            elif s == -1:
                new_Q[i] = -1
            elif s == 0:
                new_Q[i] = 0

        self.P = new_P
        self.Q = new_Q

    def backward_leapfrog_step(self, X: list[int]):
        """
        Time-Reversal Step:
        Reconstructs the EXACT prior state (Q_prev, P_prev) without needing memory storage!
        """
        # Reverse Step 2 (Recover Q_prev from Q and X)
        # Note: (Q_prev + X) mapped to Q. Invertible because:
        # if Q_prev was in {-1, 0, 1} and X in {-1, 0, 1}, algebraic total was Q + 3*c.
        prev_Q = [0] * self.width
        carry_generated = [0] * self.width

        for i in range(min(len(X), self.width)):
            # Search candidate in {-1, 0, 1}
            found = False
            for cand in [-1, 0, 1]:
                s = cand + X[i]
                expected_q = -1 if s == 2 else (1 if s == -2 else s)
                if expected_q == self.Q[i]:
                    prev_Q[i] = cand
                    c = 1 if s >= 2 else (-1 if s <= -2 else 0)
                    carry_generated[i] = c
                    found = True
                    break
            assert found, "Reversible step failed to find pre-image"

        # Reverse Step 1 (Recover P_prev from P and carry_generated)
        prev_P = list(self.P)
        for i in range(self.width):
            prev_P[i] -= carry_generated[i]

        self.Q = prev_Q
        self.P = prev_P

    def final_carry_assimilation(self) -> int:
        """Only at the very end of computation, resolve (Q, P) into a single scalar."""
        return self.energy()


def run_tests():
    print("=== Testing Hamiltonian Symplectic Accumulation ===")
    random.seed(42)
    acc = HamiltonianTernaryAccumulator(width=20)
    
    # Generate a stream of 100 ternary inputs
    inputs = []
    expected_running_sums = []
    total = 0
    for _ in range(100):
        # 8-trit random vector
        vec = [random.choice([-1, 0, 1]) for _ in range(8)] + [0] * 12
        vec_val = sum(t * (3 ** i) for i, t in enumerate(vec))
        total += vec_val
        inputs.append(vec)
        expected_running_sums.append(total)

    # 1. Forward Leapfrog Accumulation (O(1) local steps, NO ripple-carry)
    print("Running 100 forward leapfrog steps in phase space (Q, P)...")
    for i, vec in enumerate(inputs):
        acc.forward_leapfrog_step(vec)
        # Verify Hamiltonian energy invariant at every step
        assert acc.energy() == expected_running_sums[i], f"Energy invariant broken at step {i}"
    
    final_val = acc.final_carry_assimilation()
    print(f"PASS: Final accumulated value H(Q, P) = {final_val} matches reference sum {total} exactly!")

    # 2. Time-Reversal Reconstructability (ZERO intermediate memory storage)
    print("\nRunning 100 backward leapfrog steps (time reversal)...")
    for i in reversed(range(100)):
        # At step i, state should match running sum at i
        assert acc.energy() == expected_running_sums[i]
        acc.backward_leapfrog_step(inputs[i])

    # Should return to initial state (0, 0)
    assert acc.energy() == 0
    assert all(q == 0 for q in acc.Q)
    assert all(p == 0 for p in acc.P)
    print("PASS: Time reversal returned to initial state (0, 0) bit-for-bit!")
    print("ZERO memory was used to store intermediate states!")
    print("\n=== ALL HAMILTONIAN SYMPLECTIC TESTS PASSED ===")

if __name__ == "__main__":
    run_tests()
