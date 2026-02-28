namespace TestNamespace

/-- This is a simple function that adds two natural numbers.
    It demonstrates basic function definition with documentation. -/
def add (a b : Nat) : Nat :=
  a + b

/-- Multiplication of two natural numbers.
    Used for testing doc comment extraction. -/
def multiply (x y : Nat) : Nat :=
  x * y

def noDocComment (n : Nat) : Nat :=
  n + 1

/-- A theorem stating that addition is commutative.
    This is a simple example theorem. -/
theorem add_comm (a b : Nat) : add a b = add b a := by
  simp [add]
  omega

/-- Lemma about multiplication by zero. -/
lemma mul_zero (n : Nat) : multiply n 0 = 0 := by
  simp [multiply]

end TestNamespace
