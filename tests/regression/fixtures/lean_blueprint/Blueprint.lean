set_option maxHeartbeats 400000

/-- Doubling, defined before the target so it counts as trusted context. -/
def double (n : Nat) : Nat := n + n

theorem double_zero : double 0 = 0 := rfl

/-- The user's own docstring, which must survive assembly untouched. -/
theorem blueprint_target (n : Nat) : double n + 0 = 2 * n := by
  sorry

theorem after_target : True := trivial
