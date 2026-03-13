import data.real.basic
import data.real.nnreal
import Mathlib.Tactic

-- EVOLVE-BLOCK-START
theorem abelian_group {G : Type _} [Group G] (a b : G) (hab : a * b = b * a) :
  ∀ x y, x ∈ (Subgroup.closure ({a, b} : Set G)) → y ∈ (Subgroup.closure ({a, b} : Set G)) → x * y = y * x :=
by
  sorry
-- EVOLVE-BLOCK-END
:=