import data.real.basic
import data.real.nnreal
import Mathlib.Tactic

-- EVOLVE-BLOCK-START
theorem abelian_group {G : Type*} [Group G] (a b : G)
-- Assumptions
(hab : a * b = b * a)
:
-- Conjecture
(∀ x y : G,
  x ∈ (Subgroup.closure ({a, b} : Set G)) →
  y ∈ (Subgroup.closure ({a, b} : Set G)) →
  x * y = y * x)
-- EVOLVE-BLOCK-END
:=