import data.real.basic
import data.real.nnreal
import Mathlib.Tactic

-- EVOLVE-BLOCK-START
theorem abelian_group (G : Type _) [Group G] (a b : G) (H : Subgroup G := Subgroup.closure ({a, b} : Set G))
-- Assumptions
(hab : a * b = b * a)
:
-- Conjecture
(∀ x y : G, x ∈ H → y ∈ H → x * y = y * x) := by
  classical
  -- Prove stronger statements by induction on the word representing elements of H
  -- Define the auxiliary lemmas showing a and b commute with every element of H
  have ha : ∀ y ∈ H, a * y = y * a := by
    have h := Subgroup.closure_induction (S := ({a, b} : Set G)) (P := fun z => a * z = z * a)
      (by
        intro z hz
        rcases hz with hz | hz
        · subst hz; simp
        · simpa [hab] )
      (by
        intro x y hx hy
        calc
          a * (x * y) = (a * x) * y := by simp [mul_assoc]
          _ = (x * a) * y := by simp [hx]
          _ = x * (a * y) := by simp [mul_assoc]
          _ = x * (y * a) := by simp [hy]
          _ = (x * y) * a := by simp [mul_assoc])
      (by
        intro x hx
        have hx' : a * x = x * a := hx
        calc
          a * (x⁻¹) = (x * a) * x⁻¹ := by
            -- derive from hx' by multiplying on the right by x⁻¹ and manipulating
            have := congrArg (fun t => t * x⁻¹) hx'
            simpa [mul_assoc] using this
          _ = x⁻¹ * a := by
            -- rearrange
            have := congrArg (fun t => x⁻¹ * t) hx'
            simpa [mul_assoc] using this)
      (by
        intro y hy
        -- base case for y ∈ H is covered by the previous step
        exact hy)
    exact h
  have hb : ∀ y ∈ H, b * y = y * b := by
    have h := Subgroup.closure_induction (S := ({a, b} : Set G)) (P := fun z => b * z = z * b)
      (by
        intro z hz
        rcases hz with hz | hz
        · subst hz; simp
        · simpa [hab] )
      (by
        intro x y hx hy
        calc
          b * (x * y) = (b * x) * y := by simp [mul_assoc]
          _ = (x * b) * y := by simp [hx]
          _ = x * (b * y) := by simp [mul_assoc]
          _ = x * (y * b) := by simp [hy]
          _ = (x * y) * b := by simp [mul_assoc])
      (by
        intro x hx
        have hx' : b * x = x * b := hx
        calc
          b * (x⁻¹) = (x * b) * x⁻¹ := by
            have := congrArg (fun t => t * x⁻¹) hx'
            simpa [mul_assoc] using this
          _ = x⁻¹ * b := by
            have := congrArg (fun t => x⁻¹ * t) hx'
            simpa [mul_assoc] using this)
      (by
        intro y hy
        exact hy)
    exact h
  -- Now prove by induction on elements of H that they commute with every other element of H
  have hcomm : ∀ x ∈ H, ∀ z ∈ H, x * z = z * x := by
    refine Subgroup.closure_induction (S := ({a, b} : Set G)) (P := fun x => ∀ z ∈ H, x * z = z * x)
      (by
        intro x hx z hz
        rcases hx with hx | hx
        · subst hx; simp [ha z hz]
        · subst hx; simp [hb z hz])
      (by
        intro x y hx hy hz
        -- use associativity to rearrange
        have hxz : x * z = z * x := hx z hz
        have hyr : y * z = z * y := hy z hz
        calc
          (x * y) * z = x * (y * z) := by simp [mul_assoc]
          _ = x * (z * y) := by simpa [hyr]
          _ = (x * z) * y := by simp [mul_assoc]
          _ = (z * x) * y := by simpa [hxz]
          _ = z * (x * y) := by simp [mul_assoc]
      )
      (by
        intro x hx
        -- inverse case
        have hx' : ∀ z ∈ H, x * z = z * x := hx
        intro z hz
        have hx_inv : x⁻¹ * z = z * x⁻¹ := by
          have hxy : x * z = z * x := hx' z hz
          have := congrArg (fun t => t * x⁻¹) hxy
          simpa [mul_assoc] using this
        simpa [hx_inv])
      (by
        intro z hz
        exact hz)
  -- Finally, use the fact that hcomm holds to conclude the original goal
  intro x hx y hy
  have : x * y = y * x := hcomm x hx y hy
  simpa using this
-- EVOLVE-BLOCK-END
:=