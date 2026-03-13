import mathlib



theorem abelian_group {G : Type _} [Group G] (a b : G) (hab : a * b = b * a) :
  ∀ x y, x ∈ (Subgroup.closure ({a, b} : Set G)) → y ∈ (Subgroup.closure ({a, b} : Set G)) → x * y = y * x :=
by
  classical
  
  haveI : IsAbelian (Subgroup.closure ({a, b} : Set G)) := isAbelian_closure_of_comm hab
  intro x y hx hy
  
  have h := IsAbelian.mul_comm (a := (⟨x, hx⟩ : Subgroup.closure ({a, b} : Set G)))
                              (b := (⟨y, hy⟩ : Subgroup.closure ({a, b} : Set G)))
  
  have hG := congrArg Subtype.val h
  simpa using hG
