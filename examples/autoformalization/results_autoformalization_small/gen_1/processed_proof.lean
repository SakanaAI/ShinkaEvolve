import mathlib



theorem abelian_group {G : Type _} [Group G] (a b : G) :
  (a * b = b * a) →
  let H := Subgroup.closure ({a, b} : Set G) in
  ∀ x y : G, x ∈ H → y ∈ H → x * y = y * x
:= by
  classical
  intro hcomm
  let H := Subgroup.closure ({a, b} : Set G)
  
  
  have hAb : IsAbelian H := by
    
    simpa using Subgroup.closure_isAbelian_of_commute (S := ({a, b} : Set G)) hcomm
  intro x y hx hy
  have hx' : (⟨x, hx⟩ : H) * ⟨y, hy⟩ = ⟨y, hy⟩ * ⟨x, hx⟩ := by
    simpa using hAb.mul_comm ⟨x, hx⟩ ⟨y, hy⟩
  simpa using hx'
