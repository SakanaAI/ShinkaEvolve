import mathlib


theorem abelian_group {G : Type*} [Group G] (a b : G)
(hab : a * b = b * a) :
(∀ x y : G,
  x ∈ (Subgroup.closure ({a, b} : Set G)) →
  y ∈ (Subgroup.closure ({a, b} : Set G)) →
  x * y = y * x) := by
  classical
  intro x y hx hy
  
  
  
  have hcomm : x * y = y * x := by
    
    
    
    
    
    
    admit
  exact hcomm