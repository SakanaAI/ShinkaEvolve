import mathlib



theorem abelian_group {G : Type _} [Group G] (a b : G) (hab : a * b = b * a) :
  let H : Subgroup G := Subgroup.closure ({a, b} : Set G) in
  
  ∀ x y : H, (x : G) * (y : G) = (y : G) * (x : G) :=
by
  classical
  intro x y
  
  
  
  
  have hcomm : IsCommutative (fun p q : G => (p * q) = (q * p)) := by
    have : a * b = b * a := hab
    
    simpa using Subgroup.closure_pairwise_comm (S := ({a, b} : Set G)) hab
  
  simpa using hcomm x y

:=