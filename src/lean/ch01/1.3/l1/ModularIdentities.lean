namespace Ch01_1_3_L1

/--
Normalize an integer `a` modulo a **positive** modulus `m`,
returning a natural remainder in `[0, m-1]`.
-/
def modNorm (a : Int) (m : Nat) : Nat :=
  if h : m = 0 then
    0
  else
    -- Lean's `Int.emod : Int → Nat → Nat` yields a Nat remainder.
    Int.emod a m

/-- (a + b) % m with normalization into `[0, m-1]`. -/
def modAdd (a b : Int) (m : Nat) : Nat :=
  (modNorm a m + modNorm b m) % m

/-- (a - b) % m with normalization into `[0, m-1]`. -/
def modSub (a b : Int) (m : Nat) : Nat :=
  let am := modNorm a m
  let bm := modNorm b m
  (am + (m + m) - bm) % m  -- avoid negative by over-adding m

/-- (a * b) % m with normalization into `[0, m-1]`. -/
def modMul (a b : Int) (m : Nat) : Nat :=
  (modNorm a m * modNorm b m) % m

end Ch01_1_3_L1