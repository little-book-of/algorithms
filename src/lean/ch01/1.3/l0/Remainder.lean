namespace Ch01_1_3_L0

/-- Return `(q, r)` with `n = d*q + r`. Requires `d > 0`. -/
def divIdentity (n d : Nat) : Nat × Nat :=
  if d = 0 then (0, 0)  -- L0: avoid exceptions; caller should pass d>0
  else (n / d, n % d)

/-- Wrap days 0..6 (0=Mon) after `shift` days. -/
def weekShift (start shift : Nat) : Nat :=
  (start + shift) % 7

/-- `(base^exp) % mod` via binary exponentiation (all `Nat`). -/
def powMod (base exp mod : Nat) : Nat :=
  if mod = 1 then 0 else
  let rec go (b e acc : Nat) : Nat :=
    if e = 0 then acc
    else
      let acc' := if e % 2 = 1 then (acc * b) % mod else acc
      go ((b * b) % mod) (e / 2) acc'
  go (base % mod) exp (1 % mod)

end Ch01_1_3_L0
