namespace Ch01_1_3_L2

/-- (base^exp) mod n for Nat using fast exponentiation. -/
def powMod (base exp n : Nat) : Nat :=
  if n = 1 then 0 else
  let rec go (b e acc : Nat) : Nat :=
    if e = 0 then acc
    else
      let acc' := if e % 2 = 1 then (acc * b) % n else acc
      go ((b * b) % n) (e / 2) acc'
  go (base % n) exp (1 % n)

/-- Decompose n-1 = d * 2^s with d odd (n≥2). -/
def decompose (n : Nat) : Nat × Nat :=
  let mut d := n - 1
  let mut s := 0
  while d % 2 = 0 do
    d := d / 2
    s := s + 1
  (d, s)

/-- One Miller–Rabin round; `a` is the base. Returns true if `a` witnesses compositeness. -/
def tryComposite (a n : Nat) : Bool :=
  let (d, s) := decompose n
  let mut x := powMod a d n
  if x = 1 || x = n - 1 then
    false
  else
    let mut i := 1
    let mut x2 := x
    let mut witness := true
    while i < s do
      x2 := (x2 * x2) % n
      if x2 = n - 1 then
        witness := false
        i := s
      else
        i := i + 1
    witness

/-- Deterministic Miller–Rabin for 64-bit range using bases {2,3,5,7,11,13,17}. -/
def isProbablePrime64 (n : Nat) : Bool :=
  if n < 2 then false else
  let small : List Nat := [2,3,5,7,11,13,17,19,23,29]
  if small.contains n then true else
  if small.any (fun p => n % p = 0) then false else
  let bases := [2,3,5,7,11,13,17]
  not <| bases.any (fun a =>
    let a' := a % n
    a' ≠ 0 && tryComposite a' n)

end Ch01_1_3_L2