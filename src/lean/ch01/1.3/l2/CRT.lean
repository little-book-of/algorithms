import Ch01_1_3_L2.ExtGcdInv

open Ch01_1_3_L2

namespace Ch01_1_3_L2

/-- Combine x ≡ a1 (mod m1), x ≡ a2 (mod m2) assuming gcd(m1,m2)=1.
Returns (x, M=m1*m2) with 0 ≤ x < M. -/
def crtPair (a1 m1 a2 m2 : Nat) : Option (Nat × Nat) := do
  if m1 = 0 || m2 = 0 then
    none
  else
    -- Need inv(m1 mod m2, m2)
    let inv? := invMod? (Int.ofNat (m1 % m2)) m2
    let inv ← inv?
    let k := ((a2 + m2 - (a1 % m2)) % m2) * inv % m2
    let M := m1 * m2
    let x := (a1 + k * m1) % M
    some (x, M)

/-- Fold CRT over a list of pairwise-coprime congruences (ai, mi). -/
def crt (congs : List (Nat × Nat)) : Option (Nat × Nat) := do
  match congs with
  | [] => none
  | (a,m) :: rest =>
    let mut acc : Nat × Nat := (a, m)
    for (a2, m2) in rest do
      let (x, M) ← crtPair acc.fst acc.snd a2 m2
      acc := (x, M)
    some acc

end Ch01_1_3_L2