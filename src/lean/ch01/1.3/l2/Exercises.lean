import Ch01_1_3_L2.ExtGcdInv
import Ch01_1_3_L2.Totient
import Ch01_1_3_L2.MillerRabin
import Ch01_1_3_L2.CRT

open Ch01_1_3_L2

namespace Ch01_1_3_L2

/-- 1) modular inverse 7 mod 13 -> 2 -/
def ex1 : Option Nat := invMod? 7 13

/-- 2) φ(10) and Euler’s theorem check for a=3, n=10. -/
def ex2 : Nat × Bool :=
  let φ := phi 10
  let n := 10
  let a := 3 % n
  -- powMod on Nat
  let rec pow (b e : Nat) (m : Nat) : Nat :=
    if e = 0 then 1 % m
    else
      let acc := pow (b*b % m) (e/2) m
      if e % 2 = 1 then (a * pow (a*a % n) (φ/2) n) % n else acc
  -- simple binary exp specialized to (a^φ mod n)
  let rec pe (b e acc : Nat) : Nat :=
    if e = 0 then acc else
      let acc' := if e % 2 = 1 then (acc * b) % n else acc
      pe ((b*b) % n) (e/2) acc'
  let ok := pe a φ 1 = 1
  (φ, ok)

/-- 3) Fermat on 341 with base 2 -> 1 (Carmichael). -/
def ex3 : Nat :=
  let n := 341; let e := 340
  let rec pe (b e acc : Nat) : Nat :=
    if e = 0 then acc else
      let acc' := if e % 2 = 1 then (acc * b) % n else acc
      pe ((b*b) % n) (e/2) acc'
  pe (2 % n) e 1

/-- 4) Use Miller–Rabin on a small list. -/
def ex4 : List Nat :=
  [97, 341, 561, 569].filter isProbablePrime64

/-- 5) CRT: x≡1 (mod 4), x≡2 (mod 5), x≡3 (mod 7) -> 52 mod 140 -/
def ex5 : Option (Nat × Nat) := do
  let (x12, m12) ← crtPair 1 4 2 5
  crtPair x12 m12 3 7

def runExercises : IO Unit := do
  IO.println s!"Exercise 1 invmod(7,13): {ex1}"
  let (φ, ok) := ex2
  IO.println s!"Exercise 2 phi(10) & check: {φ} {ok}"
  IO.println s!"Exercise 3 Fermat 341 result: {ex3}"
  IO.println s!"Exercise 4 probable primes: {ex4}"
  IO.println s!"Exercise 5 CRT: {ex5}"

end Ch01_1_3_L2