import Ch01_1_3_L2.ExtGcdInv
import Ch01_1_3_L2.Totient
import Ch01_1_3_L2.MillerRabin
import Ch01_1_3_L2.CRT
import Ch01_1_3_L2.Exercises

open Ch01_1_3_L2

def main : IO Unit := do
  -- Extended Euclid / inverse
  let (g, x, y) := extendedGCD 240 46
  IO.println s!"extended_gcd(240,46) -> {(g, x, y)}"
  IO.println s!"invmod(3,7) = {invMod? 3 7}"

  -- Totient
  IO.println s!"phi(10) = {phi 10}"
  IO.println s!"phi(36) = {phi 36}"

  -- Miller–Rabin
  for n in [97, 561, 1105, (2147483647 : Nat)] do
    IO.println s!"is_probable_prime({n}) = {isProbablePrime64 n}"

  -- CRT: x≡2 (mod 3), x≡3 (mod 5), x≡2 (mod 7) -> 23 mod 105
  match crt [(2,3),(3,5),(2,7)] with
  | some (x, m) => IO.println s!"CRT -> {x} mod {m}"
  | none        => IO.println "CRT failed"

  -- Exercises
  IO.println ""
  runExercises