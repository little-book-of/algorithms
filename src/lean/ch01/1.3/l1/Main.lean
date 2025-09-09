import Ch01_1_3_L1.GcdLcm
import Ch01_1_3_L1.ModularIdentities
import Ch01_1_3_L1.Fractions
import Ch01_1_3_L1.Exercises

open Ch01_1_3_L1

def main : IO Unit := do
  -- GCD / LCM
  let a : Int := 252; let b : Int := 198
  IO.println s!"gcd({a}, {b}) = {gcdInt a b}"
  IO.println s!"lcm(12, 18) = {lcmInt 12 18}"

  -- Modular identities
  let x : Int := 123; let y : Int := 456; let m : Nat := 7
  IO.println s!"(x+y)%m vs modAdd: {((x + y) |> fun s => modNorm s m)} {modAdd x y m}"
  IO.println s!"(x*y)%m vs modMul: {(((x.natAbs % m) * (y.natAbs % m)) % m)} {modMul x y m}"

  -- Fraction reduction
  let r := reduceFraction 84 126
  IO.println s!"reduceFraction(84,126) = {r.num}/{r.den}"

  -- Exercises
  IO.println ""
  runExercises