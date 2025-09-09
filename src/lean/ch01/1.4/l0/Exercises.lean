import Ch01_1_4_L0.IntOverflowSim
import Ch01_1_4_L0.FloatPrecision

open Ch01_1_4_L0

namespace Ch01_1_4_L0

/-- 1) 4-bit counter: start at 14, add 1 three times. Expect (15, 0, 1). -/
def ex1 : Nat × Nat × Nat :=
  let a := addUnsignedBits 14 1 4
  let b := addUnsignedBits a 1 4
  let c := addUnsignedBits b 1 4
  (a, b, c)

/-- 2) Ten times 0.1 vs 1.0 (exact? almost-equal?). -/
def ex2 : Float × Bool × Bool :=
  let s : Float := repeatAdd 0.1 10
  (s, s == 1.0, almostEqual s 1.0 1e-9)

/-- 3) 8-bit signed addition demos: 127+1 and -1+(-1). Expect (−128, −2). -/
def ex3 : Int × Int :=
  (addSignedBits 127 1 8, addSignedBits (-1) (-1) 8)

/-- 4) Unsigned 16-bit wrap for 65535 + 1 -> 0. -/
def ex4 : Nat :=
  addUnsignedBits 65535 1 16

end Ch01_1_4_L0
