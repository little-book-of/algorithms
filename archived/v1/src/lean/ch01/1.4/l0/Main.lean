import Ch01_1_4_L0.IntOverflowSim
import Ch01_1_4_L0.FloatPrecision
import Ch01_1_4_L0.Exercises

open Ch01_1_4_L0

def main : IO Unit := do
  IO.println "=== Integer overflow (8-bit) ==="
  IO.println s!"255 + 1 (unsigned, wrap) -> {addUnsignedBits 255 1 8}"
  IO.println s!"127 + 1 (signed, wrap)   -> {addSignedBits 127 1 8}"

  let (rU, cf, of) := addWithFlags8 200 100
  IO.println s!"200 + 100 -> result={rU} (unsigned), CF={cf}, OF={of}"
  IO.println s!"Interpret result as signed: {toSignedBits rU 8}"

  IO.println "\n=== Floating-point surprises ==="
  let x : Float := 0.1 + 0.2
  IO.println s!"0.1 + 0.2 = {x}"  -- typically 0.30000000000000004
  IO.println s!"Direct equality with 0.3? {(x == 0.3)}"
  IO.println s!"Using epsilon: {almostEqual x 0.3 1e-9}"

  IO.println "\nRepeat add 0.1 ten times:"
  let s := repeatAdd 0.1 10
  IO.println s!"sum = {s}, eq1={(s == 1.0)}, almost={(almostEqual s 1.0 1e-9)}"

  IO.println "\nMix large and small:"
  let a := mixLargeSmall 1e16 1.0
  IO.println s!"1e16 + 1.0 = {a} (often unchanged)"

  IO.println "\nNaive sum may drift slightly:"
  let nums := List.replicate 10 (0.1 : Float)
  IO.println s!"sumNaive([0.1]*10) = {sumNaive nums}"

  -- Exercises (small showcase)
  IO.println ""
  let (e1a,e1b,e1c) := ex1
  IO.println s!"Exercise 1 (4-bit sequence from 14): {e1a} {e1b} {e1c}"
  let (sum, eq1, almost) := ex2
  IO.println s!"Exercise 2 (ten*0.1): sum={sum} eq1={eq1} almost={almost}"
  let (x3,y3) := ex3
  IO.println s!"Exercise 3 (signed adds): {x3} {y3}"
  IO.println s!"Exercise 4 (u16 65535+1): {ex4}"
