import Ch01_1_1_L1.DemoBases
import Ch01_1_1_L1.TwosComplement

open Ch01_1_1_L1

def runExercises : IO Unit := do
  -- Exercise 1: 100 -> octal, hex
  IO.println s!"Exercise 1 (100): oct={toOct 100} hex={toHex 100}"

  -- Exercise 2: -7 in 8-bit two’s complement
  IO.println s!"Exercise 2 (-7 in 8-bit): {toTwosComplement (-7) 8}"

  -- Exercise 3: verify 0xFF == 255
  let vFF := Nat.ofDigits 16 ['F','F']
  IO.println s!"Exercise 3 (0xFF == 255?): {vFF == 255}"

  -- Exercise 4: parse "11111001" as 8-bit two’s complement
  match fromTwosComplement "11111001" with
  | some v => IO.println s!"Exercise 4 (parse '11111001'): {v}"
  | none   => IO.println "Exercise 4: invalid input"
