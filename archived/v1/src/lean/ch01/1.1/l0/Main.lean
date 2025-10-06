import Ch01_1_1_L0.DecToBin
import Ch01_1_1_L0.BinToDec
import Ch01_1_1_L0.Exercises

def main : IO Unit := do
  let n := 42
  IO.println s!"Decimal: {n}"
  IO.println s!"Binary : {decToBin n}"

  let s := "1011"
  match binToDec s with
  | some v => IO.println s!"Binary string '{s}' -> decimal: {v}"
  | none   => IO.println s!"Invalid binary string: {s}"

  runExercises
