import Ch01_1_1_L0.DecToBin
import Ch01_1_1_L0.BinToDec

open Ch01_1_1_L0

def runExercises : IO Unit := do
  -- Exercise 1: 19 -> binary
  IO.println s!"Exercise 1 (19 -> binary): {decToBin 19}"

  -- Exercise 2: "10101" -> decimal
  match binToDec "10101" with
  | some v => IO.println s!"Exercise 2 ('10101' -> dec): {v}"
  | none   => IO.println "Invalid input"

  -- Exercise 3: 27 -> binary
  IO.println s!"Exercise 3 (27 -> binary): {decToBin 27}"

  -- Exercise 4: verify "111111" == 63
  match binToDec "111111" with
  | some v => IO.println s!"Exercise 4: 111111 -> {v} == 63? {v == 63}"
  | none   => IO.println "Invalid input"

  -- Exercise 5: why binary
  IO.println "Exercise 5: Binary suits hardware with two stable states (0/1)."
