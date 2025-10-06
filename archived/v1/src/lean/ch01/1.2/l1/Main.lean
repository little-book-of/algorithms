import Ch01_1_2_L1.DivMod
import Ch01_1_2_L1.ModuloExamples
import Ch01_1_2_L1.EfficiencyDemo
import Ch01_1_2_L1.Exercises

open Ch01_1_2_L1

def main : IO Unit := do
  -- Show division identity examples
  demoDivMod

  -- Modulo use cases: hashing, cyclic day, modular exponent
  runModuloExamples

  -- Efficiency demo: power-of-two modulus note
  runEfficiencyDemo

  -- Exercises output
  runExercises
