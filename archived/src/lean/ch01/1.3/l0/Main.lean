import Ch01_1_3_L0.Parity
import Ch01_1_3_L0.Divisibility
import Ch01_1_3_L0.Remainder
import Ch01_1_3_L0.Exercises

open Ch01_1_3_L0

def main : IO Unit := do
  -- Parity
  IO.println "Parity:"
  for n in #[10, 7] do
    IO.println s!"  {n} is_even? {isEven n}  is_odd? {isOdd n}  (bit) {parityBit n}"

  -- Divisibility
  IO.println "\nDivisibility:"
  IO.println s!"  12 divisible by 3? {isDivisible 12 3}"
  IO.println s!"  14 divisible by 5? {isDivisible 14 5}"

  -- Remainders & identity
  IO.println "\nRemainders & identity:"
  let (q, r) := divIdentity 17 5
  IO.println s!"  17 = 5*{q} + {r}"

  -- Clock
  IO.println "\nClock (7-day week):"
  IO.println s!"  Saturday(5) + 10 days -> {weekShift 5 10}"

  -- Last digit of 2^15
  IO.println "\nLast digit of 2^15:"
  IO.println s!"  {powMod 2 15 10}"

  -- Exercises
  IO.println ""
  runExercises
