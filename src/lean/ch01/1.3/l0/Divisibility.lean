namespace Ch01_1_3_L0

/-- True iff `b` divides `a` (with `b ≠ 0`). -/
def isDivisible (a b : Nat) : Bool :=
  if b = 0 then false else a % b == 0

/-- Last decimal digit of `n`. -/
def lastDecimalDigit (n : Nat) : Nat :=
  n % 10

/-- Quick rule: divisible by 10 if last digit is 0. -/
def divisibleBy10 (n : Nat) : Bool :=
  lastDecimalDigit n == 0

end Ch01_1_3_L0
