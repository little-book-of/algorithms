namespace Ch01_1_3_L0

/-- True iff `n` is even (Nat-based). -/
def isEven (n : Nat) : Bool :=
  n % 2 == 0

/-- True iff `n` is odd. -/
def isOdd (n : Nat) : Bool :=
  n % 2 == 1

/-- "even" / "odd" using the last binary bit. -/
def parityBit (n : Nat) : String :=
  if (n &&& 1) == 1 then "odd" else "even"

end Ch01_1_3_L0
