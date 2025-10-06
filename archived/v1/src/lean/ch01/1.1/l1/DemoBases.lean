namespace Ch01_1_1_L1

def toOct (n : Nat) : String :=
  String.mk (Nat.toDigits 8 n)

def toHex (n : Nat) : String :=
  (String.mk (Nat.toDigits 16 n)).toUpper

def demoBases : IO Unit := do
  let n := 42
  IO.println s!"Decimal: {n}"
  IO.println s!"Octal  : {toOct n}"
  IO.println s!"Hex    : {toHex n}"

  -- Erlang/C/Go/Python have literals, Lean does not,
  -- so we simulate by parsing:
  let octLit := String.mk (Nat.toDigits 10 (Nat.ofDigits 8 ['5','2']))
  let hexLit := String.mk (Nat.toDigits 10 (Nat.ofDigits 16 ['2','A']))
  IO.println s!"Octal literal 52 -> {octLit}"
  IO.println s!"Hex literal 2A -> {hexLit}"
