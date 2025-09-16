namespace Ch01_1_2_L1

/-- Hash bucket mapping: key ↦ [0, size). -/
def hashingExample (key size : Nat) : Nat :=
  key % size

/-- Cyclic weekday wrap (0–6). -/
def cyclicDay (start shift : Nat) : Nat :=
  (start + shift) % 7

/-- Modular exponentiation via square-and-multiply. -/
def modexp (a b m : Nat) : Nat :=
  let rec go (base : Nat) (e : Nat) (acc : Nat) : Nat :=
    if e = 0 then acc
    else
      let acc' := if e % 2 = 1 then (acc * base) % m else acc
      go ((base * base) % m) (e / 2) acc'
  go (a % m) b 1

def runModuloExamples : IO Unit := do
  IO.println s!"Hashing: key=1234, size=10 -> {hashingExample 1234 10}"
  IO.println s!"Cyclic: Saturday(5)+4 -> {cyclicDay 5 4}"
  IO.println s!"modexp(7,128,13) = {modexp 7 128 13}"

end Ch01_1_2_L1
