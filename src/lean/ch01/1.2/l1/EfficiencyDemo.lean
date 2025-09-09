namespace Ch01_1_2_L1

/--
For a power-of-two modulus m = 2^k, machine-level `n % m` equals `n & (m-1)`.
Lean’s runtime doesn’t expose bitwise ops on `Nat` as `%` synonyms here,
so we print the arithmetic result and (informally) note the bitmask equivalence.
-/
def runEfficiencyDemo : IO Unit := do
  let cases := [5, 12, 20]
  for n in cases do
    let m := 8
    IO.println s!"{n} % {m} = {n % m} (bitmask n & 7 on machines)"

end Ch01_1_2_L1
