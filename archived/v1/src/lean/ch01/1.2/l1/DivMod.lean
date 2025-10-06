namespace Ch01_1_2_L1

/-- Print the identity n = d*q + r for a few pairs. -/
def demoDivMod : IO Unit := do
  let show (n d : Nat) : IO Unit := do
    let q := n / d
    let r := n % d
    IO.println s!"{n} = {d}*{q} + {r}"
  show 47 5
  show 23 7
  show 100 9

end Ch01_1_2_L1
