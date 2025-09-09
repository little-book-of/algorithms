namespace Ch01_1_2_L0

/-- Convert a natural number to its binary representation (no prefix). -/
partial def toBinAux (n : Nat) (acc : List Char) : List Char :=
  if n == 0 then acc
  else
    let bit : Char := if n % 2 == 0 then '0' else '1'
    toBinAux (n / 2) (bit :: acc)

def toBin (n : Nat) : String :=
  if n == 0 then "0" else (toBinAux n []).asString

/-- Small binary demo mirroring the text examples. -/
def demoBinary : IO Unit := do
  let x := 11  -- 0b1011
  let y := 6   -- 0b0110
  IO.println s!"x = {toBin x} (dec {x})"
  IO.println s!"y = {toBin y} (dec {y})"
  IO.println s!"x + y = {toBin (x + y)} (dec {x + y})"
  IO.println s!"x - y = {toBin (x - y)} (dec {x - y})"

end Ch01_1_2_L0
