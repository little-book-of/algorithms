namespace Ch01_1_1_L1

/-- Encode an integer into a two’s complement bitstring of length `bits`. -/
def toTwosComplement (x : Int) (bits : Nat) : String :=
  if bits = 0 then "" else
  let mask : Nat := if bits >= 64 then (2 ^ 64 - 1) else (2 ^ bits - 1)
  let u := (x.toNat) &&& mask
  let rec loop (n : Nat) (b : Nat) (acc : List Char) : List Char :=
    if b = 0 then acc
    else
      let bit := if n % 2 = 0 then '0' else '1'
      loop (n / 2) (b - 1) (bit :: acc)
  String.mk (loop u bits [])

/-- Decode a two’s complement bitstring back to an integer. -/
def fromTwosComplement (s : String) : Option Int :=
  let bits := s.length
  let chars := s.data
  let mut val : Nat := 0
  for c in chars do
    if c = '0' then val := val * 2
    else if c = '1' then val := val * 2 + 1
    else return none
  -- check sign bit
  if s.front? = some '1' then
    let full := (1 : Int) <<< bits
    some (val.toInt - full)
  else
    some val.toInt
