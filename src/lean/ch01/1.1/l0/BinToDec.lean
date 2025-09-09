namespace Ch01_1_1_L0

def binToDec (s : String) : Option Nat :=
  let rec go (chars : List Char) (acc : Nat) : Option Nat :=
    match chars with
    | []      => some acc
    | '0'::cs => go cs (acc * 2)
    | '1'::cs => go cs (acc * 2 + 1)
    | _::cs   => none
  go s.data 0

end Ch01_1_1_L0
