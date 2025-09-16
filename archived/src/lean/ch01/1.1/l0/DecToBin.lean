namespace Ch01_1_1_L0

partial def decToBinAux (n : Nat) (acc : List Char) : List Char :=
  if n == 0 then acc
  else
    let bit := if n % 2 == 0 then '0' else '1'
    decToBinAux (n / 2) (bit :: acc)

def decToBin (n : Nat) : String :=
  if n == 0 then "0"
  else (decToBinAux n []).asString

end Ch01_1_1_L0
