namespace Ch01_1_3_L2

/-- Strip all factors p from x (Nat). -/
@[inline] def strip (x p : Nat) : Nat :=
  if p = 0 then x else
  let rec go (y : Nat) : Nat :=
    if y % p = 0 then go (y / p) else y
  go x

/-- φ(n) via trial factorization (didactic; OK for small/medium n). -/
def phi (n : Nat) : Nat :=
  if n = 0 then 0 else
  let rec loop (x res p : Nat) : Nat :=
    if p*p > x then
      if x > 1 then res - res / x else res
    else
      if p = 2 then
        if x % 2 = 0 then
          let x' := strip x 2
          loop x' (res - res / 2) 3
        else
          loop x res 3
      else
        if x % p = 0 then
          let x' := strip x p
          loop x' (res - res / p) (p+2)
        else
          loop x res (p+2)
  loop n n 2

end Ch01_1_3_L2