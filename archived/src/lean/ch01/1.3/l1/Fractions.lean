import Ch01_1_3_L1.GcdLcm

open Ch01_1_3_L1

namespace Ch01_1_3_L1

structure Fraction where
  num : Int
  den : Int
deriving Repr

/--
Reduce an integer fraction to lowest terms with **canonical sign**:
denominator strictly positive. Panics if `den = 0`.
-/
def reduceFraction (n d : Int) : Fraction :=
  if d = 0 then
    panic! "denominator cannot be zero"
  else
    let g : Nat := gcdInt n d
    -- divide integers by positive natural g (exact division as g | n, g | d)
    let gn : Int := Int.ofNat g
    let num' := n / gn
    let den' := d / gn
    if den' < 0 then
      { num := -num', den := -den' }
    else
      { num := num', den := den' }

end Ch01_1_3_L1