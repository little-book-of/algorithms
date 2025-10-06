namespace Ch01_1_2_L2

/-- Bit length for `Nat`:
- `0` ↦ `0`
- `n>0` ↦ `Nat.log2 n + 1`
This mirrors the usual definition: floor(log2(n)) + 1. -/
def bitLen (n : Nat) : Nat :=
  if n = 0 then 0 else Nat.log2 n + 1

/-- Power of two helper. -/
@[inline] def pow2 (m : Nat) : Nat := Nat.pow 2 m

/-- Split `x` at `m` low bits (i.e., base `2^m`):
returns `(high, low)` with `x = high*2^m + low`, `0 ≤ low < 2^m`. -/
def splitAt (x m : Nat) : Nat × Nat :=
  let p := pow2 m
  (x / p, x % p)

/-- Karatsuba multiplication on natural numbers.
For small values we fall back to builtin multiplication to terminate recursion. -/
partial def karatsuba (x y : Nat) : Nat :=
  if x < 10 || y < 10 then
    x * y
  else
    let n  := Nat.max (bitLen x) (bitLen y)
    let m  := n / 2
    let (xh, xl) := splitAt x m
    let (yh, yl) := splitAt y m
    let z0 := karatsuba xl yl
    let z2 := karatsuba xh yh
    let z1 := karatsuba (xl + xh) (yl + yh) - z0 - z2
    let p  := pow2 m
    z2 * (p * p) + z1 * p + z0

end Ch01_1_2_L2
