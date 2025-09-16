namespace Ch01_1_4_L1

/-- 2^n as `Int`. -/
@[inline] def twoPow (n : Nat) : Int := Int.ofNat (Nat.pow 2 n)

/-- Wrap an `Int` to signed `bits`-width (two's complement). -/
def wrapToSigned (bits : Nat) (x : Int) : Int :=
  let m   : Int := twoPow bits
  let r   : Int := x % m
  let r'  : Int := if r < 0 then r + m else r
  let sign : Int := twoPow (bits - 1)
  if r' ≥ sign then r' - m else r'

/-- i32 bounds. -/
def i32Min : Int := -twoPow 31
def i32Max : Int := twoPow 31 - 1

/-- i64 bounds. -/
def i64Min : Int := -twoPow 63
def i64Max : Int := twoPow 63 - 1

/-! ### i32 ops -/

/-- Checked add (i32): `some s` if in range, else `none`. -/
def addI32Checked (a b : Int) : Option Int :=
  let s := a + b
  if s < i32Min || s > i32Max then none else some s

/-- Wrapping add (i32): two's complement. -/
def addI32Wrapping (a b : Int) : Int :=
  wrapToSigned 32 (a + b)

/-- Saturating add (i32): clamp to bounds. -/
def addI32Saturating (a b : Int) : Int :=
  let s := a + b
  if   s > i32Max then i32Max
  else if s < i32Min then i32Min
  else s

/-- Checked mul (i32). -/
def mulI32Checked (a b : Int) : Option Int :=
  let p := a * b
  if p < i32Min || p > i32Max then none else some p

/-! ### i64 ops -/

def addI64Checked (a b : Int) : Option Int :=
  let s := a + b
  if s < i64Min || s > i64Max then none else some s

def addI64Wrapping (a b : Int) : Int :=
  wrapToSigned 64 (a + b)

def addI64Saturating (a b : Int) : Int :=
  let s := a + b
  if   s > i64Max then i64Max
  else if s < i64Min then i64Min
  else s

def mulI64Checked (a b : Int) : Option Int :=
  let p := a * b
  if p < i64Min || p > i64Max then none else some p

end Ch01_1_4_L1
