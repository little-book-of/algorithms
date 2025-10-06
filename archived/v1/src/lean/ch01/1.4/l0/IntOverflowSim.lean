namespace Ch01_1_4_L0

/-- 2^bits as Nat (bits ≥ 0). -/
@[inline] def twoPow (bits : Nat) : Nat :=
  Nat.pow 2 bits

/-- Unsigned addition with wraparound modulo 2^bits. -/
def addUnsignedBits (x y bits : Nat) : Nat :=
  let m := twoPow bits
  if m = 0 then (x + y) else (x + y) % m

/-- Interpret an unsigned value (0..2^bits-1) as signed two’s-complement (Int). -/
def toSignedBits (value bits : Nat) : Int :=
  if bits = 0 then 0
  else
    let m    := twoPow bits
    let sign := twoPow (bits - 1)
    let v    := value % m
    if v ≥ sign then (Int.ofNat v) - (Int.ofNat m) else Int.ofNat v

/-- Signed addition with wrap at `bits` using two’s-complement. -/
def addSignedBits (x y : Int) (bits : Nat) : Int :=
  if bits = 0 then 0
  else
    let m := twoPow bits
    -- normalize inputs to unsigned residues
    let norm (z : Int) : Nat := (Int.toNat ((z % (Int.ofNat m) + (Int.ofNat m)) % (Int.ofNat m)))
    let raw : Nat := (norm x + norm y) % m
    toSignedBits raw bits

/-- 8-bit add returning (resultU8, CF, OF).
CF: unsigned carry (sum ≥ 256). OF: signed overflow (two positives -> negative OR two negatives -> non-negative). -/
def addWithFlags8 (x y : Nat) : Nat × Bool × Bool :=
  let rU : Nat := addUnsignedBits x y 8
  let cf       : Bool := x + y ≥ 256
  let sx : Int := toSignedBits x 8
  let sy : Int := toSignedBits y 8
  let sr : Int := toSignedBits rU 8
  let of : Bool :=
    ((sx ≥ 0 ∧ sy ≥ 0 ∧ sr < 0) ∨ (sx < 0 ∧ sy < 0 ∧ sr ≥ 0))
  (rU, cf, of)

end Ch01_1_4_L0
