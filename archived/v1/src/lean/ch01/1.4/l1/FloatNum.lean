namespace Ch01_1_4_L1

/-- Robust float comparison: |x-y| ≤ max(rel*max(|x|,|y|), abs). -/
def almostEqual (x y : Float) (rel := 1e-12) (abs := 1e-12) : Bool :=
  let diff := Float.abs (x - y)
  let ax   := Float.abs x
  let ay   := Float.abs y
  let tol  := Float.max (rel * Float.max ax ay) abs
  diff ≤ tol

/-- Kahan (compensated) summation. -/
def kahanSum (xs : List Float) : Float :=
  let rec go (s c : Float) : List Float → Float
    | []      => s
    | x :: xs =>
      let y := x - c
      let t := s + y
      let c' := (t - s) - y
      go t c' xs
  go 0.0 0.0 xs

/-- One pairwise "round" combining adjacent pairs. -/
private def pairOnce : List Float → List Float
  | []            => []
  | [a]           => [a]
  | a :: b :: xs  => (a + b) :: pairOnce xs

/-- Pairwise/tree summation (stable, parallel-friendly). -/
partial def pairwiseSum (xs : List Float) : Float :=
  match xs with
  | []      => 0.0
  | [a]     => a
  | _       => pairwiseSum (pairOnce xs)

/-- Machine epsilon: smallest eps with 1.0 + eps ≠ 1.0. -/
partial def machineEpsilon : Float :=
  let rec go (eps : Float) :=
    if (1.0 + eps / 2.0) == 1.0 then eps else go (eps / 2.0)
  go 1.0

/-- ULP distance between two floats (IEEE-754 binary64). -/
def ulpDiff (a b : Float) : UInt64 :=
  let ua := a.toBits
  let ub := b.toBits
  let signMask : UInt64 := (1 : UInt64) <<< 63
  let norm (u : UInt64) : UInt64 :=
    if (u &&& signMask) ≠ 0 then signMask - u else u
  let na := norm ua
  let nb := norm ub
  if na ≥ nb then na - nb else nb - na

/-- Next representable float strictly greater than `x` (toward +∞). -/
def nextAfterUp (x : Float) : Float :=
  let u := x.toBits
  let signMask : UInt64 := (1 : UInt64) <<< 63
  let u' := if (u &&& signMask) = 0 then u + 1 else u - 1
  Float.ofBits u'

end Ch01_1_4_L1
