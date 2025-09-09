namespace Ch01_1_4_L0

/-- True if |x - y| < eps (simple beginner-safe comparison). -/
def almostEqual (x y eps : Float) : Bool :=
  Float.abs (x - y) < eps

/-- Add `value` to 0.0 `times` times (shows tiny drift for values like 0.1). -/
def repeatAdd (value : Float) (times : Nat) : Float :=
  let rec go (i : Nat) (acc : Float) : Float :=
    match i with
    | 0     => acc
    | i'+1  => go i' (acc + value)
  go times 0.0

/-- Adding a tiny number to a huge one may have no effect (limited precision). -/
def mixLargeSmall (large small : Float) : Float :=
  large + small

/-- Naive left-to-right sum; useful to demonstrate accumulated error. -/
def sumNaive (xs : List Float) : Float :=
  xs.foldl (fun acc v => acc + v) 0.0

end Ch01_1_4_L0
