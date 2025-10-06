namespace Ch01_1_1_L2

/-- Return an IEEE-754 NaN value (by 0/0). -/
def makeNaN : Float :=
  (0.0 : Float) / 0.0

/-- Return +∞ (by 1/0). -/
def makeInf : Float :=
  (1.0 : Float) / 0.0

/-- Heuristic NaN test: NaN ≠ NaN by IEEE-754. -/
def isNaN (x : Float) : Bool :=
  x != x

/-- Find the spacing (ULP) in + direction near `x` by halving search.
    Stops when adding half no longer changes the sum. -/
def ulpAround (x : Float) : Float :=
  let rec loop (e : Float) : Float :=
    let e' := e / 2.0
    if x + e' == x then e else loop e'
  loop 1.0

/-- Approximate next representable above `x` as `x + ulpAround x`. -/
def nextUpApprox (x : Float) : Float :=
  x + ulpAround x

/-- Approximate next representable below `x` as `x - ulpAround x`.
    Note: this is symmetric only near numbers where spacing is roughly even. -/
def nextDownApprox (x : Float) : Float :=
  x - ulpAround x

end Ch01_1_1_L2
