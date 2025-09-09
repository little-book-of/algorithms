import Ch01_1_1_L2.FloatUtils
import Ch01_1_1_L2.DecimalExact
import Ch01_1_1_L2.Exercises

open Ch01_1_1_L2

def main : IO Unit := do
  -- Rounding surprise
  let a := (0.1 : Float) + 0.2
  IO.println s!"0.1 + 0.2 = {repr a}"
  IO.println s!"Equal to 0.3? {(a == (0.3 : Float))}"

  -- NaN and Inf (via IEEE-754 behavior at runtime)
  let nan := makeNaN
  let pinf := makeInf
  IO.println s!"NaN == NaN? {(nan == nan)}"
  IO.println s!"Infinity example: {repr pinf}"

  -- ULP around 1.0 (search-based)
  let ulp1 := ulpAround 1.0
  let nup  := nextUpApprox 1.0
  let ndn  := nextDownApprox 1.0
  IO.println s!"ULP(1.0): {repr ulp1}"
  IO.println s!"nextUp(1.0) ≈ {repr nup}"
  IO.println s!"nextDown(1.0) ≈ {repr ndn}"
  IO.println s!"(nextUp - 1.0): {repr (nup - 1.0)}"

  -- Exact decimal demo with rationals
  decimalExactDemo

  -- Exercises output
  runExercises
