namespace Ch01_1_1_L2

/-- Show exact decimal arithmetic using rationals (ℚ) for contrast. -/
def decimalExactDemo : IO Unit := do
  -- Exact decimal strings modelled as rationals:
  let a : Rat := 1 / 10   -- 0.1
  let b : Rat := 1 / 5    -- 0.2
  let c : Rat := 3 / 10   -- 0.3
  let s := a + b
  IO.println s!"Rat 0.1 + 0.2 = {s} = {s.num}/{s.den}"
  IO.println s!"Equal to 0.3? {decide (s = c)}"

end Ch01_1_1_L2
