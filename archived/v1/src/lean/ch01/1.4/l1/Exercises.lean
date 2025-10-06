import Ch01_1_4_L1.IntArith
import Ch01_1_4_L1.FloatNum
import Ch01_1_4_L1.FixedPoint

open Ch01_1_4_L1

namespace Ch01_1_4_L1

/-- 1) Policy results for (i32 max + 1). -/
def ex1Policy : (Int × String × Int) :=
  let wrap := addI32Wrapping i32Max 1
  let chk  := match addI32Checked i32Max 1 with | none => "Overflow" | some _ => "OK"
  let sat  := addI32Saturating i32Max 1
  (wrap, chk, sat)

/-- 2) Float compare: 0.1 + 0.2 vs 0.3. -/
def ex2CompareFloats : (Float × Bool × Bool) :=
  let x := 0.1 + 0.2
  (x, x == 0.3, almostEqual x 0.3)

/-- 3) Pairwise vs Kahan on first 1e5 harmonic terms. -/
def ex3Summation : (Float × Float) :=
  let n := 100000
  let xs := (List.range n).map (fun i => 1.0 / (Float.ofNat (i+1)))
  (pairwiseSum xs, kahanSum xs)

/-- 4) Ledger transfer: 12.34 -> 0.34. -/
def ex4Ledger : (String × String) :=
  match (parseDollarsToCents "12.34", parseDollarsToCents "0.00", parseDollarsToCents "0.34") with
  | (Except.ok a, Except.ok b, Except.ok amt) =>
      let A := FixedPoint.Ledger.mk a
      let B := FixedPoint.Ledger.mk b
      match FixedPoint.Ledger.transfer A B amt with
      | Except.ok (A1, B1) => (formatCents A1.balanceCents, formatCents B1.balanceCents)
      | _ => ("ERR","ERR")
  | _ => ("ERR","ERR")

/-- Print exercises. -/
def runExercises : IO Unit := do
  let (w, chk, s) := ex1Policy
  IO.println s!"Exercise 1 (policy): wrap={w} checked={chk} sat={s}"
  let (sum, eq1, almost) := ex2CompareFloats
  IO.println s!"Exercise 2 (float compare): sum={sum} eq={eq1} almost={almost}"
  let (p, k) := ex3Summation
  IO.println s!"Exercise 3 (pairwise vs kahan): {p} {k}"
  let (aStr, bStr) := ex4Ledger
  IO.println s!"Exercise 4 (ledger): {aStr} {bStr}"

end Ch01_1_4_L1
