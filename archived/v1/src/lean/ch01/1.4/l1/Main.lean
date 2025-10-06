import Ch01_1_4_L1.IntArith
import Ch01_1_4_L1.FloatNum
import Ch01_1_4_L1.FixedPoint
import Ch01_1_4_L1.Exercises

open Ch01_1_4_L1

def main : IO Unit := do
  IO.println "=== Integers: checked / wrapping / saturating ==="
  IO.println s!"int32 max: {i32Max}"
  match addI32Checked i32Max 1 with
  | none   => IO.println "checked add (i32): overflow trapped ✔"
  | some _ => IO.println "checked add (i32): unexpected OK"
  IO.println s!"wrapping add (i32): {addI32Wrapping i32Max 1}"   -- -> i32Min
  IO.println s!"saturating add (i32): {addI32Saturating i32Max 1}"

  IO.println s!"wrapping add (i64): {addI64Wrapping (i64Max - 41) 100}"
  IO.println s!"saturating add (i64): {addI64Saturating (i64Max - 100) 200}"

  IO.println "\n=== Floats: compare, sum, epsilon ==="
  let x := (0.1 + 0.2)
  IO.println s!"0.1 + 0.2 = {x}"
  IO.println s!"Direct equality with 0.3? {x == 0.3}"
  IO.println s!"almostEqual(x, 0.3): {almostEqual x 0.3}"

  let n := 100000
  let arr := (List.range n).map (fun i => 1.0 / (Float.ofNat (i+1)))
  IO.println s!"pairwise_sum: {pairwiseSum arr}"
  IO.println s!"kahan_sum   : {kahanSum arr}"

  IO.println s!"machine epsilon: {machineEpsilon}"
  IO.println s!"ULP(1.0, nextAfterUp(1.0)) = {ulpDiff 1.0 (nextAfterUp 1.0)}"

  IO.println "\n=== Fixed-point money (int cents) ==="
  match (parseDollarsToCents "10.00", parseDollarsToCents "0.05", parseDollarsToCents "3.35") with
  | (Except.ok ca, Except.ok cdep, Except.ok c335) =>
      let A := FixedPoint.Ledger.mk ca
      let B := FixedPoint.Ledger.mk 0
      match FixedPoint.Ledger.transfer A B c335 with
      | Except.ok (A1, B1) =>
          IO.println s!"A: {formatCents A1.balanceCents} B: {formatCents B1.balanceCents}"
          let A2 := FixedPoint.Ledger.deposit A1 cdep
          IO.println s!"A after 5 cents deposit: {formatCents A2.balanceCents}"
      | _ => IO.println "transfer failed"
  | _ => IO.println "parse failed"

  IO.println "\n=== Exercises ==="
  runExercises
