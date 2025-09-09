namespace Ch01_1_4_L1

/-- Parse "123", "123.4", "123.45", "-0.07" to **cents** (Int).
Truncates extra fractional digits beyond 2 by policy. -/
def parseDollarsToCents (s0 : String) : Except String Int := Id.run do
  let s := s0.trim
  if s.isEmpty then
    throw "empty money literal"
  -- sign
  let (sign, rest) :=
    if s.front = '-' then (-1, s.drop 1)
    else if s.front = '+' then (1, s.drop 1)
    else (1, s)
  if rest.isEmpty then
    throw "bad money literal"
  -- split on dot
  let parts := rest.split (· == '.')
  let wholeStr := parts.getD 0 ""
  if !wholeStr.all Char.isDigit then throw "bad whole part"
  let fracStr0 := parts.getD 1 ""
  if !(fracStr0.all Char.isDigit || fracStr0.isEmpty) then
    throw "bad fractional part"
  let fracStr :=
    if fracStr0.length = 0 then "00"
    else if fracStr0.length = 1 then fracStr0.push '0'
    else fracStr0.extract 0 2   -- truncate beyond 2
  -- to numbers
  let some wholeNat := wholeStr.toNat? | throw "whole parse fail"
  let some fracNat  := fracStr.toNat?  | throw "frac parse fail"
  let cents : Int := (Int.ofNat wholeNat) * 100 + Int.ofNat fracNat
  return sign * cents

/-- Format cents to "[-]W.DD". -/
def formatCents (c : Int) : String :=
  let sign := if c < 0 then "-" else ""
  let a := Int.natAbs c
  let dollars := a / 100
  let frac := a % 100
  let fracStr := if frac < 10 then s!"0{frac}" else s!"{frac}"
  s!"{sign}{dollars}.{fracStr}"

structure Ledger where
  balanceCents : Int
deriving Repr

namespace Ledger

def mk (cents : Int) : Ledger := { balanceCents := cents }

def deposit (L : Ledger) (cents : Int) : Ledger :=
  { L with balanceCents := L.balanceCents + cents }

/-- Withdraw; returns error if insufficient funds. -/
def withdraw (L : Ledger) (cents : Int) : Except String Ledger := do
  if cents > L.balanceCents then
    throw "insufficient funds"
  return { L with balanceCents := L.balanceCents - cents }

/-- Transfer cents; rolls back on failure. -/
def transfer (from to : Ledger) (cents : Int) : Except String (Ledger × Ledger) := do
  let from1 ← from.withdraw cents
  let to1 := to.deposit cents
  return (from1, to1)

end Ledger

end Ch01_1_4_L1
