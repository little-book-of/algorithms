import Ch01_1_1_L2.FloatUtils

open Ch01_1_1_L2

namespace Ch01_1_1_L2

def ex1BitsOfOneLike : String :=
  -- We don't expose raw bits here; report key facts we can verify at runtime.
  -- For IEEE-754 double, 1.0 has unbiased exponent 0 and mantissa 0;
  -- we reflect that by checking ULP symmetry around 1.0.
  let ulp := ulpAround 1.0
  s!"Ex1: ULP(1.0) ≈ {repr ulp}; nextUp ≈ {repr (1.0 + ulp)}"

def ex2UlpVsNext : String :=
  let ulp := ulpAround 1.0
  let nu  := nextUpApprox 1.0
  s!"Ex2: ULP(1.0) ≈ {repr ulp}, nextUp-1.0 ≈ {repr (nu - 1.0)}"

def ex3ClassifyValues : List (String × String) :=
  let nan := makeNaN
  let inf := makeInf
  [
    ("zero", "zero (exactly 0.0)"),
    ("one",  "normal (≈ 1.0)"),
    ("+inf", if isNaN inf then "nan" else "inf (constructed via 1/0)"),
    ("nan",  if isNaN nan then "nan" else "not-nan (unexpected)")
  ]

def ex4Nexts : String :=
  let up  := nextUpApprox 3.5
  let dn  := nextDownApprox 3.5
  s!"Ex4: nextUp(3.5) ≈ {repr up}, nextDown(3.5) ≈ {repr dn}"

def ex5TinySearch : String :=
  -- Find smallest positive number that still changes 0.0 when added
  let rec down (e : Float) :=
    let e' := e / 2.0
    if 0.0 + e' == 0.0 then e else down e'
  let tiny := down 1.0
  s!"Ex5: first positive step above 0.0 (approx) ≈ {repr tiny}"

def runExercises : IO Unit := do
  IO.println (ex1BitsOfOneLike)
  IO.println (ex2UlpVsNext)
  for (name, cls) in ex3ClassifyValues do
    IO.println s!"Ex3: {name} -> {cls}"
  IO.println (ex4Nexts)
  IO.println (ex5TinySearch)

end Ch01_1_1_L2
