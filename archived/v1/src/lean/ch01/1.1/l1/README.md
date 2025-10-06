# ch01 / 1.1 / l1 — Beyond Binary: Octal, Hex, and Two’s Complement (Lean)

Examples for base representations and negative numbers.

## Files

* `lakefile.lean` — project configuration for Lake.
* `Main.lean` — demo runner: prints decimal, octal, hex, and exercises.
* `DemoBases.lean` — convert numbers to octal/hex strings.
* `TwosComplement.lean` — encode/decode integers in two’s complement form.
* `Exercises.lean` — small exercises to practice conversions.

## Build and Run

```
lake build
lake exe main
```

## Expected Output

```
Decimal: 42
Octal  : 52
Hex    : 2A
Octal literal 52 -> 42
Hex literal 2A -> 42
Exercise 1 (100): oct=144 hex=64
Exercise 2 (-7 in 8-bit): 11111001
Exercise 3 (0xFF == 255?): true
Exercise 4 (parse '11111001'): -7
```
