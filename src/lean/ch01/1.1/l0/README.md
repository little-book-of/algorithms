# ch01 / 1.1 / l0 — Decimal and Binary Basics (Lean)

Examples for decimal ↔ binary basics.

## Files

* `lakefile.lean` — project configuration for Lake.
* `Main.lean` — demo runner: decimal to binary, binary to decimal, and exercises.
* `DecToBin.lean` — convert natural number → binary string.
* `BinToDec.lean` — convert binary string → natural number.
* `Exercises.lean` — small exercises using the conversion functions.

## Build and Run

```
lake build
lake exe main
```

## Expected Output

```
Decimal: 42
Binary : 101010
Binary string '1011' -> decimal: 11
Exercise 1 (19 -> binary): 10011
Exercise 2 ('10101' -> dec): 21
Exercise 3 (27 -> binary): 11011
Exercise 4: 111111 -> 63 == 63? true
Exercise 5: Binary suits hardware with two stable states (0/1).
```
