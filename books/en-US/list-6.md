# Chapter 6. Mathematics for Algorithms 

# Section 51. Number Theory

### 501 Euclidean Algorithm

The Euclidean Algorithm is one of the oldest and most elegant algorithms in mathematics. It finds the greatest common divisor (gcd) of two integers, the largest number that divides both without leaving a remainder.

It's fast, simple, and forms the basis of modern number theory and cryptography.

#### What Problem Are We Solving?

We want to compute:

$$
\text{gcd}(a, b)
$$

That is, the largest integer$d$ such that$d \mid a$ and$d \mid b$.

Instead of checking every possible divisor, Euclid discovered a beautiful shortcut:

$$
\gcd(a, b) = \gcd(b, a \bmod b)
$$

Keep replacing the larger number by its remainder until one becomes zero. The last non-zero number is the gcd.

Example:

$$
\gcd(48, 18) = \gcd(18, 48 \bmod 18) = \gcd(18, 12) = \gcd(12, 6) = \gcd(6, 0) = 6
$$

So $gcd(48, 18) = 6$.

#### How Does It Work (Plain Language)?

Think of gcd like peeling layers of remainders.
Each step removes a chunk until nothing's left. The number that "survives" all remainders is the gcd.

Let's see it step by step:

| Step | a  | b  | a mod b |
| ---- | -- | -- | ------- |
| 1    | 48 | 18 | 12      |
| 2    | 18 | 12 | 6       |
| 3    | 12 | 6  | 0       |

When remainder becomes 0, stop. The other number (6) is the gcd.

Every step reduces the numbers quickly, so it runs in O(log min(a, b)) time, much faster than trying all divisors.

#### Tiny Code (Easy Versions)

C Version

```c
#include <stdio.h>

int gcd(int a, int b) {
    while (b != 0) {
        int r = a % b;
        a = b;
        b = r;
    }
    return a;
}

int main(void) {
    int a, b;
    printf("Enter a and b: ");
    scanf("%d %d", &a, &b);
    printf("gcd(%d, %d) = %d\n", a, b, gcd(a, b));
}
```

Python Version

```python
def gcd(a, b):
    while b != 0:
        a, b = b, a % b
    return a

a, b = map(int, input("Enter a and b: ").split())
print("gcd(", a, ",", b, ") =", gcd(a, b))
```

#### Why It Matters

- Core of modular arithmetic, number theory, and cryptography.
- Foundation for Extended Euclidean Algorithm (solving ax + by = gcd).
- Used in modular inverses and Chinese Remainder Theorem.
- Demonstrates algorithmic thinking: divide problem into smaller remainders.

#### A Gentle Proof (Why It Works)

If$a = bq + r$, then any divisor of$a$ and$b$ must also divide$r$.
So the set of common divisors of$(a, b)$ equals that of$(b, r)$.
Thus:

$$
\gcd(a, b) = \gcd(b, r)
$$

Repeatedly applying this equality until$r = 0$ reveals the gcd.

Because remainders shrink each step, the algorithm halts in at most$O(\log n)$ steps.

#### Try It Yourself

1. Compute gcd(84, 30) by hand.
2. Trace the steps of gcd(210, 45).
3. Modify code to count number of steps.
4. Try with large numbers (e.g. 123456, 789012).
5. Compare runtime with a naive divisor-checking method.

#### Test Cases

| a   | b   | Steps | gcd |
| --- | --- | ----- | --- |
| 48  | 18  | 3     | 6   |
| 84  | 30  | 3     | 6   |
| 210 | 45  | 3     | 15  |
| 101 | 10  | 2     | 1   |
| 270 | 192 | 5     | 6   |

#### Complexity

- Time: O(log min(a, b))
- Space: O(1) (iterative) or O(log n) (recursive)

The Euclidean Algorithm shows the power of simplicity: divide, reduce, and repeat, a timeless idea that still beats inside modern computation.

### 502 Extended Euclidean Algorithm

The Extended Euclidean Algorithm goes one step beyond the classic gcd: it not only finds the gcd(a, b) but also gives you x and y such that

$$
a \cdot x + b \cdot y = \gcd(a, b)
$$

These coefficients (x, y) are the key to solving Diophantine equations and finding modular inverses, which are essential in cryptography, modular arithmetic, and algorithm design.

#### What Problem Are We Solving?

We want to solve the linear equation:

$$
a x + b y = \gcd(a, b)
$$

Given integers a and b, we need integers x and y that satisfy this equation.
The Euclidean algorithm gives the gcd, but we can trace back its steps to express the gcd as a combination of a and b.

Example:
Find x, y such that
$$
240x + 46y = \gcd(240, 46)
$$

We know gcd(240, 46) = 2.
The Extended Euclidean Algorithm gives:
$$
2 = 240(-9) + 46(47)
$$
So (x = -9, y = 47).

#### How Does It Work (Plain Language)?

Think of it as "remembering" how each remainder was made during the gcd process.

In each step:
$$
a = bq + r \quad \Rightarrow \quad r = a - bq
$$

We recursively compute gcd(b, r), and when unwinding, we rewrite each remainder in terms of a and b.

Step-by-step for (240, 46):

| Step | a   | b  | a % b | Equation        |
| ---- | --- | -- | ----- | --------------- |
| 1    | 240 | 46 | 10    | 240 = 46×5 + 10 |
| 2    | 46  | 10 | 6     | 46 = 10×4 + 6   |
| 3    | 10  | 6  | 4     | 10 = 6×1 + 4    |
| 4    | 6   | 4  | 2     | 6 = 4×1 + 2     |
| 5    | 4   | 2  | 0     | Stop, gcd = 2   |

Now backtrack:

- 2 = 6 − 4×1
- 4 = 10 − 6×1
- 6 = 46 − 10×4
- 10 = 240 − 46×5

Substitute upward until gcd is in the form 240x + 46y.

#### Tiny Code (Easy Versions)

C Version

```c
#include <stdio.h>

int extended_gcd(int a, int b, int *x, int *y) {
    if (b == 0) {
        *x = 1;
        *y = 0;
        return a;
    }
    int x1, y1;
    int g = extended_gcd(b, a % b, &x1, &y1);
    *x = y1;
    *y = x1 - (a / b) * y1;
    return g;
}

int main(void) {
    int a, b, x, y;
    printf("Enter a and b: ");
    scanf("%d %d", &a, &b);
    int g = extended_gcd(a, b, &x, &y);
    printf("gcd(%d, %d) = %d\n", a, b, g);
    printf("Coefficients: x = %d, y = %d\n", x, y);
}
```

Python Version

```python
def extended_gcd(a, b):
    if b == 0:
        return a, 1, 0
    g, x1, y1 = extended_gcd(b, a % b)
    x = y1
    y = x1 - (a // b) * y1
    return g, x, y

a, b = map(int, input("Enter a and b: ").split())
g, x, y = extended_gcd(a, b)
print(f"gcd({a}, {b}) = {g}")
print(f"x = {x}, y = {y}")
```

#### Why It Matters

- Builds the modular inverse:$a^{-1} \equiv x \pmod{m}$ if gcd(a, m) = 1
- Solves linear Diophantine equations
- Used in RSA cryptography, CRT, and modular arithmetic
- Converts gcd into a linear combination of inputs

#### A Gentle Proof (Why It Works)

At each step:
$$
\gcd(a, b) = \gcd(b, a \bmod b)
$$
If we know $b x' + (a \bmod b) y' = \gcd(a, b)$,
and since $a \bmod b = a - b\lfloor a/b \rfloor$,
we can write:
$$
\gcd(a, b) = a y' + b(x' - \lfloor a/b \rfloor y')
$$

Set $x = y'$, $y = x' - \lfloor a/b \rfloor y'$.

This gives the recurrence for coefficients (x, y).

#### Try It Yourself

1. Solve (240x + 46y = 2) by hand.
2. Verify gcd(99, 78) = 3 and find x, y.
3. Find modular inverse of 3 mod 11 using extended gcd.
4. Modify the code to return only modular inverse.
5. Trace recursive calls for (99, 78).

#### Test Cases

| a   | b  | gcd | x   | y  | Check            |
| --- | -- | --- | --- | -- | ---------------- |
| 240 | 46 | 2   | -9  | 47 | 240(-9)+46(47)=2 |
| 99  | 78 | 3   | -11 | 14 | 99(-11)+78(14)=3 |
| 35  | 15 | 5   | 1   | -2 | 35(1)+15(-2)=5   |
| 7   | 5  | 1   | -2  | 3  | 7(-2)+5(3)=1     |

#### Complexity

- Time: O(log min(a, b))
- Space: O(log n) (recursion depth)

The Extended Euclidean Algorithm is the "memory" of gcd, not just finding the divisor, but showing how it's made.

### 503 Modular Addition

Modular addition is like wrapping numbers around a circle. Once you reach the end, you loop back to the start. This operation lies at the heart of modular arithmetic, the arithmetic of clocks, cryptography, and hashing.

When we add numbers under a modulus M, we always stay within the range [0, M−1].

#### What Problem Are We Solving?

We want to compute the sum of two numbers under modulo M:

$$
(a + b) \bmod M
$$

Example:
Let$a = 17, b = 12, M = 10$

$$
(17 + 12) \bmod 10 = 29 \bmod 10 = 9
$$

So we "wrap around" after every 10.

#### How Does It Work (Plain Language)?

Imagine a clock with M hours. If you move forward a hours, then b hours, where do you land?

You just add and take the remainder by M.

| a  | b  | M  | a + b | (a + b) mod M | Result |
| -- | -- | -- | ----- | ------------- | ------ |
| 17 | 12 | 10 | 29    | 9             | 9      |
| 8  | 7  | 5  | 15    | 0             | 0      |
| 25 | 25 | 7  | 50    | 1             | 1      |

It's addition with "wrap-around" when crossing multiples of M.

#### Tiny Code (Easy Versions)

C Version

```c
#include <stdio.h>

int mod_add(int a, int b, int M) {
    int res = (a % M + b % M) % M;
    if (res < 0) res += M; // handle negative values
    return res;
}

int main(void) {
    int a, b, M;
    printf("Enter a, b, M: ");
    scanf("%d %d %d", &a, &b, &M);
    printf("(a + b) mod M = %d\n", mod_add(a, b, M));
}
```

Python Version

```python
def mod_add(a, b, M):
    return (a % M + b % M) % M

a, b, M = map(int, input("Enter a, b, M: ").split())
print("(a + b) mod M =", mod_add(a, b, M))
```

#### Why It Matters

- Forms the base operation in modular arithmetic
- Used in cryptographic algorithms (RSA, Diffie-Hellman)
- Ensures no overflow in fixed-size computations
- Supports hashing, ring arithmetic, and checksum calculations

#### A Gentle Proof (Why It Works)

Let$a = Mq_1 + r_1$ and$b = Mq_2 + r_2$.
Then:
$$
a + b = M(q_1 + q_2) + (r_1 + r_2)
$$
So,
$$
(a + b) \bmod M = (r_1 + r_2) \bmod M = (a \bmod M + b \bmod M) \bmod M
$$
This is the modular addition property.

#### Try It Yourself

1. Compute (25 + 37) mod 12.
2. Compute (−3 + 5) mod 7.
3. Modify code to handle 3 numbers: (a + b + c) mod M.
4. Write modular subtraction: (a − b) mod M.
5. Explore pattern table for M = 5.

#### Test Cases

| a  | b  | M  | (a + b) mod M | Result |
| -- | -- | -- | ------------- | ------ |
| 17 | 12 | 10 | 9             | Ok      |
| 25 | 25 | 7  | 1             | Ok      |
| -3 | 5  | 7  | 2             | Ok      |
| 8  | 8  | 5  | 1             | Ok      |
| 0  | 9  | 4  | 1             | Ok      |

#### Complexity

- Time: O(1)
- Space: O(1)

Modular addition is arithmetic on a loop, every sum bends back into a finite world, keeping numbers tidy and elegant.

### 504 Modular Multiplication

Modular multiplication is arithmetic in a wrapped world, just like modular addition, but with repeated addition. When you multiply two numbers under a modulus M, you keep only the remainder, ensuring the result stays in [0, M−1].

It's the backbone of fast exponentiation, hashing, cryptography, and number-theoretic transforms.

#### What Problem Are We Solving?

We want to compute:

$$
(a \times b) \bmod M
$$

Example:
Let$a = 7, b = 8, M = 5$

$$
(7 \times 8) \bmod 5 = 56 \bmod 5 = 1
$$

So 7×8 "wraps" around the circle of size 5 and lands at 1.

#### How Does It Work (Plain Language)?

Think of modular multiplication as repeated modular addition:

$$
(a \times b) \bmod M = (a \bmod M + a \bmod M + \dots) \bmod M
$$
(repeat b times)

We can simplify it with this identity:
$$
(a \times b) \bmod M = ((a \bmod M) \times (b \bmod M)) \bmod M
$$

| a  | b  | M  | a×b | (a×b) mod M | Result |
| -- | -- | -- | --- | ----------- | ------ |
| 7  | 8  | 5  | 56  | 1           | 1      |
| 25 | 4  | 7  | 100 | 2           | 2      |
| 12 | 13 | 10 | 156 | 6           | 6      |

For large values, we avoid overflow by applying modular reduction at each step.

#### Tiny Code (Easy Versions)

C Version

```c
#include <stdio.h>

long long mod_mul(long long a, long long b, long long M) {
    a %= M;
    b %= M;
    long long res = (a * b) % M;
    if (res < 0) res += M; // handle negatives
    return res;
}

int main(void) {
    long long a, b, M;
    printf("Enter a, b, M: ");
    scanf("%lld %lld %lld", &a, &b, &M);
    printf("(a * b) mod M = %lld\n", mod_mul(a, b, M));
}
```

Python Version

```python
def mod_mul(a, b, M):
    return (a % M * b % M) % M

a, b, M = map(int, input("Enter a, b, M: ").split())
print("(a * b) mod M =", mod_mul(a, b, M))
```

For very large numbers, use modular multiplication by addition (to avoid overflow):

```python
def mod_mul_safe(a, b, M):
    res = 0
    a %= M
    while b > 0:
        if b % 2 == 1:
            res = (res + a) % M
        a = (2 * a) % M
        b //= 2
    return res
```

#### Why It Matters

- Essential in modular exponentiation and cryptography (RSA, Diffie-Hellman)
- Avoids overflow in arithmetic under modulus
- Core operation for Number Theoretic Transform (NTT) and hash functions
- Supports building modular inverses, power functions, and polynomial arithmetic

#### A Gentle Proof (Why It Works)

Let$a = q_1 M + r_1$,$b = q_2 M + r_2$.

Then:
$$
a \times b = M(q_1 b + q_2 a - q_1 q_2 M) + r_1 r_2
$$
So modulo M, all multiples of M vanish:
$$
(a \times b) \bmod M = (r_1 \times r_2) \bmod M
$$
Hence,
$$
(a \times b) \bmod M = ((a \bmod M) \times (b \bmod M)) \bmod M
$$

#### Try It Yourself

1. Compute (25 × 13) mod 7.
2. Try (−3 × 8) mod 5.
3. Modify code to handle negative inputs.
4. Write modular multiplication using repeated doubling (like binary exponentiation).
5. Create a table of (a×b) mod 6 for a, b ∈ [0,5].

#### Test Cases

| a          | b          | M          | (a×b) mod M | Result |
| ---------- | ---------- | ---------- | ----------- | ------ |
| 7          | 8          | 5          | 1           | Ok      |
| 25         | 4          | 7          | 2           | Ok      |
| -3         | 8          | 5          | 1           | Ok      |
| 12         | 13         | 10         | 6           | Ok      |
| 1000000000 | 1000000000 | 1000000007 | 49          | Ok      |

#### Complexity

- Time: O(1) (with built-in multiplication), O(log b) (with safe doubling)
- Space: O(1)

Modular multiplication keeps arithmetic stable and predictable, no matter how large the numbers, everything folds back neatly into the modular ring.

### 505 Modular Exponentiation

Modular exponentiation is the art of raising a number to a power under a modulus, efficiently.
Instead of multiplying (a) by itself (b) times (which would be far too slow), we square and reduce step by step.
This technique powers cryptography, hashing, number theory, and fast algorithms for huge exponents.

#### What Problem Are We Solving?

We want to compute:

$$
a^b \bmod M
$$

directly, without overflow and without looping (b) times.

Example:
Let$a = 3, b = 13, M = 7$

Naively:
$$
3^{13} = 1594323 \quad \Rightarrow \quad 1594323 \bmod 7 = 5
$$

Efficiently, we can do it with squaring:
$$
3^{13} \bmod 7 = ((3^8 \cdot 3^4 \cdot 3^1) \bmod 7) = 5
$$

#### How Does It Work (Plain Language)?

We use exponentiation by squaring.
Break the exponent into binary. For each bit, either square or square and multiply.

If (b) is even:
$$
a^b = (a^{b/2})^2
$$
If (b) is odd:
$$
a^b = a \cdot a^{b-1}
$$

Always take modulo after every multiplication to keep numbers small.

Example: $(a=3,\ b=13,\ M=7)$

| Step | b (binary) | a                                 | b  | Result  |
| ---- | ---------- | --------------------------------- | -- | ------- |
| 1    | 1101       | 3                                 | 13 | res = 1 |
| 2    | odd        | res = $(1\times3)\bmod7=3$, $a=(3\times3)\bmod7=2$, $b=6$ |    |         |
| 3    | even       | res = 3, $a=(2\times2)\bmod7=4$, $b=3$ |    |         |
| 4    | odd        | res = $(3\times4)\bmod7=5$, $a=(4\times4)\bmod7=2$, $b=1$ |    |         |
| 5    | odd        | res = $(5\times2)\bmod7=3$, $a=(2\times2)\bmod7=4$, $b=0$ |    |         |


Result = 3, which is $3^{13} \bmod 7$.

#### Tiny Code (Easy Versions)

C Version

```c
#include <stdio.h>

long long mod_pow(long long a, long long b, long long M) {
    long long res = 1;
    a %= M;
    while (b > 0) {
        if (b % 2 == 1)
            res = (res * a) % M;
        a = (a * a) % M;
        b /= 2;
    }
    return res;
}

int main(void) {
    long long a, b, M;
    printf("Enter a, b, M: ");
    scanf("%lld %lld %lld", &a, &b, &M);
    printf("%lld^%lld mod %lld = %lld\n", a, b, M, mod_pow(a, b, M));
}
```

Python Version

```python
def mod_pow(a, b, M):
    res = 1
    a %= M
    while b > 0:
        if b % 2 == 1:
            res = (res * a) % M
        a = (a * a) % M
        b //= 2
    return res

a, b, M = map(int, input("Enter a, b, M: ").split())
print(f"{a}^{b} mod {M} =", mod_pow(a, b, M))
```

Or simply:

```python
pow(a, b, M)
```

(Python's built-in pow handles this efficiently.)

#### Why It Matters

- Core operation in RSA, Diffie-Hellman, and ElGamal
- Needed for Fermat's Little Theorem and modular inverses
- Enables fast exponentiation without overflow
- Turns exponentiation from O(b) to O(log b)

#### A Gentle Proof (Why It Works)

Every exponent can be written in binary:
$$
b = \sum_{i=0}^{k} b_i \cdot 2^i
$$

Then:
$$
a^b = \prod_{i=0}^{k} (a^{2^i})^{b_i}
$$

We precompute $a^{2^i}$ by squaring, and multiply only where (b_i = 1).
Each squaring and multiplication is followed by modulo reduction, so numbers stay small.

#### Try It Yourself

1. Compute $2^{10} \bmod 1000$.
2. Compute $5^{117} \bmod 19$.
3. Modify the code to print each step (trace exponentiation).
4. Compare runtime with naive power.
5. Use pow(a, b, M) and verify same result.

#### Test Cases

| a  | b   | M    | Result | Check |
| -- | --- | ---- | ------ | ----- |
| 3  | 13  | 7    | 5      | Ok     |
| 2  | 10  | 1000 | 24     | Ok     |
| 5  | 117 | 19   | 1      | Ok     |
| 10 | 9   | 6    | 4      | Ok     |
| 7  | 222 | 13   | 9      | Ok     |

#### Complexity

- Time: O(log b)
- Space: O(1) (iterative) or O(log b) (recursive)

Modular exponentiation is how we tame huge powers, turning exponential growth into a fast, logarithmic dance under the modulus.

### 506 Modular Inverse

The modular inverse is the number that undoes multiplication under a modulus.
If $a \cdot x \equiv 1 \pmod{M}$, then (x) is called the modular inverse of (a) modulo (M).

It's the key to dividing in modular arithmetic, since division isn't directly defined, we multiply by an inverse instead.

#### What Problem Are We Solving?

We want to solve:

$$
a \cdot x \equiv 1 \pmod{M}
$$

That means find $x$ such that:

$$
(a \times x) \bmod M = 1
$$

This $x$ is called the modular multiplicative inverse of $a$ modulo $M$.

Example:

Find the inverse of $3 \pmod{11}$.

We need:

$$
3 \cdot x \equiv 1 \pmod{11}
$$

Try $x = 4$:

$$
3 \cdot 4 = 12 \equiv 1 \pmod{11}
$$

Therefore:

$$
3^{-1} \equiv 4 \pmod{11}
$$


#### How Does It Work (Plain Language)?

There are two main ways to find the modular inverse:

1. Extended Euclidean Algorithm (works for all when gcd(a, M) = 1)
2. Fermat's Little Theorem (works if M is prime)

##### 1. Extended Euclidean Method

We solve:

$$
a x + M y = 1
$$

The coefficient $x \bmod M$ is the modular inverse.

Example:

Find the inverse of $3 \bmod 11$.

Use the extended Euclidean algorithm:

$$
\begin{aligned}
11 &= 3 \times 3 + 2 \\
3  &= 2 \times 1 + 1 \\
2  &= 1 \times 2 + 0
\end{aligned}
$$

Backtrack:

$$
\begin{aligned}
1 &= 3 - 2 \times 1 \\
  &= 3 - (11 - 3 \times 3) \\
  &= 4 \times 3 - 1 \times 11
\end{aligned}
$$

So $x = 4$.

Hence,

$$
3^{-1} \equiv 4 \pmod{11}
$$


##### 2. Fermat's Little Theorem (Prime M)

If $M$ is prime and $a \not\equiv 0 \pmod{M}$, then:

$$
a^{M-1} \equiv 1 \pmod{M}
$$

Multiply both sides by $a^{-1}$:

$$
a^{M-2} \equiv a^{-1} \pmod{M}
$$

So the modular inverse is:

$$
a^{-1} = a^{M-2} \bmod M
$$

Example:

$$
3^{-1} \pmod{11} = 3^{9} \bmod 11 = 4
$$


#### Tiny Code (Easy Versions)

C Version (Extended GCD)

```c
#include <stdio.h>

long long extended_gcd(long long a, long long b, long long *x, long long *y) {
    if (b == 0) {
        *x = 1;
        *y = 0;
        return a;
    }
    long long x1, y1;
    long long g = extended_gcd(b, a % b, &x1, &y1);
    *x = y1;
    *y = x1 - (a / b) * y1;
    return g;
}

long long mod_inverse(long long a, long long M) {
    long long x, y;
    long long g = extended_gcd(a, M, &x, &y);
    if (g != 1) return -1; // no inverse if gcd ≠ 1
    x = (x % M + M) % M;
    return x;
}

int main(void) {
    long long a, M;
    printf("Enter a, M: ");
    scanf("%lld %lld", &a, &M);
    long long inv = mod_inverse(a, M);
    if (inv == -1)
        printf("No inverse exists.\n");
    else
        printf("Inverse of %lld mod %lld = %lld\n", a, M, inv);
}
```

Python Version

```python
def mod_inverse(a, M):
    def extended_gcd(a, b):
        if b == 0:
            return a, 1, 0
        g, x1, y1 = extended_gcd(b, a % b)
        x = y1
        y = x1 - (a // b) * y1
        return g, x, y

    g, x, y = extended_gcd(a, M)
    if g != 1:
        return None
    return x % M

a, M = map(int, input("Enter a, M: ").split())
inv = mod_inverse(a, M)
print("Inverse:", inv if inv is not None else "None")
```

Python (Prime Modulus with Fermat)

```python
def mod_inverse_prime(a, M):
    return pow(a, M - 2, M)
```

#### Why It Matters

- Enables division in modular arithmetic
- Critical in RSA, CRT, Elliptic Curves, and hashing
- Used to solve equations like $a x \equiv b \pmod{M}$
- Basis for solving linear congruences and modular systems

#### A Gentle Proof (Why It Works)

If $\gcd(a, M) = 1$,
By Bézout's identity:
$$
a x + M y = 1
$$
Taking modulo M:
$$
a x \equiv 1 \pmod{M}
$$
Thus, (x) is the modular inverse of (a).

#### Try It Yourself

1. Find $5^{-1} \pmod{7}$.
2. Find $10^{-1} \pmod{17}$.
3. Check which numbers have no inverse mod $8$.
4. Implement both Extended GCD and Fermat versions.
5. Solve $7x \equiv 3 \pmod{13}$ using an inverse.

#### Test Cases

| a  | M  | Inverse | Check                          |
| -- | -- | ------- | ------------------------------ |
| 3  | 11 | 4       | $3 \times 4 = 12 \equiv 1$     |
| 5  | 7  | 3       | $5 \times 3 = 15 \equiv 1$     |
| 10 | 17 | 12      | $10 \times 12 = 120 \equiv 1$  |
| 2  | 4  | None    | $\gcd(2,4) \ne 1$              |
| 7  | 13 | 2       | $7 \times 2 = 14 \equiv 1$     |


#### Complexity

- Extended GCD: O(log M)
- Fermat's (prime M): O(log M) (via modular exponentiation)
- Space: O(1) iterative

The modular inverse is the key that unlocks division in the modular world, where every valid number has its own mirror multiplier that brings you back to 1.

### 507 Chinese Remainder Theorem

The Chinese Remainder Theorem (CRT) is a beautiful bridge between congruences.
It lets you solve systems of modular equations, combining many modular worlds into one consistent solution.
Originally described over two thousand years ago, it remains a cornerstone of modern number theory and cryptography.

#### What Problem Are We Solving?

We want an integer $x$ that satisfies a system of congruences:

$$
\begin{cases}
x \equiv a_1 \pmod{m_1} \\
x \equiv a_2 \pmod{m_2} \\
\vdots \\
x \equiv a_k \pmod{m_k}
\end{cases}
$$

If the moduli $m_1,m_2,\dots,m_k$ are pairwise coprime, there is a unique solution modulo
$$
M = m_1 m_2 \cdots m_k.
$$

Example

Find $x$ such that
$$
\begin{cases}
x \equiv 2 \pmod{3} \\
x \equiv 3 \pmod{4} \\
x \equiv 2 \pmod{5}
\end{cases}
$$

Compute
$$
M = 3 \cdot 4 \cdot 5 = 60,\quad
M_1 = \frac{M}{3}=20,\quad
M_2 = \frac{M}{4}=15,\quad
M_3 = \frac{M}{5}=12.
$$

Find inverses
$$
20^{-1} \pmod{3} = 2,\quad
15^{-1} \pmod{4} = 3,\quad
12^{-1} \pmod{5} = 3.
$$

Combine
$$
x = 2\cdot 20\cdot 2 \;+\; 3\cdot 15\cdot 3 \;+\; 2\cdot 12\cdot 3
  = 80 + 135 + 72 = 287.
$$

Reduce modulo $60$
$$
x \equiv 287 \bmod 60 = 47.
$$

So the solution is
$$
x \equiv 47 \pmod{60}.
$$


#### How Does It Work (Plain Language)?

Each congruence gives a "lane" that repeats every $m_i$.

CRT finds the intersection point, the smallest $x$ where all lanes line up.

Think of modular worlds like clocks with different tick lengths.
CRT finds the time when all clocks show the specified hands simultaneously.

| Modulus | Remainder | Period | Aligns At       |
| ------- | --------- | ------ | --------------- |
| 3       | 2         | 3      | 2, 5, 8, 11, …  |
| 4       | 3         | 4      | 3, 7, 11, 15, … |
| 5       | 2         | 5      | 2, 7, 12, 17, … |

They first align at 47, and then repeat every 60.

#### Tiny Code (Easy Version)

Python Version

```python
def extended_gcd(a, b):
    if b == 0:
        return a, 1, 0
    g, x1, y1 = extended_gcd(b, a % b)
    return g, y1, x1 - (a // b) * y1

def mod_inverse(a, m):
    g, x, y = extended_gcd(a, m)
    if g != 1:
        return None
    return x % m

def crt(a, m):
    M = 1
    for mod in m:
        M *= mod
    x = 0
    for ai, mi in zip(a, m):
        Mi = M // mi
        inv = mod_inverse(Mi, mi)
        x = (x + ai * Mi * inv) % M
    return x

a = [2, 3, 2]
m = [3, 4, 5]
print("x =", crt(a, m))  # Output: 47
```

C Version (Simplified)

```c
#include <stdio.h>

long long extended_gcd(long long a, long long b, long long *x, long long *y) {
    if (b == 0) { *x = 1; *y = 0; return a; }
    long long x1, y1;
    long long g = extended_gcd(b, a % b, &x1, &y1);
    *x = y1;
    *y = x1 - (a / b) * y1;
    return g;
}

long long mod_inverse(long long a, long long m) {
    long long x, y;
    long long g = extended_gcd(a, m, &x, &y);
    if (g != 1) return -1;
    return (x % m + m) % m;
}

long long crt(int a[], int m[], int n) {
    long long M = 1, x = 0;
    for (int i = 0; i < n; i++) M *= m[i];
    for (int i = 0; i < n; i++) {
        long long Mi = M / m[i];
        long long inv = mod_inverse(Mi, m[i]);
        x = (x + (long long)a[i] * Mi * inv) % M;
    }
    return x;
}

int main(void) {
    int a[] = {2, 3, 2};
    int m[] = {3, 4, 5};
    int n = 3;
    printf("x = %lld\n", crt(a, m, n)); // Output: 47
}
```

#### Why It Matters

- Combines multiple modular systems into one unified solution
- Essential in RSA (CRT optimization)
- Used in Chinese calendar, hashing, polynomial moduli, FFTs
- Foundation for multi-modular arithmetic in big integer math

#### A Gentle Proof (Why It Works)

If the moduli $m_i$ are pairwise coprime, then $M_i = M/m_i$ is coprime with $m_i$.
So each $M_i$ has an inverse $n_i$ such that
$$
M_i \cdot n_i \equiv 1 \pmod{m_i}.
$$

The combination
$$
x = \sum_{i=1}^{k} a_i \, M_i \, n_i
$$
satisfies $x \equiv a_i \pmod{m_i}$ for all $i$.
Reducing $x$ modulo $M$ gives the smallest non-negative solution.

#### Try It Yourself

1. Solve
   $x \equiv 1 \pmod{2}$,
   $x \equiv 2 \pmod{3}$,
   $x \equiv 3 \pmod{5}$.
2. Change one modulus to not be coprime (e.g., $4, 6$) and observe what happens.
3. Implement CRT with non-coprime moduli using Garner’s algorithm.
4. Test large primes (use Python big integers).
5. Use CRT to reconstruct $x$ from residues modulo $10^{9}+7$ and $998244353$.


#### Test Cases

| System                      | Solution | Modulus | Check |
| --------------------------- | -------- | ------- | ----- |
| (2 mod 3, 3 mod 4, 2 mod 5) | 47       | 60      | Ok     |
| (1 mod 2, 2 mod 3, 3 mod 5) | 23       | 30      | Ok     |
| (3 mod 5, 1 mod 7)          | 31       | 35      | Ok     |
| (0 mod 3, 1 mod 4)          | 4        | 12      | Ok     |

#### Complexity

- Time: O(k log M) (each step uses Extended GCD)
- Space: O(k)

CRT is the harmony of modular worlds, it unifies many congruences into one elegant answer, echoing ancient arithmetic and modern encryption alike.

### 508 Binary GCD (Stein's Algorithm)

The Binary GCD algorithm, also known as Stein's algorithm, computes the greatest common divisor using bit operations instead of division.
It's often faster than the classical Euclidean algorithm, especially on binary hardware, perfect for low-level, performance-sensitive code.

#### What Problem Are We Solving?

We want to compute

$$
\gcd(a,b)
$$

without division, using only shifts, subtraction, and parity checks. This is the binary GCD (Stein’s) algorithm.

Algorithm

1. If $a=0$ return $b$. If $b=0$ return $a$.
2. Let $k=\min(v_2(a),\,v_2(b))$ where $v_2(x)$ is the number of trailing zero bits in $x$.
3. Set $a \gets a/2^{v_2(a)}$ and $b \gets b/2^{v_2(b)}$  (both become odd).
4. While $a \ne b$:
   - If $a>b$, set $a \gets a-b$; then remove factors of two: $a \gets a/2^{v_2(a)}$.
   - Else set $b \gets b-a$; then $b \gets b/2^{v_2(b)}$.
5. Return $a \cdot 2^{k}$.

Worked example

Find $\gcd(48,18)$.

- Trailing zeros: $v_2(48)=4$, $v_2(18)=1$, so $k=\min(4,1)=1$.
- Make odd:
  - $a \gets 48/2^{4}=3$
  - $b \gets 18/2^{1}=9$
- Loop:
  - $b \gets 9-3=6 \Rightarrow b \gets 6/2^{1}=3$
  - Now $a=b=3$
- Answer: $3 \cdot 2^{1}=6$

So $\gcd(48,18)=6$.

Notes

- Only uses subtraction and bit shifts.
- Complexity is $O(\log(\max(a,b)))$ with very simple operations.


#### How Does It Work (Plain Language)?

Stein's insight:

- If both numbers are even → $\gcd(a,b) = 2 \times \gcd(a/2,\, b/2)$
- If one is even → divide it by 2
- If both are odd → replace the larger by (larger − smaller)
- Repeat until they become equal

| Step | a  | b  | Operation               | Note                          |
| ---- | -- | -- | ----------------------- | ----------------------------- |
| 1    | 48 | 18 | both even → divide by 2 | $\gcd = 2 \times \gcd(24,9)$  |
| 2    | 24 | 9  | one even → divide a     | $\gcd = 2 \times \gcd(12,9)$  |
| 3    | 12 | 9  | one even → divide a     | $\gcd = 2 \times \gcd(6,9)$   |
| 4    | 6  | 9  | one even → divide a     | $\gcd = 2 \times \gcd(3,9)$   |
| 5    | 3  | 9  | both odd → $b-a=6$      | $\gcd = 2 \times \gcd(3,6)$   |
| 6    | 3  | 6  | one even → divide b     | $\gcd = 2 \times \gcd(3,3)$   |
| 7    | 3  | 3  | equal → return 3        | $\gcd = 2 \times 3 = 6$       |

Final result: $\gcd(48,18)=6$.

#### Tiny Code (Easy Versions)

C Version

```c
#include <stdio.h>

int gcd_binary(int a, int b) {
    if (a == 0) return b;
    if (b == 0) return a;

    // find power of 2 factor
    int shift = 0;
    while (((a | b) & 1) == 0) {
        a >>= 1;
        b >>= 1;
        shift++;
    }

    // make 'a' odd
    while ((a & 1) == 0) a >>= 1;

    while (b != 0) {
        while ((b & 1) == 0) b >>= 1;
        if (a > b) {
            int temp = a;
            a = b;
            b = temp;
        }
        b = b - a;
    }

    return a << shift;
}

int main(void) {
    int a, b;
    printf("Enter a and b: ");
    scanf("%d %d", &a, &b);
    printf("gcd(%d, %d) = %d\n", a, b, gcd_binary(a, b));
}
```

Python Version

```python
def gcd_binary(a, b):
    if a == 0: return b
    if b == 0: return a
    shift = 0
    while ((a | b) & 1) == 0:
        a >>= 1
        b >>= 1
        shift += 1
    while (a & 1) == 0:
        a >>= 1
    while b != 0:
        while (b & 1) == 0:
            b >>= 1
        if a > b:
            a, b = b, a
        b -= a
    return a << shift

a, b = map(int, input("Enter a, b: ").split())
print("gcd(", a, ",", b, ") =", gcd_binary(a, b))
```

#### Why It Matters

- Avoids division, uses only shifts, subtraction, and comparisons
- Fast on binary processors (hardware-friendly)
- Works for unsigned integers, useful in embedded systems
- Demonstrates the power of bitwise math for classic problems

#### A Gentle Proof (Why It Works)

The gcd rules remain the same:

- gcd(2a, 2b) = 2 × gcd(a, b)
- gcd(a, 2b) = gcd(a, b) if a is odd
- gcd(a, b) = gcd(a, b − a) if both odd and a < b

Each step preserves gcd properties while removing factors of 2 efficiently.

By induction, when a = b, that value is the gcd.

#### Try It Yourself

1. Compute gcd(48, 18) step by step.
2. Try gcd(56, 98).
3. Modify code to count operations.
4. Compare runtime with Euclidean version.
5. Use on 64-bit integers and test large inputs.

#### Test Cases

| a   | b   | gcd(a, b) | Steps |
| --- | --- | --------- | ----- |
| 48  | 18  | 6         | Ok     |
| 56  | 98  | 14        | Ok     |
| 101 | 10  | 1         | Ok     |
| 270 | 192 | 6         | Ok     |
| 0   | 8   | 8         | Ok     |

#### Complexity

- Time: O(log min(a, b))
- Space: O(1)
- Bitwise operations make it faster in practice than division-based GCD

Binary GCD is Euclid's spirit in binary form, subtraction, shifting, and symmetry in the dance of bits.

### 509 Modular Reduction

Modular reduction is the process of bringing a number back into range by taking its remainder modulo M.
It's a small but essential step in modular arithmetic, every modular algorithm uses it to keep numbers bounded and stable.

Think of it as folding an infinitely long number line into a circle of length M, and asking, "Where do we land?"

#### What Problem Are We Solving?

We want to compute:

$$
x \bmod M
$$

That is, the remainder when (x) is divided by (M), always mapped into the canonical range [0, M−1].

Example:
If$M = 10$:

| x  | x mod 10 |
| -- | -------- |
| 23 | 3        |
| 17 | 7        |
| 0  | 0        |
| -3 | 7        |

Negative numbers wrap around to a positive residue.

So we want a normalized result, never negative.

#### How Does It Work (Plain Language)?

The remainder operator `%` in most languages can produce negative results for negative inputs.
To ensure a proper modular residue, we fix it by adding M back if needed.

Rule of thumb:

$$
\text{mod}(x, M) = ((x % M) + M) % M
$$

| x  | M  | x % M | Normalized |
| -- | -- | ----- | ---------- |
| 23 | 10 | 3     | 3          |
| -3 | 10 | -3    | 7          |
| 15 | 6  | 3     | 3          |
| -8 | 5  | -3    | 2          |

This ensures x mod M ∈ [0, M−1].

#### Tiny Code (Easy Versions)

C Version

```c
#include <stdio.h>

int mod_reduce(int x, int M) {
    int r = x % M;
    if (r < 0) r += M;
    return r;
}

int main(void) {
    int x, M;
    printf("Enter x and M: ");
    scanf("%d %d", &x, &M);
    printf("%d mod %d = %d\n", x, M, mod_reduce(x, M));
}
```

Python Version

```python
def mod_reduce(x, M):
    return (x % M + M) % M

x, M = map(int, input("Enter x, M: ").split())
print(f"{x} mod {M} =", mod_reduce(x, M))
```

#### Why It Matters

- Ensures correct residues even with negative numbers
- Keeps arithmetic consistent:$(a + b) \bmod M = ((a \bmod M) + (b \bmod M)) \bmod M$
- Critical in cryptography, hashing, polynomial mod arithmetic
- Prevents overflow and sign bugs in modular computations

#### A Gentle Proof (Why It Works)

Every integer (x) can be written as:

$$
x = qM + r
$$

where (q) is the quotient and (r) is the remainder.
We want (r \in [0, M-1]).

If `%` gives negative (r), then (r + M) is the positive residue:
$$
(x \bmod M) = (x % M + M) % M
$$

This satisfies modular congruence:
$$
x \equiv r \pmod{M}
$$

#### Try It Yourself

1. Compute $-7 \bmod 5$ by hand.
2. Test code with negative inputs.
3. Build `mod_add`, `mod_sub`, `mod_mul` using normalized reduction.
4. Use reduction inside loops to prevent overflow.
5. Compare `%` vs proper mod with negative numbers in C or Python.

#### Test Cases

| x  | M  | Expected | Check |
| -- | -- | -------- | ----- |
| 23 | 10 | 3        | Ok     |
| -3 | 10 | 7        | Ok     |
| 15 | 6  | 3        | Ok     |
| -8 | 5  | 2        | Ok     |
| 0  | 9  | 0        | Ok     |

#### Complexity

- Time: O(1)
- Space: O(1)

Modular reduction is the heartbeat of modular arithmetic, every operation folds back into a circle, keeping numbers small, positive, and predictable.

### 510 Modular Linear Equation Solver

A modular linear equation is an equation of the form
$$
a x \equiv b \pmod{m}
$$
We want to find all integers $x$ that satisfy this congruence.
This is the modular version of solving $ax = b$, but in a world that wraps around at multiples of (m).

#### What Problem Are We Solving?

Given integers $a,b,m$, solve the linear congruence
$$
a x \equiv b \pmod{m}.
$$

Key facts

- Let $g=\gcd(a,m)$.  
- A solution exists iff $g \mid b$.  
- If a solution exists, there are exactly $g$ solutions modulo $m$, spaced by $m/g$.

Procedure

1) Compute $g=\gcd(a,m)$. If $g \nmid b$, no solution.  
2) Reduce:
$$
a'=\frac{a}{g},\quad b'=\frac{b}{g},\quad m'=\frac{m}{g}.
$$
Then solve the coprime congruence
$$
a' x \equiv b' \pmod{m'}.
$$
3) Find the inverse $a'^{-1} \pmod{m'}$ and set
$$
x_0 \equiv a'^{-1} b' \pmod{m'}.
$$
4) All solutions modulo $m$ are
$$
x \equiv x_0 + t\cdot \frac{m}{g} \pmod{m},\quad t=0,1,\dots,g-1.
$$

Worked example

Solve $6x \equiv 8 \pmod{14}$.

1) $g=\gcd(6,14)=2$, and $2 \mid 8$ so solutions exist.  
2) Reduce: $a'=6/2=3$, $b'=8/2=4$, $m'=14/2=7$. Solve
$$
3x \equiv 4 \pmod{7}.
$$
3) Inverse: $3^{-1}\equiv 5 \pmod{7}$, so
$$
x_0 \equiv 5\cdot 4 \equiv 20 \equiv 6 \pmod{7}.
$$
4) Lift to modulo 14. Since $m/g=7$, solutions are
$$
x \equiv 6 + t\cdot 7 \pmod{14},\quad t=0,1.
$$
Thus $x \in \{6,\,13\} \pmod{14}$.


#### How Does It Work (Plain Language)?

1. Check solvability  
   A solution exists iff $g=\gcd(a,m)$ divides $b$.

2. Reduce the congruence by $g$ 
   $$
   \frac{a}{g}\,x \equiv \frac{b}{g} \pmod{\frac{m}{g}}
   $$

3. Find the modular inverse  
   Compute $\left(\frac{a}{g}\right)^{-1} \pmod{\frac{m}{g}}$.

4. Solve for one solution, then enumerate all $g$ solutions  
   $$
   x_0 \equiv \left(\frac{a}{g}\right)^{-1}\!\left(\frac{b}{g}\right) \pmod{\frac{m}{g}}
   $$
   $$
   x \equiv x_0 + k\cdot\frac{m}{g} \pmod{m},\quad k=0,1,\ldots,g-1
   $$


#### Step-by-Step Table (Example)

| Step | Equation                 | Action                | Result            |
| ---- | ------------------------ | --------------------- | ----------------- |
| 1    | $6x \equiv 8 \pmod{14}$  | $\gcd(6,14)=2 \mid 8$ | solvable          |
| 2    | divide by $2$            | $3x \equiv 4 \pmod{7}$| simplified        |
| 3    | inverse of $3 \bmod 7$   | $5$                   | since $3\cdot5\equiv1$ |
| 4    | multiply both sides      | $x \equiv 4\cdot5 \equiv 20 \equiv 6 \pmod{7}$ | one solution |
| 5    | lift to $\pmod{14}$      | $x \in \{6,\,13\}$    | ok                |


#### Tiny Code (Easy Versions)

Python Version

```python
def extended_gcd(a, b):
    if b == 0:
        return a, 1, 0
    g, x1, y1 = extended_gcd(b, a % b)
    return g, y1, x1 - (a // b) * y1

def solve_modular_linear(a, b, m):
    g, x, y = extended_gcd(a, m)
    if b % g != 0:
        return []  # No solution
    a1, b1, m1 = a // g, b // g, m // g
    x0 = (x * b1) % m1
    return [(x0 + i * m1) % m for i in range(g)]

a, b, m = map(int, input("Enter a, b, m: ").split())
solutions = solve_modular_linear(a, b, m)
if solutions:
    print("Solutions:", solutions)
else:
    print("No solution")
```

C Version (Simplified)

```c
#include <stdio.h>

long long extended_gcd(long long a, long long b, long long *x, long long *y) {
    if (b == 0) { *x = 1; *y = 0; return a; }
    long long x1, y1;
    long long g = extended_gcd(b, a % b, &x1, &y1);
    *x = y1;
    *y = x1 - (a / b) * y1;
    return g;
}

int solve_modular(long long a, long long b, long long m, long long sol[]) {
    long long x, y;
    long long g = extended_gcd(a, m, &x, &y);
    if (b % g != 0) return 0;
    a /= g; b /= g; m /= g;
    long long x0 = ((x * b) % m + m) % m;
    for (int i = 0; i < g; i++)
        sol[i] = (x0 + i * m) % (m * g);
    return g;
}

int main(void) {
    long long a, b, m, sol[10];
    printf("Enter a, b, m: ");
    scanf("%lld %lld %lld", &a, &b, &m);
    int n = solve_modular(a, b, m, sol);
    if (n == 0) printf("No solution\n");
    else {
        printf("Solutions:");
        for (int i = 0; i < n; i++) printf(" %lld", sol[i]);
        printf("\n");
    }
}
```

#### Why It Matters

- Solves modular equations, the building block for CRT, RSA, and Diophantine systems
- Generalizes modular inverses (when b=1)
- Basis for solving linear congruence systems
- Enables modular division in non-prime moduli

#### A Gentle Proof (Why It Works)

If $a x \equiv b \pmod{m}$, then $m \mid (a x - b)$.

Let $g=\gcd(a,m)$. Divide the congruence by $g$:
$$
\frac{a}{g}\,x \equiv \frac{b}{g} \pmod{\frac{m}{g}}.
$$
Now $\gcd\!\left(\frac{a}{g},\frac{m}{g}\right)=1$, so $\left(\frac{a}{g}\right)^{-1} \pmod{\frac{m}{g}}$ exists. Multiplying both sides by this inverse gives one solution modulo $\frac{m}{g}$. Adding multiples of $\frac{m}{g}$ generates all $g$ solutions modulo $m$:
$$
x \equiv x_0 + k\cdot \frac{m}{g} \pmod{m}, \quad k=0,1,\dots,g-1.
$$

#### Try It Yourself

1. Solve $6x \equiv 8 \pmod{14}$
2. Solve $4x \equiv 2 \pmod{6}$
3. Solve $3x \equiv 2 \pmod{7}$
4. Try an unsolvable case: $4x \equiv 3 \pmod{6}$
5. Modify code to print $\gcd$ and the inverse at each step


#### Test Cases

| a | b | m  | Solutions | Check |
| - | - | -- | --------- | ----- |
| 6 | 8 | 14 | 6, 13     | Ok     |
| 4 | 2 | 6  | 2, 5      | Ok     |
| 3 | 2 | 7  | 3         | Ok     |
| 4 | 3 | 6  | None      | Ok     |

#### Complexity

- Time: O(log m)
- Space: O(1)

The modular linear solver turns arithmetic into algebra on a circle, finding where linear lines cross modular grids.

# Section 52. Primality and Factorization 

### 511 Trial Division

Trial Division is the simplest way to test if a number is prime, by checking whether any smaller number divides it evenly.
It's slow for large$n$, but perfect for building intuition, small primes, and as a subroutine inside more advanced factorization or primality tests.

#### What Problem Are We Solving?

We want to decide if an integer$n > 1$ is prime or composite.

A prime has exactly two divisors (1 and itself).
A composite has additional divisors.

Trial division checks all possible divisors up to$\sqrt{n}$.

Example
Is$n = 37$ prime?

Check divisors:
$$
\begin{aligned}
37 \bmod 2 &= 1 \
37 \bmod 3 &= 1 \
37 \bmod 4 &= 1 \
37 \bmod 5 &= 2 \
37 \bmod 6 &= 1
\end{aligned}
$$

No divisors found up to$\sqrt{37}$. Therefore, prime.

#### How Does It Work (Plain Language)

If$n = a \times b$, then one of$a$ or$b$ must satisfy$a, b \leq \sqrt{n}$.
So if$n$ has a divisor, it will appear before$\sqrt{n}$.
We check each integer in that range.

To optimize:

- First check$2$
- Then test only odd numbers

| Step | Divisor          |$n \bmod \text{Divisor}$ | Result |
| ---- | ---------------- | -------------------------- | ------ |
| 1    | 2                | 1                          | skip   |
| 2    | 3                | 1                          | skip   |
| 3    | 4                | 1                          | skip   |
| 4    | 5                | 2                          | skip   |
| 5    | 6                | 1                          | skip   |
|,    | No divisor found | Prime                  |        |

#### Tiny Code (Easy Versions)

C Version

```c
#include <stdio.h>
#include <math.h>
#include <stdbool.h>

bool is_prime(int n) {
    if (n < 2) return false;
    if (n == 2) return true;
    if (n % 2 == 0) return false;
    int limit = (int)sqrt(n);
    for (int i = 3; i <= limit; i += 2)
        if (n % i == 0)
            return false;
    return true;
}

int main(void) {
    int n;
    printf("Enter n: ");
    scanf("%d", &n);
    printf("%d is %s\n", n, is_prime(n) ? "prime" : "composite");
}
```

Python Version

```python
import math

def is_prime(n):
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    limit = int(math.sqrt(n)) + 1
    for i in range(3, limit, 2):
        if n % i == 0:
            return False
    return True

n = int(input("Enter n: "))
print(n, "is", "prime" if is_prime(n) else "composite")
```

#### Why It Matters

- Foundation for all primality tests
- Used for small$n$ and initial sieving
- Great for factorization of small numbers
- Builds intuition for$\sqrt{n}$ boundary and divisor pairs

#### A Gentle Proof (Why It Works)

If$n = a \times b$,
then one of$a, b \leq \sqrt{n}$.

If both were greater than$\sqrt{n}$, then:
$$
a \times b > \sqrt{n} \times \sqrt{n} = n
$$
which is impossible.
Thus, any factorization must include a number ≤$\sqrt{n}$.

Therefore, checking up to$\sqrt{n}$ is sufficient to confirm primality.

#### Try It Yourself

1. Check if$37, 49, 51$ are prime.
2. Modify code to print the first divisor found.
3. Extend code to list all divisors of$n$.
4. Compare runtime for$n = 10^6$ vs$n = 10^9$.
5. Combine with a sieve to skip non-prime divisors.

#### Test Cases

|$n$ | Expected  | First Divisor |
| ----- | --------- | ------------- |
| 2     | Prime     |,             |
| 3     | Prime     |,             |
| 4     | Composite | 2             |
| 9     | Composite | 3             |
| 37    | Prime     |,             |
| 49    | Composite | 7             |

#### Complexity

- Time:$O(\sqrt{n})$
- Space:$O(1)$

Trial Division is the "hello world" of primality testing, simple, certain, and fundamental to number theory.

### 512 Sieve of Eratosthenes

The Sieve of Eratosthenes is a classic and efficient algorithm for finding all prime numbers up to a given limit $n$.
Instead of testing each number individually, it eliminates multiples of known primes, leaving only primes behind.

This sieve is one of the oldest known algorithms (over 2000 years old) and remains a cornerstone of computational number theory.

#### What Problem Are We Solving?

We want to generate all primes up to $n$.

Example
Find all primes $\leq 30$:

$$
{2, 3, 5, 7, 11, 13, 17, 19, 23, 29}
$$

#### How Does It Work (Plain Language)

Think of a list of integers $2, 3, \ldots, n$.
We will repeatedly "cross out" multiples of each prime.

Steps:

1. Start with the first prime $p = 2$.
2. Cross out all multiples of $p$ starting from $p^2$.
3. Move to the next number not yet crossed out, that's the next prime.
4. Repeat until $p^2 > n$.

What remains unmarked are all the primes.

Example: $n = 30$

| Step | Prime $p$ | Remove Multiples                | Remaining Primes                     |
| ---- | --------- | ------------------------------- | ------------------------------------ |
| 1    | 2         | $4, 6, 8, 10, 12, \ldots$       | $2, 3, 5, 7, 9, 11, 13, 15, \ldots$  |
| 2    | 3         | $9, 12, 15, 18, 21, 24, 27, 30$ | $2, 3, 5, 7, 11, 13, 17, 19, 23, 29$ |
| 3    | 5         | $25, 30$                        | no change                            |
| 4    | Stop      | $5^2 = 25 > \sqrt{30}$          | Done                                 |

Final primes up to 30:

$$
{2, 3, 5, 7, 11, 13, 17, 19, 23, 29}
$$

#### Tiny Code (Easy Versions)

C Version

```c
#include <stdio.h>
#include <stdbool.h>
#include <math.h>

void sieve(int n) {
    bool is_prime[n + 1];
    for (int i = 0; i <= n; i++) is_prime[i] = true;
    is_prime[0] = is_prime[1] = false;

    for (int p = 2; p * p <= n; p++) {
        if (is_prime[p]) {
            for (int multiple = p * p; multiple <= n; multiple += p)
                is_prime[multiple] = false;
        }
    }

    printf("Primes up to %d:\n", n);
    for (int i = 2; i <= n; i++)
        if (is_prime[i]) printf("%d ", i);
    printf("\n");
}

int main(void) {
    int n;
    printf("Enter n: ");
    scanf("%d", &n);
    sieve(n);
}
```

Python Version

```python
def sieve(n):
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    p = 2
    while p * p <= n:
        if is_prime[p]:
            for multiple in range(p * p, n + 1, p):
                is_prime[multiple] = False
        p += 1
    return [i for i in range(2, n + 1) if is_prime[i]]

n = int(input("Enter n: "))
print("Primes:", sieve(n))
```

#### Why It Matters

- Efficiently generates all primes up to $n$
- Used in number theory, cryptography, factorization, and prime sieving precomputation
- Avoids repeated division
- Foundation for advanced sieves (e.g. Linear Sieve, Segmented Sieve)

#### A Gentle Proof (Why It Works)

Every composite number $n$ has a smallest prime factor $p$.
That factor $p$ will mark the composite when the algorithm reaches $p$.

Thus, by crossing out all multiples of each prime, all composites are removed, and only primes remain.

Because any composite $n = a \times b$ has at least one $a \leq \sqrt{n}$,
we can stop sieving when $p^2 > n$.

#### Try It Yourself

1. Generate primes $\leq 50$.
2. Modify code to count how many primes are found.
3. Print primes in rows of 10.
4. Compare runtime for $n = 10^4$, $10^5$, $10^6$.
5. Optimize memory: sieve only odd numbers.

#### Test Cases

| $n$ | Expected Output                      |
| --- | ------------------------------------ |
| 10  | $2, 3, 5, 7$                         |
| 20  | $2, 3, 5, 7, 11, 13, 17, 19$         |
| 30  | $2, 3, 5, 7, 11, 13, 17, 19, 23, 29$ |

#### Complexity

- Time: $O(n \log \log n)$
- Space: $O(n)$

The Sieve of Eratosthenes blends clarity and efficiency, by striking out every composite, it leaves behind the primes that shape number theory.

### 513 Sieve of Atkin

The Sieve of Atkin is a modern improvement on the Sieve of Eratosthenes.
Instead of crossing out multiples, it uses quadratic forms and modular arithmetic to determine prime candidates, then eliminates non-primes via square multiples.
It's asymptotically faster and beautifully mathematical, though more complex to implement.

#### What Problem Are We Solving?

We want to find all prime numbers up to a given limit $n$, but faster than the classic sieve.

The Sieve of Atkin uses congruence conditions based on quadratic residues to detect potential primes.

#### How Does It Work (Plain Language)

For a given integer $n$, we determine whether it is a prime candidate by checking specific modular equations:

1. Initialize an array `is_prime[0..n]` to false.

2. For every integer pair $(x, y)$ with $x, y \ge 1$, compute:

   * $n_1 = 4x^2 + y^2$

     * If $n_1 \le N$ and $n_1 \bmod 12 \in {1, 5}$, flip `is_prime[n1]`
   * $n_2 = 3x^2 + y^2$

     * If $n_2 \le N$ and $n_2 \bmod 12 = 7$, flip `is_prime[n2]`
   * $n_3 = 3x^2 - y^2$

     * If $x > y$, $n_3 \le N$, and $n_3 \bmod 12 = 11$, flip `is_prime[n3]`

3. Eliminate multiples of squares:

   * For each $k$ such that $k^2 \le N$,
     mark all multiples of $k^2$ as composite.

4. Finally, add 2 and 3 as primes.

All remaining numbers marked true are primes.

Example (small N = 50):

Start with 2, 3, and 5.
Apply modular conditions to detect others:

| Condition    | Formula | Mod Class | Candidates                  |
| ------------ | ------- | --------- | --------------------------- |
| $4x^2 + y^2$ | $1, 5$  | $12$      | $5, 13, 17, 29, 37, 41, 49$ |
| $3x^2 + y^2$ | $7$     | $12$      | $7, 19, 31, 43$             |
| $3x^2 - y^2$ | $11$    | $12$      | $11, 23, 47$                |

Then remove multiples of squares (e.g. $25, 49$).
Remaining primes up to 50:

$$
{2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47}
$$

#### Tiny Code (Easy Version)

Python Version

```python
import math

def sieve_atkin(limit):
    is_prime = [False] * (limit + 1)
    sqrt_limit = int(math.sqrt(limit)) + 1

    for x in range(1, sqrt_limit):
        for y in range(1, sqrt_limit):
            n = 4 * x * x + y * y
            if n <= limit and n % 12 in (1, 5):
                is_prime[n] = not is_prime[n]

            n = 3 * x * x + y * y
            if n <= limit and n % 12 == 7:
                is_prime[n] = not is_prime[n]

            n = 3 * x * x - y * y
            if x > y and n <= limit and n % 12 == 11:
                is_prime[n] = not is_prime[n]

    for n in range(5, sqrt_limit):
        if is_prime[n]:
            for k in range(n * n, limit + 1, n * n):
                is_prime[k] = False

    primes = [2, 3]
    primes.extend([i for i in range(5, limit + 1) if is_prime[i]])
    return primes

n = int(input("Enter n: "))
print("Primes:", sieve_atkin(n))
```

#### Why It Matters

- Faster than the Sieve of Eratosthenes for very large $n$
- Demonstrates deep connections between number theory and computation
- Uses modular patterns of quadratic residues to detect primes
- Foundation for optimized sieving algorithms

#### A Gentle Proof (Why It Works)

Every integer can be represented in quadratic forms.
Primes occur in specific modular classes:

- Primes of form $4x + 1$ satisfy $4x^2 + y^2$
- Primes of form $6x + 1$ or $6x + 5$ satisfy $3x^2 + y^2$ or $3x^2 - y^2$

By counting solutions to these congruences mod 12, one can distinguish primes from composites.
Flipping ensures each candidate is toggled odd number of times only if it meets prime conditions.

Multiples of squares are eliminated since no square factor can belong to a prime.

#### Try It Yourself

1. Generate all primes $\le 100$.
2. Compare output with Eratosthenes.
3. Measure performance for $n = 10^6$.
4. Modify code to count primes only.
5. Print candidate flips to see how toggling works.

#### Test Cases

| $n$ | Expected Output (Primes)                                 |
| --- | -------------------------------------------------------- |
| 10  | $2, 3, 5, 7$                                             |
| 30  | $2, 3, 5, 7, 11, 13, 17, 19, 23, 29$                     |
| 50  | $2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47$ |

#### Complexity

- Time: $O(n)$ (with modular arithmetic precomputation)
- Space: $O(n)$

The Sieve of Atkin turns number theory into code, using the elegance of quadratic forms to filter primes from the infinite sea of integers.

### 514 Miller–Rabin Primality Test

The Miller–Rabin test is a fast probabilistic primality test.
Given an odd integer $n > 2$, it decides whether $n$ is composite or probably prime by checking whether $a$ behaves like a witness to compositeness modulo $n$.

#### What Problem Are We Solving?

Decide primality of a large integer $n$ much faster than trial division or a full sieve, especially when $n$ can be hundreds or thousands of bits.

Input: odd $n > 2$, accuracy parameter $k$ (number of bases).
Output: composite or probably prime.

#### How Does It Work (Plain Language)

Write
$$
n - 1 = 2^s \cdot d \quad \text{with } d \text{ odd}.
$$

Repeat for $k$ random bases $a \in {2, 3, \ldots, n-2}$:

1. Compute
   $$
   x \equiv a^{,d} \bmod n.
   $$

2. If $x = 1$ or $x = n-1$, this base passes.

3. Otherwise, square up to $s-1$ times:
   $$
   x \leftarrow x^2 \bmod n.
   $$
   If at any step $x = n-1$, the base passes.
   If none hit $n-1$, declare composite.

If all $k$ bases pass, declare probably prime.
For composite $n$, a random base exposes compositeness with probability at least $1/4$, so error is at most $(1/4)^k$.

#### Tiny Code (Easy Versions)

C Version

```c
#include <stdio.h>
#include <stdint.h>

static uint64_t mul_mod(uint64_t a, uint64_t b, uint64_t m) {
    __uint128_t z =$__uint128_t$a * b % m;
    return (uint64_t)z;
}

static uint64_t pow_mod(uint64_t a, uint64_t e, uint64_t m) {
    uint64_t r = 1 % m;
    a %= m;
    while (e > 0) {
        if (e & 1) r = mul_mod(r, a, m);
        a = mul_mod(a, a, m);
        e >>= 1;
    }
    return r;
}

static int miller_rabin_base(uint64_t n, uint64_t a, uint64_t d, int s) {
    uint64_t x = pow_mod(a, d, n);
    if (x == 1 || x == n - 1) return 1;
    for (int i = 1; i < s; i++) {
        x = mul_mod(x, x, n);
        if (x == n - 1) return 1;
    }
    return 0; // composite for this base
}

int is_probable_prime(uint64_t n) {
    if (n < 2) return 0;
    for (uint64_t p : (uint64_t[]){2,3,5,7,11,13,17,19,23,0}) {
        if (p == 0) break;
        if (n % p == 0) return n == p;
    }
    // write n-1 = 2^s * d
    uint64_t d = n - 1;
    int s = 0;
    while ((d & 1) == 0) { d >>= 1; s++; }

    // Deterministic set for 64-bit integers
    uint64_t bases[] = {2, 3, 5, 7, 11, 13, 17};
    int nb = sizeof(bases) / sizeof(bases[0]);
    for (int i = 0; i < nb; i++) {
        uint64_t a = bases[i] % n;
        if (a <= 1) continue;
        if (!miller_rabin_base(n, a, d, s)) return 0;
    }
    return 1; // probably prime
}

int main(void) {
    uint64_t n;
    if (scanf("%llu", &n) != 1) return 0;
    printf("%llu is %s\n", n, is_probable_prime(n) ? "probably prime" : "composite");
    return 0;
}
```

Python Version

```python
def pow_mod(a, e, m):
    r = 1
    a %= m
    while e > 0:
        if e & 1:
            r = (r * a) % m
        a = (a * a) % m
        e >>= 1
    return r

def miller_rabin(n, bases=None):
    if n < 2:
        return False
    small_primes = [2,3,5,7,11,13,17,19,23]
    for p in small_primes:
        if n % p == 0:
            return n == p
    # write n-1 = 2^s * d
    d = n - 1
    s = 0
    while d % 2 == 0:
        d //= 2
        s += 1
    if bases is None:
        # Deterministic for 64-bit range
        bases = [2, 3, 5, 7, 11, 13, 17]
    for a in bases:
        a %= n
        if a <= 1:
            continue
        x = pow_mod(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(s - 1):
            x = (x * x) % n
            if x == n - 1:
                break
        else:
            return False
    return True

n = int(input().strip())
print("probably prime" if miller_rabin(n) else "composite")
```

#### Why It Matters

- Very fast screening test for large integers.
- Standard in cryptographic key generation pipelines.
- With a fixed base set, becomes deterministic for bounded ranges (for 64 bit integers, the given bases suffice).

#### A Gentle Proof (Why It Works)

Fermat style motivation: if $n$ is prime and $\gcd(a,n)=1$, then by Euler
$$
a^{n-1} \equiv 1 \pmod n.
$$
Stronger, by writing $n-1 = 2^s d$, the sequence
$$
a^{d},\ a^{2d},\ a^{4d},\ \ldots,\ a^{2^{s-1}d} \pmod n
$$
must land at $1$ through a chain that can only introduce the value $-1 \equiv n-1$ immediately before $1$. If the chain misses $n-1$, $a$ certifies compositeness. For composite $n$, at least three quarters of $a$ are witnesses, giving error at most $(1/4)^k$ after $k$ independent bases.

#### Try It Yourself

1. Factor $n-1$ as $2^s d$ for $n = 561$ and test bases $a = 2, 3, 5$.
2. Generate random 64 bit odd $n$ and compare Miller–Rabin against trial division up to $10^6$.
3. Replace bases with random choices and measure error frequency on Carmichael numbers.
4. Extend to a deterministic set that covers your target range.

#### Test Cases

| $n$                 | Result                      | Notes                   |
| ------------------- | --------------------------- | ----------------------- |
| $37$                | probably prime              | prime                   |
| $221 = 13 \cdot 17$ | composite                   | small factor            |
| $561$               | composite                   | Carmichael number       |
| $2^{61}-1$          | probably prime              | Mersenne candidate      |
| $10^{18}+3$         | composite or probably prime | depends on actual value |

#### Complexity

- Time: $O(k \log^3 n)$ with schoolbook modular multiplication, $k$ bases.
- Space: $O(1)$ iterative state.

Miller–Rabin gives a fast and reliable primality screen: decompose $n-1$, test a few bases, and either certify compositeness or return a very strong probably prime.

### 515 Fermat Primality Test

The Fermat primality test is one of the simplest probabilistic tests for determining whether a number is likely prime.
It's based on Fermat's Little Theorem, which states that if$n$ is prime and$a$ is not divisible by$n$, then

$$
a^{n-1} \equiv 1 \pmod{n}.
$$

If this congruence fails for some base$a$, then$n$ is definitely composite.
If it holds for several randomly chosen bases,$n$ is probably prime.

#### What Problem Are We Solving?

We want a fast check for whether$n$ is prime, especially for large$n$, where trial division or sieves are too slow.

We test whether numbers satisfy Fermat's congruence condition:

$$
a^{n-1} \bmod n = 1
$$

for random bases$a \in [2, n-2]$.

#### How Does It Work (Plain Language)

1. Choose a random integer$a$,$2 \le a \le n-2$.
2. Compute$x = a^{n-1} \bmod n$.
3. If$x \ne 1$,$n$ is composite.
4. If$x = 1$,$n$ might be prime.
5. Repeat several times with different$a$ for higher confidence.

If$n$ passes for all chosen bases, we say probably prime.
If it fails for any base, composite.

Example: Test$n = 561$ (a Carmichael number).

- Pick$a = 2$:$2^{560} \bmod 561 = 1$
- Pick$a = 3$:$3^{560} \bmod 561 = 1$
- Pick$a = 5$:$5^{560} \bmod 561 = 1$

All pass, but$561 = 3 \cdot 11 \cdot 17$ is composite.
Hence, Fermat test can be fooled by Carmichael numbers.

#### Tiny Code (Easy Versions)

Python Version

```python
import random

def pow_mod(a, e, m):
    r = 1
    a %= m
    while e > 0:
        if e & 1:
            r = (r * a) % m
        a = (a * a) % m
        e >>= 1
    return r

def fermat_test(n, k=5):
    if n < 2:
        return False
    if n in (2, 3):
        return True
    if n % 2 == 0:
        return False
    for _ in range(k):
        a = random.randint(2, n - 2)
        if pow_mod(a, n - 1, n) != 1:
            return False
    return True

n = int(input("Enter n: "))
print("Probably prime" if fermat_test(n) else "Composite")
```

C Version

```c
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

long long pow_mod(long long a, long long e, long long m) {
    long long r = 1;
    a %= m;
    while (e > 0) {
        if (e & 1) r = (r * a) % m;
        a = (a * a) % m;
        e >>= 1;
    }
    return r;
}

int fermat_test(long long n, int k) {
    if (n < 2) return 0;
    if (n == 2 || n == 3) return 1;
    if (n % 2 == 0) return 0;
    srand(time(NULL));
    for (int i = 0; i < k; i++) {
        long long a = 2 + rand() % (n - 3);
        if (pow_mod(a, n - 1, n) != 1)
            return 0;
    }
    return 1;
}

int main(void) {
    long long n;
    printf("Enter n: ");
    scanf("%lld", &n);
    printf("%lld is %s\n", n, fermat_test(n, 5) ? "probably prime" : "composite");
}
```

#### Why It Matters

- Extremely simple to implement
- Useful as a first filter before stronger tests (e.g. Miller–Rabin)
- Foundation for many probabilistic algorithms in number theory
- Helps illustrate Fermat's Little Theorem in computational form

#### A Gentle Proof (Why It Works)

If$n$ is prime and$\gcd(a, n) = 1$, Fermat's Little Theorem guarantees:

$$
a^{n-1} \equiv 1 \pmod{n}.
$$

If this fails,$n$ cannot be prime.
However, some composite numbers (Carmichael numbers) satisfy this condition for all$a$ coprime to$n$.
Thus, the test is probabilistic, not deterministic.

#### Try It Yourself

1. Test$n = 37$ with bases 2, 3, 5.
2. Try$n = 561$ and see the failure.
3. Increase$k$ (number of trials) to see stability.
4. Compare runtime with Miller–Rabin.
5. Build a composite that passes one base but fails another.

#### Test Cases

| $n$ | Bases   | Result                |
| --- | ------- | --------------------- |
| 37  | 2, 3, 5 | Pass → probably prime |
| 15  | 2       | Fail → composite      |
| 561 | 2, 3, 5 | Pass → false positive |
| 97  | 2, 3    | Pass → probably prime |

#### Complexity

- Time: $O(k \log^3 n)$ (due to modular exponentiation)
- Space: $O(1)$

The Fermat test is primality at lightning speed, but with a trickster's flaw: it can be fooled by clever composites called Carmichael numbers.

### 516 Pollard's Rho Algorithm

The Pollard's Rho algorithm is a clever randomized method for integer factorization.
It uses a simple iterative function and the birthday paradox to find nontrivial factors quickly, without trial division.
Though probabilistic, it's extremely effective for finding small factors of large numbers.

#### What Problem Are We Solving?

Given a composite number$n$, find a nontrivial factor$d$ such that
( 1 < d < n$.

Instead of checking divisibility exhaustively, Pollard's Rho uses a pseudorandom sequence modulo$n$ and detects when two values become congruent modulo a hidden factor.

#### How Does It Work (Plain Language)

We define an iteration:

$$
x_{i+1} = f(x_i) \bmod n, \quad \text{commonly } f(x) = (x^2 + c) \bmod n
$$

Two sequences running at different speeds (like a "tortoise and hare") eventually collide modulo a factor of$n$.
When they do, the gcd of their difference with$n$ gives a factor.

Algorithm Outline

1. Pick a random function$f(x) = (x^2 + c) \bmod n$
   with random$x_0$ and$c$.
2. Set$x = y = x_0$,$d = 1$.
3. While$d = 1$:

   *$x = f(x)$
   *$y = f(f(y))$
   *$d = \gcd(|x - y|, n)$
4. If$d = n$, restart with a new function.
5. If$1 < d < n$, output$d$.

Example:
Let$n = 8051 = 83 \times 97$.
Choose$f(x) = (x^2 + 1) \bmod 8051$,$x_0 = 2$.

| Step | (x) | (y) | (|x-y|) | (\gcd(|x-y|, 8051)) |
|------|------|------|------|------------------|
| 1 | 5 | 26 | 21 | 1 |
| 2 | 26 | 7474 | 7448 | 83 |

Ok Found factor$83$

#### Tiny Code (Easy Versions)

Python Version

```python
import math
import random

def f(x, c, n):
    return (x * x + c) % n

def pollard_rho(n):
    if n % 2 == 0:
        return 2
    x = random.randint(2, n - 1)
    y = x
    c = random.randint(1, n - 1)
    d = 1
    while d == 1:
        x = f(x, c, n)
        y = f(f(y, c, n), c, n)
        d = math.gcd(abs(x - y), n)
        if d == n:
            return pollard_rho(n)
    return d

n = int(input("Enter n: "))
factor = pollard_rho(n)
print(f"Nontrivial factor of {n}: {factor}")
```

C Version

```c
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

long long gcd(long long a, long long b) {
    while (b != 0) {
        long long t = b;
        b = a % b;
        a = t;
    }
    return a;
}

long long f(long long x, long long c, long long n) {
    return (x * x + c) % n;
}

long long pollard_rho(long long n) {
    if (n % 2 == 0) return 2;
    long long x = rand() % (n - 2) + 2;
    long long y = x;
    long long c = rand() % (n - 1) + 1;
    long long d = 1;

    while (d == 1) {
        x = f(x, c, n);
        y = f(f(y, c, n), c, n);
        long long diff = x > y ? x - y : y - x;
        d = gcd(diff, n);
        if (d == n) return pollard_rho(n);
    }
    return d;
}

int main(void) {
    srand(time(NULL));
    long long n;
    printf("Enter n: ");
    scanf("%lld", &n);
    long long factor = pollard_rho(n);
    printf("Nontrivial factor: %lld\n", factor);
}
```

#### Why It Matters

- Finds factors much faster than trial division
- Crucial in integer factorization, cryptanalysis, and RSA key testing
- Basis for advanced algorithms (e.g. Pollard's p–1, Brent's Rho)
- Probabilistic yet efficient for small factors

#### A Gentle Proof (Why It Works)

If two values become congruent modulo a factor$p$ of$n$:
$$
x_i \equiv x_j \pmod{p}, \quad i \ne j
$$
then$p \mid (x_i - x_j)$, and thus
$$
\gcd(|x_i - x_j|, n) = p.
$$
Because$p$ divides$n$, but not all of$n$, it reveals a nontrivial factor.

The "Rho" shape refers to the cycle formed by repeated squaring modulo$p$.

#### Try It Yourself

1. Factor$n = 8051$.
2. Try different functions$f(x) = x^2 + c$.
3. Test on $n = 91, 187, 589$.
4. Compare runtime to trial division.
5. Combine with recursion to fully factor $n$.

#### Test Cases

| $n$  | Factors | Found |
| ---- | ------- | ----- |
| 91   | 7 × 13  | 7     |
| 187  | 11 × 17 | 17    |
| 8051 | 83 × 97 | 83    |
| 2047 | 23 × 89 | 23    |

#### Complexity

- Expected Time: $O(n^{1/4})$
- Space: $O(1)$

Pollard's Rho is like chasing your own tail, but eventually, the loop gives up a hidden factor.

### 517 Pollard's p−1 Method

The Pollard's p−1 algorithm is a specialized factorization method that works best when a prime factor$p$ of$n$ has a smooth value of$p - 1$ (that is,$p-1$ factors completely into small primes).
It's one of the earliest practical improvements over trial division, simple, elegant, and effective for numbers with smooth prime factors.

#### What Problem Are We Solving?

Given a composite number$n$, we want to find a nontrivial factor$d$ such that$1 < d < n$.

This method exploits Fermat's Little Theorem:

$$
a^{p-1} \equiv 1 \pmod{p}
$$

If$p \mid n$, then$p$ divides$a^{p-1} - 1$, even if$p$ is unknown.

By computing$\gcd(a^M - 1, n)$ for a suitable$M$, we may find such$p$.

#### How Does It Work (Plain Language)

If$p$ is a prime factor of$n$ and$p-1$ divides$M$,
then$a^{M} \equiv 1 \pmod{p}$.
So$p$ divides$a^{M} - 1$.
Taking the gcd with$n$ reveals$p$.

Algorithm:

1. Choose a base$a$ (commonly 2).
2. Choose a smoothness bound$B$.
3. Compute:
   $$
   M = \text{lcm}(1, 2, 3, \ldots, B)
   $$
   and
   $$
   g = \gcd(a^M - 1, n)
   $$
4. If$1 < g < n$, return$g$ (a nontrivial factor).
   If$g = 1$, increase$B$.
   If$g = n$, choose a different$a$.

Example:
Let$n = 91 = 7 \times 13$.

Take$a = 2$,$B = 5$.
Then$M = \text{lcm}(1, 2, 3, 4, 5) = 60$.

Compute$g = \gcd(2^{60} - 1, 91)$.

$$
2^{60} - 1 \bmod 91 = 0 \implies g = 7
$$

Ok Found factor$7$

#### Tiny Code (Easy Versions)

Python Version

```python
import math
from math import gcd

def pow_mod(a, e, n):
    r = 1
    a %= n
    while e > 0:
        if e & 1:
            r = (r * a) % n
        a = (a * a) % n
        e >>= 1
    return r

def pollard_p_minus_1(n, B=10, a=2):
    M = 1
    for i in range(2, B + 1):
        M *= i // math.gcd(M, i)
    x = pow_mod(a, M, n)
    g = gcd(x - 1, n)
    if 1 < g < n:
        return g
    return None

n = int(input("Enter n: "))
factor = pollard_p_minus_1(n, B=10, a=2)
if factor:
    print(f"Nontrivial factor of {n}: {factor}")
else:
    print("No factor found, try larger B")
```

C Version

```c
#include <stdio.h>
#include <stdlib.h>

long long gcd(long long a, long long b) {
    while (b != 0) {
        long long t = b;
        b = a % b;
        a = t;
    }
    return a;
}

long long pow_mod(long long a, long long e, long long n) {
    long long r = 1 % n;
    a %= n;
    while (e > 0) {
        if (e & 1) r = (r * a) % n;
        a = (a * a) % n;
        e >>= 1;
    }
    return r;
}

long long pollard_p_minus_1(long long n, int B, long long a) {
    long long M = 1;
    for (int i = 2; i <= B; i++) {
        long long g = gcd(M, i);
        M = M / g * i;
    }
    long long x = pow_mod(a, M, n);
    long long g = gcd(x - 1, n);
    if (g > 1 && g < n) return g;
    return 0;
}

int main(void) {
    long long n;
    printf("Enter n: ");
    scanf("%lld", &n);
    long long factor = pollard_p_minus_1(n, 10, 2);
    if (factor)
        printf("Nontrivial factor: %lld\n", factor);
    else
        printf("No factor found. Try larger B.\n");
}
```

#### Why It Matters

- Excellent for factoring numbers with smooth prime factors
- Simple to implement
- Fast compared to trial division
- Builds intuition for group order and Fermat's Little Theorem

Used in:

- RSA key validation
- Elliptic curve methods (ECM) as conceptual base
- Educational number theory and cryptography

#### A Gentle Proof (Why It Works)

If$p \mid n$ and$p - 1 \mid M$,
then by Fermat's Little Theorem:
$$
a^{p-1} \equiv 1 \pmod{p}
$$
so
$$
a^{M} \equiv 1 \pmod{p}.
$$

Hence$p \mid a^M - 1$, so
$$
\gcd(a^M - 1, n) \ge p.
$$

If$p \ne n$, this gcd gives a nontrivial factor.

If$p-1$ is not smooth,$M$ must be larger to capture its factors.

#### Try It Yourself

1. Factor$91 = 7 \times 13$ with$B = 5$.
2. Try$8051 = 83 \times 97$; increase$B$ gradually.
3. Experiment with different bases$a$.
4. Compare runtime with Pollard's Rho.
5. Observe failure when$p-1$ has large prime factors.

#### Test Cases

| $n$  | Factors | Bound $B$ | Found |
| ---- | ------- | --------- | ----- |
| 91   | 7 × 13  | 5         | 7     |
| 187  | 11 × 17 | 10        | 11    |
| 589  | 19 × 31 | 15        | 19    |
| 8051 | 83 × 97 | 20        | 83    |

#### Complexity

- Time:$O(B \log^2 n)$, depends on smoothness of$p-1$
- Space:$O(1)$

Pollard's p−1 method is a mathematical keyhole, it opens composite locks when one factor's order is built from small primes.

### 518 Wheel Factorization

Wheel factorization is a deterministic optimization for trial division.
It systematically skips obvious composites by constructing a repeating pattern (the "wheel") of candidate offsets that are coprime to small primes.
This reduces the number of divisibility checks, making basic primality and factorization tests much faster.

#### What Problem Are We Solving?

We want to test whether a number $n$ is prime or factor it,
but without checking every integer up to $\sqrt{n}$.

Instead of testing all numbers, we skip those clearly divisible by small primes such as $2,3,5,7,\ldots$.
The wheel pattern encodes these skips.

#### How Does It Work (Plain Language)

1. Choose a set of small primes (the basis), for example $\{2,3,5\}$.
2. Compute the wheel size as their product:
   $$
   W = 2 \times 3 \times 5 = 30
   $$
3. Determine the residues modulo $W$ that are coprime to $W$:
   $$
   \{1,7,11,13,17,19,23,29\}
   $$
4. To test numbers up to $n$, only check candidates of the form
   $$
   kW + r \quad \text{for } r \in \text{residues}.
   $$
5. For each candidate $m$, test divisibility up to $\sqrt{m}$.

This skips about 73% of integers when using the $2\times3\times5$ wheel.

Example

Find primes $\le 50$ using the wheel with $\{2,3,5\}$.

Wheel residues mod 30:
$$
1, 7, 11, 13, 17, 19, 23, 29
$$

Candidates:
$$
1, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 49
$$

Filter by divisibility up to $\sqrt{50}$:

Primes:
$$
7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47
$$


#### Tiny Code (Easy Versions)

Python Version

```python
import math

def wheel_candidates(limit, base_primes=[2, 3, 5]):
    W = 1
    for p in base_primes:
        W *= p
    residues = [r for r in range(1, W) if all(r % p != 0 for p in base_primes)]

    candidates = []
    k = 0
    while k * W <= limit:
        for r in residues:
            num = k * W + r
            if num <= limit:
                candidates.append(num)
        k += 1
    return candidates

def is_prime(n):
    if n < 2: return False
    if n in (2, 3, 5): return True
    for p in [2, 3, 5]:
        if n % p == 0:
            return False
    for candidate in wheel_candidates(int(math.sqrt(n)) + 1):
        if n % candidate == 0:
            return False
    return True

n = int(input("Enter n: "))
print(n, "is", "prime" if is_prime(n) else "composite")
```

C Version

```c
#include <stdio.h>
#include <math.h>
#include <stdbool.h>

bool is_prime(int n) {
    if (n < 2) return false;
    if (n == 2 || n == 3 || n == 5) return true;
    if (n % 2 == 0 || n % 3 == 0 || n % 5 == 0) return false;

    int residues[] = {1, 7, 11, 13, 17, 19, 23, 29};
    int wheel = 30;
    int limit = (int)sqrt(n);

    for (int k = 0; k * wheel <= limit; k++) {
        for (int i = 0; i < 8; i++) {
            int d = k * wheel + residues[i];
            if (d > 1 && d <= limit && n % d == 0)
                return false;
        }
    }
    return true;
}

int main(void) {
    int n;
    printf("Enter n: ");
    scanf("%d", &n);
    printf("%d is %s\n", n, is_prime(n) ? "prime" : "composite");
}
```

#### Why It Matters

- Reduces trial checks by skipping obvious composites
- Speeds up trial division and sieve methods
- Reusable in wheel sieves (e.g. 2×3×5×7 wheel)
- Conceptual link between modular arithmetic and primality testing

Commonly used in:

- Optimized primality tests
- Prime sieves (wheel sieve of Eratosthenes)
- Hybrid factorization routines

#### A Gentle Proof (Why It Works)

Any number (n) that shares a factor with any base prime (p)
will appear in a non-coprime residue class modulo (W).
By checking only residues coprime to (W), we remove all such composites.

Thus, every remaining candidate is coprime to all base primes,
and we need only check divisibility by larger primes.

#### Try It Yourself

1. Build wheels for ({2, 3}), ({2, 3, 5}), and ({2, 3, 5, 7}).
2. Count how many integers each skips up to 100.
3. Compare runtime with plain trial division.
4. Use wheel to speed up a sieve implementation.
5. Print residues and visualize the pattern.

#### Test Cases

| Base Primes  | Wheel Size | Residues                     | Skip % |
| ------------ | ---------- | ---------------------------- | ------ |
| {2, 3}       | 6          | 1, 5                         | 66%    |
| {2, 3, 5}    | 30         | 1, 7, 11, 13, 17, 19, 23, 29 | 73%    |
| {2, 3, 5, 7} | 210        | 48 residues                  | 77%    |

| $n$ | Result    |
| --- | --------- |
| 7   | prime     |
| 49  | composite |
| 97  | prime     |
| 121 | composite |

#### Complexity

- Time:$O(\frac{\sqrt{n}}{\phi(W)})$, fewer checks than$O(\sqrt{n})$
- Space:$O(1)$

Wheel factorization is like building a gear that only touches promising numbers, a simple modular pattern to roll through candidates efficiently.

### 519 AKS Primality Test

The AKS primality test is the first deterministic, polynomial-time algorithm for primality testing that does not rely on unproven hypotheses.
It answers one of the oldest questions in computational number theory:

> Can we check if a number is prime in polynomial time, for sure?

Unlike probabilistic tests (Fermat, Miller–Rabin), AKS gives a definite answer, no randomness, no false positives.

#### What Problem Are We Solving?

We want to deterministically decide whether a number$n$ is prime or composite
in polynomial time, without relying on assumptions like the Riemann Hypothesis.

#### The Core Idea

A number $n$ is prime if and only if it satisfies

$$
(x + a)^n \equiv x^n + a \pmod{n}
$$

for all integers $a$.

This congruence captures the binomial property of primes:
in a prime modulus, all binomial coefficients $\binom{n}{k}$ with $0 < k < n$ vanish modulo $n$.

The AKS algorithm refines this condition into a computable primality test.

#### The Algorithm (Simplified)

Step 1. Check if $n$ is a perfect power.  
If $n = a^b$ for some integers $a, b > 1$, then $n$ is composite.

Step 2. Find the smallest integer $r$ such that

$$
\text{ord}_r(n) > (\log_2 n)^2
$$

where $\text{ord}_r(n)$ is the multiplicative order of $n$ modulo $r$.

Step 3. For each $a = 2, 3, \ldots, r$:  
If $1 < \gcd(a, n) < n$, then $n$ is composite.

Step 4. If $n \le r$, then $n$ is prime.

Step 5. For all integers $a = 1, 2, \ldots, \lfloor \sqrt{\phi(r)} \log n \rfloor$, check whether

$$
(x + a)^n \equiv x^n + a \pmod{(x^r - 1, n)}.
$$

If any test fails, $n$ is composite.  
Otherwise, $n$ is prime.


#### How Does It Work (Plain Language)

1. Perfect powers fail primality.
2. The order condition ensures$r$ is large enough to distinguish non-primes.
3. Small gcds catch trivial factors.
4. The polynomial congruence ensures that$n$ behaves like a prime under binomial expansion.

Together, these steps eliminate all composites and confirm all primes.

#### Example (Conceptual)

Let $n = 7$.

1. $7$ is not a perfect power.  
2. The smallest $r$ such that $\text{ord}_r(7) > (\log 7)^2 \approx 5.3$ is $r = 5$.  
3. No small $\gcd$ values found.  
4. $n > r$.  
5. Check
   $$
   (x + a)^7 \bmod (x^5 - 1, 7) = x^7 + a.
   $$
   All tests pass, so $n$ is prime.


#### Tiny Code (Illustrative Only)

The full AKS test is mathematically involved.
Below is a simplified prototype that captures its structure, not optimized for large (n).

Python Version

```python
import math
from math import gcd

def is_perfect_power(n):
    for b in range(2, int(math.log2(n)) + 2):
        a = round(n  (1 / b))
        if a  b == n:
            return True
    return False

def multiplicative_order(n, r):
    if gcd(n, r) != 1:
        return 0
    order = 1
    value = n % r
    while value != 1:
        value = (value * n) % r
        order += 1
        if order > r:
            return 0
    return order

def aks_is_prime(n):
    if n < 2:
        return False
    if is_perfect_power(n):
        return False

    logn2 = (math.log2(n))  2
    r = 2
    while True:
        if gcd(n, r) == 1 and multiplicative_order(n, r) > logn2:
            break
        r += 1

    for a in range(2, r + 1):
        g = gcd(a, n)
        if 1 < g < n:
            return False

    if n <= r:
        return True

    limit = int(math.sqrt(r) * math.log2(n))
    for a in range(1, limit + 1):
        # Simplified placeholder: full polynomial congruence omitted
        if pow(a, n, n) != a % n:
            return False
    return True

n = int(input("Enter n: "))
print("Prime" if aks_is_prime(n) else "Composite")
```

#### Why It Matters

- First general, deterministic, polynomial-time test
- Landmark in computational number theory (Agrawal–Kayal–Saxena, 2002)
- Theoretical foundation for all modern primality testing
- Shows primes can be recognized without randomness

#### A Gentle Proof (Why It Works)

If$n$ is prime, then by the binomial theorem:

$$
(x + a)^n = \sum_{k=0}^{n} \binom{n}{k} x^k a^{n-k} \equiv x^n + a \pmod{n}
$$

because all middle binomial coefficients are divisible by$n$.
For composites, this identity fails for some$a$,
unless$n$ has special smooth structure (caught by earlier steps).

Hence, the polynomial test is both necessary and sufficient for primality.

#### Try It Yourself

1. Test $n = 37$, $n = 97$, $n = 121$.  
2. Compare runtime with Miller–Rabin.  
3. Observe exponential slowdown for large $n$.  
4. Verify perfect power rejection.  
5. Explore small $r$ and the order function.

#### Test Cases

| $n$ | Result    | Notes                   |
| --- | --------- | ----------------------- |
| 2   | prime     | base case               |
| 7   | prime     | passes polynomial check |
| 37  | prime     | correct                 |
| 121 | composite | fails binomial identity |

#### Complexity

- Time: $O((\log n)^6)$ (original), improved to $O((\log n)^3)$  
- Space: polynomial in $\log n$

The AKS primality test transformed primality checking from an art of heuristics into a science of certainty, proving that primes are decidable in polynomial time.


### 520 Segmented Sieve

The segmented sieve is a memory-efficient variant of the Sieve of Eratosthenes, designed to generate primes within a large range $[L, R]$ without storing all numbers up to $R$.  
It is ideal when $R$ is very large (for example $10^{12}$), but the segment width $R-L$ is small enough to fit in memory.

#### What Problem Are We Solving?

We want to find all prime numbers in a range $[L, R]$, where $R$ may be extremely large.

A standard sieve up to $R$ would require $O(R)$ space, which is infeasible for $R \gg 10^8$.  
The segmented sieve solves this by dividing the range into smaller blocks and marking composites using base primes up to $\sqrt{R}$.

#### How Does It Work (Plain Language)

1. Precompute base primes up to $\sqrt{R}$ using a standard sieve.

2. For each segment $[L, R]$:

   * Mark all numbers as potentially prime.
   * For each base prime $p$:

     * Find the first multiple of $p$ in $[L, R]$:
       $$
       \text{start} = \max\left(p^2,\; \left\lceil \frac{L}{p} \right\rceil \cdot p \right)
       $$
     * Mark all multiples of $p$ as composite.

3. Remaining unmarked numbers are primes.

Repeat for each segment if the full range is too large to fit in memory.

Example: Find primes in [100, 120]

1. Compute base primes up to $\sqrt{120} = 10.9$:  
   $\{2, 3, 5, 7\}$

2. Start marking:

   * For $p = 2$: mark 100, 102, 104, …
   * For $p = 3$: mark 102, 105, 108, …
   * For $p = 5$: mark 100, 105, 110, 115, 120
   * For $p = 7$: mark 105, 112, 119

Unmarked numbers:
$$
\boxed{101, 103, 107, 109, 113}
$$
(these are the primes in [100, 120])


#### Tiny Code (Easy Versions)

Python Version

```python
import math

def simple_sieve(limit):
    mark = [True] * (limit + 1)
    mark[0] = mark[1] = False
    for i in range(2, int(math.sqrt(limit)) + 1):
        if mark[i]:
            for j in range(i * i, limit + 1, i):
                mark[j] = False
    return [i for i, is_prime in enumerate(mark) if is_prime]

def segmented_sieve(L, R):
    base_primes = simple_sieve(int(math.sqrt(R)) + 1)
    mark = [True] * (R - L + 1)
    for p in base_primes:
        start = max(p * p, ((L + p - 1) // p) * p)
        for j in range(start, R + 1, p):
            mark[j - L] = False
    if L == 1:
        mark[0] = False
    return [L + i for i, is_prime in enumerate(mark) if is_prime]

L, R = map(int, input("Enter L R: ").split())
print("Primes:", segmented_sieve(L, R))
```

C Version

```c
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <stdbool.h>

void simple_sieve(int limit, int primes, int *count) {
    bool *mark = calloc(limit + 1, sizeof(bool));
    for (int i = 2; i <= limit; i++) mark[i] = true;
    for (int i = 2; i * i <= limit; i++)
        if (mark[i])
            for (int j = i * i; j <= limit; j += i)
                mark[j] = false;

    *count = 0;
    for (int i = 2; i <= limit; i++)
        if (mark[i]) (*count)++;

    *primes = malloc(*count * sizeof(int));
    int idx = 0;
    for (int i = 2; i <= limit; i++)
        if (mark[i]) (*primes)[idx++] = i;

    free(mark);
}

void segmented_sieve(long long L, long long R) {
    int limit = sqrt(R) + 1;
    int *primes, count;
    simple_sieve(limit, &primes, &count);

    bool *mark = calloc(R - L + 1, sizeof(bool));
    for (int i = 0; i <= R - L; i++) mark[i] = true;

    for (int i = 0; i < count; i++) {
        int p = primes[i];
        long long start = (long long)p * p;
        if (start < L)
            start = ((L + p - 1) / p) * p;
        for (long long j = start; j <= R; j += p)
            mark[j - L] = false;
    }
    if (L == 1) mark[0] = false;

    for (int i = 0; i <= R - L; i++)
        if (mark[i]) printf("%lld ", L + i);
    printf("\n");

    free(primes);
    free(mark);
}

int main(void) {
    long long L, R;
    printf("Enter L R: ");
    scanf("%lld %lld", &L, &R);
    segmented_sieve(L, R);
}
```

#### Why It Matters

- Memory-efficient: handles large ranges without full sieve storage
- Essential for prime generation in big intervals (competitive programming, cryptography)
- Used in factorization and probabilistic primality tests
- Demonstrates divide-and-conquer sieving

#### A Gentle Proof (Why It Works)

Every composite number in $[L, R]$ must have a prime factor $\le \sqrt{R}$.
By marking multiples of all such base primes, all composites are removed, leaving only primes.
Each segment repeats the same logic, correctness holds per block.

#### Try It Yourself

1. Generate primes in $[100, 200]$.  
2. Test $[10^{12}, 10^{12} + 1000]$.  
3. Compare memory use with a full sieve.  
4. Implement dynamic segmenting for very large ranges.  
5. Print each segment as it is processed.

#### Test Cases

| Range $[L, R]$ | Output                  |
| -------------- | ----------------------- |
| [10, 30]       | 11, 13, 17, 19, 23, 29  |
| [100, 120]     | 101, 103, 107, 109, 113 |
| [1, 10]        | 2, 3, 5, 7              |

#### Complexity

- Time: $O((R - L + 1)\log \log R)$  
- Space: $O(\sqrt{R})$ for base primes and $O(R - L)$ for the segment

The segmented sieve works like panning for gold — processing one tray (segment) at a time to uncover primes hidden in vast numerical ranges.


# Section 53. Combinatorics 

### 521 Factorial Precomputation

Factorial precomputation is one of the most useful techniques in combinatorics and modular arithmetic.
It allows you to quickly compute values like$n!$, binomial coefficients$\binom{n}{k}$, or permutations, especially under a modulus$M$, without recomputing from scratch each time.

#### What Problem Are We Solving?

We often need$n!$ (factorial) or combinations like:

$$
\binom{n}{k} = \frac{n!}{k!(n-k)!}
$$

Directly computing factorials every time costs$O(n)$, which is too slow when$n$ is large or when we need many queries.
By precomputing all factorials once up to$n$, we can answer queries in O(1) time.

This is especially important when working modulo$M$, where division requires modular inverses.

#### How Does It Work (Plain Language)

We build an array `fact[]` such that:

$$
\text{fact}[i] = (i!) \bmod M
$$

using the recurrence:

$$
\text{fact}[0] = 1, \quad \text{fact}[i] = (i \times \text{fact}[i - 1]) \bmod M
$$

If we also need inverse factorials:

$$
\text{invfact}[n] = (\text{fact}[n])^{-1} \bmod M
$$

then we can compute all inverses in reverse:

$$
\text{invfact}[i - 1] = (i \times \text{invfact}[i]) \bmod M
$$

This lets us compute binomial coefficients fast:

$$
\binom{n}{k} = \text{fact}[n] \times \text{invfact}[k] \times \text{invfact}[n - k] \bmod M
$$

#### Example

Let$M = 10^9 + 7$,$n = 5$:

|$i$ |$i!$ |$i! \bmod M$ |
| ----- | ------ | -------------- |
| 0     | 1      | 1              |
| 1     | 1      | 1              |
| 2     | 2      | 2              |
| 3     | 6      | 6              |
| 4     | 24     | 24             |
| 5     | 120    | 120            |

$$
\binom{5}{2} = \frac{5!}{2! \cdot 3!} = \frac{120}{2 \cdot 6} = 10
$$

With modular inverses, the same holds under$M$.

#### Tiny Code (Easy Versions)

Python Version

```python
M = 109 + 7

def precompute_factorials(n):
    fact = [1] * (n + 1)
    for i in range(1, n + 1):
        fact[i] = (fact[i - 1] * i) % M
    return fact

def modinv(a, m=M):
    return pow(a, m - 2, m)  # Fermat's Little Theorem

def precompute_inverses(fact):
    n = len(fact) - 1
    invfact = [1] * (n + 1)
    invfact[n] = modinv(fact[n])
    for i in range(n, 0, -1):
        invfact[i - 1] = (invfact[i] * i) % M
    return invfact

n = 10
fact = precompute_factorials(n)
invfact = precompute_inverses(fact)

def nCr(n, r):
    if r < 0 or r > n: return 0
    return fact[n] * invfact[r] % M * invfact[n - r] % M

print("5C2 =", nCr(5, 2))
```

C Version

```c
#include <stdio.h>
#define M 1000000007
#define MAXN 1000000

long long fact[MAXN + 1], invfact[MAXN + 1];

long long modpow(long long a, long long e) {
    long long r = 1;
    while (e > 0) {
        if (e & 1) r = r * a % M;
        a = a * a % M;
        e >>= 1;
    }
    return r;
}

void precompute_factorials(int n) {
    fact[0] = 1;
    for (int i = 1; i <= n; i++)
        fact[i] = fact[i - 1] * i % M;

    invfact[n] = modpow(fact[n], M - 2);
    for (int i = n; i >= 1; i--)
        invfact[i - 1] = invfact[i] * i % M;
}

long long nCr(int n, int r) {
    if (r < 0 || r > n) return 0;
    return fact[n] * invfact[r] % M * invfact[n - r] % M;
}

int main(void) {
    precompute_factorials(1000000);
    printf("5C2 = %lld\n", nCr(5, 2));
}
```

#### Why It Matters

- Converts$O(n)$ recomputation into O(1) queries
- Fundamental for combinatorics, DP, and modular counting
- Enables quick computation of:

  * Binomial coefficients
  * Multiset combinations
  * Probability computations
  * Catalan numbers

Used in:

- Combinatorial DP
- Probability and expectation problems
- Modular combinatorics

#### A Gentle Proof (Why It Works)

Factorials grow recursively:
$$
n! = n \cdot (n-1)!
$$
So precomputing stores each step once.
Modulo arithmetic preserves multiplication structure:
$$
(ab) \bmod M = ((a \bmod M) \cdot (b \bmod M)) \bmod M
$$
and modular inverses exist when$M$ is prime, by Fermat's Little Theorem:
$$
a^{M-1} \equiv 1 \pmod{M} \implies a^{-1} \equiv a^{M-2} \pmod{M}
$$

#### Try It Yourself

1. Precompute up to$n = 10^6$ and print factorials.
2. Compute$1000! \bmod 10^9+7$.
3. Verify$\binom{n}{k} = \binom{n}{n-k}$.
4. Add memoization for varying$M$.
5. Extend to double factorials or multinomial coefficients.

#### Test Cases

|$n$ |$k$ |$n! \bmod M$ |$\binom{n}{k} \bmod M$ |
| ----- | ----- | -------------- | ------------------------ |
| 5     | 2     | 120            | 10                       |
| 10    | 3     | 3628800        | 120                      |
| 100   | 50    |,              | 538992043                |

#### Complexity

- Precomputation:$O(n)$
- Query:$O(1)$
- Space:$O(n)$

Factorial precomputation is your lookup table for combinatorics, prepare once, compute instantly.

### 522 nCr Computation

nCr computation (binomial coefficient calculation) is the backbone of combinatorics, it counts the number of ways to choose$r$ elements from a set of$n$ elements, without regard to order.
It shows up in counting, probability, DP, and combinatorial identities.

#### What Problem Are We Solving?

We want to compute:

$$
\binom{n}{r} = \frac{n!}{r!(n-r)!}
$$

directly or modulo a large prime$M$, efficiently.

For small$n$, this can be done by direct multiplication and division.
For large$n$, we must use modular arithmetic and modular inverses, since division is not defined under modulo.

#### How Does It Work (Plain Language)

We can compute$\binom{n}{r}$ in several ways:

1. Multiplicative formula (direct):

   $$
   \binom{n}{r} = \prod_{i=1}^{r} \frac{n - r + i}{i}
   $$

   Works well when$n$ and$r$ are moderate ((< 10^6)).

2. Dynamic Programming (Pascal's Triangle):

   $$
   \binom{n}{r} = \binom{n-1}{r-1} + \binom{n-1}{r}
   $$

   with base cases$\binom{n}{0} = 1$,$\binom{n}{n} = 1$.

3. Factorial precomputation (for modulo):

   Using precomputed arrays:
   $$
   \binom{n}{r} = \text{fact}[n] \cdot \text{invfact}[r] \cdot \text{invfact}[n-r] \bmod M
   $$

#### Example

Compute$\binom{5}{2}$:

$$
\binom{5}{2} = \frac{5!}{2! \cdot 3!} = \frac{120}{12} = 10
$$

Check with Pascal's identity:

$$
\binom{5}{2} = \binom{4}{1} + \binom{4}{2} = 4 + 6 = 10
$$

#### Tiny Code (Easy Versions)

1. Multiplicative Formula (No Modulo)

```python
def nCr(n, r):
    if r < 0 or r > n:
        return 0
    r = min(r, n - r)
    res = 1
    for i in range(1, r + 1):
        res = res * (n - r + i) // i
    return res

print(nCr(5, 2))  # 10
```

2. Factorial + Modular Inverse

```python
M = 109 + 7

def modpow(a, e, m=M):
    r = 1
    while e > 0:
        if e & 1:
            r = (r * a) % m
        a = (a * a) % m
        e >>= 1
    return r

def nCr_mod(n, r):
    if r < 0 or r > n:
        return 0
    fact = [1] * (n + 1)
    for i in range(1, n + 1):
        fact[i] = (fact[i - 1] * i) % M
    invfact = [1] * (n + 1)
    invfact[n] = modpow(fact[n], M - 2)
    for i in range(n, 0, -1):
        invfact[i - 1] = (invfact[i] * i) % M
    return fact[n] * invfact[r] % M * invfact[n - r] % M

print(nCr_mod(5, 2))  # 10
```

3. Pascal's Triangle (Dynamic Programming)

```python
def build_pascal(n):
    C = [[0]*(n+1) for _ in range(n+1)]
    for i in range(n+1):
        C[i][0] = C[i][i] = 1
        for j in range(1, i):
            C[i][j] = C[i-1][j-1] + C[i-1][j]
    return C

pascal = build_pascal(10)
print(pascal[5][2])  # 10
```

#### Why It Matters

- Fundamental to combinatorics

- Used in:

  * Binomial expansions
  * Probability (e.g., hypergeometric distributions)
  * Dynamic programming (e.g., counting paths)
  * Number theory (Lucas theorem)
  * Modular arithmetic combinatorics

- The backbone of:

  * Catalan numbers
  * Pascal's triangle
  * Inclusion–exclusion principles

#### A Gentle Proof (Why It Works)

Combinatorially:

- To choose$r$ items from$n$, either include a specific item or exclude it.

So:
$$
\binom{n}{r} = \binom{n-1}{r-1} + \binom{n-1}{r}
$$
with boundaries:
$$
\binom{n}{0} = \binom{n}{n} = 1
$$

Multiplicatively:
$$
\frac{n!}{r!(n-r)!} = \frac{n}{1} \times \frac{n-1}{2} \times \cdots \times \frac{n-r+1}{r}
$$

#### Try It Yourself

1. Compute$\binom{10}{3}$ manually and using code.
2. Generate the 6th row of Pascal's triangle.
3. Verify symmetry$\binom{n}{r} = \binom{n}{n-r}$.
4. Implement modulo version for$n = 10^6$.
5. Use nCr to compute Catalan numbers:
   $$
   C_n = \frac{1}{n+1}\binom{2n}{n}
   $$

#### Test Cases

| (n) | (r) | Result    | Method    |
| --- | --- | --------- | --------- |
| 5   | 2   | 10        | factorial |
| 10  | 3   | 120       | DP        |
| 100 | 50  | 538992043 | modulo    |

#### Complexity

| Method                | Time             | Space    |
| --------------------- | ---------------- | -------- |
| Multiplicative        | (O(r))           | (O(1))   |
| DP (Pascal)           | (O(n^2))         | (O(n^2)) |
| Precomputed factorial | (O(1)) per query | (O(n))   |

nCr is the counting lens of algorithms, every subset, combination, and selection passes through it.

### 523 Pascal's Triangle

Pascal's Triangle is the geometric arrangement of binomial coefficients.
Each entry represents $\binom{n}{r}$, and every row builds upon the previous one.
It's not just a pretty triangle, it's the living structure of combinatorics, binomial expansions, and dynamic programming.

#### What Problem Are We Solving?

We want to compute binomial coefficients efficiently and recurrently, without factorials or modular inverses.

We use the recursive identity:

$$
\binom{n}{r}=\binom{n-1}{r-1}+\binom{n-1}{r}
$$

with base cases:

$$
\binom{n}{0}=\binom{n}{n}=1
$$

This recurrence builds all combinations in a simple triangle, each number is the sum of the two above it.

#### How Does It Work (Plain Language)

Start with row 0: `[1]`
Each new row begins and ends with 1, and every middle element is the sum of two neighbors above.

Example:

| Row | Values        |
| --- | ------------- |
| 0   | 1             |
| 1   | 1 1           |
| 2   | 1 2 1         |
| 3   | 1 3 3 1       |
| 4   | 1 4 6 4 1     |
| 5   | 1 5 10 10 5 1 |

The value at row $n$, column $r$ equals $\binom{n}{r}$.

#### Example

Compute $\binom{5}{2}$ from Pascal's triangle:

Row 5: 1 5 10 10 5 1
→ $\binom{5}{2}=10$

Check recurrence:

$$
\binom{5}{2}=\binom{4}{1}+\binom{4}{2}=4+6=10
$$

#### Tiny Code (Easy Versions)

Python Version

```python
def pascal_triangle(n):
    C = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        C[i][0] = C[i][i] = 1
        for j in range(1, i):
            C[i][j] = C[i - 1][j - 1] + C[i - 1][j]
    return C

C = pascal_triangle(6)
for i in range(6):
    print(C[i][:i + 1])

print("C(5,2) =", C[5][2])
```

C Version

```c
#include <stdio.h>

void pascal_triangle(int n) {
    int C[n + 1][n + 1];
    for (int i = 0; i <= n; i++) {
        for (int j = 0; j <= i; j++) {
            if (j == 0 || j == i)
                C[i][j] = 1;
            else
                C[i][j] = C[i - 1][j - 1] + C[i - 1][j];
            printf("%d ", C[i][j]);
        }
        printf("\n");
    }
}

int main(void) {
    pascal_triangle(6);
    return 0;
}
```

#### Why It Matters

- Efficient $O(n^2)$ construction of all $\binom{n}{r}$
- No factorials, no large-number overflows for small $n$
- Foundation for:

  * Binomial expansions:
    $(a+b)^n=\sum_{r=0}^n\binom{n}{r}a^{n-r}b^r$
  * Combinatorial DP: counting subsets, paths, partitions
  * Probability computations

Also connects to:

- Fibonacci (diagonal sums)
- Powers of 2 (row sums)
- Sierpinski triangles (mod 2 pattern)

#### A Gentle Proof (Why It Works)

Each term counts ways to choose $r$ items from $n$:
either include a particular element or exclude it.

So:

$$
\binom{n}{r}=\binom{n-1}{r-1}+\binom{n-1}{r}
$$

where:

- $\binom{n-1}{r-1}$: choose $r-1$ from remaining after including
- $\binom{n-1}{r}$: choose $r$ from remaining after excluding

This recurrence relation constructs the entire triangle layer by layer.

#### Try It Yourself

1. Print the first 10 rows of Pascal's triangle.
2. Verify that the sum of row $n$ equals $2^n$.
3. Use triangle values to expand $(a+b)^5$.
4. Visualize pattern mod 2, get the Sierpinski triangle.
5. Use diagonals to build Fibonacci sequence.

#### Test Cases

| $n$ | $r$ | $\binom{n}{r}$ | Triangle Row             |
| --- | --- | -------------- | ------------------------ |
| 5   | 2   | 10             | [1, 5, 10, 10, 5, 1]     |
| 6   | 3   | 20             | [1, 6, 15, 20, 15, 6, 1] |

#### Complexity

- Time: $O(n^2)$
- Space: $O(n^2)$, or $O(n)$ with row compression

Pascal's Triangle is combinatorics in motion, a growing, recursive landscape where every number remembers its ancestors.

### 524 Multiset Combination

Multiset combination counts selections with repetition allowed, choosing $r$ elements from $n$ types, where each type can appear multiple times.
It's the combinatorial backbone for problems like compositions, integer partitions, stars and bars, and bag combinations.

#### What Problem Are We Solving?

We want to count the number of ways to choose $r$ items from $n$ types when duplicates are allowed.

For example, with types ${A,B,C}$ and $r=2$, valid selections are:
$$
{AA,AB,AC,BB,BC,CC}
$$
That's 6 combinations, not $\binom{3}{2}=3$.

So the formula is:

$$
\text{MultisetCombination}(n,r)=\binom{n+r-1}{r}
$$

This is the "stars and bars" formula.

#### How Does It Work (Plain Language)

Think of $r$ identical stars (items) separated into $n$ groups by $(n-1)$ dividers (bars).

Example: $n=3$, $r=4$

We need to arrange 4 stars and 2 bars:

```
- * | * | *
```

Each arrangement corresponds to one combination.

Total arrangements:

$$
\binom{n+r-1}{r}=\binom{6}{4}=15
$$

So there are 15 ways to pick 4 items from 3 types with repetition.

#### Example

Let $n=3$ (types A, B, C), $r=2$.
Formula:

$$
\binom{3+2-1}{2}=\binom{4}{2}=6
$$

List all:
AA, AB, AC, BB, BC, CC

Ok Matches formula.

#### Tiny Code (Easy Versions)

Python Version

```python
from math import comb

def multiset_combination(n, r):
    return comb(n + r - 1, r)

print("Combinations (n=3, r=2):", multiset_combination(3, 2))
```

Modulo Version (using factorial precomputation)

```python
M = 109 + 7

def modpow(a, e, m=M):
    r = 1
    while e > 0:
        if e & 1:
            r = (r * a) % m
        a = (a * a) % m
        e >>= 1
    return r

def modinv(a):
    return modpow(a, M - 2)

def nCr_mod(n, r):
    if r < 0 or r > n:
        return 0
    fact = [1] * (n + 1)
    for i in range(1, n + 1):
        fact[i] = fact[i - 1] * i % M
    return fact[n] * modinv(fact[r]) % M * modinv(fact[n - r]) % M

def multiset_combination_mod(n, r):
    return nCr_mod(n + r - 1, r)

print("Multiset Combination (n=3, r=2):", multiset_combination_mod(3, 2))
```

#### Why It Matters

- Models combinations with replacement
- Appears in:

  * Counting multisets / bags
  * Integer partitioning
  * Polynomial coefficient enumeration
  * Distributing identical balls into boxes
- Key in DP over multisets, generating functions, and probability spaces

#### A Gentle Proof (Why It Works)

Represent $r$ identical items as stars `*`
and $n-1$ dividers as bars `|`.

Example: $n=4$, $r=3$
We have $r+(n-1)=6$ symbols.
Choosing positions of $r$ stars:

$$
\binom{r+n-1}{r}
$$

Each unique arrangement corresponds to a multiset.

Hence, total combinations = $\binom{n+r-1}{r}$.

#### Try It Yourself

1. Count ways to choose 3 fruits from {apple, banana, cherry}.
2. Compute $\text{MultisetCombination}(5,2)$ and list a few examples.
3. Build a DP table using Pascal's triangle recurrence:
   $$
   f(n,r)=f(n,r-1)+f(n-1,r)
   $$
4. Use modulo arithmetic for large $n$.
5. Visualize "stars and bars" layouts for $n=4, r=3$.

#### Test Cases

| $n$ | $r$ | $\binom{n+r-1}{r}$ | Result |
| --- | --- | ------------------ | ------ |
| 3   | 2   | $\binom{4}{2}=6$   | 6      |
| 4   | 3   | $\binom{6}{3}=20$  | 20     |
| 2   | 5   | $\binom{6}{5}=6$   | 6      |

#### Complexity

- Time: $O(1)$ (with precomputed factorials)
- Space: $O(n+r)$ (for factorial storage)

Multiset combinations open up counting beyond uniqueness, when repetition is a feature, not a flaw.

### 525 Permutation Generation

Permutation generation is the process of listing all possible arrangements of a set of elements, the order now matters.
It's one of the most fundamental operations in combinatorics, recursion, and search algorithms, powering brute-force solvers, lexicographic enumeration, and backtracking frameworks.

#### What Problem Are We Solving?

We want to generate all permutations of a collection of size $n$, that is, every possible ordering.

For example, for ${1, 2, 3}$, the permutations are:

$$
{1,2,3},{1,3,2},{2,1,3},{2,3,1},{3,1,2},{3,2,1}
$$

There are $n!$ total permutations.

#### How Does It Work (Plain Language)

There are multiple strategies:

1. Recursive Backtracking

   * Choose an element
   * Permute the remaining
   * Combine

2. Lexicographic (Next Permutation)

   * Generate in sorted order by finding next lexicographic successor

3. Heap's Algorithm

   * Swap-based iterative generation in $O(n!)$ time, $O(n)$ space

Recursive Approach Example

To generate all permutations of ${1,2,3}$:

- Fix 1 → permute ${2,3}$ → ${1,2,3},{1,3,2}$
- Fix 2 → permute ${1,3}$ → ${2,1,3},{2,3,1}$
- Fix 3 → permute ${1,2}$ → ${3,1,2},{3,2,1}$

#### Tiny Code (Easy Versions)

Python Version (Recursive Backtracking)

```python
def permute(arr, path=[]):
    if not arr:
        print(path)
        return
    for i in range(len(arr)):
        permute(arr[:i] + arr[i+1:], path + [arr[i]])

permute([1, 2, 3])
```

Python Version (Using Built-in)

```python
from itertools import permutations

for p in permutations([1, 2, 3]):
    print(p)
```

C Version (Backtracking)

```c
#include <stdio.h>

void swap(int *a, int *b) {
    int t = *a; *a = *b; *b = t;
}

void permute(int *arr, int l, int r) {
    if (l == r) {
        for (int i = 0; i <= r; i++) printf("%d ", arr[i]);
        printf("\n");
        return;
    }
    for (int i = l; i <= r; i++) {
        swap(&arr[l], &arr[i]);
        permute(arr, l + 1, r);
        swap(&arr[l], &arr[i]); // backtrack
    }
}

int main(void) {
    int arr[] = {1, 2, 3};
    permute(arr, 0, 2);
}
```

#### Why It Matters

- Foundation for brute-force search, backtracking, and enumeration
- Powers:

  * Traveling Salesman Problem (TSP) brute-force
  * Permutation testing in statistics
  * Order-based search in AI / combinatorial optimization
- Useful in:

  * Combinatorics
  * String generation
  * Constraint solving

#### A Gentle Proof (Why It Works)

Each position in the permutation can hold one of the remaining unused elements.

The recurrence:

$$
P(n)=n\cdot P(n-1)
$$

Base case $P(1)=1$, so $P(n)=n!$.

Every branch in recursion represents one arrangement, total branches = $n!$.

#### Try It Yourself

1. Generate all permutations of `[1,2,3]`.
2. Print count for `n=4` (should be $24$).
3. Modify code to store, not print, permutations.
4. Implement lexicographic next-permutation method.
5. Use permutations to test all possible password orders.

#### Test Cases

| Input     | Output (Permutations)                                | Count |
| --------- | ---------------------------------------------------- | ----- |
| [1,2]     | [1,2], [2,1]                                         | 2     |
| [1,2,3]   | [1,2,3], [1,3,2], [2,1,3], [2,3,1], [3,1,2], [3,2,1] | 6     |
| [‘A',‘B'] | AB, BA                                               | 2     |

#### Complexity

- Time: $O(n!)$, total permutations
- Space: $O(n)$ recursion depth

Permutation generation is exhaustive creativity, exploring every possible order, every possible path.

### 526 Next Permutation

The Next Permutation algorithm finds the next lexicographically greater permutation of a given sequence.
It's a building block for lexicographic enumeration, letting you step through permutations in sorted order without generating all at once.

If no larger permutation exists (the sequence is in descending order), it resets to the smallest (ascending order).

#### What Problem Are We Solving?

Given a permutation (arranged sequence), find the next lexicographically larger one.

Example:
From `[1, 2, 3]`, the next permutation is `[1, 3, 2]`.
From `[3, 2, 1]`, there's no larger one, so we reset to `[1, 2, 3]`.

We want a method that transforms a sequence in-place, in O(n) time.

#### How Does It Work (Plain Language)

Imagine your sequence as a number, e.g. `[1, 2, 3]` represents 123.
You want the next bigger number made from the same digits.

Algorithm steps:

1. Find pivot, scan from right to left to find the first index `i` where
   $a[i] < a[i+1]$
   (This is the point where the next permutation can be increased.)

2. Find successor, from the right, find the smallest $a[j] > a[i]$.

3. Swap $a[i]$ and $a[j]$.

4. Reverse the suffix starting at $i+1$ (turn descending tail into ascending).

If no pivot is found, reverse the whole array (last permutation → first).

Example

From `[1, 2, 3]`:

1. Pivot at `2` (`a[1]`), since `2 < 3`.
2. Successor = `3`.
3. Swap → `[1, 3, 2]`.
4. Reverse suffix `[2]` → unchanged.
   Ok Next permutation = `[1, 3, 2]`.

From `[3, 2, 1]`:
No pivot (entirely descending) → reverse to `[1, 2, 3]`.

#### Tiny Code (Easy Versions)

Python Version

```python
def next_permutation(arr):
    n = len(arr)
    i = n - 2
    while i >= 0 and arr[i] >= arr[i + 1]:
        i -= 1
    if i == -1:
        arr.reverse()
        return False  # last permutation
    j = n - 1
    while arr[j] <= arr[i]:
        j -= 1
    arr[i], arr[j] = arr[j], arr[i]
    arr[i + 1:] = reversed(arr[i + 1:])
    return True

# Example
arr = [1, 2, 3]
next_permutation(arr)
print(arr)  # [1, 3, 2]
```

C Version

```c
#include <stdio.h>
#include <stdbool.h>

void reverse(int *a, int l, int r) {
    while (l < r) {
        int t = a[l];
        a[l++] = a[r];
        a[r--] = t;
    }
}

bool next_permutation(int *a, int n) {
    int i = n - 2;
    while (i >= 0 && a[i] >= a[i + 1]) i--;
    if (i < 0) {
        reverse(a, 0, n - 1);
        return false;
    }
    int j = n - 1;
    while (a[j] <= a[i]) j--;
    int t = a[i]; a[i] = a[j]; a[j] = t;
    reverse(a, i + 1, n - 1);
    return true;
}

int main(void) {
    int a[] = {1, 2, 3};
    int n = 3;
    next_permutation(a, n);
    for (int i = 0; i < n; i++) printf("%d ", a[i]);
}
```

#### Why It Matters

- Core tool for lexicographic enumeration
- Powers:

  * Combinatorial iteration
  * Search problems (e.g. TSP brute-force)
  * String/sequence next-step generation
- Used in C++ STL (`std::next_permutation`)

#### A Gentle Proof (Why It Works)

The suffix after pivot $a[i]$ is strictly decreasing, meaning it's the largest arrangement of that suffix.
To get the next larger permutation:

- Increase $a[i]$ minimally (swap with next larger element $a[j]$)
- Then reorder the suffix to the smallest possible arrangement (ascending)

Thus ensuring the next lexicographic order.

#### Try It Yourself

1. Step through `[1, 2, 3]` → `[1, 3, 2]` → `[2, 1, 3]` → …
2. Write loop to print all permutations in order.
3. Try on characters: `['A', 'B', 'C']`.
4. Test `[3, 2, 1]`, should reset to `[1, 2, 3]`.
5. Modify to generate previous permutation.

#### Test Cases

| Input   | Output (Next) | Note        |
| ------- | ------------- | ----------- |
| [1,2,3] | [1,3,2]       | simple      |
| [1,3,2] | [2,1,3]       | middle      |
| [3,2,1] | [1,2,3]       | wrap-around |
| [1,1,5] | [1,5,1]       | duplicates  |

#### Complexity

- Time: $O(n)$
- Space: $O(1)$

Next Permutation is your lexicographic stepper, one small swap, one tidy reverse, one giant leap in order.

### 527 Subset Generation

Subset generation is the process of listing every possible subset of a given set, including the empty set and the full set.
It's a cornerstone of combinatorial enumeration, power set construction, and many backtracking and bitmask algorithms.

#### What Problem Are We Solving?

Given a set of $n$ elements, generate all its subsets.

For example, for ${1,2,3}$, the subsets (the power set) are:

$$
\varnothing,{1},{2},{3},{1,2},{1,3},{2,3},{1,2,3}
$$

Total number of subsets = $2^n$.

#### How Does It Work (Plain Language)

There are two classic methods:

1. Recursive / Backtracking
   Choose whether to include or exclude each element.
   Build subsets depth-first.

2. Bitmask Enumeration
   Represent subsets using a binary mask of length $n$,
   where bit $i$ indicates whether the $i$-th element is included.

Example for ${1,2,3}$:

| Mask | Binary | Subset        |
| ---- | ------ | ------------- |
| 0    | 000    | $\varnothing$ |
| 1    | 001    | ${3}$         |
| 2    | 010    | ${2}$         |
| 3    | 011    | ${2,3}$       |
| 4    | 100    | ${1}$         |
| 5    | 101    | ${1,3}$       |
| 6    | 110    | ${1,2}$       |
| 7    | 111    | ${1,2,3}$     |

#### Tiny Code (Easy Versions)

Python Version (Recursive)

```python
def subsets(arr, path=[], i=0):
    if i == len(arr):
        print(path)
        return
    # Exclude current
    subsets(arr, path, i + 1)
    # Include current
    subsets(arr, path + [arr[i]], i + 1)

subsets([1, 2, 3])
```

Python Version (Bitmask)

```python
def subsets_bitmask(arr):
    n = len(arr)
    for mask in range(1 << n):
        subset = [arr[i] for i in range(n) if mask & (1 << i)]
        print(subset)

subsets_bitmask([1, 2, 3])
```

C Version (Bitmask)

```c
#include <stdio.h>

void subsets(int *arr, int n) {
    int total = 1 << n;
    for (int mask = 0; mask < total; mask++) {
        printf("{ ");
        for (int i = 0; i < n; i++) {
            if (mask & (1 << i))
                printf("%d ", arr[i]);
        }
        printf("}\n");
    }
}

int main(void) {
    int arr[] = {1, 2, 3};
    subsets(arr, 3);
}
```

#### Why It Matters

- Powers:

  * Enumerating possibilities in combinatorial problems
  * Subset sum, knapsack, bitmask DP
  * Search and optimization
- Basis for:

  * Power sets
  * Inclusion–exclusion principle
  * State-space exploration

Every subset represents one state, one choice pattern, one branch in the combinatorial tree.

#### A Gentle Proof (Why It Works)

Each element can be included or excluded independently.
That gives $2$ choices per element, so:

$$
\text{Total subsets} = 2 \times 2 \times \cdots \times 2 = 2^n
$$

Each binary mask uniquely encodes a subset.
Thus both recursive and bitmask methods visit all $2^n$ subsets exactly once.

#### Try It Yourself

1. Generate all subsets of `[1,2,3,4]`.
2. Count subsets of size 2 only.
3. Print subsets in lexicographic order.
4. Filter subsets summing to 5.
5. Combine with DP for subset sum.

#### Test Cases

| Input   | Output (Subsets)    | Count |
| ------- | ------------------- | ----- |
| [1,2]   | [], [1], [2], [1,2] | 4     |
| [1,2,3] | 8 subsets           | 8     |
| []      | []                  | 1     |

#### Complexity

- Time: $O(2^n \cdot n)$ (each subset generation takes $O(n)$)
- Space: $O(n)$ recursion stack or temporary list

Subset generation is the combinatorial chorus, every yes/no choice joins the melody of all possibilities.

### 528 Gray Code Generation

Gray codes list all $2^n$ binary strings of length $n$ so that consecutive codes differ in exactly one bit (Hamming distance $1$).
They are ideal for enumerations where small step changes reduce errors or avoid expensive recomputation.

#### What Problem Are We Solving?

Generate an ordering of all $n$-bit strings
$$
g_0,g_1,\dots,g_{2^n-1}
$$
such that for each $k\ge 1$, the Hamming distance $\mathrm{dist}(g_{k-1},g_k)=1$.

#### How Does It Work (Plain Language)

There are two classic constructions:

1. Reflect and prefix (recursive)

- Base: $G_1=[0,1]$
- To build $G_{n}$ from $G_{n-1}$:

  * Take $G_{n-1}$ and prefix each code with $0$
  * Take the reverse of $G_{n-1}$ and prefix each code with $1$
- Concatenate the two lists

2. Bit trick (iterative, index to Gray)

- The $k$-th Gray code is
  $$
  g(k)=k\oplus (k!!\gg!1)
  $$
  where $\oplus$ is bitwise XOR and $\gg$ is right shift

Both yield the same sequence up to renaming.

#### Example

For $n=3$, the reflect-and-prefix method gives:

- Start with $G_1$: $0,1$
- $G_2$: prefix $0$ to get $00,01$, prefix $1$ to reversed $G_1$ to get $11,10$
- $G_2=[00,01,11,10]$
- $G_3$: prefix $0$ to $G_2$ $\to$ $000,001,011,010$
  prefix $1$ to reversed $G_2$ $\to$ $110,111,101,100$

Final $G_3$:
$$
000,001,011,010,110,111,101,100
$$
Each consecutive pair differs in exactly one bit.

#### Tiny Code (Easy Versions)

Python Version: reflect and prefix

```python
def gray_reflect(n):
    codes = ["0", "1"]
    if n == 1:
        return codes
    for _ in range(2, n + 1):
        left = ["0" + c for c in codes]
        right = ["1" + c for c in reversed(codes)]
        codes = left + right
    return codes

print(gray_reflect(3))  # ['000','001','011','010','110','111','101','100']
```

Python Version: bit trick $g(k)=k\oplus(k>>1)$

```python
def gray_bit(n):
    return [k ^ (k >> 1) for k in range(1 << n)]

def to_bits(x, n):
    return format(x, f"0{n}b")

n = 3
codes = [to_bits(v, n) for v in gray_bit(n)]
print(codes)  # ['000','001','011','010','110','111','101','100']
```

C Version: bit trick

```c
#include <stdio.h>

unsigned gray(unsigned k) {
    return k ^ (k >> 1);
}

void print_binary(unsigned x, int n) {
    for (int i = n - 1; i >= 0; --i)
        putchar((x & (1u << i)) ? '1' : '0');
}

int main(void) {
    int n = 3;
    unsigned total = 1u << n;
    for (unsigned k = 0; k < total; ++k) {
        unsigned g = gray(k);
        print_binary(g, n);
        putchar('\n');
    }
    return 0;
}
```

#### Why It Matters

- Consecutive states differ by one bit, reducing switching errors and glitches in hardware encoders and ADCs
- Useful in gray-code counters, Hamiltonian paths on hypercubes, and search over subsets where incremental updates are cheap
- Common in combinatorial generation to minimize change between outputs

#### A Gentle Proof (Why It Works)

For the reflect-and-prefix method:

- The first half is $0\cdot G_{n-1}$ where neighbors already differ in one bit
- The second half is $1\cdot \mathrm{rev}(G_{n-1})$ where neighbors still differ in one bit
- The boundary pair differs only in the first bit because we go from $0\cdot g_0$ to $1\cdot g_0$
  By induction on $n$, consecutive codes differ in exactly one bit.

For the bit trick:

- Write $g(k)=k\oplus(k>>1)$ and $g(k+1)=(k+1)\oplus((k+1)>>1)$
- One can check that $g(k)$ and $g(k+1)$ differ only in the lowest bit that changes when incrementing $k$
  Hence Hamming distance is $1$.

#### Try It Yourself

1. Generate Gray codes for $n=1\ldots 5$ using both methods and compare.
2. Map Gray codes back to binary: inverse is $b_0=g_0$, and $b_i=b_{i-1}\oplus g_i$ for $i\ge 1$.
3. Iterate all subsets of ${0,\dots,n-1}$ in Gray order and maintain an incremental sum by flipping one element each step.
4. Use Gray order to traverse vertices of the $n$-cube.

#### Test Cases

| $n$ | Expected prefix of sequence       |
| --- | --------------------------------- |
| 1   | $0,1$                             |
| 2   | $00,01,11,10$                     |
| 3   | $000,001,011,010,110,111,101,100$ |

Also verify that every adjacent pair differs in exactly one bit.

#### Complexity

- Time: $O(2^n)$ to output all codes
- Space: $O(n)$ per code (or $O(1)$ extra using bit trick)

Gray codes give a one-bit-at-a-time tour of the hypercube, perfect for smooth transitions in algorithms and hardware.

### 529 Catalan Number DP

Catalan numbers count a wide variety of recursive structures, from valid parentheses strings to binary trees, polygon triangulations, and non-crossing paths.
They are the backbone of combinatorial recursion, dynamic programming, and context-free grammars.

#### What Problem Are We Solving?

We want to compute the $n$-th Catalan number, $C_n$, which counts:

- Valid parentheses sequences of length $2n$
- Binary search trees with $n$ nodes
- Triangulations of an $(n+2)$-gon
- Non-crossing partitions, Dyck paths, etc.

The recursive formula is:

$$
C_0 = 1
$$

$$
C_n = \sum_{i=0}^{n-1} C_i \cdot C_{n-1-i}
$$

Alternatively, the closed-form expression using binomial coefficients is:

$$
C_n = \frac{1}{n+1}\binom{2n}{n}
$$

#### How Does It Work (Plain Language)

Think of $C_n$ as counting ways to split a structure into two balanced parts.

Example (valid parentheses):
Every sequence starts with `(` and pairs with a matching `)` that divides the sequence into two smaller valid parts.
If the left part has $C_i$ ways and the right part has $C_{n-1-i}$ ways, the total is their product.
Summing across all possible splits gives the full $C_n$.

#### Example

Let's compute the first few:

| $n$ | $C_n$ |
| --- | ----- |
| 0   | 1     |
| 1   | 1     |
| 2   | 2     |
| 3   | 5     |
| 4   | 14    |
| 5   | 42    |

#### Tiny Code (Easy Versions)

Python (DP approach)

```python
def catalan(n):
    C = [0] * (n + 1)
    C[0] = 1
    for i in range(1, n + 1):
        for j in range(i):
            C[i] += C[j] * C[i - 1 - j]
    return C[n]

for i in range(6):
    print(f"C({i}) = {catalan(i)}")
```

Python (Binomial formula)

```python
from math import comb

def catalan_binom(n):
    return comb(2 * n, n) // (n + 1)

print([catalan_binom(i) for i in range(6)])
```

C Version (DP approach)

```c
#include <stdio.h>

unsigned long catalan(int n) {
    unsigned long C[n + 1];
    C[0] = 1;
    for (int i = 1; i <= n; i++) {
        C[i] = 0;
        for (int j = 0; j < i; j++)
            C[i] += C[j] * C[i - 1 - j];
    }
    return C[n];
}

int main(void) {
    for (int i = 0; i <= 5; i++)
        printf("C(%d) = %lu\n", i, catalan(i));
}
```

#### Why It Matters

Catalan numbers appear in many fundamental combinatorial and algorithmic contexts:

- Combinatorics: Dyck paths, lattice paths, stack-sortable permutations
- Dynamic programming: counting tree structures
- Parsing theory: number of valid parse trees
- Geometry: triangulations of convex polygons

They are a universal count for balanced recursive structures.

#### A Gentle Proof (Why It Works)

Each Catalan object splits into two smaller ones around a root:

$$
C_n = \sum_{i=0}^{n-1} C_i \cdot C_{n-1-i}
$$

This recurrence mirrors binary tree formation, left subtree size $i$, right subtree $n-1-i$.

The closed form follows from the binomial identity:

$$
C_n = \frac{1}{n+1}\binom{2n}{n}
$$

which arises from solving the recurrence using generating functions.

#### Try It Yourself

1. Compute $C_3$ manually using the recurrence.
2. Verify $C_4 = 14$.
3. Print all valid parentheses for $n=3$ (should be 5).
4. Compare DP and binomial implementations.
5. Use Catalan DP to count binary search trees of size $n$.

#### Test Cases

| $n$ | Expected $C_n$ |
| --- | -------------- |
| 0   | 1              |
| 1   | 1              |
| 2   | 2              |
| 3   | 5              |
| 4   | 14             |

#### Complexity

- Time: $O(n^2)$ (DP), $O(n)$ (binomial)
- Space: $O(n)$

Catalan numbers are the backbone of many recursive counting problems, capturing the essence of balance, structure, and symmetry.

### 530 Stirling Numbers

Stirling numbers count ways to partition or arrange elements under specific constraints.
They connect combinatorics, recurrence relations, and generating functions, bridging counting, algebra, and probability.

There are two families:

1. Stirling numbers of the first kind $c(n, k)$, count permutations of $n$ elements with exactly $k$ cycles.
2. Stirling numbers of the second kind $S(n, k)$, count ways to partition $n$ elements into $k$ non-empty unlabeled subsets.

#### What Problem Are We Solving?

We want to compute either:

1. First kind (signed or unsigned):

$$
c(n,k) = c(n-1,k-1) + (n-1),c(n-1,k)
$$

2. Second kind:

$$
S(n,k) = S(n-1,k-1) + k,S(n-1,k)
$$

Base cases:

$$
c(0,0)=1,\quad S(0,0)=1
$$

$$
c(n,0)=S(n,0)=0 \text{ for } n>0
$$

$$
c(0,k)=S(0,k)=0 \text{ for } k>0
$$

#### How Does It Work (Plain Language)

- First kind: building permutations

  * Place element $n$ alone as a new cycle ($c(n-1,k-1)$ ways)
  * Or insert $n$ into one of $(n-1)$ existing cycles ($ (n-1),c(n-1,k)$ ways)

- Second kind: building partitions

  * Place element $n$ alone (new subset) → $S(n-1,k-1)$
  * Or place $n$ into one of $k$ existing subsets → $k,S(n-1,k)$

Each recurrence reflects a "new item added" logic, creating new groups or joining old ones.

#### Example

For second kind $S(3,k)$:

| $n$ | $k$ | $S(n,k)$ |
| --- | --- | -------- |
| 3   | 1   | 1        |
| 3   | 2   | 3        |
| 3   | 3   | 1        |

So $S(3,2)=3$ means 3 ways to split {1,2,3} into 2 non-empty sets:

- {1,2},{3}
- {1,3},{2}
- {2,3},{1}

#### Tiny Code (Easy Versions)

Python (Stirling numbers of the second kind)

```python
def stirling2(n, k):
    if n == k == 0:
        return 1
    if n == 0 or k == 0:
        return 0
    dp = [[0]*(k+1) for _ in range(n+1)]
    dp[0][0] = 1
    for i in range(1, n+1):
        for j in range(1, k+1):
            dp[i][j] = dp[i-1][j-1] + j * dp[i-1][j]
    return dp[n][k]

for k in range(1,4):
    print(f"S(3,{k}) =", stirling2(3,k))
```

Python (Stirling numbers of the first kind)

```python
def stirling1(n, k):
    if n == k == 0:
        return 1
    if n == 0 or k == 0:
        return 0
    dp = [[0]*(k+1) for _ in range(n+1)]
    dp[0][0] = 1
    for i in range(1, n+1):
        for j in range(1, k+1):
            dp[i][j] = dp[i-1][j-1] + (i-1) * dp[i-1][j]
    return dp[n][k]
```

#### Why It Matters

Stirling numbers unify combinatorics, algebra, and analysis:

- $S(n,k)$: count of set partitions
- $c(n,k)$: count of permutation cycles
- Appear in:

  * Bell numbers: $B_n = \sum_{k=0}^n S(n,k)$
  * Factorial expansions
  * Moments in probability and statistics
  * Polynomial bases (falling/rising factorials)

They're the coefficients when expressing factorials or powers in different bases:
$$
x^n = \sum_k S(n,k),(x)_k, \quad (x)_n = \sum_k c(n,k),x^k
$$

#### A Gentle Proof (Why It Works)

For $S(n,k)$, consider adding one element to a partition of size $n-1$:

- Create new subset (choose none → $S(n-1,k-1)$)
- Join existing subset (choose one of $k$ → $kS(n-1,k)$)

Add them → recurrence proven.

For $c(n,k)$, each new element:

- Starts a new cycle ($c(n-1,k-1)$)
- Joins one of existing cycles ($(n-1)c(n-1,k)$)

#### Try It Yourself

1. Compute $S(4,2)$ manually.
2. Verify $S(4,2)=7$ and $S(4,3)=6$.
3. Print full Stirling triangle.
4. Compare $S(n,k)$ vs binomial coefficients.
5. Derive Bell numbers: $B_n=\sum_k S(n,k)$.

#### Test Cases

| $n$ | $k$ | $S(n,k)$ | $c(n,k)$ |
| --- | --- | -------- | -------- |
| 3   | 1   | 1        | 2        |
| 3   | 2   | 3        | 3        |
| 3   | 3   | 1        | 1        |
| 4   | 2   | 7        | 11       |

#### Complexity

- Time: $O(nk)$
- Space: $O(nk)$

Stirling numbers form the grammar of combinatorial counting, expressing partitions, cycles, and transformations between polynomial worlds.

# Section 54. Probability and Randomized Algorithms 

### 531 Monte Carlo Simulation

Monte Carlo simulation is a numerical method that uses random sampling to approximate solutions to deterministic or probabilistic problems.
When exact formulas are hard or impossible to compute, Monte Carlo methods estimate results by simulating many random experiments and averaging the outcomes.

#### What Problem Are We Solving?

We want to estimate a quantity (like an area, integral, or probability) by random sampling.

The core principle:

$$
\text{Expected Value} \approx \frac{1}{N}\sum_{i=1}^{N} f(X_i)
$$

where $X_i$ are random samples and $f(X_i)$ measures the contribution from each trial.

As $N \to \infty$, the average converges to the true value by the Law of Large Numbers.

#### Example: Estimating π

Imagine a unit square enclosing a quarter-circle of radius 1.
The ratio of points falling inside the circle to total points approximates $\pi/4$.

$$
\frac{\text{points inside}}{\text{total points}} \approx \frac{\pi}{4}
$$

So:

$$
\pi \approx 4 \times \frac{\text{inside}}{\text{total}}
$$

#### How Does It Work (Plain Language)

1. Define a random experiment (e.g., sample $(x,y)$ in $[0,1]\times[0,1]$)
2. Check if the sample satisfies a condition (e.g., $x^2 + y^2 \le 1$)
3. Repeat many times
4. Use the ratio (hits / total) to estimate the desired probability or value

More samples → lower error (variance $\propto 1/\sqrt{N}$)

#### Tiny Code (Easy Versions)

Python Version (Estimate π)

```python
import random

def monte_carlo_pi(samples=1000000):
    inside = 0
    for _ in range(samples):
        x, y = random.random(), random.random()
        if x*x + y*y <= 1:
            inside += 1
    return 4 * inside / samples

print("Estimated π =", monte_carlo_pi())
```

C Version

```c
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    int samples = 1000000, inside = 0;
    for (int i = 0; i < samples; i++) {
        double x = (double)rand() / RAND_MAX;
        double y = (double)rand() / RAND_MAX;
        if (x*x + y*y <= 1.0) inside++;
    }
    double pi = 4.0 * inside / samples;
    printf("Estimated π = %f\n", pi);
    return 0;
}
```

#### Why It Matters

Monte Carlo methods are essential for:

- Integration in high dimensions
- Probabilistic modeling
- Optimization and simulation
- Financial modeling (e.g., option pricing)
- Physics and engineering (e.g., particle transport, statistical mechanics)

They trade accuracy for generality, thriving when deterministic methods fail.

#### A Gentle Proof (Why It Works)

Let $X_1, X_2, \dots, X_N$ be i.i.d. random variables representing outcomes.
By the Law of Large Numbers:

$$
\frac{1}{N}\sum_{i=1}^{N} X_i \to \mathbb{E}[X]
$$

So the sample mean converges to the expected value, giving a consistent estimator.
The Central Limit Theorem ensures that error decreases as $1/\sqrt{N}$.

#### Try It Yourself

1. Estimate $\pi$ with different sample sizes.
2. Use Monte Carlo to compute $\int_0^1 x^2,dx$.
3. Estimate the probability that sum of two dice is ≥ 10.
4. Model coin flips and compare to exact probabilities.
5. Measure convergence as $N$ grows (error vs samples).

#### Test Cases

| Samples   | Estimated π | Expected Error |
| --------- | ----------- | -------------- |
| 1000      | ~3.1        | ±0.05          |
| 10,000    | ~3.14       | ±0.02          |
| 1,000,000 | ~3.1415     | ±0.001         |

#### Complexity

- Time: $O(N)$
- Space: $O(1)$

Monte Carlo simulation is statistics as computation, randomness revealing structure through repetition and averages.

### 532 Las Vegas Algorithm

A Las Vegas algorithm always returns a correct answer, but its runtime is random, it may take longer or shorter depending on luck.
Unlike Monte Carlo methods (which trade accuracy for speed), Las Vegas algorithms never compromise correctness, only time.

#### What Problem Are We Solving?

We need randomized algorithms that remain guaranteed correct, but whose execution time depends on random events.

In other words:

- Output: always correct
- Time: random variable

We want to design algorithms where randomness helps avoid worst-case behavior or simplify logic, without breaking correctness.

#### How Does It Work (Plain Language)

A Las Vegas algorithm uses randomness to guide the search, pick pivots, or sample data, but verifies correctness before returning.

If the random choice is poor, it tries again.

Example: Randomized QuickSort

QuickSort chooses a random pivot, splits data around it, and recursively sorts.
Random pivot selection ensures expected $O(n \log n)$ time, even though worst-case $O(n^2)$ still exists.

But the output is always sorted, correctness never depends on chance.

Another example: Randomized QuickSelect

Finds the $k$-th smallest element using a random pivot.
If pivot is bad, runtime worsens, but result is still correct.

#### Tiny Code (Easy Versions)

Python (Randomized QuickSort)

```python
import random

def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = random.choice(arr)
    left = [x for x in arr if x < pivot]
    mid = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + mid + quicksort(right)

print(quicksort([3, 1, 4, 1, 5, 9, 2, 6]))
```

Python (QuickSelect)

```python
import random

def quickselect(arr, k):
    pivot = random.choice(arr)
    left = [x for x in arr if x < pivot]
    mid = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    if k < len(left):
        return quickselect(left, k)
    elif k < len(left) + len(mid):
        return pivot
    else:
        return quickselect(right, k - len(left) - len(mid))

print(quickselect([7, 10, 4, 3, 20, 15], 2))  # 3rd smallest element
```

#### Why It Matters

Las Vegas algorithms combine certainty of correctness with expected efficiency:

- Sorting and selection (QuickSort, QuickSelect)
- Computational geometry (Randomized incremental algorithms)
- Graph algorithms (Randomized MSTs, planar separations)
- Data structures (Skip lists, hash tables)

They often simplify design and avoid adversarial inputs.

#### A Gentle Proof (Why It Works)

Let $T$ be runtime random variable.
Expected time:

$$
\mathbb{E}[T] = \sum_{t} t \cdot P(T = t)
$$

The algorithm terminates when a "good" random event occurs (e.g. balanced pivot).
By bounding expected cost at each step, we can show $\mathbb{E}[T] = O(f(n))$.

Correctness is ensured by deterministic verification after random choices.

#### Try It Yourself

1. Compare QuickSort with and without random pivot.
2. Measure runtime across many runs, observe variation.
3. Count recursive calls distribution for small arrays.
4. Use randomization in skip list insertion.
5. Write a retry-based algorithm (e.g., random hash probing).

#### Test Cases

| Input                 | Algorithm   | Output        | Correct? | Runtime |
| --------------------- | ----------- | ------------- | -------- | ------- |
| [3,1,4,1,5,9]         | QuickSort   | [1,1,3,4,5,9] | Yes      | varies  |
| [7,10,4,3,20,15], k=2 | QuickSelect | 7             | Yes      | varies  |

#### Complexity

- Expected Time: $O(f(n))$ (often $O(n)$ or $O(n\log n)$)
- Worst Time: still possible, but rare
- Space: depends on recursion or retries

Las Vegas algorithms gamble with time, never with truth, they always find the right answer, only the path differs each run.

### 533 Reservoir Sampling

Reservoir sampling is a clever algorithm for selecting a uniform random sample from a data stream of unknown or very large size, using only O(k) space.
It's a cornerstone of streaming algorithms, where you cannot store all the data but still want fair random selection.

#### What Problem Are We Solving?

Given a stream of $n$ items (possibly very large or unbounded), select $k$ items uniformly at random, that is, each item has equal probability $\frac{k}{n}$ of being chosen, without knowing $n$ in advance.

We want to process items one by one, keeping only a reservoir of $k$ samples.

#### How Does It Work (Plain Language)

The key idea is incremental fairness:

1. Initialize: fill the reservoir with the first $k$ elements.
2. Process stream: for each item $i$ (1-indexed) after the $k$-th,

   * generate a random integer $j \in [1, i]$
   * if $j \le k$, replace reservoir[$j$] with item $i$

This ensures each element's chance = $\frac{k}{i}$ at the $i$-th step,
and finally $\frac{k}{n}$ overall.

#### Example (k = 2)

Stream: [A, B, C, D]

1. Reservoir = [A, B]
2. $i=3$ → random $j\in[1,3]$ → say $j=2$ → replace B → [A, C]
3. $i=4$ → $j\in[1,4]$ → say $j=4$ → skip (since >2) → [A, C]

After full stream, every pair has equal probability.

#### Proof of Uniformity

At step $i$:

- Probability of being selected: $\frac{k}{i}$
- Probability of surviving future replacements:
  $$
  \prod_{t=i+1}^{n}\left(1 - \frac{1}{t}\right)=\frac{i}{n}
  $$
  Thus final probability = $\frac{k}{i} \cdot \frac{i}{n} = \frac{k}{n}$, equal for all.

#### Tiny Code (Easy Versions)

Python Version (k = 1)

```python
import random

def reservoir_sample(stream):
    result = None
    for i, item in enumerate(stream, start=1):
        if random.randint(1, i) == 1:
            result = item
    return result

data = [10, 20, 30, 40, 50]
print("Random pick:", reservoir_sample(data))
```

Python Version (k = 3)

```python
import random

def reservoir_sample_k(stream, k):
    reservoir = []
    for i, item in enumerate(stream, start=1):
        if i <= k:
            reservoir.append(item)
        else:
            j = random.randint(1, i)
            if j <= k:
                reservoir[j - 1] = item
    return reservoir

data = range(1, 11)
print(reservoir_sample_k(data, 3))
```

C Version (k = 1)

```c
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    int stream[] = {10, 20, 30, 40, 50};
    int n = 5;
    int result = stream[0];
    for (int i = 1; i < n; i++) {
        int j = rand() % (i + 1);
        if (j == 0)
            result = stream[i];
    }
    printf("Random pick: %d\n", result);
}
```

#### Why It Matters

Reservoir sampling is crucial for:

- Big data: when $n$ is too large for memory
- Streaming APIs, logs, telemetry
- Random sampling in databases and distributed systems
- Machine learning: random mini-batches, unbiased selection

It gives exact uniform probability without prior knowledge of $n$.

#### A Gentle Proof (Why It Works)

For any element $x_i$:

- Chance selected when seen = $\frac{k}{i}$
- Chance not replaced later = $\frac{i}{i+1}\cdot \frac{i+1}{i+2}\cdots\frac{n-1}{n}=\frac{i}{n}$
- Combined: $\frac{k}{i} \cdot \frac{i}{n} = \frac{k}{n}$
  Uniform for all $i$.

#### Try It Yourself

1. Run with increasing $n$ and track frequencies.
2. Verify approximate uniformity over 10,000 trials.
3. Test with $k>1$.
4. Apply to data streams from a file or API.
5. Modify for weighted sampling.

#### Test Cases

| Stream       | k | Possible Reservoir | Probability |
| ------------ | - | ------------------ | ----------- |
| [A, B, C]    | 1 | A, B, or C         | 1/3 each    |
| [1, 2, 3, 4] | 2 | Any 2-combo        | Equal       |

#### Complexity

- Time: $O(n)$ (one pass)
- Space: $O(k)$

Reservoir sampling is randomness under constraint, fair selection from endless flow, one element at a time.

### 534 Randomized QuickSort

Randomized QuickSort is the classic divide-and-conquer sorting algorithm, with a twist of randomness.
Instead of choosing a fixed pivot (like the first or last element), it picks a random pivot, ensuring that the expected runtime stays $O(n \log n)$, regardless of input order.

This simple randomization elegantly neutralizes worst-case scenarios.

#### What Problem Are We Solving?

We need a fast, in-place sorting algorithm that avoids pathological inputs.
Standard QuickSort can degrade to $O(n^2)$ on sorted data if the pivot is chosen poorly.
By picking pivots uniformly at random, we guarantee expected balance and expected $O(n \log n)$ behavior.

#### How Does It Work (Plain Language)

QuickSort partitions the array around a pivot —
elements smaller go left, greater go right —
then recursively sorts the two sides.

Randomization ensures the pivot is, on average, near the median, keeping the recursion tree balanced.

Steps:

1. Choose a random pivot index `p`.
2. Partition array so that:

   * Left: elements < pivot
   * Right: elements > pivot
3. Recursively sort both partitions.

When randomization is fair, each pivot splits roughly evenly, giving height $\approx \log n$ and total work $O(n \log n)$.

#### Example

Sort `[3, 6, 2, 1, 4]`

1. Pick random pivot → say `3`
2. Partition → `[2,1] [3] [6,4]`
3. Recurse on `[2,1]` and `[6,4]`
4. Continue until sorted: `[1,2,3,4,6]`

Different random seeds → different recursion paths, same final result.

#### Tiny Code (Easy Versions)

Python Version

```python
import random

def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = random.choice(arr)
    left = [x for x in arr if x < pivot]
    mid = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + mid + quicksort(right)

arr = [3, 6, 2, 1, 4]
print(quicksort(arr))
```

C Version

```c
#include <stdio.h>
#include <stdlib.h>

void swap(int *a, int *b) {
    int t = *a; *a = *b; *b = t;
}

int partition(int a[], int low, int high) {
    int pivot_idx = low + rand() % (high - low + 1);
    swap(&a[pivot_idx], &a[high]);
    int pivot = a[high];
    int i = low - 1;
    for (int j = low; j < high; j++) {
        if (a[j] < pivot) swap(&a[++i], &a[j]);
    }
    swap(&a[i + 1], &a[high]);
    return i + 1;
}

void quicksort(int a[], int low, int high) {
    if (low < high) {
        int pi = partition(a, low, high);
        quicksort(a, low, pi - 1);
        quicksort(a, pi + 1, high);
    }
}

int main(void) {
    int a[] = {3, 6, 2, 1, 4};
    int n = 5;
    quicksort(a, 0, n - 1);
    for (int i = 0; i < n; i++) printf("%d ", a[i]);
}
```

#### Why It Matters

Randomized QuickSort is:

- Fast in practice (low constants, cache-friendly)
- In-place (no extra arrays)
- Expected $O(n \log n)$ independent of input
- Immune to adversarial inputs and pre-sorted traps

Used widely in:

- Standard libraries (e.g. Python's `sort()` uses hybrid with randomized pivot)
- Database systems and data processing pipelines
- Teaching divide-and-conquer and randomized algorithms

#### A Gentle Proof (Why It Works)

Let $T(n)$ be expected cost.
Each partition step costs $O(n)$ comparisons.
Pivot splits array into sizes $i$ and $n-i-1$ with equal probability for each $i$.

$$
\mathbb{E}[T(n)] = \frac{1}{n}\sum_{i=0}^{n-1} (\mathbb{E}[T(i)] + \mathbb{E}[T(n-i-1)]) + O(n)
$$

Solving gives $\mathbb{E}[T(n)] = O(n \log n)$.

Variance decreases as random choices average out.

#### Try It Yourself

1. Run algorithm multiple times, note pivot sequence.
2. Sort already sorted list, no slowdown expected.
3. Compare runtime vs MergeSort.
4. Visualize recursion tree depth.
5. Implement a 3-way partition version for duplicates.

#### Test Cases

| Input       | Output      | Notes      |
| ----------- | ----------- | ---------- |
| [3,6,2,1,4] | [1,2,3,4,6] | Basic case |
| [1,2,3,4,5] | [1,2,3,4,5] | Pre-sorted |
| [5,4,3,2,1] | [1,2,3,4,5] | Reverse    |
| [2,2,2,2]   | [2,2,2,2]   | Duplicates |

#### Complexity

- Expected Time: $O(n \log n)$
- Worst Case: $O(n^2)$ (rare)
- Space: $O(\log n)$ recursion stack

Randomized QuickSort transforms luck into balance, each shuffle a safeguard, each pivot a chance for harmony.

### 535 Randomized QuickSelect

Randomized QuickSelect is a divide-and-conquer algorithm for finding the k-th smallest element in an unsorted array in expected linear time.
It's the selection twin of QuickSort, using the same partition idea, but exploring only one side of the array at each step.

#### What Problem Are We Solving?

Given an unsorted array `arr` and a rank `k` (1-indexed),
we want the element that would appear at position `k` if the array were sorted —
without fully sorting it.

Example:
In `[7, 10, 4, 3, 20, 15]`,
the 3rd smallest element is `7`.

We want to find it in expected O(n), faster than sorting ($O(n \log n)$).

#### How Does It Work (Plain Language)

1. Choose a random pivot from the array.
2. Partition elements into three groups:

   * Left: smaller than pivot
   * Middle: equal to pivot
   * Right: greater than pivot
3. Compare `k` to the sizes:

   * If `k` ≤ len(left): recurse on left
   * Else if `k` ≤ len(left) + len(mid): pivot is the answer
   * Else: recurse on right with adjusted rank

By following only the side containing the k-th element,
we cut the problem roughly in half each time.

#### Example

Find the 3rd smallest in `[7, 10, 4, 3, 20, 15]`

1. Random pivot = `10`

   * Left = `[7, 4, 3]`, Mid = `[10]`, Right = `[20, 15]`
2. len(left) = 3
   Since `k=3` ≤ 3, recurse into `[7, 4, 3]`
3. Random pivot = `4`

   * Left = `[3]`, Mid = `[4]`, Right = `[7]`
4. len(left)=1, len(mid)=1
   `k=3` > 1+1 → recurse into `[7]` with `k=1`
   → Answer = `7`

#### Tiny Code (Easy Versions)

Python Version

```python
import random

def quickselect(arr, k):
    pivot = random.choice(arr)
    left = [x for x in arr if x < pivot]
    mid = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    
    if k <= len(left):
        return quickselect(left, k)
    elif k <= len(left) + len(mid):
        return pivot
    else:
        return quickselect(right, k - len(left) - len(mid))

data = [7, 10, 4, 3, 20, 15]
print(quickselect(data, 3))  # 3rd smallest → 7
```

C Version (In-Place Partition)

```c
#include <stdio.h>
#include <stdlib.h>

void swap(int *a, int *b) { int t = *a; *a = *b; *b = t; }

int partition(int a[], int l, int r, int pivot_idx) {
    int pivot = a[pivot_idx];
    swap(&a[pivot_idx], &a[r]);
    int store = l;
    for (int i = l; i < r; i++)
        if (a[i] < pivot) swap(&a[i], &a[store++]);
    swap(&a[store], &a[r]);
    return store;
}

int quickselect(int a[], int l, int r, int k) {
    if (l == r) return a[l];
    int pivot_idx = l + rand() % (r - l + 1);
    int idx = partition(a, l, r, pivot_idx);
    int rank = idx - l + 1;
    if (k == rank) return a[idx];
    if (k < rank) return quickselect(a, l, idx - 1, k);
    return quickselect(a, idx + 1, r, k - rank);
}

int main(void) {
    int arr[] = {7, 10, 4, 3, 20, 15};
    int n = 6, k = 3;
    printf("%d\n", quickselect(arr, 0, n - 1, k));
}
```

#### Why It Matters

- Expected O(n) time, O(1) space
- Foundation for median selection and order statistics
- Used in:

  * Median-of-medians
  * Randomized algorithms
  * QuickSort optimization
  * Data summarization and sampling

In practical terms, QuickSelect is how many libraries (like `numpy.partition`) find medians and percentiles efficiently.

#### A Gentle Proof (Why It Works)

Each step partitions the array in $O(n)$ and recurses on one side.
Expected split ratio ≈ 1:2 → recurrence:

$$
T(n) = T\left(\frac{n}{2}\right) + O(n)
$$

Solving gives $T(n) = O(n)$.
Worst case ($O(n^2)$) happens only if pivot is repeatedly extreme, probability exponentially small.

#### Try It Yourself

1. Find the median (`k = n//2`) of a large random array.
2. Compare QuickSelect vs sorting time.
3. Run multiple trials; note variation in recursion depth.
4. Modify to find k-th largest.
5. Implement deterministic pivot (median-of-medians).

#### Test Cases

| Input            | k | Output | Note           |
| ---------------- | - | ------ | -------------- |
| [7,10,4,3,20,15] | 3 | 7      | 3rd smallest   |
| [5,4,3,2,1]      | 1 | 1      | smallest       |
| [2,2,2,2]        | 2 | 2      | duplicates     |
| [10]             | 1 | 10     | single element |

#### Complexity

- Expected Time: $O(n)$
- Worst Case: $O(n^2)$ (rare)
- Space: $O(1)$ (in-place)

Randomized QuickSelect is precision through chance, a direct, elegant route to the k-th element, powered by probability.

### 536 Birthday Paradox Simulation

The Birthday Paradox is a famous probability puzzle showing how quickly collisions occur in random samples.
Surprisingly, with just 23 people, there's a >50% chance two share the same birthday, even though there are 365 possible days.

A simulation helps reveal why intuition often fails.

#### What Problem Are We Solving?

We want to estimate the probability that at least two people share the same birthday in a group of size $n$.

There are two approaches:

1. Analytical formula (exact)
2. Monte Carlo simulation (empirical)

#### Analytical Probability

Let's compute $P(\text{no collision})$ first:

- 1st person: 365 choices
- 2nd person: 364
- 3rd person: 363
- ...
- $n$-th person: $(365 - n + 1)$ choices

So:

$$
P(\text{no match}) = \frac{365}{365} \times \frac{364}{365} \times \cdots \times \frac{365-n+1}{365}
$$

Then:

$$
P(\text{collision}) = 1 - P(\text{no match})
$$

For $n=23$, $P(\text{collision}) \approx 0.507$

#### How Does It Work (Plain Language)

Each new person you add increases the chance of a match, not with everyone, but with any of the previous people.
The number of pairs grows fast:

$$
\text{\# pairs} = \binom{n}{2}
$$

That quadratic growth in comparisons explains why collisions happen quickly.

A Monte Carlo simulation simply runs the experiment many times and counts how often duplicates appear.

#### Example

Try $n = 23$ people, 10,000 trials:

- ~50% of trials have at least one shared birthday
- ~50% do not

The randomness converges to the analytical result.

#### Tiny Code (Easy Versions)

Python Version

```python
import random

def birthday_collision_prob(n, trials=10000):
    count = 0
    for _ in range(trials):
        birthdays = [random.randint(1, 365) for _ in range(n)]
        if len(birthdays) != len(set(birthdays)):
            count += 1
    return count / trials

for n in [10, 20, 23, 30, 40]:
    print(f"n={n}, P(collision)≈{birthday_collision_prob(n):.3f}")
```

C Version

```c
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>

bool has_collision(int n) {
    bool seen[366] = {false};
    for (int i = 0; i < n; i++) {
        int b = rand() % 365 + 1;
        if (seen[b]) return true;
        seen[b] = true;
    }
    return false;
}

double birthday_collision_prob(int n, int trials) {
    int count = 0;
    for (int i = 0; i < trials; i++)
        if (has_collision(n)) count++;
    return (double)count / trials;
}

int main(void) {
    srand(time(NULL));
    int ns[] = {10, 20, 23, 30, 40};
    for (int i = 0; i < 5; i++) {
        int n = ns[i];
        printf("n=%d, P≈%.3f\n", n, birthday_collision_prob(n, 10000));
    }
}
```

#### Why It Matters

The Birthday Paradox illustrates collision probability, crucial for:

- Hash functions (collision analysis)
- Cryptography (birthday attacks)
- Random ID generation (UUIDs, fingerprints)
- Simulation and probability reasoning

It shows that intuition about randomness often underestimates collisions.

#### A Gentle Proof (Why It Works)

For small $n$:

$$
P(\text{no match}) = \prod_{i=0}^{n-1} \frac{365 - i}{365}
$$

Expand for $n=23$:

$$
P(\text{no match}) \approx 0.493 \
P(\text{match}) = 1 - 0.493 = 0.507
$$

Thus, with only 23 people, more likely than not that two share a birthday.

#### Try It Yourself

1. Plot $P(\text{collision})$ vs $n$.
2. Find smallest $n$ where probability > 0.9.
3. Try different "year lengths" (e.g. 500 or 1000).
4. Test for non-uniform birthdays.
5. Simulate for hash buckets ($m=2^{16}$, $n=500$).

#### Test Cases

| n  | Expected P(collision) | Simulation (approx) |
| -- | --------------------- | ------------------- |
| 10 | 0.117                 | 0.12                |
| 20 | 0.411                 | 0.41                |
| 23 | 0.507                 | 0.50                |
| 30 | 0.706                 | 0.71                |
| 40 | 0.891                 | 0.89                |

#### Complexity

- Time: $O(n \cdot \text{trials})$
- Space: $O(n)$

The Birthday Paradox is a collision lens, a window into how randomness piles up faster than intuition expects.

### 537 Random Hashing

Random hashing uses randomness to minimize collisions and distribute keys uniformly across buckets.
It's a core idea in hash tables, Bloom filters, and probabilistic data structures, where fairness and independence matter more than determinism.

#### What Problem Are We Solving?

We need to map arbitrary keys to buckets (like slots in a hash table) in a way that spreads them evenly,
even when input keys follow non-uniform or adversarial patterns.

Deterministic hash functions may cluster keys or leak structure.
Adding randomness to the hash function ensures each key behaves like a random variable,
so collisions become rare and predictable only in expectation.

#### How Does It Work (Plain Language)

A random hash function is drawn from a *family* of functions $H = {h_1, h_2, \ldots, h_m}$
before use, we fix one randomly at runtime.

Formally, a hash family is universal if for all $x \ne y$:

$$
P(h(x) = h(y)) \le \frac{1}{M}
$$

where $M$ is the number of buckets.
This guarantees expected $O(1)$ lookups.

Key idea: each new program run uses a different random hash seed, so attackers or pathological datasets can't force collisions.

#### Example (Universal Hashing)

Let $U = {0, 1, \dots, p-1}$ with prime $p$.
For parameters $a, b$ chosen uniformly from $1, \dots, p-1$, define:

$$
h_{a,b}(x) = ((a \cdot x + b) \bmod p) \bmod M
$$

This gives a universal hash family, low collision probability and simple computation.

#### Example in Action

Say $p = 17$, $M = 10$, $a = 5$, $b = 3$:

| x | h(x) = (5x+3) mod 17 mod 10 |
| - | --------------------------- |
| 1 | 8                           |
| 2 | 3                           |
| 3 | 8                           |
| 4 | 3                           |

Each key gets a pseudo-random bucket.
Collisions still possible, but not predictable, controlled by probability, not structure.

#### Tiny Code (Easy Versions)

Python Version

```python
import random

def random_hash(a, b, p, M, x):
    return ((a * x + b) % p) % M

def make_hash(p, M):
    a = random.randint(1, p - 1)
    b = random.randint(0, p - 1)
    return lambda x: ((a * x + b) % p) % M

# Example
p, M = 17, 10
h = make_hash(p, M)
data = [1, 2, 3, 4, 5]
print([h(x) for x in data])
```

C Version

```c
#include <stdio.h>
#include <stdlib.h>

int random_hash(int a, int b, int p, int M, int x) {
    return ((a * x + b) % p) % M;
}

int main(void) {
    int p = 17, M = 10;
    int a = rand() % (p - 1) + 1;
    int b = rand() % p;
    int data[] = {1, 2, 3, 4, 5};
    int n = 5;
    for (int i = 0; i < n; i++) {
        printf("x=%d -> h=%d\n", data[i], random_hash(a, b, p, M, data[i]));
    }
}
```

#### Why It Matters

- Fairness: keys are spread evenly, lowering collision risk.
- Security: prevents adversarial key selection (important in hash-flooding attacks).
- Performance: ensures expected $O(1)$ lookup even in worst input cases.
- Applications:

  * Hash tables
  * Cuckoo hashing
  * Bloom filters
  * Consistent hashing
  * Cryptographic schemes (non-cryptographic randomness)

Languages like Python, Java, and Go randomize hash seeds to defend against input-based attacks.

#### A Gentle Proof (Why It Works)

For a universal hash family $H$ over $U$ with $|H| = m$:

$$
P(h(x) = h(y)) = \frac{1}{M}, \quad x \ne y
$$

Thus, the expected number of collisions for $n$ keys is:

$$
E[\text{collisions}] = \binom{n}{2} \cdot \frac{1}{M}
$$

If $M \approx n$, expected collisions ≈ constant.

Hence, average lookup and insertion time = $O(1)$.

#### Try It Yourself

1. Compare random hash vs naive $(x \bmod M)$.
2. Plot bucket frequencies over random runs.
3. Test collision count for random seeds.
4. Implement 2-choice hashing ($h_1$, $h_2$ choose less loaded).
5. Build small universal hash table.

#### Test Cases

| Keys             | M  | Hash Type         | Expected Collisions |
| ---------------- | -- | ----------------- | ------------------- |
| 0–99             | 10 | naive $x \bmod M$ | clustered           |
| 0–99             | 10 | random $ax+b$     | balanced            |
| adversarial keys | 10 | random seed       | unpredictable       |

#### Complexity

- Time: $O(1)$ per hash
- Space: $O(1)$ for parameters

Random hashing is structured unpredictability, deterministic in code, but probabilistically fair in behavior.

### 538 Random Walk Simulation

A random walk is a path defined by a sequence of random steps.
It models countless natural and computational processes, from molecules drifting in liquid to stock prices fluctuating, and from diffusion in physics to exploration in AI.

By simulating random walks, we capture how randomness unfolds over time.

#### What Problem Are We Solving?

We want to study how a process evolves when each next step depends on random choice, not deterministic rules.

Random walks appear in:

- Physics: Brownian motion
- Finance: stock movement models
- Graph theory: Markov chains, PageRank
- Algorithms: randomized search and sampling

A simulation lets us see expected displacement, return probability, and spatial spread.

#### How Does It Work (Plain Language)

A random walk starts at an origin (e.g., $(0,0)$).
At each step, choose a direction randomly and move one unit.

Examples:

- 1D walk: step +1 or −1
- 2D walk: move north, south, east, or west
- 3D walk: move along x, y, or z axes randomly

Repeat for $n$ steps, track the position, and analyze results.

#### Example (1D)

Start at $x=0$.
For each step:

- With probability $1/2$: $x \gets x+1$
- With probability $1/2$: $x \gets x-1$

After $n$ steps, position is random variable $X_n$.
Expected value: $E[X_n] = 0$
Variance: $Var[X_n] = n$
Expected distance from origin ≈ $\sqrt{n}$

#### Example (2D)

Start at $(0,0)$
Each step: randomly move up, down, left, or right.
Plotting the path produces a meandering trajectory, a picture of diffusion.

#### Tiny Code (Easy Versions)

Python (1D Random Walk)

```python
import random

def random_walk_1d(steps):
    x = 0
    path = [x]
    for _ in range(steps):
        x += random.choice([-1, 1])
        path.append(x)
    return path

walk = random_walk_1d(100)
print("Final position:", walk[-1])
```

Python (2D Random Walk)

```python
import random

def random_walk_2d(steps):
    x, y = 0, 0
    path = [(x, y)]
    for _ in range(steps):
        dx, dy = random.choice([(1,0), (-1,0), (0,1), (0,-1)])
        x, y = x + dx, y + dy
        path.append((x, y))
    return path

walk = random_walk_2d(50)
print("Final position:", walk[-1])
```

C Version (1D)

```c
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(void) {
    srand(time(NULL));
    int steps = 100;
    int x = 0;
    for (int i = 0; i < steps; i++) {
        x += (rand() % 2) ? 1 : -1;
    }
    printf("Final position: %d\n", x);
}
```

#### Why It Matters

Random walks form the mathematical backbone of many systems:

- Physics: diffusion, Brownian motion
- Finance: random price fluctuations
- AI & RL: exploration strategies
- Graph algorithms: PageRank, cover time
- Statistics: Monte Carlo methods

They show how order emerges from randomness and why many natural processes diffuse over time.

#### A Gentle Proof (Why It Works)

Each step $S_i$ is independent with $E[S_i]=0$, $Var[S_i]=1$.
After $n$ steps:

$$
X_n = \sum_{i=1}^n S_i, \quad E[X_n]=0, \quad Var[X_n]=n
$$

Expected squared distance: $E[X_n^2] = n$
Expected absolute displacement: $\sqrt{n}$
This scaling explains diffusion's $\sqrt{t}$ behavior.

In 2D or 3D, similar logic extends:
$$
E[|X_n|^2] = n \cdot \text{step size}^2
$$

#### Try It Yourself

1. Plot the path for $n=1000$ in 1D and 2D.
2. Compare multiple runs, note variability.
3. Track average distance from origin over trials.
4. Change step probabilities (biased walk).
5. Add absorbing barriers (e.g. stop at $x=10$).

#### Test Cases

| Steps | Dimension | Expected $E[X_n]$ | Expected Distance |
| ----- | --------- | ----------------- | ----------------- |
| 10    | 1D        | 0                 | ~3.16             |
| 100   | 1D        | 0                 | ~10               |
| 100   | 2D        | (0,0)             | ~10               |
| 1000  | 1D        | 0                 | ~31.6             |

#### Complexity

- Time: $O(n)$
- Space: $O(n)$ if storing path; $O(1)$ if tracking position only

A random walk is motion without plan, step by step, direction by chance, pattern emerging only in the long run.

### 539 Coupon Collector Estimation

The Coupon Collector Problem asks:
How many random draws (with replacement) are needed to collect all distinct items from a set of size $n$?

It's a cornerstone of probabilistic analysis, used to model everything from collecting trading cards to testing coverage in randomized algorithms.

#### What Problem Are We Solving?

Suppose there are $n$ different coupons (or Pokémon, or test cases).
Each time, you draw one uniformly at random.

Question:
How many draws $T$ do you need on average to collect all $n$?

#### The Big Idea

Each new draw has a smaller chance of revealing something *new*.
At the beginning, new coupons are easy to find; toward the end, you're mostly drawing duplicates.

The expected number of trials is:

$$
E[T] = n \cdot H_n = n \left(1 + \frac{1}{2} + \frac{1}{3} + \cdots + \frac{1}{n}\right)
$$

Asymptotically:

$$
E[T] \approx n \ln n + \gamma n + \frac{1}{2}
$$

where $\gamma \approx 0.57721$ is the Euler–Mascheroni constant.

#### How Does It Work (Plain Language)

You can think of the collection process as phases:

- Phase 1: get the 1st new coupon → expected 1 draw
- Phase 2: get the 2nd new coupon → expected $\frac{n}{n-1}$ draws
- Phase 3: get the 3rd new coupon → expected $\frac{n}{n-2}$ draws
- …
- Phase $n$: last coupon → expected $\frac{n}{1}$ draws

Summing them up gives $E[T] = n H_n$.

#### Example

For $n = 5$:

$$
E[T] = 5 \cdot (1 + \frac{1}{2} + \frac{1}{3} + \frac{1}{4} + \frac{1}{5}) = 5 \times 2.283 = 11.415
$$

So you'll need about 11–12 draws on average to get all 5.

#### Tiny Code (Easy Versions)

Python (Simulation)

```python
import random

def coupon_collector(n, trials=10000):
    total = 0
    for _ in range(trials):
        collected = set()
        count = 0
        while len(collected) < n:
            coupon = random.randint(1, n)
            collected.add(coupon)
            count += 1
        total += count
    return total / trials

for n in [5, 10, 20]:
    print(f"n={n}, Expected≈{coupon_collector(n):.2f}, Theory≈{n * sum(1/i for i in range(1, n+1)):.2f}")
```

C (Simple Simulation)

```c
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>

double coupon_collector(int n, int trials) {
    int total = 0;
    bool seen[n+1];
    for (int t = 0; t < trials; t++) {
        for (int i = 1; i <= n; i++) seen[i] = false;
        int count = 0, collected = 0;
        while (collected < n) {
            int c = rand() % n + 1;
            count++;
            if (!seen[c]) { seen[c] = true; collected++; }
        }
        total += count;
    }
    return (double)total / trials;
}

int main(void) {
    srand(time(NULL));
    int ns[] = {5, 10, 20};
    for (int i = 0; i < 3; i++) {
        int n = ns[i];
        printf("n=%d, E≈%.2f\n", n, coupon_collector(n, 10000));
    }
}
```

#### Why It Matters

The Coupon Collector model shows up everywhere:

- Algorithm coverage (e.g., hashing all buckets)
- Testing & sampling (ensuring all cases appear)
- Networking (packet collection)
- Distributed systems (gossip protocols)
- Collectible games (expected cost to finish a set)

It captures the diminishing returns of randomness, the last few items always take longest.

#### A Gentle Proof (Why It Works)

Let $T_i$ = draws to get a new coupon when you already have $i-1$.

Probability of success = $\frac{n - i + 1}{n}$

So expected draws for phase $i$ = $\frac{1}{p_i} = \frac{n}{n - i + 1}$

Sum over all phases:

$$
E[T] = \sum_{i=1}^n \frac{n}{n - i + 1} = n \sum_{k=1}^n \frac{1}{k} = n H_n
$$

#### Try It Yourself

1. Simulate for different $n$ values; compare with theory.
2. Plot $E[T]/n$, should grow like $\ln n$.
3. Modify for biased probabilities (non-uniform draws).
4. Estimate probability all coupons collected by step $t$.
5. Apply to randomized load balancing (balls into bins).

#### Test Cases

| n  | Theoretical $E[T]$ | Simulation (approx) |
| -- | ------------------ | ------------------- |
| 5  | 11.42              | 11.40               |
| 10 | 29.29              | 29.20               |
| 20 | 71.94              | 72.10               |
| 50 | 224.96             | 225.10              |

#### Complexity

- Time: $O(n \cdot \text{trials})$
- Space: $O(n)$

The Coupon Collector problem is a mathematical metaphor for patience, the closer you are to completion, the rarer progress becomes.

### 540 Markov Chain Simulation

A Markov chain models a system that jumps between states with fixed transition probabilities.
The future depends only on the current state, not on the full past.
Simulation lets us estimate long-run behavior, hitting times, and steady-state distributions when analysis is hard.

#### What Problem Are We Solving?

Given a finite state space $\mathcal{S}={1,\dots,m}$ and a row-stochastic transition matrix $P\in\mathbb{R}^{m\times m}$ with
$$
P_{ij}=P(X_{t+1}=j \mid X_t=i),\quad \sum_{j=1}^m P_{ij}=1,
$$
we want to generate a trajectory $X_0,X_1,\dots,X_T$ and estimate quantities like

- stationary distribution $\pi$ such that $\pi P=\pi$
- expected reward $\mathbb{E}[f(X_t)]$
- hitting or return times

#### How Does It Work (Plain Language)

1. Choose an initial state $X_0$ (or an initial distribution $\mu$).
2. For $t=0,1,\dots,T-1$

   * From the current state $i=X_t$, draw the next state $j$ according to row $i$ of $P$.
   * Set $X_{t+1}=j$.
3. Optionally discard a burn-in prefix, then average statistics over the rest.

If the chain is irreducible and aperiodic, empirical frequencies converge to the stationary distribution.

#### Example

Two-state weather model, states ${S,R}$ for Sunny, Rainy:
$$
P=\begin{pmatrix}
0.8 & 0.2\
0.4 & 0.6
\end{pmatrix}.
$$
Start at Sunny, simulate 10,000 steps, estimate fraction of sunny days.
Theory: solve $\pi=\pi P$, $\pi_S=\tfrac{2}{3}$, $\pi_R=\tfrac{1}{3}$.

#### Tiny Code (Easy Versions)

Python Version

```python
import random

def simulate_markov(P, start, steps):
    # P: list of lists, each row sums to 1
    # start: integer state index
    x = start
    traj = [x]
    for _ in range(steps):
        r = random.random()
        cdf, nxt = 0.0, 0
        for j, p in enumerate(P[x]):
            cdf += p
            if r <= cdf:
                nxt = j
                break
        x = nxt
        traj.append(x)
    return traj

# Example: Sunny=0, Rainy=1
P = [[0.8, 0.2],
     [0.4, 0.6]]
traj = simulate_markov(P, start=0, steps=10000)
burn = 1000
pi_hat_S = sum(1 for s in traj[burn:] if s == 0) / (len(traj) - burn)
print("Estimated pi(Sunny) =", pi_hat_S)
```

C Version

```c
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int next_state(double *row, int m) {
    double r = (double)rand() / RAND_MAX, cdf = 0.0;
    for (int j = 0; j < m; j++) {
        cdf += row[j];
        if (r <= cdf) return j;
    }
    return m - 1;
}

int main(void) {
    srand((unsigned)time(NULL));
    int m = 2, steps = 10000, x = 0;
    double P[2][2] = {{0.8, 0.2}, {0.4, 0.6}};
    int sunny = 0, burn = 1000;
    for (int t = 0; t < steps; t++) {
        if (t >= burn && x == 0) sunny++;
        x = next_state(P[x], m);
    }
    double pi_hat = (double)sunny / (steps - burn);
    printf("Estimated pi(Sunny) = %.4f\n", pi_hat);
    return 0;
}
```

#### Why It Matters

- Statistics and MCMC: estimate integrals by sampling from complex distributions
- Operations research: queueing systems, reliability, inventory
- Reinforcement learning: modeling environment dynamics and returns
- Networks and graphs: PageRank, random walks, mixing time

Simulation provides estimates when closed forms are unavailable or too costly.

#### A Gentle Proof (Why It Works)

If the chain is irreducible and aperiodic with stationary distribution $\pi$, then by the ergodic theorem
$$
\frac{1}{T}\sum_{t=1}^T f(X_t)\xrightarrow[]{a.s.}\sum_{i\in\mathcal{S}} \pi_i f(i).
$$
In particular, empirical state frequencies converge to $\pi$. Burn-in reduces bias from the initial distribution.

#### Try It Yourself

1. Verify the two-state example, compare $\pi_{\text{Sunny}}$ estimate to $2/3$.
2. Build a 5-state chain with a known $\pi$ and confirm convergence.
3. Add a reward function $r(i)$, estimate long-run average reward.
4. Explore the effect of burn-in and sample size on variance.
5. Simulate a random walk on a graph, estimate visit frequencies.

#### Test Cases

| Chain                                    | Property                           | Theory       | Simulation (approx)     |
| ---------------------------------------- | ---------------------------------- | ------------ | ----------------------- |
| 2-state weather                          | $\pi_S$                            | $2/3$        | $0.66$ to $0.67$        |
| 3-state cyclic with damping              | ergodic                            | unique $\pi$ | frequencies match $\pi$ |
| Random walk on line with reflecting ends | stationary mass higher near middle | qualitative  | matches histogram       |

#### Complexity

- Time: $O(T \cdot \deg)$ where $\deg$ is average nonzero entries per row of $P$
- Space: $O(1)$ for online statistics, $O(T)$ if storing the full trajectory

Markov chain simulation turns matrix probabilities into empirical behavior, one step at a time, then averages reveal the long-run truth.

# Section 55. Sieve Methods and Modular Math 

### 541 Sieve of Eratosthenes

The Sieve of Eratosthenes is one of the oldest and most elegant algorithms in mathematics, designed to generate all prime numbers up to a given limit $n$.
It systematically marks out multiples of primes, leaving only the primes unmarked.

#### What Problem Are We Solving?

We want an efficient way to find all primes $\le n$.
A naive method tests divisibility for every number, $O(n\sqrt{n})$.
The sieve reduces this dramatically to $O(n \log\log n)$ by marking composites in bulk.

#### How Does It Work (Plain Language)

Imagine writing all numbers from $2$ to $n$.
Starting from the first unmarked number $2$, you:

1. Declare it prime.
2. Mark all multiples of $2$ as composite.
3. Move to the next unmarked number (which must be prime).
4. Repeat until $p^2 > n$.

The remaining unmarked numbers are primes.

#### Example

Let $n = 30$
Start with $2$:

| Step | Prime | Marked Multiples                                    |
| ---- | ----- | --------------------------------------------------- |
| 1    | 2     | 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30 |
| 2    | 3     | 6, 9, 12, 15, 18, 21, 24, 27, 30                    |
| 3    | 5     | 10, 15, 20, 25, 30                                  |
| 4    | 7     | $7^2 = 49 > 30$ → stop                              |

Primes left: 2, 3, 5, 7, 11, 13, 17, 19, 23, 29

#### Tiny Code (Easy Versions)

Python Version

```python
def sieve(n):
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    p = 2
    while p * p <= n:
        if is_prime[p]:
            for i in range(p * p, n + 1, p):
                is_prime[i] = False
        p += 1
    return [i for i in range(2, n + 1) if is_prime[i]]

print(sieve(30))
```

C Version

```c
#include <stdio.h>
#include <stdbool.h>
#include <math.h>

void sieve(int n) {
    bool is_prime[n+1];
    for (int i = 0; i <= n; i++) is_prime[i] = true;
    is_prime[0] = is_prime[1] = false;
    for (int p = 2; p * p <= n; p++) {
        if (is_prime[p]) {
            for (int i = p * p; i <= n; i += p)
                is_prime[i] = false;
        }
    }
    for (int i = 2; i <= n; i++)
        if (is_prime[i]) printf("%d ", i);
}

int main(void) {
    sieve(30);
}
```

#### Why It Matters

- Efficient prime generation, fundamental for number theory, cryptography, and factorization.
- Core building block for:

  * Prime tables in modular arithmetic
  * Totient computation
  * Sieve-based factorization
  * Precomputation in combinatorics (nCr mod p)
- Intuitive, a beautiful demonstration of elimination by pattern.

#### A Gentle Proof (Why It Works)

Every composite number $n$ has a smallest prime factor $p \le \sqrt{n}$.
When the sieve marks multiples of each prime up to $\sqrt{n}$, every composite is marked exactly once by its smallest prime factor.
Thus, all unmarked numbers are prime.

#### Try It Yourself

1. Run the sieve for $n=50$, count primes (should be 15).
2. Modify to return count instead of list.
3. Plot prime density vs $n$.
4. Extend to segmented sieve for large $n$.
5. Compare runtime vs naive primality testing.

#### Test Cases

| n  | Primes Found                       | Count |
| -- | ---------------------------------- | ----- |
| 10 | 2, 3, 5, 7                         | 4     |
| 20 | 2, 3, 5, 7, 11, 13, 17, 19         | 8     |
| 30 | 2, 3, 5, 7, 11, 13, 17, 19, 23, 29 | 10    |

#### Complexity

- Time: $O(n \log\log n)$
- Space: $O(n)$

The Sieve of Eratosthenes is simplicity sharpened by insight, mark the multiples, reveal the primes.

### 542 Linear Sieve

The Linear Sieve, also known as the Euler Sieve, is an optimized version of the Sieve of Eratosthenes that computes all primes up to $n$ in $O(n)$ time, marking each composite exactly once.

It avoids redundant markings by combining primes with their smallest prime factors.

#### What Problem Are We Solving?

The classical sieve marks each composite multiple times, once for every prime divisor.
In the linear sieve, each composite is generated only by its smallest prime factor (SPF), ensuring total work proportional to $n$.

We want:

- All primes $\le n$
- Optionally, the smallest prime factor `spf[x]` for each $x$

#### How Does It Work (Plain Language)

Maintain:

- A boolean array `is_prime[]`
- A list `primes[]`

Algorithm:

1. Initialize all numbers as prime (`True`).
2. For each integer $i$ from 2 to $n$:

   * If `is_prime[i]` is `True`, add $i$ to the list of primes.
   * For every prime $p$ in `primes`:

     * If $i \cdot p > n$, break.
     * Mark `is_prime[i*p] = False`.
     * If $p$ divides $i$, stop (to ensure each composite is marked only once).

This ensures each composite is marked exactly once by its smallest prime factor.

#### Example

Let $n=10$:

| i  | primes           | marked composites |
| -- | ---------------- | ----------------- |
| 2  | [2]              | 4, 6, 8, 10       |
| 3  | [2,3]            | 6, 9              |
| 4  | skip (not prime) |                   |
| 5  | [2,3,5]          | 10                |
| 6  | skip             |                   |
| 7  | [2,3,5,7]        |                   |
| 8  | skip             |                   |
| 9  | skip             |                   |
| 10 | skip             |                   |

Primes: 2, 3, 5, 7

#### Tiny Code (Easy Versions)

Python Version

```python
def linear_sieve(n):
    is_prime = [True] * (n + 1)
    primes = []
    spf = [0] * (n + 1)  # smallest prime factor
    is_prime[0] = is_prime[1] = False

    for i in range(2, n + 1):
        if is_prime[i]:
            primes.append(i)
            spf[i] = i
        for p in primes:
            if i * p > n:
                break
            is_prime[i * p] = False
            spf[i * p] = p
            if i % p == 0:
                break
    return primes, spf

pr, spf = linear_sieve(30)
print("Primes:", pr)
```

C Version

```c
#include <stdio.h>
#include <stdbool.h>

void linear_sieve(int n) {
    bool is_prime[n+1];
    int primes[n+1], spf[n+1];
    int count = 0;

    for (int i = 0; i <= n; i++) is_prime[i] = true;
    is_prime[0] = is_prime[1] = false;

    for (int i = 2; i <= n; i++) {
        if (is_prime[i]) {
            primes[count++] = i;
            spf[i] = i;
        }
        for (int j = 0; j < count; j++) {
            int p = primes[j];
            if (i * p > n) break;
            is_prime[i * p] = false;
            spf[i * p] = p;
            if (i % p == 0) break;
        }
    }

    printf("Primes: ");
    for (int i = 0; i < count; i++) printf("%d ", primes[i]);
}

int main(void) {
    linear_sieve(30);
}
```

#### Why It Matters

The linear sieve is a powerful improvement:

- Each number processed once
- Fastest possible prime sieve (tight asymptotic bound)
- Useful byproducts:

  * Smallest prime factor (SPF) table for factorization
  * Prime list for arithmetic functions (e.g. Euler's Totient)

Used widely in:

- Competitive programming
- Precomputation for modular arithmetic
- Factorization and divisor enumeration

#### A Gentle Proof (Why It Works)

Each composite $n = p \cdot m$ is marked exactly once, by its smallest prime factor $p$.
When $i = m$, the algorithm pairs $i$ with $p$:

- If $p$ divides $i$, break to prevent further markings.

Thus total operations ≈ $O(n)$.

#### Try It Yourself

1. Compare with classical sieve timings for $n=10^6$.
2. Print `spf[x]` for $x=2$ to $20$.
3. Write a function to factorize $x$ using `spf`.
4. Modify to count number of prime factors.
5. Extend to compute totient in $O(n)$.

#### Test Cases

| n   | Primes Found                       | Count |
| --- | ---------------------------------- | ----- |
| 10  | 2, 3, 5, 7                         | 4     |
| 30  | 2, 3, 5, 7, 11, 13, 17, 19, 23, 29 | 10    |
| 100 | 25 primes                          | 25    |

#### Complexity

- Time: $O(n)$
- Space: $O(n)$

The Linear Sieve refines Eratosthenes' brilliance, marking each composite once, no more, no less.

### 543 Segmented Sieve

The Segmented Sieve extends the classic sieve to handle large ranges efficiently, such as generating primes between $L$ and $R$ where $R$ may be very large (e.g. $10^{12}$), without storing all numbers up to $R$ in memory.

It divides the range into segments small enough to fit in memory, sieving each one using precomputed small primes.

#### What Problem Are We Solving?

The normal sieve of Eratosthenes needs memory proportional to $n$, making it infeasible for large upper bounds (like $10^{12}$).
The segmented sieve solves this by using a two-phase process:

1. Precompute small primes up to $\sqrt{R}$ using a standard sieve.
2. Sieve each segment $[L, R]$ using those primes.

This way, memory usage stays $O(\sqrt{R}) + O(R-L+1)$, even for very large $R$.

#### How Does It Work (Plain Language)

To generate primes in $[L, R]$:

1. Pre-sieve small primes:
   $$ \text{small\_primes} = \text{sieve}(\sqrt{R}) $$
2. Initialize a boolean segment array of size $R-L+1$ (all True).
3. For each small prime $p$:

   * Find the first multiple of $p$ in $[L, R]$:
     $$
     \text{start} = \max(p^2, \lceil \frac{L}{p} \rceil \cdot p)
     $$
   * Mark all multiples of $p$ as composite.
4. Remaining unmarked numbers are primes in $[L, R]$.

#### Example

Find primes in $[10, 30]$

1. Compute primes up to $\sqrt{30} = 5$: ${2, 3, 5}$
2. Segment = `[10..30]`, mark multiples:

   * $2$: mark 10,12,14,…,30
   * $3$: mark 12,15,18,21,24,27,30
   * $5$: mark 10,15,20,25,30
3. Unmarked → 11, 13, 17, 19, 23, 29

#### Tiny Code (Easy Versions)

Python Version

```python
import math

def simple_sieve(limit):
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for p in range(2, int(math.sqrt(limit)) + 1):
        if is_prime[p]:
            for i in range(p * p, limit + 1, p):
                is_prime[i] = False
    return [p for p in range(2, limit + 1) if is_prime[p]]

def segmented_sieve(L, R):
    limit = int(math.sqrt(R)) + 1
    primes = simple_sieve(limit)
    is_prime = [True] * (R - L + 1)

    for p in primes:
        start = max(p * p, ((L + p - 1) // p) * p)
        for i in range(start, R + 1, p):
            is_prime[i - L] = False

    if L == 1:
        is_prime[0] = False

    return [L + i for i, prime in enumerate(is_prime) if prime]

print(segmented_sieve(10, 30))
```

C Version

```c
#include <stdio.h>
#include <math.h>
#include <stdbool.h>

void simple_sieve(int limit, int primes[], int *count) {
    bool mark[limit+1];
    for (int i = 0; i <= limit; i++) mark[i] = true;
    mark[0] = mark[1] = false;
    for (int p = 2; p * p <= limit; p++)
        if (mark[p])
            for (int i = p * p; i <= limit; i += p)
                mark[i] = false;
    *count = 0;
    for (int i = 2; i <= limit; i++)
        if (mark[i]) primes[(*count)++] = i;
}

void segmented_sieve(long long L, long long R) {
    int primes[100000], count;
    int limit = (int)sqrt(R) + 1;
    simple_sieve(limit, primes, &count);
    int size = R - L + 1;
    bool is_prime[size];
    for (int i = 0; i < size; i++) is_prime[i] = true;

    for (int i = 0; i < count; i++) {
        long long p = primes[i];
        long long start = p * p;
        if (start < L) start = ((L + p - 1) / p) * p;
        for (long long j = start; j <= R; j += p)
            is_prime[j - L] = false;
    }

    if (L == 1) is_prime[0] = false;

    for (int i = 0; i < size; i++)
        if (is_prime[i]) printf("%lld ", L + i);
}

int main(void) {
    segmented_sieve(10, 30);
}
```

#### Why It Matters

The segmented sieve is essential when:

- $R$ is too large for a full array
- You need prime generation in ranges, e.g. [10⁹, 10⁹ + 10⁶]
- Building prime lists for big-number algorithms (RSA, primality tests)

It combines space efficiency with speed, leveraging precomputed small primes.

#### A Gentle Proof (Why It Works)

Every composite number in $[L,R]$ has a smallest prime factor $p \le \sqrt{R}$.
Thus, by marking all multiples of primes $\le \sqrt{R}$, we eliminate all composites.
Numbers left unmarked must be prime.

No need to store all numbers ≤ $R$, only the current segment.

#### Try It Yourself

1. Generate primes between $10^6$ and $10^6+1000$.
2. Measure memory vs full sieve.
3. Try non-square segment sizes.
4. Compare performance with classical sieve.
5. Combine with wheel factorization for extra speed.

#### Test Cases

| Range      | Output Primes       | Count |
| ---------- | ------------------- | ----- |
| [10, 30]   | 11,13,17,19,23,29   | 6     |
| [1, 10]    | 2,3,5,7             | 4     |
| [100, 120] | 101,103,107,109,113 | 5     |

#### Complexity

- Time: $O((R-L+1)\log\log R)$
- Space: $O(\sqrt{R}) + O(R-L+1)$

The Segmented Sieve scales Eratosthenes' idea to infinity, sieving range by range, memory never a barrier.

### 544 SPF (Smallest Prime Factor) Table

The Smallest Prime Factor (SPF) Table precomputes, for every integer $1 \le n \le N$, the smallest prime that divides it.
It's a powerful foundation for fast factorization, divisor functions, and multiplicative arithmetic functions, all in $O(N)$ time.

#### What Problem Are We Solving?

We often need to:

- Factorize many numbers quickly
- Compute arithmetic functions (e.g. $\varphi(n)$, $\tau(n)$, $\sigma(n)$)
- Check primality efficiently

Trial division is $O(\sqrt{n})$ per query, too slow for many queries.
With SPF precomputation, any number can be factorized in $O(\log n)$.

#### How Does It Work (Plain Language)

We build a table `spf[i]` such that:

- If $i$ is prime → `spf[i] = i`
- If $i$ is composite → `spf[i]` = smallest prime dividing $i$

We fill it using a linear sieve:

1. Start with `spf[i] = 0` (unset).
2. When visiting a prime $p$, set `spf[p] = p`.
3. For each $p$, mark $i \cdot p$ with `spf[i*p] = p` if unset.

Each composite is processed once, by its smallest prime.

#### Example

Compute `spf` for $1 \le i \le 10$:

| i  | spf[i] | Factorization |
| -- | ------ | ------------- |
| 1  | 1      |,             |
| 2  | 2      | 2             |
| 3  | 3      | 3             |
| 4  | 2      | 2×2           |
| 5  | 5      | 5             |
| 6  | 2      | 2×3           |
| 7  | 7      | 7             |
| 8  | 2      | 2×2×2         |
| 9  | 3      | 3×3           |
| 10 | 2      | 2×5           |

Factorization becomes trivial:

```
n = 84
spf[84] = 2 → 42
spf[42] = 2 → 21
spf[21] = 3 → 7
spf[7] = 7 → 1
→ 2 × 2 × 3 × 7
```

#### Tiny Code (Easy Versions)

Python Version

```python
def spf_sieve(n):
    spf = [0] * (n + 1)
    spf[1] = 1
    for i in range(2, n + 1):
        if spf[i] == 0:
            spf[i] = i
            for j in range(i * i, n + 1, i):
                if spf[j] == 0:
                    spf[j] = i
    return spf

def factorize(n, spf):
    factors = []
    while n != 1:
        factors.append(spf[n])
        n //= spf[n]
    return factors

spf = spf_sieve(100)
print(factorize(84, spf))
```

C Version

```c
#include <stdio.h>

void spf_sieve(int n, int spf[]) {
    for (int i = 0; i <= n; i++) spf[i] = 0;
    spf[1] = 1;
    for (int i = 2; i <= n; i++) {
        if (spf[i] == 0) {
            spf[i] = i;
            for (long long j = (long long)i * i; j <= n; j += i)
                if (spf[j] == 0) spf[j] = i;
        }
    }
}

void factorize(int n, int spf[]) {
    while (n != 1) {
        printf("%d ", spf[n]);
        n /= spf[n];
    }
}

int main(void) {
    int n = 100, spf[101];
    spf_sieve(n, spf);
    factorize(84, spf);
}
```

#### Why It Matters

The SPF table is a Swiss-army knife for number-theoretic algorithms:

- Prime factorization in $O(\log n)$
- Counting divisors: use exponents from SPF
- Computing totient: $\varphi(n)$ via prime factors
- Detecting square-free numbers

Used in:

- Modular arithmetic systems
- Factorization-heavy algorithms
- Competitive programming precomputations

#### A Gentle Proof (Why It Works)

Each composite $n = p \cdot m$ has smallest prime $p$.
When sieve reaches $p$, it sets `spf[n] = p`.
If a smaller prime existed, it would've marked $n$ earlier.
Thus `spf[n]` is indeed the smallest prime dividing $n$.

Each number is marked once → total time $O(n)$.

#### Try It Yourself

1. Build `spf[]` for $n=50$ and print all factorizations.
2. Write `is_prime(i)` = `(spf[i]==i)`.
3. Modify to count distinct prime factors.
4. Compute $\varphi(i)$ using SPF factors.
5. Visualize prime factor frequencies.

#### Test Cases

| n  | Factorization via SPF |
| -- | --------------------- |
| 10 | 2 × 5                 |
| 12 | 2 × 2 × 3             |
| 36 | 2 × 2 × 3 × 3         |
| 84 | 2 × 2 × 3 × 7         |

#### Complexity

- Precompute: $O(n)$
- Factorization per query: $O(\log n)$
- Space: $O(n)$

The SPF table turns factorization from division into lookup, precalculate once, reuse forever.

### 545 Möbius Function Sieve

The Möbius function $\mu(n)$ is a multiplicative arithmetic function central to inversion formulas and detecting square factors.
Values:

- $\mu(1)=1$
- $\mu(n)=0$ if $n$ has any squared prime factor
- $\mu(n)=(-1)^k$ if $n$ is a product of $k$ distinct primes

A Möbius sieve computes $\mu(1),\dots,\mu(N)$ efficiently for large $N$.

#### What Problem Are We Solving?

We want to compute $\mu(n)$ for all $1 \le n \le N$ faster than factoring each $n$ separately.

Target:

- All values of $\mu$ up to $N$ in near linear time
- Often together with a prime list and the smallest prime factor

#### How Does It Work (Plain Language)

Use a linear sieve with these invariants:

- Maintain a dynamic list of primes `primes`
- Store `mu[i]` as we go
- For each $i$ from $2$ to $N$:

  * If $i$ is prime, set `mu[i] = -1` and push to `primes`
  * For each prime $p$ in `primes`:

    * If $i p > N`, stop
    * If $p \mid i$, then:

      * `mu[i*p] = 0` because $p^2 \mid i p$
      * break to keep linear complexity
    * Else:

      * `mu[i*p] = -mu[i]` because we add a new distinct prime factor

Initialize with `mu[1] = 1`.

This marks each composite exactly once with its smallest prime factor relation.

#### Example

Compute $\mu(n)$ for $1 \le n \le 12$:

| $n$ | prime factors | squarefree | $\mu(n)$ |
| --- | ------------- | ---------- | -------- |
| 1   |,             | yes        | 1        |
| 2   | $2$           | yes        | $-1$     |
| 3   | $3$           | yes        | $-1$     |
| 4   | $2^2$         | no         | 0        |
| 5   | $5$           | yes        | $-1$     |
| 6   | $2\cdot 3$    | yes        | $+1$     |
| 7   | $7$           | yes        | $-1$     |
| 8   | $2^3$         | no         | 0        |
| 9   | $3^2$         | no         | 0        |
| 10  | $2\cdot 5$    | yes        | $+1$     |
| 11  | $11$          | yes        | $-1$     |
| 12  | $2^2\cdot 3$  | no         | 0        |

#### Tiny Code (Easy Versions)

Python Version (Linear Sieve for $\mu$)

```python
def mobius_sieve(n):
    mu = [0] * (n + 1)
    mu[1] = 1
    primes = []
    is_comp = [False] * (n + 1)

    for i in range(2, n + 1):
        if not is_comp[i]:
            primes.append(i)
            mu[i] = -1
        for p in primes:
            ip = i * p
            if ip > n:
                break
            is_comp[ip] = True
            if i % p == 0:
                mu[ip] = 0           # p^2 divides ip
                break
            else:
                mu[ip] = -mu[i]      # add new distinct prime
    return mu

# Example
mu = mobius_sieve(50)
print([ (i, mu[i]) for i in range(1, 13) ])
```

C Version

```c
#include <stdio.h>
#include <stdbool.h>

void mobius_sieve(int n, int mu[]) {
    bool comp[n + 1];
    for (int i = 0; i <= n; i++) { comp[i] = false; mu[i] = 0; }
    mu[1] = 1;

    int primes[n + 1], pc = 0;

    for (int i = 2; i <= n; i++) {
        if (!comp[i]) {
            primes[pc++] = i;
            mu[i] = -1;
        }
        for (int j = 0; j < pc; j++) {
            long long ip = 1LL * i * primes[j];
            if (ip > n) break;
            comp[ip] = true;
            if (i % primes[j] == 0) {
                mu[ip] = 0;              // squared factor
                break;
            } else {
                mu[ip] = -mu[i];         // add a new distinct prime
            }
        }
    }
}

int main(void) {
    int N = 50;
    int mu[51];
    mobius_sieve(N, mu);
    for (int i = 1; i <= 12; i++)
        printf("mu(%d) = %d\n", i, mu[i]);
    return 0;
}
```

#### Why It Matters

- Möbius inversion in number theory:
  $$
  f(n)=\sum_{d\mid n} g(d)
  \quad \Longleftrightarrow \quad
  g(n)=\sum_{d\mid n} \mu(d) f!\left(\frac{n}{d}\right)
  $$
- Detects squarefree numbers: $\mu(n)\ne 0$ iff $n$ is squarefree
- Central to:

  * Dirichlet convolutions and multiplicative functions
  * Inclusion exclusion over divisors
  * Evaluating sums like $\sum_{n\le N}\mu(n)$
  * Counting problems with gcd or coprimality constraints

#### A Gentle Proof (Why It Works)

Induct on increasing integers while maintaining:

- If $i$ is prime then $\mu(i)=-1$
- For any prime $p$:

  * If $p \mid i$ then $\mu(i p)=0$ because $p^2 \mid i p$
  * If $p \nmid i$ then $\mu(i p)=-\mu(i)$ since $i p$ is squarefree with one more distinct prime

The linear loop stops at the first prime dividing $i$, so each composite is processed exactly once.
Thus total work is proportional to $N$ and the recurrence of signs matches the definition of $\mu$.

#### Try It Yourself

1. Verify that $\sum_{d\mid n} \mu(d) = 0$ for $n>1$ and equals $1$ for $n=1$.
2. Compute count of squarefree integers up to $N$ using $\sum_{n\le N} [\mu(n)\ne 0]$.
3. Implement Dirichlet convolution with precomputed $\mu$ to invert divisor sums.
4. Compare runtime of the linear sieve versus factoring each $n$.
5. Extend the sieve to also store the smallest prime factor and reuse it.

#### Test Cases

| $n$ | $\mu(n)$ |
| --- | -------- |
| 1   | 1        |
| 2   | $-1$     |
| 3   | $-1$     |
| 4   | 0        |
| 6   | $+1$     |
| 10  | $+1$     |
| 12  | 0        |
| 30  | $-1$     |

#### Complexity

- Time: $O(N)$ using the linear sieve
- Space: $O(N)$ for arrays

The Möbius sieve delivers $\mu$ values for an entire range in one pass, making inversion and squarefree analysis practical at scale.

### 546 Euler's Totient Sieve

The Euler Totient Function $\varphi(n)$ counts how many integers $1 \le k \le n$ are coprime to $n$.
That is, numbers such that $\gcd(k,n)=1$.

We can compute all $\varphi(1), \dots, \varphi(N)$ in $O(N)$ time using a linear sieve.

#### What Problem Are We Solving?

Naively, computing $\varphi(n)$ requires prime factorization:
$$
\varphi(n) = n \prod_{p|n}\left(1 - \frac{1}{p}\right)
$$
Doing that for each $n$ individually is too slow.

We want a fast sieve to compute $\varphi$ for every $n$ up to $N$ in one pass.

#### How Does It Work (Plain Language)

We use a linear sieve similar to prime generation:

1. Start with `phi[1] = 1`.
2. For each number $i$:

   * If $i$ is prime, then $\varphi(i)=i-1$.
   * For each prime $p$:

     * If $i \cdot p > N$, stop.
     * If $p \mid i$ (i.e., $p$ divides $i$):

       * $\varphi(i \cdot p) = \varphi(i) \cdot p$
       * break (to ensure linear time)
     * Else:

       * $\varphi(i \cdot p) = \varphi(i) \cdot (p-1)$

Each number is processed exactly once, preserving $O(N)$ complexity.

#### Example

Let's compute $\varphi(n)$ for $n=1$ to $10$:

| $n$ | Prime Factors | Formula                            | $\varphi(n)$ |
| --- | ------------- | ---------------------------------- | ------------ |
| 1   |,             | 1                                  | 1            |
| 2   | $2$           | $2(1-\frac{1}{2})$                 | 1            |
| 3   | $3$           | $3(1-\frac{1}{3})$                 | 2            |
| 4   | $2^2$         | $4(1-\frac{1}{2})$                 | 2            |
| 5   | $5$           | $5(1-\frac{1}{5})$                 | 4            |
| 6   | $2\cdot3$     | $6(1-\frac{1}{2})(1-\frac{1}{3})$  | 2            |
| 7   | $7$           | $7(1-\frac{1}{7})$                 | 6            |
| 8   | $2^3$         | $8(1-\frac{1}{2})$                 | 4            |
| 9   | $3^2$         | $9(1-\frac{1}{3})$                 | 6            |
| 10  | $2\cdot5$     | $10(1-\frac{1}{2})(1-\frac{1}{5})$ | 4            |

#### Tiny Code (Easy Versions)

Python Version

```python
def totient_sieve(n):
    phi = [0] * (n + 1)
    primes = []
    phi[1] = 1
    is_comp = [False] * (n + 1)

    for i in range(2, n + 1):
        if not is_comp[i]:
            primes.append(i)
            phi[i] = i - 1
        for p in primes:
            ip = i * p
            if ip > n:
                break
            is_comp[ip] = True
            if i % p == 0:
                phi[ip] = phi[i] * p
                break
            else:
                phi[ip] = phi[i] * (p - 1)
    return phi

# Example
phi = totient_sieve(20)
for i in range(1, 11):
    print(f"phi({i}) = {phi[i]}")
```

C Version

```c
#include <stdio.h>
#include <stdbool.h>

void totient_sieve(int n, int phi[]) {
    bool comp[n + 1];
    int primes[n + 1], pc = 0;
    for (int i = 0; i <= n; i++) { comp[i] = false; phi[i] = 0; }
    phi[1] = 1;

    for (int i = 2; i <= n; i++) {
        if (!comp[i]) {
            primes[pc++] = i;
            phi[i] = i - 1;
        }
        for (int j = 0; j < pc; j++) {
            int p = primes[j];
            long long ip = 1LL * i * p;
            if (ip > n) break;
            comp[ip] = true;
            if (i % p == 0) {
                phi[ip] = phi[i] * p;
                break;
            } else {
                phi[ip] = phi[i] * (p - 1);
            }
        }
    }
}

int main(void) {
    int n = 20, phi[21];
    totient_sieve(n, phi);
    for (int i = 1; i <= 10; i++)
        printf("phi(%d) = %d\n", i, phi[i]);
}
```

#### Why It Matters

The totient function $\varphi(n)$ is foundational in:

- Euler's theorem: $a^{\varphi(n)} \equiv 1 \pmod n$ if $\gcd(a,n)=1$
- RSA encryption: $\varphi(n)$ defines modular inverses for keys
- Counting reduced fractions: number of coprime pairs
- Group theory: size of multiplicative group $(\mathbb{Z}/n\mathbb{Z})^\times$

And useful in:

- Modular arithmetic
- Cryptography
- Number-theoretic combinatorics

#### A Gentle Proof (Why It Works)

Let $i$ be current number, $p$ a prime:

- If $p\nmid i$, $\varphi(i p) = \varphi(i) \cdot (p-1)$ since we multiply by a new distinct prime.
- If $p\mid i$, $\varphi(i p) = \varphi(i) \cdot p$ because we're extending a power of an existing prime.

Every number is built from its smallest prime factor in exactly one way → linear time.

#### Try It Yourself

1. Compute $\varphi(n)$ for $n=1$ to $20$.
2. Verify $\sum_{d|n} \varphi(d) = n$.
3. Plot $\varphi(n)/n$ to visualize density of coprime numbers.
4. Use $\varphi(n)$ to compute modular inverses via Euler's theorem.
5. Adapt sieve to also store primes and smallest prime factors.

#### Test Cases

| $n$ | $\varphi(n)$ |
| --- | ------------ |
| 1   | 1            |
| 2   | 1            |
| 3   | 2            |
| 4   | 2            |
| 5   | 4            |
| 6   | 2            |
| 10  | 4            |
| 12  | 4            |
| 15  | 8            |
| 20  | 8            |

#### Complexity

- Time: $O(N)$
- Space: $O(N)$

The Euler Totient Sieve is the backbone for arithmetic, cryptographic, and modular reasoning, one pass, all $\varphi(n)$ ready.

### 547 Divisor Count Sieve

The Divisor Count Sieve precomputes the number of positive divisors $d(n)$ (also written $\tau(n)$) for all integers $1 \le n \le N$.
It's a powerful tool in number theory and combinatorics, perfect for counting factors efficiently in $O(N\log N)$ time.

#### What Problem Are We Solving?

We want to compute the number of divisors for every integer up to $N$:
$$
d(n) = \sum_{i \mid n} 1
$$
or equivalently, if the prime factorization of $n$ is
$$
n = p_1^{a_1} p_2^{a_2} \dots p_k^{a_k},
$$
then
$$
d(n) = (a_1 + 1)(a_2 + 1)\dots(a_k + 1).
$$

Naively factoring each number is too slow.
The sieve method does it in one unified pass.

#### How Does It Work (Plain Language)

We use a divisor accumulation sieve:

For each $i$ from $1$ to $N$:

- Add $1$ to every multiple of $i$ (because $i$ divides each multiple)
- In code:

  ```python
  for i in range(1, N+1):
      for j in range(i, N+1, i):
          div[j] += 1
  ```

Each integer $n$ gets incremented once for each divisor $i \mid n$.
Total operations $\sim N\log N$.

#### Example

For $N = 10$:

| $n$ | Divisors    | $d(n)$ |
| --- | ----------- | ------ |
| 1   | 1           | 1      |
| 2   | 1, 2        | 2      |
| 3   | 1, 3        | 2      |
| 4   | 1, 2, 4     | 3      |
| 5   | 1, 5        | 2      |
| 6   | 1, 2, 3, 6  | 4      |
| 7   | 1, 7        | 2      |
| 8   | 1, 2, 4, 8  | 4      |
| 9   | 1, 3, 9     | 3      |
| 10  | 1, 2, 5, 10 | 4      |

#### Tiny Code (Easy Versions)

Python Version

```python
def divisor_count_sieve(n):
    div = [0] * (n + 1)
    for i in range(1, n + 1):
        for j in range(i, n + 1, i):
            div[j] += 1
    return div

# Example
N = 10
div = divisor_count_sieve(N)
for i in range(1, N + 1):
    print(f"d({i}) = {div[i]}")
```

C Version

```c
#include <stdio.h>

void divisor_count_sieve(int n, int div[]) {
    for (int i = 0; i <= n; i++) div[i] = 0;
    for (int i = 1; i <= n; i++)
        for (int j = i; j <= n; j += i)
            div[j]++;
}

int main(void) {
    int N = 10, div[11];
    divisor_count_sieve(N, div);
    for (int i = 1; i <= N; i++)
        printf("d(%d) = %d\n", i, div[i]);
}
```

#### Why It Matters

The divisor count function $d(n)$ is essential in:

- Divisor-sum problems
- Highly composite numbers
- Counting lattice points
- Summation over divisors (e.g. $\sum_{i=1}^N d(i)$)
- Dynamic programming and combinatorial enumeration
- Number-theoretic transforms

Also appears in formulas like:
$$
\sigma_0(n) = d(n), \quad \sigma_1(n) = \text{sum of divisors}
$$

#### A Gentle Proof (Why It Works)

Each integer $i$ divides exactly $\lfloor N/i \rfloor$ numbers ≤ $N$.
So in the nested loop, $i$ contributes $+1$ to $\lfloor N/i \rfloor$ entries.
Summing over $i$ gives total operations:
$$
\sum_{i=1}^{N} \frac{N}{i} \approx N \log N
$$

This is efficient and straightforward.

#### Try It Yourself

1. Print $d(n)$ for $n = 1$ to $30$.
2. Plot $d(n)$ to see how divisor counts fluctuate.
3. Modify code to compute sum of divisors:

   ```python
   divsum[j] += i
   ```
4. Combine with totient sieve to study divisor distributions.
5. Count numbers with exactly $k$ divisors.

#### Test Cases

| $n$ | Divisors                  | $d(n)$ |
| --- | ------------------------- | ------ |
| 1   | 1                         | 1      |
| 2   | 1, 2                      | 2      |
| 4   | 1, 2, 4                   | 3      |
| 6   | 1, 2, 3, 6                | 4      |
| 8   | 1, 2, 4, 8                | 4      |
| 12  | 1, 2, 3, 4, 6, 12         | 6      |
| 30  | 1, 2, 3, 5, 6, 10, 15, 30 | 8      |

#### Complexity

- Time: $O(N\log N)$
- Space: $O(N)$

The Divisor Count Sieve is simple yet mighty, precomputing factor structure for every number with a few nested loops.

### 548 Modular Precomputation

Modular precomputation prepares tables like factorials, inverse elements, and powers modulo $M$ so that later queries run in $O(1)$ time after an $O(N)$ or $O(N \log M)$ setup.
This is the backbone for fast combinatorics, DP, and number theoretic transforms under a modulus.

#### What Problem Are We Solving?

We often need to compute repeatedly:

- $a+b$, $a-b$, $a\cdot b$, $a^k \bmod M$
- modular inverses $a^{-1} \bmod M$
- binomial coefficients $\binom{n}{r} \bmod M$

Doing these from scratch per query costs $O(\log M)$ time via exponentiation.
With precomputation we answer in $O(1)$ per query after one linear pass.

#### What Do We Precompute?

For a prime modulus $M$ and a chosen limit $N$:

- `fact[i] = i! mod M` for $0 \le i \le N$
- `inv[i] = i^{-1} mod M` for $1 \le i \le N$
- `invfact[i] = (i!)^{-1} mod M` for $0 \le i \le N$
- optional: `powA[i] = A^i mod M` for fixed bases

Then
$$
\binom{n}{r} \bmod M = \text{fact}[n]\cdot \text{invfact}[r]\cdot \text{invfact}[n-r] \bmod M
$$
in $O(1)$ time.

#### How Does It Work (Plain Language)

1. Factorials: one forward pass
2. Inverse factorials: compute $\text{invfact}[N]=\text{fact}[N]^{M-2}\bmod M$ by Fermat then run backward
3. Elementwise inverses: either from `invfact` and `fact` or linearly by the identity
   $$
   \text{inv}[1]=1,\qquad
   \text{inv}[i]=M-\left(\left\lfloor \frac{M}{i}\right\rfloor\cdot \text{inv}[M\bmod i]\right)\bmod M
   $$
   which runs in $O(N)$.

These rely on $M$ being prime so that every $1\le i<M$ is invertible.

#### Edge Cases

- If $M$ is not prime: use extended Euclid to invert numbers coprime with $M$, or use factorial tables only for indices not hitting noninvertible factors. For combinatorics with composite $M$, use prime factorization of $M$ plus CRT, or use Lucas or Garner methods when applicable.
- Range limit: choose $N$ at least as large as the maximum $n$ you will query.

#### Tiny Code (Easy Versions)

Python Version (prime modulus)

```python
M = 109 + 7

def modpow(a, e, m=M):
    r = 1
    while e:
        if e & 1: r = r * a % m
        a = a * a % m
        e >>= 1
    return r

def build_tables(N, M=109+7):
    fact = [1] * (N + 1)
    for i in range(1, N + 1):
        fact[i] = fact[i - 1] * i % M

    invfact = [1] * (N + 1)
    invfact[N] = modpow(fact[N], M - 2, M)  # Fermat
    for i in range(N, 0, -1):
        invfact[i - 1] = invfact[i] * i % M

    inv = [0] * (N + 1)
    inv[1] = 1
    for i in range(2, N + 1):
        inv[i] = (M - (M // i) * inv[M % i] % M) % M

    return fact, invfact, inv

def nCr_mod(n, r, fact, invfact, M=109+7):
    if r < 0 or r > n: return 0
    return fact[n] * invfact[r] % M * invfact[n - r] % M

# Example
N = 1_000_000
fact, invfact, inv = build_tables(N, M)
print(nCr_mod(10, 3, fact, invfact, M))  # 120
```

C Version (prime modulus)

```c
#include <stdio.h>
#include <stdint.h>

const int MOD = 1000000007;

long long modpow(long long a, long long e) {
    long long r = 1 % MOD;
    while (e) {
        if (e & 1) r = (r * a) % MOD;
        a = (a * a) % MOD;
        e >>= 1;
    }
    return r;
}

void build_tables(int N, int fact[], int invfact[], int inv[]) {
    fact[0] = 1;
    for (int i = 1; i <= N; i++) fact[i] = (long long)fact[i-1] * i % MOD;

    invfact[N] = modpow(fact[N], MOD - 2);
    for (int i = N; i >= 1; i--) invfact[i-1] = (long long)invfact[i] * i % MOD;

    inv[1] = 1;
    for (int i = 2; i <= N; i++)
        inv[i] = (int)((MOD - (long long)(MOD / i) * inv[MOD % i] % MOD) % MOD);
}

int nCr_mod(int n, int r, int fact[], int invfact[]) {
    if (r < 0 || r > n) return 0;
    return (int)((long long)fact[n] * invfact[r] % MOD * invfact[n - r] % MOD);
}

int main(void) {
    int N = 1000000;
    static int fact[1000001], invfact[1000001], inv[1000001];
    build_tables(N, fact, invfact, inv);
    printf("%d\n", nCr_mod(10, 3, fact, invfact)); // 120
    return 0;
}
```

#### Why It Matters

- Fast combinatorics: $\binom{n}{r}$, permutations, multinomials in $O(1)$
- DP under modulo: convolution like transitions, counting paths
- Number theory: modular inverses and powers ready on demand
- Competitive programming and crypto prototypes where repeated modular queries are common

#### A Gentle Proof (Why It Works)

For prime $M$, $\mathbb{Z}_M^\times$ is a field. Fermat gives $a^{M-2}\equiv a^{-1}\pmod M$ for $a \not\equiv 0$.
Backward fill yields $\text{invfact}[i-1]=\text{invfact}[i]\cdot i \bmod M$, hence $(i-1)!^{-1}$.
Then
$$
\binom{n}{r} = \frac{n!}{r!,(n-r)!} \equiv \text{fact}[n]\cdot \text{invfact}[r]\cdot \text{invfact}[n-r] \pmod M.
$$
The linear inverse identity follows from writing $i\cdot \text{inv}[i]\equiv 1$ and recursing on $M\bmod i$.

#### Try It Yourself

1. Precompute up to $N=10^7$ with memory tuning and check that $\sum_{r=0}^n \binom{n}{r}\equiv 2^n \pmod M$.
2. Add a table of powers `powA[i]` for a fixed base $A$ to answer $A^k \bmod M$ in $O(1)$.
3. Implement multinomial: $\frac{n!}{a_1!\cdots a_k!}$ via `fact` and `invfact`.
4. For composite $M$, factor $M=\prod p_i^{e_i}$, compute modulo each $p_i^{e_i}$, then combine with CRT.
5. Benchmark precompute once vs on-demand exponentiation per query.

#### Test Cases

| Query                           | Answer                         |
| ------------------------------- | ------------------------------ |
| $\binom{10}{3}\bmod 10^9+7$     | 120                            |
| $\binom{1000}{500}\bmod 10^9+7$ | computed in $O(1)$ from tables |
| $a^{-1}\bmod M$ for $a=123456$  | `inv[a]`                       |
| $A^k\bmod M$ for many $k$       | `powA[k]` if precomputed       |

#### Complexity

- Precompute: $O(N)$ time, $O(N)$ space
- Per query: $O(1)$
- One exp: a single $O(\log M)$ exponentiation to seed `invfact[N]` if you choose the backward method

Modular precomputation turns heavy arithmetic into lookups. Pay once up front, answer instantly forever after.

### 549 Fermat's Little Theorem

Fermat's Little Theorem is the cornerstone of modular arithmetic. It states that if $p$ is a prime and $a$ is not divisible by $p$, then:

$$
a^{p-1} \equiv 1 \pmod p
$$

This powerful relationship underpins modular inverses, primality tests, and exponentiation optimizations.

#### What Problem Are We Solving?

We often need to simplify or invert large modular expressions:

- Computing $a^{-1} \bmod p$
- Simplifying huge exponents like $a^k \bmod p$
- Verifying primality (Fermat, Miller–Rabin tests)

Instead of performing expensive division, Fermat's theorem gives us:

$$
a^{-1} \equiv a^{p-2} \pmod p
$$

So inversion becomes modular exponentiation, achievable in $O(\log p)$.

#### How Does It Work (Plain Language)

When $p$ is prime, multiplication modulo $p$ forms a group of $p-1$ elements (excluding $0$).
By Lagrange's theorem, every element raised to the group size equals the identity:

$$
a^{p-1} \equiv 1 \pmod p
$$

Rearranging gives the modular inverse:

$$
a \cdot a^{p-2} \equiv 1 \pmod p
$$

So $a^{p-2}$ *is* the inverse of $a$ under mod $p$.

#### Example

Let $a=3$, $p=7$ (a prime):

$$
3^{6} = 729 \equiv 1 \pmod 7
$$

And indeed:

$$
3^{5} = 243 \equiv 5 \pmod 7
$$

Since $3 \times 5 = 15 \equiv 1 \pmod 7$,
$5$ is the modular inverse of $3$ mod $7$.

#### Tiny Code (Easy Versions)

Python Version

```python
def modpow(a, e, m):
    r = 1
    a %= m
    while e:
        if e & 1:
            r = r * a % m
        a = a * a % m
        e >>= 1
    return r

def modinv(a, p):
    return modpow(a, p - 2, p)  # Fermat's little theorem

# Example
p = 7
a = 3
print(modpow(a, p - 1, p))  # should be 1
print(modinv(a, p))         # should be 5
```

C Version

```c
#include <stdio.h>

long long modpow(long long a, long long e, long long m) {
    long long r = 1 % m;
    a %= m;
    while (e) {
        if (e & 1) r = r * a % m;
        a = a * a % m;
        e >>= 1;
    }
    return r;
}

long long modinv(long long a, long long p) {
    return modpow(a, p - 2, p); // Fermat's Little Theorem
}

int main(void) {
    long long a = 3, p = 7;
    printf("a^(p-1) mod p = %lld\n", modpow(a, p - 1, p));
    printf("Inverse = %lld\n", modinv(a, p));
}
```

#### Why It Matters

- Modular Inverses: Key for division under modulus (e.g., in combinatorics $\binom{n}{r}$ mod $p$).
- Exponent Reduction: For large exponents, use periodicity modulo $p-1$.
- Primality Tests: Forms the basis of Fermat and Miller–Rabin tests.
- RSA & Cryptography: Central to modular arithmetic with primes.

#### A Gentle Proof (Why It Works)

Consider all residues ${1,2,\ldots,p-1}$ modulo prime $p$.
Multiplying each by $a$ (where $\gcd(a,p)=1$) permutes them. Thus:

$$
1\cdot2\cdots(p-1) \equiv (a\cdot1)(a\cdot2)\cdots(a\cdot(p-1)) \pmod p
$$

Cancelling $(p-1)!$ (nonzero mod $p$ by Wilson's theorem) yields:

$$
a^{p-1} \equiv 1 \pmod p
$$

#### Try It Yourself

1. Verify $a^{p-1}\equiv 1$ for various primes $p$ and bases $a$.
2. Use it to compute modular inverses: test $a^{p-2}$ for different $a$.
3. Combine with modular exponentiation to speed up combinatorial formulas.
4. Explore what fails when $p$ is composite (Fermat pseudoprimes).
5. Implement Fermat primality test using $a^{p-1}\bmod p$.

#### Test Cases

| $a$ | $p$ | $a^{p-1}\bmod p$ | $a^{p-2}\bmod p$ (inverse) |
| --- | --- | ---------------- | -------------------------- |
| 2   | 5   | 1                | 3                          |
| 3   | 7   | 1                | 5                          |
| 4   | 11  | 1                | 3                          |
| 10  | 13  | 1                | 4                          |

#### Complexity

- Modular Exponentiation: $O(\log p)$
- Space: $O(1)$

Fermat's Little Theorem transforms division into exponentiation, bringing algebraic structure into computational arithmetic.

### 550 Wilson's Theorem

Wilson's Theorem gives a remarkable characterization of prime numbers using factorials:

$$
(p-1)! \equiv -1 \pmod p
$$

That is, for a prime $p$, the factorial of $(p-1)$ leaves a remainder of $p-1$ (or equivalently $-1$) when divided by $p$.

Conversely, if this congruence holds, $p$ must be prime.

#### What Problem Are We Solving?

We want a way to test primality or understand modular inverses through factorials.

While it's not practical for large primes due to factorial growth, Wilson's theorem is conceptually elegant and connects factorials, inverses, and modular arithmetic beautifully.

It shows how the multiplicative structure modulo $p$ is cyclic and symmetric.

#### How Does It Work (Plain Language)

For a prime $p$, every number $1,2,\dots,p-1$ has a unique inverse modulo $p$, and only $1$ and $p-1$ are self-inverse.

When we multiply all of them together:

- Each pair $a \cdot a^{-1}$ contributes $1$
- $1$ and $(p-1)$ contribute $1$ and $(p-1)$

So the whole product becomes:

$$
(p-1)! \equiv 1 \cdot (p-1) \equiv -1 \pmod p
$$

#### Example

Let's test small primes:

| $p$ | $(p-1)!$      | $(p-1)! \bmod p$    | Check |
| --- | ------------- | ------------------- | ----- |
| 2   | 1             | 1 ≡ -1 mod 2        | ✔     |
| 3   | 2             | 2 ≡ -1 mod 3        | ✔     |
| 5   | 24            | 24 ≡ -1 mod 5       | ✔     |
| 7   | 720           | 720 ≡ -1 mod 7      | ✔     |
| 11  | 10! = 3628800 | 3628800 ≡ -1 mod 11 | ✔     |

Try a composite $p=6$:
$5! = 120$, and $120 \bmod 6 = 0$ → fails.

So Wilson's condition is both necessary and sufficient for primality.

#### Tiny Code (Easy Versions)

Python Version

```python
def factorial_mod(n, m):
    res = 1
    for i in range(2, n + 1):
        res = (res * i) % m
    return res

def is_prime_wilson(p):
    if p < 2:
        return False
    return factorial_mod(p - 1, p) == p - 1

# Example
for p in [2, 3, 5, 6, 7, 11]:
    print(p, is_prime_wilson(p))
```

C Version

```c
#include <stdio.h>
#include <stdbool.h>

long long factorial_mod(int n, int mod) {
    long long res = 1;
    for (int i = 2; i <= n; i++)
        res = (res * i) % mod;
    return res;
}

bool is_prime_wilson(int p) {
    if (p < 2) return false;
    return factorial_mod(p - 1, p) == p - 1;
}

int main(void) {
    int ps[] = {2, 3, 5, 6, 7, 11};
    for (int i = 0; i < 6; i++)
        printf("%d %s\n", ps[i], is_prime_wilson(ps[i]) ? "prime" : "composite");
}
```

#### Why It Matters

Wilson's Theorem connects combinatorics, modular arithmetic, and primality:

- Primality characterization: $p$ is prime $\iff (p-1)! \equiv -1 \pmod p$
- Proof of group structure: $(\mathbb{Z}/p\mathbb{Z})^\times$ is a multiplicative group
- Factorial inverses: $(p-1)!$ acts as $-1$, enabling certain modular proofs

Though inefficient for large $p$, it's conceptually vital in number theory.

#### A Gentle Proof (Why It Works)

Let $p$ be prime.
The set ${1, 2, \dots, p-1}$ under multiplication mod $p$ forms a group.

Each element $a$ has an inverse $a^{-1}$.
Multiplying all elements:

$$
(p-1)! \equiv \prod_{a=1}^{p-1} a \equiv \prod_{a=1}^{p-1} a^{-1} \equiv (p-1)!^{-1} \pmod p
$$

Thus:

$$
((p-1)!)^2 \equiv 1 \pmod p
$$

So $(p-1)! \equiv \pm 1$.
If $(p-1)! \equiv 1$, then all numbers are self-inverse, only possible for $p=2$.
For $p>2$, it must be $-1$.

Conversely, if $(p-1)! \equiv -1$, $p$ cannot be composite (since composite factorials are $0 \bmod p$).

#### Try It Yourself

1. Verify $(p-1)! \bmod p$ for small primes.
2. Check why it fails for $p=4,6,8,9$.
3. Explore what happens mod a composite number (factorial will include factors of $p$).
4. Use it to show $(p-1)! + 1$ is divisible by $p$.
5. Try optimizing factorial modulo $p$ for small ranges.

#### Test Cases

| $p$ | $(p-1)!$ | $(p-1)! \bmod p$ | Prime? |
| --- | -------- | ---------------- | ------ |
| 2   | 1        | 1                | ✔      |
| 3   | 2        | 2                | ✔      |
| 4   | 6        | 2                | ✖      |
| 5   | 24       | 4                | ✔      |
| 6   | 120      | 0                | ✖      |
| 7   | 720      | 6                | ✔      |

#### Complexity

- Time: $O(p)$ (factorial modulo computation)
- Space: $O(1)$

Though inefficient for primality testing, Wilson's Theorem beautifully bridges factorials, inverses, and primes, a gem of elementary number theory.

# Section 56. Linear Algebra 

### 551 Gaussian Elimination

Gaussian Elimination is the fundamental algorithm for solving systems of linear equations, computing determinants, and finding matrix rank.
It systematically transforms a given matrix into an upper triangular form using row operations, after which solutions can be found by back-substitution.

#### What Problem Are We Solving?

We want to solve a system of $n$ linear equations in $n$ variables:

$$
A\mathbf{x} = \mathbf{b}
$$

where
$A$ is an $n \times n$ matrix,
$\mathbf{x}$ is the vector of unknowns,
$\mathbf{b}$ is the constant vector.

Instead of guessing or manually substituting, Gaussian elimination gives a systematic, deterministic, and polynomial-time method.

#### How Does It Work (Plain Language)

We perform elementary row operations to simplify the augmented matrix $[A | b]$:

1. Forward Elimination

   * For each column, select a pivot (nonzero element).
   * Swap rows if needed (partial pivoting).
   * Eliminate all entries below the pivot to make them zero.

2. Back Substitution

   * Once in upper-triangular form, solve from the last equation upward.

This converts the system into:
$$
U\mathbf{x} = \mathbf{c}
$$
where $U$ is upper triangular, easily solvable.

#### Example

Solve:
$$
\begin{cases}
2x + y - z = 8 \
-3x - y + 2z = -11 \
-2x + y + 2z = -3
\end{cases}
$$

Step 1: Write the augmented matrix

$$
\left[
\begin{array}{rrr|r}
2 & 1 & -1 & 8 \\
-3 & -1 & 2 & -11 \\
-2 & 1 & 2 & -3
\end{array}
\right]
$$


Step 2: Eliminate below first pivot

Use pivot = 2 (row 1).

$$
R_2 \gets R_2 + \tfrac{3}{2}R_1,\qquad
R_3 \gets R_3 + R_1
$$

$$
\left[
\begin{array}{rrr|r}
2 & 1 & -1 & 8 \\
0 & \tfrac{1}{2} & \tfrac{1}{2} & 1 \\
0 & 2 & 1 & 5
\end{array}
\right]
$$

Step 3: Eliminate below second pivot

Pivot = $\tfrac{1}{2}$ (row 2).

$$
R_3 \gets R_3 - 4R_2
$$

$$
\left[
\begin{array}{rrr|r}
2 & 1 & -1 & 8 \\
0 & \tfrac{1}{2} & \tfrac{1}{2} & 1 \\
0 & 0 & -1 & 1
\end{array}
\right]
$$


Step 4: Back Substitution

From bottom up:

- $-z = 1 \implies z = -1$
- $0.5y + 0.5z = 1 \implies y = 3$
- $2x + y - z = 8 \implies 2x + 3 + 1 = 8 \implies x = 2$

Solution: $(x, y, z) = (2, 3, -1)$

#### Tiny Code (Easy Versions)

Python Version

```python
def gaussian_elimination(a, b):
    n = len(a)
    for i in range(n):
        # Pivot
        max_row = max(range(i, n), key=lambda r: abs(a[r][i]))
        a[i], a[max_row] = a[max_row], a[i]
        b[i], b[max_row] = b[max_row], b[i]

        # Eliminate below
        for j in range(i + 1, n):
            factor = a[j][i] / a[i][i]
            for k in range(i, n):
                a[j][k] -= factor * a[i][k]
            b[j] -= factor * b[i]

    # Back substitution
    x = [0] * n
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - sum(a[i][j] * x[j] for j in range(i + 1, n))) / a[i][i]
    return x

A = [[2, 1, -1], [-3, -1, 2], [-2, 1, 2]]
B = [8, -11, -3]
print(gaussian_elimination(A, B))  # [2.0, 3.0, -1.0]
```

C Version

```c
#include <stdio.h>
#include <math.h>

#define N 3

void gaussian_elimination(double a[N][N], double b[N], double x[N]) {
    for (int i = 0; i < N; i++) {
        // Pivot
        int max_row = i;
        for (int r = i + 1; r < N; r++)
            if (fabs(a[r][i]) > fabs(a[max_row][i]))
                max_row = r;
        for (int c = 0; c < N; c++) {
            double tmp = a[i][c];
            a[i][c] = a[max_row][c];
            a[max_row][c] = tmp;
        }
        double tmp = b[i]; b[i] = b[max_row]; b[max_row] = tmp;

        // Eliminate
        for (int j = i + 1; j < N; j++) {
            double factor = a[j][i] / a[i][i];
            for (int k = i; k < N; k++)
                a[j][k] -= factor * a[i][k];
            b[j] -= factor * b[i];
        }
    }

    // Back substitution
    for (int i = N - 1; i >= 0; i--) {
        double sum = 0;
        for (int j = i + 1; j < N; j++)
            sum += a[i][j] * x[j];
        x[i] = (b[i] - sum) / a[i][i];
    }
}

int main() {
    double A[N][N] = {{2, 1, -1}, {-3, -1, 2}, {-2, 1, 2}};
    double B[N] = {8, -11, -3}, X[N];
    gaussian_elimination(A, B, X);
    printf("x = %.2f, y = %.2f, z = %.2f\n", X[0], X[1], X[2]);
}
```

#### Why It Matters

Gaussian elimination is foundational in:

- Solving $A\mathbf{x}=\mathbf{b}$
- Finding determinants ($\det(A)$ is product of pivots)
- Finding inverse matrices (by applying elimination to $[A|I]$)
- Computing matrix rank (count nonzero rows after elimination)

It underpins linear algebra libraries (BLAS/LAPACK), numerical solvers, and symbolic computation.

#### A Gentle Proof (Why It Works)

Each elementary row operation corresponds to multiplication by an invertible matrix $E_i$.
After a sequence:
$$
E_k \cdots E_1 A = U
$$
where $U$ is upper triangular.
Then:
$$
A = E_1^{-1}\cdots E_k^{-1}U
$$
so the system $A\mathbf{x}=\mathbf{b}$ becomes $U\mathbf{x} = (E_k\cdots E_1)\mathbf{b}$, which is solvable by back substitution.

Each step preserves solution space, ensuring correctness.

#### Try It Yourself

1. Solve a 3×3 system using Gaussian elimination manually.
2. Modify the algorithm to return determinant = product of pivots.
3. Extend to augmented matrix to compute inverse.
4. Add partial pivoting to handle zero pivots.
5. Compare with matrix decomposition (LU).

#### Test Cases

| System                                  | Solution    |
| --------------------------------------- | ----------- |
| $2x+y-z=8,\ -3x-y+2z=-11,\ -2x+y+2z=-3$ | $(2,3,-1)$  |
| $x+y=2,\ 2x-y=0$                        | $(2/3,4/3)$ |
| $x-y=1,\ 2x+y=4$                        | $(5/3,2/3)$ |

#### Complexity

- Time: $O(n^3)$
- Space: $O(n^2)$

Gaussian Elimination is the workhorse of linear algebra, every advanced solver builds upon it.

### 552 Gauss–Jordan Elimination

Gauss–Jordan Elimination extends Gaussian elimination by continuing the reduction process until the matrix is in reduced row echelon form (RREF), not just upper-triangular.
This makes it ideal for finding inverses, testing linear independence, and solving systems directly without back-substitution.

#### What Problem Are We Solving?

We want a full, systematic method to:

- Solve $A\mathbf{x} = \mathbf{b}$
- Find $A^{-1}$ (inverse matrix)
- Identify rank, null space, and pivots

Instead of stopping at an upper-triangular system (as in Gaussian elimination), we go further to make every pivot $1$ and clear all entries above and below it.

#### How Does It Work (Plain Language)

1. Form the augmented matrix
   Combine $A$ and $\mathbf{b}$:
   $[A | \mathbf{b}]$

2. Forward elimination
   For each pivot column:

   * Choose pivot (swap if needed)
   * Scale row so pivot = 1
   * Eliminate below (make zeros under pivot)

3. Backward elimination
   For each pivot column (starting from last):

   * Eliminate above (make zeros above pivot)

At the end, $A$ becomes the identity matrix and the right-hand side gives the solution vector.

If augmenting with $I$, the right-hand side becomes $A^{-1}$.

#### Example

Solve:

$$
\begin{cases}
x + y + z = 6 \\
2y + 5z = -4 \\
2x + 5y - z = 27
\end{cases}
$$

Step 1: Write augmented matrix

$$
\left[
\begin{array}{rrr|r}
1 & 1 & 1 & 6 \\
0 & 2 & 5 & -4 \\
2 & 5 & -1 & 27
\end{array}
\right]
$$

Step 2: Eliminate below first pivot

$R_3 \gets R_3 - 2R_1$

$$
\left[
\begin{array}{rrr|r}
1 & 1 & 1 & 6 \\
0 & 2 & 5 & -4 \\
0 & 3 & -3 & 15
\end{array}
\right]
$$

Step 3: Pivot at row 2

$R_2 \gets \tfrac{1}{2}R_2$

$$
\left[
\begin{array}{rrr|r}
1 & 1 & 1 & 6 \\
0 & 1 & \tfrac{5}{2} & -2 \\
0 & 3 & -3 & 15
\end{array}
\right]
$$

Step 4: Eliminate below and above the second pivot

$R_3 \gets R_3 - 3R_2,\quad R_1 \gets R_1 - R_2$

$$
\left[
\begin{array}{rrr|r}
1 & 0 & -\tfrac{3}{2} & 8 \\
0 & 1 & \tfrac{5}{2} & -2 \\
0 & 0 & -\tfrac{21}{2} & 21
\end{array}
\right]
$$

Normalize third pivot

$R_3 \gets -\tfrac{2}{21} R_3$

$$
\left[
\begin{array}{rrr|r}
1 & 0 & -\tfrac{3}{2} & 8 \\
0 & 1 & \tfrac{5}{2} & -2 \\
0 & 0 & 1 & -2
\end{array}
\right]
$$

Eliminate above the third pivot

$R_1 \gets R_1 + \tfrac{3}{2}R_3,\quad R_2 \gets R_2 - \tfrac{5}{2}R_3$

$$
\left[
\begin{array}{rrr|r}
1 & 0 & 0 & 5 \\
0 & 1 & 0 & 3 \\
0 & 0 & 1 & -2
\end{array}
\right]
$$

Solution:

$$
x=5,\quad y=3,\quad z=-2
$$


#### Tiny Code (Easy Versions)

Python Version

```python
def gauss_jordan(a, b):
    n = len(a)
    # Augment A with b
    for i in range(n):
        a[i].append(b[i])

    for i in range(n):
        # Pivot selection
        max_row = max(range(i, n), key=lambda r: abs(a[r][i]))
        a[i], a[max_row] = a[max_row], a[i]

        # Normalize pivot row
        pivot = a[i][i]
        for j in range(i, n + 1):
            a[i][j] /= pivot

        # Eliminate all other rows
        for k in range(n):
            if k != i:
                factor = a[k][i]
                for j in range(i, n + 1):
                    a[k][j] -= factor * a[i][j]

    # Extract solution
    return [a[i][n] for i in range(n)]

A = [[1, 1, 1],
     [0, 2, 5],
     [2, 5, -1]]
B = [6, -4, 27]
print(gauss_jordan(A, B))  # [5.0, 3.0, -2.0]
```

C Version

```c
#include <stdio.h>
#include <math.h>

#define N 3

void gauss_jordan(double a[N][N+1]) {
    for (int i = 0; i < N; i++) {
        // Pivot selection
        int max_row = i;
        for (int r = i + 1; r < N; r++)
            if (fabs(a[r][i]) > fabs(a[max_row][i]))
                max_row = r;
        for (int c = 0; c <= N; c++) {
            double tmp = a[i][c];
            a[i][c] = a[max_row][c];
            a[max_row][c] = tmp;
        }

        // Normalize pivot row
        double pivot = a[i][i];
        for (int c = 0; c <= N; c++)
            a[i][c] /= pivot;

        // Eliminate other rows
        for (int r = 0; r < N; r++) {
            if (r != i) {
                double factor = a[r][i];
                for (int c = 0; c <= N; c++)
                    a[r][c] -= factor * a[i][c];
            }
        }
    }
}

int main() {
    double A[N][N+1] = {
        {1, 1, 1, 6},
        {0, 2, 5, -4},
        {2, 5, -1, 27}
    };
    gauss_jordan(A);
    for (int i = 0; i < N; i++)
        printf("x%d = %.2f\n", i + 1, A[i][N]);
}
```

#### Why It Matters

Gauss–Jordan Elimination is versatile:

- Direct solution without back-substitution
- Matrix inversion by applying to $[A|I]$
- Rank computation (number of pivots)
- Linear independence testing

It's conceptually clear and forms the basis for high-level linear algebra routines.

#### A Gentle Proof (Why It Works)

Each row operation corresponds to multiplication by an invertible matrix $E_i$.
After full reduction:
$$
E_k \cdots E_1 [A | I] = [I | A^{-1}]
$$
Thus, $A^{-1} = E_k \cdots E_1$.
The method transforms $A$ into $I$ through reversible operations, so the right-hand side evolves into $A^{-1}$.

For $A\mathbf{x} = \mathbf{b}$, augmenting $A$ with $\mathbf{b}$ gives the unique solution vector.

#### Try It Yourself

1. Solve $A\mathbf{x}=\mathbf{b}$ using full RREF.
2. Augment $A$ with $I$ and compute $A^{-1}$.
3. Count nonzero rows to find rank.
4. Implement with partial pivoting for stability.
5. Compare with LU decomposition in performance.

#### Test Cases

| System                            | Solution   |
| --------------------------------- | ---------- |
| $x+y+z=6,\ 2y+5z=-4,\ 2x+5y-z=27$ | $(5,3,-2)$ |
| $x+y=2,\ 3x-2y=1$                 | $(1,1)$    |
| $2x+y=5,\ 4x-2y=6$                | $(2,1)$    |

#### Complexity

- Time: $O(n^3)$
- Space: $O(n^2)$

Gauss–Jordan is a complete solver, producing identity on the left, and solution or inverse on the right, all in one pass.





### 553 LU Decomposition

LU Decomposition factors a matrix $A$ into the product of a lower triangular matrix $L$ and an upper triangular matrix $U$:

$$
A = L \cdot U
$$

This factorization is a workhorse of numerical linear algebra. Once $A$ is decomposed, we can solve systems $A\mathbf{x}=\mathbf{b}$ quickly for multiple right-hand sides by forward and backward substitution.

#### What Problem Are We Solving?

We want to solve $A\mathbf{x}=\mathbf{b}$ efficiently and repeatedly.

Gaussian elimination works once, but LU decomposition reuses the factorization:

- First solve $L\mathbf{y}=\mathbf{b}$ (forward substitution)
- Then solve $U\mathbf{x}=\mathbf{y}$ (back substitution)

Also useful for:

- Determinant computation ($\det(A)=\det(L)\det(U)$)
- Matrix inversion
- Numerical stability with pivoting ($PA=LU$)

#### How Does It Work (Plain Language)

LU decomposition performs the same operations as Gaussian elimination but records the multipliers in $L$.

1. Initialize

   * $L$ as identity
   * $U$ as a copy of $A$

2. Eliminate below each pivot

   * For $i$ from $0$ to $n-1$:

     * For $j>i$:
       $L[j][i] = U[j][i] / U[i][i]$
       Subtract $L[j][i] \times$ (row $i$) from row $j$ of $U$

At the end:

- $L$ has ones on the diagonal and multipliers below.
- $U$ is upper triangular.

If $A$ requires row swaps for stability, we include a permutation matrix $P$:
$$
PA = LU
$$

#### Example

Decompose
$$
A = \begin{bmatrix}
2 & 3 & 1\
4 & 7 & 7\
-2 & 4 & 5
\end{bmatrix}
$$

Step 1: Pivot at $a_{11}=2$

Eliminate below:

- Row 2: $L_{21}=4/2=2$ → Row2 = Row2 - 2*Row1
- Row 3: $L_{31}=-2/2=-1$ → Row3 = Row3 + Row1

$L = \begin{bmatrix}
1 & 0 & 0\
2 & 1 & 0\
-1 & 0 & 1
\end{bmatrix},\
U = \begin{bmatrix}
2 & 3 & 1\
0 & 1 & 5\
0 & 7 & 6
\end{bmatrix}$

Step 2: Pivot at $U_{22}=1$

Eliminate below:

- Row3: $L_{32}=7/1=7$ → Row3 = Row3 - 7*Row2

$L = \begin{bmatrix}
1 & 0 & 0\
2 & 1 & 0\
-1 & 7 & 1
\end{bmatrix},\
U = \begin{bmatrix}
2 & 3 & 1\
0 & 1 & 5\
0 & 0 & -29
\end{bmatrix}$

Check: $A = L \cdot U$

#### Tiny Code (Easy Versions)

Python Version

```python
def lu_decompose(A):
    n = len(A)
    L = [[0]*n for _ in range(n)]
    U = [[0]*n for _ in range(n)]

    for i in range(n):
        L[i][i] = 1

    for i in range(n):
        # Upper Triangular
        for k in range(i, n):
            U[i][k] = A[i][k] - sum(L[i][j] * U[j][k] for j in range(i))
        # Lower Triangular
        for k in range(i + 1, n):
            L[k][i] = (A[k][i] - sum(L[k][j] * U[j][i] for j in range(i))) / U[i][i]
    return L, U

A = [
    [2, 3, 1],
    [4, 7, 7],
    [-2, 4, 5]
$$
L, U = lu_decompose(A)
print("L =", L)
print("U =", U)
```

C Version

```c
#include <stdio.h>

#define N 3

void lu_decompose(double A[N][N], double L[N][N], double U[N][N]) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            L[i][j] = (i == j) ? 1 : 0;
            U[i][j] = 0;
        }
    }

    for (int i = 0; i < N; i++) {
        for (int k = i; k < N; k++) {
            double sum = 0;
            for (int j = 0; j < i; j++)
                sum += L[i][j] * U[j][k];
            U[i][k] = A[i][k] - sum;
        }
        for (int k = i + 1; k < N; k++) {
            double sum = 0;
            for (int j = 0; j < i; j++)
                sum += L[k][j] * U[j][i];
            L[k][i] = (A[k][i] - sum) / U[i][i];
        }
    }
}

int main(void) {
    double A[N][N] = {{2,3,1},{4,7,7},{-2,4,5}}, L[N][N], U[N][N];
    lu_decompose(A, L, U);

    printf("L:\n");
    for (int i=0;i<N;i++){ for(int j=0;j<N;j++) printf("%6.2f ",L[i][j]); printf("\n"); }
    printf("U:\n");
    for (int i=0;i<N;i++){ for(int j=0;j<N;j++) printf("%6.2f ",U[i][j]); printf("\n"); }
}
```

#### Why It Matters

- Fast solving: Reuse $LU$ to solve for multiple $\mathbf{b}$
- Determinant: $\det(A)=\prod_i U_{ii}$
- Inverse: Solve $LU\mathbf{x}=\mathbf{e}_i$ for each $i$
- Foundation: Basis of Cholesky, Crout, and Doolittle variants
- Stability: Combine with pivoting for robustness ($PA=LU$)

#### A Gentle Proof (Why It Works)

Each elimination step corresponds to multiplying $A$ by an elementary lower-triangular matrix $E_i$.
After all steps:
$$
U = E_k E_{k-1} \cdots E_1 A
$$
Then
$$
A = E_1^{-1} E_2^{-1} \cdots E_k^{-1} U
$$
Let
$$
L = E_1^{-1}E_2^{-1}\cdots E_k^{-1}
$$
so $A = L U$, with $L$ lower-triangular (unit diagonal) and $U$ upper-triangular.

#### Try It Yourself

1. Perform LU decomposition on a $3\times3$ matrix by hand.
2. Verify $A = L \cdot U$ by multiplication.
3. Solve $A\mathbf{x}=\mathbf{b}$ via forward + backward substitution.
4. Implement determinant computation via $\prod U_{ii}$.
5. Add partial pivoting (compute $P,L,U$).

#### Test Cases

| $A$                                                     | $L$                                                     | $U$                                                     |
| ------------------------------------------------------- | ------------------------------------------------------- | ------------------------------------------------------- |
| $\begin{bmatrix} 2 & 3 \\ 4 & 7 \end{bmatrix}$          | $\begin{bmatrix} 1 & 0 \\ 2 & 1 \end{bmatrix}$          | $\begin{bmatrix} 2 & 3 \\ 0 & 1 \end{bmatrix}$          |
| $\begin{bmatrix} 1 & 1 \\ 2 & 3 \end{bmatrix}$          | $\begin{bmatrix} 1 & 0 \\ 2 & 1 \end{bmatrix}$          | $\begin{bmatrix} 1 & 1 \\ 0 & 1 \end{bmatrix}$          |


#### Complexity

- Time: $O(n^3)$ (factorization)
- Solve: $O(n^2)$ per right-hand side
- Space: $O(n^2)$

LU decomposition is the backbone of numerical linear algebra, turning Gaussian elimination into a reusable, modular tool.

### 554 Cholesky Decomposition

Cholesky Decomposition is a special case of LU decomposition for symmetric positive definite (SPD) matrices.
It factors a matrix $A$ into the product of a lower triangular matrix $L$ and its transpose:

$$
A = L \cdot L^{T}
$$

This method is twice as efficient as LU decomposition and numerically more stable for SPD matrices, a favorite in optimization, machine learning, and statistics.

#### What Problem Are We Solving?

We want to solve $A\mathbf{x}=\mathbf{b}$ efficiently when $A$ is symmetric ($A=A^T$) and positive definite ($\mathbf{x}^T A \mathbf{x} > 0$ for all $\mathbf{x}\neq0$).

Instead of general elimination, we exploit symmetry to reduce work by half.

Once we find $L$, we solve:

- Forward: $L\mathbf{y}=\mathbf{b}$
- Backward: $L^{T}\mathbf{x}=\mathbf{y}$

#### How Does It Work (Plain Language)

We build $L$ row by row (or column by column), using the formulas:

For diagonal elements:
$$
L_{ii} = \sqrt{A_{ii} - \sum_{k=1}^{i-1}L_{ik}^2}
$$

For off-diagonal elements:
$$
L_{ij} = \frac{1}{L_{jj}}\Big(A_{ij} - \sum_{k=1}^{j-1}L_{ik}L_{jk}\Big), \quad i>j
$$

The upper half is just the transpose of $L$.

#### Example

Given
$$
A =
\begin{bmatrix}
4 & 12 & -16\
12 & 37 & -43\
-16 & -43 & 98
\end{bmatrix}
$$

Step 1: $L_{11} = \sqrt{4} = 2$

Step 2: $L_{21} = 12/2 = 6$, $L_{31} = -16/2 = -8$

Step 3: $L_{22} = \sqrt{37 - 6^2} = \sqrt{1} = 1$

Step 4: $L_{32} = \frac{-43 - (-8)(6)}{1} = 5$

Step 5: $L_{33} = \sqrt{98 - (-8)^2 - 5^2} = \sqrt{9} = 3$

So:
$$
L =
\begin{bmatrix}
2 & 0 & 0\
6 & 1 & 0\
-8 & 5 & 3
\end{bmatrix}
,\quad
A = L \cdot L^{T}
$$

#### Tiny Code (Easy Versions)

Python

```python
import math

def cholesky(A):
    n = len(A)
    L = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1):
            s = sum(L[i][k]*L[j][k] for k in range(j))
            if i == j:
                L[i][j] = math.sqrt(A[i][i] - s)
            else:
                L[i][j] = (A[i][j] - s) / L[j][j]
    return L

A = [
    [4, 12, -16],
    [12, 37, -43],
    [-16, -43, 98]
$$

L = cholesky(A)
for row in L:
    print(row)
```

C

```c
#include <stdio.h>
#include <math.h>

#define N 3

void cholesky(double A[N][N], double L[N][N]) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j <= i; j++) {
            double sum = 0;
            for (int k = 0; k < j; k++)
                sum += L[i][k] * L[j][k];
            if (i == j)
                L[i][j] = sqrt(A[i][i] - sum);
            else
                L[i][j] = (A[i][j] - sum) / L[j][j];
        }
    }
}

int main(void) {
    double A[N][N] = {
        {4, 12, -16},
        {12, 37, -43},
        {-16, -43, 98}
    }, L[N][N] = {0};

    cholesky(A, L);

    printf("L:\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++)
            printf("%8.3f ", L[i][j]);
        printf("\n");
    }
}
```

#### Why It Matters

- Half the work of LU decomposition.
- Numerically stable for SPD matrices.
- Essential for:

  * Least squares regression
  * Kalman filters
  * Gaussian processes
  * Covariance matrix decomposition

#### A Gentle Proof (Why It Works)

For an SPD matrix $A$, all leading principal minors are positive.
This guarantees every diagonal pivot ($A_{ii} - \sum L_{ik}^2$) is positive, so we can take square roots.

Hence $L$ exists and is unique.

We can see $A = L L^T$ because every element $A_{ij}$ is reproduced by summing the dot product of row $i$ and row $j$ of $L$.

#### Try It Yourself

1. Decompose a $3\times3$ SPD matrix by hand.
2. Multiply $L L^T$ to verify.
3. Use it to solve $A\mathbf{x}=\mathbf{b}$.
4. Compare runtime with LU decomposition.
5. Test on a covariance matrix (e.g. symmetric, positive).

#### Test Case

$$
A =
\begin{bmatrix}
25 & 15 & -5\
15 & 18 &  0\
-5 &  0 & 11
\end{bmatrix}
\Rightarrow
L =
\begin{bmatrix}
5 & 0 & 0\
3 & 3 & 0\
-1 & 1 & 3
\end{bmatrix}
$$

Check: $L L^T = A$

#### Complexity

- Time: $O(n^3/3)$
- Solve: $O(n^2)$
- Space: $O(n^2)$

Cholesky decomposition is your go-to method for fast, stable, and symmetric systems, a cornerstone in numerical analysis and machine learning.

### 555 QR Decomposition

QR Decomposition breaks a matrix $A$ into two orthogonal factors:

$$
A = Q \cdot R
$$

where $Q$ is orthogonal ($Q^T Q = I$) and $R$ is upper triangular.
This decomposition is key in solving least squares problems, eigenvalue computations, and orthogonalization tasks.

#### What Problem Are We Solving?

We often need to solve $A\mathbf{x} = \mathbf{b}$ when $A$ is not square (e.g. $m > n$).
Instead of normal equations $(A^TA)\mathbf{x}=A^T\mathbf{b}$, QR gives a more stable solution:

$$
A = Q R \implies R \mathbf{x} = Q^T \mathbf{b}
$$

No need to form $A^T A$, which can amplify numerical errors.

#### How Does It Work (Plain Language)

QR decomposition orthogonalizes the columns of $A$ step by step:

1. Start with columns $a_1, a_2, \ldots, a_n$ of $A$
2. Build orthogonal basis $q_1, q_2, \ldots, q_n$ using Gram–Schmidt
3. Normalize to get orthonormal columns of $Q$
4. Compute R as projection coefficients

For each $i$:
$$
r_{ii} = |a_i - \sum_{j=1}^{i-1}r_{ji}q_j|, \quad q_i = \frac{a_i - \sum_{j=1}^{i-1}r_{ji}q_j}{r_{ii}}
$$

Compactly:

- $Q$: orthonormal basis
- $R$: upper-triangular coefficients

Variants:

- Classical Gram–Schmidt (CGS): simple but unstable
- Modified Gram–Schmidt (MGS): more stable
- Householder reflections: best for numerical accuracy

#### Example

Let
$$
A =
\begin{bmatrix}
1 & 1 \
1 & -1 \
1 & 1
\end{bmatrix}
$$

Step 1: Take $a_1 = (1, 1, 1)^T$

$$
q_1 = \frac{a_1}{|a_1|} = \frac{1}{\sqrt{3}}(1, 1, 1)
$$

Step 2: Remove projection from $a_2$

$$
r_{12} = q_1^T a_2 = \frac{1}{\sqrt{3}}(1 + (-1) + 1) = \frac{1}{\sqrt{3}}
$$

$$
u_2 = a_2 - r_{12} q_1 = (1, -1, 1) - \frac{1}{\sqrt{3}}\cdot\frac{1}{\sqrt{3}}(1,1,1) = (1-\tfrac{1}{3}, -1-\tfrac{1}{3}, 1-\tfrac{1}{3})
$$

$$
u_2 = \left(\frac{2}{3}, -\frac{4}{3}, \frac{2}{3}\right)
$$

Normalize:

$$
q_2 = \frac{u_2}{|u_2|} = \frac{1}{\sqrt{8/3}} \left(\frac{2}{3}, -\frac{4}{3}, \frac{2}{3}\right) = \frac{1}{\sqrt{6}}(1, -2, 1)
$$

Then $Q = [q_1\ q_2]$, $R = Q^T A$

#### Tiny Code (Easy Versions)

Python

```python
import numpy as np

def qr_decompose(A):
    A = np.array(A, dtype=float)
    m, n = A.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))

    for i in range(n):
        v = A[:, i]
        for j in range(i):
            R[j, i] = np.dot(Q[:, j], A[:, i])
            v = v - R[j, i] * Q[:, j]
        R[i, i] = np.linalg.norm(v)
        Q[:, i] = v / R[i, i]
    return Q, R

A = [[1, 1], [1, -1], [1, 1]]
Q, R = qr_decompose(A)
print("Q =\n", Q)
print("R =\n", R)
```

C (Simplified Gram–Schmidt)

```c
#include <stdio.h>
#include <math.h>

#define M 3
#define N 2

void qr_decompose(double A[M][N], double Q[M][N], double R[N][N]) {
    for (int i = 0; i < N; i++) {
        for (int k = 0; k < M; k++)
            Q[k][i] = A[k][i];
        for (int j = 0; j < i; j++) {
            R[j][i] = 0;
            for (int k = 0; k < M; k++)
                R[j][i] += Q[k][j] * A[k][i];
            for (int k = 0; k < M; k++)
                Q[k][i] -= R[j][i] * Q[k][j];
        }
        R[i][i] = 0;
        for (int k = 0; k < M; k++)
            R[i][i] += Q[k][i] * Q[k][i];
        R[i][i] = sqrt(R[i][i]);
        for (int k = 0; k < M; k++)
            Q[k][i] /= R[i][i];
    }
}

int main(void) {
    double A[M][N] = {{1,1},{1,-1},{1,1}}, Q[M][N], R[N][N];
    qr_decompose(A, Q, R);

    printf("Q:\n");
    for(int i=0;i<M;i++){ for(int j=0;j<N;j++) printf("%8.3f ", Q[i][j]); printf("\n"); }

    printf("R:\n");
    for(int i=0;i<N;i++){ for(int j=0;j<N;j++) printf("%8.3f ", R[i][j]); printf("\n"); }
}
```

#### Why It Matters

- Numerical stability for least squares
- Orthogonal basis for column space
- Eigenvalue algorithms (QR iteration)
- Machine learning: regression, PCA
- Signal processing: orthogonalization, projections

#### A Gentle Proof (Why It Works)

Every full-rank matrix $A$ with independent columns can be written as:
$$
A = [a_1, a_2, \ldots, a_n] = [q_1, q_2, \ldots, q_n] R
$$

Each $a_i$ is expressed as a linear combination of the orthogonal basis vectors $q_j$:
$$
a_i = \sum_{j=1}^{i} r_{ji} q_j
$$

Collecting $q_j$ as columns of $Q$ gives $A=QR$.

#### Try It Yourself

1. Orthogonalize two 3D vectors manually.
2. Verify $Q^T Q = I$.
3. Compute $R = Q^T A$.
4. Use $QR$ to solve an overdetermined system.
5. Compare classical vs modified Gram–Schmidt.

#### Test Case

$$
A =
\begin{bmatrix}
1 & 1\
1 & 0\
0 & 1
\end{bmatrix}
,\quad
Q =
\begin{bmatrix}
\frac{1}{\sqrt{2}} & \frac{1}{\sqrt{6}}\
\frac{1}{\sqrt{2}} & -\frac{1}{\sqrt{6}}\
0 & \frac{2}{\sqrt{6}}
\end{bmatrix}
,\quad
R =
\begin{bmatrix}
\sqrt{2} & \frac{1}{\sqrt{2}}\
0 & \sqrt{\frac{3}{2}}
\end{bmatrix}
$$

Check: $A = Q R$

#### Complexity

- Time: $O(mn^2)$
- Space: $O(mn)$
- Stability: High (especially with Householder reflections)

QR decomposition is your orthogonal compass, guiding you to stable, geometric solutions in least squares, PCA, and spectral algorithms.

### 556 Matrix Inversion (Gauss–Jordan Method)

Matrix Inversion finds a matrix $A^{-1}$ such that

$$
A \cdot A^{-1} = I
$$

This operation is fundamental for solving systems, transforming spaces, and expressing linear mappings. While direct inversion is rarely used in practice (solving $A\mathbf{x}=\mathbf{b}$ is cheaper), learning how to compute it reveals the structure of linear algebra itself.

#### What Problem Are We Solving?

We want to find $A^{-1}$, the matrix that undoes $A$.
Once we have $A^{-1}$, any system $A\mathbf{x}=\mathbf{b}$ can be solved simply by:

$$
\mathbf{x} = A^{-1} \mathbf{b}
$$

But inversion is only defined when $A$ is square and non-singular (i.e. $\det(A)\neq0$).

#### How Does It Work (Plain Language)

The Gauss–Jordan method augments $A$ with the identity matrix $I$, then performs row operations to transform $A$ into $I$.
Whatever $I$ becomes on the right-hand side is $A^{-1}$.

Step-by-step:

1. Form the augmented matrix $[A | I]$
2. Apply row operations to make $A$ into $I$
3. The right half becomes $A^{-1}$

#### Example

Let
$$
A=
\begin{bmatrix}
2 & 1\\
5 & 3
\end{bmatrix}.
$$

Augment with the identity:
$$
\left[
\begin{array}{cc|cc}
2 & 1 & 1 & 0\\
5 & 3 & 0 & 1
\end{array}
\right].
$$

Step 1: $R_1 \gets \tfrac{1}{2}R_1$
$$
\left[
\begin{array}{cc|cc}
1 & \tfrac{1}{2} & \tfrac{1}{2} & 0\\
5 & 3 & 0 & 1
\end{array}
\right].
$$

Step 2: $R_2 \gets R_2 - 5R_1$
$$
\left[
\begin{array}{cc|cc}
1 & \tfrac{1}{2} & \tfrac{1}{2} & 0\\
0 & \tfrac{1}{2} & -\tfrac{5}{2} & 1
\end{array}
\right].
$$

Step 3: $R_2 \gets 2R_2$
$$
\left[
\begin{array}{cc|cc}
1 & \tfrac{1}{2} & \tfrac{1}{2} & 0\\
0 & 1 & -5 & 2
\end{array}
\right].
$$

Step 4: $R_1 \gets R_1 - \tfrac{1}{2}R_2$
$$
\left[
\begin{array}{cc|cc}
1 & 0 & 3 & -1\\
0 & 1 & -5 & 2
\end{array}
\right].
$$

Thus
$$
A^{-1}=
\begin{bmatrix}
3 & -1\\
-5 & 2
\end{bmatrix}.
$$

Check:
$$
A\,A^{-1}=
\begin{bmatrix}
2 & 1\\
5 & 3
\end{bmatrix}
\begin{bmatrix}
3 & -1\\
-5 & 2
\end{bmatrix}
=
\begin{bmatrix}
1 & 0\\
0 & 1
\end{bmatrix}.
$$


#### Tiny Code (Easy Versions)

Python

```python
def invert_matrix(A):
    n = len(A)
    # Augment with identity
    aug = [row + [int(i == j) for j in range(n)] for i, row in enumerate(A)]

    for i in range(n):
        # Make pivot 1
        pivot = aug[i][i]
        for j in range(2*n):
            aug[i][j] /= pivot

        # Eliminate other rows
        for k in range(n):
            if k != i:
                factor = aug[k][i]
                for j in range(2*n):
                    aug[k][j] -= factor * aug[i][j]

    # Extract inverse
    return [row[n:] for row in aug]

A = [[2,1],[5,3]]
A_inv = invert_matrix(A)
for row in A_inv:
    print(row)
```

C

```c
#include <stdio.h>

#define N 2

void invert_matrix(double A[N][N], double I[N][N]) {
    double aug[N][2*N];
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            aug[i][j] = A[i][j];
            aug[i][j+N] = (i == j) ? 1 : 0;
        }
    }

    for (int i = 0; i < N; i++) {
        double pivot = aug[i][i];
        for (int j = 0; j < 2*N; j++)
            aug[i][j] /= pivot;

        for (int k = 0; k < N; k++) {
            if (k == i) continue;
            double factor = aug[k][i];
            for (int j = 0; j < 2*N; j++)
                aug[k][j] -= factor * aug[i][j];
        }
    }

    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            I[i][j] = aug[i][j+N];
}

int main(void) {
    double A[N][N] = {{2,1},{5,3}}, Inv[N][N];
    invert_matrix(A, Inv);
    for (int i=0;i<N;i++){ for(int j=0;j<N;j++) printf("%6.2f ",Inv[i][j]); printf("\n"); }
}
```

#### Why It Matters

- Conceptual clarity: defines what "inverse" means
- Solving systems: $\mathbf{x}=A^{-1}\mathbf{b}$
- Geometry: undo linear transformations
- Symbolic algebra: e.g. transformations, coordinate changes

In practice, you rarely compute $A^{-1}$ explicitly, you factor and solve instead.

#### A Gentle Proof (Why It Works)

Elementary row operations correspond to multiplying by elementary matrices $E_i$.
If

$$
E_k \cdots E_2 E_1 A = I
$$

then

$$
A^{-1} = E_k \cdots E_2 E_1
$$

Each $E_i$ is invertible, so their product is invertible.

#### Try It Yourself

1. Invert a $3 \times 3$ matrix by hand.
2. Verify $A \cdot A^{-1} = I$.
3. Observe what happens if $\det(A)=0$.
4. Compare with LU-based inversion.
5. Time it vs solving $A\mathbf{x}=\mathbf{b}$.

#### Test Case

$$
A =
\begin{bmatrix}
1 & 2 & 3\
0 & 1 & 4\
5 & 6 & 0
\end{bmatrix}
\implies
A^{-1} =
\begin{bmatrix}
-24 & 18 & 5\
20 & -15 & -4\
-5 & 4 & 1
\end{bmatrix}
$$

Check: $A A^{-1} = I$

#### Complexity

- Time: $O(n^3)$
- Space: $O(n^2)$

Matrix inversion is a mirror: turning transformations back upon themselves, and teaching us that "solving" and "inverting" are two sides of the same operation.

### 557 Determinant by Elimination

The determinant of a matrix measures its scaling factor and invertibility.
Through Gaussian elimination, we can compute it efficiently by converting the matrix to upper triangular form, where the determinant equals the product of the diagonal entries, adjusted for any row swaps.

$$
\det(A) = (\text{sign}) \times \prod_{i=1}^{n} U_{ii}
$$

#### What Problem Are We Solving?

We want to compute $\det(A)$ without recursive expansion (which is $O(n!)$).
Elimination-based methods do it in $O(n^3)$, the same as LU decomposition.

The determinant tells us:

- If $\det(A)=0$: $A$ is singular (non-invertible)
- If $\det(A)\ne0$: $A$ is invertible
- The volume scaling factor of the linear transformation by $A$
- The orientation (positive = preserved, negative = flipped)

#### How Does It Work (Plain Language)

We perform elimination to form an upper triangular matrix $U$, keeping track of row swaps and scaling.

Steps:

1. Start with $A$
2. For each pivot row $i$:

   * Swap rows if pivot is zero (each swap flips determinant sign)
   * Eliminate below using row operations
     (adding multiples doesn't change determinant)
3. Once $U$ is upper triangular:
   $$
   \det(A) = (\pm 1) \times \prod_{i=1}^n U_{ii}
   $$

Only row swaps affect the sign.

#### Example

Let
$$
A =
\begin{bmatrix}
2 & 1 & 3\
4 & 2 & 6\
1 & -1 & 1
\end{bmatrix}
$$

Perform elimination:

- Row2 = Row2 - 2×Row1 → $[0, 0, 0]$
- Row3 = Row3 - ½×Row1 → $[0, -1.5, -0.5]$

Now $U =
\begin{bmatrix}
2 & 1 & 3\
0 & 0 & 0\
0 & -1.5 & -0.5
\end{bmatrix}$

A zero row appears → $\det(A)=0$.

Now let's test a non-singular case:

$$
B =
\begin{bmatrix}
2 & 1 & 1\
1 & 3 & 2\
1 & 0 & 0
\end{bmatrix}
$$

Eliminate:

- Row2 = Row2 - ½×Row1 → $[0, 2.5, 1.5]$
- Row3 = Row3 - ½×Row1 → $[0, -0.5, -0.5]$
- Row3 = Row3 + 0.2×Row2 → $[0, 0, -0.2]$

Now $U_{11}=2$, $U_{22}=2.5$, $U_{33}=-0.2$

$$
\det(B) = 2 \times 2.5 \times (-0.2) = -1
$$

#### Tiny Code (Easy Versions)

Python

```python
def determinant(A):
    n = len(A)
    A = [row[:] for row in A]
    det = 1
    sign = 1

    for i in range(n):
        # Pivoting
        if A[i][i] == 0:
            for j in range(i+1, n):
                if A[j][i] != 0:
                    A[i], A[j] = A[j], A[i]
                    sign *= -1
                    break
        pivot = A[i][i]
        if pivot == 0:
            return 0
        det *= pivot
        for j in range(i+1, n):
            factor = A[j][i] / pivot
            for k in range(i, n):
                A[j][k] -= factor * A[i][k]
    return sign * det

A = [[2,1,1],[1,3,2],[1,0,0]]
print("det =", determinant(A))
```

C

```c
#include <stdio.h>
#include <math.h>

#define N 3

double determinant(double A[N][N]) {
    double det = 1;
    int sign = 1;
    for (int i = 0; i < N; i++) {
        if (fabs(A[i][i]) < 1e-9) {
            int swap = -1;
            for (int j = i+1; j < N; j++) {
                if (fabs(A[j][i]) > 1e-9) { swap = j; break; }
            }
            if (swap == -1) return 0;
            for (int k = 0; k < N; k++) {
                double tmp = A[i][k];
                A[i][k] = A[swap][k];
                A[swap][k] = tmp;
            }
            sign *= -1;
        }
        det *= A[i][i];
        for (int j = i+1; j < N; j++) {
            double factor = A[j][i] / A[i][i];
            for (int k = i; k < N; k++)
                A[j][k] -= factor * A[i][k];
        }
    }
    return det * sign;
}

int main(void) {
    double A[N][N] = {{2,1,1},{1,3,2},{1,0,0}};
    printf("det = %.2f\n", determinant(A));
}
```

#### Why It Matters

- Invertibility check ($\det(A)\ne0$ means invertible)
- Volume scaling under linear transform
- Orientation detection (sign of determinant)
- Crucial for:

  * Jacobians in calculus
  * Change of variables
  * Eigenvalue computation

#### A Gentle Proof (Why It Works)

Gaussian elimination expresses $A$ as:

$$
A = L U
$$

with $L$ unit-lower-triangular.
Then:
$$
\det(A) = \det(L)\det(U) = 1 \times \prod_{i=1}^n U_{ii}
$$

Row swaps multiply $\det(A)$ by $-1$ each time.
Adding multiples of rows doesn't change $\det(A)$.

#### Try It Yourself

1. Compute $\det(A)$ by cofactor expansion and elimination, compare.
2. Track how sign changes with swaps.
3. Verify with a triangular matrix.
4. Observe determinant of singular matrices (should be 0).
5. Compare LU-based determinant.

#### Test Case

| Matrix $A$                                            | Determinant |
| ----------------------------------------------------- | ----------- |
| $\begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}$        | $-2$        |
| $\begin{bmatrix} 2 & 1 & 1 \\ 1 & 3 & 2 \\ 1 & 0 & 0 \end{bmatrix}$ | $-1$        |
| $\begin{bmatrix} 2 & 1 & 3 \\ 4 & 2 & 6 \\ 1 & -1 & 1 \end{bmatrix}$ | $0$         |


#### Complexity

- Time: $O(n^3)$
- Space: $O(1)$ (in-place)

Determinant by elimination turns algebraic chaos into structure, each pivot a volume factor, each swap a flip of orientation.

### 558 Rank of a Matrix

The rank of a matrix tells us how many independent rows or columns it has, in other words, the dimension of its image (column space). It's the bridge between linear independence, solvability, and dimension.

We can compute it efficiently using Gaussian elimination: transform the matrix to row echelon form (REF) and count the non-zero rows.

#### What Problem Are We Solving?

We want to measure how much information is in a matrix.
Rank answers questions like:

- Are the columns independent?
- Does $A\mathbf{x}=\mathbf{b}$ have a solution?
- What is the dimension of the span of rows/columns?

For $m\times n$ matrix $A$:
$$
\text{rank}(A) = \text{number of leading pivots in REF}
$$

#### How Does It Work (Plain Language)

1. Apply Gaussian elimination to reduce $A$ to row echelon form (REF)
2. Each non-zero row represents one pivot (independent direction)
3. Count the pivots → that's the rank

If full rank:

- $\text{rank}(A)=n$ → columns independent
- $\text{rank}(A)<n$ → some columns are dependent

In reduced row echelon form (RREF), the pivots are explicit 1s with zeros above and below.

#### Example

Let
$$
A =
\begin{bmatrix}
2 & 1 & 3\
4 & 2 & 6\
1 & -1 & 1
\end{bmatrix}
$$

Perform elimination:

Row2 = Row2 - 2×Row1 → $[0, 0, 0]$
Row3 = Row3 - ½×Row1 → $[0, -1.5, -0.5]$

Result:
$$
\begin{bmatrix}
2 & 1 & 3\
0 & -1.5 & -0.5\
0 & 0 & 0
\end{bmatrix}
$$

Two non-zero rows → rank = 2

So the rows span a 2D plane, not all of $\mathbb{R}^3$.

#### Tiny Code (Easy Versions)

Python

```python
def matrix_rank(A):
    n, m = len(A), len(A[0])
    A = [row[:] for row in A]
    rank = 0

    for col in range(m):
        # Find pivot
        pivot_row = None
        for row in range(rank, n):
            if abs(A[row][col]) > 1e-9:
                pivot_row = row
                break
        if pivot_row is None:
            continue

        # Swap to current rank row
        A[rank], A[pivot_row] = A[pivot_row], A[rank]

        # Normalize pivot row
        pivot = A[rank][col]
        A[rank] = [x / pivot for x in A[rank]]

        # Eliminate below
        for r in range(rank+1, n):
            factor = A[r][col]
            A[r] = [A[r][c] - factor*A[rank][c] for c in range(m)]

        rank += 1
    return rank

A = [[2,1,3],[4,2,6],[1,-1,1]]
print("rank =", matrix_rank(A))
```

C (Gaussian Elimination)

```c
#include <stdio.h>
#include <math.h>

#define N 3
#define M 3

int matrix_rank(double A[N][M]) {
    int rank = 0;
    for (int col = 0; col < M; col++) {
        int pivot = -1;
        for (int r = rank; r < N; r++) {
            if (fabs(A[r][col]) > 1e-9) { pivot = r; break; }
        }
        if (pivot == -1) continue;

        // Swap
        if (pivot != rank) {
            for (int c = 0; c < M; c++) {
                double tmp = A[rank][c];
                A[rank][c] = A[pivot][c];
                A[pivot][c] = tmp;
            }
        }

        // Normalize
        double div = A[rank][col];
        for (int c = 0; c < M; c++)
            A[rank][c] /= div;

        // Eliminate
        for (int r = rank + 1; r < N; r++) {
            double factor = A[r][col];
            for (int c = 0; c < M; c++)
                A[r][c] -= factor * A[rank][c];
        }

        rank++;
    }
    return rank;
}

int main(void) {
    double A[N][M] = {{2,1,3},{4,2,6},{1,-1,1}};
    printf("rank = %d\n", matrix_rank(A));
}
```

#### Why It Matters

- Dimension of span: rank = number of independent directions
- Solvability:

  * If $\text{rank}(A)=\text{rank}([A|\mathbf{b}])$, system is consistent
  * If $\text{rank}(A)<n$, infinite or no solutions
- Column space: rank = dimension of image
- Row space: same as column space dimension (rank = rank$^T$)

#### A Gentle Proof (Why It Works)

Row operations don't change the span of rows, hence don't change rank.
Once in echelon form, each non-zero row adds one linearly independent vector to the row space.

Thus:
$$
\text{rank}(A) = \text{number of pivot positions}
$$

#### Try It Yourself

1. Reduce a $3\times3$ matrix to REF.
2. Count pivot rows → rank.
3. Compare with column independence check.
4. Compute rank($A$) and rank($A^T$).
5. Test with singular matrices (rank < n).

#### Test Cases

| Matrix                                           | Rank |
| ------------------------------------------------ | ---- |
| $\begin{bmatrix} 1 & 2 \\ 3 & 6 \end{bmatrix}$   | 1    |
| $\begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}$   | 2    |
| $\begin{bmatrix} 2 & 1 & 3 \\ 4 & 2 & 6 \\ 1 & -1 & 1 \end{bmatrix}$ | 2    |


#### Complexity

- Time: $O(n^3)$
- Space: $O(n^2)$

The rank is the soul of a matrix, telling you how many truly independent voices sing in its rows and columns.

### 559 Eigenvalue Power Method

The Power Method is a simple iterative algorithm to approximate the dominant eigenvalue (the one with the largest magnitude) and its corresponding eigenvector of a matrix $A$.

It's one of the earliest and most intuitive ways to "listen" to the matrix and find its strongest direction, the one that stays steady under repeated transformation.

#### What Problem Are We Solving?

We want to find $\lambda_{\max}$ and $\mathbf{v}_{\max}$ such that:

$$
A \mathbf{v}*{\max} = \lambda*{\max} \mathbf{v}_{\max}
$$

When $A$ is large or sparse, solving the characteristic polynomial is infeasible.
The Power Method gives an iterative, low-memory approximation, crucial in numerical linear algebra, PageRank, and PCA.

#### How Does It Work (Plain Language)

1. Start with a random vector $\mathbf{x}_0$
2. Repeatedly apply $A$: $\mathbf{x}_{k+1} = A \mathbf{x}_k$
3. Normalize each step to prevent overflow
4. When $\mathbf{x}_k$ stabilizes, it aligns with the dominant eigenvector
5. The corresponding eigenvalue is approximated by the Rayleigh quotient:

$$
\lambda_k \approx \frac{\mathbf{x}_k^T A \mathbf{x}_k}{\mathbf{x}_k^T \mathbf{x}_k}
$$

Because $A^k \mathbf{x}*0$ amplifies the component in the direction of $\mathbf{v}*{\max}$.

#### Algorithm (Step-by-Step)

Given $A$ and tolerance $\varepsilon$:

1. Choose $\mathbf{x}_0$ (non-zero vector)
2. Repeat:

   * $\mathbf{y} = A \mathbf{x}$
   * $\lambda = \max(|y_i|)$ or $\mathbf{x}^T A \mathbf{x}$
   * $\mathbf{x} = \mathbf{y} / \lambda$
3. Stop when $|\mathbf{x}_{k+1} - \mathbf{x}_k| < \varepsilon$

Return $\lambda, \mathbf{x}$.

Convergence requires that $A$ has a unique largest eigenvalue.

#### Example

Let
$$
A =
\begin{bmatrix}
2 & 1\
1 & 3
\end{bmatrix}
$$

Start
$$
\mathbf{x}_0 = \begin{bmatrix} 1 \\ 1 \end{bmatrix}
$$

1. $\mathbf{y} = A\mathbf{x}_0 = [3, 4]^T$, normalize: $\mathbf{x}_1 = [0.6, 0.8]$
2. $\mathbf{y} = A\mathbf{x}_1 = [2.0, 3.0]^T$, normalize: $\mathbf{x}_2 = [0.5547, 0.8321]$
3. Repeat, $\mathbf{x}_k$ converges to eigenvector $[0.447, 0.894]$
4. $\lambda \approx 3.618$ (dominant eigenvalue)

#### Tiny Code (Easy Versions)

Python

```python
import numpy as np

def power_method(A, tol=1e-6, max_iter=1000):
    n = len(A)
    x = np.ones(n)
    lambda_old = 0

    for _ in range(max_iter):
        y = np.dot(A, x)
        lambda_new = np.max(np.abs(y))
        x = y / lambda_new
        if np.linalg.norm(x - y/np.linalg.norm(y)) < tol:
            break
        lambda_old = lambda_new
    return lambda_new, x / np.linalg.norm(x)

A = np.array([[2,1],[1,3]], dtype=float)
lam, v = power_method(A)
print("lambda ≈", lam)
print("v ≈", v)
```

C

```c
#include <stdio.h>
#include <math.h>

#define N 2
#define MAX_ITER 1000
#define TOL 1e-6

void power_method(double A[N][N], double v[N]) {
    double y[N], lambda = 0, lambda_old;
    for (int i=0;i<N;i++) v[i]=1.0;

    for (int iter=0;iter<MAX_ITER;iter++) {
        // y = A * v
        for (int i=0;i<N;i++) {
            y[i]=0;
            for (int j=0;j<N;j++) y[i]+=A[i][j]*v[j];
        }
        // estimate eigenvalue
        lambda_old = lambda;
        lambda = fabs(y[0]);
        for (int i=1;i<N;i++) if (fabs(y[i])>lambda) lambda=fabs(y[i]);
        // normalize
        for (int i=0;i<N;i++) v[i]=y[i]/lambda;

        // check convergence
        double diff=0;
        for (int i=0;i<N;i++) diff+=fabs(y[i]-lambda*v[i]);
        if (diff<TOL) break;
    }

    printf("lambda ≈ %.6f\n", lambda);
    printf("v ≈ [%.3f, %.3f]\n", v[0], v[1]);
}

int main(void) {
    double A[N][N] = {{2,1},{1,3}}, v[N];
    power_method(A, v);
}
```

#### Why It Matters

- Finds dominant eigenvalue/vector
- Works with large sparse matrices
- Foundation for:

  * PageRank (Google)
  * PCA (Principal Component Analysis)
  * Spectral methods
  * Markov chains steady states

#### A Gentle Proof (Why It Works)

If $A$ has eigen-decomposition
$$
A = V \Lambda V^{-1}
$$
and $\mathbf{x}_0 = c_1\mathbf{v}_1 + \dots + c_n\mathbf{v}_n$,

then
$$
A^k \mathbf{x}_0 = c_1\lambda_1^k\mathbf{v}_1 + \cdots + c_n\lambda_n^k\mathbf{v}_n
$$

As $k \to \infty$, $\lambda_1^k$ dominates, so direction $\to \mathbf{v}_1$.
Normalization removes scale, leaving the dominant eigenvector.

#### Try It Yourself

1. Apply to a $3\times3$ symmetric matrix.
2. Compare result with `numpy.linalg.eig`.
3. Try on a diagonal matrix, verify it finds largest diagonal.
4. Observe divergence if eigenvalues are equal in magnitude.
5. Modify to inverse power method to find smallest eigenvalue.

#### Test Case

$$
A =
\begin{bmatrix}
4 & 1\
2 & 3
\end{bmatrix},
\quad
\lambda_{\max} \approx 4.561,
\quad
\mathbf{v}_{\max} \approx
\begin{bmatrix}
0.788\
0.615
\end{bmatrix}
$$

#### Complexity

- Time: $O(kn^2)$ (for $k$ iterations)
- Space: $O(n^2)$

The Power Method is the simplest window into eigenvalues, each iteration aligns your vector more closely with the matrix's strongest echo.

### 560 Singular Value Decomposition (SVD)

The Singular Value Decomposition (SVD) is one of the most powerful and universal factorizations in linear algebra. It expresses any matrix$A$ (square or rectangular) as a product of three special matrices:

$$
A = U \Sigma V^T
$$

-$U$: orthogonal matrix of left singular vectors
-$\Sigma$: diagonal matrix of singular values (non-negative)
-$V$: orthogonal matrix of right singular vectors

SVD generalizes the eigendecomposition, works for any matrix (even non-square), and reveals deep structure, from geometry to data compression.

#### What Problem Are We Solving?

We want to decompose$A$ into simpler, interpretable parts:

- Directions of stretching (via$V$)
- Amount of stretching (via$\Sigma$)
- Resulting orthogonal directions (via$U$)

SVD is used to:

- Compute rank, null space, and range
- Perform dimensionality reduction (PCA)
- Solve least squares problems
- Compute pseudoinverse
- Perform noise reduction in signals and images

#### How Does It Work (Plain Language)

For any$A \in \mathbb{R}^{m \times n}$:

1. Compute$A^T A$ (symmetric and positive semidefinite)
2. Find eigenvalues$\lambda_i$ and eigenvectors$v_i$
3. Singular values$\sigma_i = \sqrt{\lambda_i}$
4. Form$V = [v_1, \ldots, v_n]$
5. Form$U = \frac{1}{\sigma_i} A v_i$ for non-zero$\sigma_i$
6. Assemble$\Sigma$ with$\sigma_i$ on the diagonal

Result:
$$
A = U \Sigma V^T
$$

If$A$ is$m \times n$:

-$U$:$m \times m$
-$\Sigma$:$m \times n$
-$V$:$n \times n$

#### Example

Let
$$
A=
\begin{bmatrix}
3 & 1\\
1 & 3
\end{bmatrix}.
$$

1. Compute $A^{\mathsf T}A$:
$$
A^{\mathsf T}A
=
\begin{bmatrix}
3 & 1\\
1 & 3
\end{bmatrix}
\begin{bmatrix}
3 & 1\\
1 & 3
\end{bmatrix}
=
\begin{bmatrix}
10 & 6\\
6 & 10
\end{bmatrix}.
$$

2. Eigenvalues: $16,\,4$  
   Singular values: $\sigma_1=4,\ \sigma_2=2$.

3. Right singular vectors:
$$
v_1=\frac{1}{\sqrt{2}}\begin{bmatrix}1\\1\end{bmatrix}, \quad
v_2=\frac{1}{\sqrt{2}}\begin{bmatrix}1\\-1\end{bmatrix}, \quad
V=[v_1,v_2].
$$

4. Compute $u_i=\frac{1}{\sigma_i}A v_i$:
$$
u_1=\frac{1}{4}A v_1
=\frac{1}{4}
\begin{bmatrix}3 & 1\\1 & 3\end{bmatrix}
\frac{1}{\sqrt{2}}
\begin{bmatrix}1\\1\end{bmatrix}
=\frac{1}{\sqrt{2}}\begin{bmatrix}1\\1\end{bmatrix},
$$
$$
u_2=\frac{1}{2}A v_2
=\frac{1}{2}
\begin{bmatrix}3 & 1\\1 & 3\end{bmatrix}
\frac{1}{\sqrt{2}}
\begin{bmatrix}1\\-1\end{bmatrix}
=\frac{1}{\sqrt{2}}\begin{bmatrix}1\\-1\end{bmatrix}.
$$

Thus
$$
U=\frac{1}{\sqrt{2}}
\begin{bmatrix}
1 & 1\\
1 & -1
\end{bmatrix}, \quad
\Sigma=
\begin{bmatrix}
4 & 0\\
0 & 2
\end{bmatrix}, \quad
V=U.
$$

Check: $A = U\,\Sigma\,V^{\mathsf T}$.


#### Tiny Code (Easy Versions)

Python (NumPy built-in)

```python
import numpy as np

A = np.array([[3, 1], [1, 3]], dtype=float)
U, S, Vt = np.linalg.svd(A)

print("U =\n", U)
print("S =\n", np.diag(S))
print("V^T =\n", Vt)
```

Python (Manual Approximation for 2×2)

```python
import numpy as np

def svd_2x2(A):
    ATA = A.T @ A
    eigvals, V = np.linalg.eig(ATA)
    idx = np.argsort(-eigvals)
    eigvals, V = eigvals[idx], V[:, idx]
    S = np.sqrt(eigvals)
    U = (A @ V) / S
    return U, S, V

A = np.array([[3, 1], [1, 3]], float)
U, S, V = svd_2x2(A)
print("U =", U)
print("S =", S)
print("V =", V)
```

C (Conceptual Skeleton)

```c
#include <stdio.h>
#include <math.h>

// For 2x2 SVD demonstration
void svd_2x2(double A[2][2]) {
    double a=A[0][0], b=A[0][1], c=A[1][0], d=A[1][1];
    double ATA[2][2] = {
        {a*a + c*c, a*b + c*d},
        {a*b + c*d, b*b + d*d}
    };

    double trace = ATA[0][0] + ATA[1][1];
    double det = ATA[0][0]*ATA[1][1] - ATA[0][1]*ATA[1][0];
    double s = sqrt(trace*trace/4 - det);

    double sigma1 = sqrt(trace/2 + s);
    double sigma2 = sqrt(trace/2 - s);

    printf("Singular values: %.3f, %.3f\n", sigma1, sigma2);
}

int main(void) {
    double A[2][2] = {{3,1},{1,3}};
    svd_2x2(A);
}
```

#### Why It Matters

- Universal factorization: works for any matrix
- Geometry: describes stretching and rotation
- Data science: PCA, low-rank approximation
- Numerical analysis: stable pseudoinverse
- Signal processing: denoising, compression

SVD powers tools like Latent Semantic Analysis (LSA) and Truncated SVD for dimensionality reduction.

#### A Gentle Proof (Why It Works)

From$A^T A = V \Lambda V^T$, where$\Lambda$ is diagonal with eigenvalues$\lambda_i \ge 0$.
Define$\Sigma = \sqrt{\Lambda}$.
Then:

$$
A = (A V \Sigma^{-1}) \Sigma V^T
$$

Set$U = A V \Sigma^{-1}$, which is orthogonal.
Thus$A = U \Sigma V^T$.

#### Try It Yourself

1. Decompose a$3\times2$ matrix manually.
2. Verify$U^T U = I$,$V^T V = I$.
3. Compare singular values with$\sqrt{\text{eigenvalues of } A^T A}$.
4. Use SVD to find rank (count of non-zero singular values).
5. Reconstruct$A$ from top-1 singular value for compression.

#### Test Case

$$
A =
\begin{bmatrix}
3 & 1\
1 & 3
\end{bmatrix},
\quad
U =
\frac{1}{\sqrt{2}}
\begin{bmatrix}
1 & 1\
1 & -1
\end{bmatrix},
\quad
\Sigma =
\begin{bmatrix}
4 & 0\
0 & 2
\end{bmatrix},
\quad
V = U
$$

Check:$A = U \Sigma V^T$

#### Complexity

- Time:$O(mn^2)$ for dense matrices
- Space:$O(mn)$

SVD is the master key of linear algebra, unlocking rank, eigenstructure, geometry, and data all at once.

# Section 57. FFT and NTT (Fast Transforms)

### 561 Discrete Fourier Transform (DFT)

The Discrete Fourier Transform (DFT) is the mathematical bridge between time (or spatial) domain and frequency domain. It decomposes a finite sequence of numbers into a sum of complex sinusoids, revealing the hidden frequencies that make up the signal.

The DFT is the foundation for signal processing, audio compression, image filtering, and polynomial multiplication. It's beautiful, exact, and fundamental.

#### What Problem Are We Solving?

Given a sequence of$n$ complex numbers:

$$
x = (x_0, x_1, \ldots, x_{n-1})
$$

we want to compute a new sequence$X = (X_0, X_1, \ldots, X_{n-1})$ that describes how much of each frequency is present in$x$.

The transform is defined as:

$$
X_k = \sum_{j=0}^{n-1} x_j \cdot e^{-2\pi i \frac{jk}{n}}
\quad \text{for } k = 0, 1, \ldots, n-1
$$

Each$X_k$ measures the amplitude of the complex sinusoid with frequency$k/n$.

#### How Does It Work (Plain Language)

Think of your input sequence as a chord played on a piano, a mix of multiple frequencies.
The DFT "listens" to the signal and tells you which notes (frequencies) are present and how strong they are.

At its core, it multiplies the signal by complex sinusoids$e^{-2\pi i jk/n}$, summing the results to find how strongly each sinusoid contributes.

#### Example

Let$n = 4$, and$x = (1, 2, 3, 4)$

Then:

$$
X_k = \sum_{j=0}^{3} x_j \cdot e^{-2\pi i \frac{jk}{4}}
$$

Compute each:

-$X_0 = 1 + 2 + 3 + 4 = 10$
-$X_1 = 1 + 2i - 3 - 4i = -2 - 2i$
-$X_2 = 1 - 2 + 3 - 4 = -2$
-$X_3 = 1 - 2i - 3 + 4i = -2 + 2i$

So the DFT is:

$$
X = [10, -2-2i, -2, -2+2i]
$$

#### Inverse DFT

To reconstruct the original sequence from its frequencies:

$$
x_j = \frac{1}{n} \sum_{k=0}^{n-1} X_k \cdot e^{2\pi i \frac{jk}{n}}
$$

The forward and inverse transform form a perfect pair, no information is lost.

#### Tiny Code (Easy Versions)

Python (Naive DFT)

```python
import cmath

def dft(x):
    n = len(x)
    X = []
    for k in range(n):
        s = 0
        for j in range(n):
            angle = -2j * cmath.pi * j * k / n
            s += x[j] * cmath.exp(angle)
        X.append(s)
    return X

# Example
x = [1, 2, 3, 4]
X = dft(x)
print("DFT:", X)
```

Python (Inverse DFT)

```python
def idft(X):
    n = len(X)
    x = []
    for j in range(n):
        s = 0
        for k in range(n):
            angle = 2j * cmath.pi * j * k / n
            s += X[k] * cmath.exp(angle)
        x.append(s / n)
    return x
```

C (Naive Implementation)

```c
#include <stdio.h>
#include <complex.h>
#include <math.h>

#define PI 3.14159265358979323846

void dft(int n, double complex x[], double complex X[]) {
    for (int k = 0; k < n; k++) {
        X[k] = 0;
        for (int j = 0; j < n; j++) {
            double angle = -2.0 * PI * j * k / n;
            X[k] += x[j] * cexp(I * angle);
        }
    }
}

int main(void) {
    int n = 4;
    double complex x[4] = {1, 2, 3, 4};
    double complex X[4];
    dft(n, x, X);

    for (int k = 0; k < n; k++)
        printf("X[%d] = %.2f + %.2fi\n", k, creal(X[k]), cimag(X[k]));
}
```

#### Why It Matters

- Converts time-domain signals into frequency-domain insights
- Enables filtering, compression, and pattern detection
- Fundamental in signal processing, audio/video, cryptography, machine learning, and FFT-based algorithms

#### A Gentle Proof (Why It Works)

The DFT uses the orthogonality of complex exponentials:

$$
\sum_{j=0}^{n-1} e^{-2\pi i (k-l) j / n} =
\begin{cases}
n, & \text{if } k = l, \\
0, & \text{if } k \ne l.
\end{cases}
$$


This property ensures we can isolate each frequency component uniquely and invert the transform exactly.

#### Try It Yourself

1. Compute DFT of$[1, 0, 1, 0]$.
2. Verify that applying IDFT brings back the original sequence.
3. Plot real and imaginary parts of$X_k$.
4. Observe how$X_0$ represents the average of input values.
5. Compare runtime with FFT for large$n$.

#### Test Cases

| Input$x$  | Output$X$           |
| ------------ | ---------------------- |
| [1, 1, 1, 1] | [4, 0, 0, 0]           |
| [1, 0, 1, 0] | [2, 0, 2, 0]           |
| [1, 2, 3, 4] | [10, -2-2i, -2, -2+2i] |

#### Complexity

- Time:$O(n^2)$
- Space:$O(n)$

DFT is the mathematical microscope that reveals the hidden harmonies inside data, every signal becomes a song of frequencies.

### 562 Fast Fourier Transform (FFT)

The Fast Fourier Transform (FFT) is one of the most important algorithms in computational mathematics. It takes the Discrete Fourier Transform (DFT), originally an$O(n^2)$ operation, and reduces it to$O(n \log n)$ by exploiting symmetry and recursion.
It's the beating heart behind digital signal processing, convolution, polynomial multiplication, and spectral analysis.

#### What Problem Are We Solving?

We want to compute the same transformation as the DFT:

$$
X_k = \sum_{j=0}^{n-1} x_j \cdot e^{-2\pi i \frac{jk}{n}}
$$

but faster.

A direct implementation loops over both$j$ and$k$, requiring$n^2$ operations.
FFT cleverly reorganizes computation by dividing the sequence into even and odd parts, halving the work each time.

#### How Does It Work (Plain Language)

FFT uses a divide-and-conquer idea:

1. Split the sequence into even and odd indexed elements.
2. Compute the DFT of each half (recursively).
3. Combine results using the symmetry of complex roots of unity.

This works best when$n$ is a power of 2, because we can split evenly each time until reaching single elements.

Mathematically:

Let$n = 2m$. Then:

$$
X_k = E_k + \omega_n^k O_k \
X_{k+m} = E_k - \omega_n^k O_k
$$

where:

-$E_k$ is the DFT of even-indexed terms,
-$O_k$ is the DFT of odd-indexed terms,
-$\omega_n = e^{-2\pi i / n}$ is the primitive root of unity.

#### Example

Let$n = 4$,$x = [1, 2, 3, 4]$

Split:

- Evens:$[1, 3]$
- Odds:$[2, 4]$

Compute DFT of each (size 2):

-$E = [4, -2]$
-$O = [6, -2]$

Then combine:

-$X_0 = E_0 + O_0 = 10$
-$X_1 = E_1 + \omega_4^1 O_1 = -2 + i(-2) = -2 - 2i$
-$X_2 = E_0 - O_0 = -2$
-$X_3 = E_1 - \omega_4^1 O_1 = -2 + 2i$

So$X = [10, -2-2i, -2, -2+2i]$, same as DFT but computed faster.

#### Tiny Code (Recursive FFT)

Python (Cooley–Tukey FFT)

```python
import cmath

def fft(x):
    n = len(x)
    if n == 1:
        return x
    w_n = cmath.exp(-2j * cmath.pi / n)
    w = 1
    x_even = fft(x[0::2])
    x_odd = fft(x[1::2])
    X = [0] * n
    for k in range(n // 2):
        t = w * x_odd[k]
        X[k] = x_even[k] + t
        X[k + n // 2] = x_even[k] - t
        w *= w_n
    return X

# Example
x = [1, 2, 3, 4]
X = fft(x)
print("FFT:", X)
```

C (Recursive FFT)

```c
#include <stdio.h>
#include <complex.h>
#include <math.h>

#define PI 3.14159265358979323846

void fft(int n, double complex *x) {
    if (n <= 1) return;

    double complex even[n/2], odd[n/2];
    for (int i = 0; i < n/2; i++) {
        even[i] = x[2*i];
        odd[i] = x[2*i + 1];
    }

    fft(n/2, even);
    fft(n/2, odd);

    for (int k = 0; k < n/2; k++) {
        double complex w = cexp(-2.0 * I * PI * k / n);
        double complex t = w * odd[k];
        x[k] = even[k] + t;
        x[k + n/2] = even[k] - t;
    }
}

int main() {
    double complex x[4] = {1, 2, 3, 4};
    fft(4, x);
    for (int i = 0; i < 4; i++)
        printf("X[%d] = %.2f + %.2fi\n", i, creal(x[i]), cimag(x[i]));
}
```

#### Why It Matters

- Transforms signals, images, and time-series to frequency domain in milliseconds.
- Foundation for digital filters, convolution, and compression.
- Core in machine learning (spectral methods) and physics simulations.
- Enables polynomial multiplication in$O(n \log n)$.

#### A Gentle Proof (Why It Works)

The trick is recognizing that the DFT matrix has repeated patterns due to powers of$\omega_n$:

$$
W_n =
\begin{bmatrix}
1 & 1 & 1 & 1 \\
1 & \omega_n & \omega_n^{2} & \omega_n^{3} \\
1 & \omega_n^{2} & \omega_n^{4} & \omega_n^{6} \\
1 & \omega_n^{3} & \omega_n^{6} & \omega_n^{9}
\end{bmatrix}
$$


By splitting into even and odd columns, we reuse computations recursively.
This halves the problem size at each level, leading to$\log n$ recursion depth and$n$ work per level.

Total:$O(n \log n)$

#### Try It Yourself

1. Implement FFT for$n = 8$ with random values.
2. Compare runtime with naive DFT.
3. Plot runtime vs$n$ (log-log scale).
4. Apply FFT to a sine wave and visualize peaks in frequency.
5. Multiply two polynomials using FFT convolution.

#### Test Cases

| Input$x$  | FFT$X$              |
| ------------ | ---------------------- |
| [1, 1, 1, 1] | [4, 0, 0, 0]           |
| [1, 2, 3, 4] | [10, -2-2i, -2, -2+2i] |

#### Complexity

- Time:$O(n \log n)$
- Space:$O(n)$ (recursive) or$O(1)$ (iterative)

FFT turned a quadratic computation into a nearly linear one, a leap so profound it reshaped science, engineering, and computing forever.

### 563 Cooley–Tukey FFT

The Cooley–Tukey algorithm is the most widely used implementation of the Fast Fourier Transform (FFT). It's the algorithm that made the FFT practical, elegant, and efficient—reducing the $O(n^2)$ Discrete Fourier Transform to $O(n\log n)$ by recursively decomposing it into smaller transforms.

This method leverages the divide-and-conquer principle and the symmetry of complex roots of unity, making it the backbone of nearly all FFT libraries.

#### What Problem Are We Solving?

We want to compute the Discrete Fourier Transform efficiently:

$$
X_k=\sum_{j=0}^{n-1}x_j\cdot e^{-2\pi i\frac{jk}{n}}
$$

Instead of computing all terms directly, we split the sequence into smaller pieces and reuse results—dramatically cutting down redundant computation.

#### How Does It Work (Plain Language)

Cooley–Tukey works by recursively dividing the DFT into smaller DFTs:

1. Split the input sequence into even and odd indexed elements:

   * $x_{\text{even}}=[x_0,x_2,x_4,\ldots]$
   * $x_{\text{odd}}=[x_1,x_3,x_5,\ldots]$

2. Compute two smaller DFTs of size $n/2$:

   * $E_k=\text{DFT}(x_{\text{even}})$
   * $O_k=\text{DFT}(x_{\text{odd}})$

3. Combine them using twiddle factors:

   * $\omega_n=e^{-2\pi i/n}$

The combination step:

$$
X_k=E_k+\omega_n^kO_k
$$

$$
X_{k+n/2}=E_k-\omega_n^kO_k
$$

#### Example ($n=8$)

Suppose $x=[x_0,x_1,\ldots,x_7]$

1. Split into evens and odds:

   * $[x_0,x_2,x_4,x_6]$
   * $[x_1,x_3,x_5,x_7]$

2. Recursively compute 4-point FFTs for each.

3. Combine:

   * $X_k=E_k+\omega_8^kO_k$
   * $X_{k+4}=E_k-\omega_8^kO_k$

Each recursion layer halves the problem, and there are $\log_2 n$ layers total.

#### Tiny Code (Recursive Implementation)

Python

```python
import cmath

def cooley_tukey_fft(x):
    n = len(x)
    if n == 1:
        return x
    w_n = cmath.exp(-2j * cmath.pi / n)
    w = 1
    X_even = cooley_tukey_fft(x[0::2])
    X_odd = cooley_tukey_fft(x[1::2])
    X = [0] * n
    for k in range(n // 2):
        t = w * X_odd[k]
        X[k] = X_even[k] + t
        X[k + n // 2] = X_even[k] - t
        w *= w_n
    return X

# Example
x = [1, 2, 3, 4, 5, 6, 7, 8]
X = cooley_tukey_fft(x)
print("FFT:", X)
```

C (Recursive)

```c
#include <stdio.h>
#include <complex.h>
#include <math.h>

#define PI 3.14159265358979323846

void fft(int n, double complex *x) {
    if (n <= 1) return;

    double complex even[n/2], odd[n/2];
    for (int i = 0; i < n/2; i++) {
        even[i] = x[2*i];
        odd[i] = x[2*i + 1];
    }

    fft(n/2, even);
    fft(n/2, odd);

    for (int k = 0; k < n/2; k++) {
        double complex w = cexp(-2.0 * I * PI * k / n);
        double complex t = w * odd[k];
        x[k] = even[k] + t;
        x[k + n/2] = even[k] - t;
    }
}

int main() {
    double complex x[8] = {1,2,3,4,5,6,7,8};
    fft(8, x);
    for (int i = 0; i < 8; i++)
        printf("X[%d] = %.2f + %.2fi\n", i, creal(x[i]), cimag(x[i]));
}
```

#### Why It Matters

- Reduces runtime from $O(n^2)$ to $O(n\log n)$
- Basis for audio/video processing, image transforms, and DSP
- Essential for fast polynomial multiplication, signal filtering, and spectral analysis
- Universally adopted in FFT libraries (FFTW, NumPy, cuFFT)

#### A Gentle Proof (Why It Works)

The DFT matrix $W_n$ is built from complex roots of unity:

$$
W_n =
\begin{bmatrix}
1 & 1 & 1 & \dots & 1 \\
1 & \omega_n & \omega_n^2 & \dots & \omega_n^{n-1} \\
1 & \omega_n^2 & \omega_n^4 & \dots & \omega_n^{2(n-1)} \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
1 & \omega_n^{n-1} & \omega_n^{2(n-1)} & \dots & \omega_n^{(n-1)^2}
\end{bmatrix}
$$

where $\omega_n = e^{-2\pi i / n}$ is the $n$th root of unity.


By reordering and grouping even/odd columns, we get two smaller $W_{n/2}$ blocks, multiplied by twiddle factors.
Thus, each recursive step halves the problem size—leading to $\log_2 n$ levels of computation.

#### Try It Yourself

1. Compute FFT for $[1,2,3,4,5,6,7,8]$ by hand (2 levels).
2. Verify symmetry $X_{k+n/2}=E_k-\omega_n^kO_k$.
3. Compare runtime with DFT for $n=1024$.
4. Visualize recursion tree.
5. Test on sine wave input—check peaks in frequency domain.

#### Test Cases

| Input             | Output (Magnitude)                                      |
| ----------------- | ------------------------------------------------------- |
| [1,1,1,1,1,1,1,1] | [8,0,0,0,0,0,0,0]                                       |
| [1,2,3,4,5,6,7,8] | [36,-4+9.66i,-4+4i,-4+1.66i,-4,-4-1.66i,-4-4i,-4-9.66i] |

#### Complexity

- Time: $O(n\log n)$
- Space: $O(n)$ (recursion) or $O(1)$ (iterative version)

The Cooley–Tukey FFT is more than a clever trick—it's a profound insight into symmetry and structure, transforming the way we compute, analyze, and understand signals across every field of science and engineering.

### 564 Iterative FFT

The Iterative FFT is an efficient, non-recursive implementation of the Cooley–Tukey Fast Fourier Transform. Instead of recursive calls, it computes the transform in-place, reordering elements using bit-reversal permutation and iteratively combining results in layers.

This approach is widely used in high-performance FFT libraries (like FFTW or cuFFT) because it's cache-friendly, stack-safe, and parallelizable.

#### What Problem Are We Solving?

Recursive FFT is elegant, but function calls and memory allocation overheads can slow it down. The iterative FFT eliminates recursion, doing all the same computations directly in loops.

We still want to compute:

$$
X_k=\sum_{j=0}^{n-1}x_j\cdot e^{-2\pi i\frac{jk}{n}}
$$

but we'll reuse the divide-and-conquer pattern iteratively.

#### How Does It Work (Plain Language)

The iterative FFT runs in logarithmic stages, each stage doubling the subproblem size:

1. Bit-Reversal Permutation
   Reorder the input so indices follow bit-reversed order (mirror the binary digits).
   Example: for $n=8$, indices $[0,1,2,3,4,5,6,7]$ become $[0,4,2,6,1,5,3,7]$.

2. Butterfly Computation
   Combine pairs of elements (like wings of a butterfly) using twiddle factors:
   $$
   t=\omega_n^k\cdot X_{\text{odd}}
   $$
   Then update:
   $$
   X_{\text{even}}'=X_{\text{even}}+t
   $$
   $$
   X_{\text{odd}}'=X_{\text{even}}-t
   $$

3. Iterate over stages
   Each stage merges smaller DFTs into larger ones, doubling the block size until full length.

By the end, $x$ is transformed in-place to $X$.

#### Example ($n=8$)

1. Input: $x=[x_0,x_1,x_2,x_3,x_4,x_5,x_6,x_7]$
2. Reorder via bit-reversal: $[x_0,x_4,x_2,x_6,x_1,x_5,x_3,x_7]$
3. Stage 1: combine pairs $(x_0,x_1),(x_2,x_3),\ldots$
4. Stage 2: combine blocks of size 4
5. Stage 3: combine blocks of size 8

Each stage multiplies by twiddle factors $\omega_n^k=e^{-2\pi i k/n}$, performing butterflies iteratively.

#### Tiny Code (Iterative FFT)

Python (In-Place Iterative FFT)

```python
import cmath

def bit_reverse(x):
    n = len(x)
    j = 0
    for i in range(1, n):
        bit = n >> 1
        while j & bit:
            j ^= bit
            bit >>= 1
        j ^= bit
        if i < j:
            x[i], x[j] = x[j], x[i]

def iterative_fft(x):
    n = len(x)
    bit_reverse(x)
    size = 2
    while size <= n:
        w_m = cmath.exp(-2j * cmath.pi / size)
        for k in range(0, n, size):
            w = 1
            for j in range(size // 2):
                t = w * x[k + j + size // 2]
                u = x[k + j]
                x[k + j] = u + t
                x[k + j + size // 2] = u - t
                w *= w_m
        size *= 2
    return x

# Example
x = [1, 2, 3, 4, 5, 6, 7, 8]
X = iterative_fft([complex(a, 0) for a in x])
print("FFT:", X)
```

C (In-Place Iterative FFT)

```c
#include <stdio.h>
#include <complex.h>
#include <math.h>

#define PI 3.14159265358979323846

void bit_reverse(double complex *x, int n) {
    int j = 0;
    for (int i = 1; i < n; i++) {
        int bit = n >> 1;
        while (j & bit) { j ^= bit; bit >>= 1; }
        j ^= bit;
        if (i < j) {
            double complex temp = x[i];
            x[i] = x[j];
            x[j] = temp;
        }
    }
}

void iterative_fft(double complex *x, int n) {
    bit_reverse(x, n);
    for (int size = 2; size <= n; size <<= 1) {
        double angle = -2 * PI / size;
        double complex w_m = cos(angle) + I * sin(angle);
        for (int k = 0; k < n; k += size) {
            double complex w = 1;
            for (int j = 0; j < size/2; j++) {
                double complex t = w * x[k + j + size/2];
                double complex u = x[k + j];
                x[k + j] = u + t;
                x[k + j + size/2] = u - t;
                w *= w_m;
            }
        }
    }
}
```

#### Why It Matters

- Removes recursion overhead, faster in practice
- In-place, requires no extra arrays
- Foundation for GPU FFTs, DSP hardware, and real-time systems
- Enables batch FFTs efficiently (parallelizable loops)

#### A Gentle Proof (Why It Works)

The iterative FFT simply traverses the same recursion tree bottom-up.
The bit-reversal ensures inputs line up as they would in recursive order.
Each stage merges DFTs of size $2^k$ into $2^{k+1}$, using identical butterfly equations:

$$
X_k=E_k+\omega_n^kO_k,\quad X_{k+n/2}=E_k-\omega_n^kO_k
$$

By repeating for $\log_2 n$ stages, we compute the full FFT.

#### Try It Yourself

1. Apply iterative FFT to $[1,2,3,4,5,6,7,8]$.
2. Print array after bit-reversal, confirm order.
3. Compare result to recursive FFT.
4. Plot runtime for $n=2^k$.
5. Extend to inverse FFT (change sign in exponent, divide by $n$).

#### Test Cases

| Input             | Output (Magnitude)                                      |
| ----------------- | ------------------------------------------------------- |
| [1,1,1,1,1,1,1,1] | [8,0,0,0,0,0,0,0]                                       |
| [1,2,3,4,5,6,7,8] | [36,-4+9.66i,-4+4i,-4+1.66i,-4,-4-1.66i,-4-4i,-4-9.66i] |

#### Complexity

- Time: $O(n\log n)$
- Space: $O(1)$ (in-place)

The iterative FFT replaces elegant recursion with raw efficiency—bringing the same mathematical beauty into tight, blazing-fast loops.

### 565 Inverse FFT (IFFT)

The Inverse Fast Fourier Transform (IFFT) is the mirror image of the FFT. It takes a signal from the frequency domain back to the time domain, perfectly reconstructing the original sequence.
Where FFT decomposes a signal into frequencies, IFFT reassembles it from those same components, a complete round-trip transformation.

#### What Problem Are We Solving?

Given the Fourier coefficients $X_0, X_1, \ldots, X_{n-1}$, we want to reconstruct the original signal $x_0, x_1, \ldots, x_{n-1}$.

The definition of the inverse DFT is:

$$
x_j=\frac{1}{n}\sum_{k=0}^{n-1}X_k\cdot e^{2\pi i\frac{jk}{n}}
$$

IFFT allows us to recover time-domain data after performing operations (like filtering or convolution) in the frequency domain.

#### How Does It Work (Plain Language)

IFFT works just like FFT, same butterfly structure, same recursion or iteration, but with conjugated twiddle factors and a final scaling by $1/n$.

To compute the IFFT:

1. Take the complex conjugate of all frequency components.
2. Run a forward FFT.
3. Take the complex conjugate again.
4. Divide every result by $n$.

This uses the fact that:
$$
\text{IFFT}(X)=\frac{1}{n}\cdot\overline{\text{FFT}(\overline{X})}
$$

#### Example

Let $X=[10,-2-2i,-2,-2+2i]$

1. Conjugate: $[10,-2+2i,-2,-2-2i]$
2. FFT of conjugated values $\to [4,8,12,16]$
3. Conjugate again: $[4,8,12,16]$
4. Divide by $n=4$: $[1,2,3,4]$

Recovered original: $x=[1,2,3,4]$

Perfect reconstruction confirmed.

#### Tiny Code (Easy Versions)

Python (IFFT via FFT)

```python
import cmath

def fft(x):
    n = len(x)
    if n == 1:
        return x
    w_n = cmath.exp(-2j * cmath.pi / n)
    w = 1
    X_even = fft(x[0::2])
    X_odd = fft(x[1::2])
    X = [0] * n
    for k in range(n // 2):
        t = w * X_odd[k]
        X[k] = X_even[k] + t
        X[k + n // 2] = X_even[k] - t
        w *= w_n
    return X

def ifft(X):
    n = len(X)
    X_conj = [x.conjugate() for x in X]
    x = fft(X_conj)
    x = [val.conjugate() / n for val in x]
    return x

# Example
X = [10, -2-2j, -2, -2+2j]
print("IFFT:", ifft(X))
```

C (IFFT using FFT structure)

```c
#include <stdio.h>
#include <complex.h>
#include <math.h>

#define PI 3.14159265358979323846

void fft(int n, double complex *x, int inverse) {
    if (n <= 1) return;

    double complex even[n/2], odd[n/2];
    for (int i = 0; i < n/2; i++) {
        even[i] = x[2*i];
        odd[i] = x[2*i + 1];
    }

    fft(n/2, even, inverse);
    fft(n/2, odd, inverse);

    double sign = inverse ? 2.0 * PI : -2.0 * PI;
    for (int k = 0; k < n/2; k++) {
        double complex w = cexp(I * sign * k / n);
        double complex t = w * odd[k];
        x[k] = even[k] + t;
        x[k + n/2] = even[k] - t;
        if (inverse) {
            x[k] /= 2;
            x[k + n/2] /= 2;
        }
    }
}

int main() {
    double complex X[4] = {10, -2-2*I, -2, -2+2*I};
    fft(4, X, 1); // inverse FFT
    for (int i = 0; i < 4; i++)
        printf("x[%d] = %.2f + %.2fi\n", i, creal(X[i]), cimag(X[i]));
}
```

#### Why It Matters

- Restores time-domain data from frequency components
- Used in signal reconstruction, convolution, and filter design
- Guarantees perfect reversibility with FFT
- Central to compression algorithms, image restoration, and physics simulations

#### A Gentle Proof (Why It Works)

The DFT and IFFT matrices are Hermitian inverses:

$$
F_{n}^{-1}=\frac{1}{n}\overline{F_n}^T
$$

Thus, applying FFT followed by IFFT yields the identity:

$$
\text{IFFT}(\text{FFT}(x))=x
$$

Conjugation flips the sign of exponents, and scaling by $1/n$ ensures normalization.

#### Try It Yourself

1. Compute FFT of $[1,2,3,4]$, then apply IFFT.
2. Compare reconstructed result to original.
3. Modify $X$ by zeroing high frequencies, watch smoothing effect.
4. Use IFFT for polynomial multiplication results.
5. Visualize magnitude and phase before and after IFFT.

#### Test Cases

| Input $X$           | Output $x$        |
| ------------------- | ----------------- |
| [10,-2-2i,-2,-2+2i] | [1,2,3,4]         |
| [4,0,0,0]           | [1,1,1,1]         |
| [8,0,0,0,0,0,0,0]   | [1,1,1,1,1,1,1,1] |

#### Complexity

- Time: $O(n\log n)$
- Space: $O(n)$

IFFT is the mirror step of FFT, same structure, reversed flow. It completes the cycle: from time to frequency and back, without losing a single bit of truth.

### 566 Convolution via FFT

Convolution is one of the most fundamental operations in mathematics, signal processing, and computer science. It combines two sequences into one, blending information over shifts.
But computing convolution directly takes $O(n^2)$ time. The FFT Convolution Theorem gives us a shortcut: by transforming to the frequency domain, we can do it in $O(n\log n)$ time.

#### What Problem Are We Solving?

Given two sequences $a$ and $b$ of lengths $n$ and $m$, we want their convolution $c = a * b$:

$$
c_k=\sum_{i=0}^{k}a_i\cdot b_{k-i}
$$

for $k=0,\ldots,n+m-2$.

This appears in:

- Signal processing (filtering, correlation)
- Polynomial multiplication
- Pattern matching
- Probability distributions (sum of random variables)

A direct computation is $O(nm)$. Using FFT, we can reduce it to $O(n\log n)$.

#### How Does It Work (Plain Language)

The Convolution Theorem says:

$$
\text{DFT}(a*b)=\text{DFT}(a)\cdot\text{DFT}(b)
$$

That is, convolution in time domain equals pointwise multiplication in frequency domain.

So to compute convolution fast:

1. Pad both sequences to size $N\ge n+m-1$ (power of 2).
2. Compute FFT of both: $A=\text{FFT}(a)$, $B=\text{FFT}(b)$.
3. Multiply pointwise: $C_k=A_k\cdot B_k$.
4. Apply inverse FFT: $c=\text{IFFT}(C)$.
5. Take real parts (round small errors).

#### Example

Let $a=[1,2,3]$, $b=[4,5,6]$

Expected convolution (by hand):

$$
c=[1\cdot4,1\cdot5+2\cdot4,1\cdot6+2\cdot5+3\cdot4,2\cdot6+3\cdot5,3\cdot6]
$$

$$
c=[4,13,28,27,18]
$$

FFT method:

1. Pad $a,b$ to length 8
2. $A=\text{FFT}(a)$, $B=\text{FFT}(b)$
3. $C=A\cdot B$
4. $c=\text{IFFT}(C)$
5. Round to integers $\Rightarrow [4,13,28,27,18]$

Matches perfectly.

#### Tiny Code (Easy Versions)

Python (FFT Convolution)

```python
import cmath

def fft(x):
    n = len(x)
    if n == 1:
        return x
    w_n = cmath.exp(-2j * cmath.pi / n)
    w = 1
    X_even = fft(x[0::2])
    X_odd = fft(x[1::2])
    X = [0] * n
    for k in range(n // 2):
        t = w * X_odd[k]
        X[k] = X_even[k] + t
        X[k + n // 2] = X_even[k] - t
        w *= w_n
    return X

def ifft(X):
    n = len(X)
    X_conj = [x.conjugate() for x in X]
    x = fft(X_conj)
    return [v.conjugate()/n for v in x]

def convolution(a, b):
    n = 1
    while n < len(a) + len(b) - 1:
        n *= 2
    a += [0]*(n - len(a))
    b += [0]*(n - len(b))
    A = fft(a)
    B = fft(b)
    C = [A[i]*B[i] for i in range(n)]
    c = ifft(C)
    return [round(v.real) for v in c]

# Example
a = [1, 2, 3]
b = [4, 5, 6]
print(convolution(a, b))  # [4, 13, 28, 27, 18]
```

C (FFT Convolution Skeleton)

```c
// Assume fft() and ifft() functions are implemented
void convolution(int n, double complex *a, double complex *b, double complex *c) {
    int size = 1;
    while (size < 2 * n) size <<= 1;

    // pad arrays
    for (int i = n; i < size; i++) {
        a[i] = 0;
        b[i] = 0;
    }

    fft(size, a, 0);
    fft(size, b, 0);
    for (int i = 0; i < size; i++)
        c[i] = a[i] * b[i];
    fft(size, c, 1); // inverse FFT (scaled)
}
```

#### Why It Matters

- Turns slow $O(n^2)$ convolution into $O(n\log n)$
- Core technique in polynomial multiplication, digital filters, signal correlation, neural network layers, probabilistic sums
- Used in big integer multiplication (Karatsuba, Schönhage–Strassen)

#### A Gentle Proof (Why It Works)

The DFT converts convolution to multiplication:

$$
C=\text{DFT}(a*b)=\text{DFT}(a)\cdot\text{DFT}(b)
$$

Because the DFT matrix diagonalizes cyclic shifts, it transforms convolution (which involves shifts and sums) into independent multiplications of frequencies.

Then applying $\text{IDFT}$ recovers the exact result.

#### Try It Yourself

1. Convolve $[1,2,3]$ and $[4,5,6]$ manually, then verify via FFT.
2. Pad inputs to power of 2 and observe intermediate arrays.
3. Try longer polynomials (length 1000), measure speed difference.
4. Plot inputs, their spectra, and output.
5. Implement modulo convolution (e.g. with Number Theoretic Transform).

#### Test Cases

| $a$     | $b$     | $a*b$           |
| ------- | ------- | --------------- |
| [1,2,3] | [4,5,6] | [4,13,28,27,18] |
| [1,1,1] | [1,2,3] | [1,3,6,5,3]     |
| [2,0,1] | [3,4]   | [6,8,3,4]       |

#### Complexity

- Time: $O(n\log n)$
- Space: $O(n)$

FFT Convolution is how computers multiply long numbers, combine signals, and merge patterns, faster than you could ever do by hand.

### 567 Number Theoretic Transform (NTT)

The Number Theoretic Transform (NTT) is the modular arithmetic version of the FFT.
It replaces complex numbers with integers under a finite modulus, allowing all computations to be done exactly (no floating-point error), perfect for cryptography, combinatorics, and competitive programming.

While FFT uses complex roots of unity $e^{2\pi i/n}$, NTT uses primitive roots of unity modulo a prime.

#### What Problem Are We Solving?

We want to perform polynomial multiplication (or convolution) exactly, using modular arithmetic instead of floating-point numbers.

Given two polynomials:

$$
A(x)=\sum_{i=0}^{n-1}a_i x^i,\quad B(x)=\sum_{i=0}^{m-1}b_i x^i
$$

Their product:

$$
C(x)=A(x)\cdot B(x)=\sum_{k=0}^{n+m-2}c_kx^k
$$

with

$$
c_k=\sum_{i=0}^{k}a_i\cdot b_{k-i}
$$

We want to compute all $c_k$ efficiently, under modulo $M$.

#### Key Idea

The Convolution Theorem holds in modular arithmetic too, if we can find a primitive $n$-th root of unity $g$ such that:

$$
g^n\equiv1\pmod M
$$

and

$$
g^k\not\equiv1\pmod M,\ \text{for } 0<k<n
$$

Then, we can define an NTT just like FFT:

$$
A_k=\sum_{j=0}^{n-1}a_j\cdot g^{jk}\pmod M
$$

and invert it using $g^{-1}$ and $n^{-1}\pmod M$.

#### How It Works (Plain Language)

1. Choose a modulus $M$ where $M=k\cdot 2^m+1$ (a FFT-friendly prime).
   Example: $M=998244353=119\cdot2^{23}+1$
2. Find primitive root $g$ (e.g. $g=3$).
3. Perform FFT-like butterflies with modular multiplications.
4. For inverse, use $g^{-1}$ and divide by $n$ via modular inverse.

This ensures exact results, no rounding errors.

#### Example

Let $M=17$, $n=4$, and $g=4$ since $4^4\equiv1\pmod{17}$.

Input: $a=[1,2,3,4]$

Compute NTT:

$$
A_k=\sum_{j=0}^{3}a_j\cdot g^{jk}\pmod{17}
$$

Result: $A=[10,2,16,15]$

Inverse NTT (using $g^{-1}=13$, $n^{-1}=13$ mod 17):

$$
a_j=\frac{1}{n}\sum_{k=0}^{3}A_k\cdot g^{-jk}\pmod{17}
$$

Recovered: $[1,2,3,4]$

#### Tiny Code (NTT Template)

Python

```python
MOD = 998244353
ROOT = 3  # primitive root

def modpow(a, e, m):
    res = 1
    while e:
        if e & 1:
            res = res * a % m
        a = a * a % m
        e >>= 1
    return res

def ntt(a, invert):
    n = len(a)
    j = 0
    for i in range(1, n):
        bit = n >> 1
        while j & bit:
            j ^= bit
            bit >>= 1
        j ^= bit
        if i < j:
            a[i], a[j] = a[j], a[i]
    len_ = 2
    while len_ <= n:
        wlen = modpow(ROOT, (MOD - 1) // len_, MOD)
        if invert:
            wlen = modpow(wlen, MOD - 2, MOD)
        for i in range(0, n, len_):
            w = 1
            for j in range(len_ // 2):
                u = a[i + j]
                v = a[i + j + len_ // 2] * w % MOD
                a[i + j] = (u + v) % MOD
                a[i + j + len_ // 2] = (u - v + MOD) % MOD
                w = w * wlen % MOD
        len_ <<= 1
    if invert:
        inv_n = modpow(n, MOD - 2, MOD)
        for i in range(n):
            a[i] = a[i] * inv_n % MOD

def multiply(a, b):
    n = 1
    while n < len(a) + len(b):
        n <<= 1
    a += [0]*(n - len(a))
    b += [0]*(n - len(b))
    ntt(a, False)
    ntt(b, False)
    for i in range(n):
        a[i] = a[i] * b[i] % MOD
    ntt(a, True)
    return a
```

#### Why It Matters

- Exact modular results, no floating-point rounding
- Enables polynomial multiplication, combinatorial transforms, big integer multiplication
- Critical in cryptography, lattice algorithms, and competitive programming
- Used in modern schemes (like NTT-based homomorphic encryption)

#### A Gentle Proof (Why It Works)

The NTT matrix $W_n$ with $W_{jk}=g^{jk}\pmod M$ satisfies:

$$
W_n^{-1}=\frac{1}{n}\overline{W_n}
$$

Because $g$ is a primitive $n$-th root of unity, columns of $W_n$ are orthogonal modulo $M$, and modular inverses exist (since $M$ is prime).
Hence, the forward and inverse transforms perfectly invert each other.

#### Try It Yourself

1. Use $M=17$, $g=4$, $a=[1,2,3,4]$, compute NTT manually.
2. Multiply $[1,2,3]$ and $[4,5,6]$ under mod $17$ using NTT.
3. Test mod $998244353$, length $8$.
4. Compare FFT (float) vs NTT (modular) results.
5. Observe rounding-free exactness.

#### Test Cases

| $a$     | $b$     | $a*b$ (mod 17)                          |
| ------- | ------- | --------------------------------------- |
| [1,2,3] | [4,5,6] | [4,13,28,27,18] mod 17 = [4,13,11,10,1] |
| [1,1]   | [1,1]   | [1,2,1]                                 |

#### Complexity

- Time: $O(n\log n)$
- Space: $O(n)$

The NTT is the FFT of the integers, same beauty, different universe. It merges algebra, number theory, and computation into one seamless engine of exactness.

### 568 Inverse NTT (INTT)

The Inverse Number Theoretic Transform (INTT) is the reverse operation of the NTT. It brings data back from the frequency domain to the coefficient domain, fully reconstructing the original sequence.
Just like IFFT for FFT, the INTT undoes the modular transform using the inverse root of unity and a modular scaling factor.

#### What Problem Are We Solving?

Given an NTT-transformed sequence $A=[A_0,A_1,\ldots,A_{n-1}]$, we want to recover the original sequence $a=[a_0,a_1,\ldots,a_{n-1}]$ under modulus $M$.

The definition mirrors the inverse DFT:

$$
a_j = n^{-1} \sum_{k=0}^{n-1} A_k \cdot g^{-jk} \pmod M
$$

where:

- $g$ is a primitive $n$-th root of unity mod $M$
- $n^{-1}$ is the modular inverse of $n$ under mod $M$

#### How Does It Work (Plain Language)

The INTT follows the same butterfly structure as the NTT, but:

1. Use inverse twiddle factors $g^{-1}$ instead of $g$
2. After all stages, multiply each element by $n^{-1}$ mod $M$

This ensures perfect inversion:

$$
\text{INTT}(\text{NTT}(a)) = a
$$

#### Example

Let $M=17$, $n=4$, $g=4$
Then $g^{-1}=13$ (since $4\cdot13\equiv1\pmod{17}$), $n^{-1}=13$ (since $4\cdot13\equiv1\pmod{17}$)

Suppose $A=[10,2,16,15]$

Compute:

$$
a_j=13\cdot\sum_{k=0}^{3}A_k\cdot(13)^{jk}\pmod{17}
$$

After calculation:
$a=[1,2,3,4]$

Perfect reconstruction.

#### Tiny Code (Inverse NTT)

Python (Inverse NTT)

```python
MOD = 998244353
ROOT = 3  # primitive root

def modpow(a, e, m):
    res = 1
    while e:
        if e & 1:
            res = res * a % m
        a = a * a % m
        e >>= 1
    return res

def ntt(a, invert=False):
    n = len(a)
    j = 0
    for i in range(1, n):
        bit = n >> 1
        while j & bit:
            j ^= bit
            bit >>= 1
        j ^= bit
        if i < j:
            a[i], a[j] = a[j], a[i]

    len_ = 2
    while len_ <= n:
        wlen = modpow(ROOT, (MOD - 1) // len_, MOD)
        if invert:
            wlen = modpow(wlen, MOD - 2, MOD)
        for i in range(0, n, len_):
            w = 1
            for j in range(len_ // 2):
                u = a[i + j]
                v = a[i + j + len_ // 2] * w % MOD
                a[i + j] = (u + v) % MOD
                a[i + j + len_ // 2] = (u - v + MOD) % MOD
                w = w * wlen % MOD
        len_ <<= 1

    if invert:
        inv_n = modpow(n, MOD - 2, MOD)
        for i in range(n):
            a[i] = a[i] * inv_n % MOD

# Example
A = [10, 2, 16, 15]
ntt(A, invert=True)
print("Inverse NTT:", A)  # [1,2,3,4]
```

C (Inverse NTT Skeleton)

```c
void ntt(double complex *a, int n, int invert) {
    // ... (bit-reversal same as NTT)
    for (int len = 2; len <= n; len <<= 1) {
        long long wlen = modpow(ROOT, (MOD - 1) / len, MOD);
        if (invert) wlen = modinv(wlen, MOD);
        for (int i = 0; i < n; i += len) {
            long long w = 1;
            for (int j = 0; j < len / 2; j++) {
                long long u = a[i + j];
                long long v = a[i + j + len / 2] * w % MOD;
                a[i + j] = (u + v) % MOD;
                a[i + j + len / 2] = (u - v + MOD) % MOD;
                w = w * wlen % MOD;
            }
        }
    }
    if (invert) {
        long long inv_n = modinv(n, MOD);
        for (int i = 0; i < n; i++)
            a[i] = a[i] * inv_n % MOD;
    }
}
```

#### Why It Matters

- Completes the modular FFT pipeline
- Ensures exact reconstruction of coefficients
- Core for polynomial multiplication, cryptographic transforms, error-correcting codes
- Enables precise modular computations with no floating-point errors

#### A Gentle Proof (Why It Works)

NTT and INTT are modular inverses of each other:

$$
\text{NTT}(a)=W_n\cdot a,\quad \text{INTT}(A)=n^{-1}\cdot W_n^{-1}\cdot A
$$

where $W_n$ is the Vandermonde matrix over $g$.
Because $W_nW_n^{-1}=I$, and $g$ is primitive, the two transforms form a bijection.

#### Try It Yourself

1. Compute NTT of $[1,2,3,4]$ under mod $17$, then apply INTT.
2. Verify $\text{INTT}(\text{NTT}(a))=a$.
3. Check inverse properties for different $n=8,16$.
4. Multiply polynomials using NTT + INTT pipeline.
5. Compare modular vs floating-point FFT.

#### Test Cases

| Input $A$            | Output $a$ (mod 17) |
| -------------------- | ------------------- |
| [10,2,16,15]         | [1,2,3,4]           |
| [4,13,11,10,1,0,0,0] | [1,2,3,4,5]         |

#### Complexity

- Time: $O(n\log n)$
- Space: $O(n)$

The INTT closes the circle: every modular transformation now has a perfect inverse. From number theory to code, it keeps exactness at every step.

### 569 Bluestein's Algorithm

Bluestein's Algorithm, also known as the chirp z-transform, is a clever method to compute a Discrete Fourier Transform (DFT) of arbitrary length $n$ using a Fast Fourier Transform (FFT) of length $m \ge 2n-1$.
Unlike Cooley–Tukey, it works even when $n$ is not a power of two.

#### What Problem Are We Solving?

Standard FFTs (like Cooley–Tukey) require $n$ to be a power of two for fast computation.
What if we want to compute the DFT for an arbitrary $n$, such as $n=12$, $30$, or a prime number?

Bluestein's algorithm rewrites the DFT as a convolution, which can then be computed efficiently using any FFT of sufficient size.

Given a sequence $a=[a_0,a_1,\ldots,a_{n-1}]$, we want:

$$
A_k = \sum_{j=0}^{n-1} a_j \cdot \omega_n^{jk}, \quad 0 \le k < n
$$

where $\omega_n = e^{-2\pi i / n}$.

#### How Does It Work (Plain Language)

Bluestein's algorithm transforms the DFT into a convolution problem:

$$
A_k = \sum_{j=0}^{n-1} a_j \cdot \omega_n^{j^2/2} \cdot \omega_n^{(k-j)^2/2}
$$

Define:

- $b_j = a_j \cdot \omega_n^{j^2/2}$
- $c_j = \omega_n^{-j^2/2}$

Then $A_k$ is the convolution of $b$ and $c$, scaled by $\omega_n^{k^2/2}$.

We can compute this convolution using FFT-based polynomial multiplication, even when $n$ is arbitrary.

#### Algorithm Steps

1. Precompute chirp factors $\omega_n^{j^2/2}$ and their inverses.
2. Build sequences $b_j$ and $c_j$.
3. Pad both to length $m \ge 2n-1$.
4. Perform FFT-based convolution:
   $$
   d = \text{IFFT}(\text{FFT}(b) \cdot \text{FFT}(c))
   $$
5. Extract $A_k = d_k \cdot \omega_n^{-k^2/2}$ for $k=0,\ldots,n-1$.

#### Tiny Code (Python)

```python
import cmath
import math

def next_power_of_two(x):
    return 1 << (x - 1).bit_length()

def fft(a, invert=False):
    n = len(a)
    if n == 1:
        return a
    a_even = fft(a[0::2], invert)
    a_odd = fft(a[1::2], invert)
    ang = 2 * math.pi / n * (-1 if not invert else 1)
    w, wn = 1, cmath.exp(1j * ang)
    y = [0] * n
    for k in range(n // 2):
        t = w * a_odd[k]
        y[k] = a_even[k] + t
        y[k + n // 2] = a_even[k] - t
        w *= wn
    if invert:
        for i in range(n):
            y[i] /= 2
    return y

def bluestein_dft(a):
    n = len(a)
    m = next_power_of_two(2 * n - 1)
    ang = math.pi / n
    w = [cmath.exp(-1j * ang * j * j) for j in range(n)]
    b = [a[j] * w[j] for j in range(n)] + [0] * (m - n)
    c = [w[j].conjugate() for j in range(n)] + [0] * (m - n)

    B = fft(b)
    C = fft(c)
    D = [B[i] * C[i] for i in range(m)]
    d = fft(D, invert=True)

    return [d[k] * w[k] for k in range(n)]

# Example
a = [1, 2, 3]
print("DFT via Bluestein:", bluestein_dft(a))
```

#### Why It Matters

- Handles arbitrary-length DFTs (prime $n$, non-power-of-two)
- Widely used in signal processing, polynomial arithmetic, NTT generalization
- Enables uniform FFT pipelines without length restriction
- Core idea: Chirp-z transform → Convolution → FFT

#### A Gentle Proof (Why It Works)

We rewrite DFT terms:

$$
A_k = \sum_{j=0}^{n-1} a_j \cdot \omega_n^{jk}
$$

Multiply and divide by $\omega_n^{(j^2 + k^2)/2}$:

$$
A_k = \omega_n^{-k^2/2} \sum_{j=0}^{n-1} (a_j \cdot \omega_n^{j^2/2}) \cdot \omega_n^{(k-j)^2/2}
$$

This is a discrete convolution of two sequences, computable via FFT.
Thus, DFT reduces to convolution, enabling $O(m\log m)$ performance for any $n$.

#### Try It Yourself

1. Compute DFT for $n=6$, $a=[1,2,3,4,5,6]$.
2. Compare results from Bluestein and Cooley–Tukey (if $n$ is power of two).
3. Try $n=7$ (prime), only Bluestein works efficiently.
4. Verify that applying inverse DFT returns the original sequence.
5. Explore zero-padding effect when $m>2n-1$.

#### Test Cases

| Input $a$   | $n$ | Output (approx)                                  |
| ----------- | --- | ------------------------------------------------ |
| [1,2,3]     | 3   | [6,(-1.5+0.866i),(-1.5-0.866i)]                  |
| [1,2,3,4,5] | 5   | [15,-2.5+3.44i,-2.5+0.81i,-2.5-0.81i,-2.5-3.44i] |

#### Complexity

- Time: $O(m\log m)$ where $m\ge2n-1$
- Space: $O(m)$

Bluestein's algorithm bridges the gap between power-of-two FFTs and general-length DFTs, turning every sequence into a convolutional melody computable by fast transforms.

### 570 FFT-Based Multiplication

FFT-Based Multiplication uses the Fast Fourier Transform to multiply large integers or polynomials efficiently by transforming multiplication into pointwise products in the frequency domain.

Instead of multiplying term by term (which is $O(n^2)$), we leverage the FFT to compute the convolution in $O(n\log n)$ time.

#### What Problem Are We Solving?

When multiplying two large polynomials (or integers), the naive approach requires $O(n^2)$ operations.
For large $n$, this becomes impractical.

We want to compute:

$$
C(x) = A(x) \cdot B(x)
$$

where

$$
A(x) = \sum_{i=0}^{n-1} a_i x^i,\quad B(x) = \sum_{j=0}^{n-1} b_j x^j
$$

We need coefficients $c_k$ of $C(x)$ such that

$$
c_k = \sum_{i+j=k} a_i b_j
$$

This is exactly a convolution. The FFT computes it efficiently.

#### How Does It Work (Plain Language)

1. Pad the sequences so their lengths are a power of two and large enough to hold the full result.
2. FFT-transform both sequences to frequency space.
3. Multiply the results elementwise.
4. Inverse FFT to bring back the coefficients.
5. Round to nearest integer (for integer multiplication).

It's like tuning two melodies into frequency space, multiplying harmonics, and transforming them back into time.

#### Algorithm Steps

1. Let $A$ and $B$ be coefficient arrays.
2. Choose $n$ as the smallest power of two $\ge 2 \cdot \max(\text{len}(A), \text{len}(B))$.
3. Pad $A$ and $B$ to length $n$.
4. Compute FFT$(A)$ and FFT$(B)$.
5. Compute $C'[k] = A'[k] \cdot B'[k]$.
6. Compute $C = \text{IFFT}(C')$.
7. Round real parts of $C$ to nearest integer.

#### Tiny Code (Python)

```python
import cmath
import math

def fft(a, invert=False):
    n = len(a)
    if n == 1:
        return a
    a_even = fft(a[0::2], invert)
    a_odd = fft(a[1::2], invert)
    ang = 2 * math.pi / n * (-1 if not invert else 1)
    w, wn = 1, cmath.exp(1j * ang)
    y = [0] * n
    for k in range(n // 2):
        t = w * a_odd[k]
        y[k] = a_even[k] + t
        y[k + n // 2] = a_even[k] - t
        w *= wn
    if invert:
        for i in range(n):
            y[i] /= 2
    return y

def multiply(a, b):
    n = 1
    while n < len(a) + len(b):
        n <<= 1
    fa = a + [0] * (n - len(a))
    fb = b + [0] * (n - len(b))
    FA = fft(fa)
    FB = fft(fb)
    FC = [FA[i] * FB[i] for i in range(n)]
    C = fft(FC, invert=True)
    return [round(c.real) for c in C]

# Example: Multiply (1 + 2x + 3x^2) * (4 + 5x + 6x^2)
print(multiply([1, 2, 3], [4, 5, 6]))
# Output: [4, 13, 28, 27, 18]
```

#### Why It Matters

- Foundation of big integer arithmetic (used in libraries like GMP).
- Enables fast polynomial multiplication.
- Used in cryptography, signal processing, and FFT-based convolution.
- Scales to millions of terms efficiently.

#### A Gentle Proof (Why It Works)

Let

$$
C(x) = A(x)B(x)
$$

In the frequency domain (using DFT):

$$
\text{DFT}(C) = \text{DFT}(A) \cdot \text{DFT}(B)
$$

By the Convolution Theorem, multiplication in frequency space corresponds to convolution in time:

$$
C = \text{IFFT}(\text{FFT}(A) \cdot \text{FFT}(B))
$$

Thus, we get $c_k = \sum_{i+j=k} a_i b_j$ automatically after inverse transform.

#### Try It Yourself

1. Multiply $(1+2x+3x^2)$ and $(4+5x+6x^2)$ by hand.
2. Implement integer multiplication using base-10 digits as coefficients.
3. Compare FFT vs naive multiplication for $n=1024$.
4. Try with complex or modular arithmetic.
5. Explore rounding issues and precision.

#### Test Cases

| A(x)    | B(x)    | Result          |
| ------- | ------- | --------------- |
| [1,2,3] | [4,5,6] | [4,13,28,27,18] |
| [3,2,1] | [1,0,1] | [3,2,4,2,1]     |

#### Complexity

- Time: $O(n\log n)$
- Space: $O(n)$

FFT-based multiplication turns a slow quadratic process into a symphony of frequency-domain operations, fast, elegant, and scalable.

# Section 58. Numerical Methods 

### 571 Newton–Raphson

Newton–Raphson is a fast root finding method that refines a guess $x$ for $f(x)=0$ by following the tangent line of $f$ at $x$. It has quadratic convergence near a simple root if the derivative does not vanish.

#### What Problem Are We Solving?

Given a differentiable function $f:\mathbb{R}\to\mathbb{R}$, find $x^*$ such that $f(x^*)=0$. Typical use cases include solving nonlinear equations, optimizing $1$D functions via $f^\prime(x)=0$, and as an inner step in larger numerical methods.

#### How Does It Work (Plain Language)

At a current guess $x_k$, approximate $f$ by its tangent:
$$
f(x)\approx f(x_k)+f^\prime(x_k)(x-x_k).
$$
Set this linear model to zero and solve for the intercept:
$$
x_{k+1}=x_k-\frac{f(x_k)}{f^\prime(x_k)}.
$$
Repeat until the change is small or $|f(x_k)|$ is small.

Update rule
$$
x_{k+1}=x_k-\frac{f(x_k)}{f^\prime(x_k)}.
$$

#### Example

Solve $x^2-2=0$ (square root of $2$).

Let $f(x)=x^2-2$, $f^\prime(x)=2x$, start $x_0=1$.

- $x_1=1-\frac{-1}{2}=1.5$
- $x_2=1.5-\frac{1.5^2-2}{3}=1.416\overline{6}$
- $x_3\approx1.4142157$

Rapid approach to $\sqrt{2}\approx1.41421356$.

#### Tiny Code

Python

```python
def newton(f, df, x0, tol=1e-10, max_iter=100):
    x = x0
    for _ in range(max_iter):
        fx = f(x)
        dfx = df(x)
        if dfx == 0:
            raise ZeroDivisionError("Derivative became zero")
        x_new = x - fx / dfx
        if abs(x_new - x) <= tol:
            return x_new
        x = x_new
    return x

# Example: sqrt(2)
root = newton(lambda x: x*x - 2, lambda x: 2*x, 1.0)
print(root)
```

C

```c
#include <stdio.h>
#include <math.h>

double newton(double (*f)(double), double (*df)(double),
              double x0, double tol, int max_iter) {
    double x = x0;
    for (int i = 0; i < max_iter; i++) {
        double fx = f(x), dfx = df(x);
        if (dfx == 0.0) break;
        double x_new = x - fx / dfx;
        if (fabs(x_new - x) <= tol) return x_new;
        x = x_new;
    }
    return x;
}

double f(double x){ return x*x - 2.0; }
double df(double x){ return 2.0*x; }

int main(void){
    double r = newton(f, df, 1.0, 1e-10, 100);
    printf("%.12f\n", r);
    return 0;
}
```

#### Why It Matters

- Very fast near a simple root: error roughly squares each step
- Widely used inside solvers for nonlinear systems, optimization, and implicit ODE steps
- Easy to implement when $f^\prime$ is available or cheap to approximate

#### A Gentle Proof Idea

If $f\in C^2$ and $x^*$ is a simple root with $f(x^*)=0$ and $f^\prime(x^*)\ne0$, a Taylor expansion around $x^*$ gives
$$
f(x)=f^\prime(x^*)(x-x^*)+\tfrac12 f^{\prime\prime}(\xi)(x-x^*)^2.
$$
Plugging the Newton update shows the new error $e_{k+1}=x_{k+1}-x^*$ satisfies
$$
e_{k+1}\approx -\frac{f^{\prime\prime}(x^*)}{2f^\prime(x^*)}e_k^2,
$$
which is quadratic convergence when $e_k$ is small.

#### Practical Tips

- Choose a good initial guess $x_0$ to avoid divergence
- Guard against $f^\prime(x_k)=0$ or tiny derivatives
- Use damping: $x_{k+1}=x_k-\alpha\frac{f(x_k)}{f^\prime(x_k)}$ with $0<\alpha\le1$ if steps overshoot
- Stop criteria: $|f(x_k)|\le\varepsilon$ or $|x_{k+1}-x_k|\le\varepsilon$

#### Try It Yourself

1. Solve $\cos x - x=0$ starting at $x_0=0.5$.
2. Find cube root of $5$ via $f(x)=x^3-5$.
3. Minimize $g(x)=(x-3)^2$ by applying Newton to $g^\prime(x)=0$.
4. Experiment with a poor initial guess for $f(x)=\tan x - x$ near $\pi/2$ and observe failure modes.
5. Add line search or damping and compare robustness.

#### Test Cases

| $f(x)$       | $f^\prime(x)$  | Root              | Start $x_0$ | Iter to $10^{-10}$ |
| ------------ | -------------- | ----------------- | ----------- | ------------------ |
| $x^2-2$      | $2x$           | $\sqrt{2}$        | $1.0$       | 5                  |
| $\cos x - x$ | $-,\sin x - 1$ | $\approx0.739085$ | $0.5$       | 5–6                |
| $x^3-5$      | $3x^2$         | $\sqrt[3]{5}$     | $1.5$       | 6–7                |

*(Iterations are indicative and depend on tolerances.)*

#### Complexity

- Per iteration: one $f$ and one $f^\prime$ evaluation plus $O(1)$ arithmetic
- Convergence: typically quadratic near a simple root
- Overall cost: small number of iterations for well behaved problems

Newton–Raphson is the standard $1$D root finder when derivatives are available and the initial guess is reasonable. It is simple, fast, and forms the backbone of many higher dimensional methods.

### 572 Bisection Method

The Bisection Method is one of the simplest and most reliable ways to find a root of a continuous function. It's a "divide and narrow" search for $x^*$ such that $f(x^*)=0$ within an interval where the function changes sign.

#### What Problem Are We Solving?

We want to solve $f(x)=0$ for a continuous function $f$ when we know two points $a$ and $b$ such that:

$$
f(a)\cdot f(b) < 0
$$

That means the function crosses zero between $a$ and $b$ by the Intermediate Value Theorem.

#### How Does It Work (Plain Language)

At each step:

1. Compute midpoint $m=\frac{a+b}{2}$
2. Check sign of $f(m)$

   * If $f(a)\cdot f(m) < 0$, root is between $a$ and $m$
   * Else root is between $m$ and $b$
3. Replace the interval with the new one and repeat until the interval is small enough

It halves the search space every step, like a binary search for roots.

#### Algorithm Steps

1. Start with $[a,b]$ such that $f(a)\cdot f(b)<0$
2. While $|b-a|>\varepsilon$:

   * $m=\frac{a+b}{2}$
   * If $f(m)=0$ (or close enough), stop
   * Else, pick the half where sign changes
3. Return midpoint as approximate root

#### Tiny Code

Python

```python
def bisection(f, a, b, tol=1e-10, max_iter=100):
    fa, fb = f(a), f(b)
    if fa * fb >= 0:
        raise ValueError("f(a) and f(b) must have opposite signs")
    for _ in range(max_iter):
        m = 0.5 * (a + b)
        fm = f(m)
        if abs(fm) < tol or (b - a) / 2 < tol:
            return m
        if fa * fm < 0:
            b, fb = m, fm
        else:
            a, fa = m, fm
    return 0.5 * (a + b)

# Example: root of x^3 - x - 2 = 0
root = bisection(lambda x: x3 - x - 2, 1, 2)
print(root)  # ~1.5213797
```

C

```c
#include <stdio.h>
#include <math.h>

double f(double x) { return x*x*x - x - 2; }

double bisection(double a, double b, double tol, int max_iter) {
    double fa = f(a), fb = f(b);
    if (fa * fb >= 0) return NAN;
    for (int i = 0; i < max_iter; i++) {
        double m = 0.5 * (a + b);
        double fm = f(m);
        if (fabs(fm) < tol || (b - a) / 2 < tol) return m;
        if (fa * fm < 0) {
            b = m; fb = fm;
        } else {
            a = m; fa = fm;
        }
    }
    return 0.5 * (a + b);
}

int main(void) {
    printf("%.10f\n", bisection(1, 2, 1e-10, 100));
    return 0;
}
```

#### Why It Matters

- Guaranteed convergence if $f$ is continuous and sign changes
- No derivative required (unlike Newton–Raphson)
- Robust and simple to implement
- Basis for hybrid methods like Brent's method

#### A Gentle Proof (Why It Works)

By the Intermediate Value Theorem, if $f(a)\cdot f(b)<0$ and $f$ is continuous, there exists $x^*\in[a,b]$ such that $f(x^*)=0$.

Each iteration halves the interval:
$$
|b_{k+1}-a_{k+1}|=\frac{1}{2}|b_k-a_k|
$$
After $k$ iterations:
$$
|b_k-a_k|=\frac{1}{2^k}|b_0-a_0|
$$
To achieve tolerance $\varepsilon$:
$$
k \ge \log_2\frac{|b_0-a_0|}{\varepsilon}
$$

#### Try It Yourself

1. Find root of $f(x)=x^3-x-2$ on $[1,2]$.
2. Try $f(x)=\cos x - x$ on $[0,1]$.
3. Compare iterations vs Newton–Raphson.
4. Test failure if $f(a)$ and $f(b)$ have same sign.
5. Observe how tolerance affects accuracy and iterations.

#### Test Cases

| Function     | Interval | Root (approx) | Iterations (ε=1e-6) |
| ------------ | -------- | ------------- | ------------------- |
| $x^2-2$      | [1,2]    | 1.414214      | 20                  |
| $x^3-x-2$    | [1,2]    | 1.521380      | 20                  |
| $\cos x - x$ | [0,1]    | 0.739085      | 20                  |

#### Complexity

- Time: $O(\log_2(\frac{b-a}{\varepsilon}))$
- Space: $O(1)$

The Bisection Method trades speed for certainty, it never fails when the sign condition holds, making it a cornerstone of reliable root finding.

### 573 Secant Method

The Secant Method is a root-finding algorithm that uses two initial guesses and repeatedly refines them using secant lines, straight lines passing through two recent points of the function. It's like a derivative-free Newton–Raphson: instead of using $f'(x)$, we estimate it from the slope of the secant.

#### What Problem Are We Solving?

We want to solve for $x^*$ such that $f(x^*)=0$, but sometimes we don't have the derivative $f'(x)$.
The secant method approximates it numerically:

$$
f'(x_k)\approx \frac{f(x_k)-f(x_{k-1})}{x_k-x_{k-1}}
$$

By replacing the exact derivative with this difference quotient, we still follow the Newton-like update.

#### How Does It Work (Plain Language)

Think of drawing a line through two points $(x_{k-1},f(x_{k-1}))$ and $(x_k,f(x_k))$.
That line crosses the $x$-axis at a new guess $x_{k+1}$.
Repeat the process until convergence.

The update formula is:

$$
x_{k+1}=x_k-f(x_k)\frac{x_k-x_{k-1}}{f(x_k)-f(x_{k-1})}
$$

Each iteration uses the last two estimates instead of one (like Newton).

#### Algorithm Steps

1. Start with two initial guesses $x_0$ and $x_1$ such that $f(x_0)\ne f(x_1)$.

2. Repeat until convergence:

   $$
   x_{k+1}=x_k-f(x_k)\frac{x_k-x_{k-1}}{f(x_k)-f(x_{k-1})}
   $$

3. Stop when $|x_{k+1}-x_k|<\varepsilon$ or $|f(x_{k+1})|<\varepsilon$.

#### Tiny Code

Python

```python
def secant(f, x0, x1, tol=1e-10, max_iter=100):
    f0, f1 = f(x0), f(x1)
    for _ in range(max_iter):
        if f1 == f0:
            raise ZeroDivisionError("Division by zero in secant method")
        x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
        if abs(x2 - x1) < tol:
            return x2
        x0, x1 = x1, x2
        f0, f1 = f1, f(x1)
    return x1

# Example: root of x^3 - x - 2 = 0
root = secant(lambda x: x3 - x - 2, 1, 2)
print(root)  # ~1.5213797
```

C

```c
#include <stdio.h>
#include <math.h>

double f(double x) { return x*x*x - x - 2; }

double secant(double x0, double x1, double tol, int max_iter) {
    double f0 = f(x0), f1 = f(x1);
    for (int i = 0; i < max_iter; i++) {
        if (f1 == f0) break;
        double x2 = x1 - f1 * (x1 - x0) / (f1 - f0);
        if (fabs(x2 - x1) < tol) return x2;
        x0 = x1; f0 = f1;
        x1 = x2; f1 = f(x1);
    }
    return x1;
}

int main(void) {
    printf("%.10f\n", secant(1, 2, 1e-10, 100));
    return 0;
}
```

#### Why It Matters

- No derivative needed (uses finite differences)
- Faster than Bisection (superlinear, ≈1.618 order)
- Commonly used when $f'(x)$ is expensive or unavailable
- A stepping stone to hybrid methods (e.g. Brent's method)

#### A Gentle Proof (Why It Works)

Newton's method update:

$$
x_{k+1}=x_k-\frac{f(x_k)}{f'(x_k)}
$$

Approximate $f'(x_k)$ via difference quotient:

$$
f'(x_k)\approx\frac{f(x_k)-f(x_{k-1})}{x_k-x_{k-1}}
$$

Substitute into Newton's update:

$$
x_{k+1}=x_k-f(x_k)\frac{x_k-x_{k-1}}{f(x_k)-f(x_{k-1})}
$$

Thus, it's Newton's method without explicit derivatives, retaining superlinear convergence when close to the root.

#### Try It Yourself

1. Solve $x^3-x-2=0$ with $(x_0,x_1)=(1,2)$.
2. Compare iteration count vs Newton ($x_0=1$).
3. Try $\cos x - x=0$ with $(0,1)$.
4. Observe failure when $f(x_k)=f(x_{k-1})$.
5. Add fallback to bisection for robustness.

#### Test Cases

| Function     | Initial $(x_0,x_1)$ | Root (approx) | Iterations (ε=1e-10) |
| ------------ | ------------------- | ------------- | -------------------- |
| $x^2-2$      | (1,2)               | 1.41421356    | 6                    |
| $x^3-x-2$    | (1,2)               | 1.5213797     | 7                    |
| $\cos x - x$ | (0,1)               | 0.73908513    | 6                    |

#### Complexity

- Time: $O(k)$ iterations, each $O(1)$ (1 function eval per step)
- Convergence: Superlinear ($\approx1.618$ order)
- Space: $O(1)$

The Secant Method blends the speed of Newton's method with the simplicity of bisection, a derivative-free bridge between theory and practicality.

### 574 Fixed-Point Iteration

Fixed-Point Iteration is a general method for solving equations of the form $x=g(x)$. Instead of finding where $f(x)=0$, we repeatedly apply a transformation that should converge to a stable point, the fixed point where input equals output.

#### What Problem Are We Solving?

We want to find $x^*$ such that

$$
x^* = g(x^*)
$$

If we can express a problem $f(x)=0$ as $x=g(x)$, then the solution of one is the fixed point of the other.
For example, solving $x^2-2=0$ is equivalent to solving

$$
x = \sqrt{2}
$$

or in iterative form,

$$
x_{k+1} = g(x_k) = \frac{1}{2}\left(x_k + \frac{2}{x_k}\right)
$$

#### How Does It Work (Plain Language)

Start with an initial guess $x_0$, then keep applying $g$:

$$
x_{k+1} = g(x_k)
$$

If $g$ is well-behaved (a contraction near $x^*$), this sequence will settle closer and closer to the fixed point.

You can think of it like repeatedly bouncing toward equilibrium, each bounce gets smaller until you land at $x^*$.

#### Convergence Condition

Fixed-point iteration converges if $g$ is continuous near $x^*$ and

$$
|g'(x^*)| < 1
$$

If $|g'(x^*)| > 1$, the iteration diverges.

This means the slope of $g$ at the fixed point must not be too steep, it should "pull" you in rather than push you away.

#### Algorithm Steps

1. Choose initial guess $x_0$.
2. Compute $x_{k+1} = g(x_k)$.
3. Stop if $|x_{k+1} - x_k| < \varepsilon$ or after max iterations.
4. Return $x_{k+1}$ as the approximate root.

#### Tiny Code

Python

```python
def fixed_point(g, x0, tol=1e-10, max_iter=100):
    x = x0
    for _ in range(max_iter):
        x_new = g(x)
        if abs(x_new - x) < tol:
            return x_new
        x = x_new
    return x

# Example: Solve x = cos(x)
root = fixed_point(lambda x: math.cos(x), 0.5)
print(root)  # ~0.739085
```

C

```c
#include <stdio.h>
#include <math.h>

double g(double x) { return cos(x); }

double fixed_point(double x0, double tol, int max_iter) {
    double x = x0, x_new;
    for (int i = 0; i < max_iter; i++) {
        x_new = g(x);
        if (fabs(x_new - x) < tol) return x_new;
        x = x_new;
    }
    return x;
}

int main(void) {
    printf("%.10f\n", fixed_point(0.5, 1e-10, 100));
    return 0;
}
```

#### Why It Matters

- Foundation for Newton–Raphson, Gauss–Seidel, and many nonlinear solvers.
- Conceptually simple and general.
- Shows how convergence depends on transformation design, not just the function.

#### A Gentle Proof (Why It Works)

Suppose $x^*$ is a fixed point, $g(x^*)=x^*$.
Expand $g$ near $x^*$ by Taylor approximation:

$$
g(x) = g(x^*) + g'(x^*)(x - x^*) + O((x-x^*)^2)
$$

Subtract $x^*$:

$$
x_{k+1}-x^* = g'(x^*)(x_k-x^*) + O((x_k-x^*)^2)
$$

So for small errors,
$$
|x_{k+1}-x^*| \approx |g'(x^*)||x_k-x^*|
$$

If $|g'(x^*)|<1$, the error shrinks every iteration, geometric convergence.

#### Try It Yourself

1. Solve $x=\cos(x)$ starting at $x_0=0.5$.
2. Solve $x=\sqrt{2}$ using $x_{k+1}=\frac{1}{2}(x_k+\frac{2}{x_k})$.
3. Try $x=g(x)=1+\frac{1}{x}$, watch it diverge.
4. Experiment with transformations of $f(x)=x^3-x-2$ into different $g(x)$.
5. Observe sensitivity to initial guesses and slope.

#### Test Cases

| $g(x)$                       | Start $x_0$ | Result (approx) | Converges? |
| ---------------------------- | ----------- | --------------- | ---------- |
| $\cos x$                     | 0.5         | 0.739085        | Yes        |
| $\frac{1}{2}(x+\frac{2}{x})$ | 1           | 1.41421356      | Yes        |
| $1+\frac{1}{x}$              | 1           | Divergent       | No         |

#### Complexity

- Per iteration: $O(1)$ (one function evaluation)
- Convergence: Linear (if $|g'(x^*)|<1$)
- Space: $O(1)$

Fixed-point iteration is the gentle heartbeat of numerical solving, simple, geometric, and foundational to modern root-finding and optimization algorithms.

### 575 Gaussian Quadrature

Gaussian Quadrature is a high-accuracy numerical integration method that approximates
$$
I=\int_a^b f(x),dx
$$
by evaluating $f(x)$ at optimally chosen points (nodes) and weighting them carefully.
It achieves maximum precision for a given number of evaluations, far more accurate than trapezoidal or Simpson's rules for smooth functions.

#### What Problem Are We Solving?

We want to approximate an integral numerically, but instead of equally spaced sample points, we'll choose the best possible points to minimize error.

Traditional methods sample uniformly, but Gaussian quadrature uses orthogonal polynomials (like Legendre, Chebyshev, Laguerre, or Hermite) to determine ideal nodes $x_i$ and weights $w_i$ that make the rule exact for all polynomials up to degree $2n-1$.

Formally,

$$
\int_a^b f(x),dx \approx \sum_{i=1}^n w_i,f(x_i)
$$

#### How Does It Work (Plain Language)

1. Choose a set of orthogonal polynomials on $[a,b]$ (often Legendre for standard integrals).
2. The roots of the $n$-th polynomial become the sample points $x_i$.
3. Compute corresponding weights $w_i$ so that the formula integrates all polynomials up to degree $2n-1$ exactly.
4. Evaluate $f(x_i)$, multiply by weights, and sum.

Each $x_i$ and $w_i$ is precomputed, you just plug in your function.

#### Example: Gauss–Legendre Quadrature on $[-1,1]$

For $n=2$:
$$
x_1=-\frac{1}{\sqrt{3}}, \quad x_2=\frac{1}{\sqrt{3}}, \quad w_1=w_2=1.
$$

Thus,
$$
\int_{-1}^{1} f(x),dx \approx f(-1/\sqrt{3})+f(1/\sqrt{3}).
$$

For arbitrary $[a,b]$, map via
$$
t=\frac{b-a}{2}x+\frac{a+b}{2},
$$
and scale the result by $\frac{b-a}{2}$.

#### Tiny Code

Python

```python
import numpy as np

def gauss_legendre(f, a, b, n=2):
    # nodes and weights for n=2, extendable for larger n
    x = np.array([-1/np.sqrt(3), 1/np.sqrt(3)])
    w = np.array([1.0, 1.0])
    # transform to [a,b]
    t = 0.5*(b - a)*x + 0.5*(b + a)
    return 0.5*(b - a)*np.sum(w * f(t))

# Example: integrate f(x)=x^2 from 0 to 1
result = gauss_legendre(lambda x: x2, 0, 1)
print(result)  # ~0.333333
```

C

```c
#include <stdio.h>
#include <math.h>

double f(double x) { return x*x; }

double gauss_legendre(double a, double b) {
    double x1 = -1.0/sqrt(3.0), x2 = 1.0/sqrt(3.0);
    double w1 = 1.0, w2 = 1.0;
    double t1 = 0.5*(b - a)*x1 + 0.5*(b + a);
    double t2 = 0.5*(b - a)*x2 + 0.5*(b + a);
    return 0.5*(b - a)*(w1*f(t1) + w2*f(t2));
}

int main(void) {
    printf("%.6f\n", gauss_legendre(0, 1)); // 0.333333
    return 0;
}
```

#### Why It Matters

- High precision with few points
- Exact for all polynomials up to degree $2n-1$
- Works beautifully for smooth integrands
- Forms the foundation for spectral methods, finite elements, and probability integration

#### A Gentle Proof (Why It Works)

Let $p_n(x)$ be an orthogonal polynomial of degree $n$ on $[a,b]$ with respect to a weight function $w(x)$.
Its $n$ roots $x_i$ satisfy orthogonality:

$$
\int_a^b p_n(x),p_m(x),w(x),dx=0 \quad (m<n).
$$

If $f$ is a polynomial of degree $\le2n-1$, it can be decomposed into terms up to $p_{2n-1}(x)$, and integrating with the quadrature rule using these roots yields the exact value.

Thus, Gaussian quadrature minimizes integration error within polynomial spaces.

#### Try It Yourself

1. Integrate $\sin x$ from $0$ to $\pi/2$ using 2- and 3-point Gauss–Legendre.
2. Compare with Simpson's rule.
3. Extend to $n=3$ using precomputed nodes and weights.
4. Try $f(x)=e^x$ over $[0,1]$.
5. Experiment with scaling to non-standard intervals $[a,b]$.

#### Test Cases

| Function | Interval | $n$ | Result (approx) | True Value |
| -------- | -------- | --- | --------------- | ---------- |
| $x^2$    | [0,1]    | 2   | 0.333333        | 1/3        |
| $\sin x$ | [0,π/2]  | 2   | 0.99984         | 1.00000    |
| $e^x$    | [0,1]    | 2   | 1.71828         | 1.71828    |

#### Complexity

- Time: $O(n)$ evaluations of $f$
- Space: $O(n)$ for nodes and weights
- Accuracy: exact for all polynomials of degree ≤ $2n-1$

Gaussian Quadrature shows how pure mathematics and computation intertwine, orthogonal polynomials guiding us to integrate with surgical precision.

### 576 Simpson's Rule

Simpson's Rule is a classical numerical integration method that approximates the integral of a smooth function using parabolas rather than straight lines. It combines the simplicity of the trapezoidal rule with higher-order accuracy, making it one of the most practical methods for evenly spaced data.

#### What Problem Are We Solving?

We want to approximate the definite integral

$$
I = \int_a^b f(x),dx
$$

when we only know $f(x)$ at discrete points or cannot find an analytic antiderivative.

Instead of approximating $f$ by a straight line between each pair of points, Simpson's Rule uses quadratic interpolation through every two subintervals for much higher accuracy.

#### How Does It Work (Plain Language)

Imagine you take three points $(x_0,f(x_0))$, $(x_1,f(x_1))$, $(x_2,f(x_2))$ equally spaced by $h$.
Fit a parabola through them, integrate that parabola, and repeat.

For a single parabolic arc:

$$
\int_{x_0}^{x_2} f(x),dx \approx \frac{h}{3}\big[f(x_0) + 4f(x_1) + f(x_2)\big]
$$

For $n$ subintervals (where $n$ is even):

$$
I \approx \frac{h}{3}\Big[f(x_0) + 4\sum_{i=1,3,5,\ldots}^{n-1} f(x_i) + 2\sum_{i=2,4,6,\ldots}^{n-2} f(x_i) + f(x_n)\Big]
$$

where $h = \frac{b - a}{n}$.

#### Tiny Code

Python

```python
def simpson(f, a, b, n=100):
    if n % 2 == 1:
        n += 1  # must be even
    h = (b - a) / n
    s = f(a) + f(b)
    for i in range(1, n):
        x = a + i * h
        s += 4 * f(x) if i % 2 else 2 * f(x)
    return s * h / 3

# Example: integrate sin(x) from 0 to π
import math
res = simpson(math.sin, 0, math.pi, n=100)
print(res)  # ~2.000000
```

C

```c
#include <stdio.h>
#include <math.h>

double f(double x) { return sin(x); }

double simpson(double a, double b, int n) {
    if (n % 2 == 1) n++; // must be even
    double h = (b - a) / n;
    double s = f(a) + f(b);
    for (int i = 1; i < n; i++) {
        double x = a + i * h;
        s += (i % 2 ? 4 : 2) * f(x);
    }
    return s * h / 3.0;
}

int main(void) {
    printf("%.6f\n", simpson(0, M_PI, 100)); // 2.000000
    return 0;
}
```

#### Why It Matters

- Accurate and simple for smooth functions
- Exact for cubic polynomials
- Often the best balance between accuracy and computation for tabulated data
- Forms the backbone of adaptive and composite integration schemes

#### A Gentle Proof (Why It Works)

Let $f(x)$ be approximated by a quadratic polynomial
$$
p(x) = ax^2 + bx + c
$$
that passes through $(x_0,f_0)$, $(x_1,f_1)$, $(x_2,f_2)$.
Integrating $p(x)$ over $[x_0,x_2]$ yields

$$
\int_{x_0}^{x_2} p(x),dx = \frac{h}{3}\big[f_0 + 4f_1 + f_2\big].
$$

Because the error term depends on $f^{(4)}(\xi)$, Simpson's Rule is fourth-order accurate, i.e.

$$
E = -\frac{(b-a)}{180}h^4f^{(4)}(\xi)
$$

for some $\xi\in[a,b]$.

#### Try It Yourself

1. Integrate $\sin x$ from $0$ to $\pi$ (should be $2$).
2. Integrate $x^4$ from $0$ to $1$, check exactness for polynomials ≤ degree 3.
3. Compare with trapezoidal rule for the same $n$.
4. Experiment with uneven $n$ and verify convergence.
5. Implement adaptive Simpson's rule for automatic refinement.

#### Test Cases

| Function | Interval | $n$ | Simpson Result | True Value |
| -------- | -------- | --- | -------------- | ---------- |
| $\sin x$ | [0, π]   | 100 | 1.999999       | 2.000000   |
| $x^2$    | [0, 1]   | 10  | 0.333333       | 1/3        |
| $e^x$    | [0, 1]   | 20  | 1.718282       | 1.718282   |

#### Complexity

- Time: $O(n)$ evaluations of $f(x)$
- Accuracy: $O(h^4)$
- Space: $O(1)$

Simpson's Rule is the perfect blend of simplicity and precision, a parabolic leap beyond straight-line approximations.

### 577 Trapezoidal Rule

The Trapezoidal Rule is one of the simplest numerical integration methods. It approximates the area under a curve by dividing it into trapezoids instead of rectangles, providing a linear interpolation between sampled points.

#### What Problem Are We Solving?

We want to estimate

$$
I = \int_a^b f(x),dx
$$

when we only know $f(x)$ at discrete points, or when the integral has no simple analytical form.
The idea is to approximate $f(x)$ as piecewise linear between each pair of neighboring points.

#### How Does It Work (Plain Language)

If you know $f(a)$ and $f(b)$, the simplest estimate is the area of a trapezoid:

$$
I \approx \frac{b-a}{2},[f(a)+f(b)].
$$

For multiple subintervals of equal width $h=(b-a)/n$:

$$
I \approx \frac{h}{2}\Big[f(x_0)+2\sum_{i=1}^{n-1}f(x_i)+f(x_n)\Big].
$$

Each pair of consecutive points defines a trapezoid, we sum their areas to approximate the total.

#### Tiny Code

Python

```python
def trapezoidal(f, a, b, n=100):
    h = (b - a) / n
    s = 0.5 * (f(a) + f(b))
    for i in range(1, n):
        s += f(a + i * h)
    return s * h

# Example: integrate e^x from 0 to 1
import math
res = trapezoidal(math.exp, 0, 1, 100)
print(res)  # ~1.718282
```

C

```c
#include <stdio.h>
#include <math.h>

double f(double x) { return exp(x); }

double trapezoidal(double a, double b, int n) {
    double h = (b - a) / n;
    double s = 0.5 * (f(a) + f(b));
    for (int i = 1; i < n; i++) {
        s += f(a + i * h);
    }
    return s * h;
}

int main(void) {
    printf("%.6f\n", trapezoidal(0, 1, 100)); // 1.718282
    return 0;
}
```

#### Why It Matters

- Simple and widely used as a first numerical integration method
- Robust for smooth functions
- Serves as a building block for Simpson's rule and Romberg integration
- Works directly with tabulated data

#### A Gentle Proof (Why It Works)

Over one subinterval $[x_i,x_{i+1}]$, approximate $f(x)$ by a straight line:

$$
f(x)\approx f(x_i)+\frac{f(x_{i+1})-f(x_i)}{h}(x-x_i).
$$

Integrating this line exactly gives

$$
\int_{x_i}^{x_{i+1}} f(x),dx \approx \frac{h}{2}[f(x_i)+f(x_{i+1})].
$$

Summing across all intervals yields the composite trapezoidal rule.

The error term for one interval is proportional to the curvature of $f$:

$$
E = -\frac{(b-a)}{12}h^2f''(\xi)
$$

for some $\xi\in[a,b]$. Thus, it's second-order accurate.

#### Try It Yourself

1. Integrate $\sin x$ from $0$ to $\pi$ (result $\approx2$).
2. Integrate $x^2$ from $0$ to $1$ and compare with exact $1/3$.
3. Compare error with Simpson's Rule for the same $n$.
4. Try $f(x)=1/x$ on $[1,2]$.
5. Explore how halving $h$ affects accuracy.

#### Test Cases

| Function | Interval | $n$ | Trapezoidal | True Value |
| -------- | -------- | --- | ----------- | ---------- |
| $\sin x$ | [0, π]   | 100 | 1.9998      | 2.0000     |
| $x^2$    | [0,1]    | 100 | 0.33335     | 1/3        |
| $e^x$    | [0,1]    | 100 | 1.71828     | 1.71828    |

#### Complexity

- Time: $O(n)$ function evaluations
- Accuracy: $O(h^2)$
- Space: $O(1)$

The Trapezoidal Rule is the entryway to numerical integration, intuitive, reliable, and surprisingly effective when used with fine discretization or smooth functions.

### 578 Runge–Kutta (RK4) Method

The Runge–Kutta (RK4) method is one of the most widely used techniques for numerically solving ordinary differential equations (ODEs). It provides a beautiful balance between accuracy and computational simplicity, using multiple slope evaluations within each step to achieve fourth-order precision.

#### What Problem Are We Solving?

We want to solve an initial value problem (IVP):

$$
\frac{dy}{dx}=f(x,y), \quad y(x_0)=y_0
$$

When no analytical solution exists (or is hard to find), RK4 approximates $y(x)$ step by step, with high accuracy.

#### How Does It Work (Plain Language)

Instead of taking a single slope per step (like Euler's method), RK4 samples four slopes and combines them:

$$
\begin{aligned}
k_1 &= f(x_n, y_n),\
k_2 &= f(x_n + h/2,, y_n + h k_1/2),\
k_3 &= f(x_n + h/2,, y_n + h k_2/2),\
k_4 &= f(x_n + h,, y_n + h k_3),\
y_{n+1} &= y_n + \frac{h}{6}(k_1 + 2k_2 + 2k_3 + k_4).
\end{aligned}
$$

This weighted average captures both curvature and local behavior with surprising precision.

#### Tiny Code

Python

```python
def rk4(f, x0, y0, h, n):
    x, y = x0, y0
    for _ in range(n):
        k1 = f(x, y)
        k2 = f(x + h/2, y + h*k1/2)
        k3 = f(x + h/2, y + h*k2/2)
        k4 = f(x + h, y + h*k3)
        y += h*(k1 + 2*k2 + 2*k3 + k4)/6
        x += h
    return x, y

# Example: dy/dx = y, y(0) = 1 → true solution y = e^x
import math
f = lambda x, y: y
x, y = rk4(f, 0, 1, 0.1, 10)
print(y, "vs", math.e)
```

C

```c
#include <stdio.h>
#include <math.h>

double f(double x, double y) { return y; } // dy/dx = y

void rk4(double x0, double y0, double h, int n) {
    double x = x0, y = y0;
    for (int i = 0; i < n; i++) {
        double k1 = f(x, y);
        double k2 = f(x + h/2, y + h*k1/2);
        double k3 = f(x + h/2, y + h*k2/2);
        double k4 = f(x + h, y + h*k3);
        y += h*(k1 + 2*k2 + 2*k3 + k4)/6.0;
        x += h;
    }
    printf("x=%.2f, y=%.6f\n", x, y);
}

int main(void) {
    rk4(0, 1, 0.1, 10); // y(1) ≈ e ≈ 2.718282
    return 0;
}
```

#### Why It Matters

- Fourth-order accuracy without complex derivatives
- Used in physics, engineering, and machine learning (ODE solvers)
- Foundation for adaptive step-size solvers and neural ODEs
- Much more stable and accurate than Euler or midpoint methods

#### A Gentle Proof (Why It Works)

Expand the true solution with Taylor series:

$$
y(x+h)=y(x)+h y'(x)+\frac{h^2}{2}y''(x)+\frac{h^3}{6}y^{(3)}(x)+O(h^4)
$$

RK4's combination of $k_1,k_2,k_3,k_4$ reproduces all terms up to $h^4$, giving global error $O(h^4)$.

Each $k_i$ estimates the derivative at intermediate points, forming a weighted average that captures local curvature.

#### Try It Yourself

1. Solve $dy/dx = y$ from $x=0$ to $x=1$ with $h=0.1$.
2. Compare with Euler's method.
3. Try $dy/dx = -2y$ with $y(0)=1$.
4. Visualize $k_1,k_2,k_3,k_4$ on the slope field.
5. Test with nonlinear $f(x,y)=x^2+y^2$.

#### Test Cases

| Differential Equation | Interval | Step h | Result  | True Value      |
| --------------------- | -------- | ------ | ------- | --------------- |
| $dy/dx=y$             | [0,1]    | 0.1    | 2.71828 | $e$             |
| $dy/dx=-2y$           | [0,1]    | 0.1    | 0.13534 | $e^{-2}$        |
| $dy/dx=x+y$           | [0,1]    | 0.1    | 2.7183  | analytic $2e-1$ |

#### Complexity

- Time: $O(n)$ evaluations of $f$ (4 per step)
- Accuracy: Global error $O(h^4)$
- Space: $O(1)$

The Runge–Kutta method is a masterpiece of numerical design, simple enough to code in minutes, powerful enough to drive modern simulations and control systems.

### 579 Euler's Method

Euler's Method is the simplest numerical procedure for solving ordinary differential equations (ODEs). It's the "hello world" of numerical integration, conceptually clear, easy to implement, and forms the foundation for more advanced methods like Runge–Kutta.

#### What Problem Are We Solving?

We want to approximate the solution of an initial value problem:

$$
\frac{dy}{dx}=f(x,y), \quad y(x_0)=y_0.
$$

If we can't solve this analytically, we can approximate $y(x)$ at discrete points using small steps $h$.

#### How Does It Work (Plain Language)

The key idea: use the slope at the current point to predict the next one.

At each step:

$$
y_{n+1} = y_n + h f(x_n, y_n),
$$

and

$$
x_{n+1} = x_n + h.
$$

This is just "take a small step along the tangent." The smaller $h$, the better the approximation.

#### Tiny Code

Python

```python
def euler(f, x0, y0, h, n):
    x, y = x0, y0
    for _ in range(n):
        y += h * f(x, y)
        x += h
    return x, y

# Example: dy/dx = y, y(0)=1 -> y=e^x
import math
f = lambda x, y: y
x, y = euler(f, 0, 1, 0.1, 10)
print(y, "vs", math.e)
```

C

```c
#include <stdio.h>
#include <math.h>

double f(double x, double y) { return y; }

void euler(double x0, double y0, double h, int n) {
    double x = x0, y = y0;
    for (int i = 0; i < n; i++) {
        y += h * f(x, y);
        x += h;
    }
    printf("x=%.2f, y=%.6f\n", x, y);
}

int main(void) {
    euler(0, 1, 0.1, 10); // y(1) ≈ 2.5937 vs e ≈ 2.7183
    return 0;
}
```

#### Why It Matters

- The first step into numerical ODEs
- Simple, intuitive, and educational
- Shows the tradeoff between step size and accuracy
- Used as a building block for Runge–Kutta, Heun's, and predictor–corrector methods

#### A Gentle Proof (Why It Works)

Using Taylor expansion:

$$
y(x+h) = y(x) + h y'(x) + \frac{h^2}{2} y''(\xi)
$$

Since $y'(x)=f(x,y)$, Euler's method approximates by ignoring higher-order terms:

$$
y_{n+1} \approx y_n + h f(x_n, y_n).
$$

The local error is $O(h^2)$, and the global error is $O(h)$.

#### Try It Yourself

1. Solve $dy/dx=y$ from $0$ to $1$ with $h=0.1$.
2. Decrease $h$ and observe convergence toward $e$.
3. Try $dy/dx=-2y$ and plot exponential decay.
4. Compare with Runge–Kutta for same step size.
5. Implement a version that stores and plots all $(x,y)$ pairs.

#### Test Cases

| Differential Equation | Interval | Step $h$ | Euler Result | True Value |
| --------------------- | -------- | -------- | ------------ | ---------- |
| $dy/dx=y$             | [0,1]    | 0.1      | 2.5937       | 2.7183     |
| $dy/dx=-2y$           | [0,1]    | 0.1      | 0.1615       | 0.1353     |
| $dy/dx=x+y$           | [0,1]    | 0.1      | 2.65         | 2.7183     |

#### Complexity

- Time: $O(n)$
- Accuracy: Global $O(h)$
- Space: $O(1)$

Euler's Method is the simplest step into numerical dynamics, a straight line drawn into the curved world of differential equations.

### 580 Gradient Descent (1D Numerical Optimization)

Gradient Descent is a simple yet powerful iterative algorithm for finding minima of differentiable functions. In one dimension, it's an intuitive process: move in the opposite direction of the slope until the function stops decreasing.

#### What Problem Are We Solving?

We want to find the local minimum of a real-valued function $f(x)$, that is:

$$
\min_x f(x)
$$

If $f'(x)$ exists but solving $f'(x)=0$ analytically is difficult, we can approach the minimum step-by-step using the gradient (slope).

#### How Does It Work (Plain Language)

At each iteration, move opposite to the derivative, scaled by a learning rate $\eta$:

$$
x_{t+1} = x_t - \eta f'(x_t)
$$

The derivative $f'(x_t)$ points uphill; subtracting it moves downhill. The step size $\eta$ determines how far we move.

- If $\eta$ is too small, convergence is slow.
- If $\eta$ is too large, we may overshoot or diverge.

The process repeats until $|f'(x_t)|$ becomes very small.

#### Tiny Code

Python

```python
def gradient_descent_1d(df, x0, eta=0.1, tol=1e-6, max_iter=1000):
    x = x0
    for _ in range(max_iter):
        grad = df(x)
        if abs(grad) < tol:
            break
        x -= eta * grad
    return x

# Example: minimize f(x) = x^2 -> df/dx = 2x
f_prime = lambda x: 2*x
x_min = gradient_descent_1d(f_prime, x0=5)
print(x_min)  # ≈ 0
```

C

```c
#include <stdio.h>
#include <math.h>

double df(double x) { return 2*x; } // derivative of x^2

double gradient_descent(double x0, double eta, double tol, int max_iter) {
    double x = x0;
    for (int i = 0; i < max_iter; i++) {
        double grad = df(x);
        if (fabs(grad) < tol) break;
        x -= eta * grad;
    }
    return x;
}

int main(void) {
    double xmin = gradient_descent(5.0, 0.1, 1e-6, 1000);
    printf("x_min = %.6f\n", xmin);
    return 0;
}
```

#### Why It Matters

- Fundamental to optimization, machine learning, and deep learning
- Scales naturally from 1D to high dimensions
- Helps visualize energy landscapes, convergence, and learning dynamics

#### A Gentle Proof (Why It Works)

For convex and smooth $f(x)$, the Taylor expansion gives:

$$
f(x-\eta f'(x)) \approx f(x) - \eta (f'(x))^2 + \frac{\eta^2}{2} f''(\xi) (f'(x))^2
$$

For small $\eta>0$ and $f''(\xi)>0$, the term $-\eta (f'(x))^2$ dominates, ensuring that $f(x)$ decreases with each step.

This shows convergence toward the stationary point where $f'(x)=0$.

#### Try It Yourself

1. $f(x)=x^2$ with $\eta=0.1$, starting from $x_0=5$.
2. Try $\eta=0.01$ and $\eta=1.0$, observe convergence or divergence.
3. Minimize $f(x)=(x-3)^2$; check how quickly $x\to3$.
4. Visualize each iteration on a plot of $f(x)$.
5. Modify for stochastic or momentum variants.

#### Test Cases

| Function  | Derivative | Start $x_0$ | $\eta$ | Result | True Minimum |
| --------- | ---------- | ----------- | ------ | ------ | ------------ |
| $x^2$     | $2x$       | 5           | 0.1    | 0.0000 | 0            |
| $(x-3)^2$ | $2(x-3)$   | 0           | 0.1    | 3.0000 | 3            |
| $\cos x$  | $-\sin x$  | 2           | 0.05   | 1.57   | $\pi/2$      |

#### Complexity

- Time: $O(k)$ iterations (depends on step size and tolerance)
- Accuracy: depends on $\eta$ and smoothness of $f$
- Space: $O(1)$

Gradient Descent is the simplest form of learning, a gentle step downhill that embodies the heart of optimization and machine intelligence.

# Section 59. Mathematicial Optimization 

### 581 Simplex Method

The Simplex Method is a cornerstone of mathematical optimization, a geometric algorithm for solving linear programming (LP) problems efficiently. It walks along the vertices of the feasible region (a convex polytope) until it reaches the optimal vertex, where the objective function is maximized or minimized.

#### What Problem Are We Solving?

We want to optimize a linear objective function subject to linear constraints:

$$
\text{maximize } z = c_1x_1 + c_2x_2 + \dots + c_nx_n
$$

subject to

$$
\begin{aligned}
a_{11}x_1 + a_{12}x_2 + \dots + a_{1n}x_n &\le b_1, \
a_{21}x_1 + a_{22}x_2 + \dots + a_{2n}x_n &\le b_2, \
&\vdots \
a_{m1}x_1 + a_{m2}x_2 + \dots + a_{mn}x_n &\le b_m, \
x_i &\ge 0.
\end{aligned}
$$

This is the standard form of a linear program.

The goal is to find $(x_1,\dots,x_n)$ that gives the maximum (or minimum) value of $z$.

#### How Does It Work (Plain Language)

Imagine all constraints forming a polygon (in 2D) or polyhedron (in higher dimensions).
The feasible region is convex, so the optimal point always lies on a vertex.

The Simplex method:

1. Starts at one vertex (feasible basic solution).
2. Moves along edges to an adjacent vertex that improves the objective function.
3. Repeats until no further improvement is possible, that vertex is optimal.

#### Algebraic View

1. Convert inequalities to equalities using slack variables.
   Example:
   $x_1 + x_2 \le 4$ → $x_1 + x_2 + s_1 = 4$, where $s_1 \ge 0$.

2. Represent the system in tableau form.

3. Perform pivot operations (like Gaussian elimination) to move from one basic feasible solution to another.

4. Stop when all reduced costs in the objective row are non-negative (for maximization).

#### Tiny Code (Simplified Demonstration)

Python (educational version)

```python
import numpy as np

def simplex(c, A, b):
    m, n = A.shape
    tableau = np.zeros((m+1, n+m+1))
    tableau[:m, :n] = A
    tableau[:m, n:n+m] = np.eye(m)
    tableau[:m, -1] = b
    tableau[-1, :n] = -c

    while True:
        col = np.argmin(tableau[-1, :-1])
        if tableau[-1, col] >= 0:
            break  # optimal
        ratios = [tableau[i, -1] / tableau[i, col] if tableau[i, col] > 0 else np.inf for i in range(m)]
        row = np.argmin(ratios)
        pivot = tableau[row, col]
        tableau[row, :] /= pivot
        for i in range(m+1):
            if i != row:
                tableau[i, :] -= tableau[i, col] * tableau[row, :]
    return tableau[-1, -1]

# Example: maximize z = 3x1 + 2x2
# subject to: x1 + x2 ≤ 4, x1 ≤ 2, x2 ≤ 3
c = np.array([3, 2])
A = np.array([[1, 1], [1, 0], [0, 1]])
b = np.array([4, 2, 3])
print("Max z =", simplex(c, A, b))
```

#### Why It Matters

- Foundation of operations research, economics, and optimization theory.
- Used in logistics, finance, resource allocation, and machine learning (e.g., SVMs).
- Despite exponential worst-case complexity, it is extremely fast in practice.

#### A Gentle Proof (Why It Works)

Because linear programs are convex, the optimum (if it exists) must occur at a vertex of the feasible region.
The Simplex algorithm explores vertices in a way that strictly improves the objective function until reaching a vertex with no improving adjacent vertices.

This corresponds to the condition that all reduced costs in the tableau are non-negative, indicating optimality.

#### Try It Yourself

1. Solve the LP: maximize $z = 3x_1 + 5x_2$
   subject to:
   $$
   \begin{cases}
   2x_1 + x_2 \le 8 \
   x_1 + 2x_2 \le 8 \
   x_1, x_2 \ge 0
   \end{cases}
   $$

2. Draw the feasible region and verify the optimal vertex.

3. Modify constraints and observe how the solution moves.

4. Experiment with minimization by negating the objective.

#### Test Cases

| Objective   | Constraints                       | Result $(x_1,x_2)$ | Max $z$ |
| ----------- | --------------------------------- | ------------------ | ------- |
| $3x_1+2x_2$ | $x_1+x_2\le4,\ x_1\le2,\ x_2\le3$ | (1,3)              | 9       |
| $2x_1+3x_2$ | $2x_1+x_2\le8,\ x_1+2x_2\le8$     | (2,3)              | 13      |
| $x_1+x_2$   | $x_1,x_2\le5$                     | (5,5)              | 10      |

#### Complexity

- Time: Polynomial in practice (though worst-case exponential)
- Space: $O(mn)$ for tableau storage

The Simplex Method remains one of the most elegant algorithms ever created, a geometric dance over a convex landscape, always finding the corner of greatest value.

### 582 Dual Simplex Method

The Dual Simplex Method is a close cousin of the Simplex algorithm. While the original Simplex keeps the solution feasible and moves toward optimality, the Dual Simplex does the opposite, it keeps the solution optimal and moves toward feasibility.

It's especially useful when constraints change or when we start with an infeasible solution that already satisfies optimality conditions.

#### What Problem Are We Solving?

We still solve a linear program (LP) in standard form, but we start from a tableau that is dual feasible (objective is optimal) but primal infeasible (some right-hand sides are negative).

Maximize

$$
z = c^T x
$$

subject to

$$
A x \le b, \quad x \ge 0.
$$

The dual simplex method works to restore feasibility while maintaining optimality of the reduced costs.

#### How Does It Work (Plain Language)

Think of the Simplex and Dual Simplex as mirror images:

| Step         | Simplex                    | Dual Simplex                  |
| ------------ | -------------------------- | ----------------------------- |
| Keeps        | Feasible solution          | Optimal reduced costs         |
| Fixes        | Optimality                 | Feasibility                   |
| Pivot choice | Most negative reduced cost | Most negative right-hand side |

In the Dual Simplex, at each iteration:

1. Identify a row with a negative RHS (violating feasibility).
2. Among its coefficients, choose a pivot column that keeps reduced costs non-negative after pivoting.
3. Perform a pivot to make that constraint feasible.
4. Repeat until all RHS entries are non-negative (fully feasible).

#### Algebraic Formulation

If we maintain tableau:

$$
\begin{bmatrix}
A & I & b \
c^T & 0 & z
\end{bmatrix}
$$

then the pivot condition becomes:

- Choose row $r$ with $b_r < 0$.
- Choose column $s$ where $a_{rs} < 0$ and $\frac{c_s}{a_{rs}}$ is minimal.
- Pivot on $(r,s)$.

This restores primal feasibility step by step while maintaining dual optimality.

#### Tiny Code (Illustrative Example)

Python

```python
import numpy as np

def dual_simplex(A, b, c):
    m, n = A.shape
    tableau = np.zeros((m+1, n+m+1))
    tableau[:m, :n] = A
    tableau[:m, n:n+m] = np.eye(m)
    tableau[:m, -1] = b
    tableau[-1, :n] = -c

    while np.any(tableau[:-1, -1] < 0):
        r = np.argmin(tableau[:-1, -1])
        ratios = []
        for j in range(n+m):
            if tableau[r, j] < 0:
                ratios.append(tableau[-1, j] / tableau[r, j])
            else:
                ratios.append(np.inf)
        s = np.argmin(ratios)
        tableau[r, :] /= tableau[r, s]
        for i in range(m+1):
            if i != r:
                tableau[i, :] -= tableau[i, s] * tableau[r, :]
    return tableau[-1, -1]

# Example
A = np.array([[1, 1], [2, 1]])
b = np.array([-2, 2])  # infeasible start
c = np.array([3, 2])
print("Optimal value:", dual_simplex(A, b, c))
```

#### Why It Matters

- Efficient for re-optimizing problems after small changes in constraints or RHS.
- Commonly used in branch-and-bound and cutting-plane algorithms for integer programming.
- Avoids recomputation when previous Simplex solutions become infeasible.

#### A Gentle Proof (Why It Works)

At each iteration, the pivot preserves dual feasibility, meaning the reduced costs remain non-negative, and decreases the objective value (for a maximization problem).
When all basic variables become feasible (RHS non-negative), the solution is both feasible and optimal.

Thus, convergence is guaranteed in a finite number of steps for non-degenerate cases.

#### Try It Yourself

1. Start with a Simplex tableau that has a negative RHS.
2. Apply the Dual Simplex pivot rule to fix feasibility.
3. Observe that objective value never increases (for maximization).
4. Compare the path taken with the standard Simplex method.
5. Use it to re-solve a modified LP without starting from scratch.

#### Test Cases

| Objective   | Constraints                  | Method Start      | Result                    |
| ----------- | ---------------------------- | ----------------- | ------------------------- |
| $3x_1+2x_2$ | $x_1+x_2\le2$, $x_1-x_2\ge1$ | infeasible primal | $x_1=1.5, x_2=0.5, z=5.5$ |
| $2x_1+x_2$  | $x_1-x_2\ge2$, $x_1+x_2\le6$ | infeasible primal | $x_1=2, x_2=4, z=8$       |

#### Complexity

- Time: Similar to Simplex, efficient in practice
- Space: $O(mn)$ for tableau

The Dual Simplex is the mirror that balances infeasibility and optimality, a practical algorithm for when the landscape shifts but the solution must adapt gracefully.

### 583 Interior-Point Method

The Interior-Point Method is a modern alternative to the Simplex algorithm for solving linear and convex optimization problems.
Instead of moving along the edges of the feasible region, it moves through the interior, following smooth curves toward the optimal point.

#### What Problem Are We Solving?

We want to solve a standard linear program:

$$
\text{minimize } c^T x
$$

subject to

$$
A x = b, \quad x \ge 0.
$$

This defines a convex feasible region, the intersection of half-spaces and equality constraints.
The interior-point method searches for the optimal point inside that region rather than on its boundary.

#### How Does It Work (Plain Language)

Think of the feasible region as a polyhedron.
Instead of "walking along edges" like the Simplex method, we "slide" through its interior along a smooth central path guided by both the objective and the constraints.

The method uses a barrier function to prevent the solution from crossing boundaries.
For example, to keep $x_i \ge 0$, we add a penalty term $-\mu \sum_i \ln(x_i)$ to the objective.

So we solve:

$$
\text{minimize } c^T x - \mu \sum_i \ln(x_i)
$$

where $\mu > 0$ controls how close we stay to the boundary.
As $\mu \to 0$, the solution approaches the true optimal vertex.

#### Mathematical Steps

1. Barrier problem formulation:
   $$
   \min_x ; c^T x - \mu \sum_{i=1}^n \ln(x_i)
   $$
   subject to $A x = b$.

2. First-order condition:
   $$
   A x = b, \quad X s = \mu e, \quad s = c - A^T y,
   $$
   where $X = \text{diag}(x_1, \dots, x_n)$ and $s$ are slack variables.

3. Newton update:
   Solve the linearized KKT (Karush–Kuhn–Tucker) system for $\Delta x, \Delta y, \Delta s$.

4. Step and update:
   $$
   x \leftarrow x + \alpha \Delta x, \quad y \leftarrow y + \alpha \Delta y, \quad s \leftarrow s + \alpha \Delta s,
   $$
   with step size $\alpha$ chosen to maintain positivity.

5. Reduce $\mu$ and repeat until convergence.

#### Tiny Code (Conceptual Example)

Python (illustrative version)

```python
import numpy as np

def interior_point(A, b, c, mu=1.0, tol=1e-8, max_iter=50):
    m, n = A.shape
    x = np.ones(n)
    y = np.zeros(m)
    s = np.ones(n)

    for _ in range(max_iter):
        r1 = A @ x - b
        r2 = A.T @ y + s - c
        r3 = x * s - mu * np.ones(n)

        if np.linalg.norm(r1) < tol and np.linalg.norm(r2) < tol and np.linalg.norm(r3) < tol:
            break

        # Construct Newton system
        diagX = np.diag(x)
        diagS = np.diag(s)
        M = A @ np.linalg.inv(diagS) @ diagX @ A.T
        rhs = -r1 + A @ np.linalg.inv(diagS) @ (r3 - diagX @ r2)
        dy = np.linalg.solve(M, rhs)
        ds = -r2 - A.T @ dy
        dx = (r3 - diagX @ ds) / s

        # Step size
        alpha = 0.99 * min(1, min(-x[dx < 0] / dx[dx < 0], default=1))
        x += alpha * dx
        y += alpha * dy
        s += alpha * ds
        mu *= 0.5
    return x, y, c @ x

# Example
A = np.array([[1, 1], [1, -1]])
b = np.array([1, 0])
c = np.array([1, 2])
x, y, val = interior_point(A, b, c)
print("Optimal x:", x, "Objective:", val)
```

#### Why It Matters

- Competes with or outperforms Simplex for very large-scale LPs and QPs.
- Smooth and robust, avoids corner-by-corner traversal.
- Forms the foundation of modern convex optimization and machine learning solvers (e.g., SVMs, logistic regression).

#### A Gentle Proof (Why It Works)

The logarithmic barrier ensures that the iterates remain strictly positive.
Each Newton step minimizes a local quadratic approximation of the barrier-augmented objective.
As $\mu \to 0$, the barrier term vanishes, and the solution converges to the true KKT optimal point of the original LP.

#### Try It Yourself

1. Minimize $x_1 + x_2$ subject to
   $$
   \begin{cases}
   x_1 + 2x_2 \ge 2, \
   3x_1 + x_2 \ge 3, \
   x_1, x_2 \ge 0.
   \end{cases}
   $$
2. Compare convergence with the Simplex solution.
3. Experiment with different $\mu$ reduction schedules.
4. Plot trajectory, note the smooth curve through the interior.

#### Test Cases

| Objective  | Constraints                   | Optimal $(x_1,x_2)$ | Min $c^T x$ |
| ---------- | ----------------------------- | ------------------- | ----------- |
| $x_1+x_2$  | $x_1+x_2\ge2$                 | (1,1)               | 2           |
| $x_1+2x_2$ | $x_1+2x_2\ge4$, $x_1,x_2\ge0$ | (0,2)               | 4           |
| $2x_1+x_2$ | $x_1+x_2\ge3$                 | (2,1)               | 5           |

#### Complexity

- Time: $O(n^{3.5}L)$ for linear programs (polynomial time)
- Space: $O(n^2)$ due to matrix factorizations

The Interior-Point Method is the elegant smooth traveler of optimization, gliding through the feasible space's heart while homing in on the global optimum with mathematical grace.

### 584 Gradient Descent (Unconstrained Optimization)

Gradient Descent is one of the simplest and most fundamental optimization algorithms. It is used to find the local minimum of a differentiable function by repeatedly moving in the opposite direction of the gradient, the direction of steepest descent.

#### What Problem Are We Solving?

We want to minimize a smooth function $f(x)$, where $x \in \mathbb{R}^n$:

$$
\min_x f(x)
$$

The function $f(x)$ could represent cost, loss, or error, and our goal is to find a point where the gradient (slope) is close to zero:

$$
\nabla f(x^*) = 0
$$

#### How Does It Work (Plain Language)

At each step, we update $x$ by moving *against* the gradient, since that's the direction in which $f(x)$ decreases fastest.

$$
x_{t+1} = x_t - \eta \nabla f(x_t)
$$

Here:

- $\eta > 0$ is the learning rate (step size).
- $\nabla f(x_t)$ is the gradient vector at the current point.

The algorithm continues until the gradient becomes very small or until changes in $x$ or $f(x)$ are negligible.

#### Step-by-Step Example

Let's minimize $f(x) = x^2$.
Then $\nabla f(x) = 2x$.

$$
x_{t+1} = x_t - \eta (2x_t) = (1 - 2\eta)x_t
$$

If $0 < \eta < 1$, the sequence converges to $x=0$, the global minimum.

#### Tiny Code (Simple Implementation)

Python

```python
import numpy as np

def gradient_descent(fprime, x0, eta=0.1, tol=1e-6, max_iter=1000):
    x = x0
    for _ in range(max_iter):
        grad = fprime(x)
        if np.linalg.norm(grad) < tol:
            break
        x -= eta * grad
    return x

# Example: minimize f(x) = x^2 + y^2
fprime = lambda v: np.array([2*v[0], 2*v[1]])
x_min = gradient_descent(fprime, np.array([5.0, -3.0]))
print("Minimum at:", x_min)
```

C

```c
#include <stdio.h>
#include <math.h>

void grad(double x[], double g[]) {
    g[0] = 2*x[0];
    g[1] = 2*x[1];
}

void gradient_descent(double x[], double eta, double tol, int max_iter) {
    double g[2];
    for (int i = 0; i < max_iter; i++) {
        grad(x, g);
        double norm = sqrt(g[0]*g[0] + g[1]*g[1]);
        if (norm < tol) break;
        x[0] -= eta * g[0];
        x[1] -= eta * g[1];
    }
}

int main(void) {
    double x[2] = {5.0, -3.0};
    gradient_descent(x, 0.1, 1e-6, 1000);
    printf("Minimum at (%.4f, %.4f)\n", x[0], x[1]);
    return 0;
}
```

#### Why It Matters

- Foundation for machine learning, deep learning, and optimization theory.
- Works in high dimensions with simple computation per step.
- Forms the basis of more advanced algorithms like SGD, Momentum, and Adam.

#### A Gentle Proof (Why It Works)

For convex and differentiable $f(x)$ with Lipschitz-continuous gradient ($L$-smooth), the update rule guarantees:

$$
f(x_{t+1}) \le f(x_t) - \frac{\eta}{2}|\nabla f(x_t)|^2
$$

if $0 < \eta \le \frac{1}{L}$.

This means each step decreases $f(x)$ by a quantity proportional to the squared gradient magnitude, leading to convergence.

#### Try It Yourself

1. Minimize $f(x)=x^2+y^2$ starting from $(5,-3)$.
2. Try $\eta=0.1$, $\eta=0.5$, and $\eta=1.0$, see which converges fastest.
3. Add a stopping condition based on $|f(x_{t+1}) - f(x_t)|$.
4. Visualize the path on a contour plot of $f(x,y)$.
5. Extend to non-convex functions like $f(x)=x^4 - 3x^3 + 2$.

#### Test Cases

| Function  | Gradient  | Start $x_0$ | $\eta$ | Result | True Minimum |
| --------- | --------- | ----------- | ------ | ------ | ------------ |
| $x^2$     | $2x$      | 5           | 0.1    | 0.0000 | 0            |
| $x^2+y^2$ | $(2x,2y)$ | (5,-3)      | 0.1    | (0,0)  | (0,0)        |
| $(x-2)^2$ | $2(x-2)$  | 0           | 0.1    | 2.0000 | 2            |

#### Complexity

- Time: $O(k)$ iterations (depends on learning rate and tolerance)
- Space: $O(n)$

Gradient Descent is the universal descent path, a simple, elegant method that lies at the foundation of optimization and learning across all of modern computation.

### 585 Stochastic Gradient Descent (SGD)

Stochastic Gradient Descent (SGD) is the workhorse of modern machine learning.
It extends ordinary gradient descent by using *random samples* (or mini-batches) to estimate the gradient at each step, allowing it to scale efficiently to massive datasets.

#### What Problem Are We Solving?

We aim to minimize a function defined as the average of many sample-based losses:

$$
f(x) = \frac{1}{N} \sum_{i=1}^N f_i(x)
$$

Computing the full gradient $\nabla f(x) = \frac{1}{N}\sum_i \nabla f_i(x)$ at every iteration can be very expensive when $N$ is large.

SGD avoids that by using only one (or a few) random samples per step:

$$
x_{t+1} = x_t - \eta \nabla f_{i_t}(x_t)
$$

where $i_t$ is randomly selected from ${1,2,\dots,N}$.

#### How Does It Work (Plain Language)

Instead of calculating the *exact* slope of the entire landscape, SGD takes a noisy but much cheaper estimate of the slope.
It zigzags its way downhill, sometimes overshooting, sometimes correcting, but overall, it trends toward the minimum.

This randomness acts like "built-in exploration," helping SGD escape shallow local minima in non-convex problems.

#### Algorithm Steps

1. Initialize $x_0$ (random or zero).
2. For each iteration $t$:

   * Randomly sample $i_t \in {1,\dots,N}$
   * Compute gradient estimate $g_t = \nabla f_{i_t}(x_t)$
   * Update parameter: $x_{t+1} = x_t - \eta g_t$
3. Optionally decay the learning rate $\eta_t$ over time.

Common decay schedules:
$$
\eta_t = \frac{\eta_0}{1 + \lambda t}
$$

#### Tiny Code (Simple Example)

Python

```python
import numpy as np

def sgd(fprime, x0, data, eta=0.1, epochs=1000):
    x = x0
    N = len(data)
    for t in range(epochs):
        i = np.random.randint(N)
        grad = fprime(x, data[i])
        x -= eta * grad
    return x

# Example: minimize average (x - y)^2 over samples
data = np.array([1.0, 2.0, 3.0, 4.0])
def grad(x, y): return 2*(x - y)
x_min = sgd(grad, x0=0.0, data=data)
print("Estimated minimum:", x_min)
```

C

```c
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

double grad(double x, double y) {
    return 2*(x - y);
}

double sgd(double *data, int N, double x0, double eta, int epochs) {
    double x = x0;
    srand(time(NULL));
    for (int t = 0; t < epochs; t++) {
        int i = rand() % N;
        double g = grad(x, data[i]);
        x -= eta * g;
    }
    return x;
}

int main(void) {
    double data[] = {1.0, 2.0, 3.0, 4.0};
    int N = 4;
    double xmin = sgd(data, N, 0.0, 0.1, 1000);
    printf("Estimated minimum: %.4f\n", xmin);
    return 0;
}
```

#### Why It Matters

- Crucial for training neural networks and large-scale models.
- Handles billions of data points efficiently.
- Naturally fits streaming or online learning settings.
- Randomness helps avoid bad local minima in non-convex landscapes.

#### A Gentle Proof (Why It Works)

If the learning rate $\eta_t$ satisfies
$$
\sum_t \eta_t = \infty \quad \text{and} \quad \sum_t \eta_t^2 < \infty,
$$
then under mild convexity and smoothness assumptions, SGD converges in expectation to the true minimum $x^*$.

The randomness introduces *variance*, but averaging or decreasing $\eta_t$ controls it.

#### Try It Yourself

1. Minimize $f(x) = \frac{1}{N}\sum_i (x - y_i)^2$ for random samples $y_i$.
2. Compare full gradient descent vs SGD convergence speed.
3. Add learning rate decay: $\eta_t = \eta_0/(1+0.01t)$.
4. Try mini-batch SGD (use several samples per step).
5. Plot $f(x_t)$ vs iteration, notice noisy but downward trend.

#### Test Cases

| Function  | Gradient | Dataset        | Result | True Minimum |
| --------- | -------- | -------------- | ------ | ------------ |
| $(x-y)^2$ | $2(x-y)$ | [1,2,3,4]      | ≈ 2.5  | 2.5          |
| $(x-5)^2$ | $2(x-5)$ | [5]*100        | 5.0    | 5.0          |
| $(x-y)^2$ | [1,1000] | large variance | ~500   | 500          |

#### Complexity

- Time: $O(k)$ iterations (each uses one or few samples)
- Space: $O(1)$ or $O(\text{mini-batch size})$
- Convergence: Sublinear but scalable

SGD is the heart of modern learning, a simple yet powerful idea that trades precision for speed, letting massive systems learn efficiently from a sea of data.

### 586 Newton's Method (Multivariate Optimization)

Newton's Method in multiple dimensions generalizes the one-dimensional root-finding approach to efficiently locate stationary points of smooth functions.
It uses both the gradient (first derivative) and Hessian (second derivative matrix) to make quadratic steps toward the optimum.

#### What Problem Are We Solving?

We want to minimize a smooth function $f(x)$, where $x \in \mathbb{R}^n$:

$$
\min_x f(x)
$$

At the minimum, the gradient vanishes:

$$
\nabla f(x^*) = 0.
$$

Newton's method refines the guess $x_t$ by approximating $f(x)$ locally with its second-order Taylor expansion:

$$
f(x+\Delta x) \approx f(x) + \nabla f(x)^T \Delta x + \frac{1}{2} \Delta x^T H(x) \Delta x,
$$

where $H(x)$ is the Hessian matrix.

Setting the gradient of this approximation to zero gives:

$$
H(x) \Delta x = -\nabla f(x),
$$

which leads to the update rule:

$$
x_{t+1} = x_t - H(x_t)^{-1} \nabla f(x_t).
$$

#### How Does It Work (Plain Language)

Imagine standing on a curved surface representing $f(x)$.
The gradient tells you the slope, but the Hessian tells you how the slope itself bends.
By combining them, Newton's method jumps directly toward the local minimum of that curve's quadratic approximation, often converging in just a few steps when the surface is well-behaved.

#### Tiny Code (Illustrative Example)

Python

```python
import numpy as np

def newton_multivariate(fprime, hessian, x0, tol=1e-6, max_iter=100):
    x = x0
    for _ in range(max_iter):
        g = fprime(x)
        H = hessian(x)
        if np.linalg.norm(g) < tol:
            break
        dx = np.linalg.solve(H, g)
        x -= dx
    return x

# Example: f(x, y) = x^2 + y^2
fprime = lambda v: np.array([2*v[0], 2*v[1]])
hessian = lambda v: np.array([[2, 0], [0, 2]])
x_min = newton_multivariate(fprime, hessian, np.array([5.0, -3.0]))
print("Minimum at:", x_min)
```

C (simplified)

```c
#include <stdio.h>
#include <math.h>

void gradient(double x[], double g[]) {
    g[0] = 2*x[0];
    g[1] = 2*x[1];
}

void hessian(double H[2][2]) {
    H[0][0] = 2; H[0][1] = 0;
    H[1][0] = 0; H[1][1] = 2;
}

void newton(double x[], double tol, int max_iter) {
    double g[2], H[2][2];
    for (int k = 0; k < max_iter; k++) {
        gradient(x, g);
        if (sqrt(g[0]*g[0] + g[1]*g[1]) < tol) break;
        hessian(H);
        x[0] -= g[0] / H[0][0];
        x[1] -= g[1] / H[1][1];
    }
}

int main(void) {
    double x[2] = {5.0, -3.0};
    newton(x, 1e-6, 100);
    printf("Minimum at (%.4f, %.4f)\n", x[0], x[1]);
    return 0;
}
```

#### Why It Matters

- Extremely fast near the optimum (quadratic convergence).
- The foundation for many advanced solvers, BFGS, Newton-CG, and trust-region methods.
- Central to optimization, machine learning, and numerical analysis.

#### A Gentle Proof (Why It Works)

Near a true minimum $x^*$, the function behaves almost quadratically:

$$
f(x) \approx f(x^*) + \frac{1}{2}(x-x^*)^T H(x^*)(x-x^*).
$$

Thus, the Newton update $x_{t+1}=x_t-H^{-1}\nabla f(x_t)$ effectively solves this local quadratic model exactly.
When $H$ is positive definite and $x_t$ is sufficiently close to $x^*$, convergence is quadratic, the error shrinks roughly as the square of the previous error.

#### Try It Yourself

1. Minimize $f(x,y)=x^2+y^2$ from $(5,-3)$.
2. Try a non-diagonal Hessian: $f(x,y)=x^2+xy+y^2$.
3. Compare convergence speed with Gradient Descent.
4. Observe behavior when the Hessian is not positive definite.
5. Add line search to improve robustness.

#### Test Cases

| Function              | Gradient              | Hessian                                           | Start      | Result | True Minimum |
| --------------------- | --------------------- | ------------------------------------------------- | ---------- | ------ | ------------ |
| $x^2 + y^2$           | $(2x,\,2y)$           | $\begin{bmatrix} 2 & 0 \\ 0 & 2 \end{bmatrix}$    | $(5,-3)$   | $(0,0)$ | $(0,0)$      |
| $(x-2)^2 + (y+1)^2$   | $(2(x-2),\,2(y+1))$   | $\operatorname{diag}(2,2)$                        | $(0,0)$    | $(2,-1)$ | $(2,-1)$     |
| $x^2 + xy + y^2$      | $(2x+y,\,x+2y)$       | $\begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix}$    | $(3,-2)$   | $(0,0)$ | $(0,0)$      |


#### Complexity

- Time: $O(n^3)$ per iteration (matrix inversion or linear solve)
- Space: $O(n^2)$ (store Hessian)

Newton's Method is the mathematician's scalpel, precise, elegant, and fast when the terrain is smooth, cutting straight to the heart of an optimum in just a few careful steps.

### 587 Conjugate Gradient Method

The Conjugate Gradient (CG) Method is an iterative algorithm for solving large systems of linear equations of the form

$$
A x = b
$$

where $A$ is symmetric positive definite (SPD).
It's especially powerful because it doesn't require matrix inversion or even explicit storage of $A$, only the ability to compute matrix–vector products.

CG can also be seen as a method for minimizing quadratic functions efficiently in high dimensions.

#### What Problem Are We Solving?

We want to minimize the quadratic form

$$
f(x) = \frac{1}{2}x^T A x - b^T x,
$$

which has the gradient

$$
\nabla f(x) = A x - b.
$$

Setting $\nabla f(x)=0$ gives the same equation $A x = b$.

Thus, solving $A x = b$ and minimizing $f(x)$ are equivalent.

#### How Does It Work (Plain Language)

Ordinary gradient descent may zigzag and converge slowly when contours of $f(x)$ are elongated.
The Conjugate Gradient method fixes this by ensuring each search direction is A-orthogonal (conjugate) to all previous ones, meaning each step eliminates error in a new dimension without undoing progress from earlier steps.

It can find the exact solution in at most $n$ steps (for $n$ variables) in exact arithmetic.

#### Algorithm Steps

1. Initialize $x_0$ (e.g., zeros).
2. Compute initial residual $r_0 = b - A x_0$ and set direction $p_0 = r_0$.
3. For each iteration $k=0,1,2,\dots$:
   $$
   \alpha_k = \frac{r_k^T r_k}{p_k^T A p_k}
   $$
   $$
   x_{k+1} = x_k + \alpha_k p_k
   $$
   $$
   r_{k+1} = r_k - \alpha_k A p_k
   $$
   If $|r_{k+1}|$ is small enough, stop.
   Otherwise:
   $$
   \beta_k = \frac{r_{k+1}^T r_{k+1}}{r_k^T r_k}, \quad p_{k+1} = r_{k+1} + \beta_k p_k.
   $$

Each new $p_k$ is "conjugate" to all previous directions with respect to $A$.

#### Tiny Code (Minimal Implementation)

Python

```python
import numpy as np

def conjugate_gradient(A, b, x0=None, tol=1e-6, max_iter=1000):
    n = len(b)
    x = np.zeros(n) if x0 is None else x0
    r = b - A @ x
    p = r.copy()
    for _ in range(max_iter):
        Ap = A @ p
        alpha = np.dot(r, r) / np.dot(p, Ap)
        x += alpha * p
        r_new = r - alpha * Ap
        if np.linalg.norm(r_new) < tol:
            break
        beta = np.dot(r_new, r_new) / np.dot(r, r)
        p = r_new + beta * p
        r = r_new
    return x

# Example: Solve A x = b
A = np.array([[4, 1], [1, 3]], dtype=float)
b = np.array([1, 2], dtype=float)
x = conjugate_gradient(A, b)
print("Solution:", x)
```

C

```c
#include <stdio.h>
#include <math.h>

void matvec(double A[2][2], double x[2], double y[2]) {
    y[0] = A[0][0]*x[0] + A[0][1]*x[1];
    y[1] = A[1][0]*x[0] + A[1][1]*x[1];
}

void conjugate_gradient(double A[2][2], double b[2], double x[2], int max_iter, double tol) {
    double r[2], p[2], Ap[2];
    matvec(A, x, r);
    for (int i = 0; i < 2; i++) {
        r[i] = b[i] - r[i];
        p[i] = r[i];
    }
    for (int k = 0; k < max_iter; k++) {
        matvec(A, p, Ap);
        double rr = r[0]*r[0] + r[1]*r[1];
        double alpha = rr / (p[0]*Ap[0] + p[1]*Ap[1]);
        x[0] += alpha * p[0];
        x[1] += alpha * p[1];
        for (int i = 0; i < 2; i++) r[i] -= alpha * Ap[i];
        double rr_new = r[0]*r[0] + r[1]*r[1];
        if (sqrt(rr_new) < tol) break;
        double beta = rr_new / rr;
        for (int i = 0; i < 2; i++) p[i] = r[i] + beta * p[i];
    }
}

int main(void) {
    double A[2][2] = {{4, 1}, {1, 3}};
    double b[2] = {1, 2};
    double x[2] = {0, 0};
    conjugate_gradient(A, b, x, 1000, 1e-6);
    printf("Solution: (%.4f, %.4f)\n", x[0], x[1]);
    return 0;
}
```

#### Why It Matters

- Ideal for large sparse systems, especially those from numerical PDEs and finite element methods.
- Avoids explicit matrix inversion, only needs $A p$ products.
- Core building block in machine learning, physics simulations, and scientific computing.

#### A Gentle Proof (Why It Works)

Each direction $p_k$ is chosen so that

$$
p_i^T A p_j = 0 \quad \text{for } i \ne j,
$$

ensuring orthogonality under the $A$-inner product.
This means each step eliminates error along one conjugate direction, never revisiting it.
For an $n$-dimensional system, all error components are eliminated after at most $n$ steps.

#### Try It Yourself

1. Solve $A x = b$ with 
$$
A = 
\begin{bmatrix}
4 & 1\\
1 & 3
\end{bmatrix}, 
\quad 
b = 
\begin{bmatrix}
1\\
2
\end{bmatrix}.
$$

2. Compare the result with Gaussian elimination.

3. Modify $A$ to be non-symmetric and observe failure or oscillation.

4. Add a preconditioner $M^{-1}$ to improve convergence.

5. Plot $|r_k|$ versus iterations and note the geometric decay.


#### Test Cases

| $A$                                               | $b$         | Solution $x^*$          | Iterations |
| ------------------------------------------------- | ----------- | ----------------------- | ----------- |
| $\begin{bmatrix} 4 & 1 \\ 1 & 3 \end{bmatrix}$    | $[1, 2]^T$  | $(0.0909,\, 0.6364)$    | 3           |
| $\operatorname{diag}(2, 5, 10)$                   | $[1, 1, 1]^T$ | $(0.5,\, 0.2,\, 0.1)$    | 3           |
| random SPD $(5 \times 5)$                         | random $b$  | accurate                | $< n$       |


#### Complexity

- Time: $O(k n)$ (each iteration needs one matrix–vector multiply)
- Space: $O(n)$
- Convergence: Linear but very efficient for well-conditioned $A$

The Conjugate Gradient method is the quiet power of numerical optimization, using geometry and algebra hand in hand to solve vast systems with elegant efficiency.

### 588 Lagrange Multipliers (Constrained Optimization)

The Lagrange Multiplier Method provides a systematic way to find extrema (minima or maxima) of a function subject to equality constraints.
It introduces auxiliary variables, called *Lagrange multipliers*, that enforce the constraints algebraically, transforming a constrained problem into an unconstrained one.

#### What Problem Are We Solving?

We want to minimize (or maximize) a function
$$
f(x_1, x_2, \dots, x_n)
$$
subject to one or more equality constraints:
$$
g_i(x_1, x_2, \dots, x_n) = 0, \quad i = 1, 2, \dots, m.
$$

For simplicity, start with a single constraint $g(x) = 0$.

#### The Core Idea

At an optimum, the gradient of $f$ must lie in the same direction as the gradient of the constraint $g$:

$$
\nabla f(x^*) = \lambda \nabla g(x^*),
$$

where $\lambda$ is the Lagrange multiplier.
This captures the idea that any small movement along the constraint surface cannot reduce $f$ further.

We introduce the Lagrangian function:

$$
\mathcal{L}(x, \lambda) = f(x) - \lambda g(x),
$$

and find stationary points by setting all derivatives to zero:

$$
\nabla_x \mathcal{L} = 0, \quad g(x) = 0.
$$

#### Example: Classic Two-Variable Case

Minimize
$$
f(x, y) = x^2 + y^2
$$
subject to
$$
x + y = 1.
$$

1. Construct the Lagrangian:
   $$
   \mathcal{L}(x, y, \lambda) = x^2 + y^2 - \lambda(x + y - 1)
   $$

2. Take partial derivatives and set them to zero:
   $$
   \frac{\partial \mathcal{L}}{\partial x} = 2x - \lambda = 0
   $$
   $$
   \frac{\partial \mathcal{L}}{\partial y} = 2y - \lambda = 0
   $$
   $$
   \frac{\partial \mathcal{L}}{\partial \lambda} = -(x + y - 1) = 0
   $$

3. Solve: from the first two equations, $2x = 2y = \lambda$ → $x = y$.
   Substituting into the constraint $x + y = 1$ → $x = y = 0.5$.

Thus the minimum occurs at $(x, y) = (0.5, 0.5)$ with $\lambda = 1$.

#### Tiny Code (Simple Example)

Python

```python
import sympy as sp

x, y, lam = sp.symbols('x y lam')
f = x2 + y2
g = x + y - 1
L = f - lam * g

sol = sp.solve([sp.diff(L, x), sp.diff(L, y), sp.diff(L, lam)], [x, y, lam])
print(sol)
```

Output:

```
{x: 0.5, y: 0.5, lam: 1}
```

C (conceptual numeric)

```c
#include <stdio.h>

int main(void) {
    double x = 0.5, y = 0.5, lam = 1.0;
    printf("Minimum at (%.2f, %.2f), lambda = %.2f\n", x, y, lam);
    return 0;
}
```

#### Why It Matters

- Core of constrained optimization in calculus, economics, and machine learning.
- Generalizes easily to multiple constraints and higher dimensions.
- Forms the foundation for KKT conditions (see next section) used in convex optimization and support vector machines.

#### A Gentle Proof (Geometric View)

At the optimum point, any movement tangent to the constraint surface must not change $f(x)$.
Hence, $\nabla f$ must be perpendicular to that surface, that is, parallel to $\nabla g$.
Introducing $\lambda$ allows us to equate these directions algebraically, creating solvable equations.

#### Try It Yourself

1. Minimize $f(x, y) = x^2 + y^2$ subject to $x + y = 1$.
2. Try $f(x, y) = x^2 + 2y^2$ subject to $x - y = 0$.
3. Solve with two constraints:
   $$
   g_1(x, y) = x + y - 1 = 0, \quad g_2(x, y) = x - 2y = 0.
   $$
4. Observe how $\lambda_1, \lambda_2$ act as "weights" enforcing constraints.

#### Test Cases

| Function   | Constraint  | Result $(x^*, y^*)$                     | $\lambda$ |
| ---------- | ----------- | --------------------------------------- | --------- |
| $x^2+y^2$  | $x+y=1$     | (0.5, 0.5)                              | 1         |
| $x^2+2y^2$ | $x-y=0$     | (0, 0)                                  | 0         |
| $x^2+y^2$  | $x^2+y^2=4$ | circle constraint → any point on circle | variable  |

#### Complexity

- Symbolic solution: $O(n^3)$ for solving equations.
- Numeric solution: iterative methods (Newton–Raphson, SQP) for large systems.

The method of Lagrange multipliers is the mathematical bridge between freedom and constraint, guiding optimization across boundaries defined by nature, design, or logic itself.

### 589 Karush–Kuhn–Tucker (KKT) Conditions

The Karush–Kuhn–Tucker (KKT) conditions generalize the Lagrange multiplier method to handle inequality and equality constraints in nonlinear optimization.
They form the cornerstone of modern constrained optimization, especially in convex optimization, machine learning (SVMs), and economics.

#### What Problem Are We Solving?

We want to minimize

$$
f(x)
$$

subject to

$$
g_i(x) \le 0 \quad (i = 1, \dots, m)
$$

and

$$
h_j(x) = 0 \quad (j = 1, \dots, p).
$$

Here:

- $g_i(x)$ are inequality constraints,
- $h_j(x)$ are equality constraints.

#### The Lagrangian Function

We extend the idea of the Lagrange function:

$$
\mathcal{L}(x, \lambda, \mu) = f(x) + \sum_{i=1}^m \lambda_i g_i(x) + \sum_{j=1}^p \mu_j h_j(x),
$$

where

- $\lambda_i \ge 0$ are multipliers for inequality constraints,
- $\mu_j$ are multipliers for equality constraints.

#### The KKT Conditions

For an optimal point $x^*$, there must exist $\lambda_i$ and $\mu_j$ satisfying the following four conditions:

1. Stationarity
   $$
   \nabla f(x^*) + \sum_{i=1}^m \lambda_i \nabla g_i(x^*) + \sum_{j=1}^p \mu_j \nabla h_j(x^*) = 0
   $$

2. Primal Feasibility
   $$
   g_i(x^*) \le 0, \quad h_j(x^*) = 0
   $$

3. Dual Feasibility
   $$
   \lambda_i \ge 0 \quad \text{for all } i
   $$

4. Complementary Slackness
   $$
   \lambda_i g_i(x^*) = 0 \quad \text{for all } i
   $$

Complementary slackness means that if a constraint is not "tight" (inactive), its corresponding $\lambda_i$ must be zero, it exerts no force on the solution.

#### Example: Quadratic Optimization with Constraint

Minimize
$$
f(x) = x^2
$$
subject to
$$
g(x) = 1 - x \le 0.
$$

Step 1: Write the Lagrangian
$$
\mathcal{L}(x, \lambda) = x^2 + \lambda(1 - x)
$$

Step 2: KKT conditions

- Stationarity:
  $$
  \frac{d\mathcal{L}}{dx} = 2x - \lambda = 0
  $$

- Primal feasibility:
  $$
  1 - x \le 0
  $$

- Dual feasibility:
  $$
  \lambda \ge 0
  $$

- Complementary slackness:
  $$
  \lambda(1 - x) = 0
  $$

Solve:
From stationarity, $\lambda = 2x$.
From complementary slackness, either $\lambda=0$ or $1-x=0$.

- If $\lambda=0$, then $x=0$. But $1-x=1>0$, violates feasibility.
- If $1-x=0$, then $x=1$ and $\lambda=2$.

Solution: $x^* = 1$, $\lambda^* = 2$.

#### Tiny Code (Symbolic Example)

Python

```python
import sympy as sp

x, lam = sp.symbols('x lam')
f = x2
g = 1 - x
L = f + lam * g

sol = sp.solve([sp.diff(L, x), g, lam >= 0, lam * g], [x, lam], dict=True)
print(sol)
```

Output:

```
$${x: 1, lam: 2}]
```

#### Why It Matters

- Generalizes Lagrange multipliers to handle inequality constraints.
- Fundamental to convex optimization, machine learning (SVMs), econometrics, and engineering design.
- Provides necessary conditions (and sufficient for convex problems) for optimality.

#### A Gentle Proof (Intuition)

At an optimum, any feasible perturbation $\Delta x$ must not reduce $f(x)$.
This is only possible if the gradient of $f$ lies within the cone formed by the gradients of the active constraints.
KKT multipliers $\lambda_i$ express this combination mathematically.

#### Try It Yourself

1. Minimize $f(x)=x^2+y^2$ subject to $x+y\ge1$.
2. Solve $f(x)=x^2$ with constraint $x\ge1$.
3. Compare unconstrained and constrained solutions.
4. Implement KKT solver using symbolic differentiation.

#### Test Cases

| Function  | Constraint | Result $(x^*)$ | $\lambda^*$ |
| --------- | ---------- | -------------- | ----------- |
| $x^2$     | $1-x\le0$  | 1              | 2           |
| $x^2+y^2$ | $x+y-1=0$  | $(0.5, 0.5)$   | 1           |
| $x^2$     | $x\ge0$    | 0              | 0           |

#### Complexity

- Symbolic solving: $O(n^3)$ for small systems.
- Numerical KKT solvers (e.g. Sequential Quadratic Programming): $O(n^3)$ per iteration.

The KKT conditions are the grammar of optimization, expressing how objectives and constraints negotiate balance at the frontier of possibility.

### 590 Coordinate Descent

The Coordinate Descent method is one of the simplest yet surprisingly powerful algorithms for optimization.
Instead of adjusting all variables at once, it updates one coordinate at a time, cycling through variables until convergence.
It's widely used in LASSO regression, matrix factorization, and sparse optimization.

#### What Problem Are We Solving?

We want to minimize a function

$$
f(x_1, x_2, \dots, x_n)
$$

possibly subject to simple constraints (like $x_i \ge 0$).

#### The Idea

Rather than tackling the full gradient $\nabla f(x)$ at once, we fix all variables except one and minimize with respect to that variable.

For example, in 2D:

$$
f(x, y) \rightarrow \text{first fix } y, \text{ minimize over } x; \text{ then fix } x, \text{ minimize over } y.
$$

Each update step reduces the objective, leading to convergence for convex functions.

#### Algorithm Steps

Given an initial vector $x^{(0)}$:

1. For each coordinate $i = 1, \dots, n$:

   * Define the *partial function* $f_i(x_i) = f(x_1, \dots, x_i, \dots, x_n)$, holding other variables fixed.
   * Find
     $$
     x_i^{(k+1)} = \arg \min_{x_i} f_i(x_i)
     $$
2. Repeat until convergence (when $f(x)$ changes negligibly or $|x^{(k+1)} - x^{(k)}| < \varepsilon$).

#### Example: Simple Quadratic Function

Minimize
$$
f(x, y) = (x - 2)^2 + (y - 3)^2.
$$

1. Initialize $(x, y) = (0, 0)$.
2. Fix $y=0$: minimize $(x-2)^2$ → $x=2$.
3. Fix $x=2$: minimize $(y-3)^2$ → $y=3$.
4. Done, reached $(2, 3)$ in one sweep.

For convex quadratics, coordinate descent converges linearly to the global minimum.

#### Tiny Code

Python

```python
import numpy as np

def coordinate_descent(f, grad, x0, tol=1e-6, max_iter=1000):
    x = x0.copy()
    n = len(x)
    for _ in range(max_iter):
        x_old = x.copy()
        for i in range(n):
            g = grad(x)
            x[i] -= 0.1 * g[i]  # simple step along coordinate gradient
        if np.linalg.norm(x - x_old) < tol:
            break
    return x

# Example: f(x,y) = (x-2)^2 + (y-3)^2
f = lambda v: (v[0]-2)2 + (v[1]-3)2
grad = lambda v: np.array([2*(v[0]-2), 2*(v[1]-3)])
x = coordinate_descent(f, grad, np.array([0.0, 0.0]))
print("Minimum:", x)
```

C (simple version)

```c
#include <stdio.h>
#include <math.h>

int main(void) {
    double x = 0, y = 0;
    double alpha = 0.1;
    for (int iter = 0; iter < 100; iter++) {
        double x_old = x, y_old = y;
        x -= alpha * 2 * (x - 2);
        y -= alpha * 2 * (y - 3);
        if (fabs(x - x_old) < 1e-6 && fabs(y - y_old) < 1e-6) break;
    }
    printf("Minimum at (%.6f, %.6f)\n", x, y);
    return 0;
}
```

#### Why It Matters

- Extremely simple to implement.
- Works well when $f(x)$ is separable or coordinate-wise convex.
- Scales to very high dimensions, since each step only updates one variable.
- Used in LASSO, ridge regression, support vector machines, and matrix factorization.

#### A Gentle Proof (Intuition)

At each iteration, minimizing over one coordinate cannot increase $f(x)$.
Thus, $f(x^{(k)})$ is non-increasing and converges.
For convex $f$, this limit is the global minimum.

#### Try It Yourself

1. Minimize $f(x, y) = (x-1)^2 + (2y-3)^2$.
2. Add constraint $x, y \ge 0$.
3. Replace gradient update with exact 1D minimization per coordinate.
4. Try a non-convex example, observe convergence to local minima.

#### Test Cases

| Function          | Start | Result     | Notes                    |
| ----------------- | ----- | ---------- | ------------------------ |
| $(x-2)^2+(y-3)^2$ | (0,0) | (2,3)      | Quadratic, exact minimum |
| $(x-1)^2+(y-1)^2$ | (5,5) | (1,1)      | Converges linearly       |
| $\sin(x)+\sin(y)$ | (2,2) | (π/2, π/2) | Local minimum            |

#### Complexity

- Each coordinate update: $O(1)$ (if gradient is cheap).
- One full cycle: $O(n)$.
- Total complexity: $O(nk)$ for $k$ iterations.

Coordinate Descent embodies the minimalist spirit of optimization, improving one small piece at a time, yet always moving closer to the whole solution.

# Section 60. Algebraic Tricks and Transform Techniques 

### 591 Polynomial Multiplication (FFT)

Multiply two polynomials fast by converting coefficient vectors to point values via the Fast Fourier Transform, multiplying pointwise, then transforming back. This computes convolution in $O(n\log n)$ instead of $O(n^2)$.

#### What Problem Are We Solving?

Given
$$
A(x)=\sum_{i=0}^{n-1}a_i x^i,\quad B(x)=\sum_{j=0}^{m-1}b_j x^j,
$$
compute coefficients of
$$
C(x)=A(x)B(x)=\sum_{k=0}^{n+m-2}c_k x^k,
\quad c_k=\sum_{i+j=k}a_i b_j.
$$

Naive convolution is $O(nm)$. FFT uses the Convolution Theorem to do it in $O(N\log N)$ where $N$ is a power of two with $N\ge n+m-1$.

#### How Does It Work

1. Choose size: $N=\text{next power of two}\ge n+m-1$.
2. Zero pad $a$ and $b$ to length $N$.
3. FFT both sequences: $\hat a=\operatorname{FFT}(a)$, $\hat b=\operatorname{FFT}(b)$.
4. Pointwise multiply: $\hat c_k=\hat a_k\hat b_k$.
5. Inverse FFT: $c=\operatorname{IFFT}(\hat c)$.
6. Round real parts to nearest integers if inputs are integers.

Convolution Theorem:
$$
\mathcal{F}(a*b)=\mathcal{F}(a)\odot \mathcal{F}(b).
$$

#### Tiny Code

Python (NumPy, real coefficients)

```python
import numpy as np

def poly_mul_fft(a, b):
    n = len(a) + len(b) - 1
    N = 1 << (n - 1).bit_length()
    fa = np.fft.rfft(a, N)
    fb = np.fft.rfft(b, N)
    fc = fa * fb
    c = np.fft.irfft(fc, N)[:n]
    # If inputs are integers, round to nearest int
    return np.rint(c).astype(int)

# Example
print(poly_mul_fft([1,2,3], [4,5]))  # [4,13,22,15]
```

C++17 (iterative Cooley–Tukey with std::complex)

```cpp
#include <bits/stdc++.h>
using namespace std;

using cd = complex<double>;
const double PI = acos(-1);

void fft(vector<cd>& a, bool inv){
    int n = (int)a.size();
    static vector<int> rev;
    static vector<cd> roots{0,1};
    if ((int)rev.size() != n){
        int k = __builtin_ctz(n);
        rev.assign(n,0);
        for (int i=0;i<n;i++)
            rev[i] = (rev[i>>1]>>1) | ((i&1)<<(k-1));
    }
    if ((int)roots.size() < n){
        int k = __builtin_ctz(roots.size());
        roots.resize(n);
        while ((1<<k) < n){
            double ang = 2*PI/(1<<(k+1));
            for (int i=1<<(k-1); i<(1<<k); i++){
                roots[2*i]   = roots[i];
                roots[2*i+1] = cd(cos(ang*(2*i+1-(1<<k))), sin(ang*(2*i+1-(1<<k))));
            }
            k++;
        }
    }
    for (int i=0;i<n;i++) if (i<rev[i]) swap(a[i],a[rev[i]]);
    for (int len=1; len<n; len<<=1){
        for (int i=0;i<n;i+=2*len){
            for (int j=0;j<len;j++){
                cd u=a[i+j];
                cd v=a[i+j+len]*roots[len+j];
                a[i+j]=u+v;
                a[i+j+len]=u-v;
            }
        }
    }
    if (inv){
        reverse(a.begin()+1, a.end());
        for (auto& x:a) x/=n;
    }
}

vector<long long> multiply(const vector<long long>& A, const vector<long long>& B){
    int n = 1;
    int need = (int)A.size() + (int)B.size() - 1;
    while (n < need) n <<= 1;
    vector<cd> fa(n), fb(n);
    for (size_t i=0;i<A.size();i++) fa[i] = A[i];
    for (size_t i=0;i<B.size();i++) fb[i] = B[i];
    fft(fa,false); fft(fb,false);
    for (int i=0;i<n;i++) fa[i] *= fb[i];
    fft(fa,true);
    vector<long long> C(need);
    for (int i=0;i<need;i++) C[i] = llround(fa[i].real());
    return C;
}

int main(){
    vector<long long> a={1,2,3}, b={4,5};
    auto c = multiply(a,b); // 4 13 22 15
    for (auto x:c) cout<<x<<" ";
    cout<<"\n";
}
```

#### Why It Matters

- Reduces polynomial multiplication from quadratic to quasi-linear time.
- Backbone for big integer arithmetic, signal processing, string matching via convolution, and combinatorial counting.
- Extends to multidimensional convolutions.

#### A Gentle Proof Idea

Choose $N$-th roots of unity $\omega_N^k$. The DFT evaluates a degree less than $N$ polynomial at $N$ distinct points:
$$
\hat a_k=\sum_{t=0}^{N-1} a_t \omega_N^{kt}.
$$
Pointwise multiplication gives values of $C(x)=A(x)B(x)$ at the same points:
$\hat c_k=\hat a_k\hat b_k$. Since evaluation points are distinct, inverse DFT uniquely reconstructs $c_0,\dots,c_{N-1}$.

#### Practical Tips

- Pick $N$ as a power of two for fastest FFT.
- For integer inputs, rounding after IFFT recovers exact coefficients.
- Large coefficients risk floating error. Two fixes:

  1. split coefficients into chunks and use two or three FFTs with Chinese Remainder reconstruction,
  2. use NTT over a prime modulus for exact modular convolution.
- Trim trailing zeros.

#### Try It Yourself

1. Multiply $(1+2x+3x^2)$ by $(4+5x)$ by hand and compare to FFT result.
2. Convolve two random length 10 vectors and verify against naive $O(n^2)$.
3. Measure runtime vs naive as sizes double.
4. Implement circular convolution and compare with linear convolution via zero padding.
5. Explore double-precision limits by scaling inputs.

#### Test Cases

| Input A   | Input B   | Output C           |
| --------- | --------- | ------------------ |
| $[1,2,3]$ | $[4,5]$   | $[4,13,22,15]$     |
| $[1,0,1]$ | $[1,1]$   | $[1,1,1,1]$        |
| $[2,3,4]$ | $[5,6,7]$ | $[10,27,52,45,28]$ |

#### Complexity

- Time: $O(N\log N)$ for $N\ge n+m-1$
- Space: $O(N)$

FFT based multiplication is the standard fast path for large polynomial and integer products.

### 592 Polynomial Inversion (Newton Iteration)

Polynomial inversion finds a series $B(x)$ such that

$$
A(x)B(x)\equiv1\pmod{x^n}.
$$

That means multiplying $A$ and $B$ should yield $1$ up to terms of degree $n-1$.
It's the polynomial version of computing $1/a$ using iterative refinement, built on the same Newton–Raphson principle.

#### What Problem Are We Solving?

Given a formal power series

$$
A(x)=a_0+a_1x+a_2x^2+\dots,
$$

we want another series

$$
B(x)=b_0+b_1x+b_2x^2+\dots
$$

such that

$$
A(x)B(x)=1.
$$

This is only possible if $a_0\ne0$.

#### Newton's Method for Series

We use Newton iteration, similar to numerical inversion.
Let $B_k(x)$ be our current approximation modulo $x^k$.
Then we refine it using:

$$
B_{2k}(x)=B_k(x),(2-A(x)B_k(x)) \pmod{x^{2k}}.
$$

Each iteration doubles the correct number of terms.

#### Algorithm Steps

1. Start with $B_1(x)=1/a_0$.
2. For $k=1,2,4,8,\dots$ until $k\ge n$:

   * Compute $C(x)=A(x)B_k(x)\pmod{x^{2k}}$.
   * Update $B_{2k}(x)=B_k(x),(2-C(x))\pmod{x^{2k}}$.
3. Truncate $B(x)$ to degree $n-1$.

All polynomial multiplications use FFT or NTT for speed.

#### Example

Suppose

$$
A(x)=1+x.
$$

We expect

$$
B(x)=1-x+x^2-x^3+\dots.
$$

Step 1: $B_1=1$.
Step 2: $A(x)B_1=1+x$, so

$$
B_2=B_1(2-(1+x))=1-x.
$$

Step 3:
$A(x)B_2=(1+x)(1-x)=1-x^2$,

$$
B_4=B_2(2-(1-x^2))=(1-x)(1+x^2)=1-x+x^2-x^3.
$$

And so on. Each step doubles precision.

#### Tiny Code

Python (using NumPy FFT)

```python
import numpy as np

def poly_mul(a, b):
    n = len(a) + len(b) - 1
    N = 1 << (n - 1).bit_length()
    fa = np.fft.rfft(a, N)
    fb = np.fft.rfft(b, N)
    fc = fa * fb
    c = np.fft.irfft(fc, N)[:n]
    return c

def poly_inv(a, n):
    b = np.array([1 / a[0]])
    k = 1
    while k < n:
        k *= 2
        ab = poly_mul(a[:k], b)[:k]
        b = (b * 2 - poly_mul(b, ab)[:k])[:k]
    return b[:n]

# Example: invert 1 + x
a = np.array([1.0, 1.0])
b = poly_inv(a, 8)
print(np.round(b, 3))
# [1. -1. 1. -1. 1. -1. 1. -1.]
```

#### Why It Matters

- Used for series division, modular inverses, and polynomial division in FFT-based arithmetic.
- Core primitive in formal power series computations.
- Appears in combinatorics, symbolic algebra, and computer algebra systems.

#### Intuition Behind Newton Update

If we want $AB=1$, define
$$
F(B)=A(x)B(x)-1.
$$
Newton iteration gives

$$
B_{k+1}=B_k-F(B_k)/A(x)=B_k-(A(x)B_k-1)B_k=B_k(2-A(x)B_k),
$$

which matches our polynomial update rule.

#### Try It Yourself

1. Invert $A(x)=1+x+x^2$ up to degree 8.
2. Verify by multiplying $A(x)B(x)$ and confirming all terms above constant vanish.
3. Implement with modular arithmetic under prime $p$.
4. Compare FFT-based vs naive performance.

#### Test Cases

| $A(x)$ | $B(x)$ (first terms)       | Verification                |
| ------ | -------------------------- | --------------------------- |
| $1+x$  | $1-x+x^2-x^3+\dots$        | $(1+x)(1-x+x^2-\dots)=1$    |
| $1-2x$ | $1+2x+4x^2+8x^3+\dots$     | $(1-2x)(1+2x+4x^2+\dots)=1$ |
| $2+x$  | $0.5-0.25x+0.125x^2-\dots$ | $(2+x)(B)=1$                |

#### Complexity

- Each iteration doubles precision.
- Uses FFT multiplication → $O(n\log n)$.
- Memory: $O(n)$.

Polynomial inversion by Newton iteration is a masterclass in algebraic efficiency, one simple update that doubles accuracy each time.

### 593 Polynomial Derivative

Taking the derivative of a polynomial is one of the simplest symbolic operations, yet it appears everywhere in algorithms for optimization, root finding, series manipulation, and numerical analysis.

Given a polynomial
$$
A(x)=a_0+a_1x+a_2x^2+\dots+a_nx^n,
$$
its derivative is
$$
A'(x)=a_1+2a_2x+3a_3x^2+\dots+n a_nx^{n-1}.
$$

#### What Problem Are We Solving?

We want to compute the derivative coefficients efficiently and represent $A'(x)$ in the same coefficient form as $A(x)$.
If
$$
A(x)=[a_0,a_1,\dots,a_n],
$$
then
$$
A'(x)=[a_1,2a_2,3a_3,\dots,n a_n].
$$

This operation runs in $O(n)$ time and is a key subroutine in polynomial algebra and calculus of formal power series.

#### Algorithm Steps

1. Given coefficients $a_0,\dots,a_n$.
2. For each $i$ from $1$ to $n$:

   * Compute $b_{i-1}=i\times a_i$.
3. Return coefficients $[b_0,b_1,\dots,b_{n-1}]$ of $A'(x)$.

#### Example

Let
$$
A(x)=3+2x+5x^2+4x^3.
$$

Then
$$
A'(x)=2+10x+12x^2.
$$

In coefficient form:

| Term  | Coefficient | New Coefficient |
| ----- | ----------- | --------------- |
| $x^0$ | $3$         |,               |
| $x^1$ | $2$         | $2\times1=2$    |
| $x^2$ | $5$         | $5\times2=10$   |
| $x^3$ | $4$         | $4\times3=12$   |

#### Tiny Code

Python

```python
def poly_derivative(a):
    n = len(a)
    return [i * a[i] for i in range(1, n)]

# Example
A = [3, 2, 5, 4]
print(poly_derivative(A))  # [2, 10, 12]
```

C

```c
#include <stdio.h>

int main(void) {
    double a[] = {3, 2, 5, 4};
    int n = 4;
    double d[n - 1];
    for (int i = 1; i < n; i++)
        d[i - 1] = i * a[i];
    for (int i = 0; i < n - 1; i++)
        printf("%.0f ", d[i]);
    printf("\n");
    return 0;
}
```

#### Why It Matters

- Central in Newton–Raphson root-finding, gradient descent, and optimization.
- Appears in polynomial division, Taylor series, differential equations, and symbolic algebra.
- Used in automatic differentiation and backpropagation as the algebraic foundation.

#### A Gentle Proof

From calculus:
$$
\frac{d}{dx}(x^i)=i x^{i-1}.
$$

Since $A(x)=\sum_i a_i x^i$, linearity gives
$$
A'(x)=\sum_i a_i i x^{i-1}.
$$
Thus the coefficient of $x^{i-1}$ in $A'(x)$ is $i a_i$.

#### Try It Yourself

1. Differentiate $A(x)=5x^4+2x^3-x^2+3x-7$.
2. Compute the derivative twice (second derivative).
3. Implement derivative modulo $p$ for large integer polynomials.
4. Use derivative in Newton–Raphson root updates:
   $x_{k+1}=x_k-\frac{A(x_k)}{A'(x_k)}.$

#### Test Cases

| Polynomial       | Derivative    | Result      |
| ---------------- | ------------- | ----------- |
| $3+2x+5x^2+4x^3$ | $2+10x+12x^2$ | `[2,10,12]` |
| $x^4$            | $4x^3$        | `[0,0,0,4]` |
| $1+x+x^2+x^3$    | $1+2x+3x^2$   | `[1,2,3]`   |

#### Complexity

- Time: $O(n)$
- Space: $O(n)$

Polynomial differentiation is a one-line operation, but it underlies much of algorithmic calculus, the small step that powers big changes.

### 594 Polynomial Integration

Polynomial integration is the reverse of differentiation: we find a polynomial $B(x)$ such that $B'(x)=A(x)$.
It's a simple yet vital tool in symbolic algebra, numerical integration, and generating functions.

#### What Problem Are We Solving?

Given
$$
A(x)=a_0+a_1x+a_2x^2+\dots+a_{n-1}x^{n-1},
$$
we want
$$
B(x)=\int A(x),dx=C+b_1x+b_2x^2+\dots+b_nx^n,
$$
where each coefficient satisfies
$$
b_{i+1}=\frac{a_i}{i+1}.
$$

Usually, we set the constant of integration $C=0$ for computational purposes.

#### How It Works

If $A(x)=[a_0,a_1,a_2,\dots,a_{n-1}]$,
then
$$
B(x)=[0,\frac{a_0}{1},\frac{a_1}{2},\frac{a_2}{3},\dots,\frac{a_{n-1}}{n}].
$$

Each term is divided by its new exponent index.

#### Example

Let
$$
A(x)=2+10x+12x^2.
$$

Then
$$
B(x)=2x+\frac{10}{2}x^2+\frac{12}{3}x^3=2x+5x^2+4x^3.
$$

In coefficient form:

| Term  | Coefficient in A(x) | Integrated Term | Coefficient in B(x) |
| ----- | ------------------- | --------------- | ------------------- |
| $x^0$ | 2                   | $2x$            | 2                   |
| $x^1$ | 10                  | $10x^2/2$       | 5                   |
| $x^2$ | 12                  | $12x^3/3$       | 4                   |

So $B(x)=[0,2,5,4]$.

#### Tiny Code

Python

```python
def poly_integrate(a):
    n = len(a)
    return [0.0] + [a[i] / (i + 1) for i in range(n)]

# Example
A = [2, 10, 12]
print(poly_integrate(A))  # [0.0, 2.0, 5.0, 4.0]
```

C

```c
#include <stdio.h>

int main(void) {
    double a[] = {2, 10, 12};
    int n = 3;
    double b[n + 1];
    b[0] = 0.0;
    for (int i = 0; i < n; i++)
        b[i + 1] = a[i] / (i + 1);
    for (int i = 0; i <= n; i++)
        printf("%.2f ", b[i]);
    printf("\n");
    return 0;
}
```

#### Why It Matters

- Converts differential equations into integral form.
- Core for symbolic calculus and automatic integration.
- Used in Taylor series reconstruction, antiderivative computation, and formal power series analysis.
- In computational contexts, enables integration modulo p for exact algebraic manipulation.

#### A Gentle Proof

We know from calculus:
$$
\frac{d}{dx}(x^{i+1})=(i+1)x^i.
$$

Thus, to "undo" differentiation, each term's coefficient is divided by $(i+1)$:
$$
\int a_i x^i dx = \frac{a_i}{i+1}x^{i+1}.
$$

Linearity of integration guarantees the full polynomial follows the same rule.

#### Try It Yourself

1. Integrate $A(x)=3+4x+5x^2$ with $C=0$.
2. Add a nonzero constant of integration $C=7$.
3. Verify by differentiating your result.
4. Implement integration under modulo arithmetic ($\text{mod }p$).
5. Use integration to compute area under a polynomial curve from $x=0$ to $x=1$.

#### Test Cases

| $A(x)$        | $B(x)$                          | Verification               |
| ------------- | ------------------------------- | -------------------------- |
| $2+10x+12x^2$ | $2x+5x^2+4x^3$                  | $B'(x)=A(x)$               |
| $1+x+x^2$     | $x+\frac{x^2}{2}+\frac{x^3}{3}$ | derivative recovers $A(x)$ |
| $5x^2$        | $\frac{5x^3}{3}$                | derivative gives $5x^2$    |

#### Complexity

- Time: $O(n)$
- Space: $O(n)$

Polynomial integration is straightforward but fundamental, turning discrete coefficients into a continuous curve, one fraction at a time.

### 595 Formal Power Series Composition

Formal power series (FPS) composition is the operation of substituting one series into another,
$$
C(x)=A(B(x)),
$$
where both $A(x)$ and $B(x)$ are formal power series with no concern for convergence, we only care about coefficients up to a chosen degree $n$.

It's a fundamental operation in algebraic combinatorics, symbolic computation, and generating function analysis.

#### What Problem Are We Solving?

Given
$$
A(x)=a_0+a_1x+a_2x^2+\dots,
$$
and
$$
B(x)=b_0+b_1x+b_2x^2+\dots,
$$
we want
$$
C(x)=A(B(x))=a_0+a_1B(x)+a_2(B(x))^2+a_3(B(x))^3+\dots.
$$

We compute only terms up to degree $n-1$.

#### Key Assumptions

1. Usually $b_0=0$ (so $B(x)$ has no constant term),
   otherwise $A(B(x))$ involves a constant shift.
2. The target degree $n$ limits all computations, truncate at each step.

#### How It Works

1. Initialize $C(x)=[a_0]$.
2. For each $k\ge1$:

   * Compute $(B(x))^k$ using repeated convolution or precomputed powers.
   * Multiply by $a_k$.
   * Add to $C(x)$, truncating after degree $n-1$.

Mathematically:
$$
C(x)=\sum_{k=0}^{n-1} a_k (B(x))^k \pmod{x^n}.
$$

Efficient algorithms use divide-and-conquer and FFT-based polynomial multiplications.

#### Example

Let
$$
A(x)=1+x+x^2, \quad B(x)=x+x^2.
$$

Then:
$$
A(B(x))=1+(x+x^2)+(x+x^2)^2=1+x+x^2+x^2+2x^3+x^4.
$$
Simplify:
$$
A(B(x))=1+x+2x^2+2x^3+x^4.
$$

Coefficient form: $[1,1,2,2,1]$.

#### Tiny Code

Python

```python
import numpy as np

def poly_mul(a, b, n):
    m = len(a) + len(b) - 1
    N = 1 << (m - 1).bit_length()
    fa = np.fft.rfft(a, N)
    fb = np.fft.rfft(b, N)
    fc = fa * fb
    c = np.fft.irfft(fc, N)[:n]
    return np.rint(c).astype(int)

def poly_compose(a, b, n):
    res = np.zeros(n, dtype=int)
    term = np.ones(1, dtype=int)
    for i in range(len(a)):
        if i > 0:
            term = poly_mul(term, b, n)
        res[:len(term)] += a[i] * term[:n]
    return res[:n]

# Example
A = [1, 1, 1]
B = [0, 1, 1]
print(poly_compose(A, B, 5))  # [1, 1, 2, 2, 1]
```

#### Why It Matters

- Composition is central in generating functions for combinatorial classes.
- Appears in power series reversion, functional iteration, and differential equations.
- Used in Taylor expansion of composite functions and symbolic algebra systems.
- Required for constructing exponential generating functions and series transformations.

#### A Gentle Proof

Using the formal definition:
$$
A(x)=\sum_{k=0}^\infty a_kx^k.
$$

Then substituting $B(x)$ gives
$$
A(B(x))=\sum_{k=0}^\infty a_k (B(x))^k.
$$

Since we only keep coefficients up to $x^{n-1}$, higher-degree terms vanish modulo $x^n$.
Each $(B(x))^k$ contributes terms of degree $\ge k$, ensuring finite computation.

#### Try It Yourself

1. Compute $A(B(x))$ for $A(x)=1+x+x^2$, $B(x)=x+x^2$.
2. Compare against symbolic expansion.
3. Test $A(x)=\exp(x)$ and $B(x)=\sin(x)$ up to degree 6.
4. Implement truncated FFT-based composition for large $n$.
5. Explore reversion: find $B(x)$ such that $A(B(x))=x$.

#### Test Cases

| $A(x)$    | $B(x)$  | Result up to $x^4$  |
| --------- | ------- | ------------------- |
| $1+x+x^2$ | $x+x^2$ | $1+x+2x^2+2x^3+x^4$ |
| $1+2x$    | $x+x^2$ | $1+2x+2x^2$         |
| $1+x^2$   | $x+x^2$ | $1+x^2+2x^3+x^4$    |

#### Complexity

- Naive: $O(n^2)$
- FFT-based: $O(n\log n)$ for each multiplication
- Divide-and-conquer composition: $O(n^{1.5})$ with advanced algorithms

Formal power series composition is where algebra meets structure, substituting one infinite series into another to build new functions, patterns, and generating laws.

### 596 Exponentiation by Squaring

Exponentiation by squaring is a fast method to compute powers efficiently.
Instead of multiplying repeatedly, it halves the exponent at each step, a divide-and-conquer strategy that reduces the time from $O(n)$ to $O(\log n)$ multiplications.

It's the workhorse behind modular exponentiation, fast matrix powers, and polynomial powering.

#### What Problem Are We Solving?

We want to compute
$$
a^n
$$
for integer (or polynomial, matrix) $a$ and nonnegative integer $n$.

Naively we'd multiply $a$ by itself $n$ times, which is slow.
Exponentiation by squaring uses the binary expansion of $n$ to skip unnecessary multiplications.

#### Key Idea

Use these rules:

$$
a^n =
\begin{cases}
1, & n = 0, \\[6pt]
\left(a^{\,n/2}\right)^2, & n \text{ even}, \\[6pt]
a \cdot \left(a^{\,(n-1)/2}\right)^2, & n \text{ odd}.
\end{cases}
$$


Each step halves $n$, so only $\log_2 n$ levels of recursion are needed.

#### Example

Compute $a^9$:

| Step  | Expression      | Simplified     |
| ----- | --------------- | -------------- |
| $a^9$ | $a\cdot(a^4)^2$ | uses odd rule  |
| $a^4$ | $(a^2)^2$       | uses even rule |
| $a^2$ | $(a^1)^2$       | base recursion |
| $a^1$ | $a$             | base case      |

Total: 4 multiplications instead of 8.

#### Tiny Code

Python

```python
def power(a, n):
    if n == 0:
        return 1
    if n % 2 == 0:
        half = power(a, n // 2)
        return half * half
    else:
        half = power(a, (n - 1) // 2)
        return a * half * half

print(power(3, 9))  # 19683
```

C

```c
#include <stdio.h>

long long power(long long a, long long n) {
    if (n == 0) return 1;
    if (n % 2 == 0) {
        long long half = power(a, n / 2);
        return half * half;
    } else {
        long long half = power(a, n / 2);
        return a * half * half;
    }
}

int main(void) {
    printf("%lld\n", power(3, 9)); // 19683
    return 0;
}
```

Modular version (Python)

```python
def modpow(a, n, m):
    res = 1
    a %= m
    while n > 0:
        if n & 1:
            res = (res * a) % m
        a = (a * a) % m
        n >>= 1
    return res

print(modpow(3, 200, 1000000007))  # fast
```

#### Why It Matters

- Used in modular arithmetic, cryptography, RSA, and discrete exponentiation.
- Essential for matrix exponentiation in dynamic programming and Fibonacci computation.
- Basis for polynomial powering, fast doubling, and fast modular inverse routines.

#### A Gentle Proof

Each step halves the exponent, maintaining equivalence:

For even $n$:
$$
a^n=(a^{n/2})^2.
$$
For odd $n$:
$$
a^n=a\cdot(a^{(n-1)/2})^2.
$$
Since each recursion uses $n/2$, total calls are $O(\log n)$.

#### Try It Yourself

1. Compute $2^{31}$ manually using repeated squaring.
2. Compare multiplication counts with naive $O(n)$ method.
3. Modify algorithm to work on $2\times2$ matrices.
4. Extend to polynomials using convolution for multiplication.
5. Implement an iterative version and benchmark both.

#### Test Cases

| Base $a$ | Exponent $n$ | Result        |
| -------- | ------------ | ------------- |
| $3$      | $9$          | $19683$       |
| $2$      | $10$         | $1024$        |
| $5$      | $0$          | $1$           |
| $7$      | $13$         | $96889010407$ |

#### Complexity

- Time: $O(\log n)$ multiplications
- Space: $O(\log n)$ (recursive) or $O(1)$ (iterative)

Exponentiation by squaring is the quintessential divide-and-conquer power trick, a perfect blend of simplicity, speed, and mathematical elegance.

### 597 Modular Exponentiation

Modular exponentiation efficiently computes
$$
a^b \bmod m
$$
without overflowing intermediate results.
It's fundamental in cryptography, primality testing, and modular arithmetic systems such as RSA and Diffie–Hellman.

#### What Problem Are We Solving?

We want to compute
$$
r=(a^b)\bmod m,
$$
where $a$, $b$, and $m$ are integers and $b$ may be very large.

Directly computing $a^b$ is impractical because the intermediate result grows exponentially.
We instead apply modular reduction at each multiplication step, using the rule:

$$
(xy)\bmod m=((x\bmod m)(y\bmod m))\bmod m.
$$

#### Key Idea

Combine modular arithmetic with exponentiation by squaring:

$$
a^b \bmod m =
\begin{cases}
1, & b = 0, \\[6pt]
\left((a^{\,b/2} \bmod m)^2\right) \bmod m, & b \text{ even}, \\[6pt]
\left(a \times (a^{\,(b-1)/2} \bmod m)^2\right) \bmod m, & b \text{ odd}.
\end{cases}
$$


Each step reduces both $b$ and intermediate values modulo $m$.

#### Example

Compute $3^{13}\bmod 17$:

| Step | $b$  | Operation                                     | Result   |
| ---- | ---- | --------------------------------------------- | -------- |
| 13   | odd  | $res=3$                                       | $res=3$  |
| 6    | even | square $3\to9$                                | $a=9$    |
| 3    | odd  | $res=res*a=3*9=27\Rightarrow27\bmod17=10$     | $res=10$ |
| 1    | odd  | $res=res*a=10*13=130\Rightarrow130\bmod17=11$ | Final    |

Result: $3^{13}\bmod17=11$

#### Tiny Code

Python

```python
def modexp(a, b, m):
    res = 1
    a %= m
    while b > 0:
        if b & 1:
            res = (res * a) % m
        a = (a * a) % m
        b >>= 1
    return res

print(modexp(3, 13, 17))  # 11
```

C

```c
#include <stdio.h>

long long modexp(long long a, long long b, long long m) {
    long long res = 1;
    a %= m;
    while (b > 0) {
        if (b & 1)
            res = (res * a) % m;
        a = (a * a) % m;
        b >>= 1;
    }
    return res;
}

int main(void) {
    printf("%lld\n", modexp(3, 13, 17)); // 11
    return 0;
}
```

#### Why It Matters

- Core of RSA encryption/decryption, Diffie–Hellman key exchange, and ElGamal systems.
- Essential in modular inverses, hashing, and primitive root computations.
- Enables large computations like $a^{10^{18}}\bmod m$ to run in microseconds.
- Used in Fermat, Miller–Rabin, and Carmichael tests for primality.

#### A Gentle Proof

For any modulus $m$,
$$
(a\times b)\bmod m=((a\bmod m)(b\bmod m))\bmod m.
$$
This property lets us reduce after every multiplication.

Using binary exponentiation,
$$
a^b=\prod_{i=0}^{k-1} a^{2^i\cdot d_i},
$$
where $b=\sum_i d_i2^i$.
We only multiply terms where $d_i=1$, keeping everything modulo $m$ to stay bounded.

#### Try It Yourself

1. Compute $5^{117}\bmod19$.
2. Compare against built-in `pow(5, 117, 19)` in Python.
3. Modify to handle $b<0$ using modular inverses.
4. Extend to matrices modulo $m$.
5. Prove that $a^{p-1}\bmod p=1$ for prime $p$ (Fermat's Little Theorem).

#### Test Cases

| $a$ | $b$ | $m$  | $a^b\bmod m$ |
| --- | --- | ---- | ------------ |
| 3   | 13  | 17   | 11           |
| 2   | 10  | 1000 | 24           |
| 7   | 256 | 13   | 9            |
| 5   | 117 | 19   | 1            |

#### Complexity

- Time: $O(\log b)$ multiplications
- Space: $O(1)$

Modular exponentiation is one of the most elegant and essential routines in computational number theory, small, exact, and powerful enough to secure the internet.

### 598 Fast Walsh–Hadamard Transform (FWHT)

The Fast Walsh–Hadamard Transform (FWHT) is a divide-and-conquer algorithm for computing pairwise XOR-based convolutions efficiently.
It is the discrete analog of the Fast Fourier Transform, but instead of multiplication under addition, it works under the bitwise XOR operation.

#### What Problem Are We Solving?

Given two sequences
$$
A=(a_0,a_1,\dots,a_{n-1}), \quad B=(b_0,b_1,\dots,b_{n-1}),
$$
we want their XOR convolution defined as

$$
C[k]=\sum_{i\oplus j=k}a_i b_j,
$$
where $\oplus$ is the bitwise XOR.

Naively, this requires $O(n^2)$ work. FWHT reduces it to $O(n\log n)$.

#### Key Idea

The Walsh–Hadamard Transform (WHT) maps a vector to its XOR-domain form.
We can compute XOR convolutions by transforming both sequences, multiplying pointwise, and then inverting the transform.

Let $\text{FWT}$ denote the transform.
Then

$$
C=\text{FWT}^{-1}(\text{FWT}(A)\circ\text{FWT}(B)),
$$
where $\circ$ is element-wise multiplication.

#### Transform Definition

For $n=2^k$, define the fast Walsh–Hadamard transform (FWT) recursively.

Base case:
$$
\text{FWT}([a_0]) = [a_0].
$$

Recursive step: split $A$ into $A_1$ (first half) and $A_2$ (second half). After computing $\text{FWT}(A_1)$ and $\text{FWT}(A_2)$, combine as
$$
\forall\, i=0,\dots,\tfrac{n}{2}-1:\quad
A'[i] = A_1[i] + A_2[i],\qquad
A'[i+\tfrac{n}{2}] = A_1[i] - A_2[i].
$$

Inverse transform:
$$
\text{IFWT}(A') = \frac{1}{n}\,\text{FWT}(A').
$$

#### Example

Let $A=[1,2,3,4]$. Split into $A_1=[1,2]$ and $A_2=[3,4]$, then combine
$$
[\,1+3,\ 2+4,\ 1-3,\ 2-4\,] = [\,4,6,-2,-2\,].
$$
For longer vectors, apply the same split–recurse–combine pattern until length $1$.



#### Tiny Code

Python

```python
def fwht(a, inverse=False):
    n = len(a)
    h = 1
    while h < n:
        for i in range(0, n, h * 2):
            for j in range(h):
                x = a[i + j]
                y = a[i + j + h]
                a[i + j] = x + y
                a[i + j + h] = x - y
        h *= 2
    if inverse:
        for i in range(n):
            a[i] //= n
    return a

def xor_convolution(a, b):
    n = 1
    while n < max(len(a), len(b)):
        n *= 2
    a = a + [0] * (n - len(a))
    b = b + [0] * (n - len(b))
    A = fwht(a[:])
    B = fwht(b[:])
    C = [A[i] * B[i] for i in range(n)]
    C = fwht(C, inverse=True)
    return C

print(xor_convolution([1,2,3,4],[4,3,2,1]))
```

#### Why It Matters

- XOR convolution appears in subset transforms, bitmask dynamic programming, and Boolean algebra problems.
- Used in signal processing, error-correcting codes, and polynomial transforms over GF(2).
- Key to fast computations on hypercubes and bitwise domains.

#### A Gentle Proof

Each level of recursion computes pairwise sums and differences —
a linear transformation using the Hadamard matrix $H_n$:

$$
H_n=
\begin{bmatrix}
H_{n/2} & H_{n/2}\
H_{n/2} & -H_{n/2}
\end{bmatrix}.
$$

Since $H_nH_n^T=nI$, its inverse is $\frac{1}{n}H_n^T$, giving correctness and invertibility.

#### Try It Yourself

1. Compute the FWHT of $[1,1,0,0]$.
2. Perform XOR convolution of $[1,2,3,4]$ and $[4,3,2,1]$.
3. Modify to handle floating-point or modular arithmetic.
4. Implement inverse transform explicitly and verify recovery.
5. Compare timing with naive $O(n^2)$ approach.

#### Test Cases

| Input A   | Input B   | XOR Convolution Result |
| --------- | --------- | ---------------------- |
| [1,2]     | [3,4]     | [10, -2]               |
| [1,1,0,0] | [0,1,1,0] | [2,0,0,2]              |
| [1,2,3,4] | [4,3,2,1] | [20, 0, 0, 0]          |

#### Complexity

- Time: $O(n\log n)$
- Space: $O(n)$

The Fast Walsh–Hadamard Transform is the XOR-world twin of FFT, smaller, sharper, and just as elegant.

### 598 Fast Walsh–Hadamard Transform (FWHT)

The Fast Walsh–Hadamard Transform (FWHT) is a divide-and-conquer algorithm for computing pairwise XOR-based convolutions efficiently.
It is the discrete analog of the Fast Fourier Transform, but instead of multiplication under addition, it works under the bitwise XOR operation.

#### What Problem Are We Solving?

Given two sequences
$$
A=(a_0,a_1,\dots,a_{n-1}), \quad B=(b_0,b_1,\dots,b_{n-1}),
$$
we want their XOR convolution defined as

$$
C[k]=\sum_{i\oplus j=k}a_i b_j,
$$
where $\oplus$ is the bitwise XOR.

Naively, this requires $O(n^2)$ work. FWHT reduces it to $O(n\log n)$.

#### Key Idea

The Walsh–Hadamard Transform (WHT) maps a vector to its XOR-domain form.
We can compute XOR convolutions by transforming both sequences, multiplying pointwise, and then inverting the transform.

Let $\text{FWT}$ denote the transform.
Then

$$
C=\text{FWT}^{-1}(\text{FWT}(A)\circ\text{FWT}(B)),
$$
where $\circ$ is element-wise multiplication.

#### Transform Definition

For $n = 2^k$, recursively define:

$$
\text{FWT}(A) =
\begin{cases}
[a_0], & n = 1, \\[6pt]
\text{combine } \text{FWT}(A_1) \text{ and } \text{FWT}(A_2)
 & \text{for } n > 1.
\end{cases}
$$

The combination step:
$$
A'[i] = A_1[i] + A_2[i], \qquad
A'[i + n/2] = A_1[i] - A_2[i],
\quad i = 0, \dots, n/2 - 1.
$$

To invert, divide all results by $n$ after applying the same process.

#### Example

Let $A = [1, 2, 3, 4]$.

1. Split into $[1, 2]$ and $[3, 4]$  
   Combine:
   $$
   [1 + 3,\ 2 + 4,\ 1 - 3,\ 2 - 4] = [4, 6, -2, -2].
   $$

2. Apply recursively for longer vectors.


#### Tiny Code

Python

```python
def fwht(a, inverse=False):
    n = len(a)
    h = 1
    while h < n:
        for i in range(0, n, h * 2):
            for j in range(h):
                x = a[i + j]
                y = a[i + j + h]
                a[i + j] = x + y
                a[i + j + h] = x - y
        h *= 2
    if inverse:
        for i in range(n):
            a[i] //= n
    return a

def xor_convolution(a, b):
    n = 1
    while n < max(len(a), len(b)):
        n *= 2
    a = a + [0] * (n - len(a))
    b = b + [0] * (n - len(b))
    A = fwht(a[:])
    B = fwht(b[:])
    C = [A[i] * B[i] for i in range(n)]
    C = fwht(C, inverse=True)
    return C

print(xor_convolution([1,2,3,4],[4,3,2,1]))
```

#### Why It Matters

- XOR convolution appears in subset transforms, bitmask dynamic programming, and Boolean algebra problems.
- Used in signal processing, error-correcting codes, and polynomial transforms over GF(2).
- Key to fast computations on hypercubes and bitwise domains.

#### A Gentle Proof

Each level of recursion computes pairwise sums and differences —
a linear transformation using the Hadamard matrix $H_n$:

$$
H_n=
\begin{bmatrix}
H_{n/2} & H_{n/2}\
H_{n/2} & -H_{n/2}
\end{bmatrix}.
$$

Since $H_nH_n^T=nI$, its inverse is $\frac{1}{n}H_n^T$, giving correctness and invertibility.

#### Try It Yourself

1. Compute the FWHT of $[1,1,0,0]$.
2. Perform XOR convolution of $[1,2,3,4]$ and $[4,3,2,1]$.
3. Modify to handle floating-point or modular arithmetic.
4. Implement inverse transform explicitly and verify recovery.
5. Compare timing with naive $O(n^2)$ approach.

#### Test Cases

| Input A   | Input B   | XOR Convolution Result |
| --------- | --------- | ---------------------- |
| [1,2]     | [3,4]     | [10, -2]               |
| [1,1,0,0] | [0,1,1,0] | [2,0,0,2]              |
| [1,2,3,4] | [4,3,2,1] | [20, 0, 0, 0]          |

#### Complexity

- Time: $O(n\log n)$
- Space: $O(n)$

The Fast Walsh–Hadamard Transform is the XOR-world twin of FFT, smaller, sharper, and just as elegant.

### 599 Zeta Transform

The Zeta Transform is a combinatorial transform that accumulates values over subsets or supersets.
It is especially useful in subset dynamic programming (DP over bitmasks), inclusion–exclusion, and fast subset convolutions.

#### What Problem Are We Solving?

Given a function $f(S)$ defined over subsets $S$ of a universe $U$ (size $n$),
the *Zeta Transform* produces a new function $F(S)$ that sums over all subsets (or supersets) of $S$:

- Subset version
  $$
  F(S)=\sum_{T\subseteq S}f(T)
  $$

- Superset version
  $$
  F(S)=\sum_{T\supseteq S}f(T)
  $$

The transform runs in $O(n2^n)$ naively, but can be computed in $O(n2^n)$ efficiently using bit DP.

#### Key Idea

For subset zeta transform over bitmasks of size $n$:

For each bit $i$ from $0$ to $n-1$:

- For each mask $m$ from $0$ to $2^n-1$:

  * If bit $i$ is set in $m$, then add $f[m\text{ without bit }i]$ to $f[m]$.

In code:
$$
f[m]+=f[m\setminus{i}].
$$

This efficiently accumulates all contributions of subsets.

#### Example

Let $f$ over 3-bit subsets be:

| Mask | Subset  | $f(S)$ |
| ---- | ------- | ------ |
| 000  | ∅       | 1      |
| 001  | {0}     | 2      |
| 010  | {1}     | 3      |
| 011  | {0,1}   | 4      |
| 100  | {2}     | 5      |
| 101  | {0,2}   | 6      |
| 110  | {1,2}   | 7      |
| 111  | {0,1,2} | 8      |

After subset zeta transform, $F(S)=\sum_{T\subseteq S}f(T)$.

For $S={0,1}$ (mask 011):
$$
F(011)=f(000)+f(001)+f(010)+f(011)=1+2+3+4=10.
$$

#### Tiny Code

Python

```python
def subset_zeta_transform(f):
    n = len(f).bit_length() - 1
    F = f[:]
    for i in range(n):
        for mask in range(1 << n):
            if mask & (1 << i):
                F[mask] += F[mask ^ (1 << i)]
    return F

# Example: f for subsets of size 3
f = [1,2,3,4,5,6,7,8]
print(subset_zeta_transform(f))
```

C

```c
#include <stdio.h>

void subset_zeta_transform(int *f, int n) {
    for (int i = 0; i < n; i++) {
        for (int mask = 0; mask < (1 << n); mask++) {
            if (mask & (1 << i))
                f[mask] += f[mask ^ (1 << i)];
        }
    }
}

int main(void) {
    int f[8] = {1,2,3,4,5,6,7,8};
    subset_zeta_transform(f, 3);
    for (int i = 0; i < 8; i++) printf("%d ", f[i]);
    return 0;
}
```

#### Why It Matters

- Central in subset DP, SOS DP, and bitmask convolutions.
- Used for fast inclusion–exclusion and Möbius inversion.
- Appears in counting problems, graph subset enumeration, and Boolean function transforms.
- Works hand-in-hand with the Möbius Transform to invert accumulated results.

#### A Gentle Proof

Each iteration over bit $i$ ensures every subset without bit $i$ contributes to the superset with bit $i$.
By induction, all subsets $T\subseteq S$ accumulate into $F(S)$.

#### Try It Yourself

1. Compute Zeta transform manually for $n=3$.
2. Verify inversion using Möbius inversion (next section).
3. Modify code for superset version (flip the condition).
4. Implement modulo arithmetic (e.g., $\bmod 10^9+7$).
5. Use it to compute number of subsets with a given property.

#### Test Cases

| $f(S)$            | Subset | $F(S)$            |
| ----------------- | ------ | ----------------- |
| [1,2,3,4]         | 2 bits | [1,3,4,10]        |
| [0,1,1,0,1,0,0,0] | 3 bits | [0,1,1,2,1,2,2,4] |

#### Complexity

- Time: $O(n2^n)$
- Space: $O(2^n)$

The Zeta Transform is the summation heart of subset DP, a way to see all parts of a set at once, elegantly and efficiently.

### 600 Möbius Inversion

The Möbius Inversion is the mathematical inverse of the Zeta Transform.
It allows us to recover a function $f(S)$ from its accumulated form $F(S)$ where
$$F(S)=\sum_{T\subseteq S}f(T).$$
It is a cornerstone in combinatorics, number theory, and subset dynamic programming.

#### What Problem Are We Solving?

Suppose we know $F(S)$, the total contribution from all subsets $T\subseteq S$.
We want to invert this accumulation and get back the original $f(S)$.

The inversion formula is:

$$
f(S)=\sum_{T\subseteq S}(-1)^{|S\setminus T|}F(T).
$$

This is the combinatorial analog of differentiation for sums over subsets.

#### Key Idea

Möbius inversion reverses the cumulative effect of the subset Zeta transform.
In iterative (bitwise) form, we can perform it efficiently by "subtracting contributions" instead of adding them.

For each bit $i$ from $0$ to $n-1$:

If bit $i$ is set in $S$, subtract $f[S\setminus{i}]$ from $f[S]$:

$$
f[S]-=f[S\setminus{i}].
$$

This undoes the inclusion steps done in the Zeta transform.

#### Example

Let's start from
$$
F(S)=\sum_{T\subseteq S}f(T)
$$
and we know:

| Mask | Subset | $F(S)$ |
| ---- | ------ | ------ |
| 00   | ∅      | 1      |
| 01   | {0}    | 3      |
| 10   | {1}    | 4      |
| 11   | {0,1}  | 10     |

We apply Möbius inversion:

| Step  | Subset                                   | Computation   | Result |
| ----- | ---------------------------------------- | ------------- | ------ |
| ∅     |,                                        | $f(∅)=F(∅)=1$ | 1      |
| {0}   | $f(01)=F(01)-F(00)=3-1$                  | 2             |        |
| {1}   | $f(10)=F(10)-F(00)=4-1$                  | 3             |        |
| {0,1} | $f(11)=F(11)-F(01)-F(10)+F(00)=10-3-4+1$ | 4             |        |

So we recovered $f=[1,2,3,4]$.

#### Tiny Code

Python

```python
def mobius_inversion(F):
    n = len(F).bit_length() - 1
    f = F[:]
    for i in range(n):
        for mask in range(1 << n):
            if mask & (1 << i):
                f[mask] -= f[mask ^ (1 << i)]
    return f

# Example
F = [1,3,4,10]
print(mobius_inversion(F))  # [1,2,3,4]
```

C

```c
#include <stdio.h>

void mobius_inversion(int *F, int n) {
    for (int i = 0; i < n; i++) {
        for (int mask = 0; mask < (1 << n); mask++) {
            if (mask & (1 << i))
                F[mask] -= F[mask ^ (1 << i)];
        }
    }
}

int main(void) {
    int F[4] = {1,3,4,10};
    mobius_inversion(F, 2);
    for (int i = 0; i < 4; i++) printf("%d ", F[i]);
    return 0;
}
```

#### Why It Matters

- The backbone of inclusion–exclusion principle and subset DPs.
- Used in number-theoretic inversions such as the classic integer Möbius function μ(n).
- Pairs perfectly with Zeta transform to toggle between cumulative and pointwise representations.
- Appears in fast subset convolution, polynomial transforms, and combinatorial counting.

#### A Gentle Proof

From
$$
F(S)=\sum_{T\subseteq S}f(T),
$$
we can see the system as a triangular matrix with $1$s for $T\subseteq S$.
Its inverse has entries $(-1)^{|S\setminus T|}$ for $T\subseteq S$, giving

$$
f(S)=\sum_{T\subseteq S}(-1)^{|S\setminus T|}F(T).
$$

Thus Möbius inversion exactly cancels the overcounting in the Zeta transform.

#### Try It Yourself

1. Start with $f=[1,2,3,4]$ and compute its $F$ using the Zeta transform.
2. Apply Möbius inversion to get $f$ back.
3. Extend to 3-bit subsets ($n=3$).
4. Use it to compute inclusion–exclusion counts.
5. Modify to work modulo $10^9+7$.

#### Test Cases

| $F(S)$                | Expected $f(S)$   |
| --------------------- | ----------------- |
| [1,3,4,10]            | [1,2,3,4]         |
| [0,1,1,2]             | [0,1,1,0]         |
| [2,4,6,12,8,16,20,40] | [2,2,2,4,2,4,4,8] |

#### Complexity

- Time: $O(n2^n)$
- Space: $O(2^n)$

Möbius inversion closes the circle of combinatorial transforms, the mirror image of the Zeta transform, turning sums back into their sources.

