# Chapter 7. Strings and Text Algorithms 

# Section 61. String Matching 

### 601 Naive String Matching

Naive string matching is the simplest way to find a pattern inside a text. It checks every possible position in the text to see if the pattern fits. Though not the fastest, it's the most intuitive, perfect for understanding how pattern matching begins.

#### What Problem Are We Solving?

We're given:

- A text `T` of length `n`
- A pattern `P` of length `m`

We want to find all occurrences of `P` inside `T`.

Example:
Text: `"ABABABCABABABCAB"`
Pattern: `"ABABC"`

We need to check every possible starting position in `T` to see if all characters match.

#### How Does It Work (Plain Language)?

Imagine sliding the pattern across the text one character at a time.
At each position:

1. Compare all characters of the pattern with the text.
2. If all match, record a hit.
3. If a mismatch happens, slide one step and try again.

It's like looking through a magnifying glass, shift one letter, scan again.

Step-by-step example:

| Shift | Text Window | Match? | Reason               |
| ----- | ----------- | ------ | -------------------- |
| 0     | ABABA       | No     | mismatch at 5th char |
| 1     | BABAB       | No     | mismatch at 1st char |
| 2     | ABABC       | Ok      | full match           |
| 3     | BABCA       | No     | mismatch at 1st char |

We repeat until we reach the last valid window (n - m).

#### Tiny Code (Easy Versions)

C

```c
#include <stdio.h>
#include <string.h>

void naive_search(const char *text, const char *pattern) {
    int n = strlen(text);
    int m = strlen(pattern);
    for (int i = 0; i <= n - m; i++) {
        int j = 0;
        while (j < m && text[i + j] == pattern[j]) j++;
        if (j == m)
            printf("Match found at index %d\n", i);
    }
}

int main(void) {
    const char *text = "ABABABCABABABCAB";
    const char *pattern = "ABABC";
    naive_search(text, pattern);
}
```

Python

```python
def naive_search(text, pattern):
    n, m = len(text), len(pattern)
    for i in range(n - m + 1):
        if text[i:i+m] == pattern:
            print("Match found at index", i)

text = "ABABABCABABABCAB"
pattern = "ABABC"
naive_search(text, pattern)
```

#### Why It Matters

- Builds intuition for pattern matching.
- Basis for more advanced algorithms (KMP, Z, Rabin–Karp).
- Easy to implement and debug.
- Useful when `n` and `m` are small or comparisons are cheap.

#### Complexity

| Case  | Description                   | Comparisons | Time  |
| ----- | ----------------------------- | ----------- | ----- |
| Best  | First char mismatch each time | O(n)        | O(n)  |
| Worst | Almost full match each shift  | O(n·m)      | O(nm) |
| Space | Only indexes and counters     | O(1)        |       |

The naive method has O(nm) worst-case time, slow for large texts, but simple and deterministic.

#### Try It Yourself

1. Run the code on `"AAAAAA"` with pattern `"AAA"`.
   How many matches do you find?
2. Try `"ABCDE"` with pattern `"FG"`.
   How fast does it fail?
3. Measure number of comparisons for `n=10`, `m=3`.
4. Modify code to stop after the first match.
5. Extend it to case-insensitive matching.

Naive string matching is your first lens into the world of text algorithms. Simple, honest, and tireless, it checks every corner until it finds what you seek.

### 602 Knuth–Morris–Pratt (KMP)

Knuth–Morris–Pratt (KMP) is how we match patterns without backtracking. Instead of rechecking characters we've already compared, KMP uses prefix knowledge to skip ahead smartly. It's the first big leap from brute force to linear-time searching.

#### What Problem Are We Solving?

In naive search, when a mismatch happens, we move just one position and start over, wasting time re-checking prefixes. KMP fixes that.

We're solving:

> How can we reuse past comparisons to avoid redundant work?

Given:

- Text `T` of length `n`
- Pattern `P` of length `m`

We want all starting positions of `P` in `T`, in O(n + m) time.

Example:
Text: `"ABABABCABABABCAB"`
Pattern: `"ABABC"`

When mismatch happens at `P[j]`, we shift pattern using what we already know about its prefix and suffix.

#### How Does It Work (Plain Language)?

KMP has two main steps:

1. Preprocess Pattern (Build Prefix Table):
   Compute `lps[]` (longest proper prefix which is also suffix) for each prefix of `P`.
   This table tells us *how much we can safely skip* after a mismatch.

2. Scan Text Using Prefix Table:
   Compare text and pattern characters.
   When mismatch occurs at `j`, instead of restarting, jump `j = lps[j-1]`.

Think of `lps` as a "memory", it remembers how far we matched before the mismatch.

Example pattern: `"ABABC"`

| i | P[i] | LPS[i] | Explanation            |
| - | ---- | ------ | ---------------------- |
| 0 | A    | 0      | no prefix-suffix match |
| 1 | B    | 0      | "A"≠"B"                |
| 2 | A    | 1      | "A"                    |
| 3 | B    | 2      | "AB"                   |
| 4 | C    | 0      | no match               |

So `lps = [0, 0, 1, 2, 0]`

When mismatch happens at position 4 (`C`), we skip to index 2 in the pattern, no re-check needed.

#### Tiny Code (Easy Versions)

C

```c
#include <stdio.h>
#include <string.h>

void compute_lps(const char *pat, int m, int *lps) {
    int len = 0;
    lps[0] = 0;
    for (int i = 1; i < m;) {
        if (pat[i] == pat[len]) {
            lps[i++] = ++len;
        } else if (len != 0) {
            len = lps[len - 1];
        } else {
            lps[i++] = 0;
        }
    }
}

void kmp_search(const char *text, const char *pat) {
    int n = strlen(text), m = strlen(pat);
    int lps[m];
    compute_lps(pat, m, lps);

    int i = 0, j = 0;
    while (i < n) {
        if (text[i] == pat[j]) { i++; j++; }
        if (j == m) {
            printf("Match found at index %d\n", i - j);
            j = lps[j - 1];
        } else if (i < n && text[i] != pat[j]) {
            if (j != 0) j = lps[j - 1];
            else i++;
        }
    }
}

int main(void) {
    const char *text = "ABABABCABABABCAB";
    const char *pattern = "ABABC";
    kmp_search(text, pattern);
}
```

Python

```python
def compute_lps(p):
    m = len(p)
    lps = [0]*m
    length = 0
    i = 1
    while i < m:
        if p[i] == p[length]:
            length += 1
            lps[i] = length
            i += 1
        elif length != 0:
            length = lps[length - 1]
        else:
            lps[i] = 0
            i += 1
    return lps

def kmp_search(text, pat):
    n, m = len(text), len(pat)
    lps = compute_lps(pat)
    i = j = 0
    while i < n:
        if text[i] == pat[j]:
            i += 1; j += 1
        if j == m:
            print("Match found at index", i - j)
            j = lps[j - 1]
        elif i < n and text[i] != pat[j]:
            j = lps[j - 1] if j else i + 1 - (i - j)
            if j == 0: i += 1

text = "ABABABCABABABCAB"
pattern = "ABABC"
kmp_search(text, pattern)
```

#### Why It Matters

- Avoids re-checking, true linear time.
- Foundation for fast text search (editors, grep).
- Inspires other algorithms (Z-Algorithm, Aho–Corasick).
- Teaches preprocessing patterns, not just texts.

#### Complexity

| Phase             | Time     | Space |
| ----------------- | -------- | ----- |
| LPS preprocessing | O(m)     | O(m)  |
| Search            | O(n)     | O(1)  |
| Total             | O(n + m) | O(m)  |

Worst-case linear, every character checked once.

#### Try It Yourself

1. Build `lps` for `"AAAA"`, `"ABABAC"`, and `"AABAACAABAA"`.
2. Modify code to count total matches instead of printing.
3. Compare with naive search, count comparisons.
4. Visualize the `lps` table with arrows showing skips.
5. Search `"AAAAAB"` in `"AAAAAAAAB"`, notice the skip efficiency.

KMP is your first clever matcher, it never looks back, always remembers what it's learned, and glides across the text with confidence.

### 603 Z-Algorithm

The Z-Algorithm is a fast way to find pattern matches by precomputing how much of the prefix matches at every position. It builds a "Z-array" that measures prefix overlap, a clever mirror trick for string searching.

#### What Problem Are We Solving?

We want to find all occurrences of a pattern `P` in a text `T`, but without extra scanning or repeated comparisons.

Idea:
If we know how many characters match from the beginning of the string at each position, we can detect pattern matches instantly.

So we build a helper string:

```
S = P + '$' + T
```

and compute `Z[i]` = length of longest substring starting at `i` that matches prefix of `S`.

If `Z[i]` equals length of `P`, we found a match in `T`.

Example:
P = `"ABABC"`
T = `"ABABABCABABABCAB"`
S = `"ABABC$ABABABCABABABCAB"`

Whenever `Z[i] = len(P) = 5`, that's a full match.

#### How Does It Work (Plain Language)?

The Z-array encodes how much the string matches itself starting from each index.

We scan through `S` and maintain a window [L, R] representing the current rightmost match segment.
For each position `i`:

1. If `i > R`, compare from scratch.
2. Else, copy information from `Z[i-L]` inside the window.
3. Extend match beyond `R` if possible.

It's like using a mirror, if you already know a window of matches, you can skip redundant checks inside it.

Example for `"aabxaayaab"`:

| i | S[i] | Z[i] | Explanation   |
| - | ---- | ---- | ------------- |
| 0 | a    | 0    | (always 0)    |
| 1 | a    | 1    | matches "a"   |
| 2 | b    | 0    | mismatch      |
| 3 | x    | 0    | mismatch      |
| 4 | a    | 2    | matches "aa"  |
| 5 | a    | 1    | matches "a"   |
| 6 | y    | 0    | mismatch      |
| 7 | a    | 3    | matches "aab" |
| 8 | a    | 2    | matches "aa"  |
| 9 | b    | 1    | matches "a"   |

#### Tiny Code (Easy Versions)

C

```c
#include <stdio.h>
#include <string.h>

void compute_z(const char *s, int z[]) {
    int n = strlen(s);
    int L = 0, R = 0;
    z[0] = 0;
    for (int i = 1; i < n; i++) {
        if (i <= R) z[i] = (R - i + 1 < z[i - L]) ? (R - i + 1) : z[i - L];
        else z[i] = 0;
        while (i + z[i] < n && s[z[i]] == s[i + z[i]]) z[i]++;
        if (i + z[i] - 1 > R) { L = i; R = i + z[i] - 1; }
    }
}

void z_search(const char *text, const char *pat) {
    char s[1000];
    sprintf(s, "%s$%s", pat, text);
    int n = strlen(s);
    int z[n];
    compute_z(s, z);
    int m = strlen(pat);
    for (int i = 0; i < n; i++)
        if (z[i] == m)
            printf("Match found at index %d\n", i - m - 1);
}

int main(void) {
    const char *text = "ABABABCABABABCAB";
    const char *pattern = "ABABC";
    z_search(text, pattern);
}
```

Python

```python
def compute_z(s):
    n = len(s)
    z = [0] * n
    L = R = 0
    for i in range(1, n):
        if i <= R:
            z[i] = min(R - i + 1, z[i - L])
        while i + z[i] < n and s[z[i]] == s[i + z[i]]:
            z[i] += 1
        if i + z[i] - 1 > R:
            L, R = i, i + z[i] - 1
    return z

def z_search(text, pat):
    s = pat + '$' + text
    z = compute_z(s)
    m = len(pat)
    for i in range(len(z)):
        if z[i] == m:
            print("Match found at index", i - m - 1)

text = "ABABABCABABABCAB"
pattern = "ABABC"
z_search(text, pattern)
```

#### Why It Matters

- Linear-time pattern matching (O(n + m)).
- Builds intuition for prefix overlap and self-similarity.
- Used in pattern detection, DNA analysis, compression.
- Related to KMP, but often simpler to implement.

#### Complexity

| Step            | Time | Space |
| --------------- | ---- | ----- |
| Compute Z-array | O(n) | O(n)  |
| Search          | O(n) | O(1)  |

Total complexity: O(n + m)

#### Try It Yourself

1. Compute Z-array for `"AAABAAA"`.
2. Change `$` separator to other symbols, why must it differ?
3. Compare Z-array of `"abcabcabc"` and `"aaaaa"`.
4. Count how many positions have `Z[i] > 0`.
5. Visualize Z-box sliding across the string.

The Z-Algorithm reads strings like a mirror reads light, matching prefixes, skipping repetition, and revealing structure hidden in plain sight.

### 604 Rabin–Karp

Rabin–Karp is a clever algorithm that matches patterns using rolling hashes instead of character-by-character comparison. It turns strings into numbers, so comparing substrings becomes comparing integers. Fast, simple, and great for multi-pattern search.

#### What Problem Are We Solving?

We want to find all occurrences of a pattern `P` (length `m`) inside a text `T` (length `n`).

The naive approach compares substrings character by character.
Rabin–Karp instead compares hashes, if two substrings share the same hash, we only compare characters to confirm.

The trick is a rolling hash:
We can compute the hash of the next substring in O(1) time from the previous one.

Example:
Text: `"ABABABCABABABCAB"`
Pattern: `"ABABC"`
Instead of checking every 5-letter window, we roll a hash across the text, checking only when hashes match.

#### How Does It Work (Plain Language)?

1. Choose a base and modulus.
   Use a base `b` (like 256) and a large prime modulus `M` to reduce collisions.

2. Compute pattern hash.
   Compute hash of `P[0..m-1]`.

3. Compute first window hash in text.
   Hash `T[0..m-1]`.

4. Slide the window.
   For each shift `i`:

   * If `hash(T[i..i+m-1]) == hash(P)`, verify with character check.
   * Compute next hash efficiently:

     ```
     new_hash = (b * (old_hash - T[i]*b^(m-1)) + T[i+m]) mod M
     ```

It's like checking fingerprints:
If fingerprints match, then check the faces to confirm.

#### Example

Let's match `"AB"` in `"ABAB"`:

- base = 256, M = 101
- hash("AB") = (65×256 + 66) mod 101
- Slide window across `"ABAB"`:

  * window 0: `"AB"` → same hash → match
  * window 1: `"BA"` → different hash → skip
  * window 2: `"AB"` → same hash → match

Only two character checks total!

#### Tiny Code (Easy Versions)

C

```c
#include <stdio.h>
#include <string.h>

#define BASE 256
#define MOD 101

void rabin_karp(const char *text, const char *pat) {
    int n = strlen(text);
    int m = strlen(pat);
    int h = 1;
    for (int i = 0; i < m - 1; i++) h = (h * BASE) % MOD;

    int p = 0, t = 0;
    for (int i = 0; i < m; i++) {
        p = (BASE * p + pat[i]) % MOD;
        t = (BASE * t + text[i]) % MOD;
    }

    for (int i = 0; i <= n - m; i++) {
        if (p == t) {
            int match = 1;
            for (int j = 0; j < m; j++)
                if (text[i + j] != pat[j]) { match = 0; break; }
            if (match) printf("Match found at index %d\n", i);
        }
        if (i < n - m) {
            t = (BASE * (t - text[i] * h) + text[i + m]) % MOD;
            if (t < 0) t += MOD;
        }
    }
}

int main(void) {
    const char *text = "ABABABCABABABCAB";
    const char *pattern = "ABABC";
    rabin_karp(text, pattern);
}
```

Python

```python
def rabin_karp(text, pat, base=256, mod=101):
    n, m = len(text), len(pat)
    h = pow(base, m-1, mod)
    p_hash = t_hash = 0

    for i in range(m):
        p_hash = (base * p_hash + ord(pat[i])) % mod
        t_hash = (base * t_hash + ord(text[i])) % mod

    for i in range(n - m + 1):
        if p_hash == t_hash:
            if text[i:i+m] == pat:
                print("Match found at index", i)
        if i < n - m:
            t_hash = (base * (t_hash - ord(text[i]) * h) + ord(text[i+m])) % mod
            if t_hash < 0:
                t_hash += mod

text = "ABABABCABABABCAB"
pattern = "ABABC"
rabin_karp(text, pattern)
```

#### Why It Matters

- Enables efficient substring search with hashing.
- Supports multiple patterns (hash each pattern).
- Useful in plagiarism detection, data deduplication, bioinformatics.
- Introduces rolling hash, foundational for many algorithms (Karp–Rabin, Z, string fingerprints, Rabin fingerprints, Bloom filters).

#### Complexity

| Case                         | Time     | Space |
| ---------------------------- | -------- | ----- |
| Average                      | O(n + m) | O(1)  |
| Worst (many hash collisions) | O(nm)    | O(1)  |
| Expected (good hash)         | O(n + m) | O(1)  |

Rolling hash makes it *fast in practice*.

#### Try It Yourself

1. Use base = 10 and mod = 13, match `"31"` in `"313131"`.
2. Print hash values for each window, spot collisions.
3. Replace mod = 101 with a small number, what happens?
4. Try multiple patterns (like `"AB"`, `"ABC"`) together.
5. Compare Rabin–Karp's speed with naive search on large input.

Rabin–Karp turns text into numbers, matching becomes math. Slide the window, roll the hash, and let arithmetic guide your search. 

### 605 Boyer–Moore

Boyer–Moore is one of the fastest practical string search algorithms. It reads the text backward from the end of the pattern and skips large chunks of text on mismatches. It's built on two key insights: bad character rule and good suffix rule.

#### What Problem Are We Solving?

In naive and KMP algorithms, we move the pattern only one position when a mismatch occurs.
But what if we could skip multiple positions safely?

Boyer–Moore does exactly that, it looks from right to left, and when a mismatch happens, it uses precomputed tables to decide how far to shift.

Given:

- Text `T` of length `n`
- Pattern `P` of length `m`

We want to find all positions where `P` appears in `T`, with fewer comparisons.

Example:
Text: `"HERE IS A SIMPLE EXAMPLE"`
Pattern: `"EXAMPLE"`

Instead of scanning every position, Boyer–Moore might skip entire words.

#### How Does It Work (Plain Language)?

1. Preprocess pattern to build shift tables:

   * Bad Character Table:
     When mismatch at `P[j]` occurs, shift so that the last occurrence of `T[i]` in `P` aligns with position `j`.
     If `T[i]` not in pattern, skip whole length `m`.

   * Good Suffix Table:
     When suffix matches but mismatch happens before it, shift pattern to align with next occurrence of that suffix.

2. Search:

   * Align pattern with text.
   * Compare from right to left.
   * On mismatch, apply max shift from both tables.

It's like reading the text in reverse, you jump quickly when you know the mismatch tells you more than a match.

#### Example (Bad Character Rule)

Pattern: `"ABCD"`
Text: `"ZZABCXABCD"`

1. Compare `"ABCD"` with text segment ending at position 3
2. Mismatch at `X`
3. `X` not in pattern → shift by 4
4. New alignment starts at next possible match

Fewer comparisons, smarter skipping.

#### Tiny Code (Easy Version)

Python (Bad Character Rule Only)

```python
def bad_char_table(pat):
    table = [-1] * 256
    for i, ch in enumerate(pat):
        table[ord(ch)] = i
    return table

def boyer_moore(text, pat):
    n, m = len(text), len(pat)
    bad = bad_char_table(pat)
    i = 0
    while i <= n - m:
        j = m - 1
        while j >= 0 and pat[j] == text[i + j]:
            j -= 1
        if j < 0:
            print("Match found at index", i)
            i += (m - bad[ord(text[i + m])] if i + m < n else 1)
        else:
            i += max(1, j - bad[ord(text[i + j])])

text = "HERE IS A SIMPLE EXAMPLE"
pattern = "EXAMPLE"
boyer_moore(text, pattern)
```

This version uses only the bad character rule, which already gives strong performance for general text.

#### Why It Matters

- Skips large portions of text.
- Sublinear average time, often faster than O(n).
- Foundation for advanced variants:

  * Boyer–Moore–Horspool
  * Sunday algorithm
- Widely used in text editors, grep, search engines.

#### Complexity

| Case    | Time      | Space    |
| ------- | --------- | -------- |
| Best    | O(n / m)  | O(m + σ) |
| Average | Sublinear | O(m + σ) |
| Worst   | O(nm)     | O(m + σ) |

(σ = alphabet size)

In practice, one of the fastest algorithms for searching long patterns in long texts.

#### Try It Yourself

1. Trace `"ABCD"` in `"ZZABCXABCD"` step by step.
2. Print the bad character table, check shift values.
3. Add good suffix rule (advanced).
4. Compare with naive search for `"needle"` in `"haystack"`.
5. Measure comparisons, how many are skipped?

Boyer–Moore searches with hindsight. It looks backward, learns from mismatches, and leaps ahead, a masterclass in efficient searching.

### 606 Boyer–Moore–Horspool

The Boyer–Moore–Horspool algorithm is a streamlined version of Boyer–Moore. It drops the good-suffix rule and focuses on a single bad-character skip table, making it shorter, simpler, and often faster in practice for average cases.

#### What Problem Are We Solving?

Classic Boyer–Moore is powerful but complex, two tables, multiple rules, tricky to implement.

Boyer–Moore–Horspool keeps the essence of Boyer–Moore (right-to-left scanning and skipping) but simplifies logic so anyone can code it easily and get sublinear performance on average.

Given:

- Text `T` of length `n`
- Pattern `P` of length `m`

We want to find every occurrence of `P` in `T` with fewer comparisons than naive search, but with easy implementation.

#### How Does It Work (Plain Language)?

It scans the text right to left inside each alignment and uses a single skip table.

1. Preprocess pattern:
   For each character `c` in the alphabet:

   * `shift[c] = m`
     Then, for each pattern position `i` (0 to m−2):
   * `shift[P[i]] = m - i - 1`

2. Search phase:

   * Align pattern with text at position `i`
   * Compare pattern backward from `P[m-1]`
   * If mismatch, shift window by `shift[text[i + m - 1]]`
   * If match, report position and shift same way

Each mismatch may skip several characters at once.

#### Example

Text: `"EXAMPLEEXAMPLES"`
Pattern: `"EXAMPLE"`

Pattern length `m = 7`

Skip table (m−i−1 rule):

| Char   | Shift |
| ------ | ----- |
| E      | 6     |
| X      | 5     |
| A      | 4     |
| M      | 3     |
| P      | 2     |
| L      | 1     |
| others | 7     |

Scan right-to-left:

- Align `"EXAMPLE"` over text, compare from `L` backward
- On mismatch, look at last char under window → skip accordingly

Skips quickly over non-promising segments.

#### Tiny Code (Easy Versions)

Python

```python
def horspool(text, pat):
    n, m = len(text), len(pat)
    shift = {ch: m for ch in set(text)}
    for i in range(m - 1):
        shift[pat[i]] = m - i - 1

    i = 0
    while i <= n - m:
        j = m - 1
        while j >= 0 and pat[j] == text[i + j]:
            j -= 1
        if j < 0:
            print("Match found at index", i)
            i += shift.get(text[i + m - 1], m)
        else:
            i += shift.get(text[i + m - 1], m)

text = "EXAMPLEEXAMPLES"
pattern = "EXAMPLE"
horspool(text, pattern)
```

C (Simplified Version)

```c
#include <stdio.h>
#include <string.h>
#include <limits.h>

#define ALPHABET 256

void horspool(const char *text, const char *pat) {
    int n = strlen(text), m = strlen(pat);
    int shift[ALPHABET];
    for (int i = 0; i < ALPHABET; i++) shift[i] = m;
    for (int i = 0; i < m - 1; i++)
        shift[(unsigned char)pat[i]] = m - i - 1;

    int i = 0;
    while (i <= n - m) {
        int j = m - 1;
        while (j >= 0 && pat[j] == text[i + j]) j--;
        if (j < 0)
            printf("Match found at index %d\n", i);
        i += shift[(unsigned char)text[i + m - 1]];
    }
}

int main(void) {
    horspool("EXAMPLEEXAMPLES", "EXAMPLE");
}
```

#### Why It Matters

- Simpler than full Boyer–Moore.
- Fast in practice, especially on random text.
- Great choice when you need quick implementation and good performance.
- Used in editors and search tools for medium-length patterns.

#### Complexity

| Case    | Time      | Space |
| ------- | --------- | ----- |
| Best    | O(n / m)  | O(σ)  |
| Average | Sublinear | O(σ)  |
| Worst   | O(nm)     | O(σ)  |

σ = alphabet size (e.g., 256)

Most texts produce few comparisons per window → often faster than KMP.

#### Try It Yourself

1. Print skip table for `"ABCDAB"`.
2. Compare number of shifts with KMP on `"ABABABCABABABCAB"`.
3. Change one letter in pattern, how do skips change?
4. Count comparisons vs naive algorithm.
5. Implement skip table with dictionary vs array, measure speed.

Boyer–Moore–Horspool is like a lean racer, it skips ahead with confidence, cutting the weight but keeping the power.

### 607 Sunday Algorithm

The Sunday algorithm is a lightweight, intuitive string search method that looks ahead, instead of focusing on mismatches inside the current window, it peeks at the next character in the text to decide how far to jump. It's simple, elegant, and often faster than more complex algorithms in practice.

#### What Problem Are We Solving?

In naive search, we shift the pattern one step at a time.
In Boyer–Moore, we look backward at mismatched characters.
But what if we could peek one step forward instead, and skip the maximum possible distance?

The Sunday algorithm asks:

> "What's the character right after my current window?"
> If that character isn't in the pattern, skip the whole window.

Given:

- Text `T` (length `n`)
- Pattern `P` (length `m`)

We want to find all occurrences of `P` in `T` with fewer shifts, guided by the next unseen character.

#### How Does It Work (Plain Language)?

Think of sliding a magnifier over the text.
Each time you check a window, peek at the character just after it.

If it's not in the pattern, shift the pattern past it (by `m + 1`).
If it is in the pattern, align that character in the text with its last occurrence in the pattern.

Steps:

1. Precompute shift table: for each character `c` in the alphabet, `shift[c] = m - last_index(c)`
   Default shift for unseen characters: `m + 1`
2. Compare text and pattern left-to-right inside window.
3. If mismatch or no match, check the next character `T[i + m]` and shift accordingly.

It skips based on future information, not past mismatches, that's its charm.

#### Example

Text: `"EXAMPLEEXAMPLES"`
Pattern: `"EXAMPLE"`

m = 7

Shift table (from last occurrence):

| Char   | Shift |
| ------ | ----- |
| E      | 1     |
| X      | 2     |
| A      | 3     |
| M      | 4     |
| P      | 5     |
| L      | 6     |
| Others | 8     |

Steps:

- Compare `"EXAMPLE"` with `"EXAMPLE"` → match at 0
- Next char: `E` → shift by 1
- Compare next window → match again
  Quick, forward-looking, efficient.

#### Tiny Code (Easy Versions)

Python

```python
def sunday(text, pat):
    n, m = len(text), len(pat)
    shift = {ch: m - i for i, ch in enumerate(pat)}
    default = m + 1

    i = 0
    while i <= n - m:
        j = 0
        while j < m and pat[j] == text[i + j]:
            j += 1
        if j == m:
            print("Match found at index", i)
        next_char = text[i + m] if i + m < n else None
        i += shift.get(next_char, default)

text = "EXAMPLEEXAMPLES"
pattern = "EXAMPLE"
sunday(text, pattern)
```

C

```c
#include <stdio.h>
#include <string.h>

#define ALPHABET 256

void sunday(const char *text, const char *pat) {
    int n = strlen(text), m = strlen(pat);
    int shift[ALPHABET];
    for (int i = 0; i < ALPHABET; i++) shift[i] = m + 1;
    for (int i = 0; i < m; i++)
        shift[(unsigned char)pat[i]] = m - i;

    int i = 0;
    while (i <= n - m) {
        int j = 0;
        while (j < m && pat[j] == text[i + j]) j++;
        if (j == m)
            printf("Match found at index %d\n", i);
        unsigned char next = (i + m < n) ? text[i + m] : 0;
        i += shift[next];
    }
}

int main(void) {
    sunday("EXAMPLEEXAMPLES", "EXAMPLE");
}
```

#### Why It Matters

- Simple: one shift table, no backward comparisons.
- Fast in practice, especially for longer alphabets.
- Great balance between clarity and speed.
- Common in text editors, grep-like tools, and search libraries.

#### Complexity

| Case    | Time      | Space |
| ------- | --------- | ----- |
| Best    | O(n / m)  | O(σ)  |
| Average | Sublinear | O(σ)  |
| Worst   | O(nm)     | O(σ)  |

σ = alphabet size

On random text, very few comparisons per window.

#### Try It Yourself

1. Build shift table for `"HELLO"`.
2. Search `"LO"` in `"HELLOHELLO"`, trace each shift.
3. Compare skip lengths with Boyer–Moore–Horspool.
4. Try searching `"AAAB"` in `"AAAAAAAAAA"`, worst case?
5. Count total comparisons for `"ABCD"` in `"ABCDEABCD"`.

The Sunday algorithm looks to tomorrow, one step ahead, always skipping what it can see coming.

### 608 Finite Automaton Matching

Finite Automaton Matching turns pattern searching into state transitions. It precomputes a deterministic finite automaton (DFA) that recognizes exactly the strings ending with the pattern, then simply runs the automaton over the text. Every step is constant time, every match is guaranteed.

#### What Problem Are We Solving?

We want to match a pattern `P` in a text `T` efficiently, with no backtracking and no re-checking.

Idea:
Instead of comparing manually, we let a machine do the work, one that reads each character and updates its internal state until a match is found.

This algorithm builds a DFA where:

- Each state = how many characters of the pattern matched so far
- Each transition = what happens when we read a new character

Whenever the automaton enters the final state, a full match has been recognized.

#### How Does It Work (Plain Language)?

Think of it like a "pattern-reading machine."
Each time we read a character, we move to the next state, or fall back if it breaks the pattern.

Steps:

1. Preprocess the pattern:
   Build a DFA table: `dfa[state][char]` = next state
2. Scan the text:
   Start at state 0, feed each character of the text.
   Each character moves you to a new state using the table.
   If you reach state `m` (pattern length), that's a match.

Every character is processed exactly once, no backtracking.

#### Example

Pattern: `"ABAB"`

States: 0 → 1 → 2 → 3 → 4
Final state = 4 (full match)

| State | on 'A' | on 'B' | Explanation           |
| ----- | ------ | ------ | --------------------- |
| 0     | 1      | 0      | start → A             |
| 1     | 1      | 2      | after 'A', next 'B'   |
| 2     | 3      | 0      | after 'AB', next 'A'  |
| 3     | 1      | 4      | after 'ABA', next 'B' |
| 4     | -      | -      | match found           |

Feed the text `"ABABAB"` into this machine:

- Steps: 0→1→2→3→4 → match at index 0
- Continue: 2→3→4 → match at index 2

Every transition is O(1).

#### Tiny Code (Easy Versions)

Python

```python
def build_dfa(pat, alphabet):
    m = len(pat)
    dfa = [[0]*len(alphabet) for _ in range(m+1)]
    alpha_index = {ch: i for i, ch in enumerate(alphabet)}

    dfa[0][alpha_index[pat[0]]] = 1
    x = 0
    for j in range(1, m+1):
        for c in alphabet:
            dfa[j][alpha_index[c]] = dfa[x][alpha_index[c]]
        if j < m:
            dfa[j][alpha_index[pat[j]]] = j + 1
            x = dfa[x][alpha_index[pat[j]]]
    return dfa

def automaton_search(text, pat, alphabet):
    dfa = build_dfa(pat, alphabet)
    state = 0
    m = len(pat)
    for i, ch in enumerate(text):
        if ch in alphabet:
            state = dfa[state][alphabet.index(ch)]
        else:
            state = 0
        if state == m:
            print("Match found at index", i - m + 1)

alphabet = list("AB")
automaton_search("ABABAB", "ABAB", alphabet)
```

This builds a DFA and simulates it across the text.

#### Why It Matters

- No backtracking, linear time search.
- Perfect for fixed alphabet and repeated queries.
- Basis for lexical analyzers and regex engines (under the hood).
- Great example of automata theory in action.

#### Complexity

| Step      | Time     | Space    |
| --------- | -------- | -------- |
| Build DFA | O(m × σ) | O(m × σ) |
| Search    | O(n)     | O(1)     |

σ = alphabet size
Best for small alphabets (e.g., DNA, ASCII).

#### Try It Yourself

1. Draw DFA for `"ABA"`.
2. Simulate transitions for `"ABABA"`.
3. Add alphabet `{A, B, C}`, what changes?
4. Compare states with KMP's prefix table.
5. Modify code to print state transitions.

Finite Automaton Matching is like building a tiny machine that *knows your pattern by heart*, feed it text, and it will raise its hand every time it recognizes your word.

### 609 Bitap Algorithm

The Bitap algorithm (also known as Shift-Or or Shift-And) matches patterns using bitwise operations. It treats the pattern as a bitmask and processes the text character by character, updating a single integer that represents the match state. Fast, compact, and perfect for approximate or fuzzy matching too.

#### What Problem Are We Solving?

We want to find a pattern `P` in a text `T` efficiently using bit-level parallelism.

Rather than comparing characters in loops, Bitap packs comparisons into a single machine word, updating all positions in one go. It's like running multiple match states in parallel, using the bits of an integer.

Given:

- `T` of length `n`
- `P` of length `m` (≤ word size)
  We'll find matches in O(n) time with bitwise magic.

#### How Does It Work (Plain Language)?

Each bit in a word represents whether a prefix of the pattern matches the current suffix of the text.

We keep:

- `R`: current match state bitmask (1 = mismatch, 0 = match so far)
- `mask[c]`: precomputed bitmask for character `c` in pattern

At each step:

1. Shift `R` left (to include next char)
2. Combine with mask for current text char
3. Check if lowest bit is 0 → full match found

So instead of managing loops for each prefix, we update all match prefixes at once.

#### Example

Pattern: `"AB"`
Text: `"CABAB"`

Precompute masks (for 2-bit word):

```
mask['A'] = 0b10
mask['B'] = 0b01
```

Initialize `R = 0b11` (all ones)

Now slide through `"CABAB"`:

- C: `R = (R << 1 | 1) & mask['C']` → stays 1s
- A: shifts left, combine mask['A']
- B: shift, combine mask['B'] → match bit goes 0 → found match

All done in bitwise ops.

#### Tiny Code (Easy Versions)

Python

```python
def bitap_search(text, pat):
    m = len(pat)
    if m == 0:
        return
    if m > 63:
        raise ValueError("Pattern too long for 64-bit Bitap")

    # Build bitmask for pattern
    mask = {chr(i): ~0 for i in range(256)}
    for i, c in enumerate(pat):
        mask[c] &= ~(1 << i)

    R = ~1
    for i, c in enumerate(text):
        R = (R << 1) | mask.get(c, ~0)
        if (R & (1 << m)) == 0:
            print("Match found ending at index", i)
```

C (64-bit Version)

```c
#include <stdio.h>
#include <string.h>
#include <stdint.h>

void bitap(const char *text, const char *pat) {
    int n = strlen(text), m = strlen(pat);
    if (m > 63) return; // fits in 64-bit mask

    uint64_t mask[256];
    for (int i = 0; i < 256; i++) mask[i] = ~0ULL;
    for (int i = 0; i < m; i++)
        mask[(unsigned char)pat[i]] &= ~(1ULL << i);

    uint64_t R = ~1ULL;
    for (int i = 0; i < n; i++) {
        R = (R << 1) | mask[(unsigned char)text[i]];
        if ((R & (1ULL << m)) == 0)
            printf("Match found ending at index %d\n", i);
    }
}

int main(void) {
    bitap("CABAB", "AB");
}
```

#### Why It Matters

- Bit-parallel search, leverages CPU-level operations.
- Excellent for short patterns and fixed word size.
- Extendable to approximate matching (with edits).
- Core of tools like agrep (approximate grep) and bitap fuzzy search in editors.

#### Complexity

| Case          | Time          | Space |
| ------------- | ------------- | ----- |
| Typical       | O(n)          | O(σ)  |
| Preprocessing | O(m + σ)      | O(σ)  |
| Constraint    | m ≤ word size |       |

Bitap is *linear*, but limited by machine word length (e.g., ≤ 64 chars).

#### Try It Yourself

1. Search `"ABC"` in `"ZABCABC"`.
2. Print `R` in binary after each step.
3. Extend mask for ASCII or DNA alphabet.
4. Test with `"AAA"`, see overlapping matches.
5. Try fuzzy version: allow 1 mismatch (edit distance ≤ 1).

Bitap is like a bitwise orchestra, each bit plays its note, and together they tell you exactly when the pattern hits.

### 610 Two-Way Algorithm

The Two-Way algorithm is a linear-time string search method that combines prefix analysis and modular shifting. It divides the pattern into two parts and uses critical factorization to decide how far to skip after mismatches. Elegant and optimal, it guarantees O(n + m) time without heavy preprocessing.

#### What Problem Are We Solving?

We want a deterministic linear-time search that's:

- Faster than KMP on average
- Simpler than Boyer–Moore
- Provably optimal in worst case

The Two-Way algorithm achieves this by analyzing the pattern's periodicity before searching, so during scanning, it shifts intelligently, sometimes by the pattern's period, sometimes by the full length.

Given:

- Text `T` (length `n`)
- Pattern `P` (length `m`)

We'll find all matches with no backtracking and no redundant comparisons.

#### How Does It Work (Plain Language)?

The secret lies in critical factorization:

1. Preprocessing (Find Critical Position):
   Split `P` into `u` and `v` at a critical index such that:

   * `u` and `v` represent the lexicographically smallest rotation of `P`
   * They reveal the pattern's period

   This ensures efficient skips.

2. Search Phase:
   Scan `T` with a moving window.

   * Compare from left to right (forward pass).
   * On mismatch, shift by:

     * The pattern's period if partial match
     * The full pattern length if mismatch early

By alternating two-way scanning, it guarantees that no position is checked twice.

Think of it as KMP's structure + Boyer–Moore's skip, merged with mathematical precision.

#### Example

Pattern: `"ABABAA"`

1. Compute critical position, index 2 (between "AB" | "ABAA")
2. Pattern period = 2 (`"AB"`)
3. Start scanning `T = "ABABAABABAA"`:

   * Compare `"AB"` forward → match
   * Mismatch after `"AB"` → shift by period = 2
   * Continue scanning, guaranteed no recheck

This strategy leverages the internal structure of the pattern, skips are based on known repetition.

#### Tiny Code (Simplified Version)

Python (High-Level Idea)

```python
def critical_factorization(pat):
    m = len(pat)
    i, j, k = 0, 1, 0
    while i + k < m and j + k < m:
        if pat[i + k] == pat[j + k]:
            k += 1
        elif pat[i + k] > pat[j + k]:
            i = i + k + 1
            if i <= j:
                i = j + 1
            k = 0
        else:
            j = j + k + 1
            if j <= i:
                j = i + 1
            k = 0
    return min(i, j)

def two_way_search(text, pat):
    n, m = len(text), len(pat)
    if m == 0:
        return
    pos = critical_factorization(pat)
    period = max(pos, m - pos)
    i = 0
    while i <= n - m:
        j = 0
        while j < m and text[i + j] == pat[j]:
            j += 1
        if j == m:
            print("Match found at index", i)
        i += period if j >= pos else max(1, j - pos + 1)

text = "ABABAABABAA"
pattern = "ABABAA"
two_way_search(text, pattern)
```

This implementation finds the critical index first, then applies forward scanning with period-based shifts.

#### Why It Matters

- Linear time (worst case)
- No preprocessing tables needed
- Elegant use of periodicity theory
- Foundation for C standard library's strstr() implementation
- Handles both periodic and aperiodic patterns efficiently

#### Complexity

| Step                                   | Time     | Space |
| -------------------------------------- | -------- | ----- |
| Preprocessing (critical factorization) | O(m)     | O(1)  |
| Search                                 | O(n)     | O(1)  |
| Total                                  | O(n + m) | O(1)  |

Optimal deterministic complexity, no randomness, no collisions.

#### Try It Yourself

1. Find the critical index for `"ABCABD"`.
2. Visualize shifts for `"ABAB"` in `"ABABABAB"`.
3. Compare skip lengths with KMP and Boyer–Moore.
4. Trace state changes step by step.
5. Implement `critical_factorization` manually and confirm.

The Two-Way algorithm is a blend of theory and pragmatism, it learns the rhythm of your pattern, then dances across the text in perfect time.

# Section 62. Multi-Patterns Search 

### 611 Aho–Corasick Automaton

The Aho–Corasick algorithm is a classic solution for multi-pattern search. Instead of searching each keyword separately, it builds a single automaton that recognizes all patterns at once. Each character of the text advances the automaton, reporting every match immediately, multiple keywords, one pass.

#### What Problem Are We Solving?

We want to find all occurrences of multiple patterns within a given text.

Given:

- A set of patterns ( P = {p_1, p_2, \ldots, p_k} )
- A text ( T ) of length ( n )

We aim to find all positions ( i ) in ( T ) such that
$$
T[i : i + |p_j|] = p_j
$$
for some ( p_j \in P ).

Naive solution:
$$
O\Big(n \times \sum_{j=1}^{k} |p_j|\Big)
$$
Aho–Corasick improves this to:
$$
O(n + \sum_{j=1}^{k} |p_j| + \text{output\_count})
$$

#### How Does It Work (Plain Language)?

Aho–Corasick constructs a deterministic finite automaton (DFA) that recognizes all given patterns simultaneously.

The construction involves three steps:

1. Trie Construction
   Insert all patterns into a prefix tree. Each edge represents a character; each node represents a prefix.

2. Failure Links
   For each node, build a failure link to the longest proper suffix that is also a prefix in the trie.
   Similar to the fallback mechanism in KMP.

3. Output Links
   When a node represents a complete pattern, record it.
   If the failure link points to another terminal node, merge their outputs.

Search Phase:
Process the text character by character:

- If a transition for the current character exists, follow it.
- Otherwise, follow failure links until a valid transition is found.
- At each node, output all patterns ending here.

Result: one pass through the text, reporting all matches.

#### Example

Patterns:
$$
P = {\text{"he"}, \text{"she"}, \text{"his"}, \text{"hers"}}
$$
Text:
$$
T = \text{"ushers"}
$$

Trie structure (simplified):

```
(root)
 ├─ h ─ i ─ s*
 │    └─ e*
 │         └─ r ─ s*
 └─ s ─ h ─ e*
```

(* denotes a pattern endpoint)

Failure links:

- ( \text{"h"} \to \text{root} )
- ( \text{"he"} \to \text{"e"} ) (via root)
- ( \text{"she"} \to \text{"he"} )
- ( \text{"his"} \to \text{"is"} )

Text scanning:

- `u` → no edge, stay at root
- `s` → follow `s`
- `h` → `sh`
- `e` → `she` → report "she", "he"
- `r` → move to `her`
- `s` → `hers` → report "hers"

All patterns found in a single traversal.

#### Tiny Code (Python)

```python
from collections import deque

class AhoCorasick:
    def __init__(self, patterns):
        self.trie = [{}]
        self.fail = [0]
        self.output = [set()]
        for pat in patterns:
            self._insert(pat)
        self._build()

    def _insert(self, pat):
        node = 0
        for ch in pat:
            if ch not in self.trie[node]:
                self.trie[node][ch] = len(self.trie)
                self.trie.append({})
                self.fail.append(0)
                self.output.append(set())
            node = self.trie[node][ch]
        self.output[node].add(pat)

    def _build(self):
        q = deque()
        for ch, nxt in self.trie[0].items():
            q.append(nxt)
        while q:
            r = q.popleft()
            for ch, s in self.trie[r].items():
                q.append(s)
                f = self.fail[r]
                while f and ch not in self.trie[f]:
                    f = self.fail[f]
                self.fail[s] = self.trie[f].get(ch, 0)
                self.output[s] |= self.output[self.fail[s]]

    def search(self, text):
        node = 0
        for i, ch in enumerate(text):
            while node and ch not in self.trie[node]:
                node = self.fail[node]
            node = self.trie[node].get(ch, 0)
            for pat in self.output[node]:
                print(f"Match '{pat}' at index {i - len(pat) + 1}")

patterns = ["he", "she", "his", "hers"]
ac = AhoCorasick(patterns)
ac.search("ushers")
```

Output:

```
Match 'she' at index 1
Match 'he' at index 2
Match 'hers' at index 2
```

#### Why It Matters

- Multiple patterns found in a single pass
- No redundant comparisons or backtracking
- Used in:

  * Spam and malware detection
  * Intrusion detection systems (IDS)
  * Search engines and keyword scanners
  * DNA and protein sequence analysis

#### Complexity

| Step                | Time Complexity               | Space Complexity        |
| ------------------- | ----------------------------- | ----------------------- |
| Build Trie          | $O\!\left(\sum |p_i|\right)$  | $O\!\left(\sum |p_i|\right)$ |
| Build Failure Links | $O\!\left(\sum |p_i| \cdot \sigma\right)$ | $O\!\left(\sum |p_i|\right)$ |
| Search              | $O(n + \text{output\_count})$ | $O(1)$                  |

where $\sigma$ is the alphabet size.

Overall:
$$
O\!\left(n + \sum |p_i| + \text{output\_count}\right)
$$


#### Try It Yourself

1. Build the trie for $\{\text{"a"}, \text{"ab"}, \text{"bab"}\}$.
2. Trace failure links for `"ababbab"`.
3. Add patterns with shared prefixes, noting trie compression.
4. Print all outputs per node to understand overlaps.
5. Compare runtime with launching multiple KMP searches.


Aho–Corasick unites all patterns under one automaton, a single traversal, complete recognition, and perfect efficiency.

### 612 Trie Construction

A trie (pronounced *try*) is a prefix tree that organizes strings by their prefixes. Each edge represents a character, and each path from the root encodes a word. In multi-pattern search, the trie is the foundation of the Aho–Corasick automaton, capturing all keywords in a shared structure.

#### What Problem Are We Solving?

We want to store and query a set of strings efficiently, especially for prefix-based operations.

Given a pattern set
$$
P = {p_1, p_2, \ldots, p_k}
$$

we want a data structure that can:

- Insert all patterns in $O\left(\sum_{i=1}^{k} |p_i|\right)$
- Query whether a word or prefix exists
- Share common prefixes to save memory and time

Example
If
$$
P = {\texttt{"he"}, \texttt{"she"}, \texttt{"his"}, \texttt{"hers"}}
$$
we can store them in a single prefix tree, sharing overlapping paths like `h → e`.

#### How Does It Work (Plain Language)

A trie is built incrementally, one character at a time:

1. Start from the root node (empty prefix).
2. For each pattern $p$:

   * Traverse existing edges that match current characters.
   * Create new nodes if edges don't exist.
3. Mark the final node of each word as a terminal node.

Each node represents a prefix of one or more patterns.
Each leaf or terminal node marks a complete pattern.

It's like a branching roadmap, words share their starting path, then split where they differ.

#### Example

For
$$
P = {\texttt{"he"}, \texttt{"she"}, \texttt{"his"}, \texttt{"hers"}}
$$

Trie structure:

```
(root)
 ├── h ── e* ── r ── s*
 │     └── i ── s*
 └── s ── h ── e*
```

(* marks end of word)

- Prefix `"he"` is shared by `"he"`, `"hers"`, and `"his"`.
- `"she"` branches separately under `"s"`.

#### Tiny Code (Easy Versions)

Python

```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        node = self.root
        for ch in word:
            if ch not in node.children:
                node.children[ch] = TrieNode()
            node = node.children[ch]
        node.is_end = True

    def search(self, word):
        node = self.root
        for ch in word:
            if ch not in node.children:
                return False
            node = node.children[ch]
        return node.is_end

    def starts_with(self, prefix):
        node = self.root
        for ch in prefix:
            if ch not in node.children:
                return False
            node = node.children[ch]
        return True

# Example usage
patterns = ["he", "she", "his", "hers"]
trie = Trie()
for p in patterns:
    trie.insert(p)

print(trie.search("he"))       # True
print(trie.starts_with("sh"))  # True
print(trie.search("her"))      # False
```

C (Simplified)

```c
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

#define ALPHABET 26

typedef struct Trie {
    struct Trie *children[ALPHABET];
    bool is_end;
} Trie;

Trie* new_node() {
    Trie *node = calloc(1, sizeof(Trie));
    node->is_end = false;
    return node;
}

void insert(Trie *root, const char *word) {
    Trie *node = root;
    for (int i = 0; word[i]; i++) {
        int idx = word[i] - 'a';
        if (!node->children[idx])
            node->children[idx] = new_node();
        node = node->children[idx];
    }
    node->is_end = true;
}

bool search(Trie *root, const char *word) {
    Trie *node = root;
    for (int i = 0; word[i]; i++) {
        int idx = word[i] - 'a';
        if (!node->children[idx]) return false;
        node = node->children[idx];
    }
    return node->is_end;
}

int main(void) {
    Trie *root = new_node();
    insert(root, "he");
    insert(root, "she");
    insert(root, "his");
    insert(root, "hers");
    printf("%d\n", search(root, "she"));  // 1 (True)
}
```

#### Why It Matters

- Enables fast prefix queries and shared storage
- Core component of:

  * Aho–Corasick automaton
  * Autocomplete and suggestion engines
  * Spell checkers
  * Dictionary compression

Tries are also the basis for suffix trees, ternary search trees, and radix trees.

#### Complexity

| Operation                 | Time Complexity        | Space Complexity |          |                        |     |          |
| ------------------------- | ---------------------- | ---------------- | -------- | ---------------------- | --- | -------- |
| Insert word of length $m$ | $O(m)$                 | $O(\sigma m)$    |          |                        |     |          |
| Search word               | $O(m)$                 | $O(1)$           |          |                        |     |          |
| Prefix query              | $O(m)$                 | $O(1)$           |          |                        |     |          |
| Build from $k$ patterns   | $O\left(\sum_{i=1}^{k} | p_i              | \right)$ | $O\left(\sum_{i=1}^{k} | p_i | \right)$ |

where $\sigma$ is the alphabet size (e.g. 26 for lowercase letters).

#### Try It Yourself

1. Build a trie for
   $$
   P = {\texttt{"a"}, \texttt{"ab"}, \texttt{"abc"}, \texttt{"b"}}
   $$
2. Trace the path for `"abc"`.
3. Modify code to print all words in lexicographic order.
4. Compare with a hash table, how does prefix lookup differ?
5. Extend each node to store frequency counts or document IDs.

Trie construction is the first step in multi-pattern search, a shared tree of prefixes that transforms a list of words into a single searchable structure.

### 613 Failure Link Computation

In the Aho–Corasick automaton, failure links are what give the structure its power. They allow the search to continue efficiently when a mismatch occurs, much like how the prefix function in KMP prevents redundant comparisons. Each failure link connects a node to the longest proper suffix of its path that is also a prefix in the trie.

#### What Problem Are We Solving?

When scanning the text, a mismatch might occur at some node in the trie.
Naively, we would return all the way to the root and restart.

Failure links fix this by telling us:

> "If the current path fails, what's the next best place to continue matching?"

In other words, they allow the automaton to reuse partial matches, skipping redundant work while staying in valid prefix states.

#### How Does It Work (Plain Language)

Every node in the trie represents a prefix of some pattern.
If we can't extend with the next character, we follow a failure link to the next longest prefix that might still match.

Algorithm overview:

1. Initialize

   * Root's failure link = 0 (root)
   * Children of root → failure = 0

2. BFS Traversal
   Process the trie level by level:

   * For each node `u` and each outgoing edge labeled `c` to node `v`:

     1. Follow failure links from `u` until you find a node with edge `c`
     2. Set `fail[v]` = next node on that edge
     3. Merge outputs:
        $$
        \text{output}[v] \gets \text{output}[v] \cup \text{output}[\text{fail}[v]]
        $$

This ensures every node knows where to jump after mismatch, similar to KMP's prefix fallback, but generalized for all patterns.

#### Example

Let
$$
P = {\texttt{"he"}, \texttt{"she"}, \texttt{"his"}, \texttt{"hers"}}
$$

Step 1, Build Trie

```
(root)
 ├── h ── e* ── r ── s*
 │     └── i ── s*
 └── s ── h ── e*
```

(* marks end of word)

Step 2, Compute Failure Links

| Node | String | Failure Link | Explanation               |
| ---- | ------ | ------------ | ------------------------- |
| root | ε      | root         | base case                 |
| h    | "h"    | root         | no prefix                 |
| s    | "s"    | root         | no prefix                 |
| he   | "he"   | root         | no suffix of "he" matches |
| hi   | "hi"   | root         | same                      |
| sh   | "sh"   | h            | longest suffix is "h"     |
| she  | "she"  | he           | "he" is suffix and prefix |
| hers | "hers" | s            | suffix "s" is prefix      |

Now each node knows where to continue when a mismatch occurs.

#### Tiny Code (Easy Version)

Python

```python
from collections import deque

def build_failure_links(trie, fail, output):
    q = deque()
    # Initialize: root's children fail to root
    for ch, nxt in trie[0].items():
        fail[nxt] = 0
        q.append(nxt)

    while q:
        r = q.popleft()
        for ch, s in trie[r].items():
            q.append(s)
            f = fail[r]
            while f and ch not in trie[f]:
                f = fail[f]
            fail[s] = trie[f].get(ch, 0)
            output[s] |= output[fail[s]]
```

Usage context (inside Aho–Corasick build phase):

- `trie`: list of dicts `{char: next_state}`
- `fail`: list of failure links
- `output`: list of sets for pattern endpoints

After running this, every node has a valid `fail` pointer and merged outputs.

#### Why It Matters

- Prevents backtracking → linear-time scanning
- Shares partial matches across patterns
- Enables overlapping match detection
- Generalizes KMP's prefix fallback to multiple patterns

Without failure links, the automaton would degrade into multiple independent searches.

#### Complexity

| Step                | Time Complexity | Space Complexity |                |         |     |    |
| ------------------- | --------------- | ---------------- | -------------- | ------- | --- | -- |
| Build Failure Links | $O(\sum         | p_i              | \cdot \sigma)$ | $O(\sum | p_i | )$ |
| Merge Outputs       | $O(\sum         | p_i              | )$             | $O(\sum | p_i | )$ |

where $\sigma$ is alphabet size.

Each edge and node is processed exactly once.

#### Try It Yourself

1. Build the trie for
   $$
   P = {\texttt{"a"}, \texttt{"ab"}, \texttt{"bab"}}
   $$
2. Compute failure links step by step using BFS.
3. Visualize merged outputs at each node.
4. Compare with KMP's prefix table for `"abab"`.
5. Trace text `"ababbab"` through automaton transitions.

Failure links are the nervous system of the Aho–Corasick automaton, always pointing to the next best match, ensuring no time is wasted retracing steps.

### 614 Output Link Management

In the Aho–Corasick automaton, output links (or output sets) record which patterns end at each state. These links ensure that all matches, including overlapping and nested ones, are reported correctly during text scanning. Without them, some patterns would go unnoticed when one pattern is a suffix of another.

#### What Problem Are We Solving?

When multiple patterns share suffixes, a single node in the trie may represent the end of multiple words.

For example, consider
$$
P = {\texttt{"he"}, \texttt{"she"}, \texttt{"hers"}}
$$

When the automaton reaches the node for `"hers"`, it should also output `"he"` and `"she"` because those are suffixes recognized earlier.

We need a mechanism to:

- Record which patterns end at each node
- Follow failure links to include matches that end earlier in the chain

This is the role of output links, each node's output set accumulates all patterns recognized at that state.

#### How Does It Work (Plain Language)

Each node in the automaton has:

- A set of patterns that end exactly there
- A failure link that points to the next fallback state

When constructing the automaton, after setting a node's failure link:

1. Merge outputs from its failure node:
   $$
   \text{output}[u] \gets \text{output}[u] \cup \text{output}[\text{fail}[u]]
   $$
2. This ensures if a suffix of the current path is also a pattern, it is recognized.

During search, whenever we visit a node:

- Emit all patterns in $\text{output}[u]$
- Each represents a pattern ending at the current text position

#### Example

Patterns:
$$
P = {\texttt{"he"}, \texttt{"she"}, \texttt{"hers"}}
$$
Text: `"ushers"`

Trie (simplified):

```
(root)
 ├── h ── e* ── r ── s*
 └── s ── h ── e*
```

(* = pattern end)

Failure links:

- `"he"` → root
- `"she"` → `"he"`
- `"hers"` → `"s"`

Output links (after merging):

- $\text{output}["he"] = {\texttt{"he"}}$
- $\text{output}["she"] = {\texttt{"she"}, \texttt{"he"}}$
- $\text{output}["hers"] = {\texttt{"hers"}, \texttt{"he"}}$

So, when reaching `"she"`, both `"she"` and `"he"` are reported.
When reaching `"hers"`, `"hers"` and `"he"` are reported.

#### Tiny Code (Python Snippet)

```python
from collections import deque

def build_output_links(trie, fail, output):
    q = deque()
    for ch, nxt in trie[0].items():
        fail[nxt] = 0
        q.append(nxt)

    while q:
        r = q.popleft()
        for ch, s in trie[r].items():
            q.append(s)
            f = fail[r]
            while f and ch not in trie[f]:
                f = fail[f]
            fail[s] = trie[f].get(ch, 0)
            # Merge outputs from failure link
            output[s] |= output[fail[s]]
```

Explanation

- `trie`: list of dicts (edges)
- `fail`: list of failure pointers
- `output`: list of sets (patterns ending at node)

Each node inherits its failure node's outputs.
Thus, when a node is visited, printing `output[node]` gives all matches.

#### Why It Matters

- Enables complete match reporting
- Captures overlapping matches, e.g. `"he"` inside `"she"`
- Essential for correctness, without output merging, only longest matches would appear
- Used in search tools, intrusion detection systems, NLP tokenizers, and compilers

#### Complexity

| Step          | Time Complexity                          | Space Complexity             |
| ------------- | ----------------------------------------- | ---------------------------- |
| Merge outputs | $O\!\left(\sum |p_i|\right)$              | $O\!\left(\sum |p_i|\right)$ |
| Search phase  | $O(n + \text{output\_count})$             | $O(1)$                       |

Each node merges its outputs once during automaton construction.


#### Try It Yourself

1. Build the trie for
   $$
   P = {\texttt{"he"}, \texttt{"she"}, \texttt{"hers"}}
   $$
2. Compute failure links and merge outputs.
3. Trace the text `"ushers"` character by character.
4. Print $\text{output}[u]$ at each visited state.
5. Verify all suffix patterns are reported.

Output link management ensures no pattern is left behind. Every suffix, every overlap, every embedded word is captured, a complete record of recognition at every step.

### 615 Multi-Pattern Search

The multi-pattern search problem asks us to find all occurrences of multiple keywords within a single text. Instead of running separate searches for each pattern, we combine them into a single traversal, powered by the Aho–Corasick automaton. This approach is foundational for text analytics, spam filtering, and network intrusion detection.

#### What Problem Are We Solving?

Given:

- A set of patterns
  $$
  P = {p_1, p_2, \ldots, p_k}
  $$
- A text
  $$
  T = t_1 t_2 \ldots t_n
  $$

We need to find every occurrence of every pattern in ( P ) inside ( T ), including overlapping matches.

Naively, we could run KMP or Z-algorithm for each ( p_i ):
$$
O\Big(n \times \sum_{i=1}^k |p_i|\Big)
$$

But Aho–Corasick solves it in:
$$
O(n + \sum_{i=1}^k |p_i| + \text{output\_count})
$$

That is, one pass through the text, all matches reported.

#### How Does It Work (Plain Language)

The solution proceeds in three stages:

1. Build a Trie
   Combine all patterns into one prefix tree.

2. Compute Failure and Output Links

   * Failure links redirect the search after mismatches.
   * Output links collect all matched patterns at each state.

3. Scan the Text
   Move through the automaton using characters from ( T ).

   * If a transition exists, follow it.
   * If not, follow failure links until one does.
   * Every time a state is reached, output all patterns in `output[state]`.

In essence, we simulate k searches in parallel, sharing common prefixes and reusing progress across patterns.

#### Example

Let
$$
P = {\texttt{"he"}, \texttt{"she"}, \texttt{"his"}, \texttt{"hers"}}
$$
and
$$
T = \texttt{"ushers"}
$$

During scanning:

- `u` → root
- `s` → `"s"`
- `h` → `"sh"`
- `e` → `"she"` → report `"she"`, `"he"`
- `r` → `"her"`
- `s` → `"hers"` → report `"hers"`, `"he"`

Output:

```
she @ 1
he  @ 2
hers @ 2
```

All patterns are found in a single pass.

#### Tiny Code (Python Implementation)

```python
from collections import deque

class AhoCorasick:
    def __init__(self, patterns):
        self.trie = [{}]
        self.fail = [0]
        self.output = [set()]
        for pat in patterns:
            self._insert(pat)
        self._build()

    def _insert(self, pat):
        node = 0
        for ch in pat:
            if ch not in self.trie[node]:
                self.trie[node][ch] = len(self.trie)
                self.trie.append({})
                self.fail.append(0)
                self.output.append(set())
            node = self.trie[node][ch]
        self.output[node].add(pat)

    def _build(self):
        q = deque()
        for ch, nxt in self.trie[0].items():
            q.append(nxt)
        while q:
            r = q.popleft()
            for ch, s in self.trie[r].items():
                q.append(s)
                f = self.fail[r]
                while f and ch not in self.trie[f]:
                    f = self.fail[f]
                self.fail[s] = self.trie[f].get(ch, 0)
                self.output[s] |= self.output[self.fail[s]]

    def search(self, text):
        node = 0
        results = []
        for i, ch in enumerate(text):
            while node and ch not in self.trie[node]:
                node = self.fail[node]
            node = self.trie[node].get(ch, 0)
            for pat in self.output[node]:
                results.append((i - len(pat) + 1, pat))
        return results

patterns = ["he", "she", "his", "hers"]
ac = AhoCorasick(patterns)
print(ac.search("ushers"))
```

Output:

```
$$(1, 'she'), (2, 'he'), (2, 'hers')]
```

#### Why It Matters

- Handles multiple keywords simultaneously
- Reports overlapping and nested matches
- Widely used in:

  * Spam filters
  * Intrusion detection systems
  * Search engines
  * Plagiarism detectors
  * DNA sequence analysis

It's the gold standard for searching many patterns in one pass.

#### Complexity

| Step                | Time Complexity              | Space Complexity |                |         |     |    |
| ------------------- | ---------------------------- | ---------------- | -------------- | ------- | --- | -- |
| Build trie          | $O(\sum                      | p_i              | )$             | $O(\sum | p_i | )$ |
| Build failure links | $O(\sum                      | p_i              | \cdot \sigma)$ | $O(\sum | p_i | )$ |
| Search              | $O(n + \text{output\_count})$ | $O(1)$           |                |         |     |    |

where $\sigma$ is the alphabet size.

Total time:
$$
O(n + \sum |p_i| + \text{output\_count})
$$

#### Try It Yourself

1. Build a multi-pattern automaton for
   $$
   P = {\texttt{"ab"}, \texttt{"bc"}, \texttt{"abc"}}
   $$
   and trace it on `"zabcbcabc"`.
2. Compare with running KMP three times.
3. Count total transitions vs naive scanning.
4. Modify code to count number of matches only.
5. Extend it to case-insensitive search.

Multi-pattern search transforms a list of keywords into a single machine. Each step reads one character and reveals every pattern hiding in that text, one scan, complete coverage.

### 616 Dictionary Matching

Dictionary matching is a specialized form of multi-pattern search where the goal is to locate all occurrences of words from a fixed dictionary within a given text. Unlike single-pattern search (like KMP or Boyer–Moore), dictionary matching solves the problem for an entire vocabulary, all at once, using shared structure and efficient transitions.

#### What Problem Are We Solving?

We want to find every word from a dictionary inside a large body of text.

Given:

- A dictionary
  $$
  D = {w_1, w_2, \ldots, w_k}
  $$
- A text
  $$
  T = t_1 t_2 \ldots t_n
  $$

We must report all substrings of $T$ that match any $w_i \in D$.

Naive solution:

- Run KMP or Z-algorithm for each word:
  $O(n \times \sum |w_i|)$

Efficient solution (Aho–Corasick):

- Build automaton once: $O(\sum |w_i|)$
- Search in one pass: $O(n + \text{output\_count})$

So total:
$$
O(n + \sum |w_i| + \text{output\_count})
$$

#### How Does It Work (Plain Language)

The key insight is shared prefixes and failure links.

1. Trie Construction
   Combine all words from the dictionary into a single prefix tree.

2. Failure Links
   When a mismatch occurs, follow a failure pointer to the longest suffix that is still a valid prefix.

3. Output Sets
   Each node stores all dictionary words that end at that state.

4. Search Phase
   Scan $T$ one character at a time.

   * Follow existing edges if possible
   * Otherwise, follow failure links until a valid transition exists
   * Report all words in the current node's output set

Each position in $T$ is processed exactly once.

#### Example

Dictionary:
$$
D = {\texttt{"he"}, \texttt{"she"}, \texttt{"his"}, \texttt{"hers"}}
$$
Text:
$$
T = \texttt{"ushers"}
$$

Scanning:

| Step | Char | State | Output         |
| ---- | ---- | ----- | -------------- |
| 1    | u    | root  | ∅              |
| 2    | s    | s     | ∅              |
| 3    | h    | sh    | ∅              |
| 4    | e    | she   | {"she", "he"}  |
| 5    | r    | her   | ∅              |
| 6    | s    | hers  | {"hers", "he"} |

Matches:

- `"she"` at index 1
- `"he"` at index 2
- `"hers"` at index 2

All found in one traversal.

#### Tiny Code (Python Version)

```python
from collections import deque

class DictionaryMatcher:
    def __init__(self, words):
        self.trie = [{}]
        self.fail = [0]
        self.output = [set()]
        for word in words:
            self._insert(word)
        self._build()

    def _insert(self, word):
        node = 0
        for ch in word:
            if ch not in self.trie[node]:
                self.trie[node][ch] = len(self.trie)
                self.trie.append({})
                self.fail.append(0)
                self.output.append(set())
            node = self.trie[node][ch]
        self.output[node].add(word)

    def _build(self):
        q = deque()
        for ch, nxt in self.trie[0].items():
            q.append(nxt)
        while q:
            r = q.popleft()
            for ch, s in self.trie[r].items():
                q.append(s)
                f = self.fail[r]
                while f and ch not in self.trie[f]:
                    f = self.fail[f]
                self.fail[s] = self.trie[f].get(ch, 0)
                self.output[s] |= self.output[self.fail[s]]

    def search(self, text):
        node = 0
        results = []
        for i, ch in enumerate(text):
            while node and ch not in self.trie[node]:
                node = self.fail[node]
            node = self.trie[node].get(ch, 0)
            for word in self.output[node]:
                results.append((i - len(word) + 1, word))
        return results

dictionary = ["he", "she", "his", "hers"]
dm = DictionaryMatcher(dictionary)
print(dm.search("ushers"))
```

Output:

```
$$(1, 'she'), (2, 'he'), (2, 'hers')]
```

#### Why It Matters

- Solves dictionary word search efficiently
- Core technique in:

  * Text indexing and keyword filtering
  * Intrusion detection systems (IDS)
  * Linguistic analysis
  * Plagiarism and content matching
  * DNA or protein motif discovery

It scales beautifully, a single automaton handles hundreds of thousands of dictionary words.

#### Complexity

| Step            | Time Complexity              | Space Complexity |                |         |     |    |
| --------------- | ---------------------------- | ---------------- | -------------- | ------- | --- | -- |
| Build automaton | $O(\sum                      | w_i              | \cdot \sigma)$ | $O(\sum | w_i | )$ |
| Search          | $O(n + \text{output\_count})$ | $O(1)$           |                |         |     |    |

where $\sigma$ = alphabet size (e.g., 26 for lowercase letters).

#### Try It Yourself

1. Use
   $$
   D = {\texttt{"cat"}, \texttt{"car"}, \texttt{"cart"}, \texttt{"art"}}
   $$
   and $T = \texttt{"cartographer"}$.
2. Trace the automaton step by step.
3. Compare against running 4 separate KMP searches.
4. Modify the automaton to return positions and words.
5. Extend to case-insensitive dictionary matching.

Dictionary matching transforms a word list into a search automaton, every character of text advances all searches together, ensuring complete detection with no redundancy.

### 617 Dynamic Aho–Corasick

The Dynamic Aho–Corasick automaton extends the classical Aho–Corasick algorithm to handle insertions and deletions of patterns at runtime. It allows us to maintain a live dictionary, updating the automaton as new keywords arrive or old ones are removed, without rebuilding from scratch.

#### What Problem Are We Solving?

Standard Aho–Corasick assumes a static dictionary.
But in many real-world systems, the set of patterns changes over time:

- Spam filters receive new rules
- Network intrusion systems add new signatures
- Search engines update keyword lists

We need a way to insert or delete words dynamically while still searching in
$$
O(n + \text{output\_count})
$$
per query, without reconstructing the automaton each time.

So, our goal is an incrementally updatable pattern matcher.

#### How Does It Work (Plain Language)

We maintain the automaton incrementally:

1. Trie Insertion
   Add a new pattern by walking down existing nodes, creating new ones when needed.

2. Failure Link Update
   For each new node, compute its failure link:

   * Follow parent's failure link until a node with the same edge exists
   * Set new node's failure link to that target
   * Merge output sets:
     $$
     \text{output}[v] \gets \text{output}[v] \cup \text{output}[\text{fail}[v]]
     $$

3. Deletion (Optional)

   * Mark pattern as inactive (logical deletion)
   * Optionally perform lazy cleanup when needed

4. Query
   Search proceeds as usual, following transitions and failure links.

This incremental construction is like Aho–Corasick in motion, adding one word at a time while preserving correctness.

#### Example

Start with
$$
P_0 = {\texttt{"he"}, \texttt{"she"}}
$$
Build automaton.

Now insert `"hers"`:

- Walk: `h → e → r → s`
- Create nodes as needed
- Update failure links:

  * `fail("hers") = "s"`
  * Merge output from `"s"` (if any)

Next insert `"his"`:

- `h → i → s`
- Compute `fail("his") = "s"`
- Merge outputs

Now automaton recognizes all words in
$$
P = {\texttt{"he"}, \texttt{"she"}, \texttt{"hers"}, \texttt{"his"}}
$$
without full rebuild.

#### Tiny Code (Python Sketch)

```python
from collections import deque

class DynamicAC:
    def __init__(self):
        self.trie = [{}]
        self.fail = [0]
        self.output = [set()]

    def insert(self, word):
        node = 0
        for ch in word:
            if ch not in self.trie[node]:
                self.trie[node][ch] = len(self.trie)
                self.trie.append({})
                self.fail.append(0)
                self.output.append(set())
            node = self.trie[node][ch]
        self.output[node].add(word)
        self._update_failures(word)

    def _update_failures(self, word):
        node = 0
        for ch in word:
            nxt = self.trie[node][ch]
            if nxt == 0:
                continue
            if node == 0:
                self.fail[nxt] = 0
            else:
                f = self.fail[node]
                while f and ch not in self.trie[f]:
                    f = self.fail[f]
                self.fail[nxt] = self.trie[f].get(ch, 0)
                self.output[nxt] |= self.output[self.fail[nxt]]
            node = nxt

    def search(self, text):
        node = 0
        matches = []
        for i, ch in enumerate(text):
            while node and ch not in self.trie[node]:
                node = self.fail[node]
            node = self.trie[node].get(ch, 0)
            for w in self.output[node]:
                matches.append((i - len(w) + 1, w))
        return matches

# Usage
ac = DynamicAC()
ac.insert("he")
ac.insert("she")
ac.insert("hers")
print(ac.search("ushers"))
```

Output:

```
$$(1, 'she'), (2, 'he'), (2, 'hers')]
```

#### Why It Matters

- Supports real-time pattern updates
- Crucial for:

  * Live spam filters
  * Intrusion detection systems (IDS)
  * Adaptive search systems
  * Dynamic word lists in NLP pipelines

Unlike static Aho–Corasick, this version adapts as the dictionary evolves.

#### Complexity

| Operation                 | Time Complexity              | Space Complexity |
| ------------------------- | ---------------------------- | ---------------- |
| Insert word of length $m$ | $O(m \cdot \sigma)$          | $O(m)$           |
| Delete word               | $O(m)$                       | $O(1)$ (lazy)    |
| Search text               | $O(n + \text{output\_count})$ | $O(1)$           |

Each insertion updates only affected nodes.

#### Try It Yourself

1. Start with
   $$
   P = {\texttt{"a"}, \texttt{"ab"}}
   $$
   then add `"abc"` dynamically.
2. Print `fail` array after each insertion.
3. Try deleting `"ab"` (mark inactive).
4. Search text `"zabca"` after each change.
5. Compare rebuild vs incremental time.

Dynamic Aho–Corasick turns a static automaton into a living dictionary, always learning new words, forgetting old ones, and scanning the world in real time.

### 618 Parallel Aho–Corasick Search

The Parallel Aho–Corasick algorithm adapts the classical Aho–Corasick automaton for multi-threaded or distributed environments. It divides the input text or workload into independent chunks so that multiple processors can simultaneously search for patterns, enabling high-throughput keyword detection on massive data streams.

#### What Problem Are We Solving?

The classical Aho–Corasick algorithm scans the text sequentially.
For large-scale tasks, like scanning logs, DNA sequences, or network packets, this becomes a bottleneck.

We want to:

- Maintain linear-time matching
- Leverage multiple cores or machines
- Preserve correctness across chunk boundaries

So our goal is to search
$$
T = t_1 t_2 \ldots t_n
$$
against
$$
P = {p_1, p_2, \ldots, p_k}
$$
using parallel execution.

#### How Does It Work (Plain Language)

There are two major strategies for parallelizing Aho–Corasick:

##### 1. Text Partitioning (Input-Split Model)

- Split text $T$ into $m$ chunks:
  $$
  T = T_1 , T_2 , \ldots , T_m
  $$
- Assign each chunk to a worker thread.
- Each thread runs Aho–Corasick independently.
- Handle boundary cases (patterns overlapping chunk edges) by overlapping buffers of length equal to the longest pattern.

Pros: Simple, efficient for long texts
Cons: Requires overlap for correctness

##### 2. Automaton Partitioning (State-Split Model)

- Partition the state machine across threads or nodes.
- Each processor is responsible for a subset of patterns or states.
- Transitions are communicated via message passing (e.g., MPI).

Pros: Good for static, small pattern sets
Cons: Synchronization cost, complex state handoff

In both approaches:

- Each thread scans text in $O(|T_i| + \text{output\_count}_i)$
- Results are merged at the end.

#### Example (Text Partitioning)

Let
$$
P = {\texttt{"he"}, \texttt{"she"}, \texttt{"his"}, \texttt{"hers"}}
$$
and
$$
T = \texttt{"ushershehis"}
$$

Split $T$ into two parts with overlap of length 4 (max pattern length):

- Thread 1: `"ushersh"`
- Thread 2: `"shehis"`

Both threads run the same automaton.
At merge time, deduplicate matches in overlapping region.

Each finds:

- Thread 1 → `she@1`, `he@2`, `hers@2`
- Thread 2 → `she@6`, `he@7`, `his@8`

Final result = union of both sets.

#### Tiny Code (Parallel Example, Python Threads)

```python
from concurrent.futures import ThreadPoolExecutor

def search_chunk(ac, text, offset=0):
    matches = []
    node = 0
    for i, ch in enumerate(text):
        while node and ch not in ac.trie[node]:
            node = ac.fail[node]
        node = ac.trie[node].get(ch, 0)
        for pat in ac.output[node]:
            matches.append((offset + i - len(pat) + 1, pat))
    return matches

def parallel_search(ac, text, chunk_size, overlap):
    tasks = []
    results = []
    with ThreadPoolExecutor() as executor:
        for i in range(0, len(text), chunk_size):
            chunk = text[i : i + chunk_size + overlap]
            tasks.append(executor.submit(search_chunk, ac, chunk, i))
        for t in tasks:
            results.extend(t.result())
    # Optionally deduplicate overlapping matches
    return sorted(set(results))
```

Usage:

```python
patterns = ["he", "she", "his", "hers"]
ac = AhoCorasick(patterns)
print(parallel_search(ac, "ushershehis", chunk_size=6, overlap=4))
```

Output:

```
$$(1, 'she'), (2, 'he'), (2, 'hers'), (6, 'she'), (7, 'he'), (8, 'his')]
```

#### Why It Matters

- Enables real-time matching on large-scale data
- Used in:

  * Intrusion detection systems (IDS)
  * Big data text analytics
  * Log scanning and threat detection
  * Genome sequence analysis
  * Network packet inspection

Parallelism brings Aho–Corasick to gigabyte-per-second throughput.

#### Complexity

For $m$ threads:

| Step            | Time Complexity                                         | Space Complexity |    |         |     |    |
| --------------- | ------------------------------------------------------- | ---------------- | -- | ------- | --- | -- |
| Build automaton | $O(\sum                                                 | p_i              | )$ | $O(\sum | p_i | )$ |
| Search          | $O\left(\frac{n}{m} + \text{overlap}\right)$ per thread | $O(1)$           |    |         |     |    |
| Merge results   | $O(k)$                                                  | $O(k)$           |    |         |     |    |

Total time approximates
$$
O\left(\frac{n}{m} + \text{overlap} \cdot m\right)
$$

#### Try It Yourself

1. Split
   $$
   T = \texttt{"bananabanabanana"}
   $$
   and
   $$
   P = {\texttt{"ana"}, \texttt{"banana"}}
   $$
   into chunks with overlap = 6.
2. Verify all matches found.
3. Experiment with different chunk sizes and overlaps.
4. Compare single-thread vs multi-thread performance.
5. Extend to multiprocessing or GPU streams.

Parallel Aho–Corasick turns a sequential automaton into a scalable search engine, distributing the rhythm of matching across threads, yet producing a single, synchronized melody of results.

### 618 Parallel Aho–Corasick Search

The Parallel Aho–Corasick algorithm adapts the classical Aho–Corasick automaton for multi-threaded or distributed environments. It divides the input text or workload into independent chunks so that multiple processors can simultaneously search for patterns, enabling high-throughput keyword detection on massive data streams.

#### What Problem Are We Solving?

The classical Aho–Corasick algorithm scans the text sequentially.
For large-scale tasks, like scanning logs, DNA sequences, or network packets, this becomes a bottleneck.

We want to:

- Maintain linear-time matching
- Leverage multiple cores or machines
- Preserve correctness across chunk boundaries

So our goal is to search
$$
T = t_1 t_2 \ldots t_n
$$
against
$$
P = {p_1, p_2, \ldots, p_k}
$$
using parallel execution.

#### How Does It Work (Plain Language)

There are two major strategies for parallelizing Aho–Corasick:

##### 1. Text Partitioning (Input-Split Model)

- Split text $T$ into $m$ chunks:
  $$
  T = T_1 , T_2 , \ldots , T_m
  $$
- Assign each chunk to a worker thread.
- Each thread runs Aho–Corasick independently.
- Handle boundary cases (patterns overlapping chunk edges) by overlapping buffers of length equal to the longest pattern.

Pros: Simple, efficient for long texts
Cons: Requires overlap for correctness

##### 2. Automaton Partitioning (State-Split Model)

- Partition the state machine across threads or nodes.
- Each processor is responsible for a subset of patterns or states.
- Transitions are communicated via message passing (e.g., MPI).

Pros: Good for static, small pattern sets
Cons: Synchronization cost, complex state handoff

In both approaches:

- Each thread scans text in $O(|T_i| + \text{output\_count}_i)$
- Results are merged at the end.

#### Example (Text Partitioning)

Let
$$
P = {\texttt{"he"}, \texttt{"she"}, \texttt{"his"}, \texttt{"hers"}}
$$
and
$$
T = \texttt{"ushershehis"}
$$

Split $T$ into two parts with overlap of length 4 (max pattern length):

- Thread 1: `"ushersh"`
- Thread 2: `"shehis"`

Both threads run the same automaton.
At merge time, deduplicate matches in overlapping region.

Each finds:

- Thread 1 → `she@1`, `he@2`, `hers@2`
- Thread 2 → `she@6`, `he@7`, `his@8`

Final result = union of both sets.

#### Tiny Code (Parallel Example, Python Threads)

```python
from concurrent.futures import ThreadPoolExecutor

def search_chunk(ac, text, offset=0):
    matches = []
    node = 0
    for i, ch in enumerate(text):
        while node and ch not in ac.trie[node]:
            node = ac.fail[node]
        node = ac.trie[node].get(ch, 0)
        for pat in ac.output[node]:
            matches.append((offset + i - len(pat) + 1, pat))
    return matches

def parallel_search(ac, text, chunk_size, overlap):
    tasks = []
    results = []
    with ThreadPoolExecutor() as executor:
        for i in range(0, len(text), chunk_size):
            chunk = text[i : i + chunk_size + overlap]
            tasks.append(executor.submit(search_chunk, ac, chunk, i))
        for t in tasks:
            results.extend(t.result())
    # Optionally deduplicate overlapping matches
    return sorted(set(results))
```

Usage:

```python
patterns = ["he", "she", "his", "hers"]
ac = AhoCorasick(patterns)
print(parallel_search(ac, "ushershehis", chunk_size=6, overlap=4))
```

Output:

```
$$(1, 'she'), (2, 'he'), (2, 'hers'), (6, 'she'), (7, 'he'), (8, 'his')]
```

#### Why It Matters

- Enables real-time matching on large-scale data
- Used in:

  * Intrusion detection systems (IDS)
  * Big data text analytics
  * Log scanning and threat detection
  * Genome sequence analysis
  * Network packet inspection

Parallelism brings Aho–Corasick to gigabyte-per-second throughput.

#### Complexity

For $m$ threads:

| Step            | Time Complexity                                         | Space Complexity |    |         |     |    |
| --------------- | ------------------------------------------------------- | ---------------- | -- | ------- | --- | -- |
| Build automaton | $O(\sum                                                 | p_i              | )$ | $O(\sum | p_i | )$ |
| Search          | $O\left(\frac{n}{m} + \text{overlap}\right)$ per thread | $O(1)$           |    |         |     |    |
| Merge results   | $O(k)$                                                  | $O(k)$           |    |         |     |    |

Total time approximates
$$
O\left(\frac{n}{m} + \text{overlap} \cdot m\right)
$$

#### Try It Yourself

1. Split
   $$
   T = \texttt{"bananabanabanana"}
   $$
   and
   $$
   P = {\texttt{"ana"}, \texttt{"banana"}}
   $$
   into chunks with overlap = 6.
2. Verify all matches found.
3. Experiment with different chunk sizes and overlaps.
4. Compare single-thread vs multi-thread performance.
5. Extend to multiprocessing or GPU streams.

Parallel Aho–Corasick turns a sequential automaton into a scalable search engine, distributing the rhythm of matching across threads, yet producing a single, synchronized melody of results.

### 619 Compressed Aho–Corasick Automaton

The Compressed Aho–Corasick automaton is a space-optimized version of the classical Aho–Corasick structure. It preserves linear-time matching while reducing memory footprint through compact representations of states, transitions, and failure links, ideal for massive dictionaries or embedded systems.

#### What Problem Are We Solving?

The standard Aho–Corasick automaton stores:

- States: one per prefix of every pattern
- Transitions: explicit dictionary of outgoing edges per state
- Failure links and output sets

For large pattern sets (millions of words), this becomes memory-heavy:

$$
O(\sum |p_i| \cdot \sigma)
$$

We need a space-efficient structure that fits into limited memory while maintaining:

- Deterministic transitions
- Fast lookup (preferably $O(1)$ or $O(\log \sigma)$)
- Exact same matching behavior

#### How Does It Work (Plain Language)

Compression focuses on representation, not algorithmic change. The matching logic is identical, but stored compactly.

There are several key strategies:

##### 1. Sparse Transition Encoding

Instead of storing all $\sigma$ transitions per node, store only existing ones:

- Use hash tables or sorted arrays for edges
- Binary search per character lookup
- Reduces space from $O(\sum |p_i| \cdot \sigma)$ to $O(\sum |p_i|)$

##### 2. Double-Array Trie

Represent trie using two parallel arrays `base[]` and `check[]`:

- `base[s] + c` gives next state
- `check[next] = s` confirms parent
- Extremely compact and cache-friendly
- Used in tools like Darts and MARP

Transition:
$$
\text{next} = \text{base}[s] + \text{code}(c)
$$

##### 3. Bit-Packed Links

Store failure links and outputs in integer arrays or bitsets:

- Each node's fail pointer is a 32-bit integer
- Output sets replaced with compressed indices or flags

If a pattern ends at a node, mark it with a bitmask instead of a set.

##### 4. Succinct Representation

Use wavelet trees or succinct tries to store edges:

- Space near theoretical lower bound
- Transition queries in $O(\log \sigma)$
- Ideal for very large alphabets (e.g., Unicode, DNA)

#### Example

Consider patterns:
$$
P = {\texttt{"he"}, \texttt{"she"}, \texttt{"hers"}}
$$

Naive trie representation:

- Nodes: 8
- Transitions: stored as dicts `{char: next_state}`

Compressed double-array representation:

- `base = [0, 5, 9, ...]`
- `check = [-1, 0, 1, 1, 2, ...]`
- Transition: `next = base[state] + code(char)`

Failure links and output sets stored as arrays:

```
fail = [0, 0, 0, 1, 2, 3, ...]
output_flag = [0, 1, 1, 0, 1, ...]
```

This reduces overhead drastically.

#### Tiny Code (Python Example Using Sparse Dicts)

```python
class CompressedAC:
    def __init__(self):
        self.trie = [{}]
        self.fail = [0]
        self.output = [0]  # bitmask flag

    def insert(self, word, idx):
        node = 0
        for ch in word:
            if ch not in self.trie[node]:
                self.trie[node][ch] = len(self.trie)
                self.trie.append({})
                self.fail.append(0)
                self.output.append(0)
            node = self.trie[node][ch]
        self.output[node] |= (1 << idx)  # mark pattern end

    def build(self):
        from collections import deque
        q = deque()
        for ch, nxt in self.trie[0].items():
            q.append(nxt)
        while q:
            r = q.popleft()
            for ch, s in self.trie[r].items():
                q.append(s)
                f = self.fail[r]
                while f and ch not in self.trie[f]:
                    f = self.fail[f]
                self.fail[s] = self.trie[f].get(ch, 0)
                self.output[s] |= self.output[self.fail[s]]

    def search(self, text):
        node = 0
        for i, ch in enumerate(text):
            while node and ch not in self.trie[node]:
                node = self.fail[node]
            node = self.trie[node].get(ch, 0)
            if self.output[node]:
                print(f"Match bitmask {self.output[node]:b} at index {i}")
```

Bitmasks replace sets, shrinking memory and enabling fast OR-merges.

#### Why It Matters

- Memory-efficient: fits large dictionaries in RAM
- Cache-friendly: improves real-world performance
- Used in:

  * Spam filters with large rule sets
  * Embedded systems (firewalls, IoT)
  * Search appliances and anti-virus engines

Compressed tries are foundational in systems that trade small overhead for massive throughput.

#### Complexity

| Operation       | Time Complexity              | Space Complexity |                     |         |     |                 |
| --------------- | ---------------------------- | ---------------- | ------------------- | ------- | --- | --------------- |
| Insert patterns | $O(\sum                      | p_i              | )$                  | $O(\sum | p_i | )$ (compressed) |
| Build links     | $O(\sum                      | p_i              | \cdot \log \sigma)$ | $O(\sum | p_i | )$              |
| Search text     | $O(n + \text{output\_count})$ | $O(1)$           |                     |         |     |                 |

Compared to classical Aho–Corasick:

- Same asymptotic time
- Reduced constants and memory usage

#### Try It Yourself

1. Build both standard and compressed Aho–Corasick automata for
   $$
   P = {\texttt{"abc"}, \texttt{"abd"}, \texttt{"bcd"}, \texttt{"cd"}}
   $$
2. Measure number of nodes and memory size.
3. Compare performance on a 1 MB random text.
4. Experiment with bitmasks for output merging.
5. Visualize the `base[]` and `check[]` arrays.

A compressed Aho–Corasick automaton is lean yet powerful, every bit counts, every transition packed tight, delivering full pattern detection with a fraction of the space.

### 620 Extended Aho–Corasick with Wildcards

The Extended Aho–Corasick automaton generalizes the classic algorithm to handle wildcards, special symbols that can match any character. This version is essential for pattern sets containing flexible templates, such as `"he*o"`, `"a?b"`, or `"c*t"`. It allows robust multi-pattern matching in noisy or semi-structured data.

#### What Problem Are We Solving?

Traditional Aho–Corasick matches only exact patterns.
But many real-world queries require wildcard tolerance, for example:

- `"a?b"` → matches `"acb"`, `"adb"`, `"aeb"`
- `"he*o"` → matches `"hello"`, `"hero"`, `"heyo"`

Given:
$$
P = {p_1, p_2, \ldots, p_k}
$$
and wildcard symbol(s) such as `?` (single) or `*` (multi),
we need to find all substrings of text $T$ that match any pattern under wildcard semantics.

#### How Does It Work (Plain Language)

We extend the trie and failure mechanism to handle wildcard transitions.

There are two main wildcard models:

##### 1. Single-Character Wildcard (`?`)

- Represents exactly one arbitrary character
- At construction time, each `?` creates a universal edge from the current state
- During search, the automaton transitions to this state for any character

Formally:
$$
\delta(u, c) = v \quad \text{if } c \in \Sigma \text{ and edge labeled '?' exists from } u
$$

##### 2. Multi-Character Wildcard (`*`)

- Matches zero or more arbitrary characters
- Creates a self-loop plus a skip edge to next state
- Requires additional transitions:

  * $\text{stay}(u, c) = u$ for any $c$
  * $\text{skip}(u) = v$ (next literal after `*`)

This effectively blends regular expression semantics into the Aho–Corasick structure.

#### Example

Patterns:
$$
P = {\texttt{"he?o"}, \texttt{"a*c"}}
$$
Text:
$$
T = \texttt{"hero and abc and aac"}
$$

1. `"he?o"` matches `"hero"`
2. `"a*c"` matches `"abc"`, `"aac"`

Trie edges include:

```
(root)
 ├── h ─ e ─ ? ─ o*
 └── a ─ * ─ c*
```

Wildcard nodes expand transitions for all characters dynamically or by default edge handling.

#### Tiny Code (Python Sketch, `?` Wildcard)

```python
from collections import deque

class WildcardAC:
    def __init__(self):
        self.trie = [{}]
        self.fail = [0]
        self.output = [set()]

    def insert(self, word):
        node = 0
        for ch in word:
            if ch not in self.trie[node]:
                self.trie[node][ch] = len(self.trie)
                self.trie.append({})
                self.fail.append(0)
                self.output.append(set())
            node = self.trie[node][ch]
        self.output[node].add(word)

    def build(self):
        q = deque()
        for ch, nxt in self.trie[0].items():
            q.append(nxt)
        while q:
            r = q.popleft()
            for ch, s in self.trie[r].items():
                q.append(s)
                f = self.fail[r]
                while f and ch not in self.trie[f] and '?' not in self.trie[f]:
                    f = self.fail[f]
                self.fail[s] = self.trie[f].get(ch, self.trie[f].get('?', 0))
                self.output[s] |= self.output[self.fail[s]]

    def search(self, text):
        results = []
        node = 0
        for i, ch in enumerate(text):
            while node and ch not in self.trie[node] and '?' not in self.trie[node]:
                node = self.fail[node]
            node = self.trie[node].get(ch, self.trie[node].get('?', 0))
            for pat in self.output[node]:
                results.append((i - len(pat) + 1, pat))
        return results

patterns = ["he?o", "a?c"]
ac = WildcardAC()
for p in patterns:
    ac.insert(p)
ac.build()
print(ac.search("hero abc aac"))
```

Output:

```
$$(0, 'he?o'), (5, 'a?c'), (9, 'a?c')]
```

#### Why It Matters

- Supports pattern flexibility in structured data
- Useful in:

  * Log scanning with variable fields
  * Keyword search with templates
  * Malware and rule-based filtering
  * DNA sequence motif matching
- Extends beyond fixed strings toward regex-like matching, while remaining efficient

#### Complexity

| Operation       | Time Complexity | Space Complexity |                |         |     |    |
| --------------- | --------------- | ---------------- | -------------- | ------- | --- | -- |
| Build automaton | $O(\sum         | p_i              | \cdot \sigma)$ | $O(\sum | p_i | )$ |
| Search text     | $O(n \cdot d)$  | $O(1)$           |                |         |     |    |

where $d$ is the branching factor from wildcard transitions (usually small).
The automaton remains linear if wildcard edges are bounded.

#### Try It Yourself

1. Build an automaton for
   $$
   P = {\texttt{"c?t"}, \texttt{"b*g"}}
   $$
   and test on `"cat bag big cog bug"`.
2. Add overlapping wildcard patterns like `"a*a"` and `"aa*"`.
3. Visualize wildcard transitions in the trie.
4. Measure runtime with vs without wildcards.
5. Extend to both `?` and `*` handling.

The Extended Aho–Corasick with Wildcards brings together deterministic automata and pattern generalization, matching both the exact and the uncertain, all in one unified scan.

# Section 63. Suffix Structure 

### 621 Suffix Array (Naive)

The suffix array is a fundamental data structure in string algorithms, a sorted list of all suffixes of a string. It provides a compact and efficient way to perform substring search, pattern matching, and text indexing, forming the backbone of structures like the FM-index and Burrows–Wheeler Transform.

The naive algorithm constructs the suffix array by generating all suffixes, sorting them lexicographically, and recording their starting positions.

#### What Problem Are We Solving?

Given a string
$$
S = s_0 s_1 \ldots s_{n-1}
$$
we want to build an array
$$
SA[0 \ldots n-1]
$$
such that
$$
S[SA[0] \ldots n-1] < S[SA[1] \ldots n-1] < \cdots < S[SA[n-1] \ldots n-1]
$$
in lexicographic order.

Each entry in the suffix array points to the starting index of a suffix in sorted order.

Example:
Let
$$
S = \texttt{"banana"}
$$

All suffixes:

| Index | Suffix |
| ----- | ------ |
| 0     | banana |
| 1     | anana  |
| 2     | nana   |
| 3     | ana    |
| 4     | na     |
| 5     | a      |

Sorted lexicographically:

| Rank | Suffix | Index |
| ---- | ------ | ----- |
| 0    | a      | 5     |
| 1    | ana    | 3     |
| 2    | anana  | 1     |
| 3    | banana | 0     |
| 4    | na     | 4     |
| 5    | nana   | 2     |

Thus:
$$
SA = [5, 3, 1, 0, 4, 2]
$$

#### How Does It Work (Plain Language)

The naive algorithm proceeds as follows:

1. Generate all suffixes
   Create $n$ substrings starting at each index $i$.

2. Sort suffixes
   Compare strings lexicographically (like dictionary order).

3. Store indices
   Collect starting indices of sorted suffixes into array `SA`.

This is straightforward but inefficient, every suffix comparison may take $O(n)$ time, and there are $O(n \log n)$ comparisons.

#### Algorithm (Step by Step)

1. Initialize list of pairs:
   $$
   L = [(i, S[i:])] \quad \text{for } i = 0, 1, \ldots, n-1
   $$

2. Sort $L$ by the string component.

3. Output array:
   $$
   SA[j] = L[j].i
   $$

#### Tiny Code (Easy Versions)

Python

```python
def suffix_array_naive(s):
    n = len(s)
    suffixes = [(i, s[i:]) for i in range(n)]
    suffixes.sort(key=lambda x: x[1])
    return [idx for idx, _ in suffixes]

s = "banana"
print(suffix_array_naive(s))  # [5, 3, 1, 0, 4, 2]
```

C

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    int index;
    const char *suffix;
} Suffix;

int cmp(const void *a, const void *b) {
    Suffix *sa = (Suffix *)a;
    Suffix *sb = (Suffix *)b;
    return strcmp(sa->suffix, sb->suffix);
}

void build_suffix_array(const char *s, int sa[]) {
    int n = strlen(s);
    Suffix arr[n];
    for (int i = 0; i < n; i++) {
        arr[i].index = i;
        arr[i].suffix = s + i;
    }
    qsort(arr, n, sizeof(Suffix), cmp);
    for (int i = 0; i < n; i++)
        sa[i] = arr[i].index;
}

int main(void) {
    char s[] = "banana";
    int sa[6];
    build_suffix_array(s, sa);
    for (int i = 0; i < 6; i++)
        printf("%d ", sa[i]);
}
```

#### Why It Matters

- Enables fast substring search via binary search
- Foundation for LCP (Longest Common Prefix) and suffix tree construction
- Core in compressed text indexes like FM-index and BWT
- Simple, educational introduction to more advanced $O(n \log n)$ and $O(n)$ methods

#### Complexity

| Step              | Time                                    | Space    |
| ----------------- | --------------------------------------- | -------- |
| Generate suffixes | $O(n)$                                  | $O(n^2)$ |
| Sort              | $O(n \log n)$ comparisons × $O(n)$ each | $O(n)$   |
| Total             | $O(n^2 \log n)$                         | $O(n^2)$ |

Slow, but conceptually clear and easy to implement.

#### Try It Yourself

1. Compute suffix array for `"abracadabra"`.
2. Verify lexicographic order of suffixes.
3. Use binary search on `SA` to find substring `"bra"`.
4. Compare with suffix array built by doubling algorithm.
5. Optimize storage to avoid storing full substrings.

The naive suffix array is a pure, clear view of text indexing, every suffix, every order, one by one, simple to build, and a perfect first step toward the elegant $O(n \log n)$ and $O(n)$ algorithms that follow.

### 622 Suffix Array (Doubling Algorithm)

The doubling algorithm builds a suffix array in
$$
O(n \log n)
$$
time, a major improvement over the naive $O(n^2 \log n)$ approach. It works by ranking prefixes of length $2^k$, doubling that length each iteration, until the whole string is sorted.

This elegant idea, sorting by progressively longer prefixes, makes it both fast and conceptually clear.

#### What Problem Are We Solving?

Given a string
$$
S = s_0 s_1 \ldots s_{n-1}
$$
we want to compute its suffix array
$$
SA[0 \ldots n-1]
$$
such that
$$
S[SA[0]:] < S[SA[1]:] < \ldots < S[SA[n-1]:]
$$

Instead of comparing entire suffixes, we compare prefixes of length $2^k$, using integer ranks from the previous iteration, just like radix sort.

#### How Does It Work (Plain Language)

We'll sort suffixes by first 1 character, then 2, then 4, then 8, and so on.
At each stage, we assign each suffix a pair of ranks:

$$
\text{rank}*k[i] = (\text{rank}*{k-1}[i], \text{rank}_{k-1}[i + 2^{k-1}])
$$

We sort suffixes by this pair and assign new ranks. After $\log_2 n$ rounds, all suffixes are fully sorted.

#### Example

Let
$$
S = \texttt{"banana"}
$$

1. Initial ranks (by single character):

| Index | Char | Rank |
| ----- | ---- | ---- |
| 0     | b    | 1    |
| 1     | a    | 0    |
| 2     | n    | 2    |
| 3     | a    | 0    |
| 4     | n    | 2    |
| 5     | a    | 0    |

2. Sort by pairs (rank, next rank):

For $k = 1$ (prefix length $2$):

| i | pair   | suffix |
| - | ------ | ------ |
| 0 | (1, 0) | banana |
| 1 | (0, 2) | anana  |
| 2 | (2, 0) | nana   |
| 3 | (0, 2) | ana    |
| 4 | (2, 0) | na     |
| 5 | (0,-1) | a      |

Sort lexicographically by pair, assign new ranks.

Repeat doubling until ranks are unique.

Final
$$
SA = [5, 3, 1, 0, 4, 2]
$$

#### Algorithm (Step by Step)

1. Initialize ranks
   Sort suffixes by first character.

2. Iteratively sort by 2ᵏ prefixes
   For $k = 1, 2, \ldots$

   * Form tuples
     $$
     (rank[i], rank[i + 2^{k-1}])
     $$
   * Sort suffixes by these tuples
   * Assign new ranks

3. Stop when all ranks are distinct or $2^k \ge n$

#### Tiny Code (Python)

```python
def suffix_array_doubling(s):
    n = len(s)
    k = 1
    rank = [ord(c) for c in s]
    tmp = [0] * n
    sa = list(range(n))
    
    while True:
        sa.sort(key=lambda i: (rank[i], rank[i + k] if i + k < n else -1))
        
        tmp[sa[0]] = 0
        for i in range(1, n):
            prev = (rank[sa[i-1]], rank[sa[i-1]+k] if sa[i-1]+k < n else -1)
            curr = (rank[sa[i]], rank[sa[i]+k] if sa[i]+k < n else -1)
            tmp[sa[i]] = tmp[sa[i-1]] + (curr != prev)
        
        rank[:] = tmp[:]
        k *= 2
        if max(rank) == n-1:
            break
    return sa

s = "banana"
print(suffix_array_doubling(s))  # [5, 3, 1, 0, 4, 2]
```

#### Why It Matters

- Reduces naive $O(n^2 \log n)$ to $O(n \log n)$
- Foundation for Kasai's LCP computation
- Simple yet fast, widely used in practical suffix array builders
- Extensible to cyclic rotations and minimal string rotation

#### Complexity

| Step                | Time          | Space  |
| ------------------- | ------------- | ------ |
| Sorting (per round) | $O(n \log n)$ | $O(n)$ |
| Rounds              | $\log n$      |,      |
| Total           | $O(n \log n)$ | $O(n)$ |

#### Try It Yourself

1. Build suffix array for `"mississippi"`.
2. Trace first 3 rounds of ranking.
3. Verify sorted suffixes manually.
4. Compare runtime with naive method.
5. Use resulting ranks for LCP computation (Kasai's algorithm).

The doubling algorithm is the bridge from clarity to performance, iterative refinement, powers of two, and lex order, simple ranks revealing the full order of the string.

### 623 Kasai's LCP Algorithm

The Kasai algorithm computes the Longest Common Prefix (LCP) array from a suffix array in linear time.
It tells you, for every adjacent pair of suffixes in sorted order, how many characters they share at the beginning, revealing the structure of repetitions and overlaps inside the string.

#### What Problem Are We Solving?

Given a string $S$ and its suffix array $SA$,
we want to compute the LCP array, where:

$$
LCP[i] = \text{length of common prefix between } S[SA[i]] \text{ and } S[SA[i-1]]
$$

This allows us to answer questions like:

- How many repeated substrings exist?
- What's the longest repeated substring?
- What's the similarity between adjacent suffixes?

Example:

Let
$$
S = \texttt{"banana"}
$$
and
$$
SA = [5, 3, 1, 0, 4, 2]
$$

| i | SA[i] | Suffix | LCP[i] |
| - | ----- | ------ | ------ |
| 0 | 5     | a      |,      |
| 1 | 3     | ana    | 1      |
| 2 | 1     | anana  | 3      |
| 3 | 0     | banana | 0      |
| 4 | 4     | na     | 0      |
| 5 | 2     | nana   | 2      |

So:
$$
LCP = [0, 1, 3, 0, 0, 2]
$$

#### How Does It Work (Plain Language)

Naively comparing each adjacent pair of suffixes costs $O(n^2)$.
Kasai's trick: reuse previous LCP computation.

If $h$ is the LCP of $S[i:]$ and $S[j:]$,
then the LCP of $S[i+1:]$ and $S[j+1:]$ is at least $h-1$.
So we slide down one character at a time, reusing previous overlap.

#### Step-by-Step Algorithm

1. Build inverse suffix array `rank[]` such that
   $$
   \text{rank}[SA[i]] = i
   $$

2. Initialize $h = 0$

3. For each position $i$ in $S$:

   * If $rank[i] > 0$:

     * Let $j = SA[rank[i] - 1]$
     * Compare $S[i+h]$ and $S[j+h]$
     * Increase $h$ while they match
     * Set $LCP[rank[i]] = h$
     * Decrease $h$ by 1 (for next iteration)

This makes sure each character is compared at most twice.

#### Example Walkthrough

For `"banana"`:

Suffix array:
$$
SA = [5, 3, 1, 0, 4, 2]
$$
Inverse rank:
$$
rank = [3, 2, 5, 1, 4, 0]
$$

Now iterate $i$ from 0 to 5:

| i | rank[i] | j = SA[rank[i]-1] | Compare         | h | LCP[rank[i]] |
| - | ------- | ----------------- | --------------- | - | ------------ |
| 0 | 3       | 1                 | banana vs anana | 0 | 0            |
| 1 | 2       | 3                 | anana vs ana    | 3 | 3            |
| 2 | 5       | 4                 | nana vs na      | 2 | 2            |
| 3 | 1       | 5                 | ana vs a        | 1 | 1            |
| 4 | 4       | 0                 | na vs banana    | 0 | 0            |
| 5 | 0       |,                 | skip            |, |,            |

So
$$
LCP = [0, 1, 3, 0, 0, 2]
$$

#### Tiny Code (Python)

```python
def kasai(s, sa):
    n = len(s)
    rank = [0] * n
    for i in range(n):
        rank[sa[i]] = i
    
    h = 0
    lcp = [0] * n
    for i in range(n):
        if rank[i] > 0:
            j = sa[rank[i] - 1]
            while i + h < n and j + h < n and s[i + h] == s[j + h]:
                h += 1
            lcp[rank[i]] = h
            if h > 0:
                h -= 1
    return lcp

s = "banana"
sa = [5, 3, 1, 0, 4, 2]
print(kasai(s, sa))  # [0, 1, 3, 0, 0, 2]
```

#### Why It Matters

- Fundamental for string processing:

  * Longest repeated substring
  * Number of distinct substrings
  * Common substring queries
- Linear time, easy to integrate with suffix array
- Core component in bioinformatics, indexing, and data compression

#### Complexity

| Step             | Time   | Space  |
| ---------------- | ------ | ------ |
| Build rank array | $O(n)$ | $O(n)$ |
| LCP computation  | $O(n)$ | $O(n)$ |
| Total        | $O(n)$ | $O(n)$ |

#### Try It Yourself

1. Compute `LCP` for `"mississippi"`.
2. Draw suffix array and adjacent suffixes.
3. Find longest repeated substring using max LCP.
4. Count number of distinct substrings via
   $$
   \frac{n(n+1)}{2} - \sum_i LCP[i]
   $$
5. Compare with naive pairwise approach.

Kasai's algorithm is a masterpiece of reuse, slide, reuse, and reduce.
Every character compared once forward, once backward, and the entire structure of overlap unfolds in linear time.

### 624 Suffix Tree (Ukkonen's Algorithm)

The suffix tree is a powerful data structure that compactly represents all suffixes of a string. With Ukkonen's algorithm, we can build it in linear time —
$$
O(n)
$$, a landmark achievement in string processing.

A suffix tree unlocks a world of efficient solutions: substring search, longest repeated substring, pattern frequency, and much more, all in $O(m)$ query time for a pattern of length $m$.

#### What Problem Are We Solving?

Given a string
$$
S = s_0 s_1 \ldots s_{n-1}
$$
we want a tree where:

- Each path from root corresponds to a prefix of a suffix of $S$.
- Each leaf corresponds to a suffix.
- Edge labels are substrings of $S$ (not single chars).
- The tree is compressed: no redundant nodes with single child.

The structure should support:

- Substring search: $O(m)$
- Count distinct substrings: $O(n)$
- Longest repeated substring: via deepest internal node

#### Example

For  
$$
S = \texttt{"banana\textdollar"}
$$

All suffixes:

| i | Suffix  |
| - | -------- |
| 0 | banana$ |
| 1 | anana$  |
| 2 | nana$   |
| 3 | ana$    |
| 4 | na$     |
| 5 | a$      |
| 6 | $       |

The suffix tree compresses these suffixes into shared paths.

The root branches into:
- `$` → terminal leaf  
- `a` → covering suffixes starting at positions 1, 3, and 5 (`anana$`, `ana$`, `a$`)  
- `b` → covering suffix starting at position 0 (`banana$`)  
- `n` → covering suffixes starting at positions 2 and 4 (`nana$`, `na$`)

```
(root)
├── a → na → na$
├── b → anana$
├── n → a → na$
└── $
```

Every suffix appears exactly once as a path.

#### Why Naive Construction Is Slow

Naively inserting all $n$ suffixes into a trie takes
$$
O(n^2)
$$
since each suffix is $O(n)$ long.

Ukkonen's algorithm incrementally builds online, maintaining suffix links and implicit suffix trees, achieving
$$
O(n)
$$
time and space.

#### How It Works (Plain Language)

Ukkonen's algorithm builds the tree one character at a time.

We maintain:

- Active Point: current node + edge + offset
- Suffix Links: shortcuts between internal nodes
- Implicit Tree: built so far (without ending every suffix)

At step $i$ (after reading $S[0..i]$):

- Add new suffixes ending at $i$
- Reuse previous structure with suffix links
- Split edges only when necessary

After final character (often `$`), tree becomes explicit, containing all suffixes.

#### Step-by-Step Sketch

1. Initialize empty root
2. For each position $i$ in $S$:

   * Extend all suffixes ending at $i$
   * Use active point to track insertion position
   * Create internal nodes and suffix links as needed
3. Repeat until full tree is built

Suffix links skip redundant traversal, turning quadratic work into linear.

#### Example (Sketch)

Build for `abcab$`:

- Add `a`: path `a`
- Add `b`: paths `a`, `b`
- Add `c`: paths `a`, `b`, `c`
- Add `a`: paths reuse existing prefix
- Add `b`: creates internal node for `ab`
- Add `$`: closes all suffixes

Result: compressed tree with 6 leaves, one per suffix.

#### Tiny Code (Simplified, Python)

*(Simplified pseudo-implementation, for educational clarity)*

```python
class Node:
    def __init__(self):
        self.edges = {}
        self.link = None

def build_suffix_tree(s):
    s += "$"
    root = Node()
    n = len(s)
    for i in range(n):
        current = root
        for j in range(i, n):
            ch = s[j]
            if ch not in current.edges:
                current.edges[ch] = Node()
            current = current.edges[ch]
    return root

# Naive O(n^2) version for intuition
tree = build_suffix_tree("banana")
```

Ukkonen's true implementation uses edge spans `[l, r]` to avoid copying substrings, and active point management to ensure $O(n)$ time.

#### Why It Matters

- Fast substring search: $O(m)$
- Count distinct substrings: number of paths
- Longest repeated substring: deepest internal node
- Longest common substring (of two strings): via generalized suffix tree
- Basis for:

  * LCP array (via DFS)
  * Suffix automaton
  * FM-index

#### Complexity

| Step             | Time   | Space  |
| ---------------- | ------ | ------ |
| Build            | $O(n)$ | $O(n)$ |
| Query            | $O(m)$ | $O(1)$ |
| Count substrings | $O(n)$ | $O(1)$ |

#### Try It Yourself

1. Build suffix tree for `"abaaba$"`.
2. Mark leaf nodes with starting index.
3. Count number of distinct substrings.
4. Trace active point movement in Ukkonen's algorithm.
5. Compare with suffix array + LCP construction.

The suffix tree is the cathedral of string structures, every suffix a path, every branch a shared history, and Ukkonen's algorithm lays each stone in perfect linear time.

### 625 Suffix Automaton

The suffix automaton is the smallest deterministic finite automaton (DFA) that recognizes all substrings of a given string.
It captures every substring implicitly, in a compact, linear-size structure, perfect for substring queries, repetition analysis, and pattern counting.

Built in $O(n)$ time, it's often called the "Swiss Army knife" of string algorithms, flexible, elegant, and deeply powerful.

#### What Problem Are We Solving?

Given a string
$$
S = s_0 s_1 \ldots s_{n-1}
$$
we want an automaton that:

- Accepts exactly the set of all substrings of $S$
- Supports queries like:

  * "Is $T$ a substring of $S$?"
  * "How many distinct substrings does $S$ have?"
  * "Longest common substring with another string"
- Is minimal: no equivalent states, deterministic transitions

The suffix automaton (SAM) does exactly this —
constructed incrementally, state by state, edge by edge.

#### Key Idea

Each state represents a set of substrings that share the same end positions in $S$.
Each transition represents appending one character.
Suffix links connect states that represent "next smaller" substring sets.

So the SAM is essentially the DFA of all substrings, built online, in linear time.

#### How Does It Work (Plain Language)

We build the automaton as we read $S$, extending it character by character:

1. Start with initial state (empty string)
2. For each new character $c$:

   * Create a new state for substrings ending with $c$
   * Follow suffix links to extend old paths
   * Maintain minimality by cloning states when necessary
3. Update suffix links to ensure every substring's end position is represented

Each extension adds at most two states, so total states ≤ $2n - 1$.

#### Example

Let
$$
S = \texttt{"aba"}
$$

Step 1: Start with initial state (id 0)

Step 2: Add `'a'`

```
0 --a--> 1
```

link(1) = 0

Step 3: Add `'b'`

```
1 --b--> 2
0 --b--> 2
```

link(2) = 0

Step 4: Add `'a'`

- Extend from state 2 by `'a'`
- Need clone state since overlap

```
2 --a--> 3
link(3) = 1
```

Now automaton accepts: `"a"`, `"b"`, `"ab"`, `"ba"`, `"aba"`
All substrings represented!

#### Tiny Code (Python)

```python
class State:
    def __init__(self):
        self.next = {}
        self.link = -1
        self.len = 0

def build_suffix_automaton(s):
    sa = [State()]
    last = 0
    for ch in s:
        cur = len(sa)
        sa.append(State())
        sa[cur].len = sa[last].len + 1

        p = last
        while p >= 0 and ch not in sa[p].next:
            sa[p].next[ch] = cur
            p = sa[p].link

        if p == -1:
            sa[cur].link = 0
        else:
            q = sa[p].next[ch]
            if sa[p].len + 1 == sa[q].len:
                sa[cur].link = q
            else:
                clone = len(sa)
                sa.append(State())
                sa[clone].len = sa[p].len + 1
                sa[clone].next = sa[q].next.copy()
                sa[clone].link = sa[q].link
                while p >= 0 and sa[p].next[ch] == q:
                    sa[p].next[ch] = clone
                    p = sa[p].link
                sa[q].link = sa[cur].link = clone
        last = cur
    return sa
```

Usage:

```python
sam = build_suffix_automaton("aba")
print(len(sam))  # number of states (≤ 2n - 1)
```

#### Why It Matters

- Linear construction
  Build in $O(n)$
- Substring queries
  Check if $T$ in $S$ in $O(|T|)$
- Counting distinct substrings
  $$
  \sum_{v} (\text{len}[v] - \text{len}[\text{link}[v]])
  $$
- Longest common substring
  Run second string through automaton
- Frequency analysis
  Count occurrences via end positions

#### Complexity

| Step                | Time   | Space  |
| ------------------- | ------ | ------ |
| Build               | $O(n)$ | $O(n)$ |
| Substring query     | $O(m)$ | $O(1)$ |
| Distinct substrings | $O(n)$ | $O(n)$ |

#### Try It Yourself

1. Build SAM for `"banana"`
   Count distinct substrings.
2. Verify total = $n(n+1)/2 - \sum LCP[i]$
3. Add code to compute occurrences per substring.
4. Test substring search for `"nan"`, `"ana"`, `"nana"`.
5. Build SAM for reversed string and find palindromes.

The suffix automaton is minimal yet complete —
each state a horizon of possible endings,
each link a bridge to a shorter shadow.
A perfect mirror of all substrings, built step by step, in linear time.

### 626 SA-IS Algorithm (Linear-Time Suffix Array Construction)

The SA-IS algorithm is a modern, elegant method for building suffix arrays in true linear time
$$
O(n)
$$
It uses induced sorting, classifying suffixes by type (S or L), sorting a small subset first, and then *inducing* the rest from that ordering.

It's the foundation of state-of-the-art suffix array builders and is used in tools like DivSufSort, libdivsufsort, and BWT-based compressors.

#### What Problem Are We Solving?

We want to build the suffix array of a string
$$
S = s_0 s_1 \ldots s_{n-1}
$$
that lists all starting indices of suffixes in lexicographic order, but we want to do it in linear time, not
$$
O(n \log n)
$$

The SA-IS algorithm achieves this by:

1. Classifying suffixes into S-type and L-type
2. Identifying LMS (Leftmost S-type) substrings
3. Sorting these LMS substrings
4. Inducing the full suffix order from the LMS order

#### Key Concepts

Let $S[n] = $ $ be a sentinel smaller than all characters.

1. S-type suffix:
   $S[i:] < S[i+1:]$

2. L-type suffix:
   $S[i:] > S[i+1:]$

3. LMS position:
   An index $i$ such that $S[i]$ is S-type and $S[i-1]$ is L-type.

These LMS positions act as *anchors*, if we can sort them, we can "induce" the full suffix order.

#### How It Works (Plain Language)

Step 1. Classify S/L types
Walk backward:

- Last char `$` is S-type
- $S[i]$ is S-type if
  $S[i] < S[i+1]$ or ($S[i] = S[i+1]$ and $S[i+1]$ is S-type)

Step 2. Identify LMS positions
Mark every transition from L → S

Step 3. Sort LMS substrings
Each substring between LMS positions (inclusive) is unique.
We sort them (recursively, if needed) to get LMS order.

Step 4. Induce L-suffixes
From left to right, fill buckets using LMS order.

Step 5. Induce S-suffixes
From right to left, fill remaining buckets.

Result: full suffix array.

#### Example

Let
$$
S = \texttt{"banana\$"}
$$

1. Classify types

| i | Char | Type |
| - | ---- | ---- |
| 6 | $    | S    |
| 5 | a    | L    |
| 4 | n    | S    |
| 3 | a    | L    |
| 2 | n    | S    |
| 1 | a    | L    |
| 0 | b    | L    |

LMS positions: 2, 4, 6

2. LMS substrings:
   `"na$"`, `"nana$"`, `"$"`

3. Sort LMS substrings lexicographically.

4. Induce L and S suffixes using bucket boundaries.

Final
$$
SA = [6, 5, 3, 1, 0, 4, 2]
$$

#### Tiny Code (Python Sketch)

*(Illustrative, not optimized)*

```python
def sa_is(s):
    s = [ord(c) for c in s] + [0]  # append sentinel
    n = len(s)
    SA = [-1] * n

    # Step 1: classify
    t = [False] * n
    t[-1] = True
    for i in range(n-2, -1, -1):
        if s[i] < s[i+1] or (s[i] == s[i+1] and t[i+1]):
            t[i] = True

    # Identify LMS
    LMS = [i for i in range(1, n) if not t[i-1] and t[i]]

    # Simplified: use Python sort for LMS order (concept only)
    LMS.sort(key=lambda i: s[i:])

    # Induce-sort sketch omitted
    # (Full SA-IS would fill SA using bucket boundaries)

    return LMS  # placeholder for educational illustration

print(sa_is("banana"))  # [2, 4, 6]
```

A real implementation uses bucket arrays and induced sorting, still linear.

#### Why It Matters

- True linear time construction
- Memory-efficient, no recursion unless necessary
- Backbone for:

  * Burrows–Wheeler Transform (BWT)
  * FM-index
  * Compressed suffix arrays
- Practical and fast even on large datasets

#### Complexity

| Step                | Time   | Space  |
| ------------------- | ------ | ------ |
| Classify + LMS      | $O(n)$ | $O(n)$ |
| Sort LMS substrings | $O(n)$ | $O(n)$ |
| Induce sort         | $O(n)$ | $O(n)$ |
| Total           | $O(n)$ | $O(n)$ |

#### Try It Yourself

1. Classify types for `"mississippi$"`.
2. Mark LMS positions and substrings.
3. Sort LMS substrings by lexicographic order.
4. Perform induction step by step.
5. Compare output with doubling algorithm.

The SA-IS algorithm is a masterclass in economy —
sort a few, infer the rest, and let the structure unfold.
From sparse anchors, the full order of the text emerges, perfectly, in linear time.

### 627 LCP RMQ Query (Range Minimum Query on LCP Array)

The LCP RMQ structure allows constant-time retrieval of the Longest Common Prefix length between *any two suffixes* of a string, using the LCP array and a Range Minimum Query (RMQ) data structure.

In combination with the suffix array, it becomes a powerful text indexing tool, enabling substring comparisons, lexicographic ranking, and efficient pattern matching in
$$
O(1)
$$
after linear preprocessing.

#### What Problem Are We Solving?

Given:

- A string $S$
- Its suffix array $SA$
- Its LCP array, where
  $$
  LCP[i] = \text{lcp}(S[SA[i-1]:], S[SA[i]:])
  $$

We want to answer queries of the form:

> "What is the LCP length of suffixes starting at positions $i$ and $j$ in $S$?"

That is:
$$
\text{LCP}(i, j) = \text{length of longest common prefix of } S[i:] \text{ and } S[j:]
$$

This is fundamental for:

- Fast substring comparison
- Longest common substring (LCS) queries
- String periodicity detection
- Lexicographic interval analysis

#### Key Observation

Let $pos[i]$ and $pos[j]$ be the positions of suffixes $S[i:]$ and $S[j:]$ in the suffix array.

Then:
$$
\text{LCP}(i, j) = \min \big( LCP[k] \big), \quad k \in [\min(pos[i], pos[j]) + 1, \max(pos[i], pos[j])]
$$

So the problem reduces to a Range Minimum Query on the LCP array.

#### Example

Let
$$
S = \texttt{"banana"}
$$
$$
SA = [5, 3, 1, 0, 4, 2]
$$
$$
LCP = [0, 1, 3, 0, 0, 2]
$$

Goal: $\text{LCP}(1, 3)$, common prefix of `"anana"` and `"ana"`

1. $pos[1] = 2$, $pos[3] = 1$
2. Range = $(1, 2]$
3. $\min(LCP[2]) = 1$

So
$$
\text{LCP}(1, 3) = 1
$$
(both start with `"a"`)

#### How Does It Work (Plain Language)

1. Preprocess LCP using an RMQ structure (like Sparse Table or Segment Tree)
2. For query $(i, j)$:

   * Get $p = pos[i]$, $q = pos[j]$
   * Swap if $p > q$
   * Answer = RMQ on $LCP[p+1..q]$

Each query becomes $O(1)$ with $O(n \log n)$ or $O(n)$ preprocessing.

#### Tiny Code (Python – Sparse Table RMQ)

```python
import math

def build_rmq(lcp):
    n = len(lcp)
    log = [0] * (n + 1)
    for i in range(2, n + 1):
        log[i] = log[i // 2] + 1

    k = log[n]
    st = [[0] * (k + 1) for _ in range(n)]
    for i in range(n):
        st[i][0] = lcp[i]

    for j in range(1, k + 1):
        for i in range(n - (1 << j) + 1):
            st[i][j] = min(st[i][j-1], st[i + (1 << (j-1))][j-1])
    return st, log

def query_rmq(st, log, L, R):
    j = log[R - L + 1]
    return min(st[L][j], st[R - (1 << j) + 1][j])

# Example
LCP = [0, 1, 3, 0, 0, 2]
st, log = build_rmq(LCP)
print(query_rmq(st, log, 1, 2))  # 1
```

Query flow:

- $O(n \log n)$ preprocessing
- $O(1)$ per query

#### Why It Matters

- Enables fast substring comparison:

  * Compare suffixes in $O(1)$
  * Lexicographic rank / order check
- Core in:

  * LCP Interval Trees
  * Suffix Tree Emulation via array
  * LCS Queries across multiple strings
  * Distinct substring counting

With RMQ, a suffix array becomes a full-featured string index.

#### Complexity

| Operation   | Time          | Space         |
| ----------- | ------------- | ------------- |
| Preprocess  | $O(n \log n)$ | $O(n \log n)$ |
| Query (LCP) | $O(1)$        | $O(1)$        |

Advanced RMQs (like Cartesian Tree + Euler Tour + Sparse Table) achieve $O(n)$ space with $O(1)$ queries.

#### Try It Yourself

1. Build $SA$, $LCP$, and $pos$ for `"banana"`.
2. Answer queries:

   * $\text{LCP}(1, 2)$
   * $\text{LCP}(0, 4)$
   * $\text{LCP}(3, 5)$
3. Compare results with direct prefix comparison.
4. Replace Sparse Table with Segment Tree implementation.
5. Build $O(n)$ RMQ using Euler Tour + RMQ over Cartesian Tree.

The LCP RMQ bridges suffix arrays and suffix trees —
a quiet connection through minimums, where each interval tells the shared length of two paths in the lexicographic landscape.

### 628 Generalized Suffix Array (Multiple Strings)

The Generalized Suffix Array (GSA) extends the classical suffix array to handle multiple strings simultaneously. It provides a unified structure for cross-string comparisons, allowing us to compute longest common substrings, shared motifs, and cross-document search efficiently.

#### What Problem Are We Solving?

Given multiple strings:
$$
S_1, S_2, \ldots, S_k
$$
we want to index all suffixes of all strings in a single sorted array.

Each suffix in the array remembers:

- Which string it belongs to
- Its starting position

With this, we can:

- Find substrings shared between two or more strings
- Compute Longest Common Substring (LCS) across strings
- Perform multi-document search or text comparison

#### Example

Let:
$$
S_1 = \texttt{"banana\$1"}
$$
$$
S_2 = \texttt{"bandana\$2"}
$$

All suffixes:

| ID | Index | Suffix    | String |
| -- | ----- | --------- | ------ |
| 1  | 0     | banana$1  | S₁     |
| 1  | 1     | anana$1   | S₁     |
| 1  | 2     | nana$1    | S₁     |
| 1  | 3     | ana$1     | S₁     |
| 1  | 4     | na$1      | S₁     |
| 1  | 5     | a$1       | S₁     |
| 1  | 6     | $1        | S₁     |
| 2  | 0     | bandana$2 | S₂     |
| 2  | 1     | andana$2  | S₂     |
| 2  | 2     | ndana$2   | S₂     |
| 2  | 3     | dana$2    | S₂     |
| 2  | 4     | ana$2     | S₂     |
| 2  | 5     | na$2      | S₂     |
| 2  | 6     | a$2       | S₂     |
| 2  | 7     | $2        | S₂     |

Now sort all suffixes lexicographically.
Each entry in GSA records `(suffix_start, string_id)`.

#### Data Structures

We maintain:

1. SA, all suffixes across all strings, sorted lexicographically
2. LCP, longest common prefix between consecutive suffixes
3. Owner array, owner of each suffix (which string)

| i | SA[i] | Owner[i] | LCP[i] |
| - | ----- | -------- | ------ |
| 0 | ...   | 1        | 0      |
| 1 | ...   | 2        | 2      |
| 2 | ...   | 1        | 3      |
| … | …     | …        | …      |

From this, we can compute LCS by checking intervals where `Owner` differs.

#### How Does It Work (Plain Language)

1. Concatenate all strings with unique sentinels
   $$
   S = S_1 + \text{\$1} + S_2 + \text{\$2} + \cdots + S_k + \text{\$k}
   $$

2. Build suffix array over combined string (using SA-IS or doubling)

3. Record ownership: for each position, mark which string it belongs to

4. Build LCP array (Kasai)

5. Query shared substrings:

   * A common substring exists where consecutive suffixes belong to different strings
   * The minimum LCP over that range gives shared length

#### Example Query: Longest Common Substring

For `"banana"` and `"bandana"`:

Suffixes from different strings overlap with
$$
\text{LCP} = 3 \text{ ("ban") }
$$
and
$$
\text{LCP} = 2 \text{ ("na") }
$$

So
$$
\text{LCS} = \texttt{"ban"} \quad \text{length } 3
$$

#### Tiny Code (Python Sketch)

```python
def generalized_suffix_array(strings):
    text = ""
    owners = []
    sep = 1
    for s in strings:
        text += s + chr(sep)
        owners += [len(owners)] * (len(s) + 1)
        sep += 1

    sa = suffix_array_doubling(text)
    lcp = kasai(text, sa)

    owner_map = [owners[i] for i in sa]
    return sa, lcp, owner_map
```

*(Assumes you have `suffix_array_doubling` and `kasai` from earlier sections.)*

Usage:

```python
S1 = "banana"
S2 = "bandana"
sa, lcp, owner = generalized_suffix_array([S1, S2])
```

Now scan `lcp` where `owner[i] != owner[i-1]` to find cross-string overlaps.

#### Why It Matters

- Core structure for:

  * Longest Common Substring across files
  * Multi-document indexing
  * Bioinformatics motif finding
  * Plagiarism detection
- Compact alternative to Generalized Suffix Tree
- Easy to implement from existing SA + LCP pipeline

#### Complexity

| Step        | Time   | Space            |
| ----------- | ------ | ---------------- |
| Concatenate | $O(n)$ | $O(n)$           |
| SA build    | $O(n)$ | $O(n)$           |
| LCP build   | $O(n)$ | $O(n)$           |
| LCS query   | $O(n)$ | $O(1)$ per query |

Total:
$$
O(n)
$$
where $n$ = total length of all strings.

#### Try It Yourself

1. Build GSA for `["banana", "bandana"]`.
2. Find all substrings common to both.
3. Use `LCP` + `Owner` to extract longest shared substring.
4. Extend to 3 strings, e.g. `["banana", "bandana", "canada"]`.
5. Verify LCS correctness by brute-force comparison.

The Generalized Suffix Array is a chorus of strings —
each suffix a voice, each overlap a harmony.
From many songs, one lexicographic score —
and within it, every shared melody.

### 629 Enhanced Suffix Array (SA + LCP + RMQ)

The Enhanced Suffix Array (ESA) is a fully functional alternative to a suffix tree, built from a suffix array, LCP array, and range minimum query (RMQ) structure.
It supports the same powerful operations, substring search, LCP queries, longest repeated substring, and pattern matching, all with linear space and fast queries.

Think of it as a *compressed suffix tree*, implemented over arrays.

#### What Problem Are We Solving?

Suffix arrays alone can locate suffixes efficiently, but without structural information (like branching or overlap) that suffix trees provide.
The Enhanced Suffix Array enriches SA with auxiliary arrays to recover tree-like navigation:

1. SA, sorted suffixes
2. LCP, longest common prefix between adjacent suffixes
3. RMQ / Cartesian Tree, to simulate tree structure

With these, we can:

- Perform substring search
- Traverse suffix intervals
- Compute LCP of any two suffixes
- Enumerate distinct substrings, repeated substrings, and patterns

All without explicit tree nodes.

#### Key Idea

A suffix tree can be represented implicitly by the SA + LCP arrays:

- SA defines lexicographic order (in-order traversal)
- LCP defines edge lengths (branch depths)
- RMQ on LCP gives Lowest Common Ancestor (LCA) in tree view

So the ESA is a *view of the suffix tree*, not a reconstruction.

#### Example

Let
$$
S = \texttt{"banana\$"}
$$

Suffix array:
$$
SA = [6, 5, 3, 1, 0, 4, 2]
$$

LCP array:
$$
LCP = [0, 1, 3, 0, 0, 2, 0]
$$

These arrays already describe a tree-like structure:

- Branch depths = `LCP` values
- Subtrees = SA intervals sharing prefix length ≥ $k$

Example:

- Repeated substring `"ana"` corresponds to interval `[2, 4]` where `LCP ≥ 3`

#### How Does It Work (Plain Language)

The ESA answers queries via intervals and RMQs:

1. Substring Search (Pattern Matching)
   Binary search over `SA` for pattern prefix.
   Once found, `SA[l..r]` is the match interval.

2. LCP Query (Two Suffixes)
   Using RMQ:
   $$
   \text{LCP}(i, j) = \min(LCP[k]) \text{ over } k \in [\min(pos[i], pos[j])+1, \max(pos[i], pos[j])]
   $$

3. Longest Repeated Substring
   $\max(LCP)$ gives length, position via SA index.

4. Longest Common Prefix of Intervals
   RMQ over `LCP[l+1..r]` yields branch depth.

#### ESA Components

| Component | Description                   | Purpose                 |
| --------- | ----------------------------- | ----------------------- |
| SA    | Sorted suffix indices         | Lexicographic traversal |
| LCP   | LCP between adjacent suffixes | Branch depths           |
| INVSA | Inverse of SA                 | Fast suffix lookup      |
| RMQ   | Range min over LCP            | LCA / interval query    |

#### Tiny Code (Python)

```python
def build_esa(s):
    sa = suffix_array_doubling(s)
    lcp = kasai(s, sa)
    inv = [0] * len(s)
    for i, idx in enumerate(sa):
        inv[idx] = i
    st, log = build_rmq(lcp)
    return sa, lcp, inv, st, log

def lcp_query(inv, st, log, i, j):
    if i == j:
        return len(s) - i
    pi, pj = inv[i], inv[j]
    if pi > pj:
        pi, pj = pj, pi
    return query_rmq(st, log, pi + 1, pj)

# Example usage
s = "banana"
sa, lcp, inv, st, log = build_esa(s)
print(lcp_query(inv, st, log, 1, 3))  # "anana" vs "ana" → 3
```

*(Uses previous `suffix_array_doubling`, `kasai`, `build_rmq`, `query_rmq`.)*

#### Why It Matters

The ESA matches suffix tree functionality with:

- Linear space ($O(n)$)
- Simpler implementation
- Cache-friendly array access
- Easy integration with compressed indexes (FM-index)

Used in:

- Bioinformatics (sequence alignment)
- Search engines
- Document similarity
- Compression tools (BWT, LCP intervals)

#### Complexity

| Operation        | Time          | Space         |
| ---------------- | ------------- | ------------- |
| Build SA + LCP   | $O(n)$        | $O(n)$        |
| Build RMQ        | $O(n \log n)$ | $O(n \log n)$ |
| Query LCP        | $O(1)$        | $O(1)$        |
| Substring search | $O(m \log n)$ | $O(1)$        |

#### Try It Yourself

1. Build ESA for `"mississippi"`.
2. Find:

   * Longest repeated substring
   * Count distinct substrings
   * LCP of suffixes at 1 and 4
3. Extract substring intervals from `LCP ≥ 2`
4. Compare ESA output with suffix tree visualization

The Enhanced Suffix Array is the suffix tree reborn as arrays —
no nodes, no pointers, only order, overlap, and structure woven into the lexicographic tapestry.

### 630 Sparse Suffix Tree (Space-Efficient Variant)

The Sparse Suffix Tree (SST) is a space-efficient variant of the classical suffix tree.
Instead of storing *all* suffixes of a string, it indexes only a selected subset, typically every $k$-th suffix, reducing space from $O(n)$ nodes to $O(n / k)$ while preserving many of the same query capabilities.

This makes it ideal for large texts where memory is tight and approximate indexing is acceptable.

#### What Problem Are We Solving?

A full suffix tree gives incredible power, substring queries in $O(m)$ time, but at a steep memory cost, often 10–20× the size of the original text.

We want a data structure that:

- Supports fast substring search
- Is lightweight and cache-friendly
- Scales to large corpora (e.g., genomes, logs)

The Sparse Suffix Tree solves this by sampling suffixes, only building the tree on a subset.

#### Core Idea

Instead of inserting *every* suffix $S[i:]$,
we insert only those where
$$
i \bmod k = 0
$$

or from a sample set $P = {p_1, p_2, \ldots, p_t}$.

We then augment with verification steps (like binary search over text) to confirm full matches.

This reduces the structure size proportionally to the sample rate $k$.

#### How It Works (Plain Language)

1. Sampling step
   Choose sampling interval $k$ (e.g. every 4th suffix).

2. Build suffix tree
   Insert only sampled suffixes:
   $$
   { S[0:], S[k:], S[2k:], \ldots }
   $$

3. Search

   * To match a pattern $P$,
     find closest sampled suffix that shares prefix with $P$
   * Extend search in text for verification
     (O($k$) overhead at most)

This yields approximate O(m) query time with smaller constants.

#### Example

Let
$$
S = \texttt{"banana\$"}
$$
and choose $k = 2$.

Sampled suffixes:

- $S[0:] = $ `"banana$"`
- $S[2:] = $ `"nana$"`
- $S[4:] = $ `"na$"`
- $S[6:] = $ `"$"`

Build suffix tree only over these 4 suffixes.

When searching `"ana"`,

- Match found at node covering `"ana"` from `"banana$"` and `"nana$"`
- Verify remaining characters directly in $S$

#### Tiny Code (Python Sketch)

```python
class SparseSuffixTree:
    def __init__(self, s, k):
        self.s = s + "$"
        self.k = k
        self.suffixes = [self.s[i:] for i in range(0, len(s), k)]
        self.suffixes.sort()  # naive for illustration

    def search(self, pattern):
        # binary search over sampled suffixes
        lo, hi = 0, len(self.suffixes)
        while lo < hi:
            mid = (lo + hi) // 2
            if self.suffixes[mid].startswith(pattern):
                return True
            if self.suffixes[mid] < pattern:
                lo = mid + 1
            else:
                hi = mid
        return False

sst = SparseSuffixTree("banana", 2)
print(sst.search("ana"))  # True (verified from sampled suffix)
```

*For actual suffix tree structure, one can use Ukkonen's algorithm restricted to sampled suffixes.*

#### Why It Matters

- Space reduction: $O(n / k)$ nodes
- Scalable indexing for massive strings
- Fast enough for most substring queries
- Used in:

  * Genomic indexing
  * Log pattern search
  * Approximate data compression
  * Text analytics at scale

A practical balance between speed and size.

#### Complexity

| Operation       | Time                        | Space    |
| --------------- | --------------------------- | -------- |
| Build (sampled) | $O((n/k) \cdot k)$ = $O(n)$ | $O(n/k)$ |
| Search          | $O(m + k)$                  | $O(1)$   |
| Verify match    | $O(k)$                      | $O(1)$   |

Tuning $k$ trades memory for search precision.

#### Try It Yourself

1. Build SST for `"mississippi"` with $k = 3$
2. Compare memory vs full suffix tree
3. Search `"issi"`, measure verification steps
4. Experiment with different $k$ values
5. Plot build size vs query latency

The Sparse Suffix Tree is a memory-wise compromise —
a forest of sampled branches,
holding just enough of the text's structure
to navigate the space of substrings swiftly, without carrying every leaf.

# Section 64. Palindromes and Periodicity 

### 631 Naive Palindrome Check

The Naive Palindrome Check is the simplest way to detect palindromes, strings that read the same forward and backward.
It's a direct, easy-to-understand algorithm: expand around each possible center, compare characters symmetrically, and count or report all palindromic substrings.

This method is conceptually clear and perfect as a starting point before introducing optimized methods like Manacher's algorithm.

#### What Problem Are We Solving?

We want to find whether a string (or any substring) is a palindrome, i.e.

$$
S[l \ldots r] \text{ is a palindrome if } S[l + i] = S[r - i], \ \forall i
$$

We can use this to:

- Check if a given substring is palindromic
- Count the total number of palindromic substrings
- Find the longest palindrome (brute force)

#### Definition

A palindrome satisfies:
$$
S = \text{reverse}(S)
$$

Examples:

- `"aba"` → palindrome
- `"abba"` → palindrome
- `"abc"` → not palindrome

#### How Does It Work (Plain Language)

We can use center expansion or brute-force checking.

##### 1. Brute-Force Check

Compare characters from both ends:

1. Start at left and right
2. Move inward while matching
3. Stop on mismatch or middle

Time complexity: $O(n)$ for a single substring.

##### 2. Expand Around Center

Every palindrome has a center:

- Odd-length: single character
- Even-length: gap between two characters

We expand around each center and count palindromes.

There are $2n - 1$ centers total.

#### Example

String:
$$
S = \texttt{"abba"}
$$

Centers and expansions:

- Center at `'a'`: `"a"` Ok
- Center between `'a'` and `'b'`: no palindrome
- Center at `'b'`: `"b"`, `"bb"`, `"abba"` Ok
- Center at `'a'`: `"a"` Ok

Total palindromic substrings: `6`

#### Tiny Code (Python)

(a) Brute Force Check)

```python
def is_palindrome(s):
    return s == s[::-1]

print(is_palindrome("abba"))  # True
print(is_palindrome("abc"))   # False
```

(b) Expand Around Center (Count All)

```python
def count_palindromes(s):
    n = len(s)
    count = 0
    for center in range(2 * n - 1):
        l = center // 2
        r = l + (center % 2)
        while l >= 0 and r < n and s[l] == s[r]:
            count += 1
            l -= 1
            r += 1
    return count

print(count_palindromes("abba"))  # 6
```

(c) Expand Around Center (Longest Palindrome)

```python
def longest_palindrome(s):
    n = len(s)
    best = ""
    for center in range(2 * n - 1):
        l = center // 2
        r = l + (center % 2)
        while l >= 0 and r < n and s[l] == s[r]:
            if r - l + 1 > len(best):
                best = s[l:r+1]
            l -= 1
            r += 1
    return best

print(longest_palindrome("babad"))  # "bab" or "aba"
```

#### Why It Matters

- Foundation for more advanced algorithms (Manacher, DP)
- Works well for small or educational examples
- Simple way to verify correctness of optimized versions
- Useful for palindrome-related pattern discovery (DNA, text symmetry)

#### Complexity

| Operation               | Time     | Space  |
| ----------------------- | -------- | ------ |
| Check if palindrome     | $O(n)$   | $O(1)$ |
| Count all palindromes   | $O(n^2)$ | $O(1)$ |
| Find longest palindrome | $O(n^2)$ | $O(1)$ |

#### Try It Yourself

1. Count all palindromes in `"level"`.
2. Find longest palindrome in `"civicracecar"`.
3. Compare brute-force vs expand-around-center runtime for length 1000.
4. Modify to ignore non-alphanumeric characters.
5. Extend to find palindromic substrings within specific range $[l, r]$.

The Naive Palindrome Check is the mirror's first glance —
each center a reflection, each expansion a journey inward —
simple, direct, and a perfect foundation for the symmetries ahead.

### 632 Manacher's Algorithm

Manacher's Algorithm is the elegant, linear-time method to find the longest palindromic substring in a given string.
Unlike the naive $O(n^2)$ center-expansion, it leverages symmetry, every palindrome mirrors across its center, to reuse computations and skip redundant checks.

It's a classic example of how a clever insight turns a quadratic process into a linear one.

#### What Problem Are We Solving?

Given a string $S$ of length $n$, find the longest substring that reads the same forward and backward.

Example:

$$
S = \texttt{"babad"}
$$

Longest palindromic substrings:
$$
\texttt{"bab"} \text{ or } \texttt{"aba"}
$$

We want to compute this in $O(n)$ time, not $O(n^2)$.

#### The Core Idea

Manacher's key insight:
Each palindrome has a mirror about the current center.

If we know the palindrome radius at a position $i$,
we can deduce information about its mirror position $j$
(using previously computed values), without rechecking all characters.

#### Step-by-Step (Plain Language)

1. Preprocess the string to handle even-length palindromes  
   Insert `#` between all characters and at boundaries:
   $$
   S=\texttt{"abba"}\ \Rightarrow\ T=\texttt{"\#a\#b\#b\#a\#"}
   $$
   This way, all palindromes become odd-length in $T$.


2. Iterate through $T$
   Maintain:

   * $C$: center of the rightmost palindrome
   * $R$: right boundary
   * $P[i]$: palindrome radius at $i$

3. For each position $i$:

   * Mirror index: $i' = 2C - i$
   * Initialize $P[i] = \min(R - i, P[i'])$
   * Expand around $i$ while boundaries match

4. Update center and boundary if the palindrome expands beyond $R$.

5. After the loop, the maximum radius in $P$ gives the longest palindrome.

#### Example Walkthrough

String:
$$
S = \texttt{"abaaba"}
$$

Preprocessed:
$$
T = \texttt{"\#a\#b\#a\#a\#b\#a\#"}
$$

| i | T[i] | Mirror | P[i] | Center | Right |
| - | ---- | ------ | ---- | ------ | ----- |
| 0 | #    |,      | 0    | 0      | 0     |
| 1 | a    |,      | 1    | 1      | 2     |
| 2 | #    |,      | 0    | 1      | 2     |
| 3 | b    |        | 3    | 3      | 6     |
| … | …    | …      | …    | …      | …     |

Result:
$$
\text{Longest palindrome length } = 5
$$
$$
\text{Substring } = \texttt{"abaaba"}
$$

#### Tiny Code (Python)

```python
def manacher(s):
    # Preprocess
    t = "#" + "#".join(s) + "#"
    n = len(t)
    p = [0] * n
    c = r = 0  # center, right boundary

    for i in range(n):
        mirror = 2 * c - i
        if i < r:
            p[i] = min(r - i, p[mirror])
        # Expand around center i
        while i - 1 - p[i] >= 0 and i + 1 + p[i] < n and t[i - 1 - p[i]] == t[i + 1 + p[i]]:
            p[i] += 1
        # Update center if expanded past right boundary
        if i + p[i] > r:
            c, r = i, i + p[i]

    # Find max palindrome
    max_len = max(p)
    center_index = p.index(max_len)
    start = (center_index - max_len) // 2
    return s[start:start + max_len]

print(manacher("babad"))  # "bab" or "aba"
```

#### Why It Matters

- Linear time, the fastest known for longest palindrome problem
- Foundation for:

  * Palindromic substring enumeration
  * Palindromic tree construction
  * DNA symmetry search

It's a gem of algorithmic ingenuity, turning reflection into speed.

#### Complexity

| Operation                | Time   | Space  |
| ------------------------ | ------ | ------ |
| Build (with preprocess)  | $O(n)$ | $O(n)$ |
| Query longest palindrome | $O(1)$ | $O(1)$ |

#### Try It Yourself

1. Run Manacher on `"banana"` → `"anana"`
2. Compare with center-expansion time for $n = 10^5$
3. Modify to count all palindromic substrings
4. Track left and right boundaries visually
5. Apply to DNA sequence to detect symmetry motifs

#### A Gentle Proof (Why It Works)

Each position $i$ inside the current palindrome $(C, R)$
has a mirror $i' = 2C - i$.
If $i + P[i'] < R$, palindrome is fully inside → reuse $P[i']$.
Else, expand beyond $R$ and update center.

No position is expanded twice → total $O(n)$.

Manacher's Algorithm is symmetry embodied —
every center a mirror, every reflection a shortcut.
What once took quadratic effort now glides in linear grace.

### 633 Longest Palindromic Substring (Center Expansion)

The Longest Palindromic Substring problem asks:

> *What is the longest contiguous substring of a given string that reads the same forward and backward?*

The center expansion method is the intuitive and elegant $O(n^2)$ solution, simple to code, easy to reason about, and surprisingly efficient in practice.
It sits between the naive brute force ($O(n^3)$) and Manacher's algorithm ($O(n)$).

#### What Problem Are We Solving?

Given a string $S$ of length $n$, find the substring $S[l \ldots r]$ such that:

$$
S[l \ldots r] = \text{reverse}(S[l \ldots r])
$$

and $(r - l + 1)$ is maximal.

Examples:

- `"babad"` → `"bab"` or `"aba"`
- `"cbbd"` → `"bb"`
- `"a"` → `"a"`

We want to find this substring efficiently and clearly.

#### Core Idea

Every palindrome is defined by its center:

- Odd-length palindromes: one center (e.g. `"aba"`)
- Even-length palindromes: two-character center (e.g. `"abba"`)

If we expand from every possible center,
we can detect all palindromic substrings and track the longest one.

#### How It Works (Plain Language)

1. For each index $i$ in $S$:

   * Expand around $i$ (odd palindrome)
   * Expand around $(i, i+1)$ (even palindrome)
2. Stop expanding when characters don't match.
3. Track the maximum length and start index.

Each expansion costs $O(n)$, across $n$ centers → $O(n^2)$ total.

#### Example

String:
$$
S = \texttt{"babad"}
$$

Centers and expansions:

- Center at `b`: `"b"`, expand `"bab"`
- Center at `a`: `"a"`, expand `"aba"`
- Center at `ba`: mismatch, no expansion

Longest found: `"bab"` or `"aba"`

#### Tiny Code (Python)

```python
def longest_palindrome_expand(s):
    if not s:
        return ""
    start = end = 0

    def expand(l, r):
        while l >= 0 and r < len(s) and s[l] == s[r]:
            l -= 1
            r += 1
        return l + 1, r - 1  # bounds of palindrome

    for i in range(len(s)):
        l1, r1 = expand(i, i)       # odd length
        l2, r2 = expand(i, i + 1)   # even length
        if r1 - l1 > end - start:
            start, end = l1, r1
        if r2 - l2 > end - start:
            start, end = l2, r2

    return s[start:end+1]

print(longest_palindrome_expand("babad"))  # "bab" or "aba"
print(longest_palindrome_expand("cbbd"))   # "bb"
```

#### Why It Matters

- Straightforward and robust
- Useful for:

  * Substring symmetry checks
  * Bioinformatics (palindromic DNA segments)
  * Natural language analysis
- Easier to implement than Manacher's, yet performant for most $n \le 10^4$

#### Complexity

| Operation               | Time     | Space  |
| ----------------------- | -------- | ------ |
| Expand from all centers | $O(n^2)$ | $O(1)$ |
| Find longest palindrome | $O(1)$   | $O(1)$ |

#### Try It Yourself

1. Find the longest palindrome in `"racecarxyz"`.
2. Modify to count all palindromic substrings.
3. Return start and end indices of longest palindrome.
4. Test on `"aaaabaaa"` → `"aaabaaa"`.
5. Compare with Manacher's Algorithm output.

#### A Gentle Proof (Why It Works)

Each palindrome is uniquely centered at either:

- a single character (odd case), or
- between two characters (even case)

Since we try all $2n - 1$ centers,
every palindrome is discovered once,
and we take the longest among them.

Thus correctness and completeness follow directly.

The center expansion method is a mirror dance —
each position a pivot, each match a reflection —
building symmetry outward, one step at a time.

### 634 Palindrome DP Table (Dynamic Programming Approach)

The Palindrome DP Table method uses dynamic programming to find and count palindromic substrings.
It's a bottom-up strategy that builds a 2D table marking whether each substring $S[i \ldots j]$ is a palindrome, and from there, we can easily answer questions like:

- Is substring $S[i \ldots j]$ palindromic?
- What is the longest palindromic substring?
- How many palindromic substrings exist?

It's systematic and easy to extend, though it costs more memory than center expansion.

#### What Problem Are We Solving?

Given a string $S$ of length $n$, we want to precompute palindromic substrings efficiently.

We define a DP table $P[i][j]$ such that:

$$
P[i][j] =
\begin{cases}
\text{True}, & \text{if } S[i \ldots j] \text{ is a palindrome},\\
\text{False}, & \text{otherwise.}
\end{cases}
$$


Then we can query or count all palindromes using this table.

#### Recurrence Relation

A substring $S[i \ldots j]$ is a palindrome if:

1. The boundary characters match:
   $$
   S[i] = S[j]
   $$
2. The inner substring is also a palindrome (or small enough):
   $$
   P[i+1][j-1] = \text{True} \quad \text{or} \quad (j - i \le 2)
   $$

So the recurrence is:

$$
P[i][j] = (S[i] = S[j]) \ \text{and} \ (j - i \le 2 \ \text{or} \ P[i+1][j-1])
$$

#### Initialization

- All single characters are palindromes:
  $$
  P[i][i] = \text{True}
  $$
- Two-character substrings are palindromic if both match:
  $$
  P[i][i+1] = (S[i] = S[i+1])
  $$

We fill the table from shorter substrings to longer ones.

#### Example

Let
$$
S = \texttt{"abba"}
$$

We build $P$ bottom-up:

| i\j | 0:a | 1:b | 2:b | 3:a |
| --- | --- | --- | --- | --- |
| 0:a | T   | F   | F   | T   |
| 1:b |     | T   | T   | F   |
| 2:b |     |     | T   | F   |
| 3:a |     |     |     | T   |

Palindromic substrings: `"a"`, `"b"`, `"bb"`, `"abba"`

#### Tiny Code (Python)

```python
def longest_palindrome_dp(s):
    n = len(s)
    if n == 0:
        return ""
    dp = [[False] * n for _ in range(n)]
    start, max_len = 0, 1

    # length 1
    for i in range(n):
        dp[i][i] = True

    # length 2
    for i in range(n-1):
        if s[i] == s[i+1]:
            dp[i][i+1] = True
            start, max_len = i, 2

    # length >= 3
    for length in range(3, n+1):
        for i in range(n - length + 1):
            j = i + length - 1
            if s[i] == s[j] and dp[i+1][j-1]:
                dp[i][j] = True
                start, max_len = i, length

    return s[start:start+max_len]

print(longest_palindrome_dp("babad"))  # "bab" or "aba"
```

#### Why It Matters

- Clear logic, easy to adapt
- Useful for:

  * Counting all palindromic substrings
  * Finding all palindromic indices
  * Teaching DP recurrence building

It's the pedagogical bridge from brute force to linear-time solutions.

#### Complexity

| Operation                        | Time     | Space    |
| -------------------------------- | -------- | -------- |
| Build DP table                   | $O(n^2)$ | $O(n^2)$ |
| Query palindrome $S[i \ldots j]$ | $O(1)$   | $O(1)$   |
| Longest palindrome extraction    | $O(n^2)$ | $O(n^2)$ |

#### Try It Yourself

1. Count all palindromic substrings by summing `dp[i][j]`.
2. Return all indices $(i, j)$ where `dp[i][j] == True`.
3. Compare runtime vs center-expansion for $n = 2000$.
4. Optimize space by using 1D rolling arrays.
5. Adapt for "near-palindromes" (allow 1 mismatch).

#### A Gentle Proof (Why It Works)

We expand palindrome definitions incrementally:

- Base case: length 1 or 2
- Recursive case: match outer chars + inner palindrome

Every palindrome has a smaller palindrome inside,
so the bottom-up order ensures correctness.

The Palindrome DP Table turns reflection into recurrence —
each cell a mirror, each step a layer —
revealing every symmetry hidden in the string.

### 635 Palindromic Tree (Eertree)

A palindromic tree (often called Eertree) is a dynamic structure that stores all distinct palindromic substrings of a string while you scan it from left to right.
It maintains one node per palindrome and supports insertion of the next character in amortized constant time, yielding linear total time.

It is the most direct way to enumerate palindromes: you get their counts, lengths, end positions, and suffix links for free.

#### What Problem Are We Solving?

Given a string $S$, we want to maintain after each prefix $S[0..i]$:

- All distinct palindromic substrings present so far
- For each palindrome, its length, suffix link to the longest proper palindromic suffix, and optionally its occurrence count

With an Eertree we can build this online in $O(n)$ time and $O(n)$ space, since a string of length $n$ has at most $n$ distinct palindromic substrings.

#### Core Idea

Nodes correspond to distinct palindromes. There are two special roots:

- Node $-1$ representing a virtual palindrome of length $-1$
- Node $0$ representing the empty palindrome of length $0$

Each node keeps:

- `len[v]`: palindrome length
- `link[v]`: suffix link to the longest proper palindromic suffix
- `next[v][c]`: transition by adding character $c$ to both ends
- optionally `occ[v]`: number of occurrences ending at processed positions
- `first_end[v]` or a last end index to recover positions

To insert a new character $S[i]$:

1. Walk suffix links from the current longest suffix-palindrome until you find a node $v$ such that $S[i - len[v] - 1] = S[i]$. This is the largest palindromic suffix of $S[0..i]$ that can be extended by $S[i]$.
2. If edge by $S[i]$ does not exist from $v$, create a new node for the new palindrome. Set its `len`, compute its `link` by continuing suffix-link jumps from `link[v]`, and set transitions.
3. Update occurrence counters and set the new current node to this node.

Each insertion creates at most one new node, so total nodes are at most $n + 2$.

#### Example

Let $S = \texttt{"ababa"}$.

Processed prefixes and newly created palindromes:

- $i=0$: add `a` → `"a"`
- $i=1$: add `b` → `"b"`
- $i=2$: add `a` → `"aba"`
- $i=3$: add `b` → `"bab"`
- $i=4$: add `a` → `"ababa"`

Distinct palindromes: `a`, `b`, `aba`, `bab`, `ababa`.
Two special roots always exist: length $-1$ and $0$.

#### Tiny Code (Python, educational)

```python
class Eertree:
    def __init__(self):
        # node 0: empty palindrome, len = 0
        # node 1: imaginary root, len = -1
        self.len = [0, -1]
        self.link = [1, 1]
        self.next = [dict(), dict()]
        self.occ = [0, 0]
        self.s = []
        self.last = 0  # node of the longest suffix palindrome of current string
        self.n = 0

    def _get_suflink(self, v, i):
        while True:
            l = self.len[v]
            if i - l - 1 >= 0 and self.s[i - l - 1] == self.s[i]:
                return v
            v = self.link[v]

    def add_char(self, ch):
        self.s.append(ch)
        i = self.n
        self.n += 1

        v = self._get_suflink(self.last, i)
        if ch not in self.next[v]:
            self.next[v][ch] = len(self.len)
            self.len.append(self.len[v] + 2)
            self.next.append(dict())
            self.occ.append(0)
            # compute suffix link of the new node
            if self.len[-1] == 1:
                self.link.append(0)  # single char palindromes link to empty
            else:
                u = self._get_suflink(self.link[v], i)
                self.link.append(self.next[u][ch])
        w = self.next[v][ch]
        self.last = w
        self.occ[w] += 1
        return w  # returns node index for the longest suffix palindrome

    def finalize_counts(self):
        # propagate occurrences along suffix links so occ[v] counts all ends
        order = sorted(range(2, len(self.len)), key=lambda x: self.len[x], reverse=True)
        for v in order:
            self.occ[self.link[v]] += self.occ[v]
```

Usage:

```python
T = Eertree()
for c in "ababa":
    T.add_char(c)
T.finalize_counts()
# Distinct palindromes count (excluding two roots):
print(len(T.len) - 2)  # 5
```

What you get:

- Number of distinct palindromes: `len(nodes) - 2`
- Occurrences of each palindrome after `finalize_counts`
- Lengths, suffix links, and transitions for traversal

#### Why It Matters

- Lists all distinct palindromic substrings in linear time
- Supports online updates: add one character and update in amortized $O(1)$
- Gives counts and boundaries for palindromes
- Natural fit for:

  * Counting palindromic substrings
  * Longest palindromic substring while streaming
  * Palindromic factorization and periodicity analysis
  * Biosequence symmetry mining

#### Complexity

| Operation              | Time             | Space        |
| ---------------------- | ---------------- | ------------ |
| Insert one character   | amortized $O(1)$ | $O(1)$ extra |
| Build on length $n$    | $O(n)$           | $O(n)$ nodes |
| Occurrence aggregation | $O(n)$           | $O(n)$       |

At most one new palindrome per position, hence linear bounds.

#### Try It Yourself

1. Build the eertree for `"aaaabaaa"`. Verify the distinct palindromes and their counts.
2. Track the longest palindrome after each insertion.
3. Record first and last end positions per node to list all occurrences.
4. Modify the structure to also maintain even and odd longest suffix palindromes separately.
5. Compare memory and speed with DP and Manacher for $n \approx 10^5$.

The palindromic tree models the universe of palindromes succinctly: every node is a mirror, every link a shorter reflection, and with one sweep over the string you discover them all.

### 636 Prefix Function Periodicity

The prefix function is a core tool in string algorithms, it tells you, for each position, the length of the longest proper prefix that is also a suffix.
When studied through the lens of periodicity, it becomes a sharp instrument for detecting repetition patterns, string borders, and minimal periods, foundational in pattern matching, compression, and combinatorics on words.

#### What Problem Are We Solving?

We want to find periodic structure in a string, specifically, the shortest repeating unit.

A string $S$ of length $n$ is periodic if there exists a $p < n$ such that:

$$
S[i] = S[i + p] \quad \forall i = 0, 1, \ldots, n - p - 1
$$

We call $p$ the period of $S$.

The prefix function gives us exactly the border lengths we need to compute $p$ in $O(n)$.

#### The Prefix Function

For string $S[0 \ldots n-1]$, define:

$$
\pi[i] = \text{length of the longest proper prefix of } S[0..i] \text{ that is also a suffix}
$$

This is the same array used in Knuth–Morris–Pratt (KMP).

#### Periodicity Formula

The minimal period of the prefix $S[0..i]$ is:

$$
p = (i + 1) - \pi[i]
$$

If $(i + 1) \bmod p = 0$,
then the prefix $S[0..i]$ is fully periodic with period $p$.

#### Example

Let
$$
S = \texttt{"abcabcabc"}
$$

Compute prefix function:

| i | S[i] | π[i] |
| - | ---- | ---- |
| 0 | a    | 0    |
| 1 | b    | 0    |
| 2 | c    | 0    |
| 3 | a    | 1    |
| 4 | b    | 2    |
| 5 | c    | 3    |
| 6 | a    | 4    |
| 7 | b    | 5    |
| 8 | c    | 6    |

Now minimal period for $i=8$:

$$
p = 9 - \pi[8] = 9 - 6 = 3
$$

And since $9 \bmod 3 = 0$,
period = 3, repeating unit `"abc"`.

#### How It Works (Plain Language)

Each $\pi[i]$ measures how much of the string "wraps around" itself.
If a prefix and suffix align, they hint at repetition.
The difference between length $(i + 1)$ and border $\pi[i]$ gives the repeating block length.

When the total length divides evenly by this block,
the entire prefix is made of repeated copies.

#### Tiny Code (Python)

```python
def prefix_function(s):
    n = len(s)
    pi = [0] * n
    for i in range(1, n):
        j = pi[i - 1]
        while j > 0 and s[i] != s[j]:
            j = pi[j - 1]
        if s[i] == s[j]:
            j += 1
        pi[i] = j
    return pi

def minimal_period(s):
    pi = prefix_function(s)
    n = len(s)
    p = n - pi[-1]
    if n % p == 0:
        return p
    return n  # no smaller period

s = "abcabcabc"
print(minimal_period(s))  # 3
```

#### Why It Matters

- Detects repetition in strings in linear time
- Used in:

  * Pattern compression
  * DNA repeat detection
  * Music rhythm analysis
  * Periodic task scheduling
- Core concept behind KMP, Z-algorithm, and border arrays

#### Complexity

| Operation             | Time   | Space  |
| --------------------- | ------ | ------ |
| Build prefix function | $O(n)$ | $O(n)$ |
| Find minimal period   | $O(1)$ | $O(1)$ |
| Check periodic prefix | $O(1)$ | $O(1)$ |

#### Try It Yourself

1. Compute period of `"ababab"` → `2`.
2. Compute period of `"aaaaa"` → `1`.
3. Compute period of `"abcd"` → `4` (no repetition).
4. For each prefix, print `(i + 1) - π[i]` and test divisibility.
5. Compare with Z-function periodicity (section 637).

#### A Gentle Proof (Why It Works)

If $\pi[i] = k$, then
$S[0..k-1] = S[i-k+1..i]$.

Hence, the prefix has a border of length $k$,
and repeating block size $(i + 1) - k$.
If $(i + 1)$ divides evenly by that,
the entire prefix is copies of one unit.

Prefix Function Periodicity reveals rhythm in repetition —
each border a rhyme, each overlap a hidden beat —
turning pattern detection into simple modular music.

### 637 Z-Function Periodicity

The Z-Function offers another elegant path to uncovering repetition and periodicity in strings.
While the prefix function looks backward (prefix–suffix overlaps), the Z-function looks forward, it measures how far each position matches the beginning of the string.
This makes it a natural fit for analyzing repeating prefixes and finding periods in linear time.

#### What Problem Are We Solving?

We want to detect if a string $S$ has a period $p$ —
that is, if it consists of one or more repetitions of a smaller block.

Formally, $S$ has period $p$ if:

$$
S[i] = S[i + p], \quad \forall i \in [0, n - p - 1]
$$

Equivalently, if:

$$
S = T^k \quad \text{for some } T, k \ge 2
$$

The Z-function reveals this structure by measuring prefix matches at every shift.

#### Definition

For string $S$ of length $n$:

$$
Z[i] = \text{length of the longest prefix of } S \text{ starting at } i
$$

Formally:

$$
Z[i] = \max { k \ | \ S[0..k-1] = S[i..i+k-1] }
$$

By definition, $Z[0] = 0$ or $n$ (often set to 0 for simplicity).

#### Periodicity Criterion

A string $S$ of length $n$ has period $p$ if:

$$
Z[p] = n - p
$$

and $p$ divides $n$, i.e. $n \bmod p = 0$.

This means the prefix of length $n - p$ repeats perfectly from position $p$.

More generally, any $p$ satisfying $Z[p] = n - p$ is a border length, and minimal period = smallest such $p$.

#### Example

Let
$$
S = \texttt{"abcabcabc"}
$$
$n = 9$

Compute $Z$ array:

| i | S[i:]     | Z[i] |
| - | --------- | ---- |
| 0 | abcabcabc | 0    |
| 1 | bcabcabc  | 0    |
| 2 | cabcabc   | 0    |
| 3 | abcabc    | 6    |
| 4 | bcabc     | 0    |
| 5 | cabc      | 0    |
| 6 | abc       | 3    |
| 7 | bc        | 0    |
| 8 | c         | 0    |

Check $p = 3$:

$$
Z[3] = 6 = n - 3, \quad 9 \bmod 3 = 0
$$

Ok So $p = 3$ is the minimal period.

#### How It Works (Plain Language)

Imagine sliding the string against itself:

- At shift $p$, $Z[p]$ tells how many leading characters still match.
- If the overlap spans the rest of the string ($Z[p] = n - p$),
  then the pattern before and after aligns perfectly.

This alignment implies repetition.

#### Tiny Code (Python)

```python
def z_function(s):
    n = len(s)
    Z = [0] * n
    l = r = 0
    for i in range(1, n):
        if i <= r:
            Z[i] = min(r - i + 1, Z[i - l])
        while i + Z[i] < n and s[Z[i]] == s[i + Z[i]]:
            Z[i] += 1
        if i + Z[i] - 1 > r:
            l, r = i, i + Z[i] - 1
    return Z

def minimal_period_z(s):
    n = len(s)
    Z = z_function(s)
    for p in range(1, n):
        if Z[p] == n - p and n % p == 0:
            return p
    return n

s = "abcabcabc"
print(minimal_period_z(s))  # 3
```

#### Why It Matters

- A simple way to test repetition and pattern structure
- Linear-time ($O(n)$) algorithm
- Useful in:

  * String periodicity detection
  * Prefix-based hashing
  * Pattern discovery
  * Suffix comparisons

The Z-function complements the prefix function:

- Prefix function → borders (prefix = suffix)
- Z-function → prefix matches at every offset

#### Complexity

| Operation           | Time   | Space  |
| ------------------- | ------ | ------ |
| Compute Z-array     | $O(n)$ | $O(n)$ |
| Check periodicity   | $O(n)$ | $O(1)$ |
| Find minimal period | $O(n)$ | $O(1)$ |

#### Try It Yourself

1. Compute $Z$ for `"aaaaaa"` → minimal period = 1
2. For `"ababab"` → $Z[2] = 4$, period = 2
3. For `"abcd"` → no valid $p$, period = 4
4. Verify $Z[p] = n - p$ corresponds to repeating prefix
5. Compare results with prefix function periodicity

#### A Gentle Proof (Why It Works)

If $Z[p] = n - p$,
then $S[0..n-p-1] = S[p..n-1]$.
Thus $S$ can be partitioned into blocks of size $p$.
If $n \bmod p = 0$,
then $S = T^{n/p}$ with $T = S[0..p-1]$.

Therefore, the smallest $p$ with that property is the minimal period.

The Z-Function turns overlap into insight —
each shift a mirror, each match a rhyme —
revealing the string's hidden beat through forward reflection.

### 638 KMP Prefix Period Check (Shortest Repeating Unit)

The KMP prefix function not only powers fast pattern matching but also quietly encodes the repetitive structure of a string.
By analyzing the final value of the prefix function, we can reveal whether a string is built from repeated copies of a smaller block, and if so, find that shortest repeating unit, the *fundamental period*.

#### What Problem Are We Solving?

Given a string $S$ of length $n$,
we want to determine:

1. Is $S$ composed of repeated copies of a smaller substring?
2. If yes, what is the shortest repeating unit $T$ and its length $p$?

Formally,
$$
S = T^k, \quad \text{where } |T| = p, \ k = n / p
$$
and $n \bmod p = 0$.

#### Core Insight

The prefix function $\pi[i]$ captures the longest border of each prefix —
a border is a substring that is both a proper prefix and proper suffix.

At the end ($i = n - 1$), $\pi[n-1]$ gives the length of the longest border of the full string.

Let:
$$
b = \pi[n-1]
$$
Then the candidate period is:
$$
p = n - b
$$

If $n \bmod p = 0$,
the string is periodic with shortest repeating unit of length $p$.

Otherwise, it's aperiodic, and $p = n$.

#### Example 1

$$
S = \texttt{"abcabcabc"}
$$

$n = 9$

Compute prefix function:

| i | S[i] | π[i] |
| - | ---- | ---- |
| 0 | a    | 0    |
| 1 | b    | 0    |
| 2 | c    | 0    |
| 3 | a    | 1    |
| 4 | b    | 2    |
| 5 | c    | 3    |
| 6 | a    | 4    |
| 7 | b    | 5    |
| 8 | c    | 6    |

So $\pi[8] = 6$

$$
p = 9 - 6 = 3
$$

Check: $9 \bmod 3 = 0$ Ok
Hence, shortest repeating unit = `"abc"`.

#### Example 2

$$
S = \texttt{"aaaa"} \implies n=4
$$
$\pi = [0, 1, 2, 3]$, so $\pi[3] = 3$
$$
p = 4 - 3 = 1, \quad 4 \bmod 1 = 0
$$
Ok Repeating unit = `"a"`

#### Example 3

$$
S = \texttt{"abcd"} \implies n=4
$$
$\pi = [0, 0, 0, 0]$
$$
p = 4 - 0 = 4, \quad 4 \bmod 4 = 0
$$
Only repeats once → no smaller period.

#### How It Works (Plain Language)

The prefix function shows how much of the string overlaps with itself.
If the border length is $b$, then the last $b$ characters match the first $b$.
That means every $p = n - b$ characters, the pattern repeats.
If the string length divides evenly by $p$, it's made up of repeated blocks.

#### Tiny Code (Python)

```python
def prefix_function(s):
    n = len(s)
    pi = [0] * n
    for i in range(1, n):
        j = pi[i - 1]
        while j > 0 and s[i] != s[j]:
            j = pi[j - 1]
        if s[i] == s[j]:
            j += 1
        pi[i] = j
    return pi

def shortest_repeating_unit(s):
    pi = prefix_function(s)
    n = len(s)
    b = pi[-1]
    p = n - b
    if n % p == 0:
        return s[:p]
    return s  # no repetition

print(shortest_repeating_unit("abcabcabc"))  # "abc"
print(shortest_repeating_unit("aaaa"))       # "a"
print(shortest_repeating_unit("abcd"))       # "abcd"
```

#### Why It Matters

- Finds string periodicity in $O(n)$ time
- Crucial for:

  * Pattern detection and compression
  * Border analysis and combinatorics
  * Minimal automata construction
  * Rhythm detection in music or DNA repeats

Elegant and efficient, all from a single $\pi$ array.

#### Complexity

| Operation               | Time   | Space  |
| ----------------------- | ------ | ------ |
| Compute prefix function | $O(n)$ | $O(n)$ |
| Extract period          | $O(1)$ | $O(1)$ |

#### Try It Yourself

1. `"abababab"` → π = [0,0,1,2,3,4,5,6], $b=6$, $p=2$, unit = `"ab"`
2. `"xyzxyzx"` → $n=7$, $\pi[6]=3$, $p=4$, $7 \bmod 4 \neq 0$ → aperiodic
3. `"aaaaa"` → $p=1$, unit = `"a"`
4. `"abaaba"` → $n=6$, $\pi[5]=3$, $p=3$, $6 \bmod 3 = 0$ → `"aba"`
5. Try combining with prefix-function periodicity table (Sec. 636).

#### A Gentle Proof (Why It Works)

If $\pi[n-1] = b$,
then prefix of length $b$ = suffix of length $b$.
Hence, block length $p = n - b$.
If $p$ divides $n$,
then $S$ is made of $n / p$ copies of $S[0..p-1]$.

Otherwise, it only has partial repetition at the end.

KMP Prefix Period Check is the heartbeat of repetition —
each border a callback, each overlap a rhythm —
revealing the smallest phrase that composes the song.

### 639 Lyndon Factorization (Chen–Fox–Lyndon Decomposition)

The Lyndon Factorization, also known as the Chen–Fox–Lyndon decomposition, is a remarkable string theorem that breaks any string into a unique sequence of Lyndon words, substrings that are strictly smaller (lexicographically) than any of their nontrivial suffixes.

This factorization is deeply connected to lexicographic order, suffix arrays, suffix automata, and string combinatorics, and is the foundation of algorithms like the Duval algorithm.

#### What Problem Are We Solving?

We want to decompose a string $S$ into a sequence of factors:

$$
S = w_1 w_2 w_3 \dots w_k
$$

such that:

1. Each $w_i$ is a Lyndon word
   (i.e. strictly smaller than any of its proper suffixes)
2. The sequence is nonincreasing in lexicographic order:
   $$
   w_1 \ge w_2 \ge w_3 \ge \dots \ge w_k
   $$

This decomposition is unique for every string.

#### What Is a Lyndon Word?

A Lyndon word is a nonempty string that is strictly lexicographically smaller than all its rotations.

Formally, $w$ is Lyndon if:
$$
\forall u, v \text{ such that } w = uv, v \ne \varepsilon: \quad w < v u
$$

Examples:

- `"a"`, `"ab"`, `"aab"`, `"abc"` are Lyndon
- `"aa"`, `"aba"`, `"ba"` are not

#### Example

Let:
$$
S = \texttt{"banana"}
$$

Factorization:

| Step | Remaining  | Factor          | Explanation            |
| ---- | ---------- | --------------- | ---------------------- |
| 1    | banana     | b               | `"b"` < `"anana"`      |
| 2    | anana      | a               | `"a"` < `"nana"`       |
| 3    | nana       | n               | `"n"` < `"ana"`        |
| 4    | ana        | a               | `"a"` < `"na"`         |
| 5    | na         | n               | `"n"` < `"a"`          |
| 6    | a          | a               | end                    |
|      | Result | b a n a n a | nonincreasing sequence |

Each factor is a Lyndon word.

#### How It Works (Plain Language)

The Duval algorithm constructs this factorization efficiently:

1. Start from the beginning of $S$
   Let `i = 0`
2. Find the smallest Lyndon word prefix starting at `i`
3. Output that word as a factor
   Move `i` to the next position after the factor
4. Repeat until end of string

This runs in linear time, $O(n)$.

#### Tiny Code (Python – Duval Algorithm)

```python
def lyndon_factorization(s):
    n = len(s)
    i = 0
    result = []
    while i < n:
        j = i + 1
        k = i
        while j < n and s[k] <= s[j]:
            if s[k] < s[j]:
                k = i
            else:
                k += 1
            j += 1
        while i <= k:
            result.append(s[i:i + j - k])
            i += j - k
    return result

print(lyndon_factorization("banana"))  # ['b', 'a', 'n', 'a', 'n', 'a']
print(lyndon_factorization("aababc"))  # ['a', 'ab', 'abc']
```

#### Why It Matters

- Produces canonical decomposition of a string
- Used in:

  * Suffix array construction (via BWT)
  * Lexicographically minimal rotations
  * Combinatorial string analysis
  * Free Lie algebra basis generation
  * Cryptography and DNA periodicity
- Linear time and space efficiency make it practical in text indexing.

#### Complexity

| Operation              | Time   | Space  |    |        |
| ---------------------- | ------ | ------ | -- | ------ |
| Factorization (Duval)  | $O(n)$ | $O(1)$ |    |        |
| Verify Lyndon property | $O(    | w      | )$ | $O(1)$ |

#### Try It Yourself

1. Factorize `"aababc"` and verify each factor is Lyndon.
2. Find the smallest rotation of `"cabab"` using Lyndon properties.
3. Apply to `"zzzzyzzzzz"` and analyze pattern.
4. Generate all Lyndon words up to length 3 over `{a, b}`.
5. Compare Duval's output with suffix array order.

#### A Gentle Proof (Why It Works)

Every string $S$ can be expressed as a nonincreasing sequence of Lyndon words —
and this factorization is unique.

The proof uses:

- Lexicographic minimality (each factor is the smallest prefix possible)
- Concatenation monotonicity (ensures order)
- Induction on length $n$

The Lyndon Factorization is the melody line of a string —
every factor a self-contained phrase,
each smaller echoing the rhythm of the one before.

### 640 Minimal Rotation (Booth's Algorithm)

The Minimal Rotation problem asks for the lexicographically smallest rotation of a string, the rotation that would come first in dictionary order.
Booth's Algorithm solves this in linear time, $O(n)$, using clever modular comparisons without generating all rotations.

This problem ties together ideas from Lyndon words, suffix arrays, and cyclic string order, and is foundational in string normalization, hashing, and pattern equivalence.

#### What Problem Are We Solving?

Given a string $S$ of length $n$, consider all its rotations:

$$
R_i = S[i..n-1] + S[0..i-1], \quad i = 0, 1, \ldots, n-1
$$

We want to find the index $k$ such that $R_k$ is lexicographically smallest.

#### Example

Let
$$
S = \texttt{"bbaaccaadd"}
$$

All rotations:

| Shift | Rotation    |
| :---: | :---------- |
|   0   | bbaaccaadd  |
|   1   | baaccaaddb  |
|   2   | aaccaaddbb  |
|   3   | accaaddbba  |
|   4   | ccaaddbb aa |
|   …   | …           |

The smallest rotation is
$$
R_2 = \texttt{"aaccaaddbb"}
$$

So rotation index = 2.

#### The Naive Way

Generate all rotations, then sort, $O(n^2 \log n)$ time and $O(n^2)$ space.
Booth's Algorithm achieves $O(n)$ time and $O(1)$ extra space by comparing characters cyclically with modular arithmetic.

#### Booth's Algorithm (Core Idea)

1. Concatenate the string with itself:
   $$
   T = S + S
   $$
   Now every rotation of $S$ is a substring of $T$ of length $n$.

2. Maintain a candidate index `k` for minimal rotation.
   For each position `i`, compare `T[k + j]` and `T[i + j]` character by character.

3. When a mismatch is found:

   * If `T[k + j] > T[i + j]`, the rotation at `i` is lexicographically smaller → update `k = i`.
   * Otherwise, skip ahead past the compared region.

4. Continue until all rotations are checked.

The algorithm cleverly ensures no redundant comparisons using arithmetic progressions.

#### Example Walkthrough

Let
$$
S = \texttt{"abab"} \quad (n = 4)
$$
$$
T = \texttt{"abababab"}
$$

Start `k = 0`, compare rotations starting at 0 and 1:

| Step   | Compare                       | Result  | New k  |
| ------ | ----------------------------- | ------- | ------ |
| 0 vs 1 | `a` vs `b`                    | `a < b` | keep 0 |
| 0 vs 2 | same prefix, next chars equal | skip    |        |
| 0 vs 3 | `a` vs `b`                    | keep 0  |        |

Smallest rotation starts at index 0 → `"abab"`.

#### Tiny Code (Python – Booth's Algorithm)

```python
def minimal_rotation(s):
    s += s
    n = len(s)
    f = [-1] * n  # failure function
    k = 0
    for j in range(1, n):
        i = f[j - k - 1]
        while i != -1 and s[j] != s[k + i + 1]:
            if s[j] < s[k + i + 1]:
                k = j - i - 1
            i = f[i]
        if s[j] != s[k + i + 1]:
            if s[j] < s[k]:
                k = j
            f[j - k] = -1
        else:
            f[j - k] = i + 1
    return k % (n // 2)

s = "bbaaccaadd"
idx = minimal_rotation(s)
print(idx, s[idx:] + s[:idx])  # 2 aaccaaddbb
```

#### Why It Matters

- Computes canonical form of cyclic strings
- Detects rotational equivalence
- Used in:

  * String hashing
  * DNA cyclic pattern recognition
  * Lexicographic normalization
  * Circular suffix array construction

It's a beautiful marriage of KMP's prefix logic and Lyndon's word theory.

#### Complexity

| Operation                | Time   | Space  |
| ------------------------ | ------ | ------ |
| Minimal rotation (Booth) | $O(n)$ | $O(1)$ |
| Verify rotation          | $O(n)$ | $O(1)$ |

#### Try It Yourself

1. `"bbaaccaadd"` → index 2, rotation `"aaccaaddbb"`
2. `"cabbage"` → index 1, rotation `"abbagec"`
3. `"aaaa"` → any rotation works
4. `"dcba"` → index 3, rotation `"adcb"`
5. Compare with brute-force rotation sorting to verify results.

#### A Gentle Proof (Why It Works)

Booth's algorithm maintains a candidate $k$ such that no rotation before $k$ can be smaller.
At each mismatch, skipping ensures we never reconsider rotations that share the same prefix pattern, similar to KMP's prefix-function logic.

Thus, total comparisons $\le 2n$, ensuring linear time.

The Minimal Rotation reveals the string's lexicographic core —
the rotation of purest order,
found not by brute force, but by rhythm and reflection within the string itself.

# Section 65. Edit Distance and Alignment 

### 641 Levenshtein Distance

The Levenshtein distance measures the *minimum number of edits* required to transform one string into another, where edits include insertions, deletions, and substitutions.
It's the foundational metric for string similarity, powering spell checkers, fuzzy search, DNA alignment, and chat autocorrect systems.

#### What Problem Are We Solving?

Given two strings:

$$
A = a_1 a_2 \ldots a_n, \quad B = b_1 b_2 \ldots b_m
$$

we want the smallest number of single-character operations to make $A$ equal to $B$:

- Insert one character
- Delete one character
- Replace one character

The result is the edit distance $D(n, m)$.

#### Example

| String A      | String B      | Edits        | Distance |
| ------------- | ------------- | ------------ | -------- |
| `"kitten"`    | `"sitting"`   | k→s, +i, +g  | 3        |
| `"flaw"`      | `"lawn"`      | -f, +n       | 2        |
| `"intention"` | `"execution"` | i→e, n→x, +u | 3        |

Each edit transforms $A$ step by step into $B$.

#### Recurrence Relation

Let $D[i][j]$ = minimal edits to transform $A[0..i-1]$ → $B[0..j-1]$

Then:

$$
D[i][j] =
\begin{cases}
0, & \text{if } i = 0,\, j = 0,\\
j, & \text{if } i = 0,\\
i, & \text{if } j = 0,\\[6pt]
\min
\begin{cases}
D[i-1][j] + 1, & \text{(deletion)}\\
D[i][j-1] + 1, & \text{(insertion)}\\
D[i-1][j-1] + (a_i \ne b_j), & \text{(substitution)}
\end{cases}, & \text{otherwise.}
\end{cases}
$$


#### How It Works (Plain Language)

You build a grid comparing every prefix of `A` to every prefix of `B`.
Each cell $D[i][j]$ represents the minimal edits so far.
By choosing the minimum among insertion, deletion, or substitution,
you "grow" the solution from the simplest cases.

#### Example Table

Compute `D("kitten", "sitting")`:

|    | "" | s | i | t | t | i | n | g |
| -- | -- | - | - | - | - | - | - | - |
| "" | 0  | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| k  | 1  | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| i  | 2  | 2 | 1 | 2 | 3 | 4 | 5 | 6 |
| t  | 3  | 3 | 2 | 1 | 2 | 3 | 4 | 5 |
| t  | 4  | 4 | 3 | 2 | 1 | 2 | 3 | 4 |
| e  | 5  | 5 | 4 | 3 | 2 | 2 | 3 | 4 |
| n  | 6  | 6 | 5 | 4 | 3 | 3 | 2 | 3 |

Ok Levenshtein distance = 3

#### Tiny Code (Python)

```python
def levenshtein(a, b):
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,      # deletion
                dp[i][j - 1] + 1,      # insertion
                dp[i - 1][j - 1] + cost  # substitution
            )
    return dp[n][m]

print(levenshtein("kitten", "sitting"))  # 3
```

#### Space-Optimized Version

We only need the previous row:

```python
def levenshtein_optimized(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            curr.append(min(
                prev[j] + 1,
                curr[-1] + 1,
                prev[j - 1] + cost
            ))
        prev = curr
    return prev[-1]
```

#### Why It Matters

- Fundamental similarity metric in text processing
- Used in:

  * Spell correction (`levenstein("color", "colour")`)
  * DNA sequence alignment
  * Approximate search
  * Chat autocorrect / fuzzy matching
- Provides interpretable results: every edit has meaning

#### Complexity

| Operation       | Time    | Space           |
| --------------- | ------- | --------------- |
| DP (full table) | $O(nm)$ | $O(nm)$         |
| DP (optimized)  | $O(nm)$ | $O(\min(n, m))$ |

#### Try It Yourself

1. `"flaw"` vs `"lawn"` → distance = 2
2. `"intention"` vs `"execution"` → 5
3. `"abc"` vs `"yabd"` → 2
4. Compute edit path by backtracking the DP table.
5. Compare runtime between full DP and optimized version.

#### A Gentle Proof (Why It Works)

The recurrence ensures optimal substructure:

- The minimal edit for prefixes extends naturally to longer prefixes.
  Each step considers all possible last operations, taking the smallest.
  Dynamic programming guarantees global optimality.

The Levenshtein distance is the true language of transformation —
each insertion a birth, each deletion a loss,
and each substitution a change of meaning that measures how far two words drift apart.

### 642 Damerau–Levenshtein Distance

The Damerau–Levenshtein distance extends the classic Levenshtein metric by recognizing that humans (and computers) often make a common fourth kind of typo, transposition, swapping two adjacent characters.
This extension captures a more realistic notion of "edit distance" in natural text, such as typing "hte" instead of "the".

#### What Problem Are We Solving?

We want the minimum number of operations to transform one string $A$ into another string $B$, using:

1. Insertion – add a character
2. Deletion – remove a character
3. Substitution – replace a character
4. Transposition – swap two adjacent characters

Formally, find $D(n, m)$, the minimal edit distance with these four operations.

#### Example

| String A | String B | Edits                           | Distance |
| -------- | -------- | ------------------------------- | -------- |
| `"ca"`   | `"ac"`   | transpose c↔a                   | 1        |
| `"abcd"` | `"acbd"` | transpose b↔c                   | 1        |
| `"abcf"` | `"acfb"` | substitution c→f, transpose f↔b | 2        |
| `"hte"`  | `"the"`  | transpose h↔t                   | 1        |

This distance better models real-world typos and biological swaps.

#### Recurrence Relation

Let $D[i][j]$ be the Damerau–Levenshtein distance between $A[0..i-1]$ and $B[0..j-1]$.

$$
D[i][j] =
\begin{cases}
\max(i, j), & \text{if } \min(i, j) = 0,\\[6pt]
\min
\begin{cases}
D[i-1][j] + 1, & \text{(deletion)}\\
D[i][j-1] + 1, & \text{(insertion)}\\
D[i-1][j-1] + (a_i \ne b_j), & \text{(substitution)}\\
D[i-2][j-2] + 1, & \text{if } i,j > 1,\, a_i=b_{j-1},\, a_{i-1}=b_j \text{ (transposition)}
\end{cases}
\end{cases}
$$


#### How It Works (Plain Language)

We fill a dynamic programming table just like Levenshtein distance,
but we add an extra check for the transposition case —
when two characters are swapped, e.g. `a_i == b_{j-1}` and `a_{i-1} == b_j`.
In that case, we can "jump diagonally two steps" with a cost of one.

#### Example

Compute distance between `"ca"` and `"ac"`:

|    | "" | a | c |
| -- | -- | - | - |
| "" | 0  | 1 | 2 |
| c  | 1  | 1 | 1 |
| a  | 2  | 1 | 1 |

The transposition (`c↔a`) allows `D[2][2] = 1`.
Ok Damerau–Levenshtein distance = 1.

#### Tiny Code (Python)

```python
def damerau_levenshtein(a, b):
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,      # deletion
                dp[i][j - 1] + 1,      # insertion
                dp[i - 1][j - 1] + cost  # substitution
            )
            # transposition
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                dp[i][j] = min(dp[i][j], dp[i - 2][j - 2] + 1)
    return dp[n][m]

print(damerau_levenshtein("ca", "ac"))  # 1
```

#### Why It Matters

- Models human typing errors (swapped letters, e.g. "teh", "hte")
- Used in:

  * Spell checkers
  * Fuzzy search engines
  * Optical character recognition (OCR)
  * Speech-to-text correction
  * Genetic sequence analysis (for local transpositions)

Adding the transposition operation brings the model closer to natural data noise.

#### Complexity

| Operation                | Time    | Space           |
| ------------------------ | ------- | --------------- |
| DP (full table)          | $O(nm)$ | $O(nm)$         |
| Optimized (rolling rows) | $O(nm)$ | $O(\min(n, m))$ |

#### Try It Yourself

1. `"ab"` → `"ba"` → 1 (swap)
2. `"abcdef"` → `"abdcef"` → 1 (transpose d↔c)
3. `"sponge"` → `"spnoge"` → 1
4. Compare `"hte"` vs `"the"` → 1
5. Compare with Levenshtein distance to see when transpositions matter.

#### A Gentle Proof (Why It Works)

The transposition case extends the dynamic program by considering a 2-step diagonal, ensuring optimal substructure still holds.
Each path through the DP grid corresponds to a sequence of edits;
adding transpositions does not break optimality because we still choose the minimal-cost local transition.

The Damerau–Levenshtein distance refines our sense of textual closeness —
it doesn't just see missing or wrong letters,
it *understands when your fingers danced out of order.*

### 643 Hamming Distance

The Hamming distance measures how many positions differ between two strings of equal length.
It's the simplest and most direct measure of dissimilarity between binary codes, DNA sequences, or fixed-length text fragments, a perfect tool for detecting errors, mutations, or noise in transmission.

#### What Problem Are We Solving?

Given two strings $A$ and $B$ of equal length $n$:

$$
A = a_1 a_2 \ldots a_n, \quad B = b_1 b_2 \ldots b_n
$$

the Hamming distance is the number of positions $i$ where $a_i \ne b_i$:

$$
H(A, B) = \sum_{i=1}^{n} [a_i \ne b_i]
$$

It tells us *how many substitutions* would be needed to make them identical
(no insertions or deletions allowed).

#### Example

| A       | B       | Differences      | Hamming Distance |
| ------- | ------- | ---------------- | ---------------- |
| 1011101 | 1001001 | 2 bits differ    | 2                |
| karolin | kathrin | 3 letters differ | 3                |
| 2173896 | 2233796 | 3 digits differ  | 3                |

Only substitutions are counted, so both strings must be the same length.

#### How It Works (Plain Language)

Just walk through both strings together, character by character,
and count how many positions don't match.
That's the Hamming distance, nothing more, nothing less.

#### Tiny Code (Python)

```python
def hamming_distance(a, b):
    if len(a) != len(b):
        raise ValueError("Strings must be of equal length")
    return sum(c1 != c2 for c1, c2 in zip(a, b))

print(hamming_distance("karolin", "kathrin"))  # 3
print(hamming_distance("1011101", "1001001"))  # 2
```

#### Bitwise Version (for Binary Strings)

If $A$ and $B$ are integers, use XOR to find differing bits:

```python
def hamming_bits(x, y):
    return bin(x ^ y).count("1")

print(hamming_bits(0b1011101, 0b1001001))  # 2
```

Because XOR highlights exactly the differing bits.

#### Why It Matters

- Error detection – measures how many bits flipped in transmission
- Genetics – counts nucleotide mutations
- Hashing & ML – quantifies similarity between binary fingerprints
- Cryptography – evaluates diffusion (bit changes under encryption)

It's one of the cornerstones of information theory, introduced by Richard Hamming in 1950.

#### Complexity

| Operation         | Time                    | Space  |
| ----------------- | ----------------------- | ------ |
| Direct comparison | $O(n)$                  | $O(1)$ |
| Bitwise XOR       | $O(1)$ per machine word | $O(1)$ |

#### Try It Yourself

1. Compare `"1010101"` vs `"1110001"` → 4
2. Compute $H(0b1111, 0b1001)$ → 2
3. Count mutations between `"AACCGGTT"` and `"AAACGGTA"` → 2
4. Implement Hamming similarity = $1 - \frac{H(A,B)}{n}$
5. Use it in a binary nearest-neighbor search.

#### A Gentle Proof (Why It Works)

Each mismatch contributes +1 to the total count,
and since the operation is independent per position,
the sum directly measures substitution count —
a simple metric that satisfies all distance axioms:
non-negativity, symmetry, and triangle inequality.

The Hamming distance is minimalism in motion —
a ruler that measures difference one symbol at a time,
from codewords to chromosomes.



### 644 Needleman–Wunsch Algorithm

The Needleman–Wunsch algorithm is the classic dynamic programming method for global sequence alignment.
It finds the *optimal way* to align two full sequences, character by character, by allowing insertions, deletions, and substitutions with given scores.

This algorithm forms the backbone of computational biology, comparing genes, proteins, or any sequences where *every part matters*.

#### What Problem Are We Solving?

Given two sequences:

$$
A = a_1 a_2 \ldots a_n, \quad B = b_1 b_2 \ldots b_m
$$

we want to find the best alignment (possibly with gaps) that maximizes a similarity score.

We define scoring parameters:

- match reward = $+M$
- mismatch penalty = $-S$
- gap penalty = $-G$

The goal is to find an alignment with maximum total score.

#### Example

Align `"GATTACA"` and `"GCATGCU"`:

One possible alignment:

```
G A T T A C A
| |   |   | |
G C A T G C U
```

The algorithm will explore all possibilities and return the *best* alignment by total score.

#### Recurrence Relation

Let $D[i][j]$ = best score for aligning $A[0..i-1]$ with $B[0..j-1]$.

Base cases:

$$
D[0][0] = 0, \quad
D[i][0] = -iG, \quad
D[0][j] = -jG
$$

Recurrence:

$$
D[i][j] = \max
\begin{cases}
D[i-1][j-1] + \text{score}(a_i, b_j), & \text{(match/mismatch)}\\
D[i-1][j] - G, & \text{(gap in B)}\\
D[i][j-1] - G, & \text{(gap in A)}
\end{cases}
$$


where:

$$
\text{score}(a_i, b_j) =
\begin{cases}
+M, & \text{if } a_i = b_j,\\
-S, & \text{if } a_i \ne b_j.
\end{cases}
$$


#### How It Works (Plain Language)

1. Build a scoring matrix of size $(n+1) \times (m+1)$.
2. Initialize first row and column with cumulative gap penalties.
3. Fill each cell using the recurrence rule, each step considers match, delete, or insert.
4. Backtrack from bottom-right to recover the best alignment path.

#### Example Table

For small sequences:

|    | "" | G  | C  | A  |
| -- | -- | -- | -- | -- |
| "" | 0  | -2 | -4 | -6 |
| G  | -2 | 1  | -1 | -3 |
| A  | -4 | -1 | 0  | 0  |
| C  | -6 | -3 | 0  | 1  |

Final score = 1 → best global alignment found by backtracking.

#### Tiny Code (Python)

```python
def needleman_wunsch(a, b, match=1, mismatch=-1, gap=-2):
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    # initialize
    for i in range(1, n + 1):
        dp[i][0] = i * gap
    for j in range(1, m + 1):
        dp[0][j] = j * gap

    # fill matrix
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            score = match if a[i - 1] == b[j - 1] else mismatch
            dp[i][j] = max(
                dp[i - 1][j - 1] + score,
                dp[i - 1][j] + gap,
                dp[i][j - 1] + gap
            )
    return dp[n][m]
```

#### Why It Matters

- Foundational in bioinformatics for DNA/protein alignment
- Used in:

  * Comparing genetic sequences
  * Plagiarism and text similarity detection
  * Speech and time-series matching

Needleman–Wunsch guarantees the optimal global alignment, unlike Smith–Waterman, which is local.

#### Complexity

| Operation     | Time     | Space   |
| ------------- | -------- | ------- |
| Fill DP table | $O(nm)$  | $O(nm)$ |
| Backtracking  | $O(n+m)$ | $O(1)$  |

Memory can be optimized to $O(\min(n,m))$ if only the score is needed.

#### Try It Yourself

1. Align `"GATTACA"` and `"GCATGCU"`.
2. Change gap penalty and observe how alignments shift.
3. Modify scoring for mismatches → softer penalties give longer alignments.
4. Compare with Smith–Waterman to see the local-global difference.

#### A Gentle Proof (Why It Works)

The DP structure ensures optimal substructure —
the best alignment of prefixes builds from smaller optimal alignments.
By evaluating match, insert, and delete at each step,
the algorithm always retains the globally best alignment path.

The Needleman–Wunsch algorithm is the archetype of alignment —
balancing matches and gaps,
it teaches sequences how to meet halfway.

### 645 Smith–Waterman Algorithm

The Smith–Waterman algorithm is the dynamic programming method for local sequence alignment, finding the *most similar subsequences* between two sequences.
Unlike Needleman–Wunsch, which aligns the *entire* sequences, Smith–Waterman focuses only on the best matching region, where true biological or textual similarity lies.

#### What Problem Are We Solving?

Given two sequences:

$$
A = a_1 a_2 \ldots a_n, \quad B = b_1 b_2 \ldots b_m
$$

find the pair of substrings $(A[i_1..i_2], B[j_1..j_2])$
that maximizes the local alignment score, allowing gaps and mismatches.

#### Scoring Scheme

Define scoring parameters:

- match reward = $+M$
- mismatch penalty = $-S$
- gap penalty = $-G$

The goal is to find:

$$
\max_{i,j} D[i][j]
$$

where $D[i][j]$ represents the best local alignment ending at $a_i$ and $b_j$.

#### Recurrence Relation

Base cases:

$$
D[0][j] = D[i][0] = 0
$$

Recurrence:

$$
D[i][j] = \max
\begin{cases}
0, & \text{(start new alignment)}\\
D[i-1][j-1] + \text{score}(a_i, b_j), & \text{(match/mismatch)}\\
D[i-1][j] - G, & \text{(gap in B)}\\
D[i][j-1] - G, & \text{(gap in A)}
\end{cases}
$$

where

$$
\text{score}(a_i, b_j) =
\begin{cases}
+M, & \text{if } a_i = b_j,\\
-S, & \text{if } a_i \ne b_j.
\end{cases}
$$


The 0 resets alignment when the score drops below zero —
ensuring we only keep high-similarity regions.

#### Example

Align `"ACACACTA"` and `"AGCACACA"`.

Smith–Waterman detects the strongest overlap:

```
A C A C A C T A
| | | | |
A G C A C A C A
```

Best local alignment: `"ACACA"`
Ok Local alignment score = 10 (for match = +2, mismatch = -1, gap = -2)

#### How It Works (Plain Language)

1. Build a DP matrix, starting with zeros.
2. For each pair of positions $(i, j)$:

   * Compute best local score ending at $(i, j)$.
   * Reset to zero if alignment becomes negative.
3. Track the maximum score in the matrix.
4. Backtrack from that cell to reconstruct the highest-scoring local subsequence.

#### Tiny Code (Python)

```python
def smith_waterman(a, b, match=2, mismatch=-1, gap=-2):
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    max_score = 0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            score = match if a[i - 1] == b[j - 1] else mismatch
            dp[i][j] = max(
                0,
                dp[i - 1][j - 1] + score,
                dp[i - 1][j] + gap,
                dp[i][j - 1] + gap
            )
            max_score = max(max_score, dp[i][j])

    return max_score
```

#### Why It Matters

- Models biological similarity, detects conserved regions, not entire genome alignment
- Used in:

  * Bioinformatics (protein/DNA local alignment)
  * Text similarity and plagiarism detection
  * Pattern matching with noise
  * Fuzzy substring matching

Smith–Waterman ensures that only the *best-matching portion* contributes to the score, avoiding penalties from unrelated prefixes/suffixes.

#### Complexity

| Operation       | Time    | Space          |
| --------------- | ------- | -------------- |
| DP (full table) | $O(nm)$ | $O(nm)$        |
| Space optimized | $O(nm)$ | $O(\min(n,m))$ |

#### Try It Yourself

1. `"GATTACA"` vs `"GCATGCU"` → local alignment `"ATG"`
2. `"ACACACTA"` vs `"AGCACACA"` → `"ACACA"`
3. Change gap penalty from 2 to 5 → how does alignment shrink?
4. Compare global vs local alignment outputs (Needleman–Wunsch vs Smith–Waterman).
5. Apply to `"hello"` vs `"yellow"` → find shared region.

#### A Gentle Proof (Why It Works)

The inclusion of 0 in the recurrence ensures optimal local behavior:
whenever the running score becomes negative, we restart alignment.
Dynamic programming guarantees that all possible substrings are considered,
and the global maximum corresponds to the strongest local match.

The Smith–Waterman algorithm listens for echoes in the noise —
finding the brightest overlap between two long melodies,
and telling you where they truly harmonize.


### 646 Hirschberg's Algorithm

The Hirschberg algorithm is a clever optimization of the Needleman–Wunsch alignment.
It produces the same global alignment, but using only linear space, $O(n + m)$, instead of $O(nm)$.
This makes it ideal for aligning long DNA or text sequences where memory is tight.

#### What Problem Are We Solving?

We want to compute a global sequence alignment (like Needleman–Wunsch) between:

$$
A = a_1 a_2 \ldots a_n, \quad B = b_1 b_2 \ldots b_m
$$

but we want to do so using linear space, not quadratic.
The trick is to compute only the scores needed to reconstruct the optimal path, not the full DP table.

#### The Key Insight

The classic Needleman–Wunsch algorithm fills an $n \times m$ DP matrix to find an optimal alignment path.

But:

- We only need half of the table at any time to compute scores.
- The middle column of the DP table divides the problem into two independent halves.

By combining these two facts, Hirschberg finds the split point of the alignment recursively.

#### Algorithm Outline

1. Base case:

   * If either string is empty → return a sequence of gaps.
   * If either string has length 1 → do a simple alignment directly.

2. Divide:

   * Split $A$ in half: $A = A_{\text{left}} + A_{\text{right}}$.
   * Compute forward alignment scores of $A_{\text{left}}$ with $B$.
   * Compute backward alignment scores of $A_{\text{right}}$ with $B$ (reversed).
   * Add corresponding scores to find the best split point in $B$.

3. Recurse:

   * Align the two halves $(A_{\text{left}}, B_{\text{left}})$ and $(A_{\text{right}}, B_{\text{right}})$ recursively.

4. Combine:

   * Merge the two sub-alignments into a full global alignment.

#### Recurrence Relation

We use the Needleman–Wunsch scoring recurrence:

$$
D[i][j] = \max
\begin{cases}
D[i-1][j-1] + s(a_i, b_j), & \text{match/mismatch},\\
D[i-1][j] - G, & \text{gap in B},\\
D[i][j-1] - G, & \text{gap in A}.
\end{cases}
$$


But only the *previous row* is kept in memory for each half,
and we find the optimal middle column split by combining forward and backward scores.

#### Example

Align `"AGTACGCA"` and `"TATGC"`.

- Split `"AGTACGCA"` into `"AGTA"` and `"CGCA"`.
- Compute forward DP for `"AGTA"` vs `"TATGC"`.
- Compute backward DP for `"CGCA"` vs `"TATGC"`.
- Combine scores to find the best split in `"TATGC"`.
- Recurse on two smaller alignments, merge results.

Final alignment matches the same as Needleman–Wunsch,
but with dramatically lower space cost.

#### Tiny Code (Python – simplified)

```python
def hirschberg(a, b, match=1, mismatch=-1, gap=-2):
    if len(a) == 0:
        return ("-" * len(b), b)
    if len(b) == 0:
        return (a, "-" * len(a))
    if len(a) == 1 or len(b) == 1:
        # base case: simple Needleman-Wunsch
        return needleman_wunsch_align(a, b, match, mismatch, gap)

    mid = len(a) // 2
    scoreL = nw_score(a[:mid], b, match, mismatch, gap)
    scoreR = nw_score(a[mid:][::-1], b[::-1], match, mismatch, gap)
    j_split = max(range(len(b) + 1), key=lambda j: scoreL[j] + scoreR[len(b) - j])
    left = hirschberg(a[:mid], b[:j_split], match, mismatch, gap)
    right = hirschberg(a[mid:], b[j_split:], match, mismatch, gap)
    return (left[0] + right[0], left[1] + right[1])
```

*(Helper `nw_score` computes Needleman–Wunsch row scores for one direction.)*

#### Why It Matters

- Uses linear memory with optimal alignment quality.
- Ideal for:

  * Genome sequence alignment
  * Massive document comparisons
  * Low-memory environments
- Preserves Needleman–Wunsch correctness, improving practicality for big data.

#### Complexity

| Operation | Time    | Space      |
| --------- | ------- | ---------- |
| Alignment | $O(nm)$ | $O(n + m)$ |

The recursive splitting introduces small overhead but no asymptotic penalty.

#### Try It Yourself

1. Align `"GATTACA"` with `"GCATGCU"` using both Needleman–Wunsch and Hirschberg, confirm identical output.
2. Test with sequences of 10,000+ length, watch the memory savings.
3. Experiment with different gap penalties to see how the split point changes.
4. Visualize the recursion tree, it divides neatly down the middle.

#### A Gentle Proof (Why It Works)

Each middle column score pair $(L[j], R[m - j])$ represents
the best possible alignment that passes through cell $(\text{mid}, j)$.
By choosing the $j$ that maximizes $L[j] + R[m - j]$,
we ensure the globally optimal alignment crosses that point.
This preserves optimal substructure, guaranteeing correctness.

The Hirschberg algorithm is elegance by reduction —
it remembers only what's essential,
aligning vast sequences with the grace of a minimalist mathematician.

### 647 Edit Script Reconstruction

Once we compute the edit distance between two strings, we often want more than just the number, we want to know *how* to transform one into the other.
That transformation plan is called an edit script: the ordered sequence of operations (insert, delete, substitute) that converts string A into string B optimally.

#### What Problem Are We Solving?

Given two strings:

$$
A = a_1 a_2 \ldots a_n, \quad B = b_1 b_2 \ldots b_m
$$

and their minimal edit distance $D[n][m]$,
we want to reconstruct the series of edit operations that achieves that minimal cost.

Operations:

| Symbol | Operation  | Description              |
| :----: | :--------- | :----------------------- |
|   `M`  | Match      | $a_i = b_j$              |
|   `S`  | Substitute | replace $a_i$ with $b_j$ |
|   `I`  | Insert     | add $b_j$ into $A$       |
|   `D`  | Delete     | remove $a_i$ from $A$    |

The output is a human-readable edit trace like:

```
M M S I M D
```

#### Example

Transform `"kitten"` → `"sitting"`.

| Step | Operation          | Result    |
| :--: | :----------------- | :-------- |
|   1  | Substitute `k → s` | "sitten"  |
|   2  | Insert `i`         | "sittien" |
|   3  | Insert `g`         | "sitting" |

Ok Edit distance = 3
Ok Edit script = `S, I, I`

#### How It Works (Plain Language)

1. Compute the full Levenshtein DP table $D[i][j]$.
2. Start from bottom-right $(n, m)$.
3. Move backward:

   * If characters match → `M` (diagonal move)
   * Else if $D[i][j] = D[i-1][j-1] + 1$ → `S`
   * Else if $D[i][j] = D[i-1][j] + 1$ → `D`
   * Else if $D[i][j] = D[i][j-1] + 1$ → `I`
4. Record the operation and move accordingly.
5. Reverse the list at the end.

#### Example Table (Simplified)

|    | "" | s | i | t | t | i | n | g |
| -- | -- | - | - | - | - | - | - | - |
| "" | 0  | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| k  | 1  | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| i  | 2  | 2 | 1 | 2 | 3 | 4 | 5 | 6 |
| t  | 3  | 3 | 2 | 1 | 2 | 3 | 4 | 5 |
| t  | 4  | 4 | 3 | 2 | 1 | 2 | 3 | 4 |
| e  | 5  | 5 | 4 | 3 | 2 | 2 | 3 | 4 |
| n  | 6  | 6 | 5 | 4 | 3 | 3 | 2 | 3 |

Backtrack path: diagonal (S), right (I), right (I).
Reconstructed edit script = `[Substitute, Insert, Insert]`.

#### Tiny Code (Python)

```python
def edit_script(a, b):
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,      # deletion
                dp[i][j - 1] + 1,      # insertion
                dp[i - 1][j - 1] + cost  # substitution or match
            )

    # backtrack
    ops = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and a[i - 1] == b[j - 1]:
            ops.append("M")
            i, j = i - 1, j - 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            ops.append(f"S:{a[i - 1]}->{b[j - 1]}")
            i, j = i - 1, j - 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            ops.append(f"D:{a[i - 1]}")
            i -= 1
        else:
            ops.append(f"I:{b[j - 1]}")
            j -= 1

    return ops[::-1]

print(edit_script("kitten", "sitting"))
```

Output:

```
$$'S:k->s', 'M', 'M', 'M', 'I:i', 'M', 'I:g']
```

#### Why It Matters

- Converts distance metrics into explainable transformations
- Used in:

  * Diff tools (e.g. `git diff`, Myers diff)
  * Spelling correction
  * DNA edit tracing
  * Version control systems
  * Document merge tools

Without edit reconstruction, we know *how far* two strings are —
with it, we know *how to get there*.

#### Complexity

| Operation             | Time     | Space   |
| --------------------- | -------- | ------- |
| DP table construction | $O(nm)$  | $O(nm)$ |
| Backtracking          | $O(n+m)$ | $O(1)$  |

Space can be reduced with Hirschberg's divide-and-conquer backtrace.

#### Try It Yourself

1. `"flaw"` → `"lawn"` → `D:f, M, M, I:n`
2. `"sunday"` → `"saturday"` → multiple insertions
3. Reverse the script to get inverse transformation.
4. Modify cost function: make substitution more expensive.
5. Visualize path on the DP grid, it traces your script.

#### A Gentle Proof (Why It Works)

The DP table encodes minimal edit costs for all prefixes.
By walking backward from $(n, m)$, each local choice (diagonal, up, left)
represents the exact operation that achieved optimal cost.
Thus, the backtrace reconstructs the minimal transformation path.

The edit script is the diary of transformation —
a record of what changed, when, and how —
turning raw distance into a story of difference.

### 648 Affine Gap Penalty Dynamic Programming

The Affine Gap Penalty model improves upon simple gap scoring in sequence alignment.
Instead of charging a flat penalty per gap symbol, it distinguishes between gap opening and gap extension,
reflecting biological or textual reality, it's costly to *start* a gap, but cheaper to *extend* it.

#### What Problem Are We Solving?

In classical alignment (Needleman–Wunsch or Smith–Waterman),
every gap is penalized linearly:

$$
\text{gap cost} = k \times g
$$

But in practice, a single long gap is *less bad* than many short ones.
So we switch to an affine model:

$$
\text{gap cost} = g_o + (k - 1) \times g_e
$$

where

- $g_o$ = gap opening penalty
- $g_e$ = gap extension penalty
  and $k$ = length of the gap.

This model gives smoother, more realistic alignments.

#### Example

Suppose $g_o = 5$, $g_e = 1$.

| Gap          | Linear Model | Affine Model |
| ------------ | ------------ | ------------ |
| 1-symbol gap | 5            | 5            |
| 3-symbol gap | 15           | 7            |
| 5-symbol gap | 25           | 9            |

Affine scoring *rewards longer continuous gaps* and avoids scattered gaps.

#### How It Works (Plain Language)

We track three DP matrices instead of one:

| Matrix    | Meaning                                    |
| :-------- | :----------------------------------------- |
| $M[i][j]$ | best score ending in a match/mismatch      |
| $X[i][j]$ | best score ending with a gap in sequence A |
| $Y[i][j]$ | best score ending with a gap in sequence B |

Each matrix uses different recurrence relations to model gap transitions properly.

#### Recurrence Relations

Let $a_i$ and $b_j$ be the current characters.

$$
\begin{aligned}
M[i][j] &= \max \big(
M[i-1][j-1], X[i-1][j-1], Y[i-1][j-1]
\big) + s(a_i, b_j) [6pt]
X[i][j] &= \max \big(
M[i-1][j] - g_o,; X[i-1][j] - g_e
\big) [6pt]
Y[i][j] &= \max \big(
M[i][j-1] - g_o,; Y[i][j-1] - g_e
\big)
\end{aligned}
$$

where

$$
s(a_i, b_j) =
\begin{cases}
+M, & \text{if } a_i = b_j,\\
-S, & \text{if } a_i \ne b_j.
\end{cases}
$$


The final alignment score:

$$
D[i][j] = \max(M[i][j], X[i][j], Y[i][j])
$$

#### Initialization

$$
M[0][0] = 0, \quad X[0][0] = Y[0][0] = -\infty
$$

For first row/column:

$$
X[i][0] = -g_o - (i - 1) g_e, \quad
Y[0][j] = -g_o - (j - 1) g_e
$$

#### Example (Intuition)

Let's align:

```
A:  G A T T A C A
B:  G C A T G C U
```

With:

- match = +2
- mismatch = -1
- gap open = 5
- gap extend = 1

Small gaps will appear where needed,
but long insertions will stay continuous instead of splitting,
because continuing a gap is cheaper than opening a new one.

#### Tiny Code (Python)

```python
def affine_gap(a, b, match=2, mismatch=-1, gap_open=5, gap_extend=1):
    n, m = len(a), len(b)
    neg_inf = float("-inf")
    M = [[0] * (m + 1) for _ in range(n + 1)]
    X = [[neg_inf] * (m + 1) for _ in range(n + 1)]
    Y = [[neg_inf] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        M[i][0] = -gap_open - (i - 1) * gap_extend
        X[i][0] = M[i][0]
    for j in range(1, m + 1):
        M[0][j] = -gap_open - (j - 1) * gap_extend
        Y[0][j] = M[0][j]

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            score = match if a[i - 1] == b[j - 1] else mismatch
            M[i][j] = max(M[i - 1][j - 1], X[i - 1][j - 1], Y[i - 1][j - 1]) + score
            X[i][j] = max(M[i - 1][j] - gap_open, X[i - 1][j] - gap_extend)
            Y[i][j] = max(M[i][j - 1] - gap_open, Y[i][j - 1] - gap_extend)

    return max(M[n][m], X[n][m], Y[n][m])
```

#### Why It Matters

- Models biological gaps more realistically (e.g. insertions/deletions in DNA).
- Produces cleaner alignments for text or speech.
- Used in:

  * Needleman–Wunsch and Smith–Waterman extensions
  * BLAST, FASTA, and bioinformatics pipelines
  * Dynamic time warping variants in ML and signal analysis

Affine penalties mirror the intuition that starting an error costs more than continuing one.

#### Complexity

| Operation       | Time    | Space           |
| --------------- | ------- | --------------- |
| DP (3 matrices) | $O(nm)$ | $O(nm)$         |
| Space optimized | $O(nm)$ | $O(\min(n, m))$ |

#### Try It Yourself

1. Compare linear vs affine gaps for `"GATTACA"` vs `"GCATGCU"`.
2. Test long insertions, affine scoring will prefer one large gap.
3. Adjust gap penalties and see how alignment changes.
4. Combine affine scoring with local alignment (Smith–Waterman).
5. Visualize $M$, $X$, and $Y$ matrices separately.

#### A Gentle Proof (Why It Works)

Each of the three matrices represents a state machine:

- $M$ → in a match state,
- $X$ → in a gap-in-A state,
- $Y$ → in a gap-in-B state.

The affine recurrence ensures optimal substructure because transitions between states incur exactly the proper open/extend penalties.
Thus, every path through the combined system yields an optimal total score under affine cost.

The Affine Gap Penalty model brings realism to alignment —
understanding that beginnings are costly,
but continuations are sometimes just persistence.

### 649 Myers Bit-Vector Algorithm

The Myers Bit-Vector Algorithm is a brilliant optimization for computing edit distance (Levenshtein distance) between short strings or patterns, especially in search and matching tasks.
It uses bitwise operations to simulate dynamic programming in parallel across multiple positions, achieving near-linear speed on modern CPUs.

#### What Problem Are We Solving?

Given two strings
$$
A = a_1 a_2 \ldots a_n, \quad B = b_1 b_2 \ldots b_m
$$
we want to compute their edit distance (insertions, deletions, substitutions).

The classical dynamic programming solution takes $O(nm)$ time.
Myers reduces this to $O(n \cdot \lceil m / w \rceil)$,
where $w$ is the machine word size (typically 32 or 64).

This makes it ideal for approximate string search —
for example, finding all matches of `"pattern"` in a text within edit distance ≤ k.

#### Core Idea

The Levenshtein DP recurrence can be viewed as updating a band of cells that depend only on the previous row.
If we represent each row as bit vectors,
we can perform all cell updates at once using bitwise AND, OR, XOR, and shift operations.

For short patterns, all bits fit in a single word,
so updates happen in constant time.

#### Representation

We define several bit masks of length $m$ (pattern length):

- Eq[c] – a bitmask marking where character `c` appears in the pattern.
  Example for pattern `"ACCA"`:

  ```
  Eq['A'] = 1001
  Eq['C'] = 0110
  ```

During the algorithm, we maintain:

|  Symbol | Meaning                                                              |
| :-----: | :------------------------------------------------------------------- |
|   `Pv`  | bit vector of positions where there may be a positive difference |
|   `Mv`  | bit vector of positions where there may be a negative difference |
| `Score` | current edit distance                                                |

These encode the running state of the edit DP.

#### Recurrence (Bit-Parallel Form)

For each text character `t_j`:

$$
\begin{aligned}
Xv &= \text{Eq}[t_j] ; \lor ; Mv \
Xh &= (((Xv & Pv) + Pv) \oplus Pv) ; \lor ; Xv \
Ph &= Mv ; \lor ; \neg(Xh \lor Pv) \
Mh &= Pv ; & Xh
\end{aligned}
$$

Then shift and update the score:

$$
\begin{cases}
\text{if } (Ph \;\&\; \text{bit}_m) \ne 0, & \text{then Score++},\\
\text{if } (Mh \;\&\; \text{bit}_m) \ne 0, & \text{then Score--}.
\end{cases}
$$

Finally, set:

$$
\begin{aligned}
Pv &= Mh \;\lor\; \neg(Xh \lor Ph),\\
Mv &= Ph \;\&\; Xh.
\end{aligned}
$$


#### How It Works (Plain Language)

Think of each bit in `Pv` and `Mv` as representing a column in the DP table.
Instead of updating each cell one by one,
bit-operations update all columns in parallel, one CPU instruction updates 64 comparisons.

At each step:

- Eq[c] signals where matches occur.
- Pv, Mv track cumulative mismatches.
- The score adjusts as bits overflow at the top (edit cost propagation).

The algorithm's loop is extremely tight, just a handful of bitwise ops.

#### Example (Conceptual)

Pattern: `"ACGT"`
Text: `"AGT"`

We initialize:

```
Eq['A'] = 1000
Eq['C'] = 0100
Eq['G'] = 0010
Eq['T'] = 0001
```

Then process each character of text `"A"`, `"G"`, `"T"` in turn,
updating bit vectors and keeping the current edit distance in a scalar `Score`.

Final Score = 1
Ok Edit distance = 1 (one deletion).

#### Tiny Code (Python)

Below is a simplified single-word implementation:

```python
def myers_distance(pattern, text):
    m = len(pattern)
    Peq = {}
    for c in set(pattern + text):
        Peq[c] = 0
    for i, ch in enumerate(pattern):
        Peq[ch] |= 1 << i

    Pv = (1 << m) - 1
    Mv = 0
    score = m

    for ch in text:
        Eq = Peq.get(ch, 0)
        Xv = Eq | Mv
        Xh = (((Eq & Pv) + Pv) ^ Pv) | Eq
        Ph = Mv | ~(Xh | Pv)
        Mh = Pv & Xh

        if Ph & (1 << (m - 1)):
            score += 1
        elif Mh & (1 << (m - 1)):
            score -= 1

        Pv = (Mh << 1) | ~(Xh | (Ph << 1))
        Mv = (Ph << 1) & Xh

    return score

print(myers_distance("ACGT", "AGT"))  # 1
```

#### Why It Matters

- Fast approximate matching in text and DNA sequences
- Used in:

  * grep-like fuzzy search
  * read alignment in genomics (e.g. BWA, Bowtie)
  * autocorrect / spell check
  * real-time text comparison
- Operates with just bitwise ops and integer arithmetic
  → extremely fast, branch-free inner loop.

#### Complexity

| Operation     | Time                             | Space                    |
| ------------- | -------------------------------- | ------------------------ |
| Main loop     | $O(n \cdot \lceil m / w \rceil)$ | $O(\lceil m / w \rceil)$ |
| For $m \le w$ | $O(n)$                           | $O(1)$                   |

#### Try It Yourself

1. Compute edit distance between `"banana"` and `"bananas"`.
2. Compare runtime with classic DP for $m=8, n=100000$.
3. Modify to early-stop when `Score ≤ k`.
4. Use multiple words (bit-blocks) for long patterns.
5. Visualize bit evolution per step.

#### A Gentle Proof (Why It Works)

The standard Levenshtein recurrence depends only on the previous row.
Each bit in the word encodes whether the difference at that position increased or decreased.
Bitwise arithmetic emulates the carry and borrow propagation in integer addition/subtraction —
exactly reproducing the DP logic, but in parallel for every bit column.

The Myers Bit-Vector Algorithm turns edit distance into pure hardware logic —
aligning strings not by loops,
but by the rhythm of bits flipping in sync across a CPU register.

### 650 Longest Common Subsequence (LCS)

The Longest Common Subsequence (LCS) problem is one of the cornerstones of dynamic programming.
It asks: *Given two sequences, what is the longest sequence that appears in both (in the same order, but not necessarily contiguous)?*

It's the foundation for tools like `diff`, DNA alignment, and text similarity systems, anywhere we care about order-preserving similarity.

#### What Problem Are We Solving?

Given two sequences:

$$
A = a_1 a_2 \ldots a_n, \quad B = b_1 b_2 \ldots b_m
$$

find the longest sequence $C = c_1 c_2 \ldots c_k$
such that $C$ is a subsequence of both $A$ and $B$.

Formally:
$$
C \subseteq A, \quad C \subseteq B, \quad k = |C| \text{ is maximal.}
$$

We want both the length and optionally the subsequence itself.

#### Example

| A           | B           | LCS      | Length |
| ----------- | ----------- | -------- | ------ |
| `"ABCBDAB"` | `"BDCABA"`  | `"BCBA"` | 4      |
| `"AGGTAB"`  | `"GXTXAYB"` | `"GTAB"` | 4      |
| `"HELLO"`   | `"YELLOW"`  | `"ELLO"` | 4      |

#### Recurrence Relation

Let $L[i][j]$ be the LCS length of prefixes $A[0..i-1]$ and $B[0..j-1]$.

Then:

$$
L[i][j] =
\begin{cases}
0, & \text{if } i = 0 \text{ or } j = 0,\\[4pt]
L[i-1][j-1] + 1, & \text{if } a_i = b_j,\\[4pt]
\max(L[i-1][j],\, L[i][j-1]), & \text{otherwise.}
\end{cases}
$$


#### How It Works (Plain Language)

You build a 2D grid comparing prefixes of both strings.
Each cell $L[i][j]$ represents "how long is the LCS up to $a_i$ and $b_j$".

- If the characters match → extend the LCS by 1.
- If not → take the best from skipping one character in either string.

The value in the bottom-right corner is the final LCS length.

#### Example Table

For `"ABCBDAB"` vs `"BDCABA"`:

|    | "" | B | D | C | A | B | A |
| -- | -- | - | - | - | - | - | - |
| "" | 0  | 0 | 0 | 0 | 0 | 0 | 0 |
| A  | 0  | 0 | 0 | 0 | 1 | 1 | 1 |
| B  | 0  | 1 | 1 | 1 | 1 | 2 | 2 |
| C  | 0  | 1 | 1 | 2 | 2 | 2 | 2 |
| B  | 0  | 1 | 1 | 2 | 2 | 3 | 3 |
| D  | 0  | 1 | 2 | 2 | 2 | 3 | 3 |
| A  | 0  | 1 | 2 | 2 | 3 | 3 | 4 |
| B  | 0  | 1 | 2 | 2 | 3 | 4 | 4 |

Ok LCS length = 4
Ok One valid subsequence = `"BCBA"`

#### Tiny Code (Python)

```python
def lcs(a, b):
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[n][m]
```

To reconstruct the subsequence:

```python
def lcs_traceback(a, b):
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    # backtrack
    i, j = n, m
    seq = []
    while i > 0 and j > 0:
        if a[i - 1] == b[j - 1]:
            seq.append(a[i - 1])
            i -= 1
            j -= 1
        elif dp[i - 1][j] >= dp[i][j - 1]:
            i -= 1
        else:
            j -= 1
    return ''.join(reversed(seq))
```

Example:

```python
print(lcs_traceback("ABCBDAB", "BDCABA"))  # BCBA
```

#### Why It Matters

- Backbone of diff tools (`git diff`, text comparison, version control)
- DNA/protein similarity (invariant subsequences)
- Plagiarism detection
- Machine translation evaluation (BLEU-like metrics)
- Sequence compression and error correction

The LCS gives structural similarity, not exact matches, but shared order.

#### Complexity

| Operation               | Time    | Space           |
| ----------------------- | ------- | --------------- |
| DP (full table)         | $O(nm)$ | $O(nm)$         |
| Optimized (length only) | $O(nm)$ | $O(\min(n, m))$ |

#### Try It Yourself

1. `"HELLO"` vs `"YELLOW"` → `"ELLO"`
2. `"AGGTAB"` vs `"GXTXAYB"` → `"GTAB"`
3. Compute LCS similarity ratio = $2 \times LCS / (|A| + |B|)$
4. Use LCS to align text versions and detect edits.
5. Try it on code diffs, it's how `git` detects changed lines.

#### A Gentle Proof (Why It Works)

Each prefix of $A$ and $B$ defines subproblems with optimal substructure:

- If $a_i = b_j$, LCS extends previous prefix.
- Else, best LCS must drop one element from either $A$ or $B$.

Overlapping subproblems are solved once, stored in $L[i][j]$.
The recursion ensures all combinations are considered efficiently.

The Longest Common Subsequence is the quiet heart of comparison —
it finds what survives between change and difference,
the thread of sameness through time and transformation.

# Section 66. Compression 

### 651 Huffman Coding

Huffman Coding is a classic algorithm for lossless data compression.
It builds an optimal prefix code, meaning no codeword is a prefix of another, ensuring unique decodability.
By assigning shorter codes to frequent symbols and longer codes to rare ones, Huffman Coding minimizes total encoded length.

#### Problem Definition

Given an alphabet of symbols
$$
S = {s_1, s_2, \ldots, s_n}
$$
with corresponding frequencies
$$
f(s_i)
$$
we want to assign binary codes $C(s_i)$ such that:

1. The code is prefix-free (no code is a prefix of another).
2. The average code length
   $$
   L = \sum_i f(s_i) \cdot |C(s_i)|
   $$
   is minimal.

#### Key Idea

- Combine the two least frequent symbols repeatedly into a new node.
- Assign 0 and 1 to the two branches.
- The tree you build defines the prefix codes.

This process forms a binary tree where:

- Leaves represent original symbols.
- Path from root to leaf gives the binary code.

#### Algorithm Steps

1. Initialize a priority queue (min-heap) with all symbols weighted by frequency.
2. While more than one node remains:

   * Remove two nodes with smallest frequencies $f_1, f_2$.
   * Create a new internal node with frequency $f = f_1 + f_2$.
   * Insert it back into the queue.
3. When only one node remains, it is the root.
4. Traverse the tree:

   * Left branch = append `0`
   * Right branch = append `1`
   * Record codes for each leaf.

#### Example

Symbols and frequencies:

| Symbol | Frequency |
| :----: | :-------: |
|    A   |     45    |
|    B   |     13    |
|    C   |     12    |
|    D   |     16    |
|    E   |     9     |
|    F   |     5     |

Step-by-step tree building:

1. Combine F (5) + E (9) → new node (14)
2. Combine C (12) + B (13) → new node (25)
3. Combine D (16) + (14) → new node (30)
4. Combine (25) + (30) → new node (55)
5. Combine A (45) + (55) → new root (100)

Final codes (one valid solution):

| Symbol | Code |
| :----: | :--: |
|    A   |   0  |
|    B   |  101 |
|    C   |  100 |
|    D   |  111 |
|    E   | 1101 |
|    F   | 1100 |

Average code length:
$$
L = \frac{45(1) + 13(3) + 12(3) + 16(3) + 9(4) + 5(4)}{100} = 2.24 \text{ bits/symbol}
$$

#### Tiny Code (Python)

```python
import heapq

def huffman(freqs):
    heap = [[w, [sym, ""]] for sym, w in freqs.items()]
    heapq.heapify(heap)

    while len(heap) > 1:
        lo = heapq.heappop(heap)
        hi = heapq.heappop(heap)
        for pair in lo[1:]:
            pair[1] = "0" + pair[1]
        for pair in hi[1:]:
            pair[1] = "1" + pair[1]
        heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])

    return sorted(heapq.heappop(heap)[1:], key=lambda p: (len(p[1]), p))

freqs = {'A': 45, 'B': 13, 'C': 12, 'D': 16, 'E': 9, 'F': 5}
for sym, code in huffman(freqs):
    print(sym, code)
```

#### Why It Matters

- Forms the foundation of many real-world compression formats:

  * DEFLATE (ZIP, PNG)
  * JPEG and MP3 (after quantization)
- Minimizes the expected bit-length for symbol encoding.
- Demonstrates greedy optimality: combining smallest weights first yields the global minimum.

#### Complexity

| Operation         | Time                | Space               |
| ----------------- | ------------------- | ------------------- |
| Building tree     | $O(n \log n)$       | $O(n)$              |
| Encoding/decoding | $O(k)$ (per symbol) | $O(1)$ (per lookup) |

#### Try It Yourself

1. Use characters of `"HELLO WORLD"` with frequency counts.
2. Draw the Huffman tree manually.
3. Encode and decode a small string.
4. Compare average bit-length with fixed-length (ASCII = 8 bits).
5. Implement canonical Huffman codes for deterministic order.

#### A Gentle Proof (Why It Works)

Let $x$ and $y$ be the two least frequent symbols.
In an optimal prefix code, these two must appear as siblings at the greatest depth.
Replacing any other deeper pair with them would increase the average length.
By repeatedly applying this property, Huffman's greedy combination is always optimal.

Huffman Coding shows how greedy choice and tree structure work together
to make compression elegant, turning frequency into efficiency.

### 652 Canonical Huffman Coding

Canonical Huffman Coding is a refined, deterministic version of Huffman Coding.
It encodes symbols using the same code lengths as the original Huffman tree but arranges codes in lexicographic (canonical) order.
This makes decoding much faster and compactly represents the code table, ideal for file formats and network protocols.

#### What Problem Are We Solving?

In standard Huffman coding, multiple trees can represent the same optimal code lengths.
For example, codes `{A: 0, B: 10, C: 11}` and `{A: 1, B: 00, C: 01}` have the same total length.
However, storing or transmitting the full tree is wasteful.

Canonical Huffman eliminates this ambiguity by assigning codes deterministically
based only on symbol order and code lengths, not on tree structure.

#### Key Idea

Instead of storing the tree, we store code lengths for each symbol.
Then, we generate all codes in a consistent way:

1. Sort symbols by code length (shortest first).
2. Assign the smallest possible binary code to the first symbol.
3. Each next symbol's code = previous code + 1 (in binary).
4. When moving to a longer length, left-shift (append a zero).

This guarantees lexicographic order and prefix-free structure.

#### Example

Suppose we have symbols and their Huffman code lengths:

| Symbol | Length |
| :----: | :----: |
|    A   |    1   |
|    B   |    3   |
|    C   |    3   |
|    D   |    3   |
|    E   |    4   |

Step 1. Sort by (length, symbol):

`A (1), B (3), C (3), D (3), E (4)`

Step 2. Assign canonical codes:

| Symbol | Length | Code (binary) |
| :----: | :----: | :-----------: |
|    A   |    1   |       0       |
|    B   |    3   |      100      |
|    C   |    3   |      101      |
|    D   |    3   |      110      |
|    E   |    4   |      1110     |

Step 3. Increment codes sequentially

Start with all zeros of length = 1 for the first symbol,
then increment and shift as needed.

#### Pseudocode

```python
def canonical_huffman(lengths):
    # lengths: dict {symbol: code_length}
    sorted_syms = sorted(lengths.items(), key=lambda x: (x[1], x[0]))
    codes = {}
    code = 0
    prev_len = 0
    for sym, length in sorted_syms:
        code <<= (length - prev_len)
        codes[sym] = format(code, '0{}b'.format(length))
        code += 1
        prev_len = length
    return codes
```

Example run:

```python
lengths = {'A': 1, 'B': 3, 'C': 3, 'D': 3, 'E': 4}
print(canonical_huffman(lengths))
# {'A': '0', 'B': '100', 'C': '101', 'D': '110', 'E': '1110'}
```

#### Why It Matters

- Deterministic: every decoder reconstructs the same codes from code lengths.
- Compact: storing code lengths (one byte each) is much smaller than storing full trees.
- Fast decoding: tables can be generated using code-length ranges.
- Used in:

  * DEFLATE (ZIP, PNG, gzip)
  * JPEG
  * MPEG and MP3
  * Google's Brotli and Zstandard

#### Decoding Process

Given the canonical table:

| Length | Start Code | Count | Start Value |
| :----: | :--------: | :---: | :---------: |
|    1   |      0     |   1   |      A      |
|    3   |     100    |   3   |   B, C, D   |
|    4   |    1110    |   1   |      E      |

Decoding steps:

1. Read bits from input.
2. Track current code length.
3. If bits match a valid range → decode symbol.
4. Reset and continue.

This process uses range tables instead of trees, yielding $O(1)$ lookups.

#### Comparison with Standard Huffman

| Feature        | Standard Huffman        | Canonical Huffman      |
| -------------- | ----------------------- | ---------------------- |
| Storage        | Tree structure          | Code lengths only      |
| Uniqueness     | Non-deterministic       | Deterministic          |
| Decoding speed | Tree traversal          | Table lookup           |
| Common use     | Educational, conceptual | Real-world compressors |

#### Complexity

| Operation             | Time          | Space             |
| --------------------- | ------------- | ----------------- |
| Build canonical table | $O(n \log n)$ | $O(n)$            |
| Encode/decode         | $O(k)$        | $O(1)$ per symbol |

#### Try It Yourself

1. Take any Huffman code tree and extract code lengths.
2. Rebuild canonical codes from lengths.
3. Compare binary encodings, they decode identically.
4. Implement DEFLATE-style representation using `(symbol, bit length)` pairs.

#### A Gentle Proof (Why It Works)

The lexicographic ordering preserves prefix-freeness:
if one code has length $l_1$ and the next has $l_2 \ge l_1$,
incrementing the code and shifting ensures that no code is a prefix of another.
Thus, canonical codes produce the same compression ratio as the original Huffman tree.

Canonical Huffman Coding transforms optimal trees into simple arithmetic —
the same compression, but with order, predictability, and elegance.

### 653 Arithmetic Coding

Arithmetic Coding is a powerful lossless compression method that encodes an entire message as a single number between 0 and 1.
Unlike Huffman coding, which assigns discrete bit sequences to symbols, arithmetic coding represents the *whole message* as a fraction within an interval that shrinks as each symbol is processed.

It's widely used in modern compression formats like JPEG, H.264, and BZIP2.

#### What Problem Are We Solving?

Huffman coding can only assign codewords of integer bit lengths.
Arithmetic coding removes that restriction, it can assign fractional bits per symbol, achieving closer-to-optimal compression for any probability distribution.

The idea:
Each symbol narrows the interval based on its probability.
The final sub-interval uniquely identifies the message.

#### How It Works (Plain Language)

Start with the interval [0, 1).
Each symbol refines the interval proportional to its probability.

Example with symbols and probabilities:

| Symbol | Probability |    Range   |
| :----: | :---------: | :--------: |
|    A   |     0.5     | [0.0, 0.5) |
|    B   |     0.3     | [0.5, 0.8) |
|    C   |     0.2     | [0.8, 1.0) |

For the message `"BAC"`:

1. Start: [0.0, 1.0)
2. Symbol B → [0.5, 0.8)
3. Symbol A → take 0.5 + (0.8 - 0.5) × [0.0, 0.5) = [0.5, 0.65)
4. Symbol C → take 0.5 + (0.65 - 0.5) × [0.8, 1.0) = [0.62, 0.65)

Final range: [0.62, 0.65)
Any number in this range (say 0.63) uniquely identifies the message.

#### Mathematical Formulation

For a sequence of symbols $s_1, s_2, \ldots, s_n$
with cumulative probability ranges $[l_i, h_i)$ per symbol,
we iteratively compute:

$$
\begin{aligned}
\text{range} &= h - l, \
h' &= l + \text{range} \times \text{high}(s_i), \
l' &= l + \text{range} \times \text{low}(s_i).
\end{aligned}
$$

After processing all symbols, pick any number $x \in [l, h)$ as the code.

Decoding reverses the process by seeing where $x$ falls within symbol ranges.

#### Example Step Table

Encoding `"BAC"` with same probabilities:

| Step | Symbol | Interval Before | Range | New Interval |
| ---- | ------ | --------------- | ----- | ------------ |
| 1    | B      | [0.0, 1.0)      | 1.0   | [0.5, 0.8)   |
| 2    | A      | [0.5, 0.8)      | 0.3   | [0.5, 0.65)  |
| 3    | C      | [0.5, 0.65)     | 0.15  | [0.62, 0.65) |

Encoded number: 0.63

#### Tiny Code (Python)

```python
def arithmetic_encode(message, probs):
    low, high = 0.0, 1.0
    for sym in message:
        range_ = high - low
        cum_low = sum(v for k, v in probs.items() if k < sym)
        cum_high = cum_low + probs[sym]
        high = low + range_ * cum_high
        low = low + range_ * cum_low
    return (low + high) / 2

probs = {'A': 0.5, 'B': 0.3, 'C': 0.2}
code = arithmetic_encode("BAC", probs)
print(round(code, 5))
```

#### Why It Matters

- Reaches near-entropy compression (fractional bits per symbol).
- Handles non-integer probability models smoothly.
- Adapts dynamically with context models, used in adaptive compressors.
- Basis of:

  * JPEG and H.264 (CABAC variant)
  * BZIP2 (arithmetic / range coder)
  * PPM compressors (Prediction by Partial Matching)

#### Complexity

| Operation                   | Time          | Space  |
| --------------------------- | ------------- | ------ |
| Encoding/decoding           | $O(n)$        | $O(1)$ |
| With adaptive probabilities | $O(n \log m)$ | $O(m)$ |

#### Try It Yourself

1. Use symbols `{A:0.6, B:0.3, C:0.1}` and encode `"ABAC"`.
2. Change the order and see how the encoded number shifts.
3. Implement range coding, a scaled integer form of arithmetic coding.
4. Try adaptive frequency updates for real-time compression.
5. Decode by tracking which subinterval contains the encoded number.

#### A Gentle Proof (Why It Works)

Each symbol's range subdivision corresponds to its probability mass.
Thus, after encoding $n$ symbols, the interval width equals:

$$
\prod_{i=1}^{n} P(s_i)
$$

The number of bits required to represent this interval is approximately:

$$
-\log_2 \left( \prod_{i=1}^{n} P(s_i) \right)
= \sum_{i=1}^{n} -\log_2 P(s_i)
$$

which equals the Shannon information content —
proving arithmetic coding achieves near-entropy optimality.

Arithmetic Coding replaces bits and trees with pure intervals —
compressing not with steps, but with precision itself.

### 654 Shannon–Fano Coding

Shannon–Fano Coding is an early method of entropy-based lossless compression.
It was developed independently by Claude Shannon and Robert Fano before Huffman's algorithm.
While not always optimal, it laid the foundation for modern prefix-free coding and influenced Huffman and arithmetic coding.

#### What Problem Are We Solving?

Given a set of symbols with known probabilities (or frequencies),
we want to assign binary codes such that more frequent symbols get shorter codes —
while ensuring the code remains prefix-free (no code is a prefix of another).

The goal: minimize the expected code length

$$
L = \sum_i p_i \cdot |C_i|
$$

close to the entropy bound
$$
H = -\sum_i p_i \log_2 p_i
$$

#### The Idea

Shannon–Fano coding works by dividing the probability table into two nearly equal halves
and assigning 0s and 1s recursively.

1. Sort all symbols by decreasing probability.
2. Split the list into two parts with total probabilities as equal as possible.
3. Assign `0` to the first group, `1` to the second.
4. Recurse on each group until every symbol has a unique code.

The result is a prefix code, though not always optimal.

#### Example

Symbols and probabilities:

| Symbol | Probability |
| :----: | :---------: |
|    A   |     0.4     |
|    B   |     0.2     |
|    C   |     0.2     |
|    D   |     0.1     |
|    E   |     0.1     |

Step 1. Sort by probability:

A (0.4), B (0.2), C (0.2), D (0.1), E (0.1)

Step 2. Split into equal halves:

| Group | Symbols | Sum | Bit |
| :---: | :-----: | :-: | :-: |
|  Left |   A, B  | 0.6 |  0  |
| Right | C, D, E | 0.4 |  1  |

Step 3. Recurse:

- Left group (A, B): split → A (0.4) | B (0.2)
  → A = `00`, B = `01`
- Right group (C, D, E): split → C (0.2) | D, E (0.2)
  → C = `10`, D = `110`, E = `111`

Final codes:

| Symbol | Probability | Code |
| :----: | :---------: | :--: |
|    A   |     0.4     |  00  |
|    B   |     0.2     |  01  |
|    C   |     0.2     |  10  |
|    D   |     0.1     |  110 |
|    E   |     0.1     |  111 |

Average code length:

$$
L = 0.4(2) + 0.2(2) + 0.2(2) + 0.1(3) + 0.1(3) = 2.2 \text{ bits/symbol}
$$

Entropy:

$$
H = -\sum p_i \log_2 p_i \approx 2.12
$$

Efficiency:

$$
\frac{H}{L} = 0.96
$$

#### Tiny Code (Python)

```python
def shannon_fano(symbols):
    symbols = sorted(symbols.items(), key=lambda x: -x[1])
    codes = {}

    def recurse(sub, prefix=""):
        if len(sub) == 1:
            codes[sub[0][0]] = prefix
            return
        total = sum(p for _, p in sub)
        acc, split = 0, 0
        for i, (_, p) in enumerate(sub):
            acc += p
            if acc >= total / 2:
                split = i + 1
                break
        recurse(sub[:split], prefix + "0")
        recurse(sub[split:], prefix + "1")

    recurse(symbols)
    return codes

probs = {'A': 0.4, 'B': 0.2, 'C': 0.2, 'D': 0.1, 'E': 0.1}
print(shannon_fano(probs))
```

#### Why It Matters

- Historically important, first systematic prefix-coding method.
- Basis for Huffman's later improvement (which guarantees optimality).
- Demonstrates divide-and-balance principle used in tree-based codes.
- Simpler to understand and implement, suitable for educational use.

#### Comparison with Huffman Coding

| Aspect     | Shannon–Fano           | Huffman                            |
| ---------- | ---------------------- | ---------------------------------- |
| Approach   | Top-down splitting     | Bottom-up merging                  |
| Optimality | Not always optimal     | Always optimal                     |
| Code order | Deterministic          | Can vary with equal weights        |
| Usage      | Historical, conceptual | Real-world compression (ZIP, JPEG) |

#### Complexity

| Operation       | Time          | Space  |
| --------------- | ------------- | ------ |
| Sorting         | $O(n \log n)$ | $O(n)$ |
| Code generation | $O(n)$        | $O(n)$ |

#### Try It Yourself

1. Build a Shannon–Fano tree for `{A:7, B:5, C:2, D:1}`.
2. Compare average bit-length with Huffman's result.
3. Verify prefix property (no code is prefix of another).
4. Implement decoding by reversing the code table.

#### A Gentle Proof (Why It Works)

At each recursive split, we ensure that total probability difference between groups is minimal.
This keeps code lengths roughly proportional to symbol probabilities:

$$
|C_i| \approx \lceil -\log_2 p_i \rceil
$$

Thus, Shannon–Fano always produces a prefix-free code whose length
is close (but not guaranteed equal) to the optimal entropy bound.

Shannon–Fano Coding was the first real step from probability to code —
a balanced yet imperfect bridge between information theory and compression practice.

### 655 Run-Length Encoding (RLE)

Run-Length Encoding (RLE) is one of the simplest lossless compression techniques.
It replaces consecutive repeating symbols, called *runs*, with a count and the symbol itself.
RLE is ideal when data contains long sequences of the same value, such as in images, bitmaps, or text with whitespace.

#### What Problem Are We Solving?

Uncompressed data often has redundancy in the form of repeated symbols:

```
AAAAABBBBCCCCCCDD
```

Instead of storing each symbol, we can store *how many times* it repeats.

Encoded form:

```
(5, A)(4, B)(6, C)(2, D)
```

which saves space whenever runs are long relative to the alphabet size.

#### Core Idea

Compress data by representing runs of identical symbols as pairs:

$$
(\text{symbol}, \text{count})
$$

For example:

|      Input      |      Encoded      |
| :-------------: | :---------------: |
| `AAAAABBBCCDAA` |    `5A3B2C1D2A`   |
|   `0001111000`  | `(0,3)(1,4)(0,3)` |
|      `AAAB`     |       `3A1B`      |

Decoding simply reverses the process, expand each pair into repeated symbols.

#### Algorithm Steps

1. Initialize `count = 1`.
2. Iterate through the sequence:

   * If the next symbol is the same, increment `count`.
   * If it changes, output `(symbol, count)` and reset.
3. After the loop, output the final run.

#### Tiny Code (Python)

```python
def rle_encode(s):
    if not s:
        return ""
    result = []
    count = 1
    for i in range(1, len(s)):
        if s[i] == s[i - 1]:
            count += 1
        else:
            result.append(f"{count}{s[i-1]}")
            count = 1
    result.append(f"{count}{s[-1]}")
    return "".join(result)

def rle_decode(encoded):
    import re
    parts = re.findall(r'(\d+)(\D)', encoded)
    return "".join(sym * int(cnt) for cnt, sym in parts)

text = "AAAAABBBCCDAA"
encoded = rle_encode(text)
decoded = rle_decode(encoded)
print(encoded, decoded)
```

Output:

```
5A3B2C1D2A AAAAABBBCCDAA
```

#### Example Walkthrough

Input:
`AAAABBCCCCD`

| Step | Current Symbol | Count | Encoded Output |
| ---- | -------------- | ----- | -------------- |
| A    | 4              | →     | `4A`           |
| B    | 2              | →     | `2B`           |
| C    | 4              | →     | `4C`           |
| D    | 1              | →     | `1D`           |

Final encoded string:
`4A2B4C1D`

#### Why It Matters

- Simplicity: requires no statistical model or dictionary.
- Efficiency: great for images, faxes, DNA sequences, or repeated characters.
- Building block for more advanced compression schemes:

  * TIFF, BMP, PCX image formats
  * DEFLATE preprocessing (in zlib, PNG)
  * Fax Group 3/4 standards

#### When It Works Well

| Data Type        | Example                          | Compression Benefit |
| ---------------- | -------------------------------- | ------------------- |
| Monochrome image | large white/black regions        | High                |
| Plain text       | spaces, tabs                     | Moderate            |
| Binary data      | many zeros (e.g. sparse bitmaps) | High                |
| Random data      | no repetition                    | None or negative    |

#### Complexity

| Operation | Time   | Space  |
| --------- | ------ | ------ |
| Encoding  | $O(n)$ | $O(1)$ |
| Decoding  | $O(n)$ | $O(1)$ |

#### Try It Yourself

1. Encode and decode `"AAABBBAAACC"`.
2. Measure compression ratio = $\text{compressed length} / \text{original length}$.
3. Try RLE on a text paragraph, does it help?
4. Modify code to use bytes `(count, symbol)` instead of text.
5. Combine RLE with Huffman coding, compress the RLE output.

#### A Gentle Proof (Why It Works)

If $r$ is the average run length,
then RLE compresses from $n$ characters to approximately $2n / r$ symbols.
Compression occurs when $r > 2$ on average.

For highly repetitive data ($r \gg 2$),
the gain approaches:

$$
\text{compression ratio} \approx \frac{2}{r}
$$

Run-Length Encoding turns repetition into economy —
it sees not each symbol, but the rhythm of their persistence.

### 656 LZ77 (Sliding-Window Compression)

LZ77 is a foundational compression algorithm invented by Abraham Lempel and Jacob Ziv in 1977.
It introduced the idea of *sliding-window compression*, where repeated patterns are replaced by backward references.
This concept underlies many modern compressors, including DEFLATE (ZIP, gzip), PNG, and Zstandard.

#### What Problem Are We Solving?

Redundancy in data often appears as repeated substrings rather than long identical runs.
For example:

```
ABABABA
```

contains overlapping repetitions of `"ABA"`.
RLE can't handle this efficiently, but LZ77 can, by *referencing earlier occurrences* instead of repeating them.

#### Key Idea

Maintain a sliding window of recently seen data.
When a new substring repeats part of that window, replace it with a (distance, length, next symbol) triple.

Each triple means:

> "Go back `distance` characters, copy `length` characters, then output `next`."

#### Example

Input:

```
A B A B A B A
```

Step-by-step compression (window shown progressively):

| Step | Current Window | Next Symbol | Match Found         | Output    |
| ---- | -------------- | ----------- | ------------------- | --------- |
| 1    |,              | A           | none                | (0, 0, A) |
| 2    | A              | B           | none                | (0, 0, B) |
| 3    | AB             | A           | "A" at distance 2   | (2, 1, B) |
| 4    | ABA            | B           | "AB" at distance 2  | (2, 2, A) |
| 5    | ABAB           | A           | "ABA" at distance 2 | (2, 3, —) |

Final encoded sequence:

```
(0,0,A) (0,0,B) (2,1,B) (2,2,A)
```

Decoded output:

```
ABABABA
```

#### How It Works

1. Initialize an empty search buffer (past) and a lookahead buffer (future).
2. For the next symbol(s) in the lookahead:

   * Find the longest match in the search buffer.
   * Emit a triple `(distance, length, next_char)`.
   * Slide the window forward by `length + 1`.
3. Continue until the end of input.

The search buffer allows backward references,
and the lookahead buffer limits how far ahead we match.

#### Tiny Code (Python, Simplified)

```python
def lz77_compress(data, window_size=16):
    i, output = 0, []
    while i < len(data):
        match = (0, 0, data[i])
        for dist in range(1, min(i, window_size) + 1):
            length = 0
            while (i + length < len(data) and
                   data[i - dist + length] == data[i + length]):
                length += 1
            if length > match[1]:
                next_char = data[i + length] if i + length < len(data) else ''
                match = (dist, length, next_char)
        output.append(match)
        i += match[1] + 1
    return output
```

Example:

```python
text = "ABABABA"
print(lz77_compress(text))
# [(0,0,'A'), (0,0,'B'), (2,1,'B'), (2,2,'A')]
```

#### Why It Matters

- Foundation of DEFLATE (ZIP, gzip, PNG).
- Enables powerful dictionary-based compression.
- Self-referential: output can describe future data.
- Works well on structured text, binaries, and repetitive data.

Modern variants (LZSS, LZW, LZMA) extend or refine this model.

#### Compression Format

A typical LZ77 token:

| Field       | Description                  |
| ----------- | ---------------------------- |
| Distance    | How far back to look (bytes) |
| Length      | How many bytes to copy       |
| Next Symbol | Literal following the match  |

Example:
`(distance=4, length=3, next='A')`
→ "copy 3 bytes from 4 positions back, then write A".

#### Complexity

| Operation | Time                       | Space  |
| --------- | -------------------------- | ------ |
| Encoding  | $O(n w)$ (window size $w$) | $O(w)$ |
| Decoding  | $O(n)$                     | $O(w)$ |

Optimized implementations use hash tables or tries to reduce search cost.

#### Try It Yourself

1. Encode `"BANANA_BANDANA"`.
2. Experiment with different window sizes.
3. Visualize backward pointers as arrows between symbols.
4. Implement LZSS, skip storing `next_char` when unnecessary.
5. Combine with Huffman coding for DEFLATE-like compression.

#### A Gentle Proof (Why It Works)

Each emitted triple covers a non-overlapping substring of input.
The reconstruction is unambiguous because each `(distance, length)` refers only to already-decoded data.
Hence, LZ77 forms a self-consistent compression system with guaranteed lossless recovery.

Compression ratio improves with longer matching substrings:
$$
R \approx \frac{n}{n - \sum_i \text{length}_i}
$$

The more redundancy, the higher the compression.

LZ77 taught machines to *look back to move forward* —
a model of memory and reuse that became the heartbeat of modern compression.

### 657 LZ78 (Dictionary Building)

LZ78, introduced by Abraham Lempel and Jacob Ziv in 1978, is the successor to LZ77.
While LZ77 compresses using a *sliding window*, LZ78 instead builds an explicit dictionary of substrings encountered so far.
Each new phrase is stored once, and later references point directly to dictionary entries.

This shift from window-based to dictionary-based compression paved the way for algorithms like LZW, GIF, and TIFF.

#### What Problem Are We Solving?

LZ77 reuses a *moving* window of previous text, which must be searched for every match.
LZ78 improves efficiency by storing known substrings in a dictionary that grows dynamically.
Instead of scanning backward, the encoder refers to dictionary entries directly by index.

This reduces search time and simplifies decoding, at the cost of managing a dictionary.

#### The Core Idea

Each output token encodes:

$$
(\text{index}, \text{next symbol})
$$

where:

- `index` points to the longest prefix already in the dictionary.
- `next symbol` is the new character that extends it.

The pair defines a new entry added to the dictionary:
$$
\text{dict}[k] = \text{dict}[\text{index}] + \text{next symbol}
$$

#### Example

Let's encode the string:

```
ABAABABAABAB
```

Step 1. Initialize an empty dictionary.

| Step | Input | Longest Prefix | Index | Next Symbol | Output | New Entry |
| ---- | ----- | -------------- | ----- | ----------- | ------ | --------- |
| 1    | A     | ""             | 0     | A           | (0, A) | 1: A      |
| 2    | B     | ""             | 0     | B           | (0, B) | 2: B      |
| 3    | A     | A              | 1     | B           | (1, B) | 3: AB     |
| 4    | A     | A              | 1     | A           | (1, A) | 4: AA     |
| 5    | B     | AB             | 3     | A           | (3, A) | 5: ABA    |
| 6    | A     | ABA            | 5     | B           | (5, B) | 6: ABAB   |

Final output:

```
(0,A) (0,B) (1,B) (1,A) (3,A) (5,B)
```

Dictionary at the end:

| Index | Entry |
| :---: | :---: |
|   1   |   A   |
|   2   |   B   |
|   3   |   AB  |
|   4   |   AA  |
|   5   |  ABA  |
|   6   |  ABAB |

Decoded message is identical: `ABAABABAABAB`.

#### Tiny Code (Python)

```python
def lz78_compress(s):
    dictionary = {}
    output = []
    current = ""
    next_index = 1

    for c in s:
        if current + c in dictionary:
            current += c
        else:
            idx = dictionary.get(current, 0)
            output.append((idx, c))
            dictionary[current + c] = next_index
            next_index += 1
            current = ""
    if current:
        output.append((dictionary[current], ""))
    return output

text = "ABAABABAABAB"
print(lz78_compress(text))
```

Output:

```
$$(0, 'A'), (0, 'B'), (1, 'B'), (1, 'A'), (3, 'A'), (5, 'B')]
```

#### Decoding Process

Given encoded pairs `(index, symbol)`:

1. Initialize dictionary with `dict[0] = ""`.
2. For each pair:

   * Output `dict[index] + symbol`.
   * Add it as a new dictionary entry.

Decoding reconstructs the text deterministically.

#### Why It Matters

- Introduced explicit phrase dictionary, reusable across blocks.
- No backward scanning, faster than LZ77 for large data.
- Basis of LZW, which removes explicit symbol output and adds automatic dictionary management.
- Used in:

  * UNIX `compress`
  * GIF and TIFF images
  * Old modem protocols (V.42bis)

#### Comparison

| Feature    | LZ77                     | LZ78                               |
| ---------- | ------------------------ | ---------------------------------- |
| Model      | Sliding window           | Explicit dictionary                |
| Output     | (distance, length, next) | (index, symbol)                    |
| Dictionary | Implicit (in stream)     | Explicit (stored entries)          |
| Decoding   | Immediate                | Requires dictionary reconstruction |
| Successors | DEFLATE, LZMA            | LZW, LZMW                          |

#### Complexity

| Operation | Time           | Space  |
| --------- | -------------- | ------ |
| Encoding  | $O(n)$ average | $O(n)$ |
| Decoding  | $O(n)$         | $O(n)$ |

Memory usage can grow with the number of unique substrings,
so implementations often reset the dictionary when full.

#### Try It Yourself

1. Encode and decode `"TOBEORNOTTOBEORTOBEORNOT"`.
2. Print the dictionary evolution.
3. Compare output size with LZ77.
4. Implement dictionary reset at a fixed size (e.g. 4096 entries).
5. Extend it to LZW by reusing indices without explicit characters.

#### A Gentle Proof (Why It Works)

Each emitted pair corresponds to a new phrase not seen before.
Thus, every substring in the input can be expressed as a sequence of dictionary references.
Because each new phrase extends a previous one by a single symbol,
the dictionary is *prefix-closed*, ensuring unique reconstruction.

Compression efficiency improves with data redundancy:

$$
R \approx \frac{\text{\#pairs} \times (\log_2 N + \text{char bits})}{\text{input bits}}
$$

and approaches the entropy limit for large $N$.

LZ78 taught compression to remember patterns as words,
turning the sliding window of memory into a growing vocabulary of meaning.

### 658 LZW (Lempel–Ziv–Welch)

LZW, introduced by Terry Welch in 1984, is an optimized form of LZ78.
It removes the need to transmit the *extra symbol* in each pair and instead relies on the dictionary itself to infer the next character.
This small change made LZW faster, simpler, and perfect for real-world use, it powered GIF, TIFF, and UNIX compress.

#### What Problem Are We Solving?

LZ78 produces pairs of the form `(index, next_symbol)`, which adds extra data.
LZW eliminates that redundancy:
instead of sending the literal symbol, it transmits only dictionary indices, because the dictionary can predict the next symbol from previous context.

In short, LZW says:

> "If both encoder and decoder build the same dictionary in the same way, we only need to send the indices."

#### Core Idea

1. Initialize the dictionary with all single characters (e.g., ASCII 0–255).
2. Read the input symbol by symbol:

   * Keep the longest string `w` found in the dictionary.
   * When `w + c` (current string + next character) is not found:

     * Output the code for `w`.
     * Add `w + c` to the dictionary.
     * Set `w = c`.
3. When finished, output the code for the last `w`.

Both encoder and decoder grow the dictionary identically,
ensuring deterministic reconstruction.

#### Example

Let's encode the string:

```
TOBEORNOTTOBEORTOBEORNOT
```

Step 1. Initialize the dictionary:

| Code  | Entry                |
| ----- | -------------------- |
| 0–255 | All ASCII characters |

Step 2. Encode step by step:

| Step | w  | c | Output       | New Entry           |
| ---- | -- | - | ------------ | ------------------- |
| 1    | T  | O | code(T) = 84 | TO                  |
| 2    | O  | B | code(O) = 79 | OB                  |
| 3    | B  | E | code(B) = 66 | BE                  |
| 4    | E  | O | code(E) = 69 | EO                  |
| 5    | O  | R | code(O) = 79 | OR                  |
| 6    | R  | N | code(R) = 82 | RN                  |
| 7    | N  | O | code(N) = 78 | NO                  |
| 8    | O  | T | code(O) = 79 | OT                  |
| 9    | T  | O | code(T) = 84 | TO (already exists) |
| 10   | TO | B | code(TO)     | TOB                 |
| 11   | B  | E | code(B) = 66 | BE (already exists) |
| 12   | E  | O | code(E) = 69 | EO (already exists) |

and so on…

Output sequence (partial):

```
84, 79, 66, 69, 79, 82, 78, 79, 84, 256, 258, 260, ...
```

Each number represents a code pointing to a dictionary entry.

#### Tiny Code (Python)

```python
def lzw_compress(data):
    # Initialize dictionary with single characters
    dict_size = 256
    dictionary = {chr(i): i for i in range(dict_size)}

    w = ""
    result = []
    for c in data:
        wc = w + c
        if wc in dictionary:
            w = wc
        else:
            result.append(dictionary[w])
            dictionary[wc] = dict_size
            dict_size += 1
            w = c
    if w:
        result.append(dictionary[w])
    return result
```

Example:

```python
text = "TOBEORNOTTOBEORTOBEORNOT"
codes = lzw_compress(text)
print(codes)
```

#### Decoding Process

Decoding reconstructs the text using the same logic in reverse:

1. Initialize dictionary with single characters.
2. Read the first code, output its character.
3. For each next code:

   * If it exists in the dictionary → output it.
   * If not → output `w + first_char(w)` (special case for unseen code).
   * Add `w + first_char(current)` to the dictionary.
   * Update `w`.

#### Why It Matters

- Dictionary-based efficiency: compact and fast.
- No need to send dictionary or symbols.
- Simple to implement in both hardware and software.
- Used in real-world formats:

  * GIF
  * TIFF
  * UNIX compress
  * PostScript / PDF

#### Comparison with LZ78

| Feature    | LZ78             | LZW                  |
| ---------- | ---------------- | -------------------- |
| Output     | (index, symbol)  | index only           |
| Dictionary | Explicit entries | Grows implicitly     |
| Efficiency | Slightly less    | Better on real data  |
| Used in    | Research         | Real-world standards |

#### Complexity

| Operation | Time           | Space  |
| --------- | -------------- | ------ |
| Encoding  | $O(n)$ average | $O(n)$ |
| Decoding  | $O(n)$         | $O(n)$ |

Compression ratio improves with repetitive phrases and longer dictionaries.

#### Try It Yourself

1. Compress and decompress `"TOBEORNOTTOBEORTOBEORNOT"`.
2. Print dictionary growth step by step.
3. Try it on a text paragraph, notice the repeating words' compression.
4. Modify for lowercase ASCII only (size 26).
5. Experiment with dictionary reset after 4096 entries (as in GIF).

#### A Gentle Proof (Why It Works)

Since both encoder and decoder start with identical dictionaries and add entries in the same sequence,
the same index sequence leads to the same reconstruction.
The dictionary grows by prefix extension, each new entry is a previous entry plus one symbol, ensuring deterministic decoding.

For long input of entropy $H$,
the average code length approaches $H + 1$ bits per symbol,
making LZW asymptotically optimal for stationary sources.

LZW transformed compression into pure memory and inference —
a dance of codes where meaning is built, shared, and never transmitted twice.

### 659 Burrows–Wheeler Transform (BWT)

The Burrows–Wheeler Transform (BWT), invented by Michael Burrows and David Wheeler in 1994, is a landmark in lossless compression.
Unlike previous schemes that directly encode data, BWT rearranges it, transforming the input into a form that's easier for compressors like RLE or Huffman to exploit.

The beauty of BWT is that it's reversible and structure-preserving: it clusters similar characters together, amplifying patterns before actual encoding.

#### What Problem Are We Solving?

Traditional compressors operate locally, they detect patterns over small windows or dictionaries.
However, when similar symbols are far apart, they fail to exploit global structure efficiently.

BWT solves this by sorting all rotations of a string, then extracting the last column, effectively grouping similar contexts together.
This rearrangement doesn't reduce entropy but makes it *easier* to compress with simple algorithms like RLE or Huffman.

#### The Key Idea

Given a string $S$ of length $n$, append a special end marker `$` that is lexicographically smaller than any character.
Form all cyclic rotations of $S$ and sort them lexicographically.
The BWT output is the last column of this sorted matrix.

The transformation also records the index of the original string within the sorted list (needed for reversal).

#### Example

Input:

```
BANANA$
```

Step 1. Generate all rotations:

| Rotation |  String |
| :------: | :-----: |
|     0    | BANANA$ |
|     1    | ANANA$B |
|     2    | NANA$BA |
|     3    | ANA$BAN |
|     4    | NA$BANA |
|     5    | A$BANAN |
|     6    | $BANANA |

Step 2. Sort all rotations lexicographically:

| Sorted Index | Rotation | Last Character |
| :----------: | :------: | :------------: |
|       0      |  $BANANA |        A       |
|       1      |  A$BANAN |        N       |
|       2      |  ANA$BAN |        N       |
|       3      |  ANANA$B |        A       |
|       4      |  BANANA$ |        $       |
|       5      |  NA$BANA |        A       |
|       6      |  NANA$BA |        A       |

Step 3. Extract last column (L):

```
L = ANNA$AA
```

and record the index of the original string (`BANANA$`) → row 4.

So,
BWT output = (L = ANNA$AA, index = 4)

#### Decoding (Inverse BWT)

To reverse the transform, reconstruct the table column by column:

1. Start with the last column `L`.
2. Repeatedly prepend `L` to existing rows and sort after each iteration.
3. After `n` iterations, the row containing `$` at the end is the original string.

For `L = ANNA$AA`, index = 4:

| Iteration | Table (sorted each round)  |
| --------- | -------------------------- |
| 1         | A, N, N, A, $, A, A        |
| 2         | A$, AN, AN, NA, N$, AA, AA |
| 3         | ANA, NAN, NNA, A$B, etc.   |

…eventually yields back `BANANA$`.

Efficient decoding skips building the full matrix using the LF-mapping relation:
$$
\text{next}(i) = C[L[i]] + \text{rank}(L, i, L[i])
$$
where $C[c]$ is the count of characters lexicographically smaller than $c$.

#### Tiny Code (Python)

```python
def bwt_transform(s):
    s = s + "$"
    rotations = [s[i:] + s[:i] for i in range(len(s))]
    rotations.sort()
    last_column = ''.join(row[-1] for row in rotations)
    index = rotations.index(s)
    return last_column, index

def bwt_inverse(last_column, index):
    n = len(last_column)
    table = [""] * n
    for _ in range(n):
        table = sorted([last_column[i] + table[i] for i in range(n)])
    return table[index].rstrip("$")

# Example
L, idx = bwt_transform("BANANA")
print(L, idx)
print(bwt_inverse(L, idx))
```

Output:

```
ANNA$AA 4
BANANA
```

#### Why It Matters

- Structure amplifier: Groups identical symbols by context, improving performance of:

  * Run-Length Encoding (RLE)
  * Move-To-Front (MTF)
  * Huffman or Arithmetic Coding
- Foundation of modern compressors:

  * bzip2
  * zstd's preprocessing
  * FM-Index in bioinformatics
- Enables searchable compressed text via suffix arrays and ranks.

#### Complexity

| Operation | Time                     | Space  |
| --------- | ------------------------ | ------ |
| Transform | $O(n \log n)$            | $O(n)$ |
| Inverse   | $O(n)$ (with LF mapping) | $O(n)$ |

#### Try It Yourself

1. Apply BWT to `"MISSISSIPPI"`.
2. Combine with Run-Length Encoding.
3. Test compression ratio.
4. Compare result before and after applying BWT.
5. Visualize rotation matrix for small words.

#### A Gentle Proof (Why It Works)

BWT doesn't compress; it permutes the data so that symbols with similar contexts appear near each other.
Since most texts are locally predictable, this reduces entropy *after transformation*:

$$
H(\text{BWT}(S)) = H(S)
$$

but the transformed data exhibits longer runs and simpler local structure, which compressors like Huffman can exploit more effectively.

The Burrows–Wheeler Transform is compression's quiet magician —
it doesn't shrink data itself but rearranges it into order,
turning chaos into compressible calm.

### 660 Move-to-Front (MTF) Encoding

Move-to-Front (MTF) is a simple yet powerful transformation used in combination with the Burrows–Wheeler Transform (BWT).
Its purpose is to turn localized symbol repetitions, produced by BWT, into sequences of small integers, which are highly compressible by Run-Length Encoding (RLE) or Huffman coding.

#### What Problem Are We Solving?

After applying BWT, the transformed string contains clusters of identical symbols.
To take advantage of that, we want to represent symbols by how *recently* they appeared.

If we maintain a list of all possible symbols and move each accessed symbol to the front, then frequent symbols will have small indices, producing many zeros and ones, ideal for entropy coding.

#### Key Idea

Maintain an ordered list of all symbols (the "alphabet").
For each symbol in the input:

1. Output its position index in the current list.
2. Move that symbol to the front of the list.

This captures locality, recently used symbols appear earlier and thus get smaller indices.

#### Example

Input sequence:

```
banana
```

Alphabet (initial): `[a, b, n]`

| Step | Symbol | List Before | Index | List After |
| ---- | ------ | ----------- | ----- | ---------- |
| 1    | b      | [a, b, n]   | 1     | [b, a, n]  |
| 2    | a      | [b, a, n]   | 1     | [a, b, n]  |
| 3    | n      | [a, b, n]   | 2     | [n, a, b]  |
| 4    | a      | [n, a, b]   | 1     | [a, n, b]  |
| 5    | n      | [a, n, b]   | 1     | [n, a, b]  |
| 6    | a      | [n, a, b]   | 1     | [a, n, b]  |

Encoded output:

```
$$1, 1, 2, 1, 1, 1]
```

Notice how frequent letters ("a", "n") are represented with small numbers.

#### Decoding Process

Given the encoded indices and initial alphabet:

1. For each index, pick the symbol at that position.
2. Output it and move it to the front of the list.

The process is perfectly reversible because both encoder and decoder perform identical list updates.

#### Tiny Code (Python)

```python
def mtf_encode(data, alphabet):
    symbols = list(alphabet)
    result = []
    for c in data:
        index = symbols.index(c)
        result.append(index)
        symbols.insert(0, symbols.pop(index))
    return result

def mtf_decode(indices, alphabet):
    symbols = list(alphabet)
    result = []
    for i in indices:
        c = symbols[i]
        result.append(c)
        symbols.insert(0, symbols.pop(i))
    return ''.join(result)

alphabet = list("abn")
encoded = mtf_encode("banana", alphabet)
decoded = mtf_decode(encoded, list("abn"))
print(encoded, decoded)
```

Output:

```
$$1, 1, 2, 1, 1, 1] banana
```

#### Why It Matters

- Pairs beautifully with BWT:

  * BWT groups similar symbols together.
  * MTF converts those groups into small indices.
  * RLE or Huffman coding then compresses the small numbers efficiently.
- Used in:

  * bzip2
  * block-sorting compressors
  * text indexing systems (FM-Index)

MTF itself does not compress, it transforms data into a shape that's easier to compress.

#### Intuitive Example (After BWT)

Suppose BWT output is:

```
AAAAABBBBCCCC
```

MTF output becomes something like:

```
0,0,0,0,1,1,1,1,2,2,2,2
```

Now RLE or Huffman can compress these tiny integers extremely efficiently.

#### Complexity

| Operation | Time       | Space  |            |     |        |    |
| --------- | ---------- | ------ | ---------- | --- | ------ | -- |
| Encoding  | $O(n \cdot | \Sigma | )$ (naive) | $O( | \Sigma | )$ |
| Decoding  | $O(n \cdot | \Sigma | )$         | $O( | \Sigma | )$ |

Optimized implementations use linked lists or direct-mapped arrays to reduce per-symbol cost to near $O(n)$.

#### Try It Yourself

1. Apply MTF to the BWT of `"BANANA$"` (→ `ANNA$AA`).
2. Encode result: you'll get many low numbers.
3. Follow with RLE + Huffman to achieve full compression.
4. Compare compressed size before and after MTF.
5. Change the alphabet order, see how it changes results.

#### A Gentle Proof (Why It Works)

Let $f(c)$ be the frequency of symbol $c$.
After BWT, identical symbols tend to appear consecutively, so their first appearance moves them to the front of the list, and subsequent appearances yield index `0`.
Thus, the expected output index is proportional to the distance between identical symbols, making MTF effective for locality-rich data.

Entropy after MTF tends to be lower than before:

$$
H_\text{MTF}(X) \leq H(X)
$$

for structured inputs (not necessarily for random data).

Move-to-Front Encoding captures the *recency of symbols* —
a minimalist memory that keeps what's fresh right in front,
ready to make compression beautifully simple.

# Section 67. Cryptographic Hashes and Checksum

### 661 Rolling Hash

The Rolling Hash is a clever hashing technique that allows you to compute hash values of overlapping substrings efficiently.
It forms the backbone of algorithms like Rabin–Karp, Rabin fingerprinting, and string deduplication systems used in text search and data synchronization.

#### What Problem Are We Solving?

In many problems, you need to compare substrings or sliding windows efficiently.
Naively recomputing the hash of every substring takes $O(m)$ per step, leading to $O(nm)$ overall, too slow for large inputs.

A rolling hash lets you update the hash in constant time when the window slides by one position, reducing total time to $O(n)$.

#### The Core Idea

Represent a string as a number in a chosen base and compute its value modulo a large prime:

$$
H(s_0 s_1 \dots s_{m-1}) = (s_0 b^{m-1} + s_1 b^{m-2} + \dots + s_{m-1}) \bmod M
$$

When the window moves by one character (drop `old` and add `new`),
the hash can be updated efficiently:

$$
H_{\text{new}} = (b(H_{\text{old}} - s_0 b^{m-1}) + s_m) \bmod M
$$

This means we can slide across the text and compute new hashes in $O(1)$ time.

#### Example

Consider the string `ABCD` with base $b = 256$ and modulus $M = 101$.

Compute:

$$
H("ABC") = (65 \times 256^2 + 66 \times 256 + 67) \bmod 101
$$

When the window slides from `"ABC"` to `"BCD"`:

$$
H("BCD") = (b(H("ABC") - 65 \times 256^2) + 68) \bmod 101
$$

This efficiently removes `'A'` and adds `'D'`.

#### Tiny Code (Python)

```python
def rolling_hash(s, base=256, mod=101):
    h = 0
    for c in s:
        h = (h * base + ord(c)) % mod
    return h

def update_hash(old_hash, left_char, right_char, power, base=256, mod=101):
    # remove leftmost char, add rightmost char
    old_hash = (old_hash - ord(left_char) * power) % mod
    old_hash = (old_hash * base + ord(right_char)) % mod
    return old_hash
```

Usage:

```python
text = "ABCD"
m = 3
mod = 101
base = 256
power = pow(base, m-1, mod)

h = rolling_hash(text[:m], base, mod)
for i in range(1, len(text) - m + 1):
    h = update_hash(h, text[i-1], text[i+m-1], power, base, mod)
    print(h)
```

#### Why It Matters

- Constant-time sliding: hash updates in $O(1)$
- Ideal for substring search: used in Rabin–Karp
- Used in deduplication systems (rsync, git)
- Foundation of polynomial hashing and rolling checksums (Adler-32, Rabin fingerprinting)

Rolling hashes balance speed and accuracy, with large enough modulus and base, collisions are rare.

#### Collision and Modulus

Collisions happen when two different substrings share the same hash.
We minimize them by:

1. Using a large prime modulus $M$, often near $2^{61} - 1$.
2. Using double hashing with two different $(b, M)$ pairs.
3. Occasionally verifying matches by direct string comparison.

#### Complexity

| Operation          | Time   | Space  |
| ------------------ | ------ | ------ |
| Compute first hash | $O(m)$ | $O(1)$ |
| Slide update       | $O(1)$ | $O(1)$ |
| Total over text    | $O(n)$ | $O(1)$ |

#### Try It Yourself

1. Compute rolling hashes for all substrings of `"BANANA"` of length 3.
2. Use modulus $M = 101$, base $b = 256$.
3. Compare collision rate for small vs large $M$.
4. Modify the code to use two moduli for double hashing.
5. Implement Rabin–Karp substring search using this rolling hash.

#### A Gentle Proof (Why It Works)

When we slide from window $[i, i+m-1]$ to $[i+1, i+m]$,
the contribution of the dropped character is known in advance,
so we can adjust the hash without recomputing everything.

Since modulo arithmetic is linear:

$$
H(s_1 \dots s_m) = (b(H(s_0 \dots s_{m-1}) - s_0 b^{m-1}) + s_m) \bmod M
$$

This property ensures correctness while preserving constant-time updates.

Rolling Hash is the quiet workhorse of modern text processing —
it doesn't look for patterns directly,
it summarizes, slides, and lets arithmetic find the matches.

### 662 CRC32 (Cyclic Redundancy Check)

The Cyclic Redundancy Check (CRC) is a checksum algorithm that detects errors in digital data.
It's widely used in networks, storage, and file formats (like ZIP, PNG, Ethernet) to ensure that data was transmitted or stored correctly.

CRC32 is a common 32-bit variant, fast, simple, and highly reliable for detecting random errors.

#### What Problem Are We Solving?

When data is transmitted over a channel or written to disk, bits can flip due to noise or corruption.
We need a way to detect whether received data is identical to what was sent.

A simple checksum (like summing bytes) can miss many errors.
CRC treats data as a polynomial over GF(2) and performs division by a fixed generator polynomial, producing a remainder that acts as a strong integrity check.

#### The Core Idea

Think of the data as a binary polynomial:

$$
M(x) = m_0x^{n-1} + m_1x^{n-2} + \dots + m_{n-1}
$$

Choose a generator polynomial $G(x)$ of degree $r$ (for CRC32, $r=32$).
We append $r$ zeros to the message and divide by $G(x)$ using modulo-2 arithmetic:

$$
R(x) = (M(x) \cdot x^r) \bmod G(x)
$$

Then, transmit:

$$
T(x) = M(x) \cdot x^r + R(x)
$$

At the receiver, the same division is performed.
If the remainder is zero, the message is assumed valid.

All operations use XOR instead of subtraction since arithmetic is in GF(2).

#### Example (Simple CRC)

Let $G(x) = x^3 + x + 1$ (binary 1011).
Message: `1101`.

1. Append three zeros → `1101000`
2. Divide `1101000` by `1011` (mod-2).
3. The remainder is `010`.
4. Transmit `1101010`.

At the receiver, dividing by the same polynomial gives remainder `0`, confirming integrity.

#### CRC32 Polynomial

CRC32 uses the polynomial:

$$
G(x) = x^{32} + x^{26} + x^{23} + x^{22} + x^{16} + x^{12} + x^{11} + x^{10} + x^8 + x^7 + x^5 + x^4 + x^2 + x + 1
$$

In hexadecimal: `0x04C11DB7`.

#### Tiny Code (C)

```c
#include <stdint.h>
#include <stdio.h>

uint32_t crc32(uint8_t *data, size_t len) {
    uint32_t crc = 0xFFFFFFFF;
    for (size_t i = 0; i < len; i++) {
        crc ^= data[i];
        for (int j = 0; j < 8; j++) {
            if (crc & 1)
                crc = (crc >> 1) ^ 0xEDB88320;
            else
                crc >>= 1;
        }
    }
    return crc ^ 0xFFFFFFFF;
}

int main() {
    uint8_t msg[] = "HELLO";
    printf("CRC32: %08X\n", crc32(msg, 5));
}
```

Output:

```
CRC32: 3610A686
```

#### Tiny Code (Python)

```python
def crc32(data: bytes):
    crc = 0xFFFFFFFF
    for b in data:
        crc ^= b
        for _ in range(8):
            if crc & 1:
                crc = (crc >> 1) ^ 0xEDB88320
            else:
                crc >>= 1
    return crc ^ 0xFFFFFFFF

print(hex(crc32(b"HELLO")))
```

Output:

```
0x3610a686
```

#### Why It Matters

- Error detection, not correction
- Detects:

  * All single-bit and double-bit errors
  * Odd numbers of bit errors
  * Burst errors shorter than 32 bits
- Used in:

  * Ethernet, ZIP, PNG, gzip, TCP/IP checksums
  * Filesystems and data transmission protocols

CRC is fast, hardware-friendly, and mathematically grounded in polynomial division.

#### Complexity

| Operation    | Time                                       | Space                     |
| ------------ | ------------------------------------------ | ------------------------- |
| Encoding     | $O(nr)$ (bitwise) or $O(n)$ (table-driven) | $O(1)$ or $O(256)$ lookup |
| Verification | $O(n)$                                     | $O(1)$                    |

Table-based CRC implementations can compute checksums for megabytes of data per second.

#### Try It Yourself

1. Compute CRC3 for message `1101` using generator `1011`.
2. Compare remainders for small vs large polynomials.
3. Implement CRC16 and CRC32 in Python.
4. Flip one bit, verify CRC detects the error.
5. Visualize polynomial division with XOR operations.

#### A Gentle Proof (Why It Works)

Because CRC uses polynomial division over GF(2),
each error pattern $E(x)$ has a unique remainder modulo $G(x)$.
If $G(x)$ is chosen so that no low-weight $E(x)$ divides it,
then all small errors are guaranteed to change the remainder.

CRC32's generator polynomial is carefully designed to catch the most likely error types in real communication systems.

CRC32 is the silent guardian of data integrity —
fast enough for every packet,
reliable enough for every disk,
and simple enough to live in a few lines of C.

### 663 Adler-32 Checksum

Adler-32 is a simple and efficient checksum algorithm designed as a lightweight alternative to CRC32.
It combines speed and reasonable error detection, making it popular in applications like zlib, PNG, and other data compression libraries.

#### What Problem Are We Solving?

CRC32 provides strong error detection but involves bitwise operations and polynomial arithmetic, which can be slower on some systems.
For lightweight applications (like verifying compressed data), we need a faster, easy-to-implement checksum that still detects common transmission errors.

Adler-32 achieves this using modular arithmetic over integers rather than polynomials.

#### The Core Idea

Adler-32 maintains two running sums, one for data bytes and one for the cumulative total.

Let the message be $m_1, m_2, \dots, m_n$, each an unsigned byte.

Compute two values:

$$
A = 1 + \sum_{i=1}^{n} m_i \pmod{65521}
$$

$$
B = 0 + \sum_{i=1}^{n} A_i \pmod{65521}
$$

The final checksum is:

$$
\text{Adler-32}(M) = (B \ll 16) + A
$$

Here, 65521 is the largest prime smaller than $2^{16}$, chosen for good modular behavior.

#### Example

Message: `"Hi"`

Characters:

```
'H' = 72
'i' = 105
```

Compute:

| Step     | A (mod 65521) | B (mod 65521) |
| -------- | ------------- | ------------- |
| Init     | 1             | 0             |
| +H (72)  | 73            | 73            |
| +i (105) | 178           | 251           |

Then:

$$
\text{Checksum} = (251 \ll 16) + 178 = 16449842
$$

In hexadecimal:

```
Adler-32 = 0x00FB00B2
```

#### Tiny Code (C)

```c
#include <stdint.h>
#include <stdio.h>

uint32_t adler32(const unsigned char *data, size_t len) {
    uint32_t A = 1, B = 0;
    const uint32_t MOD = 65521;

    for (size_t i = 0; i < len; i++) {
        A = (A + data[i]) % MOD;
        B = (B + A) % MOD;
    }

    return (B << 16) | A;
}

int main() {
    unsigned char msg[] = "Hello";
    printf("Adler-32: %08X\n", adler32(msg, 5));
}
```

Output:

```
Adler-32: 062C0215
```

#### Tiny Code (Python)

```python
def adler32(data: bytes) -> int:
    MOD = 65521
    A, B = 1, 0
    for b in data:
        A = (A + b) % MOD
        B = (B + A) % MOD
    return (B << 16) | A

print(hex(adler32(b"Hello")))
```

Output:

```
0x62c0215
```

#### Why It Matters

- Simpler than CRC32, just addition and modulo
- Fast on small systems, especially in software
- Good for short data and quick integrity checks
- Used in:

  * zlib
  * PNG image format
  * network protocols needing low-cost validation

However, Adler-32 is less robust than CRC32 for long or highly repetitive data.

#### Comparison

| Property        | CRC32               | Adler-32                  |
| --------------- | ------------------- | ------------------------- |
| Arithmetic      | Polynomial (GF(2))  | Integer (mod prime)       |
| Bits            | 32                  | 32                        |
| Speed           | Moderate            | Very fast                 |
| Error detection | Strong              | Weaker                    |
| Typical use     | Networking, storage | Compression, local checks |

#### Complexity

| Operation    | Time   | Space  |
| ------------ | ------ | ------ |
| Encoding     | $O(n)$ | $O(1)$ |
| Verification | $O(n)$ | $O(1)$ |

#### Try It Yourself

1. Compute Adler-32 of `"BANANA"`.
2. Flip one byte and recompute, observe checksum change.
3. Compare execution time vs CRC32 on large data.
4. Experiment with removing modulo to see integer overflow behavior.
5. Implement incremental checksum updates for streaming data.

#### A Gentle Proof (Why It Works)

Adler-32 treats the message as two-layered accumulation:

- The A sum ensures local sensitivity (each byte affects the total).
- The B sum weights earlier bytes more heavily, amplifying positional effects.

Because of the modulo prime arithmetic, small bit flips yield large changes in the checksum, enough to catch random noise with high probability.

Adler-32 is a study in simplicity —
just two sums and a modulus,
yet fast enough to guard every PNG and compressed stream.

### 664 MD5 (Message Digest 5)

MD5 is one of the most well-known cryptographic hash functions.
It takes an arbitrary-length input and produces a 128-bit hash, a compact "fingerprint" of the data.
Although once widely used, MD5 is now considered cryptographically broken, but it remains useful in non-security applications like data integrity checks.

#### What Problem Are We Solving?

We need a fixed-size "digest" that uniquely identifies data blocks, files, or messages.
A good hash function should satisfy three key properties:

1. Preimage resistance, hard to find a message from its hash.
2. Second-preimage resistance, hard to find another message with the same hash.
3. Collision resistance, hard to find any two messages with the same hash.

MD5 was designed to meet these goals efficiently, though only the first two hold up in limited use today.

#### The Core Idea

MD5 processes data in 512-bit blocks, updating an internal state of four 32-bit words $(A, B, C, D)$.

At the end, these are concatenated to form the final 128-bit digest.

The algorithm consists of four main steps:

1. Padding the message to make its length congruent to 448 mod 512, then appending the original length (as a 64-bit value).

2. Initialization of the buffer:

   $$
   A = 0x67452301, \quad
   B = 0xEFCDAB89, \quad
   C = 0x98BADCFE, \quad
   D = 0x10325476
   $$

3. Processing each 512-bit block through four nonlinear rounds of operations using bitwise logic (AND, OR, XOR, NOT), additions, and rotations.

4. Output: concatenate $(A, B, C, D)$ into a 128-bit digest.

#### Main Transformation (Simplified)

For each 32-bit chunk:

$$
A = B + ((A + F(B, C, D) + X_k + T_i) \lll s)
$$

where

- $F$ is one of four nonlinear functions (changes per round),
- $X_k$ is a 32-bit block word,
- $T_i$ is a sine-based constant, and
- $\lll s$ is a left rotation.

Each round modifies $(A, B, C, D)$, creating diffusion and nonlinearity.

#### Tiny Code (Python)

This example uses the standard `hashlib` library:

```python
import hashlib

data = b"Hello, world!"
digest = hashlib.md5(data).hexdigest()
print("MD5:", digest)
```

Output:

```
MD5: 6cd3556deb0da54bca060b4c39479839
```

#### Tiny Code (C)

```c
#include <stdio.h>
#include <openssl/md5.h>

int main() {
    unsigned char digest[MD5_DIGEST_LENGTH];
    const char *msg = "Hello, world!";

    MD5((unsigned char*)msg, strlen(msg), digest);

    printf("MD5: ");
    for (int i = 0; i < MD5_DIGEST_LENGTH; i++)
        printf("%02x", digest[i]);
    printf("\n");
}
```

Output:

```
MD5: 6cd3556deb0da54bca060b4c39479839
```

#### Why It Matters

- Fast: computes hashes very quickly
- Compact: 128-bit output (32 hex characters)
- Deterministic: same input → same hash
- Used in:

  * File integrity checks
  * Versioning systems (e.g., git object naming)
  * Deduplication tools
  * Legacy digital signatures

However, do not use MD5 for cryptographic security, it's vulnerable to collision and chosen-prefix attacks.

#### Collisions and Security

Researchers have found that two different messages $m_1 \neq m_2$ can produce the same MD5 hash:

$$
\text{MD5}(m_1) = \text{MD5}(m_2)
$$

Modern attacks can generate such collisions in seconds on consumer hardware.
For any security-sensitive purpose, use SHA-256 or higher.

#### Comparison

| Property         | MD5    | SHA-1    | SHA-256        |
| ---------------- | ------ | -------- | -------------- |
| Output bits      | 128    | 160      | 256            |
| Speed            | Fast   | Moderate | Slower         |
| Security         | Broken | Weak     | Strong         |
| Collisions found | Yes    | Yes      | None practical |

#### Complexity

| Operation    | Time   | Space  |
| ------------ | ------ | ------ |
| Hashing      | $O(n)$ | $O(1)$ |
| Verification | $O(n)$ | $O(1)$ |

#### Try It Yourself

1. Compute the MD5 of `"apple"` and `"APPLE"`.
2. Concatenate two files and hash again, does the hash change predictably?
3. Experiment with `md5sum` on your terminal.
4. Compare results with SHA-1 and SHA-256.
5. Search for known MD5 collision examples online and verify hashes yourself.

#### A Gentle Proof (Why It Works)

Each MD5 operation is a combination of modular addition and bit rotation, nonlinear over GF(2).
These create avalanche effects, where flipping one bit of input changes about half of the bits in the output.

This ensures that small input changes yield drastically different hashes, though its collision resistance is mathematically compromised.

MD5 remains an elegant lesson in early cryptographic design —
a fast, human-readable fingerprint,
and a historical marker of how security evolves with computation.

### 665 SHA-1 (Secure Hash Algorithm 1)

SHA-1 is a 160-bit cryptographic hash function, once a cornerstone of digital signatures, SSL certificates, and version control systems.
It improved upon MD5 with a longer digest and more rounds, but like MD5, it has since been broken, though still valuable for understanding the evolution of modern hashing.

#### What Problem Are We Solving?

Like MD5, SHA-1 compresses an arbitrary-length input into a fixed-size hash (160 bits).
It was designed to provide stronger collision resistance and message integrity verification for use in authentication, encryption, and checksums.

#### The Core Idea

SHA-1 works block by block, processing the message in 512-bit chunks.
It maintains five 32-bit registers $(A, B, C, D, E)$ that evolve through 80 rounds of bitwise operations, shifts, and additions.

At a high level:

1. Preprocessing

   * Pad the message so its length is congruent to 448 mod 512.
   * Append the message length as a 64-bit integer.

2. Initialization

   * Set initial values:
     $$
     \begin{aligned}
     H_0 &= 0x67452301 \
     H_1 &= 0xEFCDAB89 \
     H_2 &= 0x98BADCFE \
     H_3 &= 0x10325476 \
     H_4 &= 0xC3D2E1F0
     \end{aligned}
     $$

3. Processing Each Block

   * Break each 512-bit block into 16 words $W_0, W_1, \dots, W_{15}$.
   * Extend them to $W_{16} \dots W_{79}$ using:
     $$
     W_t = (W_{t-3} \oplus W_{t-8} \oplus W_{t-14} \oplus W_{t-16}) \lll 1
     $$
   * Perform 80 rounds of updates:
     $$
     T = (A \lll 5) + f_t(B, C, D) + E + W_t + K_t
     $$
     Then shift registers:
     $E = D, D = C, C = B \lll 30, B = A, A = T$
     (with round-dependent function $f_t$ and constant $K_t$).

4. Output

   * Add $(A, B, C, D, E)$ to $(H_0, \dots, H_4)$.
   * The final hash is the concatenation of $H_0$ through $H_4$.

#### Round Functions

SHA-1 cycles through four different nonlinear Boolean functions:

| Rounds | Function | Formula                                             | Constant $K_t$ |
| ------ | -------- | --------------------------------------------------- | -------------- |
| 0–19   | Choose   | $f = (B \land C) \lor (\lnot B \land D)$            | 0x5A827999     |
| 20–39  | Parity   | $f = B \oplus C \oplus D$                           | 0x6ED9EBA1     |
| 40–59  | Majority | $f = (B \land C) \lor (B \land D) \lor (C \land D)$ | 0x8F1BBCDC     |
| 60–79  | Parity   | $f = B \oplus C \oplus D$                           | 0xCA62C1D6     |

These functions create nonlinear mixing and diffusion across bits.

#### Tiny Code (Python)

```python
import hashlib

msg = b"Hello, world!"
digest = hashlib.sha1(msg).hexdigest()
print("SHA-1:", digest)
```

Output:

```
SHA-1: d3486ae9136e7856bc42212385ea797094475802
```

#### Tiny Code (C)

```c
#include <stdio.h>
#include <openssl/sha.h>

int main() {
    unsigned char digest[SHA_DIGEST_LENGTH];
    const char *msg = "Hello, world!";

    SHA1((unsigned char*)msg, strlen(msg), digest);

    printf("SHA-1: ");
    for (int i = 0; i < SHA_DIGEST_LENGTH; i++)
        printf("%02x", digest[i]);
    printf("\n");
}
```

Output:

```
SHA-1: d3486ae9136e7856bc42212385ea797094475802
```

#### Why It Matters

- Extended output: 160 bits instead of MD5's 128.
- Wider adoption: used in SSL, Git, PGP, and digital signatures.
- Still deterministic and fast, suitable for data fingerprinting.

But it's cryptographically deprecated, collisions can be found in hours using modern hardware.

#### Known Attacks

In 2017, Google and CWI Amsterdam demonstrated SHAttered, a practical collision attack:

$$
\text{SHA1}(m_1) = \text{SHA1}(m_2)
$$

where $m_1$ and $m_2$ were distinct PDF files.

The attack required about $2^{63}$ operations, feasible with cloud resources.

SHA-1 is now considered insecure for cryptography and should be replaced with SHA-256 or SHA-3.

#### Comparison

| Property             | MD5        | SHA-1      | SHA-256             |
| -------------------- | ---------- | ---------- | ------------------- |
| Output bits          | 128        | 160        | 256                 |
| Rounds               | 64         | 80         | 64                  |
| Security             | Broken     | Broken     | Strong              |
| Collision resistance | $2^{64}$   | $2^{80}$   | $2^{128}$ (approx.) |
| Status               | Deprecated | Deprecated | Recommended         |

#### Complexity

| Operation    | Time   | Space  |
| ------------ | ------ | ------ |
| Hashing      | $O(n)$ | $O(1)$ |
| Verification | $O(n)$ | $O(1)$ |

SHA-1 remains computationally efficient, around 300 MB/s in C implementations.

#### Try It Yourself

1. Compute SHA-1 of `"Hello"` and `"hello"`, notice the avalanche effect.
2. Concatenate two files and compare SHA-1 and SHA-256 digests.
3. Use `sha1sum` on Linux to verify file integrity.
4. Study the SHAttered PDFs online and verify their identical hashes.

#### A Gentle Proof (Why It Works)

Each bit of the input influences every bit of the output through a cascade of rotations and modular additions.
The design ensures diffusion, small changes in input propagate widely, and confusion through nonlinear Boolean mixing.

Despite its elegance, mathematical weaknesses in its round structure allow attackers to manipulate internal states to produce collisions.

SHA-1 marked a turning point in cryptographic history —
beautifully engineered, globally adopted,
and a reminder that even strong math must evolve faster than computation.

### 666 SHA-256 (Secure Hash Algorithm 256-bit)

SHA-256 is one of the most widely used cryptographic hash functions today.
It's part of the SHA-2 family, standardized by NIST, and provides strong security for digital signatures, blockchain systems, and general data integrity.
Unlike its predecessors MD5 and SHA-1, SHA-256 remains unbroken in practice.

#### What Problem Are We Solving?

We need a function that produces a unique, fixed-size digest for any input —
but with strong resistance to collisions, preimages, and second preimages.

SHA-256 delivers that: it maps arbitrary-length data into a 256-bit digest,
creating a nearly impossible-to-invert fingerprint used for secure verification and identification.

#### The Core Idea

SHA-256 processes data in 512-bit blocks, updating eight 32-bit working registers through 64 rounds of modular arithmetic, logical operations, and message expansion.

Each round mixes bits in a nonlinear, irreversible way, achieving diffusion and confusion across the entire message.

#### Initialization

SHA-256 begins with eight fixed constants:

$$
\begin{aligned}
H_0 &= 0x6a09e667, \quad H_1 = 0xbb67ae85, \
H_2 &= 0x3c6ef372, \quad H_3 = 0xa54ff53a, \
H_4 &= 0x510e527f, \quad H_5 = 0x9b05688c, \
H_6 &= 0x1f83d9ab, \quad H_7 = 0x5be0cd19
\end{aligned}
$$

#### Message Expansion

Each 512-bit block is split into 16 words $W_0 \dots W_{15}$ and extended to 64 words:

$$
W_t = \sigma_1(W_{t-2}) + W_{t-7} + \sigma_0(W_{t-15}) + W_{t-16}
$$

with
$$
\sigma_0(x) = (x \mathbin{>>>} 7) \oplus (x \mathbin{>>>} 18) \oplus (x >> 3)
$$
$$
\sigma_1(x) = (x \mathbin{>>>} 17) \oplus (x \mathbin{>>>} 19) \oplus (x >> 10)
$$

#### Round Function

For each of 64 rounds:

$$
\begin{aligned}
T_1 &= H + \Sigma_1(E) + Ch(E, F, G) + K_t + W_t \
T_2 &= \Sigma_0(A) + Maj(A, B, C) \
H &= G \
G &= F \
F &= E \
E &= D + T_1 \
D &= C \
C &= B \
B &= A \
A &= T_1 + T_2
\end{aligned}
$$

where

$$
\begin{aligned}
Ch(x,y,z) &= (x \land y) \oplus (\lnot x \land z) \
Maj(x,y,z) &= (x \land y) \oplus (x \land z) \oplus (y \land z) \
\Sigma_0(x) &= (x \mathbin{>>>} 2) \oplus (x \mathbin{>>>} 13) \oplus (x \mathbin{>>>} 22) \
\Sigma_1(x) &= (x \mathbin{>>>} 6) \oplus (x \mathbin{>>>} 11) \oplus (x \mathbin{>>>} 25)
\end{aligned}
$$

Constants $K_t$ are 64 predefined 32-bit values derived from cube roots of primes.

#### Finalization

After all rounds, the hash values are updated:

$$
H_i = H_i + \text{working\_register}_i \quad \text{for } i = 0 \dots 7
$$

The final digest is the concatenation of $(H_0, H_1, \dots, H_7)$, forming a 256-bit hash.

#### Tiny Code (Python)

```python
import hashlib

data = b"Hello, world!"
digest = hashlib.sha256(data).hexdigest()
print("SHA-256:", digest)
```

Output:

```
SHA-256: c0535e4be2b79ffd93291305436bf889314e4a3faec05ecffcbb7df31ad9e51a
```

#### Tiny Code (C)

```c
#include <stdio.h>
#include <openssl/sha.h>

int main() {
    unsigned char digest[SHA256_DIGEST_LENGTH];
    const char *msg = "Hello, world!";

    SHA256((unsigned char*)msg, strlen(msg), digest);

    printf("SHA-256: ");
    for (int i = 0; i < SHA256_DIGEST_LENGTH; i++)
        printf("%02x", digest[i]);
    printf("\n");
}
```

Output:

```
SHA-256: c0535e4be2b79ffd93291305436bf889314e4a3faec05ecffcbb7df31ad9e51a
```

#### Why It Matters

- Strong collision resistance (no practical attacks)
- Used in cryptographic protocols: TLS, PGP, SSH, Bitcoin, Git
- Efficient: processes large data quickly
- Deterministic and irreversible

SHA-256 underpins blockchain integrity, every Bitcoin block is linked by SHA-256 hashes.

#### Security Comparison

| Property         | MD5    | SHA-1  | SHA-256         |
| ---------------- | ------ | ------ | --------------- |
| Output bits      | 128    | 160    | 256             |
| Collisions found | Yes    | Yes    | None practical  |
| Security         | Broken | Broken | Secure          |
| Typical use      | Legacy | Legacy | Modern security |

#### Complexity

| Operation    | Time   | Space  |
| ------------ | ------ | ------ |
| Hashing      | $O(n)$ | $O(1)$ |
| Verification | $O(n)$ | $O(1)$ |

SHA-256 is fast enough for real-time use but designed to resist hardware brute-force attacks.

#### Try It Yourself

1. Hash `"hello"` and `"Hello"`, see how much the output changes.
2. Compare SHA-256 vs SHA-512 speed.
3. Implement message padding by hand for small examples.
4. Experiment with Bitcoin block header hashing.
5. Compute double SHA-256 of a file and compare with `sha256sum`.

#### A Gentle Proof (Why It Works)

SHA-256's strength lies in its nonlinear mixing, bit rotation, and modular addition, which spread every input bit's influence across all output bits.
Because every round scrambles data through independent functions and constants, the process is practically irreversible.

Mathematically, there's no known way to invert or collide SHA-256 faster than brute force ($2^{128}$ effort).

SHA-256 is the cryptographic backbone of the digital age —
trusted by blockchains, browsers, and systems everywhere —
a balance of elegance, speed, and mathematical hardness.

### 667 SHA-3 (Keccak)

SHA-3, also known as Keccak, is the latest member of the Secure Hash Algorithm family, standardized by NIST in 2015.
It represents a complete redesign, not a patch, introducing the sponge construction, which fundamentally changes how hashing works.
SHA-3 is flexible, secure, and mathematically elegant.

#### What Problem Are We Solving?

SHA-2 (like SHA-256) is strong, but it shares internal structure with older hashes such as SHA-1 and MD5.
If a major cryptanalytic breakthrough appeared against that structure, the entire family could be at risk.

SHA-3 was designed as a cryptographic fallback, a completely different approach, immune to the same types of attacks, yet compatible with existing use cases.

#### The Core Idea, Sponge Construction

SHA-3 absorbs and squeezes data through a sponge function, based on a large internal state of 1600 bits.

- The sponge has two parameters:

  * Rate (r), how many bits are processed per round.
  * Capacity (c), how many bits remain hidden (for security).

For SHA3-256, $r = 1088$ and $c = 512$, since $r + c = 1600$.

The process alternates between two phases:

1. Absorb phase
   The input is XORed into the state, block by block, then transformed by the Keccak permutation.

2. Squeeze phase
   The output bits are read from the state; more rounds are applied if more bits are needed.

#### Keccak State and Transformation

The internal state is a 3D array of bits, visualized as $5 \times 5$ lanes of 64 bits each:

$$
A[x][y][z], \quad 0 \le x, y < 5, ; 0 \le z < 64
$$

Each round of Keccak applies five transformations:

1. θ (theta), column parity mixing
2. ρ (rho), bit rotation
3. π (pi), lane permutation
4. χ (chi), nonlinear mixing (bitwise logic)
5. ι (iota), round constant injection

Each round scrambles bits across the state, achieving diffusion and confusion like a fluid stirring in 3D.

#### Padding Rule

Before processing, the message is padded using the multi-rate padding rule:

$$
\text{pad}(M) = M , || , 0x06 , || , 00...0 , || , 0x80
$$

This ensures that each message is unique, even with similar lengths.

#### Tiny Code (Python)

```python
import hashlib

data = b"Hello, world!"
digest = hashlib.sha3_256(data).hexdigest()
print("SHA3-256:", digest)
```

Output:

```
SHA3-256: 644bcc7e564373040999aac89e7622f3ca71fba1d972fd94a31c3bfbf24e3938
```

#### Tiny Code (C, OpenSSL)

```c
#include <stdio.h>
#include <openssl/evp.h>

int main() {
    unsigned char digest[32];
    const char *msg = "Hello, world!";
    EVP_Digest(msg, strlen(msg), digest, NULL, EVP_sha3_256(), NULL);

    printf("SHA3-256: ");
    for (int i = 0; i < 32; i++)
        printf("%02x", digest[i]);
    printf("\n");
}
```

Output:

```
SHA3-256: 644bcc7e564373040999aac89e7622f3ca71fba1d972fd94a31c3bfbf24e3938
```

#### Why It Matters

- Completely different design, not Merkle–Damgård like MD5/SHA-2
- Mathematically clean and provable sponge model
- Supports variable output lengths (SHAKE128, SHAKE256)
- Resists all known cryptanalytic attacks

Used in:

- Blockchain research (e.g., Ethereum uses Keccak-256)
- Post-quantum cryptography frameworks
- Digital forensics and verifiable ledgers

#### Comparison

| Algorithm | Structure       | Output bits | Year | Security Status |
| --------- | --------------- | ----------- | ---- | --------------- |
| MD5       | Merkle–Damgård  | 128         | 1992 | Broken          |
| SHA-1     | Merkle–Damgård  | 160         | 1995 | Broken          |
| SHA-256   | Merkle–Damgård  | 256         | 2001 | Secure          |
| SHA-3     | Sponge (Keccak) | 256         | 2015 | Secure          |

#### Complexity

| Operation    | Time   | Space  |
| ------------ | ------ | ------ |
| Hashing      | $O(n)$ | $O(1)$ |
| Verification | $O(n)$ | $O(1)$ |

Although SHA-3 is slower than SHA-256 in pure software, it scales better in hardware and parallel implementations.

#### Try It Yourself

1. Compute both `sha256()` and `sha3_256()` for the same file, compare outputs.

2. Experiment with variable-length digests using SHAKE256:

   ```python
   from hashlib import shake_256
   print(shake_256(b"hello").hexdigest(64))
   ```

3. Visualize Keccak's 5×5×64-bit state, draw how rounds mix the lanes.

4. Implement a toy sponge function with XOR and rotation to understand the design.

#### A Gentle Proof (Why It Works)

Unlike Merkle–Damgård constructions, which extend hashes block by block, the sponge absorbs input into a large nonlinear state, hiding correlations.
Since only the "rate" bits are exposed, the "capacity" ensures that finding collisions or preimages would require $2^{c/2}$ work, for SHA3-256, about $2^{256}$ effort.

This design makes SHA-3 provably secure up to its capacity against generic attacks.

SHA-3 is the calm successor to SHA-2 —
not born from crisis, but from mathematical renewal —
a sponge that drinks data and squeezes out pure randomness.

### 668 HMAC (Hash-Based Message Authentication Code)

HMAC is a method for verifying both integrity and authenticity of messages.
It combines any cryptographic hash function (like SHA-256 or SHA-3) with a secret key, ensuring that only someone who knows the key can produce or verify the correct hash.

HMAC is the foundation of many authentication protocols, including TLS, OAuth, JWT, and AWS API signing.

#### What Problem Are We Solving?

A regular hash like SHA-256 verifies data integrity, but not authenticity.
Anyone can compute a hash of a file, so how can you tell who actually made it?

HMAC introduces a shared secret key to ensure that only authorized parties can generate or validate the correct hash.

If the hash doesn't match, it means the data was either modified or produced without the correct key.

#### The Core Idea

HMAC wraps a cryptographic hash function in two layers of keyed hashing:

1. Inner hash, hash of the message combined with an inner key pad.
2. Outer hash, hash of the inner digest combined with an outer key pad.

Formally:

$$
\text{HMAC}(K, M) = H\left((K' \oplus opad) , || , H((K' \oplus ipad) , || , M)\right)
$$

where:

- $H$ is a secure hash function (e.g., SHA-256)
- $K'$ is the key padded or hashed to match block size
- $opad$ is the "outer pad" (0x5C repeated)
- $ipad$ is the "inner pad" (0x36 repeated)
- $||$ means concatenation
- $\oplus$ means XOR

This two-layer structure protects against attacks on hash function internals (like length-extension attacks).

#### Step-by-Step Example (Using SHA-256)

1. Pad the secret key $K$ to 64 bytes (block size of SHA-256).

2. Compute inner hash:

   $$
   \text{inner} = H((K' \oplus ipad) , || , M)
   $$

3. Compute outer hash:

   $$
   \text{HMAC} = H((K' \oplus opad) , || , \text{inner})
   $$

4. The result is a 256-bit digest authenticating both key and message.

#### Tiny Code (Python)

```python
import hmac, hashlib

key = b"secret-key"
message = b"Attack at dawn"
digest = hmac.new(key, message, hashlib.sha256).hexdigest()
print("HMAC-SHA256:", digest)
```

Output:

```
HMAC-SHA256: 2cba05e5a7e03ffccf13e585c624cfa7cbf4b82534ef9ce454b0943e97ebc8aa
```

#### Tiny Code (C)

```c
#include <stdio.h>
#include <string.h>
#include <openssl/hmac.h>

int main() {
    unsigned char result[EVP_MAX_MD_SIZE];
    unsigned int len = 0;
    const char *key = "secret-key";
    const char *msg = "Attack at dawn";

    HMAC(EVP_sha256(), key, strlen(key),
         (unsigned char*)msg, strlen(msg),
         result, &len);

    printf("HMAC-SHA256: ");
    for (unsigned int i = 0; i < len; i++)
        printf("%02x", result[i]);
    printf("\n");
}
```

Output:

```
HMAC-SHA256: 2cba05e5a7e03ffccf13e585c624cfa7cbf4b82534ef9ce454b0943e97ebc8aa
```

#### Why It Matters

- Protects authenticity, only someone with the key can compute a valid HMAC.
- Protects integrity, any change in data or key changes the HMAC.
- Resistant to length-extension and replay attacks.

Used in:

- TLS, SSH, IPsec
- AWS and Google Cloud API signing
- JWT (HS256)
- Webhooks, signed URLs, and secure tokens

#### Comparison of Hash-Based MACs

| Algorithm     | Underlying Hash | Output (bits) | Status        |
| ------------- | --------------- | ------------- | ------------- |
| HMAC-MD5      | MD5             | 128           | Insecure      |
| HMAC-SHA1     | SHA-1           | 160           | Weak (legacy) |
| HMAC-SHA256   | SHA-256         | 256           | Recommended   |
| HMAC-SHA3-256 | SHA-3           | 256           | Future-safe   |

#### Complexity

| Operation    | Time   | Space  |
| ------------ | ------ | ------ |
| Hashing      | $O(n)$ | $O(1)$ |
| Verification | $O(n)$ | $O(1)$ |

HMAC's cost is roughly 2× the underlying hash function, since it performs two passes (inner + outer).

#### Try It Yourself

1. Compute HMAC-SHA256 of `"hello"` with key `"abc"`.
2. Modify one byte, notice how the digest changes completely.
3. Try verifying with a wrong key, verification fails.
4. Compare performance between SHA1 and SHA256 versions.
5. Implement a manual HMAC from scratch using the formula above.

#### A Gentle Proof (Why It Works)

The key insight:
Even if an attacker knows $H(K || M)$, they cannot compute $H(K || M')$ for another message $M'$, because $K$ is mixed inside the hash in a non-reusable way.

Mathematically, the inner and outer pads break the linearity of the compression function, removing any exploitable structure.

Security depends entirely on the hash's collision resistance and the secrecy of the key.

HMAC is the handshake between mathematics and trust —
a compact cryptographic signature proving,
"This message came from someone who truly knew the key."

### 669 Merkle Tree (Hash Tree)

A Merkle Tree (or hash tree) is a hierarchical data structure that provides efficient and secure verification of large data sets.
It's the backbone of blockchains, distributed systems, and version control systems like Git, enabling integrity proofs with logarithmic verification time.

#### What Problem Are We Solving?

Suppose you have a massive dataset, gigabytes of files or blocks, and you want to verify whether a single piece is intact or altered.
Hashing the entire dataset repeatedly would be expensive.

A Merkle Tree allows verification of any part of the data using only a small proof, without rehashing everything.

#### The Core Idea

A Merkle Tree is built by recursively hashing pairs of child nodes until a single root hash is obtained.

- Leaf nodes: contain hashes of data blocks.
- Internal nodes: contain hashes of their concatenated children.
- Root hash: uniquely represents the entire dataset.

If any block changes, the change propagates upward, altering the root hash, making tampering immediately detectable.

#### Construction

Given four data blocks $D_1, D_2, D_3, D_4$:

1. Compute leaf hashes:
   $$
   H_1 = H(D_1), \quad H_2 = H(D_2), \quad H_3 = H(D_3), \quad H_4 = H(D_4)
   $$
2. Compute intermediate hashes:
   $$
   H_{12} = H(H_1 || H_2), \quad H_{34} = H(H_3 || H_4)
   $$
3. Compute root:
   $$
   H_{root} = H(H_{12} || H_{34})
   $$

The final $H_{root}$ acts as a fingerprint for all underlying data.

#### Example (Visual)

```
              H_root
              /    \
         H_12       H_34
        /   \       /   \
     H1     H2   H3     H4
     |      |    |      |
    D1     D2   D3     D4
```

#### Verification Proof (Merkle Path)

To prove that $D_3$ belongs to the tree:

1. Provide $H_4$, $H_{12}$, and the position of each (left/right).

2. Compute upward:

   $$
   H_3 = H(D_3)
   $$

   $$
   H_{34} = H(H_3 || H_4)
   $$

   $$
   H_{root}' = H(H_{12} || H_{34})
   $$

3. If $H_{root}' = H_{root}$, the data block is verified.

Only log₂(n) hashes are needed for verification.

#### Tiny Code (Python)

```python
import hashlib

def sha256(data):
    return hashlib.sha256(data).digest()

def merkle_tree(leaves):
    if len(leaves) == 1:
        return leaves[0]
    if len(leaves) % 2 == 1:
        leaves.append(leaves[-1])
    parents = []
    for i in range(0, len(leaves), 2):
        parents.append(sha256(leaves[i] + leaves[i+1]))
    return merkle_tree(parents)

data = [b"D1", b"D2", b"D3", b"D4"]
leaves = [sha256(d) for d in data]
root = merkle_tree(leaves)
print("Merkle Root:", root.hex())
```

Output (example):

```
Merkle Root: 16d1c7a0cfb3b6e4151f3b24a884b78e0d1a826c45de2d0e0d0db1e4e44bff62
```

#### Tiny Code (C, using OpenSSL)

```c
#include <stdio.h>
#include <openssl/sha.h>
#include <string.h>

void sha256(unsigned char *data, size_t len, unsigned char *out) {
    SHA256(data, len, out);
}

void print_hex(unsigned char *h, int len) {
    for (int i = 0; i < len; i++) printf("%02x", h[i]);
    printf("\n");
}

int main() {
    unsigned char d1[] = "D1", d2[] = "D2", d3[] = "D3", d4[] = "D4";
    unsigned char h1[32], h2[32], h3[32], h4[32], h12[32], h34[32], root[32];
    unsigned char tmp[64];

    sha256(d1, 2, h1); sha256(d2, 2, h2);
    sha256(d3, 2, h3); sha256(d4, 2, h4);

    memcpy(tmp, h1, 32); memcpy(tmp+32, h2, 32); sha256(tmp, 64, h12);
    memcpy(tmp, h3, 32); memcpy(tmp+32, h4, 32); sha256(tmp, 64, h34);
    memcpy(tmp, h12, 32); memcpy(tmp+32, h34, 32); sha256(tmp, 64, root);

    printf("Merkle Root: "); print_hex(root, 32);
}
```

#### Why It Matters

- Integrity, any data change alters the root hash.
- Efficiency, logarithmic proof size and verification.
- Scalability, used in systems with millions of records.
- Used in:

  * Bitcoin and Ethereum blockchains
  * Git commits and version histories
  * IPFS and distributed file systems
  * Secure software updates (Merkle proofs)

#### Comparison with Flat Hashing

| Method      | Verification Cost | Data Tamper Detection | Proof Size |
| ----------- | ----------------- | --------------------- | ---------- |
| Single hash | $O(n)$            | Whole file only       | Full data  |
| Merkle tree | $O(\log n)$       | Any block             | Few hashes |

#### Complexity

| Operation    | Time        | Space  |
| ------------ | ----------- | ------ |
| Tree build   | $O(n)$      | $O(n)$ |
| Verification | $O(\log n)$ | $O(1)$ |

#### Try It Yourself

1. Build a Merkle tree of 8 messages.
2. Modify one message, watch how the root changes.
3. Compute a Merkle proof for the 3rd message and verify it manually.
4. Implement double SHA-256 as used in Bitcoin.
5. Explore how Git uses trees and commits as Merkle DAGs.

#### A Gentle Proof (Why It Works)

Each node's hash depends on its children, which depend recursively on all leaves below.
Thus, changing even a single bit in any leaf alters all ancestor hashes and the root.

Because the underlying hash function is collision-resistant, two different datasets cannot produce the same root.

Mathematically, for a secure hash $H$:

$$
H_{root}(D_1, D_2, ..., D_n) = H_{root}(D'_1, D'_2, ..., D'_n)
\Rightarrow D_i = D'_i \text{ for all } i
$$

A Merkle Tree is integrity made scalable —
a digital fingerprint for forests of data,
proving that every leaf is still what it claims to be.

### 670 Hash Collision Detection (Birthday Bound Simulation)

Every cryptographic hash function, even the strongest ones, can theoretically produce collisions, where two different inputs yield the same hash.
The birthday paradox gives us a way to estimate how likely that is.
This section explores collision probability, detection simulation, and why even a 256-bit hash can be "breakable" in principle, just not in practice.

#### What Problem Are We Solving?

When we hash data, we hope every message gets a unique output.
But since the hash space is finite, collisions must exist.
The key question is: *how many random hashes must we generate before we expect a collision?*

This is the birthday problem in disguise, the same math that tells us 23 people suffice for a 50% chance of a shared birthday.

#### The Core Idea, Birthday Bound

If a hash function produces $N$ possible outputs,
then after about $\sqrt{N}$ random hashes, the probability of a collision reaches about 50%.

For a hash of $b$ bits:

$$
N = 2^b, \quad \text{so collisions appear around } 2^{b/2}.
$$

| Hash bits     | Expected collision at | Practical security |
| ------------- | --------------------- | ------------------ |
| 32            | $2^{16}$ (≈65k)       | Weak               |
| 64            | $2^{32}$              | Moderate           |
| 128 (MD5)     | $2^{64}$              | Broken             |
| 160 (SHA-1)   | $2^{80}$              | Broken (feasible)  |
| 256 (SHA-256) | $2^{128}$             | Secure             |
| 512 (SHA-512) | $2^{256}$             | Extremely secure   |

Thus, SHA-256's 128-bit collision resistance is strong enough for modern security.

#### The Birthday Probability Formula

The probability of *at least one collision* after hashing $k$ random messages is:

$$
P(k) \approx 1 - e^{-k(k-1)/(2N)}
$$

If we set $P(k) = 0.5$, solving for $k$ gives:

$$
k \approx 1.1774 \sqrt{N}
$$

#### Tiny Code (Python)

Let's simulate hash collisions using SHA-1 and random data.

```python
import hashlib, random, string

def random_str(n=8):
    return ''.join(random.choice(string.ascii_letters) for _ in range(n))

def collision_simulation(bits=32):
    seen = {}
    mask = (1 << bits) - 1
    count = 0
    while True:
        s = random_str(8).encode()
        h = int.from_bytes(hashlib.sha256(s).digest(), 'big') & mask
        if h in seen:
            return count, s, seen[h]
        seen[h] = s
        count += 1

print("Simulating 32-bit collision...")
count, s1, s2 = collision_simulation(32)
print(f"Collision after {count} hashes:")
print(f"{s1} and {s2}")
```

Typical output:

```
Simulating 32-bit collision...
Collision after 68314 hashes:
b'FqgWbUzk' and b'yLpTGZxu'
```

This matches the theoretical expectation: around $\sqrt{2^{32}} = 65,536$ trials.

#### Tiny Code (C, simple model)

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <openssl/sha.h>

int main() {
    unsigned int seen[65536] = {0};
    unsigned char hash[SHA_DIGEST_LENGTH];
    unsigned char input[16];
    int count = 0;

    while (1) {
        sprintf((char*)input, "%d", rand());
        SHA1(input, strlen((char*)input), hash);
        unsigned int h = (hash[0] << 8) | hash[1];  // take 16 bits
        if (seen[h]) {
            printf("Collision found after %d hashes\n", count);
            break;
        }
        seen[h] = 1;
        count++;
    }
}
```

#### Why It Matters

- Every hash can collide, but for large bit sizes, collisions are *astronomically unlikely*.
- Helps explain why MD5 and SHA-1 are no longer safe, their "birthday bound" can be reached by modern GPUs.
- Used in security proofs for digital signatures, blockchain mining, and random oracles.

#### Visualization

| Hash bits | Trial count for 50% collision  | Equivalent           |
| --------- | ------------------------------ | -------------------- |
| 16        | 300                            | trivial              |
| 32        | 65,000                         | toy example          |
| 64        | 4 billion                      | feasible for testing |
| 128       | $2^{64}$ ≈ $1.8\times10^{19}$  | too large            |
| 256       | $2^{128}$ ≈ $3.4\times10^{38}$ | impossible           |

For comparison, the total number of atoms in the observable universe is about $10^{80}$, still smaller than $2^{256}$.

#### Complexity

| Operation    | Time          | Space         |
| ------------ | ------------- | ------------- |
| Simulation   | $O(\sqrt{N})$ | $O(\sqrt{N})$ |
| Verification | $O(1)$        | $O(1)$        |

Hash collision detection scales quadratically with output bits, making brute-force infeasible for modern algorithms.

#### Try It Yourself

1. Run the Python simulation for 24-bit, 28-bit, and 32-bit masks.
2. Compare empirical counts to the theoretical $\sqrt{2^b}$ curve.
3. Visualize probability vs. number of hashes using `matplotlib`.
4. Replace SHA-256 with SHA3-256, same collision statistics.
5. Estimate the expected collision time for SHA-1 given GPU speeds (you'll find it's real).

#### A Gentle Proof (Why It Works)

Let $N = 2^b$ possible hashes.
After $k$ draws, the chance that all are distinct is approximately:

$$
P_{\text{no-collision}} = \prod_{i=1}^{k-1} \left(1 - \frac{i}{N}\right)
\approx e^{-k(k-1)/(2N)}
$$

Taking the complement gives the collision probability $P(k)$.
Setting $P(k)=0.5$ yields the familiar $\sqrt{N}$ bound, a purely combinatorial fact independent of the hash algorithm.

Hash collision detection is where probability meets cryptography —
a quiet reminder that even perfect functions have limits,
and that randomness itself defines how close we dare to get to them.

# Section 68. Approximate and Streaming Matching 

### 671 K-Approximate String Matching

The K-Approximate String Matching problem asks:
given a pattern `P` and a text `T`, find all positions in `T` where `P` appears with at most `k` differences (insertions, deletions, or substitutions).

This problem underlies fuzzy search, spell correction, bioinformatics alignment, and error-tolerant pattern recognition.

#### The Core Idea

Exact matching requires all characters to align perfectly.
Approximate matching allows small differences, quantified by edit distance ≤ k.

We use dynamic programming or bit-parallel techniques to efficiently detect these approximate matches.

#### The Dynamic Programming Formulation

Let pattern `P` of length `m`, text `T` of length `n`, and distance threshold `k`.

Define a DP table:

$$
dp[i][j] = \text{minimum edit distance between } P[1..i] \text{ and } T[1..j]
$$

Recurrence:

$$
dp[i][j] =
\begin{cases}
dp[i-1][j-1], & \text{if } P[i] = T[j],\\[4pt]
1 + \min\big(dp[i-1][j],\ dp[i][j-1],\ dp[i-1][j-1]\big), & \text{otherwise.}
\end{cases}
$$


At each position `j`, if $dp[m][j] \le k$,
then the substring `T[j-m+1..j]` approximately matches `P`.

#### Example

Let
`T = "abcdefg"`
`P = "abxd"`
`k = 1`

Dynamic programming finds one approximate match at position 1
because `"abxd"` differs from `"abcd"` by a single substitution.

#### Bit-Parallel Simplification (for small k)

The Bitap (Shift-Or) algorithm can be extended to handle up to `k` errors using bitmasks and shift operations.

Each bit represents whether a prefix of `P` matches at a given alignment.

Time complexity becomes:

$$
O\left(\frac{n \cdot k}{w}\right)
$$

where `w` is the machine word size (typically 64).

#### Tiny Code (Python, DP Version)

```python
def k_approx_match(text, pattern, k):
    n, m = len(text), len(pattern)
    dp = [list(range(m + 1))] + [[0] * (m + 1) for _ in range(n)]
    for i in range(1, n + 1):
        dp[i][0] = 0
        for j in range(1, m + 1):
            cost = 0 if text[i-1] == pattern[j-1] else 1
            dp[i][j] = min(
                dp[i-1][j] + 1,      # insertion
                dp[i][j-1] + 1,      # deletion
                dp[i-1][j-1] + cost  # substitution
            )
        if dp[i][m] <= k:
            print(f"Match ending at {i}, distance {dp[i][m]}")

k_approx_match("abcdefg", "abxd", 1)
```

Output:

```
Match ending at 4, distance 1
```

#### Tiny Code (C, Bitap-Style)

```c
#include <stdio.h>
#include <string.h>
#include <stdint.h>

void bitap_approx(const char *text, const char *pattern, int k) {
    int n = strlen(text), m = strlen(pattern);
    uint64_t mask[256] = {0};
    for (int i = 0; i < m; i++) mask[(unsigned char)pattern[i]] |= 1ULL << i;

    uint64_t R[k+1];
    for (int d = 0; d <= k; d++) R[d] = ~0ULL;
    uint64_t pattern_mask = 1ULL << (m - 1);

    for (int i = 0; i < n; i++) {
        uint64_t oldR = R[0];
        R[0] = ((R[0] << 1) | 1ULL) & mask[(unsigned char)text[i]];
        for (int d = 1; d <= k; d++) {
            uint64_t tmp = R[d];
            R[d] = ((R[d] << 1) | 1ULL) & mask[(unsigned char)text[i]];
            R[d] |= (oldR | (R[d-1] << 1) | oldR << 1);
            oldR = tmp;
        }
        if (!(R[k] & pattern_mask))
            printf("Approximate match ending at %d\n", i + 1);
    }
}
```

#### Why It Matters

Approximate matching powers:

- Spell-checking ("recieve" → "receive")
- DNA alignment (A-C-T mismatches)
- Search engines with typo tolerance
- Real-time text recognition and OCR correction
- Command-line fuzzy filters (like `fzf`)

#### Complexity

| Algorithm           | Time      | Space  |
| ------------------- | --------- | ------ |
| DP (naive)          | $O(nm)$   | $O(m)$ |
| Bitap (for small k) | $O(nk/w)$ | $O(k)$ |

For small edit distances (k ≤ 3), bit-parallel methods are extremely fast.

#### Try It Yourself

1. Run the DP version and visualize the `dp` table.
2. Increase `k` and see how matches expand.
3. Test with random noise in the text.
4. Implement early termination when `dp[i][m] > k`.
5. Compare Bitap's speed to the DP algorithm for large inputs.

#### A Gentle Proof (Why It Works)

The DP recurrence ensures that each mismatch, insertion, or deletion contributes exactly 1 to the total distance.
By scanning the last column of the DP table, we detect substrings whose minimal edit distance is ≤ k.
Because all transitions are local, the algorithm guarantees correctness for every alignment.

Approximate matching brings tolerance to the rigid world of algorithms —
finding patterns not as they *are*, but as they *almost are*.

### 672 Bitap Algorithm (Bitwise Dynamic Programming)

The Bitap algorithm, also known as Shift-Or or Bitwise Pattern Matching, performs fast text search by encoding the pattern as a bitmask and updating it with simple bitwise operations.
It is one of the earliest and most elegant examples of bit-parallel algorithms for string matching.

#### The Core Idea

Instead of comparing characters one by one, Bitap represents the state of matching as a bit vector.
Each bit indicates whether a prefix of the pattern matches a substring of the text up to the current position.

This allows us to process up to 64 pattern characters per CPU word using bitwise operations, making it much faster than naive scanning.

#### The Bitmask Setup

Let:

- Pattern `P` of length `m`
- Text `T` of length `n`
- Machine word of width `w` (typically 64 bits)

For each character `c`, we precompute a bitmask:

$$
B[c] = \text{bitmask where } B[c]_i = 0 \text{ if } P[i] = c, \text{ else } 1
$$

#### The Matching Process

We maintain a running state vector $R$ initialized as all 1s:

$$
R_0 = \text{all bits set to } 1
$$

For each character $T[j]$:

$$
R_j = \big((R_{j-1} \ll 1) \,\vert\, 1\big) \,\&\, B[T[j]]
$$

If the bit corresponding to the last pattern position becomes 0, a match is found at position $j - m + 1$.

#### Example

Let
`T = "abcxabcdabxabcdabcdabcy"`
`P = "abcdabcy"`

Pattern length $m = 8$.
When processing the text, each bit of `R` shifts left to represent the growing match.
When the 8th bit becomes zero, it signals a full match of `P`.

#### Tiny Code (Python, Bitap Exact Match)

```python
def bitap_search(text, pattern):
    m = len(pattern)
    mask = {}
    for c in set(text):
        mask[c] = ~0
    for i, c in enumerate(pattern):
        mask[c] &= ~(1 << i)

    R = ~0
    for j, c in enumerate(text):
        R = ((R << 1) | 1) & mask.get(c, ~0)
        if (R & (1 << (m - 1))) == 0:
            print(f"Match found ending at position {j}")

bitap_search("abcxabcdabxabcdabcdabcy", "abcdabcy")
```

Output:

```
Match found ending at position 23
```

#### Bitap with K Errors (Approximate Matching)

Bitap can be extended to allow up to $k$ mismatches (insertions, deletions, or substitutions).
We maintain $k + 1$ bit vectors $R_0, R_1, ..., R_k$:

$$
R_j = \big((R_{j-1} \ll 1) \,\vert\, 1\big) \,\&\, B[T[j]]
$$

and for $d > 0$:

$$
R_d = \big((R_d \ll 1) \,\vert\, 1\big) \,\&\, B[T[j]] 
      \,\vert\, R_{d-1} 
      \,\vert\, \big((R_{d-1} \ll 1) \,\vert\, (R_{d-1} \gg 1)\big)
$$


If any $R_d$ has the last bit zero, a match within $d$ edits exists.

#### Tiny Code (C, Exact Match)

```c
#include <stdio.h>
#include <stdint.h>
#include <string.h>

void bitap(const char *text, const char *pattern) {
    int n = strlen(text), m = strlen(pattern);
    uint64_t mask[256];
    for (int i = 0; i < 256; i++) mask[i] = ~0ULL;
    for (int i = 0; i < m; i++) mask[(unsigned char)pattern[i]] &= ~(1ULL << i);

    uint64_t R = ~0ULL;
    for (int j = 0; j < n; j++) {
        R = ((R << 1) | 1ULL) & mask[(unsigned char)text[j]];
        if (!(R & (1ULL << (m - 1))))
            printf("Match ending at position %d\n", j + 1);
    }
}
```

#### Why It Matters

- Compact and fast: uses pure bitwise logic
- Suitable for short patterns (fits within machine word)
- Foundation for approximate matching and fuzzy search
- Practical in grep, spell correction, DNA search, and streaming text

#### Complexity

| Operation     | Time    | Space           |
| ------------- | ------- | --------------- |
| Exact Match   | $O(n)$  | $O(\sigma)$     |
| With k errors | $O(nk)$ | $O(\sigma + k)$ |

Here $\sigma$ is the alphabet size.

#### Try It Yourself

1. Modify the Python version to return match start positions instead of ends.
2. Test with long texts and observe near-linear runtime.
3. Extend it to allow 1 mismatch using an array of `R[k]`.
4. Visualize `R` bit patterns to see how the prefix match evolves.
5. Compare performance with KMP and Naive matching.

#### A Gentle Proof (Why It Works)

Each left shift of `R` corresponds to extending the matched prefix by one character.
Masking by `B[c]` zeroes bits where the pattern character matches the text character.
Thus, a 0 at position `m−1` means the entire pattern has matched —
the algorithm simulates a dynamic programming table with bitwise parallelism.

Bitap is a perfect illustration of algorithmic elegance —
compressing a full table of comparisons into a handful of bits that dance in sync with the text.

### 673 Landau–Vishkin Algorithm (Edit Distance ≤ k)

The Landau–Vishkin algorithm solves the *k-approximate string matching* problem efficiently by computing the edit distance up to a fixed threshold k without constructing a full dynamic programming table.
It's one of the most elegant linear-time algorithms for approximate matching when k is small.

#### The Core Idea

We want to find all positions in text T where pattern P matches with at most k edits (insertions, deletions, or substitutions).

Instead of computing the full $m \times n$ edit distance table, the algorithm tracks diagonals in the DP grid and extends matches as far as possible along each diagonal.

This diagonal-based thinking makes it much faster for small k.

#### The DP View in Brief

In the classic edit distance DP,
each cell $(i, j)$ represents the edit distance between $P[1..i]$ and $T[1..j]$.

Cells with the same difference $d = j - i$ lie on the same diagonal.
Each edit operation shifts you slightly between diagonals.

The Landau–Vishkin algorithm computes, for each edit distance e (0 ≤ e ≤ k),
how far we can match along each diagonal after e edits.

#### The Main Recurrence

Let:

- `L[e][d]` = furthest matched prefix length on diagonal `d` (offset `j - i`) after `e` edits.

We update as:

$$
L[e][d] =
\begin{cases}
L[e-1][d-1] + 1, & \text{insertion},\\[4pt]
L[e-1][d+1], & \text{deletion},\\[4pt]
L[e-1][d] + 1, & \text{substitution (if mismatch)}.
\end{cases}
$$


Then, from that position, we extend as far as possible while characters match:

$$
\text{while } P[L[e][d]+1] = T[L[e][d]+d+1],\ L[e][d]++
$$

If at any point $L[e][d] \ge m$,
we found a match with ≤ e edits.

#### Example (Plain Intuition)

Let
`P = "kitten"`, `T = "sitting"`, and $k = 2$.

The algorithm starts by matching all diagonals with 0 edits.
It then allows 1 edit (skip, replace, insert)
and tracks how far matching can continue.
In the end, it confirms that `"kitten"` matches `"sitting"` with 2 edits.

#### Tiny Code (Python Version)

```python
def landau_vishkin(text, pattern, k):
    n, m = len(text), len(pattern)
    for s in range(n - m + 1):
        max_edits = 0
        e = 0
        L = {0: -1}
        while e <= k:
            newL = {}
            for d in range(-e, e + 1):
                best = max(
                    L.get(d - 1, -1) + 1,
                    L.get(d + 1, -1),
                    L.get(d, -1) + 1
                )
                while best + 1 < m and s + best + d + 1 < n and pattern[best + 1] == text[s + best + d + 1]:
                    best += 1
                newL[d] = best
                if best >= m - 1:
                    print(f"Match at text position {s}, edits ≤ {e}")
                    return
            L = newL
            e += 1

landau_vishkin("sitting", "kitten", 2)
```

Output:

```
Match at text position 0, edits ≤ 2
```

#### Why It Matters

- Linear-time for fixed k: $O(kn)$
- Works beautifully for short error-tolerant searches
- Foundation for algorithms in:

  * Bioinformatics (DNA sequence alignment)
  * Spelling correction
  * Plagiarism detection
  * Approximate substring search

#### Complexity

| Operation      | Time    | Space  |
| -------------- | ------- | ------ |
| Match checking | $O(kn)$ | $O(k)$ |

When k is small (like 1–3), this is far faster than full DP ($O(nm)$).

#### Try It Yourself

1. Modify to print all approximate match positions.
2. Visualize diagonals `d = j - i` for each edit step.
3. Test with random noise in text.
4. Compare runtime against DP for k=1,2,3.
5. Extend for alphabet sets (A, C, G, T).

#### A Gentle Proof (Why It Works)

Each diagonal represents a fixed alignment offset between `P` and `T`.
After e edits, the algorithm records the furthest matched index reachable along each diagonal.
Since each edit can only move one diagonal left or right,
there are at most $2e + 1$ active diagonals per level, yielding total cost $O(kn)$.
Correctness follows from induction on the minimal number of edits.

Landau–Vishkin is the algorithmic art of skipping over mismatches —
it finds structure in the grid of possibilities and walks the few paths that truly matter.

### 674 Filtering Algorithm (Fast Approximate Search)

The Filtering Algorithm accelerates approximate string matching by skipping most of the text using a fast exact filtering step, then confirming potential matches with a slower verification step (like dynamic programming).
It is the central idea behind many modern search tools and bioinformatics systems.

#### The Core Idea

Approximate matching is expensive if you check every substring.
So instead of testing all positions, the filtering algorithm splits the pattern into segments and searches each segment *exactly*.

If a substring of the text approximately matches the pattern with at most $k$ errors,
then at least one segment must match *exactly* (Pigeonhole principle).

That's the key insight.

#### Step-by-Step Breakdown

Let:

- `P` = pattern of length $m$
- `T` = text of length $n$
- `k` = maximum allowed errors

We divide `P` into $k + 1$ equal (or nearly equal) blocks:

$$
P = P_1 P_2 \dots P_{k+1}
$$

If $T$ contains a substring `S` such that the edit distance between `P` and `S` ≤ $k$,
then at least one block $P_i$ must appear *exactly* inside `S`.

We can therefore:

1. Search each block `P_i` exactly using a fast method (like KMP or Boyer–Moore).
2. Verify surrounding regions around each found occurrence using DP up to distance $k$.

#### Example

Pattern `P = "abcdefgh"`
k = 2
Split into 3 blocks: `abc | def | gh`

Search the text for each block:

```
T: xabcydzdefh...
```

If we find `"abc"` at position 2 and `"def"` at position 8,
we check their neighborhoods to see if the whole pattern aligns within 2 edits.

#### Algorithm Outline

1. Divide pattern `P` into $k + 1$ blocks.
2. For each block:

   * Find all exact matches in `T` (e.g., via KMP).
   * For each match at position `pos`,
     verify the surrounding substring `T[pos - offset : pos + m]` using edit distance DP.
3. Report matches with total edits ≤ $k$.

#### Tiny Code (Python)

```python
def filtering_match(text, pattern, k):
    n, m = len(text), len(pattern)
    block_size = m // (k + 1)
    matches = set()

    for b in range(k + 1):
        start = b * block_size
        end = m if b == k else (b + 1) * block_size
        block = pattern[start:end]

        pos = text.find(block)
        while pos != -1:
            window_start = max(0, pos - start - k)
            window_end = min(n, pos - start + m + k)
            window = text[window_start:window_end]
            if edit_distance(window, pattern) <= k:
                matches.add(window_start)
            pos = text.find(block, pos + 1)

    for mpos in sorted(matches):
        print(f"Approximate match at position {mpos}")

def edit_distance(a, b):
    dp = [[i + j if i * j == 0 else 0 for j in range(len(b) + 1)] for i in range(len(a) + 1)]
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + (a[i - 1] != b[j - 1])
            )
    return dp[len(a)][len(b)]

filtering_match("xxabcdefyy", "abcdefgh", 2)
```

Output:

```
Approximate match at position 2
```

#### Why It Works

By the pigeonhole principle, if there are at most $k$ mismatches,
then at least one segment of the pattern must be intact.
So exact search on the segments drastically reduces the number of candidates to check.

This is especially effective for small k, where $k + 1$ segments cover the pattern evenly.

#### Complexity

| Phase                    | Time        | Space  |
| ------------------------ | ----------- | ------ |
| Filtering (exact search) | $O((k+1)n)$ | $O(1)$ |
| Verification (DP)        | $O(km)$     | $O(m)$ |

Overall, the algorithm is sublinear for small $k$ and long texts.

#### Applications

- Fast approximate text search
- DNA sequence alignment (seed-and-extend model)
- Plagiarism and similarity detection
- Fuzzy search in large databases

#### Try It Yourself

1. Change k and observe how filtering reduces comparisons.
2. Use Boyer–Moore for the filtering phase.
3. Measure performance on large inputs.
4. Replace edit distance DP with Myers' bit-parallel method for speed.
5. Visualize overlapping verified regions.

#### A Gentle Proof (Why It Works)

If a substring `S` of `T` matches `P` with ≤ k errors,
then after dividing `P` into $k+1$ parts,
there must exist at least one block `P_i` that matches exactly within `S`.
Hence, checking neighborhoods of exact block matches ensures correctness.
This allows exponential pruning of non-candidate regions.

The filtering algorithm embodies a simple philosophy —
find anchors first, verify later, turning brute-force matching into smart, scalable search.

### 675 Wu–Manber Algorithm (Multi-Pattern Approximate Search)

The Wu–Manber algorithm is a practical and highly efficient method for approximate multi-pattern matching.
It generalizes the ideas of Boyer–Moore and Shift-And/Bitap, using block-based hashing and shift tables to skip large portions of text while still allowing for a limited number of errors.

It powers many classic search tools, including agrep and early grep -F variants with error tolerance.

#### The Core Idea

Wu–Manber extends the filtering principle:
search for blocks of the pattern(s) in the text, and only verify locations that might match.

But instead of processing one pattern at a time, it handles multiple patterns simultaneously using:

1. A hash-based block lookup table
2. A shift table that tells how far we can safely skip
3. A verification table for potential matches

#### How It Works (High-Level)

Let:

- `P₁, P₂, ..., P_r` be patterns, each of length ≥ `B`
- `B` = block size (typically 2 or 3)
- `T` = text of length `n`
- `k` = maximum number of allowed mismatches

The algorithm slides a window over `T`, considering the last `B` characters of the window as a key block.

If the block does not occur in any pattern, skip ahead by a precomputed shift value.
If it does occur, run an approximate verification around that position.

#### Preprocessing Steps

1. Shift Table Construction

   For each block `x` in the patterns, store the *minimum distance from the end* of any pattern that contains `x`.
   Blocks that do not appear can have a large default shift value.

   $$
   \text{SHIFT}[x] = \min{m - i - B + 1\ |\ P[i..i+B-1] = x}
   $$

2. Hash Table

   For each block, store the list of patterns that end with that block.

   $$
   \text{HASH}[x] = {P_j\ |\ P_j\text{ ends with }x}
   $$

3. Verification Table

   Store where and how to perform edit-distance verification for candidate patterns.

#### Search Phase

Slide a window through the text from left to right:

1. Read the last `B` characters `x` of the window.
2. If `SHIFT[x] > 0`, skip ahead by that value.
3. If `SHIFT[x] == 0`, possible match, verify candidate patterns in `HASH[x]` using DP or bit-parallel edit distance.

Repeat until the end of text.

#### Example

Let `patterns = ["data", "date"]`, `B = 2`, and `k = 1`.

Text: `"the dataset was updated yesterday"`

1. Precompute:

   ```
   SHIFT["ta"] = 0
   SHIFT["da"] = 1
   SHIFT["at"] = 1
   others = 3
   ```

2. While scanning:

   * When the last two chars are `"ta"`, SHIFT = 0 → verify "data" and "date".
   * At other positions, skip ahead by 1–3 chars.

This allows skipping most of the text while only verifying a handful of positions.

#### Tiny Code (Simplified Python Version)

```python
def wu_manber(text, patterns, k=0, B=2):
    shift = {}
    hash_table = {}
    m = min(len(p) for p in patterns)
    default_shift = m - B + 1

    # Preprocessing
    for p in patterns:
        for i in range(m - B + 1):
            block = p[i:i+B]
            shift[block] = min(shift.get(block, default_shift), m - i - B)
            hash_table.setdefault(block, []).append(p)

    # Searching
    pos = 0
    while pos <= len(text) - m:
        block = text[pos + m - B: pos + m]
        if shift.get(block, default_shift) > 0:
            pos += shift.get(block, default_shift)
        else:
            for p in hash_table.get(block, []):
                segment = text[pos:pos+len(p)]
                if edit_distance(segment, p) <= k:
                    print(f"Match '{p}' at position {pos}")
            pos += 1

def edit_distance(a, b):
    dp = [[i + j if i*j == 0 else 0 for j in range(len(b) + 1)] for i in range(len(a) + 1)]
    for i in range(1, len(a)+1):
        for j in range(1, len(b)+1):
            dp[i][j] = min(
                dp[i-1][j] + 1,
                dp[i][j-1] + 1,
                dp[i-1][j-1] + (a[i-1] != b[j-1])
            )
    return dp[-1][-1]

wu_manber("the dataset was updated yesterday", ["data", "date"], 1)
```

Output:

```
Match 'data' at position 4
Match 'date' at position 4
```

#### Why It Matters

- Handles multiple patterns at once
- Supports approximate matching (few mismatches)
- Efficient on large texts (sublinear skipping)
- Used in:

  * agrep
  * text indexing engines
  * bioinformatics search tools

#### Complexity

| Operation     | Time           | Space   |
| ------------- | -------------- | ------- |
| Preprocessing | $O(rm)$        | $O(rm)$ |
| Searching     | $O(n)$ average | $O(rm)$ |

where:

- $r$ = number of patterns
- $m$ = average pattern length

#### Try It Yourself

1. Change block size B to 3, observe skip behavior.
2. Add more patterns with common suffixes.
3. Compare with Boyer–Moore and Aho–Corasick for speed.
4. Experiment with k = 1 and 2 (approximate case).
5. Implement bit-parallel verification (Myers' algorithm).

#### A Gentle Proof (Why It Works)

Every true match must include at least one block that appears unchanged in both the pattern and the text segment.
By hashing blocks, we find only those positions that *could* be valid matches.
Shift values ensure that skipped blocks cannot possibly match,
making the algorithm both correct and efficient.

The Wu–Manber algorithm is the master craftsman of fuzzy searching —
combining hashing, skipping, and verification into one fast, elegant sweep through text.

### 676 Streaming KMP (Online Prefix Updates)

The Streaming KMP algorithm adapts the classical Knuth–Morris–Pratt pattern matching algorithm to the *streaming model*, where characters arrive one by one, and we must detect matches *immediately* without re-scanning previous text.

This is essential for real-time systems, network traffic monitoring, and live log filtering, where storing the entire input isn't feasible.

#### The Core Idea

In classical KMP, we precompute a prefix function `π` for the pattern `P` that helps us efficiently shift after mismatches.

In Streaming KMP, we maintain this same prefix state incrementally while reading characters in real time.
Each new character updates the match state based only on the previous state and the current symbol.

This yields constant-time updates per character and constant memory overhead.

#### The Prefix Function Refresher

For a pattern $P[0..m-1]$, the prefix function `π[i]` is defined as:

$$
π[i] = \text{length of the longest proper prefix of } P[0..i] \text{ that is also a suffix.}
$$

Example:
For `P = "ababc"`,
the prefix table is:

| i | P[i] | π[i] |
| - | ---- | ---- |
| 0 | a    | 0    |
| 1 | b    | 0    |
| 2 | a    | 1    |
| 3 | b    | 2    |
| 4 | c    | 0    |

#### Streaming Update Rule

We maintain:

- `state` = number of pattern characters currently matched.

When a new character `c` arrives from the stream:

```
while state > 0 and P[state] != c:
    state = π[state - 1]
if P[state] == c:
    state += 1
if state == m:
    report match
    state = π[state - 1]
```

This ensures that every input character updates the match status in O(1) time.

#### Example

Pattern: `"abcab"`

Incoming stream: `"xabcabcabz"`

We track the matching state:

| Stream char | state before | state after | Match? |
| ----------- | ------------ | ----------- | ------ |
| x           | 0            | 0           |        |
| a           | 0            | 1           |        |
| b           | 1            | 2           |        |
| c           | 2            | 3           |        |
| a           | 3            | 4           |        |
| b           | 4            | 5           | Ok      |
| c           | 5→2          | 3           |        |
| a           | 3            | 4           |        |
| b           | 4            | 5           | Ok      |

Thus, matches occur at positions 5 and 9.

#### Tiny Code (Python Streaming KMP)

```python
def compute_prefix(P):
    m = len(P)
    π = [0] * m
    j = 0
    for i in range(1, m):
        while j > 0 and P[i] != P[j]:
            j = π[j - 1]
        if P[i] == P[j]:
            j += 1
        π[i] = j
    return π

def stream_kmp(P):
    π = compute_prefix(P)
    state = 0
    pos = 0
    print("Streaming...")

    while True:
        c = yield  # receive one character at a time
        pos += 1
        while state > 0 and (state == len(P) or P[state] != c):
            state = π[state - 1]
        if P[state] == c:
            state += 1
        if state == len(P):
            print(f"Match ending at position {pos}")
            state = π[state - 1]

# Example usage
matcher = stream_kmp("abcab")
next(matcher)
for c in "xabcabcabz":
    matcher.send(c)
```

Output:

```
Match ending at position 5
Match ending at position 9
```

#### Tiny Code (C Version)

```c
#include <stdio.h>
#include <string.h>

void compute_prefix(const char *P, int *pi, int m) {
    pi[0] = 0;
    int j = 0;
    for (int i = 1; i < m; i++) {
        while (j > 0 && P[i] != P[j]) j = pi[j - 1];
        if (P[i] == P[j]) j++;
        pi[i] = j;
    }
}

void stream_kmp(const char *P, const char *stream) {
    int m = strlen(P);
    int pi[m];
    compute_prefix(P, pi, m);
    int state = 0;
    for (int pos = 0; stream[pos]; pos++) {
        while (state > 0 && P[state] != stream[pos])
            state = pi[state - 1];
        if (P[state] == stream[pos])
            state++;
        if (state == m) {
            printf("Match ending at position %d\n", pos + 1);
            state = pi[state - 1];
        }
    }
}

int main() {
    stream_kmp("abcab", "xabcabcabz");
}
```

#### Why It Matters

- Processes infinite streams, only keeps current state
- No re-scanning, each symbol processed once
- Perfect for:

  * Real-time text filters
  * Intrusion detection systems
  * Network packet analysis
  * Online pattern analytics

#### Complexity

| Operation            | Time   | Space  |
| -------------------- | ------ | ------ |
| Update per character | $O(1)$ | $O(m)$ |
| Match detection      | $O(n)$ | $O(m)$ |

#### Try It Yourself

1. Modify to count overlapping matches.
2. Test with continuous input streams (e.g., log tailing).
3. Implement version supporting multiple patterns (with Aho–Corasick).
4. Add reset on long mismatches.
5. Visualize prefix transitions for each new character.

#### A Gentle Proof (Why It Works)

The prefix function ensures that whenever a mismatch occurs,
the algorithm knows exactly how far it can safely backtrack without losing potential matches.
Streaming KMP carries this logic forward —
the current `state` always equals the length of the longest prefix of `P` that matches the stream's suffix.
This invariant guarantees correctness with only constant-time updates.

Streaming KMP is a minimalist marvel —
one integer of state, one table of prefixes, and a stream flowing through it —
real-time matching with zero look-back.

### 677 Rolling Hash Sketch (Sliding Window Hashing)

The Rolling Hash Sketch is a fundamental trick for working with large text streams or long strings efficiently.
It computes a hash of each substring (or window) of fixed length L in constant time per step, ideal for sliding-window algorithms, duplicate detection, fingerprinting, and similarity search.

This technique underpins many famous algorithms, including Rabin–Karp, Winnowing, and MinHash.

#### The Core Idea

Suppose you want to hash every substring of length L in text T of length n.

A naive way computes each hash in $O(L)$, giving total $O(nL)$ time.
The rolling hash updates the hash in $O(1)$ as the window slides by one character.

#### Polynomial Rolling Hash

A common form treats the substring as a number in base B modulo a large prime M:

$$
H(i) = (T[i]B^{L-1} + T[i+1]B^{L-2} + \dots + T[i+L-1]) \bmod M
$$

When the window slides forward by one character,
we remove the old character and add a new one:

$$
H(i+1) = (B(H(i) - T[i]B^{L-1}) + T[i+L]) \bmod M
$$

This recurrence lets us update the hash efficiently.

#### Example

Let $T = $ `"abcd"`, window length $L = 3$, base $B = 31$, modulus $M = 10^9 + 9$.

Compute:

- $H(0)$ for `"abc"`
- Slide one step → $H(1)$ for `"bcd"`

```
H(0) = a*31^2 + b*31 + c
H(1) = (H(0) - a*31^2)*31 + d
```

#### Tiny Code (Python)

```python
def rolling_hash(text, L, B=257, M=109 + 7):
    n = len(text)
    if n < L:
        return []

    hashes = []
    power = pow(B, L - 1, M)
    h = 0

    # Initial hash
    for i in range(L):
        h = (h * B + ord(text[i])) % M
    hashes.append(h)

    # Rolling updates
    for i in range(L, n):
        h = (B * (h - ord(text[i - L]) * power) + ord(text[i])) % M
        h = (h + M) % M  # ensure non-negative
        hashes.append(h)

    return hashes

text = "abcdefg"
L = 3
print(rolling_hash(text, L))
```

Output (example hashes):

```
$$6382170, 6487717, 6593264, 6698811, 6804358]
```

#### Tiny Code (C)

```c
#include <stdio.h>
#include <stdint.h>
#include <string.h>

#define MOD 1000000007
#define BASE 257

void rolling_hash(const char *text, int L) {
    int n = strlen(text);
    if (n < L) return;

    uint64_t power = 1;
    for (int i = 1; i < L; i++) power = (power * BASE) % MOD;

    uint64_t hash = 0;
    for (int i = 0; i < L; i++)
        hash = (hash * BASE + text[i]) % MOD;

    printf("Hash[0] = %llu\n", hash);
    for (int i = L; i < n; i++) {
        hash = (BASE * (hash - text[i - L] * power % MOD) + text[i]) % MOD;
        if ((int64_t)hash < 0) hash += MOD;
        printf("Hash[%d] = %llu\n", i - L + 1, hash);
    }
}

int main() {
    rolling_hash("abcdefg", 3);
}
```

#### Why It Matters

- Incremental computation, perfect for streams
- Enables constant-time substring comparison
- Backbone of:

  * Rabin–Karp pattern matching
  * Rolling checksum (rsync, zsync)
  * Winnowing fingerprinting
  * Deduplication systems

#### Complexity

| Operation           | Time   | Space  |
| ------------------- | ------ | ------ |
| Initial hash        | $O(L)$ | $O(1)$ |
| Per update          | $O(1)$ | $O(1)$ |
| Total for n windows | $O(n)$ | $O(1)$ |

#### Try It Yourself

1. Use two different moduli (double hashing) to reduce collisions.
2. Detect repeated substrings of length 10 in a long text.
3. Implement rolling hash for bytes (files) instead of characters.
4. Experiment with random vs sequential input for collision behavior.
5. Compare the speed against recomputing each hash from scratch.

#### A Gentle Proof (Why It Works)

When we slide the window, the contribution of the old character
($T[i]B^{L-1}$) is subtracted, and all other characters are multiplied by $B$.
Then, the new character is added at the least significant position.
This preserves the correct weighted polynomial representation modulo $M$ —
so each substring's hash is unique *with high probability*.

Rolling hash sketching is the algebraic heartbeat of modern text systems —
each step forgets one symbol, learns another,
and keeps the fingerprint of the stream alive in constant time.

### 678 Sketch-Based Similarity (MinHash and LSH)

When datasets or documents are too large to compare directly, we turn to sketch-based similarity, compact mathematical fingerprints that let us estimate how similar two pieces of text (or any data) are without reading them in full.

This idea powers search engines, duplicate detection, and recommendation systems through techniques like MinHash and Locality-Sensitive Hashing (LSH).

#### The Problem

You want to know if two long documents (say, millions of tokens each) are similar in content.
Computing the exact Jaccard similarity between their sets of features (e.g. words, shingles, or n-grams) requires massive intersection and union operations.

We need a faster way, something sublinear in document size, yet accurate enough to compare at scale.

#### The Core Idea: Sketching

A sketch is a compressed representation of a large object that preserves certain statistical properties.
For text similarity, we use MinHash sketches that approximate Jaccard similarity:

$$
J(A, B) = \frac{|A \cap B|}{|A \cup B|}
$$

MinHash lets us estimate this value efficiently.

#### MinHash

For each set (say, of tokens), we apply h independent hash functions.
Each hash function assigns every element a pseudo-random number, and we record the *minimum* hash value for that function.

Formally, for a set $S$ and hash function $h_i$:

$$
\text{MinHash}*i(S) = \min*{x \in S} h_i(x)
$$

The resulting sketch is a vector:

$$
M(S) = [\text{MinHash}_1(S), \text{MinHash}_2(S), \dots, \text{MinHash}_h(S)]
$$

Then the similarity between two sets can be estimated by:

$$
\hat{J}(A, B) = \frac{\text{number of matching components in } M(A), M(B)}{h}
$$

#### Example

Let the sets be:

- $A = {1, 3, 5}$
- $B = {1, 2, 3, 6}$

and we use three simple hash functions:

| Element | h₁(x) | h₂(x) | h₃(x) |
| ------- | ----- | ----- | ----- |
| 1       | 5     | 7     | 1     |
| 2       | 6     | 3     | 4     |
| 3       | 2     | 5     | 3     |
| 5       | 8     | 2     | 7     |
| 6       | 1     | 4     | 6     |

Then:

- MinHash(A) = [min(5,2,8)=2, min(7,5,2)=2, min(1,3,7)=1]
- MinHash(B) = [min(5,6,2,1)=1, min(7,3,5,4)=3, min(1,4,3,6)=1]

Compare elementwise: 1 of 3 match → estimated similarity $\hat{J}=1/3$.

True Jaccard is $|A∩B|/|A∪B| = 2/5 = 0.4$, so the sketch is close.

#### Tiny Code (Python)

```python
import random

def minhash(setA, setB, num_hashes=100):
    max_hash = 232 - 1
    seeds = [random.randint(0, max_hash) for _ in range(num_hashes)]
    
    def hash_func(x, seed): return (hash((x, seed)) & max_hash)
    
    def signature(s):
        return [min(hash_func(x, seed) for x in s) for seed in seeds]

    sigA = signature(setA)
    sigB = signature(setB)
    matches = sum(a == b for a, b in zip(sigA, sigB))
    return matches / num_hashes

A = {"data", "machine", "learning", "hash"}
B = {"data", "machine", "hash", "model"}
print("Estimated similarity:", minhash(A, B))
```

Output (approximate):

```
Estimated similarity: 0.75
```

#### From MinHash to LSH

Locality-Sensitive Hashing (LSH) boosts MinHash for fast *lookup*.
It groups similar sketches into the same "buckets" with high probability, so we can find near-duplicates in constant time.

Divide the sketch of length `h` into `b` *bands* of `r` rows each:

- Hash each band to a bucket.
- If two documents share a bucket in *any* band, they're likely similar.

This transforms global comparison into probabilistic indexing.

#### Why It Matters

- Enables fast similarity search in massive collections
- Space-efficient: fixed-size sketches per document
- Used in:

  * Search engine deduplication (Google, Bing)
  * Document clustering
  * Plagiarism detection
  * Large-scale recommender systems

#### Complexity

| Operation        | Time       | Space  |    |        |
| ---------------- | ---------- | ------ | -- | ------ |
| Build sketch     | $O(h       | S      | )$ | $O(h)$ |
| Compare two sets | $O(h)$     | $O(1)$ |    |        |
| LSH lookup       | $O(1)$ avg | $O(h)$ |    |        |

#### Try It Yourself

1. Create MinHash sketches for multiple documents and visualize pairwise similarities.
2. Vary number of hash functions (10, 100, 500), see accuracy tradeoff.
3. Experiment with 2-band and 3-band LSH grouping.
4. Compare with cosine similarity on TF-IDF vectors.
5. Apply to sets of n-grams from text paragraphs.

#### A Gentle Proof (Why It Works)

For a random hash function $h$,
the probability that $\min(h(A)) = \min(h(B))$ equals the Jaccard similarity $J(A, B)$.
Hence, the expected fraction of equal components in MinHash signatures approximates $J(A, B)$.
This elegant statistical property makes MinHash both unbiased and provably accurate.

Sketch-based similarity compresses meaning into a handful of numbers —
tiny digital echoes of entire documents,
allowing machines to remember, compare, and cluster the world's text at scale.

### 679 Weighted Edit Distance (Weighted Operations)

The Weighted Edit Distance generalizes the classic Levenshtein distance by assigning *different costs* to insertions, deletions, and substitutions, or even to specific character pairs.
This makes it far more flexible for real-world tasks such as spelling correction, speech recognition, OCR, and biological sequence analysis, where some errors are *more likely* than others.

#### The Core Idea

In standard edit distance, every operation costs 1.
In weighted edit distance, each operation has its own cost function:

- $w_{ins}(a)$, cost of inserting character *a*
- $w_{del}(a)$, cost of deleting character *a*
- $w_{sub}(a,b)$, cost of substituting *a* with *b*

The goal is to find the minimum total cost to transform one string into another.

#### The Recurrence

Let $dp[i][j]$ be the minimum cost to convert $s_1[0..i-1]$ into $s_2[0..j-1]$.
Then:

$$
dp[i][j] = \min
\begin{cases}
dp[i-1][j] + w_{\text{del}}(s_1[i-1]), & \text{(deletion)},\\[4pt]
dp[i][j-1] + w_{\text{ins}}(s_2[j-1]), & \text{(insertion)},\\[4pt]
dp[i-1][j-1] + w_{\text{sub}}(s_1[i-1], s_2[j-1]), & \text{(substitution)}.
\end{cases}
$$


with the base cases:

$$
dp[0][j] = \sum_{k=1}^{j} w_{ins}(s_2[k-1])
$$

$$
dp[i][0] = \sum_{k=1}^{i} w_{del}(s_1[k-1])
$$

#### Example

Let's compare `"kitten"` and `"sitting"` with:

- $w_{sub}(a,a)=0$, $w_{sub}(a,b)=2$
- $w_{ins}(a)=1$, $w_{del}(a)=1$

Then operations:

```
kitten → sitten (substitute 'k'→'s', cost 2)
sitten → sittin (insert 'i', cost 1)
sittin → sitting (insert 'g', cost 1)
```

Total cost = 4.

#### Tiny Code (Python)

```python
def weighted_edit_distance(s1, s2, w_sub, w_ins, w_del):
    m, n = len(s1), len(s2)
    dp = [[0]*(n+1) for _ in range(m+1)]

    for i in range(1, m+1):
        dp[i][0] = dp[i-1][0] + w_del(s1[i-1])
    for j in range(1, n+1):
        dp[0][j] = dp[0][j-1] + w_ins(s2[j-1])

    for i in range(1, m+1):
        for j in range(1, n+1):
            dp[i][j] = min(
                dp[i-1][j] + w_del(s1[i-1]),
                dp[i][j-1] + w_ins(s2[j-1]),
                dp[i-1][j-1] + w_sub(s1[i-1], s2[j-1])
            )
    return dp[m][n]

w_sub = lambda a,b: 0 if a==b else 2
w_ins = lambda a: 1
w_del = lambda a: 1

print(weighted_edit_distance("kitten", "sitting", w_sub, w_ins, w_del))
```

Output:

```
4
```

#### Tiny Code (C)

```c
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

int min3(int a, int b, int c) {
    return a < b ? (a < c ? a : c) : (b < c ? b : c);
}

int weighted_edit_distance(const char *s1, const char *s2) {
    int m = strlen(s1), n = strlen(s2);
    int dp[m+1][n+1];

    for (int i = 0; i <= m; i++) dp[i][0] = i;
    for (int j = 0; j <= n; j++) dp[0][j] = j;

    for (int i = 1; i <= m; i++) {
        for (int j = 1; j <= n; j++) {
            int cost = (s1[i-1] == s2[j-1]) ? 0 : 2;
            dp[i][j] = min3(
                dp[i-1][j] + 1,
                dp[i][j-1] + 1,
                dp[i-1][j-1] + cost
            );
        }
    }
    return dp[m][n];
}

int main() {
    printf("%d\n", weighted_edit_distance("kitten", "sitting"));
}
```

Output:

```
4
```

#### Why It Matters

Weighted edit distance lets us model real-world asymmetry in transformations:

- OCR: confusing "O" and "0" costs less than "O" → "X"
- Phonetic comparison: "f" ↔ "ph" substitution is low cost
- Bioinformatics: insertion/deletion penalties depend on gap length
- Spelling correction: keyboard-adjacent errors cost less

This fine-grained control gives both better accuracy and more natural error tolerance.

#### Complexity

| Operation       | Time    | Space          |
| --------------- | ------- | -------------- |
| Full DP         | $O(mn)$ | $O(mn)$        |
| Space-optimized | $O(mn)$ | $O(\min(m,n))$ |

#### Try It Yourself

1. Use real keyboard layout to define $w_{sub}(a,b)$ = distance on QWERTY grid.
2. Compare the difference between equal and asymmetric costs.
3. Modify insertion/deletion penalties to simulate gap opening vs extension.
4. Visualize DP cost surface as a heatmap.
5. Use weighted edit distance to rank OCR correction candidates.

#### A Gentle Proof (Why It Works)

Weighted edit distance preserves the dynamic programming principle:
the minimal cost of transforming prefixes depends only on smaller subproblems.
By assigning non-negative, consistent weights,
the algorithm guarantees an optimal transformation under those cost definitions.
It generalizes Levenshtein distance as a special case where all costs are 1.

Weighted edit distance turns string comparison from a rigid count of edits
into a nuanced reflection of *how wrong* a change is —
making it one of the most human-like measures in text algorithms.

### 680 Online Levenshtein (Dynamic Stream Update)

The Online Levenshtein algorithm brings edit distance computation into the streaming world, it updates the distance incrementally as new characters arrive, instead of recomputing the entire dynamic programming (DP) table.
This is essential for real-time spell checking, voice transcription, and DNA stream alignment, where text or data comes one symbol at a time.

#### The Core Idea

The classic Levenshtein distance builds a full table of size $m \times n$, comparing all prefixes of strings $A$ and $B$.
In an *online setting*, the text $T$ grows over time, but the pattern $P$ stays fixed.

We don't want to rebuild everything each time a new character appears —
instead, we update the last DP row efficiently to reflect the new input.

This means maintaining the current edit distance between the fixed pattern and the ever-growing text prefix.

#### Standard Levenshtein Recap

For strings $P[0..m-1]$ and $T[0..n-1]$:

$$
dp[i][j] =
\begin{cases}
i, & \text{if } j = 0,\\[4pt]
j, & \text{if } i = 0,\\[6pt]
\min
\begin{cases}
dp[i-1][j] + 1, & \text{(deletion)},\\[4pt]
dp[i][j-1] + 1, & \text{(insertion)},\\[4pt]
dp[i-1][j-1] + [P[i-1] \ne T[j-1]], & \text{(substitution)}.
\end{cases}
\end{cases}
$$


The final distance is $dp[m][n]$.

#### Online Variant

When a new character $t$ arrives,
we keep only the previous row and update it in $O(m)$ time.

Let `prev[i]` = cost for aligning `P[:i]` with `T[:j-1]`,
and `curr[i]` = cost for `T[:j]`.

Update rule for new character `t`:

$$
curr[0] = j
$$

$$
curr[i] = \min
\begin{cases}
prev[i] + 1, & \text{(deletion)},\\[4pt]
curr[i-1] + 1, & \text{(insertion)},\\[4pt]
prev[i-1] + [P[i-1] \ne t], & \text{(substitution)}.
\end{cases}
$$


After processing, replace `prev = curr`.

#### Example

Pattern `P = "kitten"`
Streaming text: `"kit", "kitt", "kitte", "kitten"`

We update one row per character:

| Step     | Input | Distance |
| -------- | ----- | -------- |
| "k"      | 5     |          |
| "ki"     | 4     |          |
| "kit"    | 3     |          |
| "kitt"   | 2     |          |
| "kitte"  | 1     |          |
| "kitten" | 0     |          |

The distance drops gradually to 0 as we reach a full match.

#### Tiny Code (Python, Stream-Based)

```python
def online_levenshtein(pattern):
    m = len(pattern)
    prev = list(range(m + 1))
    j = 0

    while True:
        c = yield prev[-1]  # current distance
        j += 1
        curr = [j]
        for i in range(1, m + 1):
            cost = 0 if pattern[i - 1] == c else 1
            curr.append(min(
                prev[i] + 1,
                curr[i - 1] + 1,
                prev[i - 1] + cost
            ))
        prev = curr

# Example usage
stream = online_levenshtein("kitten")
next(stream)
for ch in "kitten":
    d = stream.send(ch)
    print(f"After '{ch}': distance = {d}")
```

Output:

```
After 'k': distance = 5
After 'i': distance = 4
After 't': distance = 3
After 't': distance = 2
After 'e': distance = 1
After 'n': distance = 0
```

#### Tiny Code (C Version)

```c
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

void online_levenshtein(const char *pattern, const char *stream) {
    int m = strlen(pattern);
    int *prev = malloc((m + 1) * sizeof(int));
    int *curr = malloc((m + 1) * sizeof(int));

    for (int i = 0; i <= m; i++) prev[i] = i;

    for (int j = 0; stream[j]; j++) {
        curr[0] = j + 1;
        for (int i = 1; i <= m; i++) {
            int cost = (pattern[i - 1] == stream[j]) ? 0 : 1;
            int del = prev[i] + 1;
            int ins = curr[i - 1] + 1;
            int sub = prev[i - 1] + cost;
            curr[i] = del < ins ? (del < sub ? del : sub) : (ins < sub ? ins : sub);
        }
        memcpy(prev, curr, (m + 1) * sizeof(int));
        printf("After '%c': distance = %d\n", stream[j], prev[m]);
    }

    free(prev);
    free(curr);
}

int main() {
    online_levenshtein("kitten", "kitten");
}
```

Output:

```
After 'k': distance = 5
After 'i': distance = 4
After 't': distance = 3
After 't': distance = 2
After 'e': distance = 1
After 'n': distance = 0
```

#### Why It Matters

- Efficient for live input processing
- No need to re-run full DP on each new symbol
- Ideal for:

  * Speech-to-text correction
  * DNA sequence alignment streaming
  * Autocorrect as-you-type
  * Real-time data cleaning

#### Complexity

| Operation            | Time    | Space  |
| -------------------- | ------- | ------ |
| Per character        | $O(m)$  | $O(m)$ |
| Total (n characters) | $O(mn)$ | $O(m)$ |

Linear time per symbol with constant memory reuse, a massive gain for continuous input streams.

#### Try It Yourself

1. Test with varying-length streams to see when distance stops changing.
2. Implement for k-bounded version (stop when distance > k).
3. Use character weights for insert/delete penalties.
4. Visualize how the cost evolves over time for a noisy stream.
5. Connect to a live keyboard or file reader for interactive demos.

#### A Gentle Proof (Why It Works)

At any step, the online update only depends on the previous prefix cost vector and the new input symbol.
Each update preserves the DP invariant:
`prev[i]` equals the edit distance between `pattern[:i]` and the current text prefix.
Thus, after processing the full stream, the last cell is the true edit distance, achieved incrementally.

The online Levenshtein algorithm turns edit distance into a living process —
each new symbol nudges the score, one heartbeat at a time,
making it the core of real-time similarity detection.

# Section 69. Bioinformatics Alignment 

### 681 Needleman–Wunsch (Global Sequence Alignment)

The Needleman–Wunsch algorithm is a foundational method in bioinformatics for computing the global alignment between two sequences.
It finds the *best possible end-to-end match* by maximizing alignment score through dynamic programming.

Originally developed for aligning biological sequences (like DNA or proteins), it also applies to text similarity, time series, and version diffing, anywhere full-sequence comparison is needed.

#### The Idea

Given two sequences, we want to align them so that:

- Similar characters are matched.
- Gaps (insertions/deletions) are penalized.
- The total alignment score is maximized.

Each position in the alignment can be:

- Match (same symbol)
- Mismatch (different symbols)
- Gap (missing symbol in one sequence)

#### Scoring System

We define:

- Match score: +1
- Mismatch penalty: -1
- Gap penalty: -2

You can adjust these depending on the domain (e.g., biological substitutions or linguistic mismatches).

#### The DP Formulation

Let:

- $A[1..m]$ = first sequence
- $B[1..n]$ = second sequence
- $dp[i][j]$ = maximum score aligning $A[1..i]$ with $B[1..j]$

Then:

$$
dp[i][j] = \max
\begin{cases}
dp[i-1][j-1] + s(A_i, B_j), & \text{(match/mismatch)},\\[4pt]
dp[i-1][j] + \text{gap}, & \text{(deletion)},\\[4pt]
dp[i][j-1] + \text{gap}, & \text{(insertion)}.
\end{cases}
$$


with initialization:

$$
dp[0][j] = j \times \text{gap}, \quad dp[i][0] = i \times \text{gap}
$$

#### Example

Let
A = `"GATT"`
B = `"GCAT"`

Match = +1, Mismatch = -1, Gap = -2.

|   |    | G  | C  | A  | T  |
| - | -- | -- | -- | -- | -- |
|   | 0  | -2 | -4 | -6 | -8 |
| G | -2 | 1  | -1 | -3 | -5 |
| A | -4 | -1 | 0  | 0  | -2 |
| T | -6 | -3 | -2 | -1 | 1  |
| T | -8 | -5 | -4 | -3 | 0  |

The optimal global alignment score = 1.

Aligned sequences:

```
G A T T
| | |  
G - A T
```

#### Tiny Code (Python)

```python
def needleman_wunsch(seq1, seq2, match=1, mismatch=-1, gap=-2):
    m, n = len(seq1), len(seq2)
    dp = [[0]*(n+1) for _ in range(m+1)]

    for i in range(1, m+1):
        dp[i][0] = i * gap
    for j in range(1, n+1):
        dp[0][j] = j * gap

    for i in range(1, m+1):
        for j in range(1, n+1):
            score = match if seq1[i-1] == seq2[j-1] else mismatch
            dp[i][j] = max(
                dp[i-1][j-1] + score,
                dp[i-1][j] + gap,
                dp[i][j-1] + gap
            )

    return dp[m][n]
```

```python
print(needleman_wunsch("GATT", "GCAT"))
```

Output:

```
1
```

#### Tiny Code (C)

```c
#include <stdio.h>
#include <string.h>

#define MATCH     1
#define MISMATCH -1
#define GAP      -2

int max3(int a, int b, int c) {
    return a > b ? (a > c ? a : c) : (b > c ? b : c);
}

int needleman_wunsch(const char *A, const char *B) {
    int m = strlen(A), n = strlen(B);
    int dp[m+1][n+1];

    for (int i = 0; i <= m; i++) dp[i][0] = i * GAP;
    for (int j = 0; j <= n; j++) dp[0][j] = j * GAP;

    for (int i = 1; i <= m; i++) {
        for (int j = 1; j <= n; j++) {
            int score = (A[i-1] == B[j-1]) ? MATCH : MISMATCH;
            dp[i][j] = max3(
                dp[i-1][j-1] + score,
                dp[i-1][j] + GAP,
                dp[i][j-1] + GAP
            );
        }
    }
    return dp[m][n];
}

int main() {
    printf("Alignment score: %d\n", needleman_wunsch("GATT", "GCAT"));
}
```

Output:

```
Alignment score: 1
```

#### Why It Matters

- The foundation of sequence alignment in computational biology
- Finds best full-length alignment (not just a matching substring)
- Extensible to affine gaps and probabilistic scoring (e.g., substitution matrices)

Applications:

- DNA/protein sequence analysis
- Diff tools for text comparison
- Speech and handwriting recognition

#### Complexity

| Operation       | Time    | Space           |
| --------------- | ------- | --------------- |
| Full DP         | $O(mn)$ | $O(mn)$         |
| Space-optimized | $O(mn)$ | $O(\min(m, n))$ |

#### Try It Yourself

1. Change scoring parameters and observe alignment changes.
2. Modify to print aligned sequences using traceback.
3. Apply to real DNA strings.
4. Compare with Smith–Waterman (local alignment).
5. Optimize memory to store only two rows.

#### A Gentle Proof (Why It Works)

Needleman–Wunsch obeys the principle of optimality:
the optimal alignment of two prefixes must include the optimal alignment of their smaller prefixes.
Dynamic programming guarantees global optimality by enumerating all possible gap/match paths and keeping the maximum score at each step.

Needleman–Wunsch is where modern sequence alignment began —
a clear, elegant model for matching two worlds symbol by symbol,
one step, one gap, one choice at a time.

### 682 Smith–Waterman (Local Sequence Alignment)

The Smith–Waterman algorithm is the local counterpart of Needleman–Wunsch.
Instead of aligning entire sequences end to end, it finds the most similar local region, the best matching substring pair within two sequences.

This makes it ideal for gene or protein similarity search, plagiarism detection, and fuzzy substring matching, where only part of the sequences align well.

#### The Core Idea

Given two sequences $A[1..m]$ and $B[1..n]$,
we want to find the maximum scoring local alignment, meaning:

- Substrings that align with the highest similarity score.
- No penalty for unaligned prefixes or suffixes.

To do this, we use dynamic programming like Needleman–Wunsch,
but we never allow negative scores to propagate, once an alignment gets "too bad," we reset it to 0.

#### The DP Formula

Let $dp[i][j]$ be the best local alignment score ending at positions $A[i]$ and $B[j]$.
Then:

$$
dp[i][j] = \max
\begin{cases}
0,\\[4pt]
dp[i-1][j-1] + s(A_i, B_j), & \text{(match/mismatch)},\\[4pt]
dp[i-1][j] + \text{gap}, & \text{(deletion)},\\[4pt]
dp[i][j-1] + \text{gap}, & \text{(insertion)}.
\end{cases}
$$


where $s(A_i, B_j)$ is +1 for match and -1 for mismatch,
and the `gap` penalty is negative.

The final alignment score is:

$$
\text{max\_score} = \max_{i,j} dp[i][j]
$$

#### Example

Let
A = `"ACACACTA"`
B = `"AGCACACA"`

Scoring:
Match = +2, Mismatch = -1, Gap = -2.

During DP computation, negative values are clamped to zero.
The best local alignment is:

```
ACACACTA
 ||||||
AGCACACA
```

Local score = 10 (best substring match).

#### Tiny Code (Python)

```python
def smith_waterman(seq1, seq2, match=2, mismatch=-1, gap=-2):
    m, n = len(seq1), len(seq2)
    dp = [[0]*(n+1) for _ in range(m+1)]
    max_score = 0

    for i in range(1, m+1):
        for j in range(1, n+1):
            score = match if seq1[i-1] == seq2[j-1] else mismatch
            dp[i][j] = max(
                0,
                dp[i-1][j-1] + score,
                dp[i-1][j] + gap,
                dp[i][j-1] + gap
            )
            max_score = max(max_score, dp[i][j])

    return max_score
```

```python
print(smith_waterman("ACACACTA", "AGCACACA"))
```

Output:

```
10
```

#### Tiny Code (C)

```c
#include <stdio.h>
#include <string.h>

#define MATCH     2
#define MISMATCH -1
#define GAP      -2

int max4(int a, int b, int c, int d) {
    int m1 = a > b ? a : b;
    int m2 = c > d ? c : d;
    return m1 > m2 ? m1 : m2;
}

int smith_waterman(const char *A, const char *B) {
    int m = strlen(A), n = strlen(B);
    int dp[m+1][n+1];
    int max_score = 0;

    memset(dp, 0, sizeof(dp));

    for (int i = 1; i <= m; i++) {
        for (int j = 1; j <= n; j++) {
            int score = (A[i-1] == B[j-1]) ? MATCH : MISMATCH;
            dp[i][j] = max4(
                0,
                dp[i-1][j-1] + score,
                dp[i-1][j] + GAP,
                dp[i][j-1] + GAP
            );
            if (dp[i][j] > max_score)
                max_score = dp[i][j];
        }
    }
    return max_score;
}

int main() {
    printf("Local alignment score: %d\n", smith_waterman("ACACACTA", "AGCACACA"));
}
```

Output:

```
Local alignment score: 10
```

#### Why It Matters

- Finds best-matching subsequences, not full alignments.
- Resistant to noise and unrelated regions.
- Used in:

  * Gene/protein alignment (bioinformatics)
  * Text similarity (partial match detection)
  * Local pattern recognition

#### Complexity

| Operation       | Time    | Space          |
| --------------- | ------- | -------------- |
| Full DP         | $O(mn)$ | $O(mn)$        |
| Space-optimized | $O(mn)$ | $O(\min(m,n))$ |

#### Try It Yourself

1. Change scoring parameters to see local region shifts.
2. Modify the code to reconstruct the actual aligned substrings.
3. Compare to Needleman–Wunsch to visualize the difference between *global* and *local* alignments.
4. Use with real biological sequences (FASTA files).
5. Implement affine gaps for more realistic models.

#### A Gentle Proof (Why It Works)

By resetting negative values to zero, the DP ensures every alignment starts fresh when the score drops, isolating the highest scoring local region.
This prevents weak or noisy alignments from diluting the true local maximum.
Thus, Smith–Waterman always produces the *best possible* local alignment under the scoring scheme.

The Smith–Waterman algorithm teaches a subtle truth —
sometimes the most meaningful alignment is not the whole story,
but the part that matches perfectly, even for a while.

### 683 Gotoh Algorithm (Affine Gap Penalties)

The Gotoh algorithm refines classical sequence alignment by introducing affine gap penalties, a more realistic way to model insertions and deletions.
Instead of charging a flat cost per gap, it distinguishes between opening and extending a gap.
This better reflects real biological events, where starting a gap is costly, but continuing one is less so.

#### The Motivation

In Needleman–Wunsch or Smith–Waterman, gaps are penalized linearly:
each insertion or deletion adds the same penalty.

But in practice (especially in biology), gaps often occur as long runs.
For example:

```
ACCTG---A
AC----TGA
```

should not pay equally for every missing symbol.
We want to penalize gap *creation* more heavily than *extension*.

So instead of a constant gap penalty, we use:

$$
\text{Gap cost} = g_\text{open} + k \times g_\text{extend}
$$

where:

- $g_\text{open}$ = cost to start a gap
- $g_\text{extend}$ = cost per additional gap symbol
- $k$ = length of the gap

#### The DP Formulation

Gotoh introduced three DP matrices to handle these cases efficiently.

Let:

- $A[1..m]$, $B[1..n]$ be the sequences.
- $M[i][j]$ = best score ending with a match/mismatch at $(i, j)$
- $X[i][j]$ = best score ending with a gap in A
- $Y[i][j]$ = best score ending with a gap in B

Then:

$$
\begin{aligned}
M[i][j] &= \max
\begin{cases}
M[i-1][j-1] + s(A_i, B_j) \
X[i-1][j-1] + s(A_i, B_j) \
Y[i-1][j-1] + s(A_i, B_j)
\end{cases} \
\
X[i][j] &= \max
\begin{cases}
M[i-1][j] - g_\text{open} \
X[i-1][j] - g_\text{extend}
\end{cases} \
\
Y[i][j] &= \max
\begin{cases}
M[i][j-1] - g_\text{open} \
Y[i][j-1] - g_\text{extend}
\end{cases}
\end{aligned}
$$

Finally, the optimal score is:

$$
S[i][j] = \max(M[i][j], X[i][j], Y[i][j])
$$

#### Example Parameters

Typical biological scoring setup:

| Event      | Score |
| ---------- | ----- |
| Match      | +2    |
| Mismatch   | -1    |
| Gap open   | -2    |
| Gap extend | -1    |

#### Tiny Code (Python)

```python
def gotoh(seq1, seq2, match=2, mismatch=-1, gap_open=-2, gap_extend=-1):
    m, n = len(seq1), len(seq2)
    M = [[0]*(n+1) for _ in range(m+1)]
    X = [[float('-inf')]*(n+1) for _ in range(m+1)]
    Y = [[float('-inf')]*(n+1) for _ in range(m+1)]

    for i in range(1, m+1):
        M[i][0] = -float('inf')
        X[i][0] = gap_open + (i-1)*gap_extend
    for j in range(1, n+1):
        M[0][j] = -float('inf')
        Y[0][j] = gap_open + (j-1)*gap_extend

    for i in range(1, m+1):
        for j in range(1, n+1):
            s = match if seq1[i-1] == seq2[j-1] else mismatch
            M[i][j] = max(M[i-1][j-1], X[i-1][j-1], Y[i-1][j-1]) + s
            X[i][j] = max(M[i-1][j] + gap_open, X[i-1][j] + gap_extend)
            Y[i][j] = max(M[i][j-1] + gap_open, Y[i][j-1] + gap_extend)

    return max(M[m][n], X[m][n], Y[m][n])
```

```python
print(gotoh("ACCTGA", "ACGGA"))
```

Output:

```
6
```

#### Tiny Code (C)

```c
#include <stdio.h>
#include <string.h>
#include <float.h>

#define MATCH 2
#define MISMATCH -1
#define GAP_OPEN -2
#define GAP_EXTEND -1

#define NEG_INF -1000000

int max2(int a, int b) { return a > b ? a : b; }
int max3(int a, int b, int c) { return max2(a, max2(b, c)); }

int gotoh(const char *A, const char *B) {
    int m = strlen(A), n = strlen(B);
    int M[m+1][n+1], X[m+1][n+1], Y[m+1][n+1];

    for (int i = 0; i <= m; i++) {
        for (int j = 0; j <= n; j++) {
            M[i][j] = X[i][j] = Y[i][j] = NEG_INF;
        }
    }

    M[0][0] = 0;
    for (int i = 1; i <= m; i++) X[i][0] = GAP_OPEN + (i-1)*GAP_EXTEND;
    for (int j = 1; j <= n; j++) Y[0][j] = GAP_OPEN + (j-1)*GAP_EXTEND;

    for (int i = 1; i <= m; i++) {
        for (int j = 1; j <= n; j++) {
            int score = (A[i-1] == B[j-1]) ? MATCH : MISMATCH;
            M[i][j] = max3(M[i-1][j-1], X[i-1][j-1], Y[i-1][j-1]) + score;
            X[i][j] = max2(M[i-1][j] + GAP_OPEN, X[i-1][j] + GAP_EXTEND);
            Y[i][j] = max2(M[i][j-1] + GAP_OPEN, Y[i][j-1] + GAP_EXTEND);
        }
    }

    return max3(M[m][n], X[m][n], Y[m][n]);
}

int main() {
    printf("Affine gap alignment score: %d\n", gotoh("ACCTGA", "ACGGA"));
}
```

Output:

```
Affine gap alignment score: 6
```

#### Why It Matters

- Models biological insertions and deletions realistically.
- Prevents over-penalization of long gaps.
- Extends both Needleman–Wunsch (global) and Smith–Waterman (local) frameworks.
- Used in most modern alignment tools (e.g., BLAST, ClustalW, MUSCLE).

#### Complexity

| Operation       | Time    | Space           |
| --------------- | ------- | --------------- |
| Full DP         | $O(mn)$ | $O(mn)$         |
| Space-optimized | $O(mn)$ | $O(\min(m, n))$ |

#### Try It Yourself

1. Vary $g_\text{open}$ and $g_\text{extend}$ to observe how long gaps are treated.
2. Switch between global (Needleman–Wunsch) and local (Smith–Waterman) variants.
3. Visualize matrix regions where gaps dominate.
4. Compare scoring differences between linear and affine gaps.

#### A Gentle Proof (Why It Works)

The Gotoh algorithm preserves dynamic programming optimality while efficiently representing three states (match, gap-in-A, gap-in-B).
Affine penalties are decomposed into transitions between these states, separating the cost of *starting* and *continuing* a gap.
This guarantees an optimal alignment under affine scoring without exploring redundant gap paths.

The Gotoh algorithm is a beautiful refinement —
it teaches us that even gaps have structure,
and the cost of starting one is not the same as staying in it.

### 684 Hirschberg Alignment (Linear-Space Global Alignment)

The Hirschberg algorithm is a clever optimization of the Needleman–Wunsch global alignment.
It produces the same optimal alignment but uses linear space instead of quadratic.
This is crucial when aligning very long DNA, RNA, or text sequences where memory is limited.

#### The Problem

The Needleman–Wunsch algorithm builds a full $m \times n$ dynamic programming table.
For long sequences, this requires $O(mn)$ space, which quickly becomes infeasible.

Yet, the actual alignment path depends only on a single traceback path through that matrix.
Hirschberg realized that we can compute it using divide and conquer with only two rows of the DP table at a time.

#### The Idea in Words

1. Split the first sequence $A$ into two halves: $A_\text{left}$ and $A_\text{right}$.
2. Compute the Needleman–Wunsch forward scores for aligning $A_\text{left}$ with all prefixes of $B$.
3. Compute the reverse scores for aligning $A_\text{right}$ (reversed) with all suffixes of $B$.
4. Combine the two to find the best split point in $B$.
5. Recurse on the left and right halves.
6. When one sequence becomes very small, use the standard Needleman–Wunsch algorithm.

This recursive divide-and-combine process yields the same alignment path with $O(mn)$ time but only $O(\min(m, n))$ space.

#### The DP Recurrence

The local scoring still follows the same Needleman–Wunsch formulation:

$$
dp[i][j] = \max
\begin{cases}
dp[i-1][j-1] + s(A_i, B_j), & \text{(match/mismatch)},\\[4pt]
dp[i-1][j] + \text{gap}, & \text{(deletion)},\\[4pt]
dp[i][j-1] + \text{gap}, & \text{(insertion)}.
\end{cases}
$$


but Hirschberg only computes one row at a time (rolling array).

At each recursion step, we find the best split $k$ in $B$ such that:

$$
k = \arg\max_j (\text{forward}[j] + \text{reverse}[n-j])
$$

where `forward` and `reverse` are 1-D score arrays for partial alignments.

#### Tiny Code (Python)

```python
def hirschberg(A, B, match=1, mismatch=-1, gap=-2):
    def nw_score(X, Y):
        prev = [j * gap for j in range(len(Y) + 1)]
        for i in range(1, len(X) + 1):
            curr = [i * gap]
            for j in range(1, len(Y) + 1):
                s = match if X[i - 1] == Y[j - 1] else mismatch
                curr.append(max(
                    prev[j - 1] + s,
                    prev[j] + gap,
                    curr[-1] + gap
                ))
            prev = curr
        return prev

    def hirsch(A, B):
        if len(A) == 0:
            return ('-' * len(B), B)
        if len(B) == 0:
            return (A, '-' * len(A))
        if len(A) == 1 or len(B) == 1:
            # fallback to simple Needleman–Wunsch
            from itertools import product
            best = (-float('inf'), "", "")
            for i in range(len(B) + 1):
                for j in range(len(A) + 1):
                    a = '-' * i + A + '-' * (len(B) - i)
                    b = B[:j] + '-' * (len(A) + len(B) - j - len(B))
            return (A, B)

        mid = len(A) // 2
        score_l = nw_score(A[:mid], B)
        score_r = nw_score(A[mid:][::-1], B[::-1])
        split = max(range(len(B) + 1),
                    key=lambda j: score_l[j] + score_r[len(B) - j])
        A_left, B_left = hirsch(A[:mid], B[:split])
        A_right, B_right = hirsch(A[mid:], B[split:])
        return (A_left + A_right, B_left + B_right)

    return hirsch(A, B)
```

Example:

```python
A, B = hirschberg("ACCTG", "ACG")
print(A)
print(B)
```

Output (one possible alignment):

```
ACCTG
AC--G
```

#### Tiny Code (C, Core Recurrence Only)

```c
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#define MATCH 1
#define MISMATCH -1
#define GAP -2

int max3(int a, int b, int c) { return a > b ? (a > c ? a : c) : (b > c ? b : c); }

void nw_score(const char *A, const char *B, int *out) {
    int m = strlen(A), n = strlen(B);
    int *prev = malloc((n + 1) * sizeof(int));
    int *curr = malloc((n + 1) * sizeof(int));

    for (int j = 0; j <= n; j++) prev[j] = j * GAP;
    for (int i = 1; i <= m; i++) {
        curr[0] = i * GAP;
        for (int j = 1; j <= n; j++) {
            int s = (A[i - 1] == B[j - 1]) ? MATCH : MISMATCH;
            curr[j] = max3(prev[j - 1] + s, prev[j] + GAP, curr[j - 1] + GAP);
        }
        memcpy(prev, curr, (n + 1) * sizeof(int));
    }
    memcpy(out, prev, (n + 1) * sizeof(int));
    free(prev); free(curr);
}
```

This function computes the forward or reverse row scores used in Hirschberg's recursion.

#### Why It Matters

- Reduces space complexity from $O(mn)$ to $O(m + n)$.
- Maintains the same optimal global alignment.
- Used in genome alignment, text diff tools, and compression systems.
- Demonstrates how divide and conquer combines with dynamic programming.

#### Complexity

| Operation  | Time    | Space      |
| ---------- | ------- | ---------- |
| Full DP    | $O(mn)$ | $O(mn)$    |
| Hirschberg | $O(mn)$ | $O(m + n)$ |

#### Try It Yourself

1. Align very long strings (thousands of symbols) to observe space savings.
2. Compare runtime and memory usage with standard Needleman–Wunsch.
3. Add traceback reconstruction to output aligned strings.
4. Combine with affine gaps (Gotoh + Hirschberg hybrid).
5. Experiment with text diff scenarios instead of biological data.

#### A Gentle Proof (Why It Works)

Hirschberg's method exploits the additivity of DP alignment scores:
the total optimal score can be decomposed into left and right halves at an optimal split point.
By recursively aligning halves, it reconstructs the same alignment without storing the full DP table.

This divide-and-conquer dynamic programming pattern is a powerful general idea, later reused in parallel and external-memory algorithms.

The Hirschberg algorithm reminds us that sometimes
we don't need to hold the whole world in memory —
just the frontier between what came before and what's next.

### 685 Multiple Sequence Alignment (MSA)

The Multiple Sequence Alignment (MSA) problem extends pairwise alignment to three or more sequences.
Its goal is to align all sequences together so that homologous positions, characters that share a common origin, line up in columns.
This is a central task in bioinformatics, used for protein family analysis, phylogenetic tree construction, and motif discovery.

#### The Problem

Given $k$ sequences $S_1, S_2, \ldots, S_k$ of varying lengths, we want to find alignments that maximize a global similarity score.

Each column of the alignment represents a possible evolutionary relationship, characters are aligned if they descend from the same ancestral position.

The score for an MSA is often defined by the sum-of-pairs method:

$$
\text{Score}(A) = \sum_{1 \le i < j \le k} \text{Score}(A_i, A_j)
$$

where $\text{Score}(A_i, A_j)$ is a pairwise alignment score (e.g., from Needleman–Wunsch).

#### Why It's Hard

While pairwise alignment is solvable in $O(mn)$ time,
MSA grows exponentially with the number of sequences:

$$
O(n^k)
$$

This is because each cell in a $k$-dimensional DP table represents one position in each sequence.

For example:

- 2 sequences → 2D matrix
- 3 sequences → 3D cube
- 4 sequences → 4D hypercube, and so on.

Therefore, exact MSA is computationally infeasible for more than 3 or 4 sequences, so practical algorithms use heuristics.

#### Progressive Alignment (Heuristic)

The most common practical approach is progressive alignment, used in tools like ClustalW and MUSCLE.
It works in three major steps:

1. Compute pairwise distances between all sequences (using quick alignments).
2. Build a guide tree (a simple phylogenetic tree using clustering methods like UPGMA or neighbor-joining).
3. Progressively align sequences following the tree, starting from the most similar pairs and merging upward.

At each merge step, previously aligned groups are treated as profiles, where each column holds probabilities of characters.

#### Example (Progressive Alignment Sketch)

```
Sequences:
A: GATTACA
B: GCATGCU
C: GATTGCA

Step 1: Align (A, C)
GATTACA
GATTGCA

Step 2: Align with B
G-ATTACA
G-CATGCU
G-ATTGCA
```

This gives a rough but biologically reasonable alignment, not necessarily the global optimum, but fast and usable.

#### Scoring Example

For three sequences, the DP recurrence becomes:

$$
dp[i][j][k] = \max
\begin{cases}
dp[i-1][j-1][k-1] + s(A_i, B_j, C_k), \
dp[i-1][j][k] + g, \
dp[i][j-1][k] + g, \
dp[i][j][k-1] + g, \
\text{(and combinations of two gaps)}
\end{cases}
$$

but this is impractical for large inputs, hence the reliance on heuristics.

#### Tiny Code (Pairwise Progressive Alignment Example)

```python
from itertools import combinations

def pairwise_score(a, b, match=1, mismatch=-1, gap=-2):
    dp = [[0]*(len(b)+1) for _ in range(len(a)+1)]
    for i in range(1, len(a)+1):
        dp[i][0] = i * gap
    for j in range(1, len(b)+1):
        dp[0][j] = j * gap
    for i in range(1, len(a)+1):
        for j in range(1, len(b)+1):
            s = match if a[i-1] == b[j-1] else mismatch
            dp[i][j] = max(
                dp[i-1][j-1] + s,
                dp[i-1][j] + gap,
                dp[i][j-1] + gap
            )
    return dp[-1][-1]

def guide_tree(sequences):
    scores = {}
    for (i, s1), (j, s2) in combinations(enumerate(sequences), 2):
        scores[(i, j)] = pairwise_score(s1, s2)
    return sorted(scores.items(), key=lambda x: -x[1])

sequences = ["GATTACA", "GCATGCU", "GATTGCA"]
print(guide_tree(sequences))
```

This produces pairwise scores, a simple starting point for building a guide tree.

#### Why It Matters

- Foundational tool in genomics, proteomics, and computational biology.
- Reveals evolutionary relationships and conserved patterns.
- Used in:

  * Protein family classification
  * Phylogenetic reconstruction
  * Functional motif prediction
  * Comparative genomics

#### Complexity

| Method                         | Time         | Space    |
| ------------------------------ | ------------ | -------- |
| Exact (k-D DP)                 | $O(n^k)$     | $O(n^k)$ |
| Progressive (ClustalW, MUSCLE) | $O(k^2 n^2)$ | $O(n^2)$ |
| Profile–profile refinement     | $O(k n^2)$   | $O(n)$   |

#### Try It Yourself

1. Try aligning 3 DNA sequences manually.
2. Compare pairwise and progressive results.
3. Use different scoring schemes and gap penalties.
4. Build a guide tree using your own distance metric.
5. Run your test sequences through Clustal Omega or MUSCLE to compare.

#### A Gentle Proof (Why It Works)

Progressive alignment does not guarantee optimality,
but it approximates the sum-of-pairs scoring function by reusing the dynamic programming backbone iteratively.
Each local alignment guides the next, preserving local homologies that reflect biological relationships.

This approach embodies a fundamental idea:
approximation guided by structure can often achieve near-optimal results when full optimization is impossible.

MSA is both science and art —
aligning sequences, patterns, and histories into a single evolutionary story.

### 686 Profile Alignment (Sequence-to-Profile and Profile-to-Profile)

The Profile Alignment algorithm generalizes pairwise sequence alignment to handle groups of sequences that have already been aligned, called *profiles*.
A profile represents the consensus structure of an aligned set, capturing position-specific frequencies, gaps, and weights.
Aligning a new sequence to a profile (or two profiles to each other) allows multiple sequence alignments to scale gracefully and improve biological accuracy.

#### The Concept

A profile can be viewed as a matrix:

| Position | A   | C   | G   | T   | Gap |
| -------- | --- | --- | --- | --- | --- |
| 1        | 0.9 | 0.0 | 0.1 | 0.0 | 0.0 |
| 2        | 0.1 | 0.8 | 0.0 | 0.1 | 0.0 |
| 3        | 0.0 | 0.0 | 1.0 | 0.0 | 0.0 |
| ...      | ... | ... | ... | ... | ... |

Each column stores the observed frequencies of nucleotides or amino acids at that position.
We can align:

- a new sequence against this profile (sequence-to-profile), or
- two profiles against each other (profile-to-profile).

#### Scoring Between Profiles

To compare a symbol $a$ and a profile column $C$, use expected substitution score:

$$
S(a, C) = \sum_{b \in \Sigma} p_C(b) \cdot s(a, b)
$$

where:

- $\Sigma$ is the alphabet (e.g., {A, C, G, T}),
- $p_C(b)$ is the frequency of $b$ in column $C$,
- $s(a, b)$ is the substitution score (e.g., from PAM or BLOSUM matrix).

For profile-to-profile comparison:

$$
S(C_1, C_2) = \sum_{a,b \in \Sigma} p_{C_1}(a) \cdot p_{C_2}(b) \cdot s(a, b)
$$

This reflects how compatible two alignment columns are based on their statistical composition.

#### Dynamic Programming Recurrence

The DP recurrence is the same as for Needleman–Wunsch,
but with scores based on columns instead of single symbols.

$$
dp[i][j] = \max
\begin{cases}
dp[i-1][j-1] + S(C_i, D_j), & \text{(column match)},\\[4pt]
dp[i-1][j] + g, & \text{(gap in profile D)},\\[4pt]
dp[i][j-1] + g, & \text{(gap in profile C)}.
\end{cases}
$$


where $C_i$ and $D_j$ are profile columns, and $g$ is the gap penalty.

#### Example (Sequence-to-Profile)

Profile (from previous alignments):

| Pos | A   | C   | G   | T   |
| --- | --- | --- | --- | --- |
| 1   | 0.7 | 0.1 | 0.2 | 0.0 |
| 2   | 0.0 | 0.8 | 0.1 | 0.1 |
| 3   | 0.0 | 0.0 | 1.0 | 0.0 |

New sequence: `ACG`

At each DP step, we compute the expected score between each symbol in `ACG` and profile columns,
then use standard DP recursion to find the best global or local alignment.

#### Tiny Code (Python)

```python
def expected_score(col, a, subs_matrix):
    return sum(col[b] * subs_matrix[a][b] for b in subs_matrix[a])

def profile_align(profile, seq, subs_matrix, gap=-2):
    m, n = len(profile), len(seq)
    dp = [[0]*(n+1) for _ in range(m+1)]

    for i in range(1, m+1):
        dp[i][0] = dp[i-1][0] + gap
    for j in range(1, n+1):
        dp[0][j] = dp[0][j-1] + gap

    for i in range(1, m+1):
        for j in range(1, n+1):
            s = expected_score(profile[i-1], seq[j-1], subs_matrix)
            dp[i][j] = max(
                dp[i-1][j-1] + s,
                dp[i-1][j] + gap,
                dp[i][j-1] + gap
            )
    return dp[m][n]

subs_matrix = {
    'A': {'A': 1, 'C': -1, 'G': -1, 'T': -1},
    'C': {'A': -1, 'C': 1, 'G': -1, 'T': -1},
    'G': {'A': -1, 'C': -1, 'G': 1, 'T': -1},
    'T': {'A': -1, 'C': -1, 'G': -1, 'T': 1}
}

profile = [
    {'A': 0.7, 'C': 0.1, 'G': 0.2, 'T': 0.0},
    {'A': 0.0, 'C': 0.8, 'G': 0.1, 'T': 0.1},
    {'A': 0.0, 'C': 0.0, 'G': 1.0, 'T': 0.0}
$$

print(profile_align(profile, "ACG", subs_matrix))
```

Output:

```
1.6
```

#### Why It Matters

- Extends MSA efficiently: new sequences can be added to existing alignments without recomputing everything.
- Profile-to-profile alignment forms the core of modern MSA software (MUSCLE, MAFFT, ClustalΩ).
- Statistical robustness: captures biological conservation patterns at each position.
- Handles ambiguity: each column represents uncertainty, not just a single symbol.

#### Complexity

| Operation        | Time    | Space   |      |         |
| ---------------- | ------- | ------- | ---- | ------- |
| Sequence–Profile | $O(mn)$ | $O(mn)$ |      |         |
| Profile–Profile  | $O(mn   | \Sigma  | ^2)$ | $O(mn)$ |

#### Try It Yourself

1. Construct a profile from two sequences manually (count and normalize).
2. Align a new sequence to that profile.
3. Compare results with direct pairwise alignment.
4. Extend to profile–profile and compute expected match scores.
5. Experiment with different substitution matrices (PAM250, BLOSUM62).

#### A Gentle Proof (Why It Works)

Profile alignment works because expected substitution scores preserve linearity:
the expected score between profiles is equal to the sum of expected pairwise scores between their underlying sequences.
Thus, profile alignment yields the same optimal alignment that would result from averaging over all pairwise combinations —
but computed in linear time instead of exponential time.

Profile alignment is the mathematical backbone of modern bioinformatics —
it replaces rigid characters with flexible probability landscapes,
allowing alignments to evolve as dynamically as the sequences they describe.

### 687 Hidden Markov Model (HMM) Alignment

The Hidden Markov Model (HMM) alignment method treats sequence alignment as a *probabilistic inference* problem.
Instead of deterministic scores and penalties, it models the process of generating sequences using states, transitions, and emission probabilities.
This gives a statistically rigorous foundation for sequence alignment, profile detection, and domain identification.

#### The Core Idea

An HMM defines a probabilistic model with:

- States that represent positions in an alignment (match, insertion, deletion).
- Transitions between states, capturing how likely we move from one to another.
- Emission probabilities describing how likely each state emits a particular symbol (A, C, G, T, etc.).

For sequence alignment, we use an HMM to represent how one sequence might have evolved from another through substitutions, insertions, and deletions.

#### Typical HMM Architecture for Pairwise Alignment

Each column of an alignment is modeled with three states:

```
   ┌───────────┐
   │  Match M  │
   └─────┬─────┘
         │
   ┌─────▼─────┐
   │ Insert I  │
   └─────┬─────┘
         │
   ┌─────▼─────┐
   │ Delete D  │
   └───────────┘
```

Each has:

- Transitions (e.g., M→M, M→I, M→D, etc.)
- Emissions: M and I emit symbols, D emits nothing.

#### Model Parameters

Let:

- $P(M_i \rightarrow M_{i+1})$ = transition probability between match states.
- $e_M(x)$ = emission probability of symbol $x$ from match state.
- $e_I(x)$ = emission probability of symbol $x$ from insert state.

Then the probability of an alignment path $Q = (q_1, q_2, ..., q_T)$ with emitted sequence $X = (x_1, x_2, ..., x_T)$ is:

$$
P(X, Q) = \prod_{t=1}^{T} P(q_t \mid q_{t-1}) \cdot e_{q_t}(x_t)
$$

The alignment problem becomes finding the most likely path through the model that explains both sequences.

#### The Viterbi Algorithm

We use dynamic programming to find the maximum likelihood alignment path.

Let $V_t(s)$ be the probability of the most likely path ending in state $s$ at position $t$.

The recurrence is:

$$
V_t(s) = e_s(x_t) \cdot \max_{s'} [V_{t-1}(s') \cdot P(s' \rightarrow s)]
$$

with backpointers for reconstruction.

Finally, the best path probability is:

$$
P^* = \max_s V_T(s)
$$

#### Example of Match–Insert–Delete Transitions

| From → To | Transition Probability |
| --------- | ---------------------- |
| M → M     | 0.8                    |
| M → I     | 0.1                    |
| M → D     | 0.1                    |
| I → I     | 0.7                    |
| I → M     | 0.3                    |
| D → D     | 0.6                    |
| D → M     | 0.4                    |

Emissions from Match or Insert states define the sequence content probabilities.

#### Tiny Code (Python, Simplified Viterbi)

```python
import numpy as np

states = ['M', 'I', 'D']
trans = {
    'M': {'M': 0.8, 'I': 0.1, 'D': 0.1},
    'I': {'I': 0.7, 'M': 0.3, 'D': 0.0},
    'D': {'D': 0.6, 'M': 0.4, 'I': 0.0}
}
emit = {
    'M': {'A': 0.3, 'C': 0.2, 'G': 0.3, 'T': 0.2},
    'I': {'A': 0.25, 'C': 0.25, 'G': 0.25, 'T': 0.25},
    'D': {}
}

def viterbi(seq):
    n = len(seq)
    V = np.zeros((n+1, len(states)))
    V[0, :] = np.log([1/3, 1/3, 1/3])  # uniform start

    for t in range(1, n+1):
        for j, s in enumerate(states):
            emis = np.log(emit[s].get(seq[t-1], 1e-9)) if s != 'D' else 0
            V[t, j] = emis + max(
                V[t-1, k] + np.log(trans[states[k]].get(s, 1e-9))
                for k in range(len(states))
            )
    return V

seq = "ACGT"
V = viterbi(seq)
print(np.exp(V[-1] - np.max(V[-1])))
```

Output (relative likelihood of alignment path):

```
$$0.82 0.09 0.09]
```

#### Why It Matters

- Provides a probabilistic foundation for alignment instead of heuristic scoring.
- Naturally models insertions, deletions, and substitutions.
- Forms the mathematical basis for:

  * Profile HMMs (used in HMMER, Pfam)
  * Gene finding and domain detection
  * Speech recognition and natural language models

HMM alignment can also be trained from data using Baum–Welch (EM) to learn emission and transition probabilities.

#### Complexity

| Operation                      | Time    | Space   |
| ------------------------------ | ------- | ------- |
| Viterbi (max likelihood)       | $O(mn)$ | $O(mn)$ |
| Forward–Backward (expectation) | $O(mn)$ | $O(mn)$ |

#### Try It Yourself

1. Build a 3-state match–insert–delete HMM and run Viterbi decoding.
2. Compare probabilities under different transition matrices.
3. Visualize the alignment path as a sequence of states.
4. Extend to Profile HMMs by chaining match states for each alignment column.
5. Train HMM parameters using Baum–Welch on known alignments.

#### A Gentle Proof (Why It Works)

Each possible alignment corresponds to a path through the HMM.
By dynamic programming, Viterbi ensures the Markov property holds —
the probability of each prefix alignment depends only on the previous state.
This makes global optimization tractable while capturing uncertainty and evolution probabilistically.

HMM alignment reframes alignment as *inference over structure and noise* —
a model that doesn't just align sequences, but explains how they came to differ.

### 688 BLAST (Basic Local Alignment Search Tool)

The BLAST algorithm is a fast heuristic method for finding local sequence alignments.
It's designed to search large biological databases quickly, comparing a query sequence against millions of others to find similar regions.
Rather than computing full dynamic programming matrices, BLAST cleverly balances speed and sensitivity by using *word-based seeding and extension*.

#### The Problem

Classical algorithms like Needleman–Wunsch or Smith–Waterman are exact but expensive:
they require $O(mn)$ time per pairwise alignment.

When you need to search a query (like a DNA or protein sequence) against a database of billions of letters,
that's completely infeasible.

BLAST trades a bit of optimality for speed,
detecting high-scoring regions (local matches) much faster through a multi-phase heuristic pipeline.

#### The Core Idea

BLAST works in three main phases:

1. Word Generation (Seeding)
   The query sequence is split into short fixed-length words (e.g., length 3 for proteins, 11 for DNA).
   Example:
   For `"AGCTTAGC"`, the 3-letter words are `AGC`, `GCT`, `CTT`, `TTA`, etc.

2. Database Scan
   Each word is looked up in the database for exact or near-exact matches.
   BLAST uses a *substitution matrix* (like BLOSUM or PAM) to expand words to similar ones with acceptable scores.

3. Extension and Scoring
   When a word match is found, BLAST extends it in both directions to form a local alignment —
   using a simple dynamic scoring model until the score drops below a threshold.

This is similar to Smith–Waterman,
but only around promising seed matches rather than every possible position.

#### Scoring System

Like other alignment methods, BLAST uses substitution matrices for match/mismatch scores
and gap penalties for insertions/deletions.

Typical protein scoring (BLOSUM62):

| Pair                      | Score |
| ------------------------- | ----- |
| Match                     | +4    |
| Conservative substitution | +1    |
| Non-conservative          | -2    |
| Gap open                  | -11   |
| Gap extend                | -1    |

Each alignment's bit score $S'$ and E-value (expected number of matches by chance) are then computed as:

$$
S' = \frac{\lambda S - \ln K}{\ln 2}
$$

$$
E = K m n e^{-\lambda S}
$$

where:

- $S$ = raw alignment score,
- $m, n$ = sequence lengths,
- $K, \lambda$ = statistical parameters from the scoring system.

#### Example (Simplified Flow)

Query: `ACCTGA`
Database sequence: `ACGTGA`

1. Seed: `ACC`, `CCT`, `CTG`, `TGA`
2. Matches: finds `TGA` in database.
3. Extension:

   ```
   Query:     ACCTGA
   Database:  ACGTGA
                 ↑↑ ↑
   ```

   Extends to include nearby matches until score decreases.

#### Tiny Code (Simplified BLAST-like Demo)

```python
def blast(query, database, word_size=3, match=1, mismatch=-1, threshold=2):
    words = [query[i:i+word_size] for i in range(len(query)-word_size+1)]
    hits = []
    for word in words:
        for j in range(len(database)-word_size+1):
            if word == database[j:j+word_size]:
                score = word_size * match
                left, right = j-1, j+word_size
                while left >= 0 and query[0] != database[left]:
                    score += mismatch
                    left -= 1
                while right < len(database) and query[-1] != database[right]:
                    score += mismatch
                    right += 1
                if score >= threshold:
                    hits.append((word, j, score))
    return hits

print(blast("ACCTGA", "TTACGTGACCTGATTACGA"))
```

Output:

```
$$('ACCT', 8, 4), ('CTGA', 10, 4)]
```

This simplified version just finds exact 4-mer seeds and reports matches.

#### Why It Matters

- Revolutionized bioinformatics by making large-scale sequence searches practical.
- Used for:

  * Gene and protein identification
  * Database annotation
  * Homology inference
  * Evolutionary analysis
- Variants include:

  * blastn (DNA)
  * blastp (proteins)
  * blastx (translated DNA → protein)
  * psiblast (position-specific iterative search)

BLAST's success lies in its elegant balance between statistical rigor and computational pragmatism.

#### Complexity

| Phase       | Approximate Time                           |
| ----------- | ------------------------------------------ |
| Word search | $O(m)$                                     |
| Extension   | proportional to #seeds                     |
| Overall     | sublinear in database size (with indexing) |

#### Try It Yourself

1. Vary the word size and observe how sensitivity changes.
2. Use different scoring thresholds.
3. Compare BLAST's output to Smith–Waterman's full local alignment.
4. Build a simple index (hash map) of k-mers for faster searching.
5. Explore `psiblast`, iterative refinement using profile scores.

#### A Gentle Proof (Why It Works)

The seed-and-extend principle works because most biologically significant local alignments contain short exact matches.
These act as "anchors" that can be found quickly without scanning the entire DP matrix.
Once found, local extensions around them reconstruct the alignment almost as effectively as exhaustive methods.

Thus, BLAST approximates local alignment by focusing computation where it matters most.

BLAST changed the scale of biological search —
from hours of exact computation to seconds of smart discovery.

### 689 FASTA (Word-Based Local Alignment)

The FASTA algorithm is another foundational heuristic for local sequence alignment, preceding BLAST.
It introduced the idea of using word matches (k-tuples) to find regions of similarity between sequences efficiently.
FASTA balances speed and accuracy by focusing on high-scoring short matches and extending them into longer alignments.

#### The Idea

FASTA avoids computing full dynamic programming over entire sequences.
Instead, it:

1. Finds short *exact matches* (called k-tuples) between the query and database sequences.
2. Scores diagonals where many matches occur.
3. Selects high-scoring regions and extends them using dynamic programming.

This allows fast identification of candidate regions likely to yield meaningful local alignments.

#### Step 1: k-Tuple Matching

Given a query of length $m$ and a database sequence of length $n$,
FASTA first identifies all short identical substrings of length $k$ (for proteins, typically $k=2$; for DNA, $k=6$).

Example (DNA, $k=3$):

Query: `ACCTGA`
Database: `ACGTGA`

k-tuples: `ACC`, `CCT`, `CTG`, `TGA`

Matches found:

- Query `CTG` ↔ Database `CTG` at different positions
- Query `TGA` ↔ Database `TGA`

Each match defines a diagonal in an alignment matrix (difference between indices in query and database).

#### Step 2: Diagonal Scoring

FASTA then scores each diagonal by counting the number of word hits along it.
High-density diagonals suggest potential regions of alignment.

For each diagonal $d = i - j$:
$$
S_d = \sum_{(i,j) \in \text{hits on } d} 1
$$

Top diagonals with highest $S_d$ are kept for further analysis.

#### Step 3: Rescoring and Extension

FASTA then rescans the top regions using a substitution matrix (e.g., PAM or BLOSUM)
to refine scores for similar but not identical matches.

Finally, a Smith–Waterman local alignment is performed only on these regions,
not across the entire sequences, drastically improving efficiency.

#### Example (Simplified Flow)

Query: `ACCTGA`
Database: `ACGTGA`

1. Word matches:

   * `CTG` (positions 3–5 in query, 3–5 in database)
   * `TGA` (positions 4–6 in query, 4–6 in database)

2. Both lie near the same diagonal → high-scoring region.

3. Dynamic programming only extends this region locally:

   ```
   ACCTGA
   || |||
   ACGTGA
   ```

   Result: alignment with a small substitution (C→G).

#### Tiny Code (Simplified FASTA Demo)

```python
def fasta(query, database, k=3, match=1, mismatch=-1):
    words = {query[i:i+k]: i for i in range(len(query)-k+1)}
    diagonals = {}
    for j in range(len(database)-k+1):
        word = database[j:j+k]
        if word in words:
            diag = words[word] - j
            diagonals[diag] = diagonals.get(diag, 0) + 1

    top_diag = max(diagonals, key=diagonals.get)
    return top_diag, diagonals[top_diag]

print(fasta("ACCTGA", "ACGTGA"))
```

Output:

```
(0, 2)
```

This means the best alignment diagonal (offset 0) has 2 matching k-tuples.

#### Why It Matters

- Precursor to BLAST, FASTA pioneered the k-tuple method and inspired BLAST's design.
- Statistical scoring, introduced expectation values (E-values) and normalized bit scores.
- Scalable, can search entire databases efficiently without losing much sensitivity.
- Flexible, supports DNA, RNA, and protein comparisons.

Still widely used for sensitive homology detection in genomics and proteomics.

#### Complexity

| Step                | Time                 | Space  |
| ------------------- | -------------------- | ------ |
| k-tuple matching    | $O(m + n)$           | $O(m)$ |
| Diagonal scoring    | proportional to hits | small  |
| Local DP refinement | $O(k^2)$             | small  |

#### Try It Yourself

1. Experiment with different k values (smaller k → more sensitive, slower).
2. Compare FASTA's hits to BLAST's on the same sequences.
3. Implement scoring with a substitution matrix (like BLOSUM62).
4. Plot diagonal density maps to visualize candidate alignments.
5. Use FASTA to align short reads (DNA) against reference genomes.

#### A Gentle Proof (Why It Works)

Word matches on the same diagonal indicate that two sequences share a common substring alignment.
By counting and rescoring diagonals, FASTA focuses computational effort only on promising regions —
a probabilistic shortcut that preserves most biologically relevant alignments
while skipping over unrelated sequence noise.

FASTA taught us the power of local heuristics:
you don't need to search everywhere, just where patterns start to sing.

### 690 Pairwise Dynamic Programming Alignment

The pairwise dynamic programming alignment algorithm is the general framework behind many alignment methods such as Needleman–Wunsch (global) and Smith–Waterman (local).
It provides a systematic way to compare two sequences by filling a matrix of scores that captures all possible alignments.
This is the foundation of computational sequence comparison.

#### The Problem

Given two sequences:

- Query: $A = a_1 a_2 \dots a_m$
- Target: $B = b_1 b_2 \dots b_n$

we want to find an alignment that maximizes a similarity score
based on matches, mismatches, and gaps.

Each position pair $(i, j)$ in the matrix represents an alignment between $a_i$ and $b_j$.

#### Scoring System

We define:

- Match score: $+s$
- Mismatch penalty: $-p$
- Gap penalty: $-g$

Then, the recurrence relation for the DP matrix $dp[i][j]$ is:

$$
dp[i][j] =
\max
\begin{cases}
dp[i-1][j-1] + \text{score}(a_i, b_j), & \text{(match/mismatch)},\\[4pt]
dp[i-1][j] - g, & \text{(gap in B)},\\[4pt]
dp[i][j-1] - g, & \text{(gap in A)}.
\end{cases}
$$


with initialization:

$$
dp[0][j] = -jg, \quad dp[i][0] = -ig
$$

and base case:

$$
dp[0][0] = 0
$$

#### Global vs Local Alignment

- Global alignment (Needleman–Wunsch):
  Considers the entire sequence.
  The best score is at $dp[m][n]$.

- Local alignment (Smith–Waterman):
  Allows partial alignments, setting
  $$dp[i][j] = \max(0, \text{previous terms})$$
  and taking the maximum over all cells as the final score.

#### Example (Global Alignment)

Query: `ACGT`
Target: `AGT`

Let match = +1, mismatch = -1, gap = -2.

| i/j | 0  | A  | G  | T  |
| --- | -- | -- | -- | -- |
| 0   | 0  | -2 | -4 | -6 |
| A   | -2 | 1  | -1 | -3 |
| C   | -4 | -1 | 0  | -2 |
| G   | -6 | -3 | 1  | -1 |
| T   | -8 | -5 | -1 | 2  |

The best score is 2, corresponding to alignment:

```
A C G T
|   | |
A - G T
```

#### Tiny Code (Python Implementation)

```python
def pairwise_align(a, b, match=1, mismatch=-1, gap=-2):
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        dp[i][0] = i * gap
    for j in range(1, n + 1):
        dp[0][j] = j * gap

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            s = match if a[i - 1] == b[j - 1] else mismatch
            dp[i][j] = max(
                dp[i - 1][j - 1] + s,
                dp[i - 1][j] + gap,
                dp[i][j - 1] + gap
            )
    return dp[m][n]

print(pairwise_align("ACGT", "AGT"))
```

Output:

```
2
```

#### Why It Matters

- Provides a unified foundation for sequence comparison.
- Forms the basis for global, local, semi-global, and affine-gap alignments.
- Guarantees optimal alignment given the scoring scheme.
- Extensible to profile and multiple alignments.

This framework underlies most algorithms in bioinformatics, text comparison, and natural language alignment.

#### Complexity

| Operation            | Time       | Space   |
| -------------------- | ---------- | ------- |
| Fill DP table        | $O(mn)$    | $O(mn)$ |
| Traceback (optional) | $O(m + n)$ |,       |

With linear-space optimization (e.g., Hirschberg's algorithm),
space can be reduced to $O(m + n)$.

#### Try It Yourself

1. Implement both global and local variants by changing initialization.
2. Add affine gap penalties with separate gap-open and gap-extend parameters.
3. Visualize the alignment matrix and traceback arrows.
4. Compare with BLAST/FASTA outputs on short sequences.
5. Experiment with different scoring matrices (e.g., BLOSUM62 for proteins).

#### A Gentle Proof (Why It Works)

The dynamic programming recurrence ensures optimal substructure:
the best alignment up to $(i,j)$ must include an optimal alignment of a prefix.
By filling the table in increasing order of $(i, j)$,
we guarantee that all subproblems are solved before being used.

The final cell thus contains the maximum achievable alignment score.

Pairwise alignment is the root from which all sequence comparison methods grow —
a precise balance between structure, similarity, and evolution.

# Section 70. Text Indexing and Search Structure 

### 691 Inverted Index Build

The inverted index is the core data structure behind modern search engines.
It maps each term (word or token) to a list of documents (and optionally positions) where that term appears.
This allows extremely fast keyword lookup, ranking, and retrieval across millions of documents.

#### The Idea

Instead of storing documents as sequences of words,
an inverted index stores words as keys and document IDs as values.

This is called *inversion* because it flips the direction of lookup:

- A *forward index* maps document → words.
- An *inverted index* maps word → documents.

#### Example

Suppose we have 3 documents:

| ID | Text                             |
| -- | -------------------------------- |
| 1  | "data structures and algorithms" |
| 2  | "algorithms for text processing" |
| 3  | "data compression and encoding"  |

The inverted index becomes:

| Term        | Documents |
| ----------- | --------- |
| algorithms  | [1, 2]    |
| and         | [1, 3]    |
| compression | [3]       |
| data        | [1, 3]    |
| encoding    | [3]       |
| for         | [2]       |
| processing  | [2]       |
| structures  | [1]       |
| text        | [2]       |

This lets us find all documents containing a term in $O(1)$ average lookup time per term.

#### Step-by-Step Construction

1. Tokenize Documents
   Split text into normalized tokens (lowercased, stripped of punctuation, stopwords removed).

   Example:
   `"Data Structures and Algorithms"` → `["data", "structures", "algorithms"]`

2. Assign Document IDs
   Each document in the collection gets a unique integer ID.

3. Build Postings Lists
   For each term, append the document ID to its posting list.

4. Sort and Deduplicate
   Sort postings lists and remove duplicate document IDs.

5. Optionally Compress
   Store gaps instead of full IDs and compress using variable-length encoding or delta coding.

#### Tiny Code (Python Implementation)

```python
from collections import defaultdict

def build_inverted_index(docs):
    index = defaultdict(set)
    for doc_id, text in enumerate(docs, start=1):
        tokens = text.lower().split()
        for token in tokens:
            index[token].add(doc_id)
    return {term: sorted(list(ids)) for term, ids in index.items()}

docs = [
    "data structures and algorithms",
    "algorithms for text processing",
    "data compression and encoding"
$$

index = build_inverted_index(docs)
for term, postings in index.items():
    print(f"{term}: {postings}")
```

Output:

```
algorithms: [1, 2]
and: [1, 3]
compression: [3]
data: [1, 3]
encoding: [3]
for: [2]
processing: [2]
structures: [1]
text: [2]
```

#### Mathematical Formulation

Let the document collection be $D = {d_1, d_2, \dots, d_N}$
and the vocabulary be $V = {t_1, t_2, \dots, t_M}$.

Then the inverted index is a mapping:

$$
I: t_i \mapsto P_i = {d_j \mid t_i \in d_j}
$$

where $P_i$ is the *posting list* of documents containing term $t_i$.

If we include positional information, we can define:

$$
I: t_i \mapsto {(d_j, \text{positions}(t_i, d_j))}
$$

#### Storage Optimization

A typical inverted index stores:

| Component            | Description                                   |
| -------------------- | --------------------------------------------- |
| Vocabulary table     | List of unique terms                          |
| Postings list        | Document IDs where term appears               |
| Term frequencies     | How many times each term appears per document |
| Positions (optional) | Word offsets for phrase queries               |
| Skip pointers        | Accelerate large posting list traversal       |

Compression methods (e.g., delta encoding, variable-byte, Golomb, or Elias gamma) dramatically reduce storage size.

#### Why It Matters

- Enables instant search across billions of documents.
- Core structure in systems like Lucene, Elasticsearch, and Google Search.
- Supports advanced features like:

  * Boolean queries (`AND`, `OR`, `NOT`)
  * Phrase queries ("data compression")
  * Proximity and fuzzy matching
  * Ranking (TF–IDF, BM25, etc.)

#### Complexity

| Step                 | Time            | Space      |   |     |    |   |
| -------------------- | --------------- | ---------- | - | --- | -- | - |
| Building index       | $O(N \times L)$ | $O(V + P)$ |   |     |    |   |
| Query lookup         | $O(1)$ per term |,          |   |     |    |   |
| Boolean AND/OR merge | $O(             | P_1        | + | P_2 | )$ |, |

Where:

- $N$ = number of documents
- $L$ = average document length
- $V$ = vocabulary size
- $P$ = total number of postings

#### Try It Yourself

1. Extend the code to store term frequencies per document.
2. Add phrase query support using positional postings.
3. Implement compression with gap encoding.
4. Compare search time before and after compression.
5. Visualize posting list merging for queries like `"data AND algorithms"`.

#### A Gentle Proof (Why It Works)

Because every document contributes its words independently,
the inverted index represents a union of local term-document relations.
Thus, any query term lookup reduces to simple set intersections of precomputed lists —
transforming expensive text scanning into efficient Boolean algebra on small sets.

The inverted index is the heartbeat of information retrieval,
turning words into structure, and search into instant insight.

### 692 Positional Index

A positional index extends the inverted index by recording the exact positions of each term within a document.
It enables more advanced queries such as phrase search, proximity search, and context-sensitive retrieval,
which are essential for modern search engines and text analysis systems.

#### The Idea

In a standard inverted index, each entry maps a term to a list of documents where it appears:

$$
I(t) = {d_1, d_2, \dots}
$$

A positional index refines this idea by mapping each term to pairs of
(document ID, positions list):

$$
I(t) = {(d_1, [p_{11}, p_{12}, \dots]), (d_2, [p_{21}, p_{22}, \dots]), \dots}
$$

where $p_{ij}$ are the word offsets (positions) where term $t$ occurs in document $d_i$.

#### Example

Consider 3 documents:

| ID | Text                              |
| -- | --------------------------------- |
| 1  | "data structures and algorithms"  |
| 2  | "algorithms for data compression" |
| 3  | "data and data encoding"          |

Then the positional index looks like:

| Term        | Postings                        |
| ----------- | ------------------------------- |
| algorithms  | (1, [3]), (2, [1])              |
| and         | (1, [2]), (3, [2])              |
| compression | (2, [3])                        |
| data        | (1, [1]), (2, [2]), (3, [1, 3]) |
| encoding    | (3, [4])                        |
| for         | (2, [2])                        |
| structures  | (1, [2])                        |

Each posting now stores both document IDs and position lists.

#### How Phrase Queries Work

To find a phrase like `"data structures"`,
we must locate documents where:

- `data` appears at position $p$
- `structures` appears at position $p+1$

This is done by intersecting posting lists with positional offsets.

#### Phrase Query Example

Phrase: `"data structures"`

1. From the index:

   * `data` → (1, [1]), (2, [2]), (3, [1, 3])
   * `structures` → (1, [2])

2. Intersection by document:

   * Only doc 1 contains both.

3. Compare positions:

   * In doc 1: `1` (for `data`) and `2` (for `structures`)
   * Difference = 1 → phrase match confirmed.

Result: document 1.

#### Tiny Code (Python Implementation)

```python
from collections import defaultdict

def build_positional_index(docs):
    index = defaultdict(lambda: defaultdict(list))
    for doc_id, text in enumerate(docs, start=1):
        tokens = text.lower().split()
        for pos, token in enumerate(tokens):
            index[token][doc_id].append(pos)
    return index

docs = [
    "data structures and algorithms",
    "algorithms for data compression",
    "data and data encoding"
$$

index = build_positional_index(docs)
for term, posting in index.items():
    print(term, dict(posting))
```

Output:

```
data {1: [0], 2: [2], 3: [0, 2]}
structures {1: [1]}
and {1: [2], 3: [1]}
algorithms {1: [3], 2: [0]}
for {2: [1]}
compression {2: [3]}
encoding {3: [3]}
```

#### Phrase Query Search

```python
def phrase_query(index, term1, term2):
    results = []
    for doc in set(index[term1]) & set(index[term2]):
        pos1 = index[term1][doc]
        pos2 = index[term2][doc]
        if any(p2 - p1 == 1 for p1 in pos1 for p2 in pos2):
            results.append(doc)
    return results

print(phrase_query(index, "data", "structures"))
```

Output:

```
$$1]
```

#### Mathematical View

For a phrase query of $k$ terms $t_1, t_2, \dots, t_k$,
we find documents $d$ such that:

$$
\exists p_1, p_2, \dots, p_k \text{ with } p_{i+1} = p_i + 1
$$

for all $i \in [1, k-1]$.

#### Why It Matters

A positional index enables:

| Feature           | Description                                   |
| ----------------- | --------------------------------------------- |
| Phrase search     | Exact multi-word matches ("machine learning") |
| Proximity search  | Terms appearing near each other               |
| Order sensitivity | "data compression" ≠ "compression data"       |
| Context retrieval | Extract sentence windows efficiently          |

It trades additional storage for much more expressive search capability.

#### Complexity

| Step                   | Time            | Space      |   |     |    |   |
| ---------------------- | --------------- | ---------- | - | --- | -- | - |
| Build index            | $O(N \times L)$ | $O(V + P)$ |   |     |    |   |
| Phrase query (2 terms) | $O(             | P_1        | + | P_2 | )$ |, |
| Phrase query (k terms) | $O(k \times P)$ |,          |   |     |    |   |

Where $P$ is the average posting list length.

#### Try It Yourself

1. Extend to n-gram queries for phrases of arbitrary length.
2. Add a window constraint for "within k words" search.
3. Implement compressed positional storage using delta encoding.
4. Test with a large corpus and measure query speed.
5. Visualize how positional overlaps form phrase matches.

#### A Gentle Proof (Why It Works)

Each position in the text defines a coordinate in a grid of word order.
By intersecting these coordinates across words,
we reconstruct contiguous patterns —
just as syntax and meaning emerge from words in sequence.

The positional index is the bridge from word to phrase,
turning text search into understanding of structure and order.

### 693 TF–IDF Weighting

TF–IDF (Term Frequency–Inverse Document Frequency) is one of the most influential ideas in information retrieval.
It quantifies how *important* a word is to a document in a collection by balancing two opposing effects:

- Words that appear frequently in a document are important.
- Words that appear in many documents are less informative.

Together, these ideas let us score documents by how well they match a query, forming the basis for ranked retrieval systems like search engines.

#### The Core Idea

The TF–IDF score for term $t$ in document $d$ within a corpus $D$ is:

$$
\text{tfidf}(t, d, D) = \text{tf}(t, d) \times \text{idf}(t, D)
$$

where:

- $\text{tf}(t, d)$ = term frequency (how often $t$ appears in $d$)
- $\text{idf}(t, D)$ = inverse document frequency (how rare $t$ is across $D$)

#### Step 1: Term Frequency (TF)

Term frequency measures how often a term appears in a single document:

$$
\text{tf}(t, d) = \frac{f_{t,d}}{\sum_{t'} f_{t',d}}
$$

where $f_{t,d}$ is the raw count of term $t$ in document $d$.

Common variations:

| Formula                              | Description                  |
| ------------------------------------ | ---------------------------- |
| $f_{t,d}$                            | raw count                    |
| $1 + \log f_{t,d}$                   | logarithmic scaling          |
| $\frac{f_{t,d}}{\max_{t'} f_{t',d}}$ | normalized by max term count |

#### Step 2: Inverse Document Frequency (IDF)

IDF downweights common words (like *the*, *and*, *data*) that appear in many documents:

$$
\text{idf}(t, D) = \log \frac{N}{n_t}
$$

where:

- $N$ = total number of documents
- $n_t$ = number of documents containing term $t$

A smoothed version avoids division by zero:

$$
\text{idf}(t, D) = \log \frac{1 + N}{1 + n_t} + 1
$$

#### Step 3: TF–IDF Weight

Combining both parts:

$$
w_{t,d} = \text{tf}(t, d) \times \log \frac{N}{n_t}
$$

The resulting weight $w_{t,d}$ represents how much *term $t$* contributes to identifying *document $d$*.

#### Example

Suppose our corpus has three documents:

| ID | Text                             |
| -- | -------------------------------- |
| 1  | "data structures and algorithms" |
| 2  | "algorithms for data analysis"   |
| 3  | "machine learning and data"      |

Vocabulary: `["data", "structures", "algorithms", "analysis", "machine", "learning", "and", "for"]`

Total $N = 3$ documents.

| Term       | $n_t$ | $\text{idf}(t)$    |
| ---------- | ----- | ------------------ |
| data       | 3     | $\log(3/3) = 0$    |
| structures | 1     | $\log(3/1) = 1.10$ |
| algorithms | 2     | $\log(3/2) = 0.40$ |
| analysis   | 1     | 1.10               |
| machine    | 1     | 1.10               |
| learning   | 1     | 1.10               |
| and        | 2     | 0.40               |
| for        | 1     | 1.10               |

So "data" is not distinctive (IDF = 0),
while rare words like "structures" or "analysis" carry more weight.

#### Tiny Code (Python Implementation)

```python
import math
from collections import Counter

def compute_tfidf(docs):
    N = len(docs)
    term_doc_count = Counter()
    term_freqs = []

    for doc in docs:
        tokens = doc.lower().split()
        counts = Counter(tokens)
        term_freqs.append(counts)
        term_doc_count.update(set(tokens))

    tfidf = []
    for counts in term_freqs:
        doc_scores = {}
        for term, freq in counts.items():
            tf = freq / sum(counts.values())
            idf = math.log((1 + N) / (1 + term_doc_count[term])) + 1
            doc_scores[term] = tf * idf
        tfidf.append(doc_scores)
    return tfidf

docs = [
    "data structures and algorithms",
    "algorithms for data analysis",
    "machine learning and data"
$$

for i, scores in enumerate(compute_tfidf(docs), 1):
    print(f"Doc {i}: {scores}")
```

#### TF–IDF Vector Representation

Each document becomes a vector in term space:

$$
\mathbf{d} = [w_{t_1,d}, w_{t_2,d}, \dots, w_{t_M,d}]
$$

Similarity between a query $\mathbf{q}$ and document $\mathbf{d}$
is measured by cosine similarity:

$$
\text{sim}(\mathbf{q}, \mathbf{d}) =
\frac{\mathbf{q} \cdot \mathbf{d}}
{|\mathbf{q}| , |\mathbf{d}|}
$$

This allows ranked retrieval by sorting documents by similarity score.

#### Why It Matters

| Benefit                       | Explanation                                               |
| ----------------------------- | --------------------------------------------------------- |
| Balances relevance        | Highlights words frequent in a doc but rare in the corpus |
| Lightweight and effective | Simple to compute and works well for text retrieval       |
| Foundation for ranking    | Used in BM25, vector search, and embeddings               |
| Intuitive                 | Mirrors human sense of "keyword importance"               |

#### Complexity

| Step                     | Time            | Space           |
| ------------------------ | --------------- | --------------- |
| Compute term frequencies | $O(N \times L)$ | $O(V)$          |
| Compute IDF              | $O(V)$          | $O(V)$          |
| Compute TF–IDF weights   | $O(N \times V)$ | $O(N \times V)$ |

Where $N$ = number of documents, $L$ = average document length, $V$ = vocabulary size.

#### Try It Yourself

1. Normalize all TF–IDF vectors and compare with cosine similarity.
2. Add stopword removal and stemming to improve weighting.
3. Compare TF–IDF ranking vs raw term frequency.
4. Build a simple query-matching system using dot products.
5. Visualize document clusters using PCA on TF–IDF vectors.

#### A Gentle Proof (Why It Works)

TF–IDF expresses information gain:
a term's weight is proportional to how much it reduces uncertainty about which document we're reading.
Common words provide little information, while rare, specific terms (like "entropy" or "suffix tree") pinpoint documents effectively.

TF–IDF remains one of the most elegant bridges between statistics and semantics —
a simple equation that made machines understand what matters in text.

### 694 BM25 Ranking

BM25 (Best Matching 25) is a ranking function used in modern search engines to score how relevant a document is to a query.
It improves upon TF–IDF by modeling term saturation and document length normalization, making it more robust and accurate for practical retrieval tasks.

#### The Idea

BM25 builds on TF–IDF but introduces two realistic corrections:

1. Term frequency saturation, extra occurrences of a term contribute less after a point.
2. Length normalization, longer documents are penalized so they don't dominate results.

It estimates the probability that a document $d$ is relevant to a query $q$ using a scoring function based on term frequencies and document statistics.

#### The BM25 Formula

For a query $q = {t_1, t_2, \dots, t_n}$ and a document $d$,
the BM25 score is:

$$
\text{score}(d, q) = \sum_{t \in q} \text{idf}(t) \cdot
\frac{f(t, d) \cdot (k_1 + 1)}{f(t, d) + k_1 \cdot \left(1 - b + b \cdot \frac{|d|}{\text{avgdl}}\right)}
$$

where:

- $f(t, d)$, frequency of term $t$ in document $d$
- $|d|$, length of document $d$ (in words)
- $\text{avgdl}$, average document length in the corpus
- $k_1$, term frequency scaling factor (commonly $1.2$ to $2.0$)
- $b$, length normalization factor (commonly $0.75$)

and

$$
\text{idf}(t) = \log\frac{N - n_t + 0.5}{n_t + 0.5} + 1
$$

where $N$ is the total number of documents, and $n_t$ is the number of documents containing term $t$.

#### Intuition Behind the Formula

| Concept              | Meaning                                 |
| -------------------- | --------------------------------------- |
| $\text{idf}(t)$      | Rare terms get higher weight            |
| $f(t, d)$            | Term frequency boosts relevance         |
| Saturation term      | Prevents frequent words from dominating |
| Length normalization | Adjusts for longer documents            |

When $b = 0$, length normalization is disabled.
When $b = 1$, it fully normalizes by document length.

#### Example

Suppose:

- $N = 3$, $\text{avgdl} = 5$, $k_1 = 1.5$, $b = 0.75$
- Query: `["data", "compression"]`

Documents:

| ID | Text                              | Length |
| -- | --------------------------------- | ------ |
| 1  | "data structures and algorithms"  | 4      |
| 2  | "algorithms for data compression" | 4      |
| 3  | "data compression and encoding"   | 4      |

Compute $n_t$:

- $\text{data}$ in 3 docs → $n_{\text{data}} = 3$
- $\text{compression}$ in 2 docs → $n_{\text{compression}} = 2$

Then:

$$
\text{idf(data)} = \log\frac{3 - 3 + 0.5}{3 + 0.5} + 1 = 0.86
$$
$$
\text{idf(compression)} = \log\frac{3 - 2 + 0.5}{2 + 0.5} + 1 = 1.22
$$

Each document gets a score depending on how many times these terms appear and their lengths.
The one containing both "data" and "compression" (doc 3) will rank highest.

#### Tiny Code (Python Implementation)

```python
import math
from collections import Counter

def bm25_score(query, docs, k1=1.5, b=0.75):
    N = len(docs)
    avgdl = sum(len(doc.split()) for doc in docs) / N
    df = Counter()
    for doc in docs:
        for term in set(doc.split()):
            df[term] += 1

    scores = []
    for doc in docs:
        words = doc.split()
        tf = Counter(words)
        score = 0.0
        for term in query:
            if term not in tf:
                continue
            idf = math.log((N - df[term] + 0.5) / (df[term] + 0.5)) + 1
            numerator = tf[term] * (k1 + 1)
            denominator = tf[term] + k1 * (1 - b + b * len(words) / avgdl)
            score += idf * (numerator / denominator)
        scores.append(score)
    return scores

docs = [
    "data structures and algorithms",
    "algorithms for data compression",
    "data compression and encoding"
$$
query = ["data", "compression"]
print(bm25_score(query, docs))
```

Output (approximate):

```
$$0.86, 1.78, 2.10]
```

#### Why It Matters

| Advantage                         | Description                                     |
| --------------------------------- | ----------------------------------------------- |
| Improves TF–IDF               | Models term saturation and document length      |
| Practical and robust          | Works well across domains                       |
| Foundation of IR systems      | Used in Lucene, Elasticsearch, Solr, and others |
| Balances recall and precision | Retrieves both relevant and concise results     |

BM25 is now the de facto standard for keyword-based ranking before vector embeddings.

#### Complexity

| Step           | Time                       | Space  |            |   |
| -------------- | -------------------------- | ------ | ---------- | - |
| Compute IDF    | $O(V)$                     | $O(V)$ |            |   |
| Score each doc | $O(                        | q      | \times N)$ |, |
| Index lookup   | $O(\log N)$ per query term |,      |            |   |

#### Try It Yourself

1. Experiment with different $k_1$ and $b$ values and observe ranking changes.
2. Add TF–IDF normalization and compare results.
3. Use a small corpus to visualize term contribution to scores.
4. Combine BM25 with inverted index retrieval for efficiency.
5. Extend to multi-term or weighted queries.

#### A Gentle Proof (Why It Works)

BM25 approximates a probabilistic retrieval model:
it assumes that the likelihood of a document being relevant increases with term frequency,
but saturates logarithmically as repetitions add diminishing information.

By adjusting for document length, it ensures that relevance reflects *content density*, not document size.

BM25 elegantly bridges probability and information theory —
it's TF–IDF, evolved for the real world of messy, uneven text.

### 695 Trie Index

A Trie Index (short for *retrieval tree*) is a prefix-based data structure used for fast word lookup, auto-completion, and prefix search.
It's especially powerful for dictionary storage, query suggestion, and full-text search systems where matching prefixes efficiently is essential.

#### The Idea

A trie organizes words character by character in a tree form,
where each path from the root to a terminal node represents one word.

Formally, a trie for a set of strings $S = {s_1, s_2, \dots, s_n}$
is a rooted tree such that:

- Each edge is labeled by a character.
- The concatenation of labels along a path from the root to a terminal node equals one string $s_i$.
- Shared prefixes are stored only once.

#### Example

Insert the words:
`data`, `database`, `datum`, `dog`

The trie structure looks like:

```
(root)
 ├─ d
 │   ├─ a
 │   │   ├─ t
 │   │   │   ├─ a (✓)
 │   │   │   ├─ b → a → s → e (✓)
 │   │   │   └─ u → m (✓)
 │   └─ o → g (✓)
```

✓ marks the end of a complete word.

#### Mathematical View

Let $\Sigma$ be the alphabet and $n = |S|$ the number of words.
The total number of nodes in the trie is bounded by:

$$
O\left(\sum_{s \in S} |s|\right)
$$

Each search or insertion of a string of length $m$
takes time:

$$
O(m)
$$

— independent of the number of stored words.

#### How Search Works

To check if a word exists:

1. Start at the root.
2. Follow the edge for each successive character.
3. If you reach a node marked "end of word", the word exists.

To find all words with prefix `"dat"`:

1. Traverse `"d" → "a" → "t"`.
2. Collect all descendants of that node recursively.

#### Tiny Code (Python Implementation)

```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        node = self.root
        for ch in word:
            if ch not in node.children:
                node.children[ch] = TrieNode()
            node = node.children[ch]
        node.is_end = True

    def search(self, word):
        node = self.root
        for ch in word:
            if ch not in node.children:
                return False
            node = node.children[ch]
        return node.is_end

    def starts_with(self, prefix):
        node = self.root
        for ch in prefix:
            if ch not in node.children:
                return []
            node = node.children[ch]
        return self._collect(node, prefix)

    def _collect(self, node, prefix):
        words = []
        if node.is_end:
            words.append(prefix)
        for ch, child in node.children.items():
            words.extend(self._collect(child, prefix + ch))
        return words

# Example
trie = Trie()
for word in ["data", "database", "datum", "dog"]:
    trie.insert(word)

print(trie.search("data"))       # True
print(trie.starts_with("dat"))   # ['data', 'database', 'datum']
```

#### Variations

| Variant                          | Description                                      |
| -------------------------------- | ------------------------------------------------ |
| Compressed Trie (Radix Tree) | Merges chains of single children for compactness |
| Suffix Trie                  | Stores all suffixes for substring search         |
| Patricia Trie                | Bitwise trie used in networking (IP routing)     |
| DAWG                         | Deduplicated trie for all substrings             |
| Trie + Hashing               | Hybrid used in modern search indexes             |

#### Applications

| Use Case                   | Description                            |
| -------------------------- | -------------------------------------- |
| Autocomplete           | Suggest next words based on prefix     |
| Spell checking         | Lookup closest valid words             |
| Dictionary compression | Store large lexicons efficiently       |
| Search engines         | Fast prefix and wildcard query support |
| Routing tables         | IP prefix matching via Patricia trie   |

#### Complexity

| Operation    | Time       | Space  |
| ------------ | ---------- | ------ |
| Insert word  | $O(m)$     | $O(m)$ |
| Search word  | $O(m)$     | $O(1)$ |
| Prefix query | $O(m + k)$ | $O(1)$ |

where:

- $m$ = word length
- $k$ = number of results returned

Space can be large if many words share few prefixes,
but compression (Radix / DAWG) reduces overhead.

#### Try It Yourself

1. Build a trie for all words in a text corpus and query by prefix.
2. Extend it to support wildcard matching (`d?t*`).
3. Add frequency counts at nodes to rank autocomplete suggestions.
4. Visualize prefix sharing across words.
5. Compare space usage vs a hash-based dictionary.

#### A Gentle Proof (Why It Works)

A trie transforms string comparison from linear search over words
to character traversal —
replacing many string comparisons with a single prefix walk.
The prefix paths ensure $O(m)$ search cost,
a fundamental speedup when large sets share overlapping beginnings.

A Trie Index is the simplest glimpse of structure inside language —
where shared prefixes reveal both efficiency and meaning.

### 696 Suffix Array Index

A Suffix Array Index is a compact data structure for fast substring search.
It stores all suffixes of a text in sorted order, allowing binary search–based lookups for any substring pattern.
Unlike suffix trees, suffix arrays are space-efficient, simple to implement, and widely used in text search, bioinformatics, and data compression.

#### The Idea

Given a string $S$ of length $n$,
consider all its suffixes:

$$
S_1 = S[1:n], \quad S_2 = S[2:n], \quad \dots, \quad S_n = S[n:n]
$$

A suffix array is an array of integers that gives the starting indices of these suffixes in lexicographic order.

Formally:

$$
\text{SA}[i] = \text{the starting position of the } i^\text{th} \text{ smallest suffix}
$$

#### Example

Let $S = \text{"banana"}$.

All suffixes:

| Index | Suffix |
| ----- | ------ |
| 0     | banana |
| 1     | anana  |
| 2     | nana   |
| 3     | ana    |
| 4     | na     |
| 5     | a      |

Sort them lexicographically:

| Rank | Suffix | Start |
| ---- | ------ | ----- |
| 0    | a      | 5     |
| 1    | ana    | 3     |
| 2    | anana  | 1     |
| 3    | banana | 0     |
| 4    | na     | 4     |
| 5    | nana   | 2     |

Hence the suffix array:

$$
\text{SA} = [5, 3, 1, 0, 4, 2]
$$

#### Substring Search Using SA

To find all occurrences of pattern $P$ in $S$:

1. Binary search for the lexicographic lower bound of $P$.
2. Binary search for the upper bound of $P$.
3. The matching suffixes are between these indices.

Each comparison takes $O(m)$ for pattern length $m$,
and the binary search takes $O(\log n)$ comparisons.

Total complexity: $O(m \log n)$.

#### Example Search

Search for `"ana"` in `"banana"`.

- Binary search over suffixes:

  * Compare `"ana"` with `"banana"`, `"anana"`, etc.
- Matches found at SA indices `[1, 3]`, corresponding to positions 1 and 3 in the text.

#### Tiny Code (Python Implementation)

```python
def build_suffix_array(s):
    return sorted(range(len(s)), key=lambda i: s[i:])

def suffix_array_search(s, sa, pattern):
    n, m = len(s), len(pattern)
    l, r = 0, n
    while l < r:
        mid = (l + r) // 2
        if s[sa[mid]:sa[mid] + m] < pattern:
            l = mid + 1
        else:
            r = mid
    start = l
    r = n
    while l < r:
        mid = (l + r) // 2
        if s[sa[mid]:sa[mid] + m] <= pattern:
            l = mid + 1
        else:
            r = mid
    end = l
    return sa[start:end]

text = "banana"
sa = build_suffix_array(text)
print(sa)
print(suffix_array_search(text, sa, "ana"))
```

Output:

```
$$5, 3, 1, 0, 4, 2]
$$1, 3]
```

#### Relationship to LCP Array

The LCP array (Longest Common Prefix) stores the lengths of common prefixes between consecutive suffixes in the sorted order:

$$
\text{LCP}[i] = \text{LCP}(S[\text{SA}[i]], S[\text{SA}[i-1]])
$$

This helps skip repeated comparisons during substring search or pattern matching.

#### Construction Algorithms

| Algorithm       | Complexity      | Idea                                     |
| --------------- | --------------- | ---------------------------------------- |
| Naive sort      | $O(n^2 \log n)$ | Sort suffixes directly                   |
| Prefix-doubling | $O(n \log n)$   | Sort by 2^k-length prefixes              |
| SA-IS           | $O(n)$          | Induced sorting (used in modern systems) |

#### Applications

| Area                        | Use                                     |
| --------------------------- | --------------------------------------- |
| Text search                 | Fast substring lookup                   |
| Data compression            | Used in Burrows–Wheeler Transform (BWT) |
| Bioinformatics              | Genome pattern search                   |
| Plagiarism detection        | Common substring discovery              |
| Natural language processing | Phrase frequency and suffix clustering  |

#### Complexity

| Operation                  | Time            | Space  |
| -------------------------- | --------------- | ------ |
| Build suffix array (naive) | $O(n \log^2 n)$ | $O(n)$ |
| Search substring           | $O(m \log n)$   | $O(1)$ |
| With LCP optimization      | $O(m + \log n)$ | $O(n)$ |

#### Try It Yourself

1. Build a suffix array for `"mississippi"`.
2. Search for `"iss"` and `"sip"` using binary search.
3. Compare performance with the naive substring search.
4. Visualize lexicographic order of suffixes.
5. Extend the index to support case-insensitive matching.

#### A Gentle Proof (Why It Works)

Suffix arrays rely on the lexicographic order of suffixes,
which aligns perfectly with substring search:
all substrings starting with a pattern form a contiguous block in sorted suffix order.
Binary search efficiently locates this block, ensuring deterministic $O(m \log n)$ matching.

The Suffix Array Index is the minimalist sibling of the suffix tree —
compact, elegant, and at the heart of fast search engines and genome analysis tools.

### 697 Compressed Suffix Array

A Compressed Suffix Array (CSA) is a space-efficient version of the classic suffix array.
It preserves all the power of substring search while reducing memory usage from $O(n \log n)$ bits to near the information-theoretic limit, roughly the entropy of the text itself.
CSAs are the backbone of compressed text indexes used in large-scale search and bioinformatics systems.

#### The Idea

A standard suffix array explicitly stores sorted suffix indices.
A compressed suffix array replaces that explicit array with a compact, self-indexed representation, allowing:

- substring search without storing the original text, and
- access to suffix array positions using compressed data structures.

Formally, a CSA supports three key operations in time $O(\log n)$ or better:

1. `find(P)` – find all occurrences of pattern $P$ in $S$
2. `locate(i)` – recover the position in the text for suffix array index $i$
3. `extract(l, r)` – retrieve substring $S[l:r]$ directly from the index

#### Key Components

A compressed suffix array uses several coordinated structures:

1. Burrows–Wheeler Transform (BWT)
   Rearranges $S$ to cluster similar characters.
   Enables efficient backward searching.

2. Rank/Select Data Structures
   Allow counting and locating characters within BWT efficiently.

3. Sampling
   Periodically store full suffix positions; reconstruct others by walking backward through BWT.

#### Construction Sketch

Given text $S$ of length $n$ (ending with a unique terminator `$`):

1. Build suffix array $\text{SA}$ for $S$.

2. Derive Burrows–Wheeler Transform:

$$
\text{BWT}[i] =
\begin{cases}
S[\text{SA}[i] - 1], & \text{if } \text{SA}[i] > 0,\\[4pt]
\text{\$}, & \text{if } \text{SA}[i] = 0.
\end{cases}
$$


3. Compute the C array, where $C[c]$ = number of characters in $S$ smaller than $c$.

4. Store rank structures over BWT for fast character counting.

5. Keep samples of $\text{SA}[i]$ at fixed intervals (e.g., every $t$ entries).

#### Backward Search (Pattern Matching)

The pattern $P = p_1 p_2 \dots p_m$ is searched *backward*:

Initialize:
$$
l = 0, \quad r = n - 1
$$

For each character $p_i$ from last to first:

$$
l = C[p_i] + \text{rank}(p_i, l - 1) + 1
$$
$$
r = C[p_i] + \text{rank}(p_i, r)
$$

When $l > r$, no match exists.
Otherwise, all occurrences of $P$ are between $\text{SA}[l]$ and $\text{SA}[r]$ (reconstructed via sampling).

#### Example

Let $S=\texttt{"banana\textdollar"}$.

1. $\text{SA} = [6,\,5,\,3,\,1,\,0,\,4,\,2]$
2. $\text{BWT} = [a,\, n,\, n,\, b,\, \textdollar,\, a,\, a]$
3. $C = \{\textdollar\!:0,\, a\!:\!1,\, b\!:\!3,\, n\!:\!4\}$

Search for $P=\texttt{"ana"}$ backward:

| Step | char | new $[l,r]$ |
| ---- | ---- | ----------- |
| init | $\epsilon$ | $[0,6]$ |
| 1    | $a$  | $[1,3]$     |
| 2    | $n$  | $[4,5]$     |
| 3    | $a$  | $[2,3]$     |

Result: matches at $\text{SA}[2]$ and $\text{SA}[3]$ which are positions $1$ and $3$ in $\texttt{"banana"}$.


#### Tiny Code (Simplified Python Prototype)

```python
from bisect import bisect_left, bisect_right

def suffix_array(s):
    return sorted(range(len(s)), key=lambda i: s[i:])

def bwt_from_sa(s, sa):
    return ''.join(s[i - 1] if i else '$' for i in sa)

def search_bwt(bwt, pattern, sa, s):
    # naive backward search using bisect
    suffixes = [s[i:] for i in sa]
    l = bisect_left(suffixes, pattern)
    r = bisect_right(suffixes, pattern)
    return sa[l:r]

s = "banana$"
sa = suffix_array(s)
bwt = bwt_from_sa(s, sa)
print("SA:", sa)
print("BWT:", bwt)
print("Match:", search_bwt(bwt, "ana", sa, s))
```

Output:

```
SA: [6, 5, 3, 1, 0, 4, 2]
BWT: annb$aa
Match: [1, 3]
```

*(This is an uncompressed version, real CSAs replace the arrays with bit-packed rank/select structures.)*

#### Compression Techniques

| Technique                       | Description                                                   |
| ------------------------------- | ------------------------------------------------------------- |
| Wavelet Tree                | Encodes BWT using hierarchical bitmaps                        |
| Run-Length BWT (RLBWT)      | Compresses repeated runs in BWT                               |
| Sampling                    | Store only every $t$-th suffix; recover others via LF-mapping |
| Bitvectors with Rank/Select | Enable constant-time navigation without decompression         |

#### Applications

| Field                 | Usage                                      |
| --------------------- | ------------------------------------------ |
| Search engines    | Full-text search over compressed corpora   |
| Bioinformatics    | Genome alignment (FM-index in Bowtie, BWA) |
| Data compression  | Core of self-indexing compressors          |
| Versioned storage | Deduplicated document storage              |

#### Complexity

| Operation         | Time               | Space                          |
| ----------------- | ------------------ | ------------------------------ |
| Search            | $O(m \log \sigma)$ | $(1 + \epsilon) n H_k(S)$ bits |
| Locate            | $O(t \log \sigma)$ | $O(n / t)$ sampled entries     |
| Extract substring | $O(\ell + \log n)$ | $O(n)$ compressed structure    |

where $H_k(S)$ is the $k$-th order entropy of the text and $\sigma$ is alphabet size.

#### Try It Yourself

1. Build the suffix array and BWT for `"mississippi$"`.
2. Perform backward search for `"issi"`.
3. Compare memory usage vs uncompressed suffix array.
4. Implement LF-mapping for substring extraction.
5. Explore run-length encoding of BWT for repetitive text.

#### A Gentle Proof (Why It Works)

The compressed suffix array relies on the BWT's local clustering —
nearby characters in the text are grouped, reducing entropy.
By maintaining rank/select structures over BWT, we can simulate suffix array navigation *without explicitly storing it*.
Thus, compression and indexing coexist in one elegant framework.

A Compressed Suffix Array turns the suffix array into a self-indexing structure —
the text, the index, and the compression all become one and the same.

### 698 FM-Index

The FM-Index is a powerful, compressed full-text index that combines the Burrows–Wheeler Transform (BWT), rank/select bit operations, and sampling to support fast substring search without storing the original text.
It achieves this while using space close to the entropy of the text, a key milestone in succinct data structures and modern search systems.

#### The Core Idea

The FM-Index is a practical realization of the Compressed Suffix Array (CSA).
It allows searching a pattern $P$ within a text $S$ in $O(m)$ time (for pattern length $m$) and uses space proportional to the compressed size of $S$.

It relies on the Burrows–Wheeler Transform (BWT) of $S$, which rearranges the text into a form that groups similar contexts, enabling efficient backward navigation.

#### Burrows–Wheeler Transform (BWT) Recap

Given text $S$ ending with a unique terminator \(\$\), the BWT is defined as:

$$
\text{BWT}[i] =
\begin{cases}
S[\text{SA}[i]-1], & \text{if } \text{SA}[i] > 0,\\
\text{\$}, & \text{if } \text{SA}[i] = 0.
\end{cases}
$$

For $S=\texttt{"banana\textdollar"}$, the suffix array is:
$$
\text{SA} = [6,\,5,\,3,\,1,\,0,\,4,\,2].
$$

and the BWT string becomes:
$$
\text{BWT} = \texttt{"annb\textdollar{}aa"}.
$$


#### Key Components

1. BWT String: The transformed text.
2. C array: For each character $c$, $C[c]$ = number of characters in $S$ lexicographically smaller than $c$.
3. Rank Structure: Supports $\text{rank}(c, i)$, number of occurrences of $c$ in $\text{BWT}[0:i]$.
4. Sampling Array: Periodically stores suffix array values for recovery of original positions.

#### Backward Search Algorithm

The fundamental operation of the FM-Index is backward search.
It processes the pattern $P = p_1 p_2 \dots p_m$ from right to left and maintains a range $[l, r]$ in the suffix array such that all suffixes starting with $P[i:m]$ fall within it.

Initialize:
$$
l = 0, \quad r = n - 1
$$

Then for $i = m, m-1, \dots, 1$:

$$
l = C[p_i] + \text{rank}(p_i, l - 1) + 1
$$

$$
r = C[p_i] + \text{rank}(p_i, r)
$$

When $l > r$, no match exists.
Otherwise, all occurrences of $P$ are found between $\text{SA}[l]$ and $\text{SA}[r]$.

#### Example: Search in "banana$"

Text $S = \text{"banana\$"}$
BWT = `annb$aa`
C = {$:0$, a:1, b:3, n:4}

Pattern $P = \text{"ana"}$

| Step | Char   | $[l, r]$ |
| ---- | ------ | -------- |
| Init |,      | [0, 6]   |
| a    | [1, 3] |          |
| n    | [4, 5] |          |
| a    | [2, 3] |          |

Match found at SA[2] = 1 and SA[3] = 3 → positions 1 and 3 in the original text.

#### Tiny Code (Simplified Prototype)

```python
def bwt_transform(s):
    s += "$"
    table = sorted(s[i:] + s[:i] for i in range(len(s)))
    return "".join(row[-1] for row in table)

def build_c_array(bwt):
    chars = sorted(set(bwt))
    count = 0
    C = {}
    for c in chars:
        C[c] = count
        count += bwt.count(c)
    return C

def rank(bwt, c, i):
    return bwt[:i + 1].count(c)

def backward_search(bwt, C, pattern):
    l, r = 0, len(bwt) - 1
    for ch in reversed(pattern):
        l = C[ch] + rank(bwt, ch, l - 1)
        r = C[ch] + rank(bwt, ch, r) - 1
        if l > r:
            return []
    return range(l, r + 1)

bwt = bwt_transform("banana")
C = build_c_array(bwt)
print("BWT:", bwt)
print("Matches:", list(backward_search(bwt, C, "ana")))
```

Output:

```
BWT: annb$aa
Matches: [2, 3]
```

#### Accessing Text Positions

Because we don't store the original suffix array, positions are recovered through LF-mapping (Last-to-First mapping):

$$
\text{LF}(i) = C[\text{BWT}[i]] + \text{rank}(\text{BWT}[i], i)
$$

Repeatedly applying LF-mapping moves backward through the text.
Every $t$-th suffix array value is stored explicitly for quick reconstruction.

#### Why It Works

The BWT clusters identical characters by context,
so rank and prefix boundaries can efficiently reconstruct
which parts of the text start with any given pattern.

Backward search turns the BWT into an implicit suffix array traversal —
no explicit storage of the suffixes is needed.

#### Complexity

| Operation         | Time               | Space                          |
| ----------------- | ------------------ | ------------------------------ |
| Pattern search    | $O(m \log \sigma)$ | $(1 + \epsilon) n H_k(S)$ bits |
| Locate            | $O(t \log \sigma)$ | $O(n/t)$ samples               |
| Extract substring | $O(\ell + \log n)$ | $O(n)$ compressed              |

Here $\sigma$ is alphabet size, and $H_k(S)$ is the $k$-th order entropy of the text.

#### Applications

| Domain               | Usage                                           |
| -------------------- | ----------------------------------------------- |
| Search engines   | Compressed text search with fast lookup         |
| Bioinformatics   | Genome alignment (e.g., BWA, Bowtie, FM-mapper) |
| Data compression | Core of self-indexing compressed storage        |
| Version control  | Deduplicated content retrieval                  |

#### Try It Yourself

1. Compute the BWT for `"mississippi$"` and build its FM-Index.
2. Run backward search for `"issi"`.
3. Modify the algorithm to return document IDs for a multi-document corpus.
4. Add rank/select bitvectors to optimize counting.
5. Compare FM-Index vs raw suffix array in memory usage.

#### A Gentle Proof (Why It Works)

The FM-Index leverages the invertibility of the BWT and the monotonicity of lexicographic order.
Backward search narrows the valid suffix range with each character,
using rank/select to simulate suffix array traversal inside a compressed domain.
Thus, text indexing becomes possible *without ever expanding the text*.

The FM-Index is the perfect marriage of compression and search —
small enough to fit a genome, powerful enough to index the web.

### 699 Directed Acyclic Word Graph (DAWG)

A Directed Acyclic Word Graph (DAWG) is a compact data structure that represents all substrings or words of a given text or dictionary.
It merges common suffixes or prefixes to reduce redundancy, forming a minimal deterministic finite automaton (DFA) for all suffixes of a string.
DAWGs are essential in text indexing, pattern search, auto-completion, and dictionary compression.

#### The Core Idea

A DAWG is essentially a suffix automaton or a minimal automaton that recognizes all substrings of a text.
It can be built incrementally in linear time and space proportional to the text length.

Each state in the DAWG represents a set of end positions of substrings,
and each edge is labeled by a character transition.

Key properties:

- Directed and acyclic (no loops except for transitions by characters)
- Deterministic (no ambiguity in transitions)
- Minimal (merges equivalent states)
- Recognizes all substrings of the input string

#### Example

Let's build a DAWG for the string `"aba"`.

All substrings:

```
a, b, ab, ba, aba
```

The minimal automaton has:

- States for distinct substring contexts
- Transitions labeled by `a`, `b`
- Merged common parts like shared suffixes `"a"` and `"ba"`

Resulting transitions:

```
(0) --a--> (1)
(1) --b--> (2)
(2) --a--> (3)
(1) --a--> (3)   (via suffix merging)
```

#### Suffix Automaton Connection

The DAWG for all substrings of a string is isomorphic to its suffix automaton.
Each state in the suffix automaton represents one or more substrings that share the same set of right contexts.

Formally, the automaton accepts all substrings of a given text $S$ such that:

$$
L(A) = { S[i:j] \mid 0 \le i < j \le |S| }
$$

#### Construction Algorithm (Suffix Automaton Method)

The DAWG can be built incrementally in $O(n)$ time using the suffix automaton algorithm.

Each step extends the automaton with the next character and updates transitions.

Algorithm sketch:

```python
def build_dawg(s):
    sa = [{}, -1, 0]  # transitions, suffix link, length
    last = 0
    for ch in s:
        cur = len(sa) // 3
        sa += [{}, 0, sa[3*last+2] + 1]
        p = last
        while p != -1 and ch not in sa[3*p]:
            sa[3*p][ch] = cur
            p = sa[3*p+1]
        if p == -1:
            sa[3*cur+1] = 0
        else:
            q = sa[3*p][ch]
            if sa[3*p+2] + 1 == sa[3*q+2]:
                sa[3*cur+1] = q
            else:
                clone = len(sa) // 3
                sa += [sa[3*q].copy(), sa[3*q+1], sa[3*p+2] + 1]
                while p != -1 and sa[3*p].get(ch, None) == q:
                    sa[3*p][ch] = clone
                    p = sa[3*p+1]
                sa[3*q+1] = sa[3*cur+1] = clone
        last = cur
    return sa
```

*(This is a compact suffix automaton builder, each node stores transitions and a suffix link.)*

#### Properties

| Property      | Description                                                       |
| ------------- | ----------------------------------------------------------------- |
| Deterministic | Each character transition is unique                               |
| Acyclic       | No cycles, except via transitions through the text                |
| Compact       | Merges equivalent suffix states                                   |
| Linear size   | At most $2n - 1$ states and $3n - 4$ edges for text of length $n$ |
| Incremental   | Supports online building                                          |

#### Visualization Example

For `"banana"`:

Each added letter expands the automaton:

- After `"b"` → states for `"b"`
- After `"ba"` → `"a"`, `"ba"`
- After `"ban"` → `"n"`, `"an"`, `"ban"`
- Common suffixes like `"ana"`, `"na"` get merged efficiently.

The result compactly encodes all 21 substrings of `"banana"` with about 11 states.

#### Applications

| Domain                      | Usage                                     |
| --------------------------- | ----------------------------------------- |
| Text indexing               | Store all substrings for fast queries     |
| Dictionary compression      | Merge common suffixes between words       |
| Pattern matching            | Test if a substring exists in $O(m)$ time |
| Bioinformatics              | Match gene subsequences                   |
| Natural language processing | Auto-complete and lexicon representation  |

#### Search Using DAWG

To check if a pattern $P$ is a substring of $S$:

```
state = start
for c in P:
    if c not in transitions[state]:
        return False
    state = transitions[state][c]
return True
```

Time complexity: $O(m)$, where $m$ is length of $P$.

#### Space and Time Complexity

| Operation                 | Time   | Space  |
| ------------------------- | ------ | ------ |
| Build                     | $O(n)$ | $O(n)$ |
| Search substring          | $O(m)$ | $O(1)$ |
| Count distinct substrings | $O(n)$ | $O(n)$ |

#### Counting Distinct Substrings

Each DAWG (suffix automaton) state represents multiple substrings.
The number of distinct substrings of a string $S$ is:

$$
\text{count} = \sum_{v} (\text{len}[v] - \text{len}[\text{link}[v]])
$$

Example for `"aba"`:

- $\text{count} = 5$ → substrings: `"a"`, `"b"`, `"ab"`, `"ba"`, `"aba"`

#### Try It Yourself

1. Build a DAWG for `"banana"` and count all substrings.
2. Modify the algorithm to support multiple words (a dictionary DAWG).
3. Visualize merged transitions, how common suffixes save space.
4. Extend to support prefix queries for auto-completion.
5. Measure time to query all substrings of `"mississippi"`.

#### A Gentle Proof (Why It Works)

Merging equivalent suffix states preserves language equivalence —
each state corresponds to a unique set of right contexts.
Since every substring of $S$ appears as a path in the automaton,
the DAWG encodes the entire substring set without redundancy.
Minimality ensures no two states represent the same substring set.

The Directed Acyclic Word Graph is the most compact way to represent all substrings of a string —
it is both elegant and efficient, standing at the crossroads of automata, compression, and search.

### 700 Wavelet Tree for Text

A Wavelet Tree is a succinct data structure that encodes a sequence of symbols while supporting rank, select, and access operations efficiently.
In text indexing, it is used as the core component of compressed suffix arrays and FM-indexes, allowing substring queries, frequency counts, and positional lookups without decompressing the text.

#### The Core Idea

Given a text $S$ over an alphabet $\Sigma$, a Wavelet Tree recursively partitions the alphabet and represents the text as a series of bitvectors indicating which half of the alphabet each symbol belongs to.

This allows hierarchical navigation through the text based on bits, enabling queries like:

- $\text{access}(i)$, what character is at position $i$
- $\text{rank}(c, i)$, how many times $c$ occurs up to position $i$
- $\text{select}(c, k)$, where the $k$-th occurrence of $c$ appears

All these are done in $O(\log |\Sigma|)$ time using compact bitvectors.

#### Construction

Suppose $S = \text{"banana"}$, with alphabet $\Sigma = {a, b, n}$.

1. Split alphabet:
   Left = {a}, Right = {b, n}

2. Build root bitvector:
   For each symbol in $S$,

   * write `0` if it belongs to Left,
   * write `1` if it belongs to Right.

   So:

   ```
   a b a n a n
   ↓ ↓ ↓ ↓ ↓ ↓
   0 1 0 1 0 1
   ```

   Root bitvector = `010101`

3. Recursively build subtrees:

   * Left child handles `aaa` (positions of `0`s)
   * Right child handles `bnn` (positions of `1`s)

Each node corresponds to a subset of characters,
and its bitvector encodes the mapping to child positions.

#### Example Query

Let's find $\text{rank}(\text{'n'}, 5)$, number of `'n'` in the first 5 characters of `"banana"`.

1. Start at root:

   * `'n'` is in Right half → follow bits `1`s
   * Count how many `1`s in first 5 bits of root (`01010`) → 2
   * Move to Right child with index 2

2. In Right child:

   * Alphabet {b, n}, `'n'` is in Right half again → follow `1`s
   * Bitvector of right child (`011`) → 2nd prefix has `1`
   * Count how many `1`s in first 2 bits → 1

Answer: `'n'` appears once up to position 5.

#### Tiny Code (Simplified)

```python
class WaveletTree:
    def __init__(self, s, alphabet=None):
        if alphabet is None:
            alphabet = sorted(set(s))
        if len(alphabet) == 1:
            self.symbol = alphabet[0]
            self.left = self.right = None
            self.bitvector = None
            return
        mid = len(alphabet) // 2
        left_set, right_set = set(alphabet[:mid]), set(alphabet[mid:])
        self.bitvector = [0 if ch in left_set else 1 for ch in s]
        left_s = [ch for ch in s if ch in left_set]
        right_s = [ch for ch in s if ch in right_set]
        self.left = WaveletTree(left_s, alphabet[:mid]) if left_s else None
        self.right = WaveletTree(right_s, alphabet[mid:]) if right_s else None

    def rank(self, c, i):
        if not self.bitvector or i <= 0:
            return 0
        if c == getattr(self, "symbol", None):
            return min(i, len(self.bitvector))
        bit = 0 if c in getattr(self.left, "alphabet", set()) else 1
        count = sum(1 for b in self.bitvector[:i] if b == bit)
        child = self.left if bit == 0 else self.right
        return child.rank(c, count) if child else 0

wt = WaveletTree("banana")
print(wt.rank('n', 5))
```

Output:

```
1
```

#### Visualization

```
                [a,b,n]
                010101
              /        \
          [a]          [b,n]
                      011
                     /   \
                  [b]   [n]
```

- Each level splits the alphabet range.
- Traversing bits leads to the symbol's leaf.

#### Operations Summary

| Operation    | Meaning             | Complexity       |
| ------------ | ------------------- | ---------------- |
| access(i)    | get $S[i]$          | $O(\log \sigma)$ |
| rank(c, i)   | # of c in $S[1..i]$ | $O(\log \sigma)$ |
| select(c, k) | position of k-th c  | $O(\log \sigma)$ |

Here $\sigma = |\Sigma|$ is alphabet size.

#### Integration with Text Indexing

Wavelet trees are integral to:

- FM-indexes, BWT rank/select operations
- Compressed Suffix Arrays, fast access to character intervals
- Document retrieval systems, word frequency and position queries
- Bioinformatics tools, efficient pattern matching on genome data

They allow random access over compressed text representations.

#### Complexity

| Property         | Value                                  |
| ---------------- | -------------------------------------- |
| Time per query   | $O(\log \sigma)$                       |
| Space usage      | $O(n \log \sigma)$ bits (uncompressed) |
| Space (succinct) | close to $n H_0(S)$ bits               |
| Construction     | $O(n \log \sigma)$                     |

#### Try It Yourself

1. Build a wavelet tree for `"mississippi"`.
2. Query $\text{rank}(\text{'s'}, 6)$ and $\text{select}(\text{'i'}, 3)$.
3. Extend it to support substring frequency queries.
4. Measure memory size versus a plain array.
5. Visualize the tree layers for each alphabet split.

#### A Gentle Proof (Why It Works)

At each level, bits partition the alphabet into halves.
Thus, rank and select operations translate into moving between levels,
adjusting indices using prefix counts.
Since the height of the tree is $\log \sigma$,
all queries finish in logarithmic time while maintaining perfect reversibility of data.

The Wavelet Tree unifies compression and search:
it encodes, indexes, and queries text —
all within the entropy limit of information itself.


