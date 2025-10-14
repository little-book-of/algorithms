# Chapter 5. Dynamic Programming 

# Section 41. DP basic and state transitions 

### 401 Fibonacci DP

Fibonacci is the hello-world of dynamic programming, a simple sequence that teaches the power of remembering past results. Instead of recomputing subproblems over and over, we store them. The result? A huge leap from exponential to linear time.

#### What Problem Are We Solving?

The Fibonacci sequence is defined as:

$$
F(0) = 0, \quad F(1) = 1, \quad F(n) = F(n-1) + F(n-2)
$$

A naive recursive version recomputes the same values many times. For example, `F(5)` calls `F(4)` and `F(3)`, but `F(4)` also calls `F(3)` again, wasteful repetition.

Our goal is to avoid recomputation by caching results. That's dynamic programming in a nutshell.

#### How Does It Work (Plain Language)?

Think of Fibonacci as a ladder. You can climb to step `n` only if you know the number of ways to reach `n-1` and `n-2`. Instead of recalculating those steps every time, record them once, then reuse.

There are two main flavors of DP:

| Approach                   | Description               | Example                   |
| -------------------------- | ------------------------- | ------------------------- |
| Top-Down (Memoization) | Recursion + caching       | Store results in an array |
| Bottom-Up (Tabulation) | Iteration from base cases | Build array from 0 up     |

Let's visualize the state filling:

| n | F(n-2) | F(n-1) | F(n) = F(n-1) + F(n-2) |
| - | ------ | ------ | ---------------------- |
| 0 | -      | -      | 0                      |
| 1 | -      | -      | 1                      |
| 2 | 0      | 1      | 1                      |
| 3 | 1      | 1      | 2                      |
| 4 | 1      | 2      | 3                      |
| 5 | 2      | 3      | 5                      |
| 6 | 3      | 5      | 8                      |

Each new value reuses two old ones, no redundant work.

#### Tiny Code (Easy Versions)

C (Bottom-Up Fibonacci)

```c
#include <stdio.h>

int main(void) {
    int n;
    printf("Enter n: ");
    scanf("%d", &n);

    if (n == 0) { printf("0\n"); return 0; }
    if (n == 1) { printf("1\n"); return 0; }

    long long dp[n + 1];
    dp[0] = 0;
    dp[1] = 1;

    for (int i = 2; i <= n; i++) {
        dp[i] = dp[i - 1] + dp[i - 2];
    }

    printf("Fibonacci(%d) = %lld\n", n, dp[n]);
    return 0;
}
```

Python (Memoized Fibonacci)

```python
from functools import lru_cache

@lru_cache(maxsize=None)
def fib(n):
    if n <= 1:
        return n
    return fib(n-1) + fib(n-2)

n = int(input("Enter n: "))
print("Fibonacci(", n, ") =", fib(n))
```

#### Why It Matters

- Demonstrates state definition: F(n) = F(n-1) + F(n-2)
- Introduces overlapping subproblems and optimal substructure
- First step toward mastering DP intuition
- Reduces time complexity from exponential O(2ⁿ) to linear O(n)

You learn that solving once and remembering is better than solving a hundred times.

#### Step-by-Step Example

| Step | Calculation    | Memo Table (Partial) |
| ---- | -------------- | -------------------- |
| Base | F(0)=0, F(1)=1 | [0, 1, _, _, _, _]   |
| F(2) | F(1)+F(0)=1    | [0, 1, 1, _, _, _]   |
| F(3) | F(2)+F(1)=2    | [0, 1, 1, 2, _, _]   |
| F(4) | F(3)+F(2)=3    | [0, 1, 1, 2, 3, _]   |
| F(5) | F(4)+F(3)=5    | [0, 1, 1, 2, 3, 5]   |

#### Try It Yourself

1. Write Fibonacci recursively without memoization. Measure calls.
2. Add a dictionary or array for memoization, compare speeds.
3. Convert recursive to iterative (tabulation).
4. Optimize space: store only two variables instead of full array.
5. Print the full table of computed values.

#### Test Cases

| n  | Expected Output | Notes                 |
| -- | --------------- | --------------------- |
| 0  | 0               | Base case             |
| 1  | 1               | Base case             |
| 2  | 1               | 1 + 0                 |
| 5  | 5               | Sequence: 0,1,1,2,3,5 |
| 10 | 55              | Smooth growth check   |

#### Complexity

- Time: O(n) for both memoization and tabulation
- Space: O(n) for table; O(1) if optimized

Fibonacci DP is the simplest proof that remembering pays off, it's where dynamic programming begins, and efficiency is born.

### 402 Climbing Stairs

The climbing stairs problem is a friendly cousin of Fibonacci, same recurrence, different story. You're standing at the bottom of a staircase with `n` steps. You can climb 1 step or 2 steps at a time. How many distinct ways can you reach the top?

This is one of the most intuitive gateways into dynamic programming: define states, relate them recursively, and reuse past computations.

#### What Problem Are We Solving?

We want the number of distinct ways to reach step `n`.

You can reach step `n` by:

- taking 1 step from `n-1`, or
- taking 2 steps from `n-2`.

So the recurrence is:
$$
dp[n] = dp[n-1] + dp[n-2]
$$

with base cases:
$$
dp[0] = 1, \quad dp[1] = 1
$$

This is structurally identical to Fibonacci, but with a combinatorial interpretation.

#### How Does It Work (Plain Language)?

Think of each step as a checkpoint. To reach step `n`, you must come from either of the two prior checkpoints. If you already know how many ways there are to reach those, just add them.

Let's illustrate with a table:

| Step (n) | Ways to Reach | Explanation                                 |
| -------- | ------------- | ------------------------------------------- |
| 0        | 1             | Stay at ground                              |
| 1        | 1             | Single step                                 |
| 2        | 2             | (1+1), (2)                                  |
| 3        | 3             | (1+1+1), (1+2), (2+1)                       |
| 4        | 5             | (1+1+1+1), (2+1+1), (1+2+1), (1+1+2), (2+2) |
| 5        | 8             | Previous two sums: 5 = 3+2                  |

Each new value is the sum of the previous two.

#### Tiny Code (Easy Versions)

C (Bottom-Up Climbing Stairs)

```c
#include <stdio.h>

int main(void) {
    int n;
    printf("Enter number of steps: ");
    scanf("%d", &n);

    if (n == 0 || n == 1) {
        printf("Ways: 1\n");
        return 0;
    }

    long long dp[n + 1];
    dp[0] = 1;
    dp[1] = 1;

    for (int i = 2; i <= n; i++) {
        dp[i] = dp[i - 1] + dp[i - 2];
    }

    printf("Ways to climb %d steps: %lld\n", n, dp[n]);
    return 0;
}
```

Python (Space Optimized)

```python
n = int(input("Enter number of steps: "))

if n == 0 or n == 1:
    print(1)
else:
    a, b = 1, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    print("Ways to climb", n, "steps:", b)
```

#### Why It Matters

- Demonstrates state transitions: `dp[i]` depends on `dp[i-1]` and `dp[i-2]`
- Teaches bottom-up thinking and base case setup
- Shows how recurrence translates to counting problems
- Connects combinatorics with DP intuition

Climbing stairs is a great mental bridge between pure math (recurrence) and applied reasoning (counting paths).

#### Step-by-Step Example

Let's trace `n = 5`:

| i | dp[i-2] | dp[i-1] | dp[i] = dp[i-1] + dp[i-2] |
| - | ------- | ------- | ------------------------- |
| 0 | -       | -       | 1                         |
| 1 | -       | -       | 1                         |
| 2 | 1       | 1       | 2                         |
| 3 | 1       | 2       | 3                         |
| 4 | 2       | 3       | 5                         |
| 5 | 3       | 5       | 8                         |

So, 8 ways to climb 5 steps.

#### Try It Yourself

1. Modify the code to allow 1, 2, or 3 steps at a time.
2. Print the entire `dp` table for small `n`.
3. Compare recursive vs iterative solutions.
4. Try to derive a formula, notice the Fibonacci pattern.
5. What if each step had a cost? Adapt it to Min Cost Climb.

#### Test Cases

| n | Expected Output | Ways                  |
| - | --------------- | --------------------- |
| 0 | 1               | Do nothing            |
| 1 | 1               | [1]                   |
| 2 | 2               | [1+1], [2]            |
| 3 | 3               | [1+1+1], [1+2], [2+1] |
| 4 | 5               | All combinations      |
| 5 | 8               | Grows like Fibonacci  |

#### Complexity

- Time: O(n)
- Space: O(n), reducible to O(1) with two variables

Climbing stairs shows that dynamic programming isn't just math, it's about recognizing patterns in movement, growth, and memory.

### 403 Grid Paths

The grid path problem is a gentle step into 2D dynamic programming, where states depend on neighbors, not just previous elements. Imagine standing in the top-left corner of a grid, moving only right or down, trying to count how many ways lead to the bottom-right corner.

Each cell's value is determined by paths reaching it from above or from the left, a perfect metaphor for how DP builds solutions layer by layer.

#### What Problem Are We Solving?

Given an `m × n` grid, find the number of distinct paths from `(0, 0)` to `(m-1, n-1)` when you can move only:

- Right `(x, y+1)`
- Down `(x+1, y)`

The recurrence:
$$
dp[i][j] = dp[i-1][j] + dp[i][j-1]
$$
with base cases:
$$
dp[0][j] = 1, \quad dp[i][0] = 1
$$
(since only one way exists along the first row or column)

#### How Does It Work (Plain Language)?

Think of the grid like a city map, every intersection `(i, j)` can be reached from either the north `(i-1, j)` or the west `(i, j-1)`. So total routes = routes from north + routes from west.

Let's visualize a `3×3` grid (0-indexed):

| Cell (i,j) | Ways | Explanation    |
| ---------- | ---- | -------------- |
| (0,0)      | 1    | Start          |
| (0,1)      | 1    | Only from left |
| (0,2)      | 1    | Only from left |
| (1,0)      | 1    | Only from top  |
| (1,1)      | 2    | (0,1)+(1,0)=2  |
| (1,2)      | 3    | (0,2)+(1,1)=3  |
| (2,0)      | 1    | Only from top  |
| (2,1)      | 3    | (1,1)+(2,0)=3  |
| (2,2)      | 6    | (1,2)+(2,1)=6  |

So `dp[2][2] = 6` → 6 distinct paths.

#### Tiny Code (Easy Versions)

C (2D DP Table)

```c
#include <stdio.h>

int main(void) {
    int m, n;
    printf("Enter rows and cols: ");
    scanf("%d %d", &m, &n);

    long long dp[m][n];

    for (int i = 0; i < m; i++) dp[i][0] = 1;
    for (int j = 0; j < n; j++) dp[0][j] = 1;

    for (int i = 1; i < m; i++) {
        for (int j = 1; j < n; j++) {
            dp[i][j] = dp[i-1][j] + dp[i][j-1];
        }
    }

    printf("Unique paths: %lld\n", dp[m-1][n-1]);
    return 0;
}
```

Python (Space Optimized)

```python
m, n = map(int, input("Enter rows and cols: ").split())
dp = [1] * n

for _ in range(1, m):
    for j in range(1, n):
        dp[j] += dp[j - 1]

print("Unique paths:", dp[-1])
```

#### Why It Matters

- Teaches 2D DP grids
- Builds intuition for problems on lattices, matrices, grids
- Foundation for min-cost path, maze traversal, robot movement
- Encourages space optimization from 2D → 1D

From counting paths to optimizing them, this grid is your DP canvas.

#### Step-by-Step Example

For a `3×3` grid:

| i\j | 0 | 1 | 2 |
| --- | - | - | - |
| 0   | 1 | 1 | 1 |
| 1   | 1 | 2 | 3 |
| 2   | 1 | 3 | 6 |

`dp[2][2] = 6` → six unique routes.

#### Try It Yourself

1. Modify the code to handle obstacles (`0` = block, `1` = open).
2. Print the DP table.
3. Implement using recursion + memoization.
4. Add a condition for moving right, down, and diagonal.
5. Compare with combinatorial formula: $\binom{m+n-2}{m-1}$.

#### Test Cases

| Grid Size | Expected Paths | Notes                   |
| --------- | -------------- | ----------------------- |
| 1×1       | 1              | Only starting cell      |
| 2×2       | 2              | Right→Down, Down→Right  |
| 3×3       | 6              | Classic case            |
| 3×4       | 10             | Combinatorics check     |
| 4×4       | 20             | Pascal triangle pattern |

#### Complexity

- Time: O(m×n)
- Space: O(m×n), reducible to O(n)

Grid Paths reveal the essence of DP, every position depends on simpler ones. From here, you'll learn to minimize, maximize, and traverse with purpose.

### 404 Min Cost Path

The Min Cost Path problem is where counting meets optimization. Instead of asking *"How many ways can I reach the end?"*, we ask *"What's the cheapest way to get there?"*. You're moving across a grid, cell by cell, each with a cost, and your goal is to reach the bottom-right corner while minimizing the total cost.

This is one of the most fundamental path optimization problems in dynamic programming.

#### What Problem Are We Solving?

Given a matrix `cost[m][n]`, where each cell represents a non-negative cost, find the minimum total cost path from `(0, 0)` to `(m-1, n-1)`, moving only right, down, or (optionally) diagonally down-right.

The recurrence:
$$
dp[i][j] = cost[i][j] + \min(dp[i-1][j], dp[i][j-1])
$$
If diagonal moves are allowed:
$$
dp[i][j] = cost[i][j] + \min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
$$

Base case:
$$
dp[0][0] = cost[0][0]
$$

#### How Does It Work (Plain Language)?

Imagine hiking across a grid of terrain, where each cell has an energy cost. Every move you make adds to your total cost. You always want to choose the path that keeps your running total as small as possible.

The DP table records the minimum cost to reach each cell, building from the top-left to the bottom-right.

Let's see an example grid:

| Cell  | Cost |
| ----- | ---- |
| (0,0) | 1    |
| (0,1) | 3    |
| (0,2) | 5    |
| (1,0) | 2    |
| (1,1) | 1    |
| (1,2) | 2    |
| (2,0) | 4    |
| (2,1) | 3    |
| (2,2) | 1    |

We fill `dp[i][j]` = cost to reach `(i,j)`:

| i\j | 0 | 1 | 2 |
| --- | - | - | - |
| 0   | 1 | 4 | 9 |
| 1   | 3 | 2 | 4 |
| 2   | 7 | 5 | 5 |

Minimum cost = 5

#### Tiny Code (Easy Versions)

C (Bottom-Up DP)

```c
#include <stdio.h>
#define MIN(a,b) ((a)<(b)?(a):(b))

int main(void) {
    int m, n;
    printf("Enter rows and cols: ");
    scanf("%d %d", &m, &n);

    int cost[m][n];
    printf("Enter cost matrix:\n");
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            scanf("%d", &cost[i][j]);

    int dp[m][n];
    dp[0][0] = cost[0][0];

    for (int i = 1; i < m; i++) dp[i][0] = dp[i-1][0] + cost[i][0];
    for (int j = 1; j < n; j++) dp[0][j] = dp[0][j-1] + cost[0][j];

    for (int i = 1; i < m; i++)
        for (int j = 1; j < n; j++)
            dp[i][j] = cost[i][j] + MIN(dp[i-1][j], dp[i][j-1]);

    printf("Min cost: %d\n", dp[m-1][n-1]);
    return 0;
}
```

Python (Optional Diagonal Move)

```python
def min_cost_path(cost):
    m, n = len(cost), len(cost[0])
    dp = [[0]*n for _ in range(m)]
    dp[0][0] = cost[0][0]

    for i in range(1, m):
        dp[i][0] = dp[i-1][0] + cost[i][0]
    for j in range(1, n):
        dp[0][j] = dp[0][j-1] + cost[0][j]

    for i in range(1, m):
        for j in range(1, n):
            dp[i][j] = cost[i][j] + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])

    return dp[-1][-1]

cost = [
    [1, 3, 5],
    [2, 1, 2],
    [4, 3, 1]
$$

print("Min cost path:", min_cost_path(cost))
```

#### Why It Matters

- Transitions from counting paths to optimizing paths
- Introduces minimization recurrence
- Basis for many grid problems (e.g. maze solving, image traversal, shortest paths)
- Builds intuition for weighted DP

This problem is a stepping stone toward Dijkstra's and Bellman-Ford in graphs.

#### Step-by-Step Example

| Cell  | From Top | From Left | Min | Total Cost |
| ----- | -------- | --------- | --- | ---------- |
| (0,0) | -        | -         | -   | 1          |
| (0,1) | -        | 1+3       | 4   | 4          |
| (1,0) | 1+2      | -         | 3   | 3          |
| (1,1) | 4+1      | 3+1       | 4   | 4          |
| (2,2) | ...      | ...       | ... | 5          |

Answer: 5

#### Try It Yourself

1. Add diagonal moves, compare results.
2. Add blocked cells (infinite cost).
3. Modify to find maximum cost path.
4. Reconstruct the path using a `parent` table.
5. Use a priority queue (Dijkstra) for non-grid graphs.

#### Test Cases

| Grid                      | Expected Min Cost | Notes         |
| ------------------------- | ----------------- | ------------- |
| [[1]]                     | 1                 | Single cell   |
| [[1,2],[3,4]]             | 7                 | 1→2→4         |
| [[1,3,5],[2,1,2],[4,3,1]] | 5                 | Optimal route |
| [[5,9],[4,2]]             | 11                | 5→4→2         |

#### Complexity

- Time: O(m×n)
- Space: O(m×n), reducible to O(n)

Min Cost Path turns the grid into a map of decisions, each cell asks, "What's the cheapest way to reach me?" and the DP table answers with calm precision.

### 405 Coin Change (Count Ways)

You have coin denominations and an amount. How many distinct ways can you make that amount if you can use unlimited coins of each type? We count combinations where order does not matter. For example, with coins [1, 2, 5] there are 4 ways to make 5:
`5`, `2+2+1`, `2+1+1+1`, `1+1+1+1+1`.

#### What Problem Are We Solving?

Given an array `coins[]` and an integer `amount`, compute the number of combinations to form `amount` using unlimited copies of each coin.

State and recurrence:

$$
dp[x] = \text{number of ways to make sum } x
$$
$$
dp[0] = 1
$$
$$
\text{for each coin } c:\quad \text{for } x \text{ from } c \text{ to } \text{amount}:\quad dp[x] \mathrel{+=} dp[x - c]
$$

Why this order: iterating coins on the outside ensures each combination is counted once. If you loop amounts on the outside and coins inside, you would count permutations.

#### How Does It Work (Plain Language)?

Think of building the total from left to right. For each coin value `c`, you ask: if I must use `c` at least once, how many ways remain to fill `x - c`? Add those to the ways already known. Move forward increasing `x`, and repeat for the next coin. The table fills like a rolling tally.

Example with coins `[1, 2, 5]` and `amount = 5`:

| After processing | dp[0] | dp[1] | dp[2] | dp[3] | dp[4] | dp[5] | Explanation                       |
| ---------------- | ----- | ----- | ----- | ----- | ----- | ----- | --------------------------------- |
| Init             | 1     | 0     | 0     | 0     | 0     | 0     | One way to make 0: choose nothing |
| Coin 1           | 1     | 1     | 1     | 1     | 1     | 1     | Using 1s only                     |
| Coin 2           | 1     | 1     | 2     | 2     | 3     | 3     | Add ways that end with a 2        |
| Coin 5           | 1     | 1     | 2     | 2     | 3     | 4     | Add ways that end with a 5        |

Answer is `dp[5] = 4`.

#### Tiny Code (Easy Versions)

C (1D DP, combinations)

```c
#include <stdio.h>

int main(void) {
    int n, amount;
    printf("Enter number of coin types and amount: ");
    scanf("%d %d", &n, &amount);

    int coins[n];
    printf("Enter coin values: ");
    for (int i = 0; i < n; i++) scanf("%d", &coins[i]);

    // Use long long to avoid overflow for large counts
    long long dp[amount + 1];
    for (int x = 0; x <= amount; x++) dp[x] = 0;
    dp[0] = 1;

    for (int i = 0; i < n; i++) {
        int c = coins[i];
        for (int x = c; x <= amount; x++) {
            dp[x] += dp[x - c];
        }
    }

    printf("Number of ways: %lld\n", dp[amount]);
    return 0;
}
```

Python (1D DP, combinations)

```python
coins = list(map(int, input("Enter coin values: ").split()))
amount = int(input("Enter amount: "))

dp = [0] * (amount + 1)
dp[0] = 1

for c in coins:
    for x in range(c, amount + 1):
        dp[x] += dp[x - c]

print("Number of ways:", dp[amount])
```

#### Why It Matters

- Introduces the idea of unbounded knapsack counting
- Shows how loop ordering controls whether you count combinations or permutations
- Forms a foundation for many counting DPs such as integer partitions and dice sum counts
- Encourages space optimization with a single dimension

#### Step-by-Step Example

Coins `[1, 3, 4]`, amount `6`:

| Step   | Update             | dp array snapshot (index is amount) |
| ------ | ------------------ | ----------------------------------- |
| Init   | dp[0]=1            | [1, 0, 0, 0, 0, 0, 0]               |
| Coin 1 | fill x=1..6        | [1, 1, 1, 1, 1, 1, 1]               |
| Coin 3 | x=3..6 add dp[x-3] | [1, 1, 1, 2, 2, 2, 3]               |
| Coin 4 | x=4..6 add dp[x-4] | [1, 1, 1, 2, 3, 3, 4]               |

Answer: `dp[6] = 4`.

#### Try It Yourself

1. Switch to permutations counting: loop `x` outside and `coins` inside. Compare results.
2. Add a cap per coin type and convert to a bounded version.
3. Sort coins and print one valid combination using a parent pointer array.
4. Use modulo arithmetic to avoid overflow: for example `10^9+7`.
5. Extend to count ways for every `x` from `0` to `amount` and print the full table.

#### Test Cases

| Coins   | Amount | Expected Ways | Notes                             |
| ------- | ------ | ------------- | --------------------------------- |
| []      | 0      | 1             | One empty way                     |
| []      | 5      | 0             | No coins cannot form positive sum |
| [1]     | 4      | 1             | Only 1+1+1+1                      |
| [2]     | 3      | 0             | Odd cannot be formed              |
| [1,2,5] | 5      | 4             | Classic example                   |
| [2,3,7] | 12     | 4             | Combinations only                 |

#### Complexity

- Time: $O(n \times \text{amount})$ where `n` is number of coin types
- Space: $O(\text{amount})$ with 1D DP

Coin Change counting teaches you how state order and loop order shape the meaning of a DP. Once you feel this pattern, many counting problems become straightforward.

### 406 Coin Change (Min Coins)

Now we shift from *counting ways* to *finding the fewest coins*. Given a target amount and coin denominations, how can we form the sum using the minimum number of coins? This version of coin change turns counting into optimization, a small twist with big impact.

#### What Problem Are We Solving?

Given `coins[]` and a total `amount`, find the minimum number of coins needed to make up that amount.
If it's impossible, return -1.

We define:
$$
dp[x] = \text{minimum coins to make sum } x
$$
with base case:
$$
dp[0] = 0
$$
Recurrence:
$$
dp[x] = \min_{c \in coins,\ c \le x} (dp[x - c] + 1)
$$

Each `dp[x]` asks: "If I take coin `c`, what's the best I can do with the remainder?"

#### How Does It Work (Plain Language)?

You start from `0` and climb up, building the cheapest way to reach every amount.
For each `x`, you try all coins `c` ≤ `x`. If you can make `x-c`, add one coin and see if that's better than your current best.

It's like choosing the shortest route to a destination using smaller hops.

Example: coins = `[1, 3, 4]`, amount = 6

| Amount (x) | dp[x] | Explanation |
| ---------- | ----- | ----------- |
| 0          | 0     | Base case   |
| 1          | 1     | 1×1         |
| 2          | 2     | 1+1         |
| 3          | 1     | 3           |
| 4          | 1     | 4           |
| 5          | 2     | 1+4         |
| 6          | 2     | 3+3         |

So minimum = `2`.

#### Tiny Code (Easy Versions)

C (Bottom-Up DP)

```c
#include <stdio.h>
#define INF 1000000
#define MIN(a,b) ((a)<(b)?(a):(b))

int main(void) {
    int n, amount;
    printf("Enter number of coins and amount: ");
    scanf("%d %d", &n, &amount);

    int coins[n];
    printf("Enter coin values: ");
    for (int i = 0; i < n; i++) scanf("%d", &coins[i]);

    int dp[amount + 1];
    for (int x = 1; x <= amount; x++) dp[x] = INF;
    dp[0] = 0;

    for (int x = 1; x <= amount; x++) {
        for (int i = 0; i < n; i++) {
            int c = coins[i];
            if (x - c >= 0) dp[x] = MIN(dp[x], dp[x - c] + 1);
        }
    }

    if (dp[amount] == INF)
        printf("Not possible\n");
    else
        printf("Min coins: %d\n", dp[amount]);

    return 0;
}
```

Python (Straightforward DP)

```python
coins = list(map(int, input("Enter coin values: ").split()))
amount = int(input("Enter amount: "))

INF = float('inf')
dp = [INF] * (amount + 1)
dp[0] = 0

for x in range(1, amount + 1):
    for c in coins:
        if x - c >= 0:
            dp[x] = min(dp[x], dp[x - c] + 1)

print("Min coins:" if dp[amount] != INF else "Not possible", end=" ")
print(dp[amount] if dp[amount] != INF else "")
```

#### Why It Matters

- Core unbounded optimization DP
- Shows minimization recurrence with base infinity
- Illustrates subproblem dependency: `dp[x]` depends on smaller sums
- Connects directly to Knapsack, shortest path, and DP + BFS hybrids

This version teaches you to mix greedy intuition with DP correctness.

#### Step-by-Step Example

Coins `[1, 3, 4]`, amount = 6

| x | Try 1     | Try 3     | Try 4     | dp[x] |
| - | --------- | --------- | --------- | ----- |
| 0 | -         | -         | -         | 0     |
| 1 | dp[0]+1=1 | -         | -         | 1     |
| 2 | dp[1]+1=2 | -         | -         | 2     |
| 3 | dp[2]+1=3 | dp[0]+1=1 | -         | 1     |
| 4 | dp[3]+1=2 | dp[1]+1=2 | dp[0]+1=1 | 1     |
| 5 | dp[4]+1=2 | dp[2]+1=3 | dp[1]+1=2 | 2     |
| 6 | dp[5]+1=3 | dp[3]+1=2 | dp[2]+1=3 | 2     |

Answer = `dp[6] = 2`

#### Try It Yourself

1. Add code to reconstruct the actual coin set.
2. Compare greedy vs DP for coins `[1, 3, 4]`, amount = 6.
3. Modify to handle limited coin supply.
4. Use recursion + memoization and compare runtime.
5. Try edge cases: amount smaller than smallest coin.

#### Test Cases

| Coins    | Amount | Expected Output | Notes           |
| -------- | ------ | --------------- | --------------- |
| [1]      | 3      | 3               | Only 1s         |
| [2]      | 3      | -1              | Impossible      |
| [1,3,4]  | 6      | 2               | 3+3 or 4+1+1    |
| [1,2,5]  | 11     | 3               | 5+5+1           |
| [2,5,10] | 0      | 0               | No coins needed |

#### Complexity

- Time: O(n×amount)
- Space: O(amount)

Coin Change (Min Coins) is a masterclass in thinking minimally, every subproblem is a small decision toward the most efficient path.

### 407 Knapsack 0/1

The 0/1 Knapsack problem is one of the crown jewels of dynamic programming. You're given a backpack with limited capacity and a set of items, each with a weight and a value. You must decide which items to pack so that the total value is maximized without exceeding the weight limit. You can either take an item (1) or leave it (0), hence the name.

#### What Problem Are We Solving?

Given:

- `n` items, each with `weight[i]` and `value[i]`
- capacity `W`

Find the maximum total value you can carry:
$$
dp[i][w] = \text{max value using first i items with capacity } w
$$

Recurrence:
$$
dp[i][w] = \max(
dp[i-1][w], \quad
value[i-1] + dp[i-1][w - weight[i-1]]
)
$$
if `weight[i-1] <= w`, else `dp[i][w] = dp[i-1][w]`.

Base:
$$
dp[0][w] = 0, \quad dp[i][0] = 0
$$

#### How Does It Work (Plain Language)?

Think of your backpack as a *budget of space*. Each item is a trade-off:

- Include it → gain its value but lose capacity
- Exclude it → keep capacity for others

You make this decision for every item and every possible capacity.

We build a table where each cell `dp[i][w]` stores the best value you can achieve with the first `i` items and total capacity `w`.

Example:
Items = [(w=1,v=1), (w=3,v=4), (w=4,v=5), (w=5,v=7)], W = 7

| $i/w$ | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| --- | - | - | - | - | - | - | - | - |
| 0   | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1   | 0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| 2   | 0 | 1 | 1 | 4 | 5 | 5 | 5 | 5 |
| 3   | 0 | 1 | 1 | 4 | 5 | 6 | 6 | 9 |
| 4   | 0 | 1 | 1 | 4 | 5 | 7 | 8 | 9 |

Answer = `9` (items 2 + 3)

#### Tiny Code (Easy Versions)

C (2D DP Table)

```c
#include <stdio.h>

#define MAX(a,b) ((a)>(b)?(a):(b))

int main(void) {
    int n, W;
    printf("Enter number of items and capacity: ");
    scanf("%d %d", &n, &W);

    int wt[n], val[n];
    printf("Enter weights: ");
    for (int i = 0; i < n; i++) scanf("%d", &wt[i]);
    printf("Enter values: ");
    for (int i = 0; i < n; i++) scanf("%d", &val[i]);

    int dp[n + 1][W + 1];

    for (int i = 0; i <= n; i++) {
        for (int w = 0; w <= W; w++) {
            if (i == 0 || w == 0) dp[i][w] = 0;
            else if (wt[i-1] <= w)
                dp[i][w] = MAX(val[i-1] + dp[i-1][w - wt[i-1]], dp[i-1][w]);
            else
                dp[i][w] = dp[i-1][w];
        }
    }

    printf("Max value: %d\n", dp[n][W]);
    return 0;
}
```

Python (Space Optimized)

```python
weights = list(map(int, input("Enter weights: ").split()))
values = list(map(int, input("Enter values: ").split()))
W = int(input("Enter capacity: "))
n = len(weights)

dp = [0] * (W + 1)

for i in range(n):
    for w in range(W, weights[i] - 1, -1):
        dp[w] = max(dp[w], values[i] + dp[w - weights[i]])

print("Max value:", dp[W])
```

#### Why It Matters

- Teaches choice-based DP: include or exclude
- Core of resource allocation, subset selection, budgeting problems
- Foundation for advanced DPs (subset sum, partition, scheduling)
- Introduces 2D → 1D space optimization

This problem embodies the essence of decision-making in DP: to take or not to take.

#### Step-by-Step Example

Items: (w,v) = (1,1), (3,4), (4,5), (5,7), W=7

| Step | Capacity | Action            | dp                 |
| ---- | -------- | ----------------- | ------------------ |
| i=1  | w≥1      | Take (1,1)        | +1 value           |
| i=2  | w≥3      | Take (3,4)        | Replace low combos |
| i=3  | w≥4      | Combine (3+4) = 9 | Max found          |
| i=4  | w=7      | Can't beat 9      | Done               |

#### Try It Yourself

1. Print selected items using a traceback table.
2. Compare 2D vs 1D versions.
3. Add constraint for exact weight match.
4. Try variants: maximize weight, minimize count, etc.
5. Modify for fractional weights → Greedy Fractional Knapsack.

#### Test Cases

| Weights   | Values     | Capacity | Expected | Notes           |
| --------- | ---------- | -------- | -------- | --------------- |
| [1,2,3]   | [10,15,40] | 6        | 65       | Take all        |
| [2,3,4,5] | [3,4,5,6]  | 5        | 7        | (2,3)           |
| [1,3,4,5] | [1,4,5,7]  | 7        | 9        | (3,4)           |
| [2,5]     | [5,10]     | 3        | 5        | Only first fits |

#### Complexity

- Time: O(n×W)
- Space: O(n×W) → O(W) optimized

0/1 Knapsack is the archetype of dynamic programming, it's all about balancing choices, constraints, and rewards.

### 408 Knapsack Unbounded

The Unbounded Knapsack problem is the free refill version of knapsack. You still want to maximize value under a capacity limit, but now each item can be chosen multiple times. It's like packing snacks, you can grab as many as you want, as long as they fit in the bag.

#### What Problem Are We Solving?

Given:

- `n` items with `weight[i]` and `value[i]`
- capacity `W`
- Unlimited copies of each item

Find the maximum value achievable without exceeding `W`.

State:
$$
dp[w] = \text{max value for capacity } w
$$

Recurrence:
$$
dp[w] = \max_{i: weight[i] \le w} (dp[w - weight[i]] + value[i])
$$

Base:
$$
dp[0] = 0
$$

Notice that this is similar to 0/1 Knapsack, but here we reuse items. The difference lies in the order of iteration.

#### How Does It Work (Plain Language)?

Think of capacity `w` as a budget. For each capacity, you check all items, if one fits, you see what happens when you reuse it. Unlike 0/1 Knapsack (where each item can only be used once per combination), Unbounded Knapsack allows multiple selections.

| Capacity (w) | Best Value | Explanation          |
| ------------ | ---------- | -------------------- |
| 0            | 0          | Empty                |
| 1            | 15         | 1 copy of item(1,15) |
| 2            | 30         | 2 copies             |
| 3            | 45         | 3 copies             |
| 4            | 60         | 4 copies             |

(If all items have same ratio, you'll fill with the best one.)

Example:
Items: `(w,v)` = (2,4), (3,7), (4,9), W = 7

- dp[2] = 4
- dp[3] = 7
- dp[4] = 9
- dp[5] = max(dp[3]+4, dp[2]+7) = 11
- dp[6] = max(dp[4]+4, dp[3]+7, dp[2]+9) = 14
- dp[7] = max(dp[5]+4, dp[4]+7, dp[3]+9) = 16

Answer = 16

#### Tiny Code (Easy Versions)

C (1D DP, Unbounded)

```c
#include <stdio.h>
#define MAX(a,b) ((a)>(b)?(a):(b))

int main(void) {
    int n, W;
    printf("Enter number of items and capacity: ");
    scanf("%d %d", &n, &W);

    int wt[n], val[n];
    printf("Enter weights: ");
    for (int i = 0; i < n; i++) scanf("%d", &wt[i]);
    printf("Enter values: ");
    for (int i = 0; i < n; i++) scanf("%d", &val[i]);

    int dp[W + 1];
    for (int w = 0; w <= W; w++) dp[w] = 0;

    for (int i = 0; i < n; i++) {
        for (int w = wt[i]; w <= W; w++) {
            dp[w] = MAX(dp[w], val[i] + dp[w - wt[i]]);
        }
    }

    printf("Max value: %d\n", dp[W]);
    return 0;
}
```

Python (Simple Bottom-Up)

```python
weights = list(map(int, input("Enter weights: ").split()))
values = list(map(int, input("Enter values: ").split()))
W = int(input("Enter capacity: "))

dp = [0] * (W + 1)

for i in range(len(weights)):
    for w in range(weights[i], W + 1):
        dp[w] = max(dp[w], values[i] + dp[w - weights[i]])

print("Max value:", dp[W])
```

#### Why It Matters

- Demonstrates unbounded usage of elements
- Basis for coin change (min), rod cutting, integer break
- Highlights the importance of iteration order in DP
- Connects counting (how many ways) to optimization (best way)

This is where combinatorial explosion becomes manageable.

#### Step-by-Step Example

Items: (2,4), (3,7), (4,9), W = 7

| w | dp[w] | Best Choice |
| - | ----- | ----------- |
| 0 | 0     | -           |
| 1 | 0     | none fits   |
| 2 | 4     | (2)         |
| 3 | 7     | (3)         |
| 4 | 9     | (4)         |
| 5 | 11    | (2+3)       |
| 6 | 14    | (3+3)       |
| 7 | 16    | (3+4)       |

Answer: 16

#### Try It Yourself

1. Print the items used (store parent choice).
2. Compare 0/1 and Unbounded outputs.
3. Add a limit on copies, hybrid knapsack.
4. Change objective: minimize number of items.
5. Apply to Rod Cutting problem.

#### Test Cases

| Weights   | Values     | W  | Expected | Notes             |
| --------- | ---------- | -- | -------- | ----------------- |
| [2,3,4]   | [4,7,9]    | 7  | 16       | 3+4               |
| [1,2,3]   | [10,15,40] | 6  | 90       | six 1s            |
| [5,10,20] | [10,30,50] | 20 | 100      | four 5s or one 20 |
| [2,5]     | [5,10]     | 3  | 5        | only one 2        |

#### Complexity

- Time: O(n×W)
- Space: O(W)

Unbounded Knapsack is your first taste of infinite choice under constraint, a powerful idea that flows through many DP designs.

### 409 Longest Increasing Subsequence (DP)

The Longest Increasing Subsequence (LIS) problem is a classic, it's all about finding the longest chain of numbers that strictly increases. You don't have to keep them consecutive, just in order. This is a foundational DP problem that blends state definition, transitions, and comparisons beautifully.

#### What Problem Are We Solving?

Given an array `arr[]` of length `n`, find the length of the longest increasing subsequence, a sequence of indices `i₁ < i₂ < ... < iₖ` such that:
$$
arr[i₁] < arr[i₂] < \cdots < arr[iₖ]
$$

We want maximum length.

Recurrence:
$$
dp[i] = 1 + \max(dp[j]) \quad \text{for all } j < i \text{ where } arr[j] < arr[i]
$$
Otherwise:
$$
dp[i] = 1
$$

Base case:
$$
dp[0] = 1
$$

Answer:
$$
\max_i dp[i]
$$

#### How Does It Work (Plain Language)?

You look at each number and ask: *"Can I extend an increasing sequence ending before me?"*
If yes, take the longest one that fits and extend it by one.

It's like building towers, each number stacks on top of a smaller one, extending the tallest possible stack.

Example: `arr = [10, 22, 9, 33, 21, 50, 41, 60]`

| i | arr[i] | dp[i] | Reason            |
| - | ------ | ----- | ----------------- |
| 0 | 10     | 1     | start             |
| 1 | 22     | 2     | 10→22             |
| 2 | 9      | 1     | no smaller before |
| 3 | 33     | 3     | 10→22→33          |
| 4 | 21     | 2     | 10→21             |
| 5 | 50     | 4     | 10→22→33→50       |
| 6 | 41     | 4     | 10→22→33→41       |
| 7 | 60     | 5     | 10→22→33→50→60    |

Answer = 5

#### Tiny Code (Easy Versions)

C (O(n²) DP)

```c
#include <stdio.h>

#define MAX(a,b) ((a)>(b)?(a):(b))

int main(void) {
    int n;
    printf("Enter number of elements: ");
    scanf("%d", &n);

    int arr[n];
    printf("Enter array: ");
    for (int i = 0; i < n; i++) scanf("%d", &arr[i]);

    int dp[n];
    for (int i = 0; i < n; i++) dp[i] = 1;

    int ans = 1;
    for (int i = 1; i < n; i++) {
        for (int j = 0; j < i; j++) {
            if (arr[j] < arr[i])
                dp[i] = MAX(dp[i], dp[j] + 1);
        }
        ans = MAX(ans, dp[i]);
    }

    printf("Length of LIS: %d\n", ans);
    return 0;
}
```

Python (Simple DP)

```python
arr = list(map(int, input("Enter array: ").split()))
n = len(arr)
dp = [1] * n

for i in range(n):
    for j in range(i):
        if arr[j] < arr[i]:
            dp[i] = max(dp[i], dp[j] + 1)

print("Length of LIS:", max(dp))
```

#### Why It Matters

- Core sequence DP, compares pairs, tracks best chain
- Demonstrates O(n²) DP thinking
- Foundation for LCS, Edit Distance, Patience Sorting (O(n log n))
- Applied in stock analysis, genome sequences, and chain problems

This problem teaches "look back and extend", a key DP instinct.

#### Step-by-Step Example

`arr = [3, 10, 2, 1, 20]`

| i | arr[i] | dp[i] | Best Chain |
| - | ------ | ----- | ---------- |
| 0 | 3      | 1     | [3]        |
| 1 | 10     | 2     | [3,10]     |
| 2 | 2      | 1     | [2]        |
| 3 | 1      | 1     | [1]        |
| 4 | 20     | 3     | [3,10,20]  |

Answer = 3

#### Try It Yourself

1. Print the actual LIS using a `parent` array.
2. Convert to non-decreasing LIS (≤ instead of <).
3. Compare with O(n log n) binary search version.
4. Adapt for longest decreasing subsequence.
5. Apply to 2D pairs (Russian Doll Envelopes).

#### Test Cases

| arr                      | Expected | Notes                    |
| ------------------------ | -------- | ------------------------ |
| [1,2,3,4,5]              | 5        | Already increasing       |
| [5,4,3,2,1]              | 1        | Only one element         |
| [3,10,2,1,20]            | 3        | [3,10,20]                |
| [10,22,9,33,21,50,41,60] | 5        | Classic example          |
| [2,2,2,2]                | 1        | Strictly increasing only |

#### Complexity

- Time: O(n²)
- Space: O(n)

LIS is the melody of DP, every element listens to its predecessors, finds harmony, and extends the tune to its fullest length.

### 410 Edit Distance (Levenshtein)

The Edit Distance (or Levenshtein distance) problem measures how different two strings are by counting the minimum number of operations needed to transform one into the other. The allowed operations are:

- Insert
- Delete
- Replace

It's the foundation of spell checkers, DNA sequence alignment, and fuzzy search, anywhere we need to measure "how close" two sequences are.

#### What Problem Are We Solving?

Given two strings `A` and `B`, find the minimum number of operations required to convert `A` → `B`.

Let:

- `A` has length `m`
- `B` has length `n`

State:
$$
dp[i][j] = \text{min edits to convert } A[0..i-1] \text{ to } B[0..j-1]
$$

Recurrence:

1. If `A[i-1] == B[j-1]`:
   $$
   dp[i][j] = dp[i-1][j-1]
   $$
2. Else, take min of the three operations:
   $$
   dp[i][j] = 1 + \min(
   dp[i-1][j],     \text{ (delete)}
   dp[i][j-1],     \text{ (insert)}
   dp[i-1][j-1]    \text{ (replace)}
   )
   $$

Base:
$$
dp[0][j] = j,\quad dp[i][0] = i
$$
(empty string conversions)

#### How Does It Work (Plain Language)?

Imagine editing a word character by character. At each step, compare the current letters:

- If they match → no cost, move diagonally.
- If they differ → choose the cheapest fix (insert, delete, replace).

The DP table builds all prefix transformations, from small strings to full ones.

Example:
`A = "kitten"`, `B = "sitting"`

| Step          | Operation   | Result |
| ------------- | ----------- | ------ |
| Replace k → s | sitten  |        |
| Replace e → i | sittin  |        |
| Insert g      | sitting |        |

Answer = 3

#### Tiny Code (Easy Versions)

C (2D DP)

```c
#include <stdio.h>
#define MIN3(a,b,c) ((a<b?a:b)<c?(a<b?a:b):c)

int main(void) {
    char A[100], B[100];
    printf("Enter string A: ");
    scanf("%s", A);
    printf("Enter string B: ");
    scanf("%s", B);

    int m = 0, n = 0;
    while (A[m]) m++;
    while (B[n]) n++;

    int dp[m + 1][n + 1];

    for (int i = 0; i <= m; i++) dp[i][0] = i;
    for (int j = 0; j <= n; j++) dp[0][j] = j;

    for (int i = 1; i <= m; i++) {
        for (int j = 1; j <= n; j++) {
            if (A[i-1] == B[j-1]) dp[i][j] = dp[i-1][j-1];
            else dp[i][j] = 1 + MIN3(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]);
        }
    }

    printf("Edit distance: %d\n", dp[m][n]);
    return 0;
}
```

Python (Simple Version)

```python
A = input("Enter string A: ")
B = input("Enter string B: ")

m, n = len(A), len(B)
dp = [[0]*(n+1) for _ in range(m+1)]

for i in range(m+1): dp[i][0] = i
for j in range(n+1): dp[0][j] = j

for i in range(1, m+1):
    for j in range(1, n+1):
        if A[i-1] == B[j-1]:
            dp[i][j] = dp[i-1][j-1]
        else:
            dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])

print("Edit distance:", dp[m][n])
```

#### Why It Matters

- Illustrates 2D DP on strings
- Introduces transformation problems
- Forms the backbone of spell correction, DNA alignment, diff tools
- Beautifully captures state = prefix lengths pattern

Edit Distance is the dictionary definition of "step-by-step transformation."

#### Step-by-Step Example

`A = "intention"`, `B = "execution"`

| $A/B$ | "" | e | x | e | c | u | t | i | o | n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---- |
| ""  | 0  | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
| i   | 1  | 1 | 2 | 3 | 4 | 5 | 6 | 6 | 7 | 8 |
| n   | 2  | 2 | 2 | 3 | 4 | 5 | 6 | 7 | 7 | 7 |
| t   | 3  | 3 | 3 | 3 | 4 | 5 | 5 | 6 | 7 | 8 |
| e   | 4  | 3 | 4 | 3 | 4 | 5 | 6 | 6 | 7 | 8 |
| n   | 5  | 4 | 4 | 4 | 4 | 5 | 6 | 7 | 7 | 7 |
| t   | 6  | 5 | 5 | 5 | 5 | 5 | 5 | 6 | 7 | 8 |
| i   | 7  | 6 | 6 | 6 | 6 | 6 | 6 | 5 | 6 | 7 |
| o   | 8  | 7 | 7 | 7 | 7 | 7 | 7 | 6 | 5 | 6 |
| n   | 9  | 8 | 8 | 8 | 8 | 8 | 8 | 7 | 6 | 5 |

Answer = 5 edits

#### Try It Yourself

1. Print actual edit sequence (backtrack).
2. Add costs: assign different weights for insert/delete/replace.
3. Try case-insensitive variant.
4. Compare with Longest Common Subsequence.
5. Implement recursive + memoized version.

#### Test Cases

| A        | B         | Expected | Notes            |
| -------- | --------- | -------- | ---------------- |
| "kitten" | "sitting" | 3        | classic          |
| "horse"  | "ros"     | 3        | leetcode         |
| "flaw"   | "lawn"    | 2        | replace + insert |
| "abc"    | "yabd"    | 2        | insert + replace |
| ""       | "abc"     | 3        | all inserts      |

#### Complexity

- Time: O(m×n)
- Space: O(m×n), reducible to O(n)

Edit Distance teaches precision in DP: every cell means *"smallest change to fix this prefix"*. It's the language of correction, one letter at a time.

# Section 42. Classic Problems 

### 411 0/1 Knapsack

The 0/1 Knapsack is one of the most iconic problems in dynamic programming. It's the perfect example of decision-making under constraints, each item can either be taken or left, but never split or repeated. The goal is to maximize total value within a fixed capacity.

This version focuses on understanding choice, capacity, and optimal substructure, the three pillars of DP.

#### What Problem Are We Solving?

Given:

- `n` items, each with weight `w[i]` and value `v[i]`
- a knapsack of capacity `W`

We want:
$$
\text{maximize total value} \quad \sum v[i]
$$
subject to
$$
\sum w[i] \le W
$$
and each item can be used at most once.

State definition:
$$
dp[i][w] = \text{max value using first } i \text{ items with capacity } w
$$

Recurrence:
$$
dp[i][w] =
\begin{cases}
dp[i-1][w], & \text{if } w_{i-1} > w,\\
\max\big(dp[i-1][w],\ dp[i-1][w - w_{i-1}] + v_{i-1}\big), & \text{otherwise.}
\end{cases}
$$

Base case:
$$
dp[0][w] = 0, \quad dp[i][0] = 0
$$

Answer:
$$
dp[n][W]
$$

#### How Does It Work (Plain Language)?

At every step, you ask:
"Should I take this item or leave it?"
If it fits, compare:

- Not taking it → stick with previous best (`dp[i-1][w]`)
- Taking it → add its value plus best value for remaining capacity (`dp[i-1][w - weight[i]] + value[i]`)

The DP table stores the best possible value at every sub-capacity for each subset of items.

| Item | Weight | Value |
| ---- | ------ | ----- |
| 1    | 1      | 1     |
| 2    | 3      | 4     |
| 3    | 4      | 5     |
| 4    | 5      | 7     |

Capacity = 7 → Answer = 9 (items 2 + 3)

#### Tiny Code (Easy Versions)

C (2D DP Table)

```c
#include <stdio.h>
#define MAX(a,b) ((a)>(b)?(a):(b))

int main(void) {
    int n, W;
    printf("Enter number of items and capacity: ");
    scanf("%d %d", &n, &W);

    int wt[n], val[n];
    printf("Enter weights: ");
    for (int i = 0; i < n; i++) scanf("%d", &wt[i]);
    printf("Enter values: ");
    for (int i = 0; i < n; i++) scanf("%d", &val[i]);

    int dp[n + 1][W + 1];

    for (int i = 0; i <= n; i++) {
        for (int w = 0; w <= W; w++) {
            if (i == 0 || w == 0) dp[i][w] = 0;
            else if (wt[i-1] <= w)
                dp[i][w] = MAX(val[i-1] + dp[i-1][w - wt[i-1]], dp[i-1][w]);
            else
                dp[i][w] = dp[i-1][w];
        }
    }

    printf("Max value: %d\n", dp[n][W]);
    return 0;
}
```

Python (1D Optimized)

```python
weights = list(map(int, input("Enter weights: ").split()))
values = list(map(int, input("Enter values: ").split()))
W = int(input("Enter capacity: "))
n = len(weights)

dp = [0] * (W + 1)

for i in range(n):
    for w in range(W, weights[i] - 1, -1):
        dp[w] = max(dp[w], dp[w - weights[i]] + values[i])

print("Max value:", dp[W])
```

#### Why It Matters

- Introduces decision-based DP: take or skip
- Builds on recurrence intuition (state transition)
- Forms basis for subset sum, equal partition, and resource allocation
- Teaches capacity-dependent states

It's the first time you feel the *tension* between greedy desire and constrained reality, a DP classic.

#### Step-by-Step Example

Items: (w,v) = (1,1), (3,4), (4,5), (5,7), W = 7

| $i/w$ | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| --- | - | - | - | - | - | - | - | - |
| 0   | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1   | 0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| 2   | 0 | 1 | 1 | 4 | 5 | 5 | 5 | 5 |
| 3   | 0 | 1 | 1 | 4 | 5 | 6 | 6 | 9 |
| 4   | 0 | 1 | 1 | 4 | 5 | 7 | 8 | 9 |

Answer = 9

#### Try It Yourself

1. Add code to reconstruct chosen items.
2. Compare 2D vs 1D DP outputs.
3. Modify to minimize weight for a given value.
4. Visualize table transitions for small inputs.
5. Experiment with large weights, test performance.

#### Test Cases

| Weights   | Values     | W | Expected | Notes           |
| --------- | ---------- | - | -------- | --------------- |
| [1,2,3]   | [10,15,40] | 6 | 65       | all items fit   |
| [2,3,4,5] | [3,4,5,6]  | 5 | 7        | (2,3)           |
| [1,3,4,5] | [1,4,5,7]  | 7 | 9        | (3,4)           |
| [2,5]     | [5,10]     | 3 | 5        | only first fits |

#### Complexity

- Time: O(n×W)
- Space: O(n×W) → O(W) (optimized)

0/1 Knapsack is the heartbeat of DP, every decision echoes the fundamental trade-off: *to take or not to take*.

### 412 Subset Sum

The Subset Sum problem is a fundamental example of boolean dynamic programming. Instead of maximizing or minimizing, we simply ask "Is it possible?", can we pick a subset of numbers that adds up to a given target?

This problem forms the foundation for many combinatorial DP problems such as Equal Partition, Count of Subsets, Target Sum, and even Knapsack itself.

#### What Problem Are We Solving?

Given:

- An array `arr[]` of `n` positive integers
- A target sum `S`

Determine whether there exists a subset of `arr[]` whose elements sum to exactly `S`.

We define:
$$
dp[i][s] = \text{true if subset of first } i \text{ elements can form sum } s
$$

Recurrence:

- If `arr[i-1] > s`:
  [
  dp[i][s] = dp[i-1][s]
  ]
- Else:
  [
  dp[i][s] = dp[i-1][s] \lor dp[i-1][s - arr[i-1]]
  ]

Base cases:
$$
dp[0][0] = \text{true}, \quad dp[0][s>0] = \text{false}
$$

Answer:
$$
dp[n][S]
$$

#### How Does It Work (Plain Language)?

Think of it as a yes/no table:

- Rows → items
- Columns → sums

Each cell asks: *"Can I form sum `s` using the first `i` items?"*
The answer comes from either skipping or including the current item.

Example: `arr = [2, 3, 7, 8, 10]`, `S = 11`

| $i/Sum$ | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
| ----- | - | - | - | - | - | - | - | - | - | - | -- | -- |
| 0     | T | F | F | F | F | F | F | F | F | F | F  | F  |
| 1 (2) | T | F | T | F | F | F | F | F | F | F | F  | F  |
| 2 (3) | T | F | T | T | F | T | F | F | F | F | F  | F  |
| 3 (7) | T | F | T | T | F | T | F | T | T | F | T  | T  |

`dp[5][11] = True` → `[3, 8]` is one valid subset.

#### Tiny Code (Easy Versions)

C (2D Boolean Table)

```c
#include <stdio.h>

int main(void) {
    int n, S;
    printf("Enter number of elements and target sum: ");
    scanf("%d %d", &n, &S);

    int arr[n];
    printf("Enter elements: ");
    for (int i = 0; i < n; i++) scanf("%d", &arr[i]);

    int dp[n + 1][S + 1];
    for (int i = 0; i <= n; i++) dp[i][0] = 1;
    for (int s = 1; s <= S; s++) dp[0][s] = 0;

    for (int i = 1; i <= n; i++) {
        for (int s = 1; s <= S; s++) {
            if (arr[i-1] > s)
                dp[i][s] = dp[i-1][s];
            else
                dp[i][s] = dp[i-1][s] || dp[i-1][s - arr[i-1]];
        }
    }

    printf("Subset sum %s possible\n", dp[n][S] ? "is" : "is not");
    return 0;
}
```

Python (1D Optimization)

```python
arr = list(map(int, input("Enter elements: ").split()))
S = int(input("Enter target sum: "))

dp = [False] * (S + 1)
dp[0] = True

for num in arr:
    for s in range(S, num - 1, -1):
        dp[s] = dp[s] or dp[s - num]

print("Subset sum is possible" if dp[S] else "Not possible")
```

#### Why It Matters

- Introduces boolean DP (true/false states)
- Foundation for Equal Partition, Target Sum, and Count Subsets
- Closely related to 0/1 Knapsack but without values
- Perfect exercise for learning state dependency

This problem captures the logic of feasibility: *"If I could make `s - arr[i]` before, then I can make `s` now."*

#### Step-by-Step Example

Array = [2, 3, 7, 8, 10], S = 11

| Step | Consider                       | New True Sums |
| ---- | ------------------------------ | ------------- |
| 2    | [0,2]                          | {2}           |
| 3    | [0,2,3,5]                      | {3,5}         |
| 7    | [0,2,3,5,7,9,10,12]            | {7,9,10}      |
| 8    | [0,2,3,5,7,8,9,10,11,12,13,15] | {11}          |

Found 11.

#### Try It Yourself

1. Print one valid subset using a parent pointer table.
2. Count the total number of valid subsets (convert to count DP).
3. Try with duplicates, does it change anything?
4. Modify to check if sum is divisible by `k`.
5. Add negative numbers (use offset shifting).

#### Test Cases

| arr          | S  | Expected | Notes     |
| ------------ | -- | -------- | --------- |
| [2,3,7,8,10] | 11 | True     | 3+8       |
| [1,2,3]      | 5  | True     | 2+3       |
| [1,2,5]      | 4  | False    | no subset |
| [1,1,1,1]    | 2  | True     | 1+1       |
| [5,2,6,4]    | 13 | True     | 5+4+4     |

#### Complexity

- Time: O(n×S)
- Space: O(n×S) → O(S) (optimized)

Subset Sum is a cornerstone of DP, a yes/no version of Knapsack that teaches how logic flows through states, one sum at a time.

### 413 Equal Partition

The Equal Partition problem asks a natural question: can we divide a set of numbers into two subsets with equal sum?
It's a direct application of Subset Sum, reframed as a partitioning challenge.
If the total sum is even, we check if there's a subset that sums to half, that ensures the other subset sums to the same.

#### What Problem Are We Solving?

Given:

- An array `arr[]` of `n` positive integers

Determine whether it can be partitioned into two subsets whose sums are equal.

Let total sum be `S`.
We need to check:

- If `S` is odd → impossible
- If `S` is even → check if Subset Sum to `S/2` is possible

So the problem reduces to:

$$
\text{Is there a subset of } arr[] \text{ with sum } = S/2?
$$

We use the same recurrence from Subset Sum:

$$
dp[i][s] = dp[i-1][s] \lor dp[i-1][s - arr[i-1]]
$$

Base:
$$
dp[0][0] = \text{true}
$$

Answer:
$$
dp[n][S/2]
$$

#### How Does It Work (Plain Language)?

1. Compute total sum `S`.
2. If `S` is odd → cannot split evenly.
3. Otherwise, use Subset Sum DP to check if we can reach `S/2`.
   If yes, one subset forms `S/2`, and the remaining numbers automatically form the other half.

Example: `arr = [1, 5, 11, 5]`

- Sum = 22
- Target = 11
- Can we make 11?
  Yes → `[11]` and `[1,5,5]` are two halves.

#### Tiny Code (Easy Versions)

C (2D DP Table)

```c
#include <stdio.h>

int main(void) {
    int n;
    printf("Enter number of elements: ");
    scanf("%d", &n);

    int arr[n];
    printf("Enter elements: ");
    for (int i = 0; i < n; i++) scanf("%d", &arr[i]);

    int sum = 0;
    for (int i = 0; i < n; i++) sum += arr[i];

    if (sum % 2 != 0) {
        printf("Cannot partition into equal sum subsets\n");
        return 0;
    }

    int target = sum / 2;
    int dp[n + 1][target + 1];

    for (int i = 0; i <= n; i++) dp[i][0] = 1;
    for (int s = 1; s <= target; s++) dp[0][s] = 0;

    for (int i = 1; i <= n; i++) {
        for (int s = 1; s <= target; s++) {
            if (arr[i-1] > s)
                dp[i][s] = dp[i-1][s];
            else
                dp[i][s] = dp[i-1][s] || dp[i-1][s - arr[i-1]];
        }
    }

    printf("Equal partition %s possible\n", dp[n][target] ? "is" : "is not");
    return 0;
}
```

Python (1D Space Optimization)

```python
arr = list(map(int, input("Enter elements: ").split()))
S = sum(arr)

if S % 2 != 0:
    print("Cannot partition into equal subsets")
else:
    target = S // 2
    dp = [False] * (target + 1)
    dp[0] = True

    for num in arr:
        for s in range(target, num - 1, -1):
            dp[s] = dp[s] or dp[s - num]

    print("Equal partition is possible" if dp[target] else "Not possible")
```

#### Why It Matters

- Builds directly on Subset Sum
- Demonstrates problem reduction in DP
- Useful for balanced partitioning, load balancing, and fair division
- Teaches thinking in terms of state feasibility

Equal Partition shows how yes/no DPs can solve seemingly complex questions with simple logic.

#### Step-by-Step Example

`arr = [1, 5, 11, 5]`

1. `S = 22` → `S/2 = 11`
2. Use Subset Sum DP to check if `11` can be formed
3. True → subsets `[11]` and `[1,5,5]`

Another case: `arr = [1, 2, 3, 5]`

- `S = 11` (odd) → cannot partition.

#### Try It Yourself

1. Print the actual subsets (traceback table).
2. Try arrays with duplicates.
3. Compare with total sum odd case.
4. Add constraint: must use at least one element in each subset.
5. Visualize dp table for small arrays.

#### Test Cases

| arr         | Sum | Expected | Notes        |
| ----------- | --- | -------- | ------------ |
| [1,5,11,5]  | 22  | True     | 11 and 11    |
| [1,2,3,5]   | 11  | False    | odd total    |
| [3,3,3,3]   | 12  | True     | split evenly |
| [2,2,2,2,2] | 10  | True     | 5+5          |
| [1,1,3,4,7] | 16  | True     | 8+8          |

#### Complexity

- Time: O(n × S/2)
- Space: O(S/2) (optimized)

Equal Partition is the first real taste of reduction in dynamic programming, take a bigger problem, express it as Subset Sum, and solve with the same machinery.


### 414 Count of Subsets with Sum

The Count of Subsets with Sum problem extends the Subset Sum idea. Instead of asking *"Is it possible to form this sum?"*, we ask *"In how many ways can we form it?"*.
This transforms a boolean DP into a counting DP, where each state accumulates the number of combinations that yield a given sum.

#### What Problem Are We Solving?

Given:

- An array `arr[]` of `n` positive integers
- A target sum `S`

We want the number of subsets whose elements sum exactly to `S`.

We define the state:

$$
dp[i][s] = \text{number of ways to form sum } s \text{ using first } i \text{ elements}
$$

The recurrence:

$$
dp[i][s] =
\begin{cases}
dp[i-1][s], & \text{if } arr[i-1] > s,\\
dp[i-1][s] + dp[i-1][s - arr[i-1]], & \text{otherwise.}
\end{cases}
$$


Base cases:

$$
dp[0][0] = 1, \quad dp[0][s>0] = 0
$$

Final answer:

$$
dp[n][S]
$$

#### How Does It Work (Plain Language)

Each element gives two paths: include or exclude.
If you include it, you count all subsets that formed `s - arr[i-1]` before.
If you exclude it, you inherit all subsets that already formed `s`.
So each cell accumulates total combinations from both branches.

Example: `arr = [2, 3, 5, 6, 8, 10]`, `S = 10`

Ways to form 10:

- `{10}`
- `{2, 8}`
- `{2, 3, 5}`

Answer = 3

#### Tiny Code (Easy Versions)

C (2D DP Table)

```c
#include <stdio.h>

int main(void) {
    int n, S;
    printf("Enter number of elements and target sum: ");
    scanf("%d %d", &n, &S);

    int arr[n];
    printf("Enter elements: ");
    for (int i = 0; i < n; i++) scanf("%d", &arr[i]);

    int dp[n + 1][S + 1];

    for (int i = 0; i <= n; i++) dp[i][0] = 1;
    for (int s = 1; s <= S; s++) dp[0][s] = 0;

    for (int i = 1; i <= n; i++) {
        for (int s = 0; s <= S; s++) {
            if (arr[i-1] > s)
                dp[i][s] = dp[i-1][s];
            else
                dp[i][s] = dp[i-1][s] + dp[i-1][s - arr[i-1]];
        }
    }

    printf("Number of subsets: %d\n", dp[n][S]);
    return 0;
}
```

Python (1D Space Optimized)

```python
arr = list(map(int, input("Enter elements: ").split()))
S = int(input("Enter target sum: "))

dp = [0] * (S + 1)
dp[0] = 1

for num in arr:
    for s in range(S, num - 1, -1):
        dp[s] += dp[s - num]

print("Number of subsets:", dp[S])
```

#### Why It Matters

- Extends Subset Sum from feasibility to counting
- Foundation for Target Sum, Equal Partition Count, and Combinatorics DP
- Shows how a small change in recurrence changes meaning
- Demonstrates accumulation instead of boolean OR

This is where DP transitions from *logic* to *combinatorics*, from "can I?" to "how many ways?"

#### Step-by-Step Example

`arr = [2, 3, 5, 6, 8, 10]`, `S = 10`

| i | arr[i] | Ways to form 10 | Explanation          |
| - | ------ | --------------- | -------------------- |
| 1 | 2      | 0               | cannot reach 10 yet  |
| 2 | 3      | 0               | 2+3=5 only           |
| 3 | 5      | 1               | {5}                  |
| 4 | 6      | 1               | {10}                 |
| 5 | 8      | 2               | {2,8}, {10}          |
| 6 | 10     | 3               | {10}, {2,8}, {2,3,5} |

Answer = 3

#### A Gentle Proof (Why It Works)

We build `dp[i][s]` using the inclusion-exclusion principle:

To form sum `s` using first `i` items, two possibilities exist:

1. Exclude `arr[i-1]`: all subsets that form `s` remain valid
   $$ dp[i-1][s] $$
2. Include `arr[i-1]`: each subset that formed `s - arr[i-1]` now forms `s`
   $$ dp[i-1][s - arr[i-1]] $$

Thus:

$$
dp[i][s] = dp[i-1][s] + dp[i-1][s - arr[i-1]]
$$

No double counting occurs since each element is processed once, contributing to exactly one branch per subproblem.
By building layer by layer, `dp[n][S]` accumulates all valid subset combinations summing to `S`.

#### Try It Yourself

1. Print all valid subsets using recursive backtracking.
2. Modify the DP to count subsets with sum ≤ target.
3. Add duplicates and compare results.
4. Apply modulo $10^9 + 7$ to handle large counts.
5. Extend to count subsets with sum difference = D.

#### Test Cases

| arr            | S  | Expected | Notes                |
| -------------- | -- | -------- | -------------------- |
| [2,3,5,6,8,10] | 10 | 3        | {10}, {2,8}, {2,3,5} |
| [1,1,1,1]      | 2  | 6        | choose any 2         |
| [1,2,3]        | 3  | 2        | {3}, {1,2}           |
| [1,2,5]        | 4  | 0        | no subset            |
| [2,4,6,10]     | 16 | 2        | {6,10}, {2,4,10}     |

#### Complexity

- Time: $O(n \times S)$
- Space: $O(S)$

The Count of Subsets with Sum problem is a perfect illustration of how dynamic programming can evolve from feasibility to enumeration, counting every path that leads to success.

### 415 Target Sum

The Target Sum problem combines Subset Sum and sign assignment, instead of selecting elements, you assign + or − to each one so that their total equals a target value. It's a beautiful example of how DP can turn algebraic constraints into combinatorial counting.

#### What Problem Are We Solving?

Given:

- An array `arr[]` of `n` non-negative integers
- A target value `T`

Count the number of ways to assign + or − signs to elements so that:

$$
a_1 \pm a_2 \pm a_3 \pm \ldots \pm a_n = T
$$

Each element must appear once with either sign.

We define:

- Let total sum be $S = \sum arr[i]$

If we split into two subsets $P$ (positive) and $N$ (negative), we have:

$$
\begin{cases}
P + N = S \
P - N = T
\end{cases}
$$

Solve these equations:

$$
P = \frac{S + T}{2}
$$

So the problem becomes count subsets whose sum = (S + T)/2.

If $(S + T)$ is odd or $T > S$, answer = 0 (impossible).

#### Key Idea

Convert the sign problem into a subset counting problem:

$$
\text{Count subsets with sum } P = \frac{S + T}{2}
$$

Then use the recurrence from Count of Subsets with Sum:

$$
dp[i][p] =
\begin{cases}
dp[i-1][p], & \text{if } arr[i-1] > p,\\
dp[i-1][p] + dp[i-1][p - arr[i-1]], & \text{otherwise.}
\end{cases}
$$


Base case:

$$
dp[0][0] = 1
$$

Answer:

$$
dp[n][P]
$$

#### How Does It Work (Plain Language)

Think of each element as being placed on one of two sides: positive or negative.
Instead of directly simulating signs, we compute how many subsets sum to $(S + T)/2$.
That subset represents all numbers assigned `+`; the rest implicitly become `−`.

Example:
`arr = [1, 1, 2, 3]`, `T = 1`
Sum `S = 7` → $P = (7 + 1)/2 = 4$

So we count subsets summing to 4:

- `{1, 3}`
- `{1, 1, 2}`
  Answer = 2

#### Tiny Code (Easy Versions)

C (2D DP Table)

```c
#include <stdio.h>

int main(void) {
    int n, T;
    printf("Enter number of elements and target: ");
    scanf("%d %d", &n, &T);

    int arr[n];
    printf("Enter elements: ");
    for (int i = 0; i < n; i++) scanf("%d", &arr[i]);

    int S = 0;
    for (int i = 0; i < n; i++) S += arr[i];

    if ((S + T) % 2 != 0 || T > S) {
        printf("No solutions\n");
        return 0;
    }

    int P = (S + T) / 2;
    int dp[n + 1][P + 1];

    for (int i = 0; i <= n; i++) dp[i][0] = 1;
    for (int j = 1; j <= P; j++) dp[0][j] = 0;

    for (int i = 1; i <= n; i++) {
        for (int j = 0; j <= P; j++) {
            if (arr[i-1] > j)
                dp[i][j] = dp[i-1][j];
            else
                dp[i][j] = dp[i-1][j] + dp[i-1][j - arr[i-1]];
        }
    }

    printf("Number of ways: %d\n", dp[n][P]);
    return 0;
}
```

Python (1D Space Optimized)

```python
arr = list(map(int, input("Enter elements: ").split()))
T = int(input("Enter target: "))
S = sum(arr)

if (S + T) % 2 != 0 or T > S:
    print("No solutions")
else:
    P = (S + T) // 2
    dp = [0] * (P + 1)
    dp[0] = 1

    for num in arr:
        for s in range(P, num - 1, -1):
            dp[s] += dp[s - num]

    print("Number of ways:", dp[P])
```

#### Why It Matters

- Transforms sign assignment into subset counting
- Reinforces algebraic manipulation in DP
- Foundation for expression evaluation, partition problems, and probabilistic sums
- Demonstrates how mathematical reformulation simplifies state design

It's a powerful example of turning a tricky ± sum problem into a familiar counting DP.

#### Step-by-Step Example

`arr = [1, 1, 2, 3]`, `T = 1`
$S = 7$ → $P = 4$

Subsets summing to 4:

- `{1, 3}`
- `{1, 1, 2}`

Answer = 2

#### A Gentle Proof (Why It Works)

Let the positive subset sum be $P$ and negative subset sum be $N$.
We have:

$$
P - N = T \quad \text{and} \quad P + N = S
$$

Adding both:
$$
2P = S + T \implies P = \frac{S + T}{2}
$$

Thus, any valid assignment of signs corresponds exactly to one subset summing to $P$.
Every subset of sum $P$ defines a unique sign configuration:

- Numbers in $P$ → positive
- Numbers not in $P$ → negative

So counting subsets with sum $P$ is equivalent to counting all valid sign assignments.

#### Try It Yourself

1. Handle zeros (they double the count).
2. Return all possible sign configurations.
3. Check with negative `T` (same symmetry).
4. Compare brute-force enumeration with DP result.
5. Modify for constraints like "at least k positive numbers".

#### Test Cases

| arr       | T | Expected | Notes                  |
| --------- | - | -------- | ---------------------- |
| [1,1,2,3] | 1 | 2        | {1,3}, {1,1,2}         |
| [1,2,3]   | 0 | 2        | {1,2,3} and {-1,-2,-3} |
| [2,2,2,2] | 4 | 6        | many ways              |
| [1,1,1,1] | 0 | 6        | symmetric partitions   |
| [5,3,2,1] | 5 | 2        | {2,3}, {5}             |

#### Complexity

- Time: $O(n \times P)$
- Space: $O(P)$

The Target Sum problem shows how algebra and DP meet: by reinterpreting signs as subsets, you turn a puzzle of pluses and minuses into a clean combinatorial count.

### 416 Unbounded Knapsack

The Unbounded Knapsack problem is the unlimited version of the classic 0/1 Knapsack. Here, each item can be chosen multiple times, as long as total weight stays within capacity. It's one of the most elegant illustrations of reusable states in dynamic programming.

#### What Problem Are We Solving?

Given:

- `n` items, each with `weight[i]` and `value[i]`
- a knapsack of capacity `W`

Find the maximum total value achievable without exceeding capacity `W`.
Each item can be used any number of times.

We define the state:

$$
dp[w] = \text{maximum value for capacity } w
$$

Recurrence:

$$
dp[w] = \max_{i: , weight[i] \le w} \big( dp[w - weight[i]] + value[i] \big)
$$

Base:

$$
dp[0] = 0
$$

Final answer:

$$
dp[W]
$$

The key difference from 0/1 Knapsack is order of iteration —
for unbounded, we move forward through weights so items can be reused.

#### How Does It Work (Plain Language)

Think of the capacity as a ladder. At each rung `w`, you check every item:

- If it fits, you ask: *"If I take this item, what's the best I can do with the remaining space?"*
- Since items are reusable, you can add it again later.

This way, every `dp[w]` builds from smaller capacities, each possibly using the same item again.

Example:
Items = (weight, value): (2,4), (3,7), (4,9)
`W = 7`

| Capacity | dp[w] | Explanation  |
| -------- | ----- | ------------ |
| 0        | 0     | base         |
| 1        | 0     | no item fits |
| 2        | 4     | one (2,4)    |
| 3        | 7     | one (3,7)    |
| 4        | 9     | one (4,9)    |
| 5        | 11    | (2,4)+(3,7)  |
| 6        | 14    | (3,7)+(3,7)  |
| 7        | 16    | (3,7)+(4,9)  |

Answer = 16

#### Tiny Code (Easy Versions)

C (Bottom-Up 1D DP)

```c
#include <stdio.h>
#define MAX(a,b) ((a) > (b) ? (a) : (b))

int main(void) {
    int n, W;
    printf("Enter number of items and capacity: ");
    scanf("%d %d", &n, &W);

    int wt[n], val[n];
    printf("Enter weights: ");
    for (int i = 0; i < n; i++) scanf("%d", &wt[i]);
    printf("Enter values: ");
    for (int i = 0; i < n; i++) scanf("%d", &val[i]);

    int dp[W + 1];
    for (int w = 0; w <= W; w++) dp[w] = 0;

    for (int i = 0; i < n; i++) {
        for (int w = wt[i]; w <= W; w++) {
            dp[w] = MAX(dp[w], val[i] + dp[w - wt[i]]);
        }
    }

    printf("Max value: %d\n", dp[W]);
    return 0;
}
```

Python (Iterative Version)

```python
weights = list(map(int, input("Enter weights: ").split()))
values = list(map(int, input("Enter values: ").split()))
W = int(input("Enter capacity: "))

dp = [0] * (W + 1)

for i in range(len(weights)):
    for w in range(weights[i], W + 1):
        dp[w] = max(dp[w], values[i] + dp[w - weights[i]])

print("Max value:", dp[W])
```

#### Why It Matters

- Demonstrates reusable subproblems, items can appear multiple times
- Connects to Coin Change (Min Coins) and Rod Cutting
- Foundation for integer partition, resource allocation, and bounded-unbounded hybrid problems
- Teaches forward iteration logic

Unbounded Knapsack is the perfect showcase of DP with repetition.

#### Step-by-Step Example

Items: (2,4), (3,7), (4,9), `W = 7`

| w | dp[w] | Best Choice |
| - | ----- | ----------- |
| 0 | 0     | base        |
| 1 | 0     | none fits   |
| 2 | 4     | (2)         |
| 3 | 7     | (3)         |
| 4 | 9     | (4)         |
| 5 | 11    | (2,3)       |
| 6 | 14    | (3,3)       |
| 7 | 16    | (3,4)       |

Answer = 16

#### A Gentle Proof (Why It Works)

For each capacity `w`, we consider every item `i` such that `weight[i] ≤ w`.
If we choose item `i`, we gain its value plus the best value achievable for the remaining space `w - weight[i]`:

$$
dp[w] = \max_i \big( value[i] + dp[w - weight[i]] \big)
$$

Unlike 0/1 Knapsack (which must avoid reuse), this recurrence allows reuse because `dp[w - weight[i]]` is computed before `dp[w]` in the same pass, meaning item `i` can contribute multiple times.

By filling the array from `0` to `W`, every capacity's best value is derived from optimal substructures of smaller capacities.

#### Try It Yourself

1. Print chosen items by tracking predecessors.
2. Compare with 0/1 Knapsack results.
3. Add constraint: each item ≤ k copies.
4. Apply to Rod Cutting: `weight = length`, `value = price`.
5. Experiment with fractional weights (greedy fails here).

#### Test Cases

| Weights   | Values        | W  | Expected | Notes                |
| --------- | ------------- | -- | -------- | -------------------- |
| [2,3,4]   | [4,7,9]       | 7  | 16       | (3,4)                |
| [5,10,15] | [10,30,50]    | 20 | 100      | four 5s or two 10s   |
| [1,2,3]   | [10,15,40]    | 6  | 90       | six 1s               |
| [2,5]     | [5,10]        | 3  | 5        | one 2                |
| [1,3,4,5] | [10,40,50,70] | 8  | 160      | multiples of 1 and 3 |

#### Complexity

- Time: $O(n \times W)$
- Space: $O(W)$

The Unbounded Knapsack captures the essence of reusable DP states, every step builds on smaller, self-similar subproblems, stacking value piece by piece until the capacity is full.

### 417 Fractional Knapsack

The Fractional Knapsack problem is a close cousin of the 0/1 Knapsack, but with a twist. Here, you can break items into fractions, taking partial weight to maximize total value. This problem is not solved by DP; it's a greedy algorithm, serving as a contrast to show where DP is *not* needed.

#### What Problem Are We Solving?

Given:

- `n` items, each with `weight[i]` and `value[i]`
- a knapsack with capacity `W`

Find the maximum total value achievable by possibly taking fractions of items.

We define:

- Value density (ratio):
  $$
  \text{ratio}[i] = \frac{value[i]}{weight[i]}
  $$

To maximize value:

1. Sort items by decreasing ratio.
2. Take full items until you can't.
3. Take a fraction of the next one to fill the remaining capacity.

Answer is the sum of selected (full + partial) values.

#### How Does It Work (Plain Language)

If each item can be split, the best approach is take the most valuable per unit weight first.
It's like filling your bag with gold dust, start with the richest dust, then move to less valuable kinds.

Example:
Items:

| Item | Value | Weight | Ratio |
| ---- | ----- | ------ | ----- |
| 1    | 60    | 10     | 6.0   |
| 2    | 100   | 20     | 5.0   |
| 3    | 120   | 30     | 4.0   |

Capacity = 50

1. Take all of Item 1 → weight 10, value 60
2. Take all of Item 2 → weight 20, value 100
3. Take 20/30 = 2/3 of Item 3 → weight 20, value 80

Total = 60 + 100 + 80 = 240

#### Tiny Code (Easy Versions)

C (Greedy Algorithm)

```c
#include <stdio.h>

struct Item {
    int value, weight;
};

int compare(const void *a, const void *b) {
    double r1 = (double)((struct Item *)a)->value / ((struct Item *)a)->weight;
    double r2 = (double)((struct Item *)b)->value / ((struct Item *)b)->weight;
    return (r1 < r2) ? 1 : -1;
}

int main(void) {
    int n, W;
    printf("Enter number of items and capacity: ");
    scanf("%d %d", &n, &W);

    struct Item arr[n];
    printf("Enter value and weight:\n");
    for (int i = 0; i < n; i++)
        scanf("%d %d", &arr[i].value, &arr[i].weight);

    qsort(arr, n, sizeof(struct Item), compare);

    double totalValue = 0.0;
    int curWeight = 0;

    for (int i = 0; i < n; i++) {
        if (curWeight + arr[i].weight <= W) {
            curWeight += arr[i].weight;
            totalValue += arr[i].value;
        } else {
            int remain = W - curWeight;
            totalValue += arr[i].value * ((double)remain / arr[i].weight);
            break;
        }
    }

    printf("Max value: %.2f\n", totalValue);
    return 0;
}
```

Python (Greedy Implementation)

```python
items = []
n = int(input("Enter number of items: "))
W = int(input("Enter capacity: "))

for _ in range(n):
    v, w = map(int, input("Enter value and weight: ").split())
    items.append((v, w, v / w))

items.sort(key=lambda x: x[2], reverse=True)

total_value = 0.0
cur_weight = 0

for v, w, r in items:
    if cur_weight + w <= W:
        cur_weight += w
        total_value += v
    else:
        remain = W - cur_weight
        total_value += v * (remain / w)
        break

print("Max value:", round(total_value, 2))
```

#### Why It Matters

- Demonstrates where DP isn't needed, a greedy choice property
- Contrasts with 0/1 Knapsack (DP needed)
- Builds intuition for ratio-based optimization
- Appears in resource allocation, finance, optimization

The Fractional Knapsack is the "continuous" version, you don't choose *yes or no*, you pour the best parts until you run out of room.

#### Step-by-Step Example

| Item | Value | Weight | Ratio | Take |
| ---- | ----- | ------ | ----- | ---- |
| 1    | 60    | 10     | 6.0   | Full |
| 2    | 100   | 20     | 5.0   | Full |
| 3    | 120   | 30     | 4.0   | 2/3  |

Total = 240

#### A Gentle Proof (Why It Works)

If all items can be divided arbitrarily, the optimal strategy is always to take the one with the highest value density first.
Proof sketch:

1. Suppose an optimal solution skips a higher-ratio item to take a lower-ratio one.
2. Replacing part of the lower-ratio item with the higher-ratio one strictly increases total value.
3. Contradiction, thus, sorting by ratio is optimal.

This property is called the greedy choice property.
Because the problem satisfies both optimal substructure and greedy choice, a greedy algorithm suffices.

#### Try It Yourself

1. Compare results with 0/1 Knapsack for same items.
2. Add more items with identical ratios.
3. Implement sorting manually and test correctness.
4. Check behavior when capacity < smallest weight.
5. Visualize partial fill using ratio chart.

#### Test Cases

| Values             | Weights         | W  | Expected | Notes               |
| ------------------ | --------------- | -- | -------- | ------------------- |
| [60,100,120]       | [10,20,30]      | 50 | 240      | classic             |
| [10,5,15,7,6,18,3] | [2,3,5,7,1,4,1] | 15 | 55.33    | greedy mix          |
| [25,50,75]         | [5,10,15]       | 10 | 50       | full item           |
| [5,10,15]          | [1,2,3]         | 3  | 15       | take all            |
| [1,2,3]            | [3,2,1]         | 3  | 5        | highest ratio first |

#### Complexity

- Time: $O(n \log n)$ (for sorting)
- Space: $O(1)$

The Fractional Knapsack shows the power of greedy reasoning, sometimes, thinking locally optimal truly leads to the global best.

### 418 Coin Change (Min Coins)

The Coin Change (Min Coins) problem is about finding the *fewest number of coins* needed to make a given amount. Unlike the counting version, which sums all combinations, this one focuses on minimization, the shortest path to the target sum.

It's a classic unbounded DP problem, where each coin can be used multiple times.

#### What Problem Are We Solving?

Given:

- A list of coins `coins[]`
- A target amount `A`

Find the minimum number of coins needed to make amount `A`.
If it's impossible, return `-1`.

We define the state:

$$
dp[x] = \text{minimum coins required to make amount } x
$$

Recurrence:

$$
dp[x] = \min_{c \in coins,; c \le x} \big( dp[x - c] + 1 \big)
$$

Base:

$$
dp[0] = 0
$$

Final answer:

$$
dp[A]
$$

If no combination is possible, `dp[A]` will remain at infinity (or a large sentinel value).

#### How Does It Work (Plain Language)

Think of building the amount step by step.
For each value `x`, try all coins `c` that fit, and see which leads to the fewest total coins.
Each state `dp[x]` represents the shortest chain from `0` to `x`.

It's like climbing stairs to a target floor, each coin is a step size, and you want the path with the fewest steps.

Example:
`coins = [1, 3, 4]`, `A = 6`

| Amount | dp[x] | Choice |
| ------ | ----- | ------ |
| 0      | 0     | base   |
| 1      | 1     | 1      |
| 2      | 2     | 1+1    |
| 3      | 1     | 3      |
| 4      | 1     | 4      |
| 5      | 2     | 3+2    |
| 6      | 2     | 3+3    |

Answer = 2 (3 + 3)

#### Tiny Code (Easy Versions)

C (Bottom-Up DP)

```c
#include <stdio.h>
#define INF 1000000
#define MIN(a,b) ((a) < (b) ? (a) : (b))

int main(void) {
    int n, A;
    printf("Enter number of coins and amount: ");
    scanf("%d %d", &n, &A);

    int coins[n];
    printf("Enter coin values: ");
    for (int i = 0; i < n; i++) scanf("%d", &coins[i]);

    int dp[A + 1];
    for (int i = 1; i <= A; i++) dp[i] = INF;
    dp[0] = 0;

    for (int x = 1; x <= A; x++) {
        for (int i = 0; i < n; i++) {
            int c = coins[i];
            if (x - c >= 0)
                dp[x] = MIN(dp[x], dp[x - c] + 1);
        }
    }

    if (dp[A] == INF) printf("Not possible\n");
    else printf("Min coins: %d\n", dp[A]);

    return 0;
}
```

Python (Simple Iterative Version)

```python
coins = list(map(int, input("Enter coin values: ").split()))
A = int(input("Enter amount: "))

INF = float('inf')
dp = [INF] * (A + 1)
dp[0] = 0

for x in range(1, A + 1):
    for c in coins:
        if x - c >= 0:
            dp[x] = min(dp[x], dp[x - c] + 1)

print("Min coins:" if dp[A] != INF else "Not possible", end=" ")
print(dp[A] if dp[A] != INF else "")
```

#### Why It Matters

- Classic unbounded minimization DP
- Core of many resource optimization problems
- Foundation for graph shortest paths, minimum steps, edit operations
- Contrasts with counting version (same recurrence, different aggregation)

This problem shows how min replaces sum in DP to shift from "how many" to "how few".

#### Step-by-Step Example

Coins = [1, 3, 4], A = 6

| x | Choices               | dp[x] |
| - | --------------------- | ----- |
| 0 | -                     | 0     |
| 1 | dp[0]+1               | 1     |
| 2 | dp[1]+1               | 2     |
| 3 | dp[0]+1               | 1     |
| 4 | dp[0]+1               | 1     |
| 5 | min(dp[2]+1, dp[1]+1) | 2     |
| 6 | min(dp[3]+1, dp[2]+1) | 2     |

Answer = 2 (3+3)

#### A Gentle Proof (Why It Works)

The recurrence builds from smaller amounts upward.
For each amount `x`, every coin `c` offers a path from `x - c` → `x`, adding 1 step.

By induction:

- Base case: `dp[0] = 0` (no coins to make 0).
- Inductive step: assume optimal solutions exist for all `< x`.
  Then, the minimal value among all `dp[x - c] + 1` is the fewest coins to form `x`.

Since each `x` reuses optimal subsolutions, `dp[A]` is globally optimal.

#### Try It Yourself

1. Print the chosen coins (trace back `dp[x]`).
2. Add a coin that never helps (e.g., `[1, 3, 10]`, `A=6`).
3. Compare with greedy for `[1,3,4]` (fails).
4. Extend to limited coins (bounded knapsack).
5. Try larger `A` to see performance.

#### Test Cases

| Coins     | Amount | Expected | Notes        |
| --------- | ------ | -------- | ------------ |
| [1,3,4]   | 6      | 2        | 3+3          |
| [2,5]     | 3      | -1       | not possible |
| [1,2,5]   | 11     | 3        | 5+5+1        |
| [9,6,5,1] | 11     | 2        | 6+5          |
| [2,3,7]   | 12     | 3        | 3+3+6        |

#### Complexity

- Time: $O(n \times A)$
- Space: $O(A)$

The Coin Change (Min Coins) problem is where unbounded DP meets optimization, building minimal paths to a target through simple, repeated decisions.

### 419 Coin Change (Count Ways)

The Coin Change (Count Ways) problem is about how many different ways you can make a given amount using available coins. Unlike the minimization version, here every combination matters, order doesn't.

This is a perfect example of unbounded combinatorial DP, where each coin can be used multiple times, but arrangement order is irrelevant.

#### What Problem Are We Solving?

Given:

- A list of coins `coins[]`
- A target amount `A`

Find the number of distinct combinations (unordered) that sum to `A`.

We define the state:

$$
dp[x] = \text{number of ways to make amount } x
$$

Recurrence:

$$
dp[x] = \sum_{c \in coins,; c \le x} dp[x - c]
$$

Base:

$$
dp[0] = 1
$$

To avoid counting the same combination multiple times (e.g., `[1,2]` and `[2,1]`), we iterate coins first, then amount.

#### How Does It Work (Plain Language)

We count *combinations*, not *permutations*.
That means `{1,2}` and `{2,1}` are considered the same way.
So we fix each coin's order, when processing a coin, we allow it to be reused, but not reordered with future coins.

Example:
`coins = [1, 2, 5]`, `A = 5`

Ways:

- 1+1+1+1+1
- 1+1+1+2
- 1+2+2
- 5

Answer = 4

#### Tiny Code (Easy Versions)

C (Bottom-Up DP)

```c
#include <stdio.h>

int main(void) {
    int n, A;
    printf("Enter number of coins and amount: ");
    scanf("%d %d", &n, &A);

    int coins[n];
    printf("Enter coin values: ");
    for (int i = 0; i < n; i++) scanf("%d", &coins[i]);

    long long dp[A + 1];
    for (int i = 0; i <= A; i++) dp[i] = 0;
    dp[0] = 1;

    for (int i = 0; i < n; i++) {
        for (int x = coins[i]; x <= A; x++) {
            dp[x] += dp[x - coins[i]];
        }
    }

    printf("Number of ways: %lld\n", dp[A]);
    return 0;
}
```

Python (Iterative Combinations)

```python
coins = list(map(int, input("Enter coin values: ").split()))
A = int(input("Enter amount: "))

dp = [0] * (A + 1)
dp[0] = 1

for c in coins:
    for x in range(c, A + 1):
        dp[x] += dp[x - c]

print("Number of ways:", dp[A])
```

#### Why It Matters

- Foundation of combinatorial DP
- Basis for partition counting, compositional sums, and probability DP
- Reinforces loop ordering importance, changing order counts permutations instead
- Connects to integer partition problems in number theory

It teaches you that *what you count* (order vs combination) depends on how you iterate.

#### Step-by-Step Example

Coins = [1, 2, 5], A = 5

Initialize `dp = [1, 0, 0, 0, 0, 0]`

| Coin | State               | dp array (after processing) |
| ---- | ------------------- | --------------------------- |
| 1    | all                 | [1, 1, 1, 1, 1, 1]          |
| 2    | adds combos using 2 | [1, 1, 2, 2, 3, 3]          |
| 5    | adds combos using 5 | [1, 1, 2, 2, 3, 4]          |

Answer = 4

#### A Gentle Proof (Why It Works)

We fill `dp[x]` by summing contributions from each coin `c`:
every time we use coin `c`, we move from subproblem `x - c` → `x`.

$$
dp[x] = \sum_{c \in coins} dp[x - c]
$$

But we must fix coin iteration order to ensure unique combinations.
Iterating coins first ensures each combination is formed in a canonical order:

- `1` before `2` before `5`
  So `{1,2}` appears once, not twice.

By induction:

- Base: `dp[0] = 1` (one way: use nothing)
- Step: each `dp[x]` counts valid combos by extending smaller sums.

#### Try It Yourself

1. Swap loop order → count permutations.
2. Add coin `3` and compare growth.
3. Print all combinations via recursion.
4. Add modulo $10^9 + 7$ for large results.
5. Compare with Min Coins (same coins, different goal).

#### Test Cases

| Coins     | Amount | Expected | Notes                            |
| --------- | ------ | -------- | -------------------------------- |
| [1,2,5]   | 5      | 4        | classic                          |
| [2,3,5,6] | 10     | 5        | multiple combos                  |
| [1]       | 3      | 1        | only one way                     |
| [2]       | 3      | 0        | impossible                       |
| [1,2,3]   | 4      | 4        | (1+1+1+1), (1+1+2), (2+2), (1+3) |

#### Complexity

- Time: $O(n \times A)$
- Space: $O(A)$

The Coin Change (Count Ways) problem captures the *combinatorial heart* of DP, a single recurrence, but with the magic of order-aware iteration, turns counting from chaos into clarity.

### 420 Multi-Dimensional Knapsack

The Multi-Dimensional Knapsack problem (also called the Multi-Constraint Knapsack) extends the classic 0/1 Knapsack into a richer, more realistic world. Here, each item consumes multiple types of resources (weight, volume, cost, etc.), and we must respect all constraints simultaneously.

It's where the simplicity of one dimension gives way to the complexity of many.

#### What Problem Are We Solving?

Given:

- `n` items
- Each item `i` has:

  * value `v[i]`
  * weights in `m` dimensions `w[i][1..m]`
- Capacities `C[1..m]` for each dimension

Select a subset of items maximizing total value, subject to:

$$
\forall j \in [1, m]: \sum_{i \in S} w[i][j] \le C[j]
$$

State definition:

$$
dp[c_1][c_2] \ldots [c_m] = \text{maximum value achievable with capacities } (c_1, \ldots, c_m)
$$

Recurrence:

$$
dp[\vec{c}] = \max \big( dp[\vec{c}],; dp[\vec{c} - \vec{w_i}] + v[i] \big)
$$

where $\vec{c} - \vec{w_i}$ means subtracting all weights component-wise.

#### How Does It Work (Plain Language)

It's like packing a spaceship with multiple limits, weight, volume, fuel usage, and every item drains each dimension differently.
You can't just fill to a single limit; every item's cost affects all dimensions at once.

The DP grid is now multi-dimensional: you must iterate over every combination of capacities and decide to include or exclude each item.

Example (2D case):
Items:

| Item | Value | Weight | Volume |
| ---- | ----- | ------ | ------ |
| 1    | 60    | 2      | 3      |
| 2    | 100   | 3      | 4      |
| 3    | 120   | 4      | 5      |

Capacity: (W=5, V=7)

Answer: 160 (Items 1 + 2)

#### Tiny Code (2D DP Example)

C (2D Capacity, 0/1 Version)

```c
#include <stdio.h>
#define MAX(a,b) ((a) > (b) ? (a) : (b))

int main(void) {
    int n, W, V;
    printf("Enter number of items, weight cap, volume cap: ");
    scanf("%d %d %d", &n, &W, &V);

    int w[n], vol[n], val[n];
    printf("Enter weight, volume, value:\n");
    for (int i = 0; i < n; i++)
        scanf("%d %d %d", &w[i], &vol[i], &val[i]);

    int dp[W + 1][V + 1];
    for (int i = 0; i <= W; i++)
        for (int j = 0; j <= V; j++)
            dp[i][j] = 0;

    for (int i = 0; i < n; i++) {
        for (int wi = W; wi >= w[i]; wi--) {
            for (int vi = V; vi >= vol[i]; vi--) {
                dp[wi][vi] = MAX(dp[wi][vi],
                                 dp[wi - w[i]][vi - vol[i]] + val[i]);
            }
        }
    }

    printf("Max value: %d\n", dp[W][V]);
    return 0;
}
```

Python (2D DP)

```python
items = [(2,3,60), (3,4,100), (4,5,120)]
W, V = 5, 7

dp = [[0]*(V+1) for _ in range(W+1)]

for w, v, val in items:
    for wi in range(W, w-1, -1):
        for vi in range(V, v-1, -1):
            dp[wi][vi] = max(dp[wi][vi], dp[wi-w][vi-v] + val)

print("Max value:", dp[W][V])
```

#### Why It Matters

- Models real-world constraints, multiple resources
- Core of operations research, resource allocation, logistics, multi-resource scheduling
- Illustrates how DP dimensionality grows with complexity
- Forces careful state design and iteration order

When one dimension is not enough, this generalization captures tradeoffs across many.

#### Step-by-Step Example (2D)

Capacity: W=5, V=7
Items:

1. (2,3,60)
2. (3,4,100)
3. (4,5,120)

We explore subsets:

- {1} → (2,3), value=60
- {2} → (3,4), value=100
- {3} → (4,5), value=120
- {1,2} → (5,7), value=160 ✅ optimal
- {1,3} → (6,8) ❌ exceeds
- {2,3} → (7,9) ❌ exceeds

Answer = 160

#### A Gentle Proof (Why It Works)

By induction on item index and capacities:

Let $dp[i][c_1][c_2] \ldots [c_m]$ be the best value using first `i` items and capacity vector $(c_1, c_2, \ldots, c_m)$.

Two choices for each item:

1. Exclude → keep $dp[i-1][\vec{c}]$
2. Include → $dp[i-1][\vec{c} - \vec{w_i}] + v[i]$ (if feasible)

Take the max.

Since all transitions only depend on smaller capacities, and each subproblem is optimal, overall DP converges to global optimum.

#### Try It Yourself

1. Add third dimension (e.g., "time").
2. Compare with greedy (fails).
3. Visualize DP table for 2D.
4. Track chosen items with traceback.
5. Add unbounded variation (re-use items).

#### Test Cases

| Items (W,V,Val)                | Capacity (W,V) | Expected | Notes       |
| ------------------------------ | -------------- | -------- | ----------- |
| [(2,3,60),(3,4,100),(4,5,120)] | (5,7)          | 160      | (1+2)       |
| [(1,2,10),(2,3,20),(3,3,40)]   | (3,4)          | 40       | single best |
| [(2,2,8),(2,3,9),(3,4,14)]     | (4,5)          | 17       | (1+2)       |
| [(3,2,10),(2,4,12),(4,3,18)]   | (5,6)          | 22       | (1+2)       |

#### Complexity

- Time: $O(n \times C_1 \times C_2 \times \ldots \times C_m)$
- Space: $O(C_1 \times C_2 \times \ldots \times C_m)$

The Multi-Dimensional Knapsack is a reminder that every extra resource adds a new axis to your reasoning, and your DP table.

# Section 43. Sequence Problems 


### 421 Longest Increasing Subsequence (O(n^2) DP)

The Longest Increasing Subsequence (LIS) problem asks for the maximum length of a subsequence that is strictly increasing. Elements do not need to be contiguous, only in order.

#### What Problem Are We Solving?

Given an array `arr[0..n-1]`, find the maximum `k` such that there exist indices
`0 ≤ i1 < i2 < ... < ik < n` with
$$
arr[i_1] < arr[i_2] < \cdots < arr[i_k].
$$

Define the state
$$
dp[i] = \text{length of the LIS that ends at index } i.
$$

Recurrence
$$
dp[i] = 1 + \max_{;0 \le j < i,; arr[j] < arr[i]} dp[j], \quad \text{with } dp[i] \leftarrow 1 \text{ if no such } j.
$$

Answer
$$
\max_{0 \le i < n} dp[i].
$$

#### How Does It Work (Plain Language)

For each position `i`, look back at all earlier positions `j < i` with a smaller value. Any increasing subsequence ending at `j` can be extended by `arr[i]`. Pick the best among them and add one. If nothing is smaller, start a new subsequence of length 1 at `i`.

Example: `arr = [10, 22, 9, 33, 21, 50, 41, 60]`

| i | arr[i] | dp[i] | explanation            |
| - | ------ | ----- | ---------------------- |
| 0 | 10     | 1     | start                  |
| 1 | 22     | 2     | 10 → 22                |
| 2 | 9      | 1     | restart at 9           |
| 3 | 33     | 3     | 10 → 22 → 33           |
| 4 | 21     | 2     | 10 → 21                |
| 5 | 50     | 4     | 10 → 22 → 33 → 50      |
| 6 | 41     | 4     | 10 → 22 → 33 → 41      |
| 7 | 60     | 5     | 10 → 22 → 33 → 50 → 60 |

Answer is 5.

#### Tiny Code (Easy Versions)

C (O(n^2))

```c
#include <stdio.h>
#define MAX(a,b) ((a)>(b)?(a):(b))

int main(void) {
    int n;
    printf("Enter n: ");
    scanf("%d", &n);
    int arr[n];
    printf("Enter array: ");
    for (int i = 0; i < n; i++) scanf("%d", &arr[i]);

    int dp[n];
    for (int i = 0; i < n; i++) dp[i] = 1;

    int ans = 0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < i; j++) {
            if (arr[j] < arr[i]) dp[i] = MAX(dp[i], dp[j] + 1);
        }
        ans = MAX(ans, dp[i]);
    }
    printf("LIS length: %d\n", ans);
    return 0;
}
```

Python (O(n^2))

```python
arr = list(map(int, input("Enter array: ").split()))
n = len(arr)
dp = [1] * n
for i in range(n):
    for j in range(i):
        if arr[j] < arr[i]:
            dp[i] = max(dp[i], dp[j] + 1)
print("LIS length:", max(dp) if dp else 0)
```

#### Why It Matters

- Archetypal sequence DP: define state on prefixes and extend with a choice.
- Foundation for LCS, Edit Distance, patience sorting LIS in O(n log n), and 2D variants like Russian Doll Envelopes.
- Useful in ranking, time series smoothing, and chain scheduling.

#### Step by Step Example

`arr = [3, 10, 2, 1, 20]`

| i | arr[i] | candidates (dp[j]+1) with (arr[j] < arr[i])       | dp[i] |
| - | ------ | ------------------------------------------------- | ----- |
| 0 | 3      |,                                                 | 1     |
| 1 | 10     | dp[0]+1 = 2                                       | 2     |
| 2 | 2      |,                                                 | 1     |
| 3 | 1      |,                                                 | 1     |
| 4 | 20     | max( dp[0]+1=2, dp[1]+1=3, dp[2]+1=2, dp[3]+1=2 ) | 3     |

Answer is 3.

#### A Gentle Proof (Why It Works)

Let `OPT(i)` denote the LIS length that ends exactly at index `i`. Any LIS ending at `i` must either be just `[arr[i]]` of length 1, or extend a strictly smaller element at some `j < i`. Therefore
$$
OPT(i) = \max\left( 1,; 1 + \max_{j<i,;arr[j]<arr[i]} OPT(j) \right).
$$
This depends only on optimal values from smaller indices, hence dynamic programming applies. The overall LIS is the maximum over all end positions:
$$
\text{LIS} = \max_i OPT(i).
$$
By induction on `i`, the recurrence computes `OPT(i)` correctly, so the final maximum is optimal.

#### Try It Yourself

1. Recover an actual LIS: keep a `parent[i]` pointing to the `j` that gave the best transition.
2. Change to nondecreasing subsequence: replace `<` with `<=`.
3. Compare with the O(n log n) patience sorting method and verify both lengths match.
4. Compute the number of LIS of maximum length using a parallel `cnt[i]`.
5. Extend to 2D pairs `(a,b)` by sorting on `a` and running LIS on `b` with careful tie handling.

#### Test Cases

| arr                      | expected |
| ------------------------ | -------- |
| [1,2,3,4,5]              | 5        |
| [5,4,3,2,1]              | 1        |
| [3,10,2,1,20]            | 3        |
| [10,22,9,33,21,50,41,60] | 5        |
| [2,2,2,2]                | 1        |

#### Complexity

- Time: (O(n^2))
- Space: (O(n))

This O(n^2) DP is the clearest path to LIS: define the end, look back to smaller ends, and grow the longest chain.

### 422 LIS (Patience Sorting) – O(n log n) Optimized

The Longest Increasing Subsequence (LIS) can be solved faster than the classic (O(n^2)) DP by using a clever idea inspired by patience sorting. Instead of building all sequences, we maintain a minimal tail array, each element represents the smallest possible tail of an increasing subsequence of a given length.

#### What Problem Are We Solving?

Given an array `arr[0..n-1]`, find the length of the longest strictly increasing subsequence in O(n log n) time.

We want
$$
\text{LIS length} = \max k \text{ such that } \exists i_1 < i_2 < \cdots < i_k,; arr[i_1] < arr[i_2] < \cdots < arr[i_k].
$$

#### Key Idea

Maintain an array `tails[]` where
`tails[len]` = smallest tail value of any increasing subsequence of length `len+1`.

For each element `x` in the array:

1. Use binary search in `tails` to find the first position `pos` with `tails[pos] ≥ x`.
2. Replace `tails[pos]` with `x` (we found a better tail).
3. If `x` is larger than all tails, append it, subsequence grows.

At the end, `len(tails)` = LIS length.

#### How Does It Work (Plain Language)

Think of placing numbers onto piles (like patience solitaire):

- Each pile's top is the smallest possible number ending an increasing subsequence of that length.
- When a new number comes, place it on the leftmost pile whose top is ≥ the number.
- If none exists, start a new pile.

The number of piles equals the LIS length.

Example:
`arr = [10, 22, 9, 33, 21, 50, 41, 60]`

Process:

| x  | tails (after processing x) |
| -- | -------------------------- |
| 10 | [10]                       |
| 22 | [10, 22]                   |
| 9  | [9, 22]                    |
| 33 | [9, 22, 33]                |
| 21 | [9, 21, 33]                |
| 50 | [9, 21, 33, 50]            |
| 41 | [9, 21, 33, 41]            |
| 60 | [9, 21, 33, 41, 60]        |

Answer = 5

#### Tiny Code (Easy Versions)

C (Using Binary Search)

```c
#include <stdio.h>

int lower_bound(int arr[], int len, int x) {
    int l = 0, r = len;
    while (l < r) {
        int mid = (l + r) / 2;
        if (arr[mid] < x) l = mid + 1;
        else r = mid;
    }
    return l;
}

int main(void) {
    int n;
    printf("Enter n: ");
    scanf("%d", &n);

    int a[n];
    printf("Enter array: ");
    for (int i = 0; i < n; i++) scanf("%d", &a[i]);

    int tails[n], len = 0;

    for (int i = 0; i < n; i++) {
        int pos = lower_bound(tails, len, a[i]);
        tails[pos] = a[i];
        if (pos == len) len++;
    }

    printf("LIS length: %d\n", len);
    return 0;
}
```

Python (Using bisect)

```python
import bisect

arr = list(map(int, input("Enter array: ").split()))
tails = []

for x in arr:
    pos = bisect.bisect_left(tails, x)
    if pos == len(tails):
        tails.append(x)
    else:
        tails[pos] = x

print("LIS length:", len(tails))
```

#### Why It Matters

- Reduces LIS from $O(n^2)$ to $O(n \log n)$
- Introduces binary search in DP transitions
- Demonstrates state compression, we track only *tails*, not all subproblems
- Serves as basis for LIS reconstruction, LDS, Longest Bitonic Subsequence, and 2D LIS

This technique shows how mathematical insight can collapse a DP table into a minimal structure.

#### Step-by-Step Example

`arr = [3, 10, 2, 1, 20]`

| x  | tails     |
| -- | --------- |
| 3  | [3]       |
| 10 | [3,10]    |
| 2  | [2,10]    |
| 1  | [1,10]    |
| 20 | [1,10,20] |

Answer = 3

#### A Gentle Proof (Why It Works)

Invariant:

- `tails[k]` = minimal possible tail of any increasing subsequence of length `k+1`.

When we place `x`:

- Replacing a tail keeps subsequences valid (shorter or equal tail → more chance to extend).
- Appending `x` grows the length by one.

By induction:

- Each `tails[k]` is nondecreasing with length.
- Final size of `tails` equals the LIS length, because every pile represents a distinct subsequence length.

#### Try It Yourself

1. Track predecessors to reconstruct one LIS.
2. Modify to nondecreasing subsequence with `bisect_right`.
3. Compare counts with (O(n^2)) version.
4. Visualize piles after each insertion.
5. Use on 2D sorted pairs `(a,b)` for envelope problems.

#### Test Cases

| arr                      | Expected | Notes              |
| ------------------------ | -------- | ------------------ |
| [10,22,9,33,21,50,41,60] | 5        | classic            |
| [3,10,2,1,20]            | 3        | {3,10,20}          |
| [1,2,3,4,5]              | 5        | already increasing |
| [5,4,3,2,1]              | 1        | decreasing         |
| [2,2,2,2]                | 1        | constant           |

#### Complexity

- Time: $O(n \log n)$ (binary search per element)
- Space: $O(n)$

The Patience Sorting LIS turns a quadratic DP into a sleek logarithmic method, a masterclass in trading space for insight.

### 423 Longest Common Subsequence (LCS)

The Longest Common Subsequence (LCS) problem finds the longest sequence that appears in the same relative order (not necessarily contiguous) in both strings. It is one of the most fundamental two-sequence DPs, and the basis of algorithms like diff, edit distance, and DNA alignment.

#### What Problem Are We Solving?

Given two sequences
$$
X = x_1, x_2, \dots, x_m,\quad Y = y_1, y_2, \dots, y_n
$$
find the length of the longest sequence that is a subsequence of both.

Define the state:

$$
dp[i][j] = \text{LCS length of prefixes } X[0..i-1],, Y[0..j-1]
$$

Recurrence:

$$
dp[i][j] =
\begin{cases}
0, & \text{if } i = 0 \text{ or } j = 0,\\
dp[i-1][j-1] + 1, & \text{if } x_{i-1} = y_{j-1},\\
\max\big(dp[i-1][j],\, dp[i][j-1]\big), & \text{if } x_{i-1} \ne y_{j-1}.
\end{cases}
$$


Answer:

$$
dp[m][n]
$$

#### How Does It Work (Plain Language)

We build a grid where each cell `dp[i][j]` represents the LCS of the first `i` characters of `X` and the first `j` characters of `Y`.

- If the characters match, extend the subsequence diagonally.
- If not, skip one character (either from `X` or `Y`) and take the better result.

Think of it as aligning the two strings, step by step, and keeping the longest matching order.

Example:
`X = "ABCBDAB"`, `Y = "BDCAB"`

The longest common subsequence is `"BCAB"`, length 4.

#### Tiny Code (Easy Versions)

C (Classic 2D DP)

```c
#include <stdio.h>
#define MAX(a,b) ((a) > (b) ? (a) : (b))

int main(void) {
    char X[100], Y[100];
    printf("Enter first string: ");
    scanf("%s", X);
    printf("Enter second string: ");
    scanf("%s", Y);

    int m = 0, n = 0;
    while (X[m]) m++;
    while (Y[n]) n++;

    int dp[m + 1][n + 1];
    for (int i = 0; i <= m; i++)
        for (int j = 0; j <= n; j++)
            dp[i][j] = 0;

    for (int i = 1; i <= m; i++) {
        for (int j = 1; j <= n; j++) {
            if (X[i - 1] == Y[j - 1])
                dp[i][j] = dp[i - 1][j - 1] + 1;
            else
                dp[i][j] = MAX(dp[i - 1][j], dp[i][j - 1]);
        }
    }

    printf("LCS length: %d\n", dp[m][n]);
    return 0;
}
```

Python (2D DP)

```python
X = input("Enter first string: ")
Y = input("Enter second string: ")

m, n = len(X), len(Y)
dp = [[0]*(n+1) for _ in range(m+1)]

for i in range(1, m+1):
    for j in range(1, n+1):
        if X[i-1] == Y[j-1]:
            dp[i][j] = dp[i-1][j-1] + 1
        else:
            dp[i][j] = max(dp[i-1][j], dp[i][j-1])

print("LCS length:", dp[m][n])
```

#### Why It Matters

- Classic two-dimensional DP template
- Core for Edit Distance, Sequence Alignment, Diff tools
- Demonstrates subproblem reuse via overlapping prefixes
- Helps understand table-filling and backtracking reconstruction

LCS is where dynamic programming meets string similarity.

#### Step-by-Step Example

`X = "ABCBDAB"`, `Y = "BDCAB"`

| i | j | X[i-1], Y[j-1] | dp[i][j] | Explanation      |
| - | - | -------------- | -------- | ---------------- |
| 1 | 1 | A, B           | 0        | mismatch         |
| 2 | 1 | B, B           | 1        | match            |
| 3 | 2 | C, D           | 1        | carry max        |
| 4 | 3 | B, C           | 1        | carry max        |
| 5 | 4 | D, A           | 2        | later match      |
| 7 | 5 | B, B           | 4        | full subsequence |

Answer = 4 (`"BCAB"`)

#### A Gentle Proof (Why It Works)

By induction on `i` and `j`:

- Base: $dp[0][j] = dp[i][0] = 0$ (empty prefix)
- If $x_{i-1} = y_{j-1}$, every common subsequence of `X[0..i-2]` and `Y[0..j-2]` can be extended by this match.
- If not equal, longest subsequence must exclude one character, hence `max` of left and top cells.

Since each subproblem depends only on smaller prefixes, filling the table row by row ensures all dependencies are ready.

#### Try It Yourself

1. Reconstruct the actual LCS (store direction or traceback).
2. Modify to handle case-insensitive matches.
3. Compare with Edit Distance formula.
4. Visualize table diagonal matches.
5. Use it to find diff between two lines of text.

#### Test Cases

| X         | Y         | Expected | Notes     |
| --------- | --------- | -------- | --------- |
| "ABCBDAB" | "BDCAB"   | 4        | "BCAB"    |
| "AGGTAB"  | "GXTXAYB" | 4        | "GTAB"    |
| "AAAA"    | "AA"      | 2        | subset    |
| "ABC"     | "DEF"     | 0        | none      |
| ""        | "ABC"     | 0        | base case |

#### Complexity

- Time: $O(m \times n)$
- Space: $O(m \times n)$, or $O(\min(m,n))$ with rolling arrays

The Longest Common Subsequence teaches you to align two worlds, character by character, building similarity from shared order, not proximity.

### 424 Edit Distance (Levenshtein)

The Edit Distance (or Levenshtein Distance) problem measures how *different* two strings are by counting the minimum number of operations needed to transform one into the other. The allowed operations are usually insert, delete, and replace.

This is one of the most elegant two-dimensional DPs, it captures transformation cost between sequences step by step.

#### What Problem Are We Solving?

Given two strings
$$
X = x_1, x_2, \dots, x_m,\quad Y = y_1, y_2, \dots, y_n
$$
find the minimum number of operations to convert `X` into `Y`, using:

- Insert a character
- Delete a character
- Replace a character

We define the state:

$$
dp[i][j] = \text{minimum edits to convert } X[0..i-1] \text{ into } Y[0..j-1]
$$

Recurrence:

$$
dp[i][j] =
\begin{cases}
i, & \text{if } j = 0,\\
j, & \text{if } i = 0,\\
dp[i-1][j-1], & \text{if } x_{i-1} = y_{j-1},\\
1 + \min\big(dp[i-1][j],\ dp[i][j-1],\ dp[i-1][j-1]\big), & \text{if } x_{i-1} \ne y_{j-1}.
\end{cases}
$$

- $dp[i-1][j] + 1$ → delete  
- $dp[i][j-1] + 1$ → insert  
- $dp[i-1][j-1] + 1$ → replace


Answer:

$$
dp[m][n]
$$

#### How Does It Work (Plain Language)

We build a 2D grid comparing prefixes of both strings.

Each cell answers: *"What's the cheapest way to make `X[:i]` look like `Y[:j]`?"*

- If characters match, carry over the diagonal value.
- If they differ, take the smallest cost among inserting, deleting, or replacing.

Think of typing corrections: every operation moves you closer to the target.

Example:
`X = "kitten"`, `Y = "sitting"`
Operations:

- Replace `k` → `s`
- Replace `e` → `i`
- Insert `g`

Answer = 3

#### Tiny Code (Easy Versions)

C (2D DP Table)

```c
#include <stdio.h>
#define MIN(a,b) ((a)<(b)?(a):(b))

int min3(int a, int b, int c) {
    int m = (a < b) ? a : b;
    return (m < c) ? m : c;
}

int main(void) {
    char X[100], Y[100];
    printf("Enter first string: ");
    scanf("%s", X);
    printf("Enter second string: ");
    scanf("%s", Y);

    int m = 0, n = 0;
    while (X[m]) m++;
    while (Y[n]) n++;

    int dp[m + 1][n + 1];

    for (int i = 0; i <= m; i++) dp[i][0] = i;
    for (int j = 0; j <= n; j++) dp[0][j] = j;

    for (int i = 1; i <= m; i++) {
        for (int j = 1; j <= n; j++) {
            if (X[i-1] == Y[j-1])
                dp[i][j] = dp[i-1][j-1];
            else
                dp[i][j] = 1 + min3(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]);
        }
    }

    printf("Edit distance: %d\n", dp[m][n]);
    return 0;
}
```

Python (Compact Version)

```python
X = input("Enter first string: ")
Y = input("Enter second string: ")

m, n = len(X), len(Y)
dp = [[0]*(n+1) for _ in range(m+1)]

for i in range(m+1):
    dp[i][0] = i
for j in range(n+1):
    dp[0][j] = j

for i in range(1, m+1):
    for j in range(1, n+1):
        if X[i-1] == Y[j-1]:
            dp[i][j] = dp[i-1][j-1]
        else:
            dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])

print("Edit distance:", dp[m][n])
```

#### Why It Matters

- Foundation for spell checking, DNA alignment, fuzzy matching, and diff tools
- Demonstrates multi-option recurrence (3 choices per state)
- Basis for weighted edit distances (cost per operation)
- Shows how to encode sequence transformation into DP

It's one of the cleanest examples where DP reveals the shortest transformation path.

#### Step-by-Step Example

`X = "kitten"`, `Y = "sitting"`

| i | j | X[:i]    | Y[:j]     | dp[i][j] | Explanation |
| - | - | -------- | --------- | -------- | ----------- |
| 0 | 0 | ""       | ""        | 0        | base        |
| 1 | 1 | "k"      | "s"       | 1        | replace     |
| 2 | 2 | "ki"     | "si"      | 1        | carry       |
| 3 | 3 | "kit"    | "sit"     | 1        | carry       |
| 4 | 4 | "kitt"   | "sitt"    | 1        | carry       |
| 5 | 5 | "kitte"  | "sitti"   | 2        | replace     |
| 6 | 6 | "kitten" | "sittin"  | 2        | replace     |
| 6 | 7 | "kitten" | "sitting" | 3        | insert      |

Answer = 3

#### A Gentle Proof (Why It Works)

For each prefix pair `(i, j)`:

- If last chars match: no new cost, inherit `dp[i-1][j-1]`.
- Else:

  * Delete `X[i-1]` → `dp[i-1][j] + 1`
  * Insert `Y[j-1]` → `dp[i][j-1] + 1`
  * Replace `X[i-1]` with `Y[j-1]` → `dp[i-1][j-1] + 1`

We choose the minimal option.
By induction on `(i, j)`, every `dp[i][j]` is optimal, since it uses optimal subsolutions from smaller prefixes.

#### Try It Yourself

1. Print the sequence of operations (traceback from `dp[m][n]`).
2. Change costs: make replace = 2, others = 1.
3. Compare with LCS: `EditDistance = m + n - 2 × LCS`.
4. Handle insert/delete only (turn it into LCS variant).
5. Try with words like `"intention"` → `"execution"`.

#### Test Cases

| X        | Y         | Expected | Notes                    |
| -------- | --------- | -------- | ------------------------ |
| "kitten" | "sitting" | 3        | replace, replace, insert |
| "flaw"   | "lawn"    | 2        | replace, insert          |
| "abc"    | "abc"     | 0        | same                     |
| "abc"    | "yabd"    | 2        | replace, insert          |
| ""       | "abc"     | 3        | inserts                  |

#### Complexity

- Time: $O(m \times n)$
- Space: $O(m \times n)$, or $O(\min(m,n))$ with rolling rows

The Edit Distance captures the very essence of transformation, how to reshape one structure into another, one careful operation at a time.

### 425 Longest Palindromic Subsequence

The Longest Palindromic Subsequence (LPS) problem finds the longest sequence that reads the same forward and backward, not necessarily contiguous.
It's a classic two-dimensional DP, and a mirror image of the Longest Common Subsequence (LCS), but here, we compare a string with its reverse.

#### What Problem Are We Solving?

Given a string
$$
S = s_1, s_2, \dots, s_n
$$
find the length of the longest subsequence that is a palindrome.

Define the state:

$$
dp[i][j] = \text{LPS length in substring } S[i..j]
$$

Recurrence:

$$
dp[i][j] =
\begin{cases}
1, & \text{if } i = j,\\
2 + dp[i+1][j-1], & \text{if } s_i = s_j,\\
\max\big(dp[i+1][j],\, dp[i][j-1]\big), & \text{if } s_i \ne s_j.
\end{cases}
$$

Base case:
$$
dp[i][i] = 1
$$

Final answer:
$$
dp[0][n-1]
$$


#### How Does It Work (Plain Language)

We expand outward between two indices `i` and `j`:

- If the characters match, they can wrap a smaller palindrome inside.
- If not, skip one character (either start or end) and try again.

Think of it as *folding the string onto itself*, one matching pair at a time.

Example:
`S = "bbbab"`

LPS = `"bbbb"` (length 4)

#### Tiny Code (Easy Versions)

C (Bottom-Up 2D DP)

```c
#include <stdio.h>
#define MAX(a,b) ((a) > (b) ? (a) : (b))

int main(void) {
    char S[100];
    printf("Enter string: ");
    scanf("%s", S);

    int n = 0;
    while (S[n]) n++;

    int dp[n][n];
    for (int i = 0; i < n; i++) dp[i][i] = 1;

    for (int len = 2; len <= n; len++) {
        for (int i = 0; i <= n - len; i++) {
            int j = i + len - 1;
            if (S[i] == S[j] && len == 2)
                dp[i][j] = 2;
            else if (S[i] == S[j])
                dp[i][j] = 2 + dp[i+1][j-1];
            else
                dp[i][j] = MAX(dp[i+1][j], dp[i][j-1]);
        }
    }

    printf("LPS length: %d\n", dp[0][n-1]);
    return 0;
}
```

Python (Clean DP Version)

```python
S = input("Enter string: ")
n = len(S)
dp = [[0]*n for _ in range(n)]

for i in range(n):
    dp[i][i] = 1

for length in range(2, n+1):
    for i in range(n - length + 1):
        j = i + length - 1
        if S[i] == S[j]:
            dp[i][j] = 2 + (dp[i+1][j-1] if length > 2 else 0)
        else:
            dp[i][j] = max(dp[i+1][j], dp[i][j-1])

print("LPS length:", dp[0][n-1])
```

#### Why It Matters

- Core DP on substrings (interval DP)
- Connects to LCS:
  [
  \text{LPS}(S) = \text{LCS}(S, \text{reverse}(S))
  ]
- Foundation for Palindrome Partitioning, String Reconstruction, and DNA symmetry problems
- Teaches two-pointer DP intuition

The LPS shows how symmetry and substructure intertwine.

#### Step-by-Step Example

`S = "bbbab"`

| i | j | S[i..j] | dp[i][j] | Reason           |
| - | - | ------- | -------- | ---------------- |
| 0 | 0 | b       | 1        | single char      |
| 1 | 1 | b       | 1        | single char      |
| 2 | 2 | b       | 1        | single char      |
| 3 | 3 | a       | 1        | single char      |
| 4 | 4 | b       | 1        | single char      |
| 2 | 4 | "bab"   | 3        | b + a + b        |
| 1 | 4 | "bbab"  | 3        | wrap b's         |
| 0 | 4 | "bbbab" | 4        | b + (bb a b) + b |

Answer = 4 (`"bbbb"`)

#### A Gentle Proof (Why It Works)

For substring `S[i..j]`:

- If `s_i == s_j`:
  both ends can contribute to a longer palindrome, add 2 around `dp[i+1][j-1]`.

- If `s_i != s_j`:
  one of them can't be in the palindrome, skip either `i` or `j` and take max.

By filling increasing substring lengths, every subproblem is solved before it's needed.

#### Try It Yourself

1. Reconstruct one longest palindrome using backtracking.
2. Compare with `LCS(S, reverse(S))` result.
3. Try on `"cbbd"`, `"agbdba"`.
4. Modify to count number of distinct palindromic subsequences.
5. Visualize table diagonals (bottom-up growth).

#### Test Cases

| S        | Expected | Notes           |
| -------- | -------- | --------------- |
| "bbbab"  | 4        | "bbbb"          |
| "cbbd"   | 2        | "bb"            |
| "agbdba" | 5        | "abdba"         |
| "abcd"   | 1        | any single char |
| "aaa"    | 3        | whole string    |

#### Complexity

- Time: (O(n^2))
- Space: (O(n^2)), reducible with rolling arrays

The Longest Palindromic Subsequence is a mirror held up to your string, revealing the symmetry hidden within.

### 426 Shortest Common Supersequence (SCS)

The Shortest Common Supersequence (SCS) problem asks for the shortest string that contains both given strings as subsequences.
It's like merging two sequences together without breaking order, balancing overlap and inclusion.

This problem is a close companion to LCS, in fact, its length can be directly expressed in terms of the Longest Common Subsequence.

#### What Problem Are We Solving?

Given two strings
$$
X = x_1, x_2, \dots, x_m,\quad Y = y_1, y_2, \dots, y_n
$$
find the length of the shortest string that contains both as subsequences.

Define the state:

$$
dp[i][j] = \text{length of SCS of prefixes } X[0..i-1] \text{ and } Y[0..j-1]
$$

Recurrence:

$$
dp[i][j] =
\begin{cases}
i, & \text{if } j = 0,\\
j, & \text{if } i = 0,\\
1 + dp[i-1][j-1], & \text{if } x_{i-1} = y_{j-1},\\
1 + \min\big(dp[i-1][j],\ dp[i][j-1]\big), & \text{if } x_{i-1} \ne y_{j-1}.
\end{cases}
$$

Answer:
$$
dp[m][n]
$$

Alternate formula:
$$
\text{SCS length} = m + n - \text{LCS length}
$$


#### How Does It Work (Plain Language)

If two characters match, you include it once and move diagonally.
If they differ, include one character and move toward the smaller subproblem (skipping one side).
You're building the shortest merged string preserving both orders.

Think of it as stitching the two sequences together with minimal redundancy.

Example:
`X = "AGGTAB"`, `Y = "GXTXAYB"`

LCS = `"GTAB"` (length 4)

So:
$$
\text{SCS length} = 6 + 7 - 4 = 9
$$

SCS = `"AGXGTXAYB"`

#### Tiny Code (Easy Versions)

C (DP Table)

```c
#include <stdio.h>
#define MIN(a,b) ((a) < (b) ? (a) : (b))

int main(void) {
    char X[100], Y[100];
    printf("Enter first string: ");
    scanf("%s", X);
    printf("Enter second string: ");
    scanf("%s", Y);

    int m = 0, n = 0;
    while (X[m]) m++;
    while (Y[n]) n++;

    int dp[m + 1][n + 1];
    for (int i = 0; i <= m; i++) dp[i][0] = i;
    for (int j = 0; j <= n; j++) dp[0][j] = j;

    for (int i = 1; i <= m; i++) {
        for (int j = 1; j <= n; j++) {
            if (X[i-1] == Y[j-1])
                dp[i][j] = 1 + dp[i-1][j-1];
            else
                dp[i][j] = 1 + MIN(dp[i-1][j], dp[i][j-1]);
        }
    }

    printf("SCS length: %d\n", dp[m][n]);
    return 0;
}
```

Python (Straightforward DP)

```python
X = input("Enter first string: ")
Y = input("Enter second string: ")

m, n = len(X), len(Y)
dp = [[0]*(n+1) for _ in range(m+1)]

for i in range(m+1):
    dp[i][0] = i
for j in range(n+1):
    dp[0][j] = j

for i in range(1, m+1):
    for j in range(1, n+1):
        if X[i-1] == Y[j-1]:
            dp[i][j] = 1 + dp[i-1][j-1]
        else:
            dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1])

print("SCS length:", dp[m][n])
```

#### Why It Matters

- Shows merging sequences with order preservation
- Ties directly to LCS via the formula
- Useful for file merging, version control, and sequence alignment
- Demonstrates minimal superstructure over two DPs

It's the "union" counterpart to the "intersection" of LCS.

#### Step-by-Step Example

`X = "AGGTAB"`, `Y = "GXTXAYB"`

| Step        | Action | Result   |
| ----------- | ------ | -------- |
| Compare A,G | differ | choose A |
| Compare G,G | match  | add G    |
| Compare G,X | differ | add X    |
| Compare G,T | differ | add T    |
| Compare T,X | differ | add X    |
| Compare A,A | match  | add A    |
| Compare B,B | match  | add B    |

SCS = `"AGXGTXAYB"`

Length = 9

#### A Gentle Proof (Why It Works)

Every SCS must include all characters of both strings in order.

- If last chars match: append once → `1 + dp[i-1][j-1]`
- Else, shortest option comes from skipping one character from either string.

By induction on `(i, j)`, since subproblems solve strictly smaller prefixes, we get optimal length.

The equivalence
$$
|SCS| = m + n - |LCS|
$$
follows because overlapping LCS chars are counted twice when summing lengths, and must be subtracted once.

#### Try It Yourself

1. Reconstruct actual SCS string (traceback from `dp[m][n]`).
2. Verify `|SCS| = |X| + |Y| - |LCS|`.
3. Compare SCS vs concatenation `X + Y`.
4. Apply to sequences with no overlap.
5. Test with identical strings (SCS = same string).

#### Test Cases

| X         | Y         | Expected Length | Notes          |
| --------- | --------- | --------------- | -------------- |
| "AGGTAB"  | "GXTXAYB" | 9               | overlap GTAB   |
| "ABCBDAB" | "BDCAB"   | 9               | shares BCAB    |
| "HELLO"   | "GEEK"    | 8               | no big overlap |
| "AB"      | "AB"      | 2               | identical      |
| "AB"      | "CD"      | 4               | disjoint       |

#### Complexity

- Time: $O(m \times n)$
- Space: $O(m \times n)$, or $O(\min(m,n))$ for length only

The Shortest Common Supersequence weaves two strings into one, the tightest possible thread that holds both stories together.

### 427 Longest Repeated Subsequence

The Longest Repeated Subsequence (LRS) of a string is the longest subsequence that appears at least twice in the string without reusing the same index position. It is like LCS of a string with itself, with an extra constraint to avoid matching a character with itself.

#### What Problem Are We Solving?

Given a string
$$
S = s_1, s_2, \dots, s_n
$$
find the length of the longest subsequence that occurs at least twice in (S) with disjoint index positions.

Define the state by comparing the string with itself:

$$
dp[i][j] = \text{LRS length for } S[1..i] \text{ and } S[1..j]
$$

Recurrence:

$$
dp[i][j] =
\begin{cases}
0, & \text{if } i = 0 \text{ or } j = 0,\\
1 + dp[i-1][j-1], & \text{if } s_i = s_j \text{ and } i \ne j,\\
\max\big(dp[i-1][j],\, dp[i][j-1]\big), & \text{otherwise.}
\end{cases}
$$

Answer:
$$
dp[n][n]
$$


The key difference from LCS is the constraint $i \ne j$ to prevent matching the same occurrence of a character.

#### How Does It Work (Plain Language)

Think of aligning the string with itself. You are looking for common subsequences, but you are not allowed to match a character to its identical position. When characters match at different positions, you extend the repeated subsequence. When they do not match or are at the same position, you take the best from skipping one side.

Example:
`S = "aabebcdd"`
One LRS is `"abd"` with length 3.

#### Tiny Code (Easy Versions)

C (2D DP)

```c
#include <stdio.h>
#define MAX(a,b) ((a) > (b) ? (a) : (b))

int main(void) {
    char S[1005];
    printf("Enter string: ");
    scanf("%s", S);

    int n = 0; while (S[n]) n++;

    int dp[n + 1][n + 1];
    for (int i = 0; i <= n; i++)
        for (int j = 0; j <= n; j++)
            dp[i][j] = 0;

    for (int i = 1; i <= n; i++) {
        for (int j = 1; j <= n; j++) {
            if (S[i-1] == S[j-1] && i != j)
                dp[i][j] = 1 + dp[i-1][j-1];
            else
                dp[i][j] = MAX(dp[i-1][j], dp[i][j-1]);
        }
    }

    printf("LRS length: %d\n", dp[n][n]);
    return 0;
}
```

Python (Straightforward DP)

```python
S = input("Enter string: ")
n = len(S)
dp = [[0]*(n+1) for _ in range(n+1)]

for i in range(1, n+1):
    for j in range(1, n+1):
        if S[i-1] == S[j-1] and i != j:
            dp[i][j] = 1 + dp[i-1][j-1]
        else:
            dp[i][j] = max(dp[i-1][j], dp[i][j-1])

print("LRS length:", dp[n][n])
```

#### Why It Matters

- Illustrates transforming a problem into LCS on the same string with a simple constraint.
- Useful for detecting repeated patterns and compression signals.
- Builds intuition for self-alignment DPs and index constraints.

#### Step by Step Example

`S = "aabebcdd"`
Let us look at the alignment idea:

- Matching pairs at different indices:

  * `a` at positions 1 and 2 can contribute, but not with the same index.
  * `b` at positions 3 and 6.
  * `d` at positions 7 and 8.

A valid repeated subsequence is `"abd"` using indices `(1,3,7)` and `(2,6,8)`.
Length (= 3).

#### A Gentle Proof (Why It Works)

Consider LCS of (S) with itself:

- If you allowed matches at the same indices, you would trivially match every character with itself and get (n).
- By forbidding matches where (i = j), any character contributes only when there exists another occurrence at a different index.
- The recurrence mirrors LCS but enforces (i \ne j).
- By induction on (i, j), the table accumulates exactly the lengths of repeated subsequences, and the maximum at (dp[n][n]) is the LRS length.

#### Try It Yourself

1. Reconstruct one LRS by tracing back from (dp[n][n]) while respecting (i \ne j).
2. Modify to count the number of distinct LRS of maximum length.
3. Compare LRS and LPS on the same string to see structural differences.
4. Handle ties when reconstructing to get the lexicographically smallest LRS.
5. Test behavior on strings with all unique characters.

#### Test Cases

| S          | Expected LRS length | One LRS |
| ---------- | ------------------- | ------- |
| "aabebcdd" | 3                   | "abd"   |
| "axxxy"    | 2                   | "xx"    |
| "aaaa"     | 3                   | "aaa"   |
| "abc"      | 0                   | ""      |
| "aaba"     | 2                   | "aa"    |

#### Complexity

- Time: (O(n^2))
- Space: (O(n^2))

The Longest Repeated Subsequence is LCS turned inward. Compare the string with itself, forbid identical positions, and the repeated pattern reveals itself.

### 428 String Interleaving

The String Interleaving problem asks whether a string $S$ can be formed by interleaving (or weaving together) two other strings $X$ and $Y$ while preserving the relative order of characters from each.

It's a dynamic programming problem that elegantly captures sequence merging under order constraints, similar in spirit to merging two sorted lists.

#### What Problem Are We Solving?

Given three strings $X$, $Y$, and $S$, determine if $S$ is a valid interleaving of $X$ and $Y$.

We define the state:

$$
dp[i][j] = \text{True if } S[0..i+j-1] \text{ can be formed by interleaving } X[0..i-1] \text{ and } Y[0..j-1]
$$

Recurrence:

$$
dp[i][j] =
\begin{cases}
dp[i-1][j], & \text{if } X[i-1] = S[i+j-1] \text{ and } dp[i-1][j],\\
dp[i][j-1], & \text{if } Y[j-1] = S[i+j-1] \text{ and } dp[i][j-1],\\
dp[i-1][j] \lor dp[i][j-1], & \text{if both conditions hold.}
\end{cases}
$$


Base conditions:

$$
dp[0][0] = \text{True}
$$

$$
dp[i][0] = dp[i-1][0] \land (X[i-1] = S[i-1])
$$

$$
dp[0][j] = dp[0][j-1] \land (Y[j-1] = S[j-1])
$$

Answer:

$$
dp[m][n]
$$

where $m = |X|$, $n = |Y|$.

#### How Does It Work (Plain Language)

You have two input strings $X$ and $Y$, and you're asked whether you can merge them in order to get $S$.

Each step, decide whether the next character in $S$ should come from $X$ or $Y$, as long as you don't break the order within either.

Imagine reading from two ribbons of characters, you can switch between them but never rearrange within a ribbon.

Example:
$X = \text{"abc"}$, $Y = \text{"def"}$, $S = \text{"adbcef"}$

Valid interleaving: $a$ (from $X$), $d$ (from $Y$), $b$ (from $X$), $c$ (from $X$), $e$ (from $Y$), $f$ (from $Y$)

#### Tiny Code (Easy Versions)

C (2D Boolean DP)

```c
#include <stdio.h>
#include <stdbool.h>
#include <string.h>

bool isInterleave(char *X, char *Y, char *S) {
    int m = strlen(X), n = strlen(Y);
    if (m + n != strlen(S)) return false;

    bool dp[m+1][n+1];
    dp[0][0] = true;

    for (int i = 1; i <= m; i++)
        dp[i][0] = dp[i-1][0] && X[i-1] == S[i-1];

    for (int j = 1; j <= n; j++)
        dp[0][j] = dp[0][j-1] && Y[j-1] == S[j-1];

    for (int i = 1; i <= m; i++) {
        for (int j = 1; j <= n; j++) {
            dp[i][j] = false;
            if (X[i-1] == S[i+j-1]) dp[i][j] |= dp[i-1][j];
            if (Y[j-1] == S[i+j-1]) dp[i][j] |= dp[i][j-1];
        }
    }

    return dp[m][n];
}

int main(void) {
    char X[100], Y[100], S[200];
    printf("Enter X: "); scanf("%s", X);
    printf("Enter Y: "); scanf("%s", Y);
    printf("Enter S: "); scanf("%s", S);

    if (isInterleave(X, Y, S))
        printf("Yes, S is an interleaving of X and Y.\n");
    else
        printf("No, S cannot be formed.\n");

    return 0;
}
```

Python (Simple DP Table)

```python
X = input("Enter X: ")
Y = input("Enter Y: ")
S = input("Enter S: ")

m, n = len(X), len(Y)
if len(S) != m + n:
    print("No")
else:
    dp = [[False]*(n+1) for _ in range(m+1)]
    dp[0][0] = True

    for i in range(1, m+1):
        dp[i][0] = dp[i-1][0] and X[i-1] == S[i-1]
    for j in range(1, n+1):
        dp[0][j] = dp[0][j-1] and Y[j-1] == S[j-1]

    for i in range(1, m+1):
        for j in range(1, n+1):
            dp[i][j] = (X[i-1] == S[i+j-1] and dp[i-1][j]) or \
                       (Y[j-1] == S[i+j-1] and dp[i][j-1])

    print("Yes" if dp[m][n] else "No")
```

#### Why It Matters

- Demonstrates two-sequence merging under order constraints
- Core idea behind path interleaving, merge scheduling, and string weaving
- Good stepping stone for problems involving 2D grid DPs and string constraints

#### Step-by-Step Example

$X = \text{"abc"}$, $Y = \text{"def"}$, $S = \text{"adbcef"}$

| i | j | X[:i] | Y[:j] | dp[i][j] | Explanation       |
| - | - | ----- | ----- | -------- | ----------------- |
| 0 | 0 | ""    | ""    | T        | base              |
| 1 | 0 | a     | ""    | T        | a from X          |
| 1 | 1 | a     | d     | T        | d from Y          |
| 2 | 1 | ab    | d     | T        | b from X          |
| 3 | 1 | abc   | d     | F        | cannot match next |
| 3 | 2 | abc   | de    | T        | e from Y          |
| 3 | 3 | abc   | def   | T        | f from Y          |

Answer: True

#### A Gentle Proof (Why It Works)

At any point, we have used $i + j$ characters from $S$:

- If last came from $X$: $X[i-1] = S[i+j-1]$ and $dp[i-1][j]$ was True
- If last came from $Y$: $Y[j-1] = S[i+j-1]$ and $dp[i][j-1]$ was True

By filling the table left-to-right, top-to-bottom, every prefix is validated before combining.
Inductive reasoning ensures correctness for all prefixes.

#### Try It Yourself

1. Print one valid interleaving path
2. Modify to count total interleavings
3. Handle strings with duplicate characters carefully
4. Try on examples where $X$ and $Y$ share common prefixes
5. Extend to three strings interleaving

#### Test Cases

| X     | Y     | S        | Expected |
| ----- | ----- | -------- | -------- |
| "abc" | "def" | "adbcef" | True     |
| "ab"  | "cd"  | "abcd"   | True     |
| "ab"  | "cd"  | "acbd"   | True     |
| "ab"  | "cd"  | "acdb"   | False    |
| "aa"  | "ab"  | "aaba"   | True     |

#### Complexity

- Time: $O(m \times n)$
- Space: $O(m \times n)$, can be reduced to $O(n)$

The String Interleaving problem is about harmony, weaving two sequences together, letter by letter, in perfect order.

### 429 Sequence Alignment (Bioinformatics)

The Sequence Alignment problem asks how to best align two sequences (often DNA, RNA, or proteins) to measure their similarity, allowing for gaps and mismatches. It forms the foundation of bioinformatics, string similarity, and edit-based scoring systems.

Unlike edit distance, sequence alignment assigns scores for matches, mismatches, and gaps, and seeks a maximum score, not a minimal edit count.

#### What Problem Are We Solving?

Given two sequences
$$
X = x_1, x_2, \dots, x_m, \quad Y = y_1, y_2, \dots, y_n
$$
and scoring rules:

- $+1$ for a match
- $-1$ for a mismatch
- $-2$ for a gap (insertion/deletion)

we want to find an alignment of $X$ and $Y$ that maximizes the total score.

We define the state:

$$
dp[i][j] = \text{maximum alignment score between } X[0..i-1] \text{ and } Y[0..j-1]
$$

Recurrence:

$$
dp[i][j] =
\max
\begin{cases}
dp[i-1][j-1] + \text{score}(x_{i-1}, y_{j-1}) \
dp[i-1][j] + \text{gap penalty} \
dp[i][j-1] + \text{gap penalty}
\end{cases}
$$

Base:

$$
dp[i][0] = i \times \text{gap penalty}, \quad dp[0][j] = j \times \text{gap penalty}
$$

Answer: $dp[m][n]$

#### How Does It Work (Plain Language)

We fill a grid where each cell $(i, j)$ represents the best score to align the first $i$ characters of $X$ with the first $j$ of $Y$.

At each step, we decide:

1. Match/Mismatch ($x_{i-1}$ with $y_{j-1}$)
2. Insert a gap in $Y$ (skip character in $X$)
3. Insert a gap in $X$ (skip character in $Y$)

The final cell holds the optimal alignment score.
Tracing back reveals the aligned strings, with dashes representing gaps.

Example:

$X = \text{"GATTACA"}$
$Y = \text{"GCATGCU"}$

One alignment:

```
G A T T A C A -
| |   |   | |
G - C A T G C U
```

#### Tiny Code (Easy Versions)

C (Global Alignment / Needleman–Wunsch)

```c
#include <stdio.h>
#define MAX(a,b) ((a) > (b) ? (a) : (b))

int score(char a, char b) {
    return a == b ? 1 : -1;
}

int main(void) {
    char X[100], Y[100];
    printf("Enter X: "); scanf("%s", X);
    printf("Enter Y: "); scanf("%s", Y);

    int m = 0, n = 0;
    while (X[m]) m++;
    while (Y[n]) n++;

    int gap = -2;
    int dp[m+1][n+1];

    for (int i = 0; i <= m; i++) dp[i][0] = i * gap;
    for (int j = 0; j <= n; j++) dp[0][j] = j * gap;

    for (int i = 1; i <= m; i++) {
        for (int j = 1; j <= n; j++) {
            int match = dp[i-1][j-1] + score(X[i-1], Y[j-1]);
            int delete = dp[i-1][j] + gap;
            int insert = dp[i][j-1] + gap;
            dp[i][j] = MAX(match, MAX(delete, insert));
        }
    }

    printf("Max alignment score: %d\n", dp[m][n]);
    return 0;
}
```

Python (Clean Version)

```python
X = input("Enter X: ")
Y = input("Enter Y: ")

m, n = len(X), len(Y)
match, mismatch, gap = 1, -1, -2

dp = [[0]*(n+1) for _ in range(m+1)]

for i in range(1, m+1):
    dp[i][0] = i * gap
for j in range(1, n+1):
    dp[0][j] = j * gap

for i in range(1, m+1):
    for j in range(1, n+1):
        score = match if X[i-1] == Y[j-1] else mismatch
        dp[i][j] = max(
            dp[i-1][j-1] + score,
            dp[i-1][j] + gap,
            dp[i][j-1] + gap
        )

print("Max alignment score:", dp[m][n])
```

#### Why It Matters

- Foundational in bioinformatics (DNA, RNA, protein comparison)
- Used in spell correction, plagiarism detection, text similarity
- Shows a weighted DP with scores, not just counts
- Demonstrates path reconstruction with multiple decisions

This is the generalization of edit distance to *scored alignment*.

#### Step-by-Step Example

Let $X = \text{"AGT"}$, $Y = \text{"GTT"}$
Match $= +1$, Mismatch $= -1$, Gap $= -2$

| i | j | $X[0..i]$ | $Y[0..j]$ | $dp[i][j]$ | Choice   |
| - | - | --------- | --------- | ---------- | -------- |
| 1 | 1 | A, G      | G         | -1         | mismatch |
| 2 | 2 | AG, GT    | GT        | +0         | align G  |
| 3 | 3 | AGT, GTT  | GTT       | +1         | align T  |

Answer = +1

#### A Gentle Proof (Why It Works)

Each $dp[i][j]$ represents the best possible score achievable aligning $X[0..i-1]$ and $Y[0..j-1]$.
Induction ensures correctness:

- Base: $dp[0][j], dp[i][0]$ handle leading gaps
- Step: At each $(i, j)$, you consider all valid transitions, match/mismatch, insert, delete, and take the max.
  Thus $dp[m][n]$ is globally optimal.

#### Try It Yourself

1. Trace back to print alignment with gaps
2. Try different scoring systems
3. Compare global (Needleman–Wunsch) vs local (Smith–Waterman) alignment
4. Handle affine gaps (gap opening + extension)
5. Visualize grid paths as alignments

#### Test Cases

| X         | Y         | Expected           | Notes        |
| --------- | --------- | ------------------ | ------------ |
| "AGT"     | "GTT"     | 1                  | one match    |
| "GATTACA" | "GCATGCU" | depends on scoring | classic      |
| "ABC"     | "ABC"     | 3                  | all match    |
| "ABC"     | "DEF"     | -3                 | all mismatch |
| "A"       | "AAA"     | -2                 | gaps added   |

#### Complexity

- Time: $O(m \times n)$
- Space: $O(m \times n)$ (can be reduced with row-rolling)

The Sequence Alignment problem teaches that similarity is not just about matches, it's about balancing alignments, mismatches, and gaps to find the best correspondence between two sequences.

### 430 Diff Algorithm (Myers / DP)

The Diff Algorithm compares two sequences and finds their shortest edit script (SES), the minimal sequence of insertions and deletions required to transform one into the other.
It's the heart of tools like `git diff` and `diff`, providing human-readable change summaries.

The Myers Algorithm is the most famous linear-space implementation, but the DP formulation builds on edit distance and LCS intuition.

#### What Problem Are We Solving?

Given two strings
$$
X = x_1, x_2, \dots, x_m, \quad Y = y_1, y_2, \dots, y_n
$$
find a minimal sequence of edits (insertions and deletions) to turn $X$ into $Y$.

Each edit transforms one sequence toward the other, and matching characters are left untouched.

The minimal number of edits equals:

$$
\text{SES length} = m + n - 2 \times \text{LCS length}
$$

We can also explicitly trace the path to recover the diff.

#### Recurrence (DP Formulation)

Let $dp[i][j]$ be the minimal number of edits to convert $X[0..i-1]$ into $Y[0..j-1]$.

Then:

$$
dp[i][j] =
\begin{cases}
i, & \text{if } j = 0,\\
j, & \text{if } i = 0,\\
dp[i-1][j-1], & \text{if } x_{i-1} = y_{j-1},\\
1 + \min\big(dp[i-1][j],\, dp[i][j-1]\big), & \text{if } x_{i-1} \ne y_{j-1}.
\end{cases}
$$

Answer:
$$
dp[m][n]
$$

The traceback reconstructs the sequence of operations:  
keeping, deleting, or inserting characters to transform $X$ into $Y$.


#### How Does It Work (Plain Language)

Imagine aligning two sequences line by line.
When characters match, move diagonally (no cost).
If they differ, you must either delete from $X$ or insert from $Y$.

By walking through a grid of all prefix pairs, you can find the shortest edit path, the same logic as `git diff`.

Example:
$X = \text{"ABCABBA"}$
$Y = \text{"CBABAC"}$

One minimal diff:

```
- A
  B
  C
+ B
  A
  B
- B
+ A
  C
```

#### Tiny Code (Easy Versions)

C (DP Traceback for Diff)

```c
#include <stdio.h>
#define MIN(a,b) ((a)<(b)?(a):(b))

int main(void) {
    char X[100], Y[100];
    printf("Enter X: "); scanf("%s", X);
    printf("Enter Y: "); scanf("%s", Y);

    int m = 0, n = 0;
    while (X[m]) m++;
    while (Y[n]) n++;

    int dp[m+1][n+1];
    for (int i = 0; i <= m; i++) dp[i][0] = i;
    for (int j = 0; j <= n; j++) dp[0][j] = j;

    for (int i = 1; i <= m; i++) {
        for (int j = 1; j <= n; j++) {
            if (X[i-1] == Y[j-1])
                dp[i][j] = dp[i-1][j-1];
            else
                dp[i][j] = 1 + MIN(dp[i-1][j], dp[i][j-1]);
        }
    }

    printf("Edit distance: %d\n", dp[m][n]);

    // Traceback
    int i = m, j = n;
    printf("Diff:\n");
    while (i > 0 || j > 0) {
        if (i > 0 && j > 0 && X[i-1] == Y[j-1]) {
            printf("  %c\n", X[i-1]);
            i--; j--;
        } else if (i > 0 && dp[i][j] == dp[i-1][j] + 1) {
            printf("- %c\n", X[i-1]);
            i--;
        } else {
            printf("+ %c\n", Y[j-1]);
            j--;
        }
    }

    return 0;
}
```

Python (Simple Diff Reconstruction)

```python
X = input("Enter X: ")
Y = input("Enter Y: ")

m, n = len(X), len(Y)
dp = [[0]*(n+1) for _ in range(m+1)]

for i in range(m+1):
    dp[i][0] = i
for j in range(n+1):
    dp[0][j] = j

for i in range(1, m+1):
    for j in range(1, n+1):
        if X[i-1] == Y[j-1]:
            dp[i][j] = dp[i-1][j-1]
        else:
            dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1])

i, j = m, n
ops = []
while i > 0 or j > 0:
    if i > 0 and j > 0 and X[i-1] == Y[j-1]:
        ops.append(f"  {X[i-1]}")
        i -= 1; j -= 1
    elif i > 0 and dp[i][j] == dp[i-1][j] + 1:
        ops.append(f"- {X[i-1]}")
        i -= 1
    else:
        ops.append(f"+ {Y[j-1]}")
        j -= 1

print("Edit distance:", dp[m][n])
print("Diff:")
for op in reversed(ops):
    print(op)
```

#### Why It Matters

- Foundation of version control systems (`git diff`, `patch`)
- Minimizes edit operations for transformation
- Simplifies merge conflict resolution
- Builds upon LCS and Edit Distance concepts
- Demonstrates traceback-based reconstruction

The diff is the human-readable face of dynamic programming, turning tables into insight.

#### Step-by-Step Example

$X = \text{"ABCABBA"}$, $Y = \text{"CBABAC"}$

| Operation | Explanation                       |
| --------- | --------------------------------- |
| - A       | X starts with 'A' not in Y prefix |
| B         | match                             |
| C         | match                             |
| + B       | insert                            |
| A         | match                             |
| B         | match                             |
| - B       | delete                            |
| + A       | insert                            |
| C         | match                             |

Total edits = 4 (minimal)

#### A Gentle Proof (Why It Works)

- If $x_{i-1} = y_{j-1}$, skip both (no cost).
- Otherwise, best path must add a new operation (insert/delete).
- $dp[i][j]$ stores minimal edits for prefixes.
  By induction, $dp[m][n]$ is the minimal SES length.

#### Try It Yourself

1. Print diff symbols (+, -, space) in alignment view
2. Compare diff path with LCS path
3. Try with real words: `"kitten"` → `"sitting"`
4. Modify cost: insertion/deletion weights
5. Use colors or indentation to visualize output

#### Test Cases

| X         | Y         | Edits | Notes           |
| --------- | --------- | ----- | --------------- |
| "ABCABBA" | "CBABAC"  | 4     | classic example |
| "abc"     | "abc"     | 0     | identical       |
| "abc"     | "def"     | 6     | all replaced    |
| "kitten"  | "sitting" | 3     | classic         |
| ""        | "xyz"     | 3     | all inserts     |

#### Complexity

- Time: $O(m \times n)$
- Space: $O(m \times n)$ (traceback needs full table)

The Diff Algorithm transforms comparison into storytelling, showing precisely how one sequence evolves into another, one edit at a time.

# Section 44. Matrix and Chain Problems 

### 431 Matrix Chain Multiplication

The Matrix Chain Multiplication problem asks for the most efficient way to parenthesize a product of matrices so that the total number of scalar multiplications is minimized. Matrix multiplication is associative, so the order of multiplication can change the cost dramatically even though the result is the same shape.

#### What Problem Are We Solving?

Given a chain of matrices
$$
A_1A_2\cdots A_n
$$
with dimensions
$$
A_i \text{ is } p_{i-1}\times p_i \quad (i=1..n),
$$
choose the parenthesization that minimizes scalar multiplications.

Define the state:

$$
dp[i][j] = \text{minimum cost to multiply } A_iA_{i+1}\cdots A_j
$$

Recurrence:

$$
dp[i][i] = 0
$$
$$
dp[i][j] = \min_{i\le k<j},\bigl(dp[i][k] + dp[k+1][j] + p_{i-1}p_kp_j\bigr)
$$

The last term is the cost of multiplying the two resulting matrices from the split at (k).

Answer:

$$
dp[1][n]
$$

Optionally keep a split table (split[i][j]) storing the (k) achieving the minimum to reconstruct the optimal parenthesization.

#### How Does It Work (Plain Language)

Matrix-chain multiplication DP

Recurrence
$$
\begin{aligned}
m[i,i] &= 0,\\
m[i,j] &= \min_{i \le k < j}\Big(m[i,k] + m[k+1,j] + p_{i-1}\,p_k\,p_j\Big)\qquad (i<j),
\end{aligned}
$$
where matrices are $A_i$ of size $p_{i-1}\times p_i$.

Example with dimensions
$$
p=[10,\,30,\,5,\,60],\quad
A_1:10\times30,\ A_2:30\times5,\ A_3:5\times60.
$$

Two ways to parenthesize:

1) $(A_1A_2)A_3$  
Cost
$$
(10\cdot 30\cdot 5) + (10\cdot 5\cdot 60)
= 1500 + 3000 = 4500.
$$

2) $A_1(A_2A_3)$  
Cost
$$
(30\cdot 5\cdot 60) + (10\cdot 30\cdot 60)
= 9000 + 18000 = 27000.
$$

Minimum cost is $4500$, achieved by $(A_1A_2)A_3$ with split $k=1$.


#### Tiny Code (Easy Versions)

C (DP with reconstruction)

```c
#include <stdio.h>
#include <limits.h>

void print_optimal(int i, int j, int split[105][105]) {
    if (i == j) { printf("A%d", i); return; }
    printf("(");
    int k = split[i][j];
    print_optimal(i, k, split);
    printf(" x ");
    print_optimal(k+1, j, split);
    printf(")");
}

int main(void) {
    int n;
    printf("Enter number of matrices: ");
    scanf("%d", &n);
    int p[n+1];
    printf("Enter dimensions p0..pn: ");
    for (int i = 0; i <= n; i++) scanf("%d", &p[i]);

    long long dp[105][105];
    int split[105][105];

    for (int i = 1; i <= n; i++) dp[i][i] = 0;

    for (int len = 2; len <= n; len++) {
        for (int i = 1; i + len - 1 <= n; i++) {
            int j = i + len - 1;
            dp[i][j] = LLONG_MAX;
            for (int k = i; k < j; k++) {
                long long cost = dp[i][k] + dp[k+1][j] + 1LL*p[i-1]*p[k]*p[j];
                if (cost < dp[i][j]) {
                    dp[i][j] = cost;
                    split[i][j] = k;
                }
            }
        }
    }

    printf("Min scalar multiplications: %lld\n", dp[1][n]);
    printf("Optimal parenthesization: ");
    print_optimal(1, n, split);
    printf("\n");
    return 0;
}
```

Python (DP with reconstruction)

```python
def matrix_chain_order(p):
    n = len(p) - 1
    dp = [[0]*(n+1) for _ in range(n+1)]
    split = [[0]*(n+1) for _ in range(n+1)]

    for length in range(2, n+1):
        for i in range(1, n - length + 2):
            j = i + length - 1
            dp[i][j] = float('inf')
            for k in range(i, j):
                cost = dp[i][k] + dp[k+1][j] + p[i-1]*p[k]*p[j]
                if cost < dp[i][j]:
                    dp[i][j] = cost
                    split[i][j] = k
    return dp, split

def build_solution(split, i, j):
    if i == j:
        return f"A{i}"
    k = split[i][j]
    return "(" + build_solution(split, i, k) + " x " + build_solution(split, k+1, j) + ")"

p = list(map(int, input("Enter p0..pn: ").split()))
dp, split = matrix_chain_order(p)
n = len(p) - 1
print("Min scalar multiplications:", dp[1][n])
print("Optimal parenthesization:", build_solution(split, 1, n))
```

#### Why It Matters

- Canonical example of interval DP and optimal binary partitioning
- Shows how associativity allows many evaluation orders with different costs
- Appears in query plan optimization, automatic differentiation scheduling, graphics and compiler optimization

#### Step by Step Example

For (p = [5, 10, 3, 12, 5, 50, 6]) with (n=6):

- Try all splits for each subchain length
- The DP eventually yields (dp[1][6] = 2010) and an optimal structure like (((A_1(A_2A_3))((A_4A_5)A_6)))
  Exact parentheses can vary among ties but the minimal cost is unique here.

#### A Gentle Proof (Why It Works)

Let (OPT(i,j)) be the optimal cost for $A_i\cdots A_j$. In any optimal solution, the last multiplication splits the chain at some (k) with $i\le k<j$. The two sides must themselves be optimal, otherwise replacing one side by a better solution improves the total, contradicting optimality. Therefore
$$
OPT(i,j) = \min_{i\le k<j}\bigl(OPT(i,k)+OPT(k+1,j)+p_{i-1}p_kp_j\bigr),
$$
with (OPT(i,i)=0). Since each subproblem uses strictly shorter chains, filling by increasing length computes all needed values before they are used.

#### Try It Yourself

1. Print not only one but all optimal parenthesizations when multiple (k) tie.
2. Add a second objective like minimizing depth after minimizing cost.
3. Compare greedy choices vs DP on random instances.
4. Extend to a cost model with addition cost or cache reuse.
5. Visualize the DP table and splits along diagonals.

#### Test Cases

| p (p0..pn)         | n | Expected min cost | One optimal parentheses                |
| ------------------ | - | ----------------- | -------------------------------------- |
| [10,30,5,60]       | 3 | 4500              | (A1 x A2) x A3                         |
| [5,10,3,12,5,50,6] | 6 | 2010              | one optimal structure                  |
| [40,20,30,10,30]   | 4 | 26000             | ((A1 x (A2 x A3)) x A4) or tie variant |
| [10,20,30]         | 2 | 6000              | A1 x A2                                |
| [2,3,4,5]          | 3 | 64                | (A1 x A2) x A3                         |

#### Complexity

- Time: (O(n^3)) for the triple loop over (i,j,k)
- Space: (O(n^2)) for the DP and split tables

Matrix Chain Multiplication is the textbook pattern for interval DP: pick a split, combine optimal subchains, and account for the boundary multiplication cost.

### 432 Boolean Parenthesization

The Boolean Parenthesization problem (also called the Boolean Expression Evaluation problem) asks: *Given a boolean expression consisting of `T` (true), `F` (false), and operators (`&`, `|`, `^`), how many ways can we parenthesize it so that it evaluates to `True`?*

It's a classic DP over intervals problem where we explore all possible splits between operators, combining sub-results based on logic rules.

#### What Problem Are We Solving?

Given a boolean expression string like
`T|F&T^T`,
count the number of ways to parenthesize it so that it evaluates to True.

We must consider both True and False counts for sub-expressions.

Let:

- $dpT[i][j]$ = number of ways $expr[i..j]$ evaluates to True
- $dpF[i][j]$ = number of ways $expr[i..j]$ evaluates to False

If expression length is $n$, then we only consider operands at even indices and operators at odd indices.

#### Recurrence

For every split at operator $k$ between $i$ and $j$:

Let $op = expr[k]$

Compute:

$$
\text{TotalTrue} =
\begin{cases}
dpT[i][k-1]\cdot dpT[k+1][j], & \text{if } op=\land,\\
dpT[i][k-1]\cdot dpT[k+1][j]\;+\;dpT[i][k-1]\cdot dpF[k+1][j]\;+\;dpF[i][k-1]\cdot dpT[k+1][j], & \text{if } op=\lor,\\
dpT[i][k-1]\cdot dpF[k+1][j]\;+\;dpF[i][k-1]\cdot dpT[k+1][j], & \text{if } op=\oplus.
\end{cases}
$$

Similarly for $dpF[i][j]$ using complementary logic.

Base cases:

$$
dpT[i][i] = 1 \text{ if expr[i] = 'T' else } 0
$$
$$
dpF[i][i] = 1 \text{ if expr[i] = 'F' else } 0
$$

Answer = $dpT[0][n-1]$

#### How Does It Work (Plain Language)

We cut the expression at each operator and combine the truth counts of the left and right sides according to boolean logic.

For each subexpression, we record:

- how many parenthesizations make it True
- how many make it False

Then we combine smaller subproblems to get bigger ones, just like Matrix Chain Multiplication, but using logic rules.

Example:
Expression = `T|F&T`

We can group as:

1. `(T|F)&T` → True
2. `T|(F&T)` → True

Answer = 2

#### Tiny Code (Easy Versions)

C (Bottom-Up DP)

```c
#include <stdio.h>
#include <string.h>

#define MAX 105

int dpT[MAX][MAX], dpF[MAX][MAX];

int main(void) {
    char expr[105];
    printf("Enter expression (T/F with &|^): ");
    scanf("%s", expr);

    int n = strlen(expr);
    for (int i = 0; i < n; i += 2) {
        dpT[i][i] = (expr[i] == 'T');
        dpF[i][i] = (expr[i] == 'F');
    }

    for (int len = 3; len <= n; len += 2) {
        for (int i = 0; i + len - 1 < n; i += 2) {
            int j = i + len - 1;
            dpT[i][j] = dpF[i][j] = 0;

            for (int k = i + 1; k < j; k += 2) {
                char op = expr[k];
                int lT = dpT[i][k-1], lF = dpF[i][k-1];
                int rT = dpT[k+1][j], rF = dpF[k+1][j];

                if (op == '&') {
                    dpT[i][j] += lT * rT;
                    dpF[i][j] += lT * rF + lF * rT + lF * rF;
                } else if (op == '|') {
                    dpT[i][j] += lT * rT + lT * rF + lF * rT;
                    dpF[i][j] += lF * rF;
                } else if (op == '^') {
                    dpT[i][j] += lT * rF + lF * rT;
                    dpF[i][j] += lT * rT + lF * rF;
                }
            }
        }
    }

    printf("Number of ways to get True: %d\n", dpT[0][n-1]);
    return 0;
}
```

Python (Readable DP)

```python
expr = input("Enter expression (T/F with &|^): ")
n = len(expr)

dpT = [[0]*n for _ in range(n)]
dpF = [[0]*n for _ in range(n)]

for i in range(0, n, 2):
    dpT[i][i] = 1 if expr[i] == 'T' else 0
    dpF[i][i] = 1 if expr[i] == 'F' else 0

for length in range(3, n+1, 2):
    for i in range(0, n-length+1, 2):
        j = i + length - 1
        for k in range(i+1, j, 2):
            op = expr[k]
            lT, lF = dpT[i][k-1], dpF[i][k-1]
            rT, rF = dpT[k+1][j], dpF[k+1][j]

            if op == '&':
                dpT[i][j] += lT * rT
                dpF[i][j] += lT * rF + lF * rT + lF * rF
            elif op == '|':
                dpT[i][j] += lT * rT + lT * rF + lF * rT
                dpF[i][j] += lF * rF
            else:  # '^'
                dpT[i][j] += lT * rF + lF * rT
                dpF[i][j] += lT * rT + lF * rF

print("Ways to evaluate to True:", dpT[0][n-1])
```

#### Why It Matters

- Classic interval DP pattern with logical combination
- Shows how state splitting applies beyond arithmetic
- Foundation for boolean circuit optimization and expression counting problems
- Reinforces divide by operator technique

#### Step-by-Step Example

Expression = `T|F&T`

Subproblems:

| Subexpr | Ways True | Ways False |   |
| ------- | --------- | ---------- | - |
| T       | 1         | 0          |   |
| F       | 0         | 1          |   |
| T       | 1         | 0          |   |
| T       | F         | 1          | 0 |
| F&T     | 0         | 1          |   |
| T       | F&T       | 2          | 0 |

Answer = 2

#### A Gentle Proof (Why It Works)

Each subexpression can be split at an operator $op_k$. The truth count of the whole depends only on the truth counts of its parts and the operator's truth table. By combining all possible $k$ recursively, we count all valid parenthesizations. Overlapping subproblems arise when evaluating the same substring, so memoization or bottom-up filling ensures efficiency.

#### Try It Yourself

1. Extend to count False outcomes too.
2. Add modulo $10^9+7$ for large counts.
3. Print one valid parenthesization.
4. Try on expressions like `T^T^F` or `T|F&T^T`.
5. Modify rules for custom logic systems.

#### Test Cases

| Expression | Expected True Count |   |   |
| ---------- | ------------------- | - | - |
| T          | F&T                 | 2 |   |
| T^T^F      | 0                   |   |   |
| T^F        | F                   | 2 |   |
| T&F        | T                   | 2 |   |
| T          | T&F                 | F | 5 |

#### Complexity

- Time: $O(n^3)$ (split for every operator)
- Space: $O(n^2)$ (2 DP tables)

The Boolean Parenthesization problem is the logic mirror of Matrix Chain Multiplication, instead of minimizing cost, we're counting truth through combinatorial structure.

### 433 Burst Balloons

The Burst Balloons problem is a classic interval DP challenge. You're given a row of balloons, each with a number representing coins. When you burst a balloon, you gain coins equal to the product of its number and the numbers of its immediate neighbors. The task is to determine the maximum coins you can collect by choosing the optimal order of bursting.

#### What Problem Are We Solving?

Given an array `nums` of length `n`, when you burst balloon `i`, you gain
$$
\text{coins} = nums[i-1] \times nums[i] \times nums[i+1]
$$
where `nums[i-1]` and `nums[i+1]` are the adjacent balloons still unburst.

After bursting `i`, it is removed from the sequence, changing neighbor relationships.

We want to maximize total coins by choosing the best bursting order.

To simplify boundary conditions, pad the array with 1s at both ends:
$$
val = [1] + nums + [1]
$$

Define DP state:
$$
dp[i][j] = \text{maximum coins obtainable by bursting all balloons between } i \text{ and } j
$$

Recurrence:
$$
dp[i][j] = \max_{k \in (i, j)} \Big( dp[i][k] + dp[k][j] + val[i] \cdot val[k] \cdot val[j] \Big)
$$

Answer:
$$
dp[0][n+1]
$$

#### How Does It Work (Plain Language)

Instead of thinking *"Which balloon to pop next?"*, think *"Which balloon to pop last?"* between two boundaries.

By fixing the last balloon `k` between `i` and `j`, its neighbors are guaranteed to be `i` and `j` at that moment, so the coins earned are easy to compute:
`val[i] * val[k] * val[j]`.

Then we solve the smaller subproblems:

- `dp[i][k]`: best coins from bursting balloons between `i` and `k`
- `dp[k][j]`: best coins from bursting between `k` and `j`

Combine and take the best split.

This is the reverse of the intuitive "first burst" approach, making the subproblems independent.

#### Tiny Code (Easy Versions)

C (Bottom-Up DP)

```c
#include <stdio.h>
#include <string.h>

#define MAX 305
#define max(a,b) ((a)>(b)?(a):(b))

int main(void) {
    int n;
    printf("Enter number of balloons: ");
    scanf("%d", &n);

    int nums[MAX], val[MAX];
    printf("Enter balloon values: ");
    for (int i = 1; i <= n; i++) scanf("%d", &nums[i]);

    val[0] = val[n+1] = 1;
    for (int i = 1; i <= n; i++) val[i] = nums[i];

    int dp[MAX][MAX];
    memset(dp, 0, sizeof(dp));

    for (int len = 2; len <= n + 1; len++) {
        for (int i = 0; i + len <= n + 1; i++) {
            int j = i + len;
            for (int k = i + 1; k < j; k++) {
                int cost = val[i] * val[k] * val[j] + dp[i][k] + dp[k][j];
                dp[i][j] = max(dp[i][j], cost);
            }
        }
    }

    printf("Max coins: %d\n", dp[0][n+1]);
    return 0;
}
```

Python (Interval DP)

```python
nums = list(map(int, input("Enter balloon values: ").split()))
val = [1] + nums + [1]
n = len(nums)
dp = [[0]*(n+2) for _ in range(n+2)]

for length in range(2, n+2):
    for i in range(0, n+2-length):
        j = i + length
        for k in range(i+1, j):
            dp[i][j] = max(dp[i][j], val[i]*val[k]*val[j] + dp[i][k] + dp[k][j])

print("Max coins:", dp[0][n+1])
```

#### Why It Matters

- Exemplifies interval DP structure: choose a pivot balloon as the "last" action
- Shows how reverse reasoning simplifies state independence
- Appears in optimization over chains, trees, brackets, and games
- Foundation for polygon triangulation and matrix multiplication variants

#### Step-by-Step Example

Example: `nums = [3, 1, 5, 8]`

Pad → `val = [1, 3, 1, 5, 8, 1]`

Compute `dp[i][j]` for increasing intervals:

- Interval (1,4): choose `k=2` or `3`, compare costs
- Gradually expand to full (0,5):
  Optimal = 167 coins

One optimal order:
`burst 1 → 5 → 3 → 8 → 1`

#### A Gentle Proof (Why It Works)

Each interval `(i,j)` can only depend on smaller intervals `(i,k)` and `(k,j)` because the last balloon `k` divides the chain cleanly. By fixing `k` as last, we ensure both sides are independent, they share no unburst balloons. Since every subproblem is smaller, bottom-up DP fills states without cycles. Thus, optimal substructure and overlapping subproblems guarantee correctness.

#### Try It Yourself

1. Implement a top-down memoized version with recursion.
2. Visualize the DP table as a triangle showing optimal splits.
3. Add reconstruction to print the burst order.
4. Try `[1,2,3]`, `[1,5]`, `[7,9,8]` to check intuition.
5. Compare performance for `n=300`.

#### Test Cases

| nums      | Max Coins |
| --------- | --------- |
| [3,1,5,8] | 167       |
| [1,5]     | 10        |
| [2,2,2]   | 12        |
| [1,2,3]   | 12        |
| [9]       | 9         |

#### Complexity

- Time: $O(n^3)$ for all subintervals and splits
- Space: $O(n^2)$ for DP table

The Burst Balloons problem captures the essence of interval DP: choose the last action, build subproblems on either side, and let structure guide optimal order.

### 434 Optimal BST

The Optimal Binary Search Tree problem asks for the BST shape that minimizes the expected search cost given access frequencies. Even though all BSTs hold the same keys in-order, different shapes can have very different average lookup depths.

#### What Problem Are We Solving?

We have sorted keys
$$K_1<K_2<\dots<K_n,$$
with successful-search probabilities (p_1,\dots,p_n). Optionally, we may include probabilities (q_0,\dots,q_n) for unsuccessful searches in the gaps between keys (classical formulation).

Goal: build a BST over these keys that minimizes expected comparisons.

Two standard DP models:

1. Full model with gaps ((p_i,q_i)). This is the textbook version.
2. Simplified model with only (p_i). Useful when you only care about successful hits.

#### DP: Full Model With Gaps

Define DP over intervals of keys $(K_i,\dots,K_j)$ with gaps on both sides.

Weight (total probability mass in the interval, including gaps):
$$
w[i][j]=
\begin{cases}
q_{i-1}, & i>j,\\
w[i][j-1] + p_j + q_j, & i\le j.
\end{cases}
$$

Expected cost (including internal node comparisons):
$$
e[i][j]=
\begin{cases}
q_{i-1}, & i>j,\\
\displaystyle \min_{r=i}^{j}\big( e[i][r-1] + e[r+1][j] + w[i][j] \big), & i\le j.
\end{cases}
$$

Answer: $e[1][n]$.  
If you also keep $root[i][j]$ that stores the minimizing $r$, you can reconstruct the tree.

#### DP: Simplified Model (success-only)

Sometimes you only have hit frequencies $f_i$. Let the cost count depth with each comparison adding 1. Define
$$
dp[i][j]
= \text{minimum total weighted depth for } K_i,\ldots,K_j
\text{ when the root contributes 1 per key below it.}
$$

A convenient formulation uses prefix sums
$$
S[k]=\sum_{t=1}^{k} f_t, \qquad W(i,j)=S[j]-S[i-1].
$$

Recurrence:
$$
dp[i][j]=
\begin{cases}
0, & i>j,\\[4pt]
\displaystyle \min_{r=i}^{j}\big(dp[i][r-1]+dp[r+1][j]\big)+W(i,j), & i\le j.
\end{cases}
$$

Answer: $dp[1][n]$.  
The extra $W(i,j)$ accounts for the fact that choosing any root increases the depth of all keys in its subtrees by 1.


#### How Does It Work (Plain Language)

Pick the root for a range of keys. Every key not chosen as root sits one level deeper, so its cost increases. The DP tries every candidate root and adds:

- the optimal cost of the left subtree
- the optimal cost of the right subtree
- the penalty for pushing all nonroot keys one level deeper (their total frequency)

Choose the root that minimizes this sum for every interval.

#### Tiny Code (Easy Versions)

C (full model with gaps (p_i,q_i))

```c
#include <stdio.h>
#include <float.h>

#define MAXN 205
#define MIN(a,b) ((a)<(b)?(a):(b))

double e[MAXN][MAXN], w[MAXN][MAXN];
int rootIdx[MAXN][MAXN];

int main(void) {
    int n;
    printf("Enter n: ");
    scanf("%d", &n);

    double p[MAXN], q[MAXN];
    printf("Enter p1..pn: ");
    for (int i = 1; i <= n; i++) scanf("%lf", &p[i]);
    printf("Enter q0..qn: ");
    for (int i = 0; i <= n; i++) scanf("%lf", &q[i]);

    for (int i = 1; i <= n+1; i++) {
        e[i][i-1] = q[i-1];
        w[i][i-1] = q[i-1];
    }

    for (int len = 1; len <= n; len++) {
        for (int i = 1; i+len-1 <= n; i++) {
            int j = i + len - 1;
            w[i][j] = w[i][j-1] + p[j] + q[j];
            e[i][j] = DBL_MAX;
            for (int r = i; r <= j; r++) {
                double cost = e[i][r-1] + e[r+1][j] + w[i][j];
                if (cost < e[i][j]) {
                    e[i][j] = cost;
                    rootIdx[i][j] = r;
                }
            }
        }
    }

    printf("Optimal expected cost: %.6f\n", e[1][n]);
    // rootIdx[i][j] holds the chosen root for reconstruction
    return 0;
}
```

Python (simplified model with hit frequencies (f_i))

```python
f = list(map(float, input("Enter f1..fn: ").split()))
n = len(f)
# 1-index for convenience
f = [0.0] + f
S = [0.0]*(n+1)
for i in range(1, n+1):
    S[i] = S[i-1] + f[i]

def W(i, j):
    return 0.0 if i > j else S[j] - S[i-1]

dp = [[0.0]*(n+2) for _ in range(n+2)]
root = [[0]*(n+2) for _ in range(n+2)]

for length in range(1, n+1):
    for i in range(1, n - length + 2):
        j = i + length - 1
        best, arg = float('inf'), -1
        for r in range(i, j+1):
            cost = dp[i][r-1] + dp[r+1][j] + W(i, j)
            if cost < best:
                best, arg = cost, r
        dp[i][j], root[i][j] = best, arg

print("Optimal weighted cost:", round(dp[1][n], 6))
# root[i][j] gives a root choice for reconstruction
```

#### Why It Matters

- Models biased queries where some keys are far more popular
- Canonical interval DP with a split and an additive per-interval penalty
- Basis for query plan optimization, autocompletion tries, and decision tree shaping
- Leads to advanced speedups like Knuth optimization when conditions hold

#### Step by Step Example

Simplified model with (f = [0.3, 0.2, 0.5]) for (K_1<K_2<K_3).

- For length 1: (dp[i][i] = f_i)
- For interval ([1,2]): try roots 1 or 2

  * (r=1:\ dp[1][0]+dp[2][2]+(f_1+f_2)=0+0.2+0.5=0.7)
  * (r=2:\ dp[1][1]+dp[3][2]+0.5=0.3+0+0.5=0.8)
    choose (r=1).
- For ([1,3]): try (r=1,2,3) with penalty (W(1,3)=1.0)
  compute and pick the minimum. The DP returns the best shape and cost.

#### A Gentle Proof (Why It Works)

Consider an optimal tree for keys ([i,j]) whose root is (r). All keys other than (K_r) move one level deeper, adding exactly (W(i,j)-p_r) to their cumulative cost. Splitting at (r) separates the instance into two independent subproblems ([i,r-1]) and ([r+1,j]). If either subtree were not optimal, replacing it by a better one would reduce total cost, contradicting optimality. Thus the recurrence that scans all roots and adds the interval weight is correct. The gap model adds (q)-probabilities to the interval weight (w[i][j]) and yields the classical formula.

#### Try It Yourself

1. Reconstruct the tree using the stored `root` table and print it in preorder.
2. Compare the full model ((p,q)) versus the simplified model on the same data.
3. Normalize frequencies so (\sum p_i + \sum q_i = 1) and interpret (e[1][n]) as expected comparisons.
4. Experiment with Knuth optimization when the quadrangle inequality holds to reduce time toward (O(n^2)).
5. Stress test with skewed distributions where one key dominates.

#### Test Cases

| Keys   | Model        | Params                                      | Expected behavior                                    |
| ------ | ------------ | ------------------------------------------- | ---------------------------------------------------- |
| 3 keys | simplified   | (f=[0.3,0.2,0.5])                           | root tends to be key 3 or 1 depending on split costs |
| 4 keys | simplified   | (f=[1,1,1,1])                               | more balanced tree wins                              |
| 3 keys | full ((p,q)) | (p=[0.3,0.2,0.4],\ q=[0.02,0.02,0.03,0.03]) | gaps shift the optimal root                          |
| 1 key  | either       | single freq                                 | cost equals (p_1) or (q_0+ p_1 + q_1)                |

#### Complexity

- Time: (O(n^3)) with the naive triple loop over ((i,j,r))
- Space: (O(n^2)) for DP and root tables

The Optimal BST DP captures a universal pattern: choose a split, add a per-interval penalty that reflects depth inflation, and combine optimal subtrees for the minimal expected search cost.

### 435 Polygon Triangulation

The Polygon Triangulation problem is a foundational geometric DP challenge. Given a convex polygon, the task is to divide it into non-overlapping triangles by drawing non-intersecting diagonals, minimizing the total weight—often the sum of triangle areas or edge lengths. It's structurally similar to Matrix Chain Multiplication, with intervals, splits, and additive costs.

#### What Problem Are We Solving?

Given a convex polygon with vertices (V_0, V_1, \dots, V_{n-1}), we want to triangulate it (partition into triangles using diagonals) such that the total cost is minimized.

Define cost as:
$$
\text{cost}(i,j,k) = \text{weight of triangle }(V_i, V_j, V_k)
$$
where weight could be:

- Area
- Perimeter
- Squared edge length sum (for generality)

We define the DP state:
$$
dp[i][j] = \text{minimum triangulation cost between } V_i \text{ and } V_j
$$

Recurrence:
$$
dp[i][j] = \min_{i<k<j}\Big(dp[i][k] + dp[k][j] + \text{cost}(i,j,k)\Big)
$$

Base case:
$$
dp[i][i+1] = 0
$$

Final answer: (dp[0][n-1])

#### How Does It Work (Plain Language)

You pick a vertex (k) between (i) and (j) to form a triangle ((V_i, V_k, V_j)).
This triangle splits the polygon into two smaller polygons:

- One from (V_i) to (V_k)
- Another from (V_k) to (V_j)

We recursively find their optimal triangulations and add the triangle's cost.

This is a divide-and-conquer on geometry.
Every choice of diagonal corresponds to a split in the DP.

#### Tiny Code (Easy Versions)

C (Using area as cost)

```c
#include <stdio.h>
#include <math.h>
#include <float.h>

#define MAX 105
#define min(a,b) ((a)<(b)?(a):(b))

typedef struct { double x, y; } Point;

double dist(Point a, Point b) {
    double dx = a.x - b.x, dy = a.y - b.y;
    return sqrt(dx*dx + dy*dy);
}

double triangle_cost(Point a, Point b, Point c) {
    double s = (dist(a,b) + dist(b,c) + dist(c,a)) / 2.0;
    double area = sqrt(s * (s - dist(a,b)) * (s - dist(b,c)) * (s - dist(c,a)));
    return area;
}

int main(void) {
    int n;
    printf("Enter number of vertices: ");
    scanf("%d", &n);
    Point v[MAX];
    for (int i = 0; i < n; i++)
        scanf("%lf %lf", &v[i].x, &v[i].y);

    double dp[MAX][MAX];
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            dp[i][j] = 0.0;

    for (int len = 2; len < n; len++) {
        for (int i = 0; i + len < n; i++) {
            int j = i + len;
            dp[i][j] = DBL_MAX;
            for (int k = i + 1; k < j; k++) {
                double cost = dp[i][k] + dp[k][j] + triangle_cost(v[i], v[j], v[k]);
                dp[i][j] = min(dp[i][j], cost);
            }
        }
    }

    printf("Min triangulation cost: %.4f\n", dp[0][n-1]);
    return 0;
}
```

Python (Perimeter cost)

```python
import math

def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

def triangle_cost(a, b, c):
    return dist(a,b) + dist(b,c) + dist(c,a)

n = int(input("Enter number of vertices: "))
v = [tuple(map(float, input().split())) for _ in range(n)]
dp = [[0]*n for _ in range(n)]

for length in range(2, n):
    for i in range(n - length):
        j = i + length
        dp[i][j] = float('inf')
        for k in range(i+1, j):
            dp[i][j] = min(dp[i][j],
                           dp[i][k] + dp[k][j] + triangle_cost(v[i], v[k], v[j]))

print("Min triangulation cost:", round(dp[0][n-1], 4))
```

#### Why It Matters

- Canonical geometric DP using intervals and triple splits
- Underpins graphics, meshing, computational geometry, and 3D modeling
- Shows that Matrix Chain Multiplication and Polygon Triangulation share a structural template
- Reinforces how spatial reasoning maps to recurrence formulation

#### Step-by-Step Example

Consider a quadrilateral with vertices:
$$
V_0=(0,0),; V_1=(1,0),; V_2=(1,1),; V_3=(0,1)
$$

Two triangulations:

1. Diagonal (V_0V_2): triangles ((V_0,V_1,V_2)) and ((V_0,V_2,V_3))
2. Diagonal (V_1V_3): triangles ((V_0,V_1,V_3)) and ((V_1,V_2,V_3))

Both give same area (=1).
DP would compute both and take the minimum.

#### A Gentle Proof (Why It Works)

Every triangulation must include exactly (n-3) diagonals.
Fixing a triangle ((V_i, V_k, V_j)) that uses diagonal (V_iV_j) partitions the polygon into two smaller convex polygons.
Since subproblems do not overlap except at the boundary, their optimal solutions combine to the global optimum.
By evaluating all (k) between (i) and (j), we guarantee we find the optimal split.
The recurrence enumerates all possible triangulations implicitly.

#### Try It Yourself

1. Change the cost function to perimeter instead of area.
2. Print the sequence of triangles chosen by storing split points.
3. Visualize the triangulation order in 2D.
4. Compare complexity vs brute force enumeration.
5. Implement in memoized recursion style.

#### Test Cases

| Vertices                               | Expected Min Cost (Area) |
| -------------------------------------- | ------------------------ |
| Square (0,0),(1,0),(1,1),(0,1)         | 1.0                      |
| Triangle (0,0),(1,0),(0,1)             | 0.5                      |
| Pentagon (0,0),(1,0),(2,1),(1,2),(0,1) | Varies with shape        |
| (0,0),(2,0),(2,2),(0,2)                | 4.0                      |

#### Complexity

- Time: $O(n^3)$ (triply nested loop)
- Space: $O(n^2)$ (DP table)

Polygon Triangulation is the geometric twin of matrix-chain optimization, same recurrence, new meaning.

### 436 Matrix Path Sum

The Matrix Path Sum problem asks for a path from the top-left to the bottom-right of a grid that optimizes a score, typically the minimum total sum of visited cells, moving only right or down.

#### What Problem Are We Solving?

Given an $m\times n$ matrix $A$ of integers, find the minimum cost to go from $(0,0)$ to $(m-1,n-1)$ using moves ${,\text{right},\text{down},}$.

State:
$$
dp[i][j]=\text{minimum path sum to reach cell }(i,j)
$$

Recurrence:
$$
dp[i][j]=A[i][j]+\min\big(dp[i-1][j],,dp[i][j-1]\big)
$$

Borders:
$$
dp[0][0]=A[0][0],\quad
dp[i][0]=A[i][0]+dp[i-1][0],\quad
dp[0][j]=A[0][j]+dp[0][j-1]
$$

Answer:
$$
dp[m-1][n-1]
$$

Space optimization: keep one row
$$
\text{row}[j]=A[i][j]+\min(\text{row}[j],,\text{row}[j-1])
$$

#### How Does It Work (Plain Language)

Each cell's best cost equals its value plus the better of the two ways you could have arrived: from above or from the left. Build the table row by row until the destination cell is filled.

Example:
$$
A=
\begin{bmatrix}
1&3&1\
1&5&1\
4&2&1
\end{bmatrix}
$$
Optimal sum is $1+1+3+1+1=7$ via path $(0,0)\to(1,0)\to(1,1)\to(1,2)\to(2,2)$.

#### Tiny Code (Easy Versions)

C (2D DP)

```c
#include <stdio.h>
#define MIN(a,b) ((a)<(b)?(a):(b))

int main(void) {
    int m, n;
    scanf("%d %d", &m, &n);
    int A[m][n], dp[m][n];
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            scanf("%d", &A[i][j]);

    dp[0][0] = A[0][0];
    for (int i = 1; i < m; i++) dp[i][0] = A[i][0] + dp[i-1][0];
    for (int j = 1; j < n; j++) dp[0][j] = A[0][j] + dp[0][j-1];

    for (int i = 1; i < m; i++)
        for (int j = 1; j < n; j++)
            dp[i][j] = A[i][j] + MIN(dp[i-1][j], dp[i][j-1]);

    printf("%d\n", dp[m-1][n-1]);
    return 0;
}
```

Python (1D Space Optimized)

```python
m, n = map(int, input().split())
A = [list(map(int, input().split())) for _ in range(m)]

row = [0]*n
row[0] = A[0][0]
for j in range(1, n):
    row[j] = row[j-1] + A[0][j]

for i in range(1, m):
    row[0] += A[i][0]
    for j in range(1, n):
        row[j] = A[i][j] + min(row[j], row[j-1])

print(row[-1])
```

#### Why It Matters

- Core template for grid DP and shortest path on DAGs
- Basis for image seam carving, robot motion on grids, spreadsheet cost flows
- Demonstrates classic DP patterns: table fill, border initialization, and space rolling

#### Step-by-Step Example

For
$$
A=
\begin{bmatrix}
5&1&3\
2&8&1\
4&2&1
\end{bmatrix}
$$
compute row by row:

- First row $dp$: $[5,6,9]$
- First column $dp$: $[5,7,11]$
- Fill inner:

  * $dp[1][1]=8+\min(6,7)=14$
  * $dp[1][2]=1+\min(9,14)=10$
  * $dp[2][1]=2+\min(14,11)=13$
  * $dp[2][2]=1+\min(10,13)=11$

Answer $=11$.

#### A Gentle Proof (Why It Works)

Any optimal path to $(i,j)$ must come from either $(i-1,j)$ or $(i,j-1)$ because moves are only right or down. If a cheaper route existed that did not pass through the cheaper of these two, replacing the prefix with the cheaper subpath would reduce the total cost, which is a contradiction. Hence the local recurrence using the minimum of top and left yields the global optimum when filled in topological order.

#### Try It Yourself

1. Reconstruct the path: keep a parent pointer from each $(i,j)$ to argmin of top or left.
2. Maximize sum instead of minimize by changing $\min$ to $\max$.
3. Add obstacles: mark blocked cells with $+\infty$ and skip them.
4. Allow moves right, down, and diagonal down-right. Extend the recurrence to three predecessors.
5. Use large matrices and compare 2D DP vs 1D rolling performance.

#### Test Cases

| Matrix                    | Expected |
| ------------------------- | -------- |
| [[1,3,1],[1,5,1],[4,2,1]] | 7        |
| [[5]]                     | 5        |
| [[1,2,3],[4,5,6]]         | 12       |
| [[1,1,1],[1,1,1],[1,1,1]] | 5        |
| [[5,1,3],[2,8,1],[4,2,1]] | 11       |

#### Complexity

- Time: $O(mn)$
- Space: $O(mn)$ with full table or $O(n)$ with row rolling

Matrix Path Sum is the go-to pattern for grid costs: initialize borders, sweep the table, and each cell is its value plus the best way to arrive.

### 437 Largest Square Submatrix

The Largest Square Submatrix problem asks for the size of the biggest square of 1s in a binary matrix. It's a staple 2D DP problem, each cell's value tells how large a square can end at that position.

#### What Problem Are We Solving?

Given a binary matrix $A$ of size $m\times n$, find the side length of the largest all-1s square.

We define
$$
dp[i][j] = \text{side length of the largest square whose bottom right corner is at } (i,j).
$$

Recurrence:
$$
dp[i][j]=
\begin{cases}
A[i][j], & i=0 \text{ or } j=0,\\
0, & A[i][j]=0,\\
\min\{dp[i-1][j],\, dp[i][j-1],\, dp[i-1][j-1]\}+1, & A[i][j]=1.
\end{cases}
$$

Answer:
$$
\max_{i,j} dp[i][j].
$$


#### How Does It Work (Plain Language)

Each cell says: *"How big a square of 1s can I end?"*
If it's a 1, we look up, left, and up-left, the smallest of those tells how big a square we can extend.

Intuition:

- A square can grow only if all its borders can.
- So a 1 at ((i,j)) grows a square of size `1 + min(three neighbors)`.

Example:

```
1 0 1 0 0
1 0 1 1 1
1 1 1 1 1
1 0 0 1 0
```

The largest square has side length 3.

#### Tiny Code (Easy Versions)

C

```c
#include <stdio.h>
#define MAX(a,b) ((a)>(b)?(a):(b))
#define MIN(a,b) ((a)<(b)?(a):(b))

int main(void) {
    int m, n;
    scanf("%d %d", &m, &n);
    int A[m][n], dp[m][n];
    for (int i=0; i<m; i++)
        for (int j=0; j<n; j++)
            scanf("%d", &A[i][j]);

    int max_side = 0;
    for (int i=0; i<m; i++) {
        for (int j=0; j<n; j++) {
            if (A[i][j] == 0) dp[i][j] = 0;
            else if (i==0 || j==0) dp[i][j] = 1;
            else dp[i][j] = MIN(MIN(dp[i-1][j], dp[i][j-1]), dp[i-1][j-1]) + 1;
            if (dp[i][j] > max_side) max_side = dp[i][j];
        }
    }
    printf("%d\n", max_side);
    return 0;
}
```

Python

```python
m, n = map(int, input().split())
A = [list(map(int, input().split())) for _ in range(m)]

dp = [[0]*n for _ in range(m)]
max_side = 0

for i in range(m):
    for j in range(n):
        if A[i][j] == 1:
            if i == 0 or j == 0:
                dp[i][j] = 1
            else:
                dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1
            max_side = max(max_side, dp[i][j])

print(max_side)
```

#### Why It Matters

- Foundation of 2D DP problems (local recurrence on neighbors).
- Directly extends to rectangles, histograms, obstacles, or weighted cells.
- Used in image processing, pattern detection, bioinformatics grids.
- Demonstrates how spatial structure can be captured by overlapping subproblems.

#### Step-by-Step Example

Matrix:
$$
A =
\begin{bmatrix}
1 & 1 & 1 \
1 & 1 & 1 \
0 & 1 & 1
\end{bmatrix}
$$

Fill DP:

| A | DP |
| - | -- |
| 1 | 1  |
| 1 | 1  |
| 1 | 1  |

Row 2:

- $dp[1][1] = \min(1,1,1) + 1 = 2$
- $dp[1][2] = \min(1,1,1) + 1 = 2$

Row 3:

- $dp[2][2] = \min(2,2,1) + 1 = 2$

$\text{Max} = 2 \Rightarrow \text{square side} = 2$


#### A Gentle Proof (Why It Works)

A cell ((i,j)) can be the bottom-right corner of a square of side (k) iff:

- The cells directly above, left, and diagonal up-left can each form a square of side (k-1).
  So (dp[i][j]) is the largest (k) satisfying these.
  Filling top-down ensures each needed neighbor is ready, and taking the min keeps the square aligned.

#### Try It Yourself

1. Modify to return area instead of side length.
2. Handle obstacles (cells with -1) as blocked.
3. Adapt to maximum rectangle of 1s (hint: histogram DP per row).
4. Output coordinates of top-left cell of the largest square.
5. Compare time between full and 1D-rolled DP.

#### Test Cases

| Matrix                    | Result |
| ------------------------- | ------ |
| [[1,1,1],[1,1,1],[1,1,1]] | 3      |
| [[0,1],[1,1]]             | 2      |
| [[1,0,1],[1,1,1],[1,1,0]] | 2      |
| [[1]]                     | 1      |
| [[0,0],[0,0]]             | 0      |

#### Complexity

- Time: (O(mn))
- Space: (O(mn)) or (O(n)) with rolling rows

This problem is a clean showcase of local DP propagation, each cell grows the memory of its best square from three neighbors.

### 438 Max Rectangle in Binary Matrix

The Max Rectangle in Binary Matrix problem asks for the area of the largest rectangle containing only 1s in a binary matrix. It's a powerful combination of 2D DP and stack-based histogram algorithms, every row is treated as the base of a histogram, and we compute the largest rectangle there.

#### What Problem Are We Solving?

Given a binary matrix $A$ of size $m \times n$, find the maximum rectangular area consisting entirely of 1s.

Interpret each row as a histogram of heights and update per row:
$$
\text{height}[j]=
\begin{cases}
\text{height}[j]+1, & A[i][j]=1,\\
0, & A[i][j]=0.
\end{cases}
$$
At each row, compute the Largest Rectangle in Histogram on $\text{height}$.

Equivalent 2D recurrence for heights:
$$
\text{height}[i][j]=
\begin{cases}
A[i][j]+\text{height}[i-1][j], & A[i][j]=1,\\
0, & A[i][j]=0.
\end{cases}
$$

Answer:
$$
\max_i \operatorname{LargestRectangle}\big(\text{height}[i]\big).
$$


#### How Does It Work (Plain Language)

Think of the matrix as stacked histograms:

- Each row builds on top of the one above.
- A `1` extends the height; a `0` resets it.
- For each row, we ask: "What's the largest rectangle if this row were the bottom?"

This converts a 2D problem into (m) histogram problems.

Example:

```
1 0 1 0 0
1 0 1 1 1
1 1 1 1 1
1 0 0 1 0
```

The largest rectangle of 1s has area 6.

#### Tiny Code (Easy Versions)

C (Using Stack for Histogram)

```c
#include <stdio.h>
#define MAX 205
#define MAX2(a,b) ((a)>(b)?(a):(b))

int largest_histogram(int *h, int n) {
    int stack[MAX], top = -1, maxA = 0;
    for (int i = 0; i <= n; i++) {
        int cur = (i == n ? 0 : h[i]);
        while (top >= 0 && h[stack[top]] >= cur) {
            int height = h[stack[top--]];
            int width = (top < 0 ? i : i - stack[top] - 1);
            int area = height * width;
            if (area > maxA) maxA = area;
        }
        stack[++top] = i;
    }
    return maxA;
}

int main(void) {
    int m, n;
    scanf("%d %d", &m, &n);
    int A[MAX][MAX];
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            scanf("%d", &A[i][j]);

    int height[MAX] = {0}, maxArea = 0;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++)
            height[j] = A[i][j] ? height[j] + 1 : 0;
        int area = largest_histogram(height, n);
        if (area > maxArea) maxArea = area;
    }
    printf("%d\n", maxArea);
    return 0;
}
```

Python (Stack-Based)

```python
def largest_histogram(h):
    stack, maxA = [], 0
    h.append(0)
    for i, x in enumerate(h):
        while stack and h[stack[-1]] >= x:
            height = h[stack.pop()]
            width = i if not stack else i - stack[-1] - 1
            maxA = max(maxA, height * width)
        stack.append(i)
    h.pop()
    return maxA

m, n = map(int, input().split())
A = [list(map(int, input().split())) for _ in range(m)]
height = [0]*n
maxA = 0

for i in range(m):
    for j in range(n):
        height[j] = height[j] + 1 if A[i][j] else 0
    maxA = max(maxA, largest_histogram(height))

print(maxA)
```

#### Why It Matters

- Core of 2D maximal area problems
- Connects DP (height propagation) and stack algorithms (histogram)
- Used in image segmentation, pattern recognition, binary masks
- Template for maximal submatrix under constraints

#### Step-by-Step Example

For:
$$
A=
\begin{bmatrix}
1 & 1 & 1 \
1 & 1 & 1 \
1 & 0 & 1
\end{bmatrix}
$$

Row 1: `[1,1,1]` → largest histogram = 3
Row 2: `[2,2,2]` → largest histogram = 6
Row 3: `[3,0,3]` → largest histogram = 3
Max = 6

#### A Gentle Proof (Why It Works)

Each rectangle in the matrix can be identified by its bottom row and column range.
The height array at row (i) encodes exactly the number of consecutive 1s above each column, including row (i).
Thus every maximal rectangle's bottom row is considered once, and the largest histogram algorithm ensures that for each height combination, the maximal area is found.
Therefore, iterating all rows yields the global optimum.

#### Try It Yourself

1. Modify to return coordinates of top-left and bottom-right corners.
2. Extend to max rectangle of 0s by flipping bits.
3. Compare to Largest Square Submatrix, same idea, different recurrence.
4. Use rolling arrays for memory reduction.
5. Visualize histogram growth row by row.

#### Test Cases

| Matrix                                            | Expected Max Area |
| ------------------------------------------------- | ----------------- |
| [[1,0,1,0,0],[1,0,1,1,1],[1,1,1,1,1],[1,0,0,1,0]] | 6                 |
| [[1,1,1],[1,1,1],[1,1,1]]                         | 9                 |
| [[0,0],[0,0]]                                     | 0                 |
| [[1]]                                             | 1                 |
| [[1,0,1],[1,1,1],[1,1,0]]                         | 4                 |

#### Complexity

- Time: (O(mn)) (each cell pushed/popped once across all rows)
- Space: (O(n)) for histogram and stack

This problem elegantly layers row-wise DP and histogram optimization, a universal method for maximal rectangles in 2D grids.

### 439 Submatrix Sum Queries

The Submatrix Sum Queries problem asks for the sum of all elements inside many rectangular regions of a 2D array. With a 2D prefix sum DP table, each query can be answered in $O(1)$ time after $O(mn)$ preprocessing.

#### What Problem Are We Solving?

Given an $m\times n$ matrix $A$ and many queries of the form $(r_1,c_1,r_2,c_2)$ with $0\le r_1\le r_2<m$ and $0\le c_1\le c_2<n$, compute:

$$
\text{Sum}(r_1,c_1,r_2,c_2)=\sum_{i=r_1}^{r_2}\sum_{j=c_1}^{c_2}A[i][j]
$$

Define the 2D prefix sum $P$ using 1-based indexing:

$$
P[i][j]=\sum_{x=1}^{i}\sum_{y=1}^{j}A[x-1][y-1], \quad 1\le i\le m,\ 1\le j\le n
$$

with $P[0][*]=P[*][0]=0$.

Then any submatrix sum is:

$$
S=P[r_2+1][c_2+1]-P[r_1][c_2+1]-P[r_2+1][c_1]+P[r_1][c_1]
$$

#### How Does It Work (Plain Language)

Precompute cumulative sums from the top-left corner.
The sum of a rectangle is then the big prefix up to its bottom-right, minus the two prefixes above and left, plus back the overlap you subtracted twice.
This is just inclusion-exclusion in 2D.

Example:

$$
A =
\begin{bmatrix}
1 & 2 & 3 \\
4 & 5 & 6 \\
7 & 8 & 9
\end{bmatrix}
$$

Query $(r_1,c_1,r_2,c_2)=(1,1,2,2)$ over
$$
\begin{bmatrix}
5 & 6 \\
8 & 9
\end{bmatrix}
$$
gives $5+6+8+9=28$.


#### Tiny Code (Easy Versions)

C (2D Prefix Sum, many queries in O(1) each)

```c
#include <stdio.h>

int main(void) {
    int m, n, q;
    scanf("%d %d", &m, &n);
    long long A[m][n];
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            scanf("%lld", &A[i][j]);

    long long P[m+1][n+1];
    for (int i = 0; i <= m; i++) P[i][0] = 0;
    for (int j = 0; j <= n; j++) P[0][j] = 0;

    for (int i = 1; i <= m; i++) {
        long long rowsum = 0;
        for (int j = 1; j <= n; j++) {
            rowsum += A[i-1][j-1];
            P[i][j] = P[i-1][j] + rowsum;
        }
    }

    scanf("%d", &q);
    while (q--) {
        int r1, c1, r2, c2;
        scanf("%d %d %d %d", &r1, &c1, &r2, &c2);
        long long S = P[r2+1][c2+1] - P[r1][c2+1] - P[r2+1][c1] + P[r1][c1];
        printf("%lld\n", S);
    }
    return 0;
}
```

Python (2D Prefix Sum)

```python
m, n = map(int, input().split())
A = [list(map(int, input().split())) for _ in range(m)]

P = [[0]*(n+1) for _ in range(m+1)]
for i in range(1, m+1):
    rowsum = 0
    for j in range(1, n+1):
        rowsum += A[i-1][j-1]
        P[i][j] = P[i-1][j] + rowsum

q = int(input())
for _ in range(q):
    r1, c1, r2, c2 = map(int, input().split())
    S = P[r2+1][c2+1] - P[r1][c2+1] - P[r2+1][c1] + P[r1][c1]
    print(S)
```

#### Why It Matters

- Turns many 2D range sum queries into $O(1)$ time each after $O(mn)$ preprocessing
- Fundamental for integral images, heatmaps, terrain elevation maps, and data analytics
- Building block for maximum submatrix sum, range average, and density queries

#### Step-by-Step Example

Let
$$
A =
\begin{bmatrix}
2 & -1 & 3 \\
0 & 4 & 5 \\
7 & 2 & -6
\end{bmatrix}
$$

Query $(0,1,2,2)$ covers the submatrix
$$
\begin{bmatrix}
-1 & 3 \\
4 & 5 \\
2 & -6
\end{bmatrix}
$$
whose sum is
$$
-1 + 3 + 4 + 5 + 2 - 6 = 7.
$$


Check formula:

$$
S = P[3][3]-P[0][3]-P[3][1]+P[0][1] = 12-4-7+6 = 7
$$

#### A Gentle Proof (Why It Works)

By definition,
$P[i][j]=\sum_{x\le i}\sum_{y\le j}A[x-1][y-1]$.

For rectangle $[r_1..r_2]\times[c_1..c_2]$:

- $P[r_2+1][c_2+1]$: total up to bottom-right
- subtract $P[r_1][c_2+1]$: remove rows above
- subtract $P[r_2+1][c_1]$: remove columns left
- add $P[r_1][c_1]$: restore overlap

Thus, the inclusion-exclusion identity holds.

#### Try It Yourself

1. Extend to 3D prefix sums for cuboid queries
2. Support range average (divide sum by area)
3. Add modulo arithmetic for large sums
4. Handle sparse updates with a 2D Fenwick tree
5. Precompute prefix sum for probability maps or heat distributions

#### Test Cases

| Matrix                      | Query $(r_1,c_1,r_2,c_2)$ | Expected |
| --------------------------- | ------------------------- | -------- |
| [[1,2],[3,4]]               | (0,0,1,1)                 | 10       |
| [[1,2,3],[4,5,6],[7,8,9]]   | (1,1,2,2)                 | 28       |
| [[2,-1,3],[0,4,5],[7,2,-6]] | (0,1,2,2)                 | 7        |
| [[5]]                       | (0,0,0,0)                 | 5        |
| [[1,0,1],[0,1,0],[1,0,1]]   | (0,0,2,2)                 | 5        |

#### Complexity

- Preprocessing: $O(mn)$
- Query: $O(1)$
- Space: $O(mn)$

2D prefix sums are a foundational DP tool: preprocess once, then every submatrix sum is instant.

### 440 Palindrome Partitioning

The Palindrome Partitioning problem asks you to divide a string into the fewest possible substrings such that each substring is a palindrome. It's a quintessential interval DP problem where we explore all split points, using precomputed palindrome checks to accelerate the recurrence.

#### What Problem Are We Solving?

Given a string $s$ of length $n$, find the minimum number of cuts needed so that every substring is a palindrome.

For example:
$s=\text{"aab"}$
The best partition is `"aa" | "b"`, needing 1 cut.

We define:

- $dp[i]$ = minimum cuts needed for substring $s[0..i]$
- $pal[i][j] = 1$ if $s[i..j]$ is palindrome, else $0$

Recurrence:
$$
dp[i] =
\begin{cases}
0, & \text{if } pal[0][i] = 1,\\
\min_{0 \le j < i,\ pal[j+1][i] = 1} (dp[j] + 1), & \text{otherwise.}
\end{cases}
$$

Precompute $pal[i][j]$ using:
$$
pal[i][j] = (s[i]=s[j]) \land (j-i<2 \lor pal[i+1][j-1])
$$

Answer: $dp[n-1]$

#### How Does It Work (Plain Language)

We want to cut the string at points where the right substring is a palindrome.
For each index $i$, we find all $j<i$ such that $s[j+1..i]$ is palindrome and take the minimum over $dp[j]+1$.

To avoid $O(n^3)$, we first precompute $pal[i][j]$ in $O(n^2)$.

Example:

```
s = "aab"
pal = 
a a b
1 1 0
  1 0
    1
```

Cuts:

- $dp[0]=0$ (`a`)
- $dp[1]=0$ (`aa`)
- $dp[2]=1$ (`aa|b`)
  → answer = 1

#### Tiny Code (Easy Versions)

C (Bottom-Up DP)

```c
#include <stdio.h>
#include <string.h>
#include <limits.h>

#define MIN(a,b) ((a)<(b)?(a):(b))
#define MAXN 1005

int main(void) {
    char s[MAXN];
    scanf("%s", s);
    int n = strlen(s);
    int pal[MAXN][MAXN] = {0};
    int dp[MAXN];

    for (int i = 0; i < n; i++) pal[i][i] = 1;
    for (int len = 2; len <= n; len++) {
        for (int i = 0; i + len - 1 < n; i++) {
            int j = i + len - 1;
            if (s[i] == s[j]) {
                if (len == 2 || pal[i+1][j-1]) pal[i][j] = 1;
            }
        }
    }

    for (int i = 0; i < n; i++) {
        if (pal[0][i]) { dp[i] = 0; continue; }
        dp[i] = INT_MAX;
        for (int j = 0; j < i; j++) {
            if (pal[j+1][i] && dp[j] + 1 < dp[i])
                dp[i] = dp[j] + 1;
        }
    }

    printf("%d\n", dp[n-1]);
    return 0;
}
```

Python (Readable)

```python
s = input().strip()
n = len(s)
pal = [[False]*n for _ in range(n)]
dp = [0]*n

for i in range(n):
    pal[i][i] = True
for length in range(2, n+1):
    for i in range(n-length+1):
        j = i + length - 1
        if s[i] == s[j] and (length == 2 or pal[i+1][j-1]):
            pal[i][j] = True

for i in range(n):
    if pal[0][i]:
        dp[i] = 0
    else:
        dp[i] = min(dp[j]+1 for j in range(i) if pal[j+1][i])

print(dp[-1])
```

#### Why It Matters

- Teaches interval DP with string-based state
- Used in text segmentation, DNA sequence analysis, code parsing
- Builds intuition for partitioning problems and precomputation synergy

#### Step-by-Step Example

$s = \text{"banana"}$

Palindromic substrings:

- Single letters
- `"ana"` at positions (1–3), (3–5)
- `"anana"` at (1–5)

Compute $dp$:

- $dp[0]=0$
- $dp[1]=1$ (`b|a`)
- $dp[2]=1$ (`ba|n`)
- $dp[3]=1$ (`b|ana`)
- $dp[4]=2$ (`b|an|an`)
- $dp[5]=1$ (`b|anana`)
  → Answer = 1

#### A Gentle Proof (Why It Works)

A valid partition ends at some position $i$.
If $s[j+1..i]$ is palindrome, then the cost to partition up to $i$ is $dp[j]+1$.
The optimal must choose the best such $j$.
By precomputing all palindrome substrings, each $dp[i]$ depends only on smaller indices, satisfying the principle of optimality.

#### Try It Yourself

1. Return the actual partitions using a parent array.
2. Modify to count all partitions instead of minimizing.
3. Adapt to palindromic subsequences (different structure).
4. Visualize DP and palindrome tables side by side.
5. Benchmark naive vs precomputed palindrome approaches.

#### Test Cases

| s         | Expected Cuts | Example Partition |         |     |
| --------- | ------------- | ----------------- | ------- | --- |
| "aab"     | 1             | "aa"              | "b"     |     |
| "racecar" | 0             | "racecar"         |         |     |
| "banana"  | 1             | "b"               | "anana" |     |
| "abc"     | 2             | "a"               | "b"     | "c" |
| "a"       | 0             | "a"               |         |     |

#### Complexity

- Time: $O(n^2)$
- Space: $O(n^2)$ (can reduce palindrome table)

Palindrome Partitioning is a model example of DP with precomputation, revealing how structure (palindromes) enables efficient segmentation.

# Section 45. Bitmask DP and Traveling Salesman 

### 441 Traveling Salesman Problem (TSP), Bitmask DP (Held–Karp)

The Traveling Salesman Problem asks for the shortest tour that visits every city exactly once and returns to the start. With dynamic programming over subsets, we can solve it in $O(n^2 2^n)$, which is optimal up to polynomial factors for exact exponential algorithms.

#### What Problem Are We Solving?

Given $n$ cities and a distance matrix $dist[i][j]$, find the minimum tour length that starts at city $0$, visits all cities once, and returns to $0$.

State (Held–Karp):

- Let $dp[mask][i]$ be the minimum cost to start at $0$, visit exactly the set of cities in bitmask $mask$ (where bit $k$ set means city $k$ is visited), and end at city $i$.
- Base: $dp[1\ll 0][0]=0$

Transition:

$$
dp[mask][i] = \min_{j \in mask,, j\ne i}; (dp[mask\setminus{i}][j] + dist[j][i])
$$

Answer:

$$
\min_{i\ne 0}; dp[(1\ll n)-1][i] + dist[i][0]
$$

Path reconstruction: store the predecessor for each $(mask,i)$.

#### How Does It Work (Plain Language)

We build tours by growing the set of visited cities. For every subset and last city $i$, we ask:

> "What was the best way to visit this subset without $i$, then hop from $j$ to $i$?"

This reuses smaller subproblems to solve larger ones, until all cities are visited.

#### Tiny Code (Easy Versions)

C (Bitmask DP with Reconstruction)

```c
#include <stdio.h>
#include <limits.h>

#define MAXN 20
#define INF  1000000000

int dist[MAXN][MAXN];
int prevCity[1<<MAXN][MAXN];
int dp[1<<MAXN][MAXN];

int main(void) {
    int n;
    scanf("%d", &n);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            scanf("%d", &dist[i][j]);

    int N = 1 << n;
    for (int m = 0; m < N; m++)
        for (int i = 0; i < n; i++)
            dp[m][i] = INF, prevCity[m][i] = -1;

    dp[1<<0][0] = 0;

    for (int mask = 0; mask < N; mask++) {
        for (int i = 0; i < n; i++) if (mask & (1<<i)) {
            int pmask = mask ^ (1<<i);
            if (pmask == 0) continue;
            for (int j = 0; j < n; j++) if (pmask & (1<<j)) {
                int val = dp[pmask][j] + dist[j][i];
                if (val < dp[mask][i]) {
                    dp[mask][i] = val;
                    prevCity[mask][i] = j;
                }
            }
        }
    }

    int all = N - 1, best = INF, last = -1;
    for (int i = 1; i < n; i++) {
        int val = dp[all][i] + dist[i][0];
        if (val < best) best = val, last = i;
    }
    printf("Min tour cost: %d\n", best);

    int path[MAXN+1], cnt = n;
    int mask = all, cur = last;
    path[--cnt] = cur;
    while (cur != 0) {
        int p = prevCity[mask][cur];
        mask ^= (1<<cur);
        cur = p;
        path[--cnt] = cur;
    }
    printf("Tour: ");
    for (int i = 0; i < n; i++) printf("%d ", path[i]);
    printf("0\n");
    return 0;
}
```

Python (Compact)

```python
import sys
INF = 1015

n = int(sys.stdin.readline())
dist = [list(map(int, sys.stdin.readline().split())) for _ in range(n)]

N = 1 << n
dp = [[INF]*n for _ in range(N)]
dp[1][0] = 0

for mask in range(N):
    if (mask & 1) == 0:
        continue
    for i in range(n):
        if (mask & (1<<i)) == 0:
            continue
        pmask = mask ^ (1<<i)
        if pmask == 0:
            continue
        for j in range(n):
            if (pmask & (1<<j)) == 0:
                continue
            dp[mask][i] = min(dp[mask][i], dp[pmask][j] + dist[j][i])

allmask = N - 1
ans = min(dp[allmask][i] + dist[i][0] for i in range(1, n))
print(ans)
```

#### Why It Matters

- Canonical bitmask DP example
- Exact solution with best-known time complexity for general TSP
- Template for subset-state DP problems: assignment, routing, path cover, Steiner tree

#### Step-by-Step Example

Suppose

$$
dist =
\begin{bmatrix}
0 & 10 & 15 & 20 \\
10 & 0 & 35 & 25 \\
15 & 35 & 0 & 30 \\
20 & 25 & 30 & 0
\end{bmatrix}
$$

The optimal tour is $0 \to 1 \to 3 \to 2 \to 0$ with cost $10+25+30+15=80$.


#### A Gentle Proof (Why It Works)

For any subset $S$ containing $0$ and endpoint $i\in S$,
the optimal path visiting $S$ and ending at $i$ must come from
some $j\in S\setminus{i}$ visiting $S\setminus{i}$ optimally and then edge $j\to i$.

Thus:

$$
dp[S][i] = \min_{j\in S\setminus{i}}(dp[S\setminus{i}][j] + dist[j][i])
$$

By processing subsets in increasing size, dependencies are always ready before use.

#### Try It Yourself

1. Reconstruct the tour path
2. Add must-visit or forbidden cities
3. Run on $n=15$ and observe scaling
4. Adapt to asymmetric TSP ($dist[i][j]\ne dist[j][i]$)
5. Compare with brute-force $O(n!)$

#### Test Cases

| n | dist                                                  | Expected |
| - | ----------------------------------------------------- | -------- |
| 2 | [[0,5],[5,0]]                                         | 10       |
| 3 | [[0,1,10],[1,0,2],[10,2,0]]                           | 13       |
| 4 | [[0,10,15,20],[10,0,35,25],[15,35,0,30],[20,25,30,0]] | 80       |
| 4 | [[0,3,4,2],[3,0,1,5],[4,1,0,6],[2,5,6,0]]             | 12       |

#### Complexity

- Time: $O(n^2 2^n)$
- Space: $O(n 2^n)$

Held–Karp DP is the foundation for exponential-time optimization over subsets, bridging combinatorial search and dynamic programming.

### 442 Subset DP (Over Subsets of States)

Subset DP is a powerful pattern where each DP state represents a subset of elements. It's used when problems depend on combinations of items, masks, or visited sets. You define transitions based on smaller subsets, building up to larger ones.

#### What Problem Are We Solving?

We want to compute some function $dp[S]$ over all subsets $S$ of a universe of size $n$ (where $S$ is represented as a bitmask).

Each $dp[S]$ depends on smaller subsets of $S$, typically by adding or removing one element at a time.

Common forms:

- Subset sums: $dp[S] = \sum_{T \subset S} f[T]$
- Maximums over subsets: $dp[S] = \max_{T \subset S} f[T]$
- Counting configurations: $dp[S] = \sum_{i \in S} dp[S \setminus {i}]$

The key idea: use bit operations and iterate through submasks efficiently.

#### How Does It Work (Plain Language)

Think of each subset as a state.
For example, if $n=3$, the subsets are:

| Mask | Binary | Subset  |
| ---- | ------ | ------- |
| 0    | 000    | ∅       |
| 1    | 001    | {0}     |
| 2    | 010    | {1}     |
| 3    | 011    | {0,1}   |
| 4    | 100    | {2}     |
| 5    | 101    | {0,2}   |
| 6    | 110    | {1,2}   |
| 7    | 111    | {0,1,2} |

Transitions depend on the structure of the problem:

- Additive (sum over submasks)
- Combinational (merge results)
- Stepwise (add/remove one bit)

#### Example 1: Sum Over Subsets (SOS DP)

We want $F[S] = \sum_{T \subseteq S} A[T]$.
Naively $O(3^n)$, but SOS DP does it in $O(n2^n)$.

Algorithm:

```c
for (int i = 0; i < n; i++)
  for (int mask = 0; mask < (1<<n); mask++)
    if (mask & (1<<i))
      F[mask] += F[mask ^ (1<<i)];
```

Each iteration adds contributions from subsets missing bit $i$.

#### Example 2: Counting Paths on Subsets

Suppose we count Hamiltonian paths on subsets:
$$
dp[S][i] = \sum_{j \in S, j \ne i} dp[S\setminus{i}][j]
$$
with base $dp[{i}][i]=1$.

Iterate all subsets, and for each subset and endpoint, sum over possible predecessors.

#### Tiny Code (Easy Versions)

C (Sum Over Subsets Example)

```c
#include <stdio.h>

int main(void) {
    int n = 3;
    int A[8] = {1, 2, 3, 4, 5, 6, 7, 8}; // arbitrary base values
    int F[8];
    for (int mask = 0; mask < (1<<n); mask++) F[mask] = A[mask];

    for (int i = 0; i < n; i++)
        for (int mask = 0; mask < (1<<n); mask++)
            if (mask & (1<<i))
                F[mask] += F[mask ^ (1<<i)];

    for (int mask = 0; mask < (1<<n); mask++)
        printf("F[%d] = %d\n", mask, F[mask]);

    return 0;
}
```

Python (Sum Over Subsets)

```python
n = 3
A = [1,2,3,4,5,6,7,8]
F = A[:]
for i in range(n):
    for mask in range(1<<n):
        if mask & (1<<i):
            F[mask] += F[mask ^ (1<<i)]
print(F)
```

#### Why It Matters

- Foundation for bitmask combinatorics
- Speeds up subset convolutions, inclusion-exclusion, and fast zeta transforms
- Essential in Steiner Tree DP, bitmask knapsack, and TSP variants

#### Step-by-Step Example

Let $A = [1,2,3,4,5,6,7,8]$, $n=3$.

We want $F[S]=\sum_{T\subseteq S}A[T]$.

- $F[0]=A[0]=1$
- $F[1]=A[1]+A[0]=3$
- $F[3]=A[3]+A[2]+A[1]+A[0]=10$

At the end:
$F = [1,3,4,10,5,12,13,36]$

#### A Gentle Proof (Why It Works)

Each mask represents a subset $S$.
When we iterate bit $i$, we add $F[S\setminus{i}]$ to $F[S]$ if $i\in S$.
This propagates values from smaller subsets to larger ones, accumulating all submask contributions.
The loop order ensures every submask is processed before supersets containing it.

#### Try It Yourself

1. Implement Sum Over Supersets (SOS Superset) by flipping the condition.
2. Compute $\max_{T\subseteq S} A[T]$ instead of sum.
3. Use subset DP to count number of ways to cover a set with given subsets.
4. Combine subset DP with bitcount(mask) to handle per-size transitions.
5. Visualize subset lattice as a hypercube traversal.

#### Test Cases

| A                 | Expected F (Sum Over Subsets) |
| ----------------- | ----------------------------- |
| [1,2,3,4,5,6,7,8] | [1,3,4,10,5,12,13,36]         |
| [0,1,0,1,0,1,0,1] | [0,1,0,2,0,2,0,4]             |
| [1,0,0,0,0,0,0,0] | [1,1,1,1,1,1,1,1]             |

#### Complexity

- Time: $O(n2^n)$
- Space: $O(2^n)$

Subset DP is a unifying pattern for problems on sets, once you see the bitmask, think "DP over subsets".

### 443 Hamiltonian Path DP (State Compression)

The Hamiltonian Path DP problem asks for the shortest path that visits every vertex exactly once, without needing to return to the start. It's a close sibling of the Traveling Salesman Problem (TSP) but without the final return edge. Using bitmask DP, we can solve it in $O(n^2 2^n)$.

#### What Problem Are We Solving?

Given a complete or weighted directed graph with $n$ vertices and a cost matrix $dist[i][j]$, find the minimum-cost path that visits all vertices exactly once.

We don't need to return to the starting node (unlike TSP).

Define the DP state:

- $dp[mask][i]$: the minimum cost to visit exactly the set of vertices in $mask$ and end at vertex $i$.

Base case:

$$
dp[1<<i][i] = 0 \quad \forall i
$$

Transition:

$$
dp[mask][i] = \min_{j \in mask,\ j \ne i}\big(dp[mask \setminus {i}][j] + dist[j][i]\big)
$$

Answer:

$$
\min_{i} dp[(1<<n)-1][i]
$$

#### How Does It Work (Plain Language)

Imagine we're constructing paths step by step:

1. Each mask represents which vertices we've already visited.
2. Each endpoint $i$ means we finish the path at vertex $i$.
3. We build $dp[mask][i]$ from smaller subsets by adding one last vertex $i$.

At each step, we check all $j$ that could be the previous vertex in the path.

No need to add $dist[i][start]$ because we don't return, it's a path, not a cycle.

#### Tiny Code (Easy Versions)

C (Hamiltonian Path DP)

```c
#include <stdio.h>
#include <limits.h>

#define MAXN 20
#define INF 1000000000

int dist[MAXN][MAXN];
int dp[1<<MAXN][MAXN];

int main(void) {
    int n;
    scanf("%d", &n);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            scanf("%d", &dist[i][j]);

    int N = 1 << n;
    for (int mask = 0; mask < N; mask++)
        for (int i = 0; i < n; i++)
            dp[mask][i] = INF;

    for (int i = 0; i < n; i++)
        dp[1<<i][i] = 0;

    for (int mask = 0; mask < N; mask++) {
        for (int i = 0; i < n; i++) {
            if (!(mask & (1<<i))) continue;
            int pmask = mask ^ (1<<i);
            if (pmask == 0) continue;
            for (int j = 0; j < n; j++) {
                if (!(pmask & (1<<j))) continue;
                int cost = dp[pmask][j] + dist[j][i];
                if (cost < dp[mask][i]) dp[mask][i] = cost;
            }
        }
    }

    int all = N - 1;
    int best = INF;
    for (int i = 0; i < n; i++)
        if (dp[all][i] < best)
            best = dp[all][i];

    printf("Minimum Hamiltonian path cost: %d\n", best);
    return 0;
}
```

Python (Compact Version)

```python
INF = 1015
n = int(input())
dist = [list(map(int, input().split())) for _ in range(n)]

N = 1 << n
dp = [[INF]*n for _ in range(N)]
for i in range(n):
    dp[1<<i][i] = 0

for mask in range(N):
    for i in range(n):
        if not (mask & (1<<i)): continue
        pmask = mask ^ (1<<i)
        if pmask == 0: continue
        for j in range(n):
            if not (pmask & (1<<j)): continue
            dp[mask][i] = min(dp[mask][i], dp[pmask][j] + dist[j][i])

ans = min(dp[N-1][i] for i in range(n))
print(ans)
```

#### Why It Matters

- Fundamental state compression DP pattern
- Useful when the problem involves visiting all nodes exactly once
- Core for path planning, ordering constraints, and bitmask search

#### Step-by-Step Example

Let

$$
dist =
\begin{bmatrix}
0 & 1 & 4 \
1 & 0 & 2 \
4 & 2 & 0
\end{bmatrix}
$$

Paths:

- $0 \to 1 \to 2$: $1 + 2 = 3$
- $0 \to 2 \to 1$: $4 + 2 = 6$
- $1 \to 0 \to 2$: $1 + 4 = 5$

So minimum path = 3.
$dp[(1<<3)-1] = dp[7] = [5,3,5]$, $\min=3$.

#### A Gentle Proof (Why It Works)

We apply optimal substructure:

For each subset $S$ and endpoint $i$,
the optimal Hamiltonian path to $(S,i)$ must extend an optimal path to $(S\setminus{i}, j)$ by one edge $j\to i$.
This recurrence ensures no city is revisited and all are included once.

Each $dp[mask][i]$ depends only on smaller masks, so it can be built bottom-up.

#### Try It Yourself

1. Add start node constraint (fix path must start at 0).
2. Recover the actual path using a `parent` array.
3. Modify for maximum path (replace `min` with `max`).
4. Adapt for directed graphs with asymmetric costs.
5. Use bit tricks like `mask & -mask` to iterate bits efficiently.

#### Test Cases

| n | dist                                      | Expected |
| - | ----------------------------------------- | -------- |
| 3 | [[0,1,4],[1,0,2],[4,2,0]]                 | 3        |
| 4 | [[0,3,1,5],[3,0,6,7],[1,6,0,2],[5,7,2,0]] | 6        |
| 2 | [[0,5],[5,0]]                             | 5        |
| 3 | [[0,9,9],[9,0,1],[9,1,0]]                 | 10       |

#### Complexity

- Time: $O(n^2 2^n)$
- Space: $O(n 2^n)$

Hamiltonian Path DP is the core structure for problems involving traversal of all nodes exactly once, simple, powerful, and a template for countless variants.

### 444 Assignment Problem DP (Mask over Tasks)

The Assignment Problem asks for the minimum total cost to assign $n$ workers to $n$ tasks with each worker doing exactly one task and each task done by exactly one worker. Besides the Hungarian algorithm, a clean solution for small $n$ is bitmask DP over subsets of tasks.

#### What Problem Are We Solving?

Given a cost matrix $C$ where $C[i][j]$ is the cost for worker $i$ to do task $j$, find the minimum cost perfect matching between workers and tasks.

State definition:

- Let $mask$ encode which tasks have been taken.
- Let $i=\text{popcount}(mask)$ be the number of already assigned workers, meaning we are about to assign worker $i$.

DP state:

$$
dp[mask]=\text{minimum total cost to assign the first } \text{popcount}(mask)\text{ workers to the set of tasks in }mask
$$

Transition:

$$
dp[mask\cup{j}]=\min\big(dp[mask]+C[i][j]\big)\quad\text{for all tasks }j\notin mask,\ i=\text{popcount}(mask)
$$

Base and answer:

$$
dp[0]=0,\qquad \text{Answer}=dp[(1\ll n)-1]
$$

#### How Does It Work (Plain Language)

We process workers in order $0,1,\dots,n-1$. The bitmask tells which tasks are already used. For the next worker $i$, try assigning any free task $j$, add its cost, and carry the best. This builds up all partial matchings until all tasks are taken.

#### Tiny Code (Easy Versions)

C (Bitmask DP)

```c
#include <stdio.h>
#include <limits.h>

#define MAXN 20
#define INF  1000000000

int C[MAXN][MAXN];
int dp[1<<MAXN];

int main(void) {
    int n;
    scanf("%d", &n);
    for (int i=0;i<n;i++)
        for (int j=0;j<n;j++)
            scanf("%d",&C[i][j]);

    int N=1<<n;
    for (int m=0;m<N;m++) dp[m]=INF;
    dp[0]=0;

    for (int mask=0;mask<N;mask++) {
        int i = __builtin_popcount(mask);
        if (i>=n || dp[mask]==INF) continue;
        for (int j=0;j<n;j++) if (!(mask&(1<<j))) {
            int nmask = mask|(1<<j);
            if (dp[mask]+C[i][j] < dp[nmask])
                dp[nmask] = dp[mask]+C[i][j];
        }
    }
    printf("%d\n", dp[N-1]);
    return 0;
}
```

Python (Compact)

```python
import sys
INF = 1015

n = int(sys.stdin.readline())
C = [list(map(int, sys.stdin.readline().split())) for _ in range(n)]

N = 1 << n
dp = [INF]*N
dp[0] = 0

for mask in range(N):
    i = bin(mask).count("1")
    if i >= n or dp[mask] == INF:
        continue
    for j in range(n):
        if (mask >> j) & 1: 
            continue
        dp[mask | (1 << j)] = min(dp[mask | (1 << j)], dp[mask] + C[i][j])

print(dp[N-1])
```

#### Why It Matters

- Canonical example of state compression DP on subsets
- Simple and reliable when $n\le 20$ or so
- Baseline to compare with Hungarian algorithm and min cost flow
- Template for richer constraints: forbidden pairs, bonuses, prerequisites

#### Step-by-Step Example

Let

$$
C=
\begin{bmatrix}
9&2&7\
6&4&3\
5&8&1
\end{bmatrix}
$$

One optimal assignment is worker $0\to1$ (2), $1\to2$ (3), $2\to0$ (5) for total $2+3+5=10$. The DP explores all masks, always extending by one free task for the next worker.

#### A Gentle Proof (Why It Works)

Let $S$ be a set of tasks and $i=|S|$. Any optimal partial assignment of the first $i$ workers to $S$ must end by assigning worker $i-1$ to some $j\in S$. Removing task $j$ yields an optimal solution to $(S\setminus{j}, i-1)$ plus $C[i-1][j]$. Reversing this gives the forward transition from $dp[mask]$ to $dp[mask\cup{j}]$ with $i=\text{popcount}(mask)$. Since each transition increases the mask, filling masks in increasing popcount topologically respects dependencies.

#### Try It Yourself

1. Reconstruct the assignment by storing a parent task for each $mask$.
2. Add forbidden pairs by skipping those $(i,j)$.
3. Add a bonus matrix $B$ and minimize $\sum(C[i][j]-B[i][j])$.
4. Handle rectangular $n\times m$ by padding the smaller side with zero dummy costs.
5. Compare runtime against Hungarian on random instances.

#### Test Cases

| $C$                                             | Expected min |
| ----------------------------------------------- | ------------ |
| $\begin{bmatrix}1 & 2 \\ 2 & 1\end{bmatrix}$    | $2$          |
| $\begin{bmatrix}9 & 2 & 7 \\ 6 & 4 & 3 \\ 5 & 8 & 1\end{bmatrix}$ | $10$ |
| $\begin{bmatrix}4 & 1 & 3 \\ 2 & 0 & 5 \\ 3 & 2 & 2\end{bmatrix}$ | $5$  |
| $\begin{bmatrix}10\end{bmatrix}$                | $10$         |


#### Complexity

- Time: $O(n^2 2^n)$
- Space: $O(2^n)$

Bitmask DP for the assignment problem is a tidy blueprint: iterate masks, assign the next worker to a free task, and keep the cheapest extension until all tasks are taken.

### 445 Partition into Two Sets (Balanced Load)

The Partition Problem asks whether a given set of numbers can be split into two subsets with equal sum. In its optimization form, we aim to minimize the difference between subset sums. It's a classic subset DP example that models balanced workloads, resource allocation, and load balancing.

#### What Problem Are We Solving?

Given an array $A[0..n-1]$, partition it into two subsets $S_1$ and $S_2$ such that the difference of sums is minimized:

$$
\text{minimize } |sum(S_1) - sum(S_2)|
$$

Equivalently, find a subset with sum as close as possible to half the total:

$$
\text{target} = \left\lfloor \frac{\sum A}{2} \right\rfloor
$$

We use DP to find all achievable sums up to `target`.

Define boolean DP:

$$
dp[i][s] = 1 \text{ if some subset of the first } i \text{ elements has sum } s
$$

Transition:

$$
dp[i][s] = dp[i-1][s] \lor dp[i-1][s-A[i-1]] \quad \text{if } s\ge A[i-1]
$$

Answer:

$$
\text{Find largest } s\le target \text{ with } dp[n][s]=1,\ \text{difference} = total - 2s
$$

#### How Does It Work (Plain Language)

Think of it as filling a knapsack of capacity `target`.
Each item can either go into the subset or stay out.
We try all combinations of sums up to half the total, beyond that, the second subset mirrors it.
The closest sum to `target` yields the minimal difference.

#### Tiny Code (Easy Versions)

C (Tabulation)

```c
#include <stdio.h>
#include <string.h>
#define MAXN 100
#define MAXSUM 10000

int main(void) {
    int n, A[MAXN];
    scanf("%d", &n);
    int total = 0;
    for (int i = 0; i < n; i++) {
        scanf("%d", &A[i]);
        total += A[i];
    }

    int target = total / 2;
    int dp[MAXSUM + 1] = {0};
    dp[0] = 1;

    for (int i = 0; i < n; i++) {
        for (int s = target; s >= A[i]; s--) {
            if (dp[s - A[i]]) dp[s] = 1;
        }
    }

    int best = 0;
    for (int s = target; s >= 0; s--) {
        if (dp[s]) { best = s; break; }
    }

    int diff = total - 2 * best;
    printf("Minimal difference: %d\n", diff);
    return 0;
}
```

Python (Concise Version)

```python
A = list(map(int, input().split()))
total = sum(A)
target = total // 2
dp = [False]*(target+1)
dp[0] = True

for x in A:
    for s in range(target, x-1, -1):
        if dp[s-x]:
            dp[s] = True

for s in range(target, -1, -1):
    if dp[s]:
        print("Minimal difference:", total - 2*s)
        break
```

#### Why It Matters

- Models balanced partitioning of workloads, memory, or resources
- Foundation for Subset Sum and Knapsack problems
- Introduces boolean DP over sums, a crucial building block for combinatorial search

#### Step-by-Step Example

Let $A=[1,6,11,5]$.
Total $=23$, target $=11$.

Feasible sums:
${0,1,5,6,7,11,12,16,17,18,23}$

Best $s=11$, minimal difference $23-2\cdot11=1$.

Partition: $[11]$ and $[1,5,6]$.

#### A Gentle Proof (Why It Works)

Each element either belongs to subset $S_1$ or $S_2$.
Let $s_1$ be sum of $S_1$, $s_2=total-s_1$.
We want $|s_1-s_2|$ minimized → $|total-2s_1|$.
By exploring all achievable sums $s_1\le total/2$, we find the $s_1$ closest to half.
Boolean DP tracks reachability using inclusion-exclusion transitions.

#### Try It Yourself

1. Count number of balanced partitions (replace boolean with integer DP).
2. Add constraint "each subset must have at least $k$ elements."
3. Extend to multi-set partitions (3 or more subsets).
4. Visualize reachable sums as boolean array transitions.
5. Compare with brute-force subset enumeration.

#### Test Cases

| A            | Minimal Difference | Partition      |
| ------------ | ------------------ | -------------- |
| [1,6,11,5]   | 1                  | [11] [1,5,6]   |
| [3,1,4,2,2]  | 0                  | [3,2] [4,2]    |
| [1,2,7]      | 4                  | [7] [1,2]      |
| [2,2,2,2]    | 0                  | [2,2] [2,2]    |
| [10,20,15,5] | 0                  | [10,15] [20,5] |

#### Complexity

- Time: $O(n\cdot sum)$
- Space: $O(sum)$

The Partition DP is a gentle bridge from Subset Sum to balanced optimization, teaching how combinatorial structure guides numerical state transitions.

### 446 Count Hamiltonian Cycles (Bitmask Enumeration)

The Hamiltonian Cycle Counting problem asks: given a graph, how many distinct Hamiltonian cycles (closed tours visiting each vertex exactly once) exist?
Unlike the shortest-path variants, this version focuses on counting all possible cycles using bitmask DP.

#### What Problem Are We Solving?

Given a graph $G=(V,E)$ with $|V|=n$, count the number of distinct Hamiltonian cycles starting and ending at vertex $0$ that visit every vertex exactly once.

We define:

- $dp[mask][i]$ = number of Hamiltonian paths starting at $0$, visiting all vertices in $mask$, and ending at $i$
- Base: $dp[1<<0][0]=1$

Transition:

$$
dp[mask][i] = \sum_{j\in mask,, j\ne i,,(j,i)\in E} dp[mask\setminus{i}][j]
$$

Answer (number of cycles):

$$
\sum_{i=1}^{n-1} dp[(1<<n)-1][i] \text{ if } (i,0)\in E
$$

Each valid end vertex $i$ must connect back to $0$ to complete the cycle.

#### How Does It Work (Plain Language)

We build all possible paths from vertex $0$ that cover a subset of vertices and end at some $i$.
For each step, we extend smaller paths by adding a new endpoint $i$.
When all vertices are visited ($mask=(1<<n)-1$), we check which endpoints connect back to $0$.
Summing these gives the total number of Hamiltonian cycles.

This is similar to the Held–Karp DP, but the operation is addition (counting) instead of minimization.

#### Tiny Code (Easy Versions)

C (Bitmask Counting DP)

```c
#include <stdio.h>
#include <string.h>

#define MAXN 20

long long dp[1<<MAXN][MAXN];
int adj[MAXN][MAXN];

int main(void) {
    int n, m;
    scanf("%d %d", &n, &m);
    memset(adj, 0, sizeof(adj));
    for (int e = 0; e < m; e++) {
        int u, v;
        scanf("%d %d", &u, &v);
        adj[u][v] = adj[v][u] = 1;
    }

    dp[1][0] = 1; // start at vertex 0

    for (int mask = 1; mask < (1 << n); mask++) {
        for (int i = 0; i < n; i++) {
            if (!(mask & (1 << i))) continue;
            for (int j = 0; j < n; j++) {
                if (i == j || !(mask & (1 << j))) continue;
                if (adj[j][i])
                    dp[mask][i] += dp[mask ^ (1 << i)][j];
            }
        }
    }

    long long total = 0;
    int full = (1 << n) - 1;
    for (int i = 1; i < n; i++) {
        if (adj[i][0]) total += dp[full][i];
    }

    printf("%lld\n", total / 2); // divide by 2 for undirected graphs
    return 0;
}
```

Python (Compact Version)

```python
n, m = map(int, input().split())
adj = [[0]*n for _ in range(n)]
for _ in range(m):
    u, v = map(int, input().split())
    adj[u][v] = adj[v][u] = 1

N = 1 << n
dp = [[0]*n for _ in range(N)]
dp[1][0] = 1

for mask in range(N):
    for i in range(n):
        if not (mask & (1<<i)): continue
        for j in range(n):
            if i==j or not (mask & (1<<j)): continue
            if adj[j][i]:
                dp[mask][i] += dp[mask ^ (1<<i)][j]

full = N-1
total = sum(dp[full][i] for i in range(1,n) if adj[i][0])
print(total // 2)
```

#### Why It Matters

- Demonstrates DP counting over subsets
- Foundation for counting Hamiltonian paths, cycle covers, and tours
- Appears in graph enumeration, combinatorial design, and traveling salesman counting

#### Step-by-Step Example

Let the graph be a square (4-cycle):

Vertices: $0,1,2,3$

Edges: $(0,1),(1,2),(2,3),(3,0)$ and $(0,3),(3,2),(2,1),(1,0)$

All Hamiltonian cycles:

- $0\to1\to2\to3\to0$
- $0\to3\to2\to1\to0$

So total = 2.

The DP constructs these paths incrementally by adding one vertex at a time.

#### A Gentle Proof (Why It Works)

Each DP state $(mask,i)$ corresponds to unique partial paths visiting $mask$ and ending at $i$.
To form a path ending at $i$, we must come from $j\in mask\setminus{i}$ with edge $(j,i)$.
This ensures each path is counted exactly once.

At the full mask, we have all Hamiltonian paths starting at $0$ and ending at $i$; connecting $i\to0$ closes the cycle.

Divide by 2 for undirected graphs since each cycle is counted twice (once clockwise, once counterclockwise).

#### Try It Yourself

1. Remove division by 2 for directed graphs.
2. Count Hamiltonian paths (no return edge).
3. Modify to track path sequences using parent arrays.
4. For large $n$, compare with inclusion-exclusion counting.
5. Implement memoized recursion (top-down version).

#### Test Cases

| Graph                | Expected        |
| -------------------- | --------------- |
| Triangle $(0-1-2-0)$ | 1               |
| Square $(0-1-2-3-0)$ | 2               |
| Line $(0-1-2)$       | 0               |
| Complete graph $K_4$ | $(4-1)!/2 = 3$  |
| $K_5$                | $(5-1)!/2 = 12$ |

#### Complexity

- Time: $O(n^2 2^n)$
- Space: $O(n 2^n)$

Counting Hamiltonian cycles via DP elegantly blends subset enumeration with path counting, offering an exact combinatorial count for small graphs.

### 447 Steiner Tree DP

The Steiner Tree problem asks for the minimum-cost subgraph that connects a given set of terminals in a weighted graph. You may use extra non-terminal vertices (Steiner nodes) if that reduces total cost. The classic exact DP for small numbers of terminals is the Dreyfus–Wagner subset DP.

#### What Problem Are We Solving?

Input: an undirected connected graph with nonnegative edge weights, and a terminal set $T={t_1,\dots,t_k}$.

Goal: find a minimum-weight tree that connects all vertices in $T$.

We use a subset DP over terminals and a root vertex:

- Precompute all-pairs shortest paths $dist[u][v]$.
- DP state: $dp[S][v]=$ minimum cost of a tree that connects all terminals in subset $S\subseteq T$ and whose tree is rooted at vertex $v$.

Initialization:
$$
dp[{t_i}][v]=dist[v][t_i]
$$

Combine subsets at the same root:
$$
dp[S][v]=\min_{\emptyset\ne A\subset S}\big(dp[A][v]+dp[S\setminus A][v]\big)
$$

Then allow the root to move via shortest paths:
$$
dp[S][v]=\min_{u}\big(dp[S][u]+dist[u][v]\big)
$$

Answer:
$$
\min_{v}dp[T][v]
$$

In practice we alternate subset-combine at fixed $v$ and a multi-source shortest-path relaxation for each $S$.

#### How Does It Work (Plain Language)

Think of building a Steiner tree by gluing together optimal trees for smaller terminal subsets at a meeting vertex $v$. After merging, you are allowed to slide that meeting point anywhere in the graph using shortest paths. Repeat for all subsets until you cover all terminals.

#### Tiny Code (Easy Version, Python)

Dreyfus–Wagner with Floyd–Warshall for $dist$ and Dijkstra-based relax per subset. Works for small $k$.

```python
import heapq

INF = 1015

def floyd_warshall(n, w):
    dist = [row[:] for row in w]
    for i in range(n):
        dist[i][i] = min(dist[i][i], 0)
    for k in range(n):
        for i in range(n):
            dik = dist[i][k]
            if dik == INF: continue
            rowi = dist[i]
            rowk = dist[k]
            for j in range(n):
                nd = dik + rowk[j]
                if nd < rowi[j]: rowi[j] = nd
    return dist

def steiner_tree(n, edges, terminals):
    # Build dense weight matrix
    w = [[INF]*n for _ in range(n)]
    for u in range(n):
        w[u][u] = 0
    for u, v, c in edges:
        if c < w[u][v]:
            w[u][v] = w[v][u] = c

    dist = floyd_warshall(n, w)

    k = len(terminals)
    term_index = {t:i for i,t in enumerate(terminals)}

    # dp[mask][v]
    dp = [[INF]*n for _ in range(1<<k)]
    for t in terminals:
        m = 1 << term_index[t]
        for v in range(n):
            dp[m][v] = dist[v][t]

    # subset DP
    for mask in range(1, 1<<k):
        # combine proper nonempty A subset at same root v
        sub = (mask-1) & mask
        while sub:
            other = mask ^ sub
            if other:
                for v in range(n):
                    val = dp[sub][v] + dp[other][v]
                    if val < dp[mask][v]:
                        dp[mask][v] = val
            sub = (sub-1) & mask

        # root-move relaxation by Dijkstra over complete graph with dist metric
        # This is equivalent to: dp[mask] = metric-closure shortest-path from sources with potentials dp[mask]
        # Implement 1 run of Dijkstra with initial costs dp[mask][*]
        pq = [(dp[mask][v], v) for v in range(n)]
        heapq.heapify(pq)
        seen = [False]*n
        while pq:
            d,v = heapq.heappop(pq)
            if seen[v]: continue
            seen[v] = True
            if d > dp[mask][v]: continue
            row = dist[v]
            for u in range(n):
                nd = d + row[u]
                if nd < dp[mask][u]:
                    dp[mask][u] = nd
                    heapq.heappush(pq, (nd, u))

    full = (1<<k) - 1
    return min(dp[full])

# Example usage:
# n = 5
# edges = [(0,1,1),(1,2,1),(2,3,1),(3,4,1),(0,4,10),(1,4,2)]
# terminals = [0,3,4]
# print(steiner_tree(n, edges, terminals))
```

Notes:

- For sparse graphs you can skip Floyd–Warshall and run Dijkstra inside the relax step using the original adjacency. Using the metric closure as above is simple and correct for nonnegative weights.

#### Why It Matters

- Exact algorithm for Steiner trees when the number of terminals $k$ is small
- Standard approach in VLSI routing, network design, and phylogenetics
- Teaches a powerful pattern: subset merge at a root plus shortest-path relaxation

#### Step-by-Step Example

Suppose a 5-vertex path $0$–$1$–$2$–$3$–$4$ with unit weights and terminals $T={0,3,4}$.

- Singletons: $dp[{0}][v]=dist[v][0]$, etc.
- Combine ${3}$ and ${4}$ at $v=3$ or $v=4$, then relax along the path.
- Finally combine with ${0}$; the optimal tree is edges $(0,1),(1,2),(2,3),(3,4)$ with total cost $4$.

#### A Gentle Proof (Why It Works)

Let $S\subseteq T$ and $v$ be a meeting vertex of an optimal Steiner tree for $S$. The tree decomposes into two subtrees whose terminal sets partition $S$, and both subtrees meet at $v$. Hence
$$
dp[S][v]\le dp[A][v]+dp[S\setminus A][v].
$$
Conversely, any combination at $v$ plus a shortest-path relocation of $v$ to another vertex is no worse than explicitly wiring with edges, due to triangle inequality from the metric closure. Induction over $|S|$ yields optimality.

#### Try It Yourself

1. Replace the Dijkstra relax with Bellman–Ford to allow zero edges with tight potentials.
2. Extract the actual Steiner tree: store the best split and predecessor during relaxation.
3. Compare metric-closure method vs relaxing on the original sparse graph.
4. Benchmark vs MST over terminals to see the benefit of Steiner vertices.
5. Add a constraint that certain vertices are forbidden as Steiner nodes.

#### Test Cases

| Graph                                               | Terminals | Expected Steiner cost |
| --------------------------------------------------- | --------- | --------------------- |
| Path 0–1–2–3–4 with unit edges                      | {0,3,4}   | 4                     |
| Triangle 0–1–2 with unit edges                      | {0,2}     | 1                     |
| Square 0–1–2–3 with unit edges, diagonal 1–3 cost 1 | {0,2,3}   | 2                     |
| Star center 0 to 1..4 all cost 1                    | {1,2,3,4} | 4                     |

#### Complexity

- With metric closure and subset merge: $O(3^k\cdot n + 2^k\cdot n\log n)$ typical
- Memory: $O(2^k\cdot n)$

The Dreyfus–Wagner DP is the go-to exact method when $k$ is small: combine subsets at a root, then relax through shortest paths to let Steiner nodes emerge automatically.

### 448 SOS DP (Sum Over Subsets)

Sum Over Subsets (SOS DP) is a clever bitmask dynamic programming technique for computing values aggregated over all subsets of each mask efficiently, without enumerating all subset pairs explicitly.

#### What Problem Are We Solving?

Suppose you have an array `f[mask]` defined for all bitmasks of length `n`, and you want to compute:

$$
g[mask] = \sum_{sub \subseteq mask} f[sub]
$$

Naively, this requires iterating over all subsets of each mask, which takes $O(3^n)$.
With SOS DP, we can compute all $g[mask]$ in $O(n2^n)$ time.

#### How Does It Work (Plain Language)

Think of each bit in the mask as a "dimension."
For each bit position `i`, if that bit is set in the mask, we can inherit contributions from the version where the bit was off.
We build up sums by iterating over each bit dimension and folding smaller subsets upward.

#### Transition Formula

Let `dp[mask]` initially equal `f[mask]`.
Then for each bit `i` from 0 to `n-1`:

```text
if mask has bit i set:
    dp[mask] += dp[mask ^ (1 << i)]
```

After processing all bits, `dp[mask]` holds the sum over all subsets of `mask`.

#### Example

Let $n=3$, masks from `000` to `111`.

If $f[mask]=1$ for all masks, then:

| mask | subsets           | g[mask] |
| ---- | ----------------- | ------- |
| 000  | {000}             | 1       |
| 001  | {000,001}         | 2       |
| 010  | {000,010}         | 2       |
| 011  | {000,001,010,011} | 4       |
| 100  | {000,100}         | 2       |
| 111  | all 8 subsets     | 8       |

The DP folds these results bit by bit.

#### Tiny Code (Easy Versions)

C

```c
#include <stdio.h>
#define MAXN (1<<20)

long long f[MAXN], dp[MAXN];

int main(void) {
    int n;
    scanf("%d", &n);
    int N = 1 << n;
    for (int mask = 0; mask < N; mask++) {
        scanf("%lld", &f[mask]);
        dp[mask] = f[mask];
    }

    for (int i = 0; i < n; i++) {
        for (int mask = 0; mask < N; mask++) {
            if (mask & (1 << i)) {
                dp[mask] += dp[mask ^ (1 << i)];
            }
        }
    }

    for (int mask = 0; mask < N; mask++)
        printf("%lld ", dp[mask]);
    printf("\n");
}
```

Python

```python
n = int(input("n: "))
N = 1 << n
f = list(map(int, input().split()))
dp = f[:]

for i in range(n):
    for mask in range(N):
        if mask & (1 << i):
            dp[mask] += dp[mask ^ (1 << i)]

print(dp)
```

#### Why It Matters

- Core trick in bitmask convolution, subset transforms, XOR convolution, and mobius inversion.
- Reduces $O(3^n)$ subset loops to $O(n2^n)$.
- Common in problems over subset sums, probabilistic DP, counting states, and game theory.

#### A Gentle Proof (Why It Works)

We can represent each mask as a binary vector of $n$ bits.
Each bit dimension $i$ adds subsets with $i$th bit unset.
Inductively, after processing bit $i$, each mask accumulates contributions from all subsets differing only in bits $\le i$.
By the end, every subset of `mask` has been included exactly once.

Formally, for each subset $sub\subseteq mask$, there exists a sequence of bit additions leading from $sub$ to $mask$, ensuring its inclusion.

#### Try It Yourself

1. Reverse the process to compute Sum Over Supersets instead.
2. Modify to compute product over subsets.
3. Apply to count subsets satisfying parity conditions.
4. Use SOS DP to precompute subset sums before a subset convolution.
5. Combine with inclusion-exclusion for faster combinatorial counting.

#### Test Cases

| n | f (input) | Expected dp (output) |
| - | --------- | -------------------- |
| 2 | 1 1 1 1   | 1 2 2 4              |
| 2 | 1 2 3 4   | 1 3 4 10             |
| 3 | all 1s    | 1 2 2 4 2 4 4 8      |

#### Complexity

- Time: $O(n2^n)$
- Space: $O(2^n)$

SOS DP is a cornerstone of bitmask dynamic programming, revealing structure across subsets without enumerating them explicitly.

### 449 Bitmask Knapsack (State Compression)

The Bitmask Knapsack technique encodes subsets of items using bitmasks, allowing you to represent selections, transitions, and constraints compactly. It's a bridge between subset enumeration and dynamic programming, especially useful when the number of items is small (e.g. $n \le 20$) but the value/weight range is large.

#### What Problem Are We Solving?

Given $n$ items, each with weight $w_i$ and value $v_i$, and capacity $W$, choose a subset with total weight ≤ $W$ to maximize total value.

Instead of indexing DP by capacity, we can enumerate subsets via bitmask:

$$
best = \max_{\text{subset}} \sum_{i \in \text{subset}} v_i \quad \text{such that } \sum_{i \in \text{subset}} w_i \le W
$$

Each subset corresponds to an integer mask, where bit $i$ indicates inclusion of item $i$.

#### How Does It Work (Plain Language)

A bitmask is a snapshot of which items are taken. You precompute total weight and total value for each subset.
Then simply iterate all masks, filter by capacity, and keep the best.

It's exponential ($2^n$) but works when $n$ is small and weights are large, where classical DP by weight is infeasible.

#### Transition Formula

For each mask:

- Compute
  $$
  total_w = \sum_{i:mask_i=1} w_i
  $$
  $$
  total_v = \sum_{i:mask_i=1} v_i
  $$
- If $total_w \le W$, update answer:
  $$
  best = \max(best, total_v)
  $$

Or incrementally:

$$
dp[mask] = \sum_{i:mask_i=1} v_i
$$
$$
weight[mask] = \sum_{i:mask_i=1} w_i
$$

#### Tiny Code (Easy Versions)

C

```c
#include <stdio.h>

#define MAXN 20
#define MAXMASK (1<<MAXN)

int main(void) {
    int n, W;
    scanf("%d %d", &n, &W);
    int w[n], v[n];
    for (int i = 0; i < n; i++) scanf("%d %d", &w[i], &v[i]);

    int N = 1 << n;
    int best = 0;

    for (int mask = 0; mask < N; mask++) {
        int total_w = 0, total_v = 0;
        for (int i = 0; i < n; i++) {
            if (mask & (1 << i)) {
                total_w += w[i];
                total_v += v[i];
            }
        }
        if (total_w <= W && total_v > best)
            best = total_v;
    }

    printf("%d\n", best);
}
```

Python

```python
n, W = map(int, input().split())
w, v = [], []
for _ in range(n):
    wi, vi = map(int, input().split())
    w.append(wi)
    v.append(vi)

best = 0
for mask in range(1 << n):
    total_w = total_v = 0
    for i in range(n):
        if mask & (1 << i):
            total_w += w[i]
            total_v += v[i]
    if total_w <= W:
        best = max(best, total_v)

print(best)
```

#### Why It Matters

- Works well when $n$ is small (e.g. $n \le 20$) but weights/values are large
- Natural fit for meet-in-the-middle and subset enumeration
- Simplifies reasoning about combinations, constraints, and transitions
- Found in traveling salesman variants, set packing, and team selection problems

#### Step-by-Step Example

Items:

| i | w | v |
| - | - | - |
| 0 | 3 | 4 |
| 1 | 4 | 5 |
| 2 | 2 | 3 |

Capacity $W=6$.

Subsets:

| mask | items   | weight | value | feasible |
| ---- | ------- | ------ | ----- | -------- |
| 000  | {}      | 0      | 0     | ✓        |
| 001  | {0}     | 3      | 4     | ✓        |
| 010  | {1}     | 4      | 5     | ✓        |
| 011  | {0,1}   | 7      | 9     | ✗        |
| 100  | {2}     | 2      | 3     | ✓        |
| 101  | {0,2}   | 5      | 7     | ✓        |
| 110  | {1,2}   | 6      | 8     | ✓        |
| 111  | {0,1,2} | 9      | 12    | ✗        |

Best feasible = mask `110` → value 8.

#### A Gentle Proof (Why It Works)

Each subset is a distinct combination of items. Enumerating all $2^n$ subsets guarantees completeness.
Feasibility is checked via total weight ≤ $W$, ensuring no invalid subset contributes.
Maximization over all feasible subsets returns the global optimum.

No overlapping subproblems, so no need for memoization. The entire search space is finite and explored.

#### Try It Yourself

1. Print all feasible subsets and their total values.
2. Combine with bitcount to restrict subset size.
3. Use meet-in-the-middle: split items into halves, enumerate each half, then merge results.
4. Extend to multi-dimensional capacity $(W_1,W_2,...)$.
5. Adapt to minimize weight for a target value instead.

#### Test Cases

| Items             | W | Expected |
| ----------------- | - | -------- |
| (3,4),(4,5),(2,3) | 6 | 8        |
| (2,3),(3,4),(4,5) | 5 | 7        |
| (1,1),(2,2),(3,3) | 3 | 3        |

#### Complexity

- Time: $O(n2^n)$
- Space: $O(1)$ (no DP table needed)

Bitmask Knapsack is a brute-force DP with compression, a go-to technique for small $n$, offering full flexibility when classical capacity-indexed DP would blow up.

### 450 Bitmask Independent Set (Graph Subset Optimization)

The Bitmask Independent Set DP enumerates all subsets of vertices in a graph to find the maximum-weight independent set, a set of vertices with no edges between any pair. It's a classic exponential DP, efficient for small graphs ($n \le 20$), using bit operations for adjacency and feasibility.

#### What Problem Are We Solving?

Given an undirected graph $G=(V,E)$ with vertex weights $w_i$, find:

$$
\max \sum_{i \in S} w_i \quad \text{subject to } \forall (u,v) \in E,\ u,v \notin S
$$

That is, choose a subset $S$ of vertices with no adjacent pairs, maximizing total weight.

We represent each subset $S$ by a bitmask, where bit $i=1$ means vertex $i$ is included.

#### How Does It Work (Plain Language)

We iterate over all $2^n$ subsets.
For each subset, we check whether it forms an independent set by ensuring no edge connects two chosen vertices.
If it's valid, sum its vertex weights and update the best.

Precompute adjacency masks to quickly test validity.

#### Transition / Check

For each mask:

1. Validity  
   A subset is independent if, for all vertices $i$ included, it contains no neighbor:
   $$
   (\,adj[i] \mathbin{\&} mask\,) = 0
   $$
   where $adj[i]$ is the bitmask of neighbors of vertex $i$.

2. Value
   $$
   value(mask) = \sum_{i:\,mask_i=1} w_i
   $$

3. Best
   $$
   best = \max\!\bigl(best,\ value(mask)\bigr)
   $$


#### Tiny Code (Easy Versions)

C

```c
#include <stdio.h>

#define MAXN 20

int main(void) {
    int n, m;
    scanf("%d %d", &n, &m);
    int w[n];
    for (int i = 0; i < n; i++) scanf("%d", &w[i]);

    int adj[n];
    for (int i = 0; i < n; i++) adj[i] = 0;
    for (int e = 0; e < m; e++) {
        int u, v;
        scanf("%d %d", &u, &v);
        adj[u] |= 1 << v;
        adj[v] |= 1 << u;
    }

    int N = 1 << n;
    int best = 0;

    for (int mask = 0; mask < N; mask++) {
        int ok = 1, val = 0;
        for (int i = 0; i < n; i++) {
            if (mask & (1 << i)) {
                if (adj[i] & mask) { ok = 0; break; }
                val += w[i];
            }
        }
        if (ok && val > best) best = val;
    }

    printf("%d\n", best);
}
```

Python

```python
n, m = map(int, input().split())
w = list(map(int, input().split()))

adj = [0]*n
for _ in range(m):
    u, v = map(int, input().split())
    adj[u] |= 1 << v
    adj[v] |= 1 << u

best = 0
for mask in range(1 << n):
    ok = True
    val = 0
    for i in range(n):
        if mask & (1 << i):
            if adj[i] & mask:
                ok = False
                break
            val += w[i]
    if ok:
        best = max(best, val)

print(best)
```

#### Why It Matters

- Solves maximum independent set (MIS) for small graphs
- Useful for exact solutions in constraint problems, treewidth-based DP, or bitmask search
- A building block in graph coloring, maximum clique, dominating set, and subset optimization
- Adaptable for unweighted (count) or weighted (sum) versions

#### Step-by-Step Example

Graph: 4 vertices, edges $(0,1), (1,2), (2,3)$
Weights: $[3, 2, 4, 1]$

Independent sets:

| Mask | Set   | Valid | Value |
| ---- | ----- | ----- | ----- |
| 0000 | ∅     | ✓     | 0     |
| 0001 | {0}   | ✓     | 3     |
| 0010 | {1}   | ✓     | 2     |
| 0100 | {2}   | ✓     | 4     |
| 1000 | {3}   | ✓     | 1     |
| 0101 | {0,2} | ✗     |,     |
| 1001 | {0,3} | ✓     | 4     |
| 1100 | {2,3} | ✗     |,     |
| 1010 | {1,3} | ✓     | 3     |

Best = 4 (either {2} or {0,3})

#### A Gentle Proof (Why It Works)

Each subset represents a candidate solution.
A subset is feasible iff it contains no adjacent pair, ensured by the adjacency mask check.
Since every subset is tested, the algorithm finds the global optimum by enumeration.

Bitmask feasibility check $(adj[i] \mathbin{\&} mask) == 0$ ensures constant-time validation per vertex, keeping complexity tight.


#### Try It Yourself

1. Modify to count all independent sets.
2. Restrict to subsets of exact size $k$.
3. Add memoization to prune invalid masks early.
4. Combine with meet-in-the-middle for $n=40$.
5. Flip edges to find maximum clique (complement graph).

#### Test Cases

| Graph         | Weights     | Expected |
| ------------- | ----------- | -------- |
| Chain 0–1–2–3 | [3,2,4,1]   | 4        |
| Star center 0 | [5,1,1,1,1] | 5        |
| Triangle      | [1,2,3]     | 3        |
| Empty graph   | [2,2,2]     | 6        |

#### Complexity

- Time: $O(n2^n)$
- Space: $O(n)$

Bitmask Independent Set DP explores all subsets systematically, perfect when graphs are small but weights or constraints are complex.

# Section 46. Digit DP and SOS DP

### 451 Count Numbers with Property (Digit DP)

Digit DP is a method for counting numbers within a range that satisfy digit-level constraints, like no leading zeros, digit sum, specific digits, or forbidden patterns. This algorithm builds results by processing digits one by one while maintaining states for prefix constraints and current properties.

#### What Problem Are We Solving?

Given an integer $N$ and a property $P$ (like "sum of digits is even"), count how many integers in $[0, N]$ satisfy $P$.

Example: Count numbers $\le 327$ where the sum of digits is even.

We process each digit from most significant to least significant, maintaining:

- pos: current digit index
- sum: accumulated property (e.g. sum of digits mod 2)
- tight: whether we are bounded by $N$'s prefix (0 = free, 1 = still tight)
- leading: whether we've only seen leading zeros (optional)

State:
$$
dp[pos][sum][tight]
$$

Transition over next digit `d` (0..limit):

- Update `next_sum = (sum + d) % 2`
- If `tight=1`, limit = digit at pos in $N$, else 9
- Move to next position

Answer is sum over all valid end states satisfying the property.

#### How Does It Work (Plain Language)

Digit DP works like counting with awareness:
At each step, you choose the next digit, updating what you know (like current sum), while respecting the upper bound.
By the time you process all digits, you've counted all valid numbers, no need to iterate up to $N$!

#### Transition Formula

Let $S$ be the string of digits of $N$. For each state:

$$
dp[pos][sum][tight] = \sum_{d=0}^{limit} dp[pos+1][(sum+d)\bmod M][tight \land (d==limit)]
$$

with base:
$$
dp[len][sum][tight] = 1 \text{ if property holds, else } 0
$$

Example: Property = sum of digits even → $sum%2=0$

#### Tiny Code (Easy Versions)

Python (Count numbers ≤ N with even digit sum)

```python
from functools import lru_cache

def count_even_sum(n):
    digits = list(map(int, str(n)))
    m = len(digits)

    @lru_cache(None)
    def dp(pos, sum_mod2, tight):
        if pos == m:
            return 1 if sum_mod2 == 0 else 0
        limit = digits[pos] if tight else 9
        total = 0
        for d in range(limit + 1):
            total += dp(pos + 1, (sum_mod2 + d) % 2, tight and d == limit)
        return total

    return dp(0, 0, True)

N = int(input("Enter N: "))
print(count_even_sum(N))
```

C (Recursive Memoized DP)

```c
#include <stdio.h>
#include <string.h>

int digits[20];
long long memo[20][2][2];
int len;

long long dp(int pos, int sum_mod2, int tight) {
    if (pos == len) return sum_mod2 == 0;
    if (memo[pos][sum_mod2][tight] != -1) return memo[pos][sum_mod2][tight];

    int limit = tight ? digits[pos] : 9;
    long long res = 0;
    for (int d = 0; d <= limit; d++) {
        res += dp(pos + 1, (sum_mod2 + d) % 2, tight && (d == limit));
    }
    return memo[pos][sum_mod2][tight] = res;
}

long long solve(long long n) {
    len = 0;
    while (n > 0) {
        digits[len++] = n % 10;
        n /= 10;
    }
    for (int i = 0; i < len / 2; i++) {
        int tmp = digits[i];
        digits[i] = digits[len - 1 - i];
        digits[len - 1 - i] = tmp;
    }
    memset(memo, -1, sizeof(memo));
    return dp(0, 0, 1);
}

int main(void) {
    long long n;
    scanf("%lld", &n);
    printf("%lld\n", solve(n));
}
```

#### Why It Matters

- Fundamental to digit-based counting problems
- Efficiently handles constraints on digits, sum, mod, parity, forbidden patterns
- Avoids looping through large ranges (works in $O(\text{len} \times \text{state})$)
- Core idea behind counting numbers with:

  * Even digit sum
  * Specific digits (e.g. no 4)
  * Digits increasing/decreasing
  * Remainder mod M conditions

#### Step-by-Step Example

Count numbers ≤ 327 with even digit sum.

We track sum mod 2:

- Start: pos=0, sum=0, tight=1
- First digit: choose 0..3

  * if 3 chosen → sum=1, next tight=1
  * else → free (tight=0)
- Continue until last digit
- At end, count where sum=0 (even)

Result: 164 numbers.

#### A Gentle Proof (Why It Works)

At each position, `tight` ensures we never exceed N, and recursive branching over digits ensures coverage of all valid prefixes.
By caching identical subproblems (same `pos`, `sum`, `tight`), we avoid recomputation.
Thus, total states = $O(len \times property_space \times 2)$.

#### Try It Yourself

1. Count numbers ≤ N with sum of digits divisible by 3.
2. Count numbers with no consecutive equal digits.
3. Count numbers with at most k nonzero digits.
4. Count numbers with digit product < M.
5. Adapt for range [L, R] via `solve(R) - solve(L-1)`.

#### Test Cases

| N   | Property       | Expected |
| --- | -------------- | -------- |
| 9   | Even digit sum | 5        |
| 20  | Even digit sum | 10       |
| 327 | Even digit sum | 164      |

#### Complexity

- Time: $O(len \times M \times 2)$
- Space: $O(len \times M \times 2)$

Digit DP transforms combinatorial counting into state-driven reasoning, digit by digit, a foundational trick for number-theoretic dynamic programming.

### 452 Count Without Adjacent Duplicates

This Digit DP problem counts numbers within a range that do not contain adjacent identical digits, a classic example where the state must remember the previous digit to enforce adjacency rules.

#### What Problem Are We Solving?

Given an integer $N$, count how many integers in $[0, N]$ have no two consecutive equal digits.

For example, up to $N = 1234$:

- Valid: 1203 (no repeats)
- Invalid: 1224 (two 2's together)

We'll use Digit DP to explore all digit sequences up to $N$, ensuring no adjacent duplicates.

#### DP State

Let the string of digits of $N$ be `S`, with length `len`.

State:
$$
dp[pos][prev][tight][leading]
$$

Where:

- `pos`: current index (0-based)
- `prev`: previous digit (0–9, or 10 if none yet)
- `tight`: whether prefix equals $N$ so far (1 = bound, 0 = free)
- `leading`: whether we've seen only leading zeros (1 = true)

Each state counts valid completions from position `pos` onward.

#### Transition

At position `pos`:

- Choose next digit `d` from 0 to `limit` (where `limit = S[pos]` if `tight = 1`, else 9)
- Skip if `d == prev` and not `leading` (no adjacent duplicates)
- Next state:

  * `next_prev = d` if not `leading` else 10
  * `next_tight = tight and (d == limit)`
  * `next_leading = leading and (d == 0)`

Sum all valid transitions.

Base:
$$
dp[len][prev][tight][leading] = 1
$$

when `pos == len` (end of number).

#### How Does It Work (Plain Language)

We're building numbers digit by digit:

- `tight` keeps us within bounds.
- `prev` remembers the last chosen digit, to prevent repeating it.
- `leading` helps skip leading zeros (which don't count as duplicates).

By caching all combinations, we count every valid number exactly once.

#### Tiny Code (Easy Versions)

Python

```python
from functools import lru_cache

def count_no_adjacent(N):
    digits = list(map(int, str(N)))
    m = len(digits)

    @lru_cache(None)
    def dp(pos, prev, tight, leading):
        if pos == m:
            return 1  # valid number
        limit = digits[pos] if tight else 9
        total = 0
        for d in range(limit + 1):
            if not leading and d == prev:
                continue  # no adjacent duplicates
            total += dp(
                pos + 1,
                d if not leading else 10,
                tight and d == limit,
                leading and d == 0
            )
        return total

    return dp(0, 10, True, True)

N = int(input("Enter N: "))
print(count_no_adjacent(N))
```

C

```c
#include <stdio.h>
#include <string.h>

int digits[20];
long long memo[20][11][2][2];
int len;

long long dp(int pos, int prev, int tight, int leading) {
    if (pos == len) return 1;
    if (memo[pos][prev][tight][leading] != -1) return memo[pos][prev][tight][leading];

    int limit = tight ? digits[pos] : 9;
    long long res = 0;
    for (int d = 0; d <= limit; d++) {
        if (!leading && d == prev) continue;
        int next_prev = leading && d == 0 ? 10 : d;
        int next_tight = tight && (d == limit);
        int next_leading = leading && (d == 0);
        res += dp(pos + 1, next_prev, next_tight, next_leading);
    }
    return memo[pos][prev][tight][leading] = res;
}

long long solve(long long n) {
    len = 0;
    while (n > 0) {
        digits[len++] = n % 10;
        n /= 10;
    }
    for (int i = 0; i < len / 2; i++) {
        int tmp = digits[i];
        digits[i] = digits[len - 1 - i];
        digits[len - 1 - i] = tmp;
    }
    memset(memo, -1, sizeof(memo));
    return dp(0, 10, 1, 1);
}

int main(void) {
    long long n;
    scanf("%lld", &n);
    printf("%lld\n", solve(n));
}
```

#### Why It Matters

- Classic Digit DP with "previous digit" state
- Enables constraints like:

  * No adjacent repeats
  * No increasing/decreasing sequences
  * No forbidden pairs
- Useful in pattern counting, PIN code generation, license plate validation

#### Step-by-Step Example

Count numbers ≤ 120 with no adjacent duplicates:

- Leading zeros allowed at first
- E.g. `101` ✓, `110` ✗
- DP checks each digit:

  * `1?0` (pos=0, prev=10)
  * For next digit: skip equal to prev
- Total valid count = 91

#### A Gentle Proof (Why It Works)

For each position, transitions ensure:

- Only digits ≤ bound if `tight=1`
- No consecutive duplicates via `d != prev`
- Leading zeros treated specially (ignored in duplicate check)

By iterating through all valid digits and memoizing results, each subproblem (prefix constraint + previous digit) is solved once, ensuring completeness and correctness.

#### Try It Yourself

1. Count numbers ≤ N with no equal adjacent digits and sum of digits even.
2. Count numbers ≤ N with strictly increasing digits.
3. Modify to disallow adjacent 0's only.
4. Combine with mod constraints (digit sum mod M).
5. Extend to handle exactly k equal pairs.

#### Test Cases

| N    | Expected | Notes           |
| ---- | -------- | --------------- |
| 9    | 10       | 0–9 all valid   |
| 11   | 10       | 10 invalid      |
| 100  | 91       | Only 9 invalids |
| 1234 | 820      | Approx result   |

#### Complexity

- Time: $O(len \times 11 \times 2 \times 2 \times 10)$
- Space: $O(len \times 11 \times 2 \times 2)$

Digit DP elegantly enforces local digit constraints (like adjacency) through memory of the previous digit, enabling fast counting across massive ranges.

### 453 Sum of Digits in Range

This Digit DP problem computes the sum of digits of all numbers in a given range $[0, N]$.
Instead of enumerating numbers, we accumulate digit contributions position by position, respecting upper bounds.

#### What Problem Are We Solving?

Given a number $N$, compute:

$$
S(N) = \sum_{x=0}^{N} \text{sum\_of\_digits}(x)
$$

For example:

- $S(13) = 1+0 + 1+1 + 1+2 + 1+3 = 55$

The goal is to compute $S(N)$ efficiently in $O(\text{len} \times 2 \times M)$, not $O(N)$.

You can also handle ranges:
$$
S(L,R) = S(R) - S(L-1)
$$

#### DP State

Let $S$ = list of digits of $N$.

We define a recursive function:
$$
dp[pos][tight][sum]
$$

But instead of counting numbers, we also accumulate the total digit sum contribution.

So the function returns (count, total_sum), a pair:

- `count`: number of valid numbers
- `sum`: sum of digits over all valid numbers

State:

- `pos`: current position (0..len-1)
- `tight`: whether we are still bounded by prefix
- `leading`: whether only leading zeros so far

#### Transition

At each position, choose digit `d` in `[0, limit]`
(limit = digit at pos if `tight = 1`, else 9)

Let `(cnt_next, sum_next)` = result from next position.

We add current digit's contribution:
$$
total_sum += sum_next + d \times cnt_next
$$

If `leading` is true and `d=0`, then we don't count that as a "real" leading digit.

Base case:
$$
dp[len][tight][leading] = (1, 0)
$$

(one valid number, sum = 0)

#### How Does It Work (Plain Language)

Each recursive call counts how many numbers are possible from this prefix, and how much total digit sum they produce.

When you pick a digit `d`:

- `d` contributes `d * cnt_next` to all numbers in this branch
- Plus whatever the rest of the digits contribute recursively

By caching `(count, sum)` per state, we reuse computations for repeated prefixes.

#### Tiny Code (Easy Version)

Python

```python
from functools import lru_cache

def sum_of_digits_upto(N):
    digits = list(map(int, str(N)))
    m = len(digits)

    @lru_cache(None)
    def dp(pos, tight, leading):
        if pos == m:
            return (1, 0)  # (count, sum)

        limit = digits[pos] if tight else 9
        total_count, total_sum = 0, 0

        for d in range(limit + 1):
            cnt_next, sum_next = dp(
                pos + 1,
                tight and (d == limit),
                leading and d == 0
            )
            total_count += cnt_next
            total_sum += sum_next + (0 if leading and d == 0 else d * cnt_next)

        return (total_count, total_sum)

    return dp(0, True, True)[1]

def sum_of_digits_range(L, R):
    return sum_of_digits_upto(R) - sum_of_digits_upto(L - 1)

# Example
L, R = map(int, input("Enter L R: ").split())
print(sum_of_digits_range(L, R))
```

C (Recursive Pair Return via struct)

```c
#include <stdio.h>
#include <string.h>

typedef struct { long long count, sum; } Pair;

int digits[20];
Pair memo[20][2][2];
int vis[20][2][2];
int len;

Pair dp(int pos, int tight, int leading) {
    if (pos == len) return (Pair){1, 0};
    if (vis[pos][tight][leading]) return memo[pos][tight][leading];
    vis[pos][tight][leading] = 1;

    int limit = tight ? digits[pos] : 9;
    long long total_count = 0, total_sum = 0;

    for (int d = 0; d <= limit; d++) {
        Pair next = dp(pos + 1, tight && (d == limit), leading && (d == 0));
        total_count += next.count;
        total_sum += next.sum + (leading && d == 0 ? 0 : (long long)d * next.count);
    }

    return memo[pos][tight][leading] = (Pair){total_count, total_sum};
}

long long solve(long long n) {
    if (n < 0) return 0;
    len = 0;
    while (n > 0) {
        digits[len++] = n % 10;
        n /= 10;
    }
    for (int i = 0; i < len / 2; i++) {
        int tmp = digits[i];
        digits[i] = digits[len - 1 - i];
        digits[len - 1 - i] = tmp;
    }
    memset(vis, 0, sizeof(vis));
    return dp(0, 1, 1).sum;
}

int main(void) {
    long long L, R;
    scanf("%lld %lld", &L, &R);
    printf("%lld\n", solve(R) - solve(L - 1));
}
```

#### Why It Matters

- Computes digit sums over huge ranges in logarithmic time
- Basis for many digit aggregation problems (count of 1's, digit sum mod M, etc.)
- Extensible to:

  * Counting even/odd digits
  * Weighted digit sums (like $d \times 10^{pos}$)
  * Property-based aggregation (e.g., sum of squares)

#### Step-by-Step Example

Compute sum of digits for all numbers ≤ 13:

| Number | Sum |
| ------ | --- |
| 0      | 0   |
| 1      | 1   |
| 2      | 2   |
| 3      | 3   |
| 4      | 4   |
| 5      | 5   |
| 6      | 6   |
| 7      | 7   |
| 8      | 8   |
| 9      | 9   |
| 10     | 1   |
| 11     | 2   |
| 12     | 3   |
| 13     | 4   |

Total = 55 ✅

DP builds this by digit:

- Tens digit → repeats 10 times
- Ones digit → contributes 0–9 repeatedly

#### A Gentle Proof (Why It Works)

Each position contributes its digit value multiplied by the number of combinations of remaining positions.
Digit DP captures this recursively:
If a digit `d` is fixed at position `pos`, every completion of later positions includes that digit once, hence `d * count_of_suffix`. Summing over all digits at all positions gives the total sum.

#### Try It Yourself

1. Count sum of even digits only.
2. Compute sum of digits mod M.
3. Compute sum of squared digits.
4. Count total number of digits written in range.
5. Compute weighted sum (like `d * 10^pos` contributions).

#### Test Cases

| Range  | Expected |
| ------ | -------- |
| 0–9    | 45       |
| 0–13   | 55       |
| 10–99  | 855      |
| 1–1000 | 13501    |

#### Complexity

- Time: $O(\text{len} \times 2 \times 2 \times 10)$
- Space: $O(\text{len} \times 2 \times 2)$

Digit DP can aggregate digit-level properties over massive intervals, this sum version is its canonical "count + accumulate" template.

### 454 Count with Mod Condition (Digit Sum mod M)

Count numbers in a range whose digit sum satisfies a modular condition. The standard pattern tracks the digit-sum modulo $M$ while respecting the upper bound.

#### What Problem Are We Solving?

Given integers $N,M,K$, count how many $x\in[0,N]$ satisfy:
$$
\big(\text{sum\_digits}(x)\big)\bmod M=K
$$
For a general range $[L,R]$, use $f(R)-f(L-1)$.

#### DP State

Let the decimal string of $N$ be $S$, length $m$.

State:
$$
dp[pos][mod][tight][leading]
$$

- $pos$: index in $S$ (0-based, left to right)
- $mod$: current value of digit-sum modulo $M$
- $tight\in{0,1}$: still equal to $S$ prefix or already below
- $leading\in{0,1}$: still placing only leading zeros

Goal: count completions with final $mod = K$.

Base:
$$
dp[m][mod][tight][leading] =
\begin{cases}
1, & mod = K,\\
0, & \text{otherwise.}
\end{cases}
$$

Transition (choose next digit $d$):

- $limit = S[pos]$ if $tight = 1$, else $9$
- Next states:
  - $next\_tight = tight \land (d = limit)$  
  - $next\_leading = leading \land (d = 0)$  

$$
  next\_mod =
    \begin{cases}
    mod, & \text{if } next\_leading = 1,\\
    (mod + d) \bmod M, & \text{otherwise.}
    \end{cases}
$$

Then
$$
dp[pos][mod][tight][leading]
= \sum_{d=0}^{limit}
dp[pos+1][next\_mod][next\_tight][next\_leading].
$$



#### Tiny Code (Easy Versions)

Python

```python
from functools import lru_cache

def count_mod_sum_upto(N, M, K):
    S = list(map(int, str(N)))
    m = len(S)

    @lru_cache(None)
    def dp(pos, mod, tight, leading):
        if pos == m:
            return 1 if mod == K else 0
        limit = S[pos] if tight else 9
        total = 0
        for d in range(limit + 1):
            ntight = tight and (d == limit)
            nleading = leading and (d == 0)
            nmod = mod if nleading else (mod + d) % M
            total += dp(pos + 1, nmod, ntight, nleading)
        return total

    return dp(0, 0, True, True)

def count_mod_sum_range(L, R, M, K):
    if L <= 0:
        return count_mod_sum_upto(R, M, K)
    return count_mod_sum_upto(R, M, K) - count_mod_sum_upto(L - 1, M, K)

# Example:
# print(count_mod_sum_range(0, 327, 7, 3))
```

C

```c
#include <stdio.h>
#include <string.h>

long long memo[20][200][2][2];
char vis[20][200][2][2];
int digits[20], mlen, M, K;

long long solve_dp(int pos, int mod, int tight, int leading) {
    if (pos == mlen) return mod == K;
    if (vis[pos][mod][tight][leading]) return memo[pos][mod][tight][leading];
    vis[pos][mod][tight][leading] = 1;

    int limit = tight ? digits[pos] : 9;
    long long total = 0;
    for (int d = 0; d <= limit; d++) {
        int ntight = tight && (d == limit);
        int nleading = leading && (d == 0);
        int nmod = nleading ? mod : (mod + d) % M;
        total += solve_dp(pos + 1, nmod, ntight, nleading);
    }
    return memo[pos][mod][tight][leading] = total;
}

long long count_upto(long long N, int m, int k) {
    if (N < 0) return 0;
    M = m; K = k;
    int tmp[20], len = 0;
    while (N > 0) { tmp[len++] = (int)(N % 10); N /= 10; }
    if (len == 0) tmp[len++] = 0;
    for (int i = 0; i < len; i++) digits[i] = tmp[len - 1 - i];
    mlen = len;
    memset(vis, 0, sizeof(vis));
    return solve_dp(0, 0, 1, 1);
}

long long count_range(long long L, long long R, int m, int k) {
    return count_upto(R, m, k) - count_upto(L - 1, m, k);
}

// Example usage in main:
// int main(){ long long L=0,R=327; int M=7,K=3; printf("%lld\n", count_range(L,R,M,K)); }
```

Notes:

- The array bound `200` for `mod` assumes $M\le 200$. Increase if needed.

#### Why It Matters

- Core Digit DP for number-theory style constraints
- Handles many variants by changing the carried statistic to modulo form
- Building block for problems like:

  * Count with digit sum equal to $S$ (set $M$ large enough and target $K=S$ with careful state)
  * Count with digit sum in a set (sum over multiple $K$)
  * Multi-condition states (e.g., digit sum mod $M$ and parity)

#### Step-by-Step Example

Count $x\in[0,99]$ with digit sum mod $3$ equal to $0$.

- The DP carries $mod\in{0,1,2}$.
- At each position, branch over $d\in[0..9]$ with tight until you pass the bound 99.
- The answer is $34$.

(This small case can also be verified by combinatorics: roughly one third of two-digit-with-leading-zero numbers.)

#### A Gentle Proof (Why It Works)

Every number corresponds to one path of digit choices.
The state $(pos,mod,tight,leading)$ uniquely captures all information that influences future feasibility and the final condition $mod=K$.
Since the transition only depends on the current state and chosen digit, memoizing these states yields a complete and non-overlapping partition of the search space.

#### Try It Yourself

1. Count numbers with digit sum mod $M$ in a set $S$ by summing answers over $K\in S$.
2. Count numbers with digit sum exactly $S$ by replacing $mod$ with a bounded sum state and capping at $S$.
3. Combine with no adjacent duplicates by adding a `prev` digit to the state.
4. Compute numbers with sum of squares of digits mod $M$ equal to $K$.
5. Extend to base-$B$ by changing the limit from 9 to $B-1$.

#### Test Cases

| $N$ | $M$ | $K$ | Expected idea       |
| --- | --- | --- | ------------------- |
| 9   | 3   | 0   | 4 numbers (0,3,6,9) |
| 20  | 2   | 0   | about half of 0..20 |
| 99  | 3   | 0   | 34                  |
| 327 | 7   | 3   | computed via code   |

#### Complexity

- Time: $O(len\cdot M\cdot 2\cdot 2\cdot 10)$
- Space: $O(len\cdot M\cdot 2\cdot 2)$

This is the standard mod-carry Digit DP: thread the property through the digits modulo $M$, respect the bound with `tight`, and account for leading zeros cleanly.

### 455 Count of Increasing Digits

We want to count all integers in a range that have strictly increasing digits, every digit is larger than the one before it. For example, `123`, `149`, and `7` qualify, but `133`, `321`, or `224` do not.

#### What Problem Are We Solving?

Given an upper bound $N$, count integers $x$ in $[0, N]$ such that:

$$
\text{digits}(x) = [d_0, d_1, \dots, d_k] \implies d_0 < d_1 < \dots < d_k
$$

Example: For $N=500$, valid numbers include `1, 2, …, 9, 12, 13, …, 49, 123, 134, …, 489`, etc.

We can model this with Digit DP tracking the previous digit to ensure the increasing condition.

#### DP State

Let $S$ = list of digits of $N$.

State:
$$
dp[pos][prev][tight][leading]
$$

Where:

- `pos`: index of current digit (0-based)
- `prev`: last chosen digit (0–9, or 10 for "none yet")
- `tight`: whether we're still prefix-equal to $N$
- `leading`: whether we've only placed leading zeros (so far no real digits)

#### Transition

At each position:

- Determine `limit = S[pos]` if `tight=1`, else 9.
- Loop `d` from 0 to `limit`.
- Skip `d <= prev` if not `leading` (must strictly increase).
- Update:

  * `next_tight = tight and (d == limit)`
  * `next_leading = leading and (d == 0)`
  * `next_prev = prev if next_leading else d`

Sum results of recursive calls.

Base:
$$
dp[len][prev][tight][leading] = 1
$$
since one valid number is formed.

#### How Does It Work (Plain Language)

We build the number one digit at a time:

- If we've started (not leading), each new digit must be greater than the previous one.
- If we're still leading, any zero is fine.
- `tight` ensures we never exceed $N$'s prefix.
  By exploring all possible digits under these rules, we count every strictly increasing number ≤ $N$.

#### Tiny Code (Easy Versions)

Python

```python
from functools import lru_cache

def count_increasing_digits(N):
    S = list(map(int, str(N)))
    m = len(S)

    @lru_cache(None)
    def dp(pos, prev, tight, leading):
        if pos == m:
            return 1  # reached end, valid number
        limit = S[pos] if tight else 9
        total = 0
        for d in range(limit + 1):
            if not leading and d <= prev:
                continue  # must strictly increase
            ntight = tight and (d == limit)
            nleading = leading and (d == 0)
            nprev = prev if nleading else d
            total += dp(pos + 1, nprev, ntight, nleading)
        return total

    return dp(0, 10, True, True)

# Example
N = int(input("Enter N: "))
print(count_increasing_digits(N))
```

C

```c
#include <stdio.h>
#include <string.h>

long long memo[20][11][2][2];
char vis[20][11][2][2];
int digits[20], len;

long long dp(int pos, int prev, int tight, int leading) {
    if (pos == len) return 1;
    if (vis[pos][prev][tight][leading]) return memo[pos][prev][tight][leading];
    vis[pos][prev][tight][leading] = 1;

    int limit = tight ? digits[pos] : 9;
    long long res = 0;
    for (int d = 0; d <= limit; d++) {
        if (!leading && d <= prev) continue;
        int ntight = tight && (d == limit);
        int nleading = leading && (d == 0);
        int nprev = nleading ? prev : d;
        res += dp(pos + 1, nprev, ntight, nleading);
    }

    return memo[pos][prev][tight][leading] = res;
}

long long solve(long long n) {
    if (n < 0) return 0;
    len = 0;
    while (n > 0) {
        digits[len++] = n % 10;
        n /= 10;
    }
    if (len == 0) digits[len++] = 0;
    for (int i = 0; i < len / 2; i++) {
        int t = digits[i];
        digits[i] = digits[len - 1 - i];
        digits[len - 1 - i] = t;
    }
    memset(vis, 0, sizeof(vis));
    return dp(0, 10, 1, 1);
}

int main(void) {
    long long N;
    scanf("%lld", &N);
    printf("%lld\n", solve(N));
}
```

#### Why It Matters

- Builds understanding of monotonic digit constraints
- Template for counting:

  * Strictly increasing digits
  * Strictly decreasing digits
  * Non-decreasing digits (just change condition to `d < prev`)
- Appears in combinatorial enumeration and digit ordering problems

#### Step-by-Step Example

For $N=130$:

Valid numbers include:

```
0–9
12, 13
23
...
123
```

Invalid examples:

- `11` (equal digits)
- `21` (decreasing)

DP automatically filters these based on the `d > prev` rule.

#### A Gentle Proof (Why It Works)

Each path corresponds to a unique number ≤ $N$.
The rule `d > prev` enforces strictly increasing order.
The DP ensures no overcounting because each prefix `(pos, prev, tight, leading)` fully determines future choices.

#### Try It Yourself

1. Modify to count strictly decreasing numbers (`d < prev`).
2. Count non-decreasing numbers (`d >= prev`).
3. Enforce exact length $k$ via a `len_used` parameter.
4. Add mod conditions (sum mod M).
5. Compute sum of all increasing numbers instead of count.

#### Test Cases

| N   | Expected | Notes                            |
| --- | -------- | -------------------------------- |
| 9   | 10       | 0–9 valid                        |
| 12  | 12       | 10 numbers 0–9, plus 12 and 13   |
| 99  | 45       | All 1–2-digit increasing numbers |
| 321 | 84       | Derived by DP                    |

#### Complexity

- Time: $O(len \times 11 \times 2 \times 2 \times 10)$
- Space: $O(len \times 11 \times 2 \times 2)$

Digit DP with monotonic digit constraints transforms ordering problems into state-space counting, a fundamental technique for combinatorial digit analysis.

### 456 Count with Forbidden Digits

Count how many integers in a range avoid a given set of forbidden digits. This is a basic Digit DP where the state remembers whether we are still tight to the upper bound and whether we have only placed leading zeros.

#### What Problem Are We Solving?

Given $N$ and a set $F\subseteq{0,1,\dots,9}$ of forbidden digits, count integers $x\in[0,N]$ whose standard decimal representation contains no digit from $F$.

Convention: leading zeros are allowed during the DP but do not count as real digits, so a leading zero is always permitted even if $0\in F$.

For a general range $[L,R]$ return $f(R)-f(L-1)$.

#### DP State

Let $S$ be the digit list of $N$, length $m$.

State:
$$
dp[pos][tight][leading]
$$

- $pos$: index in $[0,m)$
- $tight\in{0,1}$: 1 if the prefix equals $N$ so far
- $leading\in{0,1}$: 1 if all chosen digits are leading zeros

#### Transition

At position $pos$ choose digit $d\in[0,\text{limit}]$, where $\text{limit}=S[pos]$ if $tight=1$ else $9$.

Reject $d$ if it is a real digit that is forbidden:

- If $leading=1$ and $d=0$, accept regardless of $F$.
- Otherwise require $d\notin F$.

Next state:

- $next_tight = tight\land(d=\text{limit})$
- $next_leading = leading\land(d=0)$

Recurrence:
$$
dp[pos][tight][leading] = \sum_{d=0}^{\text{limit}} \mathbf{1}\big(\text{allowed}(d,leading)\big)\cdot dp[pos+1][next_tight][next_leading]
$$

Base:
$$
dp[m][tight][leading]=1
$$

Answer is $dp[0][1][1]$.

#### How Does It Work (Plain Language)

We build the number left to right. If we have not surpassed $N$ yet, the next digit is restricted by $N$ at that position. Leading zeros are virtual padding and do not trigger the forbidden check. The DP counts all valid completions from each prefix.

#### Tiny Code (Easy Versions)

Python

```python
from functools import lru_cache

def count_without_forbidden(N, forbidden):
    S = list(map(int, str(N)))
    m = len(S)
    F = set(forbidden)

    @lru_cache(None)
    def dp(pos, tight, leading):
        if pos == m:
            return 1
        limit = S[pos] if tight else 9
        total = 0
        for d in range(limit + 1):
            ntight = tight and (d == limit)
            nleading = leading and (d == 0)
            # allow leading zero regardless of F
            if not nleading and d in F:
                continue
            total += dp(pos + 1, ntight, nleading)
        return total

    return dp(0, True, True)

def count_range_without_forbidden(L, R, forbidden):
    if L <= 0:
        return count_without_forbidden(R, forbidden)
    return count_without_forbidden(R, forbidden) - count_without_forbidden(L - 1, forbidden)

# Example:
# print(count_range_without_forbidden(0, 327, {3,4}))
```

C

```c
#include <stdio.h>
#include <string.h>

long long memo[20][2][2];
char vis[20][2][2];
int digits[20], len;
int forbid[10];

long long dp(int pos, int tight, int leading) {
    if (pos == len) return 1;
    if (vis[pos][tight][leading]) return memo[pos][tight][leading];
    vis[pos][tight][leading] = 1;

    int limit = tight ? digits[pos] : 9;
    long long total = 0;

    for (int d = 0; d <= limit; d++) {
        int ntight = tight && (d == limit);
        int nleading = leading && (d == 0);
        if (!nleading && forbid[d]) continue; // real digit must not be forbidden
        total += dp(pos + 1, ntight, nleading);
    }
    return memo[pos][tight][leading] = total;
}

long long solve_upto(long long N) {
    if (N < 0) return 0;
    len = 0;
    if (N == 0) digits[len++] = 0;
    while (N > 0) { digits[len++] = (int)(N % 10); N /= 10; }
    for (int i = 0; i < len/2; i++) {
        int t = digits[i]; digits[i] = digits[len-1-i]; digits[len-1-i] = t;
    }
    memset(vis, 0, sizeof(vis));
    return dp(0, 1, 1);
}

// Example main
// int main(void){
//     long long L,R; int k,x;
//     scanf("%lld %lld %d",&L,&R,&k);
//     memset(forbid,0,sizeof(forbid));
//     for(int i=0;i<k;i++){ scanf("%d",&x); forbid[x]=1; }
//     long long ans = solve_upto(R) - solve_upto(L-1);
//     printf("%lld\n", ans);
// }
```

#### Why It Matters

- Canonical Digit DP that filters digits by a local constraint
- Models problems with digit blacklists, keypad rules, license formats, or numeral-system restrictions
- Serves as a base to combine with additional states like digit sum or adjacency constraints

#### Step-by-Step Example

Let $F={3,4}$ and $N=327$.

- At each position, digits ${3,4}$ are disallowed unless we are still in leading zeros.
- The DP explores all prefixes bounded by $327$ and sums valid completions.
- Use the Python snippet to compute the exact count.

#### A Gentle Proof (Why It Works)

Every number in $[0,N]$ corresponds to a unique path of digit choices. The predicate allowed$(d,leading)$ ensures that once a real digit is placed, it is not forbidden. The pair $(pos,tight)$ ensures we do not exceed $N$. Since subproblems depend only on these three parameters, memoization counts each equivalence class of prefixes exactly once.

#### Try It Yourself

1. Forbid multiple digits, e.g. $F={1,3,7}$.
2. Forbid a set that includes $0$ and verify that leading zeros still pass.
3. Combine with a sum modulo condition by adding a $mod$ state.
4. Combine with no adjacent duplicates by adding a $prev$ state.
5. Switch to base $B$ by changing the limit from $9$ to $B-1$.

#### Test Cases

| $N$  | $F$     | Expected idea                                                  |
| ---- | ------- | -------------------------------------------------------------- |
| 99   | ${9}$   | Count of two-digit-with-leading-zero numbers using digits 0..8 |
| 327  | ${3,4}$ | Computed by code                                               |
| 1000 | ${1}$   | All numbers without digit 1                                    |
| 0    | any $F$ | 1 (the number 0)                                               |

#### Complexity

- Time: $O(len\cdot 2\cdot 2\cdot 10)$
- Space: $O(len\cdot 2\cdot 2)$

This pattern is the simplest Digit DP guard: screen digits against a blacklist while handling bounds and leading zeros correctly.

### 457 SOS DP Subset Sum

Sum Over Subsets (SOS) DP is a powerful bitmask technique used to precompute values over all subsets of a given mask. One of its core applications is the Subset Sum over bitmasks, efficiently computing $f(S) = \sum_{T \subseteq S} g(T)$ for all $S$.

#### What Problem Are We Solving?

Given an array `g` of size $2^n$ indexed by bitmasks, compute a new array `f` such that:

$$
f[S] = \sum_{T \subseteq S} g[T]
$$

A naive approach iterates through all subsets for each $S$, which takes $O(3^n)$. SOS DP reduces this to $O(n \cdot 2^n)$, making it feasible for $n \le 20$.

#### How Does It Work (Plain Language)

We treat each bit as a dimension. For each bit position $i$ (from 0 to $n-1$):

- For each mask $S$:

  * If bit $i$ is set in $S$, add contribution from $S$ with bit $i$ cleared.

This accumulates sums over all subsets, one bit at a time.

#### Recurrence

Let `f` initially equal `g`. Then:

$$
\begin{aligned}
&\text{for } i = 0,\dots,n-1:\\
&\quad \text{for } S = 0,\dots,\ \texttt{(1<<n)}-1:\\
&\quad\quad \text{if } \bigl(S \mathbin{\&} \texttt{(1<<i)}\bigr) \ne 0:\quad
f[S] \mathrel{+}= f\!\left[S^{\text{without } i}\right]
\end{aligned}
$$

where $S^{\text{without } i} = S \oplus \texttt{(1<<i)}$ removes bit $i$.

After processing all bits, `f[S]` holds the sum over all subsets of $S$.


#### Tiny Code (Easy Versions)

Python

```python
def sos_subset_sum(g, n):
    f = g[:]  # copy
    for i in range(n):
        for S in range(1 << n):
            if S & (1 << i):
                f[S] += f[S ^ (1 << i)]
    return f

# Example
n = 3
g = [1,2,3,4,5,6,7,8]  # g[mask]
f = sos_subset_sum(g, n)
print(f)
```

C

```c
#include <stdio.h>

void sos_subset_sum(long long f[], int n) {
    for (int i = 0; i < n; i++) {
        for (int S = 0; S < (1 << n); S++) {
            if (S & (1 << i)) {
                f[S] += f[S ^ (1 << i)];
            }
        }
    }
}

int main() {
    int n = 3;
    long long f[1 << 3] = {1,2,3,4,5,6,7,8};
    sos_subset_sum(f, n);
    for (int i = 0; i < (1 << n); i++) printf("%lld ", f[i]);
    return 0;
}
```

#### Why It Matters

- Foundation for bitmask DP transforms (e.g. subset convolution, inclusion-exclusion).
- Enables fast enumeration of subset properties (sums, counts, etc.).
- Reusable building block in probabilistic DP, polynomial transforms, and game DP.

#### Step-by-Step Example

Let $n=2$, masks $00,01,10,11$ and $g=[1,2,3,4]$:

Initialize $f=g$.

1. Bit $i=0$:

   * $S=01$: $f[01]+=f[00] \implies 2+1=3$
   * $S=11$: $f[11]+=f[10] \implies 4+3=7$

$f=[1,3,3,7]$

2. Bit $i=1$:

   * $S=10$: $f[10]+=f[00] \implies 3+1=4$
   * $S=11$: $f[11]+=f[01] \implies 7+3=10$

Final $f=[1,3,4,10]$

Check:

- $f[11] = g[00]+g[01]+g[10]+g[11] = 1+2+3+4=10$ ✓

#### A Gentle Proof (Why It Works)

Each bit is processed independently.
At iteration $i$, each mask $S$ accumulates contributions from all subsets differing only at bit $i$.
After processing all bits, every subset $T\subseteq S$ is visited exactly once.

By induction:

- Base: $i=0$, $f[S]$ contains $g[S]$.
- Step: adding $f[S^{\text{without }i}]$ ensures inclusion of subsets missing bit $i$.

Thus, after $n$ passes, $f[S]$ sums over all subsets.

#### Try It Yourself

1. Change sum to product (if $f[S]*=f[S^{\text{without }i}]$).
2. Compute $f[S]=\sum_{T\supseteq S} g[T]$ (see Superset DP).
3. Combine SOS DP with inclusion-exclusion to count valid subsets.
4. Apply to subset convolution problems.
5. Use modulo arithmetic for large values.

#### Test Cases

| n | g (input)         | f (output)        |
| - | ----------------- | ----------------- |
| 2 | [1,2,3,4]         | [1,3,4,10]        |
| 3 | [1,1,1,1,1,1,1,1] | [1,2,2,4,2,4,4,8] |
| 3 | [0,1,2,3,4,5,6,7] | computed by code  |

#### Complexity

- Time: $O(n \cdot 2^n)$
- Space: $O(2^n)$

SOS DP transforms the exponential subset-sum enumeration into a structured linear pass across dimensions, making subset-based computation tractable for small $n$.

### 458 SOS DP Superset Sum

Sum Over Supersets computes for every mask the aggregate over all of its supersets. It complements the usual SOS DP over subsets.

#### What Problem Are We Solving?

Given an array `g` of size $2^n$ indexed by bitmasks, compute `h` such that
$$
h[S]=\sum_{T\supseteq S}g[T].
$$
Naively this is $O(3^n)$. With SOS Superset DP it is $O(n\cdot 2^n)$.

#### How Does It Work (Plain Language)

Process bits one by one. For each bit $i$, if a mask $S$ has bit $i$ unset, then every superset that turns this bit on is of the form $S\cup{i}=S\oplus(1<<i)$. So we can accumulate from that neighbor upward.

#### Recurrence

Initialize $h = g$. For each bit $i = 0..n-1$:

For every mask $S$:

If $(S \mathbin{\&} (1 << i)) == 0$, then
    $$
    h[S] += h[S \oplus (1 << i)].
    $$
    After all bits, $h[S]$ equals the sum over all supersets of $S$.


#### Tiny Code

Python

```python
def sos_superset_sum(g, n):
    h = g[:]  # copy
    for i in range(n):
        for S in range(1 << n):
            if (S & (1 << i)) == 0:
                h[S] += h[S | (1 << i)]
    return h

# Example
n = 3
g = [1,2,3,4,5,6,7,8]  # g[mask]
h = sos_superset_sum(g, n)
print(h)
```

C

```c
#include <stdio.h>

void sos_superset_sum(long long h[], int n){
    for(int i=0;i<n;i++){
        for(int S=0;S<(1<<n);S++){
            if((S&(1<<i))==0){
                h[S]+=h[S|(1<<i)];
            }
        }
    }
}

int main(){
    int n=3;
    long long h[1<<3]={1,2,3,4,5,6,7,8};
    sos_superset_sum(h,n);
    for(int i=0;i<(1<<n);i++) printf("%lld ", h[i]);
    return 0;
}
```

#### Why It Matters

- Dual of subset SOS DP.
- Core for transforms like zeta and Möbius on the subset lattice.
- Useful for queries like: for each feature set $S$, aggregate values over all supersets that contain $S$.

#### Step-by-Step Example

Let $n=2$, masks $00,01,10,11$ and $g=[1,2,3,4]$.

Process bit $i=0$:

- $S=00$: $h[00]+=h[01]\Rightarrow 1+2=3$
- $S=10$: $h[10]+=h[11]\Rightarrow 3+4=7$

Now $h=[3,2,7,4]$.

Process bit $i=1$:

- $S=00$: $h[00]+=h[10]\Rightarrow 3+7=10$
- $S=01$: $h[01]+=h[11]\Rightarrow 2+4=6$

Final $h=[10,6,7,4]$, which matches

- $h[00]=g[00]+g[01]+g[10]+g[11]=10$
- $h[01]=g[01]+g[11]=6$
- $h[10]=g[10]+g[11]=7$
- $h[11]=g[11]=4$.

#### A Gentle Proof (Why It Works)

Fix a bit order. When processing bit $i$, for any $S$ with bit $i$ unset, every superset of $S$ either keeps bit $i$ off or turns it on. Before processing $i$, $h[S]$ accumulates supersets with bit $i$ off. Adding $h[S\cup{i}]$ brings in all supersets with bit $i$ on. Induct over bits to conclude all supersets are included exactly once.

#### Try It Yourself

1. Convert this to compute maximum over supersets by replacing plus with max.
2. Combine with subset SOS to precompute both directions for fast subset-superset queries.
3. Apply modulo arithmetic to prevent overflow.
4. Implement the Möbius inversion on supersets to invert the transform.
5. Extend to bitwise operations where aggregation depends on bit counts.

#### Test Cases

| n | g input           | h output          |
| - | ----------------- | ----------------- |
| 2 | [1,2,3,4]         | [10,6,7,4]        |
| 3 | all 1s            | [8,4,4,2,4,2,2,1] |
| 3 | [0,1,2,3,4,5,6,7] | computed by code  |

#### Complexity

- Time: $O(n\cdot 2^n)$
- Space: $O(2^n)$

SOS Superset DP is the natural mirror of subset SOS. Use it whenever queries demand aggregating over all sets that contain a given mask.

### 459 XOR Basis DP

The XOR Basis DP technique helps count or generate all possible XOR values from a set of numbers efficiently. It constructs a linear basis over GF(2) and enables solving problems like counting distinct XORs, finding minimum/maximum XOR, and combining with digit DP or bitmask states.

#### What Problem Are We Solving?

Given a list of numbers $A = [a_1, a_2, \dots, a_n]$, we want to:

- Find how many distinct XOR values can be formed from subsets of $A$.
- Or find maximum/minimum possible XOR.
- Or answer queries on possible XOR combinations.

The XOR operation forms a vector space over $\mathbb{F}_2$, and each number contributes a vector. The XOR basis provides a compact representation of all subset XORs.

#### Core Idea

Maintain an array `basis` representing independent bit vectors. Insert each number into the basis (Gaussian elimination over GF(2)):

- For each bit from high to low, if that bit is set and not represented, store the number.
- If it is already represented, XOR with the current basis vector to reduce it.

At the end, the number of independent vectors is the rank $r$, and the number of distinct XORs is $2^r$.

#### DP Perspective

The state represents a basis built from a prefix of the array.
You can define:

$$
dp[i] = \text{XOR basis after processing first } i \text{ elements}
$$

To count distinct XORs after all elements:

$$
\text{count} = 2^{\text{rank}}
$$

If you need to build combinations (e.g. count of XOR < M), combine basis construction with digit DP constraints.

#### Tiny Code (Easy Versions)

Python

```python
def xor_basis(arr):
    basis = []
    for x in arr:
        for b in basis:
            x = min(x, x ^ b)
        if x:
            basis.append(x)
    return basis

def count_distinct_xors(arr):
    basis = xor_basis(arr)
    return 1 << len(basis)  # 2^rank

# Example
A = [3, 10, 5]
basis = xor_basis(A)
print("Basis:", basis)
print("Distinct XORs:", count_distinct_xors(A))
```

C

```c
#include <stdio.h>

int insert_basis(int basis[], int *sz, int x) {
    for (int i = 0; i < *sz; i++) {
        if ((x ^ basis[i]) < x) x ^= basis[i];
    }
    if (x == 0) return 0;
    basis[(*sz)++] = x;
    return 1;
}

int main() {
    int arr[] = {3, 10, 5};
    int n = 3, basis[32], sz = 0;

    for (int i = 0; i < n; i++)
        insert_basis(basis, &sz, arr[i]);

    printf("Rank: %d\nDistinct XORs: %d\n", sz, 1 << sz);
}
```

#### Why It Matters

- Forms the foundation for subset XOR problems.
- Used in:

  * Counting distinct XORs
  * Maximum XOR subset
  * XOR-constrained digit DP
  * Gaussian elimination in $\mathbb{F}_2$

It's the bitwise analog of linear algebra, solving over GF(2).

#### Step-by-Step Example

Let $A = [3, 10, 5]$:

- Binary: $3=011$, $10=1010$, $5=0101$
- Insert 3 → basis = {3}
- Insert 10 → independent, basis = {3,10}
- Insert 5 → can be reduced: $5⊕3=6$, $6⊕10=12$ → independent, basis = {3,10,5}

Rank $r=3$, number of distinct XORs = $2^3=8$.

All subset XORs:

```
0, 3, 5, 6, 10, 11, 12, 15
```

#### A Gentle Proof (Why It Works)

Each basis vector represents a new independent bit dimension.
Every subset XOR corresponds to a linear combination over $\mathbb{F}_2$ of the basis.
If there are $r$ independent vectors, there are $2^r$ possible linear combinations (subsets), hence $2^r$ distinct XORs.

#### Try It Yourself

1. Modify to find maximum XOR subset (XOR greedily from MSB down).
2. Combine with digit DP to count numbers with XOR constraints ($< M$).
3. Track reconstruction: which subset forms a target XOR.
4. Apply to path XOR queries in trees (via prefix basis merging).
5. Extend to multiset bases or online updates.

#### Test Cases

| Input    | Output | Notes                 |
| -------- | ------ | --------------------- |
| [3,10,5] | 8      | 3 independent vectors |
| [1,2,3]  | 4      | Rank = 2              |
| [1,1,1]  | 2      | Rank = 1              |

#### Complexity

- Time: $O(n \cdot \text{bitwidth})$
- Space: $O(\text{bitwidth})$

XOR Basis DP is the digital geometry of sets under XOR, every number a vector, every subset a linear combination, every question a path through binary space.

### 460 Digit DP for Palindromes

Digit DP for Palindromes counts all numbers within a given range that are palindromic, numbers that read the same forward and backward. It's a symmetric DP that constructs digits from both ends simultaneously, respecting tight bounds from the original number.

#### What Problem Are We Solving?

Given two integers $L$ and $R$, count the palindromes in $[L, R]$.

Example: in $[1, 200]$, palindromes are
$1,2,\dots,9,11,22,\dots,99,101,111,\dots,191$, total 28.

Naively iterating and checking each number is $O(N)$, which is too slow for large $N$.
We want an $O(d \times 10^{d/2})$ approach using Digit DP with symmetry.

#### Core Idea

We can build a number digit by digit from the outside in, ensuring the number remains a palindrome at each step.

For a given length $len$, we only need to choose digits for the first half; the second half is determined.

The tight constraints ensure we stay $\le R$ (or $\le L-1$ for inclusive ranges).

#### DP Definition

Let $S$ = digits of $N$, length $len$.

State:
$$
dp[pos][tight][leading]
$$

Where:

- `pos` is the current index from the left half ($0 \le pos < \frac{len}{2}$)
- `tight` means prefix equals $N$ so far
- `leading` means we've placed only leading zeros

The recursion places one digit at `pos`, mirrors it at `len-1-pos`, and recurses inward.

#### Transition

For each `digit` in $[0, limit]$:

- If `leading` and `digit==0`, we can skip counting it as a real digit.
- Mirror digit into the symmetric position.
- Update tightness if `digit == limit`.
- Recurse inward until halfway.

When reaching middle, count 1 valid palindrome.

#### Algorithm Outline

To count palindromes ≤ $N$:

1. Convert $N$ to string `S`
2. Run `dfs(pos=0, tight=True, leading=True)`
3. If building full palindrome (not half-only), check mirrored structure

Count in $[L, R]$ as:
$$
f(R) - f(L-1)
$$

#### Tiny Code (Easy Versions)

Python

```python
from functools import lru_cache

def count_palindromes_upto(N):
    S = list(map(int, str(N)))
    n = len(S)

    @lru_cache(None)
    def dfs(pos, tight, leading, half):
        if pos == (n + 1) // 2:
            return 1  # one palindrome formed
        limit = S[pos] if tight else 9
        total = 0
        for d in range(limit + 1):
            if leading and d == 0:
                total += dfs(pos + 1, tight and d == limit, True, half + [0])
            else:
                total += dfs(pos + 1, tight and d == limit, False, half + [d])
        return total

    return dfs(0, True, True, ())

def count_palindromes(L, R):
    return count_palindromes_upto(R) - count_palindromes_upto(L - 1)

# Example
print(count_palindromes(1, 200))
```

C (half-construction)

```c
#include <stdio.h>
#include <string.h>

long long count_palindromes_upto(long long N) {
    if (N < 0) return 0;
    char s[20];
    sprintf(s, "%lld", N);
    int len = strlen(s);
    int half = (len + 1) / 2;
    long long count = 0;

    // Build prefix half and mirror
    for (int mask = 0; mask < (1 << (half * 4)); mask++) {
        // Conceptual only, use recursion or base-10 enumeration for actual code
    }

    // Simpler approach: iterate half, build full palindrome, check ≤ N
    long long start = 1;
    for (int i = 1; i <= len; i++) {
        int half_len = (i + 1) / 2;
        long long base = 1;
        for (int j = 1; j < half_len; j++) base *= 10;
        for (long long x = base; x < base * 10; x++) {
            long long y = x;
            if (i % 2) y /= 10;
            long long z = x;
            while (y > 0) {
                z = z * 10 + (y % 10);
                y /= 10;
            }
            if (z <= N) count++;
        }
    }
    return count;
}

int main(void) {
    long long L = 1, R = 200;
    printf("%lld\n", count_palindromes_upto(R) - count_palindromes_upto(L - 1));
}
```

#### Why It Matters

- Palindromic counting appears in:

  * Digit constraints (e.g. special number sets)
  * Symmetric number combinatorics
  * Patterned sequence generation
- Builds intuition for bidirectional DP where digits mirror.

#### Step-by-Step Example

Let $N = 200$:

- Length 3 → half = 2
- Choose first half (00–19), mirror → `00–99` → `0,11,22,...,191`
- Apply bounds: only ≤200 → count = 28

#### A Gentle Proof (Why It Works)

Every palindrome is uniquely determined by its first half.
For each valid half respecting the upper bound, exactly one mirrored number exists.
Tightness ensures no overflow beyond $N$.
Leading-zero handling ensures numbers like `00100` are excluded.

#### Try It Yourself

1. Count even-length palindromes only.
2. Count palindromes with a fixed digit sum.
3. Modify to count palindromic primes by adding primality check.
4. Combine with digit constraints (e.g. no 3s).
5. Count numbers that become palindromes after reversal operations.

#### Test Cases

| L   | R   | Output | Notes                    |
| --- | --- | ------ | ------------------------ |
| 1   | 9   | 9      | Single-digit palindromes |
| 10  | 99  | 9      | 11,22,...,99             |
| 1   | 200 | 28     | Up to 191                |
| 100 | 999 | 90     | All 3-digit palindromes  |

#### Complexity

- Time: $O(d \times 10^{d/2})$
- Space: $O(d)$

Digit DP for Palindromes bridges arithmetic and symmetry, constructing mirrored structures digit by digit under bound constraints.

# Section 47. DP Optimizations 

### 461 Divide & Conquer DP (Monotone Optimization)

When a DP transition is a 1D convolution of the form
$dp[i][j]=\min\limits_{k<j}\big(dp[i-1][k]+C(k,j)\big)$,
and the argmin is monotone ($opt[i][j]\le opt[i][j+1]$), you can compute a whole layer $dp[i][*]$ in $O(n)$ splits with divide and conquer instead of $O(n^2)$. Typical total is $O(K,n\log n)$ or $O(K,n)$ depending on implementation.

#### What Problem Are We Solving?

Speed up DP layers with transitions
$$
dp[j]=\min_{k<j}\big(prev[k]+C(k,j)\big),
$$
for $j$ in an interval, where the optimal index $opt[j]\in[\text{optL},\text{optR}]$ and satisfies
$$
opt[j]\le opt[j+1]\quad\text{(monotone decision property).}
$$

This structure appears in:

- $K$-partitioning of arrays with convex segment cost
- 1D facility placement and line breaking with convex penalties
- Some shortest path on DAG layers with convex arc costs

#### How Does It Work (Plain Language)

Compute the current DP layer on a segment $[L,R]$ by solving the midpoint $M$, searching its best $k$ only in $[optL,optR]$. The best index $opt[M]$ splits the problem:

- Left half $[L,M-1]$ only needs candidates in $[optL,opt[M]]$
- Right half $[M+1,R]$ only needs $[opt[M],optR]$

Recursively repeat until intervals are size 1. Monotonicity guarantees these candidate ranges.

#### Preconditions checklist

You can use divide and conquer DP if:

1. Transition is $dp[j]=\min_{k<j}(prev[k]+C(k,j))$.
2. The optimal index is monotone in $j$. A sufficient condition is quadrangle inequality or Monge property of $C$:
   $$
   C(a,c)+C(b,d)\le C(a,d)+C(b,c)\quad\text{for }a\le b\le c\le d.
   $$

#### Tiny Code (Template)

C++ style pseudocode (drop in C with minor edits)

```cpp
// Compute one layer dp_cur[lo..hi], given dp_prev and cost C(k,j).
// Assumes optimal indices are monotone.
void compute(int lo, int hi, int optL, int optR,
             const vector<long long>& dp_prev,
             vector<long long>& dp_cur,
             auto&& cost) {
    if (lo > hi) return;
    int mid = (lo + hi) >> 1;

    long long best = LLONG_MAX;
    int best_k = -1;
    int start = optL, end = min(optR, mid - 1);
    for (int k = start; k <= end; ++k) {
        long long val = dp_prev[k] + cost(k, mid);
        if (val < best) {
            best = val;
            best_k = k;
        }
    }
    dp_cur[mid] = best;

    // Recurse with narrowed opt ranges
    compute(lo, mid - 1, optL, best_k, dp_prev, dp_cur, cost);
    compute(mid + 1, hi, best_k, optR, dp_prev, dp_cur, cost);
}
```

Python (clear and compact)

```python
INF = 1018

def compute(lo, hi, optL, optR, dp_prev, dp_cur, cost):
    if lo > hi:
        return
    mid = (lo + hi) // 2
    best_val, best_k = INF, -1
    end = min(optR, mid - 1)
    for k in range(optL, end + 1):
        v = dp_prev[k] + cost(k, mid)
        if v < best_val:
            best_val, best_k = v, k
    dp_cur[mid] = best_val
    compute(lo, mid - 1, optL, best_k, dp_prev, dp_cur, cost)
    compute(mid + 1, hi, best_k, optR, dp_prev, dp_cur, cost)
```

How to use
For $i=1..K$: call `compute(1,n,0,n-1, dp_prev, dp_cur, cost)` then swap layers. Index ranges depend on your base cases.

#### Example: K partitions with convex segment cost

Given array $a[1..n]$, let prefix sums $S[j]=\sum_{t=1}^j a[t]$. Suppose segment cost is
$$
C(k,j)=\big(S[j]-S[k]\big)^2,
$$
which is convex and satisfies quadrangle inequality. The DP
$$
dp[i][j]=\min_{k<j}\big(dp[i-1][k]+C(k,j)\big)
$$
has monotone argmins, so one layer can be computed with the template. Total roughly $O(K,n\log n)$.

#### Why It Matters

- Cuts a quadratic DP layer down to near linear
- Simple to implement compared to more advanced tricks
- Pairs well with prefix sums for $C(k,j)$ evaluation
- Core technique in editors line breaking, clustering in 1D, histogram smoothing

#### Step by Step on a small instance

Let $a=[1,3,2,4]$, $K=2$, $C(k,j)=(S[j]-S[k])^2$.

1. Initialize $dp[0][0]=0$, $dp[0][j>0]=+\infty$.
2. Layer $i=1$: compute $dp[1][j]$ by scanning $k<j$. Argmins are nondecreasing.
3. Layer $i=2$: call `compute(1,n,0,n-1, ...)`. The recursion halves the target interval and narrows candidate $k$ ranges by monotonicity.

#### A Gentle Proof (Why It Works)

Let $opt[j]$ be a minimizer for $dp[j]$. If $C$ is Monge, then for $x<y$ and $u<v$:
$$
C(u,x)+C(v,y)\le C(u,y)+C(v,x).
$$
Assume by contradiction $opt[x]>opt[y]$. Using optimality of those indices and the inequality above yields a contradiction. Hence $opt$ is nondecreasing.

The recursion evaluates $dp[mid]$ using candidates in $[optL,optR]$. The found $opt[mid]$ splits feasible candidates:

- Any optimal index for $j<mid$ is in $[optL,opt[mid]]$
- Any optimal index for $j>mid$ is in $[opt[mid],optR]$
  Induct over segments to show each $dp[j]$ is computed with exactly the needed candidate set and no essential candidate is excluded.

#### Try It Yourself

1. Replace $C(k,j)$ by $\alpha,(S[j]-S[k])^2+\beta,(j-k)$ and verify monotonicity still holds.
2. Use the template to speed up line breaking with raggedness penalty.
3. Benchmark naive $O(n^2)$ vs divide and conquer on random convex costs.
4. Combine with space optimization by keeping only two layers.
5. Contrast with Knuth optimization and Convex Hull Trick and decide which applies for your cost.

#### Test Cases

- Small convex cost:

  * $a=[2,1,3], K=2$, $C(k,j)=(S[j]-S[k])^2$
  * Compare naive and D&C outputs, they must match.
- Linear plus convex mix:

  * $a=[1,1,1,1], K=3$, $C(k,j)=(S[j]-S[k])^2+(j-k)$
- Edge cases:

  * $n=1$ any $K\ge1$
  * All zeros array, any $K$

#### Complexity

- One layer with divide and conquer: $O(n\log n)$ evaluations of $C$ in the simple form, often written as $O(n)$ splits with logarithmic recursion depth.
- Full DP: $O(K,n\log n)$ time, $O(n)$ or $O(K,n)$ space depending on whether you store all layers.

Divide and conquer DP is your go to when the argmin slides to the right as $j$ grows. It is small code, big speedup.

### 462 Knuth Optimization

Knuth Optimization is a special-case speedup for certain DP transitions where quadrangle inequality and monotonicity of optimal decisions hold. It improves $O(n^2)$ dynamic programs to $O(n)$ per layer, often used in optimal binary search tree, matrix chain, and interval partitioning problems.

#### What Problem Are We Solving?

We want to optimize DP recurrences of the form:
$$
dp[i][j] = \min_{k \in [i, j-1]} \big(dp[i][k] + dp[k+1][j] + C(i, j)\big)
$$
for $1 \le i \le j \le n$, where $C(i,j)$ is a cost function satisfying quadrangle inequality.

Naively, this is $O(n^3)$. With Knuth optimization, we cut it to $O(n^2)$ by restricting the search range using a monotonic property of the optimal split point.

#### Key Condition

Knuth optimization applies when:

1. Quadrangle Inequality:
   $$
   C(a, c) + C(b, d) \le C(a, d) + C(b, c), \quad \forall a \le b \le c \le d
   $$
2. Monotonicity of Argmin:
   $$
   opt[i][j-1] \le opt[i][j] \le opt[i+1][j]
   $$
   These ensure that the optimal split $k$ for $[i,j]$ moves rightward as intervals slide.

#### How Does It Work (Plain Language)

When computing $dp[i][j]$, the best partition point $k$ lies between $opt[i][j-1]$ and $opt[i+1][j]$.
So instead of scanning the full $[i, j-1]$, we limit to a narrow window.
This reduces work from $O(n^3)$ to $O(n^2)$.

We fill intervals in increasing length order, maintaining and reusing `opt[i][j]`.

#### Step-by-Step Recurrence

For all $i$:
$$
dp[i][i] = 0
$$

Then for lengths $len = 2..n$:

```text
for i in 1..n-len+1:
    j = i + len - 1
    dp[i][j] = ∞
    for k in opt[i][j-1]..opt[i+1][j]:
        val = dp[i][k] + dp[k+1][j] + C(i,j)
        if val < dp[i][j]:
            dp[i][j] = val
            opt[i][j] = k
```

#### Tiny Code (Easy Versions)

Python

```python
INF = 1018

def knuth_optimization(n, cost):
    dp = [[0]*n for _ in range(n)]
    opt = [[0]*n for _ in range(n)]

    for i in range(n):
        opt[i][i] = i

    for length in range(2, n+1):
        for i in range(0, n-length+1):
            j = i + length - 1
            dp[i][j] = INF
            start = opt[i][j-1]
            end = opt[i+1][j] if i+1 <= j else j-1
            for k in range(start, end+1):
                val = dp[i][k] + dp[k+1][j] + cost(i, j)
                if val < dp[i][j]:
                    dp[i][j] = val
                    opt[i][j] = k
    return dp[0][n-1]
```

C

```c
#include <stdio.h>
#include <limits.h>

#define INF 1000000000000000LL
#define N 505

long long dp[N][N], opt[N][N];

long long cost(int i, int j); // user-defined cost function

long long knuth(int n) {
    for (int i = 0; i < n; i++) {
        dp[i][i] = 0;
        opt[i][i] = i;
    }
    for (int len = 2; len <= n; len++) {
        for (int i = 0; i + len - 1 < n; i++) {
            int j = i + len - 1;
            dp[i][j] = INF;
            int start = opt[i][j-1];
            int end = opt[i+1][j];
            if (end == 0) end = j - 1;
            if (start > end) { int tmp = start; start = end; end = tmp; }
            for (int k = start; k <= end; k++) {
                long long val = dp[i][k] + dp[k+1][j] + cost(i, j);
                if (val < dp[i][j]) {
                    dp[i][j] = val;
                    opt[i][j] = k;
                }
            }
        }
    }
    return dp[0][n-1];
}
```

#### Why It Matters

Knuth Optimization turns a cubic DP into quadratic without approximations. It's especially useful for:

- Optimal Binary Search Trees (OBST)
- Matrix Chain Multiplication
- Merging Stones / File Merging
- Bracket Parsing / Partitioning

It's a precise algebraic optimization based on Monge arrays and convexity.

#### Step-by-Step Example

Consider merging files with sizes $[10,20,30]$.
$C(i,j)=\text{sum of file sizes from i to j}$.
We fill $dp[i][j]$ with minimal total cost of merging segment $[i..j]$.
Argmins move monotonically, so Knuth optimization applies.

#### A Gentle Proof (Why It Works)

If $C$ satisfies quadrangle inequality:
$$
C(a, c) + C(b, d) \le C(a, d) + C(b, c),
$$
then combining two adjacent subproblems will never cause the optimal cut to move left.
Hence, $opt[i][j-1] \le opt[i][j] \le opt[i+1][j]$.
Thus, restricting the search range preserves correctness while cutting redundant checks.

#### Try It Yourself

1. Apply Knuth optimization to Optimal BST:
   $$
   dp[i][j]=\min_{k\in[i,j]}(dp[i][k-1]+dp[k+1][j]+w[i][j])
   $$
2. Use it in Merging Stones (sum-cost merge).
3. Compare with Divide & Conquer DP, both need monotonicity, but Knuth's has fixed quadratic structure.
4. Verify monotonicity by printing $opt[i][j]$.
5. Prove $C(i,j)=\text{prefix}[j]-\text{prefix}[i-1]$ satisfies the condition.

#### Test Cases

| Case           | Description              | Expected Complexity |
| -------------- | ------------------------ | ------------------- |
| File merging   | [10, 20, 30]             | $O(n^2)$            |
| Optimal BST    | Sorted keys, frequencies | $O(n^2)$            |
| Merging Stones | Equal weights            | Monotone $opt$      |

#### Complexity

- Time: $O(n^2)$
- Space: $O(n^2)$

Knuth Optimization is the elegant midpoint between full DP and convexity tricks, precise, predictable, and optimal whenever cost satisfies Monge structure.

### 463 Convex Hull Trick (CHT)

The Convex Hull Trick speeds up DP transitions of the form
$$
dp[i] = \min_{k<i}\big(dp[k] + m_k \cdot x_i + b_k\big)
$$
when the slopes $m_k$ are monotonic (increasing or decreasing) and the $x_i$ queries are also monotonic.
It replaces $O(n^2)$ scanning with $O(n)$ amortized or $O(\log n)$ query time using a dynamic hull.

#### What Problem Are We Solving?

We have a DP recurrence like:
$$
dp[i] = \min_{k < i}\big(dp[k] + m_k \cdot x_i + b_k\big)
$$
where:

- $m_k$ (slope) and $b_k$ (intercept) define lines,
- $x_i$ is the query coordinate,
- we want the minimum (or maximum) value over all $k$.

This appears in:

- Line DP (e.g. segmented linear costs),
- Divide and Conquer DP (convex variant),
- Knuth-like DPs with linear penalties,
- Aliens trick and Li Chao Trees for non-monotone cases.

#### When It Applies

1. Transition fits $dp[i] = \min_k(dp[k] + m_k x_i + b_k)$
2. $m_k$'s are monotonic (non-decreasing or non-increasing)
3. $x_i$ queries are sorted (non-decreasing)

Then you can use a deque-based CHT for $O(1)$ amortized per insertion/query.

If slopes or queries are not monotonic, use Li Chao Tree (next algorithm).

#### How Does It Work (Plain Language)

Each $k$ defines a line $y = m_k x + b_k$.
The DP asks: for current $x_i$, which previous line gives the smallest value?
All lines together form a lower envelope, a piecewise minimum curve.
We maintain this hull incrementally and query the minimum efficiently.

If slopes are sorted, each new line intersects the previous hull at one point,
and old lines become useless after their intersection point.

#### Tiny Code (Easy Versions)

Python (Monotone CHT)

```python
class CHT:
    def __init__(self):
        self.lines = []  # (m, b)
    
    def bad(self, l1, l2, l3):
        # Check if l2 is unnecessary between l1 and l3
        return (l3[1] - l1[1]) * (l1[0] - l2[0]) <= (l2[1] - l1[1]) * (l1[0] - l3[0])

    def add(self, m, b):
        self.lines.append((m, b))
        while len(self.lines) >= 3 and self.bad(self.lines[-3], self.lines[-2], self.lines[-1]):
            self.lines.pop(-2)

    def query(self, x):
        # queries x in increasing order
        while len(self.lines) >= 2 and \
              self.lines[0][0]*x + self.lines[0][1] >= self.lines[1][0]*x + self.lines[1][1]:
            self.lines.pop(0)
        m, b = self.lines[0]
        return m*x + b

# Example: dp[i] = min(dp[k] + m[k]*x[i] + b[k])
dp = [0]*5
x = [1, 2, 3, 4, 5]
m = [2, 1, 3, 4, 5]
b = [5, 4, 3, 2, 1]

cht = CHT()
cht.add(m[0], b[0])
for i in range(1, 5):
    dp[i] = cht.query(x[i-1])
    cht.add(m[i], dp[i] + b[i])
```

C (Deque Implementation)

```c
#include <stdio.h>

typedef struct { long long m, b; } Line;
Line hull[100005];
int sz = 0, ptr = 0;

double intersect(Line a, Line b) {
    return (double)(b.b - a.b) / (a.m - b.m);
}

int bad(Line a, Line b, Line c) {
    return (c.b - a.b)*(a.m - b.m) <= (b.b - a.b)*(a.m - c.m);
}

void add_line(long long m, long long b) {
    Line L = {m, b};
    while (sz >= 2 && bad(hull[sz-2], hull[sz-1], L)) sz--;
    hull[sz++] = L;
}

long long query(long long x) {
    while (ptr + 1 < sz && hull[ptr+1].m * x + hull[ptr+1].b <= hull[ptr].m * x + hull[ptr].b)
        ptr++;
    return hull[ptr].m * x + hull[ptr].b;
}
```

#### Why It Matters

- Reduces $O(n^2)$ DP with linear transition to $O(n)$ or $O(n\log n)$.
- Extremely common in optimization tasks:

  * Convex cost partitioning
  * Slope trick extensions
  * Dynamic programming with linear penalties
- Foundation for Li Chao Trees and Slope Trick.

#### Step-by-Step Example

Suppose
$$
dp[i] = \min_{k<i}\big(dp[k] + a_k \cdot b_i + c_k\big)
$$
Given:

- $a = [2,4,6]$
- $b = [1,2,3]$
- $c = [5,4,2]$

Each step:

1. Add line $y = m_k x + b_k = a_k x + (dp[k] + c_k)$
2. Query at $x_i = b[i]$ to get min value.

CHT keeps only useful lines forming lower envelope.

#### A Gentle Proof (Why It Works)

For monotonic slopes, intersection points are sorted.
Once a line becomes worse than the next at a certain $x$,
it will never be optimal again for larger $x$.
Therefore, we can pop it from the deque —
each line enters and leaves once → $O(n)$ amortized.

#### Try It Yourself

1. Adapt for maximum query (flip signs).
2. Combine with DP: $dp[i] = \min_k(dp[k] + m_k x_i + b_k)$.
3. Add Li Chao Tree for unsorted slopes/queries.
4. Visualize lower envelope intersection points.
5. Compare with Slope Trick (piecewise-linear potentials).

#### Test Cases

| m       | b       | x | Expected min |
| ------- | ------- | - | ------------ |
| [1,2,3] | [0,1,3] | 1 | 1            |
| [2,1]   | [5,4]   | 2 | 6            |
| [1,3,5] | [2,2,2] | 4 | 14           |

#### Complexity

- Time: $O(n)$ amortized (monotone queries) or $O(n\log n)$ (Li Chao)
- Space: $O(n)$

Convex Hull Trick is the bridge between geometry and DP, every line a subproblem, every envelope a minimization frontier.

### 464 Li Chao Tree

The Li Chao Tree is a dynamic data structure for maintaining a set of lines and efficiently querying the minimum (or maximum) value at any given $x$. Unlike the Convex Hull Trick, it works even when slopes and query points are arbitrary and unordered.

#### What Problem Are We Solving?

We want to handle DP recurrences (or cost functions) of the form:

$$
dp[i] = \min_{j < i} (m_j \cdot x_i + b_j)
$$

Each $j$ contributes a line $y = m_jx + b_j$.
We must find the minimum value among all added lines at a given $x_i$.

Unlike the Convex Hull Trick (which needs monotonic $x$ or $m$),
Li Chao Tree handles any insertion or query order, no sorting required.

#### When It Applies

Li Chao Tree applies when:

- You need to add arbitrary lines ($m_j, b_j$) over time
- You must query arbitrary $x$ in any order
- You want min or max queries efficiently
- Slopes and queries are not monotonic

This makes it ideal for:

- DP with arbitrary slopes
- Online queries
- Geometry problems involving lower envelopes
- Line container queries in computational geometry

#### How Does It Work (Plain Language)

The Li Chao Tree divides the $x$-axis into segments.
Each node represents an interval, storing one line that's currently optimal over part (or all) of that range.

When a new line is added:

- Compare it with the current line in the interval
- Swap if it's better at the midpoint
- Recursively insert into one child (left/right), narrowing the range

When querying:

- Descend the tree using $x$
- Combine values of lines encountered
- Return the minimum (or maximum)

This yields $O(\log X)$ time per insertion and query,
where $X$ is the range of $x$ values (discretized if necessary).

#### Tiny Code (Easy Version)

Python (Min Version)

```python
INF = 1018

class Line:
    def __init__(self, m, b):
        self.m = m
        self.b = b
    def value(self, x):
        return self.m * x + self.b

class Node:
    def __init__(self, l, r):
        self.l = l
        self.r = r
        self.line = None
        self.left = None
        self.right = None

class LiChaoTree:
    def __init__(self, l, r):
        self.root = Node(l, r)
    
    def _add(self, node, new_line):
        l, r = node.l, node.r
        m = (l + r) // 2
        if node.line is None:
            node.line = new_line
            return
        
        left_better = new_line.value(l) < node.line.value(l)
        mid_better = new_line.value(m) < node.line.value(m)
        
        if mid_better:
            node.line, new_line = new_line, node.line
        
        if r - l == 0:
            return
        if left_better != mid_better:
            if not node.left:
                node.left = Node(l, m)
            self._add(node.left, new_line)
        else:
            if not node.right:
                node.right = Node(m + 1, r)
            self._add(node.right, new_line)
    
    def add_line(self, m, b):
        self._add(self.root, Line(m, b))
    
    def _query(self, node, x):
        if node is None:
            return INF
        res = node.line.value(x) if node.line else INF
        m = (node.l + node.r) // 2
        if x <= m:
            return min(res, self._query(node.left, x))
        else:
            return min(res, self._query(node.right, x))
    
    def query(self, x):
        return self._query(self.root, x)

# Example usage
tree = LiChaoTree(0, 100)
tree.add_line(2, 3)
tree.add_line(-1, 10)
print(tree.query(5))  # minimum value among all lines at x=5
```

#### Why It Matters

- Handles arbitrary slopes and queries
- Efficient for online DP and geometry optimization
- Generalizes CHT (works without monotonic constraints)
- Can be used for both min and max queries (just flip inequalities)

#### Step-by-Step Example

Suppose we insert lines:

1. $y = 2x + 3$
2. $y = -x + 10$

Then query at $x = 5$:

- First line: $2(5) + 3 = 13$
- Second line: $-5 + 10 = 5$
  → answer = 5

The tree ensures each query returns the best line without brute force.

#### A Gentle Proof (Why It Works)

Each interval stores a line that's locally optimal at some midpoint.
If a new line is better at one endpoint, it must eventually overtake the existing line, so the intersection lies in that half.

By recursing on halves, we ensure the correct line is chosen for every $x$.

The tree height is $\log X$,
and each insertion affects at most $\log X$ nodes.

Hence:
$$
T(n) = O(n \log X)
$$

#### Try It Yourself

1. Implement Li Chao Tree for max queries (invert comparisons).
2. Add $n$ random lines and query random $x$.
3. Apply it to DP with linear cost:
   $$
   dp[i] = \min_{j < i}(dp[j] + a_jx_i + b_j)
   $$
4. Visualize segment splits and stored lines.
5. Compare with Convex Hull Trick on monotonic test cases.

#### Test Cases

| Lines | Queries | Range    | Time       | Works |
| ----- | ------- | -------- | ---------- | ----- |
| 10    | 10      | [0, 100] | O(n log X) | ✓     |
| 1e5   | 1e5     | [0, 1e9] | O(n log X) | ✓     |

#### Complexity

- Time: $O(\log X)$ per insert/query
- Space: $O(\log X)$ per line (tree nodes)

The Li Chao Tree is your line oracle, always ready to give the best line, no matter how chaotic your slopes and queries become.

### 465 Slope Trick

The Slope Trick is a dynamic programming optimization technique for problems involving piecewise-linear convex functions. It lets you maintain and update the shape of a convex cost function efficiently, especially when transitions involve operations like adding absolute values, shifting minima, or combining convex shapes.

#### What Problem Are We Solving?

Many DP problems involve minimizing a cost function that changes shape over time, such as:

$$
dp[i] = \min_x (dp[i-1](x) + |x - a_i|)
$$

Here, $dp[i]$ is not a single value but a function of $x$.
The Slope Trick is how we maintain this function efficiently as a sequence of linear segments.

This comes up when:

- You need to add |x - a| terms
- You need to shift the whole function left or right
- You need to add constants or merge minima

Rather than storing the full function, we store key "breakpoints" and update in logarithmic time.

#### When It Applies

Slope Trick applies when:

- The cost function is convex and piecewise linear
- Each transition is of the form:

  * $f(x) + |x - a|$
  * $f(x + c)$ or $f(x - c)$
  * $\min_x(f(x)) + c$
- You need to track minimal cost across shifting choices

Common in:

- Median DP
- Path alignment
- Convex smoothing
- Minimizing sum of absolute differences
- Cost balancing problems

#### How Does It Work (Plain Language)

Instead of recomputing the entire cost function each time,
we maintain two priority queues (heaps) that track where the slope changes.

Think of the cost function as a mountain made of straight lines:

- Adding $|x - a|$ means putting a "tent" centered at $a$
- Moving $x$ left/right shifts the mountain
- The minimum point can move but remains easy to find

We track the slope's left and right breakpoints using heaps:

- Left heap (max-heap): stores slopes to the left of minimum
- Right heap (min-heap): stores slopes to the right

Each operation updates these heaps in $O(\log n)$.

#### Example Problem

Minimize:
$$
dp[i] = \min_x(dp[i-1](x) + |x - a_i|)
$$

We want the minimal total distance to all $a_1, a_2, ..., a_i$.

The optimal $x$ is the median of all $a$'s seen so far.

Slope Trick maintains this function efficiently.

#### Tiny Code (Easy Version)

Python

```python
import heapq

class SlopeTrick:
    def __init__(self):
        self.left = []   # max-heap (store negative values)
        self.right = []  # min-heap
        self.min_cost = 0

    def add_abs(self, a):
        if not self.left:
            heapq.heappush(self.left, -a)
            heapq.heappush(self.right, a)
            return
        if a < -self.left[0]:
            heapq.heappush(self.left, -a)
            val = -heapq.heappop(self.left)
            heapq.heappush(self.right, val)
            self.min_cost += -self.left[0] - a
        else:
            heapq.heappush(self.right, a)
            val = heapq.heappop(self.right)
            heapq.heappush(self.left, -val)
            self.min_cost += a - val

    def get_min(self):
        return self.min_cost

# Example
st = SlopeTrick()
for a in [3, 1, 4, 1, 5]:
    st.add_abs(a)
print("Minimum cost:", st.get_min())
```

This structure efficiently tracks the cost of minimizing the sum of absolute differences.

#### Why It Matters

- Reduces function-based DPs into heap updates
- Elegant solution for convex minimization
- Handles |x - a|, shift, and constant add in $O(\log n)$
- Avoids discretization of continuous $x$

#### Step-by-Step Example

Suppose we add points sequentially: $a = [3, 1, 4]$

1. Add $|x - 3|$: min at $x = 3$
2. Add $|x - 1|$: min moves to $x = 2$
3. Add $|x - 4|$: min moves to $x = 3$

Heaps track these balance points dynamically.
Total cost is sum of minimal shifts.

#### A Gentle Proof (Why It Works)

Adding $|x - a|$ modifies slope:

- For $x < a$, slope increases by $-1$
- For $x > a$, slope increases by $+1$

Thus, the function stays convex.
Heaps store where slope crosses zero (the minimum).

Balancing heaps keeps slopes equalized, ensuring minimum at the median.
Each operation maintains convexity and updates cost correctly.

#### Try It Yourself

1. Implement `add_shift(c)` to shift function horizontally.
2. Solve:
   $$
   dp[i] = \min_x(dp[i-1](x) + |x - a_i|)
   $$
   for a list of $a_i$
3. Add `add_constant(c)` for vertical shifts.
4. Track the running median using heaps.
5. Visualize slope evolution, it should always form a "V" shape.

#### Test Cases

| Input        | Expected Minimum |
| ------------ | ---------------- |
| [3]          | 0                |
| [3, 1]       | 2                |
| [3, 1, 4]    | 3                |
| [3, 1, 4, 1] | 5                |

#### Complexity

- Time: $O(n\log n)$
- Space: $O(n)$

The Slope Trick is like origami for DP, you fold and shift convex functions to shape the minimal path, one segment at a time.

### 466 Monotonic Queue Optimization

Monotonic Queue Optimization is a dynamic programming acceleration technique for recurrences involving sliding windows or range-limited minima. It replaces naive scanning ($O(nk)$) with a monotonic deque that finds optimal states in $O(n)$.

#### What Problem Are We Solving?

We want to optimize DP of the form:

$$
dp[i] = \min_{j \in [i-k,, i-1]} (dp[j] + cost(j, i))
$$

or simpler, when $cost(j, i)$ is monotonic or separable, like $w_i$ or $c(i-j)$, we can maintain a window of candidate $j$'s.

This pattern appears in:

- Sliding window DPs
- Shortest path in DAGs with window constraints
- Queue scheduling problems
- Constrained subsequence or segment DPs

#### When It Applies

You can apply Monotonic Queue Optimization when:

- The transition uses contiguous ranges of $j$ (like a window)
- The cost function is monotonic, allowing pruning of bad states
- You want to find $\min$ or $\max$ over a sliding window efficiently

Common forms:

- $dp[i] = \min_{j \in [i-k, i]} (dp[j] + c[j])$
- $dp[i] = \max_{j \in [i-k, i]} (dp[j] + w[i])$

This trick does not require convexity, only monotonic ordering in the transition range.

#### How Does It Work (Plain Language)

Instead of checking all $k$ previous states for each $i$,
we maintain a deque of indices that are still potentially optimal.

At each step:

1. Remove old indices (outside window)
2. Pop worse states (whose value is greater than the new one)
3. Front of deque gives the best $j$ for current $i$

This ensures the deque is monotonic (increasing or decreasing depending on min/max).

#### Example Recurrence

$$
dp[i] = \min_{j \in [i-k, i-1]} (dp[j] + w_i)
$$

Since $w_i$ doesn't depend on $j$, we just need $\min dp[j]$ over the last $k$ indices.

#### Tiny Code (Easy Version)

Python

```python
from collections import deque

def min_sliding_window_dp(arr, k):
    n = len(arr)
    dp = [0] * n
    dq = deque()
    
    for i in range(n):
        # Remove out-of-window
        while dq and dq[0] < i - k:
            dq.popleft()
        # Pop worse elements
        while dq and dp[dq[-1]] >= dp[i - 1] if i > 0 else False:
            dq.pop()
        # Push current
        dq.append(i)
        # Compute dp
        dp[i] = arr[i] + (dp[dq[0]] if dq else 0)
    return dp
```

C

```c
#include <stdio.h>
#define N 100000
#define INF 1000000000

int dp[N], a[N], q[N];

int main() {
    int n = 5, k = 2;
    int front = 0, back = 0;

    int arr[5] = {3, 1, 4, 1, 5};
    dp[0] = arr[0];
    q[back++] = 0;

    for (int i = 1; i < n; i++) {
        while (front < back && q[front] < i - k) front++;
        dp[i] = dp[q[front]] + arr[i];
        while (front < back && dp[q[back - 1]] >= dp[i]) back--;
        q[back++] = i;
    }

    for (int i = 0; i < n; i++) printf("%d ", dp[i]);
}
```

#### Why It Matters

- Converts range-min DPs from $O(nk)$ → $O(n)$
- Essential for problems with window constraints
- Avoids heap overhead (constant-time updates)
- Extremely simple and robust

#### Step-by-Step Example

Let $arr = [3,1,4,1,5]$, $k = 2$

At $i = 2$:

- Candidates: $j \in [0, 1]$
- dp[0]=3, dp[1]=4
- dq = [1] after pruning worse values
- dp[2] = arr[2] + dp[1] = 4 + 4 = 8

Deque moves as window slides, always holding potential minima.

#### A Gentle Proof (Why It Works)

At each $i$:

- Remove indices $< i-k$ (out of range)
- Maintain monotonic order of dp-values in deque
- The front always gives the smallest $dp[j]$ in window
- Because each element is pushed and popped once, total operations = $O(n)$

Thus, overall complexity is linear.

#### Try It Yourself

1. Implement max version by reversing comparisons.
2. Apply to $dp[i] = \min_{j \in [i-k, i]} (dp[j] + c_i)$
3. Visualize deque evolution per step.
4. Solve constrained path problems with limited jump size.
5. Compare runtime with naive $O(nk)$ approach.

#### Test Cases

| Input        | k | Expected    |
| ------------ | - | ----------- |
| [3,1,4,1,5]  | 2 | fast min DP |
| [10,9,8,7,6] | 3 | decreasing  |
| [1,2,3,4,5]  | 1 | simple      |

#### Complexity

- Time: $O(n)$
- Space: $O(k)$

Monotonic Queue Optimization is your sliding window oracle, keeping just the right candidates, and tossing the rest without looking back.

### 467 Bitset DP

Bitset DP is a performance optimization technique that uses bit-level parallelism to speed up dynamic programming, especially when state transitions involve Boolean operations over large ranges. By representing states as bits, multiple transitions can be processed simultaneously using fast bitwise operators.

#### What Problem Are We Solving?

We want to optimize DPs like:

$$
dp[i] = \text{reachable states after considering first } i \text{ elements}
$$

Often in subset sum, knapsack, path existence, or mask propagation, we deal with states where:

- Each state is true/false
- Transition is shifting or combining bits

For example, in Subset Sum:
$$
dp[i][s] = dp[i-1][s] \lor dp[i-1][s - a_i]
$$
We can compress this into a bitset shift.

#### When It Applies

You can use Bitset DP when:

- States are Boolean (true/false)
- Transition is shift-based or additive
- State space is dense and bounded

Common use cases:

- Subset Sum ($O(nS / w)$)
- Bounded Knapsack
- Graph reachability
- Palindromic substrings DP
- Counting with bit masks

Here, $w$ is word size (e.g. 64), giving up to 64x speedup.

#### How Does It Work (Plain Language)

Represent each DP layer as a bitset, each bit indicates whether a state is reachable.

For Subset Sum:

- Initially: `dp[0] = 1` (sum = 0 reachable)
- For each number `a`:

  * Shift left by `a` → new reachable sums
  * Combine: `dp |= dp << a`

Example:
Adding 3 to {0, 2, 5} means shift left by 3 → {3, 5, 8}.

All in one CPU instruction!

#### Example Recurrence

$$
dp[s] = dp[s] \lor dp[s - a]
$$

Bitset form:

$$
dp = dp \lor (dp \ll a)
$$

#### Tiny Code (Easy Version)

Python (using int bitset)

```python
def subset_sum(nums, target):
    dp = 1  # bit 0 = reachable sum 0
    for a in nums:
        dp |= dp << a
    return (dp >> target) & 1  # check if bit target is set

print(subset_sum([3, 2, 7], 5))  # True (3+2)
print(subset_sum([3, 2, 7], 6))  # False
```

C (using bitset)

```c
#include <stdio.h>
#include <string.h>

#define MAXS 10000
#define W 64
unsigned long long dp[MAXS / W + 1];

void setbit(int i) { dp[i / W] |= 1ULL << (i % W); }
int getbit(int i) { return (dp[i / W] >> (i % W)) & 1; }

int main() {
    memset(dp, 0, sizeof(dp));
    setbit(0); // sum 0 reachable
    int a[] = {3, 2, 7}, n = 3;
    for (int i = 0; i < n; i++) {
        int v = a[i];
        for (int j = MAXS / W; j >= 0; j--) {
            unsigned long long shifted = dp[j] << v;
            if (j + v / W + 1 <= MAXS / W)
                dp[j + v / W + 1] |= dp[j] >> (W - (v % W));
            dp[j] |= shifted;
        }
    }
    printf("Sum 5 reachable? %d\n", getbit(5));
}
```

#### Why It Matters

- Exploits hardware parallelism
- Ideal for dense Boolean DP
- Works for subset sums, range transitions, graph masks
- Achieves massive speedups with simple operations

#### Step-by-Step Example

Suppose `nums = [2, 3]`, `target = 5`

Start: `dp = 1` → {0}

After `2`:
`dp << 2` = {2}
`dp |= dp << 2` = {0, 2}

After `3`:
`dp << 3` = {3, 5}
`dp |= dp << 3` = {0, 2, 3, 5}

Bit 5 is set → sum 5 reachable ✅

#### A Gentle Proof (Why It Works)

Each shift corresponds to adding an element to a subset sum.
Bitwise OR merges reachable sums.
No overlap conflict, each bit is unique to a sum.
After all shifts, all sums formed by subsets are represented.

Each shift-OR runs in $O(S / w)$ time,
where $S$ = target sum, $w$ = word size.

#### Try It Yourself

1. Implement bounded knapsack via repeated shift-and-OR.
2. Count distinct subset sums (popcount of dp).
3. Apply to palindrome DP: $dp[i][j] = s[i] == s[j] \land dp[i+1][j-1]$.
4. Visualize bit patterns after each step.
5. Benchmark vs normal DP on large $S$.

#### Test Cases

| nums         | target | Result |
| ------------ | ------ | ------ |
| [3, 2, 7]    | 5      | True   |
| [3, 2, 7]    | 6      | False  |
| [1, 2, 3, 4] | 10     | True   |

#### Complexity

- Time: $O(nS / w)$
- Space: $O(S / w)$

Bitset DP is your Boolean supercharger, turning slow loops into blinding-fast bitwise moves.

### 468 Offline DP Queries

Offline DP Queries are a strategy to handle queries on a dynamic programming state space by reordering or batching them for efficient computation. Instead of answering queries as they arrive (online), we process them *after sorting or grouping*, enabling faster transitions or range updates.

#### What Problem Are We Solving?

You may have a DP or recurrence that evolves over time, and a set of queries asking for values at specific states or intervals, like:

- "What is $dp[x]$ after all updates?"
- "What is the min cost among indices in [L, R]?"
- "How many reachable states satisfy condition C?"

Naively answering queries as they appear leads to repeated recomputation.
By processing them offline, we exploit sorting, prefix accumulation, or data structure reuse.

#### When It Applies

Offline DP Query methods apply when:

- Queries can be sorted (by time, index, or key)
- Transitions or states evolve monotonically
- You can batch updates and reuse results

Common cases:

- Range DP queries: $dp[i]$ over [L, R]
- Monotonic state DPs (like convex hull or segment DP)
- Mo's algorithm on DP states
- Incremental DPs where $dp[i]$ is finalized before querying

#### How Does It Work (Plain Language)

Instead of answering as we go, we:

1. Collect all queries
2. Sort them by a relevant dimension (like time or index)
3. Process DP transitions incrementally
4. Answer queries once the needed states are available

Think of it as "moving forward once" and answering everything you pass.

By decoupling query order from input order,
you avoid recomputation and exploit monotonic progression of DP.

#### Example Problem

You're asked $q$ queries:

> For each $x_i$, what is the minimum $dp[j] + cost(j, x_i)$ over all $j \le x_i$?

Naively, $O(nq)$.
Offline, sort queries by $x_i$,
process $j = 1 \ldots n$,
and maintain current DP structure (like a segment tree or convex hull).

#### Tiny Code (Easy Version)

Python (sorted queries with running DP)

```python
def offline_dp(arr, queries):
    # arr defines dp transitions
    # queries = [(x, idx)]
    n = len(arr)
    dp = [0] * (n + 1)
    res = [0] * len(queries)

    queries.sort()  # sort by x
    ptr = 0
    for i in range(1, n + 1):
        dp[i] = dp[i-1] + arr[i-1]
        # process queries with x == i
        while ptr < len(queries) and queries[ptr][0] == i:
            _, idx = queries[ptr]
            res[idx] = dp[i]
            ptr += 1
    return res

arr = [3, 1, 4, 1, 5]
queries = [(3, 0), (5, 1)]
print(offline_dp(arr, queries))  # [sum of first 3, sum of first 5]
```

C (sorted queries)

```c
#include <stdio.h>
#include <stdlib.h>

typedef struct { int x, idx; } Query;

int cmp(const void *a, const void *b) {
    return ((Query*)a)->x - ((Query*)b)->x;
}

int main() {
    int arr[] = {3, 1, 4, 1, 5}, n = 5;
    Query q[] = {{3,0},{5,1}};
    int dp[6] = {0}, res[2];

    qsort(q, 2, sizeof(Query), cmp);
    int ptr = 0;
    for (int i = 1; i <= n; i++) {
        dp[i] = dp[i-1] + arr[i-1];
        while (ptr < 2 && q[ptr].x == i) {
            res[q[ptr].idx] = dp[i];
            ptr++;
        }
    }

    printf("%d %d\n", res[0], res[1]); // dp[3], dp[5]
}
```

#### Why It Matters

- Converts repeated query updates into one forward pass
- Enables range optimizations (segment trees, CHT, etc.)
- Reduces complexity from $O(nq)$ to $O(n + q \log n)$ or better
- Essential for problems mixing queries + DP updates

#### Step-by-Step Example

Suppose we compute cumulative $dp[i] = dp[i-1] + a[i]$
and queries ask $dp[x]$ for random $x$:

Naive:

- Recompute each query: $O(qn)$

Offline:

- Sort queries by $x$
- Single pass $O(n + q)$

Same principle applies to complex DPs if queries depend on monotone indices.

#### A Gentle Proof (Why It Works)

If DP states evolve monotonically in one dimension (index or time),
then after computing $dp[1]$ to $dp[x]$,
the answer to all queries with bound ≤ $x$ is final.

Sorting ensures we never recompute older states,
and every query sees exactly what it needs, no more, no less.

Thus, each DP transition and query is processed once,
yielding $O(n + q)$ total complexity.

#### Try It Yourself

1. Implement offline queries for prefix sums
2. Combine with Convex Hull Trick for sorted $x_i$
3. Use segment tree for range min DP queries
4. Implement offline Subset Sum queries (by sum ≤ X)
5. Compare performance with online queries

#### Test Cases

| arr         | queries | Output  |
| ----------- | ------- | ------- |
| [3,1,4,1,5] | [3,5]   | [8,14]  |
| [2,2,2]     | [1,2,3] | [2,4,6] |

#### Complexity

- Time: $O(n + q \log q)$
- Space: $O(n + q)$

Offline DP Queries are your time travelers, answer questions from the future by rearranging them into a single efficient sweep through the past.

### 469 DP + Segment Tree

DP + Segment Tree is a hybrid optimization pattern that combines dynamic programming with a segment tree (or Fenwick tree) to handle transitions involving range queries (min, max, sum) efficiently. It's especially useful when each DP state depends on a range of previous states, rather than a single index.

#### What Problem Are We Solving?

We want to compute:

$$
dp[i] = \min_{l_i \le j \le r_i}(dp[j] + cost[j])
$$

or more generally,

$$
dp[i] = \text{aggregate over range [L(i), R(i)] of some function of } dp[j]
$$

When transitions span intervals, naive iteration over each range is $O(n^2)$.
A segment tree reduces this to $O(n \log n)$ by supporting range queries and point updates.

#### When It Applies

Use DP + Segment Tree when:

- Transitions depend on intervals or ranges
- The DP recurrence is monotonic in index order
- You need fast min, max, or sum over subsets

Typical problems:

- Range-based knapsack variants
- Sequence partitioning with range cost
- Interval scheduling DP
- Increasing subsequences with weight
- Pathfinding with segment bounds

#### How Does It Work (Plain Language)

Instead of looping over all $j$ to find the best previous state:

1. Store $dp[j]$ values in a segment tree
2. Query for the minimum/maximum over $[L(i), R(i)]$
3. Add transition cost and store $dp[i]$ back

This way, every step:

- Query = $O(\log n)$
- Update = $O(\log n)$
  Total = $O(n \log n)$

#### Example Recurrence

$$
dp[i] = \min_{j < i,, a_j < a_i}(dp[j]) + cost(i)
$$

If $a_i$ values can be ordered or compressed,
we can query the segment tree for all $a_j < a_i$ efficiently.

#### Tiny Code (Easy Version)

Python

```python
INF = 1018

class SegmentTree:
    def __init__(self, n):
        self.N = 1
        while self.N < n:
            self.N *= 2
        self.data = [INF] * (2 * self.N)
    def update(self, i, val):
        i += self.N
        self.data[i] = val
        while i > 1:
            i //= 2
            self.data[i] = min(self.data[2*i], self.data[2*i+1])
    def query(self, l, r):
        l += self.N
        r += self.N
        res = INF
        while l < r:
            if l % 2:
                res = min(res, self.data[l])
                l += 1
            if r % 2:
                r -= 1
                res = min(res, self.data[r])
            l //= 2
            r //= 2
        return res

def dp_segment_tree(arr):
    n = len(arr)
    dp = [INF] * n
    seg = SegmentTree(n)
    dp[0] = arr[0]
    seg.update(0, dp[0])
    for i in range(1, n):
        best = seg.query(max(0, i - 2), i)  # e.g., range [i-2, i-1]
        dp[i] = arr[i] + best
        seg.update(i, dp[i])
    return dp

arr = [3, 1, 4, 1, 5]
print(dp_segment_tree(arr))
```

C

```c
#include <stdio.h>
#define INF 1000000000
#define N 100005

int seg[4*N], dp[N], arr[N];

int min(int a, int b) { return a < b ? a : b; }

void update(int idx, int val, int id, int l, int r) {
    if (l == r) { seg[id] = val; return; }
    int mid = (l + r) / 2;
    if (idx <= mid) update(idx, val, 2*id, l, mid);
    else update(idx, val, 2*id+1, mid+1, r);
    seg[id] = min(seg[2*id], seg[2*id+1]);
}

int query(int ql, int qr, int id, int l, int r) {
    if (qr < l || r < ql) return INF;
    if (ql <= l && r <= qr) return seg[id];
    int mid = (l + r) / 2;
    return min(query(ql, qr, 2*id, l, mid),
               query(ql, qr, 2*id+1, mid+1, r));
}

int main() {
    int arr[] = {3,1,4,1,5}, n = 5;
    for (int i = 0; i < 4*N; i++) seg[i] = INF;
    dp[0] = arr[0];
    update(0, dp[0], 1, 0, n-1);
    for (int i = 1; i < n; i++) {
        int best = query(i-2 >= 0 ? i-2 : 0, i-1, 1, 0, n-1);
        dp[i] = arr[i] + best;
        update(i, dp[i], 1, 0, n-1);
    }
    for (int i = 0; i < n; i++) printf("%d ", dp[i]);
}
```

#### Why It Matters

- Handles range transitions efficiently
- Reduces quadratic DPs to $O(n \log n)$
- Works for both min and max recurrences
- Combines with coordinate compression for complex ranges

#### Step-by-Step Example

Let $arr = [3, 1, 4, 1, 5]$, and
$$
dp[i] = arr[i] + \min_{j \in [i-2, i-1]} dp[j]
$$

- $i=0$: $dp[0]=3$
- $i=1$: query $[0,0]$, $dp[1]=1+3=4$
- $i=2$: query $[0,1]$, $dp[2]=4+1=5$
- $i=3$: query $[1,2]$, $dp[3]=1+4=5$
- $i=4$: query $[2,3]$, $dp[4]=5+4=9$

#### A Gentle Proof (Why It Works)

Segment trees store range minima.
Each DP state only depends on previously finalized values.
As you move $i$ forward, you query and update disjoint ranges.
Hence total complexity:

$$
O(n \log n) \text{ (n queries + n updates)}
$$

No recomputation, each transition is resolved via the tree in logarithmic time.

#### Try It Yourself

1. Change recurrence to max and adjust segment tree.
2. Solve weighted LIS: $dp[i] = w_i + \max_{a_j < a_i} dp[j]$.
3. Combine with coordinate compression for arbitrary $a_i$.
4. Visualize segment tree contents over iterations.
5. Apply to interval scheduling with overlapping windows.

#### Test Cases

| arr         | Range     | Output      |
| ----------- | --------- | ----------- |
| [3,1,4,1,5] | [i-2,i-1] | [3,4,5,5,9] |
| [2,2,2,2]   | [i-1,i-1] | [2,4,6,8]   |

#### Complexity

- Time: $O(n \log n)$
- Space: $O(n)$

Segment Tree + DP is your range oracle, answering every interval dependency without scanning the whole past.

### 470 Divide & Conquer Knapsack

Divide & Conquer Knapsack is an optimization method that accelerates dynamic programming for large-capacity knapsack problems by recursively splitting the item set and combining results, rather than building a full $O(nW)$ DP table. It is especially powerful when you need to reconstruct solutions or handle queries across subsets.

#### What Problem Are We Solving?

The classic 0/1 Knapsack Problem is:

$$
dp[i][w] = \max(dp[i-1][w],, dp[i-1][w - w_i] + v_i)
$$

where $w_i$ is the item's weight and $v_i$ its value.

This standard DP costs $O(nW)$ in time and space, which becomes infeasible when $n$ or $W$ is large.

Divide & Conquer Knapsack tackles this by splitting items into halves and solving subproblems recursively, a strategy similar to meet-in-the-middle, but adapted to DP.

#### When It Applies

Use Divide & Conquer Knapsack when:

- You have many items ($n > 1000$)
- Capacity $W$ is large, but manageable via combinations
- You need partial solution reconstruction
- You want to handle batch queries (e.g., best value for each capacity range)

Common contexts:

- Large $n$, moderate $W$ (split across subsets)
- Enumerating feasible states
- Offline processing of item sets
- Recursive solution generation (for decision trees or subset enumeration)

#### How Does It Work (Plain Language)

Instead of building one giant DP table,
we split the item list into halves:

- Solve left half → get all achievable $(weight, value)$ pairs
- Solve right half → same
- Merge results efficiently (like convolution or sweep)

By recursively combining subproblems,
you reduce total recomputation and enable parallel merging of feasible subsets.

If $n = 2^k$, recursion depth = $O(\log n)$, and each merge costs $O(2^{n/2})$, much faster than $O(nW)$ when $W$ is large.

#### Example Recurrence

Let `solve(l, r)` compute all feasible pairs for items $[l, r)$:

```text
If r - l == 1:
    return {(0,0), (w_l, v_l)}
Else:
    mid = (l + r) / 2
    left = solve(l, mid)
    right = solve(mid, r)
    return combine(left, right)
```

`combine` merges pairs from left and right (like merging sorted lists, keeping only Pareto-optimal pairs).

#### Tiny Code (Easy Version)

Python

```python
def combine(left, right, W):
    res = []
    for w1, v1 in left:
        for w2, v2 in right:
            w = w1 + w2
            if w <= W:
                res.append((w, v1 + v2))
    # Keep only best value per weight (Pareto frontier)
    res.sort()
    best = []
    cur = -1
    for w, v in res:
        if v > cur:
            best.append((w, v))
            cur = v
    return best

def solve(items, W):
    n = len(items)
    if n == 1:
        w, v = items[0]
        return [(0, 0), (w, v)] if w <= W else [(0, 0)]
    mid = n // 2
    left = solve(items[:mid], W)
    right = solve(items[mid:], W)
    return combine(left, right, W)

items = [(3, 4), (4, 5), (7, 10), (8, 11)]
W = 10
print(solve(items, W))  # [(0,0),(3,4),(4,5),(7,10),(8,11),(10,14)]
```

#### Why It Matters

- Avoids full $O(nW)$ DP when $W$ is large
- Enables offline merging and solution reconstruction
- Useful in meet-in-the-middle optimization
- Can handle dynamic constraints by recombining subsets

#### Step-by-Step Example

Items:
$(3,4), (4,5), (7,10), (8,11)$, $W = 10$

Split:

- Left half: $(3,4), (4,5)$ → feasible = {(0,0),(3,4),(4,5),(7,9)}
- Right half: $(7,10), (8,11)$ → {(0,0),(7,10),(8,11),(15,21)}

Combine all $(w_L + w_R, v_L + v_R)$ ≤ 10:

- (0,0), (3,4), (4,5), (7,9), (7,10), (8,11), (10,14)

Pareto-optimal:

- (0,0), (3,4), (4,5), (7,10), (10,14)

Max value for $W=10$: 14

#### A Gentle Proof (Why It Works)

By recursively splitting:

- Each subset's combinations are enumerated in $O(2^{n/2})$
- Merge step ensures only non-dominated states are carried forward
- Recursion covers all subsets exactly once

Thus, total cost ≈ $O(2^{n/2})$ instead of $O(nW)$.

For moderate $n$ (≤40), this is dramatically faster.

For large $n$ with constraints (bounded weights), merges reduce to $O(n \log n)$ per layer.

#### Try It Yourself

1. Implement value-only knapsack (max value ≤ W)
2. Visualize Pareto frontier after each combine
3. Use recursion tree to print intermediate DP states
4. Compare against standard $O(nW)$ DP results
5. Extend to multi-dimensional weights

#### Test Cases

| Items                       | W  | Result |
| --------------------------- | -- | ------ |
| [(3,4),(4,5),(7,10),(8,11)] | 10 | 14     |
| [(1,1),(2,2),(3,3)]         | 3  | 3      |
| [(2,3),(3,4),(4,5)]         | 5  | 7      |

#### Complexity

- Time: $O(2^{n/2} \cdot n)$ (meet-in-the-middle)
- Space: $O(2^{n/2})$

Divide & Conquer Knapsack is your recursive craftsman, building optimal subsets by combining halves, not filling tables.

# Section 48. Tree DP and Rerooting 

### 471 Subtree Sum DP

Subtree Sum DP is one of the most fundamental patterns in tree dynamic programming. It computes the sum of values in every node's subtree using a simple post-order traversal. Once you know how to aggregate over subtrees, you can extend the same idea to handle sizes, depths, counts, or any associative property.

#### What Problem Are We Solving?

Given a rooted tree where each node has a value, compute for every node the sum of values in its subtree (including itself).

For a node $u$ with children $v_1, v_2, \dots, v_k$, the subtree sum is:

$$
dp[u] = value[u] + \sum_{v \in children(u)} dp[v]
$$

This idea generalizes to many forms of aggregation, such as counting nodes, finding subtree size, or computing subtree products.

#### How Does It Work (Plain Language)

Think of each node as a small calculator.
When a node finishes computing its children's sums, it adds them all up, plus its own value.
This is post-order traversal, compute from leaves upward.

#### Step-by-Step Example

Consider the tree:

```
       1(5)
      /   \
   2(3)   3(2)
   / \
4(1) 5(4)
```

Values:

- Node 1 → 5
- Node 2 → 3
- Node 3 → 2
- Node 4 → 1
- Node 5 → 4

We compute bottom-up:

- $dp[4] = 1$
- $dp[5] = 4$
- $dp[2] = 3 + 1 + 4 = 8$
- $dp[3] = 2$
- $dp[1] = 5 + 8 + 2 = 15$

So the subtree sums are:

| Node | Subtree Sum |
| ---- | ----------- |
| 1    | 15          |
| 2    | 8           |
| 3    | 2           |
| 4    | 1           |
| 5    | 4           |

#### Tiny Code (Easy Version)

C

```c
#include <stdio.h>
#define MAXN 100

int n;
int value[MAXN];
int adj[MAXN][MAXN], deg[MAXN];
int dp[MAXN];

int dfs(int u, int parent) {
    dp[u] = value[u];
    for (int i = 0; i < deg[u]; i++) {
        int v = adj[u][i];
        if (v == parent) continue;
        dp[u] += dfs(v, u);
    }
    return dp[u];
}

int main() {
    n = 5;
    int edges[][2] = {{1,2},{1,3},{2,4},{2,5}};
    for (int i = 0; i < 4; i++) {
        int a = edges[i][0], b = edges[i][1];
        adj[a][deg[a]++] = b;
        adj[b][deg[b]++] = a;
    }
    int vals[] = {0,5,3,2,1,4};
    for (int i = 1; i <= n; i++) value[i] = vals[i];
    dfs(1, -1);
    for (int i = 1; i <= n; i++) printf("dp[%d] = %d\n", i, dp[i]);
}
```

Python

```python
from collections import defaultdict
n = 5
edges = [(1,2),(1,3),(2,4),(2,5)]
value = {1:5, 2:3, 3:2, 4:1, 5:4}

g = defaultdict(list)
for u,v in edges:
    g[u].append(v)
    g[v].append(u)

dp = {}

def dfs(u, p):
    dp[u] = value[u]
    for v in g[u]:
        if v == p: continue
        dfs(v, u)
        dp[u] += dp[v]

dfs(1, -1)
print(dp)
```

#### Why It Matters

- A core tree DP pattern: many problems reduce to aggregating over subtrees
- Forms the basis of rerooting DP, tree diameter, and centroid decomposition
- Used in computing subtree sizes, subtree XOR, sum of depths, subtree counts

Once you master subtree DP, you can generalize to:

- Max/min subtree values
- Counting paths through nodes
- Dynamic rerooting transitions

#### A Gentle Proof (Why It Works)

By induction on tree depth:

- Base Case: For a leaf node $u$, $dp[u] = value[u]$, correct by definition.
- Inductive Step: Assume all children $v$ have correct $dp[v]$.
  Then $dp[u] = value[u] + \sum dp[v]$ correctly accumulates all values in $u$'s subtree.

Since every node is visited once and every edge twice, total cost is $O(n)$.

#### Try It Yourself

1. Modify the code to compute subtree size instead of sum
2. Track maximum value in each subtree
3. Extend to compute sum of depths per subtree
4. Add rerooting to compute subtree sum for every root
5. Use input parser to build arbitrary trees

#### Test Cases

| Tree     | Values        | Subtree Sums    |
| -------- | ------------- | --------------- |
| 1–2–3    | {1:1,2:2,3:3} | {1:6, 2:5, 3:3} |
| 1–2, 1–3 | {1:5,2:2,3:1} | {1:8, 2:2, 3:1} |

#### Complexity

- Time: $O(n)$ (each node visited once)
- Space: $O(n)$ recursion + adjacency

Subtree Sum DP is your first brush with tree dynamics, one traversal, full insight.

### 472 Diameter DP

Diameter DP computes the longest path in a tree, the *diameter*. Unlike shortest paths, the diameter is measured by the greatest distance between any two nodes, not necessarily passing through the root. Using dynamic programming, we can derive this in a single DFS traversal by combining the two deepest child paths at every node.

#### What Problem Are We Solving?

Given a tree with $n$ nodes (unweighted or weighted), find the diameter, i.e. the length of the longest simple path between any two nodes.

For an unweighted tree, this is measured in edges or nodes; for a weighted tree, in sum of edge weights.

We define a DP recurrence:

$$
dp[u] = \text{length of longest downward path from } u
$$

At each node, the diameter candidate is the sum of the two longest child paths:

$$
diameter = \max(diameter, top1 + top2)
$$

#### How Does It Work (Plain Language)

Think of every node as a hub connecting paths from its children.
The longest path passing through a node is formed by picking its two deepest child paths and joining them.
We collect this as we perform a post-order DFS.

In the end, the global maximum across all nodes is the tree's diameter.

#### Step-by-Step Example

Tree:

```
      1
     / \
    2   3
   / \
  4   5
```

Each edge has weight 1.

Compute longest downward paths:

- Leaves (4, 5, 3): $dp=0$
- Node 2: $dp[2] = 1 + \max(dp[4], dp[5]) = 1 + 0 = 1$
- Node 1: $dp[1] = 1 + \max(dp[2], dp[3]) = 1 + 1 = 2$

Now compute diameter:

- At node 2: top1=0, top2=0 → local diameter=0
- At node 1: top1=1 (from 2), top2=1 (from 3) → local diameter=2

So the tree diameter = 2 edges (path 4–2–1–3)

#### Tiny Code (Easy Version)

C

```c
#include <stdio.h>
#define MAXN 100

int n;
int adj[MAXN][MAXN], deg[MAXN];
int diameter = 0;

int dfs(int u, int p) {
    int top1 = 0, top2 = 0;
    for (int i = 0; i < deg[u]; i++) {
        int v = adj[u][i];
        if (v == p) continue;
        int depth = 1 + dfs(v, u);
        if (depth > top1) {
            top2 = top1;
            top1 = depth;
        } else if (depth > top2) {
            top2 = depth;
        }
    }
    if (top1 + top2 > diameter) diameter = top1 + top2;
    return top1;
}

int main() {
    n = 5;
    int edges[][2] = {{1,2},{1,3},{2,4},{2,5}};
    for (int i = 0; i < 4; i++) {
        int a = edges[i][0], b = edges[i][1];
        adj[a][deg[a]++] = b;
        adj[b][deg[b]++] = a;
    }
    dfs(1, -1);
    printf("Tree diameter: %d edges\n", diameter);
}
```

Python

```python
from collections import defaultdict

g = defaultdict(list)
edges = [(1,2),(1,3),(2,4),(2,5)]
for u,v in edges:
    g[u].append(v)
    g[v].append(u)

diameter = 0

def dfs(u, p):
    global diameter
    top1 = top2 = 0
    for v in g[u]:
        if v == p: continue
        depth = 1 + dfs(v, u)
        if depth > top1:
            top2 = top1
            top1 = depth
        elif depth > top2:
            top2 = depth
    diameter = max(diameter, top1 + top2)
    return top1

dfs(1, -1)
print("Tree diameter:", diameter)
```

#### Why It Matters

- Central building block for tree analysis, network radius, center finding
- Used in problems involving longest paths, tree heights, centroid decomposition
- Serves as a key step in rerooting or centroid algorithms

#### A Gentle Proof (Why It Works)

Let's prove correctness by induction:

- Base Case: A leaf node has $dp[u] = 0$, no contribution beyond itself.
- Inductive Step: For each internal node $u$, if all children $v$ correctly compute their longest downward paths $dp[v]$, then combining the two largest gives the longest path through $u$.
  Since every path in a tree passes through some lowest common ancestor $u$, our DFS finds the true maximum globally.

#### Try It Yourself

1. Modify the code for weighted edges
2. Return both endpoints of the diameter path
3. Compare with two-pass BFS method (pick farthest node twice)
4. Extend to compute height of each subtree alongside
5. Visualize recursion tree with local diameters

#### Test Cases

| Tree                           | Diameter |
| ------------------------------ | -------- |
| Line: 1–2–3–4                  | 3 edges  |
| Star: 1–{2,3,4,5}              | 2 edges  |
| Balanced binary tree (depth 2) | 4 edges  |

#### Complexity

- Time: $O(n)$
- Space: $O(n)$ recursion stack

Diameter DP is the lens that reveals the tree's longest breath, one sweep, full span.

### 473 Independent Set DP

Independent Set DP finds the largest set of nodes in a tree such that no two chosen nodes are adjacent. This is a classic tree dynamic programming problem, showcasing the fundamental trade-off between inclusion and exclusion at each node.

#### What Problem Are We Solving?

Given a tree with $n$ nodes (each possibly having a weight), find the maximum-weight independent set, a subset of nodes such that no two connected nodes are selected.

For each node $u$, we maintain two DP states:

- $dp[u][0]$: maximum value in $u$'s subtree when $u$ is not chosen
- $dp[u][1]$: maximum value in $u$'s subtree when $u$ is chosen

The recurrence:

$$
dp[u][0] = \sum_{v \in children(u)} \max(dp[v][0], dp[v][1])
$$

$$
dp[u][1] = value[u] + \sum_{v \in children(u)} dp[v][0]
$$

The answer is $\max(dp[root][0], dp[root][1])$.

#### How Does It Work (Plain Language)

Each node decides:

- If it includes itself, it excludes its children
- If it excludes itself, it can take the best of each child

This "take-or-skip" strategy flows bottom-up from leaves to root.

#### Step-by-Step Example

Tree:

```
      1(3)
     /   \
   2(2)  3(1)
   /
 4(4)
```

Values in parentheses.

Compute DP from bottom:

- Node 4:
  $dp[4][0]=0$, $dp[4][1]=4$

- Node 2:
  $dp[2][0]=\max(dp[4][0],dp[4][1])=4$
  $dp[2][1]=2+dp[4][0]=2+0=2$

- Node 3:
  $dp[3][0]=0$, $dp[3][1]=1$

- Node 1:
  $dp[1][0]=\max(dp[2][0],dp[2][1])+\max(dp[3][0],dp[3][1])=4+1=5$
  $dp[1][1]=3+dp[2][0]+dp[3][0]=3+4+0=7$

Answer: $\max(5,7)=7$

Best set = {1,4}

#### Tiny Code (Easy Version)

C

```c
#include <stdio.h>
#define MAXN 100

int n;
int value[MAXN];
int adj[MAXN][MAXN], deg[MAXN];
int dp[MAXN][2];
int visited[MAXN];

void dfs(int u, int p) {
    dp[u][0] = 0;
    dp[u][1] = value[u];
    for (int i = 0; i < deg[u]; i++) {
        int v = adj[u][i];
        if (v == p) continue;
        dfs(v, u);
        dp[u][0] += (dp[v][0] > dp[v][1] ? dp[v][0] : dp[v][1]);
        dp[u][1] += dp[v][0];
    }
}

int main() {
    n = 4;
    int edges[][2] = {{1,2},{1,3},{2,4}};
    for (int i = 0; i < 3; i++) {
        int a = edges[i][0], b = edges[i][1];
        adj[a][deg[a]++] = b;
        adj[b][deg[b]++] = a;
    }
    int vals[] = {0,3,2,1,4};
    for (int i = 1; i <= n; i++) value[i] = vals[i];
    dfs(1, -1);
    int ans = dp[1][0] > dp[1][1] ? dp[1][0] : dp[1][1];
    printf("Max independent set sum: %d\n", ans);
}
```

Python

```python
from collections import defaultdict

g = defaultdict(list)
edges = [(1,2),(1,3),(2,4)]
for u,v in edges:
    g[u].append(v)
    g[v].append(u)

value = {1:3, 2:2, 3:1, 4:4}
dp = {}

def dfs(u, p):
    include = value[u]
    exclude = 0
    for v in g[u]:
        if v == p: continue
        dfs(v, u)
        include += dp[v][0]
        exclude += max(dp[v][0], dp[v][1])
    dp[u] = (exclude, include)

dfs(1, -1)
print("Max independent set sum:", max(dp[1]))
```

#### Why It Matters

- Foundation for tree-based constraint problems
- Used in network stability, resource allocation, scheduling
- Extensible to weighted graphs, forests, ranged constraints

Patterns derived from this:

- Vertex cover (complement)
- House robber on trees
- Dynamic inclusion-exclusion states

#### A Gentle Proof (Why It Works)

We prove by induction:

- Base case: Leaf node $u$ has $dp[u][1]=value[u]$, $dp[u][0]=0$, correct.
- Inductive step: For any node $u$, if all subtrees compute optimal values,
  including $u$ adds its value and excludes children ($dp[v][0]$),
  excluding $u$ allows best child choice ($\max(dp[v][0], dp[v][1])$).
  Thus each subtree is optimal, ensuring global optimality.

#### Try It Yourself

1. Extend to weighted edges (where cost is per edge)
2. Modify to reconstruct chosen nodes
3. Implement for forest (multiple trees)
4. Compare with vertex cover DP
5. Apply to House Robber III (Leetcode 337)

#### Test Cases

| Tree           | Values            | Answer        |
| -------------- | ----------------- | ------------- |
| 1–2–3          | {1:3,2:2,3:1}     | {1,3} sum=4   |
| Star 1–{2,3,4} | {1:5, others:3}   | {2,3,4} sum=9 |
| Chain 1–2–3–4  | {1:1,2:4,3:5,4:4} | {2,4} sum=8   |

#### Complexity

- Time: $O(n)$
- Space: $O(n)$ recursion

Independent Set DP captures the tree's quiet balance, every node's choice echoes through its branches.

### 474 Vertex Cover DP

Vertex Cover DP solves a classic tree optimization problem: choose the smallest set of nodes such that every edge in the tree has at least one endpoint selected. This complements the Independent Set DP, together they form a dual pair in combinatorial optimization.

#### What Problem Are We Solving?

Given a tree with $n$ nodes, find the minimum vertex cover, i.e. a smallest subset of nodes such that every edge $(u,v)$ has $u$ or $v$ in the set.

We define two states for each node $u$:

- $dp[u][0]$: minimum size of vertex cover in subtree rooted at $u$, when $u$ is not included
- $dp[u][1]$: minimum size of vertex cover in subtree rooted at $u$, when $u$ is included

Recurrence:

$$
dp[u][0] = \sum_{v \in children(u)} dp[v][1]
$$

$$
dp[u][1] = 1 + \sum_{v \in children(u)} \min(dp[v][0], dp[v][1])
$$

If $u$ is not in the cover, all its children must be included to cover edges $(u,v)$.
If $u$ is included, each child can choose whether or not to join the cover.

Answer: $\min(dp[root][0], dp[root][1])$

#### How Does It Work (Plain Language)

Each node decides whether to take responsibility for covering edges or delegate that responsibility to its children.
This is a mutual-exclusion constraint:

- If $u$ is excluded, its children must be included.
- If $u$ is included, each child is free to choose.

#### Step-by-Step Example

Tree:

```
    1
   / \
  2   3
 / \
4   5
```

We compute bottom-up:

- Leaves 4,5:
  $dp[4][0]=0$, $dp[4][1]=1$
  $dp[5][0]=0$, $dp[5][1]=1$

- Node 2:
  $dp[2][0]=dp[4][1]+dp[5][1]=2$
  $dp[2][1]=1+\min(dp[4][0],dp[4][1])+\min(dp[5][0],dp[5][1])=1+0+0=1$

- Node 3:
  $dp[3][0]=0$, $dp[3][1]=1$

- Node 1:
  $dp[1][0]=dp[2][1]+dp[3][1]=1+1=2$
  $dp[1][1]=1+\min(dp[2][0],dp[2][1])+\min(dp[3][0],dp[3][1])=1+1+0=2$

Result: $\min(2,2)=2$

Minimum vertex cover size = 2

#### Tiny Code (Easy Version)

C

```c
#include <stdio.h>
#define MAXN 100

int n;
int adj[MAXN][MAXN], deg[MAXN];
int dp[MAXN][2];
int visited[MAXN];

void dfs(int u, int p) {
    dp[u][0] = 0;
    dp[u][1] = 1;
    for (int i = 0; i < deg[u]; i++) {
        int v = adj[u][i];
        if (v == p) continue;
        dfs(v, u);
        dp[u][0] += dp[v][1];
        dp[u][1] += (dp[v][0] < dp[v][1] ? dp[v][0] : dp[v][1]);
    }
}

int main() {
    n = 5;
    int edges[][2] = {{1,2},{1,3},{2,4},{2,5}};
    for (int i = 0; i < 4; i++) {
        int a = edges[i][0], b = edges[i][1];
        adj[a][deg[a]++] = b;
        adj[b][deg[b]++] = a;
    }
    dfs(1, -1);
    int ans = dp[1][0] < dp[1][1] ? dp[1][0] : dp[1][1];
    printf("Minimum vertex cover: %d\n", ans);
}
```

Python

```python
from collections import defaultdict

g = defaultdict(list)
edges = [(1,2),(1,3),(2,4),(2,5)]
for u,v in edges:
    g[u].append(v)
    g[v].append(u)

dp = {}

def dfs(u, p):
    include = 1
    exclude = 0
    for v in g[u]:
        if v == p: continue
        dfs(v, u)
        exclude += dp[v][1]
        include += min(dp[v][0], dp[v][1])
    dp[u] = (exclude, include)

dfs(1, -1)
print("Minimum vertex cover:", min(dp[1]))
```

#### Why It Matters

- Fundamental for constraint satisfaction problems on trees
- Dual to Independent Set DP (by complement)
- Used in network design, task monitoring, sensor placement

Many graph algorithms (on trees) rely on this cover-or-skip dichotomy, including:

- Dominating sets
- Guard problems
- Minimum cameras in binary tree (Leetcode 968)

#### A Gentle Proof (Why It Works)

By induction:

- Base case: Leaf $u$:
  $dp[u][0]=0$ (if not covered, edge must be covered by parent), $dp[u][1]=1$ (include self).
- Inductive step:
  If all children have optimal covers,

  * Excluding $u$ forces inclusion of all $v$ (ensures edges $(u,v)$ covered).
  * Including $u$ allows flexible optimal choices for children.
    Each node's local decision yields global minimality since the tree has no cycles.

#### Try It Yourself

1. Print actual cover set (backtrack from DP)
2. Extend to weighted vertex cover (replace count with sum of weights)
3. Compare with Independent Set DP, show complement sizes
4. Implement iterative version using post-order
5. Apply to Minimum Cameras in Binary Tree

#### Test Cases

| Tree             | Result         | Cover |
| ---------------- | -------------- | ----- |
| 1–2–3            | 1–2–3          | {2}   |
| Star 1–{2,3,4,5} | {1}            |       |
| Chain 1–2–3–4    | {2,4} or {1,3} |       |

#### Complexity

- Time: $O(n)$
- Space: $O(n)$

Vertex Cover DP shows how including a node protects its edges, balancing economy and completeness across the tree.

### 475 Path Counting DP

Path Counting DP is a gentle entry point into tree combinatorics, it helps you count how many distinct paths exist under certain conditions, such as from root to leaves, between pairs, or with specific constraints. It builds intuition for counting with structure rather than brute force.

#### What Problem Are We Solving?

Given a tree with $n$ nodes, we want to count paths satisfying some property. The simplest form is counting the number of root-to-leaf paths, but we can generalize to:

- Total number of paths between all pairs
- Paths with weight constraints
- Paths with certain node properties

We'll start with the fundamental version, root-to-leaf paths.

Define:

$$
dp[u] = \text{number of paths starting at } u
$$

For a rooted tree, each node's paths equal the sum of paths from its children. If $u$ is a leaf, it contributes $1$ path (just itself):

$$
dp[u] =
\begin{cases}
1, & \text{if } u \text{ is a leaf},\\
\displaystyle\sum_{v \in \text{children}(u)} dp[v], & \text{otherwise.}
\end{cases}
$$

The total number of root-to-leaf paths = $dp[root]$.

#### How Does It Work (Plain Language)

Start at the leaves, each leaf is one complete path.
Each parent accumulates paths from its children:
"Every path from my child forms one from me too."
This propagates upward until the root holds the total count.

#### Step-by-Step Example

Tree:

```
      1
     / \
    2   3
   / \
  4   5
```

Compute $dp$ from leaves upward:

- $dp[4] = 1$, $dp[5] = 1$, $dp[3] = 1$
- $dp[2] = dp[4] + dp[5] = 2$
- $dp[1] = dp[2] + dp[3] = 3$

So there are 3 root-to-leaf paths:

1. 1–2–4
2. 1–2–5
3. 1–3

#### Tiny Code (Easy Version)

C

```c
#include <stdio.h>
#define MAXN 100

int n;
int adj[MAXN][MAXN], deg[MAXN];
int dp[MAXN];

int dfs(int u, int p) {
    int count = 0;
    int isLeaf = 1;
    for (int i = 0; i < deg[u]; i++) {
        int v = adj[u][i];
        if (v == p) continue;
        isLeaf = 0;
        count += dfs(v, u);
    }
    if (isLeaf) return dp[u] = 1;
    return dp[u] = count;
}

int main() {
    n = 5;
    int edges[][2] = {{1,2},{1,3},{2,4},{2,5}};
    for (int i = 0; i < 4; i++) {
        int a = edges[i][0], b = edges[i][1];
        adj[a][deg[a]++] = b;
        adj[b][deg[b]++] = a;
    }
    dfs(1, -1);
    printf("Root-to-leaf paths: %d\n", dp[1]);
}
```

Python

```python
from collections import defaultdict

g = defaultdict(list)
edges = [(1,2),(1,3),(2,4),(2,5)]
for u,v in edges:
    g[u].append(v)
    g[v].append(u)

dp = {}

def dfs(u, p):
    is_leaf = True
    count = 0
    for v in g[u]:
        if v == p: continue
        is_leaf = False
        count += dfs(v, u)
    dp[u] = 1 if is_leaf else count
    return dp[u]

dfs(1, -1)
print("Root-to-leaf paths:", dp[1])
```

#### Why It Matters

- Foundation for counting problems on trees
- Forms the basis of path-sum DP, tree DP rerooting, and combinatorial enumeration
- Essential in probabilistic models and decision trees
- Useful for probability propagation and branching process simulation

#### A Gentle Proof (Why It Works)

We can prove by induction:

- Base case: Leaf node $u$ has exactly one path, itself.
  So $dp[u]=1$.
- Inductive step: Assume all children $v$ compute correct counts $dp[v]$.
  Then $dp[u] = \sum dp[v]$ counts all distinct root-to-leaf paths passing through $u$.

Since every path is uniquely identified by its first branching decision, we never double-count.

#### Try It Yourself

1. Modify to count all simple paths (pairs $(u,v)$).
2. Add edge weights and count paths with total sum $\le K$.
3. Track and print all root-to-leaf paths using recursion stack.
4. Extend to directed acyclic graphs (DAGs).
5. Combine with rerooting to count paths through each node.

#### Test Cases

| Tree                | Paths   |
| ------------------- | ------- |
| 1–2–3               | 1 path  |
| Star (1–{2,3,4})    | 3 paths |
| Binary tree depth 2 | 3 paths |

#### Complexity

- Time: $O(n)$ (each node visited once)
- Space: $O(n)$ recursion

Path Counting DP shows how structure transforms into number, one traversal, all paths accounted.

### 476 DP on Rooted Tree

DP on Rooted Tree is the most general pattern of tree dynamic programming, it teaches you to reason about *states* on hierarchical structures. Every subtree contributes partial answers, and a parent combines them. This is the building block for almost every tree-based DP: sums, counts, distances, constraints, and beyond.

#### What Problem Are We Solving?

We want to compute a property for each node based on its subtree, things like:

- Subtree sum
- Subtree size
- Maximum depth
- Path counts
- Modular products
- Combinatorial counts

Given a rooted tree, we define a DP function that recursively collects results from each child and aggregates them.

Generic form:

$$
dp[u] = f(u, {dp[v] : v \in children(u)})
$$

You define:

- Base case (usually for leaves)
- Transition function (combine children's results)
- Merge operation (sum, max, min, multiply, etc.)

#### How Does It Work (Plain Language)

You think from the leaves upward.
Each node:

1. Collects results from its children.
2. Applies a combining function.
3. Stores a final value.

This bottom-up reasoning mirrors *post-order traversal*, solve children first, then parent.

The power is that $f$ can represent *any operation*: sum, min, max, or even bitmask merge.

#### Step-by-Step Example

Let's compute subtree size for every node (number of nodes in its subtree):

Recurrence:

$$
dp[u] = 1 + \sum_{v \in children(u)} dp[v]
$$

Tree:

```
      1
     / \
    2   3
   / \
  4   5
```

Compute bottom-up:

- $dp[4]=1$, $dp[5]=1$
- $dp[2]=1+1+1=3$
- $dp[3]=1$
- $dp[1]=1+3+1=5$

So:

- $dp[1]=5$
- $dp[2]=3$
- $dp[3]=1$
- $dp[4]=1$
- $dp[5]=1$

#### Tiny Code (Easy Version)

C

```c
#include <stdio.h>
#define MAXN 100

int n;
int adj[MAXN][MAXN], deg[MAXN];
int dp[MAXN];

int dfs(int u, int p) {
    dp[u] = 1; // count itself
    for (int i = 0; i < deg[u]; i++) {
        int v = adj[u][i];
        if (v == p) continue;
        dp[u] += dfs(v, u);
    }
    return dp[u];
}

int main() {
    n = 5;
    int edges[][2] = {{1,2},{1,3},{2,4},{2,5}};
    for (int i = 0; i < 4; i++) {
        int a = edges[i][0], b = edges[i][1];
        adj[a][deg[a]++] = b;
        adj[b][deg[b]++] = a;
    }
    dfs(1, -1);
    for (int i = 1; i <= n; i++)
        printf("dp[%d] = %d\n", i, dp[i]);
}
```

Python

```python
from collections import defaultdict

g = defaultdict(list)
edges = [(1,2),(1,3),(2,4),(2,5)]
for u,v in edges:
    g[u].append(v)
    g[v].append(u)

dp = {}

def dfs(u, p):
    dp[u] = 1
    for v in g[u]:
        if v == p: continue
        dp[u] += dfs(v, u)
    return dp[u]

dfs(1, -1)
print(dp)
```

#### Why It Matters

- Core template for any tree DP problem
- Powers algorithms like subtree sum, depth counting, modular product aggregation, path count, and rerooting
- Foundation for advanced rerooting DP, where answers depend on parent and sibling states

Once you master this pattern, you can:

- Change the recurrence → change the problem
- Add constraints → introduce multiple DP states
- Extend to graphs with DAG structure

#### A Gentle Proof (Why It Works)

Induction on tree height:

- Base case: Leaf $u$ → $dp[u]$ initialized to base (e.g. 1 or value[u]).
- Inductive step: Suppose all children $v$ compute correct $dp[v]$.
  Then $f(u, {dp[v]})$ aggregates subtree results correctly.

Since trees are acyclic, post-order guarantees children are processed first, correctness follows.

#### Try It Yourself

1. Change recurrence to compute sum of subtree values.
2. Compute maximum subtree depth.
3. Track count of leaves in each subtree.
4. Extend to two-state DP, e.g. include/exclude logic.
5. Combine with rerooting to compute value for every root.

#### Test Cases

| Tree             | Subtree Sizes     |
| ---------------- | ----------------- |
| 1–2–3            | {1:3, 2:2, 3:1}   |
| Star 1–{2,3,4,5} | {1:5, others:1}   |
| Chain 1–2–3–4    | {1:4,2:3,3:2,4:1} |

#### Complexity

- Time: $O(n)$
- Space: $O(n)$ recursion

DP on Rooted Tree is your *canvas*, define $f$, and paint the structure's logic across its branches.

### 477 Rerooting Technique

The Rerooting Technique is a powerful pattern in tree dynamic programming that allows you to compute results for every node as root, in linear time. Instead of recalculating from scratch for each root, we "reroot" efficiently by reusing already computed subtree results.

#### What Problem Are We Solving?

Suppose we've computed a property (like subtree sum, subtree size, distance sum) rooted at a fixed node, say node $1$.
Now we want to compute the same property for all nodes as root, for example:

- Sum of distances from each node to all others
- Size or cost of each subtree when rooted differently
- Count of paths or contributions that depend on root position

We define:

$$
dp[u] = f({dp[v]}_{v \in children(u)})
$$

But we also want $res[u]$, the answer when the tree is rooted at $u$.

By rerooting, we can transfer results along edges, moving the root and updating only local contributions.

#### How Does It Work (Plain Language)

We first perform a post-order DFS to compute $dp[u]$ for every node as if the root were fixed (e.g. node 1).

Then, a second pre-order DFS "pushes" results outward —
when moving root from parent $u$ to child $v$, we adjust contributions:

- Remove child's part from parent
- Add parent's part to child

Each rerooting step updates $res[v]$ from $res[u]$ in constant or small time.

This two-pass structure is the hallmark of rerooting DP:

1. Downward pass: gather subtree results
2. Upward pass: propagate parent contributions

#### Step-by-Step Example: Sum of Distances

Goal: for each node, compute sum of distances to all other nodes.

Tree:

```
      1
     / \
    2   3
   / \
  4   5
```

Step 1: Post-order (subtree sums + sizes)

For each node:

- $subtree_size[u]$ = number of nodes in subtree of $u$
- $dp[u]$ = sum of distances from $u$ to nodes in its subtree

Recurrence:

$$
subtree_size[u] = 1 + \sum subtree_size[v]
$$

$$
dp[u] = \sum (dp[v] + subtree_size[v])
$$

At root (1), $dp[1]=8$ (sum of distances from 1 to all nodes).

Step 2: Pre-order (rerooting)

When rerooting from $u$ to $v$:

- Moving root away from $v$ adds $(n - subtree_size[v])$
- Moving root toward $v$ subtracts $subtree_size[v]$

So:
$$
dp[v] = dp[u] + (n - 2 \times subtree_size[v])
$$

Now every node has its distance sum in $O(n)$.

#### Tiny Code (Sum of Distances)

C

```c
#include <stdio.h>
#define MAXN 100

int n;
int adj[MAXN][MAXN], deg[MAXN];
int dp[MAXN], subtree[MAXN], res[MAXN];

void dfs1(int u, int p) {
    subtree[u] = 1;
    dp[u] = 0;
    for (int i = 0; i < deg[u]; i++) {
        int v = adj[u][i];
        if (v == p) continue;
        dfs1(v, u);
        subtree[u] += subtree[v];
        dp[u] += dp[v] + subtree[v];
    }
}

void dfs2(int u, int p) {
    res[u] = dp[u];
    for (int i = 0; i < deg[u]; i++) {
        int v = adj[u][i];
        if (v == p) continue;
        int pu = dp[u], pv = dp[v];
        int su = subtree[u], sv = subtree[v];

        // move root u -> v
        dp[u] -= dp[v] + subtree[v];
        subtree[u] -= subtree[v];
        dp[v] += dp[u] + subtree[u];
        subtree[v] += subtree[u];

        dfs2(v, u);

        // restore
        dp[v] = pv;
        dp[u] = pu;
        subtree[v] = sv;
        subtree[u] = su;
    }
}

int main() {
    n = 5;
    int edges[][2] = {{1,2},{1,3},{2,4},{2,5}};
    for (int i = 0; i < 4; i++) {
        int a = edges[i][0], b = edges[i][1];
        adj[a][deg[a]++] = b;
        adj[b][deg[b]++] = a;
    }
    dfs1(1, -1);
    dfs2(1, -1);
    for (int i = 1; i <= n; i++)
        printf("res[%d] = %d\n", i, res[i]);
}
```

Python

```python
from collections import defaultdict
g = defaultdict(list)
edges = [(1,2),(1,3),(2,4),(2,5)]
for u,v in edges:
    g[u].append(v)
    g[v].append(u)

n = 5
dp = {i:0 for i in range(1,n+1)}
sub = {i:1 for i in range(1,n+1)}
res = {}

def dfs1(u,p):
    sub[u]=1
    dp[u]=0
    for v in g[u]:
        if v==p: continue
        dfs1(v,u)
        sub[u]+=sub[v]
        dp[u]+=dp[v]+sub[v]

def dfs2(u,p):
    res[u]=dp[u]
    for v in g[u]:
        if v==p: continue
        pu,pv=dp[u],dp[v]
        su,sv=sub[u],sub[v]

        dp[u]-=dp[v]+sub[v]
        sub[u]-=sub[v]
        dp[v]+=dp[u]+sub[u]
        sub[v]+=sub[u]

        dfs2(v,u)

        dp[u],dp[v]=pu,pv
        sub[u],sub[v]=su,sv

dfs1(1,-1)
dfs2(1,-1)
print(res)
```

#### Why It Matters

- Compute answers for all nodes in $O(n)$
- Essential in distance sums, rerooted subtree queries, centroid-based algorithms
- Core pattern in Tree Rerooting DP problems on AtCoder, Codeforces, Leetcode

Rerooting transforms one-root logic into every-root knowledge.

#### A Gentle Proof (Why It Works)

1. First pass ensures each node knows its subtree contribution.
2. Second pass applies a constant-time update to shift the root:

   * Remove child's contribution
   * Add parent's complement

Since each edge is traversed twice, total cost is linear.

#### Try It Yourself

1. Count number of nodes in subtree for every possible root.
2. Compute sum of depths for every root.
3. Modify recurrence for product of subtree values.
4. Apply to tree balancing: minimize total distance.
5. Extend to weighted trees.

#### Test Cases

| Tree  | Root | Distance Sum |
| ----- | ---- | ------------ |
| 1–2–3 | 1    | 3            |
| 1–2–3 | 2    | 2            |
| 1–2–3 | 3    | 3            |

#### Complexity

- Time: $O(n)$
- Space: $O(n)$

Rerooting DP is your algorithmic kaleidoscope, spin the root, and watch all perspectives appear.

### 478 Distance Sum Rerooting

Distance Sum Rerooting is a classic and elegant application of the rerooting technique. It computes, for every node, the sum of distances to all other nodes in a tree, in just O(n) time. It's one of the cleanest examples showing how rerooting transforms local subtree data into global insight.

#### What Problem Are We Solving?

Given a tree with $n$ nodes, for every node $u$, compute:

$$
res[u] = \sum_{v=1}^{n} \text{dist}(u, v)
$$

A naïve approach (running BFS from every node) takes $O(n^2)$.
We'll do it in two DFS passes using rerooting DP.

#### How Does It Work (Plain Language)

1. First pass (post-order):
   Root the tree at an arbitrary node (say, 1).
   Compute:

   * $subtree[u]$: size of subtree rooted at $u$
   * $dp[u]$: sum of distances from $u$ to all nodes in its subtree

   Recurrence:

   $$
   subtree[u] = 1 + \sum_{v \in children(u)} subtree[v]
   $$

   $$
   dp[u] = \sum_{v \in children(u)} (dp[v] + subtree[v])
   $$

2. Second pass (pre-order):
   Use rerooting to compute $res[v]$ from $res[u]$:

   $$
   res[v] = res[u] + (n - 2 \times subtree[v])
   $$

   * Moving root from $u$ to $v$:

     * Nodes inside $v$'s subtree get 1 closer
     * Nodes outside get 1 farther
   * So net change = $-subtree[v] + (n - subtree[v]) = n - 2 \times subtree[v]$

3. $res[1] = dp[1]$ (initial result for root)

After this, each node has its sum of distances, all in linear time.

#### Step-by-Step Example

Tree:

```
      1
     / \
    2   3
   / \
  4   5
```

Step 1 (Post-order)

- $dp[4]=0,\ dp[5]=0,\ dp[3]=0$
- $subtree[4]=1,\ subtree[5]=1,\ subtree[3]=1$
- $subtree[2]=1+1+1=3,\ dp[2]=dp[4]+dp[5]+subtree[4]+subtree[5]=0+0+1+1=2$
- $subtree[1]=1+3+1=5,\ dp[1]=dp[2]+dp[3]+subtree[2]+subtree[3]=2+0+3+1=6$

So $res[1]=6$.

Step 2 (Reroot)

- $res[2]=res[1]+(5-2\times3)=6-1=5$
- $res[3]=res[1]+(5-2\times1)=6+3=9$
- $res[4]=res[2]+(5-2\times1)=5+3=8$
- $res[5]=res[2]+(5-2\times1)=5+3=8$

✅ Final:

- $res[1]=6$, $res[2]=5$, $res[3]=9$, $res[4]=8$, $res[5]=8$

#### Tiny Code (Easy Version)

C

```c
#include <stdio.h>
#define MAXN 100

int n;
int adj[MAXN][MAXN], deg[MAXN];
int subtree[MAXN], dp[MAXN], res[MAXN];

void dfs1(int u, int p) {
    subtree[u] = 1;
    dp[u] = 0;
    for (int i = 0; i < deg[u]; i++) {
        int v = adj[u][i];
        if (v == p) continue;
        dfs1(v, u);
        subtree[u] += subtree[v];
        dp[u] += dp[v] + subtree[v];
    }
}

void dfs2(int u, int p) {
    res[u] = dp[u];
    for (int i = 0; i < deg[u]; i++) {
        int v = adj[u][i];
        if (v == p) continue;
        dp[v] = dp[u] + (n - 2 * subtree[v]);
        dfs2(v, u);
    }
}

int main() {
    n = 5;
    int edges[][2] = {{1,2},{1,3},{2,4},{2,5}};
    for (int i = 0; i < 4; i++) {
        int a = edges[i][0], b = edges[i][1];
        adj[a][deg[a]++] = b;
        adj[b][deg[b]++] = a;
    }
    dfs1(1, -1);
    dfs2(1, -1);
    for (int i = 1; i <= n; i++)
        printf("Sum of distances from %d = %d\n", i, res[i]);
}
```

Python

```python
from collections import defaultdict

g = defaultdict(list)
edges = [(1,2),(1,3),(2,4),(2,5)]
for u,v in edges:
    g[u].append(v)
    g[v].append(u)

n = 5
dp = {i:0 for i in range(1,n+1)}
sub = {i:1 for i in range(1,n+1)}
res = {}

def dfs1(u,p):
    sub[u]=1
    dp[u]=0
    for v in g[u]:
        if v==p: continue
        dfs1(v,u)
        sub[u]+=sub[v]
        dp[u]+=dp[v]+sub[v]

def dfs2(u,p):
    res[u]=dp[u]
    for v in g[u]:
        if v==p: continue
        dp[v]=dp[u]+(n-2*sub[v])
        dfs2(v,u)

dfs1(1,-1)
dfs2(1,-1)
print(res)
```

#### Why It Matters

- Computes sum of distances for every node in linear time
- A foundational rerooting example, applies to many other metrics (sums, products, min/max)
- Extensible to weighted edges, directed trees, and centroid decomposition
- Useful in graph analysis, network latency, tree balancing, dynamic centers

#### A Gentle Proof (Why It Works)

Each reroot step adjusts the sum of distances by accounting for nodes that become closer or farther:

- Nodes in the new root's subtree ($subtree[v]$): distances decrease by 1
- Others ($n - subtree[v]$): distances increase by 1

So:

$$
res[v] = res[u] + (n - 2 \times subtree[v])
$$

By induction across edges, each node gets correct total distance.

#### Try It Yourself

1. Extend to weighted edges.
2. Compute average distance per node.
3. Combine with centroid finding (node minimizing $res[u]$).
4. Visualize change in $res$ as root slides.
5. Adapt for directed rooted trees.

#### Test Cases

| Tree  | Node | Result |
| ----- | ---- | ------ |
| 1–2–3 | 1    | 3      |
| 1–2–3 | 2    | 2      |
| 1–2–3 | 3    | 3      |

#### Complexity

- Time: $O(n)$
- Space: $O(n)$

Distance Sum Rerooting shows the beauty of symmetry in trees, move the root, update the world, keep it all in balance.

### 479 Tree Coloring DP

Tree Coloring DP is a versatile pattern for solving coloring and labeling problems on trees under local constraints, for example, counting how many valid colorings exist when adjacent nodes cannot share a color, or minimizing cost under adjacency restrictions. It combines state definition with child aggregation, forming one of the most common templates in competitive programming.

#### What Problem Are We Solving?

Given a tree with $n$ nodes, we want to color each node with one of $k$ colors so that no two adjacent nodes share the same color, and count the number of valid colorings.

Formally, find the number of ways to assign $color[u] \in {1,2,\dots,k}$
such that for every edge $(u,v)$, $color[u] \ne color[v]$.

We can also generalize:

- Weighted versions (cost per color)
- Restricted versions (pre-colored nodes)
- Modular counting ($\bmod\ 10^9+7$)

Here, we'll solve the basic unweighted counting version.

#### How Does It Work (Plain Language)

We do a rooted DP where each node decides its color and multiplies the valid combinations from its children.

If a node $u$ is colored $c$, then each child $v$ can take any color except $c$.

So for each node:

$$
dp[u][c] = \prod_{v \in children(u)} \sum_{\substack{c' = 1 \ c' \ne c}}^{k} dp[v][c']
$$

Finally, the total count is:

$$
\text{Answer} = \sum_{c=1}^{k} dp[root][c]
$$

Because the tree is acyclic, we can safely combine subtrees without overcounting.

#### Step-by-Step Example

Tree:

```
    1
   / \
  2   3
```

$k=3$ colors (1,2,3)

Start from leaves:

- For leaf node $v$, $dp[v][c] = 1$ for all $c \in {1,2,3}$ (any color works)

Now node 2 and 3:

- $dp[2] = dp[3] = [1,1,1]$

At node 1:

- $dp[1][1] = \prod_{child} \sum_{c' \ne 1} dp[child][c'] = (1+1)*(1+1)=4$
- $dp[1][2] = (1+1)*(1+1)=4$
- $dp[1][3] = (1+1)*(1+1)=4$

Total = $4+4+4 = 12$ valid colorings

Manual check: each of 3 colors for node 1 × 2 choices per child × 2 children = 12 ✅

#### Tiny Code (Easy Version)

C

```c
#include <stdio.h>
#define MAXN 100
#define MAXK 10
#define MOD 1000000007

int n, k;
int adj[MAXN][MAXN], deg[MAXN];
long long dp[MAXN][MAXK+1];

void dfs(int u, int p) {
    for (int c = 1; c <= k; c++) dp[u][c] = 1;
    for (int i = 0; i < deg[u]; i++) {
        int v = adj[u][i];
        if (v == p) continue;
        dfs(v, u);
        for (int c = 1; c <= k; c++) {
            long long sum = 0;
            for (int c2 = 1; c2 <= k; c2++) {
                if (c2 == c) continue;
                sum = (sum + dp[v][c2]) % MOD;
            }
            dp[u][c] = (dp[u][c] * sum) % MOD;
        }
    }
}

int main() {
    n = 3; k = 3;
    int edges[][2] = {{1,2},{1,3}};
    for (int i = 0; i < 2; i++) {
        int a = edges[i][0], b = edges[i][1];
        adj[a][deg[a]++] = b;
        adj[b][deg[b]++] = a;
    }
    dfs(1, -1);
    long long ans = 0;
    for (int c = 1; c <= k; c++) ans = (ans + dp[1][c]) % MOD;
    printf("Total colorings: %lld\n", ans);
}
```

Python

```python
from collections import defaultdict

MOD = 109 + 7
n, k = 3, 3
edges = [(1,2),(1,3)]

g = defaultdict(list)
for u,v in edges:
    g[u].append(v)
    g[v].append(u)

dp = {}

def dfs(u, p):
    dp[u] = [1]*(k+1)
    for v in g[u]:
        if v == p: continue
        dfs(v, u)
        new = [0]*(k+1)
        for c in range(1, k+1):
            total = 0
            for c2 in range(1, k+1):
                if c2 == c: continue
                total = (total + dp[v][c2]) % MOD
            new[c] = (dp[u][c] * total) % MOD
        dp[u] = new

dfs(1, -1)
ans = sum(dp[1][1:]) % MOD
print("Total colorings:", ans)
```

#### Why It Matters

- Solves coloring, labeling, and assignment problems on trees
- Foundation for constraint satisfaction DPs
- Extensible to weighted, modular, and partial pre-colored versions
- Appears in graph theory, combinatorics, and tree-structured probabilistic models

With small tweaks, it becomes:

- Minimum-cost coloring (replace `+` with `min`)
- Constraint coloring (prune invalid colors)
- Modular counting (for combinatorics)

#### A Gentle Proof (Why It Works)

By induction on tree height:

- Base case: Leaf node $u$: $dp[u][c]=1$ (can take any color)
- Inductive step: Suppose each child $v$ has computed correct $dp[v][c']$.
  For node $u$ colored $c$, all children $v$ must choose colors $c' \ne c$.
  Summing and multiplying ensures we count all valid combinations.

No overlap or omission occurs because trees have no cycles.

#### Try It Yourself

1. Add modular constraint (e.g. $k=10^5$).
2. Extend to pre-colored nodes: fix certain $dp[u][c] = 0/1$.
3. Modify recurrence for weighted coloring (cost per color).
4. Optimize with prefix-suffix products for large $k$.
5. Apply to binary tree coloring with parity constraints.

#### Test Cases

| Tree           | n | k | Result |
| -------------- | - | - | ------ |
| 1–2            | 2 | 2 | 2      |
| 1–2–3          | 3 | 3 | 12     |
| Star 1–{2,3,4} | 4 | 3 | 24     |

#### Complexity

- Time: $O(n \times k^2)$
- Space: $O(n \times k)$

Tree Coloring DP is your combinatorial paintbrush, define local rules, traverse once, and color the whole forest with logic.

### 480 Binary Search on Tree DP

Binary Search on Tree DP is a hybrid strategy combining tree dynamic programming with binary search over an answer space. It's especially useful when the feasibility of a condition is monotonic, for example, when asking "is there a subtree/path satisfying constraint X under threshold T?" and the answer changes from false → true as T increases.

#### What Problem Are We Solving?

Given a tree with weights or values on nodes or edges, we want to find a minimum (or maximum) threshold $T$ such that a property holds, e.g.:

- Longest path with all edge weights ≤ $T$
- Smallest $T$ such that there exists a subtree of sum ≥ $S$
- Minimal limit where a valid DP state becomes achievable

We binary search over $T$, and for each guess, we run a DP on the tree to check if the condition is satisfied.

#### How Does It Work (Plain Language)

1. Identify a monotonic property, one that, once true, stays true (or once false, stays false).
2. Define a check(T) function using Tree DP that returns whether the property holds.
3. Apply binary search over $T$ to find the smallest (or largest) value satisfying the condition.

#### Example: Longest Path Under Limit

We're given a weighted tree with edge weights $w(u,v)$.
Find the maximum path length such that all edges ≤ T.
We want the minimum T for which path length ≥ L.

Steps:

1. Binary search over $T$
2. For each $T$, build a subgraph of edges ≤ $T$
3. Run DP on tree (e.g. diameter DP) to check if a path of length ≥ L exists

#### DP Design

We use a DFS-based DP that computes, for each node:

$$
dp[u] = \text{length of longest downward path under } T
$$

and combine two best child paths to check if the diameter ≥ L.

#### Tiny Code (Feasibility Check)

C

```c
#include <stdio.h>
#include <string.h>
#define MAXN 100
#define INF 1000000000

int n, L;
int adj[MAXN][MAXN], w[MAXN][MAXN], deg[MAXN];
int best;

int dfs(int u, int p, int T) {
    int top1 = 0, top2 = 0;
    for (int i = 0; i < deg[u]; i++) {
        int v = adj[u][i];
        if (v == p || w[u][v] > T) continue;
        int len = dfs(v, u, T) + 1;
        if (len > top1) { top2 = top1; top1 = len; }
        else if (len > top2) top2 = len;
    }
    if (top1 + top2 >= L) best = 1;
    return top1;
}

int check(int T) {
    best = 0;
    dfs(1, -1, T);
    return best;
}

int main() {
    n = 4; L = 3;
    int edges[3][3] = {{1,2,3},{2,3,5},{3,4,7}};
    for (int i = 0; i < 3; i++) {
        int a=edges[i][0], b=edges[i][1], c=edges[i][2];
        adj[a][deg[a]] = b; w[a][b] = c; deg[a]++;
        adj[b][deg[b]] = a; w[b][a] = c; deg[b]++;
    }
    int lo = 0, hi = 10, ans = -1;
    while (lo <= hi) {
        int mid = (lo + hi)/2;
        if (check(mid)) { ans = mid; hi = mid - 1; }
        else lo = mid + 1;
    }
    printf("Minimum T: %d\n", ans);
}
```

Python

```python
from collections import defaultdict

g = defaultdict(list)
edges = [(1,2,3),(2,3,5),(3,4,7)]
n, L = 4, 3
for u,v,w in edges:
    g[u].append((v,w))
    g[v].append((u,w))

def dfs(u, p, T):
    top1 = top2 = 0
    global ok
    for v,w in g[u]:
        if v == p or w > T: continue
        length = dfs(v, u, T) + 1
        if length > top1:
            top2 = top1
            top1 = length
        elif length > top2:
            top2 = length
    if top1 + top2 >= L:
        ok = True
    return top1

def check(T):
    global ok
    ok = False
    dfs(1, -1, T)
    return ok

lo, hi = 0, 10
ans = -1
while lo <= hi:
    mid = (lo + hi)//2
    if check(mid):
        ans = mid
        hi = mid - 1
    else:
        lo = mid + 1
print("Minimum T:", ans)
```

#### Why It Matters

- Many threshold optimization problems rely on binary search + DP
- Ideal when cost / limit interacts with tree-based structure
- Useful in network design, path constraints, tree queries, game theory

Examples:

- Smallest edge weight for connectivity
- Minimal node cost for subtree property
- Path feasibility under resource constraint

#### A Gentle Proof (Why It Works)

If the property is monotonic, binary search guarantees correctness:

- If a condition holds at $T$, it holds at all $T' > T$
- So, we can search for the smallest satisfying $T$

Tree DP correctly checks feasibility because it enumerates all root-to-leaf and child-to-child paths under threshold $T$.

#### Try It Yourself

1. Modify to maximize value (reverse monotonicity).
2. Replace edge constraint with node value ≤ T.
3. Use DP to count paths, not just check existence.
4. Apply to maximum subtree sum under bound.
5. Extend to k-colored constraints (binary search over cost).

#### Test Cases

| n | L | Edges                 | Output |
| - | - | --------------------- | ------ |
| 4 | 3 | (1-2:3, 2-3:5, 3-4:7) | 5      |
| 3 | 2 | (1-2:1, 2-3:2)        | 2      |

#### Complexity

- DP per check: $O(n)$
- Binary search: $\log(\text{range})$
- Total: $O(n\log C)$ where $C$ is max edge weight

Binary Search on Tree DP bridges feasibility logic and optimization, use it whenever monotonic thresholds and tree states meet.

# Section 49. DP Reconstruction and Traceback

### 481 Reconstruct LCS

Reconstructing the Longest Common Subsequence (LCS) means not just computing its length, but tracing back the actual sequence that two strings share in order. This step turns abstract DP tables into tangible answers, a common need in bioinformatics, text diffing, and alignment tasks.

#### What Problem Are We Solving?

Given two sequences $A$ (length $n$) and $B$ (length $m$), find the longest subsequence common to both (not necessarily contiguous).

We first build a DP table for LCS length:

$$
dp[i][j] =
\begin{cases}
0, & \text{if } i = 0 \text{ or } j = 0,\\
dp[i-1][j-1] + 1, & \text{if } A[i-1] = B[j-1],\\
\max(dp[i-1][j],\ dp[i][j-1]), & \text{otherwise.}
\end{cases}
$$


Then we trace back from $dp[n][m]$ to reconstruct the sequence.

#### How Does It Work (Plain Language)

1. Compute the LCS length table using standard DP.
2. Start from the bottom-right corner ($dp[n][m]$).
3. Trace back:

   * If $A[i-1] == B[j-1]$: add that character and move diagonally ($i-1, j-1$)
   * Else move to the direction with larger dp value
4. Reverse the collected sequence.

#### Example

Let $A = \text{"ABCBDAB"}$, $B = \text{"BDCABA"}$

DP length table leads to result "BCBA".

#### Tiny Code

C

```c
#include <stdio.h>
#include <string.h>

#define MAX 100

int dp[MAX][MAX];
char A[MAX], B[MAX];
char lcs[MAX];

int main() {
    scanf("%s %s", A, B);
    int n = strlen(A), m = strlen(B);
    for (int i = 1; i <= n; i++)
        for (int j = 1; j <= m; j++)
            if (A[i-1] == B[j-1])
                dp[i][j] = dp[i-1][j-1] + 1;
            else
                dp[i][j] = dp[i-1][j] > dp[i][j-1] ? dp[i-1][j] : dp[i][j-1];

    // Reconstruct
    int i = n, j = m, k = dp[n][m];
    lcs[k] = '\0';
    while (i > 0 && j > 0) {
        if (A[i-1] == B[j-1]) {
            lcs[--k] = A[i-1];
            i--; j--;
        } else if (dp[i-1][j] >= dp[i][j-1])
            i--;
        else
            j--;
    }
    printf("LCS: %s\n", lcs);
}
```

Python

```python
def reconstruct_lcs(A, B):
    n, m = len(A), len(B)
    dp = [[0]*(m+1) for _ in range(n+1)]
    for i in range(1, n+1):
        for j in range(1, m+1):
            if A[i-1] == B[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])

    i, j = n, m
    res = []
    while i > 0 and j > 0:
        if A[i-1] == B[j-1]:
            res.append(A[i-1])
            i -= 1
            j -= 1
        elif dp[i-1][j] >= dp[i][j-1]:
            i -= 1
        else:
            j -= 1
    return ''.join(reversed(res))

A = input("A: ")
B = input("B: ")
print("LCS:", reconstruct_lcs(A, B))
```

#### Why It Matters

- Core of diff, merge, and DNA alignment tools
- Demonstrates how DP solutions can reconstruct actual solutions, not just counts
- Foundation for traceback techniques in many DP problems

#### A Gentle Proof (Why It Works)

At each step:

- If $A[i-1] = B[j-1]$, the character must belong to the LCS, so we include it and move diagonally.
- Otherwise, the longer LCS lies in the direction of the greater dp value, hence we follow that path.
  By starting from $dp[n][m]$ and moving backward, we guarantee each included character is part of at least one optimal solution.

Since we collect in reverse order, reversing yields the correct sequence.

#### Try It Yourself

1. Trace the LCS of "ABCBDAB" and "BDCABA" by hand.
2. Modify to find one of all possible LCSs (handle ties).
3. Extend for case-insensitive comparison.
4. Adapt code to return indices of matching characters.
5. Visualize path arrows in DP table.

#### Test Cases

| A         | B         | LCS    |
| --------- | --------- | ------ |
| "ABCBDAB" | "BDCABA"  | "BCBA" |
| "AGGTAB"  | "GXTXAYB" | "GTAB" |
| "AXYT"    | "AYZX"    | "AY"   |

#### Complexity

- Time: $O(nm)$
- Space: $O(nm)$ (can optimize to $O(\min(n,m))$ for length only)

Reconstruct LCS is your first step from number tables to actual solutions, bridging reasoning and reality.

### 482 Reconstruct LIS

Reconstructing the Longest Increasing Subsequence (LIS) means finding not just the length of the longest increasing sequence, but the actual subsequence. This is a classic step beyond computing DP values, it's about tracing *how* we got there.

#### What Problem Are We Solving?

Given a sequence of numbers $A = [a_1, a_2, \dots, a_n]$, we want to find the longest strictly increasing subsequence. The DP version computes LIS length in $O(n^2)$, but here we focus on reconstruction.

We define:

$$
dp[i] = \text{length of LIS ending at } i
$$

and a parent array to track predecessors:

$$
parent[i] = \text{index of previous element in LIS ending at } i
$$

Finally, we backtrack from the index of the maximum $dp[i]$ to recover the sequence.

#### How Does It Work (Plain Language)

1. Compute dp[i]: longest LIS ending at $A[i]$.
2. Track parent[i]: where this sequence came from.
3. Find max length index, call it `best`.
4. Backtrack using `parent` array.
5. Reverse the reconstructed list.

#### Example

For $A = [10, 22, 9, 33, 21, 50, 41, 60]$

We get:

- `dp = [1, 2, 1, 3, 2, 4, 4, 5]`
- LIS length = 5
- Sequence = `[10, 22, 33, 50, 60]`

#### Tiny Code

C

```c
#include <stdio.h>

int main() {
    int A[] = {10, 22, 9, 33, 21, 50, 41, 60};
    int n = sizeof(A) / sizeof(A[0]);
    int dp[n], parent[n];

    for (int i = 0; i < n; i++) {
        dp[i] = 1;
        parent[i] = -1;
        for (int j = 0; j < i; j++) {
            if (A[j] < A[i] && dp[j] + 1 > dp[i]) {
                dp[i] = dp[j] + 1;
                parent[i] = j;
            }
        }
    }

    // Find index of LIS
    int best = 0;
    for (int i = 1; i < n; i++)
        if (dp[i] > dp[best]) best = i;

    // Reconstruct LIS
    int lis[100], len = 0;
    for (int i = best; i != -1; i = parent[i])
        lis[len++] = A[i];

    printf("LIS: ");
    for (int i = len - 1; i >= 0; i--)
        printf("%d ", lis[i]);
    printf("\n");
}
```

Python

```python
def reconstruct_lis(A):
    n = len(A)
    dp = [1]*n
    parent = [-1]*n

    for i in range(n):
        for j in range(i):
            if A[j] < A[i] and dp[j] + 1 > dp[i]:
                dp[i] = dp[j] + 1
                parent[i] = j

    best = max(range(n), key=lambda i: dp[i])

    res = []
    while best != -1:
        res.append(A[best])
        best = parent[best]
    return res[::-1]

A = [10, 22, 9, 33, 21, 50, 41, 60]
print("LIS:", reconstruct_lis(A))
```

#### Why It Matters

- Transforms abstract DP into real sequence output
- Useful in scheduling, stock analysis, subsequence pattern recognition
- Teaches traceback technique with parent tracking, reused across many problems

#### A Gentle Proof (Why It Works)

By definition, $dp[i]$ records the LIS length ending at $i$. Whenever we update $dp[i] = dp[j] + 1$, we've extended the best LIS ending at $j$. Recording `parent[i] = j` lets us reconstruct that path.

The element with the maximum $dp[i]$ must end one LIS, and by backtracking through parents, we trace exactly one valid increasing subsequence achieving the max length.

#### Try It Yourself

1. Trace LIS reconstruction for $[3, 10, 2, 1, 20]$
2. Modify to return all LIS sequences (handle equal-length ties).
3. Adapt code for non-decreasing LIS.
4. Combine with binary search LIS for $O(n \log n)$ + parent tracking.
5. Visualize `parent` links as arrows between indices.

#### Test Cases

| Input                           | LIS                  | Length |
| ------------------------------- | -------------------- | ------ |
| [10, 22, 9, 33, 21, 50, 41, 60] | [10, 22, 33, 50, 60] | 5      |
| [3, 10, 2, 1, 20]               | [3, 10, 20]          | 3      |
| [50, 3, 10, 7, 40, 80]          | [3, 7, 40, 80]       | 4      |

#### Complexity

- Time: $O(n^2)$
- Space: $O(n)$

Reconstruct LIS is a gentle bridge from computing a number to seeing the story behind it, each element tracing its lineage through the DP table.

### 483 Reconstruct Knapsack

Reconstructing the Knapsack solution means identifying which items form the optimal value, not just knowing the maximum value. This is the difference between understanding *what's possible* and *what to choose*.

#### What Problem Are We Solving?

Given:

- $n$ items with values $v[i]$ and weights $w[i]$
- Capacity $W$

We want:

- Maximize total value without exceeding $W$
- Recover the chosen items

The 0/1 knapsack DP table is defined as:

$$
dp[i][w] =
\begin{cases}
0, & \text{if } i = 0 \text{ or } w = 0,\\
dp[i-1][w], & \text{if } w_i > w,\\
\max(dp[i-1][w],\ dp[i-1][w - w_i] + v_i), & \text{otherwise.}
\end{cases}
$$


To reconstruct, we backtrack from $dp[n][W]$:

- If $dp[i][w] \neq dp[i-1][w]$, then item $i$ was included
- Subtract its weight, move to $i-1$

#### How Does It Work (Plain Language)

1. Build the standard 0/1 knapsack DP table.
2. Start from bottom-right corner $(n, W)$.
3. Compare $dp[i][w]$ vs $dp[i-1][w]$:

   * If different, include item $i$, update $w -= w_i$.
4. Continue until $i=0$.
5. Reverse selected items for correct order.

#### Example

Let:

| Item | Value | Weight |
| ---- | ----- | ------ |
| 1    | 60    | 10     |
| 2    | 100   | 20     |
| 3    | 120   | 30     |

Capacity $W = 50$

Optimal value = 220
Chosen items = {2, 3}

#### Tiny Code

C

```c
#include <stdio.h>

#define N 4
#define W 50

int main() {
    int val[] = {0, 60, 100, 120};
    int wt[] = {0, 10, 20, 30};
    int dp[N][W+1];

    for (int i = 0; i < N; i++)
        for (int w = 0; w <= W; w++)
            dp[i][w] = 0;

    for (int i = 1; i < N; i++) {
        for (int w = 1; w <= W; w++) {
            if (wt[i] <= w) {
                int include = val[i] + dp[i-1][w-wt[i]];
                int exclude = dp[i-1][w];
                dp[i][w] = include > exclude ? include : exclude;
            } else dp[i][w] = dp[i-1][w];
        }
    }

    printf("Max Value: %d\n", dp[N-1][W]);
    printf("Items Taken: ");

    int w = W;
    for (int i = N-1; i > 0; i--) {
        if (dp[i][w] != dp[i-1][w]) {
            printf("%d ", i);
            w -= wt[i];
        }
    }
    printf("\n");
}
```

Python

```python
def reconstruct_knapsack(values, weights, W):
    n = len(values)
    dp = [[0]*(W+1) for _ in range(n+1)]

    for i in range(1, n+1):
        for w in range(W+1):
            if weights[i-1] <= w:
                dp[i][w] = max(dp[i-1][w],
                               values[i-1] + dp[i-1][w - weights[i-1]])
            else:
                dp[i][w] = dp[i-1][w]

    # Reconstruction
    w = W
    chosen = []
    for i in range(n, 0, -1):
        if dp[i][w] != dp[i-1][w]:
            chosen.append(i-1)
            w -= weights[i-1]

    chosen.reverse()
    return dp[n][W], chosen

values = [60, 100, 120]
weights = [10, 20, 30]
W = 50
value, items = reconstruct_knapsack(values, weights, W)
print("Max value:", value)
print("Items:", items)
```

#### Why It Matters

- Turns value tables into actionable decisions
- Essential in optimization problems (resource allocation, budgeting)
- Demonstrates traceback logic from DP matrix

#### A Gentle Proof (Why It Works)

Every $dp[i][w]$ represents the best value using the first $i$ items under capacity $w$.
If $dp[i][w] \neq dp[i-1][w]$, item $i$ was critical in improving value, so it must be included.
Reducing $w$ by its weight and moving up repeats the same logic, tracing one optimal solution.

#### Try It Yourself

1. Modify for multiple optimal solutions (store parent paths).
2. Implement space-optimized DP and reconstruct with backtracking info.
3. Adapt for unbounded knapsack (reuse items).
4. Add total weight output.
5. Visualize reconstruction arrows from $dp[n][W]$.

#### Test Cases

| Values         | Weights      | W  | Max Value | Items  |
| -------------- | ------------ | -- | --------- | ------ |
| [60, 100, 120] | [10, 20, 30] | 50 | 220       | [1, 2] |
| [10, 20, 30]   | [1, 1, 1]    | 2  | 50        | [1, 2] |
| [5, 4, 6, 3]   | [2, 3, 4, 5] | 5  | 7         | [0, 1] |

#### Complexity

- Time: $O(nW)$
- Space: $O(nW)$ (can reduce to $O(W)$ for value-only)

Reconstruction transforms knapsack from a math result to a real-world selection list, revealing *which* items make the optimum possible.

### 484 Edit Distance Alignment

Edit Distance tells us *how different* two strings are, alignment reconstruction shows *exactly where* they differ. By tracing the path of operations (insert, delete, substitute), we can visualize the full transformation.

#### What Problem Are We Solving?

Given two strings $A$ (length $n$) and $B$ (length $m$), compute not only the edit distance but also the alignment that transforms $A$ into $B$ using the minimum number of operations.

We define:

$$
dp[i][j] =
\begin{cases}
0, & \text{if } i = 0 \text{ and } j = 0,\\
i, & \text{if } j = 0,\\
j, & \text{if } i = 0,\\[6pt]
\displaystyle
\min\!\begin{cases}
dp[i-1][j] + 1, & \text{deletion},\\
dp[i][j-1] + 1, & \text{insertion},\\
dp[i-1][j-1] + \text{cost}(A[i-1], B[j-1]), & \text{replace or match.}
\end{cases}
\end{cases}
$$


Then, we trace back from $dp[n][m]$ to list operations in reverse.

#### How Does It Work (Plain Language)

1. Build standard Levenshtein DP table.
2. Start from $dp[n][m]$.
3. Move:

   * Diagonal: match or replace
   * Up: delete
   * Left: insert
4. Record operation at each move.
5. Reverse the sequence for final alignment.

#### Example

Let $A=\text{"kitten"}$, $B=\text{"sitting"}$.

Operations:

| Step     | Action    | Result           |
| -------- | --------- | ---------------- |
| k → k    | match     | kitten / sitting |
| i → i    | match     | kitten / sitting |
| t → t    | match     | kitten / sitting |
| t → t    | match     | kitten / sitting |
| e → i    | replace   | kitti n          |
| insert g | insertion | kitting          |

Edit distance = 3 (replace e→i, insert g, insert n)

#### Tiny Code

C

```c
#include <stdio.h>
#include <string.h>

#define MAX 100

int dp[MAX][MAX];

int min3(int a, int b, int c) {
    return a < b ? (a < c ? a : c) : (b < c ? b : c);
}

int main() {
    char A[MAX], B[MAX];
    scanf("%s %s", A, B);
    int n = strlen(A), m = strlen(B);

    for (int i = 0; i <= n; i++) dp[i][0] = i;
    for (int j = 0; j <= m; j++) dp[0][j] = j;

    for (int i = 1; i <= n; i++) {
        for (int j = 1; j <= m; j++) {
            int cost = (A[i-1] == B[j-1]) ? 0 : 1;
            dp[i][j] = min3(dp[i-1][j] + 1,
                            dp[i][j-1] + 1,
                            dp[i-1][j-1] + cost);
        }
    }

    printf("Edit Distance: %d\n", dp[n][m]);
    printf("Operations:\n");

    int i = n, j = m;
    while (i > 0 || j > 0) {
        if (i > 0 && j > 0 && dp[i][j] == dp[i-1][j-1] && A[i-1] == B[j-1]) {
            printf("Match %c\n", A[i-1]);
            i--; j--;
        }
        else if (i > 0 && j > 0 && dp[i][j] == dp[i-1][j-1] + 1) {
            printf("Replace %c -> %c\n", A[i-1], B[j-1]);
            i--; j--;
        }
        else if (i > 0 && dp[i][j] == dp[i-1][j] + 1) {
            printf("Delete %c\n", A[i-1]);
            i--;
        }
        else {
            printf("Insert %c\n", B[j-1]);
            j--;
        }
    }
}
```

Python

```python
def edit_distance_alignment(A, B):
    n, m = len(A), len(B)
    dp = [[0]*(m+1) for _ in range(n+1)]

    for i in range(n+1):
        dp[i][0] = i
    for j in range(m+1):
        dp[0][j] = j

    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = 0 if A[i-1] == B[j-1] else 1
            dp[i][j] = min(dp[i-1][j] + 1,
                           dp[i][j-1] + 1,
                           dp[i-1][j-1] + cost)

    i, j = n, m
    ops = []
    while i > 0 or j > 0:
        if i > 0 and j > 0 and A[i-1] == B[j-1]:
            ops.append(f"Match {A[i-1]}")
            i -= 1; j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + 1:
            ops.append(f"Replace {A[i-1]} -> {B[j-1]}")
            i -= 1; j -= 1
        elif i > 0 and dp[i][j] == dp[i-1][j] + 1:
            ops.append(f"Delete {A[i-1]}")
            i -= 1
        else:
            ops.append(f"Insert {B[j-1]}")
            j -= 1

    return dp[n][m], list(reversed(ops))

dist, ops = edit_distance_alignment("kitten", "sitting")
print("Edit Distance:", dist)
print("Alignment:")
for op in ops:
    print(op)
```

#### Why It Matters

- Powers diff, spell checkers, and DNA sequence alignment
- Essential for text transformation visualization
- Reinforces traceback logic from DP tables

#### A Gentle Proof (Why It Works)

Each $dp[i][j]$ is minimal over three possibilities:

- Insertion: adding $B[j-1]$
- Deletion: removing $A[i-1]$
- Replacement: changing $A[i-1]$ to $B[j-1]$

The optimal path through the DP grid, via moves left, up, or diagonal, exactly records the sequence of minimal edits. Reversing the trace yields the transformation.

#### Try It Yourself

1. Compute alignment for "intention" → "execution"
2. Add operation count summary (insertions, deletions, replacements)
3. Visualize grid with arrows (↑, ←, ↖)
4. Modify cost: substitution = 2, insertion/deletion = 1
5. Return both alignment string and operation list

#### Test Cases

| A      | B        | Distance | Alignment (Ops)                 |
| ------ | -------- | -------- | ------------------------------- |
| kitten | sitting  | 3        | Replace e→i, Insert n, Insert g |
| sunday | saturday | 3        | Insert a, Insert t, Replace n→r |
| horse  | ros      | 3        | Delete h, Replace o→r, Delete e |

#### Complexity

- Time: $O(nm)$
- Space: $O(nm)$ (can reduce to $O(\min(n,m))$ without reconstruction)

Edit Distance Alignment transforms a distance metric into a step-by-step story, showing exactly how one word becomes another.

### 485 Matrix Chain Parentheses

Matrix Chain Multiplication gives us the minimum number of multiplications, but reconstruction tells us how to parenthesize, the *order* of multiplication that achieves that cost. Without this step, we know the cost, but not the recipe.

#### What Problem Are We Solving?

Given a sequence of matrices $A_1, A_2, \dots, A_n$ with dimensions
$p_0 \times p_1, p_1 \times p_2, \dots, p_{n-1} \times p_n$,
we want to determine the optimal parenthesization that minimizes scalar multiplications.

The cost DP is:

$$
dp[i][j] =
\begin{cases}
0 & \text{if } i = j \
\min_{i \le k < j} (dp[i][k] + dp[k+1][j] + p_{i-1} \cdot p_k \cdot p_j)
\end{cases}
$$

To reconstruct the solution, we maintain a split table $split[i][j]$ indicating the index $k$ where the optimal split occurs.

#### How Does It Work (Plain Language)

1. Compute cost table using bottom-up DP.
2. Track split point $k$ at each subproblem.
3. Recurse:

   * Base: if $i==j$, return $A_i$
   * Otherwise: `(` + solve($i$, $k$) + solve($k+1$, $j$) + `)`

This yields the exact parenthesization.

#### Example

Matrix dimensions: $[40, 20, 30, 10, 30]$

There are 4 matrices:

- $A_1: 40\times20$
- $A_2: 20\times30$
- $A_3: 30\times10$
- $A_4: 10\times30$

Optimal order:
$((A_1(A_2A_3))A_4)$
Minimal cost: 26000

#### Tiny Code

C

```c
#include <stdio.h>
#include <limits.h>

#define N 5

int dp[N][N];
int split[N][N];

int min(int a, int b) { return a < b ? a : b; }

void print_paren(int i, int j) {
    if (i == j) {
        printf("A%d", i);
        return;
    }
    printf("(");
    print_paren(i, split[i][j]);
    print_paren(split[i][j] + 1, j);
    printf(")");
}

int main() {
    int p[] = {40, 20, 30, 10, 30};
    int n = 4;

    for (int i = 1; i <= n; i++) dp[i][i] = 0;

    for (int len = 2; len <= n; len++) {
        for (int i = 1; i <= n - len + 1; i++) {
            int j = i + len - 1;
            dp[i][j] = INT_MAX;
            for (int k = i; k < j; k++) {
                int cost = dp[i][k] + dp[k+1][j] + p[i-1]*p[k]*p[j];
                if (cost < dp[i][j]) {
                    dp[i][j] = cost;
                    split[i][j] = k;
                }
            }
        }
    }

    printf("Minimum cost: %d\n", dp[1][n]);
    printf("Optimal order: ");
    print_paren(1, n);
    printf("\n");
}
```

Python

```python
def matrix_chain_order(p):
    n = len(p) - 1
    dp = [[0]* (n+1) for _ in range(n+1)]
    split = [[0]* (n+1) for _ in range(n+1)]

    for l in range(2, n+1):
        for i in range(1, n-l+2):
            j = i + l - 1
            dp[i][j] = float('inf')
            for k in range(i, j):
                cost = dp[i][k] + dp[k+1][j] + p[i-1]*p[k]*p[j]
                if cost < dp[i][j]:
                    dp[i][j] = cost
                    split[i][j] = k

    def build(i, j):
        if i == j: return f"A{i}"
        k = split[i][j]
        return f"({build(i, k)}{build(k+1, j)})"

    return dp[1][n], build(1, n)

p = [40, 20, 30, 10, 30]
cost, order = matrix_chain_order(p)
print("Min Cost:", cost)
print("Order:", order)
```

#### Why It Matters

- Converts abstract cost table into concrete plan
- Foundation of query optimization, compiler expression parsing
- Shows how split tracking yields human-readable structure

#### A Gentle Proof (Why It Works)

At each subchain $(i, j)$, DP tries all $k$ splits.
The chosen $k$ minimizing cost is stored in `split[i][j]`.
By recursively applying these stored splits, we follow the same decision tree that generated the minimal cost.
Thus reconstruction yields the exact sequence of multiplications.

#### Try It Yourself

1. Try $p=[10, 20, 30, 40, 30]$ and verify order.
2. Add printing of subproblem cost for each pair $(i,j)$.
3. Modify to return tree structure instead of string.
4. Visualize with nested parentheses tree.
5. Extend to show intermediate matrix dimensions at each step.

#### Test Cases

| Dimensions       | Cost  | Parenthesization |
| ---------------- | ----- | ---------------- |
| [40,20,30,10,30] | 26000 | ((A1(A2A3))A4)   |
| [10,20,30]       | 6000  | (A1A2A3)         |
| [10,20,30,40,30] | 30000 | ((A1A2)(A3A4))   |

#### Complexity

- Time: $O(n^3)$
- Space: $O(n^2)$

Matrix Chain Parentheses turns cost minimization into concrete strategy, showing *not just how much*, but *how exactly*.

### 486 Coin Change Reconstruction

In the Coin Change problem, we usually count the minimum coins or total ways. Reconstruction, however, asks: *which exact coins make up the solution?*
This bridges the gap between number answers and actual combinations.

#### What Problem Are We Solving?

Given:

- A set of coin denominations $coins = [c_1, c_2, \dots, c_n]$
- A target sum $S$

We want to:

1. Compute the minimum number of coins needed (classic DP)
2. Reconstruct one optimal combination of coins that achieves $S$

We define:

$$
dp[x] =
\begin{cases}
0, & \text{if } x = 0,\\
1 + \displaystyle\min_{c \le x}\bigl(dp[x - c]\bigr), & \text{if } x > 0.
\end{cases}
$$


And record which coin gave the best solution:

$$
choice[x] = c \text{ that minimizes } dp[x-c]
$$

#### How Does It Work (Plain Language)

1. Build DP array from $0$ to $S$.
2. For each amount $x$, try every coin $c$.
3. Keep track of:

   * The minimum coin count (`dp[x]`)
   * The coin used (`choice[x]`)
4. After filling, trace back from $S$:
   repeatedly subtract `choice[x]` until reaching 0.

#### Example

Coins = [1, 3, 4], Target $S = 6$

DP steps:

| x | dp[x] | choice[x] |
| - | ----- | --------- |
| 0 | 0     | -         |
| 1 | 1     | 1         |
| 2 | 2     | 1         |
| 3 | 1     | 3         |
| 4 | 1     | 4         |
| 5 | 2     | 1         |
| 6 | 2     | 3         |

Optimal combination: [3, 3]

#### Tiny Code

C

```c
#include <stdio.h>
#include <limits.h>

#define MAX 100

int main() {
    int coins[] = {1, 3, 4};
    int n = 3;
    int S = 6;
    int dp[MAX], choice[MAX];

    dp[0] = 0;
    choice[0] = -1;

    for (int i = 1; i <= S; i++) {
        dp[i] = INT_MAX;
        choice[i] = -1;
        for (int j = 0; j < n; j++) {
            int c = coins[j];
            if (c <= i && dp[i-c] + 1 < dp[i]) {
                dp[i] = dp[i-c] + 1;
                choice[i] = c;
            }
        }
    }

    printf("Min coins: %d\n", dp[S]);
    printf("Combination: ");
    int x = S;
    while (x > 0) {
        printf("%d ", choice[x]);
        x -= choice[x];
    }
    printf("\n");
}
```

Python

```python
def coin_change_reconstruct(coins, S):
    dp = [float('inf')] * (S + 1)
    choice = [-1] * (S + 1)
    dp[0] = 0

    for x in range(1, S + 1):
        for c in coins:
            if c <= x and dp[x - c] + 1 < dp[x]:
                dp[x] = dp[x - c] + 1
                choice[x] = c

    if dp[S] == float('inf'):
        return None, []

    comb = []
    while S > 0:
        comb.append(choice[S])
        S -= choice[S]

    return dp[-1], comb

coins = [1, 3, 4]
S = 6
count, comb = coin_change_reconstruct(coins, S)
print("Min coins:", count)
print("Combination:", comb)
```

#### Why It Matters

- Converts abstract DP result into practical plan
- Critical in finance, vending systems, resource allocation
- Reinforces traceback technique for linear DP problems

#### A Gentle Proof (Why It Works)

By definition, $dp[x] = 1 + dp[x-c]$ for optimal $c$.
Thus the optimal last step for $x$ must use coin $choice[x] = c$.
Repeatedly subtracting this $c$ gives a valid sequence ending at $0$.
Each step reduces the problem size while preserving optimality (greedy by DP).

#### Try It Yourself

1. Try $coins=[1,3,4]$, $S=10$
2. Modify to return all optimal combinations (if multiple)
3. Extend for limited coin counts
4. Visualize table $(x, dp[x], choice[x])$
5. Adapt for non-canonical systems (like [1, 3, 5, 7])

#### Test Cases

| Coins      | S  | Min Coins | Combination |
| ---------- | -- | --------- | ----------- |
| [1, 3, 4]  | 6  | 2         | [3, 3]      |
| [1, 2, 5]  | 11 | 3         | [5, 5, 1]   |
| [2, 5, 10] | 7  | ∞         | []          |

#### Complexity

- Time: $O(S \times n)$
- Space: $O(S)$

Coin Change Reconstruction transforms "how many" into "which ones", building not just an answer, but a clear path to it.

### 487 Path Reconstruction DP

Path reconstruction in DP is the art of retracing your steps through a cost or distance table to find the *exact route* that led to the optimal answer. It's not enough to know *how far*, you want to know *how you got there*.

#### What Problem Are We Solving?

Given a grid (or graph) where each cell has a cost, we compute the minimum path cost from a start cell $(0,0)$ to a destination $(n-1, m-1)$ using only right or down moves.
Now, instead of just reporting the minimal cost, we'll reconstruct the path.

We define:

$$
dp[i][j] =
\begin{cases}
grid[0][0], & \text{if } i = 0 \text{ and } j = 0,\\
grid[i][j] + \min\bigl(dp[i-1][j],\ dp[i][j-1]\bigr), & \text{otherwise.}
\end{cases}
$$


We also maintain a parent table `parent[i][j]` to remember whether we came from top or left.

#### How Does It Work (Plain Language)

1. Fill dp[i][j] with the minimum cost to reach each cell.
2. Track the move that led to this cost:

   * If $dp[i][j]$ came from $dp[i-1][j]$, parent = "up"
   * Else parent = "left"
3. Start from destination $(n-1,m-1)$ and backtrack using `parent`.
4. Reverse the reconstructed list for the correct order.

#### Example

Grid:

| 1 | 3 | 1 |
| - | - | - |
| 1 | 5 | 1 |
| 4 | 2 | 1 |

Minimal path sum: 7
Path: $(0,0)\rightarrow(0,1)\rightarrow(0,2)\rightarrow(1,2)\rightarrow(2,2)$

#### Tiny Code

C

```c
#include <stdio.h>
#include <limits.h>

#define N 3
#define M 3

int grid[N][M] = {
    {1, 3, 1},
    {1, 5, 1},
    {4, 2, 1}
};

int dp[N][M];
char parent[N][M]; // 'U' = up, 'L' = left

int min(int a, int b) { return a < b ? a : b; }

int main() {
    dp[0][0] = grid[0][0];

    // First row
    for (int j = 1; j < M; j++) {
        dp[0][j] = dp[0][j-1] + grid[0][j];
        parent[0][j] = 'L';
    }
    // First column
    for (int i = 1; i < N; i++) {
        dp[i][0] = dp[i-1][0] + grid[i][0];
        parent[i][0] = 'U';
    }

    // Fill rest
    for (int i = 1; i < N; i++) {
        for (int j = 1; j < M; j++) {
            if (dp[i-1][j] < dp[i][j-1]) {
                dp[i][j] = dp[i-1][j] + grid[i][j];
                parent[i][j] = 'U';
            } else {
                dp[i][j] = dp[i][j-1] + grid[i][j];
                parent[i][j] = 'L';
            }
        }
    }

    printf("Min path sum: %d\n", dp[N-1][M-1]);

    // Backtrack
    int i = N - 1, j = M - 1;
    int path[100][2], len = 0;
    while (!(i == 0 && j == 0)) {
        path[len][0] = i;
        path[len][1] = j;
        len++;
        if (parent[i][j] == 'U') i--;
        else j--;
    }
    path[len][0] = 0; path[len][1] = 0;
    len++;

    printf("Path: ");
    for (int k = len - 1; k >= 0; k--)
        printf("(%d,%d) ", path[k][0], path[k][1]);
    printf("\n");
}
```

Python

```python
def min_path_sum_path(grid):
    n, m = len(grid), len(grid[0])
    dp = [[0]*m for _ in range(n)]
    parent = [['']*m for _ in range(n)]

    dp[0][0] = grid[0][0]

    for j in range(1, m):
        dp[0][j] = dp[0][j-1] + grid[0][j]
        parent[0][j] = 'L'
    for i in range(1, n):
        dp[i][0] = dp[i-1][0] + grid[i][0]
        parent[i][0] = 'U'

    for i in range(1, n):
        for j in range(1, m):
            if dp[i-1][j] < dp[i][j-1]:
                dp[i][j] = dp[i-1][j] + grid[i][j]
                parent[i][j] = 'U'
            else:
                dp[i][j] = dp[i][j-1] + grid[i][j]
                parent[i][j] = 'L'

    # Backtrack
    path = []
    i, j = n-1, m-1
    while not (i == 0 and j == 0):
        path.append((i, j))
        if parent[i][j] == 'U':
            i -= 1
        else:
            j -= 1
    path.append((0, 0))
    path.reverse()

    return dp[-1][-1], path

grid = [[1,3,1],[1,5,1],[4,2,1]]
cost, path = min_path_sum_path(grid)
print("Min cost:", cost)
print("Path:", path)
```

#### Why It Matters

- Translates numerical DP into navigable routes
- Key in pathfinding, robot navigation, route planning
- Demonstrates parent-pointer technique for 2D grids

#### A Gentle Proof (Why It Works)

By construction, $dp[i][j]$ stores the minimal cost to reach $(i,j)$.
Since each cell depends only on top and left, storing the better source as `parent[i][j]` ensures each step back leads to a valid prefix of an optimal path.
Following parents reconstructs one such optimal path.

#### Try It Yourself

1. Try on a $4\times4$ grid with random costs.
2. Modify to allow diagonal moves.
3. Extend for maximum path sum (change min→max).
4. Visualize path arrows (↑, ←).
5. Adapt for graph shortest path with adjacency matrix.

#### Test Cases

| Grid                      | Result | Path                            |
| ------------------------- | ------ | ------------------------------- |
| [[1,3,1],[1,5,1],[4,2,1]] | 7      | [(0,0),(0,1),(0,2),(1,2),(2,2)] |
| [[1,2,3],[4,5,6]]         | 12     | [(0,0),(0,1),(0,2),(1,2)]       |

#### Complexity

- Time: $O(nm)$
- Space: $O(nm)$

Path Reconstruction DP turns shortest paths into visible journeys, showing every choice that built the optimum.

### 488 Sequence Reconstruction

Sequence Reconstruction is the process of recovering an entire sequence from partial or implicit information, typically from DP tables, prefix relations, or pairwise constraints. It is a bridge between solving a problem and interpreting its answer as a sequence.

#### What Problem Are We Solving?

You often solve DP problems that count or score possible sequences, but what if you need to recover one valid sequence (or even all)?
For example:

1. Given the LIS length, reconstruct one LIS.
2. Given partial orders, reconstruct a sequence that satisfies them.
3. Given prefix sums, rebuild the original array.

Here, we'll explore a general pattern: rebuild the sequence using parent or predecessor states tracked during DP.

#### Example: Reconstruct Longest Increasing Subsequence

Given an array `arr`, we first compute `dp[i]` = length of LIS ending at `i`.
We then track predecessors using `parent[i]` to rebuild the actual subsequence.

#### Recurrence

$$
dp[i] = 1 + \max_{j<i,\ arr[j]<arr[i]} dp[j]
$$

with
$$
parent[i] = \arg\max_{j<i,\ arr[j]<arr[i]} dp[j]
$$

After computing `dp`, we find the index of max(dp), then backtrack using `parent`.

#### How Does It Work (Plain Language)

1. Run the LIS DP as usual.
2. Whenever we update `dp[i]`, store which previous index gave that improvement.
3. After finishing, find the end index of the best LIS.
4. Walk backward using `parent` until `-1`.
5. Reverse the collected indices, that's your LIS.

#### Tiny Code

Python

```python
def reconstruct_lis(arr):
    n = len(arr)
    dp = [1] * n
    parent = [-1] * n

    for i in range(n):
        for j in range(i):
            if arr[j] < arr[i] and dp[j] + 1 > dp[i]:
                dp[i] = dp[j] + 1
                parent[i] = j

    length = max(dp)
    idx = dp.index(length)

    # Backtrack
    lis = []
    while idx != -1:
        lis.append(arr[idx])
        idx = parent[idx]
    lis.reverse()
    return lis

arr = [10, 9, 2, 5, 3, 7, 101, 18]
print(reconstruct_lis(arr))  # [2, 3, 7, 18]
```

#### C Version

```c
#include <stdio.h>

int main() {
    int arr[] = {10, 9, 2, 5, 3, 7, 101, 18};
    int n = sizeof(arr)/sizeof(arr[0]);
    int dp[n], parent[n];
    for (int i = 0; i < n; i++) {
        dp[i] = 1;
        parent[i] = -1;
    }

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < i; j++) {
            if (arr[j] < arr[i] && dp[j] + 1 > dp[i]) {
                dp[i] = dp[j] + 1;
                parent[i] = j;
            }
        }
    }

    // find max index
    int max_len = 0, idx = 0;
    for (int i = 0; i < n; i++) {
        if (dp[i] > max_len) {
            max_len = dp[i];
            idx = i;
        }
    }

    // reconstruct
    int lis[n], len = 0;
    while (idx != -1) {
        lis[len++] = arr[idx];
        idx = parent[idx];
    }

    printf("LIS: ");
    for (int i = len - 1; i >= 0; i--) printf("%d ", lis[i]);
    printf("\n");
}
```

#### Why It Matters

- Shows how DP tables contain full structure, not just values
- Useful in bioinformatics, diff tools, edit tracing, sequence alignment
- Forms the foundation for traceback algorithms

#### A Gentle Proof (Why It Works)

By induction on index `i`:

- Base case: first element, LIS = `[arr[i]]`
- Inductive step: each `parent[i]` points to the previous LIS endpoint giving max length
  Thus, following parent pointers from the max element recreates a valid LIS, and reversing yields forward order.

#### Try It Yourself

1. Change condition to `arr[j] > arr[i]` → Longest Decreasing Subsequence.
2. Modify to track all LIS sequences.
3. Print indices instead of values.
4. Extend to two dimensions (nested envelopes).
5. Combine with binary search LIS to get $O(n \log n)$ reconstruction.

#### Test Cases

| Input                 | LIS         |
| --------------------- | ----------- |
| [10,9,2,5,3,7,101,18] | [2,3,7,18]  |
| [3,10,2,1,20]         | [3,10,20]   |
| [50,3,10,7,40,80]     | [3,7,40,80] |

#### Complexity

- Time: $O(n^2)$
- Space: $O(n)$

Sequence Reconstruction turns numerical answers into narrative sequences, revealing how each element fits into the optimal story.

### 489 Multi-Choice Reconstruction

Multi-Choice Reconstruction is about retracing selections when a DP problem allows multiple choices per state, such as picking from categories, groups, or configurations. It extends simple parent tracking into multi-dimensional or multi-decision DP, reconstructing a full combination of choices that led to the optimal answer.

#### What Problem Are We Solving?

Some DP problems involve choosing one option from several categories, such as:

1. Multi-choice Knapsack, each group has several items; you can pick at most one.
2. Course Scheduling, pick one time slot per subject to maximize free time.
3. Machine Assignment, choose one machine per job for minimal cost.

We need to not only compute the optimal value, but also reconstruct which choices were made across categories.

#### Example: Multi-Choice Knapsack

Given `G` groups, each containing several items `(weight, value)`, select one item per group such that the total weight ≤ W and value is maximized.

#### State Definition

Let $dp[g][w]$ = max value using first $g$ groups with total weight $w$.
We will track which item in each group contributed to this value.

#### Recurrence

$$
dp[g][w] = \max_{(w_i, v_i) \in group[g]} \big(dp[g-1][w - w_i] + v_i\big)
$$

To reconstruct, we store:

$$
choice[g][w] = i \text{ such that } dp[g][w] \text{ achieved by item } i
$$

#### How Does It Work (Plain Language)

1. For each group, for each capacity, try every item in the group.
2. Pick the one that gives the highest value.
3. Store which item index gave that best value in `choice`.
4. After filling the table, backtrack from `(G, W)` using `choice` to rebuild selected items.

#### Tiny Code

Python

```python
def multi_choice_knapsack(groups, W):
    G = len(groups)
    dp = [[0] * (W + 1) for _ in range(G + 1)]
    choice = [[-1] * (W + 1) for _ in range(G + 1)]

    for g in range(1, G + 1):
        for w in range(W + 1):
            for idx, (wt, val) in enumerate(groups[g - 1]):
                if wt <= w and dp[g - 1][w - wt] + val > dp[g][w]:
                    dp[g][w] = dp[g - 1][w - wt] + val
                    choice[g][w] = idx

    # Backtrack
    w = W
    selected = []
    for g in range(G, 0, -1):
        idx = choice[g][w]
        if idx != -1:
            wt, val = groups[g - 1][idx]
            selected.append((g - 1, idx, wt, val))
            w -= wt
    selected.reverse()
    return dp[G][W], selected

groups = [
    [(3, 5), (2, 3)], 
    [(4, 6), (1, 2), (3, 4)], 
    [(2, 4), (1, 1)]
$$
print(multi_choice_knapsack(groups, 7))
```

Output:

```
(13, [(0, 0, 3, 5), (1, 1, 1, 2), (2, 0, 2, 4)])
```

#### Why It Matters

- Many optimization problems involve multiple nested decisions.
- Useful in resource allocation, scheduling, and multi-constraint planning.
- Reconstruction helps explain why the DP made each choice, crucial for debugging and interpretation.

#### A Gentle Proof (Why It Works)

We proceed by induction on `g` (group count):

- Base Case: $g=1$, choose the best item under capacity $w$.
- Inductive Step: assume all optimal choices up to group $g-1$ are correct.
  For group $g$, each `dp[g][w]` is built from `dp[g-1][w-w_i] + v_i`, and storing the index `i` ensures reconstructing one valid optimal chain backward from $(G,W)$.

Thus, each backtracked choice sequence corresponds to one optimal solution.

#### Try It Yourself

1. Add a limit on total number of groups selected.
2. Modify for multiple item selections per group.
3. Print group name instead of index.
4. Extend to 3D DP (group × capacity × budget).
5. Reconstruct second-best solution by skipping one choice.

#### Test Cases

| Groups                                                | W | Output                            |
| ----------------------------------------------------- | - | --------------------------------- |
| `[[(3,5),(2,3)], [(4,6),(1,2),(3,4)], [(2,4),(1,1)]]` | 7 | Value=13, picks=(3,5),(1,2),(2,4) |
| `[[(2,3)], [(2,2),(3,5)]]`                            | 5 | Value=8                           |
| `[[(1,1),(2,4)], [(2,2),(3,5)]]`                      | 4 | Value=6                           |

#### Complexity

- Time: $O(G \cdot W \cdot K)$ where $K$ = max group size
- Space: $O(G \cdot W)$

Multi-Choice Reconstruction turns layered decision DPs into understandable sequences, revealing exactly what was chosen and why.

### 490 Traceback Visualization

Traceback Visualization is about seeing how a DP algorithm reconstructs its answer, turning invisible state transitions into a clear path of decisions. It converts a DP table into a narrative of moves, showing how each optimal solution is formed step by step.

#### What Problem Are We Solving?

Most DP problems compute optimal values but hide how those values were reached.
Traceback visualization helps us answer:

- Which transitions were taken?
- How do we get from the base case to the solution?
- What pattern does the DP follow through its table?

You're not changing the algorithm, you're revealing its story.

Common examples:

- Longest Common Subsequence (LCS): arrows tracing matches.
- Edit Distance: diagonal for match, up for delete, left for insert.
- Matrix Path Problems: arrows showing minimal path sum.
- Knapsack: table highlights selected cells.

#### How Does It Work (Plain Language)

We reconstruct the DP solution visually:

1. Compute `dp` table as usual.
2. Start from the final state (e.g. `dp[n][m]`).
3. Move backward following transitions that created the optimal value.
4. Record each step (arrow, direction, or explanation).
5. Draw path or print trace.

Each cell's transition reveals why it was chosen, minimal, maximal, or matching condition.

#### Example: Edit Distance Visualization

Given strings `A = "kitten"`, `B = "sitting"`,
we compute $dp[i][j]$ = min edit distance between prefixes $A[0..i)$ and $B[0..j)$.

We then trace back:

- If $A[i-1] = B[j-1]$: diagonal (match)
- Else:

  * if $dp[i][j] = dp[i-1][j-1] + 1$: substitution
  * if $dp[i][j] = dp[i-1][j] + 1$: deletion
  * if $dp[i][j] = dp[i][j-1] + 1$: insertion

Trace path: bottom-right → top-left

#### Tiny Code

Python

```python
def edit_distance_trace(a, b):
    n, m = len(a), len(b)
    dp = [[0]*(m+1) for _ in range(n+1)]

    for i in range(n+1):
        dp[i][0] = i
    for j in range(m+1):
        dp[0][j] = j

    for i in range(1, n+1):
        for j in range(1, m+1):
            if a[i-1] == b[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j-1], dp[i-1][j], dp[i][j-1])

    # Traceback
    i, j = n, m
    trace = []
    while i > 0 or j > 0:
        if i > 0 and j > 0 and a[i-1] == b[j-1]:
            trace.append(f"Match {a[i-1]}")
            i -= 1; j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + 1:
            trace.append(f"Substitute {a[i-1]} -> {b[j-1]}")
            i -= 1; j -= 1
        elif i > 0 and dp[i][j] == dp[i-1][j] + 1:
            trace.append(f"Delete {a[i-1]}")
            i -= 1
        else:
            trace.append(f"Insert {b[j-1]}")
            j -= 1
    trace.reverse()
    return dp[n][m], trace

dist, steps = edit_distance_trace("kitten", "sitting")
print("Distance:", dist)
print("\n".join(steps))
```

Output:

```
Distance: 3
Substitute k -> s
Match i
Match t
Match t
Substitute e -> i
Match n
Insert g
```

#### Why It Matters

- Turns abstract numbers into concrete reasoning.
- Great for teaching and debugging DPs.
- Bridges computation and human understanding.
- Helps verify optimal path or detect incorrect transitions.

#### A Gentle Proof (Why It Works)

By construction, each $dp[i][j]$ is the minimal cost achievable.
At reconstruction, we select any predecessor $(i', j')$ satisfying the recurrence equality:

$$
dp[i][j] = f(dp[i'][j']) + cost
$$

Thus, following these equalities in reverse forms a valid optimal path.
Since all $dp$ states are filled bottom-up, each predecessor was computed earlier, ensuring consistency.

#### Try It Yourself

1. Visualize LCS: arrows for match and direction.
2. Visualize min path sum in a grid: print arrows or coordinates.
3. Add ASCII arrows (↖ ↑ ←) to the traceback.
4. Generate a heatmap with colored path.
5. Show decision explanation (e.g. "Insert g at pos 6").

#### Test Cases

| Input                      | Output Trace             |
| -------------------------- | ------------------------ |
| ("abc", "abc")             | Matches only             |
| ("kitten", "sitting")      | Substitute, Insert steps |
| ("intention", "execution") | Mixed edits              |
| ("horse", "ros")           | Delete + Substitute path |

#### Complexity

- Time: same as DP, $O(nm)$
- Space: $O(nm)$ for table + path

Traceback visualization transforms the final DP value into a living path, so learners see how the algorithm's reasoning unfolds across the grid.

# Section 50. Meta-DP and Optimization Templates 

### 491 State Compression Template

State Compression is a powerful technique for reducing the dimensionality of dynamic programming problems by encoding multiple substate variables into a single integer mask. This allows us to handle problems that depend on subsets, visited states, or boolean configurations efficiently using bitwise operations.

#### What Problem Are We Solving?

Many DP problems depend on a combination of states:

- Which elements have been used?
- Which nodes have been visited?
- What subset is currently active?

Representing these directly can explode in complexity.
Instead, we compress the state into an integer bitmask, where each bit indicates the inclusion or status of an element.

For example:

- Traveling Salesman: mask tracks visited cities.
- Subset Sum: mask tracks sums possible with subsets.
- Graph DP: mask encodes chosen vertices.

#### How Does It Work (Plain Language)

We represent subsets or multi-variable states with bitmasks:

- $1$ bit → included, $0$ bit → excluded.
- For $n$ elements, we have $2^n$ possible masks.

Each DP state looks like:

$$
dp[mask] = \text{best result for subset represented by } mask
$$

Transitions iterate over bits set/unset in the mask, updating dependent states.

Key operations:

- `mask | (1 << i)` → include element i
- `mask & (1 << i)` → check if i included
- `mask ^ (1 << i)` → toggle inclusion
- `mask & -mask` → extract lowest set bit

#### Example: Subset DP Template

$$
dp[mask] = \min_{i \in mask} \big( dp[mask \setminus {i}] + cost[i] \big)
$$

Here, each `mask` represents a combination of items, and we build solutions incrementally by adding one element at a time.

#### Tiny Code

C

```c
#include <stdio.h>
#include <limits.h>

int min(int a, int b) { return a < b ? a : b; }

int main(void) {
    int n = 3;
    int cost[] = {3, 2, 5};
    int dp[1 << 3];

    for (int mask = 0; mask < (1 << n); mask++)
        dp[mask] = INT_MAX / 2;

    dp[0] = 0;

    for (int mask = 1; mask < (1 << n); mask++) {
        for (int i = 0; i < n; i++) {
            if (mask & (1 << i)) {
                int prev = mask ^ (1 << i);
                dp[mask] = min(dp[mask], dp[prev] + cost[i]);
            }
        }
    }

    printf("Minimum total cost: %d\n", dp[(1 << n) - 1]);
}
```

Python

```python
from math import inf

n = 3
cost = [3, 2, 5]
dp = [inf] * (1 << n)
dp[0] = 0

for mask in range(1, 1 << n):
    for i in range(n):
        if mask & (1 << i):
            dp[mask] = min(dp[mask], dp[mask ^ (1 << i)] + cost[i])

print("Minimum total cost:", dp[(1 << n) - 1])
```

#### Why It Matters

- Compresses exponential states into manageable integer masks.
- Enables elegant solutions for combinatorial problems.
- Essential for TSP, Assignment, Subset DP, and Bitmask Knapsack.
- Fits perfectly with iterative DP loops.

#### A Gentle Proof (Why It Works)

If $dp[S]$ stores the optimal result for subset $S$,
and every transition moves from smaller to larger subsets via one addition:

$$
dp[S] = \min_{i \in S} \big( dp[S \setminus {i}] + cost[i] \big)
$$

Then by induction:

- Base case: $dp[\emptyset]$ is known (often 0).
- Inductive step: each subset $S$ builds on smaller subsets.
  All subsets are processed in increasing order of size, ensuring correctness.

#### Try It Yourself

1. Implement Subset DP for Sum Over Subsets (SOS DP).
2. Solve Traveling Salesman using state compression.
3. Adapt to Assignment Problem ($n!$ → $2^n n$ states).
4. Use mask parity (even/odd bits) for combinatorial constraints.
5. Print masks in binary to visualize transitions.

#### Test Cases

| Input            | Description           | Output |
| ---------------- | --------------------- | ------ |
| cost = [3, 2, 5] | choose all 3 elements | 10     |
| cost = [1, 2]    | 2 elements            | 3      |
| cost = [5]       | single item           | 5      |

#### Complexity

- Time: $O(n \cdot 2^n)$
- Space: $O(2^n)$

State Compression DP is your gateway to subset reasoning, compact, powerful, and fundamental for solving exponential combinatorial spaces with structure.

### 492 Transition Optimization Template

Transition Optimization is a core technique for improving the efficiency of DP transitions by precomputing or structuring recurrence updates. Many DP recurrences involve nested loops or repeated evaluations that can be simplified through mathematical properties, monotonicity, or auxiliary data structures.

#### What Problem Are We Solving?

In many DPs, each state depends on a range or set of previous states:

$$
dp[i] = \min_{j < i} \big( dp[j] + cost(j, i) \big)
$$

Naively, this takes $O(n^2)$ time.
But if $cost(j, i)$ has special structure (monotonicity, convexity, quadrangle inequality), we can reduce it to $O(n \log n)$ or even $O(n)$ using optimized transitions.

Transition optimization finds patterns or data structures to accelerate these computations.

#### How Does It Work (Plain Language)

When you notice repeated transitions like:

```c
for (int i = 1; i <= n; i++)
    for (int j = 0; j < i; j++)
        dp[i] = min(dp[i], dp[j] + cost(j, i));
```

...you're paying an $O(n^2)$ cost.
But often, `cost(j, i)` follows a pattern (e.g. linear, convex, or monotonic), so we can optimize:

- Monotonic Queue Optimization: for sliding window minimums.
- Divide & Conquer DP: when optimal j's move monotonically.
- Convex Hull Trick: when $cost(j, i) = m_j \cdot x_i + b_j$ is linear.
- Knuth Optimization: when quadrangle inequality holds.

Each approach precomputes or narrows transitions.

#### Example Transition (Generic)

$$
dp[i] = \min_{j < i} \big( dp[j] + f(j, i) \big)
$$

If $f$ satisfies the Monge property or quadrangle inequality, we can determine that the optimal $j$ moves in one direction only (monotonic).
That means we can use divide & conquer or pointer tricks to find it efficiently.

#### Tiny Code

C (Naive Transition)

```c
for (int i = 1; i <= n; i++) {
    dp[i] = INF;
    for (int j = 0; j < i; j++) {
        int candidate = dp[j] + cost(j, i);
        if (candidate < dp[i])
            dp[i] = candidate;
    }
}
```

C (Optimized with Monotonic Pointer)

```c
int ptr = 0;
for (int i = 1; i <= n; i++) {
    while (ptr + 1 < i && better(ptr + 1, ptr, i))
        ptr++;
    dp[i] = dp[ptr] + cost(ptr, i);
}
```

Here `better(a, b, i)` checks whether `a` gives a smaller cost than `b` for `dp[i]`.

#### Python (Sliding Window Optimization)

```python
from collections import deque

dp = [0] * (n + 1)
q = deque([0])

for i in range(1, n + 1):
    while len(q) >= 2 and better(q[1], q[0], i):
        q.popleft()
    j = q[0]
    dp[i] = dp[j] + cost(j, i)
    while len(q) >= 2 and cross(q[-2], q[-1], i):
        q.pop()
    q.append(i)
```

This structure appears in Convex Hull Trick and Monotonic Queue Optimization.

#### Why It Matters

- Reduces $O(n^2)$ → $O(n \log n)$ or $O(n)$ transitions.
- Exploits structure (monotonicity, convexity) in DP cost functions.
- Powers major optimizations:

  * Knuth Optimization
  * Divide & Conquer DP
  * Convex Hull Trick
  * Slope Trick
  * Monotone Queue DP

#### A Gentle Proof (Why It Works)

If the recurrence satisfies Monotonicity of the Argmin, i.e.:

$$
opt[i] \le opt[i+1]
$$

then the best transition index $j$ moves non-decreasingly.
This means we can find optimal $j$ for all $i$ in one sweep, using either:

- Two-pointer traversal (Monotone Queue)
- Divide & Conquer recursion (Knuth or D&C DP)
- Line container (Convex Hull Trick)

By exploiting this structure, we avoid recomputation.

#### Try It Yourself

1. Identify a DP where each state depends on a range of previous states.
2. Check if `cost(j, i)` satisfies monotonic or convex properties.
3. Apply divide & conquer optimization to reduce $O(n^2)$.
4. Implement Convex Hull Trick for linear cost forms.
5. Use deque-based Monotonic Queue for sliding range DP.

#### Test Cases

| Case                               | Recurrence      | Optimization      |
| ---------------------------------- | --------------- | ----------------- |
| $dp[i] = \min_{j<i}(dp[j]+c(i-j))$ | $c$ convex      | Convex Hull Trick |
| $dp[i] = \min_{j<i}(dp[j]+w(j,i))$ | Monotone argmin | Divide & Conquer  |
| $dp[i] = \min_{j<i}(dp[j]) + a_i$  | sliding window  | Monotonic Queue   |

#### Complexity

- Time: $O(n)$ to $O(n \log n)$ (depends on method)
- Space: $O(n)$

Transition Optimization is the art of seeing structure in cost, once you spot monotonicity or convexity, your DP becomes faster, cleaner, and smarter.

### 493 Space Optimization Template

Space Optimization is the art of trimming away unused dimensions in a DP table by realizing that only a limited subset of previous states is needed at each step. Many classic DPs that start with large $O(n^2)$ or $O(nm)$ tables can be reduced to rolling arrays or single-row updates, cutting memory usage drastically.

#### What Problem Are We Solving?

Dynamic Programming often uses multi-dimensional arrays:

$$
dp[i][j] = \text{answer using first } i \text{ items with capacity } j
$$

But not all dimensions are necessary.
If each state $dp[i]$ only depends on previous row $dp[i-1]$, we can reuse memory, keeping just two rows (or even one).

Space Optimization lets us move from $O(nm)$ to $O(m)$, or from 2D → 1D, or 3D → 2D, without changing logic.

#### How Does It Work (Plain Language)

DP updates come from previous states, not all states.

For example, in 0/1 Knapsack:

$$
dp[i][w] = \max(dp[i-1][w], dp[i-1][w - wt[i]] + val[i])
$$

Only `dp[i-1][*]` is needed when computing `dp[i][*]`.
So we can collapse the DP table into a single array `dp[w]`, updating it in reverse (to avoid overwriting states we still need).

If transitions depend on current or previous row, choose direction carefully:

- 0/1 Knapsack → reverse loop
- Unbounded Knapsack → forward loop

#### Example Transformation

Before (2D DP)

```c
int dp[n+1][W+1];
for (int i = 1; i <= n; i++)
  for (int w = 0; w <= W; w++)
    dp[i][w] = max(dp[i-1][w], dp[i-1][w-wt[i]] + val[i]);
```

After (1D DP)

```c
int dp[W+1] = {0};
for (int i = 1; i <= n; i++)
  for (int w = W; w >= wt[i]; w--)
    dp[w] = max(dp[w], dp[w-wt[i]] + val[i]);
```

#### Tiny Code

C (Rolling Array Example)

```c
#include <stdio.h>
#define max(a,b) ((a)>(b)?(a):(b))

int main(void) {
    int n = 3, W = 5;
    int wt[] = {0, 2, 3, 4};
    int val[] = {0, 4, 5, 7};
    int dp[6] = {0};

    for (int i = 1; i <= n; i++)
        for (int w = W; w >= wt[i]; w--)
            dp[w] = max(dp[w], dp[w - wt[i]] + val[i]);

    printf("Max value: %d\n", dp[W]);
}
```

Python (1D Rolling)

```python
n, W = 3, 5
wt = [2, 3, 4]
val = [4, 5, 7]

dp = [0] * (W + 1)

for i in range(n):
    for w in range(W, wt[i] - 1, -1):
        dp[w] = max(dp[w], dp[w - wt[i]] + val[i])

print("Max value:", dp[W])
```

#### Why It Matters

- Reduces memory from $O(nm)$ → $O(m)$.
- Makes large DP problems feasible under memory limits.
- Reveals dependency structure in transitions.
- Forms the backbone of iterative bottom-up optimization.

Space optimization is vital for:

- Knapsack, LCS, LIS
- Grid path counting
- Partition problems
- Digit DP (carry compression)

#### A Gentle Proof (Why It Works)

Let's define $dp[i][j]$ depending only on $dp[i-1][*]$.
Since each new row is computed solely from the previous one:

$$
dp[i][j] = f(dp[i-1][j], dp[i-1][j-w_i])
$$

So at iteration $i$, once `dp[i][*]` is complete, `dp[i-1][*]` is never used again.
By updating in reverse (to preserve dependencies), the 2D table can be rolled into one.

Formally, space can be reduced from $O(nm)$ to $O(m)$
if and only if:

1. Each $dp[i]$ depends on $dp[i-1]$, not $dp[i]$ itself.
2. Transition direction ensures previous states remain unmodified.

#### Try It Yourself

1. Convert your 0/1 Knapsack to 1D DP.
2. Space-optimize the LCS table (2D → 2 rows).
3. Apply to "Climbing Stairs" ($dp[i]$ only needs last 2 values).
4. For Unbounded Knapsack, try forward updates.
5. Compare memory usage before and after.

#### Test Cases

| Problem      | Original Space | Optimized Space |
| ------------ | -------------- | --------------- |
| 0/1 Knapsack | $O(nW)$        | $O(W)$          |
| LCS          | $O(nm)$        | $O(2m)$         |
| Fibonacci    | $O(n)$         | $O(1)$          |

#### Complexity

- Time: unchanged
- Space: reduced by 1 dimension
- Tradeoff: direction of iteration matters

Space Optimization is a quiet revolution: by recognizing independence between layers, we free our algorithms from unnecessary memory, one dimension at a time.

### 494 Multi-Dimensional DP Template

Multi-Dimensional DP extends classic one- or two-dimensional formulations into higher-dimensional state spaces, capturing problems where multiple independent variables evolve together. Each dimension corresponds to a decision axis, time, position, capacity, or some discrete property, making it possible to express rich combinatorial or structural relationships.

#### What Problem Are We Solving?

Some problems require tracking more than one evolving parameter:

- Knapsack with two capacities → $dp[i][w_1][w_2]$
- String interleaving → $dp[i][j][k]$
- Dice sum counting → $dp[i][sum][count]$
- Grid with keys → $dp[x][y][mask]$

When multiple independent factors drive state transitions, a single index DP cannot capture them. Multi-Dimensional DP encodes joint state evolution explicitly.

#### How Does It Work (Plain Language)

We define a DP table where each axis tracks a property:

$$
dp[a][b][c] = \text{best result with parameters } (a, b, c)
$$

Transitions update along one or more dimensions:

$$
dp[a][b][c] = \min/\max(\text{transitions from neighbors})
$$

Think of this as traversing a grid of states, where each move modifies several parameters. The key idea is to fill the table systematically based on topological or nested loops that respect dependency order.

#### Example Recurrence

Multi-dimensional structure often looks like:

$$
dp[i][j][k] = f(dp[i-1][j'][k'], \text{cost}(i, j, k))
$$

Example (2D Knapsack):

$$
dp[i][w_1][w_2] = \max(dp[i-1][w_1][w_2],\ dp[i-1][w_1-wt_1[i]][w_2-wt_2[i]] + val[i])
$$

#### Tiny Code

C (2D Knapsack)

```c
#include <stdio.h>
#define max(a,b) ((a)>(b)?(a):(b))

int main(void) {
    int n = 3, W1 = 5, W2 = 5;
    int wt1[] = {0, 2, 3, 4};
    int wt2[] = {0, 1, 2, 3};
    int val[] = {0, 4, 5, 6};
    int dp[4][6][6] = {0};

    for (int i = 1; i <= n; i++) {
        for (int w1 = 0; w1 <= W1; w1++) {
            for (int w2 = 0; w2 <= W2; w2++) {
                dp[i][w1][w2] = dp[i-1][w1][w2];
                if (w1 >= wt1[i] && w2 >= wt2[i]) {
                    dp[i][w1][w2] = max(dp[i][w1][w2],
                        dp[i-1][w1 - wt1[i]][w2 - wt2[i]] + val[i]);
                }
            }
        }
    }

    printf("Max value: %d\n", dp[n][W1][W2]);
}
```

Python (3D Example: String Interleaving)

```python
s1, s2, s3 = "ab", "cd", "acbd"
n1, n2, n3 = len(s1), len(s2), len(s3)

dp = [[[False]*(n3+1) for _ in range(n2+1)] for _ in range(n1+1)]
dp[0][0][0] = True

for i in range(n1+1):
    for j in range(n2+1):
        for k in range(n3+1):
            if k == 0: continue
            if i > 0 and s1[i-1] == s3[k-1] and dp[i-1][j][k-1]:
                dp[i][j][k] = True
            if j > 0 and s2[j-1] == s3[k-1] and dp[i][j-1][k-1]:
                dp[i][j][k] = True

print("Interleaving possible:", dp[n1][n2][n3])
```

#### Why It Matters

- Captures multi-factor problems elegantly
- Handles constraints coupling (capacity, index, sum)
- Enables state compression when reduced
- Common in:

  * Multi-resource allocation
  * Interleaving / sequence merging
  * Multi-knapsack / bounded subset
  * Grid navigation with additional properties

Multi-dimensional DPs form the foundation of generalized search spaces, where each variable adds a dimension of reasoning.

#### A Gentle Proof (Why It Works)

By induction over the outermost dimension:

If $dp[i][*][*]$ depends only on $dp[i-1][*][*]$,
and each transition moves from smaller to larger indices,
then the DP fills in topological order, ensuring correctness.

Each additional dimension multiplies the state space but does not alter dependency direction.
Thus, correctness holds as long as we respect dimension order and initialize base cases properly.

#### Try It Yourself

1. Solve 2D Knapsack with dual capacity.
2. Implement string interleaving check with 3D DP.
3. Model shortest path in 3D grid using $dp[x][y][z]$.
4. Add bitmask dimension for subset tracking.
5. Optimize memory using rolling or compression.

#### Test Cases

| Problem             | Dimensions        | Example State    | Output     |
| ------------------- | ----------------- | ---------------- | ---------- |
| 2D Knapsack         | 3D (item, w1, w2) | $dp[i][w1][w2]$  | Max value  |
| String Interleaving | 3D                | $dp[i][j][k]$    | True/False |
| Grid with Keys      | 3D                | $dp[x][y][mask]$ | Min steps  |

#### Complexity

- Time: $O(\text{product of dimensions})$
- Space: same order; compressible via rolling
- Tradeoff: richer state space vs feasibility

Multi-Dimensional DP is your tool for multi-constraint reasoning, when life refuses to fit in one dimension, let your DP grow an extra axis.

### 495 Decision Monotonicity

Decision Monotonicity is a structural property in DP recurrences that allows us to optimize transition search. When the optimal decision index for $dp[i]$ moves in one direction (non-decreasing) as $i$ increases, we can reduce a naive $O(n^2)$ DP to $O(n \log n)$ or even $O(n)$ using divide-and-conquer or two-pointer techniques.

#### What Problem Are We Solving?

In many DPs, each state $dp[i]$ is computed by choosing a best transition point $j < i$:

$$
dp[i] = \min_{0 \le j < i} \big( dp[j] + cost(j, i) \big)
$$

This naive recurrence requires trying all previous states for every $i$, leading to $O(n^2)$ time.
But if the index of the optimal $j$ (called $opt[i]$) satisfies:

$$
opt[i] \le opt[i+1]
$$

then the decision index moves monotonically, and we can search efficiently, either by divide & conquer DP or sliding pointer optimization.

#### How Does It Work (Plain Language)

If as $i$ increases, the best $j$ never moves backward, we can reuse or narrow the search for each next state.

In other words:

- The "best split point" for $i=10$ will be at or after the best split for $i=9$.
- No need to re-check smaller $j$ again.
- You can sweep $j$ forward or recursively restrict the range.

This property appears when $cost(j, i)$ satisfies certain quadrangle inequalities or convexity conditions.

#### Example Recurrence

$$
dp[i] = \min_{j < i} \big( dp[j] + (i-j)^2 \big)
$$

Here, as $i$ grows, larger $j$ become more favorable because $(i-j)^2$ penalizes small gaps. Thus, $opt[i]$ increases monotonically.

Another example:
$$
dp[i] = \min_{j < i} \big( dp[j] + c[j] \cdot a[i] \big)
$$
where $a[i]$ is increasing, the convex hull trick applies, and optimal lines appear in increasing order.

#### Tiny Code (Two-Pointer Monotonic Search)

C

```c
int n = ...;
int dp[MAXN];
int opt[MAXN];
for (int i = 1; i <= n; i++) {
    dp[i] = INF;
    int start = opt[i-1];
    for (int j = start; j <= i; j++) {
        int val = dp[j] + cost(j, i);
        if (val < dp[i]) {
            dp[i] = val;
            opt[i] = j;
        }
    }
}
```

Each $opt[i]$ begins searching from $opt[i-1]$, cutting redundant checks.

#### Python (Divide & Conquer Optimization)

```python
def solve(l, r, optL, optR):
    if l > r: return
    mid = (l + r) // 2
    best = (float('inf'), -1)
    for j in range(optL, min(optR, mid) + 1):
        val = dp[j] + cost(j, mid)
        if val < best[0]:
            best = (val, j)
    dp[mid] = best[0]
    opt = best[1]
    solve(l, mid - 1, optL, opt)
    solve(mid + 1, r, opt, optR)

solve(1, n, 0, n-1)
```

#### Why It Matters

- Reduces complexity from $O(n^2)$ → $O(n \log n)$ or $O(n)$
- Enables Divide & Conquer DP, Knuth Optimization, and Convex Hull Trick
- Builds foundation for structured cost functions
- Helps identify monotonic transitions in scheduling, partitioning, or chain DPs

#### A Gentle Proof (Why It Works)

If $opt[i] \le opt[i+1]$, then $dp[i]$'s optimal transition comes from no earlier than $opt[i-1]$.
Thus, we can safely restrict search intervals:

$$
dp[i] = \min_{j \in [opt[i-1], i-1]} f(j, i)
$$

The proof follows from quadrangle inequality:

$$
f(a, c) + f(b, d) \le f(a, d) + f(b, c)
$$

which ensures convex-like structure and monotone decisions.

By induction:

- Base: $opt[1]$ known.
- Step: if $opt[i] \le opt[i+1]$, then the recurrence preserves order.

#### Try It Yourself

1. Implement a divide & conquer DP with $opt$ tracking.
2. Verify monotonicity of $opt[i]$ experimentally for a sample cost.
3. Apply to partitioning problems like Divide Array into K Segments.
4. Compare $O(n^2)$ vs optimized $O(n \log n)$ performance.
5. Check if your cost satisfies quadrangle inequality.

#### Test Cases

| Recurrence                                 | Property          | Optimization     |
| ------------------------------------------ | ----------------- | ---------------- |
| $dp[i] = \min_{j<i}(dp[j]+(i-j)^2)$        | convex            | monotone opt     |
| $dp[i] = \min_{j<i}(dp[j]+a[i]\cdot b[j])$ | increasing $a[i]$ | convex hull      |
| Segment DP                                 | $cost(l,r)$ Monge | divide & conquer |

#### Complexity

- Time: $O(n \log n)$ (divide & conquer) or $O(n)$ (two-pointer)
- Space: $O(n)$

Decision Monotonicity is the hidden geometry of DP, once you spot that the "best index" moves only forward, your algorithm speeds up dramatically.

### 496 Monge Array Optimization

Monge Array Optimization is a powerful tool for accelerating dynamic programming when the cost matrix satisfies a special inequality known as the Monge property. It guarantees that the argmin of each row moves monotonically across columns, allowing us to use Divide & Conquer DP or SMAWK algorithm for subquadratic optimization.

#### What Problem Are We Solving?

Consider a DP of the form:

$$
dp[i][j] = \min_{k < j} \big(dp[i-1][k] + cost[k][j]\big)
$$

If the cost matrix $cost[k][j]$ satisfies the Monge property, we can compute all $dp[i][j]$ in $O(n \log n)$ or $O(n)$ per layer, instead of the naive $O(n^2)$.

This pattern appears in:

- Partition DP (divide sequence into segments)
- Matrix Chain / Knuth DP
- Optimal merge / segmentation

#### How Does It Work (Plain Language)

The Monge property states that for all $a < b$ and $c < d$:

$$
cost[a][c] + cost[b][d] \le cost[a][d] + cost[b][c]
$$

This means the difference in cost is consistent across diagonals, implying convexity in two dimensions.
As a result, the optimal split point moves monotonically:

$$
opt[i][j] \le opt[i][j+1]
$$

We can therefore restrict our search range for $dp[i][j]$ using Divide & Conquer optimization.

#### Example Recurrence

For segment partitioning:

$$
dp[i][j] = \min_{k < j} \big( dp[i-1][k] + cost[k][j] \big)
$$

If $cost[k][j]$ is Monge, then $opt[i][j] \le opt[i][j+1]$.
Thus, when computing $dp[i][j]$, we only need to search $k$ in $[opt[i][j-1], opt[i][j+1]]$.

#### Tiny Code (Divide & Conquer over Monge Matrix)

C (Template)

```c
void compute(int i, int l, int r, int optL, int optR) {
    if (l > r) return;
    int mid = (l + r) / 2;
    int best_k = -1;
    long long best_val = LLONG_MAX;
    for (int k = optL; k <= optR && k < mid; k++) {
        long long val = dp_prev[k] + cost[k][mid];
        if (val < best_val) {
            best_val = val;
            best_k = k;
        }
    }
    dp[mid] = best_val;
    compute(i, l, mid - 1, optL, best_k);
    compute(i, mid + 1, r, best_k, optR);
}
```

Each recursive call computes a segment's midpoint and recursively narrows the search range based on monotonicity.

Python (Monge DP Skeleton)

```python
def compute(i, l, r, optL, optR):
    if l > r:
        return
    mid = (l + r) // 2
    best = (float('inf'), -1)
    for k in range(optL, min(optR, mid) + 1):
        val = dp_prev[k] + cost[k][mid]
        if val < best[0]:
            best = (val, k)
    dp[mid] = best[0]
    opt[mid] = best[1]
    compute(i, l, mid - 1, optL, best[1])
    compute(i, mid + 1, r, best[1], optR)
```

#### Why It Matters

- Exploits Monge property to skip redundant transitions
- Reduces 2D DP to $O(n \log n)$ or even $O(n)$ per layer
- Powers optimizations like:

  * Divide & Conquer DP
  * Knuth Optimization (special Monge case)
  * SMAWK algorithm (row minima in Monge arrays)

Used in:

- Sequence segmentation
- Matrix chain multiplication
- Optimal BST
- Inventory / scheduling models

#### A Gentle Proof (Why It Works)

If $cost$ satisfies Monge inequality:

$$
cost[a][c] + cost[b][d] \le cost[a][d] + cost[b][c]
$$

then:

$$
opt[j] \le opt[j+1]
$$

That is, as $j$ increases, the best $k$ (split point) cannot move backward.
Hence, when computing $dp[j]$, we can reuse or narrow the search interval using the previous opt index.

This monotonicity of argmin is the key to divide-and-conquer speedups.

#### Try It Yourself

1. Verify Monge property for your cost function.
2. Implement the Divide & Conquer DP template.
3. Test on partition DP with convex segment cost.
4. Compare $O(n^2)$ vs optimized $O(n \log n)$ runtime.
5. Explore SMAWK for row minima in Monge matrices.

#### Test Cases

| Cost Function                    | Monge? | Optimization |   |    |
| -------------------------------- | ------ | ------------ | - | -- |
| $cost[a][b] = (sum[b]-sum[a])^2$ | ✅      | Yes          |   |    |
| $cost[a][b] = (b-a)^2$           | ✅      | Yes          |   |    |
| $cost[a][b] =                    | b-a    | $            | ❌ | No |

#### Complexity

- Time: $O(n \log n)$ per layer
- Space: $O(n)$
- Layers: multiply by $k$ if multi-stage DP

Monge Array Optimization transforms a naive DP table into a structured landscape, once your costs align, transitions fall neatly into place with logarithmic grace.

### 497 Divide & Conquer Template

Divide & Conquer DP is a technique for optimizing DP transitions when the optimal transition index exhibits monotonicity. By recursively dividing the problem and searching only within a limited range for each midpoint, we reduce complexity from $O(n^2)$ to $O(n \log n)$ or even $O(n)$ per layer.

#### What Problem Are We Solving?

Many DP formulations involve transitions like:

$$
dp[i] = \min_{j < i} \big( dp[j] + cost(j, i) \big)
$$

If $opt[i] \le opt[i+1]$, meaning the best transition index moves monotonically forward, we can use divide and conquer to find optimal $j$ efficiently instead of scanning all $j < i$.

This structure is common in:

- Partition DP (divide array into $k$ segments)
- Monge or Convex cost problems
- Segment-based recurrence with monotone argmin

#### How Does It Work (Plain Language)

We recursively divide the range $[L, R]$, compute $dp[mid]$ using the best transition from a restricted interval $[optL, optR]$, then:

- Left half $[L, mid-1]$ searches $[optL, opt[mid]]$
- Right half $[mid+1, R]$ searches $[opt[mid], optR]$

By maintaining monotone search boundaries, we ensure correctness and avoid redundant checks.

Think of it as a guided binary search over DP indices, powered by structural guarantees.

#### Example Recurrence

$$
dp[i] = \min_{j < i} \big( dp[j] + cost(j, i) \big)
$$

If $cost$ satisfies quadrangle inequality or Monge property, then:

$$
opt[i] \le opt[i+1]
$$

Thus, we can recursively compute $dp$ over subranges.

#### Tiny Code (C)

```c
#include <stdio.h>
#include <limits.h>

#define INF 1000000000
#define min(a,b) ((a)<(b)?(a):(b))

int n;
int dp[10005], prev_dp[10005];

// Example cost function (prefix sums)
int prefix[10005];
int cost(int j, int i) {
    int sum = prefix[i] - prefix[j];
    return sum * sum;
}

void compute(int l, int r, int optL, int optR) {
    if (l > r) return;
    int mid = (l + r) / 2;
    int best_k = -1;
    int best_val = INF;

    for (int k = optL; k <= optR && k < mid; k++) {
        int val = prev_dp[k] + cost(k, mid);
        if (val < best_val) {
            best_val = val;
            best_k = k;
        }
    }

    dp[mid] = best_val;

    // Recurse left and right halves
    compute(l, mid - 1, optL, best_k);
    compute(mid + 1, r, best_k, optR);
}

int main(void) {
    n = 5;
    int arr[] = {0, 1, 2, 3, 4, 5};
    for (int i = 1; i <= n; i++) prefix[i] = prefix[i-1] + arr[i];
    for (int i = 0; i <= n; i++) prev_dp[i] = i*i;

    compute(1, n, 0, n-1);

    for (int i = 1; i <= n; i++) printf("dp[%d] = %d\n", i, dp[i]);
}
```

Python

```python
def cost(j, i):
    s = prefix[i] - prefix[j]
    return s * s

def compute(l, r, optL, optR):
    if l > r:
        return
    mid = (l + r) // 2
    best = (float('inf'), -1)
    for k in range(optL, min(optR, mid) + 1):
        val = prev_dp[k] + cost(k, mid)
        if val < best[0]:
            best = (val, k)
    dp[mid], opt[mid] = best
    compute(l, mid - 1, optL, best[1])
    compute(mid + 1, r, best[1], optR)

n = 5
arr = [0, 1, 2, 3, 4, 5]
prefix = [0]
for x in arr: prefix.append(prefix[-1] + x)
prev_dp = [i*i for i in range(len(arr))]
dp = [0]*(n+1)
opt = [0]*(n+1)
compute(1, n, 0, n-1)
print(dp[1:])
```

#### Why It Matters

- Reduces complexity dramatically: $O(n \log n)$ per layer
- Works on structured recurrences with monotonic $opt[i]$
- Forms backbone for:

  * Knuth Optimization
  * Monge Array DP
  * Segment Partition DP

You can think of it as "binary search for DP transitions."

#### A Gentle Proof (Why It Works)

If $opt[i] \le opt[i+1]$, then each $dp[mid]$'s optimal index $k^*$ lies between $optL$ and $optR$.
When dividing the range:

- Left child ($L, mid-1$) searches $[optL, k^*]$
- Right child ($mid+1, R$) searches $[k^*, optR]$

By induction, every segment explores only valid transitions.
Since each $k$ is visited $O(\log n)$ times, total time is $O(n \log n)$.

#### Try It Yourself

1. Implement partition DP with convex segment cost.
2. Verify monotonicity of $opt[i]$ numerically.
3. Compare $O(n^2)$ vs optimized $O(n \log n)$.
4. Combine with space optimization (roll arrays).
5. Extend to multi-layer DP (e.g., k-partition).

#### Test Cases

| Recurrence                                  | Property | Optimization |        |     |
| ------------------------------------------- | -------- | ------------ | ------ | --- |
| $dp[i]=\min_{j<i}(dp[j]+(sum[i]-sum[j])^2)$ | convex   | yes          |        |     |
| $dp[i]=\min_{j<i}(dp[j]+                    | i-j      | )$           | linear | yes |
| $dp[i]=\min_{j<i}(dp[j]+cost[j][i])$        | Monge    | yes          |        |     |

#### Complexity

- Time: $O(n \log n)$ per layer
- Space: $O(n)$
- Layers: multiply by $k$ for multi-stage DP

Divide & Conquer DP is your scalpel for quadratic DPs, once you find monotonicity, you slice complexity cleanly in half at every level.

### 498 Rerooting Template

Rerooting DP is a powerful tree dynamic programming pattern that lets you compute results for every node as the root, efficiently reusing computations from parent-to-child transitions. It's like rotating the tree root through all nodes without recomputing everything from scratch.

#### What Problem Are We Solving?

Given a tree, we often want to compute a property for each node as if it were the root. For example:

- Sum of distances to all nodes
- Size of subtree or value based on children
- Number of valid colorings rooted at each node

Naively, you could rerun DP for every node, $O(n^2)$, but rerooting reduces this to $O(n)$ or $O(n \log n)$ by cleverly reusing partial results.

#### How Does It Work (Plain Language)

1. First pass (postorder): compute DP values bottom-up for a fixed root (usually node 1).
2. Second pass (preorder): propagate results top-down, rerooting along each edge and combining parent contributions.

When moving the root from `u` to `v`:

- Remove `v`'s contribution from `u`'s DP.
- Add `u`'s contribution (excluding `v`) into `v`'s DP.

This way, each node inherits a correct rerooted DP in one traversal.

#### Example Problem

Compute sum of distances from every node to all others.

Let:

- $dp[u]$ = sum of distances from $u$ to all nodes in its subtree
- $sz[u]$ = size of subtree of $u$

We can reroot using:
$$
dp[v] = dp[u] - sz[v] + (n - sz[v])
$$
when moving root from $u$ to child $v$.

#### Tiny Code (C)

```c
#include <stdio.h>
#include <vector>

#define MAXN 100005
using namespace std;

vector<int> adj[MAXN];
int n;
int sz[MAXN];
long long dp[MAXN];
long long ans[MAXN];

void dfs1(int u, int p) {
    sz[u] = 1;
    dp[u] = 0;
    for (int v : adj[u]) if (v != p) {
        dfs1(v, u);
        sz[u] += sz[v];
        dp[u] += dp[v] + sz[v];
    }
}

void dfs2(int u, int p) {
    ans[u] = dp[u];
    for (int v : adj[u]) if (v != p) {
        long long dp_u = dp[u], dp_v = dp[v];
        int sz_u = sz[u], sz_v = sz[v];

        // Move root from u to v
        dp[u] -= dp[v] + sz[v];
        sz[u] -= sz[v];
        dp[v] += dp[u] + sz[u];
        sz[v] += sz[u];

        dfs2(v, u);

        // Restore
        dp[u] = dp_u; dp[v] = dp_v;
        sz[u] = sz_u; sz[v] = sz_v;
    }
}

int main(void) {
    scanf("%d", &n);
    for (int i = 0; i < n-1; i++) {
        int u, v;
        scanf("%d%d", &u, &v);
        adj[u].push_back(v);
        adj[v].push_back(u);
    }
    dfs1(1, -1);
    dfs2(1, -1);
    for (int i = 1; i <= n; i++)
        printf("Sum of distances from %d: %lld\n", i, ans[i]);
}
```

Python

```python
from collections import defaultdict

n = 5
adj = defaultdict(list)
edges = [(1,2),(1,3),(3,4),(3,5)]
for u,v in edges:
    adj[u].append(v)
    adj[v].append(u)

sz = [0]*(n+1)
dp = [0]*(n+1)
ans = [0]*(n+1)

def dfs1(u,p):
    sz[u] = 1
    dp[u] = 0
    for v in adj[u]:
        if v == p: continue
        dfs1(v,u)
        sz[u] += sz[v]
        dp[u] += dp[v] + sz[v]

def dfs2(u,p):
    ans[u] = dp[u]
    for v in adj[u]:
        if v == p: continue
        dp_u, dp_v = dp[u], dp[v]
        sz_u, sz_v = sz[u], sz[v]

        dp[u] -= dp[v] + sz[v]
        sz[u] -= sz[v]
        dp[v] += dp[u] + sz[u]
        sz[v] += sz[u]

        dfs2(v,u)

        dp[u], dp[v] = dp_u, dp_v
        sz[u], sz[v] = sz_u, sz_v

dfs1(1,-1)
dfs2(1,-1)

for i in range(1,n+1):
    print(f"Sum of distances from {i}: {ans[i]}")
```

#### Why It Matters

- Enables $O(n)$ computation of per-node DP values.
- Reuses child and parent information via reversible transitions.
- Crucial for:

  * Distance sums
  * Subtree aggregations
  * Coloring and constraint propagation

You can reroot any tree once you know how to move contributions.

#### A Gentle Proof (Why It Works)

The rerooting relation ensures:

- $dp[u]$ stores full-tree values when rooted at $u$.
- When rerooting to $v$, subtract $v$'s contribution from $u$, then add $u$'s contribution to $v$.

Because each edge is traversed twice, total complexity is $O(n)$.

This is a direct application of DP reusability under tree decomposition.

#### Try It Yourself

1. Compute subtree sums and reroot to get sum of values at distance ≤ k.
2. Apply rerooting to count paths passing through each node.
3. Modify transitions for tree coloring or centroid scoring.
4. Visualize contribution flow parent↔child.

#### Test Cases

| Tree               | Query         | Output     |
| ------------------ | ------------- | ---------- |
| Line (1–2–3–4)     | Distance sums | 6, 4, 4, 6 |
| Star (1–2,1–3,1–4) | Distance sums | 3, 5, 5, 5 |
| Balanced tree      | Aggregation   | symmetric  |

#### Complexity

- Time: $O(n)$
- Space: $O(n)$

Rerooting DP is your "walk the tree" trick, one bottom-up pass, one top-down pass, and you know what every node would see if it stood at the root.

### 499 Iterative DP Pattern

Iterative (bottom-up) dynamic programming is the most systematic and efficient way to compute state-based solutions. Instead of recursion and memoization, we explicitly build tables in increasing order of dependency, turning recurrence relations into simple loops.

#### What Problem Are We Solving?

When you have a recurrence like:

$$
dp[i] = f(dp[i-1], dp[i-2], \ldots)
$$

you don't need recursion, you can iterate from base to target.
This approach avoids call stack overhead, ensures predictable memory access, and simplifies debugging.

Iterative DP is ideal for:

- Counting problems (e.g. Fibonacci, climbing stairs)
- Path minimization (e.g. shortest path, knapsack)
- Sequence alignment (e.g. LCS, edit distance)

#### How Does It Work (Plain Language)

1. Define the state $dp[i]$: what does it represent?
2. Identify base cases (e.g. $dp[0]$, $dp[1]$).
3. Establish transition using smaller states.
4. Iterate from smallest to largest index, ensuring dependencies are filled before use.
5. Extract result (e.g. $dp[n]$ or $\max_i dp[i]$).

The iteration order must match dependency direction.

#### Example: Climbing Stairs

You can climb either 1 or 2 steps at a time.
Number of ways to reach step $n$:

$$
dp[i] = dp[i-1] + dp[i-2]
$$

with base cases $dp[0] = 1$, $dp[1] = 1$.

#### Tiny Code (C)

```c
#include <stdio.h>

int main() {
    int n = 5;
    int dp[6];
    dp[0] = 1;
    dp[1] = 1;
    for (int i = 2; i <= n; i++)
        dp[i] = dp[i-1] + dp[i-2];
    printf("Ways to climb %d stairs: %d\n", n, dp[n]);
}
```

Python

```python
n = 5
dp = [0]*(n+1)
dp[0] = dp[1] = 1
for i in range(2, n+1):
    dp[i] = dp[i-1] + dp[i-2]
print(f"Ways to climb {n} stairs: {dp[n]}")
```

#### Why It Matters

- Performance: Iteration eliminates recursion overhead.
- Clarity: Each state is computed once, in a known order.
- Memory Optimization: You can reduce space when only recent states are needed (rolling array).
- Foundation: All advanced DPs (knapsack, edit distance, LIS) can be written iteratively.

#### A Gentle Proof (Why It Works)

If $dp[i]$ depends only on smaller indices, then filling $dp[0 \ldots n]$ in order guarantees correctness.

By induction:

- Base cases true by definition.
- Assuming $dp[0..i-1]$ correct, then $dp[i] = f(dp[0..i-1])$ produces correct result.

No state is used before it's computed.

#### Try It Yourself

1. Implement iterative Fibonacci with constant space.
2. Convert recursive knapsack into iterative table form.
3. Write bottom-up LCS for two strings.
4. Try 2D iterative DP for grid paths.

#### Test Cases

| Input  | Expected Output |
| ------ | --------------- |
| $n=0$  | 1               |
| $n=1$  | 1               |
| $n=5$  | 8               |
| $n=10$ | 89              |

#### Complexity

- Time: $O(n)$
- Space: $O(n)$ (or $O(1)$ with rolling array)

Iterative DP is the canonical form, the simplest, most direct way to think about recursion unrolled into loops.

### 500 Memoization Template

Memoization is the top-down form of dynamic programming, you solve the problem recursively, but store answers so you never recompute the same state twice. It's the natural bridge between pure recursion and iterative DP.

#### What Problem Are We Solving?

Many recursive problems revisit the same subproblems multiple times.
For example, Fibonacci recursion:

$$
F(n) = F(n-1) + F(n-2)
$$

recomputes $F(k)$ many times. Memoization avoids this by caching results after the first computation.

Whenever your recursion tree overlaps, memoization converts exponential time into polynomial time.

#### How Does It Work (Plain Language)

1. Define the state: what parameters describe your subproblem?
2. Check if cached: if already solved, return memoized value.
3. Recurse: compute using smaller states.
4. Store result before returning.
5. Return the cached value next time it's needed.

Memoization is ideal for:

- Recursive definitions (Fibonacci, Knapsack, LCS)
- Combinatorial counting with overlapping subproblems
- Tree/graph traversal with repeated subpaths

#### Example: Fibonacci with Memoization

$$
F(n) =
\begin{cases}
1, & n \le 1,\\
F(n-1) + F(n-2), & \text{otherwise.}
\end{cases}
$$


We store each $F(k)$ the first time it's computed.

#### Tiny Code (C)

```c
#include <stdio.h>

int memo[100];

int fib(int n) {
    if (n <= 1) return 1;
    if (memo[n] != 0) return memo[n];
    return memo[n] = fib(n-1) + fib(n-2);
}

int main() {
    int n = 10;
    printf("Fib(%d) = %d\n", n, fib(n));
}
```

Python

```python
memo = {}
def fib(n):
    if n <= 1:
        return 1
    if n in memo:
        return memo[n]
    memo[n] = fib(n-1) + fib(n-2)
    return memo[n]

print(fib(10))
```

#### Why It Matters

- Bridges recursion and iteration: You keep the elegance of recursion with the performance of DP.
- Faster prototypes: Great for quickly building correct solutions.
- Easier to reason: You only define recurrence, not filling order.
- Transition step: Helps derive bottom-up equivalents later.

#### A Gentle Proof (Why It Works)

We prove correctness by induction:

- Base case: $dp[0]$ and $dp[1]$ defined directly.
- Inductive step: Each call to $f(n)$ only uses smaller arguments $f(k)$, which are correct by the inductive hypothesis.
- Caching: Ensures each $f(k)$ computed exactly once, guaranteeing $O(n)$ total calls.

Thus, memoization preserves recursion semantics while achieving optimal time.

#### Try It Yourself

1. Write memoized knapsack with signature `solve(i, w)`
2. Memoize subset sum (`solve(i, sum)`)
3. Build LCS recursively with `(i, j)` as state
4. Compare memoized and bottom-up versions for runtime

#### Test Cases

| Input     | Expected Output |
| --------- | --------------- |
| $fib(0)$  | 1               |
| $fib(1)$  | 1               |
| $fib(5)$  | 8               |
| $fib(10)$ | 89              |

#### Complexity

- Time: $O(n)$ (each state computed once)
- Space: $O(n)$ recursion + cache

Memoization is the conceptual core of DP, it reveals how subproblems overlap and prepares you for crafting iterative solutions.
