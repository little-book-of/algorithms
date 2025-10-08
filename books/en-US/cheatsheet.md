# The Cheatsheet 

### Page 1. Big Picture and Complexity

A quick reference for understanding algorithms, efficiency, and growth rates.
Keep this sheet beside you as you read or code.

#### What Is an Algorithm?

An algorithm is a clear, step-by-step process that solves a problem.

| Property      | Description                               |
| ------------- | ----------------------------------------- |
| Precise       | Each step is unambiguous                  |
| Finite        | Must stop after a certain number of steps |
| Effective     | Every step is doable by machine or human  |
| Deterministic | Same input, same output (usually)         |

Think of it like a recipe:

- Input: ingredients
- Steps: instructions
- Output: final dish

#### Core Qualities

| Concept     | Question to Ask                        |
| ----------- | -------------------------------------- |
| Correctness | Does it always solve the problem       |
| Termination | Does it eventually stop                |
| Complexity  | How much time and space it needs       |
| Clarity     | Is it easy to understand and implement |

#### Why Complexity Matters

Different algorithms grow differently as input size $n$ increases.

| Growth Rate  | Example Algorithm        | Effect When $n$ Doubles |
| ------------ | ------------------------ | ----------------------- |
| $O(1)$       | Hash lookup              | No change               |
| $O(\log n)$  | Binary search            | Slight increase         |
| $O(n)$       | Linear scan              | Doubled                 |
| $O(n\log n)$ | Merge sort               | Slightly more than 2×   |
| $O(n^2)$     | Bubble sort              | Quadrupled              |
| $O(2^n)$     | Subset generation        | Explodes                |
| $O(n!)$      | Brute-force permutations | Unusable beyond $n=10$  |

#### Measuring Time and Space

| Measure          | Meaning                                     | Example                      |
| ---------------- | ------------------------------------------- | ---------------------------- |
| Time Complexity  | Number of operations                        | Loop from 1 to $n$: $O(n)$   |
| Space Complexity | Memory usage (stack, heap, data structures) | Recursive call depth: $O(n)$ |

Simple rules:

- Sequential steps: sum of costs
- Nested loops: product of sizes
- Recursion: use recurrence relations

#### Common Patterns

| Pattern                       | Cost Formula         | Complexity   |
| ----------------------------- | -------------------- | ------------ |
| Single Loop (1 to $n$)        | $T(n) = n$           | $O(n)$       |
| Nested Loops ($n \times n$)   | $T(n) = n^2$         | $O(n^2)$     |
| Halving Each Step             | $T(n) = \log_2 n$    | $O(\log n)$  |
| Divide and Conquer (2 halves) | $T(n) = 2T(n/2) + n$ | $O(n\log n)$ |

#### Doubling Rule

Run algorithm for $n$ and $2n$:

| Observation       | Likely Complexity |
| ----------------- | ----------------- |
| Constant time     | $O(1)$            |
| Time doubles      | $O(n)$            |
| Time quadruples   | $O(n^2)$          |
| Time × log factor | $O(n\log n)$      |

#### Tiny Code: Binary Search

```python
def binary_search(arr, x):
    lo, hi = 0, len(arr) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if arr[mid] == x:
            return mid
        elif arr[mid] < x:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1
```

Complexity:
$$T(n) = T(n/2) + 1 \Rightarrow O(\log n)$$

#### Common Pitfalls

| Issue                     | Tip                                       |
| ------------------------- | ----------------------------------------- |
| Off-by-one error          | Check loop bounds carefully               |
| Infinite loop             | Ensure termination condition is reachable |
| Midpoint overflow (C/C++) | Use `mid = lo + (hi - lo) / 2`            |
| Unsorted data in search   | Binary search only works on sorted input  |

#### Quick Growth Summary

| Type         | Formula Example | Description            |
| ------------ | --------------- | ---------------------- |
| Constant     | $1$             | Fixed time             |
| Logarithmic  | $\log n$        | Divide each time       |
| Linear       | $n$             | Step through all items |
| Linearithmic | $n \log n$      | Sort-like complexity   |
| Quadratic    | $n^2$           | Double loop            |
| Cubic        | $n^3$           | Triple nested loops    |
| Exponential  | $2^n$           | All subsets            |
| Factorial    | $n!$            | All permutations       |

#### Simple Rule of Thumb

Trace small examples by hand.
Count steps, memory, and recursion depth.
You'll see how growth behaves before running code.

### Page 2. Recurrences and Master Theorem

This page helps you break down recursive algorithms and estimate their runtime using recurrences.

#### What Is a Recurrence?

A recurrence relation expresses a problem's cost $T(n)$ in terms of smaller subproblems.

Typical structure:

$$
T(n) = a , T\left(\frac{n}{b}\right) + f(n)
$$

where:

- $a$ = number of subproblems
- $b$ = factor by which input shrinks
- $f(n)$ = extra work per call (merge, combine, etc.)

#### Common Recurrences

| Algorithm          | Recurrence Form       | Solution          |
| ------------------ | --------------------- | ----------------- |
| Binary Search      | $T(n)=T(n/2)+1$       | $O(\log n)$       |
| Merge Sort         | $T(n)=2T(n/2)+n$      | $O(n\log n)$      |
| Quick Sort (avg)   | $T(n)=2T(n/2)+O(n)$   | $O(n\log n)$      |
| Quick Sort (worst) | $T(n)=T(n-1)+O(n)$    | $O(n^2)$          |
| Matrix Multiply    | $T(n)=8T(n/2)+O(n^2)$ | $O(n^3)$          |
| Karatsuba          | $T(n)=3T(n/2)+O(n)$   | $O(n^{\log_2 3})$ |

#### Solving Recurrences

There are several methods to solve them:

| Method         | Description                      | Best For           |
| -------------- | -------------------------------- | ------------------ |
| Iteration      | Expand step by step              | Simple recurrences |
| Substitution   | Guess and prove with induction   | Verification       |
| Recursion Tree | Visualize total work per level   | Divide and conquer |
| Master Theorem | Shortcut for $T(n)=aT(n/b)+f(n)$ | Standard forms     |

#### The Master Theorem

Given $$T(n) = aT(n/b) + f(n)$$

Let $$n^{\log_b a}$$ be the "critical term"

| Case | Condition                                                           | Result                                    |
| ---- | ------------------------------------------------------------------- | ----------------------------------------- |
| 1    | If $f(n) = O(n^{\log_b a - \varepsilon})$                           | $T(n) = \Theta(n^{\log_b a})$             |
| 2    | If $f(n) = \Theta(n^{\log_b a}\log^k n)$                            | $T(n) = \Theta(n^{\log_b a}\log^{k+1} n)$ |
| 3    | If $f(n) = \Omega(n^{\log_b a + \varepsilon})$ and regularity holds | $T(n) = \Theta(f(n))$                     |

#### Examples

| Algorithm         | $a$ | $b$ | $f(n)$ | Case | $T(n)$                 |
| ----------------- | --- | --- | ------ | ---- | ---------------------- |
| Merge Sort        | 2   | 2   | $n$    | 2    | $\Theta(n\log n)$      |
| Binary Search     | 1   | 2   | $1$    | 1    | $\Theta(\log n)$       |
| Strassen Multiply | 7   | 2   | $n^2$  | 2    | $\Theta(n^{\log_2 7})$ |
| Quick Sort (avg)  | 2   | 2   | $n$    | 2    | $\Theta(n\log n)$      |

#### Recursion Tree Visualization

Break cost into levels:

Example: $T(n)=2T(n/2)+n$

| Level | #Nodes | Work per Node | Total Work |
| ----- | ------ | ------------- | ---------- |
| 0     | 1      | $n$           | $n$        |
| 1     | 2      | $n/2$         | $n$        |
| 2     | 4      | $n/4$         | $n$        |
| ...   | ...    | ...           | ...        |

Sum across $\log_2 n$ levels:

$$T(n) = n \log_2 n$$

#### Tiny Code: Fast Exponentiation

Compute $a^n$ efficiently.

```python
def power(a, n):
    res = 1
    while n > 0:
        if n % 2 == 1:
            res *= a
        a *= a
        n //= 2
    return res
```

Recurrence:

$$T(n) = T(n/2) + O(1) \Rightarrow O(\log n)$$

#### Iteration Method Example

Solve $T(n)=T(n/2)+n$

Expand:

$$
\begin{aligned}
T(n) &= T(n/2) + n \
&= T(n/4) + n/2 + n \
&= T(n/8) + n/4 + n/2 + n \
&= \ldots + n(1 + 1/2 + 1/4 + \ldots) \
&= O(n)
\end{aligned}
$$

#### Common Forms

| Form                | Result       |
| ------------------- | ------------ |
| $T(n)=T(n-1)+O(1)$  | $O(n)$       |
| $T(n)=T(n/2)+O(1)$  | $O(\log n)$  |
| $T(n)=2T(n/2)+O(1)$ | $O(n)$       |
| $T(n)=2T(n/2)+O(n)$ | $O(n\log n)$ |
| $T(n)=T(n/2)+O(n)$  | $O(n)$       |

#### Quick Checklist

1. Identify $a$, $b$, and $f(n)$
2. Compare $f(n)$ to $n^{\log_b a}$
3. Apply correct case
4. Confirm assumptions (regularity)
5. State final complexity

Understanding recurrences helps you estimate performance before coding.
Always look for subproblem count, size, and merge cost.

### Page 3. Sorting at a Glance

Sorting is one of the most common algorithmic tasks. This page helps you quickly compare sorting methods, their complexity, stability, and when to use them.

#### Why Sorting Matters

Sorting organizes data so that searches, merges, and analyses become efficient.
Many problems become simpler once the input is sorted.

#### Quick Comparison Table

| Algorithm      | Best Case    | Average Case | Worst Case   | Space       | Stable | In-Place | Notes                        |
| -------------- | ------------ | ------------ | ------------ | ----------- | ------ | -------- | ---------------------------- |
| Bubble Sort    | $O(n)$       | $O(n^2)$     | $O(n^2)$     | $O(1)$      | Yes    | Yes      | Simple, educational          |
| Selection Sort | $O(n^2)$     | $O(n^2)$     | $O(n^2)$     | $O(1)$      | No     | Yes      | Few swaps                    |
| Insertion Sort | $O(n)$       | $O(n^2)$     | $O(n^2)$     | $O(1)$      | Yes    | Yes      | Great for small/partial sort |
| Merge Sort     | $O(n\log n)$ | $O(n\log n)$ | $O(n\log n)$ | $O(n)$      | Yes    | No       | Stable, divide and conquer   |
| Quick Sort     | $O(n\log n)$ | $O(n\log n)$ | $O(n^2)$     | $O(\log n)$ | No     | Yes      | Fast average, in place       |
| Heap Sort      | $O(n\log n)$ | $O(n\log n)$ | $O(n\log n)$ | $O(1)$      | No     | Yes      | Not stable                   |
| Counting Sort  | $O(n+k)$     | $O(n+k)$     | $O(n+k)$     | $O(n+k)$    | Yes    | No       | Integer keys only            |
| Radix Sort     | $O(d(n+k))$  | $O(d(n+k))$  | $O(d(n+k))$  | $O(n+k)$    | Yes    | No       | Sort by digits               |
| Bucket Sort    | $O(n+k)$     | $O(n+k)$     | $O(n^2)$     | $O(n)$      | Yes    | No       | Uniform distribution needed  |

#### Choosing a Sorting Algorithm

| Situation                           | Best Choice           |
| ----------------------------------- | --------------------- |
| Small array or nearly sorted data   | Insertion Sort        |
| Stable required, general case       | Merge Sort or Timsort |
| In-place and fast on average        | Quick Sort            |
| Guarantee worst-case $O(n\log n)$   | Heap Sort             |
| Small integer keys or limited range | Counting or Radix     |
| External sorting (large data)       | External Merge Sort   |

#### Tiny Code: Insertion Sort

Simple and intuitive for beginners.

```python
def insertion_sort(a):
    for i in range(1, len(a)):
        key = a[i]
        j = i - 1
        while j >= 0 and a[j] > key:
            a[j + 1] = a[j]
            j -= 1
        a[j + 1] = key
    return a
```

Complexity:
$$T(n) = O(n^2)$$ average, $$O(n)$$ best (already sorted)

#### Divide and Conquer Sorts

##### Merge Sort

Splits list, sorts halves, merges results.

Recurrence:
$$T(n) = 2T(n/2) + O(n) = O(n\log n)$$

Tiny Code:

```python
def merge_sort(a):
    if len(a) <= 1:
        return a
    mid = len(a)//2
    L = merge_sort(a[:mid])
    R = merge_sort(a[mid:])
    i = j = 0
    res = []
    while i < len(L) and j < len(R):
        if L[i] <= R[j]:
            res.append(L[i]); i += 1
        else:
            res.append(R[j]); j += 1
    res.extend(L[i:]); res.extend(R[j:])
    return res
```

##### Quick Sort

Pick pivot, partition, sort subarrays.

Recurrence:
$$T(n) = T(k) + T(n-k-1) + O(n)$$
Average case: $$O(n\log n)$$
Worst case: $$O(n^2)$$

Tiny Code:

```python
def quick_sort(a):
    if len(a) <= 1:
        return a
    pivot = a[len(a)//2]
    left  = [x for x in a if x < pivot]
    mid   = [x for x in a if x == pivot]
    right = [x for x in a if x > pivot]
    return quick_sort(left) + mid + quick_sort(right)
```

#### Stable vs Unstable

| Property | Description                        | Example               |
| -------- | ---------------------------------- | --------------------- |
| Stable   | Equal elements keep original order | Merge Sort, Insertion |
| Unstable | May reorder equal elements         | Quick, Heap           |

#### Visualization Tips

| Pattern   | Description                     |
| --------- | ------------------------------- |
| Bubble    | Compare and swap adjacent       |
| Selection | Select min each pass            |
| Insertion | Grow sorted region step by step |
| Merge     | Divide, conquer, merge          |
| Quick     | Partition and recurse           |
| Heap      | Build heap, extract repeatedly  |

#### Summary Table

| Type           | Category           | Complexity   | Stable    | Space    |
| -------------- | ------------------ | ------------ | --------- | -------- |
| Simple         | Bubble, Selection  | $O(n^2)$     | Varies    | $O(1)$   |
| Insertion      | Incremental        | $O(n^2)$     | Yes       | $O(1)$   |
| Divide/Conquer | Merge, Quick       | $O(n\log n)$ | Merge yes | Merge no |
| Distribution   | Counting, Radix    | $O(n+k)$     | Yes       | $O(n+k)$ |
| Hybrid         | Timsort, IntroSort | $O(n\log n)$ | Yes       | Varies   |

When in doubt, start with Timsort (Python) or std::sort (C++) which adapt dynamically.

### Page 4. Searching and Selection

Searching means finding what you need from a collection. Selection means picking specific elements such as the smallest, largest, or k-th element. This page summarizes both.

#### Searching Basics

| Type          | Description                       | Data Requirement | Complexity          |
| ------------- | --------------------------------- | ---------------- | ------------------- |
| Linear Search | Check one by one                  | None             | $O(n)$              |
| Binary Search | Divide range by 2 each step       | Sorted           | $O(\log n)$         |
| Jump Search   | Skip ahead fixed steps            | Sorted           | $O(\sqrt n)$        |
| Interpolation | Guess position based on value     | Sorted, uniform  | $O(\log\log n)$ avg |
| Exponential   | Expand window, then binary search | Sorted           | $O(\log n)$         |

#### Linear Search

Simple but slow for large inputs.

```python
def linear_search(a, x):
    for i, v in enumerate(a):
        if v == x:
            return i
    return -1
```

Complexity:
$$T(n) = O(n)$$

#### Binary Search

Fast on sorted lists.

```python
def binary_search(a, x):
    lo, hi = 0, len(a) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if a[mid] == x:
            return mid
        elif a[mid] < x:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1
```

Complexity:
$$T(n) = T(n/2) + 1 \Rightarrow O(\log n)$$

#### Binary Search Variants

| Variant     | Goal                           | Return Value                 |
| ----------- | ------------------------------ | ---------------------------- |
| Lower Bound | First index where $a[i] \ge x$ | Position of first ≥ x        |
| Upper Bound | First index where $a[i] > x$   | Position of first > x        |
| Count Range | `upper_bound - lower_bound`    | Count of $x$ in sorted array |

#### Common Binary Search Pitfalls

| Problem                      | Fix                           |
| ---------------------------- | ----------------------------- |
| Infinite loop                | Update bounds correctly       |
| Off-by-one                   | Check mid inclusion carefully |
| Unsuitable for unsorted data | Sort or use hash-based search |
| Overflow (C/C++)             | `mid = lo + (hi - lo) / 2`    |

#### Exponential Search

Used for unbounded or large sorted lists.

1. Check positions $1, 2, 4, 8, ...$ until $a[i] \ge x$
2. Binary search in last found interval

Complexity:
$$O(\log n)$$

#### Selection Problems

Find the $k$-th smallest or largest element.

| Task              | Example Use Case            | Algorithm        | Complexity   |
| ----------------- | --------------------------- | ---------------- | ------------ |
| Min / Max         | Smallest / largest element  | Linear Scan      | $O(n)$       |
| k-th Smallest     | Order statistic             | Quickselect      | Avg $O(n)$   |
| Median            | Middle element              | Quickselect      | Avg $O(n)$   |
| Top-k Elements    | Partial sort                | Heap / Partition | $O(n\log k)$ |
| Median of Medians | Worst-case linear selection | Deterministic    | $O(n)$       |

#### Tiny Code: Quickselect (k-th smallest)

```python
import random

def quickselect(a, k):
    if len(a) == 1:
        return a[0]
    pivot = random.choice(a)
    left  = [x for x in a if x < pivot]
    mid   = [x for x in a if x == pivot]
    right = [x for x in a if x > pivot]

    if k < len(left):
        return quickselect(left, k)
    elif k < len(left) + len(mid):
        return pivot
    else:
        return quickselect(right, k - len(left) - len(mid))
```

Complexity:
Average $O(n)$, Worst $O(n^2)$

#### Tiny Code: Lower Bound

```python
def lower_bound(a, x):
    lo, hi = 0, len(a)
    while lo < hi:
        mid = (lo + hi) // 2
        if a[mid] < x:
            lo = mid + 1
        else:
            hi = mid
    return lo
```

#### Hash-Based Searching

When order does not matter, hashing gives near constant lookup.

| Operation | Average | Worst  |
| --------- | ------- | ------ |
| Insert    | $O(1)$  | $O(n)$ |
| Search    | $O(1)$  | $O(n)$ |
| Delete    | $O(1)$  | $O(n)$ |

Best for large, unsorted collections.

#### Summary Table

| Scenario                   | Recommended Approach | Complexity  |
| -------------------------- | -------------------- | ----------- |
| Small array                | Linear Search        | $O(n)$      |
| Large, sorted array        | Binary Search        | $O(\log n)$ |
| Unbounded range            | Exponential Search   | $O(\log n)$ |
| Need k-th smallest element | Quickselect          | Avg $O(n)$  |
| Many lookups               | Hash Table           | Avg $O(1)$  |

#### Quick Tips

- Always check whether data is sorted before applying binary search.
- Quickselect is great when you only need the k-th element, not a full sort.
- Use hash maps for fast lookups on unsorted data.

### Page 5. Core Data Structures

Data structures organize data for efficient access and modification.
Choosing the right one often makes an algorithm simple and fast.

#### Arrays and Lists

| Structure       | Access | Search | Insert End       | Insert Middle        | Delete               | Notes             |
| --------------- | ------ | ------ | ---------------- | -------------------- | -------------------- | ----------------- |
| Static Array    | $O(1)$ | $O(n)$ | N/A              | $O(n)$               | $O(n)$               | Fixed size        |
| Dynamic Array   | $O(1)$ | $O(n)$ | Amortized $O(1)$ | $O(n)$               | $O(n)$               | Auto-resizing     |
| Linked List (S) | $O(n)$ | $O(n)$ | $O(1)$ head      | $O(1)$ if node known | $O(1)$ if node known | Sequential access |
| Linked List (D) | $O(n)$ | $O(n)$ | $O(1)$ head/tail | $O(1)$ if node known | $O(1)$ if node known | Two-way traversal |

- Singly linked lists: next pointer only
- Doubly linked lists: next and prev pointers
- Dynamic arrays use *doubling* to grow capacity

#### Tiny Code: Dynamic Array Resize (Python-like)

```python
def resize(arr, new_cap):
    new = [None] * new_cap
    for i in range(len(arr)):
        new[i] = arr[i]
    return new
```

Doubling capacity keeps amortized append $O(1)$.

#### Stacks and Queues

| Structure    | Push   | Pop    | Peek   | Notes                      |
| ------------ | ------ | ------ | ------ | -------------------------- |
| Stack (LIFO) | $O(1)$ | $O(1)$ | $O(1)$ | Undo operations, recursion |
| Queue (FIFO) | $O(1)$ | $O(1)$ | $O(1)$ | Scheduling, BFS            |
| Deque        | $O(1)$ | $O(1)$ | $O(1)$ | Insert/remove both ends    |

#### Tiny Code: Stack

```python
stack = []
stack.append(x)   # push
x = stack.pop()   # pop
```

#### Tiny Code: Queue

```python
from collections import deque

q = deque()
q.append(x)   # enqueue
x = q.popleft()  # dequeue
```

#### Priority Queue (Heap)

Stores elements so the smallest (or largest) is always on top.

| Operation   | Complexity  |
| ----------- | ----------- |
| Insert      | $O(\log n)$ |
| Extract min | $O(\log n)$ |
| Peek min    | $O(1)$      |
| Build heap  | $O(n)$      |

Tiny Code:

```python
import heapq
heap = []
heapq.heappush(heap, value)
x = heapq.heappop(heap)
```

Heaps are used in Dijkstra, Prim, and scheduling.

#### Hash Tables

| Operation | Average | Worst  | Notes                               |
| --------- | ------- | ------ | ----------------------------------- |
| Insert    | $O(1)$  | $O(n)$ | Hash collisions increase cost       |
| Search    | $O(1)$  | $O(n)$ | Good hash + low load factor helps   |
| Delete    | $O(1)$  | $O(n)$ | Usually open addressing or chaining |

Key ideas:

- Compute index using hash function: `index = hash(key) % capacity`
- Resolve collisions by chaining or probing

#### Tiny Code: Hash Map (Simplified)

```python
table = [[] for _ in range(8)]
def put(key, value):
    i = hash(key) % len(table)
    for kv in table[i]:
        if kv[0] == key:
            kv[1] = value
            return
    table[i].append([key, value])
```

#### Sets

A hash-based collection of unique elements.

| Operation | Average Complexity |
| --------- | ------------------ |
| Add       | $O(1)$             |
| Search    | $O(1)$             |
| Remove    | $O(1)$             |

Used for membership checks and duplicates removal.

#### Union-Find (Disjoint Set)

Keeps track of connected components.
Two main operations:

- find(x): get representative of x
- union(a,b): merge sets of a and b

With path compression + union by rank → nearly $O(1)$.

Tiny Code:

```python
class DSU:
    def __init__(self, n):
        self.p = list(range(n))
        self.r = [0]*n
    def find(self, x):
        if self.p[x] != x:
            self.p[x] = self.find(self.p[x])
        return self.p[x]
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb: return
        if self.r[ra] < self.r[rb]: ra, rb = rb, ra
        self.p[rb] = ra
        if self.r[ra] == self.r[rb]:
            self.r[ra] += 1
```

#### Summary Table

| Category     | Structure       | Use Case                          |
| ------------ | --------------- | --------------------------------- |
| Sequence     | Array, List     | Ordered data                      |
| LIFO/FIFO    | Stack, Queue    | Recursion, scheduling             |
| Priority     | Heap            | Best-first selection, PQ problems |
| Hash-based   | Hash Table, Set | Fast lookups, uniqueness          |
| Connectivity | Union-Find      | Graph components, clustering      |

#### Quick Tips

- Choose array when random access matters.
- Choose list when insertions/deletions frequent.
- Choose stack or queue for control flow.
- Choose heap for priority.
- Choose hash table for constant lookups.
- Choose DSU for disjoint sets or graph merging.

### Page 6. Graphs Quick Use

Graphs model connections between objects.
They appear everywhere: maps, networks, dependencies, and systems.
This page gives you a compact view of common graph algorithms.

#### Graph Basics

A graph has vertices (nodes) and edges (connections).

| Type               | Description                  |
| ------------------ | ---------------------------- |
| Undirected         | Edges go both ways           |
| Directed (Digraph) | Edges have direction         |
| Weighted           | Edges carry cost or distance |
| Unweighted         | All edges cost 1             |

#### Representations

| Representation   | Space    | Best For              | Notes                      |
| ---------------- | -------- | --------------------- | -------------------------- |
| Adjacency List   | $O(V+E)$ | Sparse graphs         | Common in practice         |
| Adjacency Matrix | $O(V^2)$ | Dense graphs          | Constant-time edge lookup  |
| Edge List        | $O(E)$   | Edge-based algorithms | Easy to iterate over edges |

Adjacency List Example (Python):

```python
graph = {
    0: [(1, 2), (2, 5)],
    1: [(2, 1)],
    2: []
}
```

Each tuple `(neighbor, weight)` represents an edge.

#### Traversals

##### Breadth-First Search (BFS)

Visits layer by layer (good for shortest paths in unweighted graphs).

```python
from collections import deque
def bfs(adj, s):
    dist = {s: 0}
    q = deque([s])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if v not in dist:
                dist[v] = dist[u] + 1
                q.append(v)
    return dist
```

Complexity: $O(V+E)$

##### Depth-First Search (DFS)

Explores deeply before backtracking.

```python
def dfs(adj, u, visited):
    visited.add(u)
    for v in adj[u]:
        if v not in visited:
            dfs(adj, v, visited)
```

Complexity: $O(V+E)$

#### Shortest Path Algorithms

| Algorithm      | Works On          | Negative Edges | Complexity       | Notes                   |
| -------------- | ----------------- | -------------- | ---------------- | ----------------------- |
| BFS            | Unweighted        | No             | $O(V+E)$         | Shortest hops           |
| Dijkstra       | Weighted (nonneg) | No             | $O((V+E)\log V)$ | Uses priority queue     |
| Bellman-Ford   | Weighted          | Yes            | $O(VE)$          | Detects negative cycles |
| Floyd-Warshall | All pairs         | Yes            | $O(V^3)$         | DP approach             |

#### Tiny Code: Dijkstra's Algorithm

```python
import heapq

def dijkstra(adj, s):
    INF = 1018
    dist = [INF] * len(adj)
    dist[s] = 0
    pq = [(0, s)]
    while pq:
        d, u = heapq.heappop(pq)
        if d != dist[u]: 
            continue
        for v, w in adj[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(pq, (nd, v))
    return dist
```

#### Topological Sort (DAGs only)

Orders nodes so every edge $(u,v)$ goes from earlier to later.

| Method      | Idea                         | Complexity |
| ----------- | ---------------------------- | ---------- |
| DFS-based   | Post-order stack reversal    | $O(V+E)$   |
| Kahn's Algo | Remove nodes with indegree 0 | $O(V+E)$   |

#### Minimum Spanning Tree (MST)

Connect all nodes with minimum total weight.

| Algorithm | Idea                       | Complexity   | Notes                     |
| --------- | -------------------------- | ------------ | ------------------------- |
| Kruskal   | Sort edges, use Union-Find | $O(E\log E)$ | Works well with edge list |
| Prim      | Grow tree using PQ         | $O(E\log V)$ | Starts from any vertex    |

#### Tiny Code: Kruskal MST

```python
def kruskal(edges, n):
    parent = list(range(n))
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    res = 0
    for w, u, v in sorted(edges):
        ru, rv = find(u), find(v)
        if ru != rv:
            res += w
            parent[rv] = ru
    return res
```

#### Strongly Connected Components (SCC)

Subsets where every node can reach every other.
Use Kosaraju or Tarjan algorithm, both $O(V+E)$.

#### Cycle Detection

| Graph Type | Method               | Notes                      |
| ---------- | -------------------- | -------------------------- |
| Undirected | DFS with parent      | Edge to non-parent visited |
| Directed   | DFS with color/state | Back edge found = cycle    |

#### Summary Table

| Task                       | Algorithm        | Complexity   | Notes                   |
| -------------------------- | ---------------- | ------------ | ----------------------- |
| Visit all nodes            | DFS / BFS        | $O(V+E)$     | Traversal               |
| Shortest path (unweighted) | BFS              | $O(V+E)$     | Counts edges            |
| Shortest path (weighted)   | Dijkstra         | $O(E\log V)$ | No negative weights     |
| Negative edges allowed     | Bellman-Ford     | $O(VE)$      | Detects negative cycles |
| All-pairs shortest path    | Floyd-Warshall   | $O(V^3)$     | DP matrix               |
| MST                        | Kruskal / Prim   | $O(E\log V)$ | Minimal connection cost |
| DAG order                  | Topological Sort | $O(V+E)$     | Only for DAGs           |

#### Quick Tips

- Use BFS for shortest path in unweighted graphs.
- Use Dijkstra if weights are nonnegative.
- Use Union-Find for Kruskal MST.
- Use Topological Sort for dependency resolution.
- Always check for negative edges before using Dijkstra.

### Page 7. Dynamic Programming Quick Use

Dynamic Programming (DP) is about solving big problems by breaking them into overlapping subproblems and reusing their solutions. This page helps you see patterns quickly.

#### When to Use DP

You can usually apply DP if:

| Symptom                 | Meaning                             |
| ----------------------- | ----------------------------------- |
| Optimal Substructure    | Best solution uses best of subparts |
| Overlapping Subproblems | Same subresults appear again        |
| Decision + Recurrence   | State transitions can be defined    |

#### DP Styles

| Style               | Description               | Example                    |
| ------------------- | ------------------------- | -------------------------- |
| Top-down (Memo)     | Recursion + cache results | Fibonacci with memoization |
| Bottom-up (Tabular) | Iterative fill table      | Knapsack table             |
| Space-optimized     | Reuse previous row/state  | Rolling arrays             |

#### Fibonacci Example

Recurrence:
$$F(n)=F(n-1)+F(n-2),\quad F(0)=0,F(1)=1$$

##### Top-down (Memoization)

```python
def fib(n, memo={}):
    if n <= 1:
        return n
    if n not in memo:
        memo[n] = fib(n-1, memo) + fib(n-2, memo)
    return memo[n]
```

##### Bottom-up (Tabulation)

```python
def fib(n):
    dp = [0, 1]
    for i in range(2, n + 1):
        dp.append(dp[i-1] + dp[i-2])
    return dp[n]
```

#### Steps to Solve DP Problems

1. Define State
   Example: $dp[i]$ = best answer for first $i$ items
2. Define Transition
   Example: $dp[i]=\max(dp[i-1], value[i]+dp[i-weight[i]])$
3. Set Base Cases
   Example: $dp[0]=0$
4. Choose Order
   Bottom-up or Top-down
5. Return Answer
   Often $dp[n]$ or $dp[target]$

#### Common DP Categories

| Category  | Example Problems                   | State Form                   |
| --------- | ---------------------------------- | ---------------------------- |
| Sequence  | LIS, LCS, Edit Distance            | $dp[i][j]$ over prefixes     |
| Subset    | Knapsack, Subset Sum               | $dp[i][w]$ capacity-based    |
| Partition | Palindrome Partitioning, Equal Sum | $dp[i]$ cut-based            |
| Grid      | Min Path Sum, Unique Paths         | $dp[i][j]$ over cells        |
| Counting  | Coin Change Count, Stairs          | Add ways from subproblems    |
| Interval  | Matrix Chain, Burst Balloons       | $dp[i][j]$ range subproblem  |
| Bitmask   | TSP, Assignment                    | $dp[mask][i]$ subset states  |
| Digit     | Count numbers with constraint      | $dp[pos][tight][sum]$ digits |
| Tree      | Rerooting, Subtree DP              | $dp[u]$ over children        |

#### Classic Problems

| Problem                     | State Definition                   | Transition                             |
| --------------------------- | ---------------------------------- | -------------------------------------- |
| Climbing Stairs             | $dp[i]=$ ways to reach step i      | $dp[i]=dp[i-1]+dp[i-2]$                |
| Coin Change (Count Ways)    | $dp[x]=$ ways to make sum x        | $dp[x]+=dp[x-coin]$                    |
| 0/1 Knapsack                | $dp[w]=$ max value under weight w  | $dp[w]=\max(dp[w],dp[w-w_i]+v_i)$      |
| Longest Increasing Subseq.  | $dp[i]=$ LIS ending at i           | if $a[j]<a[i]$, $dp[i]=dp[j]+1$        |
| Edit Distance               | $dp[i][j]=$ edit cost              | min(insert,delete,replace)             |
| Matrix Chain Multiplication | $dp[i][j]=$ min cost mult subchain | $dp[i][j]=\min_k(dp[i][k]+dp[k+1][j])$ |

#### Tiny Code: 0/1 Knapsack (1D optimized)

```python
def knapsack(weights, values, W):
    dp = [0]*(W+1)
    for i in range(len(weights)):
        for w in range(W, weights[i]-1, -1):
            dp[w] = max(dp[w], dp[w-weights[i]] + values[i])
    return dp[W]
```

#### Sequence Alignment Example

Edit Distance Recurrence:

$$
dp[i][j] =
\begin{cases}
dp[i-1][j-1], & \text{if } s[i] = t[j],\\
1 + \min(dp[i-1][j],\ dp[i][j-1],\ dp[i-1][j-1]), & \text{otherwise.}
\end{cases}
$$

#### Optimization Techniques

| Technique             | When to Use              | Example                 |
| --------------------- | ------------------------ | ----------------------- |
| Space Optimization    | 2D → 1D states reuse     | Knapsack, LCS           |
| Prefix/Suffix Precomp | Range aggregates         | Sum/Min queries         |
| Divide & Conquer DP   | Monotonic decisions      | Matrix Chain            |
| Convex Hull Trick     | Linear transition minima | DP on lines             |
| Bitset DP             | Large boolean states     | Subset sum optimization |

#### Debugging Tips

- Print partial `dp` arrays to see progress.
- Check base cases carefully.
- Ensure loops match transition dependencies.
- Always confirm the recurrence before coding.

### Page 8. Mathematics for Algorithms Quick Use

Mathematics builds the foundation for algorithmic reasoning.
This page collects essential formulas and methods every programmer should know.

#### Number Theory Essentials

| Topic            | Description             | Formula / Idea                       |
| ---------------- | ----------------------- | ------------------------------------ |
| GCD (Euclidean)  | Greatest common divisor | $gcd(a,b)=gcd(b,a%b)$                |
| Extended GCD     | Solve $ax+by=gcd(a,b)$  | Backtrack coefficients               |
| LCM              | Least common multiple   | $lcm(a,b)=\frac{a\cdot b}{gcd(a,b)}$ |
| Modular Addition | Add under modulo M      | $(a+b)\bmod M$                       |
| Modular Multiply | Multiply under modulo M | $(a\cdot b)\bmod M$                  |
| Modular Inverse  | $a^{-1}\bmod M$         | $a^{M-2}\bmod M$ if M is prime       |
| Modular Exponent | Fast exponentiation     | Square and multiply                  |
| CRT              | Combine congruences     | Solve system $x\equiv a_i\pmod{m_i}$ |

Tiny Code (Modular Exponentiation):

```python
def modpow(a, n, M):
    res = 1
    while n:
        if n & 1:
            res = res * a % M
        a = a * a % M
        n >>= 1
    return res
```

#### Primality and Factorization

| Algorithm             | Use Case                | Complexity       | Notes                  |
| --------------------- | ----------------------- | ---------------- | ---------------------- |
| Trial Division        | Small n                 | $O(\sqrt{n})$    | Simple                 |
| Sieve of Eratosthenes | Generate primes         | $O(n\log\log n)$ | Classic prime sieve    |
| Miller–Rabin          | Probabilistic primality | $O(k\log^3 n)$   | Fast for big n         |
| Pollard Rho           | Factor composite        | $O(n^{1/4})$     | Randomized             |
| Sieve of Atkin        | Faster variant          | $O(n)$           | Complex implementation |

#### Combinatorics

| Formula                                                       | Description             |
| ------------------------------------------------------------- | ----------------------- |
| $n! = n\cdot(n-1)\cdots1$                                     | Factorial               |
| $\binom{n}{k}=\dfrac{n!}{k!(n-k)!}$                           | Number of combinations  |
| $P(n,k)=\dfrac{n!}{(n-k)!}$                                   | Number of permutations  |
| Pascal's Rule: $\binom{n}{k}=\binom{n-1}{k}+\binom{n-1}{k-1}$ | Build Pascal's Triangle |
| Catalan: $C_n=\dfrac{1}{n+1}\binom{2n}{n}$                    | Parentheses counting    |

Tiny Code (nCr with factorials mod M):

```python
def nCr(n, r, fact, inv):
    return fact[n]*inv[r]%M*inv[n-r]%M
```

#### Probability Basics

| Concept        | Formula or Idea                              |                             |                |
| -------------- | -------------------------------------------- | --------------------------- | -------------- |
| Probability    | $P(A)=\frac{\text{favorable}}{\text{total}}$ |                             |                |
| Complement     | $P(\bar{A})=1-P(A)$                          |                             |                |
| Union          | $P(A\cup B)=P(A)+P(B)-P(A\cap B)$            |                             |                |
| Conditional    | $P(A                                         | B)=\frac{P(A\cap B)}{P(B)}$ |                |
| Bayes' Theorem | $P(A                                         | B)=\frac{P(B                | A)P(A)}{P(B)}$ |
| Expected Value | $E[X]=\sum x_iP(x_i)$                        |                             |                |
| Variance       | $Var(X)=E[X^2]-E[X]^2$                       |                             |                |

#### Linear Algebra Core

| Operation            | Formula / Method               | Complexity |
| -------------------- | ------------------------------ | ---------- |
| Gaussian Elimination | Solve $Ax=b$                   | $O(n^3)$   |
| Determinant          | Product of pivots              | $O(n^3)$   |
| Matrix Multiply      | $(AB)*{ij}=\sum_kA*{ik}B_{kj}$ | $O(n^3)$   |
| Transpose            | $A^T_{ij}=A_{ji}$              | $O(n^2)$   |
| LU Decomposition     | $A=LU$ (lower, upper)          | $O(n^3)$   |
| Cholesky             | $A=LL^T$ (symmetric pos. def.) | $O(n^3)$   |
| Power Method         | Dominant eigenvalue estimation | iterative  |

Tiny Code (Gaussian Elimination Skeleton):

```python
for i in range(n):
    pivot = a[i][i]
    for j in range(i, n+1):
        a[i][j] /= pivot
    for k in range(n):
        if k != i:
            ratio = a[k][i]
            for j in range(i, n+1):
                a[k][j] -= ratio*a[i][j]
```

#### Fast Transforms

| Transform | Use Case               | Complexity   | Notes           |
| --------- | ---------------------- | ------------ | --------------- |
| FFT       | Polynomial convolution | $O(n\log n)$ | Complex numbers |
| NTT       | Modular convolution    | $O(n\log n)$ | Prime modulus   |
| FWT (XOR) | XOR-based convolution  | $O(n\log n)$ | Subset DP       |

FFT Equation:

$$
X_k = \sum_{n=0}^{N-1} x_n e^{-2\pi i kn/N}
$$

#### Numerical Methods

| Method         | Purpose           | Formula or Idea                                           |
| -------------- | ----------------- | --------------------------------------------------------- |
| Bisection      | Root-finding      | Midpoint halve until $f(x)=0$                             |
| Newton–Raphson | Fast convergence  | $x_{n+1}=x_n-\frac{f(x_n)}{f'(x_n)}$                      |
| Secant Method  | Approx derivative | $x_{n+1}=x_n-f(x_n)\frac{x_n-x_{n-1}}{f(x_n)-f(x_{n-1})}$ |
| Simpson's Rule | Integration       | $\int_a^bf(x)dx\approx\frac{h}{3}(f(a)+4f(m)+f(b))$       |

#### Optimization and Calculus

| Concept              | Formula / Idea                                             |
| -------------------- | ---------------------------------------------------------- |
| Derivative           | $f'(x)=\lim_{h\to0}\frac{f(x+h)-f(x)}{h}$                  |
| Gradient Descent     | $x_{k+1}=x_k-\eta\nabla f(x_k)$                            |
| Lagrange Multipliers | $\nabla f=\lambda\nabla g$                                 |
| Convex Function      | $f(\lambda x+(1-\lambda)y)\le\lambda f(x)+(1-\lambda)f(y)$ |

Tiny Code (Gradient Descent):

```python
x = x0
for _ in range(1000):
    grad = df(x)
    x -= lr * grad
```

#### Algebraic Tricks

| Topic             | Formula / Use                         |                             |                       |
| ----------------- | ------------------------------------- | --------------------------- | --------------------- |
| Exponentiation    | $a^n$ via square-multiply             |                             |                       |
| Polynomial Deriv. | $(ax^n)' = n\cdot a x^{n-1}$          |                             |                       |
| Integration       | $\int x^n dx = \frac{x^{n+1}}{n+1}+C$ |                             |                       |
| Möbius Inversion  | $f(n)=\sum_{d                         | n}g(d)\implies g(n)=\sum_{d | n}\mu(d)\cdot f(n/d)$ |

#### Quick Reference Table

| Domain         | Must-Know Algorithm        |
| -------------- | -------------------------- |
| Number Theory  | GCD, Mod Exp, CRT          |
| Combinatorics  | Pascal, Factorial, Catalan |
| Probability    | Bayes, Expected Value      |
| Linear Algebra | Gaussian Elimination       |
| Transforms     | FFT, NTT                   |
| Optimization   | Gradient Descent           |

### Page 9. Strings and Text Algorithms Quick Use

Strings are sequences of characters used in text search, matching, and transformation.
This page gives quick references to classical and modern string techniques.

#### String Fundamentals

| Concept         | Description                            | Example                |
| --------------- | -------------------------------------- | ---------------------- |
| Alphabet        | Set of symbols                         | `{a, b, c}`            |
| String Length   | Number of characters                   | `"hello"` → 5          |
| Substring       | Continuous part of string              | `"ell"` in `"hello"`   |
| Subsequence     | Ordered subset (not necessarily cont.) | `"hlo"` from `"hello"` |
| Prefix / Suffix | Starts / ends part of string           | `"he"`, `"lo"`         |

Indexing: Most algorithms use 0-based indexing.

#### String Search Overview

| Algorithm    | Complexity   | Description                    |
| ------------ | ------------ | ------------------------------ |
| Naive Search | $O(nm)$      | Check all positions            |
| KMP          | $O(n+m)$     | Prefix-suffix skip table       |
| Z-Algorithm  | $O(n+m)$     | Precompute match lengths       |
| Rabin–Karp   | $O(n+m)$ avg | Rolling hash check             |
| Boyer–Moore  | $O(n/m)$ avg | Backward scan, skip mismatches |

#### KMP Prefix Function

Compute prefix-suffix matches for pattern.

| Step    | Meaning                                                      |
| ------- | ------------------------------------------------------------ |
| $pi[i]$ | Longest proper prefix that is also suffix for $pattern[0:i]$ |

Tiny Code:

```python
def prefix_function(p):
    pi = [0]*len(p)
    j = 0
    for i in range(1, len(p)):
        while j > 0 and p[i] != p[j]:
            j = pi[j-1]
        if p[i] == p[j]:
            j += 1
        pi[i] = j
    return pi
```

Search uses `pi` to skip mismatches.

#### Z-Algorithm

Computes length of substring starting at i matching prefix.

| Step   | Meaning                                         |
| ------ | ----------------------------------------------- |
| $Z[i]$ | Longest substring starting at i matching prefix |

Use `$S = pattern + '$' + text$` to find pattern occurrences.

#### Rabin–Karp Rolling Hash

| Idea | Compute hash for window of text, slide, compare |
| ---- | ----------------------------------------------- |

Hash Function:
$$
h(s) = (s_0p^{n-1} + s_1p^{n-2} + \dots + s_{n-1}) \bmod M
$$

Update efficiently when sliding one character.

Tiny Code:

```python
def rolling_hash(s, base=257, mod=109+7):
    h = 0
    for ch in s:
        h = (h*base + ord(ch)) % mod
    return h
```

#### Advanced Pattern Matching

| Algorithm    | Use Case                  | Complexity   |
| ------------ | ------------------------- | ------------ |
| Boyer–Moore  | Large alphabet            | $O(n/m)$ avg |
| Sunday       | Last char shift heuristic | $O(n)$ avg   |
| Bitap        | Approximate match         | $O(nm/w)$    |
| Aho–Corasick | Multi-pattern search      | $O(n+z)$     |

#### Aho–Corasick Automaton

Build a trie from patterns and compute failure links.

| Step         | Description             |
| ------------ | ----------------------- |
| Build Trie   | Add all patterns        |
| Failure Link | Fallback to next prefix |
| Output Link  | Record pattern match    |

Tiny Code Sketch:

```python
from collections import deque

def build_ac(patterns):
    trie = [{}]
    fail = [0]
    for pat in patterns:
        node = 0
        for c in pat:
            node = trie[node].setdefault(c, len(trie))
            if node == len(trie):
                trie.append({})
                fail.append(0)
    # compute failure links
    q = deque()
    for c in trie[0]:
        q.append(trie[0][c])
    while q:
        u = q.popleft()
        for c, v in trie[u].items():
            f = fail[u]
            while f and c not in trie[f]:
                f = fail[f]
            fail[v] = trie[f].get(c, 0)
            q.append(v)
    return trie, fail
```

#### Suffix Structures

| Structure        | Purpose                         | Build Time       |
| ---------------- | ------------------------------- | ---------------- |
| Suffix Array     | Sorted list of suffix indices   | $O(n\log n)$     |
| LCP Array        | Longest Common Prefix of suffix | $O(n)$           |
| Suffix Tree      | Trie of suffixes                | $O(n)$ (Ukkonen) |
| Suffix Automaton | Minimal DFA of substrings       | $O(n)$           |

Suffix Array Doubling Approach:

- Rank substrings of length $2^k$
- Sort and merge using pairs of ranks

LCP via Kasai's Algorithm:
$$
LCP[i]=\text{common prefix of } S[SA[i]:], S[SA[i-1]:]
$$

#### Palindrome Detection

| Algorithm            | Description                   | Complexity |
| -------------------- | ----------------------------- | ---------- |
| Manacher's Algorithm | Longest palindromic substring | $O(n)$     |
| DP Table             | Check substring palindrome    | $O(n^2)$   |
| Center Expansion     | Expand around center          | $O(n^2)$   |

Manacher's Core:

- Transform with separators (`#`)
- Track radius of palindrome around each center

#### Edit Distance Family

| Algorithm            | Description           | Complexity                 |
| -------------------- | --------------------- | -------------------------- |
| Levenshtein Distance | Insert/Delete/Replace | $O(nm)$                    |
| Damerau–Levenshtein  | Add transposition     | $O(nm)$                    |
| Hirschberg           | Space-optimized LCS   | $O(nm)$ time, $O(n)$ space |

Recurrence:
$$
dp[i][j]=\min
\begin{cases}
dp[i-1][j]+1 \
dp[i][j-1]+1 \
dp[i-1][j-1]+(s_i\neq t_j)
\end{cases}
$$

#### Compression Techniques

| Algorithm         | Type             | Idea                               |
| ----------------- | ---------------- | ---------------------------------- |
| Huffman Coding    | Prefix code      | Shorter codes for frequent chars   |
| Arithmetic Coding | Range encoding   | Fractional interval representation |
| LZ77 / LZ78       | Dictionary-based | Reuse earlier substrings           |
| BWT + MTF + RLE   | Block sorting    | Group similar chars before coding  |

Huffman Principle:
Shorter bit strings assigned to higher frequency symbols.

#### Hashing and Checksums

| Algorithm    | Use Case          | Notes                 |
| ------------ | ----------------- | --------------------- |
| CRC32        | Error detection   | Simple polynomial mod |
| MD5          | Hash (legacy)     | Not secure            |
| SHA-256      | Secure hash       | Cryptographic         |
| Rolling Hash | Substring compare | Used in Rabin–Karp    |

#### Quick Reference

| Task                  | Algorithm          | Complexity  |
| --------------------- | ------------------ | ----------- |
| Single pattern search | KMP / Z            | $O(n+m)$    |
| Multi-pattern search  | Aho–Corasick       | $O(n+z)$    |
| Approximate search    | Bitap / Wu–Manber  | $O(kn)$     |
| Substring queries     | Suffix Array + LCP | $O(\log n)$ |
| Palindromes           | Manacher           | $O(n)$      |
| Compression           | Huffman / LZ77     | variable    |
| Edit distance         | DP table           | $O(nm)$     |

### Page 10. Geometry, Graphics, and Spatial Algorithms Quick Use

Geometry helps us solve problems about shapes, distances, and spatial relationships.
This page summarizes core computational geometry techniques with simple formulas and examples.

#### Coordinate Basics

| Concept            | Description                                  | Formula / Example                        |   |   |   |             |
| ------------------ | -------------------------------------------- | ---------------------------------------- | - | - | - | ----------- |
| Point Distance     | Distance between $(x_1,y_1)$ and $(x_2,y_2)$ | $d=\sqrt{(x_2-x_1)^2+(y_2-y_1)^2}$       |   |   |   |             |
| Midpoint           | Between two points                           | $(\frac{x_1+x_2}{2}, \frac{y_1+y_2}{2})$ |   |   |   |             |
| Dot Product        | Angle & projection                           | $\vec{a}\cdot\vec{b}=                    | a |   | b | \cos\theta$ |
| Cross Product (2D) | Signed area, orientation                     | $a\times b = a_xb_y - a_yb_x$            |   |   |   |             |
| Orientation Test   | CCW, CW, collinear check                     | $\text{sign}(a\times b)$                 |   |   |   |             |

Tiny Code (Orientation Test):

```python
def orient(a, b, c):
    val = (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])
    return 0 if val == 0 else (1 if val > 0 else -1)
```

#### Convex Hull

Find the smallest convex polygon enclosing all points.

| Algorithm         | Complexity   | Notes                        |
| ----------------- | ------------ | ---------------------------- |
| Graham Scan       | $O(n\log n)$ | Sort by angle, use stack     |
| Andrew's Monotone | $O(n\log n)$ | Sort by x, build upper/lower |
| Jarvis March      | $O(nh)$      | Wrap hull, h = hull size     |
| Chan's Algorithm  | $O(n\log h)$ | Output-sensitive hull        |

Steps:

1. Sort points
2. Build lower hull
3. Build upper hull
4. Concatenate

#### Closest Pair of Points

Divide-and-conquer approach.

| Step              | Description                     |
| ----------------- | ------------------------------- |
| Split by x        | Divide points into halves       |
| Recurse and merge | Track min distance across strip |

Complexity: $O(n\log n)$

Formula:
$$
d(p,q)=\sqrt{(x_p-x_q)^2+(y_p-y_q)^2}
$$

#### Line Intersection

Two segments $(p_1,p_2)$ and $(q_1,q_2)$ intersect if:

1. Orientations differ
2. Segments overlap on line if collinear

Tiny Code:

```python
def intersect(p1, p2, q1, q2):
    o1 = orient(p1, p2, q1)
    o2 = orient(p1, p2, q2)
    o3 = orient(q1, q2, p1)
    o4 = orient(q1, q2, p2)
    return o1 != o2 and o3 != o4
```

#### Polygon Area (Shoelace Formula)

For vertices $(x_i, y_i)$ in order:

$$
A=\frac{1}{2}\left|\sum_{i=0}^{n-1}(x_iy_{i+1}-x_{i+1}y_i)\right|
$$

Tiny Code:

```python
def area(poly):
    s = 0
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i+1)%n]
        s += x1*y2 - x2*y1
    return abs(s)/2
```

#### Point in Polygon

| Method         | Idea                   | Complexity |
| -------------- | ---------------------- | ---------- |
| Ray Casting    | Count edge crossings   | $O(n)$     |
| Winding Number | Track signed rotations | $O(n)$     |
| Convex Test    | Check all orientations | $O(n)$     |

Ray Casting: Odd number of crossings → inside.

#### Rotating Calipers

Used for:

- Polygon diameter (farthest pair)
- Minimum bounding box
- Width and antipodal pairs

Idea: Sweep around convex hull using tangents.
Complexity: $O(n)$ after hull.

#### Sweep Line Techniques

| Problem              | Method               | Complexity       |
| -------------------- | -------------------- | ---------------- |
| Closest Pair         | Active set by y      | $O(n\log n)$     |
| Segment Intersection | Event-based sweeping | $O((n+k)\log n)$ |
| Rectangle Union Area | Vertical edge events | $O(n\log n)$     |
| Skyline Problem      | Merge by height      | $O(n\log n)$     |

Use balanced trees or priority queues for active sets.

#### Circle Geometry

| Concept                 | Formula                   |
| ----------------------- | ------------------------- |
| Equation                | $(x-x_c)^2+(y-y_c)^2=r^2$ |
| Tangent Length          | $\sqrt{d^2-r^2}$          |
| Two-Circle Intersection | Distance-based geometry   |

#### Spatial Data Structures

| Structure | Use Case                 | Notes                         |
| --------- | ------------------------ | ----------------------------- |
| KD-Tree   | Nearest neighbor search  | Axis-aligned splits           |
| R-Tree    | Range queries            | Bounding boxes hierarchy      |
| Quadtree  | 2D recursive subdivision | Graphics, collision detection |
| Octree    | 3D extension             | Volumetric partitioning       |
| BSP Tree  | Planar splits            | Rendering, collision          |

#### Rasterization and Graphics

| Algorithm       | Purpose                | Notes                        |
| --------------- | ---------------------- | ---------------------------- |
| Bresenham Line  | Draw line integer grid | No floating point            |
| Midpoint Circle | Circle rasterization   | Symmetry exploitation        |
| Scanline Fill   | Polygon fill algorithm | Sort edges, horizontal sweep |
| Z-Buffer        | Hidden surface removal | Per-pixel depth comparison   |
| Phong Shading   | Smooth lighting        | Interpolate normals          |

#### Pathfinding in Space

| Algorithm        | Description             | Notes              |
| ---------------- | ----------------------- | ------------------ |
| A*               | Heuristic shortest path | $f(n)=g(n)+h(n)$   |
| Theta*           | Any-angle path          | Shortcut-based     |
| RRT / RRT*       | Random exploration      | Robotics planning  |
| PRM              | Probabilistic roadmap   | Sampled graph      |
| Visibility Graph | Connect visible points  | Geometric planning |

#### Quick Summary

| Task                           | Algorithm          | Complexity   |
| ------------------------------ | ------------------ | ------------ |
| Convex Hull                    | Graham / Andrew    | $O(n\log n)$ |
| Closest Pair                   | Divide and Conquer | $O(n\log n)$ |
| Segment Intersection Detection | Sweep Line         | $O(n\log n)$ |
| Point in Polygon               | Ray Casting        | $O(n)$       |
| Polygon Area                   | Shoelace Formula   | $O(n)$       |
| Nearest Neighbor Search        | KD-Tree            | $O(\log n)$  |
| Pathfinding                    | A*                 | $O(E\log V)$ |

#### Tip

- Always sort points for geometry preprocessing.
- Use cross product for orientation tests.
- Prefer integer arithmetic when possible to avoid floating errors.

### Page 11. Systems, Databases, and Distributed Algorithms Quick Use

Systems and databases rely on algorithms that manage memory, concurrency, persistence, and coordination.
This page gives an overview of the most important ones.

#### Concurrency Control

Ensures correctness when multiple transactions or threads run at once.

| Method                  | Idea                                     | Notes                           |
| ----------------------- | ---------------------------------------- | ------------------------------- |
| Two-Phase Locking (2PL) | Acquire locks, then release after commit | Guarantees serializability      |
| Strict 2PL              | Hold all locks until commit              | Prevents cascading aborts       |
| Conservative 2PL        | Lock all before execution                | Deadlock-free but less parallel |
| Timestamp Ordering      | Order by timestamps                      | May abort late transactions     |
| Multiversion CC (MVCC)  | Readers get snapshots                    | Used in PostgreSQL, InnoDB      |
| Optimistic CC (OCC)     | Validate at commit                       | Best for low conflict workloads |

#### Tiny Code: Timestamp Ordering

```python
# Simplified
if write_ts[x] > txn_ts or read_ts[x] > txn_ts:
    abort()
else:
    write_ts[x] = txn_ts
```

Each object tracks read and write timestamps.

#### Deadlocks

Circular waits among transactions.

| Detection | Build Wait-For Graph, detect cycle |
| Prevention | Wait-Die (old waits) / Wound-Wait (young aborts) |

Detection Complexity: $O(V+E)$

Tiny Code (Wait-For Graph Cycle Check):

```python
def has_cycle(graph):
    visited, stack = set(), set()
    def dfs(u):
        visited.add(u)
        stack.add(u)
        for v in graph[u]:
            if v not in visited and dfs(v): return True
            if v in stack: return True
        stack.remove(u)
        return False
    return any(dfs(u) for u in graph)
```

#### Logging and Recovery

| Technique       | Description                 | Notes                     |
| --------------- | --------------------------- | ------------------------- |
| Write-Ahead Log | Log before data             | Ensures durability        |
| ARIES           | Analysis, Redo, Undo phases | Industry standard         |
| Checkpointing   | Save consistent snapshot    | Speeds recovery           |
| Shadow Paging   | Copy-on-write updates       | Simpler but less flexible |

Recovery after crash:

1. Analysis: find active transactions
2. Redo: reapply committed changes
3. Undo: revert uncommitted ones

#### Indexing

Accelerates lookups and range queries.

| Index Type      | Description            | Notes                       |
| --------------- | ---------------------- | --------------------------- |
| B-Tree / B+Tree | Balanced multiway tree | Disk-friendly               |
| Hash Index      | Exact match only       | No range queries            |
| GiST / R-Tree   | Spatial data           | Bounding box hierarchy      |
| Inverted Index  | Text search            | Maps token to document list |

B+Tree Complexity: $O(\log_B N)$ (B = branching factor)

Tiny Code (Binary Search in Index):

```python
def search(node, key):
    i = bisect_left(node.keys, key)
    if i < len(node.keys) and node.keys[i] == key:
        return node.values[i]
    if node.is_leaf:
        return None
    return search(node.children[i], key)
```

#### Query Processing

| Step           | Description                     |
| -------------- | ------------------------------- |
| Parsing        | Build abstract syntax tree      |
| Optimization   | Reorder joins, pick indices     |
| Execution Plan | Choose algorithm per operator   |
| Execution      | Evaluate iterators or pipelines |

Common join strategies:

| Join Type       | Complexity           | Notes         |
| --------------- | -------------------- | ------------- |
| Nested Loop     | $O(nm)$              | Simple, slow  |
| Hash Join       | $O(n+m)$             | Build + probe |
| Sort-Merge Join | $O(n\log n+m\log m)$ | Sorted inputs |

#### Caching and Replacement

| Policy     | Description                 | Notes                     |
| ---------- | --------------------------- | ------------------------- |
| LRU        | Evict least recently used   | Simple, temporal locality |
| LFU        | Evict least frequently used | Good for stable patterns  |
| ARC / LIRS | Adaptive hybrid             | Handles mixed workloads   |
| Random     | Random eviction             | Simple, fair              |

Tiny Code (LRU using OrderedDict):

```python
from collections import OrderedDict

class LRU:
    def __init__(self, cap):
        self.cap = cap
        self.cache = OrderedDict()
    def get(self, k):
        if k not in self.cache: return -1
        self.cache.move_to_end(k)
        return self.cache[k]
    def put(self, k, v):
        if k in self.cache: self.cache.move_to_end(k)
        self.cache[k] = v
        if len(self.cache) > self.cap: self.cache.popitem(last=False)
```

#### Distributed Systems Core

| Problem         | Description                 | Typical Solution   |
| --------------- | --------------------------- | ------------------ |
| Consensus       | Agree on value across nodes | Paxos, Raft        |
| Leader Election | Pick coordinator            | Bully, Raft        |
| Replication     | Maintain copies             | Log replication    |
| Partitioning    | Split data                  | Consistent hashing |
| Membership      | Detect nodes                | Gossip protocols   |

#### Raft Consensus (Simplified)

| Phase       | Action                     |
| ----------- | -------------------------- |
| Election    | Nodes vote, elect leader   |
| Replication | Leader appends log entries |
| Commitment  | Once majority acknowledge  |

Safety: Committed entries never change.
Liveness: New leader elected on failure.

Tiny Code Sketch:

```python
if vote_request.term > term:
    term = vote_request.term
    voted_for = candidate
```

#### Consistent Hashing

Distributes keys across nodes smoothly.

| Step                   | Description              |
| ---------------------- | ------------------------ |
| Hash each node to ring | e.g. hash(node_id)       |
| Hash each key          | Find next node clockwise |
| Add/remove node        | Only nearby keys move    |

Used in: Dynamo, Cassandra, Memcached.

#### Fault Tolerance Patterns

| Pattern             | Description                | Example          |
| ------------------- | -------------------------- | ---------------- |
| Replication         | Multiple copies            | Primary-backup   |
| Checkpointing       | Save progress periodically | ML training      |
| Heartbeats          | Liveness detection         | Cluster managers |
| Retry + Backoff     | Handle transient failures  | API calls        |
| Quorum Reads/Writes | Require majority agreement | Cassandra        |

#### Distributed Coordination

| Tool / Protocol | Description              | Example Use      |
| --------------- | ------------------------ | ---------------- |
| ZooKeeper       | Centralized coordination | Locks, config    |
| Raft            | Distributed consensus    | Log replication  |
| Etcd            | Key-value store on Raft  | Cluster metadata |

#### Summary Table

| Topic        | Algorithm / Concept | Complexity  | Notes                 |
| ------------ | ------------------- | ----------- | --------------------- |
| Locking      | 2PL, MVCC, OCC      | varies      | Transaction isolation |
| Deadlock     | Wait-Die, Detection | $O(V+E)$    | Graph-based check     |
| Recovery     | ARIES, WAL          | varies      | Crash recovery        |
| Indexing     | B+Tree, Hash Index  | $O(\log N)$ | Faster queries        |
| Join         | Hash / Sort-Merge   | varies      | Query optimization    |
| Cache        | LRU, LFU            | $O(1)$      | Data locality         |
| Consensus    | Raft, Paxos         | $O(n)$ msg  | Fault tolerance       |
| Partitioning | Consistent Hashing  | $O(1)$ avg  | Scalability           |

#### Quick Tips

- Always ensure serializability in concurrency.
- Use MVCC for read-heavy workloads.
- ARIES ensures durability via WAL.
- For scalability, partition and replicate wisely.
- Consensus is required for shared state correctness.

### Page 12. Algorithms for AI, ML, and Optimization Quick Use

This page gathers classical algorithms that power modern AI and machine learning systems, from clustering and classification to gradient-based learning and metaheuristics.

#### Classical Machine Learning Algorithms

| Category       | Algorithm             | Core Idea                                               | Complexity        |
| -------------- | --------------------- | ------------------------------------------------------- | ----------------- |
| Clustering     | k-Means               | Assign to nearest centroid, update centers              | $O(nkt)$          |
| Clustering     | k-Medoids (PAM)       | Representative points as centers                        | $O(k(n-k)^2)$     |
| Clustering     | Gaussian Mixture (EM) | Soft assignments via probabilities                      | $O(nkd)$ per iter |
| Classification | Naive Bayes           | Apply Bayes rule with feature independence              | $O(nd)$           |
| Classification | Logistic Regression   | Linear + sigmoid activation                             | $O(nd)$           |
| Classification | SVM (Linear)          | Maximize margin via convex optimization                 | $O(nd)$ approx    |
| Classification | k-NN                  | Vote from nearest neighbors                             | $O(nd)$ per query |
| Trees          | Decision Tree (CART)  | Recursive splitting by impurity                         | $O(nd\log n)$     |
| Projection     | LDA / PCA             | Find projection maximizing variance or class separation | $O(d^3)$          |

#### Tiny Code: k-Means

```python
import random, math

def kmeans(points, k, iters=100):
    centroids = random.sample(points, k)
    for _ in range(iters):
        groups = [[] for _ in range(k)]
        for p in points:
            idx = min(range(k), key=lambda i: (p[0]-centroids[i][0])2 + (p[1]-centroids[i][1])2)
            groups[idx].append(p)
        new_centroids = []
        for g in groups:
            if g:
                x = sum(p[0] for p in g)/len(g)
                y = sum(p[1] for p in g)/len(g)
                new_centroids.append((x,y))
            else:
                new_centroids.append(random.choice(points))
        if centroids == new_centroids: break
        centroids = new_centroids
    return centroids
```

#### Linear Models

| Model               | Formula                  | Loss Function                       |
| ------------------- | ------------------------ | ----------------------------------- |
| Linear Regression   | $\hat{y}=w^Tx+b$         | MSE: $\frac{1}{n}\sum(y-\hat{y})^2$ |
| Logistic Regression | $\hat{y}=\sigma(w^Tx+b)$ | Cross-Entropy                       |
| Ridge Regression    | Linear + $L_2$ penalty   | $L=\text{MSE}+\lambda|w|^2$         |
| Lasso Regression    | Linear + $L_1$ penalty   | $L=\text{MSE}+\lambda|w|_1$         |

Tiny Code (Gradient Descent for Linear Regression):

```python
def train(X, y, lr=0.01, epochs=1000):
    w = [0]*len(X[0])
    b = 0
    for _ in range(epochs):
        for i in range(len(y)):
            y_pred = sum(w[j]*X[i][j] for j in range(len(w))) + b
            err = y_pred - y[i]
            for j in range(len(w)):
                w[j] -= lr * err * X[i][j]
            b -= lr * err
    return w, b
```

#### Decision Trees and Ensembles

| Algorithm         | Description                 | Notes                       |
| ----------------- | --------------------------- | --------------------------- |
| ID3 / C4.5 / CART | Split by info gain or Gini  | Recursive, interpretable    |
| Random Forest     | Bagging + Decision Trees    | Reduces variance            |
| Gradient Boosting | Sequential residual fitting | XGBoost, LightGBM, CatBoost |
| AdaBoost          | Weighted weak learners      | Sensitive to noise          |

Impurity Measures:

- Gini: $1-\sum p_i^2$
- Entropy: $-\sum p_i\log_2p_i$

#### Support Vector Machines (SVM)

Finds a maximum margin hyperplane.

Objective:
$$
\min_{w,b} \frac{1}{2}|w|^2 + C\sum\xi_i
$$
subject to $y_i(w^Tx_i+b)\ge1-\xi_i$

Kernel trick enables nonlinear separation:
$$K(x_i,x_j)=\phi(x_i)\cdot\phi(x_j)$$

#### Neural Network Fundamentals

| Component  | Description                 |
| ---------- | --------------------------- |
| Neuron     | $y=\sigma(w\cdot x+b)$      |
| Activation | Sigmoid, ReLU, Tanh         |
| Loss       | MSE, Cross-Entropy          |
| Training   | Gradient Descent + Backprop |
| Optimizers | SGD, Adam, RMSProp          |

Forward Propagation:
$$a^{(l)} = \sigma(W^{(l)}a^{(l-1)}+b^{(l)})$$
Backpropagation computes gradients layer by layer.

#### Gradient Descent Variants

| Variant    | Idea                      | Notes              |
| ---------- | ------------------------- | ------------------ |
| Batch      | Use all data each step    | Stable but slow    |
| Stochastic | Update per sample         | Noisy, fast        |
| Mini-batch | Group updates             | Common practice    |
| Momentum   | Add velocity term         | Faster convergence |
| Adam       | Adaptive moment estimates | Most popular       |

Update Rule:
$$
w = w - \eta \cdot \frac{\partial L}{\partial w}
$$

#### Unsupervised Learning

| Algorithm   | Description               | Notes               |
| ----------- | ------------------------- | ------------------- |
| PCA         | Variance-based projection | Eigen decomposition |
| ICA         | Independent components    | Signal separation   |
| t-SNE       | Preserve local structure  | Visualization only  |
| Autoencoder | NN reconstruction model   | Dimensionality red. |

PCA Formula:
Covariance $C=\frac{1}{n}X^TX$, eigenvectors of $C$ are principal axes.

#### Probabilistic Models

| Model            | Description              | Notes                |                           |     |
| ---------------- | ------------------------ | -------------------- | ------------------------- | --- |
| Naive Bayes      | Independence assumption  | $P(y                 | x)\propto P(y)\prod P(x_i | y)$ |
| HMM              | Sequential hidden states | Viterbi for decoding |                           |     |
| Markov Chains    | Transition probabilities | $P(x_t               | x_{t-1})$                 |     |
| Gaussian Mixture | Soft clustering          | EM algorithm         |                           |     |

#### Optimization and Metaheuristics

| Algorithm           | Category        | Notes                         |
| ------------------- | --------------- | ----------------------------- |
| Gradient Descent    | Convex Opt.     | Differentiable objectives     |
| Newton's Method     | Second-order    | Uses Hessian                  |
| Simulated Annealing | Prob. search    | Escape local minima           |
| Genetic Algorithm   | Evolutionary    | Population-based search       |
| PSO (Swarm)         | Collective move | Inspired by flocking behavior |
| Hill Climbing       | Greedy search   | Local optimization            |

#### Reinforcement Learning Core

| Concept        | Description              | Example          |
| -------------- | ------------------------ | ---------------- |
| Agent          | Learner/decision maker   | Robot, policy    |
| Environment    | Provides states, rewards | Game, simulation |
| Policy         | Mapping state → action   | $\pi(s)=a$       |
| Value Function | Expected return          | $V(s)$, $Q(s,a)$ |

Q-Learning Update:
$$
Q(s,a)\leftarrow Q(s,a)+\alpha(r+\gamma\max_{a'}Q(s',a')-Q(s,a))
$$

Tiny Code:

```python
Q[s][a] += alpha * (r + gamma * max(Q[s_next]) - Q[s][a])
```

#### AI Search Algorithms

| Algorithm   | Description              | Complexity       | Notes                     |
| ----------- | ------------------------ | ---------------- | ------------------------- |
| BFS         | Shortest path unweighted | $O(V+E)$         | Level order search        |
| DFS         | Deep exploration         | $O(V+E)$         | Backtracking              |
| A* Search   | Informed, uses heuristic | $O(E\log V)$     | $f(n)=g(n)+h(n)$          |
| IDA*        | Iterative deepening A*   | Memory efficient | Optimal if $h$ admissible |
| Beam Search | Keep best k states       | Approximate      | NLP decoding              |

#### Evaluation Metrics

| Task           | Metric                      | Formula / Meaning                      |
| -------------- | --------------------------- | -------------------------------------- |
| Classification | Accuracy, Precision, Recall | $\frac{TP}{TP+FP}$, $\frac{TP}{TP+FN}$ |
| Regression     | RMSE, MAE, $R^2$            | Fit and error magnitude                |
| Clustering     | Silhouette Score            | Cohesion vs separation                 |
| Ranking        | MAP, NDCG                   | Order-sensitive                        |

Confusion Matrix:

|          | Pred + | Pred - |
| -------- | ------ | ------ |
| Actual + | TP     | FN     |
| Actual - | FP     | TN     |

#### Summary

| Category       | Algorithm Example             | Notes                    |
| -------------- | ----------------------------- | ------------------------ |
| Clustering     | k-Means, GMM                  | Unsupervised grouping    |
| Classification | Logistic, SVM, Trees          | Supervised labeling      |
| Regression     | Linear, Ridge, Lasso          | Predict continuous value |
| Optimization   | GD, Adam, Simulated Annealing | Minimize loss            |
| Probabilistic  | Bayes, HMM, EM                | Uncertainty modeling     |
| Reinforcement  | Q-Learning, SARSA             | Reward-based learning    |


