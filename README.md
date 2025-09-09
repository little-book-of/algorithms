# The Little Book of Algorithms

The Little Book of Algorithms is part of the Little Book series — short, rigorous, and approachable books on core topics in mathematics, computer science, and systems.
This book covers algorithms: from linear data structures and fundamental techniques to paradigms and graph complexity.

## Formats

- [Download PDF](docs/book.pdf) — print-ready
- [Download EPUB](docs/book.epub) — e-reader friendly
- [View LaTeX](docs/book-latex/book.tex) — `.tex` source
- [Read on GitHub Pages](https://little-book-of.github.io/multilinear-algebra/) — online website

## Quick Start (Quarto)

```bash
# 1) Install Quarto (https://quarto.org)
# 2) Clone this repo, then:

# HTML (default output)
quarto render . --to html

# PDF (requires a TeX distribution)
quarto render . --to pdf

# ePub
quarto render . --to epub
```

By default, Quarto writes HTML to `_site/`. Adjust outputs in `_quarto.yml` as needed.

## Repository Layout

```
/
├── chapter_00.qmd  # Chapter 0 — Foundations
├── chapter_01.qmd  # Chapter 1 — Numbers
├── chapter_02.qmd  # Chapter 2 — Arrays
├── ...             # Chapters 03..26 (see full ToC below)
├── _quarto.yml     # Quarto book config (lists chapter_*.qmd)
├── figs/           # Diagrams & illustrations used across chapters
└── src/            # Multi-language reference implementations
    ├── python/     # e.g., src/python/ch08_sorting/timsort_demo.py
    ├── c/          # e.g., src/c/ch12_hashing/linear_probing.c
    ├── go/         # e.g., src/go/ch21_flow/dinic.go
    ├── erlang/     # e.g., src/erlang/ch05_queues/circ_buffer.erl
    └── lean/       # e.g., src/lean/ch00_foundations/Inv.lean
```

Chapters live at the repo root as `chapter_XX.qmd` and are included in `_quarto.yml` in numeric order.
`src/` contains runnable, tested implementations that mirror the chapters by `chXX_*` folder names for easy cross-navigation.

## Labeling with Code

The book content is implemented across five languages:

- Python: clarity and rapid prototyping
- C: data layout and performance at low level
- Go: concurrency and modern systems coding
- Erlang: message passing and fault tolerance
- Lean: mechanized proofs and invariants

This multi-language approach makes algorithmic ideas transferable across paradigms.

## How to Read (L0 → L1 → L2)

Each section is written in three progressive layers:

- L0 (Beginner) — intuition and analogies
- L1 (Intermediate) — algorithms, proofs, trade-offs
- L2 (Advanced) — real systems, internals, and deeper theory

Skim L0 to build intuition, work L1 for competence, and dip into L2 for depth or production perspectives.

## Full Table of Contents — Five Volumes

Chapters are numbered 00–26 across the whole book. Each `chapter_XX.qmd` corresponds to one line below.

### Volume I — Structures Linéaires

- Chapter 0 Foundations

  * 0.1 Definition
  * 0.2 Correctness
  * 0.3 Efficiency
  * 0.4 Recursion
- Chapter 1 Numbers

  * 1.1 Representation
  * 1.2 Arithmetic
  * 1.3 Properties
  * 1.4 Growth & Precision
- Chapter 2 Arrays

  * 2.1 Access & Mutation
  * 2.2 Dynamic Arrays
  * 2.3 Multidimensional Arrays
  * 2.4 Prefix Sums & Tricks
- Chapter 3 Strings

  * 3.1 Encodings
  * 3.2 Operations
  * 3.3 Pattern Matching
  * 3.4 Suffix Structures
- Chapter 4 Linked Lists

  * 4.1 Singly Linked Lists
  * 4.2 Doubly Linked Lists
  * 4.3 Circular Lists
  * 4.4 Skip Lists
- Chapter 5 Stacks & Queues

  * 5.1 Stacks
  * 5.2 Queues
  * 5.3 Deques

### Volume II — Algorithmes Fondamentaux

- Chapter 6 Searching

  * 6.1 Linear Search
  * 6.2 Binary Search
  * 6.3 Hash Lookup
- Chapter 7 Selection

  * 7.1 Randomized Selection
  * 7.2 Median of Medians
- Chapter 8 Sorting

  * 8.1 Quadratic Sorts
  * 8.2 Divide-and-Conquer Sorts
  * 8.3 Linear-Time Sorts
  * 8.4 Hybrid Sorts
- Chapter 9 Amortized Analysis

  * 9.1 Aggregate Method
  * 9.2 Accounting Method
  * 9.3 Potential Method
  * 9.4 Union-Find Analysis

### Volume III — Structures Hiérarchiques

- Chapter 10 Tree Fundamentals

  * 10.1 Model & Terminology
  * 10.2 Representations
  * 10.3 Traversals
  * 10.4 Metrics & Invariants
- Chapter 11 Heaps & Priority Queues

  * 11.1 Heap Property
  * 11.2 Core Operations
  * 11.3 Variants & Meldability
- Chapter 12 Binary Search Trees

  * 12.1 Invariant & Operations
  * 12.2 Height & Degeneration
  * 12.3 Order Operations
- Chapter 13 Balanced Trees & Ordered Maps

  * 13.1 Rotations
  * 13.2 AVL Trees
  * 13.3 Red-Black Trees
  * 13.4 Treaps & Skiplists
- Chapter 14 Range Queries

  * 14.1 Fenwick Trees
  * 14.2 Segment Trees
  * 14.3 Sparse Tables
  * 14.4 Euler Tour + RMQ
- Chapter 15 Vector Databases

  * 15.1 Motivation & Embeddings
  * 15.2 Approximate Nearest Neighbors
  * 15.3 Graph-based Indexes (HNSW)
  * 15.4 Product Quantization & IVF
  * 15.5 Serving & Updating

### Volume IV — Paradigmes Algorithmiques

- Chapter 16 Divide-and-Conquer
- Chapter 17 Greedy
- Chapter 18 Dynamic Programming
- Chapter 19 Backtracking & Search

### Volume V — Graphes et Complexité

- Chapter 20 Graph Basics
- Chapter 21 DAGs & SCC
- Chapter 22 Shortest Paths
- Chapter 23 Flows & Matchings
- Chapter 24 Tree Algorithms
- Chapter 25 Complexity & Limits
- Chapter 26 External & Cache-Oblivious
- Chapter 27 Probabilistic & Streaming
- Chapter 28 Engineering

## Learning Matrix

The Little Book of Algorithms is designed as a layered curriculum: every concept is introduced at three levels — L0 (Beginner), L1 (Intermediate), L2 (Advanced). The matrix shows how topics progress across the volumes.

| Volume                         | Chapter                   | Section            | L0                     | L1                      | L2                          |
| ------------------------------ | ------------------------- | ------------------ | ---------------------- | ----------------------- | --------------------------- |
| I — Structures Linéaires       | Foundations               | Definition         | Recipes                | ADTs                    | Library/runtime             |
|                                |                           | Correctness        | Invariants             | Proof sketches          | Fuzzing, PBT                |
|                                |                           | Efficiency         | Growth rates           | Time-space tradeoffs    | Cache, GC                   |
|                                |                           | Recursion          | Factorial, Fibonacci   | Tracing, base cases     | Tail recursion              |
|                                | Numbers                   | Representation     | Decimal, binary        | Hex, two’s complement   | Floating-point internals    |
|                                |                           | Arithmetic         | Addition, subtraction  | Division, modulo        | Fast multiplication         |
|                                |                           | Properties         | Even/odd, divisibility | Modular arithmetic      | Number-theoretic transforms |
|                                |                           | Growth & Precision | Overflow               | Big integers            | IEEE 754                    |
|                                | Arrays                    | Access & Mutation  | Boxes in row           | Indexing, updates       | Kernel tables               |
|                                |                           | Dynamic Arrays     | Resizing               | Amortized analysis      | Memory allocators           |
|                                |                           | Multidimensional   | Grids                  | Row/col-major           | Cache blocking              |
|                                |                           | Prefix sums        | Running totals         | Sliding windows         | Fenwick baseline            |
|                                | Strings                   | Encodings          | ASCII/UTF-8            | Unicode handling        | Compression                 |
|                                |                           | Operations         | Concatenation          | Palindrome              | SIMD ops                    |
|                                |                           | Pattern Matching   | Substring search       | KMP, Rabin-Karp         | Suffix automaton            |
|                                |                           | Suffix Structures  | Suffix arrays          | LCP tables              | Suffix trees                |
|                                | Linked Lists              | Singly             | Next pointers          | Inserts, deletes        | Allocator pools             |
|                                |                           | Doubly             | Prev + next            | Bidirectional traversal | VFS dentries                |
|                                |                           | Circular           | Wrap around            | Ring buffer             | Concurrency use             |
|                                |                           | Skip Lists         | Probabilistic levels   | Expected O(log n)       | Concurrent skiplists        |
|                                | Stacks & Queues           | Stacks             | Plate analogy          | Array/list impls        | Call stack, VM frames       |
|                                |                           | Queues             | Line analogy           | Linked vs circular      | Lock-free queues            |
|                                |                           | Deques             | Double-ended           | Implementations         | OS schedulers               |
| II — Algorithmes Fondamentaux  | Searching                 | Linear             | Scan                   | Binary search           | Hash table internals        |
|                                |                           | Selection          | Randomized             | Quickselect             | Median-of-medians           |
|                                |                           | Sorting            | Quadratic sorts        | Merge, quicksort        | Timsort, introsort          |
|                                |                           | Amortized Analysis | Aggregate              | Accounting              | Potential method            |
| III — Structures Hiérarchiques | Tree Fundamentals         | Terminology        | Node, edge             | Rooted/unrooted         | Prüfer, Catalan             |
|                                |                           | Traversals         | BFS/DFS intuition      | Recursive impls         | Parallel traversals         |
|                                | Heaps                     | Heap property      | Parent-child rule      | Sift-up/down            | Cache-aware heaps           |
|                                |                           | Variants           | Min/max heaps          | d-ary heaps             | Fibonacci, pairing          |
|                                | Binary Search Trees       | Invariant          | Left < key < right     | Insert, delete          | Concurrency issues          |
|                                |                           | Height             | Degeneration           | Random inserts          | DSW rebalancing             |
|                                | Balanced Trees            | Rotations          | Tilt, straighten       | Implementing            | Proof of invariants         |
|                                |                           | AVL                | Balance factor         | Rebalance ops           | Perf tradeoffs              |
|                                |                           | Red-Black          | Coloring rules         | Insert/delete fixups    | Kernel rbtree               |
|                                |                           | Treaps/Skiplists   | Randomization          | Expected log n          | Redis zset                  |
|                                | Range Queries             | Fenwick            | Prefix sums            | i&-i trick              | 2D Fenwick                  |
|                                |                           | Segment Trees      | Interval splits        | Lazy propagation        | Parallel ST                 |
|                                |                           | Sparse Tables      | Precompute             | RMQ                     | Succinct encodings          |
|                                |                           | Euler Tour + RMQ   | Flatten trees          | LCA                     | Fischer-Heun                |
|                                | Vector Databases          | Motivation         | Embeddings             | ANN search              | PQ, IVF, HNSW               |
|                                |                           | Indexes            | LSH basics             | HNSW, graph search      | Hybrid systems              |
|                                |                           | Serving            | Batch build            | Updates                 | Distributed serving         |
| IV — Paradigmes Algorithmiques | Divide-and-Conquer        | Concept            | Split problems         | Master theorem          | Strassen, FFT               |
|                                | Greedy                    | Concept            | Local choice           | Exchange argument       | Matroid, submodularity      |
|                                | Dynamic Programming       | Concept            | Memoization            | Knapsack, LCS           | Convex hull trick           |
|                                | Backtracking              | Concept            | Try & undo             | Pruning                 | Branch & bound              |
| V — Graphes et Complexité      | Graph Basics              | Representation     | Cities & roads         | List vs matrix          | CSR, compression            |
|                                |                           | Traversals         | Ripples, maze          | BFS/DFS impls           | SCC detection               |
|                                | DAGs & SCC                | Toposort           | Task ordering          | DFS/BFS impls           | Build systems               |
|                                |                           | SCC                | —                      | Kosaraju, Tarjan        | Compiler opts               |
|                                | Shortest Paths            | Basics             | GPS analogy            | Dijkstra, Bellman-Ford  | A\*, Johnson                |
|                                | Flows & Matchings         | Max flow           | Pipes analogy          | Edmonds-Karp, Dinic     | Min-cut, matchings          |
|                                | Tree Algorithms           | Basics             | Height, diameter       | LCA                     | HLD                         |
|                                | Complexity & Limits       | Growth             | Big-O                  | Lower bounds            | SAT solvers                 |
|                                |                           | Randomized         | Shuffle                | Monte Carlo             | Reservoir sampling          |
|                                | External & Cache          | Basics             | RAM vs disk            | External sort           | Cache-oblivious             |
|                                | Probabilistic & Streaming | Basics             | —                      | Count-Min, HLL          | Telemetry                   |
|                                | Engineering               | Basics             | Trace inputs           | Property tests          | Profiling                   |

## Contributing

- Improve explanations, add exercises, or contribute code to `src/python`, `src/c`, `src/go`, `src/erlang`, `src/lean`.
- Keep code chapter-scoped (e.g., `src/go/ch16_dp/knapsack.go`).
- Prefer small, runnable examples with comments; include benchmarks where relevant.
- PRs welcome; please include tests or property checks where applicable.

## License

Creative Commons CC BY-NC-SA 4.0 for text and figures.
Code samples are MIT unless a file header states otherwise.

