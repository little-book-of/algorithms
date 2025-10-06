# The Plan 

## Chapter 1. Foundations of Algorithms

### 1. What Is an Algorithm?

| #  | Algorithm                  | Note                                               |
| -- | -------------------------- | -------------------------------------------------- |
| 1  | Euclid’s GCD               | Oldest known algorithm for greatest common divisor |
| 2  | Sieve of Eratosthenes      | Generate primes efficiently                        |
| 3  | Binary Search              | Divide and conquer search                          |
| 4  | Exponentiation by Squaring | Fast power computation                             |
| 5  | Long Division              | Classic step-by-step arithmetic                    |
| 6  | Modular Addition Algorithm | Wrap-around arithmetic                             |
| 7  | Base Conversion Algorithm  | Convert between number systems                     |
| 8  | Factorial Computation      | Recursive and iterative approaches                 |
| 9  | Fibonacci Sequence         | Recursive vs. dynamic computation                  |
| 10 | Tower of Hanoi             | Recursive problem-solving pattern                  |


### 2. Measuring Time and Space

| #  | Algorithm                  | Note                                |
| -- | -------------------------- | ----------------------------------- |
| 11 | Counting Operations        | Manual step-counting for complexity |
| 12 | Loop Analysis              | Evaluate time cost of loops         |
| 13 | Recurrence Expansion       | Analyze recursive costs             |
| 14 | Amortized Analysis         | Average per-operation cost          |
| 15 | Space Counting             | Stack and heap tracking             |
| 16 | Memory Footprint Estimator | Track per-variable usage            |
| 17 | Time Complexity Table      | Map O(1)...O(n²)...O(2ⁿ)            |
| 18 | Space-Time Tradeoff        | Cache vs. recomputation             |
| 19 | Profiling Algorithm        | Empirical time measurement          |
| 20 | Benchmarking Framework     | Compare algorithm performance       |


### 3. Big-O, Big-Theta, Big-Omega

| #  | Algorithm                      | Note                                  |
| -- | ------------------------------ | ------------------------------------- |
| 21 | Growth Rate Comparator         | Compare asymptotic behaviors          |
| 22 | Dominant Term Extractor        | Simplify runtime expressions          |
| 23 | Limit-Based Complexity Test    | Using limits for asymptotics          |
| 24 | Summation Simplifier           | Sum of arithmetic/geometric sequences |
| 25 | Recurrence Tree Method         | Visualize recursive costs             |
| 26 | Master Theorem Evaluator       | Solve T(n) recurrences                |
| 27 | Big-Theta Proof Builder        | Bounding upper and lower limits       |
| 28 | Big-Omega Case Finder          | Best-case scenario analysis           |
| 29 | Empirical Complexity Estimator | Measure via doubling experiments      |
| 30 | Complexity Class Identifier    | Match runtime to known class          |


### 4. Algorithmic Paradigms (Greedy, Divide and Conquer, DP)

| #  | Algorithm                   | Note                       |
| -- | --------------------------- | -------------------------- |
| 31 | Greedy Coin Change          | Local optimal step-by-step |
| 32 | Huffman Coding              | Greedy compression tree    |
| 33 | Merge Sort                  | Divide and conquer sort    |
| 34 | Binary Search               | Divide and conquer search  |
| 35 | Karatsuba Multiplication    | Recursive divide & conquer |
| 36 | Matrix Chain Multiplication | DP with substructure       |
| 37 | Longest Common Subsequence  | Classic DP problem         |
| 38 | Rod Cutting                 | DP optimization            |
| 39 | Activity Selection          | Greedy scheduling          |
| 40 | Optimal Merge Patterns      | Greedy file merging        |


### 5. Recurrence Relations

| #  | Algorithm                 | Note                               |
| -- | ------------------------- | ---------------------------------- |
| 41 | Linear Recurrence Solver  | Closed-form for linear recurrences |
| 42 | Master Theorem            | Divide-and-conquer complexity      |
| 43 | Substitution Method       | Inductive proof approach           |
| 44 | Iteration Method          | Expand recurrence step-by-step     |
| 45 | Generating Functions      | Transform recurrences              |
| 46 | Matrix Exponentiation     | Solve linear recurrences fast      |
| 47 | Recurrence to DP Table    | Tabulation approach                |
| 48 | Divide & Combine Template | Convert recurrence into algorithm  |
| 49 | Memoized Recursive Solver | Store overlapping results          |
| 50 | Characteristic Polynomial | Solve homogeneous recurrence       |


### 6. Searching Basics

| #  | Algorithm               | Note                           |
| -- | ----------------------- | ------------------------------ |
| 51 | Linear Search           | Sequential element scan        |
| 52 | Binary Search           | Midpoint halving               |
| 53 | Jump Search             | Block skip linear              |
| 54 | Exponential Search      | Doubling step size             |
| 55 | Interpolation Search    | Estimate position by value     |
| 56 | Ternary Search          | Divide into thirds             |
| 57 | Fibonacci Search        | Golden ratio search            |
| 58 | Sentinel Search         | Early termination optimization |
| 59 | Bidirectional Search    | Meet-in-the-middle             |
| 60 | Search in Rotated Array | Adapted binary search          |


### 7. Sorting Basics

| #  | Algorithm      | Note                     |
| -- | -------------- | ------------------------ |
| 61 | Bubble Sort    | Adjacent swap sort       |
| 62 | Selection Sort | Find minimum each pass   |
| 63 | Insertion Sort | Incremental build sort   |
| 64 | Shell Sort     | Gap-based insertion      |
| 65 | Merge Sort     | Divide-and-conquer       |
| 66 | Quick Sort     | Partition-based          |
| 67 | Heap Sort      | Binary heap order        |
| 68 | Counting Sort  | Integer key distribution |
| 69 | Radix Sort     | Digit-by-digit           |
| 70 | Bucket Sort    | Group into ranges        |


### 8. Data Structures Overview

| #  | Algorithm                  | Note                    |
| -- | -------------------------- | ----------------------- |
| 71 | Stack Push/Pop             | LIFO operations         |
| 72 | Queue Enqueue/Dequeue      | FIFO operations         |
| 73 | Singly Linked List         | Linear node chain       |
| 74 | Doubly Linked List         | Bidirectional traversal |
| 75 | Hash Table Insertion       | Key-value indexing      |
| 76 | Binary Search Tree Insert  | Ordered node placement  |
| 77 | Heapify                    | Build heap in-place     |
| 78 | Union-Find Operations      | Disjoint-set management |
| 79 | Graph Adjacency List Build | Sparse representation   |
| 80 | Trie Insertion/Search      | Prefix tree for strings |


### 9. Graphs and Trees Overview

| #  | Algorithm                    | Note                          |
| -- | ---------------------------- | ----------------------------- |
| 81 | DFS Traversal                | Depth-first exploration       |
| 82 | BFS Traversal                | Level-order exploration       |
| 83 | Topological Sort             | DAG ordering                  |
| 84 | Minimum Spanning Tree        | Kruskal/Prim overview         |
| 85 | Dijkstra’s Shortest Path     | Weighted graph shortest route |
| 86 | Bellman-Ford                 | Handle negative edges         |
| 87 | Floyd-Warshall               | All-pairs shortest path       |
| 88 | Union-Find for MST           | Edge grouping                 |
| 89 | Tree Traversals              | Inorder, Preorder, Postorder  |
| 90 | LCA (Lowest Common Ancestor) | Common node in tree           |


### 10. Algorithm Design Patterns

| #   | Algorithm               | Note                   |
| --- | ----------------------- | ---------------------- |
| 91  | Brute Force             | Try all possibilities  |
| 92  | Greedy Choice           | Local optimum per step |
| 93  | Divide and Conquer      | Break and merge        |
| 94  | Dynamic Programming     | Reuse subproblems      |
| 95  | Backtracking            | Explore with undo      |
| 96  | Branch and Bound        | Prune search space     |
| 97  | Randomized Algorithm    | Inject randomness      |
| 98  | Approximation Algorithm | Near-optimal solution  |
| 99  | Online Algorithm        | Step-by-step decision  |
| 100 | Hybrid Strategy         | Combine paradigms      |




## Chapter 2. Sorting and Searching

### 11. Elementary Sorting (Bubble, Insertion, Selection)

| #   | Algorithm             | Note                                  |
| --- | --------------------- | ------------------------------------- |
| 101 | Bubble Sort           | Swap adjacent out-of-order elements   |
| 102 | Improved Bubble Sort  | Stop early if already sorted          |
| 103 | Cocktail Shaker Sort  | Bidirectional bubble pass             |
| 104 | Selection Sort        | Select smallest element each pass     |
| 105 | Double Selection Sort | Find both min and max each pass       |
| 106 | Insertion Sort        | Insert each element into correct spot |
| 107 | Binary Insertion Sort | Use binary search for position        |
| 108 | Gnome Sort            | Simple insertion-like with swaps      |
| 109 | Odd-Even Sort         | Parallel-friendly comparison sort     |
| 110 | Stooge Sort           | Recursive quirky educational sort     |


### 12. Divide-and-Conquer Sorting (Merge, Quick, Heap)

| #   | Algorithm               | Note                             |
| --- | ----------------------- | -------------------------------- |
| 111 | Merge Sort              | Recursive divide and merge       |
| 112 | Iterative Merge Sort    | Bottom-up non-recursive version  |
| 113 | Quick Sort              | Partition-based recursive sort   |
| 114 | Hoare Partition Scheme  | Classic quicksort partition      |
| 115 | Lomuto Partition Scheme | Simpler but less efficient       |
| 116 | Randomized Quick Sort   | Avoid worst-case pivot           |
| 117 | Heap Sort               | Heapify + extract max repeatedly |
| 118 | 3-Way Quick Sort        | Handle duplicates efficiently    |
| 119 | External Merge Sort     | Disk-based merge for large data  |
| 120 | Parallel Merge Sort     | Divide work among threads        |


### 13. Counting and Distribution Sorts (Counting, Radix, Bucket)

| #   | Algorithm                | Note                                  |
| --- | ------------------------ | ------------------------------------- |
| 121 | Counting Sort            | Count key occurrences                 |
| 122 | Stable Counting Sort     | Preserve order of equals              |
| 123 | Radix Sort (LSD)         | Least significant digit first         |
| 124 | Radix Sort (MSD)         | Most significant digit first          |
| 125 | Bucket Sort              | Distribute into buckets               |
| 126 | Pigeonhole Sort          | Simple bucket variant                 |
| 127 | Flash Sort               | Distribution with in-place correction |
| 128 | Postman Sort             | Stable multi-key sort                 |
| 129 | Address Calculation Sort | Hash-like distribution                |
| 130 | Spread Sort              | Hybrid radix/quick strategy           |


### 14. Hybrid Sorts (IntroSort, Timsort)

| #   | Algorithm            | Note                             |
| --- | -------------------- | -------------------------------- |
| 131 | IntroSort            | Quick + Heap fallback            |
| 132 | TimSort              | Merge + Insertion + Runs         |
| 133 | Dual-Pivot QuickSort | Modern quicksort optimization    |
| 134 | SmoothSort           | Heap-like adaptive sort          |
| 135 | Block Merge Sort     | Cache-efficient merge variant    |
| 136 | Adaptive Merge Sort  | Adjusts to partially sorted data |
| 137 | PDQSort              | Pattern-defeating quicksort      |
| 138 | WikiSort             | Stable in-place merge            |
| 139 | GrailSort            | In-place stable mergesort        |
| 140 | Adaptive Hybrid Sort | Dynamically selects strategy     |


### 15. Special Sorts (Cycle, Gnome, Comb, Pancake)

| #   | Algorithm           | Note                          |
| --- | ------------------- | ----------------------------- |
| 141 | Cycle Sort          | Minimal writes                |
| 142 | Comb Sort           | Shrinking gap bubble          |
| 143 | Gnome Sort          | Insertion-like with swaps     |
| 144 | Cocktail Sort       | Two-way bubble                |
| 145 | Pancake Sort        | Flip-based sorting            |
| 146 | Bitonic Sort        | Parallel network sorting      |
| 147 | Odd-Even Merge Sort | Sorting network design        |
| 148 | Sleep Sort          | Uses timing as order key      |
| 149 | Bead Sort           | Simulates gravity             |
| 150 | Bogo Sort           | Randomly permute until sorted |


### 16. Linear and Binary Search

| #   | Algorithm                   | Note                         |
| --- | --------------------------- | ---------------------------- |
| 151 | Linear Search               | Scan sequentially            |
| 152 | Linear Search (Sentinel)    | Guard element at end         |
| 153 | Binary Search (Iterative)   | Halve interval each loop     |
| 154 | Binary Search (Recursive)   | Halve interval via recursion |
| 155 | Binary Search (Lower Bound) | First >= target              |
| 156 | Binary Search (Upper Bound) | First > target               |
| 157 | Exponential Search          | Double step size             |
| 158 | Jump Search                 | Jump fixed steps then linear |
| 159 | Fibonacci Search            | Golden-ratio style jumps     |
| 160 | Uniform Binary Search       | Avoid recomputing midpoints  |


### 17. Interpolation and Exponential Search

| #   | Algorithm                      | Note                         |
| --- | ------------------------------ | ---------------------------- |
| 161 | Interpolation Search           | Estimate index by value      |
| 162 | Recursive Interpolation Search | Divide by estimated midpoint |
| 163 | Exponential Search             | Double and binary refine     |
| 164 | Doubling Search                | Generic exponential pattern  |
| 165 | Galloping Search               | Used in TimSort merges       |
| 166 | Unbounded Binary Search        | Find bounds dynamically      |
| 167 | Root-Finding Bisection         | Search zero-crossing         |
| 168 | Golden Section Search          | Optimize unimodal function   |
| 169 | Fibonacci Search (Optimum)     | Similar to golden search     |
| 170 | Jump + Binary Hybrid           | Combined probing strategy    |


### 18. Selection Algorithms (Quickselect, Median of Medians)

| #   | Algorithm                 | Note                      |
| --- | ------------------------- | ------------------------- |
| 171 | Quickselect               | Partition-based selection |
| 172 | Median of Medians         | Deterministic pivot       |
| 173 | Randomized Select         | Random pivot version      |
| 174 | Binary Search on Answer   | Range-based selection     |
| 175 | Order Statistics Tree     | BST with rank queries     |
| 176 | Tournament Tree Selection | Hierarchical comparison   |
| 177 | Heap Select (Min-Heap)    | Maintain top-k elements   |
| 178 | Partial QuickSort         | Sort partial prefix       |
| 179 | BFPRT Algorithm           | Linear-time selection     |
| 180 | Kth Largest Stream        | Streaming selection       |


### 19. Range Searching and Nearest Neighbor

| #   | Algorithm                  | Note                        |
| --- | -------------------------- | --------------------------- |
| 181 | Binary Search Range        | Find lower and upper bounds |
| 182 | Segment Tree Query         | Sum/min/max over interval   |
| 183 | Fenwick Tree Query         | Efficient prefix sums       |
| 184 | Interval Tree Search       | Overlap queries             |
| 185 | KD-Tree Search             | Spatial nearest neighbor    |
| 186 | R-Tree Query               | Range search in geometry    |
| 187 | Range Minimum Query (RMQ)  | Sparse table approach       |
| 188 | Mo’s Algorithm             | Offline query reordering    |
| 189 | Sweep Line Range Search    | Sort + scan technique       |
| 190 | Ball Tree Nearest Neighbor | Metric-space search         |


### 20. Search Optimizations and Variants

| #   | Algorithm                      | Note                     |
| --- | ------------------------------ | ------------------------ |
| 191 | Binary Search with Tolerance   | For floating values      |
| 192 | Ternary Search                 | Unimodal optimization    |
| 193 | Hash-Based Search              | O(1) expected lookup     |
| 194 | Bloom Filter Lookup            | Probabilistic membership |
| 195 | Cuckoo Hash Search             | Dual-hash relocation     |
| 196 | Robin Hood Hashing             | Equalize probe lengths   |
| 197 | Jump Consistent Hashing        | Stable hash assignment   |
| 198 | Prefix Search in Trie          | Auto-completion lookup   |
| 199 | Pattern Search in Suffix Array | Fast substring lookup    |
| 200 | Search in Infinite Array       | Dynamic bound finding    |


## Chapter 3. Data Structures in Action

### 21. Arrays, Linked Lists, Stacks, Queues

| #   | Algorithm                        | Note                              |
| --- | -------------------------------- | --------------------------------- |
| 201 | Dynamic Array Resize             | Doubling strategy for capacity    |
| 202 | Circular Array Implementation    | Wrap-around indexing              |
| 203 | Singly Linked List Insert/Delete | Basic node manipulation           |
| 204 | Doubly Linked List Insert/Delete | Two-way linkage                   |
| 205 | Stack Push/Pop                   | LIFO structure                    |
| 206 | Queue Enqueue/Dequeue            | FIFO structure                    |
| 207 | Deque Implementation             | Double-ended queue                |
| 208 | Circular Queue                   | Fixed-size queue with wrap-around |
| 209 | Stack via Queue                  | Implement stack using two queues  |
| 210 | Queue via Stack                  | Implement queue using two stacks  |


### 22. Hash Tables and Variants (Cuckoo, Robin Hood, Consistent)

| #   | Algorithm            | Note                             |
| --- | -------------------- | -------------------------------- |
| 211 | Hash Table Insertion | Key-value pair with modulo       |
| 212 | Linear Probing       | Resolve collisions sequentially  |
| 213 | Quadratic Probing    | Nonlinear probing sequence       |
| 214 | Double Hashing       | Alternate hash on collision      |
| 215 | Cuckoo Hashing       | Two-table relocation strategy    |
| 216 | Robin Hood Hashing   | Equalize probe length fairness   |
| 217 | Chained Hash Table   | Linked list buckets              |
| 218 | Perfect Hashing      | No-collision mapping             |
| 219 | Consistent Hashing   | Stable distribution across nodes |
| 220 | Dynamic Rehashing    | Resize on load factor threshold  |


### 23. Heaps (Binary, Fibonacci, Pairing)

| #   | Algorithm                    | Note                           |
| --- | ---------------------------- | ------------------------------ |
| 221 | Binary Heap Insert           | Bubble-up maintenance          |
| 222 | Binary Heap Delete           | Heapify-down maintenance       |
| 223 | Build Heap (Heapify)         | Bottom-up O(n) build           |
| 224 | Heap Sort                    | Extract max repeatedly         |
| 225 | Min Heap Implementation      | For smallest element access    |
| 226 | Max Heap Implementation      | For largest element access     |
| 227 | Fibonacci Heap Insert/Delete | Amortized efficient operations |
| 228 | Pairing Heap Merge           | Lightweight mergeable heap     |
| 229 | Binomial Heap Merge          | Merge trees of equal order     |
| 230 | Leftist Heap Merge           | Maintain rank-skewed heap      |


### 24. Balanced Trees (AVL, Red-Black, Splay, Treap)

| #   | Algorithm              | Note                         |
| --- | ---------------------- | ---------------------------- |
| 231 | AVL Tree Insert        | Rotate to maintain balance   |
| 232 | AVL Tree Delete        | Balance after deletion       |
| 233 | Red-Black Tree Insert  | Color fix and rotations      |
| 234 | Red-Black Tree Delete  | Maintain invariants          |
| 235 | Splay Tree Access      | Move accessed node to root   |
| 236 | Treap Insert           | Priority-based rotation      |
| 237 | Treap Delete           | Randomized balance           |
| 238 | Weight Balanced Tree   | Maintain subtree weights     |
| 239 | Scapegoat Tree Rebuild | Rebalance on size threshold  |
| 240 | AA Tree                | Simplified red-black variant |


### 25. Segment Trees and Fenwick Trees

| #   | Algorithm               | Note                         |
| --- | ----------------------- | ---------------------------- |
| 241 | Build Segment Tree      | Recursive construction       |
| 242 | Range Sum Query         | Recursive or iterative query |
| 243 | Range Update            | Lazy propagation technique   |
| 244 | Point Update            | Modify single element        |
| 245 | Fenwick Tree Build      | Incremental binary index     |
| 246 | Fenwick Update          | Update cumulative sums       |
| 247 | Fenwick Query           | Prefix sum retrieval         |
| 248 | Segment Tree Merge      | Combine child results        |
| 249 | Persistent Segment Tree | Maintain history of versions |
| 250 | 2D Segment Tree         | For matrix range queries     |


### 26. Disjoint Set Union (Union-Find)

| #   | Algorithm            | Note                          |
| --- | -------------------- | ----------------------------- |
| 251 | Make-Set             | Initialize each element       |
| 252 | Find                 | Locate representative         |
| 253 | Union                | Merge two sets                |
| 254 | Union by Rank        | Attach smaller tree to larger |
| 255 | Path Compression     | Flatten tree structure        |
| 256 | DSU with Rollback    | Support undo operations       |
| 257 | DSU on Tree          | Track subtree connectivity    |
| 258 | Kruskal’s MST        | Edge selection with DSU       |
| 259 | Connected Components | Group graph nodes             |
| 260 | Offline Query DSU    | Handle dynamic unions         |


### 27. Probabilistic Data Structures (Bloom, Count-Min, HyperLogLog)

| #   | Algorithm             | Note                           |
| --- | --------------------- | ------------------------------ |
| 261 | Bloom Filter Insert   | Hash to bit array              |
| 262 | Bloom Filter Query    | Probabilistic membership check |
| 263 | Counting Bloom Filter | Support deletions via counters |
| 264 | Cuckoo Filter         | Space-efficient alternative    |
| 265 | Count-Min Sketch      | Approximate frequency table    |
| 266 | HyperLogLog           | Cardinality estimation         |
| 267 | Flajolet-Martin       | Early probabilistic counting   |
| 268 | MinHash               | Estimate Jaccard similarity    |
| 269 | Reservoir Sampling    | Random k-sample stream         |
| 270 | Skip Bloom Filter     | Range queries on Bloom         |


### 28. Skip Lists and B-Trees

| #   | Algorithm           | Note                          |
| --- | ------------------- | ----------------------------- |
| 271 | Skip List Insert    | Probabilistic layered list    |
| 272 | Skip List Delete    | Adjust pointers               |
| 273 | Skip List Search    | Jump via tower levels         |
| 274 | B-Tree Insert       | Split on overflow             |
| 275 | B-Tree Delete       | Merge on underflow            |
| 276 | B+ Tree Search      | Leaf-based sequential scan    |
| 277 | B+ Tree Range Query | Efficient ordered access      |
| 278 | B* Tree             | More space-efficient variant  |
| 279 | Adaptive Radix Tree | Byte-wise branching           |
| 280 | Trie Compression    | Path compression optimization |


### 29. Persistent and Functional Data Structures

| #   | Algorithm                 | Note                     |
| --- | ------------------------- | ------------------------ |
| 281 | Persistent Stack          | Keep all versions        |
| 282 | Persistent Array          | Copy-on-write segments   |
| 283 | Persistent Segment Tree   | Versioned updates        |
| 284 | Persistent Linked List    | Immutable nodes          |
| 285 | Functional Queue          | Amortized reverse lists  |
| 286 | Finger Tree               | Fast concat and split    |
| 287 | Zipper Structure          | Localized mutation       |
| 288 | Red-Black Persistent Tree | Immutable balanced tree  |
| 289 | Trie with Versioning      | Historical string lookup |
| 290 | Persistent Union-Find     | Time-travel connectivity |


### 30. Advanced Trees and Range Queries

| #   | Algorithm              | Note                      |
| --- | ---------------------- | ------------------------- |
| 291 | Sparse Table Build     | Static range min/max      |
| 292 | Cartesian Tree         | RMQ to LCA transformation |
| 293 | Segment Tree Beats     | Handle complex queries    |
| 294 | Merge Sort Tree        | Range count queries       |
| 295 | Wavelet Tree           | Rank/select by value      |
| 296 | KD-Tree                | Multidimensional queries  |
| 297 | Range Tree             | Orthogonal range queries  |
| 298 | Fenwick 2D Tree        | Matrix prefix sums        |
| 299 | Treap Split/Merge      | Range-based treap ops     |
| 300 | Mo’s Algorithm on Tree | Offline subtree queries   |


## Chapter 4. Graph Algorithms

### 31. Traversals (DFS, BFS, Iterative Deepening)

| #   | Algorithm                           | Note                                |
| --- | ----------------------------------- | ----------------------------------- |
| 301 | Depth-First Search (Recursive)      | Explore deeply before backtracking  |
| 302 | Depth-First Search (Iterative)      | Stack-based exploration             |
| 303 | Breadth-First Search (Queue)        | Level-order traversal               |
| 304 | Iterative Deepening DFS             | Combine depth-limit + completeness  |
| 305 | Bidirectional BFS                   | Search from both ends               |
| 306 | DFS on Grid                         | Maze solving / connected components |
| 307 | BFS on Grid                         | Shortest path in unweighted graph   |
| 308 | Multi-Source BFS                    | Parallel layer expansion            |
| 309 | Topological Sort (DFS-based)        | DAG ordering                        |
| 310 | Topological Sort (Kahn’s Algorithm) | In-degree tracking                  |


### 32. Strongly Connected Components (Tarjan, Kosaraju)

| #   | Algorithm                   | Note                      |
| --- | --------------------------- | ------------------------- |
| 311 | Kosaraju’s Algorithm        | Two-pass DFS              |
| 312 | Tarjan’s Algorithm          | Low-link discovery        |
| 313 | Gabow’s Algorithm           | Stack pair tracking       |
| 314 | SCC DAG Construction        | Condensed component graph |
| 315 | SCC Online Merge            | Incremental condensation  |
| 316 | Component Label Propagation | Iterative labeling        |
| 317 | Path-Based SCC              | DFS with path stack       |
| 318 | Kosaraju Parallel Version   | SCC via parallel DFS      |
| 319 | Dynamic SCC Maintenance     | Add/remove edges          |
| 320 | SCC for Weighted Graph      | Combine with edge weights |


### 33. Shortest Paths (Dijkstra, Bellman-Ford, A*, Johnson)

| #   | Algorithm                 | Note                        |
| --- | ------------------------- | --------------------------- |
| 321 | Dijkstra (Binary Heap)    | Greedy edge relaxation      |
| 322 | Dijkstra (Fibonacci Heap) | Improved priority queue     |
| 323 | Bellman-Ford              | Negative weights support    |
| 324 | SPFA (Queue Optimization) | Faster average Bellman-Ford |
| 325 | A* Search                 | Heuristic-guided path       |
| 326 | Floyd–Warshall            | All-pairs shortest path     |
| 327 | Johnson’s Algorithm       | All-pairs using reweighting |
| 328 | 0-1 BFS                   | Deque-based shortest path   |
| 329 | Dial’s Algorithm          | Integer weight buckets      |
| 330 | Multi-Source Dijkstra     | Multiple starting points    |


### 34. Shortest Path Variants (0–1 BFS, Bidirectional, Heuristic A*)

| #   | Algorithm                   | Note                               |
| --- | --------------------------- | ---------------------------------- |
| 331 | 0–1 BFS                     | For edges with weight 0 or 1       |
| 332 | Bidirectional Dijkstra      | Meet in the middle                 |
| 333 | A* with Euclidean Heuristic | Spatial shortest path              |
| 334 | ALT Algorithm               | A* landmarks + triangle inequality |
| 335 | Contraction Hierarchies     | Preprocessing for road networks    |
| 336 | CH Query Algorithm          | Shortcut-based routing             |
| 337 | Bellman-Ford Queue Variant  | Early termination                  |
| 338 | Dijkstra with Early Stop    | Halt on target                     |
| 339 | Goal-Directed Search        | Restrict expansion direction       |
| 340 | Yen’s K Shortest Paths      | Enumerate multiple best paths      |


### 35. Minimum Spanning Trees (Kruskal, Prim, Borůvka)

| #   | Algorithm                          | Note                    |
| --- | ---------------------------------- | ----------------------- |
| 341 | Kruskal’s Algorithm                | Sort edges + union-find |
| 342 | Prim’s Algorithm (Heap)            | Grow MST from seed      |
| 343 | Prim’s Algorithm (Adj Matrix)      | Dense graph variant     |
| 344 | Borůvka’s Algorithm                | Component merging       |
| 345 | Reverse-Delete MST                 | Remove heavy edges      |
| 346 | MST via Dijkstra Trick             | For positive weights    |
| 347 | Dynamic MST Maintenance            | Handle edge updates     |
| 348 | Minimum Bottleneck Spanning Tree   | Max edge minimization   |
| 349 | Manhattan MST                      | Grid graph optimization |
| 350 | Euclidean MST (Kruskal + Geometry) | Use Delaunay graph      |


### 36. Flows (Ford–Fulkerson, Edmonds–Karp, Dinic)

| #   | Algorithm                        | Note                         |
| --- | -------------------------------- | ---------------------------- |
| 351 | Ford–Fulkerson                   | Augmenting path method       |
| 352 | Edmonds–Karp                     | BFS-based Ford–Fulkerson     |
| 353 | Dinic’s Algorithm                | Level graph + blocking flow  |
| 354 | Push–Relabel                     | Local preflow push           |
| 355 | Capacity Scaling                 | Speed-up with capacity tiers |
| 356 | Cost Scaling                     | Min-cost optimization        |
| 357 | Min-Cost Max-Flow (Bellman-Ford) | Costed augmenting paths      |
| 358 | Min-Cost Max-Flow (SPFA)         | Faster average               |
| 359 | Circulation with Demands         | Generalized flow formulation |
| 360 | Successive Shortest Path         | Incremental min-cost updates |


### 37. Cuts (Stoer–Wagner, Karger, Gomory–Hu)

| #   | Algorithm                      | Note                         |
| --- | ------------------------------ | ---------------------------- |
| 361 | Stoer–Wagner Minimum Cut       | Global min cut               |
| 362 | Karger’s Randomized Cut        | Contract edges randomly      |
| 363 | Karger–Stein                   | Recursive randomized cut     |
| 364 | Gomory–Hu Tree                 | All-pairs min-cut            |
| 365 | Max-Flow Min-Cut               | Duality theorem application  |
| 366 | Stoer–Wagner Repeated Phase    | Multiple passes              |
| 367 | Dynamic Min Cut                | Maintain on edge update      |
| 368 | Minimum s–t Cut (Edmonds–Karp) | Based on flow                |
| 369 | Approximate Min Cut            | Random sampling              |
| 370 | Min k-Cut                      | Partition graph into k parts |


### 38. Matchings (Hopcroft–Karp, Hungarian, Blossom)

| #   | Algorithm                      | Note                      |
| --- | ------------------------------ | ------------------------- |
| 371 | Bipartite Matching (DFS)       | Simple augmenting path    |
| 372 | Hopcroft–Karp                  | O(E√V) bipartite matching |
| 373 | Hungarian Algorithm            | Weighted assignment       |
| 374 | Kuhn–Munkres                   | Max-weight matching       |
| 375 | Blossom Algorithm              | General graph matching    |
| 376 | Edmonds’ Blossom Shrinking     | Odd cycle contraction     |
| 377 | Greedy Matching                | Fast approximate          |
| 378 | Stable Marriage (Gale–Shapley) | Stable pairing            |
| 379 | Weighted b-Matching            | Capacity-constrained      |
| 380 | Maximal Matching               | Local greedy maximal set  |


### 39. Tree Algorithms (LCA, HLD, Centroid Decomposition)

| #   | Algorithm                  | Note                        |
| --- | -------------------------- | --------------------------- |
| 381 | Euler Tour LCA             | Flatten tree to array       |
| 382 | Binary Lifting LCA         | Jump powers of two          |
| 383 | Tarjan’s LCA (Offline DSU) | Query via union-find        |
| 384 | Heavy-Light Decomposition  | Decompose paths             |
| 385 | Centroid Decomposition     | Recursive split on centroid |
| 386 | Tree Diameter (DFS Twice)  | Farthest pair               |
| 387 | Tree DP                    | Subtree-based optimization  |
| 388 | Rerooting DP               | Compute all roots’ answers  |
| 389 | Binary Search on Tree      | Edge weight constraints     |
| 390 | Virtual Tree               | Build on query subset       |


### 40. Advanced Graph Algorithms and Tricks

| #   | Algorithm                           | Note                          |
| --- | ----------------------------------- | ----------------------------- |
| 391 | Topological DP                      | DP on DAG order               |
| 392 | SCC Condensed Graph DP              | Meta-graph processing         |
| 393 | Eulerian Path                       | Trail covering all edges      |
| 394 | Hamiltonian Path                    | NP-complete exploration       |
| 395 | Chinese Postman                     | Eulerian circuit with repeats |
| 396 | Hierholzer’s Algorithm              | Construct Eulerian circuit    |
| 397 | Johnson’s Cycle Finding             | Enumerate all cycles          |
| 398 | Transitive Closure (Floyd–Warshall) | Reachability matrix           |
| 399 | Graph Coloring (Backtracking)       | Constraint satisfaction       |
| 400 | Articulation Points & Bridges       | Critical structure detection  |


## Chapter 5. Dynamic Programming

### 41. DP Basics and State Transitions

| #   | Algorithm                           | Note                            |
| --- | ----------------------------------- | ------------------------------- |
| 401 | Fibonacci DP                        | Classic top-down vs bottom-up   |
| 402 | Climbing Stairs                     | Count paths with small steps    |
| 403 | Grid Paths                          | DP over 2D lattice              |
| 404 | Min Cost Path                       | Accumulate minimal sums         |
| 405 | Coin Change (Count Ways)            | Combinatorial sums              |
| 406 | Coin Change (Min Coins)             | Minimize step count             |
| 407 | Knapsack 0/1                        | Select items under weight limit |
| 408 | Knapsack Unbounded                  | Repeatable items                |
| 409 | Longest Increasing Subsequence (DP) | Subsequence optimization        |
| 410 | Edit Distance (Levenshtein)         | Measure similarity step-by-step |


### 42. Classic Problems (Knapsack, Subset Sum, Coin Change)

| #   | Algorithm                  | Note                              |
| --- | -------------------------- | --------------------------------- |
| 411 | 0/1 Knapsack               | Value maximization under capacity |
| 412 | Subset Sum                 | Boolean feasibility DP            |
| 413 | Equal Partition            | Divide set into equal halves      |
| 414 | Count of Subsets with Sum  | Counting variant                  |
| 415 | Target Sum                 | DP with +/- transitions           |
| 416 | Unbounded Knapsack         | Reuse items                       |
| 417 | Fractional Knapsack        | Greedy + DP comparison            |
| 418 | Coin Change (Min Coins)    | DP shortest path                  |
| 419 | Coin Change (Count Ways)   | Combinatorial counting            |
| 420 | Multi-Dimensional Knapsack | Capacity in multiple dimensions   |


### 43. Sequence Problems (LIS, LCS, Edit Distance)

| #   | Algorithm                           | Note                          |
| --- | ----------------------------------- | ----------------------------- |
| 421 | Longest Increasing Subsequence      | O(n²) DP                      |
| 422 | LIS (Patience Sorting)              | O(n log n) optimized          |
| 423 | Longest Common Subsequence          | Two-sequence DP               |
| 424 | Edit Distance (Levenshtein)         | Transform operations          |
| 425 | Longest Palindromic Subsequence     | Symmetric DP                  |
| 426 | Shortest Common Supersequence       | Merge sequences               |
| 427 | Longest Repeated Subsequence        | DP with overlap               |
| 428 | String Interleaving                 | Merge with order preservation |
| 429 | Sequence Alignment (Bioinformatics) | Gap penalties                 |
| 430 | Diff Algorithm (Myers/DP)           | Minimal edit path             |


### 44. Matrix and Chain Problems

| #   | Algorithm                      | Note                  |
| --- | ------------------------------ | --------------------- |
| 431 | Matrix Chain Multiplication    | Parenthesization cost |
| 432 | Boolean Parenthesization       | Count true outcomes   |
| 433 | Burst Balloons                 | Interval DP           |
| 434 | Optimal BST                    | Weighted search cost  |
| 435 | Polygon Triangulation          | DP over partitions    |
| 436 | Matrix Path Sum                | DP on 2D grid         |
| 437 | Largest Square Submatrix       | Dynamic growth check  |
| 438 | Max Rectangle in Binary Matrix | Histogram + DP        |
| 439 | Submatrix Sum Queries          | Prefix sum DP         |
| 440 | Palindrome Partitioning        | DP with cuts          |


### 45. Bitmask DP and Traveling Salesman

| #   | Algorithm                 | Note                            |
| --- | ------------------------- | ------------------------------- |
| 441 | Traveling Salesman (TSP)  | Visit all cities                |
| 442 | Subset DP                 | Over subsets of states          |
| 443 | Hamiltonian Path DP       | State compression               |
| 444 | Assignment Problem DP     | Mask over tasks                 |
| 445 | Partition into Two Sets   | Balanced load                   |
| 446 | Count Hamiltonian Cycles  | Bitmask enumeration             |
| 447 | Steiner Tree DP           | Minimal connection of terminals |
| 448 | SOS DP (Sum Over Subsets) | Precompute sums                 |
| 449 | Bitmask Knapsack          | State compression               |
| 450 | Bitmask Independent Set   | Graph subset optimization       |


### 46. Digit DP and SOS DP

| #   | Algorithm                         | Note                     |
| --- | --------------------------------- | ------------------------ |
| 451 | Count Numbers with Property       | Digit-state transitions  |
| 452 | Count Without Adjacent Duplicates | Adjacent constraints     |
| 453 | Sum of Digits in Range            | Carry-dependent states   |
| 454 | Count with Mod Condition          | DP over digit sum mod M  |
| 455 | Count of Increasing Digits        | Ordered constraint       |
| 456 | Count with Forbidden Digits       | Exclusion transitions    |
| 457 | SOS DP Subset Sum                 | Sum over bitmask subsets |
| 458 | SOS DP Superset Sum               | Sum over supersets       |
| 459 | XOR Basis DP                      | Combine digit and bit DP |
| 460 | Digit DP for Palindromes          | Symmetric digit state    |


### 47. DP Optimizations (Divide & Conquer, Convex Hull Trick, Knuth)

| #   | Algorithm                    | Note                           |
| --- | ---------------------------- | ------------------------------ |
| 461 | Divide & Conquer DP          | Monotone decision property     |
| 462 | Knuth Optimization           | DP with quadrangle inequality  |
| 463 | Convex Hull Trick            | Linear recurrence min queries  |
| 464 | Li Chao Tree                 | Segment-based hull maintenance |
| 465 | Slope Trick                  | Piecewise-linear optimization  |
| 466 | Monotonic Queue Optimization | Sliding DP state               |
| 467 | Bitset DP                    | Speed via bit-parallel         |
| 468 | Offline DP Queries           | Preprocessing state            |
| 469 | DP + Segment Tree            | Range-based optimization       |
| 470 | Divide & Conquer Knapsack    | Split-space DP                 |


### 48. Tree DP and Rerooting

| #   | Algorithm                | Note                      |
| --- | ------------------------ | ------------------------- |
| 471 | Subtree Sum DP           | Aggregate values          |
| 472 | Diameter DP              | Max path via child states |
| 473 | Independent Set DP       | Choose or skip nodes      |
| 474 | Vertex Cover DP          | Tree constraint problem   |
| 475 | Path Counting DP         | Count root-leaf paths     |
| 476 | DP on Rooted Tree        | Bottom-up aggregation     |
| 477 | Rerooting Technique      | Compute for all roots     |
| 478 | Distance Sum Rerooting   | Efficient recomputation   |
| 479 | Tree Coloring DP         | Combinatorial counting    |
| 480 | Binary Search on Tree DP | Monotonic transitions     |


### 49. DP Reconstruction and Traceback

| #   | Algorithm                   | Note                           |
| --- | --------------------------- | ------------------------------ |
| 481 | Reconstruct LCS             | Backtrack table                |
| 482 | Reconstruct LIS             | Track predecessors             |
| 483 | Reconstruct Knapsack        | Recover selected items         |
| 484 | Edit Distance Alignment     | Trace insert/delete/substitute |
| 485 | Matrix Chain Parentheses    | Rebuild parenthesization       |
| 486 | Coin Change Reconstruction  | Backtrack last used coin       |
| 487 | Path Reconstruction DP      | Trace minimal route            |
| 488 | Sequence Reconstruction     | Rebuild from states            |
| 489 | Multi-Choice Reconstruction | Combine best subpaths          |
| 490 | Traceback Visualization     | Visual DP backtrack tool       |


### 50. Meta-DP and Optimization Templates

| #   | Algorithm                        | Note                        |
| --- | -------------------------------- | --------------------------- |
| 491 | State Compression Template       | Represent subsets compactly |
| 492 | Transition Optimization Template | Precompute transitions      |
| 493 | Space Optimization Template      | Rolling arrays              |
| 494 | Multi-Dimensional DP Template    | Nested loops version        |
| 495 | Decision Monotonicity            | Optimization hint           |
| 496 | Monge Array Optimization         | Matrix property leverage    |
| 497 | Divide & Conquer Template        | Half-split recursion        |
| 498 | Rerooting Template               | Generalized tree DP         |
| 499 | Iterative DP Pattern             | Bottom-up unrolling         |
| 500 | Memoization Template             | Recursive caching skeleton  |


## Chapter 6. Mathematics for Algorithms

### 51. Number Theory (GCD, Modular Arithmetic, CRT)

| #   | Algorithm                      | Note                      |
| --- | ------------------------------ | ------------------------- |
| 501 | Euclidean Algorithm            | Compute gcd(a, b)         |
| 502 | Extended Euclidean Algorithm   | Solve ax + by = gcd(a, b) |
| 503 | Modular Addition               | Add under modulo M        |
| 504 | Modular Multiplication         | Multiply under modulo M   |
| 505 | Modular Exponentiation         | Fast power mod M          |
| 506 | Modular Inverse                | Compute a⁻¹ mod M         |
| 507 | Chinese Remainder Theorem      | Combine modular systems   |
| 508 | Binary GCD (Stein’s Algorithm) | Bitwise gcd               |
| 509 | Modular Reduction              | Normalize residues        |
| 510 | Modular Linear Equation Solver | Solve ax ≡ b (mod m)      |


### 52. Primality and Factorization (Miller–Rabin, Pollard Rho)

| #   | Algorithm                   | Note                          |
| --- | --------------------------- | ----------------------------- |
| 511 | Trial Division              | Simple prime test             |
| 512 | Sieve of Eratosthenes       | Generate primes up to n       |
| 513 | Sieve of Atkin              | Faster sieve variant          |
| 514 | Miller–Rabin Primality Test | Probabilistic primality       |
| 515 | Fermat Primality Test       | Modular power check           |
| 516 | Pollard’s Rho               | Randomized factorization      |
| 517 | Pollard’s p−1 Method        | Factor using smoothness       |
| 518 | Wheel Factorization         | Skip known composites         |
| 519 | AKS Primality Test          | Deterministic polynomial test |
| 520 | Segmented Sieve             | Prime generation for large n  |


### 53. Combinatorics (Permutations, Combinations, Subsets)

| #   | Algorithm                | Note                       |
| --- | ------------------------ | -------------------------- |
| 521 | Factorial Precomputation | Build n! table             |
| 522 | nCr Computation          | Use Pascal’s or factorials |
| 523 | Pascal’s Triangle        | Binomial coefficients      |
| 524 | Multiset Combination     | Repetition allowed         |
| 525 | Permutation Generation   | Lexicographic order        |
| 526 | Next Permutation         | STL-style increment        |
| 527 | Subset Generation        | Bitmask or recursion       |
| 528 | Gray Code Generation     | Single-bit flips           |
| 529 | Catalan Number DP        | Count valid parentheses    |
| 530 | Stirling Numbers         | Partition counting         |


### 54. Probability and Randomized Algorithms

| #   | Algorithm                   | Note                          |
| --- | --------------------------- | ----------------------------- |
| 531 | Monte Carlo Simulation      | Approximate via randomness    |
| 532 | Las Vegas Algorithm         | Always correct, variable time |
| 533 | Reservoir Sampling          | Uniform sampling from stream  |
| 534 | Randomized QuickSort        | Expected O(n log n)           |
| 535 | Randomized QuickSelect      | Random pivot                  |
| 536 | Birthday Paradox Simulation | Probability collision         |
| 537 | Random Hashing              | Reduce collision chance       |
| 538 | Random Walk Simulation      | State transitions             |
| 539 | Coupon Collector Estimation | Expected trials               |
| 540 | Markov Chain Simulation     | Transition matrix sampling    |


### 55. Sieve Methods and Modular Math

| #   | Algorithm                         | Note                           |
| --- | --------------------------------- | ------------------------------ |
| 541 | Sieve of Eratosthenes             | Base prime sieve               |
| 542 | Linear Sieve                      | O(n) sieve variant             |
| 543 | Segmented Sieve                   | Range prime generation         |
| 544 | SPF (Smallest Prime Factor) Table | Factorization via sieve        |
| 545 | Möbius Function Sieve             | Multiplicative function calc   |
| 546 | Euler’s Totient Sieve             | Compute φ(n) for all n         |
| 547 | Divisor Count Sieve               | Count divisors efficiently     |
| 548 | Modular Precomputation            | Store inverses, factorials     |
| 549 | Fermat Little Theorem             | a^(p−1) ≡ 1 mod p              |
| 550 | Wilson’s Theorem                  | Prime test via factorial mod p |


### 56. Linear Algebra (Gaussian Elimination, LU, SVD)

| #   | Algorithm                       | Note                            |
| --- | ------------------------------- | ------------------------------- |
| 551 | Gaussian Elimination            | Solve Ax = b                    |
| 552 | Gauss-Jordan Elimination        | Reduced row echelon             |
| 553 | LU Decomposition                | Factor A into L·U               |
| 554 | Cholesky Decomposition          | A = L·Lᵀ for SPD                |
| 555 | QR Decomposition                | Orthogonal factorization        |
| 556 | Matrix Inversion (Gauss-Jordan) | Find A⁻¹                        |
| 557 | Determinant by Elimination      | Product of pivots               |
| 558 | Rank of Matrix                  | Count non-zero rows             |
| 559 | Eigenvalue Power Method         | Approximate dominant eigenvalue |
| 560 | Singular Value Decomposition    | A = UΣVᵀ                        |


### 57. FFT and NTT (Fast Transforms)

| #   | Algorithm                        | Note                         |
| --- | -------------------------------- | ---------------------------- |
| 561 | Discrete Fourier Transform (DFT) | O(n²) baseline               |
| 562 | Fast Fourier Transform (FFT)     | O(n log n) convolution       |
| 563 | Cooley–Tukey FFT                 | Recursive divide and conquer |
| 564 | Iterative FFT                    | In-place bit reversal        |
| 565 | Inverse FFT                      | Recover time-domain          |
| 566 | Convolution via FFT              | Polynomial multiplication    |
| 567 | Number Theoretic Transform (NTT) | Modulo prime FFT             |
| 568 | Inverse NTT                      | Modular inverse transform    |
| 569 | Bluestein’s Algorithm            | FFT of arbitrary size        |
| 570 | FFT-Based Multiplication         | Big integer product          |


### 58. Numerical Methods (Newton, Simpson, Runge–Kutta)

| #   | Algorithm             | Note                          |
| --- | --------------------- | ----------------------------- |
| 571 | Newton–Raphson        | Root finding via tangent      |
| 572 | Bisection Method      | Interval halving              |
| 573 | Secant Method         | Approximate derivative        |
| 574 | Fixed-Point Iteration | x = f(x) convergence          |
| 575 | Gaussian Quadrature   | Weighted integration          |
| 576 | Simpson’s Rule        | Piecewise quadratic integral  |
| 577 | Trapezoidal Rule      | Linear interpolation integral |
| 578 | Runge–Kutta (RK4)     | ODE solver                    |
| 579 | Euler’s Method        | Step-by-step ODE              |
| 580 | Gradient Descent (1D) | Numerical optimization        |


### 59. Mathematical Optimization (Simplex, Gradient, Convex)

| #   | Algorithm                      | Note                        |
| --- | ------------------------------ | --------------------------- |
| 581 | Simplex Method                 | Linear programming solver   |
| 582 | Dual Simplex Method            | Solve dual constraints      |
| 583 | Interior-Point Method          | Convex optimization         |
| 584 | Gradient Descent               | Unconstrained optimization  |
| 585 | Stochastic Gradient Descent    | Sample-based updates        |
| 586 | Newton’s Method (Multivariate) | Quadratic convergence       |
| 587 | Conjugate Gradient             | Solve SPD systems           |
| 588 | Lagrange Multipliers           | Constrained optimization    |
| 589 | KKT Conditions Solver          | Convex constraint handling  |
| 590 | Coordinate Descent             | Sequential variable updates |


### 60. Algebraic Tricks and Transform Techniques

| #   | Algorithm                       | Note                        |
| --- | ------------------------------- | --------------------------- |
| 591 | Polynomial Multiplication (FFT) | Fast convolution            |
| 592 | Polynomial Inversion            | Newton iteration            |
| 593 | Polynomial Derivative           | Term-wise multiply by index |
| 594 | Polynomial Integration          | Divide by index+1           |
| 595 | Formal Power Series Composition | Substitute series           |
| 596 | Exponentiation by Squaring      | Fast powering               |
| 597 | Modular Exponentiation          | Fast power mod M            |
| 598 | Fast Walsh–Hadamard Transform   | XOR convolution             |
| 599 | Zeta Transform                  | Subset summation            |
| 600 | Möbius Inversion                | Recover original from sums  |


## Chapter 7. Strings and Text Algorithms

### 61. String Matching (KMP, Z, Rabin–Karp, Boyer–Moore)

| #   | Algorithm                 | Note                            |
| --- | ------------------------- | ------------------------------- |
| 601 | Naive String Matching     | Compare every position          |
| 602 | Knuth–Morris–Pratt (KMP)  | Prefix function skipping        |
| 603 | Z-Algorithm               | Match using Z-values            |
| 604 | Rabin–Karp                | Rolling hash comparison         |
| 605 | Boyer–Moore               | Backward skip based on mismatch |
| 606 | Boyer–Moore–Horspool      | Simplified shift table          |
| 607 | Sunday Algorithm          | Last-character shift            |
| 608 | Finite Automaton Matching | DFA-based matching              |
| 609 | Bitap Algorithm           | Bitmask approximate matching    |
| 610 | Two-Way Algorithm         | Optimal linear matching         |


### 62. Multi-Pattern Search (Aho–Corasick)

| #   | Algorithm                  | Note                        |
| --- | -------------------------- | --------------------------- |
| 611 | Aho–Corasick Automaton     | Trie + failure links        |
| 612 | Trie Construction          | Prefix tree build           |
| 613 | Failure Link Computation   | BFS for transitions         |
| 614 | Output Link Management     | Handle overlapping patterns |
| 615 | Multi-Pattern Search       | Find all keywords           |
| 616 | Dictionary Matching        | Find multiple substrings    |
| 617 | Dynamic Aho–Corasick       | Add/remove patterns         |
| 618 | Parallel AC Search         | Multi-threaded traversal    |
| 619 | Compressed AC Automaton    | Memory-optimized            |
| 620 | Extended AC with Wildcards | Flexible matching           |


### 63. Suffix Structures (Suffix Array, Suffix Tree, LCP)

| #   | Algorithm                | Note                        |
| --- | ------------------------ | --------------------------- |
| 621 | Suffix Array (Naive)     | Sort all suffixes           |
| 622 | Suffix Array (Doubling)  | O(n log n) rank-based       |
| 623 | Kasai’s LCP Algorithm    | Longest common prefix       |
| 624 | Suffix Tree (Ukkonen)    | Linear-time online          |
| 625 | Suffix Automaton         | Minimal DFA of substrings   |
| 626 | SA-IS Algorithm          | O(n) suffix array           |
| 627 | LCP RMQ Query            | Range minimum for substring |
| 628 | Generalized Suffix Array | Multiple strings            |
| 629 | Enhanced Suffix Array    | Combine SA + LCP            |
| 630 | Sparse Suffix Tree       | Space-efficient variant     |


### 64. Palindromes and Periodicity (Manacher)

| #   | Algorithm                            | Note                               |
| --- | ------------------------------------ | ---------------------------------- |
| 631 | Naive Palindrome Check               | Expand around center               |
| 632 | Manacher’s Algorithm                 | O(n) longest palindrome            |
| 633 | Longest Palindromic Substring        | Center expansion                   |
| 634 | Palindrome DP Table                  | Substring boolean matrix           |
| 635 | Palindromic Tree (Eertree)           | Track distinct palindromes         |
| 636 | Prefix Function Periodicity          | Detect repetition patterns         |
| 637 | Z-Function Periodicity               | Identify periodic suffix           |
| 638 | KMP Prefix Period Check              | Shortest repeating unit            |
| 639 | Lyndon Factorization                 | Decompose string into Lyndon words |
| 640 | Minimal Rotation (Booth’s Algorithm) | Lexicographically minimal shift    |


### 65. Edit Distance and Alignment

| #   | Algorithm                  | Note                       |
| --- | -------------------------- | -------------------------- |
| 641 | Levenshtein Distance       | Insert/delete/replace cost |
| 642 | Damerau–Levenshtein        | Swap included              |
| 643 | Hamming Distance           | Count differing bits       |
| 644 | Needleman–Wunsch           | Global alignment           |
| 645 | Smith–Waterman             | Local alignment            |
| 646 | Hirschberg’s Algorithm     | Memory-optimized alignment |
| 647 | Edit Script Reconstruction | Backtrack operations       |
| 648 | Affine Gap Penalty DP      | Varying gap cost           |
| 649 | Myers Bit-Vector Algorithm | Fast edit distance         |
| 650 | Longest Common Subsequence | Alignment by inclusion     |


### 66. Compression (Huffman, Arithmetic, LZ77, BWT)

| #   | Algorithm                 | Note                        |
| --- | ------------------------- | --------------------------- |
| 651 | Huffman Coding            | Optimal prefix tree         |
| 652 | Canonical Huffman         | Deterministic ordering      |
| 653 | Arithmetic Coding         | Interval probability coding |
| 654 | Shannon–Fano Coding       | Early prefix method         |
| 655 | Run-Length Encoding (RLE) | Repeat compression          |
| 656 | LZ77                      | Sliding-window match        |
| 657 | LZ78                      | Dictionary building         |
| 658 | LZW                       | Variant used in GIF         |
| 659 | Burrows–Wheeler Transform | Block reordering            |
| 660 | Move-to-Front Encoding    | Locality boosting transform |


### 67. Cryptographic Hashes and Checksums

| #   | Algorithm                | Note                         |
| --- | ------------------------ | ---------------------------- |
| 661 | Rolling Hash             | Polynomial mod-based         |
| 662 | CRC32                    | Cyclic redundancy check      |
| 663 | Adler-32                 | Lightweight checksum         |
| 664 | MD5                      | Legacy cryptographic hash    |
| 665 | SHA-1                    | Deprecated hash function     |
| 666 | SHA-256                  | Secure hash standard         |
| 667 | SHA-3 (Keccak)           | Sponge construction          |
| 668 | HMAC                     | Keyed message authentication |
| 669 | Merkle Tree              | Hierarchical hashing         |
| 670 | Hash Collision Detection | Birthday bound simulation    |


### 68. Approximate and Streaming Matching

| #   | Algorithm                | Note                             |
| --- | ------------------------ | -------------------------------- |
| 671 | K-Approximate Matching   | Allow k mismatches               |
| 672 | Bitap Algorithm          | Bitwise dynamic programming      |
| 673 | Landau–Vishkin Algorithm | Edit distance ≤ k                |
| 674 | Filtering Algorithm      | Fast approximate search          |
| 675 | Wu–Manber                | Multi-pattern approximate search |
| 676 | Streaming KMP            | Online prefix updates            |
| 677 | Rolling Hash Sketch      | Sliding window hashing           |
| 678 | Sketch-based Similarity  | MinHash / LSH variants           |
| 679 | Weighted Edit Distance   | Weighted operations              |
| 680 | Online Levenshtein       | Dynamic stream update            |


### 69. Bioinformatics Alignment (Needleman–Wunsch, Smith–Waterman)

| #   | Algorithm                         | Note                      |
| --- | --------------------------------- | ------------------------- |
| 681 | Needleman–Wunsch                  | Global sequence alignment |
| 682 | Smith–Waterman                    | Local alignment           |
| 683 | Gotoh Algorithm                   | Affine gap penalties      |
| 684 | Hirschberg Alignment              | Linear-space alignment    |
| 685 | Multiple Sequence Alignment (MSA) | Progressive methods       |
| 686 | Profile Alignment                 | Align sequence to profile |
| 687 | Hidden Markov Model Alignment     | Probabilistic alignment   |
| 688 | BLAST                             | Heuristic local search    |
| 689 | FASTA                             | Word-based alignment      |
| 690 | Pairwise DP Alignment             | General DP framework      |


### 70. Text Indexing and Search Structures

| #   | Algorithm                          | Note                       |
| --- | ---------------------------------- | -------------------------- |
| 691 | Inverted Index Build               | Word-to-document mapping   |
| 692 | Positional Index                   | Store word positions       |
| 693 | TF-IDF Weighting                   | Importance scoring         |
| 694 | BM25 Ranking                       | Modern ranking formula     |
| 695 | Trie Index                         | Prefix search structure    |
| 696 | Suffix Array Index                 | Substring search           |
| 697 | Compressed Suffix Array            | Space-optimized            |
| 698 | FM-Index                           | BWT-based compressed index |
| 699 | DAWG (Directed Acyclic Word Graph) | Shared suffix graph        |
| 700 | Wavelet Tree for Text              | Rank/select on sequences   |


## Chapter 8. Geometry, Graphics, and Spatial Algorithms

### 71. Convex Hull (Graham, Andrew, Chan)

| #   | Algorithm                    | Note                                     |
| --- | ---------------------------- | ---------------------------------------- |
| 701 | Gift Wrapping (Jarvis March) | Wrap hull one point at a time            |
| 702 | Graham Scan                  | Sort by angle, maintain stack            |
| 703 | Andrew’s Monotone Chain      | Sort by x, upper + lower hull            |
| 704 | Chan’s Algorithm             | Output-sensitive O(n log h)              |
| 705 | QuickHull                    | Divide-and-conquer hull                  |
| 706 | Incremental Convex Hull      | Add points one by one                    |
| 707 | Divide & Conquer Hull        | Merge two partial hulls                  |
| 708 | 3D Convex Hull               | Extend to 3D geometry                    |
| 709 | Dynamic Convex Hull          | Maintain hull with inserts               |
| 710 | Rotating Calipers            | Compute diameter, width, antipodal pairs |


### 72. Closest Pair and Segment Intersection

| #   | Algorithm                       | Note                          |
| --- | ------------------------------- | ----------------------------- |
| 711 | Closest Pair (Divide & Conquer) | Split, merge minimal distance |
| 712 | Closest Pair (Sweep Line)       | Maintain active window        |
| 713 | Brute Force Closest Pair        | Check all O(n²) pairs         |
| 714 | Bentley–Ottmann                 | Find all line intersections   |
| 715 | Segment Intersection Test       | Cross product orientation     |
| 716 | Line Sweep for Segments         | Event-based intersection      |
| 717 | Intersection via Orientation    | CCW test                      |
| 718 | Circle Intersection             | Geometry of two circles       |
| 719 | Polygon Intersection            | Clip overlapping polygons     |
| 720 | Nearest Neighbor Pair           | Combine KD-tree + search      |


### 73. Line Sweep and Plane Sweep Algorithms

| #   | Algorithm                              | Note                             |
| --- | -------------------------------------- | -------------------------------- |
| 721 | Sweep Line for Events                  | Process sorted events            |
| 722 | Interval Scheduling                    | Select non-overlapping intervals |
| 723 | Rectangle Union Area                   | Sweep edges to count area        |
| 724 | Segment Intersection (Bentley–Ottmann) | Detect all crossings             |
| 725 | Skyline Problem                        | Merge height profiles            |
| 726 | Closest Pair Sweep                     | Maintain active set              |
| 727 | Circle Arrangement                     | Sweep and count regions          |
| 728 | Sweep for Overlapping Rectangles       | Detect collisions                |
| 729 | Range Counting                         | Count points in rectangle        |
| 730 | Plane Sweep for Triangles              | Polygon overlay computation      |


### 74. Delaunay and Voronoi Diagrams

| #   | Algorithm                            | Note                             |
| --- | ------------------------------------ | -------------------------------- |
| 731 | Delaunay Triangulation (Incremental) | Add points, maintain Delaunay    |
| 732 | Delaunay (Divide & Conquer)          | Merge triangulations             |
| 733 | Delaunay (Fortune’s Sweep)           | O(n log n) construction          |
| 734 | Voronoi Diagram (Fortune’s)          | Sweep line beachline             |
| 735 | Incremental Voronoi                  | Update on insertion              |
| 736 | Bowyer–Watson                        | Empty circle criterion           |
| 737 | Duality Transform                    | Convert between Voronoi/Delaunay |
| 738 | Power Diagram                        | Weighted Voronoi                 |
| 739 | Lloyd’s Relaxation                   | Smooth Voronoi cells             |
| 740 | Voronoi Nearest Neighbor             | Region-based lookup              |


### 75. Point in Polygon and Polygon Triangulation

| #   | Algorithm                              | Note                     |
| --- | -------------------------------------- | ------------------------ |
| 741 | Ray Casting                            | Count edge crossings     |
| 742 | Winding Number                         | Angle sum method         |
| 743 | Convex Polygon Point Test              | Orientation checks       |
| 744 | Ear Clipping Triangulation             | Remove ears iteratively  |
| 745 | Monotone Polygon Triangulation         | Sweep line triangulation |
| 746 | Delaunay Triangulation                 | Optimal triangle quality |
| 747 | Convex Decomposition                   | Split into convex parts  |
| 748 | Polygon Area (Shoelace Formula)        | Signed area computation  |
| 749 | Minkowski Sum                          | Add shapes geometrically |
| 750 | Polygon Intersection (Weiler–Atherton) | Clip overlapping shapes  |


### 76. Spatial Data Structures (KD, R-tree)

| #   | Algorithm                         | Note                     |
| --- | --------------------------------- | ------------------------ |
| 751 | KD-Tree Build                     | Recursive median split   |
| 752 | KD-Tree Search                    | Axis-aligned query       |
| 753 | Range Search KD-Tree              | Orthogonal query         |
| 754 | Nearest Neighbor KD-Tree          | Closest point search     |
| 755 | R-Tree Build                      | Bounding box hierarchy   |
| 756 | R*-Tree                           | Optimized split strategy |
| 757 | Quad Tree                         | Spatial decomposition    |
| 758 | Octree                            | 3D spatial decomposition |
| 759 | BSP Tree (Binary Space Partition) | Split by planes          |
| 760 | Morton Order (Z-Curve)            | Spatial locality index   |


### 77. Rasterization and Scanline Techniques

| #   | Algorithm                     | Note                         |
| --- | ----------------------------- | ---------------------------- |
| 761 | Bresenham’s Line Algorithm    | Efficient integer drawing    |
| 762 | Midpoint Circle Algorithm     | Circle rasterization         |
| 763 | Scanline Fill                 | Polygon interior fill        |
| 764 | Edge Table Fill               | Sort edges by y              |
| 765 | Z-Buffer Algorithm            | Hidden surface removal       |
| 766 | Painter’s Algorithm           | Sort by depth                |
| 767 | Gouraud Shading               | Vertex interpolation shading |
| 768 | Phong Shading                 | Normal interpolation         |
| 769 | Anti-Aliasing (Supersampling) | Smooth jagged edges          |
| 770 | Scanline Polygon Clipping     | Efficient clipping           |


### 78. Computer Vision (Canny, Hough, SIFT)

| #   | Algorithm                                | Note                           |
| --- | ---------------------------------------- | ------------------------------ |
| 771 | Canny Edge Detector                      | Gradient + hysteresis          |
| 772 | Sobel Operator                           | Gradient magnitude filter      |
| 773 | Hough Transform (Lines)                  | Accumulator for line detection |
| 774 | Hough Transform (Circles)                | Radius-based accumulator       |
| 775 | Harris Corner Detector                   | Eigenvalue-based corners       |
| 776 | FAST Corner Detector                     | Intensity circle test          |
| 777 | SIFT (Scale-Invariant Feature Transform) | Keypoint detection             |
| 778 | SURF (Speeded-Up Robust Features)        | Faster descriptor              |
| 779 | ORB (Oriented FAST + BRIEF)              | Binary robust feature          |
| 780 | RANSAC                                   | Robust model fitting           |


### 79. Pathfinding in Space (A*, RRT, PRM)

| #   | Algorithm                           | Note                          |
| --- | ----------------------------------- | ----------------------------- |
| 781 | A* Search                           | Heuristic pathfinding         |
| 782 | Dijkstra for Grid                   | Weighted shortest path        |
| 783 | Theta*                              | Any-angle pathfinding         |
| 784 | Jump Point Search                   | Grid acceleration             |
| 785 | RRT (Rapidly-Exploring Random Tree) | Random sampling tree          |
| 786 | RRT*                                | Optimal variant with rewiring |
| 787 | PRM (Probabilistic Roadmap)         | Graph sampling planner        |
| 788 | Visibility Graph                    | Connect visible vertices      |
| 789 | Potential Field Pathfinding         | Gradient-based navigation     |
| 790 | Bug Algorithms                      | Simple obstacle avoidance     |


### 80. Computational Geometry Variants and Applications

| #   | Algorithm                        | Note                     |
| --- | -------------------------------- | ------------------------ |
| 791 | Convex Polygon Intersection      | Clip convex sets         |
| 792 | Minkowski Sum                    | Shape convolution        |
| 793 | Rotating Calipers                | Closest/farthest pair    |
| 794 | Half-Plane Intersection          | Feasible region          |
| 795 | Line Arrangement                 | Count regions            |
| 796 | Point Location (Trapezoidal Map) | Query region lookup      |
| 797 | Voronoi Nearest Facility         | Region query             |
| 798 | Delaunay Mesh Generation         | Triangulation refinement |
| 799 | Smallest Enclosing Circle        | Welzl’s algorithm        |
| 800 | Collision Detection (SAT)        | Separating axis theorem  |


## Chapter 9. Systems, Databases, and Distributed Algorithms

### 81. Concurrency Control (2PL, MVCC, OCC)

| #   | Algorithm                               | Note                          |
| --- | --------------------------------------- | ----------------------------- |
| 801 | Two-Phase Locking (2PL)                 | Acquire-then-release locks    |
| 802 | Strict 2PL                              | Hold locks until commit       |
| 803 | Conservative 2PL                        | Prevent deadlocks via prelock |
| 804 | Timestamp Ordering                      | Schedule by timestamps        |
| 805 | Multiversion Concurrency Control (MVCC) | Snapshot isolation            |
| 806 | Optimistic Concurrency Control (OCC)    | Validate at commit            |
| 807 | Serializable Snapshot Isolation         | Merge read/write sets         |
| 808 | Lock-Free Algorithm                     | Atomic CAS updates            |
| 809 | Wait-Die / Wound-Wait                   | Deadlock prevention policies  |
| 810 | Deadlock Detection (Wait-for Graph)     | Cycle detection in waits      |


### 82. Logging, Recovery, and Commit Protocols

| #   | Algorithm                 | Note                      |
| --- | ------------------------- | ------------------------- |
| 811 | Write-Ahead Logging (WAL) | Log before commit         |
| 812 | ARIES Recovery            | Re-do/undo with LSNs      |
| 813 | Shadow Paging             | Copy-on-write persistence |
| 814 | Two-Phase Commit (2PC)    | Coordinator-driven commit |
| 815 | Three-Phase Commit (3PC)  | Non-blocking variant      |
| 816 | Checkpointing             | Save state for recovery   |
| 817 | Undo Logging              | Rollback uncommitted      |
| 818 | Redo Logging              | Reapply committed         |
| 819 | Quorum Commit             | Majority agreement        |
| 820 | Consensus Commit          | Combine 2PC + Paxos       |


### 83. Scheduling (Round Robin, EDF, Rate-Monotonic)

| #   | Algorithm                       | Note                            |
| --- | ------------------------------- | ------------------------------- |
| 821 | First-Come First-Served (FCFS)  | Sequential job order            |
| 822 | Shortest Job First (SJF)        | Optimal average wait            |
| 823 | Round Robin (RR)                | Time-slice fairness             |
| 824 | Priority Scheduling             | Weighted selection              |
| 825 | Multilevel Queue                | Tiered priority queues          |
| 826 | Earliest Deadline First (EDF)   | Real-time optimal               |
| 827 | Rate Monotonic Scheduling (RMS) | Fixed periodic priority         |
| 828 | Lottery Scheduling              | Probabilistic fairness          |
| 829 | Multilevel Feedback Queue       | Adaptive behavior               |
| 830 | Fair Queuing (FQ)               | Flow-based proportional sharing |


### 84. Caching and Replacement (LRU, LFU, CLOCK)

| #   | Algorithm                              | Note                       |
| --- | -------------------------------------- | -------------------------- |
| 831 | LRU (Least Recently Used)              | Evict oldest used          |
| 832 | LFU (Least Frequently Used)            | Evict lowest frequency     |
| 833 | FIFO Cache                             | Simple queue eviction      |
| 834 | CLOCK Algorithm                        | Approximate LRU            |
| 835 | ARC (Adaptive Replacement Cache)       | Mix of recency + frequency |
| 836 | Two-Queue (2Q)                         | Separate recent/frequent   |
| 837 | LIRS (Low Inter-reference Recency Set) | Predict reuse distance     |
| 838 | TinyLFU                                | Frequency sketch admission |
| 839 | Random Replacement                     | Simple stochastic policy   |
| 840 | Belady’s Optimal                       | Evict farthest future use  |


### 85. Networking (Routing, Congestion Control)

| #   | Algorithm                              | Note                    |
| --- | -------------------------------------- | ----------------------- |
| 841 | Dijkstra’s Routing                     | Shortest path routing   |
| 842 | Bellman–Ford Routing                   | Distance-vector routing |
| 843 | Link-State Routing (OSPF)              | Global view routing     |
| 844 | Distance-Vector Routing (RIP)          | Local neighbor updates  |
| 845 | Path Vector (BGP)                      | Route advertisement     |
| 846 | Flooding                               | Broadcast to all nodes  |
| 847 | Spanning Tree Protocol                 | Loop-free topology      |
| 848 | Congestion Control (AIMD)              | TCP window control      |
| 849 | Random Early Detection (RED)           | Queue preemptive drop   |
| 850 | ECN (Explicit Congestion Notification) | Mark packets early      |


### 86. Distributed Consensus (Paxos, Raft, PBFT)

| #   | Algorithm                                  | Note                              |
| --- | ------------------------------------------ | --------------------------------- |
| 851 | Basic Paxos                                | Majority consensus                |
| 852 | Multi-Paxos                                | Sequence of agreements            |
| 853 | Raft                                       | Log replication + leader election |
| 854 | Viewstamped Replication                    | Alternative consensus design      |
| 855 | PBFT (Practical Byzantine Fault Tolerance) | Byzantine safety                  |
| 856 | Zab (Zookeeper Atomic Broadcast)           | Broadcast + ordering              |
| 857 | EPaxos                                     | Leaderless fast path              |
| 858 | VRR (Virtual Ring Replication)             | Log around ring                   |
| 859 | Two-Phase Commit with Consensus            | Transactional commit              |
| 860 | Chain Replication                          | Ordered state replication         |


### 87. Load Balancing and Rate Limiting

| #   | Algorithm                  | Note                          |
| --- | -------------------------- | ----------------------------- |
| 861 | Round Robin Load Balancing | Sequential distribution       |
| 862 | Weighted Round Robin       | Proportional to weight        |
| 863 | Least Connections          | Pick least loaded node        |
| 864 | Consistent Hashing         | Map requests stably           |
| 865 | Power of Two Choices       | Sample and choose lesser load |
| 866 | Random Load Balancing      | Simple uniform random         |
| 867 | Token Bucket               | Rate-based limiter            |
| 868 | Leaky Bucket               | Steady flow shaping           |
| 869 | Sliding Window Counter     | Rolling time window           |
| 870 | Fixed Window Counter       | Resettable counter limiter    |


### 88. Search and Indexing (Inverted, BM25, WAND)

| #   | Algorithm                   | Note                        |
| --- | --------------------------- | --------------------------- |
| 871 | Inverted Index Construction | Word → document list        |
| 872 | Positional Index Build      | Store term positions        |
| 873 | TF-IDF Scoring              | Term frequency weighting    |
| 874 | BM25 Ranking                | Modern scoring model        |
| 875 | Boolean Retrieval           | Logical AND/OR/NOT          |
| 876 | WAND Algorithm              | Efficient top-k retrieval   |
| 877 | Block-Max WAND (BMW)        | Early skipping optimization |
| 878 | Impact-Ordered Indexing     | Sort by contribution        |
| 879 | Tiered Indexing             | Prioritize high-score docs  |
| 880 | DAAT vs SAAT Evaluation     | Document vs score-at-a-time |


### 89. Compression and Encoding in Systems

| #   | Algorithm                 | Note                             |
| --- | ------------------------- | -------------------------------- |
| 881 | Run-Length Encoding (RLE) | Simple repetition encoding       |
| 882 | Huffman Coding            | Optimal variable-length code     |
| 883 | Arithmetic Coding         | Fractional interval coding       |
| 884 | Delta Encoding            | Store differences                |
| 885 | Variable Byte Encoding    | Compact integers                 |
| 886 | Elias Gamma Coding        | Prefix integer encoding          |
| 887 | Rice Coding               | Unary + remainder scheme         |
| 888 | Snappy                    | Fast block compression           |
| 889 | Zstandard (Zstd)          | Modern adaptive codec            |
| 890 | LZ4                       | High-speed dictionary compressor |


### 90. Fault Tolerance and Replication

| #   | Algorithm                  | Note                     |
| --- | -------------------------- | ------------------------ |
| 891 | Primary–Backup Replication | One leader, one standby  |
| 892 | Quorum Replication         | Majority write/read rule |
| 893 | Chain Replication          | Ordered consistency      |
| 894 | Gossip Protocol            | Epidemic state exchange  |
| 895 | Anti-Entropy Repair        | Periodic reconciliation  |
| 896 | Erasure Coding             | Redundant data blocks    |
| 897 | Checksum Verification      | Detect corruption        |
| 898 | Heartbeat Monitoring       | Liveness detection       |
| 899 | Leader Election (Bully)    | Highest ID wins          |
| 900 | Leader Election (Ring)     | Token-based rotation     |


## Chapter 10. AI, ML, and Optimization

### 91. Classical ML (k-means, Naive Bayes, SVM, Decision Trees)

| #   | Algorithm                          | Note                               |
| --- | ---------------------------------- | ---------------------------------- |
| 901 | k-Means Clustering                 | Partition by centroid iteration    |
| 902 | k-Medoids (PAM)                    | Cluster by exemplars               |
| 903 | Gaussian Mixture Model (EM)        | Soft probabilistic clustering      |
| 904 | Naive Bayes Classifier             | Probabilistic feature independence |
| 905 | Logistic Regression                | Sigmoid linear classifier          |
| 906 | Perceptron                         | Online linear separator            |
| 907 | Decision Tree (CART)               | Recursive partition by impurity    |
| 908 | ID3 Algorithm                      | Information gain splitting         |
| 909 | k-Nearest Neighbors (kNN)          | Distance-based classification      |
| 910 | Linear Discriminant Analysis (LDA) | Projection for separation          |


### 92. Ensemble Methods (Bagging, Boosting, Random Forests)

| #   | Algorithm         | Note                              |
| --- | ----------------- | --------------------------------- |
| 911 | Bagging           | Bootstrap aggregation             |
| 912 | Random Forest     | Ensemble of decision trees        |
| 913 | AdaBoost          | Weighted error correction         |
| 914 | Gradient Boosting | Sequential residual fitting       |
| 915 | XGBoost           | Optimized gradient boosting       |
| 916 | LightGBM          | Histogram-based leaf growth       |
| 917 | CatBoost          | Ordered boosting for categoricals |
| 918 | Stacking          | Meta-model ensemble               |
| 919 | Voting Classifier | Majority aggregation              |
| 920 | Snapshot Ensemble | Averaged checkpoints              |


### 93. Gradient Methods (SGD, Adam, RMSProp)

| #   | Algorithm                         | Note                        |
| --- | --------------------------------- | --------------------------- |
| 921 | Gradient Descent                  | Batch full-gradient step    |
| 922 | Stochastic Gradient Descent (SGD) | Sample-wise updates         |
| 923 | Mini-Batch SGD                    | Tradeoff speed and variance |
| 924 | Momentum                          | Add velocity to descent     |
| 925 | Nesterov Accelerated Gradient     | Lookahead correction        |
| 926 | AdaGrad                           | Adaptive per-parameter rate |
| 927 | RMSProp                           | Exponential moving average  |
| 928 | Adam                              | Momentum + adaptive rate    |
| 929 | AdamW                             | Decoupled weight decay      |
| 930 | L-BFGS                            | Limited-memory quasi-Newton |


### 94. Deep Learning (Backpropagation, Dropout, Normalization)

| #   | Algorithm                | Note                       |
| --- | ------------------------ | -------------------------- |
| 931 | Backpropagation          | Gradient chain rule        |
| 932 | Xavier/He Initialization | Scaled variance init       |
| 933 | Dropout                  | Random neuron deactivation |
| 934 | Batch Normalization      | Normalize per batch        |
| 935 | Layer Normalization      | Normalize per feature      |
| 936 | Gradient Clipping        | Prevent explosion          |
| 937 | Early Stopping           | Prevent overfitting        |
| 938 | Weight Decay             | Regularization via penalty |
| 939 | Learning Rate Scheduling | Dynamic LR adjustment      |
| 940 | Residual Connections     | Skip layer improvement     |


### 95. Sequence Models (Viterbi, Beam Search, CTC)

| #   | Algorithm                                   | Note                         |
| --- | ------------------------------------------- | ---------------------------- |
| 941 | Hidden Markov Model (Forward–Backward)      | Probabilistic sequence model |
| 942 | Viterbi Algorithm                           | Most probable path           |
| 943 | Baum–Welch                                  | EM training for HMMs         |
| 944 | Beam Search                                 | Top-k path exploration       |
| 945 | Greedy Decoding                             | Fast approximate decoding    |
| 946 | Connectionist Temporal Classification (CTC) | Unaligned sequence training  |
| 947 | Attention Mechanism                         | Weighted context aggregation |
| 948 | Transformer Decoder                         | Self-attention stack         |
| 949 | Seq2Seq with Attention                      | Encoder-decoder framework    |
| 950 | Pointer Network                             | Output index selection       |


### 96. Metaheuristics (GA, SA, PSO, ACO)

| #   | Algorithm                         | Note                           |
| --- | --------------------------------- | ------------------------------ |
| 951 | Genetic Algorithm (GA)            | Evolutionary optimization      |
| 952 | Simulated Annealing (SA)          | Temperature-controlled search  |
| 953 | Tabu Search                       | Memory of forbidden moves      |
| 954 | Particle Swarm Optimization (PSO) | Velocity-based search          |
| 955 | Ant Colony Optimization (ACO)     | Pheromone-guided path          |
| 956 | Differential Evolution (DE)       | Vector-based mutation          |
| 957 | Harmony Search                    | Music-inspired improvisation   |
| 958 | Firefly Algorithm                 | Brightness-attraction movement |
| 959 | Bee Colony Optimization           | Explore-exploit via scouts     |
| 960 | Hill Climbing                     | Local incremental improvement  |


### 97. Reinforcement Learning (Q-learning, Policy Gradients)

| #   | Algorithm                          | Note                        |
| --- | ---------------------------------- | --------------------------- |
| 961 | Monte Carlo Control                | Average returns             |
| 962 | Temporal Difference (TD) Learning  | Bootstrap updates           |
| 963 | SARSA                              | On-policy TD learning       |
| 964 | Q-Learning                         | Off-policy TD learning      |
| 965 | Double Q-Learning                  | Reduce overestimation       |
| 966 | Deep Q-Network (DQN)               | Neural Q approximator       |
| 967 | REINFORCE                          | Policy gradient by sampling |
| 968 | Actor–Critic                       | Value-guided policy update  |
| 969 | PPO (Proximal Policy Optimization) | Clipped surrogate objective |
| 970 | DDPG / SAC                         | Continuous action RL        |


### 98. Approximation and Online Algorithms

| #   | Algorithm                        | Note                                   |
| --- | -------------------------------- | -------------------------------------- |
| 971 | Greedy Set Cover                 | ln(n)-approximation                    |
| 972 | Vertex Cover Approximation       | Double-matching heuristic              |
| 973 | Traveling Salesman Approximation | MST-based 2-approx                     |
| 974 | k-Center Approximation           | Farthest-point heuristic               |
| 975 | Online Paging (LRU)              | Competitive analysis                   |
| 976 | Online Matching (Ranking)        | Adversarial input resilience           |
| 977 | Online Knapsack                  | Ratio-based acceptance                 |
| 978 | Competitive Ratio Evaluation     | Bound worst-case performance           |
| 979 | PTAS / FPTAS Schemes             | Polynomial approximation               |
| 980 | Primal–Dual Method               | Approximate combinatorial optimization |


### 99. Fairness, Causal Inference, and Robust Optimization

| #   | Algorithm                            | Note                          |
| --- | ------------------------------------ | ----------------------------- |
| 981 | Reweighting for Fairness             | Adjust sample weights         |
| 982 | Demographic Parity Constraint        | Equalize positive rates       |
| 983 | Equalized Odds                       | Align error rates             |
| 984 | Adversarial Debiasing                | Learn fair representations    |
| 985 | Causal DAG Discovery                 | Graphical causal inference    |
| 986 | Propensity Score Matching            | Estimate treatment effect     |
| 987 | Instrumental Variable Estimation     | Handle confounders            |
| 988 | Robust Optimization                  | Worst-case aware optimization |
| 989 | Distributionally Robust Optimization | Minimax over uncertainty sets |
| 990 | Counterfactual Fairness              | Simulate do-interventions     |


### 100. AI Planning, Search, and Learning Systems

| #    | Algorithm                       | Note                          |
| ---- | ------------------------------- | ----------------------------- |
| 991  | Breadth-First Search (BFS)      | Uninformed search             |
| 992  | Depth-First Search (DFS)        | Backtracking search           |
| 993  | A* Search                       | Heuristic guided              |
| 994  | Iterative Deepening A* (IDA*)   | Memory-bounded heuristic      |
| 995  | Uniform Cost Search             | Expand by path cost           |
| 996  | Monte Carlo Tree Search (MCTS)  | Exploration vs exploitation   |
| 997  | Minimax                         | Game tree evaluation          |
| 998  | Alpha–Beta Pruning              | Prune unneeded branches       |
| 999  | STRIPS Planning                 | Action-based state transition |
| 1000 | Hierarchical Task Network (HTN) | Structured AI planning        |

