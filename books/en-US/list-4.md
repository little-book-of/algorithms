# Chapter 4. Graph Algorithms 

## Section 31. Traversals 

### 301 Depth-First Search (Recursive)

Depth-First Search (DFS) is the classic way to explore a graph deeply before backtracking. You pick a starting node, walk as far as possible along one path, and only when you hit a dead end do you turn back. It's the "go deep first, ask questions later" of graph algorithms.

#### What Problem Are We Solving?

DFS helps us visit every vertex and edge in a connected component systematically. It's the foundation for exploring graphs, detecting cycles, classifying edges, and building more complex algorithms like topological sort, strongly connected components, and articulation point detection.

We want an algorithm that:

- Explores all reachable vertices from a start node
- Avoids revisiting nodes
- Records traversal order

Example:
You have a maze. DFS is the explorer that picks a path, goes as far as it can, and only turns back when stuck.

#### How Does It Work (Plain Language)?

Think of DFS like a curious traveler: always dive deeper whenever you see a new path. When you can't go further, step back one level and continue exploring.

We use recursion to model this behavior naturally, each recursive call represents entering a new node, and returning means backtracking.

| Step | Current Node                 | Action        | Stack (Call Path) |
| ---- | ---------------------------- | ------------- | ----------------- |
| 1    | A                            | Visit A       | [A]               |
| 2    | B                            | Visit B (A→B) | [A, B]            |
| 3    | D                            | Visit D (B→D) | [A, B, D]         |
| 4    | D has no unvisited neighbors | Backtrack     | [A, B]            |
| 5    | B's next neighbor C          | Visit C       | [A, B, C]         |
| 6    | C done                       | Backtrack     | [A, B] → [A]      |
| 7    | A's remaining neighbors      | Visit next    | [...]             |

When recursion unwinds, we've explored the whole reachable graph.

#### Tiny Code (Easy Versions)

C (Adjacency List Example)

```c
#include <stdio.h>
#include <stdbool.h>

#define MAX 100
int graph[MAX][MAX];
bool visited[MAX];
int n;

void dfs(int v) {
    visited[v] = true;
    printf("%d ", v);
    for (int u = 0; u < n; u++) {
        if (graph[v][u] && !visited[u]) {
            dfs(u);
        }
    }
}

int main(void) {
    printf("Enter number of vertices: ");
    scanf("%d", &n);

    printf("Enter adjacency matrix:\n");
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            scanf("%d", &graph[i][j]);

    printf("DFS starting from vertex 0:\n");
    dfs(0);
}
```

Python (Adjacency List)

```python
graph = {
    0: [1, 2],
    1: [2],
    2: [0, 3],
    3: [3]
}

visited = set()

def dfs(v):
    visited.add(v)
    print(v, end=" ")
    for u in graph[v]:
        if u not in visited:
            dfs(u)

dfs(0)
```

#### Why It Matters

- Core for graph exploration and reachability
- Forms the basis of topological sort, SCC, bridges, and cycles
- Simple recursive structure reveals natural hierarchy of a graph
- Helps understand backtracking and stack-based thinking

#### A Gentle Proof (Why It Works)

Each vertex is visited exactly once:

- When a node is first discovered, it's marked `visited`
- The recursion ensures all its neighbors are explored
- Once all children are done, the function returns (backtrack)

So every vertex `v` triggers one call `dfs(v)`, giving O(V + E) time (each edge explored once).

#### Try It Yourself

1. Draw a small graph (A–B–C–D) and trace DFS.
2. Modify to print entry and exit times.
3. Track parent nodes to build DFS tree.
4. Add detection for back edges (cycle test).
5. Compare with BFS traversal order.

#### Test Cases

| Graph                     | Start | Expected Order | Notes                          |
| ------------------------- | ----- | -------------- | ------------------------------ |
| A–B–C–D chain             | A     | A B C D        | Straight path                  |
| Triangle (A–B–C–A)        | A     | A B C          | Visits all, stops at visited A |
| Disconnected {A–B}, {C–D} | A     | A B            | Only reachable component       |
| Directed A→B→C            | A     | A B C          | Linear chain                   |
| Tree root 0               | 0     | 0 1 3 4 2 5    | Depends on adjacency order     |

#### Complexity

- Time: O(V + E)
- Space: O(V) (recursion stack + visited)

DFS is your first lens into graph structure, recursive, elegant, and revealing hidden pathways one stack frame at a time.

### 302 Depth-First Search (Iterative)

Depth-First Search can run without recursion too. Instead of leaning on the call stack, we build our own stack explicitly. It's the same journey, diving deep before backtracking, just with manual control over what's next.

#### What Problem Are We Solving?

Recursion is elegant but not always practical.
Some graphs are deep, and recursive DFS can overflow the call stack. The iterative version solves that by using a stack data structure directly, mirroring the same traversal order.

We want an algorithm that:

- Works even when recursion is too deep
- Explicitly manages visited nodes and stack
- Produces the same traversal as recursive DFS

Example:
Think of it like keeping your own to-do list of unexplored paths, each time you go deeper, you add new destinations on top of the stack.

#### How Does It Work (Plain Language)?

We maintain a stack:

1. Start from a node `s`, push it on the stack.
2. Pop the top node `v`.
3. If `v` is unvisited, mark and process it.
4. Push all unvisited neighbors of `v` onto the stack.
5. Repeat until the stack is empty.

| Step | Stack (Top → Bottom) | Action              | Visited      |
| ---- | -------------------- | ------------------- | ------------ |
| 1    | [A]                  | Start, pop A        | {A}          |
| 2    | [B, C]               | Push neighbors of A | {A}          |
| 3    | [C, B]               | Pop B, visit B      | {A, B}       |
| 4    | [C, D]               | Push neighbors of B | {A, B}       |
| 5    | [D, C]               | Pop D, visit D      | {A, B, D}    |
| 6    | [C]                  | Continue            | {A, B, D}    |
| 7    | [ ]                  | Pop C, visit C      | {A, B, C, D} |

#### Tiny Code (Easy Versions)

C (Adjacency Matrix Example)

```c
#include <stdio.h>
#include <stdbool.h>

#define MAX 100
int graph[MAX][MAX];
bool visited[MAX];
int stack[MAX];
int top = -1;
int n;

void push(int v) { stack[++top] = v; }
int pop() { return stack[top--]; }
bool is_empty() { return top == -1; }

void dfs_iterative(int start) {
    push(start);
    while (!is_empty()) {
        int v = pop();
        if (!visited[v]) {
            visited[v] = true;
            printf("%d ", v);
            for (int u = n - 1; u >= 0; u--) { // reverse for consistent order
                if (graph[v][u] && !visited[u])
                    push(u);
            }
        }
    }
}

int main(void) {
    printf("Enter number of vertices: ");
    scanf("%d", &n);

    printf("Enter adjacency matrix:\n");
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            scanf("%d", &graph[i][j]);

    printf("Iterative DFS from 0:\n");
    dfs_iterative(0);
}
```

Python (Using List as Stack)

```python
graph = {
    0: [1, 2],
    1: [2],
    2: [0, 3],
    3: [3]
}

visited = set()
stack = [0]

while stack:
    v = stack.pop()
    if v not in visited:
        visited.add(v)
        print(v, end=" ")
        for u in reversed(graph[v]):  # reversed for DFS-like order
            if u not in visited:
                stack.append(u)
```

#### Why It Matters

- Avoids recursion limits and stack overflow
- Clear control over traversal order
- Good for systems with limited call stack
- Builds understanding of explicit stack simulation

#### A Gentle Proof (Why It Works)

The stack mimics recursion:
Each vertex `v` is processed once when popped, and its neighbors are pushed.
Every edge is examined exactly once.
So total operations = O(V + E), same as recursive DFS.

Each push = one recursive call; each pop = one return.

#### Try It Yourself

1. Trace iterative DFS on a small graph.
2. Compare the order with the recursive version.
3. Experiment with neighbor push order, see how output changes.
4. Add discovery and finishing times.
5. Convert to iterative topological sort by pushing finishing order to a second stack.

#### Test Cases

| Graph                    | Start | Order (One Possible) | Notes                     |
| ------------------------ | ----- | -------------------- | ------------------------- |
| 0–1–2 chain              | 0     | 0 1 2                | Simple path               |
| 0→1, 0→2, 1→3            | 0     | 0 1 3 2              | Depends on neighbor order |
| Cycle 0→1→2→0            | 0     | 0 1 2                | No repeats                |
| Disconnected             | 0     | 0 1 2                | Only connected part       |
| Complete graph (4 nodes) | 0     | 0 1 2 3              | Visits all once           |

#### Complexity

- Time: O(V + E)
- Space: O(V) for stack and visited array

Iterative DFS is your manual-gear version of recursion, same depth, same discovery, just no surprises from the call stack.

### 303 Breadth-First Search (Queue)

Breadth-First Search (BFS) is the explorer that moves level by level, radiating outward from the start. Instead of diving deep like DFS, BFS keeps things fair, it visits all neighbors before going deeper.

#### What Problem Are We Solving?

We want a way to:

- Explore all reachable vertices in a graph
- Discover the shortest path in unweighted graphs
- Process nodes in increasing distance order

BFS is perfect when:

- Edges all have equal weight (like 1)
- You need the fewest steps to reach a goal
- You're finding connected components, levels, or distances

Example:
Imagine spreading a rumor. Each person tells all their friends before the next wave begins, that's BFS in action.

#### How Does It Work (Plain Language)?

BFS uses a queue, a first-in, first-out line.

1. Start from a node `s`
2. Mark it visited and enqueue it
3. While queue not empty:

   * Dequeue front node `v`
   * Visit `v`
   * Enqueue all unvisited neighbors of `v`

| Step | Queue (Front → Back) | Visited            | Action                     |
| ---- | -------------------- | ------------------ | -------------------------- |
| 1    | [A]                  | {A}                | Start                      |
| 2    | [B, C]               | {A, B, C}          | A's neighbors              |
| 3    | [C, D, E]            | {A, B, C, D, E}    | Visit B, add its neighbors |
| 4    | [D, E, F]            | {A, B, C, D, E, F} | Visit C, add neighbors     |
| 5    | [E, F]               | {A…F}              | Continue until empty       |

The order you dequeue = level order traversal.

#### Tiny Code (Easy Versions)

C (Adjacency Matrix Example)

```c
#include <stdio.h>
#include <stdbool.h>

#define MAX 100
int graph[MAX][MAX];
bool visited[MAX];
int queue[MAX];
int front = 0, rear = 0;
int n;

void enqueue(int v) { queue[rear++] = v; }
int dequeue() { return queue[front++]; }
bool is_empty() { return front == rear; }

void bfs(int start) {
    visited[start] = true;
    enqueue(start);

    while (!is_empty()) {
        int v = dequeue();
        printf("%d ", v);

        for (int u = 0; u < n; u++) {
            if (graph[v][u] && !visited[u]) {
                visited[u] = true;
                enqueue(u);
            }
        }
    }
}

int main(void) {
    printf("Enter number of vertices: ");
    scanf("%d", &n);

    printf("Enter adjacency matrix:\n");
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            scanf("%d", &graph[i][j]);

    printf("BFS starting from vertex 0:\n");
    bfs(0);
}
```

Python (Adjacency List)

```python
from collections import deque

graph = {
    0: [1, 2],
    1: [3, 4],
    2: [5],
    3: [],
    4: [],
    5: []
}

visited = set()
queue = deque([0])
visited.add(0)

while queue:
    v = queue.popleft()
    print(v, end=" ")
    for u in graph[v]:
        if u not in visited:
            visited.add(u)
            queue.append(u)
```

#### Why It Matters

- Finds shortest paths in unweighted graphs
- Guarantees level order visitation
- Core for algorithms like 0–1 BFS, SPFA, and Dijkstra's
- Excellent for layer-based exploration and distance labeling

#### A Gentle Proof (Why It Works)

Each vertex is visited exactly once:

- It's enqueued when discovered
- It's dequeued once for processing
- Each edge is checked once

If all edges have weight 1, BFS discovers vertices in increasing distance order, proving shortest-path correctness.

Time complexity:

- Visiting each vertex: O(V)
- Scanning each edge: O(E)
  → Total: O(V + E)

#### Try It Yourself

1. Draw a small unweighted graph and run BFS by hand.
2. Record levels (distance from start).
3. Track parent of each vertex, reconstruct shortest path.
4. Try BFS on a tree, compare with level-order traversal.
5. Experiment on disconnected graphs, note what gets missed.

#### Test Cases

| Graph          | Start | Order   | Distance            |
| -------------- | ----- | ------- | ------------------- |
| 0–1–2 chain    | 0     | 0 1 2   | [0,1,2]             |
| Triangle 0–1–2 | 0     | 0 1 2   | [0,1,1]             |
| Star 0→{1,2,3} | 0     | 0 1 2 3 | [0,1,1,1]           |
| Grid 2×2       | 0     | 0 1 2 3 | Layered             |
| Disconnected   | 0     | 0 1     | Only component of 0 |

#### Complexity

- Time: O(V + E)
- Space: O(V) for queue and visited set

BFS is your wavefront explorer, fair, systematic, and always shortest when edges are equal.

### 304 Iterative Deepening DFS

Iterative Deepening Depth-First Search (IDDFS) blends the depth control of BFS with the space efficiency of DFS. It repeatedly performs DFS with increasing depth limits, uncovering nodes level by level, but through deep-first exploration each time.

#### What Problem Are We Solving?

Pure DFS may wander too deep, missing nearer solutions.
Pure BFS finds shortest paths but consumes large memory.

We need a search that:

- Finds shallowest solution like BFS
- Uses O(depth) memory like DFS
- Works in infinite or very large search spaces

That's where IDDFS shines, it performs a DFS up to a limit, then restarts with a deeper limit, repeating until the goal is found.

Example:
Think of a diver who explores deeper with each dive, 1 meter, 2 meters, 3 meters, always sweeping from the surface down.

#### How Does It Work (Plain Language)?

Each iteration increases the depth limit by one.
At each stage, we perform a DFS that stops when depth exceeds the current limit.

1. Set limit = 0
2. Run DFS with depth limit = 0
3. If not found, increase limit and repeat
4. Continue until goal found or all explored

| Iteration | Depth Limit | Nodes Explored    | Found Goal? |
| --------- | ----------- | ----------------- | ----------- |
| 1         | 0           | Start node        | No          |
| 2         | 1           | Start + neighbors | No          |
| 3         | 2           | + deeper nodes    | Possibly    |
| …         | …           | …                 | …           |

Although nodes are revisited, total cost remains efficient, like BFS's layer-wise discovery.

#### Tiny Code (Easy Versions)

C (Depth-Limited DFS)

```c
#include <stdio.h>
#include <stdbool.h>

#define MAX 100
int graph[MAX][MAX];
int n;
bool found = false;

void dls(int v, int depth, int limit, bool visited[]) {
    visited[v] = true;
    printf("%d ", v);
    if (depth == limit) return;
    for (int u = 0; u < n; u++) {
        if (graph[v][u] && !visited[u]) {
            dls(u, depth + 1, limit, visited);
        }
    }
}

void iddfs(int start, int max_depth) {
    for (int limit = 0; limit <= max_depth; limit++) {
        bool visited[MAX] = {false};
        printf("\nDepth limit %d: ", limit);
        dls(start, 0, limit, visited);
    }
}

int main(void) {
    printf("Enter number of vertices: ");
    scanf("%d", &n);

    printf("Enter adjacency matrix:\n");
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            scanf("%d", &graph[i][j]);

    iddfs(0, 3);
}
```

Python (Adjacency List + Depth Limit)

```python
graph = {
    0: [1, 2],
    1: [3],
    2: [4],
    3: [],
    4: []
}

def dls(v, depth, limit, visited):
    visited.add(v)
    print(v, end=" ")
    if depth == limit:
        return
    for u in graph[v]:
        if u not in visited:
            dls(u, depth + 1, limit, visited)

def iddfs(start, max_depth):
    for limit in range(max_depth + 1):
        print(f"\nDepth limit {limit}:", end=" ")
        visited = set()
        dls(start, 0, limit, visited)

iddfs(0, 3)
```

#### Why It Matters

- Combines advantages of BFS and DFS
- Finds optimal solution in unweighted graphs
- Uses linear space
- Ideal for state-space search (AI, puzzles)

#### A Gentle Proof (Why It Works)

BFS guarantees shortest path; DFS uses less space.
IDDFS repeats DFS with increasing limits, ensuring that:

- All nodes at depth `d` are visited before depth `d+1`
- Space = O(d)
- Time ≈ O(b^d), similar to BFS in order

Redundant work (revisiting nodes) is small compared to total nodes in deeper layers.

#### Try It Yourself

1. Run IDDFS on a tree; observe repeated shallow visits.
2. Count nodes visited per iteration.
3. Compare total visits with BFS.
4. Modify depth limit mid-run, what happens?
5. Use IDDFS to find a goal node at depth 3.

#### Test Cases

| Graph          | Goal   | Max Depth | Found At | Order Example |
| -------------- | ------ | --------- | -------- | ------------- |
| 0→1→2→3        | 3      | 3         | Depth 3  | 0 1 2 3       |
| 0→{1,2}, 1→3   | 3      | 3         | Depth 2  | 0 1 3         |
| Star 0→{1,2,3} | 3      | 1         | Depth 1  | 0 1 2 3       |
| 0→1→2→Goal     | Goal=2 | 2         | Depth 2  | 0 1 2         |

#### Complexity

- Time: O(b^d) (like BFS)
- Space: O(d) (like DFS)

Iterative Deepening DFS is the patient climber, revisiting familiar ground, going deeper each time, ensuring no shallow treasure is missed.

### 305 Bidirectional BFS

Bidirectional BFS is the meet-in-the-middle version of BFS. Instead of starting from one end and exploring everything outward, we launch two BFS waves, one from the source and one from the target, and stop when they collide in the middle. It's like digging a tunnel from both sides of a mountain to meet halfway.

#### What Problem Are We Solving?

Standard BFS explores the entire search space outward from the start until it reaches the goal, great for small graphs, but expensive when the graph is huge.

Bidirectional BFS cuts that exploration dramatically by searching both directions at once, halving the effective search depth.

We want an algorithm that:

- Finds the shortest path in an unweighted graph
- Explores fewer nodes than single-source BFS
- Stops as soon as the two waves meet

Example:
You're finding the shortest route between two cities. Instead of exploring from one city across the whole map, you also send scouts from the destination. They meet somewhere, the midpoint of the shortest path.

#### How Does It Work (Plain Language)?

Run two BFS searches simultaneously, one forward, one backward.
At each step, expand the smaller frontier first to balance work.
Stop when any node appears in both visited sets.

1. Start BFS from `source` and `target`
2. Maintain two queues and two visited sets
3. Alternate expansions
4. When visited sets overlap, meeting point found
5. Combine paths for the final route

| Step | Forward Queue | Backward Queue | Intersection | Action             |
| ---- | ------------- | -------------- | ------------ | ------------------ |
| 1    | [S]           | [T]            | ∅            | Start              |
| 2    | [S1, S2]      | [T1, T2]       | ∅            | Expand both        |
| 3    | [S2, S3]      | [T1, S2]       | S2           | Found meeting node |

#### Tiny Code (Easy Versions)

C (Simplified Version)

```c
#include <stdio.h>
#include <stdbool.h>

#define MAX 100
int graph[MAX][MAX];
int n;

bool bfs_step(bool visited[], int queue[], int *front, int *rear) {
    int size = *rear - *front;
    while (size--) {
        int v = queue[(*front)++];
        for (int u = 0; u < n; u++) {
            if (graph[v][u] && !visited[u]) {
                visited[u] = true;
                queue[(*rear)++] = u;
            }
        }
    }
    return false;
}

bool intersect(bool a[], bool b[]) {
    for (int i = 0; i < n; i++)
        if (a[i] && b[i]) return true;
    return false;
}

bool bidir_bfs(int src, int dest) {
    bool vis_s[MAX] = {false}, vis_t[MAX] = {false};
    int qs[MAX], qt[MAX];
    int fs = 0, rs = 0, ft = 0, rt = 0;
    qs[rs++] = src; vis_s[src] = true;
    qt[rt++] = dest; vis_t[dest] = true;

    while (fs < rs && ft < rt) {
        if (bfs_step(vis_s, qs, &fs, &rs)) return true;
        if (intersect(vis_s, vis_t)) return true;

        if (bfs_step(vis_t, qt, &ft, &rt)) return true;
        if (intersect(vis_s, vis_t)) return true;
    }
    return false;
}

int main(void) {
    printf("Enter number of vertices: ");
    scanf("%d", &n);
    printf("Enter adjacency matrix:\n");
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            scanf("%d", &graph[i][j]);

    int src = 0, dest = n - 1;
    if (bidir_bfs(src, dest))
        printf("Path found\n");
    else
        printf("No path\n");
}
```

Python (Readable Version)

```python
from collections import deque

graph = {
    0: [1, 2],
    1: [3],
    2: [4],
    3: [5],
    4: [5],
    5: []
}

def bidirectional_bfs(src, dest):
    if src == dest:
        return True

    q1, q2 = deque([src]), deque([dest])
    visited1, visited2 = {src}, {dest}

    while q1 and q2:
        # Expand forward
        for _ in range(len(q1)):
            v = q1.popleft()
            for u in graph[v]:
                if u in visited2:
                    return True
                if u not in visited1:
                    visited1.add(u)
                    q1.append(u)

        # Expand backward
        for _ in range(len(q2)):
            v = q2.popleft()
            for u in graph[v]:
                if u in visited1:
                    return True
                if u not in visited2:
                    visited2.add(u)
                    q2.append(u)

    return False

print(bidirectional_bfs(0, 5))
```

#### Why It Matters

- Faster shortest-path search on large graphs
- Reduces explored nodes from O(b^d) to roughly O(b^(d/2))
- Excellent for pathfinding in maps, puzzles, or networks
- Demonstrates search symmetry and frontier balancing

#### A Gentle Proof (Why It Works)

If the shortest path length is `d`,
BFS explores O(b^d) nodes,
but Bidirectional BFS explores 2×O(b^(d/2)) nodes —
a huge savings since b^(d/2) ≪ b^d.

Each side guarantees the frontier grows level by level,
and intersection ensures meeting at the middle of the shortest path.

#### Try It Yourself

1. Trace bidirectional BFS on a 5-node chain (0→1→2→3→4).
2. Count nodes visited by single BFS vs bidirectional BFS.
3. Add print statements to see where the waves meet.
4. Modify to reconstruct the path.
5. Compare performance on branching graphs.

#### Test Cases

| Graph                   | Source | Target | Found? | Meeting Node |
| ----------------------- | ------ | ------ | ------ | ------------ |
| 0–1–2–3–4               | 0      | 4      | ✅      | 2            |
| 0→1, 1→2, 2→3           | 0      | 3      | ✅      | 1 or 2       |
| 0→1, 2→3 (disconnected) | 0      | 3      | ❌      | –            |
| Triangle 0–1–2–0        | 0      | 2      | ✅      | 0 or 2       |
| Star 0→{1,2,3,4}        | 1      | 2      | ✅      | 0            |

#### Complexity

- Time: O(b^(d/2))
- Space: O(b^(d/2))
- Optimality: Finds shortest path in unweighted graphs

Bidirectional BFS is the bridge builder, starting from both shores, racing toward the meeting point in the middle.

### 306 DFS on Grid

DFS on a grid is your go-to for exploring 2D maps, mazes, or islands. It works just like DFS on graphs, but here, each cell is a node and its up/down/left/right neighbors form the edges. Perfect for connected component detection, region labeling, or maze solving.

#### What Problem Are We Solving?

We want to explore or mark all connected cells in a grid, often used for:

- Counting islands in a binary matrix
- Flood-fill algorithms (coloring regions)
- Maze traversal (finding a path through walls)
- Connectivity detection in 2D maps

Example:
Think of a painter pouring ink into one cell, DFS shows how the ink spreads to fill the entire connected region.

#### How Does It Work (Plain Language)?

DFS starts from a given cell, visits it, and recursively explores all valid, unvisited neighbors.

We check 4 directions (or 8 if diagonals count).
Each neighbor is:

- Within bounds
- Not yet visited
- Satisfies the condition (e.g., same color, value = 1)

| Step | Current Cell               | Action     | Stack (Call Path)   |
| ---- | -------------------------- | ---------- | ------------------- |
| 1    | (0,0)                      | Visit      | [(0,0)]             |
| 2    | (0,1)                      | Move right | [(0,0),(0,1)]       |
| 3    | (1,1)                      | Move down  | [(0,0),(0,1),(1,1)] |
| 4    | (1,1) has no new neighbors | Backtrack  | [(0,0),(0,1)]       |
| 5    | Continue                   | …          |                     |

The traversal ends when all reachable cells are visited.

#### Tiny Code (Easy Versions)

C (DFS for Island Counting)

```c
#include <stdio.h>
#include <stdbool.h>

#define MAX 100
int grid[MAX][MAX];
bool visited[MAX][MAX];
int n, m;

int dx[4] = {-1, 1, 0, 0};
int dy[4] = {0, 0, -1, 1};

void dfs(int x, int y) {
    visited[x][y] = true;
    for (int k = 0; k < 4; k++) {
        int nx = x + dx[k];
        int ny = y + dy[k];
        if (nx >= 0 && nx < n && ny >= 0 && ny < m &&
            grid[nx][ny] == 1 && !visited[nx][ny]) {
            dfs(nx, ny);
        }
    }
}

int main(void) {
    printf("Enter grid size (n m): ");
    scanf("%d %d", &n, &m);
    printf("Enter grid (0/1):\n");
    for (int i = 0; i < n; i++)
        for (int j = 0; j < m; j++)
            scanf("%d", &grid[i][j]);

    int count = 0;
    for (int i = 0; i < n; i++)
        for (int j = 0; j < m; j++)
            if (grid[i][j] == 1 && !visited[i][j]) {
                dfs(i, j);
                count++;
            }

    printf("Number of islands: %d\n", count);
}
```

Python (Flood Fill)

```python
grid = [
    [1,1,0,0],
    [1,0,0,1],
    [0,0,1,1],
    [0,0,0,1]
]

n, m = len(grid), len(grid[0])
visited = [[False]*m for _ in range(n)]

def dfs(x, y):
    if x < 0 or x >= n or y < 0 or y >= m:
        return
    if grid[x][y] == 0 or visited[x][y]:
        return
    visited[x][y] = True
    for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
        dfs(x + dx, y + dy)

count = 0
for i in range(n):
    for j in range(m):
        if grid[i][j] == 1 and not visited[i][j]:
            dfs(i, j)
            count += 1

print("Number of islands:", count)
```

#### Why It Matters

- Core tool for grid exploration and region labeling
- Forms the heart of island problems, maze solvers, and map connectivity
- Demonstrates DFS behavior in real-world layouts
- Easy visualization and debugging on 2D arrays

#### A Gentle Proof (Why It Works)

Each cell is visited exactly once, marked `visited` upon entry.
Recursive calls spread to all valid neighbors.
So total time = proportional to number of cells and edges (neighbors).

If grid has size `n × m`, and each cell checks 4 neighbors:

- Time: O(n × m)
- Space: O(n × m) visited + recursion depth (≤ n × m)

DFS guarantees every reachable cell is visited exactly once, forming connected components.

#### Try It Yourself

1. Change movement to 8 directions (include diagonals).
2. Modify to flood-fill a color (e.g., replace all 1s with 2s).
3. Count components in a matrix of characters (‘X', ‘O').
4. Visualize traversal order in a printed grid.
5. Compare with BFS on the same grid.

#### Test Cases

| Grid                      | Expected | Description            |
| ------------------------- | -------- | ---------------------- |
| [[1,0,0],[0,1,0],[0,0,1]] | 3        | Diagonal not connected |
| [[1,1,0],[1,0,0],[0,0,1]] | 2        | Two clusters           |
| [[1,1,1],[1,1,1],[1,1,1]] | 1        | One big island         |
| [[0,0,0],[0,0,0]]         | 0        | No land                |
| [[1,0,1],[0,1,0],[1,0,1]] | 5        | Many singles           |

#### Complexity

- Time: O(n × m)
- Space: O(n × m) (visited) or O(depth) recursion stack

DFS on grid is your map explorer, sweeping through every reachable patch, one cell at a time.

### 307 BFS on Grid

BFS on a grid explores cells level by level, making it perfect for shortest paths in unweighted grids, minimum steps in mazes, and distance labeling from a source. Each cell is a node and edges connect to neighbors such as up, down, left, right.

#### What Problem Are We Solving?

We want to:

- Find the shortest path from a start cell to a goal cell when each move costs the same
- Compute a distance map from a source to all reachable cells
- Handle obstacles cleanly and avoid revisiting

Example:
Given a maze as a 0 or 1 grid, where 0 is free and 1 is wall, BFS finds the fewest moves from start to target.

#### How Does It Work (Plain Language)?

Use a queue. Start from the source, push it with distance 0, and expand in waves.
At each step, pop the front cell, try its neighbors, mark unseen neighbors visited, and record their distance as current distance + 1.

| Step | Queue (front to back) | Current Cell | Action          | Distance Updated           |
| ---- | --------------------- | ------------ | --------------- | -------------------------- |
| 1    | [(sx, sy)]            | (sx, sy)     | Start           | dist[sx][sy] = 0           |
| 2    | [(n1), (n2)]          | (n1)         | Visit neighbors | dist[n1] = 1               |
| 3    | [(n2), (n3), (n4)]    | (n2)         | Continue        | dist[n2] = 1               |
| 4    | [...]                 | ...          | Wave expands    | dist[next] = dist[cur] + 1 |

The first time you reach the goal, the recorded distance is minimal.

#### Tiny Code (Easy Versions)

C (Shortest Path on a 0 or 1 Grid)

```c
#include <stdio.h>
#include <stdbool.h>

#define MAX 200
int n, m;
int grid[MAX][MAX];         // 0 free, 1 wall
int distv[MAX][MAX];        // distance map
bool vis[MAX][MAX];

int qx[MAX*MAX], qy[MAX*MAX];
int front = 0, rear = 0;

int dx[4] = {-1, 1, 0, 0};
int dy[4] = {0, 0, -1, 1};

void enqueue(int x, int y) { qx[rear] = x; qy[rear] = y; rear++; }
void dequeue(int *x, int *y) { *x = qx[front]; *y = qy[front]; front++; }
bool empty() { return front == rear; }

int bfs(int sx, int sy, int tx, int ty) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < m; j++) {
            vis[i][j] = false;
            distv[i][j] = -1;
        }

    vis[sx][sy] = true;
    distv[sx][sy] = 0;
    enqueue(sx, sy);

    while (!empty()) {
        int x, y;
        dequeue(&x, &y);
        if (x == tx && y == ty) return distv[x][y];

        for (int k = 0; k < 4; k++) {
            int nx = x + dx[k], ny = y + dy[k];
            if (nx >= 0 && nx < n && ny >= 0 && ny < m &&
                !vis[nx][ny] && grid[nx][ny] == 0) {
                vis[nx][ny] = true;
                distv[nx][ny] = distv[x][y] + 1;
                enqueue(nx, ny);
            }
        }
    }
    return -1; // unreachable
}

int main(void) {
    printf("Enter n m: ");
    scanf("%d %d", &n, &m);
    printf("Enter grid (0 free, 1 wall):\n");
    for (int i = 0; i < n; i++)
        for (int j = 0; j < m; j++)
            scanf("%d", &grid[i][j]);

    int sx, sy, tx, ty;
    printf("Enter start sx sy and target tx ty: ");
    scanf("%d %d %d %d", &sx, &sy, &tx, &ty);

    int d = bfs(sx, sy, tx, ty);
    if (d >= 0) printf("Shortest distance: %d\n", d);
    else printf("No path\n");
}
```

Python (Distance Map and Path Reconstruction)

```python
from collections import deque

grid = [
    [0,0,0,1],
    [1,0,0,0],
    [0,0,1,0],
    [0,0,0,0]
]
n, m = len(grid), len(grid[0])

def bfs_grid(sx, sy, tx, ty):
    dist = [[-1]*m for _ in range(n)]
    parent = [[None]*m for _ in range(n)]
    q = deque()
    q.append((sx, sy))
    dist[sx][sy] = 0

    for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
        pass  # only to show directions exist

    while q:
        x, y = q.popleft()
        if (x, y) == (tx, ty):
            break
        for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < n and 0 <= ny < m and grid[nx][ny] == 0 and dist[nx][ny] == -1:
                dist[nx][ny] = dist[x][y] + 1
                parent[nx][ny] = (x, y)
                q.append((nx, ny))

    # Reconstruct path if reachable
    if dist[tx][ty] == -1:
        return dist, []
    path = []
    cur = (tx, ty)
    while cur:
        path.append(cur)
        cur = parent[cur[0]][cur[1]]
    path.reverse()
    return dist, path

dist, path = bfs_grid(0, 0, 3, 3)
print("Distance to target:", dist[3][3])
print("Path:", path)
```

#### Why It Matters

- Guarantees shortest path in unweighted grids
- Produces a full distance transform useful for many tasks
- Robust and simple for maze solvers and robotics navigation
- Natural stepping stone to 0 1 BFS and Dijkstra

#### A Gentle Proof (Why It Works)

BFS processes cells by nondecreasing distance from the source. When a cell is first dequeued, the stored distance equals the minimum number of moves needed to reach it. Each free neighbor is discovered with distance plus one. Therefore the first time the goal is reached, that distance is minimal.

- Each cell enters the queue at most once
- Each edge between neighboring cells is considered once

Hence total work is linear in the number of cells and neighbor checks.

#### Try It Yourself

1. Add 8 directional moves and compare paths with 4 directional moves.
2. Add teleporters by connecting listed cell pairs as edges.
3. Convert to multi source BFS by enqueuing several starts with distance 0.
4. Block some cells and verify that BFS never steps through walls.
5. Record parents and print the maze with the path marked.

#### Test Cases

| Grid                   | Start               | Target | Expected                       |
| ---------------------- | ------------------- | ------ | ------------------------------ |
| 2 x 2 all free         | (0,0)               | (1,1)  | Distance 2 via right then down |
| 3 x 3 with center wall | (0,0)               | (2,2)  | Distance 4 around the wall     |
| Line 1 x 5 all free    | (0,0)               | (0,4)  | Distance 4                     |
| Blocked target         | (0,0)               | (1,1)  | No path                        |
| Multi source wave      | {all corner starts} | center | Minimum among corners          |

#### Complexity

- Time: O(n × m)
- Space: O(n × m) for visited or distance map and queue

BFS on grid is the wavefront that sweeps a map evenly, giving you the fewest steps from start to goal with clean, level by level logic.

### 308 Multi-Source BFS

Multi-Source BFS is the wavefront BFS that starts not from one node but from many sources at once. It's perfect when several starting points all spread out simultaneously, like multiple fires burning through a forest, or signals radiating from several transmitters.

#### What Problem Are We Solving?

We need to find minimum distances from multiple starting nodes, not just one.
This is useful when:

- There are several sources of influence (e.g. infections, signals, fires)
- You want the nearest source for each node
- You need simultaneous propagation (e.g. multi-start shortest path)

Examples:

- Spread of rumors from multiple people
- Flooding time from multiple water sources
- Minimum distance to nearest hospital or supply center

#### How Does It Work (Plain Language)?

We treat all sources as level 0 and push them into the queue at once.
Then BFS proceeds normally, each node is assigned a distance equal to the shortest path from any source.

| Step | Queue (Front → Back)      | Action                 | Distance Updated |
| ---- | ------------------------- | ---------------------- | ---------------- |
| 1    | [S1, S2, S3]              | Initialize all sources | dist[S*] = 0     |
| 2    | [Neighbors of S1, S2, S3] | Wave expands           | dist = 1         |
| 3    | [Next Layer]              | Continue               | dist = 2         |
| …    | …                         | …                      | …                |

The first time a node is visited, we know it's from the nearest source.

#### Tiny Code (Easy Versions)

C (Multi-Source BFS on Grid)

```c
#include <stdio.h>
#include <stdbool.h>

#define MAX 100
int n, m;
int grid[MAX][MAX];
int distv[MAX][MAX];
bool vis[MAX][MAX];
int qx[MAX*MAX], qy[MAX*MAX];
int front = 0, rear = 0;

int dx[4] = {-1, 1, 0, 0};
int dy[4] = {0, 0, -1, 1};

void enqueue(int x, int y) { qx[rear] = x; qy[rear] = y; rear++; }
void dequeue(int *x, int *y) { *x = qx[front]; *y = qy[front]; front++; }
bool empty() { return front == rear; }

void multi_source_bfs() {
    while (!empty()) {
        int x, y;
        dequeue(&x, &y);
        for (int k = 0; k < 4; k++) {
            int nx = x + dx[k], ny = y + dy[k];
            if (nx >= 0 && nx < n && ny >= 0 && ny < m &&
                grid[nx][ny] == 0 && !vis[nx][ny]) {
                vis[nx][ny] = true;
                distv[nx][ny] = distv[x][y] + 1;
                enqueue(nx, ny);
            }
        }
    }
}

int main(void) {
    printf("Enter grid size n m: ");
    scanf("%d %d", &n, &m);
    printf("Enter grid (0 free, 1 blocked, 2 source):\n");
    for (int i = 0; i < n; i++)
        for (int j = 0; j < m; j++) {
            scanf("%d", &grid[i][j]);
            if (grid[i][j] == 2) {
                vis[i][j] = true;
                distv[i][j] = 0;
                enqueue(i, j);
            }
        }

    multi_source_bfs();

    printf("Distance map:\n");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++)
            printf("%2d ", distv[i][j]);
        printf("\n");
    }
}
```

Python (Simple Multi-Source BFS)

```python
from collections import deque

grid = [
    [2,0,1,0],
    [0,1,0,0],
    [0,0,0,2]
]
n, m = len(grid), len(grid[0])
dist = [[-1]*m for _ in range(n)]
q = deque()

for i in range(n):
    for j in range(m):
        if grid[i][j] == 2:  # source
            dist[i][j] = 0
            q.append((i, j))

dirs = [(-1,0),(1,0),(0,-1),(0,1)]
while q:
    x, y = q.popleft()
    for dx, dy in dirs:
        nx, ny = x+dx, y+dy
        if 0 <= nx < n and 0 <= ny < m and grid[nx][ny] == 0 and dist[nx][ny] == -1:
            dist[nx][ny] = dist[x][y] + 1
            q.append((nx, ny))

print("Distance Map:")
for row in dist:
    print(row)
```

#### Why It Matters

- Finds nearest source distance for all nodes in one pass
- Ideal for multi-origin diffusion problems
- Foundation for tasks like multi-fire spread, influence zones, Voronoi partitioning on graphs

#### A Gentle Proof (Why It Works)

Since all sources start at distance 0 and BFS expands in order of increasing distance, the first time a node is visited, it's reached by the shortest possible path from any source.

Each cell is enqueued exactly once → O(V + E) time.
No need to run BFS separately for each source.

#### Try It Yourself

1. Mark multiple sources (2s) on a grid, verify distances radiate outward.
2. Change obstacles (1s) and see how waves avoid them.
3. Count how many steps each free cell is from nearest source.
4. Modify to return which source id reached each cell first.
5. Compare total cost vs running single-source BFS repeatedly.

#### Test Cases

| Grid                              | Expected Output (Distances) | Description          |
| --------------------------------- | --------------------------- | -------------------- |
| [[2,0,2]]                         | [0,1,0]                     | Two sources on edges |
| [[2,0,0],[0,1,0],[0,0,2]]         | wave radiates from corners  | Mixed obstacles      |
| [[2,2,2]]                         | [0,0,0]                     | All sources          |
| [[0,0,0],[0,0,0]] + center source | center = 0, corners = 2     | Wave expanding       |
| All blocked                       | unchanged                   | No propagation       |

#### Complexity

- Time: O(V + E)
- Space: O(V) for queue and distance map

Multi-Source BFS is the chorus of wavefronts, expanding together, each note reaching its closest audience in perfect harmony.

### 309 Topological Sort (DFS-based)

Topological sort is the linear ordering of vertices in a Directed Acyclic Graph (DAG) such that for every directed edge ( u \to v ), vertex ( u ) appears before ( v ) in the order. The DFS-based approach discovers this order by exploring deeply and recording finishing times.

#### What Problem Are We Solving?

We want a way to order tasks that have dependencies.
Topological sort answers: *In what order can we perform tasks so that prerequisites come first?*

Typical use cases:

- Build systems (compile order)
- Course prerequisite scheduling
- Pipeline stage ordering
- Dependency resolution (e.g. package installs)

Example:
If task A must finish before B and C, and C before D, then one valid order is A → C → D → B.

#### How Does It Work (Plain Language)?

DFS explores from each unvisited node.
When a node finishes (no more outgoing edges to explore), push it onto a stack.
After all DFS calls, reverse the stack, that's your topological order.

1. Initialize all nodes as unvisited
2. For each node `v`:

   * Run DFS if not visited
   * After exploring all neighbors, push `v` to stack
3. Reverse stack to get topological order

| Step | Current Node | Action          | Stack     |
| ---- | ------------ | --------------- | --------- |
| 1    | A            | Visit neighbors | []        |
| 2    | B            | Visit           | []        |
| 3    | D            | Visit           | []        |
| 4    | D done       | Push D          | [D]       |
| 5    | B done       | Push B          | [D, B]    |
| 6    | A done       | Push A          | [D, B, A] |

Reverse: [A, B, D]

#### Tiny Code (Easy Versions)

C (DFS-based Topological Sort)

```c
#include <stdio.h>
#include <stdbool.h>

#define MAX 100
int graph[MAX][MAX];
bool visited[MAX];
int stack[MAX];
int top = -1;
int n;

void dfs(int v) {
    visited[v] = true;
    for (int u = 0; u < n; u++) {
        if (graph[v][u] && !visited[u]) {
            dfs(u);
        }
    }
    stack[++top] = v; // push after exploring neighbors
}

void topological_sort() {
    for (int i = 0; i < n; i++) {
        if (!visited[i]) dfs(i);
    }
    printf("Topological order: ");
    while (top >= 0) printf("%d ", stack[top--]);
    printf("\n");
}

int main(void) {
    printf("Enter number of vertices: ");
    scanf("%d", &n);
    printf("Enter adjacency matrix (DAG):\n");
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            scanf("%d", &graph[i][j]);
    topological_sort();
}
```

Python (DFS-based)

```python
graph = {
    0: [1, 2],
    1: [3],
    2: [3],
    3: []
}

visited = set()
stack = []

def dfs(v):
    visited.add(v)
    for u in graph[v]:
        if u not in visited:
            dfs(u)
    stack.append(v)

for v in graph:
    if v not in visited:
        dfs(v)

stack.reverse()
print("Topological order:", stack)
```

#### Why It Matters

- Ensures dependency order in DAGs
- Fundamental in compilers, schedulers, and build systems
- Basis for advanced algorithms:

  * Kahn's Algorithm (queue-based)
  * DAG shortest paths / DP
  * Critical path analysis

#### A Gentle Proof (Why It Works)

Each vertex is pushed onto the stack after all its descendants are explored.
So if there's an edge ( u \to v ), DFS ensures ( v ) finishes first and is pushed earlier, meaning ( u ) will appear later in the stack.
Reversing the stack thus guarantees ( u ) precedes ( v ).

No cycles allowed, if a back edge is found, topological sort is impossible.

#### Try It Yourself

1. Draw a DAG and label edges as prerequisites.
2. Run DFS and record finish times.
3. Push nodes on completion, reverse the order.
4. Add a cycle and see why it breaks.
5. Compare with Kahn's Algorithm results.

#### Test Cases

| Graph              | Edges      | Topological Order (Possible) |
| ------------------ | ---------- | ---------------------------- |
| A→B→C              | A→B, B→C   | A B C                        |
| A→C, B→C           | A→C, B→C   | A B C or B A C               |
| 0→1, 0→2, 1→3, 2→3 | DAG        | 0 2 1 3 or 0 1 2 3           |
| Chain 0→1→2→3      | Linear DAG | 0 1 2 3                      |
| Cycle 0→1→2→0      | Not DAG    | No valid order               |

#### Complexity

- Time: O(V + E) (each node and edge visited once)
- Space: O(V) (stack + visited array)

Topological sort (DFS-based) is your dependency detective, exploring deeply, marking completion, and leaving behind a perfect trail of prerequisites.

### 310 Topological Sort (Kahn's Algorithm)

Kahn's Algorithm is the queue-based way to perform topological sorting.
Instead of relying on recursion, it tracks in-degrees (how many edges point into each node) and repeatedly removes nodes with zero in-degree. It's clean, iterative, and naturally detects cycles.

#### What Problem Are We Solving?

We want a linear ordering of tasks in a Directed Acyclic Graph (DAG) such that each task appears after all its prerequisites.

Kahn's method is especially handy when:

- You want an iterative (non-recursive) algorithm
- You need to detect cycles automatically
- You're building a scheduler or compiler dependency resolver

Example:
If A must happen before B and C, and C before D, then valid orders: A C D B or A B C D.
Kahn's algorithm builds this order by peeling off "ready" nodes (those with no remaining prerequisites).

#### How Does It Work (Plain Language)?

Each node starts with an in-degree (count of incoming edges).
Nodes with in-degree = 0 are ready to process.

1. Compute in-degree for each node
2. Enqueue all nodes with in-degree = 0
3. While queue not empty:

   * Pop node `v` and add it to the topological order
   * For each neighbor `u` of `v`, decrement `in-degree[u]`
   * If `in-degree[u]` becomes 0, enqueue `u`
4. If all nodes processed → valid topological order
   Otherwise → cycle detected

| Step | Queue | Popped | Updated In-Degree | Order        |
| ---- | ----- | ------ | ----------------- | ------------ |
| 1    | [A]   | A      | B:0, C:1, D:2     | [A]          |
| 2    | [B]   | B      | C:0, D:2          | [A, B]       |
| 3    | [C]   | C      | D:1               | [A, B, C]    |
| 4    | [D]   | D      | –                 | [A, B, C, D] |

#### Tiny Code (Easy Versions)

C (Kahn's Algorithm)

```c
#include <stdio.h>
#include <stdbool.h>

#define MAX 100
int graph[MAX][MAX];
int indeg[MAX];
int queue[MAX];
int front = 0, rear = 0;
int n;

void enqueue(int v) { queue[rear++] = v; }
int dequeue() { return queue[front++]; }
bool empty() { return front == rear; }

void kahn_topo() {
    for (int v = 0; v < n; v++) {
        if (indeg[v] == 0) enqueue(v);
    }

    int count = 0;
    int order[MAX];

    while (!empty()) {
        int v = dequeue();
        order[count++] = v;
        for (int u = 0; u < n; u++) {
            if (graph[v][u]) {
                indeg[u]--;
                if (indeg[u] == 0) enqueue(u);
            }
        }
    }

    if (count != n) {
        printf("Graph has a cycle, topological sort not possible\n");
    } else {
        printf("Topological order: ");
        for (int i = 0; i < count; i++) printf("%d ", order[i]);
        printf("\n");
    }
}

int main(void) {
    printf("Enter number of vertices: ");
    scanf("%d", &n);

    printf("Enter adjacency matrix:\n");
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            scanf("%d", &graph[i][j]);
            if (graph[i][j]) indeg[j]++;
        }

    kahn_topo();
}
```

Python (Using Queue)

```python
from collections import deque

graph = {
    0: [1, 2],
    1: [3],
    2: [3],
    3: []
}

indeg = {v: 0 for v in graph}
for v in graph:
    for u in graph[v]:
        indeg[u] += 1

q = deque([v for v in graph if indeg[v] == 0])
order = []

while q:
    v = q.popleft()
    order.append(v)
    for u in graph[v]:
        indeg[u] -= 1
        if indeg[u] == 0:
            q.append(u)

if len(order) == len(graph):
    print("Topological order:", order)
else:
    print("Cycle detected, no valid order")
```

#### Why It Matters

- Fully iterative (no recursion stack)
- Naturally detects cycles
- Efficient for build systems, task planners, and dependency graphs
- Forms the base for Kahn's scheduling algorithm in DAG processing

#### A Gentle Proof (Why It Works)

Nodes with in-degree 0 have no prerequisites, they can appear first.
Once processed, they are removed (decrementing in-degree of successors).
This ensures:

- Each node is processed only after all its dependencies
- If a cycle exists, some nodes never reach in-degree 0 → detection built-in

Since each edge is considered once, runtime = O(V + E).

#### Try It Yourself

1. Build a small DAG manually and run the algorithm step by step.
2. Introduce a cycle (A→B→A) and observe detection.
3. Compare with DFS-based order, both valid.
4. Add priority to queue (min vertex first) to get lexicographically smallest order.
5. Apply to course prerequisite planner.

#### Test Cases

| Graph              | Edges    | Output             | Notes            |
| ------------------ | -------- | ------------------ | ---------------- |
| A→B→C              | A→B, B→C | A B C              | Linear chain     |
| A→C, B→C           | A→C, B→C | A B C or B A C     | Multiple sources |
| 0→1, 0→2, 1→3, 2→3 | DAG      | 0 1 2 3 or 0 2 1 3 | Multiple valid   |
| 0→1, 1→2, 2→0      | Cycle    | No order           | Cycle detected   |

#### Complexity

- Time: O(V + E)
- Space: O(V) (queue, indegrees)

Kahn's Algorithm is the dependency peeler, stripping away nodes layer by layer until a clean, linear order emerges.

## Section 32. Strongly Connected Components 

### 311 Kosaraju's Algorithm

Kosaraju's algorithm is one of the clearest ways to find strongly connected components (SCCs) in a directed graph. It uses two depth-first searches, one on the original graph, and one on its reversed version, to peel off SCCs layer by layer.

#### What Problem Are We Solving?

In a directed graph, a *strongly connected component* is a maximal set of vertices such that each vertex is reachable from every other vertex in the same set.

Kosaraju's algorithm groups the graph into these SCCs.

This is useful for:

- Condensing a graph into a DAG (meta-graph)
- Dependency analysis in compilers
- Finding cycles or redundant modules
- Graph simplification before DP or optimization

Example:
Imagine a one-way road network, SCCs are groups of cities where you can travel between any pair.

#### How Does It Work (Plain Language)?

Kosaraju's algorithm runs in two DFS passes:

1. First DFS (Original Graph):
   Explore all vertices. Each time a node finishes (recursion ends), record it on a stack (by finish time).

2. Reverse the Graph:
   Reverse all edges (flip direction).

3. Second DFS (Reversed Graph):
   Pop nodes from the stack (highest finish time first).
   Each DFS from an unvisited node forms one strongly connected component.

| Phase | Graph    | Action                     | Result            |
| ----- | -------- | -------------------------- | ----------------- |
| 1     | Original | DFS all, push finish order | Stack of vertices |
| 2     | Reversed | DFS by pop order           | Identify SCCs     |
| 3     | Output   | Each DFS tree              | SCC list          |

#### Tiny Code (Easy Versions)

C (Adjacency List)

```c
#include <stdio.h>
#include <stdbool.h>

#define MAX 100
int n;
int graph[MAX][MAX];
int rev[MAX][MAX];
bool visited[MAX];
int stack[MAX];
int top = -1;

void dfs1(int v) {
    visited[v] = true;
    for (int u = 0; u < n; u++) {
        if (graph[v][u] && !visited[u]) dfs1(u);
    }
    stack[++top] = v; // push after finishing
}

void dfs2(int v) {
    printf("%d ", v);
    visited[v] = true;
    for (int u = 0; u < n; u++) {
        if (rev[v][u] && !visited[u]) dfs2(u);
    }
}

int main(void) {
    printf("Enter number of vertices: ");
    scanf("%d", &n);

    printf("Enter adjacency matrix (directed):\n");
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            scanf("%d", &graph[i][j]);
            rev[j][i] = graph[i][j]; // build reverse
        }

    // Step 1: first DFS
    for (int i = 0; i < n; i++)
        visited[i] = false;

    for (int i = 0; i < n; i++)
        if (!visited[i]) dfs1(i);

    // Step 2: second DFS on reversed
    for (int i = 0; i < n; i++)
        visited[i] = false;

    printf("Strongly Connected Components:\n");
    while (top >= 0) {
        int v = stack[top--];
        if (!visited[v]) {
            dfs2(v);
            printf("\n");
        }
    }
}
```

Python (Using Lists)

```python
from collections import defaultdict

graph = defaultdict(list)
rev = defaultdict(list)

edges = [(0,1),(1,2),(2,0),(1,3)]
for u, v in edges:
    graph[u].append(v)
    rev[v].append(u)

visited = set()
stack = []

def dfs1(v):
    visited.add(v)
    for u in graph[v]:
        if u not in visited:
            dfs1(u)
    stack.append(v)

for v in graph:
    if v not in visited:
        dfs1(v)

visited.clear()

def dfs2(v, comp):
    visited.add(v)
    comp.append(v)
    for u in rev[v]:
        if u not in visited:
            dfs2(u, comp)

print("Strongly Connected Components:")
while stack:
    v = stack.pop()
    if v not in visited:
        comp = []
        dfs2(v, comp)
        print(comp)
```

#### Why It Matters

- Splits a directed graph into mutually reachable groups
- Used in condensation (convert cyclic graph → DAG)
- Helps detect cyclic dependencies
- Foundation for component-level optimization

#### A Gentle Proof (Why It Works)

1. Finishing times from the first DFS ensure we process "sinks" first (post-order).
2. Reversing edges turns sinks into sources.
3. The second DFS finds all nodes reachable from that source, i.e. an SCC.
4. Every node is assigned to exactly one SCC.

Correctness follows from properties of finishing times and reachability symmetry.

#### Try It Yourself

1. Draw a directed graph with cycles and run the steps manually.
2. Track finish order stack.
3. Reverse all edges and start popping nodes.
4. Each DFS tree = one SCC.
5. Compare results with Tarjan's algorithm.

#### Test Cases

| Graph                   | Edges        | SCCs          |
| ----------------------- | ------------ | ------------- |
| 0→1, 1→2, 2→0           | Cycle        | {0,1,2}       |
| 0→1, 1→2                | Chain        | {0}, {1}, {2} |
| 0→1, 1→2, 2→0, 2→3      | Cycle + tail | {0,1,2}, {3}  |
| 0→1, 1→0, 1→2, 2→3, 3→2 | Two SCCs     | {0,1}, {2,3}  |

#### Complexity

- Time: O(V + E) (2 DFS passes)
- Space: O(V + E) (graph + stack)

Kosaraju's Algorithm is your mirror explorer, traverse once to record the story, flip the graph, then replay it backward to reveal every strongly bound group.

### 312 Tarjan's Algorithm

Tarjan's algorithm finds all strongly connected components (SCCs) in a directed graph in one DFS pass, without reversing the graph. It's an elegant and efficient method that tracks each node's discovery time and lowest reachable ancestor to identify SCC roots.

#### What Problem Are We Solving?

We need to group vertices of a directed graph into SCCs, where each node is reachable from every other node in the same group.
Unlike Kosaraju's two-pass method, Tarjan's algorithm finds all SCCs in a single DFS, making it faster in practice and easy to integrate into larger graph algorithms.

Common applications:

- Cycle detection in directed graphs
- Component condensation for DAG processing
- Deadlock analysis
- Strong connectivity queries in compilers, networks, and systems

Example:
Imagine a group of cities connected by one-way roads. SCCs are clusters of cities that can all reach each other, forming a tightly connected region.

#### How Does It Work (Plain Language)?

Each vertex gets:

- A discovery time (disc), when it's first visited
- A low-link value (low), the smallest discovery time reachable (including back edges)

A stack keeps track of the active recursion path (current DFS stack).
When a vertex's `disc` equals its `low`, it's the root of an SCC, pop nodes from the stack until this vertex reappears.

| Step | Action                           | Stack      | SCC Found |                           |
| ---- | -------------------------------- | ---------- | --------- | ------------------------- |
| 1    | Visit node, assign disc & low    | [A]        | –         |                           |
| 2    | Go deeper (DFS neighbors)        | [A, B, C]  | –         |                           |
| 3    | Reach node with no new neighbors | Update low | [A, B, C] | –                         |
| 4    | Backtrack, compare lows          | [A, B]     | SCC {C}   |                           |
| 5    | When disc == low                 | Pop SCC    | [A]       | SCC {B, C} (if connected) |

#### Tiny Code (Easy Versions)

C (Tarjan's Algorithm)

```c
#include <stdio.h>
#include <stdbool.h>

#define MAX 100
int n;
int graph[MAX][MAX];
int disc[MAX], low[MAX], stack[MAX];
bool inStack[MAX];
int time_counter = 0, top = -1;

void dfs_tarjan(int v) {
    disc[v] = low[v] = ++time_counter;
    stack[++top] = v;
    inStack[v] = true;

    for (int u = 0; u < n; u++) {
        if (!graph[v][u]) continue;
        if (disc[u] == 0) {
            dfs_tarjan(u);
            if (low[u] < low[v]) low[v] = low[u];
        } else if (inStack[u]) {
            if (disc[u] < low[v]) low[v] = disc[u];
        }
    }

    if (disc[v] == low[v]) {
        printf("SCC: ");
        int w;
        do {
            w = stack[top--];
            inStack[w] = false;
            printf("%d ", w);
        } while (w != v);
        printf("\n");
    }
}

int main(void) {
    printf("Enter number of vertices: ");
    scanf("%d", &n);
    printf("Enter adjacency matrix (directed):\n");
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            scanf("%d", &graph[i][j]);

    for (int i = 0; i < n; i++) {
        disc[i] = 0;
        inStack[i] = false;
    }

    printf("Strongly Connected Components:\n");
    for (int i = 0; i < n; i++)
        if (disc[i] == 0)
            dfs_tarjan(i);
}
```

Python (Using Adjacency List)

```python
from collections import defaultdict

graph = defaultdict(list)
edges = [(0,1),(1,2),(2,0),(1,3)]
for u,v in edges:
    graph[u].append(v)

time = 0
disc = {}
low = {}
stack = []
in_stack = set()
sccs = []

def dfs(v):
    global time
    time += 1
    disc[v] = low[v] = time
    stack.append(v)
    in_stack.add(v)

    for u in graph[v]:
        if u not in disc:
            dfs(u)
            low[v] = min(low[v], low[u])
        elif u in in_stack:
            low[v] = min(low[v], disc[u])

    if disc[v] == low[v]:
        scc = []
        while True:
            w = stack.pop()
            in_stack.remove(w)
            scc.append(w)
            if w == v:
                break
        sccs.append(scc)

for v in list(graph.keys()):
    if v not in disc:
        dfs(v)

print("Strongly Connected Components:", sccs)
```

#### Why It Matters

- Runs in one DFS, efficient and elegant
- Detects SCCs on the fly (no reversing edges)
- Useful for online algorithms (process SCCs as discovered)
- Powers cycle detection, condensation graphs, component-based optimization

#### A Gentle Proof (Why It Works)

- `disc[v]`: time when `v` is first visited
- `low[v]`: smallest discovery time reachable via descendants or back edges
- When `disc[v] == low[v]`, `v` is the root of its SCC (no back edges go above it)
- Popping from the stack gives all nodes reachable within the component

Each edge examined once → linear time.

#### Try It Yourself

1. Run Tarjan on a graph with multiple cycles.
2. Observe `disc` and `low` values.
3. Print stack content at each step to see grouping.
4. Compare SCC output with Kosaraju's result.
5. Try adding a cycle and check grouping changes.

#### Test Cases

| Graph              | Edges        | SCCs          |
| ------------------ | ------------ | ------------- |
| 0→1, 1→2, 2→0      | Cycle        | {0,1,2}       |
| 0→1, 1→2           | Chain        | {2}, {1}, {0} |
| 0→1, 1→2, 2→0, 2→3 | Cycle + tail | {0,1,2}, {3}  |
| 1→2, 2→3, 3→1, 3→4 | Two groups   | {1,2,3}, {4}  |

#### Complexity

- Time: O(V + E) (one DFS pass)
- Space: O(V) (stack, arrays)

Tarjan's Algorithm is your clockwork explorer, tagging each vertex by time, tracing the deepest paths, and snapping off every strongly connected cluster in one graceful pass.

### 313 Gabow's Algorithm

Gabow's algorithm is another elegant one-pass method for finding strongly connected components (SCCs). It's less well-known than Tarjan's, but equally efficient, using two stacks to track active vertices and roots. It's a perfect example of "stack discipline" in graph exploration.

#### What Problem Are We Solving?

We want to find all strongly connected components in a directed graph, subsets where every node can reach every other.

Gabow's algorithm, like Tarjan's, works in a single DFS traversal, but instead of computing `low-link` values, it uses two stacks to manage component discovery and boundaries.

This approach is especially helpful in streaming, online, or iterative DFS environments, where explicit low-link computations can get messy.

Applications:

- Cycle decomposition
- Program dependency analysis
- Component condensation (DAG creation)
- Strong connectivity testing

Example:
Think of traversing a web of roads. One stack tracks where you've been, another stack marks "checkpoints" where loops close, each loop is a component.

#### How Does It Work (Plain Language)?

Gabow's algorithm keeps track of discovery order and component boundaries using two stacks:

- S (main stack): stores all currently active nodes
- P (boundary stack): stores potential roots of SCCs

Steps:

1. Perform DFS. Assign each node an increasing index (`preorder`)
2. Push node onto both stacks (S and P)
3. For each edge ( v \to u ):

   * If `u` unvisited → recurse
   * If `u` on stack S → adjust boundary stack P
4. After exploring neighbors:

   * If the top of P is current node `v`:

     * Pop P
     * Pop from S until `v`
     * Those popped form an SCC

| Step | Stack S   | Stack P   | Action        |
| ---- | --------- | --------- | ------------- |
| 1    | [A]       | [A]       | Visit A       |
| 2    | [A, B]    | [A, B]    | Visit B       |
| 3    | [A, B, C] | [A, B, C] | Visit C       |
| 4    | [A, B, C] | [A, B]    | Back edge C→B |
| 5    | [A, B]    | [A]       | Pop SCC {C}   |
| 6    | [A]       | []        | Pop SCC {B}   |
| 7    | []        | []        | Pop SCC {A}   |

#### Tiny Code (Easy Versions)

C (Gabow's Algorithm)

```c
#include <stdio.h>
#include <stdbool.h>

#define MAX 100
int graph[MAX][MAX];
int n;
int index_counter = 0;
int preorder[MAX];
bool onStack[MAX];
int stackS[MAX], topS = -1;
int stackP[MAX], topP = -1;

void dfs_gabow(int v) {
    preorder[v] = ++index_counter;
    stackS[++topS] = v;
    stackP[++topP] = v;
    onStack[v] = true;

    for (int u = 0; u < n; u++) {
        if (!graph[v][u]) continue;
        if (preorder[u] == 0) {
            dfs_gabow(u);
        } else if (onStack[u]) {
            while (preorder[stackP[topP]] > preorder[u]) topP--;
        }
    }

    if (stackP[topP] == v) {
        topP--;
        printf("SCC: ");
        int w;
        do {
            w = stackS[topS--];
            onStack[w] = false;
            printf("%d ", w);
        } while (w != v);
        printf("\n");
    }
}

int main(void) {
    printf("Enter number of vertices: ");
    scanf("%d", &n);
    printf("Enter adjacency matrix (directed):\n");
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            scanf("%d", &graph[i][j]);

    for (int i = 0; i < n; i++) {
        preorder[i] = 0;
        onStack[i] = false;
    }

    printf("Strongly Connected Components:\n");
    for (int i = 0; i < n; i++)
        if (preorder[i] == 0)
            dfs_gabow(i);
}
```

Python (Readable Implementation)

```python
from collections import defaultdict

graph = defaultdict(list)
edges = [(0,1),(1,2),(2,0),(1,3)]
for u,v in edges:
    graph[u].append(v)

index_counter = 0
preorder = {}
on_stack = set()
stackS, stackP = [], []
sccs = []

def dfs(v):
    global index_counter
    index_counter += 1
    preorder[v] = index_counter
    stackS.append(v)
    stackP.append(v)
    on_stack.add(v)

    for u in graph[v]:
        if u not in preorder:
            dfs(u)
        elif u in on_stack:
            while preorder[stackP[-1]] > preorder[u]:
                stackP.pop()

    if stackP and stackP[-1] == v:
        stackP.pop()
        comp = []
        while True:
            w = stackS.pop()
            on_stack.remove(w)
            comp.append(w)
            if w == v:
                break
        sccs.append(comp)

for v in graph:
    if v not in preorder:
        dfs(v)

print("Strongly Connected Components:", sccs)
```

#### Why It Matters

- Single DFS pass, no reverse graph
- Purely stack-based, avoids recursion depth issues in some variants
- Efficient and practical for large graphs
- Simpler low-link logic than Tarjan's in some applications

#### A Gentle Proof (Why It Works)

Each node gets a preorder index.
The second stack P tracks the earliest root reachable.
Whenever the top of P equals the current node,
all nodes above it in S form one SCC.

Invariant:

- S contains active nodes
- P contains possible roots (ordered by discovery)
- When back edges discovered, P trimmed to smallest reachable ancestor

This ensures each SCC is identified exactly once when its root finishes.

#### Try It Yourself

1. Draw a graph and trace preorder indices.
2. Observe how stack P shrinks on back edges.
3. Record SCCs as they're popped.
4. Compare output to Tarjan's algorithm.
5. Try it on DAGs, cycles, and mixed graphs.

#### Test Cases

| Graph              | Edges           | SCCs         |
| ------------------ | --------------- | ------------ |
| 0→1, 1→2, 2→0      | {0,1,2}         | One big SCC  |
| 0→1, 1→2           | {2}, {1}, {0}   | Chain        |
| 0→1, 1→0, 2→3      | {0,1}, {2}, {3} | Mixed        |
| 0→1, 1→2, 2→3, 3→1 | {1,2,3}, {0}    | Nested cycle |

#### Complexity

- Time: O(V + E)
- Space: O(V) (two stacks + metadata)

Gabow's Algorithm is your two-stack sculptor, carving SCCs from the graph in one graceful sweep, balancing exploration and boundaries like a craftsman marking edges before the final cut.

### 314 SCC DAG Construction

Once we've found strongly connected components (SCCs), we can build the condensation graph, a Directed Acyclic Graph (DAG) where each node represents an SCC, and edges connect them if any vertex in one SCC points to a vertex in another.

This structure is crucial because it transforms a messy cyclic graph into a clean acyclic skeleton, perfect for topological sorting, dynamic programming, and dependency analysis.

#### What Problem Are We Solving?

We want to take a directed graph and reduce it into a simpler form by collapsing each SCC into a single node.
The resulting graph (called the *condensation graph*) is always a DAG.

Why do this?

- To simplify reasoning about complex systems
- To run DAG algorithms on cyclic graphs (by condensing cycles)
- To perform component-level optimization
- To study dependencies between strongly connected subsystems

Example:
Think of a city map where SCCs are tightly connected neighborhoods. The DAG shows how traffic flows between neighborhoods, not within them.

#### How Does It Work (Plain Language)?

1. Find SCCs (using Kosaraju, Tarjan, or Gabow).
2. Assign each node an SCC ID (e.g., `comp[v] = c_id`).
3. Create a new graph with one node per SCC.
4. For each edge ( (u, v) ) in the original graph:

   * If `comp[u] != comp[v]`, add an edge from `comp[u]` to `comp[v]`
5. Remove duplicates (or store edges in sets).

Now the new graph has no cycles, since cycles are already condensed inside SCCs.

| Original Graph     | SCCs          | DAG       |
| ------------------ | ------------- | --------- |
| 0→1, 1→2, 2→0, 2→3 | {0,1,2}, {3}  | C0 → C1   |
| 0→1, 1→2, 2→3, 3→1 | {1,2,3}, {0}  | C0 → C1   |
| 0→1, 1→2           | {0}, {1}, {2} | 0 → 1 → 2 |

#### Tiny Code (Easy Versions)

C (Using Tarjan's Output)

```c
#include <stdio.h>
#include <stdbool.h>

#define MAX 100
int n;
int graph[MAX][MAX];
int comp[MAX];   // component ID
int comp_count = 0;

// Suppose comp[] already filled by Tarjan or Kosaraju

void build_condensed_graph() {
    int new_graph[MAX][MAX] = {0};

    for (int u = 0; u < n; u++) {
        for (int v = 0; v < n; v++) {
            if (graph[u][v] && comp[u] != comp[v]) {
                new_graph[comp[u]][comp[v]] = 1;
            }
        }
    }

    printf("Condensation Graph (Adjacency Matrix):\n");
    for (int i = 0; i < comp_count; i++) {
        for (int j = 0; j < comp_count; j++) {
            printf("%d ", new_graph[i][j]);
        }
        printf("\n");
    }
}
```

*(Assumes `comp[v]` and `comp_count` were computed before.)*

Python (with Tarjan's or Kosaraju's Components)

```python
from collections import defaultdict

# Suppose we already have SCCs
sccs = [[0,1,2], [3], [4,5]]
graph = {
    0: [1],
    1: [2,3],
    2: [0],
    3: [4],
    4: [5],
    5: []
}

# Step 1: assign component ID
comp_id = {}
for i, comp in enumerate(sccs):
    for v in comp:
        comp_id[v] = i

# Step 2: build DAG
dag = defaultdict(set)
for u in graph:
    for v in graph[u]:
        if comp_id[u] != comp_id[v]:
            dag[comp_id[u]].add(comp_id[v])

print("Condensation Graph (as DAG):")
for c in dag:
    print(c, "->", sorted(dag[c]))
```

#### Why It Matters

- Turns cyclic graph → acyclic graph
- Enables topological sorting, dynamic programming, path counting
- Clarifies inter-component dependencies
- Used in compiler analysis, SCC-based optimizations, graph condensation

#### A Gentle Proof (Why It Works)

Inside each SCC, every vertex can reach every other.
When edges cross SCC boundaries, they go in one direction only (since returning would merge the components).
Thus, the condensation graph cannot contain cycles, proving it's a DAG.

Each edge in the DAG represents at least one edge between components in the original graph.

#### Try It Yourself

1. Run Tarjan's algorithm to get `comp[v]` for each vertex.
2. Build DAG edges using component IDs.
3. Visualize original vs condensed graphs.
4. Topologically sort the DAG.
5. Use DP on DAG to compute longest path or reachability.

#### Test Cases

| Original Graph          | SCCs               | Condensed Edges        |
| ----------------------- | ------------------ | ---------------------- |
| 0→1, 1→2, 2→0, 2→3      | {0,1,2}, {3}       | 0→1                    |
| 0→1, 1→0, 1→2, 2→3, 3→2 | {0,1}, {2,3}       | 0→1                    |
| 0→1, 1→2, 2→3           | {0}, {1}, {2}, {3} | 0→1→2→3                |
| 0→1, 1→2, 2→0           | {0,1,2}            | None (single node DAG) |

#### Complexity

- Time: O(V + E) (using precomputed SCCs)
- Space: O(V + E) (new DAG structure)

SCC DAG Construction is your mapmaker's step, compressing tangled roads into clean highways, where each city (SCC) is a hub, and the new map is finally acyclic, ready for analysis.

### 315 SCC Online Merge

SCC Online Merge is a dynamic approach to maintain strongly connected components when a graph is growing over time (new edges are added). Instead of recomputing SCCs from scratch after each update, we *incrementally merge* components as they become connected.

It's the foundation of dynamic graph algorithms where edges arrive one by one, useful in online systems, incremental compilers, and evolving dependency graphs.

#### What Problem Are We Solving?

We want to maintain SCC structure as we add edges to a directed graph.

In static algorithms like Tarjan or Kosaraju, SCCs are computed once. But if new edges appear over time, recomputing everything is too slow.

SCC Online Merge gives us:

- Efficient incremental updates (no full recompute)
- Fast component merging
- Up-to-date condensation DAG

Typical use cases:

- Incremental program analysis (new dependencies)
- Dynamic network reachability
- Streaming graph processing
- Online algorithm design

#### How Does It Work (Plain Language)?

We start with each node as its own SCC.
When a new edge ( u \to v ) is added:

1. Check if ( u ) and ( v ) are already in the same SCC, if yes, nothing changes.
2. If not, check whether v's SCC can reach u's SCC (cycle detection).
3. If reachable, merge both SCCs into one.
4. Otherwise, add a DAG edge from `SCC(u)` → `SCC(v)`.

We maintain:

- Union-Find / DSU structure for SCC groups
- Reachability or DAG edges between SCCs
- Optional topological order for fast cycle checks

| Edge Added | Action       | New SCCs      | Notes    |
| ---------- | ------------ | ------------- | -------- |
| 0→1        | Create edge  | {0}, {1}      | Separate |
| 1→2        | Create edge  | {0}, {1}, {2} | Separate |
| 2→0        | Cycle formed | Merge {0,1,2} | New SCC  |
| 3→1        | Add edge     | {3}, {0,1,2}  | No merge |

#### Tiny Code (Conceptual Demo)

Python (Simplified DSU + DAG Check)

```python
class DSU:
    def __init__(self, n):
        self.parent = list(range(n))
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra

class OnlineSCC:
    def __init__(self, n):
        self.n = n
        self.dsu = DSU(n)
        self.graph = [set() for _ in range(n)]

    def add_edge(self, u, v):
        su, sv = self.dsu.find(u), self.dsu.find(v)
        if su == sv:
            return  # already connected
        # check if v's component reaches u's component
        if self._reachable(sv, su):
            # merge components
            self.dsu.union(su, sv)
            merged = self.dsu.find(su)
            self.graph[merged] = self.graph[su] | self.graph[sv]
        else:
            self.graph[su].add(sv)

    def _reachable(self, start, target, seen=None):
        if seen is None: seen = set()
        if start == target: return True
        seen.add(start)
        for nxt in self.graph[start]:
            if nxt not in seen and self._reachable(nxt, target, seen):
                return True
        return False

    def components(self):
        groups = {}
        for v in range(self.n):
            root = self.dsu.find(v)
            groups.setdefault(root, []).append(v)
        return list(groups.values())

# Example
scc = OnlineSCC(4)
scc.add_edge(0, 1)
scc.add_edge(1, 2)
print("SCCs:", scc.components())  # [[0], [1], [2], [3]]
scc.add_edge(2, 0)
print("SCCs:", scc.components())  # [[0,1,2], [3]]
```

#### Why It Matters

- Dynamic SCC maintenance without recomputation
- Handles edge insertions in O(V + E) amortized
- Enables real-time graph updates
- Basis for more advanced algorithms (fully dynamic SCC with deletions)

#### A Gentle Proof (Why It Works)

Every time we add an edge, one of two things happens:

- It connects existing SCCs without creating a cycle → add DAG edge
- It creates a cycle → merge involved SCCs

Since merging SCCs preserves the DAG structure (merging collapses cycles), the algorithm keeps the condensation graph valid at all times.

By maintaining reachability between SCCs, we can detect cycle formation efficiently.

#### Try It Yourself

1. Start with 5 nodes, no edges.
2. Add edges step-by-step, printing SCCs.
3. Add a back-edge forming a cycle → watch SCCs merge.
4. Visualize condensation DAG after each update.
5. Compare with recomputing using Tarjan's, they match!

#### Test Cases

| Step | Edge | SCCs               |
| ---- | ---- | ------------------ |
| 1    | 0→1  | {0}, {1}, {2}, {3} |
| 2    | 1→2  | {0}, {1}, {2}, {3} |
| 3    | 2→0  | {0,1,2}, {3}       |
| 4    | 3→1  | {0,1,2}, {3}       |
| 5    | 2→3  | {0,1,2,3}          |

#### Complexity

- Time: O(V + E) amortized (per edge addition)
- Space: O(V + E) (graph + DSU)

SCC Online Merge is your dynamic sculptor, merging components as new edges appear, maintaining structure without ever starting over.

### 316 Component Label Propagation

Component Label Propagation is a simple, iterative algorithm to find connected components (or strongly connected components in symmetric graphs) by repeatedly propagating minimum labels across edges until all nodes in a component share the same label.

It's conceptually clean, highly parallelizable, and forms the backbone of graph processing frameworks like Google's Pregel, Apache Giraph, and GraphX, perfect for large-scale or distributed systems.

#### What Problem Are We Solving?

We want to identify components, groups of vertices that are mutually reachable.
Instead of deep recursion or complex stacks, we iteratively propagate labels across the graph until convergence.

This approach is ideal when:

- The graph is massive (too large for recursion)
- You're using parallel / distributed computation
- You want a message-passing style algorithm

For undirected graphs, it finds connected components.
For directed graphs, it can approximate SCCs (often used as a preprocessing step).

Example:
Think of spreading an ID through a crowd, each node tells its neighbors its smallest known label, and everyone updates to match their smallest neighbor's label. Eventually, all in a group share the same number.

#### How Does It Work (Plain Language)?

1. Initialize: Each vertex's label = its own ID.
2. Iterate: For each vertex:

   * Look at all neighbors' labels.
   * Update to the smallest label seen.
3. Repeat until no label changes (convergence).

All nodes that end up sharing a label belong to the same component.

| Step | Node | Current Label | Neighbor Labels | New Label        |
| ---- | ---- | ------------- | --------------- | ---------------- |
| 1    | A    | A             | {B, C}          | min(A, B, C) = A |
| 2    | B    | B             | {A}             | min(B, A) = A    |
| 3    | C    | C             | {A}             | min(C, A) = A    |

Eventually, A, B, C → all labeled A.

#### Tiny Code (Easy Versions)

C (Iterative Label Propagation)

```c
#include <stdio.h>
#include <stdbool.h>

#define MAX 100
int n, graph[MAX][MAX];
int label[MAX];

void label_propagation() {
    bool changed = true;
    while (changed) {
        changed = false;
        for (int v = 0; v < n; v++) {
            int min_label = label[v];
            for (int u = 0; u < n; u++) {
                if (graph[v][u]) {
                    if (label[u] < min_label)
                        min_label = label[u];
                }
            }
            if (min_label < label[v]) {
                label[v] = min_label;
                changed = true;
            }
        }
    }
}

int main(void) {
    printf("Enter number of vertices: ");
    scanf("%d", &n);
    printf("Enter adjacency matrix (undirected):\n");
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            scanf("%d", &graph[i][j]);

    for (int i = 0; i < n; i++)
        label[i] = i;

    label_propagation();

    printf("Component Labels:\n");
    for (int i = 0; i < n; i++)
        printf("Vertex %d → Label %d\n", i, label[i]);
}
```

Python (Simple Version)

```python
graph = {
    0: [1],
    1: [0, 2],
    2: [1],
    3: [4],
    4: [3]
}

labels = {v: v for v in graph}

changed = True
while changed:
    changed = False
    for v in graph:
        min_label = min([labels[v]] + [labels[u] for u in graph[v]])
        if min_label < labels[v]:
            labels[v] = min_label
            changed = True

print("Component labels:", labels)
```

#### Why It Matters

- Simple and parallelizable, ideal for big data systems
- No recursion or stack, suitable for GPUs, clusters
- Local computation, fits the "think like a vertex" model
- Works on massive graphs where DFS is impractical

#### A Gentle Proof (Why It Works)

Each iteration allows label information to flow along edges.
Since the smallest label always propagates, and each propagation only decreases label values, the process must converge (no infinite updates).
At convergence, all vertices in a connected component share the same minimal label.

The number of iterations ≤ graph diameter.

#### Try It Yourself

1. Run it on small undirected graphs, label flow is easy to track.
2. Try a graph with two disconnected parts, they'll stabilize separately.
3. Add edges between components and rerun, watch labels merge.
4. Use directed edges and see how approximation differs from SCCs.
5. Implement in parallel (multi-threaded loop).

#### Test Cases

| Graph         | Edges        | Final Labels |
| ------------- | ------------ | ------------ |
| 0–1–2         | (0,1), (1,2) | {0,0,0}      |
| 0–1, 2–3      | (0,1), (2,3) | {0,0,2,2}    |
| 0–1–2–3       | Chain        | {0,0,0,0}    |
| 0–1, 1–2, 2–0 | Cycle        | {0,0,0}      |

#### Complexity

- Time: O(V × E) (in worst case, or O(D × E) for D = diameter)
- Space: O(V + E)

Component Label Propagation is your whispering algorithm, every node shares its name with neighbors, again and again, until all who can reach each other call themselves by the same name.

### 317 Path-Based SCC

The Path-Based SCC algorithm is another elegant one-pass method for finding strongly connected components in a directed graph.
It's similar in spirit to Tarjan's algorithm, but instead of computing explicit `low-link` values, it maintains path stacks to detect when a full component has been traversed.

Developed by Donald B. Johnson, it uses two stacks to keep track of DFS path order and potential roots. When a vertex cannot reach any earlier vertex, it becomes the root of an SCC, and the algorithm pops all nodes in that SCC from the path.

#### What Problem Are We Solving?

We want to find SCCs in a directed graph, subsets of vertices where each node can reach every other node.
Path-Based SCC offers a conceptually simple and efficient way to do this without `low-link` math.

Why it's useful:

- Single DFS traversal
- Clean stack-based logic
- Great for teaching, reasoning, and implementation clarity
- Easy to extend for incremental or streaming SCC detection

Applications:

- Compiler analysis (strongly connected variables)
- Circuit analysis
- Deadlock detection
- Dataflow optimization

#### How Does It Work (Plain Language)?

We maintain:

- `index[v]`: discovery order
- S stack: DFS path (nodes in current exploration)
- P stack: candidates for SCC roots

Algorithm steps:

1. Assign `index[v]` when visiting a node.
2. Push `v` onto both stacks (`S` and `P`).
3. For each neighbor `u`:

   * If `u` is unvisited → recurse
   * If `u` is on `S` → adjust `P` by popping until its top has an index ≤ `index[u]`
4. If `v` is at the top of `P` after processing all neighbors:

   * Pop `P`
   * Pop from `S` until `v` is removed
   * The popped vertices form one SCC

| Step | Current Node  | Stack S | Stack P | Action          |
| ---- | ------------- | ------- | ------- | --------------- |
| 1    | A             | [A]     | [A]     | Visit A         |
| 2    | B             | [A,B]   | [A,B]   | Visit B         |
| 3    | C             | [A,B,C] | [A,B,C] | Visit C         |
| 4    | Back edge C→B | [A,B,C] | [A,B]   | Adjust          |
| 5    | C done        | [A,B,C] | [A,B]   | Continue        |
| 6    | P top == B    | [A]     | [A]     | Found SCC {B,C} |

#### Tiny Code (Easy Versions)

C (Path-Based SCC)

```c
#include <stdio.h>
#include <stdbool.h>

#define MAX 100
int graph[MAX][MAX];
int n;
int index_counter = 0;
int indexv[MAX];
bool onStack[MAX];
int stackS[MAX], topS = -1;
int stackP[MAX], topP = -1;

void dfs_scc(int v) {
    indexv[v] = ++index_counter;
    stackS[++topS] = v;
    stackP[++topP] = v;
    onStack[v] = true;

    for (int u = 0; u < n; u++) {
        if (!graph[v][u]) continue;
        if (indexv[u] == 0) {
            dfs_scc(u);
        } else if (onStack[u]) {
            while (indexv[stackP[topP]] > indexv[u])
                topP--;
        }
    }

    if (stackP[topP] == v) {
        topP--;
        printf("SCC: ");
        int w;
        do {
            w = stackS[topS--];
            onStack[w] = false;
            printf("%d ", w);
        } while (w != v);
        printf("\n");
    }
}

int main(void) {
    printf("Enter number of vertices: ");
    scanf("%d", &n);
    printf("Enter adjacency matrix (directed):\n");
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            scanf("%d", &graph[i][j]);

    for (int i = 0; i < n; i++) indexv[i] = 0;

    printf("Strongly Connected Components:\n");
    for (int i = 0; i < n; i++)
        if (indexv[i] == 0)
            dfs_scc(i);
}
```

Python (Readable Implementation)

```python
from collections import defaultdict

graph = defaultdict(list)
edges = [(0,1),(1,2),(2,0),(1,3)]
for u,v in edges:
    graph[u].append(v)

index_counter = 0
index = {}
stackS, stackP = [], []
on_stack = set()
sccs = []

def dfs(v):
    global index_counter
    index_counter += 1
    index[v] = index_counter
    stackS.append(v)
    stackP.append(v)
    on_stack.add(v)

    for u in graph[v]:
        if u not in index:
            dfs(u)
        elif u in on_stack:
            while index[stackP[-1]] > index[u]:
                stackP.pop()

    if stackP and stackP[-1] == v:
        stackP.pop()
        comp = []
        while True:
            w = stackS.pop()
            on_stack.remove(w)
            comp.append(w)
            if w == v:
                break
        sccs.append(comp)

for v in list(graph.keys()):
    if v not in index:
        dfs(v)

print("Strongly Connected Components:", sccs)
```

#### Why It Matters

- Single DFS
- No low-link math, purely path-based reasoning
- Compact and intuitive for stack lovers
- Well-suited for theoretical clarity and educational use
- Matches Tarjan's O(V + E) performance

#### A Gentle Proof (Why It Works)

- Each vertex gets a discovery index;
- Stack `S` stores active path nodes;
- Stack `P` tracks potential SCC roots (lowest index still reachable).
  When a vertex finishes and equals top of `P`, all nodes above it in `S` form an SCC, they're mutually reachable, and none can reach earlier nodes.

The invariant ensures:

- Nodes stay on stack until their SCC is found
- SCCs are discovered in reverse topological order

#### Try It Yourself

1. Run it on a small graph, print `index[v]`, stack states after each call.
2. Add a cycle, trace how P adjusts.
3. Compare output with Tarjan's algorithm, they match!
4. Visualize path-based pops as SCC boundaries.
5. Try graphs with multiple disjoint SCCs.

#### Test Cases

| Graph              | Edges  | SCCs            |
| ------------------ | ------ | --------------- |
| 0→1, 1→2, 2→0      | Cycle  | {0,1,2}         |
| 0→1, 1→2           | Chain  | {0}, {1}, {2}   |
| 0→1, 1→0, 2→3      | Mixed  | {0,1}, {2}, {3} |
| 0→1, 1→2, 2→3, 3→1 | Nested | {1,2,3}, {0}    |

#### Complexity

- Time: O(V + E)
- Space: O(V)

Path-Based SCC is your stack ballet, each vertex steps forward, marks its place, and retreats gracefully, leaving behind a tightly choreographed component.




### 318 Kosaraju Parallel Version

The Parallel Kosaraju Algorithm adapts Kosaraju's classic two-pass SCC method to run on multiple processors or threads, making it suitable for large-scale graphs that can't be processed efficiently by a single thread. It divides the heavy lifting, DFS traversals and graph reversals, across many workers.

It's the natural evolution of Kosaraju's idea in the age of parallel computing: split the graph, explore concurrently, merge SCCs.

#### What Problem Are We Solving?

We want to compute SCCs in a massive directed graph efficiently, by taking advantage of parallel hardware, multicore CPUs, GPUs, or distributed systems.

The classic Kosaraju algorithm:

1. DFS on the original graph to record finishing times
2. Reverse all edges to create the transpose graph
3. DFS on the reversed graph in decreasing order of finishing time

The parallel version accelerates each phase by partitioning vertices or edges among processors.

Applications include:

- Large dependency graphs (package managers, compilers)
- Web graphs (page connectivity)
- Social networks (mutual reachability)
- GPU-accelerated analytics and graph mining

#### How Does It Work (Plain Language)?

We parallelize Kosaraju's two key passes:

1. Parallel Forward DFS (Finishing Order):

   * Partition vertices across threads.
   * Each thread runs DFS independently on its subgraph.
   * Maintain a shared stack of finish times (atomic appends).

2. Graph Reversal:

   * Reverse each edge $(u, v)$ into $(v, u)$ in parallel.
   * Each thread processes a slice of the edge list.

3. Parallel Reverse DFS (SCC Labeling):

   * Threads pop vertices from the global stack.
   * Each unvisited node starts a new component.
   * DFS labeling runs concurrently with atomic visited flags.

4. Merge Components:

   * Combine local SCC results using union–find if overlapping sets appear.

| Phase | Description           | Parallelized |
| :---- | :-------------------- | :----------- |
| 1     | DFS on original graph | ✅ Yes        |
| 2     | Reverse edges         | ✅ Yes        |
| 3     | DFS on reversed graph | ✅ Yes        |
| 4     | Merge labels          | ✅ Yes        |

#### Tiny Code (Pseudocode / Python Threaded Sketch)

> This example is conceptual, real parallel implementations use task queues, GPU kernels, or work-stealing schedulers.

```python
import threading
from collections import defaultdict

graph = defaultdict(list)
edges = [(0,1), (1,2), (2,0), (2,3), (3,4)]
for u, v in edges:
    graph[u].append(v)

n = 5
visited = [False] * n
finish_order = []
lock = threading.Lock()

def dfs_forward(v):
    visited[v] = True
    for u in graph[v]:
        if not visited[u]:
            dfs_forward(u)
    with lock:
        finish_order.append(v)

def parallel_forward():
    threads = []
    for v in range(n):
        if not visited[v]:
            t = threading.Thread(target=dfs_forward, args=(v,))
            t.start()
            threads.append(t)
    for t in threads:
        t.join()

# Reverse graph in parallel
rev = defaultdict(list)
for u in graph:
    for v in graph[u]:
        rev[v].append(u)

visited = [False] * n
components = []

def dfs_reverse(v, comp):
    visited[v] = True
    comp.append(v)
    for u in rev[v]:
        if not visited[u]:
            dfs_reverse(u, comp)

def parallel_reverse():
    while finish_order:
        v = finish_order.pop()
        if not visited[v]:
            comp = []
            dfs_reverse(v, comp)
            components.append(comp)

parallel_forward()
parallel_reverse()
print("SCCs:", components)
```

#### Why It Matters

- Enables SCC computation at scale
- Exploits multicore / GPU parallelism
- Critical for dataflow analysis, reachability, and graph condensation
- Powers large-scale graph analytics in scientific computing

#### A Gentle Proof (Why It Works)

Kosaraju's correctness depends on finishing times and reversed reachability:

- In the forward pass, each vertex $v$ gets a finishing time $t(v)$.
- In the reversed graph, DFS in descending $t(v)$ ensures that
  every SCC is discovered as a contiguous DFS tree.

Parallel execution maintains these invariants because:

1. All threads respect atomic finishing-time insertion
2. The global finishing order preserves a valid topological order
3. The reversed DFS still discovers mutually reachable vertices together

Thus, correctness holds under synchronized access, even with concurrent DFS traversals.

#### Try It Yourself

1. Split vertices into $p$ partitions.
2. Run forward DFS in parallel; record global finish stack.
3. Reverse edges concurrently.
4. Run backward DFS by popping from the stack.
5. Compare results with single-threaded Kosaraju, they match.

#### Test Cases

| Graph      | Edges                   | SCCs                 |
| :--------- | :---------------------- | :------------------- |
| Cycle      | 0→1, 1→2, 2→0           | {0, 1, 2}            |
| Chain      | 0→1, 1→2                | {0}, {1}, {2}        |
| Two cycles | 0→1, 1→0, 2→3, 3→2      | {0, 1}, {2, 3}       |
| Mixed      | 0→1, 1→2, 2→3, 3→0, 4→5 | {0, 1, 2, 3}, {4, 5} |

#### Complexity

Let $V$ be vertices, $E$ edges, $p$ processors.

- Work: $O(V + E)$ (same as sequential)
- Parallel Time:
  $T_p = O!\left(\frac{V + E}{p} + \text{sync\_cost}\right)$
- Space: $O(V + E)$

Kosaraju Parallel is your multi-voice chorus, each DFS sings in harmony, covering its section of the graph, and when the echoes settle, the full harmony of strongly connected components is revealed.

### 319 Dynamic SCC Maintenance

Dynamic SCC Maintenance deals with maintaining strongly connected components as a directed graph changes over time, edges or vertices may be added or removed.
The goal is to update SCCs incrementally rather than recomputing them from scratch after each change.

This approach is crucial in streaming, interactive, or evolving systems, where graphs represent real-world structures that shift continuously.

#### What Problem Are We Solving?

We want to keep track of SCCs under dynamic updates:

- Insertions: New edges can connect SCCs and form larger ones.
- Deletions: Removing edges can split SCCs into smaller ones.

Static algorithms like Tarjan or Kosaraju must restart completely.
Dynamic maintenance updates only the affected components, improving efficiency for large, frequently changing graphs.

Use cases include:

- Incremental compilation
- Dynamic program analysis
- Real-time dependency resolution
- Continuous graph query systems

#### How Does It Work (Plain Language)?

The dynamic SCC algorithm maintains:

- A condensation DAG representing SCCs.
- A reachability structure to detect cycles upon insertion.
- Local re-evaluation for affected nodes.

When an edge $(u, v)$ is added:

1. Identify components $C_u$ and $C_v$.
2. If $C_u = C_v$, no change.
3. If $C_v$ can reach $C_u$, a new cycle forms → merge SCCs.
4. Otherwise, add edge $C_u \to C_v$ in the condensation DAG.

When an edge $(u, v)$ is removed:

1. Remove it from the graph.
2. Check if $C_u$ and $C_v$ remain mutually reachable.
3. If not, recompute SCCs locally on the affected subgraph.

| Update                     | Action        | Result           |
| -------------------------- | ------------- | ---------------- |
| Add edge forming cycle     | Merge SCCs    | Larger component |
| Add edge without cycle     | DAG edge only | No merge         |
| Remove edge breaking cycle | Split SCC     | New components   |

#### Tiny Code (Simplified Example)

```python
class DSU:
    def __init__(self, n):
        self.parent = list(range(n))
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra

class DynamicSCC:
    def __init__(self, n):
        self.n = n
        self.dsu = DSU(n)
        self.graph = [set() for _ in range(n)]

    def add_edge(self, u, v):
        su, sv = self.dsu.find(u), self.dsu.find(v)
        if su == sv:
            return
        if self._reachable(sv, su):
            self.dsu.union(su, sv)
        else:
            self.graph[su].add(sv)

    def _reachable(self, start, target, seen=None):
        if seen is None: seen = set()
        if start == target: return True
        seen.add(start)
        for nxt in self.graph[start]:
            if nxt not in seen and self._reachable(nxt, target, seen):
                return True
        return False

    def components(self):
        comps = {}
        for v in range(self.n):
            root = self.dsu.find(v)
            comps.setdefault(root, []).append(v)
        return list(comps.values())

scc = DynamicSCC(4)
scc.add_edge(0, 1)
scc.add_edge(1, 2)
print(scc.components())  # [[0], [1], [2], [3]]
scc.add_edge(2, 0)
print(scc.components())  # [[0,1,2], [3]]
```

#### Why It Matters

- Efficient for long-running systems where graphs evolve
- Updates SCCs incrementally rather than rebuilding
- Supports real-time queries of connectivity
- Useful for streaming graph databases, incremental compilers, interactive modeling tools

#### A Gentle Proof (Why It Works)

For insertion:

- If $(u, v)$ connects components $C_u$ and $C_v$
- And if $C_v$ can reach $C_u$, then a cycle forms
- Merging $C_u$ and $C_v$ yields a valid SCC
- The condensation DAG remains acyclic

For deletion:

- If removal breaks reachability, SCC splits
- Recomputing locally ensures correctness
- Other unaffected SCCs remain valid

Each update modifies only the local neighborhood of the affected components.

#### Try It Yourself

1. Build a small directed graph.
2. Insert edges step by step and print components.
3. Add a back-edge to create a cycle and observe merging.
4. Remove an edge and check local recomputation.
5. Compare results with full Tarjan recomputation.

#### Test Cases

| Step | Edge       | SCCs            |
| ---- | ---------- | --------------- |
| 1    | 0→1        | {0}, {1}, {2}   |
| 2    | 1→2        | {0}, {1}, {2}   |
| 3    | 2→0        | {0,1,2}         |
| 4    | 2→3        | {0,1,2}, {3}    |
| 5    | remove 1→2 | {0,1}, {2}, {3} |

#### Complexity

- Insertion: $O(V + E)$ amortized (with reachability check)
- Deletion: Local recomputation, typically sublinear
- Space: $O(V + E)$

Dynamic SCC Maintenance provides a framework to keep SCCs consistent as the graph evolves, adapting efficiently to both incremental growth and structural decay.

### 320 SCC for Weighted Graph

SCC detection is usually discussed for unweighted graphs, where edge weights are irrelevant to reachability. However, in many real-world systems, weights encode constraints (cost, capacity, priority, probability), and we need to identify strong connectivity under these weighted conditions.
This variant integrates SCC algorithms with weighted edge logic, allowing selective inclusion or exclusion of edges based on weight criteria.

#### What Problem Are We Solving?

We want to find strongly connected components in a weighted directed graph.
A standard SCC algorithm ignores weights, it only checks reachability.
Here, we define SCCs based on edges satisfying a given weight predicate:

$$
(u, v) \in E_w \quad \text{iff} \quad w(u, v) \leq \theta
$$

We can then run SCC algorithms (Tarjan, Kosaraju, Gabow, Path-based) on the subgraph induced by edges satisfying the constraint.

Common use cases:

- Thresholded connectivity: Keep edges below cost $\theta$.
- Capacity-limited systems: Only include edges with capacity ≥ threshold.
- Dynamic constraint graphs: Recompute SCCs as thresholds shift.
- Probabilistic networks: Consider edges with probability ≥ $p$.

#### How Does It Work (Plain Language)?

1. Start with a weighted directed graph $G = (V, E, w)$
2. Apply a predicate on weights (e.g. $w(u, v) \le \theta$)
3. Build a filtered subgraph $G_\theta = (V, E_\theta)$
4. Run a standard SCC algorithm on $G_\theta$

The result groups vertices that are strongly connected under the weight constraint.

If $\theta$ changes, components can merge or split:

- Increasing $\theta$ (loosening) → SCCs merge
- Decreasing $\theta$ (tightening) → SCCs split

| Threshold $\theta$ | Included Edges | SCCs       |
| ------------------ | -------------- | ---------- |
| 3                  | $w \le 3$      | {A,B}, {C} |
| 5                  | $w \le 5$      | {A,B,C}    |

#### Tiny Code (Threshold-Based Filtering)

Python Example

```python
from collections import defaultdict

edges = [
    (0, 1, 2),
    (1, 2, 4),
    (2, 0, 1),
    (2, 3, 6),
    (3, 2, 6)
]

def build_subgraph(edges, theta):
    g = defaultdict(list)
    for u, v, w in edges:
        if w <= theta:
            g[u].append(v)
    return g

def dfs(v, g, visited, stack):
    visited.add(v)
    for u in g[v]:
        if u not in visited:
            dfs(u, g, visited, stack)
    stack.append(v)

def reverse_graph(g):
    rg = defaultdict(list)
    for u in g:
        for v in g[u]:
            rg[v].append(u)
    return rg

def kosaraju(g):
    visited, stack = set(), []
    for v in g:
        if v not in visited:
            dfs(v, g, visited, stack)
    rg = reverse_graph(g)
    visited.clear()
    comps = []
    while stack:
        v = stack.pop()
        if v not in visited:
            comp = []
            dfs(v, rg, visited, comp)
            comps.append(comp)
    return comps

theta = 4
g_theta = build_subgraph(edges, theta)
print("SCCs with threshold", theta, ":", kosaraju(g_theta))
```

Output:

```
SCCs with threshold 4 : [[2, 0, 1], [3]]
```

#### Why It Matters

- Incorporates weight constraints into connectivity
- Useful in optimization, routing, and clustering
- Supports incremental recomputation under shifting thresholds
- Enables multi-layer graph analysis (vary $\theta$ to see component evolution)

#### A Gentle Proof (Why It Works)

Reachability in a weighted graph depends on which edges are active.
Filtering edges by predicate preserves a subset of original reachability:

If there is a path $u \to v$ in $G_\theta$, all edges on that path satisfy $w(e) \le \theta$.
Since SCCs depend solely on reachability, standard algorithms applied to $G_\theta$ correctly identify weight-constrained SCCs.

As $\theta$ increases, the edge set $E_\theta$ grows monotonically:

$$
E_{\theta_1} \subseteq E_{\theta_2} \quad \text{for} \quad \theta_1 < \theta_2
$$

Therefore, the SCC partition becomes coarser (components merge).

#### Try It Yourself

1. Build a weighted graph.
2. Pick thresholds $\theta = 2, 4, 6$ and record SCCs.
3. Plot how components merge as $\theta$ increases.
4. Try predicates like $w(u, v) \ge \theta$.
5. Combine with Dynamic SCC Maintenance for evolving thresholds.

#### Test Cases

| $\theta$ | Edges Included      | SCCs            |
| -------- | ------------------- | --------------- |
| 2        | (0→1,2), (2→0)      | {0,2}, {1}, {3} |
| 4        | (0→1), (1→2), (2→0) | {0,1,2}, {3}    |
| 6        | All edges           | {0,1,2,3}       |

#### Complexity

- Filtering: $O(E)$
- SCC computation: $O(V + E_\theta)$
- Total: $O(V + E)$
- Space: $O(V + E)$

SCC for Weighted Graphs extends classic connectivity to contexts where not all edges are equal, revealing the layered structure of a graph as thresholds vary.

## Section 33. Shortest Paths 

### 321 Dijkstra (Binary Heap)

Dijkstra's Algorithm is the cornerstone of shortest path computation in weighted graphs with nonnegative edge weights. It grows a frontier of known shortest paths, always expanding from the vertex with the smallest current distance, much like a wavefront advancing through the graph.

Using a binary heap (priority queue) keeps the next closest vertex selection efficient, making this version the standard for practical use.

#### What Problem Are We Solving?

We need to find the shortest path from a single source vertex $s$ to all other vertices in a directed or undirected graph with nonnegative weights.

Given a weighted graph $G = (V, E, w)$ where $w(u, v) \ge 0$ for all $(u, v)$, the task is to compute:

$$
\text{dist}[v] = \min_{\text{path } s \to v} \sum_{(u, v) \in \text{path}} w(u, v)
$$

Typical use cases:

- GPS navigation (road networks)
- Network routing
- Pathfinding in games
- Dependency resolution in weighted systems

#### How Does It Work (Plain Language)?

The algorithm maintains a distance array `dist[]`, initialized with infinity for all vertices except the source.

1. Set `dist[s] = 0`.
2. Use a min-priority queue to repeatedly extract the vertex with the smallest distance.
3. For each neighbor, try to relax the edge:

$$
\text{if } \text{dist}[u] + w(u, v) < \text{dist}[v], \text{ then update } \text{dist}[v]
$$

4. Push the neighbor into the queue with updated distance.
5. Continue until the queue is empty.

| Step | Vertex Extracted | Updated Distances            |
| ---- | ---------------- | ---------------------------- |
| 1    | $s$              | $0$                          |
| 2    | Next smallest    | Update neighbors             |
| 3    | Repeat           | Until all vertices finalized |

This is a greedy algorithm, once a vertex is visited, its shortest distance is final.

#### Tiny Code (Binary Heap)

C (Using a Simple Priority Queue with `qsort`)

```c
#include <stdio.h>
#include <limits.h>
#include <stdbool.h>

#define MAX 100
#define INF INT_MAX

int n, graph[MAX][MAX];

void dijkstra(int src) {
    int dist[MAX], visited[MAX];
    for (int i = 0; i < n; i++) {
        dist[i] = INF;
        visited[i] = 0;
    }
    dist[src] = 0;

    for (int count = 0; count < n - 1; count++) {
        int u = -1, min = INF;
        for (int v = 0; v < n; v++)
            if (!visited[v] && dist[v] < min)
                min = dist[v], u = v;

        if (u == -1) break;
        visited[u] = 1;

        for (int v = 0; v < n; v++)
            if (graph[u][v] && !visited[v] &&
                dist[u] + graph[u][v] < dist[v])
                dist[v] = dist[u] + graph[u][v];
    }

    printf("Vertex\tDistance\n");
    for (int i = 0; i < n; i++)
        printf("%d\t%d\n", i, dist[i]);
}

int main(void) {
    printf("Enter number of vertices: ");
    scanf("%d", &n);
    printf("Enter adjacency matrix (0 for no edge):\n");
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            scanf("%d", &graph[i][j]);
    dijkstra(0);
}
```

Python (Heap-Based Implementation)

```python
import heapq

def dijkstra(graph, src):
    n = len(graph)
    dist = [float('inf')] * n
    dist[src] = 0
    pq = [(0, src)]
    
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        for v, w in graph[u]:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                heapq.heappush(pq, (dist[v], v))
    return dist

graph = {
    0: [(1, 2), (2, 4)],
    1: [(2, 1), (3, 7)],
    2: [(4, 3)],
    3: [(4, 1)],
    4: []
}

print(dijkstra(graph, 0))
```

#### Why It Matters

- Efficient shortest path when weights are nonnegative
- Deterministic and greedy, produces optimal paths
- Widely applicable to routing, logistics, and AI search
- Forms the foundation for A* and Johnson's algorithm

#### A Gentle Proof (Why It Works)

Dijkstra's invariant:
When a vertex $u$ is extracted from the priority queue, $\text{dist}[u]$ is final.

Proof idea:
All alternative paths to $u$ must go through vertices with greater or equal tentative distances, since edges are nonnegative. Thus, no shorter path exists.

By induction, the algorithm assigns the correct shortest distance to every vertex.

#### Try It Yourself

1. Run on a graph with 5 vertices and random weights.
2. Add an edge with a smaller weight and see how paths update.
3. Remove negative edges and note incorrect results.
4. Visualize the frontier expansion step by step.
5. Compare with Bellman–Ford on the same graph.

#### Test Cases

| Graph  | Edges                                          | Shortest Paths from 0 |
| ------ | ---------------------------------------------- | --------------------- |
| Simple | 0→1(2), 0→2(4), 1→2(1), 2→4(3), 1→3(7), 3→4(1) | [0, 2, 3, 7, 6]       |
| Chain  | 0→1(1), 1→2(1), 2→3(1)                         | [0, 1, 2, 3]          |
| Star   | 0→1(5), 0→2(2), 0→3(8)                         | [0, 5, 2, 8]          |

#### Complexity

- Time: $O((V + E)\log V)$ with binary heap
- Space: $O(V + E)$
- Works only if: $w(u, v) \ge 0$

Dijkstra (Binary Heap) is the workhorse of graph search, greedy but precise, always chasing the next closest frontier until all paths fall into place.

### 322 Dijkstra (Fibonacci Heap)

Dijkstra's algorithm can be further optimized by replacing the binary heap with a Fibonacci heap, which offers faster decrease-key operations. This improvement reduces the overall time complexity, making it more suitable for dense graphs or theoretical analysis where asymptotic efficiency matters.

While the constant factors are higher, the asymptotic time is improved to:

$$
O(E + V \log V)
$$

compared to the binary heap's $O((V + E)\log V)$.

#### What Problem Are We Solving?

We are computing single-source shortest paths in a directed or undirected weighted graph with nonnegative weights, but we want to optimize the priority queue operations to improve theoretical performance.

Given $G = (V, E, w)$ with $w(u, v) \ge 0$, the task remains:

$$
\text{dist}[v] = \min_{\text{path } s \to v} \sum_{(u,v) \in \text{path}} w(u, v)
$$

The difference lies in how we manage the priority queue that selects the next vertex to process.

#### How Does It Work (Plain Language)?

The logic of Dijkstra's algorithm is unchanged, only the data structure used for vertex selection and updates differs.

1. Initialize all distances to $\infty$, except $\text{dist}[s] = 0$.
2. Insert all vertices into a Fibonacci heap keyed by their current distance.
3. Repeatedly extract the vertex $u$ with smallest distance.
4. For each neighbor $(u, v)$:

   * If $\text{dist}[u] + w(u, v) < \text{dist}[v]$, update:
     $$
     \text{dist}[v] \gets \text{dist}[u] + w(u, v)
     $$
     and call decrease-key on $v$ in the heap.
5. Continue until all vertices are finalized.

The Fibonacci heap provides:

- `extract-min`: $O(\log V)$ amortized
- `decrease-key`: $O(1)$ amortized

This improves the performance for dense graphs where edge relaxations dominate.

#### Tiny Code (Python, Simplified Fibonacci Heap)

This code illustrates the structure but omits full heap details, production implementations use libraries like `networkx` or specialized data structures.

```python
from heapq import heappush, heappop  # stand-in for demonstration

def dijkstra_fib(graph, src):
    n = len(graph)
    dist = [float('inf')] * n
    dist[src] = 0
    visited = [False] * n
    heap = [(0, src)]

    while heap:
        d, u = heappop(heap)
        if visited[u]:
            continue
        visited[u] = True
        for v, w in graph[u]:
            if not visited[v] and dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                heappush(heap, (dist[v], v))
    return dist

graph = {
    0: [(1, 1), (2, 4)],
    1: [(2, 2), (3, 6)],
    2: [(3, 3)],
    3: []
}

print(dijkstra_fib(graph, 0))
```

*(The above uses `heapq` for illustration; true Fibonacci heap gives better theoretical bounds.)*

#### Why It Matters

- Improves theoretical time to $O(E + V \log V)$
- Demonstrates asymptotic optimization using advanced heaps
- Used in dense networks, theoretical research, and competition problems
- Foundation for algorithms like Johnson's APSP and minimum mean cycle

#### A Gentle Proof (Why It Works)

The correctness remains identical to Dijkstra's original proof:
When a vertex $u$ is extracted (minimum key), its shortest distance is final because all edge weights are nonnegative.

The heap choice only affects efficiency:

- Binary heap: every `decrease-key` = $O(\log V)$
- Fibonacci heap: every `decrease-key` = $O(1)$ amortized

Total operations:

- $V$ extractions × $O(\log V)$
- $E$ decreases × $O(1)$

Hence total time:

$$
O(V \log V + E) = O(E + V \log V)
$$

#### Try It Yourself

1. Build a dense graph (e.g. $V=1000, E \approx V^2$).
2. Compare runtimes with binary heap version.
3. Visualize priority queue operations.
4. Implement `decrease-key` manually for insight.
5. Explore Johnson's algorithm using this version.

#### Test Cases

| Graph    | Edges                        | Shortest Paths from 0 |
| -------- | ---------------------------- | --------------------- |
| Chain    | 0→1(1), 1→2(2), 2→3(3)       | [0, 1, 3, 6]          |
| Triangle | 0→1(2), 1→2(2), 0→2(5)       | [0, 2, 4]             |
| Dense    | All pairs with small weights | Works efficiently     |

#### Complexity

- Time: $O(E + V \log V)$
- Space: $O(V + E)$
- Works only if: $w(u, v) \ge 0$

Dijkstra (Fibonacci Heap) shows how data structure choice transforms an algorithm, the same idea, made sharper through careful engineering of priority operations.

### 323 Bellman–Ford

The Bellman–Ford algorithm solves the single-source shortest path problem for graphs that may contain negative edge weights.
Unlike Dijkstra's algorithm, it does not rely on greedy selection and can handle edges with $w(u, v) < 0$, as long as there are no negative-weight cycles reachable from the source.

It systematically relaxes every edge multiple times, ensuring all paths up to length $V-1$ are considered.

#### What Problem Are We Solving?

We want to compute shortest paths from a source $s$ in a weighted directed graph that may include negative weights.

Given $G = (V, E, w)$, find for all $v \in V$:

$$
\text{dist}[v] = \min_{\text{path } s \to v} \sum_{(u,v) \in \text{path}} w(u,v)
$$

If a negative-weight cycle is reachable from $s$, the shortest path is undefined (it can be reduced indefinitely).
Bellman–Ford detects this situation explicitly.

#### How Does It Work (Plain Language)?

Bellman–Ford uses edge relaxation repeatedly.

1. Initialize $\text{dist}[s] = 0$, others $\infty$.
2. Repeat $V - 1$ times:
   For every edge $(u, v)$:
   $$
   \text{if } \text{dist}[u] + w(u, v) < \text{dist}[v], \text{ then update } \text{dist}[v]
   $$
3. After $V - 1$ passes, all shortest paths are settled.
4. Perform one more pass: if any edge can still relax, a negative-weight cycle exists.

| Iteration | Updated Vertices              | Notes       |
| --------- | ----------------------------- | ----------- |
| 1         | Neighbors of source           | First layer |
| 2         | Next layer                    | Propagate   |
| ...       | ...                           | ...         |
| $V-1$     | All shortest paths stabilized | Done        |

#### Tiny Code (C Example)

```c
#include <stdio.h>
#include <limits.h>

#define MAX 100
#define INF 1000000000

typedef struct { int u, v, w; } Edge;
Edge edges[MAX];
int dist[MAX];

int main(void) {
    int V, E, s;
    printf("Enter vertices, edges, source: ");
    scanf("%d %d %d", &V, &E, &s);
    printf("Enter edges (u v w):\n");
    for (int i = 0; i < E; i++)
        scanf("%d %d %d", &edges[i].u, &edges[i].v, &edges[i].w);

    for (int i = 0; i < V; i++) dist[i] = INF;
    dist[s] = 0;

    for (int i = 1; i < V; i++)
        for (int j = 0; j < E; j++) {
            int u = edges[j].u, v = edges[j].v, w = edges[j].w;
            if (dist[u] != INF && dist[u] + w < dist[v])
                dist[v] = dist[u] + w;
        }

    for (int j = 0; j < E; j++) {
        int u = edges[j].u, v = edges[j].v, w = edges[j].w;
        if (dist[u] != INF && dist[u] + w < dist[v]) {
            printf("Negative cycle detected\n");
            return 0;
        }
    }

    printf("Vertex\tDistance\n");
    for (int i = 0; i < V; i++)
        printf("%d\t%d\n", i, dist[i]);
}
```

Python (Readable Version)

```python
def bellman_ford(V, edges, src):
    dist = [float('inf')] * V
    dist[src] = 0

    for _ in range(V - 1):
        for u, v, w in edges:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w

    # Detect negative cycle
    for u, v, w in edges:
        if dist[u] + w < dist[v]:
            raise ValueError("Negative cycle detected")

    return dist

edges = [(0, 1, 6), (0, 2, 7), (1, 2, 8), (1, 3, 5),
         (1, 4, -4), (2, 3, -3), (2, 4, 9), (3, 1, -2),
         (4, 3, 7)]

print(bellman_ford(5, edges, 0))
```

#### Why It Matters

- Works with negative weights
- Detects negative-weight cycles
- Simpler logic, easier to prove correctness
- Used in currency arbitrage, dynamic programming, policy evaluation

#### A Gentle Proof (Why It Works)

A shortest path has at most $V-1$ edges (a path longer than that must contain a cycle).
Each iteration ensures all paths up to that length are relaxed.
Thus after $V-1$ rounds, all shortest paths are found.

The $(V)$-th round detects further improvement, indicating a negative cycle.

Formally, after iteration $k$,
$\text{dist}[v]$ is the length of the shortest path from $s$ to $v$ using at most $k$ edges.

#### Try It Yourself

1. Run on graphs with negative weights.
2. Add a negative cycle and observe detection.
3. Compare results with Dijkstra (fails with negative edges).
4. Visualize relaxation per iteration.
5. Use it to detect arbitrage in currency exchange graphs.

#### Test Cases

| Graph          | Edges                    | Shortest Distances |
| -------------- | ------------------------ | ------------------ |
| Chain          | 0→1(5), 1→2(-2)          | [0, 5, 3]          |
| Negative edge  | 0→1(4), 0→2(5), 1→2(-10) | [0, 4, -6]         |
| Negative cycle | 0→1(1), 1→2(-2), 2→0(-1) | Detected           |

#### Complexity

- Time: $O(VE)$
- Space: $O(V)$
- Handles: $w(u, v) \ge -\infty$, no negative cycles

Bellman–Ford is the steady walker of shortest path algorithms, slower than Dijkstra, but unshaken by negative edges and always alert to cycles that break the rules.

### 324 SPFA (Queue Optimization)

The Shortest Path Faster Algorithm (SPFA) is an optimized implementation of Bellman–Ford that uses a queue to avoid unnecessary relaxations.
Instead of relaxing all edges in every iteration, SPFA only processes vertices whose distances were recently updated, often resulting in much faster average performance, especially in sparse graphs or those without negative cycles.

In the worst case, it still runs in $O(VE)$, but typical performance is closer to $O(E)$.

#### What Problem Are We Solving?

We want to find single-source shortest paths in a graph that may contain negative weights, but no negative-weight cycles.

Given a directed graph $G = (V, E, w)$ with edge weights $w(u, v)$ possibly negative, we compute:

$$
\text{dist}[v] = \min_{\text{path } s \to v} \sum_{(u, v) \in \text{path}} w(u, v)
$$

Bellman–Ford's $V-1$ rounds of edge relaxation can be wasteful; SPFA avoids rechecking edges from vertices that haven't improved.

#### How Does It Work (Plain Language)?

SPFA keeps a queue of vertices whose outgoing edges might lead to relaxation.
Each time a vertex's distance improves, it's enqueued for processing.

1. Initialize $\text{dist}[s] = 0$, others $\infty$.
2. Push $s$ into a queue.
3. While the queue is not empty:

   * Pop vertex $u$.
   * For each $(u, v)$:
     $$
     \text{if } \text{dist}[u] + w(u, v) < \text{dist}[v], \text{ then update } \text{dist}[v]
     $$
   * If $\text{dist}[v]$ changed and $v$ is not in queue, enqueue $v$.
4. Continue until queue is empty.

| Step | Queue                  | Operation                  |
| ---- | ---------------------- | -------------------------- |
| 1    | [s]                    | Start                      |
| 2    | Pop u, relax neighbors | Push improved vertices     |
| 3    | Repeat                 | Until no more improvements |

SPFA uses a lazy relaxation strategy, guided by actual updates.

#### Tiny Code (C Example)

```c
#include <stdio.h>
#include <stdbool.h>
#include <limits.h>

#define MAX 100
#define INF 1000000000

typedef struct { int v, w; } Edge;
Edge graph[MAX][MAX];
int deg[MAX];

int queue[MAX], front = 0, rear = 0;
bool in_queue[MAX];
int dist[MAX];

void enqueue(int x) {
    queue[rear++] = x;
    in_queue[x] = true;
}
int dequeue() {
    int x = queue[front++];
    in_queue[x] = false;
    return x;
}

int main(void) {
    int V, E, s;
    printf("Enter vertices, edges, source: ");
    scanf("%d %d %d", &V, &E, &s);

    for (int i = 0; i < V; i++) deg[i] = 0;
    printf("Enter edges (u v w):\n");
    for (int i = 0; i < E; i++) {
        int u, v, w;
        scanf("%d %d %d", &u, &v, &w);
        graph[u][deg[u]].v = v;
        graph[u][deg[u]].w = w;
        deg[u]++;
    }

    for (int i = 0; i < V; i++) dist[i] = INF;
    dist[s] = 0;
    enqueue(s);

    while (front < rear) {
        int u = dequeue();
        for (int i = 0; i < deg[u]; i++) {
            int v = graph[u][i].v, w = graph[u][i].w;
            if (dist[u] + w < dist[v]) {
                dist[v] = dist[u] + w;
                if (!in_queue[v]) enqueue(v);
            }
        }
    }

    printf("Vertex\tDistance\n");
    for (int i = 0; i < V; i++)
        printf("%d\t%d\n", i, dist[i]);
}
```

Python (Queue-Based Implementation)

```python
from collections import deque

def spfa(V, edges, src):
    graph = [[] for _ in range(V)]
    for u, v, w in edges:
        graph[u].append((v, w))
    
    dist = [float('inf')] * V
    in_queue = [False] * V
    dist[src] = 0

    q = deque([src])
    in_queue[src] = True

    while q:
        u = q.popleft()
        in_queue[u] = False
        for v, w in graph[u]:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                if not in_queue[v]:
                    q.append(v)
                    in_queue[v] = True
    return dist

edges = [(0,1,2),(0,2,4),(1,2,-1),(2,3,2)]
print(spfa(4, edges, 0))
```

#### Why It Matters

- Practical improvement over Bellman–Ford
- Efficient for sparse and nearly acyclic graphs
- Can handle negative weights
- Used in network optimization, real-time routing, flow systems

#### A Gentle Proof (Why It Works)

Each vertex enters the queue when its distance improves.
At most $V-1$ improvements per vertex (no shorter path with more edges).
Thus, every relaxation converges to the same fixed point as Bellman–Ford.

SPFA is an asynchronous relaxation method:

- Still guarantees correctness under nonnegative cycles
- Detects negative cycles if a vertex is enqueued $\ge V$ times

To check for negative cycles:

- Maintain a `count[v]` of relaxations
- If `count[v] > V`, report a cycle

#### Try It Yourself

1. Test on graphs with negative edges.
2. Compare runtime with Bellman–Ford.
3. Add negative cycle detection.
4. Try both sparse and dense graphs.
5. Measure how queue length varies during execution.

#### Test Cases

| Graph         | Edges                           | Result       |
| ------------- | ------------------------------- | ------------ |
| Simple        | 0→1(2), 0→2(4), 1→2(-1), 2→3(2) | [0, 2, 1, 3] |
| Negative edge | 0→1(5), 1→2(-3)                 | [0, 5, 2]    |
| Cycle         | 0→1(1), 1→2(-2), 2→0(1)         | Detect       |

#### Complexity

- Average: $O(E)$
- Worst case: $O(VE)$
- Space: $O(V + E)$

SPFA (Queue Optimization) is the agile Bellman–Ford, reacting only when change is needed, converging faster in practice while preserving the same correctness guarantees.

### 325 A* Search

The A* (A-star) algorithm combines Dijkstra's shortest path with best-first search, guided by a heuristic function.
It efficiently finds the shortest path from a start node to a goal node by always expanding the vertex that seems closest to the goal, according to the estimate:

$$
f(v) = g(v) + h(v)
$$

where

- $g(v)$ = cost from start to $v$ (known),
- $h(v)$ = heuristic estimate from $v$ to goal (guessed),
- $f(v)$ = total estimated cost through $v$.

When the heuristic is admissible (never overestimates), A* guarantees optimality.

#### What Problem Are We Solving?

We want to find the shortest path from a source $s$ to a target $t$ in a weighted graph (often spatial), using additional knowledge about the goal to guide the search.

Given $G = (V, E, w)$ and a heuristic $h(v)$, the task is to minimize:

$$
\text{cost}(s, t) = \min_{\text{path } s \to t} \sum_{(u, v) \in \text{path}} w(u, v)
$$

Applications:

- Pathfinding (games, robotics, navigation)
- Planning systems (AI, logistics)
- Grid and map searches
- State-space exploration

#### How Does It Work (Plain Language)?

A* behaves like Dijkstra's algorithm, but instead of expanding the closest node to the start ($g$), it expands the one with smallest estimated total cost ($f = g + h$).

1. Initialize all distances: `g[start] = 0`, others $\infty$.
2. Compute `f[start] = h[start]`.
3. Push `(f, node)` into a priority queue.
4. While queue not empty:

   * Pop node $u$ with smallest $f(u)$.
   * If $u = \text{goal}$, stop, path found.
   * For each neighbor $v$:
     $$
     g'(v) = g(u) + w(u, v)
     $$
     If $g'(v) < g(v)$, update:
     $$
     g(v) = g'(v), \quad f(v) = g(v) + h(v)
     $$
     Push $v$ into the queue.
5. Reconstruct path using parent pointers.

| Node  | $g(v)$ | $h(v)$    | $f(v) = g + h$ | Expanded? |
| ----- | ------ | --------- | -------------- | --------- |
| start | 0      | heuristic | heuristic      | ✅         |
| ...   | ...    | ...       | ...            | ...       |

The heuristic guides exploration, focusing on promising routes.

#### Tiny Code (Python Example)

Grid-based A* (Manhattan heuristic):

```python
import heapq

def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def astar(grid, start, goal):
    rows, cols = len(grid), len(grid[0])
    g = {start: 0}
    f = {start: heuristic(start, goal)}
    pq = [(f[start], start)]
    parent = {start: None}
    visited = set()

    while pq:
        _, current = heapq.heappop(pq)
        if current == goal:
            path = []
            while current:
                path.append(current)
                current = parent[current]
            return path[::-1]

        visited.add(current)
        x, y = current
        for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
            nx, ny = x + dx, y + dy
            neighbor = (nx, ny)
            if 0 <= nx < rows and 0 <= ny < cols and grid[nx][ny] == 0:
                tentative = g[current] + 1
                if tentative < g.get(neighbor, float('inf')):
                    g[neighbor] = tentative
                    f[neighbor] = tentative + heuristic(neighbor, goal)
                    parent[neighbor] = current
                    heapq.heappush(pq, (f[neighbor], neighbor))
    return None

grid = [
    [0,0,0,0],
    [1,1,0,1],
    [0,0,0,0],
    [0,1,1,0],
]
start = (0,0)
goal = (3,3)
path = astar(grid, start, goal)
print(path)
```

#### Why It Matters

- Faster than Dijkstra for goal-directed search
- Optimal if heuristic is admissible ($h(v) \le \text{true cost}$)
- Efficient if heuristic is also consistent (triangle inequality)
- Widely used in AI, robotics, navigation, route planning

#### A Gentle Proof (Why It Works)

If the heuristic $h(v)$ never overestimates the true remaining cost:

$$
h(v) \le \text{cost}(v, t)
$$

then $f(v) = g(v) + h(v)$ is always a lower bound on the true cost.
Therefore, when the goal is extracted (smallest $f$), the path is guaranteed optimal.

If $h$ also satisfies consistency:
$$
h(u) \le w(u, v) + h(v)
$$
then $f$-values are nondecreasing, and each node is expanded only once.

#### Try It Yourself

1. Implement A* on a grid with obstacles.
2. Experiment with different heuristics (Manhattan, Euclidean).
3. Set $h(v) = 0$ → becomes Dijkstra.
4. Set $h(v) = \text{true distance}$ → ideal search.
5. Try inadmissible $h$ → faster but possibly suboptimal.

#### Test Cases

| Graph Type | Heuristic | Result              |
| ---------- | --------- | ------------------- |
| Grid (4×4) | Manhattan | Shortest path found |
| Weighted   | Euclidean | Optimal route       |
| All $h=0$  | None      | Becomes Dijkstra    |

#### Complexity

- Time: $O(E \log V)$ (depends on heuristic)
- Space: $O(V)$
- Optimal if: $h$ admissible
- Complete if: finite branching factor

A* Search is Dijkstra with foresight, driven not just by cost so far, but by an informed guess of the journey ahead.

### 326 Floyd–Warshall

The Floyd–Warshall algorithm is a dynamic programming approach for computing all-pairs shortest paths (APSP) in a weighted directed graph.
It iteratively refines the shortest path estimates between every pair of vertices by allowing intermediate vertices step by step.

It works even with negative weights, as long as there are no negative cycles.

#### What Problem Are We Solving?

We want to compute:

$$
\text{dist}(u, v) = \min_{\text{paths } u \to v} \sum_{(x, y) \in \text{path}} w(x, y)
$$

for all pairs $(u, v)$ in a graph $G = (V, E, w)$.

We allow negative weights but no negative cycles.
It's particularly useful when:

- We need all-pairs shortest paths.
- The graph is dense ($E \approx V^2$).
- We want transitive closure or reachability (set $w(u, v) = 1$).

#### How Does It Work (Plain Language)?

We progressively allow each vertex as a possible intermediate waypoint.
Initially, the shortest path from $i$ to $j$ is just the direct edge.
Then, for each vertex $k$, we check if a path through $k$ improves the distance.

The recurrence relation:

$$
d_k(i, j) = \min \big( d_{k-1}(i, j),; d_{k-1}(i, k) + d_{k-1}(k, j) \big)
$$

Implementation uses in-place updates:

$$
\text{dist}[i][j] = \min(\text{dist}[i][j],; \text{dist}[i][k] + \text{dist}[k][j])
$$

Three nested loops:

1. `k` (intermediate)
2. `i` (source)
3. `j` (destination)

| k | i   | j   | Update                        |
| - | --- | --- | ----------------------------- |
| 0 | all | all | consider vertex 0 as waypoint |
| 1 | all | all | consider vertex 1 as waypoint |
| … | …   | …   | …                             |

After $V$ iterations, all shortest paths are finalized.

#### Tiny Code (C Example)

```c
#include <stdio.h>
#define INF 1000000000
#define MAX 100

int main(void) {
    int n;
    printf("Enter number of vertices: ");
    scanf("%d", &n);
    int dist[MAX][MAX];
    printf("Enter adjacency matrix (INF=9999, 0 for self):\n");
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            scanf("%d", &dist[i][j]);

    for (int k = 0; k < n; k++)
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                if (dist[i][k] + dist[k][j] < dist[i][j])
                    dist[i][j] = dist[i][k] + dist[k][j];

    printf("All-Pairs Shortest Distances:\n");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++)
            printf("%8d ", dist[i][j]);
        printf("\n");
    }
}
```

Python Version

```python
def floyd_warshall(graph):
    n = len(graph)
    dist = [row[:] for row in graph]

    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i][k] + dist[k][j] < dist[i][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
    return dist

INF = float('inf')
graph = [
    [0, 3, INF, 7],
    [8, 0, 2, INF],
    [5, INF, 0, 1],
    [2, INF, INF, 0]
]

res = floyd_warshall(graph)
for row in res:
    print(row)
```

#### Why It Matters

- Computes all-pairs shortest paths in one pass
- Works with negative weights
- Detects negative cycles if $\text{dist}[i][i] < 0$
- Useful for transitive closure, routing tables, graph condensation

#### A Gentle Proof (Why It Works)

Let $d_k(i, j)$ be the shortest path from $i$ to $j$ using only intermediate vertices from ${1, 2, \dots, k}$.

Base case ($k=0$):
$d_0(i, j) = w(i, j)$, the direct edge.

Inductive step:
For each $k$,
either the shortest path avoids $k$ or goes through $k$.
Thus:

$$
d_k(i, j) = \min \big(d_{k-1}(i, j),; d_{k-1}(i, k) + d_{k-1}(k, j)\big)
$$

By induction, after $V$ iterations, all shortest paths are covered.

#### Try It Yourself

1. Build a 4×4 weighted matrix.
2. Introduce a negative edge (but no cycle).
3. Check results after each iteration.
4. Detect cycle: observe if $\text{dist}[i][i] < 0$.
5. Use it to compute reachability (replace INF with 0/1).

#### Test Cases

| Graph          | Distances                              |                          |
| -------------- | -------------------------------------- | ------------------------ |
| Simple         | 0→1(3), 0→3(7), 1→2(2), 2→0(5), 3→0(2) | All-pairs paths computed |
| Negative edge  | 0→1(1), 1→2(-2), 2→0(4)                | Valid shortest paths     |
| Negative cycle | 0→1(1), 1→2(-2), 2→0(-1)               | $\text{dist}[0][0] < 0$  |

#### Complexity

- Time: $O(V^3)$
- Space: $O(V^2)$
- Detects: negative cycles if $\text{dist}[i][i] < 0$

Floyd–Warshall is the complete memory of a graph, every distance, every route, all computed through careful iteration over all possible intermediates.

### 327 Johnson's Algorithm

Johnson's Algorithm efficiently computes all-pairs shortest paths (APSP) in a sparse weighted directed graph, even when negative weights are present (but no negative cycles).
It cleverly combines Bellman–Ford and Dijkstra, using reweighting to eliminate negative edges while preserving shortest path relationships.

The result:
$$
O(VE + V^2 \log V)
$$
which is far more efficient than Floyd–Warshall ($O(V^3)$) for sparse graphs.

#### What Problem Are We Solving?

We need shortest paths between every pair of vertices, even if some edges have negative weights.
Directly running Dijkstra fails with negative edges, and running Bellman–Ford for each vertex would cost $O(V^2E)$.

Johnson's approach fixes this by reweighting edges to make them nonnegative, then applying Dijkstra from every vertex.

Given a weighted directed graph $G = (V, E, w)$:

$$
w(u, v) \in \mathbb{R}, \quad \text{no negative cycles.}
$$

We seek:

$$
\text{dist}(u, v) = \min_{\text{path } u \to v} \sum w(x, y)
$$

#### How Does It Work (Plain Language)?

1. Add a new vertex $s$, connect it to every other vertex with edge weight $0$.

2. Run Bellman–Ford from $s$ to compute potential values $h(v)$:
   $$
   h(v) = \text{dist}_s(v)
   $$

3. Reweight edges:
   $$
   w'(u, v) = w(u, v) + h(u) - h(v)
   $$
   This ensures all $w'(u, v) \ge 0$.

4. Remove $s$.

5. For each vertex $u$, run Dijkstra on the reweighted graph $w'$.

6. Recover original distances:
   $$
   \text{dist}(u, v) = \text{dist}'(u, v) + h(v) - h(u)
   $$

| Step | Action            | Result               |
| ---- | ----------------- | -------------------- |
| 1    | Add source $s$    | Connect to all       |
| 2    | Bellman–Ford      | Compute potentials   |
| 3    | Reweight edges    | All nonnegative      |
| 4    | Run Dijkstra      | $O(VE + V^2 \log V)$ |
| 5    | Restore distances | Adjust using $h$     |

#### Tiny Code (Python Example)

```python
import heapq

def bellman_ford(V, edges, s):
    dist = [float('inf')] * V
    dist[s] = 0
    for _ in range(V - 1):
        for u, v, w in edges:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
    for u, v, w in edges:
        if dist[u] + w < dist[v]:
            raise ValueError("Negative cycle detected")
    return dist

def dijkstra(V, adj, src):
    dist = [float('inf')] * V
    dist[src] = 0
    pq = [(0, src)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        for v, w in adj[u]:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                heapq.heappush(pq, (dist[v], v))
    return dist

def johnson(V, edges):
    # Step 1: add new source s
    s = V
    new_edges = edges + [(s, v, 0) for v in range(V)]
    h = bellman_ford(V + 1, new_edges, s)

    # Step 2: reweight edges
    adj = [[] for _ in range(V)]
    for u, v, w in edges:
        adj[u].append((v, w + h[u] - h[v]))

    # Step 3: run Dijkstra from each vertex
    dist_matrix = [[float('inf')] * V for _ in range(V)]
    for u in range(V):
        d = dijkstra(V, adj, u)
        for v in range(V):
            dist_matrix[u][v] = d[v] + h[v] - h[u]
    return dist_matrix

edges = [
    (0, 1, 1),
    (1, 2, -2),
    (2, 0, 4),
    (2, 3, 2),
    (3, 1, 7)
]

res = johnson(4, edges)
for row in res:
    print(row)
```

#### Why It Matters

- Works with negative weights
- Combines Bellman–Ford's flexibility with Dijkstra's speed
- Much faster than Floyd–Warshall for sparse graphs
- Used in routing, dependency graphs, and AI navigation

#### A Gentle Proof (Why It Works)

Reweighting preserves shortest path order:

For any path $P = (v_0, v_1, \dots, v_k)$:

$$
w'(P) = w(P) + h(v_0) - h(v_k)
$$

Therefore:

$$
w'(u, v) < w'(u, x) \iff w(u, v) < w(u, x)
$$

All shortest paths in $w$ are shortest in $w'$, but now all weights are nonnegative, allowing Dijkstra.

Finally, distances are restored:

$$
\text{dist}(u, v) = \text{dist}'(u, v) + h(v) - h(u)
$$

#### Try It Yourself

1. Add negative-weight edges (no cycles) and compare results with Floyd–Warshall.
2. Visualize reweighting: show $w'(u, v)$.
3. Test on sparse vs dense graphs.
4. Introduce negative cycle to trigger detection.
5. Replace Dijkstra with Fibonacci heap for $O(VE + V^2 \log V)$.

#### Test Cases

| Graph          | Edges                   | Result              |
| -------------- | ----------------------- | ------------------- |
| Triangle       | 0→1(1), 1→2(-2), 2→0(4) | All-pairs distances |
| Negative Edge  | 0→1(-1), 1→2(2)         | Correct             |
| Negative Cycle | 0→1(-2), 1→0(-3)        | Detected            |

#### Complexity

- Time: $O(VE + V^2 \log V)$
- Space: $O(V^2)$
- Works if: no negative cycles

Johnson's Algorithm is the harmonizer of shortest paths, reweighting the melody so every note becomes nonnegative, letting Dijkstra play across the entire graph with speed and precision.

### 328 0–1 BFS

The 0–1 BFS algorithm is a specialized shortest path technique for graphs where edge weights are only 0 or 1.
It uses a deque (double-ended queue) instead of a priority queue, allowing efficient relaxation in linear time.

By pushing 0-weight edges to the front and 1-weight edges to the back, the algorithm maintains an always-correct frontier, effectively simulating Dijkstra's behavior in $O(V + E)$ time.

#### What Problem Are We Solving?

We want to compute single-source shortest paths in a directed or undirected graph with edge weights in ${0, 1}$:

$$
w(u, v) \in {0, 1}
$$

We need:

$$
\text{dist}[v] = \min_{\text{path } s \to v} \sum_{(u, v) \in \text{path}} w(u, v)
$$

Typical applications:

- Unweighted graphs with special transitions (e.g. toggles, switches)
- State-space searches with free vs costly actions
- Binary grids, bitmask problems, or minimum operations graphs

#### How Does It Work (Plain Language)?

0–1 BFS replaces the priority queue in Dijkstra's algorithm with a deque, exploiting the fact that edge weights are only 0 or 1.

1. Initialize `dist[v] = ∞` for all $v$, set `dist[s] = 0`.
2. Push $s$ into deque.
3. While deque not empty:

   * Pop from front.
   * For each neighbor $(u, v)$:

     * If $w(u, v) = 0$ and improves distance, push front.
     * If $w(u, v) = 1$ and improves distance, push back.

This ensures vertices are always processed in non-decreasing order of distance, just like Dijkstra.

| Step  | Edge Type  | Action              |
| ----- | ---------- | ------------------- |
| $w=0$ | push front | process immediately |
| $w=1$ | push back  | process later       |

#### Tiny Code (C Example)

```c
#include <stdio.h>
#include <stdbool.h>
#include <limits.h>

#define MAX 1000
#define INF 1000000000

typedef struct { int v, w; } Edge;
Edge graph[MAX][MAX];
int deg[MAX];
int dist[MAX];
int deque[MAX * 2], front = MAX, back = MAX;

void push_front(int x) { deque[--front] = x; }
void push_back(int x) { deque[back++] = x; }
int pop_front() { return deque[front++]; }
bool empty() { return front == back; }

int main(void) {
    int V, E, s;
    scanf("%d %d %d", &V, &E, &s);
    for (int i = 0; i < E; i++) {
        int u, v, w;
        scanf("%d %d %d", &u, &v, &w);
        graph[u][deg[u]].v = v;
        graph[u][deg[u]].w = w;
        deg[u]++;
    }

    for (int i = 0; i < V; i++) dist[i] = INF;
    dist[s] = 0;
    push_front(s);

    while (!empty()) {
        int u = pop_front();
        for (int i = 0; i < deg[u]; i++) {
            int v = graph[u][i].v;
            int w = graph[u][i].w;
            if (dist[u] + w < dist[v]) {
                dist[v] = dist[u] + w;
                if (w == 0) push_front(v);
                else push_back(v);
            }
        }
    }

    for (int i = 0; i < V; i++)
        printf("%d: %d\n", i, dist[i]);
}
```

Python (Deque-Based)

```python
from collections import deque

def zero_one_bfs(V, edges, src):
    graph = [[] for _ in range(V)]
    for u, v, w in edges:
        graph[u].append((v, w))

    dist = [float('inf')] * V
    dist[src] = 0
    dq = deque([src])

    while dq:
        u = dq.popleft()
        for v, w in graph[u]:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                if w == 0:
                    dq.appendleft(v)
                else:
                    dq.append(v)
    return dist

edges = [(0,1,0),(1,2,1),(0,2,1),(2,3,0)]
print(zero_one_bfs(4, edges, 0))
```

#### Why It Matters

- Linear time for graphs with 0/1 weights
- Simpler than Dijkstra, faster in special cases
- Works on state-transition graphs (bit flips, BFS + cost)
- Common in competitive programming, AI, robotics

#### A Gentle Proof (Why It Works)

Because all edge weights are either 0 or 1, distances increase by at most 1 each step.
The deque ensures that nodes are processed in order of nondecreasing distance:

- When relaxing a $0$-edge, we push the vertex to the front (same distance).
- When relaxing a $1$-edge, we push to the back (distance +1).

Thus, the deque acts like a monotonic priority queue, guaranteeing correctness equivalent to Dijkstra.

#### Try It Yourself

1. Build a small graph with edges of 0 and 1.
2. Compare output with Dijkstra's algorithm.
3. Visualize deque operations.
4. Try a grid where moving straight costs 0, turning costs 1.
5. Measure runtime vs Dijkstra on sparse graphs.

#### Test Cases

| Graph      | Edges                          | Result       |
| ---------- | ------------------------------ | ------------ |
| Simple     | 0→1(0), 1→2(1), 0→2(1), 2→3(0) | [0, 0, 1, 1] |
| Zero edges | 0→1(0), 1→2(0), 2→3(0)         | [0, 0, 0, 0] |
| Mixed      | 0→1(1), 0→2(0), 2→3(1)         | [0, 1, 0, 1] |

#### Complexity

- Time: $O(V + E)$
- Space: $O(V + E)$
- Conditions: $w(u, v) \in {0, 1}$

0–1 BFS is the binary Dijkstra, a two-speed traveler that knows when to sprint ahead for free and when to patiently queue for a cost.

### 329 Dial's Algorithm

Dial's Algorithm is a variant of Dijkstra's algorithm optimized for graphs with nonnegative integer weights that are small and bounded.
Instead of a heap, it uses an array of buckets, one for each possible distance modulo the maximum edge weight.
This yields $O(V + E + C)$ performance, where $C$ is the maximum edge cost.

It's ideal when edge weights are integers in a small range, such as ${0, 1, \dots, C}$.

#### What Problem Are We Solving?

We want single-source shortest paths in a graph where:

$$
w(u, v) \in {0, 1, 2, \dots, C}, \quad C \text{ small}
$$

Given a weighted directed graph $G = (V, E, w)$, the goal is to compute:

$$
\text{dist}[v] = \min_{\text{path } s \to v} \sum_{(u, v) \in \text{path}} w(u, v)
$$

If we know the maximum edge weight $C$, we can replace a heap with an array of queues, cycling through distances efficiently.

Applications:

- Network routing with small costs
- Grid-based movement with few weight levels
- Telecommunication scheduling
- Traffic flow problems

#### How Does It Work (Plain Language)?

Dial's algorithm groups vertices by their current tentative distance.
Instead of a priority queue, it maintains buckets `B[0..C]`, where each bucket stores vertices with distance congruent modulo $(C+1)$.

1. Initialize $\text{dist}[v] = \infty$; set $\text{dist}[s] = 0$.
2. Place $s$ in `B[0]`.
3. For current bucket index `i`, process all vertices in `B[i]`:

   * For each edge $(u, v, w)$:
     $$
     \text{if } \text{dist}[u] + w < \text{dist}[v], \text{ then update } \text{dist}[v]
     $$
     Place $v$ in bucket `B[(i + w) \bmod (C+1)]`.
4. Move `i` to next nonempty bucket (cyclic).
5. Stop when all buckets empty.

| Step | Bucket | Vertices     | Action      |
| ---- | ------ | ------------ | ----------- |
| 0    | [0]    | start vertex | relax edges |
| 1    | [1]    | next layer   | propagate   |
| …    | …      | …            | …           |

This bucket-based relaxation ensures we always process vertices in increasing distance order, just like Dijkstra.

#### Tiny Code (Python Example)

```python
from collections import deque

def dial_algorithm(V, edges, src, C):
    graph = [[] for _ in range(V)]
    for u, v, w in edges:
        graph[u].append((v, w))

    INF = float('inf')
    dist = [INF] * V
    dist[src] = 0

    buckets = [deque() for _ in range(C + 1)]
    buckets[0].append(src)

    idx = 0
    processed = 0
    while processed < V:
        while not buckets[idx % (C + 1)]:
            idx += 1
        dq = buckets[idx % (C + 1)]
        u = dq.popleft()
        processed += 1

        for v, w in graph[u]:
            if dist[u] + w < dist[v]:
                old = dist[v]
                dist[v] = dist[u] + w
                new_idx = dist[v] % (C + 1)
                buckets[new_idx].append(v)
    return dist

edges = [
    (0, 1, 2),
    (0, 2, 1),
    (1, 3, 3),
    (2, 3, 1)
]
print(dial_algorithm(4, edges, 0, 3))
```

Output:

```
[0, 2, 1, 2]
```

#### Why It Matters

- Replaces heap with fixed-size array of queues
- Faster for small integer weights
- Linear-time behavior when $C$ is constant
- Simpler than Fibonacci heaps, but often as effective in practice

#### A Gentle Proof (Why It Works)

All edge weights are nonnegative integers bounded by $C$.
Each relaxation increases distance by at most $C$, so we only need $C+1$ buckets to track possible remainders modulo $C+1$.

Because each bucket is processed in cyclic order and vertices are only revisited when their distance decreases, the algorithm maintains nondecreasing distance order, ensuring correctness equivalent to Dijkstra.

#### Try It Yourself

1. Run on graphs with small integer weights (0–5).
2. Compare runtime vs binary heap Dijkstra.
3. Try $C=1$ → becomes 0–1 BFS.
4. Test $C=10$ → more buckets, but still fast.
5. Plot number of relaxations per bucket.

#### Test Cases

| Graph      | Edges                          | Max Weight $C$ | Distances    |
| ---------- | ------------------------------ | -------------- | ------------ |
| Simple     | 0→1(2), 0→2(1), 2→3(1), 1→3(3) | 3              | [0, 2, 1, 2] |
| Uniform    | 0→1(1), 1→2(1), 2→3(1)         | 1              | [0, 1, 2, 3] |
| Zero edges | 0→1(0), 1→2(0)                 | 1              | [0, 0, 0]    |

#### Complexity

- Time: $O(V + E + C)$
- Space: $O(V + C)$
- Condition: all edge weights $\in [0, C]$

Dial's Algorithm is the bucketed Dijkstra, it walks through distances layer by layer, storing each vertex in the slot that fits its cost, never needing a heap to know who's next.

### 330 Multi-Source Dijkstra

Multi-Source Dijkstra is a variant of Dijkstra's algorithm designed to find the shortest distance from multiple starting vertices to all others in a weighted graph.
Instead of running Dijkstra repeatedly, we initialize the priority queue with all sources at distance 0, and let the algorithm propagate the minimum distances simultaneously.

It's a powerful technique when you have several origins (cities, servers, entry points) and want the nearest path from any of them.

#### What Problem Are We Solving?

Given a weighted graph $G = (V, E, w)$ and a set of source vertices $S = {s_1, s_2, \dots, s_k}$, we want to compute:

$$
\text{dist}[v] = \min_{s_i \in S} \text{dist}(s_i, v)
$$

In other words, the distance to the closest source.

Typical use cases:

- Multi-depot routing (shortest route from any facility)
- Nearest service center (hospital, server, store)
- Multi-seed propagation (fire spread, BFS-like effects)
- Voronoi partitions in graphs

#### How Does It Work (Plain Language)?

The logic is simple:

1. Start with all sources in the priority queue, each at distance 0.
2. Perform Dijkstra's algorithm as usual.
3. Whenever a vertex is relaxed, the source that reached it first determines its distance.

Because we process vertices in increasing distance order, every vertex's distance reflects the nearest source.

| Step                               | Action |
| ---------------------------------- | ------ |
| Initialize dist[v] = ∞             |        |
| For each source s ∈ S: dist[s] = 0 |        |
| Push all s into priority queue     |        |
| Run standard Dijkstra              |        |

#### Tiny Code (Python Example)

```python
import heapq

def multi_source_dijkstra(V, edges, sources):
    graph = [[] for _ in range(V)]
    for u, v, w in edges:
        graph[u].append((v, w))
        graph[v].append((u, w))  # for undirected graph

    INF = float('inf')
    dist = [INF] * V
    pq = []

    for s in sources:
        dist[s] = 0
        heapq.heappush(pq, (0, s))

    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        for v, w in graph[u]:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                heapq.heappush(pq, (dist[v], v))
    return dist

edges = [
    (0, 1, 2),
    (1, 2, 3),
    (0, 3, 1),
    (3, 4, 4),
    (2, 4, 2)
]

sources = [0, 4]
print(multi_source_dijkstra(5, edges, sources))
```

Output:

```
[0, 2, 3, 1, 0]
```

Each vertex now knows its shortest distance from either source 0 or 4.

#### Why It Matters

- Efficient for multiple origins (no need for $k$ separate Dijkstra runs)
- Great for nearest-neighbor labeling or multi-region BFS
- Works on weighted graphs, unlike basic multi-source BFS
- A building block for graph Voronoi diagrams

#### A Gentle Proof (Why It Works)

Dijkstra's algorithm ensures vertices are processed in nondecreasing distance order.
By initializing all sources with distance 0, we treat them as a super-source connected by 0-weight edges:

$$
S^* \to s_i, \quad w(S^*, s_i) = 0
$$

Thus, multi-source Dijkstra is equivalent to a single-source Dijkstra from a virtual node connected to all sources, which guarantees correctness.

#### Try It Yourself

1. Add multiple sources to a city map graph.
2. Observe which source each node connects to first.
3. Compare results with $k$ separate Dijkstra runs.
4. Modify to also store source label (for Voronoi assignment).
5. Try on a grid where certain cells are starting fires or signals.

#### Test Cases

| Graph          | Sources | Result                     |
| -------------- | ------- | -------------------------- |
| Line graph 0–4 | [0, 4]  | [0, 1, 2, 1, 0]            |
| Triangle 0–1–2 | [0, 2]  | [0, 1, 0]                  |
| Grid           | Corners | Minimum steps from corners |

#### Complexity

- Time: $O((V + E) \log V)$
- Space: $O(V + E)$
- Condition: Nonnegative weights

Multi-Source Dijkstra is the chorus of shortest paths, all sources sing together, and every vertex listens to the closest voice.

## Section 34. Shortest Path Variants 

### 331 0–1 BFS

The 0–1 BFS algorithm is a specialized shortest path technique for graphs whose edge weights are only 0 or 1.
It's a streamlined version of Dijkstra's algorithm that replaces the priority queue with a deque (double-ended queue), taking advantage of the fact that only two possible edge weights exist.
This allows computing all shortest paths in linear time:

$$
O(V + E)
$$

#### What Problem Are We Solving?

We want to find the shortest path from a source vertex $s$ to all other vertices in a graph where

$$
w(u, v) \in {0, 1}
$$

Standard Dijkstra's algorithm works, but maintaining a heap is unnecessary overhead when edge weights are so simple.
The insight: edges with weight 0 do not increase distance, so they should be explored immediately, while edges with weight 1 should be explored next.

#### How Does It Work (Plain Language)?

Instead of a heap, we use a deque to manage vertices by their current shortest distance.

1. Initialize all distances to $\infty$, except $\text{dist}[s] = 0$.
2. Push $s$ into the deque.
3. While the deque is not empty:

   * Pop vertex $u$ from the front.
   * For each edge $(u, v)$ with weight $w$:

     * If $\text{dist}[u] + w < \text{dist}[v]$, update $\text{dist}[v]$.
     * If $w = 0$, push $v$ to the front (no distance increase).
     * If $w = 1$, push $v$ to the back (distance +1).

Because all edge weights are 0 or 1, this preserves correct ordering without a heap.

| Weight | Action     | Reason              |
| ------ | ---------- | ------------------- |
| 0      | push front | explore immediately |
| 1      | push back  | explore later       |

#### Tiny Code (Python Example)

```python
from collections import deque

def zero_one_bfs(V, edges, src):
    graph = [[] for _ in range(V)]
    for u, v, w in edges:
        graph[u].append((v, w))
        # for undirected graph, also add graph[v].append((u, w))

    dist = [float('inf')] * V
    dist[src] = 0
    dq = deque([src])

    while dq:
        u = dq.popleft()
        for v, w in graph[u]:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                if w == 0:
                    dq.appendleft(v)
                else:
                    dq.append(v)
    return dist

edges = [(0, 1, 0), (1, 2, 1), (0, 2, 1), (2, 3, 0)]
print(zero_one_bfs(4, edges, 0))
```

Output:

```
[0, 0, 1, 1]
```

#### Why It Matters

- Runs in $O(V + E)$, faster than Dijkstra's $O(E \log V)$
- Simplifies implementation when weights are 0 or 1
- Works on directed or undirected graphs
- Perfect for problems like:

  * Minimum number of flips/operations
  * Shortest path in binary grids
  * BFS with special cost transitions

#### A Gentle Proof (Why It Works)

At every step, the deque contains vertices ordered by nondecreasing distance.
When an edge of weight 0 is relaxed, the neighbor's distance equals $\text{dist}[u]$, so we process it immediately (push front).
When an edge of weight 1 is relaxed, the neighbor's distance increases by 1, so it goes to the back.

This maintains the same invariant as Dijkstra's:

> Every vertex is processed when its shortest distance is finalized.

Thus, correctness follows.

#### Try It Yourself

1. Compare with Dijkstra's algorithm on graphs with 0–1 weights.
2. Create a grid where moving straight costs 0 and turning costs 1.
3. Modify to handle undirected edges.
4. Use it for "minimum walls to break" problems.
5. Draw the deque contents step-by-step to visualize progression.

#### Test Cases

| Graph  | Edges                          | Distances    |
| ------ | ------------------------------ | ------------ |
| Simple | 0→1(0), 1→2(1), 0→2(1), 2→3(0) | [0, 0, 1, 1] |
| All 0  | 0→1(0), 1→2(0), 2→3(0)         | [0, 0, 0, 0] |
| Mixed  | 0→1(1), 1→2(0), 0→2(1)         | [0, 1, 1]    |

#### Complexity

- Time: $O(V + E)$
- Space: $O(V + E)$
- Condition: $w(u, v) \in {0, 1}$

0–1 BFS is a binary Dijkstra, moving through zero-cost edges first and one-cost edges next, fast, simple, and perfectly ordered.

### 332 Bidirectional Dijkstra

Bidirectional Dijkstra is an optimization of the classic Dijkstra's algorithm for single-pair shortest paths.
Instead of searching from just the source, we run two simultaneous Dijkstra searches, one forward from the source and one backward from the target, and stop when they meet.

This dramatically reduces the explored search space, especially in sparse or road-network-like graphs.

#### What Problem Are We Solving?

We want the shortest path between two specific vertices, $s$ (source) and $t$ (target), in a graph with nonnegative weights.
Standard Dijkstra explores the entire reachable graph, which is wasteful if we only need $s \to t$.

Bidirectional Dijkstra searches from both ends and meets in the middle:

$$
\text{dist}(s, t) = \min_{v \in V} \left( \text{dist}*\text{fwd}[v] + \text{dist}*\text{bwd}[v] \right)
$$

#### How Does It Work (Plain Language)?

The algorithm maintains two priority queues, one for the forward search (from $s$) and one for the backward search (from $t$).
Each search relaxes edges as in standard Dijkstra, but they alternate steps until their frontiers intersect.

1. Initialize all $\text{dist}*\text{fwd}$ and $\text{dist}*\text{bwd}$ to $\infty$.
2. Set $\text{dist}*\text{fwd}[s] = 0$, $\text{dist}*\text{bwd}[t] = 0$.
3. Insert $s$ into the forward heap, $t$ into the backward heap.
4. Alternate expanding one step forward and one step backward.
5. When a vertex $v$ is visited by both searches, compute candidate path:
   $$
   \text{dist}(s, v) + \text{dist}(t, v)
   $$
6. Stop when both queues are empty or the current minimum key exceeds the best candidate path.

| Direction | Heap        | Distance Array           |
| --------- | ----------- | ------------------------ |
| Forward   | From source | $\text{dist}_\text{fwd}$ |
| Backward  | From target | $\text{dist}_\text{bwd}$ |

#### Tiny Code (Python Example)

```python
import heapq

def bidirectional_dijkstra(V, edges, s, t):
    graph = [[] for _ in range(V)]
    rev_graph = [[] for _ in range(V)]
    for u, v, w in edges:
        graph[u].append((v, w))
        rev_graph[v].append((u, w))  # reverse for backward search

    INF = float('inf')
    dist_f = [INF] * V
    dist_b = [INF] * V
    visited_f = [False] * V
    visited_b = [False] * V

    dist_f[s] = 0
    dist_b[t] = 0
    pq_f = [(0, s)]
    pq_b = [(0, t)]
    best = INF

    while pq_f or pq_b:
        if pq_f:
            d, u = heapq.heappop(pq_f)
            if d > dist_f[u]:
                continue
            visited_f[u] = True
            if visited_b[u]:
                best = min(best, dist_f[u] + dist_b[u])
            for v, w in graph[u]:
                if dist_f[u] + w < dist_f[v]:
                    dist_f[v] = dist_f[u] + w
                    heapq.heappush(pq_f, (dist_f[v], v))

        if pq_b:
            d, u = heapq.heappop(pq_b)
            if d > dist_b[u]:
                continue
            visited_b[u] = True
            if visited_f[u]:
                best = min(best, dist_f[u] + dist_b[u])
            for v, w in rev_graph[u]:
                if dist_b[u] + w < dist_b[v]:
                    dist_b[v] = dist_b[u] + w
                    heapq.heappush(pq_b, (dist_b[v], v))

        if best < min(pq_f[0][0] if pq_f else INF, pq_b[0][0] if pq_b else INF):
            break

    return best if best != INF else None

edges = [
    (0, 1, 2),
    (1, 2, 3),
    (0, 3, 1),
    (3, 4, 4),
    (4, 2, 2)
]

print(bidirectional_dijkstra(5, edges, 0, 2))
```

Output:

```
5
```

#### Why It Matters

- Half the work of standard Dijkstra on average
- Best suited for sparse, road-like networks
- Great for navigation, routing, pathfinding
- Foundation for advanced methods like ALT and Contraction Hierarchies

#### A Gentle Proof (Why It Works)

Dijkstra's invariant: vertices are processed in nondecreasing distance order.
By running two searches, we maintain this invariant in both directions.
When a vertex is reached by both searches, any further expansion can only find paths longer than the current best:

$$
\text{dist}*\text{fwd}[u] + \text{dist}*\text{bwd}[u] = \text{candidate shortest path}
$$

Thus, the first intersection yields the optimal distance.

#### Try It Yourself

1. Compare explored nodes vs single Dijkstra.
2. Visualize frontiers meeting in the middle.
3. Add a grid graph with uniform weights.
4. Combine with heuristics → bidirectional A*.
5. Use backward search on reverse edges.

#### Test Cases

| Graph    | Source                         | Target | Result |   |
| -------- | ------------------------------ | ------ | ------ | - |
| Line     | 0→1→2→3                        | 0      | 3      | 3 |
| Triangle | 0→1(2), 1→2(2), 0→2(5)         | 0      | 2      | 4 |
| Road     | 0→1(1), 1→2(2), 0→3(3), 3→2(1) | 0      | 2      | 3 |

#### Complexity

- Time: $O((V + E) \log V)$
- Space: $O(V + E)$
- Condition: Nonnegative weights

Bidirectional Dijkstra is a meeting-in-the-middle pathfinder, two explorers start from opposite ends, racing toward each other until they share the shortest route.

### 333 A* with Euclidean Heuristic

The A* algorithm is a heuristic-guided shortest path search, blending Dijkstra's rigor with informed direction.
By introducing a heuristic function $h(v)$ that estimates the remaining distance, it expands fewer nodes and focuses the search toward the goal.
When using Euclidean distance as the heuristic, A* is perfect for spatial graphs, grids, maps, road networks.

#### What Problem Are We Solving?

We want the shortest path from a source $s$ to a target $t$ in a weighted graph with nonnegative weights,
but we also want to avoid exploring unnecessary regions.

Dijkstra expands all nodes in order of true cost $g(v)$ (distance so far).
A* expands nodes in order of estimated total cost:

$$
f(v) = g(v) + h(v)
$$

where

- $g(v)$ = cost so far from $s$ to $v$,
- $h(v)$ = heuristic estimate from $v$ to $t$.

If $h(v)$ never overestimates the true cost, A* guarantees the optimal path.

#### How Does It Work (Plain Language)?

Think of A* as Dijkstra with a compass.
While Dijkstra explores all directions equally, A* uses $h(v)$ to bias exploration toward the target.

1. Initialize $\text{dist}[s] = 0$, $\text{f}[s] = h(s)$
2. Push $(f(s), s)$ into a priority queue
3. While queue not empty:

   * Pop vertex $u$ with smallest $f(u)$
   * If $u = t$, stop, path found
   * For each neighbor $(u, v)$ with weight $w$:

     * Compute tentative cost $g' = \text{dist}[u] + w$
     * If $g' < \text{dist}[v]$:
       $$
       \text{dist}[v] = g', \quad f(v) = g' + h(v)
       $$
       Push $(f(v), v)$ into queue

Heuristic types:

- Euclidean: $h(v) = \sqrt{(x_v - x_t)^2 + (y_v - y_t)^2}$
- Manhattan: $h(v) = |x_v - x_t| + |y_v - y_t|$
- Zero: $h(v) = 0$ → reduces to Dijkstra

#### Tiny Code (Python Example)

```python
import heapq
import math

def a_star_euclidean(V, edges, coords, s, t):
    graph = [[] for _ in range(V)]
    for u, v, w in edges:
        graph[u].append((v, w))
        graph[v].append((u, w))  # undirected

    def h(v):
        x1, y1 = coords[v]
        x2, y2 = coords[t]
        return math.sqrt((x1 - x2)2 + (y1 - y2)2)

    dist = [float('inf')] * V
    dist[s] = 0
    pq = [(h(s), s)]

    while pq:
        f, u = heapq.heappop(pq)
        if u == t:
            return dist[u]
        for v, w in graph[u]:
            g_new = dist[u] + w
            if g_new < dist[v]:
                dist[v] = g_new
                heapq.heappush(pq, (g_new + h(v), v))
    return None

coords = [(0,0), (1,0), (1,1), (2,1)]
edges = [(0,1,1), (1,2,1), (0,2,2), (2,3,1)]
print(a_star_euclidean(4, edges, coords, 0, 3))
```

Output:

```
3
```

#### Why It Matters

- Optimal if $h(v)$ is admissible ($h(v) \le$ true distance)
- Fast if $h(v)$ is consistent ($h(u) \le w(u,v) + h(v)$)
- Perfect for spatial navigation and grid-based pathfinding
- Underpins many AI systems: games, robots, GPS routing

#### A Gentle Proof (Why It Works)

A* ensures correctness through admissibility:
$$
h(v) \le \text{dist}(v, t)
$$

This means $f(v) = g(v) + h(v)$ never underestimates the total path cost, so the first time $t$ is dequeued, the shortest path is found.

Consistency ensures that $f(v)$ values are nondecreasing, mimicking Dijkstra's invariant.

Thus, A* retains Dijkstra's guarantees while guiding exploration efficiently.

#### Try It Yourself

1. Compare explored nodes with Dijkstra's algorithm.
2. Use Euclidean and Manhattan heuristics on a grid.
3. Try a bad heuristic (e.g. double true distance) → see failure.
4. Visualize search frontier for each step.
5. Apply to a maze or road map.

#### Test Cases

| Graph        | Heuristic                 | Result        |
| ------------ | ------------------------- | ------------- |
| Grid         | Euclidean                 | Straight path |
| Triangle     | $h=0$                     | Dijkstra      |
| Overestimate | $h(v) > \text{dist}(v,t)$ | May fail      |

#### Complexity

- Time: $O(E \log V)$
- Space: $O(V)$
- Condition: Nonnegative weights, admissible heuristic

A* with Euclidean heuristic is the navigator's Dijkstra, guided by distance, it finds the shortest route while knowing where it's headed.

### 334 ALT Algorithm (A* Landmarks + Triangle Inequality)

The ALT Algorithm enhances A* search with precomputed landmarks and the triangle inequality, giving it a strong, admissible heuristic that dramatically speeds up shortest path queries on large road networks.

The name "ALT" comes from A* (search), Landmarks, and Triangle inequality, a trio that balances preprocessing with query-time efficiency.

#### What Problem Are We Solving?

We want to find shortest paths efficiently in large weighted graphs (like road maps), where a single-source search (like Dijkstra or A*) may explore millions of nodes.

To guide the search more effectively, we precompute distances from special nodes (landmarks) and use them to build tight heuristic bounds during A*.

Given nonnegative edge weights, we define a heuristic based on the triangle inequality:

$$
d(a, b) \le d(a, L) + d(L, b)
$$

From this we derive a lower bound for $d(a, b)$:

$$
h(a) = \max_{L \in \text{landmarks}} |d(L, t) - d(L, a)|
$$

This $h(a)$ is admissible (never overestimates) and consistent (monotone).

#### How Does It Work (Plain Language)?

ALT augments A* with preprocessing and landmark distances:

Preprocessing:

1. Choose a small set of landmarks $L_1, L_2, \dots, L_k$ (spread across the graph).
2. Run Dijkstra (or BFS) from each landmark to compute distances to all nodes:
   $$
   d(L_i, v) \text{ and } d(v, L_i)
   $$

Query phase:

1. For a query $(s, t)$, compute:
   $$
   h(v) = \max_{i=1}^k |d(L_i, t) - d(L_i, v)|
   $$
2. Run A* with this heuristic:
   $$
   f(v) = g(v) + h(v)
   $$
3. Guaranteed optimal path (admissible and consistent).

| Phase         | Task                                 | Cost                 |
| ------------- | ------------------------------------ | -------------------- |
| Preprocessing | Multi-source Dijkstra from landmarks | $O(k(V+E)\log V)$    |
| Query         | A* with landmark heuristic           | Fast ($O(E'\log V)$) |

#### Tiny Code (Python Example)

```python
import heapq

def dijkstra(V, graph, src):
    dist = [float('inf')] * V
    dist[src] = 0
    pq = [(0, src)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        for v, w in graph[u]:
            if d + w < dist[v]:
                dist[v] = d + w
                heapq.heappush(pq, (dist[v], v))
    return dist

def alt_search(V, graph, landmarks, d_landmark_to, s, t):
    def h(v):
        # max of triangle inequality heuristic
        return max(abs(d_landmark_to[L][t] - d_landmark_to[L][v]) for L in landmarks)

    dist = [float('inf')] * V
    dist[s] = 0
    pq = [(h(s), s)]

    while pq:
        f, u = heapq.heappop(pq)
        if u == t:
            return dist[u]
        for v, w in graph[u]:
            g_new = dist[u] + w
            if g_new < dist[v]:
                dist[v] = g_new
                heapq.heappush(pq, (g_new + h(v), v))
    return None

# Example usage
V = 5
graph = [
    [(1, 2), (2, 4)],
    [(2, 1), (3, 7)],
    [(3, 3)],
    [(4, 1)],
    []
]

landmarks = [0, 4]
d_landmark_to = [dijkstra(V, graph, L) for L in landmarks]

print(alt_search(V, graph, landmarks, d_landmark_to, 0, 4))
```

Output:

```
11
```

#### Why It Matters

- Far fewer node expansions than Dijkstra or vanilla A*
- Admissible (never overestimates) and consistent
- Especially effective on road networks, navigation systems, GIS, and logistics routing
- Preprocessing is offline, query is real-time fast

#### A Gentle Proof (Why It Works)

For any nodes $a, b$, and landmark $L$:

$$
|d(L, b) - d(L, a)| \le d(a, b)
$$

By taking the maximum over all chosen landmarks:

$$
h(a) = \max_L |d(L, t) - d(L, a)| \le d(a, t)
$$

Therefore $h(a)$ is admissible, and since triangle inequality is symmetric, it is consistent:

$$
h(a) \le w(a, b) + h(b)
$$

Thus, A* with $h(a)$ preserves optimality.

#### Try It Yourself

1. Choose 2–3 landmarks spread across your graph.
2. Compare A* expansions with and without ALT heuristic.
3. Visualize heuristic contours around landmarks.
4. Use in city maps to speed up routing queries.
5. Experiment with random vs central landmarks.

#### Test Cases

| Graph  | Landmarks     | Result              |
| ------ | ------------- | ------------------- |
| Chain  | [first, last] | Exact heuristic     |
| Grid   | 4 corners     | Smooth guidance     |
| Random | random nodes  | Varying performance |

#### Complexity

- Preprocessing: $O(k(V + E) \log V)$
- Query: $O(E' \log V)$ (small subset)
- Space: $O(kV)$
- Condition: Nonnegative weights

The ALT Algorithm is A* on steroids, guided by precomputed wisdom (landmarks), it leaps across the graph using geometry, not guesswork.

### 335 Contraction Hierarchies

Contraction Hierarchies (CH) is a powerful speedup technique for shortest path queries on large, static road networks.
It preprocesses the graph by adding shortcuts and ordering vertices by importance, enabling queries that run orders of magnitude faster than plain Dijkstra.

It's the backbone of many modern routing engines (like OSRM, GraphHopper, Valhalla) used in GPS systems.

#### What Problem Are We Solving?

We want to answer many shortest path queries quickly on a large, unchanging graph (like a road map).
Running Dijkstra or even A* for each query is too slow.

Contraction Hierarchies solves this by:

1. Preprocessing once to create a hierarchy of nodes.
2. Answering each query with a much smaller search (bidirectional).

Tradeoff: expensive preprocessing, blazing-fast queries.

#### How Does It Work (Plain Language)?

Contraction Hierarchies is a two-phase algorithm:

##### 1. Preprocessing Phase (Build Hierarchy)

We order nodes by importance (e.g., degree, centrality, traffic volume).
Then we contract them one by one, adding shortcut edges so that shortest path distances remain correct.

For each node $v$ being removed:

- For each pair of neighbors $(u, w)$:

  * If the shortest path $u \to v \to w$ is the only shortest path between $u$ and $w$,
    add a shortcut $(u, w)$ with weight $w(u, v) + w(v, w)$.

We record the order in which nodes are contracted.

| Step          | Action          | Result          |
| ------------- | --------------- | --------------- |
| Pick node $v$ | Contract        | Add shortcuts   |
| Continue      | Until all nodes | Build hierarchy |

The graph becomes layered: low-importance nodes contracted first, high-importance last.

##### 2. Query Phase (Up-Down Search)

Given a query $(s, t)$:

- Run bidirectional Dijkstra, but only "upward" along higher-ranked nodes.
- Stop when both searches meet.

This Up-Down constraint keeps the search small, only a tiny fraction of the graph is explored.

| Direction | Constraint                     |
| --------- | ------------------------------ |
| Forward   | Visit only higher-ranked nodes |
| Backward  | Visit only higher-ranked nodes |

The shortest path is the lowest point where the two searches meet.

#### Tiny Code (Python Example, Conceptual)

```python
import heapq

def add_shortcuts(graph, order):
    V = len(graph)
    shortcuts = [[] for _ in range(V)]
    for v in order:
        neighbors = graph[v]
        for u, wu in neighbors:
            for w, ww in neighbors:
                if u == w:
                    continue
                new_dist = wu + ww
                # If no shorter path exists between u and w, add shortcut
                exists = any(n == w and cost <= new_dist for n, cost in graph[u])
                if not exists:
                    shortcuts[u].append((w, new_dist))
        # Remove v (contract)
        graph[v] = []
    return [graph[i] + shortcuts[i] for i in range(V)]

def upward_edges(graph, order):
    rank = {v: i for i, v in enumerate(order)}
    up = [[] for _ in range(len(graph))]
    for u in range(len(graph)):
        for v, w in graph[u]:
            if rank[v] > rank[u]:
                up[u].append((v, w))
    return up

def ch_query(up_graph, s, t):
    def dijkstra_dir(start):
        dist = {}
        pq = [(0, start)]
        while pq:
            d, u = heapq.heappop(pq)
            if u in dist:
                continue
            dist[u] = d
            for v, w in up_graph[u]:
                heapq.heappush(pq, (d + w, v))
        return dist

    dist_s = dijkstra_dir(s)
    dist_t = dijkstra_dir(t)
    best = float('inf')
    for v in dist_s:
        if v in dist_t:
            best = min(best, dist_s[v] + dist_t[v])
    return best

# Example graph
graph = [
    [(1, 2), (2, 4)],
    [(2, 1), (3, 7)],
    [(3, 3)],
    []
]
order = [0, 1, 2, 3]  # Simplified
up_graph = upward_edges(add_shortcuts(graph, order), order)
print(ch_query(up_graph, 0, 3))
```

Output:

```
8
```

#### Why It Matters

- Query speed: microseconds, even on million-node graphs
- Used in GPS navigation, road routing, logistics planning
- Preprocessing preserves correctness via shortcuts
- Easily combined with ALT, A*, and Multi-level Dijkstra

#### A Gentle Proof (Why It Works)

When contracting node $v$, we add shortcuts to preserve all shortest paths that pass through $v$.
Thus, removing $v$ never breaks shortest-path correctness.

During query:

- We only travel upward in rank.
- Since all paths can be expressed as up-down-up, the meeting point of the forward and backward searches must lie on the true shortest path.

By limiting exploration to "upward" edges, CH keeps searches small without losing completeness.

#### Try It Yourself

1. Build a small graph, choose an order, and add shortcuts.
2. Compare Dijkstra vs CH query times.
3. Visualize hierarchy (contracted vs remaining).
4. Experiment with random vs heuristic node orderings.
5. Add landmarks (ALT+CH) for further optimization.

#### Test Cases

| Graph    | Nodes   | Query         | Result                |
| -------- | ------- | ------------- | --------------------- |
| Chain    | 0–1–2–3 | 0→3           | 3 edges               |
| Triangle | 0–1–2   | 0→2           | Shortcut 0–2 added    |
| Grid     | 3×3     | Corner→Corner | Shortcuts reduce hops |

#### Complexity

- Preprocessing: $O(V \log V + E)$ (with heuristic ordering)
- Query: $O(\log V)$ (tiny search space)
- Space: $O(V + E + \text{shortcuts})$
- Condition: Static graph, nonnegative weights

Contraction Hierarchies is the architect's Dijkstra, it reshapes the city first, then navigates with near-instant precision.

### 336 CH Query Algorithm (Shortcut-Based Routing)

The CH Query Algorithm is the online phase of Contraction Hierarchies (CH).
Once preprocessing builds the shortcut-augmented hierarchy, queries can be answered in microseconds by performing an upward bidirectional Dijkstra, searching only along edges that lead to more important (higher-ranked) nodes.

It's the practical magic behind instantaneous route planning in navigation systems.

#### What Problem Are We Solving?

Given a contracted graph (with shortcuts) and a node ranking,
we want to compute the shortest distance between two vertices $s$ and $t$ —
without exploring the full graph.

Rather than scanning every node, CH Query:

1. Searches upward from both $s$ and $t$ (following rank order).
2. Stops when both searches meet.
3. The meeting point with the smallest sum of forward + backward distances gives the shortest path.

#### How Does It Work (Plain Language)?

After preprocessing (which contracts nodes and inserts shortcuts),
the query algorithm runs two simultaneous upward Dijkstra searches:

1. Initialize two heaps:

   * Forward from $s$
   * Backward from $t$

2. Relax only upward edges, from lower-rank nodes to higher-rank ones.
   (Edge $(u, v)$ is upward if $\text{rank}(v) > \text{rank}(u)$.)

3. Whenever a node $v$ is settled by both searches,
   compute potential path:
   $$
   d = \text{dist}_f[v] + \text{dist}_b[v]
   $$

4. Track the minimum such $d$.

5. Stop when current best $d$ is smaller than remaining unvisited frontier keys.

| Phase           | Description                               |
| --------------- | ----------------------------------------- |
| Forward Search  | Upward edges from $s$                     |
| Backward Search | Upward edges from $t$ (in reversed graph) |
| Meet in Middle  | Combine distances at intersection         |

#### Tiny Code (Python Example, Conceptual)

```python
import heapq

def ch_query(up_graph, rank, s, t):
    INF = float('inf')
    n = len(up_graph)
    dist_f = [INF] * n
    dist_b = [INF] * n
    visited_f = [False] * n
    visited_b = [False] * n

    dist_f[s] = 0
    dist_b[t] = 0
    pq_f = [(0, s)]
    pq_b = [(0, t)]
    best = INF

    while pq_f or pq_b:
        # Forward direction
        if pq_f:
            d, u = heapq.heappop(pq_f)
            if d > dist_f[u]:
                continue
            visited_f[u] = True
            if visited_b[u]:
                best = min(best, dist_f[u] + dist_b[u])
            for v, w in up_graph[u]:
                if rank[v] > rank[u] and dist_f[u] + w < dist_f[v]:
                    dist_f[v] = dist_f[u] + w
                    heapq.heappush(pq_f, (dist_f[v], v))

        # Backward direction
        if pq_b:
            d, u = heapq.heappop(pq_b)
            if d > dist_b[u]:
                continue
            visited_b[u] = True
            if visited_f[u]:
                best = min(best, dist_f[u] + dist_b[u])
            for v, w in up_graph[u]:
                if rank[v] > rank[u] and dist_b[u] + w < dist_b[v]:
                    dist_b[v] = dist_b[u] + w
                    heapq.heappush(pq_b, (dist_b[v], v))

        # Early stop condition
        min_frontier = min(
            pq_f[0][0] if pq_f else INF,
            pq_b[0][0] if pq_b else INF
        )
        if min_frontier >= best:
            break

    return best if best != INF else None

# Example: upward graph with rank
up_graph = [
    [(1, 2), (2, 4)],  # 0
    [(3, 7)],          # 1
    [(3, 3)],          # 2
    []                 # 3
]
rank = [0, 1, 2, 3]
print(ch_query(up_graph, rank, 0, 3))
```

Output:

```
8
```

#### Why It Matters

- Ultra-fast queries, typically microseconds on large graphs
- No heuristics, 100% exact shortest paths
- Used in real-time GPS navigation, logistics optimization, map routing APIs
- Can be combined with ALT, turn penalties, or time-dependent weights

#### A Gentle Proof (Why It Works)

During preprocessing, every node $v$ is contracted with shortcuts preserving all shortest paths.
Each valid shortest path can be represented as an up–down path:

- ascending ranks (up), then descending ranks (down).

By running upward-only searches from both $s$ and $t$, we explore the up segments of all such paths.
The first meeting point $v$ with
$$
\text{dist}_f[v] + \text{dist}_b[v]
$$
minimal corresponds to the optimal path.

#### Try It Yourself

1. Visualize ranks and edges before and after contraction.
2. Compare number of visited nodes with full Dijkstra.
3. Combine with landmarks (ALT) for even faster queries.
4. Measure query times on grid vs road-like graphs.
5. Add path reconstruction by storing parent pointers.

#### Test Cases

| Graph            | Query (s,t)      | Result  | Nodes Visited   |
| ---------------- | ---------------- | ------- | --------------- |
| 0–1–2–3          | (0,3)            | 3 edges | 4 (Dijkstra: 4) |
| 0–1–2–3–4 (line) | (0,4)            | path=4  | 5 (Dijkstra: 5) |
| Grid (5×5)       | (corner, corner) | correct | ~20x fewer      |

#### Complexity

- Preprocessing: handled by CH build ($O(V \log V + E)$)
- Query: $O(\log V)$ (tiny frontier)
- Space: $O(V + E + \text{shortcuts})$
- Condition: Nonnegative weights, static graph

The CH Query Algorithm is the express lane of graph search —
prebuilt shortcuts and a smart hierarchy let it fly from source to target in a fraction of the time.

### 337 Bellman–Ford Queue Variant (Early Termination SPFA)

The Bellman–Ford Queue Variant, commonly known as SPFA (*Shortest Path Faster Algorithm*), improves upon the standard Bellman–Ford by using a queue to relax only active vertices, those whose distances were updated.
In practice, it's often much faster, though in the worst case it still runs in $O(VE)$.

It's a clever hybrid: Bellman–Ford's correctness, Dijkstra's selectivity.

#### What Problem Are We Solving?

We want to compute shortest paths from a single source $s$ in a weighted graph that may include negative edges (but no negative cycles):

$$
w(u, v) \in \mathbb{R}, \quad \text{and no negative cycles}
$$

The classic Bellman–Ford relaxes all edges $V-1$ times —
too slow when most nodes don't change often.

SPFA optimizes by:

- Tracking which nodes were updated last round,
- Pushing them into a queue,
- Relaxing only their outgoing edges.

#### How Does It Work (Plain Language)?

The algorithm uses a queue of "active" nodes.
When a node's distance improves, we enqueue it (if not already in the queue).
Each iteration pulls one node, relaxes its neighbors, and enqueues those whose distances improve.

It's like a wavefront of updates spreading only where needed.

| Step | Action                                                             |
| ---- | ------------------------------------------------------------------ |
| 1    | Initialize $\text{dist}[s] = 0$, others $\infty$                   |
| 2    | Push $s$ into queue                                                |
| 3    | While queue not empty:                                             |
|      | Pop $u$, relax $(u, v)$                                            |
|      | If $\text{dist}[u] + w(u,v) < \text{dist}[v]$, update and push $v$ |
| 4    | Optional: detect negative cycles                                   |

When every node leaves the queue with no new updates, we're done, early termination!

#### Tiny Code (C Example)

```c
#include <stdio.h>
#include <stdbool.h>
#include <limits.h>

#define MAXV 1000
#define INF 1000000000

typedef struct { int v, w; } Edge;
Edge graph[MAXV][MAXV];
int deg[MAXV], dist[MAXV];
bool in_queue[MAXV];
int queue[MAXV*10], front, back;

void spfa(int V, int s) {
    for (int i = 0; i < V; i++) dist[i] = INF, in_queue[i] = false;
    front = back = 0;
    dist[s] = 0;
    queue[back++] = s;
    in_queue[s] = true;

    while (front != back) {
        int u = queue[front++];
        if (front == MAXV*10) front = 0;
        in_queue[u] = false;

        for (int i = 0; i < deg[u]; i++) {
            int v = graph[u][i].v, w = graph[u][i].w;
            if (dist[u] + w < dist[v]) {
                dist[v] = dist[u] + w;
                if (!in_queue[v]) {
                    queue[back++] = v;
                    if (back == MAXV*10) back = 0;
                    in_queue[v] = true;
                }
            }
        }
    }
}
```

Python (Simplified)

```python
from collections import deque

def spfa(V, edges, src):
    graph = [[] for _ in range(V)]
    for u, v, w in edges:
        graph[u].append((v, w))

    dist = [float('inf')] * V
    inq = [False] * V
    dist[src] = 0

    dq = deque([src])
    inq[src] = True

    while dq:
        u = dq.popleft()
        inq[u] = False
        for v, w in graph[u]:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                if not inq[v]:
                    dq.append(v)
                    inq[v] = True
    return dist
```

#### Why It Matters

- Often faster than Bellman–Ford on sparse graphs
- Early termination when no further updates
- Handles negative edges safely
- Basis for Min-Cost Max-Flow (SPFA version)
- Popular in competitive programming and network routing

#### A Gentle Proof (Why It Works)

Bellman–Ford relies on edge relaxation propagating shortest paths through $V-1$ iterations.

SPFA dynamically schedules relaxations:

- Each vertex enters the queue only when $\text{dist}$ improves.
- Since each relaxation respects edge constraints, shortest paths converge after at most $V-1$ relaxations per vertex.

Thus, SPFA preserves correctness and can terminate earlier when convergence is reached.

Worst case (e.g. negative-weight grid), each vertex relaxes $O(V)$ times, so time is $O(VE)$.
But in practice, it's near $O(E)$.

#### Try It Yourself

1. Run on graphs with negative weights but no cycles.
2. Compare steps with Bellman–Ford, fewer relaxations.
3. Add a counter per node to detect negative cycles.
4. Use as core for Min-Cost Max-Flow.
5. Test dense vs sparse graphs, watch runtime difference.

#### Test Cases

| Graph                   | Edges        | Result     |
| ----------------------- | ------------ | ---------- |
| 0→1(2), 1→2(-1), 0→2(4) | no neg cycle | [0, 2, 1]  |
| 0→1(1), 1→2(2), 2→0(-4) | neg cycle    | detect     |
| 0→1(5), 0→2(2), 2→1(-3) | no cycle     | [0, -1, 2] |

#### Complexity

- Time: $O(VE)$ worst, often $O(E)$ average
- Space: $O(V + E)$
- Condition: Negative edges allowed, no negative cycles

The Bellman–Ford Queue Variant is the smart scheduler —
instead of looping blindly, it listens for updates and moves only where change happens.

### 338 Dijkstra with Early Stop

Dijkstra with Early Stop is a target-aware optimization of the classic Dijkstra's algorithm.
It leverages the fact that Dijkstra processes vertices in nondecreasing order of distance, so as soon as the target node is extracted from the priority queue, its shortest distance is final and the search can safely terminate.

This simple tweak can cut runtime dramatically for single-pair shortest path queries.

#### What Problem Are We Solving?

In standard Dijkstra's algorithm, the search continues until all reachable vertices have been settled, even if we only care about one destination $t$.
That's wasteful for point-to-point routing, where we only need $\text{dist}(s, t)$.

The Early Stop version stops immediately when $t$ is extracted from the priority queue:

$$
\text{dist}(t) = \text{final shortest distance}
$$

#### How Does It Work (Plain Language)?

Same setup as Dijkstra, but with an exit condition:

1. Initialize $\text{dist}[s] = 0$, all others $\infty$.
2. Push $(0, s)$ into priority queue.
3. While queue not empty:

   * Pop $(d, u)$ with smallest tentative distance.
   * If $u = t$: stop, we found the shortest path.
   * For each neighbor $(v, w)$:

     * If $\text{dist}[u] + w < \text{dist}[v]$, relax and push.

Because Dijkstra's invariant guarantees we always pop nodes in order of increasing distance,
the first time we pop $t$, we have found its true shortest distance.

| Step                 | Action           |
| -------------------- | ---------------- |
| Extract min node $u$ | Expand neighbors |
| If $u = t$           | Stop immediately |
| Else                 | Continue         |

No need to explore the entire graph.

#### Tiny Code (Python Example)

```python
import heapq

def dijkstra_early_stop(V, edges, s, t):
    graph = [[] for _ in range(V)]
    for u, v, w in edges:
        graph[u].append((v, w))
        graph[v].append((u, w))  # undirected

    INF = float('inf')
    dist = [INF] * V
    dist[s] = 0
    pq = [(0, s)]

    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        if u == t:
            return dist[t]
        for v, w in graph[u]:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                heapq.heappush(pq, (dist[v], v))
    return None  # unreachable
```

Example:

```python
edges = [
    (0, 1, 2),
    (1, 2, 3),
    (0, 3, 1),
    (3, 4, 4),
    (4, 2, 2)
]
print(dijkstra_early_stop(5, edges, 0, 2))
```

Output:

```
5
```

#### Why It Matters

- Early termination = fewer expansions = faster queries
- Ideal for point-to-point routing
- Combines well with A* (guided early stop) and ALT
- Simple to implement (just one `if` condition)

Used in:

- GPS navigation (city-to-city routes)
- Network routing (specific endpoints)
- Game AI pathfinding

#### A Gentle Proof (Why It Works)

Dijkstra's correctness relies on the fact that when a node is extracted from the heap,
its shortest distance is final (no smaller distance can appear later).

Thus, when $t$ is extracted:
$$
\text{dist}(t) = \min_{u \in V} \text{dist}(u)
$$
and any other unvisited vertex has $\text{dist}[v] \ge \text{dist}[t]$.

Therefore, stopping immediately when $u = t$ preserves correctness.

#### Try It Yourself

1. Compare runtime vs full Dijkstra on large graphs.
2. Visualize heap operations with and without early stop.
3. Apply to grid graphs (start vs goal corner).
4. Combine with A* to reduce visited nodes even more.
5. Test on disconnected graphs (target unreachable).

#### Test Cases

| Graph           | Source      | Target           | Result             | Notes      |
| --------------- | ----------- | ---------------- | ------------------ | ---------- |
| Chain 0–1–2–3   | 0           | 3                | 3 edges            | Stops at 3 |
| Star (0–others) | 0           | 4                | direct edge        | 1 step     |
| Grid            | (0,0)→(n,n) | Early stop saves | Many nodes skipped |            |

#### Complexity

- Time: $O(E \log V)$ (best-case ≪ full graph)
- Space: $O(V + E)$
- Condition: Nonnegative weights

Dijkstra with Early Stop is the sniper version —
it locks onto the target and halts the moment the job's done, saving every wasted move.

### 339 Goal-Directed Search

Goal-Directed Search is a general strategy for focusing graph exploration toward a specific target, rather than scanning the entire space.
It modifies shortest-path algorithms (like BFS or Dijkstra) by biasing expansions in the direction of the goal using geometry, landmarks, or precomputed heuristics.

When the bias is admissible (never overestimates cost), it still guarantees optimal paths while greatly reducing explored nodes.

#### What Problem Are We Solving?

In standard BFS or Dijkstra, exploration radiates uniformly outward, even in directions that clearly don't lead to the target.
For large graphs (grids, road maps), that's wasteful.

We need a way to steer the search toward the destination without losing correctness.

Formally, for source $s$ and target $t$, we want to find

$$
\text{dist}(s, t) = \min_{\text{path } P: s \to t} \sum_{(u,v)\in P} w(u,v)
$$

while visiting as few nodes as possible.

#### How Does It Work (Plain Language)?

Goal-directed search attaches a heuristic bias to each node's priority in the queue,
so nodes closer (or more promising) to the target are expanded earlier.

Typical scoring rule:

$$
f(v) = g(v) + \lambda \cdot h(v)
$$

where

- $g(v)$ = distance from $s$ so far,
- $h(v)$ = heuristic estimate from $v$ to $t$,
- $\lambda$ = bias factor (often $1$ for A*, or $<1$ for partial guidance).

Admissible heuristics ($h(v) \le \text{true distance}(v,t)$) preserve optimality.
Even simple directional heuristics, like Euclidean or Manhattan distance, can dramatically reduce search space.

| Heuristic Type | Definition                       | Use Case         |   |               |   |             |
| -------------- | -------------------------------- | ---------------- | - | ------------- | - | ----------- |
| Euclidean      | $\sqrt{(x_v-x_t)^2+(y_v-y_t)^2}$ | geometric graphs |   |               |   |             |
| Manhattan      | $                                | x_v-x_t          | + | y_v-y_t       | $ | grid worlds |
| Landmark (ALT) | $                                | d(L,t)-d(L,v)    | $ | road networks |   |             |

#### Tiny Code (Python Example)

```python
import heapq
import math

def goal_directed_search(V, edges, coords, s, t):
    graph = [[] for _ in range(V)]
    for u, v, w in edges:
        graph[u].append((v, w))
        graph[v].append((u, w))  # undirected

    def heuristic(v):
        x1, y1 = coords[v]
        x2, y2 = coords[t]
        return math.sqrt((x1 - x2)2 + (y1 - y2)2)

    dist = [float('inf')] * V
    dist[s] = 0
    pq = [(heuristic(s), s)]

    while pq:
        f, u = heapq.heappop(pq)
        if u == t:
            return dist[u]
        for v, w in graph[u]:
            g_new = dist[u] + w
            if g_new < dist[v]:
                dist[v] = g_new
                f_v = g_new + heuristic(v)
                heapq.heappush(pq, (f_v, v))
    return None
```

Example:

```python
coords = [(0,0), (1,0), (1,1), (2,1)]
edges = [(0,1,1), (1,2,1), (0,2,2), (2,3,1)]
print(goal_directed_search(4, edges, coords, 0, 3))
```

Output:

```
3
```

#### Why It Matters

- Fewer node expansions than unguided Dijkstra
- Naturally integrates with A*, ALT, CH, landmark heuristics
- Applicable to navigation, pathfinding, planning
- Easily adapted to grids, 3D spaces, or weighted networks

Used in:

- GPS navigation
- AI game agents
- Robotics (motion planning)

#### A Gentle Proof (Why It Works)

When $h(v)$ is admissible,
$$
h(v) \le \text{true distance}(v,t),
$$
then $f(v) = g(v) + h(v)$ never underestimates the cost of the optimal path through $v$.

Therefore, the first time the target $t$ is popped from the queue,
$\text{dist}(t)$ is guaranteed to be the true shortest distance.

If $h$ is also consistent, then
$$
h(u) \le w(u,v) + h(v),
$$
which ensures the priority order behaves like Dijkstra's, preserving monotonicity.

#### Try It Yourself

1. Run with $h=0$ → becomes normal Dijkstra.
2. Try different $\lambda$:

   * $\lambda=1$ → A*
   * $\lambda<1$ → softer guidance (more exploration).
3. Compare expansions with plain Dijkstra on a grid.
4. Visualize frontier growth, goal-directed forms a cone instead of a circle.
5. Test non-admissible $h$ (may speed up but lose optimality).

#### Test Cases

| Graph          | Heuristic | Result   | Nodes Visited |
| -------------- | --------- | -------- | ------------- |
| 4-node line    | Euclidean | 3        | fewer         |
| Grid 5×5       | Manhattan | Optimal  | ~½ nodes      |
| Zero heuristic | $h=0$     | Dijkstra | all nodes     |

#### Complexity

- Time: $O(E \log V)$ (fewer relaxations in practice)
- Space: $O(V)$
- Condition: Nonnegative weights, admissible $h(v)$

Goal-Directed Search is the compass-guided Dijkstra —
it still guarantees the shortest route, but marches confidently toward the goal instead of wandering in every direction.

### 340 Yen's K Shortest Paths

Yen's Algorithm finds not just the single shortest path, but the K shortest loopless paths between two nodes in a weighted directed graph.
It's a natural extension of Dijkstra's, instead of stopping at the first solution, it systematically explores path deviations to discover the next-best routes in ascending order of total length.

Used widely in network routing, multi-route planning, and alternatives in navigation systems.

#### What Problem Are We Solving?

Given a graph $G = (V, E)$ with nonnegative edge weights, a source $s$, and a target $t$,
we want to compute the first $K$ distinct shortest paths:

$$
P_1, P_2, \dots, P_K
$$

ordered by total weight:

$$
\text{len}(P_1) \le \text{len}(P_2) \le \cdots \le \text{len}(P_K)
$$

Each path must be simple (no repeated nodes).

#### How Does It Work (Plain Language)?

Yen's Algorithm builds upon Dijkstra's algorithm and the deviation path concept.

1. Compute the shortest path $P_1$ using Dijkstra.
2. For each $i = 2, \dots, K$:

   * Let $P_{i-1}$ be the previous path.
   * For each node (spur node) in $P_{i-1}$:

     * Split the path into root path (prefix up to spur node).
     * Temporarily remove:

       * Any edge that would recreate a previously found path.
       * Any node in the root path (except spur node) to prevent cycles.
     * Run Dijkstra from spur node to $t$.
     * Combine root path + spur path → candidate path.
   * Among all candidates, choose the shortest one not yet selected.
   * Add it as $P_i$.

Repeat until $K$ paths are found or no candidates remain.

| Step | Operation               | Purpose              |
| ---- | ----------------------- | -------------------- |
| 1    | $P_1$ = Dijkstra(s, t)  | base path            |
| 2    | Deviation from prefixes | explore alternatives |
| 3    | Collect candidates      | min-heap             |
| 4    | Select shortest         | next path            |

#### Tiny Code (Python Example)

A simplified version for small graphs:

```python
import heapq

def dijkstra(graph, s, t):
    pq = [(0, s, [s])]
    seen = set()
    while pq:
        d, u, path = heapq.heappop(pq)
        if u == t:
            return (d, path)
        if u in seen: 
            continue
        seen.add(u)
        for v, w in graph[u]:
            if v not in seen:
                heapq.heappush(pq, (d + w, v, path + [v]))
    return None

def yen_k_shortest_paths(graph, s, t, K):
    A = []
    B = []
    first = dijkstra(graph, s, t)
    if not first:
        return A
    A.append(first)
    for k in range(1, K):
        prev_path = A[k - 1][1]
        for i in range(len(prev_path) - 1):
            spur_node = prev_path[i]
            root_path = prev_path[:i + 1]
            removed_edges = []
            # Temporarily remove edges
            for d, p in A:
                if p[:i + 1] == root_path and i + 1 < len(p):
                    u = p[i]
                    v = p[i + 1]
                    for e in graph[u]:
                        if e[0] == v:
                            graph[u].remove(e)
                            removed_edges.append((u, e))
                            break
            spur = dijkstra(graph, spur_node, t)
            if spur:
                total_path = root_path[:-1] + spur[1]
                total_cost = sum(graph[u][v][1] for u, v in zip(total_path, total_path[1:])) if False else spur[0] + sum(0 for _ in root_path)
                if (total_cost, total_path) not in B:
                    B.append((total_cost, total_path))
            for u, e in removed_edges:
                graph[u].append(e)
        if not B:
            break
        B.sort(key=lambda x: x[0])
        A.append(B.pop(0))
    return A
```

Example:

```python
graph = {
    0: [(1, 1), (2, 2)],
    1: [(2, 1), (3, 3)],
    2: [(3, 1)],
    3: []
}
print(yen_k_shortest_paths(graph, 0, 3, 3))
```

Output:

```
[(3, [0, 1, 2, 3]), (4, [0, 2, 3]), (5, [0, 1, 3])]
```

#### Why It Matters

- Provides multiple distinct routes, not just one
- Used in multi-path routing, backup planning, logistics optimization
- Guarantees simple paths (no loops)
- Reuses Dijkstra, easy to integrate with existing solvers

#### A Gentle Proof (Why It Works)

Every $P_i$ is generated by deviating from earlier paths at some spur node, ensuring uniqueness.
Dijkstra ensures each spur path is locally shortest.
Since all candidates are stored and selected by global order, the sequence $P_1, P_2, \dots, P_K$ is globally sorted by total cost.

Thus:
$$
\text{len}(P_1) \le \text{len}(P_2) \le \cdots \le \text{len}(P_K)
$$

and each path is simple.

#### Try It Yourself

1. Generate the 3 shortest paths in a small network.
2. Compare with Eppstein's algorithm (faster for large K).
3. Remove edges dynamically and re-run, observe reordering.
4. Visualize path tree (root = $s$, branches = deviations).
5. Adapt for undirected or bi-directional graphs.

#### Test Cases

| Graph         | $s$        | $t$ | $K$              | Paths                            |
| ------------- | ---------- | --- | ---------------- | -------------------------------- |
| Line 0–1–2–3  | 0          | 3   | 2                | one main, one deviation          |
| Triangle      | 0          | 2   | 3                | 0–1–2, 0–2, 0–1–0–2 (if allowed) |
| Weighted grid | start–goal | 3   | alternate routes |                                  |

#### Complexity

- Time: $O(K \cdot V \cdot (E + V \log V))$
- Space: $O(KV)$
- Condition: Nonnegative weights, finite paths

Yen's Algorithm is the explorer's Dijkstra, it doesn't stop at the first road found but keeps discovering better detours, one deviation at a time.

## Section 35. Minimum Spanning Trees 

### 341 Kruskal's Algorithm

Kruskal's algorithm builds a minimum spanning tree (MST) by repeatedly adding the lightest edge that does not create a cycle.
It uses a disjoint set union (DSU) data structure to track components efficiently.

#### What Problem Are We Solving?

Given a connected, undirected, weighted graph $G=(V,E,w)$, find a subset $T\subseteq E$ such that:

- $T$ connects all vertices,
- $|T|=|V|-1$,
- $\sum_{e\in T}w(e)$ is minimized.

If the graph is not connected, Kruskal builds a minimum spanning forest.

#### How Does It Work (Plain Language)?

1. Sort all edges by nondecreasing weight.
2. Initialize DSU with each vertex in its own set.
3. Scan edges in order. For edge $(u,v)$:

   * If $u$ and $v$ are in different sets, add the edge to the MST and union their sets.
   * Otherwise, skip it to avoid a cycle.
4. Stop when you have $|V|-1$ edges.

| Step          | Action               | DSU effect               |
| ------------- | -------------------- | ------------------------ |
| Sort edges    | Light to heavy       | None                     |
| Check $(u,v)$ | If find(u) ≠ find(v) | union(u,v) and keep edge |
| Cycle case    | If find(u) = find(v) | skip edge                |

#### Tiny Code

C (DSU + Kruskal)

```c
#include <stdio.h>
#include <stdlib.h>

typedef struct { int u, v; int w; } Edge;

typedef struct {
    int *p, *r;
    int n;
} DSU;

DSU make_dsu(int n){
    DSU d; d.n = n;
    d.p = malloc(n * sizeof(int));
    d.r = malloc(n * sizeof(int));
    for(int i=0;i<n;i++){ d.p[i]=i; d.r[i]=0; }
    return d;
}
int find(DSU *d, int x){
    if(d->p[x]!=x) d->p[x]=find(d, d->p[x]);
    return d->p[x];
}
void unite(DSU *d, int a, int b){
    a=find(d,a); b=find(d,b);
    if(a==b) return;
    if(d->r[a]<d->r[b]) d->p[a]=b;
    else if(d->r[b]<d->r[a]) d->p[b]=a;
    else { d->p[b]=a; d->r[a]++; }
}
int cmp_edge(const void* A, const void* B){
    Edge *a=(Edge*)A, *b=(Edge*)B;
    return a->w - b->w;
}

int main(void){
    int n, m;
    scanf("%d %d", &n, &m);
    Edge *E = malloc(m * sizeof(Edge));
    for(int i=0;i<m;i++) scanf("%d %d %d", &E[i].u, &E[i].v, &E[i].w);

    qsort(E, m, sizeof(Edge), cmp_edge);
    DSU d = make_dsu(n);

    int taken = 0;
    long long cost = 0;
    for(int i=0;i<m && taken < n-1;i++){
        int a = find(&d, E[i].u), b = find(&d, E[i].v);
        if(a != b){
            unite(&d, a, b);
            cost += E[i].w;
            taken++;
        }
    }
    if(taken != n-1) { printf("Graph not connected\n"); }
    else { printf("MST cost: %lld\n", cost); }
    return 0;
}
```

Python (DSU + Kruskal)

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
        a, b = self.find(a), self.find(b)
        if a == b: return False
        if self.r[a] < self.r[b]: self.p[a] = b
        elif self.r[b] < self.r[a]: self.p[b] = a
        else: self.p[b] = a; self.r[a] += 1
        return True

def kruskal(n, edges):
    edges = sorted(edges)
    dsu = DSU(n)
    cost = 0
    mst = []
    for w, u, v in edges:
        if dsu.union(u, v):
            cost += w
            mst.append((u, v, w))
            if len(mst) == n-1:
                break
    return cost, mst

n = 4
edges = [(1,0,1),(4,0,2),(3,1,2),(2,1,3),(5,2,3)]
print(kruskal(n, edges))
```

#### Why It Matters

- Simple and fast with sorting plus DSU.
- Works well on sparse graphs.
- Produces an MST that minimizes total edge weight.
- Easy to adapt for minimum spanning forest on disconnected graphs.

#### A Gentle Proof (Why It Works)

Kruskal relies on the cut property:
Let $S \subset V$ be any proper subset and consider the cut $(S, V \setminus S)$.
The lightest edge crossing the cut is safe to include in some MST.

Sorting edges by weight and always taking the next lightest that connects two different components is equivalent to repeatedly applying the cut property to the partition defined by current DSU components.
This never creates a cycle and never excludes the possibility of an optimal MST.
By induction on the number of chosen edges, the final set is an MST.

#### Try It Yourself

1. Generate random sparse graphs and compare Kruskal vs Prim.
2. Remove an edge from the MST and recompute to observe changes.
3. Force ties in edge weights and confirm multiple valid MSTs.
4. Run on a disconnected graph to obtain a minimum spanning forest.
5. Log chosen edges to visualize the growth of components.

#### Test Cases

| Graph        | Edges $(u,v,w)$                             | MST Edges         | Cost       |
| ------------ | ------------------------------------------- | ----------------- | ---------- |
| Triangle     | (0,1,1), (1,2,2), (0,2,3)                   | (0,1,1), (1,2,2)  | 3          |
| Square       | (0,1,1), (1,2,1), (2,3,1), (3,0,1), (0,2,2) | any 3 of weight 1 | 3          |
| Disconnected | two separate triangles                      | MST per component | sum of two |

#### Complexity

- Time: $O(E \log E)$ for sorting plus almost linear DSU, which is $O(E \alpha(V))$. Usually written as $O(E \log E)$ or $O(E \log V)$.
- Space: $O(V + E)$.
- Output size: $|V|-1$ edges for a connected graph.

Kruskal is the sort then stitch approach to MSTs. Sort edges globally, then stitch components locally with DSU until the tree locks into place.

### 342 Prim's Algorithm (Heap)

Prim's algorithm builds a minimum spanning tree (MST) by growing a connected subtree one vertex at a time, always choosing the lightest edge that connects a vertex inside the tree to one outside.
It's a greedy algorithm, often implemented with a min-heap for efficiency.

#### What Problem Are We Solving?

Given a connected, undirected, weighted graph $G=(V,E,w)$, find a subset of edges $T\subseteq E$ such that:

- $T$ connects all vertices,
- $|T|=|V|-1$,
- $\sum_{e\in T}w(e)$ is minimized.

Prim's algorithm grows the MST from a single seed vertex.

#### How Does It Work (Plain Language)?

1. Choose any starting vertex $s$.
2. Initialize all vertices with $\text{key}[v]=\infty$, except $\text{key}[s]=0$.
3. Use a min-heap (priority queue) keyed by edge weight.
4. Repeatedly extract the vertex $u$ with the smallest key (edge weight):

   * Mark $u$ as part of the MST.
   * For each neighbor $v$ of $u$:

     * If $v$ is not yet in the MST and $w(u,v)<\text{key}[v]$, update $\text{key}[v]=w(u,v)$ and set $\text{parent}[v]=u$.
5. Continue until all vertices are included.

| Step            | Action               | Description       |
| --------------- | -------------------- | ----------------- |
| Initialize      | Choose start vertex  | $\text{key}[s]=0$ |
| Extract min     | Add lightest edge    | Expands MST       |
| Relax neighbors | Update cheaper edges | Maintain frontier |

#### Tiny Code

Python (Min-Heap Version)

```python
import heapq

def prim_mst(V, edges, start=0):
    graph = [[] for _ in range(V)]
    for u, v, w in edges:
        graph[u].append((w, v))
        graph[v].append((w, u))  # undirected

    visited = [False] * V
    pq = [(0, start, -1)]  # (weight, vertex, parent)
    mst_edges = []
    total_cost = 0

    while pq:
        w, u, parent = heapq.heappop(pq)
        if visited[u]:
            continue
        visited[u] = True
        total_cost += w
        if parent != -1:
            mst_edges.append((parent, u, w))
        for weight, v in graph[u]:
            if not visited[v]:
                heapq.heappush(pq, (weight, v, u))
    return total_cost, mst_edges

edges = [(0,1,2),(0,2,3),(1,2,1),(1,3,4),(2,3,5)]
cost, mst = prim_mst(4, edges)
print(cost, mst)
```

Output:

```
7 [(0,1,2), (1,2,1), (1,3,4)]
```

#### Why It Matters

- Suitable for dense graphs (especially with adjacency lists and heaps).
- Builds MST incrementally (like Dijkstra).
- Great for online construction where the tree must stay connected.
- Easier to integrate with adjacency matrix for small graphs.

#### A Gentle Proof (Why It Works)

Prim's algorithm obeys the cut property:
At each step, consider the cut between the current MST set $S$ and the remaining vertices $V \setminus S$.
The lightest edge crossing that cut is always safe to include.
By repeatedly choosing the minimum such edge, Prim's maintains a valid MST prefix.
When all vertices are added, the resulting tree is minimal.

#### Try It Yourself

1. Run Prim's on a dense graph vs Kruskal, compare edge choices.
2. Visualize the growing frontier.
3. Try with adjacency matrix (without heap).
4. Test on disconnected graph, each component forms its own tree.
5. Replace heap with a simple array to see $O(V^2)$ version.

#### Test Cases

| Graph    | Edges $(u,v,w)$                    | MST Edges                 | Cost |
| -------- | ---------------------------------- | ------------------------- | ---- |
| Triangle | (0,1,1), (1,2,2), (0,2,3)          | (0,1,1), (1,2,2)          | 3    |
| Square   | (0,1,2), (1,2,3), (2,3,1), (3,0,4) | (2,3,1), (0,1,2), (1,2,3) | 6    |
| Line     | (0,1,5), (1,2,1), (2,3,2)          | (1,2,1), (2,3,2), (0,1,5) | 8    |

#### Complexity

- Time: $O(E\log V)$ with heap, $O(V^2)$ with array.
- Space: $O(V + E)$
- Output size: $|V|-1$ edges

Prim's is the grow from seed approach to MSTs. It builds the tree step by step, always pulling the next lightest edge from the frontier.

### 343 Prim's Algorithm (Adjacency Matrix)

This is the array-based version of Prim's algorithm, optimized for dense graphs.
Instead of using a heap, it directly scans all vertices to find the next minimum key vertex at each step.
The logic is identical to Prim's heap version but trades priority queues for simple loops.

#### What Problem Are We Solving?

Given a connected, undirected, weighted graph $G=(V,E,w)$, represented by an adjacency matrix $W$, we want to construct an MST, a subset of edges that:

- connects all vertices,
- contains $|V|-1$ edges,
- and minimizes total weight $\sum w(e)$.

The adjacency matrix form simplifies edge lookups and is ideal for dense graphs, where $E\approx V^2$.

#### How Does It Work (Plain Language)?

1. Start from an arbitrary vertex (say $0$).
2. Initialize $\text{key}[v]=\infty$ for all vertices, except $\text{key}[0]=0$.
3. Maintain a set $\text{inMST}[v]$ marking vertices already included.
4. Repeat $V-1$ times:

   * Choose the vertex $u$ not yet in MST with the smallest $\text{key}[u]$.
   * Add $u$ to MST.
   * For each vertex $v$, if $W[u][v]$ is smaller than $\text{key}[v]$, update it and record parent $v\gets u$.

At each iteration, one vertex joins the MST, the one connected by the lightest edge to the existing set.

| Step        | Operation         | Description              |
| ----------- | ----------------- | ------------------------ |
| Initialize  | $\text{key}[0]=0$ | start from vertex 0      |
| Extract min | find smallest key | next vertex to add       |
| Update      | relax edges       | update keys of neighbors |

#### Tiny Code

C (Adjacency Matrix Version)

```c
#include <stdio.h>
#include <limits.h>
#include <stdbool.h>

#define INF 1000000000
#define N 100

int minKey(int key[], bool inMST[], int n) {
    int min = INF, idx = -1;
    for (int v = 0; v < n; v++)
        if (!inMST[v] && key[v] < min)
            min = key[v], idx = v;
    return idx;
}

void primMatrix(int graph[N][N], int n) {
    int key[N], parent[N];
    bool inMST[N];
    for (int i = 0; i < n; i++)
        key[i] = INF, inMST[i] = false;
    key[0] = 0, parent[0] = -1;

    for (int count = 0; count < n - 1; count++) {
        int u = minKey(key, inMST, n);
        inMST[u] = true;
        for (int v = 0; v < n; v++)
            if (graph[u][v] && !inMST[v] && graph[u][v] < key[v]) {
                parent[v] = u;
                key[v] = graph[u][v];
            }
    }
    int total = 0;
    for (int i = 1; i < n; i++) {
        printf("%d - %d: %d\n", parent[i], i, graph[i][parent[i]]);
        total += graph[i][parent[i]];
    }
    printf("MST cost: %d\n", total);
}

int main() {
    int n = 5;
    int graph[N][N] = {
        {0,2,0,6,0},
        {2,0,3,8,5},
        {0,3,0,0,7},
        {6,8,0,0,9},
        {0,5,7,9,0}
    };
    primMatrix(graph, n);
    return 0;
}
```

Output:

```
0 - 1: 2
1 - 2: 3
0 - 3: 6
1 - 4: 5
MST cost: 16
```

Python Version

```python
def prim_matrix(graph):
    V = len(graph)
    key = [float('inf')] * V
    parent = [-1] * V
    in_mst = [False] * V
    key[0] = 0

    for _ in range(V - 1):
        u = min((key[v], v) for v in range(V) if not in_mst[v])[1]
        in_mst[u] = True
        for v in range(V):
            if graph[u][v] != 0 and not in_mst[v] and graph[u][v] < key[v]:
                key[v] = graph[u][v]
                parent[v] = u

    edges = [(parent[i], i, graph[i][parent[i]]) for i in range(1, V)]
    cost = sum(w for _, _, w in edges)
    return cost, edges

graph = [
    [0,2,0,6,0],
    [2,0,3,8,5],
    [0,3,0,0,7],
    [6,8,0,0,9],
    [0,5,7,9,0]
]
print(prim_matrix(graph))
```

Output:

```
(16, [(0,1,2), (1,2,3), (0,3,6), (1,4,5)])
```

#### Why It Matters

- Simple to implement with adjacency matrices.
- Best for dense graphs where $E \approx V^2$.
- Avoids the complexity of heaps.
- Easy to visualize and debug for classroom or teaching use.

#### A Gentle Proof (Why It Works)

Like the heap-based version, this variant relies on the cut property:
At each step, the chosen edge connects a vertex inside the tree to one outside with minimum weight, so it is always safe.
Each iteration expands the MST without cycles, and after $V-1$ iterations, all vertices are connected.

#### Try It Yourself

1. Run on a complete graph, expect $V-1$ smallest edges.
2. Modify weights, see how edge choices change.
3. Compare with heap-based Prim on runtime as $V$ grows.
4. Implement in $O(V^2)$ and confirm complexity experimentally.
5. Test on disconnected graph, see where algorithm stops.

#### Test Cases

| Graph                 | MST Edges                          | Cost                     |            |
| --------------------- | ---------------------------------- | ------------------------ | ---------- |
| 5-node matrix         | (0,1,2), (1,2,3), (0,3,6), (1,4,5) | 16                       |            |
| 3-node triangle       | (0,1,1), (1,2,2)                   | 3                        |            |
| Dense complete 4-node | 6 edges                            | chooses lightest 3 edges | sum of min |

#### Complexity

- Time: $O(V^2)$
- Space: $O(V^2)$ (matrix)
- Output size: $|V|-1$ edges

Prim (Adjacency Matrix) is the classic dense-graph version, it trades speed for simplicity and predictable access time.

### 344 Borůvka's Algorithm

Borůvka's algorithm is one of the earliest MST algorithms (1926), designed to build the minimum spanning tree by repeatedly connecting each component with its cheapest outgoing edge.
It operates in phases, merging components until a single tree remains.

#### What Problem Are We Solving?

Given a connected, undirected, weighted graph $G = (V, E, w)$, we want to find a minimum spanning tree (MST), a subset $T \subseteq E$ such that:

- $T$ connects all vertices,
- $|T| = |V| - 1$,
- $\sum_{e \in T} w(e)$ is minimized.

Borůvka's approach grows multiple subtrees in parallel, making it highly suitable for parallel or distributed computation.

#### How Does It Work (Plain Language)?

The algorithm works in rounds. Each round connects every component to another via its lightest outgoing edge.

1. Start with each vertex as its own component.
2. For each component, find the minimum-weight edge that connects it to another component.
3. Add all these edges to the MST, they are guaranteed safe (by the cut property).
4. Merge components connected by these edges.
5. Repeat until only one component remains.

Each round at least halves the number of components, so the algorithm finishes in $O(\log V)$ phases.

| Step | Operation                        | Description            |
| ---- | -------------------------------- | ---------------------- |
| 1    | Initialize components            | each vertex alone      |
| 2    | Find cheapest edge per component | lightest outgoing      |
| 3    | Add edges                        | merge components       |
| 4    | Repeat                           | until single component |

#### Tiny Code

Python (DSU Version)

```python
class DSU:
    def __init__(self, n):
        self.p = list(range(n))
        self.r = [0] * n
    def find(self, x):
        if self.p[x] != x:
            self.p[x] = self.find(self.p[x])
        return self.p[x]
    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a == b: return False
        if self.r[a] < self.r[b]: self.p[a] = b
        elif self.r[a] > self.r[b]: self.p[b] = a
        else: self.p[b] = a; self.r[a] += 1
        return True

def boruvka_mst(V, edges):
    dsu = DSU(V)
    mst = []
    total_cost = 0
    components = V

    while components > 1:
        cheapest = [-1] * V
        for i, (u, v, w) in enumerate(edges):
            set_u = dsu.find(u)
            set_v = dsu.find(v)
            if set_u != set_v:
                if cheapest[set_u] == -1 or edges[cheapest[set_u]][2] > w:
                    cheapest[set_u] = i
                if cheapest[set_v] == -1 or edges[cheapest[set_v]][2] > w:
                    cheapest[set_v] = i
        for i in range(V):
            if cheapest[i] != -1:
                u, v, w = edges[cheapest[i]]
                if dsu.union(u, v):
                    mst.append((u, v, w))
                    total_cost += w
                    components -= 1
    return total_cost, mst

edges = [(0,1,1), (0,2,3), (1,2,1), (1,3,4), (2,3,5)]
print(boruvka_mst(4, edges))
```

Output:

```
(6, [(0,1,1), (1,2,1), (1,3,4)])
```

#### Why It Matters

- Naturally parallelizable (each component acts independently).
- Simple and elegant, based on repeated application of the cut property.
- Ideal for sparse graphs and distributed systems.
- Often used in hybrid MST algorithms (combining with Kruskal/Prim).

#### A Gentle Proof (Why It Works)

By the cut property, for each component $C$, the cheapest edge leaving $C$ is always safe to include in the MST.
Since edges are chosen simultaneously across all components, and no cycles are created within a single phase (components merge only across cuts), each round preserves correctness.
After each phase, components merge, and the process repeats until all are unified.

Each iteration reduces the number of components by at least half, ensuring $O(\log V)$ phases.

#### Try It Yourself

1. Run on a small graph and trace phases manually.
2. Compare with Kruskal's sorted-edge approach.
3. Add parallel logging to visualize simultaneous merges.
4. Observe how components shrink exponentially.
5. Mix with Kruskal: use Borůvka until few components remain, then switch.

#### Test Cases

| Graph        | Edges $(u,v,w)$                 | MST                     | Cost          |
| ------------ | ------------------------------- | ----------------------- | ------------- |
| Triangle     | (0,1,1),(1,2,2),(0,2,3)         | (0,1,1),(1,2,2)         | 3             |
| Square       | (0,1,1),(1,2,2),(2,3,1),(3,0,2) | (0,1,1),(2,3,1),(1,2,2) | 4             |
| Dense 4-node | 6 edges                         | builds MST in 2 phases  | verified cost |

#### Complexity

- Phases: $O(\log V)$
- Time per phase: $O(E)$
- Total Time: $O(E \log V)$
- Space: $O(V + E)$

Borůvka's algorithm is the parallel grow-and-merge strategy for MSTs, every component reaches out through its lightest edge until the whole graph becomes one connected tree.

### 345 Reverse-Delete MST

The Reverse-Delete Algorithm builds a minimum spanning tree (MST) by starting with the full graph and repeatedly removing edges, but only when their removal does not disconnect the graph.
It's the conceptual mirror image of Kruskal's algorithm.

#### What Problem Are We Solving?

Given a connected, undirected, weighted graph $G = (V, E, w)$, we want to find an MST —
a spanning tree that connects all vertices with minimum total weight.

Instead of adding edges like Kruskal, we start with all edges and delete them one by one,
making sure the graph remains connected after each deletion.

#### How Does It Work (Plain Language)?

1. Sort all edges by decreasing weight.
2. Initialize the working graph $T = G$.
3. For each edge $(u,v)$ in that order:

   * Temporarily remove $(u,v)$ from $T$.
   * If $u$ and $v$ are still connected in $T$, permanently delete the edge (it's not needed).
   * Otherwise, restore it (it's essential).
4. When all edges are processed, $T$ is the MST.

This approach ensures only indispensable edges remain, forming a spanning tree of minimal total weight.

| Step | Action                | Description            |   |   |   |     |
| ---- | --------------------- | ---------------------- | - | - | - | --- |
| 1    | Sort edges descending | heavy edges first      |   |   |   |     |
| 2    | Remove edge           | test connectivity      |   |   |   |     |
| 3    | Keep if needed        | if removal disconnects |   |   |   |     |
| 4    | Stop                  | when $                 | T | = | V | -1$ |

#### Tiny Code

Python (Using DFS for Connectivity Check)

```python
def dfs(graph, start, visited):
    visited.add(start)
    for v, _ in graph[start]:
        if v not in visited:
            dfs(graph, v, visited)

def is_connected(graph, u, v):
    visited = set()
    dfs(graph, u, visited)
    return v in visited

def reverse_delete_mst(V, edges):
    edges = sorted(edges, key=lambda x: x[2], reverse=True)
    graph = {i: [] for i in range(V)}
    for u, v, w in edges:
        graph[u].append((v, w))
        graph[v].append((u, w))

    mst_cost = sum(w for _, _, w in edges)

    for u, v, w in edges:
        # remove edge
        graph[u] = [(x, wx) for x, wx in graph[u] if x != v]
        graph[v] = [(x, wx) for x, wx in graph[v] if x != u]

        if is_connected(graph, u, v):
            mst_cost -= w  # edge not needed
        else:
            # restore edge
            graph[u].append((v, w))
            graph[v].append((u, w))

    mst_edges = []
    visited = set()
    def collect(u):
        visited.add(u)
        for v, w in graph[u]:
            if (v, w) not in visited:
                mst_edges.append((u, v, w))
                if v not in visited:
                    collect(v)
    collect(0)
    return mst_cost, mst_edges

edges = [(0,1,1), (0,2,2), (1,2,3), (1,3,4), (2,3,5)]
print(reverse_delete_mst(4, edges))
```

Output:

```
(7, [(0,1,1), (0,2,2), (1,3,4)])
```

#### Why It Matters

- Simple dual perspective to Kruskal's algorithm.
- Demonstrates the cycle property:

  > In any cycle, the edge with the largest weight cannot be in an MST.
- Good for teaching proofs and conceptual understanding.
- Can be used for verifying MSTs or constructing them in reverse.

#### A Gentle Proof (Why It Works)

The cycle property states:

> For any cycle in a graph, the edge with the largest weight cannot belong to the MST.

By sorting edges in descending order and removing each maximum-weight edge that lies in a cycle (i.e., when $u$ and $v$ remain connected without it), we eliminate all non-MST edges.
When no such edge remains, the result is an MST.

Since each deletion preserves connectivity, the final graph is a spanning tree with minimal total weight.

#### Try It Yourself

1. Run on a triangle graph, see heaviest edge removed.
2. Compare deletion order with Kruskal's addition order.
3. Visualize graph at each step.
4. Replace DFS with BFS or Union-Find for speed.
5. Use to verify MST output from another algorithm.

#### Test Cases

| Graph    | Edges $(u,v,w)$                 | MST                     | Cost |
| -------- | ------------------------------- | ----------------------- | ---- |
| Triangle | (0,1,1),(1,2,2),(0,2,3)         | (0,1,1),(1,2,2)         | 3    |
| Square   | (0,1,1),(1,2,2),(2,3,3),(3,0,4) | (0,1,1),(1,2,2),(2,3,3) | 6    |
| Line     | (0,1,2),(1,2,1),(2,3,3)         | all edges               | 6    |

#### Complexity

- Time: $O(E(E+V))$ with naive DFS (each edge removal checks connectivity).
  With Union-Find optimizations or precomputed structures, it can be reduced.
- Space: $O(V + E)$.

Reverse-Delete is the subtract instead of add view of MSTs, peel away heavy edges until only the essential light structure remains.

### 346 MST via Dijkstra Trick

This variant constructs a Minimum Spanning Tree (MST) using a Dijkstra-like process, repeatedly expanding the tree by selecting the lightest edge connecting any vertex inside the tree to one outside.
It's essentially Prim's algorithm recast through Dijkstra's lens, showing the deep parallel between shortest-path and spanning-tree growth.

#### What Problem Are We Solving?

Given a connected, undirected, weighted graph $G=(V,E,w)$ with nonnegative edge weights, we want an MST —
a subset $T \subseteq E$ that connects all vertices with
$$
|T| = |V| - 1, \quad \text{and} \quad \sum_{e \in T} w(e) \text{ minimized.}
$$

While Dijkstra's algorithm builds shortest-path trees based on *path cost*, this version builds an MST by using edge weights directly as "distances" from the current tree.

#### How Does It Work (Plain Language)?

This is Prim's algorithm in disguise:
Instead of tracking the shortest path from a root, we track the lightest edge connecting each vertex to the growing tree.

1. Initialize all vertices with $\text{key}[v]=\infty$, except start vertex $s$ with $\text{key}[s]=0$.
2. Use a priority queue (min-heap) keyed by $\text{key}[v]$.
3. Repeatedly extract vertex $u$ with smallest key.
4. For each neighbor $v$:

   * If $v$ not yet in tree and $w(u,v)<\text{key}[v]$,
     update $\text{key}[v]=w(u,v)$ and record parent.
5. Repeat until all vertices are included.

Unlike Dijkstra, we do not sum edge weights, we only care about the minimum edge to reach each vertex.

| Step            | Dijkstra                                   | MST via Dijkstra Trick                        |
| --------------- | ------------------------------------------ | --------------------------------------------- |
| Distance update | $\text{dist}[v] = \text{dist}[u] + w(u,v)$ | $\text{key}[v] = \min(\text{key}[v], w(u,v))$ |
| Priority        | Path cost                                  | Edge cost                                     |
| Goal            | Shortest path                              | MST                                           |

#### Tiny Code

Python (Heap Implementation)

```python
import heapq

def mst_dijkstra_trick(V, edges, start=0):
    graph = [[] for _ in range(V)]
    for u, v, w in edges:
        graph[u].append((v, w))
        graph[v].append((u, w))

    in_tree = [False] * V
    key = [float('inf')] * V
    parent = [-1] * V
    key[start] = 0
    pq = [(0, start)]

    total_cost = 0
    while pq:
        w, u = heapq.heappop(pq)
        if in_tree[u]:
            continue
        in_tree[u] = True
        total_cost += w

        for v, wt in graph[u]:
            if not in_tree[v] and wt < key[v]:
                key[v] = wt
                parent[v] = u
                heapq.heappush(pq, (wt, v))

    mst_edges = [(parent[v], v, key[v]) for v in range(V) if parent[v] != -1]
    return total_cost, mst_edges

edges = [(0,1,2),(0,2,3),(1,2,1),(1,3,4),(2,3,5)]
print(mst_dijkstra_trick(4, edges))
```

Output:

```
(7, [(0,1,2), (1,2,1), (1,3,4)])
```

#### Why It Matters

- Demonstrates the duality between MST and shortest-path search.
- Provides a conceptual bridge between Prim and Dijkstra.
- Useful for teaching and algorithmic unification.
- Intuitive when already familiar with Dijkstra's structure.

#### A Gentle Proof (Why It Works)

At each iteration, we maintain a set $S$ of vertices already in the MST.
The cut property guarantees that the lightest edge connecting $S$ to $V\setminus S$ is always safe to include.

By storing these edge weights in a priority queue and always selecting the smallest,
we exactly follow this property, thus constructing an MST incrementally.

The algorithm stops when all vertices are included, yielding an MST.

#### Try It Yourself

1. Compare line by line with Dijkstra, only one change in relaxation.
2. Run on complete graphs, observe star-like MSTs.
3. Try graphs with multiple equal edges, see tie-handling.
4. Replace heap with array, check $O(V^2)$ version.
5. Visualize with frontier highlighting, edges instead of distances.

#### Test Cases

| Graph    | Edges $(u,v,w)$                 | MST                     | Cost |
| -------- | ------------------------------- | ----------------------- | ---- |
| Triangle | (0,1,1),(1,2,2),(0,2,3)         | (0,1,1),(1,2,2)         | 3    |
| Square   | (0,1,2),(1,2,3),(2,3,1),(3,0,4) | (2,3,1),(0,1,2),(1,2,3) | 6    |
| Line     | (0,1,5),(1,2,2),(2,3,1)         | all edges               | 8    |

#### Complexity

- Time: $O(E \log V)$ (heap) or $O(V^2)$ (array)
- Space: $O(V + E)$
- Output size: $|V|-1$ edges

MST via Dijkstra Trick is Prim's algorithm reimagined, it replaces distance summation with edge minimization, proving how two classic graph ideas share one greedy heart.

### 347 Dynamic MST Maintenance

Dynamic MST maintenance addresses the question:
how can we update an MST efficiently when the underlying graph changes, edges are added, removed, or their weights change, without rebuilding from scratch?

This problem arises in systems where the graph evolves over time, such as networks, road maps, or real-time optimization systems.

#### What Problem Are We Solving?

Given a graph $G=(V,E,w)$ and its MST $T$, we want to maintain $T$ under updates:

- Edge Insertion: add new edge $(u,v,w)$
- Edge Deletion: remove existing edge $(u,v)$
- Edge Weight Update: change weight $w(u,v)$

Naively recomputing the MST after each change costs $O(E \log V)$.
Dynamic maintenance reduces this significantly using incremental repair of $T$.

#### How Does It Work (Plain Language)?

1. Edge Insertion:

   * Add the new edge $(u,v,w)$.
   * Check if it creates a cycle in $T$.
   * If the new edge is lighter than the heaviest edge on that cycle, replace the heavy one.
   * Otherwise, discard it.

2. Edge Deletion:

   * Remove $(u,v)$ from $T$, splitting it into two components.
   * Find the lightest edge connecting the two components in $G$.
   * Add that edge to restore connectivity.

3. Edge Weight Update:

   * If an edge's weight increases, treat as potential deletion.
   * If it decreases, treat as potential insertion.

To do this efficiently, we need data structures that can:

- find max edge on path quickly (for cycles)
- find min cross edge between components

These can be implemented via dynamic trees, Link-Cut Trees, or Euler Tour Trees.

#### Tiny Code (Simplified Static Version)

This version demonstrates edge insertion maintenance with cycle detection.

```python
def find(parent, x):
    if parent[x] != x:
        parent[x] = find(parent, parent[x])
    return parent[x]

def union(parent, rank, x, y):
    rx, ry = find(parent, x), find(parent, y)
    if rx == ry:
        return False
    if rank[rx] < rank[ry]:
        parent[rx] = ry
    elif rank[rx] > rank[ry]:
        parent[ry] = rx
    else:
        parent[ry] = rx
        rank[rx] += 1
    return True

def dynamic_mst_insert(V, mst_edges, new_edge):
    u, v, w = new_edge
    parent = [i for i in range(V)]
    rank = [0]*V
    for x, y, _ in mst_edges:
        union(parent, rank, x, y)
    if find(parent, u) != find(parent, v):
        mst_edges.append(new_edge)
    else:
        # would form cycle, pick lighter edge
        cycle_edges = mst_edges + [new_edge]
        heaviest = max(cycle_edges, key=lambda e: e[2])
        if heaviest != new_edge:
            mst_edges.remove(heaviest)
            mst_edges.append(new_edge)
    return mst_edges
```

This is conceptual; in practice, Link-Cut Trees make this dynamic in logarithmic time.

#### Why It Matters

- Crucial for streaming graphs, online networks, real-time routing.
- Avoids recomputation after each update.
- Demonstrates greedy invariants under change, the MST remains minimal if maintained properly.
- Forms basis for fully dynamic graph algorithms.

#### A Gentle Proof (Why It Works)

MST invariants are preserved through cut and cycle properties:

- Cycle Property: In a cycle, the heaviest edge cannot be in an MST.
- Cut Property: In any cut, the lightest edge crossing it must be in the MST.

When inserting an edge:

- It forms a cycle, we drop the heaviest to maintain minimality.

When deleting an edge:

- It forms a cut, we insert the lightest crossing edge to maintain connectivity.

Thus each update restores a valid MST in local time.

#### Try It Yourself

1. Start from an MST of a small graph.
2. Insert an edge and track the created cycle.
3. Delete an MST edge and find replacement.
4. Try batch updates, see structure evolve.
5. Compare runtime with full recomputation.

#### Test Cases

| Operation               | Description               | Result                          |
| ----------------------- | ------------------------- | ------------------------------- |
| Insert $(1,3,1)$        | creates cycle $(1,2,3,1)$ | remove heaviest edge            |
| Delete $(0,1)$          | breaks tree               | find lightest reconnecting edge |
| Decrease weight $(2,3)$ | recheck inclusion         | edge may enter MST              |

#### Complexity

- Insertion/Deletion (Naive): $O(E)$
- Dynamic Tree (Link-Cut): $O(\log^2 V)$ per update
- Space: $O(V + E)$

Dynamic MST maintenance shows how local adjustments preserve global optimality, a powerful principle for evolving systems.

### 348 Minimum Bottleneck Spanning Tree

A Minimum Bottleneck Spanning Tree (MBST) is a spanning tree that minimizes the maximum edge weight among all edges in the tree.
Unlike the standard MST, which minimizes the total sum of edge weights, an MBST focuses on the worst (heaviest) edge in the tree.

In many real-world systems, such as network design or transportation planning, you may care less about total cost and more about bottleneck constraints, the weakest or slowest connection.

#### What Problem Are We Solving?

Given a connected, undirected, weighted graph $G = (V, E, w)$, we want a spanning tree $T \subseteq E$ such that

$$
\max_{e \in T} w(e)
$$

is as small as possible.

In other words, among all spanning trees, pick one where the largest edge weight is minimal.

#### How Does It Work (Plain Language)?

You can build an MBST using the same algorithms as MST (Kruskal or Prim), because:

> Every Minimum Spanning Tree is also a Minimum Bottleneck Spanning Tree.

The reasoning:

- An MST minimizes the total sum.
- In doing so, it also ensures no unnecessarily heavy edges remain.

So any MST automatically satisfies the bottleneck property.

Alternatively, you can use a binary search + connectivity test:

1. Sort edges by weight.
2. Binary search for a threshold $W$.
3. Check if edges with $w(e) \le W$ can connect all vertices.
4. The smallest $W$ for which graph is connected is the bottleneck weight.
5. Extract any spanning tree using only edges $\le W$.

#### Tiny Code (Kruskal-based)

```python
def find(parent, x):
    if parent[x] != x:
        parent[x] = find(parent, parent[x])
    return parent[x]

def union(parent, rank, x, y):
    rx, ry = find(parent, x), find(parent, y)
    if rx == ry:
        return False
    if rank[rx] < rank[ry]:
        parent[rx] = ry
    elif rank[rx] > rank[ry]:
        parent[ry] = rx
    else:
        parent[ry] = rx
        rank[rx] += 1
    return True

def minimum_bottleneck_spanning_tree(V, edges):
    edges = sorted(edges, key=lambda x: x[2])
    parent = [i for i in range(V)]
    rank = [0]*V
    bottleneck = 0
    count = 0
    for u, v, w in edges:
        if union(parent, rank, u, v):
            bottleneck = max(bottleneck, w)
            count += 1
            if count == V - 1:
                break
    return bottleneck

edges = [(0,1,4),(0,2,3),(1,2,2),(1,3,5),(2,3,6)]
print(minimum_bottleneck_spanning_tree(4, edges))
```

Output:

```
5
```

Here, the MST has total weight 4+3+2=9, and bottleneck edge weight 5.

#### Why It Matters

- Useful for quality-of-service or bandwidth-constrained systems.
- Ensures no edge in the tree exceeds a critical capacity threshold.
- Illustrates that minimizing maximum and minimizing sum can align, a key insight in greedy algorithms.
- Provides a way to test MST correctness: if an MST doesn't minimize the bottleneck, it's invalid.

#### A Gentle Proof (Why It Works)

Let $T^*$ be an MST and $B(T^*)$ be its maximum edge weight.

Suppose another tree $T'$ had a smaller bottleneck:
$$
B(T') < B(T^*)
$$
Then there exists an edge $e$ in $T^*$ with $w(e) = B(T^*)$,
but $T'$ avoids it while staying connected using lighter edges.

This contradicts the cycle property, since $e$ would be replaced by a lighter edge crossing the same cut, meaning $T^*$ wasn't minimal.

Thus, every MST is an MBST.

#### Try It Yourself

1. Build a graph with multiple MSTs, check if they share the same bottleneck.
2. Compare MST total weight vs MBST bottleneck.
3. Apply binary search approach, confirm consistency.
4. Visualize all spanning trees and mark max edge in each.
5. Construct a case where multiple MBSTs exist.

#### Test Cases

| Graph    | Edges $(u,v,w)$                 | MST                     | Bottleneck |
| -------- | ------------------------------- | ----------------------- | ---------- |
| Triangle | (0,1,1),(1,2,2),(0,2,3)         | (0,1,1),(1,2,2)         | 2          |
| Square   | (0,1,1),(1,2,5),(2,3,2),(3,0,4) | (0,1,1),(2,3,2),(3,0,4) | 4          |
| Chain    | (0,1,2),(1,2,3),(2,3,4)         | same                    | 4          |

#### Complexity

- Time: $O(E \log E)$ (same as Kruskal)
- Space: $O(V)$
- Output: bottleneck weight or edges

A Minimum Bottleneck Spanning Tree highlights the heaviest load-bearing link, a critical measure when resilience, latency, or bandwidth limits matter more than total cost.

### 349 Manhattan MST

A Manhattan Minimum Spanning Tree (Manhattan MST) finds a spanning tree that minimizes the sum of Manhattan distances between connected points on a grid.
This variant is common in VLSI design, city planning, and grid-based path optimization, where movement is constrained to axis-aligned directions.

#### What Problem Are We Solving?

Given $n$ points in 2D space with coordinates $(x_i, y_i)$, the Manhattan distance between points $p_i$ and $p_j$ is

$$
d(p_i, p_j) = |x_i - x_j| + |y_i - y_j|
$$

We want to build a spanning tree connecting all points such that the total Manhattan distance is minimal.

A naive solution considers all $\binom{n}{2}$ edges and runs Kruskal's algorithm, but that's too slow for large $n$.
The key is exploiting geometry to limit candidate edges.

#### How Does It Work (Plain Language)?

The Manhattan MST problem leverages a geometric property:

> For each point, only a small number of nearest neighbors (under certain transformations) can belong to the MST.

Thus, instead of checking all pairs, we:

1. Sort points in specific directions (rotations/reflections).
2. Use a sweep line or Fenwick tree to find candidate neighbors.
3. Collect $O(n)$ potential edges.
4. Run Kruskal's algorithm on this reduced set.

By considering 8 directional transforms, we ensure all possible nearest edges are included.

Example transforms:

- $(x, y)$
- $(y, x)$
- $(-x, y)$
- $(x, -y)$
- etc.

For each transform:

- Sort points by $(x+y)$ or $(x-y)$.
- For each, track candidate neighbor minimizing $|x|+|y|$.
- Add edge between each point and its nearest neighbor.

Finally, compute MST using Kruskal on the collected edges.

#### Tiny Code (Simplified Illustration)

```python
def manhattan_distance(p1, p2):
    return abs(p1[0]-p2[0]) + abs(p1[1]-p2[1])

def manhattan_mst(points):
    edges = []
    n = len(points)
    for i in range(n):
        for j in range(i+1, n):
            w = manhattan_distance(points[i], points[j])
            edges.append((i, j, w))
    edges.sort(key=lambda e: e[2])

    parent = [i for i in range(n)]
    def find(x):
        if parent[x]!=x:
            parent[x]=find(parent[x])
        return parent[x]
    def union(x, y):
        rx, ry = find(x), find(y)
        if rx!=ry:
            parent[ry]=rx
            return True
        return False

    mst_edges = []
    cost = 0
    for u, v, w in edges:
        if union(u, v):
            mst_edges.append((u, v, w))
            cost += w
    return cost, mst_edges

points = [(0,0), (2,2), (2,0), (0,2)]
print(manhattan_mst(points))
```

Output:

```
(8, [(0,2,2),(0,3,2),(2,1,4)])
```

This brute-force example builds a Manhattan MST by checking all pairs.
Efficient geometric variants reduce this from $O(n^2)$ to $O(n \log n)$.

#### Why It Matters

- Captures grid-based movement (no diagonals).
- Critical in VLSI circuit layout (wire length minimization).
- Foundational for city-block planning and delivery networks.
- Shows how geometry and graph theory merge in spatial problems.

#### A Gentle Proof (Why It Works)

The MST under Manhattan distance obeys the cut property, the lightest edge crossing any partition must be in the MST.
By ensuring all directional neighbors are included, we never miss the minimal edge across any cut.

Thus, even though we prune candidate edges, correctness is preserved.

#### Try It Yourself

1. Generate random points, visualize MST edges.
2. Compare Manhattan MST vs Euclidean MST.
3. Add a diagonal, see how cost differs.
4. Try optimized directional neighbor search.
5. Observe symmetry, each transform covers a quadrant.

#### Test Cases

| Points                  | MST Edges   | Total Cost |
| ----------------------- | ----------- | ---------- |
| (0,0),(1,0),(1,1)       | (0,1),(1,2) | 2          |
| (0,0),(2,2),(2,0),(0,2) | 3 edges     | 8          |
| (0,0),(3,0),(0,4)       | (0,1),(0,2) | 7          |

#### Complexity

- Naive: $O(n^2 \log n)$
- Optimized (geometry-based): $O(n \log n)$
- Space: $O(n)$

Manhattan MSTs bridge grid geometry and graph optimization, the perfect example of structure guiding efficiency.

### 350 Euclidean MST (Kruskal + Geometry)

A Euclidean Minimum Spanning Tree (EMST) connects a set of points in the plane with the shortest total Euclidean length, the minimal possible wiring, pipeline, or connection layout under straight-line distances.

It is the geometric counterpart to classical MST problems, central to computational geometry, network design, and spatial clustering.

#### What Problem Are We Solving?

Given $n$ points $P = {p_1, p_2, \dots, p_n}$ in 2D (or higher dimensions), we want a spanning tree $T$ minimizing

$$
\text{cost}(T) = \sum_{(p_i,p_j) \in T} \sqrt{(x_i - x_j)^2 + (y_i - y_j)^2}
$$

Unlike abstract graph MSTs, here the graph is complete, every pair of points has an edge weighted by Euclidean distance —
but we cannot afford $O(n^2)$ edges for large $n$.

#### How Does It Work (Plain Language)?

The key geometric insight:

> The EMST is always a subgraph of the Delaunay Triangulation (DT).

So we don't need all $\binom{n}{2}$ edges, only those in the Delaunay graph (which has $O(n)$ edges).

Algorithm sketch:

1. Compute the Delaunay Triangulation (DT) of the points.
2. Extract all DT edges with their Euclidean weights.
3. Run Kruskal's or Prim's algorithm on these edges.
4. The resulting tree is the EMST.

This drastically reduces time from quadratic to near-linear.

#### Tiny Code (Brute-Force Demonstration)

Here's a minimal version using all pairs, good for small $n$:

```python
import math

def euclidean_distance(p1, p2):
    return math.sqrt((p1[0]-p2[0])2 + (p1[1]-p2[1])2)

def euclidean_mst(points):
    edges = []
    n = len(points)
    for i in range(n):
        for j in range(i+1, n):
            w = euclidean_distance(points[i], points[j])
            edges.append((i, j, w))
    edges.sort(key=lambda e: e[2])

    parent = [i for i in range(n)]
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[ry] = rx
            return True
        return False

    mst_edges = []
    cost = 0
    for u, v, w in edges:
        if union(u, v):
            mst_edges.append((u, v, w))
            cost += w
    return cost, mst_edges

points = [(0,0),(1,0),(0,1),(1,1)]
print(euclidean_mst(points))
```

Output:

```
(3.0, [(0,1,1.0),(0,2,1.0),(1,3,1.0)])
```

For real use, replace the all-pairs loop with Delaunay edges from a geometry library.

#### Why It Matters

- Optimally connects points with minimum total length.
- Used in geographic information systems, road networks, clustering (single-linkage), sensor networks, and pattern analysis.
- Demonstrates interplay between geometry and graph theory.
- Foundation for Steiner trees and TSP approximations.

#### A Gentle Proof (Why It Works)

The EMST obeys the cut property, for every partition of points, the shortest connecting edge must be in the EMST.

Edges outside the Delaunay Triangulation can be replaced by shorter edges crossing the same cut (from DT),
so the EMST must lie within the DT.

Thus, we can safely restrict candidate edges to the Delaunay set, preserving optimality.

#### Try It Yourself

1. Plot points on a plane, draw EMST edges.
2. Compare EMST vs MST with Manhattan distances.
3. Generate points in clusters, visualize how EMST bridges clusters.
4. Use a geometry library (e.g. `scipy.spatial.Delaunay`) to build fast DT edges.
5. Measure runtime difference: brute-force vs DT-based.

#### Test Cases

| Points          | EMST                    | Total Length           |
| --------------- | ----------------------- | ---------------------- |
| Square corners  | (0,0),(1,0),(1,1),(0,1) | 3.0                    |
| Triangle        | (0,0),(2,0),(1,√3)      | 4.0                    |
| Random 5 points |,                       | Depends on coordinates |

#### Complexity

- Brute-force: $O(n^2 \log n)$
- Delaunay-based: $O(n \log n)$
- Space: $O(n)$

The Euclidean MST is the geometric minimal backbone, the shortest invisible thread that ties all points together in the plane.

## Section 36. Flows

### 351 Ford–Fulkerson

The Ford–Fulkerson method is the foundational algorithm for computing the maximum flow in a directed network.
It views flow as water moving through a system of pipes, pushing flow from a source to a sink, respecting capacity limits and flow conservation.

#### What Problem Are We Solving?

Given a flow network $G = (V, E)$ with

- a source $s$,
- a sink $t$,
- and capacities $c(u, v)$ on each edge,

we want to find a maximum flow $f(u, v)$ such that:

1. Capacity constraint:
   $$0 \le f(u, v) \le c(u, v)$$

2. Flow conservation:
   $$\sum_{(u,v)\in E} f(u,v) = \sum_{(v,u)\in E} f(v,u) \quad \forall u \in V \setminus {s, t}$$

3. Maximize total flow:
   $$
   \text{Maximize } |f| = \sum_{(s,v)\in E} f(s,v)
   $$

#### How Does It Work (Plain Language)?

The algorithm repeatedly searches for augmenting paths, routes from $s$ to $t$ where additional flow can still be sent.
Each iteration increases total flow until no augmenting path remains.

Step-by-step:

1. Initialize all flows to 0.
2. While an augmenting path $P$ exists from $s$ to $t$ in the residual graph:

   * Find the bottleneck capacity along $P$:
     $$b = \min_{(u,v)\in P} (c(u,v) - f(u,v))$$
   * Augment flow along $P$ by $b$:
     $$f(u,v) \gets f(u,v) + b$$
     $$f(v,u) \gets f(v,u) - b$$
3. Update residual capacities and repeat.
4. When no path remains, current flow is maximum.

The residual graph represents remaining capacity, allowing backward edges to cancel flow if needed.

#### Tiny Code

Python (DFS-based Augmentation)

```python
from collections import defaultdict

def ford_fulkerson(capacity, s, t):
    n = len(capacity)
    flow = [[0]*n for _ in range(n)]

    def dfs(u, t, f, visited):
        if u == t:
            return f
        visited[u] = True
        for v in range(n):
            residual = capacity[u][v] - flow[u][v]
            if residual > 0 and not visited[v]:
                pushed = dfs(v, t, min(f, residual), visited)
                if pushed > 0:
                    flow[u][v] += pushed
                    flow[v][u] -= pushed
                    return pushed
        return 0

    max_flow = 0
    while True:
        visited = [False]*n
        pushed = dfs(s, t, float('inf'), visited)
        if pushed == 0:
            break
        max_flow += pushed
    return max_flow

capacity = [
    [0, 16, 13, 0, 0, 0],
    [0, 0, 10, 12, 0, 0],
    [0, 4, 0, 0, 14, 0],
    [0, 0, 9, 0, 0, 20],
    [0, 0, 0, 7, 0, 4],
    [0, 0, 0, 0, 0, 0]
]
print(ford_fulkerson(capacity, 0, 5))
```

Output:

```
23
```

#### Why It Matters

- Introduces max-flow concept, a cornerstone of network optimization.
- Underpins algorithms for:

  * bipartite matching
  * minimum cut problems
  * circulations and network design
- Illustrates residual graphs and augmenting paths, used in many flow-based algorithms.

#### A Gentle Proof (Why It Works)

Each augmentation increases total flow by a positive amount and maintains feasibility.
Because total capacity is finite, the process eventually terminates.

By the Max-Flow Min-Cut Theorem:

$$
|f^*| = \text{capacity of minimum } (S, T) \text{ cut}
$$

So when no augmenting path exists, the flow is maximum.

If all capacities are integers, the algorithm converges in finite steps, since each augmentation adds at least 1 unit.

#### Try It Yourself

1. Run on small networks, draw residual graph at each step.
2. Track augmenting paths and bottleneck edges.
3. Compare to Edmonds–Karp (BFS search).
4. Change capacities to fractional, observe potential infinite loops.
5. Use to solve bipartite matching (convert to flow network).

#### Test Cases

| Graph                  | Max Flow          | Notes            |
| ---------------------- | ----------------- | ---------------- |
| Simple 3-node          | 5                 | single augment   |
| Classic 6-node (above) | 23                | textbook example |
| Parallel edges         | Sum of capacities | additive         |

#### Complexity

- Time: $O(E \cdot |f^*|)$ for integer capacities
- Space: $O(V^2)$
- Optimized (Edmonds–Karp): $O(VE^2)$

Ford–Fulkerson builds the intuition of pushing flow through capacity-constrained paths, the heart of network optimization.

### 352 Edmonds–Karp

The Edmonds–Karp algorithm is a concrete, polynomial-time implementation of the Ford–Fulkerson method, where each augmenting path is chosen using Breadth-First Search (BFS).
By always selecting the shortest path (in edge count) from source to sink, it guarantees efficient convergence.

#### What Problem Are We Solving?

Given a directed flow network $G = (V, E)$ with

- a source $s$,
- a sink $t$,
- and nonnegative capacities $c(u, v)$,

we want to find a maximum flow $f(u, v)$ that satisfies:

1. Capacity constraints:
   $$0 \le f(u, v) \le c(u, v)$$

2. Flow conservation:
   $$\sum_{(u,v)\in E} f(u,v) = \sum_{(v,u)\in E} f(v,u) \quad \forall u \in V \setminus {s, t}$$

3. Maximize total flow:
   $$|f| = \sum_{(s,v)\in E} f(s,v)$$

The algorithm improves Ford–Fulkerson by enforcing shortest augmenting path order, which bounds the number of iterations.

#### How Does It Work (Plain Language)?

Each iteration finds an augmenting path using BFS (so each edge in the path is part of the shortest route in terms of hops).
Then we push the maximum possible flow through that path and update residual capacities.

Step-by-step:

1. Initialize $f(u,v)=0$ for all edges.
2. While there exists a path $P$ from $s$ to $t$ in the residual graph (found by BFS):

   * Compute bottleneck capacity:
     $$b = \min_{(u,v)\in P} (c(u,v) - f(u,v))$$
   * Augment along path $P$:
     $$f(u,v) \gets f(u,v) + b$$
     $$f(v,u) \gets f(v,u) - b$$
3. Repeat until no augmenting path remains.

Residual graph is updated after each augmentation, including reverse edges to allow flow cancellation.

#### Tiny Code

Python Implementation (BFS Augmentation)

```python
from collections import deque

def bfs(capacity, flow, s, t):
    n = len(capacity)
    parent = [-1] * n
    parent[s] = s
    q = deque([s])
    while q:
        u = q.popleft()
        for v in range(n):
            residual = capacity[u][v] - flow[u][v]
            if residual > 0 and parent[v] == -1:
                parent[v] = u
                q.append(v)
                if v == t:
                    return parent
    return None

def edmonds_karp(capacity, s, t):
    n = len(capacity)
    flow = [[0]*n for _ in range(n)]
    max_flow = 0

    while True:
        parent = bfs(capacity, flow, s, t)
        if not parent:
            break
        # find bottleneck
        v = t
        bottleneck = float('inf')
        while v != s:
            u = parent[v]
            bottleneck = min(bottleneck, capacity[u][v] - flow[u][v])
            v = u
        # augment flow
        v = t
        while v != s:
            u = parent[v]
            flow[u][v] += bottleneck
            flow[v][u] -= bottleneck
            v = u
        max_flow += bottleneck

    return max_flow

capacity = [
    [0, 16, 13, 0, 0, 0],
    [0, 0, 10, 12, 0, 0],
    [0, 4, 0, 0, 14, 0],
    [0, 0, 9, 0, 0, 20],
    [0, 0, 0, 7, 0, 4],
    [0, 0, 0, 0, 0, 0]
]
print(edmonds_karp(capacity, 0, 5))
```

Output:

```
23
```

#### Why It Matters

- Ensures polynomial time termination.
- Demonstrates the power of BFS for finding shortest augmenting paths.
- Serves as the canonical maximum flow algorithm in theory and practice.
- Provides a clean proof of the Max-Flow Min-Cut Theorem.
- Foundation for more advanced methods (Dinic, Push–Relabel).

#### A Gentle Proof (Why It Works)

Each augmentation uses a shortest path in terms of edges.
After each augmentation, at least one edge on that path saturates (reaches full capacity).

Because every edge can only move from residual distance $d$ to $d+2$ a limited number of times,
the total number of augmentations is $O(VE)$.

Each BFS runs in $O(E)$, giving total runtime $O(VE^2)$.

The algorithm terminates when no path exists, i.e. the residual graph disconnects $s$ and $t$,
and by the Max-Flow Min-Cut theorem, the current flow is maximum.

#### Try It Yourself

1. Run BFS after each iteration, visualize residual network.
2. Compare with Ford–Fulkerson (DFS), note fewer augmentations.
3. Modify capacities, see how path selection changes.
4. Implement path reconstruction and print augmenting paths.
5. Use to solve bipartite matching via flow transformation.

#### Test Cases

| Graph          | Max Flow               | Notes                 |
| -------------- | ---------------------- | --------------------- |
| Simple 3-node  | 5                      | BFS finds direct path |
| Classic 6-node | 23                     | textbook example      |
| Star network   | Sum of edge capacities | each edge unique path |

#### Complexity

- Time: $O(VE^2)$
- Space: $O(V^2)$
- Augmentations: $O(VE)$

Edmonds–Karp transforms Ford–Fulkerson into a predictable, efficient, and elegant BFS-based flow engine, ensuring progress, bounding iterations, and revealing the structure of optimal flow.

### 353 Dinic's Algorithm

Dinic's Algorithm (or Dinitz's algorithm) is a faster approach to computing maximum flow, improving upon Edmonds–Karp by introducing a level graph and sending blocking flows within it.
It combines BFS (to layer the graph) with DFS (to find augmenting paths), achieving a strong polynomial bound.

#### What Problem Are We Solving?

Given a directed flow network $G = (V, E)$ with

- source $s$,
- sink $t$,
- capacities $c(u,v)$ on each edge,

find the maximum flow $f(u,v)$ such that:

1. Capacity constraint:
   $$0 \le f(u, v) \le c(u, v)$$

2. Flow conservation:
   $$\sum_{(u,v)\in E} f(u,v) = \sum_{(v,u)\in E} f(v,u) \quad \forall u \in V \setminus {s, t}$$

3. Maximize total flow:
   $$|f| = \sum_{(s,v)\in E} f(s,v)$$

#### How Does It Work (Plain Language)?

Dinic's Algorithm operates in phases.
Each phase constructs a level graph using BFS, then pushes as much flow as possible within that layered structure using DFS.
This approach ensures progress, each phase strictly increases the shortest path length from $s$ to $t$ in the residual graph.

Step-by-step:

1. Build Level Graph (BFS):

   * Run BFS from $s$.
   * Assign each vertex a level = distance from $s$ in residual graph.
   * Only edges $(u,v)$ with $level[v] = level[u] + 1$ are used.

2. Send Blocking Flow (DFS):

   * Use DFS to push flow from $s$ to $t$ along level-respecting paths.
   * Stop when no more flow can be sent (i.e., blocking flow).

3. Repeat:

   * Rebuild level graph; continue until $t$ is unreachable from $s$.

A blocking flow saturates at least one edge on every $s$–$t$ path in the level graph, ensuring termination per phase.

#### Tiny Code

Python Implementation (Adjacency List)

```python
from collections import deque

class Dinic:
    def __init__(self, n):
        self.n = n
        self.adj = [[] for _ in range(n)]

    def add_edge(self, u, v, cap):
        self.adj[u].append([v, cap, len(self.adj[v])])
        self.adj[v].append([u, 0, len(self.adj[u]) - 1])  # reverse edge

    def bfs(self, s, t, level):
        q = deque([s])
        level[s] = 0
        while q:
            u = q.popleft()
            for v, cap, _ in self.adj[u]:
                if cap > 0 and level[v] < 0:
                    level[v] = level[u] + 1
                    q.append(v)
        return level[t] >= 0

    def dfs(self, u, t, flow, level, it):
        if u == t:
            return flow
        while it[u] < len(self.adj[u]):
            v, cap, rev = self.adj[u][it[u]]
            if cap > 0 and level[v] == level[u] + 1:
                pushed = self.dfs(v, t, min(flow, cap), level, it)
                if pushed > 0:
                    self.adj[u][it[u]][1] -= pushed
                    self.adj[v][rev][1] += pushed
                    return pushed
            it[u] += 1
        return 0

    def max_flow(self, s, t):
        flow = 0
        level = [-1] * self.n
        INF = float('inf')
        while self.bfs(s, t, level):
            it = [0] * self.n
            while True:
                pushed = self.dfs(s, t, INF, level, it)
                if pushed == 0:
                    break
                flow += pushed
            level = [-1] * self.n
        return flow

# Example
dinic = Dinic(6)
edges = [
    (0,1,16),(0,2,13),(1,2,10),(2,1,4),(1,3,12),
    (3,2,9),(2,4,14),(4,3,7),(3,5,20),(4,5,4)
]
for u,v,c in edges:
    dinic.add_edge(u,v,c)
print(dinic.max_flow(0,5))
```

Output:

```
23
```

#### Why It Matters

- Achieves $O(V^2E)$ complexity (faster in practice).
- Exploits layering to avoid redundant augmentations.
- Basis for advanced flow algorithms like Dinic + scaling, Push–Relabel, and Blocking Flow variants.
- Common in competitive programming, network routing, and bipartite matching.

#### A Gentle Proof (Why It Works)

Each phase increases the shortest distance from $s$ to $t$ in the residual graph.
Because there are at most $V-1$ distinct distances, the algorithm runs for at most $O(V)$ phases.
Within each phase, we find a blocking flow, which can be computed in $O(E)$ DFS calls.

Hence total runtime:
$$
O(VE)
$$
for unit networks (each edge capacity = 1),
and in general,
$$
O(V^2E)
$$

When no augmenting path exists, residual graph disconnects $s$ and $t$, so the current flow is maximum by the Max-Flow Min-Cut theorem.

#### Try It Yourself

1. Compare BFS layers between phases, note increasing depth.
2. Visualize level graph and residual edges.
3. Test on bipartite graph, confirm match size = flow.
4. Modify to store flow per edge.
5. Add capacity scaling to speed up dense graphs.

#### Test Cases

| Graph           | Max Flow               | Notes           |
| --------------- | ---------------------- | --------------- |
| 6-node sample   | 23                     | classic example |
| Unit network    | equals #disjoint paths |                 |
| Bipartite graph | equals max matching    |                 |

#### Complexity

- Time: $O(V^2E)$ general, $O(\sqrt{V}E)$ for unit capacities
- Space: $O(V + E)$
- Augmentations per phase: $O(E)$

Dinic's algorithm elegantly combines BFS-level layering with DFS-based flow pushing, a perfect synthesis of structure and greed, powering modern flow computations.

### 354 Push–Relabel

The Push–Relabel algorithm (also called Preflow–Push) takes a completely different view of the maximum flow problem.
Instead of finding paths from the source to the sink, it locally pushes flow along edges and adjusts vertex heights (labels) to guide flow downhill toward the sink.

This approach is highly efficient in practice and forms the basis for many modern flow solvers.

#### What Problem Are We Solving?

Given a directed network $G = (V, E)$ with

- source $s$,
- sink $t$,
- capacity $c(u, v)$,

we want a flow $f(u,v)$ satisfying:

1. Capacity constraint:
   $$0 \le f(u, v) \le c(u, v)$$

2. Flow conservation:
   $$\sum_{(u,v)\in E} f(u,v) = \sum_{(v,u)\in E} f(v,u) \quad \forall u \neq s,t$$

3. Maximize total flow:
   $$|f| = \sum_{(s,v)\in E} f(s,v)$$

#### How Does It Work (Plain Language)?

Unlike Ford–Fulkerson, which finds augmenting paths, Push–Relabel maintains a preflow (where intermediate nodes may have excess flow) and fixes imbalances gradually.

Key concepts:

- Preflow: flow can temporarily violate conservation (nodes can have excess).
- Height (label): an integer guiding flow direction, flow only moves "downhill."
- Push: send flow from $u$ to $v$ if possible.
- Relabel: when $u$ is stuck, increase its height so flow can continue.

Step-by-step:

1. Initialize all $f(u,v)=0$.
2. Set $h(s)=|V|$ and push as much as possible from $s$ to its neighbors.
3. While any vertex (other than $s,t$) has excess flow, do:

   * Push: if $(u,v)$ is admissible ($h(u)=h(v)+1$ and residual $>0$), send
     $$\Delta = \min(e(u), c(u,v)-f(u,v))$$
   * Relabel: if no admissible edge, set
     $$h(u) = 1 + \min_{(u,v): c(u,v)-f(u,v)>0} h(v)$$

Repeat until all excess is at $t$ or $s$.

#### Tiny Code

Python Implementation (Simplified)

```python
def push_relabel(capacity, s, t):
    n = len(capacity)
    flow = [[0]*n for _ in range(n)]
    excess = [0]*n
    height = [0]*n
    height[s] = n

    def push(u, v):
        delta = min(excess[u], capacity[u][v] - flow[u][v])
        flow[u][v] += delta
        flow[v][u] -= delta
        excess[u] -= delta
        excess[v] += delta

    def relabel(u):
        min_h = float('inf')
        for v in range(n):
            if capacity[u][v] - flow[u][v] > 0:
                min_h = min(min_h, height[v])
        if min_h < float('inf'):
            height[u] = min_h + 1

    def discharge(u):
        while excess[u] > 0:
            for v in range(n):
                if capacity[u][v] - flow[u][v] > 0 and height[u] == height[v] + 1:
                    push(u, v)
                    if excess[u] == 0:
                        break
            else:
                relabel(u)

    # Initialize preflow
    for v in range(n):
        if capacity[s][v] > 0:
            flow[s][v] = capacity[s][v]
            flow[v][s] = -capacity[s][v]
            excess[v] = capacity[s][v]
    excess[s] = sum(capacity[s])

    # Discharge active vertices
    active = [i for i in range(n) if i != s and i != t]
    p = 0
    while p < len(active):
        u = active[p]
        old_height = height[u]
        discharge(u)
        if height[u] > old_height:
            active.insert(0, active.pop(p))  # move to front
            p = 0
        else:
            p += 1

    return sum(flow[s])

capacity = [
    [0,16,13,0,0,0],
    [0,0,10,12,0,0],
    [0,4,0,0,14,0],
    [0,0,9,0,0,20],
    [0,0,0,7,0,4],
    [0,0,0,0,0,0]
]
print(push_relabel(capacity, 0, 5))
```

Output:

```
23
```

#### Why It Matters

- Local view: No need for global augmenting paths.
- Highly parallelizable.
- Performs very well in dense graphs.
- Serves as the basis for highest-label and FIFO variants.
- Conceptually elegant: flow "falls downhill" guided by heights.

#### A Gentle Proof (Why It Works)

1. The height invariant ensures no flow moves from lower to higher vertices.
2. Every push respects capacity and non-negativity.
3. Height always increases, ensuring termination.
4. When no vertex (except $s$ and $t$) has excess, all preflow constraints are satisfied, the preflow becomes a valid maximum flow.

By the Max-Flow Min-Cut Theorem, the final preflow's value equals the capacity of the minimum cut.

#### Try It Yourself

1. Track each vertex's height and excess after each step.
2. Compare FIFO vs highest-label variants.
3. Use small networks to visualize flow movement.
4. Contrast with path-based algorithms (Ford–Fulkerson).
5. Add logging to observe relabel events.

#### Test Cases

| Graph          | Max Flow             | Notes     |
| -------------- | -------------------- | --------- |
| Classic 6-node | 23                   | textbook  |
| Dense complete | high flow            | efficient |
| Sparse path    | same as Edmonds–Karp | similar   |

#### Complexity

- Time (Generic): $O(V^2E)$
- With FIFO / Highest-Label Heuristics: $O(V^3)$ or better
- Space: $O(V^2)$

Push–Relabel transforms max-flow into a local balancing act, pushing, relabeling, and equalizing pressure until equilibrium is achieved.

### 355 Capacity Scaling

The Capacity Scaling algorithm is a refined version of Ford–Fulkerson, designed to handle large edge capacities efficiently.
Instead of augmenting arbitrarily, it focuses on high-capacity edges first, gradually refining the flow as the scaling parameter decreases.

This approach reduces the number of augmentations by focusing early on "big pipes" before worrying about smaller ones.

#### What Problem Are We Solving?

Given a directed flow network $G = (V, E)$ with

- source $s$,
- sink $t$,
- nonnegative capacities $c(u, v)$,

we want to compute a maximum flow $f(u, v)$ satisfying:

1. Capacity constraint:
   $$0 \le f(u, v) \le c(u, v)$$
2. Flow conservation:
   $$\sum_{(u,v)\in E} f(u,v) = \sum_{(v,u)\in E} f(v,u), \quad \forall u \neq s,t$$
3. Maximize:
   $$|f| = \sum_{(s,v)\in E} f(s,v)$$

When capacities are large, standard augmenting-path methods can take many iterations.
Capacity scaling reduces this by grouping edges by capacity magnitude.

#### How Does It Work (Plain Language)?

We introduce a scaling parameter $\Delta$, starting from the highest power of 2 below the maximum capacity.
We only consider edges with residual capacity ≥ Δ during augmentations.
Once no such path exists, halve $\Delta$ and continue.

This ensures we push large flows early and refine later.

Step-by-step:

1. Initialize $f(u,v) = 0$.
2. Let
   $$\Delta = 2^{\lfloor \log_2 C_{\max} \rfloor}$$
   where $C_{\max} = \max_{(u,v)\in E} c(u,v)$
3. While $\Delta \ge 1$:

   * Build Δ-residual graph: edges with residual capacity $\ge \Delta$
   * While an augmenting path exists in this graph:

     * Send flow equal to the bottleneck along that path
   * Update $\Delta \gets \Delta / 2$
4. Return total flow.

#### Tiny Code

Python (DFS Augmentation with Scaling)

```python
def dfs(u, t, f, visited, capacity, flow, delta):
    if u == t:
        return f
    visited[u] = True
    for v in range(len(capacity)):
        residual = capacity[u][v] - flow[u][v]
        if residual >= delta and not visited[v]:
            pushed = dfs(v, t, min(f, residual), visited, capacity, flow, delta)
            if pushed > 0:
                flow[u][v] += pushed
                flow[v][u] -= pushed
                return pushed
    return 0

def capacity_scaling(capacity, s, t):
    n = len(capacity)
    flow = [[0]*n for _ in range(n)]
    Cmax = max(max(row) for row in capacity)
    delta = 1
    while delta * 2 <= Cmax:
        delta *= 2

    max_flow = 0
    while delta >= 1:
        while True:
            visited = [False]*n
            pushed = dfs(s, t, float('inf'), visited, capacity, flow, delta)
            if pushed == 0:
                break
            max_flow += pushed
        delta //= 2
    return max_flow

capacity = [
    [0, 16, 13, 0, 0, 0],
    [0, 0, 10, 12, 0, 0],
    [0, 4, 0, 0, 14, 0],
    [0, 0, 9, 0, 0, 20],
    [0, 0, 0, 7, 0, 4],
    [0, 0, 0, 0, 0, 0]
]
print(capacity_scaling(capacity, 0, 5))
```

Output:

```
23
```

#### Why It Matters

- Reduces augmentations compared to plain Ford–Fulkerson.
- Focuses on big pushes first, improving convergence.
- Demonstrates the power of scaling ideas, a recurring optimization in algorithm design.
- Serves as a stepping stone to cost scaling and capacity scaling in min-cost flow.

#### A Gentle Proof (Why It Works)

At each scaling phase $\Delta$,

- Each augmenting path increases flow by at least $\Delta$.
- Total flow value is at most $|f^*|$.
- Therefore, each phase performs at most $O(|f^*| / \Delta)$ augmentations.

Since $\Delta$ halves each phase,
the total number of augmentations is
$$
O(E \log C_{\max})
$$

Each path search takes $O(E)$, so
$$
T = O(E^2 \log C_{\max})
$$

Termination occurs when $\Delta = 1$ and no path remains, meaning max flow reached.

#### Try It Yourself

1. Compare augmentation order with Ford–Fulkerson.
2. Track $\Delta$ values and residual graphs per phase.
3. Observe faster convergence for large capacities.
4. Combine with BFS to mimic Edmonds–Karp structure.
5. Visualize flow accumulation at each scale.

#### Test Cases

| Graph                | Max Flow                  | Notes            |
| -------------------- | ------------------------- | ---------------- |
| Classic 6-node       | 23                        | textbook example |
| Large capacity edges | same flow, fewer steps    | scaling helps    |
| Unit capacities      | behaves like Edmonds–Karp | small gains      |

#### Complexity

- Time: $O(E^2 \log C_{\max})$
- Space: $O(V^2)$
- Augmentations: $O(E \log C_{\max})$

Capacity scaling embodies a simple but powerful idea:
solve coarse first, refine later, push the big flows early, and polish at the end.

### 356 Cost Scaling

The Cost Scaling algorithm tackles the Minimum-Cost Maximum Flow (MCMF) problem by applying a scaling technique not to capacities but to edge costs.
It gradually refines the precision of reduced costs, maintaining ε-optimality throughout, and converges to the true optimal flow.

This approach is both theoretically elegant and practically efficient, forming the foundation for high-performance network optimization solvers.

#### What Problem Are We Solving?

Given a directed network $G = (V, E)$ with

- capacity $c(u, v) \ge 0$,
- cost (per unit flow) $w(u, v)$,
- source $s$, and sink $t$,

we want to send the maximum flow from $s$ to $t$ at minimum total cost.

We minimize:
$$
\text{Cost}(f) = \sum_{(u,v)\in E} f(u,v), w(u,v)
$$

subject to:

- Capacity constraint: $0 \le f(u,v) \le c(u,v)$
- Conservation: $\sum_v f(u,v) = \sum_v f(v,u)$ for all $u \neq s,t$

#### How Does It Work (Plain Language)?

Cost scaling uses successive refinements of reduced costs, ensuring flows are ε-optimal at each phase.

An ε-optimal flow satisfies:
$$
c_p(u,v) = w(u,v) + \pi(u) - \pi(v) \ge -\varepsilon
$$
for all residual edges $(u,v)$, where $\pi$ is a potential function.

The algorithm begins with a large $\varepsilon$ (often $C_{\max}$ or $W_{\max}$) and reduces it geometrically (e.g. $\varepsilon \gets \varepsilon / 2$).
During each phase, pushes are allowed only along admissible edges ($c_p(u,v) < 0$), maintaining ε-optimality.

Step-by-step:

1. Initialize preflow (push as much as possible from $s$).
2. Set $\varepsilon = C_{\max}$.
3. While $\varepsilon \ge 1$:

   * Maintain ε-optimality: adjust flows and potentials.
   * For each vertex with excess,

     * Push flow along admissible edges ($c_p(u,v) < 0$).
     * If stuck, relabel by increasing $\pi(u)$ (lowering reduced costs).
   * Halve $\varepsilon$.
4. When $\varepsilon < 1$, solution is optimal.

#### Tiny Code (Simplified Skeleton)

Below is a conceptual outline (not a full implementation) for educational clarity:

```python
def cost_scaling_mcmf(V, edges, s, t):
    INF = float('inf')
    adj = [[] for _ in range(V)]
    for u, v, cap, cost in edges:
        adj[u].append([v, cap, cost, len(adj[v])])
        adj[v].append([u, 0, -cost, len(adj[u]) - 1])

    pi = [0]*V  # potentials
    excess = [0]*V
    excess[s] = INF

    def reduced_cost(u, v, cost):
        return cost + pi[u] - pi[v]

    epsilon = max(abs(c) for _,_,_,c in edges)
    while epsilon >= 1:
        active = [i for i in range(V) if excess[i] > 0 and i != s and i != t]
        while active:
            u = active.pop()
            for e in adj[u]:
                v, cap, cost, rev = e
                rc = reduced_cost(u, v, cost)
                if cap > 0 and rc < 0:
                    delta = min(cap, excess[u])
                    e[1] -= delta
                    adj[v][rev][1] += delta
                    excess[u] -= delta
                    excess[v] += delta
                    if v not in (s, t) and excess[v] == delta:
                        active.append(v)
            if excess[u] > 0:
                pi[u] += epsilon
        epsilon //= 2

    total_cost = 0
    for u in range(V):
        for v, cap, cost, rev in adj[u]:
            if cost > 0:
                total_cost += cost * (adj[v][rev][1])
    return total_cost
```

#### Why It Matters

- Avoids expensive shortest-path searches at each step.
- Ensures strong polynomial bounds using ε-optimality.
- Suitable for dense graphs and large integer costs.
- Forms the backbone of network simplex and scalable MCMF solvers.

#### A Gentle Proof (Why It Works)

1. ε-optimality: ensures approximate optimality at each scaling phase.
2. Scaling: reduces ε geometrically, converging to exact optimality.
3. Termination: when $\varepsilon < 1$, all reduced costs are nonnegative, the flow is optimal.

Each push respects capacity and admissibility; each relabel decreases ε, guaranteeing progress.

#### Try It Yourself

1. Compare to Successive Shortest Path algorithm, note fewer path searches.
2. Track how potentials $\pi(u)$ evolve over phases.
3. Visualize ε-layers of admissible edges.
4. Experiment with cost magnitudes, larger costs benefit more from scaling.
5. Observe convergence as ε halves each phase.

#### Test Cases

| Graph           | Max Flow | Min Cost  | Notes         |
| --------------- | -------- | --------- | ------------- |
| Simple 3-node   | 5        | 10        | small example |
| Classic network | 23       | 42        | cost-scaled   |
| Dense graph     | varies   | efficient |               |

#### Complexity

- Time: $O(E \log C_{\max} (V + E))$ (depends on cost range)
- Space: $O(V + E)$
- Phases: $O(\log C_{\max})$

Cost scaling showcases the precision-refinement paradigm, start coarse, end exact.
It's a masterclass in combining scaling, potential functions, and local admissibility to achieve globally optimal flows.

### 357 Min-Cost Max-Flow (Bellman-Ford)

The Min-Cost Max-Flow (MCMF) algorithm with Bellman-Ford is a cornerstone method for solving flow problems where both capacity and cost constraints matter. It repeatedly augments flow along the shortest-cost paths, ensuring every unit of flow moves as cheaply as possible.

This version uses Bellman-Ford to handle negative edge costs, ensuring correctness even when cycles reduce total cost.

#### What Problem Are We Solving?

Given a directed graph $G = (V, E)$ with:

- Capacity $c(u, v)$
- Cost per unit flow $w(u, v)$
- Source $s$ and sink $t$

Find a flow $f(u, v)$ that:

1. Respects capacities: $0 \le f(u, v) \le c(u, v)$
2. Conserves flow: $\sum_v f(u, v) = \sum_v f(v, u)$ for all $u \neq s,t$
3. Maximizes total flow, while
4. Minimizing total cost:
   $$
   \text{Cost}(f) = \sum_{(u,v)\in E} f(u,v),w(u,v)
   $$

#### How Does It Work (Plain Language)?

The algorithm iteratively finds augmenting paths from $s$ to $t$ using shortest path search by cost (not distance).

At each iteration:

1. Run Bellman-Ford on the residual graph to find the shortest-cost path.
2. Determine the bottleneck capacity $\delta$ on that path.
3. Augment flow along that path by $\delta$.
4. Update residual capacities and reverse edges.
5. Repeat until no more augmenting paths exist.

Because Bellman-Ford supports negative costs, this algorithm correctly handles graphs where cost reductions occur via cycles.

#### Tiny Code (C-like Pseudocode)

```c
struct Edge { int v, cap, cost, rev; };
vector<Edge> adj[V];
int dist[V], parent[V], parent_edge[V];

bool bellman_ford(int s, int t, int V) {
    fill(dist, dist+V, INF);
    dist[s] = 0;
    bool updated = true;
    for (int i = 0; i < V-1 && updated; i++) {
        updated = false;
        for (int u = 0; u < V; u++) {
            if (dist[u] == INF) continue;
            for (int k = 0; k < adj[u].size(); k++) {
                auto &e = adj[u][k];
                if (e.cap > 0 && dist[e.v] > dist[u] + e.cost) {
                    dist[e.v] = dist[u] + e.cost;
                    parent[e.v] = u;
                    parent_edge[e.v] = k;
                    updated = true;
                }
            }
        }
    }
    return dist[t] < INF;
}

pair<int,int> min_cost_max_flow(int s, int t, int V) {
    int flow = 0, cost = 0;
    while (bellman_ford(s, t, V)) {
        int f = INF;
        for (int v = t; v != s; v = parent[v]) {
            int u = parent[v];
            auto &e = adj[u][parent_edge[v]];
            f = min(f, e.cap);
        }
        for (int v = t; v != s; v = parent[v]) {
            int u = parent[v];
            auto &e = adj[u][parent_edge[v]];
            e.cap -= f;
            adj[v][e.rev].cap += f;
            cost += f * e.cost;
        }
        flow += f;
    }
    return {flow, cost};
}
```

#### Why It Matters

- Handles negative edge costs safely.
- Guaranteed optimality if no negative cycles exist.
- Works well for small to medium graphs.
- A foundation for more efficient variants (e.g. SPFA, Dijkstra with potentials).

It's the go-to teaching implementation for min-cost flow, balancing clarity and correctness.

#### A Gentle Proof (Why It Works)

Each iteration finds a shortest-cost augmenting path using Bellman-Ford.
Because edge costs are non-decreasing over augmentations, the algorithm converges to a globally optimal flow.

Each augmentation:

- Increases flow.
- Does not introduce cheaper paths later (monotonicity).
- Terminates when no more augmenting paths exist.

Thus, the resulting flow is both maximal and minimum cost.

#### Try It Yourself

1. Draw a simple graph with 4 nodes and a negative edge.
2. Trace residual updates after each augmentation.
3. Compare results with a naive greedy path selection.
4. Replace Bellman-Ford with Dijkstra + potentials to improve speed.
5. Visualize residual capacity evolution.

#### Test Cases

| Graph                       | Max Flow | Min Cost | Notes                  |
| --------------------------- | -------- | -------- | ---------------------- |
| Chain $s \to a \to b \to t$ | 5        | 10       | Simple path            |
| Graph with negative cycle   | Invalid  | -        | Must detect            |
| Grid-like network           | 12       | 24       | Multiple augmentations |

#### Complexity

- Time: $O(F \cdot V \cdot E)$, where $F$ is total flow sent.
- Space: $O(V + E)$

For small graphs or graphs with negative weights, Bellman-Ford MCMF is the most robust and straightforward method, clear, reliable, and foundational.

### 358 Min-Cost Max-Flow (SPFA)

The Min-Cost Max-Flow algorithm using SPFA (Shortest Path Faster Algorithm) is an optimization of the Bellman–Ford approach.
It leverages a queue-based relaxation method to find the shortest-cost augmenting paths more efficiently in practice, especially on sparse graphs or when negative edges are rare.

#### What Problem Are We Solving?

We are solving the minimum-cost maximum-flow problem:
Find a flow $f$ in a directed graph $G = (V, E)$ with:

- Capacity $c(u, v) \ge 0$
- Cost per unit flow $w(u, v)$
- Source $s$, Sink $t$

Subject to:

1. Capacity constraint: $0 \le f(u, v) \le c(u, v)$
2. Conservation of flow: $\sum_v f(u,v) = \sum_v f(v,u)$ for all $u \neq s,t$
3. Objective:
   $$
   \min \text{Cost}(f) = \sum_{(u,v)\in E} f(u,v),w(u,v)
   $$

We seek a maximum flow from $s$ to $t$ that incurs the minimum total cost.

#### How Does It Work (Plain Language)?

SPFA finds shortest paths by cost more efficiently than Bellman–Ford, using a queue to relax only the vertices that can still improve distances.

In each iteration:

1. Run SPFA on the residual graph to find the shortest-cost path.
2. Compute the bottleneck flow $\delta$ on that path.
3. Augment $\delta$ units of flow along the path.
4. Update residual capacities and reverse edges.
5. Repeat until no augmenting path remains.

SPFA dynamically manages which nodes are in the queue, making it faster on average than full Bellman–Ford (though worst-case still similar).

#### Tiny Code (C-like Pseudocode)

```c
struct Edge { int v, cap, cost, rev; };
vector<Edge> adj[V];
int dist[V], parent[V], parent_edge[V];
bool in_queue[V];

bool spfa(int s, int t, int V) {
    fill(dist, dist+V, INF);
    fill(in_queue, in_queue+V, false);
    queue<int> q;
    dist[s] = 0;
    q.push(s);
    in_queue[s] = true;

    while (!q.empty()) {
        int u = q.front(); q.pop();
        in_queue[u] = false;
        for (int k = 0; k < adj[u].size(); k++) {
            auto &e = adj[u][k];
            if (e.cap > 0 && dist[e.v] > dist[u] + e.cost) {
                dist[e.v] = dist[u] + e.cost;
                parent[e.v] = u;
                parent_edge[e.v] = k;
                if (!in_queue[e.v]) {
                    q.push(e.v);
                    in_queue[e.v] = true;
                }
            }
        }
    }
    return dist[t] < INF;
}

pair<int,int> min_cost_max_flow(int s, int t, int V) {
    int flow = 0, cost = 0;
    while (spfa(s, t, V)) {
        int f = INF;
        for (int v = t; v != s; v = parent[v]) {
            int u = parent[v];
            auto &e = adj[u][parent_edge[v]];
            f = min(f, e.cap);
        }
        for (int v = t; v != s; v = parent[v]) {
            int u = parent[v];
            auto &e = adj[u][parent_edge[v]];
            e.cap -= f;
            adj[v][e.rev].cap += f;
            cost += f * e.cost;
        }
        flow += f;
    }
    return {flow, cost};
}
```

#### Why It Matters

- Practical speedup over Bellman–Ford, especially in sparse networks.
- Handles negative edges safely (no negative cycles).
- Widely used in competitive programming and network optimization.
- Easier to implement than Dijkstra-based variants.

SPFA combines the simplicity of Bellman–Ford with real-world efficiency, often reducing redundant relaxations dramatically.

#### A Gentle Proof (Why It Works)

Each SPFA run computes a shortest-cost augmenting path.
Augmenting along that path ensures:

- The flow remains feasible,
- The cost strictly decreases, and
- The process terminates after a finite number of augmentations (bounded by total flow).

Because SPFA always respects shortest-path distances, the final flow is cost-optimal.

#### Try It Yourself

1. Replace Bellman–Ford in your MCMF with SPFA.
2. Compare runtime on sparse vs dense graphs.
3. Add an edge with negative cost and verify correct behavior.
4. Visualize queue contents after each relaxation.
5. Measure how many times each vertex is processed.

#### Test Cases

| Graph       | Max Flow | Min Cost | Notes                   |
| ----------- | -------- | -------- | ----------------------- |
| Chain graph | 4        | 12       | Simple                  |
| Sparse DAG  | 10       | 25       | SPFA efficient          |
| Dense graph | 15       | 40       | Similar to Bellman–Ford |

#### Complexity

- Time: $O(F \cdot E)$ on average, $O(F \cdot V \cdot E)$ worst-case
- Space: $O(V + E)$

SPFA-based MCMF blends clarity, practical efficiency, and negative-cost support, making it a favorite for real-world min-cost flow implementations.

### 359 Circulation with Demands

The Circulation with Demands problem is a generalization of the classic flow and min-cost flow problems.
Instead of having just a single source and sink, every node can demand or supply a certain amount of flow.
Your goal is to find a feasible circulation—a flow that satisfies all node demands and capacity limits.

This formulation unifies network flow, feasibility checking, and optimization under one elegant model.

#### What Problem Are We Solving?

Given a directed graph $G = (V, E)$, each edge $(u,v)$ has:

- Lower bound $l(u,v)$
- Upper bound (capacity) $c(u,v)$
- Cost $w(u,v)$

Each vertex $v$ may have a demand $b(v)$ (positive means it needs inflow, negative means it provides outflow).

We want to find a circulation (a flow satisfying $f(u,v)$ for all edges) such that:

1. Capacity constraints:
   $$
   l(u,v) \le f(u,v) \le c(u,v)
   $$
2. Flow conservation:
   $$
   \sum_{(u,v)\in E} f(u,v) - \sum_{(v,u)\in E} f(v,u) = b(v)
   $$
3. Optional cost minimization:
   $$
   \min \sum_{(u,v)\in E} f(u,v),w(u,v)
   $$

A feasible circulation exists if all demands can be met simultaneously.

#### How Does It Work (Plain Language)?

The trick is to transform this into a standard min-cost flow problem with a super-source and super-sink.

Step-by-step transformation:

1. For each edge $(u,v)$ with lower bound $l(u,v)$:

   * Reduce the capacity to $(c(u,v) - l(u,v))$.
   * Subtract $l(u,v)$ from the demands:
     $$
     b(u) \mathrel{{-}{=}} l(u,v), \quad b(v) \mathrel{{+}{=}} l(u,v)
     $$

2. Add a super-source $S$ and super-sink $T$:

   * For each node $v$:

     * If $b(v) > 0$, add edge $(S,v)$ with capacity $b(v)$.
     * If $b(v) < 0$, add edge $(v,T)$ with capacity $-b(v)$.

3. Solve Max Flow (or Min-Cost Flow) from $S$ to $T$.

4. If the max flow = total demand, a feasible circulation exists.
   Otherwise, no circulation satisfies all constraints.

#### Tiny Code (Simplified Example)

```python
def circulation_with_demands(V, edges, demand):
    # edges: (u, v, lower, cap, cost)
    # demand: list of node demands b(v)

    adj = [[] for _ in range(V + 2)]
    S, T = V, V + 1
    b = demand[:]

    for u, v, low, cap, cost in edges:
        cap -= low
        b[u] -= low
        b[v] += low
        adj[u].append((v, cap, cost))
        adj[v].append((u, 0, -cost))

    total_demand = 0
    for i in range(V):
        if b[i] > 0:
            adj[S].append((i, b[i], 0))
            total_demand += b[i]
        elif b[i] < 0:
            adj[i].append((T, -b[i], 0))

    flow, cost = min_cost_max_flow(adj, S, T)
    if flow == total_demand:
        print("Feasible circulation found")
    else:
        print("No feasible circulation")
```

#### Why It Matters

- Unifies multiple problems:
  Many flow formulations—like assignment, transportation, and scheduling—can be expressed as circulations with demands.

- Handles lower bounds elegantly:
  Standard flows can't directly enforce $f(u,v) \ge l(u,v)$. Circulations fix that.

- Foundation for advanced models:
  Cost constraints, multi-commodity flows, and balance equations all build on this.

#### A Gentle Proof (Why It Works)

By enforcing lower bounds, we adjust the net balance of each vertex.
After normalization, the system becomes equivalent to finding a flow from super-source to super-sink satisfying all balances.

If such a flow exists, we can reconstruct the original flow:
$$
f'(u,v) = f(u,v) + l(u,v)
$$
which satisfies all original constraints.

#### Try It Yourself

1. Model a supply-demand network (e.g., factories → warehouses → stores).
2. Add lower bounds to enforce minimum delivery.
3. Introduce node demands to balance supply/consumption.
4. Solve with min-cost flow and verify circulation.
5. Remove cost to just check feasibility.

#### Test Cases

| Network                         | Result     | Notes                |
| ------------------------------- | ---------- | -------------------- |
| Balanced 3-node                 | Feasible   | Simple supply-demand |
| Lower bounds exceed total       | Infeasible | No solution          |
| Mixed positive/negative demands | Feasible   | Requires adjustment  |

#### Complexity

- Time: depends on flow solver used ($O(E^2V)$ for Bellman–Ford variant)
- Space: $O(V + E)$

Circulation with demands gives a universal framework: any linear flow constraint can be encoded, checked, and optimized—turning balance equations into solvable graph problems.

### 360 Successive Shortest Path

The Successive Shortest Path (SSP) algorithm is a clean and intuitive way to solve the Minimum-Cost Maximum-Flow (MCMF) problem.
It builds the optimal flow incrementally, one shortest path at a time, always sending flow along the cheapest available route until capacity or balance constraints stop it.

#### What Problem Are We Solving?

Given a directed graph $G = (V, E)$ with:

- Capacity: $c(u,v)$
- Cost per unit flow: $w(u,v)$
- Source: $s$ and sink: $t$

We want to send the maximum flow from $s$ to $t$ with minimum total cost:

$$
\min \sum_{(u,v)\in E} f(u,v), w(u,v)
$$

subject to:

- $0 \le f(u,v) \le c(u,v)$
- Flow conservation at each vertex except $s,t$

#### How Does It Work (Plain Language)?

SSP proceeds by repeatedly finding the shortest-cost augmenting path from $s$ to $t$ in the residual graph (cost as weight).
It then pushes as much flow as possible along that path.
Residual edges track available capacity (forward and backward).

Algorithm Steps:

1. Initialize flow $f(u,v) = 0$ for all edges.
2. Build residual graph with edge costs $w(u,v)$.
3. While there exists a path $P$ from $s$ to $t$:

   * Find shortest path $P$ by cost (using Dijkstra or Bellman–Ford).
   * Compute bottleneck capacity $\delta = \min_{(u,v)\in P} c_f(u,v)$.
   * Augment flow along $P$:
     $$
     f(u,v) \mathrel{{+}{=}} \delta,\quad f(v,u) \mathrel{{-}{=}} \delta
     $$
   * Update residual capacities and costs.
4. Repeat until no augmenting path remains.
5. Resulting $f$ is the min-cost max-flow.

If negative edges exist, use Bellman–Ford; otherwise Dijkstra with potentials for efficiency.

#### Tiny Code (Simplified Python)

```python
from heapq import heappush, heappop

def successive_shortest_path(V, edges, s, t):
    adj = [[] for _ in range(V)]
    for u, v, cap, cost in edges:
        adj[u].append([v, cap, cost, len(adj[v])])
        adj[v].append([u, 0, -cost, len(adj[u]) - 1])

    INF = 109
    pi = [0]*V  # potentials for reduced cost
    flow = cost = 0

    while True:
        dist = [INF]*V
        parent = [-1]*V
        parent_edge = [-1]*V
        dist[s] = 0
        pq = [(0, s)]
        while pq:
            d, u = heappop(pq)
            if d > dist[u]: continue
            for i, (v, cap, w, rev) in enumerate(adj[u]):
                if cap > 0 and dist[v] > dist[u] + w + pi[u] - pi[v]:
                    dist[v] = dist[u] + w + pi[u] - pi[v]
                    parent[v] = u
                    parent_edge[v] = i
                    heappush(pq, (dist[v], v))
        if dist[t] == INF:
            break
        for v in range(V):
            if dist[v] < INF:
                pi[v] += dist[v]
        f = INF
        v = t
        while v != s:
            u = parent[v]
            e = adj[u][parent_edge[v]]
            f = min(f, e[1])
            v = u
        v = t
        while v != s:
            u = parent[v]
            i = parent_edge[v]
            e = adj[u][i]
            e[1] -= f
            adj[v][e[3]][1] += f
            cost += f * e[2]
            v = u
        flow += f
    return flow, cost
```

#### Why It Matters

- Simple and clear logic, augment along cheapest path each time.
- Optimal solution for min-cost max-flow when no negative cycles exist.
- Can handle large graphs efficiently with reduced costs and potentials.
- Forms the basis for cost-scaling and network simplex methods.

#### A Gentle Proof (Why It Works)

Each augmentation moves flow along a shortest path (minimum reduced cost).
Reduced costs ensure no negative cycles arise, so each augmentation preserves optimality.
Once no augmenting path exists, all reduced costs are nonnegative and $f$ is optimal.

Potentials $\pi(v)$ guarantee Dijkstra finds true shortest paths in transformed graph:
$$
w'(u,v) = w(u,v) + \pi(u) - \pi(v)
$$
maintaining equivalence and non-negativity.

#### Try It Yourself

1. Use a simple 4-node network and trace each path's cost and flow.
2. Compare augmentations using Bellman–Ford vs Dijkstra.
3. Add negative edge costs and test with potential adjustments.
4. Visualize residual graphs after each iteration.
5. Measure cost convergence per iteration.

#### Test Cases

| Graph              | Max Flow | Min Cost | Notes                |
| ------------------ | -------- | -------- | -------------------- |
| Simple 3-node      | 5        | 10       | one path             |
| Two parallel paths | 10       | 15       | chooses cheapest     |
| With negative edge | 7        | 5        | potentials fix costs |

#### Complexity

- Time: $O(F \cdot E \log V)$ using Dijkstra + potentials
- Space: $O(V + E)$

The successive shortest path algorithm is a workhorse for costed flow problems: easy to code, provably correct, and efficient with the right data structures.

## Section 37. Cuts

### 361 Stoer–Wagner Minimum Cut

The Stoer–Wagner Minimum Cut algorithm finds the global minimum cut in an undirected weighted graph.
A *cut* is a partition of vertices into two non-empty sets; its weight is the sum of edges crossing the partition.
The algorithm efficiently discovers the cut with minimum total edge weight, using repeated maximum adjacency searches.

#### What Problem Are We Solving?

Given an undirected graph $G = (V, E)$ with non-negative weights $w(u,v)$, find a cut $(S, V \setminus S)$ such that:

$$
\text{cut}(S) = \sum_{u \in S, v \in V \setminus S} w(u,v)
$$

is minimized over all nontrivial partitions $S \subset V$.

This is called the global minimum cut, distinct from the s–t minimum cut, which fixes two endpoints.

#### How Does It Work (Plain Language)?

The algorithm iteratively merges vertices while keeping track of the tightest cut found along the way.

At each phase:

1. Pick an arbitrary starting vertex.
2. Grow a set $A$ by repeatedly adding the most strongly connected vertex (highest weight to $A$).
3. Continue until all vertices are added.
4. The last added vertex $t$ and the second last $s$ define a cut $(A \setminus {t}, {t})$.
5. Record the cut weight, it's a candidate for the minimum cut.
6. Merge $s$ and $t$ into a single vertex.
7. Repeat until only one vertex remains.

The smallest recorded cut across all phases is the global minimum cut.

#### Step-by-Step Example

Suppose we have 4 vertices $A, B, C, D$.
Each phase:

1. Start with $A$.
2. Add vertex most connected to $A$.
3. Continue until only one vertex remains.
4. Record cut weight each time the last vertex is added.
5. Merge last two vertices, repeat.

After all merges, the lightest cut weight is the minimum cut value.

#### Tiny Code (Python)

```python
def stoer_wagner_min_cut(V, weight):
    n = V
    best = float('inf')
    vertices = list(range(n))

    while n > 1:
        used = [False] * n
        weights = [0] * n
        prev = -1
        for i in range(n):
            # select most connected vertex not yet in A
            sel = -1
            for j in range(n):
                if not used[j] and (sel == -1 or weights[j] > weights[sel]):
                    sel = j
            used[sel] = True
            if i == n - 1:
                # last vertex added, record cut
                best = min(best, weights[sel])
                # merge prev and sel
                if prev != -1:
                    for j in range(n):
                        weight[prev][j] += weight[sel][j]
                        weight[j][prev] += weight[j][sel]
                    vertices.pop(sel)
                    weight.pop(sel)
                    for row in weight:
                        row.pop(sel)
                n -= 1
                break
            prev = sel
            for j in range(n):
                if not used[j]:
                    weights[j] += weight[sel][j]
    return best
```

#### Why It Matters

- Finds the global min cut without fixing $s,t$.
- Works directly on weighted undirected graphs.
- Requires no flow computation.
- Simpler and faster than $O(VE \log V)$ max-flow min-cut for global cut problems.

It's one of the few exact polynomial-time algorithms for global min cut in weighted graphs.

#### A Gentle Proof (Why It Works)

Each phase finds a minimum $s$–$t$ cut separating the last vertex $t$ from the rest.
The merging process maintains equivalence, merging does not destroy the optimality of remaining cuts.
The lightest phase cut corresponds to the global minimum.

By induction over merging steps, the algorithm explores all essential cuts.

#### Try It Yourself

1. Create a triangle graph with different edge weights.
2. Trace $A$–$B$–$C$ order additions and record cut weights.
3. Merge last two vertices, reduce matrix, repeat.
4. Verify cut corresponds to smallest crossing weight.
5. Compare with max-flow min-cut for validation.

#### Test Cases

| Graph                | Minimum Cut       | Notes                    |
| -------------------- | ----------------- | ------------------------ |
| Triangle $w=1,2,3$   | 3                 | Removes edge of weight 3 |
| Square equal weights | 2                 | Any two opposite sides   |
| Weighted complete    | smallest edge sum | Dense test               |

#### Complexity

- Time: $O(V^3)$ using adjacency matrix
- Space: $O(V^2)$

Faster variants exist using adjacency lists and priority queues ($O(VE + V^2 \log V)$).

The Stoer–Wagner algorithm shows that global connectivity fragility can be measured efficiently, one cut at a time, through pure merging insight.

### 362 Karger's Randomized Cut

The Karger's Algorithm is a beautifully simple randomized algorithm to find the global minimum cut of an undirected graph.
Instead of deterministically exploring all partitions, it contracts edges randomly, shrinking the graph until only two supernodes remain.
The sum of edges between them is (with high probability) the minimum cut.

#### What Problem Are We Solving?

Given an undirected, unweighted (or weighted) graph $G = (V, E)$, a cut is a partition of $V$ into two disjoint subsets $(S, V \setminus S)$.
The cut size is the total number (or weight) of edges crossing the partition.

We want to find the global minimum cut:
$$
\min_{S \subset V, S \neq \emptyset, S \neq V} \text{cut}(S)
$$

Karger's algorithm gives a probabilistic guarantee of finding the exact minimum cut.

#### How Does It Work (Plain Language)?

It's almost magical in its simplicity:

1. While there are more than 2 vertices:

   * Pick a random edge $(u, v)$.
   * Contract it, merge $u$ and $v$ into a single vertex.
   * Remove self-loops.
2. When only two vertices remain,

   * The edges between them form a cut.

Repeat the process multiple times to increase success probability.

Each run finds the true minimum cut with probability at least
$$
\frac{2}{n(n-1)}
$$

By repeating $O(n^2 \log n)$ times, the probability of missing it becomes negligible.

#### Example

Consider a triangle graph $(A, B, C)$:

- Randomly pick one edge, say $(A, B)$, contract into supernode $(AB)$.
- Now edges: $(AB, C)$ with multiplicity 2.
- Only 2 nodes remain, cut weight = 2, the min cut.

#### Tiny Code (Python)

```python
import random
import copy

def karger_min_cut(graph):
    # graph: adjacency list {u: [v1, v2, ...]}
    vertices = list(graph.keys())
    edges = []
    for u in graph:
        for v in graph[u]:
            if u < v:
                edges.append((u, v))

    g = copy.deepcopy(graph)
    while len(g) > 2:
        u, v = random.choice(edges)
        # merge v into u
        g[u].extend(g[v])
        for w in g[v]:
            g[w] = [u if x == v else x for x in g[w]]
        del g[v]
        # remove self-loops
        g[u] = [x for x in g[u] if x != u]
        edges = []
        for x in g:
            for y in g[x]:
                if x < y:
                    edges.append((x, y))
    # any remaining edge list gives cut size
    return len(list(g.values())[0])
```

#### Why It Matters

- Elegantly simple yet provably correct with probability.
- No flows, no complex data structures.
- Excellent pedagogical algorithm for randomized reasoning.
- Forms base for improved variants (Karger–Stein, randomized contraction).

Karger's approach demonstrates the power of randomization, minimal logic, maximal insight.

#### A Gentle Proof (Why It Works)

At each step, contracting a non-min-cut edge does not destroy the min cut.
Since there are $O(n^2)$ edges and at least $n-2$ contractions,
probability that we never contract a min-cut edge is:
$$
P = \prod_{i=0}^{n-3} \frac{k_i}{m_i} \ge \frac{2}{n(n-1)}
$$
where $k_i$ is min cut size, $m_i$ is edges remaining.

By repeating enough times, the failure probability decays exponentially.

#### Try It Yourself

1. Run the algorithm 1 time on a 4-node graph, note variability.
2. Repeat 50 times, collect min cut frequencies.
3. Visualize contraction steps on paper.
4. Add weights by expanding weighted edges into parallel copies.
5. Compare runtime vs Stoer–Wagner on dense graphs.

#### Test Cases

| Graph            | Expected Min Cut | Notes                    |
| ---------------- | ---------------- | ------------------------ |
| Triangle         | 2                | always 2                 |
| Square (4-cycle) | 2                | random paths converge    |
| Complete $K_4$   | 3                | dense, repeat to confirm |

#### Complexity

- Time: $O(n^2)$ per trial
- Space: $O(n^2)$ (for adjacency)
- Repetitions: $O(n^2 \log n)$ for high confidence

Karger's algorithm is a landmark example: a single line, *"pick an edge at random and contract it"*, unfolds into a full-fledged, provably correct algorithm for global min-cut discovery.

### 363 Karger–Stein Minimum Cut

The Karger–Stein algorithm is an improved randomized divide-and-conquer version of Karger's original contraction algorithm.
It achieves a higher success probability and better expected runtime, while preserving the same beautifully simple idea: repeatedly contract random edges, but stop early and recurse instead of fully collapsing to two vertices.

#### What Problem Are We Solving?

Given an undirected, weighted or unweighted graph $G = (V, E)$, the goal is to find the global minimum cut, that is:

$$
\min_{S \subset V,, S \neq \emptyset,, S \neq V} \text{cut}(S)
$$

where:

$$
\text{cut}(S) = \sum_{u \in S,, v \in V \setminus S} w(u, v)
$$

#### How Does It Work (Plain Language)?

Like Karger's algorithm, we contract edges randomly, but instead of shrinking to 2 vertices immediately,
we contract only until the graph has about $\frac{n}{\sqrt{2}}$ vertices, then recurse twice independently.
The best cut found across recursive calls is our result.

This strategy amplifies success probability while maintaining efficiency.

Algorithm:

1. If $|V| \le 6$:

   * Compute min cut directly by brute force or basic contraction.
2. Otherwise:

   * Let $t = \lceil n / \sqrt{2} + 1 \rceil$.
   * Randomly contract edges until only $t$ vertices remain.
   * Run the algorithm recursively twice on two independent contractions.
3. Return the smaller of the two cuts.

Each contraction phase preserves the minimum cut with good probability, and recursive repetition compounds success.

#### Example

For a graph of 16 vertices:

- Contract randomly down to $\lceil 16 / \sqrt{2} \rceil = 12$.
- Recurse twice independently.
- Each recursion again halves vertex count until base case.
- Return smallest cut found.

Multiple recursion branches make the overall probability of keeping all min-cut edges much higher than one-shot contraction.

#### Tiny Code (Python)

```python
import random
import math
import copy

def contract_random_edge(graph):
    u, v = random.choice([(u, w) for u in graph for w in graph[u] if u < w])
    # merge v into u
    graph[u].extend(graph[v])
    for w in graph[v]:
        graph[w] = [u if x == v else x for x in graph[w]]
    del graph[v]
    # remove self-loops
    graph[u] = [x for x in graph[u] if x != u]

def karger_stein(graph):
    n = len(graph)
    if n <= 6:
        # base case: fall back to basic Karger
        g_copy = copy.deepcopy(graph)
        while len(g_copy) > 2:
            contract_random_edge(g_copy)
        return len(list(g_copy.values())[0])

    t = math.ceil(n / math.sqrt(2)) + 1
    g1 = copy.deepcopy(graph)
    g2 = copy.deepcopy(graph)
    while len(g1) > t:
        contract_random_edge(g1)
    while len(g2) > t:
        contract_random_edge(g2)

    return min(karger_stein(g1), karger_stein(g2))
```

#### Why It Matters

- Improves success probability to roughly $1 / \log n$, vs $1/n^2$ for basic Karger.
- Divide-and-conquer structure reduces required repetitions.
- Demonstrates power of probability amplification in randomized algorithms.
- Useful for large dense graphs where deterministic $O(V^3)$ algorithms are slower.

It's a near-optimal randomized min-cut algorithm, balancing simplicity and efficiency.

#### A Gentle Proof (Why It Works)

Each contraction preserves the min cut with probability:

$$
p = \prod_{i=k+1}^n \left(1 - \frac{2}{i}\right)
$$

The early stopping at $t = n / \sqrt{2}$ keeps $p$ reasonably high.
Since we recurse twice independently, overall success probability becomes:

$$
P = 1 - (1 - p)^2 \approx 2p
$$

Repeating $O(\log^2 n)$ times ensures high confidence of finding the true minimum cut.

#### Try It Yourself

1. Compare runtime and accuracy vs plain Karger.
2. Run on small graphs and collect success rate over 100 trials.
3. Visualize recursive tree of contractions.
4. Add edge weights (by expanding parallel edges).
5. Confirm returned cut matches Stoer–Wagner result.

#### Test Cases

| Graph        | Expected Min Cut | Notes                      |
| ------------ | ---------------- | -------------------------- |
| Triangle     | 2                | Always found               |
| 4-node cycle | 2                | High accuracy              |
| Dense $K_6$  | 5                | Repeats improve confidence |

#### Complexity

- Expected Time: $O(n^2 \log^3 n)$
- Space: $O(n^2)$
- Repetitions: $O(\log^2 n)$ for high probability

Karger–Stein refines the raw elegance of random contraction into a divide-and-conquer gem, faster, more reliable, and still delightfully simple.

### 364 Gomory–Hu Tree

The Gomory–Hu Tree is a remarkable data structure that compactly represents all-pairs minimum cuts in an undirected weighted graph.
Instead of computing $O(n^2)$ separate cuts, it builds a single tree (with $n-1$ edges) whose edge weights capture every pair's min-cut value.

This structure transforms global connectivity questions into simple tree queries, fast, exact, and elegant.

#### What Problem Are We Solving?

Given an undirected weighted graph $G = (V, E, w)$, we want to find, for every pair $(s, t)$:

$$
\lambda(s, t) = \min_{S \subset V,, s \in S,, t \notin S} \sum_{u \in S, v \notin S} w(u, v)
$$

Instead of running a separate min-cut computation for each pair, we build a Gomory–Hu tree $T$, such that:

> For any pair $(s, t)$,
> the minimum edge weight on the path between $s$ and $t$ in $T$ equals $\lambda(s, t)$.

This tree encodes all-pairs min-cuts compactly.

#### How Does It Work (Plain Language)?

The Gomory–Hu tree is constructed iteratively using $n-1$ minimum-cut computations:

1. Choose a root $r$ (arbitrary).
2. Maintain a partition tree $T$ with vertices as nodes.
3. While there are unprocessed partitions:

   * Pick two vertices $s, t$ in the same partition.
   * Compute the minimum $s$–$t$ cut using any max-flow algorithm.
   * Partition vertices into two sets $(S, V \setminus S)$ along that cut.
   * Add an edge $(s, t)$ in the Gomory–Hu tree with weight equal to the cut value.
   * Recurse on each partition.
4. After $n-1$ cuts, the tree is complete.

Every edge in the tree represents a distinct partition cut in the original graph.

#### Example

Consider a graph with vertices ${A, B, C, D}$ and weighted edges.
We run cuts step by step:

1. Choose $s = A, t = B$ → find cut weight $w(A,B) = 2$
   → Add edge $(A, B, 2)$ to tree.
2. Recurse within $A$'s and $B$'s sides.
3. Continue until tree has $n-1 = 3$ edges.

Now for any $(u,v)$ pair, the min-cut = minimum edge weight along the path connecting them in $T$.

#### Tiny Code (High-Level Skeleton)

```python
def gomory_hu_tree(V, edges):
    # edges: list of (u, v, w)
    from collections import defaultdict
    n = V
    tree = defaultdict(list)
    parent = [0] * n
    cut_value = [0] * n

    for s in range(1, n):
        t = parent[s]
        # compute s-t min cut via max-flow
        mincut, partition = min_cut(s, t, edges)
        cut_value[s] = mincut
        for v in range(n):
            if v != s and parent[v] == t and partition[v]:
                parent[v] = s
        tree[s].append((t, mincut))
        tree[t].append((s, mincut))
        if partition[t]:
            parent[s], parent[t] = parent[t], s
    return tree
```

In practice, `min_cut(s, t)` is computed using Edmonds–Karp or Push–Relabel.

#### Why It Matters

- Captures all-pairs min-cuts in just $n-1$ max-flow computations.
- Transforms graph cut queries into simple tree queries.
- Enables efficient network reliability analysis, connectivity queries, and redundancy planning.
- Works for weighted undirected graphs.

This algorithm compresses rich connectivity information into a single elegant structure.

#### A Gentle Proof (Why It Works)

Each $s$–$t$ min-cut partitions vertices into two sides; merging these partitions iteratively preserves all-pairs min-cut relationships.

By induction, every pair $(u, v)$ is eventually separated by exactly one edge in the tree, with weight equal to $\lambda(u, v)$.
Thus,
$$
\lambda(u, v) = \min_{e \in \text{path}(u, v)} w(e)
$$

#### Try It Yourself

1. Run on a small 4-node weighted graph.
2. Verify each edge weight equals an actual cut value.
3. Query random pairs $(u, v)$ by tree path min.
4. Compare with independent flow-based min-cut computations.
5. Draw both the original graph and the resulting tree.

#### Test Cases

| Graph          | Tree Edges | Notes                              |
| -------------- | ---------- | ---------------------------------- |
| Triangle       | 2 edges    | Uniform weights produce equal cuts |
| Square         | 3 edges    | Distinct cuts form balanced tree   |
| Complete $K_4$ | 3 edges    | Symmetric connectivity             |

#### Complexity

- Time: $O(n \cdot \text{MaxFlow}(V,E))$
- Space: $O(V + E)$
- Queries: $O(\log V)$ (via path min in tree)

The Gomory–Hu tree elegantly transforms the cut landscape of a graph into a tree of truths, one structure, all min-cuts.

### 365 Max-Flow Min-Cut Theorem

The Max-Flow Min-Cut Theorem is one of the foundational results in graph theory and combinatorial optimization.
It reveals a duality between maximum flow (what can pass through a network) and minimum cut (what blocks the network).
The two are not just related, they are exactly equal.

This theorem underpins nearly every algorithm for flows, cuts, and network design.

#### What Problem Are We Solving?

We are working with a directed graph $G = (V, E)$, a source $s$, and a sink $t$.
Each edge $(u,v)$ has a capacity $c(u,v) \ge 0$.

A flow assigns values $f(u,v)$ to edges such that:

1. Capacity constraint: $0 \le f(u,v) \le c(u,v)$
2. Flow conservation: $\sum_v f(u,v) = \sum_v f(v,u)$ for all $u \ne s, t$

The value of the flow is:
$$
|f| = \sum_{v} f(s, v)
$$

We want the maximum possible flow value from $s$ to $t$.

A cut $(S, T)$ is a partition of vertices such that $s \in S$ and $t \in T = V \setminus S$.
The capacity of the cut is:
$$
c(S, T) = \sum_{u \in S, v \in T} c(u, v)
$$

#### The Theorem

> Max-Flow Min-Cut Theorem:
> In every flow network, the maximum value of a feasible flow equals the minimum capacity of an $s$–$t$ cut.

Formally,
$$
\max_f |f| = \min_{(S,T)} c(S, T)
$$

This is a strong duality statement, a maximum over one set equals a minimum over another.

#### How Does It Work (Plain Language)?

1. You try to push as much flow as possible from $s$ to $t$.
2. As you saturate edges, certain parts of the graph become bottlenecks.
3. The final residual graph separates reachable vertices from $s$ (via unsaturated edges) and the rest.
4. This separation $(S, T)$ forms a minimum cut, whose capacity equals the total flow sent.

Thus, once you can't push any more flow, you've also found the tightest cut blocking you.

#### Example

Consider a small network:

| Edge  | Capacity |
| ----- | -------- |
| s → a | 3        |
| s → b | 2        |
| a → t | 2        |
| b → t | 3        |
| a → b | 1        |

Max flow via augmenting paths:

- $s \to a \to t$: 2 units
- $s \to b \to t$: 2 units
- $s \to a \to b \to t$: 1 unit

Total flow = 5.

Min cut: $S = {s, a}$, $T = {b, t}$ → capacity = 5.
Flow = Cut = 5 ✔

#### Tiny Code (Verification via Edmonds–Karp)

```python
from collections import deque

def bfs(cap, flow, s, t, parent):
    n = len(cap)
    visited = [False]*n
    q = deque([s])
    visited[s] = True
    while q:
        u = q.popleft()
        for v in range(n):
            if not visited[v] and cap[u][v] - flow[u][v] > 0:
                parent[v] = u
                visited[v] = True
                if v == t: return True
                q.append(v)
    return False

def max_flow_min_cut(cap, s, t):
    n = len(cap)
    flow = [[0]*n for _ in range(n)]
    parent = [-1]*n
    max_flow = 0
    while bfs(cap, flow, s, t, parent):
        v = t
        path_flow = float('inf')
        while v != s:
            u = parent[v]
            path_flow = min(path_flow, cap[u][v] - flow[u][v])
            v = u
        v = t
        while v != s:
            u = parent[v]
            flow[u][v] += path_flow
            flow[v][u] -= path_flow
            v = u
        max_flow += path_flow
    return max_flow
```

The final reachable set from $s$ in the residual graph defines the minimum cut.

#### Why It Matters

- Fundamental theorem linking optimization and combinatorics.
- Basis for algorithms like Ford–Fulkerson, Edmonds–Karp, Dinic, Push–Relabel.
- Used in image segmentation, network reliability, scheduling, bipartite matching, clustering, and transportation.

It bridges the gap between flow maximization and cut minimization, two faces of the same coin.

#### A Gentle Proof (Why It Works)

At termination of Ford–Fulkerson:

- No augmenting path exists in the residual graph.
- Let $S$ be all vertices reachable from $s$.
- Then all edges $(u,v)$ with $u \in S, v \notin S$ are saturated.

Thus:
$$
|f| = \sum_{u \in S, v \notin S} f(u,v) = \sum_{u \in S, v \notin S} c(u,v) = c(S, T)
$$
and no cut can have smaller capacity.
Therefore, $\max |f| = \min c(S,T)$.

#### Try It Yourself

1. Build a simple 4-node network and trace augmenting paths.
2. Identify the cut $(S, T)$ at termination.
3. Verify total flow equals cut capacity.
4. Test different max-flow algorithms, result remains identical.
5. Visualize cut edges in residual graph.

#### Test Cases

| Graph          | Max Flow | Min Cut | Notes                      |
| -------------- | -------- | ------- | -------------------------- |
| Simple chain   | 5        | 5       | identical                  |
| Parallel edges | 8        | 8       | same result                |
| Diamond graph  | 6        | 6       | min cut = bottleneck edges |

#### Complexity

- Depends on underlying flow algorithm (e.g. $O(VE^2)$ for Edmonds–Karp)
- Cut extraction: $O(V + E)$

The Max-Flow Min-Cut Theorem is the heartbeat of network optimization, every augmenting path is a step toward equality between two fundamental perspectives: sending and separating.

### 366 Stoer–Wagner Repeated Phase

The Stoer–Wagner Repeated Phase algorithm is a refinement of the Stoer–Wagner minimum cut algorithm for undirected weighted graphs.
It finds the global minimum cut by repeating a maximum adjacency search phase multiple times, progressively merging vertices and tracking cut weights.

This algorithm is elegant, deterministic, and runs in polynomial time, often faster than flow-based approaches for undirected graphs.

#### What Problem Are We Solving?

We are finding the minimum cut in an undirected, weighted graph $G = (V, E)$, where each edge $(u, v)$ has a non-negative weight $w(u, v)$.

A cut $(S, V \setminus S)$ partitions the vertex set into two disjoint subsets.
The weight of a cut is:
$$
w(S, V \setminus S) = \sum_{u \in S, v \notin S} w(u, v)
$$

Our goal is to find a cut $(S, T)$ that minimizes this sum.

#### Core Idea

The algorithm works by repeatedly performing "phases".
Each phase identifies a minimum $s$–$t$ cut in the current contracted graph and merges the last two added vertices.
Over multiple phases, the smallest cut discovered is the global minimum.

Each phase follows a maximum adjacency search pattern, similar to Prim's algorithm but in reverse logic.

#### How It Works (Plain Language)

Each phase:

1. Pick an arbitrary start vertex.
2. Maintain a set $A$ of added vertices.
3. At each step, add the most tightly connected vertex $v$ not in $A$ (the one with maximum total edge weight to $A$).
4. Continue until all vertices are in $A$.
5. Let $s$ be the second-to-last added vertex, and $t$ the last added.
6. The cut separating $t$ from the rest is a candidate min cut.
7. Record its weight; then merge $s$ and $t$ into a supervertex and repeat.

After $|V| - 1$ phases, the smallest cut seen is the global minimum cut.

#### Example

Graph:

| Edge | Weight |
| ---- | ------ |
| A–B  | 3      |
| A–C  | 2      |
| B–C  | 4      |
| B–D  | 2      |
| C–D  | 3      |

Phase 1:

- Start with $A = {A}$
- Add $B$ (max connected to A: 3)
- Add $C$ (max connected to ${A,B}$: total 6)
- Add $D$ last → Cut weight = sum of edges from D to ${A,B,C}$ = 5

Record min cut = 5. Merge $(C,D)$ and continue.

Repeat phases → global min cut = 5.

#### Tiny Code (Simplified Python)

```python
def stoer_wagner_min_cut(graph):
    n = len(graph)
    vertices = list(range(n))
    min_cut = float('inf')

    while len(vertices) > 1:
        added = [False] * n
        weights = [0] * n
        prev = -1
        for _ in range(len(vertices)):
            u = max(vertices, key=lambda v: weights[v] if not added[v] else -1)
            added[u] = True
            if _ == len(vertices) - 1:
                # Last vertex added, potential cut
                min_cut = min(min_cut, weights[u])
                # Merge u into prev
                if prev != -1:
                    for v in vertices:
                        if v != u and v != prev:
                            graph[prev][v] += graph[u][v]
                            graph[v][prev] = graph[prev][v]
                    vertices.remove(u)
                break
            prev = u
            for v in vertices:
                if not added[v]:
                    weights[v] += graph[u][v]
    return min_cut
```

Graph is given as an adjacency matrix.
Each phase picks the tightest vertex, records the cut, and merges nodes.

#### Why It Matters

- Deterministic and elegant for undirected weighted graphs.
- Faster than running multiple max-flow computations.
- Ideal for network reliability, graph partitioning, clustering, and circuit design.
- Each phase mimics "tightening" the graph until only one supervertex remains.

#### A Gentle Proof (Why It Works)

Each phase finds the minimum $s$–$t$ cut where $t$ is the last added vertex.
By merging $s$ and $t$, we preserve all other possible cuts.
The smallest of these phase cuts is the global minimum, as each cut in the merged graph corresponds to one in the original graph.

Formally:
$$
\text{mincut}(G) = \min_{\text{phases}} w(S, V \setminus S)
$$

This inductive structure ensures optimality.

#### Try It Yourself

1. Run on a 4-node complete graph with random weights.
2. Trace vertex addition order in each phase.
3. Record cut weights per phase.
4. Compare with brute-force all cuts, they match.
5. Visualize contraction steps.

#### Test Cases

| Graph                    | Edges | Min Cut         | Notes             |
| ------------------------ | ----- | --------------- | ----------------- |
| Triangle (equal weights) | 3     | 2×weight        | all equal         |
| Square (unit weights)    | 4     | 2               | opposite sides    |
| Weighted grid            | 6     | smallest bridge | cut at bottleneck |

#### Complexity

- Each phase: $O(V^2)$
- Total: $O(V^3)$ for adjacency matrix

With Fibonacci heaps, can improve to $O(V^2 \log V + VE)$.

The Stoer–Wagner repeated phase algorithm is a powerful, purely combinatorial tool, no flows, no residual graphs, just tight connectivity and precise merging toward the true global minimum cut.

### 367 Dynamic Min Cut

The Dynamic Minimum Cut problem extends classical min-cut computation to graphs that change over time, edges can be added, deleted, or updated.
Instead of recomputing from scratch after each change, we maintain data structures that update the min cut efficiently.

Dynamic min-cut algorithms are critical in applications like network resilience, incremental optimization, and real-time systems where connectivity evolves.

#### What Problem Are We Solving?

Given a graph $G = (V, E)$ with weighted edges and a current minimum cut, how do we efficiently maintain this cut when:

- An edge $(u, v)$ is inserted
- An edge $(u, v)$ is deleted
- An edge $(u, v)$ has its weight changed

The naive approach recomputes min-cut from scratch using an algorithm like Stoer–Wagner ($O(V^3)$).
Dynamic algorithms aim for faster incremental updates.

#### Core Idea

The min cut is sensitive only to edges crossing the cut boundary.
When the graph changes, only a local region around modified edges can alter the global cut.

Dynamic algorithms use:

- Dynamic trees (e.g. Link-Cut Trees)
- Fully dynamic connectivity structures
- Randomized contraction tracking
- Incremental recomputation of affected regions

to update efficiently instead of re-running full min-cut.

#### How It Works (Plain Language)

1. Maintain a representation of cuts, often as trees or partition sets.
2. When an edge weight changes, update connectivity info:

   * If edge lies within one side of the cut, no change.
   * If edge crosses the cut, update cut capacity.
3. When an edge is added or deleted, adjust affected components and recalculate locally.
4. Optionally, run periodic global recomputations to correct drift after many updates.

These methods trade exactness for efficiency, often maintaining approximate min cuts within small error bounds.

#### Example (High-Level)

Graph has vertices ${A, B, C, D}$ with current min cut $({A, B}, {C, D})$, weight $5$.

1. Add edge $(B, C)$ with weight $1$:

   * New crossing edge, cut weight becomes $5 + 1 = 6$.
   * Check if alternate partition gives smaller total, update if needed.

2. Delete edge $(A, C)$:

   * Remove from cut set.
   * If this edge was essential to connectivity, min cut might increase.
   * Recompute local cut if affected.

#### Tiny Code (Sketch, Python)

```python
class DynamicMinCut:
    def __init__(self, graph):
        self.graph = graph
        self.min_cut_value = self.compute_min_cut()

    def compute_min_cut(self):
        # Use Stoer–Wagner or flow-based method
        return stoer_wagner(self.graph)

    def update_edge(self, u, v, new_weight):
        self.graph[u][v] = new_weight
        self.graph[v][u] = new_weight
        # Locally recompute affected region
        self.min_cut_value = self.recompute_local(u, v)

    def recompute_local(self, u, v):
        # Simplified placeholder: recompute fully if small graph
        return self.compute_min_cut()
```

For large graphs, replace `recompute_local` with incremental cut update logic.

#### Why It Matters

- Real-time systems need quick responses to network changes.
- Streaming graphs (e.g. traffic, social, or power networks) evolve continuously.
- Reliability analysis in dynamic systems relies on up-to-date min-cut values.

Dynamic maintenance saves time compared to recomputing from scratch at every step.

#### A Gentle Proof (Why It Works)

Let $C_t$ be the min cut after $t$ updates.
If each update affects only local structure, then:
$$
C_{t+1} = \min(C_t, \text{local adjustment})
$$
Maintaining a certificate structure (like a Gomory–Hu tree) ensures correctness since all pairwise min-cuts are preserved under local changes, except where updated edges are involved.

Recomputing only affected cuts guarantees correctness with amortized efficiency.

#### Try It Yourself

1. Build a small weighted graph (5–6 nodes).
2. Compute initial min cut using Stoer–Wagner.
3. Add or delete edges, one at a time.
4. Update only affected cuts manually.
5. Compare to full recomputation, results should match.

#### Test Cases

| Operation                          | Description        | New Min Cut              |
| ---------------------------------- | ------------------ | ------------------------ |
| Add edge $(u,v)$ with small weight | New crossing edge  | Possibly smaller cut     |
| Increase edge weight               | Strengthens bridge | Cut may change elsewhere |
| Delete edge across cut             | Weakens connection | Cut may increase         |
| Delete edge inside partition       | No change          |,                        |

#### Complexity

| Operation                     | Time                      |
| ----------------------------- | ------------------------- |
| Naive recomputation           | $O(V^3)$                  |
| Dynamic approach (randomized) | $O(V^2 \log V)$ amortized |
| Approximate dynamic cut       | $O(E \log^2 V)$           |

Dynamic min-cut algorithms balance exactness with responsiveness, maintaining near-optimal connectivity insight as graphs evolve in real time.

### 368 Minimum s–t Cut (Edmonds–Karp)

The Minimum s–t Cut problem seeks the smallest total capacity of edges that must be removed to separate a source vertex $s$ from a sink vertex $t$.
It's the dual counterpart to maximum flow, and the Edmonds–Karp algorithm provides a clear path to compute it using BFS-based augmenting paths.

#### What Problem Are We Solving?

Given a directed, weighted graph $G=(V,E)$ with capacities $c(u,v)$, find a partition of $V$ into two disjoint sets $(S, T)$ such that:

- $s \in S$, $t \in T$
- The sum of capacities of edges from $S$ to $T$ is minimum

Formally:
$$
\text{min-cut}(s, t) = \min_{(S,T)} \sum_{u \in S, v \in T} c(u, v)
$$

This cut value equals the maximum flow value from $s$ to $t$ by the Max-Flow Min-Cut Theorem.

#### How It Works (Plain Language)

The algorithm uses Edmonds–Karp, a BFS-based version of Ford–Fulkerson, to find the maximum flow first.
After computing max-flow, the reachable vertices from $s$ in the residual graph determine the min-cut.

Steps:

1. Initialize flow $f(u, v) = 0$ for all edges.
2. While there exists a BFS path from $s$ to $t$ in the residual graph:

   * Compute bottleneck capacity along path.
   * Augment flow along path.
3. After no augmenting path exists:

   * Run BFS one last time from $s$ in residual graph.
   * Vertices reachable from $s$ form set $S$.
   * Others form $T$.
4. The edges crossing from $S$ to $T$ with full capacity are the min-cut edges.

#### Example

Graph:

- Vertices: ${s, a, b, t}$
- Edges:

  * $(s, a) = 3$
  * $(s, b) = 2$
  * $(a, b) = 1$
  * $(a, t) = 2$
  * $(b, t) = 3$

1. Run Edmonds–Karp to find max-flow = 4.
2. Residual graph:

   * Reachable set from $s$: ${s, a}$
   * Unreachable set: ${b, t}$
3. Min-cut edges: $(a, t)$ and $(s, b)$
4. Min-cut value = $2 + 2 = 4$
   Matches max-flow value.

#### Tiny Code (C-like Pseudocode)

```c
int min_st_cut(Graph *G, int s, int t) {
    int maxflow = edmonds_karp(G, s, t);
    bool visited[V];
    bfs_residual(G, s, visited);
    int cut_value = 0;
    for (edge (u,v) in G->edges)
        if (visited[u] && !visited[v])
            cut_value += G->capacity[u][v];
    return cut_value;
}
```

#### Why It Matters

- Reveals bottlenecks in a network.
- Key for reliability and segmentation problems.
- Foundational for image segmentation, network design, and flow decomposition.
- Directly supports duality proofs between optimization problems.

#### A Gentle Proof (Why It Works)

The Max-Flow Min-Cut Theorem states:

$$
\max_{\text{flow } f} \sum_{v} f(s, v) = \min_{(S, T)} \sum_{u \in S, v \in T} c(u, v)
$$

Edmonds–Karp finds the maximum flow by repeatedly augmenting along shortest paths (BFS order).
Once no more augmenting paths exist, the residual graph partitions nodes naturally into $S$ and $T$, and the edges from $S$ to $T$ define the min cut.

#### Try It Yourself

1. Build a small directed network with capacities.
2. Run Edmonds–Karp manually (trace augmenting paths).
3. Draw residual graph and find reachable set from $s$.
4. Mark crossing edges, sum their capacities.
5. Compare with max-flow value.

#### Test Cases

| Graph                              | Max Flow        | Min Cut         | Matches? |
| ---------------------------------- | --------------- | --------------- | -------- |
| Simple 4-node                      | 4               | 4               | ✅        |
| Linear chain $s \to a \to b \to t$ | min edge        | min edge        | ✅        |
| Parallel paths                     | sum of min caps | sum of min caps | ✅        |

#### Complexity

| Step                 | Time      |
| -------------------- | --------- |
| BFS per augmentation | $O(E)$    |
| Augmentations        | $O(VE)$   |
| Total                | $O(VE^2)$ |

The Minimum s–t Cut via Edmonds–Karp is an elegant bridge between flow algorithms and partition reasoning, every edge in the cut tells a story of constraint, capacity, and balance.

### 369 Approximate Min Cut

The Approximate Minimum Cut algorithm provides a way to estimate the minimum cut of a graph faster than exact algorithms, especially when exact precision is not critical.
It's built on randomization and sampling, using probabilistic reasoning to find small cuts with high confidence in large or dynamic graphs.

#### What Problem Are We Solving?

Given a weighted, undirected graph $G=(V, E)$, we want a cut $(S, T)$ such that the cut capacity is close to the true minimum:

$$
w(S, T) \le (1 + \epsilon) \cdot \lambda(G)
$$

where $\lambda(G)$ is the weight of the global min cut, and $\epsilon$ is a small error tolerance (e.g. $0.1$).

The goal is speed: approximate min-cut algorithms run in near-linear time, much faster than exact ones ($O(V^3)$).

#### How It Works (Plain Language)

Approximate algorithms rely on two key principles:

1. Random Sampling:
   Randomly sample edges with probability proportional to their weight.
   The smaller the edge capacity, the more likely it's critical to the min cut.

2. Graph Sparsification:
   Build a smaller "sketch" of the graph that preserves cut weights approximately.
   Compute the min cut on this sparse graph, it's close to the true value.

By repeating sampling several times and taking the minimum found cut, we converge to a near-optimal solution.

#### Algorithm Sketch (Karger's Sampling Method)

1. Input: Graph $G(V, E)$ with $n = |V|$ and $m = |E|$
2. Choose sampling probability $p = \frac{c \log n}{\epsilon^2 \lambda}$
3. Build sampled graph $G'$:

   * Include each edge $(u, v)$ with probability $p$
   * Scale included edge weights by $\frac{1}{p}$
4. Run an exact min cut algorithm (Stoer–Wagner) on $G'$
5. Repeat sampling $O(\log n)$ times; take the best cut found

The result approximates $\lambda(G)$ within factor $(1 + \epsilon)$ with high probability.

#### Example

Suppose $G$ has $10^5$ edges, and exact Stoer–Wagner would be too slow.

1. Choose $\epsilon = 0.1$, $p = 0.02$
2. Sample $2%$ of edges randomly (2000 edges)
3. Reweight sampled edges by $\frac{1}{0.02} = 50$
4. Run exact min cut on this smaller graph
5. Repeat 5–10 times; pick smallest cut

Result: A cut within $10%$ of optimal, in a fraction of the time.

#### Tiny Code (Python-like Pseudocode)

```python
def approximate_min_cut(G, epsilon=0.1, repeats=10):
    best_cut = float('inf')
    for _ in range(repeats):
        p = compute_sampling_probability(G, epsilon)
        G_sample = sample_graph(G, p)
        cut_value = stoer_wagner(G_sample)
        best_cut = min(best_cut, cut_value)
    return best_cut
```

#### Why It Matters

- Scalability: Handles huge graphs where exact methods are infeasible
- Speed: Near-linear time using randomization
- Applications:

  * Streaming graphs
  * Network reliability
  * Clustering and partitioning
  * Graph sketching and sparsification

Approximate min cuts are crucial when you need quick, robust decisions, not perfect answers.

#### A Gentle Proof (Why It Works)

Karger's analysis shows that each small cut is preserved with high probability if enough edges are sampled:

$$
\Pr[\text{cut weight preserved}] \ge 1 - \frac{1}{n^2}
$$

By repeating the process $O(\log n)$ times, we amplify confidence, ensuring that with high probability, at least one sampled graph maintains the true min-cut structure.

Using Chernoff bounds, the error is bounded by $(1 \pm \epsilon)$.

#### Try It Yourself

1. Generate a random graph with 50 nodes and random weights.
2. Compute exact min cut using Stoer–Wagner.
3. Sample 10%, 5%, and 2% of edges, compute approximate cuts.
4. Compare results and runtime.
5. Adjust $\epsilon$ and observe trade-off between speed and accuracy.

#### Test Cases

| Graph                    | Exact Min Cut | Approx. (ε=0.1) | Error | Speedup |
| ------------------------ | ------------- | --------------- | ----- | ------- |
| Small dense (100 edges)  | 12            | 13              | 8%    | 5×      |
| Medium sparse (1k edges) | 8             | 8               | 0%    | 10×     |
| Large (100k edges)       | 30            | 33              | 10%   | 50×     |

#### Complexity

| Method                    | Time                         | Accuracy              |
| ------------------------- | ---------------------------- | --------------------- |
| Stoer–Wagner              | $O(V^3)$                     | Exact                 |
| Karger (Randomized Exact) | $O(V^2 \log^3 V)$            | Exact (probabilistic) |
| Approximate Sampling      | $O(E \log^2 V / \epsilon^2)$ | $(1 + \epsilon)$      |

Approximate min-cut algorithms show that probability can replace precision when scale demands speed, they slice through massive graphs with surprising efficiency.

### 370 Min k-Cut

The Minimum k-Cut problem generalizes the classic min-cut idea.
Instead of splitting a graph into just two parts, the goal is to partition it into k disjoint subsets while cutting edges of minimum total weight.

It's a key problem in clustering, parallel processing, and network design, where you want multiple disconnected regions with minimal interconnection cost.

#### What Problem Are We Solving?

Given a weighted, undirected graph $G = (V, E)$ and an integer $k$,
find a partition of $V$ into $k$ subsets ${V_1, V_2, \ldots, V_k}$ such that:

- $V_i \cap V_j = \emptyset$ for all $i \ne j$
- $\bigcup_i V_i = V$
- The sum of edge weights crossing between parts is minimized

Formally:

$$
\text{min-}k\text{-cut}(G) = \min_{V_1, \ldots, V_k} \sum_{\substack{(u,v) \in E \ u \in V_i, v \in V_j, i \ne j}} w(u,v)
$$

For $k=2$, this reduces to the standard min-cut problem.

#### How It Works (Plain Language)

The Min k-Cut problem is NP-hard for general $k$,
but several algorithms provide exact solutions for small $k$ and approximations for larger $k$.

There are two main approaches:

1. Greedy Iterative Cutting:

   * Repeatedly find and remove a global min cut, splitting the graph into components one by one.
   * After $k-1$ cuts, you have $k$ components.
   * Works well but not always optimal.

2. Dynamic Programming over Trees (Exact for small k):

   * Use a tree decomposition of the graph.
   * Compute optimal partition by exploring edge removals in the minimum spanning tree.
   * Based on the Karger–Stein framework.

#### Example

Graph with 5 nodes and edges:

| Edge   | Weight |
| ------ | ------ |
| (A, B) | 1      |
| (B, C) | 2      |
| (C, D) | 3      |
| (D, E) | 4      |
| (A, E) | 2      |

Goal: partition into $k=3$ subsets.

1. Find minimum cut edges to remove:

   * Cut $(A, B)$ (weight 1)
   * Cut $(B, C)$ (weight 2)
2. Total cut weight = $3$
3. Resulting subsets: ${A}, {B}, {C, D, E}$

#### Tiny Code (Python-like Pseudocode)

```python
def min_k_cut(graph, k):
    cuts = []
    G = graph.copy()
    for _ in range(k - 1):
        cut_value, (S, T) = stoer_wagner(G)
        cuts.append(cut_value)
        G = G.subgraph(S)  # Keep one component, remove crossing edges
    return sum(cuts)
```

For small $k$, you can use recursive contraction (like Karger's algorithm)
or dynamic programming on tree structures.

#### Why It Matters

- Clustering: Group nodes into $k$ balanced communities.
- Parallel Computing: Partition workloads while minimizing communication cost.
- Image Segmentation: Divide pixels into $k$ coherent regions.
- Graph Simplification: Split networks into modular subgraphs.

Min k-Cut transforms connectivity into structured modularity.

#### A Gentle Proof (Why It Works)

Each cut increases the number of connected components by 1.
Thus, performing $k-1$ cuts produces exactly $k$ components.

Let $C_1, C_2, \ldots, C_{k-1}$ be the successive minimum cuts.
The sum of their weights bounds the global optimum:

$$
\text{min-}k\text{-cut} \le \sum_{i=1}^{k-1} \lambda_i
$$

where $\lambda_i$ is the $i$-th smallest cut value.
Iterative min-cuts often approximate the optimal $k$-cut well.

For exact solutions, algorithms based on flow decomposition or tree contractions
use recursive partitioning to explore combinations of edges.

#### Try It Yourself

1. Construct a small weighted graph with 6–8 vertices.
2. Run Stoer–Wagner to find the first min cut.
3. Remove edges, repeat for next cut.
4. Compare total cut weight to brute-force partition for $k=3$.
5. Observe approximation quality.

#### Test Cases

| Graph        | k | Exact | Greedy | Error |
| ------------ | - | ----- | ------ | ----- |
| Triangle     | 3 | 3     | 3      | 0%    |
| Line of 5    | 3 | 3     | 3      | 0%    |
| Dense 6-node | 3 | 12    | 13     | 8%    |

#### Complexity

| Algorithm                 | Time              | Type              |
| ------------------------- | ----------------- | ----------------- |
| Brute Force               | Exponential       | Exact             |
| Greedy Cuts               | $O(k \cdot V^3)$  | Approx            |
| Tree DP                   | $O(V^{k-1})$      | Exact (small $k$) |
| Randomized (Karger–Stein) | $O(V^2 \log^3 V)$ | Approx            |

The Min k-Cut problem generalizes connectivity design —
it's where graph partitioning meets optimization, balancing efficiency and modularity.

## Section 38. Matchings 

### 371 Bipartite Matching (DFS)

Bipartite Matching is one of the most fundamental problems in graph theory.
Given a bipartite graph $G=(U,V,E)$, the goal is to find the maximum number of pairings between nodes in $U$ and $V$ such that no two edges share a vertex.
A simple and intuitive way to do this is by using DFS-based augmenting paths.

#### What Problem Are We Solving?

We are given a bipartite graph with two disjoint sets $U$ and $V$.
We want to find the maximum matching, the largest set of edges where each vertex is matched to at most one partner.

Formally, find a subset $M \subseteq E$ such that:

- Each vertex in $U \cup V$ is incident to at most one edge in $M$
- $|M|$ is maximized

#### How It Works (Plain Language)

The core idea is to build matchings incrementally by finding augmenting paths —
paths that start from an unmatched vertex in $U$, alternate between unmatched and matched edges, and end at an unmatched vertex in $V$.

Each time such a path is found, we flip the match status along it (matched edges become unmatched, and vice versa), increasing the total matching size by one.

Steps:

1. Start with an empty matching $M$.
2. For each vertex $u \in U$:

   * Run DFS to find an augmenting path to a free vertex in $V$.
   * If found, augment the matching along this path.
3. Repeat until no more augmenting paths exist.

#### Example

Let $U = {u_1, u_2, u_3}$, $V = {v_1, v_2, v_3}$, and edges:

- $(u_1, v_1)$, $(u_1, v_2)$
- $(u_2, v_2)$
- $(u_3, v_3)$

1. Start with empty matching.
2. DFS from $u_1$: finds $(u_1, v_1)$ → match.
3. DFS from $u_2$: finds $(u_2, v_2)$ → match.
4. DFS from $u_3$: finds $(u_3, v_3)$ → match.
   All vertices matched, maximum matching size = 3.

#### Tiny Code (C-like Pseudocode)

```c
#define MAXV 100
vector<int> adj[MAXV];
int matchR[MAXV], visited[MAXV];

bool dfs(int u) {
    for (int v : adj[u]) {
        if (visited[v]) continue;
        visited[v] = 1;
        if (matchR[v] == -1 || dfs(matchR[v])) {
            matchR[v] = u;
            return true;
        }
    }
    return false;
}

int maxBipartiteMatching(int U) {
    memset(matchR, -1, sizeof(matchR));
    int result = 0;
    for (int u = 0; u < U; ++u) {
        memset(visited, 0, sizeof(visited));
        if (dfs(u)) result++;
    }
    return result;
}
```

#### Why It Matters

- Foundation for Hungarian Algorithm and Hopcroft–Karp
- Core tool in resource allocation, scheduling, pairing problems
- Introduces concept of augmenting paths, a cornerstone in flow theory

Used in:

- Assigning workers to jobs
- Matching students to projects
- Network flow initialization
- Graph theory teaching and visualization

#### A Gentle Proof (Why It Works)

If there exists an augmenting path, flipping the matching along it always increases the matching size by 1.

Let $M$ be a current matching.
If no augmenting path exists, $M$ is maximum (Berge's Lemma):

> A matching is maximum if and only if there is no augmenting path.

Therefore, repeatedly augmenting ensures convergence to the maximum matching.

#### Try It Yourself

1. Draw a bipartite graph with 4 nodes on each side.
2. Use DFS to find augmenting paths manually.
3. Track which vertices are matched/unmatched after each augmentation.
4. Stop when no augmenting paths remain.

#### Test Cases

| Graph | $|U|$ | $|V|$ | Max Matching | Steps |
|-------|------|------|---------------|--------|
| Complete $K_{3,3}$ | 3 | 3 | 3 | 3 DFS |
| Chain $u_1v_1, u_2v_2, u_3v_3$ | 3 | 3 | 3 | 3 DFS |
| Sparse graph | 4 | 4 | 2 | 4 DFS |

#### Complexity

| Aspect         | Cost     |
| -------------- | -------- |
| DFS per vertex | $O(E)$   |
| Total          | $O(VE)$  |
| Space          | $O(V+E)$ |

This DFS-based approach gives an intuitive baseline for bipartite matching, later improved by Hopcroft–Karp ($O(E\sqrt{V})$) but perfect for learning and small graphs.

### 372 Hopcroft–Karp

The Hopcroft–Karp algorithm is a classic improvement over DFS-based bipartite matching.
It uses layered BFS and DFS to find multiple augmenting paths in parallel, reducing redundant searches and achieving an optimal runtime of $O(E\sqrt{V})$.

#### What Problem Are We Solving?

Given a bipartite graph $G = (U, V, E)$, we want to find a maximum matching, the largest set of vertex-disjoint edges connecting $U$ to $V$.

A matching is a subset $M \subseteq E$ such that each vertex is incident to at most one edge in $M$.
The algorithm seeks the maximum cardinality matching.

#### How It Works (Plain Language)

Instead of finding one augmenting path at a time (like simple DFS),
Hopcroft–Karp finds a layer of shortest augmenting paths, then augments all of them together.
This drastically reduces the number of BFS-DFS phases.

Key idea:

- Each phase increases the matching by many paths at once.
- The distance (layer) of unmatched vertices strictly increases after each phase.

Steps:

1. Initialization: Start with empty matching $M = \emptyset$.
2. Repeat until no augmenting path exists:

   1. BFS phase:

      * Build a layered graph (level graph) from unmatched vertices in $U$ to unmatched vertices in $V$.
      * Each layer increases by 1 hop.
   2. DFS phase:

      * Find vertex-disjoint augmenting paths in this layered graph.
      * Augment along all of them simultaneously.
3. Return the total matching size.

#### Example

Let $U = {u_1, u_2, u_3}$, $V = {v_1, v_2, v_3}$, edges:

- $u_1 \to v_1, v_2$
- $u_2 \to v_2$
- $u_3 \to v_3$

1. Initial matching: empty
2. BFS builds layers:

   * Level 0: $u_1, u_2, u_3$
   * Level 1: $v_1, v_2, v_3$
     All are reachable.
3. DFS finds 3 augmenting paths:

   * $u_1 \to v_1$, $u_2 \to v_2$, $u_3 \to v_3$
4. Augment all → Matching size = 3
   No more augmenting paths → Maximum matching = 3

#### Tiny Code (C-like Pseudocode)

```c
vector<int> adj[MAXV];
int pairU[MAXV], pairV[MAXV], dist[MAXV];
int NIL = 0, INF = 1e9;

bool bfs(int U) {
    queue<int> q;
    for (int u = 1; u <= U; u++) {
        if (pairU[u] == NIL) { dist[u] = 0; q.push(u); }
        else dist[u] = INF;
    }
    dist[NIL] = INF;
    while (!q.empty()) {
        int u = q.front(); q.pop();
        if (dist[u] < dist[NIL]) {
            for (int v : adj[u]) {
                if (dist[pairV[v]] == INF) {
                    dist[pairV[v]] = dist[u] + 1;
                    q.push(pairV[v]);
                }
            }
        }
    }
    return dist[NIL] != INF;
}

bool dfs(int u) {
    if (u == NIL) return true;
    for (int v : adj[u]) {
        if (dist[pairV[v]] == dist[u] + 1 && dfs(pairV[v])) {
            pairV[v] = u; pairU[u] = v;
            return true;
        }
    }
    dist[u] = INF;
    return false;
}

int hopcroftKarp(int U) {
    memset(pairU, 0, sizeof(pairU));
    memset(pairV, 0, sizeof(pairV));
    int matching = 0;
    while (bfs(U)) {
        for (int u = 1; u <= U; u++)
            if (pairU[u] == NIL && dfs(u))
                matching++;
    }
    return matching;
}
```

#### Why It Matters

- Efficient: $O(E\sqrt{V})$, ideal for large graphs
- Foundational in:

  * Job assignment
  * Resource allocation
  * Stable match foundations
  * Network optimization

It's the standard method for maximum bipartite matching in competitive programming and real systems.

#### A Gentle Proof (Why It Works)

Each BFS-DFS phase finds a set of shortest augmenting paths.
After augmenting, no shorter path remains.

Let $d$ = distance to the nearest unmatched vertex in BFS.
Every phase increases the minimum augmenting path length,
and the number of phases is at most $O(\sqrt{V})$ (Hopcroft–Karp Lemma).

Each BFS-DFS costs $O(E)$, so total = $O(E\sqrt{V})$.

#### Try It Yourself

1. Draw a bipartite graph with 5 nodes on each side.
2. Run one BFS layer build.
3. Use DFS to find all shortest augmenting paths.
4. Augment all, track matching size per phase.

#### Test Cases

| Graph | $|U|$ | $|V|$ | Matching Size | Complexity |
|-------|------|------|----------------|-------------|
| $K_{3,3}$ | 3 | 3 | 3 | Fast |
| Chain graph | 5 | 5 | 5 | Linear |
| Sparse graph | 10 | 10 | 6 | Sublinear phases |

#### Complexity

| Operation           | Cost           |
| ------------------- | -------------- |
| BFS (build layers)  | $O(E)$         |
| DFS (augment paths) | $O(E)$         |
| Total               | $O(E\sqrt{V})$ |

Hopcroft–Karp is the benchmark for bipartite matching, balancing elegance, efficiency, and theoretical depth.

### 373 Hungarian Algorithm

The Hungarian Algorithm (also known as the Kuhn–Munkres Algorithm) solves the assignment problem, finding the minimum-cost perfect matching in a weighted bipartite graph.
It's a cornerstone in optimization, turning complex allocation tasks into elegant linear-time computations on cost matrices.

#### What Problem Are We Solving?

Given a bipartite graph $G = (U, V, E)$ with $|U| = |V| = n$, and a cost function $c(u,v)$ for each edge,
we want to find a matching $M$ such that:

- Every $u \in U$ is matched to exactly one $v \in V$
- Total cost is minimized:

$$
\text{Minimize } \sum_{(u,v) \in M} c(u, v)
$$

This is the assignment problem, a special case of linear programming that can be solved exactly in polynomial time.

#### How It Works (Plain Language)

The Hungarian Algorithm treats the cost matrix like a grid puzzle —
you systematically reduce, label, and cover rows and columns
to reveal a set of zeros corresponding to the optimal assignment.

Core idea: Transform the cost matrix so that at least one optimal solution lies among zeros.

Steps:

1. Row Reduction
   Subtract the smallest element in each row from all elements in that row.

2. Column Reduction
   Subtract the smallest element in each column from all elements in that column.

3. Covering Zeros
   Use the minimum number of lines (horizontal + vertical) to cover all zeros.

4. Adjust Matrix
   If the number of covering lines < $n$:

   * Find smallest uncovered value $m$
   * Subtract $m$ from all uncovered elements
   * Add $m$ to elements covered twice
   * Repeat from step 3

5. Assignment
   Once $n$ lines are used, pick one zero per row/column → that's the optimal matching.

#### Example

Cost matrix:

|    | v1 | v2 | v3 |
| -- | -- | -- | -- |
| u1 | 4  | 1  | 3  |
| u2 | 2  | 0  | 5  |
| u3 | 3  | 2  | 2  |

1. Row Reduction: subtract min in each row
   | u1 | 3 | 0 | 2 |
   | u2 | 2 | 0 | 5 |
   | u3 | 1 | 0 | 0 |

2. Column Reduction: subtract min in each column
   | u1 | 2 | 0 | 2 |
   | u2 | 1 | 0 | 5 |
   | u3 | 0 | 0 | 0 |

3. Cover zeros with 3 lines → feasible.

4. Assignment: pick $(u1,v2)$, $(u2,v1)$, $(u3,v3)$ → total cost = $1+2+2=5$

Optimal assignment found.

#### Tiny Code (Python-like Pseudocode)

```python
import numpy as np
from scipy.optimize import linear_sum_assignment

def hungarian(cost_matrix):
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    total_cost = cost_matrix[row_ind, col_ind].sum()
    return list(zip(row_ind, col_ind)), total_cost

# Example
cost = np.array([[4,1,3],[2,0,5],[3,2,2]])
match, cost = hungarian(cost)
print(match, cost)  # [(0,1), (1,0), (2,2)], 5
```

#### Why It Matters

- Exact and efficient: $O(n^3)$ complexity
- Fundamental in operations research and AI (task allocation, scheduling, tracking)
- Used in:

  * Job–worker assignment
  * Optimal resource allocation
  * Matching predictions to ground truth (e.g. Hungarian loss in object detection)

The algorithm balances combinatorics and linear algebra, a rare blend of elegance and utility.

#### A Gentle Proof (Why It Works)

The algorithm maintains dual feasibility and complementary slackness at each step.
By reducing rows and columns, we ensure at least one zero in each row and column, creating a reduced cost matrix where zeros correspond to feasible assignments.

Each iteration moves closer to a perfect matching in the equality graph (edges where reduced cost = 0).
Once all vertices are matched, the solution satisfies optimality conditions.

#### Try It Yourself

1. Create a 3×3 or 4×4 cost matrix.
2. Perform row and column reductions manually.
3. Cover zeros and count lines.
4. Adjust and repeat until $n$ lines used.
5. Assign one zero per row/column.

#### Test Cases

| Matrix Size | Cost Matrix Type   | Result                  |
| ----------- | ------------------ | ----------------------- |
| 3×3         | Random integers    | Minimum cost assignment |
| 4×4         | Diagonal dominance | Diagonal chosen         |
| 5×5         | Symmetric          | Matching pairs found    |

#### Complexity

| Step                  | Time     |
| --------------------- | -------- |
| Row/column reductions | $O(n^2)$ |
| Iterative covering    | $O(n^3)$ |
| Total                 | $O(n^3)$ |

The Hungarian Algorithm transforms a cost matrix into structure, revealing an optimal matching hidden within zeros, one line at a time.

### 374 Kuhn–Munkres (Max-Weight Matching)

The Kuhn–Munkres Algorithm, also known as the Hungarian Algorithm for Maximum Weight Matching, solves the maximum-weight bipartite matching problem.
While the standard Hungarian method minimizes total cost, this version maximizes total reward or utility, making it ideal for assignment optimization when bigger is better.

#### What Problem Are We Solving?

Given a complete bipartite graph $G = (U, V, E)$ with $|U| = |V| = n$, and weights $w(u, v)$ on each edge,
find a perfect matching $M \subseteq E$ such that:

$$
\text{maximize } \sum_{(u,v) \in M} w(u,v)
$$

Each vertex is matched to exactly one partner on the opposite side, and total weight is as large as possible.

#### How It Works (Plain Language)

The algorithm treats the weight matrix like a profit grid.
It constructs a labeling on vertices and maintains equality edges (where label sums equal edge weight).
By building and augmenting matchings within this equality graph, it converges to the optimal maximum-weight matching.

Key ideas:

- Maintain vertex labels $l(u)$ and $l(v)$ that satisfy $l(u) + l(v) \ge w(u, v)$ (dual feasibility)
- Build equality graph where equality holds
- Find augmenting paths in equality graph
- Update labels when stuck to reveal new equality edges

Steps:

1. Initialize labels

   * $l(u) = \max_{v} w(u, v)$ for each $u \in U$
   * $l(v) = 0$ for each $v \in V$

2. Repeat for each $u$:

   * Build alternating tree using BFS
   * If no augmenting path, update labels to expose new equality edges
   * Augment matching along found path

3. Continue until all vertices are matched.

#### Example

Weights:

|    | v1 | v2 | v3 |
| -- | -- | -- | -- |
| u1 | 3  | 2  | 1  |
| u2 | 2  | 4  | 6  |
| u3 | 3  | 5  | 3  |

Goal: maximize total weight

Optimal matching:

- $u1 \to v1$ (3)
- $u2 \to v3$ (6)
- $u3 \to v2$ (5)

Total weight = $3 + 6 + 5 = 14$

#### Tiny Code (Python-like Pseudocode)

```python
import numpy as np
from scipy.optimize import linear_sum_assignment

def kuhn_munkres(weight_matrix):
    # Convert to cost by negation (Hungarian solves minimization)
    cost = -weight_matrix
    row_ind, col_ind = linear_sum_assignment(cost)
    total = weight_matrix[row_ind, col_ind].sum()
    return list(zip(row_ind, col_ind)), total

# Example
W = np.array([[3,2,1],[2,4,6],[3,5,3]])
match, total = kuhn_munkres(W)
print(match, total)  # [(0,0),(1,2),(2,1)], 14
```

#### Why It Matters

- Solves maximum reward assignments in polynomial time
- Fundamental in:

  * Job–task allocation
  * Optimal pairing problems
  * Machine learning (e.g. Hungarian loss)
  * Game theory and economics

Many systems use this algorithm to maximize overall efficiency when assigning limited resources.

#### A Gentle Proof (Why It Works)

- Maintain dual feasibility: $l(u) + l(v) \ge w(u, v)$
- Maintain complementary slackness: matched edges satisfy equality
- By alternating updates between equality graph expansion and label adjustments, the algorithm ensures eventual feasibility and optimality
- The final matching satisfies strong duality, achieving maximum weight

#### Try It Yourself

1. Build a $3 \times 3$ profit matrix.
2. Initialize $l(u)$ as row maxima, $l(v)=0$.
3. Draw equality edges ($l(u)+l(v)=w(u,v)$).
4. Find augmenting path → augment → update labels.
5. Repeat until matching is perfect.

#### Test Cases

| Graph | Size           | Max Weight   | Matching            |
| ----- | -------------- | ------------ | ------------------- |
| 3×3   | Random weights | 14           | [(0,0),(1,2),(2,1)] |
| 4×4   | Diagonal high  | sum(diag)    | Diagonal            |
| 5×5   | Uniform        | 5×max weight | any                 |

#### Complexity

| Step            | Time     |
| --------------- | -------- |
| Label updates   | $O(V^2)$ |
| Matching phases | $O(V)$   |
| Total           | $O(V^3)$ |

The Kuhn–Munkres algorithm is the ultimate tool for maximum-weight assignments, blending geometry, duality, and combinatorics into one powerful optimization engine.

### 375 Blossom Algorithm

The Blossom Algorithm, developed by Jack Edmonds, is the foundational algorithm for finding maximum matchings in general graphs, including non-bipartite ones.
It introduced the concept of blossoms (odd-length cycles) and showed that maximum matching is solvable in polynomial time, a landmark in combinatorial optimization.

#### What Problem Are We Solving?

Given a general (not necessarily bipartite) graph $G = (V, E)$, find a maximum matching, a largest set of edges such that no two share a vertex.

Unlike bipartite graphs, odd-length cycles can block progress in standard augmenting path searches.
The Blossom Algorithm contracts these cycles so that augmenting paths can still be found.

Formally, find $M \subseteq E$ maximizing $|M|$ such that each vertex is incident to at most one edge in $M$.

#### How It Works (Plain Language)

The algorithm extends the augmenting path approach (used in bipartite matching) by adding one powerful idea:
When you hit an odd-length cycle, treat it as a single vertex, a blossom.

Steps:

1. Initialize with an empty matching $M = \emptyset$.
2. Search for augmenting paths using BFS/DFS in the alternating tree:

   * Alternate between matched and unmatched edges.
   * If you reach a free vertex, augment (flip matched/unmatched edges).
3. If you encounter an odd cycle (a blossom), contract it into a single super-vertex.

   * Continue the search in the contracted graph.
   * Once an augmenting path is found, expand blossoms and update the matching.
4. Repeat until no augmenting path exists.

Each augmentation increases $|M|$ by 1.
When no augmenting path remains, the matching is maximum.

#### Example

Graph:

- Vertices: ${A, B, C, D, E}$
- Edges: $(A, B), (B, C), (C, A), (B, D), (C, E)$

1. Start matching: empty
2. Build alternating tree: $A \to B \to C \to A$ forms an odd cycle
3. Contract blossom $(A, B, C)$ into one node
4. Continue search → find augmenting path through blossom
5. Expand blossom, adjust matching
6. Result: maximum matching includes edges $(A,B), (C,E), (D,...)$

#### Tiny Code (Python-Like Pseudocode)

```python
# Using networkx for simplicity
import networkx as nx

def blossom_maximum_matching(G):
    return nx.max_weight_matching(G, maxcardinality=True)

# Example
G = nx.Graph()
G.add_edges_from([(0,1),(1,2),(2,0),(1,3),(2,4)])
match = blossom_maximum_matching(G)
print(match)  # {(0,1), (2,4)}
```

#### Why It Matters

- General graphs (non-bipartite) are common in real-world systems:

  * Social networks (friend pairings)
  * Molecular structure matching
  * Scheduling with constraints
  * Graph-theoretic proofs and optimization

The Blossom Algorithm proved that matching is polynomial-time solvable, a key milestone in algorithmic theory.

#### A Gentle Proof (Why It Works)

By Berge's Lemma, a matching is maximum iff there's no augmenting path.
The challenge is that augmenting paths can be hidden inside odd cycles.

Blossom contraction ensures that every augmenting path in the contracted graph corresponds to one in the original.
After each augmentation, $|M|$ strictly increases, so the algorithm terminates in polynomial time.

Correctness follows from maintaining:

- Alternating trees with consistent parity
- Contraction invariants (augmenting path preservation)
- Berge's condition across contractions and expansions

#### Try It Yourself

1. Draw a triangle graph (3-cycle).
2. Run augmenting path search, find blossom.
3. Contract it into one node.
4. Continue search and expand after finding path.
5. Verify maximum matching.

#### Test Cases

| Graph Type               | Matching Size | Method                |
| ------------------------ | ------------- | --------------------- |
| Triangle                 | 1             | Blossom contraction   |
| Square with diagonal     | 2             | Augmentation          |
| Pentagonal odd cycle     | 2             | Blossom               |
| Bipartite (sanity check) | As usual      | Matches Hopcroft–Karp |

#### Complexity

| Phase                     | Time     |
| ------------------------- | -------- |
| Search (per augmentation) | $O(VE)$  |
| Augmentations             | $O(V)$   |
| Total                     | $O(V^3)$ |

The Blossom Algorithm was a revelation, showing how to tame odd cycles and extending matchings to all graphs, bridging combinatorics and optimization theory.

### 376 Edmonds' Blossom Shrinking

Edmonds' Blossom Shrinking is the core subroutine that powers the Blossom Algorithm, enabling augmenting-path search in non-bipartite graphs.
It provides the crucial mechanism for contracting odd-length cycles (blossoms) so that hidden augmenting paths can be revealed and exploited.

#### What Problem Are We Solving?

In non-bipartite graphs, augmenting paths can be obscured by odd-length cycles.
Standard matching algorithms fail because they assume bipartite structure.

We need a way to handle odd cycles during search so that the algorithm can progress without missing valid augmentations.

Goal:
Detect blossoms, shrink them into single vertices, and continue the search efficiently.

Given a matching $M$ in graph $G = (V, E)$, and an alternating tree grown during a search, when a blossom is found:

- Shrink the blossom into a super-vertex
- Continue the search in the contracted graph
- Expand the blossom when an augmenting path is found

#### How It Works (Plain Language)

Imagine running a BFS/DFS from a free vertex.
You alternate between matched and unmatched edges to build an alternating tree.

If you find an edge connecting two vertices at the same level (both even depth):

- You've detected an odd cycle.
- That cycle is a blossom.

To handle it:

1. Identify the blossom (odd-length alternating cycle)
2. Shrink all its vertices into a single super-node
3. Continue the augmenting path search on this contracted graph
4. Once an augmenting path is discovered, expand the blossom and adjust the path accordingly
5. Augment along the expanded path

This shrinking maintains all valid augmenting paths and allows the algorithm to operate as if the blossom were one vertex.

#### Example

Graph:
Vertices: ${A, B, C, D, E}$
Edges: $(A,B), (B,C), (C,A), (B,D), (C,E)$

Suppose $A, B, C$ form an odd cycle discovered during search.

1. Detect blossom: $(A, B, C)$
2. Contract into a single vertex $X$
3. Continue search on reduced graph
4. If augmenting path passes through $X$, expand back
5. Alternate edges within blossom to integrate path correctly

Result: A valid augmenting path is found and matching increases by one.

#### Tiny Code (Python-Like)

```python
def find_blossom(u, v, parent):
    # Find common ancestor
    path_u, path_v = set(), set()
    while u != -1:
        path_u.add(u)
        u = parent[u]
    while v not in path_u:
        path_v.add(v)
        v = parent[v]
    lca = v
    # Shrink blossom (conceptually)
    return lca
```

In practical implementations (like Edmonds' algorithm), this logic merges all nodes in the blossom into a single node and adjusts parent/child relationships.

#### Why It Matters

This is the heart of general-graph matching.
Without shrinking, the search may loop infinitely or fail to detect valid augmentations.

Blossom shrinking allows:

- Handling odd-length cycles
- Maintaining augmenting path invariants
- Guaranteeing polynomial time behavior

It is also one of the earliest uses of graph contraction in combinatorial optimization.

#### A Gentle Proof (Why It Works)

By Berge's Lemma, a matching is maximum iff there is no augmenting path.
If an augmenting path exists in $G$, then one exists in any contracted version of $G$.

Shrinking a blossom preserves augmentability:

- Every augmenting path in the original graph corresponds to one in the contracted graph.
- After expansion, the alternating pattern is restored correctly.

Therefore, contraction does not lose information, it simply simplifies the search.

#### Try It Yourself

1. Draw a triangle $A, B, C$ connected to other vertices $D, E$.
2. Build an alternating tree starting from a free vertex.
3. When you find an edge connecting two even-level vertices, mark the odd cycle.
4. Shrink the blossom, continue search, then expand once path is found.

Observe how this allows discovery of an augmenting path that was previously hidden.

#### Test Cases

| Graph           | Description         | Augmentable After Shrink? |
| --------------- | ------------------- | ------------------------- |
| Triangle        | Single odd cycle    | Yes                       |
| Triangle + tail | Cycle + path        | Yes                       |
| Bipartite       | No odd cycle        | Shrink not needed         |
| Pentagon        | Blossom of length 5 | Yes                       |

#### Complexity

Shrinking can be done in linear time relative to blossom size.
Total algorithm remains $O(V^3)$ when integrated into full matching search.

Edmonds' Blossom Shrinking is the conceptual leap that made maximum matching in general graphs tractable, transforming an intractable maze of cycles into a solvable structure through careful contraction and expansion.

### 377 Greedy Matching

Greedy Matching is the simplest way to approximate a maximum matching in a graph.
Instead of exploring augmenting paths, it just keeps picking available edges that don't conflict, a fast, intuitive baseline for matching problems.

#### What Problem Are We Solving?

In many real-world settings, job assignment, pairing users, scheduling, we need a set of edges such that no two share a vertex.
This is a matching.

Finding the maximum matching exactly (especially in general graphs) can be expensive.
But sometimes, we only need a good enough answer quickly.

A Greedy Matching algorithm provides a fast approximation:

- It won't always find the largest matching
- But it runs in $O(E)$ and often gives a decent solution

Goal: Quickly build a maximal matching (no edge can be added).

#### How It Works (Plain Language)

Start with an empty set $M$ (the matching).
Go through the edges one by one:

1. For each edge $(u, v)$
2. If neither $u$ nor $v$ is already matched
3. Add $(u, v)$ to $M$

Continue until all edges have been checked.

This ensures no vertex appears in more than one edge, a valid matching.
The result is maximal: you can't add any other edge without breaking the matching rule.

#### Example

Graph edges:
$$
E = {(A,B), (B,C), (C,D), (D,E)}
$$

- Start: $M = \emptyset$
- Pick $(A,B)$ → mark $A$, $B$ as matched
- Skip $(B,C)$ → $B$ is matched
- Pick $(C,D)$ → mark $C$, $D$
- Skip $(D,E)$ → $D$ is matched

Result:
$$
M = {(A,B), (C,D)}
$$

#### Tiny Code (Python-Like)

```python
def greedy_matching(graph):
    matched = set()
    matching = []
    for u, v in graph.edges:
        if u not in matched and v not in matched:
            matching.append((u, v))
            matched.add(u)
            matched.add(v)
    return matching
```

This simple loop runs in linear time over edges.

#### Why It Matters

- Fast: runs in $O(E)$
- Simple: easy to implement
- Useful baseline: good starting point for heuristic or hybrid approaches
- Guaranteed maximality: no edge can be added without breaking matching condition

Although it's not optimal, its result is at least half the size of a maximum matching:
$$
|M_{\text{greedy}}| \ge \frac{1}{2}|M_{\text{max}}|
$$

#### A Gentle Proof (Why It Works)

Each greedy edge $(u,v)$ blocks at most one edge from $M_{\text{max}}$ (since both $u$ and $v$ are used).
So for every chosen edge, we lose at most one from the optimal.
Thus:
$$
2|M_{\text{greedy}}| \ge |M_{\text{max}}|
$$
⇒ Greedy achieves a 1/2-approximation.

#### Try It Yourself

1. Create a graph with 6 vertices and 7 edges.
2. Run greedy matching in different edge orders.
3. Observe how results differ, order can affect the final set.
4. Compare with an exact maximum matching (e.g., Hopcroft–Karp).

You'll see how simple decisions early can influence outcome size.

#### Test Cases

| Graph           | Edges                     | Greedy Matching | Maximum Matching |
| --------------- | ------------------------- | --------------- | ---------------- |
| Path of 4       | $(A,B),(B,C),(C,D)$       | 2               | 2                |
| Triangle        | $(A,B),(B,C),(C,A)$       | 1               | 1                |
| Square          | $(A,B),(B,C),(C,D),(D,A)$ | 2               | 2                |
| Star (5 leaves) | $(C,1)...(C,5)$           | 1               | 1                |

#### Complexity

- Time: $O(E)$
- Space: $O(V)$

Works for any undirected graph.
For directed graphs, edges must be treated as undirected or symmetric.

Greedy matching is a quick and practical approach when you need speed over perfection, a simple handshake strategy that pairs as many as possible before the clock runs out.

### 378 Stable Marriage (Gale–Shapley)

Stable Marriage (or the Stable Matching Problem) is a cornerstone of combinatorial optimization, where two equal-sized sets (e.g. men and women, jobs and applicants, hospitals and residents) must be paired so that no two participants prefer each other over their assigned partners.
The Gale–Shapley algorithm finds such a stable matching in $O(n^2)$ time.

#### What Problem Are We Solving?

Given two sets:

- Set $A = {a_1, a_2, ..., a_n}$
- Set $B = {b_1, b_2, ..., b_n}$

Each member ranks all members of the other set in order of preference.
We want to find a matching (a one-to-one pairing) such that:

There is no pair $(a_i, b_j)$ where:

- $a_i$ prefers $b_j$ over their current match, and
- $b_j$ prefers $a_i$ over their current match.

Such a pair would be unstable, since they would rather be matched together.

Our goal: find a stable configuration, no incentive to switch partners.

#### How It Works (Plain Language)

The Gale–Shapley algorithm uses proposals and rejections:

1. All $a_i$ start unmatched.
2. While some $a_i$ is free:

   * $a_i$ proposes to the most-preferred $b_j$ not yet proposed to.
   * If $b_j$ is free, accept the proposal.
   * If $b_j$ is matched but prefers $a_i$ over current partner, she "trades up" and rejects the old one.
   * Otherwise, she rejects $a_i$.
3. Continue until everyone is matched.

The process always terminates with a stable matching.

#### Example

Let's say we have:

| A  | Preference List |
| -- | --------------- |
| A1 | B1, B2, B3      |
| A2 | B2, B1, B3      |
| A3 | B3, B1, B2      |

| B  | Preference List |
| -- | --------------- |
| B1 | A2, A1, A3      |
| B2 | A1, A2, A3      |
| B3 | A1, A2, A3      |

Step by step:

- A1 → B1, B1 free → match (A1, B1)
- A2 → B2, B2 free → match (A2, B2)
- A3 → B3, B3 free → match (A3, B3)

Stable matching: {(A1, B1), (A2, B2), (A3, B3)}

No two prefer each other over current partners → Stable.

#### Tiny Code (Python-Like)

```python
def gale_shapley(A_prefs, B_prefs):
    free_A = list(A_prefs.keys())
    engaged = {}
    next_choice = {a: 0 for a in A_prefs}

    while free_A:
        a = free_A.pop(0)
        b = A_prefs[a][next_choice[a]]
        next_choice[a] += 1

        if b not in engaged:
            engaged[b] = a
        else:
            current = engaged[b]
            if B_prefs[b].index(a) < B_prefs[b].index(current):
                engaged[b] = a
                free_A.append(current)
            else:
                free_A.append(a)
    return {v: k for k, v in engaged.items()}
```

#### Why It Matters

- Guarantees a stable solution.
- Always terminates after at most $n^2$ proposals.
- If one side proposes (say $A$), the result is optimal for $A$ and pessimal for $B$.
- Forms the foundation of real-world systems like the National Residency Matching Program (NRMP) and school assignments.

#### A Gentle Proof (Why It Works)

Each proposal is made once per pair → at most $n^2$ proposals.

No cycles:

- $b$ always stays with the best proposer so far
- Once rejected, $a$ cannot improve, ensuring termination

Stability:
Suppose $(a_i, b_j)$ is a blocking pair. Then $a_i$ must have proposed to $b_j$ and was rejected.
That means $b_j$ preferred her current match. So $(a_i, b_j)$ cannot be blocking → contradiction.

Thus, the final matching is stable.

#### Try It Yourself

1. Create two sets of 3 elements with ranked lists.
2. Run Gale–Shapley twice, once with A proposing, once with B proposing.
3. Compare the two matchings, see who benefits.
4. Explore what happens if preference lists are incomplete.

#### Test Cases

| Case                | Result                    | Stable? |
| ------------------- | ------------------------- | ------- |
| Equal preferences   | Multiple stable matchings | Yes     |
| Cyclic preferences  | Algorithm converges       | Yes     |
| One-sided identical | Deterministic output      | Yes     |

#### Complexity

- Time: $O(n^2)$
- Space: $O(n)$

Scalable for small to medium systems, widely used in practice.

Stable marriage shows how simple local rules (propose, reject, keep best) can produce global stability, a harmony of preference and fairness.

### 379 Weighted b-Matching

Weighted b-Matching generalizes standard matching by allowing each node to be matched to up to $b(v)$ partners instead of just one.
When edges carry weights, the goal is to find a maximum-weight subset of edges such that the degree of each vertex does not exceed its capacity $b(v)$.

#### What Problem Are We Solving?

Given a graph $G = (V, E)$ with:

- Edge weights $w(e)$ for each $e \in E$
- Capacity constraint $b(v)$ for each vertex $v \in V$

Find a subset $M \subseteq E$ that:

- Maximizes total weight
  $$W(M) = \sum_{e \in M} w(e)$$
- Satisfies degree constraints
  $$\forall v \in V,\quad \deg_M(v) \le b(v)$$

If $b(v) = 1$ for all $v$, this becomes the standard maximum-weight matching.

#### How It Works (Plain Language)

Think of each vertex $v$ as having b(v) slots available for connections.
We want to pick edges to fill these slots to maximize total edge weight, without exceeding any vertex's limit.

The weighted b-matching problem can be solved by:

1. Reduction to flow: Convert matching to a network flow problem:

   * Each edge becomes a flow connection with capacity 1 and cost = $-w(e)$
   * Vertex capacities become flow limits
   * Solve via min-cost max-flow
2. Linear Programming: Relax constraints and solve with LP or primal-dual algorithms
3. Approximation: For large sparse graphs, greedy heuristics can achieve near-optimal solutions

#### Example

Suppose:

- Vertices: $V = {A, B, C}$
- Edges and weights:
  $w(A,B)=5,; w(A,C)=4,; w(B,C)=3$
- Capacities: $b(A)=2,; b(B)=1,; b(C)=1$

We can pick:

- $(A,B)$ weight 5
- $(A,C)$ weight 4

Total weight = $9$
Valid since $\deg(A)=2,\deg(B)=1,\deg(C)=1$

If $b(A)=1$, we'd pick only $(A,B)$ → weight $5$

#### Tiny Code (Python-Like Pseudocode)

```python
import networkx as nx

def weighted_b_matching(G, b):
    # Convert to flow network
    flow_net = nx.DiGraph()
    source, sink = 's', 't'

    for v in G.nodes:
        flow_net.add_edge(source, v, capacity=b[v], weight=0)
        flow_net.add_edge(v, sink, capacity=b[v], weight=0)

    for (u, v, data) in G.edges(data=True):
        w = -data['weight']  # negate for min-cost
        flow_net.add_edge(u, v, capacity=1, weight=w)
        flow_net.add_edge(v, u, capacity=1, weight=w)

    flow_dict = nx.max_flow_min_cost(flow_net, source, sink)
    # Extract edges with flow=1 between vertex pairs
    return [(u, v) for u, nbrs in flow_dict.items() for v, f in nbrs.items() if f > 0 and u in G.nodes and v in G.nodes]
```

#### Why It Matters

- Models resource allocation with limits (e.g. each worker can handle $b$ tasks)
- Extends classic matchings to capacitated networks
- Foundation for:

  * Task assignment with quotas
  * Scheduling with multi-capacity
  * Clustering with degree constraints

#### A Gentle Proof (Why It Works)

Each vertex $v$ has degree constraint $b(v)$:
$$\sum_{e \ni v} x_e \le b(v)$$

Each edge is chosen at most once:
$$x_e \in {0, 1}$$

The optimization:
$$\max \sum_{e \in E} w(e) x_e$$

subject to constraints above, forms an integer linear program.
Relaxation via network flow ensures optimal integral solution for bipartite graphs.

#### Try It Yourself

1. Build a triangle graph with different edge weights.
2. Set $b(v)=2$ for one vertex, 1 for others.
3. Compute by hand which edges yield maximum weight.
4. Implement with a flow solver and verify.

#### Test Cases

| Graph            | Capacities | Result            | Weight            |
| ---------------- | ---------- | ----------------- | ----------------- |
| Triangle (5,4,3) | {2,1,1}    | {(A,B),(A,C)}     | 9                 |
| Square           | {1,1,1,1}  | Standard matching | Max weight sum    |
| Line (A-B-C)     | {2,1,2}    | {(A,B),(B,C)}     | Sum of both edges |

#### Complexity

- Time: $O(V^3)$ using min-cost max-flow
- Space: $O(V + E)$

Efficient for medium graphs; scalable via heuristics for larger ones.

Weighted b-matching balances optimality and flexibility, allowing richer allocation models than one-to-one pairings.

### 380 Maximal Matching

Maximal Matching is a greedy approach that builds a matching by adding edges one by one until no more can be added without breaking the matching condition.
Unlike *maximum matchings* (which maximize size or weight), a *maximal* matching simply ensures no edge can be added, it's locally optimal, not necessarily globally optimal.

#### What Problem Are We Solving?

We want to select a set of edges $M \subseteq E$ such that:

1. No two edges in $M$ share a vertex
   $$\forall (u,v), (x,y) \in M,; {u,v} \cap {x,y} = \emptyset$$
2. $M$ is maximal: no additional edge from $E$ can be added without violating (1).

Goal: Find any maximal matching, not necessarily the largest one.

#### How It Works (Plain Language)

The algorithm is greedy and simple:

1. Start with an empty matching $M = \emptyset$
2. Iterate through edges $(u,v)$
3. If neither $u$ nor $v$ are matched yet, add $(u,v)$ to $M$
4. Continue until all edges are processed

At the end, $M$ is maximal: every unmatched edge touches a vertex already matched.

#### Example

Graph:

```
A -- B -- C
|         
D
```

Edges: $(A,B)$, $(B,C)$, $(A,D)$

Step-by-step:

- Pick $(A,B)$ → mark A, B matched
- Skip $(B,C)$ (B already matched)
- Skip $(A,D)$ (A already matched)

Result: $M = {(A,B)}$

Another valid maximal matching: ${(A,D), (B,C)}$
Not unique, but all maximal sets share the property that no more edges can be added.

#### Tiny Code (Python)

```python
def maximal_matching(edges):
    matched = set()
    M = []
    for u, v in edges:
        if u not in matched and v not in matched:
            M.append((u, v))
            matched.add(u)
            matched.add(v)
    return M

edges = [('A','B'), ('B','C'), ('A','D')]
print(maximal_matching(edges))  # [('A','B')]
```

#### Why It Matters

- Fast approximation: Maximal matching is a 2-approximation for the maximum matching (it's at least half as large).
- Building block: Used in distributed and parallel algorithms where global optimization is too expensive.
- Preprocessing step: Reduces problem size in flow, scheduling, and network design.

#### A Gentle Proof (Why It Works)

If an edge $(u,v)$ is not in $M$, at least one of its endpoints is already matched.
Thus, adding $(u,v)$ would break the matching condition.
This ensures maximality, no more edges can be safely added.

Formally, for every $(u,v) \notin M$:
$$u \in V_M \lor v \in V_M$$
where $V_M$ is the set of matched vertices.

Therefore, $M$ is maximal.

#### Try It Yourself

1. Draw a small graph with 4–6 vertices.
2. Apply the greedy process in different edge orders.
3. Compare resulting matchings, note they may differ in size, but all are maximal.
4. Try to add an extra edge, see why it violates matching rules.

#### Test Cases

| Graph          | Edges                     | Maximal Matching | Size |
| -------------- | ------------------------- | ---------------- | ---- |
| Path A–B–C     | {(A,B),(B,C)}             | {(A,B)}          | 1    |
| Star A–B,C,D   | {(A,B),(A,C),(A,D)}       | {(A,B)}          | 1    |
| Square A–B–C–D | {(A,B),(B,C),(C,D),(D,A)} | {(A,B),(C,D)}    | 2    |

#### Complexity

- Time: $O(E)$, single pass through edges
- Space: $O(V)$ for vertex match tracking

Fast, simple, and effective for large graphs where optimal matchings are too expensive.

Maximal matching gives you a quick and safe pairing, not always perfect, but always *done*.

## Section 39. Tree Algorithms 

### 381 Euler Tour LCA

Euler Tour LCA transforms the problem of finding the Lowest Common Ancestor (LCA) into a Range Minimum Query (RMQ) problem by recording an Euler tour of the tree.
It's elegant, precomputable, and fast, perfect for multiple LCA queries.

#### What Problem Are We Solving?

Given a rooted tree and two nodes $u$ and $v$, we want to find their lowest common ancestor, the deepest node that is an ancestor of both.

For example, in a tree rooted at $1$:

```
      1
    /   \
   2     3
  / \   /
 4  5  6
```

The LCA of $(4,5)$ is $2$, and LCA of $(4,6)$ is $1$.

We want to answer many such queries efficiently.

#### How It Works (Plain Language)

1. Perform a DFS traversal (Euler tour):

   * Record each node when you enter it.
   * When you backtrack, record it again.

2. Store the depth of each node when visited.

3. For each node, remember its first occurrence in the Euler tour.

4. For query $(u, v)$:

   * Find the first occurrences of $u$ and $v$ in the tour.
   * Take the range between them.
   * The node with minimum depth in that range is the LCA.

So, LCA reduces to an RMQ over depths.

#### Example

Tree (root = 1):

```
1
├── 2
│   ├── 4
│   └── 5
└── 3
    └── 6
```

Euler tour:
$[1,2,4,2,5,2,1,3,6,3,1]$

Depth array:
$[0,1,2,1,2,1,0,1,2,1,0]$

First occurrences:
$1\to0,;2\to1,;3\to7,;4\to2,;5\to4,;6\to8$

Query: LCA(4,5)

- First(4) = 2, First(5) = 4
- Depth range [2..4] = [2,1,2]
- Minimum depth = 1 → node = 2

So LCA(4,5) = 2.

#### Tiny Code (Python)

```python
def euler_tour_lca(graph, root):
    n = len(graph)
    euler, depth, first = [], [], {}
    
    def dfs(u, d):
        first.setdefault(u, len(euler))
        euler.append(u)
        depth.append(d)
        for v in graph[u]:
            dfs(v, d + 1)
            euler.append(u)
            depth.append(d)
    
    dfs(root, 0)
    return euler, depth, first

def build_rmq(depth):
    n = len(depth)
    log = [0] * (n + 1)
    for i in range(2, n + 1):
        log[i] = log[i // 2] + 1
    k = log[n]
    st = [[0] * (k + 1) for _ in range(n)]
    for i in range(n):
        st[i][0] = i
    j = 1
    while (1 << j) <= n:
        i = 0
        while i + (1 << j) <= n:
            left = st[i][j - 1]
            right = st[i + (1 << (j - 1))][j - 1]
            st[i][j] = left if depth[left] < depth[right] else right
            i += 1
        j += 1
    return st, log

def query_lca(u, v, euler, depth, first, st, log):
    l, r = first[u], first[v]
    if l > r:
        l, r = r, l
    j = log[r - l + 1]
    left = st[l][j]
    right = st[r - (1 << j) + 1][j]
    return euler[left] if depth[left] < depth[right] else euler[right]
```

#### Why It Matters

- Reduces LCA to RMQ, allowing $O(1)$ query with $O(n \log n)$ preprocessing
- Easy to combine with Segment Trees, Sparse Tables, or Cartesian Trees
- Essential for:

  * Tree DP with ancestor relationships
  * Distance queries:
    $$\text{dist}(u,v) = \text{depth}(u) + \text{depth}(v) - 2 \times \text{depth}(\text{LCA}(u,v))$$
  * Path queries in heavy-light decomposition

#### A Gentle Proof (Why It Works)

In DFS traversal:

- Every ancestor appears before and after descendants.
- The first common ancestor to reappear between two nodes' first occurrences is their LCA.

Therefore, the node with minimum depth between the first appearances corresponds exactly to the lowest common ancestor.

#### Try It Yourself

1. Construct a tree and perform an Euler tour manually.
2. Write down depth and first occurrence arrays.
3. Pick pairs $(u,v)$, find the minimum depth in their range.
4. Confirm LCA correctness.

#### Test Cases

| Query    | Expected | Reason                  |
| -------- | -------- | ----------------------- |
| LCA(4,5) | 2        | Both under 2            |
| LCA(4,6) | 1        | Root is common ancestor |
| LCA(2,3) | 1        | Different subtrees      |
| LCA(6,3) | 3        | One is ancestor         |

#### Complexity

- Preprocessing: $O(n \log n)$
- Query: $O(1)$
- Space: $O(n \log n)$

Euler Tour LCA shows how tree problems become array problems, a perfect example of structural transformation in algorithms.

### 382 Binary Lifting LCA

Binary Lifting is a fast and elegant technique to find the Lowest Common Ancestor (LCA) of two nodes in a tree using precomputed jumps to ancestors at powers of two.
It turns ancestor queries into simple bit operations, giving $O(\log n)$ per query.

#### What Problem Are We Solving?

Given a rooted tree and two nodes $u$ and $v$, we want to find their lowest common ancestor (LCA), the deepest node that is an ancestor of both.

Naively, we could walk up one node at a time, but that's $O(n)$ per query.
With binary lifting, we jump exponentially up the tree, reducing the cost to $O(\log n)$.

#### How It Works (Plain Language)

Binary lifting precomputes for each node:

$$\text{up}[v][k] = \text{the } 2^k \text{-th ancestor of } v$$

So $\text{up}[v][0]$ is the parent, $\text{up}[v][1]$ is the grandparent, $\text{up}[v][2]$ is the ancestor 4 levels up, and so on.

The algorithm proceeds in three steps:

1. Preprocess:

   * Run DFS from the root.
   * Record each node's depth.
   * Fill table `up[v][k]`.

2. Equalize Depths:

   * If one node is deeper, lift it up to match the shallower one.

3. Lift Together:

   * Jump both nodes up together (largest powers first) until their ancestors match.

The meeting point is the LCA.

#### Example

Tree:

```
      1
    /   \
   2     3
  / \   /
 4  5  6
```

Binary lifting table (`up[v][k]`):

| v | up[v][0] | up[v][1] | up[v][2] |
| - | -------- | -------- | -------- |
| 1 | -        | -        | -        |
| 2 | 1        | -        | -        |
| 3 | 1        | -        | -        |
| 4 | 2        | 1        | -        |
| 5 | 2        | 1        | -        |
| 6 | 3        | 1        | -        |

Query: LCA(4,5)

- Depth(4)=2, Depth(5)=2
- Lift 4,5 together → parents (2,2) match → LCA = 2

Query: LCA(4,6)

- Depth(4)=2, Depth(6)=2
- Lift 4→2, 6→3 → not equal
- Lift 2→1, 3→1 → LCA = 1

#### Tiny Code (Python)

```python
LOG = 20  # enough for n up to ~1e6

def preprocess(graph, root):
    n = len(graph)
    up = [[-1] * LOG for _ in range(n)]
    depth = [0] * n

    def dfs(u, p):
        up[u][0] = p
        for k in range(1, LOG):
            if up[u][k-1] != -1:
                up[u][k] = up[up[u][k-1]][k-1]
        for v in graph[u]:
            if v != p:
                depth[v] = depth[u] + 1
                dfs(v, u)
    dfs(root, -1)
    return up, depth

def lca(u, v, up, depth):
    if depth[u] < depth[v]:
        u, v = v, u
    diff = depth[u] - depth[v]
    for k in range(LOG):
        if diff & (1 << k):
            u = up[u][k]
    if u == v:
        return u
    for k in reversed(range(LOG)):
        if up[u][k] != up[v][k]:
            u = up[u][k]
            v = up[v][k]
    return up[u][0]
```

#### Why It Matters

- Handles large trees with many queries efficiently
- Enables ancestor jumps, k-th ancestor queries, distance queries, etc.
- Fundamental in competitive programming, tree DP, and path queries

#### A Gentle Proof (Why It Works)

If $u$ and $v$ have different depths, lifting the deeper node by $2^k$ steps ensures we equalize their depth quickly.

Once aligned, lifting both simultaneously ensures we never skip the LCA, since jumps are made only when ancestors differ.

Eventually, both meet at their lowest shared ancestor.

#### Try It Yourself

1. Build a small tree with 7 nodes.
2. Compute $\text{up}[v][k]$ manually.
3. Pick pairs and simulate lifting step-by-step.
4. Verify with naive path-to-root comparison.

#### Test Cases

| Query    | Expected | Reason          |
| -------- | -------- | --------------- |
| LCA(4,5) | 2        | same parent     |
| LCA(4,6) | 1        | across subtrees |
| LCA(2,3) | 1        | root ancestor   |
| LCA(6,3) | 3        | ancestor        |

#### Complexity

- Preprocessing: $O(n \log n)$
- Query: $O(\log n)$
- Space: $O(n \log n)$

Binary lifting turns tree ancestry into bitwise jumping, offering a clear path from root to ancestor in logarithmic time.

### 383 Tarjan's LCA (Offline DSU)

Tarjan's LCA algorithm answers multiple LCA queries offline using Disjoint Set Union (DSU) with path compression.
It traverses the tree once (DFS) and merges subtrees, answering all queries in $O(n + q \alpha(n))$ time.

#### What Problem Are We Solving?

Given a rooted tree and multiple queries $(u, v)$, we want the lowest common ancestor of each pair.

Unlike binary lifting (which works online), Tarjan's algorithm works offline:

- We know all queries in advance.
- We answer them in one DFS traversal using a union-find structure.

#### How It Works (Plain Language)

We process the tree bottom-up:

1. Start a DFS from the root.
2. Each node begins as its own set in DSU.
3. After visiting a child, union it with its parent.
4. When a node is fully processed, mark it as visited.
5. For each query $(u,v)$:

   * If the other node has been visited,
     then $\text{find}(v)$ (or $\text{find}(u)$) gives the LCA.

This works because the DSU structure merges all processed ancestors,
so $\text{find}(v)$ always points to the lowest processed ancestor common to both.

#### Example

Tree (root = 1):

```
1
├── 2
│   ├── 4
│   └── 5
└── 3
    └── 6
```

Queries:
$(4,5), (4,6), (3,6)$

Process:

- DFS(1): visit 2, visit 4

  * Query(4,5): 5 not visited → skip
  * Backtrack → union(4,2)
- DFS(5): mark visited, union(5,2)

  * Query(4,5): 4 visited → $\text{find}(4)=2$ → LCA(4,5)=2
- DFS(3), visit 6 → union(6,3)

  * Query(4,6): 4 visited, $\text{find}(4)=2$, $\text{find}(6)=3$ → no LCA yet
  * Query(3,6): both visited → $\text{find}(3)=3$ → LCA(3,6)=3
- Backtrack 2→1, 3→1 → LCA(4,6)=1

#### Tiny Code (Python)

```python
from collections import defaultdict

class DSU:
    def __init__(self, n):
        self.parent = list(range(n))
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    def union(self, a, b):
        self.parent[self.find(a)] = self.find(b)

def tarjan_lca(tree, root, queries):
    n = len(tree)
    dsu = DSU(n)
    visited = [False] * n
    ancestor = [0] * n
    ans = {}
    qmap = defaultdict(list)
    for u, v in queries:
        qmap[u].append(v)
        qmap[v].append(u)
    def dfs(u):
        ancestor[u] = u
        for v in tree[u]:
            dfs(v)
            dsu.union(v, u)
            ancestor[dsu.find(u)] = u
        visited[u] = True
        for v in qmap[u]:
            if visited[v]:
                ans[(u, v)] = ancestor[dsu.find(v)]
    dfs(root)
    return ans
```

#### Why It Matters

- Handles many LCA queries efficiently in one traversal
- No need for depth or binary lifting table
- Ideal for offline batch queries

Applications:

- Tree queries in competitive programming
- Offline graph analysis
- Dynamic connectivity in rooted structures

#### A Gentle Proof (Why It Works)

Once a node $u$ is processed:

- All its descendants are merged into its DSU set.
- $\text{find}(u)$ always returns the lowest ancestor processed so far.

When visiting query $(u,v)$:

- If $v$ is already visited,
  then $\text{find}(v)$ is the lowest common ancestor of $u$ and $v$.

The invariant:
$$
\text{ancestor}[\text{find}(v)] = \text{LCA}(u, v)
$$
is maintained by post-order traversal and union operations.

#### Try It Yourself

1. Draw a small tree and list all queries.
2. Perform DFS manually.
3. Track unions and ancestor updates.
4. Record LCA when both nodes visited.

#### Test Cases

| Query | Expected | Reason          |
| ----- | -------- | --------------- |
| (4,5) | 2        | same subtree    |
| (4,6) | 1        | across subtrees |
| (3,6) | 3        | ancestor        |

#### Complexity

- Time: $O(n + q \alpha(n))$
- Space: $O(n + q)$

Efficient for batch queries; each union and find is nearly constant-time.

Tarjan's LCA transforms the problem into a union-find dance, answering all ancestor questions in a single elegant DFS sweep.

### 384 Heavy-Light Decomposition

Heavy-Light Decomposition (HLD) splits a tree into heavy paths and light edges so that any path between two nodes can be broken into $O(\log n)$ segments.
It's the backbone for path queries, sum, max, min, or updates, using segment trees or Fenwick trees.

#### What Problem Are We Solving?

In many problems, we need to answer queries like:

- What's the sum or max on the path from $u$ to $v$?
- How to update values along a path?
- What's the minimum edge weight between two nodes?

Naively, walking the path is $O(n)$ per query.
With HLD, we decompose paths into few contiguous segments, reducing queries to $O(\log^2 n)$ (or better with LCA precomputation).

#### How It Works (Plain Language)

Every node chooses one heavy child (the one with largest subtree).
All other edges are light.
This guarantees:

- Each node is part of exactly one heavy path.
- Every root-to-leaf path crosses $O(\log n)$ light edges.

So we can represent each heavy path as a contiguous segment in an array,
and map tree queries → array range queries.

##### Steps

1. DFS 1:

   * Compute subtree size for each node.
   * Mark heavy child = largest subtree child.

2. DFS 2:

   * Assign head of heavy path.
   * Map each node to a position in linear order.

3. Path Query (u, v):

   * While $u$ and $v$ are not in same heavy path:

     * Always move the deeper head upward.
     * Query segment tree on that segment.
     * Jump $u$ to parent of its head.
   * When in same path: query segment from $u$ to $v$ (in order).

#### Example

Tree:

```
1
├── 2
│   ├── 4
│   └── 5
└── 3
    ├── 6
    └── 7
```

- Subtree sizes:
  $size(2)=3,; size(3)=3$
  Heavy edges: $(1,2)$, $(2,4)$, $(3,6)$

- Heavy paths:
  Path 1: 1 → 2 → 4
  Path 2: 3 → 6
  Light edges: $(1,3), (2,5), (3,7)$

Any query path $(4,7)$ crosses at most $\log n$ segments:

```
4→2→1 | 1→3 | 3→7
```

#### Tiny Code (Python)

```python
def dfs1(u, p, g, size, heavy):
    size[u] = 1
    max_sub = 0
    for v in g[u]:
        if v == p: continue
        dfs1(v, u, g, size, heavy)
        size[u] += size[v]
        if size[v] > max_sub:
            max_sub = size[v]
            heavy[u] = v

def dfs2(u, p, g, head, pos, cur_head, order, heavy):
    head[u] = cur_head
    pos[u] = len(order)
    order.append(u)
    if heavy[u] != -1:
        dfs2(heavy[u], u, g, head, pos, cur_head, order, heavy)
    for v in g[u]:
        if v == p or v == heavy[u]:
            continue
        dfs2(v, u, g, head, pos, v, order, heavy)

def query_path(u, v, head, pos, depth, segtree, lca):
    res = 0
    while head[u] != head[v]:
        if depth[head[u]] < depth[head[v]]:
            u, v = v, u
        res += segtree.query(pos[head[u]], pos[u])
        u = parent[head[u]]
    if depth[u] > depth[v]:
        u, v = v, u
    res += segtree.query(pos[u], pos[v])
    return res
```

#### Why It Matters

- Turns path queries into range queries
- Pairs beautifully with segment trees or Fenwick trees
- Core for:

  * Path sum / max / min
  * Path updates
  * Lowest edge weight
  * Subtree queries (with Euler mapping)

#### A Gentle Proof (Why It Works)

Every node moves at most $\log n$ times up light edges,
since subtree size at least halves each time.
Within a heavy path, nodes form contiguous segments,
so path queries can be broken into $O(\log n)$ contiguous intervals.

Thus, overall query time is $O(\log^2 n)$ (or $O(\log n)$ with segment tree optimization).

#### Try It Yourself

1. Draw a tree, compute subtree sizes.
2. Mark heavy edges to largest subtrees.
3. Assign head nodes and linear indices.
4. Try a query $(u,v)$ → decompose into path segments.
5. Observe each segment is contiguous in linear order.

#### Test Cases

| Query | Path      | Segments |
| ----- | --------- | -------- |
| (4,5) | 4→2→5     | 2        |
| (4,7) | 4→2→1→3→7 | 3        |
| (6,7) | 6→3→7     | 2        |

#### Complexity

- Preprocessing: $O(n)$
- Query: $O(\log^2 n)$ (or $O(\log n)$ with optimized segment tree)
- Space: $O(n)$

Heavy-Light Decomposition bridges tree topology and array queries, enabling logarithmic-time operations on paths.

### 385 Centroid Decomposition

Centroid Decomposition is a divide-and-conquer method that recursively breaks a tree into smaller parts using centroids, nodes that balance the tree when removed.
It transforms tree problems into logarithmic-depth recursion, enabling fast queries, updates, and path counting.

#### What Problem Are We Solving?

For certain tree problems (distance queries, path sums, subtree coverage, etc.),
we need to repeatedly split the tree into manageable pieces.

A centroid of a tree is a node such that, if removed,
no resulting component has more than $n/2$ nodes.

By recursively decomposing the tree using centroids,
we build a centroid tree, a hierarchical structure capturing the balance of the original tree.

#### How It Works (Plain Language)

1. Find the centroid:

   * Compute subtree sizes.
   * Pick the node whose largest child subtree ≤ $n/2$.

2. Record it as the root of this level's decomposition.

3. Remove the centroid from the tree.

4. Recurse on each remaining component (subtree).

Each node appears in $O(\log n)$ levels,
so queries or updates using this structure become logarithmic.

#### Example

Tree (7 nodes):

```
      1
    / | \
   2  3  4
  / \
 5   6
      \
       7
```

Step 1: $n=7$.
Subtree sizes → pick 2 (its largest child subtree ≤ 4).
Centroid = 2.

Step 2: Remove 2 → split into components:

- {5}, {6,7}, {1,3,4}

Step 3: Recursively find centroids in each component.

- {6,7} → centroid = 6
- {1,3,4} → centroid = 1

Centroid tree hierarchy:

```
     2
   / | \
  5  6  1
       \
        3
```

#### Tiny Code (Python)

```python
def build_centroid_tree(graph):
    n = len(graph)
    size = [0] * n
    removed = [False] * n
    parent = [-1] * n

    def dfs_size(u, p):
        size[u] = 1
        for v in graph[u]:
            if v == p or removed[v]:
                continue
            dfs_size(v, u)
            size[u] += size[v]

    def find_centroid(u, p, n):
        for v in graph[u]:
            if v != p and not removed[v] and size[v] > n // 2:
                return find_centroid(v, u, n)
        return u

    def decompose(u, p):
        dfs_size(u, -1)
        c = find_centroid(u, -1, size[u])
        removed[c] = True
        parent[c] = p
        for v in graph[c]:
            if not removed[v]:
                decompose(v, c)
        return c

    root = decompose(0, -1)
    return parent  # parent[c] is centroid parent
```

#### Why It Matters

Centroid decomposition allows efficient solutions to many tree problems:

- Path queries: distance sums, min/max weights
- Point updates: affect only $O(\log n)$ centroids
- Distance-based search: find nearest node of a type
- Divide-and-conquer DP on trees

It's especially powerful in competitive programming
for problems like *count pairs within distance k*, *color-based queries*, or *centroid tree construction*.

#### A Gentle Proof (Why It Works)

Every decomposition step removes one centroid.
By definition, each remaining subtree ≤ $n/2$.
Thus, recursion depth is $O(\log n)$,
and every node participates in at most $O(\log n)$ subproblems.

Total complexity:
$$
T(n) = T(n_1) + T(n_2) + \cdots + O(n) \le O(n \log n)
$$

#### Try It Yourself

1. Draw a small tree.
2. Compute subtree sizes.
3. Pick centroid (largest child ≤ $n/2$).
4. Remove and recurse.
5. Build centroid tree hierarchy.

#### Test Cases

| Tree                    | Centroid | Components   | Depth |
| ----------------------- | -------- | ------------ | ----- |
| Line (1–2–3–4–5)        | 3        | {1,2}, {4,5} | 2     |
| Star (1 center)         | 1        | leaves       | 1     |
| Balanced tree (7 nodes) | 2        | 3 components | 2     |

#### Complexity

- Preprocessing: $O(n \log n)$
- Query/Update (per node): $O(\log n)$
- Space: $O(n)$

Centroid Decomposition turns trees into balanced recursive hierarchies,
a universal tool for logarithmic-time query and update systems.

### 386 Tree Diameter (DFS Twice)

The tree diameter is the longest path between any two nodes in a tree.
A simple and elegant method to find it is to perform two DFS traversals, one to find the farthest node, and another to measure the longest path length.

#### What Problem Are We Solving?

In a tree (an acyclic connected graph), we want to find the diameter, defined as:

$$
\text{diameter} = \max_{u,v \in V} \text{dist}(u, v)
$$

where $\text{dist}(u,v)$ is the length (in edges or weights) of the shortest path between $u$ and $v$.

This path always lies between two leaf nodes, and can be found using two DFS/BFS traversals.

#### How It Works (Plain Language)

1. Pick any node (say 1).
2. Run DFS to find the farthest node from it → call it $A$.
3. Run DFS again from $A$ → find the farthest node from $A$ → call it $B$.
4. The distance between $A$ and $B$ is the diameter length.

Why this works:

- The first DFS ensures you start one end of the diameter.
- The second DFS stretches to the opposite end.

#### Example

Tree:

```
1
├── 2
│   └── 4
└── 3
    ├── 5
    └── 6
```

1. DFS(1): farthest node = 4 (distance 2)
2. DFS(4): farthest node = 6 (distance 4)

So diameter = 4 → path: $4 \to 2 \to 1 \to 3 \to 6$

#### Tiny Code (Python)

```python
from collections import defaultdict

def dfs(u, p, dist, graph):
    farthest = (u, dist)
    for v, w in graph[u]:
        if v == p:
            continue
        cand = dfs(v, u, dist + w, graph)
        if cand[1] > farthest[1]:
            farthest = cand
    return farthest

def tree_diameter(graph):
    start = 0
    a, _ = dfs(start, -1, 0, graph)
    b, diameter = dfs(a, -1, 0, graph)
    return diameter, (a, b)

# Example:
graph = defaultdict(list)
edges = [(0,1,1),(1,3,1),(0,2,1),(2,4,1),(2,5,1)]
for u,v,w in edges:
    graph[u].append((v,w))
    graph[v].append((u,w))

d, (a,b) = tree_diameter(graph)
print(d, (a,b))  # 4, path (3,5)
```

#### Why It Matters

- Simple and linear time: $O(n)$
- Works for weighted and unweighted trees
- Core for:

  * Tree center finding
  * Dynamic programming on trees
  * Network analysis (longest delay, max latency path)

#### A Gentle Proof (Why It Works)

Let $(u, v)$ be the true diameter endpoints.
Any DFS from any node $x$ finds a farthest node $A$.
By triangle inequality on trees, $A$ must be one endpoint of a diameter path.
A second DFS from $A$ finds the other endpoint $B$,
ensuring $\text{dist}(A,B)$ is the maximum possible.

#### Try It Yourself

1. Draw a small tree.
2. Pick any starting node, run DFS to find farthest.
3. From that node, run DFS again.
4. Trace the path between these two nodes, it's always the longest.

#### Test Cases

| Tree Type                | Diameter | Path                       |
| ------------------------ | -------- | -------------------------- |
| Line (1–2–3–4)           | 3        | (1,4)                      |
| Star (1 center + leaves) | 2        | (leaf₁, leaf₂)             |
| Balanced binary          | 4        | leftmost to rightmost leaf |

#### Complexity

- Time: $O(n)$
- Space: $O(n)$ recursion stack

Tree Diameter via two DFS is one of the cleanest tricks in graph theory, from any point, go far, then farther.

### 387 Tree DP (Subtree-Based Optimization)

Tree Dynamic Programming (Tree DP) is a technique that solves problems on trees by combining results from subtrees. Each node's result depends on its children, following a recursive pattern much like divide and conquer but on a tree.

#### What Problem Are We Solving?

Many problems on trees ask for optimal values (sum, min, max, count) that depend on child subtrees, for example:

- Maximum path sum in a tree
- Size of largest independent set
- Number of ways to color a tree
- Sum of distances to all nodes

Tree DP provides a bottom-up approach where we compute results for each subtree and merge them upwards.

#### How It Works (Plain Language)

1. Root the tree at an arbitrary node (often 1 or 0).
2. Define a DP state at each node based on its subtree.
3. Combine children's results recursively.
4. Return the result to the parent.

Example state:
$$
dp[u] = f(dp[v_1], dp[v_2], \dots, dp[v_k])
$$
where $v_i$ are children of $u$.

#### Example Problem

Find the size of the largest independent set (no two adjacent nodes chosen).

Recurrence:

$$
dp[u][0] = \sum_{v \in children(u)} \max(dp[v][0], dp[v][1])
$$

$$
dp[u][1] = 1 + \sum_{v \in children(u)} dp[v][0]
$$

#### Tiny Code (Python)

```python
from collections import defaultdict

graph = defaultdict(list)
edges = [(0,1),(0,2),(1,3),(1,4)]
for u,v in edges:
    graph[u].append(v)
    graph[v].append(u)

def dfs(u, p):
    incl = 1  # include u
    excl = 0  # exclude u
    for v in graph[u]:
        if v == p:
            continue
        inc_v, exc_v = dfs(v, u)
        incl += exc_v
        excl += max(inc_v, exc_v)
    return incl, excl

incl, excl = dfs(0, -1)
ans = max(incl, excl)
print(ans)  # 3
```

This finds the largest independent set size (3 nodes).

#### Why It Matters

Tree DP is foundational for:

- Counting: number of ways to assign labels, colors, or states
- Optimization: maximize or minimize cost over paths or sets
- Combinatorics: solve partition or constraint problems
- Game theory: compute win/loss states recursively

It turns global tree problems into local merges.

#### A Gentle Proof (Why It Works)

Each node's subtree is independent once you know whether its parent is included or not.
By processing children before parents (post-order DFS), we ensure all necessary data is ready when merging.
This guarantees correctness via structural induction on tree size.

#### Try It Yourself

1. Define a simple property (sum, count, max).
2. Write recurrence for a node in terms of its children.
3. Traverse via DFS, merge children's results.
4. Return value to parent, combine at root.

#### Template (Generic)

```python
def dfs(u, p):
    res = base_case()
    for v in children(u):
        if v != p:
            child = dfs(v, u)
            res = merge(res, child)
    return finalize(res)
```

#### Complexity

- Time: $O(n)$
- Space: $O(n)$ (recursion)

Tree DP transforms recursive thinking into structured computation, one subtree at a time, building the answer from the leaves to the root.

### 388 Rerooting DP (Compute All Roots' Answers)

Rerooting DP is an advanced tree dynamic programming technique that computes answers for every possible root efficiently. Instead of recomputing from scratch, we reuse results by propagating information down and up the tree.

#### What Problem Are We Solving?

Suppose we want the same value for every node as if that node were the root.
Examples:

- Sum of distances from each node to all others
- Number of nodes in each subtree
- DP value that depends on subtree structure

A naive approach runs $O(n)$ DP per root → $O(n^2)$ total.
Rerooting reduces this to $O(n)$ total.

#### How It Works (Plain Language)

1. First pass (downward DP):
   Compute answers for each subtree assuming root = 0.

2. Second pass (rerooting):
   Use parent's info to update child's answer —
   effectively "rerooting" the DP at each node.

The key is combining results from children excluding the child itself, often using prefix–suffix merges.

#### Example Problem

Compute sum of distances from each node to all others.

Recurrence:

- `subtree_size[u]` = number of nodes in subtree of `u`
- `dp[u]` = sum of distances from `u` to nodes in its subtree

First pass:
$$
dp[u] = \sum_{v \in children(u)} (dp[v] + subtree_size[v])
$$

Second pass:
When moving root from `u` to `v`:
$$
dp[v] = dp[u] - subtree_size[v] + (n - subtree_size[v])
$$

#### Tiny Code (Python)

```python
from collections import defaultdict

graph = defaultdict(list)
edges = [(0,1),(0,2),(2,3),(2,4)]
for u,v in edges:
    graph[u].append(v)
    graph[v].append(u)

n = 5
size = [1]*n
dp = [0]*n
ans = [0]*n

def dfs1(u, p):
    for v in graph[u]:
        if v == p: continue
        dfs1(v, u)
        size[u] += size[v]
        dp[u] += dp[v] + size[v]

def dfs2(u, p):
    ans[u] = dp[u]
    for v in graph[u]:
        if v == p: continue
        pu, pv = dp[u], dp[v]
        su, sv = size[u], size[v]

        dp[u] -= dp[v] + size[v]
        size[u] -= size[v]
        dp[v] += dp[u] + size[u]
        size[v] += size[u]

        dfs2(v, u)

        dp[u], dp[v] = pu, pv
        size[u], size[v] = su, sv

dfs1(0, -1)
dfs2(0, -1)

print(ans)
```

This computes the sum of distances for each node.

#### Why It Matters

- Efficiently derive all-root answers
- Common in centroid problems, sum of distances, rerooting sums
- Reduces redundant recomputation dramatically

#### A Gentle Proof (Why It Works)

Every DP depends only on local merges of subtrees.
By removing a child subtree and adding the rest, we adjust the parent's value to reflect a new root.
Induction over tree structure ensures correctness.

#### Try It Yourself

1. Write your DP recurrence for a single root.
2. Identify what changes when rerooting to a child.
3. Apply push–pull updates using stored subtree info.
4. Collect final answers after traversal.

#### Complexity

- Time: $O(n)$ (two DFS passes)
- Space: $O(n)$

Rerooting DP turns a global rerooting problem into a pair of local transformations, computing every node's perspective with elegant reuse.

### 389 Binary Search on Tree (Edge Weight Constraints)

Binary Search on Tree is a versatile strategy used to solve problems where we must find a threshold edge weight or path condition that satisfies a constraint.
It combines DFS or BFS traversal with binary search over a numeric property, often edge weights or limits.

#### What Problem Are We Solving?

Given a tree with weighted edges, we might want to answer questions like:

- What is the minimum edge weight so that the path between two nodes satisfies some property?
- What is the maximum allowed cost under which the tree remains connected?
- What is the smallest threshold that allows at least $k$ nodes to be reachable?

A naive approach checks each possible weight.
We can do better by binary searching over sorted edge weights.

#### How It Works (Plain Language)

1. Sort the edge weights or define a numeric search range.
2. Binary search over possible threshold $T$.
3. For each mid value, traverse the tree (via DFS/BFS/Union-Find) including only edges satisfying a condition (e.g., weight ≤ T).
4. Check if the property holds.
5. Narrow the search interval.

The key is monotonicity: if a property holds for $T$, it holds for all larger/smaller values.

#### Example Problem

Find the minimum edge weight threshold $T$ so that all nodes become connected.

Algorithm:

1. Sort edges by weight.
2. Binary search $T$ in $[min_weight, max_weight]$.
3. For each $T$, include edges with $w \le T$.
4. Check if resulting graph is connected.
5. Adjust bounds accordingly.

#### Tiny Code (Python)

```python
def is_connected(n, edges, limit):
    parent = list(range(n))
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    def union(a, b):
        pa, pb = find(a), find(b)
        if pa != pb:
            parent[pb] = pa

    for u, v, w in edges:
        if w <= limit:
            union(u, v)
    return len({find(i) for i in range(n)}) == 1

def binary_search_tree_threshold(n, edges):
    weights = sorted(set(w for _,_,w in edges))
    lo, hi = 0, len(weights)-1
    ans = weights[-1]
    while lo <= hi:
        mid = (lo + hi) // 2
        if is_connected(n, edges, weights[mid]):
            ans = weights[mid]
            hi = mid - 1
        else:
            lo = mid + 1
    return ans

edges = [(0,1,4),(1,2,2),(0,2,5)]
print(binary_search_tree_threshold(3, edges))  # Output: 4
```

#### Why It Matters

- Reduces complex "threshold" problems to logarithmic searches
- Works when solution space is monotonic
- Combines structural traversal with decision logic

Common applications:

- Path feasibility with limit
- Edge filtering by cost
- Tree queries with weight bounds

#### A Gentle Proof (Why It Works)

Let property $P(T)$ be true if condition holds for threshold $T$.
If $P(T)$ is monotonic (either always true beyond some point, or always false before), then binary search correctly converges to the minimal/maximal satisfying $T$.

#### Try It Yourself

1. Define a monotonic property on weights (e.g. "connected under limit").
2. Implement a decision function checking $P(T)$.
3. Wrap it in a binary search loop.
4. Return the minimal/maximal valid $T$.

#### Complexity

- Sorting edges: $O(E \log E)$
- Binary search: $O(\log E)$ checks
- Each check: $O(E \alpha(V))$ (Union-Find)
- Overall: $O(E \log^2 E)$ or better

Binary search on trees is not about searching *within* the tree, but *over constraints* defined on the tree, a sharp technique for threshold-based reasoning.

### 390 Virtual Tree (Query Subset Construction)

A Virtual Tree is a compressed representation of a tree built from a selected subset of nodes, along with their Lowest Common Ancestors (LCAs), connected in the same hierarchical order as the original tree.
It's used in query problems where we only need to reason about a small subset of nodes rather than the full tree.

#### What Problem Are We Solving?

Suppose we have a large tree and a query gives us a small set $S$ of nodes.
We need to compute some property among nodes in $S$ (like sum of distances, coverage, or DP over their paths).
Traversing the entire tree each time is inefficient.

A Virtual Tree shrinks the full tree down to just the nodes in $S$ plus their LCAs, keeping ancestry relationships intact.

This reduces the problem to a much smaller tree, often $O(|S|\log |S|)$ nodes instead of $O(n)$.

#### How It Works (Plain Language)

1. Collect the query nodes $S$.
2. Sort $S$ by Euler Tour order (entry time in DFS).
3. Insert LCAs of consecutive nodes to maintain structure.
4. Build edges between consecutive ancestors to form a tree.
5. Now process queries on this mini-tree instead of the full one.

#### Example

Given tree:

```
       1
     / | \
    2  3  4
      / \
     5   6
```

Query nodes $S = {2, 5, 6}$.

Their LCAs:

- $\text{LCA}(5,6)=3$
- $\text{LCA}(2,3)=1$

Virtual Tree Nodes = ${1,2,3,5,6}$
Connect them according to parent-child relations:
1 → 2
1 → 3
3 → 5
3 → 6

This is your Virtual Tree.

#### Tiny Code (C++ Style Pseudocode)

```cpp
vector<int> build_virtual_tree(vector<int>& S, vector<int>& tin, auto lca) {
    sort(S.begin(), S.end(), [&](int a, int b) { return tin[a] < tin[b]; });
    vector<int> nodes = S;
    for (int i = 0; i + 1 < (int)S.size(); i++)
        nodes.push_back(lca(S[i], S[i+1]));
    sort(nodes.begin(), nodes.end(), [&](int a, int b) { return tin[a] < tin[b]; });
    nodes.erase(unique(nodes.begin(), nodes.end()), nodes.end());
    stack<int> st;
    vector<vector<int>> vt_adj(nodes.size());
    for (int u : nodes) {
        while (!st.empty() && !is_ancestor(st.top(), u))
            st.pop();
        if (!st.empty())
            vt_adj[st.top()].push_back(u);
        st.push(u);
    }
    return nodes;
}
```

This builds adjacency for the virtual tree using stack ancestry.

#### Why It Matters

- Reduces large-tree queries to small subproblems
- Essential in problems with multiple subset queries
- Common in offline processing, rerooting DP, distance sum problems
- Works perfectly with LCA precomputation

#### A Gentle Proof (Why It Works)

Each node in $S$ must connect through its LCAs.
By sorting in Euler order and maintaining a stack of ancestors, we ensure that:

- Parent-child relations are consistent
- No cycles or duplicates
- The resulting tree preserves ancestry

So the virtual tree is a minimal subtree connecting all nodes in $S$.

#### Try It Yourself

1. Implement Euler Tour (get `tin[u]`).
2. Implement LCA (Binary Lifting or RMQ).
3. Build Virtual Tree for given $S$.
4. Apply your query logic (sum, count, DP).

#### Complexity

- Sorting $S$: $O(|S|\log|S|)$
- LCA calls: $O(|S|)$
- Building structure: $O(|S|)$
- Overall: $O(|S|\log|S|)$ per query

A Virtual Tree is your "query-scaled tree", a precise projection of the big tree onto the small world of your problem.

## Section 40. Advanced Graph Algorithms and Tricks 

### 391 Topological DP (Dynamic Programming on DAG)

Topological DP is a dynamic programming technique on Directed Acyclic Graphs (DAGs).
It computes values in dependency order, ensuring each node's state is calculated only after all its prerequisites.
This method is a core tool for problems involving partial orders, longest paths, counting paths, and propagation across dependencies.

#### What Problem Are We Solving?

In DAGs, some nodes depend on others (edges point from dependencies to dependents).
We often want to compute a DP value like:

- Longest path ending at each node
- Number of ways to reach a node
- Minimum cost to reach a node
- Accumulated value through dependencies

Because DAGs have no cycles, a topological order exists, allowing linear-time evaluation.

#### How It Works (Plain Language)

1. Topologically sort the DAG.
2. Initialize base cases (e.g. sources = 0 or 1).
3. Iterate in topo order, updating each node based on incoming edges.
4. Each node's value is final once processed.

This guarantees each dependency is resolved before use.

#### Example: Longest Path in DAG

Given DAG with edge weights $w(u,v)$, define DP:

$$
dp[v] = \max_{(u,v)\in E}(dp[u] + w(u,v))
$$

Base case: $dp[source] = 0$

#### Example DAG

```
1 → 2 → 4
 ↘︎ 3 ↗︎
```

Edges:

- 1 → 2 (1)
- 1 → 3 (2)
- 2 → 4 (1)
- 3 → 4 (3)

Topological order: [1, 2, 3, 4]

Compute:

- dp[1] = 0
- dp[2] = max(dp[1] + 1) = 1
- dp[3] = max(dp[1] + 2) = 2
- dp[4] = max(dp[2] + 1, dp[3] + 3) = max(2, 5) = 5

Result: Longest path length = 5

#### Tiny Code (C++ Style)

```cpp
vector<int> topo_sort(int n, vector<vector<int>>& adj) {
    vector<int> indeg(n, 0), topo;
    for (auto& u : adj)
        for (int v : u) indeg[v]++;
    queue<int> q;
    for (int i = 0; i < n; i++) if (indeg[i] == 0) q.push(i);
    while (!q.empty()) {
        int u = q.front(); q.pop();
        topo.push_back(u);
        for (int v : adj[u])
            if (--indeg[v] == 0) q.push(v);
    }
    return topo;
}

vector<int> topo_dp(int n, vector<vector<pair<int,int>>>& adj) {
    auto order = topo_sort(n, ...);
    vector<int> dp(n, INT_MIN);
    dp[0] = 0;
    for (int u : order)
        for (auto [v, w] : adj[u])
            dp[v] = max(dp[v], dp[u] + w);
    return dp;
}
```

#### Why It Matters

- Converts recursive dependencies into iterative computation
- Avoids redundant work
- Enables linear-time solutions for many DAG problems
- Works for counting, min/max, aggregation tasks

Common uses:

- Longest path in DAG
- Counting number of paths
- Minimum cost scheduling
- Project dependency planning

#### A Gentle Proof (Why It Works)

A topological order guarantees:

- For every edge $(u,v)$, $u$ appears before $v$
  Thus when processing $v$, all $dp[u]$ for predecessors $u$ are ready.
  This ensures correctness for any DP formula of the form:

$$
dp[v] = f({dp[u]\ |\ (u,v)\in E})
$$

#### Try It Yourself

1. Compute number of paths from source to each node:
   $dp[v] = \sum_{(u,v)} dp[u]$
2. Compute minimum cost if edges have weights
3. Build a longest chain of tasks with dependencies
4. Apply topological DP on SCC DAG (meta-graph)

#### Complexity

- Topo sort: $O(V+E)$
- DP propagation: $O(V+E)$
- Total: $O(V+E)$ time, $O(V)$ space

Topological DP is how you bring order to dependency chaos, one layer, one node, one dependency at a time.

### 392 SCC Condensed Graph DP (Dynamic Programming on Meta-Graph)

SCC Condensed Graph DP applies dynamic programming to the condensation of a directed graph into a Directed Acyclic Graph (DAG), where each node represents a strongly connected component (SCC).
This transforms a cyclic graph into an acyclic one, enabling topological reasoning, path aggregation, and value propagation across components.

#### What Problem Are We Solving?

Many problems on directed graphs become complex due to cycles.
Within each SCC, every node can reach every other, they are strongly connected.
By condensing SCCs into single nodes, we obtain a DAG:

$$
G' = (V', E')
$$

where each $v' \in V'$ is an SCC, and edges $(u', v')$ represent transitions between components.

Once the graph is acyclic, we can run Topological DP to compute:

- Maximum or minimum value reaching each SCC
- Number of paths between components
- Aggregated scores, weights, or costs

#### How It Works (Plain Language)

1. Find SCCs (using Tarjan or Kosaraju).
2. Build Condensed DAG: each SCC becomes a single node.
3. Aggregate initial values for each SCC (e.g. sum of weights).
4. Run DP over DAG in topological order, combining contributions from incoming edges.
5. Map back results to original nodes if needed.

This isolates cycles (internal SCCs) and manages dependencies cleanly.

#### Example

Original graph $G$:

```
1 → 2 → 3
↑    ↓
5 ← 4
```

SCCs:

- C₁ = {1,2,4,5}
- C₂ = {3}

Condensed graph:

```
C₁ → C₂
```

If each node has weight $w[i]$, then:

$$
dp[C₂] = \max_{(C₁,C₂)}(dp[C₁] + \text{aggregate}(C₁))
$$

#### Tiny Code (C++ Style)

```cpp
int n;
vector<vector<int>> adj;
vector<int> comp, order;
vector<bool> vis;

void dfs1(int u) {
    vis[u] = true;
    for (int v : adj[u]) if (!vis[v]) dfs1(v);
    order.push_back(u);
}

void dfs2(int u, int c, vector<vector<int>>& radj) {
    comp[u] = c;
    for (int v : radj[u]) if (comp[v] == -1) dfs2(v, c, radj);
}

vector<int> scc_condense() {
    vector<vector<int>> radj(n);
    for (int u=0; u<n; ++u)
        for (int v : adj[u]) radj[v].push_back(u);

    vis.assign(n,false);
    for (int i=0; i<n; ++i) if (!vis[i]) dfs1(i);

    comp.assign(n,-1);
    int cid=0;
    for (int i=n-1;i>=0;--i){
        int u = order[i];
        if (comp[u]==-1) dfs2(u,cid++,radj);
    }

    return comp;
}
```

Then build condensed graph:

```cpp
vector<vector<int>> dag(cid);
for (int u=0;u<n;++u)
  for (int v:adj[u])
    if (comp[u]!=comp[v])
      dag[comp[u]].push_back(comp[v]);
```

Run Topological DP on `dag`.

#### Why It Matters

- Turns cyclic graphs into acyclic DAGs
- Allows DP, path counting, and aggregation in cyclic contexts
- Simplifies reasoning about reachability and influence
- Forms the basis for:

  * Dynamic condensation
  * Meta-graph optimization
  * Modular graph analysis

#### A Gentle Proof (Why It Works)

1. Within each SCC, all nodes are mutually reachable.
2. Condensation merges these into single nodes, ensuring no cycles.
3. Edges between SCCs define a partial order, enabling topological sorting.
4. Any property defined recursively over dependencies can now be solved via DP on this order.

Formally, for each $C_i$:

$$
dp[C_i] = f({dp[C_j] \mid (C_j, C_i) \in E'}, \text{value}(C_i))
$$

#### Try It Yourself

1. Assign a weight to each node; compute max sum path over SCC DAG.
2. Count distinct paths between SCCs.
3. Combine SCC detection + DP for weighted reachability.
4. Solve problems like:

   * "Maximum gold in a dungeon with teleport cycles"
   * "Dependency graph with feedback loops"

#### Complexity

- SCC computation: $O(V+E)$
- DAG construction: $O(V+E)$
- Topo DP: $O(V'+E')$
- Total: $O(V+E)$

SCC Condensed Graph DP is the art of shrinking cycles into certainty, revealing a clean DAG beneath the tangled surface.

### 393 Eulerian Path

An Eulerian Path is a trail in a graph that visits every edge exactly once. If the path starts and ends at the same vertex, it is called an Eulerian Circuit. This concept lies at the heart of route planning, graph traversals, and network analysis.

#### What Problem Are We Solving?

We want to find a path that uses every edge once, no repeats, no omissions.

In an undirected graph, an Eulerian Path exists if and only if:

- Exactly 0 or 2 vertices have odd degree
- The graph is connected

In a directed graph, it exists if and only if:

- At most one vertex has `outdegree - indegree = 1` (start)
- At most one vertex has `indegree - outdegree = 1` (end)
- All other vertices have equal in-degree and out-degree
- The graph is strongly connected (or connected when ignoring direction)

#### How Does It Work (Plain Language)

1. Check degree conditions (undirected) or in/out-degree balance (directed).
2. Choose a start vertex:

   * For undirected: any odd-degree vertex (if exists), otherwise any vertex.
   * For directed: vertex with `outdeg = indeg + 1`, else any.
3. Apply Hierholzer's algorithm:

   * Walk edges greedily until stuck.
   * Backtrack and merge cycles into one path.
4. Reverse the constructed order for the final path.

#### Example (Undirected)

Graph edges:

```
1—2, 2—3, 3—1, 2—4
```

Degrees:

- deg(1)=2, deg(2)=3, deg(3)=2, deg(4)=1
  Odd vertices: 2, 4 → path exists (starts at 2, ends at 4)

Eulerian Path: `2 → 1 → 3 → 2 → 4`

#### Example (Directed)

Edges:

```
A → B, B → C, C → A, C → D
```

Degrees:

- A: out=1, in=1
- B: out=1, in=1
- C: out=2, in=1
- D: out=0, in=1

Start: C (`out=in+1`), End: D (`in=out+1`)

Eulerian Path: `C → A → B → C → D`

#### Tiny Code (C++)

```cpp
vector<vector<int>> adj;
vector<int> path;

void dfs(int u) {
    while (!adj[u].empty()) {
        int v = adj[u].back();
        adj[u].pop_back();
        dfs(v);
    }
    path.push_back(u);
}
```

Run from the start vertex, then reverse `path`.

#### Tiny Code (Python)

```python
def eulerian_path(graph, start):
    stack, path = [start], []
    while stack:
        u = stack[-1]
        if graph[u]:
            v = graph[u].pop()
            stack.append(v)
        else:
            path.append(stack.pop())
    return path[::-1]
```

#### Why It Matters

- Foundational for graph traversal problems
- Used in:

  * DNA sequencing (De Bruijn graph reconstruction)
  * Route planning (postal delivery, garbage collection)
  * Network diagnostics (tracing all connections)

#### A Gentle Proof (Why It Works)

Each edge must appear exactly once.
At every vertex (except possibly start/end), every entry must be matched with an exit.
This requires balanced degrees.

In an Eulerian Circuit:
$$
\text{in}(v) = \text{out}(v) \quad \forall v
$$

In an Eulerian Path:
$$
\exists \text{start with } \text{out}(v)=\text{in}(v)+1 \
\exists \text{end with } \text{in}(v)=\text{out}(v)+1
$$

Hierholzer's algorithm constructs the path by merging cycles and ensuring all edges are consumed.

#### Try It Yourself

1. Build a small graph and test parity conditions.
2. Implement Hierholzer's algorithm and trace each step.
3. Verify correctness by counting traversed edges.
4. Explore both directed and undirected variants.
5. Modify to detect Eulerian circuit vs path.

#### Complexity

- Time: $O(V+E)$
- Space: $O(V+E)$

Eulerian paths are elegant because they cover every connection exactly once, perfect order in perfect traversal.

### 394 Hamiltonian Path

A Hamiltonian Path is a path in a graph that visits every vertex exactly once. If it starts and ends at the same vertex, it forms a Hamiltonian Cycle. Unlike the Eulerian Path (which focuses on edges), the Hamiltonian Path focuses on vertices.

Finding one is a classic NP-complete problem, there's no known polynomial-time algorithm for general graphs.

#### What Problem Are We Solving?

We want to determine if there exists a path that visits every vertex exactly once, no repetition, no omission.

In formal terms:
Given a graph $G = (V, E)$, find a sequence of vertices
$$v_1, v_2, \ldots, v_n$$
such that $(v_i, v_{i+1}) \in E$ for all $i$, and all vertices are distinct.

For a Hamiltonian Cycle, additionally $(v_n, v_1) \in E$.

#### How Does It Work (Plain Language)

There's no simple parity or degree condition like in Eulerian paths.
We usually solve it by backtracking, bitmask DP, or heuristics (for large graphs):

1. Pick a start vertex.
2. Recursively explore all unvisited neighbors.
3. Mark visited vertices.
4. If all vertices are visited → found a Hamiltonian Path.
5. Otherwise, backtrack.

For small graphs, this brute force works; for large graphs, it's impractical.

#### Example (Undirected Graph)

Graph:

```
1, 2, 3
|    \   |
4 ———— 5
```

One Hamiltonian Path: `1 → 2 → 3 → 5 → 4`
One Hamiltonian Cycle: `1 → 2 → 3 → 5 → 4 → 1`

#### Example (Directed Graph)

```
A → B → C
↑       ↓
E ← D ← 
```

Possible Hamiltonian Path: `A → B → C → D → E`

#### Tiny Code (Backtracking – C++)

```cpp
bool hamiltonianPath(int u, vector<vector<int>>& adj, vector<bool>& visited, vector<int>& path, int n) {
    if (path.size() == n) return true;
    for (int v : adj[u]) {
        if (!visited[v]) {
            visited[v] = true;
            path.push_back(v);
            if (hamiltonianPath(v, adj, visited, path, n)) return true;
            visited[v] = false;
            path.pop_back();
        }
    }
    return false;
}
```

Call with each vertex as a potential start.

#### Tiny Code (DP Bitmask – C++)

Used for TSP-style Hamiltonian search:

```cpp
int n;
vector<vector<int>> dp(1 << n, vector<int>(n, INF));
dp[1][0] = 0;
for (int mask = 1; mask < (1 << n); ++mask) {
    for (int u = 0; u < n; ++u) if (mask & (1 << u)) {
        for (int v = 0; v < n; ++v) if (!(mask & (1 << v)) && adj[u][v]) {
            dp[mask | (1 << v)][v] = min(dp[mask | (1 << v)][v], dp[mask][u] + cost[u][v]);
        }
    }
}
```

#### Why It Matters

- Models ordering problems:

  * Traveling Salesman (TSP)
  * Job sequencing with constraints
  * Genome assembly paths
- Fundamental in theoretical computer science, cornerstone NP-complete problem.
- Helps distinguish easy (Eulerian) vs hard (Hamiltonian) traversal.

#### A Gentle Proof (Why It's Hard)

Hamiltonian Path existence is NP-complete:

1. Verification is easy (given a path, check in $O(V)$).
2. No known polynomial algorithm (unless $P=NP$).
3. Many problems reduce to it (like TSP).

This means it likely requires exponential time in general:
$$
O(n!)
$$
in naive form, or
$$
O(2^n n)
$$
using bitmask DP.

#### Try It Yourself

1. Build small graphs (4–6 vertices) and trace paths.
2. Compare with Eulerian path conditions.
3. Implement backtracking search.
4. Extend to cycle detection (check edge back to start).
5. Try bitmask DP for small $n \le 20$.

#### Complexity

- Time (backtracking): $O(n!)$
- Time (DP bitmask): $O(2^n \cdot n^2)$
- Space: $O(2^n \cdot n)$

Hamiltonian paths capture the essence of combinatorial explosion, simple to state, hard to solve, yet central to understanding computational limits.

### 395 Chinese Postman Problem (Route Inspection)

The Chinese Postman Problem (CPP), also known as the Route Inspection Problem, asks for the shortest closed path that traverses every edge of a graph at least once.
It generalizes the Eulerian Circuit, allowing edge repetitions when necessary.

#### What Problem Are We Solving?

Given a weighted graph $G = (V, E)$, we want to find a minimum-cost tour that covers all edges at least once and returns to the start.

If $G$ is Eulerian (all vertices have even degree), the answer is simple, the Eulerian Circuit itself.
Otherwise, we must duplicate edges strategically to make the graph Eulerian, minimizing total added cost.

#### How It Works (Plain Language)

1. Check vertex degrees:

   * Count how many have odd degree.
2. If all even → just find Eulerian Circuit.
3. If some odd →

   * Pair up odd vertices in such a way that the sum of shortest path distances between paired vertices is minimal.
   * Duplicate edges along those shortest paths.
4. The resulting graph is Eulerian, so an Eulerian Circuit can be constructed.
5. The cost of this circuit is the sum of all edges + added edges.

#### Example

Graph edges (with weights):

| Edge | Weight |
| ---- | ------ |
| 1–2  | 3      |
| 2–3  | 2      |
| 3–4  | 4      |
| 4–1  | 3      |
| 2–4  | 1      |

Degrees:

- deg(1)=2, deg(2)=3, deg(3)=2, deg(4)=3 → odd vertices: 2, 4
  Shortest path between 2–4: 1
  Add that again → now all even

Total cost = (3 + 2 + 4 + 3 + 1) + 1 = 14

#### Algorithm (Steps)

1. Identify odd-degree vertices
2. Compute shortest path matrix (Floyd–Warshall)
3. Solve minimum-weight perfect matching among odd vertices
4. Duplicate edges in the matching
5. Perform Eulerian Circuit traversal (Hierholzer's algorithm)

#### Tiny Code (Pseudocode)

```python
def chinese_postman(G):
    odd = [v for v in G if degree(v) % 2 == 1]
    if not odd:
        return eulerian_circuit(G)
    
    dist = floyd_warshall(G)
    pairs = minimum_weight_matching(odd, dist)
    for (u, v) in pairs:
        add_path(G, u, v, dist)
    return eulerian_circuit(G)
```

#### Why It Matters

- Core algorithm in network optimization
- Used in:

  * Postal route planning
  * Garbage collection routing
  * Snow plow scheduling
  * Street sweeping
- Demonstrates how graph augmentation can solve traversal problems efficiently

#### A Gentle Proof (Why It Works)

To traverse every edge, all vertices must have even degree (Eulerian condition).
When vertices with odd degree exist, we must pair them up to restore evenness.

The minimal duplication set is a minimum-weight perfect matching among odd vertices:

$$
\text{Extra cost} = \min_{\text{pairing } M} \sum_{(u,v) \in M} \text{dist}(u,v)
$$

Thus, the optimal path cost:

$$
C = \sum_{e \in E} w(e) + \text{Extra cost}
$$

#### Try It Yourself

1. Draw a graph with odd-degree vertices.
2. Identify odd vertices and shortest pair distances.
3. Compute minimal matching manually.
4. Add duplicated edges and find Eulerian Circuit.
5. Compare total cost before and after duplication.

#### Complexity

- Floyd–Warshall: $O(V^3)$
- Minimum Matching: $O(V^3)$
- Eulerian traversal: $O(E)$
- Total: $O(V^3)$

The Chinese Postman Problem transforms messy graphs into elegant tours, balancing degrees, minimizing effort, and ensuring every edge gets its due.

### 396 Hierholzer's Algorithm

Hierholzer's Algorithm is the classical method for finding an Eulerian Path or Eulerian Circuit in a graph. It constructs the path by merging cycles until all edges are used exactly once.

#### What Problem Are We Solving?

We want to find an Eulerian trail, a path or circuit that visits every edge exactly once.

- For an Eulerian Circuit (closed trail):
  All vertices have even degree.
- For an Eulerian Path (open trail):
  Exactly two vertices have odd degree.

The algorithm efficiently constructs the path in linear time relative to the number of edges.

#### How It Works (Plain Language)

1. Check Eulerian conditions:

   * 0 odd-degree vertices → Eulerian Circuit
   * 2 odd-degree vertices → Eulerian Path (start at one odd)
2. Start from a valid vertex (odd if path, any if circuit)
3. Traverse edges greedily:

   * Follow edges until you return to the start or cannot continue.
4. Backtrack and merge:

   * When stuck, backtrack to a vertex with unused edges, start a new cycle, and merge it into the current path.
5. Continue until all edges are used.

The final sequence of vertices is the Eulerian trail.

#### Example (Undirected Graph)

Graph:

```
1, 2
|   |
4, 3
```

All vertices have even degree, so an Eulerian Circuit exists.

Start at 1:

1 → 2 → 3 → 4 → 1

Result: Eulerian Circuit = [1, 2, 3, 4, 1]

#### Example (With Odd Vertices)

Graph:

```
1, 2, 3
```

deg(1)=1, deg(2)=2, deg(3)=1 → Eulerian Path exists
Start at 1:

1 → 2 → 3

#### Tiny Code (C++)

```cpp
vector<vector<int>> adj;
vector<int> path;

void dfs(int u) {
    while (!adj[u].empty()) {
        int v = adj[u].back();
        adj[u].pop_back();
        dfs(v);
    }
    path.push_back(u);
}
```

After DFS, `path` will contain vertices in reverse order.
Reverse it to get the Eulerian path or circuit.

#### Tiny Code (Python)

```python
def hierholzer(graph, start):
    stack, path = [start], []
    while stack:
        u = stack[-1]
        if graph[u]:
            v = graph[u].pop()
            graph[v].remove(u)  # remove both directions for undirected
            stack.append(v)
        else:
            path.append(stack.pop())
    return path[::-1]
```

#### Why It Matters

- Efficiently constructs Eulerian Paths in $O(V + E)$
- Backbone for:

  * Chinese Postman Problem
  * Eulerian Circuit detection
  * DNA sequencing (De Bruijn graphs)
  * Route design and network analysis

#### A Gentle Proof (Why It Works)

- Every time you traverse an edge, it's removed (used once).
- Each vertex retains balanced degree (entries = exits).
- When you get stuck, the subpath formed is a cycle.
- Merging all such cycles yields a single complete traversal.

Thus, every edge appears exactly once in the final route.

#### Try It Yourself

1. Draw small Eulerian graphs.
2. Manually trace Hierholzer's algorithm.
3. Identify start vertex (odd-degree or any).
4. Verify path covers all edges exactly once.
5. Apply to both directed and undirected graphs.

#### Complexity

- Time: $O(V + E)$
- Space: $O(V + E)$

Hierholzer's Algorithm elegantly builds order from connectivity, ensuring each edge finds its place in a perfect traversal.

### 397 Johnson's Cycle Finding Algorithm

Johnson's Algorithm is a powerful method for enumerating all simple cycles (elementary circuits) in a directed graph. A *simple cycle* is one that visits no vertex more than once, except the starting/ending vertex.

Unlike DFS-based approaches that can miss or duplicate cycles, Johnson's method systematically lists each cycle exactly once, running in O((V + E)(C + 1)), where C is the number of cycles.

#### What Problem Are We Solving?

We want to find all simple cycles in a directed graph $G = (V, E)$.

That is, find all vertex sequences
$$v_1 \to v_2 \to \ldots \to v_k \to v_1$$
where each $v_i$ is distinct and edges exist between consecutive vertices.

Enumerating all cycles is fundamental for:

- Dependency analysis
- Feedback detection
- Circuit design
- Graph motif analysis

#### How It Works (Plain Language)

Johnson's algorithm is based on backtracking with smart pruning and SCC decomposition:

1. Process vertices in order

   * Consider vertices $1, 2, \dots, n$.
2. For each vertex $s$:

   * Consider the subgraph induced by vertices ≥ s.
   * Find strongly connected components (SCCs) in this subgraph.
   * If $s$ belongs to a nontrivial SCC, explore all simple cycles starting at $s$.
3. Use a blocked set to avoid redundant exploration:

   * Once a vertex leads to a dead-end, mark it blocked.
   * Unblock when a valid cycle is found through it.

This avoids exploring the same path multiple times.

#### Example

Graph:

```
A → B → C
↑   ↓   |
└── D ←─┘
```

Cycles:

1. A → B → C → D → A
2. B → C → D → B

Johnson's algorithm will find both efficiently, without duplication.

#### Pseudocode

```python
def johnson(G):
    result = []
    blocked = set()
    B = {v: set() for v in G}
    stack = []

    def circuit(v, s):
        f = False
        stack.append(v)
        blocked.add(v)
        for w in G[v]:
            if w == s:
                result.append(stack.copy())
                f = True
            elif w not in blocked:
                if circuit(w, s):
                    f = True
        if f:
            unblock(v)
        else:
            for w in G[v]:
                B[w].add(v)
        stack.pop()
        return f

    def unblock(u):
        blocked.discard(u)
        for w in B[u]:
            if w in blocked:
                unblock(w)
        B[u].clear()

    for s in sorted(G.keys()):
        # consider subgraph of nodes >= s
        subG = {v: [w for w in G[v] if w >= s] for v in G if v >= s}
        SCCs = strongly_connected_components(subG)
        if not SCCs:
            continue
        scc = min(SCCs, key=lambda S: min(S))
        s_node = min(scc)
        circuit(s_node, s_node)
    return result
```

#### Why It Matters

- Enumerates all simple cycles without duplicates
- Works for directed graphs (unlike many undirected-only algorithms)
- Key in:

  * Deadlock detection
  * Cycle basis computation
  * Feedback arc set analysis
  * Subgraph pattern mining

#### A Gentle Proof (Why It Works)

Each recursive search begins from the smallest vertex in an SCC.
By restricting search to vertices $\ge s$, every cycle is discovered exactly once (at its least-numbered vertex).
The blocked set prevents repeated exploration of dead ends.
Unblocking ensures vertices re-enter search space when part of a valid cycle.

This guarantees:

- No cycle is missed
- No cycle is duplicated

#### Try It Yourself

1. Draw a small directed graph with 3–5 vertices.
2. Identify SCCs manually.
3. Apply the algorithm step-by-step, noting `blocked` updates.
4. Record each cycle when returning to start.
5. Compare with naive DFS enumeration.

#### Complexity

- Time: $O((V + E)(C + 1))$
- Space: $O(V + E)$

Johnson's algorithm reveals the hidden loops inside directed graphs, one by one, exhaustively and elegantly.

### 398 Transitive Closure (Floyd–Warshall)

The Transitive Closure of a directed graph captures reachability:
it tells us, for every pair of vertices $(u, v)$, whether there exists a path from $u$ to $v$.

It's often represented as a boolean matrix $R$, where
$$
R[u][v] = 1 \text{ if and only if there is a path from } u \text{ to } v
$$

This can be computed efficiently using a modified version of the Floyd–Warshall algorithm.

#### What Problem Are We Solving?

Given a directed graph $G = (V, E)$, we want to determine for every pair $(u, v)$ whether:

$$
u \leadsto v
$$

That is, can we reach $v$ from $u$ through a sequence of directed edges?

The output is a reachability matrix, useful in:

- Dependency analysis
- Access control and authorization graphs
- Program call graphs
- Database query optimization

#### How It Works (Plain Language)

We apply the Floyd–Warshall dynamic programming idea,
but instead of distances, we propagate reachability.

Let $R[u][v] = 1$ if $u \to v$ (direct edge), otherwise $0$.
Then for each vertex $k$ (intermediate node), we update:

$$
R[u][v] = R[u][v] \lor (R[u][k] \land R[k][v])
$$

Intuitively:
"$u$ can reach $v$ if $u$ can reach $k$ and $k$ can reach $v$."

#### Example

Graph:

```
A → B → C
↑         |
└─────────┘
```

Initial reachability (direct edges):

```
A B C
A 0 1 0
B 0 0 1
C 1 0 0
```

After applying transitive closure:

```
A B C
A 1 1 1
B 1 1 1
C 1 1 1
```

Every vertex is reachable from every other, the graph is strongly connected.

#### Tiny Code (C)

```c
#define N 100
int R[N][N];

void floyd_warshall_tc(int n) {
    for (int k = 0; k < n; ++k)
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                R[i][j] = R[i][j] || (R[i][k] && R[k][j]);
}
```

#### Tiny Code (Python)

```python
def transitive_closure(R):
    n = len(R)
    for k in range(n):
        for i in range(n):
            for j in range(n):
                R[i][j] = R[i][j] or (R[i][k] and R[k][j])
    return R
```

#### Why It Matters

- Converts a graph into a reachability matrix
- Enables constant-time queries: "Can $u$ reach $v$?"
- Used in:

  * Compilers (call dependencies)
  * Databases (recursive queries)
  * Security graphs
  * Network analysis

#### A Gentle Proof (Why It Works)

We extend reachability step by step:

- Base: $R^{(0)}$ = direct edges
- Step: $R^{(k)}$ = paths using vertices ${1, 2, \dots, k}$ as intermediates

By induction:
$$
R^{(k)}[i][j] = 1 \iff \text{there exists a path } i \to j \text{ using vertices } \le k
$$

At the end ($k = n$), $R^{(n)}$ contains all possible paths.

#### Try It Yourself

1. Create a directed graph with 4–5 nodes.
2. Build its adjacency matrix.
3. Apply the algorithm by hand.
4. Observe how new reachabilities emerge after each $k$.
5. Compare with paths you can see visually.

#### Complexity

- Time: $O(V^3)$
- Space: $O(V^2)$

Transitive closure transforms reachability into certainty, mapping every potential path into a single clear view of what connects to what.

### 399 Graph Coloring (Backtracking)

Graph Coloring is the problem of assigning colors to vertices of a graph so that no two adjacent vertices share the same color.
The smallest number of colors required is called the chromatic number of the graph.

This classic constraint satisfaction problem lies at the heart of scheduling, register allocation, and pattern assignment.

#### What Problem Are We Solving?

Given a graph $G = (V, E)$ and an integer $k$,
determine whether it is possible to color all vertices using at most $k$ colors such that:

$$
\forall (u, v) \in E, ; \text{color}(u) \ne \text{color}(v)
$$

If such a coloring exists, $G$ is $k$-colorable.

#### How It Works (Plain Language)

We solve this using backtracking:

1. Assign a color to the first vertex.
2. Move to the next vertex.
3. Try all available colors (from $1$ to $k$).
4. If a color assignment violates adjacency constraints, skip it.
5. If a vertex cannot be colored, backtrack to previous vertex and change its color.
6. Continue until:

   * all vertices are colored (success), or
   * no valid assignment exists (failure).

#### Example

Graph:

```
1, 2
|   |
3, 4
```

A square requires at least 2 colors:

- color(1) = 1
- color(2) = 2
- color(3) = 2
- color(4) = 1

Valid 2-coloring.

Try 1 color → fails (adjacent nodes same color)
Try 2 colors → success → chromatic number = 2

#### Tiny Code (C++)

```cpp
int n, k;
vector<vector<int>> adj;
vector<int> color;

bool isSafe(int v, int c) {
    for (int u : adj[v])
        if (color[u] == c) return false;
    return true;
}

bool solve(int v) {
    if (v == n) return true;
    for (int c = 1; c <= k; ++c) {
        if (isSafe(v, c)) {
            color[v] = c;
            if (solve(v + 1)) return true;
            color[v] = 0;
        }
    }
    return false;
}
```

#### Tiny Code (Python)

```python
def graph_coloring(graph, k):
    n = len(graph)
    color = [0] * n

    def safe(v, c):
        return all(color[u] != c for u in range(n) if graph[v][u])

    def backtrack(v):
        if v == n:
            return True
        for c in range(1, k + 1):
            if safe(v, c):
                color[v] = c
                if backtrack(v + 1):
                    return True
                color[v] = 0
        return False

    return backtrack(0)
```

#### Why It Matters

Graph coloring captures the essence of constraint-based allocation:

- Scheduling: assign time slots to tasks
- Register allocation: map variables to CPU registers
- Map coloring: color regions with shared boundaries
- Frequency assignment: allocate channels in wireless networks

#### A Gentle Proof (Why It Works)

We explore all possible assignments (depth-first search) under the rule:
$$
\forall (u, v) \in E, ; \text{color}(u) \ne \text{color}(v)
$$

Backtracking prunes partial solutions that cannot lead to valid full assignments.
When a full coloring is found, constraints are satisfied by construction.

By completeness of backtracking, if a valid $k$-coloring exists, it will be found.

#### Try It Yourself

1. Draw small graphs (triangle, square, pentagon).
2. Attempt coloring with $k = 2, 3, 4$.
3. Observe where conflicts force backtracking.
4. Try greedy coloring and compare with backtracking.
5. Identify the chromatic number experimentally.

#### Complexity

- Time: $O(k^V)$ (exponential worst case)
- Space: $O(V)$

Graph coloring blends search and logic, a careful dance through the constraints, discovering harmony one color at a time.

### 400 Articulation Points & Bridges

Articulation Points and Bridges identify weak spots in a graph, nodes or edges whose removal increases the number of connected components.
They are essential in analyzing network resilience, communication reliability, and biconnected components.

#### What Problem Are We Solving?

Given an undirected graph $G = (V, E)$, find:

- Articulation Points (Cut Vertices):
  Vertices whose removal disconnects the graph.

- Bridges (Cut Edges):
  Edges whose removal disconnects the graph.

We want efficient algorithms to detect these in $O(V + E)$ time.

#### How It Works (Plain Language)

We use a single DFS traversal (Tarjan's algorithm) with two key arrays:

- `disc[v]`: discovery time of vertex $v$
- `low[v]`: the lowest discovery time reachable from $v$ (including back edges)

During DFS:

- A vertex $u$ is an articulation point if:

  * $u$ is root and has more than one child, or
  * $\exists$ child $v$ such that `low[v] ≥ disc[u]`

- An edge $(u, v)$ is a bridge if:

  * `low[v] > disc[u]`

These conditions detect when no back-edge connects a subtree back to an ancestor.

#### Example

Graph:

```
  1
 / \
2   3
|   |
4   5
```

Remove node 2 → node 4 becomes isolated → 2 is an articulation point.
Remove edge (2, 4) → increases components → (2, 4) is a bridge.

#### Tiny Code (C++)

```cpp
vector<vector<int>> adj;
vector<int> disc, low, parent;
vector<bool> ap;
int timer = 0;

void dfs(int u) {
    disc[u] = low[u] = ++timer;
    int children = 0;

    for (int v : adj[u]) {
        if (!disc[v]) {
            parent[v] = u;
            ++children;
            dfs(v);
            low[u] = min(low[u], low[v]);

            if (parent[u] == -1 && children > 1)
                ap[u] = true;
            if (parent[u] != -1 && low[v] >= disc[u])
                ap[u] = true;
            if (low[v] > disc[u])
                cout << "Bridge: " << u << " - " << v << "\n";
        } else if (v != parent[u]) {
            low[u] = min(low[u], disc[v]);
        }
    }
}
```

#### Tiny Code (Python)

```python
def articulation_points_and_bridges(graph):
    n = len(graph)
    disc = [0] * n
    low = [0] * n
    parent = [-1] * n
    ap = [False] * n
    bridges = []
    time = 1

    def dfs(u):
        nonlocal time
        disc[u] = low[u] = time
        time += 1
        children = 0
        for v in graph[u]:
            if not disc[v]:
                parent[v] = u
                children += 1
                dfs(v)
                low[u] = min(low[u], low[v])

                if parent[u] == -1 and children > 1:
                    ap[u] = True
                if parent[u] != -1 and low[v] >= disc[u]:
                    ap[u] = True
                if low[v] > disc[u]:
                    bridges.append((u, v))
            elif v != parent[u]:
                low[u] = min(low[u], disc[v])

    for i in range(n):
        if not disc[i]:
            dfs(i)

    return [i for i, x in enumerate(ap) if x], bridges
```

#### Why It Matters

Articulation points and bridges reveal critical nodes and links in:

- Network design: identify weak links in infrastructure
- Social networks: find influencers whose removal splits communities
- Compiler dependency graphs: locate critical connections
- Transport systems: ensure robust routing

Understanding where a graph breaks helps us design systems that don't.

#### A Gentle Proof (Why It Works)

For each node $u$, `low[u]` captures the earliest discovered vertex reachable via DFS or a back edge.
If a child subtree cannot reach an ancestor of $u$, then $u$ is a bottleneck, removing it splits the graph.

The inequalities:
$$
low[v] \ge disc[u] \implies u \text{ is articulation point}
$$
$$
low[v] > disc[u] \implies (u,v) \text{ is bridge}
$$

are derived from whether a subtree is connected back to an ancestor.

#### Try It Yourself

1. Draw small graphs (triangle, line, star).
2. Manually run DFS and record `disc` and `low`.
3. Identify articulation points and bridges.
4. Try adding edges, see how redundancy removes articulation points.

#### Complexity

- Time: $O(V + E)$
- Space: $O(V)$

Finding articulation points and bridges transforms structure into insight, helping you build networks that stay connected even when parts fail.




