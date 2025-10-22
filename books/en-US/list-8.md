# Chapter 8. Geometry, Graphics and Spatial Algorithms 

# Section 71. Convex Hull 

### 701 Gift Wrapping (Jarvis March)

Gift Wrapping, or Jarvis March, is one of the simplest and most intuitive algorithms for finding the convex hull of a set of points, the smallest convex polygon that encloses them all. Think of it like wrapping a rubber band around nails on a board.

It "wraps" the hull one point at a time by repeatedly selecting the most counterclockwise point until it returns to the start.

#### What Problem Are We Solving?

Given n points in the plane, we want to compute their convex hull, the polygon formed by connecting the outermost points in order. The convex hull is fundamental in geometry, graphics, and robotics.

Formally, the convex hull H(S) of a set S is the smallest convex set containing S.

We want an algorithm that:

* Finds all points on the hull.
* Orders them along the perimeter.
* Works reliably even with collinear points.

Jarvis March is conceptually simple and good for small or nearly convex sets.

#### How Does It Work (Plain Language)?

Imagine standing at the leftmost point and walking around the outside, always turning as left as possible (counterclockwise). That ensures we trace the hull boundary.

Algorithm steps:

| Step | Description                                                                                           |
| ---- | ----------------------------------------------------------------------------------------------------- |
| 1    | Start from the leftmost (or lowest) point.                                                            |
| 2    | Choose the next point p such that all other points lie to the right of the line (current, p). |
| 3    | Move to p, add it to the hull.                                                                    |
| 4    | Repeat until you return to the start.                                                                 |

This mimics "wrapping" around all points, hence Gift Wrapping.

#### Example Walkthrough

Suppose we have 6 points:
A(0,0), B(2,1), C(1,2), D(3,3), E(0,3), F(3,0)

Start at A(0,0) (leftmost).
From A, the most counterclockwise point is E(0,3).
From E, turn leftmost again → D(3,3).
From D → F(3,0).
From F → back to A.

Hull = [A, E, D, F]

#### Tiny Code (Easy Version)

C

```c
#include <stdio.h>

typedef struct { double x, y; } Point;

int orientation(Point a, Point b, Point c) {
    double val = (b.y - a.y) * (c.x - b.x) - 
                 (b.x - a.x) * (c.y - b.y);
    if (val == 0) return 0;      // collinear
    return (val > 0) ? 1 : 2;    // 1: clockwise, 2: counterclockwise
}

void convexHull(Point pts[], int n) {
    if (n < 3) return;
    int hull[100], h = 0;
    int l = 0;
    for (int i = 1; i < n; i++)
        if (pts[i].x < pts[l].x)
            l = i;

    int p = l, q;
    do {
        hull[h++] = p;
        q = (p + 1) % n;
        for (int i = 0; i < n; i++)
            if (orientation(pts[p], pts[i], pts[q]) == 2)
                q = i;
        p = q;
    } while (p != l);

    printf("Convex Hull:\n");
    for (int i = 0; i < h; i++)
        printf("(%.1f, %.1f)\n", pts[hull[i]].x, pts[hull[i]].y);
}

int main(void) {
    Point pts[] = {{0,0},{2,1},{1,2},{3,3},{0,3},{3,0}};
    int n = sizeof(pts)/sizeof(pts[0]);
    convexHull(pts, n);
}
```

Python

```python
def orientation(a, b, c):
    val = (b[1]-a[1])*(c[0]-b[0]) - (b[0]-a[0])*(c[1]-b[1])
    if val == 0: return 0
    return 1 if val > 0 else 2

def convex_hull(points):
    n = len(points)
    if n < 3: return []
    l = min(range(n), key=lambda i: points[i][0])
    hull = []
    p = l
    while True:
        hull.append(points[p])
        q = (p + 1) % n
        for i in range(n):
            if orientation(points[p], points[i], points[q]) == 2:
                q = i
        p = q
        if p == l: break
    return hull

pts = [(0,0),(2,1),(1,2),(3,3),(0,3),(3,0)]
print("Convex Hull:", convex_hull(pts))
```

#### Why It Matters

* Simple and intuitive: easy to visualize and implement.
* Works on any set of points, even non-sorted.
* Output-sensitive: time depends on number of hull points *h*.
* Good baseline for comparing more advanced algorithms (Graham, Chan).

Applications:

* Robotics and path planning (boundary detection)
* Computer graphics (collision envelopes)
* GIS and mapping (territory outline)
* Clustering and outlier detection

#### Try It Yourself

1. Try with points forming a square, triangle, or concave shape.
2. Add collinear points, see if they're included.
3. Visualize each orientation step (plot arrows).
4. Count comparisons (to verify O(nh)).
5. Compare with Graham Scan and Andrew's Monotone Chain.

#### Test Cases

| Points                         | Hull Output         | Notes             |
| ------------------------------ | ------------------- | ----------------- |
| Square (0,0),(0,1),(1,0),(1,1) | All 4 points        | Perfect rectangle |
| Triangle (0,0),(2,0),(1,1)     | 3 points            | Simple convex     |
| Concave shape                  | Outer boundary only | Concavity ignored |
| Random points                  | Varies              | Always convex     |

#### Complexity

* Time: O(nh), where *h* = hull points count
* Space: O(h) for output list

Gift Wrapping is your first compass in computational geometry, follow the leftmost turns, and the shape of your data reveals itself.

### 702 Graham Scan

Graham Scan is a fast, elegant algorithm for finding the convex hull of a set of points. It works by sorting the points by angle around an anchor and then scanning to build the hull while maintaining a stack of turning directions.

Think of it like sorting all your stars around a basepoint, then tracing the outermost ring without stepping back inside.

#### What Problem Are We Solving?

Given n points on a plane, we want to find the convex hull, the smallest convex polygon enclosing all points.

Unlike Gift Wrapping, which walks around points one by one, Graham Scan sorts them first, then efficiently traces the hull in a single pass.

We need:

* A consistent ordering (polar angle)
* A way to test turns (orientation)

#### How Does It Work (Plain Language)?

1. Pick the anchor point, the one with the lowest y (and lowest x if tie).
2. Sort all points by polar angle with respect to the anchor.
3. Scan through sorted points, maintaining a stack of hull vertices.
4. For each point, check the last two in the stack:

   * If they make a non-left turn (clockwise), pop the last one.
   * Keep doing this until it turns left (counterclockwise).
   * Push the new point.
5. At the end, the stack holds the convex hull in order.

#### Example Walkthrough

Points:
A(0,0), B(2,1), C(1,2), D(3,3), E(0,3), F(3,0)

1. Anchor: A(0,0)
2. Sort by polar angle → F(3,0), B(2,1), D(3,3), C(1,2), E(0,3)
3. Scan:

   * Start [A, F, B]
   * Check next D → left turn → push
   * Next C → right turn → pop D
   * Push C → check with B, still right turn → pop B
   * Continue until all are scanned
     Hull: A(0,0), F(3,0), D(3,3), E(0,3)

#### Tiny Code (Easy Version)

C

```c
#include <stdio.h>
#include <stdlib.h>

typedef struct { double x, y; } Point;

Point anchor;

int orientation(Point a, Point b, Point c) {
    double val = (b.y - a.y) * (c.x - b.x) - 
                 (b.x - a.x) * (c.y - b.y);
    if (val == 0) return 0;
    return (val > 0) ? 1 : 2;
}

double dist(Point a, Point b) {
    double dx = a.x - b.x, dy = a.y - b.y;
    return dx * dx + dy * dy;
}

int compare(const void *p1, const void *p2) {
    Point a = *(Point *)p1, b = *(Point *)p2;
    int o = orientation(anchor, a, b);
    if (o == 0)
        return dist(anchor, a) < dist(anchor, b) ? -1 : 1;
    return (o == 2) ? -1 : 1;
}

void grahamScan(Point pts[], int n) {
    int ymin = 0;
    for (int i = 1; i < n; i++)
        if (pts[i].y < pts[ymin].y ||
           (pts[i].y == pts[ymin].y && pts[i].x < pts[ymin].x))
            ymin = i;

    Point temp = pts[0];
    pts[0] = pts[ymin];
    pts[ymin] = temp;
    anchor = pts[0];

    qsort(pts + 1, n - 1, sizeof(Point), compare);

    Point stack[100];
    int top = 2;
    stack[0] = pts[0];
    stack[1] = pts[1];
    stack[2] = pts[2];

    for (int i = 3; i < n; i++) {
        while (orientation(stack[top - 1], stack[top], pts[i]) != 2)
            top--;
        stack[++top] = pts[i];
    }

    printf("Convex Hull:\n");
    for (int i = 0; i <= top; i++)
        printf("(%.1f, %.1f)\n", stack[i].x, stack[i].y);
}

int main() {
    Point pts[] = {{0,0},{2,1},{1,2},{3,3},{0,3},{3,0}};
    int n = sizeof(pts)/sizeof(pts[0]);
    grahamScan(pts, n);
}
```

Python

```python
def orientation(a, b, c):
    val = (b[1]-a[1])*(c[0]-b[0]) - (b[0]-a[0])*(c[1]-b[1])
    if val == 0: return 0
    return 1 if val > 0 else 2

def graham_scan(points):
    n = len(points)
    anchor = min(points, key=lambda p: (p[1], p[0]))
    sorted_pts = sorted(points, key=lambda p: (
        atan2(p[1]-anchor[1], p[0]-anchor[0]), (p[0]-anchor[0])2 + (p[1]-anchor[1])2
    ))

    hull = []
    for p in sorted_pts:
        while len(hull) >= 2 and orientation(hull[-2], hull[-1], p) != 2:
            hull.pop()
        hull.append(p)
    return hull

from math import atan2
pts = [(0,0),(2,1),(1,2),(3,3),(0,3),(3,0)]
print("Convex Hull:", graham_scan(pts))
```

#### Why It Matters

* Efficient: O(n log n) from sorting; scanning is linear.
* Robust: Handles collinearity with tie-breaking.
* Canonical: Foundational convex hull algorithm in computational geometry.

Applications:

* Graphics: convex outlines, mesh simplification
* Collision detection and physics
* GIS boundary analysis
* Clustering hulls and convex enclosures

#### Try It Yourself

1. Plot 10 random points, sort them by angle.
2. Trace turns manually to see the hull shape.
3. Add collinear points, test tie-breaking.
4. Compare with Jarvis March for same data.
5. Measure performance as n grows.

#### Test Cases

| Points                    | Hull           | Note             |
| ------------------------- | -------------- | ---------------- |
| Square corners            | All 4          | Classic hull     |
| Triangle + interior point | 3 outer points | Interior ignored |
| Collinear points          | Endpoints only | Correct          |
| Random scatter            | Outer ring     | Verified shape   |

#### Complexity

* Time: O(n log n)
* Space: O(n) for sorting + stack

Graham Scan blends geometry and order, sort the stars, follow the turns, and the hull emerges clean and sharp.

### 703 Andrew's Monotone Chain

Andrew's Monotone Chain is a clean, efficient convex hull algorithm that's both easy to implement and fast in practice. It's essentially a simplified variant of Graham Scan, but instead of sorting by angle, it sorts by x-coordinate and constructs the hull in two sweeps, one for the lower hull, one for the upper.

Think of it as building a fence twice, once along the bottom, then along the top, and joining them together into a complete boundary.

#### What Problem Are We Solving?

Given n points, find their convex hull, the smallest convex polygon enclosing them.

Andrew's algorithm provides:

* Deterministic sorting by x (and y)
* A simple loop-based build (no angle math)
* An O(n log n) solution, matching Graham Scan

It's widely used for simplicity and numerical stability.

#### How Does It Work (Plain Language)?

1. Sort all points lexicographically by x, then y.
2. Build lower hull:

   * Traverse points left to right.
   * While the last two points + new one make a non-left turn, pop the last.
   * Push new point.
3. Build upper hull:

   * Traverse points right to left.
   * Repeat the same popping rule.
4. Concatenate lower + upper hulls, excluding duplicate endpoints.

You end up with the full convex hull in counterclockwise order.

#### Example Walkthrough

Points:
A(0,0), B(2,1), C(1,2), D(3,3), E(0,3), F(3,0)

1. Sort by x → A(0,0), E(0,3), C(1,2), B(2,1), D(3,3), F(3,0)

2. Lower hull

   * Start A(0,0), E(0,3) → right turn → pop E
   * Add C(1,2), B(2,1), F(3,0) → keep left turns only
     → Lower hull: [A, B, F]

3. Upper hull

   * Start F(3,0), D(3,3), E(0,3), A(0,0) → maintain left turns
     → Upper hull: [F, D, E, A]

4. Combine (remove duplicates):
   Hull: [A, B, F, D, E]

#### Tiny Code (Easy Version)

C

```c
#include <stdio.h>
#include <stdlib.h>

typedef struct { double x, y; } Point;

int cmp(const void *a, const void *b) {
    Point p = *(Point*)a, q = *(Point*)b;
    if (p.x == q.x) return (p.y > q.y) - (p.y < q.y);
    return (p.x > q.x) - (p.x < q.x);
}

double cross(Point o, Point a, Point b) {
    return (a.x - o.x)*(b.y - o.y) - (a.y - o.y)*(b.x - o.x);
}

void monotoneChain(Point pts[], int n) {
    qsort(pts, n, sizeof(Point), cmp);

    Point hull[200];
    int k = 0;

    // Build lower hull
    for (int i = 0; i < n; i++) {
        while (k >= 2 && cross(hull[k-2], hull[k-1], pts[i]) <= 0)
            k--;
        hull[k++] = pts[i];
    }

    // Build upper hull
    for (int i = n-2, t = k+1; i >= 0; i--) {
        while (k >= t && cross(hull[k-2], hull[k-1], pts[i]) <= 0)
            k--;
        hull[k++] = pts[i];
    }

    k--; // last point is same as first

    printf("Convex Hull:\n");
    for (int i = 0; i < k; i++)
        printf("(%.1f, %.1f)\n", hull[i].x, hull[i].y);
}

int main() {
    Point pts[] = {{0,0},{2,1},{1,2},{3,3},{0,3},{3,0}};
    int n = sizeof(pts)/sizeof(pts[0]);
    monotoneChain(pts, n);
}
```

Python

```python
def cross(o, a, b):
    return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])

def monotone_chain(points):
    points = sorted(points)
    lower = []
    for p in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper = []
    for p in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    return lower[:-1] + upper[:-1]

pts = [(0,0),(2,1),(1,2),(3,3),(0,3),(3,0)]
print("Convex Hull:", monotone_chain(pts))
```

#### Why It Matters

* Simpler than Graham Scan, no polar sorting needed
* Stable and robust against collinear points
* Commonly used in practice due to clean implementation
* Good starting point for 2D computational geometry

Applications:

* 2D collision detection
* Convex envelopes in graphics
* Bounding regions in mapping
* Hull preprocessing for advanced geometry (Voronoi, Delaunay)

#### Try It Yourself

1. Sort by x and draw points by hand.
2. Step through both passes (lower, upper).
3. Visualize popping during non-left turns.
4. Add collinear points, verify handling.
5. Compare hulls with Graham Scan and Jarvis March.

#### Test Cases

| Points                  | Hull                | Notes             |
| ----------------------- | ------------------- | ----------------- |
| Square (4 corners)      | 4 corners           | Classic rectangle |
| Triangle + center point | Outer 3 only        | Center ignored    |
| Collinear points        | 2 endpoints         | Handled           |
| Random scatter          | Correct convex ring | Stable            |

#### Complexity

* Time: O(n log n) (sorting dominates)
* Space: O(n)

Andrew's Monotone Chain is geometry at its cleanest, sort, sweep, stitch, a simple loop carves the perfect boundary.

### 704 Chan's Algorithm

Chan's Algorithm is a clever output-sensitive convex hull algorithm, meaning its running time depends not just on the total number of points *n*, but also on the number of points *h* that actually form the hull. It smartly combines Graham Scan and Jarvis March to get the best of both worlds.

Think of it like organizing a big crowd by grouping them, tracing each group's boundary, and then merging those outer lines into one smooth hull.

#### What Problem Are We Solving?

We want to find the convex hull of a set of *n* points, but we don't want to pay the full cost of sorting all of them if only a few are on the hull.

Chan's algorithm solves this with:

* Subproblem decomposition (divide into chunks)
* Fast local hulls (via Graham Scan)
* Efficient merging (via wrapping)

Result:
O(n log h) time, faster when *h* is small.

#### How Does It Work (Plain Language)?

Chan's algorithm works in three main steps:

| Step | Description                                                                                                                          |
| ---- | ------------------------------------------------------------------------------------------------------------------------------------ |
| 1    | Partition points into groups of size *m*.                                                                                        |
| 2    | For each group, compute local convex hull (using Graham Scan).                                                                   |
| 3    | Use Gift Wrapping (Jarvis March) across all hulls to find the global one, but limit the number of hull vertices explored to *m*. |

If it fails (h > m), double m and repeat.

This "guess and check" approach ensures you find the full hull in *O(n log h)* time.

#### Example Walkthrough

Imagine 30 points, but only 6 form the hull.

1. Choose *m = 4*, so you have about 8 groups.
2. Compute hull for each group with Graham Scan (fast).
3. Combine by wrapping around, at each step, pick the next tangent across all hulls.
4. If more than *m* steps are needed, double *m* → *m = 8*, repeat.
5. When all hull vertices are found, stop.

Result: Global convex hull with minimal extra work.

#### Tiny Code (Conceptual Pseudocode)

This algorithm is intricate, but here's a simple conceptual version:

```python
def chans_algorithm(points):
    import math
    n = len(points)
    m = 1
    while True:
        m = min(2*m, n)
        groups = [points[i:i+m] for i in range(0, n, m)]

        # Step 1: compute local hulls
        local_hulls = [graham_scan(g) for g in groups]

        # Step 2: merge using wrapping
        hull = []
        start = min(points)
        p = start
        for k in range(m):
            hull.append(p)
            q = None
            for H in local_hulls:
                # choose tangent point on each local hull
                cand = tangent_from_point(p, H)
                if q is None or orientation(p, q, cand) == 2:
                    q = cand
            p = q
            if p == start:
                return hull
```

Key idea: combine small hulls efficiently without reprocessing all points each time.

#### Why It Matters

* Output-sensitive: best performance when hull size is small.
* Bridges theory and practice, shows how combining algorithms can reduce asymptotic cost.
* Demonstrates divide and conquer + wrapping synergy.
* Important theoretical foundation for higher-dimensional hulls.

Applications:

* Geometric computing frameworks
* Robotics path envelopes
* Computational geometry libraries
* Performance-critical mapping or collision systems

#### Try It Yourself

1. Try with small *h* (few hull points) and large *n*, note faster performance.
2. Compare running time with Graham Scan.
3. Visualize groups and their local hulls.
4. Track doubling of *m* per iteration.
5. Measure performance growth as hull grows.

#### Test Cases

| Points                       | Hull                | Notes               |
| ---------------------------- | ------------------- | ------------------- |
| 6-point convex set           | All points          | Single iteration    |
| Dense cluster + few outliers | Outer boundary only | Output-sensitive    |
| Random 2D                    | Correct hull        | Matches Graham Scan |
| 1,000 points, 10 hull        | O(n log 10)         | Very fast           |

#### Complexity

* Time: O(n log h)
* Space: O(n)
* Best for: Small hull size relative to total points

Chan's Algorithm is geometry's quiet optimizer, it guesses, tests, and doubles back, wrapping the world one layer at a time.

### 705 QuickHull

QuickHull is a divide-and-conquer algorithm for finding the convex hull, conceptually similar to QuickSort, but in geometry. It recursively splits the set of points into smaller groups, finding extreme points and building the hull piece by piece.

Imagine you're stretching a rubber band around nails: pick the farthest nails, draw a line, and split the rest into those above and below that line. Repeat until every segment is "tight."

#### What Problem Are We Solving?

Given n points, we want to construct the convex hull, the smallest convex polygon containing all points.

QuickHull achieves this by:

* Choosing extreme points as anchors
* Partitioning the set into subproblems
* Recursively finding farthest points forming hull edges

It's intuitive and often fast on average, though can degrade to *O(n²)* in worst cases (e.g. all points on the hull).

#### How Does It Work (Plain Language)?

1. Find leftmost and rightmost points (A and B). These form a baseline of the hull.
2. Split points into two groups:

   * Above line AB
   * Below line AB
3. For each side:

   * Find the point C farthest from AB.
   * This forms a triangle ABC.
   * Any points inside triangle ABC are not on the hull.
   * Recur on the outer subsets (A–C and C–B).
4. Combine the recursive hulls from both sides.

Each recursive step adds one vertex, the farthest point, building the hull piece by piece.

#### Example Walkthrough

Points:
A(0,0), B(4,0), C(2,3), D(1,1), E(3,1)

1. Leftmost = A(0,0), Rightmost = B(4,0)
2. Points above AB = {C}, below AB = {}
3. Farthest from AB (above) = C(2,3)
   → Hull edge: A–C–B
4. No points left below AB → done

Hull = [A(0,0), C(2,3), B(4,0)]

#### Tiny Code (Easy Version)

C

```c
#include <stdio.h>
#include <math.h>

typedef struct { double x, y; } Point;

double cross(Point a, Point b, Point c) {
    return (b.x - a.x)*(c.y - a.y) - (b.y - a.y)*(c.x - a.x);
}

double distance(Point a, Point b, Point c) {
    return fabs(cross(a, b, c));
}

void quickHullRec(Point pts[], int n, Point a, Point b, int side) {
    int idx = -1;
    double maxDist = 0;

    for (int i = 0; i < n; i++) {
        double val = cross(a, b, pts[i]);
        if ((side * val) > 0 && fabs(val) > maxDist) {
            idx = i;
            maxDist = fabs(val);
        }
    }

    if (idx == -1) {
        printf("(%.1f, %.1f)\n", a.x, a.y);
        printf("(%.1f, %.1f)\n", b.x, b.y);
        return;
    }

    quickHullRec(pts, n, a, pts[idx], -cross(a, pts[idx], b) < 0 ? 1 : -1);
    quickHullRec(pts, n, pts[idx], b, -cross(pts[idx], b, a) < 0 ? 1 : -1);
}

void quickHull(Point pts[], int n) {
    int min = 0, max = 0;
    for (int i = 1; i < n; i++) {
        if (pts[i].x < pts[min].x) min = i;
        if (pts[i].x > pts[max].x) max = i;
    }
    Point A = pts[min], B = pts[max];
    quickHullRec(pts, n, A, B, 1);
    quickHullRec(pts, n, A, B, -1);
}

int main() {
    Point pts[] = {{0,0},{4,0},{2,3},{1,1},{3,1}};
    int n = sizeof(pts)/sizeof(pts[0]);
    printf("Convex Hull:\n");
    quickHull(pts, n);
}
```

Python

```python
def cross(a, b, c):
    return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])

def distance(a, b, c):
    return abs(cross(a, b, c))

def quickhull_rec(points, a, b, side):
    idx, max_dist = -1, 0
    for i, p in enumerate(points):
        val = cross(a, b, p)
        if side * val > 0 and abs(val) > max_dist:
            idx, max_dist = i, abs(val)
    if idx == -1:
        return [a, b]
    c = points[idx]
    return (quickhull_rec(points, a, c, -1 if cross(a, c, b) > 0 else 1) +
            quickhull_rec(points, c, b, -1 if cross(c, b, a) > 0 else 1))

def quickhull(points):
    points = sorted(points)
    a, b = points[0], points[-1]
    return list({*quickhull_rec(points, a, b, 1), *quickhull_rec(points, a, b, -1)})

pts = [(0,0),(4,0),(2,3),(1,1),(3,1)]
print("Convex Hull:", quickhull(pts))
```

#### Why It Matters

* Elegant and recursive, conceptually simple.
* Good average-case performance for random points.
* Divide-and-conquer design teaches geometric recursion.
* Intuitive visualization for teaching convex hulls.

Applications:

* Geometric modeling
* Game development (collision envelopes)
* Path planning and mesh simplification
* Visualization tools for spatial datasets

#### Try It Yourself

1. Plot random points and walk through recursive splits.
2. Add collinear points and see how they're handled.
3. Compare step count to Graham Scan.
4. Time on sparse vs dense hulls.
5. Trace recursive tree visually, each node is a hull edge.

#### Test Cases

| Points                  | Hull           | Notes                   |
| ----------------------- | -------------- | ----------------------- |
| Triangle                | 3 points       | Simple hull             |
| Square corners + center | 4 corners      | Center ignored          |
| Random scatter          | Outer ring     | Matches others          |
| All collinear           | Endpoints only | Handles degenerate case |

#### Complexity

* Average: O(n log n)
* Worst: O(n²)
* Space: O(n) (recursion stack)

QuickHull is the geometric sibling of QuickSort, split, recurse, and join the pieces into a clean convex boundary.

### 706 Incremental Convex Hull

The Incremental Convex Hull algorithm builds the hull step by step, starting from a small convex set (like a triangle) and inserting points one at a time, updating the hull dynamically as each point is added.

It's like growing a soap bubble around points: each new point either floats inside (ignored) or pushes out the bubble wall (updates the hull).

#### What Problem Are We Solving?

Given n points, we want to construct their convex hull.

Instead of sorting or splitting (as in Graham or QuickHull), the incremental method:

* Builds an initial hull from a few points
* Adds each remaining point
* Updates the hull edges when new points extend the boundary

This pattern generalizes nicely to higher dimensions, making it foundational for 3D hulls and computational geometry libraries.

#### How Does It Work (Plain Language)?

1. Start with a small hull (e.g. first 3 non-collinear points).
2. For each new point P:

   * Check if P is inside the current hull.
   * If not:

     * Find all visible edges (edges facing P).
     * Remove those edges from the hull.
     * Connect P to the boundary of the visible region.
3. Continue until all points are processed.

The hull grows incrementally, always staying convex.

#### Example Walkthrough

Points:
A(0,0), B(4,0), C(2,3), D(1,1), E(3,2)

1. Start hull with {A, B, C}.
2. Add D(1,1): lies inside hull → ignore.
3. Add E(3,2): lies on boundary or inside → ignore.

Hull remains [A, B, C].

If you added F(5,1):

* F lies outside, so update hull to include it → [A, B, F, C]

#### Tiny Code (Easy Version)

C (Conceptual)

```c
#include <stdio.h>
#include <stdlib.h>

typedef struct { double x, y; } Point;

double cross(Point a, Point b, Point c) {
    return (b.x - a.x)*(c.y - a.y) - (b.y - a.y)*(c.x - a.x);
}

// Simplified incremental hull for 2D (no edge pruning)
void incrementalHull(Point pts[], int n) {
    // Start with first 3 points forming a triangle
    Point hull[100];
    int h = 3;
    for (int i = 0; i < 3; i++) hull[i] = pts[i];

    for (int i = 3; i < n; i++) {
        Point p = pts[i];
        int visible[100], count = 0;

        // Mark edges visible from p
        for (int j = 0; j < h; j++) {
            Point a = hull[j];
            Point b = hull[(j+1)%h];
            if (cross(a, b, p) > 0) visible[count++] = j;
        }

        // If none visible, point is inside
        if (count == 0) continue;

        // Remove visible edges and insert new connections (simplified)
        // Here: we just print added point for demo
        printf("Adding point (%.1f, %.1f) to hull\n", p.x, p.y);
    }

    printf("Final hull (approx):\n");
    for (int i = 0; i < h; i++)
        printf("(%.1f, %.1f)\n", hull[i].x, hull[i].y);
}

int main() {
    Point pts[] = {{0,0},{4,0},{2,3},{1,1},{3,2},{5,1}};
    int n = sizeof(pts)/sizeof(pts[0]);
    incrementalHull(pts, n);
}
```

Python (Simplified)

```python
def cross(a, b, c):
    return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])

def is_inside(hull, p):
    for i in range(len(hull)):
        a, b = hull[i], hull[(i+1)%len(hull)]
        if cross(a, b, p) > 0:
            return False
    return True

def incremental_hull(points):
    hull = points[:3]
    for p in points[3:]:
        if not is_inside(hull, p):
            hull.append(p)
            # In practice, re-sort hull in CCW order
            hull = sorted(hull, key=lambda q: (q[0], q[1]))
    return hull

pts = [(0,0),(4,0),(2,3),(1,1),(3,2),(5,1)]
print("Convex Hull:", incremental_hull(pts))
```

#### Why It Matters

* Conceptually simple, easy to extend to 3D and higher.
* Online: can update hull dynamically as points stream in.
* Used in real-time simulations, collision detection, and geometry libraries.
* Foundation for dynamic hull maintenance (next section).

Applications:

* Incremental geometry algorithms
* Data streams and real-time convexity checks
* Building Delaunay or Voronoi structures incrementally

#### Try It Yourself

1. Add points one by one, draw hull at each step.
2. Observe how interior points don't change the hull.
3. Try random insertion orders, hull stays consistent.
4. Compare with Graham Scan's static approach.
5. Extend to 3D using visible-face detection.

#### Test Cases

| Points                   | Hull            | Notes          |
| ------------------------ | --------------- | -------------- |
| Triangle + inside points | Outer 3         | Inside ignored |
| Square + center point    | Corners only    | Works          |
| Random points            | Outer ring      | Verified       |
| Incremental additions    | Correct updates | Dynamic hull   |

#### Complexity

* Time: O(n²) naive, O(n log n) with optimization
* Space: O(h)

The incremental method teaches geometry's patience, one point at a time, reshaping the boundary as the world grows.

### 707 Divide & Conquer Hull

The Divide & Conquer Hull algorithm builds the convex hull by splitting the set of points into halves, recursively computing hulls for each half, and then merging them, much like Merge Sort, but for geometry.

Imagine cutting your set of points into two clouds, wrapping each cloud separately, then stitching the two wraps into one smooth boundary.

#### What Problem Are We Solving?

Given n points on a plane, we want to construct their convex hull.

The divide and conquer approach provides:

* A clean O(n log n) runtime
* Elegant structure (recursion + merge)
* Strong foundation for higher-dimensional hulls

It's a canonical example of applying divide and conquer to geometric data.

#### How Does It Work (Plain Language)?

1. Sort all points by x-coordinate.
2. Divide the points into two halves.
3. Recursively compute the convex hull for each half.
4. Merge the two hulls:

   * Find upper tangent: the line touching both hulls from above
   * Find lower tangent: the line touching both from below
   * Remove interior points between tangents
   * Join remaining points to form the merged hull

This process repeats until all points are enclosed in one convex boundary.

#### Example Walkthrough

Points:
A(0,0), B(4,0), C(2,3), D(1,1), E(3,2)

1. Sort by x: [A, D, C, E, B]
2. Divide: Left = [A, D, C], Right = [E, B]
3. Hull(Left) = [A, C]
   Hull(Right) = [E, B]
4. Merge:

   * Find upper tangent → connects C and E
   * Find lower tangent → connects A and B
     Hull = [A, B, E, C]

#### Tiny Code (Conceptual Pseudocode)

To illustrate the logic (omitting low-level tangent-finding details):

```python
def divide_conquer_hull(points):
    n = len(points)
    if n <= 3:
        # Base: simple convex polygon
        return sorted(points)
    
    mid = n // 2
    left = divide_conquer_hull(points[:mid])
    right = divide_conquer_hull(points[mid:])
    return merge_hulls(left, right)

def merge_hulls(left, right):
    # Find upper and lower tangents
    upper = find_upper_tangent(left, right)
    lower = find_lower_tangent(left, right)
    # Combine points between tangents
    hull = []
    i = left.index(upper[0])
    while left[i] != lower[0]:
        hull.append(left[i])
        i = (i + 1) % len(left)
    hull.append(lower[0])
    j = right.index(lower[1])
    while right[j] != upper[1]:
        hull.append(right[j])
        j = (j + 1) % len(right)
    hull.append(upper[1])
    return hull
```

In practice, tangent-finding uses orientation tests and cyclic traversal.

#### Why It Matters

* Elegant recursion: geometry meets algorithm design.
* Balanced performance: deterministic O(n log n).
* Ideal for batch processing or parallel implementations.
* Extends well to 3D convex hulls (divide in planes).

Applications:

* Computational geometry toolkits
* Spatial analysis and map merging
* Parallel geometry processing
* Geometry-based clustering

#### Try It Yourself

1. Draw 10 points, split by x-midpoint.
2. Build hulls for left and right manually.
3. Find upper/lower tangents and merge.
4. Compare result to Graham Scan.
5. Trace recursion tree (like merge sort).

#### Test Cases

| Points           | Hull           | Notes            |
| ---------------- | -------------- | ---------------- |
| Triangle         | 3 points       | Simple base case |
| Square           | All corners    | Perfect merge    |
| Random scatter   | Outer boundary | Verified         |
| Collinear points | Endpoints only | Correct          |

#### Complexity

* Time: O(n log n)
* Space: O(n)
* Best Case: Balanced splits → efficient merges

Divide & Conquer Hull is geometric harmony, each half finds its shape, and together they trace the perfect outline of all points.

### 708 3D Convex Hull

The 3D Convex Hull is the natural extension of the planar hull into space. Instead of connecting points into a polygon, you connect them into a polyhedron, a 3D envelope enclosing all given points.

Think of it as wrapping a shrink film around scattered pebbles in 3D space, it tightens into a surface formed by triangular faces.

#### What Problem Are We Solving?

Given n points in 3D, find the convex polyhedron (set of triangular faces) that completely encloses them.

We want to compute:

* Vertices (points on the hull)
* Edges (lines between them)
* Faces (planar facets forming the surface)

The goal:
A minimal set of faces such that every point lies inside or on the hull.

#### How Does It Work (Plain Language)?

Several algorithms extend from 2D to 3D, but one classic approach is the Incremental 3D Hull:

| Step | Description                                                                   |
| ---- | ----------------------------------------------------------------------------- |
| 1    | Start with a non-degenerate tetrahedron (4 points not on the same plane). |
| 2    | For each remaining point P:                                                   |
|      | – Identify visible faces (faces where P is outside).                      |
|      | – Remove those faces (forming a "hole").                                      |
|      | – Create new faces connecting P to the boundary of the hole.              |
| 3    | Continue until all points are processed.                                      |
| 4    | The remaining faces define the 3D convex hull.                            |

Each insertion either adds new faces or lies inside and is ignored.

#### Example Walkthrough

Points:
A(0,0,0), B(1,0,0), C(0,1,0), D(0,0,1), E(1,1,1)

1. Start with base tetrahedron: A, B, C, D
2. Add E(1,1,1):

   * Find faces visible from E
   * Remove them
   * Connect E to boundary edges of the visible region
3. New hull has 5 vertices, forming a convex polyhedron.

#### Tiny Code (Conceptual Pseudocode)

A high-level idea, practical versions use complex data structures (face adjacency, conflict graph):

```python
def incremental_3d_hull(points):
    hull = initialize_tetrahedron(points)
    for p in points:
        if point_inside_hull(hull, p):
            continue
        visible_faces = [f for f in hull if face_visible(f, p)]
        hole_edges = find_boundary_edges(visible_faces)
        hull = [f for f in hull if f not in visible_faces]
        for e in hole_edges:
            hull.append(make_face(e, p))
    return hull
```

Each face is represented by a triple of points (a, b, c), with orientation tests via determinants or triple products.

#### Why It Matters

* Foundation for 3D geometry, meshes, solids, and physics.
* Used in computational geometry, graphics, CAD, physics engines.
* Forms building blocks for:

  * Delaunay Triangulation (3D)
  * Voronoi Diagrams (3D)
  * Convex decomposition and collision detection

Applications:

* 3D modeling and rendering
* Convex decomposition (physics engines)
* Spatial analysis, convex enclosures
* Game geometry, mesh simplification

#### Try It Yourself

1. Start with 4 non-coplanar points, visualize the tetrahedron.
2. Add one point outside and sketch new faces.
3. Add a point inside, confirm no hull change.
4. Compare 3D hulls for cube corners, random points, sphere samples.
5. Use a geometry viewer to visualize updates step-by-step.

#### Test Cases

| Points                  | Hull Output | Notes            |
| ----------------------- | ----------- | ---------------- |
| 4 non-coplanar          | Tetrahedron | Base case        |
| Cube corners            | 8 vertices  | Classic box hull |
| Random points on sphere | All points  | Convex set       |
| Random interior points  | Only outer  | Inner ignored    |

#### Complexity

* Time: O(n log n) average, O(n²) worst-case
* Space: O(n)

The 3D Convex Hull lifts geometry into space, from wrapping a string to wrapping a surface, it turns scattered points into shape.

### 709 Dynamic Convex Hull

A Dynamic Convex Hull is a data structure (and algorithm family) that maintains the convex hull as points are inserted (and sometimes deleted), without recomputing the entire hull from scratch.

Think of it like a living rubber band that flexes and tightens as you add or remove pegs, always adjusting itself to stay convex.

#### What Problem Are We Solving?

Given a sequence of updates (insertions or deletions of points), we want to maintain the current convex hull efficiently, so that:

* Insert(point) adjusts the hull in sublinear time.
* Query() returns the hull or answers questions (area, diameter, point location).
* Delete(point) (optional) removes a point and repairs the hull.

A dynamic hull is crucial when data evolves, streaming points, moving agents, or incremental datasets.

#### How Does It Work (Plain Language)?

Several strategies exist depending on whether we need full dynamism (inserts + deletes) or semi-dynamic (inserts only):

| Variant                   | Idea                                         | Complexity                    |
| ------------------------- | -------------------------------------------- | ----------------------------- |
| Semi-Dynamic          | Only insertions, maintain hull incrementally | O(log n) amortized per insert |
| Fully Dynamic         | Both insertions and deletions                | O(log² n) per update          |
| Online Hull (1D / 2D) | Maintain upper & lower chains separately     | Logarithmic updates           |

Common structure:

1. Split hull into upper and lower chains.
2. Store each chain in a balanced BST or ordered set.
3. On insert:

   * Locate insertion position by x-coordinate.
   * Check for turn direction (orientation tests).
   * Remove interior points (not convex) and add new vertex.
4. On delete:

   * Remove vertex, re-link neighbors, recheck convexity.

#### Example Walkthrough (Semi-Dynamic)

Start with empty hull.
Insert points one by one:

1. Add A(0,0) → hull = [A]
2. Add B(2,0) → hull = [A, B]
3. Add C(1,2) → hull = [A, B, C]
4. Add D(3,1):

   * Upper hull = [A, C, D]
   * Lower hull = [A, B, D]
     Hull updates dynamically without recomputing all points.

If D lies inside, skip it.
If D extends hull, remove covered edges and reinsert.

#### Tiny Code (Python Sketch)

A simple incremental hull using sorted chains:

```python
def cross(o, a, b):
    return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])

class DynamicHull:
    def __init__(self):
        self.upper = []
        self.lower = []

    def insert(self, p):
        self._insert_chain(self.upper, p, 1)
        self._insert_chain(self.lower, p, -1)

    def _insert_chain(self, chain, p, sign):
        chain.append(p)
        chain.sort()  # maintain order by x
        while len(chain) >= 3 and sign * cross(chain[-3], chain[-2], chain[-1]) <= 0:
            del chain[-2]

    def get_hull(self):
        return self.lower[:-1] + self.upper[::-1][:-1]

# Example
dh = DynamicHull()
for p in [(0,0),(2,0),(1,2),(3,1)]:
    dh.insert(p)
print("Hull:", dh.get_hull())
```

#### Why It Matters

* Real-time geometry: used in moving point sets, games, robotics.
* Streaming analytics: convex envelopes of live data.
* Incremental algorithms: maintain convexity without full rebuild.
* Data structures research: connects geometry to balanced trees.

Applications:

* Collision detection (objects moving step-by-step)
* Real-time visualization
* Geometric median or bounding region updates
* Computational geometry libraries (CGAL, Boost.Geometry)

#### Try It Yourself

1. Insert points one by one, sketch hull after each.
2. Try inserting an interior point (no hull change).
3. Insert a point outside, watch edges removed and added.
4. Extend code to handle deletions.
5. Compare with Incremental Hull (static order).

#### Test Cases

| Operation             | Result             | Notes                 |
| --------------------- | ------------------ | --------------------- |
| Insert outer points   | Expanding hull     | Expected growth       |
| Insert interior point | No change          | Stable                |
| Insert collinear      | Adds endpoint      | Interior ignored      |
| Delete hull vertex    | Reconnect boundary | Fully dynamic variant |

#### Complexity

* Semi-Dynamic (insert-only): O(log n) amortized per insert
* Fully Dynamic: O(log² n) per update
* Query (return hull): O(h)

The dynamic convex hull is a shape that grows with time, a memory of extremes, always ready for the next point to bend its boundary.

### 710 Rotating Calipers

The Rotating Calipers technique is a geometric powerhouse, a way to systematically explore pairs of points, edges, or directions on a convex polygon by "rotating" a set of imaginary calipers around its boundary.

It's like placing a pair of measuring arms around the convex hull, rotating them in sync, and recording distances, widths, or diameters at every step.

#### What Problem Are We Solving?

Once you have a convex hull, many geometric quantities can be computed efficiently using rotating calipers:

* Farthest pair (diameter)
* Minimum width / bounding box
* Closest pair of parallel edges
* Antipodal point pairs
* Polygon area and width in given direction

It transforms geometric scanning into an O(n) walk, no nested loops needed.

#### How Does It Work (Plain Language)?

1. Start with a convex polygon (points ordered CCW).
2. Imagine a caliper, a line touching one vertex, with another parallel line touching the opposite edge.
3. Rotate these calipers around the hull:

   * At each step, advance the side whose next edge causes the smaller rotation.
   * Measure whatever quantity you need (distance, area, width).
4. Stop when calipers make a full rotation.

Every "event" (vertex alignment) corresponds to an antipodal pair, useful for finding extremal distances.

#### Example Walkthrough: Farthest Pair (Diameter)

Hull: A(0,0), B(4,0), C(4,3), D(0,3)

1. Start with edge AB and find point farthest from AB (D).
2. Rotate calipers to next edge (BC), advance opposite point as needed.
3. Continue rotating until full sweep.
4. Track max distance found → here: between A(0,0) and C(4,3)

Result: Diameter = 5

#### Tiny Code (Python)

Farthest pair (diameter) using rotating calipers on a convex hull:

```python
from math import dist

def rotating_calipers(hull):
    n = len(hull)
    if n == 1:
        return (hull[0], hull[0], 0)
    if n == 2:
        return (hull[0], hull[1], dist(hull[0], hull[1]))

    def area2(a, b, c):
        return abs((b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0]))

    max_d = 0
    best_pair = (hull[0], hull[0])
    j = 1
    for i in range(n):
        ni = (i + 1) % n
        while area2(hull[i], hull[ni], hull[(j+1)%n]) > area2(hull[i], hull[ni], hull[j]):
            j = (j + 1) % n
        d = dist(hull[i], hull[j])
        if d > max_d:
            max_d = d
            best_pair = (hull[i], hull[j])
    return best_pair + (max_d,)

# Example hull (square)
hull = [(0,0),(4,0),(4,3),(0,3)]
a, b, d = rotating_calipers(hull)
print(f"Farthest pair: {a}, {b}, distance={d:.2f}")
```

#### Why It Matters

* Elegant O(n) solutions for many geometric problems
* Turns geometric search into synchronized sweeps
* Used widely in computational geometry, graphics, and robotics
* Core step in bounding box, minimum width, and collision algorithms

Applications:

* Shape analysis (diameter, width, bounding box)
* Collision detection (support functions in physics engines)
* Robotics (clearance computation)
* GIS and mapping (directional hull properties)

#### Try It Yourself

1. Draw a convex polygon.
2. Place a pair of parallel lines tangent to two opposite edges.
3. Rotate them and record farthest point pairs.
4. Compare with brute force O(n²) distance check.
5. Extend to compute minimum-area bounding box.

#### Test Cases

| Hull              | Result             | Notes        |
| ----------------- | ------------------ | ------------ |
| Square 4×3        | A(0,0)-C(4,3)      | Diagonal = 5 |
| Triangle          | Longest edge       | Works        |
| Regular hexagon   | Opposite vertices  | Symmetric    |
| Irregular polygon | Antipodal max pair | Verified     |

#### Complexity

* Time: O(n) (linear scan around hull)
* Space: O(1)

Rotating Calipers is geometry's precision instrument, smooth, synchronized, and exact, it measures the world by turning gently around its edges.

# Section 72. Closest Pair and Segment Algorithms 

### 711 Closest Pair (Divide & Conquer)

The Closest Pair (Divide & Conquer) algorithm finds the two points in a set that are closest together, faster than brute force. It cleverly combines sorting, recursion, and geometric insight to achieve O(n log n) time.

Think of it as zooming in on pairs step by step: split the plane, solve each side, then check only the narrow strip where cross-boundary pairs might hide.

#### What Problem Are We Solving?

Given n points in the plane, find the pair (p, q) with the smallest Euclidean distance:

$$
d(p, q) = \sqrt{(p_x - q_x)^2 + (p_y - q_y)^2}
$$

A naive solution checks all pairs (O(n²)), but divide-and-conquer reduces the work by cutting the problem in half and only merging near-boundary candidates.

#### How Does It Work (Plain Language)?

1. Sort all points by x-coordinate.  
2. Divide the points into two halves: left and right.  
3. Recursively find the closest pair in each half → distances $d_L$ and $d_R$.  
4. Let $d = \min(d_L, d_R)$.  
5. Merge step:
   - Collect points within distance $d$ of the dividing line (a vertical strip).  
   - Sort these strip points by y.  
   - For each point, only check the next few neighbors (at most 7) in y-order.  
6. The smallest distance found in these checks is the answer.

This restriction, "check only a few nearby points," is what keeps the algorithm $O(n \log n)$.

#### Example Walkthrough

Points:  
A(0,0), B(3,4), C(1,1), D(4,5), E(2,2)

1. Sort by x → [A(0,0), C(1,1), E(2,2), B(3,4), D(4,5)]  
2. Split into Left [A, C, E] and Right [B, D].  
3. Left recursion → closest = A–C = $\sqrt{2}$  
   Right recursion → closest = B–D = $\sqrt{2}$  
   So $d = \min(\sqrt{2}, \sqrt{2}) = \sqrt{2}$.  
4. Strip near divide ($x \approx 2$) → E(2,2), B(3,4), D(4,5)  
   Check pairs:  
   - E–B = $\sqrt{5}$  
   - E–D = $\sqrt{10}$  
   No smaller distance found.  

Result: Closest Pair = (A, C), distance = $\sqrt{2}$.


#### Tiny Code (Easy Version)

C

```c
#include <stdio.h>
#include <math.h>
#include <float.h>
#include <stdlib.h>

typedef struct { double x, y; } Point;

int cmpX(const void* a, const void* b) {
    Point *p = (Point*)a, *q = (Point*)b;
    return (p->x > q->x) - (p->x < q->x);
}

int cmpY(const void* a, const void* b) {
    Point *p = (Point*)a, *q = (Point*)b;
    return (p->y > q->y) - (p->y < q->y);
}

double dist(Point a, Point b) {
    double dx = a.x - b.x, dy = a.y - b.y;
    return sqrt(dx*dx + dy*dy);
}

double brute(Point pts[], int n) {
    double min = DBL_MAX;
    for (int i=0; i<n; i++)
        for (int j=i+1; j<n; j++)
            if (dist(pts[i], pts[j]) < min)
                min = dist(pts[i], pts[j]);
    return min;
}

double stripClosest(Point strip[], int size, double d) {
    double min = d;
    qsort(strip, size, sizeof(Point), cmpY);
    for (int i=0; i<size; i++)
        for (int j=i+1; j<size && (strip[j].y - strip[i].y) < min; j++)
            if (dist(strip[i], strip[j]) < min)
                min = dist(strip[i], strip[j]);
    return min;
}

double closestRec(Point pts[], int n) {
    if (n <= 3) return brute(pts, n);
    int mid = n/2;
    Point midPoint = pts[mid];

    double dl = closestRec(pts, mid);
    double dr = closestRec(pts+mid, n-mid);
    double d = dl < dr ? dl : dr;

    Point strip[1000];
    int j=0;
    for (int i=0; i<n; i++)
        if (fabs(pts[i].x - midPoint.x) < d)
            strip[j++] = pts[i];
    return fmin(d, stripClosest(strip, j, d));
}

double closestPair(Point pts[], int n) {
    qsort(pts, n, sizeof(Point), cmpX);
    return closestRec(pts, n);
}

int main() {
    Point pts[] = {{0,0},{3,4},{1,1},{4,5},{2,2}};
    int n = sizeof(pts)/sizeof(pts[0]);
    printf("Closest distance = %.3f\n", closestPair(pts, n));
}
```

Python

```python
from math import sqrt

def dist(a,b):
    return sqrt((a[0]-b[0])2 + (a[1]-b[1])2)

def brute(pts):
    n = len(pts)
    d = float('inf')
    for i in range(n):
        for j in range(i+1,n):
            d = min(d, dist(pts[i], pts[j]))
    return d

def strip_closest(strip, d):
    strip.sort(key=lambda p: p[1])
    m = len(strip)
    for i in range(m):
        for j in range(i+1, m):
            if (strip[j][1] - strip[i][1]) >= d:
                break
            d = min(d, dist(strip[i], strip[j]))
    return d

def closest_pair(points):
    n = len(points)
    if n <= 3:
        return brute(points)
    mid = n // 2
    midx = points[mid][0]
    d = min(closest_pair(points[:mid]), closest_pair(points[mid:]))
    strip = [p for p in points if abs(p[0]-midx) < d]
    return min(d, strip_closest(strip, d))

pts = [(0,0),(3,4),(1,1),(4,5),(2,2)]
pts.sort()
print("Closest distance:", closest_pair(pts))
```

#### Why It Matters

* Classic example of divide & conquer in geometry.
* Efficient and elegant, the leap from O(n²) to O(n log n).
* Builds intuition for other planar algorithms (Delaunay, Voronoi).

Applications:

* Clustering (detect near neighbors)
* Collision detection (find minimal separation)
* Astronomy / GIS (closest stars, cities)
* Machine learning (nearest-neighbor initialization)

#### Try It Yourself

1. Try random 2D points, verify result vs brute force.
2. Add collinear points, confirm distance along line.
3. Visualize split and strip, draw dividing line and strip area.
4. Extend to 3D closest pair (check z too).
5. Measure runtime as n doubles.

#### Test Cases

| Points                        | Closest Pair | Distance   |
| ----------------------------- | ------------ | ---------- |
| (0,0),(1,1),(2,2)             | (0,0)-(1,1)  | √2         |
| (0,0),(3,4),(1,1),(4,5),(2,2) | (0,0)-(1,1)  | √2         |
| Random                        | Verified     | O(n log n) |
| Duplicate points              | Distance = 0 | Edge case  |

#### Complexity

* Time: O(n log n)
* Space: O(n)
* Brute Force: O(n²) for comparison

Divide-and-conquer finds structure in chaos, sorting, splitting, and merging until the closest pair stands alone.

### 712 Closest Pair (Sweep Line)

The Closest Pair (Sweep Line) algorithm is a beautifully efficient O(n log n) technique that scans the plane from left to right, maintaining a sliding window (or "active set") of candidate points that could form the closest pair.

Think of it as sweeping a vertical line across a field of stars, as each star appears, you check only its close neighbors, not the whole sky.

#### What Problem Are We Solving?

Given n points in 2D space, we want to find the pair with the minimum Euclidean distance.

Unlike Divide & Conquer, which splits recursively, the Sweep Line solution processes points incrementally, one at a time, maintaining an active set of points close enough in x to be possible contenders.

This approach is intuitive, iterative, and particularly nice to implement with balanced search trees or ordered sets.

#### How Does It Work (Plain Language)?

1. Sort points by x-coordinate.
2. Initialize an empty active set (sorted by y).
3. Sweep from left to right:

   * For each point p,

     * Remove points whose x-distance from p exceeds the current best distance d (they're too far left).
     * In the remaining active set, only check points whose y-distance < d.
     * Update d if a closer pair is found.
   * Insert p into the active set.
4. Continue until all points are processed.

Since each point enters and leaves the active set once, and each is compared with a constant number of nearby points, total time is O(n log n).

#### Example Walkthrough

Points:
A(0,0), B(3,4), C(1,1), D(2,2), E(4,5)

1. Sort by x → [A, C, D, B, E]
2. Start with A → active = {A}, d = ∞
3. Add C: dist(A,C) = √2 → d = √2
4. Add D: check neighbors (A,C) → C–D = √2 (no improvement)
5. Add B: remove A (B.x - A.x > √2), check C–B (dist > √2), D–B (dist = √5)
6. Add E: remove C (E.x - C.x > √2), check D–E, B–E
    Closest Pair: (A, C) with distance √2

#### Tiny Code (Easy Version)

Python

```python
from math import sqrt
import bisect

def dist(a, b):
    return sqrt((a[0]-b[0])2 + (a[1]-b[1])2)

def closest_pair_sweep(points):
    points.sort(key=lambda p: p[0])  # sort by x
    active = []
    best = float('inf')
    best_pair = None

    for p in points:
        # Remove points too far in x
        while active and p[0] - active[0][0] > best:
            active.pop(0)

        # Filter active points by y range
        candidates = [q for q in active if abs(q[1] - p[1]) < best]

        # Check each candidate
        for q in candidates:
            d = dist(p, q)
            if d < best:
                best = d
                best_pair = (p, q)

        # Insert current point (keep sorted by y)
        bisect.insort(active, p, key=lambda r: r[1] if hasattr(bisect, "insort") else 0)

    return best_pair, best

# Example
pts = [(0,0),(3,4),(1,1),(2,2),(4,5)]
pair, d = closest_pair_sweep(pts)
print("Closest pair:", pair, "distance:", round(d,3))
```

*(Note: `bisect` can't sort by key directly; in real code use `sortedcontainers` or a balanced tree.)*

C (Pseudocode)
In C, implement with:

* `qsort` by x
* Balanced BST (by y) for active set
* Window update and neighbor checks
  (Real implementations use AVL trees or ordered arrays)

#### Why It Matters

* Incremental and online: processes one point at a time.
* Conceptual simplicity, a geometric sliding window.
* Practical alternative to divide & conquer.

Applications:

* Streaming geometry
* Real-time collision detection
* Nearest-neighbor estimation
* Computational geometry visualizations

#### Try It Yourself

1. Step through manually with sorted points.
2. Track how the active set shrinks and grows.
3. Add interior points and see how many are compared.
4. Try 1,000 random points, verify fast runtime.
5. Compare with Divide & Conquer approach, same result, different path.

#### Test Cases

| Points                | Closest Pair            | Distance   | Notes       |
| --------------------- | ----------------------- | ---------- | ----------- |
| (0,0),(1,1),(2,2)     | (0,0)-(1,1)             | √2         | Simple line |
| Random scatter        | Correct pair            | O(n log n) | Efficient   |
| Clustered near origin | Finds nearest neighbors | Works      |             |
| Duplicates            | Distance 0              | Edge case  |             |

#### Complexity

* Time: O(n log n)
* Space: O(n)
* Active Set Size: O(n) (usually small window)

The Sweep Line is geometry's steady heartbeat, moving left to right, pruning the past, and focusing only on the nearby present to find the closest pair.

### 713 Brute Force Closest Pair

The Brute Force Closest Pair algorithm is the simplest way to find the closest two points in a set, you check every possible pair and pick the one with the smallest distance.

It's the geometric equivalent of "try them all," a perfect first step for understanding how smarter algorithms improve upon it.

#### What Problem Are We Solving?

Given n points on a plane, we want to find the pair (p, q) with the smallest Euclidean distance:

$$
d(p, q) = \sqrt{(p_x - q_x)^2 + (p_y - q_y)^2}
$$

Brute force means:

* Compare each pair once.
* Track the minimum distance found so far.
* Return the pair with that distance.

It's slow, O(n²), but straightforward and unbeatable in simplicity.

#### How Does It Work (Plain Language)?

1. Initialize best distance ( d = \infty ).
2. Loop over all points ( i = 1..n-1 ):

   * For each ( j = i+1..n ), compute distance ( d(i, j) ).
   * If ( d(i, j) < d ), update ( d ) and store pair.
3. Return the smallest ( d ) and its pair.

Because each pair is checked exactly once, it's easy to reason about, and perfect for small datasets or testing.

#### Example Walkthrough

Points:
A(0,0), B(3,4), C(1,1), D(2,2)

Pairs and distances:

* A–B = 5
* A–C = √2
* A–D = √8
* B–C = √13
* B–D = √5
* C–D = √2

 Minimum distance = √2 (pairs A–C and C–D)
Return first or all minimal pairs.

#### Tiny Code (Easy Version)

C

```c
#include <stdio.h>
#include <math.h>
#include <float.h>

typedef struct { double x, y; } Point;

double dist(Point a, Point b) {
    double dx = a.x - b.x, dy = a.y - b.y;
    return sqrt(dx*dx + dy*dy);
}

void closestPairBrute(Point pts[], int n) {
    double best = DBL_MAX;
    Point p1, p2;
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            double d = dist(pts[i], pts[j]);
            if (d < best) {
                best = d;
                p1 = pts[i];
                p2 = pts[j];
            }
        }
    }
    printf("Closest Pair: (%.1f, %.1f) and (%.1f, %.1f)\n", p1.x, p1.y, p2.x, p2.y);
    printf("Distance: %.3f\n", best);
}

int main() {
    Point pts[] = {{0,0},{3,4},{1,1},{2,2}};
    int n = sizeof(pts)/sizeof(pts[0]);
    closestPairBrute(pts, n);
}
```

Python

```python
from math import sqrt

def dist(a, b):
    return sqrt((a[0]-b[0])2 + (a[1]-b[1])2)

def closest_pair_brute(points):
    best = float('inf')
    pair = None
    n = len(points)
    for i in range(n):
        for j in range(i+1, n):
            d = dist(points[i], points[j])
            if d < best:
                best = d
                pair = (points[i], points[j])
    return pair, best

pts = [(0,0),(3,4),(1,1),(2,2)]
pair, d = closest_pair_brute(pts)
print("Closest pair:", pair, "distance:", round(d,3))
```

#### Why It Matters

* Foundation for understanding divide-and-conquer and sweep line improvements.
* Small n → simplest, most reliable method.
* Useful for testing optimized algorithms.
* A gentle introduction to geometric iteration and distance functions.

Applications:

* Educational baseline for geometric problems
* Verification in computational geometry toolkits
* Debugging optimized implementations
* Very small point sets (n < 100)

#### Try It Yourself

1. Add 5–10 random points, list all pair distances manually.
2. Check correctness against optimized versions.
3. Extend to 3D, just add a z term.
4. Modify for Manhattan distance.
5. Print all equally minimal pairs (ties).

#### Test Cases

| Points                  | Closest Pair | Distance          |
| ----------------------- | ------------ | ----------------- |
| (0,0),(1,1),(2,2)       | (0,0)-(1,1)  | √2                |
| (0,0),(3,4),(1,1),(2,2) | (0,0)-(1,1)  | √2                |
| Random                  | Verified     | Matches optimized |
| Duplicates              | Distance = 0 | Edge case         |

#### Complexity

* Time: O(n²)
* Space: O(1)

Brute Force is geometry's first instinct, simple, certain, and slow, but a solid foundation for all the cleverness that follows.

### 714 Bentley–Ottmann

The Bentley–Ottmann algorithm is a classical sweep line method that efficiently finds all intersection points among a set of line segments in the plane.
It runs in

$$
O\big((n + k)\log n\big)
$$

time, where $n$ is the number of segments and $k$ is the number of intersections.

The key insight is to move a vertical sweep line across the plane, maintaining an active set of intersecting segments ordered by $y$, and using an event queue to process only three types of points: segment starts, segment ends, and discovered intersections.

#### What Problem Are We Solving?

Given $n$ line segments, we want to compute all intersection points between them.

A naive approach checks all pairs:

$$
\binom{n}{2} = \frac{n(n-1)}{2}
$$

which leads to $O(n^2)$ time.
The Bentley–Ottmann algorithm reduces this to $O\big((n + k)\log n\big)$ by only testing neighboring segments in the sweep line's active set.

#### How Does It Work (Plain Language)?

We maintain two data structures during the sweep:

1. Event Queue (EQ), all x-sorted events:
   segment starts, segment ends, and discovered intersections.
2. Active Set (AS), all segments currently intersected by the sweep line, sorted by y-coordinate.

The sweep progresses from left to right:

| Step | Description                                                                                                        |
| ---- | ------------------------------------------------------------------------------------------------------------------ |
| 1    | Initialize the event queue with all segment endpoints.                                                         |
| 2    | Sweep from left to right across all events.                                                                    |
| 3    | For each event $p$:                                                                                                |
| a.   | If $p$ is a segment start, insert the segment into AS and test for intersections with its immediate neighbors. |
| b.   | If $p$ is a segment end, remove the segment from AS.                                                           |
| c.   | If $p$ is an intersection, record it, swap the two intersecting segments in AS, and check their new neighbors. |
| 4    | Continue until the event queue is empty.                                                                           |

Each operation on the event queue or active set takes $O(\log n)$ time, using balanced search trees.

#### Example Walkthrough

Segments:

* $S_1: (0,0)\text{–}(4,4)$
* $S_2: (0,4)\text{–}(4,0)$
* $S_3: (1,3)\text{–}(3,3)$

Event queue (sorted by $x$):
$(0,0), (0,4), (1,3), (2,2), (3,3), (4,0), (4,4)$

Process:

1. At $x=0$: insert $S_1, S_2$. They intersect at $(2,2)$ → schedule intersection event.
2. At $x=1$: insert $S_3$; check $S_1, S_2, S_3$ for local intersections.
3. At $x=2$: process $(2,2)$, swap $S_1, S_2$, recheck neighbors.
4. Continue; all intersections discovered.

 Output: intersection $(2,2)$.

#### Tiny Code (Conceptual Python)

A simplified sketch of the algorithm (real implementation requires a priority queue and balanced tree):

```python
from collections import namedtuple
Event = namedtuple("Event", ["x", "y", "type", "segment"])

def orientation(a, b, c):
    return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])

def intersects(s1, s2):
    a, b = s1
    c, d = s2
    o1 = orientation(a, b, c)
    o2 = orientation(a, b, d)
    o3 = orientation(c, d, a)
    o4 = orientation(c, d, b)
    return (o1 * o2 < 0) and (o3 * o4 < 0)

def bentley_ottmann(segments):
    events = []
    for s in segments:
        (x1, y1), (x2, y2) = s
        if x1 > x2:
            s = ((x2, y2), (x1, y1))
        events.append((x1, y1, 'start', s))
        events.append((x2, y2, 'end', s))
    events.sort()

    active = []
    intersections = []

    for x, y, etype, s in events:
        if etype == 'start':
            active.append(s)
            for other in active:
                if other != s and intersects(s, other):
                    intersections.append((x, y))
        elif etype == 'end':
            active.remove(s)

    return intersections

segments = [((0,0),(4,4)), ((0,4),(4,0)), ((1,3),(3,3))]
print("Intersections:", bentley_ottmann(segments))
```

#### Why It Matters

* Efficient: $O((n + k)\log n)$ vs. $O(n^2)$
* Elegant: only neighboring segments are checked
* General-purpose: fundamental for event-driven geometry

Applications:

* CAD systems (curve crossings)
* GIS (map overlays, road intersections)
* Graphics (segment collision detection)
* Robotics (motion planning, visibility graphs)

#### A Gentle Proof (Why It Works)

At any sweep position, the segments in the active set are ordered by their $y$-coordinate.
When two segments intersect, their order must swap at the intersection point.

Hence:

* Every intersection is revealed exactly once when the sweep reaches its $x$-coordinate.
* Only neighboring segments can swap; thus only local checks are needed.
* Each event (insert, delete, or intersection) requires $O(\log n)$ time for balanced tree operations.

Total cost:

$$
O\big((n + k)\log n\big)
$$

where $n$ contributes endpoints and $k$ contributes discovered intersections.

#### Try It Yourself

1. Draw several segments that intersect at various points.
2. Sort all endpoints by $x$-coordinate.
3. Simulate the sweep: maintain an active set sorted by $y$.
4. At each event, check only adjacent segments.
5. Verify each intersection appears once and only once.
6. Compare with a brute-force $O(n^2)$ method.

#### Test Cases

| Segments                  | Intersections | Notes                   |
| ------------------------- | ------------- | ----------------------- |
| Two diagonals of a square | 1             | Intersection at center  |
| Five-point star           | 10            | All pairs intersect     |
| Parallel lines            | 0             | No intersections        |
| Random crossings          | Verified      | Matches expected output |

#### Complexity

$$
\text{Time: } O\big((n + k)\log n\big), \quad
\text{Space: } O(n)
$$

The Bentley–Ottmann algorithm is a model of geometric precision, sweeping across the plane, maintaining order, and revealing every crossing exactly once.

### 715 Segment Intersection Test

The Segment Intersection Test is the fundamental geometric routine that checks whether two line segments intersect in the plane.
It forms the building block for many larger algorithms, from polygon clipping to sweep line methods like Bentley–Ottmann.

At its heart is a simple principle: two segments intersect if and only if they straddle each other, determined by orientation tests using cross products.

#### What Problem Are We Solving?

Given two segments:

* $S_1 = (p_1, q_1)$
* $S_2 = (p_2, q_2)$

we want to determine whether they intersect, either at a point inside both segments or at an endpoint.

Mathematically, $S_1$ and $S_2$ intersect if:

1. The two segments cross each other, or
2. They are collinear and overlap.

#### How Does It Work (Plain Language)?

We use orientation tests to check the relative position of points.

For any three points $a, b, c$, define:

$$
\text{orient}(a, b, c) = (b_x - a_x)(c_y - a_y) - (b_y - a_y)(c_x - a_x)
$$

* $\text{orient}(a, b, c) > 0$: $c$ is left of the line $ab$
* $\text{orient}(a, b, c) < 0$: $c$ is right of the line $ab$
* $\text{orient}(a, b, c) = 0$: points are collinear

For segments $(p_1, q_1)$ and $(p_2, q_2)$:

Compute orientations:

* $o_1 = \text{orient}(p_1, q_1, p_2)$
* $o_2 = \text{orient}(p_1, q_1, q_2)$
* $o_3 = \text{orient}(p_2, q_2, p_1)$
* $o_4 = \text{orient}(p_2, q_2, q_1)$

Two segments properly intersect if:

$$
(o_1 \neq o_2) \quad \text{and} \quad (o_3 \neq o_4)
$$

If any $o_i = 0$, check if the corresponding point lies on the segment (collinear overlap).

#### Example Walkthrough

Segments:

* $S_1: (0,0)\text{–}(4,4)$
* $S_2: (0,4)\text{–}(4,0)$

Compute orientations:

| Pair                                   | Value | Meaning    |
| -------------------------------------- | ----- | ---------- |
| $o_1 = \text{orient}(0,0),(4,4),(0,4)$ | $> 0$ | left turn  |
| $o_2 = \text{orient}(0,0),(4,4),(4,0)$ | $< 0$ | right turn |
| $o_3 = \text{orient}(0,4),(4,0),(0,0)$ | $< 0$ | right turn |
| $o_4 = \text{orient}(0,4),(4,0),(4,4)$ | $> 0$ | left turn  |

Since $o_1 \neq o_2$ and $o_3 \neq o_4$, the segments intersect at $(2,2)$.

#### Tiny Code (C)

```c
#include <stdio.h>

typedef struct { double x, y; } Point;

double orient(Point a, Point b, Point c) {
    return (b.x - a.x)*(c.y - a.y) - (b.y - a.y)*(c.x - a.x);
}

int onSegment(Point a, Point b, Point c) {
    return b.x <= fmax(a.x, c.x) && b.x >= fmin(a.x, c.x) &&
           b.y <= fmax(a.y, c.y) && b.y >= fmin(a.y, c.y);
}

int intersect(Point p1, Point q1, Point p2, Point q2) {
    double o1 = orient(p1, q1, p2);
    double o2 = orient(p1, q1, q2);
    double o3 = orient(p2, q2, p1);
    double o4 = orient(p2, q2, q1);

    if (o1*o2 < 0 && o3*o4 < 0) return 1;

    if (o1 == 0 && onSegment(p1, p2, q1)) return 1;
    if (o2 == 0 && onSegment(p1, q2, q1)) return 1;
    if (o3 == 0 && onSegment(p2, p1, q2)) return 1;
    if (o4 == 0 && onSegment(p2, q1, q2)) return 1;

    return 0;
}

int main() {
    Point a={0,0}, b={4,4}, c={0,4}, d={4,0};
    printf("Intersect? %s\n", intersect(a,b,c,d) ? "Yes" : "No");
}
```

#### Tiny Code (Python)

```python
def orient(a, b, c):
    return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])

def on_segment(a, b, c):
    return (min(a[0], c[0]) <= b[0] <= max(a[0], c[0]) and
            min(a[1], c[1]) <= b[1] <= max(a[1], c[1]))

def intersect(p1, q1, p2, q2):
    o1 = orient(p1, q1, p2)
    o2 = orient(p1, q1, q2)
    o3 = orient(p2, q2, p1)
    o4 = orient(p2, q2, q1)

    if o1*o2 < 0 and o3*o4 < 0:
        return True
    if o1 == 0 and on_segment(p1, p2, q1): return True
    if o2 == 0 and on_segment(p1, q2, q1): return True
    if o3 == 0 and on_segment(p2, p1, q2): return True
    if o4 == 0 and on_segment(p2, q1, q2): return True
    return False

print(intersect((0,0),(4,4),(0,4),(4,0)))
```

#### Why It Matters

* Core primitive for many geometry algorithms
* Enables polygon intersection, clipping, and triangulation
* Used in computational geometry, GIS, CAD, and physics engines

Applications:

* Detecting collisions or crossings
* Building visibility graphs
* Checking self-intersections in polygons
* Foundation for sweep line and clipping algorithms

#### A Gentle Proof (Why It Works)

For segments $AB$ and $CD$ to intersect, they must straddle each other.
That is, $C$ and $D$ must lie on different sides of $AB$, and $A$ and $B$ must lie on different sides of $CD$.

The orientation function $\text{orient}(a,b,c)$ gives the signed area of triangle $(a,b,c)$.
If the signs of $\text{orient}(A,B,C)$ and $\text{orient}(A,B,D)$ differ, $C$ and $D$ are on opposite sides of $AB$.

Thus, if:

$$
\text{sign}(\text{orient}(A,B,C)) \neq \text{sign}(\text{orient}(A,B,D))
$$

and

$$
\text{sign}(\text{orient}(C,D,A)) \neq \text{sign}(\text{orient}(C,D,B))
$$

then the two segments must cross.
Collinear cases ($\text{orient}=0$) are handled separately by checking for overlap.

#### Try It Yourself

1. Draw two crossing segments, verify signs of orientations.
2. Try parallel non-intersecting segments, confirm test returns false.
3. Test collinear overlapping segments.
4. Extend to 3D (use vector cross products).
5. Combine with bounding box checks for faster filtering.

#### Test Cases

| Segments                        | Result    | Notes              |
| ------------------------------- | --------- | ------------------ |
| $(0,0)-(4,4)$ and $(0,4)-(4,0)$ | Intersect | Cross at $(2,2)$   |
| $(0,0)-(4,0)$ and $(5,0)-(6,0)$ | No        | Disjoint collinear |
| $(0,0)-(4,0)$ and $(2,0)-(6,0)$ | Yes       | Overlapping        |
| $(0,0)-(4,0)$ and $(0,1)-(4,1)$ | No        | Parallel           |

#### Complexity

$$
\text{Time: } O(1), \quad \text{Space: } O(1)
$$

The segment intersection test is geometry's atomic operation, a single, precise check built from cross products and orientation logic.

### 716 Line Sweep for Segments

The Line Sweep for Segments algorithm is a general event-driven framework for detecting intersections, overlaps, or coverage among many line segments efficiently.
It processes events (segment starts, ends, and intersections) in sorted order using a moving vertical sweep line and a balanced tree to track active segments.

This is the conceptual backbone behind algorithms like Bentley–Ottmann, rectangle union area, and overlap counting.

#### What Problem Are We Solving?

Given a set of $n$ segments (or intervals) on the plane, we want to efficiently:

* Detect intersections among them
* Count overlaps or coverage
* Compute union or intersection regions

A naive approach would compare every pair ($O(n^2)$), but a sweep line avoids unnecessary checks by maintaining only the local neighborhood of segments currently intersecting the sweep.

#### How Does It Work (Plain Language)?

We conceptually slide a vertical line across the plane from left to right, processing key events in x-sorted order:

| Step                                                                                       | Description                                                                          |
| ------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------ |
| 1                                                                                          | Event queue (EQ): all segment endpoints and known intersections, sorted by $x$.  |
| 2                                                                                          | Active set (AS): segments currently intersecting the sweep line, ordered by $y$. |
| 3                                                                                          | Process each event $e$ from left to right:                                           |
|   a. Start event: insert segment into AS; check intersection with immediate neighbors. |                                                                                      |
|   b. End event: remove segment from AS.                                                |                                                                                      |
|   c. Intersection event: report intersection; swap segment order; check new neighbors. |                                                                                      |
| 4                                                                                          | Continue until EQ is empty.                                                          |

At each step, the active set contains only those segments that are currently "alive" under the sweep line. Only neighbor pairs in AS can intersect.

#### Example Walkthrough

Segments:

* $S_1: (0,0)\text{–}(4,4)$
* $S_2: (0,4)\text{–}(4,0)$
* $S_3: (1,3)\text{–}(3,3)$

Events (sorted by x):
$(0,0), (0,4), (1,3), (2,2), (3,3), (4,0), (4,4)$

Steps:

1. At $x=0$: insert $S_1, S_2$ → check intersection $(2,2)$ → enqueue event.
2. At $x=1$: insert $S_3$ → check against neighbors $S_1$, $S_2$.
3. At $x=2$: process intersection event $(2,2)$ → swap order of $S_1$, $S_2$.
4. Continue until all segments processed.

 Output: intersection point $(2,2)$.

#### Tiny Code (Python Concept)

A conceptual skeleton for segment sweeping:

```python
from bisect import insort
from collections import namedtuple

Event = namedtuple("Event", ["x", "y", "type", "segment"])

def orientation(a, b, c):
    return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])

def intersect(s1, s2):
    a, b = s1
    c, d = s2
    o1 = orientation(a, b, c)
    o2 = orientation(a, b, d)
    o3 = orientation(c, d, a)
    o4 = orientation(c, d, b)
    return (o1*o2 < 0 and o3*o4 < 0)

def sweep_segments(segments):
    events = []
    for s in segments:
        (x1, y1), (x2, y2) = s
        if x1 > x2: s = ((x2, y2), (x1, y1))
        events += [(x1, y1, 'start', s), (x2, y2, 'end', s)]
    events.sort()

    active = []
    intersections = []

    for x, y, t, s in events:
        if t == 'start':
            insort(active, s)
            # check neighbors
            for other in active:
                if other != s and intersect(s, other):
                    intersections.append((x, y))
        elif t == 'end':
            active.remove(s)
    return intersections

segments = [((0,0),(4,4)), ((0,4),(4,0)), ((1,3),(3,3))]
print("Intersections:", sweep_segments(segments))
```

#### Why It Matters

* Unified approach for many geometry problems
* Forms the base of Bentley–Ottmann, rectangle union, and sweep circle algorithms
* Efficient: local checks instead of global comparisons

Applications:

* Detecting collisions or intersections
* Computing union area of shapes
* Event-driven simulations
* Visibility graphs and motion planning

#### A Gentle Proof (Why It Works)

At each $x$-coordinate, the active set represents the current "slice" of segments under the sweep line.

Key invariants:

1. The active set is ordered by y-coordinate, reflecting vertical order at the sweep line.
2. Two segments can only intersect if they are adjacent in this ordering.
3. Every intersection corresponds to a swap in order, so each is discovered once.

Each event (insert, remove, swap) takes $O(\log n)$ with balanced trees.
Each intersection adds one event, so total complexity:

$$
O\big((n + k)\log n\big)
$$

where $k$ is the number of intersections.

#### Try It Yourself

1. Draw several segments, label start and end events.
2. Sort events by $x$, step through the sweep.
3. Maintain a vertical ordering at each step.
4. Add a horizontal segment, see it overlap multiple active segments.
5. Count intersections and confirm correctness.

#### Test Cases

| Segments                         | Intersections | Notes                   |
| -------------------------------- | ------------- | ----------------------- |
| $(0,0)$–$(4,4)$, $(0,4)$–$(4,0)$ | 1             | Cross at $(2,2)$        |
| Parallel non-overlapping         | 0             | No intersection         |
| Horizontal overlaps              | Multiple      | Shared region           |
| Random crossings                 | Verified      | Matches expected output |

#### Complexity

$$
\text{Time: } O\big((n + k)\log n\big), \quad
\text{Space: } O(n)
$$

The line sweep framework is the geometric scheduler, moving steadily across the plane, tracking active shapes, and catching every event exactly when it happens.

### 717 Intersection via Orientation (CCW Test)

The Intersection via Orientation method, often called the CCW test (Counter-Clockwise test), is one of the simplest and most elegant tools in computational geometry. It determines whether two line segments intersect by analyzing their orientations, that is, whether triples of points turn clockwise or counterclockwise.

It's a clean, purely algebraic way to reason about geometry without explicitly solving equations for line intersections.

#### What Problem Are We Solving?

Given two line segments:

* $S_1 = (p_1, q_1)$
* $S_2 = (p_2, q_2)$

we want to determine if they intersect, either at a point inside both segments or at an endpoint.

The CCW test works entirely with determinants (cross products), avoiding floating-point divisions and handling edge cases like collinearity.

#### How Does It Work (Plain Language)?

For three points $a, b, c$, define the orientation function:

$$
\text{orient}(a, b, c) = (b_x - a_x)(c_y - a_y) - (b_y - a_y)(c_x - a_x)
$$

* $\text{orient}(a,b,c) > 0$ → counter-clockwise turn (CCW)
* $\text{orient}(a,b,c) < 0$ → clockwise turn (CW)
* $\text{orient}(a,b,c) = 0$ → collinear

For two segments $(p_1, q_1)$ and $(p_2, q_2)$, we compute:

$$
\begin{aligned}
o_1 &= \text{orient}(p_1, q_1, p_2) \
o_2 &= \text{orient}(p_1, q_1, q_2) \
o_3 &= \text{orient}(p_2, q_2, p_1) \
o_4 &= \text{orient}(p_2, q_2, q_1)
\end{aligned}
$$

The two segments intersect if and only if:

$$
(o_1 \neq o_2) \quad \text{and} \quad (o_3 \neq o_4)
$$

This ensures that each segment straddles the other.

If any $o_i = 0$, we check for collinear overlap using a bounding-box test.

#### Example Walkthrough

Segments:

* $S_1: (0,0)\text{–}(4,4)$
* $S_2: (0,4)\text{–}(4,0)$

Compute orientations:

| Expression                         | Value | Meaning |
| ---------------------------------- | ----- | ------- |
| $o_1 = \text{orient}(0,0,4,4,0,4)$ | $> 0$ | CCW     |
| $o_2 = \text{orient}(0,0,4,4,4,0)$ | $< 0$ | CW      |
| $o_3 = \text{orient}(0,4,4,0,0,0)$ | $< 0$ | CW      |
| $o_4 = \text{orient}(0,4,4,0,4,4)$ | $> 0$ | CCW     |

Because $o_1 \neq o_2$ and $o_3 \neq o_4$, the segments intersect at $(2,2)$.

#### Tiny Code (Python)

```python
def orient(a, b, c):
    return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])

def on_segment(a, b, c):
    return (min(a[0], c[0]) <= b[0] <= max(a[0], c[0]) and
            min(a[1], c[1]) <= b[1] <= max(a[1], c[1]))

def intersect(p1, q1, p2, q2):
    o1 = orient(p1, q1, p2)
    o2 = orient(p1, q1, q2)
    o3 = orient(p2, q2, p1)
    o4 = orient(p2, q2, q1)

    if o1 * o2 < 0 and o3 * o4 < 0:
        return True  # general case

    # Special cases: collinear overlap
    if o1 == 0 and on_segment(p1, p2, q1): return True
    if o2 == 0 and on_segment(p1, q2, q1): return True
    if o3 == 0 and on_segment(p2, p1, q2): return True
    if o4 == 0 and on_segment(p2, q1, q2): return True

    return False

# Example
print(intersect((0,0),(4,4),(0,4),(4,0)))  # True
```

#### Tiny Code (C)

```c
#include <stdio.h>

typedef struct { double x, y; } Point;

double orient(Point a, Point b, Point c) {
    return (b.x - a.x)*(c.y - a.y) - (b.y - a.y)*(c.x - a.x);
}

int onSegment(Point a, Point b, Point c) {
    return b.x <= fmax(a.x, c.x) && b.x >= fmin(a.x, c.x) &&
           b.y <= fmax(a.y, c.y) && b.y >= fmin(a.y, c.y);
}

int intersect(Point p1, Point q1, Point p2, Point q2) {
    double o1 = orient(p1, q1, p2);
    double o2 = orient(p1, q1, q2);
    double o3 = orient(p2, q2, p1);
    double o4 = orient(p2, q2, q1);

    if (o1*o2 < 0 && o3*o4 < 0) return 1;

    if (o1 == 0 && onSegment(p1, p2, q1)) return 1;
    if (o2 == 0 && onSegment(p1, q2, q1)) return 1;
    if (o3 == 0 && onSegment(p2, p1, q2)) return 1;
    if (o4 == 0 && onSegment(p2, q1, q2)) return 1;

    return 0;
}

int main() {
    Point a={0,0}, b={4,4}, c={0,4}, d={4,0};
    printf("Intersect? %s\n", intersect(a,b,c,d) ? "Yes" : "No");
}
```

#### Why It Matters

* Fundamental primitive in geometry and computational graphics
* Forms the core of polygon intersection, clipping, and triangulation
* Numerically stable, avoids divisions or floating-point slopes
* Used in collision detection, pathfinding, and geometry kernels

Applications:

* Detecting intersections in polygon meshes
* Checking path crossings in navigation systems
* Implementing clipping algorithms (e.g., Weiler–Atherton)

#### A Gentle Proof (Why It Works)

A segment $AB$ and $CD$ intersect if each pair of endpoints straddles the other segment.
The orientation function $\text{orient}(A,B,C)$ gives the signed area of the triangle $(A,B,C)$.

* If $\text{orient}(A,B,C)$ and $\text{orient}(A,B,D)$ have opposite signs, then $C$ and $D$ are on different sides of $AB$.
* Similarly, if $\text{orient}(C,D,A)$ and $\text{orient}(C,D,B)$ have opposite signs, then $A$ and $B$ are on different sides of $CD$.

Therefore, if:

$$
\text{sign}(\text{orient}(A,B,C)) \neq \text{sign}(\text{orient}(A,B,D))
$$

and

$$
\text{sign}(\text{orient}(C,D,A)) \neq \text{sign}(\text{orient}(C,D,B))
$$

then the two segments must cross.
If any orientation is $0$, we simply check whether the collinear point lies within the segment bounds.

#### Try It Yourself

1. Sketch two crossing segments; label orientation signs at each vertex.
2. Try non-intersecting and parallel cases, confirm orientation tests differ.
3. Check collinear overlapping segments.
4. Implement a version that counts intersections among many segments.
5. Compare with brute-force coordinate intersection.

#### Test Cases

| Segments                            | Result    | Notes              |
| ----------------------------------- | --------- | ------------------ |
| $(0,0)$–$(4,4)$ and $(0,4)$–$(4,0)$ | Intersect | Cross at $(2,2)$   |
| $(0,0)$–$(4,0)$ and $(5,0)$–$(6,0)$ | No        | Disjoint collinear |
| $(0,0)$–$(4,0)$ and $(2,0)$–$(6,0)$ | Yes       | Overlap            |
| $(0,0)$–$(4,0)$ and $(0,1)$–$(4,1)$ | No        | Parallel lines     |

#### Complexity

$$
\text{Time: } O(1), \quad \text{Space: } O(1)
$$

The CCW test distills intersection detection into a single algebraic test, a foundation of geometric reasoning built from orientation signs.

### 718 Circle Intersection

The Circle Intersection problem asks whether two circles intersect, and if so, to compute their intersection points.
It's a classic example of blending algebraic geometry with spatial reasoning, used in collision detection, Venn diagrams, and range queries.

Two circles can have 0, 1, 2, or infinite (coincident) intersection points, depending on their relative positions.

#### What Problem Are We Solving?

Given two circles:

* $C_1$: center $(x_1, y_1)$, radius $r_1$
* $C_2$: center $(x_2, y_2)$, radius $r_2$

we want to determine:

1. Do they intersect?
2. If yes, what are the intersection points?

#### How Does It Work (Plain Language)?

Let the distance between centers be:

$$
d = \sqrt{(x_2 - x_1)^2 + (y_2 - y_1)^2}
$$

Now compare $d$ with $r_1$ and $r_2$:

| Condition          | Meaning                                      |                  |                                                      |
| ------------------ | -------------------------------------------- | ---------------- | ---------------------------------------------------- |
| $d > r_1 + r_2$    | Circles are separate (no intersection)   |                  |                                                      |
| $d = r_1 + r_2$    | Circles touch externally (1 point)       |                  |                                                      |
| $                  | r_1 - r_2                                    | < d < r_1 + r_2$ | Circles intersect (2 points)                     |
| $d =               | r_1 - r_2                                    | $                | Circles touch internally (1 point)               |
| $d <               | r_1 - r_2                                    | $                | One circle is inside the other (no intersection) |
| $d = 0, r_1 = r_2$ | Circles are coincident (infinite points) |                  |                                                      |

If they intersect ($|r_1 - r_2| < d < r_1 + r_2$), the intersection points can be computed geometrically.

#### Derivation of Intersection Points

We find the line of intersection between the two circles.

Let:

$$
a = \frac{r_1^2 - r_2^2 + d^2}{2d}
$$

Then, the point $P$ on the line connecting centers where the intersection chord crosses is:

$$
P_x = x_1 + a \cdot \frac{x_2 - x_1}{d}
$$
$$
P_y = y_1 + a \cdot \frac{y_2 - y_1}{d}
$$

The height from $P$ to each intersection point is:

$$
h = \sqrt{r_1^2 - a^2}
$$

The intersection points are:

$$
(x_3, y_3) = \big(P_x \pm h \cdot \frac{y_2 - y_1}{d},; P_y \mp h \cdot \frac{x_2 - x_1}{d}\big)
$$

These two points represent the intersection of the circles.

#### Example Walkthrough

Circles:

* $C_1: (0, 0), r_1 = 5$
* $C_2: (6, 0), r_2 = 5$

Compute:

* $d = 6$
* $r_1 + r_2 = 10$, $|r_1 - r_2| = 0$
  So $|r_1 - r_2| < d < r_1 + r_2$ → 2 intersection points

Then:

$$
a = \frac{5^2 - 5^2 + 6^2}{2 \cdot 6} = 3
$$
$$
h = \sqrt{5^2 - 3^2} = 4
$$

$P = (3, 0)$ → intersection points:

$$
(x_3, y_3) = (3, \pm 4)
$$

 Intersections: $(3, 4)$ and $(3, -4)$

#### Tiny Code (Python)

```python
from math import sqrt

def circle_intersection(x1, y1, r1, x2, y2, r2):
    dx, dy = x2 - x1, y2 - y1
    d = sqrt(dx*dx + dy*dy)
    if d > r1 + r2 or d < abs(r1 - r2) or d == 0 and r1 == r2:
        return []
    a = (r1*r1 - r2*r2 + d*d) / (2*d)
    h = sqrt(r1*r1 - a*a)
    xm = x1 + a * dx / d
    ym = y1 + a * dy / d
    rx = -dy * (h / d)
    ry =  dx * (h / d)
    return [(xm + rx, ym + ry), (xm - rx, ym - ry)]

print(circle_intersection(0, 0, 5, 6, 0, 5))
```

#### Tiny Code (C)

```c
#include <stdio.h>
#include <math.h>

void circle_intersection(double x1, double y1, double r1,
                         double x2, double y2, double r2) {
    double dx = x2 - x1, dy = y2 - y1;
    double d = sqrt(dx*dx + dy*dy);

    if (d > r1 + r2 || d < fabs(r1 - r2) || (d == 0 && r1 == r2)) {
        printf("No unique intersection\n");
        return;
    }

    double a = (r1*r1 - r2*r2 + d*d) / (2*d);
    double h = sqrt(r1*r1 - a*a);
    double xm = x1 + a * dx / d;
    double ym = y1 + a * dy / d;
    double rx = -dy * (h / d);
    double ry =  dx * (h / d);

    printf("Intersection points:\n");
    printf("(%.2f, %.2f)\n", xm + rx, ym + ry);
    printf("(%.2f, %.2f)\n", xm - rx, ym - ry);
}

int main() {
    circle_intersection(0,0,5,6,0,5);
}
```

#### Why It Matters

* Fundamental geometric building block
* Used in collision detection, Venn diagrams, circle packing, sensor range overlap
* Enables circle clipping, lens area computation, and circle graph construction

Applications:

* Graphics (drawing arcs, blending circles)
* Robotics (sensing overlap)
* Physics engines (sphere–sphere collision)
* GIS (circular buffer intersection)

#### A Gentle Proof (Why It Works)

The two circle equations are:

$$
(x - x_1)^2 + (y - y_1)^2 = r_1^2
$$
$$
(x - x_2)^2 + (y - y_2)^2 = r_2^2
$$

Subtracting eliminates squares and yields a linear equation for the line connecting intersection points (the radical line).
Solving this line together with one circle's equation gives two symmetric points, derived via $a$ and $h$ from the geometry of chords.

Thus, the solution is exact and symmetric, and naturally handles 0, 1, or 2 intersections depending on $d$.

#### Try It Yourself

1. Draw two overlapping circles and compute $d$, $a$, $h$.
2. Compare geometric sketch with computed points.
3. Test tangent circles ($d = r_1 + r_2$).
4. Test nested circles ($d < |r_1 - r_2|$).
5. Extend to 3D sphere–sphere intersection (circle of intersection).

#### Test Cases

| Circle 1     | Circle 2     | Result                |
| ------------ | ------------ | --------------------- |
| $(0,0), r=5$ | $(6,0), r=5$ | $(3, 4)$, $(3, -4)$   |
| $(0,0), r=3$ | $(6,0), r=3$ | Tangent (1 point)     |
| $(0,0), r=2$ | $(0,0), r=2$ | Coincident (infinite) |
| $(0,0), r=2$ | $(5,0), r=2$ | No intersection       |

#### Complexity

$$
\text{Time: } O(1), \quad \text{Space: } O(1)
$$

Circle intersection blends algebra and geometry, a precise construction revealing where two round worlds meet.

### 719 Polygon Intersection

The Polygon Intersection problem asks us to compute the overlapping region (or intersection) between two polygons.
It's a fundamental operation in computational geometry, forming the basis for clipping, boolean operations, map overlays, and collision detection.

There are several standard methods:

* Sutherland–Hodgman (clip subject polygon against convex clip polygon)
* Weiler–Atherton (general polygons with holes)
* Greiner–Hormann (robust for complex shapes)

#### What Problem Are We Solving?

Given two polygons $P$ and $Q$, we want to compute:

$$
R = P \cap Q
$$

where $R$ is the intersection polygon, representing the region common to both.

For convex polygons, intersection is straightforward; for concave or self-intersecting polygons, careful clipping is needed.

#### How Does It Work (Plain Language)?

Let's describe the classic Sutherland–Hodgman approach (for convex clipping polygons):

1. Initialize: Let the output polygon = subject polygon.
2. Iterate over each edge of the clip polygon.
3. Clip the current output polygon against the clip edge:

   * Keep points inside the edge.
   * Compute intersection points for edges crossing the boundary.
4. After all edges processed, the remaining polygon is the intersection.

This works because every edge trims the subject polygon step by step.

#### Key Idea

For a directed edge $(C_i, C_{i+1})$ of the clip polygon, a point $P$ is inside if:

$$
(C_{i+1} - C_i) \times (P - C_i) \ge 0
$$

This uses the cross product to check orientation relative to the clip edge.

Each polygon edge pair may produce at most one intersection point.

#### Example Walkthrough

Clip polygon (square):
$(0,0)$, $(5,0)$, $(5,5)$, $(0,5)$

Subject polygon (triangle):
$(2,-1)$, $(6,2)$, $(2,6)$

Process edges:

1. Clip against bottom edge $(0,0)$–$(5,0)$ → remove points below $y=0$
2. Clip against right edge $(5,0)$–$(5,5)$ → cut off $x>5$
3. Clip against top $(5,5)$–$(0,5)$ → trim above $y=5$
4. Clip against left $(0,5)$–$(0,0)$ → trim $x<0$

 Output polygon: a pentagon representing the overlap inside the square.

#### Tiny Code (Python)

```python
def inside(p, cp1, cp2):
    return (cp2[0]-cp1[0])*(p[1]-cp1[1]) > (cp2[1]-cp1[1])*(p[0]-cp1[0])

def intersection(s, e, cp1, cp2):
    dc = (cp1[0]-cp2[0], cp1[1]-cp2[1])
    dp = (s[0]-e[0], s[1]-e[1])
    n1 = cp1[0]*cp2[1] - cp1[1]*cp2[0]
    n2 = s[0]*e[1] - s[1]*e[0]
    denom = dc[0]*dp[1] - dc[1]*dp[0]
    if denom == 0: return e
    x = (n1*dp[0] - n2*dc[0]) / denom
    y = (n1*dp[1] - n2*dc[1]) / denom
    return (x, y)

def suth_hodg_clip(subject, clip):
    output = subject
    for i in range(len(clip)):
        input_list = output
        output = []
        cp1 = clip[i]
        cp2 = clip[(i+1)%len(clip)]
        for j in range(len(input_list)):
            s = input_list[j-1]
            e = input_list[j]
            if inside(e, cp1, cp2):
                if not inside(s, cp1, cp2):
                    output.append(intersection(s, e, cp1, cp2))
                output.append(e)
            elif inside(s, cp1, cp2):
                output.append(intersection(s, e, cp1, cp2))
    return output

subject = [(2,-1),(6,2),(2,6)]
clip = [(0,0),(5,0),(5,5),(0,5)]
print(suth_hodg_clip(subject, clip))
```

#### Why It Matters

* Core of polygon operations: intersection, union, difference
* Used in clipping pipelines, rendering, CAD, GIS
* Efficient ($O(nm)$) for $n$-vertex subject and $m$-vertex clip polygon
* Stable for convex clipping polygons

Applications:

* Graphics: clipping polygons to viewport
* Mapping: overlaying shapes, zoning regions
* Simulation: detecting overlapping regions
* Computational geometry: polygon boolean ops

#### A Gentle Proof (Why It Works)

Each clip edge defines a half-plane.
The intersection of convex polygons equals the intersection of all half-planes bounding the clip polygon.

Formally:
$$
R = P \cap \bigcap_{i=1}^{m} H_i
$$
where $H_i$ is the half-plane on the interior side of clip edge $i$.

At each step, we take the polygon–half-plane intersection, which is itself convex.
Thus, after clipping against all edges, we obtain the exact intersection.

Since each vertex can generate at most one intersection per edge, the total complexity is $O(nm)$.

#### Try It Yourself

1. Draw a triangle and clip it against a square, follow each step.
2. Try reversing clip and subject polygons.
3. Test degenerate cases (no intersection, full containment).
4. Compare convex vs concave clip polygons.
5. Extend to Weiler–Atherton for non-convex shapes.

#### Test Cases

| Subject Polygon        | Clip Polygon | Result                 |
| ---------------------- | ------------ | ---------------------- |
| Triangle across square | Square       | Clipped pentagon       |
| Fully inside           | Square       | Unchanged              |
| Fully outside          | Square       | Empty                  |
| Overlapping rectangles | Both         | Intersection rectangle |

#### Complexity

$$
\text{Time: } O(nm), \quad \text{Space: } O(n + m)
$$

Polygon intersection is geometry's boolean operator, trimming shapes step by step until only the shared region remains.

### 720 Nearest Neighbor Pair (with KD-Tree)

The Nearest Neighbor Pair problem asks us to find the pair of points that are closest together in a given set, a fundamental question in computational geometry and spatial data analysis.

It underpins algorithms in clustering, graphics, machine learning, and collision detection, and can be solved efficiently using divide and conquer, sweep line, or spatial data structures like KD-Trees.

#### What Problem Are We Solving?

Given a set of $n$ points $P = {p_1, p_2, \dots, p_n}$ in the plane, find two distinct points $(p_i, p_j)$ such that the Euclidean distance

$$
d(p_i, p_j) = \sqrt{(x_i - x_j)^2 + (y_i - y_j)^2}
$$

is minimized.

Naively checking all $\binom{n}{2}$ pairs takes $O(n^2)$ time.
We want an $O(n \log n)$ or better solution.

#### How Does It Work (Plain Language)?

We'll focus on the KD-Tree approach, which efficiently supports nearest-neighbor queries in low-dimensional space.

A KD-Tree (k-dimensional tree) recursively partitions space along coordinate axes:

1. Build phase

   * Sort points by $x$, split at median → root node
   * Recursively build left (smaller $x$) and right (larger $x$) subtrees
   * Alternate axis at each depth ($x$, $y$, $x$, $y$, …)

2. Query phase (for each point)

   * Traverse KD-Tree to find nearest candidate
   * Backtrack to check subtrees that might contain closer points
   * Maintain global minimum distance and pair

By leveraging axis-aligned bounding boxes, many regions are pruned (ignored) early.

#### Step-by-Step (Conceptual)

1. Build KD-Tree in $O(n \log n)$.
2. For each point $p$, search for its nearest neighbor in $O(\log n)$ expected time.
3. Track global minimum pair $(p, q)$ with smallest distance.

#### Example Walkthrough

Points:
$$
P = {(1,1), (4,4), (5,1), (7,2)}
$$

1. Build KD-Tree splitting by $x$:
   root = $(4,4)$
   left subtree = $(1,1)$
   right subtree = $(5,1),(7,2)$

2. Query nearest for each:

   * $(1,1)$ → nearest = $(4,4)$ ($d=4.24$)
   * $(4,4)$ → nearest = $(5,1)$ ($d=3.16$)
   * $(5,1)$ → nearest = $(7,2)$ ($d=2.24$)
   * $(7,2)$ → nearest = $(5,1)$ ($d=2.24$)

 Closest pair: $(5,1)$ and $(7,2)$

#### Tiny Code (Python)

```python
from math import sqrt

def dist(a, b):
    return sqrt((a[0]-b[0])2 + (a[1]-b[1])2)

def build_kdtree(points, depth=0):
    if not points: return None
    k = 2
    axis = depth % k
    points.sort(key=lambda p: p[axis])
    mid = len(points) // 2
    return {
        'point': points[mid],
        'left': build_kdtree(points[:mid], depth+1),
        'right': build_kdtree(points[mid+1:], depth+1)
    }

def nearest_neighbor(tree, target, depth=0, best=None):
    if tree is None: return best
    point = tree['point']
    if best is None or dist(target, point) < dist(target, best):
        best = point
    axis = depth % 2
    next_branch = tree['left'] if target[axis] < point[axis] else tree['right']
    best = nearest_neighbor(next_branch, target, depth+1, best)
    return best

points = [(1,1), (4,4), (5,1), (7,2)]
tree = build_kdtree(points)
best_pair = None
best_dist = float('inf')
for p in points:
    q = nearest_neighbor(tree, p)
    if q != p:
        d = dist(p, q)
        if d < best_dist:
            best_pair = (p, q)
            best_dist = d
print("Closest pair:", best_pair, "Distance:", best_dist)
```

#### Tiny Code (C, Conceptual Sketch)

Building a full KD-tree in C is more elaborate, but core logic:

```c
double dist(Point a, Point b) {
    double dx = a.x - b.x, dy = a.y - b.y;
    return sqrt(dx*dx + dy*dy);
}

// Recursively split points along x or y based on depth
Node* build_kdtree(Point* points, int n, int depth) {
    // Sort by axis, select median as root
    // Recurse for left and right
}

// Search nearest neighbor recursively with pruning
void nearest_neighbor(Node* root, Point target, Point* best, double* bestDist, int depth) {
    // Compare current point, recurse in promising branch
    // Backtrack if other branch may contain closer point
}
```

#### Why It Matters

* Avoids $O(n^2)$ brute-force
* Scales well for moderate dimensions (2D, 3D)
* Generalizes to range search, radius queries, clustering

Applications:

* Graphics (object proximity, mesh simplification)
* Machine learning (k-NN classification)
* Robotics (nearest obstacle detection)
* Spatial databases (geo queries)

#### A Gentle Proof (Why It Works)

Each recursive partition defines a half-space where points are stored.
When searching, we always explore the side containing the query point, but must check the other side if the hypersphere around the query point crosses the partition plane.

Since each level splits data roughly in half, the expected number of visited nodes is $O(\log n)$.
Building the tree is $O(n \log n)$ by recursive median finding.

Overall nearest-pair complexity:

$$
O(n \log n)
$$

#### Try It Yourself

1. Draw 10 random points, compute brute-force pair.
2. Build a KD-tree manually (alternate x/y).
3. Trace nearest neighbor search steps.
4. Compare search order and pruning decisions.
5. Extend to 3D points.

#### Test Cases

| Points                | Result        | Notes               |
| --------------------- | ------------- | ------------------- |
| (0,0), (1,1), (3,3)   | (0,0)-(1,1)   | $d=\sqrt2$          |
| (1,1), (2,2), (2,1.1) | (2,2)-(2,1.1) | Closest             |
| Random 10 pts         | Verified      | Matches brute force |

#### Complexity

$$
\text{Time: } O(n \log n), \quad \text{Space: } O(n)
$$

The Nearest Neighbor Pair is geometry's instinct, finding the closest companionship in a crowded space with elegant divide-and-search reasoning.

# Section 73. Line Sweep and Plane Sweep Algorithms 

### 721 Sweep Line for Events

The Sweep Line Algorithm is a unifying framework for solving many geometric problems by processing events in sorted order along a moving line (usually vertical).
It transforms spatial relationships into a temporal sequence, allowing us to track intersections, overlaps, or active objects efficiently using a dynamic active set.

This paradigm lies at the heart of algorithms like Bentley–Ottmann, closest pair, rectangle union, and skyline problems.

#### What Problem Are We Solving?

We want to process geometric events, points, segments, rectangles, circles, that interact in the plane.
The challenge: many spatial problems become simple if we consider only what's active at a specific sweep position.

For example:

* In intersection detection, only neighboring segments can intersect.
* In rectangle union, only active intervals contribute to total area.
* In skyline computation, only the tallest current height matters.

So we reformulate the problem:

> Move a sweep line across the plane, handle events one by one, and update the active set as geometry enters or leaves.

#### How Does It Work (Plain Language)?

1. Event Queue (EQ)

   * All critical points sorted by $x$ (or time).
   * Each event marks a start, end, or change (like intersection).

2. Active Set (AS)

   * Stores currently "active" objects that intersect the sweep line.
   * Maintained in a structure ordered by another coordinate (like $y$).

3. Main Loop
   Process each event in sorted order:

   * Insert new geometry into AS.
   * Remove expired geometry.
   * Query or update relationships (neighbors, counts, intersections).

4. Continue until EQ is empty.

Each step is logarithmic with balanced trees, so total complexity is $O((n+k)\log n)$, where $k$ is number of interactions (e.g. intersections).

#### Example Walkthrough

Let's take line segment intersection as an example:

Segments:

* $S_1: (0,0)$–$(4,4)$
* $S_2: (0,4)$–$(4,0)$

Events: endpoints sorted by $x$:
$(0,0)$, $(0,4)$, $(4,0)$, $(4,4)$

Steps:

1. At $x=0$, insert $S_1$, $S_2$.
2. Check active set order → detect intersection at $(2,2)$ → enqueue intersection event.
3. At $x=2$, process intersection → swap order in AS.
4. Continue → all intersections reported.

 Result: intersection point $(2,2)$ found by event-driven sweep.

#### Tiny Code (Python Sketch)

```python
import heapq

def sweep_line(events):
    heapq.heapify(events)  # min-heap by x
    active = set()
    while events:
        x, event_type, obj = heapq.heappop(events)
        if event_type == 'start':
            active.add(obj)
        elif event_type == 'end':
            active.remove(obj)
        elif event_type == 'intersection':
            print("Intersection at x =", x)
        # handle neighbors in active set if needed
```

Usage:

* Fill `events` with tuples `(x, type, object)`
* Insert / remove from `active` as sweep proceeds

#### Tiny Code (C Skeleton)

```c
#include <stdio.h>
#include <stdlib.h>

typedef struct { double x; int type; int id; } Event;

int cmp(const void* a, const void* b) {
    double x1 = ((Event*)a)->x, x2 = ((Event*)b)->x;
    return (x1 < x2) ? -1 : (x1 > x2);
}

void sweep_line(Event* events, int n) {
    qsort(events, n, sizeof(Event), cmp);
    for (int i = 0; i < n; i++) {
        if (events[i].type == 0) printf("Start event at x=%.2f\n", events[i].x);
        if (events[i].type == 1) printf("End event at x=%.2f\n", events[i].x);
    }
}
```

#### Why It Matters

* Universal pattern in computational geometry
* Turns 2D problems into sorted 1D scans
* Enables efficient detection of intersections, unions, and counts
* Used in graphics, GIS, simulation, CAD

Applications:

* Bentley–Ottmann (line intersections)
* Rectangle union area
* Range counting and queries
* Plane subdivision and visibility graphs

#### A Gentle Proof (Why It Works)

At any moment, only objects crossing the sweep line can influence the outcome.
By processing events in sorted order, we guarantee that:

* Every change in geometric relationships happens at an event.
* Between events, the structure of the active set remains stable.

Thus, we can maintain local state (neighbors, counts, maxima) incrementally, never revisiting old positions.

For $n$ input elements and $k$ interactions, total cost:

$$
O((n + k)\log n)
$$

since each insert, delete, or neighbor check is $O(\log n)$.

#### Try It Yourself

1. Draw segments and sort endpoints by $x$.
2. Sweep a vertical line and track which segments it crosses.
3. Record every time two segments change order → intersection!
4. Try rectangles or intervals, observe how active set changes.

#### Test Cases

| Input                        | Expected        | Notes                  |
| ---------------------------- | --------------- | ---------------------- |
| 2 segments crossing          | 1 intersection  | at center              |
| 3 segments crossing pairwise | 3 intersections | all detected           |
| Non-overlapping              | none            | active set stays small |

#### Complexity

$$
\text{Time: } O((n + k)\log n), \quad \text{Space: } O(n)
$$

The sweep line is geometry's conveyor belt, sliding across space, updating the world one event at a time.

### 722 Interval Scheduling

The Interval Scheduling algorithm is a cornerstone of greedy optimization on the line.
Given a set of time intervals, each representing a job or task, the goal is to select the maximum number of non-overlapping intervals.
This simple yet profound algorithm forms the heart of resource allocation, timeline planning, and spatial scheduling problems.

#### What Problem Are We Solving?

Given $n$ intervals:

$$
I_i = [s_i, f_i), \quad i = 1, \ldots, n
$$

we want to find the largest subset of intervals such that no two overlap.  
Formally, find $S \subseteq \{1, \ldots, n\}$ such that for all $i, j \in S$,

$$
[s_i, f_i) \cap [s_j, f_j) = \emptyset
$$

and $|S|$ is maximized.

Example:

| Interval | Start | Finish |
| --------- | ------ | ------- |
| $I_1$     | 1      | 4       |
| $I_2$     | 3      | 5       |
| $I_3$     | 0      | 6       |
| $I_4$     | 5      | 7       |
| $I_5$     | 8      | 9       |

Optimal schedule: $I_1, I_4, I_5$ (3 intervals)

#### How Does It Work (Plain Language)?

The greedy insight:

> Always pick the interval that finishes earliest, then discard all overlapping ones, and repeat.

Reasoning:

- Finishing early leaves more room for future tasks.  
- No earlier finish can increase the count; it only blocks later intervals.


#### Algorithm (Greedy Strategy)

1. Sort intervals by finishing time $f_i$.
2. Initialize empty set $S$.
3. For each interval $I_i$ in order:

   * If $I_i$ starts after or at the finish time of last selected interval → select it.
4. Return $S$.

#### Example Walkthrough

Input:
$(1,4), (3,5), (0,6), (5,7), (8,9)$

1. Sort by finish:
   $(1,4), (3,5), (0,6), (5,7), (8,9)$

2. Start with $(1,4)$

   * Next $(3,5)$ overlaps → skip
   * $(0,6)$ overlaps → skip
   * $(5,7)$ fits → select
   * $(8,9)$ fits → select

 Output: $(1,4), (5,7), (8,9)$

#### Tiny Code (Python)

```python
def interval_scheduling(intervals):
    intervals.sort(key=lambda x: x[1])  # sort by finish time
    selected = []
    current_end = float('-inf')
    for (s, f) in intervals:
        if s >= current_end:
            selected.append((s, f))
            current_end = f
    return selected

intervals = [(1,4),(3,5),(0,6),(5,7),(8,9)]
print("Optimal schedule:", interval_scheduling(intervals))
```

#### Tiny Code (C)

```c
#include <stdio.h>
#include <stdlib.h>

typedef struct { int s, f; } Interval;

int cmp(const void *a, const void *b) {
    return ((Interval*)a)->f - ((Interval*)b)->f;
}

void interval_scheduling(Interval arr[], int n) {
    qsort(arr, n, sizeof(Interval), cmp);
    int last_finish = -1;
    for (int i = 0; i < n; i++) {
        if (arr[i].s >= last_finish) {
            printf("(%d, %d)\n", arr[i].s, arr[i].f);
            last_finish = arr[i].f;
        }
    }
}

int main() {
    Interval arr[] = {{1,4},{3,5},{0,6},{5,7},{8,9}};
    interval_scheduling(arr, 5);
}
```

#### Why It Matters

* Greedy proof: earliest finishing interval never harms optimality
* Foundation for resource scheduling, CPU job selection, meeting room planning
* Basis for weighted variants, interval partitioning, segment trees

Applications:

* CPU process scheduling
* Railway or runway slot allocation
* Event planning and booking systems
* Non-overlapping task assignment

#### A Gentle Proof (Why It Works)

Let $S^*$ be an optimal solution, and $I_g$ be the earliest finishing interval chosen by the greedy algorithm.
We can transform $S^*$ so that it also includes $I_g$ without reducing its size, by replacing any overlapping interval with $I_g$.

Hence by induction:

* The greedy algorithm always finds an optimal subset.

Total running time is dominated by sorting:

$$
O(n \log n)
$$

#### Try It Yourself

1. Draw intervals on a line, simulate greedy selection.
2. Add overlapping intervals and see which get skipped.
3. Compare to a brute-force approach (check all subsets).
4. Extend to weighted interval scheduling (with DP).

#### Test Cases

| Intervals                     | Optimal Schedule  | Count |
| ----------------------------- | ----------------- | ----- |
| (1,4),(3,5),(0,6),(5,7),(8,9) | (1,4),(5,7),(8,9) | 3     |
| (0,2),(1,3),(2,4),(3,5)       | (0,2),(2,4)       | 2     |
| (1,10),(2,3),(3,4),(4,5)      | (2,3),(3,4),(4,5) | 3     |

#### Complexity

$$
\text{Time: } O(n \log n), \quad \text{Space: } O(1)
$$

The Interval Scheduling algorithm is the epitome of greedy elegance, choosing the earliest finish, one decision at a time, to paint the longest non-overlapping path across the timeline.

### 723 Rectangle Union Area

The Rectangle Union Area algorithm computes the total area covered by a set of axis-aligned rectangles.
Overlaps should be counted only once, even if multiple rectangles cover the same region.

This problem is a classic demonstration of the sweep line technique combined with interval management, transforming a 2D geometry question into a sequence of 1D range computations.

#### What Problem Are We Solving?

Given $n$ rectangles aligned with coordinate axes,
each rectangle $R_i = [x_1, x_2) \times [y_1, y_2)$,

we want to compute the total area of their union:

$$
A = \text{area}\left(\bigcup_{i=1}^n R_i\right)
$$

Overlapping regions must only be counted once.
Brute-force grid enumeration is too expensive, we need a geometric, event-driven approach.

#### How Does It Work (Plain Language)?

We use a vertical sweep line across the $x$-axis:

1. Events:
   Each rectangle generates two events:

   * at $x_1$: add vertical interval $[y_1, y_2)$
   * at $x_2$: remove vertical interval $[y_1, y_2)$

2. Active Set:
   During the sweep, maintain a structure storing active y-intervals, representing where the sweep line currently intersects rectangles.

3. Area Accumulation:
   As the sweep line moves from $x_i$ to $x_{i+1}$,
   the covered y-length ($L$) is computed from the active set,
   and the contributed area is:

   $$
   A += L \times (x_{i+1} - x_i)
   $$

By processing all $x$-events in sorted order, we capture all additions/removals and accumulate exact area.

#### Example Walkthrough

Rectangles:

1. $(1, 1, 3, 3)$
2. $(2, 2, 4, 4)$

Events:

* $x=1$: add [1,3]
* $x=2$: add [2,4]
* $x=3$: remove [1,3]
* $x=4$: remove [2,4]

Step-by-step:

| Interval                  | x-range | y-covered | Area |
| ------------------------- | ------- | --------- | ---- |
| [1,3]                     | 1→2     | 2         | 2    |
| [1,3]+[2,4] → merge [1,4] | 2→3     | 3         | 3    |
| [2,4]                     | 3→4     | 2         | 2    |

 Total area = 2 + 3 + 2 = 7

#### Tiny Code (Python)

```python
def union_area(rectangles):
    events = []
    for (x1, y1, x2, y2) in rectangles:
        events.append((x1, 1, y1, y2))  # start
        events.append((x2, -1, y1, y2)) # end
    events.sort()  # sort by x

    def compute_y_length(active):
        # merge intervals
        merged, last_y2, total = [], -float('inf'), 0
        for y1, y2 in sorted(active):
            y1 = max(y1, last_y2)
            if y2 > y1:
                total += y2 - y1
                last_y2 = y2
        return total

    active, prev_x, area = [], 0, 0
    for x, typ, y1, y2 in events:
        area += compute_y_length(active) * (x - prev_x)
        if typ == 1:
            active.append((y1, y2))
        else:
            active.remove((y1, y2))
        prev_x = x
    return area

rects = [(1,1,3,3),(2,2,4,4)]
print("Union area:", union_area(rects))  # 7
```

#### Tiny Code (C, Conceptual)

```c
typedef struct { double x; int type; double y1, y2; } Event;
```

* Sort events by $x$
* Maintain active intervals (linked list or segment tree)
* Compute merged $y$-length and accumulate $L \times \Delta x$

Efficient implementations use segment trees to track coverage counts and total length in $O(\log n)$ per update.

#### Why It Matters

* Foundational for computational geometry, GIS, graphics
* Handles union area, perimeter, volume (higher-dim analogues)
* Basis for collision areas, coverage computation, map overlays

Applications:

* Rendering overlapping rectangles
* Land or parcel union areas
* Collision detection (2D bounding boxes)
* CAD and layout design tools

#### A Gentle Proof (Why It Works)

At each sweep position, all changes occur at event boundaries ($x_i$).
Between $x_i$ and $x_{i+1}$, the set of active intervals remains fixed.
Hence, area can be computed incrementally:

$$
A = \sum_{i} L_i \cdot (x_{i+1} - x_i)
$$

where $L_i$ is total $y$-length covered at $x_i$.
Since every insertion/removal updates only local intervals, correctness follows from maintaining the union of active intervals.

#### Try It Yourself

1. Draw 2–3 overlapping rectangles.
2. List their $x$-events.
3. Sweep and track active $y$-intervals.
4. Merge overlaps to compute $L_i$.
5. Multiply by $\Delta x$ for each step.

#### Test Cases

| Rectangles           | Expected Area | Notes             |
| -------------------- | ------------- | ----------------- |
| (1,1,3,3), (2,2,4,4) | 7             | partial overlap   |
| (0,0,1,1), (1,0,2,1) | 2             | disjoint          |
| (0,0,2,2), (1,1,3,3) | 7             | overlap at corner |

#### Complexity

$$
\text{Time: } O(n \log n), \quad \text{Space: } O(n)
$$

The Rectangle Union Area algorithm turns a complex 2D union into a 1D sweep with active interval merging, precise, elegant, and scalable.

### 724 Segment Intersection (Bentley–Ottmann Variant)

The Segment Intersection problem asks us to find all intersection points among a set of $n$ line segments in the plane.
The Bentley–Ottmann algorithm is the canonical sweep line approach, improving naive $O(n^2)$ pairwise checking to

$$
O\big((n + k)\log n\big)
$$

where $k$ is the number of intersection points.

This variant is a direct application of the event-driven sweep line method specialized for segments.

#### What Problem Are We Solving?

Given $n$ line segments

$$
S = { s_1, s_2, \ldots, s_n }
$$

we want to compute the set of all intersection points between any two segments.
We need both which segments intersect and where.

#### Naive vs. Sweep Line

* Naive approach:
  Check all $\binom{n}{2}$ pairs → $O(n^2)$ time.
  Even for small $n$, this is wasteful when few intersections exist.

* Sweep Line (Bentley–Ottmann):

  * Process events in increasing $x$ order
  * Maintain active segments ordered by $y$
  * Only neighboring segments can intersect → local checks only

This turns a quadratic search into an output-sensitive algorithm.

#### How Does It Work (Plain Language)

We move a vertical sweep line from left to right, handling three event types:

| Event Type   | Description                                  |
| ------------ | -------------------------------------------- |
| Start        | Add segment to active set                    |
| End          | Remove segment from active set               |
| Intersection | Two segments cross; record point, swap order |

The active set is kept sorted by segment height ($y$) at the sweep line.
When a new segment is inserted, we only test its neighbors for intersection.
After a swap, we only test new adjacent pairs.

#### Example Walkthrough

Segments:

* $S_1: (0,0)$–$(4,4)$
* $S_2: (0,4)$–$(4,0)$
* $S_3: (1,0)$–$(1,3)$

Event queue (sorted by $x$):
$(0,0)$, $(0,4)$, $(1,0)$, $(1,3)$, $(4,0)$, $(4,4)$

1. $x=0$: Insert $S_1$, $S_2$ → check pair → intersection $(2,2)$ found.
2. $x=1$: Insert $S_3$, check with neighbors, no new intersections.
3. $x=2$: Process intersection $(2,2)$, swap order of $S_1$, $S_2$.
4. Continue → remove as sweep passes segment ends.

Output: intersection point $(2,2)$.

#### Geometric Test: Orientation

Given segments $AB$ and $CD$, they intersect if and only if

$$
\text{orient}(A, B, C) \ne \text{orient}(A, B, D)
$$

and

$$
\text{orient}(C, D, A) \ne \text{orient}(C, D, B)
$$

This uses cross product orientation to test if points are on opposite sides.

#### Tiny Code (Python)

```python
import heapq

def orient(a, b, c):
    return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])

def intersect(a, b, c, d):
    o1 = orient(a, b, c)
    o2 = orient(a, b, d)
    o3 = orient(c, d, a)
    o4 = orient(c, d, b)
    return o1*o2 < 0 and o3*o4 < 0

def bentley_ottmann(segments):
    events = []
    for s in segments:
        (x1,y1),(x2,y2) = s
        if x1 > x2:
            s = ((x2,y2),(x1,y1))
        events.append((x1, 'start', s))
        events.append((x2, 'end', s))
    heapq.heapify(events)

    active, intersections = [], []
    while events:
        x, typ, seg = heapq.heappop(events)
        if typ == 'start':
            active.append(seg)
            for other in active:
                if other != seg and intersect(seg[0], seg[1], other[0], other[1]):
                    intersections.append(x)
        elif typ == 'end':
            active.remove(seg)
    return intersections

segments = [((0,0),(4,4)), ((0,4),(4,0)), ((1,0),(1,3))]
print("Intersections:", bentley_ottmann(segments))
```

#### Why It Matters

* Output-sensitive: scales with actual number of intersections
* Core of geometry engines, CAD tools, and graphics pipelines
* Used in polygon clipping, mesh overlay, and map intersection

Applications:

* Detecting segment crossings in vector maps
* Overlaying geometric layers in GIS
* Path intersection detection (roads, wires, edges)
* Preprocessing for triangulation and visibility graphs

#### A Gentle Proof (Why It Works)

Every intersection event corresponds to a swap in the vertical order of segments.
Since order changes only at intersections, all are discovered by processing:

1. Insertions/Deletions (start/end events)
2. Swaps (intersection events)

We never miss or duplicate an intersection because only neighboring pairs can intersect between events.

Total operations:

* $n$ starts, $n$ ends, $k$ intersections → $O(n + k)$ events
* Each event uses $O(\log n)$ operations (heap/tree)

Therefore

$$
O\big((n + k)\log n\big)
$$

#### Try It Yourself

1. Draw segments with multiple crossings.
2. Sort endpoints by $x$.
3. Sweep and maintain ordered active set.
4. Record intersections as swaps occur.
5. Compare with brute-force pair checking.

#### Test Cases

| Segments            | Intersections |
| ------------------- | ------------- |
| Diagonals of square | 1             |
| Grid crossings      | Multiple      |
| Parallel lines      | 0             |
| Random segments     | Verified      |

#### Complexity

$$
\text{Time: } O((n + k)\log n), \quad \text{Space: } O(n + k)
$$

The Bentley–Ottmann variant of segment intersection is the benchmark technique, a precise dance of events and swaps that captures every crossing once, and only once.

### 725 Skyline Problem

The Skyline Problem is a classic geometric sweep line challenge: given a collection of rectangular buildings in a cityscape, compute the outline (or silhouette) that forms the skyline when viewed from afar.

This is a quintessential divide-and-conquer and line sweep example, converting overlapping rectangles into a piecewise height function that rises and falls as the sweep progresses.

#### What Problem Are We Solving?

Each building $B_i$ is defined by three numbers:

$$
B_i = (x_{\text{left}}, x_{\text{right}}, h)
$$

We want to compute the skyline, a sequence of critical points:

$$
$$(x_1, h_1), (x_2, h_2), \ldots, (x_m, 0)]
$$

such that the upper contour of all buildings is traced exactly once.

Example input:

| Building | Left | Right | Height |
| -------- | ---- | ----- | ------ |
| 1        | 2    | 9     | 10     |
| 2        | 3    | 7     | 15     |
| 3        | 5    | 12    | 12     |

Output:

$$
$$(2,10), (3,15), (7,12), (12,0)]
$$

#### How Does It Work (Plain Language)

The skyline changes only at building edges, left or right sides.
We treat each edge as an event in a sweep line moving from left to right:

1. At left edge ($x_\text{left}$): add building height to active set.
2. At right edge ($x_\text{right}$): remove height from active set.
3. After each event, the skyline height = max(active set).
4. If height changes, append $(x, h)$ to result.

This efficiently constructs the outline by tracking current tallest building.

#### Example Walkthrough

Input:
$(2,9,10), (3,7,15), (5,12,12)$

Events:

* (2, start, 10)
* (3, start, 15)
* (5, start, 12)
* (7, end, 15)
* (9, end, 10)
* (12, end, 12)

Steps:

| x  | Event    | Active Heights | Max Height | Output |
| -- | -------- | -------------- | ---------- | ------ |
| 2  | Start 10 | {10}           | 10         | (2,10) |
| 3  | Start 15 | {10,15}        | 15         | (3,15) |
| 5  | Start 12 | {10,15,12}     | 15         | –      |
| 7  | End 15   | {10,12}        | 12         | (7,12) |
| 9  | End 10   | {12}           | 12         | –      |
| 12 | End 12   | {}             | 0          | (12,0) |

Output skyline:
$$
$$(2,10), (3,15), (7,12), (12,0)]
$$

#### Tiny Code (Python)

```python
import heapq

def skyline(buildings):
    events = []
    for L, R, H in buildings:
        events.append((L, -H))  # start
        events.append((R, H))   # end
    events.sort()

    result = []
    heap = [0]  # max-heap (store negatives)
    prev_max = 0
    active = {}

    for x, h in events:
        if h < 0:  # start
            heapq.heappush(heap, h)
        else:      # end
            active[h] = active.get(h, 0) + 1  # mark for removal
        # Clean up ended heights
        while heap and active.get(-heap[0], 0):
            active[-heap[0]] -= 1
            if active[-heap[0]] == 0:
                del active[-heap[0]]
            heapq.heappop(heap)
        curr_max = -heap[0]
        if curr_max != prev_max:
            result.append((x, curr_max))
            prev_max = curr_max
    return result

buildings = [(2,9,10), (3,7,15), (5,12,12)]
print("Skyline:", skyline(buildings))
```

#### Tiny Code (C, Conceptual)

```c
typedef struct { int x, h, type; } Event;
```

1. Sort events by $x$ (and height for tie-breaking).
2. Use balanced tree (multiset) to maintain active heights.
3. On start, insert height; on end, remove height.
4. Record changes in max height as output points.

#### Why It Matters

* Demonstrates event-based sweeping with priority queues
* Core in rendering, city modeling, interval aggregation
* Dual of rectangle union, here we care about upper contour, not area

Applications:

* Cityscape rendering
* Range aggregation visualization
* Histogram or bar merge outlines
* Shadow or coverage profiling

#### A Gentle Proof (Why It Works)

The skyline changes only at edges, since interior points are covered continuously.
Between edges, the set of active buildings is constant, so the max height is constant.

By processing all edges in order and recording each height change, we reconstruct the exact upper envelope.

Each insertion/removal is $O(\log n)$ (heap), and there are $2n$ events:

$$
O(n \log n)
$$

#### Try It Yourself

1. Draw 2–3 overlapping buildings.
2. Sort all edges by $x$.
3. Sweep and track active heights.
4. Record output every time the max height changes.
5. Verify with manual outline tracing.

#### Test Cases

| Buildings                   | Skyline                       |
| --------------------------- | ----------------------------- |
| (2,9,10),(3,7,15),(5,12,12) | (2,10),(3,15),(7,12),(12,0)   |
| (1,3,3),(2,4,4),(5,6,1)     | (1,3),(2,4),(4,0),(5,1),(6,0) |
| (1,2,1),(2,3,2),(3,4,3)     | (1,1),(2,2),(3,3),(4,0)       |

#### Complexity

$$
\text{Time: } O(n \log n), \quad \text{Space: } O(n)
$$

The Skyline Problem captures the rising and falling rhythm of geometry, a stepwise silhouette built from overlapping shapes and the elegance of sweeping through events.

### 726 Closest Pair Sweep

The Closest Pair Sweep algorithm finds the minimum distance between any two points in the plane using a sweep line and an active set.
It's one of the most elegant examples of combining sorting, geometry, and locality, transforming an $O(n^2)$ search into an $O(n \log n)$ algorithm.

#### What Problem Are We Solving?

Given $n$ points $P = {p_1, p_2, \ldots, p_n}$ in the plane,
find two distinct points $(p_i, p_j)$ such that their Euclidean distance is minimal:

$$
d(p_i, p_j) = \sqrt{(x_i - x_j)^2 + (y_i - y_j)^2}
$$

We want both the distance and the pair that achieves it.

A naive $O(n^2)$ algorithm checks all pairs.
We'll do better using a sweep line and spatial pruning.

#### How Does It Work (Plain Language)

The key insight:
When sweeping from left to right, only points within a narrow vertical strip can be the closest pair.

Algorithm outline:

1. Sort points by $x$-coordinate.
2. Maintain an active set (ordered by $y$) of points within the current strip width (equal to best distance found so far).
3. For each new point:

   * Remove points with $x$ too far left.
   * Compare only to points within $\delta$ vertically, where $\delta$ is current best distance.
   * Update best distance if smaller found.
4. Continue until all points processed.

This works because in a $\delta \times 2\delta$ strip, at most 6 points can be close enough to improve the best distance.

#### Example Walkthrough

Points:
$$
P = {(1,1), (2,3), (3,2), (5,5)}
$$

1. Sort by $x$: $(1,1), (2,3), (3,2), (5,5)$
2. Start with first point $(1,1)$
3. Add $(2,3)$ → $d=\sqrt{5}$
4. Add $(3,2)$ → compare with last 2 points

   * $d((2,3),(3,2)) = \sqrt{2}$ → best $\delta = \sqrt{2}$
5. Add $(5,5)$ → $x$ difference > $\delta$ from $(1,1)$, remove it

   * Compare $(5,5)$ with $(2,3),(3,2)$ → no smaller found

Output: closest pair $(2,3),(3,2)$, distance $\sqrt{2}$.

#### Tiny Code (Python)

```python
from math import sqrt
import bisect

def dist(a, b):
    return sqrt((a[0]-b[0])2 + (a[1]-b[1])2)

def closest_pair(points):
    points.sort()  # sort by x
    best = float('inf')
    best_pair = None
    active = []  # sorted by y
    j = 0

    for i, p in enumerate(points):
        x, y = p
        while (x - points[j][0]) > best:
            active.remove(points[j])
            j += 1
        pos = bisect.bisect_left(active, (y - best, -float('inf')))
        while pos < len(active) and active[pos][0] <= y + best:
            d = dist(p, active[pos])
            if d < best:
                best, best_pair = d, (p, active[pos])
            pos += 1
        bisect.insort(active, p)
    return best, best_pair

points = [(1,1), (2,3), (3,2), (5,5)]
print("Closest pair:", closest_pair(points))
```

#### Tiny Code (C, Conceptual Sketch)

```c
typedef struct { double x, y; } Point;
double dist(Point a, Point b) {
    double dx = a.x - b.x, dy = a.y - b.y;
    return sqrt(dx*dx + dy*dy);
}
// Sort by x, maintain active set (balanced BST by y)
// For each new point, remove far x, search nearby y, update best distance
```

Efficient implementations use balanced search trees or ordered lists.

#### Why It Matters

* Classic in computational geometry
* Combines sorting + sweeping + local search
* Model for spatial algorithms using geometric pruning

Applications:

* Nearest neighbor search
* Clustering and pattern recognition
* Motion planning (min separation)
* Spatial indexing and range queries

#### A Gentle Proof (Why It Works)

At each step, points farther than $\delta$ in $x$ cannot improve the best distance.
In the $\delta$-strip, each point has at most 6 neighbors (packing argument in a $\delta \times \delta$ grid).
Thus, total comparisons are linear after sorting.

Overall complexity:

$$
O(n \log n)
$$

from initial sort and logarithmic insertions/removals.

#### Try It Yourself

1. Plot a few points.
2. Sort by $x$.
3. Sweep from left to right.
4. Keep strip width $\delta$, check only local neighbors.
5. Compare with brute-force for verification.

#### Test Cases

| Points                  | Closest Pair | Distance            |
| ----------------------- | ------------ | ------------------- |
| (1,1),(2,3),(3,2),(5,5) | (2,3),(3,2)  | $\sqrt{2}$          |
| (0,0),(1,0),(2,0)       | (0,0),(1,0)  | 1                   |
| Random 10 points        | Verified     | Matches brute force |

#### Complexity

$$
\text{Time: } O(n \log n), \quad \text{Space: } O(n)
$$

The Closest Pair Sweep is geometry's precision tool, narrowing the search to a moving strip and comparing only those neighbors that truly matter.

### 727 Circle Arrangement Sweep

The Circle Arrangement Sweep algorithm computes the arrangement of a set of circles, the subdivision of the plane induced by all circle arcs and their intersection points.
It's a generalization of line and segment arrangements, extended to curved edges, requiring event-driven sweeping and geometric reasoning.

#### What Problem Are We Solving?

Given $n$ circles
$$
C_i: (x_i, y_i, r_i)
$$
we want to compute their arrangement: the decomposition of the plane into faces, edges, and vertices formed by intersections between circles.

A simpler variant focuses on counting intersections and constructing intersection points.

Each pair of circles can intersect at most two points, so there can be at most

$$
O(n^2)
$$
intersection points.

#### Why It's Harder Than Lines

* A circle introduces nonlinear boundaries.
* The sweep line must handle arc segments, not just straight intervals.
* Events occur at circle start/end x-coordinates and at intersection points.

This means each circle enters and exits the sweep twice, and new intersections can emerge dynamically.

#### How Does It Work (Plain Language)

The sweep line moves left to right, intersecting circles as vertical slices.
We maintain an active set of circle arcs currently intersecting the line.

At each event:

1. Leftmost point ($x_i - r_i$): insert circle arc.
2. Rightmost point ($x_i + r_i$): remove circle arc.
3. Intersection points: when two arcs cross, schedule intersection event.

Each time arcs are inserted or swapped, check local neighbors for intersections (like Bentley–Ottmann, but with curved segments).

#### Circle–Circle Intersection Formula

Two circles:

$$
(x_1, y_1, r_1), \quad (x_2, y_2, r_2)
$$

Distance between centers:

$$
d = \sqrt{(x_2 - x_1)^2 + (y_2 - y_1)^2}
$$

If $|r_1 - r_2| \le d \le r_1 + r_2$, they intersect at two points:

$$
a = \frac{r_1^2 - r_2^2 + d^2}{2d}
$$

$$
h = \sqrt{r_1^2 - a^2}
$$

Then intersection coordinates:

$$
x_3 = x_1 + a \cdot \frac{x_2 - x_1}{d} \pm h \cdot \frac{y_2 - y_1}{d}
$$

$$
y_3 = y_1 + a \cdot \frac{y_2 - y_1}{d} \mp h \cdot \frac{x_2 - x_1}{d}
$$

Each intersection point becomes an event in the sweep.

#### Example (3 Circles)

Circles:

* $C_1: (0,0,2)$
* $C_2: (3,0,2)$
* $C_3: (1.5,2,1.5)$

Each pair intersects in 2 points → up to 6 intersection points.
The arrangement has vertices (intersections), edges (arcs), and faces (regions).

The sweep processes:

* $x = -2$: $C_1$ starts
* $x = 1$: $C_2$ starts
* $x = 1.5$: intersection events
* $x = 3$: $C_3$ starts
* $x = 5$: circles end

#### Tiny Code (Python Sketch)

```python
from math import sqrt

def circle_intersections(c1, c2):
    (x1, y1, r1), (x2, y2, r2) = c1, c2
    dx, dy = x2 - x1, y2 - y1
    d = sqrt(dx*dx + dy*dy)
    if d > r1 + r2 or d < abs(r1 - r2) or d == 0:
        return []
    a = (r1*r1 - r2*r2 + d*d) / (2*d)
    h = sqrt(r1*r1 - a*a)
    xm = x1 + a * dx / d
    ym = y1 + a * dy / d
    xs1 = xm + h * dy / d
    ys1 = ym - h * dx / d
    xs2 = xm - h * dy / d
    ys2 = ym + h * dx / d
    return [(xs1, ys1), (xs2, ys2)]

def circle_arrangement(circles):
    events = []
    for i, c1 in enumerate(circles):
        for j, c2 in enumerate(circles[i+1:], i+1):
            pts = circle_intersections(c1, c2)
            events.extend(pts)
    return sorted(events)

circles = [(0,0,2), (3,0,2), (1.5,2,1.5)]
print("Intersections:", circle_arrangement(circles))
```

This simplified version enumerates intersection points, suitable for event scheduling.

#### Why It Matters

* Foundation for geometric arrangements with curved objects
* Used in motion planning, robotics, cellular coverage, CAD
* Step toward full algebraic geometry arrangements (conics, ellipses)

Applications:

* Cellular network planning (coverage overlaps)
* Path regions for robots
* Venn diagrams and spatial reasoning
* Graph embedding on circular arcs

#### A Gentle Proof (Why It Works)

Each circle adds at most two intersections with others;
Each intersection event is processed once;
At most $O(n^2)$ intersections, each with $O(\log n)$ handling (tree insertion/removal).

Therefore:

$$
O(n^2 \log n)
$$

The correctness follows from local adjacency: only neighboring arcs can swap during events, so all intersections are captured.

#### Try It Yourself

1. Draw 3 circles overlapping partially.
2. Compute pairwise intersections.
3. Mark points, connect arcs in clockwise order.
4. Sweep from leftmost to rightmost.
5. Count faces (regions) formed.

#### Test Cases

| Circles       | Intersections | Faces     |
| ------------- | ------------- | --------- |
| 2 overlapping | 2             | 3 regions |
| 3 overlapping | 6             | 8 regions |
| Disjoint      | 0             | n regions |

#### Complexity

$$
\text{Time: } O(n^2 \log n), \quad \text{Space: } O(n^2)
$$

The Circle Arrangement Sweep transforms smooth geometry into discrete structure, every arc, crossing, and face traced by a patient sweep across the plane.

### 728 Sweep for Overlapping Rectangles

The Sweep for Overlapping Rectangles algorithm detects intersections or collisions among a set of axis-aligned rectangles.
It's a practical and elegant use of line sweep methods for 2D collision detection, spatial joins, and layout engines.

#### What Problem Are We Solving?

Given $n$ axis-aligned rectangles

$$
R_i = [x_{1i}, x_{2i}] \times [y_{1i}, y_{2i}]
$$

we want to find all pairs $(R_i, R_j)$ that overlap, meaning

$$
 [x_{1i}, x_{2i}] \cap [x_{1j}, x_{2j}] \ne \emptyset
$$
and
$$
 [y_{1i}, y_{2i}] \cap [y_{1j}, y_{2j}] \ne \emptyset
$$

This is a common subproblem in graphics, GIS, and physics engines.

#### Naive Approach

Check every pair of rectangles:
$$
O(n^2)
$$

Too slow when $n$ is large.

We'll use a sweep line along $x$, maintaining an active set of rectangles whose x-intervals overlap the current position.

#### How Does It Work (Plain Language)

We process events in increasing $x$:

* Start event: at $x_{1i}$, rectangle enters active set.
* End event: at $x_{2i}$, rectangle leaves active set.

At each insertion, we check new rectangle against all active rectangles for y-overlap.

Because active rectangles all overlap in $x$, we only need to test $y$-intervals.

#### Example Walkthrough

Rectangles:

| ID | $x_1$ | $x_2$ | $y_1$ | $y_2$ |
| -- | ----- | ----- | ----- | ----- |
| R1 | 1     | 4     | 1     | 3     |
| R2 | 2     | 5     | 2     | 4     |
| R3 | 6     | 8     | 0     | 2     |

Events (sorted by $x$):
$(1,\text{start},R1)$, $(2,\text{start},R2)$, $(4,\text{end},R1)$, $(5,\text{end},R2)$, $(6,\text{start},R3)$, $(8,\text{end},R3)$

Sweep:

1. $x=1$: Add R1 → active = {R1}.
2. $x=2$: Add R2 → check overlap with R1:

   * $[1,3] \cap [2,4] = [2,3] \ne \emptyset$ → overlap found (R1, R2).
3. $x=4$: Remove R1.
4. $x=5$: Remove R2.
5. $x=6$: Add R3.
6. $x=8$: Remove R3.

Output: Overlap pair (R1, R2).

#### Overlap Condition

Two rectangles $R_i, R_j$ overlap iff

$$
x_{1i} < x_{2j} \ \text{and}\ x_{2i} > x_{1j}
$$

and

$$
y_{1i} < y_{2j} \ \text{and}\ y_{2i} > y_{1j}
$$

#### Tiny Code (Python)

```python
def overlaps(r1, r2):
    return not (r1[1] <= r2[0] or r2[1] <= r1[0] or 
                r1[3] <= r2[2] or r2[3] <= r1[2])

def sweep_rectangles(rects):
    events = []
    for i, (x1, x2, y1, y2) in enumerate(rects):
        events.append((x1, 'start', i))
        events.append((x2, 'end', i))
    events.sort()
    active = []
    result = []
    for x, typ, idx in events:
        if typ == 'start':
            for j in active:
                if overlaps(rects[idx], rects[j]):
                    result.append((idx, j))
            active.append(idx)
        else:
            active.remove(idx)
    return result

rects = [(1,4,1,3),(2,5,2,4),(6,8,0,2)]
print("Overlaps:", sweep_rectangles(rects))
```

#### Tiny Code (C Sketch)

```c
typedef struct { double x1, x2, y1, y2; } Rect;
int overlaps(Rect a, Rect b) {
    return !(a.x2 <= b.x1 || b.x2 <= a.x1 ||
             a.y2 <= b.y1 || b.y2 <= a.y1);
}
```

Use an array of events, sort by $x$, maintain active list.

#### Why It Matters

* Core idea behind broad-phase collision detection
* Used in 2D games, UI layout engines, spatial joins
* Extends easily to 3D box intersection via multi-axis sweep

Applications:

* Physics simulations (bounding box overlap)
* Spatial query systems (R-tree verification)
* CAD layout constraint checking

#### A Gentle Proof (Why It Works)

* The active set contains exactly rectangles that overlap current $x$.
* By checking only these, we cover all possible overlaps once.
* Each insertion/removal: $O(\log n)$ (with balanced tree).
* Each pair tested only when $x$-ranges overlap.

Total time:

$$
O((n + k) \log n)
$$

where $k$ is number of overlaps.

#### Try It Yourself

1. Draw overlapping rectangles on a grid.
2. Sort edges by $x$.
3. Sweep and maintain active list.
4. At each insertion, test $y$-overlap with actives.
5. Record overlaps, verify visually.

#### Test Cases

| Rectangles               | Overlaps        |
| ------------------------ | --------------- |
| R1(1,4,1,3), R2(2,5,2,4) | (R1,R2)         |
| Disjoint rectangles      | None            |
| Nested rectangles        | All overlapping |

#### Complexity

$$
\text{Time: } O((n + k)\log n), \quad \text{Space: } O(n)
$$

The Sweep for Overlapping Rectangles is a geometric sentinel, sliding across the plane, keeping track of active shapes, and spotting collisions with precision.

### 729 Range Counting

Range Counting asks: given many points in the plane, how many lie inside an axis-aligned query rectangle.
It is a staple of geometric data querying, powering interactive plots, maps, and database indices.

#### What Problem Are We Solving?

Input: a static set of $n$ points $P = {(x_i,y_i)}_{i=1}^n$.
Queries: for rectangles $R = [x_L, x_R] \times [y_B, y_T]$, return

$$
\#\{(x,y) \in P \mid x_L \le x \le x_R,\; y_B \le y \le y_T\}.
$$


We want fast query time, ideally sublinear, after a one-time preprocessing step.

#### How Does It Work (Plain Language)

Several classic structures support orthogonal range counting.

1. Sorted by x + Fenwick over y (offline or sweep):
   Sort points by $x$. Sort queries by $x_R$. Sweep in $x$, adding points to a Fenwick tree keyed by their compressed $y$.
   The count for $[x_L,x_R]\times[y_B,y_T]$ equals:
   $$
   \text{count}(x \le x_R, y \in [y_B,y_T]) - \text{count}(x < x_L, y \in [y_B,y_T]).
   $$
   Time: $O((n + q)\log n)$ offline.

2. Range Tree (static, online):
   Build a balanced BST on $x$. Each node stores a sorted list of the $y$ values in its subtree.
   A 2D query decomposes the $x$-range into $O(\log n)$ canonical nodes, and in each node we binary search the $y$ list to count how many lie in $[y_B,y_T]$.
   Time: query $O(\log^2 n)$, space $O(n \log n)$.
   With fractional cascading, query improves to $O(\log n)$.

3. Fenwick of Fenwicks or Segment tree of Fenwicks:
   Index by $x$ with a Fenwick tree. Each Fenwick node stores another Fenwick over $y$.
   Fully online updates and queries in $O(\log^2 n)$ with $O(n \log n)$ space after coordinate compression.

#### Example Walkthrough

Points: $(1,1), (2,3), (3,2), (5,4), (6,1)$
Query: $R = [2,5] \times [2,4]$

Points inside: $(2,3), (3,2), (5,4)$
Answer: $3$.

#### Tiny Code 1: Offline Sweep with Fenwick (Python)

```python
# Offline orthogonal range counting:
# For each query [xL,xR]x[yB,yT], compute F(xR, yB..yT) - F(xL-ε, yB..yT)

from bisect import bisect_left, bisect_right

class Fenwick:
    def __init__(self, n):
        self.n = n
        self.fw = [0]*(n+1)
    def add(self, i, v=1):
        while i <= self.n:
            self.fw[i] += v
            i += i & -i
    def sum(self, i):
        s = 0
        while i > 0:
            s += self.fw[i]
            i -= i & -i
        return s
    def range_sum(self, l, r):
        if r < l: return 0
        return self.sum(r) - self.sum(l-1)

def offline_range_count(points, queries):
    # points: list of (x,y)
    # queries: list of (xL,xR,yB,yT)
    ys = sorted({y for _,y in points} | {q[2] for q in queries} | {q[3] for q in queries})
    def y_id(y): return bisect_left(ys, y) + 1

    # prepare events: add points up to x, then answer queries ending at that x
    events = []
    for i,(x,y) in enumerate(points):
        events.append((x, 0, i))  # point event
    Fq = []  # queries on xR
    Gq = []  # queries on xL-1
    for qi,(xL,xR,yB,yT) in enumerate(queries):
        Fq.append((xR, 1, qi))
        Gq.append((xL-1, 2, qi))
    events += Fq + Gq
    events.sort()

    fw = Fenwick(len(ys))
    ansR = [0]*len(queries)
    ansL = [0]*len(queries)

    for x,typ,idx in events:
        if typ == 0:
            _,y = points[idx]
            fw.add(y_id(y), 1)
        elif typ == 1:
            xL,xR,yB,yT = queries[idx]
            l = bisect_left(ys, yB) + 1
            r = bisect_right(ys, yT)
            ansR[idx] = fw.range_sum(l, r)
        else:
            xL,xR,yB,yT = queries[idx]
            l = bisect_left(ys, yB) + 1
            r = bisect_right(ys, yT)
            ansL[idx] = fw.range_sum(l, r)

    return [ansR[i] - ansL[i] for i in range(len(queries))]

# demo
points = [(1,1),(2,3),(3,2),(5,4),(6,1)]
queries = [(2,5,2,4), (1,6,1,1)]
print(offline_range_count(points, queries))  # [3, 2]
```

#### Tiny Code 2: Static Range Tree Query Idea (Python, conceptual)

```python
# Build: sort points by x, recursively split;
# at each node store the y-sorted list for binary counting.

from bisect import bisect_left, bisect_right

class RangeTree:
    def __init__(self, pts):
        # pts sorted by x
        self.xs = [p[0] for p in pts]
        self.ys = sorted(p[1] for p in pts)
        self.left = self.right = None
        if len(pts) > 1:
            mid = len(pts)//2
            self.left = RangeTree(pts[:mid])
            self.right = RangeTree(pts[mid:])

    def count_y(self, yB, yT):
        L = bisect_left(self.ys, yB)
        R = bisect_right(self.ys, yT)
        return R - L

    def query(self, xL, xR, yB, yT):
        # count points with x in [xL,xR] and y in [yB,yT]
        if xR < self.xs[0] or xL > self.xs[-1]:
            return 0
        if xL <= self.xs[0] and self.xs[-1] <= xR:
            return self.count_y(yB, yT)
        if not self.left:  # leaf
            return int(xL <= self.xs[0] <= xR and yB <= self.ys[0] <= yT)
        return self.left.query(xL,xR,yB,yT) + self.right.query(xL,xR,yB,yT)

pts = sorted([(1,1),(2,3),(3,2),(5,4),(6,1)])
rt = RangeTree(pts)
print(rt.query(2,5,2,4))  # 3
```

#### Why It Matters

* Core primitive for spatial databases and analytic dashboards
* Underlies heatmaps, density queries, and windowed aggregations
* Extends to higher dimensions with $k$-d trees and range trees

Applications:
map viewports, time window counts, GIS filtering, interactive brushing and linking.

#### A Gentle Proof (Why It Works)

For the range tree: the $x$-range $[x_L,x_R]$ decomposes into $O(\log n)$ canonical nodes of a balanced BST.
Each canonical node stores its subtree's $y$ values in sorted order.
Counting in $[y_B,y_T]$ at a node costs $O(\log n)$ by binary searches.
Summing over $O(\log n)$ nodes yields $O(\log^2 n)$ per query.
With fractional cascading, the second-level searches reuse pointers so all counts are found in $O(\log n)$.

#### Try It Yourself

1. Implement offline counting with a Fenwick tree and coordinate compression.
2. Compare against naive $O(n)$ per query to verify.
3. Build a range tree and time $q$ queries for varying $n$.
4. Add updates: switch to Fenwick of Fenwicks for dynamic points.
5. Extend to 3D with a tree of trees for orthogonal boxes.

#### Test Cases

| Points                          | Query rectangle           | Expected |
| ------------------------------- | ------------------------- | -------- |
| $(1,1),(2,3),(3,2),(5,4),(6,1)$ | $[2,5]\times[2,4]$        | 3        |
| same                            | $[1,6]\times[1,1]$        | 2        |
| $(0,0),(10,10)$                 | $[1,9]\times[1,9]$        | 0        |
| grid $3\times 3$                | center $[1,2]\times[1,2]$ | 4        |

#### Complexity

* Offline sweep with Fenwick: preprocessing plus queries in $O((n+q)\log n)$
* Range tree: build $O(n \log n)$, query $O(\log^2 n)$ or $O(\log n)$ with fractional cascading
* Segment or Fenwick of Fenwicks: dynamic updates and queries in $O(\log^2 n)$

Range counting turns spatial selection into log-time queries by layering search trees and sorted auxiliary lists.

### 729 Range Counting

Range Counting asks: given many points in the plane, how many lie inside an axis-aligned query rectangle.
It is a staple of geometric data querying, powering interactive plots, maps, and database indices.

#### What Problem Are We Solving?

Input: a static set of $n$ points $P = {(x_i,y_i)}_{i=1}^n$.
Queries: for rectangles $R = [x_L, x_R] \times [y_B, y_T]$, return

$$
\#\{(x,y) \in P \mid x_L \le x \le x_R,\; y_B \le y \le y_T\}.
$$


We want fast query time, ideally sublinear, after a one-time preprocessing step.

#### How Does It Work (Plain Language)

Several classic structures support orthogonal range counting.

1. Sorted by x + Fenwick over y (offline or sweep):
   Sort points by $x$. Sort queries by $x_R$. Sweep in $x$, adding points to a Fenwick tree keyed by their compressed $y$.
   The count for $[x_L,x_R]\times[y_B,y_T]$ equals:
   $$
   \text{count}(x \le x_R, y \in [y_B,y_T]) - \text{count}(x < x_L, y \in [y_B,y_T]).
   $$
   Time: $O((n + q)\log n)$ offline.

2. Range Tree (static, online):
   Build a balanced BST on $x$. Each node stores a sorted list of the $y$ values in its subtree.
   A 2D query decomposes the $x$-range into $O(\log n)$ canonical nodes, and in each node we binary search the $y$ list to count how many lie in $[y_B,y_T]$.
   Time: query $O(\log^2 n)$, space $O(n \log n)$.
   With fractional cascading, query improves to $O(\log n)$.

3. Fenwick of Fenwicks or Segment tree of Fenwicks:
   Index by $x$ with a Fenwick tree. Each Fenwick node stores another Fenwick over $y$.
   Fully online updates and queries in $O(\log^2 n)$ with $O(n \log n)$ space after coordinate compression.

#### Example Walkthrough

Points: $(1,1), (2,3), (3,2), (5,4), (6,1)$
Query: $R = [2,5] \times [2,4]$

Points inside: $(2,3), (3,2), (5,4)$
Answer: $3$.

#### Tiny Code 1: Offline Sweep with Fenwick (Python)

```python
# Offline orthogonal range counting:
# For each query [xL,xR]x[yB,yT], compute F(xR, yB..yT) - F(xL-ε, yB..yT)

from bisect import bisect_left, bisect_right

class Fenwick:
    def __init__(self, n):
        self.n = n
        self.fw = [0]*(n+1)
    def add(self, i, v=1):
        while i <= self.n:
            self.fw[i] += v
            i += i & -i
    def sum(self, i):
        s = 0
        while i > 0:
            s += self.fw[i]
            i -= i & -i
        return s
    def range_sum(self, l, r):
        if r < l: return 0
        return self.sum(r) - self.sum(l-1)

def offline_range_count(points, queries):
    # points: list of (x,y)
    # queries: list of (xL,xR,yB,yT)
    ys = sorted({y for _,y in points} | {q[2] for q in queries} | {q[3] for q in queries})
    def y_id(y): return bisect_left(ys, y) + 1

    # prepare events: add points up to x, then answer queries ending at that x
    events = []
    for i,(x,y) in enumerate(points):
        events.append((x, 0, i))  # point event
    Fq = []  # queries on xR
    Gq = []  # queries on xL-1
    for qi,(xL,xR,yB,yT) in enumerate(queries):
        Fq.append((xR, 1, qi))
        Gq.append((xL-1, 2, qi))
    events += Fq + Gq
    events.sort()

    fw = Fenwick(len(ys))
    ansR = [0]*len(queries)
    ansL = [0]*len(queries)

    for x,typ,idx in events:
        if typ == 0:
            _,y = points[idx]
            fw.add(y_id(y), 1)
        elif typ == 1:
            xL,xR,yB,yT = queries[idx]
            l = bisect_left(ys, yB) + 1
            r = bisect_right(ys, yT)
            ansR[idx] = fw.range_sum(l, r)
        else:
            xL,xR,yB,yT = queries[idx]
            l = bisect_left(ys, yB) + 1
            r = bisect_right(ys, yT)
            ansL[idx] = fw.range_sum(l, r)

    return [ansR[i] - ansL[i] for i in range(len(queries))]

# demo
points = [(1,1),(2,3),(3,2),(5,4),(6,1)]
queries = [(2,5,2,4), (1,6,1,1)]
print(offline_range_count(points, queries))  # [3, 2]
```

#### Tiny Code 2: Static Range Tree Query Idea (Python, conceptual)

```python
# Build: sort points by x, recursively split;
# at each node store the y-sorted list for binary counting.

from bisect import bisect_left, bisect_right

class RangeTree:
    def __init__(self, pts):
        # pts sorted by x
        self.xs = [p[0] for p in pts]
        self.ys = sorted(p[1] for p in pts)
        self.left = self.right = None
        if len(pts) > 1:
            mid = len(pts)//2
            self.left = RangeTree(pts[:mid])
            self.right = RangeTree(pts[mid:])

    def count_y(self, yB, yT):
        L = bisect_left(self.ys, yB)
        R = bisect_right(self.ys, yT)
        return R - L

    def query(self, xL, xR, yB, yT):
        # count points with x in [xL,xR] and y in [yB,yT]
        if xR < self.xs[0] or xL > self.xs[-1]:
            return 0
        if xL <= self.xs[0] and self.xs[-1] <= xR:
            return self.count_y(yB, yT)
        if not self.left:  # leaf
            return int(xL <= self.xs[0] <= xR and yB <= self.ys[0] <= yT)
        return self.left.query(xL,xR,yB,yT) + self.right.query(xL,xR,yB,yT)

pts = sorted([(1,1),(2,3),(3,2),(5,4),(6,1)])
rt = RangeTree(pts)
print(rt.query(2,5,2,4))  # 3
```

#### Why It Matters

* Core primitive for spatial databases and analytic dashboards
* Underlies heatmaps, density queries, and windowed aggregations
* Extends to higher dimensions with $k$-d trees and range trees

Applications:
map viewports, time window counts, GIS filtering, interactive brushing and linking.

#### A Gentle Proof (Why It Works)

For the range tree: the $x$-range $[x_L,x_R]$ decomposes into $O(\log n)$ canonical nodes of a balanced BST.
Each canonical node stores its subtree's $y$ values in sorted order.
Counting in $[y_B,y_T]$ at a node costs $O(\log n)$ by binary searches.
Summing over $O(\log n)$ nodes yields $O(\log^2 n)$ per query.
With fractional cascading, the second-level searches reuse pointers so all counts are found in $O(\log n)$.

#### Try It Yourself

1. Implement offline counting with a Fenwick tree and coordinate compression.
2. Compare against naive $O(n)$ per query to verify.
3. Build a range tree and time $q$ queries for varying $n$.
4. Add updates: switch to Fenwick of Fenwicks for dynamic points.
5. Extend to 3D with a tree of trees for orthogonal boxes.

#### Test Cases

| Points                          | Query rectangle           | Expected |
| ------------------------------- | ------------------------- | -------- |
| $(1,1),(2,3),(3,2),(5,4),(6,1)$ | $[2,5]\times[2,4]$        | 3        |
| same                            | $[1,6]\times[1,1]$        | 2        |
| $(0,0),(10,10)$                 | $[1,9]\times[1,9]$        | 0        |
| grid $3\times 3$                | center $[1,2]\times[1,2]$ | 4        |

#### Complexity

* Offline sweep with Fenwick: preprocessing plus queries in $O((n+q)\log n)$
* Range tree: build $O(n \log n)$, query $O(\log^2 n)$ or $O(\log n)$ with fractional cascading
* Segment or Fenwick of Fenwicks: dynamic updates and queries in $O(\log^2 n)$

Range counting turns spatial selection into log-time queries by layering search trees and sorted auxiliary lists.

### 730 Plane Sweep for Triangles

The Plane Sweep for Triangles algorithm computes intersections, overlaps, or arrangements among a collection of triangles in the plane.
It extends line- and segment-based sweeps to polygonal elements, managing both edges and faces as events.

#### What Problem Are We Solving?

Given $n$ triangles
$$
T_i = {(x_{i1}, y_{i1}), (x_{i2}, y_{i2}), (x_{i3}, y_{i3})}
$$
we want to compute:

* All intersections among triangle edges
* Overlapping regions (union area or intersection polygons)
* Overlay decomposition: the full planar subdivision induced by triangle boundaries

Such sweeps are essential in mesh overlay, computational geometry kernels, and computer graphics.

#### Naive Approach

Compare all triangle pairs $(T_i, T_j)$ and their 9 edge pairs.
Time:
$$
O(n^2)
$$
Too expensive for large meshes or spatial data.

We improve using plane sweep over edges and events.

#### How Does It Work (Plain Language)

A triangle is composed of 3 line segments.
We treat every triangle edge as a segment event and process with a segment sweep line:

1. Convert all triangle edges into a list of segments.
2. Sort all segment endpoints by $x$.
3. Sweep line moves left to right.
4. Maintain an active set of edges intersecting the sweep.
5. When two edges intersect, record intersection point and, if needed, subdivide geometry.

If computing overlay, intersections subdivide triangles into planar faces.

#### Example Walkthrough

Triangles:

* $T_1$: $(1,1)$, $(4,1)$, $(2,3)$
* $T_2$: $(2,0)$, $(5,2)$, $(3,4)$

1. Extract edges:

   * $T_1$: $(1,1)-(4,1)$, $(4,1)-(2,3)$, $(2,3)-(1,1)$
   * $T_2$: $(2,0)-(5,2)$, $(5,2)-(3,4)$, $(3,4)-(2,0)$

2. Collect all endpoints, sort by $x$:
   $x = 1, 2, 3, 4, 5$

3. Sweep:

   * $x=1$: add edges from $T_1$
   * $x=2$: add edges from $T_2$; check intersections with current active set
   * find intersection between $T_1$'s sloping edge and $T_2$'s base edge
   * record intersection
   * update geometry if overlay needed

Output: intersection point(s), overlapping region polygon.

#### Geometric Predicates

For edges $(A,B)$ and $(C,D)$:
check intersection with orientation tests:

$$
\text{orient}(A,B,C) \ne \text{orient}(A,B,D)
$$
and
$$
\text{orient}(C,D,A) \ne \text{orient}(C,D,B)
$$

Intersections subdivide edges and update event queue.

#### Tiny Code (Python Sketch)

```python
def orient(a, b, c):
    return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])

def intersect(a,b,c,d):
    o1 = orient(a,b,c)
    o2 = orient(a,b,d)
    o3 = orient(c,d,a)
    o4 = orient(c,d,b)
    return o1*o2 < 0 and o3*o4 < 0

def sweep_triangles(triangles):
    segments = []
    for tri in triangles:
        for i in range(3):
            a, b = tri[i], tri[(i+1)%3]
            if a[0] > b[0]:
                a, b = b, a
            segments.append((a,b))
    events = []
    for s in segments:
        events.append((s[0][0],'start',s))
        events.append((s[1][0],'end',s))
    events.sort()
    active = []
    intersections = []
    for x,typ,seg in events:
        if typ == 'start':
            for s in active:
                if intersect(seg[0],seg[1],s[0],s[1]):
                    intersections.append(x)
            active.append(seg)
        else:
            active.remove(seg)
    return intersections

triangles = [[(1,1),(4,1),(2,3)],[(2,0),(5,2),(3,4)]]
print("Intersections:", sweep_triangles(triangles))
```

This basic form can be extended to compute actual intersection coordinates and polygons.

#### Why It Matters

* Fundamental for overlay of meshes, polygon unions, intersection areas
* Used in finite element meshing, map overlay, geometry engines
* Generalizes segment sweeps to polygonal inputs

Applications:

* CAD/CAE analysis
* GIS overlay operations
* Triangulated map intersection
* Rendering and occlusion detection

#### A Gentle Proof (Why It Works)

Each triangle contributes three edges, total $3n$ edges.
Each intersection event occurs when two edges cross.
The Bentley–Ottmann framework ensures every intersection is detected once, by local adjacency in the active set.

Total complexity:

$$
O((n + k)\log n)
$$

where $k$ is number of intersections among edges.

#### Try It Yourself

1. Draw two triangles overlapping partially.
2. Extract edges, sort endpoints by $x$.
3. Sweep, track active edges.
4. Mark each intersection.
5. Compare to brute-force intersection of all edge pairs.

#### Test Cases

| Triangles             | Intersections | Description                 |
| --------------------- | ------------- | --------------------------- |
| Disjoint              | 0             | Non-overlapping             |
| Partially overlapping | >0            | Edge crossings              |
| Nested                | 0             | One triangle inside another |
| Crossing edges        | 2             | Intersecting boundaries     |

#### Complexity

$$
\text{Time: } O((n + k)\log n), \quad \text{Space: } O(n + k)
$$

The Plane Sweep for Triangles weaves polygon edges through the sweep line, tracing every crossing precisely, building the foundation for polygon overlays and mesh operations.

# Section 74. Delaunay and Voronoi Diagrams 

### 731 Delaunay Triangulation (Incremental)

Delaunay Triangulation is a fundamental structure in computational geometry.
Given a set of points in the plane, it connects them into triangles such that no point lies inside the circumcircle of any triangle.
The incremental algorithm builds this triangulation step by step, inserting one point at a time and locally restoring the Delaunay condition.

#### What Problem Are We Solving?

Given $n$ points $P = {p_1, p_2, \ldots, p_n}$ in the plane,
construct a triangulation $T$ such that for every triangle $\triangle abc$ in $T$:

$$
\text{No other point } p \in P \text{ lies inside the circumcircle of } \triangle abc.
$$

This property leads to well-shaped triangles and maximized minimum angles,
making Delaunay triangulations ideal for mesh generation, interpolation, and graphics.

#### Core Idea

Start with a super-triangle that contains all points.
Insert points one by one, and after each insertion, update local connectivity to maintain the empty circle property.

#### How Does It Work (Plain Language)

1. Initialize:
   Create a large triangle enclosing all input points.

2. Insert each point $p_i$:

   * Find the triangle that contains $p_i$.
   * Split it into sub-triangles connecting $p_i$ to its vertices.

3. Legalize edges:

   * For each new edge, check the Delaunay condition.
   * If violated (neighbor's opposite point inside circumcircle), flip the edge.

4. Repeat until all points are inserted.

5. Remove triangles touching the super-triangle vertices.

#### Example Walkthrough

Points:
$P = {A(0,0), B(2,0), C(1,2), D(1,1)}$

1. Super-triangle covers all points.
2. Insert $A, B, C$ → initial triangle $\triangle ABC$.
3. Insert $D(1,1)$ → split into $\triangle ABD$, $\triangle BCD$, $\triangle CAD$.
4. Check each edge for circumcircle violation.
5. Flip edges if needed.

Output: triangulation satisfying empty circle condition.

#### Delaunay Condition (Empty Circle Test)

For triangle with vertices $a,b,c$ and point $p$,
$p$ lies inside the circumcircle if the determinant is positive:

$$
\begin{vmatrix}
a_x & a_y & a_x^2 + a_y^2 & 1 \\
b_x & b_y & b_x^2 + b_y^2 & 1 \\
c_x & c_y & c_x^2 + c_y^2 & 1 \\
p_x & p_y & p_x^2 + p_y^2 & 1
\end{vmatrix} > 0
$$



If true, flip the edge opposite $p$ to restore Delaunay property.

#### Tiny Code (Python Sketch)

```python
import math

def circumcircle_contains(a, b, c, p):
    ax, ay = a
    bx, by = b
    cx, cy = c
    px, py = p
    mat = [
        [ax - px, ay - py, (ax - px)2 + (ay - py)2],
        [bx - px, by - py, (bx - px)2 + (by - py)2],
        [cx - px, cy - py, (cx - px)2 + (cy - py)2],
    ]
    det = (
        mat[0][0] * (mat[1][1]*mat[2][2] - mat[2][1]*mat[1][2])
        - mat[1][0] * (mat[0][1]*mat[2][2] - mat[2][1]*mat[0][2])
        + mat[2][0] * (mat[0][1]*mat[1][2] - mat[1][1]*mat[0][2])
    )
    return det > 0

def incremental_delaunay(points):
    # Placeholder: real implementation would use edge-flip structure
    # Here we return list of triangles in pseudocode form
    return [("triangulation", points)]
```

This pseudocode shows the circumcircle test, core of the legalization step.
Full implementation maintains edge adjacency and triangle flipping.

#### Why It Matters

* Produces high-quality meshes (no skinny triangles)
* Used in terrain modeling, mesh refinement, finite element methods
* Forms basis of Voronoi diagrams (its dual)

Applications:

* 3D modeling and rendering
* Scientific computing and simulation
* GIS interpolation (TIN models)
* Computational geometry toolkits (CGAL, Shapely)

#### A Gentle Proof (Why It Works)

The incremental algorithm maintains Delaunay property at each step:

* Initially, super-triangle satisfies it trivially.
* Each insertion subdivides existing triangle(s).
* Edge flips restore local optimality.

Because every insertion preserves the empty circle condition,
the final triangulation is globally Delaunay.

Time complexity depends on insertion order and point distribution:

$$
O(n^2) \text{ worst case}, \quad O(n \log n) \text{ average case.}
$$

#### Try It Yourself

1. Draw three points, form triangle.
2. Add a fourth inside, connect to all vertices.
3. Check each edge's circumcircle test.
4. Flip any violating edges.
5. Repeat for more points.

Observe how the triangulation adapts to stay Delaunay.

#### Test Cases

| Points            | Triangulation         |
| ----------------- | --------------------- |
| (0,0),(2,0),(1,2) | Single triangle       |
| + (1,1)           | 3 triangles, Delaunay |
| Random 10 points  | Valid triangulation   |

#### Complexity

$$
\text{Time: } O(n \log n) \text{ (average)}, \quad O(n^2) \text{ (worst)}
$$
$$
\text{Space: } O(n)
$$

The Incremental Delaunay Triangulation builds geometry like a sculptor, point by point, flipping edges until every triangle fits the empty-circle harmony.

### 732 Delaunay (Divide & Conquer)

The Divide & Conquer Delaunay Triangulation algorithm constructs the Delaunay triangulation by recursively dividing the point set, triangulating subproblems, and merging them with geometric precision.
It's one of the most elegant and efficient methods, achieving
$$O(n \log n)$$
time complexity while guaranteeing the empty-circle property.

#### What Problem Are We Solving?

Given $n$ points $P = {p_1, p_2, \ldots, p_n}$ in the plane,
find a triangulation such that for each triangle $\triangle abc$:

$$
\text{No other point } p \in P \text{ lies inside the circumcircle of } \triangle abc
$$

We seek a globally Delaunay structure, built recursively from local solutions.

#### How Does It Work (Plain Language)

The Divide & Conquer method parallels merge sort:

1. Sort points by $x$-coordinate.
2. Divide the set into two halves $P_L$ and $P_R$.
3. Recursively triangulate each half to get $T_L$ and $T_R$.
4. Merge the two triangulations:

   * Find the lower common tangent connecting $T_L$ and $T_R$.
   * Then zip upward, adding new Delaunay edges until reaching the upper tangent.
   * Remove edges that violate the empty-circle condition during merging.

After merging, $T = T_L \cup T_R$ is the full Delaunay triangulation.

#### Key Geometric Step: Merging

To merge two Delaunay triangulations:

1. Find the base edge that connects the lowest visible points (the lower tangent).
2. Iteratively add edges connecting points that form valid Delaunay triangles.
3. Flip edges if they violate the circumcircle condition.
4. Continue upward until the upper tangent is reached.

This "zipper" merge creates a seamless, globally valid triangulation.

#### Example Walkthrough

Points: $P = {(0,0), (2,0), (1,2), (4,0), (5,2)}$

1. Sort by $x$: $(0,0), (1,2), (2,0), (4,0), (5,2)$
2. Divide: left half $(0,0),(1,2),(2,0)$, right half $(4,0),(5,2)$
3. Triangulate each half:

   * $T_L$: $\triangle (0,0),(1,2),(2,0)$
   * $T_R$: $\triangle (4,0),(5,2)$
4. Merge:

   * Find lower tangent $(2,0)-(4,0)$
   * Add connecting edges, test with empty-circle condition
   * Final triangulation: Delaunay over all five points

#### Delaunay Test (Empty Circle Check)

For each candidate edge $(a,b)$ connecting left and right sides,
test whether adding a third vertex $c$ maintains Delaunay property:

$$
\begin{vmatrix}
a_x & a_y & a_x^2 + a_y^2 & 1 \\
b_x & b_y & b_x^2 + b_y^2 & 1 \\
c_x & c_y & c_x^2 + c_y^2 & 1 \\
p_x & p_y & p_x^2 + p_y^2 & 1
\end{vmatrix} \le 0
$$


If violated (positive determinant), remove or flip edge.

#### Tiny Code (Python Sketch)

```python
def delaunay_divide(points):
    points = sorted(points)
    if len(points) <= 3:
        # base case: direct triangulation
        return [tuple(points)]
    mid = len(points)//2
    left = delaunay_divide(points[:mid])
    right = delaunay_divide(points[mid:])
    return merge_delaunay(left, right)

def merge_delaunay(left, right):
    # Placeholder merge; real version finds tangents and flips edges
    return left + right
```

This skeleton shows recursive structure; real implementations maintain adjacency, compute tangents, and apply empty-circle checks.

#### Why It Matters

* Optimal time complexity $O(n \log n)$
* Elegant divide-and-conquer paradigm
* Basis for Fortune's sweep and advanced triangulators
* Ideal for static point sets, terrain meshes, GIS models

Applications:

* Terrain modeling (TIN generation)
* Scientific simulation (finite element meshes)
* Voronoi diagram construction (via dual graph)
* Computational geometry libraries (CGAL, Triangle)

#### A Gentle Proof (Why It Works)

1. Base case: small sets are trivially Delaunay.
2. Inductive step: merging preserves Delaunay property since:

   * All edges created during merge satisfy local empty-circle test.
   * The merge only connects boundary vertices visible to each other.

Therefore, by induction, the final triangulation is Delaunay.

Each merge step takes linear time, and there are $\log n$ levels:

$$
T(n) = 2T(n/2) + O(n) = O(n \log n)
$$

#### Try It Yourself

1. Plot 6–8 points sorted by $x$.
2. Divide into two halves, triangulate each.
3. Draw lower tangent, connect visible vertices.
4. Flip any edges violating empty-circle property.
5. Verify final triangulation satisfies Delaunay rule.

#### Test Cases

| Points           | Triangulation Type           |
| ---------------- | ---------------------------- |
| 3 points         | Single triangle              |
| 5 points         | Merged triangles             |
| Random 10 points | Valid Delaunay triangulation |

#### Complexity

$$
\text{Time: } O(n \log n), \quad \text{Space: } O(n)
$$

The Divide & Conquer Delaunay algorithm builds harmony through balance, splitting the plane, solving locally, and merging globally into a perfect empty-circle mosaic.

### 733 Delaunay (Fortune's Sweep)

The Fortune's Sweep Algorithm is a brilliant plane-sweep approach to constructing the Delaunay triangulation and its dual, the Voronoi diagram, in
$$
O(n \log n)
$$
time.
It elegantly slides a sweep line (or parabola) across the plane, maintaining a dynamic structure called the beach line to trace the evolution of Voronoi edges, from which the Delaunay edges can be derived.

#### What Problem Are We Solving?

Given $n$ points $P = {p_1, p_2, \ldots, p_n}$ (called *sites*), construct their Delaunay triangulation, a set of triangles such that no point lies inside the circumcircle of any triangle.

Its dual graph, the Voronoi diagram, partitions the plane into cells, one per point, containing all locations closer to that point than to any other.

Fortune's algorithm constructs both structures simultaneously, efficiently.

#### Key Insight

As a sweep line moves downward, the frontier of influence of each site forms a parabolic arc.
The beach line is the union of all active arcs.
Voronoi edges appear where arcs meet; Delaunay edges connect sites whose arcs share a boundary.

The algorithm processes two kinds of events:

1. Site Events, when a new site is reached by the sweep line
2. Circle Events, when arcs vanish as the beach line reshapes (three arcs meet in a circle)

#### How Does It Work (Plain Language)

1. Sort all sites by $y$-coordinate (top to bottom).

2. Sweep a horizontal line downward:

   * At each site event, insert a new parabolic arc into the beach line.
   * Update intersections to create Voronoi/Delaunay edges.

3. At each circle event, remove the disappearing arc (when three arcs meet at a vertex of the Voronoi diagram).

4. Maintain:

   * Event queue: upcoming site/circle events
   * Beach line: balanced tree of arcs
   * Output edges: Voronoi edges / Delaunay edges (dual)

5. Continue until all events are processed.

6. Close all remaining open edges at the bounding box.

The Delaunay triangulation is recovered by connecting sites that share a Voronoi edge.

#### Example Walkthrough

Points:

* $A(2,6), B(5,5), C(3,3)$

1. Sort by $y$: $A, B, C$
2. Sweep down:

   * Site $A$: create new arc
   * Site $B$: new arc splits existing arc, new breakpoint → Voronoi edge starts
   * Site $C$: another split, more breakpoints
3. Circle event: arcs merge → Voronoi vertex, record Delaunay triangle
4. Output: three Voronoi cells, Delaunay triangle connecting $A, B, C$

#### Data Structures

| Structure                    | Purpose                            |
| ---------------------------- | ---------------------------------- |
| Event queue (priority queue) | Site & circle events sorted by $y$ |
| Beach line (balanced BST)    | Active arcs (parabolas)            |
| Output edge list             | Voronoi / Delaunay edges           |

#### Tiny Code (Pseudocode)

```python
def fortunes_algorithm(points):
    points.sort(key=lambda p: -p[1])  # top to bottom
    event_queue = [(p[1], 'site', p) for p in points]
    beach_line = []
    voronoi_edges = []
    while event_queue:
        y, typ, data = event_queue.pop(0)
        if typ == 'site':
            insert_arc(beach_line, data)
        else:
            remove_arc(beach_line, data)
        update_edges(beach_line, voronoi_edges)
    return voronoi_edges
```

This sketch omits details but shows the event-driven sweep structure.

#### Why It Matters

* Optimal $O(n \log n)$ Delaunay / Voronoi construction
* Avoids complex global flipping
* Beautiful geometric interpretation: parabolas + sweep line
* Foundation of computational geometry libraries (e.g., CGAL, Boost, Qhull)

Applications:

* Nearest neighbor search (Voronoi regions)
* Terrain and mesh generation
* Cellular coverage modeling
* Motion planning and influence maps

#### A Gentle Proof (Why It Works)

Each site and circle triggers at most one event, giving $O(n)$ events.
Each event takes $O(\log n)$ time (insertion/removal/search in balanced tree).
All edges satisfy local Delaunay condition because arcs are created only when parabolas meet (equal distance frontier).

Therefore, total complexity:

$$
O(n \log n)
$$

Correctness follows from:

* Sweep line maintains valid partial Voronoi/Delaunay structure
* Every Delaunay edge is created exactly once (dual to Voronoi edges)

#### Try It Yourself

1. Plot 3–5 points on paper.
2. Imagine a line sweeping downward.
3. Draw parabolic arcs from each point (distance loci).
4. Mark intersections (Voronoi edges).
5. Connect adjacent points, Delaunay edges appear naturally.

#### Test Cases

| Points                    | Delaunay Triangles | Voronoi Cells   |
| ------------------------- | ------------------ | --------------- |
| 3 points forming triangle | 1                  | 3               |
| 4 non-collinear points    | 2                  | 4               |
| Grid points               | many               | grid-like cells |

#### Complexity

$$
\text{Time: } O(n \log n), \quad \text{Space: } O(n)
$$

The Fortune's Sweep Algorithm reveals the deep duality of geometry, as a moving parabola traces invisible boundaries, triangles and cells crystallize from pure distance symmetry.

### 734 Voronoi Diagram (Fortune's Sweep)

The Voronoi Diagram partitions the plane into regions, each region consists of all points closest to a specific site.
Fortune's Sweep Line Algorithm constructs this structure in
$$O(n \log n)$$
time, using the same framework as the Delaunay sweep, since the two are duals.

#### What Problem Are We Solving?

Given a set of $n$ sites
$$
P = {p_1, p_2, \ldots, p_n}
$$
each at $(x_i, y_i)$, the Voronoi diagram divides the plane into cells:

$$
V(p_i) = { q \mid d(q, p_i) \le d(q, p_j), \ \forall j \ne i }
$$

Each Voronoi cell $V(p_i)$ is a convex polygon (for distinct sites).
Edges are perpendicular bisectors between pairs of sites.
Vertices are circumcenters of triples of sites.

#### Why Use Fortune's Algorithm?

Naive approach: compute all pairwise bisectors ($O(n^2)$), then intersect them.
Fortune's method improves to
$$O(n \log n)$$
by sweeping a line and maintaining parabolic arcs that define the beach line, the evolving boundary between processed and unprocessed regions.

#### How Does It Work (Plain Language)

The sweep line moves top-down (decreasing $y$), dynamically tracing the frontier of influence for each site.

1. Site Events

   * When sweep reaches a new site, insert a new parabola arc into the beach line.
   * Intersections between arcs become breakpoints, which form Voronoi edges.

2. Circle Events

   * When three consecutive arcs converge, the middle one disappears.
   * The convergence point is a Voronoi vertex (circumcenter of three sites).

3. Event Queue

   * Sorted by $y$ coordinate (priority queue).
   * Each processed event updates the beach line and outputs edges.

4. Termination

   * When all events processed, extend unfinished edges to bounding box.

The output is a full Voronoi diagram, and by duality, its Delaunay triangulation.

#### Example Walkthrough

Sites:
$A(2,6), B(5,5), C(3,3)$

Steps:

1. Sweep starts at top (site A).
2. Insert A → beach line = single arc.
3. Reach B → insert new arc, two arcs meet → start Voronoi edge.
4. Reach C → new arc, more edges form.
5. Circle event where arcs converge → Voronoi vertex at circumcenter.
6. Sweep completes → edges finalized, diagram closed.

Output: 3 Voronoi cells, 3 vertices, 3 Delaunay edges.

#### Beach Line Representation

The beach line is a sequence of parabolic arcs, stored in a balanced BST keyed by $x$-order.
Breakpoints between arcs trace Voronoi edges.

When a site is inserted, it splits an existing arc.
When a circle event triggers, an arc disappears, creating a vertex.

#### Tiny Code (Pseudocode)

```python
def voronoi_fortune(points):
    points.sort(key=lambda p: -p[1])  # top to bottom
    event_queue = [(p[1], 'site', p) for p in points]
    beach_line = []
    voronoi_edges = []
    while event_queue:
        y, typ, data = event_queue.pop(0)
        if typ == 'site':
            insert_arc(beach_line, data)
        else:
            remove_arc(beach_line, data)
        update_edges(beach_line, voronoi_edges)
    return voronoi_edges
```

This high-level structure emphasizes the event-driven nature of the algorithm.
Implementations use specialized data structures for arcs, breakpoints, and circle event scheduling.

#### Why It Matters

* Constructs Voronoi and Delaunay simultaneously
* Optimal $O(n \log n)$ complexity
* Robust for large-scale geometric data
* Foundation of spatial structures in computational geometry

Applications:

* Nearest-neighbor search
* Spatial partitioning in games and simulations
* Facility location and influence maps
* Mesh generation (via Delaunay dual)

#### A Gentle Proof (Why It Works)

Each site event adds exactly one arc → $O(n)$ site events.
Each circle event removes one arc → $O(n)$ circle events.
Each event processed in $O(\log n)$ (tree updates, priority queue ops).

Thus, total:

$$
O(n \log n)
$$

Correctness follows from geometry of parabolas:

* Breakpoints always move monotonically
* Each Voronoi vertex is created exactly once
* Beach line evolves without backtracking

#### Try It Yourself

1. Plot 3–5 points.
2. Draw perpendicular bisectors pairwise.
3. Note intersections (Voronoi vertices).
4. Connect edges into convex polygons.
5. Compare to Fortune's sweep behavior.

#### Test Cases

| Sites           | Voronoi Regions | Vertices | Delaunay Edges |
| --------------- | --------------- | -------- | -------------- |
| 3 points        | 3               | 1        | 3              |
| 4 non-collinear | 4               | 3        | 5              |
| Grid 3x3        | 9               | many     | lattice mesh   |

#### Complexity

$$
\text{Time: } O(n \log n), \quad \text{Space: } O(n)
$$

The Fortune's Sweep for Voronoi Diagrams is geometry in motion, parabolas rising and falling under a moving horizon, tracing invisible borders that define proximity and structure.

### 735 Incremental Voronoi

The Incremental Voronoi Algorithm builds a Voronoi diagram step by step by inserting one site at a time, updating the existing diagram locally rather than recomputing from scratch.
It's conceptually simple and forms the basis of dynamic and online Voronoi systems.

#### What Problem Are We Solving?

We want to construct or update a Voronoi diagram for a set of points
$$
P = {p_1, p_2, \ldots, p_n}
$$
so that for each site $p_i$, its Voronoi cell contains all points closer to $p_i$ than to any other site.

In static algorithms (like Fortune's sweep), all points must be known upfront.
But what if we want to add sites incrementally, one at a time, and update the diagram locally?

That's exactly what this algorithm enables.

#### How Does It Work (Plain Language)

1. Start simple: begin with a single site, its cell is the entire plane (bounded by a large box).
2. Insert next site:

   * Locate which cell contains it.
   * Compute perpendicular bisector between new and existing site.
   * Clip existing cells using the bisector.
   * The new site's cell is formed from regions closer to it than any others.
3. Repeat for all sites.

Each insertion modifies only nearby cells, not the entire diagram, this local nature is key.

#### Example Walkthrough

Sites:
$A(2,2)$ → $B(6,2)$ → $C(4,5)$

1. Start with A: single cell (whole bounding box).
2. Add B:

   * Draw perpendicular bisector between A and B.
   * Split plane vertically → two cells.
3. Add C:

   * Draw bisector with A and B.
   * Intersect bisectors to form three Voronoi regions.

Each new site carves its influence area by cutting existing cells.

#### Geometric Steps (Insert Site $p$)

1. Locate containing cell: find which cell $p$ lies in.
2. Find affected cells: these are the neighbors whose regions are closer to $p$ than some part of their area.
3. Compute bisectors between $p$ and each affected site.
4. Clip and rebuild cell polygons.
5. Update adjacency graph of neighboring cells.

#### Data Structures

| Structure            | Purpose                        |
| -------------------- | ------------------------------ |
| Cell list            | Polygon boundaries per site    |
| Site adjacency graph | For efficient neighbor lookups |
| Bounding box         | For finite diagram truncation  |

Optional acceleration:

* Delaunay triangulation: dual structure for locating cells faster
* Spatial index (KD-tree) for cell search

#### Tiny Code (Pseudocode)

```python
def incremental_voronoi(points, bbox):
    diagram = init_diagram(points[0], bbox)
    for p in points[1:]:
        cell = locate_cell(diagram, p)
        neighbors = find_neighbors(cell)
        for q in neighbors:
            bisector = perpendicular_bisector(p, q)
            clip_cells(diagram, p, q, bisector)
        add_cell(diagram, p)
    return diagram
```

This pseudocode highlights progressive construction by bisector clipping.

#### Why It Matters

* Simple concept, easy to visualize and implement
* Local updates, only nearby regions change
* Works for dynamic systems (adding/removing points)
* Dual to incremental Delaunay triangulation

Applications:

* Online facility location
* Dynamic sensor coverage
* Real-time influence mapping
* Game AI regions (unit territories)

#### A Gentle Proof (Why It Works)

Each step maintains the Voronoi property:

* Every region is intersection of half-planes
* Each insertion adds new bisectors, refining the partition
* No recomputation of unaffected regions

Time complexity depends on how efficiently we locate affected cells.
Naively $O(n^2)$, but with Delaunay dual and point location:
$$
O(n \log n)
$$

#### Try It Yourself

1. Draw bounding box and one point (site).
2. Insert second point, draw perpendicular bisector.
3. Insert third, draw bisectors to all sites, clip overlapping regions.
4. Shade each Voronoi cell, check boundaries are equidistant from two sites.
5. Repeat for more points.

#### Test Cases

| Sites   | Result                      |
| ------- | --------------------------- |
| 1 site  | Whole box                   |
| 2 sites | Two half-planes             |
| 3 sites | Three convex polygons       |
| 5 sites | Complex polygon arrangement |

#### Complexity

$$
\text{Time: } O(n^2) \text{ naive}, \quad O(n \log n) \text{ with Delaunay assist}
$$
$$
\text{Space: } O(n)
$$

The Incremental Voronoi Algorithm grows the diagram like crystal formation, each new point carves its own region, reshaping the world around it with clean geometric cuts.

### 736 Bowyer–Watson

The Bowyer–Watson Algorithm is a simple yet powerful incremental method for building a Delaunay triangulation.
Each new point is inserted one at a time, and the algorithm locally re-triangulates the region affected by that insertion, ensuring the empty-circle property remains true.

It is one of the most intuitive and widely used Delaunay construction methods.

#### What Problem Are We Solving?

We want to construct a Delaunay triangulation for a set of points
$$
P = {p_1, p_2, \ldots, p_n}
$$
such that every triangle satisfies the empty-circle property:

$$
\text{For every triangle } \triangle abc, \text{ no other point } p \in P \text{ lies inside its circumcircle.}
$$

We build the triangulation incrementally, maintaining validity after each insertion.

#### How Does It Work (Plain Language)

Think of the plane as a stretchy mesh. Each time you add a point:

1. You find all triangles whose circumcircles contain the new point (the "bad triangles").
2. You remove those triangles, they no longer satisfy the Delaunay condition.
3. The boundary of the removed region forms a polygonal cavity.
4. You connect the new point to every vertex on that boundary.
5. The result is a new triangulation that's still Delaunay.

Repeat until all points are inserted.

#### Step-by-Step Example

Points: $A(0,0), B(5,0), C(2.5,5), D(2.5,2)$

1. Initialize with a super-triangle that encloses all points.
2. Insert $A, B, C$ → base triangle.
3. Insert $D$:

   * Find triangles whose circumcircles contain $D$.
   * Remove them (forming a "hole").
   * Reconnect $D$ to boundary vertices of the hole.
4. Resulting triangulation satisfies Delaunay property.

#### Geometric Core: The Cavity

For each new point $p$:

* Find all triangles $\triangle abc$ with
  $$p \text{ inside } \text{circumcircle}(a, b, c).$$
* Remove those triangles.
* Collect all boundary edges shared by only one bad triangle, they form the cavity polygon.
* Connect $p$ to each boundary edge to form new triangles.

#### Tiny Code (Pseudocode)

```python
def bowyer_watson(points):
    tri = [super_triangle(points)]
    for p in points:
        bad_tris = [t for t in tri if in_circumcircle(p, t)]
        boundary = find_boundary(bad_tris)
        for t in bad_tris:
            tri.remove(t)
        for edge in boundary:
            tri.append(make_triangle(edge[0], edge[1], p))
    tri = [t for t in tri if not shares_vertex_with_super(t)]
    return tri
```

Key helper:

* `in_circumcircle(p, triangle)` tests if point lies inside circumcircle
* `find_boundary` identifies edges not shared by two removed triangles

#### Why It Matters

* Simple and robust, easy to implement
* Handles incremental insertion naturally
* Basis for many dynamic Delaunay systems
* Dual to Incremental Voronoi (each insertion updates local cells)

Applications:

* Mesh generation (finite elements, 2D/3D)
* GIS terrain modeling
* Particle simulations
* Spatial interpolation (e.g. natural neighbor)

#### A Gentle Proof (Why It Works)

Each insertion removes only triangles that violate the empty-circle property, then adds new triangles that preserve it.

By induction:

1. Base triangulation (super-triangle) is valid.
2. Each insertion preserves local Delaunay condition.
3. Therefore, the entire triangulation remains Delaunay.

Complexity:

* Naive search for bad triangles: $O(n)$ per insertion
* Total: $O(n^2)$
* With spatial indexing / point location:
  $$O(n \log n)$$

#### Try It Yourself

1. Draw 3 points → initial triangle.
2. Add a new point inside.
3. Draw circumcircles for all triangles, mark those containing the new point.
4. Remove them; connect the new point to the boundary.
5. Observe all triangles now satisfy empty-circle rule.

#### Test Cases

| Points          | Triangles          | Property                |
| --------------- | ------------------ | ----------------------- |
| 3 points        | 1 triangle         | trivially Delaunay      |
| 4 points        | 2 triangles        | both empty-circle valid |
| Random 6 points | multiple triangles | valid triangulation     |

#### Complexity

$$
\text{Time: } O(n^2) \text{ naive}, \quad O(n \log n) \text{ optimized}
$$
$$
\text{Space: } O(n)
$$

The Bowyer–Watson Algorithm is like sculpting with triangles, each new point gently reshapes the mesh, carving out cavities and stitching them back with perfect geometric balance.

### 737 Duality Transform

The Duality Transform reveals the deep connection between Delaunay triangulations and Voronoi diagrams, they are geometric duals.
Every Voronoi edge corresponds to a Delaunay edge, and every Voronoi vertex corresponds to a Delaunay triangle circumcenter.

By understanding this duality, we can construct one structure from the other, no need to compute both separately.

#### What Problem Are We Solving?

We often need both the Delaunay triangulation (for connectivity) and the Voronoi diagram (for spatial partitioning).

Rather than building each independently, we can use duality:

* Build Delaunay triangulation, derive Voronoi.
* Or build Voronoi diagram, derive Delaunay.

This saves computation and highlights the symmetry of geometry.

#### Dual Relationship

Let $P = {p_1, p_2, \ldots, p_n}$ be a set of sites in the plane.

1. Vertices:

   * Each Voronoi vertex corresponds to the circumcenter of a Delaunay triangle.

2. Edges:

   * Each Voronoi edge is perpendicular to its dual Delaunay edge.
   * It connects circumcenters of adjacent Delaunay triangles.

3. Faces:

   * Each Voronoi cell corresponds to a site vertex in Delaunay.

So:
$$
\text{Voronoi(Dual)} = \text{Delaunay(Primal)}
$$

and vice versa.

#### How Does It Work (Plain Language)

Start with Delaunay triangulation:

1. For each triangle, compute its circumcenter.
2. Connect circumcenters of adjacent triangles (triangles sharing an edge).
3. These connections form Voronoi edges.
4. The collection of these edges forms the Voronoi diagram.

Alternatively, start with Voronoi diagram:

1. Each cell's site becomes a vertex.
2. Connect two sites if their cells share a boundary → Delaunay edge.
3. Triangles form by linking triplets of mutually adjacent cells.

#### Example Walkthrough

Sites: $A(2,2), B(6,2), C(4,5)$

1. Delaunay triangulation: triangle $ABC$.
2. Circumcenter of $\triangle ABC$ = Voronoi vertex.
3. Draw perpendicular bisectors between pairs $(A,B), (B,C), (C,A)$.
4. These form Voronoi edges meeting at the circumcenter.

Now:

* Voronoi edges ⟷ Delaunay edges
* Voronoi vertex ⟷ Delaunay triangle

Duality complete.

#### Algebraic Dual (Point-Line Transform)

In computational geometry, we often use point-line duality:

$$
(x, y) \longleftrightarrow y = mx - c
$$

or more commonly:

$$
(x, y) \mapsto y = ax - b
$$

In this sense:

* A point in primal space corresponds to a line in dual space.
* Incidence and order are preserved:

  * Points above/below line ↔ lines above/below point.

Used in convex hull and half-plane intersection computations.

#### Tiny Code (Python Sketch)

```python
def delaunay_to_voronoi(delaunay):
    voronoi_vertices = [circumcenter(t) for t in delaunay.triangles]
    voronoi_edges = []
    for e in delaunay.shared_edges():
        c1 = circumcenter(e.tri1)
        c2 = circumcenter(e.tri2)
        voronoi_edges.append((c1, c2))
    return voronoi_vertices, voronoi_edges
```

Here, `circumcenter(triangle)` computes the center of the circumcircle.

#### Why It Matters

* Unifies two core geometric structures
* Enables conversion between triangulation and partition
* Essential for mesh generation, pathfinding, spatial queries
* Simplifies algorithms: compute one, get both

Applications:

* Terrain modeling: triangulate elevation, derive regions
* Nearest neighbor: Voronoi search
* Computational physics: Delaunay meshes, Voronoi volumes
* AI navigation: region adjacency via duality

#### A Gentle Proof (Why It Works)

In a Delaunay triangulation:

* Each triangle satisfies empty-circle property.
* The circumcenter of adjacent triangles is equidistant from two sites.

Thus, connecting circumcenters of adjacent triangles gives edges equidistant from two sites, by definition, Voronoi edges.

So the dual of a Delaunay triangulation is exactly the Voronoi diagram.

Formally:
$$
\text{Delaunay}(P) = \text{Voronoi}^*(P)
$$

#### Try It Yourself

1. Plot 4 non-collinear points.
2. Construct Delaunay triangulation.
3. Draw circumcircles and locate circumcenters.
4. Connect circumcenters of adjacent triangles → Voronoi edges.
5. Observe perpendicularity to original Delaunay edges.

#### Test Cases

| Sites    | Delaunay Triangles | Voronoi Vertices |
| -------- | ------------------ | ---------------- |
| 3        | 1                  | 1                |
| 4        | 2                  | 2                |
| Random 6 | 4–6                | Many             |

#### Complexity

If either structure is known:
$$
\text{Conversion Time: } O(n)
$$
$$
\text{Space: } O(n)
$$

The Duality Transform is geometry's mirror, each edge, face, and vertex reflected across a world of perpendiculars, revealing two sides of the same elegant truth.

### 738 Power Diagram (Weighted Voronoi)

A Power Diagram (also called a Laguerre–Voronoi diagram) is a generalization of the Voronoi diagram where each site has an associated weight.
Instead of simple Euclidean distance, we use the power distance, which shifts or shrinks regions based on these weights.

This allows modeling influence zones where some points "push harder" than others, ideal for applications like additively weighted nearest neighbor and circle packing.

#### What Problem Are We Solving?

In a standard Voronoi diagram, each site $p_i$ owns all points $q$ closer to it than to any other site:
$$
V(p_i) = { q \mid d(q, p_i) \le d(q, p_j), \ \forall j \ne i }.
$$

In a Power Diagram, each site $p_i$ has a weight $w_i$, and cells are defined by power distance:
$$
\pi_i(q) = | q - p_i |^2 - w_i.
$$

A point $q$ belongs to the power cell of $p_i$ if:
$$
\pi_i(q) \le \pi_j(q) \quad \forall j \ne i.
$$

When all weights $w_i = 0$, we recover the classic Voronoi diagram.

#### How Does It Work (Plain Language)

Think of each site as a circle (or sphere) with radius determined by its weight.
Instead of pure distance, we compare power distances:

* Larger weights mean stronger influence (bigger circle).
* Smaller weights mean weaker influence.

A point $q$ chooses the site whose power distance is smallest.

This creates tilted bisectors (not perpendicular), and cells may disappear entirely if they're dominated by neighbors.

#### Example Walkthrough

Sites with weights:

* $A(2,2), w_A = 1$
* $B(6,2), w_B = 0$
* $C(4,5), w_C = 4$

Compute power bisector between $A$ and $B$:

$$
|q - A|^2 - w_A = |q - B|^2 - w_B
$$

Expanding and simplifying yields a linear equation, a shifted bisector:
$$
2(x_B - x_A)x + 2(y_B - y_A)y = (x_B^2 + y_B^2 - w_B) - (x_A^2 + y_A^2 - w_A)
$$

Thus, boundaries remain straight lines, but not centered between sites.

#### Algorithm (High-Level)

1. Input: Sites $p_i = (x_i, y_i)$ with weights $w_i$.
2. Compute all pairwise bisectors using power distance.
3. Intersect bisectors to form polygonal cells.
4. Clip cells to bounding box.
5. (Optional) Use dual weighted Delaunay triangulation (regular triangulation) for efficiency.

#### Geometric Dual: Regular Triangulation

The dual of a power diagram is a regular triangulation, built using lifted points in 3D:

Map each site $(x_i, y_i, w_i)$ to 3D point $(x_i, y_i, x_i^2 + y_i^2 - w_i)$.

The lower convex hull of these lifted points, projected back to 2D, gives the power diagram.

#### Tiny Code (Python Sketch)

```python
def power_bisector(p1, w1, p2, w2):
    (x1, y1), (x2, y2) = p1, p2
    a = 2 * (x2 - x1)
    b = 2 * (y2 - y1)
    c = (x22 + y22 - w2) - (x12 + y12 - w1)
    return (a, b, c)  # line ax + by = c

def power_diagram(points, weights):
    cells = []
    for i, p in enumerate(points):
        cell = bounding_box()
        for j, q in enumerate(points):
            if i == j: continue
            a, b, c = power_bisector(p, weights[i], q, weights[j])
            cell = halfplane_intersect(cell, a, b, c)
        cells.append(cell)
    return cells
```

Each cell is built as an intersection of half-planes defined by weighted bisectors.

#### Why It Matters

* Generalization of Voronoi for weighted influence
* Produces regular triangulation duals
* Supports non-uniform density modeling

Applications:

* Physics: additively weighted fields
* GIS: territory with varying influence
* Computational geometry: circle packing
* Machine learning: power diagrams for weighted clustering

#### A Gentle Proof (Why It Works)

Each cell is defined by linear inequalities:
$$
\pi_i(q) \le \pi_j(q)
$$
which are half-planes.
The intersection of these half-planes forms a convex polygon (possibly empty).

Thus, each cell:

* Is convex
* Covers all space (union of cells)
* Is disjoint from others

Dual structure: regular triangulation, maintaining weighted Delaunay property (empty *power circle*).

Complexity:
$$
O(n \log n)
$$
using incremental or lifting methods.

#### Try It Yourself

1. Draw two points with different weights.
2. Compute power bisector, note it's not equidistant.
3. Add third site, see how regions shift by weight.
4. Increase weight of one site, watch its cell expand.

#### Test Cases

| Sites     | Weights   | Diagram           |
| --------- | --------- | ----------------- |
| 2 equal   | same      | vertical bisector |
| 2 unequal | one large | shifted boundary  |
| 3 varied  | mixed     | tilted polygons   |

#### Complexity

$$
\text{Time: } O(n \log n), \quad \text{Space: } O(n)
$$

The Power Diagram bends geometry to influence, every weight warps the balance of space, redrawing the borders of proximity and power.

### 739 Lloyd's Relaxation

Lloyd's Relaxation (also called Lloyd's Algorithm) is an iterative process that refines a Voronoi diagram by repeatedly moving each site to the centroid of its Voronoi cell.
The result is a Centroidal Voronoi Tessellation (CVT), a diagram where each region's site is also its center of mass.

It's a geometric smoothing method that transforms irregular partitions into beautifully uniform, balanced layouts.

#### What Problem Are We Solving?

A standard Voronoi diagram partitions space by proximity, but cell shapes can be irregular or skewed if sites are unevenly distributed.

We want a balanced diagram where:

* Cells are compact and similar in size
* Sites are located at cell centroids

Lloyd's relaxation solves this by iterative refinement.

#### How Does It Work (Plain Language)

Start with a random set of points and a bounding region.

Then repeat:

1. Compute Voronoi diagram of current sites.
2. Find centroid of each Voronoi cell (average of all points in that region).
3. Move each site to its cell's centroid.
4. Repeat until sites converge (movement is small).

Over time, sites spread out evenly, forming a blue-noise distribution, ideal for sampling and meshing.

#### Step-by-Step Example

1. Initialize 10 random sites in a square.
2. Compute Voronoi diagram.
3. For each cell, compute centroid:
   $$
   c_i = \frac{1}{A_i} \int_{V_i} (x, y) , dA
   $$
4. Replace site position $p_i$ with centroid $c_i$.
5. Repeat 5–10 iterations.

Result: smoother, more regular cells with nearly equal areas.

#### Tiny Code (Python Sketch)

```python
import numpy as np
from scipy.spatial import Voronoi

def lloyd_relaxation(points, bounds, iterations=5):
    for _ in range(iterations):
        vor = Voronoi(points)
        new_points = []
        for region_index in vor.point_region:
            region = vor.regions[region_index]
            if -1 in region or len(region) == 0:
                continue
            polygon = [vor.vertices[i] for i in region]
            centroid = np.mean(polygon, axis=0)
            new_points.append(centroid)
        points = np.array(new_points)
    return points
```

This simple implementation uses Scipy's Voronoi and computes centroids as polygon averages.

#### Why It Matters

* Produces uniform partitions, smooth and balanced
* Generates blue-noise distributions (useful for sampling)
* Used in meshing, texture generation, and Poisson disk sampling
* Converges quickly (a few iterations often suffice)

Applications:

* Mesh generation (finite elements, simulations)
* Sampling for graphics / procedural textures
* Clustering (k-means is a discrete analogue)
* Lattice design and territory optimization

#### A Gentle Proof (Why It Works)

Each iteration reduces an energy functional:

$$
E = \sum_i \int_{V_i} | q - p_i |^2 , dq
$$

This measures total squared distance from sites to points in their regions.
Moving $p_i$ to the centroid minimizes $E_i$ locally.

As iterations continue:

* Energy decreases monotonically
* System converges to fixed point where each $p_i$ is centroid of $V_i$

At convergence:
$$
p_i = c_i
$$
Each cell is a centroidal Voronoi region.

#### Try It Yourself

1. Scatter random points on paper.
2. Draw Voronoi cells.
3. Estimate centroids (visually or with grid).
4. Move points to centroids.
5. Redraw Voronoi.
6. Repeat, see pattern become uniform.

#### Test Cases

| Sites     | Iterations | Result             |
| --------- | ---------- | ------------------ |
| 10 random | 0          | irregular Voronoi  |
| 10 random | 3          | smoother, balanced |
| 10 random | 10         | uniform CVT        |

#### Complexity

Each iteration:

* Voronoi computation: $O(n \log n)$
* Centroid update: $O(n)$

Total:
$$
O(k n \log n)
$$
for $k$ iterations.

Lloyd's Relaxation polishes randomness into order, each iteration a gentle nudge toward harmony, transforming scattered points into a balanced, geometric mosaic.

### 740 Voronoi Nearest Neighbor

The Voronoi Nearest Neighbor query is a natural application of the Voronoi diagram, once the diagram is constructed, nearest-neighbor lookups become instantaneous.
Each query point simply falls into a Voronoi cell, and the site defining that cell is its closest neighbor.

This makes Voronoi structures perfect for spatial search, proximity analysis, and geometric classification.

#### What Problem Are We Solving?

Given a set of sites
$$
P = {p_1, p_2, \ldots, p_n}
$$
and a query point $q$, we want to find the nearest site:
$$
p^* = \arg\min_{p_i \in P} | q - p_i |.
$$

A Voronoi diagram partitions space so that every point $q$ inside a cell $V(p_i)$ satisfies:
$$
| q - p_i | \le | q - p_j |, \ \forall j \ne i.
$$

Thus, locating $q$'s cell immediately reveals its nearest neighbor.

#### How Does It Work (Plain Language)

1. Preprocess: Build Voronoi diagram from sites.
2. Query: Given a new point $q$, determine which Voronoi cell it lies in.
3. Answer: The site that generated that cell is the nearest neighbor.

This transforms nearest-neighbor search from computation (distance comparisons) into geometry (region lookup).

#### Example Walkthrough

Sites:

* $A(2,2)$
* $B(6,2)$
* $C(4,5)$

Construct Voronoi diagram → three convex cells.

Query: $q = (5,3)$

* Check which region contains $q$ → belongs to cell of $B$.
* So nearest neighbor is $B(6,2)$.

#### Algorithm (High-Level)

1. Build Voronoi diagram (any method, e.g. Fortune's sweep).
2. Point location:

   * Use spatial index or planar subdivision search (e.g. trapezoidal map).
   * Query point $q$ → find containing polygon.
3. Return associated site.

Optional optimization: if many queries are expected, build a point-location data structure for $O(\log n)$ queries.

#### Tiny Code (Python Sketch)

```python
from scipy.spatial import Voronoi, KDTree

def voronoi_nearest(points, queries):
    vor = Voronoi(points)
    tree = KDTree(points)
    result = []
    for q in queries:
        dist, idx = tree.query(q)
        result.append((q, points[idx], dist))
    return result
```

Here we combine Voronoi geometry (for understanding) with KD-tree (for practical speed).

In exact Voronoi lookup, each query uses point-location in the planar subdivision.

#### Why It Matters

* Turns nearest-neighbor search into constant-time lookup (after preprocessing)
* Enables spatial partitioning for clustering, navigation, simulation
* Forms foundation for:

  * Nearest facility location
  * Path planning (region transitions)
  * Interpolation (e.g. nearest-site assignment)
  * Density estimation, resource allocation

Used in:

* GIS (find nearest hospital, school, etc.)
* Robotics (navigation zones)
* Physics (Voronoi cells in particle systems)
* ML (nearest centroid classifiers)

#### A Gentle Proof (Why It Works)

By definition, each Voronoi cell $V(p_i)$ satisfies:
$$
V(p_i) = { q \mid | q - p_i | \le | q - p_j | \ \forall j \ne i }.
$$

So if $q \in V(p_i)$, then:
$$
| q - p_i | = \min_{p_j \in P} | q - p_j |.
$$

Therefore, locating $q$'s cell gives the correct nearest neighbor.
Efficient point location (via planar search) ensures $O(\log n)$ query time.

#### Try It Yourself

1. Draw 4 sites on paper.
2. Construct Voronoi diagram.
3. Pick a random query point.
4. See which cell contains it, that's your nearest site.
5. Verify by computing distances manually.

#### Test Cases

| Sites                  | Query        | Nearest                 |
| ---------------------- | ------------ | ----------------------- |
| A(0,0), B(4,0)         | (1,1)        | A                       |
| A(2,2), B(6,2), C(4,5) | (5,3)        | B                       |
| Random 5 sites         | random query | site of containing cell |

#### Complexity

* Preprocessing (Voronoi build): $O(n \log n)$
* Query (point location): $O(\log n)$
* Space: $O(n)$

The Voronoi Nearest Neighbor method replaces brute-force distance checks with elegant geometry, every query resolved by finding where it lives, not how far it travels.

# Section 75. Point in Polygon and Polygon Triangulation 

### 741 Ray Casting

The Ray Casting Algorithm (also known as the even–odd rule) is a simple and elegant method to determine whether a point lies inside or outside a polygon.
It works by shooting an imaginary ray from the query point and counting how many times it crosses the polygon's edges.

If the number of crossings is odd, the point is inside.
If even, the point is outside.

#### What Problem Are We Solving?

Given:

* A polygon defined by vertices
  $$P = {v_1, v_2, \ldots, v_n}$$
* A query point
  $$q = (x_q, y_q)$$

Determine whether $q$ lies inside, outside, or on the boundary of the polygon.

This test is fundamental in:

* Computational geometry
* Computer graphics (hit-testing)
* Geographic Information Systems (point-in-polygon)
* Collision detection

#### How Does It Work (Plain Language)

Imagine shining a light ray horizontally to the right from the query point $q$.
Each time the ray intersects a polygon edge, we flip an inside/outside flag.

* If ray crosses an edge odd number of times → point is inside
* If even → point is outside

Special care is needed when:

* The ray passes exactly through a vertex
* The point lies exactly on an edge

#### Step-by-Step Procedure

1. Set `count = 0`.
2. For each polygon edge $(v_i, v_{i+1})$:

   * Check if the horizontal ray from $q$ intersects the edge.
   * If yes, increment `count`.
3. If `count` is odd, $q$ is inside.
   If even, $q$ is outside.

Edge intersection condition (for an edge between $(x_i, y_i)$ and $(x_j, y_j)$):

* Ray intersects if:
  $$
  y_q \in [\min(y_i, y_j), \max(y_i, y_j))
  $$
  and
  $$
  x_q < x_i + \frac{(y_q - y_i)(x_j - x_i)}{(y_j - y_i)}
  $$

#### Example Walkthrough

Polygon: square
$$
(1,1), (5,1), (5,5), (1,5)
$$
Query point $q(3,3)$

* Cast ray to the right from $(3,3)$
* Intersects left edge $(1,1)-(1,5)$ once → count = 1
* Intersects top/bottom edges? no
  → Odd crossings → Inside

Query point $q(6,3)$

* No intersections → count = 0 → Outside

#### Tiny Code (Python Example)

```python
def point_in_polygon(point, polygon):
    x, y = point
    inside = False
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        if ((y1 > y) != (y2 > y)):
            x_intersect = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < x_intersect:
                inside = not inside
    return inside
```

This implements the even–odd rule via a simple parity flip.

#### Why It Matters

* Intuitive and easy to implement
* Works for any simple polygon (convex or concave)
* Foundation for:

  * Point-in-region tests
  * Filling polygons (graphics rasterization)
  * GIS spatial joins

Applications:

* Graphics: hit detection, clipping
* Robotics: occupancy checks
* Mapping: geographic containment
* Simulation: spatial inclusion tests

#### A Gentle Proof (Why It Works)

Each time the ray crosses an edge, the point transitions from outside to inside or vice versa.
Since the polygon boundary is closed, the total number of crossings determines final parity.

Formally:
$$
\text{Inside}(q) = \text{count}(q) \bmod 2
$$

Edges with shared vertices don't double-count if handled consistently (open upper bound).

#### Try It Yourself

1. Draw any polygon on graph paper.
2. Pick a point $q$ and draw a ray to the right.
3. Count edge crossings.
4. Check parity (odd → inside, even → outside).
5. Move $q$ near edges to test special cases.

#### Test Cases

| Polygon            | Point           | Result        |
| ------------------ | --------------- | ------------- |
| Square (1,1)-(5,5) | (3,3)           | Inside        |
| Square (1,1)-(5,5) | (6,3)           | Outside       |
| Triangle           | (edge midpoint) | On boundary   |
| Concave polygon    | interior notch  | Still correct |

#### Complexity

$$
\text{Time: } O(n), \quad \text{Space: } O(1)
$$

The Ray Casting Algorithm is like shining a light through geometry, each crossing flips your perspective, revealing whether the point lies within or beyond the shape's shadow.

### 742 Winding Number

The Winding Number Algorithm is a robust method for the point-in-polygon test.
Unlike Ray Casting, which simply counts crossings, it measures how many times the polygon winds around the query point, capturing not only inside/outside status but also orientation (clockwise vs counterclockwise).

If the winding number is nonzero, the point is inside; if it's zero, it's outside.

#### What Problem Are We Solving?

Given:

* A polygon $P = {v_1, v_2, \ldots, v_n}$
* A query point $q = (x_q, y_q)$

Determine whether $q$ lies inside or outside the polygon, including concave and self-intersecting cases.

The winding number is defined as the total angle swept by the polygon edges around the point:
$$
w(q) = \frac{1}{2\pi} \sum_{i=1}^{n} \Delta\theta_i
$$
where $\Delta\theta_i$ is the signed angle between consecutive edges from $q$.

#### How Does It Work (Plain Language)

Imagine walking along the polygon edges and watching the query point from your path:

* As you traverse, the point seems to rotate around you.
* Each turn contributes an angle to the winding sum.
* If the total turn equals $2\pi$ (or $-2\pi$), you've wrapped around the point once → inside.
* If the total turn equals $0$, you never circled the point → outside.

This is like counting how many times you loop around the point.

#### Step-by-Step Procedure

1. Initialize $w = 0$.
2. For each edge $(v_i, v_{i+1})$:

   * Compute vectors:
     $$
     \mathbf{u} = v_i - q, \quad \mathbf{v} = v_{i+1} - q
     $$
   * Compute signed angle:
     $$
     \Delta\theta = \text{atan2}(\det(\mathbf{u}, \mathbf{v}), \mathbf{u} \cdot \mathbf{v})
     $$
   * Add to total: $w += \Delta\theta$
3. If $|w| > \pi$, the point is inside; else, outside.

Or equivalently:
$$
\text{Inside if } w / 2\pi \ne 0
$$

#### Example Walkthrough

Polygon:
$(0,0), (4,0), (4,4), (0,4)$
Query point $q(2,2)$

At each edge, compute signed turn around $q$.
Total angle sum = $2\pi$ → inside

Query point $q(5,2)$
Total angle sum = $0$ → outside

#### Orientation Handling

The sign of $\Delta\theta$ depends on polygon direction:

* Counterclockwise (CCW) → positive angles
* Clockwise (CW) → negative angles

Winding number can thus also reveal orientation:

* $+1$ → inside CCW
* $-1$ → inside CW
* $0$ → outside

#### Tiny Code (Python Example)

```python
import math

def winding_number(point, polygon):
    xq, yq = point
    w = 0.0
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        u = (x1 - xq, y1 - yq)
        v = (x2 - xq, y2 - yq)
        det = u[0]*v[1] - u[1]*v[0]
        dot = u[0]*v[0] + u[1]*v[1]
        angle = math.atan2(det, dot)
        w += angle
    return abs(round(w / (2 * math.pi))) > 0
```

This computes the total angle swept and checks if it's approximately $2\pi$.

#### Why It Matters

* More robust than Ray Casting (handles self-intersections)
* Works for concave and complex polygons
* Captures orientation information
* Used in computational geometry libraries (CGAL, GEOS, Shapely)

Applications:

* Geospatial analysis (inside boundary detection)
* Graphics (fill rules, even–odd vs nonzero winding)
* Collision detection in irregular shapes
* Vector rendering (SVG uses winding rule)

#### A Gentle Proof (Why It Works)

Each edge contributes an angle turn around $q$.
By summing all such turns, we measure net rotation.
If the polygon encloses $q$, the path wraps around once (total $2\pi$).
If $q$ is outside, turns cancel out (total $0$).

Formally:
$$
w(q) = \frac{1}{2\pi} \sum_{i=1}^{n} \text{atan2}(\det(\mathbf{u_i}, \mathbf{v_i}), \mathbf{u_i} \cdot \mathbf{v_i})
$$
and $w(q) \ne 0$ iff $q$ is enclosed.

#### Try It Yourself

1. Draw a concave polygon (e.g. star shape).
2. Pick a point inside a concavity.
3. Ray Casting may misclassify, but Winding Number will not.
4. Compute angles visually, sum them up.
5. Note sign indicates orientation.

#### Test Cases

| Polygon            | Point  | Result              |
| ------------------ | ------ | ------------------- |
| Square (0,0)-(4,4) | (2,2)  | Inside              |
| Square             | (5,2)  | Outside             |
| Star               | center | Inside              |
| Star               | tip    | Outside             |
| Clockwise polygon  | (2,2)  | Winding number = -1 |

#### Complexity

$$
\text{Time: } O(n), \quad \text{Space: } O(1)
$$

The Winding Number Algorithm doesn't just ask how many times a ray crosses a boundary, it listens to the rotation of space around the point, counting full revolutions to reveal enclosure.

### 743 Convex Polygon Point Test

The Convex Polygon Point Test is a fast and elegant method to determine whether a point lies inside, outside, or on the boundary of a convex polygon.
It relies purely on orientation tests, the cross product signs between the query point and every polygon edge.

Because convex polygons have a consistent "direction" of turn, this method works in linear time and with no branching complexity.

#### What Problem Are We Solving?

Given:

* A convex polygon $P = {v_1, v_2, \ldots, v_n}$
* A query point $q = (x_q, y_q)$

We want to test whether $q$ lies:

* Inside $P$
* On the boundary of $P$
* Outside $P$

This test is specialized for convex polygons, where all interior angles are $\le 180^\circ$ and edges are oriented consistently (clockwise or counterclockwise).

#### How Does It Work (Plain Language)

In a convex polygon, all vertices turn in the same direction (say CCW).
A point is inside if it is always to the same side of every edge.

To test this:

1. Loop through all edges $(v_i, v_{i+1})$.
2. For each edge, compute the cross product between edge vector and the vector from vertex to query point:
   $$
   \text{cross}((v_{i+1} - v_i), (q - v_i))
   $$
3. Record the sign (positive, negative, or zero).
4. If all signs are non-negative (or non-positive) → point is inside or on boundary.
5. If signs differ → point is outside.

#### Cross Product Test

For two vectors
$\mathbf{a} = (x_a, y_a)$, $\mathbf{b} = (x_b, y_b)$

The 2D cross product is:
$$
\text{cross}(\mathbf{a}, \mathbf{b}) = a_x b_y - a_y b_x
$$

In geometry:

* $\text{cross} > 0$: $\mathbf{b}$ is to the left of $\mathbf{a}$ (CCW turn)
* $\text{cross} < 0$: $\mathbf{b}$ is to the right (CW turn)
* $\text{cross} = 0$: points are collinear

#### Step-by-Step Example

Polygon (CCW):
$(0,0), (4,0), (4,4), (0,4)$

Query point $q(2,2)$

Compute for each edge:

| Edge        | Cross product               | Sign |
| ----------- | --------------------------- | ---- |
| (0,0)-(4,0) | $(4,0) \times (2,2) = 8$    | +    |
| (4,0)-(4,4) | $(0,4) \times (-2,2) = 8$   | +    |
| (4,4)-(0,4) | $(-4,0) \times (-2,-2) = 8$ | +    |
| (0,4)-(0,0) | $(0,-4) \times (2,-2) = 8$  | +    |

All positive → Inside

#### Tiny Code (Python Example)

```python
def convex_point_test(point, polygon):
    xq, yq = point
    n = len(polygon)
    sign = 0
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        cross = (x2 - x1) * (yq - y1) - (y2 - y1) * (xq - x1)
        if cross != 0:
            if sign == 0:
                sign = 1 if cross > 0 else -1
            elif sign * cross < 0:
                return "Outside"
    return "Inside/On Boundary"
```

This version detects sign changes efficiently and stops early when mismatch appears.

#### Why It Matters

* Fast, linear time with small constant
* Robust, handles all convex polygons
* No need for trigonometry, angles, or intersection tests
* Works naturally with integer coordinates

Applications:

* Collision checks for convex shapes
* Graphics clipping (Sutherland–Hodgman)
* Convex hull membership tests
* Computational geometry libraries (CGAL, Shapely)

#### A Gentle Proof (Why It Works)

In a convex polygon, all points inside must be to the same side of each edge.
Orientation sign indicates which side a point is on.
If signs differ, point must cross boundary → outside.

Thus:
$$
q \in P \iff \forall i, \ \text{sign}(\text{cross}(v_{i+1} - v_i, q - v_i)) = \text{constant}
$$

This follows from convexity: the polygon lies entirely within a single half-plane for each edge.

#### Try It Yourself

1. Draw a convex polygon (triangle, square, hexagon).
2. Pick a point inside, test sign of cross products.
3. Pick a point outside, note at least one flip in sign.
4. Try a point on boundary, one cross = 0, others same sign.

#### Test Cases

| Polygon            | Point           | Result      |
| ------------------ | --------------- | ----------- |
| Square (0,0)-(4,4) | (2,2)           | Inside      |
| Square             | (5,2)           | Outside     |
| Triangle           | (edge midpoint) | On boundary |
| Hexagon            | (center)        | Inside      |

#### Complexity

$$
\text{Time: } O(n), \quad \text{Space: } O(1)
$$

The Convex Polygon Point Test reads geometry like a compass, always checking direction, ensuring the point lies safely within the consistent turn of a convex path.

### 744 Ear Clipping Triangulation

The Ear Clipping Algorithm is a simple, geometric way to triangulate a simple polygon (convex or concave).
It works by iteratively removing "ears", small triangles that can be safely cut off without crossing the polygon's interior, until only one triangle remains.

This method is widely used in computer graphics, meshing, and geometry processing because it's easy to implement and numerically stable.

#### What Problem Are We Solving?

Given a simple polygon
$$
P = {v_1, v_2, \ldots, v_n}
$$
we want to decompose it into non-overlapping triangles whose union exactly equals $P$.

Triangulation is foundational for:

* Rendering and rasterization
* Finite element analysis
* Computational geometry algorithms

For a polygon with $n$ vertices, every triangulation produces exactly $n-2$ triangles.

#### How Does It Work (Plain Language)

An ear of a polygon is a triangle formed by three consecutive vertices $(v_{i-1}, v_i, v_{i+1})$ such that:

1. The triangle lies entirely inside the polygon, and
2. It contains no other vertex of the polygon inside it.

The algorithm repeatedly clips ears:

1. Identify a vertex that forms an ear.
2. Remove it (and the ear triangle) from the polygon.
3. Repeat until only one triangle remains.

Each "clip" reduces the polygon size by one vertex.

#### Ear Definition (Formal)

Triangle $\triangle (v_{i-1}, v_i, v_{i+1})$ is an ear if:

1. $\triangle$ is convex:
   $$
   \text{cross}(v_i - v_{i-1}, v_{i+1} - v_i) > 0
   $$
2. No other vertex $v_j$ (for $j \ne i-1,i,i+1$) lies inside $\triangle$.

#### Step-by-Step Example

Polygon (CCW): $(0,0), (4,0), (4,4), (2,2), (0,4)$

1. Check each vertex for convexity.
2. Vertex $(4,0)$ forms an ear, triangle $(0,0),(4,0),(4,4)$ contains no other vertices.
3. Clip ear → remove $(4,0)$.
4. Repeat with smaller polygon.
5. Continue until only one triangle remains.

Result: triangulation = 3 triangles.

#### Tiny Code (Python Example)

```python
def is_convex(a, b, c):
    return (b[0] - a[0])*(c[1] - a[1]) - (b[1] - a[1])*(c[0] - a[0]) > 0

def point_in_triangle(p, a, b, c):
    cross1 = (b[0] - a[0])*(p[1] - a[1]) - (b[1] - a[1])*(p[0] - a[0])
    cross2 = (c[0] - b[0])*(p[1] - b[1]) - (c[1] - b[1])*(p[0] - b[0])
    cross3 = (a[0] - c[0])*(p[1] - c[1]) - (a[1] - c[1])*(p[0] - c[0])
    return (cross1 >= 0 and cross2 >= 0 and cross3 >= 0) or (cross1 <= 0 and cross2 <= 0 and cross3 <= 0)

def ear_clipping(polygon):
    triangles = []
    vertices = polygon[:]
    while len(vertices) > 3:
        n = len(vertices)
        for i in range(n):
            a, b, c = vertices[i-1], vertices[i], vertices[(i+1) % n]
            if is_convex(a, b, c):
                ear = True
                for p in vertices:
                    if p not in (a, b, c) and point_in_triangle(p, a, b, c):
                        ear = False
                        break
                if ear:
                    triangles.append((a, b, c))
                    vertices.pop(i)
                    break
    triangles.append(tuple(vertices))
    return triangles
```

This version removes one ear per iteration and terminates after $n-3$ iterations.

#### Why It Matters

* Simple to understand and implement
* Works for any simple polygon (convex or concave)
* Produces consistent triangulations
* Forms basis for many advanced meshing algorithms

Applications:

* Rendering polygons (OpenGL tessellation)
* Physics collision meshes
* Geometric modeling (e.g. GIS, FEM)

#### A Gentle Proof (Why It Works)

Every simple polygon has at least two ears (Meisters' Theorem).
Each ear is a valid triangle that doesn't overlap others.
By clipping one ear per step, the polygon's boundary shrinks, preserving simplicity.
Thus, the algorithm always terminates with $n-2$ triangles.

Time complexity (naive):
$$
O(n^2)
$$
Using spatial acceleration (e.g., adjacency lists):
$$
O(n \log n)
$$

#### Try It Yourself

1. Draw a concave polygon.
2. Find convex vertices.
3. Test each for ear condition (no other vertex inside).
4. Clip ear, redraw polygon.
5. Repeat until full triangulation.

#### Test Cases

| Polygon      | Vertices | Triangles |
| ------------ | -------- | --------- |
| Triangle     | 3        | 1         |
| Convex Quad  | 4        | 2         |
| Concave Pent | 5        | 3         |
| Star shape   | 8        | 6         |

#### Complexity

$$
\text{Time: } O(n^2), \quad \text{Space: } O(n)
$$

The Ear Clipping Triangulation slices geometry like origami, one ear at a time, until every fold becomes a perfect triangle.

### 745 Monotone Polygon Triangulation

The Monotone Polygon Triangulation algorithm is a powerful and efficient method for triangulating y-monotone polygons, polygons whose edges never "backtrack" along the y-axis.
Because of this property, we can sweep from top to bottom, connecting diagonals in a well-ordered fashion, achieving an elegant $O(n)$ time complexity.

#### What Problem Are We Solving?

Given a y-monotone polygon (its boundary can be split into a left and right chain that are both monotonic in y),
we want to split it into non-overlapping triangles.

A polygon is y-monotone if any horizontal line intersects its boundary at most twice.
This structure guarantees that each vertex can be handled incrementally using a stack-based sweep.

We want a triangulation with:

* No edge crossings
* Linear-time construction
* Stable structure for rendering and geometry

#### How Does It Work (Plain Language)

Think of sweeping a horizontal line from top to bottom.
At each vertex, you decide whether to connect it with diagonals to previous vertices, forming new triangles.

The key idea:

1. Sort vertices by y (descending)
2. Classify each vertex as belonging to the left or right chain
3. Use a stack to manage the active chain of vertices
4. Pop and connect when you can form valid diagonals
5. Continue until only the base edge remains

At the end, you get a full triangulation of the polygon.

#### Step-by-Step (Conceptual Flow)

1. Input: a y-monotone polygon
2. Sort vertices in descending y order
3. Initialize stack with first two vertices
4. For each next vertex $v_i$:

   * If $v_i$ is on the opposite chain,
     connect $v_i$ to all vertices in stack, then reset the stack.
   * Else,
     pop vertices forming convex turns, add diagonals, and push $v_i$
5. Continue until one chain remains.

#### Example

Polygon (y-monotone):

```
v1 (top)
|\
| \
|  \
v2  v3
|    \
|     v4
|    /
v5--v6 (bottom)
```

1. Sort vertices by y
2. Identify left chain (v1, v2, v5, v6), right chain (v1, v3, v4, v6)
3. Sweep from top
4. Add diagonals between chains as you descend
5. Triangulation completed in linear time.

#### Tiny Code (Python Pseudocode)

```python
def monotone_triangulation(vertices):
    # vertices sorted by descending y
    stack = [vertices[0], vertices[1]]
    triangles = []
    for i in range(2, len(vertices)):
        current = vertices[i]
        if on_opposite_chain(current, stack[-1]):
            while len(stack) > 1:
                top = stack.pop()
                triangles.append((current, top, stack[-1]))
            stack = [stack[-1], current]
        else:
            top = stack.pop()
            while len(stack) > 0 and is_convex(current, top, stack[-1]):
                triangles.append((current, top, stack[-1]))
                top = stack.pop()
            stack.extend([top, current])
    return triangles
```

Here `on_opposite_chain` and `is_convex` are geometric tests
using cross products and chain labeling.

#### Why It Matters

* Optimal $O(n)$ algorithm for monotone polygons
* A crucial step in general polygon triangulation (used after decomposition)
* Used in:

  * Graphics rendering (OpenGL tessellation)
  * Map engines (GIS)
  * Mesh generation and computational geometry libraries

#### A Gentle Proof (Why It Works)

In a y-monotone polygon:

* The boundary has no self-intersections
* The sweep line always encounters vertices in consistent topological order
* Each new vertex can only connect to visible predecessors

Thus, each edge and vertex is processed once, producing $n-2$ triangles with no redundant operations.

Time complexity:
$$
O(n)
$$

Each vertex is pushed and popped at most once.

#### Try It Yourself

1. Draw a y-monotone polygon (like a mountain slope).
2. Mark left and right chains.
3. Sweep from top to bottom, connecting diagonals.
4. Track stack operations and triangles formed.
5. Verify triangulation produces $n-2$ triangles.

#### Test Cases

| Polygon            | Vertices | Triangles | Time   |
| ------------------ | -------- | --------- | ------ |
| Convex             | 5        | 3         | $O(5)$ |
| Y-Monotone Hexagon | 6        | 4         | $O(6)$ |
| Concave Monotone   | 7        | 5         | $O(7)$ |

#### Complexity

$$
\text{Time: } O(n), \quad \text{Space: } O(n)
$$

The Monotone Polygon Triangulation flows like a waterfall, sweeping smoothly down the polygon's shape, splitting it into perfect, non-overlapping triangles with graceful precision.

### 746 Delaunay Triangulation (Optimal Triangle Quality)

The Delaunay Triangulation is one of the most elegant and fundamental constructions in computational geometry.
It produces a triangulation of a set of points such that no point lies inside the circumcircle of any triangle.
This property maximizes the minimum angle of all triangles, avoiding skinny, sliver-shaped triangles, making it ideal for meshing, interpolation, and graphics.

#### What Problem Are We Solving?

Given a finite set of points
$$
P = {p_1, p_2, \ldots, p_n}
$$
in the plane, we want to connect them into non-overlapping triangles satisfying the Delaunay condition:

> For every triangle in the triangulation, the circumcircle contains no other point of $P$ in its interior.

This gives us a Delaunay Triangulation, noted for:

* Optimal angle quality (max-min angle property)
* Duality with the Voronoi Diagram
* Robustness for interpolation and simulation

#### How Does It Work (Plain Language)

Imagine inflating circles through every triplet of points.
A circle "belongs" to a triangle if no other point is inside it.
The triangulation that respects this rule is the Delaunay triangulation.

Several methods can construct it:

1. Incremental Insertion (Bowyer–Watson): add one point at a time
2. Divide and Conquer: recursively merge Delaunay sets
3. Fortune's Sweep Line: $O(n \log n)$ algorithm
4. Flipping Edges: enforce the empty circle property

Each ensures no triangle violates the empty circumcircle rule.

#### Delaunay Condition (Empty Circumcircle Test)

For triangle with vertices $a(x_a,y_a)$, $b(x_b,y_b)$, $c(x_c,y_c)$ and a query point $p(x_p,y_p)$:

Compute determinant:

$$
\begin{vmatrix}
x_a & y_a & x_a^2 + y_a^2 & 1 \\
x_b & y_b & x_b^2 + y_b^2 & 1 \\
x_c & y_c & x_c^2 + y_c^2 & 1 \\
x_p & y_p & x_p^2 + y_p^2 & 1
\end{vmatrix}
$$


* If result > 0, point $p$ is inside the circumcircle → violates Delaunay
* If ≤ 0, triangle satisfies Delaunay condition

#### Step-by-Step (Bowyer–Watson Method)

1. Start with a super-triangle enclosing all points.
2. For each point $p$:

   * Find all triangles whose circumcircle contains $p$
   * Remove them (forming a cavity)
   * Connect $p$ to all edges on the cavity boundary
3. Repeat until all points are added.
4. Remove triangles connected to the super-triangle's vertices.

#### Tiny Code (Python Sketch)

```python
def delaunay(points):
    # assume helper functions: circumcircle_contains, super_triangle
    triangles = [super_triangle(points)]
    for p in points:
        bad = [t for t in triangles if circumcircle_contains(t, p)]
        edges = []
        for t in bad:
            for e in t.edges():
                if e not in edges:
                    edges.append(e)
                else:
                    edges.remove(e)
        for t in bad:
            triangles.remove(t)
        for e in edges:
            triangles.append(Triangle(e[0], e[1], p))
    return [t for t in triangles if not t.shares_super()]
```

This incremental construction runs in $O(n^2)$, or $O(n \log n)$ with acceleration.

#### Why It Matters

* Quality guarantee: avoids skinny triangles
* Dual structure: forms the basis of Voronoi Diagrams
* Stability: small input changes → small triangulation changes
* Applications:

  * Terrain modeling
  * Mesh generation (FEM, CFD)
  * Interpolation (Natural Neighbor, Sibson)
  * Computer graphics and GIS

#### A Gentle Proof (Why It Works)

For any set of points in general position (no 4 cocircular):

* Delaunay triangulation exists and is unique
* It maximizes minimum angle among all triangulations
* Edge flips restore Delaunay condition:
  if two triangles share an edge and violate the condition,
  flipping the edge increases the smallest angle.

Thus, repeatedly flipping until no violations yields a valid Delaunay triangulation.

#### Try It Yourself

1. Plot random points on a plane.
2. Connect them arbitrarily, then check circumcircles.
3. Flip edges that violate the Delaunay condition.
4. Compare before/after, note improved triangle shapes.
5. Overlay Voronoi diagram (they're dual structures).

#### Test Cases

| Points               | Method           | Triangles | Notes                      |
| -------------------- | ---------------- | --------- | -------------------------- |
| 3 pts                | trivial          | 1         | Always Delaunay            |
| 4 pts forming square | flip-based       | 2         | Diagonal with empty circle |
| Random 10 pts        | incremental      | 16        | Delaunay mesh              |
| Grid points          | divide & conquer | many      | uniform mesh               |

#### Complexity

$$
\text{Time: } O(n \log n), \quad \text{Space: } O(n)
$$

The Delaunay Triangulation builds harmony in the plane, every triangle balanced, every circle empty, every angle wide, a geometry that's both efficient and beautiful.

### 747 Convex Decomposition

The Convex Decomposition algorithm breaks a complex polygon into smaller convex pieces.
Since convex polygons are much easier to work with, for collision detection, rendering, and geometry operations, this decomposition step is often essential in computational geometry and graphics systems.

#### What Problem Are We Solving?

Given a simple polygon (possibly concave), we want to divide it into convex sub-polygons such that:

1. The union of all sub-polygons equals the original polygon.
2. Sub-polygons do not overlap.
3. Each sub-polygon is convex, all interior angles ≤ 180°.

Convex decomposition helps transform difficult geometric tasks (like intersection, clipping, physics simulation) into simpler convex cases.

#### How Does It Work (Plain Language)

Concave polygons "bend inward."
To make them convex, we draw diagonals that split concave regions apart.
The idea:

1. Find vertices that are reflex (interior angle > 180°).
2. Draw diagonals from each reflex vertex to visible non-adjacent vertices inside the polygon.
3. Split the polygon along these diagonals.
4. Repeat until every resulting piece is convex.

You can think of it like cutting folds out of a paper shape until every piece lies flat.

#### Reflex Vertex Test

For vertex sequence $(v_{i-1}, v_i, v_{i+1})$ (CCW order),
compute cross product:

$$
\text{cross}(v_{i+1} - v_i, v_{i-1} - v_i)
$$

* If the result < 0, $v_i$ is reflex (concave turn).
* If > 0, $v_i$ is convex.

Reflex vertices mark where diagonals may be drawn.

#### Step-by-Step Example

Polygon (CCW):
$(0,0), (4,0), (4,2), (2,1), (4,4), (0,4)$

1. Compute orientation at each vertex, $(2,1)$ is reflex.
2. From $(2,1)$, find a visible vertex on the opposite chain (e.g., $(0,4)$).
3. Add diagonal $(2,1)$–$(0,4)$ → polygon splits into two convex parts.
4. Each resulting polygon passes the convexity test.

#### Tiny Code (Python Example)

```python
def cross(o, a, b):
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

def is_reflex(prev, curr, nxt):
    return cross(curr, nxt, prev) < 0

def convex_decomposition(polygon):
    parts = [polygon]
    i = 0
    while i < len(parts):
        poly = parts[i]
        n = len(poly)
        found = False
        for j in range(n):
            if is_reflex(poly[j-1], poly[j], poly[(j+1)%n]):
                for k in range(n):
                    if k not in (j-1, j, (j+1)%n):
                        # naive visibility check (for simplicity)
                        parts.append(poly[j:k+1])
                        parts.append(poly[k:] + poly[:j+1])
                        parts.pop(i)
                        found = True
                        break
            if found: break
        if not found: i += 1
    return parts
```

This basic structure finds reflex vertices and splits polygon recursively.

#### Why It Matters

Convex decomposition underlies many geometry systems:

* Physics engines (Box2D, Chipmunk, Bullet):
  collisions computed per convex part.
* Graphics pipelines:
  rasterization and tessellation simplify to convex polygons.
* Computational geometry:
  many algorithms (e.g., point-in-polygon, intersection) are easier for convex sets.

#### A Gentle Proof (Why It Works)

Every simple polygon can be decomposed into convex polygons using diagonals that lie entirely inside the polygon.
There exists a guaranteed upper bound of $n-3$ diagonals (from polygon triangulation).
Since every convex polygon is trivially decomposed into itself, the recursive cutting terminates.

Thus, convex decomposition is both finite and complete.

#### Try It Yourself

1. Draw a concave polygon (like an arrow or "L" shape).
2. Mark reflex vertices.
3. Add diagonals connecting reflex vertices to visible points.
4. Verify each resulting piece is convex.
5. Count: total triangles ≤ $n-2$.

#### Test Cases

| Polygon     | Vertices | Convex Parts | Notes                    |
| ----------- | -------- | ------------ | ------------------------ |
| Convex      | 5        | 1            | Already convex           |
| Concave "L" | 6        | 2            | Single diagonal split    |
| Star shape  | 8        | 5            | Multiple reflex cuts     |
| Irregular   | 10       | 4            | Sequential decomposition |

#### Complexity

$$
\text{Time: } O(n^2), \quad \text{Space: } O(n)
$$

The Convex Decomposition algorithm untangles geometry piece by piece,
turning a complicated shape into a mosaic of simple, convex forms, the building blocks of computational geometry.

### 748 Polygon Area (Shoelace Formula)

The Shoelace Formula (also called Gauss's Area Formula) is a simple and elegant way to compute the area of any simple polygon, convex or concave, directly from its vertex coordinates.

It's called the "shoelace" method because when you multiply and sum the coordinates in a crisscross pattern, it looks just like lacing up a shoe.

#### What Problem Are We Solving?

Given a polygon defined by its ordered vertices
$$
P = {(x_1, y_1), (x_2, y_2), \ldots, (x_n, y_n)}
$$
we want to find its area efficiently without subdividing or integrating.

The polygon is assumed to be simple (edges do not cross) and closed, meaning $v_{n+1} = v_1$.

#### How Does It Work (Plain Language)

To find the polygon's area, take the sum of the cross-products of consecutive coordinates, one way and the other:

1. Multiply each $x_i$ by the next vertex's $y_{i+1}$.
2. Multiply each $y_i$ by the next vertex's $x_{i+1}$.
3. Subtract the two sums.
4. Take half of the absolute value.

That's it. The pattern of products forms a "shoelace" when written out, hence the name.

#### Formula

$$
A = \frac{1}{2} \Bigg| \sum_{i=1}^{n} (x_i y_{i+1} - x_{i+1} y_i) \Bigg|
$$

where $(x_{n+1}, y_{n+1}) = (x_1, y_1)$ to close the polygon.

#### Example

Polygon:
$(0,0), (4,0), (4,3), (0,4)$

Compute step by step:

| i | $x_i$ | $y_i$ | $x_{i+1}$ | $y_{i+1}$ | $x_i y_{i+1}$ | $y_i x_{i+1}$ |
| - | ----- | ----- | --------- | --------- | ------------- | ------------- |
| 1 | 0     | 0     | 4         | 0         | 0             | 0             |
| 2 | 4     | 0     | 4         | 3         | 12            | 0             |
| 3 | 4     | 3     | 0         | 4         | 0             | 12            |
| 4 | 0     | 4     | 0         | 0         | 0             | 0             |

Now compute:
$$
A = \frac{1}{2} |(0 + 12 + 0 + 0) - (0 + 0 + 12 + 0)| = \frac{1}{2} |12 - 12| = 0
$$

Oops, that means we must check vertex order (CW vs CCW).
Reordering gives positive area:

$$
A = \frac{1}{2} |12 + 16 + 0 + 0 - (0 + 0 + 0 + 0)| = 14
$$

So area = 14 square units.

#### Tiny Code (Python Example)

```python
def polygon_area(points):
    n = len(points)
    area = 0.0
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0

poly = [(0,0), (4,0), (4,3), (0,4)]
print(polygon_area(poly))  # Output: 14.0
```

This version works for both convex and concave polygons, as long as vertices are ordered consistently (CW or CCW).

#### Why It Matters

* Simple and exact (integer arithmetic works perfectly)
* No trigonometry or decomposition needed
* Used everywhere: GIS, CAD, graphics, robotics
* Works for any 2D polygon defined by vertex coordinates.

Applications:

* Compute land parcel areas
* Polygon clipping algorithms
* Geometry-based physics
* Vector graphics (SVG path areas)

#### A Gentle Proof (Why It Works)

The shoelace formula is derived from the line integral form of Green's Theorem:

$$
A = \frac{1}{2} \oint (x,dy - y,dx)
$$

Discretizing along polygon edges gives:

$$
A = \frac{1}{2} \sum_{i=1}^{n} (x_i y_{i+1} - x_{i+1} y_i)
$$

The absolute value ensures area is positive regardless of orientation (CW or CCW).

#### Try It Yourself

1. Take any polygon, triangle, square, or irregular shape.
2. Write coordinates in order.
3. Multiply across, sum one way and subtract the other.
4. Take half the absolute value.
5. Verify by comparing to known geometric area.

#### Test Cases

| Polygon                               | Vertices | Expected Area            |
| ------------------------------------- | -------- | ------------------------ |
| Triangle (0,0),(4,0),(0,3)            | 3        | 6                        |
| Rectangle (0,0),(4,0),(4,3),(0,3)     | 4        | 12                       |
| Parallelogram (0,0),(5,0),(6,3),(1,3) | 4        | 15                       |
| Concave shape                         | 5        | consistent with geometry |

#### Complexity

$$
\text{Time: } O(n), \quad \text{Space: } O(1)
$$

The Shoelace Formula is geometry's arithmetic poetry —
a neat crisscross of numbers that quietly encloses a shape's entire area in a single line of algebra.

### 749 Minkowski Sum

The Minkowski Sum is a geometric operation that combines two shapes by adding their coordinates point by point.
It's a cornerstone in computational geometry, robotics, and motion planning, used for modeling reachable spaces, expanding obstacles, and combining shapes in a mathematically precise way.

#### What Problem Are We Solving?

Given two sets of points (or shapes) in the plane:

$$
A, B \subset \mathbb{R}^2
$$

the Minkowski Sum is defined as the set of all possible sums of one point from $A$ and one from $B$:

$$
A \oplus B = {, a + b \mid a \in A,, b \in B ,}
$$

Intuitively, we "sweep" one shape around another, summing their coordinates, the result is a new shape that represents all possible combinations of positions.

#### How Does It Work (Plain Language)

Think of $A$ and $B$ as two polygons.
To compute $A \oplus B$:

1. Take every vertex in $A$ and add every vertex in $B$.
2. Collect all resulting points.
3. Compute the convex hull of that set.

If both $A$ and $B$ are convex, their Minkowski sum is also convex, and can be computed efficiently by merging edges in sorted angular order (like merging two convex polygons).

If $A$ or $B$ is concave, you can decompose them into convex parts first, compute all pairwise sums, and merge the results.

#### Geometric Meaning

If you think of $B$ as an "object" and $A$ as a "region,"
then $A \oplus B$ represents all locations that $B$ can occupy if its reference point moves along $A$.

For example:

* In robotics, $A$ can be the robot, $B$ can be obstacles, the sum gives all possible collision configurations.
* In graphics, it's used for shape expansion, offsetting, and collision detection.

#### Step-by-Step Example

Let:
$$
A = {(0,0), (2,0), (1,1)}, \quad B = {(0,0), (1,0), (0,1)}
$$

Compute all pairwise sums:

| $a$   | $b$   | $a+b$ |
| ----- | ----- | ----- |
| (0,0) | (0,0) | (0,0) |
| (0,0) | (1,0) | (1,0) |
| (0,0) | (0,1) | (0,1) |
| (2,0) | (0,0) | (2,0) |
| (2,0) | (1,0) | (3,0) |
| (2,0) | (0,1) | (2,1) |
| (1,1) | (0,0) | (1,1) |
| (1,1) | (1,0) | (2,1) |
| (1,1) | (0,1) | (1,2) |

Convex hull of all these points = Minkowski sum polygon.

#### Tiny Code (Python Example)

```python
from itertools import product

def minkowski_sum(A, B):
    points = [(a[0]+b[0], a[1]+b[1]) for a, b in product(A, B)]
    return convex_hull(points)

def convex_hull(points):
    points = sorted(set(points))
    if len(points) <= 1:
        return points
    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
    lower, upper = [], []
    for p in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]
```

This computes all sums and builds a convex hull around them.

#### Why It Matters

* Collision detection:
  $A \oplus (-B)$ tells whether shapes intersect (if origin ∈ sum).
* Motion planning:
  Expanding obstacles by robot shape simplifies pathfinding.
* Graphics and CAD:
  Used for offsetting, buffering, and morphological operations.
* Convex analysis:
  Models addition of convex functions and support sets.

#### A Gentle Proof (Why It Works)

For convex sets $A$ and $B$,
the Minkowski Sum preserves convexity:

$$
\lambda_1 (a_1 + b_1) + \lambda_2 (a_2 + b_2)
= (\lambda_1 a_1 + \lambda_2 a_2) + (\lambda_1 b_1 + \lambda_2 b_2)
\in A \oplus B
$$

for all $\lambda_1, \lambda_2 \ge 0$ and $\lambda_1 + \lambda_2 = 1$.

Thus, $A \oplus B$ is convex.
The sum geometrically represents the vector addition of all points, a direct application of convexity's closure under linear combinations.

#### Try It Yourself

1. Start with two convex polygons (like a square and a triangle).
2. Add every vertex pair and plot the points.
3. Take the convex hull, that's your Minkowski sum.
4. Try flipping one shape ($-B$), the sum shrinks into an intersection test.

#### Test Cases

| Shape A      | Shape B          | Resulting Shape  |
| ------------ | ---------------- | ---------------- |
| Triangle     | Triangle         | Hexagon          |
| Square       | Square           | Larger square    |
| Line segment | Circle           | Thickened line   |
| Polygon      | Negative polygon | Collision region |

#### Complexity

$$
\text{Time: } O(n + m), \quad \text{for convex polygons of sizes } n, m
$$

(using the angular merge algorithm)

The Minkowski Sum is geometry's way of adding ideas —
each shape extends the other, and together they define everything reachable, combinable, and possible within space.

### 750 Polygon Intersection (Weiler–Atherton Clipping)

The Weiler–Atherton Algorithm is a classic and versatile method for computing the intersection, union, or difference of two arbitrary polygons, even concave ones with holes.
It's the geometric heart of clipping systems used in computer graphics, CAD, and geospatial analysis.

#### What Problem Are We Solving?

Given two polygons:

* Subject polygon $S$
* Clip polygon $C$

we want to find the intersection region $S \cap C$, or optionally the union ($S \cup C$) or difference ($S - C$).

Unlike simpler algorithms (like Sutherland–Hodgman) that only handle convex polygons,
Weiler–Atherton works for any simple polygon, convex, concave, or with holes.

#### How Does It Work (Plain Language)

The idea is to walk along the edges of both polygons, switching between them at intersection points, to trace the final clipped region.

Think of it as walking along $S$, and whenever you hit the border of $C$, you decide whether to enter or exit the clipping area.
This path tracing builds the final intersection polygon.

#### Step-by-Step Outline

1. Find intersection points
   Compute all intersections between edges of $S$ and $C$.
   Insert these points into both polygons' vertex lists in correct order.

2. Label intersections as "entry" or "exit"
   Depending on whether you're entering or leaving $C$ when following $S$'s boundary.

3. Traverse polygons

   * Start at an unvisited intersection.
   * If it's an entry, follow along $S$ until you hit the next intersection.
   * Switch to $C$ and continue tracing along its boundary.
   * Alternate between polygons until you return to the starting point.

4. Repeat until all intersections are visited.
   Each closed traversal gives one part of the final result (may be multiple disjoint polygons).

#### Intersection Geometry (Mathematical Test)

For segments $A_1A_2$ and $B_1B_2$,
we compute intersection using the parametric line equations:

$$
A_1 + t(A_2 - A_1) = B_1 + u(B_2 - B_1)
$$

Solving for $t$ and $u$:

$$
t = \frac{(B_1 - A_1) \times (B_2 - B_1)}{(A_2 - A_1) \times (B_2 - B_1)}, \quad
u = \frac{(B_1 - A_1) \times (A_2 - A_1)}{(A_2 - A_1) \times (B_2 - B_1)}
$$

If $0 \le t, u \le 1$, the segments intersect at:

$$
P = A_1 + t(A_2 - A_1)
$$

#### Tiny Code (Python Example)

This sketch shows the conceptual structure (omitting numerical edge cases):

```python
def weiler_atherton(subject, clip):
    intersections = []
    for i in range(len(subject)):
        for j in range(len(clip)):
            p1, p2 = subject[i], subject[(i+1)%len(subject)]
            q1, q2 = clip[j], clip[(j+1)%len(clip)]
            ip = segment_intersection(p1, p2, q1, q2)
            if ip:
                intersections.append(ip)
                subject.insert(i+1, ip)
                clip.insert(j+1, ip)

    result = []
    visited = set()
    for ip in intersections:
        if ip in visited: continue
        polygon = []
        current = ip
        in_subject = True
        while True:
            polygon.append(current)
            visited.add(current)
            next_poly = subject if in_subject else clip
            idx = next_poly.index(current)
            current = next_poly[(idx + 1) % len(next_poly)]
            if current in intersections:
                in_subject = not in_subject
            if current == ip:
                break
        result.append(polygon)
    return result
```

This captures the algorithmic structure, in practice, geometric libraries (like Shapely, CGAL, GEOS) handle precision and topology robustly.

#### Why It Matters

* Handles complex polygons (concave, holes, multiple intersections)
* Works for all boolean operations (intersection, union, difference)
* Foundation for:

  * Computer graphics clipping (rendering polygons inside viewports)
  * GIS spatial analysis (overlay operations)
  * 2D CAD modeling (cutting and merging shapes)

#### A Gentle Proof (Why It Works)

By alternating traversal between polygons at intersection points,
the algorithm preserves topological continuity, the final polygon boundary follows valid edges from both $S$ and $C$.
Because intersections divide polygons into connected boundary fragments,
and every traversal alternates between "inside" and "outside" regions,
each closed path corresponds to a valid piece of the intersection.

Thus, correctness follows from:

* Consistent orientation (CW or CCW)
* Accurate inside/outside tests
* Complete traversal of all intersections

#### Try It Yourself

1. Draw two overlapping polygons (one convex, one concave).
2. Find all intersection points between edges.
3. Label each as entering or exiting.
4. Follow the edges alternating between polygons, trace the intersection region.
5. Fill it, that's $S \cap C$.

#### Test Cases

| Subject Polygon | Clip Polygon | Operation            | Result        |
| --------------- | ------------ | -------------------- | ------------- |
| Rectangle       | Triangle     | Intersection         | Triangle cap  |
| Concave         | Rectangle    | Intersection         | Clipped shape |
| Two rectangles  | Overlap      | Union                | Combined box  |
| Star and circle | Intersection | Complex curve region |               |

#### Complexity

$$
\text{Time: } O((n + m)^2), \quad \text{Space: } O(n + m)
$$

Optimized implementations use spatial indexing to accelerate intersection tests.

The Weiler–Atherton Algorithm turns polygon overlap into a walk along boundaries —
entering, exiting, and rejoining, tracing the precise geometry of how two shapes truly meet.

# Section 76. Spatial Data Structures

### 751 KD-Tree Build

The KD-Tree (short for *k-dimensional tree*) is a data structure used to organize points in a k-dimensional space for fast nearest neighbor and range queries.
It's a recursive, space-partitioning structure, dividing the space with axis-aligned hyperplanes, much like slicing the world into halves again and again.

#### What Problem Are We Solving?

Given a set of points
$$
P = {p_1, p_2, \ldots, p_n} \subset \mathbb{R}^k
$$
we want to build a structure that lets us answer geometric queries efficiently, such as:

* "Which point is nearest to $(x, y, z)$?"
* "Which points lie within this bounding box?"

Instead of checking all points every time ($O(n)$ per query), we build a KD-tree once, enabling searches in $O(\log n)$ on average.

#### How Does It Work (Plain Language)

A KD-tree is a binary tree that recursively splits points by coordinate axes:

1. Choose a splitting axis (e.g., $x$, then $y$, then $x$, … cyclically).
2. Find the median point along that axis.
3. Create a node storing that point, this is your split plane.
4. Recursively build:

   * Left subtree → points with coordinate less than median
   * Right subtree → points with coordinate greater than median

Each node divides space into two half-spaces, creating a hierarchy of nested bounding boxes.

#### Step-by-Step Example (2D)

Points:
$(2,3), (5,4), (9,6), (4,7), (8,1), (7,2)$

Build process:

| Step | Axis | Median Point | Split | Left Points       | Right Points |
| ---- | ---- | ------------ | ----- | ----------------- | ------------ |
| 1    | x    | (7,2)        | x=7   | (2,3),(5,4),(4,7) | (8,1),(9,6)  |
| 2    | y    | (5,4)        | y=4   | (2,3)             | (4,7)        |
| 3    | y    | (8,1)        | y=1   |,                 | (9,6)        |

Final structure:

```
        (7,2)
       /     \
   (5,4)     (8,1)
   /   \        \
(2,3) (4,7)     (9,6)
```

#### Tiny Code (Python Example)

```python
def build_kdtree(points, depth=0):
    if not points:
        return None
    k = len(points[0])
    axis = depth % k
    points.sort(key=lambda p: p[axis])
    median = len(points) // 2
    return {
        'point': points[median],
        'left': build_kdtree(points[:median], depth + 1),
        'right': build_kdtree(points[median + 1:], depth + 1)
    }
```

This recursive builder sorts points by alternating axes and picks the median at each level.

#### Why It Matters

The KD-tree is one of the core structures in computational geometry, with widespread applications:

* Nearest Neighbor Search, find closest points in $O(\log n)$ time
* Range Queries, count or collect points in an axis-aligned box
* Ray Tracing & Graphics, accelerate visibility and intersection checks
* Machine Learning, speed up k-NN classification or clustering
* Robotics / Motion Planning, organize configuration spaces

#### A Gentle Proof (Why It Works)

At each recursion, the median split ensures that:

* The tree height is roughly $\log_2 n$
* Each search descends only one branch per dimension, pruning large portions of space

Thus, building is $O(n \log n)$ on average (due to sorting),
and queries are logarithmic under balanced conditions.

Formally, at each level:
$$
T(n) = 2T(n/2) + O(n) \Rightarrow T(n) = O(n \log n)
$$

#### Try It Yourself

1. Write down 8 random 2D points.
2. Sort them by x-axis, pick median → root node.
3. Recursively sort left and right halves by y-axis → next splits.
4. Draw boundaries (vertical and horizontal lines) for each split.
5. Visualize the partitioning as rectangular regions.

#### Test Cases

| Points       | Dimensions | Expected Depth | Notes            |
| ------------ | ---------- | -------------- | ---------------- |
| 7 random     | 2D         | ~3             | Balanced splits  |
| 1000 random  | 3D         | ~10            | Median-based     |
| 10 collinear | 1D         | 10             | Degenerate chain |
| Grid points  | 2D         | log₂(n)        | Uniform regions  |

#### Complexity

| Operation         | Time              | Space       |
| ----------------- | ----------------- | ----------- |
| Build             | $O(n \log n)$     | $O(n)$      |
| Search            | $O(\log n)$ (avg) | $O(\log n)$ |
| Worst-case search | $O(n)$            | $O(\log n)$ |

The KD-tree is like a geometric filing cabinet —
each split folds space neatly into halves, letting you find the nearest point with just a few elegant comparisons instead of searching the entire world.

### 752 KD-Tree Search

Once a KD-tree is built, the real power comes from fast search operations, finding points near a query location without scanning the entire dataset.
The search exploits the recursive spatial partitioning of the KD-tree, pruning large parts of space that can't possibly contain the nearest point.

#### What Problem Are We Solving?

Given:

* A set of points $P \subset \mathbb{R}^k$ organized in a KD-tree
* A query point $q = (q_1, q_2, \ldots, q_k)$

we want to find:

1. The nearest neighbor of $q$ (point with minimal Euclidean distance)
2. Or all points within a given range (axis-aligned region or radius)

Instead of $O(n)$ brute force, KD-tree search achieves average $O(\log n)$ query time.

#### How Does It Work (Plain Language)

The search descends the KD-tree recursively:

1. At each node, compare the query coordinate on the current split axis.
2. Move into the subtree that contains the query point.
3. When reaching a leaf, record it as the current best.
4. Backtrack:

   * If the hypersphere around the best point so far crosses the splitting plane,
     search the other subtree too (there might be a closer point).
   * Otherwise, prune that branch, it cannot contain a nearer point.
5. Return the closest found.

This pruning is the heart of KD-tree efficiency.

#### Step-by-Step Example (2D Nearest Neighbor)

Points (from the previous build):
$(2,3), (5,4), (9,6), (4,7), (8,1), (7,2)$

Tree root: $(7,2)$, split on x

Query: $q = (9,2)$

1. Compare $q_x=9$ to split $x=7$ → go right subtree
2. Compare $q_y=2$ to split $y=1$ → go up subtree to $(9,6)$
3. Compute distance $(9,6)$ → $d=4$
4. Backtrack: check if circle of radius 4 crosses x=7 plane → yes → explore left
5. Left child $(8,1)$ → $d=1.41$ → better
6. Done → nearest = $(8,1)$

#### Tiny Code (Python Example)

```python
import math

def distance2(a, b):
    return sum((a[i] - b[i])2 for i in range(len(a)))

def nearest_neighbor(tree, point, depth=0, best=None):
    if tree is None:
        return best

    k = len(point)
    axis = depth % k
    next_branch = None
    opposite_branch = None

    if point[axis] < tree['point'][axis]:
        next_branch = tree['left']
        opposite_branch = tree['right']
    else:
        next_branch = tree['right']
        opposite_branch = tree['left']

    best = nearest_neighbor(next_branch, point, depth + 1, best)

    if best is None or distance2(point, tree['point']) < distance2(point, best):
        best = tree['point']

    if (point[axis] - tree['point'][axis])2 < distance2(point, best):
        best = nearest_neighbor(opposite_branch, point, depth + 1, best)

    return best
```

This function recursively explores only necessary branches, pruning away others that can't contain closer points.

#### Why It Matters

KD-tree search is the backbone of many algorithms:

* Machine Learning: k-nearest neighbors (k-NN), clustering
* Computer Graphics: ray-object intersection
* Robotics / Motion Planning: nearest sample search
* Simulation / Physics: proximity detection
* GIS / Spatial Databases: region and radius queries

Without KD-tree search, these tasks would scale linearly with data size.

#### A Gentle Proof (Why It Works)

KD-tree search correctness relies on two geometric facts:

1. The splitting plane divides space into disjoint regions.
2. The nearest neighbor must lie either:

   * in the same region as the query, or
   * across the split, but within a distance smaller than the current best radius.

Thus, pruning based on the distance between $q$ and the splitting plane never eliminates a possible nearer point.
By visiting subtrees only when necessary, the search remains both complete and efficient.

#### Try It Yourself

1. Build a KD-tree for points in 2D.
2. Query a random point, trace recursive calls.
3. Draw the search region and visualize pruned subtrees.
4. Increase data size, note how query time remains near $O(\log n)$.

#### Test Cases

| Query | Expected Nearest | Distance | Notes             |
| ----- | ---------------- | -------- | ----------------- |
| (9,2) | (8,1)            | 1.41     | On right branch   |
| (3,5) | (4,7)            | 2.23     | Left-heavy region |
| (5,3) | (5,4)            | 1.00     | Exact axis match  |

#### Complexity

| Operation                    | Time        | Space       |
| ---------------------------- | ----------- | ----------- |
| Average search               | $O(\log n)$ | $O(\log n)$ |
| Worst-case (degenerate tree) | $O(n)$      | $O(\log n)$ |

The KD-tree search walks through geometric space like a detective with perfect intuition —
checking only where it must, skipping where it can, and always stopping when it finds the closest possible answer.

### 753 Range Search in KD-Tree

The Range Search in a KD-tree is a geometric query that retrieves all points within a given axis-aligned region (a rectangle in 2D, box in 3D, or hyper-rectangle in higher dimensions).
It's a natural extension of KD-tree traversal, but instead of finding one nearest neighbor, we collect all points lying inside a target window.

#### What Problem Are We Solving?

Given:

* A KD-tree containing $n$ points in $k$ dimensions
* A query region (for example, in 2D):
  $$
  R = [x_{\min}, x_{\max}] \times [y_{\min}, y_{\max}]
  $$

we want to find all points $p = (p_1, \ldots, p_k)$ such that:

$$
x_{\min} \le p_1 \le x_{\max}, \quad
y_{\min} \le p_2 \le y_{\max}, \quad \ldots
$$

In other words, all points that lie *inside* the axis-aligned box $R$.

#### How Does It Work (Plain Language)

The algorithm recursively visits KD-tree nodes and prunes branches that can't possibly intersect the query region:

1. At each node, compare the splitting coordinate with the region's bounds.
2. If the node's point lies inside $R$, record it.
3. If the left subtree could contain points inside $R$, search left.
4. If the right subtree could contain points inside $R$, search right.
5. Stop when subtrees fall completely outside the region.

This approach avoids visiting most nodes, only those whose regions overlap the query box.

#### Step-by-Step Example (2D)

KD-tree (from before):

```
        (7,2)
       /     \
   (5,4)     (8,1)
   /   \        \
(2,3) (4,7)     (9,6)
```

Query region:
$$
x \in [4, 8], \quad y \in [1, 5]
$$

Search process:

1. Root (7,2) inside → record.
2. Left child (5,4) inside → record.
3. (2,3) left of region → prune.
4. (4,7) y=7 > 5 → prune.
5. Right child (8,1) inside → record.
6. (9,6) x > 8 → prune.

Result:
Points inside region = (7,2), (5,4), (8,1)

#### Tiny Code (Python Example)

```python
def range_search(tree, region, depth=0, found=None):
    if tree is None:
        return found or []
    if found is None:
        found = []
    k = len(tree['point'])
    axis = depth % k
    point = tree['point']

    # check if point inside region
    if all(region[i][0] <= point[i] <= region[i][1] for i in range(k)):
        found.append(point)

    # explore subtrees if overlapping region
    if region[axis][0] <= point[axis]:
        range_search(tree['left'], region, depth + 1, found)
    if region[axis][1] >= point[axis]:
        range_search(tree['right'], region, depth + 1, found)
    return found
```

Example usage:

```python
region = [(4, 8), (1, 5)]  # x and y bounds
results = range_search(kdtree, region)
```

#### Why It Matters

Range queries are foundational in spatial computing:

* Database indexing (R-tree, KD-tree) → fast filtering
* Graphics → find objects in viewport or camera frustum
* Robotics → retrieve local neighbors for collision checking
* Machine learning → clustering within spatial limits
* GIS systems → spatial joins and map queries

KD-tree range search combines geometric logic with efficient pruning, making it practical for high-speed applications.

#### A Gentle Proof (Why It Works)

Each node in a KD-tree defines a hyper-rectangular region of space.
If this region lies entirely outside the query box, none of its points can be inside, so we safely skip it.
Otherwise, we recurse.

The total number of nodes visited is:
$$
O(n^{1 - \frac{1}{k}} + m)
$$
where $m$ is the number of reported points,
a known bound from multidimensional search theory.

Thus, range search is output-sensitive: it scales with how many points you actually find.

#### Try It Yourself

1. Build a KD-tree with random 2D points.
2. Define a bounding box $[x_1,x_2]\times[y_1,y_2]$.
3. Trace recursive calls, note which branches are pruned.
4. Visualize the query region, confirm returned points fall inside.

#### Test Cases

| Region                   | Expected Points     | Notes           |
| ------------------------ | ------------------- | --------------- |
| $x\in[4,8], y\in[1,5]$   | (5,4), (7,2), (8,1) | 3 points inside |
| $x\in[0,3], y\in[2,4]$   | (2,3)               | Single match    |
| $x\in[8,9], y\in[0,2]$   | (8,1)               | On boundary     |
| $x\in[10,12], y\in[0,5]$ | ∅                   | Empty result    |

#### Complexity

| Operation       | Time                 | Space       |
| --------------- | -------------------- | ----------- |
| Range search    | $O(n^{1 - 1/k} + m)$ | $O(\log n)$ |
| Average (2D–3D) | $O(\sqrt{n} + m)$    | $O(\log n)$ |

The KD-tree range search is like sweeping a flashlight over geometric space —
it illuminates only the parts you care about, leaving the rest in darkness,
and reveals just the points shining inside your query window.

### 754 Nearest Neighbor Search in KD-Tree

The Nearest Neighbor (NN) Search is one of the most important operations on a KD-tree.
It finds the point (or several points) in a dataset that are closest to a given query point in Euclidean space, a problem that appears in clustering, machine learning, graphics, and robotics.

#### What Problem Are We Solving?

Given:

* A set of points $P = {p_1, p_2, \ldots, p_n} \subset \mathbb{R}^k$
* A KD-tree built on those points
* A query point $q \in \mathbb{R}^k$

We want to find:

$$
p^* = \arg\min_{p_i \in P} |p_i - q|
$$

the point $p^*$ closest to $q$ by Euclidean distance (or sometimes Manhattan or cosine distance).

#### How Does It Work (Plain Language)

KD-tree NN search works by recursively descending into the tree, just like a binary search in multiple dimensions.

1. Start at the root.
   Compare the query coordinate along the node's split axis.
   Go left or right depending on whether the query is smaller or greater.

2. Recurse until a leaf node.
   That leaf's point becomes your initial best.

3. Backtrack up the tree.
   At each node:

   * Update the best point if the node's point is closer.
   * Check if the hypersphere around the query (radius = current best distance) crosses the splitting plane.
   * If it does, explore the other subtree, there could be a closer point across the plane.
   * If not, prune that branch.

4. Terminate when you've returned to the root.

Result: the best point is guaranteed to be the true nearest neighbor.

#### Step-by-Step Example (2D)

Points:
$(2,3), (5,4), (9,6), (4,7), (8,1), (7,2)$

KD-tree root: $(7,2)$, split on $x$

Query point: $q = (9,2)$

| Step | Node      | Axis                | Action                | Best (dist²)                      |              |
| ---- | --------- | ------------------- | --------------------- | --------------------------------- | ------------ |
| 1    | (7,2)     | x                   | go right (9 > 7)      | (7,2), $d=4$                      |              |
| 2    | (8,1)     | y                   | go up (2 > 1)         | (8,1), $d=2$                      |              |
| 3    | (9,6)     | y                   | distance = 17 → worse | (8,1), $d=2$                      |              |
| 4    | backtrack | check split plane $ | 9-7                   | =2$, equals $r=√2$ → explore left | (8,1), $d=2$ |
| 5    | done      |,                   |,                     | nearest = (8,1)                   |              |

#### Tiny Code (Python Example)

```python
import math

def dist2(a, b):
    return sum((a[i] - b[i])2 for i in range(len(a)))

def kd_nearest(tree, query, depth=0, best=None):
    if tree is None:
        return best
    k = len(query)
    axis = depth % k
    next_branch = None
    opposite_branch = None

    point = tree['point']
    if query[axis] < point[axis]:
        next_branch, opposite_branch = tree['left'], tree['right']
    else:
        next_branch, opposite_branch = tree['right'], tree['left']

    best = kd_nearest(next_branch, query, depth + 1, best)
    if best is None or dist2(query, point) < dist2(query, best):
        best = point

    # Check if other branch could contain closer point
    if (query[axis] - point[axis])2 < dist2(query, best):
        best = kd_nearest(opposite_branch, query, depth + 1, best)

    return best
```

Usage:

```python
nearest = kd_nearest(kdtree, (9,2))
print("Nearest:", nearest)
```

#### Why It Matters

Nearest neighbor search appears everywhere:

* Machine Learning

  * k-NN classifier
  * Clustering (k-means, DBSCAN)
* Computer Graphics

  * Ray tracing acceleration
  * Texture lookup, sampling
* Robotics

  * Path planning (PRM, RRT*)
  * Obstacle proximity
* Simulation

  * Particle systems and spatial interactions

KD-tree NN search cuts average query time from $O(n)$ to $O(\log n)$, making it practical for real-time use.

#### A Gentle Proof (Why It Works)

The pruning rule is geometrically sound because of two properties:

1. Each subtree lies entirely on one side of the splitting plane.
2. If the query's hypersphere (radius = current best distance) doesn't intersect that plane, no closer point can exist on the other side.

Thus, only subtrees whose bounding region overlaps the sphere are explored, guaranteeing both correctness and efficiency.

In balanced cases:
$$
T(n) \approx O(\log n)
$$
and in degenerate (unbalanced) trees:
$$
T(n) = O(n)
$$

#### Try It Yourself

1. Build a KD-tree for 10 random 2D points.
2. Query a point and trace the recursion.
3. Draw hypersphere of best distance, see which branches are skipped.
4. Compare with brute-force nearest, verify same result.

#### Test Cases

| Query | Expected NN | Distance | Notes             |
| ----- | ----------- | -------- | ----------------- |
| (9,2) | (8,1)       | 1.41     | Right-heavy query |
| (3,5) | (4,7)       | 2.23     | Deep left search  |
| (7,2) | (7,2)       | 0        | Exact hit         |

#### Complexity

| Operation | Average     | Worst Case |
| --------- | ----------- | ---------- |
| Search    | $O(\log n)$ | $O(n)$     |
| Space     | $O(n)$      | $O(n)$     |

The KD-tree nearest neighbor search is like intuition formalized —
it leaps directly to where the answer must be, glances sideways only when geometry demands,
and leaves the rest of the space quietly untouched.

### 755 R-Tree Build

The R-tree is a powerful spatial indexing structure designed to handle rectangles, polygons, and spatial objects, not just points.
It's used in databases, GIS systems, and graphics engines for efficient range queries, overlap detection, and nearest object search.

While KD-trees partition space by coordinate axes, R-trees partition space by bounding boxes that tightly enclose data objects or smaller groups of objects.

#### What Problem Are We Solving?

We need an index structure that supports:

* Fast search for objects overlapping a query region
* Efficient insertions and deletions
* Dynamic growth without rebalancing from scratch

The R-tree provides all three, making it ideal for dynamic, multidimensional spatial data (rectangles, polygons, regions).

#### How It Works (Plain Language)

The idea is to group nearby objects and represent them by their Minimum Bounding Rectangles (MBRs):

1. Each leaf node stores entries of the form `(MBR, object)`.
2. Each internal node stores entries of the form `(MBR, child-pointer)`,
   where MBR covers all child rectangles.
3. The root node's MBR covers the entire dataset.

When inserting or searching, the algorithm traverses these nested bounding boxes, pruning subtrees that do not intersect the query region.

#### Building an R-Tree (Bulk Loading)

There are two main approaches to build an R-tree:

##### 1. Incremental Insertion (Dynamic)

Insert each object one by one using the ChooseSubtree rule:

1. Start from the root.
2. At each level, choose the child whose MBR needs least enlargement to include the new object.
3. If the child overflows (too many entries), split it using a heuristic like Quadratic Split or Linear Split.
4. Update parent MBRs upward.

##### 2. Bulk Loading (Static)

For large static datasets, sort objects by spatial order (e.g., Hilbert or Z-order curve), then pack them level by level to minimize overlap.

#### Example (2D Rectangles)

Suppose we have 8 objects, each with bounding boxes:

| Object | Rectangle $(x_{\min}, y_{\min}, x_{\max}, y_{\max})$ |
| ------ | ---------------------------------------------------- |
| A      | (1, 1, 2, 2)                                         |
| B      | (2, 2, 3, 3)                                         |
| C      | (8, 1, 9, 2)                                         |
| D      | (9, 3, 10, 4)                                        |
| E      | (5, 5, 6, 6)                                         |
| F      | (6, 6, 7, 7)                                         |
| G      | (3, 8, 4, 9)                                         |
| H      | (4, 9, 5, 10)                                        |

If each node can hold 4 entries, we might group as:

* Node 1 → {A, B, C, D}
  MBR = (1,1,10,4)
* Node 2 → {E, F, G, H}
  MBR = (3,5,7,10)
* Root → {Node 1, Node 2}
  MBR = (1,1,10,10)

This hierarchical nesting enables fast region queries.

#### Tiny Code (Python Example)

A simplified static R-tree builder:

```python
def build_rtree(objects, max_entries=4):
    if len(objects) <= max_entries:
        return {'children': objects, 'leaf': True,
                'mbr': compute_mbr(objects)}

    # sort by x-center for grouping
    objects.sort(key=lambda o: (o['mbr'][0] + o['mbr'][2]) / 2)
    groups = [objects[i:i+max_entries] for i in range(0, len(objects), max_entries)]

    children = [{'children': g, 'leaf': True, 'mbr': compute_mbr(g)} for g in groups]
    return {'children': children, 'leaf': False, 'mbr': compute_mbr(children)}

def compute_mbr(items):
    xmin = min(i['mbr'][0] for i in items)
    ymin = min(i['mbr'][1] for i in items)
    xmax = max(i['mbr'][2] for i in items)
    ymax = max(i['mbr'][3] for i in items)
    return (xmin, ymin, xmax, ymax)
```

#### Why It Matters

R-trees are widely used in:

* Spatial Databases (PostGIS, SQLite's R-Tree extension)
* Game Engines (collision and visibility queries)
* GIS Systems (map data indexing)
* CAD and Graphics (object selection and culling)
* Robotics / Simulation (spatial occupancy grids)

R-trees generalize KD-trees to handle *objects with size and shape*, not just points.

#### A Gentle Proof (Why It Works)

R-tree correctness depends on two geometric invariants:

1. Every child's bounding box is fully contained within its parent's MBR.
2. Every leaf MBR covers its stored object.

Because the structure preserves these containment relationships,
any query that intersects a parent box must check only relevant subtrees,
ensuring completeness and correctness.

The efficiency comes from minimizing overlap between sibling MBRs,
which reduces unnecessary subtree visits.

#### Try It Yourself

1. Create several rectangles and visualize their bounding boxes.
2. Group them manually into MBR clusters.
3. Draw the nested rectangles that represent parent nodes.
4. Perform a query like "all objects intersecting (2,2)-(6,6)" and trace which boxes are visited.

#### Test Cases

| Query Box     | Expected Results | Notes         |
| ------------- | ---------------- | ------------- |
| (1,1)-(3,3)   | A, B             | within Node 1 |
| (5,5)-(7,7)   | E, F             | within Node 2 |
| (8,2)-(9,4)   | C, D             | right group   |
| (0,0)-(10,10) | all              | full overlap  |

#### Complexity

| Operation | Average     | Worst Case |
| --------- | ----------- | ---------- |
| Search    | $O(\log n)$ | $O(n)$     |
| Insert    | $O(\log n)$ | $O(n)$     |
| Space     | $O(n)$      | $O(n)$     |

The R-tree is the quiet geometry librarian —
it files shapes neatly into nested boxes,
so that when you ask "what's nearby?",
it opens only the drawers that matter.

### 756 R*-Tree

The R*-Tree is an improved version of the R-tree that focuses on minimizing overlap and coverage between bounding boxes.
By carefully choosing where and how to insert and split entries, it achieves much better performance for real-world spatial queries.

It is the default index in many modern spatial databases (like PostGIS and SQLite) because it handles dynamic insertions efficiently while keeping query times low.

#### What Problem Are We Solving?

In a standard R-tree, bounding boxes can overlap significantly.
This causes search inefficiency, since a query region may need to explore multiple overlapping subtrees.

The R*-Tree solves this by refining two operations:

1. Insertion, tries to minimize both area and overlap increase.
2. Split, reorganizes entries to reduce future overlap.

As a result, the tree maintains tighter bounding boxes and faster search times.

#### How It Works (Plain Language)

R*-Tree adds a few enhancements on top of the regular R-tree algorithm:

1. ChooseSubtree

   * Select the child whose bounding box requires the smallest *enlargement* to include the new entry.
   * If multiple choices exist, prefer the one with smaller *overlap area* and smaller *total area*.

2. Forced Reinsert

   * When a node overflows, instead of splitting immediately, remove a small fraction of entries (typically 30%),
     and reinsert them higher up in the tree.
   * This "shake-up" redistributes objects and improves spatial clustering.

3. Split Optimization

   * When splitting is inevitable, use heuristics to minimize overlap and perimeter rather than just area.

4. Reinsertion Cascades

   * Reinsertions can propagate upward, slightly increasing insert cost,
     but producing tighter and more balanced trees.

#### Example (2D Rectangles)

Suppose we are inserting a new rectangle $R_{\text{new}}$ into a node that already contains:

| Rectangle | Area | Overlap with others |
| --------- | ---- | ------------------- |
| A         | 4    | small               |
| B         | 6    | large               |
| C         | 5    | moderate            |

In a normal R-tree, we might choose A or B arbitrarily if enlargement is similar.
In an R*-tree, we prefer the child that minimizes:

$$
\Delta \text{Overlap} + \Delta \text{Area}
$$

and if still tied, the one with smaller perimeter.

This yields spatially compact, low-overlap partitions.

#### Tiny Code (Conceptual Pseudocode)

```python
def choose_subtree(node, rect):
    best = None
    best_metric = float('inf')
    for child in node.children:
        enlargement = area_enlargement(child.mbr, rect)
        overlap_increase = overlap_delta(node.children, child, rect)
        metric = (overlap_increase, enlargement, area(child.mbr))
        if metric < best_metric:
            best_metric = metric
            best = child
    return best

def insert_rstar(node, rect, obj):
    if node.is_leaf:
        node.entries.append((rect, obj))
        if len(node.entries) > MAX_ENTRIES:
            handle_overflow(node)
    else:
        child = choose_subtree(node, rect)
        insert_rstar(child, rect, obj)
        node.mbr = recompute_mbr(node.entries)
```

#### Why It Matters

R*-Trees are used in nearly every spatial system where performance matters:

* Databases: PostgreSQL / PostGIS, SQLite, MySQL
* GIS and mapping: real-time region and proximity queries
* Computer graphics: visibility culling and collision detection
* Simulation and robotics: spatial occupancy grids
* Machine learning: range queries on embeddings or high-dimensional data

They represent a balance between update cost and query speed that works well in both static and dynamic datasets.

#### A Gentle Proof (Why It Works)

Let each node's MBR be $B_i$ and the query region $Q$.
For every child node, overlap is defined as:

$$
\text{Overlap}(B_i, B_j) = \text{Area}(B_i \cap B_j)
$$

When inserting a new entry, R*-Tree tries to minimize:

$$
\Delta \text{Overlap} + \Delta \text{Area} + \lambda \times \Delta \text{Margin}
$$

for some small $\lambda$.
This heuristic empirically minimizes the expected number of nodes visited during a query.

Over time, the tree converges toward a balanced, low-overlap hierarchy,
which is why it consistently outperforms the basic R-tree.

#### Try It Yourself

1. Insert rectangles into both an R-tree and an R*-tree.
2. Compare the bounding box overlap at each level.
3. Run a range query, count how many nodes each algorithm visits.
4. Visualize, R*-tree boxes will be more compact and disjoint.

#### Test Cases

| Operation              | Basic R-Tree     | R*-Tree          | Comment                 |
| ---------------------- | ---------------- | ---------------- | ----------------------- |
| Insert 1000 rectangles | Overlap 60%      | Overlap 20%      | R*-Tree clusters better |
| Query (region)         | 45 nodes visited | 18 nodes visited | Faster search           |
| Bulk load              | Similar time     | Slightly slower  | But better structure    |

#### Complexity

| Operation | Average     | Worst Case |
| --------- | ----------- | ---------- |
| Search    | $O(\log n)$ | $O(n)$     |
| Insert    | $O(\log n)$ | $O(n)$     |
| Space     | $O(n)$      | $O(n)$     |

The R*-Tree is the patient cartographer's upgrade —
it doesn't just file shapes into drawers,
it reorganizes them until every map edge lines up cleanly,
so when you look for something, you find it fast and sure.

### 757 Quad Tree

The Quad Tree is a simple yet elegant spatial data structure used to recursively subdivide a two-dimensional space into four quadrants (or regions).
It is ideal for indexing spatial data like images, terrains, game maps, and geometric objects that occupy distinct regions of the plane.

Unlike KD-trees that split by coordinate value, a Quad Tree splits space itself, not the data, dividing the plane into equal quadrants at each level.

#### What Problem Are We Solving?

We want a way to represent spatial occupancy or hierarchical subdivision efficiently for 2D data.
Typical goals include:

* Storing and querying geometric data (points, rectangles, regions).
* Supporting fast lookup: *"What's in this area?"*
* Enabling hierarchical simplification or rendering (e.g., in computer graphics or GIS).

Quad trees make it possible to store both sparse and dense regions efficiently by adapting their depth locally.

#### How It Works (Plain Language)

Think of a large square region containing all your data.

1. Start with the root square (the whole region).
2. If the region contains more than a threshold number of points (say, 1 or 4), subdivide it into 4 equal quadrants:

   * NW (north-west)
   * NE (north-east)
   * SW (south-west)
   * SE (south-east)
3. Recursively repeat subdivision for each quadrant that still contains too many points.
4. Each leaf node then holds a small number of points or objects.

This creates a tree whose structure mirrors the spatial distribution of data, deeper where it's dense, shallower where it's sparse.

#### Example (Points in 2D Space)

Suppose we have these 2D points in a 10×10 grid:
$(1,1), (2,3), (8,2), (9,8), (4,6)$

* The root square covers $(0,0)$–$(10,10)$.
* It subdivides at midpoint $(5,5)$.

  * NW: $(0,5)$–$(5,10)$ → contains $(4,6)$
  * NE: $(5,5)$–$(10,10)$ → contains $(9,8)$
  * SW: $(0,0)$–$(5,5)$ → contains $(1,1), (2,3)$
  * SE: $(5,0)$–$(10,5)$ → contains $(8,2)$

This hierarchical layout makes region queries intuitive and fast.

#### Tiny Code (Python Example)

```python
class QuadTree:
    def __init__(self, boundary, capacity=1):
        self.boundary = boundary  # (x, y, w, h)
        self.capacity = capacity
        self.points = []
        self.divided = False

    def insert(self, point):
        x, y = point
        bx, by, w, h = self.boundary
        if not (bx <= x < bx + w and by <= y < by + h):
            return False  # out of bounds

        if len(self.points) < self.capacity:
            self.points.append(point)
            return True
        else:
            if not self.divided:
                self.subdivide()
            return (self.nw.insert(point) or self.ne.insert(point) or
                    self.sw.insert(point) or self.se.insert(point))

    def subdivide(self):
        bx, by, w, h = self.boundary
        hw, hh = w / 2, h / 2
        self.nw = QuadTree((bx, by + hh, hw, hh), self.capacity)
        self.ne = QuadTree((bx + hw, by + hh, hw, hh), self.capacity)
        self.sw = QuadTree((bx, by, hw, hh), self.capacity)
        self.se = QuadTree((bx + hw, by, hw, hh), self.capacity)
        self.divided = True
```

Usage:

```python
qt = QuadTree((0, 0, 10, 10), 1)
for p in [(1,1), (2,3), (8,2), (9,8), (4,6)]:
    qt.insert(p)
```

#### Why It Matters

Quad Trees are foundational in computer graphics, GIS, and robotics:

* Image processing: storing pixels or regions for compression and filtering.
* Game engines: collision detection, visibility queries, terrain simplification.
* Geographic data: hierarchical tiling for map rendering.
* Robotics: occupancy grids for path planning.

They adapt naturally to spatial density, storing more detail where needed.

#### A Gentle Proof (Why It Works)

Let the dataset have $n$ points, with uniform distribution in a 2D region of area $A$.
Each subdivision reduces the area per node by a factor of 4, and the expected number of nodes is proportional to $O(n)$ if the distribution is not pathological.

For uniformly distributed points:
$$
\text{Height} \approx O(\log_4 n)
$$

And query cost for rectangular regions is:
$$
T(n) = O(\sqrt{n})
$$
in practice, since only relevant quadrants are visited.

The adaptive depth ensures that dense clusters are represented compactly, while sparse areas remain shallow.

#### Try It Yourself

1. Insert 20 random points into a Quad Tree and draw it (each subdivision as a smaller square).
2. Perform a query: "All points in rectangle (3,3)-(9,9)" and count nodes visited.
3. Compare with a brute-force scan.
4. Try reducing capacity to 1 or 2, see how the structure deepens.

#### Test Cases

| Query Rectangle | Expected Points | Notes                |
| --------------- | --------------- | -------------------- |
| (0,0)-(5,5)     | (1,1), (2,3)    | Lower-left quadrant  |
| (5,0)-(10,5)    | (8,2)           | Lower-right quadrant |
| (5,5)-(10,10)   | (9,8)           | Upper-right quadrant |
| (0,5)-(5,10)    | (4,6)           | Upper-left quadrant  |

#### Complexity

| Operation       | Average       | Worst Case |
| --------------- | ------------- | ---------- |
| Insert          | $O(\log n)$   | $O(n)$     |
| Search (region) | $O(\sqrt{n})$ | $O(n)$     |
| Space           | $O(n)$        | $O(n)$     |

The Quad Tree is like a painter's grid —
it divides the world just enough to notice where color changes,
keeping the canvas both detailed and simple to navigate.

### 758 Octree

The Octree is the 3D extension of the Quad Tree.
Instead of dividing space into four quadrants, it divides a cube into eight octants, recursively.
This simple idea scales beautifully from 2D maps to 3D worlds, perfect for graphics, physics, and spatial simulations.

Where a Quad Tree helps us reason about pixels and tiles, an Octree helps us reason about voxels, volumes, and objects in 3D.

#### What Problem Are We Solving?

We need a data structure to represent and query 3D spatial information efficiently.

Typical goals:

* Store and locate 3D points, meshes, or objects.
* Perform collision detection or visibility culling.
* Represent volumetric data (e.g., 3D scans, densities, occupancy grids).
* Speed up ray tracing or rendering by hierarchical pruning.

An Octree balances detail and efficiency, dividing dense regions finely while keeping sparse areas coarse.

#### How It Works (Plain Language)

An Octree divides space recursively:

1. Start with a cube containing all data points or objects.
2. If a cube contains more than a threshold number of items (e.g., 4), subdivide it into 8 equal sub-cubes (octants).
3. Each node stores pointers to its children, which cover:

   * Front-Top-Left (FTL)
   * Front-Top-Right (FTR)
   * Front-Bottom-Left (FBL)
   * Front-Bottom-Right (FBR)
   * Back-Top-Left (BTL)
   * Back-Top-Right (BTR)
   * Back-Bottom-Left (BBL)
   * Back-Bottom-Right (BBR)
4. Recursively subdivide until each leaf cube contains few enough objects.

This recursive space partition forms a hierarchical map of 3D space.

#### Example (Points in 3D Space)

Imagine these 3D points (in a cube from (0,0,0) to (8,8,8)):
$(1,2,3), (7,6,1), (3,5,4), (6,7,7), (2,1,2)$

The first subdivision occurs at the cube's center $(4,4,4)$.
Each child cube covers one of eight octants:

* $(0,0,0)-(4,4,4)$ → contains $(1,2,3), (2,1,2)$
* $(4,0,0)-(8,4,4)$ → contains $(7,6,1)$ (later excluded due to y>4)
* $(0,4,4)-(4,8,8)$ → contains $(3,5,4)$
* $(4,4,4)-(8,8,8)$ → contains $(6,7,7)$

Each sub-cube subdivides only if needed, creating a locally adaptive representation.

#### Tiny Code (Python Example)

```python
class Octree:
    def __init__(self, boundary, capacity=2):
        self.boundary = boundary  # (x, y, z, size)
        self.capacity = capacity
        self.points = []
        self.children = None

    def insert(self, point):
        x, y, z = point
        bx, by, bz, s = self.boundary
        if not (bx <= x < bx + s and by <= y < by + s and bz <= z < bz + s):
            return False  # point out of bounds

        if len(self.points) < self.capacity:
            self.points.append(point)
            return True

        if self.children is None:
            self.subdivide()

        for child in self.children:
            if child.insert(point):
                return True
        return False

    def subdivide(self):
        bx, by, bz, s = self.boundary
        hs = s / 2
        self.children = []
        for dx in [0, hs]:
            for dy in [0, hs]:
                for dz in [0, hs]:
                    self.children.append(Octree((bx + dx, by + dy, bz + dz, hs), self.capacity))
```

#### Why It Matters

Octrees are a cornerstone of modern 3D computation:

* Computer graphics: view frustum culling, shadow mapping, ray tracing.
* Physics engines: broad-phase collision detection.
* 3D reconstruction: storing voxelized scenes (e.g., Kinect, LiDAR).
* GIS and simulations: volumetric data and spatial queries.
* Robotics: occupancy mapping in 3D environments.

Because Octrees adapt to data density, they dramatically reduce memory and query time in 3D problems.

#### A Gentle Proof (Why It Works)

At each level, the cube divides into $8$ smaller cubes.
If a region is uniformly filled, the height of the tree is:

$$
h = O(\log_8 n) = O(\log n)
$$

Each query visits only the cubes that overlap the query region.
Thus, the expected query time is sublinear:

$$
T_{\text{query}} = O(n^{2/3})
$$

For sparse data, the number of active nodes is much smaller than $n$,
so in practice both insert and query run near $O(\log n)$.

#### Try It Yourself

1. Insert random 3D points in a cube $(0,0,0)$–$(8,8,8)$.
2. Draw a recursive cube diagram showing which regions subdivide.
3. Query: "Which points lie within $(2,2,2)$–$(6,6,6)$?"
4. Compare with brute-force search.

#### Test Cases

| Query Cube      | Expected Points  | Notes             |
| --------------- | ---------------- | ----------------- |
| (0,0,0)-(4,4,4) | (1,2,3), (2,1,2) | Lower octant      |
| (4,4,4)-(8,8,8) | (6,7,7)          | Upper far octant  |
| (2,4,4)-(4,8,8) | (3,5,4)          | Upper near octant |

#### Complexity

| Operation       | Average      | Worst Case |
| --------------- | ------------ | ---------- |
| Insert          | $O(\log n)$  | $O(n)$     |
| Search (region) | $O(n^{2/3})$ | $O(n)$     |
| Space           | $O(n)$       | $O(n)$     |

The Octree is the quiet architect of 3D space —
it builds invisible scaffolds inside volume and light,
where each cube knows just enough of its world to keep everything fast, clean, and infinite.

### 759 BSP Tree (Binary Space Partition Tree)

A BSP Tree, or *Binary Space Partitioning Tree*, is a data structure for recursively subdividing space using planes.
While quadtrees and octrees divide space into fixed quadrants or cubes, BSP trees divide it by arbitrary hyperplanes, making them incredibly flexible for geometry, visibility, and rendering.

This structure was a major breakthrough in computer graphics and computational geometry, used in early 3D engines like *DOOM* and still powering CAD, physics, and spatial reasoning systems today.

#### What Problem Are We Solving?

We need a general, efficient way to:

* Represent and query complex 2D or 3D scenes.
* Determine visibility (what surfaces are seen first).
* Perform collision detection, ray tracing, or CSG (constructive solid geometry).

Unlike quadtrees or octrees that assume axis-aligned splits, a BSP tree can partition space by any plane, perfectly fitting complex geometry.

#### How It Works (Plain Language)

1. Start with a set of geometric primitives (lines, polygons, or polyhedra).
2. Pick one as the splitting plane.
3. Divide all other objects into two sets:

   * Front set: those lying in front of the plane.
   * Back set: those behind the plane.
4. Recursively partition each side with new planes until each region contains a small number of primitives.

The result is a binary tree:

* Each internal node represents a splitting plane.
* Each leaf node represents a convex subspace (a region of space fully divided).

#### Example (2D Illustration)

Imagine you have three lines dividing a 2D plane:

* Line A: vertical
* Line B: diagonal
* Line C: horizontal

Each line divides space into two half-planes.
After all splits, you end up with convex regions (non-overlapping cells).

Each region corresponds to a leaf in the BSP tree,
and traversing the tree in front-to-back order gives a correct painter's algorithm rendering —
drawing closer surfaces over farther ones.

#### Step-by-Step Summary

1. Choose a splitting polygon or plane (e.g., one from your object list).
2. Classify every other object as in front, behind, or intersecting the plane.

   * If it intersects, split it along the plane.
3. Recursively build the tree for front and back sets.
4. For visibility or ray tracing, traverse nodes in order depending on the viewer position relative to the plane.

#### Tiny Code (Simplified Python Pseudocode)

```python
class BSPNode:
    def __init__(self, plane, front=None, back=None):
        self.plane = plane
        self.front = front
        self.back = back

def build_bsp(objects):
    if not objects:
        return None
    plane = objects[0]  # pick splitting plane
    front, back = [], []
    for obj in objects[1:]:
        side = classify(obj, plane)
        if side == 'front':
            front.append(obj)
        elif side == 'back':
            back.append(obj)
        else:  # intersecting
            f_part, b_part = split(obj, plane)
            front.append(f_part)
            back.append(b_part)
    return BSPNode(plane, build_bsp(front), build_bsp(back))
```

Here `classify` determines which side of the plane an object lies on,
and `split` divides intersecting objects along that plane.

#### Why It Matters

BSP Trees are essential in:

* 3D rendering engines, sorting polygons for the painter's algorithm.
* Game development, efficient visibility and collision queries.
* Computational geometry, point-in-polygon and ray intersection tests.
* CSG modeling, combining solids with boolean operations (union, intersection, difference).
* Robotics and simulation, representing free and occupied 3D space.

#### A Gentle Proof (Why It Works)

Every splitting plane divides space into two convex subsets.
Since convex regions never overlap, each point in space belongs to exactly one leaf.

For $n$ splitting planes, the number of convex regions formed is $O(n^2)$ in 2D and $O(n^3)$ in 3D,
but queries can be answered in logarithmic time on average by traversing only relevant branches.

Mathematically,
if $Q$ is the query point and $P_i$ are the planes,
then each comparison
$$
\text{sign}(a_i x + b_i y + c_i z + d_i)
$$
guides traversal, producing a deterministic, spatially consistent partition.

#### Try It Yourself

1. Draw 3 polygons and use each as a splitting plane.
2. Color the resulting regions after each split.
3. Store them in a BSP tree (front and back).
4. Render polygons back-to-front from a given viewpoint, you'll notice no depth sorting errors.

#### Test Cases

| Scene                     | Planes | Regions | Use                     |
| ------------------------- | ------ | ------- | ----------------------- |
| Simple room               | 3      | 8       | Visibility ordering     |
| Indoor map                | 20     | 200+    | Collision and rendering |
| CSG model (cube ∩ sphere) | 6      | 50+     | Boolean modeling        |

#### Complexity

| Operation | Average       | Worst Case |
| --------- | ------------- | ---------- |
| Build     | $O(n \log n)$ | $O(n^2)$   |
| Query     | $O(\log n)$   | $O(n)$     |
| Space     | $O(n)$        | $O(n^2)$   |

The BSP Tree is the geometric philosopher's tool —
it slices the world with planes of thought,
sorting front from back, visible from hidden,
until every region is clear, and nothing overlaps in confusion.

### 760 Morton Order (Z-Curve)

The Morton Order, also known as the Z-Order Curve, is a clever way to map multidimensional data (2D, 3D, etc.) into one dimension while preserving spatial locality.
It's not a tree by itself, but it underpins many spatial data structures, including quadtrees, octrees, and R-trees, because it allows hierarchical indexing without explicitly storing the tree.

It's called "Z-order" because when visualized, the traversal path of the curve looks like a repeating Z pattern across space.

#### What Problem Are We Solving?

We want a way to linearize spatial data so that nearby points in space remain nearby in sorted order.
That's useful for:

* Sorting and indexing spatial data efficiently.
* Bulk-loading spatial trees like R-trees or B-trees.
* Improving cache locality and disk access in databases.
* Building memory-efficient hierarchical structures.

Morton order provides a compact and computationally cheap way to do this by using bit interleaving.

#### How It Works (Plain Language)

Take two or three coordinates, for example, $(x, y)$ in 2D or $(x, y, z)$ in 3D —
and interleave their bits to create a single Morton code (integer).

For 2D:

1. Convert $x$ and $y$ to binary.
   Example: $x = 5 = (101)_2$, $y = 3 = (011)_2$.
2. Interleave bits: take one bit from $x$, one from $y$, alternating:
   $x_2 y_2 x_1 y_1 x_0 y_0$.
3. The result $(100111)_2 = 39$ is the Morton code for $(5, 3)$.

This number represents the Z-order position of the point.

When you sort points by Morton code, nearby coordinates tend to stay near each other in the sorted order —
so 2D or 3D proximity translates roughly into 1D proximity.

#### Example (2D Visualization)

| Point $(x, y)$ | Binary $(x, y)$ | Morton Code | Order |
| -------------- | --------------- | ----------- | ----- |
| (0, 0)         | (000, 000)      | 000000      | 0     |
| (1, 0)         | (001, 000)      | 000001      | 1     |
| (0, 1)         | (000, 001)      | 000010      | 2     |
| (1, 1)         | (001, 001)      | 000011      | 3     |
| (2, 2)         | (010, 010)      | 001100      | 12    |

Plotting these in 2D gives the characteristic "Z" shape, recursively repeated at each scale.

#### Tiny Code (Python Example)

```python
def interleave_bits(x, y):
    z = 0
    for i in range(32):  # assuming 32-bit coordinates
        z |= ((x >> i) & 1) << (2 * i)
        z |= ((y >> i) & 1) << (2 * i + 1)
    return z

def morton_2d(points):
    return sorted(points, key=lambda p: interleave_bits(p[0], p[1]))

points = [(1,0), (0,1), (1,1), (2,2), (0,0)]
print(morton_2d(points))
```

This produces the Z-order traversal of the points.

#### Why It Matters

Morton order bridges geometry and data systems:

* Databases: Used for bulk-loading R-trees (called *packed R-trees*).
* Graphics: Texture mipmapping and spatial sampling.
* Parallel computing: Block decomposition of grids (spatial cache efficiency).
* Numerical simulation: Adaptive mesh refinement indexing.
* Vector databases: Fast approximate nearest neighbor grouping.

Because it preserves *spatial locality* and supports *bitwise computation*, it's much faster than sorting by Euclidean distance or using complex data structures for initial indexing.

#### A Gentle Proof (Why It Works)

The Z-curve recursively subdivides space into quadrants (in 2D) or octants (in 3D), visiting them in a depth-first order.
At each recursion level, the most significant interleaved bits determine which quadrant or octant a point belongs to.

For a 2D point $(x, y)$:

$$
M(x, y) = \sum_{i=0}^{b-1} \left[ (x_i \cdot 2^{2i}) + (y_i \cdot 2^{2i+1}) \right]
$$

where $x_i, y_i$ are the bits of $x$ and $y$.

This mapping preserves hierarchical proximity:
if two points share their first $k$ bits in interleaved form,
they lie within the same $2^{-k}$-sized region of space.

#### Try It Yourself

1. Write down binary coordinates for 8 points $(x, y)$ in a 4×4 grid.
2. Interleave their bits to get Morton codes.
3. Sort by the codes, then plot points to see the "Z" pattern.
4. Observe that nearby points share many leading bits in their codes.

#### Test Cases

| $(x, y)$ | Morton Code | Binary | Result        |
| -------- | ----------- | ------ | ------------- |
| (0, 0)   | 0           | 0000   | Start         |
| (1, 0)   | 1           | 0001   | Right         |
| (0, 1)   | 2           | 0010   | Up            |
| (1, 1)   | 3           | 0011   | Upper-right   |
| (2, 0)   | 4           | 0100   | Next quadrant |

#### Complexity

| Operation      | Time                 | Space  |
| -------------- | -------------------- | ------ |
| Encoding (2D)  | $O(b)$               | $O(1)$ |
| Sorting        | $O(n \log n)$        | $O(n)$ |
| Query locality | $O(1)$ (approximate) |,      |

The Morton Order (Z-Curve) is the mathematician's compass —
it traces a single line that dances through every cell of a grid,
folding multidimensional worlds into a one-dimensional thread,
without forgetting who's close to whom.

# Section 77. Rasterization and Scanline Techniques 

### 761 Bresenham's Line Algorithm

The Bresenham's Line Algorithm is a foundational algorithm in computer graphics that draws a straight line between two points using only integer arithmetic.
It avoids floating-point operations, making it both fast and precise, perfect for raster displays, pixel art, and embedded systems.

Invented by Jack Bresenham in 1962 for early IBM plotters, it remains one of the most elegant examples of turning continuous geometry into discrete computation.

#### What Problem Are We Solving?

We want to draw a straight line from $(x_0, y_0)$ to $(x_1, y_1)$ on a pixel grid.
But computers can only light up discrete pixels, not continuous values.

A naïve approach would compute $y = m x + c$ and round each result,
but that uses slow floating-point arithmetic and accumulates rounding errors.

Bresenham's algorithm solves this by using incremental integer updates
and a decision variable to choose which pixel to light next.

#### How It Works (Plain Language)

Imagine walking from one end of the line to the other, pixel by pixel.
At each step, you decide:

> "Should I go straight east, or northeast?"

That decision depends on how far the true line is from the midpoint between these two candidate pixels.

Bresenham uses a decision parameter $d$ that tracks the difference between the ideal line and the rasterized path.

For a line with slope $0 \le m \le 1$, the algorithm works like this:

1. Start at $(x_0, y_0)$
2. Compute the deltas:
   $$
   \Delta x = x_1 - x_0, \quad \Delta y = y_1 - y_0
   $$
3. Initialize the decision parameter:
   $$
   d = 2\Delta y - \Delta x
   $$
4. For each $x$ from $x_0$ to $x_1$:

   * Plot $(x, y)$
   * If $d > 0$, increment $y$ and update
     $$
     d = d + 2(\Delta y - \Delta x)
     $$
   * Else, update
     $$
     d = d + 2\Delta y
     $$

This process traces the line using only additions and subtractions.

#### Example

Let's draw a line from $(2, 2)$ to $(8, 5)$.

$$
\Delta x = 6, \quad \Delta y = 3
$$
Initial $d = 2\Delta y - \Delta x = 0$.

| Step | (x, y) | d  | Action                     |
| ---- | ------ | -- | -------------------------- |
| 1    | (2, 2) | 0  | Plot                       |
| 2    | (3, 2) | +6 | $d>0$, increment y → (3,3) |
| 3    | (4, 3) | -6 | $d<0$, stay                |
| 4    | (5, 3) | +6 | increment y → (5,4)        |
| 5    | (6, 4) | -6 | stay                       |
| 6    | (7, 4) | +6 | increment y → (7,5)        |
| 7    | (8, 5) |,  | done                       |

Line drawn: (2,2), (3,3), (4,3), (5,4), (6,4), (7,5), (8,5).

#### Tiny Code (C Example)

```c
#include <stdio.h>
#include <stdlib.h>

void bresenham_line(int x0, int y0, int x1, int y1) {
    int dx = abs(x1 - x0);
    int dy = abs(y1 - y0);
    int sx = (x0 < x1) ? 1 : -1;
    int sy = (y0 < y1) ? 1 : -1;
    int err = dx - dy;

    while (1) {
        printf("(%d, %d)\n", x0, y0);
        if (x0 == x1 && y0 == y1) break;
        int e2 = 2 * err;
        if (e2 > -dy) { err -= dy; x0 += sx; }
        if (e2 < dx)  { err += dx; y0 += sy; }
    }
}
```

This version handles all slopes and directions symmetrically.

#### Why It Matters

Bresenham's algorithm is one of the earliest and most influential rasterization methods.
It's still used today in:

* 2D and 3D graphics renderers
* CAD software
* Printer drivers and plotters
* Microcontrollers and display systems
* Teaching integer arithmetic and geometry in computer science

It's not just an algorithm, it's a bridge between geometry and computation.

#### A Gentle Proof (Why It Works)

The true line equation is $y = m x + b$, where $m = \frac{\Delta y}{\Delta x}$.
The midpoint between two candidate pixels differs from the true line by an error $\varepsilon$.
Bresenham tracks a scaled version of this error as $d$, doubling it to avoid fractions:

$$
d = 2(\Delta y x - \Delta x y + C)
$$

When $d > 0$, the midpoint lies below the true line, so we step diagonally.
When $d < 0$, it lies above, so we step horizontally.
Because updates are constant-time integer additions, accuracy and efficiency are guaranteed.

#### Try It Yourself

1. Draw a line between $(0, 0)$ and $(10, 6)$ on grid paper.
2. Apply the update rules manually, you'll see the same pattern emerge.
3. Modify the algorithm for steep slopes ($m > 1$) by swapping roles of x and y.
4. Visualize how the decision variable controls vertical steps.

#### Test Cases

| Points      | Slope | Pixels Drawn      |
| ----------- | ----- | ----------------- |
| (0,0)-(5,2) | 0.4   | Gentle line       |
| (0,0)-(2,5) | >1    | Swap roles        |
| (2,2)-(8,5) | 0.5   | Classic test      |
| (5,5)-(0,0) | -1    | Reverse direction |

#### Complexity

| Operation | Time                     | Space  |
| --------- | ------------------------ | ------ |
| Draw Line | $O(\Delta x + \Delta y)$ | $O(1)$ |

The Bresenham Line Algorithm is the poet's ruler of the pixel world —
it draws with precision, one integer at a time,
turning algebra into art on the digital canvas.

### 762 Midpoint Circle Algorithm

The Midpoint Circle Algorithm is the circular counterpart of Bresenham's line algorithm.
It draws a perfect circle using only integer arithmetic, no trigonometry, no floating-point computation, by exploiting the circle's symmetry and a clever midpoint decision rule.

This algorithm is the heart of classic raster graphics, driving everything from retro games to low-level graphics libraries and display drivers.

#### What Problem Are We Solving?

We want to draw a circle centered at $(x_c, y_c)$ with radius $r$ on a discrete pixel grid.
The equation of the circle is:

$$
x^2 + y^2 = r^2
$$

Naïvely, we could compute each $y$ from $x$ using the formula $y = \sqrt{r^2 - x^2}$,
but that requires slow square roots and floating-point arithmetic.

The Midpoint Circle Algorithm eliminates these with an incremental, integer-based approach.

#### How It Works (Plain Language)

1. Start at the topmost point $(0, r)$.
2. Move outward along x and decide at each step whether to move south or south-east,
   depending on which pixel's center is closer to the true circle.
3. Use the circle's symmetry to draw eight points per iteration —
   one in each octant around the circle.

The algorithm relies on a decision variable $d$ that measures how far the midpoint lies from the circle boundary.

#### Step-by-Step Formulation

At each step, we evaluate the circle function:

$$
f(x, y) = x^2 + y^2 - r^2
$$

We want to know whether the midpoint between candidate pixels is inside or outside the circle.
The decision parameter is updated incrementally as we move.

1. Initialize:
   $$
   x = 0, \quad y = r
   $$
   $$
   d = 1 - r
   $$

2. Repeat until $x > y$:

   * Plot the eight symmetric points:
     $(\pm x + x_c, \pm y + y_c)$ and $(\pm y + x_c, \pm x + y_c)$
   * If $d < 0$, choose East (E) pixel and update
     $$
     d = d + 2x + 3
     $$
   * Else, choose South-East (SE) pixel and update
     $$
     d = d + 2(x - y) + 5, \quad y = y - 1
     $$
   * In both cases, increment $x = x + 1$

#### Example

Circle center $(0, 0)$, radius $r = 5$.

| Step | (x, y) | d  | Action       |
| ---- | ------ | -- | ------------ |
| 0    | (0, 5) | -4 | E → (1, 5)   |
| 1    | (1, 5) | -1 | E → (2, 5)   |
| 2    | (2, 5) | +4 | SE → (3, 4)  |
| 3    | (3, 4) | +1 | SE → (4, 3)  |
| 4    | (4, 3) | +7 | SE → (5, 2)  |
| 5    | (5, 2) |,  | Stop (x > y) |

Plotting the eight symmetric points for each iteration completes the circle.

#### Tiny Code (C Example)

```c
#include <stdio.h>

void midpoint_circle(int xc, int yc, int r) {
    int x = 0, y = r;
    int d = 1 - r;

    while (x <= y) {
        // 8 symmetric points
        printf("(%d,%d) (%d,%d) (%d,%d) (%d,%d)\n",
               xc + x, yc + y, xc - x, yc + y,
               xc + x, yc - y, xc - x, yc - y);
        printf("(%d,%d) (%d,%d) (%d,%d) (%d,%d)\n",
               xc + y, yc + x, xc - y, yc + x,
               xc + y, yc - x, xc - y, yc - x);

        if (d < 0) {
            d += 2 * x + 3;
        } else {
            d += 2 * (x - y) + 5;
            y--;
        }
        x++;
    }
}
```

#### Why It Matters

The Midpoint Circle Algorithm is used in:

* Low-level graphics libraries (e.g., SDL, OpenGL rasterizer base)
* Embedded systems and display firmware
* Digital art and games for drawing circles and arcs
* Geometric reasoning for symmetry and integer geometry examples

It forms a perfect pair with Bresenham's line algorithm, both based on discrete decision logic rather than continuous math.

#### A Gentle Proof (Why It Works)

The midpoint test evaluates whether the midpoint between two pixel candidates lies inside or outside the ideal circle:

If $f(x + 1, y - 0.5) < 0$, the midpoint is inside → choose E.
Otherwise, it's outside → choose SE.

By rearranging terms, the incremental update is derived:

$$
d_{k+1} =
\begin{cases}
d_k + 2x_k + 3, & \text{if } d_k < 0 \\
d_k + 2(x_k - y_k) + 5, & \text{if } d_k \ge 0
\end{cases}
$$


Since all terms are integers, the circle can be rasterized precisely with integer arithmetic.

#### Try It Yourself

1. Draw a circle centered at $(0,0)$ with $r=5$.
2. Compute $d$ step-by-step using the rules above.
3. Mark eight symmetric points at each iteration.
4. Compare to the mathematical circle, they align perfectly.

#### Test Cases

| Center   | Radius | Points Drawn | Symmetry |
| -------- | ------ | ------------ | -------- |
| (0, 0)   | 3      | 24           | Perfect  |
| (10, 10) | 5      | 40           | Perfect  |
| (0, 0)   | 10     | 80           | Perfect  |

#### Complexity

| Operation   | Time   | Space  |
| ----------- | ------ | ------ |
| Draw Circle | $O(r)$ | $O(1)$ |

The Midpoint Circle Algorithm is geometry's quiet craftsman —
it draws a perfect loop with nothing but integers and symmetry,
turning a pure equation into a dance of pixels on a square grid.

### 763 Scanline Fill

The Scanline Fill Algorithm is a classic polygon-filling technique in computer graphics.
It colors the interior of a polygon efficiently, one horizontal line (or *scanline*) at a time.
Rather than testing every pixel, it determines where each scanline enters and exits the polygon and fills only between those points.

This method forms the foundation of raster graphics, renderers, and vector-to-pixel conversions.

#### What Problem Are We Solving?

We need to fill the inside of a polygon, all pixels that lie within its boundary —
using an efficient, deterministic process that works on a discrete grid.

A brute-force approach would test every pixel to see if it's inside the polygon (using ray casting or winding rules),
but that's expensive.

The Scanline Fill Algorithm converts this into a row-by-row filling problem using intersection points.

#### How It Works (Plain Language)

1. Imagine horizontal lines sweeping from top to bottom across the polygon.
2. Each scanline may intersect the polygon's edges multiple times.
3. The rule:

   * Fill pixels between pairs of intersections (entering and exiting the polygon).

Thus, each scanline becomes a simple sequence of *on-off* regions: fill between every alternate pair of x-intersections.

#### Step-by-Step Procedure

1. Build an Edge Table (ET)

   * For every polygon edge, record:

     * Minimum y (start scanline)
     * Maximum y (end scanline)
     * x-coordinate of the lower endpoint
     * Inverse slope ($1/m$)
   * Store these edges sorted by their minimum y.

2. Initialize an Active Edge Table (AET), empty at the start.

3. For each scanline y:

   * Add edges from the ET whose minimum y equals the current scanline.
   * Remove edges from the AET whose maximum y equals the current scanline.
   * Sort the AET by current x.
   * Fill pixels between each pair of x-intersections.
   * For each edge in AET, update its x:
     $$
     x_{\text{new}} = x_{\text{old}} + \frac{1}{m}
     $$

4. Repeat until the AET is empty.

This procedure efficiently handles convex and concave polygons.

#### Example

Polygon: vertices $(2,2), (6,2), (4,6)$

| Edge        | y_min | y_max | x_at_y_min | 1/m  |
| ----------- | ----- | ----- | ---------- | ---- |
| (2,2)-(6,2) | 2     | 2     | 2          |,    |
| (6,2)-(4,6) | 2     | 6     | 6          | -0.5 |
| (4,6)-(2,2) | 2     | 6     | 2          | +0.5 |

Scanline progression:

| y | Active Edges           | x-intersections | Fill              |
| - | ---------------------- | --------------- | ----------------- |
| 2 |,                      |,               | Edge starts       |
| 3 | (2,6,-0.5), (2,6,+0.5) | x = 2.5, 5.5    | Fill (3, 2.5→5.5) |
| 4 | ...                    | x = 3, 5        | Fill (4, 3→5)     |
| 5 | ...                    | x = 3.5, 4.5    | Fill (5, 3.5→4.5) |
| 6 |,                      |,               | Done              |

#### Tiny Code (Python Example)

```python
def scanline_fill(polygon):
    # polygon = [(x0,y0), (x1,y1), ...]
    n = len(polygon)
    edges = []
    for i in range(n):
        x0, y0 = polygon[i]
        x1, y1 = polygon[(i + 1) % n]
        if y0 == y1:
            continue  # skip horizontal edges
        if y0 > y1:
            x0, y0, x1, y1 = x1, y1, x0, y0
        inv_slope = (x1 - x0) / (y1 - y0)
        edges.append([y0, y1, x0, inv_slope])

    edges.sort(key=lambda e: e[0])
    y = int(edges[0][0])
    active = []

    while active or edges:
        # Add new edges
        while edges and edges[0][0] == y:
            active.append(edges.pop(0))
        # Remove finished edges
        active = [e for e in active if e[1] > y]
        # Sort and find intersections
        x_list = [e[2] for e in active]
        x_list.sort()
        # Fill between pairs
        for i in range(0, len(x_list), 2):
            print(f"Fill line at y={y} from x={x_list[i]} to x={x_list[i+1]}")
        # Update x
        for e in active:
            e[2] += e[3]
        y += 1
```

#### Why It Matters

* Core of polygon rasterization in 2D rendering engines.
* Used in fill tools, graphics APIs, and hardware rasterizers.
* Handles concave and complex polygons efficiently.
* Demonstrates the power of incremental updates and scanline coherence in graphics.

It's the algorithm behind how your screen fills regions in vector graphics or how CAD software shades polygons.

#### A Gentle Proof (Why It Works)

A polygon alternates between being *inside* and *outside* at every edge crossing.
For each scanline, filling between every pair of intersections guarantees:

$$
\forall x \in [x_{2i}, x_{2i+1}], \ (x, y) \text{ is inside the polygon.}
$$

Since we only process active edges and update x incrementally,
each operation is $O(1)$ per edge per scanline, yielding total linear complexity in the number of edges times scanlines.

#### Try It Yourself

1. Draw a triangle on grid paper.
2. For each horizontal line, mark where it enters and exits the triangle.
3. Fill between those intersections.
4. Observe how the filled region exactly matches the polygon interior.

#### Test Cases

| Polygon         | Vertices | Filled Scanlines |
| --------------- | -------- | ---------------- |
| Triangle        | 3        | 4                |
| Rectangle       | 4        | 4                |
| Concave L-shape | 6        | 8                |
| Complex polygon | 8        | 10–12            |

#### Complexity

| Operation    | Time       | Space  |
| ------------ | ---------- | ------ |
| Fill Polygon | $O(n + H)$ | $O(n)$ |

where $H$ = number of scanlines in the bounding box.

The Scanline Fill Algorithm is like painting with a ruler —
it glides across the canvas line by line,
filling every space with calm precision until the whole shape glows solid.

### 764 Edge Table Fill

The Edge Table Fill Algorithm is a refined and efficient form of the scanline polygon fill.
It uses an explicit Edge Table (ET) and Active Edge Table (AET) to manage polygon boundaries, enabling fast and structured filling of even complex shapes.

This method is often implemented inside graphics hardware and rendering libraries because it minimizes redundant work while ensuring precise polygon filling.

#### What Problem Are We Solving?

When filling polygons using scanlines, we need to know exactly where each scanline enters and exits the polygon.
Instead of recomputing intersections every time, the Edge Table organizes edges so that updates are done incrementally as the scanline moves.

The Edge Table Fill Algorithm improves on basic scanline filling by storing precomputed edge data in buckets keyed by y-coordinates.

#### How It Works (Plain Language)

1. Build an Edge Table (ET), one bucket for each scanline $y$ where edges start.
2. Build an Active Edge Table (AET), dynamic list of edges that intersect the current scanline.
3. For each scanline $y$:

   * Add edges from the ET that start at $y$.
   * Remove edges that end at $y$.
   * Sort active edges by current x.
   * Fill pixels between pairs of x-values.
   * Update x for each edge incrementally using its slope.

#### Edge Table (ET) Structure

Each edge is stored with:

| Field | Meaning                                   |
| ----- | ----------------------------------------- |
| y_max | Scanline where the edge ends              |
| x     | x-coordinate at y_min                     |
| 1/m   | Inverse slope (increment for each y step) |

Edges are inserted into the ET bucket corresponding to their starting y_min.

#### Step-by-Step Example

Consider a polygon with vertices:
$(3,2), (6,5), (3,8), (1,5)$

Compute edges:

| Edge        | y_min | y_max | x | 1/m  |
| ----------- | ----- | ----- | - | ---- |
| (3,2)-(6,5) | 2     | 5     | 3 | +1   |
| (6,5)-(3,8) | 5     | 8     | 6 | -1   |
| (3,8)-(1,5) | 5     | 8     | 1 | +1   |
| (1,5)-(3,2) | 2     | 5     | 1 | 0.67 |

ET (grouped by y_min):

| y | Edges                     |
| - | ------------------------- |
| 2 | [(5, 3, 1), (5, 1, 0.67)] |
| 5 | [(8, 6, -1), (8, 1, +1)]  |

Then the scanline filling begins at y=2.

At each step:

* Add edges from ET[y] to AET.
* Sort AET by x.
* Fill between pairs.
* Update x by $x = x + 1/m$.

#### Tiny Code (Python Example)

```python
def edge_table_fill(polygon):
    ET = {}
    for i in range(len(polygon)):
        x0, y0 = polygon[i]
        x1, y1 = polygon[(i+1) % len(polygon)]
        if y0 == y1:
            continue
        if y0 > y1:
            x0, y0, x1, y1 = x1, y1, x0, y0
        inv_slope = (x1 - x0) / (y1 - y0)
        ET.setdefault(int(y0), []).append({
            'ymax': int(y1),
            'x': float(x0),
            'inv_slope': inv_slope
        })

    y = min(ET.keys())
    AET = []
    while AET or y in ET:
        if y in ET:
            AET.extend(ET[y])
        AET = [e for e in AET if e['ymax'] > y]
        AET.sort(key=lambda e: e['x'])
        for i in range(0, len(AET), 2):
            x1, x2 = AET[i]['x'], AET[i+1]['x']
            print(f"Fill line at y={y}: from x={x1:.2f} to x={x2:.2f}")
        for e in AET:
            e['x'] += e['inv_slope']
        y += 1
```

#### Why It Matters

The Edge Table Fill algorithm is central to polygon rasterization in:

* 2D graphics renderers (e.g., OpenGL's polygon pipeline)
* CAD systems for filled vector drawings
* Font rasterization and game graphics
* GPU scan converters

It reduces redundant computation, making it ideal for hardware or software rasterization loops.

#### A Gentle Proof (Why It Works)

For each scanline, the AET maintains exactly the set of edges intersecting that line.
Since each edge is linear, its intersection x increases by $\frac{1}{m}$ per scanline.
Thus the algorithm ensures consistency:

$$
x_{y+1} = x_y + \frac{1}{m}
$$

The alternating fill rule (inside–outside) guarantees that we fill every interior pixel once and only once.

#### Try It Yourself

1. Draw a pentagon on graph paper.
2. Create a table of edges with y_min, y_max, x, and 1/m.
3. For each scanline, mark entry and exit x-values and fill between them.
4. Compare your filled area to the exact polygon, it will match perfectly.

#### Test Cases

| Polygon   | Vertices | Type              | Filled Correctly          |
| --------- | -------- | ----------------- | ------------------------- |
| Triangle  | 3        | Convex            | Yes                       |
| Rectangle | 4        | Convex            | Yes                       |
| Concave   | 6        | Non-convex        | Yes                       |
| Star      | 10       | Self-intersecting | Partial (depends on rule) |

#### Complexity

| Operation    | Time       | Space  |
| ------------ | ---------- | ------ |
| Fill Polygon | $O(n + H)$ | $O(n)$ |

where $n$ is the number of edges, $H$ is the number of scanlines.

The Edge Table Fill Algorithm is the disciplined craftsman of polygon filling —
it organizes edges like tools in a box,
then works steadily scan by scan,
turning abstract vertices into solid, filled forms.

### 765 Z-Buffer Algorithm

The Z-Buffer Algorithm (or Depth Buffering) is the foundation of modern 3D rendering.
It determines which surface of overlapping 3D objects is visible at each pixel by comparing depth (z-values).

This algorithm is simple, robust, and widely implemented in hardware, every GPU you use today performs a version of it billions of times per second.

#### What Problem Are We Solving?

When projecting 3D objects onto a 2D screen, many surfaces overlap along the same pixel column.
We need to decide which one is closest to the camera, and hence visible.

Naïve solutions sort polygons globally, but that becomes difficult for intersecting or complex shapes.
The Z-Buffer Algorithm solves this by working *per pixel*, maintaining a running record of the closest object so far.

#### How It Works (Plain Language)

The idea is to maintain two buffers of the same size as the screen:

1. Frame Buffer (Color Buffer), stores final color of each pixel.
2. Depth Buffer (Z-Buffer), stores the z-coordinate (depth) of the nearest surface seen so far.

Algorithm steps:

1. Initialize the Z-Buffer with a large value (e.g., infinity).
2. For each polygon:

   * Compute its projection on the screen.
   * For each pixel inside the polygon:

     * Compute its depth z.
     * If $z < z_{\text{buffer}}[x, y]$,
       update both buffers:

       $$
       z_{\text{buffer}}[x, y] = z
       $$

       $$
       \text{frame}[x, y] = \text{polygon\_color}
       $$
3. After all polygons are processed, the frame buffer contains the visible image.

#### Step-by-Step Example

Suppose we render two triangles overlapping in screen space:
Triangle A (blue) and Triangle B (red).

For a given pixel $(x, y)$:

* Triangle A has depth $z_A = 0.45$
* Triangle B has depth $z_B = 0.3$

Since $z_B < z_A$, the red pixel from Triangle B is visible.

#### Mathematical Details

If the polygon is a plane given by

$$
ax + by + cz + d = 0,
$$

then we can compute $z$ for each pixel as

$$
z = -\frac{ax + by + d}{c}.
$$

During rasterization, $z$ can be incrementally interpolated across the polygon, just like color or texture coordinates.

#### Tiny Code (C Example)

```c
#include <stdio.h>
#include <float.h>

#define WIDTH 800
#define HEIGHT 600

typedef struct {
    float zbuffer[HEIGHT][WIDTH];
    unsigned int framebuffer[HEIGHT][WIDTH];
} Scene;

void clear(Scene* s) {
    for (int y = 0; y < HEIGHT; y++)
        for (int x = 0; x < WIDTH; x++) {
            s->zbuffer[y][x] = FLT_MAX;
            s->framebuffer[y][x] = 0; // background color
        }
}

void plot(Scene* s, int x, int y, float z, unsigned int color) {
    if (z < s->zbuffer[y][x]) {
        s->zbuffer[y][x] = z;
        s->framebuffer[y][x] = color;
    }
}
```

Each pixel compares its new depth with the stored one, a single `if` statement ensures correct visibility.

#### Why It Matters

* Used in all modern GPUs (OpenGL, Direct3D, Vulkan).
* Handles arbitrary overlapping geometry without sorting.
* Supports texture mapping, lighting, and transparency when combined with blending.
* Provides a per-pixel accuracy model of visibility, essential for photorealistic rendering.

#### A Gentle Proof (Why It Works)

For any pixel $(x, y)$, the visible surface is the one with the minimum z among all polygons projecting onto that pixel:

$$
z_{\text{visible}}(x, y) = \min_i z_i(x, y).
$$

By checking and updating this minimum incrementally as we draw, the Z-Buffer algorithm ensures that no farther surface overwrites a nearer one.

Because the depth buffer is initialized to $\infty$,
every first pixel write succeeds, and every later one is conditionally replaced only if closer.

#### Try It Yourself

1. Render two overlapping rectangles with different z-values.
2. Plot them in reverse order, notice that the front one still appears in front.
3. Visualize the z-buffer, closer surfaces have smaller values (brighter if visualized inversely).

#### Test Cases

| Scene                         | Expected Result              |
| ----------------------------- | ---------------------------- |
| Two overlapping triangles     | Foremost visible             |
| Cube rotating in space        | Faces correctly occluded     |
| Multiple intersecting objects | Correct visibility per pixel |

#### Complexity

| Operation  | Time            | Space           |
| ---------- | --------------- | --------------- |
| Per pixel  | $O(1)$          | $O(1)$          |
| Full frame | $O(W \times H)$ | $O(W \times H)$ |

The Z-Buffer Algorithm is the quiet guardian of every rendered image —
it watches every pixel's depth, ensuring that what you see
is exactly what lies closest in your virtual world.

### 766 Painter's Algorithm

The Painter's Algorithm is one of the earliest and simplest methods for hidden surface removal in 3D graphics.
It mimics how a painter works: by painting distant surfaces first, then closer ones over them, until the final visible image emerges.

Though it has been largely superseded by the Z-buffer in modern systems, it remains conceptually elegant and still useful in certain rendering pipelines and visualization tasks.

#### What Problem Are We Solving?

When multiple 3D polygons overlap in screen space, we need to determine which parts of each should be visible.
Instead of testing each pixel's depth (as in the Z-buffer), the Painter's Algorithm resolves this by drawing entire polygons in sorted order by depth.

The painter paints the farthest wall first, then the nearer ones, so that closer surfaces naturally overwrite those behind them.

#### How It Works (Plain Language)

1. Compute the average depth (z) for each polygon.
2. Sort all polygons in descending order of depth (farthest first).
3. Draw polygons one by one onto the image buffer, closer ones overwrite pixels of farther ones.

This works well when objects do not intersect and their depth ordering is consistent.

#### Step-by-Step Example

Imagine three rectangles stacked in depth:

| Polygon | Average z | Color |
| ------- | --------- | ----- |
| A       | 0.9       | Blue  |
| B       | 0.5       | Red   |
| C       | 0.2       | Green |

Sort by z: A → B → C

Paint them in order:

1. Draw A (blue, farthest)
2. Draw B (red, mid)
3. Draw C (green, nearest)

Result: The nearest (green) polygon hides parts of the others.

#### Handling Overlaps

If two polygons overlap in projection and cannot be easily depth-ordered (e.g., they intersect or cyclically overlap),
then recursive subdivision or hybrid approaches are needed:

1. Split polygons along their intersection lines.
2. Reorder the resulting fragments.
3. Draw them in correct order.

This ensures visibility correctness, at the cost of extra geometry computation.

#### Tiny Code (Python Example)

```python
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

polygons = [
    {'points': [(1,1),(5,1),(3,4)], 'z':0.8, 'color':'skyblue'},
    {'points': [(2,2),(6,2),(4,5)], 'z':0.5, 'color':'salmon'},
    {'points': [(3,3),(7,3),(5,6)], 'z':0.2, 'color':'limegreen'},
$$

# Sort by z (farthest first)
sorted_polygons = sorted(polygons, key=lambda p: p['z'], reverse=True)

fig, ax = plt.subplots()
for p in sorted_polygons:
    ax.add_patch(Polygon(p['points'], closed=True, facecolor=p['color'], edgecolor='black'))
ax.set_xlim(0,8)
ax.set_ylim(0,7)
ax.set_aspect('equal')
plt.show()
```

This draws the polygons back-to-front, exactly like a painter layering colors on canvas.

#### Why It Matters

* Intuitive, easy to implement.
* Works directly with polygon-level data, no need for per-pixel depth comparisons.
* Used in 2D rendering engines, vector graphics, and scene sorting.
* Forms the conceptual basis for more advanced visibility algorithms.

It's often used when:

* Rendering order can be precomputed (no intersection).
* You're simulating transparent surfaces or simple orthographic scenes.

#### A Gentle Proof (Why It Works)

Let polygons $P_1, P_2, ..., P_n$ have depths $z_1, z_2, ..., z_n$.
If $z_i > z_j$ for all pixels of $P_i$ behind $P_j$, then painting in descending $z$ order ensures that:

$$
\forall (x, y): \text{color}(x, y) = \text{color of nearest visible polygon at that pixel}.
$$

This holds because later polygons overwrite earlier ones in the frame buffer.

However, when polygons intersect, this depth order is not transitive and fails, hence the need for subdivision or alternative algorithms like the Z-buffer.

#### Try It Yourself

1. Draw three overlapping polygons on paper.
2. Assign z-values to each and order them back-to-front.
3. "Paint" them in that order, see how near ones cover the far ones.
4. Now create intersecting shapes, observe where ordering breaks.

#### Test Cases

| Scene                    | Works Correctly?          |
| ------------------------ | ------------------------- |
| Non-overlapping polygons | Yes                       |
| Nested polygons          | Yes                       |
| Intersecting polygons    | No (requires subdivision) |
| Transparent polygons     | Yes (with alpha blending) |

#### Complexity

| Operation     | Time          | Space                              |
| ------------- | ------------- | ---------------------------------- |
| Sort polygons | $O(n \log n)$ | $O(n)$                             |
| Draw polygons | $O(n)$        | $O(W \times H)$ (for frame buffer) |

The Painter's Algorithm captures a fundamental truth of graphics:
sometimes visibility is not about computation but about order —
the art of laying down layers until the scene emerges, one brushstroke at a time.

### 767 Gouraud Shading

Gouraud Shading is a classic method for producing smooth color transitions across a polygon surface.
Instead of assigning a single flat color to an entire face, it interpolates colors at the vertices and shades each pixel by gradually blending them.

It was one of the first algorithms to bring *smooth lighting* to computer graphics, fast, elegant, and easy to implement.

#### What Problem Are We Solving?

Flat shading gives each polygon a uniform color.
This looks artificial because the boundaries between adjacent polygons are sharply visible.

Gouraud Shading solves this by making the color vary smoothly across the surface, simulating how light reflects gradually on curved objects.

#### How It Works (Plain Language)

1. Compute vertex normals, the average of the normals of all faces sharing a vertex.

2. Compute vertex intensities using a lighting model (usually Lambertian reflection):

   $$
   I_v = k_d (L \cdot N_v) + I_{\text{ambient}}
   $$

   where

   * $L$ is the light direction
   * $N_v$ is the vertex normal
   * $k_d$ is diffuse reflectivity

3. For each polygon:

   * Interpolate the vertex intensities along each scanline.
   * Fill the interior pixels by interpolating intensity horizontally.

This gives smooth gradients across the surface with low computational cost.

#### Mathematical Form

Let vertices have intensities $I_1, I_2, I_3$.
For any interior point $(x, y)$, its intensity $I(x, y)$ is computed by barycentric interpolation:

$$
I(x, y) = \alpha I_1 + \beta I_2 + \gamma I_3
$$

where $\alpha + \beta + \gamma = 1$ and $\alpha, \beta, \gamma$ are barycentric coordinates of $(x, y)$ relative to the triangle.

#### Step-by-Step Example

Suppose a triangle has vertex intensities:

| Vertex | Coordinates | Intensity |
| ------ | ----------- | --------- |
| A      | (1, 1)      | 0.2       |
| B      | (5, 1)      | 0.8       |
| C      | (3, 4)      | 0.5       |

Then every point inside the triangle blends these values smoothly,
producing a gradient from dark at A to bright at B and medium at C.

#### Tiny Code (Python Example)

```python
import numpy as np
import matplotlib.pyplot as plt

def barycentric(x, y, x1, y1, x2, y2, x3, y3):
    det = (y2 - y3)*(x1 - x3) + (x3 - x2)*(y1 - y3)
    a = ((y2 - y3)*(x - x3) + (x3 - x2)*(y - y3)) / det
    b = ((y3 - y1)*(x - x3) + (x1 - x3)*(y - y3)) / det
    c = 1 - a - b
    return a, b, c

# triangle vertices and intensities
x1, y1, i1 = 1, 1, 0.2
x2, y2, i2 = 5, 1, 0.8
x3, y3, i3 = 3, 4, 0.5

img = np.zeros((6, 7))
for y in range(6):
    for x in range(7):
        a, b, c = barycentric(x, y, x1, y1, x2, y2, x3, y3)
        if a >= 0 and b >= 0 and c >= 0:
            img[y, x] = a*i1 + b*i2 + c*i3

plt.imshow(img, origin='lower', cmap='inferno')
plt.show()
```

This demonstrates how pixel colors can be smoothly blended based on vertex light intensities.

#### Why It Matters

* Introduced realistic shading into polygonal graphics.
* Forms the basis for hardware lighting in OpenGL and Direct3D.
* Efficient, all operations are linear interpolations, suitable for rasterization hardware.
* Used in both 3D modeling software and real-time engines before Phong shading became common.

#### A Gentle Proof (Why It Works)

The intensity on a plane varies linearly if lighting is computed at the vertices and interpolated.
For a triangle defined by vertices $A, B, C$, the light intensity at any interior point satisfies:

$$
\nabla^2 I(x, y) = 0
$$

since the interpolation is linear, and therefore continuous across edges.
Adjacent polygons sharing vertices have matching intensities at those vertices, giving a smooth overall appearance.

#### Try It Yourself

1. Create a triangle mesh (even a cube).
2. Compute vertex normals by averaging face normals.
3. Use the formula $I_v = k_d (L \cdot N_v)$ for each vertex.
4. Interpolate vertex intensities across each triangle and visualize the result.

Try rotating the light vector, you'll see how shading changes dynamically.

#### Test Cases

| Model          | Shading Type | Visual Result             |
| -------------- | ------------ | ------------------------- |
| Cube (flat)    | Flat         | Faceted look              |
| Cube (Gouraud) | Smooth       | Blended edges             |
| Sphere         | Gouraud      | Soft lighting             |
| Terrain        | Gouraud      | Natural gradient lighting |

#### Complexity

| Operation               | Time            | Space           |
| ----------------------- | --------------- | --------------- |
| Per vertex lighting     | $O(V)$          | $O(V)$          |
| Per pixel interpolation | $O(W \times H)$ | $O(W \times H)$ |

The Gouraud Shading algorithm was a key step in the evolution of realism in graphics —
a bridge between geometric form and visual smoothness,
where light glides softly across a surface instead of snapping from face to face.

### 768 Phong Shading

Phong Shading refines Gouraud Shading by interpolating normals instead of intensities, producing more accurate highlights and smooth lighting across curved surfaces.
It was a breakthrough for realism in computer graphics, capturing glossy reflections, specular highlights, and gentle light falloff with elegance.

#### What Problem Are We Solving?

Gouraud Shading interpolates colors between vertices, which can miss small, bright highlights (like a shiny spot on a sphere) if they occur between vertices.
Phong Shading fixes this by interpolating the surface normals per pixel, then recomputing lighting at every pixel.

This yields smoother, more physically accurate results, especially for curved and reflective surfaces.

#### How It Works (Plain Language)

1. Compute vertex normals as the average of the normals of all adjacent faces.
2. For each pixel inside a polygon:

   * Interpolate the normal vector $N(x, y)$ using barycentric interpolation.
   * Normalize it to unit length.
   * Apply the lighting equation at that pixel using $N(x, y)$.
3. Compute lighting (per pixel) using a standard illumination model such as Phong reflection:

   $$
   I(x, y) = k_a I_a + k_d (L \cdot N) I_l + k_s (R \cdot V)^n I_l
   $$

   where

   * $k_a, k_d, k_s$ are ambient, diffuse, and specular coefficients
   * $L$ is light direction
   * $N$ is surface normal at pixel
   * $R$ is reflection vector
   * $V$ is view direction
   * $n$ is shininess (specular exponent)

#### Step-by-Step Example

1. For each vertex of a triangle, store its normal vector $N_1, N_2, N_3$.
2. For each pixel inside the triangle:

   * Interpolate $N(x, y)$ using
     $$
     N(x, y) = \alpha N_1 + \beta N_2 + \gamma N_3
     $$
   * Normalize:
     $$
     N'(x, y) = \frac{N(x, y)}{|N(x, y)|}
     $$
   * Compute the illumination with the Phong model at that pixel.

The highlight intensity changes smoothly across the surface, producing a realistic reflection spot.

#### Tiny Code (Python Example)

```python
import numpy as np

def normalize(v):
    return v / np.linalg.norm(v)

def phong_shading(N, L, V, ka=0.1, kd=0.7, ks=0.8, n=10):
    N = normalize(N)
    L = normalize(L)
    V = normalize(V)
    R = 2 * np.dot(N, L) * N - L
    I = ka + kd * max(np.dot(N, L), 0) + ks * (max(np.dot(R, V), 0)  n)
    return np.clip(I, 0, 1)
```

At each pixel, interpolate `N`, then call `phong_shading(N, L, V)` to compute its color intensity.

#### Why It Matters

* Produces visually smooth shading and accurate specular highlights.
* Became the foundation for per-pixel lighting in modern graphics hardware.
* Accurately models curved surfaces without increasing polygon count.
* Ideal for glossy, metallic, or reflective materials.

#### A Gentle Proof (Why It Works)

Lighting is a nonlinear function of surface orientation:
the specular term $(R \cdot V)^n$ depends strongly on the local angle.
By interpolating normals, Phong Shading preserves this angular variation within each polygon.

Mathematically, Gouraud shading computes:

$$
I(x, y) = \alpha I_1 + \beta I_2 + \gamma I_3,
$$

whereas Phong computes:

$$
I(x, y) = f(\alpha N_1 + \beta N_2 + \gamma N_3),
$$

where $f(N)$ is the lighting function.
Since lighting is nonlinear in $N$, interpolating normals gives a more faithful approximation.

#### Try It Yourself

1. Render a sphere using flat shading, Gouraud shading, and Phong shading, compare results.
2. Place a single light source to one side, only Phong will capture the circular specular highlight.
3. Experiment with $n$ (shininess):

   * Low $n$ → matte surface.
   * High $n$ → shiny reflection.

#### Test Cases

| Model    | Shading Type | Result                           |
| -------- | ------------ | -------------------------------- |
| Cube     | Flat         | Faceted faces                    |
| Sphere   | Gouraud      | Smooth, missing highlights       |
| Sphere   | Phong        | Smooth with bright specular spot |
| Car body | Phong        | Realistic metal reflection       |

#### Complexity

| Operation            | Time            | Space           |
| -------------------- | --------------- | --------------- |
| Per-pixel lighting   | $O(W \times H)$ | $O(W \times H)$ |
| Normal interpolation | $O(W \times H)$ | $O(1)$          |

Phong Shading was the leap from *smooth color* to *smooth light*.
By bringing per-pixel illumination, it bridged geometry and optics —
making surfaces gleam, curves flow, and reflections shimmer like the real world.

### 769 Anti-Aliasing (Supersampling)

Anti-Aliasing smooths the jagged edges that appear when we draw diagonal or curved lines on a pixel grid.
The most common approach, Supersampling Anti-Aliasing (SSAA), works by rendering the scene at a higher resolution and averaging neighboring pixels to produce smoother edges.

It's a cornerstone of high-quality graphics, turning harsh stair-steps into soft, continuous shapes.

#### What Problem Are We Solving?

Digital images are made of square pixels, but most shapes in the real world aren't.
When we render a diagonal line or curve, pixelation creates visible aliasing, those "staircase" edges that look rough or flickery when moving.

Aliasing arises from undersampling, not enough pixel samples to represent fine details.
Anti-aliasing fixes this by increasing sampling density or blending between regions.

#### How It Works (Plain Language)

Supersampling takes multiple color samples for each pixel and averages them:

1. For each pixel, divide it into $k \times k$ subpixels.
2. Compute the color for each subpixel using the scene geometry and shading.
3. Average all subpixel colors to produce the final pixel color.

This way, the pixel color reflects partial coverage, how much of the pixel is covered by the object versus the background.

#### Example

Imagine a black line crossing a white background diagonally.
If a pixel is half-covered by the line, it will appear gray after supersampling,
because the average of white (background) and black (line) subpixels is gray.

So instead of harsh transitions, you get smooth gradients at edges.

#### Mathematical Form

If each pixel is divided into $m$ subpixels, the final color is:

$$
C_{\text{pixel}} = \frac{1}{m} \sum_{i=1}^{m} C_i
$$

where $C_i$ are the colors of each subpixel sample.

The higher $m$, the smoother the image, at the cost of more computation.

#### Step-by-Step Algorithm

1. Choose supersampling factor $s$ (e.g., 2×2, 4×4, 8×8).

2. For each pixel $(x, y)$:

   * For each subpixel $(i, j)$:
     $$
     x' = x + \frac{i + 0.5}{s}, \quad y' = y + \frac{j + 0.5}{s}
     $$

     * Compute color $C_{ij}$ at $(x', y')$.
   * Average:
     $$
     C(x, y) = \frac{1}{s^2} \sum_{i=0}^{s-1}\sum_{j=0}^{s-1} C_{ij}
     $$

3. Store $C(x, y)$ in the final frame buffer.

#### Tiny Code (Python Pseudocode)

```python
import numpy as np

def supersample(render_func, width, height, s=4):
    image = np.zeros((height, width, 3))
    for y in range(height):
        for x in range(width):
            color_sum = np.zeros(3)
            for i in range(s):
                for j in range(s):
                    x_sub = x + (i + 0.5) / s
                    y_sub = y + (j + 0.5) / s
                    color_sum += render_func(x_sub, y_sub)
            image[y, x] = color_sum / (s * s)
    return image
```

Here `render_func` computes the color of a subpixel, the heart of the renderer.

#### Why It Matters

* Reduces jagged edges (spatial aliasing).
* Improves motion smoothness when objects move (temporal aliasing).
* Enhances overall image realism and visual comfort.
* Still forms the conceptual foundation of modern techniques like MSAA, FXAA, and TAA.

#### A Gentle Proof (Why It Works)

Aliasing arises when we sample a continuous signal (the image) below the Nyquist rate —
high-frequency details "fold" into visible artifacts.

Supersampling increases the effective sampling rate,
and averaging acts as a low-pass filter, removing frequencies above the pixel grid's limit.

Mathematically, if $I(x, y)$ is the true image intensity,
the rendered pixel value becomes:

$$
I_{\text{pixel}} = \frac{1}{A} \iint_{A} I(x, y) , dx, dy
$$

which is the *area average* of the continuous image over the pixel region —
a physically accurate model of how a real display emits light.

#### Try It Yourself

1. Render a black diagonal line on a white 100×100 grid.
2. Without anti-aliasing, observe the jagged edge.
3. Apply 4× supersampling (2×2 per pixel).
4. Compare, edges will appear smoother and more natural.

#### Test Cases

| Image          | Sampling | Result              |
| -------------- | -------- | ------------------- |
| Diagonal line  | 1×1      | Jagged edges        |
| Diagonal line  | 2×2      | Noticeably smoother |
| Circle outline | 4×4      | Smooth curvature    |
| Text rendering | 8×8      | Crisp and readable  |

#### Complexity

| Operation                | Time            | Space           |
| ------------------------ | --------------- | --------------- |
| Supersampling factor $s$ | $O(s^2)$        | $O(W \times H)$ |
| Filtering                | $O(W \times H)$ | $O(1)$          |

Supersampling Anti-Aliasing softens the hard geometry of pixels into something the eye perceives as continuous.
It's how the digital canvas learns to whisper curves,
not just shout squares.

### 770 Scanline Polygon Clipping

Scanline Polygon Clipping is an efficient technique for trimming polygons to a given window or viewport using a horizontal sweep (scanline) approach.
It's used in 2D rendering pipelines to clip polygons before rasterization, keeping only the visible portion that falls inside the display region.

This algorithm combines geometric precision with raster efficiency, operating line-by-line rather than edge-by-edge.

#### What Problem Are We Solving?

When drawing polygons on the screen, only part of them may lie within the viewing window.
We must clip (cut) the polygons so that pixels outside the window are not drawn.

Classical polygon clipping algorithms like Sutherland–Hodgman work edge-by-edge.
Scanline Polygon Clipping instead operates per row (scanline), which matches how rasterization works —
making it faster and easier to integrate with rendering pipelines.

#### How It Works (Plain Language)

1. Represent the clipping region (usually a rectangle) and the polygon to be drawn.
2. Sweep a horizontal scanline from top to bottom.
3. For each scanline:

   * Find all intersections of the polygon edges with this scanline.
   * Sort the intersection points by x-coordinate.
   * Fill pixels between each *pair* of intersections that lie inside the clipping region.
4. Continue for all scanlines within the window bounds.

This way, the algorithm naturally clips the polygon —
since only intersections within the viewport are considered.

#### Example

Consider a triangle overlapping the edges of a 10×10 window.
At scanline $y = 5$, it may intersect polygon edges at $x = 3$ and $x = 7$.
Pixels $(4, 5)$ through $(6, 5)$ are filled; all others ignored.

At the next scanline $y = 6$, the intersections might shift to $x = 4$ and $x = 6$,
automatically forming the clipped interior.

#### Mathematical Form

For each polygon edge connecting $(x_1, y_1)$ and $(x_2, y_2)$,
find intersection with scanline $y = y_s$ using linear interpolation:

$$
x = x_1 + (y_s - y_1) \frac{(x_2 - x_1)}{(y_2 - y_1)}
$$

Include only intersections where $y_s$ lies within the edge's vertical span.

After sorting intersections $(x_1', x_2', x_3', x_4', ...)$,
fill between pairs $(x_1', x_2'), (x_3', x_4'), ...$ —
each pair represents an inside segment of the polygon.

#### Tiny Code (Simplified C Example)

```c
typedef struct { float x1, y1, x2, y2; } Edge;

void scanline_clip(Edge *edges, int n, int ymin, int ymax, int width) {
    for (int y = ymin; y <= ymax; y++) {
        float inter[100]; int k = 0;
        for (int i = 0; i < n; i++) {
            float y1 = edges[i].y1, y2 = edges[i].y2;
            if ((y >= y1 && y < y2) || (y >= y2 && y < y1)) {
                float x = edges[i].x1 + (y - y1) * (edges[i].x2 - edges[i].x1) / (y2 - y1);
                inter[k++] = x;
            }
        }
        // sort intersections
        for (int i = 0; i < k - 1; i++)
            for (int j = i + 1; j < k; j++)
                if (inter[i] > inter[j]) { float t = inter[i]; inter[i] = inter[j]; inter[j] = t; }

        // fill between pairs
        for (int i = 0; i < k; i += 2)
            for (int x = (int)inter[i]; x < (int)inter[i+1]; x++)
                if (x >= 0 && x < width) plot_pixel(x, y);
    }
}
```

This example clips and fills polygons scanline by scanline.

#### Why It Matters

* Perfectly integrates with rasterization, same scanline order.
* Avoids complex polygon clipping math.
* Works efficiently on hardware pipelines and software renderers alike.
* Still used in embedded systems, 2D games, and vector graphics engines.

#### A Gentle Proof (Why It Works)

The polygon boundary alternates between "entering" and "exiting" the filled region as you move horizontally across a scanline.
Thus, intersections always occur in even pairs,
and filling between them reproduces exactly the polygon's interior.

If clipping limits are $x_{\min}$ and $x_{\max}$,
the algorithm only fills within these bounds,
so the output region is effectively:

$$
\text{clipped polygon} = P \cap [x_{\min}, x_{\max}] \times [y_{\min}, y_{\max}]
$$

ensuring precise clipping without geometric recomputation.

#### Try It Yourself

1. Draw a polygon partly outside a rectangular window.
2. Move a horizontal line from top to bottom, mark intersection points each step.
3. Connect them pairwise, the shaded region is your clipped polygon.
4. Observe how the clipping region trims edges automatically.

#### Test Cases

| Polygon                    | Window | Result                         |
| -------------------------- | ------ | ------------------------------ |
| Triangle inside            | 10×10  | No change                      |
| Square crossing edge       | 10×10  | Trimmed at border              |
| Star partially outside     | 10×10  | Only visible interior rendered |
| Polygon completely outside | 10×10  | No output                      |

#### Complexity

| Operation    | Time            | Space  |
| ------------ | --------------- | ------ |
| Per scanline | $O(E)$          | $O(E)$ |
| Overall      | $O(E \times H)$ | $O(E)$ |

where $E$ is the number of edges and $H$ is the height (number of scanlines).

Scanline Polygon Clipping brings geometry down to the level of the raster itself —
a steady sweep line that reveals only what truly belongs on screen,
turning polygons into visible art one row at a time.

# Section 78. Computer Vision 

### 771 Canny Edge Detector

The Canny Edge Detector is one of the most influential algorithms in computer vision for detecting edges with precision and stability.
It combines gradient analysis, noise reduction, and non-maximum suppression to extract clear, single-pixel-wide edges from complex images.

Developed by John F. Canny in 1986, it remains a gold standard for edge detection today.

#### What Problem Are We Solving?

Edges mark boundaries between objects or regions with distinct intensity changes.
Detecting them is crucial for tasks like object recognition, segmentation, and shape analysis.

Naïve edge detection using gradients or Sobel filters often produces noisy, thick, or broken edges.
Canny's method provides:

* Low error (true edges detected)
* Good localization (edges precisely positioned)
* Minimal response (each edge detected once)

#### How It Works (Plain Language)

The Canny algorithm unfolds in five conceptual steps:

1. Noise Reduction
   Smooth the image using a Gaussian filter to reduce high-frequency noise:
   $$
   I_s = I * G_{\sigma}
   $$
   where $G_{\sigma}$ is a Gaussian kernel with standard deviation $\sigma$.

2. Gradient Computation
   Compute intensity gradients using partial derivatives:
   $$
   G_x = \frac{\partial I_s}{\partial x}, \quad G_y = \frac{\partial I_s}{\partial y}
   $$
   Then find the gradient magnitude and direction:
   $$
   M(x, y) = \sqrt{G_x^2 + G_y^2}, \quad \theta(x, y) = \arctan\left(\frac{G_y}{G_x}\right)
   $$

3. Non-Maximum Suppression
   Thin the edges by keeping only local maxima in the gradient direction.
   For each pixel, compare $M(x, y)$ to neighbors along $\theta(x, y)$, keep it only if it's larger.

4. Double Thresholding
   Use two thresholds $T_{\text{high}}$ and $T_{\text{low}}$ to classify pixels:

   * $M > T_{\text{high}}$: strong edge
   * $T_{\text{low}} < M \leq T_{\text{high}}$: weak edge
   * $M \leq T_{\text{low}}$: non-edge

5. Edge Tracking by Hysteresis
   Weak edges connected to strong edges are kept; others are discarded.
   This ensures continuity of real edges while filtering noise.

#### Step-by-Step Example

For a grayscale image:

1. Smooth with a $5 \times 5$ Gaussian filter ($\sigma = 1.0$).
2. Compute $G_x$ and $G_y$ using Sobel operators.
3. Compute gradient magnitude $M$.
4. Suppress non-maxima, keeping only local peaks.
5. Apply thresholds (e.g., $T_{\text{low}} = 0.1$, $T_{\text{high}} = 0.3$).
6. Link weak edges to strong ones using connectivity.

The final result: thin, continuous contours outlining real structures.

#### Tiny Code (Python Example with NumPy)

```python
import cv2
import numpy as np

# Load grayscale image
img = cv2.imread("input.jpg", cv2.IMREAD_GRAYSCALE)

# Apply Gaussian blur
blur = cv2.GaussianBlur(img, (5, 5), 1.0)

# Compute gradients
Gx = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
Gy = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)
mag = np.sqrt(Gx2 + Gy2)
angle = np.arctan2(Gy, Gx)

# Use OpenCV's hysteresis thresholding for simplicity
edges = cv2.Canny(img, 100, 200)

cv2.imwrite("edges.jpg", edges)
```

This captures all five stages compactly using OpenCV's built-in pipeline.

#### Why It Matters

* Detects edges reliably even in noisy conditions.
* Provides subpixel precision when implemented with interpolation.
* Balances sensitivity and noise control using Gaussian smoothing and hysteresis thresholds.
* Forms the foundation for higher-level vision tasks like contour tracing, feature extraction, and segmentation.

#### A Gentle Proof (Why It Works)

Canny formulated edge detection as an optimization problem,
seeking an operator that maximizes signal-to-noise ratio while maintaining localization and minimal response.

By modeling edges as intensity ramps corrupted by Gaussian noise, he derived that the optimal edge detector is based on the first derivative of a Gaussian:

$$
h(x) = -x e^{-\frac{x^2}{2\sigma^2}}
$$

Hence the algorithm's design naturally balances smoothing (to suppress noise) and differentiation (to detect edges).

#### Try It Yourself

1. Apply Canny to a photo at different $\sigma$ values, observe how larger $\sigma$ blurs small details.
2. Experiment with thresholds $(T_{\text{low}}, T_{\text{high}})$.

   * Too low: noise appears as edges.
   * Too high: real edges disappear.
3. Compare Canny's results to simple Sobel or Prewitt filters.

#### Test Cases

| Image           | $\sigma$ | Thresholds | Result                    |
| --------------- | -------- | ---------- | ------------------------- |
| Simple shapes   | 1.0      | (50, 150)  | Crisp boundaries          |
| Noisy texture   | 2.0      | (80, 200)  | Clean edges               |
| Face photo      | 1.2      | (70, 180)  | Facial contours preserved |
| Satellite image | 3.0      | (100, 250) | Large-scale outlines      |

#### Complexity

| Operation            | Time            | Space           |
| -------------------- | --------------- | --------------- |
| Gradient computation | $O(W \times H)$ | $O(W \times H)$ |
| Non-max suppression  | $O(W \times H)$ | $O(1)$          |
| Hysteresis tracking  | $O(W \times H)$ | $O(W \times H)$ |

The Canny Edge Detector transformed how computers perceive structure in images —
a union of calculus, probability, and geometry that finds beauty in the boundaries of things.

### 772 Sobel Operator

The Sobel Operator is a simple and powerful tool for edge detection and gradient estimation in images.
It measures how brightness changes in both horizontal and vertical directions, producing an image where edges appear as regions of high intensity.

Although conceptually simple, it remains a cornerstone in computer vision and digital image processing.

#### What Problem Are We Solving?

Edges are where the intensity in an image changes sharply, often indicating object boundaries, textures, or features.
To find them, we need a way to estimate the gradient (rate of change) of the image intensity.

The Sobel Operator provides a discrete approximation of this derivative using convolution masks, while also applying slight smoothing to reduce noise.

#### How It Works (Plain Language)

The Sobel method uses two $3 \times 3$ convolution kernels to estimate gradients:

* Horizontal gradient ($G_x$):
  $$
  G_x =
  \begin{bmatrix}
  -1 & 0 & +1 \
  -2 & 0 & +2 \
  -1 & 0 & +1
  \end{bmatrix}
  $$

* Vertical gradient ($G_y$):
  $$
  G_y =
  \begin{bmatrix}
  +1 & +2 & +1 \
  0 & 0 & 0 \
  -1 & -2 & -1
  \end{bmatrix}
  $$

You convolve these kernels with the image to get the rate of intensity change in x and y directions.

#### Computing the Gradient

1. For each pixel $(x, y)$:
   $$
   G_x(x, y) = (I * K_x)(x, y), \quad G_y(x, y) = (I * K_y)(x, y)
   $$
2. Compute gradient magnitude:
   $$
   M(x, y) = \sqrt{G_x^2 + G_y^2}
   $$
3. Compute gradient direction:
   $$
   \theta(x, y) = \arctan\left(\frac{G_y}{G_x}\right)
   $$

High magnitude values correspond to strong edges.

#### Step-by-Step Example

For a small 3×3 patch of an image:

| Pixel | Value |     |
| ----- | ----- | --- |
| 10    | 10    | 10  |
| 10    | 50    | 80  |
| 10    | 80    | 100 |

Convolving with $G_x$ and $G_y$ gives:

* $G_x = (+1)(80 - 10) + (+2)(100 - 10) = 320$
* $G_y = (+1)(10 - 10) + (+2)(10 - 80) = -140$

So:
$$
M = \sqrt{320^2 + (-140)^2} \approx 349.3
$$

A strong edge is detected there.

#### Tiny Code (Python Example)

```python
import cv2
import numpy as np

img = cv2.imread("input.jpg", cv2.IMREAD_GRAYSCALE)

# Compute Sobel gradients
Gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
Gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

# Magnitude and angle
magnitude = np.sqrt(Gx2 + Gy2)
angle = np.arctan2(Gy, Gx)

cv2.imwrite("sobel_edges.jpg", np.uint8(np.clip(magnitude, 0, 255)))
```

#### Why It Matters

* Fast and easy to implement.
* Produces good edge maps for well-lit, low-noise images.
* Forms a core part of many larger algorithms (e.g., Canny Edge Detector).
* Ideal for feature extraction in robotics, medical imaging, and computer vision preprocessing.

#### A Gentle Proof (Why It Works)

The Sobel kernels are a discrete approximation of partial derivatives with a built-in smoothing effect.
For an image intensity function $I(x, y)$, the continuous derivatives are:

$$
\frac{\partial I}{\partial x} \approx I(x + 1, y) - I(x - 1, y)
$$

The central difference scheme is combined with vertical (or horizontal) weights `[1, 2, 1]`
to suppress noise and emphasize central pixels, making Sobel robust to small fluctuations.

#### Try It Yourself

1. Apply Sobel filters separately for $x$ and $y$.
2. Visualize $G_x$ (vertical edges) and $G_y$ (horizontal edges).
3. Combine magnitudes to see full edge strength.
4. Experiment with different image types, portraits, text, natural scenes.

#### Test Cases

| Image                    | Kernel Size | Output Characteristics     |
| ------------------------ | ----------- | -------------------------- |
| Text on white background | 3×3         | Clear letter edges         |
| Landscape                | 3×3         | Good object outlines       |
| Noisy photo              | 5×5         | Slight blurring but stable |
| Medical X-ray            | 3×3         | Highlights bone contours   |

#### Complexity

| Operation             | Time            | Space           |
| --------------------- | --------------- | --------------- |
| Convolution           | $O(W \times H)$ | $O(W \times H)$ |
| Magnitude + Direction | $O(W \times H)$ | $O(1)$          |

The Sobel Operator is simplicity at its sharpest —
a small 3×3 window that reveals the geometry of light,
turning subtle intensity changes into the edges that define form and structure.

### 773 Hough Transform (Lines)

The Hough Transform is a geometric algorithm that detects lines, circles, and other parametric shapes in images.
It converts edge points in image space into a parameter space, where patterns become peaks, making it robust against noise and missing data.

For lines, it's one of the most elegant ways to find all straight lines in an image, even when the edges are broken or scattered.

#### What Problem Are We Solving?

After edge detection (like Canny or Sobel), we have a set of pixels likely belonging to edges.
But we still need to find continuous geometric structures, especially lines that connect these points.

A naive method would try to fit lines directly in the image, but that's unstable when edges are incomplete.
The Hough Transform solves this by accumulating votes in a transformed space where all possible lines can be represented.

#### How It Works (Plain Language)

A line in Cartesian coordinates can be written as
$$
y = mx + b,
$$
but this form fails for vertical lines ($m \to \infty$).
So we use the polar form instead:

$$
\rho = x \cos \theta + y \sin \theta
$$

where

* $\rho$ is the perpendicular distance from the origin to the line,
* $\theta$ is the angle between the x-axis and the line's normal.

Each edge pixel $(x, y)$ represents all possible lines passing through it.
In parameter space $(\rho, \theta)$, that pixel corresponds to a sinusoidal curve.

Where multiple curves intersect → that point $(\rho, \theta)$ represents a line supported by many edge pixels.

#### Step-by-Step Algorithm

1. Initialize an accumulator array $A[\rho, \theta]$ (all zeros).
2. For each edge pixel $(x, y)$:

   * For each $\theta$ from $0$ to $180^\circ$:
     $$
     \rho = x \cos \theta + y \sin \theta
     $$
     Increment accumulator cell $A[\rho, \theta]$.
3. Find all accumulator peaks where votes exceed a threshold.
   Each peak $(\rho_i, \theta_i)$ corresponds to a detected line.
4. Convert these back into image space for visualization.

#### Example

Suppose three points all lie roughly along a diagonal edge.
Each of their sinusoidal curves in $(\rho, \theta)$ space intersect near $(\rho = 50, \theta = 45^\circ)$ —
so a strong vote appears there.

That point corresponds to the line
$$
x \cos 45^\circ + y \sin 45^\circ = 50,
$$
or equivalently, $y = -x + c$ in image space.

#### Tiny Code (Python Example using OpenCV)

```python
import cv2
import numpy as np

# Read and preprocess image
img = cv2.imread("edges.jpg", cv2.IMREAD_GRAYSCALE)

# Use Canny to get edge map
edges = cv2.Canny(img, 100, 200)

# Apply Hough Transform
lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)

# Draw detected lines
output = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
for rho, theta in lines[:, 0]:
    a, b = np.cos(theta), np.sin(theta)
    x0, y0 = a * rho, b * rho
    x1, y1 = int(x0 + 1000 * (-b)), int(y0 + 1000 * (a))
    x2, y2 = int(x0 - 1000 * (-b)), int(y0 - 1000 * (a))
    cv2.line(output, (x1, y1), (x2, y2), (0, 0, 255), 2)

cv2.imwrite("hough_lines.jpg", output)
```

#### Why It Matters

* Detects lines, boundaries, and axes even with gaps or noise.
* Tolerant to missing pixels, lines emerge from consensus, not continuity.
* Foundation for many tasks:

  * Lane detection in self-driving cars
  * Document alignment
  * Shape recognition
  * Industrial inspection

#### A Gentle Proof (Why It Works)

Each edge pixel contributes evidence for all lines passing through it.
If $N$ points lie approximately on the same line, their sinusoidal curves intersect in $(\rho, \theta)$ space,
producing a large vote count $A[\rho, \theta] = N$.

This intersection property effectively turns collinearity in image space
into concentration in parameter space, allowing detection via simple thresholding.

Formally:
$$
A(\rho, \theta) = \sum_{x, y \in \text{edges}} \delta(\rho - x \cos \theta - y \sin \theta)
$$

Peaks in $A$ correspond to dominant linear structures.

#### Try It Yourself

1. Run Canny edge detection on a simple shape (e.g., a rectangle).
2. Apply Hough Transform and visualize accumulator peaks.
3. Change the vote threshold to see how smaller or weaker lines appear/disappear.
4. Experiment with different $\Delta \theta$ resolutions for accuracy vs. speed.

#### Test Cases

| Image            | Expected Lines | Notes                                |
| ---------------- | -------------- | ------------------------------------ |
| Square shape     | 4              | Detects all edges                    |
| Road photo       | 2–3            | Lane lines found                     |
| Grid pattern     | Many           | Regular peaks in accumulator         |
| Noisy background | Few            | Only strong consistent edges survive |

#### Complexity

| Operation         | Time                 | Space                |
| ----------------- | -------------------- | -------------------- |
| Vote accumulation | $O(N \cdot K)$       | $O(R \times \Theta)$ |
| Peak detection    | $O(R \times \Theta)$ | $O(1)$               |

where

* $N$ = number of edge pixels
* $K$ = number of $\theta$ values sampled

The Hough Transform turns geometry into statistics —
every edge pixel casts its vote, and when enough pixels agree,
a line quietly emerges from the noise, crisp and certain.

### 774 Hough Transform (Circles)

The Hough Transform for Circles extends the line-based version of the transform to detect circular shapes.
Instead of finding straight-line alignments, it finds sets of points that lie on the perimeter of possible circles.
It's especially useful when circles are partially visible or obscured by noise.

#### What Problem Are We Solving?

Edges give us candidate pixels for boundaries, but we often need to detect specific geometric shapes, like circles, ellipses, or arcs.
Circle detection is vital in tasks such as:

* Detecting coins, pupils, or holes in objects
* Recognizing road signs and circular logos
* Locating circular patterns in microscopy or astronomy

A circle is defined by its center $(a, b)$ and radius $r$.
The goal is to find all $(a, b, r)$ that fit enough edge points.

#### How It Works (Plain Language)

A circle can be expressed as:
$$
(x - a)^2 + (y - b)^2 = r^2
$$

For each edge pixel $(x, y)$, every possible circle that passes through it satisfies this equation.
Each $(x, y)$ thus votes for all possible centers $(a, b)$ for a given radius $r$.

Where many votes accumulate → that's the circle's center.

When radius is unknown, the algorithm searches in 3D parameter space $(a, b, r)$:

* $a$: x-coordinate of center
* $b$: y-coordinate of center
* $r$: radius

#### Step-by-Step Algorithm

1. Edge Detection
   Use Canny or Sobel to get an edge map.

2. Initialize Accumulator
   Create a 3D array $A[a, b, r]$ filled with zeros.

3. Voting Process
   For each edge pixel $(x, y)$ and each candidate radius $r$:

   * Compute possible centers:
     $$
     a = x - r \cos \theta, \quad b = y - r \sin \theta
     $$
     for $\theta$ in $[0, 2\pi]$.
   * Increment accumulator cell $A[a, b, r]$.

4. Find Peaks
   Local maxima in $A[a, b, r]$ indicate detected circles.

5. Output
   Convert back to image space, draw circles with detected $(a, b, r)$.

#### Example

Imagine a 100×100 image with an edge circle of radius 30 centered at (50, 50).
Each edge point votes for all possible $(a, b)$ centers corresponding to that radius.
At $(50, 50)$, the votes align and produce a strong peak, revealing the circle's center.

#### Tiny Code (Python Example using OpenCV)

```python
import cv2
import numpy as np

# Read grayscale image
img = cv2.imread("input.jpg", cv2.IMREAD_GRAYSCALE)
edges = cv2.Canny(img, 100, 200)

# Detect circles
circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, dp=1.2,
                           minDist=20, param1=100, param2=30,
                           minRadius=10, maxRadius=80)

output = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

if circles is not None:
    circles = np.uint16(np.around(circles))
    for (x, y, r) in circles[0, :]:
        cv2.circle(output, (x, y), r, (0, 255, 0), 2)
        cv2.circle(output, (x, y), 2, (0, 0, 255), 3)

cv2.imwrite("hough_circles.jpg", output)
```

#### Why It Matters

* Detects circular objects even when partially visible.
* Robust to noise and gaps in edges.
* Handles varying radius ranges efficiently with optimized implementations (e.g., OpenCV's `HOUGH_GRADIENT`).
* Useful across fields, from robotics to biology to astronomy.

#### A Gentle Proof (Why It Works)

For every point $(x, y)$, the circle equation
$$
(x - a)^2 + (y - b)^2 = r^2
$$
describes a locus of possible centers $(a, b)$.

By accumulating votes from many points, true circle centers emerge as strong intersections in parameter space.
Mathematically:
$$
A(a, b, r) = \sum_{x, y \in \text{edges}} \delta((x - a)^2 + (y - b)^2 - r^2)
$$

Peaks in $A$ correspond to circles supported by many edge points.

#### Try It Yourself

1. Use a simple image with one circle, test detection accuracy.
2. Add Gaussian noise, see how thresholds affect results.
3. Detect multiple circles with different radii.
4. Try on real images (coins, wheels, clock faces).

#### Test Cases

| Image            | Radius Range | Result               | Notes                        |
| ---------------- | ------------ | -------------------- | ---------------------------- |
| Synthetic circle | 10–50        | Perfect detection    | Simple edge pattern          |
| Coins photo      | 20–100       | Multiple detections  | Overlapping circles          |
| Clock dial       | 30–80        | Clean edges          | Works even with partial arcs |
| Noisy image      | 10–80        | Some false positives | Can adjust `param2`          |

#### Complexity

| Operation      | Time                   | Space                  |
| -------------- | ---------------------- | ---------------------- |
| Voting         | $O(N \cdot R)$         | $O(A \cdot B \cdot R)$ |
| Peak detection | $O(A \cdot B \cdot R)$ | $O(1)$                 |

Where:

* $N$ = number of edge pixels
* $R$ = number of radius values tested
* $(A, B)$ = possible center coordinates

The Hough Transform for Circles brings geometry to life —
every pixel's whisper of curvature accumulates into a clear voice of shape,
revealing circles hidden in the noise and geometry woven through the image.

### 775 Harris Corner Detector

The Harris Corner Detector identifies *corners*, points where image intensity changes sharply in multiple directions.
These points are ideal for tracking, matching, and recognizing patterns across frames or views.
Unlike edge detectors (which respond to one direction of change), corner detectors respond to two.

#### What Problem Are We Solving?

Corners are stable, distinctive features, ideal landmarks for tasks like:

* Object recognition
* Image stitching
* Optical flow
* 3D reconstruction

A good corner detector should be:

1. Repeatable (found under different lighting/viewing conditions)
2. Accurate (precise localization)
3. Efficient (fast to compute)

The Harris Detector achieves all three using image gradients and a simple mathematical test.

#### How It Works (Plain Language)

Consider shifting a small window around an image.
If the window is flat, the pixel values barely change.
If it's along an edge, intensity changes in *one direction*.
If it's at a corner, intensity changes in *two directions*.

We can quantify that using local gradient information.

#### Mathematical Formulation

1. For a window centered at $(x, y)$, define the change in intensity after a shift $(u, v)$ as:

   $$
   E(u, v) = \sum_{x, y} w(x, y) [I(x + u, y + v) - I(x, y)]^2
   $$

   where $w(x, y)$ is a Gaussian weighting function.

2. Using a Taylor expansion for small shifts:

   $$
   I(x + u, y + v) \approx I(x, y) + I_x u + I_y v
   $$

   Substituting and simplifying gives:

   $$
   E(u, v) = [u \ v]
   \begin{bmatrix}
   A & C \
   C & B
   \end{bmatrix}
   \begin{bmatrix}
   u \
   v
   \end{bmatrix}
   $$

   where
   $A = \sum w(x, y) I_x^2$,
   $B = \sum w(x, y) I_y^2$,
   $C = \sum w(x, y) I_x I_y$.

   The $2\times2$ matrix
   $$
   M =
   \begin{bmatrix}
   A & C \
   C & B
   \end{bmatrix}
   $$
   is called the structure tensor or second-moment matrix.

#### Corner Response Function

To determine if a point is flat, edge, or corner, we examine the eigenvalues $\lambda_1, \lambda_2$ of $M$:

| Case  | $\lambda_1$ | $\lambda_2$ | Type |
| ----- | ----------- | ----------- | ---- |
| Small | Small       | Flat region |      |
| Large | Small       | Edge        |      |
| Large | Large       | Corner      |      |

Instead of computing eigenvalues explicitly, Harris proposed a simpler function:

$$
R = \det(M) - k (\operatorname{trace}(M))^2
$$

where
$\det(M) = AB - C^2$,
$\operatorname{trace}(M) = A + B$,
and $k$ is typically between $0.04$ and $0.06$.

If $R$ is large and positive → corner.
If $R$ is negative → edge.
If $R$ is small → flat area.

#### Tiny Code (Python Example using OpenCV)

```python
import cv2
import numpy as np

img = cv2.imread('input.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)

dst = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)
dst = cv2.dilate(dst, None)

img[dst > 0.01 * dst.max()] = [0, 0, 255]
cv2.imwrite('harris_corners.jpg', img)
```

#### Why It Matters

* Detects stable, distinctive keypoints for matching and tracking.
* Simple and computationally efficient.
* Basis for modern detectors like Shi–Tomasi, FAST, and ORB.
* Excellent for camera motion analysis, SLAM, and stereo vision.

#### A Gentle Proof (Why It Works)

At a true corner, both gradient directions carry significant information.
The structure tensor $M$ captures these gradients through its eigenvalues.

When both $\lambda_1$ and $\lambda_2$ are large,
the local intensity function changes sharply regardless of shift direction,
which is precisely what defines a corner.

The response $R$ measures this curvature indirectly through $\det(M)$ and $\operatorname{trace}(M)$,
avoiding expensive eigenvalue computation but preserving their geometric meaning.

#### Try It Yourself

1. Apply Harris to a chessboard image, perfect for corners.
2. Change parameter $k$ and threshold, watch how many corners are detected.
3. Try on natural images or faces, note that textured regions generate many responses.

#### Test Cases

| Image         | Expected Corners | Notes                               |
| ------------- | ---------------- | ----------------------------------- |
| Checkerboard  | ~80              | Clear sharp corners                 |
| Road sign     | 4–8              | Strong edges, stable corners        |
| Natural scene | Many             | Textures produce multiple responses |
| Blurred photo | Few              | Corners fade as gradients weaken    |

#### Complexity

| Operation            | Time            | Space           |
| -------------------- | --------------- | --------------- |
| Gradient computation | $O(W \times H)$ | $O(W \times H)$ |
| Tensor + response    | $O(W \times H)$ | $O(W \times H)$ |
| Non-max suppression  | $O(W \times H)$ | $O(1)$          |

The Harris Corner Detector finds where light bends the most in an image —
the crossroads of brightness, where information density peaks,
and where geometry and perception quietly agree that "something important is here."

### 776 FAST Corner Detector

The FAST (Features from Accelerated Segment Test) corner detector is a lightning-fast alternative to the Harris detector.
It skips heavy matrix math and instead uses a simple intensity comparison test around each pixel to determine if it is a corner.
FAST is widely used in real-time applications such as SLAM, AR tracking, and mobile vision due to its remarkable speed and simplicity.

#### What Problem Are We Solving?

The Harris detector, while accurate, involves computing gradients and matrix operations for every pixel, expensive for large or real-time images.
FAST instead tests whether a pixel's neighborhood shows sharp brightness contrast in multiple directions,
a hallmark of corner-like behavior, but without using derivatives.

The key idea:

> A pixel is a corner if a set of pixels around it are significantly brighter or darker than it by a certain threshold.

#### How It Works (Plain Language)

1. Consider a circle of 16 pixels around each candidate pixel $p$.
   These are spaced evenly (Bresenham circle of radius 3).

2. For each neighbor pixel $x$, compare its intensity $I(x)$ to $I(p)$:

   * Brighter if $I(x) > I(p) + t$
   * Darker if $I(x) < I(p) - t$

3. A pixel $p$ is declared a corner if there exists a contiguous arc of $n$ pixels
   (usually $n = 12$ out of 16) that are all brighter or all darker than $I(p)$ by the threshold $t$.

4. Perform non-maximum suppression to keep only the strongest corners.

This test avoids floating-point computation entirely and is therefore ideal for embedded or real-time systems.

#### Mathematical Description

Let $I(p)$ be the intensity at pixel $p$, and $S_{16}$ be the 16 pixels around it.
Then $p$ is a corner if there exists a sequence of $n$ contiguous pixels $x_i$ in $S_{16}$ satisfying

$$
I(x_i) > I(p) + t \quad \forall i
$$
or
$$
I(x_i) < I(p) - t \quad \forall i
$$

for a fixed threshold $t$.

#### Step-by-Step Algorithm

1. Precompute the 16 circle offsets.
2. For each pixel $p$:

   * Compare four key pixels (1, 5, 9, 13) to quickly reject most candidates.
   * If at least three of these are all brighter or all darker, proceed to the full 16-pixel test.
3. Mark $p$ as a corner if a contiguous segment of $n$ satisfies the intensity rule.
4. Apply non-maximum suppression to refine corner locations.

#### Tiny Code (Python Example using OpenCV)

```python
import cv2

img = cv2.imread('input.jpg', cv2.IMREAD_GRAYSCALE)

# Initialize FAST detector
fast = cv2.FastFeatureDetector_create(threshold=30, nonmaxSuppression=True)

# Detect keypoints
kp = fast.detect(img, None)

# Draw and save result
img_out = cv2.drawKeypoints(img, kp, None, color=(0,255,0))
cv2.imwrite('fast_corners.jpg', img_out)
```

#### Why It Matters

* Extremely fast and simple, no gradient, no matrix math.
* Suitable for real-time tracking, mobile AR, and robot navigation.
* Used as the base for higher-level descriptors like ORB (Oriented FAST and Rotated BRIEF).
* Corner response is based purely on intensity contrast, making it efficient on low-power hardware.

#### A Gentle Proof (Why It Works)

A corner is where brightness changes sharply in multiple directions.
The circular test simulates this by requiring a sequence of consistently brighter or darker pixels around a center.
If intensity varies only in one direction, the contiguous condition fails, the pattern is an edge, not a corner.

The test effectively measures multi-directional contrast, approximating the same intuition as Harris
but using simple integer comparisons instead of differential analysis.

#### Try It Yourself

1. Run FAST on a high-resolution image; note how quickly corners appear.
2. Increase or decrease the threshold $t$ to control sensitivity.
3. Compare results with Harris, are the corners similar in location but faster to compute?
4. Disable `nonmaxSuppression` to see the raw response map.

#### Test Cases

| Image         | Threshold | Corners Detected | Observation                 |
| ------------- | --------- | ---------------- | --------------------------- |
| Checkerboard  | 30        | ~100             | Very stable detection       |
| Textured wall | 20        | 300–400          | High density due to texture |
| Natural photo | 40        | 60–120           | Reduced to strong features  |
| Low contrast  | 15        | Few              | Fails in flat lighting      |

#### Complexity

| Operation           | Time            | Space  |
| ------------------- | --------------- | ------ |
| Pixel comparison    | $O(W \times H)$ | $O(1)$ |
| Non-max suppression | $O(W \times H)$ | $O(1)$ |

The runtime depends only on the image size, not the gradient or window size.

The FAST Corner Detector trades mathematical elegance for speed and practicality.
It listens to the rhythm of brightness around each pixel —
and when that rhythm changes sharply in many directions, it says, simply and efficiently,
"here lies a corner."

### 777 SIFT (Scale-Invariant Feature Transform)

The SIFT (Scale-Invariant Feature Transform) algorithm finds distinctive, repeatable keypoints in images, robust to scale, rotation, and illumination changes.
It not only detects corners or blobs but also builds descriptors, small numeric fingerprints that allow features to be matched across different images.
This makes SIFT a foundation for image stitching, 3D reconstruction, and object recognition.

#### What Problem Are We Solving?

A corner detector like Harris or FAST works only at a fixed scale and orientation.
But in real-world vision tasks, objects appear at different sizes, angles, and lighting.

SIFT solves this by detecting scale- and rotation-invariant features.
Its key insight: build a *scale space* and locate stable patterns (extrema) that persist across levels of image blur.

#### How It Works (Plain Language)

The algorithm has four main stages:

1. Scale-space construction, progressively blur the image using Gaussians.
2. Keypoint detection, find local extrema across both space and scale.
3. Orientation assignment, compute gradient direction to make rotation invariant.
4. Descriptor generation, capture the local gradient pattern into a 128-dimensional vector.

Each step strengthens invariance: first scale, then rotation, then illumination.

#### 1. Scale-Space Construction

A *scale space* is created by repeatedly blurring the image with Gaussian filters of increasing standard deviation $\sigma$.

$$
L(x, y, \sigma) = G(x, y, \sigma) * I(x, y)
$$

where

$$
G(x, y, \sigma) = \frac{1}{2\pi\sigma^2} e^{-(x^2 + y^2)/(2\sigma^2)}
$$

To detect stable structures, compute the Difference of Gaussians (DoG):

$$
D(x, y, \sigma) = L(x, y, k\sigma) - L(x, y, \sigma)
$$

The DoG approximates the Laplacian of Gaussian, a blob detector.

#### 2. Keypoint Detection

A pixel is a keypoint if it is a local maximum or minimum in a $3\times3\times3$ neighborhood (across position and scale).
This means it's larger or smaller than its 26 neighbors in both space and scale.

Low-contrast and edge-like points are discarded to improve stability.

#### 3. Orientation Assignment

For each keypoint, compute local image gradients:

$$
m(x, y) = \sqrt{(L_x)^2 + (L_y)^2}, \quad \theta(x, y) = \tan^{-1}(L_y / L_x)
$$

A histogram of gradient directions (0–360°) is built within a neighborhood around the keypoint.
The peak of this histogram defines the keypoint's orientation.
If there are multiple strong peaks, multiple orientations are assigned.

This gives rotation invariance.

#### 4. Descriptor Generation

For each oriented keypoint, take a $16 \times 16$ region around it, divided into $4 \times 4$ cells.
For each cell, compute an 8-bin gradient orientation histogram, weighted by magnitude and Gaussian falloff.

This yields $4 \times 4 \times 8 = 128$ numbers, the SIFT descriptor vector.

Finally, normalize the descriptor to reduce lighting effects.

#### Tiny Code (Python Example using OpenCV)

```python
import cv2

img = cv2.imread('input.jpg', cv2.IMREAD_GRAYSCALE)

# Create SIFT detector
sift = cv2.SIFT_create()

# Detect keypoints and descriptors
kp, des = sift.detectAndCompute(img, None)

# Draw keypoints
img_out = cv2.drawKeypoints(img, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imwrite('sift_features.jpg', img_out)
```

#### Why It Matters

* Scale- and rotation-invariant.
* Robust to noise, lighting, and affine transformations.
* Forms the basis of many modern feature matchers (e.g., SURF, ORB, AKAZE).
* Critical for panoramic stitching, 3D reconstruction, and localization.

#### A Gentle Proof (Why It Works)

The Gaussian scale-space ensures that keypoints persist under changes in scale.
Because the Laplacian of Gaussian is invariant to scaling, detecting extrema in the Difference of Gaussians approximates this behavior efficiently.

Assigning dominant gradient orientation ensures rotational invariance:
$$
f'(x', y') = f(R_\theta x, R_\theta y)
$$
The descriptor's normalized histograms make it robust to illumination scaling:
$$
\frac{f'(x, y)}{||f'(x, y)||} = \frac{k f(x, y)}{||k f(x, y)||} = \frac{f(x, y)}{||f(x, y)||}
$$

#### Try It Yourself

1. Run SIFT on the same object at different scales, observe consistent keypoints.
2. Rotate the image 45°, check that SIFT matches corresponding points.
3. Use `cv2.BFMatcher()` to visualize matching between two images.

#### Test Cases

| Scene                       | Expected Matches | Observation                      |
| --------------------------- | ---------------- | -------------------------------- |
| Same object, different zoom | 50–100           | Stable matches                   |
| Rotated view                | 50+              | Keypoints preserved              |
| Low light                   | 30–60            | Gradients still distinct         |
| Different objects           | 0                | Descriptors reject false matches |

#### Complexity

| Step                   | Time                     | Space                    |
| ---------------------- | ------------------------ | ------------------------ |
| Gaussian pyramid       | $O(W \times H \times S)$ | $O(W \times H \times S)$ |
| DoG extrema detection  | $O(W \times H \times S)$ | $O(W \times H)$          |
| Descriptor computation | $O(K)$                   | $O(K)$                   |

where $S$ = number of scales per octave, $K$ = number of keypoints.

The SIFT algorithm captures visual structure that survives transformation —
like the bones beneath the skin of an image, unchanged when it grows, turns, or dims.
It sees not pixels, but *patterns that persist through change*.

### 778 SURF (Speeded-Up Robust Features)

The SURF (Speeded-Up Robust Features) algorithm is a streamlined, faster alternative to SIFT.
It retains robustness to scale, rotation, and illumination but replaces heavy Gaussian operations with box filters and integral images,
making it ideal for near real-time applications like tracking and recognition.

#### What Problem Are We Solving?

SIFT is powerful but computationally expensive —
especially the Gaussian pyramids and 128-dimensional descriptors.

SURF tackles this by:

* Using integral images for constant-time box filtering.
* Approximating the Hessian determinant for keypoint detection.
* Compressing descriptors for faster matching.

The result: SIFT-level accuracy at a fraction of the cost.

#### How It Works (Plain Language)

1. Detect interest points using an approximate Hessian matrix.
2. Assign orientation using Haar-wavelet responses.
3. Build descriptors from intensity gradients (but fewer and coarser than SIFT).

Each part is designed to use integer arithmetic and fast summations via integral images.

#### 1. Integral Image

An integral image allows fast computation of box filter sums:

$$
I_{\text{int}}(x, y) = \sum_{i \le x, j \le y} I(i, j)
$$

Any rectangular region sum can then be computed in $O(1)$ using only four array accesses.

#### 2. Keypoint Detection (Hessian Approximation)

SURF uses the Hessian determinant to find blob-like regions:

$$
H(x, y, \sigma) =
\begin{bmatrix}
L_{xx}(x, y, \sigma) & L_{xy}(x, y, \sigma) \
L_{xy}(x, y, \sigma) & L_{yy}(x, y, \sigma)
\end{bmatrix}
$$

and computes the determinant:

$$
\det(H) = L_{xx} L_{yy} - (0.9L_{xy})^2
$$

where derivatives are approximated with box filters of different sizes.
Local maxima across space and scale are retained as keypoints.

#### 3. Orientation Assignment

For each keypoint, compute Haar wavelet responses in $x$ and $y$ directions within a circular region.
A sliding orientation window (typically $60^\circ$ wide) finds the dominant direction.

This ensures rotation invariance.

#### 4. Descriptor Generation

The area around each keypoint is divided into a $4 \times 4$ grid.
For each cell, compute four features based on Haar responses:

$$
(v_x, v_y, |v_x|, |v_y|)
$$

These are concatenated into a 64-dimensional descriptor (vs 128 in SIFT).

For better matching, the descriptor is normalized:

$$
\hat{v} = \frac{v}{||v||}
$$

#### Tiny Code (Python Example using OpenCV)

```python
import cv2

img = cv2.imread('input.jpg', cv2.IMREAD_GRAYSCALE)

# Initialize SURF (may require nonfree module)
surf = cv2.xfeatures2d.SURF_create(hessianThreshold=400)

# Detect keypoints and descriptors
kp, des = surf.detectAndCompute(img, None)

# Draw and save results
img_out = cv2.drawKeypoints(img, kp, None, (255,0,0), 4)
cv2.imwrite('surf_features.jpg', img_out)
```

#### Why It Matters

* Faster than SIFT, robust to blur, scale, and rotation.
* Works well for object recognition, registration, and tracking.
* Reduced descriptor dimensionality (64) enables faster matching.
* Can run efficiently on mobile and embedded hardware.

#### A Gentle Proof (Why It Works)

The determinant of the Hessian captures local curvature —
strong positive curvature in both directions indicates a blob or corner-like structure.
Using integral images ensures that even large-scale filters can be computed in constant time:

$$
\text{BoxSum}(x_1, y_1, x_2, y_2) =
I_{\text{int}}(x_2, y_2) - I_{\text{int}}(x_2, y_1) - I_{\text{int}}(x_1, y_2) + I_{\text{int}}(x_1, y_1)
$$

Thus SURF's speedup comes directly from mathematical simplification —
replacing convolution with difference-of-box sums without losing the geometric essence of the features.

#### Try It Yourself

1. Compare SURF and SIFT keypoints on the same image.
2. Adjust `hessianThreshold`, higher values yield fewer but more stable keypoints.
3. Test on rotated or scaled versions of the image to verify invariance.

#### Test Cases

| Image          | Detector Threshold | Keypoints | Descriptor Dim | Notes                      |
| -------------- | ------------------ | --------- | -------------- | -------------------------- |
| Checkerboard   | 400                | 80        | 64             | Stable grid corners        |
| Landscape      | 300                | 400       | 64             | Rich texture               |
| Rotated object | 400                | 70        | 64             | Orientation preserved      |
| Noisy image    | 200                | 200       | 64             | Still detects stable blobs |

#### Complexity

| Step             | Time            | Space           |
| ---------------- | --------------- | --------------- |
| Integral image   | $O(W \times H)$ | $O(W \times H)$ |
| Hessian response | $O(W \times H)$ | $O(1)$          |
| Descriptor       | $O(K)$          | $O(K)$          |

where $K$ is the number of detected keypoints.

The SURF algorithm captures the essence of SIFT in half the time —
a feat of mathematical efficiency,
turning the elegance of continuous Gaussian space into a set of fast, discrete filters
that see the world sharply and swiftly.

### 779 ORB (Oriented FAST and Rotated BRIEF)

The ORB (Oriented FAST and Rotated BRIEF) algorithm combines the speed of FAST with the descriptive power of BRIEF —
producing a lightweight yet highly effective feature detector and descriptor.
It's designed for real-time vision tasks like SLAM, AR tracking, and image matching,
and is fully open and patent-free, unlike SIFT or SURF.

#### What Problem Are We Solving?

SIFT and SURF are powerful but computationally expensive and historically patented.
FAST is extremely fast but lacks orientation or descriptors.
BRIEF is compact but not rotation invariant.

ORB unifies all three goals:

* FAST keypoints
* Rotation invariance
* Binary descriptors (for fast matching)

All in one efficient pipeline.

#### How It Works (Plain Language)

1. Detect corners using FAST.
2. Assign orientation to each keypoint based on image moments.
3. Compute a rotated BRIEF descriptor around the keypoint.
4. Use binary Hamming distance for matching.

It's both rotation- and scale-invariant, compact, and lightning fast.

#### 1. Keypoint Detection (FAST)

ORB starts with the FAST detector to find candidate corners.

For each pixel $p$ and its circular neighborhood $S_{16}$:

* If at least 12 contiguous pixels in $S_{16}$ are all brighter or darker than $p$ by a threshold $t$,
  then $p$ is a corner.

To improve stability, ORB applies FAST on a Gaussian pyramid,
capturing features across multiple scales.

#### 2. Orientation Assignment

Each keypoint is given an orientation using intensity moments:

$$
m_{pq} = \sum_x \sum_y x^p y^q I(x, y)
$$

The centroid of the patch is:

$$
C = \left( \frac{m_{10}}{m_{00}}, \frac{m_{01}}{m_{00}} \right)
$$

and the orientation is given by:

$$
\theta = \tan^{-1}\left(\frac{m_{01}}{m_{10}}\right)
$$

This ensures the descriptor can be aligned to the dominant direction.

#### 3. Descriptor Generation (Rotated BRIEF)

BRIEF (Binary Robust Independent Elementary Features)
builds a binary string from pairwise intensity comparisons in a patch.

For $n$ random pairs of pixels $(p_i, q_i)$ in a patch around the keypoint:

$$
\tau(p_i, q_i) =
\begin{cases}
1, & \text{if } I(p_i) < I(q_i) \\
0, & \text{otherwise}
\end{cases}
$$


The descriptor is the concatenation of these bits, typically 256 bits long.

In ORB, this sampling pattern is rotated by the keypoint's orientation $\theta$,
giving rotation invariance.

#### 4. Matching (Hamming Distance)

ORB descriptors are binary strings,
so feature matching uses Hamming distance —
the number of differing bits between two descriptors.

This makes matching incredibly fast with bitwise XOR operations.

#### Tiny Code (Python Example using OpenCV)

```python
import cv2

img = cv2.imread('input.jpg', cv2.IMREAD_GRAYSCALE)

# Initialize ORB
orb = cv2.ORB_create(nfeatures=500)

# Detect keypoints and descriptors
kp, des = orb.detectAndCompute(img, None)

# Draw results
img_out = cv2.drawKeypoints(img, kp, None, color=(0,255,0))
cv2.imwrite('orb_features.jpg', img_out)
```

#### Why It Matters

* Fast like FAST, descriptive like SIFT, compact like BRIEF.
* Binary descriptors make matching up to 10× faster than SIFT/SURF.
* Fully free and open-source, ideal for commercial use.
* Core component in SLAM, robotics, and mobile computer vision.

#### A Gentle Proof (Why It Works)

The orientation step ensures rotational invariance.
Let $I'(x, y)$ be a rotated version of $I(x, y)$ by angle $\theta$.
Then the centroid-based orientation guarantees that:

$$
BRIEF'(p_i, q_i) = BRIEF(R_{-\theta} p_i, R_{-\theta} q_i)
$$

meaning the same keypoint produces the same binary descriptor after rotation.

Hamming distance is a metric for binary vectors,
so matching remains efficient and robust even under moderate illumination changes.

#### Try It Yourself

1. Detect ORB keypoints on two rotated versions of the same image.
2. Use `cv2.BFMatcher(cv2.NORM_HAMMING)` to match features.
3. Compare speed with SIFT, notice how fast ORB runs.
4. Increase `nfeatures` and test the tradeoff between accuracy and runtime.

#### Test Cases

| Scene           | Keypoints | Descriptor Length | Matching Speed | Notes                     |
| --------------- | --------- | ----------------- | -------------- | ------------------------- |
| Checkerboard    | ~500      | 256 bits          | Fast           | Stable grid corners       |
| Rotated object  | ~400      | 256 bits          | Fast           | Rotation preserved        |
| Low contrast    | ~200      | 256 bits          | Fast           | Contrast affects FAST     |
| Real-time video | 300–1000  | 256 bits          | Real-time      | Works on embedded devices |

#### Complexity

| Step               | Time            | Space  |
| ------------------ | --------------- | ------ |
| FAST detection     | $O(W \times H)$ | $O(1)$ |
| BRIEF descriptor   | $O(K)$          | $O(K)$ |
| Matching (Hamming) | $O(K \log K)$   | $O(K)$ |

where $K$ = number of keypoints.

The ORB algorithm is the street-smart hybrid of computer vision —
it knows SIFT's elegance, BRIEF's thrift, and FAST's hustle.
Quick on its feet, rotation-aware, and bitwise efficient,
it captures structure with speed that even hardware loves.

### 780 RANSAC (Random Sample Consensus)

The RANSAC (Random Sample Consensus) algorithm is a robust method for estimating models from data that contain outliers.
It repeatedly fits models to random subsets of points and selects the one that best explains the majority of data.
In computer vision, RANSAC is a backbone of feature matching, homography estimation, and motion tracking, it finds structure amid noise.

#### What Problem Are We Solving?

Real-world data is messy.
When matching points between two images, some correspondences are wrong, these are outliers.
If you run standard least-squares fitting, even a few bad matches can ruin your model.

RANSAC solves this by embracing randomness:
it tests many small subsets, trusting consensus rather than precision from any single sample.

#### How It Works (Plain Language)

RANSAC's idea is simple:

1. Randomly pick a minimal subset of data points.
2. Fit a model to this subset.
3. Count how many other points agree with this model within a tolerance, these are inliers.
4. Keep the model with the largest inlier set.
5. Optionally, refit the model using all inliers for precision.

You don't need all the data, just enough agreement.

#### Mathematical Overview

Let:

* $N$ = total number of data points
* $s$ = number of points needed to fit the model (e.g. $s=2$ for a line, $s=4$ for a homography)
* $p$ = probability that at least one random sample is free of outliers
* $\epsilon$ = fraction of outliers

Then the required number of iterations $k$ is:

$$
k = \frac{\log(1 - p)}{\log(1 - (1 - \epsilon)^s)}
$$

This tells us how many random samples to test for a given confidence.

#### Example: Line Fitting

Given 2D points, we want to find the best line $y = mx + c$.

1. Randomly select two points.
2. Compute slope $m$ and intercept $c$.
3. Count how many other points lie within distance $d$ of this line:

$$
\text{error}(x_i, y_i) = \frac{|y_i - (mx_i + c)|}{\sqrt{1 + m^2}}
$$

4. The line with the largest number of inliers is chosen as the best.

#### Tiny Code (Python Example)

```python
import numpy as np
import random

def ransac_line(points, n_iter=1000, threshold=1.0):
    best_m, best_c, best_inliers = None, None, []
    for _ in range(n_iter):
        sample = random.sample(points, 2)
        (x1, y1), (x2, y2) = sample
        if x2 == x1:
            continue
        m = (y2 - y1) / (x2 - x1)
        c = y1 - m * x1
        inliers = []
        for (x, y) in points:
            err = abs(y - (m*x + c)) / np.sqrt(1 + m2)
            if err < threshold:
                inliers.append((x, y))
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_m, best_c = m, c
    return best_m, best_c, best_inliers
```

#### Why It Matters

* Robust to outliers, works even if 50–80% of the data is bad.
* Model-agnostic, can fit lines, planes, fundamental matrices, homographies, etc.
* Simple and flexible, only needs a model-fitting routine and an error metric.

Used everywhere from:

* Image stitching (homography estimation)
* Stereo vision (epipolar geometry)
* 3D reconstruction
* Motion estimation in robotics

#### A Gentle Proof (Why It Works)

Each random subset has a probability $(1 - \epsilon)^s$ of containing only inliers.
After $k$ iterations, the probability that no sample is pure is $(1 - (1 - \epsilon)^s)^k$.
Setting this equal to $1 - p$ gives the iteration formula above.

Thus, after enough random trials, RANSAC almost certainly finds a model supported by the majority —
without being swayed by the minority of outliers.

#### Try It Yourself

1. Generate a noisy dataset with 20% outliers and fit a line using RANSAC.
2. Compare with least-squares, notice how RANSAC stays stable.
3. Apply to feature matching between two photos using `cv2.findHomography(..., cv2.RANSAC)`.

#### Test Cases

| Data                | Outlier Ratio | Model                  | Inlier Rate | Notes                   |
| ------------------- | ------------- | ---------------------- | ----------- | ----------------------- |
| Line points + noise | 20%           | $y = mx + c$           | 95%         | Perfect recovery        |
| Plane in 3D         | 40%           | $ax + by + cz + d = 0$ | 90%         | Robust to bad data      |
| Homography          | 50%           | 3×3 matrix             | 85%         | Used in image stitching |
| Random noise        | 90%           | N/A                    | Low         | Cannot converge         |

#### Complexity

| Step               | Time           | Space  |
| ------------------ | -------------- | ------ |
| Sampling & fitting | $O(k \cdot s)$ | $O(1)$ |
| Inlier counting    | $O(kN)$        | $O(1)$ |

Overall: $O(kN)$, where $k$ depends on desired confidence and outlier ratio.

The RANSAC algorithm is the skeptic's way of seeing truth —
it ignores the crowd, listens to a few honest voices,
and keeps sampling until consensus reveals the right line through the noise.

# Section 79. Pathfinding in Space 

### 781 A* Search

The A* (A-star) algorithm is one of the most elegant and efficient pathfinding algorithms ever designed.
It finds the shortest path between a start and goal node in a graph while minimizing total cost.
Used everywhere from navigation systems and robotics to games and AI planning, A* balances greedy search and uniform cost search through a clever use of heuristics.

#### What Problem Are We Solving?

Given a set of connected nodes (or a grid map) with movement costs between them,
we want the shortest, least-cost path from a start point to a goal point.

Unlike Dijkstra's algorithm, which explores in all directions,
A* uses a heuristic to guide the search toward the goal, much faster and still guaranteed to find the optimal path (under certain conditions).

#### How It Works (Plain Language)

A* keeps two key quantities for each node:

* $g(n)$, cost from start to this node
* $h(n)$, estimated cost from this node to the goal (the heuristic)
* $f(n) = g(n) + h(n)$, total estimated cost through this node

It expands the node with the lowest $f(n)$ until the goal is reached.
The heuristic keeps the search focused; $g$ ensures optimality.

#### Step-by-Step Algorithm

1. Initialize two sets:

   * Open list, nodes to be evaluated (start node initially)
   * Closed list, nodes already evaluated

2. For the current node:

   * Compute $f(n) = g(n) + h(n)$
   * Choose the node with lowest $f(n)$ in the open list
   * Move it to the closed list

3. For each neighbor:

   * Compute tentative $g_{new} = g(\text{current}) + \text{cost(current, neighbor)}$
   * If neighbor not in open list or $g_{new}$ is smaller, update it:

     * $g(\text{neighbor}) = g_{new}$
     * $f(\text{neighbor}) = g_{new} + h(\text{neighbor})$
     * Set parent to current

4. Stop when goal node is selected for expansion.

#### Heuristic Examples

| Domain            | Heuristic Function $h(n)$                             | Property   |   |           |   |            |
| ----------------- | ----------------------------------------------------- | ---------- | - | --------- | - | ---------- |
| Grid (4-neighbor) | Manhattan distance $                                  | x_1 - x_2  | + | y_1 - y_2 | $ | Admissible |
| Grid (8-neighbor) | Euclidean distance $\sqrt{(x_1-x_2)^2 + (y_1-y_2)^2}$ | Admissible |   |           |   |            |
| Weighted graph    | Minimum edge weight × remaining nodes                 | Admissible |   |           |   |            |

A heuristic is admissible if it never overestimates the true cost to the goal.
If it's also consistent, A* guarantees optimality without revisiting nodes.

#### Tiny Code (Python Example)

```python
import heapq

def a_star(start, goal, neighbors, heuristic):
    open_set = [(0, start)]
    came_from = {}
    g = {start: 0}

    while open_set:
        _, current = heapq.heappop(open_set)
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            return path[::-1]

        for next_node, cost in neighbors(current):
            new_g = g[current] + cost
            if next_node not in g or new_g < g[next_node]:
                g[next_node] = new_g
                f = new_g + heuristic(next_node, goal)
                heapq.heappush(open_set, (f, next_node))
                came_from[next_node] = current
    return None
```

#### Why It Matters

* Optimal and complete (with admissible heuristics)
* Efficient, explores only promising paths
* Widely used in:

  * GPS navigation
  * Video game AI (NPC movement)
  * Robot motion planning
  * Graph-based optimization problems

A* is a beautiful example of how a simple idea, combining real cost and estimated cost, produces deep practical power.

#### A Gentle Proof (Why It Works)

Let $f(n) = g(n) + h(n)$.
If $h(n)$ never overestimates the true distance to the goal,
then the first time the goal node is selected for expansion,
the path found must have the minimum cost.

Formally, for admissible $h$:

$$
h(n) \le h^*(n)
$$

where $h^*(n)$ is the true cost to goal.
Thus $f(n)$ is always a lower bound on the total cost through $n$,
and A* never misses the globally optimal path.

#### Try It Yourself

1. Implement A* on a 2D grid, mark walls as obstacles.
2. Try different heuristics (Manhattan, Euclidean, zero).
3. Compare to Dijkstra, notice how A* expands fewer nodes.
4. Visualize the open/closed lists, it's like watching reasoning unfold on a map.

#### Test Cases

| Grid Size | Obstacles  | Heuristic       | Result                  |
| --------- | ---------- | --------------- | ----------------------- |
| 5×5       | None       | Manhattan       | Straight path           |
| 10×10     | Random 20% | Manhattan       | Detour path found       |
| 50×50     | Maze       | Euclidean       | Efficient shortest path |
| 100×100   | 30%        | Zero (Dijkstra) | Slower but same path    |

#### Complexity

| Term  | Meaning                              | Typical Value            |
| ----- | ------------------------------------ | ------------------------ |
| Time  | $O(E)$ worst case, usually much less | Depends on heuristic     |
| Space | $O(V)$                               | Store open + closed sets |

With an admissible heuristic, A* can approach linear time in sparse maps, remarkably efficient for a general optimal search.

The A* algorithm walks the line between foresight and discipline —
it doesn't wander aimlessly like Dijkstra,
nor does it leap blindly like Greedy Best-First.
It *plans*, balancing knowledge of the road traveled and intuition of the road ahead.

### 782 Dijkstra for Grid

Dijkstra's algorithm is the classic foundation of shortest-path computation.
In its grid-based version, it systematically explores all reachable nodes in order of increasing cost, guaranteeing the shortest route to every destination.
While A* adds a heuristic, Dijkstra operates purely on accumulated distance,
making it the gold standard for unbiased, optimal pathfinding when no goal direction or heuristic is known.

#### What Problem Are We Solving?

Given a 2D grid (or any weighted graph),
each cell has edges to its neighbors with movement cost $w \ge 0$.
We want to find the minimum total cost path from a source to all other nodes —
or to a specific goal if one exists.

Dijkstra ensures that once a node's cost is finalized,
no shorter path to it can ever exist.

#### How It Works (Plain Language)

1. Assign distance $d = 0$ to the start cell and $∞$ to all others.
2. Place the start in a priority queue.
3. Repeatedly pop the node with the lowest current cost.
4. For each neighbor, compute tentative distance:

   $$
   d_{new} = d_{current} + w(current, neighbor)
   $$

   If $d_{new}$ is smaller, update the neighbor's distance and reinsert it into the queue.
5. Continue until all nodes are processed or the goal is reached.

Each node is "relaxed" exactly once,
ensuring efficiency and optimality.

#### Example (4-neighbor grid)

Consider a grid where moving horizontally or vertically costs 1:

$$
\text{Start} = (0, 0), \quad \text{Goal} = (3, 3)
$$

After each expansion, the wavefront of known minimal distances expands outward:

| Step | Frontier Cells      | Cost |
| ---- | ------------------- | ---- |
| 1    | (0,0)               | 0    |
| 2    | (0,1), (1,0)        | 1    |
| 3    | (0,2), (1,1), (2,0) | 2    |
| ...  | ...                 | ...  |
| 6    | (3,3)               | 6    |

#### Tiny Code (Python Example)

```python
import heapq

def dijkstra_grid(start, goal, grid):
    rows, cols = len(grid), len(grid[0])
    dist = {start: 0}
    pq = [(0, start)]
    came_from = {}

    def neighbors(cell):
        x, y = cell
        for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
            nx, ny = x+dx, y+dy
            if 0 <= nx < rows and 0 <= ny < cols and grid[nx][ny] == 0:
                yield (nx, ny), 1  # cost 1

    while pq:
        d, current = heapq.heappop(pq)
        if current == goal:
            break
        for (nxt, cost) in neighbors(current):
            new_d = d + cost
            if new_d < dist.get(nxt, float('inf')):
                dist[nxt] = new_d
                came_from[nxt] = current
                heapq.heappush(pq, (new_d, nxt))
    return dist, came_from
```

#### Why It Matters

* Optimal and complete, always finds the shortest path.
* Foundation for modern algorithms like A*, Bellman–Ford, and Floyd–Warshall.
* Versatile, works for grids, networks, and weighted graphs.
* Deterministic, explores all equally good paths without heuristic bias.

Used in:

* Network routing (e.g., OSPF, BGP)
* Game AI for exploration zones
* Path planning for autonomous robots

#### A Gentle Proof (Why It Works)

The key invariant:
when a node $u$ is removed from the priority queue, its shortest path distance $d(u)$ is final.

Proof sketch:
If there were a shorter path to $u$, some intermediate node $v$ on that path would have a smaller tentative distance,
so $v$ would have been extracted before $u$.
Thus, $d(u)$ cannot be improved afterward.

This guarantees optimality with non-negative edge weights.

#### Try It Yourself

1. Run Dijkstra on a grid with different obstacle patterns.
2. Modify edge weights to simulate terrain (e.g., grass = 1, mud = 3).
3. Compare explored nodes with A*, notice how Dijkstra expands evenly, while A* focuses toward the goal.
4. Implement an 8-direction version and measure the path difference.

#### Test Cases

| Grid    | Obstacle % | Cost Metric      | Result                      |
| ------- | ---------- | ---------------- | --------------------------- |
| 5×5     | 0          | Uniform          | Straight line               |
| 10×10   | 20         | Uniform          | Detour found                |
| 10×10   | 0          | Variable weights | Follows low-cost path       |
| 100×100 | 30         | Uniform          | Expands all reachable cells |

#### Complexity

| Operation                 | Time                              | Space           |
| ------------------------- | --------------------------------- | --------------- |
| Priority queue operations | $O((V + E)\log V)$                | $O(V)$          |
| Grid traversal            | $O(W \times H \log (W \times H))$ | $O(W \times H)$ |

In uniform-cost grids, it behaves like a breadth-first search with weighted precision.

The Dijkstra algorithm is the calm and methodical explorer of the algorithmic world —
it walks outward in perfect order, considering every possible road until all are measured,
guaranteeing that every destination receives the shortest, fairest path possible.

### 783 Theta* (Any-Angle Pathfinding)

Theta* is an extension of A* that allows any-angle movement on grids, producing paths that look more natural and shorter than those constrained to 4 or 8 directions.
It bridges the gap between discrete grid search and continuous geometric optimization, making it a favorite for robotics, drone navigation, and games where agents move freely through open space.

#### What Problem Are We Solving?

In classic A*, movement is limited to grid directions (up, down, diagonal).
Even if the optimal geometric path is straight, A* produces jagged "staircase" routes.

Theta* removes this restriction by checking line-of-sight between nodes:
if the current node's parent can directly see a successor,
it connects them without following grid edges, yielding a smoother, shorter path.

#### How It Works (Plain Language)

Theta* works like A* but modifies how parent connections are made.

For each neighbor `s'` of the current node `s`:

1. Check if `parent(s)` has line-of-sight to `s'`.

   * If yes, set
     $$
     g(s') = g(parent(s)) + \text{dist}(parent(s), s')
     $$
     and
     $$
     parent(s') = parent(s)
     $$
2. Otherwise, behave like standard A*:
   $$
   g(s') = g(s) + \text{dist}(s, s')
   $$
   and
   $$
   parent(s') = s
   $$
3. Update the priority queue with
   $$
   f(s') = g(s') + h(s')
   $$

This simple geometric relaxation gives near-optimal continuous paths
without increasing asymptotic complexity.

#### Tiny Code (Python Example)

```python
import heapq, math

def line_of_sight(grid, a, b):
    # Bresenham-style line check
    x0, y0 = a
    x1, y1 = b
    dx, dy = abs(x1-x0), abs(y1-y0)
    sx, sy = (1 if x1 > x0 else -1), (1 if y1 > y0 else -1)
    err = dx - dy
    while True:
        if grid[x0][y0] == 1:
            return False
        if (x0, y0) == (x1, y1):
            return True
        e2 = 2*err
        if e2 > -dy: err -= dy; x0 += sx
        if e2 < dx: err += dx; y0 += sy

def theta_star(grid, start, goal, heuristic):
    rows, cols = len(grid), len(grid[0])
    g = {start: 0}
    parent = {start: start}
    open_set = [(heuristic(start, goal), start)]

    def dist(a, b): return math.hypot(a[0]-b[0], a[1]-b[1])

    while open_set:
        _, s = heapq.heappop(open_set)
        if s == goal:
            path = []
            while s != parent[s]:
                path.append(s)
                s = parent[s]
            path.append(start)
            return path[::-1]
        for dx in [-1,0,1]:
            for dy in [-1,0,1]:
                if dx == dy == 0: continue
                s2 = (s[0]+dx, s[1]+dy)
                if not (0 <= s2[0] < rows and 0 <= s2[1] < cols): continue
                if grid[s2[0]][s2[1]] == 1: continue
                if line_of_sight(grid, parent[s], s2):
                    new_g = g[parent[s]] + dist(parent[s], s2)
                    if new_g < g.get(s2, float('inf')):
                        g[s2] = new_g
                        parent[s2] = parent[s]
                        f = new_g + heuristic(s2, goal)
                        heapq.heappush(open_set, (f, s2))
                else:
                    new_g = g[s] + dist(s, s2)
                    if new_g < g.get(s2, float('inf')):
                        g[s2] = new_g
                        parent[s2] = s
                        f = new_g + heuristic(s2, goal)
                        heapq.heappush(open_set, (f, s2))
    return None
```

#### Why It Matters

* Produces smooth, realistic paths for agents and robots.
* Closer to Euclidean shortest paths than grid-based A*.
* Retains admissibility if the heuristic is consistent and
  the grid has uniform costs.
* Works well in open fields, drone navigation, autonomous driving, and RTS games.

#### A Gentle Proof (Why It Works)

Theta* modifies A*'s parent linkage to reduce path length:
If `parent(s)` and `s'` have line-of-sight,
then

$$
g'(s') = g(parent(s)) + d(parent(s), s')
$$

is always ≤

$$
g(s) + d(s, s')
$$

since the direct connection is shorter or equal.
Thus, Theta* never overestimates cost, it preserves A*'s optimality
under Euclidean distance and obstacle-free visibility assumptions.

#### Try It Yourself

1. Run Theta* on a grid with few obstacles.
2. Compare the path to A*: Theta* produces gentle diagonals instead of jagged corners.
3. Increase obstacle density, watch how paths adapt smoothly.
4. Try different heuristics (Manhattan vs Euclidean).

#### Test Cases

| Map Type         | A* Path Length | Theta* Path Length | Visual Smoothness |
| ---------------- | -------------- | ------------------ | ----------------- |
| Open grid        | 28.0           | 26.8               | Smooth            |
| Sparse obstacles | 33.2           | 30.9               | Natural arcs      |
| Maze-like        | 52.5           | 52.5               | Equal (blocked)   |
| Random field     | 41.7           | 38.2               | Cleaner motion    |

#### Complexity

| Operation            | Time                         | Space  |
| -------------------- | ---------------------------- | ------ |
| Search               | $O(E \log V)$                | $O(V)$ |
| Line-of-sight checks | $O(L)$ average per expansion |        |

Theta* runs close to A*'s complexity but trades a small overhead for smoother paths and fewer turns.

Theta* is the geometry-aware evolution of A*:
it looks not just at costs but also *visibility*,
weaving direct lines where others see only squares —
turning jagged motion into elegant, continuous travel.

### 784 Jump Point Search (Grid Acceleration)

Jump Point Search (JPS) is an optimization of A* specifically for uniform-cost grids.
It prunes away redundant nodes by "jumping" in straight lines until a significant decision point (a jump point) is reached.
The result is the same optimal path as A*, but found much faster, often several times faster, with fewer node expansions.

#### What Problem Are We Solving?

A* on a uniform grid expands many unnecessary nodes:
when moving straight in an open area, A* explores each cell one by one.
But if all costs are equal, we don't need to stop at every cell —
only when something *changes* (an obstacle or forced turn).

JPS speeds things up by skipping over these "uninteresting" cells
while maintaining full optimality.

#### How It Works (Plain Language)

1. Start from the current node and move along a direction $(dx, dy)$.

2. Continue jumping in that direction until:

   * You hit an obstacle, or
   * You find a forced neighbor (a node with an obstacle beside it that forces a turn), or
   * You reach the goal.

3. Each jump point is treated as a node in A*.

4. Recursively apply jumps in possible directions from each jump point.

This greatly reduces the number of nodes considered while preserving correctness.

#### Tiny Code (Simplified Python Version)

```python
import heapq

def jump(grid, x, y, dx, dy, goal):
    rows, cols = len(grid), len(grid[0])
    while 0 <= x < rows and 0 <= y < cols and grid[x][y] == 0:
        if (x, y) == goal:
            return (x, y)
        # Forced neighbors
        if dx != 0 and dy != 0:
            if (grid[x - dx][y + dy] == 1 and grid[x - dx][y] == 0) or \
               (grid[x + dx][y - dy] == 1 and grid[x][y - dy] == 0):
                return (x, y)
        elif dx != 0:
            if (grid[x + dx][y + 1] == 1 and grid[x][y + 1] == 0) or \
               (grid[x + dx][y - 1] == 1 and grid[x][y - 1] == 0):
                return (x, y)
        elif dy != 0:
            if (grid[x + 1][y + dy] == 1 and grid[x + 1][y] == 0) or \
               (grid[x - 1][y + dy] == 1 and grid[x - 1][y] == 0):
                return (x, y)
        x += dx
        y += dy
    return None

def heuristic(a, b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

def jump_point_search(grid, start, goal):
    open_set = [(0, start)]
    g = {start: 0}
    came_from = {}
    directions = [(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]

    while open_set:
        _, current = heapq.heappop(open_set)
        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return path[::-1]
        for dx, dy in directions:
            jp = jump(grid, current[0]+dx, current[1]+dy, dx, dy, goal)
            if not jp: continue
            new_g = g[current] + heuristic(current, jp)
            if new_g < g.get(jp, float('inf')):
                g[jp] = new_g
                f = new_g + heuristic(jp, goal)
                heapq.heappush(open_set, (f, jp))
                came_from[jp] = current
    return None
```

#### Why It Matters

* Produces the exact same optimal paths as A*,
  but visits far fewer nodes.
* Excellent for large open grids or navigation meshes.
* Retains A*'s optimality and completeness.

Applications include:

* Game AI pathfinding (especially real-time movement)
* Simulation and robotics in uniform environments
* Large-scale map routing

#### A Gentle Proof (Why It Works)

Every path found by JPS corresponds to a valid A* path.
The key observation:
if moving straight doesn't reveal any new neighbors or forced choices,
then intermediate nodes contribute no additional optimal paths.

Formally, pruning these nodes preserves all shortest paths,
because they can be reconstructed by linear interpolation between jump points.
Thus, JPS is path-equivalent to A* under uniform cost.

#### Try It Yourself

1. Run A* and JPS on an open 100×100 grid.

   * Compare node expansions and time.
2. Add random obstacles and see how the number of jumps changes.
3. Visualize jump points, they appear at corners and turning spots.
4. Measure speedup: JPS often reduces expansions by 5×–20×.

#### Test Cases

| Grid Type             | A* Expansions | JPS Expansions | Speedup         |
| --------------------- | ------------- | -------------- | --------------- |
| 50×50 open            | 2500          | 180            | 13.9×           |
| 100×100 open          | 10,000        | 450            | 22×             |
| 100×100 20% obstacles | 7,200         | 900            | 8×              |
| Maze                  | 4,800         | 4,700          | 1× (same as A*) |

#### Complexity

| Term         | Time          | Space  |
| ------------ | ------------- | ------ |
| Average case | $O(k \log n)$ | $O(n)$ |
| Worst case   | $O(n \log n)$ | $O(n)$ |

JPS's performance gain depends heavily on obstacle layout —
the fewer obstacles, the greater the acceleration.

Jump Point Search is a masterclass in search pruning —
it sees that straight paths are already optimal,
skipping the monotony of uniform exploration,
and leaping forward only when a true decision must be made.

### 785 Rapidly-Exploring Random Tree (RRT)

The Rapidly-Exploring Random Tree (RRT) algorithm is a cornerstone of motion planning in robotics and autonomous navigation.
It builds a tree by sampling random points in space and connecting them to the nearest known node that can reach them without collision.
RRTs are particularly useful in high-dimensional, continuous configuration spaces where grid-based algorithms are inefficient.

#### What Problem Are We Solving?

When planning motion for a robot, vehicle, or arm, the configuration space may be continuous and complex.
Instead of discretizing space into cells, RRT samples random configurations and incrementally explores reachable regions,
eventually finding a valid path from the start to the goal.

#### How It Works (Plain Language)

1. Start with a tree `T` initialized at the start position.
2. Sample a random point `x_rand` in configuration space.
3. Find the nearest node `x_near` in the existing tree.
4. Move a small step from `x_near` toward `x_rand` to get `x_new`.
5. If the segment between them is collision-free, add `x_new` to the tree with `x_near` as its parent.
6. Repeat until the goal is reached or a maximum number of samples is reached.

Over time, the tree spreads rapidly into unexplored areas —
hence the name *rapidly-exploring*.

#### Tiny Code (Python Example)

```python
import random, math

def distance(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

def steer(a, b, step):
    d = distance(a, b)
    if d < step:
        return b
    return (a[0] + step*(b[0]-a[0])/d, a[1] + step*(b[1]-a[1])/d)

def rrt(start, goal, is_free, step=1.0, max_iter=5000, goal_bias=0.05):
    tree = {start: None}
    for _ in range(max_iter):
        x_rand = goal if random.random() < goal_bias else (random.uniform(0,100), random.uniform(0,100))
        x_near = min(tree.keys(), key=lambda p: distance(p, x_rand))
        x_new = steer(x_near, x_rand, step)
        if is_free(x_near, x_new):
            tree[x_new] = x_near
            if distance(x_new, goal) < step:
                tree[goal] = x_new
                return tree
    return tree

def reconstruct(tree, goal):
    path = [goal]
    while tree[path[-1]] is not None:
        path.append(tree[path[-1]])
    return path[::-1]
```

Here `is_free(a, b)` is a collision-checking function that ensures motion between points is valid.

#### Why It Matters

* Scalable to high dimensions: works in spaces where grids or Dijkstra become infeasible.
* Probabilistic completeness: if a solution exists, the probability of finding it approaches 1 as samples increase.
* Foundation for RRT* and PRM algorithms.
* Common in:

  * Autonomous drone and car navigation
  * Robotic arm motion planning
  * Game and simulation environments

#### A Gentle Proof (Why It Works)

Let $X_{\text{free}}$ be the obstacle-free region of configuration space.
At each iteration, RRT samples uniformly from $X_{\text{free}}$.
Since $X_{\text{free}}$ is bounded and has non-zero measure,
every region has a positive probability of being sampled.

The tree's nearest-neighbor expansion ensures that new nodes always move closer to unexplored areas.
Thus, as the number of iterations grows, the probability that the tree reaches the goal region tends to 1 —
probabilistic completeness.

#### Try It Yourself

1. Simulate RRT on a 2D grid with circular obstacles.
2. Visualize how the tree expands, it "fans out" from the start into free space.
3. Add more obstacles and observe how branches naturally grow around them.
4. Adjust `step` and `goal_bias` for smoother or faster convergence.

#### Test Cases

| Scenario    | Space     | Obstacles        | Success Rate | Avg. Path Length |
| ----------- | --------- | ---------------- | ------------ | ---------------- |
| Empty space | 2D        | 0%               | 100%         | 140              |
| 20% blocked | 2D        | random           | 90%          | 165              |
| Maze        | 2D        | narrow corridors | 75%          | 210              |
| 3D space    | spherical | 30%              | 85%          | 190              |

#### Complexity

| Operation               | Time                                  | Space  |
| ----------------------- | ------------------------------------- | ------ |
| Nearest-neighbor search | $O(N)$ (naive), $O(\log N)$ (KD-tree) | $O(N)$ |
| Total iterations        | $O(N \log N)$ expected                | $O(N)$ |

RRT is the adventurous explorer of motion planning:
instead of mapping every inch of the world,
it sends out probing branches that reach deeper into the unknown
until one of them finds a path home.

### 786 Rapidly-Exploring Random Tree Star (RRT*)

RRT* is the optimal variant of the classic Rapidly-Exploring Random Tree (RRT).
While RRT quickly finds a valid path, it does not guarantee that the path is the *shortest*.
RRT* improves upon it by gradually refining the tree —
rewiring nearby nodes to minimize total path cost and converge toward the optimal solution over time.

#### What Problem Are We Solving?

RRT is fast and complete but *suboptimal*:
its paths can be jagged or longer than necessary.
In motion planning, optimality matters, whether for energy, safety, or aesthetics.

RRT* keeps RRT's exploratory nature but adds a rewiring step that locally improves paths.
As sampling continues, the path cost monotonically decreases and converges to the optimal path length.

#### How It Works (Plain Language)

Each iteration performs three main steps:

1. Sample and extend:
   Pick a random point `x_rand`, find the nearest node `x_nearest`,
   and steer toward it to create `x_new` (as in RRT).

2. Choose best parent:
   Find all nodes within a radius $r_n$ of `x_new`.
   Among them, pick the node that gives the lowest total cost to reach `x_new`.

3. Rewire:
   For every neighbor `x_near` within $r_n$,
   check if going through `x_new` provides a shorter path.
   If so, update `x_near`'s parent to `x_new`.

This continuous refinement makes RRT* *asymptotically optimal*:
as the number of samples grows, the solution converges to the global optimum.

#### Tiny Code (Python Example)

```python
import random, math

def distance(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

def steer(a, b, step):
    d = distance(a, b)
    if d < step:
        return b
    return (a[0] + step*(b[0]-a[0])/d, a[1] + step*(b[1]-a[1])/d)

def rrt_star(start, goal, is_free, step=1.0, max_iter=5000, radius=5.0):
    tree = {start: None}
    cost = {start: 0}
    for _ in range(max_iter):
        x_rand = (random.uniform(0,100), random.uniform(0,100))
        x_nearest = min(tree.keys(), key=lambda p: distance(p, x_rand))
        x_new = steer(x_nearest, x_rand, step)
        if not is_free(x_nearest, x_new): 
            continue
        # Find nearby nodes
        neighbors = [p for p in tree if distance(p, x_new) < radius and is_free(p, x_new)]
        # Choose best parent
        x_parent = min(neighbors + [x_nearest], key=lambda p: cost[p] + distance(p, x_new))
        tree[x_new] = x_parent
        cost[x_new] = cost[x_parent] + distance(x_parent, x_new)
        # Rewire
        for p in neighbors:
            new_cost = cost[x_new] + distance(x_new, p)
            if new_cost < cost[p] and is_free(x_new, p):
                tree[p] = x_new
                cost[p] = new_cost
        # Check for goal
        if distance(x_new, goal) < step:
            tree[goal] = x_new
            cost[goal] = cost[x_new] + distance(x_new, goal)
            return tree, cost
    return tree, cost
```

#### Why It Matters

* Asymptotically optimal: path quality improves as samples increase.
* Retains probabilistic completeness like RRT.
* Produces smooth, efficient, and safe trajectories.
* Used in:

  * Autonomous vehicle path planning
  * UAV navigation
  * Robotic manipulators
  * High-dimensional configuration spaces

#### A Gentle Proof (Why It Works)

Let $c^*$ be the optimal path cost between start and goal.
RRT* ensures that as the number of samples $n \to \infty$:

$$
P(\text{cost}(RRT^*) \to c^*) = 1
$$

because:

* The sampling distribution is uniform over free space.
* Each rewire locally minimizes the cost function.
* The connection radius $r_n \sim (\log n / n)^{1/d}$
  ensures with high probability that all nearby nodes can eventually connect.

Hence, the algorithm converges to the optimal solution almost surely.

#### Try It Yourself

1. Run both RRT and RRT* on the same obstacle map.
2. Visualize the difference: RRT*'s tree looks denser and smoother.
3. Observe how the total path cost decreases as iterations increase.
4. Adjust the radius parameter to balance exploration and refinement.

#### Test Cases

| Scenario         | RRT Path Length | RRT* Path Length | Improvement |
| ---------------- | --------------- | ---------------- | ----------- |
| Empty space      | 140             | 123              | 12% shorter |
| Sparse obstacles | 165             | 142              | 14% shorter |
| Maze corridor    | 210             | 198              | 6% shorter  |
| 3D environment   | 190             | 172              | 9% shorter  |

#### Complexity

| Operation               | Time                  | Space  |
| ----------------------- | --------------------- | ------ |
| Nearest neighbor search | $O(\log N)$ (KD-tree) | $O(N)$ |
| Rewiring per iteration  | $O(\log N)$ average   | $O(N)$ |
| Total iterations        | $O(N \log N)$         | $O(N)$ |

RRT* is the refined dreamer among planners —
it starts with quick guesses like its ancestor RRT,
then pauses to reconsider, rewiring its path
until every step moves not just forward, but better.

### 787 Probabilistic Roadmap (PRM)

The Probabilistic Roadmap (PRM) algorithm is a two-phase motion planning method used for multi-query pathfinding in high-dimensional continuous spaces.
Instead of exploring from a single start point like RRT, PRM samples many random points first, connects them into a graph (roadmap), and then uses standard graph search (like Dijkstra or A*) to find paths between any two configurations.

#### What Problem Are We Solving?

For robots or systems that need to perform many queries in the same environment, such as navigating between different destinations —
it is inefficient to rebuild a tree from scratch each time (like RRT).
PRM solves this by precomputing a roadmap of feasible connections through the free configuration space.
Once built, queries can be answered quickly.

#### How It Works (Plain Language)

PRM consists of two phases:

1. Learning phase (Roadmap Construction):

   * Randomly sample $N$ points (configurations) in the free space.
   * Discard points that collide with obstacles.
   * For each valid point, connect it to its $k$ nearest neighbors
     if a straight-line connection between them is collision-free.
   * Store these nodes and edges as a graph (the roadmap).

2. Query phase (Path Search):

   * Connect the start and goal points to nearby roadmap nodes.
   * Use a graph search algorithm (like Dijkstra or A*) to find the shortest path on the roadmap.

Over time, the roadmap becomes denser, increasing the likelihood of finding optimal paths.

#### Tiny Code (Python Example)

```python
import random, math, heapq

def distance(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

def is_free(a, b):
    # Placeholder collision checker (always free)
    return True

def build_prm(num_samples=100, k=5):
    points = [(random.uniform(0,100), random.uniform(0,100)) for _ in range(num_samples)]
    edges = {p: [] for p in points}
    for p in points:
        neighbors = sorted(points, key=lambda q: distance(p, q))[1:k+1]
        for q in neighbors:
            if is_free(p, q):
                edges[p].append(q)
                edges[q].append(p)
    return points, edges

def dijkstra(edges, start, goal):
    dist = {p: float('inf') for p in edges}
    prev = {p: None for p in edges}
    dist[start] = 0
    pq = [(0, start)]
    while pq:
        d, u = heapq.heappop(pq)
        if u == goal:
            break
        for v in edges[u]:
            alt = d + distance(u, v)
            if alt < dist[v]:
                dist[v] = alt
                prev[v] = u
                heapq.heappush(pq, (alt, v))
    path, u = [], goal
    while u:
        path.append(u)
        u = prev[u]
    return path[::-1]
```

#### Why It Matters

* Ideal for multi-query motion planning.
* Probabilistically complete: as the number of samples increases, the probability of finding a path (if one exists) approaches 1.
* Common in:

  * Mobile robot navigation
  * Autonomous vehicle route maps
  * High-dimensional robotic arm planning
  * Virtual environments and games

#### A Gentle Proof (Why It Works)

Let $X_{\text{free}}$ be the free space of configurations.
Uniform random sampling ensures that as the number of samples $n \to \infty$,
the set of samples becomes dense in $X_{\text{free}}$.

If the connection radius $r_n$ satisfies:

$$
r_n \ge c \left( \frac{\log n}{n} \right)^{1/d}
$$

(where $d$ is the dimension of space),
then with high probability the roadmap graph becomes connected.

Thus, any two configurations in the same free-space component
can be connected by a path through the roadmap,
making PRM *probabilistically complete*.

#### Try It Yourself

1. Build a PRM with 100 random points and connect each to 5 nearest neighbors.
2. Add circular obstacles, observe how the roadmap avoids them.
3. Query multiple start-goal pairs using the same roadmap.
4. Measure path quality as sample size increases.

#### Test Cases

| Samples | Neighbors (k) | Connectivity | Avg. Path Length | Query Time (ms) |
| ------- | ------------- | ------------ | ---------------- | --------------- |
| 50      | 3             | 80%          | 160              | 0.8             |
| 100     | 5             | 95%          | 140              | 1.2             |
| 200     | 8             | 99%          | 125              | 1.5             |
| 500     | 10            | 100%         | 118              | 2.0             |

#### Complexity

| Operation               | Time          | Space      |
| ----------------------- | ------------- | ---------- |
| Sampling                | $O(N)$        | $O(N)$     |
| Nearest neighbor search | $O(N \log N)$ | $O(N)$     |
| Path query (A*)         | $O(E \log V)$ | $O(V + E)$ |

PRM is the cartographer of motion planning —
it first surveys the terrain with scattered landmarks,
links the reachable ones into a living map,
and lets travelers chart their course swiftly through its probabilistic roads.

### 788 Visibility Graph

The Visibility Graph algorithm is a classical geometric method for shortest path planning in a 2D environment with polygonal obstacles.
It connects all pairs of points (vertices) that can "see" each other directly, meaning the straight line between them does not intersect any obstacle.
Then, it applies a shortest-path algorithm like Dijkstra or A* on this graph to find the optimal route.

#### What Problem Are We Solving?

Imagine a robot navigating a room with walls or obstacles.
We want the shortest collision-free path between a start and a goal point.
Unlike grid-based or sampling methods, the Visibility Graph gives an exact geometric path, often touching the corners of obstacles.

#### How It Works (Plain Language)

1. Collect all vertices of obstacles, plus the start and goal points.
2. For each pair of vertices $(v_i, v_j)$:

   * Draw a segment between them.
   * If the segment does not intersect any obstacle edges, they are *visible*.
   * Add an edge $(v_i, v_j)$ to the graph, weighted by Euclidean distance.
3. Run a shortest-path algorithm (Dijkstra or A*) between start and goal.
4. The resulting path follows obstacle corners where visibility changes.

This produces the optimal path (in Euclidean distance) within the polygonal world.

#### Tiny Code (Python Example)

```python
import math, itertools, heapq

def distance(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

def intersect(a, b, c, d):
    # Basic line segment intersection test
    def ccw(p, q, r):
        return (r[1]-p[1])*(q[0]-p[0]) > (q[1]-p[1])*(r[0]-p[0])
    return (ccw(a,c,d) != ccw(b,c,d)) and (ccw(a,b,c) != ccw(a,b,d))

def build_visibility_graph(points, obstacles):
    edges = {p: [] for p in points}
    for p, q in itertools.combinations(points, 2):
        if not any(intersect(p, q, o[i], o[(i+1)%len(o)]) for o in obstacles for i in range(len(o))):
            edges[p].append((q, distance(p,q)))
            edges[q].append((p, distance(p,q)))
    return edges

def dijkstra(graph, start, goal):
    dist = {v: float('inf') for v in graph}
    prev = {v: None for v in graph}
    dist[start] = 0
    pq = [(0, start)]
    while pq:
        d, u = heapq.heappop(pq)
        if u == goal: break
        for v, w in graph[u]:
            alt = d + w
            if alt < dist[v]:
                dist[v] = alt
                prev[v] = u
                heapq.heappush(pq, (alt, v))
    path = []
    u = goal
    while u is not None:
        path.append(u)
        u = prev[u]
    return path[::-1]
```

#### Why It Matters

* Produces exact shortest paths in polygonal environments.
* Relies purely on geometry, no discretization or random sampling.
* Common in:

  * Robotics (path planning around obstacles)
  * Video games (navigation meshes and pathfinding)
  * Computational geometry teaching and testing
  * Architectural layout and urban planning tools

#### A Gentle Proof (Why It Works)

If all obstacles are polygonal and motion is allowed along straight lines between visible vertices,
then any optimal path can be represented as a sequence of visible vertices (start → corner → corner → goal).

Formally, between two consecutive tangential contacts with obstacles,
the path must be a straight segment; otherwise, it can be shortened.

Thus, the shortest obstacle-avoiding path exists within the visibility graph's edges.

#### Try It Yourself

1. Create a map with polygonal obstacles (rectangles, triangles, etc.).
2. Plot the visibility graph, edges connecting visible vertices.
3. Observe that the shortest path "hugs" obstacle corners.
4. Compare the result with grid-based A*, you'll see how geometric methods give exact minimal paths.

#### Test Cases

| Scenario             | Obstacles | Vertices | Path Type              | Result  |
| -------------------- | --------- | -------- | ---------------------- | ------- |
| Empty plane          | 0         | 2        | Straight line          | Optimal |
| One rectangle        | 4         | 6        | Tangential corner path | Optimal |
| Maze walls           | 12        | 20       | Multi-corner path      | Optimal |
| Triangular obstacles | 9         | 15       | Mixed edges            | Optimal |

#### Complexity

| Operation                | Time       | Space    |
| ------------------------ | ---------- | -------- |
| Edge visibility checks   | $O(V^2 E)$ | $O(V^2)$ |
| Shortest path (Dijkstra) | $O(V^2)$   | $O(V)$   |

Here $V$ is the number of vertices and $E$ the number of obstacle edges.

The Visibility Graph is the geometric purist of motion planners —
it trusts straight lines and clear sight,
tracing paths that just graze the edges of obstacles,
as if geometry itself whispered the way forward.

### 789 Potential Field Pathfinding

Potential Field Pathfinding treats navigation as a problem of physics.
The robot moves under the influence of an artificial potential field:
the goal attracts it like gravity, and obstacles repel it like electric charges.
This approach transforms planning into a continuous optimization problem where motion naturally flows downhill in potential energy.

#### What Problem Are We Solving?

Pathfinding in cluttered spaces can be tricky.
Classical algorithms like A* work on discrete grids, but many real environments are continuous.
Potential fields provide a smooth, real-valued framework for navigation, intuitive, lightweight, and reactive.

The challenge? Avoiding local minima, where the robot gets stuck in a valley of forces before reaching the goal.

#### How It Works (Plain Language)

1. Define a potential function over space:

   * Attractive potential toward the goal:
     $$
     U_{att}(x) = \frac{1}{2} k_{att} , |x - x_{goal}|^2
     $$

   * Repulsive potential away from obstacles:
$$
U_{rep}(x) =
\begin{cases}
\frac{1}{2} k_{rep} \left(\frac{1}{d(x)} - \frac{1}{d_0}\right)^2, & d(x) < d_0 \\
0, & d(x) \ge d_0
\end{cases}
$$

     where $d(x)$ is the distance to the nearest obstacle and $d_0$ is the influence radius.

2. Compute the resultant force (negative gradient of potential):
   $$
   F(x) = -\nabla U(x) = F_{att}(x) + F_{rep}(x)
   $$

3. Move the robot a small step in the direction of the force until it reaches the goal (or gets trapped).

#### Tiny Code (Python Example)

```python
import numpy as np

def attractive_force(pos, goal, k_att=1.0):
    return -k_att * (pos - goal)

def repulsive_force(pos, obstacles, k_rep=100.0, d0=5.0):
    total = np.zeros(2)
    for obs in obstacles:
        diff = pos - obs
        d = np.linalg.norm(diff)
        if d < d0 and d > 1e-6:
            total += k_rep * ((1/d - 1/d0) / d3) * diff
    return total

def potential_field_path(start, goal, obstacles, step=0.5, max_iter=1000):
    pos = np.array(start, dtype=float)
    goal = np.array(goal, dtype=float)
    path = [tuple(pos)]
    for _ in range(max_iter):
        F_att = attractive_force(pos, goal)
        F_rep = repulsive_force(pos, obstacles)
        F = F_att + F_rep
        pos += step * F / np.linalg.norm(F)
        path.append(tuple(pos))
        if np.linalg.norm(pos - goal) < 1.0:
            break
    return path
```

#### Why It Matters

* Continuous-space pathfinding: works directly in $\mathbb{R}^2$ or $\mathbb{R}^3$.
* Computationally light: no grid or graph construction.
* Reactive: adapts to changes in obstacles dynamically.
* Used in:

  * Autonomous drones and robots
  * Crowd simulation
  * Local motion control systems

#### A Gentle Proof (Why It Works)

The total potential function
$$
U(x) = U_{att}(x) + U_{rep}(x)
$$
is differentiable except at obstacle boundaries.
At any point, the direction of steepest descent $-\nabla U(x)$
points toward the nearest minimum of $U(x)$.
If $U$ is convex (no local minima besides the goal), the gradient descent path
converges to the goal configuration.

However, in nonconvex environments, multiple minima may exist.
Hybrid methods (like adding random perturbations or combining with A*) can escape these traps.

#### Try It Yourself

1. Define a 2D map with circular obstacles.
2. Visualize the potential field as a heatmap.
3. Trace how the path slides smoothly toward the goal.
4. Introduce a narrow passage, observe how tuning $k_{rep}$ affects avoidance.
5. Combine with A* for global + local planning.

#### Test Cases

| Environment   | Obstacles | Behavior                     | Result          |
| ------------- | --------- | ---------------------------- | --------------- |
| Empty space   | 0         | Direct path                  | Reaches goal    |
| One obstacle  | 1         | Smooth curve around obstacle | Success         |
| Two obstacles | 2         | Avoids both                  | Success         |
| Narrow gap    | 2 close   | Local minimum possible       | Partial success |

#### Complexity

| Operation                  | Time               | Space  |
| -------------------------- | ------------------ | ------ |
| Force computation per step | $O(N_{obstacles})$ | $O(1)$ |
| Total iterations           | $O(T)$             | $O(T)$ |

where $T$ is the number of movement steps.

Potential Field Pathfinding is like navigating by invisible gravity —
every point in space whispers a direction,
the goal pulls gently, the walls push firmly,
and the traveler learns the shape of the world through motion itself.

### 790 Bug Algorithms

Bug Algorithms are a family of simple reactive pathfinding methods for mobile robots that use only local sensing, no maps, no global planning, just a feel for where the goal lies and whether an obstacle is blocking the way.
They're ideal for minimalist robots or real-world navigation where uncertainty is high.

#### What Problem Are We Solving?

When a robot moves toward a goal but encounters obstacles it didn't anticipate, it needs a way to recover without a global map.
Traditional planners like A* or RRT assume full knowledge of the environment.
Bug algorithms, by contrast, make decisions on the fly, using only what the robot can sense.

#### How It Works (Plain Language)

All Bug algorithms share two phases:

1. Move toward the goal in a straight line until hitting an obstacle.
2. Follow the obstacle's boundary until a better route to the goal becomes available.

Different versions define "better route" differently:

| Variant        | Strategy                                                                   |
| -------------- | -------------------------------------------------------------------------- |
| Bug1       | Trace the entire obstacle, find the closest point to the goal, then leave. |
| Bug2       | Follow the obstacle until the line to the goal is clear again.             |
| TangentBug | Use range sensors to estimate visibility and switch paths intelligently.   |

#### Example: Bug2 Algorithm

1. Start at $S$, move toward goal $G$ along the line $SG$.
2. If an obstacle is hit, follow its boundary while measuring distance to $G$.
3. When the direct line to $G$ becomes visible again, leave the obstacle and continue.
4. Stop when $G$ is reached or no progress can be made.

This uses *only* local sensing and position awareness relative to the goal.

#### Tiny Code (Python Example)

```python
import math

def distance(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

def bug2(start, goal, obstacles, step=1.0, max_iter=1000):
    pos = list(start)
    path = [tuple(pos)]
    for _ in range(max_iter):
        # Direct motion toward goal
        dir_vec = [goal[0]-pos[0], goal[1]-pos[1]]
        dist = math.hypot(*dir_vec)
        if dist < 1.0:
            path.append(tuple(goal))
            break
        dir_vec = [dir_vec[0]/dist, dir_vec[1]/dist]
        next_pos = [pos[0]+step*dir_vec[0], pos[1]+step*dir_vec[1]]
        # Simple obstacle check
        hit = any(distance(next_pos, o) < 2.0 for o in obstacles)
        if hit:
            # Follow boundary (simplified)
            next_pos[0] += step * dir_vec[1]
            next_pos[1] -= step * dir_vec[0]
        pos = next_pos
        path.append(tuple(pos))
    return path
```

#### Why It Matters

* Requires only local sensing, no precomputed map.
* Works in unknown or dynamic environments.
* Computationally cheap and robust to sensor noise.
* Commonly used in:

  * Low-cost autonomous robots
  * Simple drones or rovers
  * Embedded microcontroller systems

#### A Gentle Proof (Why It Works)

Bug algorithms guarantee goal reachability if:

* The robot can detect when it reaches the goal, and
* The environment is bounded with finite obstacles.

Because each boundary is followed deterministically and revisited positions are avoided,
the path length is bounded and the robot will either:

1. Reach the goal, or
2. Prove that no path exists (after exploring all obstacles).

In formal terms, Bug2 achieves *completeness* under sensor constraints.

#### Try It Yourself

1. Place a circular obstacle between start and goal.
2. Simulate Bug2, watch the robot hit the obstacle, trace its edge, then resume toward the goal.
3. Add more obstacles, note how path complexity grows.
4. Compare with A* or RRT, Bug2 paths are longer but computed instantly.

#### Test Cases

| Environment        | Obstacles   | Result               | Path Type       |
| ------------------ | ----------- | -------------------- | --------------- |
| Empty space        | 0           | Straight line        | Direct          |
| Single obstacle    | 1           | Wraps around         | Success         |
| Multiple obstacles | 3           | Sequential avoidance | Success         |
| Enclosed goal      | 1 enclosing | No path              | Detects failure |

#### Complexity

| Operation                | Time   | Space  |
| ------------------------ | ------ | ------ |
| Local sensing and update | $O(1)$ | $O(1)$ |
| Total path traversal     | $O(L)$ | $O(1)$ |

where $L$ is the total obstacle boundary length encountered.

Bug algorithms are the wanderers of robotics —
they don't see the whole map, only what lies before them,
yet through patience and persistence, they find their way home.

# Section 80. Computational Geometry Variants and Applications 

### 791 Convex Polygon Intersection

The Convex Polygon Intersection algorithm computes the region formed by the overlap of two convex polygons.
Since convex polygons have no internal concavities, the intersection itself is also convex and can be efficiently found by geometric clipping or incremental edge traversal.

#### What Problem Are We Solving?

Given two convex polygons ( P ) and ( Q ), we want to find their intersection polygon ( R = P \cap Q ).
This is fundamental in computational geometry, computer graphics (clipping), and collision detection.

Convexity guarantees that:

* Every line segment between two points inside a polygon remains inside it.
* Intersection can be computed in linear time with respect to the number of edges.

#### How It Works (Plain Language)

There are two common approaches:

1. Half-plane Intersection (Sutherland–Hodgman):
Clip one polygon against each edge of the other.

* Start with all vertices of ( P ).
* For each edge of ( Q ), keep only points inside that half-plane.
* The result after all edges is ( P \cap Q ).

2. Edge Traversal (Divide and Walk):
Walk around both polygons simultaneously, advancing edges by comparing angles,
and collect intersection and inclusion points.

Both rely on convexity: at most two intersections per edge pair, and edges stay ordered by angle.

#### Mathematical Core

For each directed edge of polygon ( Q ), represented as ( (q_i, q_{i+1}) ),
define a half-plane:
$$
H_i = { x \in \mathbb{R}^2 : (q_{i+1} - q_i) \times (x - q_i) \ge 0 }
$$

Then, the intersection polygon is:
$$
P \cap Q = P \cap \bigcap_i H_i
$$

Each clipping step reduces ( P ) by cutting away parts outside the current half-plane.

#### Tiny Code (Python Example)

```python
def cross(o, a, b):
    return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])

def intersect(p1, p2, q1, q2):
    A1, B1 = p2[1]-p1[1], p1[0]-p2[0]
    C1 = A1*p1[0] + B1*p1[1]
    A2, B2 = q2[1]-q1[1], q1[0]-q2[0]
    C2 = A2*q1[0] + B2*q1[1]
    det = A1*B2 - A2*B1
    if abs(det) < 1e-9:
        return None
    return ((B2*C1 - B1*C2)/det, (A1*C2 - A2*C1)/det)

def clip_polygon(poly, edge_start, edge_end):
    new_poly = []
    for i in range(len(poly)):
        curr, nxt = poly[i], poly[(i+1)%len(poly)]
        inside_curr = cross(edge_start, edge_end, curr) >= 0
        inside_next = cross(edge_start, edge_end, nxt) >= 0
        if inside_curr and inside_next:
            new_poly.append(nxt)
        elif inside_curr and not inside_next:
            new_poly.append(intersect(curr, nxt, edge_start, edge_end))
        elif not inside_curr and inside_next:
            new_poly.append(intersect(curr, nxt, edge_start, edge_end))
            new_poly.append(nxt)
    return [p for p in new_poly if p]

def convex_intersection(P, Q):
    result = P
    for i in range(len(Q)):
        result = clip_polygon(result, Q[i], Q[(i+1)%len(Q)])
        if not result:
            break
    return result
```

#### Why It Matters

* Core operation in polygon clipping (used in rendering pipelines).
* Basis for collision detection between convex objects.
* Applied in computational geometry, GIS, and physics engines.
* Serves as a building block for more complex geometric algorithms (e.g., Minkowski sums, SAT).

#### A Gentle Proof (Why It Works)

Each edge of polygon ( Q ) defines a linear inequality describing its interior.
Intersecting ( P ) with one half-plane maintains convexity.
Successively applying all constraints from ( Q ) preserves both convexity and boundedness.

Since each clipping step removes vertices linearly,
the total complexity is ( O(n + m) ) for polygons with ( n ) and ( m ) vertices.

#### Try It Yourself

1. Create two convex polygons ( P ) and ( Q ).
2. Use the clipping code to compute ( P \cap Q ).
3. Visualize them, the resulting shape is always convex.
4. Experiment with disjoint, tangent, and fully-contained configurations.

#### Test Cases

| Polygon P             | Polygon Q     | Intersection Type      |
| --------------------- | ------------- | ---------------------- |
| Overlapping triangles | Quadrilateral | Convex quadrilateral   |
| Square inside square  | Offset        | Smaller convex polygon |
| Disjoint              | Far apart     | Empty                  |
| Touching edge         | Adjacent      | Line segment           |

#### Complexity

| Operation        | Time            | Space  |
| ---------------- | --------------- | ------ |
| Clipping         | $O(n + m)$      | $O(n)$ |
| Half-plane tests | $O(n)$ per edge | $O(1)$ |

Convex polygon intersection is the architect of geometric overlap —
cutting shapes not by brute force, but by logic,
tracing the quiet frontier where two convex worlds meet and share common ground.

### 792 Minkowski Sum

The Minkowski Sum is a fundamental geometric operation that combines two sets of points by vector addition.
In computational geometry, it is often used to model shape expansion, collision detection, and path planning, for example, growing one object by the shape of another.

#### What Problem Are We Solving?

Suppose we have two convex shapes, ( A ) and ( B ).
We want a new shape $A \oplus B$ that represents all possible sums of one point from each shape.

Formally, this captures how much space one object would occupy if it "slides around" another —
a key idea in motion planning and collision geometry.

#### How It Works (Plain Language)

Given two sets $A, B \subset \mathbb{R}^2$:
$$
A \oplus B = { a + b \mid a \in A, b \in B }
$$

In other words, take every point in ( A ) and translate it by every point in ( B ),
then take the union of all those translations.

When ( A ) and ( B ) are convex polygons, the Minkowski sum is also convex.
Its boundary can be constructed efficiently by merging edges in order of their angles.

#### Geometric Intuition

* Adding a circle to a polygon "rounds" its corners (used in configuration space expansion).
* Adding a robot shape to obstacles effectively grows obstacles by the robot's size —
  reducing path planning to a point navigation problem in the expanded space.

#### Mathematical Form

If ( A ) and ( B ) are convex polygons with vertices
$A = (a_1, \dots, a_n)$ and $B = (b_1, \dots, b_m)$,
and both listed in counterclockwise order,
then the Minkowski sum polygon can be computed by edge-wise merging:

$$
A \oplus B = \text{conv}{ a_i + b_j }
$$

#### Tiny Code (Python Example)

```python
import math

def cross(a, b):
    return a[0]*b[1] - a[1]*b[0]

def minkowski_sum(A, B):
    # assume convex, CCW ordered
    i, j = 0, 0
    n, m = len(A), len(B)
    result = []
    while i < n or j < m:
        result.append((A[i % n][0] + B[j % m][0],
                       A[i % n][1] + B[j % m][1]))
        crossA = cross((A[(i+1)%n][0]-A[i%n][0], A[(i+1)%n][1]-A[i%n][1]),
                       (B[(j+1)%m][0]-B[j%m][0], B[(j+1)%m][1]-B[j%m][1]))
        if crossA >= 0:
            i += 1
        if crossA <= 0:
            j += 1
    return result
```

#### Why It Matters

* Collision detection:
  Two convex shapes ( A ) and ( B ) intersect if and only if
  $(A \oplus (-B))$ contains the origin.

* Motion planning:
  Expanding obstacles by the robot's shape simplifies pathfinding.

* Computational geometry:
  Used to build configuration spaces and approximate complex shape interactions.

#### A Gentle Proof (Why It Works)

For convex polygons, the Minkowski sum can be obtained by adding their support functions:
$$
h_{A \oplus B}(u) = h_A(u) + h_B(u)
$$
where $h_S(u) = \max_{x \in S} u \cdot x$ gives the farthest extent of shape ( S ) along direction ( u ).
The boundary of $A \oplus B$ is formed by combining the edges of ( A ) and ( B ) in ascending angular order,
preserving convexity.

This yields an $O(n + m)$ construction algorithm.

#### Try It Yourself

1. Start with two convex polygons (e.g., triangle and square).
2. Compute their Minkowski sum, the result should "blend" their shapes.
3. Add a small circle shape to see how corners become rounded.
4. Visualize how this process enlarges one shape by another's geometry.

#### Test Cases

| Shape A          | Shape B          | Resulting Shape   | Notes                  |
| ---------------- | ---------------- | ----------------- | ---------------------- |
| Triangle         | Square           | Hexagonal shape   | Convex                 |
| Rectangle        | Circle           | Rounded rectangle | Used in robot planning |
| Two squares      | Same orientation | Larger square     | Scaled up              |
| Irregular convex | Small polygon    | Smoothed edges    | Convex preserved       |

#### Complexity

| Operation           | Time                  | Space      |
| ------------------- | --------------------- | ---------- |
| Edge merging        | $O(n + m)$            | $O(n + m)$ |
| Convex hull cleanup | $O((n + m)\log(n+m))$ | $O(n + m)$ |

The Minkowski Sum is geometry's combinatorial melody —
every point in one shape sings in harmony with every point in another,
producing a new, unified figure that reveals how objects truly meet in space.

### 793 Rotating Calipers

The Rotating Calipers technique is a geometric method used to solve a variety of convex polygon problems efficiently.
It gets its name from the mental image of a pair of calipers rotating around a convex shape, always touching it at two parallel supporting lines.

This method allows for elegant linear-time computation of geometric quantities like width, diameter, minimum bounding box, or farthest point pairs.

#### What Problem Are We Solving?

Given a convex polygon, we often need to compute geometric measures such as:

* The diameter (largest distance between two vertices).
* The width (minimum distance between two parallel lines enclosing the polygon).
* The smallest enclosing rectangle (minimum-area bounding box).

A naive approach would check all pairs of points, $O(n^2)$ work.
Rotating calipers do it in linear time, leveraging convexity and geometry.

#### How It Works (Plain Language)

1. Start with the convex polygon vertices in counterclockwise order.
2. Identify an initial pair of antipodal points, points lying on parallel supporting lines.
3. Rotate a pair of calipers around the polygon's edges, maintaining contact at antipodal vertices.
4. For each edge direction, compute the relevant measurement (distance, width, etc.).
5. Record the minimum or maximum value as needed.

Because each edge and vertex is visited at most once, total time is ( O(n) ).

#### Example: Finding the Diameter of a Convex Polygon

The diameter is the longest distance between any two points on the convex hull.

1. Compute the convex hull of the points (if not already convex).
2. Initialize two pointers at antipodal points.
3. For each vertex ( i ), move the opposite vertex ( j ) while
   the area (or cross product) increases:
   $$
   |(P_{i+1} - P_i) \times (P_{j+1} - P_i)| > |(P_{i+1} - P_i) \times (P_j - P_i)|
   $$
4. Record the maximum distance $d = | P_i - P_j |$.

#### Tiny Code (Python Example)

```python
import math

def distance(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

def rotating_calipers(points):
    # points: list of convex hull vertices in CCW order
    n = len(points)
    if n < 2:
        return 0
    max_dist = 0
    j = 1
    for i in range(n):
        next_i = (i + 1) % n
        while abs((points[next_i][0]-points[i][0]) * 
                  (points[(j+1)%n][1]-points[i][1]) - 
                  (points[next_i][1]-points[i][1]) * 
                  (points[(j+1)%n][0]-points[i][0])) > abs(
                  (points[next_i][0]-points[i][0]) * 
                  (points[j][1]-points[i][1]) - 
                  (points[next_i][1]-points[i][1]) * 
                  (points[j][0]-points[i][0])):
            j = (j + 1) % n
        max_dist = max(max_dist, distance(points[i], points[j]))
    return max_dist
```

#### Why It Matters

* Efficient: Only ( O(n) ) time for problems that naïvely take $O(n^2)$.
* Versatile: Works for multiple geometry tasks, distance, width, bounding boxes.
* Geometrically intuitive: Mimics physical measurement around shapes.
* Used in:

  * Collision detection and bounding boxes
  * Shape analysis and convex geometry
  * Robotics and computational geometry education

#### A Gentle Proof (Why It Works)

For a convex polygon, every direction of rotation corresponds to a unique pair of support lines.
Each line contacts one vertex or edge of the polygon.
As the polygon rotates by 180°, each vertex becomes a support point exactly once.

Thus, the total number of steps equals the number of vertices,
and the maximum distance or minimum width must occur at one of these antipodal pairs.

This is a direct geometric consequence of convexity and the support function
$h_P(u) = \max_{x \in P} (u \cdot x)$.

#### Try It Yourself

1. Generate a convex polygon (e.g., a hexagon).
2. Apply rotating calipers to compute:

   * Maximum distance (diameter).
   * Minimum distance between parallel sides (width).
   * Smallest bounding rectangle area.
3. Visualize the calipers rotating, they always stay tangent to opposite sides.

#### Test Cases

| Polygon   | Vertices | Quantity     | Result           |
| --------- | -------- | ------------ | ---------------- |
| Square    | 4        | Diameter     | √2 × side length |
| Rectangle | 4        | Width        | Shorter side     |
| Triangle  | 3        | Diameter     | Longest edge     |
| Hexagon   | 6        | Bounding box | Matches symmetry |

#### Complexity

| Operation                 | Time          | Space  |
| ------------------------- | ------------- | ------ |
| Edge traversal            | $O(n)$        | $O(1)$ |
| Convex hull preprocessing | $O(n \log n)$ | $O(n)$ |

The Rotating Calipers technique is geometry's compass in motion —
gliding gracefully around convex shapes,
measuring distances and widths in perfect rotational harmony.

### 794 Half-Plane Intersection

The Half-Plane Intersection algorithm finds the common region that satisfies a collection of linear inequalities, each representing a half-plane in the plane.
This is a core geometric operation for computational geometry, linear programming, and visibility computations, defining convex regions efficiently.

#### What Problem Are We Solving?

Given a set of lines in the plane, each defining a half-plane (the region on one side of a line),
find the intersection polygon of all those half-planes.

Each half-plane can be written as a linear inequality:
$$
a_i x + b_i y + c_i \le 0
$$
The intersection of these regions forms a convex polygon (possibly empty or unbounded).

Applications include:

* Linear feasibility regions
* Visibility polygons
* Clipping convex shapes
* Solving small 2D linear programs geometrically

#### How It Works (Plain Language)

1. Represent each half-plane by its boundary line and a direction (the "inside").
2. Sort all half-planes by the angle of their boundary line.
3. Process them one by one, maintaining the current intersection polygon (or deque).
4. Whenever adding a new half-plane, clip the current polygon by that half-plane.
5. The result after processing all half-planes is the intersection region.

The convexity of half-planes guarantees that their intersection is convex.

#### Mathematical Form

A half-plane is defined by the inequality:
$$
a_i x + b_i y + c_i \le 0
$$

The intersection region is:
$$
R = \bigcap_{i=1}^n { (x, y) : a_i x + b_i y + c_i \le 0 }
$$

Each boundary line divides the plane into two parts;
we iteratively eliminate the "outside" portion.

#### Tiny Code (Python Example)

```python
import math

EPS = 1e-9

def intersect(L1, L2):
    a1, b1, c1 = L1
    a2, b2, c2 = L2
    det = a1*b2 - a2*b1
    if abs(det) < EPS:
        return None
    x = (b1*c2 - b2*c1) / det
    y = (c1*a2 - c2*a1) / det
    return (x, y)

def inside(point, line):
    a, b, c = line
    return a*point[0] + b*point[1] + c <= EPS

def clip_polygon(poly, line):
    result = []
    n = len(poly)
    for i in range(n):
        curr, nxt = poly[i], poly[(i+1)%n]
        inside_curr = inside(curr, line)
        inside_next = inside(nxt, line)
        if inside_curr and inside_next:
            result.append(nxt)
        elif inside_curr and not inside_next:
            result.append(intersect((nxt[0]-curr[0], nxt[1]-curr[1], 0), line))
        elif not inside_curr and inside_next:
            result.append(intersect((nxt[0]-curr[0], nxt[1]-curr[1], 0), line))
            result.append(nxt)
    return [p for p in result if p]

def half_plane_intersection(lines, bound_box=10000):
    # Start with a large square region
    poly = [(-bound_box,-bound_box), (bound_box,-bound_box),
            (bound_box,bound_box), (-bound_box,bound_box)]
    for line in lines:
        poly = clip_polygon(poly, line)
        if not poly:
            break
    return poly
```

#### Why It Matters

* Computational geometry core: underlies convex clipping and linear feasibility.
* Linear programming visualization: geometric version of simplex.
* Graphics and vision: used in clipping, shadow casting, and visibility.
* Path planning and robotics: defines safe navigation zones.

#### A Gentle Proof (Why It Works)

Each half-plane corresponds to a linear constraint in $\mathbb{R}^2$.
The intersection of convex sets is convex, so the result must also be convex.

The iterative clipping procedure successively applies intersections:
$$
P_{k+1} = P_k \cap H_{k+1}
$$
At every step, the polygon remains convex and shrinks monotonically (or becomes empty).

The final polygon $P_n$ satisfies all constraints simultaneously.

#### Try It Yourself

1. Represent constraints like:

   * $x \ge 0$
   * $y \ge 0$
   * $x + y \le 5$
2. Convert them to line coefficients and pass to `half_plane_intersection()`.
3. Plot the resulting polygon, it will be the triangle bounded by those inequalities.

Try adding or removing constraints to see how the feasible region changes.

#### Test Cases

| Constraints                            | Resulting Shape       | Notes                  |
| -------------------------------------- | --------------------- | ---------------------- |
| 3 inequalities forming a triangle      | Finite convex polygon | Feasible               |
| Parallel constraints facing each other | Infinite strip        | Unbounded              |
| Inconsistent inequalities              | Empty set             | No intersection        |
| Rectangle constraints                  | Square                | Simple bounded polygon |

#### Complexity

| Operation          | Time          | Space  |
| ------------------ | ------------- | ------ |
| Polygon clipping   | $O(n \log n)$ | $O(n)$ |
| Incremental update | $O(n)$        | $O(n)$ |

Half-Plane Intersection is geometry's language of constraints —
each line a rule, each half-plane a promise,
and their intersection, the elegant shape of all that is possible.

### 795 Line Arrangement

A Line Arrangement is the subdivision of the plane formed by a set of lines.
It is one of the most fundamental constructions in computational geometry, used to study the combinatorial complexity of planar structures and to build algorithms for point location, visibility, and geometric optimization.

#### What Problem Are We Solving?

Given ( n ) lines in the plane,
we want to find how they divide the plane into regions, called faces, along with their edges and vertices.

For example:

* 2 lines divide the plane into 4 regions.
* 3 lines (no parallels, no 3 lines meeting at one point) divide the plane into 7 regions.
* In general, ( n ) lines divide the plane into
  $$
  \frac{n(n+1)}{2} + 1
  $$
  regions at most.

Applications include:

* Computing intersections and visibility maps
* Motion planning and path decomposition
* Constructing trapezoidal maps for point location
* Studying combinatorial geometry and duality

#### How It Works (Plain Language)

A line arrangement is constructed incrementally:

1. Start with an empty plane (1 region).
2. Add one line at a time.
3. Each new line intersects all previous lines, splitting some regions into two.

If the lines are in general position (no parallels, no 3 concurrent lines),
the number of new regions formed by the ( k )-th line is ( k ).

Hence, the total number of regions after ( n ) lines is:
$$
R(n) = 1 + \sum_{k=1}^{n} k = 1 + \frac{n(n+1)}{2}
$$

#### Geometric Structure

Each arrangement divides the plane into:

* Vertices (intersection points of lines)
* Edges (line segments between intersections)
* Faces (regions bounded by edges)

The total numbers satisfy Euler's planar formula:
$$
V - E + F = 1 + C
$$
where ( C ) is the number of connected components (for lines, ( C = 1 )).

#### Tiny Code (Python Example)

This snippet constructs intersections and counts faces for small inputs.

```python
import itertools

def intersect(l1, l2):
    (a1,b1,c1), (a2,b2,c2) = l1, l2
    det = a1*b2 - a2*b1
    if abs(det) < 1e-9:
        return None
    x = (b1*c2 - b2*c1) / det
    y = (c1*a2 - c2*a1) / det
    return (x, y)

def line_arrangement(lines):
    points = []
    for (l1, l2) in itertools.combinations(lines, 2):
        p = intersect(l1, l2)
        if p:
            points.append(p)
    return len(points), len(lines), 1 + len(points) + len(lines)
```

Example:

```python
lines = [(1, -1, 0), (0, 1, -1), (1, 1, -2)]
print(line_arrangement(lines))
```

#### Why It Matters

* Combinatorial geometry: helps bound the complexity of geometric structures.
* Point location: foundation for efficient spatial queries.
* Motion planning: subdivides space into navigable regions.
* Algorithm design: leads to data structures like the trapezoidal map and arrangements in dual space.

#### A Gentle Proof (Why It Works)

When adding the ( k )-th line:

* It can intersect all previous ( k - 1 ) lines in distinct points.
* These intersections divide the new line into ( k ) segments.
* Each segment cuts through one region, creating exactly ( k ) new regions.

Thus:
$$
R(n) = R(n-1) + n
$$
with ( R(0) = 1 ).
By summation:
$$
R(n) = 1 + \frac{n(n+1)}{2}
$$
This argument relies only on general position, no parallel or coincident lines.

#### Try It Yourself

1. Draw 1, 2, 3, and 4 lines in general position.
2. Count regions, you'll get 2, 4, 7, 11.
3. Verify the recurrence ( R(n) = R(n-1) + n ).
4. Try making lines parallel or concurrent, the count will drop.

#### Test Cases

| Lines (n) | Max Regions | Notes                        |
| --------- | ----------- | ---------------------------- |
| 1         | 2           | Divides plane in half        |
| 2         | 4           | Crossed lines                |
| 3         | 7           | No parallels, no concurrency |
| 4         | 11          | Adds 4 new regions           |
| 5         | 16          | Continues quadratic growth   |

#### Complexity

| Operation                | Time     | Space    |
| ------------------------ | -------- | -------- |
| Intersection computation | $O(n^2)$ | $O(n^2)$ |
| Incremental arrangement  | $O(n^2)$ | $O(n^2)$ |

The Line Arrangement is geometry's combinatorial playground —
each new line adds complexity, intersections, and order,
turning a simple plane into a lattice of relationships and regions.

### 796 Point Location (Trapezoidal Map)

The Point Location problem asks: given a planar subdivision (for example, a collection of non-intersecting line segments that divide the plane into regions), determine which region contains a given point.
The Trapezoidal Map method solves this efficiently using geometry and randomization.

#### What Problem Are We Solving?

Given a set of non-intersecting line segments, preprocess them so we can answer queries of the form:

> For a point $(x, y)$, which face (region) of the subdivision contains it?

Applications include:

* Finding where a point lies in a planar map or mesh
* Ray tracing and visibility problems
* Geographic Information Systems (GIS)
* Computational geometry algorithms using planar subdivisions

#### How It Works (Plain Language)

1. Build a trapezoidal decomposition:
   Extend a vertical line upward and downward from each endpoint until it hits another segment or infinity.
   These lines partition the plane into trapezoids (possibly unbounded).

2. Build a search structure (DAG):
   Store the trapezoids and their adjacency in a directed acyclic graph.
   Each internal node represents a test (is the point to the left/right of a segment or above/below a vertex?).
   Each leaf corresponds to one trapezoid.

3. Query:
   To locate a point, traverse the DAG using the geometric tests until reaching a leaf, that leaf's trapezoid contains the point.

This structure allows $O(\log n)$ expected query time after $O(n \log n)$ expected preprocessing.

#### Mathematical Sketch

For each segment set $S$:

* Build vertical extensions at endpoints → set of vertical slabs.
* Each trapezoid bounded by at most four edges:

  * top and bottom by input segments
  * left and right by vertical lines

The total number of trapezoids is linear in $n$.

#### Tiny Code (Python Example, Simplified)

Below is a conceptual skeleton; real implementations use geometric libraries.

```python
import bisect

class TrapezoidMap:
    def __init__(self, segments):
        self.segments = sorted(segments, key=lambda s: min(s[0][0], s[1][0]))
        self.x_coords = sorted({x for seg in segments for (x, _) in seg})
        self.trapezoids = self._build_trapezoids()

    def _build_trapezoids(self):
        traps = []
        for i in range(len(self.x_coords)-1):
            x1, x2 = self.x_coords[i], self.x_coords[i+1]
            traps.append(((x1, x2), None))
        return traps

    def locate_point(self, x):
        i = bisect.bisect_right(self.x_coords, x) - 1
        return self.trapezoids[max(0, min(i, len(self.trapezoids)-1))]
```

This toy version partitions the x-axis into trapezoids; real versions include y-bounds and adjacency.

#### Why It Matters

* Fast queries: expected $O(\log n)$ point-location.
* Scalable structure: linear space in the number of segments.
* Broad utility: building block for Voronoi diagrams, visibility, and polygon clipping.
* Elegant randomization: randomized incremental construction keeps it simple and robust.

#### A Gentle Proof (Why It Works)

In the randomized incremental construction:

1. Each new segment interacts with only $O(1)$ trapezoids in expectation.
2. The structure maintains expected $O(n)$ trapezoids and $O(n)$ nodes in the DAG.
3. Searching requires only $O(\log n)$ decisions on average.

Thus, the expected performance bounds are:
$$
\text{Preprocessing: } O(n \log n), \quad \text{Query: } O(\log n)
$$

#### Try It Yourself

1. Draw a few line segments without intersections.
2. Extend vertical lines from endpoints to form trapezoids.
3. Pick random points and trace which trapezoid they fall in.
4. Observe how queries become simple comparisons of coordinates.

#### Test Cases

| Input Segments         | Query Point | Output Region       |
| ---------------------- | ----------- | ------------------- |
| Horizontal line y = 1  | (0, 0)      | Below segment       |
| Two crossing diagonals | (1, 1)      | Intersection region |
| Polygon edges          | (2, 3)      | Inside polygon      |
| Empty set              | (x, y)      | Unbounded region    |

#### Complexity

| Operation       | Expected Time | Space  |
| --------------- | ------------- | ------ |
| Build structure | $O(n \log n)$ | $O(n)$ |
| Point query     | $O(\log n)$   | $O(1)$ |

The Trapezoidal Map turns geometry into logic —
each segment defines a rule,
each trapezoid a case,
and every query finds its home through elegant spatial reasoning.

### 797 Voronoi Nearest Facility

The Voronoi Nearest Facility algorithm assigns every point in the plane to its nearest facility among a given set of sites.
The resulting structure, called a Voronoi diagram, partitions space into cells, each representing the region of points closest to a specific facility.

#### What Problem Are We Solving?

Given a set of $n$ facilities (points) $S = {p_1, p_2, \dots, p_n}$, and a query point $q$,
we want to find the facility $p_i$ minimizing the distance:
$$
d(q, p_i) = \min_{1 \le i \le n} \sqrt{(x_q - x_i)^2 + (y_q - y_i)^2}
$$

The Voronoi region of a facility $p_i$ is the set of all points closer to $p_i$ than to any other facility:
$$
V(p_i) = { q \in \mathbb{R}^2 \mid d(q, p_i) \le d(q, p_j), , \forall j \ne i }
$$

#### How It Works (Plain Language)

1. Compute the Voronoi diagram for the given facilities, a planar partition of the space.
2. Each cell corresponds to one facility and contains all points for which that facility is the nearest.
3. To answer a nearest-facility query:

   * Locate which cell the query point lies in.
   * The cell's generator point is the nearest facility.

Efficient data structures allow $O(\log n)$ query time after $O(n \log n)$ preprocessing.

#### Mathematical Geometry

The boundary between two facilities $p_i$ and $p_j$ is the perpendicular bisector of the segment joining them:
$$
(x - x_i)^2 + (y - y_i)^2 = (x - x_j)^2 + (y - y_j)^2
$$
Simplifying gives:
$$
2(x_j - x_i)x + 2(y_j - y_i)y = (x_j^2 + y_j^2) - (x_i^2 + y_i^2)
$$

Each pair contributes a bisector line, and their intersections define Voronoi vertices.

#### Tiny Code (Python Example)

```python
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt

points = [(1,1), (5,2), (3,5), (7,7)]
vor = Voronoi(points)

fig = voronoi_plot_2d(vor)
plt.plot([p[0] for p in points], [p[1] for p in points], 'ro')
plt.show()
```

To locate a point's nearest facility:

```python
import numpy as np
def nearest_facility(points, q):
    points = np.array(points)
    dists = np.linalg.norm(points - np.array(q), axis=1)
    return np.argmin(dists)
```

#### Why It Matters

* Location optimization: Assign customers to nearest warehouses or service centers.
* Computational geometry: Core primitive in spatial analysis and meshing.
* GIS and logistics: Used in region partitioning and demand modeling.
* Robotics and coverage: Useful in territory planning, clustering, and sensor distribution.

#### A Gentle Proof (Why It Works)

Every boundary in the Voronoi diagram is defined by equidistant points between two facilities.
The plane is partitioned such that each location belongs to the region of the closest site.

Convexity holds because:
$$
V(p_i) = \bigcap_{j \ne i} { q : d(q, p_i) \le d(q, p_j) }
$$
and each inequality defines a half-plane, so their intersection is convex.

Thus, every Voronoi region is convex, and every query has a unique nearest facility.

#### Try It Yourself

1. Place three facilities on a grid.
2. Draw perpendicular bisectors between every pair.
3. Each intersection defines a Voronoi vertex.
4. Pick any random point, check which region it falls into.
   That facility is its nearest neighbor.

#### Test Cases

| Facilities          | Query Point | Nearest |
| ------------------- | ----------- | ------- |
| (0,0), (5,0)        | (2,1)       | (0,0)   |
| (1,1), (4,4), (7,1) | (3,3)       | (4,4)   |
| (2,2), (6,6)        | (5,3)       | (6,6)   |

#### Complexity

| Operation              | Time          | Space  |
| ---------------------- | ------------- | ------ |
| Build Voronoi diagram  | $O(n \log n)$ | $O(n)$ |
| Query nearest facility | $O(\log n)$   | $O(1)$ |

The Voronoi Nearest Facility algorithm captures a simple yet profound truth:
each place on the map belongs to the facility it loves most —
the one that stands closest, by pure geometric destiny.

### 798 Delaunay Mesh Generation

Delaunay Mesh Generation creates high-quality triangular meshes from a set of points, optimizing for numerical stability and smoothness.
It's a cornerstone in computational geometry, finite element methods (FEM), and computer graphics.

#### What Problem Are We Solving?

Given a set of points $P = {p_1, p_2, \dots, p_n}$, we want to construct a triangulation (a division into triangles) such that:

1. No point lies inside the circumcircle of any triangle.
2. Triangles are as "well-shaped" as possible, avoiding skinny, degenerate shapes.

This is known as the Delaunay Triangulation.

#### How It Works (Plain Language)

1. Start with a bounding triangle that contains all points.
2. Insert points one by one:

   * For each new point, find all triangles whose circumcircle contains it.
   * Remove those triangles, forming a polygonal hole.
   * Connect the new point to the vertices of the hole to form new triangles.
3. Remove any triangle connected to the bounding vertices.

The result is a triangulation maximizing the minimum angle among all triangles.

#### Mathematical Criterion

For any triangle $\triangle ABC$ with circumcircle passing through points $A$, $B$, and $C$,
a fourth point $D$ violates the Delaunay condition if it lies inside that circle.

This can be tested via determinant:

$$
\begin{vmatrix}
x_A & y_A & x_A^2 + y_A^2 & 1 \\
x_B & y_B & x_B^2 + y_B^2 & 1 \\
x_C & y_C & x_C^2 + y_C^2 & 1 \\
x_D & y_D & x_D^2 + y_D^2 & 1
\end{vmatrix} > 0
$$

If the determinant is positive, $D$ lies inside the circumcircle —
hence, the triangulation must be flipped to restore the Delaunay property.

#### Tiny Code (Python Example)

```python
import numpy as np
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt

points = np.random.rand(10, 2)
tri = Delaunay(points)

plt.triplot(points[:,0], points[:,1], tri.simplices)
plt.plot(points[:,0], points[:,1], 'o')
plt.show()
```

This snippet generates a 2D Delaunay triangulation and plots it.

#### Why It Matters

* Finite Element Analysis (FEA): provides well-conditioned meshes for simulations.
* Terrain and surface modeling: builds smooth, non-overlapping triangulations.
* Computer graphics: used in tessellation, shading, and 3D modeling.
* Scientific computing: enables stable numerical interpolation.

#### A Gentle Proof (Why It Works)

Delaunay triangulation maximizes the minimum angle among all triangulations of $P$.
This avoids thin, elongated triangles that cause instability.

Key geometric duality:

* The Delaunay Triangulation is the dual of the Voronoi Diagram.
* Each Delaunay edge connects points whose Voronoi cells share a boundary.

Thus, constructing one automatically defines the other.

#### Try It Yourself

1. Plot a few random points on paper.
2. Draw their circumcircles and find intersections that don't contain any other points.
3. Connect those points, you've built a Delaunay triangulation manually.
4. Now perturb one point slightly, notice how the structure adjusts while staying valid.

#### Test Cases

| Input Points     | Resulting Triangles | Notes                  |
| ---------------- | ------------------- | ---------------------- |
| 4 corner points  | 2 triangles         | Simple square split    |
| Random 5 points  | 5–6 triangles       | Depends on convex hull |
| 10 random points | ≈ 2n triangles      | Typical planar density |

#### Complexity

| Operation           | Expected Time    | Space  |
| ------------------- | ---------------- | ------ |
| Build triangulation | $O(n \log n)$    | $O(n)$ |
| Point insertion     | $O(\log n)$      | $O(1)$ |
| Edge flip           | $O(1)$ amortized |,      |

#### Variants and Extensions

* Constrained Delaunay Triangulation (CDT): preserves specific edges.
* 3D Delaunay Tetrahedralization: extends to spatial meshes.
* Adaptive refinement: improves triangle quality by inserting new points.
* Anisotropic Delaunay: accounts for directional metrics.

The Delaunay mesh is where geometry meets stability —
a network of triangles that knows how to stay balanced, elegant, and efficient.

### 799 Smallest Enclosing Circle (Welzl's Algorithm)

The Smallest Enclosing Circle problem finds the smallest possible circle that contains all given points in a plane.
It is also known as the Minimum Enclosing Circle or Bounding Circle problem.

#### What Problem Are We Solving?

Given a set of points $P = {p_1, p_2, \dots, p_n}$ in 2D space, find the circle with minimum radius $r$ and center $c = (x, y)$ such that:

$$
\forall p_i \in P, \quad |p_i - c| \le r
$$

This circle "wraps" all the points as tightly as possible, like stretching a rubber band around them and fitting the smallest possible circle.

#### How It Works (Plain Language)

The Welzl algorithm solves this efficiently using randomized incremental construction:

1. Shuffle the points randomly.
2. Build the enclosing circle incrementally:

   * Start with no points, where the circle is undefined.
   * For each new point:

     * If the point is inside the current circle, do nothing.
     * If it lies outside, rebuild the circle so that it includes this new point.
3. The circle can be defined by:

   * Two points (when they are the diameter), or
   * Three points (when they define a unique circle through all).

Expected time complexity: O(n).

#### Geometric Construction

1. Two points (A, B):
   The circle's center is the midpoint, radius is half the distance:
   $$
   c = \frac{A + B}{2}, \quad r = \frac{|A - B|}{2}
   $$

2. Three points (A, B, C):
   The circle is the unique one passing through all three.
   Using perpendicular bisectors:

   $$
   \begin{aligned}
   D &= 2(A_x(B_y - C_y) + B_x(C_y - A_y) + C_x(A_y - B_y)) \
   U_x &= \frac{(A_x^2 + A_y^2)(B_y - C_y) + (B_x^2 + B_y^2)(C_y - A_y) + (C_x^2 + C_y^2)(A_y - B_y)}{D} \
   U_y &= \frac{(A_x^2 + A_y^2)(C_x - B_x) + (B_x^2 + B_y^2)(A_x - C_x) + (C_x^2 + C_y^2)(B_x - A_x)}{D}
   \end{aligned}
   $$

   The circle's center is $(U_x, U_y)$, radius $r = |A - U|$.

#### Tiny Code (Python Example)

```python
import math, random

def dist(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

def circle_two_points(a, b):
    center = ((a[0]+b[0])/2, (a[1]+b[1])/2)
    radius = dist(a, b)/2
    return center, radius

def circle_three_points(a, b, c):
    ax, ay = a; bx, by = b; cx, cy = c
    d = 2*(ax*(by-cy) + bx*(cy-ay) + cx*(ay-by))
    ux = ((ax2+ay2)*(by-cy) + (bx2+by2)*(cy-ay) + (cx2+cy2)*(ay-by)) / d
    uy = ((ax2+ay2)*(cx-bx) + (bx2+by2)*(ax-cx) + (cx2+cy2)*(bx-ax)) / d
    center = (ux, uy)
    radius = dist(center, a)
    return center, radius

def welzl(points):
    random.shuffle(points)
    def mec(pts, boundary):
        if not pts or len(boundary) == 3:
            if len(boundary) == 0:
                return ((0, 0), 0)
            if len(boundary) == 1:
                return (boundary[0], 0)
            if len(boundary) == 2:
                return circle_two_points(*boundary)
            return circle_three_points(*boundary)
        p = pts.pop()
        c, r = mec(pts, boundary)
        if dist(c, p) <= r:
            pts.append(p)
            return c, r
        res = mec(pts, boundary + [p])
        pts.append(p)
        return res
    return mec(points[:], [])
```

#### Why It Matters

* Geometric bounding: Used in collision detection and bounding volume hierarchies.
* Clustering and spatial statistics: Encloses points tightly for area estimation.
* Graphics and robotics: Simplifies shape approximations.
* Data visualization: Computes compact enclosing shapes.

#### A Gentle Proof (Why It Works)

At most three points define the minimal enclosing circle:

* One point → circle of radius 0.
* Two points → smallest circle with that segment as diameter.
* Three points → smallest circle passing through them.

By random insertion, each point has a small probability of requiring a rebuild, leading to expected O(n) time complexity.

#### Try It Yourself

1. Choose a few points and sketch them on graph paper.
2. Find the pair of farthest points, draw the circle through them.
3. Add another point outside, adjust the circle to include it.
4. Observe when three points define the exact smallest circle.

#### Test Cases

| Input Points        | Smallest Enclosing Circle (center, radius) |
| ------------------- | ------------------------------------------ |
| (0,0), (1,0)        | ((0.5, 0), 0.5)                            |
| (0,0), (0,2), (2,0) | ((1,1), √2)                                |
| (1,1), (2,2), (3,1) | ((2,1.5), √1.25)                           |

#### Complexity

| Operation    | Expected Time | Space  |
| ------------ | ------------- | ------ |
| Build Circle | $O(n)$        | $O(1)$ |
| Verify       | $O(n)$        | $O(1)$ |

The Welzl algorithm reveals a simple truth in geometry:
the smallest circle that embraces all points is never fragile —
it's perfectly balanced, defined by the few that reach its edge.

### 800 Collision Detection (Separating Axis Theorem)

The Separating Axis Theorem (SAT) is a fundamental geometric principle for detecting whether two convex shapes are intersecting.
It provides both a proof of intersection and a way to compute minimal separating distance when they do not overlap.

#### What Problem Are We Solving?

Given two convex polygons (or convex polyhedra in 3D), determine whether they collide, meaning their interiors overlap, or are disjoint.

For convex shapes $A$ and $B$, the Separating Axis Theorem states:

> Two convex shapes do not intersect if and only if there exists a line (axis) along which their projections do not overlap.

That line is called the separating axis.

#### How It Works (Plain Language)

1. For each edge of both polygons:

   * Compute the normal vector (perpendicular to the edge).
   * Treat that normal as a potential separating axis.
2. Project both polygons onto the axis:
   $$
   \text{projection} = [\min(v \cdot n), \max(v \cdot n)]
   $$
   where $v$ is a vertex and $n$ is the unit normal.
3. If there exists an axis where the projections do not overlap,
   then the polygons are not colliding.
4. If all projections overlap, the polygons intersect.

#### Mathematical Test

For a given axis $n$:

$$
A_{\text{min}} = \min_{a \in A}(a \cdot n), \quad A_{\text{max}} = \max_{a \in A}(a \cdot n)
$$
$$
B_{\text{min}} = \min_{b \in B}(b \cdot n), \quad B_{\text{max}} = \max_{b \in B}(b \cdot n)
$$

If
$$
A_{\text{max}} < B_{\text{min}} \quad \text{or} \quad B_{\text{max}} < A_{\text{min}}
$$
then a separating axis exists → no collision.

Otherwise, projections overlap → collision.

#### Tiny Code (Python Example)

```python
import numpy as np

def project(polygon, axis):
    dots = [np.dot(v, axis) for v in polygon]
    return min(dots), max(dots)

def overlap(a_proj, b_proj):
    return not (a_proj[1] < b_proj[0] or b_proj[1] < a_proj[0])

def sat_collision(polygon_a, polygon_b):
    polygons = [polygon_a, polygon_b]
    for poly in polygons:
        for i in range(len(poly)):
            p1, p2 = poly[i], poly[(i+1) % len(poly)]
            edge = np.subtract(p2, p1)
            axis = np.array([-edge[1], edge[0]])  # perpendicular normal
            axis = axis / np.linalg.norm(axis)
            if not overlap(project(polygon_a, axis), project(polygon_b, axis)):
                return False
    return True
```

#### Why It Matters

* Physics engines: Core for detecting collisions between objects in 2D and 3D.
* Game development: Efficient for convex polygons, bounding boxes, and polyhedra.
* Robotics: Used in motion planning and obstacle avoidance.
* CAD systems: Helps test intersections between parts or surfaces.

#### A Gentle Proof (Why It Works)

Each convex polygon can be described as the intersection of half-planes.
If two convex sets do not intersect, there must exist at least one hyperplane that separates them completely.

Projecting onto the normal vectors of all edges covers all potential separating directions.
If no separation is found, the sets overlap.

This follows directly from the Hyperplane Separation Theorem in convex geometry.

#### Try It Yourself

1. Draw two rectangles or convex polygons on paper.
2. Compute normals for each edge.
3. Project both polygons onto each normal and compare intervals.
4. If you find one axis with no overlap, that's your separating axis.

#### Test Cases

| Shape A                 | Shape B                    | Result       |
| ----------------------- | -------------------------- | ------------ |
| Overlapping squares     | Shifted by less than width | Collision    |
| Non-overlapping squares | Shifted by more than width | No collision |
| Triangle vs rectangle   | Touching edge              | Collision    |
| Triangle vs rectangle   | Fully separated            | No collision |

#### Complexity

| Operation                            | Time       | Space  |
| ------------------------------------ | ---------- | ------ |
| Collision test (2D convex)           | $O(n + m)$ | $O(1)$ |
| Collision test (3D convex polyhedra) | $O(n + m)$ | $O(1)$ |

$n$ and $m$ are the number of edges (or faces).

#### Extensions

* 3D version: Use face normals and cross-products of edges as axes.
* GJK algorithm: A faster alternative for arbitrary convex shapes.
* EPA (Expanding Polytope Algorithm): Finds penetration depth after collision.
* Broad-phase detection: Combine SAT with bounding volumes for efficiency.

The Separating Axis Theorem captures the essence of collision logic —
to find contact, we only need to look for space between.
If no space exists, the objects are already meeting.



