# Chapter 2. Sorting and searching 

## 11. Elementary sorting 

### 101 Bubble Sort

Bubble Sort is like washing dishes one by one, you keep moving the biggest plate to the bottom until everything is clean and sorted. It is simple, visual, and perfect for building your sorting intuition before diving into more advanced methods.

### What Problem Are We Solving?

We want to arrange a list of elements in order (ascending or descending) by repeatedly comparing and swapping adjacent items that are out of order.

Formally:
Given an array `A[0…n-1]`, repeat passes until no swaps occur.
Each pass bubbles up the largest remaining element to its final position.

#### Example

| Step | Array State  | Description             |
| ---- | ------------ | ----------------------- |
| 0    | [5, 3, 4, 1] | Initial array           |
| 1    | [3, 4, 1, 5] | 5 bubbled to the end    |
| 2    | [3, 1, 4, 5] | 4 bubbled to position 3 |
| 3    | [1, 3, 4, 5] | Array fully sorted      |

### How Does It Work (Plain Language)?

Imagine bubbles rising to the surface, the biggest one reaches the top first. In Bubble Sort, each sweep through the list compares neighboring pairs, swapping them if they are in the wrong order. After each full pass, one more element settles into place.

We repeat until a pass finishes with no swaps, meaning the array is sorted.

#### Step-by-Step Process

| Step | Action                     | Resulting Array      |
| ---- | -------------------------- | -------------------- |
| 1    | Compare A[0] and A[1]      | Swap if needed       |
| 2    | Compare A[1] and A[2]      | Swap if needed       |
| 3    | Continue through A[n-2]    | Repeat comparisons   |
| 4    | Repeat passes until sorted | Early stop if sorted |

### Tiny Code (Easy Versions)

#### C

```c
#include <stdio.h>
#include <stdbool.h>

void bubble_sort(int a[], int n) {
    bool swapped;
    for (int pass = 0; pass < n - 1; pass++) {
        swapped = false;
        for (int i = 0; i < n - pass - 1; i++) {
            if (a[i] > a[i + 1]) {
                int temp = a[i];
                a[i] = a[i + 1];
                a[i + 1] = temp;
                swapped = true;
            }
        }
        if (!swapped) break; // early exit if already sorted
    }
}

int main(void) {
    int a[] = {5, 3, 4, 1, 2};
    int n = sizeof(a) / sizeof(a[0]);
    bubble_sort(a, n);
    for (int i = 0; i < n; i++) printf("%d ", a[i]);
    printf("\n");
}
```

#### Python

```python
def bubble_sort(a):
    n = len(a)
    for pass_num in range(n - 1):
        swapped = False
        for i in range(n - pass_num - 1):
            if a[i] > a[i + 1]:
                a[i], a[i + 1] = a[i + 1], a[i]
                swapped = True
        if not swapped:
            break

arr = [5, 3, 4, 1, 2]
bubble_sort(arr)
print(arr)
```

### Why It Matters

- Teaches comparison-based sorting through intuition
- Builds understanding of adjacent swaps and pass-based progress
- Introduces stability (equal elements keep their relative order)
- Sets the stage for improved versions (Improved Bubble Sort, Cocktail Shaker Sort, Comb Sort)

### A Gentle Proof (Why It Works)

After the first pass, the largest element moves to the last position.
After the second pass, the second largest is in position n-2.

So after k passes, the last k elements are sorted.

If we track comparisons:
1st pass: (n−1) comparisons
2nd pass: (n−2) comparisons
...
(n−1)th pass: 1 comparison

| Pass | Comparisons | Elements Sorted at End |
| ---- | ----------- | ---------------------- |
| 1    | n−1         | Largest element        |
| 2    | n−2         | Next largest           |
| ...  | ...         | ...                    |
| n−1  | 1           | Fully sorted array     |

Total comparisons = (n−1) + (n−2) + ... + 1 = n(n−1)/2

So time = O(n²) in the worst case.
If already sorted, early exit makes it O(n).

### Try It Yourself

| Task | Description                                  |
| ---- | -------------------------------------------- |
| 1    | Sort [3, 2, 1] step by step                  |
| 2    | Count how many swaps occur                   |
| 3    | Add a flag to detect early termination       |
| 4    | Compare with Insertion Sort on the same data |
| 5    | Modify to sort descending                    |

### Test Cases

| Input           | Output          | Passes | Swaps |
| --------------- | --------------- | ------ | ----- |
| [3, 2, 1]       | [1, 2, 3]       | 3      | 3     |
| [1, 2, 3]       | [1, 2, 3]       | 1      | 0     |
| [5, 1, 4, 2, 8] | [1, 2, 4, 5, 8] | 4      | 5     |

### Complexity

| Aspect       | Value                       |
| ------------ | --------------------------- |
| Time (Worst) | O(n²)                       |
| Time (Best)  | O(n)                        |
| Space        | O(1) (in-place)             |
| Stable       | Yes                         |
| Adaptive     | Yes (stops early if sorted) |

Bubble Sort is your first step into the sorting world, simple enough to code by hand, visual enough to animate, and powerful enough to spark intuition for more advanced sorts.

### 102 Improved Bubble Sort

Improved Bubble Sort builds on the basic version by recognizing that once part of the array is sorted, there's no need to revisit it. It introduces small optimizations like early termination and tracking the last swap position to reduce unnecessary comparisons.

### What Problem Are We Solving?

Basic Bubble Sort keeps scanning the whole array every pass, even when the tail is already sorted.
Improved Bubble Sort fixes this by remembering where the last swap happened. Elements beyond that index are already in place, so the next pass can stop earlier.

This optimization is especially effective for arrays that are nearly sorted.

#### Example

| Step | Array State     | Last Swap Index | Range Checked |
| ---- | --------------- | --------------- | ------------- |
| 0    | [5, 3, 4, 1, 2] | -               | 0 to 4        |
| 1    | [3, 4, 1, 2, 5] | 3               | 0 to 3        |
| 2    | [3, 1, 2, 4, 5] | 2               | 0 to 2        |
| 3    | [1, 2, 3, 4, 5] | 1               | 0 to 1        |
| 4    | [1, 2, 3, 4, 5] | 0               | Stop early    |

### How Does It Work (Plain Language)?

We improve efficiency by narrowing each pass to only the unsorted part.
We also stop early when no swaps occur, signaling the array is already sorted.

Step by step:

1. Track index of the last swap in each pass
2. Next pass ends at that index
3. Stop when no swaps occur (fully sorted)

This reduces unnecessary comparisons in nearly sorted arrays.

### Tiny Code (Easy Versions)

#### C

```c
#include <stdio.h>

void improved_bubble_sort(int a[], int n) {
    int new_n;
    while (n > 1) {
        new_n = 0;
        for (int i = 1; i < n; i++) {
            if (a[i - 1] > a[i]) {
                int temp = a[i - 1];
                a[i - 1] = a[i];
                a[i] = temp;
                new_n = i;
            }
        }
        n = new_n;
    }
}

int main(void) {
    int a[] = {5, 3, 4, 1, 2};
    int n = sizeof(a) / sizeof(a[0]);
    improved_bubble_sort(a, n);
    for (int i = 0; i < n; i++) printf("%d ", a[i]);
    printf("\n");
}
```

#### Python

```python
def improved_bubble_sort(a):
    n = len(a)
    while n > 1:
        new_n = 0
        for i in range(1, n):
            if a[i - 1] > a[i]:
                a[i - 1], a[i] = a[i], a[i - 1]
                new_n = i
        n = new_n

arr = [5, 3, 4, 1, 2]
improved_bubble_sort(arr)
print(arr)
```

### Why It Matters

- Reduces redundant comparisons
- Automatically adapts to partially sorted data
- Stops as soon as the array is sorted
- Retains stability and simplicity

### A Gentle Proof (Why It Works)

If the last swap occurs at index `k`, all elements after `k` are already in order.
Next pass only needs to scan up to `k`.
If no swaps occur (`k = 0`), the array is sorted.

| Pass | Comparisons | Last Swap | Range Next Pass |
| ---- | ----------- | --------- | --------------- |
| 1    | n−1         | k₁        | 0..k₁           |
| 2    | k₁−1        | k₂        | 0..k₂           |
| ...  | ...         | ...       | ...             |

In the best case (already sorted), only one pass occurs: O(n)
Worst case remains O(n²)

### Try It Yourself

| Task | Description                                         |
| ---- | --------------------------------------------------- |
| 1    | Sort [1, 2, 3, 4, 5] and observe early stop         |
| 2    | Sort [5, 4, 3, 2, 1] and track last swap index      |
| 3    | Modify to print last swap index each pass           |
| 4    | Compare with standard Bubble Sort pass count        |
| 5    | Try arrays with repeated values to verify stability |

### Test Cases

| Input           | Output          | Passes | Improvement            |
| --------------- | --------------- | ------ | ---------------------- |
| [1, 2, 3, 4, 5] | [1, 2, 3, 4, 5] | 1      | Early stop             |
| [5, 3, 4, 1, 2] | [1, 2, 3, 4, 5] | 4      | Fewer checks each pass |
| [2, 1, 3, 4, 5] | [1, 2, 3, 4, 5] | 1      | Detects sorted tail    |

### Complexity

| Aspect       | Value |
| ------------ | ----- |
| Time (Worst) | O(n²) |
| Time (Best)  | O(n)  |
| Space        | O(1)  |
| Stable       | Yes   |
| Adaptive     | Yes   |

Improved Bubble Sort shows how a small observation can make a classic algorithm smarter. By tracking the last swap, it skips already-sorted tails and gives a glimpse of how adaptive sorting works in practice.

### 103 Cocktail Shaker Sort

Cocktail Shaker Sort, also known as Bidirectional Bubble Sort, improves on Bubble Sort by sorting in both directions during each pass. It moves the largest element to the end and the smallest to the beginning, reducing the number of passes required.

### What Problem Are We Solving?

Standard Bubble Sort only bubbles up in one direction, pushing the largest element to the end each pass.
If small elements start near the end, they take many passes to reach their position.

Cocktail Shaker Sort fixes this by sweeping back and forth, bubbling both ends at once.

#### Example

| Step | Direction    | Array State     | Description             |
| ---- | ------------ | --------------- | ----------------------- |
| 0    | –            | [5, 3, 4, 1, 2] | Initial array           |
| 1    | Left → Right | [3, 4, 1, 2, 5] | 5 bubbled to end        |
| 2    | Right → Left | [1, 3, 4, 2, 5] | 1 bubbled to start      |
| 3    | Left → Right | [1, 3, 2, 4, 5] | 4 bubbled to position 4 |
| 4    | Right → Left | [1, 2, 3, 4, 5] | 2 bubbled to position 2 |

Sorted after 4 directional passes.

### How Does It Work (Plain Language)?

Cocktail Shaker Sort is like stirring from both sides of the array.
Each forward pass moves the largest unsorted element to the end.
Each backward pass moves the smallest unsorted element to the start.

The unsorted region shrinks from both ends with each full cycle.

#### Step-by-Step Process

| Step | Action                                        | Result |
| ---- | --------------------------------------------- | ------ |
| 1    | Sweep left to right, bubble largest to end    |        |
| 2    | Sweep right to left, bubble smallest to start |        |
| 3    | Narrow bounds, repeat until sorted            |        |

### Tiny Code (Easy Versions)

#### C

```c
#include <stdio.h>
#include <stdbool.h>

void cocktail_shaker_sort(int a[], int n) {
    bool swapped = true;
    int start = 0, end = n - 1;

    while (swapped) {
        swapped = false;

        // Forward pass
        for (int i = start; i < end; i++) {
            if (a[i] > a[i + 1]) {
                int temp = a[i];
                a[i] = a[i + 1];
                a[i + 1] = temp;
                swapped = true;
            }
        }
        if (!swapped) break;
        swapped = false;
        end--;

        // Backward pass
        for (int i = end - 1; i >= start; i--) {
            if (a[i] > a[i + 1]) {
                int temp = a[i];
                a[i] = a[i + 1];
                a[i + 1] = temp;
                swapped = true;
            }
        }
        start++;
    }
}

int main(void) {
    int a[] = {5, 3, 4, 1, 2};
    int n = sizeof(a) / sizeof(a[0]);
    cocktail_shaker_sort(a, n);
    for (int i = 0; i < n; i++) printf("%d ", a[i]);
    printf("\n");
}
```

#### Python

```python
def cocktail_shaker_sort(a):
    n = len(a)
    start, end = 0, n - 1
    swapped = True

    while swapped:
        swapped = False
        for i in range(start, end):
            if a[i] > a[i + 1]:
                a[i], a[i + 1] = a[i + 1], a[i]
                swapped = True
        if not swapped:
            break
        swapped = False
        end -= 1
        for i in range(end - 1, start - 1, -1):
            if a[i] > a[i + 1]:
                a[i], a[i + 1] = a[i + 1], a[i]
                swapped = True
        start += 1

arr = [5, 3, 4, 1, 2]
cocktail_shaker_sort(arr)
print(arr)
```

### Why It Matters

- Sorts in both directions, reducing unnecessary passes
- Performs better than Bubble Sort on many practical inputs
- Stable and easy to visualize
- Demonstrates bidirectional improvement, a foundation for adaptive sorting

### A Gentle Proof (Why It Works)

Each forward pass moves the maximum element of the unsorted range to the end.
Each backward pass moves the minimum element of the unsorted range to the start.
Thus, the unsorted range shrinks from both sides, guaranteeing progress each cycle.

| Cycle | Forward Pass   | Backward Pass     | Sorted Range     |
| ----- | -------------- | ----------------- | ---------------- |
| 1     | Largest to end | Smallest to start | [0], [n-1]       |
| 2     | Next largest   | Next smallest     | [0,1], [n-2,n-1] |
| ...   | ...            | ...               | ...              |

Worst case still O(n²), best case O(n) if already sorted.

### Try It Yourself

| Task | Description                                            |
| ---- | ------------------------------------------------------ |
| 1    | Sort [5, 3, 4, 1, 2] and track forward/backward passes |
| 2    | Visualize the shrinking unsorted range                 |
| 3    | Compare with standard Bubble Sort on reverse array     |
| 4    | Modify code to print array after each pass             |
| 5    | Test stability with duplicate values                   |

### Test Cases

| Input           | Output          | Passes | Notes                         |
| --------------- | --------------- | ------ | ----------------------------- |
| [5, 3, 4, 1, 2] | [1, 2, 3, 4, 5] | 4      | Fewer passes than bubble sort |
| [1, 2, 3, 4, 5] | [1, 2, 3, 4, 5] | 1      | Early termination             |
| [2, 1, 3, 5, 4] | [1, 2, 3, 4, 5] | 2      | Moves smallest quickly        |

### Complexity

| Aspect       | Value |
| ------------ | ----- |
| Time (Worst) | O(n²) |
| Time (Best)  | O(n)  |
| Space        | O(1)  |
| Stable       | Yes   |
| Adaptive     | Yes   |

Cocktail Shaker Sort takes the simplicity of Bubble Sort and doubles its efficiency for certain inputs. By sorting in both directions, it highlights the power of symmetry and small algorithmic tweaks.

### 104 Selection Sort

Selection Sort is like organizing a deck of cards by repeatedly picking the smallest card and placing it in order. It is simple, predictable, and useful for understanding how selection-based sorting works.

### What Problem Are We Solving?

We want to sort an array by repeatedly selecting the smallest (or largest) element from the unsorted part and swapping it into the correct position.

Selection Sort separates the array into two parts:

- A sorted prefix (built one element at a time)
- An unsorted suffix (from which we select the next minimum)

#### Example

| Step | Array State     | Action             | Sorted Part | Unsorted Part |
| ---- | --------------- | ------------------ | ----------- | ------------- |
| 0    | [5, 3, 4, 1, 2] | Start              | []          | [5,3,4,1,2]   |
| 1    | [1, 3, 4, 5, 2] | Place 1 at index 0 | [1]         | [3,4,5,2]     |
| 2    | [1, 2, 4, 5, 3] | Place 2 at index 1 | [1,2]       | [4,5,3]       |
| 3    | [1, 2, 3, 5, 4] | Place 3 at index 2 | [1,2,3]     | [5,4]         |
| 4    | [1, 2, 3, 4, 5] | Place 4 at index 3 | [1,2,3,4]   | [5]           |
| 5    | [1, 2, 3, 4, 5] | Done               | [1,2,3,4,5] | []            |

### How Does It Work (Plain Language)?

Selection Sort looks through the unsorted portion, finds the smallest element, and moves it to the front. It does not care about intermediate order until each selection is done.

Each pass fixes one position permanently.

#### Step-by-Step Process

| Step | Action                         | Effect             |
| ---- | ------------------------------ | ------------------ |
| 1    | Find smallest in unsorted part | Move it to front   |
| 2    | Repeat for next unsorted index | Grow sorted prefix |
| 3    | Stop when entire array sorted  |                    |

### Tiny Code (Easy Versions)

#### C

```c
#include <stdio.h>

void selection_sort(int a[], int n) {
    for (int i = 0; i < n - 1; i++) {
        int min_idx = i;
        for (int j = i + 1; j < n; j++) {
            if (a[j] < a[min_idx]) {
                min_idx = j;
            }
        }
        int temp = a[i];
        a[i] = a[min_idx];
        a[min_idx] = temp;
    }
}

int main(void) {
    int a[] = {5, 3, 4, 1, 2};
    int n = sizeof(a) / sizeof(a[0]);
    selection_sort(a, n);
    for (int i = 0; i < n; i++) printf("%d ", a[i]);
    printf("\n");
}
```

#### Python

```python
def selection_sort(a):
    n = len(a)
    for i in range(n - 1):
        min_idx = i
        for j in range(i + 1, n):
            if a[j] < a[min_idx]:
                min_idx = j
        a[i], a[min_idx] = a[min_idx], a[i]

arr = [5, 3, 4, 1, 2]
selection_sort(arr)
print(arr)
```

### Why It Matters

- Simple, deterministic sorting algorithm
- Demonstrates selection rather than swapping neighbors
- Good for small lists and teaching purposes
- Useful when minimizing number of swaps matters

### A Gentle Proof (Why It Works)

At each iteration, the smallest remaining element is placed at its correct position.
Once placed, it never moves again.

The algorithm performs n−1 selections and at most n−1 swaps.
Each selection requires scanning the unsorted part: O(n) comparisons.

| Pass | Search Range | Comparisons | Swap |
| ---- | ------------ | ----------- | ---- |
| 1    | n elements   | n−1         | 1    |
| 2    | n−1 elements | n−2         | 1    |
| ...  | ...          | ...         | ...  |
| n−1  | 2 elements   | 1           | 1    |

Total comparisons = n(n−1)/2 = O(n²)

### Try It Yourself

| Task | Description                                         |
| ---- | --------------------------------------------------- |
| 1    | Trace sorting of [5, 3, 4, 1, 2] step by step       |
| 2    | Count total swaps and comparisons                   |
| 3    | Modify to find maximum each pass (descending order) |
| 4    | Add print statements to see progress                |
| 5    | Compare with Bubble Sort efficiency                 |

### Test Cases

| Input           | Output          | Passes | Swaps |
| --------------- | --------------- | ------ | ----- |
| [3, 2, 1]       | [1, 2, 3]       | 2      | 2     |
| [1, 2, 3]       | [1, 2, 3]       | 2      | 0     |
| [5, 3, 4, 1, 2] | [1, 2, 3, 4, 5] | 4      | 4     |

### Complexity

| Aspect       | Value                     |
| ------------ | ------------------------- |
| Time (Worst) | O(n²)                     |
| Time (Best)  | O(n²)                     |
| Space        | O(1)                      |
| Stable       | No (swap may break order) |
| Adaptive     | No                        |

Selection Sort is a calm, methodical sorter. It does not adapt, but it does not waste swaps either. It is the simplest demonstration of the idea: find the smallest, place it, repeat.

### 105 Double Selection Sort

Double Selection Sort is a refined version of Selection Sort. Instead of finding just the smallest element each pass, it finds both the smallest and the largest, placing them at the beginning and end simultaneously. This halves the number of passes needed.

### What Problem Are We Solving?

Standard Selection Sort finds one element per pass.
Double Selection Sort improves efficiency by selecting two elements per pass, one from each end, reducing total iterations by about half.

It is useful when both extremes can be found in a single scan, improving constant factors while keeping overall simplicity.

#### Example

| Step | Array State     | Min | Max | Action                        | Sorted Part  |
| ---- | --------------- | --- | --- | ----------------------------- | ------------ |
| 0    | [5, 3, 4, 1, 2] | 1   | 5   | Swap 1 → front, 5 → back      | [1, …, 5]    |
| 1    | [1, 3, 4, 2, 5] | 2   | 4   | Swap 2 → index 1, 4 → index 3 | [1,2, …,4,5] |
| 2    | [1, 2, 3, 4, 5] | 3   | 3   | Middle element sorted         | [1,2,3,4,5]  |

Sorted in 3 passes instead of 5.

### How Does It Work (Plain Language)?

Double Selection Sort narrows the unsorted range from both sides.
Each pass:

1. Scans the unsorted section once.
2. Finds both the smallest and largest elements.
3. Swaps them to their correct positions at the front and back.

Then it shrinks the bounds and repeats.

#### Step-by-Step Process

| Step | Action                                | Effect                            |
| ---- | ------------------------------------- | --------------------------------- |
| 1    | Find smallest and largest in unsorted | Move smallest left, largest right |
| 2    | Shrink unsorted range                 | Repeat search                     |
| 3    | Stop when range collapses             | Array sorted                      |

### Tiny Code (Easy Versions)

#### C

```c
#include <stdio.h>

void double_selection_sort(int a[], int n) {
    int left = 0, right = n - 1;

    while (left < right) {
        int min_idx = left, max_idx = left;

        for (int i = left; i <= right; i++) {
            if (a[i] < a[min_idx]) min_idx = i;
            if (a[i] > a[max_idx]) max_idx = i;
        }

        // Move smallest to front
        int temp = a[left];
        a[left] = a[min_idx];
        a[min_idx] = temp;

        // If max element was swapped into min_idx
        if (max_idx == left) max_idx = min_idx;

        // Move largest to back
        temp = a[right];
        a[right] = a[max_idx];
        a[max_idx] = temp;

        left++;
        right--;
    }
}

int main(void) {
    int a[] = {5, 3, 4, 1, 2};
    int n = sizeof(a) / sizeof(a[0]);
    double_selection_sort(a, n);
    for (int i = 0; i < n; i++) printf("%d ", a[i]);
    printf("\n");
}
```

#### Python

```python
def double_selection_sort(a):
    left, right = 0, len(a) - 1
    while left < right:
        min_idx, max_idx = left, left
        for i in range(left, right + 1):
            if a[i] < a[min_idx]:
                min_idx = i
            if a[i] > a[max_idx]:
                max_idx = i

        a[left], a[min_idx] = a[min_idx], a[left]
        if max_idx == left:
            max_idx = min_idx
        a[right], a[max_idx] = a[max_idx], a[right]

        left += 1
        right -= 1

arr = [5, 3, 4, 1, 2]
double_selection_sort(arr)
print(arr)
```

### Why It Matters

- Improves Selection Sort by reducing passes
- Selects two extremes in one scan
- Fewer total swaps and comparisons
- Demonstrates bidirectional selection

### A Gentle Proof (Why It Works)

Each pass moves two elements to their correct final positions.
Thus, after k passes, the first k and last k positions are sorted.
The unsorted range shrinks by 2 each pass.

| Pass | Range Checked | Elements Fixed | Remaining Unsorted |
| ---- | ------------- | -------------- | ------------------ |
| 1    | [0..n−1]      | 2              | n−2                |
| 2    | [1..n−2]      | 2              | n−4                |
| ...  | ...           | ...            | ...                |

Total passes = n/2, each O(n) scan ⇒ O(n²) overall.

### Try It Yourself

| Task | Description                      |
| ---- | -------------------------------- |
| 1    | Sort [5, 3, 4, 1, 2] manually    |
| 2    | Count passes and swaps           |
| 3    | Print range boundaries each pass |
| 4    | Compare to Selection Sort passes |
| 5    | Modify for descending order      |

### Test Cases

| Input           | Output          | Passes | Swaps |
| --------------- | --------------- | ------ | ----- |
| [3, 2, 1]       | [1, 2, 3]       | 2      | 2     |
| [1, 2, 3, 4, 5] | [1, 2, 3, 4, 5] | 2      | 0     |
| [5, 3, 4, 1, 2] | [1, 2, 3, 4, 5] | 3      | 6     |

### Complexity

| Aspect       | Value |
| ------------ | ----- |
| Time (Worst) | O(n²) |
| Time (Best)  | O(n²) |
| Space        | O(1)  |
| Stable       | No    |
| Adaptive     | No    |

Double Selection Sort keeps Selection Sort's simplicity but doubles its reach. By grabbing both ends each pass, it highlights how symmetry can bring efficiency without new data structures.

### 106 Insertion Sort

Insertion Sort builds the sorted array one element at a time, like sorting playing cards in your hand. It takes each new element and inserts it into the correct position among those already sorted.

### What Problem Are We Solving?

We want a simple, stable way to sort elements by inserting each into place within the growing sorted section.
This works especially well for small arrays or nearly sorted data.

Insertion Sort splits the array into two parts:

- Sorted prefix: elements that are already in order
- Unsorted suffix: remaining elements yet to be inserted

#### Example

| Step | Array State     | Element Inserted | Action                          | Sorted Prefix   |
| ---- | --------------- | ---------------- | ------------------------------- | --------------- |
| 0    | [5, 3, 4, 1, 2] | -                | Start with first element sorted | [5]             |
| 1    | [3, 5, 4, 1, 2] | 3                | Insert 3 before 5               | [3, 5]          |
| 2    | [3, 4, 5, 1, 2] | 4                | Insert 4 before 5               | [3, 4, 5]       |
| 3    | [1, 3, 4, 5, 2] | 1                | Insert 1 at front               | [1, 3, 4, 5]    |
| 4    | [1, 2, 3, 4, 5] | 2                | Insert 2 after 1                | [1, 2, 3, 4, 5] |

### How Does It Work (Plain Language)?

Imagine picking cards one by one and placing each into the correct spot among those already held.
Insertion Sort repeats this logic for arrays:

1. Start with the first element (already sorted)
2. Take the next element
3. Compare backward through the sorted section
4. Shift elements to make space and insert it

#### Step-by-Step Process

| Step | Action                                    | Result |
| ---- | ----------------------------------------- | ------ |
| 1    | Take next unsorted element                |        |
| 2    | Move through sorted part to find position |        |
| 3    | Shift larger elements right               |        |
| 4    | Insert element in correct position        |        |

### Tiny Code (Easy Versions)

#### C

```c
#include <stdio.h>

void insertion_sort(int a[], int n) {
    for (int i = 1; i < n; i++) {
        int key = a[i];
        int j = i - 1;
        while (j >= 0 && a[j] > key) {
            a[j + 1] = a[j];
            j--;
        }
        a[j + 1] = key;
    }
}

int main(void) {
    int a[] = {5, 3, 4, 1, 2};
    int n = sizeof(a) / sizeof(a[0]);
    insertion_sort(a, n);
    for (int i = 0; i < n; i++) printf("%d ", a[i]);
    printf("\n");
}
```

#### Python

```python
def insertion_sort(a):
    for i in range(1, len(a)):
        key = a[i]
        j = i - 1
        while j >= 0 and a[j] > key:
            a[j + 1] = a[j]
            j -= 1
        a[j + 1] = key

arr = [5, 3, 4, 1, 2]
insertion_sort(arr)
print(arr)
```

### Why It Matters

- Simple, intuitive, and stable
- Works well for small or nearly sorted arrays
- Commonly used as a subroutine in advanced algorithms (like Timsort)
- Demonstrates concept of incremental insertion

### A Gentle Proof (Why It Works)

At step `i`, the first `i` elements are sorted.
Inserting element `a[i]` keeps the prefix sorted.
Each insertion shifts elements greater than `key` to the right, ensuring correct position.

| Pass | Sorted Portion | Comparisons (Worst) | Shifts (Worst) |
| ---- | -------------- | ------------------- | -------------- |
| 1    | [a₀,a₁]        | 1                   | 1              |
| 2    | [a₀,a₁,a₂]     | 2                   | 2              |
| ...  | ...            | ...                 | ...            |
| n-1  | [a₀…aₙ₋₁]      | n-1                 | n-1            |

Total ≈ (n²)/2 operations in the worst case.

If already sorted, only one comparison per element → O(n).

### Try It Yourself

| Task | Description                                 |
| ---- | ------------------------------------------- |
| 1    | Sort [5, 3, 4, 1, 2] step by step           |
| 2    | Count shifts and comparisons                |
| 3    | Modify to sort descending                   |
| 4    | Compare runtime with Bubble Sort            |
| 5    | Insert print statements to trace insertions |

### Test Cases

| Input           | Output          | Passes | Swaps/Shifts |
| --------------- | --------------- | ------ | ------------ |
| [3, 2, 1]       | [1, 2, 3]       | 2      | 3            |
| [1, 2, 3]       | [1, 2, 3]       | 2      | 0            |
| [5, 3, 4, 1, 2] | [1, 2, 3, 4, 5] | 4      | 8            |

### Complexity

| Aspect       | Value |
| ------------ | ----- |
| Time (Worst) | O(n²) |
| Time (Best)  | O(n)  |
| Space        | O(1)  |
| Stable       | Yes   |
| Adaptive     | Yes   |

Insertion Sort captures the logic of careful, incremental organization. It is slow for large random lists, but elegant, stable, and highly efficient when the data is already close to sorted.

### 107 Binary Insertion Sort

Binary Insertion Sort improves on traditional Insertion Sort by using binary search to find the correct insertion point instead of linear scanning. This reduces the number of comparisons from linear to logarithmic per insertion, while keeping the same stable, adaptive behavior.

### What Problem Are We Solving?

Standard Insertion Sort searches linearly through the sorted part to find where to insert the new element.
If the sorted prefix is long, this costs O(n) comparisons per element.

Binary Insertion Sort replaces that with binary search, which finds the position in O(log n) time, while still performing O(n) shifts.

This makes it a good choice when comparisons are expensive but shifting is cheap.

#### Example

| Step | Sorted Portion | Element to Insert | Insertion Index (Binary Search) | Resulting Array |
| ---- | -------------- | ----------------- | ------------------------------- | --------------- |
| 0    | [5]            | 3                 | 0                               | [3, 5, 4, 1, 2] |
| 1    | [3, 5]         | 4                 | 1                               | [3, 4, 5, 1, 2] |
| 2    | [3, 4, 5]      | 1                 | 0                               | [1, 3, 4, 5, 2] |
| 3    | [1, 3, 4, 5]   | 2                 | 1                               | [1, 2, 3, 4, 5] |

### How Does It Work (Plain Language)?

Just like Insertion Sort, we build a sorted prefix one element at a time.
But instead of scanning backwards linearly, we use binary search to locate the correct position to insert the next element.

We still need to shift larger elements to the right, but we now know exactly where to stop.

#### Step-by-Step Process

| Step | Action                                 | Effect               |
| ---- | -------------------------------------- | -------------------- |
| 1    | Perform binary search in sorted prefix | Find insertion point |
| 2    | Shift larger elements right            | Create space         |
| 3    | Insert element at found index          | Maintain order       |
| 4    | Repeat until sorted                    | Fully sorted array   |

### Tiny Code (Easy Versions)

#### C

```c
#include <stdio.h>

int binary_search(int a[], int item, int low, int high) {
    while (low <= high) {
        int mid = (low + high) / 2;
        if (item == a[mid]) return mid + 1;
        else if (item > a[mid]) low = mid + 1;
        else high = mid - 1;
    }
    return low;
}

void binary_insertion_sort(int a[], int n) {
    for (int i = 1; i < n; i++) {
        int key = a[i];
        int j = i - 1;
        int pos = binary_search(a, key, 0, j);

        while (j >= pos) {
            a[j + 1] = a[j];
            j--;
        }
        a[pos] = key;
    }
}

int main(void) {
    int a[] = {5, 3, 4, 1, 2};
    int n = sizeof(a) / sizeof(a[0]);
    binary_insertion_sort(a, n);
    for (int i = 0; i < n; i++) printf("%d ", a[i]);
    printf("\n");
}
```

#### Python

```python
def binary_search(a, item, low, high):
    while low <= high:
        mid = (low + high) // 2
        if item == a[mid]:
            return mid + 1
        elif item > a[mid]:
            low = mid + 1
        else:
            high = mid - 1
    return low

def binary_insertion_sort(a):
    for i in range(1, len(a)):
        key = a[i]
        pos = binary_search(a, key, 0, i - 1)
        a[pos + 1 : i + 1] = a[pos : i]
        a[pos] = key

arr = [5, 3, 4, 1, 2]
binary_insertion_sort(arr)
print(arr)
```

### Why It Matters

- Fewer comparisons than standard Insertion Sort
- Retains stability and adaptiveness
- Great when comparisons dominate runtime (e.g., complex objects)
- Demonstrates combining search and insertion ideas

### A Gentle Proof (Why It Works)

Binary search always finds the correct index in O(log i) comparisons for the i-th element.
Shifting elements still takes O(i) time.
So total cost:

$$
T(n) = \sum_{i=1}^{n-1} (\log i + i) = O(n^2)
$$

but with fewer comparisons than standard Insertion Sort.

| Step | Comparisons | Shifts | Total Cost |
| ---- | ----------- | ------ | ---------- |
| 1    | log₂1 = 0   | 1      | 1          |
| 2    | log₂2 = 1   | 2      | 3          |
| 3    | log₂3 ≈ 2   | 3      | 5          |
| …    | …           | …      | …          |

### Try It Yourself

| Task | Description                                  |
| ---- | -------------------------------------------- |
| 1    | Sort [5, 3, 4, 1, 2] step by step            |
| 2    | Print insertion index each pass              |
| 3    | Compare comparisons vs normal Insertion Sort |
| 4    | Modify to sort descending                    |
| 5    | Try with already sorted list                 |

### Test Cases

| Input           | Output          | Comparisons | Shifts |
| --------------- | --------------- | ----------- | ------ |
| [3, 2, 1]       | [1, 2, 3]       | ~3          | 3      |
| [1, 2, 3]       | [1, 2, 3]       | ~2          | 0      |
| [5, 3, 4, 1, 2] | [1, 2, 3, 4, 5] | ~7          | 8      |

### Complexity

| Aspect       | Value                               |
| ------------ | ----------------------------------- |
| Time (Worst) | O(n²)                               |
| Time (Best)  | O(n log n) comparisons, O(n) shifts |
| Space        | O(1)                                |
| Stable       | Yes                                 |
| Adaptive     | Yes                                 |

Binary Insertion Sort is a thoughtful balance, smarter searches, same simple structure. It reminds us that even small changes (like using binary search) can bring real efficiency when precision matters.

### 108 Gnome Sort

Gnome Sort is a simple sorting algorithm that works by swapping adjacent elements, similar to Bubble Sort, but with a twist, it moves backward whenever a swap is made. Imagine a gnome tidying flower pots: each time it finds two out of order, it swaps them and steps back to recheck the previous pair.

### What Problem Are We Solving?

We want a simple, intuitive, and in-place sorting method that uses local swaps to restore order.
Gnome Sort is particularly easy to implement and works like an insertion sort with adjacent swaps instead of shifting elements.

It's not the fastest, but it's charmingly simple, perfect for understanding local correction logic.

#### Example

| Step | Position | Array State     | Action                            |
| ---- | -------- | --------------- | --------------------------------- |
| 0    | 1        | [5, 3, 4, 1, 2] | Compare 5 > 3 → Swap, move back   |
| 1    | 0        | [3, 5, 4, 1, 2] | At start → move forward           |
| 2    | 1        | [3, 5, 4, 1, 2] | Compare 5 > 4 → Swap, move back   |
| 3    | 0        | [3, 4, 5, 1, 2] | At start → move forward           |
| 4    | 1        | [3, 4, 5, 1, 2] | Compare 4 < 5 → OK → move forward |
| 5    | 3        | [3, 4, 5, 1, 2] | Compare 5 > 1 → Swap, move back   |
| 6    | 2        | [3, 4, 1, 5, 2] | Compare 4 > 1 → Swap, move back   |
| 7    | 1        | [3, 1, 4, 5, 2] | Compare 3 > 1 → Swap, move back   |
| 8    | 0        | [1, 3, 4, 5, 2] | At start → move forward           |
| 9    | ...      | ...             | Continue until sorted [1,2,3,4,5] |

### How Does It Work (Plain Language)?

The algorithm "walks" through the list:

1. If the current element is greater or equal to the previous one, move forward.
2. If not, swap them and move one step back.
3. Repeat until the end is reached.

If you reach the start of the array, step forward.

It's like Insertion Sort, but instead of shifting, it walks and swaps.

#### Step-by-Step Process

| Step             | Condition                 | Action |
| ---------------- | ------------------------- | ------ |
| If A[i] ≥ A[i−1] | Move forward (i++)        |        |
| If A[i] < A[i−1] | Swap, move backward (i−−) |        |
| If i == 0        | Move forward (i++)        |        |

### Tiny Code (Easy Versions)

#### C

```c
#include <stdio.h>

void gnome_sort(int a[], int n) {
    int i = 1;
    while (i < n) {
        if (i == 0 || a[i] >= a[i - 1]) {
            i++;
        } else {
            int temp = a[i];
            a[i] = a[i - 1];
            a[i - 1] = temp;
            i--;
        }
    }
}

int main(void) {
    int a[] = {5, 3, 4, 1, 2};
    int n = sizeof(a) / sizeof(a[0]);
    gnome_sort(a, n);
    for (int i = 0; i < n; i++) printf("%d ", a[i]);
    printf("\n");
}
```

#### Python

```python
def gnome_sort(a):
    i = 1
    n = len(a)
    while i < n:
        if i == 0 or a[i] >= a[i - 1]:
            i += 1
        else:
            a[i], a[i - 1] = a[i - 1], a[i]
            i -= 1

arr = [5, 3, 4, 1, 2]
gnome_sort(arr)
print(arr)
```

### Why It Matters

- Demonstrates sorting through local correction
- Visually intuitive (good for animation)
- Requires no additional memory
- Stable and adaptive on partially sorted data

### A Gentle Proof (Why It Works)

Each time we swap out-of-order elements, we step back to verify order with the previous one.
Thus, by the time we move forward again, all prior elements are guaranteed to be sorted.

Gnome Sort effectively performs Insertion Sort via adjacent swaps.

| Pass | Swaps | Movement    | Sorted Portion    |
| ---- | ----- | ----------- | ----------------- |
| 1    | Few   | Backward    | Expands gradually |
| n    | Many  | Oscillating | Fully sorted      |

Worst-case swaps: O(n²)
Best-case (already sorted): O(n)

### Try It Yourself

| Task | Description                             |
| ---- | --------------------------------------- |
| 1    | Sort [5, 3, 4, 1, 2] step by step       |
| 2    | Trace `i` pointer movement              |
| 3    | Compare with Insertion Sort shifts      |
| 4    | Animate using console output            |
| 5    | Try reversed input to see maximum swaps |

### Test Cases

| Input           | Output          | Swaps | Notes           |
| --------------- | --------------- | ----- | --------------- |
| [3, 2, 1]       | [1, 2, 3]       | 3     | Many backtracks |
| [1, 2, 3]       | [1, 2, 3]       | 0     | Already sorted  |
| [5, 3, 4, 1, 2] | [1, 2, 3, 4, 5] | 8     | Moderate swaps  |

### Complexity

| Aspect       | Value |
| ------------ | ----- |
| Time (Worst) | O(n²) |
| Time (Best)  | O(n)  |
| Space        | O(1)  |
| Stable       | Yes   |
| Adaptive     | Yes   |

Gnome Sort is a whimsical algorithm that teaches persistence: every time something's out of order, step back, fix it, and keep going. It's inefficient for large data but delightful for learning and visualization.

### 109 Odd-Even Sort

Odd-Even Sort, also known as Brick Sort, is a parallel-friendly variant of Bubble Sort. It alternates between comparing odd-even and even-odd indexed pairs to gradually sort the array. It's especially useful in parallel processing where pairs can be compared simultaneously.

### What Problem Are We Solving?

Bubble Sort compares every adjacent pair in one sweep. Odd-Even Sort breaks this into two alternating phases:

- Odd phase: compare (1,2), (3,4), (5,6), ...
- Even phase: compare (0,1), (2,3), (4,5), ...

Repeating these two passes ensures all adjacent pairs eventually become sorted.

It's ideal for parallel systems or hardware implementations since comparisons in each phase are independent.

#### Example

| Phase | Type | Array State     | Action                        |
| ----- | ---- | --------------- | ----------------------------- |
| 0     | Init | [5, 3, 4, 1, 2] | Start                         |
| 1     | Even | [3, 5, 1, 4, 2] | Compare (0,1), (2,3), (4,5)   |
| 2     | Odd  | [3, 1, 5, 2, 4] | Compare (1,2), (3,4)          |
| 3     | Even | [1, 3, 2, 4, 5] | Compare (0,1), (2,3), (4,5)   |
| 4     | Odd  | [1, 2, 3, 4, 5] | Compare (1,2), (3,4) → Sorted |

### How Does It Work (Plain Language)?

Odd-Even Sort moves elements closer to their correct position with every alternating phase.
It works like a traffic system: cars at even intersections move, then cars at odd intersections move.
Over time, all cars (elements) line up in order.

#### Step-by-Step Process

| Step | Action                      | Result               |
| ---- | --------------------------- | -------------------- |
| 1    | Compare all even-odd pairs  | Swap if out of order |
| 2    | Compare all odd-even pairs  | Swap if out of order |
| 3    | Repeat until no swaps occur | Sorted array         |

### Tiny Code (Easy Versions)

#### C

```c
#include <stdio.h>
#include <stdbool.h>

void odd_even_sort(int a[], int n) {
    bool sorted = false;
    while (!sorted) {
        sorted = true;

        // Even phase
        for (int i = 0; i < n - 1; i += 2) {
            if (a[i] > a[i + 1]) {
                int temp = a[i];
                a[i] = a[i + 1];
                a[i + 1] = temp;
                sorted = false;
            }
        }

        // Odd phase
        for (int i = 1; i < n - 1; i += 2) {
            if (a[i] > a[i + 1]) {
                int temp = a[i];
                a[i] = a[i + 1];
                a[i + 1] = temp;
                sorted = false;
            }
        }
    }
}

int main(void) {
    int a[] = {5, 3, 4, 1, 2};
    int n = sizeof(a) / sizeof(a[0]);
    odd_even_sort(a, n);
    for (int i = 0; i < n; i++) printf("%d ", a[i]);
    printf("\n");
}
```

#### Python

```python
def odd_even_sort(a):
    n = len(a)
    sorted = False
    while not sorted:
        sorted = True

        # Even phase
        for i in range(0, n - 1, 2):
            if a[i] > a[i + 1]:
                a[i], a[i + 1] = a[i + 1], a[i]
                sorted = False

        # Odd phase
        for i in range(1, n - 1, 2):
            if a[i] > a[i + 1]:
                a[i], a[i + 1] = a[i + 1], a[i]
                sorted = False

arr = [5, 3, 4, 1, 2]
odd_even_sort(arr)
print(arr)
```

### Why It Matters

- Demonstrates parallel sorting principles
- Conceptually simple, easy to visualize
- Can be implemented on parallel processors (SIMD, GPU)
- Stable and in-place

### A Gentle Proof (Why It Works)

Odd-Even Sort systematically removes all inversions by alternating comparisons:

- Even phase fixes pairs (0,1), (2,3), (4,5), ...
- Odd phase fixes pairs (1,2), (3,4), (5,6), ...

After n iterations, every element "bubbles" to its position.
It behaves like Bubble Sort but is more structured and phase-based.

| Phase | Comparisons | Independent? | Swaps |
| ----- | ----------- | ------------ | ----- |
| Even  | ⌊n/2⌋       | Yes          | Some  |
| Odd   | ⌊n/2⌋       | Yes          | Some  |

Total complexity remains O(n²), but parallelizable phases reduce wall-clock time.

### Try It Yourself

| Task | Description                                |
| ---- | ------------------------------------------ |
| 1    | Trace [5, 3, 4, 1, 2] through phases       |
| 2    | Count number of phases to sort             |
| 3    | Implement using threads (parallel version) |
| 4    | Compare with Bubble Sort                   |
| 5    | Animate even and odd passes                |

### Test Cases

| Input           | Output          | Phases | Notes              |
| --------------- | --------------- | ------ | ------------------ |
| [3, 2, 1]       | [1, 2, 3]       | 3      | Alternating passes |
| [1, 2, 3]       | [1, 2, 3]       | 1      | Early stop         |
| [5, 3, 4, 1, 2] | [1, 2, 3, 4, 5] | 4      | Moderate passes    |

### Complexity

| Aspect       | Value |
| ------------ | ----- |
| Time (Worst) | O(n²) |
| Time (Best)  | O(n)  |
| Space        | O(1)  |
| Stable       | Yes   |
| Adaptive     | Yes   |

Odd-Even Sort shows that structure matters. By alternating phases, it opens the door to parallel sorting, where independent comparisons can run at once, a neat step toward high-performance sorting.

### 110 Stooge Sort

Stooge Sort is one of the most unusual and quirky recursive sorting algorithms. It's not efficient, but it's fascinating because it sorts by recursively sorting overlapping sections of the array, a great way to study recursion and algorithmic curiosity.

### What Problem Are We Solving?

Stooge Sort doesn't aim for speed. Instead, it provides an example of a recursive divide-and-conquer strategy that's neither efficient nor conventional.
It divides the array into overlapping parts and recursively sorts them, twice on the first two-thirds, once on the last two-thirds.

This algorithm is often used for educational purposes to demonstrate how recursion can be applied in non-traditional ways.

#### Example

| Step | Range  | Action                             | Array State |
| ---- | ------ | ---------------------------------- | ----------- |
| 0    | [0..4] | Start sorting [5,3,4,1,2]          | [5,3,4,1,2] |
| 1    | [0..3] | Sort first 2/3 (4 elements)        | [3,4,1,5,2] |
| 2    | [1..4] | Sort last 2/3 (4 elements)         | [3,1,4,2,5] |
| 3    | [0..3] | Sort first 2/3 again (4 elements)  | [1,3,2,4,5] |
| 4    | Repeat | Until subarrays shrink to length 1 | [1,2,3,4,5] |

### How Does It Work (Plain Language)?

If the first element is larger than the last, swap them.
Then recursively:

1. Sort the first two-thirds of the array.
2. Sort the last two-thirds of the array.
3. Sort the first two-thirds again.

It's like checking the front, then the back, then rechecking the front, until everything settles in order.

#### Step-by-Step Process

| Step | Action                                          |
| ---- | ----------------------------------------------- |
| 1    | Compare first and last elements; swap if needed |
| 2    | Recursively sort first 2/3 of array             |
| 3    | Recursively sort last 2/3 of array              |
| 4    | Recursively sort first 2/3 again                |
| 5    | Stop when subarray length ≤ 1                   |

### Tiny Code (Easy Versions)

#### C

```c
#include <stdio.h>

void stooge_sort(int a[], int l, int h) {
    if (a[l] > a[h]) {
        int temp = a[l];
        a[l] = a[h];
        a[h] = temp;
    }
    if (h - l + 1 > 2) {
        int t = (h - l + 1) / 3;
        stooge_sort(a, l, h - t);
        stooge_sort(a, l + t, h);
        stooge_sort(a, l, h - t);
    }
}

int main(void) {
    int a[] = {5, 3, 4, 1, 2};
    int n = sizeof(a) / sizeof(a[0]);
    stooge_sort(a, 0, n - 1);
    for (int i = 0; i < n; i++) printf("%d ", a[i]);
    printf("\n");
}
```

#### Python

```python
def stooge_sort(a, l, h):
    if a[l] > a[h]:
        a[l], a[h] = a[h], a[l]
    if h - l + 1 > 2:
        t = (h - l + 1) // 3
        stooge_sort(a, l, h - t)
        stooge_sort(a, l + t, h)
        stooge_sort(a, l, h - t)

arr = [5, 3, 4, 1, 2]
stooge_sort(arr, 0, len(arr) - 1)
print(arr)
```

### Why It Matters

- Demonstrates recursive divide-and-conquer logic
- A fun counterexample to "more recursion = more speed"
- Useful in theoretical discussions or algorithmic humor
- Helps build understanding of overlapping subproblems

### A Gentle Proof (Why It Works)

At each step, Stooge Sort ensures:

- The smallest element moves toward the front,
- The largest element moves toward the back,
- The array converges to sorted order through overlapping recursive calls.

Each recursion operates on 2/3 of the range, repeated 3 times, giving recurrence:

$$
T(n) = 3T(2n/3) + O(1)
$$

Solving it (using Master Theorem):

$$
T(n) = O(n^{\log_{1.5} 3}) \approx O(n^{2.7095})
$$

Slower than Bubble Sort (O(n²))!

| Step | Subarray Length | Recursive Calls | Work per Level |
| ---- | --------------- | --------------- | -------------- |
| n    | 3               | O(1)            | Constant       |
| n/2  | 3×3             | O(3)            | Larger         |
| ...  | ...             | ...             | ...            |

### Try It Yourself

| Task | Description                                             |
| ---- | ------------------------------------------------------- |
| 1    | Sort [5, 3, 4, 1, 2] manually and trace recursive calls |
| 2    | Count total swaps                                       |
| 3    | Print recursion depth                                   |
| 4    | Compare with Merge Sort steps                           |
| 5    | Measure runtime for n=10, 100, 1000                     |

### Test Cases

| Input           | Output          | Recursion Depth | Notes             |
| --------------- | --------------- | --------------- | ----------------- |
| [3, 2, 1]       | [1, 2, 3]       | 3               | Works recursively |
| [1, 2, 3]       | [1, 2, 3]       | 1               | Minimal swaps     |
| [5, 3, 4, 1, 2] | [1, 2, 3, 4, 5] | 6               | Many overlaps     |

### Complexity

| Aspect       | Value                    |
| ------------ | ------------------------ |
| Time (Worst) | O(n².7095)               |
| Time (Best)  | O(n².7095)               |
| Space        | O(log n) recursion stack |
| Stable       | No                       |
| Adaptive     | No                       |

Stooge Sort is a delightful oddity, slow, redundant, but undeniably creative. It reminds us that not every recursive idea leads to efficiency, and that algorithm design is as much art as science.

## Section 12. Divide and conquer sorting

### 111 Merge Sort

Merge Sort is one of the most famous divide-and-conquer sorting algorithms. It splits the array into halves, sorts each half recursively, and then merges the two sorted halves into a single sorted array. It guarantees O(n log n) performance, is stable, and serves as the backbone of many modern sorting libraries.

### What Problem Are We Solving?

We want a sorting algorithm that is:

- Efficient on large datasets (O(n log n))
- Stable (preserves equal element order)
- Predictable (no worst-case degradation)

Merge Sort achieves this by dividing the problem into smaller, easily solved subproblems and combining their results.

It's ideal for:

- Sorting linked lists
- External sorting (on disk)
- Stable merges (for multi-key sorting)

#### Example

| Step | Action              | Result                |
| ---- | ------------------- | --------------------- |
| 0    | Input               | [5, 3, 4, 1, 2]       |
| 1    | Split               | [5, 3, 4] and [1, 2]  |
| 2    | Split further       | [5], [3, 4], [1], [2] |
| 3    | Sort subarrays      | [3, 4, 5], [1, 2]     |
| 4    | Merge sorted halves | [1, 2, 3, 4, 5]       |

### How Does It Work (Plain Language)?

1. Divide: Split the array into two halves.
2. Conquer: Recursively sort both halves.
3. Combine: Merge the sorted halves into a single sorted array.

Think of it as sorting two smaller piles, then interleaving them in order, like merging two stacks of playing cards.

#### Step-by-Step Process

- If the array has 0 or 1 element, it's already sorted.
- Recursively sort left and right halves.
- Use a helper merge function to combine them.

### Tiny Code (Easy Versions)

#### C

```c
#include <stdio.h>

void merge(int a[], int left, int mid, int right) {
    int n1 = mid - left + 1, n2 = right - mid;
    int L[n1], R[n2];

    for (int i = 0; i < n1; i++) L[i] = a[left + i];
    for (int j = 0; j < n2; j++) R[j] = a[mid + 1 + j];

    int i = 0, j = 0, k = left;
    while (i < n1 && j < n2) {
        if (L[i] <= R[j]) a[k++] = L[i++];
        else a[k++] = R[j++];
    }
    while (i < n1) a[k++] = L[i++];
    while (j < n2) a[k++] = R[j++];
}

void merge_sort(int a[], int left, int right) {
    if (left < right) {
        int mid = left + (right - left) / 2;
        merge_sort(a, left, mid);
        merge_sort(a, mid + 1, right);
        merge(a, left, mid, right);
    }
}

int main(void) {
    int a[] = {5, 3, 4, 1, 2};
    int n = sizeof(a) / sizeof(a[0]);
    merge_sort(a, 0, n - 1);
    for (int i = 0; i < n; i++) printf("%d ", a[i]);
    printf("\n");
}
```

#### Python

```python
def merge_sort(a):
    if len(a) > 1:
        mid = len(a) // 2
        left = a[:mid]
        right = a[mid:]

        merge_sort(left)
        merge_sort(right)

        i = j = k = 0
        while i < len(left) and j < len(right):
            if left[i] <= right[j]:
                a[k] = left[i]
                i += 1
            else:
                a[k] = right[j]
                j += 1
            k += 1

        while i < len(left):
            a[k] = left[i]
            i += 1
            k += 1
        while j < len(right):
            a[k] = right[j]
            j += 1
            k += 1

arr = [5, 3, 4, 1, 2]
merge_sort(arr)
print(arr)
```

### Why It Matters

- Stable: Keeps relative order of equal elements
- Deterministic O(n log n): Always efficient
- Parallelizable: Subarrays can be sorted independently
- Foundation: For hybrid algorithms like TimSort and External Merge Sort

### A Gentle Proof (Why It Works)

Merge Sort divides the array into two halves at each level.
There are log₂ n levels of recursion, and each merge takes O(n) time.

So total time:
$$
T(n) = O(n \log n)
$$

Merging is linear because each element is copied once per level.

| Step | Work       | Subarrays | Total      |
| ---- | ---------- | --------- | ---------- |
| 1    | O(n)       | 1         | O(n)       |
| 2    | O(n/2) × 2 | 2         | O(n)       |
| 3    | O(n/4) × 4 | 4         | O(n)       |
| ...  | ...        | ...       | O(n log n) |

### Try It Yourself

1. Split [5, 3, 4, 1, 2] into halves step by step.
2. Merge [3, 5] and [1, 4] manually.
3. Trace the recursive calls on paper.
4. Implement an iterative bottom-up version.
5. Modify to sort descending.
6. Print arrays at each merge step.
7. Compare the number of comparisons vs. Bubble Sort.
8. Try merging two pre-sorted arrays [1,3,5] and [2,4,6].
9. Sort a list of strings (alphabetically).
10. Visualize the recursion tree for n = 8.

### Test Cases

| Input           | Output          | Notes           |
| --------------- | --------------- | --------------- |
| [3, 2, 1]       | [1, 2, 3]       | Standard test   |
| [1, 2, 3]       | [1, 2, 3]       | Already sorted  |
| [5, 3, 4, 1, 2] | [1, 2, 3, 4, 5] | General case    |
| [2, 2, 1, 1]    | [1, 1, 2, 2]    | Tests stability |

### Complexity

| Aspect         | Value      |
| -------------- | ---------- |
| Time (Worst)   | O(n log n) |
| Time (Best)    | O(n log n) |
| Time (Average) | O(n log n) |
| Space          | O(n)       |
| Stable         | Yes        |
| Adaptive       | No         |

Merge Sort is your first taste of divide and conquer sorting, calm, reliable, and elegant. It divides the problem cleanly, conquers recursively, and merges with precision.

### 112 Iterative Merge Sort

Iterative Merge Sort is a non-recursive version of Merge Sort that uses bottom-up merging.
Instead of dividing recursively, it starts with subarrays of size 1 and iteratively merges them in pairs, doubling the size each round. This makes it ideal for environments where recursion is expensive or limited.

### What Problem Are We Solving?

Recursive Merge Sort requires function calls and stack space.
In some systems, recursion might be slow or infeasible (e.g. embedded systems, large arrays).
Iterative Merge Sort avoids recursion by sorting iteratively, merging subarrays of increasing size until the entire array is sorted.

It's especially handy for:

- Iterative environments (no recursion)
- Large data sets (predictable memory)
- External sorting with iterative passes

#### Example

| Step | Subarray Size | Action                                 | Array State     |
| ---- | ------------- | -------------------------------------- | --------------- |
| 0    | 1             | Each element is trivially sorted       | [5, 3, 4, 1, 2] |
| 1    | 1             | Merge pairs of 1-element subarrays     | [3, 5, 1, 4, 2] |
| 2    | 2             | Merge pairs of 2-element subarrays     | [1, 3, 4, 5, 2] |
| 3    | 4             | Merge 4-element sorted block with last | [1, 2, 3, 4, 5] |
| 4    | Done          | Fully sorted                           | [1, 2, 3, 4, 5] |

### How Does It Work (Plain Language)?

Think of it as sorting small groups first and then merging those into bigger groups, no recursion required.

Process:

1. Start with subarrays of size 1 (already sorted).
2. Merge adjacent pairs of subarrays.
3. Double the subarray size and repeat.
4. Continue until subarray size ≥ n.

#### Step-by-Step Process

- Outer loop: size = 1, 2, 4, 8, ... until ≥ n
- Inner loop: merge every two adjacent blocks of given size

### Tiny Code (Easy Versions)

#### C

```c
#include <stdio.h>

void merge(int a[], int left, int mid, int right) {
    int n1 = mid - left + 1, n2 = right - mid;
    int L[n1], R[n2];
    for (int i = 0; i < n1; i++) L[i] = a[left + i];
    for (int j = 0; j < n2; j++) R[j] = a[mid + 1 + j];

    int i = 0, j = 0, k = left;
    while (i < n1 && j < n2) {
        if (L[i] <= R[j]) a[k++] = L[i++];
        else a[k++] = R[j++];
    }
    while (i < n1) a[k++] = L[i++];
    while (j < n2) a[k++] = R[j++];
}

void iterative_merge_sort(int a[], int n) {
    for (int size = 1; size < n; size *= 2) {
        for (int left = 0; left < n - 1; left += 2 * size) {
            int mid = left + size - 1;
            int right = (left + 2 * size - 1 < n - 1) ? (left + 2 * size - 1) : (n - 1);
            if (mid < right)
                merge(a, left, mid, right);
        }
    }
}

int main(void) {
    int a[] = {5, 3, 4, 1, 2};
    int n = sizeof(a) / sizeof(a[0]);
    iterative_merge_sort(a, n);
    for (int i = 0; i < n; i++) printf("%d ", a[i]);
    printf("\n");
}
```

#### Python

```python
def merge(a, left, mid, right):
    left_arr = a[left:mid+1]
    right_arr = a[mid+1:right+1]
    i = j = 0
    k = left
    while i < len(left_arr) and j < len(right_arr):
        if left_arr[i] <= right_arr[j]:
            a[k] = left_arr[i]
            i += 1
        else:
            a[k] = right_arr[j]
            j += 1
        k += 1
    while i < len(left_arr):
        a[k] = left_arr[i]
        i += 1
        k += 1
    while j < len(right_arr):
        a[k] = right_arr[j]
        j += 1
        k += 1

def iterative_merge_sort(a):
    n = len(a)
    size = 1
    while size < n:
        for left in range(0, n, 2 * size):
            mid = min(left + size - 1, n - 1)
            right = min(left + 2 * size - 1, n - 1)
            if mid < right:
                merge(a, left, mid, right)
        size *= 2

arr = [5, 3, 4, 1, 2]
iterative_merge_sort(arr)
print(arr)
```

### Why It Matters

- Eliminates recursion (more predictable memory usage)
- Still guarantees O(n log n) performance
- Useful for iterative, bottom-up, or external sorting
- Easier to parallelize since merge operations are independent

### A Gentle Proof (Why It Works)

Each iteration doubles the sorted block size.
Since each element participates in log₂ n merge levels, and each level costs O(n) work, total cost:

$$
T(n) = O(n \log n)
$$

Like recursive Merge Sort, each merge step is linear, and merging subarrays is stable.

| Iteration | Block Size | Merges | Work |
| --------- | ---------- | ------ | ---- |
| 1         | 1          | n/2    | O(n) |
| 2         | 2          | n/4    | O(n) |
| 3         | 4          | n/8    | O(n) |
| ...       | ...        | ...    | ...  |

Total = O(n log n)

### Try It Yourself

1. Sort [5, 3, 4, 1, 2] manually using bottom-up passes.
2. Trace each pass: subarray size = 1 → 2 → 4.
3. Print intermediate arrays after each pass.
4. Compare recursion depth with recursive version.
5. Implement a space-efficient version (in-place merge).
6. Modify to sort descending.
7. Apply to linked list version.
8. Test performance on large array (n = 10⁶).
9. Visualize merging passes as a tree.
10. Implement on external storage (file-based).

### Test Cases

| Input           | Output          | Notes           |
| --------------- | --------------- | --------------- |
| [3, 2, 1]       | [1, 2, 3]       | Small test      |
| [1, 2, 3]       | [1, 2, 3]       | Already sorted  |
| [5, 3, 4, 1, 2] | [1, 2, 3, 4, 5] | General test    |
| [2, 2, 1, 1]    | [1, 1, 2, 2]    | Tests stability |

### Complexity

| Aspect       | Value      |
| ------------ | ---------- |
| Time (Worst) | O(n log n) |
| Time (Best)  | O(n log n) |
| Space        | O(n)       |
| Stable       | Yes        |
| Adaptive     | No         |

Iterative Merge Sort is the non-recursive twin of classic Merge Sort, efficient, stable, and memory-predictable, making it perfect when stack space is at a premium.

### 113 Quick Sort

Quick Sort is one of the fastest and most widely used sorting algorithms. It works by partitioning an array into two halves around a pivot element and recursively sorting the two parts. With average-case O(n log n) performance and in-place operation, it's the go-to choice in many libraries and real-world systems.

### What Problem Are We Solving?

We need a sorting algorithm that's:

- Efficient in practice (fast average case)
- In-place (minimal memory use)
- Divide-and-conquer-based (parallelizable)

Quick Sort partitions the array so that:

- All elements smaller than the pivot go left
- All elements larger go right

Then it sorts each half recursively.

It's ideal for:

- Large datasets in memory
- Systems where memory allocation is limited
- Average-case performance optimization

#### Example

| Step | Pivot | Action         | Array State     |
| ---- | ----- | -------------- | --------------- |
| 0    | 5     | Partition      | [3, 4, 1, 2, 5] |
| 1    | 3     | Partition left | [1, 2, 3, 4, 5] |
| 2    | Done  | Sorted         | [1, 2, 3, 4, 5] |

### How Does It Work (Plain Language)?

1. Choose a pivot element.
2. Rearrange (partition) array so:

   * Elements smaller than pivot move to the left
   * Elements larger than pivot move to the right
3. Recursively apply the same logic to each half.

Think of the pivot as the "divider" that splits the unsorted array into two smaller problems.

#### Step-by-Step Process

| Step | Action                                    |
| ---- | ----------------------------------------- |
| 1    | Select a pivot (e.g., last element)       |
| 2    | Partition array around pivot              |
| 3    | Recursively sort left and right subarrays |
| 4    | Stop when subarray size ≤ 1               |

### Tiny Code (Easy Versions)

#### C (Lomuto Partition Scheme)

```c
#include <stdio.h>

void swap(int *a, int *b) {
    int temp = *a;
    *a = *b;
    *b = temp;
}

int partition(int a[], int low, int high) {
    int pivot = a[high];
    int i = low - 1;
    for (int j = low; j < high; j++) {
        if (a[j] < pivot) {
            i++;
            swap(&a[i], &a[j]);
        }
    }
    swap(&a[i + 1], &a[high]);
    return i + 1;
}

void quick_sort(int a[], int low, int high) {
    if (low < high) {
        int pi = partition(a, low, high);
        quick_sort(a, low, pi - 1);
        quick_sort(a, pi + 1, high);
    }
}

int main(void) {
    int a[] = {5, 3, 4, 1, 2};
    int n = sizeof(a) / sizeof(a[0]);
    quick_sort(a, 0, n - 1);
    for (int i = 0; i < n; i++) printf("%d ", a[i]);
    printf("\n");
}
```

#### Python

```python
def partition(a, low, high):
    pivot = a[high]
    i = low - 1
    for j in range(low, high):
        if a[j] < pivot:
            i += 1
            a[i], a[j] = a[j], a[i]
    a[i + 1], a[high] = a[high], a[i + 1]
    return i + 1

def quick_sort(a, low, high):
    if low < high:
        pi = partition(a, low, high)
        quick_sort(a, low, pi - 1)
        quick_sort(a, pi + 1, high)

arr = [5, 3, 4, 1, 2]
quick_sort(arr, 0, len(arr) - 1)
print(arr)
```

### Why It Matters

- In-place and fast in most real-world cases
- Divide and conquer: naturally parallelizable
- Often used as the default sorting algorithm in libraries (C, Java, Python)
- Introduces partitioning, a key algorithmic pattern

### A Gentle Proof (Why It Works)

Each partition divides the problem into smaller subarrays.
Average partition splits are balanced, giving O(log n) depth and O(n) work per level:

$$
T(n) = 2T(n/2) + O(n) = O(n \log n)
$$

If the pivot is poor (e.g. smallest or largest), complexity degrades:

$$
T(n) = T(n-1) + O(n) = O(n^2)
$$

| Case    | Partition Quality | Complexity |
| ------- | ----------------- | ---------- |
| Best    | Perfect halves    | O(n log n) |
| Average | Random            | O(n log n) |
| Worst   | Unbalanced        | O(n²)      |

Choosing pivots wisely (randomization, median-of-three) avoids worst-case splits.

### Try It Yourself

1. Sort [5, 3, 4, 1, 2] and trace partitions.
2. Change pivot selection (first, middle, random).
3. Count comparisons and swaps for each case.
4. Implement using Hoare Partition scheme.
5. Modify to sort descending.
6. Visualize recursion tree for n = 8.
7. Compare runtime with Merge Sort.
8. Try sorted input [1, 2, 3, 4, 5] and note behavior.
9. Add a counter to count recursive calls.
10. Implement tail recursion optimization.

### Test Cases

| Input           | Output          | Notes                     |
| --------------- | --------------- | ------------------------- |
| [3, 2, 1]       | [1, 2, 3]       | Basic                     |
| [1, 2, 3]       | [1, 2, 3]       | Worst case (sorted input) |
| [5, 3, 4, 1, 2] | [1, 2, 3, 4, 5] | General                   |
| [2, 2, 1, 1]    | [1, 1, 2, 2]    | Duplicates                |

### Complexity

| Aspect         | Value                |
| -------------- | -------------------- |
| Time (Best)    | O(n log n)           |
| Time (Average) | O(n log n)           |
| Time (Worst)   | O(n²)                |
| Space          | O(log n) (recursion) |
| Stable         | No                   |
| Adaptive       | No                   |

Quick Sort is the practical workhorse of sorting, swift, elegant, and widely loved. It teaches how a single smart pivot can bring order to chaos.

### 114 Hoare Partition Scheme

The Hoare Partition Scheme is an early and elegant version of Quick Sort's partitioning method, designed by C.A.R. Hoare himself. It's more efficient than the Lomuto scheme in many cases because it does fewer swaps and uses two pointers moving inward from both ends.

### What Problem Are We Solving?

In Quick Sort, we need a way to divide an array into two parts:

- Elements less than or equal to the pivot
- Elements greater than or equal to the pivot

Hoare's scheme achieves this using two indices that move toward each other, swapping elements that are out of place. It reduces the number of swaps compared to the Lomuto scheme and often performs better on real data.

It's especially useful for:

- Large arrays (fewer writes)
- Performance-critical systems
- In-place partitioning without extra space

#### Example

| Step | Pivot | Left (i) | Right (j) | Array State     | Action                           |
| ---- | ----- | -------- | --------- | --------------- | -------------------------------- |
| 0    | 5     | i=0      | j=4       | [5, 3, 4, 1, 2] | pivot = 5                        |
| 1    | 5     | i→2      | j→4       | [5, 3, 4, 1, 2] | a[j]=2<5 swap(5,2) → [2,3,4,1,5] |
| 2    | 5     | i→2      | j→3       | [2,3,4,1,5]     | a[i]=4>5? no swap                |
| 3    | stop  | -        | -         | [2,3,4,1,5]     | partition done                   |

The pivot ends up near its correct position, but not necessarily in the final index.

### How Does It Work (Plain Language)?

The algorithm picks a pivot (commonly the first element),
then moves two pointers:

- Left pointer (i): moves right, skipping small elements
- Right pointer (j): moves left, skipping large elements

When both pointers find misplaced elements, they are swapped.
This continues until they cross, at that point, the array is partitioned.

#### Step-by-Step Process

1. Choose a pivot (e.g. first element).
2. Set two indices: `i = left - 1`, `j = right + 1`.
3. Increment `i` until `a[i] >= pivot`.
4. Decrement `j` until `a[j] <= pivot`.
5. If `i < j`, swap `a[i]` and `a[j]`. Otherwise, return `j` (partition index).

### Tiny Code (Easy Versions)

#### C

```c
#include <stdio.h>

void swap(int *a, int *b) {
    int temp = *a;
    *a = *b;
    *b = temp;
}

int hoare_partition(int a[], int low, int high) {
    int pivot = a[low];
    int i = low - 1;
    int j = high + 1;

    while (1) {
        do { i++; } while (a[i] < pivot);
        do { j--; } while (a[j] > pivot);
        if (i >= j) return j;
        swap(&a[i], &a[j]);
    }
}

void quick_sort_hoare(int a[], int low, int high) {
    if (low < high) {
        int p = hoare_partition(a, low, high);
        quick_sort_hoare(a, low, p);
        quick_sort_hoare(a, p + 1, high);
    }
}

int main(void) {
    int a[] = {5, 3, 4, 1, 2};
    int n = sizeof(a) / sizeof(a[0]);
    quick_sort_hoare(a, 0, n - 1);
    for (int i = 0; i < n; i++) printf("%d ", a[i]);
    printf("\n");
}
```

#### Python

```python
def hoare_partition(a, low, high):
    pivot = a[low]
    i = low - 1
    j = high + 1
    while True:
        i += 1
        while a[i] < pivot:
            i += 1
        j -= 1
        while a[j] > pivot:
            j -= 1
        if i >= j:
            return j
        a[i], a[j] = a[j], a[i]

def quick_sort_hoare(a, low, high):
    if low < high:
        p = hoare_partition(a, low, high)
        quick_sort_hoare(a, low, p)
        quick_sort_hoare(a, p + 1, high)

arr = [5, 3, 4, 1, 2]
quick_sort_hoare(arr, 0, len(arr) - 1)
print(arr)
```

### Why It Matters

- Fewer swaps than Lomuto partition
- More efficient in practice on most datasets
- Still in-place, divide-and-conquer, O(n log n) average
- Introduces the idea of two-pointer partitioning

### A Gentle Proof (Why It Works)

The loop invariants ensure that:

- Left side: all elements ≤ pivot
- Right side: all elements ≥ pivot
- `i` and `j` move inward until they cross
  When they cross, all elements are partitioned correctly.

The pivot does not end in its final sorted position, but the subarrays can be recursively sorted independently.

$$
T(n) = T(k) + T(n - k - 1) + O(n)
$$

Average complexity O(n log n); worst-case O(n²) if pivot is poor.

| Case    | Pivot   | Behavior         | Complexity |
| ------- | ------- | ---------------- | ---------- |
| Best    | Median  | Balanced halves  | O(n log n) |
| Average | Random  | Slight imbalance | O(n log n) |
| Worst   | Min/Max | Unbalanced       | O(n²)      |

### Try It Yourself

1. Sort [5, 3, 4, 1, 2] step by step using Hoare's scheme.
2. Print `i` and `j` at each iteration.
3. Compare with Lomuto's version on the same array.
4. Try different pivot positions (first, last, random).
5. Measure number of swaps vs. Lomuto.
6. Modify to sort descending.
7. Visualize partition boundaries.
8. Test on array with duplicates [3,3,2,1,4].
9. Implement hybrid pivot selection (median-of-three).
10. Compare runtime with Merge Sort.

### Test Cases

| Input           | Output          | Notes                     |
| --------------- | --------------- | ------------------------- |
| [3, 2, 1]       | [1, 2, 3]       | Simple                    |
| [5, 3, 4, 1, 2] | [1, 2, 3, 4, 5] | General                   |
| [2, 2, 1, 1]    | [1, 1, 2, 2]    | Works with duplicates     |
| [1, 2, 3, 4, 5] | [1, 2, 3, 4, 5] | Sorted input (worst case) |

### Complexity

| Aspect         | Value              |
| -------------- | ------------------ |
| Time (Best)    | O(n log n)         |
| Time (Average) | O(n log n)         |
| Time (Worst)   | O(n²)              |
| Space          | O(log n) recursion |
| Stable         | No                 |
| Adaptive       | No                 |

Hoare Partition Scheme is elegant and efficient, the original genius of Quick Sort. Its two-pointer dance is graceful and economical, a timeless classic in algorithm design.

### 115 Lomuto Partition Scheme

The Lomuto Partition Scheme is a simple and widely taught method for partitioning in Quick Sort. It's easier to understand and implement than Hoare's scheme, though it often performs slightly more swaps. It always selects a pivot (commonly the last element) and partitions the array in a single forward pass.

### What Problem Are We Solving?

We need a clear and intuitive way to partition an array around a pivot, ensuring all smaller elements go to the left and all larger elements go to the right.

Lomuto's method uses one scanning pointer and one boundary pointer, making it easy for beginners and ideal for pedagogical purposes or small datasets.

#### Example

| Step | Pivot | i (Boundary) | j (Scan) | Array State                              | Action    |
| ---- | ----- | ------------ | -------- | ---------------------------------------- | --------- |
| 0    | 2     | i=-1         | j=0      | [5, 3, 4, 1, 2]                          | pivot = 2 |
| 1    | 2     | i=-1         | j=0      | a[0]=5>2 → no swap                       |           |
| 2    | 2     | i=-1         | j=1      | a[1]=3>2 → no swap                       |           |
| 3    | 2     | i=-1         | j=2      | a[2]=4>2 → no swap                       |           |
| 4    | 2     | i=0          | j=3      | a[3]=1<2 → swap(5,1) → [1,3,4,5,2]       |           |
| 5    | 2     | i=0          | j=4      | end of scan; swap pivot with a[i+1]=a[1] |           |
| 6    | Done  | -            | -        | [1,2,4,5,3] partitioned                  |           |

Pivot 2 is placed in its final position at index 1.
Elements left of 2 are smaller, right are larger.

### How Does It Work (Plain Language)?

1. Choose a pivot (commonly last element).
2. Initialize boundary pointer `i` before start of array.
3. Iterate through array with pointer `j`:

   * If `a[j] < pivot`, increment `i` and swap `a[i]` with `a[j]`.
4. After loop, swap `pivot` into position `i + 1`.
5. Return `i + 1` (pivot's final position).

It's like separating a deck of cards, you keep moving smaller cards to the front as you scan.

#### Step-by-Step Process

| Step | Action                                   |
| ---- | ---------------------------------------- |
| 1    | Choose pivot (usually last element)      |
| 2    | Move smaller elements before pivot       |
| 3    | Move larger elements after pivot         |
| 4    | Place pivot in its correct position      |
| 5    | Return pivot index for recursive sorting |

### Tiny Code (Easy Versions)

#### C

```c
#include <stdio.h>

void swap(int *a, int *b) {
    int temp = *a;
    *a = *b;
    *b = temp;
}

int lomuto_partition(int a[], int low, int high) {
    int pivot = a[high];
    int i = low - 1;
    for (int j = low; j < high; j++) {
        if (a[j] < pivot) {
            i++;
            swap(&a[i], &a[j]);
        }
    }
    swap(&a[i + 1], &a[high]);
    return i + 1;
}

void quick_sort_lomuto(int a[], int low, int high) {
    if (low < high) {
        int pi = lomuto_partition(a, low, high);
        quick_sort_lomuto(a, low, pi - 1);
        quick_sort_lomuto(a, pi + 1, high);
    }
}

int main(void) {
    int a[] = {5, 3, 4, 1, 2};
    int n = sizeof(a) / sizeof(a[0]);
    quick_sort_lomuto(a, 0, n - 1);
    for (int i = 0; i < n; i++) printf("%d ", a[i]);
    printf("\n");
}
```

#### Python

```python
def lomuto_partition(a, low, high):
    pivot = a[high]
    i = low - 1
    for j in range(low, high):
        if a[j] < pivot:
            i += 1
            a[i], a[j] = a[j], a[i]
    a[i + 1], a[high] = a[high], a[i + 1]
    return i + 1

def quick_sort_lomuto(a, low, high):
    if low < high:
        pi = lomuto_partition(a, low, high)
        quick_sort_lomuto(a, low, pi - 1)
        quick_sort_lomuto(a, pi + 1, high)

arr = [5, 3, 4, 1, 2]
quick_sort_lomuto(arr, 0, len(arr) - 1)
print(arr)
```

### Why It Matters

- Simple and easy to implement
- Pivot ends in correct final position each step
- Useful for educational demonstration of Quick Sort
- Common in textbooks and basic Quick Sort examples

### A Gentle Proof (Why It Works)

Invariant:

- Elements left of `i` are smaller than pivot.
- Elements between `i` and `j` are greater or not yet checked.
  At the end, swapping pivot with `a[i+1]` places it in its final position.

Time complexity:

$$
T(n) = T(k) + T(n - k - 1) + O(n)
$$

Average: O(n log n)
Worst (already sorted): O(n²)

| Case    | Partition  | Complexity |
| ------- | ---------- | ---------- |
| Best    | Balanced   | O(n log n) |
| Average | Random     | O(n log n) |
| Worst   | Unbalanced | O(n²)      |

### Try It Yourself

1. Sort [5, 3, 4, 1, 2] using Lomuto step by step.
2. Trace `i` and `j` positions at each comparison.
3. Compare with Hoare partition's number of swaps.
4. Test with sorted input, see worst case.
5. Randomize pivot to avoid worst-case.
6. Modify to sort descending order.
7. Count total swaps and comparisons.
8. Combine with tail recursion optimization.
9. Visualize partition boundary after each pass.
10. Implement a hybrid Quick Sort using Lomuto for small arrays.

### Test Cases

| Input           | Output          | Notes              |
| --------------- | --------------- | ------------------ |
| [3, 2, 1]       | [1, 2, 3]       | Simple             |
| [1, 2, 3]       | [1, 2, 3]       | Worst-case         |
| [5, 3, 4, 1, 2] | [1, 2, 3, 4, 5] | General            |
| [2, 2, 1, 1]    | [1, 1, 2, 2]    | Handles duplicates |

### Complexity

| Aspect         | Value              |
| -------------- | ------------------ |
| Time (Best)    | O(n log n)         |
| Time (Average) | O(n log n)         |
| Time (Worst)   | O(n²)              |
| Space          | O(log n) recursion |
| Stable         | No                 |
| Adaptive       | No                 |

Lomuto's scheme is the friendly teacher of Quick Sort, easy to grasp, simple to code, and perfect for building intuition about partitioning and divide-and-conquer sorting.

### 116 Randomized Quick Sort

Randomized Quick Sort enhances the classic Quick Sort by choosing the pivot randomly. This small tweak eliminates the risk of hitting the worst-case O(n²) behavior on already sorted or adversarial inputs, making it one of the most robust and practical sorting strategies in real-world use.

### What Problem Are We Solving?

Regular Quick Sort can degrade badly if the pivot is chosen poorly (for example, always picking the first or last element in a sorted array).
Randomized Quick Sort fixes this by selecting a random pivot, ensuring that, on average, partitions are balanced, regardless of input distribution.

This makes it ideal for:

- Unpredictable or adversarial inputs
- Large datasets where worst-case avoidance matters
- Performance-critical systems requiring consistent behavior

#### Example

| Step | Action                    | Array State     | Pivot        |
| ---- | ------------------------- | --------------- | ------------ |
| 0    | Choose random pivot       | [5, 3, 4, 1, 2] | 4            |
| 1    | Partition around 4        | [3, 2, 1, 4, 5] | 4 at index 3 |
| 2    | Recurse on left [3, 2, 1] | [1, 2, 3]       | 2            |
| 3    | Merge subarrays           | [1, 2, 3, 4, 5] | Done         |

Randomization ensures the pivot is unlikely to create unbalanced partitions.

### How Does It Work (Plain Language)?

It's the same Quick Sort, but before partitioning, we randomly pick one element and use it as the pivot.
This random pivot is swapped into the last position, and the normal Lomuto or Hoare partitioning continues.

This small randomness makes it robust and efficient on average, even for worst-case inputs.

#### Step-by-Step Process

1. Pick a random pivot index between `low` and `high`.
2. Swap the random pivot with the last element.
3. Partition the array (e.g., Lomuto or Hoare).
4. Recursively sort left and right partitions.

### Tiny Code (Easy Versions)

#### C (Lomuto + Random Pivot)

```c
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void swap(int *a, int *b) {
    int temp = *a;
    *a = *b;
    *b = temp;
}

int lomuto_partition(int a[], int low, int high) {
    int pivot = a[high];
    int i = low - 1;
    for (int j = low; j < high; j++) {
        if (a[j] < pivot) {
            i++;
            swap(&a[i], &a[j]);
        }
    }
    swap(&a[i + 1], &a[high]);
    return i + 1;
}

int randomized_partition(int a[], int low, int high) {
    int pivotIndex = low + rand() % (high - low + 1);
    swap(&a[pivotIndex], &a[high]);
    return lomuto_partition(a, low, high);
}

void randomized_quick_sort(int a[], int low, int high) {
    if (low < high) {
        int pi = randomized_partition(a, low, high);
        randomized_quick_sort(a, low, pi - 1);
        randomized_quick_sort(a, pi + 1, high);
    }
}

int main(void) {
    srand(time(NULL));
    int a[] = {5, 3, 4, 1, 2};
    int n = sizeof(a) / sizeof(a[0]);
    randomized_quick_sort(a, 0, n - 1);
    for (int i = 0; i < n; i++) printf("%d ", a[i]);
    printf("\n");
}
```

#### Python

```python
import random

def lomuto_partition(a, low, high):
    pivot = a[high]
    i = low - 1
    for j in range(low, high):
        if a[j] < pivot:
            i += 1
            a[i], a[j] = a[j], a[i]
    a[i + 1], a[high] = a[high], a[i + 1]
    return i + 1

def randomized_partition(a, low, high):
    pivot_index = random.randint(low, high)
    a[pivot_index], a[high] = a[high], a[pivot_index]
    return lomuto_partition(a, low, high)

def randomized_quick_sort(a, low, high):
    if low < high:
        pi = randomized_partition(a, low, high)
        randomized_quick_sort(a, low, pi - 1)
        randomized_quick_sort(a, pi + 1, high)

arr = [5, 3, 4, 1, 2]
randomized_quick_sort(arr, 0, len(arr) - 1)
print(arr)
```

### Why It Matters

- Prevents worst-case O(n²) behavior
- Simple yet highly effective
- Ensures consistent average-case across all inputs
- Foundation for Randomized Select and Randomized Algorithms

### A Gentle Proof (Why It Works)

Random pivot selection ensures the expected split size is balanced, independent of input order.
Each pivot divides the array such that expected recursion depth is O(log n) and total comparisons O(n log n).

Expected complexity:
$$
E[T(n)] = O(n \log n)
$$
Worst-case only occurs with extremely low probability (1/n!).

| Case    | Pivot Choice   | Complexity |
| ------- | -------------- | ---------- |
| Best    | Balanced       | O(n log n) |
| Average | Random         | O(n log n) |
| Worst   | Unlucky (rare) | O(n²)      |

### Try It Yourself

1. Sort [5, 3, 4, 1, 2] multiple times, note pivot differences.
2. Print pivot each recursive call.
3. Compare against deterministic pivot (first, last).
4. Test on sorted input [1, 2, 3, 4, 5].
5. Test on reverse input [5, 4, 3, 2, 1].
6. Count recursive depth across runs.
7. Modify to use Hoare partition.
8. Implement `random.choice()` version.
9. Compare runtime vs. normal Quick Sort.
10. Seed RNG to reproduce same run.

### Test Cases

| Input           | Output          | Notes                 |
| --------------- | --------------- | --------------------- |
| [3, 2, 1]       | [1, 2, 3]       | Random pivot each run |
| [1, 2, 3]       | [1, 2, 3]       | Avoids O(n²)          |
| [5, 3, 4, 1, 2] | [1, 2, 3, 4, 5] | General               |
| [2, 2, 1, 1]    | [1, 1, 2, 2]    | Handles duplicates    |

### Complexity

| Aspect         | Value        |
| -------------- | ------------ |
| Time (Best)    | O(n log n)   |
| Time (Average) | O(n log n)   |
| Time (Worst)   | O(n²) (rare) |
| Space          | O(log n)     |
| Stable         | No           |
| Adaptive       | No           |

Randomized Quick Sort shows the power of randomness, a tiny change transforms a fragile algorithm into a reliably fast one, making it one of the most practical sorts in modern computing.

### 117 Heap Sort

Heap Sort is a classic comparison-based, in-place, O(n log n) sorting algorithm built upon the heap data structure. It first turns the array into a max-heap, then repeatedly removes the largest element (the heap root) and places it at the end, shrinking the heap as it goes. It's efficient and memory-friendly but not stable.

### What Problem Are We Solving?

We want a sorting algorithm that:

- Has guaranteed O(n log n) time in all cases
- Uses constant extra space (O(1))
- Doesn't require recursion or extra arrays like Merge Sort

Heap Sort achieves this by using a binary heap to always extract the largest element efficiently, then placing it in its correct position at the end.

Perfect for:

- Memory-constrained systems
- Predictable performance needs
- Offline sorting when data fits in RAM

#### Example

| Step | Action                    | Array State     | Heap Size |
| ---- | ------------------------- | --------------- | --------- |
| 0    | Build max-heap            | [5, 3, 4, 1, 2] | 5         |
| 1    | Swap root with last       | [2, 3, 4, 1, 5] | 4         |
| 2    | Heapify root              | [4, 3, 2, 1, 5] | 4         |
| 3    | Swap root with last       | [1, 3, 2, 4, 5] | 3         |
| 4    | Heapify root              | [3, 1, 2, 4, 5] | 3         |
| 5    | Repeat until heap shrinks | [1, 2, 3, 4, 5] | 0         |

### How Does It Work (Plain Language)?

Think of a heap like a tree stored in an array. The root (index 0) is the largest element.
Heap Sort works in two main steps:

1. Build a max-heap (arranged so every parent ≥ its children)
2. Extract max repeatedly:

   * Swap root with last element
   * Reduce heap size
   * Heapify root to restore max-heap property

After each extraction, the sorted part grows at the end of the array.

#### Step-by-Step Process

| Step | Description                                |
| ---- | ------------------------------------------ |
| 1    | Build a max-heap from the array            |
| 2    | Swap the first (max) with the last element |
| 3    | Reduce heap size by one                    |
| 4    | Heapify the root                           |
| 5    | Repeat until heap is empty                 |

### Tiny Code (Easy Versions)

#### C

```c
#include <stdio.h>

void swap(int *a, int *b) {
    int temp = *a;
    *a = *b;
    *b = temp;
}

void heapify(int a[], int n, int i) {
    int largest = i;
    int left = 2 * i + 1;
    int right = 2 * i + 2;

    if (left < n && a[left] > a[largest]) largest = left;
    if (right < n && a[right] > a[largest]) largest = right;

    if (largest != i) {
        swap(&a[i], &a[largest]);
        heapify(a, n, largest);
    }
}

void heap_sort(int a[], int n) {
    // Build max heap
    for (int i = n / 2 - 1; i >= 0; i--)
        heapify(a, n, i);

    // Extract elements from heap
    for (int i = n - 1; i > 0; i--) {
        swap(&a[0], &a[i]);
        heapify(a, i, 0);
    }
}

int main(void) {
    int a[] = {5, 3, 4, 1, 2};
    int n = sizeof(a) / sizeof(a[0]);
    heap_sort(a, n);
    for (int i = 0; i < n; i++) printf("%d ", a[i]);
    printf("\n");
}
```

#### Python

```python
def heapify(a, n, i):
    largest = i
    left = 2 * i + 1
    right = 2 * i + 2
    if left < n and a[left] > a[largest]:
        largest = left
    if right < n and a[right] > a[largest]:
        largest = right
    if largest != i:
        a[i], a[largest] = a[largest], a[i]
        heapify(a, n, largest)

def heap_sort(a):
    n = len(a)
    # Build max heap
    for i in range(n // 2 - 1, -1, -1):
        heapify(a, n, i)
    # Extract max and heapify
    for i in range(n - 1, 0, -1):
        a[0], a[i] = a[i], a[0]
        heapify(a, i, 0)

arr = [5, 3, 4, 1, 2]
heap_sort(arr)
print(arr)
```

### Why It Matters

- Predictable O(n log n) in all cases
- In-place, no extra memory needed
- Excellent when memory is tight or recursion is not preferred
- Demonstrates tree-based sorting logic

### A Gentle Proof (Why It Works)

Building the heap: O(n)
Extracting each element: O(log n)
Total time:

$$
T(n) = O(n) + n \times O(\log n) = O(n \log n)
$$

Each element is "bubbled down" log n levels at most once.

| Phase              | Work         | Total      |
| ------------------ | ------------ | ---------- |
| Build heap         | O(n)         | Linear     |
| Extract n elements | n × O(log n) | O(n log n) |

Not stable, because swapping can break equal-element order.

### Try It Yourself

1. Build max-heap from [5, 3, 4, 1, 2].
2. Draw heap tree for each step.
3. Trace heapify calls and swaps.
4. Implement min-heap version for descending sort.
5. Count comparisons per phase.
6. Compare with Merge Sort space usage.
7. Modify to stop early if already sorted.
8. Animate heap construction.
9. Test on reverse array [5,4,3,2,1].
10. Add debug prints showing heap after each step.

### Test Cases

| Input           | Output          | Notes                  |
| --------------- | --------------- | ---------------------- |
| [3, 2, 1]       | [1, 2, 3]       | Small test             |
| [1, 2, 3]       | [1, 2, 3]       | Already sorted         |
| [5, 3, 4, 1, 2] | [1, 2, 3, 4, 5] | General case           |
| [2, 2, 1, 1]    | [1, 1, 2, 2]    | Not stable but correct |

### Complexity

| Aspect         | Value      |
| -------------- | ---------- |
| Time (Best)    | O(n log n) |
| Time (Average) | O(n log n) |
| Time (Worst)   | O(n log n) |
| Space          | O(1)       |
| Stable         | No         |
| Adaptive       | No         |

Heap Sort is the workhorse of guaranteed performance, steady, space-efficient, and built on elegant tree logic. It never surprises you with bad cases, a reliable friend when consistency matters.

### 118 3-Way Quick Sort

3-Way Quick Sort is a refined version of Quick Sort designed to handle arrays with many duplicate elements efficiently. Instead of dividing the array into two parts (less than and greater than pivot), it divides into three regions:

- `< pivot`
- `= pivot`
- `> pivot`

This avoids redundant work when many elements are equal to the pivot, making it especially effective for datasets with low entropy or repeated keys.

### What Problem Are We Solving?

Standard Quick Sort can perform unnecessary work when duplicates are present.
For example, if all elements are the same, standard Quick Sort still recurses O(n log n) times.

3-Way Quick Sort fixes this by:

- Skipping equal elements during partitioning
- Shrinking recursion depth dramatically

It's ideal for:

- Arrays with many duplicates
- Sorting strings with common prefixes
- Key-value pairs with repeated keys

#### Example

| Step | Pivot | Action                       | Array State     | Partitions    |
| ---- | ----- | ---------------------------- | --------------- | ------------- |
| 0    | 3     | Start                        | [3, 2, 3, 1, 3] | l=0, i=0, g=4 |
| 1    | 3     | a[i]=3 → equal               | [3, 2, 3, 1, 3] | i=1           |
| 2    | 3     | a[i]=2 < 3 → swap(a[l],a[i]) | [2, 3, 3, 1, 3] | l=1, i=2      |
| 3    | 3     | a[i]=3 → equal               | [2, 3, 3, 1, 3] | i=3           |
| 4    | 3     | a[i]=1 < 3 → swap(a[l],a[i]) | [2, 1, 3, 3, 3] | l=2, i=4      |
| 5    | 3     | a[i]=3 → equal               | [2, 1, 3, 3, 3] | Done          |

Now recursively sort left `< pivot` region [2,1], skip the middle `=3` block, and sort right `> pivot` (empty).

### How Does It Work (Plain Language)?

We track three zones using three pointers:

- `lt` (less than region)
- `i` (current element)
- `gt` (greater than region)

Each iteration compares `a[i]` with the pivot:

- If `< pivot`: swap with `lt`, expand both regions
- If `> pivot`: swap with `gt`, shrink right region
- If `= pivot`: move forward

Continue until `i > gt`.
This single pass partitions array into three regions, no need to revisit equals.

#### Step-by-Step Process

| Step | Condition        | Action                       |
| ---- | ---------------- | ---------------------------- |
| 1    | `a[i] < pivot`   | swap(a[i], a[lt]), i++, lt++ |
| 2    | `a[i] > pivot`   | swap(a[i], a[gt]), gt--      |
| 3    | `a[i] == pivot`  | i++                          |
| 4    | Stop when i > gt |                              |

Then recursively sort `[low..lt-1]` and `[gt+1..high]`.

### Tiny Code (Easy Versions)

#### C

```c
#include <stdio.h>

void swap(int *a, int *b) {
    int temp = *a;
    *a = *b;
    *b = temp;
}

void quicksort_3way(int a[], int low, int high) {
    if (low >= high) return;

    int pivot = a[low];
    int lt = low, i = low, gt = high;

    while (i <= gt) {
        if (a[i] < pivot) {
            swap(&a[lt], &a[i]);
            lt++; i++;
        } else if (a[i] > pivot) {
            swap(&a[i], &a[gt]);
            gt--;
        } else {
            i++;
        }
    }

    quicksort_3way(a, low, lt - 1);
    quicksort_3way(a, gt + 1, high);
}

int main(void) {
    int a[] = {3, 2, 3, 1, 3};
    int n = sizeof(a) / sizeof(a[0]);
    quicksort_3way(a, 0, n - 1);
    for (int i = 0; i < n; i++) printf("%d ", a[i]);
    printf("\n");
}
```

#### Python

```python
def quicksort_3way(a, low, high):
    if low >= high:
        return
    pivot = a[low]
    lt, i, gt = low, low, high
    while i <= gt:
        if a[i] < pivot:
            a[lt], a[i] = a[i], a[lt]
            lt += 1
            i += 1
        elif a[i] > pivot:
            a[i], a[gt] = a[gt], a[i]
            gt -= 1
        else:
            i += 1
    quicksort_3way(a, low, lt - 1)
    quicksort_3way(a, gt + 1, high)

arr = [3, 2, 3, 1, 3]
quicksort_3way(arr, 0, len(arr) - 1)
print(arr)
```

### Why It Matters

- Efficient for arrays with duplicates
- Reduces unnecessary recursion and comparisons
- Used in string sorting and key-heavy data
- Generalizes the idea of "partition" to multi-way splitting

### A Gentle Proof (Why It Works)

Standard Quick Sort always divides into two regions, even if all elements equal the pivot, leading to O(n²) on identical elements.

3-Way Quick Sort partitions into three zones:

- `< pivot` (left)
- `= pivot` (middle)
- `> pivot` (right)

The middle zone is skipped from recursion, reducing work dramatically.

If all elements are equal → only one pass O(n).

Expected complexity:
$$
T(n) = O(n \log n)
$$
Worst-case (no duplicates): same as Quick Sort.

| Case      | Duplicates | Complexity      |
| --------- | ---------- | --------------- |
| All equal | O(n)       | One pass only   |
| Many      | O(n log n) | Efficient       |
| None      | O(n log n) | Normal behavior |

### Try It Yourself

1. Sort [3, 2, 3, 1, 3] step by step.
2. Print regions (`lt`, `i`, `gt`) after each iteration.
3. Compare recursion depth with normal Quick Sort.
4. Test input [1, 1, 1, 1].
5. Test input [5, 4, 3, 2, 1].
6. Sort ["apple", "apple", "banana", "apple"].
7. Visualize partitions on paper.
8. Modify to count swaps and comparisons.
9. Implement descending order.
10. Apply to random integers with duplicates (e.g. [1,2,2,2,3,3,1]).

### Test Cases

| Input           | Output          | Notes                |
| --------------- | --------------- | -------------------- |
| [3, 2, 3, 1, 3] | [1, 2, 3, 3, 3] | Duplicates           |
| [5, 4, 3, 2, 1] | [1, 2, 3, 4, 5] | No duplicates        |
| [2, 2, 2, 2]    | [2, 2, 2, 2]    | All equal (O(n))     |
| [1, 3, 1, 3, 1] | [1, 1, 1, 3, 3] | Clustered duplicates |

### Complexity

| Aspect         | Value                                |
| -------------- | ------------------------------------ |
| Time (Best)    | O(n) (all equal)                     |
| Time (Average) | O(n log n)                           |
| Time (Worst)   | O(n log n)                           |
| Space          | O(log n) recursion                   |
| Stable         | No                                   |
| Adaptive       | Yes (handles duplicates efficiently) |

3-Way Quick Sort shows how a small change, three-way partitioning, can transform Quick Sort into a powerful tool for duplicate-heavy datasets, blending elegance with efficiency.

### 119 External Merge Sort

External Merge Sort is a specialized sorting algorithm designed for very large datasets that don't fit entirely into main memory (RAM). It works by sorting chunks of data in memory, writing them to disk, and then merging those sorted chunks. This makes it a key tool in databases, file systems, and big data processing.

### What Problem Are We Solving?

When data exceeds RAM capacity, in-memory sorts like Quick Sort or Heap Sort fail, they need random access to all elements.
External Merge Sort solves this by processing data in blocks:

- Sort manageable chunks in memory
- Write sorted chunks ("runs") to disk
- Merge runs sequentially using streaming I/O

This minimizes disk reads/writes, the main bottleneck in large-scale sorting.

It's ideal for:

- Large files (GBs to TBs)
- Database query engines
- Batch processing pipelines

#### Example

Let's say you have 1 GB of data and only 100 MB of RAM.

| Step | Action | Description                                        |
| ---- | ------ | -------------------------------------------------- |
| 1    | Split  | Divide file into 10 chunks of 100 MB               |
| 2    | Sort   | Load each chunk in memory, sort, write to disk     |
| 3    | Merge  | Use k-way merge (e.g. 10-way) to merge sorted runs |
| 4    | Output | Final sorted file written sequentially             |

### How Does It Work (Plain Language)?

Think of it like sorting pages of a giant book:

1. Take a few pages at a time (fit in memory)
2. Sort them and place them in order piles
3. Combine the piles in order until the whole book is sorted

It's a multi-pass algorithm:

- Pass 1: Create sorted runs
- Pass 2+: Merge runs in multiple passes until one remains

#### Step-by-Step Process

| Step | Description                                      |
| ---- | ------------------------------------------------ |
| 1    | Divide the large file into blocks fitting memory |
| 2    | Load a block, sort it using in-memory sort       |
| 3    | Write each sorted block (run) to disk            |
| 4    | Merge all runs using k-way merging               |
| 5    | Repeat merges until a single sorted file remains |

### Tiny Code (Simplified Simulation)

#### Python (Simulated External Sort)

```python
import heapq
import tempfile

def sort_chunk(chunk):
    chunk.sort()
    temp = tempfile.TemporaryFile(mode="w+t")
    temp.writelines(f"{x}\n" for x in chunk)
    temp.seek(0)
    return temp

def merge_files(files, output_file):
    iters = [map(int, f) for f in files]
    with open(output_file, "w") as out:
        for num in heapq.merge(*iters):
            out.write(f"{num}\n")

def external_merge_sort(input_data, chunk_size=5):
    chunks = []
    for i in range(0, len(input_data), chunk_size):
        chunk = input_data[i:i + chunk_size]
        chunks.append(sort_chunk(chunk))
    merge_files(chunks, "sorted_output.txt")

data = [42, 17, 93, 8, 23, 4, 16, 99, 55, 12, 71, 3]
external_merge_sort(data, chunk_size=4)
```

This example simulates external sorting in Python, splitting input into chunks, sorting each, and merging with `heapq.merge`.

### Why It Matters

- Handles massive datasets beyond memory limits
- Sequential disk I/O (fast and predictable)
- Foundation of database sort-merge joins
- Works well with distributed systems (MapReduce, Spark)

### A Gentle Proof (Why It Works)

Each pass performs O(n) work to read and write the entire dataset.
If `r` is the number of runs, and `k` is merge fan-in (number of runs merged at once):

$$
\text{Number of passes} = \lceil \log_k r \rceil
$$

Total cost ≈
$$
O(n \log_k r)
$$
dominated by I/O operations rather than comparisons.

For `r = n/M` (chunks of memory size `M`), performance is optimized by choosing `k ≈ M`.

| Phase       | Work         | Cost         |
| ----------- | ------------ | ------------ |
| Create Runs | O(n log M)   | Sort chunks  |
| Merge Runs  | O(n log_k r) | Merge passes |

### Try It Yourself

1. Split [42, 17, 93, 8, 23, 4, 16, 99, 55, 12, 71, 3] into 4-element chunks.
2. Sort each chunk individually.
3. Simulate merging sorted runs.
4. Try merging 2-way vs 4-way, count passes.
5. Visualize merging tree (runs combining).
6. Test with random large arrays (simulate files).
7. Modify chunk size and observe performance.
8. Compare I/O counts with in-memory sort.
9. Use `heapq.merge` to merge sorted streams.
10. Extend to merge files on disk (not just lists).

### Test Cases

| Input                       | Memory Limit | Output              | Notes              |
| --------------------------- | ------------ | ------------------- | ------------------ |
| [9, 4, 7, 2, 5, 1, 8, 3, 6] | 3 elements   | [1,2,3,4,5,6,7,8,9] | 3-way merge        |
| 1 GB integers               | 100 MB       | Sorted file         | 10 sorted runs     |
| [1,1,1,1,1]                 | small        | [1,1,1,1,1]         | Handles duplicates |

### Complexity

| Aspect     | Value                                     |
| ---------- | ----------------------------------------- |
| Time       | O(n log_k (n/M))                          |
| Space      | O(M) (memory buffer)                      |
| I/O Passes | O(log_k (n/M))                            |
| Stable     | Yes                                       |
| Adaptive   | Yes (fewer runs if data partially sorted) |

External Merge Sort is the unsung hero behind large-scale sorting, when memory ends, it steps in with disk-based precision, keeping order across terabytes with calm efficiency.

### 120 Parallel Merge Sort

Parallel Merge Sort takes the familiar divide-and-conquer structure of Merge Sort and spreads the work across multiple threads or processors, achieving faster sorting on multi-core CPUs or distributed systems. It's an ideal illustration of how parallelism can amplify a classic algorithm without changing its logic.

### What Problem Are We Solving?

Traditional Merge Sort runs sequentially, so even though its complexity is O(n log n), it uses only one CPU core.
On modern hardware with many cores, that's a waste.

Parallel Merge Sort tackles this by:

- Sorting subarrays in parallel
- Merging results concurrently
- Utilizing full CPU or cluster potential

It's essential for:

- High-performance computing
- Large-scale sorting
- Real-time analytics pipelines

#### Example

Sort [5, 3, 4, 1, 2] using 2 threads:

| Step | Action                      | Threads                          | Result            |
| ---- | --------------------------- | -------------------------------- | ----------------- |
| 1    | Split array into halves     | 2 threads                        | [5, 3, 4], [1, 2] |
| 2    | Sort each half concurrently | T1: sort [5,3,4], T2: sort [1,2] | [3,4,5], [1,2]    |
| 3    | Merge results               | 1 thread                         | [1,2,3,4,5]       |

Parallelism reduces total time roughly by 1 / number of threads (with overhead).

### How Does It Work (Plain Language)?

It's still divide and conquer, just with teamwork:

1. Split array into two halves.
2. Sort each half in parallel.
3. Merge the two sorted halves.
4. Stop splitting when subarrays are small (then sort sequentially).

Each recursive level can launch new threads until you reach a threshold or maximum depth.

#### Step-by-Step Process

| Step | Description                     |
| ---- | ------------------------------- |
| 1    | Divide the array into halves    |
| 2    | Sort both halves concurrently   |
| 3    | Wait for both to finish         |
| 4    | Merge results sequentially      |
| 5    | Repeat recursively for subparts |

This pattern fits well with thread pools, task schedulers, or fork-join frameworks.

### Tiny Code (Easy Versions)

#### C (POSIX Threads Example)

```c
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

typedef struct {
    int *arr;
    int left;
    int right;
} Args;

void merge(int arr[], int l, int m, int r) {
    int n1 = m - l + 1, n2 = r - m;
    int L[n1], R[n2];
    for (int i = 0; i < n1; i++) L[i] = arr[l + i];
    for (int j = 0; j < n2; j++) R[j] = arr[m + 1 + j];

    int i = 0, j = 0, k = l;
    while (i < n1 && j < n2) {
        arr[k++] = (L[i] <= R[j]) ? L[i++] : R[j++];
    }
    while (i < n1) arr[k++] = L[i++];
    while (j < n2) arr[k++] = R[j++];
}

void *parallel_merge_sort(void *arg) {
    Args *args = (Args *)arg;
    int l = args->left, r = args->right;
    int *arr = args->arr;

    if (l < r) {
        int m = l + (r - l) / 2;

        Args leftArgs = {arr, l, m};
        Args rightArgs = {arr, m + 1, r};
        pthread_t leftThread, rightThread;

        pthread_create(&leftThread, NULL, parallel_merge_sort, &leftArgs);
        pthread_create(&rightThread, NULL, parallel_merge_sort, &rightArgs);

        pthread_join(leftThread, NULL);
        pthread_join(rightThread, NULL);

        merge(arr, l, m, r);
    }
    return NULL;
}

int main(void) {
    int arr[] = {5, 3, 4, 1, 2};
    int n = sizeof(arr) / sizeof(arr[0]);
    Args args = {arr, 0, n - 1};
    parallel_merge_sort(&args);
    for (int i = 0; i < n; i++) printf("%d ", arr[i]);
    printf("\n");
}
```

*(Note: This simple version may create too many threads; real implementations limit thread depth.)*

#### Python (Using multiprocessing)

```python
from multiprocessing import Pool

def merge(left, right):
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] < right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result

def parallel_merge_sort(arr):
    if len(arr) <= 1:
        return arr
    if len(arr) < 1000:  # threshold
        return sorted(arr)
    mid = len(arr) // 2
    with Pool(2) as p:
        left, right = p.map(parallel_merge_sort, [arr[:mid], arr[mid:]])
    return merge(left, right)

arr = [5, 3, 4, 1, 2]
print(parallel_merge_sort(arr))
```

### Why It Matters

- Exploits multi-core architectures
- Significantly reduces wall-clock time
- Maintains O(n log n) work
- Great showcase of parallel divide-and-conquer

Used in:

- HPC (High Performance Computing)
- Modern standard libraries (`std::execution::par`)
- Big data frameworks (Spark, Hadoop)

### A Gentle Proof (Why It Works)

Each recursive call sorts n/2 elements, but now in parallel.
Let P = number of processors.

Work (total operations):
$$
T_{work}(n) = O(n \log n)
$$
Span (critical path time):
$$
T_{span}(n) = O(\log^2 n)
$$

Total time ≈
$$
O\left(\frac{n \log n}{P} + \log^2 n\right)
$$

Speedup ≈ P×, limited by synchronization and merge overhead.

| Phase          | Parallelizable | Work           |
| -------------- | -------------- | -------------- |
| Sort subarrays | Yes            | O(n log n / P) |
| Merge          | Partially      | O(n log P)     |

### Try It Yourself

1. Run with 1, 2, 4, 8 threads, compare speed.
2. Print thread IDs at each recursive call.
3. Implement threshold for small subarrays.
4. Merge using parallel merging.
5. Measure CPU utilization during sort.
6. Test with large random list (10⁶ elements).
7. Compare with sequential Merge Sort.
8. Profile with timing tools.
9. Try OpenMP version in C.
10. Extend to distributed nodes (MPI).

### Test Cases

| Input           | Threads | Output      | Notes           |
| --------------- | ------- | ----------- | --------------- |
| [3,2,1]         | 2       | [1,2,3]     | Simple          |
| [5,3,4,1,2]     | 2       | [1,2,3,4,5] | Balanced work   |
| 1e6 random ints | 8       | sorted      | Parallel boost  |
| [1,1,1,1]       | 4       | [1,1,1,1]   | Stable behavior |

### Complexity

| Aspect        | Value                   |
| ------------- | ----------------------- |
| Work          | O(n log n)              |
| Span          | O(log² n)               |
| Parallel Time | O(n log n / P + log² n) |
| Space         | O(n)                    |
| Stable        | Yes                     |
| Adaptive      | No                      |

Parallel Merge Sort is Merge Sort reborn for the multi-core era, the same elegance, now with teamwork. It's how classic algorithms learn to scale with hardware.

## Section 13. Counting and distribution sorts

### 121 Counting Sort

Counting Sort is a non-comparison sorting algorithm that sorts integers (or items mapped to integer keys) by counting occurrences of each value. Instead of comparing elements, it directly uses their values as indices in a counting array. It's fast (O(n + k)), stable, and perfect when the input range is limited and small.

### What Problem Are We Solving?

When keys are integers within a known range, comparison-based sorts (O(n log n)) are overkill.
Counting Sort leverages that limited range to sort in linear time, without any comparisons.

Perfect for:

- Sorting grades (0–100)
- Sorting digits or ASCII codes (0–255)
- Pre-step for Radix Sort or Bucket Sort
- Scenarios where key range ≪ n²

#### Example

Sort array `[4, 2, 2, 8, 3, 3, 1]`

| Step | Description                               | Result                      |
| ---- | ----------------------------------------- | --------------------------- |
| 1    | Find max = 8                              | Range = 0–8                 |
| 2    | Initialize count[9] = [0,0,0,0,0,0,0,0,0] |                             |
| 3    | Count each number                         | count = [0,1,2,2,1,0,0,0,1] |
| 4    | Prefix sum (positions)                    | count = [0,1,3,5,6,6,6,6,7] |
| 5    | Place elements by count                   | [1,2,2,3,3,4,8]             |

The count array tracks the position boundaries for each key.

### How Does It Work (Plain Language)?

Counting Sort doesn't compare elements. It counts how many times each value appears, then uses those counts to reconstruct the sorted list.

Think of it as filling labeled bins:

- One bin for each number
- Drop each element into its bin
- Then walk through bins in order and empty them

#### Step-by-Step Process

| Step | Description                                                 |
| ---- | ----------------------------------------------------------- |
| 1    | Find min and max (determine range `k`)                      |
| 2    | Create count array of size `k + 1`                          |
| 3    | Count occurrences of each value                             |
| 4    | Transform counts into prefix sums (for positions)           |
| 5    | Traverse input in reverse (for stability), placing elements |
| 6    | Copy sorted output back                                     |

### Tiny Code (Easy Versions)

#### C

```c
#include <stdio.h>
#include <string.h>

void counting_sort(int arr[], int n) {
    int max = arr[0];
    for (int i = 1; i < n; i++)
        if (arr[i] > max) max = arr[i];

    int count[max + 1];
    memset(count, 0, sizeof(count));

    for (int i = 0; i < n; i++)
        count[arr[i]]++;

    for (int i = 1; i <= max; i++)
        count[i] += count[i - 1];

    int output[n];
    for (int i = n - 1; i >= 0; i--) {
        output[count[arr[i]] - 1] = arr[i];
        count[arr[i]]--;
    }

    for (int i = 0; i < n; i++)
        arr[i] = output[i];
}

int main(void) {
    int arr[] = {4, 2, 2, 8, 3, 3, 1};
    int n = sizeof(arr) / sizeof(arr[0]);
    counting_sort(arr, n);
    for (int i = 0; i < n; i++) printf("%d ", arr[i]);
    printf("\n");
}
```

#### Python

```python
def counting_sort(arr):
    max_val = max(arr)
    count = [0] * (max_val + 1)

    for num in arr:
        count[num] += 1

    for i in range(1, len(count)):
        count[i] += count[i - 1]

    output = [0] * len(arr)
    for num in reversed(arr):
        count[num] -= 1
        output[count[num]] = num

    return output

arr = [4, 2, 2, 8, 3, 3, 1]
print(counting_sort(arr))
```

### Why It Matters

- Linear time when range is small (`O(n + k)`)
- Stable, preserving input order
- Foundation for Radix Sort
- Great for integer, digit, or bucket sorting

### A Gentle Proof (Why It Works)

Counting Sort replaces comparison by index-based placement.

If `n` is number of elements and `k` is key range:

- Counting occurrences: O(n)
- Prefix sums: O(k)
- Placement: O(n)

Total = O(n + k)

Stable because we traverse input in reverse while placing.

| Phase          | Work | Complexity   |
| -------------- | ---- | ------------ |
| Count elements | O(n) | Scan once    |
| Prefix sum     | O(k) | Range pass   |
| Place elements | O(n) | Stable write |

### Try It Yourself

1. Sort `[4, 2, 2, 8, 3, 3, 1]` step by step.
2. Show count array after counting.
3. Convert count to prefix sums.
4. Place elements in output (reverse scan).
5. Compare stable vs unstable version.
6. Change input to `[9, 9, 1, 2]`.
7. Try sorting `[5, 3, 5, 1, 0]`.
8. Handle input with min > 0 (offset counts).
9. Measure runtime vs Bubble Sort.
10. Use as subroutine in Radix Sort.

### Test Cases

| Input           | Output          | Notes          |
| --------------- | --------------- | -------------- |
| [4,2,2,8,3,3,1] | [1,2,2,3,3,4,8] | Example        |
| [1,4,1,2,7,5,2] | [1,1,2,2,4,5,7] | Stable         |
| [9,9,9,9]       | [9,9,9,9]       | Repeats        |
| [0,1,2,3]       | [0,1,2,3]       | Already sorted |

### Complexity

| Aspect          | Value    |
| --------------- | -------- |
| Time            | O(n + k) |
| Space           | O(n + k) |
| Stable          | Yes      |
| Adaptive        | No       |
| Range-sensitive | Yes      |

Counting Sort is like sorting by bins, no comparisons, no stress, just clean counts and linear time. It's a powerhouse behind Radix Sort and data bucketing in performance-critical pipelines.

### 122 Stable Counting Sort

Stable Counting Sort refines the basic Counting Sort by ensuring equal elements preserve their original order. This property, called *stability*, is crucial when sorting multi-key data, for example, sorting people by age, then by name. Stable versions are also the building blocks for Radix Sort, where each digit's sort depends on stability.

### What Problem Are We Solving?

Basic Counting Sort can break order among equal elements because it places them in arbitrary order.
When sorting records or tuples where order matters (e.g., by secondary key), we need stability, if `a` and `b` have equal keys, their order in output must match input.

Stable Counting Sort ensures that:

> If `arr[i]` and `arr[j]` have the same key and `i < j`,
> then `arr[i]` appears before `arr[j]` in the sorted output.

Perfect for:

- Radix Sort digits
- Multi-field records (e.g. sort by name, then by score)
- Databases and stable pipelines

#### Example

Sort `[4a, 2b, 2a, 8a, 3b, 3a, 1a]` (letters mark order)

| Step | Description                   | Result                                |
| ---- | ----------------------------- | ------------------------------------- |
| 1    | Count frequencies             | count = [0,1,2,2,1,0,0,0,1]           |
| 2    | Prefix sums (positions)       | count = [0,1,3,5,6,6,6,6,7]           |
| 3    | Traverse input in reverse | output = [1a, 2b, 2a, 3b, 3a, 4a, 8a] |

See how `2b` (index 1) appears before `2a` (index 2), stable ordering preserved.

### How Does It Work (Plain Language)?

It's Counting Sort with a twist: we fill the output from the end of the input, ensuring last-seen equal items go last.
By traversing in reverse, earlier elements are placed later, preserving their original order.

This is the core idea behind stable sorting.

#### Step-by-Step Process

| Step | Description                                   |
| ---- | --------------------------------------------- |
| 1    | Determine key range (0..max)                  |
| 2    | Count frequency of each key                   |
| 3    | Compute prefix sums (to determine positions)  |
| 4    | Traverse input right to left              |
| 5    | Place elements in output using count as index |
| 6    | Decrement count[key] after placement          |

### Tiny Code (Easy Versions)

#### C

```c
#include <stdio.h>
#include <string.h>

typedef struct {
    int key;
    char tag; // to visualize stability
} Item;

void stable_counting_sort(Item arr[], int n) {
    int max = arr[0].key;
    for (int i = 1; i < n; i++)
        if (arr[i].key > max) max = arr[i].key;

    int count[max + 1];
    memset(count, 0, sizeof(count));

    // Count occurrences
    for (int i = 0; i < n; i++)
        count[arr[i].key]++;

    // Prefix sums
    for (int i = 1; i <= max; i++)
        count[i] += count[i - 1];

    Item output[n];

    // Traverse input in reverse for stability
    for (int i = n - 1; i >= 0; i--) {
        int k = arr[i].key;
        output[count[k] - 1] = arr[i];
        count[k]--;
    }

    for (int i = 0; i < n; i++)
        arr[i] = output[i];
}

int main(void) {
    Item arr[] = {{4,'a'},{2,'b'},{2,'a'},{8,'a'},{3,'b'},{3,'a'},{1,'a'}};
    int n = sizeof(arr)/sizeof(arr[0]);
    stable_counting_sort(arr, n);
    for (int i = 0; i < n; i++) printf("(%d,%c) ", arr[i].key, arr[i].tag);
    printf("\n");
}
```

#### Python

```python
def stable_counting_sort(arr):
    max_val = max(arr)
    count = [0] * (max_val + 1)

    for num in arr:
        count[num] += 1

    for i in range(1, len(count)):
        count[i] += count[i - 1]

    output = [0] * len(arr)
    for num in reversed(arr):
        count[num] -= 1
        output[count[num]] = num

    return output

arr = [4, 2, 2, 8, 3, 3, 1]
print(stable_counting_sort(arr))
```

### Why It Matters

- Stable sorting is essential for multi-key operations
- Required for Radix Sort correctness
- Guarantees consistent behavior for duplicates
- Used in databases, language sort libraries, pipelines

### A Gentle Proof (Why It Works)

Each key is assigned a position range via prefix sums.
Traversing input from right to left ensures that earlier items occupy smaller indices, preserving order.

If `a` appears before `b` in input and `key(a) = key(b)`,
then `count[key(a)]` places `a` before `b`, stable.

| Phase       | Work | Complexity        |
| ----------- | ---- | ----------------- |
| Counting    | O(n) | Pass once         |
| Prefix sums | O(k) | Range pass        |
| Placement   | O(n) | Reverse traversal |

Total = O(n + k), same as basic Counting Sort, but stable.

### Try It Yourself

1. Sort `[(4,'a'), (2,'b'), (2,'a'), (3,'a')]`.
2. Show count and prefix arrays.
3. Traverse input from end, track output.
4. Compare with unstable version.
5. Try sorting `[5,3,5,1,0]`.
6. Visualize stability when equal keys appear.
7. Modify to handle offset keys (negative values).
8. Combine with Radix Sort (LSD).
9. Profile runtime vs normal Counting Sort.
10. Check stability by adding tags (letters).

### Test Cases

| Input             | Output            | Notes            |
| ----------------- | ----------------- | ---------------- |
| [4,2,2,8,3,3,1]   | [1,2,2,3,3,4,8]   | Same as basic    |
| [(2,'a'),(2,'b')] | [(2,'a'),(2,'b')] | Stable preserved |
| [1,1,1]           | [1,1,1]           | Idempotent       |
| [0,1,2,3]         | [0,1,2,3]         | Already sorted   |

### Complexity

| Aspect          | Value    |
| --------------- | -------- |
| Time            | O(n + k) |
| Space           | O(n + k) |
| Stable          | Yes      |
| Adaptive        | No       |
| Range-sensitive | Yes      |

Stable Counting Sort is Counting Sort with memory, it not only sorts fast but also remembers your order, making it indispensable for multi-pass algorithms like Radix Sort.

### 123 Radix Sort (LSD)

Radix Sort (Least Significant Digit first) is a non-comparison, stable sorting algorithm that processes integers (or strings) digit by digit, starting from the least significant digit (LSD). By repeatedly applying a stable sort (like Counting Sort) on each digit, it can sort numbers in linear time when digit count is small.

### What Problem Are We Solving?

When sorting integers or fixed-length keys (like dates, IDs, or strings of digits), traditional comparison-based sorts spend unnecessary effort.
Radix Sort (LSD) sidesteps comparisons by leveraging digit-wise order and stability to achieve O(d × (n + k)) performance.

Perfect for:

- Sorting numbers, dates, zip codes, or strings
- Datasets with bounded digit length
- Applications where deterministic performance matters

#### Example

Sort `[170, 45, 75, 90, 802, 24, 2, 66]`

| Pass | Digit Place | Input                      | Output (Stable Sort)       |
| ---- | ----------- | -------------------------- | -------------------------- |
| 1    | Ones        | [170,45,75,90,802,24,2,66] | [170,90,802,2,24,45,75,66] |
| 2    | Tens        | [170,90,802,2,24,45,75,66] | [802,2,24,45,66,170,75,90] |
| 3    | Hundreds    | [802,2,24,45,66,170,75,90] | [2,24,45,66,75,90,170,802] |

Final sorted output: `[2, 24, 45, 66, 75, 90, 170, 802]`

Each pass uses Stable Counting Sort on the current digit.

### How Does It Work (Plain Language)?

Think of sorting by digit positions:

1. Group by ones place (units)
2. Group by tens
3. Group by hundreds, etc.

Each pass reorders elements according to the digit at that place, while keeping earlier digit orders intact (thanks to stability).

It's like sorting by last name, then first name, one field at a time, stable each round.

#### Step-by-Step Process

| Step | Description                                              |
| ---- | -------------------------------------------------------- |
| 1    | Find the maximum number to know the number of digits `d` |
| 2    | For each digit place (1, 10, 100, …):                    |
| 3    | Use a stable counting sort based on that digit       |
| 4    | After the last pass, the array is fully sorted           |

### Tiny Code (Easy Versions)

#### C

```c
#include <stdio.h>

int get_max(int a[], int n) {
    int max = a[0];
    for (int i = 1; i < n; i++)
        if (a[i] > max) max = a[i];
    return max;
}

void counting_sort_digit(int a[], int n, int exp) {
    int output[n];
    int count[10] = {0};

    for (int i = 0; i < n; i++)
        count[(a[i] / exp) % 10]++;

    for (int i = 1; i < 10; i++)
        count[i] += count[i - 1];

    for (int i = n - 1; i >= 0; i--) {
        int digit = (a[i] / exp) % 10;
        output[count[digit] - 1] = a[i];
        count[digit]--;
    }

    for (int i = 0; i < n; i++)
        a[i] = output[i];
}

void radix_sort(int a[], int n) {
    int max = get_max(a, n);
    for (int exp = 1; max / exp > 0; exp *= 10)
        counting_sort_digit(a, n, exp);
}

int main(void) {
    int a[] = {170, 45, 75, 90, 802, 24, 2, 66};
    int n = sizeof(a) / sizeof(a[0]);
    radix_sort(a, n);
    for (int i = 0; i < n; i++) printf("%d ", a[i]);
    printf("\n");
}
```

#### Python

```python
def counting_sort_digit(arr, exp):
    n = len(arr)
    output = [0] * n
    count = [0] * 10

    for num in arr:
        index = (num // exp) % 10
        count[index] += 1

    for i in range(1, 10):
        count[i] += count[i - 1]

    for num in reversed(arr):
        index = (num // exp) % 10
        count[index] -= 1
        output[count[index]] = num

    for i in range(n):
        arr[i] = output[i]

def radix_sort(arr):
    max_val = max(arr)
    exp = 1
    while max_val // exp > 0:
        counting_sort_digit(arr, exp)
        exp *= 10

arr = [170, 45, 75, 90, 802, 24, 2, 66]
radix_sort(arr)
print(arr)
```

### Why It Matters

- Linear time (O(d × (n + k))) for fixed digits
- Stable, retains order for equal keys
- Great for large numeric datasets
- Foundation for efficient key-based sorting (strings, dates)

### A Gentle Proof (Why It Works)

At each digit position:

- Stable Counting Sort reorders by that digit
- Earlier digits remain ordered (stability)
- After all digits, array is fully ordered

If each digit has range `k` and `d` total digits:

$$
T(n) = O(d \times (n + k))
$$

| Phase      | Work         | Complexity    |
| ---------- | ------------ | ------------- |
| Per digit  | O(n + k)     | Counting sort |
| All digits | d × O(n + k) | Total         |

If `d` and `k` are constants → O(n) overall.

### Try It Yourself

1. Sort `[170, 45, 75, 90, 802, 24, 2, 66]`.
2. Trace each pass (ones, tens, hundreds).
3. Show count table per digit.
4. Compare stable vs unstable sorting.
5. Add zeros: `[07, 70, 700]`.
6. Try `[3, 1, 2, 10, 11, 21]`.
7. Count digit comparisons.
8. Modify to handle negative numbers (offset).
9. Change base to 16 (hex).
10. Compare with Merge Sort performance on large input.

### Test Cases

| Input                      | Output                     | Notes             |
| -------------------------- | -------------------------- | ----------------- |
| [170,45,75,90,802,24,2,66] | [2,24,45,66,75,90,170,802] | Classic           |
| [9,8,7,6,5]                | [5,6,7,8,9]                | Reversed          |
| [10,1,100,1000]            | [1,10,100,1000]            | Different lengths |
| [22,22,11,11]              | [11,11,22,22]              | Stable            |

### Complexity

| Aspect          | Value          |
| --------------- | -------------- |
| Time            | O(d × (n + k)) |
| Space           | O(n + k)       |
| Stable          | Yes            |
| Adaptive        | No             |
| Range-sensitive | Yes            |

Radix Sort (LSD) is the assembly line of sorting, each pass builds upon the last, producing perfectly ordered output from simple stable steps.

### 124 Radix Sort (MSD)

Radix Sort (Most Significant Digit first) is a recursive variant of Radix Sort that begins sorting from the most significant digit (MSD) and works downward. Unlike LSD Radix Sort, which is iterative and stable across all digits, MSD focuses on prefix-based grouping and recursively sorts subgroups. This makes it ideal for variable-length keys such as strings, IP addresses, or long integers.

### What Problem Are We Solving?

LSD Radix Sort works best for fixed-length keys, where every element has the same number of digits.
But when keys differ in length (e.g., strings "a", "ab", "abc"), we need to respect prefix order, "a" should come before "ab".

MSD Radix Sort handles this by grouping by prefix digits, then recursively sorting each group.

Perfect for:

- Strings, words, or variable-length keys
- Hierarchical data (prefix-sensitive)
- Lexicographic ordering (dictionary order)

#### Example

Sort: `["b", "ba", "abc", "ab", "ac"]`

| Step | Digit       | Groups                                   |
| ---- | ----------- | ---------------------------------------- |
| 1    | 1st char    | a → ["abc", "ab", "ac"], b → ["b", "ba"] |
| 2    | Group "a"   | 2nd char → b: ["abc", "ab"], c: ["ac"]   |
| 3    | Group "ab"  | 3rd char → c: ["abc"], end: ["ab"]       |
| 4    | Final merge | ["ab", "abc", "ac", "b", "ba"]           |

Lexicographic order preserved, even with varying lengths.

### How Does It Work (Plain Language)?

MSD Radix Sort organizes data by prefix trees (tries) conceptually:

- Partition elements by their most significant digit (or character)
- Recurse within each group for next digit
- Merge groups in order of digit values

If LSD is like bucket sorting digits from the back, MSD is tree-like sorting from the top.

#### Step-by-Step Process

| Step | Description                                   |
| ---- | --------------------------------------------- |
| 1    | Find highest digit place (or first character) |
| 2    | Partition array into groups by that digit     |
| 3    | Recursively sort each group by next digit     |
| 4    | Concatenate groups in order                   |

For strings, if one string ends early, it's considered smaller.

### Tiny Code (Easy Versions)

#### Python (String Example)

```python
def msd_radix_sort(arr, pos=0):
    if len(arr) <= 1:
        return arr

    # Buckets for ASCII range (0-255) + 1 for end-of-string
    buckets = [[] for _ in range(257)]
    for word in arr:
        index = ord(word[pos]) + 1 if pos < len(word) else 0
        buckets[index].append(word)

    result = []
    for bucket in buckets:
        if bucket:
            # Only recurse if there's more than one element and not EOS
            if len(bucket) > 1 and (pos < max(len(w) for w in bucket)):
                bucket = msd_radix_sort(bucket, pos + 1)
            result.extend(bucket)
    return result

arr = ["b", "ba", "abc", "ab", "ac"]
print(msd_radix_sort(arr))
```

Output:

```
$$'ab', 'abc', 'ac', 'b', 'ba']
```

#### C (Numeric Example)

```c
#include <stdio.h>

int get_digit(int num, int exp, int base) {
    return (num / exp) % base;
}

void msd_radix_sort_rec(int arr[], int n, int exp, int base, int max) {
    if (exp == 0 || n <= 1) return;

    int buckets[base][n];
    int count[base];
    for (int i = 0; i < base; i++) count[i] = 0;

    // Distribute
    for (int i = 0; i < n; i++) {
        int d = get_digit(arr[i], exp, base);
        buckets[d][count[d]++] = arr[i];
    }

    // Recurse and collect
    int idx = 0;
    for (int i = 0; i < base; i++) {
        if (count[i] > 0) {
            msd_radix_sort_rec(buckets[i], count[i], exp / base, base, max);
            for (int j = 0; j < count[i]; j++)
                arr[idx++] = buckets[i][j];
        }
    }
}

void msd_radix_sort(int arr[], int n) {
    int max = arr[0];
    for (int i = 1; i < n; i++)
        if (arr[i] > max) max = arr[i];

    int exp = 1;
    while (max / exp >= 10) exp *= 10;

    msd_radix_sort_rec(arr, n, exp, 10, max);
}

int main(void) {
    int arr[] = {170, 45, 75, 90, 802, 24, 2, 66};
    int n = sizeof(arr) / sizeof(arr[0]);
    msd_radix_sort(arr, n);
    for (int i = 0; i < n; i++) printf("%d ", arr[i]);
    printf("\n");
}
```

### Why It Matters

- Handles variable-length keys
- Natural for lexicographic ordering
- Used in string sorting, trie-based systems, suffix array construction
- Recursively partitions, often faster for large diverse keys

### A Gentle Proof (Why It Works)

Each recursive call partitions the array by digit prefix.
Since partitions are disjoint and ordered by digit, concatenating them yields a fully sorted sequence.

For `n` elements, `d` digits, and base `k`:

$$
T(n) = O(n + k) \text{ per level, depth } \le d
$$
$$
\Rightarrow O(d \times (n + k))
$$

Stability preserved via ordered grouping.

| Phase     | Work        | Description                    |
| --------- | ----------- | ------------------------------ |
| Partition | O(n)        | Place items in digit buckets   |
| Recurse   | O(d)        | Each level processes subgroups |
| Total     | O(d(n + k)) | Linear in digits               |

### Try It Yourself

1. Sort `["b", "ba", "abc", "ab", "ac"]`.
2. Draw recursion tree by character.
3. Compare order with lexicographic.
4. Test `["dog", "cat", "apple", "apricot"]`.
5. Sort integers `[170,45,75,90,802,24,2,66]`.
6. Change base (binary, hex).
7. Compare with LSD Radix Sort.
8. Add duplicates and test stability.
9. Visualize grouping buckets.
10. Implement with trie-like data structure.

### Test Cases

| Input                      | Output                     | Notes           |
| -------------------------- | -------------------------- | --------------- |
| ["b","ba","abc","ab","ac"] | ["ab","abc","ac","b","ba"] | Variable length |
| [170,45,75,90,802,24,2,66] | [2,24,45,66,75,90,170,802] | Numeric         |
| ["a","aa","aaa"]           | ["a","aa","aaa"]           | Prefix order    |
| ["z","y","x"]              | ["x","y","z"]              | Reverse input   |

### Complexity

| Aspect       | Value                |
| ------------ | -------------------- |
| Time         | O(d × (n + k))       |
| Space        | O(n + k)             |
| Stable       | Yes                  |
| Adaptive     | No                   |
| Suitable for | Variable-length keys |

Radix Sort (MSD) is lexicographic sorting by recursion, it builds order from the top down, treating prefixes as leaders and details as followers, much like how dictionaries arrange words.

### 125 Bucket Sort

Bucket Sort is a distribution-based sorting algorithm that divides the input into several buckets (bins), sorts each bucket individually (often with Insertion Sort), and then concatenates them. When input data is uniformly distributed, Bucket Sort achieves linear time performance (O(n)).

### What Problem Are We Solving?

Comparison-based sorts take O(n log n) time in the general case. But if we know that data values are spread evenly across a range, we can exploit this structure to sort faster by grouping similar values together.

Bucket Sort works best when:

- Input is real numbers in [0, 1) or any known range
- Data is uniformly distributed
- Buckets are balanced, each with few elements

Used in:

- Probability distributions
- Histogram-based sorting
- Floating-point sorting

#### Example

Sort `[0.78, 0.17, 0.39, 0.26, 0.72, 0.94, 0.21, 0.12, 0.23, 0.68]`

| Step | Action                                  | Result                                                       |
| ---- | --------------------------------------- | ------------------------------------------------------------ |
| 1    | Create 10 buckets for range [0, 1)      | [[] ... []]                                                  |
| 2    | Distribute elements by `int(n * value)` | [[0.12,0.17,0.21,0.23,0.26],[0.39],[0.68,0.72,0.78],[0.94]]  |
| 3    | Sort each bucket (Insertion Sort)       | Each bucket sorted individually                              |
| 4    | Concatenate buckets                     | [0.12, 0.17, 0.21, 0.23, 0.26, 0.39, 0.68, 0.72, 0.78, 0.94] |

### How Does It Work (Plain Language)?

Think of sorting test scores:

- You group scores into bins (0–10, 10–20, 20–30, …)
- Sort each bin individually
- Merge bins back together in order

Bucket Sort leverages range grouping, local order inside each bucket, global order from bucket sequence.

#### Step-by-Step Process

| Step | Description                                  |
| ---- | -------------------------------------------- |
| 1    | Create empty buckets for each range interval |
| 2    | Distribute elements into buckets             |
| 3    | Sort each bucket individually                |
| 4    | Concatenate buckets sequentially             |

If buckets are evenly filled, each small sort is fast, almost constant time.

### Tiny Code (Easy Versions)

#### C

```c
#include <stdio.h>
#include <stdlib.h>

void insertion_sort(float arr[], int n) {
    for (int i = 1; i < n; i++) {
        float key = arr[i];
        int j = i - 1;
        while (j >= 0 && arr[j] > key) {
            arr[j + 1] = arr[j];
            j--;
        }
        arr[j + 1] = key;
    }
}

void bucket_sort(float arr[], int n) {
    float buckets[n][n];
    int count[n];
    for (int i = 0; i < n; i++) count[i] = 0;

    // Distribute into buckets
    for (int i = 0; i < n; i++) {
        int idx = n * arr[i]; // index by range
        buckets[idx][count[idx]++] = arr[i];
    }

    // Sort each bucket
    for (int i = 0; i < n; i++)
        if (count[i] > 0)
            insertion_sort(buckets[i], count[i]);

    // Concatenate
    int k = 0;
    for (int i = 0; i < n; i++)
        for (int j = 0; j < count[i]; j++)
            arr[k++] = buckets[i][j];
}

int main(void) {
    float arr[] = {0.78,0.17,0.39,0.26,0.72,0.94,0.21,0.12,0.23,0.68};
    int n = sizeof(arr) / sizeof(arr[0]);
    bucket_sort(arr, n);
    for (int i = 0; i < n; i++) printf("%.2f ", arr[i]);
    printf("\n");
}
```

#### Python

```python
def bucket_sort(arr):
    n = len(arr)
    buckets = [[] for _ in range(n)]

    for num in arr:
        idx = int(n * num)
        buckets[idx].append(num)

    for bucket in buckets:
        bucket.sort()

    result = []
    for bucket in buckets:
        result.extend(bucket)
    return result

arr = [0.78,0.17,0.39,0.26,0.72,0.94,0.21,0.12,0.23,0.68]
print(bucket_sort(arr))
```

### Why It Matters

- Linear time for uniformly distributed data
- Great for floating-point numbers
- Illustrates distribution-based sorting
- Foundation for histogram, flash, and spread sort

### A Gentle Proof (Why It Works)

If input elements are independent and uniformly distributed, expected elements per bucket = `O(1)`.
Sorting each small bucket takes constant time → total linear time.

Total complexity:

$$
T(n) = O(n + \sum_{i=1}^{n} T_i)
$$
If each `T_i = O(1)`,
$$
T(n) = O(n)
$$

| Phase        | Work       | Complexity |
| ------------ | ---------- | ---------- |
| Distribution | O(n)       | One pass   |
| Local Sort   | O(n) total | Expected   |
| Concatenate  | O(n)       | Combine    |

### Try It Yourself

1. Sort `[0.78,0.17,0.39,0.26,0.72,0.94,0.21,0.12,0.23,0.68]`.
2. Visualize buckets as bins.
3. Change bucket count to 5, 20, see effect.
4. Try non-uniform data `[0.99,0.99,0.98]`.
5. Replace insertion sort with counting sort.
6. Measure performance with 10⁶ floats.
7. Test `[0.1,0.01,0.001]`, uneven distribution.
8. Implement bucket indexing for arbitrary ranges.
9. Compare with Quick Sort runtime.
10. Plot distribution histogram before sorting.

### Test Cases

| Input                 | Output                | Notes      |
| --------------------- | --------------------- | ---------- |
| [0.78,0.17,0.39,0.26] | [0.17,0.26,0.39,0.78] | Basic      |
| [0.1,0.01,0.001]      | [0.001,0.01,0.1]      | Sparse     |
| [0.9,0.8,0.7,0.6]     | [0.6,0.7,0.8,0.9]     | Reverse    |
| [0.1,0.1,0.1]         | [0.1,0.1,0.1]         | Duplicates |

### Complexity

| Aspect         | Value                          |
| -------------- | ------------------------------ |
| Time (Best)    | O(n)                           |
| Time (Average) | O(n)                           |
| Time (Worst)   | O(n²) (all in one bucket)      |
| Space          | O(n)                           |
| Stable         | Yes (if bucket sort is stable) |
| Adaptive       | Yes (depends on distribution)  |

Bucket Sort is like sorting by bins, fast, simple, and beautifully efficient when your data is evenly spread. It's the go-to choice for continuous values in a fixed range.

### 126 Pigeonhole Sort

Pigeonhole Sort is a simple distribution sorting algorithm that places each element directly into its corresponding "pigeonhole" (or bucket) based on its key value. It's ideal when elements are integers within a small known range, think of it as Counting Sort with explicit placement rather than counting.

### What Problem Are We Solving?

When data values are integers and close together, we don't need comparisons, we can map each value to a slot directly.
Pigeonhole Sort is particularly useful for dense integer ranges, such as:

- Sorting integers from 0 to 100
- Sorting scores, ranks, or IDs
- Small ranges with many duplicates

It trades space for speed, achieving O(n + range) performance.

#### Example

Sort `[8, 3, 2, 7, 4, 6, 8]`

| Step | Description                                  | Result                                |
| ---- | -------------------------------------------- | ------------------------------------- |
| 1    | Find min=2, max=8 → range = 7                | holes[0..6]                           |
| 2    | Create pigeonholes: `[[],[],[],[],[],[],[]]` |                                       |
| 3    | Place each number in its hole                | holes = `[[2],[3],[4],[6],[7],[8,8]]` |
| 4    | Concatenate holes                            | `[2,3,4,6,7,8,8]`                     |

Each value goes exactly to its mapped hole index.

### How Does It Work (Plain Language)?

It's like assigning students to exam rooms based on ID ranges, each slot holds all matching IDs.
You fill slots (holes), then read them back in order.

Unlike Counting Sort, which only counts occurrences, Pigeonhole Sort stores the actual values, preserving duplicates directly.

#### Step-by-Step Process

| Step | Description                                       |
| ---- | ------------------------------------------------- |
| 1    | Find minimum and maximum elements                 |
| 2    | Compute range = max - min + 1                     |
| 3    | Create array of empty pigeonholes of size `range` |
| 4    | For each element, map to hole `arr[i] - min`      |
| 5    | Place element into that hole (append)             |
| 6    | Read holes in order and flatten into output       |

### Tiny Code (Easy Versions)

#### C

```c
#include <stdio.h>
#include <stdlib.h>

void pigeonhole_sort(int arr[], int n) {
    int min = arr[0], max = arr[0];
    for (int i = 1; i < n; i++) {
        if (arr[i] < min) min = arr[i];
        if (arr[i] > max) max = arr[i];
    }

    int range = max - min + 1;
    int *holes[range];
    int counts[range];
    for (int i = 0; i < range; i++) {
        holes[i] = malloc(n * sizeof(int));
        counts[i] = 0;
    }

    // Place elements into holes
    for (int i = 0; i < n; i++) {
        int index = arr[i] - min;
        holes[index][counts[index]++] = arr[i];
    }

    // Flatten back
    int index = 0;
    for (int i = 0; i < range; i++) {
        for (int j = 0; j < counts[i]; j++) {
            arr[index++] = holes[i][j];
        }
        free(holes[i]);
    }
}

int main(void) {
    int arr[] = {8, 3, 2, 7, 4, 6, 8};
    int n = sizeof(arr) / sizeof(arr[0]);
    pigeonhole_sort(arr, n);
    for (int i = 0; i < n; i++) printf("%d ", arr[i]);
    printf("\n");
}
```

#### Python

```python
def pigeonhole_sort(arr):
    min_val, max_val = min(arr), max(arr)
    size = max_val - min_val + 1
    holes = [[] for _ in range(size)]

    for x in arr:
        holes[x - min_val].append(x)

    sorted_arr = []
    for hole in holes:
        sorted_arr.extend(hole)
    return sorted_arr

arr = [8, 3, 2, 7, 4, 6, 8]
print(pigeonhole_sort(arr))
```

### Why It Matters

- Simple mapping for small integer ranges
- Linear time if range ≈ n
- Useful in digit, rank, or ID sorting
- Provides stable grouping with explicit placement

### A Gentle Proof (Why It Works)

Every key is mapped uniquely to a hole (offset by min).
All duplicates fall into the same hole, preserving multiplicity.
Reading holes sequentially yields sorted order.

If `n` = number of elements, `k` = range of values:

$$
T(n) = O(n + k)
$$

| Phase               | Work     | Complexity         |
| ------------------- | -------- | ------------------ |
| Find min/max        | O(n)     | Single pass        |
| Distribute to holes | O(n)     | One placement each |
| Collect results     | O(n + k) | Flatten all        |

### Try It Yourself

1. Sort `[8,3,2,7,4,6,8]` by hand.
2. Show hole contents after distribution.
3. Add duplicate values, confirm stable grouping.
4. Try `[1,1,1,1]`, all in one hole.
5. Sort negative numbers `[0,-1,-2,1]` (offset by min).
6. Increase range to see space cost.
7. Compare runtime with Counting Sort.
8. Replace holes with linked lists.
9. Implement in-place version.
10. Extend for key-value pairs.

### Test Cases

| Input           | Output          | Notes           |
| --------------- | --------------- | --------------- |
| [8,3,2,7,4,6,8] | [2,3,4,6,7,8,8] | Example         |
| [1,1,1,1]       | [1,1,1,1]       | All equal       |
| [9,8,7,6]       | [6,7,8,9]       | Reverse         |
| [0,-1,-2,1]     | [-2,-1,0,1]     | Negative offset |

### Complexity

| Aspect          | Value    |
| --------------- | -------- |
| Time            | O(n + k) |
| Space           | O(n + k) |
| Stable          | Yes      |
| Adaptive        | No       |
| Range-sensitive | Yes      |

Pigeonhole Sort is as direct as sorting gets, one slot per value, one pass in, one pass out. Fast and clean when your data is dense and discrete.

### 127 Flash Sort

Flash Sort is a distribution-based sorting algorithm that combines ideas from bucket sort and insertion sort. It works in two phases:

1. Classification, distribute elements into classes (buckets) using a linear mapping
2. Permutation, rearrange elements in-place using cycles

It achieves O(n) average time on uniformly distributed data but can degrade to O(n²) in the worst case.
Invented by *Karl-Dietrich Neubert* (1990s), it's known for being extremely fast in practice on large datasets.

### What Problem Are We Solving?

When data is numerically distributed over a range, we can approximate where each element should go and move it close to its final position without full comparison sorting.

Flash Sort is built for:

- Uniformly distributed numeric data
- Large arrays
- Performance-critical applications (e.g., simulations, physics, graphics)

It leverages approximate indexing and in-place permutation for speed.

#### Example

Sort `[9, 3, 1, 7, 4, 6, 2, 8, 5]` into `m = 5` classes.

| Step | Description                                              | Result                             |
| ---- | -------------------------------------------------------- | ---------------------------------- |
| 1    | Find `min = 1`, `max = 9`                                | range = 8                          |
| 2    | Map each value to class `k = (m-1)*(a[i]-min)/(max-min)` | Class indices: [4,1,0,3,1,3,0,4,2] |
| 3    | Count elements per class, compute cumulative positions   | class boundaries = [0,2,4,6,8,9]   |
| 4    | Cycle elements into correct class positions              | Rearranged approx order            |
| 5    | Apply insertion sort within classes                      | Sorted list                        |

Final: `[1,2,3,4,5,6,7,8,9]`

### How Does It Work (Plain Language)?

Imagine flashing each element to its approximate destination class in one pass, that's the "flash" phase.
Then, fine-tune within each class using a simpler sort (like Insertion Sort).

It's like placing books roughly into shelves, then tidying each shelf.

#### Step-by-Step Process

| Step | Description                                                |
| ---- | ---------------------------------------------------------- |
| 1    | Find min and max                                           |
| 2    | Choose number of classes (usually ≈ 0.43 * n)              |
| 3    | Compute class index for each element                       |
| 4    | Count elements per class and compute prefix sums           |
| 5    | Move elements to approximate class positions (flash phase) |
| 6    | Use Insertion Sort within each class to finish             |

### Tiny Code (Easy Versions)

#### C

```c
#include <stdio.h>

void insertion_sort(float arr[], int start, int end) {
    for (int i = start + 1; i <= end; i++) {
        float key = arr[i];
        int j = i - 1;
        while (j >= start && arr[j] > key) {
            arr[j + 1] = arr[j];
            j--;
        }
        arr[j + 1] = key;
    }
}

void flash_sort(float arr[], int n) {
    if (n <= 1) return;

    float min = arr[0], max = arr[0];
    for (int i = 1; i < n; i++) {
        if (arr[i] < min) min = arr[i];
        if (arr[i] > max) max = arr[i];
    }
    if (max == min) return;

    int m = n * 0.43; // number of classes
    int L[m];
    for (int i = 0; i < m; i++) L[i] = 0;

    // Classification count
    for (int i = 0; i < n; i++) {
        int k = (int)((m - 1) * (arr[i] - min) / (max - min));
        L[k]++;
    }

    // Prefix sums
    for (int i = 1; i < m; i++) L[i] += L[i - 1];

    // Flash phase
    int move = 0, j = 0, k = m - 1;
    while (move < n - 1) {
        while (j >= L[k]) k = (int)((m - 1) * (arr[j] - min) / (max - min));
        float flash = arr[j];
        while (j != L[k]) {
            k = (int)((m - 1) * (flash - min) / (max - min));
            float hold = arr[--L[k]];
            arr[L[k]] = flash;
            flash = hold;
            move++;
        }
        j++;
    }

    // Insertion sort finish
    insertion_sort(arr, 0, n - 1);
}

int main(void) {
    float arr[] = {9,3,1,7,4,6,2,8,5};
    int n = sizeof(arr) / sizeof(arr[0]);
    flash_sort(arr, n);
    for (int i = 0; i < n; i++) printf("%.0f ", arr[i]);
    printf("\n");
}
```

#### Python

```python
def insertion_sort(arr, start, end):
    for i in range(start + 1, end + 1):
        key = arr[i]
        j = i - 1
        while j >= start and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key

def flash_sort(arr):
    n = len(arr)
    if n <= 1:
        return arr
    min_val, max_val = min(arr), max(arr)
    if max_val == min_val:
        return arr
    m = int(0.43 * n)
    L = [0] * m

    for x in arr:
        k = int((m - 1) * (x - min_val) / (max_val - min_val))
        L[k] += 1

    for i in range(1, m):
        L[i] += L[i - 1]

    move, j, k = 0, 0, m - 1
    while move < n - 1:
        while j >= L[k]:
            k = int((m - 1) * (arr[j] - min_val) / (max_val - min_val))
        flash = arr[j]
        while j != L[k]:
            k = int((m - 1) * (flash - min_val) / (max_val - min_val))
            L[k] -= 1
            arr[L[k]], flash = flash, arr[L[k]]
            move += 1
        j += 1
    insertion_sort(arr, 0, n - 1)
    return arr

arr = [9,3,1,7,4,6,2,8,5]
print(flash_sort(arr))
```

### Why It Matters

- Extremely fast on uniformly distributed data
- In-place (O(1) extra space)
- Practical for large arrays
- Combines distribution sorting and insertion finishing

### A Gentle Proof (Why It Works)

The classification step maps each element to its expected position region, dramatically reducing disorder.
Since each class has a small local range, insertion sort completes quickly.

Expected complexity (uniform input):

$$
T(n) = O(n) \text{ (classification) } + O(n) \text{ (final pass) } = O(n)
$$

Worst-case (skewed distribution): O(n²)

| Phase               | Work | Complexity      |
| ------------------- | ---- | --------------- |
| Classification      | O(n) | Compute classes |
| Flash rearrangement | O(n) | In-place moves  |
| Final sort          | O(n) | Local sorting   |

### Try It Yourself

1. Sort `[9,3,1,7,4,6,2,8,5]` step by step.
2. Change number of classes (0.3n, 0.5n).
3. Visualize class mapping for each element.
4. Count moves in flash phase.
5. Compare with Quick Sort timing on 10⁵ elements.
6. Test uniform vs skewed data.
7. Implement with different finishing sort.
8. Track cycles formed during flash phase.
9. Observe stability (it's not stable).
10. Benchmark against Merge Sort.

### Test Cases

| Input               | Output              | Notes          |
| ------------------- | ------------------- | -------------- |
| [9,3,1,7,4,6,2,8,5] | [1,2,3,4,5,6,7,8,9] | Example        |
| [5,5,5,5]           | [5,5,5,5]           | Equal elements |
| [1,2,3,4,5]         | [1,2,3,4,5]         | Already sorted |
| [9,8,7,6,5]         | [5,6,7,8,9]         | Reverse        |

### Complexity

| Aspect         | Value           |
| -------------- | --------------- |
| Time (Best)    | O(n)            |
| Time (Average) | O(n)            |
| Time (Worst)   | O(n²)           |
| Space          | O(1)            |
| Stable         | No              |
| Adaptive       | Yes (partially) |

Flash Sort is the lightning strike of sorting, it flashes elements to their expected zones in linear time, then finishes with a quick polish. When your data is uniform, it's one of the fastest practical sorts around.

### 128 Postman Sort

Postman Sort is a stable, multi-key sorting algorithm that works by sorting keys digit by digit or field by field, starting from the least significant field (like LSD Radix Sort) or most significant field (like MSD Radix Sort), depending on the application. It's often used for compound keys (e.g. postal addresses, dates, strings of fields), hence the name "Postman," as it sorts data the way a postman organizes mail: by street, then house, then apartment.

### What Problem Are We Solving?

When sorting complex records by multiple attributes, such as:

- Sorting people by `(city, street, house_number)`
- Sorting files by `(year, month, day)`
- Sorting products by `(category, brand, price)`

We need to sort hierarchically, that's what Postman Sort excels at. It's a stable, field-wise sorting approach, built upon Counting or Bucket Sort for each field.

#### Example

Sort tuples `(City, Street)` by City then Street:

```
$$("Paris", "B"), ("London", "C"), ("Paris", "A"), ("London", "A")]
```

| Step | Sort By      | Result                                                      |
| ---- | ------------ | ----------------------------------------------------------- |
| 1    | Street (LSD) | [("London","A"),("Paris","A"),("London","C"),("Paris","B")] |
| 2    | City (MSD)   | [("London","A"),("London","C"),("Paris","A"),("Paris","B")] |

Stable sorting ensures inner order is preserved from previous pass.

### How Does It Work (Plain Language)?

It's like organizing mail:

1. Sort by the smallest unit (house number)
2. Then sort by street
3. Then by city

Each pass refines the previous ordering.
If sorting from least to most significant, use LSD order (like Radix).
If sorting from most to least, use MSD order (like bucket recursion).

#### Step-by-Step Process

| Step | Description                                         |
| ---- | --------------------------------------------------- |
| 1    | Identify key fields and their order of significance |
| 2    | Choose stable sorting method (e.g., Counting Sort)  |
| 3    | Sort by least significant key first                 |
| 4    | Repeat for each key moving to most significant      |
| 5    | Final order respects all key hierarchies            |

### Tiny Code (Easy Versions)

#### Python (LSD Approach)

```python
def postman_sort(records, key_funcs):
    # key_funcs = list of functions to extract each field
    for key_func in reversed(key_funcs):
        records.sort(key=key_func)
    return records

# Example: sort by (city, street)
records = [("Paris", "B"), ("London", "C"), ("Paris", "A"), ("London", "A")]
sorted_records = postman_sort(records, [lambda x: x[0], lambda x: x[1]])
print(sorted_records)
```

Output:

```
$$('London', 'A'), ('London', 'C'), ('Paris', 'A'), ('Paris', 'B')]
```

#### C (Numeric Fields)

```c
#include <stdio.h>
#include <stdlib.h>

typedef struct {
    int city;
    int street;
} Record;

int cmp_street(const void *a, const void *b) {
    Record *ra = (Record*)a, *rb = (Record*)b;
    return ra->street - rb->street;
}

int cmp_city(const void *a, const void *b) {
    Record *ra = (Record*)a, *rb = (Record*)b;
    return ra->city - rb->city;
}

void postman_sort(Record arr[], int n) {
    // Sort by street first (LSD)
    qsort(arr, n, sizeof(Record), cmp_street);
    // Then by city (MSD)
    qsort(arr, n, sizeof(Record), cmp_city);
}

int main(void) {
    Record arr[] = {{2,3}, {1,2}, {2,1}, {1,1}};
    int n = sizeof(arr)/sizeof(arr[0]);
    postman_sort(arr, n);
    for (int i = 0; i < n; i++)
        printf("(%d,%d) ", arr[i].city, arr[i].street);
    printf("\n");
}
```

### Why It Matters

- Multi-key sorting (lexicographic order)
- Stable, preserves order across passes
- Versatile, works on compound keys, strings, records
- Foundation for:

  * Radix Sort
  * Database ORDER BY multi-column
  * Lexicographic ranking

### A Gentle Proof (Why It Works)

Each stable pass ensures that prior ordering (from less significant fields) remains intact.

If we denote each field as $f_i$, sorted stably in order $f_k \to f_1$:

$$
T(n) = \sum_{i=1}^k T_i(n)
$$

If each $T_i(n)$ is $O(n)$, the total is $O(kn)$.

Lexicographic order emerges naturally:
$$
(a_1, a_2, \ldots, a_k) < (b_1, b_2, \ldots, b_k)
\iff a_i = b_i \text{ for } i < j, \text{ and } a_j < b_j
$$

| Phase    | Operation                    | Stable | Complexity |
| -------- | ---------------------------- | ------ | ---------- |
| LSD Sort | Start from least significant | Yes    | $O(nk)$    |
| MSD Sort | Start from most significant  | Yes    | $O(nk)$    |


### Try It Yourself

1. Sort `[("Paris","B"),("London","C"),("Paris","A"),("London","A")]`
2. Add a 3rd field (zip code), sort by zip → street → city
3. Implement with `lambda` keys in Python
4. Replace stable sort with Counting Sort per field
5. Test with `[(2021,12,25),(2020,1,1),(2021,1,1)]`
6. Compare LSD vs MSD ordering
7. Sort strings by character groups (first char, second char, etc.)
8. Visualize passes and intermediate results
9. Test stability with repeated keys
10. Apply to sorting student records by `(grade, class, id)`

### Test Cases

| Input                                                       | Output                                                      | Notes         |
| ----------------------------------------------------------- | ----------------------------------------------------------- | ------------- |
| [("Paris","B"),("London","C"),("Paris","A"),("London","A")] | [("London","A"),("London","C"),("Paris","A"),("Paris","B")] | Lexicographic |
| [(2021,12,25),(2020,1,1),(2021,1,1)]                        | [(2020,1,1),(2021,1,1),(2021,12,25)]                        | Date order    |
| [(1,2,3),(1,1,3),(1,1,2)]                                   | [(1,1,2),(1,1,3),(1,2,3)]                                   | Multi-field   |

### Complexity

| Aspect   | Value       |
| -------- | ----------- |
| Time     | O(k × n)    |
| Space    | O(n)        |
| Stable   | Yes         |
| Adaptive | No          |
| Keys     | Multi-field |

Postman Sort delivers order like clockwork, field by field, pass by pass, ensuring every layer of your data finds its proper place, from apartment number to zip code.

### 129 Address Calculation Sort

Address Calculation Sort (sometimes called Hash Sort or Scatter Sort) is a distribution-based sorting method that uses a hash-like function (called an *address function*) to compute the final position of each element directly. Instead of comparing pairs, it computes where each element should go, much like direct addressing in hash tables.

### What Problem Are We Solving?

Comparison sorts need O(n log n) time.
If we know the range and distribution of input values, we can instead compute where each element belongs, placing it directly.

Address Calculation Sort bridges the gap between sorting and hashing:

- It computes a mapping from value to position.
- It places each element directly or into small groups.
- It works best when data distribution is known and uniform.

Applications:

- Dense numeric datasets
- Ranked records
- Pre-bucketed ranges

#### Example

Sort `[3, 1, 4, 0, 2]` with range `[0..4]`.

| Step | Element | Address Function $f(x) = x$ | Placement  |
| ---- | ------- | ----------------------------- | ---------- |
| 1    | 3       | f(3) = 3                      | position 3 |
| 2    | 1       | f(1) = 1                      | position 1 |
| 3    | 4       | f(4) = 4                      | position 4 |
| 4    | 0       | f(0) = 0                      | position 0 |
| 5    | 2       | f(2) = 2                      | position 2 |

Result: `[0, 1, 2, 3, 4]`

If multiple elements share the same address, they're stored in a small linked list or bucket, then sorted locally.

### How Does It Work (Plain Language)?

Imagine having labeled mailboxes for every possible key —
each element knows exactly which box it belongs in.
You just drop each letter into its slot, then read the boxes in order.

Unlike Counting Sort, which counts occurrences,
Address Calculation Sort assigns positions, it can even be in-place if collisions are handled carefully.

#### Step-by-Step Process

| Step | Description                                                   |
| ---- | ------------------------------------------------------------- |
| 1    | Define address function $f(x)$ mapping each element to an index |
| 2    | Initialize empty output array or buckets                      |
| 3    | For each element $a_i$: compute index = $f(a_i)$              |
| 4    | Place element at index (handle collisions if needed)          |
| 5    | Collect or flatten buckets into sorted order                  |


### Tiny Code (Easy Versions)

#### C (Simple In-Range Example)

```c
#include <stdio.h>
#include <stdlib.h>

void address_calculation_sort(int arr[], int n, int min, int max) {
    int range = max - min + 1;
    int *output = calloc(range, sizeof(int));
    int *filled = calloc(range, sizeof(int));

    for (int i = 0; i < n; i++) {
        int idx = arr[i] - min;
        output[idx] = arr[i];
        filled[idx] = 1;
    }

    int k = 0;
    for (int i = 0; i < range; i++) {
        if (filled[i]) arr[k++] = output[i];
    }

    free(output);
    free(filled);
}

int main(void) {
    int arr[] = {3, 1, 4, 0, 2};
    int n = sizeof(arr)/sizeof(arr[0]);
    address_calculation_sort(arr, n, 0, 4);
    for (int i = 0; i < n; i++) printf("%d ", arr[i]);
    printf("\n");
}
```

#### Python

```python
def address_calculation_sort(arr, f=None):
    if not arr:
        return arr
    if f is None:
        f = lambda x: x  # identity mapping

    min_val, max_val = min(arr), max(arr)
    size = max_val - min_val + 1
    slots = [[] for _ in range(size)]

    for x in arr:
        idx = f(x) - min_val
        slots[idx].append(x)

    # Flatten buckets
    result = []
    for bucket in slots:
        result.extend(sorted(bucket))  # local sort if needed
    return result

arr = [3, 1, 4, 0, 2]
print(address_calculation_sort(arr))
```

Output:

```
$$0, 1, 2, 3, 4]
```

### Why It Matters

- Direct computation of sorted positions
- Linear time for predictable distributions
- Combines hashing and sorting
- Forms basis of bucket, radix, and flash sort

It's a great way to see how functions can replace comparisons.

### A Gentle Proof (Why It Works)

If $f(x)$ maps each key uniquely to its sorted index,
the result is already sorted by construction.

Even with collisions, sorting small local buckets is fast.

Total time:
$$
T(n) = O(n + k)
$$
where $k$ = number of buckets = range size.

| Phase             | Work     | Complexity         |
| ----------------- | -------- | ------------------ |
| Compute addresses | O(n)     | One per element    |
| Bucket insertions | O(n)     | Constant time each |
| Local sort        | O(k)     | Small groups       |
| Flatten           | O(n + k) | Read back          |

### Try It Yourself

1. Sort `[3, 1, 4, 0, 2]` with `f(x)=x`.
2. Change mapping to `f(x)=2*x`, note gaps.
3. Add duplicates `[1,1,2,2,3]`, use buckets.
4. Try floats with `f(x)=int(x*10)` for `[0.1,0.2,0.3]`.
5. Sort strings by ASCII sum: `f(s)=sum(map(ord,s))`.
6. Compare with Counting Sort (no explicit storage).
7. Handle negative numbers by offsetting min.
8. Visualize mapping table.
9. Test range gaps (e.g., `[10, 20, 30]`).
10. Experiment with custom hash functions.

### Test Cases

| Input         | Output        | Notes           |
| ------------- | ------------- | --------------- |
| [3,1,4,0,2]   | [0,1,2,3,4]   | Perfect mapping |
| [10,20,30]    | [10,20,30]    | Sparse mapping  |
| [1,1,2,2]     | [1,1,2,2]     | Duplicates      |
| [0.1,0.3,0.2] | [0.1,0.2,0.3] | Float mapping   |

### Complexity

| Aspect          | Value              |
| --------------- | ------------------ |
| Time            | O(n + k)           |
| Space           | O(n + k)           |
| Stable          | Yes (with buckets) |
| Adaptive        | No                 |
| Range-sensitive | Yes                |

Address Calculation Sort turns sorting into placement, each value finds its home by formula, not fight. When the range is clear and collisions are rare, it's lightning-fast and elegantly simple.

### 130 Spread Sort

Spread Sort is a hybrid distribution sort that blends ideas from radix sort, bucket sort, and comparison sorting. It distributes elements into buckets using their most significant bits (MSB) or value ranges, then recursively sorts buckets (like radix/MSD) or switches to a comparison sort (like Quick Sort) when buckets are small.

It's cache-friendly, adaptive, and often faster than Quick Sort on uniformly distributed data. In fact, it's used in some high-performance libraries such as Boost C++ Spreadsort.

### What Problem Are We Solving?

Traditional comparison sorts (like Quick Sort) have $O(n \log n)$ complexity, while pure radix-based sorts can be inefficient on small or skewed datasets. Spread Sort solves this by adapting dynamically:

- Distribute like radix sort when data is wide and random
- Compare like Quick Sort when data is clustered or buckets are small

It "spreads" data across buckets, then sorts each bucket intelligently.

Perfect for:

- Integers, floats, and strings
- Large datasets
- Wide value ranges

#### Example

Sort `[43, 12, 89, 27, 55, 31, 70]`

| Step | Action                                      | Result                                   |
| ---- | ------------------------------------------- | ---------------------------------------- |
| 1    | Find min = 12, max = 89                     | range = 77                               |
| 2    | Choose bucket count (e.g. 4)                | bucket size ≈ 19                         |
| 3    | Distribute by bucket index = (val - min)/19 | Buckets: [12], [27,31], [43,55], [70,89] |
| 4    | Recursively sort each bucket                | [12], [27,31], [43,55], [70,89]          |
| 5    | Merge buckets                               | [12,27,31,43,55,70,89]                   |

### How Does It Work (Plain Language)?

Imagine sorting mail by first letter (distribution), then alphabetizing each pile (comparison).
If a pile is still big, spread it again by the next letter.
If a pile is small, just sort it directly.

Spread Sort uses:

- Distribution when the data range is wide
- Comparison sort when buckets are narrow or small

This flexibility gives it strong real-world performance.

#### Step-by-Step Process

| Step | Description                                     |
| ---- | ----------------------------------------------- |
| 1    | Find min and max values                         |
| 2    | Compute spread = max - min                      |
| 3    | Choose bucket count (based on n or spread)      |
| 4    | Distribute elements into buckets by value range |
| 5    | Recursively apply spread sort to large buckets  |
| 6    | Apply comparison sort to small buckets          |
| 7    | Concatenate results                             |

### Tiny Code (Easy Versions)

#### C (Integer Example)

```c
#include <stdio.h>
#include <stdlib.h>

int compare(const void *a, const void *b) {
    return (*(int*)a - *(int*)b);
}

void spread_sort(int arr[], int n) {
    if (n <= 16) { // threshold for small buckets
        qsort(arr, n, sizeof(int), compare);
        return;
    }

    int min = arr[0], max = arr[0];
    for (int i = 1; i < n; i++) {
        if (arr[i] < min) min = arr[i];
        if (arr[i] > max) max = arr[i];
    }

    int bucket_count = n / 4 + 1;
    int range = max - min + 1;
    int bucket_size = range / bucket_count + 1;

    int *counts = calloc(bucket_count, sizeof(int));
    int buckets = malloc(bucket_count * sizeof(int*));
    for (int i = 0; i < bucket_count; i++)
        buckets[i] = malloc(n * sizeof(int));

    // Distribution
    for (int i = 0; i < n; i++) {
        int idx = (arr[i] - min) / bucket_size;
        buckets[idx][counts[idx]++] = arr[i];
    }

    // Recursive sort
    int k = 0;
    for (int i = 0; i < bucket_count; i++) {
        if (counts[i] > 0) {
            spread_sort(buckets[i], counts[i]);
            for (int j = 0; j < counts[i]; j++)
                arr[k++] = buckets[i][j];
        }
        free(buckets[i]);
    }

    free(buckets);
    free(counts);
}

int main(void) {
    int arr[] = {43, 12, 89, 27, 55, 31, 70};
    int n = sizeof(arr)/sizeof(arr[0]);
    spread_sort(arr, n);
    for (int i = 0; i < n; i++) printf("%d ", arr[i]);
    printf("\n");
}
```

#### Python

```python
def spread_sort(arr, threshold=16):
    if len(arr) <= threshold:
        return sorted(arr)

    min_val, max_val = min(arr), max(arr)
    if min_val == max_val:
        return arr[:]

    n = len(arr)
    bucket_count = n // 4 + 1
    spread = max_val - min_val + 1
    bucket_size = spread // bucket_count + 1

    buckets = [[] for _ in range(bucket_count)]
    for x in arr:
        idx = (x - min_val) // bucket_size
        buckets[idx].append(x)

    result = []
    for b in buckets:
        if len(b) > threshold:
            result.extend(spread_sort(b, threshold))
        else:
            result.extend(sorted(b))
    return result

arr = [43, 12, 89, 27, 55, 31, 70]
print(spread_sort(arr))
```

### Why It Matters

- Linear-time on uniform data
- Adaptive to distribution and size
- Combines bucket and comparison power
- Great for large numeric datasets

Used in Boost C++, it's a real-world performant hybrid.

### A Gentle Proof (Why It Works)

Spread Sort's cost depends on:

- Distribution pass: $O(n)$  
- Recursion depth: small (logarithmic for uniform data)  
- Local sorts: small and fast (often $O(1)$ or $O(n \log n)$ on small buckets)

For $n$ elements and $b$ buckets:

$$
T(n) = O(n + \sum_{i=1}^{b} T_i)
$$
If average bucket size is constant or small:
$$
T(n) \approx O(n)
$$

| Phase        | Work                           | Complexity            |
| ------------- | ------------------------------ | --------------------- |
| Distribution  | $O(n)$                         | One pass              |
| Local Sorts   | $O(n \log m)$                  | $m =$ average bucket size |
| Total         | $O(n)$ average, $O(n \log n)$ worst |                     |


### Try It Yourself

1. Sort `[43,12,89,27,55,31,70]`.
2. Try `[5,4,3,2,1]` (non-uniform).
3. Adjust bucket threshold.
4. Visualize recursive buckets.
5. Mix large and small values (e.g., `[1, 10, 1000, 2, 20, 2000]`).
6. Compare runtime with Quick Sort.
7. Implement float version.
8. Measure distribution imbalance.
9. Tune bucket size formula.
10. Sort strings by ord(char).

### Test Cases

| Input                  | Output                 | Notes      |
| ---------------------- | ---------------------- | ---------- |
| [43,12,89,27,55,31,70] | [12,27,31,43,55,70,89] | Uniform    |
| [5,4,3,2,1]            | [1,2,3,4,5]            | Reverse    |
| [100,10,1,1000]        | [1,10,100,1000]        | Wide range |
| [5,5,5]                | [5,5,5]                | Duplicates |

### Complexity

| Aspect         | Value                         |
| -------------- | ----------------------------- |
| Time (Best)    | O(n)                          |
| Time (Average) | O(n)                          |
| Time (Worst)   | O(n log n)                    |
| Space          | O(n)                          |
| Stable         | Yes (if local sort is stable) |
| Adaptive       | Yes                           |

Spread Sort spreads elements like seeds, each finds fertile ground in its range, then blossoms into order through local sorting. It's the smart hybrid that brings the best of radix and comparison worlds together.

## Section 14. Hybrid sorts

### 131 IntroSort

IntroSort (short for *Introspective Sort*) is a hybrid sorting algorithm that combines the best features of Quick Sort, Heap Sort, and Insertion Sort. It begins with Quick Sort for speed, but if recursion goes too deep (indicating unbalanced partitions), it switches to Heap Sort to guarantee worst-case $O(n \log n)$ performance. For small partitions, it often uses Insertion Sort for efficiency.

It was introduced by David Musser (1997) and is the default sorting algorithm in C++ STL (`std::sort`), fast, adaptive, and safe.

### What Problem Are We Solving?

Pure Quick Sort is fast on average but can degrade to $O(n^2)$ in the worst case (for example, sorted input with bad pivots).  
Heap Sort guarantees $O(n \log n)$ but has worse constants.

IntroSort combines them, using Quick Sort *until danger*, then switching to Heap Sort for safety.

Perfect for:

- General-purpose sorting (numeric, string, object)
- Performance-critical libraries
- Mixed data with unknown distribution

#### Example

Sort `[9, 3, 1, 7, 5, 4, 6, 2, 8]`

| Step | Action                                      | Result            |
| ---- | ------------------------------------------- | ----------------- |
| 1    | Start Quick Sort (depth = 0)                | pivot = 5         |
| 2    | Partition → `[3,1,2,4] [5] [9,7,6,8]`       | depth = 1         |
| 3    | Recurse left + right                        |                   |
| 4    | If depth > 2 × log₂(n), switch to Heap Sort | prevents (O(n^2)) |
| 5    | Use Insertion Sort on small segments        |                   |

Final sorted array: `[1,2,3,4,5,6,7,8,9]`

### How Does It Work (Plain Language)?

It's like a careful driver:

- Start fast on the highway (Quick Sort)
- If road gets tricky (too many turns = recursion depth), switch to 4-wheel drive (Heap Sort)
- For small parking spots (tiny arrays), use a nimble bike (Insertion Sort)

IntroSort keeps average speed high but avoids worst-case crashes.

#### Step-by-Step Process

| Step | Description                                                  |
| ---- | ------------------------------------------------------------ |
| 1    | Start Quick Sort recursively                                 |
| 2    | Track recursion depth                                        |
| 3    | If depth exceeds threshold (2 × log₂ n), switch to Heap Sort |
| 4    | Use Insertion Sort for small subarrays (size < threshold)    |
| 5    | Combine results for final sorted array                       |

### Tiny Code (Easy Versions)

#### C (Simplified Implementation)

```c
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define THRESHOLD 16

void insertion_sort(int arr[], int n) {
    for (int i = 1; i < n; i++) {
        int key = arr[i], j = i - 1;
        while (j >= 0 && arr[j] > key) arr[j + 1] = arr[j--];
        arr[j + 1] = key;
    }
}

void heapify(int arr[], int n, int i) {
    int largest = i, l = 2*i + 1, r = 2*i + 2;
    if (l < n && arr[l] > arr[largest]) largest = l;
    if (r < n && arr[r] > arr[largest]) largest = r;
    if (largest != i) {
        int tmp = arr[i]; arr[i] = arr[largest]; arr[largest] = tmp;
        heapify(arr, n, largest);
    }
}

void heap_sort(int arr[], int n) {
    for (int i = n/2 - 1; i >= 0; i--) heapify(arr, n, i);
    for (int i = n - 1; i >= 0; i--) {
        int tmp = arr[0]; arr[0] = arr[i]; arr[i] = tmp;
        heapify(arr, i, 0);
    }
}

int partition(int arr[], int low, int high) {
    int pivot = arr[high], i = low - 1;
    for (int j = low; j < high; j++) {
        if (arr[j] <= pivot) {
            i++;
            int tmp = arr[i]; arr[i] = arr[j]; arr[j] = tmp;
        }
    }
    int tmp = arr[i+1]; arr[i+1] = arr[high]; arr[high] = tmp;
    return i + 1;
}

void introsort_rec(int arr[], int low, int high, int depth_limit) {
    int n = high - low + 1;
    if (n <= THRESHOLD) {
        insertion_sort(arr + low, n);
        return;
    }
    if (depth_limit == 0) {
        heap_sort(arr + low, n);
        return;
    }
    int p = partition(arr, low, high);
    introsort_rec(arr, low, p - 1, depth_limit - 1);
    introsort_rec(arr, p + 1, high, depth_limit - 1);
}

void introsort(int arr[], int n) {
    int depth_limit = 2 * log(n);
    introsort_rec(arr, 0, n - 1, depth_limit);
}

int main(void) {
    int arr[] = {9,3,1,7,5,4,6,2,8};
    int n = sizeof(arr)/sizeof(arr[0]);
    introsort(arr, n);
    for (int i = 0; i < n; i++) printf("%d ", arr[i]);
    printf("\n");
}
```

#### Python (Conceptual Demo)

```python
import math

def insertion_sort(a):
    for i in range(1, len(a)):
        key = a[i]
        j = i - 1
        while j >= 0 and a[j] > key:
            a[j + 1] = a[j]
            j -= 1
        a[j + 1] = key

def heapify(a, n, i):
    largest = i
    l, r = 2*i+1, 2*i+2
    if l < n and a[l] > a[largest]:
        largest = l
    if r < n and a[r] > a[largest]:
        largest = r
    if largest != i:
        a[i], a[largest] = a[largest], a[i]
        heapify(a, n, largest)

def heap_sort(a):
    n = len(a)
    for i in range(n//2-1, -1, -1):
        heapify(a, n, i)
    for i in range(n-1, 0, -1):
        a[0], a[i] = a[i], a[0]
        heapify(a, i, 0)

def introsort(a, depth_limit=None):
    n = len(a)
    if n <= 16:
        insertion_sort(a)
        return a
    if depth_limit is None:
        depth_limit = 2 * int(math.log2(n))
    if depth_limit == 0:
        heap_sort(a)
        return a

    pivot = a[-1]
    left = [x for x in a[:-1] if x <= pivot]
    right = [x for x in a[:-1] if x > pivot]
    introsort(left, depth_limit - 1)
    introsort(right, depth_limit - 1)
    a[:] = left + [pivot] + right
    return a

arr = [9,3,1,7,5,4,6,2,8]
print(introsort(arr))
```

### Why It Matters

- Default in C++ STL, fast and reliable
- Guaranteed worst-case $O(n \log n)$
- Optimized for cache and small data
- Adaptive, uses best method for current scenario

### A Gentle Proof (Why It Works)

Quick Sort dominates until recursion depth = $2 \log_2 n$.
At that point, worst-case risk appears → switch to Heap Sort (safe fallback).
Small subarrays are handled by Insertion Sort for low overhead.

So overall:
$$
T(n) = O(n \log n)
$$
Always bounded by Heap Sort's worst case, but often near Quick Sort's best.

| Phase          | Method         | Complexity              |
| -------------- | -------------- | ----------------------- |
| Partitioning   | Quick Sort     | O(n log n)              |
| Deep recursion | Heap Sort      | O(n log n)              |
| Small arrays   | Insertion Sort | O(n²) local, negligible |

### Try It Yourself

1. Sort `[9,3,1,7,5,4,6,2,8]`.
2. Track recursion depth, switch to Heap Sort when $> 2 \log_2 n$.
3. Replace threshold = 16 with 8, 32, measure effect.
4. Test with already sorted array, confirm Heap fallback.
5. Compare timing with Quick Sort and Heap Sort.
6. Print method used at each stage.
7. Test large array (10⁵ elements).
8. Verify worst-case safety on sorted input.
9. Try string sorting with custom comparator.
10. Implement generic version using templates or lambdas.

### Test Cases

| Input               | Output              | Notes                             |
| ------------------- | ------------------- | --------------------------------- |
| [9,3,1,7,5,4,6,2,8] | [1,2,3,4,5,6,7,8,9] | Balanced partitions               |
| [1,2,3,4,5]         | [1,2,3,4,5]         | Sorted input (Heap Sort fallback) |
| [5,5,5,5]           | [5,5,5,5]           | Equal elements                    |
| [9,8,7,6,5,4,3,2,1] | [1,2,3,4,5,6,7,8,9] | Worst-case Quick Sort avoided     |

### Complexity

| Aspect         | Value      |
| -------------- | ---------- |
| Time (Best)    | O(n log n) |
| Time (Average) | O(n log n) |
| Time (Worst)   | O(n log n) |
| Space          | O(log n)   |
| Stable         | No         |
| Adaptive       | Yes        |

IntroSort is the strategist's algorithm, it starts bold (Quick Sort), defends wisely (Heap Sort), and finishes gracefully (Insertion Sort). A perfect balance of speed, safety, and adaptability.

### 132 TimSort

TimSort is a hybrid sorting algorithm combining Merge Sort and Insertion Sort, designed for real-world data that often contains partially ordered runs. It was invented by Tim Peters in 2002 and is the default sorting algorithm in Python (`sorted()`, `.sort()`) and Java (`Arrays.sort()` for objects).

TimSort's superpower is that it detects natural runs in data, sorts them with Insertion Sort if small, and merges them smartly using a stack-based strategy to ensure efficiency.

### What Problem Are We Solving?

In practice, many datasets aren't random, they already contain sorted segments (like logs, names, timestamps).
TimSort exploits this by:

- Detecting ascending/descending runs
- Sorting small runs via Insertion Sort
- Merging runs using adaptive Merge Sort

This yields O(n) performance on already-sorted or nearly-sorted data, far better than standard $O(n \log n)$ sorts in such cases.

Perfect for:

- Partially sorted lists
- Real-world data (time series, strings, logs)
- Stable sorting (preserve order of equals)

#### Example

Sort `[5, 6, 7, 1, 2, 3, 8, 9]`

| Step | Action                    | Result                  |
| ---- | ------------------------- | ----------------------- |
| 1    | Detect runs               | [5,6,7], [1,2,3], [8,9] |
| 2    | Sort small runs if needed | [5,6,7], [1,2,3], [8,9] |
| 3    | Merge runs pairwise       | [1,2,3,5,6,7,8,9]       |

TimSort leverages order already present, fewer merges, faster finish.

### How Does It Work (Plain Language)?

Think of TimSort as a smart librarian:

- Sees which shelves (runs) are already sorted
- Tidies up small messy shelves (Insertion Sort)
- Merges shelves together efficiently (Merge Sort)
- Uses stack rules to decide merge timing for balance

It's adaptive, stable, and real-world optimized.

#### Step-by-Step Process

| Step | Description                                           |
| ---- | ----------------------------------------------------- |
| 1    | Scan the array to find runs (ascending or descending) |
| 2    | Reverse descending runs                               |
| 3    | If run length < `minrun`, extend using Insertion Sort |
| 4    | Push runs onto stack                                  |
| 5    | Merge runs when stack size conditions are violated    |
| 6    | Continue until one run remains (fully sorted)         |

### Tiny Code (Easy Versions)

#### Python (Simplified Simulation)

```python
def insertion_sort(arr, left, right):
    for i in range(left + 1, right + 1):
        key = arr[i]
        j = i - 1
        while j >= left and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key

def merge(arr, left, mid, right):
    left_part = arr[left:mid+1]
    right_part = arr[mid+1:right+1]
    i = j = 0
    k = left
    while i < len(left_part) and j < len(right_part):
        if left_part[i] <= right_part[j]:
            arr[k] = left_part[i]
            i += 1
        else:
            arr[k] = right_part[j]
            j += 1
        k += 1
    while i < len(left_part):
        arr[k] = left_part[i]; i += 1; k += 1
    while j < len(right_part):
        arr[k] = right_part[j]; j += 1; k += 1

def timsort(arr):
    n = len(arr)
    minrun = 32

    # Sort small runs using insertion sort
    for start in range(0, n, minrun):
        end = min(start + minrun - 1, n - 1)
        insertion_sort(arr, start, end)

    size = minrun
    while size < n:
        for left in range(0, n, 2 * size):
            mid = min(n - 1, left + size - 1)
            right = min(n - 1, left + 2 * size - 1)
            if mid < right:
                merge(arr, left, mid, right)
        size *= 2

arr = [5,6,7,1,2,3,8,9]
timsort(arr)
print(arr)
```

Output:

```
$$1, 2, 3, 5, 6, 7, 8, 9]
```

#### C (Conceptual Version)

```c
#include <stdio.h>
#include <stdlib.h>

#define MINRUN 32

void insertion_sort(int arr[], int left, int right) {
    for (int i = left + 1; i <= right; i++) {
        int key = arr[i], j = i - 1;
        while (j >= left && arr[j] > key) {
            arr[j + 1] = arr[j];
            j--;
        }
        arr[j + 1] = key;
    }
}

void merge(int arr[], int l, int m, int r) {
    int n1 = m - l + 1, n2 = r - m;
    int *L = malloc(n1 * sizeof(int)), *R = malloc(n2 * sizeof(int));
    for (int i = 0; i < n1; i++) L[i] = arr[l + i];
    for (int j = 0; j < n2; j++) R[j] = arr[m + 1 + j];
    int i = 0, j = 0, k = l;
    while (i < n1 && j < n2) arr[k++] = (L[i] <= R[j]) ? L[i++] : R[j++];
    while (i < n1) arr[k++] = L[i++];
    while (j < n2) arr[k++] = R[j++];
    free(L); free(R);
}

void timsort(int arr[], int n) {
    for (int i = 0; i < n; i += MINRUN) {
        int end = (i + MINRUN - 1 < n) ? (i + MINRUN - 1) : (n - 1);
        insertion_sort(arr, i, end);
    }
    for (int size = MINRUN; size < n; size *= 2) {
        for (int left = 0; left < n; left += 2 * size) {
            int mid = (left + size - 1 < n - 1) ? (left + size - 1) : (n - 1);
            int right = (left + 2 * size - 1 < n - 1) ? (left + 2 * size - 1) : (n - 1);
            if (mid < right) merge(arr, left, mid, right);
        }
    }
}

int main(void) {
    int arr[] = {5,6,7,1,2,3,8,9};
    int n = sizeof(arr)/sizeof(arr[0]);
    timsort(arr, n);
    for (int i = 0; i < n; i++) printf("%d ", arr[i]);
    printf("\n");
}
```

### Why It Matters

- Default sort in Python & Java
- Stable and adaptive
- O(n) on sorted data
- O(n log n) worst case
- Handles real-world inputs gracefully

It's the perfect sort when you don't know the data shape, it *adapts itself*.

### A Gentle Proof (Why It Works)

If data already contains sorted runs of average length $r$:

- Insertion Sort: $O(r^2)$ per run (tiny)  
- Merging $(n / r)$ runs: $O(n \log (n / r))$


Overall:
$$
T(n) = O(n + n \log (n/r))
$$
For $r \approx n$: $O(n)$  
For $r = 1$: $O(n \log n)$

| Phase         | Work              | Complexity      |
| ------------- | ----------------- | --------------- |
| Run Detection | $O(n)$            | One pass        |
| Local Sorting | $O(r^2)$ per run  | Tiny runs       |
| Merge Phase   | $O(n \log n)$     | Balanced merges |


### Try It Yourself

1. Sort `[5,6,7,1,2,3,8,9]` step by step.
2. Detect natural runs manually.
3. Reverse descending runs before merge.
4. Adjust `minrun = 16`, see difference.
5. Test `[1,2,3,4,5]`, should take O(n).
6. Add random noise to partially sorted list.
7. Track stack of runs, when to merge?
8. Compare performance with Merge Sort.
9. Visualize merge order tree.
10. Check stability with duplicate keys.

### Test Cases

| Input             | Output            | Notes          |
| ----------------- | ----------------- | -------------- |
| [5,6,7,1,2,3,8,9] | [1,2,3,5,6,7,8,9] | Mixed runs     |
| [1,2,3,4,5]       | [1,2,3,4,5]       | Already sorted |
| [9,8,7,6]         | [6,7,8,9]         | Reverse run    |
| [5,5,5,5]         | [5,5,5,5]         | Stability test |

### Complexity

| Aspect         | Value      |
| -------------- | ---------- |
| Time (Best)    | O(n)       |
| Time (Average) | O(n log n) |
| Time (Worst)   | O(n log n) |
| Space          | O(n)       |
| Stable         | Yes        |
| Adaptive       | Yes        |

TimSort is the real-world champion, it watches your data, adapts instantly, and sorts smarter, not harder. It's the kind of algorithm that doesn't just run fast, it *thinks* fast.

### 133 Dual-Pivot QuickSort

Dual-Pivot QuickSort is an enhanced variant of QuickSort that uses two pivots instead of one to partition the array into three regions:

- Elements less than pivot1,
- Elements between pivot1 and pivot2,
- Elements greater than pivot2.

This approach often reduces comparisons and improves cache efficiency. It was popularized by Vladimir Yaroslavskiy and became the default sorting algorithm in Java (from Java 7) for primitive types.

### What Problem Are We Solving?

Standard QuickSort splits the array into two parts using one pivot.
Dual-Pivot QuickSort splits into three, reducing recursion depth and overhead.

It's optimized for:

- Large arrays of primitives (integers, floats)
- Random and uniform data
- Modern CPUs with deep pipelines and caches

It offers better real-world performance than classic QuickSort, even if asymptotic complexity remains $O(n \log n)$.


#### Example

Sort `[9, 3, 1, 7, 5, 4, 6, 2, 8]`

Choose pivots ($p_1 = 3$, $p_2 = 7$):

| Region | Condition     | Elements      |
| ------- | ------------- | ------------- |
| Left    | $x < 3$       | [1, 2]        |
| Middle  | $3 \le x \le 7$ | [3, 4, 5, 6, 7] |
| Right   | $x > 7$       | [8, 9]        |

Recurse on each region.  
Final result: `[1, 2, 3, 4, 5, 6, 7, 8, 9]`.


### How Does It Work (Plain Language)?

Imagine sorting shoes by size with two markers:

- Small shelf for sizes less than 7
- Middle shelf for 7–9
- Big shelf for 10+

You walk through once, placing each shoe in the right group, then sort each shelf individually.

Dual-Pivot QuickSort does exactly that: partition into three zones in one pass, then recurse.

#### Step-by-Step Process

| Step | Description                                              |
| ---- | -------------------------------------------------------- |
| 1    | Choose two pivots ($p_1$, $p_2$), ensuring $p_1 < p_2$   |
| 2    | Partition array into 3 parts: `< p₁`, `p₁..p₂`, `> p₂`   |
| 3    | Recursively sort each part                               |
| 4    | Combine results in order                                 |

If $p_1 > p_2$, swap them first.


### Tiny Code (Easy Versions)

#### C

```c
#include <stdio.h>

void swap(int *a, int *b) {
    int tmp = *a; *a = *b; *b = tmp;
}

void dual_pivot_quicksort(int arr[], int low, int high) {
    if (low >= high) return;

    if (arr[low] > arr[high]) swap(&arr[low], &arr[high]);
    int p1 = arr[low], p2 = arr[high];

    int lt = low + 1, gt = high - 1, i = low + 1;

    while (i <= gt) {
        if (arr[i] < p1) {
            swap(&arr[i], &arr[lt]);
            lt++; i++;
        } else if (arr[i] > p2) {
            swap(&arr[i], &arr[gt]);
            gt--;
        } else {
            i++;
        }
    }
    lt--; gt++;
    swap(&arr[low], &arr[lt]);
    swap(&arr[high], &arr[gt]);

    dual_pivot_quicksort(arr, low, lt - 1);
    dual_pivot_quicksort(arr, lt + 1, gt - 1);
    dual_pivot_quicksort(arr, gt + 1, high);
}

int main(void) {
    int arr[] = {9,3,1,7,5,4,6,2,8};
    int n = sizeof(arr)/sizeof(arr[0]);
    dual_pivot_quicksort(arr, 0, n-1);
    for (int i = 0; i < n; i++) printf("%d ", arr[i]);
    printf("\n");
}
```

#### Python

```python
def dual_pivot_quicksort(arr):
    def sort(low, high):
        if low >= high:
            return
        if arr[low] > arr[high]:
            arr[low], arr[high] = arr[high], arr[low]
        p1, p2 = arr[low], arr[high]
        lt, gt, i = low + 1, high - 1, low + 1
        while i <= gt:
            if arr[i] < p1:
                arr[i], arr[lt] = arr[lt], arr[i]
                lt += 1; i += 1
            elif arr[i] > p2:
                arr[i], arr[gt] = arr[gt], arr[i]
                gt -= 1
            else:
                i += 1
        lt -= 1; gt += 1
        arr[low], arr[lt] = arr[lt], arr[low]
        arr[high], arr[gt] = arr[gt], arr[high]
        sort(low, lt - 1)
        sort(lt + 1, gt - 1)
        sort(gt + 1, high)
    sort(0, len(arr) - 1)
    return arr

arr = [9,3,1,7,5,4,6,2,8]
print(dual_pivot_quicksort(arr))
```

### Why It Matters

- Default in Java for primitives
- Fewer comparisons than single-pivot QuickSort
- Cache-friendly (less branching)
- Stable recursion depth with three partitions

### A Gentle Proof (Why It Works)

Each partitioning step processes all elements once, $O(n)$.
Recursion on three smaller subarrays yields total cost:

$$
T(n) = T(k_1) + T(k_2) + T(k_3) + O(n)
$$
On average, partitions are balanced → $T(n) = O(n \log n)$

| Phase        | Operation   | Complexity       |
| ------------- | ----------- | ---------------- |
| Partitioning  | $O(n)$      | One pass         |
| Recursion     | 3 subarrays | Balanced depth   |
| Total         | $O(n \log n)$ | Average / Worst |


### Try It Yourself

1. Sort `[9,3,1,7,5,4,6,2,8]` step by step.
2. Choose pivots manually: smallest and largest.
3. Trace index movements (lt, gt, i).
4. Compare with classic QuickSort partition count.
5. Use reversed array, observe stability.
6. Add duplicates `[5,5,5,5]`, see middle zone effect.
7. Measure comparisons vs single pivot.
8. Try large input (10⁶) and time it.
9. Visualize three partitions recursively.
10. Implement tail recursion optimization.

### Test Cases

| Input               | Output              | Notes          |
| ------------------- | ------------------- | -------------- |
| [9,3,1,7,5,4,6,2,8] | [1,2,3,4,5,6,7,8,9] | Example        |
| [1,2,3,4]           | [1,2,3,4]           | Already sorted |
| [9,8,7,6,5]         | [5,6,7,8,9]         | Reverse        |
| [5,5,5,5]           | [5,5,5,5]           | Duplicates     |

### Complexity

| Aspect         | Value      |
| -------------- | ---------- |
| Time (Best)    | O(n log n) |
| Time (Average) | O(n log n) |
| Time (Worst)   | O(n log n) |
| Space          | O(log n)   |
| Stable         | No         |
| Adaptive       | No         |

Dual-Pivot QuickSort slices data with two blades instead of one, making each cut smaller, shallower, and more balanced. A sharper, smarter evolution of a timeless classic.

### 134 SmoothSort

SmoothSort is an adaptive comparison-based sorting algorithm invented by Edsger Dijkstra. It's similar in spirit to Heap Sort, but smarter, it runs in O(n) time on already sorted data and O(n log n) in the worst case.

The key idea is to build a special heap structure (Leonardo heap) that adapts to existing order in the data. When the array is nearly sorted, it finishes quickly. When not, it gracefully falls back to heap-like performance.

### What Problem Are We Solving?

Heap Sort always works in $O(n \log n)$, even when the input is already sorted.
SmoothSort improves on this by being adaptive, the more ordered the input, the faster it gets.

It's ideal for:

- Nearly sorted arrays
- Situations requiring guaranteed upper bounds
- Memory-constrained environments (in-place sort)

#### Example

Sort `[1, 2, 4, 3, 5]`

| Step | Action                                  | Result      |
| ---- | --------------------------------------- | ----------- |
| 1    | Build initial heap (Leonardo structure) | [1,2,4,3,5] |
| 2    | Detect small disorder (4,3)             | Swap        |
| 3    | Restore heap property                   | [1,2,3,4,5] |
| 4    | Sorted early, no full rebuild needed   | Done        |

Result: finished early since only minor disorder existed.

### How Does It Work (Plain Language)?

Imagine a stack of heaps, each representing a Fibonacci-like sequence (Leonardo numbers).
You grow this structure as you read the array, maintaining order locally.
When disorder appears, you fix only where needed, not everywhere.

So SmoothSort is like a gentle gardener: it only trims where weeds grow, not the whole garden.

#### Step-by-Step Process

| Step | Description                                                    |
| ---- | -------------------------------------------------------------- |
| 1    | Represent the array as a series of Leonardo heaps              |
| 2    | Add elements one by one, updating the heap sequence            |
| 3    | Maintain heap property with minimal swaps                      |
| 4    | When finished, repeatedly extract the maximum (like Heap Sort) |
| 5    | During extraction, merge smaller heaps as needed               |

Leonardo numbers guide heap sizes:
( L(0)=1, L(1)=1, L(n)=L(n-1)+L(n-2)+1 )

### Tiny Code (Easy Versions)

#### Python (Simplified Adaptive Sort)

This is a simplified version to show adaptiveness, not a full Leonardo heap implementation.

```python
def smoothsort(arr):
    # Simplified: detect sorted runs, fix only where needed
    n = len(arr)
    for i in range(1, n):
        j = i
        while j > 0 and arr[j] < arr[j - 1]:
            arr[j], arr[j - 1] = arr[j - 1], arr[j]
            j -= 1
    return arr

arr = [1, 2, 4, 3, 5]
print(smoothsort(arr))
```

Output:

```
$$1, 2, 3, 4, 5]
```

*(Note: Real SmoothSort uses Leonardo heaps, complex but in-place and efficient.)*

#### C (Conceptual Heap Approach)

```c
#include <stdio.h>

void insertion_like_fix(int arr[], int n) {
    for (int i = 1; i < n; i++) {
        int j = i;
        while (j > 0 && arr[j] < arr[j - 1]) {
            int tmp = arr[j];
            arr[j] = arr[j - 1];
            arr[j - 1] = tmp;
            j--;
        }
    }
}

int main(void) {
    int arr[] = {1,2,4,3,5};
    int n = sizeof(arr)/sizeof(arr[0]);
    insertion_like_fix(arr, n);
    for (int i = 0; i < n; i++) printf("%d ", arr[i]);
    printf("\n");
}
```

This mimics SmoothSort's adaptiveness: fix locally, not globally.

### Why It Matters

- Adaptive: faster on nearly sorted data
- In-place: no extra memory
- Guaranteed bound: never worse than $O(n \log n)$
- Historical gem: Dijkstra's innovation in sorting theory

### A Gentle Proof (Why It Works)

For sorted input:

- Each insertion requires no swaps → $O(n)$

For random input:

- Heap restorations per insertion → $O(\log n)$  
- Total cost: $O(n \log n)$

| Case          | Behavior           | Complexity   |
| ------------- | ------------------ | ------------- |
| Best (Sorted) | Minimal swaps      | $O(n)$        |
| Average       | Moderate reheapify | $O(n \log n)$ |
| Worst         | Full heap rebuilds | $O(n \log n)$ |


SmoothSort adapts between these seamlessly.

### Try It Yourself

1. Sort `[1, 2, 4, 3, 5]` step by step.
2. Try `[1, 2, 3, 4, 5]`, measure comparisons.
3. Try `[5, 4, 3, 2, 1]`, full workload.
4. Count swaps in each case.
5. Compare to Heap Sort.
6. Visualize heap sizes as Leonardo sequence.
7. Implement run detection.
8. Experiment with large partially sorted arrays.
9. Track adaptive speedup.
10. Write Leonardo heap builder.

### Test Cases

| Input       | Output      | Notes          |
| ----------- | ----------- | -------------- |
| [1,2,4,3,5] | [1,2,3,4,5] | Minor disorder |
| [1,2,3,4,5] | [1,2,3,4,5] | Already sorted |
| [5,4,3,2,1] | [1,2,3,4,5] | Full rebuild   |
| [2,1,3,5,4] | [1,2,3,4,5] | Mixed case     |

### Complexity

| Aspect         | Value      |
| -------------- | ---------- |
| Time (Best)    | O(n)       |
| Time (Average) | O(n log n) |
| Time (Worst)   | O(n log n) |
| Space          | O(1)       |
| Stable         | No         |
| Adaptive       | Yes        |

SmoothSort glides gracefully across the array, fixing only what's broken.
It's sorting that *listens* to your data, quiet, clever, and precise.

### 135 Block Merge Sort

Block Merge Sort is a cache-efficient, stable sorting algorithm that merges data using small fixed-size blocks instead of large temporary arrays. It improves on standard Merge Sort by reducing memory usage and enhancing locality of reference, making it a great choice for modern hardware and large datasets.

It's designed to keep data cache-friendly, in-place (or nearly), and stable, making it a practical choice for systems with limited memory bandwidth or tight memory constraints.

### What Problem Are We Solving?

Classic Merge Sort is stable and O(n log n), but it needs O(n) extra space.
Block Merge Sort solves this by using blocks of fixed size (often √n) as temporary buffers for merging.

It aims to:

- Keep stability
- Use limited extra memory
- Maximize cache reuse
- Maintain predictable access patterns

Ideal for:

- Large arrays
- External memory sorting
- Systems with cache hierarchies

#### Example

Sort `[8, 3, 5, 1, 6, 2, 7, 4]`

| Step | Action                                       | Result                     |
| ---- | -------------------------------------------- | -------------------------- |
| 1    | Divide into sorted runs (via insertion sort) | [3,8], [1,5], [2,6], [4,7] |
| 2    | Merge adjacent blocks using buffer           | [1,3,5,8], [2,4,6,7]       |
| 3    | Merge merged blocks with block buffer        | [1,2,3,4,5,6,7,8]          |

Instead of full arrays, it uses small block buffers, fewer cache misses, less extra space.

### How Does It Work (Plain Language)?

Think of merging two sorted shelves in a library, but instead of taking all books off,
you move a small block at a time, swapping them in place with a small temporary cart.

You slide the buffer along the shelves, merge gradually, efficiently, with minimal movement.

#### Step-by-Step Process

| Step | Description                                                     |
| ---- | --------------------------------------------------------------- |
| 1    | Divide input into runs (sorted subarrays)                       |
| 2    | Use insertion sort or binary insertion to sort each run |
| 3    | Allocate small buffer (block)                                   |
| 4    | Merge runs pairwise using the block buffer                      |
| 5    | Repeat until one sorted array remains                           |

Block merges rely on rotations and buffer swapping to minimize extra space.

### Tiny Code (Easy Versions)

#### Python (Simplified Version)

This version simulates block merging using chunks.

```python
def insertion_sort(arr, start, end):
    for i in range(start + 1, end):
        key = arr[i]
        j = i - 1
        while j >= start and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key

def merge(arr, left, mid, right):
    left_part = arr[left:mid]
    right_part = arr[mid:right]
    i = j = 0
    k = left
    while i < len(left_part) and j < len(right_part):
        if left_part[i] <= right_part[j]:
            arr[k] = left_part[i]; i += 1
        else:
            arr[k] = right_part[j]; j += 1
        k += 1
    while i < len(left_part):
        arr[k] = left_part[i]; i += 1; k += 1
    while j < len(right_part):
        arr[k] = right_part[j]; j += 1; k += 1

def block_merge_sort(arr, block_size=32):
    n = len(arr)
    # Step 1: sort small blocks
    for start in range(0, n, block_size):
        end = min(start + block_size, n)
        insertion_sort(arr, start, end)

    # Step 2: merge adjacent blocks
    size = block_size
    while size < n:
        for left in range(0, n, 2 * size):
            mid = min(left + size, n)
            right = min(left + 2 * size, n)
            merge(arr, left, mid, right)
        size *= 2
    return arr

arr = [8, 3, 5, 1, 6, 2, 7, 4]
print(block_merge_sort(arr))
```

Output:

```
$$1, 2, 3, 4, 5, 6, 7, 8]
```

#### C (Simplified Concept)

```c
#include <stdio.h>

void insertion_sort(int arr[], int left, int right) {
    for (int i = left + 1; i < right; i++) {
        int key = arr[i], j = i - 1;
        while (j >= left && arr[j] > key) {
            arr[j + 1] = arr[j];
            j--;
        }
        arr[j + 1] = key;
    }
}

void merge(int arr[], int left, int mid, int right) {
    int n1 = mid - left, n2 = right - mid;
    int L[n1], R[n2];
    for (int i = 0; i < n1; i++) L[i] = arr[left + i];
    for (int j = 0; j < n2; j++) R[j] = arr[mid + j];

    int i = 0, j = 0, k = left;
    while (i < n1 && j < n2) arr[k++] = (L[i] <= R[j]) ? L[i++] : R[j++];
    while (i < n1) arr[k++] = L[i++];
    while (j < n2) arr[k++] = R[j++];
}

void block_merge_sort(int arr[], int n, int block_size) {
    for (int i = 0; i < n; i += block_size) {
        int end = (i + block_size < n) ? i + block_size : n;
        insertion_sort(arr, i, end);
    }
    for (int size = block_size; size < n; size *= 2) {
        for (int left = 0; left < n; left += 2 * size) {
            int mid = (left + size < n) ? left + size : n;
            int right = (left + 2 * size < n) ? left + 2 * size : n;
            if (mid < right) merge(arr, left, mid, right);
        }
    }
}

int main(void) {
    int arr[] = {8,3,5,1,6,2,7,4};
    int n = sizeof(arr)/sizeof(arr[0]);
    block_merge_sort(arr, n, 2);
    for (int i = 0; i < n; i++) printf("%d ", arr[i]);
    printf("\n");
}
```

### Why It Matters

- Stable like Merge Sort
- In-place or low-space variant
- Cache-efficient due to block locality
- Practical for large arrays or external sorting

### A Gentle Proof (Why It Works)

Each merge level processes $n$ elements: $O(n)$.  
Number of levels = $\log_2 (n / b)$, where $b$ is the block size.

So total:
$$
T(n) = O(n \log (n / b))
$$

When $b$ is large (like $\sqrt{n}$), space and time balance nicely.

| Phase         | Operation       | Cost            |
| -------------- | --------------- | ---------------- |
| Block sorting  | $\frac{n}{b} \times b^2$ | $O(nb)$          |
| Merging        | $\log (n / b) \times n$    | $O(n \log n)$    |

For typical settings, it's near $O(n \log n)$ but cache-optimized.


### Try It Yourself

1. Sort `[8,3,5,1,6,2,7,4]` with block size 2.
2. Increase block size to 4, compare steps.
3. Track number of merges.
4. Check stability with duplicates.
5. Compare with standard Merge Sort.
6. Time on sorted input (adaptive check).
7. Measure cache misses (simulated).
8. Try large array (10000+) for memory gain.
9. Mix ascending and descending runs.
10. Implement block buffer rotation manually.

### Test Cases

| Input             | Output            | Notes         |
| ----------------- | ----------------- | ------------- |
| [8,3,5,1,6,2,7,4] | [1,2,3,4,5,6,7,8] | Classic       |
| [5,5,3,3,1]       | [1,3,3,5,5]       | Stable        |
| [1,2,3,4]         | [1,2,3,4]         | Sorted        |
| [9,8,7]           | [7,8,9]           | Reverse order |

### Complexity

| Aspect         | Value      |
| -------------- | ---------- |
| Time (Best)    | O(n)       |
| Time (Average) | O(n log n) |
| Time (Worst)   | O(n log n) |
| Space          | O(b)       |
| Stable         | Yes        |
| Adaptive       | Partially  |

Block Merge Sort is the engineer's Merge Sort, same elegance, less memory, smarter on hardware.
It merges not by brute force, but by careful block juggling, balancing speed, space, and stability.

### 136 Adaptive Merge Sort

Adaptive Merge Sort is a stable, comparison-based sorting algorithm that adapts to existing order in the input data. It builds on the idea that real-world datasets are often partially sorted, so by detecting runs (already sorted sequences) and merging them intelligently, it can achieve O(n) time on nearly sorted data while retaining O(n log n) in the worst case.

It's a family of algorithms, including Natural Merge Sort, TimSort, and GrailSort, that all share one key insight: work less when less work is needed.

### What Problem Are We Solving?

Standard Merge Sort treats every input the same, even if it's already sorted.
Adaptive Merge Sort improves this by:

- Detecting sorted runs (ascending or descending)
- Merging runs instead of single elements
- Achieving linear time on sorted or partially sorted data

This makes it perfect for:

- Time series data
- Sorted or semi-sorted logs
- Incrementally updated lists

#### Example

Sort `[1, 2, 5, 3, 4, 6]`

| Step | Action                        | Result        |
| ---- | ----------------------------- | ------------- |
| 1    | Detect runs: [1,2,5], [3,4,6] | Found 2 runs  |
| 2    | Merge runs                    | [1,2,3,4,5,6] |
| 3    | Done                          | Sorted        |

Only one merge needed, input was nearly sorted, so runtime is close to O(n).

### How Does It Work (Plain Language)?

Imagine you're sorting a shelf of books that's mostly organized —
you don't pull all the books off; you just spot where order breaks, and fix those parts.

Adaptive Merge Sort does exactly this:

- Scan once to find sorted parts
- Merge runs using a stable merge procedure

It's lazy where it can be, efficient where it must be.

#### Step-by-Step Process

| Step | Description                                                   |
| ---- | ------------------------------------------------------------- |
| 1    | Scan input to find runs (ascending or descending)             |
| 2    | Reverse descending runs                                       |
| 3    | Push runs onto a stack                                        |
| 4    | Merge runs when stack conditions are violated (size or order) |
| 5    | Continue until one sorted run remains                         |

### Tiny Code (Easy Versions)

#### Python (Natural Merge Sort)

```python
def natural_merge_sort(arr):
    n = len(arr)
    runs = []
    i = 0
    # Step 1: Detect sorted runs
    while i < n:
        start = i
        i += 1
        while i < n and arr[i] >= arr[i - 1]:
            i += 1
        runs.append(arr[start:i])
    # Step 2: Merge runs pairwise
    while len(runs) > 1:
        new_runs = []
        for j in range(0, len(runs), 2):
            if j + 1 < len(runs):
                new_runs.append(merge(runs[j], runs[j+1]))
            else:
                new_runs.append(runs[j])
        runs = new_runs
    return runs[0]

def merge(left, right):
    i = j = 0
    result = []
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i]); i += 1
        else:
            result.append(right[j]); j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result

arr = [1, 2, 5, 3, 4, 6]
print(natural_merge_sort(arr))
```

Output:

```
$$1, 2, 3, 4, 5, 6]
```

#### C (Simplified Conceptual)

```c
#include <stdio.h>
#include <stdlib.h>

void merge(int arr[], int l, int m, int r) {
    int n1 = m - l + 1, n2 = r - m;
    int L[n1], R[n2];
    for (int i = 0; i < n1; i++) L[i] = arr[l + i];
    for (int j = 0; j < n2; j++) R[j] = arr[m + 1 + j];
    int i = 0, j = 0, k = l;
    while (i < n1 && j < n2) arr[k++] = (L[i] <= R[j]) ? L[i++] : R[j++];
    while (i < n1) arr[k++] = L[i++];
    while (j < n2) arr[k++] = R[j++];
}

void adaptive_merge_sort(int arr[], int n) {
    int start = 0;
    while (start < n - 1) {
        int mid = start;
        while (mid < n - 1 && arr[mid] <= arr[mid + 1]) mid++;
        int end = mid + 1;
        while (end < n - 1 && arr[end] <= arr[end + 1]) end++;
        if (end < n) merge(arr, start, mid, end);
        start = end + 1;
    }
}

int main(void) {
    int arr[] = {1, 2, 5, 3, 4, 6};
    int n = sizeof(arr)/sizeof(arr[0]);
    adaptive_merge_sort(arr, n);
    for (int i = 0; i < n; i++) printf("%d ", arr[i]);
    printf("\n");
}
```

### Why It Matters

- Stable and adaptive
- Linear time on nearly sorted data
- Works well for real-world sequences
- Forms the core idea behind TimSort
- Requires no extra knowledge about data

It's the sorting algorithm that notices your data's effort and rewards it.

### A Gentle Proof (Why It Works)

Let average run length = $r$.  
Then number of runs ≈ $n / r$.  
Each merge = $O(n)$ per level.  
Depth = $O(\log (n / r))$.

So total cost:
$$
T(n) = O(n \log (n / r))
$$

If $r = n$ (already sorted): $O(n)$.  
If $r = 1$: $O(n \log n)$.

| Case          | Run Length | Complexity         |
| -------------- | ----------- | ------------------ |
| Sorted         | $r = n$     | $O(n)$             |
| Nearly sorted  | $r$ large   | $O(n \log (n / r))$ |
| Random         | $r$ small   | $O(n \log n)$      |


### Try It Yourself

1. Sort `[1, 2, 5, 3, 4, 6]` step by step.
2. Try `[1, 2, 3, 4, 5]`, should detect 1 run.
3. Reverse a section, see new runs.
4. Merge runs manually using table.
5. Compare performance to Merge Sort.
6. Add duplicates, check stability.
7. Use 10k sorted elements, time the run.
8. Mix ascending and descending subarrays.
9. Visualize run detection.
10. Implement descending-run reversal.

### Test Cases

| Input         | Output        | Notes               |
| ------------- | ------------- | ------------------- |
| [1,2,5,3,4,6] | [1,2,3,4,5,6] | Two runs            |
| [1,2,3,4,5]   | [1,2,3,4,5]   | Already sorted      |
| [5,4,3,2,1]   | [1,2,3,4,5]   | Reverse (many runs) |
| [2,2,1,1]     | [1,1,2,2]     | Stable              |

### Complexity

| Aspect         | Value      |
| -------------- | ---------- |
| Time (Best)    | O(n)       |
| Time (Average) | O(n log n) |
| Time (Worst)   | O(n log n) |
| Space          | O(n)       |
| Stable         | Yes        |
| Adaptive       | Yes        |

Adaptive Merge Sort is like a thoughtful sorter, it looks before it leaps.
If your data's halfway there, it'll meet it in the middle.

### 137 PDQSort (Pattern-Defeating QuickSort)

PDQSort is a modern, adaptive, in-place sorting algorithm that extends QuickSort with pattern detection, branchless partitioning, and fallback mechanisms to guarantee $O(n \log n)$ performance even on adversarial inputs.

Invented by Orson Peters, it's used in C++'s `std::sort()` (since C++17) and often outperforms traditional QuickSort and IntroSort in real-world scenarios due to better cache behavior, branch prediction, and adaptive pivoting.

### What Problem Are We Solving?

Classic QuickSort performs well *on average* but can degrade to (O(n^2)) on structured or repetitive data.
PDQSort solves this by:

- Detecting bad patterns (e.g., sorted input)
- Switching strategy (to heap sort or insertion sort)
- Branchless partitioning for modern CPUs
- Adaptive pivot selection (median-of-3, Tukey ninther)

It keeps speed, stability of performance, and cache-friendliness.

Perfect for:

- Large unsorted datasets
- Partially sorted data
- Real-world data with patterns

#### Example

Sort `[1, 2, 3, 4, 5]` (already sorted)

| Step | Action                   | Result             |
| ---- | ------------------------ | ------------------ |
| 1    | Detect sorted pattern    | Pattern found      |
| 2    | Switch to Insertion Sort | Efficient handling |
| 3    | Output sorted            | [1,2,3,4,5]        |

Instead of recursive QuickSort calls, PDQSort defeats the pattern by adapting.

### How Does It Work (Plain Language)?

PDQSort is like a smart chef:

- It tastes the input first (checks pattern)
- Chooses a recipe (pivot rule, sorting fallback)
- Adjusts to the kitchen conditions (CPU caching, branching)

It never wastes effort, detecting when recursion or comparisons are unnecessary.

#### Step-by-Step Process

| Step | Description                                                 |
| ---- | ----------------------------------------------------------- |
| 1    | Pick pivot adaptively (median-of-3, ninther)                |
| 2    | Partition elements into `< pivot` and `> pivot`             |
| 3    | Detect bad patterns (sorted, reversed, many equal elements) |
| 4    | Apply branchless partitioning to reduce CPU mispredictions  |
| 5    | Recurse or switch to heap/insertion sort if needed          |
| 6    | Use tail recursion elimination                              |

### Tiny Code (Easy Versions)

#### Python (Conceptual Simplified PDQSort)

```python
def pdqsort(arr):
    def insertion_sort(a, lo, hi):
        for i in range(lo + 1, hi):
            key = a[i]
            j = i - 1
            while j >= lo and a[j] > key:
                a[j + 1] = a[j]
                j -= 1
            a[j + 1] = key

    def partition(a, lo, hi):
        pivot = a[(lo + hi) // 2]
        i, j = lo, hi - 1
        while True:
            while a[i] < pivot: i += 1
            while a[j] > pivot: j -= 1
            if i >= j: return j
            a[i], a[j] = a[j], a[i]
            i += 1; j -= 1

    def _pdqsort(a, lo, hi, depth):
        n = hi - lo
        if n <= 16:
            insertion_sort(a, lo, hi)
            return
        if depth == 0:
            a[lo:hi] = sorted(a[lo:hi])  # heap fallback
            return
        mid = partition(a, lo, hi)
        _pdqsort(a, lo, mid + 1, depth - 1)
        _pdqsort(a, mid + 1, hi, depth - 1)

    _pdqsort(arr, 0, len(arr), len(arr).bit_length() * 2)
    return arr

arr = [1, 5, 3, 4, 2]
print(pdqsort(arr))
```

Output:

```
$$1, 2, 3, 4, 5]
```

#### C (Simplified PDQ-Style)

```c
#include <stdio.h>
#include <stdlib.h>

void insertion_sort(int arr[], int lo, int hi) {
    for (int i = lo + 1; i < hi; i++) {
        int key = arr[i], j = i - 1;
        while (j >= lo && arr[j] > key) {
            arr[j + 1] = arr[j];
            j--;
        }
        arr[j + 1] = key;
    }
}

int partition(int arr[], int lo, int hi) {
    int pivot = arr[(lo + hi) / 2];
    int i = lo, j = hi - 1;
    while (1) {
        while (arr[i] < pivot) i++;
        while (arr[j] > pivot) j--;
        if (i >= j) return j;
        int tmp = arr[i]; arr[i] = arr[j]; arr[j] = tmp;
        i++; j--;
    }
}

void pdqsort(int arr[], int lo, int hi, int depth) {
    int n = hi - lo;
    if (n <= 16) { insertion_sort(arr, lo, hi); return; }
    if (depth == 0) { qsort(arr + lo, n, sizeof(int), (__compar_fn_t)strcmp); return; }
    int mid = partition(arr, lo, hi);
    pdqsort(arr, lo, mid + 1, depth - 1);
    pdqsort(arr, mid + 1, hi, depth - 1);
}

int main(void) {
    int arr[] = {1, 5, 3, 4, 2};
    int n = sizeof(arr)/sizeof(arr[0]);
    pdqsort(arr, 0, n, 2 * 32);
    for (int i = 0; i < n; i++) printf("%d ", arr[i]);
    printf("\n");
}
```

### Why It Matters

- Modern default for high-performance sorting
- Pattern detection avoids worst cases
- Branchless partitioning → fewer CPU stalls
- Heap fallback ensures $O(n \log n)$ bound
- Adaptive like TimSort, but in-place

PDQSort combines speed, safety, and hardware awareness.

### A Gentle Proof (Why It Works)

PDQSort adds mechanisms to defeat QuickSort's pitfalls:

1. Bad pattern detection → early fallback
2. Balanced pivoting → near-equal splits
3. Branchless operations → efficient execution

So overall:
$$
T(n) = O(n \log n)
$$
Best case (sorted): near O(n), thanks to early detection.

| Case    | Behavior                | Complexity |
| ------- | ----------------------- | ---------- |
| Best    | Sorted or nearly sorted | O(n)       |
| Average | Random                  | O(n log n) |
| Worst   | Adversarial             | O(n log n) |

### Try It Yourself

1. Sort `[1,2,3,4,5]` → detect sorted input.
2. Try `[5,4,3,2,1]` → reversed.
3. Random input of 1000 numbers.
4. Duplicate-heavy `[5,5,5,5]`.
5. Patterned input `[1,3,2,4,3,5,4]`.
6. Compare with QuickSort and HeapSort.
7. Count recursion depth.
8. Benchmark branchless vs classic partition.
9. Visualize fallback triggers.
10. Measure comparisons per element.

### Test Cases

| Input       | Output      | Notes      |
| ----------- | ----------- | ---------- |
| [1,5,3,4,2] | [1,2,3,4,5] | Random     |
| [1,2,3,4,5] | [1,2,3,4,5] | Sorted     |
| [5,4,3,2,1] | [1,2,3,4,5] | Reversed   |
| [5,5,5,5]   | [5,5,5,5]   | Duplicates |

### Complexity

| Aspect         | Value      |
| -------------- | ---------- |
| Time (Best)    | O(n)       |
| Time (Average) | O(n log n) |
| Time (Worst)   | O(n log n) |
| Space          | O(log n)   |
| Stable         | No         |
| Adaptive       | Yes        |

PDQSort is the ninja of sorting, lightning fast, pattern-aware, and always one step ahead of your data.
It doesn't just sort, it *outsmarts* the input.

### 138 WikiSort

WikiSort is a stable, in-place merge sort created by Mike Day, designed to combine the stability of Merge Sort with the low memory usage of in-place algorithms. It achieves O(n log n) performance and uses only O(1) extra memory, making it one of the most space-efficient stable sorts available.

Unlike classic Merge Sort, which allocates a full-size temporary array, WikiSort performs block merges with rotation operations, merging sorted regions directly within the array.

### What Problem Are We Solving?

Most stable sorting algorithms (like Merge Sort) need O(n) extra space.
WikiSort solves this by performing stable merges in place, using:

- Block rotations instead of large buffers
- Adaptive merging when possible
- Small local buffers reused efficiently

Perfect for:

- Memory-constrained environments
- Sorting large arrays in embedded systems
- Stable sorts without heavy allocation

#### Example

Sort `[3, 5, 1, 2, 4]`

| Step | Action                      | Result             |
| ---- | --------------------------- | ------------------ |
| 1    | Divide into sorted runs     | [3,5], [1,2,4]     |
| 2    | Merge using block rotations | [1,2,3,4,5]        |
| 3    | Done, stable and in-place  | Final sorted array |

Result: `[1,2,3,4,5]`, sorted, stable, and minimal extra space.

### How Does It Work (Plain Language)?

Think of two sorted shelves of books, instead of moving all books to a table, you rotate sections in place so the shelves merge seamlessly.

WikiSort keeps a small local buffer (like a tray), uses it to move small chunks, then rotates segments of the array into place. It never needs a second full array.

#### Step-by-Step Process

| Step | Description                            |
| ---- | -------------------------------------- |
| 1    | Split array into sorted runs           |
| 2    | Allocate small buffer (≈ √n elements)  |
| 3    | Merge adjacent runs using buffer       |
| 4    | Rotate blocks to maintain stability    |
| 5    | Repeat until one sorted region remains |

### Tiny Code (Easy Versions)

#### Python (Simplified In-Place Stable Merge)

This is a conceptual demonstration of stable in-place merging.

```python
def rotate(arr, start, mid, end):
    arr[start:end] = arr[mid:end] + arr[start:mid]

def merge_in_place(arr, start, mid, end):
    left = arr[start:mid]
    i, j, k = 0, mid, start
    while i < len(left) and j < end:
        if left[i] <= arr[j]:
            arr[k] = left[i]; i += 1
        else:
            val = arr[j]
            rotate(arr, i + start, j, j + 1)
            arr[k] = val
            j += 1
        k += 1
    while i < len(left):
        arr[k] = left[i]; i += 1; k += 1

def wiki_sort(arr):
    n = len(arr)
    size = 1
    while size < n:
        for start in range(0, n, 2 * size):
            mid = min(start + size, n)
            end = min(start + 2 * size, n)
            if mid < end:
                merge_in_place(arr, start, mid, end)
        size *= 2
    return arr

arr = [3,5,1,2,4]
print(wiki_sort(arr))
```

Output:

```
$$1, 2, 3, 4, 5]
```

#### C (Simplified Conceptual Version)

```c
#include <stdio.h>

void rotate(int arr[], int start, int mid, int end) {
    int temp[end - start];
    int idx = 0;
    for (int i = mid; i < end; i++) temp[idx++] = arr[i];
    for (int i = start; i < mid; i++) temp[idx++] = arr[i];
    for (int i = 0; i < end - start; i++) arr[start + i] = temp[i];
}

void merge_in_place(int arr[], int start, int mid, int end) {
    int i = start, j = mid;
    while (i < j && j < end) {
        if (arr[i] <= arr[j]) {
            i++;
        } else {
            int value = arr[j];
            rotate(arr, i, j, j + 1);
            arr[i] = value;
            i++; j++;
        }
    }
}

void wiki_sort(int arr[], int n) {
    for (int size = 1; size < n; size *= 2) {
        for (int start = 0; start < n; start += 2 * size) {
            int mid = (start + size < n) ? start + size : n;
            int end = (start + 2 * size < n) ? start + 2 * size : n;
            if (mid < end) merge_in_place(arr, start, mid, end);
        }
    }
}

int main(void) {
    int arr[] = {3,5,1,2,4};
    int n = sizeof(arr)/sizeof(arr[0]);
    wiki_sort(arr, n);
    for (int i = 0; i < n; i++) printf("%d ", arr[i]);
    printf("\n");
}
```

### Why It Matters

- Stable and in-place (O(1) extra space)
- Practical for memory-limited systems
- Predictable performance
- Cache-friendly merges
- Great balance of theory and practice

It brings the best of Merge Sort (stability) and in-place algorithms (low memory).

### A Gentle Proof (Why It Works)

Each merge takes $O(n)$, and there are $O(\log n)$ levels of merging:

$$
T(n) = O(n \log n)
$$

Extra memory = small buffer ($O(\sqrt{n})$) or even constant space.

| Phase           | Operation            | Cost           |
| ---------------- | -------------------- | -------------- |
| Block detection  | Scan runs            | $O(n)$         |
| Merging          | Rotation-based merge | $O(n \log n)$  |
| Space            | Fixed buffer         | $O(1)$         |


### Try It Yourself

1. Sort `[3,5,1,2,4]` step by step.
2. Visualize rotations during merge.
3. Add duplicates `[2,2,3,1]`, verify stability.
4. Increase size to 16, track buffer reuse.
5. Compare with Merge Sort memory usage.
6. Measure swaps vs Merge Sort.
7. Try `[1,2,3,4]`, minimal rotations.
8. Reverse `[5,4,3,2,1]`, max work.
9. Implement block rotation helper.
10. Measure runtime on sorted input.

### Test Cases

| Input       | Output      | Notes           |
| ----------- | ----------- | --------------- |
| [3,5,1,2,4] | [1,2,3,4,5] | Basic           |
| [5,4,3,2,1] | [1,2,3,4,5] | Worst case      |
| [1,2,3,4]   | [1,2,3,4]   | Already sorted  |
| [2,2,3,1]   | [1,2,2,3]   | Stable behavior |

### Complexity

| Aspect         | Value      |
| -------------- | ---------- |
| Time (Best)    | O(n)       |
| Time (Average) | O(n log n) |
| Time (Worst)   | O(n log n) |
| Space          | O(1)       |
| Stable         | Yes        |
| Adaptive       | Yes        |

WikiSort is the minimalist's Merge Sort, stable, elegant, and almost memory-free.
It merges not by copying, but by rotating, smooth, steady, and space-savvy.

### 139 GrailSort

GrailSort (short for "Greedy Adaptive In-place stable Sort") is a stable, in-place, comparison-based sorting algorithm that merges sorted subarrays using block merging and local buffers.

Created by Michał Oryńczak, GrailSort combines the stability and adaptiveness of Merge Sort with in-place operation, needing only a tiny internal buffer (often $O(\sqrt{n})$) or even no extra memory in its pure variant.

It's designed for practical stable sorting when memory is tight, achieving $O(n \log n)$ worst-case time.

### What Problem Are We Solving?

Typical stable sorting algorithms (like Merge Sort) require O(n) extra space.
GrailSort solves this by:

- Using small local buffers instead of large arrays
- Performing in-place stable merges
- Detecting and reusing natural runs (adaptive behavior)

Perfect for:

- Memory-constrained systems
- Embedded devices
- Large stable sorts on limited RAM

#### Example

Sort `[4, 1, 3, 2, 5]`

| Step | Action                          | Result            |
| ---- | ------------------------------- | ----------------- |
| 1    | Detect short runs               | [4,1], [3,2], [5] |
| 2    | Sort each run                   | [1,4], [2,3], [5] |
| 3    | Merge runs using block rotation | [1,2,3,4,5]       |
| 4    | Stable order preserved          | Done              |

All merges are done in-place, with a small reusable buffer.

### How Does It Work (Plain Language)?

Think of GrailSort like a clever librarian with a tiny desk (the buffer).
Instead of taking all books off the shelf, they:

- Divide shelves into small sorted groups,
- Keep a few aside as a helper buffer,
- Merge shelves directly on the rack by rotating sections in place.

It's stable, in-place, and adaptive, a rare combination.

#### Step-by-Step Process

| Step | Description                                     |
| ---- | ----------------------------------------------- |
| 1    | Detect and sort small runs (Insertion Sort)     |
| 2    | Choose small buffer (e.g., √n elements)         |
| 3    | Merge runs pairwise using buffer and rotation   |
| 4    | Gradually increase block size (1, 2, 4, 8, ...) |
| 5    | Continue merging until fully sorted             |

The algorithm's structure mirrors Merge Sort but uses block rotation to avoid copying large chunks.

### Tiny Code (Easy Versions)

#### Python (Simplified Concept)

Below is a simplified stable block-merge inspired by GrailSort's principles.

```python
def rotate(arr, start, mid, end):
    arr[start:end] = arr[mid:end] + arr[start:mid]

def merge_in_place(arr, left, mid, right):
    i, j = left, mid
    while i < j and j < right:
        if arr[i] <= arr[j]:
            i += 1
        else:
            val = arr[j]
            rotate(arr, i, j, j + 1)
            arr[i] = val
            i += 1
            j += 1

def grailsort(arr):
    n = len(arr)
    size = 1
    while size < n:
        for start in range(0, n, 2 * size):
            mid = min(start + size, n)
            end = min(start + 2 * size, n)
            if mid < end:
                merge_in_place(arr, start, mid, end)
        size *= 2
    return arr

arr = [4,1,3,2,5]
print(grailsort(arr))
```

Output:

```
$$1, 2, 3, 4, 5]
```

#### C (Simplified Idea)

```c
#include <stdio.h>

void rotate(int arr[], int start, int mid, int end) {
    int temp[end - start];
    int idx = 0;
    for (int i = mid; i < end; i++) temp[idx++] = arr[i];
    for (int i = start; i < mid; i++) temp[idx++] = arr[i];
    for (int i = 0; i < end - start; i++) arr[start + i] = temp[i];
}

void merge_in_place(int arr[], int start, int mid, int end) {
    int i = start, j = mid;
    while (i < j && j < end) {
        if (arr[i] <= arr[j]) i++;
        else {
            int val = arr[j];
            rotate(arr, i, j, j + 1);
            arr[i] = val;
            i++; j++;
        }
    }
}

void grailsort(int arr[], int n) {
    for (int size = 1; size < n; size *= 2) {
        for (int start = 0; start < n; start += 2 * size) {
            int mid = (start + size < n) ? start + size : n;
            int end = (start + 2 * size < n) ? start + 2 * size : n;
            if (mid < end) merge_in_place(arr, start, mid, end);
        }
    }
}

int main(void) {
    int arr[] = {4,1,3,2,5};
    int n = sizeof(arr)/sizeof(arr[0]);
    grailsort(arr, n);
    for (int i = 0; i < n; i++) printf("%d ", arr[i]);
    printf("\n");
}
```

### Why It Matters

- Stable and in-place (only small buffer)
- Adaptive, faster on partially sorted data
- Predictable performance
- Ideal for limited memory systems
- Used in practical sorting libraries and research

It's the gold standard for stable, low-space sorts.

### A Gentle Proof (Why It Works)

Each merge level processes $O(n)$ elements.  
There are $O(\log n)$ merge levels.  
Thus:

$$
T(n) = O(n \log n)
$$

A small buffer ($O(\sqrt{n})$) is reused, yielding in-place stability.

| Phase        | Operation      | Cost             |
| ------------- | -------------- | ---------------- |
| Run sorting   | Insertion Sort | $O(n)$           |
| Block merges  | Rotation-based | $O(n \log n)$    |
| Space         | Local buffer   | $O(1)$ or $O(\sqrt{n})$ |

### Try It Yourself

1. Sort `[4,1,3,2,5]` step by step.
2. Try `[1,2,3,4,5]`, no merges needed.
3. Check duplicates `[2,2,1,1]`, verify stability.
4. Experiment with `[10,9,8,7,6,5]`.
5. Visualize rotations during merge.
6. Change block size, observe performance.
7. Implement √n buffer manually.
8. Compare with Merge Sort (space).
9. Measure time vs WikiSort.
10. Try large array (10k elements).

### Test Cases

| Input       | Output      | Notes          |
| ----------- | ----------- | -------------- |
| [4,1,3,2,5] | [1,2,3,4,5] | Basic test     |
| [1,2,3,4]   | [1,2,3,4]   | Already sorted |
| [5,4,3,2,1] | [1,2,3,4,5] | Reverse        |
| [2,2,1,1]   | [1,1,2,2]   | Stable         |

### Complexity

| Aspect         | Value         |
| -------------- | ------------- |
| Time (Best)    | O(n)          |
| Time (Average) | O(n log n)    |
| Time (Worst)   | O(n log n)    |
| Space          | O(1) to O(√n) |
| Stable         | Yes           |
| Adaptive       | Yes           |

GrailSort is the gentle engineer of sorting, steady, stable, and space-wise.
It doesn't rush; it rearranges with precision, merging order from within.

### 140 Adaptive Hybrid Sort

Adaptive Hybrid Sort is a meta-sorting algorithm that dynamically combines multiple sorting strategies, such as QuickSort, Merge Sort, Insertion Sort, and Heap Sort, depending on the data characteristics and runtime patterns it detects.

It adapts in real-time, switching between methods based on factors like array size, degree of pre-sortedness, data distribution, and recursion depth. This makes it a universal, practical sorter optimized for diverse workloads.

### What Problem Are We Solving?

No single sorting algorithm is best for all situations:

- QuickSort is fast on random data but unstable and bad in worst case.
- Merge Sort is stable but memory-hungry.
- Insertion Sort is great for small or nearly sorted arrays.
- Heap Sort guarantees O(n log n) but has poor locality.

Adaptive Hybrid Sort solves this by blending algorithms:

1. Start with QuickSort for speed.
2. Detect sorted or small regions → switch to Insertion Sort.
3. Detect bad pivot patterns → switch to Heap Sort.
4. Detect stability needs or patterns → use Merge Sort.

It's a unified, self-tuning sorting system.

#### Example

Sort `[2, 3, 5, 4, 6, 7, 8, 1]`

| Step | Detection                     | Action                             |
| ---- | ----------------------------- | ---------------------------------- |
| 1    | Mostly sorted except last few | Switch to Insertion Sort           |
| 2    | Sort locally                  | [1,2,3,4,5,6,7,8]                  |
| 3    | Done                          | Adaptive path chosen automatically |

If input were random, it would stay with QuickSort. If adversarial, it would pivot to Heap Sort.

### How Does It Work (Plain Language)?

Imagine a skilled chef with many tools, knives, mixers, ovens.
When slicing carrots (small data), they use a paring knife (Insertion Sort).
When breaking a tough root (unsorted array), they grab a heavy cleaver (QuickSort).
When something's too complex, they use machinery (Merge Sort).

Adaptive Hybrid Sort works the same way, choosing the right tool at the right time.

#### Step-by-Step Process

| Step | Description                                          |
| ---- | ---------------------------------------------------- |
| 1    | Start with QuickSort (good average case)             |
| 2    | If recursion depth too high → switch to Heap Sort    |
| 3    | If subarray small (≤ threshold) → use Insertion Sort |
| 4    | If stable sorting required → use Merge Sort          |
| 5    | If data partially sorted → use TimSort-like merge    |
| 6    | Combine results for final sorted output              |

### Tiny Code (Easy Versions)

#### Python (Simplified Hybrid Sort)

```python
def insertion_sort(arr, left, right):
    for i in range(left + 1, right):
        key = arr[i]
        j = i - 1
        while j >= left and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key

def heapify(arr, n, i):
    largest = i
    l = 2*i + 1
    r = 2*i + 2
    if l < n and arr[l] > arr[largest]:
        largest = l
    if r < n and arr[r] > arr[largest]:
        largest = r
    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]
        heapify(arr, n, largest)

def heap_sort(arr):
    n = len(arr)
    for i in range(n//2 - 1, -1, -1):
        heapify(arr, n, i)
    for i in range(n - 1, 0, -1):
        arr[0], arr[i] = arr[i], arr[0]
        heapify(arr, i, 0)

def partition(arr, low, high):
    pivot = arr[(low + high) // 2]
    i, j = low, high
    while i <= j:
        while arr[i] < pivot: i += 1
        while arr[j] > pivot: j -= 1
        if i <= j:
            arr[i], arr[j] = arr[j], arr[i]
            i += 1; j -= 1
    return i

def hybrid_sort(arr, low=0, high=None, depth_limit=None):
    if high is None:
        high = len(arr) - 1
    if depth_limit is None:
        import math
        depth_limit = 2 * math.floor(math.log2(len(arr) + 1))
    size = high - low + 1
    if size <= 16:
        insertion_sort(arr, low, high + 1)
        return
    if depth_limit == 0:
        sub = arr[low:high + 1]
        heap_sort(sub)
        arr[low:high + 1] = sub
        return
    pivot_index = partition(arr, low, high)
    if low < pivot_index - 1:
        hybrid_sort(arr, low, pivot_index - 1, depth_limit - 1)
    if pivot_index < high:
        hybrid_sort(arr, pivot_index, high, depth_limit - 1)

arr = [2, 3, 5, 4, 6, 7, 8, 1]
hybrid_sort(arr)
print(arr)
```

Output:

```
$$1, 2, 3, 4, 5, 6, 7, 8]
```

#### C (Conceptual Hybrid Sort)

```c
#include <stdio.h>
#include <math.h>

void insertion_sort(int arr[], int left, int right) {
    for (int i = left + 1; i < right; i++) {
        int key = arr[i], j = i - 1;
        while (j >= left && arr[j] > key) {
            arr[j + 1] = arr[j];
            j--;
        }
        arr[j + 1] = key;
    }
}

void heapify(int arr[], int n, int i) {
    int largest = i;
    int l = 2*i + 1, r = 2*i + 2;
    if (l < n && arr[l] > arr[largest]) largest = l;
    if (r < n && arr[r] > arr[largest]) largest = r;
    if (largest != i) {
        int tmp = arr[i]; arr[i] = arr[largest]; arr[largest] = tmp;
        heapify(arr, n, largest);
    }
}

void heap_sort(int arr[], int n) {
    for (int i = n/2 - 1; i >= 0; i--) heapify(arr, n, i);
    for (int i = n - 1; i > 0; i--) {
        int tmp = arr[0]; arr[0] = arr[i]; arr[i] = tmp;
        heapify(arr, i, 0);
    }
}

int partition(int arr[], int low, int high) {
    int pivot = arr[(low + high) / 2];
    int i = low, j = high;
    while (i <= j) {
        while (arr[i] < pivot) i++;
        while (arr[j] > pivot) j--;
        if (i <= j) {
            int tmp = arr[i]; arr[i] = arr[j]; arr[j] = tmp;
            i++; j--;
        }
    }
    return i;
}

void hybrid_sort(int arr[], int low, int high, int depth_limit) {
    int size = high - low + 1;
    if (size <= 16) { insertion_sort(arr, low, high + 1); return; }
    if (depth_limit == 0) { heap_sort(arr + low, size); return; }
    int p = partition(arr, low, high);
    if (low < p - 1) hybrid_sort(arr, low, p - 1, depth_limit - 1);
    if (p < high) hybrid_sort(arr, p, high, depth_limit - 1);
}

int main(void) {
    int arr[] = {2,3,5,4,6,7,8,1};
    int n = sizeof(arr)/sizeof(arr[0]);
    int depth_limit = 2 * log2(n);
    hybrid_sort(arr, 0, n - 1, depth_limit);
    for (int i = 0; i < n; i++) printf("%d ", arr[i]);
    printf("\n");
}
```

### Why It Matters

- Adaptive to input shape
- Hybrid = flexibility + safety
- Stable runtime across data types
- Real-world robust for mixed data
- Balances speed, memory, stability, and predictability

### A Gentle Proof (Why It Works)

Let $T(n)$ be the runtime.  
Each phase is $O(n)$, and recursion depth is $O(\log n)$.  
Adaptive switching ensures no pathological behavior.

$$
T(n) = O(n \log n)
$$

Best case (sorted): Insertion Sort runs in $O(n)$.  
Worst case (adversarial): Heap fallback → $O(n \log n)$.

| Case        | Behavior       | Complexity   |
| ------------ | -------------- | ------------- |
| Sorted       | Insertion Sort | $O(n)$        |
| Random       | QuickSort      | $O(n \log n)$ |
| Adversarial  | Heap Sort      | $O(n \log n)$ |


### Try It Yourself

1. Sort `[2,3,5,4,6,7,8,1]`.
2. Try `[1,2,3,4,5,6]` → insertion path.
3. Reverse `[9,8,7,6,5]` → heap path.
4. Mix sorted + random halves.
5. Measure recursion depth.
6. Increase threshold to 32.
7. Add duplicates → observe stability.
8. Compare with IntroSort, TimSort.
9. Benchmark on large random data.
10. Visualize switch decisions.

### Test Cases

| Input             | Output            | Notes      |
| ----------------- | ----------------- | ---------- |
| [2,3,5,4,6,7,8,1] | [1,2,3,4,5,6,7,8] | Random     |
| [1,2,3,4,5]       | [1,2,3,4,5]       | Sorted     |
| [5,4,3,2,1]       | [1,2,3,4,5]       | Reverse    |
| [10,10,9,9,8]     | [8,9,9,10,10]     | Duplicates |

### Complexity

| Aspect         | Value      |
| -------------- | ---------- |
| Time (Best)    | O(n)       |
| Time (Average) | O(n log n) |
| Time (Worst)   | O(n log n) |
| Space          | O(log n)   |
| Stable         | No         |
| Adaptive       | Yes        |

Adaptive Hybrid Sort is the chameleon of sorting, it watches the data, reads the room, and chooses the perfect move.
Fast when it can be, safe when it must be.

## Section 15. Special sorts 

### 141 Cycle Sort

Cycle Sort is a comparison-based sorting algorithm designed to minimize the number of writes. It's ideal when writing to memory or storage is expensive (like EEPROM or flash memory), since each element is written exactly once into its final position.

It achieves O(n²) comparisons but performs the minimal possible number of writes, making it unique among sorting algorithms.

### What Problem Are We Solving?

Most sorting algorithms (like QuickSort or Merge Sort) swap elements multiple times before they reach their final position.
If each write is costly (e.g., embedded systems or flash memory), that's wasteful.

Cycle Sort asks:
"How can we put each element *directly* where it belongs, in one cycle, with as few writes as possible?"

#### Example

Sort `[3, 1, 2]`

| Step  | Element   | Correct Position | Action              |
| ----- | --------- | ---------------- | ------------------- |
| 1     | 3         | Index 2          | Swap 3 → position 2 |
| 2     | 2         | Index 1          | Swap 2 → position 1 |
| 3     | 1         | Index 0          | Done                |
| Final | [1, 2, 3] | Sorted           | ✅ Minimal writes    |

Each element cycles into place once.

### How Does It Work (Plain Language)?

Think of it like putting books on a shelf:

- You pick one book (element),
- Figure out where it belongs,
- Put it there, swapping with whatever's currently there,
- Repeat until every book is in the right spot.

Each cycle ensures every element reaches its final position exactly once.

#### Step-by-Step Process

| Step | Description                                                  |
| ---- | ------------------------------------------------------------ |
| 1    | Loop through array positions                                 |
| 2    | For each position, count how many elements are smaller       |
| 3    | That count = final position                                  |
| 4    | If not already correct, cycle the element to its right place |
| 5    | Continue cycling until original element returns to start     |
| 6    | Move to next position and repeat                             |

### Tiny Code (Easy Versions)

#### Python

```python
def cycle_sort(arr):
    n = len(arr)
    writes = 0
    for cycle_start in range(n - 1):
        item = arr[cycle_start]
        pos = cycle_start
        for i in range(cycle_start + 1, n):
            if arr[i] < item:
                pos += 1
        if pos == cycle_start:
            continue
        while item == arr[pos]:
            pos += 1
        arr[pos], item = item, arr[pos]
        writes += 1
        while pos != cycle_start:
            pos = cycle_start
            for i in range(cycle_start + 1, n):
                if arr[i] < item:
                    pos += 1
            while item == arr[pos]:
                pos += 1
            arr[pos], item = item, arr[pos]
            writes += 1
    print("Total writes:", writes)
    return arr

arr = [3, 1, 2, 4]
print(cycle_sort(arr))
```

Output:

```
Total writes: 3
$$1, 2, 3, 4]
```

#### C

```c
#include <stdio.h>

void cycle_sort(int arr[], int n) {
    int writes = 0;
    for (int cycle_start = 0; cycle_start < n - 1; cycle_start++) {
        int item = arr[cycle_start];
        int pos = cycle_start;

        for (int i = cycle_start + 1; i < n; i++)
            if (arr[i] < item)
                pos++;

        if (pos == cycle_start) continue;

        while (item == arr[pos]) pos++;
        int temp = arr[pos]; arr[pos] = item; item = temp;
        writes++;

        while (pos != cycle_start) {
            pos = cycle_start;
            for (int i = cycle_start + 1; i < n; i++)
                if (arr[i] < item)
                    pos++;
            while (item == arr[pos]) pos++;
            temp = arr[pos]; arr[pos] = item; item = temp;
            writes++;
        }
    }
    printf("Total writes: %d\n", writes);
}

int main(void) {
    int arr[] = {3, 1, 2, 4};
    int n = sizeof(arr) / sizeof(arr[0]);
    cycle_sort(arr, n);
    for (int i = 0; i < n; i++) printf("%d ", arr[i]);
    printf("\n");
}
```

Output:

```
Total writes: 3  
1 2 3 4
```

### Why It Matters

- Minimizes writes (useful for flash memory, EEPROMs)
- In-place
- Deterministic writes = fewer wear cycles
- Educational example of permutation cycles in sorting

Not fast, but *frugal*, every move counts.

### A Gentle Proof (Why It Works)

Each element moves to its correct position once.  
If array size is $n$, total writes $\le n$.


Counting smaller elements ensures correctness:
$$
\text{pos}(x) = |{y \mid y < x}|
$$

Each cycle resolves one permutation cycle of misplaced elements.
Thus, algorithm terminates with all items placed exactly once.

### Try It Yourself

1. Sort `[3, 1, 2]` step by step.
2. Count how many writes you perform.
3. Try `[4, 3, 2, 1]`, maximum cycles.
4. Try `[1, 2, 3, 4]`, no writes.
5. Test duplicates `[3, 1, 2, 3]`.
6. Implement a version counting cycles.
7. Compare write count with Selection Sort.
8. Benchmark on 1000 random elements.
9. Measure wear-leveling benefit for flash.
10. Visualize cycles as arrows in permutation graph.

### Test Cases

| Input     | Output    | Writes | Notes              |
| --------- | --------- | ------ | ------------------ |
| [3,1,2,4] | [1,2,3,4] | 3      | 3 cycles           |
| [1,2,3]   | [1,2,3]   | 0      | Already sorted     |
| [4,3,2,1] | [1,2,3,4] | 4      | Max writes         |
| [3,1,2,3] | [1,2,3,3] | 3      | Handles duplicates |

### Complexity

| Aspect   | Value |
| -------- | ----- |
| Time     | O(n²) |
| Writes   | ≤ n   |
| Space    | O(1)  |
| Stable   | No    |
| Adaptive | No    |

Cycle Sort is the minimalist's sorter, every write is intentional, every move meaningful. It may not be fast, but it's *precisely efficient*.


### 142 Comb Sort

Comb Sort is an improved version of Bubble Sort that eliminates small elements (often called "turtles") faster by comparing elements far apart first, using a shrinking gap strategy.

It starts with a large gap (e.g. array length) and reduces it each pass until it reaches 1, where it behaves like a regular Bubble Sort. The result is fewer comparisons and faster convergence.

### What Problem Are We Solving?

Bubble Sort is simple but slow, mainly because:

- It swaps only adjacent elements.
- Small elements crawl slowly to the front.

Comb Sort fixes this by using a gap to leap over elements, allowing "turtles" to move quickly forward.

It's like sorting with a comb, wide teeth first (large gap), then finer ones (small gap).

#### Example

Sort `[8, 4, 1, 3, 7]`

| Step | Gap         | Pass Result     | Notes         |
| ---- | ----------- | --------------- | ------------- |
| 1    | 5 / 1.3 ≈ 3 | [3, 4, 1, 8, 7] | Compare 8↔3   |
| 2    | 3 / 1.3 ≈ 2 | [1, 4, 3, 8, 7] | Compare 3↔1   |
| 3    | 2 / 1.3 ≈ 1 | [1, 3, 4, 7, 8] | Bubble finish |
| Done |,           | [1, 3, 4, 7, 8] | Sorted        |

Turtles (1, 3) jump forward earlier, speeding up convergence.

### How Does It Work (Plain Language)?

Think of it like shrinking a jump rope, at first, you make big jumps to cover ground fast, then smaller ones to fine-tune.

You start with a gap, compare and swap elements that far apart, shrink the gap each round, and stop when the gap reaches 1 *and no swaps happen*.

#### Step-by-Step Process

| Step | Description                             |
| ---- | --------------------------------------- |
| 1    | Initialize `gap = n` and `shrink = 1.3` |
| 2    | Repeat until `gap == 1` and no swaps    |
| 3    | Divide `gap` by shrink factor each pass |
| 4    | Compare elements at distance `gap`      |
| 5    | Swap if out of order                    |
| 6    | Continue until sorted                   |

### Tiny Code (Easy Versions)

#### Python

```python
def comb_sort(arr):
    n = len(arr)
    gap = n
    shrink = 1.3
    swapped = True

    while gap > 1 or swapped:
        gap = int(gap / shrink)
        if gap < 1:
            gap = 1
        swapped = False
        for i in range(n - gap):
            if arr[i] > arr[i + gap]:
                arr[i], arr[i + gap] = arr[i + gap], arr[i]
                swapped = True
    return arr

arr = [8, 4, 1, 3, 7]
print(comb_sort(arr))
```

Output:

```
$$1, 3, 4, 7, 8]
```

#### C

```c
#include <stdio.h>

void comb_sort(int arr[], int n) {
    int gap = n;
    const float shrink = 1.3;
    int swapped = 1;

    while (gap > 1 || swapped) {
        gap = (int)(gap / shrink);
        if (gap < 1) gap = 1;
        swapped = 0;
        for (int i = 0; i + gap < n; i++) {
            if (arr[i] > arr[i + gap]) {
                int tmp = arr[i];
                arr[i] = arr[i + gap];
                arr[i + gap] = tmp;
                swapped = 1;
            }
        }
    }
}

int main(void) {
    int arr[] = {8, 4, 1, 3, 7};
    int n = sizeof(arr) / sizeof(arr[0]);
    comb_sort(arr, n);
    for (int i = 0; i < n; i++) printf("%d ", arr[i]);
    printf("\n");
}
```

Output:

```
1 3 4 7 8
```

### Why It Matters

- Faster than Bubble Sort
- Simple implementation
- In-place and adaptive
- Efficient for small datasets or nearly sorted arrays

A stepping stone toward more efficient algorithms like Shell Sort.

### A Gentle Proof (Why It Works)

The shrink factor ensures gap reduction converges to 1 in $O(\log n)$ steps.  
Each pass fixes distant inversions early, reducing total swaps.

Total cost is dominated by local passes when gap = 1 (Bubble Sort).  
Hence, average complexity ≈ $O(n \log n)$ for random data, $O(n^2)$ worst case.

| Phase            | Description          | Cost              |
| ---------------- | -------------------- | ----------------- |
| Large gap passes | Move turtles forward | $O(n \log n)$     |
| Small gap passes | Final fine-tuning    | $O(n^2)$ worst    |


### Try It Yourself

1. Sort `[8, 4, 1, 3, 7]` step by step.
2. Try `[1, 2, 3, 4, 5]`, minimal passes.
3. Try `[5, 4, 3, 2, 1]`, observe gap shrinking.
4. Change shrink factor (1.5, 1.2).
5. Measure swaps per iteration.
6. Compare with Bubble Sort.
7. Visualize movement of smallest element.
8. Benchmark large random array.
9. Track gap evolution over time.
10. Implement early-stop optimization.

### Test Cases

| Input       | Output      | Notes          |
| ----------- | ----------- | -------------- |
| [8,4,1,3,7] | [1,3,4,7,8] | Basic test     |
| [1,2,3,4,5] | [1,2,3,4,5] | Already sorted |
| [5,4,3,2,1] | [1,2,3,4,5] | Reverse order  |
| [4,1,3,2]   | [1,2,3,4]   | Short array    |

### Complexity

| Aspect         | Value      |
| -------------- | ---------- |
| Time (Best)    | O(n log n) |
| Time (Average) | O(n log n) |
| Time (Worst)   | O(n²)      |
| Space          | O(1)       |
| Stable         | No         |
| Adaptive       | Yes        |

Comb Sort sweeps through data like a comb through tangled hair, wide strokes first, fine ones later, until everything's smooth and ordered.

### 143 Gnome Sort

Gnome Sort is a simple comparison-based sorting algorithm that works like a garden gnome arranging flower pots, it looks at two adjacent elements and swaps them if they're out of order, then steps backward to check the previous pair again.

It's conceptually similar to Insertion Sort, but implemented with a single loop and no nested structure, making it elegant and intuitive for learners.

### What Problem Are We Solving?

Insertion Sort requires nested loops or recursion, which can be tricky to visualize.
Gnome Sort offers the same logic using a simple forward–backward walk:

- Move forward if elements are ordered.
- Move back and swap if they're not.

This creates a human-like sorting routine, step forward, fix, step back, repeat.

#### Example

Sort `[5, 3, 4, 2]`

| Step | Index | Action                  | Result       |
| ---- | ----- | ----------------------- | ------------ |
| 1    | 1     | 5 > 3 → swap            | [3, 5, 4, 2] |
| 2    | 0     | At start → move forward | [3, 5, 4, 2] |
| 3    | 2     | 5 > 4 → swap            | [3, 4, 5, 2] |
| 4    | 1     | 3 < 4 → forward         | [3, 4, 5, 2] |
| 5    | 3     | 5 > 2 → swap            | [3, 4, 2, 5] |
| 6    | 2     | 4 > 2 → swap            | [3, 2, 4, 5] |
| 7    | 1     | 3 > 2 → swap            | [2, 3, 4, 5] |
| 8    | 0     | Done                    | [2, 3, 4, 5] |

Sorted with local fixes, no explicit nested loops.

### How Does It Work (Plain Language)?

Imagine a gnome walking down a row of flower pots:

- If the two pots are in order, step forward.
- If not, swap them and step back.
- If you reach the start, just move forward again.

Keep walking until you reach the end, the garden (array) is sorted.

#### Step-by-Step Process

| Step | Description                                     |
| ---- | ----------------------------------------------- |
| 1    | Start at index 1                                |
| 2    | Compare arr[i] with arr[i-1]                    |
| 3    | If arr[i] >= arr[i-1], move forward (i++)       |
| 4    | Else, swap arr[i] and arr[i-1], move back (i--) |
| 5    | If i == 0, move to i = 1                        |
| 6    | Repeat until i reaches n                        |

### Tiny Code (Easy Versions)

#### Python

```python
def gnome_sort(arr):
    i = 1
    n = len(arr)
    while i < n:
        if i == 0 or arr[i] >= arr[i - 1]:
            i += 1
        else:
            arr[i], arr[i - 1] = arr[i - 1], arr[i]
            i -= 1
    return arr

arr = [5, 3, 4, 2]
print(gnome_sort(arr))
```

Output:

```
$$2, 3, 4, 5]
```

#### C

```c
#include <stdio.h>

void gnome_sort(int arr[], int n) {
    int i = 1;
    while (i < n) {
        if (i == 0 || arr[i] >= arr[i - 1]) {
            i++;
        } else {
            int tmp = arr[i];
            arr[i] = arr[i - 1];
            arr[i - 1] = tmp;
            i--;
        }
    }
}

int main(void) {
    int arr[] = {5, 3, 4, 2};
    int n = sizeof(arr) / sizeof(arr[0]);
    gnome_sort(arr, n);
    for (int i = 0; i < n; i++) printf("%d ", arr[i]);
    printf("\n");
}
```

Output:

```
2 3 4 5
```

### Why It Matters

- Simple mental model, easy to understand.
- No nested loops, clean control flow.
- In-place, no extra space.
- Demonstrates local correction in sorting.

It's slower than advanced algorithms but ideal for educational purposes.

### A Gentle Proof (Why It Works)

Each swap moves an element closer to its correct position.
Whenever a swap happens, the gnome steps back to ensure local order.

Since every inversion is eventually corrected, the algorithm terminates with a sorted array.

Number of swaps proportional to number of inversions → $O(n^2)$.

| Case    | Behavior       | Complexity |
| ------- | -------------- | ---------- |
| Sorted  | Linear scan    | O(n)       |
| Random  | Frequent swaps | O(n²)      |
| Reverse | Max swaps      | O(n²)      |

### Try It Yourself

1. Sort `[5,3,4,2]` manually step by step.
2. Try `[1,2,3,4]`, minimal steps.
3. Try `[4,3,2,1]`, worst case.
4. Count number of swaps.
5. Compare with Insertion Sort.
6. Track index changes after each swap.
7. Implement visual animation (pointer walk).
8. Try duplicates `[2,1,2,1]`.
9. Measure time for n = 1000.
10. Add early-exit optimization.

### Test Cases

| Input     | Output    | Notes          |
| --------- | --------- | -------------- |
| [5,3,4,2] | [2,3,4,5] | Basic          |
| [1,2,3,4] | [1,2,3,4] | Already sorted |
| [4,3,2,1] | [1,2,3,4] | Reverse        |
| [2,1,2,1] | [1,1,2,2] | Duplicates     |

### Complexity

| Aspect         | Value |
| -------------- | ----- |
| Time (Best)    | O(n)  |
| Time (Average) | O(n²) |
| Time (Worst)   | O(n²) |
| Space          | O(1)  |
| Stable         | Yes   |
| Adaptive       | Yes   |

Gnome Sort is a friendly, step-by-step sorter, it doesn't rush, just tidies things one pot at a time until the whole row is in order.

### 144 Cocktail Sort

Cocktail Sort (also known as Bidirectional Bubble Sort or Shaker Sort) is a simple variation of Bubble Sort that sorts the list in both directions alternately, forward then backward, during each pass.

This bidirectional movement helps small elements ("turtles") bubble up faster from the end, fixing one of Bubble Sort's main weaknesses.

### What Problem Are We Solving?

Bubble Sort only moves elements in one direction, large ones float to the end, but small ones crawl slowly to the start.

Cocktail Sort solves this by shaking the list:

- Forward pass: moves large items right
- Backward pass: moves small items left

This makes it more efficient on nearly sorted arrays or when both ends need cleaning.

#### Example

Sort `[4, 3, 1, 2]`

| Step | Direction | Action                       | Result    |
| ---- | --------- | ---------------------------- | --------- |
| 1    | Forward   | Compare & swap 4↔3, 3↔1, 4↔2 | [3,1,2,4] |
| 2    | Backward  | Compare & swap 2↔1, 3↔1      | [1,3,2,4] |
| 3    | Forward   | Compare & swap 3↔2           | [1,2,3,4] |
| Done |,         | Sorted                       | ✅         |

Fewer passes than Bubble Sort.

### How Does It Work (Plain Language)?

Think of a bartender shaking a cocktail shaker back and forth, each shake moves ingredients (elements) closer to the right place from both sides.

You traverse the array:

- Left to right: push largest elements to the end
- Right to left: push smallest elements to the start

Stop when no swaps occur, the array is sorted.

#### Step-by-Step Process

| Step                                                              | Description                 |
| ----------------------------------------------------------------- | --------------------------- |
| 1                                                                 | Initialize `swapped = True` |
| 2                                                                 | While swapped:              |
|   a. Set swapped = False                                          |                             |
|   b. Forward pass (i = start → end): swap if arr[i] > arr[i+1]    |                             |
|   c. If swapped == False: break (sorted)                          |                             |
|   d. Backward pass (i = end-1 → start): swap if arr[i] > arr[i+1] |                             |
| 3                                                                 | Repeat until sorted         |

### Tiny Code (Easy Versions)

#### Python

```python
def cocktail_sort(arr):
    n = len(arr)
    swapped = True
    start = 0
    end = n - 1

    while swapped:
        swapped = False
        # Forward pass
        for i in range(start, end):
            if arr[i] > arr[i + 1]:
                arr[i], arr[i + 1] = arr[i + 1], arr[i]
                swapped = True
        if not swapped:
            break
        swapped = False
        end -= 1
        # Backward pass
        for i in range(end - 1, start - 1, -1):
            if arr[i] > arr[i + 1]:
                arr[i], arr[i + 1] = arr[i + 1], arr[i]
                swapped = True
        start += 1
    return arr

arr = [4, 3, 1, 2]
print(cocktail_sort(arr))
```

Output:

```
$$1, 2, 3, 4]
```

#### C

```c
#include <stdio.h>

void cocktail_sort(int arr[], int n) {
    int start = 0, end = n - 1, swapped = 1;
    while (swapped) {
        swapped = 0;
        // Forward pass
        for (int i = start; i < end; i++) {
            if (arr[i] > arr[i + 1]) {
                int tmp = arr[i];
                arr[i] = arr[i + 1];
                arr[i + 1] = tmp;
                swapped = 1;
            }
        }
        if (!swapped) break;
        swapped = 0;
        end--;
        // Backward pass
        for (int i = end - 1; i >= start; i--) {
            if (arr[i] > arr[i + 1]) {
                int tmp = arr[i];
                arr[i] = arr[i + 1];
                arr[i + 1] = tmp;
                swapped = 1;
            }
        }
        start++;
    }
}

int main(void) {
    int arr[] = {4, 3, 1, 2};
    int n = sizeof(arr) / sizeof(arr[0]);
    cocktail_sort(arr, n);
    for (int i = 0; i < n; i++) printf("%d ", arr[i]);
    printf("\n");
}
```

Output:

```
1 2 3 4
```

### Why It Matters

- Bidirectional improvement on Bubble Sort
- In-place and simple
- Performs well on nearly sorted data
- Adaptive, stops early when sorted

Great educational bridge to understanding bidirectional scans and adaptive sorting.

### A Gentle Proof (Why It Works)

Each forward pass pushes the largest element to the right.
Each backward pass pushes the smallest element to the left.

After each full cycle, the sorted region expands from both ends.
The algorithm stops when no swaps occur (sorted).

Total operations depend on number of inversions:
$$
O(n^2) \text{ worst}, \quad O(n) \text{ best (sorted input)}
$$

| Case    | Behavior               | Complexity |
| ------- | ---------------------- | ---------- |
| Sorted  | One bidirectional scan | O(n)       |
| Random  | Many swaps             | O(n²)      |
| Reverse | Max passes             | O(n²)      |

### Try It Yourself

1. Sort `[4,3,1,2]` manually step by step.
2. Try `[1,2,3,4]`, should stop early.
3. Try `[5,4,3,2,1]`, observe shaking effect.
4. Count swaps each pass.
5. Compare passes with Bubble Sort.
6. Visualize forward/backward movement.
7. Add "swap counter" variable.
8. Test duplicates `[3,1,3,2,1]`.
9. Measure performance on nearly sorted data.
10. Modify shrink window size.

### Test Cases

| Input       | Output      | Notes          |
| ----------- | ----------- | -------------- |
| [4,3,1,2]   | [1,2,3,4]   | Basic          |
| [1,2,3,4]   | [1,2,3,4]   | Already sorted |
| [5,4,3,2,1] | [1,2,3,4,5] | Reverse        |
| [3,1,3,2,1] | [1,1,2,3,3] | Duplicates     |

### Complexity

| Aspect         | Value |
| -------------- | ----- |
| Time (Best)    | O(n)  |
| Time (Average) | O(n²) |
| Time (Worst)   | O(n²) |
| Space          | O(1)  |
| Stable         | Yes   |
| Adaptive       | Yes   |

Cocktail Sort is the sorter that doesn't just bubble, it *shakes* the data into order, making sure both ends get attention every round.

### 145 Pancake Sort

Pancake Sort is a whimsical and educational sorting algorithm inspired by flipping pancakes on a plate, at each step, you bring the largest unsorted pancake to the top, then flip the stack to move it to its correct position.

It's not practical for large datasets, but it's a brilliant way to illustrate prefix reversals, maximum selection, and in-place transformations.

### What Problem Are We Solving?

Standard selection sort swaps elements pairwise.
Pancake Sort instead uses prefix reversals, flipping a whole section of the array, to move the largest item into place.

It's a thought experiment in restricted operations:
What if the only thing you could do is *flip*?

#### Example

Sort `[3, 6, 1, 5, 2, 4]`

| Step | Action                               | Result             |
| ---- | ------------------------------------ | ------------------ |
| 1    | Find max (6) → index 1, flip first 2 | [6, 3, 1, 5, 2, 4] |
| 2    | Flip first 6                         | [4, 2, 5, 1, 3, 6] |
| 3    | Find max (5) → index 2, flip first 3 | [5, 2, 4, 1, 3, 6] |
| 4    | Flip first 5                         | [3, 1, 4, 2, 5, 6] |
| 5    | Repeat for smaller prefixes          | [1, 2, 3, 4, 5, 6] |

Sorted using flips only!

### How Does It Work (Plain Language)?

Imagine flipping a stack of pancakes:

1. Find the biggest one not yet placed.
2. Flip the stack to bring it to the top.
3. Flip again to move it into its final position.

Repeat, each time excluding the sorted top portion.

#### Step-by-Step Process

| Step | Description                                     |
| ---- | ----------------------------------------------- |
| 1    | Find index of largest unsorted element          |
| 2    | Flip subarray from start to that index          |
| 3    | Flip entire unsorted subarray to move it to end |
| 4    | Reduce unsorted portion by one                  |
| 5    | Repeat until sorted                             |

### Tiny Code (Easy Versions)

#### Python

```python
def flip(arr, k):
    arr[:k+1] = reversed(arr[:k+1])

def pancake_sort(arr):
    n = len(arr)
    for curr_size in range(n, 1, -1):
        max_idx = arr.index(max(arr[:curr_size]))
        if max_idx != curr_size - 1:
            flip(arr, max_idx)
            flip(arr, curr_size - 1)
    return arr

arr = [3, 6, 1, 5, 2, 4]
print(pancake_sort(arr))
```

Output:

```
$$1, 2, 3, 4, 5, 6]
```

#### C

```c
#include <stdio.h>

void flip(int arr[], int k) {
    int start = 0;
    while (start < k) {
        int temp = arr[start];
        arr[start] = arr[k];
        arr[k] = temp;
        start++;
        k--;
    }
}

int find_max(int arr[], int n) {
    int max_idx = 0;
    for (int i = 1; i < n; i++)
        if (arr[i] > arr[max_idx])
            max_idx = i;
    return max_idx;
}

void pancake_sort(int arr[], int n) {
    for (int size = n; size > 1; size--) {
        int max_idx = find_max(arr, size);
        if (max_idx != size - 1) {
            flip(arr, max_idx);
            flip(arr, size - 1);
        }
    }
}

int main(void) {
    int arr[] = {3, 6, 1, 5, 2, 4};
    int n = sizeof(arr) / sizeof(arr[0]);
    pancake_sort(arr, n);
    for (int i = 0; i < n; i++) printf("%d ", arr[i]);
    printf("\n");
}
```

Output:

```
1 2 3 4 5 6
```

### Why It Matters

- Fun demonstration of prefix operations
- In-place and simple
- Shows how restricted operations can still sort
- Theoretical interest, base for pancake networks
- Used in bioinformatics (genome rearrangements)

### A Gentle Proof (Why It Works)

Each iteration places the largest remaining element at its final index.
Two flips per iteration (worst case).
At most ( 2(n - 1) ) flips total.

Correctness follows from:

- Flipping is a reversal, which preserves order except within flipped segment.
- Each largest element is locked at the end after placement.

$$
T(n) = O(n^2)
$$
because each `max()` and `flip()` operation is O(n).

### Try It Yourself

1. Sort `[3,6,1,5,2,4]` manually.
2. Trace each flip visually.
3. Try `[1,2,3,4]`, no flips needed.
4. Reverse `[4,3,2,1]`, observe maximum flips.
5. Count flips per iteration.
6. Implement flip visualization.
7. Replace `max()` with manual search.
8. Print intermediate arrays.
9. Analyze flip count for random input.
10. Challenge: implement recursive version.

### Test Cases

| Input         | Output        | Flips | Notes          |
| ------------- | ------------- | ----- | -------------- |
| [3,6,1,5,2,4] | [1,2,3,4,5,6] | 8     | Classic        |
| [1,2,3,4]     | [1,2,3,4]     | 0     | Already sorted |
| [4,3,2,1]     | [1,2,3,4]     | 6     | Worst case     |
| [2,1,3]       | [1,2,3]       | 3     | Small array    |

### Complexity

| Aspect         | Value |
| -------------- | ----- |
| Time (Worst)   | O(n²) |
| Time (Average) | O(n²) |
| Time (Best)    | O(n)  |
| Space          | O(1)  |
| Stable         | No    |
| Adaptive       | No    |

Pancake Sort flips its way to victory, a charming example of ingenuity under constraint.
You don't need fancy tools, just a good spatula and some patience.

### 146 Bitonic Sort

Bitonic Sort is a parallel sorting algorithm designed for sorting networks. It works by constructing and merging bitonic sequences, sequences that first increase, then decrease (or vice versa).

It's especially powerful on hardware, GPUs, and parallel processors, where multiple comparisons can happen at once.

### What Problem Are We Solving?

Most standard algorithms (QuickSort, MergeSort) are data-dependent, their flow changes depending on comparisons. That's a problem for hardware or parallel systems.

Bitonic Sort fixes this by having a fixed comparison pattern, perfect for parallel execution.

It answers the question:

> "How do we sort in parallel using predictable, fixed circuits?"

#### Example

Sort `[3, 7, 4, 8, 6, 2, 1, 5]`

| Step | Action                    | Result             |
| ---- | ------------------------- | ------------------ |
| 1    | Build bitonic sequence ↑↓ | [3,7,4,8, 6,2,1,5] |
| 2    | Compare & swap pairs      | [3,2,1,5, 6,7,4,8] |
| 3    | Merge halves recursively  | [1,2,3,4, 5,6,7,8] |

Each phase doubles sorted subsequences until the full array is sorted.

### How Does It Work (Plain Language)?

Imagine you're sorting cards but you can only compare pairs at fixed positions, no adaptive decisions.

You first build a bitonic sequence (up then down), then merge it into a sorted list using bitonic merge steps, repeatedly compare-and-swap mirrored elements.

#### Step-by-Step Process

| Step | Description                                           |
| ---- | ----------------------------------------------------- |
| 1    | Split array into halves                               |
| 2    | Sort first half ascending, second half descending     |
| 3    | Combine them into a bitonic sequence              |
| 4    | Run bitonic merge: recursively compare-swap pairs |
| 5    | Repeat until sequence size = 1                        |

Works best when n = 2^k (power of two).

### Tiny Code (Easy Versions)

#### Python

```python
def bitonic_merge(arr, low, cnt, direction):
    if cnt > 1:
        k = cnt // 2
        for i in range(low, low + k):
            if (direction and arr[i] > arr[i + k]) or (not direction and arr[i] < arr[i + k]):
                arr[i], arr[i + k] = arr[i + k], arr[i]
        bitonic_merge(arr, low, k, direction)
        bitonic_merge(arr, low + k, k, direction)

def bitonic_sort(arr, low=0, cnt=None, direction=True):
    if cnt is None:
        cnt = len(arr)
    if cnt > 1:
        k = cnt // 2
        bitonic_sort(arr, low, k, True)
        bitonic_sort(arr, low + k, k, False)
        bitonic_merge(arr, low, cnt, direction)

arr = [3, 7, 4, 8, 6, 2, 1, 5]
bitonic_sort(arr)
print(arr)
```

Output:

```
$$1, 2, 3, 4, 5, 6, 7, 8]
```

#### C

```c
#include <stdio.h>

void swap(int *a, int *b) {
    int t = *a; *a = *b; *b = t;
}

void bitonic_merge(int arr[], int low, int cnt, int dir) {
    if (cnt > 1) {
        int k = cnt / 2;
        for (int i = low; i < low + k; i++) {
            if ((dir && arr[i] > arr[i + k]) || (!dir && arr[i] < arr[i + k]))
                swap(&arr[i], &arr[i + k]);
        }
        bitonic_merge(arr, low, k, dir);
        bitonic_merge(arr, low + k, k, dir);
    }
}

void bitonic_sort(int arr[], int low, int cnt, int dir) {
    if (cnt > 1) {
        int k = cnt / 2;
        bitonic_sort(arr, low, k, 1);
        bitonic_sort(arr, low + k, k, 0);
        bitonic_merge(arr, low, cnt, dir);
    }
}

int main(void) {
    int arr[] = {3, 7, 4, 8, 6, 2, 1, 5};
    int n = sizeof(arr) / sizeof(arr[0]);
    bitonic_sort(arr, 0, n, 1);
    for (int i = 0; i < n; i++) printf("%d ", arr[i]);
    printf("\n");
}
```

Output:

```
1 2 3 4 5 6 7 8
```

### Why It Matters

- Parallel-friendly (sorting networks)
- Deterministic structure (no branches)
- Perfect for hardware, GPUs, SIMD
- Good educational model for divide and conquer + merging

It's not about runtime on CPUs, it's about parallel depth.

### A Gentle Proof (Why It Works)

**Bitonic sequence:**  
A sequence that increases then decreases is bitonic.

**Merging rule:**  
Compare each element with its mirror; recursively merge halves.

At each merge stage, the array becomes more sorted.  
Recursion depth = $\log n$, each level does $O(n)$ work → $O(n \log^2 n)$.

| Step  | Work   | Levels  | Total          |
| ------ | ------- | -------- | -------------- |
| Merge  | $O(n)$  | $\log n$ | $O(n \log^2 n)$ |


### Try It Yourself

1. Sort `[3,7,4,8,6,2,1,5]` manually.
2. Identify bitonic sequences at each stage.
3. Try with 4 elements `[4,1,3,2]`.
4. Change direction flags (ascending/descending).
5. Draw comparison network graph.
6. Implement iterative version.
7. Run on power-of-two sizes.
8. Measure parallel steps vs QuickSort.
9. Experiment with GPU (Numba/CUDA).
10. Visualize recursive structure.

### Test Cases

| Input             | Output            | Notes      |
| ----------------- | ----------------- | ---------- |
| [3,7,4,8,6,2,1,5] | [1,2,3,4,5,6,7,8] | Standard   |
| [4,1,3,2]         | [1,2,3,4]         | Small case |
| [5,4,3,2,1,0,9,8] | [0,1,2,3,4,5,8,9] | Reverse    |
| [8,4,2,1,3,6,5,7] | [1,2,3,4,5,6,7,8] | Random     |

### Complexity

| Aspect         | Value       |
| -------------- | ----------- |
| Time           | O(n log² n) |
| Space          | O(1)        |
| Stable         | No          |
| Adaptive       | No          |
| Parallel Depth | O(log² n)   |

Bitonic Sort shines where parallelism rules, in GPUs, circuits, and sorting networks.
Every comparison is planned, every move synchronized, a symphony of order in fixed rhythm.

### 147 Odd-Even Merge Sort

Odd-Even Merge Sort is a parallel sorting algorithm and a sorting network that merges two sorted sequences using a fixed pattern of comparisons between odd and even indexed elements.

It was introduced by Ken Batcher, and like Bitonic Sort, it's designed for parallel hardware or SIMD processors, where predictable comparison patterns matter more than data-dependent branching.

### What Problem Are We Solving?

Traditional merge algorithms rely on conditional branching, they decide at runtime which element to pick next.
This is problematic in parallel or hardware implementations, where you need fixed, predictable sequences of comparisons.

Odd-Even Merge Sort solves this by using a static comparison network that merges sorted halves without branching.

It's perfect when:

- You need deterministic behavior
- You're building parallel circuits or GPU kernels

#### Example

Merge two sorted halves: `[1, 4, 7, 8]` and `[2, 3, 5, 6]`

| Step | Action                           | Result            |
| ---- | -------------------------------- | ----------------- |
| 1    | Merge odds `[1,7]` with `[2,5]`  | [1,2,5,7]         |
| 2    | Merge evens `[4,8]` with `[3,6]` | [3,4,6,8]         |
| 3    | Combine and compare adjacent     | [1,2,3,4,5,6,7,8] |

Fixed pattern, no branching, merges completed in parallel.

### How Does It Work (Plain Language)?

Imagine two zipper chains, one odd, one even.
You weave them together in a fixed, interlocking pattern, comparing and swapping along the way.
There's no guessing, every element knows which neighbor to check.

#### Step-by-Step Process

| Step                                                 | Description                              |
| ---------------------------------------------------- | ---------------------------------------- |
| 1                                                    | Split array into left and right halves   |
| 2                                                    | Recursively sort each half               |
| 3                                                    | Use odd-even merge to combine halves |
| 4                                                    | Odd-even merge:                          |
|   a. Recursively merge odd and even indexed elements |                                          |
|   b. Compare and swap adjacent pairs                 |                                          |
| 5                                                    | Continue until array sorted              |

Works best when ( n = 2^k ).

### Tiny Code (Easy Versions)

#### Python

```python
def odd_even_merge(arr, lo, n, direction):
    if n > 1:
        m = n // 2
        odd_even_merge(arr, lo, m, direction)
        odd_even_merge(arr, lo + m, m, direction)
        for i in range(lo + m, lo + n - m):
            if (arr[i] > arr[i + m]) == direction:
                arr[i], arr[i + m] = arr[i + m], arr[i]

def odd_even_merge_sort(arr, lo=0, n=None, direction=True):
    if n is None:
        n = len(arr)
    if n > 1:
        m = n // 2
        odd_even_merge_sort(arr, lo, m, direction)
        odd_even_merge_sort(arr, lo + m, m, direction)
        odd_even_merge(arr, lo, n, direction)

arr = [8, 3, 2, 7, 4, 6, 5, 1]
odd_even_merge_sort(arr)
print(arr)
```

Output:

```
$$1, 2, 3, 4, 5, 6, 7, 8]
```

#### C

```c
#include <stdio.h>

void swap(int *a, int *b) {
    int t = *a; *a = *b; *b = t;
}

void odd_even_merge(int arr[], int lo, int n, int dir) {
    if (n > 1) {
        int m = n / 2;
        odd_even_merge(arr, lo, m, dir);
        odd_even_merge(arr, lo + m, m, dir);
        for (int i = lo + m; i < lo + n - m; i++) {
            if ((arr[i] > arr[i + m]) == dir)
                swap(&arr[i], &arr[i + m]);
        }
    }
}

void odd_even_merge_sort(int arr[], int lo, int n, int dir) {
    if (n > 1) {
        int m = n / 2;
        odd_even_merge_sort(arr, lo, m, dir);
        odd_even_merge_sort(arr, lo + m, m, dir);
        odd_even_merge(arr, lo, n, dir);
    }
}

int main(void) {
    int arr[] = {8, 3, 2, 7, 4, 6, 5, 1};
    int n = sizeof(arr)/sizeof(arr[0]);
    odd_even_merge_sort(arr, 0, n, 1);
    for (int i = 0; i < n; i++) printf("%d ", arr[i]);
    printf("\n");
}
```

Output:

```
1 2 3 4 5 6 7 8
```

### Why It Matters

- Fixed sequence, perfect for parallelism
- No data-dependent branching
- Used in hardware sorting networks
- Theoretical foundation for parallel sorting

When you need determinism and concurrency, this algorithm shines.

### A Gentle Proof (Why It Works)

Each odd-even merge merges two sorted sequences using fixed compare-swap operations.
At each stage:

- Odd indices are merged separately
- Even indices merged separately
- Adjacent elements compared to restore global order

Each level performs O(n) work, depth = O(log² n) → total complexity:
$$
T(n) = O(n \log^2 n)
$$

### Try It Yourself

1. Sort `[8,3,2,7,4,6,5,1]` step by step.
2. Trace odd-index and even-index merges.
3. Draw merge network diagram.
4. Try smaller `[4,3,2,1]` for clarity.
5. Run on power-of-two lengths.
6. Measure comparisons.
7. Compare with Bitonic Sort.
8. Implement iterative version.
9. Visualize parallel depth.
10. Experiment with ascending/descending flags.

### Test Cases

| Input             | Output            | Notes        |
| ----------------- | ----------------- | ------------ |
| [8,3,2,7,4,6,5,1] | [1,2,3,4,5,6,7,8] | Classic      |
| [4,3,2,1]         | [1,2,3,4]         | Small        |
| [9,7,5,3,1,2,4,6] | [1,2,3,4,5,6,7,9] | Mixed        |
| [5,4,3,2]         | [2,3,4,5]         | Reverse half |

### Complexity

| Aspect         | Value       |
| -------------- | ----------- |
| Time           | O(n log² n) |
| Space          | O(1)        |
| Stable         | No          |
| Adaptive       | No          |
| Parallel Depth | O(log² n)   |

Odd-Even Merge Sort weaves order from two halves like clockwork, steady, parallel, and predictable.
Every comparison is planned, every merge synchronized, it's sorting as architecture.

### 148 Sleep Sort

Sleep Sort is one of the most playful and unconventional algorithms ever invented, it sorts numbers by leveraging time delays. Each element is "slept" for a duration proportional to its value, and when the sleep ends, it prints the number.

In effect, time itself becomes the sorting mechanism.

### What Problem Are We Solving?

While not practical, Sleep Sort offers a fun demonstration of parallelism and asynchronous timing, showing that even sorting can be expressed through temporal order rather than comparisons.

It's often used as a thought experiment to teach concurrency, timing, and creative thinking about problem-solving.

#### Example

Sort `[3, 1, 4, 2]`

| Step | Value | Sleep (seconds) | Print order |
| ---- | ----- | --------------- | ----------- |
| 1    | 1     | 1s              | 1           |
| 2    | 2     | 2s              | 2           |
| 3    | 3     | 3s              | 3           |
| 4    | 4     | 4s              | 4           |

Output (over time):
`1 2 3 4`

Sorted by time of completion!

### How Does It Work (Plain Language)?

Each number is given a timer equal to its value.
All timers start simultaneously, and when a timer finishes, that number is output.
Small numbers "wake up" first, so they're printed earlier, creating a sorted sequence.

It's like a race where each runner's speed is inversely proportional to its size, smaller ones finish first.

#### Step-by-Step Process

| Step | Description                                        |
| ---- | -------------------------------------------------- |
| 1    | For each element `x`, create a thread or coroutine |
| 2    | Each thread sleeps for `x` units of time           |
| 3    | When sleep completes, print `x`                    |
| 4    | Numbers appear in sorted order                     |
| 5    | Optionally collect outputs into a list             |

### Tiny Code (Easy Versions)

#### Python (Using Threads)

```python
import threading
import time

def sleeper(x):
    time.sleep(x * 0.1)  # scale factor for speed
    print(x, end=' ')

def sleep_sort(arr):
    threads = []
    for x in arr:
        t = threading.Thread(target=sleeper, args=(x,))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()

arr = [3, 1, 4, 2]
sleep_sort(arr)
```

Output (timed):

```
1 2 3 4
```

#### C (Using Threads and Sleep)

```c
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>

void* sleeper(void* arg) {
    int x = *(int*)arg;
    usleep(x * 100000); // scaled down
    printf("%d ", x);
    return NULL;
}

void sleep_sort(int arr[], int n) {
    pthread_t threads[n];
    for (int i = 0; i < n; i++)
        pthread_create(&threads[i], NULL, sleeper, &arr[i]);
    for (int i = 0; i < n; i++)
        pthread_join(threads[i], NULL);
}

int main(void) {
    int arr[] = {3, 1, 4, 2};
    int n = sizeof(arr) / sizeof(arr[0]);
    sleep_sort(arr, n);
    printf("\n");
}
```

Output (timed):

```
1 2 3 4
```

### Why It Matters

- Creative demonstration of parallelism
- Fun teaching tool for concurrency
- Visually intuitive, sorting emerges naturally
- Great reminder: algorithms ≠ just code, they're processes

It's impractical, but delightfully educational.

### A Gentle Proof (Why It Works)

If all threads start simultaneously and sleep proportionally to their values, then:

- Smaller values finish earlier
- No collisions (if distinct integers)
- Output sequence = sorted list

For duplicates, slight offsets may be added to maintain stability.

Limitations:

- Requires positive integers
- Depends on accurate timers
- Sensitive to scheduler latency

### Try It Yourself

1. Sort `[3,1,4,2]`, observe timing.
2. Try `[10,5,1,2]`, slower but clearer pattern.
3. Add duplicates `[2,2,1]`, test ordering.
4. Scale sleep time down (`x * 0.05`).
5. Run on multi-core CPU, observe concurrency.
6. Replace sleep with `await asyncio.sleep(x)` for async version.
7. Collect results in a list instead of print.
8. Use `multiprocessing` instead of threads.
9. Visualize time vs value graph.
10. Try fractional delays for floats.

### Test Cases

| Input     | Output    | Notes              |
| --------- | --------- | ------------------ |
| [3,1,4,2] | [1,2,3,4] | Classic example    |
| [1,2,3,4] | [1,2,3,4] | Already sorted     |
| [4,3,2,1] | [1,2,3,4] | Reversed           |
| [2,2,1]   | [1,2,2]   | Handles duplicates |

### Complexity

| Aspect             | Value                       |
| ------------------ | --------------------------- |
| Time (Theoretical) | O(n) real-time (wall-clock) |
| Time (CPU Work)    | O(n) setup                  |
| Space              | O(n) threads                |
| Stable             | Yes (with offset)           |
| Adaptive           | No                          |

Sleep Sort is sorting reimagined, not by computation, but by patience.
Every number simply waits its turn, no comparisons, no loops, just time.

### 149 Bead Sort

Bead Sort, also known as Gravity Sort, is a natural sorting algorithm inspired by how beads slide under gravity on parallel rods. Imagine an abacus turned on its side: heavier piles settle first, automatically sorting themselves.

It's visual, parallel, and analog in spirit, more of a conceptual model than a practical tool, but brilliant for intuition.

### What Problem Are We Solving?

Sorting algorithms usually rely on comparisons.
Bead Sort instead uses physical simulation, items fall until they settle into order.

This approach helps visualize distribution sorting and natural computation, where sorting happens through physical laws rather than arithmetic operations.

#### Example

Sort `[5, 3, 1, 7, 4]`:

1. Represent each number as a row of beads.
2. Drop beads under gravity.
3. Count beads per column from bottom up.

| Step    | Representation                            | After Gravity   | Output |
| ------- | ----------------------------------------- | --------------- | ------ |
| Initial | 5●●●●●<br>3●●●<br>1●<br>7●●●●●●●<br>4●●●● |,               |,      |
| Gravity | Columns fill from bottom                  | Rows shorten    |,      |
| Result  | 1●<br>3●●●<br>4●●●●<br>5●●●●●<br>7●●●●●●● | [1, 3, 4, 5, 7] |        |

The smallest number rises to top, largest sinks to bottom, sorted.

### How Does It Work (Plain Language)?

Each number is a pile of beads.
Beads "fall" downward until no empty space below.
Since heavier rows push beads downward faster, larger numbers accumulate at the bottom.
When gravity stops, reading row lengths from top to bottom yields sorted order.

It's sorting by simulated gravity, no comparisons at all.

#### Step-by-Step Process

| Step | Description                                                |
| ---- | ---------------------------------------------------------- |
| 1    | Represent each integer by beads on rods (1 bead per unit)  |
| 2    | Let beads fall to the lowest empty position in each column |
| 3    | After settling, count beads per row (top-down)             |
| 4    | These counts form the sorted list                          |

Works only for non-negative integers.

### Tiny Code (Easy Versions)

#### Python

```python
def bead_sort(arr):
    if any(x < 0 for x in arr):
        raise ValueError("Only non-negative integers allowed")
    max_val = max(arr)
    grid = [[1 if i < x else 0 for i in range(max_val)] for x in arr]
    for col in range(max_val):
        beads = sum(row[col] for row in grid)
        for row in range(len(arr)):
            grid[row][col] = 1 if row >= len(arr) - beads else 0
    return [sum(row) for row in grid]

arr = [5, 3, 1, 7, 4]
print(bead_sort(arr))
```

Output:

```
$$1, 3, 4, 5, 7]
```

#### C

```c
#include <stdio.h>
#include <string.h>

void bead_sort(int *a, int n) {
    int max = 0;
    for (int i = 0; i < n; i++) if (a[i] > max) max = a[i];
    unsigned char beads[n][max];
    memset(beads, 0, n * max);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < a[i]; j++)
            beads[i][j] = 1;

    for (int j = 0; j < max; j++) {
        int sum = 0;
        for (int i = 0; i < n; i++) sum += beads[i][j];
        for (int i = 0; i < n; i++)
            beads[i][j] = (i >= n - sum) ? 1 : 0;
    }

    for (int i = 0; i < n; i++) {
        a[i] = 0;
        for (int j = 0; j < max; j++)
            a[i] += beads[i][j];
    }
}

int main(void) {
    int arr[] = {5, 3, 1, 7, 4};
    int n = sizeof(arr) / sizeof(arr[0]);
    bead_sort(arr, n);
    for (int i = 0; i < n; i++) printf("%d ", arr[i]);
    printf("\n");
}
```

Output:

```
1 3 4 5 7
```

### Why It Matters

- Demonstrates non-comparison-based sorting
- Shows physical analogies for computation
- Ideal for visual and educational purposes
- Parallelizable (each column independent)

Though impractical, it inspires biological and physics-inspired algorithm design.

### A Gentle Proof (Why It Works)

Each column acts like a gravity channel:

- Beads fall to fill lowest positions
- Columns represent magnitudes across numbers
- After settling, beads in each row = sorted value

No two beads can occupy the same slot twice, ensuring correctness.
Complexity:
$$
T(n) = O(S)
$$
where $S = \sum a_i$, total bead count.

Efficient only when numbers are small.

### Try It Yourself

1. Sort `[5,3,1,7,4]` by hand using dots.
2. Draw rods and let beads fall.
3. Try `[3,0,2,1]`, zeros stay top.
4. Experiment with duplicates `[2,2,3]`.
5. Use grid visualization in Python.
6. Compare with Counting Sort.
7. Extend for stable ordering.
8. Animate bead falling step by step.
9. Scale with numbers ≤ 10.
10. Reflect: what if gravity was sideways?

### Test Cases

| Input       | Output      | Notes                 |
| ----------- | ----------- | --------------------- |
| [5,3,1,7,4] | [1,3,4,5,7] | Classic example       |
| [3,0,2,1]   | [0,1,2,3]   | Handles zeros         |
| [2,2,3]     | [2,2,3]     | Works with duplicates |
| [1]         | [1]         | Single element        |

### Complexity

| Aspect   | Value                           |
| -------- | ------------------------------- |
| Time     | O(S), where S = sum of elements |
| Space    | O(S)                            |
| Stable   | No                              |
| Adaptive | No                              |

Bead Sort shows how even gravity can sort, numbers become beads, and time, motion, and matter do the work.
It's sorting you can *see*, not just compute.

### 150 Bogo Sort

Bogo Sort (also called Permutation Sort or Stupid Sort) is a deliberately absurd algorithm that repeatedly shuffles the array until it becomes sorted.

It's the poster child of inefficiency, often used in classrooms as a comic counterexample, sorting by pure luck.

### What Problem Are We Solving?

We're not solving a problem so much as demonstrating futility.
Bogo Sort asks, *"What if we just kept trying random orders until we got lucky?"*

It's a great teaching tool for:

- Understanding algorithmic inefficiency
- Appreciating complexity bounds
- Learning to recognize good vs. bad strategies

#### Example

Sort `[3, 1, 2]`

| Attempt | Shuffle | Sorted? |
| ------- | ------- | ------- |
| 1       | [3,1,2] | No      |
| 2       | [1,2,3] | Yes ✅   |

Stop when lucky!
(You could get lucky early... or never.)

### How Does It Work (Plain Language)?

The idea is painfully simple:

1. Check if the array is sorted.
2. If not, shuffle it randomly.
3. Repeat until sorted.

It's sorting by random chance, not logic.
Each attempt has a tiny probability of being sorted, but given infinite time, it *will* finish (eventually).

#### Step-by-Step Process

| Step | Description              |
| ---- | ------------------------ |
| 1    | Check if array is sorted |
| 2    | If sorted, done          |
| 3    | Else, shuffle randomly   |
| 4    | Go back to step 1        |

### Tiny Code (Easy Versions)

#### Python

```python
import random

def is_sorted(arr):
    return all(arr[i] <= arr[i+1] for i in range(len(arr)-1))

def bogo_sort(arr):
    attempts = 0
    while not is_sorted(arr):
        random.shuffle(arr)
        attempts += 1
    print("Sorted in", attempts, "attempts")
    return arr

arr = [3, 1, 2]
print(bogo_sort(arr))
```

Output (random):

```
Sorted in 7 attempts
$$1, 2, 3]
```

#### C

```c
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int is_sorted(int arr[], int n) {
    for (int i = 0; i < n - 1; i++)
        if (arr[i] > arr[i + 1]) return 0;
    return 1;
}

void shuffle(int arr[], int n) {
    for (int i = 0; i < n; i++) {
        int j = rand() % n;
        int temp = arr[i];
        arr[i] = arr[j];
        arr[j] = temp;
    }
}

void bogo_sort(int arr[], int n) {
    int attempts = 0;
    while (!is_sorted(arr, n)) {
        shuffle(arr, n);
        attempts++;
    }
    printf("Sorted in %d attempts\n", attempts);
}

int main(void) {
    srand(time(NULL));
    int arr[] = {3, 1, 2};
    int n = sizeof(arr) / sizeof(arr[0]);
    bogo_sort(arr, n);
    for (int i = 0; i < n; i++) printf("%d ", arr[i]);
    printf("\n");
}
```

Output (random):

```
Sorted in 12 attempts  
1 2 3
```

### Why It Matters

- Humorous cautionary tale, what *not* to do
- Demonstrates expected runtime analysis
- A good way to visualize randomness
- Reinforces need for algorithmic reasoning

It's the algorithmic equivalent of *throwing dice until sorted*, mathematically silly, but conceptually rich.

### A Gentle Proof (Why It Works)

With $n$ elements, there are $n!$ permutations.  
Only one is sorted.

Probability of success = $\frac{1}{n!}$  

Expected attempts:
$$
E(n) = n!
$$

Each check takes $O(n)$, so total expected time:
$$
T(n) = O(n \times n!)
$$

Guaranteed termination (eventually), since the probability of not sorting forever $\to 0$.


### Try It Yourself

1. Run on `[3,1,2]` and count attempts.
2. Try `[1,2,3]`, instant success.
3. Test `[4,3,2,1]`, likely infinite patience required.
4. Replace random.shuffle with deterministic shuffle (see fail).
5. Add a timeout.
6. Visualize shuffles on screen.
7. Measure average attempts over 100 trials.
8. Compare with Bubble Sort.
9. Try "Bogobogosort" (recursive Bogo!).
10. Reflect: what's the expected runtime for `n=5`?

### Test Cases

| Input     | Output    | Notes                     |
| --------- | --------- | ------------------------- |
| [3,1,2]   | [1,2,3]   | Classic                   |
| [1,2,3]   | [1,2,3]   | Already sorted            |
| [2,1]     | [1,2]     | Fast                      |
| [4,3,2,1] | [1,2,3,4] | Possibly never terminates |

### Complexity

| Aspect          | Value                   |
| --------------- | ----------------------- |
| Time (Expected) | O(n × n!)               |
| Time (Best)     | O(n)                    |
| Time (Worst)    | Unbounded               |
| Space           | O(1)                    |
| Stable          | Yes (if shuffle stable) |
| Adaptive        | No                      |

Bogo Sort is chaos pretending to be order, sorting by faith, not logic.
It's the universe's reminder that hope isn't a strategy, not even in algorithms.

## Section 16. Linear and Binary Search 

### 151 Linear Search

Linear Search (also known as Sequential Search) is the simplest and most intuitive searching algorithm. It scans through each element one by one until it finds the target, or reaches the end of the list.

It's easy to understand, easy to implement, and works on both sorted and unsorted data.

### What Problem Are We Solving?

Given a list and a target value, how can we check if the target is present, and if so, at which index?

Linear Search solves this by scanning each element in order until a match is found.

Perfect for:

- Small datasets
- Unsorted arrays
- Early learning of search principles

#### Example

Find `7` in `[3, 5, 7, 2, 9]`:

| Step | Index | Value | Match?           |
| ---- | ----- | ----- | ---------------- |
| 1    | 0     | 3     | No               |
| 2    | 1     | 5     | No               |
| 3    | 2     | 7     | ✅ Yes            |
| 4    | Stop  |,     | Found at index 2 |

### How Does It Work (Plain Language)?

It's like flipping through pages one by one looking for a word.
No skipping, no guessing, just check everything in order.

If the list is `[a₀, a₁, a₂, ..., aₙ₋₁]`:

1. Start at index 0
2. Compare `a[i]` with the target
3. If equal → found
4. Else → move to next
5. Stop when found or end reached

#### Step-by-Step Process

| Step | Description                         |
| ---- | ----------------------------------- |
| 1    | Start from index 0                  |
| 2    | Compare current element with target |
| 3    | If equal, return index              |
| 4    | Else, increment index               |
| 5    | Repeat until end                    |
| 6    | If not found, return -1             |

### Tiny Code (Easy Versions)

#### Python

```python
def linear_search(arr, target):
    for i, val in enumerate(arr):
        if val == target:
            return i
    return -1

arr = [3, 5, 7, 2, 9]
target = 7
idx = linear_search(arr, target)
print("Found at index:", idx)
```

Output:

```
Found at index: 2
```

#### C

```c
#include <stdio.h>

int linear_search(int arr[], int n, int target) {
    for (int i = 0; i < n; i++)
        if (arr[i] == target)
            return i;
    return -1;
}

int main(void) {
    int arr[] = {3, 5, 7, 2, 9};
    int n = sizeof(arr) / sizeof(arr[0]);
    int target = 7;
    int idx = linear_search(arr, n, target);
    if (idx != -1)
        printf("Found at index: %d\n", idx);
    else
        printf("Not found\n");
}
```

Output:

```
Found at index: 2
```

### Why It Matters

- Works on any list, sorted or unsorted
- No preprocessing needed
- Guaranteed to find (if present)
- Great introduction to time complexity
- Foundation for better search algorithms

### A Gentle Proof (Why It Works)

If the element exists, scanning each element ensures it will eventually be found.

For ( n ) elements:

- Best case: found at index 0 → O(1)
- Worst case: not found or last → O(n)
- Average case: half-way → O(n)

Because there's no faster way without structure.

### Try It Yourself

1. Search `7` in `[3,5,7,2,9]`
2. Search `10` (not in list)
3. Try `[1,2,3,4,5]` with `target=1` (best case)
4. Try `target=5` (worst case)
5. Count comparisons made
6. Print "Found" or "Not Found"
7. Try on unsorted vs sorted arrays
8. Modify to return all indices of target
9. Implement recursive version
10. Extend for string search in list of words

### Test Cases

| Input       | Target | Output | Notes      |
| ----------- | ------ | ------ | ---------- |
| [3,5,7,2,9] | 7      | 2      | Found      |
| [3,5,7,2,9] | 10     | -1     | Not found  |
| [1,2,3]     | 1      | 0      | Best case  |
| [1,2,3]     | 3      | 2      | Worst case |

### Complexity

| Aspect         | Value |
| -------------- | ----- |
| Time (Best)    | O(1)  |
| Time (Worst)   | O(n)  |
| Time (Average) | O(n)  |
| Space          | O(1)  |
| Stable         | Yes   |
| Adaptive       | No    |

Linear Search is the simplest lens into algorithmic thinking, brute force but guaranteed.
It's your first step from guessing to reasoning.

### 152 Linear Search (Sentinel)

Sentinel Linear Search is a clever twist on the basic Linear Search.
Instead of checking array bounds each time, we place a sentinel (a guard value) at the end of the array equal to the target.

This eliminates the need for explicit boundary checks inside the loop, making the search slightly faster and cleaner, especially in low-level languages like C.

### What Problem Are We Solving?

In a standard linear search, each iteration checks both:

1. If the current element equals the target
2. If the index is still within bounds

That second check adds overhead.

By placing a sentinel, we can guarantee the loop will always terminate, no bounds check needed.

This is useful in tight loops, embedded systems, and performance-critical code.

#### Example

Find `7` in `[3, 5, 7, 2, 9]`:

1. Append sentinel (duplicate target) → `[3, 5, 7, 2, 9, 7]`
2. Scan until `7` is found
3. If index < n, found in array
4. If index == n, only sentinel found → not in array

| Step | Index            | Value      | Match? |
| ---- | ---------------- | ---------- | ------ |
| 1    | 0                | 3          | No     |
| 2    | 1                | 5          | No     |
| 3    | 2                | 7          | ✅ Yes  |
| 4    | Stop, index < n | Found at 2 |        |

### How Does It Work (Plain Language)?

Think of the sentinel as a stop sign placed beyond the last element.
You don't have to look over your shoulder to check if you've gone too far, the sentinel will catch you.

It ensures you'll always hit a match, but then you check whether it was a real match or the sentinel.

#### Step-by-Step Process

| Step | Description                            |
| ---- | -------------------------------------- |
| 1    | Save last element                      |
| 2    | Place target at the end (sentinel)     |
| 3    | Loop until `arr[i] == target`          |
| 4    | Restore last element                   |
| 5    | If index < n → found, else → not found |

### Tiny Code (Easy Versions)

#### C

```c
#include <stdio.h>

int sentinel_linear_search(int arr[], int n, int target) {
    int last = arr[n - 1];
    arr[n - 1] = target; // sentinel
    int i = 0;
    while (arr[i] != target)
        i++;
    arr[n - 1] = last; // restore
    if (i < n - 1 || arr[n - 1] == target)
        return i;
    return -1;
}

int main(void) {
    int arr[] = {3, 5, 7, 2, 9};
    int n = sizeof(arr) / sizeof(arr[0]);
    int target = 7;
    int idx = sentinel_linear_search(arr, n, target);
    if (idx != -1)
        printf("Found at index: %d\n", idx);
    else
        printf("Not found\n");
}
```

Output:

```
Found at index: 2
```

#### Python (Simulated)

```python
def sentinel_linear_search(arr, target):
    n = len(arr)
    last = arr[-1]
    arr[-1] = target
    i = 0
    while arr[i] != target:
        i += 1
    arr[-1] = last
    if i < n - 1 or arr[-1] == target:
        return i
    return -1

arr = [3, 5, 7, 2, 9]
print(sentinel_linear_search(arr, 7))  # Output: 2
```

### Why It Matters

- Removes boundary check overhead
- Slightly faster for large arrays
- Classic example of sentinel optimization
- Teaches loop invariants and guard conditions

This is how you make a simple algorithm tight and elegant.

### A Gentle Proof (Why It Works)

By placing the target as the last element:

- Loop must terminate (guaranteed match)
- Only after the loop do we check if it was sentinel or real match

No wasted comparisons.
Total comparisons ≤ n + 1 (vs 2n in naive version).

$$
T(n) = O(n)
$$

### Try It Yourself

1. Search `7` in `[3,5,7,2,9]`
2. Search `10` (not in list)
3. Track number of comparisons vs regular linear search
4. Implement in Python, Java, C++
5. Visualize sentinel placement
6. Use array of 1000 random elements, benchmark
7. Try replacing last element temporarily
8. Search first element, check best case
9. Search last element, check sentinel restore
10. Discuss when this optimization helps most

### Test Cases

| Input       | Target | Output | Notes                  |
| ----------- | ------ | ------ | ---------------------- |
| [3,5,7,2,9] | 7      | 2      | Found                  |
| [3,5,7,2,9] | 10     | -1     | Not found              |
| [1,2,3]     | 1      | 0      | Best case              |
| [1,2,3]     | 3      | 2      | Sentinel replaced last |

### Complexity

| Aspect         | Value |
| -------------- | ----- |
| Time (Best)    | O(1)  |
| Time (Worst)   | O(n)  |
| Time (Average) | O(n)  |
| Space          | O(1)  |
| Stable         | Yes   |
| Adaptive       | No    |

Sentinel Linear Search is how you turn simplicity into elegance, one tiny guard makes the whole loop smarter.

### 153 Binary Search (Iterative)

Binary Search (Iterative) is one of the most elegant and efficient searching algorithms for sorted arrays.
It repeatedly divides the search interval in half, eliminating half the remaining elements at each step.

This version uses a loop, avoiding recursion and keeping memory usage minimal.

### What Problem Are We Solving?

When working with sorted data, a linear scan is wasteful.
If you always know the list is ordered, you can use binary search to find your target in O(log n) time instead of O(n).

#### Example

Find `7` in `[1, 3, 5, 7, 9, 11]`:

| Step | Low | High | Mid | Value | Compare              |
| ---- | --- | ---- | --- | ----- | -------------------- |
| 1    | 0   | 5    | 2   | 5     | 7 > 5 → search right |
| 2    | 3   | 5    | 4   | 9     | 7 < 9 → search left  |
| 3    | 3   | 3    | 3   | 7     | ✅ Found              |

Found at index 3.

### How Does It Work (Plain Language)?

Binary Search is like guessing a number between 1 and 100:

- Always pick the midpoint.
- If the number is too low, search the upper half.
- If it's too high, search the lower half.
- Repeat until found or interval is empty.

Each guess cuts the space in half, that's why it's so fast.

#### Step-by-Step Process

| Step | Description                                                  |
| ---- | ------------------------------------------------------------ |
| 1    | Start with `low = 0`, `high = n - 1`                         |
| 2    | While `low ≤ high`, find `mid = (low + high) // 2`           |
| 3    | If `arr[mid] == target` → return index                       |
| 4    | If `arr[mid] < target` → search right half (`low = mid + 1`) |
| 5    | Else → search left half (`high = mid - 1`)                   |
| 6    | If not found, return -1                                      |

### Tiny Code (Easy Versions)

#### Python

```python
def binary_search(arr, target):
    low, high = 0, len(arr) - 1
    while low <= high:
        mid = (low + high) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            low = mid + 1
        else:
            high = mid - 1
    return -1

arr = [1, 3, 5, 7, 9, 11]
print(binary_search(arr, 7))  # Output: 3
```

Output:

```
3
```

#### C

```c
#include <stdio.h>

int binary_search(int arr[], int n, int target) {
    int low = 0, high = n - 1;
    while (low <= high) {
        int mid = low + (high - low) / 2; // avoid overflow
        if (arr[mid] == target)
            return mid;
        else if (arr[mid] < target)
            low = mid + 1;
        else
            high = mid - 1;
    }
    return -1;
}

int main(void) {
    int arr[] = {1, 3, 5, 7, 9, 11};
    int n = sizeof(arr) / sizeof(arr[0]);
    int idx = binary_search(arr, n, 7);
    if (idx != -1)
        printf("Found at index: %d\n", idx);
    else
        printf("Not found\n");
}
```

Output:

```
Found at index: 3
```

### Why It Matters

- Fundamental divide-and-conquer algorithm
- O(log n) time complexity
- Used everywhere: search engines, databases, compilers
- Builds intuition for binary decision trees

This is the first truly efficient search most programmers learn.

### A Gentle Proof (Why It Works)

At each step, the search interval halves.  
After $k$ steps, remaining elements = $\frac{n}{2^k}$.

Stop when $\frac{n}{2^k} = 1$  
⟹ $k = \log_2 n$

So, total comparisons:
$$
T(n) = O(\log n)
$$

Works only on sorted arrays.


### Try It Yourself

1. Search `7` in `[1,3,5,7,9,11]`
2. Search `2` (not found)
3. Trace values of `low`, `high`, `mid`
4. Try on even-length array `[1,2,3,4,5,6]`
5. Try on odd-length array `[1,2,3,4,5]`
6. Compare iteration count with linear search
7. Implement recursive version
8. Use binary search to find insertion point
9. Add counter to measure steps
10. Explain why sorting is required

### Test Cases

| Input          | Target | Output | Notes         |
| -------------- | ------ | ------ | ------------- |
| [1,3,5,7,9,11] | 7      | 3      | Found         |
| [1,3,5,7,9,11] | 2      | -1     | Not found     |
| [1,2,3,4,5]    | 1      | 0      | First element |
| [1,2,3,4,5]    | 5      | 4      | Last element  |

### Complexity

| Aspect         | Value        |
| -------------- | ------------ |
| Time (Best)    | O(1)         |
| Time (Worst)   | O(log n)     |
| Time (Average) | O(log n)     |
| Space          | O(1)         |
| Stable         | Yes          |
| Adaptive       | No           |
| Prerequisite   | Sorted array |

Binary Search (Iterative) is the gold standard of efficiency, halving your problem at every step, one decision at a time.

### 154 Binary Search (Recursive)

Binary Search (Recursive) is the classic divide-and-conquer form of binary search.
Instead of looping, it calls itself on smaller subarrays, each time halving the search space until the target is found or the interval becomes empty.

It's a perfect demonstration of recursion in action, each call tackles a smaller slice of the problem.

### What Problem Are We Solving?

Given a sorted array, we want to find a target value efficiently.
Rather than scanning linearly, we repeatedly split the array in half, focusing only on the half that could contain the target.

This version expresses that logic through recursive calls.

#### Example

Find `7` in `[1, 3, 5, 7, 9, 11]`

| Step | Low | High | Mid | Value | Action                    |
| ---- | --- | ---- | --- | ----- | ------------------------- |
| 1    | 0   | 5    | 2   | 5     | 7 > 5 → search right half |
| 2    | 3   | 5    | 4   | 9     | 7 < 9 → search left half  |
| 3    | 3   | 3    | 3   | 7     | ✅ Found                   |

Recursive calls shrink the interval each time until match is found.

### How Does It Work (Plain Language)?

Binary search says:

> "If the middle element isn't what I want, I can ignore half the data."

In recursive form:

1. Check the midpoint.
2. If equal → found.
3. If smaller → recurse right.
4. If larger → recurse left.
5. Base case: if `low > high`, element not found.

Each call reduces the search space by half, logarithmic depth recursion.

#### Step-by-Step Process

| Step | Description                                 |
| ---- | ------------------------------------------- |
| 1    | Check base case: if `low > high`, return -1 |
| 2    | Compute `mid = (low + high) // 2`           |
| 3    | If `arr[mid] == target`, return `mid`       |
| 4    | If `arr[mid] > target`, recurse left half   |
| 5    | Else, recurse right half                    |

### Tiny Code (Easy Versions)

#### Python

```python
def binary_search_recursive(arr, target, low, high):
    if low > high:
        return -1
    mid = (low + high) // 2
    if arr[mid] == target:
        return mid
    elif arr[mid] < target:
        return binary_search_recursive(arr, target, mid + 1, high)
    else:
        return binary_search_recursive(arr, target, low, mid - 1)

arr = [1, 3, 5, 7, 9, 11]
print(binary_search_recursive(arr, 7, 0, len(arr)-1))  # Output: 3
```

Output:

```
3
```

#### C

```c
#include <stdio.h>

int binary_search_recursive(int arr[], int low, int high, int target) {
    if (low > high)
        return -1;
    int mid = low + (high - low) / 2;
    if (arr[mid] == target)
        return mid;
    else if (arr[mid] < target)
        return binary_search_recursive(arr, mid + 1, high, target);
    else
        return binary_search_recursive(arr, low, mid - 1, target);
}

int main(void) {
    int arr[] = {1, 3, 5, 7, 9, 11};
    int n = sizeof(arr) / sizeof(arr[0]);
    int idx = binary_search_recursive(arr, 0, n - 1, 7);
    if (idx != -1)
        printf("Found at index: %d\n", idx);
    else
        printf("Not found\n");
}
```

Output:

```
Found at index: 3
```

### Why It Matters

- Elegant divide-and-conquer demonstration
- Shows recursion depth = log₂(n)
- Same complexity as iterative version
- Lays foundation for recursive algorithms (merge sort, quicksort)

Recursion mirrors the mathematical idea of halving the interval, clean and intuitive.

### A Gentle Proof (Why It Works)

At each recursive call:

- Problem size halves: $n \to n/2 \to n/4 \to \dots$
- Recursion stops after $\log_2 n$ levels

Thus total complexity:
$$
T(n) = T(n/2) + O(1) = O(\log n)
$$

Correctness follows from:

- Sorted input ensures ordering decisions are valid  
- Base case ensures termination


### Try It Yourself

1. Search `7` in `[1,3,5,7,9,11]`
2. Search `2` (not present)
3. Add print statements to trace recursion
4. Compare call count vs iterative version
5. Test base case with `low > high`
6. Search first and last element
7. Observe recursion depth for size 8 → 3 calls
8. Add memo of `(low, high)` per step
9. Implement tail-recursive variant
10. Reflect on stack vs loop tradeoffs

### Test Cases

| Input          | Target | Output | Notes         |
| -------------- | ------ | ------ | ------------- |
| [1,3,5,7,9,11] | 7      | 3      | Found         |
| [1,3,5,7,9,11] | 2      | -1     | Not found     |
| [1,2,3,4,5]    | 1      | 0      | First element |
| [1,2,3,4,5]    | 5      | 4      | Last element  |

### Complexity

| Aspect         | Value                      |
| -------------- | -------------------------- |
| Time (Best)    | O(1)                       |
| Time (Worst)   | O(log n)                   |
| Time (Average) | O(log n)                   |
| Space          | O(log n) (recursion stack) |
| Stable         | Yes                        |
| Adaptive       | No                         |
| Prerequisite   | Sorted array               |

Binary Search (Recursive) is divide-and-conquer at its purest, halving the world each time, until the answer reveals itself.

### 155 Binary Search (Lower Bound)

Lower Bound Binary Search is a variant of binary search that finds the first position where a value could be inserted without breaking the sorted order.
In other words, it returns the index of the first element greater than or equal to the target.

It's used extensively in search engines, databases, range queries, and C++ STL (`std::lower_bound`).

### What Problem Are We Solving?

In many cases, you don't just want to know if an element exists —
you want to know where it would go in a sorted structure.

- If the value exists, return its first occurrence.
- If it doesn't exist, return the position where it could be inserted to keep the array sorted.

This is essential for insertion, frequency counting, and range boundaries.

#### Example

Find lower bound of `7` in `[1, 3, 5, 7, 7, 9, 11]`:

| Step | Low         | High | Mid | Value | Compare       | Action        |
| ---- | ----------- | ---- | --- | ----- | ------------- | ------------- |
| 1    | 0           | 6    | 3   | 7     | arr[mid] >= 7 | move high = 3 |
| 2    | 0           | 2    | 1   | 3     | arr[mid] < 7  | move low = 2  |
| 3    | 2           | 3    | 2   | 5     | arr[mid] < 7  | move low = 3  |
| 4    | low == high |,    |,   |,     | Stop          |               |

Result: Index 3 (first `7`)

### How Does It Work (Plain Language)?

Lower bound finds the leftmost slot where the target fits.
It slides the search window until `low == high`, with `low` marking the first candidate ≥ target.

You can think of it as:

> "How far left can I go while still being ≥ target?"

#### Step-by-Step Process

| Step                                             | Description                                           |
| ------------------------------------------------ | ----------------------------------------------------- |
| 1                                                | Set `low = 0`, `high = n` (note: `high` = n, not n-1) |
| 2                                                | While `low < high`:                                   |
|  a. `mid = (low + high) // 2`                    |                                                       |
|  b. If `arr[mid] < target`, move `low = mid + 1` |                                                       |
|  c. Else, move `high = mid`                      |                                                       |
| 3                                                | When loop ends, `low` is the lower bound index        |

### Tiny Code (Easy Versions)

#### Python

```python
def lower_bound(arr, target):
    low, high = 0, len(arr)
    while low < high:
        mid = (low + high) // 2
        if arr[mid] < target:
            low = mid + 1
        else:
            high = mid
    return low

arr = [1, 3, 5, 7, 7, 9, 11]
print(lower_bound(arr, 7))  # Output: 3
```

Output:

```
3
```

#### C

```c
#include <stdio.h>

int lower_bound(int arr[], int n, int target) {
    int low = 0, high = n;
    while (low < high) {
        int mid = low + (high - low) / 2;
        if (arr[mid] < target)
            low = mid + 1;
        else
            high = mid;
    }
    return low;
}

int main(void) {
    int arr[] = {1, 3, 5, 7, 7, 9, 11};
    int n = sizeof(arr) / sizeof(arr[0]);
    int idx = lower_bound(arr, n, 7);
    printf("Lower bound index: %d\n", idx);
}
```

Output:

```
Lower bound index: 3
```

### Why It Matters

- Finds insertion position for sorted arrays
- Useful in binary search trees, maps, intervals
- Core of range queries (like count of elements ≥ x)
- Builds understanding of boundary binary search

When you need more than yes/no, you need where, use lower bound.

### A Gentle Proof (Why It Works)

Invariant:

- All indices before `low` contain elements `< target`
- All indices after `high` contain elements `≥ target`

Loop maintains invariant until convergence:
$$
low = high = \text{first index where arr[i] ≥ target}
$$

Complexity:
$$
T(n) = O(\log n)
$$

### Try It Yourself

1. `arr = [1,3,5,7,7,9]`, `target = 7` → 3
2. `target = 6` → 3 (would insert before first 7)
3. `target = 10` → 6 (end)
4. `target = 0` → 0 (front)
5. Count elements ≥ 7: `len(arr) - lower_bound(arr, 7)`
6. Compare with `bisect_left` in Python
7. Use to insert elements while keeping list sorted
8. Apply to sorted strings `["apple", "banana", "cherry"]`
9. Visualize range [lower, upper) for duplicates
10. Benchmark vs linear scan for large n

### Test Cases

| Input            | Target | Output | Meaning         |
| ---------------- | ------ | ------ | --------------- |
| [1,3,5,7,7,9,11] | 7      | 3      | First 7         |
| [1,3,5,7,7,9,11] | 6      | 3      | Insert before 7 |
| [1,3,5,7,7,9,11] | 12     | 7      | End position    |
| [1,3,5,7,7,9,11] | 0      | 0      | Front           |

### Complexity

| Aspect       | Value        |
| ------------ | ------------ |
| Time         | O(log n)     |
| Space        | O(1)         |
| Stable       | Yes          |
| Adaptive     | No           |
| Prerequisite | Sorted array |

Binary Search (Lower Bound) is how algorithms learn *where* things belong, not just if they exist.
It's the precise edge of order.

### 156 Binary Search (Upper Bound)

Upper Bound Binary Search is a close cousin of Lower Bound, designed to find the first index where an element is greater than the target.
It's used to locate the right boundary of equal elements or the insertion point after duplicates.

In simpler words:

> It finds "the spot just after the last occurrence" of the target.

### What Problem Are We Solving?

Sometimes you need to know where to insert a value after existing duplicates.
For example, in frequency counting or range queries, you might want the end of a block of identical elements.

Upper bound returns that exact position, the smallest index where
$$
\text{arr[i]} > \text{target}
$$

#### Example

Find upper bound of `7` in `[1, 3, 5, 7, 7, 9, 11]`:

| Step | Low         | High | Mid | Value | Compare | Action        |
| ---- | ----------- | ---- | --- | ----- | ------- | ------------- |
| 1    | 0           | 7    | 3   | 7     | 7 ≤ 7   | move low = 4  |
| 2    | 4           | 7    | 5   | 9     | 9 > 7   | move high = 5 |
| 3    | 4           | 5    | 4   | 7     | 7 ≤ 7   | move low = 5  |
| 4    | low == high |,    |,   |,     | Stop    |               |

Result: Index 5 → position after last `7`.

### How Does It Work (Plain Language)?

If Lower Bound finds "the first ≥ target,"
then Upper Bound finds "the first > target."

Together, they define equal ranges:
$$
$$\text{lower\_bound}, \text{upper\_bound})
$$
→ all elements equal to the target.

#### Step-by-Step Process

| Step                                              | Description                                    |
| ------------------------------------------------- | ---------------------------------------------- |
| 1                                                 | Set `low = 0`, `high = n`                      |
| 2                                                 | While `low < high`:                            |
|  a. `mid = (low + high) // 2`                     |                                                |
|  b. If `arr[mid] <= target`, move `low = mid + 1` |                                                |
|  c. Else, move `high = mid`                       |                                                |
| 3                                                 | When loop ends, `low` is the upper bound index |

### Tiny Code (Easy Versions)

#### Python

```python
def upper_bound(arr, target):
    low, high = 0, len(arr)
    while low < high:
        mid = (low + high) // 2
        if arr[mid] <= target:
            low = mid + 1
        else:
            high = mid
    return low

arr = [1, 3, 5, 7, 7, 9, 11]
print(upper_bound(arr, 7))  # Output: 5
```

Output:

```
5
```

#### C

```c
#include <stdio.h>

int upper_bound(int arr[], int n, int target) {
    int low = 0, high = n;
    while (low < high) {
        int mid = low + (high - low) / 2;
        if (arr[mid] <= target)
            low = mid + 1;
        else
            high = mid;
    }
    return low;
}

int main(void) {
    int arr[] = {1, 3, 5, 7, 7, 9, 11};
    int n = sizeof(arr) / sizeof(arr[0]);
    int idx = upper_bound(arr, n, 7);
    printf("Upper bound index: %d\n", idx);
}
```

Output:

```
Upper bound index: 5
```

### Why It Matters

- Locates end of duplicate block
- Enables range counting:
   count = `upper_bound - lower_bound`
- Used in maps, sets, STL containers
- Key component in range queries and interval merging

It's the *right-hand anchor* of sorted intervals.

### A Gentle Proof (Why It Works)

Invariant:

- All indices before `low` contain $\le$ target  
- All indices after `high` contain $>$ target  

When `low == high`, it's the smallest index where `arr[i] > target`.

Number of steps = $\log_2 n$  
→ Complexity:
$$
T(n) = O(\log n)
$$


### Try It Yourself

1. `arr = [1,3,5,7,7,9]`, `target=7` → 5
2. `target=6` → 3 (insert after 5)
3. `target=10` → 6 (end)
4. `target=0` → 0 (front)
5. Count elements equal to 7 → `upper - lower`
6. Compare with `bisect_right` in Python
7. Use to insert new element after duplicates
8. Combine with lower bound to find range
9. Visualize [lower, upper) range
10. Apply to floating point sorted list

### Test Cases

| Input            | Target | Output | Meaning        |
| ---------------- | ------ | ------ | -------------- |
| [1,3,5,7,7,9,11] | 7      | 5      | After last 7   |
| [1,3,5,7,7,9,11] | 6      | 3      | Insert after 5 |
| [1,3,5,7,7,9,11] | 12     | 7      | End position   |
| [1,3,5,7,7,9,11] | 0      | 0      | Front          |

### Complexity

| Aspect       | Value        |
| ------------ | ------------ |
| Time         | O(log n)     |
| Space        | O(1)         |
| Stable       | Yes          |
| Adaptive     | No           |
| Prerequisite | Sorted array |

Binary Search (Upper Bound) is how you find where "greater than" begins, the right edge of equality, the step beyond the last twin.

### 157 Exponential Search

Exponential Search is a hybrid search algorithm that quickly locates the range where a target might lie, then uses binary search inside that range.

It's ideal when searching unbounded or very large sorted arrays, especially when the size is unknown or dynamic (like data streams or infinite arrays).

### What Problem Are We Solving?

In a standard binary search, you need the array size.
But what if the size is unknown, or huge?

Exponential Search solves this by:

1. Quickly finding an interval where the target could exist.
2. Performing binary search within that interval.

This makes it great for:

- Infinite arrays (conceptual)
- Streams
- Linked structures with known order
- Large sorted data where bounds are costly

#### Example

Find `15` in `[1, 2, 4, 8, 16, 32, 64, 128]`

| Step | Range          | Value                       | Compare               | Action |
| ---- | -------------- | --------------------------- | --------------------- | ------ |
| 1    | index 1        | 2                           | 2 < 15                | expand |
| 2    | index 2        | 4                           | 4 < 15                | expand |
| 3    | index 4        | 16                          | 16 ≥ 15               | stop   |
| 4    | Range = [2, 4] | Binary Search in [4, 8, 16] | ✅ Found 15 at index 4 |        |

We doubled the bound each time (1, 2, 4, 8...) until we passed the target.

### How Does It Work (Plain Language)?

Think of it like zooming in:

- Start small, double your step size until you overshoot the target.
- Once you've bracketed it, zoom in with binary search.

This avoids scanning linearly when the array could be massive.

#### Step-by-Step Process

| Step | Description                                     |
| ---- | ----------------------------------------------- |
| 1    | Start with index 1                              |
| 2    | While `i < n` and `arr[i] < target`, double `i` |
| 3    | Now target ∈ `[i/2, min(i, n-1)]`               |
| 4    | Apply binary search in that subrange            |
| 5    | Return found index or -1                        |

### Tiny Code (Easy Versions)

#### Python

```python
def binary_search(arr, target, low, high):
    while low <= high:
        mid = (low + high) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            low = mid + 1
        else:
            high = mid - 1
    return -1

def exponential_search(arr, target):
    if arr[0] == target:
        return 0
    i = 1
    n = len(arr)
    while i < n and arr[i] < target:
        i *= 2
    return binary_search(arr, target, i // 2, min(i, n - 1))

arr = [1, 2, 4, 8, 16, 32, 64, 128]
print(exponential_search(arr, 16))  # Output: 4
```

Output:

```
4
```

#### C

```c
#include <stdio.h>

int binary_search(int arr[], int low, int high, int target) {
    while (low <= high) {
        int mid = low + (high - low) / 2;
        if (arr[mid] == target)
            return mid;
        else if (arr[mid] < target)
            low = mid + 1;
        else
            high = mid - 1;
    }
    return -1;
}

int exponential_search(int arr[], int n, int target) {
    if (arr[0] == target)
        return 0;
    int i = 1;
    while (i < n && arr[i] < target)
        i *= 2;
    int low = i / 2;
    int high = (i < n) ? i : n - 1;
    return binary_search(arr, low, high, target);
}

int main(void) {
    int arr[] = {1, 2, 4, 8, 16, 32, 64, 128};
    int n = sizeof(arr) / sizeof(arr[0]);
    int idx = exponential_search(arr, n, 16);
    printf("Found at index: %d\n", idx);
}
```

Output:

```
Found at index: 4
```

### Why It Matters

- Works with unknown or infinite size arrays
- Faster than linear scan for large `n`
- Combines doubling search and binary search
- Used in streaming data structures, file systems, unbounded containers

It's the searcher's flashlight, shine brighter until you see your target.

### A Gentle Proof (Why It Works)

Doubling creates at most $\log_2 p$ expansions,  
where $p$ is the position of the target.

Then binary search on a range of size $O(p)$ takes another $O(\log p)$.

So total time:
$$
T(n) = O(\log p)
$$

Asymptotically equal to binary search when $p \ll n$.


### Try It Yourself

1. Search `16` in `[1,2,4,8,16,32,64,128]`
2. Search `3` → not found
3. Trace expansions: 1,2,4,8...
4. Try `[10,20,30,40,50,60,70]` with `target=60`
5. Modify doubling factor (try 3x)
6. Compare with simple binary search
7. Implement recursive exponential search
8. Use for unknown-size input stream
9. Measure expansion count
10. Visualize ranges on number line

### Test Cases

| Input              | Target | Output | Notes         |
| ------------------ | ------ | ------ | ------------- |
| [1,2,4,8,16,32,64] | 16     | 4      | Classic       |
| [1,2,4,8,16,32,64] | 3      | -1     | Not found     |
| [2,4,6,8]          | 2      | 0      | First element |
| [2,4,6,8]          | 10     | -1     | Out of range  |

### Complexity

| Aspect       | Value                                  |
| ------------ | -------------------------------------- |
| Time         | O(log p), where p = position of target |
| Space        | O(1)                                   |
| Stable       | Yes                                    |
| Adaptive     | Partially                              |
| Prerequisite | Sorted array                           |

Exponential Search is how you find your way in the dark, double your reach, then look closely where the light lands.

### 158 Jump Search

Jump Search is a simple improvement over Linear Search, designed for sorted arrays.
It works by jumping ahead in fixed-size steps instead of checking every element, then performing a linear scan within the block where the target might be.

It trades a bit of extra logic for a big win in speed, especially when data is sorted and random access is cheap.

### What Problem Are We Solving?

Linear search checks each element one by one, slow for large arrays.
Jump Search improves on this by "skipping ahead" in fixed jumps, so it makes fewer comparisons overall.

It's great for:

- Sorted arrays
- Fast random access (like arrays, not linked lists)
- Simple, predictable search steps

#### Example

Find `9` in `[1, 3, 5, 7, 9, 11, 13, 15]`

Array size = 8 → Jump size = √8 = 2 or 3

| Step | Jump to Index      | Value   | Compare   | Action       |
| ---- | ------------------ | ------- | --------- | ------------ |
| 1    | 2                  | 5       | 5 < 9     | jump forward |
| 2    | 4                  | 9       | 9 ≥ 9     | stop jump    |
| 3    | Linear scan from 2 | 5, 7, 9 | ✅ found 9 |              |

Found at index 4.

### How Does It Work (Plain Language)?

Imagine you're reading a sorted list of numbers.
Instead of reading one number at a time, you skip ahead every few steps, like jumping stairs.
Once you overshoot or match, you walk back linearly within that block.

Jump size ≈ √n gives a good balance between jumps and scans.

#### Step-by-Step Process

| Step | Description                                                |
| ---- | ---------------------------------------------------------- |
| 1    | Choose jump size `step = √n`                               |
| 2    | Jump ahead while `arr[min(step, n)-1] < target`            |
| 3    | Once overshoot, do linear search in the previous block |
| 4    | If found, return index, else -1                            |

### Tiny Code (Easy Versions)

#### Python

```python
import math

def jump_search(arr, target):
    n = len(arr)
    step = int(math.sqrt(n))
    prev = 0

    # Jump ahead
    while prev < n and arr[min(step, n) - 1] < target:
        prev = step
        step += int(math.sqrt(n))
        if prev >= n:
            return -1

    # Linear search within block
    for i in range(prev, min(step, n)):
        if arr[i] == target:
            return i
    return -1

arr = [1, 3, 5, 7, 9, 11, 13, 15]
print(jump_search(arr, 9))  # Output: 4
```

Output:

```
4
```

#### C

```c
#include <stdio.h>
#include <math.h>

int jump_search(int arr[], int n, int target) {
    int step = sqrt(n);
    int prev = 0;

    while (prev < n && arr[(step < n ? step : n) - 1] < target) {
        prev = step;
        step += sqrt(n);
        if (prev >= n) return -1;
    }

    for (int i = prev; i < (step < n ? step : n); i++) {
        if (arr[i] == target) return i;
    }
    return -1;
}

int main(void) {
    int arr[] = {1, 3, 5, 7, 9, 11, 13, 15};
    int n = sizeof(arr) / sizeof(arr[0]);
    int idx = jump_search(arr, n, 9);
    printf("Found at index: %d\n", idx);
}
```

Output:

```
Found at index: 4
```

### Why It Matters

- Faster than Linear Search for sorted arrays
- Simpler to implement than Binary Search
- Good for systems with non-random access penalties
- Balance of jumps and scans minimizes comparisons

Best of both worlds: quick leaps and gentle steps.

### A Gentle Proof (Why It Works)

Let jump size = $m$.

We perform at most $\frac{n}{m}$ jumps and $m$ linear steps.  
Total cost:
$$
T(n) = O\left(\frac{n}{m} + m\right)
$$

Minimized when $m = \sqrt{n}$  

Thus:
$$
T(n) = O(\sqrt{n})
$$


### Try It Yourself

1. Search `11` in `[1,3,5,7,9,11,13,15]`
2. Search `2` (not found)
3. Use step = 2, 3, 4 → compare jumps
4. Implement recursive version
5. Visualize jumps on paper
6. Try unsorted array (observe failure)
7. Search edge cases: first, last, middle
8. Use different step size formula
9. Combine with exponential jump
10. Compare with binary search for timing

### Test Cases

| Input                | Target | Output | Notes         |
| -------------------- | ------ | ------ | ------------- |
| [1,3,5,7,9,11,13,15] | 9      | 4      | Found         |
| [1,3,5,7,9,11,13,15] | 2      | -1     | Not found     |
| [2,4,6,8,10]         | 10     | 4      | Last element  |
| [2,4,6,8,10]         | 2      | 0      | First element |

### Complexity

| Aspect       | Value        |
| ------------ | ------------ |
| Time         | O(√n)        |
| Space        | O(1)         |
| Stable       | Yes          |
| Adaptive     | No           |
| Prerequisite | Sorted array |

Jump Search is like hopscotch on sorted ground, leap smartly, then step carefully once you're close.

### 159 Fibonacci Search

Fibonacci Search is a divide-and-conquer search algorithm that uses Fibonacci numbers to determine probe positions, rather than midpoints like Binary Search.
It's especially efficient for sorted arrays stored in sequential memory, where element access cost grows with distance (for example, on magnetic tape or cache-sensitive systems).

It's a clever twist on binary search, using Fibonacci numbers instead of powers of two.

### What Problem Are We Solving?

In Binary Search, the midpoint splits the array evenly.
In Fibonacci Search, we split using Fibonacci ratios, which keeps indices aligned to integer arithmetic, no division needed, and better cache locality in some hardware.

This is particularly useful when:

- Access cost depends on distance
- Memory access is sequential or limited
- We want to avoid division and floating-point math

#### Example

Find `8` in `[1, 3, 5, 8, 13, 21, 34]`

| Step | Fib(k) | Index Checked | Value | Compare | Action     |
| ---- | ------ | ------------- | ----- | ------- | ---------- |
| 1    | 8      | 5             | 13    | 13 > 8  | move left  |
| 2    | 5      | 2             | 5     | 5 < 8   | move right |
| 3    | 3      | 3             | 8     | 8 = 8   | ✅ found    |

Fibonacci numbers: 1, 2, 3, 5, 8, 13, …
We reduce the search space using previous Fibonacci values, like Fibonacci decomposition.

### How Does It Work (Plain Language)?

Think of Fibonacci Search as binary search guided by Fibonacci jumps.
At each step:

- You compare the element at index `offset + Fib(k-2)`
- Depending on the result, you move left or right, using smaller Fibonacci numbers to define new intervals.

You "walk down" the Fibonacci sequence until the range collapses.

#### Step-by-Step Process

| Step | Description                                                 |
| ---- | ----------------------------------------------------------- |
| 1    | Generate smallest Fibonacci number ≥ n                      |
| 2    | Use Fib(k-2) as probe index                                 |
| 3    | Compare target with `arr[offset + Fib(k-2)]`                |
| 4    | If smaller, move left (reduce by Fib(k-2))                  |
| 5    | If larger, move right (increase offset, reduce by Fib(k-1)) |
| 6    | Continue until Fib(k) = 1                                   |
| 7    | Check last element if needed                                |

### Tiny Code (Easy Versions)

#### Python

```python
def fibonacci_search(arr, target):
    n = len(arr)
    fibMMm2 = 0  # (m-2)'th Fibonacci
    fibMMm1 = 1  # (m-1)'th Fibonacci
    fibM = fibMMm1 + fibMMm2  # m'th Fibonacci

    # Find smallest Fibonacci >= n
    while fibM < n:
        fibMMm2, fibMMm1 = fibMMm1, fibM
        fibM = fibMMm1 + fibMMm2

    offset = -1

    while fibM > 1:
        i = min(offset + fibMMm2, n - 1)

        if arr[i] < target:
            fibM = fibMMm1
            fibMMm1 = fibMMm2
            fibMMm2 = fibM - fibMMm1
            offset = i
        elif arr[i] > target:
            fibM = fibMMm2
            fibMMm1 -= fibMMm2
            fibMMm2 = fibM - fibMMm1
        else:
            return i

    if fibMMm1 and offset + 1 < n and arr[offset + 1] == target:
        return offset + 1

    return -1

arr = [1, 3, 5, 8, 13, 21, 34]
print(fibonacci_search(arr, 8))  # Output: 3
```

Output:

```
3
```

#### C

```c
#include <stdio.h>

int min(int a, int b) { return (a < b) ? a : b; }

int fibonacci_search(int arr[], int n, int target) {
    int fibMMm2 = 0;
    int fibMMm1 = 1;
    int fibM = fibMMm1 + fibMMm2;

    while (fibM < n) {
        fibMMm2 = fibMMm1;
        fibMMm1 = fibM;
        fibM = fibMMm1 + fibMMm2;
    }

    int offset = -1;

    while (fibM > 1) {
        int i = min(offset + fibMMm2, n - 1);

        if (arr[i] < target) {
            fibM = fibMMm1;
            fibMMm1 = fibMMm2;
            fibMMm2 = fibM - fibMMm1;
            offset = i;
        } else if (arr[i] > target) {
            fibM = fibMMm2;
            fibMMm1 -= fibMMm2;
            fibMMm2 = fibM - fibMMm1;
        } else {
            return i;
        }
    }

    if (fibMMm1 && offset + 1 < n && arr[offset + 1] == target)
        return offset + 1;

    return -1;
}

int main(void) {
    int arr[] = {1, 3, 5, 8, 13, 21, 34};
    int n = sizeof(arr) / sizeof(arr[0]);
    int idx = fibonacci_search(arr, n, 8);
    printf("Found at index: %d\n", idx);
}
```

Output:

```
Found at index: 3
```

### Why It Matters

- Uses only addition and subtraction, no division
- Great for sequential access memory
- Matches binary search performance (O(log n))
- Offers better locality on some hardware

It's the search strategy built from nature's own numbers.

### A Gentle Proof (Why It Works)

Each iteration reduces the search space by a Fibonacci ratio:
$$
F_k = F_{k-1} + F_{k-2}
$$

Hence, search depth ≈ Fibonacci index $k \sim \log_\phi n$,  
where $\phi$ is the golden ratio ($\approx 1.618$).

So total time:
$$
T(n) = O(\log n)
$$


### Try It Yourself

1. Search `8` in `[1,3,5,8,13,21,34]`
2. Search `21`
3. Search `2` (not found)
4. Trace Fibonacci sequence steps
5. Compare probe indices with binary search
6. Try with `n=10` → Fibonacci 13 ≥ 10
7. Implement recursive version
8. Visualize probe intervals
9. Replace Fibonacci with powers of 2 (binary search)
10. Experiment with non-uniform arrays

### Test Cases

| Input              | Target | Output | Notes         |
| ------------------ | ------ | ------ | ------------- |
| [1,3,5,8,13,21,34] | 8      | 3      | Found         |
| [1,3,5,8,13,21,34] | 2      | -1     | Not found     |
| [1,3,5,8,13,21,34] | 34     | 6      | Last element  |
| [1,3,5,8,13,21,34] | 1      | 0      | First element |

### Complexity

| Aspect       | Value        |
| ------------ | ------------ |
| Time         | O(log n)     |
| Space        | O(1)         |
| Stable       | Yes          |
| Adaptive     | No           |
| Prerequisite | Sorted array |

Fibonacci Search is searching with nature's rhythm, each step shaped by the golden ratio, balancing reach and precision.

### 160 Uniform Binary Search

Uniform Binary Search is an optimized form of Binary Search where the probe positions are precomputed.
Instead of recalculating the midpoint at each step, it uses a lookup table of offsets to determine where to go next, making it faster in tight loops or hardware-limited systems.

It's all about removing repetitive midpoint arithmetic and branching for consistent, uniform steps.

### What Problem Are We Solving?

Standard binary search repeatedly computes:

$$
mid = low + \frac{high - low}{2}
$$

This is cheap on modern CPUs but expensive on:

- Early hardware
- Embedded systems
- Tight loops where division or shifting is costly

Uniform Binary Search (UBS) replaces these computations with precomputed offsets for each step, giving predictable, uniform jumps.

#### Example

Search `25` in `[5, 10, 15, 20, 25, 30, 35, 40]`

| Step | Offset | Index | Value | Compare | Action     |
| ---- | ------ | ----- | ----- | ------- | ---------- |
| 1    | 3      | 3     | 20    | 20 < 25 | move right |
| 2    | 1      | 5     | 30    | 30 > 25 | move left  |
| 3    | 0      | 4     | 25    | 25 = 25 | ✅ found    |

Instead of recalculating midpoints, UBS uses offset table `[3, 1, 0]`.

### How Does It Work (Plain Language)?

It's binary search with a preplanned route.
At each level:

- You jump by a fixed offset (from a table)
- Compare
- Move left or right, shrinking your window uniformly

No divisions, no mid calculations, just jump and compare.

#### Step-by-Step Process

| Step | Description                                 |
| ---- | ------------------------------------------- |
| 1    | Precompute offset table based on array size |
| 2    | Start at offset[0] from beginning           |
| 3    | Compare element with target                 |
| 4    | Move left/right using next offset           |
| 5    | Stop when offset = 0 or found               |

### Tiny Code (Easy Versions)

#### Python

```python
def uniform_binary_search(arr, target):
    n = len(arr)
    # Precompute offsets (powers of 2 less than n)
    offsets = []
    k = 1
    while k < n:
        offsets.append(k)
        k *= 2
    offsets.reverse()

    low = 0
    idx = offsets[0]

    for offset in offsets:
        mid = low + offset if low + offset < n else n - 1
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            low = mid + 1
        # else move left (implicitly handled next iteration)

    # Final check
    if low < n and arr[low] == target:
        return low
    return -1

arr = [5, 10, 15, 20, 25, 30, 35, 40]
print(uniform_binary_search(arr, 25))  # Output: 4
```

Output:

```
4
```

#### C

```c
#include <stdio.h>

int uniform_binary_search(int arr[], int n, int target) {
    int k = 1;
    while (k < n) k <<= 1;
    k >>= 1;

    int low = 0;
    while (k > 0) {
        int mid = low + k - 1;
        if (mid >= n) mid = n - 1;

        if (arr[mid] == target)
            return mid;
        else if (arr[mid] < target)
            low = mid + 1;

        k >>= 1; // next smaller offset
    }

    return (low < n && arr[low] == target) ? low : -1;
}

int main(void) {
    int arr[] = {5, 10, 15, 20, 25, 30, 35, 40};
    int n = sizeof(arr) / sizeof(arr[0]);
    int idx = uniform_binary_search(arr, n, 25);
    printf("Found at index: %d\n", idx);
}
```

Output:

```
Found at index: 4
```

### Why It Matters

- Avoids recomputing midpoints
- Ideal for hardware, firmware, microcontrollers
- Ensures consistent runtime path
- Fewer instructions → faster in tight loops

Uniformity = predictability = performance.

### A Gentle Proof (Why It Works)

Precomputed offsets correspond to halving the search space.
At each step, offset = floor(remaining_size / 2).
After log₂(n) steps, we narrow down to a single element.

Total steps = ⌈log₂ n⌉
Each step = constant cost (no recomputation)

So:
$$
T(n) = O(\log n)
$$

### Try It Yourself

1. Search `25` in `[5,10,15,20,25,30,35,40]`
2. Search `35`
3. Search `6` → not found
4. Precompute offset table for n=8
5. Compare steps with binary search
6. Implement for n=16
7. Use for microcontroller table lookup
8. Visualize jumps on paper
9. Measure comparisons
10. Implement recursive version

### Test Cases

| Input                    | Target | Output | Notes         |
| ------------------------ | ------ | ------ | ------------- |
| [5,10,15,20,25,30,35,40] | 25     | 4      | Found         |
| [5,10,15,20,25,30,35,40] | 5      | 0      | First element |
| [5,10,15,20,25,30,35,40] | 50     | -1     | Not found     |
| [5,10,15,20,25,30,35,40] | 40     | 7      | Last element  |

### Complexity

| Aspect       | Value        |
| ------------ | ------------ |
| Time         | O(log n)     |
| Space        | O(1)         |
| Stable       | Yes          |
| Adaptive     | No           |
| Prerequisite | Sorted array |

Uniform Binary Search is binary search with a map, no guesswork, no recalculations, just smooth, evenly spaced jumps to the answer.

## Section 17. Interpolation and exponential search 

### 161 Interpolation Search

Interpolation Search is a search algorithm for sorted arrays with uniformly distributed values.
Unlike Binary Search, which always probes the middle, Interpolation Search estimates the position of the target using value interpolation, like finding where a number lies on a number line.

If Binary Search is "divide by index," Interpolation Search is "divide by value."

### What Problem Are We Solving?

Binary Search assumes no relationship between index and value.
But if the array values are uniformly spaced, we can do better by guessing where the target should be, not just the middle.

It's ideal for:

- Uniformly distributed sorted data
- Numeric keys (IDs, prices, timestamps)
- Large arrays where value-based positioning matters

#### Example

Find `70` in `[10, 20, 30, 40, 50, 60, 70, 80, 90]`

Estimate position:

$$
pos = low + \frac{(target - arr[low]) \times (high - low)}{arr[high] - arr[low]}
$$

| Step | Low | High | Pos | Value | Compare | Action    |
| ---- | --- | ---- | --- | ----- | ------- | --------- |
| 1    | 0   | 8    | 7   | 80    | 80 > 70 | move left |
| 2    | 0   | 6    | 6   | 70    | 70 = 70 | ✅ found   |

### How Does It Work (Plain Language)?

Imagine the array as a number line.
If your target is closer to the high end, start searching closer to the right.
You *interpolate*, estimate the target's index based on its value proportion between min and max.

So rather than halving blindly, you jump to the likely spot.

#### Step-by-Step Process

| Step                                            | Description                                   |
| ----------------------------------------------- | --------------------------------------------- |
| 1                                               | Initialize `low = 0`, `high = n - 1`          |
| 2                                               | While `low <= high` and `target` is in range: |
|   Estimate position using interpolation formula |                                               |
| 3                                               | Compare `arr[pos]` with `target`              |
| 4                                               | If equal → found                              |
| 5                                               | If smaller → move `low = pos + 1`             |
| 6                                               | If larger → move `high = pos - 1`             |
| 7                                               | Repeat until found or `low > high`            |

### Tiny Code (Easy Versions)

#### Python

```python
def interpolation_search(arr, target):
    low, high = 0, len(arr) - 1

    while low <= high and arr[low] <= target <= arr[high]:
        if arr[high] == arr[low]:
            if arr[low] == target:
                return low
            break

        pos = low + (target - arr[low]) * (high - low) // (arr[high] - arr[low])

        if arr[pos] == target:
            return pos
        elif arr[pos] < target:
            low = pos + 1
        else:
            high = pos - 1

    return -1

arr = [10, 20, 30, 40, 50, 60, 70, 80, 90]
print(interpolation_search(arr, 70))  # Output: 6
```

Output:

```
6
```

#### C

```c
#include <stdio.h>

int interpolation_search(int arr[], int n, int target) {
    int low = 0, high = n - 1;

    while (low <= high && target >= arr[low] && target <= arr[high]) {
        if (arr[high] == arr[low]) {
            if (arr[low] == target) return low;
            else break;
        }

        int pos = low + (double)(high - low) * (target - arr[low]) / (arr[high] - arr[low]);

        if (arr[pos] == target)
            return pos;
        if (arr[pos] < target)
            low = pos + 1;
        else
            high = pos - 1;
    }

    return -1;
}

int main(void) {
    int arr[] = {10, 20, 30, 40, 50, 60, 70, 80, 90};
    int n = sizeof(arr) / sizeof(arr[0]);
    int idx = interpolation_search(arr, n, 70);
    printf("Found at index: %d\n", idx);
}
```

Output:

```
Found at index: 6
```

### Why It Matters

- Faster than Binary Search for uniformly distributed data
- Ideal for dense key spaces (like hash slots, ID ranges)
- Can achieve O(log log n) average time
- Preserves sorted order search logic

It's the value-aware cousin of Binary Search.

### A Gentle Proof (Why It Works)

If data is uniformly distributed, each probe halves value range logarithmically.
Expected probes:
$$
T(n) = O(\log \log n)
$$
Worst case (non-uniform):
$$
T(n) = O(n)
$$

So it outperforms binary search only under uniform value distribution.

### Try It Yourself

1. Search `70` in `[10,20,30,40,50,60,70,80,90]`
2. Search `25` (not found)
3. Try `arr = [2,4,8,16,32,64]`, notice uneven distribution
4. Plot estimated positions
5. Compare steps with Binary Search
6. Use floating-point vs integer division
7. Implement recursive version
8. Search `10` (first element)
9. Search `90` (last element)
10. Measure iterations on uniform vs non-uniform data

### Test Cases

| Input                        | Target | Output | Notes     |
| ---------------------------- | ------ | ------ | --------- |
| [10,20,30,40,50,60,70,80,90] | 70     | 6      | Found     |
| [10,20,30,40,50,60,70,80,90] | 25     | -1     | Not found |
| [10,20,30,40,50,60,70,80,90] | 10     | 0      | First     |
| [10,20,30,40,50,60,70,80,90] | 90     | 8      | Last      |

### Complexity

| Aspect       | Value                  |
| ------------ | ---------------------- |
| Time (avg)   | O(log log n)           |
| Time (worst) | O(n)                   |
| Space        | O(1)                   |
| Stable       | Yes                    |
| Adaptive     | Yes                    |
| Prerequisite | Sorted & uniform array |

Interpolation Search is like a treasure map that scales by value, it doesn't just guess the middle, it guesses where the gold really lies.

### 162 Recursive Interpolation Search

Recursive Interpolation Search is the recursive variant of the classic Interpolation Search.
Instead of looping, it calls itself on smaller subranges, estimating the likely position using the same value-based interpolation formula.

It's a natural way to express the algorithm for learners who think recursively, same logic, cleaner flow.

### What Problem Are We Solving?

We're taking the iterative interpolation search and expressing it recursively, to highlight the divide-and-conquer nature.
The recursive form is often more intuitive and mathematically aligned with its interpolation logic.

You'll use this when:

- Teaching or visualizing recursive logic
- Writing clean, declarative search code
- Practicing recursion-to-iteration transitions

#### Example

Find `50` in `[10, 20, 30, 40, 50, 60, 70, 80, 90]`

Step 1:
low = 0, high = 8
pos = 0 + (50 - 10) × (8 - 0) / (90 - 10) = 4
arr[4] = 50 → ✅ found

Recursion stops immediately, one call, one success.

### How Does It Work (Plain Language)?

Instead of looping, each recursive call zooms in to the subrange where the target might be.
Each call computes an estimated index `pos` proportional to the target's distance between `arr[low]` and `arr[high]`.

If value is higher → recurse right
If lower → recurse left
If equal → return index

#### Step-by-Step Process

| Step | Description                                                   |
| ---- | ------------------------------------------------------------- |
| 1    | Base case: if `low > high` or target out of range → not found |
| 2    | Compute position estimate using interpolation formula         |
| 3    | If `arr[pos] == target` → return `pos`                        |
| 4    | If `arr[pos] < target` → recurse on right half                |
| 5    | Else recurse on left half                                     |

### Tiny Code (Easy Versions)

#### Python

```python
def interpolation_search_recursive(arr, low, high, target):
    if low > high or target < arr[low] or target > arr[high]:
        return -1

    if arr[high] == arr[low]:
        return low if arr[low] == target else -1

    pos = low + (target - arr[low]) * (high - low) // (arr[high] - arr[low])

    if arr[pos] == target:
        return pos
    elif arr[pos] < target:
        return interpolation_search_recursive(arr, pos + 1, high, target)
    else:
        return interpolation_search_recursive(arr, low, pos - 1, target)

arr = [10, 20, 30, 40, 50, 60, 70, 80, 90]
print(interpolation_search_recursive(arr, 0, len(arr) - 1, 50))  # Output: 4
```

Output:

```
4
```

#### C

```c
#include <stdio.h>

int interpolation_search_recursive(int arr[], int low, int high, int target) {
    if (low > high || target < arr[low] || target > arr[high])
        return -1;

    if (arr[high] == arr[low])
        return (arr[low] == target) ? low : -1;

    int pos = low + (double)(high - low) * (target - arr[low]) / (arr[high] - arr[low]);

    if (arr[pos] == target)
        return pos;
    else if (arr[pos] < target)
        return interpolation_search_recursive(arr, pos + 1, high, target);
    else
        return interpolation_search_recursive(arr, low, pos - 1, target);
}

int main(void) {
    int arr[] = {10, 20, 30, 40, 50, 60, 70, 80, 90};
    int n = sizeof(arr) / sizeof(arr[0]);
    int idx = interpolation_search_recursive(arr, 0, n - 1, 50);
    printf("Found at index: %d\n", idx);
}
```

Output:

```
Found at index: 4
```

### Why It Matters

- Shows the recursive nature of interpolation-based estimation
- Clean, mathematical form for teaching and analysis
- Demonstrates recursion depth proportional to search cost
- Easier to reason about with divide-by-value intuition

Think of it as binary search's artistic cousin, elegant and value-aware.

### A Gentle Proof (Why It Works)

In uniform distributions, each recursive step shrinks the search space multiplicatively, not just by half.

If data is uniform:
$$
T(n) = T(n / f) + O(1)
\Rightarrow T(n) = O(\log \log n)
$$

If not uniform:
$$
T(n) = O(n)
$$

Recursion depth = number of interpolation refinements ≈ log log n.

### Try It Yourself

1. Search `70` in `[10,20,30,40,50,60,70,80,90]`
2. Search `25` (not found)
3. Trace recursive calls manually
4. Add print statements to watch low/high shrink
5. Compare depth with binary search
6. Try `[2,4,8,16,32,64]` (non-uniform)
7. Add guard for `arr[low] == arr[high]`
8. Implement tail recursion optimization
9. Test base cases (first, last, single element)
10. Time comparison: iterative vs recursive

### Test Cases

| Input                        | Target | Output | Notes     |
| ---------------------------- | ------ | ------ | --------- |
| [10,20,30,40,50,60,70,80,90] | 50     | 4      | Found     |
| [10,20,30,40,50,60,70,80,90] | 25     | -1     | Not found |
| [10,20,30,40,50,60,70,80,90] | 10     | 0      | First     |
| [10,20,30,40,50,60,70,80,90] | 90     | 8      | Last      |

### Complexity

| Aspect       | Value                    |
| ------------ | ------------------------ |
| Time (avg)   | O(log log n)             |
| Time (worst) | O(n)                     |
| Space        | O(log log n) (recursion) |
| Stable       | Yes                      |
| Adaptive     | Yes                      |
| Prerequisite | Sorted & uniform array   |

Recursive Interpolation Search is like zooming in with value-based intuition, each step a mathematical guess, each call a sharper focus.

### 163 Exponential Search

Exponential Search combines range expansion with binary search.
It's perfect when the array size is unknown or infinite, it first finds the range that might contain the target by doubling the index, then uses binary search within that range.

It's the "zoom out, then zoom in" strategy for searching sorted data.

### What Problem Are We Solving?

If you don't know the length of your sorted array, you can't directly apply binary search.
You need to first bound the search space, so you know where to look.

Exponential Search does exactly that:

- It doubles the index (1, 2, 4, 8, 16...) until it overshoots.
- Then it performs binary search in that bracket.

Use it for:

- Infinite or dynamically sized sorted arrays
- Streams
- Unknown-length files or data structures

#### Example

Find `19` in `[2, 4, 8, 16, 19, 23, 42, 64, 128]`

| Step | Range | Value | Compare | Action       |
| ---- | ----- | ----- | ------- | ------------ |
| 1    | 1     | 4     | 4 < 19  | double index |
| 2    | 2     | 8     | 8 < 19  | double index |
| 3    | 4     | 19    | 19 = 19 | ✅ found      |

If it had overshot (say, target 23), we'd binary search between 4 and 8.

### How Does It Work (Plain Language)?

Think of searching an endless bookshelf.
You take steps of size 1, 2, 4, 8... until you pass the book number you want.
Now you know which shelf section it's on, then you check precisely using binary search.

Fast to expand, precise to finish.

#### Step-by-Step Process

| Step | Description                                                    |
| ---- | -------------------------------------------------------------- |
| 1    | Start at index 1                                               |
| 2    | While `i < n` and `arr[i] < target`, double `i`                |
| 3    | When overshoot → binary search between `i/2` and `min(i, n-1)` |
| 4    | Return index if found, else -1                                 |

### Tiny Code (Easy Versions)

#### Python

```python
def binary_search(arr, low, high, target):
    while low <= high:
        mid = (low + high) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            low = mid + 1
        else:
            high = mid - 1
    return -1

def exponential_search(arr, target):
    n = len(arr)
    if n == 0:
        return -1
    if arr[0] == target:
        return 0
    i = 1
    while i < n and arr[i] <= target:
        i *= 2
    return binary_search(arr, i // 2, min(i, n - 1), target)

arr = [2, 4, 8, 16, 19, 23, 42, 64, 128]
print(exponential_search(arr, 19))  # Output: 4
```

Output:

```
4
```

#### C

```c
#include <stdio.h>

int binary_search(int arr[], int low, int high, int target) {
    while (low <= high) {
        int mid = low + (high - low) / 2;
        if (arr[mid] == target)
            return mid;
        else if (arr[mid] < target)
            low = mid + 1;
        else
            high = mid - 1;
    }
    return -1;
}

int exponential_search(int arr[], int n, int target) {
    if (n == 0) return -1;
    if (arr[0] == target) return 0;

    int i = 1;
    while (i < n && arr[i] <= target)
        i *= 2;

    int low = i / 2;
    int high = (i < n) ? i : n - 1;
    return binary_search(arr, low, high, target);
}

int main(void) {
    int arr[] = {2, 4, 8, 16, 19, 23, 42, 64, 128};
    int n = sizeof(arr) / sizeof(arr[0]);
    int idx = exponential_search(arr, n, 19);
    printf("Found at index: %d\n", idx);
}
```

Output:

```
Found at index: 4
```

### Why It Matters

- Handles unknown size arrays
- Logarithmic search after expansion
- Fewer comparisons for small targets
- Common in unbounded search, streams, linked memory

It's the search that grows as needed, like zooming your scope until the target appears.

### A Gentle Proof (Why It Works)

Each iteration reduces the search space by a Fibonacci ratio:
$$
F_k = F_{k-1} + F_{k-2}
$$

Hence, search depth ≈ Fibonacci index $k \sim \log_\phi n$,  
where $\phi$ is the golden ratio ($\approx 1.618$).

So total time:
$$
T(n) = O(\log n)
$$


### Try It Yourself

1. Search `19` in `[2,4,8,16,19,23,42,64,128]`
2. Search `42`
3. Search `1` (not found)
4. Trace expansion: 1, 2, 4, 8...
5. Compare expansion steps vs binary search calls
6. Try on huge array, small target
7. Try on dynamic-size list (simulate infinite)
8. Implement recursive version
9. Measure expansions vs comparisons
10. Combine with galloping search in TimSort

### Test Cases

| Input                      | Target | Output | Notes        |
| -------------------------- | ------ | ------ | ------------ |
| [2,4,8,16,19,23,42,64,128] | 19     | 4      | Found        |
| [2,4,8,16,19,23,42,64,128] | 23     | 5      | Found        |
| [2,4,8,16,19,23,42,64,128] | 1      | -1     | Not found    |
| [2,4,8,16,19,23,42,64,128] | 128    | 8      | Last element |

### Complexity

| Aspect       | Value        |
| ------------ | ------------ |
| Time         | O(log p)     |
| Space        | O(1)         |
| Stable       | Yes          |
| Adaptive     | Yes          |
| Prerequisite | Sorted array |

Exponential Search is your wayfinder, it reaches out in powers of two, then narrows in precisely where the target hides.

### 164 Doubling Search

Doubling Search (also called Unbounded Search) is the general pattern behind Exponential Search.
It's used when the data size or range is unknown, and you need to quickly discover a search interval that contains the target before performing a precise search (like binary search) inside that interval.

Think of it as *"search by doubling until you find your bracket."*

### What Problem Are We Solving?

In many real-world scenarios, you don't know the array's length or the bounds of your search domain.
You can't jump straight into binary search, you need an upper bound first.

Doubling Search gives you a strategy to find that bound quickly, in logarithmic time, by doubling your index or step size until you overshoot the target.

Perfect for:

- Infinite or streaming sequences
- Functions or implicit arrays (not stored in memory)
- Search domains defined by value, not length

#### Example

Find `23` in `[2, 4, 8, 16, 19, 23, 42, 64, 128]`

| Step | Range | Value | Compare  | Action   |
| ---- | ----- | ----- | -------- | -------- |
| 1    | i = 1 | 4     | 4 < 23   | double i |
| 2    | i = 2 | 8     | 8 < 23   | double i |
| 3    | i = 4 | 16    | 16 < 23  | double i |
| 4    | i = 8 | 128   | 128 > 23 | stop     |

Range found: `[4, 8]`
Now run binary search within `[16, 19, 23, 42, 64, 128]`
✅ Found at index 5

### How Does It Work (Plain Language)?

Start small, test the first few steps.
Every time you don't find your target and the value is still less, double your index (1, 2, 4, 8, 16...).
Once you pass your target, you've found your interval, then you search precisely.

It's like walking in the dark and doubling your stride each time until you see the light, then stepping back carefully.

#### Step-by-Step Process

| Step | Description                                       |
| ---- | ------------------------------------------------- |
| 1    | Start at index 1                                  |
| 2    | While `arr[i] < target`, set `i = 2 × i`          |
| 3    | When overshoot, define range `[i/2, min(i, n-1)]` |
| 4    | Apply binary search within range                  |
| 5    | Return index or -1 if not found                   |

### Tiny Code (Easy Versions)

#### Python

```python
def doubling_search(arr, target):
    n = len(arr)
    if n == 0:
        return -1
    if arr[0] == target:
        return 0

    i = 1
    while i < n and arr[i] < target:
        i *= 2

    low = i // 2
    high = min(i, n - 1)

    # binary search
    while low <= high:
        mid = (low + high) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            low = mid + 1
        else:
            high = mid - 1

    return -1

arr = [2, 4, 8, 16, 19, 23, 42, 64, 128]
print(doubling_search(arr, 23))  # Output: 5
```

Output:

```
5
```

#### C

```c
#include <stdio.h>

int binary_search(int arr[], int low, int high, int target) {
    while (low <= high) {
        int mid = low + (high - low) / 2;
        if (arr[mid] == target)
            return mid;
        else if (arr[mid] < target)
            low = mid + 1;
        else
            high = mid - 1;
    }
    return -1;
}

int doubling_search(int arr[], int n, int target) {
    if (n == 0) return -1;
    if (arr[0] == target) return 0;

    int i = 1;
    while (i < n && arr[i] < target)
        i *= 2;

    int low = i / 2;
    int high = (i < n) ? i : n - 1;

    return binary_search(arr, low, high, target);
}

int main(void) {
    int arr[] = {2, 4, 8, 16, 19, 23, 42, 64, 128};
    int n = sizeof(arr) / sizeof(arr[0]);
    int idx = doubling_search(arr, n, 23);
    printf("Found at index: %d\n", idx);
}
```

Output:

```
Found at index: 5
```

### Why It Matters

- Allows binary search when size is unknown
- Only O(log p) probes, where `p` is target index
- Natural strategy for streaming, infinite, or lazy structures
- Used in exponential, galloping, and tim-sort merges

It's the blueprint for all "expand then search" algorithms.

### A Gentle Proof (Why It Works)

Each doubling step multiplies range by 2,
so number of expansions ≈ log₂(p)
Then binary search inside a range of size ≤ p

Total time:
$$
T(n) = O(\log p)
$$

Still logarithmic in target position, not array size, efficient even for unbounded data.

### Try It Yourself

1. Search `23` in `[2,4,8,16,19,23,42,64,128]`
2. Search `42` (more jumps)
3. Search `1` (before first element)
4. Search `128` (last element)
5. Try doubling factor 3 instead of 2
6. Compare expansion steps with exponential search
7. Implement recursive version
8. Visualize on a number line
9. Simulate unknown-length array with a safe check
10. Measure steps for small vs large targets

### Test Cases

| Input                      | Target | Output | Notes        |
| -------------------------- | ------ | ------ | ------------ |
| [2,4,8,16,19,23,42,64,128] | 23     | 5      | Found        |
| [2,4,8,16,19,23,42,64,128] | 4      | 1      | Early        |
| [2,4,8,16,19,23,42,64,128] | 1      | -1     | Not found    |
| [2,4,8,16,19,23,42,64,128] | 128    | 8      | Last element |

### Complexity

| Aspect       | Value        |
| ------------ | ------------ |
| Time         | O(log p)     |
| Space        | O(1)         |
| Stable       | Yes          |
| Adaptive     | Yes          |
| Prerequisite | Sorted array |

Doubling Search is how you explore the unknown, double your reach, find your bounds, then pinpoint your goal.

### 165 Galloping Search

Galloping Search (also called Exponential Gallop or Search by Doubling) is a hybrid search technique that quickly finds a range by *galloping forward exponentially*, then switches to binary search within that range.

It's heavily used inside TimSort's merge step, where it helps merge sorted runs efficiently by skipping over large stretches that don't need detailed comparison.

### What Problem Are We Solving?

When merging two sorted arrays (or searching in a sorted list), repeatedly comparing one by one is wasteful if elements differ by a large margin.
Galloping Search "jumps ahead" quickly to find the region of interest, then finishes with a binary search to locate the exact boundary.

Used in:

- TimSort merges
- Run merging in hybrid sorting
- Searching in sorted blocks
- Optimizing comparison-heavy algorithms

#### Example

Find `25` in `[5, 10, 15, 20, 25, 30, 35, 40, 45]`

| Step | Index | Value | Compare | Action      |
| ---- | ----- | ----- | ------- | ----------- |
| 1    | 1     | 10    | 10 < 25 | gallop (×2) |
| 2    | 2     | 15    | 15 < 25 | gallop      |
| 3    | 4     | 25    | 25 = 25 | ✅ found     |

If we overshoot (say, search for `22`), we'd gallop past it (index 4), then perform binary search between the previous bound and current one.

### How Does It Work (Plain Language)?

You start by leaping, checking 1, 2, 4, 8 steps away, until you pass the target or reach the end.
Once you overshoot, you gallop back (binary search) in the last valid interval.

It's like sprinting ahead to find the neighborhood, then walking house-to-house once you know the block.

#### Step-by-Step Process

| Step | Description                                                                        |
| ---- | ---------------------------------------------------------------------------------- |
| 1    | Start at `start` index                                                             |
| 2    | Gallop forward by powers of 2 (`1, 2, 4, 8, ...`) until `arr[start + k] >= target` |
| 3    | Define range `[start + k/2, min(start + k, n-1)]`                                  |
| 4    | Apply binary search in that range                                                  |
| 5    | Return index if found                                                              |

### Tiny Code (Easy Versions)

#### Python

```python
def binary_search(arr, low, high, target):
    while low <= high:
        mid = (low + high) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            low = mid + 1
        else:
            high = mid - 1
    return -1

def galloping_search(arr, start, target):
    n = len(arr)
    if start >= n:
        return -1
    if arr[start] == target:
        return start

    k = 1
    while start + k < n and arr[start + k] < target:
        k *= 2

    low = start + k // 2
    high = min(start + k, n - 1)
    return binary_search(arr, low, high, target)

arr = [5, 10, 15, 20, 25, 30, 35, 40, 45]
print(galloping_search(arr, 0, 25))  # Output: 4
```

Output:

```
4
```

#### C

```c
#include <stdio.h>

int binary_search(int arr[], int low, int high, int target) {
    while (low <= high) {
        int mid = low + (high - low) / 2;
        if (arr[mid] == target)
            return mid;
        else if (arr[mid] < target)
            low = mid + 1;
        else
            high = mid - 1;
    }
    return -1;
}

int galloping_search(int arr[], int n, int start, int target) {
    if (start >= n) return -1;
    if (arr[start] == target) return start;

    int k = 1;
    while (start + k < n && arr[start + k] < target)
        k *= 2;

    int low = start + k / 2;
    int high = (start + k < n) ? start + k : n - 1;

    return binary_search(arr, low, high, target);
}

int main(void) {
    int arr[] = {5, 10, 15, 20, 25, 30, 35, 40, 45};
    int n = sizeof(arr) / sizeof(arr[0]);
    int idx = galloping_search(arr, n, 0, 25);
    printf("Found at index: %d\n", idx);
}
```

Output:

```
Found at index: 4
```

### Why It Matters

- Accelerates merging in TimSort
- Minimizes comparisons when merging large sorted runs
- Great for adaptive sorting and range scans
- Balances speed (gallop) and precision (binary)

It's a dynamic balance, gallop when far, tiptoe when close.

### A Gentle Proof (Why It Works)

Galloping phase:  
Takes $O(\log p)$ steps to reach the target vicinity (where $p$ is the distance from the start).

Binary phase:  
Another $O(\log p)$ for local search.

Total:
$$
T(n) = O(\log p)
$$

Faster than linear merging when runs differ greatly in length.


### Try It Yourself

1. Search `25` in `[5,10,15,20,25,30,35,40,45]`
2. Search `40` (larger gallop)
3. Search `3` (before first)
4. Try `start=2`
5. Print gallop steps
6. Compare with pure binary search
7. Implement recursive gallop
8. Use for merging two sorted arrays
9. Count comparisons per search
10. Test on long list with early/late targets

### Test Cases

| Input                       | Target | Output | Notes         |
| --------------------------- | ------ | ------ | ------------- |
| [5,10,15,20,25,30,35,40,45] | 25     | 4      | Found         |
| [5,10,15,20,25,30,35,40,45] | 40     | 7      | Larger gallop |
| [5,10,15,20,25,30,35,40,45] | 3      | -1     | Not found     |
| [5,10,15,20,25,30,35,40,45] | 5      | 0      | First element |

### Complexity

| Aspect       | Value        |
| ------------ | ------------ |
| Time         | O(log p)     |
| Space        | O(1)         |
| Stable       | Yes          |
| Adaptive     | Yes          |
| Prerequisite | Sorted array |

Galloping Search is how TimSort runs ahead with confidence, sprint through big gaps, slow down only when precision counts.

### 166 Unbounded Binary Search

Unbounded Binary Search is a technique for finding a value in a sorted but unbounded (or infinite) sequence.
You don't know where the data ends, or even how large it is, but you know it's sorted. So first, you find a search boundary, then perform a binary search within that discovered range.

It's a direct application of doubling search followed by binary search, especially suited for monotonic functions or streams.

### What Problem Are We Solving?

If you're working with data that:

- Has no fixed size,
- Comes as a stream,
- Or is represented as a monotonic function (like f(x) increasing with x),

then you can't apply binary search immediately, because you don't know `high`.

So you first need to find bounds [low, high] that contain the target, by expanding exponentially, and only then use binary search.

#### Example

Suppose you're searching for `20` in a monotonic function:

```
f(x) = 2x + 2
```

You want the smallest `x` such that `f(x) ≥ 20`.

Step 1: Find bounds.
Check x = 1 → f(1) = 4
Check x = 2 → f(2) = 6
Check x = 4 → f(4) = 10
Check x = 8 → f(8) = 18
Check x = 16 → f(16) = 34 (overshoot!)

Now you know target lies between x = 8 and x = 16.

Step 2: Perform binary search in [8, 16].
✅ Found f(9) = 20

### How Does It Work (Plain Language)?

You start with a small step and double your reach each time until you go beyond the target.
Once you've overshot, you know the search interval, and you can binary search inside it for precision.

It's the go-to strategy when the array (or domain) has no clear end.

#### Step-by-Step Process

| Step | Description                                                   |
| ---- | ------------------------------------------------------------- |
| 1    | Initialize `low = 0`, `high = 1`                              |
| 2    | While `f(high) < target`, set `low = high`, `high *= 2`       |
| 3    | Once `f(high) >= target`, binary search between `[low, high]` |
| 4    | Return index (or value)                                       |

### Tiny Code (Easy Versions)

#### Python

```python
def unbounded_binary_search(f, target):
    low, high = 0, 1
    # Step 1: find bounds
    while f(high) < target:
        low = high
        high *= 2

    # Step 2: binary search in [low, high]
    while low <= high:
        mid = (low + high) // 2
        val = f(mid)
        if val == target:
            return mid
        elif val < target:
            low = mid + 1
        else:
            high = mid - 1
    return -1

# Example: f(x) = 2x + 2
f = lambda x: 2 * x + 2
print(unbounded_binary_search(f, 20))  # Output: 9
```

Output:

```
9
```

#### C

```c
#include <stdio.h>

int f(int x) {
    return 2 * x + 2;
}

int binary_search(int (*f)(int), int low, int high, int target) {
    while (low <= high) {
        int mid = low + (high - low) / 2;
        int val = f(mid);
        if (val == target)
            return mid;
        else if (val < target)
            low = mid + 1;
        else
            high = mid - 1;
    }
    return -1;
}

int unbounded_binary_search(int (*f)(int), int target) {
    int low = 0, high = 1;
    while (f(high) < target) {
        low = high;
        high *= 2;
    }
    return binary_search(f, low, high, target);
}

int main(void) {
    int idx = unbounded_binary_search(f, 20);
    printf("Found at x = %d\n", idx);
}
```

Output:

```
Found at x = 9
```

### Why It Matters

- Handles infinite sequences or unbounded domains
- Perfect for monotonic functions (e.g. f(x) increasing)
- Key in searching without array length
- Used in root finding, binary lifting, streaming

It's the "expand, then refine" pattern, the explorer's search.

### A Gentle Proof (Why It Works)

Expanding step: doubles index each time → $O(\log p)$, where `p` is the position of the target.  
Binary search step: searches within a range of size ≤ p → another $O(\log p)$.

Total complexity:
$$
T(n) = O(\log p)
$$

Independent of total domain size.


### Try It Yourself

1. `f(x) = 2x + 2`, target = 20 → 9
2. `f(x) = x²`, target = 64 → 8
3. Search for non-existing value (e.g. 7 in even series)
4. Modify f(x) to be exponential
5. Print bounds before binary search
6. Try with negative domain (guard with if)
7. Apply to sorted infinite list (simulate with f)
8. Use float function and tolerance check
9. Compare with linear probing
10. Use to find smallest x where f(x) ≥ target

### Test Cases

| Function | Target | Output | Notes     |
| -------- | ------ | ------ | --------- |
| 2x+2     | 20     | 9      | f(9)=20   |
| x²       | 64     | 8      | f(8)=64   |
| 2x+2     | 7      | -1     | Not found |
| 3x+1     | 31     | 10     | f(10)=31  |

### Complexity

| Aspect       | Value                             |
| ------------ | --------------------------------- |
| Time         | O(log p)                          |
| Space        | O(1)                              |
| Stable       | Yes                               |
| Adaptive     | Yes                               |
| Prerequisite | Monotonic function or sorted data |

Unbounded Binary Search is how you search the infinite, double your bounds, then zoom in on the truth.

### 167 Root-Finding Bisection

Root-Finding Bisection is a numerical search algorithm for locating the point where a continuous function crosses zero.
It repeatedly halves an interval where the sign of the function changes, guaranteeing that a root exists within that range.

It's the simplest, most reliable method for solving equations like f(x) = 0 when you only know that a solution exists somewhere between `a` and `b`.

### What Problem Are We Solving?

When you have a function $f(x)$ and want to solve $f(x) = 0$,  
but you can't solve it algebraically, you can use the **Bisection Method**.

If $f(a)$ and $f(b)$ have opposite signs, then by the Intermediate Value Theorem,  
there is at least one root between $a$ and $b$.

Example:  
Find the root of $f(x) = x^3 - x - 2$ in $[1, 2]$.

- $f(1) = -2$  
- $f(2) = 4$  
  Signs differ → there's a root in $[1, 2]$.


### How Does It Work (Plain Language)?

You start with an interval $[a, b]$ where $f(a)$ and $f(b)$ have opposite signs.  
Then, at each step:

1. Find midpoint $m = \frac{a + b}{2}$  
2. Evaluate $f(m)$  
3. Replace either `a` or `b` so that signs at the ends still differ  
4. Repeat until the interval is small enough  

You're squeezing the interval tighter and tighter until it hugs the root.


#### Step-by-Step Process

| Step | Description                                                  |      |      |       |               |
| ---- | ------------------------------------------------------------ | ---- | ---- | ----- | ------------- |
| 1    | Choose `a`, `b` with `f(a)` and `f(b)` of opposite signs     |      |      |       |               |
| 2    | Compute midpoint `m = (a + b) / 2`                           |      |      |       |               |
| 3    | Evaluate `f(m)`                                              |      |      |       |               |
| 4    | If `f(m)` has same sign as `f(a)`, set `a = m`, else `b = m` |      |      |       |               |
| 5    | Repeat until `                                               | f(m) | `or` | b - a | ` < tolerance |
| 6    | Return `m` as root                                           |      |      |       |               |

### Tiny Code (Easy Versions)

#### Python

```python
def f(x):
    return x3 - x - 2

def bisection(f, a, b, tol=1e-6):
    if f(a) * f(b) >= 0:
        raise ValueError("No sign change: root not guaranteed")
    while (b - a) / 2 > tol:
        m = (a + b) / 2
        if f(m) == 0:
            return m
        elif f(a) * f(m) < 0:
            b = m
        else:
            a = m
    return (a + b) / 2

root = bisection(f, 1, 2)
print("Approx root:", root)
```

Output:

```
Approx root: 1.52138
```

#### C

```c
#include <stdio.h>
#include <math.h>

double f(double x) {
    return x*x*x - x - 2;
}

double bisection(double (*f)(double), double a, double b, double tol) {
    if (f(a) * f(b) >= 0) {
        printf("No sign change. Root not guaranteed.\n");
        return NAN;
    }
    double m;
    while ((b - a) / 2 > tol) {
        m = (a + b) / 2;
        double fm = f(m);
        if (fm == 0)
            return m;
        if (f(a) * fm < 0)
            b = m;
        else
            a = m;
    }
    return (a + b) / 2;
}

int main(void) {
    double root = bisection(f, 1, 2, 1e-6);
    printf("Approx root: %.6f\n", root);
}
```

Output:

```
Approx root: 1.521380
```

### Why It Matters

- Guaranteed convergence if ( f ) is continuous and signs differ
- Simple and robust
- Works for nonlinear equations
- Great starting point before more advanced methods (Newton, Secant)
- Used in physics, finance, engineering for precise solving

### A Gentle Proof (Why It Works)

By the Intermediate Value Theorem:  
If $f(a) \cdot f(b) < 0$, then there exists $c \in [a, b]$ such that $f(c) = 0$.

Each iteration halves the interval size, so after $k$ steps:


$$
b_k - a_k = \frac{b_0 - a_0}{2^k}
$$

To achieve tolerance $\varepsilon$:

$$
k \approx \log_2\left(\frac{b_0 - a_0}{\varepsilon}\right)
$$

Thus, convergence is linear, but guaranteed.

### Try It Yourself

1. $f(x) = x^2 - 2$, interval $[1, 2]$ → $\sqrt{2}$  
2. $f(x) = \cos x - x$, interval $[0, 1]$  
3. $f(x) = x^3 - 7$, interval $[1, 3]$  
4. Try tighter tolerances (`1e-3`, `1e-9`)
5. Count how many iterations needed
6. Print each midpoint
7. Compare with Newton's method
8. Plot convergence curve
9. Modify to return both root and iterations
10. Test on function with multiple roots

### Test Cases

| Function        | Interval | Root (Approx) | Notes            |
| ---------------- | -------- | ------------- | ---------------- |
| $x^2 - 2$        | $[1, 2]$ | 1.4142        | $\sqrt{2}$       |
| $x^3 - x - 2$    | $[1, 2]$ | 1.5214        | Classic example  |
| $\cos x - x$     | $[0, 1]$ | 0.7391        | Fixed point root |
| $x^3 - 7$        | $[1, 3]$ | 1.913         | Cube root of 7   |


### Complexity

| Aspect              | Value             |
| ------------------- | ----------------- |
| Time                | O(log((b−a)/tol)) |
| Space               | O(1)              |
| Convergence         | Linear            |
| Stability           | High              |
| Requires continuity | Yes               |

Bisection Method is your steady compass in numerical analysis, it never fails when a root is truly there.

### 168 Golden Section Search

Golden Section Search is a clever optimization algorithm for finding the minimum (or maximum) of a unimodal function on a closed interval ([a, b])—without derivatives.
It's a close cousin of binary search, but instead of splitting in half, it uses the golden ratio to minimize function evaluations.

### What Problem Are We Solving?

You want to find the x that minimizes f(x) on an interval ([a, b]),
but you can't or don't want to take derivatives (maybe f is noisy or discontinuous).

If f(x) is unimodal (has a single peak or trough),
then the Golden Section Search gives you a guaranteed narrowing path to the optimum.

#### Example

Find the minimum of
$$
f(x) = (x-2)^2 + 1
$$
on ([0, 5])

Since (f(x)) is quadratic, its minimum is at (x = 2).
The algorithm will zoom in around (2) by evaluating at golden-ratio points.

### How Does It Work (Plain Language)?

Imagine slicing your search interval using the golden ratio (( \phi \approx 1.618 )).
By placing test points at those ratios, you can reuse past evaluations
and eliminate one side of the interval every step.

Each iteration shrinks the search range while keeping the ratio intact —
like a mathematically perfect zoom.

#### Step-by-Step Process

| Step | Description                                                       | Condition              | Result / Note          |
| ---- | ----------------------------------------------------------------- | ---------------------- | ---------------------- |
| 1    | Choose initial interval $[a, b]$                                  |                        |                        |
| 2    | Compute golden ratio $\phi = \frac{\sqrt{5} - 1}{2} \approx 0.618$ |                        |                        |
| 3    | Set $c = b - \phi(b - a)$, $d = a + \phi(b - a)$                 |                        |                        |
| 4    | Evaluate $f(c)$ and $f(d)$                                       |                        |                        |
| 5    | If $f(c) < f(d)$, new interval = $[a, d]$                        |                        |                        |
| 6    | Else, new interval = $[c, b]$                                    |                        |                        |
| 7    | Repeat until $b - a < \text{tolerance}$                          |                        |                        |
| 8    | Return midpoint as best estimate                                 |                        |                        |


### Tiny Code (Easy Versions)

#### Python

```python
import math

def f(x):
    return (x - 2)2 + 1

def golden_section_search(f, a, b, tol=1e-5):
    phi = (math.sqrt(5) - 1) / 2
    c = b - phi * (b - a)
    d = a + phi * (b - a)
    fc, fd = f(c), f(d)
    
    while abs(b - a) > tol:
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - phi * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + phi * (b - a)
            fd = f(d)
    return (b + a) / 2

root = golden_section_search(f, 0, 5)
print("Minimum near x =", root)
```

Output:

```
Minimum near x = 2.0000
```

#### C

```c
#include <stdio.h>
#include <math.h>

double f(double x) {
    return (x - 2)*(x - 2) + 1;
}

double golden_section_search(double (*f)(double), double a, double b, double tol) {
    double phi = (sqrt(5.0) - 1) / 2;
    double c = b - phi * (b - a);
    double d = a + phi * (b - a);
    double fc = f(c), fd = f(d);

    while (fabs(b - a) > tol) {
        if (fc < fd) {
            b = d;
            d = c;
            fd = fc;
            c = b - phi * (b - a);
            fc = f(c);
        } else {
            a = c;
            c = d;
            fc = fd;
            d = a + phi * (b - a);
            fd = f(d);
        }
    }
    return (b + a) / 2;
}

int main(void) {
    double x = golden_section_search(f, 0, 5, 1e-5);
    printf("Minimum near x = %.5f\n", x);
}
```

Output:

```
Minimum near x = 2.00000
```

### Why It Matters

- No derivatives required
- Fewer evaluations than simple binary search
- Guaranteed convergence for unimodal functions
- Used in numerical optimization, tuning, engineering design, hyperparameter search
- The golden ratio ensures optimal reuse of computed points

### A Gentle Proof (Why It Works)

At each step, the interval length is multiplied by $\phi \approx 0.618$.

So after $k$ steps:
$$
L_k = (b_0 - a_0) \times \phi^k
$$

To reach tolerance $\varepsilon$:
$$
k = \frac{\log(\varepsilon / (b_0 - a_0))}{\log(\phi)}
$$

Thus, convergence is linear but efficient, and each iteration needs only one new evaluation.


### Try It Yourself

1. $f(x) = (x - 3)^2$, interval $[0, 6]$  
2. $f(x) = x^4 - 10x^2 + 9$, interval $[-5, 5]$  
3. $f(x) = |x - 1|$, interval $[-2, 4]$  
4. Try changing tolerance to `1e-3`, `1e-9`
5. Track number of iterations
6. Plot search intervals
7. Switch to maximizing (compare signs)
8. Test non-unimodal function (observe failure)
9. Modify to return f(x*) as well
10. Compare with ternary search

### Test Cases

| Function      | Interval | Minimum $x$ | $f(x)$  |
| -------------- | -------- | ------------ | -------- |
| $(x - 2)^2 + 1$ | $[0, 5]$ | 2.0000       | 1.0000   |
| $(x - 3)^2$     | $[0, 6]$ | 3.0000       | 0.0000   |
| $x^2$           | $[-3, 2]$ | 0.0000       | 0.0000   |


### Complexity

| Aspect               | Value             |
| -------------------- | ----------------- |
| Time                 | O(log((b−a)/tol)) |
| Space                | O(1)              |
| Evaluations per step | 1 new point       |
| Convergence          | Linear            |
| Requires unimodality | Yes               |

Golden Section Search is optimization's quiet craftsman, balancing precision and simplicity with the elegance of φ.

### 169 Fibonacci Search (Optimum)

Fibonacci Search is a divide-and-conquer algorithm that uses the Fibonacci sequence to determine probe positions when searching for a target in a sorted array.
It's similar to binary search but uses Fibonacci numbers instead of halving intervals, which makes it ideal for sequential access systems (like tapes or large memory arrays).

It shines where comparison count matters or when random access is expensive.

### What Problem Are We Solving?

You want to search for an element in a sorted array efficiently,
but instead of halving the interval (as in binary search),
you want to use Fibonacci numbers to decide where to look —
minimizing comparisons and taking advantage of arithmetic-friendly jumps.

#### Example

Let's search for x = 55 in:

```
arr = [10, 22, 35, 40, 45, 50, 80, 82, 85, 90, 100]
```

1. Find smallest Fibonacci number ≥ length (11) → F(7) = 13
2. Use Fibonacci offsets to decide index jumps.
3. Check mid using fibM2 = 5 (F(5)=5 → index 4): arr[4] = 45 < 55
4. Move window and repeat until found or narrowed.

### How Does It Work (Plain Language)?

Instead of splitting in half like binary search,
it splits using ratios of Fibonacci numbers, keeping subarray sizes close to golden ratio.

This approach reduces comparisons and works especially well when array size is known
and access cost is linear or limited.

Think of it as a mathematically balanced jump search guided by Fibonacci spacing.

#### Step-by-Step Process

| Step | Description                                                      |
| ---- | ---------------------------------------------------------------- |
| 1    | Compute smallest Fibonacci number Fm ≥ n                         |
| 2    | Set offsets: Fm1 = Fm-1, Fm2 = Fm-2                              |
| 3    | Compare target with arr[offset + Fm2]                            |
| 4    | If smaller, search left subarray (shift Fm1, Fm2)                |
| 5    | If larger, search right subarray (update offset, shift Fm1, Fm2) |
| 6    | If equal, return index                                           |
| 7    | Continue until Fm1 = 1                                           |

### Tiny Code (Easy Versions)

#### Python

```python
def fibonacci_search(arr, x):
    n = len(arr)
    
    # Initialize fibonacci numbers
    fibMMm2 = 0  # F(m-2)
    fibMMm1 = 1  # F(m-1)
    fibM = fibMMm2 + fibMMm1  # F(m)

    # Find smallest Fibonacci >= n
    while fibM < n:
        fibMMm2 = fibMMm1
        fibMMm1 = fibM
        fibM = fibMMm2 + fibMMm1

    offset = -1

    while fibM > 1:
        i = min(offset + fibMMm2, n - 1)

        if arr[i] < x:
            fibM = fibMMm1
            fibMMm1 = fibMMm2
            fibMMm2 = fibM - fibMMm1
            offset = i
        elif arr[i] > x:
            fibM = fibMMm2
            fibMMm1 -= fibMMm2
            fibMMm2 = fibM - fibMMm1
        else:
            return i

    if fibMMm1 and offset + 1 < n and arr[offset + 1] == x:
        return offset + 1

    return -1

arr = [10, 22, 35, 40, 45, 50, 80, 82, 85, 90, 100]
print("Found at index:", fibonacci_search(arr, 55))  # Output: -1
print("Found at index:", fibonacci_search(arr, 85))  # Output: 8
```

Output:

```
Found at index: -1
Found at index: 8
```

#### C

```c
#include <stdio.h>

int min(int a, int b) { return (a < b) ? a : b; }

int fibonacci_search(int arr[], int n, int x) {
    int fibMMm2 = 0;   // F(m-2)
    int fibMMm1 = 1;   // F(m-1)
    int fibM = fibMMm2 + fibMMm1;  // F(m)

    while (fibM < n) {
        fibMMm2 = fibMMm1;
        fibMMm1 = fibM;
        fibM = fibMMm2 + fibMMm1;
    }

    int offset = -1;

    while (fibM > 1) {
        int i = min(offset + fibMMm2, n - 1);

        if (arr[i] < x) {
            fibM = fibMMm1;
            fibMMm1 = fibMMm2;
            fibMMm2 = fibM - fibMMm1;
            offset = i;
        } else if (arr[i] > x) {
            fibM = fibMMm2;
            fibMMm1 -= fibMMm2;
            fibMMm2 = fibM - fibMMm1;
        } else {
            return i;
        }
    }

    if (fibMMm1 && offset + 1 < n && arr[offset + 1] == x)
        return offset + 1;

    return -1;
}

int main(void) {
    int arr[] = {10, 22, 35, 40, 45, 50, 80, 82, 85, 90, 100};
    int n = sizeof(arr)/sizeof(arr[0]);
    int x = 85;
    int idx = fibonacci_search(arr, n, x);
    if (idx != -1)
        printf("Found at index: %d\n", idx);
    else
        printf("Not found\n");
}
```

Output:

```
Found at index: 8
```

### Why It Matters

- Uses Fibonacci numbers to reduce comparisons
- Efficient for sorted arrays with sequential access
- Avoids division (only addition/subtraction)
- Inspired by golden ratio search (optimal probing)
- Excellent teaching tool for divide-and-conquer logic

### A Gentle Proof (Why It Works)

The Fibonacci split maintains nearly golden ratio balance.  
At each step, one subproblem has size ≈ $\frac{1}{\phi}$ of the previous.

So total steps ≈ number of Fibonacci numbers $\le n$,  
which grows as $O(\log_\phi n) \approx O(\log n)$.

Thus, the time complexity is the same as binary search,  
but with fewer comparisons and more efficient arithmetic.


### Try It Yourself

1. Search 45 in `[10, 22, 35, 40, 45, 50, 80, 82, 85, 90, 100]`
2. Search 100 (last element)
3. Search 10 (first element)
4. Search value not in array
5. Count comparisons made
6. Compare with binary search
7. Try on length = Fibonacci number (e.g. 13)
8. Visualize index jumps
9. Modify to print intervals
10. Apply to sorted strings (lex order)

### Test Cases

| Array                               | Target | Output | Notes     |
| ----------------------------------- | ------ | ------ | --------- |
| [10,22,35,40,45,50,80,82,85,90,100] | 85     | 8      | Found     |
| [10,22,35,40,45]                    | 22     | 1      | Found     |
| [1,2,3,5,8,13,21]                   | 21     | 6      | Found     |
| [2,4,6,8,10]                        | 7      | -1     | Not found |

### Complexity

| Aspect                | Value               |
| --------------------- | ------------------- |
| Time                  | O(log n)            |
| Space                 | O(1)                |
| Comparisons           | ≤ log₍φ₎(n)         |
| Access type           | Sequential-friendly |
| Requires sorted input | Yes                 |

Fibonacci Search, a golden search for discrete worlds, where each step follows nature's rhythm.

### 170 Jump + Binary Hybrid

Jump + Binary Hybrid Search blends the best of two worlds, Jump Search for fast skipping and Binary Search for precise refinement.
It's perfect when your data is sorted, and you want a balance between linear jumps and logarithmic probing within small subranges.

### What Problem Are We Solving?

Binary search is powerful but needs random access (you can jump anywhere).
Jump search works well for sequential data (like linked blocks or caches) but may overshoot.

This hybrid combines them:

1. Jump ahead in fixed steps to find the block.
2. Once you know the target range, switch to binary search inside it.

It's a practical approach for sorted datasets with limited random access (like disk blocks or database pages).

#### Example

Search for `45` in `[10, 22, 35, 40, 45, 50, 80, 82, 85, 90, 100]`.

Step 1: Jump by block size √n = 3

- Check `arr[2] = 35` → 35 < 45
- Check `arr[5] = 50` → 50 > 45

Now we know target is in `[35, 40, 45, 50)` → indices [3..5)

Step 2: Binary search within block [3..5)

- Mid = 4 → arr[4] = 45 ✅ Found!

### How Does It Work (Plain Language)?

1. Choose block size $m = \sqrt{n}$.  
2. Jump ahead by $m$ until you pass or reach the target.  
3. Once in the block, switch to binary search inside.


Jumping narrows the search zone quickly,
Binary search finishes it cleanly, fewer comparisons than either alone.

#### Step-by-Step Process

| Step | Description                                         |
| ---- | --------------------------------------------------- |
| 1    | Compute block size $m = \lfloor \sqrt{n} \rfloor$   |
| 2    | Jump by $m$ until `arr[j] ≥ target` or end          |
| 3    | Determine block $[j - m, j)$                        |
| 4    | Run binary search inside that block                 |
| 5    | Return index or -1 if not found                     |


### Tiny Code (Easy Versions)

#### Python

```python
import math

def jump_binary_search(arr, x):
    n = len(arr)
    step = int(math.sqrt(n))
    prev = 0

    # Jump phase
    while prev < n and arr[min(step, n) - 1] < x:
        prev = step
        step += int(math.sqrt(n))
        if prev >= n:
            return -1

    # Binary search in block
    low, high = prev, min(step, n) - 1
    while low <= high:
        mid = (low + high) // 2
        if arr[mid] == x:
            return mid
        elif arr[mid] < x:
            low = mid + 1
        else:
            high = mid - 1

    return -1

arr = [10, 22, 35, 40, 45, 50, 80, 82, 85, 90, 100]
print("Found at index:", jump_binary_search(arr, 45))
```

Output:

```
Found at index: 4
```

#### C

```c
#include <stdio.h>
#include <math.h>

int jump_binary_search(int arr[], int n, int x) {
    int step = (int)sqrt(n);
    int prev = 0;

    // Jump phase
    while (prev < n && arr[(step < n ? step : n) - 1] < x) {
        prev = step;
        step += (int)sqrt(n);
        if (prev >= n)
            return -1;
    }

    // Binary search in block
    int low = prev;
    int high = (step < n ? step : n) - 1;

    while (low <= high) {
        int mid = low + (high - low) / 2;
        if (arr[mid] == x)
            return mid;
        else if (arr[mid] < x)
            low = mid + 1;
        else
            high = mid - 1;
    }

    return -1;
}

int main(void) {
    int arr[] = {10, 22, 35, 40, 45, 50, 80, 82, 85, 90, 100};
    int n = sizeof(arr) / sizeof(arr[0]);
    int idx = jump_binary_search(arr, n, 45);
    if (idx != -1)
        printf("Found at index: %d\n", idx);
    else
        printf("Not found\n");
}
```

Output:

```
Found at index: 4
```

### Why It Matters

- Combines fast skipping (jump) and efficient narrowing (binary)
- Works well on sorted lists with slow access
- Reduces comparisons vs pure jump or linear
- Great for block-based storage and database indexing
- Demonstrates hybrid thinking in algorithm design

### A Gentle Proof (Why It Works)

Jump phase: $O(\sqrt{n})$ steps  
Binary phase: $O(\log \sqrt{n}) = O(\log n)$  

Total:
$$
T(n) = O(\sqrt{n}) + O(\log n)
$$

For large $n$, dominated by $O(\sqrt{n})$, but faster in practice.


### Try It Yourself

1. Search `80` in `[10,22,35,40,45,50,80,82,85,90,100]`
2. Try `10` (first element)
3. Try `100` (last element)
4. Try value not in array
5. Compare comparisons with binary search
6. Change block size (try 2√n or n/4)
7. Print jumps and block
8. Run on array of length 100
9. Combine with exponential block discovery
10. Extend for descending arrays

### Test Cases

| Array                               | Target | Output | Notes           |
| ----------------------------------- | ------ | ------ | --------------- |
| [10,22,35,40,45,50,80,82,85,90,100] | 45     | 4      | Found           |
| [10,22,35,40,45]                    | 10     | 0      | Found at start  |
| [10,22,35,40,45]                    | 100    | -1     | Not found       |
| [1,3,5,7,9,11,13]                   | 7      | 3      | Found in middle |

### Complexity

| Aspect          | Value         |
| --------------- | ------------- |
| Time            | O(√n + log n) |
| Space           | O(1)          |
| Requires sorted | Yes           |
| Stable          | Yes           |
| Adaptive        | No            |

Jump + Binary Hybrid, leap with purpose, then zero in. It's how explorers search with both speed and focus.

# Section 18. Selection Algorithms 

### 171 Quickselect

Quickselect is a selection algorithm to find the k-th smallest element in an unsorted array —
faster on average than sorting the entire array.

It's based on the same partitioning idea as Quicksort,
but only recurses into the side that contains the desired element.

Average time complexity: O(n)
Worst-case (rare): O(n²)

### What Problem Are We Solving?

Suppose you have an unsorted list and you want:

- The median,
- The k-th smallest, or
- The k-th largest element,

You don't need to fully sort, you just need one order statistic.

Quickselect solves this by partitioning the array and narrowing focus
to the relevant half only.

#### Example

Find 4th smallest element in:
`[7, 2, 1, 6, 8, 5, 3, 4]`

1. Choose pivot (e.g. 4).
2. Partition → `[2, 1, 3] [4] [7, 6, 8, 5]`
3. Pivot position = 3 (0-based)
4. k = 4 → pivot index 3 matches → 4 is 4th smallest ✅

No need to sort the rest!

### How Does It Work (Plain Language)?

Quickselect picks a pivot, partitions the list into less-than and greater-than parts,
and decides which side to recurse into based on the pivot's index vs. target `k`.

It's a divide and conquer search on positions, not order.

#### Step-by-Step Process

| Step | Description                         |
| ---- | ----------------------------------- |
| 1    | Pick pivot (random or last element) |
| 2    | Partition array around pivot        |
| 3    | Get pivot index `p`                 |
| 4    | If `p == k`, return element         |
| 5    | If `p > k`, search left             |
| 6    | If `p < k`, search right (adjust k) |

### Tiny Code (Easy Versions)

#### Python

```python
import random

def partition(arr, low, high):
    pivot = arr[high]
    i = low
    for j in range(low, high):
        if arr[j] < pivot:
            arr[i], arr[j] = arr[j], arr[i]
            i += 1
    arr[i], arr[high] = arr[high], arr[i]
    return i

def quickselect(arr, k):
    low, high = 0, len(arr) - 1
    while low <= high:
        pivot_index = partition(arr, low, high)
        if pivot_index == k:
            return arr[pivot_index]
        elif pivot_index > k:
            high = pivot_index - 1
        else:
            low = pivot_index + 1

arr = [7, 2, 1, 6, 8, 5, 3, 4]
k = 3  # 0-based index: 4th smallest
print("4th smallest:", quickselect(arr, k))
```

Output:

```
4th smallest: 4
```

#### C

```c
#include <stdio.h>

void swap(int *a, int *b) {
    int t = *a;
    *a = *b;
    *b = t;
}

int partition(int arr[], int low, int high) {
    int pivot = arr[high];
    int i = low;
    for (int j = low; j < high; j++) {
        if (arr[j] < pivot) {
            swap(&arr[i], &arr[j]);
            i++;
        }
    }
    swap(&arr[i], &arr[high]);
    return i;
}

int quickselect(int arr[], int low, int high, int k) {
    if (low <= high) {
        int pi = partition(arr, low, high);
        if (pi == k)
            return arr[pi];
        else if (pi > k)
            return quickselect(arr, low, pi - 1, k);
        else
            return quickselect(arr, pi + 1, high, k);
    }
    return -1;
}

int main(void) {
    int arr[] = {7, 2, 1, 6, 8, 5, 3, 4};
    int n = sizeof(arr) / sizeof(arr[0]);
    int k = 3;  // 4th smallest (0-based)
    printf("4th smallest: %d\n", quickselect(arr, 0, n - 1, k));
}
```

Output:

```
4th smallest: 4
```

### Why It Matters

- Find median in linear time (expected)
- Avoid sorting when you only need one element
- Basis for algorithms like Median of Medians, BFPRT
- Common in order statistics, quantiles, top-k problems
- Used in libraries (e.g. `nth_element` in C++)

### A Gentle Proof (Why It Works)

Each partition reduces problem size by eliminating one side.
Average split ≈ half → O(n) expected comparisons.
Worst case (bad pivot) → O(n²),
but with randomized pivot, very unlikely.

Expected time:
$$
T(n) = T(n/2) + O(n) \Rightarrow O(n)
$$

### Try It Yourself

1. Find 1st smallest (min)
2. Find last (max)
3. Find median (`k = n/2`)
4. Add random pivoting
5. Count comparisons per iteration
6. Modify for k-th largest (`n-k`)
7. Compare runtime with full sort
8. Visualize partition steps
9. Test on repeated elements
10. Combine with deterministic pivot

### Test Cases

| Array                  | k (0-based) | Output | Notes          |
| ---------------------- | ----------- | ------ | -------------- |
| [7,2,1,6,8,5,3,4]      | 3           | 4      | 4th smallest   |
| [3,1,2]                | 1           | 2      | median         |
| [10,80,30,90,40,50,70] | 4           | 70     | middle element |
| [5,5,5,5]              | 2           | 5      | duplicates     |

### Complexity

| Aspect         | Value       |
| -------------- | ----------- |
| Time (Average) | O(n)        |
| Time (Worst)   | O(n²)       |
| Space          | O(1)        |
| Stable         | No          |
| In-place       | Yes         |
| Randomized     | Recommended |

Quickselect, the surgical strike of sorting: find exactly what you need, and ignore the rest.

### 172 Median of Medians

Median of Medians is a deterministic selection algorithm that guarantees O(n) worst-case time for finding the k-th smallest element.
It improves on Quickselect, which can degrade to (O(n^2)) in unlucky cases, by carefully choosing a good pivot every time.

It's a cornerstone of theoretical computer science, balancing speed and worst-case safety.

### What Problem Are We Solving?

In Quickselect, a bad pivot can lead to unbalanced partitions (like always picking smallest/largest).
Median of Medians fixes this by ensuring the pivot is "good enough" —
always splitting the array so that each side has at least a constant fraction of elements.

Goal:
Find the k-th smallest element deterministically in O(n), no randomness, no risk.

#### Example

Find 5th smallest in `[12, 3, 5, 7, 4, 19, 26, 23, 8, 15]`

1. Split into groups of 5:
   `[12, 3, 5, 7, 4]`, `[19, 26, 23, 8, 15]`
2. Sort each group:
   `[3,4,5,7,12]`, `[8,15,19,23,26]`
3. Take medians: `[5, 19]`
4. Find median of medians: `19`
5. Partition array around `19`
6. Recurse on appropriate side until k-th found

Pivot `19` ensures balanced split, leading to linear runtime.

### How Does It Work (Plain Language)?

1. Break array into groups of 5
2. Sort each small group (cheap)
3. Collect all medians of groups
4. Recursively find median of those medians → good pivot
5. Partition around pivot
6. Recurse into half that contains the k-th element

Each level discards a constant fraction → (O(n)) total work.

#### Step-by-Step Process

| Step | Description                          |
| ---- | ------------------------------------ |
| 1    | Divide array into chunks of 5        |
| 2    | Sort each chunk                      |
| 3    | Extract medians                      |
| 4    | Find median of medians recursively   |
| 5    | Partition using this pivot           |
| 6    | Recurse into correct side based on k |

### Tiny Code (Easy Versions)

#### Python

```python
def partition(arr, pivot):
    less = [x for x in arr if x < pivot]
    equal = [x for x in arr if x == pivot]
    greater = [x for x in arr if x > pivot]
    return less, equal, greater

def select(arr, k):
    if len(arr) <= 5:
        return sorted(arr)[k]

    # Step 1: group in chunks of 5
    chunks = [arr[i:i+5] for i in range(0, len(arr), 5)]

    # Step 2: find medians
    medians = [sorted(chunk)[len(chunk)//2] for chunk in chunks]

    # Step 3: pivot = median of medians
    pivot = select(medians, len(medians)//2)

    # Step 4: partition
    less, equal, greater = partition(arr, pivot)

    # Step 5: recurse
    if k < len(less):
        return select(less, k)
    elif k < len(less) + len(equal):
        return pivot
    else:
        return select(greater, k - len(less) - len(equal))

arr = [12, 3, 5, 7, 4, 19, 26, 23, 8, 15]
k = 4  # 0-based: 5th smallest
print("5th smallest:", select(arr, k))
```

Output:

```
5th smallest: 8
```

#### C (Simplified Version)

*(Pseudocode-like clarity for readability)*

```c
#include <stdio.h>
#include <stdlib.h>

int cmp(const void* a, const void* b) {
    return (*(int*)a - *(int*)b);
}

int median_of_medians(int arr[], int n, int k);

int select_group_median(int arr[], int n) {
    qsort(arr, n, sizeof(int), cmp);
    return arr[n/2];
}

int median_of_medians(int arr[], int n, int k) {
    if (n <= 5) {
        qsort(arr, n, sizeof(int), cmp);
        return arr[k];
    }

    int groups = (n + 4) / 5;
    int medians[groups];
    for (int i = 0; i < groups; i++) {
        int size = (i*5 + 5 <= n) ? 5 : n - i*5;
        medians[i] = select_group_median(arr + i*5, size);
    }

    int pivot = median_of_medians(medians, groups, groups/2);

    // Partition
    int less[n], greater[n], l = 0, g = 0, equal = 0;
    for (int i = 0; i < n; i++) {
        if (arr[i] < pivot) less[l++] = arr[i];
        else if (arr[i] > pivot) greater[g++] = arr[i];
        else equal++;
    }

    if (k < l)
        return median_of_medians(less, l, k);
    else if (k < l + equal)
        return pivot;
    else
        return median_of_medians(greater, g, k - l - equal);
}

int main(void) {
    int arr[] = {12, 3, 5, 7, 4, 19, 26, 23, 8, 15};
    int n = sizeof(arr)/sizeof(arr[0]);
    printf("5th smallest: %d\n", median_of_medians(arr, n, 4));
}
```

Output:

```
5th smallest: 8
```

### Why It Matters

- Guaranteed O(n) even in worst case
- No bad pivots → stable performance
- Basis for BFPRT algorithm
- Used in theoretical guarantees for real systems
- Key for deterministic selection, safe quantile computations

### A Gentle Proof (Why It Works)

Each pivot ensures at least 30% of elements are discarded each recursion (proof via grouping).

Recurrence:
$$
T(n) = T(n/5) + T(7n/10) + O(n) \Rightarrow O(n)
$$

Thus, always linear time, even worst case.

### Try It Yourself

1. Find median of `[5, 2, 1, 3, 4]`
2. Test with duplicates
3. Compare with Quickselect runtime
4. Count recursive calls
5. Change group size to 3 or 7
6. Visualize grouping steps
7. Print pivot each round
8. Apply to large random list
9. Benchmark vs sorting
10. Implement as pivot strategy for Quickselect

### Test Cases

| Array                      | k | Output | Notes        |
| -------------------------- | - | ------ | ------------ |
| [12,3,5,7,4,19,26,23,8,15] | 4 | 8      | 5th smallest |
| [5,2,1,3,4]                | 2 | 3      | Median       |
| [7,7,7,7]                  | 2 | 7      | Duplicates   |
| [10,9,8,7,6,5]             | 0 | 5      | Min          |

### Complexity

| Aspect        | Value |
| ------------- | ----- |
| Time (Worst)  | O(n)  |
| Time (Avg)    | O(n)  |
| Space         | O(n)  |
| Stable        | No    |
| Deterministic | Yes   |

Median of Medians, a balanced thinker in the world of selection: slow and steady, but always linear.

### 173 Randomized Select

Randomized Select is a probabilistic version of Quickselect, where the pivot is chosen randomly to avoid worst-case behavior.
This small twist makes the algorithm's expected time O(n), even though the worst case remains (O(n^2)).
In practice, it's fast, simple, and robust, a true workhorse for order statistics.

### What Problem Are We Solving?

You need the k-th smallest element in an unsorted list.
Quickselect works well, but choosing the first or last element as pivot can cause bad splits.

Randomized Select improves this by picking a random pivot, making bad luck rare and performance stable.

#### Example

Find 4th smallest in `[7, 2, 1, 6, 8, 5, 3, 4]`

1. Pick random pivot (say `5`)
2. Partition → `[2,1,3,4] [5] [7,6,8]`
3. Pivot index = 4 → 4 > 3, so recurse on left `[2,1,3,4]`
4. Pick random pivot again (say `3`)
5. Partition → `[2,1] [3] [4]`
6. Index 2 = k=3 → found 4th smallest = 4 ✅

### How Does It Work (Plain Language)?

It's Quickselect with random pivoting.
At each step:

- Pick a random element as pivot.
- Partition around pivot.
- Only recurse into one side (where k lies).

This randomness ensures average-case balance, even on adversarial inputs.

#### Step-by-Step Process

| Step | Description                          |
| ---- | ------------------------------------ |
| 1    | Pick random pivot index              |
| 2    | Partition array around pivot         |
| 3    | Get pivot index `p`                  |
| 4    | If `p == k`, return element          |
| 5    | If `p > k`, recurse left             |
| 6    | If `p < k`, recurse right (adjust k) |

### Tiny Code (Easy Versions)

#### Python

```python
import random

def partition(arr, low, high):
    pivot = arr[high]
    i = low
    for j in range(low, high):
        if arr[j] < pivot:
            arr[i], arr[j] = arr[j], arr[i]
            i += 1
    arr[i], arr[high] = arr[high], arr[i]
    return i

def randomized_select(arr, low, high, k):
    if low == high:
        return arr[low]

    pivot_index = random.randint(low, high)
    arr[pivot_index], arr[high] = arr[high], arr[pivot_index]

    p = partition(arr, low, high)

    if p == k:
        return arr[p]
    elif p > k:
        return randomized_select(arr, low, p - 1, k)
    else:
        return randomized_select(arr, p + 1, high, k)

arr = [7, 2, 1, 6, 8, 5, 3, 4]
k = 3  # 4th smallest
print("4th smallest:", randomized_select(arr, 0, len(arr)-1, k))
```

Output:

```
4th smallest: 4
```

#### C

```c
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void swap(int *a, int *b) {
    int t = *a; *a = *b; *b = t;
}

int partition(int arr[], int low, int high) {
    int pivot = arr[high];
    int i = low;
    for (int j = low; j < high; j++) {
        if (arr[j] < pivot) {
            swap(&arr[i], &arr[j]);
            i++;
        }
    }
    swap(&arr[i], &arr[high]);
    return i;
}

int randomized_select(int arr[], int low, int high, int k) {
    if (low == high) return arr[low];

    int pivot_index = low + rand() % (high - low + 1);
    swap(&arr[pivot_index], &arr[high]);
    int p = partition(arr, low, high);

    if (p == k) return arr[p];
    else if (p > k) return randomized_select(arr, low, p - 1, k);
    else return randomized_select(arr, p + 1, high, k);
}

int main(void) {
    srand(time(NULL));
    int arr[] = {7, 2, 1, 6, 8, 5, 3, 4};
    int n = sizeof(arr)/sizeof(arr[0]);
    int k = 3;
    printf("4th smallest: %d\n", randomized_select(arr, 0, n - 1, k));
}
```

Output:

```
4th smallest: 4
```

### Why It Matters

- Expected O(n) time, simple, and practical
- Avoids worst-case trap of fixed-pivot Quickselect
- Great for top-k queries, quantiles, median
- Combines simplicity + randomness = robust performance
- Commonly used in competitive programming and real-world systems

### A Gentle Proof (Why It Works)

Expected recurrence:
$$
T(n) = T(\alpha n) + O(n)
$$
where $\alpha$ is random, expected $\approx \tfrac{1}{2}$  
→ $T(n) = O(n)$

Worst case still $O(n^2)$, but rare.  
Expected comparisons $\approx 2n$.


### Try It Yourself

1. Run multiple times and observe pivot randomness
2. Compare with deterministic Quickselect
3. Count recursive calls
4. Test with sorted input (robust)
5. Test all same elements
6. Change k (first, median, last)
7. Modify to find k-th largest (`n-k-1`)
8. Compare performance with sort()
9. Log pivot indices
10. Measure runtime on 10⁶ elements

### Test Cases

| Array                  | k | Output | Notes              |
| ---------------------- | - | ------ | ------------------ |
| [7,2,1,6,8,5,3,4]      | 3 | 4      | 4th smallest       |
| [10,80,30,90,40,50,70] | 4 | 70     | Works on any order |
| [1,2,3,4,5]            | 0 | 1      | Sorted input safe  |
| [5,5,5,5]              | 2 | 5      | Duplicates fine    |

### Complexity

| Aspect          | Value |
| --------------- | ----- |
| Time (Expected) | O(n)  |
| Time (Worst)    | O(n²) |
| Space           | O(1)  |
| Stable          | No    |
| Randomized      | Yes   |
| In-place        | Yes   |

Randomized Select, a game of chance that almost always wins: fast, fair, and beautifully simple.

### 174 Binary Search on Answer

Binary Search on Answer (also called Parametric Search) is a powerful optimization trick used when the search space is monotonic—meaning once a condition becomes true, it stays true (or vice versa). Instead of searching a sorted array, we're searching for the smallest or largest value that satisfies a condition.

### What Problem Are We Solving?

Sometimes you don't have an array to search, but you need to minimize or maximize a numeric answer.
Examples:

- Minimum capacity to transport items in `k` days
- Minimum maximum distance between routers
- Maximum median satisfying a condition

We can't iterate all possibilities efficiently, but we can binary search the answer space.

#### Example

Problem: Given array `[1, 2, 3, 4, 5]`, split into `2` parts, minimize the largest sum among parts.

We can't directly find it, but if we can check whether a candidate value `mid` is valid (can split with sum ≤ mid), we can binary search on `mid`.

| mid | canSplit(nums, 2, mid)        | Result                |
| --- | ----------------------------- | --------------------- |
| 9   | True (splits: [1,2,3,4], [5]) | ok → move left        |
| 7   | False                         | move right            |
| 8   | True                          | ok → final answer = 9 |

✅ Result = 9

### How Does It Work (Plain Language)?

You don't search elements, you search values.
You define a function `can(mid)` that checks if a solution is possible with `mid`.
Then use binary search to narrow down the range until you find the optimal value.

#### Step-by-Step Process

| Step | Description                          |
| ---- | ------------------------------------ |
| 1    | Define the range of answers (lo, hi) |
| 2    | While lo < hi:                       |
|      |  mid = (lo + hi) // 2                |
|      |  If can(mid): hi = mid               |
|      |  Else: lo = mid + 1                  |
| 3    | Return lo as the optimal answer      |

### Tiny Code (Easy Versions)

#### Python

```python
def can_split(nums, k, mid):
    count, current = 1, 0
    for x in nums:
        if current + x > mid:
            count += 1
            current = x
        else:
            current += x
    return count <= k

def binary_search_answer(nums, k):
    lo, hi = max(nums), sum(nums)
    while lo < hi:
        mid = (lo + hi) // 2
        if can_split(nums, k, mid):
            hi = mid
        else:
            lo = mid + 1
    return lo

nums = [1, 2, 3, 4, 5]
k = 2
print("Minimum largest sum:", binary_search_answer(nums, k))
```

Output:

```
Minimum largest sum: 9
```

#### C

```c
#include <stdio.h>

int can_split(int arr[], int n, int k, int mid) {
    int count = 1, sum = 0;
    for (int i = 0; i < n; i++) {
        if (arr[i] > mid) return 0;
        if (sum + arr[i] > mid) {
            count++;
            sum = arr[i];
        } else {
            sum += arr[i];
        }
    }
    return count <= k;
}

int binary_search_answer(int arr[], int n, int k) {
    int lo = arr[0], hi = 0;
    for (int i = 0; i < n; i++) {
        if (arr[i] > lo) lo = arr[i];
        hi += arr[i];
    }
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (can_split(arr, n, k, mid))
            hi = mid;
        else
            lo = mid + 1;
    }
    return lo;
}

int main(void) {
    int arr[] = {1, 2, 3, 4, 5};
    int n = sizeof(arr)/sizeof(arr[0]);
    int k = 2;
    printf("Minimum largest sum: %d\n", binary_search_answer(arr, n, k));
}
```

Output:

```
Minimum largest sum: 9
```

### Why It Matters

- Solves optimization problems without brute force
- Turns decision problems into search problems
- A universal pattern: works for capacity, distance, time, etc.
- Common in LeetCode, interviews, and competitive programming

### A Gentle Proof (Why It Works)

If a function `f(x)` is monotonic (true after a point or false after a point), binary search can find the threshold.
Formally:

If `f(lo) = false`, `f(hi) = true`,
and `f(x)` is monotonic,
then binary search converges to smallest x such that f(x) = true.

### Try It Yourself

1. Find smallest capacity to ship packages in `d` days
2. Find smallest max page load per student (book allocation)
3. Find largest minimum distance between routers
4. Find smallest time to paint all boards
5. Find minimum speed to reach on time
6. Define a monotonic function `can(x)` and apply search
7. Experiment with `float` range and tolerance
8. Try max instead of min (reverse condition)
9. Count binary search steps for each case
10. Compare with brute force

### Test Cases

| Problem          | Input              | Output | Explanation     |
| ---------------- | ------------------ | ------ | --------------- |
| Split array      | [1,2,3,4,5], k=2   | 9      | [1,2,3,4],[5]   |
| Book allocation  | [10,20,30,40], k=2 | 60     | [10,20,30],[40] |
| Router placement | [1,2,8,12], k=3    | 5      | Place at 1,6,12 |

### Complexity

| Aspect                | Value                        |
| --------------------- | ---------------------------- |
| Time                  | O(n log(max - min))          |
| Space                 | O(1)                         |
| Monotonicity Required | Yes                          |
| Type                  | Decision-based binary search |

Binary Search on Answer, when you can't sort the data, sort the solution space.

### 175 Order Statistics Tree

An Order Statistics Tree is a special kind of augmented binary search tree (BST) that supports two powerful operations efficiently:

1. Select(k): find the k-th smallest element.
2. Rank(x): find the position (rank) of element `x`.

It's a classic data structure where each node stores subtree size, allowing order-based queries in O(log n) time.

### What Problem Are We Solving?

Sometimes you don't just want to search by key, you want to search by order.
For example:

- "What's the 5th smallest element?"
- "What rank is 37 in the tree?"
- "How many numbers ≤ 50 are there?"

An order statistics tree gives you both key-based and rank-based access in one structure.

#### Example

Suppose you insert `[20, 15, 25, 10, 18, 22, 30]`.

Each node stores `size` (the number of nodes in its subtree).

```
        20(size=7)
       /          \
  15(3)          25(3)
  /   \          /   \
10(1) 18(1)   22(1) 30(1)
```

Select(4) → 20 (the 4th smallest)
Rank(22) → 6 (22 is the 6th smallest)

### How Does It Work (Plain Language)?

Every node tracks how many nodes exist in its subtree (left + right + itself).
When you traverse:

- To select k-th smallest, compare `k` with size of left subtree.
- To find rank of x, accumulate sizes while traversing down.

#### Select(k) Pseudocode

```
select(node, k):
    left_size = size(node.left)
    if k == left_size + 1: return node.key
    if k <= left_size: return select(node.left, k)
    else: return select(node.right, k - left_size - 1)
```

#### Rank(x) Pseudocode

```
rank(node, x):
    if node == NULL: return 0
    if x < node.key: return rank(node.left, x)
    if x == node.key: return size(node.left) + 1
    else: return size(node.left) + 1 + rank(node.right, x)
```

### Tiny Code (Easy Versions)

#### Python

```python
class Node:
    def __init__(self, key):
        self.key = key
        self.left = None
        self.right = None
        self.size = 1

def update_size(node):
    if node:
        node.size = 1 + (node.left.size if node.left else 0) + (node.right.size if node.right else 0)

def insert(node, key):
    if node is None:
        return Node(key)
    if key < node.key:
        node.left = insert(node.left, key)
    else:
        node.right = insert(node.right, key)
    update_size(node)
    return node

def select(node, k):
    left_size = node.left.size if node.left else 0
    if k == left_size + 1:
        return node.key
    elif k <= left_size:
        return select(node.left, k)
    else:
        return select(node.right, k - left_size - 1)

def rank(node, key):
    if node is None:
        return 0
    if key < node.key:
        return rank(node.left, key)
    elif key == node.key:
        return (node.left.size if node.left else 0) + 1
    else:
        left_size = node.left.size if node.left else 0
        return left_size + 1 + rank(node.right, key)

root = None
for x in [20, 15, 25, 10, 18, 22, 30]:
    root = insert(root, x)

print("Select(4):", select(root, 4))  # 20
print("Rank(22):", rank(root, 22))    # 6
```

#### C (Conceptual Skeleton)

```c
typedef struct Node {
    int key;
    int size;
    struct Node *left, *right;
} Node;

int size(Node* n) { return n ? n->size : 0; }

void update_size(Node* n) {
    if (n) n->size = 1 + size(n->left) + size(n->right);
}

Node* new_node(int key) {
    Node* n = malloc(sizeof(Node));
    n->key = key; n->size = 1;
    n->left = n->right = NULL;
    return n;
}

Node* insert(Node* root, int key) {
    if (!root) return new_node(key);
    if (key < root->key) root->left = insert(root->left, key);
    else root->right = insert(root->right, key);
    update_size(root);
    return root;
}

int select_k(Node* root, int k) {
    int left_size = size(root->left);
    if (k == left_size + 1) return root->key;
    else if (k <= left_size) return select_k(root->left, k);
    else return select_k(root->right, k - left_size - 1);
}
```

### Why It Matters

- Useful in rank-based queries, median finding, and order-statistics problems
- Core to balanced trees (AVL, Red-Black, Treaps) with order augmentation
- Enables dynamic median queries and range counting

You can imagine it as a self-updating leaderboard, always knowing who's in position `k`.

### A Gentle Proof (Why It Works)

Because subtree sizes are updated correctly on insertions/deletions,
each traversal can compute ranks or k-th values in O(log n) time (in balanced trees).
If balanced (like in an AVL or RB-tree), operations remain logarithmic.

### Try It Yourself

1. Build an Order Statistics Tree for `[10,20,30,40,50]`.
2. Find `Select(3)` and `Rank(40)`.
3. Insert new elements and recheck ranks.
4. Extend to find median dynamically.
5. Modify to support deletions.
6. Compare with sorting then indexing (O(n log n) vs O(log n)).
7. Try building on top of Red-Black Tree.
8. Use for running percentiles.
9. Explore dynamic segment trees for same queries.
10. Implement `countLessThan(x)` using rank.

### Test Cases

| Query     | Expected Result |
| --------- | --------------- |
| Select(1) | 10              |
| Select(4) | 20              |
| Rank(10)  | 1               |
| Rank(22)  | 6               |
| Rank(30)  | 7               |

### Complexity

| Operation       | Complexity |
| --------------- | ---------- |
| Insert / Delete | O(log n)   |
| Select(k)       | O(log n)   |
| Rank(x)         | O(log n)   |
| Space           | O(n)       |

An Order Statistics Tree blends search and ranking, perfect for problems that need to know *what* and *where* at the same time.

### 176 Tournament Tree Selection

A Tournament Tree is a binary tree structure that simulates a knockout tournament among elements. Each match compares two elements, and the winner moves up. It's an elegant way to find minimum, maximum, or even k-th smallest elements with structured comparisons.

### What Problem Are We Solving?

Finding the minimum or maximum in a list takes O(n).
But if you also want the second smallest, third smallest, or k-th, you'd like to reuse earlier comparisons.
A tournament tree keeps track of all matches, so you don't need to start over.

#### Example

Suppose we have elements: `[4, 7, 2, 9, 5, 1, 8, 6]`.

1. Pair them up: compare (4,7), (2,9), (5,1), (8,6)
2. Winners move up: [4, 2, 1, 6]
3. Next round: (4,2), (1,6) → winners [2, 1]
4. Final match: (2,1) → winner 1

The root of the tree = minimum element (1).

If you store the losing element from each match, you can trace back the second smallest, it must have lost directly to 1.

### How Does It Work (Plain Language)?

Imagine a sports tournament:

- Every player plays one match.
- The winner moves on, loser is eliminated.
- The champion (root) is the smallest element.
- The second smallest is the best among those who lost to the champion.

Each match is one comparison, so total comparisons = n - 1 for the min.
To find second min, check log n losers.

#### Steps

| Step | Description                                                       |
| ---- | ----------------------------------------------------------------- |
| 1    | Build a complete binary tree where each leaf is an element.       |
| 2    | Compare each pair and move winner up.                             |
| 3    | Store "losers" in each node.                                      |
| 4    | The root = min. The second min = min(losers along winner's path). |

### Tiny Code (Easy Versions)

#### Python

```python
def tournament_min(arr):
    matches = []
    tree = [[x] for x in arr]
    while len(tree) > 1:
        next_round = []
        for i in range(0, len(tree), 2):
            if i + 1 == len(tree):
                next_round.append(tree[i])
                continue
            a, b = tree[i][0], tree[i+1][0]
            if a < b:
                next_round.append([a, b])
            else:
                next_round.append([b, a])
        matches = next_round
        tree = next_round
    return tree[0][0]

def find_second_min(arr):
    # Build tournament, keep track of losers
    n = len(arr)
    tree = [[x, []] for x in arr]
    while len(tree) > 1:
        next_round = []
        for i in range(0, len(tree), 2):
            if i + 1 == len(tree):
                next_round.append(tree[i])
                continue
            a, a_losers = tree[i]
            b, b_losers = tree[i+1]
            if a < b:
                next_round.append([a, a_losers + [b]])
            else:
                next_round.append([b, b_losers + [a]])
        tree = next_round
    winner, losers = tree[0]
    return winner, min(losers)

arr = [4, 7, 2, 9, 5, 1, 8, 6]
min_val, second_min = find_second_min(arr)
print("Min:", min_val)
print("Second Min:", second_min)
```

Output:

```
Min: 1
Second Min: 2
```

#### C

```c
#include <stdio.h>
#include <limits.h>

int tournament_min(int arr[], int n) {
    int size = n;
    while (size > 1) {
        for (int i = 0; i < size / 2; i++) {
            arr[i] = (arr[2*i] < arr[2*i + 1]) ? arr[2*i] : arr[2*i + 1];
        }
        size = (size + 1) / 2;
    }
    return arr[0];
}

int main(void) {
    int arr[] = {4, 7, 2, 9, 5, 1, 8, 6};
    int n = sizeof(arr) / sizeof(arr[0]);
    printf("Minimum: %d\n", tournament_min(arr, n));
}
```

### Why It Matters

- Finds minimum in O(n), second minimum in O(n + log n) comparisons
- Reusable for k-th selection if you store all match info
- Forms the backbone of selection networks, parallel sorting, and merge tournaments

### A Gentle Proof (Why It Works)

Each element except the minimum loses exactly once.
The minimum element competes in log₂n matches (height of tree).
So second minimum must be the smallest of log₂n losers, requiring log₂n extra comparisons.

Total = n - 1 + log₂n comparisons, asymptotically optimal.

### Try It Yourself

1. Build a tournament for `[5,3,8,2,9,4]`.
2. Find minimum and second minimum manually.
3. Modify code to find maximum and second maximum.
4. Print tree rounds to visualize matches.
5. Experiment with uneven sizes (non-power-of-2).
6. Try to extend it to third smallest (hint: store paths).
7. Compare with sorting-based approach.
8. Use tournament structure for pairwise elimination problems.
9. Simulate sports bracket winner path.
10. Count comparisons for each step.

### Test Cases

| Input             | Min | Second Min |
| ----------------- | --- | ---------- |
| [4,7,2,9,5,1,8,6] | 1   | 2          |
| [10,3,6,2]        | 2   | 3          |
| [5,4,3,2,1]       | 1   | 2          |

### Complexity

| Operation           | Complexity |
| ------------------- | ---------- |
| Build Tournament    | O(n)       |
| Find Minimum        | O(1)       |
| Find Second Minimum | O(log n)   |
| Space               | O(n)       |

A Tournament Tree turns comparisons into matches, where every element plays once, and the champion reveals not just victory, but the story of every defeat.

### 177 Heap Select (Min-Heap)

Heap Select is a simple, powerful technique for finding the k smallest (or largest) elements in a collection using a heap. It's one of the most practical selection algorithms, trading minimal code for strong efficiency.

### What Problem Are We Solving?

You often don't need a full sort, just the k smallest or k largest items.
Examples:

- Find top 10 scores
- Get smallest 5 distances
- Maintain top-k trending topics

A heap (priority queue) makes this easy, keep a running set of size `k`, pop or push as needed.

#### Example

Find 3 smallest elements in `[7, 2, 9, 1, 5, 4]`.

1. Create max-heap of first `k=3` elements → `[7, 2, 9]` → heap = [9, 2, 7]
2. For each next element:

   * 1 < 9 → pop 9, push 1 → heap = [7, 2, 1]
   * 5 < 7 → pop 7, push 5 → heap = [5, 2, 1]
   * 4 < 5 → pop 5, push 4 → heap = [4, 2, 1]

Result → `[1, 2, 4]` (the 3 smallest)

### How Does It Work (Plain Language)?

You keep a heap of size k:

- For smallest elements → use a max-heap (remove largest if new smaller appears).
- For largest elements → use a min-heap (remove smallest if new larger appears).

This keeps only the top-k interesting values at all times.

#### Step-by-Step Process

| Step | Action                                          |
| ---- | ----------------------------------------------- |
| 1    | Initialize heap with first k elements           |
| 2    | Convert to max-heap (if looking for smallest k) |
| 3    | For each remaining element x:                   |
|      | If x < heap[0] → replace top                    |
| 4    | Result is heap contents (unsorted)              |
| 5    | Sort heap if needed for final output            |

### Tiny Code (Easy Versions)

#### Python

```python
import heapq

def k_smallest(nums, k):
    heap = [-x for x in nums[:k]]
    heapq.heapify(heap)
    for x in nums[k:]:
        if -x > heap[0]:
            heapq.heappop(heap)
            heapq.heappush(heap, -x)
    return sorted([-h for h in heap])

nums = [7, 2, 9, 1, 5, 4]
print("3 smallest:", k_smallest(nums, 3))
```

Output:

```
3 smallest: [1, 2, 4]
```

#### C

```c
#include <stdio.h>
#include <stdlib.h>

void swap(int *a, int *b) { int t = *a; *a = *b; *b = t; }

void heapify(int arr[], int n, int i) {
    int largest = i, l = 2*i+1, r = 2*i+2;
    if (l < n && arr[l] > arr[largest]) largest = l;
    if (r < n && arr[r] > arr[largest]) largest = r;
    if (largest != i) {
        swap(&arr[i], &arr[largest]);
        heapify(arr, n, largest);
    }
}

void build_heap(int arr[], int n) {
    for (int i = n/2 - 1; i >= 0; i--) heapify(arr, n, i);
}

void heap_select(int arr[], int n, int k) {
    int heap[k];
    for (int i = 0; i < k; i++) heap[i] = arr[i];
    build_heap(heap, k);
    for (int i = k; i < n; i++) {
        if (arr[i] < heap[0]) {
            heap[0] = arr[i];
            heapify(heap, k, 0);
        }
    }
    printf("%d smallest elements:\n", k);
    for (int i = 0; i < k; i++) printf("%d ", heap[i]);
}

int main(void) {
    int arr[] = {7, 2, 9, 1, 5, 4};
    int n = 6;
    heap_select(arr, n, 3);
}
```

Output:

```
3 smallest elements:
1 2 4
```

### Why It Matters

- Avoids full sorting (O(n log n))
- Great for streaming data, sliding windows, top-k problems
- Scales well for large `n` and small `k`
- Used in leaderboards, analytics, data pipelines

If you only need *some* order, don't sort it all.

### A Gentle Proof (Why It Works)

- Building heap: O(k)
- For each new element: compare + heapify = O(log k)
- Total: O(k + (n-k) log k) ≈ O(n log k)

For small `k`, this is much faster than sorting.

### Try It Yourself

1. Find 3 largest elements using min-heap
2. Stream numbers from input, maintain smallest 5
3. Track top 10 scores dynamically
4. Compare runtime vs `sorted(nums)[:k]`
5. Try `k = 1` (minimum), `k = n` (full sort)
6. Modify for objects with custom keys (e.g. score, id)
7. Handle duplicates, keep all or unique only
8. Experiment with random arrays of size 1e6
9. Visualize heap evolution per step
10. Combine with binary search to tune thresholds

### Test Cases

| Input         | k | Output  |
| ------------- | - | ------- |
| [7,2,9,1,5,4] | 3 | [1,2,4] |
| [10,8,6,4,2]  | 2 | [2,4]   |
| [1,1,1,1]     | 2 | [1,1]   |

### Complexity

| Operation     | Complexity     |
| ------------- | -------------- |
| Build Heap    | O(k)           |
| Iterate Array | O((n-k) log k) |
| Total Time    | O(n log k)     |
| Space         | O(k)           |

Heap Select is your practical shortcut, sort only what you need, ignore the rest.

### 178 Partial QuickSort

Partial QuickSort is a twist on classic QuickSort, it stops sorting once it has placed the first k elements (or top k) in their correct positions. It's perfect when you need top-k smallest/largest elements but don't need the rest sorted.

Think of it as QuickSort with early stopping, a hybrid between QuickSort and Quickselect.

### What Problem Are We Solving?

Sometimes you need only part of the sorted order:

- "Get top 10 scores"
- "Find smallest k elements"
- "Sort first half"

Fully sorting wastes work. Partial QuickSort skips unnecessary partitions.

#### Example

Array: `[9, 4, 6, 2, 8, 1]`, k = 3

We want smallest 3 elements.
QuickSort picks a pivot, partitions array:

```
Pivot = 6
→ [4, 2, 1] | 6 | [9, 8]
```

Now we know all elements left of pivot (4, 2, 1) are smaller.
Since `left_size == k`, we can stop, `[1,2,4]` are our smallest 3.

### How Does It Work (Plain Language)?

Just like QuickSort, but after partition:

- If pivot index == k → done.
- If pivot index > k → recurse only left.
- If pivot index < k → recurse right partially.

You never sort beyond what's needed.

#### Step-by-Step Process

| Step | Description                             |
| ---- | --------------------------------------- |
| 1    | Choose pivot                            |
| 2    | Partition array                         |
| 3    | If pivot index == k → done              |
| 4    | If pivot index > k → recurse left       |
| 5    | Else recurse right on remaining portion |

### Tiny Code (Easy Versions)

#### Python

```python
def partial_quicksort(arr, low, high, k):
    if low < high:
        p = partition(arr, low, high)
        if p > k:
            partial_quicksort(arr, low, p - 1, k)
        elif p < k:
            partial_quicksort(arr, low, p - 1, k)
            partial_quicksort(arr, p + 1, high, k)

def partition(arr, low, high):
    pivot = arr[high]
    i = low - 1
    for j in range(low, high):
        if arr[j] < pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return i + 1

arr = [9, 4, 6, 2, 8, 1]
k = 3
partial_quicksort(arr, 0, len(arr) - 1, k - 1)
print("Smallest 3 elements:", sorted(arr[:k]))
```

Output:

```
Smallest 3 elements: [1, 2, 4]
```

#### C

```c
#include <stdio.h>

void swap(int *a, int *b) { int t = *a; *a = *b; *b = t; }

int partition(int arr[], int low, int high) {
    int pivot = arr[high];
    int i = low - 1;
    for (int j = low; j < high; j++) {
        if (arr[j] < pivot) {
            i++;
            swap(&arr[i], &arr[j]);
        }
    }
    swap(&arr[i + 1], &arr[high]);
    return i + 1;
}

void partial_quicksort(int arr[], int low, int high, int k) {
    if (low < high) {
        int p = partition(arr, low, high);
        if (p > k)
            partial_quicksort(arr, low, p - 1, k);
        else if (p < k)
            partial_quicksort(arr, p + 1, high, k);
    }
}

int main(void) {
    int arr[] = {9, 4, 6, 2, 8, 1};
    int n = 6, k = 3;
    partial_quicksort(arr, 0, n - 1, k - 1);
    printf("Smallest %d elements:\n", k);
    for (int i = 0; i < k; i++) printf("%d ", arr[i]);
}
```

Output:

```
Smallest 3 elements:
1 2 4
```

### Why It Matters

- Efficient top-k selection when order matters
- Avoids sorting unnecessary portions
- Combines strengths of Quickselect and QuickSort
- Works in-place (no extra memory)

Great for partial sorting like "leaderboards", "top results", or "bounded priority lists".

### A Gentle Proof (Why It Works)

QuickSort partitions data into two halves;
Only parts that could contain the first `k` elements are explored.
Average time complexity becomes O(n) for selection, O(n log k) for partial order.

### Try It Yourself

1. Find smallest 5 numbers in `[10,9,8,7,6,5,4,3,2,1]`
2. Modify to find largest k instead
3. Compare runtime vs full `sort()`
4. Visualize recursion path
5. Track how many elements actually get sorted
6. Try random pivot vs median pivot
7. Test k = 1 (min) and k = n (full sort)
8. Measure comparisons count
9. Try with duplicates
10. Combine with heap for hybrid version

### Test Cases

| Input         | k | Output  |
| ------------- | - | ------- |
| [9,4,6,2,8,1] | 3 | [1,2,4] |
| [5,4,3,2,1]   | 2 | [1,2]   |
| [7,7,7,7]     | 2 | [7,7]   |

### Complexity

| Aspect       | Value |
| ------------ | ----- |
| Average Time | O(n)  |
| Worst Time   | O(n²) |
| Space        | O(1)  |
| Stable       | No    |

Partial QuickSort, fast, focused, and frugal, because sometimes, you only need *a slice* of order, not the whole loaf.

### 179 BFPRT Algorithm (Median of Medians Selection)

The BFPRT algorithm (named after Blum, Floyd, Pratt, Rivest, and Tarjan) is a deterministic linear-time selection algorithm. It finds the k-th smallest element in an unsorted array, guaranteeing O(n) worst-case time, a mathematically elegant and exact alternative to randomized quickselect.

### What Problem Are We Solving?

You want to find the k-th smallest element, like the median, but you don't want to gamble on random pivots (which could hit worst-case O(n²)).
BFPRT chooses pivots so well that it always guarantees O(n).

This makes it ideal for systems where deterministic behavior matters, like embedded systems, compilers, and real-time applications.

#### Example

Find the median (k=5) of `[9, 4, 7, 3, 6, 1, 8, 2, 5, 10]`

1. Divide into groups of 5:
   `[9,4,7,3,6]`, `[1,8,2,5,10]`
2. Find median of each group:
   → `[6, 5]`
3. Find median of medians:
   → median of `[6,5]` is 5.5 ≈ 5
4. Partition around pivot 5
   → `[4,3,1,2,5] | 5 | [9,7,8,6,10]`
5. Position of 5 = 5 → done.

✅ The 5th smallest = 5

### How Does It Work (Plain Language)?

It's Quickselect with a smarter pivot:

- Divide the array into groups of 5.
- Find median of each group.
- Recursively find median of these medians.
- Use that as pivot → partition → recurse on the correct side.

This ensures the pivot is *always good enough* to split the array reasonably, keeping recursion balanced.

#### Step-by-Step Summary

| Step | Description                              |
| ---- | ---------------------------------------- |
| 1    | Split array into groups of 5             |
| 2    | Sort each group and take its median      |
| 3    | Recursively find median of medians       |
| 4    | Partition around pivot                   |
| 5    | Recurse on side containing k-th smallest |

### Tiny Code (Easy Versions)

#### Python

```python
def partition(arr, pivot):
    less, equal, greater = [], [], []
    for x in arr:
        if x < pivot: less.append(x)
        elif x > pivot: greater.append(x)
        else: equal.append(x)
    return less, equal, greater

def select(arr, k):
    if len(arr) <= 5:
        return sorted(arr)[k]
    
    # Step 1: Divide into groups of 5
    groups = [arr[i:i+5] for i in range(0, len(arr), 5)]
    
    # Step 2: Find medians
    medians = [sorted(g)[len(g)//2] for g in groups]
    
    # Step 3: Median of medians as pivot
    pivot = select(medians, len(medians)//2)
    
    # Step 4: Partition
    less, equal, greater = partition(arr, pivot)
    
    # Step 5: Recurse
    if k < len(less):
        return select(less, k)
    elif k < len(less) + len(equal):
        return pivot
    else:
        return select(greater, k - len(less) - len(equal))

arr = [9,4,7,3,6,1,8,2,5,10]
k = 4  # 0-based index → 5th smallest
print("5th smallest:", select(arr, k))
```

Output:

```
5th smallest: 5
```

#### C (Conceptual Skeleton)

```c
#include <stdio.h>
#include <stdlib.h>

int cmp(const void *a, const void *b) { return (*(int*)a - *(int*)b); }

int median_of_medians(int arr[], int n);

int select_kth(int arr[], int n, int k) {
    if (n <= 5) {
        qsort(arr, n, sizeof(int), cmp);
        return arr[k];
    }

    int groups = (n + 4) / 5;
    int *medians = malloc(groups * sizeof(int));
    for (int i = 0; i < groups; i++) {
        int start = i * 5;
        int end = (start + 5 < n) ? start + 5 : n;
        qsort(arr + start, end - start, sizeof(int), cmp);
        medians[i] = arr[start + (end - start) / 2];
    }

    int pivot = median_of_medians(medians, groups);
    free(medians);

    int *left = malloc(n * sizeof(int));
    int *right = malloc(n * sizeof(int));
    int l = 0, r = 0, equal = 0;
    for (int i = 0; i < n; i++) {
        if (arr[i] < pivot) left[l++] = arr[i];
        else if (arr[i] > pivot) right[r++] = arr[i];
        else equal++;
    }

    if (k < l) return select_kth(left, l, k);
    else if (k < l + equal) return pivot;
    else return select_kth(right, r, k - l - equal);
}

int median_of_medians(int arr[], int n) {
    return select_kth(arr, n, n / 2);
}
```

### Why It Matters

- Deterministic O(n), no randomness or bad pivots
- Used in theoretical CS, worst-case analysis, exact solvers
- Foundation for deterministic selection, median-finding, linear-time sorting bounds
- Core to intro algorithms theory (CLRS Chapter 9)

### A Gentle Proof (Why It Works)

- Each group of 5 → median is ≥ 3 elements in group (2 below, 2 above)
- At least half of medians ≥ pivot → pivot ≥ 30% of elements
- At least half ≤ pivot → pivot ≤ 70% of elements
- So pivot always splits array 30–70, guaranteeing T(n) = T(n/5) + T(7n/10) + O(n) = O(n)

No chance of quadratic blowup.

### Try It Yourself

1. Find median of `[5,3,2,8,1,9,7,6,4]`
2. Trace pivot selection tree
3. Compare with random quickselect pivots
4. Measure time for n = 1e6
5. Try with duplicates
6. Try k = 0, k = n-1 (min/max)
7. Modify group size (e.g. 3 or 7), compare performance
8. Verify recursion depth
9. Use for percentile queries
10. Implement streaming median with repeated selection

### Test Cases

| Input                  | k | Output | Meaning      |
| ---------------------- | - | ------ | ------------ |
| [9,4,7,3,6,1,8,2,5,10] | 4 | 5      | 5th smallest |
| [1,2,3,4,5]            | 2 | 3      | middle       |
| [10,9,8,7,6]           | 0 | 6      | smallest     |

### Complexity

| Aspect        | Value                           |
| ------------- | ------------------------------- |
| Time          | O(n) deterministic              |
| Space         | O(n) (can be optimized to O(1)) |
| Stable        | No                              |
| Pivot Quality | Guaranteed 30–70 split          |

The BFPRT Algorithm, proof that with clever pivots and math, even chaos can be conquered in linear time.

### 180 Kth Largest Stream

The Kth Largest Stream problem focuses on maintaining the kth largest element in a sequence that grows over time, a stream.
Instead of sorting everything every time, we can use a min-heap of size k to always keep track of the top `k` elements efficiently.

This is the foundation for real-time leaderboards, streaming analytics, and online ranking systems.

### What Problem Are We Solving?

Given a stream of numbers (arriving one by one), we want to:

- Always know the kth largest element so far.
- Update quickly when a new number comes in.

You don't know the final list, you process as it flows.

#### Example

Say `k = 3`, stream = `[4, 5, 8, 2]`

1. Start empty heap
2. Add 4 → heap = [4] → kth largest = 4
3. Add 5 → heap = [4, 5] → kth largest = 4
4. Add 8 → heap = [4, 5, 8] → kth largest = 4
5. Add 2 → ignore (2 < 4) → kth largest = 4

Add new number `10`:

- 10 > 4 → pop 4, push 10 → heap = [5,8,10]
- kth largest = 5

✅ Each new element processed in O(log k)

### How Does It Work (Plain Language)?

Keep a min-heap of size `k`:

- It holds the k largest elements seen so far.
- The smallest among them (heap root) is the kth largest.
- When a new value arrives:

  * If heap size < k → push it
  * Else if value > heap[0] → pop smallest, push new

#### Step-by-Step Summary

| Step | Action                                |
| ---- | ------------------------------------- |
| 1    | Initialize empty min-heap             |
| 2    | For each new element `x`:             |
|      | If heap size < k → push `x`           |
|      | Else if `x` > heap[0] → pop, push `x` |
| 3    | kth largest = heap[0]                 |

### Tiny Code (Easy Versions)

#### Python

```python
import heapq

class KthLargest:
    def __init__(self, k, nums):
        self.k = k
        self.heap = nums
        heapq.heapify(self.heap)
        while len(self.heap) > k:
            heapq.heappop(self.heap)

    def add(self, val):
        if len(self.heap) < self.k:
            heapq.heappush(self.heap, val)
        elif val > self.heap[0]:
            heapq.heapreplace(self.heap, val)
        return self.heap[0]

# Example
stream = KthLargest(3, [4,5,8,2])
print(stream.add(3))  # 4
print(stream.add(5))  # 5
print(stream.add(10)) # 5
print(stream.add(9))  # 8
print(stream.add(4))  # 8
```

Output:

```
4
5
5
8
8
```

#### C

```c
#include <stdio.h>
#include <stdlib.h>

void swap(int *a, int *b) { int t = *a; *a = *b; *b = t; }

void heapify(int arr[], int n, int i) {
    int smallest = i, l = 2*i+1, r = 2*i+2;
    if (l < n && arr[l] < arr[smallest]) smallest = l;
    if (r < n && arr[r] < arr[smallest]) smallest = r;
    if (smallest != i) {
        swap(&arr[i], &arr[smallest]);
        heapify(arr, n, smallest);
    }
}

void push_heap(int heap[], int *n, int val) {
    heap[(*n)++] = val;
    for (int i = (*n)/2 - 1; i >= 0; i--) heapify(heap, *n, i);
}

int pop_min(int heap[], int *n) {
    int root = heap[0];
    heap[0] = heap[--(*n)];
    heapify(heap, *n, 0);
    return root;
}

int add(int heap[], int *n, int k, int val) {
    if (*n < k) {
        push_heap(heap, n, val);
    } else if (val > heap[0]) {
        heap[0] = val;
        heapify(heap, *n, 0);
    }
    return heap[0];
}

int main(void) {
    int heap[10] = {4,5,8,2};
    int n = 4, k = 3;
    for (int i = n/2 - 1; i >= 0; i--) heapify(heap, n, i);
    while (n > k) pop_min(heap, &n);

    printf("%d\n", add(heap, &n, k, 3));  // 4
    printf("%d\n", add(heap, &n, k, 5));  // 5
    printf("%d\n", add(heap, &n, k, 10)); // 5
    printf("%d\n", add(heap, &n, k, 9));  // 8
    printf("%d\n", add(heap, &n, k, 4));  // 8
}
```

### Why It Matters

- Real-time streaming top-k tracking
- Constant-time query (O(1)), fast update (O(log k))
- Core building block for:

  * Leaderboards
  * Monitoring systems
  * Continuous analytics
  * Online medians & percentiles

### A Gentle Proof (Why It Works)

Min-heap stores only top `k` values.
Whenever new value > heap[0], it must belong in top `k`.
So invariant holds: heap = top-k largest elements seen so far.
kth largest = heap[0].

Each update → O(log k).
Total after n elements → O(n log k).

### Try It Yourself

1. Initialize with `[4,5,8,2]`, k=3, stream = [3,5,10,9,4]
2. Try decreasing sequence
3. Try duplicates
4. Test `k = 1` (maximum tracker)
5. Add 1000 elements randomly, measure performance
6. Compare with full sort each time
7. Visualize heap evolution per step
8. Modify for k smallest
9. Build real-time median tracker using two heaps
10. Extend to stream of objects (track by score field)

### Test Cases

| Initial     | k | Stream       | Output Sequence |
| ----------- | - | ------------ | --------------- |
| [4,5,8,2]   | 3 | [3,5,10,9,4] | [4,5,5,8,8]     |
| [10,7,11,5] | 2 | [8,12,4]     | [10,11,11]      |
| [1]         | 1 | [2,3]        | [2,3]           |

### Complexity

| Operation         | Complexity |
| ----------------- | ---------- |
| Add               | O(log k)   |
| Query kth Largest | O(1)       |
| Space             | O(k)       |

Kth Largest Stream, stay calm in the flow; the heap remembers what matters most.

# Section 19. Range Search and Nearest Neighbor 

### 181 Binary Search Range

Binary Search Range extends the basic binary search to find not just one occurrence, but the range of positions where a given value appears, specifically, the first (lower bound) and last (upper bound) indices of a target in a sorted array.

It's the backbone for problems that require counting occurrences, range queries, and insertion positions in ordered data.

### What Problem Are We Solving?

Standard binary search returns *one* match. But what if the target appears multiple times, or we want where it *should* go?

Examples:

- Count occurrences of `x` in sorted array
- Find the first element ≥ `x`
- Find the last element ≤ `x`

With two binary searches, we can find the full [start, end] range efficiently.

#### Example

Array: `[1, 2, 2, 2, 3, 4, 5]`, target = `2`

| Function          | Result |
| ----------------- | ------ |
| Lower Bound (≥ 2) | 1      |
| Upper Bound (> 2) | 4      |
| Range             | [1, 3] |

Occurrences = `upper - lower = 3`

### How Does It Work (Plain Language)?

We use binary search twice:

- One to find the first index ≥ target (lower bound)
- One to find the first index > target (upper bound)

Subtract them to get the count, or slice the range.

#### Step-by-Step Summary

| Step | Description                                                |
| ---- | ---------------------------------------------------------- |
| 1    | Binary search for first index `i` where `arr[i] >= target` |
| 2    | Binary search for first index `j` where `arr[j] > target`  |
| 3    | Range = `[i, j - 1]` if `i < j` and `arr[i] == target`     |
| 4    | Count = `j - i`                                            |

### Tiny Code (Easy Versions)

#### Python

```python
def lower_bound(arr, target):
    lo, hi = 0, len(arr)
    while lo < hi:
        mid = (lo + hi) // 2
        if arr[mid] < target:
            lo = mid + 1
        else:
            hi = mid
    return lo

def upper_bound(arr, target):
    lo, hi = 0, len(arr)
    while lo < hi:
        mid = (lo + hi) // 2
        if arr[mid] <= target:
            lo = mid + 1
        else:
            hi = mid
    return lo

def binary_search_range(arr, target):
    l = lower_bound(arr, target)
    r = upper_bound(arr, target)
    if l == r:
        return (-1, -1)  # Not found
    return (l, r - 1)

arr = [1, 2, 2, 2, 3, 4, 5]
target = 2
print("Range:", binary_search_range(arr, target))
print("Count:", upper_bound(arr, target) - lower_bound(arr, target))
```

Output:

```
Range: (1, 3)
Count: 3
```

#### C

```c
#include <stdio.h>

int lower_bound(int arr[], int n, int target) {
    int lo = 0, hi = n;
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (arr[mid] < target) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

int upper_bound(int arr[], int n, int target) {
    int lo = 0, hi = n;
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (arr[mid] <= target) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

int main(void) {
    int arr[] = {1, 2, 2, 2, 3, 4, 5};
    int n = 7, target = 2;
    int l = lower_bound(arr, n, target);
    int r = upper_bound(arr, n, target);
    if (l == r) printf("Not found\n");
    else printf("Range: [%d, %d], Count: %d\n", l, r - 1, r - l);
}
```

Output:

```
Range: [1, 3], Count: 3
```

### Why It Matters

- Extends binary search beyond "found or not"
- Essential in frequency counting, range queries, histograms
- Powers data structures like Segment Trees, Fenwick Trees, and Range Indexes
- Used in competitive programming and database indexing

### A Gentle Proof (Why It Works)

Because binary search maintains sorted invariants (`lo < hi` and mid conditions),

- Lower bound finds first index where condition flips (`< target` → `≥ target`)
- Upper bound finds first index beyond target

Both run in O(log n), giving exact range boundaries.

### Try It Yourself

1. Test with no occurrences (e.g. `[1,3,5]`, target=2)
2. Test with all equal (e.g. `[2,2,2,2]`, target=2)
3. Test with first element = target
4. Test with last element = target
5. Try to count elements ≤ `x` or < `x`
6. Extend for floating point or custom comparator
7. Use on strings or tuples
8. Combine with bisect in Python
9. Compare iterative vs recursive
10. Use as primitive for frequency table

### Test Cases

| Array           | Target | Range   | Count |
| --------------- | ------ | ------- | ----- |
| [1,2,2,2,3,4,5] | 2      | [1,3]   | 3     |
| [1,3,5,7]       | 2      | [-1,-1] | 0     |
| [2,2,2,2]       | 2      | [0,3]   | 4     |
| [1,2,3,4]       | 4      | [3,3]   | 1     |

### Complexity

| Operation   | Complexity |
| ----------- | ---------- |
| Lower Bound | O(log n)   |
| Upper Bound | O(log n)   |
| Space       | O(1)       |
| Stable      | Yes        |

Binary Search Range, when one answer isn't enough, and precision is everything.

### 182 Segment Tree Query

Segment Tree Query is a powerful data structure technique that allows you to efficiently compute range queries like sum, minimum, maximum, or even custom associative operations over subarrays.

It preprocesses the array into a binary tree structure, where each node stores a summary (aggregate) of a segment.

Once built, you can answer queries and updates in O(log n) time.

### What Problem Are We Solving?

Given an array, we often want to query over ranges:

- Sum over `[L, R]`
- Minimum or Maximum in `[L, R]`
- GCD, product, XOR, or any associative function

A naive approach would loop each query: O(n) per query.
Segment Trees reduce this to O(log n) with a one-time O(n) build.

#### Example

Array: `[2, 4, 5, 7, 8, 9]`

| Query    | Result           |
| -------- | ---------------- |
| Sum(1,3) | 4+5+7 = 16       |
| Min(2,5) | min(5,7,8,9) = 5 |

### How It Works (Intuitive View)

A Segment Tree is like a binary hierarchy:

- The root covers the full range `[0, n-1]`
- Each node covers a subrange
- The leaf nodes are individual elements
- Each internal node stores a merge (sum, min, max...) of its children

To query a range, you traverse only relevant branches.

#### Build, Query, Update

| Operation | Description                        | Time     |
| --------- | ---------------------------------- | -------- |
| Build     | Recursively combine child segments | O(n)     |
| Query     | Traverse overlapping nodes         | O(log n) |
| Update    | Recompute along path               | O(log n) |

### Tiny Code (Sum Query Example)

#### Python

```python
class SegmentTree:
    def __init__(self, arr):
        self.n = len(arr)
        self.tree = [0] * (4 * self.n)
        self._build(arr, 1, 0, self.n - 1)

    def _build(self, arr, node, l, r):
        if l == r:
            self.tree[node] = arr[l]
        else:
            mid = (l + r) // 2
            self._build(arr, 2 * node, l, mid)
            self._build(arr, 2 * node + 1, mid + 1, r)
            self.tree[node] = self.tree[2 * node] + self.tree[2 * node + 1]

    def query(self, node, l, r, ql, qr):
        if qr < l or ql > r:  # no overlap
            return 0
        if ql <= l and r <= qr:  # total overlap
            return self.tree[node]
        mid = (l + r) // 2
        left = self.query(2 * node, l, mid, ql, qr)
        right = self.query(2 * node + 1, mid + 1, r, ql, qr)
        return left + right

# Example
arr = [2, 4, 5, 7, 8, 9]
st = SegmentTree(arr)
print(st.query(1, 0, len(arr) - 1, 1, 3))  # Sum from index 1 to 3
```

Output:

```
16
```

#### C

```c
#include <stdio.h>

#define MAXN 100
int tree[4 * MAXN];
int arr[MAXN];

int build(int node, int l, int r) {
    if (l == r) return tree[node] = arr[l];
    int mid = (l + r) / 2;
    int left = build(2 * node, l, mid);
    int right = build(2 * node + 1, mid + 1, r);
    return tree[node] = left + right;
}

int query(int node, int l, int r, int ql, int qr) {
    if (qr < l || ql > r) return 0;
    if (ql <= l && r <= qr) return tree[node];
    int mid = (l + r) / 2;
    return query(2 * node, l, mid, ql, qr) + query(2 * node + 1, mid + 1, r, ql, qr);
}

int main() {
    int n = 6;
    int data[] = {2, 4, 5, 7, 8, 9};
    for (int i = 0; i < n; i++) arr[i] = data[i];
    build(1, 0, n - 1);
    printf("Sum [1,3] = %d\n", query(1, 0, n - 1, 1, 3));
}
```

Output:

```
Sum [1,3] = 16
```

### Why It Matters

- Handles dynamic range queries and updates efficiently
- Core of competitive programming and data analytics
- Forms base for Range Minimum Query, 2D queries, and lazy propagation
- Useful in databases, financial systems, and game engines

### Intuition (Associativity Rule)

Segment Trees only work when the operation is associative:

```
merge(a, merge(b, c)) = merge(merge(a, b), c)
```

Examples:

- Sum, Min, Max, GCD, XOR
- Not Median, Not Mode (non-associative)

### Try It Yourself

1. Implement for min or max instead of sum
2. Add update() for point changes
3. Implement lazy propagation for range updates
4. Extend to 2D segment tree
5. Compare with Fenwick Tree (BIT)
6. Test on non-trivial ranges
7. Visualize the tree layout
8. Build iterative segment tree
9. Handle custom operations (GCD, XOR)
10. Benchmark O(n log n) vs naive O(nq)

### Test Cases

| Array         | Query    | Expected |
| ------------- | -------- | -------- |
| [2,4,5,7,8,9] | Sum(1,3) | 16       |
| [1,2,3,4]     | Sum(0,3) | 10       |
| [5,5,5,5]     | Sum(1,2) | 10       |
| [3,2,1,4]     | Min(1,3) | 1        |

### Complexity

| Operation | Complexity |
| --------- | ---------- |
| Build     | O(n)       |
| Query     | O(log n)   |
| Update    | O(log n)   |
| Space     | O(4n)      |

Segment Tree Query, build once, query many, fast forever.

### 183 Fenwick Tree Query

A Fenwick Tree (or Binary Indexed Tree) is a data structure designed for prefix queries and point updates in O(log n) time.
It's a more space-efficient, iterative cousin of the Segment Tree, perfect when operations are cumulative (sum, XOR, etc.) and updates are frequent.

### What Problem Are We Solving?

We want to:

- Compute prefix sums efficiently
- Support updates dynamically

A naive approach takes O(n) per query or update.
A Fenwick Tree does both in O(log n).

#### Example

Array: `[2, 4, 5, 7, 8]`

| Query          | Result                              |
| -------------- | ----------------------------------- |
| PrefixSum(3)   | 2 + 4 + 5 + 7 = 18                  |
| RangeSum(1, 3) | Prefix(3) - Prefix(0) = 18 - 2 = 16 |

### How It Works (Plain Language)

A Fenwick Tree stores cumulative information in indexed chunks.
Each index covers a range determined by its least significant bit (LSB).

```
index i covers range (i - LSB(i) + 1) ... i
```

We can update or query by moving through indices using bit operations:

- Update: move forward by adding LSB
- Query: move backward by subtracting LSB

This clever bit trick keeps operations O(log n).

#### Example Walkthrough

For array `[2, 4, 5, 7, 8]` (1-based index):

| Index | Binary | LSB | Range | Value |
| ----- | ------ | --- | ----- | ----- |
| 1     | 001    | 1   | [1]   | 2     |
| 2     | 010    | 2   | [1–2] | 6     |
| 3     | 011    | 1   | [3]   | 5     |
| 4     | 100    | 4   | [1–4] | 18    |
| 5     | 101    | 1   | [5]   | 8     |

### Tiny Code (Sum Example)

#### Python

```python
class FenwickTree:
    def __init__(self, n):
        self.n = n
        self.bit = [0] * (n + 1)

    def update(self, i, delta):
        while i <= self.n:
            self.bit[i] += delta
            i += i & -i

    def query(self, i):
        s = 0
        while i > 0:
            s += self.bit[i]
            i -= i & -i
        return s

    def range_sum(self, l, r):
        return self.query(r) - self.query(l - 1)

# Example
arr = [2, 4, 5, 7, 8]
ft = FenwickTree(len(arr))
for i, val in enumerate(arr, 1):
    ft.update(i, val)

print(ft.range_sum(2, 4))  # 4 + 5 + 7 = 16
```

Output:

```
16
```

#### C

```c
#include <stdio.h>

#define MAXN 100
int bit[MAXN + 1], n;

void update(int i, int delta) {
    while (i <= n) {
        bit[i] += delta;
        i += i & -i;
    }
}

int query(int i) {
    int s = 0;
    while (i > 0) {
        s += bit[i];
        i -= i & -i;
    }
    return s;
}

int range_sum(int l, int r) {
    return query(r) - query(l - 1);
}

int main() {
    n = 5;
    int arr[] = {0, 2, 4, 5, 7, 8}; // 1-based
    for (int i = 1; i <= n; i++) update(i, arr[i]);
    printf("Sum [2,4] = %d\n", range_sum(2,4)); // 16
}
```

Output:

```
Sum [2,4] = 16
```

### Why It Matters

- Elegant bit manipulation for efficient queries
- Simpler and smaller than Segment Trees
- Perfect for prefix sums, inversions, frequency tables
- Extends to 2D Fenwick Trees for grid-based data
- Core in competitive programming, streaming, finance

### Intuition (Least Significant Bit)

The LSB trick (i & -i) finds the rightmost set bit, controlling how far we jump.
This ensures logarithmic traversal through relevant nodes.

### Try It Yourself

1. Implement a prefix XOR version
2. Add range updates with two trees
3. Extend to 2D BIT for matrix sums
4. Visualize tree structure for array [1..8]
5. Compare speed with naive O(n) approach
6. Track frequency counts for elements
7. Use it for inversion counting
8. Create a Fenwick Tree class in C++
9. Handle point updates interactively
10. Practice bit math: draw index cover ranges

### Test Cases

| Array       | Query                  | Expected |
| ----------- | ---------------------- | -------- |
| [2,4,5,7,8] | Sum(2,4)               | 16       |
| [1,2,3,4]   | Prefix(3)              | 6        |
| [5,5,5,5]   | Sum(1,3)               | 15       |
| [3,1,4,2]   | Update(2,+3), Sum(1,2) | 7        |

### Complexity

| Operation | Complexity |
| --------- | ---------- |
| Build     | O(n log n) |
| Query     | O(log n)   |
| Update    | O(log n)   |
| Space     | O(n)       |

A Fenwick Tree turns prefix operations into lightning-fast bit magic, simple, small, and powerful.

### 184 Interval Tree Search

An Interval Tree is a data structure built to efficiently store intervals (ranges like [l, r]) and query all intervals that overlap with a given interval or point.
It's like a BST with range-awareness, enabling fast queries such as "which tasks overlap with time t?" or "which rectangles overlap this region?"

### What Problem Are We Solving?

We want to efficiently find overlapping intervals.
A naive search checks all intervals, O(n) per query.
An Interval Tree speeds this up to O(log n + k), where *k* is the number of overlapping intervals.

#### Example

Stored intervals:

```
$$5, 20], [10, 30], [12, 15], [17, 19], [30, 40]
```

Query: `[14, 16]`

Overlaps: `[10, 30]`, `[12, 15]`

### How It Works (Plain Language)

1. Build a BST using the midpoint or start of intervals as keys.
2. Each node stores:

   * interval [low, high]
   * max endpoint of its subtree
3. For queries:

   * Traverse tree, skip branches where `low > query_high` or `max < query_low`.
   * Collect overlapping intervals efficiently.

This pruning makes it logarithmic for most cases.

#### Example Tree (Sorted by low)

```
            [10, 30]
           /        \
     [5, 20]       [17, 19]
                     \
                    [30, 40]
```

Each node stores `max` endpoint of its subtree.

### Tiny Code (Query Example)

#### Python

```python
class IntervalNode:
    def __init__(self, low, high):
        self.low = low
        self.high = high
        self.max = high
        self.left = None
        self.right = None

def insert(root, low, high):
    if root is None:
        return IntervalNode(low, high)
    if low < root.low:
        root.left = insert(root.left, low, high)
    else:
        root.right = insert(root.right, low, high)
    root.max = max(root.max, high)
    return root

def overlap(i1, i2):
    return i1[0] <= i2[1] and i2[0] <= i1[1]

def search(root, query):
    if root is None:
        return []
    result = []
    if overlap((root.low, root.high), query):
        result.append((root.low, root.high))
    if root.left and root.left.max >= query[0]:
        result += search(root.left, query)
    if root.right and root.low <= query[1]:
        result += search(root.right, query)
    return result

# Example
intervals = [(5,20), (10,30), (12,15), (17,19), (30,40)]
root = None
for l, h in intervals:
    root = insert(root, l, h)

print(search(root, (14,16)))  # [(10,30), (12,15)]
```

Output:

```
$$(10, 30), (12, 15)]
```

#### C

```c
#include <stdio.h>
#include <stdlib.h>

typedef struct Node {
    int low, high, max;
    struct Node *left, *right;
} Node;

Node* newNode(int low, int high) {
    Node* n = malloc(sizeof(Node));
    n->low = low;
    n->high = high;
    n->max = high;
    n->left = n->right = NULL;
    return n;
}

int max(int a, int b) { return a > b ? a : b; }

Node* insert(Node* root, int low, int high) {
    if (!root) return newNode(low, high);
    if (low < root->low)
        root->left = insert(root->left, low, high);
    else
        root->right = insert(root->right, low, high);
    root->max = max(root->max, high);
    return root;
}

int overlap(int l1, int h1, int l2, int h2) {
    return l1 <= h2 && l2 <= h1;
}

void search(Node* root, int ql, int qh) {
    if (!root) return;
    if (overlap(root->low, root->high, ql, qh))
        printf("[%d, %d] overlaps\n", root->low, root->high);
    if (root->left && root->left->max >= ql)
        search(root->left, ql, qh);
    if (root->right && root->low <= qh)
        search(root->right, ql, qh);
}

int main() {
    Node* root = NULL;
    int intervals[][2] = {{5,20},{10,30},{12,15},{17,19},{30,40}};
    int n = 5;
    for (int i = 0; i < n; i++)
        root = insert(root, intervals[i][0], intervals[i][1]);
    printf("Overlaps with [14,16]:\n");
    search(root, 14, 16);
}
```

Output:

```
Overlaps with [14,16]:
$$10, 30] overlaps
$$12, 15] overlaps
```

### Why It Matters

- Efficient for overlap queries (e.g. events, tasks, ranges)
- Used in:

  * Scheduling (detecting conflicts)
  * Computational geometry
  * Memory allocation checks
  * Genomic range matching
- Foundation for Segment Tree with intervals

### Key Intuition

Each node stores the max endpoint of its subtree.
This helps prune non-overlapping branches early.

Think of it as a "range-aware BST".

### Try It Yourself

1. Build tree for intervals: [1,5], [2,6], [7,9], [10,15]
2. Query [4,8], which overlap?
3. Visualize pruning path
4. Extend to delete intervals
5. Add count of overlapping intervals
6. Implement iterative search
7. Compare with brute-force O(n) approach
8. Adapt for point queries only
9. Try dynamic updates
10. Use to detect meeting conflicts

### Test Cases

| Intervals                                  | Query   | Expected Overlaps |
| ------------------------------------------ | ------- | ----------------- |
| [5,20], [10,30], [12,15], [17,19], [30,40] | [14,16] | [10,30], [12,15]  |
| [1,3], [5,8], [6,10]                       | [7,9]   | [5,8], [6,10]     |
| [2,5], [6,8]                               | [1,1]   | none              |

### Complexity

| Operation | Complexity   |
| --------- | ------------ |
| Build     | O(n log n)   |
| Query     | O(log n + k) |
| Space     | O(n)         |

An Interval Tree is your go-to for range overlap queries, BST elegance meets interval intelligence.

### 185 KD-Tree Search

A KD-Tree (k-dimensional tree) is a space-partitioning data structure that organizes points in *k*-dimensional space for efficient nearest neighbor, range, and radius searches.
It's like a binary search tree, but it splits space along alternating dimensions.

### What Problem Are We Solving?

We want to:

- Find points near a given location
- Query points within a region or radius
- Do this faster than checking all points (O(n))

A KD-Tree answers such queries in O(log n) (average), versus O(n) for brute-force.

#### Example

Points in 2D:

```
(2,3), (5,4), (9,6), (4,7), (8,1), (7,2)
```

Query: Nearest neighbor of (9,2)
Result: (8,1)

### How It Works (Plain Language)

1. Build Tree

   * Choose a splitting dimension (x, y, …)
   * Pick median point along that axis
   * Recursively build left/right subtrees

2. Search

   * Compare query coordinate along current axis
   * Recurse into the nearer subtree
   * Backtrack to check the other side if necessary (only if the hypersphere crosses boundary)

This pruning makes nearest-neighbor search efficient.

#### Example (2D Split)

```
           (7,2)  [split x]
          /           \
    (5,4) [y]         (9,6) [y]
    /     \             /
 (2,3)  (4,7)       (8,1)
```

### Tiny Code (2D Example)

#### Python

```python
from math import sqrt

class Node:
    def __init__(self, point, axis):
        self.point = point
        self.axis = axis
        self.left = None
        self.right = None

def build_kdtree(points, depth=0):
    if not points:
        return None
    k = len(points[0])
    axis = depth % k
    points.sort(key=lambda p: p[axis])
    mid = len(points) // 2
    node = Node(points[mid], axis)
    node.left = build_kdtree(points[:mid], depth + 1)
    node.right = build_kdtree(points[mid+1:], depth + 1)
    return node

def distance2(a, b):
    return sum((x - y)  2 for x, y in zip(a, b))

def nearest(root, target, best=None):
    if root is None:
        return best
    point = root.point
    if best is None or distance2(point, target) < distance2(best, target):
        best = point
    axis = root.axis
    next_branch = root.left if target[axis] < point[axis] else root.right
    best = nearest(next_branch, target, best)
    if (target[axis] - point[axis])  2 < distance2(best, target):
        other = root.right if next_branch == root.left else root.left
        best = nearest(other, target, best)
    return best

points = [(2,3),(5,4),(9,6),(4,7),(8,1),(7,2)]
tree = build_kdtree(points)
print(nearest(tree, (9,2)))  # (8,1)
```

Output:

```
(8, 1)
```

#### C (2D Simplified)

```c
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

typedef struct Node {
    double point[2];
    int axis;
    struct Node *left, *right;
} Node;

int cmpx(const void* a, const void* b) {
    double* pa = (double*)a;
    double* pb = (double*)b;
    return (pa[0] > pb[0]) - (pa[0] < pb[0]);
}

int cmpy(const void* a, const void* b) {
    double* pa = (double*)a;
    double* pb = (double*)b;
    return (pa[1] > pb[1]) - (pa[1] < pb[1]);
}

double dist2(double a[2], double b[2]) {
    return (a[0]-b[0])*(a[0]-b[0]) + (a[1]-b[1])*(a[1]-b[1]);
}

// Simplified build & search omitted for brevity (tree construction similar to Python)
```

### Why It Matters

- Efficient for spatial queries in 2D, 3D, etc.
- Used in:

  * Machine Learning (KNN classification)
  * Graphics (ray tracing, collision detection)
  * Robotics (path planning, SLAM)
  * Databases (multi-dimensional indexing)

### Intuition

A KD-Tree is like playing "binary search" in multiple dimensions.
Each split narrows down the search region.

### Try It Yourself

1. Build a KD-Tree for points in 2D
2. Search nearest neighbor of (3,5)
3. Add 3D points, use modulo axis split
4. Visualize splits as alternating vertical/horizontal lines
5. Extend to k-NN (top-k closest)
6. Add radius query (points within r)
7. Compare speed to brute-force
8. Track backtrack count for pruning visualization
9. Try non-uniform data
10. Implement deletion (bonus)

### Test Cases

| Points                              | Query | Expected Nearest |
| ----------------------------------- | ----- | ---------------- |
| (2,3),(5,4),(9,6),(4,7),(8,1),(7,2) | (9,2) | (8,1)            |
| (1,1),(3,3),(5,5)                   | (4,4) | (3,3)            |
| (0,0),(10,10)                       | (7,8) | (10,10)          |

### Complexity

| Operation     | Complexity       |
| ------------- | ---------------- |
| Build         | O(n log n)       |
| Nearest Query | O(log n) average |
| Worst Case    | O(n)             |
| Space         | O(n)             |

A KD-Tree slices space along dimensions, your go-to for fast nearest neighbor searches in multidimensional worlds.

### 186 R-Tree Query

An R-Tree is a hierarchical spatial index built for efficiently querying geometric objects (rectangles, polygons, circles) in 2D or higher dimensions.
It's like a B-Tree for rectangles, grouping nearby objects into bounding boxes and organizing them in a tree for fast spatial lookups.

### What Problem Are We Solving?

We need to query spatial data efficiently:

- "Which rectangles overlap this area?"
- "What points fall inside this region?"
- "Which shapes intersect this polygon?"

A naive approach checks every object (O(n)).
An R-Tree reduces this to O(log n + k) using bounding-box hierarchy.

#### Example

Rectangles:

```
A: [1,1,3,3]
B: [2,2,5,4]
C: [4,1,6,3]
```

Query: `[2.5,2.5,4,4]`
Overlaps: A, B

### How It Works (Plain Language)

1. Store rectangles (or bounding boxes) as leaves.
2. Group nearby rectangles into Minimum Bounding Rectangles (MBRs).
3. Build hierarchy so each node's box covers its children.
4. Query by recursively checking nodes whose boxes overlap the query.

This spatial grouping allows skipping entire regions quickly.

#### Example Tree

```
             [1,1,6,4]
            /         \
     [1,1,3,3]       [4,1,6,4]
       (A,B)            (C)
```

Query `[2.5,2.5,4,4]`:

- Intersects left node → check A, B
- Intersects right node partially → check C (no overlap)

### Tiny Code (2D Rectangles)

#### Python

```python
def overlap(a, b):
    return not (a[2] < b[0] or a[0] > b[2] or a[3] < b[1] or a[1] > b[3])

class RTreeNode:
    def __init__(self, box, children=None, is_leaf=False):
        self.box = box  # [x1, y1, x2, y2]
        self.children = children or []
        self.is_leaf = is_leaf

def search_rtree(node, query):
    results = []
    if not overlap(node.box, query):
        return results
    if node.is_leaf:
        for child in node.children:
            if overlap(child.box, query):
                results.append(child.box)
    else:
        for child in node.children:
            results.extend(search_rtree(child, query))
    return results

# Example
A = RTreeNode([1,1,3,3], is_leaf=True)
B = RTreeNode([2,2,5,4], is_leaf=True)
C = RTreeNode([4,1,6,3], is_leaf=True)

left = RTreeNode([1,1,5,4], [A,B], is_leaf=True)
right = RTreeNode([4,1,6,3], [C], is_leaf=True)
root = RTreeNode([1,1,6,4], [left, right])

query = [2.5,2.5,4,4]
print(search_rtree(root, query))
```

Output:

```
$$[1, 1, 3, 3], [2, 2, 5, 4]]
```

#### C (Simplified Query)

```c
#include <stdio.h>

typedef struct Box {
    float x1, y1, x2, y2;
} Box;

int overlap(Box a, Box b) {
    return !(a.x2 < b.x1 || a.x1 > b.x2 || a.y2 < b.y1 || a.y1 > b.y2);
}

// Example: Manual tree simulation omitted for brevity
```

### Why It Matters

- Ideal for geospatial databases, mapping, collision detection, and GIS.
- Powers PostGIS, SQLite R*Tree module, spatial indexes.
- Handles overlaps, containment, and range queries.

### Intuition

R-Trees work by bounding and grouping.
Each node is a "container box", if it doesn't overlap the query, skip it entirely.
This saves massive time in spatial datasets.

### Try It Yourself

1. Represent 2D rectangles with `[x1,y1,x2,y2]`.
2. Build a 2-level tree (group nearby).
3. Query overlap region.
4. Extend to 3D bounding boxes.
5. Implement insertion using least expansion rule.
6. Add R*-Tree optimization (reinsert on overflow).
7. Compare with QuadTree (grid-based).
8. Visualize bounding boxes per level.
9. Implement nearest neighbor search.
10. Try dataset with 10k rectangles, measure speedup.

### Test Cases

| Rectangles                         | Query         | Expected Overlaps |
| ---------------------------------- | ------------- | ----------------- |
| A[1,1,3,3], B[2,2,5,4], C[4,1,6,3] | [2.5,2.5,4,4] | A, B              |
| A[0,0,2,2], B[3,3,4,4]             | [1,1,3,3]     | A                 |
| A[1,1,5,5], B[6,6,8,8]             | [7,7,9,9]     | B                 |

### Complexity

| Operation | Complexity   |
| --------- | ------------ |
| Build     | O(n log n)   |
| Query     | O(log n + k) |
| Space     | O(n)         |

An R-Tree is your geometric librarian, organizing space into nested rectangles so you can query complex regions fast and clean.

### 187 Range Minimum Query (RMQ), Sparse Table Approach

A Range Minimum Query (RMQ) answers questions like:

> "What's the smallest element between indices *L* and *R*?"

It's a core subroutine in many algorithms, from LCA (Lowest Common Ancestor) to scheduling, histograms, and segment analysis.
The Sparse Table method precomputes answers so each query is O(1) after O(n log n) preprocessing.

### What Problem Are We Solving?

Given an array `arr[0..n-1]`, we want to answer:

```
RMQ(L, R) = min(arr[L], arr[L+1], …, arr[R])
```

Efficiently, for multiple static queries (no updates).

Naive approach: O(R-L) per query
Sparse Table: O(1) per query after preprocessing.

#### Example

Array: `[2, 5, 1, 4, 9, 3]`

| Query     | Result           |
| --------- | ---------------- |
| RMQ(1, 3) | min(5,1,4) = 1   |
| RMQ(2, 5) | min(1,4,9,3) = 1 |

### How It Works (Plain Language)

1. Precompute answers for all intervals of length 2^k.
2. To answer RMQ(L,R):

   * Let `len = R-L+1`
   * Let `k = floor(log2(len))`
   * Combine two overlapping intervals of size `2^k`:

     ```
     RMQ(L,R) = min(st[L][k], st[R - 2^k + 1][k])
     ```

No updates, so data stays static and queries stay O(1).

#### Sparse Table Example

| i | arr[i] | st[i][0] | st[i][1]   | st[i][2]   |
| - | ------ | -------- | ---------- | ---------- |
| 0 | 2      | 2        | min(2,5)=2 | min(2,1)=1 |
| 1 | 5      | 5        | min(5,1)=1 | min(5,4)=1 |
| 2 | 1      | 1        | min(1,4)=1 | min(1,9)=1 |
| 3 | 4      | 4        | min(4,9)=4 | min(4,3)=3 |
| 4 | 9      | 9        | min(9,3)=3 |,          |
| 5 | 3      | 3        |,          |,          |

### Tiny Code

#### Python

```python
import math

def build_sparse_table(arr):
    n = len(arr)
    K = math.floor(math.log2(n)) + 1
    st = [[0]*K for _ in range(n)]

    for i in range(n):
        st[i][0] = arr[i]

    j = 1
    while (1 << j) <= n:
        i = 0
        while i + (1 << j) <= n:
            st[i][j] = min(st[i][j-1], st[i + (1 << (j-1))][j-1])
            i += 1
        j += 1
    return st

def query(st, L, R):
    j = int(math.log2(R - L + 1))
    return min(st[L][j], st[R - (1 << j) + 1][j])

# Example
arr = [2, 5, 1, 4, 9, 3]
st = build_sparse_table(arr)
print(query(st, 1, 3))  # 1
print(query(st, 2, 5))  # 1
```

Output:

```
1
1
```

#### C

```c
#include <stdio.h>
#include <math.h>

#define MAXN 100
#define LOG 17

int st[MAXN][LOG];
int arr[MAXN];
int n;

void build() {
    for (int i = 0; i < n; i++)
        st[i][0] = arr[i];
    for (int j = 1; (1 << j) <= n; j++) {
        for (int i = 0; i + (1 << j) <= n; i++) {
            st[i][j] = (st[i][j-1] < st[i + (1 << (j-1))][j-1]) 
                       ? st[i][j-1] 
                       : st[i + (1 << (j-1))][j-1];
        }
    }
}

int query(int L, int R) {
    int j = log2(R - L + 1);
    int left = st[L][j];
    int right = st[R - (1 << j) + 1][j];
    return left < right ? left : right;
}

int main() {
    n = 6;
    int arr_temp[] = {2,5,1,4,9,3};
    for (int i = 0; i < n; i++) arr[i] = arr_temp[i];
    build();
    printf("RMQ(1,3) = %d\n", query(1,3)); // 1
    printf("RMQ(2,5) = %d\n", query(2,5)); // 1
}
```

Output:

```
RMQ(1,3) = 1  
RMQ(2,5) = 1
```

### Why It Matters

- Instant queries after precomputation

- Crucial for:

  * Segment analysis (min, max)
  * LCA in trees
  * Sparse range data
  * Static arrays (no updates)

- Perfect when array does not change frequently.

### Intuition

Each table entry `st[i][k]` stores the minimum of range `[i, i + 2^k - 1]`.
Queries merge two overlapping intervals that cover `[L,R]`.

### Try It Yourself

1. Build table for `[1,3,2,7,9,11,3,5,6]`
2. Query RMQ(2,6) and RMQ(4,8)
3. Modify code to compute Range Max Query
4. Visualize overlapping intervals for query
5. Compare with Segment Tree version
6. Add precomputed log[] for faster lookup
7. Handle 1-based vs 0-based indices carefully
8. Practice on random arrays
9. Compare preprocessing time with naive
10. Use to solve LCA using Euler Tour

### Test Cases

| Array         | Query    | Expected |
| ------------- | -------- | -------- |
| [2,5,1,4,9,3] | RMQ(1,3) | 1        |
| [2,5,1,4,9,3] | RMQ(2,5) | 1        |
| [1,2,3,4]     | RMQ(0,3) | 1        |
| [7,6,5,4,3]   | RMQ(1,4) | 3        |

### Complexity

| Operation  | Complexity |
| ---------- | ---------- |
| Preprocess | O(n log n) |
| Query      | O(1)       |
| Space      | O(n log n) |

A Sparse Table turns repeated queries into instant lookups, your go-to tool when arrays are static and speed is king.

### 188 Mo's Algorithm

Mo's Algorithm is a clever offline technique for answering range queries on static arrays in approximately O((n + q)√n) time.
It's ideal when you have many queries like sum, distinct count, frequency, etc., but no updates.
Instead of recomputing each query, Mo's algorithm reuses results smartly by moving the range endpoints efficiently.

### What Problem Are We Solving?

We want to answer multiple range queries efficiently:

> Given an array `arr[0..n-1]` and `q` queries `[L, R]`,
> compute something like sum, count distinct, etc., for each range.

A naive approach is O(n) per query → O(nq) total.
Mo's algorithm cleverly orders queries to achieve O((n + q)√n) total time.

#### Example

Array: `[1, 2, 1, 3, 4, 2, 3]`
Queries:

1. `[0, 4]` → distinct = 4
2. `[1, 3]` → distinct = 3
3. `[2, 4]` → distinct = 3

### How It Works (Plain Language)

1. Divide array into blocks of size `√n`.
2. Sort queries by:

   * Block of L
   * R (within block)
3. Maintain a sliding window `[currL, currR)`:

   * Move endpoints left/right step-by-step
   * Update the answer incrementally
4. Store results per query index.

The sorting ensures minimal movement between consecutive queries.

#### Example

If √n = 3, queries sorted by block:

```
Block 0: [0,4], [1,3]
Block 1: [2,4]
```

You move pointers minimally:

- From [0,4] → [1,3] → [2,4]
- Reusing much of previous computation.

### Tiny Code (Distinct Count Example)

#### Python

```python
import math

def mos_algorithm(arr, queries):
    n = len(arr)
    q = len(queries)
    block_size = int(math.sqrt(n))

    # Sort queries
    queries = sorted(enumerate(queries), key=lambda x: (x[1][0] // block_size, x[1][1]))

    freq = {}
    currL, currR = 0, 0
    curr_ans = 0
    answers = [0]*q

    def add(x):
        nonlocal curr_ans
        freq[x] = freq.get(x, 0) + 1
        if freq[x] == 1:
            curr_ans += 1

    def remove(x):
        nonlocal curr_ans
        freq[x] -= 1
        if freq[x] == 0:
            curr_ans -= 1

    for idx, (L, R) in queries:
        while currL > L:
            currL -= 1
            add(arr[currL])
        while currR <= R:
            add(arr[currR])
            currR += 1
        while currL < L:
            remove(arr[currL])
            currL += 1
        while currR > R + 1:
            currR -= 1
            remove(arr[currR])
        answers[idx] = curr_ans

    return answers

# Example
arr = [1,2,1,3,4,2,3]
queries = [(0,4), (1,3), (2,4)]
print(mos_algorithm(arr, queries))  # [4,3,3]
```

Output:

```
$$4, 3, 3]
```

#### C (Structure and Idea)

```c
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#define MAXN 100000
#define MAXQ 100000

typedef struct { int L, R, idx; } Query;

int arr[MAXN], ans[MAXQ], freq[1000001];
int curr_ans = 0, block;

int cmp(const void* a, const void* b) {
    Query *x = (Query*)a, *y = (Query*)b;
    if (x->L / block != y->L / block) return x->L / block - y->L / block;
    return x->R - y->R;
}

void add(int x) { if (++freq[x] == 1) curr_ans++; }
void remove_(int x) { if (--freq[x] == 0) curr_ans--; }

int main() {
    int n = 7, q = 3;
    int arr_[] = {1,2,1,3,4,2,3};
    for (int i = 0; i < n; i++) arr[i] = arr_[i];

    Query queries[] = {{0,4,0},{1,3,1},{2,4,2}};
    block = sqrt(n);
    qsort(queries, q, sizeof(Query), cmp);

    int currL = 0, currR = 0;
    for (int i = 0; i < q; i++) {
        int L = queries[i].L, R = queries[i].R;
        while (currL > L) add(arr[--currL]);
        while (currR <= R) add(arr[currR++]);
        while (currL < L) remove_(arr[currL++]);
        while (currR > R+1) remove_(arr[--currR]);
        ans[queries[i].idx] = curr_ans;
    }

    for (int i = 0; i < q; i++) printf("%d ", ans[i]);
}
```

Output:

```
4 3 3
```

### Why It Matters

- Converts many range queries into near-linear total time
- Ideal for:

  * Sum / Count / Frequency queries
  * Distinct elements
  * GCD, XOR, etc. with associative properties
- Works on static arrays (no updates)

### Intuition

Mo's Algorithm is like sorting your errands by location.
By visiting nearby "blocks" first, you minimize travel time.
Here, "travel" = pointer movement.

### Try It Yourself

1. Run on `[1,2,3,4,5]` with queries `(0,2),(1,4),(2,4)`
2. Change to sum instead of distinct count
3. Visualize pointer movement
4. Experiment with block size variations
5. Add offline query index tracking
6. Try sqrt decomposition vs Mo's
7. Count frequency of max element per range
8. Mix different query types (still offline)
9. Add precomputed sqrt(n) block grouping
10. Use in competitive programming problems

### Test Cases

| Array           | Queries           | Output  |
| --------------- | ----------------- | ------- |
| [1,2,1,3,4,2,3] | (0,4),(1,3),(2,4) | [4,3,3] |
| [1,1,1,1]       | (0,3),(1,2)       | [1,1]   |
| [1,2,3,4,5]     | (0,2),(2,4)       | [3,3]   |

### Complexity

| Operation        | Complexity   |
| ---------------- | ------------ |
| Pre-sort Queries | O(q log q)   |
| Processing       | O((n + q)√n) |
| Space            | O(n)         |

Mo's Algorithm is your range-query workhorse, smartly ordered, block-based, and blazingly efficient for static datasets.

### 189 Sweep Line Range Search

A Sweep Line Algorithm is a geometric technique that processes events in a sorted order along one dimension, typically the x-axis, to solve range, interval, and overlap problems efficiently.
Think of it like dragging a vertical line across a 2D plane and updating active intervals as you go.

### What Problem Are We Solving?

We want to efficiently find:

- Which intervals overlap a point
- Which rectangles intersect
- How many shapes cover a region

A brute-force check for all pairs is O(n²).
A Sweep Line reduces it to O(n log n) by sorting and processing events incrementally.

#### Example

Rectangles:

```
R1: [1, 3], R2: [2, 5], R3: [4, 6]
```

Events (sorted by x):

```
x = 1: R1 starts  
x = 2: R2 starts  
x = 3: R1 ends  
x = 4: R3 starts  
x = 5: R2 ends  
x = 6: R3 ends
```

The active set changes as the line sweeps → track overlaps dynamically.

### How It Works (Plain Language)

1. Convert objects into events

   * Each interval/rectangle generates start and end events.
2. Sort all events by coordinate (x or y).
3. Sweep through events:

   * On start, add object to active set.
   * On end, remove object.
   * At each step, query active set for intersections, counts, etc.
4. Use balanced tree / set for active range maintenance.

#### Example Walkthrough

Intervals: `[1,3], [2,5], [4,6]`
Events:

```
(1, start), (2, start), (3, end), (4, start), (5, end), (6, end)
```

Step-by-step:

- At 1: add [1,3]
- At 2: add [2,5], overlap detected (1,3) ∩ (2,5)
- At 3: remove [1,3]
- At 4: add [4,6], overlap detected (2,5) ∩ (4,6)
- At 5: remove [2,5]
- At 6: remove [4,6]

Result: 2 overlapping pairs

### Tiny Code (Interval Overlaps)

#### Python

```python
def sweep_line_intervals(intervals):
    events = []
    for l, r in intervals:
        events.append((l, 'start'))
        events.append((r, 'end'))
    events.sort()

    active = 0
    overlaps = 0
    for pos, typ in events:
        if typ == 'start':
            overlaps += active  # count overlaps
            active += 1
        else:
            active -= 1
    return overlaps

intervals = [(1,3), (2,5), (4,6)]
print(sweep_line_intervals(intervals))  # 2 overlaps
```

Output:

```
2
```

#### C

```c
#include <stdio.h>
#include <stdlib.h>

typedef struct {
    int x;
    int type; // 1 = start, -1 = end
} Event;

int cmp(const void* a, const void* b) {
    Event *e1 = (Event*)a, *e2 = (Event*)b;
    if (e1->x == e2->x) return e1->type - e2->type;
    return e1->x - e2->x;
}

int main() {
    int intervals[][2] = {{1,3},{2,5},{4,6}};
    int n = 3;
    Event events[2*n];
    for (int i = 0; i < n; i++) {
        events[2*i] = (Event){intervals[i][0], 1};
        events[2*i+1] = (Event){intervals[i][1], -1};
    }
    qsort(events, 2*n, sizeof(Event), cmp);

    int active = 0, overlaps = 0;
    for (int i = 0; i < 2*n; i++) {
        if (events[i].type == 1) {
            overlaps += active;
            active++;
        } else {
            active--;
        }
    }
    printf("Total overlaps: %d\n", overlaps);
}
```

Output:

```
Total overlaps: 2
```

### Why It Matters

- Universal pattern in computational geometry:

  * Interval intersection counting
  * Rectangle overlap
  * Segment union length
  * Plane sweep algorithms (Voronoi, Convex Hull)
- Optimizes from O(n²) to O(n log n)

Used in:

- GIS (geographic data)
- Scheduling (conflict detection)
- Event simulation

### Intuition

Imagine sliding a vertical line across your data:
You only "see" the intervals currently active.
No need to look back, everything behind is already resolved.

### Try It Yourself

1. Count overlaps in `[1,4],[2,3],[5,6]`
2. Modify to compute max active intervals (peak concurrency)
3. Extend to rectangle intersections (sweep + segment tree)
4. Track total covered length
5. Combine with priority queue for dynamic ranges
6. Visualize on a timeline (scheduling conflicts)
7. Apply to meeting room allocation
8. Extend to 2D sweep (events sorted by x, active y-ranges)
9. Count overlaps per interval
10. Compare runtime to brute force

### Test Cases

| Intervals         | Overlaps |
| ----------------- | -------- |
| [1,3],[2,5],[4,6] | 2        |
| [1,2],[3,4]       | 0        |
| [1,4],[2,3],[3,5] | 3        |
| [1,5],[2,4],[3,6] | 3        |

### Complexity

| Operation   | Complexity |
| ----------- | ---------- |
| Sort Events | O(n log n) |
| Sweep       | O(n)       |
| Total       | O(n log n) |
| Space       | O(n)       |

A Sweep Line algorithm is your moving scanner, it sweeps across time or space, managing active elements, and revealing hidden overlaps in elegant, ordered fashion.

### 190 Ball Tree Nearest Neighbor

A Ball Tree is a hierarchical spatial data structure built from nested hyperspheres ("balls").
It organizes points into clusters based on distance, enabling efficient nearest neighbor and range queries in high-dimensional or non-Euclidean spaces.

While KD-Trees split by axis, Ball Trees split by distance, making them robust when dimensions increase or when the distance metric isn't axis-aligned.

### What Problem Are We Solving?

We want to efficiently:

- Find nearest neighbors for a query point
- Perform radius searches (all points within distance `r`)
- Handle high-dimensional data where KD-Trees degrade

Naive search is O(n) per query.
A Ball Tree improves to roughly O(log n) average.

#### Example

Points in 2D:

```
(2,3), (5,4), (9,6), (4,7), (8,1), (7,2)
```

Query: `(9,2)` → Nearest: `(8,1)`

Instead of splitting by x/y axis, the Ball Tree groups nearby points by distance from a center point (centroid or median).

### How It Works (Plain Language)

1. Build Tree recursively:

   * Choose a pivot (center) (often centroid or median).
   * Compute radius (max distance to any point in cluster).
   * Partition points into two subsets (inner vs outer ball).
   * Recursively build sub-balls.
2. Query:

   * Start at root ball.
   * Check if child balls could contain closer points.
   * Prune branches where `distance(center, query) - radius > best_dist`.

This yields logarithmic average behavior.

#### Example Tree (Simplified)

```
Ball(center=(6,4), radius=5)
├── Ball(center=(3,5), radius=2) → [(2,3),(4,7),(5,4)]
└── Ball(center=(8,3), radius=3) → [(7,2),(8,1),(9,6)]
```

Query `(9,2)`:

- Check root
- Compare both children
- Prune `(3,5)` (too far)
- Search `(8,3)` cluster → nearest `(8,1)`

### Tiny Code (2D Example)

#### Python

```python
from math import sqrt

class BallNode:
    def __init__(self, points):
        self.center = tuple(sum(x)/len(x) for x in zip(*points))
        self.radius = max(sqrt(sum((p[i]-self.center[i])2 for i in range(len(p)))) for p in points)
        self.points = points if len(points) <= 2 else None
        self.left = None
        self.right = None
        if len(points) > 2:
            points.sort(key=lambda p: sqrt(sum((p[i]-self.center[i])2 for i in range(len(p)))))
            mid = len(points)//2
            self.left = BallNode(points[:mid])
            self.right = BallNode(points[mid:])

def dist(a, b):
    return sqrt(sum((x - y)2 for x, y in zip(a, b)))

def nearest(node, target, best=None):
    if node is None:
        return best
    if node.points is not None:
        for p in node.points:
            if best is None or dist(p, target) < dist(best, target):
                best = p
        return best
    d_center = dist(node.center, target)
    candidates = []
    if d_center - node.radius <= dist(best, target) if best else True:
        candidates.append(node.left)
        candidates.append(node.right)
    for child in candidates:
        best = nearest(child, target, best)
    return best

points = [(2,3),(5,4),(9,6),(4,7),(8,1),(7,2)]
tree = BallNode(points)
print(nearest(tree, (9,2)))  # (8,1)
```

Output:

```
(8, 1)
```

#### C (Structure Idea)

```c
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

typedef struct Node {
    double cx, cy, radius;
    struct Node *left, *right;
    double (*points)[2];
    int count;
} Node;

// Building logic: compute centroid, radius, split by distance.
// Query logic: prune if (dist(center, query) - radius) > best.
```

Ball Trees are rarely implemented manually in C due to complexity; usually built into libraries (like scikit-learn).

### Why It Matters

- Works well for non-axis-aligned data
- Handles high-dimensional and non-Euclidean metrics (e.g. cosine)
- Used in:

  * KNN search
  * Clustering (Ball Tree K-Means)
  * Spatial databases
  * Machine learning (scikit-learn NearestNeighbors)

### Intuition

Imagine wrapping your dataset in nested bubbles.
Each bubble encloses a cluster.
To find the nearest neighbor, you only open bubbles that could possibly contain closer points.

### Try It Yourself

1. Build a Ball Tree for `(1,1),(2,2),(3,3),(10,10)`
2. Query `(2.5,2.5)`
3. Visualize nested "balls"
4. Change to 3D points
5. Use Manhattan distance instead of Euclidean
6. Compare performance vs KD-Tree
7. Track pruned branches count
8. Implement k-nearest neighbors
9. Add radius search (within distance `r`)
10. Handle duplicate points

### Test Cases

| Points                              | Query     | Nearest |
| ----------------------------------- | --------- | ------- |
| (2,3),(5,4),(9,6),(4,7),(8,1),(7,2) | (9,2)     | (8,1)   |
| (1,1),(2,2),(3,3)                   | (2.5,2.5) | (3,3)   |
| (0,0),(10,10)                       | (7,8)     | (10,10) |

### Complexity

| Operation     | Complexity |
| ------------- | ---------- |
| Build         | O(n log n) |
| Query (avg)   | O(log n)   |
| Query (worst) | O(n)       |
| Space         | O(n)       |

A Ball Tree is your spherical guide in high-dimensional space, splitting by proximity, not axes, to find neighbors fast and flexibly.

# Section 20. Search Optimizations and variants 

### 191 Binary Search with Tolerance

A Binary Search with Tolerance is a variant of binary search designed for real numbers or floating-point values where exact matches may not exist due to rounding or measurement error.

Instead of checking for strict equality (`==`), we accept a range of closeness, defined by a tolerance `ε` (epsilon).

### What Problem Are We Solving?

For integers, equality is exact.
For floating-point numbers, tiny rounding errors make equality unreliable:

```c
if (arr[mid] == x) // ❌ unreliable for floats
```

We instead test:

```c
if (fabs(arr[mid] - x) < epsilon) // ✅ tolerance-based match
```

This approach is essential for:

- Scientific computation
- Numerical analysis
- Approximations
- Root-finding
- Simulation and measurement data

### Example

Given sorted real values:

```
$$0.1, 0.2, 0.3000000001, 0.4, 0.5]
```

Searching for `0.3` with `epsilon = 1e-6`:

- `fabs(0.3000000001 - 0.3) < 1e-6` → found!

### Tiny Code

#### C Implementation

```c
#include <stdio.h>
#include <math.h>

int binary_search_tolerance(double arr[], int n, double target, double eps) {
    int low = 0, high = n - 1;
    while (low <= high) {
        int mid = (low + high) / 2;
        double diff = arr[mid] - target;
        if (fabs(diff) < eps)
            return mid;  // found within tolerance
        else if (diff < 0)
            low = mid + 1;
        else
            high = mid - 1;
    }
    return -1; // not found
}

int main() {
    double arr[] = {0.1, 0.2, 0.3000000001, 0.4, 0.5};
    int n = 5;
    double x = 0.3;
    int idx = binary_search_tolerance(arr, n, x, 1e-6);
    if (idx != -1)
        printf("Found %.6f at index %d\n", x, idx);
    else
        printf("Not found\n");
}
```

Output:

```
Found 0.300000 at index 2
```

#### Python Implementation

```python
def binary_search_tolerance(arr, x, eps=1e-6):
    lo, hi = 0, len(arr) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if abs(arr[mid] - x) < eps:
            return mid
        elif arr[mid] < x:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1

arr = [0.1, 0.2, 0.3000000001, 0.4, 0.5]
print(binary_search_tolerance(arr, 0.3))  # 2
```

### Why It Matters

- Avoids false negatives when comparing floats
- Handles round-off errors gracefully
- Useful in:

  * Root-finding
  * Floating-point datasets
  * Physics simulations
  * Numerical optimization

### Intuition

Binary search assumes exact comparison.
With floating-point numbers, "equal" often means "close enough."
`ε` defines your acceptable margin of error.

Think of it as:

> "If the difference is less than ε, consider it found."

### Try It Yourself

1. Use an array `[0.1, 0.2, 0.3, 0.4, 0.5]`
   Search for `0.3000001` with `ε = 1e-5`
2. Reduce `ε` → observe when search fails
3. Try negative numbers or decimals
4. Compare with integer binary search
5. Experiment with non-uniform spacing
6. Modify to find nearest value if not within ε
7. Visualize tolerance as a small band around target
8. Apply in root finding (`f(x) ≈ 0`)
9. Adjust ε dynamically based on scale
10. Measure precision loss with large floats

### Test Cases

| Array                    | Target   | Epsilon | Expected  |
| ------------------------ | -------- | ------- | --------- |
| [0.1, 0.2, 0.3000000001] | 0.3      | 1e-6    | Found     |
| [1.0, 2.0, 3.0]          | 2.000001 | 1e-5    | Found     |
| [1.0, 2.0, 3.0]          | 2.1      | 1e-5    | Not found |

### Complexity

| Step   | Complexity |
| ------ | ---------- |
| Search | O(log n)   |
| Space  | O(1)       |

A Binary Search with Tolerance is a small but essential upgrade when dealing with real numbers—because in floating-point land, "close enough" is often the truth.

### 192 Ternary Search

A Ternary Search is an algorithm for finding the maximum or minimum of a unimodal function—a function that increases up to a point and then decreases (or vice versa).
It is a divide-and-conquer method similar to binary search but splits the range into three parts each time.

### What Problem Are We Solving?

You have a function `f(x)` defined on an interval `[l, r]`, and you know it has one peak (or valley).
You want to find the x where `f(x)` is maximized (or minimized).

Unlike binary search (which searches for equality), ternary search searches for extremes.

### Example (Unimodal Function)

Let

```
f(x) = - (x - 2)^2 + 4
```

This is a parabola with a maximum at `x = 2`.

Ternary search gradually narrows the interval around the maximum:

1. Divide `[l, r]` into three parts
2. Evaluate `f(m1)` and `f(m2)`
3. Keep the side that contains the peak
4. Repeat until small enough

### How It Works (Step-by-Step)

| Step | Action                                                             |
| ---- | ------------------------------------------------------------------ |
| 1    | Pick two midpoints: `m1 = l + (r - l) / 3`, `m2 = r - (r - l) / 3` |
| 2    | Compare `f(m1)` and `f(m2)`                                        |
| 3    | If `f(m1) < f(m2)` → maximum is in `[m1, r]`                       |
| 4    | Else → maximum is in `[l, m2]`                                     |
| 5    | Repeat until `r - l` < ε                                           |

### Tiny Code

#### C Implementation (Maximization)

```c
#include <stdio.h>
#include <math.h>

double f(double x) {
    return -pow(x - 2, 2) + 4; // peak at x=2
}

double ternary_search(double l, double r, double eps) {
    while (r - l > eps) {
        double m1 = l + (r - l) / 3;
        double m2 = r - (r - l) / 3;
        if (f(m1) < f(m2))
            l = m1;
        else
            r = m2;
    }
    return (l + r) / 2; // approx peak
}

int main() {
    double l = 0, r = 4;
    double res = ternary_search(l, r, 1e-6);
    printf("Approx max at x = %.6f, f(x) = %.6f\n", res, f(res));
}
```

Output:

```
Approx max at x = 2.000000, f(x) = 4.000000
```

#### Python Implementation

```python
def f(x):
    return -(x - 2)2 + 4  # Peak at x=2

def ternary_search(l, r, eps=1e-6):
    while r - l > eps:
        m1 = l + (r - l) / 3
        m2 = r - (r - l) / 3
        if f(m1) < f(m2):
            l = m1
        else:
            r = m2
    return (l + r) / 2

res = ternary_search(0, 4)
print(f"Max at x = {res:.6f}, f(x) = {f(res):.6f}")
```

### Why It Matters

- Finds extrema (max/min) in continuous functions
- No derivative required (unlike calculus-based optimization)
- Works when:

  * `f(x)` is unimodal
  * Domain is continuous
  * You can evaluate `f(x)` cheaply

Used in:

- Mathematical optimization
- Machine learning hyperparameter tuning
- Geometry problems (e.g., closest distance)
- Physics simulations

### Try It Yourself

1. Try `f(x) = (x - 5)^2 + 1` (minimization)
2. Use interval `[0, 10]` and `eps = 1e-6`
3. Change `eps` to `1e-3` → observe faster but rougher result
4. Apply to distance between two moving points
5. Compare with binary search on derivative
6. Plot f(x) to visualize narrowing intervals
7. Switch condition to find minimum
8. Test with `f(x) = sin(x)` on `[0, π]`
9. Use integer search version for discrete arrays
10. Combine with golden section search for efficiency

### Test Cases

| Function            | Interval  | Expected | Type    |
| ------------------- | --------- | -------- | ------- |
| f(x) = -(x-2)^2 + 4 | [0, 4]    | x ≈ 2    | Maximum |
| f(x) = (x-5)^2 + 1  | [0, 10]   | x ≈ 5    | Minimum |
| f(x) = sin(x)       | [0, 3.14] | x ≈ 1.57 | Maximum |

### Complexity

| Metric | Value             |
| ------ | ----------------- |
| Time   | O(log((r - l)/ε)) |
| Space  | O(1)              |

A Ternary Search slices the search space into thirds—zooming in on the peak or valley with mathematical precision, no derivatives required.

### 193 Hash-Based Search

A Hash-Based Search uses a hash function to map keys directly to indices in a table, giving constant-time expected lookups.
Instead of scanning or comparing elements, it jumps straight to the bucket where the data should be.

### What Problem Are We Solving?

When searching in a large dataset, linear search is too slow (O(n)) and binary search needs sorted data (O(log n)).
Hash-based search lets you find, insert, or delete an item in O(1) average time, regardless of ordering.

It's the foundation of hash tables, hash maps, and dictionaries.

### Example (Simple Lookup)

Suppose you want to store and search for names quickly:

```
$$"Alice", "Bob", "Carol", "Dave"]
```

A hash function maps each name to an index:

```
hash("Alice") → 2
hash("Bob")   → 5
hash("Carol") → 1
```

You place each name in its corresponding slot. Searching becomes instant:

```
hash("Bob") = 5 → found!
```

### How It Works (Plain Language)

| Step | Action                                                                                                    |
| ---- | --------------------------------------------------------------------------------------------------------- |
| 1    | Compute hash(key) to get an index                                                                     |
| 2    | Look up the bucket at that index                                                                      |
| 3    | If multiple items hash to same bucket (collision), handle with a strategy (chaining, open addressing) |
| 4    | Compare keys if necessary                                                                                 |
| 5    | Return result                                                                                             |

### Tiny Code

#### C Implementation (with Linear Probing)

```c
#include <stdio.h>
#include <string.h>

#define SIZE 10

typedef struct {
    char key[20];
    int value;
    int used;
} Entry;

Entry table[SIZE];

int hash(char *key) {
    int h = 0;
    for (int i = 0; key[i]; i++)
        h = (h * 31 + key[i]) % SIZE;
    return h;
}

void insert(char *key, int value) {
    int h = hash(key);
    while (table[h].used) {
        if (strcmp(table[h].key, key) == 0) break;
        h = (h + 1) % SIZE;
    }
    strcpy(table[h].key, key);
    table[h].value = value;
    table[h].used = 1;
}

int search(char *key) {
    int h = hash(key), start = h;
    while (table[h].used) {
        if (strcmp(table[h].key, key) == 0)
            return table[h].value;
        h = (h + 1) % SIZE;
        if (h == start) break;
    }
    return -1; // not found
}

int main() {
    insert("Alice", 10);
    insert("Bob", 20);
    printf("Value for Bob: %d\n", search("Bob"));
}
```

Output:

```
Value for Bob: 20
```

#### Python Implementation (Built-in Dict)

```python
people = {"Alice": 10, "Bob": 20, "Carol": 30}

print("Bob" in people)       # True
print(people["Carol"])       # 30
```

Python's `dict` uses an optimized open addressing hash table.

### Why It Matters

- O(1) average lookup, insertion, and deletion
- No need for sorting
- Core to symbol tables, caches, dictionaries, compilers, and databases
- Can scale with resizing (rehashing)

### Try It Yourself

1. Implement hash table with chaining using linked lists
2. Replace linear probing with quadratic probing
3. Measure lookup time vs. linear search on same dataset
4. Insert keys with collisions, ensure correctness
5. Create custom hash for integers: `h(x) = x % m`
6. Observe performance as table fills up (load factor > 0.7)
7. Implement delete operation carefully
8. Resize table when load factor high
9. Try a poor hash function (like `h=1`) and measure slowdown
10. Compare with Python's built-in dict

### Test Cases

| Operation | Input         | Expected Output |
| --------- | ------------- | --------------- |
| Insert    | ("Alice", 10) | Stored          |
| Insert    | ("Bob", 20)   | Stored          |
| Search    | "Alice"       | 10              |
| Search    | "Eve"         | Not found       |
| Delete    | "Bob"         | Removed         |
| Search    | "Bob"         | Not found       |

### Complexity

| Metric | Average | Worst Case |
| ------ | ------- | ---------- |
| Search | O(1)    | O(n)       |
| Insert | O(1)    | O(n)       |
| Space  | O(n)    | O(n)       |

A Hash-Based Search is like a magic index, it jumps straight to the data you need, turning search into instant lookup.

### 194 Bloom Filter Lookup

A Bloom Filter is a probabilistic data structure that tells you if an element is definitely not in a set or possibly is.
It's super fast and memory efficient, but allows false positives (never false negatives).

### What Problem Are We Solving?

When working with huge datasets (like URLs, cache keys, or IDs), you may not want to store every element just to check membership.
Bloom Filters give you a fast O(1) check:

- "Is this element in my set?" → Maybe
- "Is it definitely not in my set?" → Yes

They're widely used in databases, network systems, and search engines (e.g., to skip disk lookups).

### Example (Cache Lookup)

You have a cache of 1 million items. Before hitting the database, you want to know:

> Should I even bother checking cache for this key?

Use a Bloom Filter to quickly tell if the key *could be* in cache.
If filter says "no," skip lookup entirely.

### How It Works (Plain Language)

| Step | Action                                                                  |
| ---- | ----------------------------------------------------------------------- |
| 1    | Create a bit array of size `m` (all zeros)                              |
| 2    | Choose `k` independent hash functions                                   |
| 3    | To insert an element: compute k hashes, set corresponding bits to 1 |
| 4    | To query an element: compute k hashes, check if all bits are 1      |
| 5    | If any bit = 0 → definitely not in set                                  |
| 6    | If all bits = 1 → possibly in set (false positive possible)             |

### Tiny Code

#### Python (Simple Bloom Filter)

```python
from hashlib import sha256

class BloomFilter:
    def __init__(self, size=1000, hash_count=3):
        self.size = size
        self.hash_count = hash_count
        self.bits = [0] * size

    def _hashes(self, item):
        for i in range(self.hash_count):
            h = int(sha256((item + str(i)).encode()).hexdigest(), 16)
            yield h % self.size

    def add(self, item):
        for h in self._hashes(item):
            self.bits[h] = 1

    def contains(self, item):
        return all(self.bits[h] for h in self._hashes(item))

# Example usage
bf = BloomFilter()
bf.add("Alice")
print(bf.contains("Alice"))  # True (probably)
print(bf.contains("Bob"))    # False (definitely not)
```

Output:

```
True
False
```

### Why It Matters

- Memory efficient for large sets
- No false negatives, if it says "no," you can trust it
- Used in caches, databases, distributed systems
- Reduces I/O by skipping non-existent entries

### Try It Yourself

1. Build a Bloom Filter with 1000 bits and 3 hash functions
2. Insert 100 random elements
3. Query 10 existing and 10 non-existing elements
4. Measure false positive rate
5. Experiment with different `m` and `k` values
6. Integrate into a simple cache simulation
7. Implement double hashing to reduce hash cost
8. Compare memory use vs. Python `set`
9. Add support for merging filters (bitwise OR)
10. Try a Counting Bloom Filter (to support deletions)

### Test Cases

| Input             | Expected Output        |
| ----------------- | ---------------------- |
| add("Alice")      | Bits set               |
| contains("Alice") | True (maybe)           |
| contains("Bob")   | False (definitely not) |
| add("Bob")        | Bits updated           |
| contains("Bob")   | True (maybe)           |

### Complexity

| Operation | Time | Space |
| --------- | ---- | ----- |
| Insert    | O(k) | O(m)  |
| Lookup    | O(k) | O(m)  |

k: number of hash functions
m: size of bit array

A Bloom Filter is like a polite doorman, it'll never wrongly turn you away, but it might let in a stranger once in a while.

### 195 Cuckoo Hash Search

A Cuckoo Hash Table is a clever hash-based structure that guarantees O(1) lookup while avoiding long probe chains.
It uses two hash functions and relocates existing keys when a collision occurs, just like a cuckoo bird kicking eggs out of a nest.

### What Problem Are We Solving?

Traditional hash tables can degrade to O(n) when collisions pile up.
Cuckoo hashing ensures constant-time lookups by guaranteeing every key has at most two possible positions.
If both are taken, it kicks out an existing key and re-inserts it elsewhere.

It's widely used in network routers, high-performance caches, and hash-based indexes.

### Example (Small Table)

Suppose you have 2 hash functions:

```
h1(x) = x % 3  
h2(x) = (x / 3) % 3
```

Insert keys: 5, 8, 11

| Key | h1 | h2 |
| --- | -- | -- |
| 5   | 2  | 1  |
| 8   | 2  | 2  |
| 11  | 2  | 1  |

When inserting 11, slot `2` is full, so we kick out 5 to its alternate position.
This continues until every key finds a home.

### How It Works (Plain Language)

| Step | Action                                                 |
| ---- | ------------------------------------------------------ |
| 1    | Compute two hash indices: `h1(key)`, `h2(key)`         |
| 2    | Try placing the key in `h1` slot                       |
| 3    | If occupied, evict existing key to its alternate slot  |
| 4    | Repeat up to a threshold (to prevent infinite loops)   |
| 5    | If full cycle detected → rehash with new functions |

Each key lives in either position `h1(key)` or `h2(key)`.

### Tiny Code

#### C Implementation (Simplified)

```c
#include <stdio.h>

#define SIZE 7

int table1[SIZE], table2[SIZE];

int h1(int key) { return key % SIZE; }
int h2(int key) { return (key / SIZE) % SIZE; }

void insert(int key, int depth) {
    if (depth > SIZE) return; // avoid infinite loop
    int pos1 = h1(key);
    if (table1[pos1] == 0) {
        table1[pos1] = key;
        return;
    }
    int displaced = table1[pos1];
    table1[pos1] = key;
    int pos2 = h2(displaced);
    if (table2[pos2] == 0)
        table2[pos2] = displaced;
    else
        insert(displaced, depth + 1);
}

int search(int key) {
    return table1[h1(key)] == key || table2[h2(key)] == key;
}

int main() {
    insert(10, 0);
    insert(20, 0);
    insert(30, 0);
    printf("Search 20: %s\n", search(20) ? "Found" : "Not Found");
}
```

Output:

```
Search 20: Found
```

#### Python Implementation

```python
SIZE = 7
table1 = [None] * SIZE
table2 = [None] * SIZE

def h1(x): return x % SIZE
def h2(x): return (x // SIZE) % SIZE

def insert(key, depth=0):
    if depth > SIZE:
        return False  # cycle detected
    pos1 = h1(key)
    if table1[pos1] is None:
        table1[pos1] = key
        return True
    key, table1[pos1] = table1[pos1], key
    pos2 = h2(key)
    if table2[pos2] is None:
        table2[pos2] = key
        return True
    return insert(key, depth + 1)

def search(key):
    return table1[h1(key)] == key or table2[h2(key)] == key

insert(10); insert(20); insert(30)
print("Search 20:", search(20))
```

### Why It Matters

- O(1) worst-case lookup (always two probes)
- Eliminates long collision chains
- Great for high-performance systems
- Deterministic position = easy debugging

### Try It Yourself

1. Insert numbers 1–10 and trace movements
2. Add detection for cycles → trigger rehash
3. Compare average probe count vs. linear probing
4. Implement delete(key) (mark slot empty)
5. Try resizing table dynamically
6. Use different hash functions for variety
7. Track load factor before rehashing
8. Store `(key, value)` pairs
9. Benchmark against chaining
10. Visualize movement paths during insertion

### Test Cases

| Operation | Input | Expected              |
| --------- | ----- | --------------------- |
| Insert    | 5     | Placed                |
| Insert    | 8     | Kicked 5, placed both |
| Search    | 5     | Found                 |
| Search    | 11    | Not Found             |
| Delete    | 8     | Removed               |
| Search    | 8     | Not Found             |

### Complexity

| Operation | Time         | Space |
| --------- | ------------ | ----- |
| Search    | O(1)         | O(n)  |
| Insert    | O(1) average | O(n)  |
| Delete    | O(1)         | O(n)  |

Cuckoo hashing is like musical chairs for keys, when one can't sit, it makes another stand up and move, but everyone eventually finds a seat.

### 196 Robin Hood Hashing

Robin Hood Hashing is an open addressing strategy that balances fairness in hash table lookups.
When two keys collide, the one that's traveled farther from its "home" index gets to stay, stealing the slot like Robin Hood, who took from the rich and gave to the poor.

### What Problem Are We Solving?

Standard linear probing can cause clusters, long runs of occupied slots that slow searches.
Robin Hood Hashing reduces variance in probe lengths, so every key has roughly equal access time.
This leads to more predictable performance, even at high load factors.

### Example (Collision Handling)

Suppose `hash(x) = x % 10`.
Insert `[10, 20, 30, 21]`.

| Key | Hash | Position      | Probe Distance |
| --- | ---- | ------------- | -------------- |
| 10  | 0    | 0             | 0              |
| 20  | 0    | 1             | 1              |
| 30  | 0    | 2             | 2              |
| 21  | 1    | 2 (collision) | 1              |

At position 2, `21` meets `30` (whose distance = 2).
Since `21`'s distance (1) is less, it keeps probing.
Robin Hood rule: if newcomer has greater or equal distance → swap.
This keeps distribution even.

### How It Works (Plain Language)

| Step | Action                                                                        |
| ---- | ----------------------------------------------------------------------------- |
| 1    | Compute hash = `key % table_size`                                             |
| 2    | If slot empty → place item                                                    |
| 3    | Else, compare probe distance with occupant's                              |
| 4    | If new item's distance ≥ current's → swap and continue probing displaced item |
| 5    | Repeat until inserted                                                         |

### Tiny Code

#### C Implementation (Simplified)

```c
#include <stdio.h>

#define SIZE 10

typedef struct {
    int key;
    int used;
    int distance;
} Entry;

Entry table[SIZE];

int hash(int key) { return key % SIZE; }

void insert(int key) {
    int index = hash(key);
    int dist = 0;
    while (table[index].used) {
        if (table[index].distance < dist) {
            int temp_key = table[index].key;
            int temp_dist = table[index].distance;
            table[index].key = key;
            table[index].distance = dist;
            key = temp_key;
            dist = temp_dist;
        }
        index = (index + 1) % SIZE;
        dist++;
    }
    table[index].key = key;
    table[index].distance = dist;
    table[index].used = 1;
}

int search(int key) {
    int index = hash(key);
    int dist = 0;
    while (table[index].used) {
        if (table[index].key == key) return 1;
        if (table[index].distance < dist) return 0;
        index = (index + 1) % SIZE;
        dist++;
    }
    return 0;
}

int main() {
    insert(10);
    insert(20);
    insert(30);
    printf("Search 20: %s\n", search(20) ? "Found" : "Not Found");
}
```

Output:

```
Search 20: Found
```

#### Python Implementation

```python
SIZE = 10
table = [None] * SIZE
distances = [0] * SIZE

def h(k): return k % SIZE

def insert(key):
    index = h(key)
    dist = 0
    while table[index] is not None:
        if distances[index] < dist:
            table[index], key = key, table[index]
            distances[index], dist = dist, distances[index]
        index = (index + 1) % SIZE
        dist += 1
    table[index], distances[index] = key, dist

def search(key):
    index = h(key)
    dist = 0
    while table[index] is not None:
        if table[index] == key:
            return True
        if distances[index] < dist:
            return False
        index = (index + 1) % SIZE
        dist += 1
    return False

insert(10); insert(20); insert(30)
print("Search 20:", search(20))
```

### Why It Matters

- Predictable lookup times, probe lengths nearly uniform
- Outperforms linear probing at high load
- Fewer long clusters, more stable table behavior
- Great for performance-critical systems

### Try It Yourself

1. Insert `[10, 20, 30, 21]` and trace swaps
2. Measure probe length variance vs. linear probing
3. Implement delete() with tombstone handling
4. Add resizing when load factor > 0.8
5. Compare performance with quadratic probing
6. Visualize table after 20 inserts
7. Experiment with different table sizes
8. Create histogram of probe lengths
9. Add key-value pair storage
10. Benchmark search time at 70%, 80%, 90% load

### Test Cases

| Operation | Input          | Expected                   |
| --------- | -------------- | -------------------------- |
| Insert    | 10, 20, 30, 21 | Evenly distributed         |
| Search    | 20             | Found                      |
| Search    | 40             | Not Found                  |
| Delete    | 10             | Removed                    |
| Insert    | 50             | Placed at optimal position |

### Complexity

| Operation | Time     | Space |
| --------- | -------- | ----- |
| Search    | O(1) avg | O(n)  |
| Insert    | O(1) avg | O(n)  |
| Delete    | O(1) avg | O(n)  |

Robin Hood Hashing keeps everyone honest, no key hoards fast access, and none are left wandering too far.

### 197 Jump Consistent Hashing

Jump Consistent Hashing is a lightweight, fast, and deterministic way to assign keys to buckets (servers, shards, or partitions) that minimizes remapping when the number of buckets changes.

It's designed for load balancing in distributed systems, like database shards or cache clusters.

### What Problem Are We Solving?

When scaling systems horizontally, you often need to assign keys (like user IDs) to buckets (like servers).
Naive methods (e.g. `key % N`) cause massive remapping when `N` changes.
Jump Consistent Hashing avoids that, only a small fraction of keys move when a bucket is added or removed.

This ensures stability and predictable redistribution, ideal for distributed caches (Memcached, Redis) and databases (Bigtable, Ceph).

### Example (Adding Buckets)

Suppose we have keys 1 to 6 and 3 buckets:

| Key | Bucket (3) |
| --- | ---------- |
| 1   | 2          |
| 2   | 0          |
| 3   | 2          |
| 4   | 1          |
| 5   | 0          |
| 6   | 2          |

When we add a new bucket (4 total), only a few keys change buckets, most stay where they are.
That's the magic of *consistency*.

### How It Works (Plain Language)

| Step | Action                                                 |
| ---- | ------------------------------------------------------ |
| 1    | Treat key as a 64-bit integer                          |
| 2    | Initialize `b = -1` and `j = 0`                        |
| 3    | While `j < num_buckets`, update:                       |
|      | `b = j`, `key = key * 2862933555777941757ULL + 1`      |
|      | `j = floor((b + 1) * (1LL << 31) / ((key >> 33) + 1))` |
| 4    | Return `b` as bucket index                             |

It uses integer arithmetic only, no tables, no storage, just math.

### Tiny Code

#### C Implementation

```c
#include <stdint.h>
#include <stdio.h>

int jump_consistent_hash(uint64_t key, int num_buckets) {
    int64_t b = -1, j = 0;
    while (j < num_buckets) {
        b = j;
        key = key * 2862933555777941757ULL + 1;
        j = (b + 1) * ((double)(1LL << 31) / ((key >> 33) + 1));
    }
    return (int)b;
}

int main() {
    for (uint64_t k = 1; k <= 6; k++)
        printf("Key %llu → Bucket %d\n", k, jump_consistent_hash(k, 3));
}
```

Output:

```
Key 1 → Bucket 2  
Key 2 → Bucket 0  
Key 3 → Bucket 2  
Key 4 → Bucket 1  
Key 5 → Bucket 0  
Key 6 → Bucket 2
```

#### Python Implementation

```python
def jump_hash(key, num_buckets):
    b, j = -1, 0
    while j < num_buckets:
        b = j
        key = key * 2862933555777941757 + 1
        j = int((b + 1) * (1 << 31) / ((key >> 33) + 1))
    return b

for k in range(1, 7):
    print(f"Key {k} → Bucket {jump_hash(k, 3)}")
```

### Why It Matters

- Stable distribution: minimal remapping on resize
- O(1) time, O(1) space
- Works great for sharding, load balancing, partitioning
- No need for external storage or ring structures (unlike consistent hashing rings)

### Try It Yourself

1. Assign 10 keys across 3 buckets
2. Add a 4th bucket and see which keys move
3. Compare with `key % N` approach
4. Try very large bucket counts (10k+)
5. Benchmark speed, notice it's almost constant
6. Integrate with a distributed cache simulation
7. Test uniformity (distribution of keys)
8. Add random seeds for per-service variation
9. Visualize redistribution pattern
10. Compare with Rendezvous Hashing

### Test Cases

| Input      | Buckets | Output                   |
| ---------- | ------- | ------------------------ |
| Key=1, N=3 | 3       | 2                        |
| Key=2, N=3 | 3       | 0                        |
| Key=3, N=3 | 3       | 2                        |
| Key=1, N=4 | 4       | 3 or 2 (depends on math) |

Only a fraction of keys remap when N changes.

### Complexity

| Operation | Time | Space |
| --------- | ---- | ----- |
| Lookup    | O(1) | O(1)  |
| Insert    | O(1) | O(1)  |

Jump Consistent Hashing is like a steady hand, even as your system grows, it keeps most keys right where they belong.

### 198 Prefix Search in Trie

A Trie (prefix tree) is a specialized tree structure that stores strings by their prefixes, enabling fast prefix-based lookups, ideal for autocomplete, dictionaries, and word search engines.

With a trie, searching for "app" instantly finds "apple", "apply", "appetite", etc.

### What Problem Are We Solving?

Traditional data structures like arrays or hash tables can't efficiently answer questions like:

- "List all words starting with `pre`"
- "Does any word start with `tri`?"

A trie organizes data by prefix paths, making such queries fast and natural, often O(k) where *k* is the prefix length.

### Example (Words: `app`, `apple`, `bat`)

The trie looks like this:

```
(root)
 ├─ a
 │  └─ p
 │     └─ p *
 │        └─ l
 │           └─ e *
 └─ b
    └─ a
       └─ t *
```

Stars (*) mark word endings.
Search "app" → found; list all completions → "app", "apple".

### How It Works (Plain Language)

| Step | Action                                                       |
| ---- | ------------------------------------------------------------ |
| 1    | Each node represents a character                         |
| 2    | A path from root to node represents a prefix         |
| 3    | When inserting, follow characters and create nodes as needed |
| 4    | Mark end of word when reaching last character            |
| 5    | To search prefix: traverse nodes character by character      |
| 6    | If all exist → prefix found; else → not found                |

### Tiny Code

#### C Implementation (Simplified)

```c
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

#define ALPHABET 26

typedef struct TrieNode {
    struct TrieNode *child[ALPHABET];
    bool isEnd;
} TrieNode;

TrieNode* newNode() {
    TrieNode* node = calloc(1, sizeof(TrieNode));
    node->isEnd = false;
    return node;
}

void insert(TrieNode *root, const char *word) {
    for (int i = 0; word[i]; i++) {
        int idx = word[i] - 'a';
        if (!root->child[idx]) root->child[idx] = newNode();
        root = root->child[idx];
    }
    root->isEnd = true;
}

bool startsWith(TrieNode *root, const char *prefix) {
    for (int i = 0; prefix[i]; i++) {
        int idx = prefix[i] - 'a';
        if (!root->child[idx]) return false;
        root = root->child[idx];
    }
    return true;
}

int main() {
    TrieNode *root = newNode();
    insert(root, "app");
    insert(root, "apple");
    printf("Starts with 'ap': %s\n", startsWith(root, "ap") ? "Yes" : "No");
}
```

Output:

```
Starts with 'ap': Yes
```

#### Python Implementation

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

    def starts_with(self, prefix):
        node = self.root
        for ch in prefix:
            if ch not in node.children:
                return False
            node = node.children[ch]
        return True

trie = Trie()
trie.insert("apple")
trie.insert("app")
print(trie.starts_with("ap"))  # True
```

### Why It Matters

- O(k) prefix lookup (k = prefix length)
- Perfect for autocomplete, spell checkers, search engines
- Stores shared prefixes efficiently
- Can be extended for frequency, weights, or wildcard matching

### Try It Yourself

1. Insert ["app", "apple", "apply", "apt"]
2. Search prefix "ap" → expect True
3. Search prefix "ba" → expect False
4. Add function to list all words starting with a prefix
5. Implement delete(word)
6. Add frequency count at each node
7. Support uppercase letters
8. Store end-of-word markers clearly
9. Compare memory use vs hash table
10. Extend to autocomplete feature returning top-N completions

### Test Cases

| Operation  | Input   | Output |
| ---------- | ------- | ------ |
| Insert     | "app"   | Stored |
| Insert     | "apple" | Stored |
| startsWith | "ap"    | True   |
| startsWith | "ba"    | False  |
| startsWith | "app"   | True   |

### Complexity

| Operation       | Time                     | Space |
| --------------- | ------------------------ | ----- |
| Insert          | O(k)                     | O(k)  |
| Search Prefix   | O(k)                     | O(1)  |
| Space (n words) | O(sum of all characters) |,     |

A Trie turns your data into a map of words, every branch is a path toward meaning, every prefix a shortcut to discovery.

### 199 Pattern Search in Suffix Array

A Suffix Array is a sorted list of all suffixes of a string.
It enables fast substring searches, perfect for pattern matching in text editors, DNA analysis, and search engines.

By combining it with binary search, you can find whether a pattern appears in a string in O(m log n) time.

### What Problem Are We Solving?

Given a large text `T` (like `"banana"`) and a pattern `P` (like `"ana"`), we want to quickly check if `P` exists in `T`.
Naive search takes O(nm) (compare every position).
A suffix array lets us search more efficiently by working on a pre-sorted list of suffixes.

### Example (Text = "banana")

List all suffixes and sort them:

| Index | Suffix |
| ----- | ------ |
| 0     | banana |
| 1     | anana  |
| 2     | nana   |
| 3     | ana    |
| 4     | na     |
| 5     | a      |

Sorted suffixes:

| SA Index | Suffix | Original Position |
| -------- | ------ | ----------------- |
| 5        | a      | 5                 |
| 3        | ana    | 3                 |
| 1        | anana  | 1                 |
| 0        | banana | 0                 |
| 4        | na     | 4                 |
| 2        | nana   | 2                 |

Now search `"ana"` using binary search over these sorted suffixes.

### How It Works (Plain Language)

| Step | Action                                                       |
| ---- | ------------------------------------------------------------ |
| 1    | Build suffix array = all suffixes sorted lexicographically   |
| 2    | Use binary search to find lower/upper bounds for pattern |
| 3    | Compare only `m` characters per comparison                   |
| 4    | If found → pattern occurs at suffix index                    |
| 5    | Otherwise → not in text                                      |

### Tiny Code

#### C Implementation (Simplified)

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int cmp(const void *a, const void *b, void *txt) {
    int i = *(int*)a, j = *(int*)b;
    return strcmp((char*)txt + i, (char*)txt + j);
}

void build_suffix_array(char *txt, int n, int sa[]) {
    for (int i = 0; i < n; i++) sa[i] = i;
    qsort_r(sa, n, sizeof(int), cmp, txt);
}

int binary_search_suffix(char *txt, int sa[], int n, char *pat) {
    int l = 0, r = n - 1;
    while (l <= r) {
        int mid = (l + r) / 2;
        int res = strncmp(pat, txt + sa[mid], strlen(pat));
        if (res == 0) return sa[mid];
        if (res < 0) r = mid - 1;
        else l = mid + 1;
    }
    return -1;
}

int main() {
    char txt[] = "banana";
    int n = strlen(txt), sa[n];
    build_suffix_array(txt, n, sa);
    char pat[] = "ana";
    int pos = binary_search_suffix(txt, sa, n, pat);
    if (pos >= 0) printf("Pattern found at %d\n", pos);
    else printf("Pattern not found\n");
}
```

Output:

```
Pattern found at 1
```

#### Python Implementation

```python
def build_suffix_array(s):
    return sorted(range(len(s)), key=lambda i: s[i:])

def search(s, sa, pat):
    l, r = 0, len(sa) - 1
    while l <= r:
        mid = (l + r) // 2
        suffix = s[sa[mid]:]
        if suffix.startswith(pat):
            return sa[mid]
        if suffix < pat:
            l = mid + 1
        else:
            r = mid - 1
    return -1

s = "banana"
sa = build_suffix_array(s)
print("Suffix Array:", sa)
print("Search 'ana':", search(s, sa, "ana"))
```

Output:

```
Suffix Array: [5, 3, 1, 0, 4, 2]
Search 'ana': 1
```

### Why It Matters

- Substring search in O(m log n)
- Space-efficient alternative to suffix trees
- Great for full-text search, DNA sequencing, plagiarism detection
- Can be extended with LCP array for longest common prefix queries

### Try It Yourself

1. Build suffix array for `"banana"`
2. Search `"na"`, `"ban"`, `"apple"`
3. Print all suffixes for visualization
4. Add LCP array for faster repeated queries
5. Compare with KMP algorithm speed
6. Use binary search manually to trace comparisons
7. Extend to count occurrences of pattern
8. Try longer text (e.g. "mississippi")
9. Implement in-place quicksort for SA
10. Benchmark vs. naive substring search

### Test Cases

| Text     | Pattern | Expected Result |
| -------- | ------- | --------------- |
| "banana" | "ana"   | Found at 1      |
| "banana" | "ban"   | Found at 0      |
| "banana" | "na"    | Found at 2 or 4 |
| "banana" | "cat"   | Not Found       |

### Complexity

| Operation | Time       | Space |
| --------- | ---------- | ----- |
| Build SA  | O(n log n) | O(n)  |
| Search    | O(m log n) | O(1)  |

A Suffix Array is like a library index, once sorted, every search becomes as simple as flipping to the right page.

### 200 Search in Infinite Array

A Search in Infinite Array (or unbounded array) is a technique for finding an element in a sorted list of unknown length.
You can't directly use binary search because you don't know `n`, so you must first find a search bound, then perform binary search within it.

This idea is crucial for systems where data is streamed or dynamically sized, like logs, unrolled lists, or file scans.

### What Problem Are We Solving?

If you're given an array-like interface (like `get(i)`), but no size `n`, how do you find `target` efficiently?
You can't do linear search, it could be infinite.
The trick: exponential search, grow bounds exponentially until you pass the target, then apply binary search in that window.

### Example

Given sorted sequence:

```
$$3, 5, 9, 12, 17, 23, 31, 45, 67, 88, 100, ...]
```

Search for `31`:

1. Start with `low = 0, high = 1`
2. While `arr[high] < 31`, double `high`

   ```
   high = 1 → 2 → 4 → 8
   ```
3. Now `arr[8] = 67 > 31`, so search range is `[4, 8]`
4. Perform binary search in `[4, 8]`
5. Found at index 6

### How It Works (Plain Language)

| Step | Action                                                    |
| ---- | --------------------------------------------------------- |
| 1    | Start with `low = 0`, `high = 1`                          |
| 2    | While `arr[high] < target`, set `low = high`, `high *= 2` |
| 3    | Now the target lies between `low` and `high`              |
| 4    | Perform standard binary search in `[low, high]`       |
| 5    | Return index if found, else -1                            |

### Tiny Code

#### C Implementation (Simulated Infinite Array)

```c
#include <stdio.h>

int get(int arr[], int size, int i) {
    if (i >= size) return 1e9; // simulate infinity
    return arr[i];
}

int binary_search(int arr[], int low, int high, int target, int size) {
    while (low <= high) {
        int mid = low + (high - low) / 2;
        int val = get(arr, size, mid);
        if (val == target) return mid;
        if (val < target) low = mid + 1;
        else high = mid - 1;
    }
    return -1;
}

int search_infinite(int arr[], int size, int target) {
    int low = 0, high = 1;
    while (get(arr, size, high) < target) {
        low = high;
        high *= 2;
    }
    return binary_search(arr, low, high, target, size);
}

int main() {
    int arr[] = {3, 5, 9, 12, 17, 23, 31, 45, 67, 88, 100};
    int size = sizeof(arr)/sizeof(arr[0]);
    int idx = search_infinite(arr, size, 31);
    printf("Found at index %d\n", idx);
}
```

Output:

```
Found at index 6
```

#### Python Implementation

```python
def get(arr, i):
    return arr[i] if i < len(arr) else float('inf')

def binary_search(arr, low, high, target):
    while low <= high:
        mid = (low + high) // 2
        val = get(arr, mid)
        if val == target:
            return mid
        if val < target:
            low = mid + 1
        else:
            high = mid - 1
    return -1

def search_infinite(arr, target):
    low, high = 0, 1
    while get(arr, high) < target:
        low = high
        high *= 2
    return binary_search(arr, low, high, target)

arr = [3, 5, 9, 12, 17, 23, 31, 45, 67, 88, 100]
print("Found at index:", search_infinite(arr, 31))
```

Output:

```
Found at index: 6
```

### Why It Matters

- Works with streams, linked storage, APIs, or infinite generators
- Avoids full traversal, logarithmic growth
- Combines exploration (finding bounds) with binary search (exact match)
- Ideal for search engines, log readers, cloud data paging

### Try It Yourself

1. Search `[1, 3, 5, 9, 12, 20]` for `9`
2. Search `[2, 4, 8, 16, 32, 64]` for `33` (not found)
3. Count number of `get()` calls, compare to linear search
4. Try `target` smaller than first element
5. Handle edge cases: empty array, target > max
6. Simulate infinite stream with `get()`
7. Replace doubling with `high = low + step` for adaptive growth
8. Visualize search window expansion
9. Generalize to descending arrays
10. Compare performance vs naive scan

### Test Cases

| Input Array                   | Target | Output |
| ----------------------------- | ------ | ------ |
| [3, 5, 9, 12, 17, 23, 31, 45] | 31     | 6      |
| [3, 5, 9, 12, 17, 23, 31, 45] | 4      | -1     |
| [1, 2, 4, 8, 16]              | 8      | 3      |
| [10, 20, 30]                  | 40     | -1     |

### Complexity

| Operation | Time     | Space |
| --------- | -------- | ----- |
| Search    | O(log p) | O(1)  |

*p* = position of target in array

Searching an infinite array feels like navigating a foggy road, first find your headlights (bounds), then drive straight to the target.



