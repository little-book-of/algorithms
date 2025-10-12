# Chapter 3. Data Structure in Action 

## Section 21. Arrays, linked lists, stacks, queues 

### 201 Dynamic Array Resize

Dynamic array resizing is how we keep push operations fast while growing storage as needed. The idea is simple: when the array is full, allocate a bigger one, copy the elements, and keep going. If we double the capacity on each resize, the average cost per push stays constant.

#### What Problem Are We Solving?

A fixed array has a fixed size. Real programs do not always know n in advance. We want an array that supports `push` at end with near constant time, grows automatically, and keeps elements in contiguous memory for cache friendliness.

Goal: Provide `push`, `pop`, `get`, `set` on a resizable array with amortized O(1) time for push.

#### How Does It Work (Plain Language)?

Keep two numbers: size and capacity.

1. Start with a small capacity (for example, 1 or 8).
2. On `push`, if `size < capacity`, write the element and increase `size`.
3. If `size == capacity`, allocate new storage with capacity doubled, copy old elements, release the old block, then push.
4. Optionally, if many `pop`s reduce `size` far below `capacity`, shrink by halving to avoid wasted space.

Why doubling?
Doubling keeps the number of costly resizes small. Most pushes are cheap writes. Only occasionally do we pay for copying.

Example Steps (Growth Simulation)

| Step | Size Before | Capacity Before | Action                                       | Size After | Capacity After |
| ---- | ----------- | --------------- | -------------------------------------------- | ---------- | -------------- |
| 1    | 0           | 1               | push(1)                                      | 1          | 1              |
| 2    | 1           | 1               | full → resize to 2, copy 1 element, push(2)  | 2          | 2              |
| 3    | 2           | 2               | full → resize to 4, copy 2 elements, push(3) | 3          | 4              |
| 4    | 3           | 4               | push(4)                                      | 4          | 4              |
| 5    | 4           | 4               | full → resize to 8, copy 4 elements, push(5) | 5          | 8              |

Notice how capacity doubles occasionally while most pushes cost O(1).

#### Tiny Code (Easy Versions)

C

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    size_t size;
    size_t capacity;
    int *data;
} DynArray;

int da_init(DynArray *a, size_t init_cap) {
    if (init_cap == 0) init_cap = 1;
    a->data = (int *)malloc(init_cap * sizeof(int));
    if (!a->data) return 0;
    a->size = 0;
    a->capacity = init_cap;
    return 1;
}

int da_resize(DynArray *a, size_t new_cap) {
    int *p = (int *)realloc(a->data, new_cap * sizeof(int));
    if (!p) return 0;
    a->data = p;
    a->capacity = new_cap;
    return 1;
}

int da_push(DynArray *a, int x) {
    if (a->size == a->capacity) {
        size_t new_cap = a->capacity * 2;
        if (!da_resize(a, new_cap)) return 0;
    }
    a->data[a->size++] = x;
    return 1;
}

int da_pop(DynArray *a, int *out) {
    if (a->size == 0) return 0;
    *out = a->data[--a->size];
    if (a->capacity > 1 && a->size <= a->capacity / 4) {
        size_t new_cap = a->capacity / 2;
        if (new_cap == 0) new_cap = 1;
        da_resize(a, new_cap);
    }
    return 1;
}

int main(void) {
    DynArray a;
    if (!da_init(&a, 1)) return 1;
    for (int i = 0; i < 10; ++i) da_push(&a, i);
    for (size_t i = 0; i < a.size; ++i) printf("%d ", a.data[i]);
    printf("\nsize=%zu cap=%zu\n", a.size, a.capacity);
    free(a.data);
    return 0;
}
```

Python

```python
class DynArray:
    def __init__(self, init_cap=1):
        self._cap = max(1, init_cap)
        self._n = 0
        self._data = [None] * self._cap

    def _resize(self, new_cap):
        new = [None] * new_cap
        for i in range(self._n):
            new[i] = self._data[i]
        self._data = new
        self._cap = new_cap

    def push(self, x):
        if self._n == self._cap:
            self._resize(self._cap * 2)
        self._data[self._n] = x
        self._n += 1

    def pop(self):
        if self._n == 0:
            raise IndexError("pop from empty DynArray")
        self._n -= 1
        x = self._data[self._n]
        self._data[self._n] = None
        if self._cap > 1 and self._n <= self._cap // 4:
            self._resize(max(1, self._cap // 2))
        return x

    def __getitem__(self, i):
        if not 0 <= i < self._n:
            raise IndexError("index out of range")
        return self._data[i]
```

#### Why It Matters

- Amortized O(1) push for dynamic growth
- Contiguous memory for cache performance
- Simple API that underlies vectors, array lists, and many scripting language arrays
- Balanced memory usage via optional shrinking

#### A Gentle Proof (Why It Works)

Accounting view for doubling
Charge each push a constant credit (e.g. 3 units).

1. A normal push costs 1 unit and stores 2 credits with the element.
2. When resizing from capacity k → 2k, copying k elements costs k units, paid by saved credits.
3. The new element pays its own 1 unit.

Thus amortized cost per push is O(1).

Growth factor choice
Any factor > 1 gives amortized O(1).

- Doubling → fewer resizes, more memory slack
- 1.5x → less slack, more frequent copies
- Too small (<1.2) → breaks amortized bound

#### Try It Yourself

1. Simulate pushes 1..32 for factors 2.0 and 1.5. Count resizes.
2. Add `reserve(n)` to preallocate capacity.
3. Implement shrinking when `size <= capacity / 4`.
4. Replace `int` with a struct, measure copy cost.
5. Use potential method: Φ = 2·size − capacity.

#### Test Cases

| Operation Sequence | Capacity Trace (Factor 2) | Notes               |
| ------------------ | ------------------------- | ------------------- |
| push 1..8          | 1 → 2 → 4 → 8             | resize at 1, 2, 4   |
| push 9             | 8 → 16                    | resize before write |
| push 10..16        | 16                        | no resize           |
| pop 16..9          | 16                        | shrinking optional  |
| pop 8              | 16 → 8                    | shrink at 25% load  |

Edge Cases

- Push on full array triggers one resize before write
- Pop on empty array should error or return false
- `reserve(n)` larger than current capacity must preserve data
- Shrink never reduces capacity below size

#### Complexity

- Time:

  * push amortized O(1)
  * push worst case O(n) on resize
  * pop amortized O(1)
  * get/set O(1)
- Space: O(n), capacity ≤ constant × size

Dynamic array resizing turns a rigid array into a flexible container. Grow when needed, copy occasionally, and enjoy constant-time pushes on average.

### 202 Circular Array Implementation

A circular array (or ring buffer) stores elements in a fixed-size array while allowing wrap-around indexing. It is perfect for implementing queues and buffers where old data is overwritten or processed in a first-in-first-out manner.

#### What Problem Are We Solving?

A regular array wastes space if we only move front and rear pointers forward. A circular array solves this by wrapping indices around when they reach the end, so all slots can be reused without shifting elements.

Goal:
Efficiently support enqueue, dequeue, peek, and size in O(1) time using a fixed-size buffer with wrap-around indexing.

#### How Does It Work (Plain Language)?

Maintain:

- `front`: index of the first element
- `rear`: index of the next free slot
- `count`: number of elements in the buffer
- `capacity`: maximum number of elements

Wrap-around indexing uses modulo arithmetic:

```text
next_index = (current_index + 1) % capacity
```

When adding or removing, always increment `front` or `rear` with this rule.

Example Steps (Wrap-around Simulation)

| Step | Operation   | Front | Rear | Count | Array State      | Note           |
| ---- | ----------- | ----- | ---- | ----- | ---------------- | -------------- |
| 1    | enqueue(10) | 0     | 1    | 1     | [10, _, _, _]    | rear advanced  |
| 2    | enqueue(20) | 0     | 2    | 2     | [10, 20, _, _]   |                |
| 3    | enqueue(30) | 0     | 3    | 3     | [10, 20, 30, _]  |                |
| 4    | dequeue()   | 1     | 3    | 2     | [_, 20, 30, _]   | front advanced |
| 5    | enqueue(40) | 1     | 0    | 3     | [40, 20, 30, _]  | wrap-around    |
| 6    | enqueue(50) | 1     | 1    | 4     | [40, 20, 30, 50] | full queue     |

#### Tiny Code (Easy Versions)

C

```c
#include <stdio.h>
#include <stdlib.h>

typedef struct {
    int *data;
    int front;
    int rear;
    int count;
    int capacity;
} CircularQueue;

int cq_init(CircularQueue *q, int capacity) {
    q->data = malloc(capacity * sizeof(int));
    if (!q->data) return 0;
    q->capacity = capacity;
    q->front = 0;
    q->rear = 0;
    q->count = 0;
    return 1;
}

int cq_enqueue(CircularQueue *q, int x) {
    if (q->count == q->capacity) return 0; // full
    q->data[q->rear] = x;
    q->rear = (q->rear + 1) % q->capacity;
    q->count++;
    return 1;
}

int cq_dequeue(CircularQueue *q, int *out) {
    if (q->count == 0) return 0; // empty
    *out = q->data[q->front];
    q->front = (q->front + 1) % q->capacity;
    q->count--;
    return 1;
}

int main(void) {
    CircularQueue q;
    cq_init(&q, 4);
    cq_enqueue(&q, 10);
    cq_enqueue(&q, 20);
    cq_enqueue(&q, 30);
    int val;
    cq_dequeue(&q, &val);
    cq_enqueue(&q, 40);
    cq_enqueue(&q, 50); // should fail if capacity=4
    printf("Front value: %d\n", q.data[q.front]);
    free(q.data);
    return 0;
}
```

Python

```python
class CircularQueue:
    def __init__(self, capacity):
        self._cap = capacity
        self._data = [None] * capacity
        self._front = 0
        self._rear = 0
        self._count = 0

    def enqueue(self, x):
        if self._count == self._cap:
            raise OverflowError("queue full")
        self._data[self._rear] = x
        self._rear = (self._rear + 1) % self._cap
        self._count += 1

    def dequeue(self):
        if self._count == 0:
            raise IndexError("queue empty")
        x = self._data[self._front]
        self._front = (self._front + 1) % self._cap
        self._count -= 1
        return x

    def peek(self):
        if self._count == 0:
            raise IndexError("queue empty")
        return self._data[self._front]
```

#### Why It Matters

- Enables constant time queue operations
- No element shifting needed
- Efficient use of fixed memory
- Backbone for circular buffers, task schedulers, streaming pipelines, and audio/video buffers

#### A Gentle Proof (Why It Works)

Modulo indexing ensures positions loop back when reaching array end.
If capacity is n, then all indices stay in `[0, n-1]`.
No overflow occurs since `front`, `rear` always wrap around.
The condition `count == capacity` detects full queue, and `count == 0` detects empty.

Thus each operation touches only one element and updates O(1) variables.

#### Try It Yourself

1. Implement a circular buffer with overwrite (old data replaced).
2. Add `is_full()` and `is_empty()` helpers.
3. Simulate producer-consumer with one enqueue per producer tick and one dequeue per consumer tick.
4. Extend to store structs instead of integers.

#### Test Cases

| Operation Sequence | Front | Rear | Count | Array State      | Notes       |
| ------------------ | ----- | ---- | ----- | ---------------- | ----------- |
| enqueue(10)        | 0     | 1    | 1     | [10, _, _, _]    | normal push |
| enqueue(20)        | 0     | 2    | 2     | [10, 20, _, _]   |             |
| enqueue(30)        | 0     | 3    | 3     | [10, 20, 30, _]  |             |
| dequeue()          | 1     | 3    | 2     | [_, 20, 30, _]   |             |
| enqueue(40)        | 1     | 0    | 3     | [40, 20, 30, _]  | wrap-around |
| enqueue(50)        | 1     | 1    | 4     | [40, 20, 30, 50] | full        |

Edge Cases

- Enqueue when full → error or overwrite
- Dequeue when empty → error
- Modulo indexing avoids overflow
- Works with any capacity ≥ 1

#### Complexity

- Time: O(1) for enqueue, dequeue, peek
- Space: O(n) fixed buffer

Circular arrays provide elegant constant-time queues with wrap-around indexing, a small trick that powers big systems.

### 203 Singly Linked List Insert/Delete

A singly linked list is a chain of nodes, each pointing to the next. It grows and shrinks dynamically without preallocating memory. Operations like insert and delete rely on pointer adjustments rather than shifting elements.

#### What Problem Are We Solving?

Static arrays are fixed-size and require shifting elements for insertions or deletions in the middle. Singly linked lists solve this by connecting elements through pointers, allowing efficient O(1) insertion and deletion (given a node reference).

Goal:
Support insert, delete, and traversal on a structure that can grow dynamically without reallocating or shifting elements.

#### How Does It Work (Plain Language)?

Each node stores two fields:

- `data` (the value)
- `next` (pointer to the next node)

The list has a `head` pointer to the first node.
Insertion and deletion work by updating the `next` pointers.

Example Steps (Insertion at Position)

| Step | Operation           | Node Updated | Before         | After          | Notes               |
| ---- | ------------------- | ------------ | -------------- | -------------- | ------------------- |
| 1    | create list         | -            | empty          | head = NULL    | list starts empty   |
| 2    | insert(10) at head  | new node     | NULL           | [10]           | head → 10           |
| 3    | insert(20) at head  | new node     | [10]           | [20 → 10]      | head updated        |
| 4    | insert(30) after 20 | new node     | [20 → 10]      | [20 → 30 → 10] | pointer rerouted    |
| 5    | delete(30)          | node 20      | [20 → 30 → 10] | [20 → 10]      | bypass removed node |

Key idea:
To insert, point the new node's `next` to the target's next, then link the target's `next` to the new node.
To delete, bypass the target node by linking previous node's `next` to the one after target.

#### Tiny Code (Easy Versions)

C

```c
#include <stdio.h>
#include <stdlib.h>

typedef struct Node {
    int data;
    struct Node *next;
} Node;

Node* insert_head(Node* head, int value) {
    Node* new_node = malloc(sizeof(Node));
    new_node->data = value;
    new_node->next = head;
    return new_node; // new head
}

Node* insert_after(Node* node, int value) {
    if (!node) return NULL;
    Node* new_node = malloc(sizeof(Node));
    new_node->data = value;
    new_node->next = node->next;
    node->next = new_node;
    return new_node;
}

Node* delete_value(Node* head, int value) {
    if (!head) return NULL;
    if (head->data == value) {
        Node* tmp = head->next;
        free(head);
        return tmp;
    }
    Node* prev = head;
    Node* cur = head->next;
    while (cur) {
        if (cur->data == value) {
            prev->next = cur->next;
            free(cur);
            break;
        }
        prev = cur;
        cur = cur->next;
    }
    return head;
}

void print_list(Node* head) {
    for (Node* p = head; p; p = p->next)
        printf("%d -> ", p->data);
    printf("NULL\n");
}

int main(void) {
    Node* head = NULL;
    head = insert_head(head, 10);
    head = insert_head(head, 20);
    insert_after(head, 30);
    print_list(head);
    head = delete_value(head, 30);
    print_list(head);
    return 0;
}
```

Python

```python
class Node:
    def __init__(self, data):
        self.data = data
        self.next = None

class LinkedList:
    def __init__(self):
        self.head = None

    def insert_head(self, data):
        node = Node(data)
        node.next = self.head
        self.head = node

    def insert_after(self, prev, data):
        if prev is None:
            return
        node = Node(data)
        node.next = prev.next
        prev.next = node

    def delete_value(self, value):
        cur = self.head
        prev = None
        while cur:
            if cur.data == value:
                if prev:
                    prev.next = cur.next
                else:
                    self.head = cur.next
                return
            prev = cur
            cur = cur.next

    def print_list(self):
        cur = self.head
        while cur:
            print(cur.data, end=" -> ")
            cur = cur.next
        print("NULL")
```

#### Why It Matters

- Provides dynamic growth with no preallocation
- Enables O(1) insertion and deletion at head or given position
- Useful in stacks, queues, hash table buckets, and adjacency lists
- Foundation for more complex data structures (trees, graphs)

#### A Gentle Proof (Why It Works)

Each operation only changes a constant number of pointers.

- Insert: 2 assignments → O(1)
- Delete: find node O(n), then 1 reassignment
  Because pointers are updated locally, the rest of the list remains valid.

#### Try It Yourself

1. Implement insert_tail for O(n) tail insertion.
2. Write reverse() to flip the list in place.
3. Add size() to count nodes.
4. Experiment with deleting from head and middle.

#### Test Cases

| Operation              | Input          | Result         | Notes           |
| ---------------------- | -------------- | -------------- | --------------- |
| insert_head(10)        | []             | [10]           | creates head    |
| insert_head(20)        | [10]           | [20 → 10]      | new head        |
| insert_after(head, 30) | [20 → 10]      | [20 → 30 → 10] | pointer reroute |
| delete_value(30)       | [20 → 30 → 10] | [20 → 10]      | node removed    |
| delete_value(40)       | [20 → 10]      | [20 → 10]      | no change       |

Edge Cases

- Delete from empty list → no effect
- Insert after NULL → no effect
- Delete head node → update head pointer

#### Complexity

- Time: insert O(1), delete O(n) (if searching by value), traversal O(n)
- Space: O(n) for n nodes

Singly linked lists teach pointer manipulation, where structure grows one node at a time, and every link matters.

### 204 Doubly Linked List Insert/Delete

A doubly linked list extends the singly linked list by adding backward links. Each node points to both its previous and next neighbor, making insertions and deletions easier in both directions.

#### What Problem Are We Solving?

Singly linked lists cannot traverse backward and deleting a node requires access to its predecessor. Doubly linked lists fix this by storing two pointers per node, allowing constant-time insertions and deletions at any position when you have a node reference.

Goal:
Support bidirectional traversal and efficient local insert/delete operations with O(1) pointer updates.

#### How Does It Work (Plain Language)?

Each node stores:

- `data`: the value
- `prev`: pointer to the previous node
- `next`: pointer to the next node

The list tracks two ends:

- `head` points to the first node
- `tail` points to the last node

Example Steps (Insertion and Deletion)

| Step | Operation        | Target | Before         | After          | Note               |
| ---- | ---------------- | ------ | -------------- | -------------- | ------------------ |
| 1    | create list      | -      | empty          | [10]           | head=tail=10       |
| 2    | insert_front(20) | head   | [10]           | [20 ⇄ 10]      | head updated       |
| 3    | insert_back(30)  | tail   | [20 ⇄ 10]      | [20 ⇄ 10 ⇄ 30] | tail updated       |
| 4    | delete(10)       | middle | [20 ⇄ 10 ⇄ 30] | [20 ⇄ 30]      | pointers bypass 10 |

Each operation only touches nearby nodes, no need to shift elements.

#### Tiny Code (Easy Versions)

C

```c
#include <stdio.h>
#include <stdlib.h>

typedef struct Node {
    int data;
    struct Node* prev;
    struct Node* next;
} Node;

Node* insert_front(Node* head, int value) {
    Node* node = malloc(sizeof(Node));
    node->data = value;
    node->prev = NULL;
    node->next = head;
    if (head) head->prev = node;
    return node;
}

Node* insert_back(Node* head, int value) {
    Node* node = malloc(sizeof(Node));
    node->data = value;
    node->next = NULL;
    if (!head) {
        node->prev = NULL;
        return node;
    }
    Node* tail = head;
    while (tail->next) tail = tail->next;
    tail->next = node;
    node->prev = tail;
    return head;
}

Node* delete_value(Node* head, int value) {
    Node* cur = head;
    while (cur && cur->data != value) cur = cur->next;
    if (!cur) return head; // not found
    if (cur->prev) cur->prev->next = cur->next;
    else head = cur->next; // deleting head
    if (cur->next) cur->next->prev = cur->prev;
    free(cur);
    return head;
}

void print_forward(Node* head) {
    for (Node* p = head; p; p = p->next) printf("%d ⇄ ", p->data);
    printf("NULL\n");
}
```

Python

```python
class Node:
    def __init__(self, data):
        self.data = data
        self.prev = None
        self.next = None

class DoublyLinkedList:
    def __init__(self):
        self.head = None

    def insert_front(self, data):
        node = Node(data)
        node.next = self.head
        if self.head:
            self.head.prev = node
        self.head = node

    def insert_back(self, data):
        node = Node(data)
        if not self.head:
            self.head = node
            return
        cur = self.head
        while cur.next:
            cur = cur.next
        cur.next = node
        node.prev = cur

    def delete_value(self, value):
        cur = self.head
        while cur:
            if cur.data == value:
                if cur.prev:
                    cur.prev.next = cur.next
                else:
                    self.head = cur.next
                if cur.next:
                    cur.next.prev = cur.prev
                return
            cur = cur.next

    def print_forward(self):
        cur = self.head
        while cur:
            print(cur.data, end=" ⇄ ")
            cur = cur.next
        print("NULL")
```

#### Why It Matters

- Bidirectional traversal (forward/backward)
- O(1) insertion and deletion when node is known
- Foundation for deque, LRU cache, and text editor buffers
- Enables clean list reversal and splice operations

#### A Gentle Proof (Why It Works)

Each node update modifies at most four pointers:

- Insert: new node's `next`, `prev` + neighbor links
- Delete: predecessor's `next`, successor's `prev`
  Since each step is constant work, operations are O(1) given node reference.

Traversal still costs O(n), but local edits are efficient.

#### Try It Yourself

1. Implement reverse() by swapping `prev` and `next` pointers.
2. Add tail pointer to allow O(1) insert_back.
3. Support bidirectional iteration.
4. Implement `pop_front()` and `pop_back()`.

#### Test Cases

| Operation        | Input          | Output         | Notes          |
| ---------------- | -------------- | -------------- | -------------- |
| insert_front(10) | []             | [10]           | head=tail=10   |
| insert_front(20) | [10]           | [20 ⇄ 10]      | new head       |
| insert_back(30)  | [20 ⇄ 10]      | [20 ⇄ 10 ⇄ 30] | new tail       |
| delete_value(10) | [20 ⇄ 10 ⇄ 30] | [20 ⇄ 30]      | middle removed |
| delete_value(20) | [20 ⇄ 30]      | [30]           | head removed   |

Edge Cases

- Deleting from empty list → no effect
- Inserting on empty list sets both head and tail
- Properly maintain both directions after each update

#### Complexity

- Time:

  * Insert/Delete (given node): O(1)
  * Search: O(n)
  * Traverse: O(n)
- Space: O(n) (2 pointers per node)

Doubly linked lists bring symmetry, move forward, move back, and edit in constant time.

### 205 Stack Push/Pop

A stack is a simple but powerful data structure that follows the LIFO (Last In, First Out) rule. The most recently added element is the first to be removed, like stacking plates where you can only take from the top.

#### What Problem Are We Solving?

We often need to reverse order or track nested operations, such as function calls, parentheses, undo/redo, and expression evaluation. A stack gives us a clean way to manage this behavior with push (add) and pop (remove).

Goal:
Support push, pop, peek, and is_empty in O(1) time, maintaining LIFO order.

#### How Does It Work (Plain Language)?

Think of a vertical pile.

- Push(x): Place x on top.
- Pop(): Remove the top element.
- Peek(): View top without removing.

Implementation can use either an array (fixed or dynamic) or a linked list.

Example Steps (Array Stack Simulation)

| Step | Operation | Stack (Top → Bottom) | Note          |
| ---- | --------- | -------------------- | ------------- |
| 1    | push(10)  | [10]                 | first element |
| 2    | push(20)  | [20, 10]             | top is 20     |
| 3    | push(30)  | [30, 20, 10]         |               |
| 4    | pop()     | [20, 10]             | 30 removed    |
| 5    | peek()    | top = 20             | top unchanged |

#### Tiny Code (Easy Versions)

C (Array-Based Stack)

```c
#include <stdio.h>
#include <stdlib.h>

typedef struct {
    int *data;
    int top;
    int capacity;
} Stack;

int stack_init(Stack *s, int cap) {
    s->data = malloc(cap * sizeof(int));
    if (!s->data) return 0;
    s->capacity = cap;
    s->top = -1;
    return 1;
}

int stack_push(Stack *s, int x) {
    if (s->top + 1 == s->capacity) return 0; // full
    s->data[++s->top] = x;
    return 1;
}

int stack_pop(Stack *s, int *out) {
    if (s->top == -1) return 0; // empty
    *out = s->data[s->top--];
    return 1;
}

int stack_peek(Stack *s, int *out) {
    if (s->top == -1) return 0;
    *out = s->data[s->top];
    return 1;
}

int main(void) {
    Stack s;
    stack_init(&s, 5);
    stack_push(&s, 10);
    stack_push(&s, 20);
    int val;
    stack_pop(&s, &val);
    printf("Popped: %d\n", val);
    stack_peek(&s, &val);
    printf("Top: %d\n", val);
    free(s.data);
    return 0;
}
```

Python (List as Stack)

```python
class Stack:
    def __init__(self):
        self._data = []

    def push(self, x):
        self._data.append(x)

    def pop(self):
        if not self._data:
            raise IndexError("pop from empty stack")
        return self._data.pop()

    def peek(self):
        if not self._data:
            raise IndexError("peek from empty stack")
        return self._data[-1]

    def is_empty(self):
        return len(self._data) == 0

# Example
s = Stack()
s.push(10)
s.push(20)
print(s.pop())  # 20
print(s.peek()) # 10
```

#### Why It Matters

- Core to recursion, parsing, and backtracking
- Underpins function call stacks in programming languages
- Natural structure for undo, reverse, and balanced parentheses
- Simplicity with wide applications in algorithms

#### A Gentle Proof (Why It Works)

Each operation touches only the top element or index.

- `push`: increment top, write value
- `pop`: read top, decrement top
  All O(1) time.
  The LIFO property ensures correct reverse order for function calls and nested scopes.

#### Try It Yourself

1. Implement stack with linked list nodes.
2. Extend capacity dynamically when full.
3. Use stack to check balanced parentheses in a string.
4. Reverse a list using stack operations.

#### Test Cases

| Operation | Input    | Output                 | Notes           |
| --------- | -------- | ---------------------- | --------------- |
| push(10)  | []       | [10]                   | top = 10        |
| push(20)  | [10]     | [20, 10]               | top = 20        |
| pop()     | [20, 10] | returns 20, stack [10] |                 |
| peek()    | [10]     | returns 10             |                 |
| pop()     | [10]     | returns 10, stack []   | empty after pop |

Edge Cases

- Pop from empty → error or return false
- Peek from empty → error
- Overflow if array full (unless dynamic)

#### Complexity

- Time: push O(1), pop O(1), peek O(1)
- Space: O(n) for n elements

Stacks embody disciplined memory, last in, first out, a minimalist model of control flow and history.

### 206 Queue Enqueue/Dequeue

A queue is the mirror twin of a stack. It follows FIFO (First In, First Out), the first element added is the first one removed, like people waiting in line.

#### What Problem Are We Solving?

We need a structure where items are processed in arrival order: scheduling tasks, buffering data, breadth-first search, or managing print jobs. A queue lets us enqueue at the back and dequeue from the front, simple and fair.

Goal:
Support enqueue, dequeue, peek, and is_empty in O(1) time using a circular layout or linked list.

#### How Does It Work (Plain Language)?

A queue has two ends:

- front → where items are removed
- rear → where items are added

Operations:

- `enqueue(x)` adds at `rear`
- `dequeue()` removes from `front`
- `peek()` looks at the `front` item

If implemented with a circular array, wrap indices using modulo arithmetic.

Example Steps (FIFO Simulation)

| Step | Operation   | Queue (Front → Rear) | Front | Rear | Note          |
| ---- | ----------- | -------------------- | ----- | ---- | ------------- |
| 1    | enqueue(10) | [10]                 | 0     | 1    | first element |
| 2    | enqueue(20) | [10, 20]             | 0     | 2    | rear advanced |
| 3    | enqueue(30) | [10, 20, 30]         | 0     | 3    |               |
| 4    | dequeue()   | [20, 30]             | 1     | 3    | 10 removed    |
| 5    | enqueue(40) | [20, 30, 40]         | 1     | 0    | wrap-around   |

#### Tiny Code (Easy Versions)

C (Circular Array Queue)

```c
#include <stdio.h>
#include <stdlib.h>

typedef struct {
    int *data;
    int front;
    int rear;
    int count;
    int capacity;
} Queue;

int queue_init(Queue *q, int cap) {
    q->data = malloc(cap * sizeof(int));
    if (!q->data) return 0;
    q->capacity = cap;
    q->front = 0;
    q->rear = 0;
    q->count = 0;
    return 1;
}

int enqueue(Queue *q, int x) {
    if (q->count == q->capacity) return 0; // full
    q->data[q->rear] = x;
    q->rear = (q->rear + 1) % q->capacity;
    q->count++;
    return 1;
}

int dequeue(Queue *q, int *out) {
    if (q->count == 0) return 0; // empty
    *out = q->data[q->front];
    q->front = (q->front + 1) % q->capacity;
    q->count--;
    return 1;
}

int queue_peek(Queue *q, int *out) {
    if (q->count == 0) return 0;
    *out = q->data[q->front];
    return 1;
}

int main(void) {
    Queue q;
    queue_init(&q, 4);
    enqueue(&q, 10);
    enqueue(&q, 20);
    enqueue(&q, 30);
    int val;
    dequeue(&q, &val);
    printf("Dequeued: %d\n", val);
    enqueue(&q, 40);
    enqueue(&q, 50); // will fail if capacity = 4
    queue_peek(&q, &val);
    printf("Front: %d\n", val);
    free(q.data);
    return 0;
}
```

Python (List as Queue using deque)

```python
from collections import deque

class Queue:
    def __init__(self):
        self._data = deque()

    def enqueue(self, x):
        self._data.append(x)

    def dequeue(self):
        if not self._data:
            raise IndexError("dequeue from empty queue")
        return self._data.popleft()

    def peek(self):
        if not self._data:
            raise IndexError("peek from empty queue")
        return self._data[0]

    def is_empty(self):
        return len(self._data) == 0

# Example
q = Queue()
q.enqueue(10)
q.enqueue(20)
print(q.dequeue())  # 10
print(q.peek())     # 20
```

#### Why It Matters

- Enforces fairness (first come, first served)
- Foundation for BFS, schedulers, buffers, pipelines
- Easy to implement and reason about
- Natural counterpart to stack

#### A Gentle Proof (Why It Works)

Each operation updates only front or rear index, not entire array.
Circular indexing ensures constant-time wrap-around:

```text
rear = (rear + 1) % capacity
front = (front + 1) % capacity
```

All operations touch O(1) data and fields, so runtime stays O(1).

#### Try It Yourself

1. Implement a linked-list-based queue.
2. Add `is_full()` and `is_empty()` checks.
3. Write a queue-based BFS on a simple graph.
4. Compare linear vs circular queue behavior.

#### Test Cases

| Operation   | Queue (Front → Rear) | Front | Rear | Count | Notes       |
| ----------- | -------------------- | ----- | ---- | ----- | ----------- |
| enqueue(10) | [10]                 | 0     | 1    | 1     |             |
| enqueue(20) | [10, 20]             | 0     | 2    | 2     |             |
| enqueue(30) | [10, 20, 30]         | 0     | 3    | 3     |             |
| dequeue()   | [20, 30]             | 1     | 3    | 2     | removes 10  |
| enqueue(40) | [20, 30, 40]         | 1     | 0    | 3     | wrap-around |
| peek()      | front=20             | 1     | 0    | 3     | check front |

Edge Cases

- Dequeue from empty → error
- Enqueue to full → overflow
- Works seamlessly with wrap-around

#### Complexity

- Time: enqueue O(1), dequeue O(1), peek O(1)
- Space: O(n) for fixed buffer or dynamic growth

Queues bring fairness to data, what goes in first comes out first, steady and predictable.

### 207 Deque Implementation

A deque (double-ended queue) is a flexible container that allows adding and removing elements from both ends, a blend of stack and queue behavior.

#### What Problem Are We Solving?

Stacks restrict you to one end, queues to two fixed roles. Sometimes we need both: insert at front or back, pop from either side. Deques power sliding window algorithms, palindrome checks, undo-redo systems, and task schedulers.

Goal:
Support `push_front`, `push_back`, `pop_front`, `pop_back`, and `peek_front/back` in O(1) time.

#### How Does It Work (Plain Language)?

Deques can be built using:

- A circular array (using wrap-around indexing)
- A doubly linked list (bidirectional pointers)

Operations:

- `push_front(x)` → insert before front
- `push_back(x)` → insert after rear
- `pop_front()` → remove front element
- `pop_back()` → remove rear element

Example Steps (Circular Array Simulation)

| Step | Operation     | Front | Rear | Deque (Front → Rear) | Note              |
| ---- | ------------- | ----- | ---- | -------------------- | ----------------- |
| 1    | push_back(10) | 0     | 1    | [10]                 | first item        |
| 2    | push_back(20) | 0     | 2    | [10, 20]             | rear grows        |
| 3    | push_front(5) | 3     | 2    | [5, 10, 20]          | wrap-around front |
| 4    | pop_back()    | 3     | 1    | [5, 10]              | 20 removed        |
| 5    | pop_front()   | 0     | 1    | [10]                 | 5 removed         |

#### Tiny Code (Easy Versions)

C (Circular Array Deque)

```c
#include <stdio.h>
#include <stdlib.h>

typedef struct {
    int *data;
    int front;
    int rear;
    int count;
    int capacity;
} Deque;

int dq_init(Deque *d, int cap) {
    d->data = malloc(cap * sizeof(int));
    if (!d->data) return 0;
    d->capacity = cap;
    d->front = 0;
    d->rear = 0;
    d->count = 0;
    return 1;
}

int dq_push_front(Deque *d, int x) {
    if (d->count == d->capacity) return 0; // full
    d->front = (d->front - 1 + d->capacity) % d->capacity;
    d->data[d->front] = x;
    d->count++;
    return 1;
}

int dq_push_back(Deque *d, int x) {
    if (d->count == d->capacity) return 0;
    d->data[d->rear] = x;
    d->rear = (d->rear + 1) % d->capacity;
    d->count++;
    return 1;
}

int dq_pop_front(Deque *d, int *out) {
    if (d->count == 0) return 0;
    *out = d->data[d->front];
    d->front = (d->front + 1) % d->capacity;
    d->count--;
    return 1;
}

int dq_pop_back(Deque *d, int *out) {
    if (d->count == 0) return 0;
    d->rear = (d->rear - 1 + d->capacity) % d->capacity;
    *out = d->data[d->rear];
    d->count--;
    return 1;
}

int main(void) {
    Deque d;
    dq_init(&d, 4);
    dq_push_back(&d, 10);
    dq_push_back(&d, 20);
    dq_push_front(&d, 5);
    int val;
    dq_pop_back(&d, &val);
    printf("Popped back: %d\n", val);
    dq_pop_front(&d, &val);
    printf("Popped front: %d\n", val);
    free(d.data);
    return 0;
}
```

Python (Deque using collections)

```python
from collections import deque

d = deque()
d.append(10)       # push_back
d.appendleft(5)    # push_front
d.append(20)
print(d.pop())     # pop_back -> 20
print(d.popleft()) # pop_front -> 5
print(d)           # deque([10])
```

#### Why It Matters

- Generalizes both stack and queue
- Core tool for sliding window maximum, palindrome checks, BFS with state, and task buffers
- Doubly-ended flexibility with constant-time operations
- Ideal for systems needing symmetric access

#### A Gentle Proof (Why It Works)

Circular indexing ensures wrap-around in O(1).
Each operation moves one index and changes one value:

- push → set value + adjust index
- pop → adjust index + read value

No shifting or reallocation needed, so all operations remain O(1).

#### Try It Yourself

1. Implement deque with a doubly linked list.
2. Add `peek_front()` and `peek_back()`.
3. Simulate a sliding window maximum algorithm.
4. Compare performance of deque vs list for queue-like tasks.

#### Test Cases

| Operation     | Front | Rear | Count | Deque (Front → Rear) | Notes      |
| ------------- | ----- | ---- | ----- | -------------------- | ---------- |
| push_back(10) | 0     | 1    | 1     | [10]                 | init       |
| push_back(20) | 0     | 2    | 2     | [10, 20]             |            |
| push_front(5) | 3     | 2    | 3     | [5, 10, 20]          | wrap front |
| pop_back()    | 3     | 1    | 2     | [5, 10]              | 20 removed |
| pop_front()   | 0     | 1    | 1     | [10]                 | 5 removed  |

Edge Cases

- Push on full deque → error or resize
- Pop on empty deque → error
- Wrap-around correctness is critical

#### Complexity

- Time: push/pop front/back O(1)
- Space: O(n)

Deques are the agile queues, you can act from either side, fast and fair.

### 208 Circular Queue

A circular queue is a queue optimized for fixed-size buffers where indices wrap around automatically. It's widely used in real-time systems, network packet buffers, and streaming pipelines to reuse space efficiently.

#### What Problem Are We Solving?

A linear queue wastes space after several dequeues, since front indices move forward. A circular queue solves this by wrapping the indices, making every slot reusable.

Goal:
Implement a queue with fixed capacity where enqueue and dequeue both take O(1) time and space is used cyclically.

#### How Does It Work (Plain Language)?

A circular queue keeps track of:

- `front`: index of the first element
- `rear`: index of the next position to insert
- `count`: number of elements

Use modulo arithmetic for wrap-around:

```text
next_index = (current_index + 1) % capacity
```

Key Conditions

- Full when `count == capacity`
- Empty when `count == 0`

Example Steps (Wrap-around Simulation)

| Step | Operation   | Front | Rear | Count | Queue State      | Note           |
| ---- | ----------- | ----- | ---- | ----- | ---------------- | -------------- |
| 1    | enqueue(10) | 0     | 1    | 1     | [10, _, _, _]    |                |
| 2    | enqueue(20) | 0     | 2    | 2     | [10, 20, _, _]   |                |
| 3    | enqueue(30) | 0     | 3    | 3     | [10, 20, 30, _]  |                |
| 4    | dequeue()   | 1     | 3    | 2     | [_, 20, 30, _]   | front advanced |
| 5    | enqueue(40) | 1     | 0    | 3     | [40, 20, 30, _]  | wrap-around    |
| 6    | enqueue(50) | 1     | 1    | 4     | [40, 20, 30, 50] | full           |

#### Tiny Code (Easy Versions)

C

```c
#include <stdio.h>
#include <stdlib.h>

typedef struct {
    int *data;
    int front;
    int rear;
    int count;
    int capacity;
} CircularQueue;

int cq_init(CircularQueue *q, int cap) {
    q->data = malloc(cap * sizeof(int));
    if (!q->data) return 0;
    q->capacity = cap;
    q->front = 0;
    q->rear = 0;
    q->count = 0;
    return 1;
}

int cq_enqueue(CircularQueue *q, int x) {
    if (q->count == q->capacity) return 0; // full
    q->data[q->rear] = x;
    q->rear = (q->rear + 1) % q->capacity;
    q->count++;
    return 1;
}

int cq_dequeue(CircularQueue *q, int *out) {
    if (q->count == 0) return 0; // empty
    *out = q->data[q->front];
    q->front = (q->front + 1) % q->capacity;
    q->count--;
    return 1;
}

int cq_peek(CircularQueue *q, int *out) {
    if (q->count == 0) return 0;
    *out = q->data[q->front];
    return 1;
}

int main(void) {
    CircularQueue q;
    cq_init(&q, 4);
    cq_enqueue(&q, 10);
    cq_enqueue(&q, 20);
    cq_enqueue(&q, 30);
    int val;
    cq_dequeue(&q, &val);
    printf("Dequeued: %d\n", val);
    cq_enqueue(&q, 40);
    cq_enqueue(&q, 50); // should fail if full
    cq_peek(&q, &val);
    printf("Front: %d\n", val);
    free(q.data);
    return 0;
}
```

Python

```python
class CircularQueue:
    def __init__(self, capacity):
        self._cap = capacity
        self._data = [None] * capacity
        self._front = 0
        self._rear = 0
        self._count = 0

    def enqueue(self, x):
        if self._count == self._cap:
            raise OverflowError("Queue full")
        self._data[self._rear] = x
        self._rear = (self._rear + 1) % self._cap
        self._count += 1

    def dequeue(self):
        if self._count == 0:
            raise IndexError("Queue empty")
        x = self._data[self._front]
        self._front = (self._front + 1) % self._cap
        self._count -= 1
        return x

    def peek(self):
        if self._count == 0:
            raise IndexError("Queue empty")
        return self._data[self._front]

# Example
q = CircularQueue(4)
q.enqueue(10)
q.enqueue(20)
q.enqueue(30)
print(q.dequeue())  # 10
q.enqueue(40)
print(q.peek())     # 20
```

#### Why It Matters

- Efficient space reuse, no wasted slots
- Predictable memory usage for real-time systems
- Backbone of buffering systems (audio, network, streaming)
- Fast O(1) operations, no shifting elements

#### A Gentle Proof (Why It Works)

Since both `front` and `rear` wrap using modulo, all operations remain in range `[0, capacity-1]`. Each operation modifies a fixed number of variables. Thus:

- enqueue: 1 write + 2 updates
- dequeue: 1 read + 2 updates
  No shifting, no resizing → all O(1) time.

#### Try It Yourself

1. Add `is_full()` and `is_empty()` helpers.
2. Implement an overwrite mode where new enqueues overwrite oldest data.
3. Visualize index movement for multiple wraps.
4. Use a circular queue to simulate a producer-consumer buffer.

#### Test Cases

| Operation   | Front | Rear | Count | Queue (Front → Rear) | Notes         |
| ----------- | ----- | ---- | ----- | -------------------- | ------------- |
| enqueue(10) | 0     | 1    | 1     | [10, _, _, _]        | first element |
| enqueue(20) | 0     | 2    | 2     | [10, 20, _, _]       |               |
| enqueue(30) | 0     | 3    | 3     | [10, 20, 30, _]      |               |
| dequeue()   | 1     | 3    | 2     | [_, 20, 30, _]       | 10 removed    |
| enqueue(40) | 1     | 0    | 3     | [40, 20, 30, _]      | wrap-around   |
| enqueue(50) | 1     | 1    | 4     | [40, 20, 30, 50]     | full queue    |

Edge Cases

- Enqueue when full → reject or overwrite
- Dequeue when empty → error
- Wrap-around indexing must handle 0 correctly

#### Complexity

- Time: enqueue O(1), dequeue O(1), peek O(1)
- Space: O(n) fixed buffer

Circular queues are the heartbeat of real-time data flows, steady, cyclic, and never wasting a byte.

### 209 Stack via Queue

A stack via queue is a playful twist, implementing LIFO behavior using FIFO tools. It shows how one structure can simulate another by combining basic operations cleverly.

#### What Problem Are We Solving?

Sometimes we're limited to queue operations (`enqueue`, `dequeue`) but still want a stack's last-in-first-out order. We can simulate `push` and `pop` using one or two queues.

Goal:
Build a stack that supports `push`, `pop`, and `peek` in O(1) or O(n) time (depending on strategy) using only queue operations.

#### How Does It Work (Plain Language)?

Two main strategies:

1. Push costly: rotate elements after every push so front is always top.
2. Pop costly: enqueue normally, but rotate during pop.

We'll show push costly version, simpler conceptually.

Idea:
Each `push` enqueues new item, then rotates all older elements behind it, so that last pushed is always at the front (ready to pop).

Example Steps (Push Costly)

| Step | Operation | Queue (Front → Rear) | Note                     |
| ---- | --------- | -------------------- | ------------------------ |
| 1    | push(10)  | [10]                 | only one element         |
| 2    | push(20)  | [20, 10]             | rotated so 20 is front   |
| 3    | push(30)  | [30, 20, 10]         | rotated again            |
| 4    | pop()     | [20, 10]             | 30 removed               |
| 5    | push(40)  | [40, 20, 10]         | rotation maintains order |

#### Tiny Code (Easy Versions)

C (Using Two Queues)

```c
#include <stdio.h>
#include <stdlib.h>

typedef struct {
    int *data;
    int front, rear, count, capacity;
} Queue;

int q_init(Queue *q, int cap) {
    q->data = malloc(cap * sizeof(int));
    if (!q->data) return 0;
    q->front = 0;
    q->rear = 0;
    q->count = 0;
    q->capacity = cap;
    return 1;
}

int q_enqueue(Queue *q, int x) {
    if (q->count == q->capacity) return 0;
    q->data[q->rear] = x;
    q->rear = (q->rear + 1) % q->capacity;
    q->count++;
    return 1;
}

int q_dequeue(Queue *q, int *out) {
    if (q->count == 0) return 0;
    *out = q->data[q->front];
    q->front = (q->front + 1) % q->capacity;
    q->count--;
    return 1;
}

typedef struct {
    Queue q1, q2;
} StackViaQueue;

int svq_init(StackViaQueue *s, int cap) {
    return q_init(&s->q1, cap) && q_init(&s->q2, cap);
}

int svq_push(StackViaQueue *s, int x) {
    q_enqueue(&s->q2, x);
    int val;
    while (s->q1.count) {
        q_dequeue(&s->q1, &val);
        q_enqueue(&s->q2, val);
    }
    // swap q1 and q2
    Queue tmp = s->q1;
    s->q1 = s->q2;
    s->q2 = tmp;
    return 1;
}

int svq_pop(StackViaQueue *s, int *out) {
    return q_dequeue(&s->q1, out);
}

int svq_peek(StackViaQueue *s, int *out) {
    if (s->q1.count == 0) return 0;
    *out = s->q1.data[s->q1.front];
    return 1;
}

int main(void) {
    StackViaQueue s;
    svq_init(&s, 10);
    svq_push(&s, 10);
    svq_push(&s, 20);
    svq_push(&s, 30);
    int val;
    svq_pop(&s, &val);
    printf("Popped: %d\n", val);
    svq_peek(&s, &val);
    printf("Top: %d\n", val);
    free(s.q1.data);
    free(s.q2.data);
    return 0;
}
```

Python

```python
from collections import deque

class StackViaQueue:
    def __init__(self):
        self.q = deque()

    def push(self, x):
        n = len(self.q)
        self.q.append(x)
        # rotate all older elements
        for _ in range(n):
            self.q.append(self.q.popleft())

    def pop(self):
        if not self.q:
            raise IndexError("pop from empty stack")
        return self.q.popleft()

    def peek(self):
        if not self.q:
            raise IndexError("peek from empty stack")
        return self.q[0]

# Example
s = StackViaQueue()
s.push(10)
s.push(20)
s.push(30)
print(s.pop())  # 30
print(s.peek()) # 20
```

#### Why It Matters

- Demonstrates duality of data structures (stack built on queue)
- Reinforces operation trade-offs (costly push vs costly pop)
- Great teaching example for algorithmic simulation
- Builds insight into complexity and resource usage

#### A Gentle Proof (Why It Works)

In push costly approach:

- Each push rotates all previous elements behind the new one → new element becomes front.
- Pop simply dequeues from front → correct LIFO order.

So order is maintained: newest always exits first.

#### Try It Yourself

1. Implement pop costly variant (push O(1), pop O(n)).
2. Add `is_empty()` helper.
3. Compare total number of operations for n pushes and pops.
4. Extend to generic types with structs or templates.

#### Test Cases

| Operation | Queue (Front → Rear) | Notes          |
| --------- | -------------------- | -------------- |
| push(10)  | [10]                 | single element |
| push(20)  | [20, 10]             | rotated        |
| push(30)  | [30, 20, 10]         | rotated        |
| pop()     | [20, 10]             | returns 30     |
| peek()    | [20, 10]             | returns 20     |

Edge Cases

- Pop/peek from empty → error
- Capacity reached → reject push
- Works even with single queue (rotation after push)

#### Complexity

- Push: O(n)
- Pop/Peek: O(1)
- Space: O(n)

Stack via queue proves constraints breed creativity, same data, different dance.

### 210 Queue via Stack

A queue via stack flips the story: build FIFO behavior using LIFO tools. It's a classic exercise in algorithmic inversion, showing how fundamental operations can emulate each other with clever order manipulation.

#### What Problem Are We Solving?

Suppose you only have stacks (with `push`, `pop`, `peek`) but need queue behavior (with `enqueue`, `dequeue`). We want to process items in arrival order, first in, first out, even though stacks operate last in, first out.

Goal:
Implement `enqueue`, `dequeue`, and `peek` for a queue using only stack operations.

#### How Does It Work (Plain Language)?

Two stacks are enough:

- inbox: where we push new items (enqueue)
- outbox: where we pop old items (dequeue)

When dequeuing, if `outbox` is empty, we move all items from `inbox` to `outbox`, reversing their order so oldest items are on top.

This reversal step restores FIFO behavior.

Example Steps (Two-Stack Method)

| Step | Operation   | inbox (Top → Bottom) | outbox (Top → Bottom) | Note                |
| ---- | ----------- | -------------------- | --------------------- | ------------------- |
| 1    | enqueue(10) | [10]                 | []                    |                     |
| 2    | enqueue(20) | [20, 10]             | []                    |                     |
| 3    | enqueue(30) | [30, 20, 10]         | []                    |                     |
| 4    | dequeue()   | []                   | [10, 20, 30]          | transfer + pop(10)  |
| 5    | enqueue(40) | [40]                 | [20, 30]              | mixed state         |
| 6    | dequeue()   | [40]                 | [30]                  | pop(20) from outbox |

#### Tiny Code (Easy Versions)

C (Using Two Stacks)

```c
#include <stdio.h>
#include <stdlib.h>

typedef struct {
    int *data;
    int top;
    int capacity;
} Stack;

int stack_init(Stack *s, int cap) {
    s->data = malloc(cap * sizeof(int));
    if (!s->data) return 0;
    s->top = -1;
    s->capacity = cap;
    return 1;
}

int stack_push(Stack *s, int x) {
    if (s->top + 1 == s->capacity) return 0;
    s->data[++s->top] = x;
    return 1;
}

int stack_pop(Stack *s, int *out) {
    if (s->top == -1) return 0;
    *out = s->data[s->top--];
    return 1;
}

int stack_peek(Stack *s, int *out) {
    if (s->top == -1) return 0;
    *out = s->data[s->top];
    return 1;
}

int stack_empty(Stack *s) {
    return s->top == -1;
}

typedef struct {
    Stack in, out;
} QueueViaStack;

int qvs_init(QueueViaStack *q, int cap) {
    return stack_init(&q->in, cap) && stack_init(&q->out, cap);
}

int qvs_enqueue(QueueViaStack *q, int x) {
    return stack_push(&q->in, x);
}

int qvs_shift(QueueViaStack *q) {
    int val;
    while (!stack_empty(&q->in)) {
        stack_pop(&q->in, &val);
        stack_push(&q->out, val);
    }
    return 1;
}

int qvs_dequeue(QueueViaStack *q, int *out) {
    if (stack_empty(&q->out)) qvs_shift(q);
    return stack_pop(&q->out, out);
}

int qvs_peek(QueueViaStack *q, int *out) {
    if (stack_empty(&q->out)) qvs_shift(q);
    return stack_peek(&q->out, out);
}

int main(void) {
    QueueViaStack q;
    qvs_init(&q, 10);
    qvs_enqueue(&q, 10);
    qvs_enqueue(&q, 20);
    qvs_enqueue(&q, 30);
    int val;
    qvs_dequeue(&q, &val);
    printf("Dequeued: %d\n", val);
    qvs_enqueue(&q, 40);
    qvs_peek(&q, &val);
    printf("Front: %d\n", val);
    free(q.in.data);
    free(q.out.data);
    return 0;
}
```

Python (Two Stack Queue)

```python
class QueueViaStack:
    def __init__(self):
        self.inbox = []
        self.outbox = []

    def enqueue(self, x):
        self.inbox.append(x)

    def dequeue(self):
        if not self.outbox:
            while self.inbox:
                self.outbox.append(self.inbox.pop())
        if not self.outbox:
            raise IndexError("dequeue from empty queue")
        return self.outbox.pop()

    def peek(self):
        if not self.outbox:
            while self.inbox:
                self.outbox.append(self.inbox.pop())
        if not self.outbox:
            raise IndexError("peek from empty queue")
        return self.outbox[-1]

# Example
q = QueueViaStack()
q.enqueue(10)
q.enqueue(20)
q.enqueue(30)
print(q.dequeue())  # 10
q.enqueue(40)
print(q.peek())     # 20
```

#### Why It Matters

- Demonstrates queue emulation with stack operations
- Core teaching example for data structure duality
- Helps in designing abstract interfaces under constraints
- Underpins some streaming and buffering systems

#### A Gentle Proof (Why It Works)

Each transfer (`inbox → outbox`) reverses order once, restoring FIFO sequence.

- Enqueue pushes to inbox (LIFO)
- Dequeue pops from outbox (LIFO of reversed)
  So overall effect = FIFO

Transfers happen only when outbox is empty, so amortized cost per operation is O(1).

#### Try It Yourself

1. Implement a single-stack recursive version.
2. Add `is_empty()` helper.
3. Measure amortized vs worst-case complexity.
4. Extend to generic data type.

#### Test Cases

| Operation   | inbox (Top→Bottom) | outbox (Top→Bottom) | Result     | Notes          |
| ----------- | ------------------ | ------------------- | ---------- | -------------- |
| enqueue(10) | [10]               | []                  |            |                |
| enqueue(20) | [20, 10]           | []                  |            |                |
| enqueue(30) | [30, 20, 10]       | []                  |            |                |
| dequeue()   | []                 | [10, 20, 30]        | returns 10 | transfer + pop |
| enqueue(40) | [40]               | [20, 30]            |            |                |
| dequeue()   | [40]               | [30]                | returns 20 |                |

Edge Cases

- Dequeue from empty queue → error
- Multiple dequeues trigger one transfer
- Outbox reused efficiently

#### Complexity

- Time: amortized O(1) per operation (worst-case O(n) on transfer)
- Space: O(n)

Queue via stack shows symmetry, turn LIFO into FIFO with one clever reversal.

## Section 22. Hash Tables and Variants 

### 211 Hash Table Insertion

A hash table stores key-value pairs for lightning-fast lookups. It uses a hash function to map keys to array indices, letting us access data in near-constant time.

#### What Problem Are We Solving?

We need a data structure that can insert, search, and delete by key efficiently, without scanning every element. Arrays give random access by index; hash tables extend that power to arbitrary keys.

Goal:
Map each key to a slot via a hash function and resolve any collisions gracefully.

#### How Does It Work (Plain Language)?

A hash table uses a hash function to convert a key into an index:

```text
index = hash(key) % capacity
```

When inserting a new (key, value):

1. Compute hash index.
2. If slot is empty → place pair there.
3. If occupied → handle collision (chaining or open addressing).

We'll use separate chaining (linked list per slot) as the simplest method.

Example Steps (Separate Chaining)

| Step | Key      | Hash(key) | Index | Action                       |
| ---- | -------- | --------- | ----- | ---------------------------- |
| 1    | "apple"  | 42        | 2     | Insert ("apple", 10)         |
| 2    | "banana" | 15        | 3     | Insert ("banana", 20)        |
| 3    | "pear"   | 18        | 2     | Collision → chain in index 2 |
| 4    | "peach"  | 21        | 1     | Insert new pair              |

Table after insertions:

| Index | Chain                        |
| ----- | ---------------------------- |
| 0     | -                            |
| 1     | ("peach", 40)                |
| 2     | ("apple", 10) → ("pear", 30) |
| 3     | ("banana", 20)               |

#### Tiny Code (Easy Versions)

C (Separate Chaining Example)

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TABLE_SIZE 5

typedef struct Node {
    char *key;
    int value;
    struct Node *next;
} Node;

typedef struct {
    Node *buckets[TABLE_SIZE];
} HashTable;

unsigned int hash(const char *key) {
    unsigned int h = 0;
    while (*key) h = h * 31 + *key++;
    return h % TABLE_SIZE;
}

HashTable* ht_create() {
    HashTable *ht = malloc(sizeof(HashTable));
    for (int i = 0; i < TABLE_SIZE; i++) ht->buckets[i] = NULL;
    return ht;
}

void ht_insert(HashTable *ht, const char *key, int value) {
    unsigned int idx = hash(key);
    Node *node = ht->buckets[idx];
    while (node) {
        if (strcmp(node->key, key) == 0) { node->value = value; return; }
        node = node->next;
    }
    Node *new_node = malloc(sizeof(Node));
    new_node->key = strdup(key);
    new_node->value = value;
    new_node->next = ht->buckets[idx];
    ht->buckets[idx] = new_node;
}

int ht_search(HashTable *ht, const char *key, int *out) {
    unsigned int idx = hash(key);
    Node *node = ht->buckets[idx];
    while (node) {
        if (strcmp(node->key, key) == 0) { *out = node->value; return 1; }
        node = node->next;
    }
    return 0;
}

int main(void) {
    HashTable *ht = ht_create();
    ht_insert(ht, "apple", 10);
    ht_insert(ht, "pear", 30);
    ht_insert(ht, "banana", 20);
    int val;
    if (ht_search(ht, "pear", &val))
        printf("pear: %d\n", val);
    return 0;
}
```

Python (Dictionary Simulation)

```python
class HashTable:
    def __init__(self, size=5):
        self.size = size
        self.table = [[] for _ in range(size)]

    def _hash(self, key):
        return hash(key) % self.size

    def insert(self, key, value):
        idx = self._hash(key)
        for pair in self.table[idx]:
            if pair[0] == key:
                pair[1] = value
                return
        self.table[idx].append([key, value])

    def search(self, key):
        idx = self._hash(key)
        for k, v in self.table[idx]:
            if k == key:
                return v
        return None

# Example
ht = HashTable()
ht.insert("apple", 10)
ht.insert("pear", 30)
ht.insert("banana", 20)
print(ht.search("pear"))  # 30
```

#### Why It Matters

- Provides average O(1) access, insert, delete
- Backbone of symbol tables, maps, sets, and dictionaries
- Used in caches, compilers, and indexing systems
- Introduces key idea: hash function + collision handling

#### A Gentle Proof (Why It Works)

Let table size = m, number of keys = n.
If hash spreads keys uniformly, expected chain length = α = n/m (load factor).

- Average lookup: O(1 + α)
- Keep α ≤ 1 → near constant time
  If collisions are minimized, operations stay fast.

#### Try It Yourself

1. Implement update to modify existing key's value.
2. Add delete(key) to remove entries from chain.
3. Experiment with different hash functions (e.g. djb2, FNV-1a).
4. Measure time vs load factor.

#### Test Cases

| Operation | Key      | Value | Result  | Notes      |
| --------- | -------- | ----- | ------- | ---------- |
| insert    | "apple"  | 10    | success | new key    |
| insert    | "banana" | 20    | success | new key    |
| insert    | "apple"  | 15    | update  | key exists |
| search    | "banana" |       | 20      | found      |
| search    | "grape"  |       | None    | not found  |

Edge Cases

- Insert duplicate → update value
- Search non-existent → return None
- Table full (open addressing) → needs rehash

#### Complexity

- Time: average O(1), worst O(n) (all collide)
- Space: O(n + m) (keys + table slots)

Hash table insertion is the art of turning chaos into order, hash, map, resolve, and store.

### 212 Linear Probing

Linear probing is one of the simplest collision resolution strategies in open addressing hash tables. When a collision occurs, it looks for the next empty slot by moving step by step through the table, wrapping around if needed.

#### What Problem Are We Solving?

When two keys hash to the same index, where do we store the new one?
Instead of chaining nodes, linear probing searches the next available slot, keeping all data inside the array.

Goal:
Resolve collisions by scanning linearly from the point of conflict until an empty slot is found.

#### How Does It Work (Plain Language)?

When inserting a key:

1. Compute index = `hash(key) % capacity`.
2. If slot empty → place key there.
3. If occupied → move to `(index + 1) % capacity`.
4. Repeat until an empty slot is found or table is full.

Lookups and deletions follow the same probe sequence until key is found or empty slot encountered.

Example Steps (Capacity = 7)

| Step | Key | Hash(key) | Index | Action                   |
| ---- | --- | --------- | ----- | ------------------------ |
| 1    | 10  | 3         | 3     | Place at index 3         |
| 2    | 24  | 3         | 4     | Collision → move to 4    |
| 3    | 31  | 3         | 5     | Collision → move to 5    |
| 4    | 17  | 3         | 6     | Collision → move to 6    |
| 5    | 38  | 3         | 0     | Wrap around → place at 0 |

Final Table

| Index | Value |
| ----- | ----- |
| 0     | 38    |
| 1     | -     |
| 2     | -     |
| 3     | 10    |
| 4     | 24    |
| 5     | 31    |
| 6     | 17    |

#### Tiny Code (Easy Versions)

C

```c
#include <stdio.h>
#include <stdlib.h>

#define CAPACITY 7
#define EMPTY -1
#define DELETED -2

typedef struct {
    int *table;
} HashTable;

int hash(int key) { return key % CAPACITY; }

HashTable* ht_create() {
    HashTable *ht = malloc(sizeof(HashTable));
    ht->table = malloc(sizeof(int) * CAPACITY);
    for (int i = 0; i < CAPACITY; i++) ht->table[i] = EMPTY;
    return ht;
}

void ht_insert(HashTable *ht, int key) {
    int idx = hash(key);
    for (int i = 0; i < CAPACITY; i++) {
        int pos = (idx + i) % CAPACITY;
        if (ht->table[pos] == EMPTY || ht->table[pos] == DELETED) {
            ht->table[pos] = key;
            return;
        }
    }
    printf("Table full, cannot insert %d\n", key);
}

int ht_search(HashTable *ht, int key) {
    int idx = hash(key);
    for (int i = 0; i < CAPACITY; i++) {
        int pos = (idx + i) % CAPACITY;
        if (ht->table[pos] == EMPTY) return -1;
        if (ht->table[pos] == key) return pos;
    }
    return -1;
}

void ht_delete(HashTable *ht, int key) {
    int pos = ht_search(ht, key);
    if (pos != -1) ht->table[pos] = DELETED;
}

int main(void) {
    HashTable *ht = ht_create();
    ht_insert(ht, 10);
    ht_insert(ht, 24);
    ht_insert(ht, 31);
    ht_insert(ht, 17);
    ht_insert(ht, 38);
    for (int i = 0; i < CAPACITY; i++)
        printf("[%d] = %d\n", i, ht->table[i]);
    return 0;
}
```

Python

```python
class LinearProbingHash:
    def __init__(self, size=7):
        self.size = size
        self.table = [None] * size

    def _hash(self, key):
        return key % self.size

    def insert(self, key):
        idx = self._hash(key)
        for i in range(self.size):
            pos = (idx + i) % self.size
            if self.table[pos] is None or self.table[pos] == "DELETED":
                self.table[pos] = key
                return
        raise OverflowError("Hash table full")

    def search(self, key):
        idx = self._hash(key)
        for i in range(self.size):
            pos = (idx + i) % self.size
            if self.table[pos] is None:
                return None
            if self.table[pos] == key:
                return pos
        return None

    def delete(self, key):
        pos = self.search(key)
        if pos is not None:
            self.table[pos] = "DELETED"

# Example
ht = LinearProbingHash()
for k in [10, 24, 31, 17, 38]:
    ht.insert(k)
print(ht.table)
print("Search 17 at:", ht.search(17))
```

#### Why It Matters

- Simplest form of open addressing
- Keeps all entries inside one array (no extra memory)
- Excellent cache performance
- Forms the basis for modern in-place hash maps

#### A Gentle Proof (Why It Works)

Every key follows the same probe sequence during insert, search, and delete.
So if a key is in the table, search will find it; if it's not, search will hit an empty slot and stop.
Uniform hashing ensures average probe length ≈ 1 / (1 - α), where α = n / m is load factor.

#### Try It Yourself

1. Implement `resize()` when load factor > 0.7.
2. Test insertion order and wrap-around behavior.
3. Compare linear probing vs chaining performance.
4. Visualize clustering as load increases.

#### Test Cases

| Operation | Key | Result       | Notes         |
| --------- | --- | ------------ | ------------- |
| insert    | 10  | index 3      | no collision  |
| insert    | 24  | index 4      | move 1 slot   |
| insert    | 31  | index 5      | move 2 slots  |
| insert    | 17  | index 6      | move 3 slots  |
| insert    | 38  | index 0      | wrap-around   |
| search    | 17  | found at 6   | linear search |
| delete    | 24  | mark deleted | slot reusable |

Edge Cases

- Table full → insertion fails
- Deleted slots reused
- Must stop on EMPTY, not DELETED

#### Complexity

- Time:

  * Average O(1) if α small
  * Worst O(n) if full cluster
- Space: O(n)

Linear probing walks straight lines through collisions, simple, local, and fast when load is low.

### 213 Quadratic Probing

Quadratic probing improves upon linear probing by reducing primary clustering. Instead of stepping through every slot one by one, it jumps in quadratic increments, spreading colliding keys more evenly across the table.

#### What Problem Are We Solving?

In linear probing, consecutive occupied slots cause clustering, leading to long probe chains and degraded performance.
Quadratic probing breaks up these runs by using nonlinear probe sequences.

Goal:
Resolve collisions by checking indices offset by quadratic values, `+1², +2², +3², …`, reducing clustering while keeping predictable probe order.

#### How Does It Work (Plain Language)?

When inserting a key:

1. Compute `index = hash(key) % capacity`.
2. If slot empty → insert.
3. If occupied → try `(index + 1²) % capacity`, `(index + 2²) % capacity`, etc.
4. Continue until an empty slot is found or table is full.

Lookups and deletions follow the same probe sequence.

Example Steps (Capacity = 7)

| Step | Key | Hash(key) | Probes (sequence) | Final Slot |
| ---- | --- | --------- | ----------------- | ---------- |
| 1    | 10  | 3         | 3                 | 3          |
| 2    | 24  | 3         | 3, 4              | 4          |
| 3    | 31  | 3         | 3, 4, 0           | 0          |
| 4    | 17  | 3         | 3, 4, 0, 2        | 2          |

Table after insertions:

| Index | Value |
| ----- | ----- |
| 0     | 31    |
| 1     | -     |
| 2     | 17    |
| 3     | 10    |
| 4     | 24    |
| 5     | -     |
| 6     | -     |

#### Tiny Code (Easy Versions)

C

```c
#include <stdio.h>
#include <stdlib.h>

#define CAPACITY 7
#define EMPTY -1
#define DELETED -2

typedef struct {
    int *table;
} HashTable;

int hash(int key) { return key % CAPACITY; }

HashTable* ht_create() {
    HashTable *ht = malloc(sizeof(HashTable));
    ht->table = malloc(sizeof(int) * CAPACITY);
    for (int i = 0; i < CAPACITY; i++) ht->table[i] = EMPTY;
    return ht;
}

void ht_insert(HashTable *ht, int key) {
    int idx = hash(key);
    for (int i = 0; i < CAPACITY; i++) {
        int pos = (idx + i * i) % CAPACITY;
        if (ht->table[pos] == EMPTY || ht->table[pos] == DELETED) {
            ht->table[pos] = key;
            return;
        }
    }
    printf("Table full, cannot insert %d\n", key);
}

int ht_search(HashTable *ht, int key) {
    int idx = hash(key);
    for (int i = 0; i < CAPACITY; i++) {
        int pos = (idx + i * i) % CAPACITY;
        if (ht->table[pos] == EMPTY) return -1;
        if (ht->table[pos] == key) return pos;
    }
    return -1;
}

int main(void) {
    HashTable *ht = ht_create();
    ht_insert(ht, 10);
    ht_insert(ht, 24);
    ht_insert(ht, 31);
    ht_insert(ht, 17);
    for (int i = 0; i < CAPACITY; i++)
        printf("[%d] = %d\n", i, ht->table[i]);
    return 0;
}
```

Python

```python
class QuadraticProbingHash:
    def __init__(self, size=7):
        self.size = size
        self.table = [None] * size

    def _hash(self, key):
        return key % self.size

    def insert(self, key):
        idx = self._hash(key)
        for i in range(self.size):
            pos = (idx + i * i) % self.size
            if self.table[pos] is None or self.table[pos] == "DELETED":
                self.table[pos] = key
                return
        raise OverflowError("Hash table full")

    def search(self, key):
        idx = self._hash(key)
        for i in range(self.size):
            pos = (idx + i * i) % self.size
            if self.table[pos] is None:
                return None
            if self.table[pos] == key:
                return pos
        return None

# Example
ht = QuadraticProbingHash()
for k in [10, 24, 31, 17]:
    ht.insert(k)
print(ht.table)
print("Search 24 at:", ht.search(24))
```

#### Why It Matters

- Reduces primary clustering seen in linear probing
- Keeps keys more evenly distributed
- Avoids extra pointers (everything stays in one array)
- Useful in hash tables where space is tight and locality helps

#### A Gentle Proof (Why It Works)

Probe sequence:
$$
i = 0, 1, 2, 3, \ldots
$$
Index at step *i* is
$$
(index + i^2) \bmod m
$$
If table size *m* is prime, this sequence visits up to ⌈m/2⌉ distinct slots before repeating, guaranteeing an empty slot is found if load factor < 0.5.

Thus all operations follow predictable, finite sequences.

#### Try It Yourself

1. Compare clustering with linear probing under same keys.
2. Experiment with different table sizes (prime vs composite).
3. Implement deletion markers properly.
4. Visualize probe paths with small tables.

#### Test Cases

| Operation | Key | Probe Sequence | Slot  | Notes          |
| --------- | --- | -------------- | ----- | -------------- |
| insert    | 10  | 3              | 3     | no collision   |
| insert    | 24  | 3, 4           | 4     | 1 step         |
| insert    | 31  | 3, 4, 0        | 0     | 2 steps        |
| insert    | 17  | 3, 4, 0, 2     | 2     | 3 steps        |
| search    | 31  | 3 → 4 → 0      | found | quadratic path |

Edge Cases

- Table full → insertion fails
- Requires table size prime for full coverage
- Needs load factor < 0.5 to avoid infinite loops

#### Complexity

- Time: average O(1), worst O(n)
- Space: O(n)

Quadratic probing trades a straight line for a curve, spreading collisions smoothly across the table.

### 214 Double Hashing

Double hashing uses two independent hash functions to minimize collisions. When a conflict occurs, it jumps forward by a second hash value, creating probe sequences unique to each key and greatly reducing clustering.

#### What Problem Are We Solving?

Linear and quadratic probing both suffer from clustering patterns, especially when keys share similar initial indices. Double hashing breaks this pattern by introducing a second hash function that defines each key's step size.

Goal:
Use two hash functions to determine probe sequence:
$$
\text{index}_i = (h_1(key) + i \cdot h_2(key)) \bmod m
$$
This produces independent probe paths and avoids overlap among keys.

#### How Does It Work (Plain Language)?

When inserting a key:

1. Compute primary hash: `h1 = key % capacity`.
2. Compute step size: `h2 = 1 + (key % (capacity - 1))` (never zero).
3. Try `h1`; if occupied, try `(h1 + h2) % m`, `(h1 + 2*h2) % m`, etc.
4. Repeat until an empty slot is found.

Same pattern applies for search and delete.

Example Steps (Capacity = 7)

| Step | Key | h₁(key) | h₂(key) | Probe Sequence | Final Slot |
| ---- | --- | ------- | ------- | -------------- | ---------- |
| 1    | 10  | 3       | 4       | 3              | 3          |
| 2    | 24  | 3       | 4       | 3 → 0          | 0          |
| 3    | 31  | 3       | 4       | 3 → 0 → 4      | 4          |
| 4    | 17  | 3       | 3       | 3 → 6          | 6          |

Final Table

| Index | Value |
| ----- | ----- |
| 0     | 24    |
| 1     | -     |
| 2     | -     |
| 3     | 10    |
| 4     | 31    |
| 5     | -     |
| 6     | 17    |

#### Tiny Code (Easy Versions)

C

```c
#include <stdio.h>
#include <stdlib.h>

#define CAPACITY 7
#define EMPTY -1
#define DELETED -2

typedef struct {
    int *table;
} HashTable;

int h1(int key) { return key % CAPACITY; }
int h2(int key) { return 1 + (key % (CAPACITY - 1)); }

HashTable* ht_create() {
    HashTable *ht = malloc(sizeof(HashTable));
    ht->table = malloc(sizeof(int) * CAPACITY);
    for (int i = 0; i < CAPACITY; i++) ht->table[i] = EMPTY;
    return ht;
}

void ht_insert(HashTable *ht, int key) {
    int idx1 = h1(key);
    int step = h2(key);
    for (int i = 0; i < CAPACITY; i++) {
        int pos = (idx1 + i * step) % CAPACITY;
        if (ht->table[pos] == EMPTY || ht->table[pos] == DELETED) {
            ht->table[pos] = key;
            return;
        }
    }
    printf("Table full, cannot insert %d\n", key);
}

int ht_search(HashTable *ht, int key) {
    int idx1 = h1(key), step = h2(key);
    for (int i = 0; i < CAPACITY; i++) {
        int pos = (idx1 + i * step) % CAPACITY;
        if (ht->table[pos] == EMPTY) return -1;
        if (ht->table[pos] == key) return pos;
    }
    return -1;
}

int main(void) {
    HashTable *ht = ht_create();
    ht_insert(ht, 10);
    ht_insert(ht, 24);
    ht_insert(ht, 31);
    ht_insert(ht, 17);
    for (int i = 0; i < CAPACITY; i++)
        printf("[%d] = %d\n", i, ht->table[i]);
    return 0;
}
```

Python

```python
class DoubleHash:
    def __init__(self, size=7):
        self.size = size
        self.table = [None] * size

    def _h1(self, key):
        return key % self.size

    def _h2(self, key):
        return 1 + (key % (self.size - 1))

    def insert(self, key):
        h1 = self._h1(key)
        h2 = self._h2(key)
        for i in range(self.size):
            pos = (h1 + i * h2) % self.size
            if self.table[pos] is None or self.table[pos] == "DELETED":
                self.table[pos] = key
                return
        raise OverflowError("Hash table full")

    def search(self, key):
        h1 = self._h1(key)
        h2 = self._h2(key)
        for i in range(self.size):
            pos = (h1 + i * h2) % self.size
            if self.table[pos] is None:
                return None
            if self.table[pos] == key:
                return pos
        return None

# Example
ht = DoubleHash()
for k in [10, 24, 31, 17]:
    ht.insert(k)
print(ht.table)
print("Search 24 at:", ht.search(24))
```

#### Why It Matters

- Minimizes primary and secondary clustering
- Probe sequences depend on key, not shared among colliding keys
- Achieves uniform distribution when both hash functions are good
- Forms basis for high-performance open addressing maps

#### A Gentle Proof (Why It Works)

If capacity m is prime and h₂(key) never equals 0,
then each key generates a unique probe sequence covering all slots:
$$
\text{indices} = {h_1, h_1 + h_2, h_1 + 2h_2, \ldots} \bmod m
$$
Thus an empty slot is always reachable, and searches find all candidates.

Expected probe count ≈ $\frac{1}{1 - \alpha}$, same as other open addressing, but with lower clustering.

#### Try It Yourself

1. Experiment with different h₂ functions (e.g. `7 - key % 7`).
2. Compare probe lengths with linear and quadratic probing.
3. Visualize probe paths for small table sizes.
4. Test with composite vs prime capacities.

#### Test Cases

| Operation | Key | h₁   | h₂    | Probe Sequence | Final Slot |
| --------- | --- | ---- | ----- | -------------- | ---------- |
| insert    | 10  | 3    | 4     | 3              | 3          |
| insert    | 24  | 3    | 4     | 3, 0           | 0          |
| insert    | 31  | 3    | 4     | 3, 0, 4        | 4          |
| insert    | 17  | 3    | 3     | 3, 6           | 6          |
| search    | 24  | 3, 0 | found |                |            |

Edge Cases

- h₂(key) must be nonzero
- m should be prime for full coverage
- Poor hash choice → incomplete coverage

#### Complexity

- Time: average O(1), worst O(n)
- Space: O(n)

Double hashing turns collisions into a graceful dance, two hash functions weaving paths that rarely cross.

### 215 Cuckoo Hashing

Cuckoo hashing takes inspiration from nature: like the cuckoo bird laying eggs in multiple nests, each key has more than one possible home. If a spot is taken, it *kicks out* the current occupant, which then moves to its alternate home, ensuring fast and predictable lookups.

#### What Problem Are We Solving?

Traditional open addressing methods (linear, quadratic, double hashing) may degrade under high load factors, causing long probe sequences. Cuckoo hashing guarantees constant-time lookups by giving each key multiple possible positions.

Goal:
Use two hash functions and relocate keys upon collision, maintaining O(1) search and insert time.

#### How Does It Work (Plain Language)?

Each key has two candidate slots, determined by two hash functions:
$$
h_1(k), \quad h_2(k)
$$

When inserting a key:

1. Try `h1(k)` → if empty, place it.
2. If occupied → *kick out* the existing key.
3. Reinsert the displaced key into its alternate slot.
4. Repeat until all keys placed or cycle detected (then rehash).

Example Steps (Capacity = 7)

| Step | Key | h₁(key) | h₂(key) | Action                    |
| ---- | --- | ------- | ------- | ------------------------- |
| 1    | 10  | 3       | 5       | Slot 3 empty → place 10   |
| 2    | 24  | 3       | 4       | Slot 3 occupied → move 10 |
| 3    | 10  | 5       | 3       | Slot 5 empty → place 10   |
| 4    | 31  | 3       | 6       | Slot 3 empty → place 31   |

Final Table

| Index | Value |
| ----- | ----- |
| 0     | -     |
| 1     | -     |
| 2     | -     |
| 3     | 31    |
| 4     | 24    |
| 5     | 10    |
| 6     | -     |

Every key is accessible in O(1) by checking two positions only.

#### Tiny Code (Easy Versions)

C (Two-Table Cuckoo Hashing)

```c
#include <stdio.h>
#include <stdlib.h>

#define CAPACITY 7
#define EMPTY -1
#define MAX_RELOCATIONS 10

typedef struct {
    int table1[CAPACITY];
    int table2[CAPACITY];
} CuckooHash;

int h1(int key) { return key % CAPACITY; }
int h2(int key) { return (key / CAPACITY) % CAPACITY; }

void init(CuckooHash *ht) {
    for (int i = 0; i < CAPACITY; i++) {
        ht->table1[i] = EMPTY;
        ht->table2[i] = EMPTY;
    }
}

int insert(CuckooHash *ht, int key) {
    int pos, tmp, loop_guard = 0;
    for (int i = 0; i < MAX_RELOCATIONS; i++) {
        pos = h1(key);
        if (ht->table1[pos] == EMPTY) {
            ht->table1[pos] = key;
            return 1;
        }
        // kick out
        tmp = ht->table1[pos];
        ht->table1[pos] = key;
        key = tmp;
        pos = h2(key);
        if (ht->table2[pos] == EMPTY) {
            ht->table2[pos] = key;
            return 1;
        }
        tmp = ht->table2[pos];
        ht->table2[pos] = key;
        key = tmp;
    }
    printf("Cycle detected, rehash needed\n");
    return 0;
}

int search(CuckooHash *ht, int key) {
    int pos1 = h1(key);
    int pos2 = h2(key);
    if (ht->table1[pos1] == key || ht->table2[pos2] == key) return 1;
    return 0;
}

int main(void) {
    CuckooHash ht;
    init(&ht);
    insert(&ht, 10);
    insert(&ht, 24);
    insert(&ht, 31);
    for (int i = 0; i < CAPACITY; i++)
        printf("[%d] T1=%d T2=%d\n", i, ht.table1[i], ht.table2[i]);
    return 0;
}
```

Python

```python
class CuckooHash:
    def __init__(self, size=7):
        self.size = size
        self.table1 = [None] * size
        self.table2 = [None] * size
        self.max_reloc = 10

    def _h1(self, key): return key % self.size
    def _h2(self, key): return (key // self.size) % self.size

    def insert(self, key):
        for _ in range(self.max_reloc):
            idx1 = self._h1(key)
            if self.table1[idx1] is None:
                self.table1[idx1] = key
                return
            key, self.table1[idx1] = self.table1[idx1], key  # swap

            idx2 = self._h2(key)
            if self.table2[idx2] is None:
                self.table2[idx2] = key
                return
            key, self.table2[idx2] = self.table2[idx2], key  # swap
        raise RuntimeError("Cycle detected, rehash needed")

    def search(self, key):
        return key in self.table1 or key in self.table2

# Example
ht = CuckooHash()
for k in [10, 24, 31]:
    ht.insert(k)
print("Table1:", ht.table1)
print("Table2:", ht.table2)
print("Search 24:", ht.search(24))
```

#### Why It Matters

- O(1) lookup, always two slots per key
- Avoids clustering entirely
- Excellent for high load factors (up to 0.5–0.9)
- Simple predictable probe path
- Great choice for hardware tables (e.g. network routing)

#### A Gentle Proof (Why It Works)

Each key has at most two possible homes.

- If both occupied, displacement ensures eventual convergence (or detects a cycle).
- Cycle length bounded → rehash needed rarely.

Expected insertion time = O(1) amortized; search always 2 checks only.

#### Try It Yourself

1. Implement rehash when cycle detected.
2. Add delete(key) and test reinsert.
3. Visualize displacement chain on insertions.
4. Compare performance with double hashing.

#### Test Cases

| Operation | Key | h₁   | h₂    | Action                  |
| --------- | --- | ---- | ----- | ----------------------- |
| insert    | 10  | 3    | 5     | place at 3              |
| insert    | 24  | 3    | 4     | displace 10 → move to 5 |
| insert    | 31  | 3    | 6     | place at 3              |
| search    | 10  | 3, 5 | found |                         |

Edge Cases

- Cycle detected → requires rehash
- Both tables full → resize
- Must limit relocation attempts

#### Complexity

- Time:

  * Lookup: O(1)
  * Insert: O(1) amortized
- Space: O(2n)

Cuckoo hashing keeps order in the nest, every key finds a home, or the table learns to rebuild its world.

### 216 Robin Hood Hashing

Robin Hood hashing is a clever twist on open addressing: when a new key collides, it compares its "distance from home" with the current occupant. If the new key has traveled farther, it *steals* the slot, redistributing probe distances more evenly and keeping variance low.

#### What Problem Are We Solving?

In linear probing, unlucky keys might travel long distances while others sit close to their home. This leads to probe sequence imbalance, long searches for some keys, short for others.
Robin Hood hashing "robs" near-home keys to help far-away ones, minimizing the maximum probe distance.

Goal:
Equalize probe distances by swapping keys so that no key is "too far" behind others.

#### How Does It Work (Plain Language)?

Each entry remembers its probe distance = number of steps from its original hash slot.
When inserting a new key:

1. Compute `index = hash(key) % capacity`.
2. If slot empty → insert.
3. If occupied → compare probe distances.

   * If newcomer's distance > occupant's distance → swap them.
   * Continue insertion for the displaced key.

Example Steps (Capacity = 7)

| Step | Key | Hash(key) | Probe Distance | Action                                                                   |
| ---- | --- | --------- | -------------- | ------------------------------------------------------------------------ |
| 1    | 10  | 3         | 0              | Place at 3                                                               |
| 2    | 24  | 3         | 0              | Collision → move to 4 (dist=1)                                           |
| 3    | 31  | 3         | 0              | Collision → dist=0 < 0? no → move → dist=1 < 1? no → dist=2 → place at 5 |
| 4    | 17  | 3         | 0              | Collision chain → compare and swap if farther                            |

This ensures all keys stay near their home index, fairer access for all.

Result Table:

| Index | Key | Dist |
| ----- | --- | ---- |
| 3     | 10  | 0    |
| 4     | 24  | 1    |
| 5     | 31  | 2    |

#### Tiny Code (Easy Versions)

C

```c
#include <stdio.h>
#include <stdlib.h>

#define CAPACITY 7
#define EMPTY -1

typedef struct {
    int key;
    int dist; // probe distance
} Slot;

typedef struct {
    Slot *table;
} HashTable;

int hash(int key) { return key % CAPACITY; }

HashTable* ht_create() {
    HashTable *ht = malloc(sizeof(HashTable));
    ht->table = malloc(sizeof(Slot) * CAPACITY);
    for (int i = 0; i < CAPACITY; i++) {
        ht->table[i].key = EMPTY;
        ht->table[i].dist = 0;
    }
    return ht;
}

void ht_insert(HashTable *ht, int key) {
    int idx = hash(key);
    int dist = 0;

    while (1) {
        if (ht->table[idx].key == EMPTY) {
            ht->table[idx].key = key;
            ht->table[idx].dist = dist;
            return;
        }

        if (dist > ht->table[idx].dist) {
            // swap keys
            int tmp_key = ht->table[idx].key;
            int tmp_dist = ht->table[idx].dist;
            ht->table[idx].key = key;
            ht->table[idx].dist = dist;
            key = tmp_key;
            dist = tmp_dist;
        }

        idx = (idx + 1) % CAPACITY;
        dist++;
        if (dist >= CAPACITY) {
            printf("Table full\n");
            return;
        }
    }
}

int ht_search(HashTable *ht, int key) {
    int idx = hash(key);
    int dist = 0;
    while (ht->table[idx].key != EMPTY && dist <= ht->table[idx].dist) {
        if (ht->table[idx].key == key) return idx;
        idx = (idx + 1) % CAPACITY;
        dist++;
    }
    return -1;
}

int main(void) {
    HashTable *ht = ht_create();
    ht_insert(ht, 10);
    ht_insert(ht, 24);
    ht_insert(ht, 31);
    for (int i = 0; i < CAPACITY; i++) {
        if (ht->table[i].key != EMPTY)
            printf("[%d] key=%d dist=%d\n", i, ht->table[i].key, ht->table[i].dist);
    }
    return 0;
}
```

Python

```python
class RobinHoodHash:
    def __init__(self, size=7):
        self.size = size
        self.table = [None] * size
        self.dist = [0] * size

    def _hash(self, key):
        return key % self.size

    def insert(self, key):
        idx = self._hash(key)
        d = 0
        while True:
            if self.table[idx] is None:
                self.table[idx] = key
                self.dist[idx] = d
                return
            # Robin Hood swap if newcomer is farther
            if d > self.dist[idx]:
                key, self.table[idx] = self.table[idx], key
                d, self.dist[idx] = self.dist[idx], d
            idx = (idx + 1) % self.size
            d += 1
            if d >= self.size:
                raise OverflowError("Table full")

    def search(self, key):
        idx = self._hash(key)
        d = 0
        while self.table[idx] is not None and d <= self.dist[idx]:
            if self.table[idx] == key:
                return idx
            idx = (idx + 1) % self.size
            d += 1
        return None

# Example
ht = RobinHoodHash()
for k in [10, 24, 31]:
    ht.insert(k)
print(list(zip(range(ht.size), ht.table, ht.dist)))
print("Search 24:", ht.search(24))
```

#### Why It Matters

- Balances access time across all keys
- Minimizes variance of probe lengths
- Outperforms linear probing under high load
- Elegant fairness principle, long-traveling keys get priority

#### A Gentle Proof (Why It Works)

By ensuring all probe distances are roughly equal, worst-case search cost ≈ average search cost.
Keys never get "stuck" behind long clusters, and searches terminate early when probe distance exceeds that of existing slot.

Average search cost ≈ O(1 + α), but with *smaller variance* than standard linear probing.

#### Try It Yourself

1. Insert keys in different orders and compare probe distances.
2. Implement deletion (mark deleted and shift neighbors).
3. Track average probe distance as load grows.
4. Compare fairness with standard linear probing.

#### Test Cases

| Operation | Key | Home | Final Slot | Dist | Notes             |
| --------- | --- | ---- | ---------- | ---- | ----------------- |
| insert    | 10  | 3    | 3          | 0    | first insert      |
| insert    | 24  | 3    | 4          | 1    | collision         |
| insert    | 31  | 3    | 5          | 2    | further collision |
| search    | 24  | 3→4  | found      |      |                   |

Edge Cases

- Table full → stop insertion
- Must cap distance to prevent infinite loop
- Deletion requires rebalancing neighbors

#### Complexity

- Time: average O(1), worst O(n)
- Space: O(n)

Robin Hood hashing brings justice to collisions, no key left wandering too far from home.

### 217 Chained Hash Table

A chained hash table is the classic solution for handling collisions, instead of squeezing every key into the array, each bucket holds a linked list (or chain) of entries that share the same hash index.

#### What Problem Are We Solving?

With open addressing, collisions force you to probe for new slots inside the array.
Chaining solves collisions externally, every index points to a small dynamic list, so multiple keys can share the same slot without crowding.

Goal:
Use linked lists (chains) to store colliding keys at the same hash index, keeping insert, search, and delete simple and efficient on average.

#### How Does It Work (Plain Language)?

Each array index stores a pointer to a linked list of key-value pairs.
When inserting:

1. Compute index = `hash(key) % capacity`.
2. Traverse chain to check if key exists.
3. If not, append new node to the front (or back).

Searching and deleting follow the same index and chain.

Example (Capacity = 5)

| Step | Key   | Hash(key) | Index | Action             |
| ---- | ----- | --------- | ----- | ------------------ |
| 1    | "cat" | 2         | 2     | Place in chain[2]  |
| 2    | "dog" | 4         | 4     | Place in chain[4]  |
| 3    | "bat" | 2         | 2     | Append to chain[2] |
| 4    | "ant" | 2         | 2     | Append to chain[2] |

Table Structure

| Index | Chain                 |
| ----- | --------------------- |
| 0     | -                     |
| 1     | -                     |
| 2     | "cat" → "bat" → "ant" |
| 3     | -                     |
| 4     | "dog"                 |

#### Tiny Code (Easy Versions)

C

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CAPACITY 5

typedef struct Node {
    char *key;
    int value;
    struct Node *next;
} Node;

typedef struct {
    Node *buckets[CAPACITY];
} HashTable;

unsigned int hash(const char *key) {
    unsigned int h = 0;
    while (*key) h = h * 31 + *key++;
    return h % CAPACITY;
}

HashTable* ht_create() {
    HashTable *ht = malloc(sizeof(HashTable));
    for (int i = 0; i < CAPACITY; i++) ht->buckets[i] = NULL;
    return ht;
}

void ht_insert(HashTable *ht, const char *key, int value) {
    unsigned int idx = hash(key);
    Node *node = ht->buckets[idx];
    while (node) {
        if (strcmp(node->key, key) == 0) { node->value = value; return; }
        node = node->next;
    }
    Node *new_node = malloc(sizeof(Node));
    new_node->key = strdup(key);
    new_node->value = value;
    new_node->next = ht->buckets[idx];
    ht->buckets[idx] = new_node;
}

int ht_search(HashTable *ht, const char *key, int *out) {
    unsigned int idx = hash(key);
    Node *node = ht->buckets[idx];
    while (node) {
        if (strcmp(node->key, key) == 0) { *out = node->value; return 1; }
        node = node->next;
    }
    return 0;
}

void ht_delete(HashTable *ht, const char *key) {
    unsigned int idx = hash(key);
    Node curr = &ht->buckets[idx];
    while (*curr) {
        if (strcmp((*curr)->key, key) == 0) {
            Node *tmp = *curr;
            *curr = (*curr)->next;
            free(tmp->key);
            free(tmp);
            return;
        }
        curr = &(*curr)->next;
    }
}

int main(void) {
    HashTable *ht = ht_create();
    ht_insert(ht, "cat", 1);
    ht_insert(ht, "bat", 2);
    ht_insert(ht, "ant", 3);
    int val;
    if (ht_search(ht, "bat", &val))
        printf("bat: %d\n", val);
    ht_delete(ht, "bat");
    if (!ht_search(ht, "bat", &val))
        printf("bat deleted\n");
    return 0;
}
```

Python

```python
class ChainedHash:
    def __init__(self, size=5):
        self.size = size
        self.table = [[] for _ in range(size)]

    def _hash(self, key):
        return hash(key) % self.size

    def insert(self, key, value):
        idx = self._hash(key)
        for pair in self.table[idx]:
            if pair[0] == key:
                pair[1] = value
                return
        self.table[idx].append([key, value])

    def search(self, key):
        idx = self._hash(key)
        for k, v in self.table[idx]:
            if k == key:
                return v
        return None

    def delete(self, key):
        idx = self._hash(key)
        self.table[idx] = [p for p in self.table[idx] if p[0] != key]

# Example
ht = ChainedHash()
ht.insert("cat", 1)
ht.insert("bat", 2)
ht.insert("ant", 3)
print(ht.table)
print("Search bat:", ht.search("bat"))
ht.delete("bat")
print(ht.table)
```

#### Why It Matters

- Simple and reliable collision handling
- Load factor can exceed 1 (chains absorb overflow)
- Deletion is straightforward (just remove node)
- Performance stable even at high load (if hash spread is uniform)

#### A Gentle Proof (Why It Works)

Expected chain length = load factor $\alpha = \frac{n}{m}$.
Each operation traverses a single chain, so average cost = O(1 + α).
Uniform hash distribution ensures α remains small → operations ≈ O(1).

#### Try It Yourself

1. Implement dynamic resizing when average chain length grows.
2. Compare prepend vs append strategies.
3. Measure average search steps as table fills.
4. Replace linked list with balanced tree (for high α).

#### Test Cases

| Operation | Key   | Index | Chain Result          | Notes       |
| --------- | ----- | ----- | --------------------- | ----------- |
| insert    | "cat" | 2     | ["cat"]               |             |
| insert    | "bat" | 2     | ["bat", "cat"]        | collision   |
| insert    | "ant" | 2     | ["ant", "bat", "cat"] | chain grows |
| search    | "bat" | 2     | found                 |             |
| delete    | "bat" | 2     | ["ant", "cat"]        | removed     |

Edge Cases

- Many keys same index → long chains
- Poor hash function → uneven distribution
- Needs memory for pointers/nodes

#### Complexity

- Time: average O(1), worst O(n) (all in one chain)
- Space: O(n + m)

Chained hashing turns collisions into conversation, if one bucket's full, it just lines them up neatly in a list.

### 218 Perfect Hashing

Perfect hashing is the dream scenario for hash tables, no collisions at all. Every key maps to a unique slot, so lookups, inserts, and deletes all take O(1) time *worst case*, not just on average.

#### What Problem Are We Solving?

Most hashing strategies (linear probing, chaining, cuckoo) deal with collisions after they happen.
Perfect hashing eliminates them entirely by designing a collision-free hash function for a fixed key set.

Goal:
Construct a hash function $h(k)$ such that all keys map to distinct indices.

#### How Does It Work (Plain Language)?

If the set of keys is known in advance (static set), we can carefully choose or build a hash function that gives each key a unique slot.

Two main types:

1. Perfect Hashing, no collisions.
2. Minimal Perfect Hashing, no collisions and table size = number of keys.

Simple Example (Keys = {10, 24, 31, 17}, Capacity = 7)

Let's find a function:
$$
h(k) = (a \cdot k + b) \bmod 7
$$
We can search for coefficients a and b that produce unique indices:

| Key | h(k) = (2k + 1) mod 7 | Index       |
| --- | --------------------- | ----------- |
| 10  | (21) % 7 = 0          | 0           |
| 24  | (49) % 7 = 0          | ❌ collision |
| 31  | (63) % 7 = 0          | ❌ collision |

So we try another pair (a=3, b=2):

| Key | (3k + 2) mod 7 | Index       |
| --- | -------------- | ----------- |
| 10  | 5              | 5           |
| 24  | 4              | 4           |
| 31  | 4              | ❌ collision |
| 17  | 6              | 6           |

Eventually, we find a mapping with no repeats by adjusting parameters or using two-level construction.

#### Two-Level Perfect Hashing

Practical perfect hashing often uses a two-level scheme:

1. Top level: Hash keys into buckets.
2. Second level: Each bucket gets its own small hash table with its own perfect hash function.

This ensures zero collisions overall, with total space ≈ O(n).

Process:

- Each bucket of size *b* gets a secondary table of size *b²*.
- Use a second hash $h_i$ for that bucket to place all keys uniquely.

#### Tiny Code (Easy Version)

Python (Two-Level Static Perfect Hashing)

```python
import random

class PerfectHash:
    def __init__(self, keys):
        self.n = len(keys)
        self.size = self.n
        self.buckets = [[] for _ in range(self.size)]
        self.secondary = [None] * self.size

        # First-level hashing
        a, b = 3, 5  # fixed small hash parameters
        def h1(k): return (a * k + b) % self.size

        # Distribute keys into buckets
        for k in keys:
            self.buckets[h1(k)].append(k)

        # Build second-level tables
        for i, bucket in enumerate(self.buckets):
            if not bucket:
                continue
            m = len(bucket)  2
            table = [None] * m
            found = False
            while not found:
                found = True
                a2, b2 = random.randint(1, m - 1), random.randint(0, m - 1)
                def h2(k): return (a2 * k + b2) % m
                table = [None] * m
                for k in bucket:
                    pos = h2(k)
                    if table[pos] is not None:
                        found = False
                        break
                    table[pos] = k
            self.secondary[i] = (table, a2, b2)

        self.h1 = h1

    def search(self, key):
        i = self.h1(key)
        table, a2, b2 = self.secondary[i]
        m = len(table)
        pos = (a2 * key + b2) % m
        return table[pos] == key

# Example
keys = [10, 24, 31, 17]
ph = PerfectHash(keys)
print([len(b) for b in ph.buckets])
print("Search 24:", ph.search(24))
print("Search 11:", ph.search(11))
```

#### Why It Matters

- Guaranteed O(1) worst-case lookup
- No clustering, no collisions, no chains
- Ideal for static key sets (e.g. reserved keywords in a compiler, routing tables)
- Memory predictable, access blazing fast

#### A Gentle Proof (Why It Works)

Let $n=|S|$ be the number of keys.  
With a truly random hash family into $m$ buckets, the collision probability for two distinct keys is $1/m$.

Choose $m=n^2$. Then the expected number of collisions is
$$
\mathbb{E}[C]=\binom{n}{2}\cdot \frac{1}{m}
=\frac{n(n-1)}{2n^2}<\tfrac12.
$$
By Markov, $\Pr[C\ge1]\le \mathbb{E}[C]<\tfrac12$, so $\Pr[C=0]>\tfrac12$.  
Therefore a collision-free hash exists. In practice, try random seeds until one has $C=0$, or use a deterministic construction for perfect hashing.


#### Try It Yourself

1. Generate perfect hash for small static set {"if", "else", "for", "while"}.
2. Build minimal perfect hash (table size = n).
3. Compare lookup times with standard dict.
4. Visualize second-level hash table sizes.

#### Test Cases

| Operation  | Keys          | Result  | Notes            |
| ---------- | ------------- | ------- | ---------------- |
| build      | [10,24,31,17] | success | each slot unique |
| search     | 24            | True    | found            |
| search     | 11            | False   | not in table     |
| collisions | none          |         | perfect mapping  |

Edge Cases

- Works only for static sets (no dynamic inserts)
- Building may require rehash trials
- Memory ↑ with quadratic secondary tables

#### Complexity

- Build: O(n²) (search for collision-free mapping)
- Lookup: O(1)
- Space: O(n) to O(n²) depending on method

Perfect hashing is like finding the perfect key for every lock, built once, opens instantly, never collides.

### 219 Consistent Hashing

Consistent hashing is a collision-handling strategy designed for *distributed systems* rather than single in-memory tables. It ensures that when nodes (servers, caches, or shards) join or leave, only a small fraction of keys need to be remapped, making it the backbone of scalable, fault-tolerant architectures.

#### What Problem Are We Solving?

In traditional hashing (e.g. `hash(key) % n`), when the number of servers `n` changes, almost every key's location changes. That's disastrous for caching, databases, or load balancing.

Consistent hashing fixes this by mapping both keys and servers into the same hash space, and placing keys near their nearest server clockwise, minimizing reassignments when the system changes.

Goal:
Achieve stable key distribution under dynamic server counts with minimal movement and good balance.

#### How Does It Work (Plain Language)?

1. Imagine a hash ring, numbers 0 through (2^{m}-1) arranged in a circle.
2. Each node (server) and key is hashed to a position on the ring.
3. A key is assigned to the next server clockwise from its hash position.
4. When a node joins/leaves, only keys in its immediate region move.

Example (Capacity = 2¹⁶)

| Item   | Hash | Placed On Ring         | Owner |
| ------ | ---- | ---------------------- | ----- |
| Node A | 1000 | •                      |       |
| Node B | 4000 | •                      |       |
| Node C | 8000 | •                      |       |
| Key K₁ | 1200 | → Node B               |       |
| Key K₂ | 8500 | → Node A (wrap-around) |       |

If Node B leaves, only keys in its segment (1000–4000) move, everything else stays put.

#### Improving Load Balance

To prevent uneven distribution, each node is represented by multiple virtual nodes (vnodes), each vnode gets its own hash.
This smooths the key spread across all nodes.

Example:

- Node A → hashes to 1000, 6000
- Node B → hashes to 3000, 9000

Keys are assigned to closest vnode clockwise.

#### Tiny Code (Easy Versions)

Python (Simple Consistent Hashing with Virtual Nodes)

```python
import bisect
import hashlib

def hash_fn(key):
    return int(hashlib.md5(str(key).encode()).hexdigest(), 16)

class ConsistentHash:
    def __init__(self, nodes=None, vnodes=3):
        self.ring = []
        self.map = {}
        self.vnodes = vnodes
        if nodes:
            for n in nodes:
                self.add_node(n)

    def add_node(self, node):
        for i in range(self.vnodes):
            h = hash_fn(f"{node}-{i}")
            self.map[h] = node
            bisect.insort(self.ring, h)

    def remove_node(self, node):
        for i in range(self.vnodes):
            h = hash_fn(f"{node}-{i}")
            self.ring.remove(h)
            del self.map[h]

    def get_node(self, key):
        if not self.ring:
            return None
        h = hash_fn(key)
        idx = bisect.bisect(self.ring, h) % len(self.ring)
        return self.map[self.ring[idx]]

# Example
ch = ConsistentHash(["A", "B", "C"], vnodes=2)
print("Key 42 ->", ch.get_node(42))
ch.remove_node("B")
print("After removing B, Key 42 ->", ch.get_node(42))
```

#### Why It Matters

- Minimizes key remapping when nodes change (≈ 1/n of keys move)
- Enables elastic scaling for caches, DB shards, and distributed stores
- Used in systems like Amazon Dynamo, Cassandra, Riak, and memcached clients
- Balances load using virtual nodes
- Decouples hash function from node count

#### A Gentle Proof (Why It Works)

Let $N$ be the number of nodes and $K$ the number of keys.  
Each node is responsible for a fraction $\frac{1}{N}$ of the ring.

When one node leaves, only its segment’s keys move, about $K/N$.  
So the expected remapping fraction is $\frac{1}{N}$.

Adding virtual nodes (vnodes) increases uniformity.  
With $V$ vnodes per physical node, the variance of the per-node load fraction scales as $\approx \frac{1}{V}$.


#### Try It Yourself

1. Add and remove nodes, track how many keys move.
2. Experiment with different vnode counts (1, 10, 100).
3. Visualize hash ring, mark nodes and key positions.
4. Simulate caching: assign 1000 keys, remove one node, count moves.

#### Test Cases

| Operation   | Node(s) | Keys      | Result             | Notes           |
| ----------- | ------- | --------- | ------------------ | --------------- |
| add_nodes   | A, B, C | [1..1000] | distributed evenly |                 |
| remove_node | B       | [1..1000] | ~1/3 keys moved    | stability check |
| add_node    | D       | [1..1000] | ~1/4 keys remapped |                 |
| lookup      | 42      | -> Node C | consistent mapping |                 |

Edge Cases

- Empty ring → return None
- Duplicate nodes → handle via unique vnode IDs
- Without vnodes → uneven load

#### Complexity

- Lookup: $O(\log n)$ (binary search in the ring)  
- Insert/Delete node: $O(v \log n)$  
- Space: $O(n \times v)$  

Consistent hashing keeps order in the storm — servers may come and go, but most keys stay right where they belong.


### 220 Dynamic Rehashing

Dynamic rehashing is how a hash table gracefully adapts as data grows or shrinks. Instead of being trapped in a fixed-size array, the table *resizes itself*, rebuilding its layout so that load stays balanced and lookups remain fast.

#### What Problem Are We Solving?

When a hash table fills up, collisions become frequent, degrading performance to O(n).
We need a mechanism to maintain a low load factor (ratio of elements to capacity) by resizing and rehashing automatically.

Goal:
Detect when load factor crosses a threshold, allocate a larger array, and rehash all keys into their new positions efficiently.

#### How Does It Work (Plain Language)?

1. Monitor load factor

   $$
   \alpha = \frac{n}{m}
   $$
   where $n$ is the number of elements and $m$ is the table size.

2. Trigger rehash

   - If $\alpha > 0.75$, expand the table (for example, double the capacity).  
   - If $\alpha < 0.25$, shrink it (optional).

3. Rebuild

   - Create a new table with the updated capacity.  
   - Reinsert each key using the new hash function modulo the new capacity.


Example Steps

| Step | Capacity | Items | Load Factor | Action      |
| ---- | -------- | ----- | ----------- | ----------- |
| 1    | 4        | 2     | 0.5         | ok          |
| 2    | 4        | 3     | 0.75        | ok          |
| 3    | 4        | 4     | 1.0         | resize to 8 |
| 4    | 8        | 4     | 0.5         | rehashed    |

Every key gets new position because `hash(key) % new_capacity` changes.

#### Incremental Rehashing

Instead of rehashing all keys at once (costly spike), incremental rehashing spreads work across operations:

- Maintain both old and new tables.
- Rehash a few entries per insert/search until old table empty.

This keeps amortized O(1) performance, even during resize.

#### Tiny Code (Easy Versions)

C (Simple Doubling Rehash)

```c
#include <stdio.h>
#include <stdlib.h>

#define INIT_CAP 4

typedef struct {
    int *keys;
    int size;
    int count;
} HashTable;

int hash(int key, int size) { return key % size; }

HashTable* ht_create(int size) {
    HashTable *ht = malloc(sizeof(HashTable));
    ht->keys = malloc(sizeof(int) * size);
    for (int i = 0; i < size; i++) ht->keys[i] = -1;
    ht->size = size;
    ht->count = 0;
    return ht;
}

void ht_resize(HashTable *ht, int new_size) {
    printf("Resizing from %d to %d\n", ht->size, new_size);
    int *old_keys = ht->keys;
    int old_size = ht->size;

    ht->keys = malloc(sizeof(int) * new_size);
    for (int i = 0; i < new_size; i++) ht->keys[i] = -1;

    ht->size = new_size;
    ht->count = 0;
    for (int i = 0; i < old_size; i++) {
        if (old_keys[i] != -1) {
            int key = old_keys[i];
            int idx = hash(key, new_size);
            while (ht->keys[idx] != -1) idx = (idx + 1) % new_size;
            ht->keys[idx] = key;
            ht->count++;
        }
    }
    free(old_keys);
}

void ht_insert(HashTable *ht, int key) {
    float load = (float)ht->count / ht->size;
    if (load > 0.75) ht_resize(ht, ht->size * 2);

    int idx = hash(key, ht->size);
    while (ht->keys[idx] != -1) idx = (idx + 1) % ht->size;
    ht->keys[idx] = key;
    ht->count++;
}

int main(void) {
    HashTable *ht = ht_create(INIT_CAP);
    for (int i = 0; i < 10; i++) ht_insert(ht, i * 3);
    for (int i = 0; i < ht->size; i++)
        printf("[%d] = %d\n", i, ht->keys[i]);
    return 0;
}
```

Python

```python
class DynamicHash:
    def __init__(self, cap=4):
        self.cap = cap
        self.size = 0
        self.table = [None] * cap

    def _hash(self, key):
        return hash(key) % self.cap

    def _rehash(self, new_cap):
        old_table = self.table
        self.table = [None] * new_cap
        self.cap = new_cap
        self.size = 0
        for key in old_table:
            if key is not None:
                self.insert(key)

    def insert(self, key):
        if self.size / self.cap > 0.75:
            self._rehash(self.cap * 2)
        idx = self._hash(key)
        while self.table[idx] is not None:
            idx = (idx + 1) % self.cap
        self.table[idx] = key
        self.size += 1

# Example
ht = DynamicHash()
for k in [10, 24, 31, 17, 19, 42, 56, 77]:
    ht.insert(k)
print(ht.table)
```

#### Why It Matters

- Keeps load factor stable for O(1) operations
- Prevents clustering in open addressing
- Supports unbounded growth
- Foundation for dynamic dictionaries, maps, caches

#### A Gentle Proof (Why It Works)

If each rehash doubles capacity, total cost of N inserts = O(N).
Each element is moved O(1) times (once per doubling), so amortized cost per insert is O(1).

Incremental rehashing further ensures no single operation is expensive, work spread evenly.

#### Try It Yourself

1. Add printouts to observe load factor at each insert.
2. Implement shrink when load < 0.25.
3. Implement incremental rehash with two tables.
4. Compare doubling vs prime-capacity growth.

#### Test Cases

| Step | Capacity | Items | Load Factor | Action   |
| ---- | -------- | ----- | ----------- | -------- |
| 1    | 4        | 2     | 0.5         | none     |
| 2    | 4        | 3     | 0.75        | ok       |
| 3    | 4        | 4     | 1.0         | resize   |
| 4    | 8        | 4     | 0.5         | rehashed |

Edge Cases

- Rehash must handle deleted slots correctly
- Avoid resizing too frequently (hysteresis)
- Keep hash function consistent across resizes

#### Complexity

- Average Insert/Search/Delete: O(1)
- Amortized Insert: O(1)
- Worst-case Resize: O(n)

Dynamic rehashing is the table's heartbeat, it expands when full, contracts when idle, always keeping operations smooth and steady.

## Section 23. Heaps 

### 221 Binary Heap Insert

A binary heap is a complete binary tree stored in an array that maintains the heap property, each parent is smaller (min-heap) or larger (max-heap) than its children.
The insert operation keeps the heap ordered by *bubbling up* the new element until it finds the right place.

#### What Problem Are We Solving?

We need a data structure that efficiently gives access to the minimum (or maximum) element, while supporting fast insertions.

A binary heap offers:

- `O(1)` access to min/max
- `O(log n)` insertion and deletion
- `O(n)` build time for initial heap

Goal:
Insert a new element while maintaining the heap property and completeness.

#### How Does It Work (Plain Language)?

A heap is stored as an array representing a complete binary tree.
Each node at index `i` has:

- Parent: `(i - 1) / 2`
- Left child: `2i + 1`
- Right child: `2i + 2`

Insertion Steps (Min-Heap)

1. Append the new element at the end (bottom level, rightmost).
2. Compare it with its parent.
3. If smaller (min-heap) or larger (max-heap), swap.
4. Repeat until the heap property is restored.

Example (Min-Heap)

Insert sequence: [10, 24, 5, 31]

| Step      | Array           | Action                      |
| --------- | --------------- | --------------------------- |
| Start     | [ ]             | empty                       |
| Insert 10 | [10]            | root only                   |
| Insert 24 | [10, 24]        | 24 > 10, no swap            |
| Insert 5  | [10, 24, 5]     | 5 < 10 → swap → [5, 24, 10] |
| Insert 31 | [5, 24, 10, 31] | 31 > 24, no swap            |

Final heap: [5, 24, 10, 31]

Tree view:

```
        5
      /   \
    24     10
   /
 31
```

#### Tiny Code (Easy Versions)

C

```c
#include <stdio.h>
#define MAX 100

typedef struct {
    int arr[MAX];
    int size;
} MinHeap;

void swap(int *a, int *b) {
    int tmp = *a; *a = *b; *b = tmp;
}

void heap_insert(MinHeap *h, int val) {
    int i = h->size++;
    h->arr[i] = val;

    // bubble up
    while (i > 0) {
        int parent = (i - 1) / 2;
        if (h->arr[i] >= h->arr[parent]) break;
        swap(&h->arr[i], &h->arr[parent]);
        i = parent;
    }
}

void heap_print(MinHeap *h) {
    for (int i = 0; i < h->size; i++) printf("%d ", h->arr[i]);
    printf("\n");
}

int main(void) {
    MinHeap h = {.size = 0};
    int vals[] = {10, 24, 5, 31};
    for (int i = 0; i < 4; i++) heap_insert(&h, vals[i]);
    heap_print(&h);
}
```

Python

```python
class MinHeap:
    def __init__(self):
        self.arr = []

    def _parent(self, i): return (i - 1) // 2

    def insert(self, val):
        self.arr.append(val)
        i = len(self.arr) - 1
        while i > 0:
            p = self._parent(i)
            if self.arr[i] >= self.arr[p]:
                break
            self.arr[i], self.arr[p] = self.arr[p], self.arr[i]
            i = p

    def __repr__(self):
        return str(self.arr)

# Example
h = MinHeap()
for x in [10, 24, 5, 31]:
    h.insert(x)
print(h)
```

#### Why It Matters

- Fundamental for priority queues
- Core of Dijkstra's shortest path, Prim's MST, and schedulers
- Supports efficient `insert`, `extract-min/max`, and `peek`
- Used in heapsort and event-driven simulations

#### A Gentle Proof (Why It Works)

Each insertion bubbles up at most height of heap = $\log_2 n$.
Since heap always remains complete, the structure is balanced, ensuring logarithmic operations.

Heap property is preserved because every swap ensures parent ≤ child (min-heap).

#### Try It Yourself

1. Insert elements in descending order → observe bubbling.
2. Switch comparisons for max-heap.
3. Print tree level by level after each insert.
4. Implement `extract_min()` to remove root and restore heap.

#### Test Cases

| Operation | Input           | Output          | Notes             |
| --------- | --------------- | --------------- | ----------------- |
| Insert    | [10, 24, 5, 31] | [5, 24, 10, 31] | min-heap property |
| Insert    | [3, 2, 1]       | [1, 3, 2]       | swap chain        |
| Insert    | [10]            | [10]            | single element    |
| Insert    | []              | [x]             | empty start       |

Edge Cases

- Full array (static heap) → resize needed
- Negative values → handled same
- Duplicate keys → order preserved

#### Complexity

- Insert: O(log n)
- Search: O(n) (unsorted beyond heap property)
- Space: O(n)

Binary heap insertion is the heartbeat of priority queues, each element climbs to its rightful place, one gentle swap at a time.

### 222 Binary Heap Delete

Deleting from a binary heap means removing the root element (the minimum in a min-heap or maximum in a max-heap) while keeping both the heap property and complete tree structure intact.
To do this, we swap the root with the last element, remove the last, and bubble down the new root until the heap is valid again.

#### What Problem Are We Solving?

We want to remove the highest-priority element quickly (min or max) from a heap, without breaking the structure.

In a min-heap, the smallest value always lives at index 0.
In a max-heap, the largest value lives at index 0.

Goal:
Efficiently remove the root and restore order in O(log n) time.

#### How Does It Work (Plain Language)?

1. Swap root (index 0) with last element.
2. Remove last element (now root value is gone).
3. Heapify down (bubble down) from the root:

   * Compare with children.
   * Swap with smaller (min-heap) or larger (max-heap) child.
   * Repeat until heap property restored.

Example (Min-Heap)

Start: `[5, 24, 10, 31]`
Remove min (5):

| Step | Action                                            | Array           |              |
| ---- | ------------------------------------------------- | --------------- | ------------ |
| 1    | Swap root (5) with last (31)                      | [31, 24, 10, 5] |              |
| 2    | Remove last                                       | [31, 24, 10]    |              |
| 3    | Compare 31 with children (24, 10) → smallest = 10 | swap            | [10, 24, 31] |
| 4    | Stop (31 > no child)                              | [10, 24, 31]    |              |

Result: `[10, 24, 31]`, still a valid min-heap.

#### Tiny Code (Easy Versions)

C

```c
#include <stdio.h>
#define MAX 100

typedef struct {
    int arr[MAX];
    int size;
} MinHeap;

void swap(int *a, int *b) {
    int tmp = *a; *a = *b; *b = tmp;
}

void heapify_down(MinHeap *h, int i) {
    int smallest = i;
    int left = 2*i + 1;
    int right = 2*i + 2;

    if (left < h->size && h->arr[left] < h->arr[smallest])
        smallest = left;
    if (right < h->size && h->arr[right] < h->arr[smallest])
        smallest = right;

    if (smallest != i) {
        swap(&h->arr[i], &h->arr[smallest]);
        heapify_down(h, smallest);
    }
}

int heap_delete_min(MinHeap *h) {
    if (h->size == 0) return -1;
    int root = h->arr[0];
    h->arr[0] = h->arr[h->size - 1];
    h->size--;
    heapify_down(h, 0);
    return root;
}

int main(void) {
    MinHeap h = {.arr = {5, 24, 10, 31}, .size = 4};
    int val = heap_delete_min(&h);
    printf("Deleted: %d\n", val);
    for (int i = 0; i < h.size; i++) printf("%d ", h.arr[i]);
    printf("\n");
}
```

Python

```python
class MinHeap:
    def __init__(self):
        self.arr = []

    def _parent(self, i): return (i - 1) // 2
    def _left(self, i): return 2 * i + 1
    def _right(self, i): return 2 * i + 2

    def insert(self, val):
        self.arr.append(val)
        i = len(self.arr) - 1
        while i > 0 and self.arr[i] < self.arr[self._parent(i)]:
            p = self._parent(i)
            self.arr[i], self.arr[p] = self.arr[p], self.arr[i]
            i = p

    def _heapify_down(self, i):
        smallest = i
        left, right = self._left(i), self._right(i)
        n = len(self.arr)
        if left < n and self.arr[left] < self.arr[smallest]:
            smallest = left
        if right < n and self.arr[right] < self.arr[smallest]:
            smallest = right
        if smallest != i:
            self.arr[i], self.arr[smallest] = self.arr[smallest], self.arr[i]
            self._heapify_down(smallest)

    def delete_min(self):
        if not self.arr:
            return None
        root = self.arr[0]
        last = self.arr.pop()
        if self.arr:
            self.arr[0] = last
            self._heapify_down(0)
        return root

# Example
h = MinHeap()
for x in [5, 24, 10, 31]:
    h.insert(x)
print("Before:", h.arr)
print("Deleted:", h.delete_min())
print("After:", h.arr)
```

#### Why It Matters

- Key operation in priority queues
- Core of Dijkstra's and Prim's algorithms
- Basis of heap sort (repeated delete-min)
- Ensures efficient extraction of extreme element

#### A Gentle Proof (Why It Works)

Each delete operation:

- Constant-time root removal
- Logarithmic heapify-down (height of tree = log n)
  → Total cost: O(log n)

The heap property holds because every swap moves a larger (min-heap) or smaller (max-heap) element down to children, ensuring local order at each step.

#### Try It Yourself

1. Delete repeatedly to sort the array (heap sort).
2. Try max-heap delete (reverse comparisons).
3. Visualize swaps after each deletion.
4. Test on ascending/descending input sequences.

#### Test Cases

| Input        | Operation | Output | Heap After | Notes      |
| ------------ | --------- | ------ | ---------- | ---------- |
| [5,24,10,31] | delete    | 5      | [10,24,31] | valid      |
| [1,3,2]      | delete    | 1      | [2,3]      | OK         |
| [10]         | delete    | 10     | []         | empty heap |
| []           | delete    | None   | []         | safe       |

Edge Cases

- Empty heap → return sentinel
- Single element → clears heap
- Duplicates handled naturally

#### Complexity

- Delete root: O(log n)
- Space: O(n)

Deleting from a heap is like removing the top card from a neat stack, replace it, sift it down, and balance restored.

### 223 Build Heap (Heapify)

Heapify (Build Heap) is the process of constructing a valid binary heap from an unsorted array in O(n) time. Instead of inserting elements one by one, we reorganize the array *in place* so every parent satisfies the heap property.

#### What Problem Are We Solving?

If we insert each element individually into an empty heap, total time is O(n log n).
But we can do better.
By heapifying from the bottom up, we can build the entire heap in O(n), crucial for heapsort and initializing priority queues efficiently.

Goal:
Turn any array into a valid min-heap or max-heap quickly.

#### How Does It Work (Plain Language)?

A heap stored in an array represents a complete binary tree:

- For node `i`, children are `2i + 1` and `2i + 2`

To build the heap:

1. Start from the last non-leaf node = `(n / 2) - 1`.
2. Apply heapify-down (sift down) to ensure the subtree rooted at `i` satisfies heap property.
3. Move upwards to the root, repeating the process.

Each subtree becomes a valid heap, and when done, the whole array is a heap.

Example (Min-Heap)

Start: `[31, 10, 24, 5, 12, 7]`

| Step  | i | Subtree      | Action         | Result                 |
| ----- | - | ------------ | -------------- | ---------------------- |
| Start | - | full array   | -              | [31, 10, 24, 5, 12, 7] |
| 1     | 2 | (24, 7)      | 24 > 7 → swap  | [31, 10, 7, 5, 12, 24] |
| 2     | 1 | (10, 5, 12)  | 10 > 5 → swap  | [31, 5, 7, 10, 12, 24] |
| 3     | 0 | (31, 5, 7)   | 31 > 5 → swap  | [5, 31, 7, 10, 12, 24] |
| 4     | 1 | (31, 10, 12) | 31 > 10 → swap | [5, 10, 7, 31, 12, 24] |

Final heap: [5, 10, 7, 31, 12, 24]

#### Tiny Code (Easy Versions)

C (Bottom-Up Build Heap)

```c
#include <stdio.h>

#define MAX 100

typedef struct {
    int arr[MAX];
    int size;
} MinHeap;

void swap(int *a, int *b) {
    int tmp = *a; *a = *b; *b = tmp;
}

void heapify_down(MinHeap *h, int i) {
    int smallest = i;
    int left = 2 * i + 1;
    int right = 2 * i + 2;

    if (left < h->size && h->arr[left] < h->arr[smallest])
        smallest = left;
    if (right < h->size && h->arr[right] < h->arr[smallest])
        smallest = right;
    if (smallest != i) {
        swap(&h->arr[i], &h->arr[smallest]);
        heapify_down(h, smallest);
    }
}

void build_heap(MinHeap *h) {
    for (int i = h->size / 2 - 1; i >= 0; i--)
        heapify_down(h, i);
}

int main(void) {
    MinHeap h = {.arr = {31, 10, 24, 5, 12, 7}, .size = 6};
    build_heap(&h);
    for (int i = 0; i < h.size; i++) printf("%d ", h.arr[i]);
    printf("\n");
}
```

Python

```python
class MinHeap:
    def __init__(self, arr):
        self.arr = arr
        self.size = len(arr)
        self.build_heap()

    def _left(self, i): return 2 * i + 1
    def _right(self, i): return 2 * i + 2

    def _heapify_down(self, i):
        smallest = i
        l, r = self._left(i), self._right(i)
        if l < self.size and self.arr[l] < self.arr[smallest]:
            smallest = l
        if r < self.size and self.arr[r] < self.arr[smallest]:
            smallest = r
        if smallest != i:
            self.arr[i], self.arr[smallest] = self.arr[smallest], self.arr[i]
            self._heapify_down(smallest)

    def build_heap(self):
        for i in range(self.size // 2 - 1, -1, -1):
            self._heapify_down(i)

# Example
arr = [31, 10, 24, 5, 12, 7]
h = MinHeap(arr)
print(h.arr)
```

#### Why It Matters

- Builds a valid heap in O(n) (not O(n log n))
- Used in heapsort initialization
- Efficient for constructing priority queues from bulk data
- Guarantees balanced tree structure automatically

#### A Gentle Proof (Why It Works)

Each node at depth `d` takes O(height) = O(log(n/d)) work.
There are more nodes at lower depths (less work each), fewer at top (more work each).
Sum across levels yields O(n) total time, not O(n log n).

Hence, bottom-up heapify is asymptotically optimal.

#### Try It Yourself

1. Run with different array sizes, random orders.
2. Compare time vs inserting one-by-one.
3. Flip comparisons for max-heap.
4. Visualize swaps as a tree.

#### Test Cases

| Input             | Output            | Notes          |
| ----------------- | ----------------- | -------------- |
| [31,10,24,5,12,7] | [5,10,7,31,12,24] | valid min-heap |
| [3,2,1]           | [1,2,3]           | heapified      |
| [10]              | [10]              | single         |
| []                | []                | empty safe     |

Edge Cases

- Already heap → no change
- Reverse-sorted input → many swaps
- Duplicates handled correctly

#### Complexity

- Time: O(n)
- Space: O(1) (in-place)

Heapify is the quiet craftsman, shaping chaos into order with gentle swaps from the ground up.

### 224 Heap Sort

Heap Sort is a classic comparison-based sorting algorithm that uses a binary heap to organize data and extract elements in sorted order. By building a max-heap (or min-heap) and repeatedly removing the root, we achieve a fully sorted array in O(n log n) time, with no extra space.

#### What Problem Are We Solving?

We want a fast, in-place sorting algorithm that:

- Doesn't require recursion (like mergesort)
- Has predictable O(n log n) behavior
- Avoids worst-case quadratic time (like quicksort)

Goal:
Use the heap's structure to repeatedly select the next largest (or smallest) element efficiently.

#### How Does It Work (Plain Language)?

1. Build a max-heap from the unsorted array.
2. Repeat until heap is empty:

   * Swap root (max element) with last element.
   * Reduce heap size by one.
   * Heapify-down the new root to restore heap property.

The array becomes sorted in ascending order (for max-heap).

Example (Ascending Sort)

Start: `[5, 31, 10, 24, 7]`

| Step | Action               | Array              |
| ---- | -------------------- | ------------------ |
| 1    | Build max-heap       | [31, 24, 10, 5, 7] |
| 2    | Swap 31 ↔ 7, heapify | [24, 7, 10, 5, 31] |
| 3    | Swap 24 ↔ 5, heapify | [10, 7, 5, 24, 31] |
| 4    | Swap 10 ↔ 5, heapify | [5, 7, 10, 24, 31] |
| 5    | Sorted               | [5, 7, 10, 24, 31] |

#### Tiny Code (Easy Versions)

C (In-place Heap Sort)

```c
#include <stdio.h>

void swap(int *a, int *b) {
    int tmp = *a; *a = *b; *b = tmp;
}

void heapify(int arr[], int n, int i) {
    int largest = i;
    int left = 2*i + 1;
    int right = 2*i + 2;

    if (left < n && arr[left] > arr[largest])
        largest = left;
    if (right < n && arr[right] > arr[largest])
        largest = right;

    if (largest != i) {
        swap(&arr[i], &arr[largest]);
        heapify(arr, n, largest);
    }
}

void heap_sort(int arr[], int n) {
    // build max-heap
    for (int i = n / 2 - 1; i >= 0; i--)
        heapify(arr, n, i);

    // extract elements
    for (int i = n - 1; i > 0; i--) {
        swap(&arr[0], &arr[i]);
        heapify(arr, i, 0);
    }
}

int main(void) {
    int arr[] = {5, 31, 10, 24, 7};
    int n = sizeof(arr)/sizeof(arr[0]);
    heap_sort(arr, n);
    for (int i = 0; i < n; i++) printf("%d ", arr[i]);
    printf("\n");
}
```

Python

```python
def heapify(arr, n, i):
    largest = i
    l, r = 2*i + 1, 2*i + 2

    if l < n and arr[l] > arr[largest]:
        largest = l
    if r < n and arr[r] > arr[largest]:
        largest = r

    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]
        heapify(arr, n, largest)

def heap_sort(arr):
    n = len(arr)
    # build max-heap
    for i in range(n//2 - 1, -1, -1):
        heapify(arr, n, i)
    # extract
    for i in range(n - 1, 0, -1):
        arr[0], arr[i] = arr[i], arr[0]
        heapify(arr, i, 0)

# Example
arr = [5, 31, 10, 24, 7]
heap_sort(arr)
print(arr)
```

#### Why It Matters

- Guaranteed O(n log n) runtime (worst-case safe)
- In-place, no extra arrays
- Great teaching example of heap structure in action
- Used in systems with strict space limits

#### A Gentle Proof (Why It Works)

1. Building the heap = O(n) (bottom-up heapify).
2. Each extraction = O(log n), repeated n times → O(n log n).
3. Heap property ensures root always holds largest element.

So final array is sorted after repeated root extractions.

#### Try It Yourself

1. Switch comparisons for min-heap sort (descending).
2. Print array after each swap to see process.
3. Compare with quicksort and mergesort.
4. Test large random arrays.

#### Test Cases

| Input          | Output         | Notes         |
| -------------- | -------------- | ------------- |
| [5,31,10,24,7] | [5,7,10,24,31] | ascending     |
| [10,9,8,7]     | [7,8,9,10]     | reverse input |
| [3]            | [3]            | single        |
| []             | []             | empty         |

Edge Cases

- Repeated values handled fine
- Already sorted input still O(n log n)
- Stable? ❌ (order not preserved)

#### Complexity

- Time: O(n log n)
- Space: O(1)
- Stable: No

Heap sort is the mountain climber's algorithm, pulling the biggest to the top, one step at a time, until order reigns from summit to base.

### 225 Min Heap Implementation

A min-heap is a binary tree where each parent is smaller than or equal to its children. Stored compactly in an array, it guarantees O(1) access to the smallest element and O(log n) insertion and deletion, perfect for priority queues and scheduling systems.

#### What Problem Are We Solving?

We need a data structure that:

- Quickly gives us the minimum element
- Supports efficient insertions and deletions
- Maintains order dynamically

A min-heap achieves this balance, always keeping the smallest element at the root while remaining compact and complete.

#### How Does It Work (Plain Language)?

A binary min-heap is stored as an array, where:

- `parent(i) = (i - 1) / 2`
- `left(i) = 2i + 1`
- `right(i) = 2i + 2`

Operations:

1. Insert(x):

   * Add `x` at the end.
   * "Bubble up" while `x < parent(x)`.

2. ExtractMin():

   * Remove root.
   * Move last element to root.
   * "Bubble down" while parent > smallest child.

3. Peek():

   * Return `arr[0]`.

Example

Start: `[ ]`
Insert sequence: 10, 24, 5, 31, 7

| Step | Action    | Array                    |             |
| ---- | --------- | ------------------------ | ----------- |
| 1    | insert 10 | [10]                     |             |
| 2    | insert 24 | [10, 24]                 |             |
| 3    | insert 5  | bubble up → swap with 10 | [5, 24, 10] |
| 4    | insert 31 | [5, 24, 10, 31]          |             |
| 5    | insert 7  | [5, 7, 10, 31, 24]       |             |

ExtractMin:

- Swap root with last (5 ↔ 24) → `[24, 7, 10, 31, 5]`
- Remove last → `[24, 7, 10, 31]`
- Bubble down → `[7, 24, 10, 31]`

#### Tiny Code (Easy Versions)

C

```c
#include <stdio.h>
#define MAX 100

typedef struct {
    int arr[MAX];
    int size;
} MinHeap;

void swap(int *a, int *b) {
    int tmp = *a; *a = *b; *b = tmp;
}

void heapify_up(MinHeap *h, int i) {
    while (i > 0) {
        int parent = (i - 1) / 2;
        if (h->arr[i] >= h->arr[parent]) break;
        swap(&h->arr[i], &h->arr[parent]);
        i = parent;
    }
}

void heapify_down(MinHeap *h, int i) {
    int smallest = i;
    int left = 2*i + 1;
    int right = 2*i + 2;

    if (left < h->size && h->arr[left] < h->arr[smallest]) smallest = left;
    if (right < h->size && h->arr[right] < h->arr[smallest]) smallest = right;

    if (smallest != i) {
        swap(&h->arr[i], &h->arr[smallest]);
        heapify_down(h, smallest);
    }
}

void insert(MinHeap *h, int val) {
    h->arr[h->size] = val;
    heapify_up(h, h->size);
    h->size++;
}

int extract_min(MinHeap *h) {
    if (h->size == 0) return -1;
    int root = h->arr[0];
    h->arr[0] = h->arr[--h->size];
    heapify_down(h, 0);
    return root;
}

int peek(MinHeap *h) {
    return h->size > 0 ? h->arr[0] : -1;
}

int main(void) {
    MinHeap h = {.size = 0};
    int vals[] = {10, 24, 5, 31, 7};
    for (int i = 0; i < 5; i++) insert(&h, vals[i]);
    printf("Min: %d\n", peek(&h));
    printf("Extracted: %d\n", extract_min(&h));
    for (int i = 0; i < h.size; i++) printf("%d ", h.arr[i]);
    printf("\n");
}
```

Python

```python
class MinHeap:
    def __init__(self):
        self.arr = []

    def _parent(self, i): return (i - 1) // 2
    def _left(self, i): return 2 * i + 1
    def _right(self, i): return 2 * i + 2

    def insert(self, val):
        self.arr.append(val)
        i = len(self.arr) - 1
        while i > 0 and self.arr[i] < self.arr[self._parent(i)]:
            p = self._parent(i)
            self.arr[i], self.arr[p] = self.arr[p], self.arr[i]
            i = p

    def extract_min(self):
        if not self.arr:
            return None
        root = self.arr[0]
        last = self.arr.pop()
        if self.arr:
            self.arr[0] = last
            self._heapify_down(0)
        return root

    def _heapify_down(self, i):
        smallest = i
        l, r = self._left(i), self._right(i)
        if l < len(self.arr) and self.arr[l] < self.arr[smallest]:
            smallest = l
        if r < len(self.arr) and self.arr[r] < self.arr[smallest]:
            smallest = r
        if smallest != i:
            self.arr[i], self.arr[smallest] = self.arr[smallest], self.arr[i]
            self._heapify_down(smallest)

    def peek(self):
        return self.arr[0] if self.arr else None

# Example
h = MinHeap()
for x in [10, 24, 5, 31, 7]:
    h.insert(x)
print("Heap:", h.arr)
print("Min:", h.peek())
print("Extracted:", h.extract_min())
print("After:", h.arr)
```

#### Why It Matters

- Backbone of priority queues, Dijkstra's, Prim's, A*
- Always gives minimum element in O(1)
- Compact array-based structure (no pointers)
- Excellent for dynamic, ordered sets

#### A Gentle Proof (Why It Works)

Each insertion or deletion affects only a single path (height = log n).
Since every swap improves local order, heap property restored in O(log n).
At any time, parent ≤ children → global min at root.

#### Try It Yourself

1. Implement a max-heap variant.
2. Track the number of swaps per insert.
3. Combine with heap sort by repeated extract_min.
4. Visualize heap as a tree diagram.

#### Test Cases

| Operation     | Input          | Output | Heap After     | Notes     |
| ------------- | -------------- | ------ | -------------- | --------- |
| Insert        | [10,24,5,31,7] | -      | [5,7,10,31,24] | valid     |
| Peek          | -              | 5      | [5,7,10,31,24] | smallest  |
| ExtractMin    | -              | 5      | [7,24,10,31]   | reordered |
| Empty Extract | []             | None   | []             | safe      |

Edge Cases

- Duplicates → handled fine
- Empty heap → return sentinel
- Negative numbers → no problem

#### Complexity

| Operation  | Time     | Space |
| ---------- | -------- | ----- |
| Insert     | O(log n) | O(1)  |
| ExtractMin | O(log n) | O(1)  |
| Peek       | O(1)     | O(1)  |

A min-heap is the quiet organizer, always keeping the smallest task at the top, ready when you are.

### 226 Max Heap Implementation

A max-heap is a binary tree where each parent is greater than or equal to its children. Stored compactly in an array, it guarantees O(1) access to the largest element and O(log n) insertion and deletion, making it ideal for scheduling, leaderboards, and priority-based systems.

#### What Problem Are We Solving?

We want a data structure that efficiently maintains a dynamic set of elements while allowing us to:

- Access the largest element quickly
- Insert and delete efficiently
- Maintain order automatically

A max-heap does exactly that, it always bubbles the biggest element to the top.

#### How Does It Work (Plain Language)?

A binary max-heap is stored as an array, with these relationships:

- `parent(i) = (i - 1) / 2`
- `left(i) = 2i + 1`
- `right(i) = 2i + 2`

Operations:

1. Insert(x):

   * Add `x` at the end.
   * "Bubble up" while `x > parent(x)`.

2. ExtractMax():

   * Remove root (maximum).
   * Move last element to root.
   * "Bubble down" while parent < larger child.

3. Peek():

   * Return `arr[0]` (max element).

Example

Start: `[ ]`
Insert sequence: 10, 24, 5, 31, 7

| Step | Action    | Array                       |
| ---- | --------- | --------------------------- |
| 1    | insert 10 | [10]                        |
| 2    | insert 24 | bubble up → [24, 10]        |
| 3    | insert 5  | [24, 10, 5]                 |
| 4    | insert 31 | bubble up → [31, 24, 5, 10] |
| 5    | insert 7  | [31, 24, 5, 10, 7]          |

ExtractMax:

- Swap root with last (31 ↔ 7): `[7, 24, 5, 10, 31]`
- Remove last: `[7, 24, 5, 10]`
- Bubble down: swap 7 ↔ 24 → `[24, 10, 5, 7]`

#### Tiny Code (Easy Versions)

C

```c
#include <stdio.h>
#define MAX 100

typedef struct {
    int arr[MAX];
    int size;
} MaxHeap;

void swap(int *a, int *b) {
    int tmp = *a; *a = *b; *b = tmp;
}

void heapify_up(MaxHeap *h, int i) {
    while (i > 0) {
        int parent = (i - 1) / 2;
        if (h->arr[i] <= h->arr[parent]) break;
        swap(&h->arr[i], &h->arr[parent]);
        i = parent;
    }
}

void heapify_down(MaxHeap *h, int i) {
    int largest = i;
    int left = 2 * i + 1;
    int right = 2 * i + 2;

    if (left < h->size && h->arr[left] > h->arr[largest]) largest = left;
    if (right < h->size && h->arr[right] > h->arr[largest]) largest = right;

    if (largest != i) {
        swap(&h->arr[i], &h->arr[largest]);
        heapify_down(h, largest);
    }
}

void insert(MaxHeap *h, int val) {
    h->arr[h->size] = val;
    heapify_up(h, h->size);
    h->size++;
}

int extract_max(MaxHeap *h) {
    if (h->size == 0) return -1;
    int root = h->arr[0];
    h->arr[0] = h->arr[--h->size];
    heapify_down(h, 0);
    return root;
}

int peek(MaxHeap *h) {
    return h->size > 0 ? h->arr[0] : -1;
}

int main(void) {
    MaxHeap h = {.size = 0};
    int vals[] = {10, 24, 5, 31, 7};
    for (int i = 0; i < 5; i++) insert(&h, vals[i]);
    printf("Max: %d\n", peek(&h));
    printf("Extracted: %d\n", extract_max(&h));
    for (int i = 0; i < h.size; i++) printf("%d ", h.arr[i]);
    printf("\n");
}
```

Python

```python
class MaxHeap:
    def __init__(self):
        self.arr = []

    def _parent(self, i): return (i - 1) // 2
    def _left(self, i): return 2 * i + 1
    def _right(self, i): return 2 * i + 2

    def insert(self, val):
        self.arr.append(val)
        i = len(self.arr) - 1
        while i > 0 and self.arr[i] > self.arr[self._parent(i)]:
            p = self._parent(i)
            self.arr[i], self.arr[p] = self.arr[p], self.arr[i]
            i = p

    def extract_max(self):
        if not self.arr:
            return None
        root = self.arr[0]
        last = self.arr.pop()
        if self.arr:
            self.arr[0] = last
            self._heapify_down(0)
        return root

    def _heapify_down(self, i):
        largest = i
        l, r = self._left(i), self._right(i)
        if l < len(self.arr) and self.arr[l] > self.arr[largest]:
            largest = l
        if r < len(self.arr) and self.arr[r] > self.arr[largest]:
            largest = r
        if largest != i:
            self.arr[i], self.arr[largest] = self.arr[largest], self.arr[i]
            self._heapify_down(largest)

    def peek(self):
        return self.arr[0] if self.arr else None

# Example
h = MaxHeap()
for x in [10, 24, 5, 31, 7]:
    h.insert(x)
print("Heap:", h.arr)
print("Max:", h.peek())
print("Extracted:", h.extract_max())
print("After:", h.arr)
```

#### Why It Matters

- Priority queues that need fast access to maximum
- Scheduling highest-priority tasks
- Tracking largest elements dynamically
- Used in heap sort, selection problems, top-k queries

#### A Gentle Proof (Why It Works)

Each operation adjusts at most height = log n levels.
Since all swaps move greater elements upward, heap property (parent ≥ children) is restored in logarithmic time.
Thus the root always holds the maximum.

#### Try It Yourself

1. Convert to min-heap by flipping comparisons.
2. Use heap to implement k-largest elements finder.
3. Trace swaps after each insert/delete.
4. Visualize heap as a tree diagram.

#### Test Cases

| Operation     | Input          | Output | Heap After     | Notes       |
| ------------- | -------------- | ------ | -------------- | ----------- |
| Insert        | [10,24,5,31,7] | -      | [31,24,5,10,7] | max at root |
| Peek          | -              | 31     | [31,24,5,10,7] | largest     |
| ExtractMax    | -              | 31     | [24,10,5,7]    | heap fixed  |
| Empty Extract | []             | None   | []             | safe        |

Edge Cases

- Duplicates → handled fine
- Empty heap → sentinel
- Negative numbers → valid

#### Complexity

| Operation  | Time     | Space |
| ---------- | -------- | ----- |
| Insert     | O(log n) | O(1)  |
| ExtractMax | O(log n) | O(1)  |
| Peek       | O(1)     | O(1)  |

A max-heap is the summit keeper, always crowning the greatest, ensuring the hierarchy remains true from root to leaf.

### 227 Fibonacci Heap Insert/Delete

A Fibonacci heap is a meldable heap optimized for very fast amortized operations. It keeps a collection of heap-ordered trees with minimal structural work on most operations, banking work for occasional consolidations. Classic result: `insert`, `find-min`, and `decrease-key` run in O(1) amortized, while `extract-min` runs in O(log n) amortized.

#### What Problem Are We Solving?

We want ultra fast priority queue ops in algorithms that call `decrease-key` a lot, like Dijkstra and Prim. Binary and pairing heaps give `decrease-key` in `O(log n)` or good constants, but Fibonacci heaps achieve amortized O(1) for `insert` and `decrease-key`, improving theoretical bounds.

Goal:
Maintain a set of heap-ordered trees where most updates only touch a few pointers, delaying consolidation until `extract-min`.

#### How Does It Work (Plain Language)?

Structure highlights

- A root list of trees, each obeying min-heap order.
- A pointer to the global minimum root.
- Each node stores degree, parent, child, and a mark bit used by `decrease-key`.
- Root list is a circular doubly linked list, children lists are similar.

Core ideas

- Insert adds a 1 node tree into the root list and updates `min` if needed.
- Delete is typically implemented as `decrease-key(x, -inf)` then `extract-min`.
- Extract-min removes the min root, promotes its children to the root list, then consolidates roots by linking trees of equal degree until all root degrees are unique.

#### How Does Insert Work

Steps for `insert(x)`

1. Make a singleton node `x`.
2. Splice `x` into the root list.
3. Update `min` if `x.key < min.key`.

Example Steps (Root List View)

| Step | Action    | Root List      | Min |
| ---- | --------- | -------------- | --- |
| 1    | insert 12 | [12]           | 12  |
| 2    | insert 7  | [12, 7]        | 7   |
| 3    | insert 25 | [12, 7, 25]    | 7   |
| 4    | insert 3  | [12, 7, 25, 3] | 3   |

All inserts are O(1) pointer splices.

#### How Does Delete Work

To delete an arbitrary node `x`, standard approach

1. `decrease-key(x, -inf)` so `x` becomes the minimum.
2. `extract-min()` to remove it.

Delete inherits the `extract-min` cost.

#### Tiny Code (Educational Skeleton)

This is a minimal sketch to illustrate structure and the two requested operations. It omits full `decrease-key` and cascading cuts to keep focus. In practice, a complete Fibonacci heap also implements `decrease-key` and `extract-min` with consolidation arrays.

Python (didactic skeleton for insert and delete via extract-min path)

```python
class FibNode:
    def __init__(self, key):
        self.key = key
        self.degree = 0
        self.mark = False
        self.parent = None
        self.child = None
        # circular doubly linked list pointers
        self.left = self
        self.right = self

def _splice(a, b):
    # insert node b to the right of node a in a circular list
    b.right = a.right
    b.left = a
    a.right.left = b
    a.right = b

def _remove_from_list(x):
    x.left.right = x.right
    x.right.left = x.left
    x.left = x.right = x

class FibHeap:
    def __init__(self):
        self.min = None
        self.n = 0

    def insert(self, key):
        x = FibNode(key)
        if self.min is None:
            self.min = x
        else:
            _splice(self.min, x)
            if x.key < self.min.key:
                self.min = x
        self.n += 1
        return x

    def _merge_root_list(self, other_min):
        if other_min is None:
            return
        # concat circular lists: self.min and other_min
        a = self.min
        b = other_min
        a_right = a.right
        b_left = b.left
        a.right = b
        b.left = a
        a_right.left = b_left
        b_left.right = a_right
        if b.key < self.min.key:
            self.min = b

    def extract_min(self):
        z = self.min
        if z is None:
            return None
        # promote children to root list
        if z.child:
            c = z.child
            nodes = []
            cur = c
            while True:
                nodes.append(cur)
                cur = cur.right
                if cur == c:
                    break
            for x in nodes:
                x.parent = None
                _remove_from_list(x)
                _splice(z, x)  # into root list near z
        # remove z from root list
        if z.right == z:
            self.min = None
        else:
            nxt = z.right
            _remove_from_list(z)
            self.min = nxt
            self._consolidate()  # real impl would link equal degree trees
        self.n -= 1
        return z.key

    def _consolidate(self):
        # Placeholder: a full implementation uses an array A[0..floor(log_phi n)]
        # to link roots of equal degree until all degrees unique.
        pass

    def delete(self, node):
        # In full Fibonacci heap:
        # decrease_key(node, -inf), then extract_min
        # Here we approximate by manual min-bump if node is the min
        # and ignore cascading cuts for brevity.
        # For correctness in real use, implement decrease_key and cuts.
        node.key = float("-inf")
        if self.min and node.key < self.min.key:
            self.min = node
        return self.extract_min()

# Example
H = FibHeap()
n1 = H.insert(12)
n2 = H.insert(7)
n3 = H.insert(25)
n4 = H.insert(3)
print("Extracted min:", H.extract_min())  # 3
```

Note This skeleton shows how `insert` stitches into the root list and how `extract_min` would promote children. A production version must implement `decrease-key` with cascading cuts and `_consolidate` that links trees of equal degree.

#### Why It Matters

- Theoretical speedups for graph algorithms heavy on `decrease-key`
- Meld operation can be O(1) by concatenating root lists
- Amortized guarantees backed by potential function analysis

#### A Gentle Proof (Why It Works)

Amortized analysis uses a potential function based on number of trees in the root list and number of marked nodes.

- `insert` only adds a root and maybe updates `min`, decreasing or slightly increasing potential, so O(1) amortized.
- `decrease-key` cuts a node and possibly cascades parent cuts, but marks bound the total number of cascades over a sequence, giving O(1) amortized.
- `extract-min` triggers consolidation. The number of distinct degrees is O(log n), so linking costs O(log n) amortized.

#### Try It Yourself

1. Complete `_consolidate` with an array indexed by degree, linking roots of equal degree until unique.
2. Implement `decrease_key(x, new_key)` with cut and cascading cut rules.
3. Add `union(H1, H2)` that concatenates root lists and picks the smaller `min`.
4. Benchmark against binary and pairing heaps on workloads with many `decrease-key` operations.

#### Test Cases

| Operation    | Input              | Expected              | Notes                                       |
| ------------ | ------------------ | --------------------- | ------------------------------------------- |
| insert       | 12, 7, 25, 3       | min = 3               | simple root list updates                    |
| extract_min  | after above        | returns 3             | children promoted, consolidate              |
| delete(node) | delete 7           | 7 removed             | via decrease to minus infinity then extract |
| meld         | union of two heaps | new min = min(m1, m2) | O(1) concat                                 |

Edge Cases

- Extract from empty heap returns None
- Duplicate keys are fine
- Large inserts without extract build many small trees until consolidation

#### Complexity

| Operation    | Amortized Time                     |
| ------------ | ---------------------------------- |
| insert       | O(1)                               |
| find-min     | O(1)                               |
| decrease-key | O(1)                               |
| extract-min  | O(log n)                           |
| delete       | O(log n) via decrease then extract |

Fibonacci heaps trade strict order for lazy elegance. Most ops are tiny pointer shuffles, and the heavy lifting happens rarely during consolidation.

### 228 Pairing Heap Merge

A pairing heap is a simple, pointer-based, meldable heap that is famously fast in practice. Its secret weapon is the merge operation: link two heap roots by making the larger-key root a child of the smaller-key root. Many other operations reduce to a small number of merges.

#### What Problem Are We Solving?

We want a priority queue with a tiny constant factor and very simple code, while keeping theoretical guarantees close to Fibonacci heaps. Pairing heaps offer extremely quick `insert`, `meld`, and often `decrease-key`, with `delete-min` powered by a lightweight multi-merge.

Goal:
Represent a heap as a tree of nodes and implement merge so that all higher-level operations can be expressed as sequences of merges.

#### How Does It Work (Plain Language)?

Each node has: key, first child, and next sibling. The heap is just a pointer to the root.
To merge two heaps `A` and `B`:

1. If one is empty, return the other.
2. Compare roots.
3. Make the larger-root heap the new child of the smaller-root heap by linking it as the smaller root's first child.

Other ops via merge

- Insert(x): make a 1-node heap and `merge(root, x)`.
- Find-min: the root's key.
- Delete-min: remove the root, then merge its children in two passes

  1. Left to right, pairwise merge adjacent siblings.
  2. Right to left, merge the resulting heaps back into one.
- Decrease-key(x, new): cut `x` from its place, set `x.key = new`, then `merge(root, x)`.

Example Steps (Merge only)

| Step | Heap A (root) | Heap B (root) | Action          | New Root |
| ---- | ------------- | ------------- | --------------- | -------- |
| 1    | 7             | 12            | link 12 under 7 | 7        |
| 2    | 7             | 3             | link 7 under 3  | 3        |
| 3    | 3             | 9             | link 9 under 3  | 3        |

Resulting root is the minimum of all merged heaps.

#### Tiny Code (Easy Versions)

C (merge, insert, find-min, delete-min two-pass)

```c
#include <stdio.h>
#include <stdlib.h>

typedef struct Node {
    int key;
    struct Node *child;
    struct Node *sibling;
} Node;

Node* make_node(int key) {
    Node* n = malloc(sizeof(Node));
    n->key = key; n->child = NULL; n->sibling = NULL;
    return n;
}

Node* merge(Node* a, Node* b) {
    if (!a) return b;
    if (!b) return a;
    if (b->key < a->key) { Node* t = a; a = b; b = t; }
    // make b the first child of a
    b->sibling = a->child;
    a->child = b;
    return a;
}

Node* insert(Node* root, int key) {
    return merge(root, make_node(key));
}

Node* merge_pairs(Node* first) {
    if (!first || !first->sibling) return first;
    Node* a = first;
    Node* b = first->sibling;
    Node* rest = b->sibling;
    a->sibling = b->sibling = NULL;
    return merge(merge(a, b), merge_pairs(rest));
}

Node* delete_min(Node* root, int* out) {
    if (!root) return NULL;
    *out = root->key;
    Node* new_root = merge_pairs(root->child);
    free(root);
    return new_root;
}

int main(void) {
    Node* h = NULL;
    h = insert(h, 7);
    h = insert(h, 12);
    h = insert(h, 3);
    h = insert(h, 9);
    int m;
    h = delete_min(h, &m);
    printf("Deleted min: %d\n", m); // 3
    return 0;
}
```

Python (succinct pairing heap)

```python
class Node:
    __slots__ = ("key", "child", "sibling")
    def __init__(self, key):
        self.key = key
        self.child = None
        self.sibling = None

def merge(a, b):
    if not a: return b
    if not b: return a
    if b.key < a.key:
        a, b = b, a
    b.sibling = a.child
    a.child = b
    return a

def merge_pairs(first):
    if not first or not first.sibling:
        return first
    a, b, rest = first, first.sibling, first.sibling.sibling
    a.sibling = b.sibling = None
    return merge(merge(a, b), merge_pairs(rest))

class PairingHeap:
    def __init__(self): self.root = None
    def find_min(self): return None if not self.root else self.root.key
    def insert(self, x):
        self.root = merge(self.root, Node(x))
    def meld(self, other):
        self.root = merge(self.root, other.root)
    def delete_min(self):
        if not self.root: return None
        m = self.root.key
        self.root = merge_pairs(self.root.child)
        return m

# Example
h = PairingHeap()
for x in [7, 12, 3, 9]:
    h.insert(x)
print(h.find_min())      # 3
print(h.delete_min())    # 3
print(h.find_min())      # 7
```

#### Why It Matters

- Incredibly simple code yet high performance in practice
- Meld is constant time pointer work
- Excellent for workloads mixing frequent inserts and decrease-keys
- A strong practical alternative to Fibonacci heaps

#### A Gentle Proof (Why It Works)

Merge correctness

- After linking, the smaller root remains parent, so heap order holds at the root and along the newly attached subtree.

Delete-min two-pass

- Pairwise merges reduce the number of trees while keeping roots small.
- The second right-to-left fold merges larger partial heaps into a single heap.
- Analyses show `delete-min` runs in O(log n) amortized; `insert` and `meld` in O(1) amortized.
- `decrease-key` is conjectured O(1) amortized in practice and near that in theory under common models.

#### Try It Yourself

1. Implement `decrease_key(node, new_key)`: cut the node from its parent and `merge(root, node)` after lowering its key.
2. Add a handle table to access nodes for fast decrease-key.
3. Benchmark against binary and Fibonacci heaps on Dijkstra workloads.
4. Visualize delete-min's two-pass pairing on random trees.

#### Test Cases

| Operation  | Input          | Output                 | Notes                  |
| ---------- | -------------- | ---------------------- | ---------------------- |
| insert     | 7, 12, 3, 9    | min = 3                | root tracks global min |
| meld       | meld two heaps | new min is min of both | constant-time link     |
| delete_min | after above    | 3                      | two-pass pairing       |
| delete_min | next           | 7                      | heap restructures      |

Edge Cases

- Merging with empty heap returns the other heap
- Duplicate keys behave naturally
- Single-node heap deletes to empty safely

#### Complexity

| Operation    | Amortized Time                                | Notes           |
| ------------ | --------------------------------------------- | --------------- |
| meld         | O(1)                                          | core primitive  |
| insert       | O(1)                                          | merge singleton |
| find-min     | O(1)                                          | root key        |
| delete-min   | O(log n)                                      | two-pass merge  |
| decrease-key | O(1) practical, near O(1) amortized in models | cut+merge       |

Pairing heaps make merging feel effortless: one comparison, a couple of pointers, and you are done.

### 229 Binomial Heap Merge

A binomial heap is a set of binomial trees that act like a binary counter for priority queues. The star move is merge: combining two heaps is like adding two binary numbers. Trees of the same degree collide, you link one under the other, and carry to the next degree.

#### What Problem Are We Solving?

We want a priority queue that supports fast meld (union) while keeping simple, provable bounds. Binomial heaps deliver:

- `meld` in O(log n)
- `insert` in O(1) amortized by melding a 1-node heap
- `find-min` in O(log n)
- `delete-min` in O(log n) via a meld with reversed children

Goal:
Represent the heap as a sorted list of binomial trees and merge two heaps by walking these lists, linking equal-degree roots.

#### How Does It Work (Plain Language)?

Binomial tree facts

- A binomial tree of degree `k` has `2^k` nodes.
- Each degree occurs at most once in a binomial heap.
- Roots are kept in increasing order of degree.

Merging two heaps H1 and H2

1. Merge the root lists by degree like a sorted list merge.
2. Walk the combined list. Whenever two consecutive trees have the same degree, link them: make the larger-key root a child of the smaller-key root, increasing the degree by 1.
3. Use a carry idea just like binary addition.

Link(u, v)

- Precondition: `degree(u) == degree(v)`.
- After linking, `min(u.key, v.key)` becomes parent and degree increases by 1.

Example Steps (degrees in parentheses)

Start

- H1 roots: [2(0), 7(1), 12(3)]
- H2 roots: [3(0), 9(2)]

1 Merge lists by degree

- Combined: [2(0), 3(0), 7(1), 9(2), 12(3)]

2 Resolve equal degrees

- Link 2(0) and 3(0) under min root → 2 becomes parent: 2(1)
- Now list: [2(1), 7(1), 9(2), 12(3)]
- Link 2(1) and 7(1) → 2 becomes parent: 2(2)
- Now list: [2(2), 9(2), 12(3)]
- Link 2(2) and 9(2) → 2 becomes parent: 2(3)
- Now list: [2(3), 12(3)]
- Link 2(3) and 12(3) → 2 becomes parent: 2(4)

Final heap has root list [2(4)] with 2 as global min.

#### Tiny Code (Easy Versions)

C (merge plus link, minimal skeleton)

```c
#include <stdio.h>
#include <stdlib.h>

typedef struct Node {
    int key;
    int degree;
    struct Node* parent;
    struct Node* child;
    struct Node* sibling; // next root or next sibling in child list
} Node;

Node* make_node(int key){
    Node* x = calloc(1, sizeof(Node));
    x->key = key;
    return x;
}

// Make 'y' a child of 'x' assuming x->key <= y->key and degree equal
static Node* link_tree(Node* x, Node* y){
    y->parent = x;
    y->sibling = x->child;
    x->child = y;
    x->degree += 1;
    return x;
}

// Merge root lists by degree (no linking yet)
static Node* merge_root_lists(Node* a, Node* b){
    if(!a) return b;
    if(!b) return a;
    Node dummy = {0};
    Node* tail = &dummy;
    while(a && b){
        if(a->degree <= b->degree){
            tail->sibling = a; a = a->sibling;
        }else{
            tail->sibling = b; b = b->sibling;
        }
        tail = tail->sibling;
    }
    tail->sibling = a ? a : b;
    return dummy.sibling;
}

// Union with carry logic
Node* binomial_union(Node* h1, Node* h2){
    Node* head = merge_root_lists(h1, h2);
    if(!head) return NULL;

    Node* prev = NULL;
    Node* curr = head;
    Node* next = curr->sibling;

    while(next){
        if(curr->degree != next->degree || (next->sibling && next->sibling->degree == curr->degree)){
            prev = curr;
            curr = next;
        }else{
            if(curr->key <= next->key){
                curr->sibling = next->sibling;
                curr = link_tree(curr, next);
            }else{
                if(prev) prev->sibling = next;
                else head = next;
                curr = link_tree(next, curr);
            }
        }
        next = curr->sibling;
    }
    return head;
}

// Convenience: insert by union with 1-node heap
Node* insert(Node* heap, int key){
    return binomial_union(heap, make_node(key));
}

int main(void){
    Node* h1 = NULL;
    Node* h2 = NULL;
    h1 = insert(h1, 2);
    h1 = insert(h1, 7);
    h1 = insert(h1, 12); // degrees will normalize after unions
    h2 = insert(h2, 3);
    h2 = insert(h2, 9);
    Node* h = binomial_union(h1, h2);
    // h now holds the merged heap; find-min is a scan of root list
    for(Node* r = h; r; r = r->sibling)
        printf("root key=%d deg=%d\n", r->key, r->degree);
    return 0;
}
```

Python (succinct union and link)

```python
class Node:
    __slots__ = ("key", "degree", "parent", "child", "sibling")
    def __init__(self, key):
        self.key = key
        self.degree = 0
        self.parent = None
        self.child = None
        self.sibling = None

def link_tree(x, y):
    # assume x.key <= y.key and degrees equal
    y.parent = x
    y.sibling = x.child
    x.child = y
    x.degree += 1
    return x

def merge_root_lists(a, b):
    if not a: return b
    if not b: return a
    dummy = Node(-1)
    t = dummy
    while a and b:
        if a.degree <= b.degree:
            t.sibling, a = a, a.sibling
        else:
            t.sibling, b = b, b.sibling
        t = t.sibling
    t.sibling = a if a else b
    return dummy.sibling

def binomial_union(h1, h2):
    head = merge_root_lists(h1, h2)
    if not head: return None
    prev, curr, nxt = None, head, head.sibling
    while nxt:
        if curr.degree != nxt.degree or (nxt.sibling and nxt.sibling.degree == curr.degree):
            prev, curr, nxt = curr, nxt, nxt.sibling
        else:
            if curr.key <= nxt.key:
                curr.sibling = nxt.sibling
                curr = link_tree(curr, nxt)
                nxt = curr.sibling
            else:
                if prev: prev.sibling = nxt
                else: head = nxt
                curr = link_tree(nxt, curr)
                nxt = curr.sibling
    return head

def insert(heap, key):
    return binomial_union(heap, Node(key))

# Example
h1 = None
for x in [2, 7, 12]:
    h1 = insert(h1, x)
h2 = None
for x in [3, 9]:
    h2 = insert(h2, x)
h = binomial_union(h1, h2)
roots = []
r = h
while r:
    roots.append((r.key, r.degree))
    r = r.sibling
print(roots)  # e.g. [(2, 4)] or a small set of unique degrees with min root smallest
```

#### Why It Matters

- Meld-friendly: merging heaps is first-class, not an afterthought
- Clean, provable bounds with a binary-counter intuition
- Foundation for Fibonacci heaps and variations
- Great when frequent melds are required in algorithms or multi-queue systems

#### A Gentle Proof (Why It Works)

Merging root lists produces degrees in nondecreasing order. Linking only happens between adjacent roots of the same degree, producing exactly one tree of each degree after all carries settle. This mirrors binary addition: each link corresponds to carrying a 1 to the next bit. Since the maximum degree is O(log n), the merge performs O(log n) links and scans, giving O(log n) time.

#### Try It Yourself

1. Implement `find_min` by scanning root list and keep a pointer to the min root.
2. Implement `delete_min`: remove min root, reverse its child list into a separate heap, then `union`.
3. Add `decrease_key` by cutting and reinserting a node in the root list, then fixing parent order.
4. Compare union time with binary heap and pairing heap on large random workloads.

#### Test Cases

| Case             | H1 Roots (deg)      | H2 Roots (deg) | Result                   | Notes                 |
| ---------------- | ------------------- | -------------- | ------------------------ | --------------------- |
| Simple merge     | [2(0), 7(1)]        | [3(0)]         | roots unique after link  | 2 becomes parent of 3 |
| Chain of carries | [2(0), 7(1), 12(3)] | [3(0), 9(2)]   | single root 2(4)         | cascading links       |
| Insert by union  | H with roots        | plus [x(0)]    | merged in O(1) amortized | single carry possible |

Edge Cases

- Merging with empty heap returns the other heap
- Duplicate keys work; tie break arbitrarily
- Maintain stable sibling pointers when linking

#### Complexity

| Operation    | Time                                        |
| ------------ | ------------------------------------------- |
| meld (union) | O(log n)                                    |
| insert       | O(1) amortized via union with 1-node heap   |
| find-min     | O(log n) scan or keep pointer for O(1) peek |
| delete-min   | O(log n)                                    |
| decrease-key | O(log n) typical implementation             |

Merging binomial heaps feels like adding binary numbers: equal degrees collide, link, and carry forward until the structure is tidy and the minimum stands at a root.

### 230 Leftist Heap Merge

A leftist heap is a binary tree heap optimized for efficient merges. Its structure skews to the left so that merging two heaps can be done recursively in O(log n) time. The clever trick is storing each node's null path length (npl), ensuring the shortest path to a null child is always on the right, which keeps merges shallow.

#### What Problem Are We Solving?

We want a merge-friendly heap with simpler structure than Fibonacci or pairing heaps but faster merges than standard binary heaps. The leftist heap is a sweet spot: fast merges, simple code, and still supports all key heap operations.

Goal:
Design a heap that keeps its shortest subtree on the right, so recursive merges stay logarithmic.

#### How Does It Work (Plain Language)?

Each node stores:

- `key` – value used for ordering
- `left`, `right` – child pointers
- `npl` – null path length (distance to nearest null)

Rules:

1. Heap order: parent key ≤ child keys (min-heap)
2. Leftist property: `npl(left) ≥ npl(right)`

Merge(a, b):

1. If one is null, return the other.
2. Compare roots, smaller root becomes new root.
3. Recursively merge `a.right` and `b`.
4. After merge, swap children if needed to keep leftist property.
5. Update `npl`.

Other operations via merge

- Insert(x): merge heap with single-node heap `x`.
- DeleteMin(): merge left and right subtrees of root.

Example (Min-Heap Merge)

Heap A: root 3

```
   3
  / \
 5   9
```

Heap B: root 4

```
  4
 / \
 8  10
```

Merge(3, 4):

- 3 < 4 → new root 3
- Merge right(9) with heap 4
- After merge:

```
     3
    / \
   5   4
      / \
     8  10
        /
       9
```

Rebalance by swapping if right's npl > left's → ensures leftist shape.

#### Tiny Code (Easy Versions)

C (Merge, Insert, Delete-Min)

```c
#include <stdio.h>
#include <stdlib.h>

typedef struct Node {
    int key;
    int npl;
    struct Node *left, *right;
} Node;

Node* make_node(int key) {
    Node* n = malloc(sizeof(Node));
    n->key = key;
    n->npl = 0;
    n->left = n->right = NULL;
    return n;
}

int npl(Node* x) { return x ? x->npl : -1; }

Node* merge(Node* a, Node* b) {
    if (!a) return b;
    if (!b) return a;
    if (b->key < a->key) { Node* t = a; a = b; b = t; }
    a->right = merge(a->right, b);
    // maintain leftist property
    if (npl(a->left) < npl(a->right)) {
        Node* t = a->left;
        a->left = a->right;
        a->right = t;
    }
    a->npl = npl(a->right) + 1;
    return a;
}

Node* insert(Node* h, int key) {
    return merge(h, make_node(key));
}

Node* delete_min(Node* h, int* out) {
    if (!h) return NULL;
    *out = h->key;
    Node* new_root = merge(h->left, h->right);
    free(h);
    return new_root;
}

int main(void) {
    Node* h1 = NULL;
    int vals[] = {5, 3, 9, 7, 4};
    for (int i = 0; i < 5; i++)
        h1 = insert(h1, vals[i]);

    int m;
    h1 = delete_min(h1, &m);
    printf("Deleted min: %d\n", m);
    return 0;
}
```

Python

```python
class Node:
    __slots__ = ("key", "npl", "left", "right")
    def __init__(self, key):
        self.key = key
        self.npl = 0
        self.left = None
        self.right = None

def npl(x): return x.npl if x else -1

def merge(a, b):
    if not a: return b
    if not b: return a
    if b.key < a.key:
        a, b = b, a
    a.right = merge(a.right, b)
    # enforce leftist property
    if npl(a.left) < npl(a.right):
        a.left, a.right = a.right, a.left
    a.npl = npl(a.right) + 1
    return a

def insert(h, key):
    return merge(h, Node(key))

def delete_min(h):
    if not h: return None, None
    m = h.key
    h = merge(h.left, h.right)
    return m, h

# Example
h = None
for x in [5, 3, 9, 7, 4]:
    h = insert(h, x)
m, h = delete_min(h)
print("Deleted min:", m)
```

#### Why It Matters

- Fast merge with simple recursion
- Ideal for priority queues with frequent unions
- Cleaner implementation than Fibonacci heaps
- Guarantees logarithmic merge and delete-min

#### A Gentle Proof (Why It Works)

The null path length (npl) ensures that the right spine is always the shortest. Therefore, each merge step recurses down only one right path, not both. This bounds recursion depth by O(log n).
Every other operation (insert, delete-min) is defined as a small number of merges, hence O(log n).

#### Try It Yourself

1. Trace the merge process for two 3-node heaps.
2. Visualize `npl` values after each step.
3. Implement `find_min()` (just return root key).
4. Try making a max-heap variant by flipping comparisons.

#### Test Cases

| Operation   | Input         | Output     | Notes                       |
| ----------- | ------------- | ---------- | --------------------------- |
| Insert      | [5,3,9,7,4]   | root=3     | min-heap property holds     |
| DeleteMin   | remove 3      | new root=4 | leftist property maintained |
| Merge       | [3,5] + [4,6] | root=3     | right spine ≤ log n         |
| Empty merge | None + [5]    | [5]        | safe                        |

Edge Cases

- Merge with null heap → returns the other
- Duplicate keys → ties fine
- Single node → npl = 0

#### Complexity

| Operation | Time     | Space |
| --------- | -------- | ----- |
| Merge     | O(log n) | O(1)  |
| Insert    | O(log n) | O(1)  |
| DeleteMin | O(log n) | O(1)  |
| FindMin   | O(1)     | O(1)  |

The leftist heap is like a river always bending toward the left, shaping itself so merges flow swiftly and naturally.

## Section 24. Balanced Trees 

### 231 AVL Tree Insert

An AVL tree is a self-balancing binary search tree where the height difference (balance factor) between the left and right subtrees of any node is at most 1. This invariant guarantees O(log n) lookup, insertion, and deletion, making AVL trees a classic example of maintaining order dynamically.

#### What Problem Are We Solving?

Ordinary binary search trees can become skewed (like linked lists) after unlucky insertions, degrading performance to O(n). The AVL tree restores balance automatically after each insertion, ensuring searches and updates stay fast.

Goal:
Maintain a balanced search tree by rotating nodes after insertions so that height difference ≤ 1 everywhere.

#### How Does It Work (Plain Language)?

1. Insert the key as in a normal BST.
2. Walk back up the recursion updating heights.
3. Check balance factor = `height(left) - height(right)`.
4. If it's outside {−1, 0, +1}, perform one of four rotations to restore balance:

| Case             | Pattern                           | Fix                              |
| ---------------- | --------------------------------- | -------------------------------- |
| Left-Left (LL)   | Inserted into left-left subtree   | Rotate right                     |
| Right-Right (RR) | Inserted into right-right subtree | Rotate left                      |
| Left-Right (LR)  | Inserted into left-right subtree  | Rotate left at child, then right |
| Right-Left (RL)  | Inserted into right-left subtree  | Rotate right at child, then left |

Example

Insert 30, 20, 10

- 30 → root
- 20 → left of 30
- 10 → left of 20
  → imbalance at 30: balance factor = 2
  → LL case → right rotate on 30

Balanced tree:

```
     20
    /  \
   10   30
```

#### Step-by-Step Example

Insert sequence: 10, 20, 30, 40, 50

| Step | Insert | Tree (in-order)  | Imbalance  | Rotation | Root After |
| ---- | ------ | ---------------- | ---------- | -------- | ---------- |
| 1    | 10     | [10]             | -          | -        | 10         |
| 2    | 20     | [10,20]          | balanced   | -        | 10         |
| 3    | 30     | [10,20,30]       | at 10 (RR) | left     | 20         |
| 4    | 40     | [10,20,30,40]    | balanced   | -        | 20         |
| 5    | 50     | [10,20,30,40,50] | at 20 (RR) | left     | 30         |

Balanced final tree:

```
     30
    /  \
   20   40
  /       \
 10        50
```

#### Tiny Code (Easy Versions)

C

```c
#include <stdio.h>
#include <stdlib.h>

typedef struct Node {
    int key, height;
    struct Node *left, *right;
} Node;

int height(Node* n) { return n ? n->height : 0; }
int max(int a, int b) { return a > b ? a : b; }

Node* new_node(int key) {
    Node* n = malloc(sizeof(Node));
    n->key = key;
    n->height = 1;
    n->left = n->right = NULL;
    return n;
}

Node* rotate_right(Node* y) {
    Node* x = y->left;
    Node* T2 = x->right;
    x->right = y;
    y->left = T2;
    y->height = 1 + max(height(y->left), height(y->right));
    x->height = 1 + max(height(x->left), height(x->right));
    return x;
}

Node* rotate_left(Node* x) {
    Node* y = x->right;
    Node* T2 = y->left;
    y->left = x;
    x->right = T2;
    x->height = 1 + max(height(x->left), height(x->right));
    y->height = 1 + max(height(y->left), height(y->right));
    return y;
}

int balance(Node* n) { return n ? height(n->left) - height(n->right) : 0; }

Node* insert(Node* node, int key) {
    if (!node) return new_node(key);
    if (key < node->key) node->left = insert(node->left, key);
    else if (key > node->key) node->right = insert(node->right, key);
    else return node; // no duplicates

    node->height = 1 + max(height(node->left), height(node->right));
    int bf = balance(node);

    // LL
    if (bf > 1 && key < node->left->key)
        return rotate_right(node);
    // RR
    if (bf < -1 && key > node->right->key)
        return rotate_left(node);
    // LR
    if (bf > 1 && key > node->left->key) {
        node->left = rotate_left(node->left);
        return rotate_right(node);
    }
    // RL
    if (bf < -1 && key < node->right->key) {
        node->right = rotate_right(node->right);
        return rotate_left(node);
    }
    return node;
}

void inorder(Node* root) {
    if (!root) return;
    inorder(root->left);
    printf("%d ", root->key);
    inorder(root->right);
}

int main(void) {
    Node* root = NULL;
    int keys[] = {10, 20, 30, 40, 50};
    for (int i = 0; i < 5; i++)
        root = insert(root, keys[i]);
    inorder(root);
    printf("\n");
}
```

Python

```python
class Node:
    def __init__(self, key):
        self.key = key
        self.left = self.right = None
        self.height = 1

def height(n): return n.height if n else 0
def balance(n): return height(n.left) - height(n.right) if n else 0

def rotate_right(y):
    x, T2 = y.left, y.left.right
    x.right, y.left = y, T2
    y.height = 1 + max(height(y.left), height(y.right))
    x.height = 1 + max(height(x.left), height(x.right))
    return x

def rotate_left(x):
    y, T2 = x.right, x.right.left
    y.left, x.right = x, T2
    x.height = 1 + max(height(x.left), height(x.right))
    y.height = 1 + max(height(y.left), height(y.right))
    return y

def insert(node, key):
    if not node: return Node(key)
    if key < node.key: node.left = insert(node.left, key)
    elif key > node.key: node.right = insert(node.right, key)
    else: return node

    node.height = 1 + max(height(node.left), height(node.right))
    bf = balance(node)

    if bf > 1 and key < node.left.key:  # LL
        return rotate_right(node)
    if bf < -1 and key > node.right.key:  # RR
        return rotate_left(node)
    if bf > 1 and key > node.left.key:  # LR
        node.left = rotate_left(node.left)
        return rotate_right(node)
    if bf < -1 and key < node.right.key:  # RL
        node.right = rotate_right(node.right)
        return rotate_left(node)
    return node

def inorder(root):
    if not root: return
    inorder(root.left)
    print(root.key, end=' ')
    inorder(root.right)

# Example
root = None
for k in [10, 20, 30, 40, 50]:
    root = insert(root, k)
inorder(root)
```

#### Why It Matters

- Guarantees O(log n) operations
- Prevents degeneration into linear chains
- Clear rotation-based balancing logic
- Basis for other balanced trees (e.g., Red-Black)

#### A Gentle Proof (Why It Works)

Each insertion may unbalance at most one node, the lowest ancestor of the inserted node.
A single rotation (or double rotation) restores the balance factor of that node to {−1, 0, +1},
and updates all affected heights in constant time.
Thus each insertion performs O(1) rotations, O(log n) recursive updates.

#### Try It Yourself

1. Insert 30, 20, 10 → LL case
2. Insert 10, 30, 20 → LR case
3. Insert 30, 10, 20 → RL case
4. Insert 10, 20, 30 → RR case
5. Draw each tree before and after rotation

#### Test Cases

| Sequence     | Rotation Type | Final Root | Height |
| ------------ | ------------- | ---------- | ------ |
| [30, 20, 10] | LL            | 20         | 2      |
| [10, 30, 20] | LR            | 20         | 2      |
| [30, 10, 20] | RL            | 20         | 2      |
| [10, 20, 30] | RR            | 20         | 2      |

Edge Cases

- Duplicate keys ignored
- Insertion into empty tree → new node
- All ascending or descending inserts → balanced via rotations

#### Complexity

| Operation | Time     | Space          |
| --------- | -------- | -------------- |
| Insert    | O(log n) | O(h) recursion |
| Search    | O(log n) | O(h)           |
| Delete    | O(log n) | O(h)           |

The AVL tree is a careful gardener, pruning imbalance wherever it grows, so your searches always find a straight path.

### 232 AVL Tree Delete

Deleting a node in an AVL tree is like removing a block from a carefully balanced tower, you take it out, then perform rotations to restore equilibrium. The key is to combine BST deletion rules with balance factor checks and rebalancing up the path.

#### What Problem Are We Solving?

Deletion in a plain BST can break the shape, making it skewed and inefficient. In an AVL tree, we want to:

1. Remove a node (using standard BST deletion)
2. Recalculate heights and balance factors
3. Restore balance with rotations

This ensures the tree remains height-balanced, keeping operations at O(log n).

#### How Does It Work (Plain Language)?

1. Find the node as in a normal BST.
2. Delete it:

   * If leaf → remove directly.
   * If one child → replace with child.
   * If two children → find inorder successor, copy its value, delete it recursively.
3. Walk upward, update height and balance factor at each ancestor.
4. Apply one of four rotation cases if unbalanced:

| Case | Condition                           | Fix                              |
| ---- | ----------------------------------- | -------------------------------- |
| LL   | balance > 1 and balance(left) ≥ 0   | Rotate right                     |
| LR   | balance > 1 and balance(left) < 0   | Rotate left at child, then right |
| RR   | balance < -1 and balance(right) ≤ 0 | Rotate left                      |
| RL   | balance < -1 and balance(right) > 0 | Rotate right at child, then left |

#### Example

Delete 10 from

```
     20
    /  \
   10   30
```

- Remove leaf 10
- Node 20: balance = 0 → balanced

Now delete 30:

```
    20
   /
 10
```

- balance(20) = +1 → still balanced

Delete 10 next:

```
20
```

Tree becomes single node, still AVL.

#### Step-by-Step Example

Insert [10, 20, 30, 40, 50, 25]
Then delete 40

| Step | Action       | Imbalance            | Rotation | Root |
| ---- | ------------ | -------------------- | -------- | ---- |
| 1    | Delete 40    | at 30 (balance = -2) | RL       | 30   |
| 2    | After rotate | balanced             | -        | 30   |

#### Tiny Code (Easy Versions)

C

```c
#include <stdio.h>
#include <stdlib.h>

typedef struct Node {
    int key, height;
    struct Node *left, *right;
} Node;

int height(Node* n) { return n ? n->height : 0; }
int max(int a, int b) { return a > b ? a : b; }

Node* new_node(int key) {
    Node* n = malloc(sizeof(Node));
    n->key = key; n->height = 1;
    n->left = n->right = NULL;
    return n;
}

Node* rotate_right(Node* y) {
    Node* x = y->left;
    Node* T2 = x->right;
    x->right = y;
    y->left = T2;
    y->height = 1 + max(height(y->left), height(y->right));
    x->height = 1 + max(height(x->left), height(x->right));
    return x;
}

Node* rotate_left(Node* x) {
    Node* y = x->right;
    Node* T2 = y->left;
    y->left = x;
    x->right = T2;
    x->height = 1 + max(height(x->left), height(x->right));
    y->height = 1 + max(height(y->left), height(y->right));
    return y;
}

int balance(Node* n) { return n ? height(n->left) - height(n->right) : 0; }

Node* min_node(Node* n) {
    Node* cur = n;
    while (cur->left) cur = cur->left;
    return cur;
}

Node* insert(Node* root, int key) {
    if (!root) return new_node(key);
    if (key < root->key) root->left = insert(root->left, key);
    else if (key > root->key) root->right = insert(root->right, key);
    else return root;

    root->height = 1 + max(height(root->left), height(root->right));
    int bf = balance(root);

    // Rebalance
    if (bf > 1 && key < root->left->key) return rotate_right(root);
    if (bf < -1 && key > root->right->key) return rotate_left(root);
    if (bf > 1 && key > root->left->key) {
        root->left = rotate_left(root->left);
        return rotate_right(root);
    }
    if (bf < -1 && key < root->right->key) {
        root->right = rotate_right(root->right);
        return rotate_left(root);
    }
    return root;
}

Node* delete(Node* root, int key) {
    if (!root) return root;

    if (key < root->key) root->left = delete(root->left, key);
    else if (key > root->key) root->right = delete(root->right, key);
    else {
        // Node with one or no child
        if (!root->left || !root->right) {
            Node* tmp = root->left ? root->left : root->right;
            if (!tmp) { tmp = root; root = NULL; }
            else *root = *tmp;
            free(tmp);
        } else {
            Node* tmp = min_node(root->right);
            root->key = tmp->key;
            root->right = delete(root->right, tmp->key);
        }
    }
    if (!root) return root;

    root->height = 1 + max(height(root->left), height(root->right));
    int bf = balance(root);

    // Rebalance
    if (bf > 1 && balance(root->left) >= 0)
        return rotate_right(root);
    if (bf > 1 && balance(root->left) < 0) {
        root->left = rotate_left(root->left);
        return rotate_right(root);
    }
    if (bf < -1 && balance(root->right) <= 0)
        return rotate_left(root);
    if (bf < -1 && balance(root->right) > 0) {
        root->right = rotate_right(root->right);
        return rotate_left(root);
    }
    return root;
}

void inorder(Node* r) {
    if (!r) return;
    inorder(r->left);
    printf("%d ", r->key);
    inorder(r->right);
}

int main(void) {
    Node* root = NULL;
    int keys[] = {10, 20, 30, 40, 50, 25};
    for (int i = 0; i < 6; i++) root = insert(root, keys[i]);
    root = delete(root, 40);
    inorder(root);
    printf("\n");
}
```

Python

```python
class Node:
    def __init__(self, key):
        self.key = key
        self.height = 1
        self.left = self.right = None

def height(n): return n.height if n else 0
def balance(n): return height(n.left) - height(n.right) if n else 0

def rotate_right(y):
    x, T2 = y.left, y.left.right
    x.right, y.left = y, T2
    y.height = 1 + max(height(y.left), height(y.right))
    x.height = 1 + max(height(x.left), height(x.right))
    return x

def rotate_left(x):
    y, T2 = x.right, x.right.left
    y.left, x.right = x, T2
    x.height = 1 + max(height(x.left), height(x.right))
    y.height = 1 + max(height(y.left), height(y.right))
    return y

def min_node(n):
    while n.left: n = n.left
    return n

def insert(r, k):
    if not r: return Node(k)
    if k < r.key: r.left = insert(r.left, k)
    elif k > r.key: r.right = insert(r.right, k)
    else: return r
    r.height = 1 + max(height(r.left), height(r.right))
    bf = balance(r)
    if bf > 1 and k < r.left.key: return rotate_right(r)
    if bf < -1 and k > r.right.key: return rotate_left(r)
    if bf > 1 and k > r.left.key:
        r.left = rotate_left(r.left); return rotate_right(r)
    if bf < -1 and k < r.right.key:
        r.right = rotate_right(r.right); return rotate_left(r)
    return r

def delete(r, k):
    if not r: return r
    if k < r.key: r.left = delete(r.left, k)
    elif k > r.key: r.right = delete(r.right, k)
    else:
        if not r.left: return r.right
        elif not r.right: return r.left
        temp = min_node(r.right)
        r.key = temp.key
        r.right = delete(r.right, temp.key)
    r.height = 1 + max(height(r.left), height(r.right))
    bf = balance(r)
    if bf > 1 and balance(r.left) >= 0: return rotate_right(r)
    if bf > 1 and balance(r.left) < 0:
        r.left = rotate_left(r.left); return rotate_right(r)
    if bf < -1 and balance(r.right) <= 0: return rotate_left(r)
    if bf < -1 and balance(r.right) > 0:
        r.right = rotate_right(r.right); return rotate_left(r)
    return r

def inorder(r):
    if not r: return
    inorder(r.left); print(r.key, end=' '); inorder(r.right)

root = None
for k in [10,20,30,40,50,25]:
    root = insert(root, k)
root = delete(root, 40)
inorder(root)
```

#### Why It Matters

- Keeps search, insert, delete all O(log n)
- Auto-rebalances after removal
- Shows how rotations maintain structure consistency
- Foundation for all balanced trees (like Red-Black, AVL variants)

#### A Gentle Proof (Why It Works)

Every deletion changes subtree height by at most 1.
Each ancestor's balance factor is recomputed; if imbalance found, a single rotation (or double rotation) restores balance.
At most O(log n) nodes are visited, and each fix is O(1).

#### Try It Yourself

1. Build [10, 20, 30, 40, 50, 25], then delete 50
2. Observe RR rotation at 30
3. Delete 10, check rebalancing at 20
4. Delete all sequentially, confirm sorted order

#### Test Cases

| Insert Sequence     | Delete | Rotation | Root After | Balanced |
| ------------------- | ------ | -------- | ---------- | -------- |
| [10,20,30,40,50,25] | 40     | RL       | 30         | ✅        |
| [10,20,30]          | 10     | RR       | 20         | ✅        |
| [30,20,10]          | 30     | LL       | 20         | ✅        |

Edge Cases

- Deleting from empty tree → safe
- Single node → becomes NULL
- Duplicate keys → ignored

#### Complexity

| Operation | Time     | Space          |
| --------- | -------- | -------------- |
| Delete    | O(log n) | O(h) recursion |
| Search    | O(log n) | O(h)           |
| Insert    | O(log n) | O(h)           |

An AVL deletion is a gentle art, remove the key, rebalance the branches, and harmony is restored.

### 233 Red-Black Tree Insert

A Red-Black Tree (RBT) is a self-balancing binary search tree that uses color bits (red or black) to control balance indirectly. Unlike AVL trees that balance by height, RBTs balance by color rules, allowing more flexible, faster insertions with fewer rotations.

#### What Problem Are We Solving?

A plain BST can degrade into a linked list with O(n) operations.
An RBT maintains a near-balanced height, ensuring O(log n) for search, insert, and delete.
Instead of exact height balance like AVL, RBTs enforce color invariants that keep paths roughly equal.

#### Red-Black Tree Properties

1. Each node is red or black.
2. The root is always black.
3. Null (NIL) nodes are considered black.
4. No two consecutive red nodes (no red parent with red child).
5. Every path from a node to its descendant NIL nodes has the same number of black nodes.

These rules guarantee height ≤ 2 × log₂(n + 1).

#### How Does It Work (Plain Language)?

1. Insert node like in a normal BST (color it red).
2. Fix violations if any property breaks.
3. Use rotations and recoloring based on where the red node appears.

| Case   | Condition                                 | Fix                                           |
| ------ | ----------------------------------------- | --------------------------------------------- |
| Case 1 | New node is root                          | Recolor black                                 |
| Case 2 | Parent black                              | No fix needed                                 |
| Case 3 | Parent red, Uncle red                     | Recolor parent & uncle black, grandparent red |
| Case 4 | Parent red, Uncle black, Triangle (LR/RL) | Rotate to line up                             |
| Case 5 | Parent red, Uncle black, Line (LL/RR)     | Rotate and recolor grandparent                |

#### Example

Insert sequence: 10, 20, 30

1. Insert 10 → root → black
2. Insert 20 → red child → balanced
3. Insert 30 → red parent (20) → Case 5 (RR) → rotate left on 10
   → recolor root black, children red

Result:

```
    20(B)
   /    \
10(R)   30(R)
```

#### Step-by-Step Example

Insert [7, 3, 18, 10, 22, 8, 11, 26]

| Step | Insert | Violation                   | Fix              | Root |
| ---- | ------ | --------------------------- | ---------------- | ---- |
| 1    | 7      | root                        | make black       | 7(B) |
| 2    | 3      | parent black                | none             | 7(B) |
| 3    | 18     | parent black                | none             | 7(B) |
| 4    | 10     | parent red, uncle red       | recolor, move up | 7(B) |
| 5    | 22     | parent black                | none             | 7(B) |
| 6    | 8      | parent red, uncle red       | recolor          | 7(B) |
| 7    | 11     | parent red, uncle black, LR | rotate + recolor | 7(B) |
| 8    | 26     | parent black                | none             | 7(B) |

Final tree balanced by color invariants.

#### Tiny Code (Easy Versions)

C (Simplified)

```c
#include <stdio.h>
#include <stdlib.h>

typedef enum { RED, BLACK } Color;

typedef struct Node {
    int key;
    Color color;
    struct Node *left, *right, *parent;
} Node;

Node* new_node(int key) {
    Node* n = malloc(sizeof(Node));
    n->key = key;
    n->color = RED;
    n->left = n->right = n->parent = NULL;
    return n;
}

void rotate_left(Node root, Node* x) {
    Node* y = x->right;
    x->right = y->left;
    if (y->left) y->left->parent = x;
    y->parent = x->parent;
    if (!x->parent) *root = y;
    else if (x == x->parent->left) x->parent->left = y;
    else x->parent->right = y;
    y->left = x;
    x->parent = y;
}

void rotate_right(Node root, Node* y) {
    Node* x = y->left;
    y->left = x->right;
    if (x->right) x->right->parent = y;
    x->parent = y->parent;
    if (!y->parent) *root = x;
    else if (y == y->parent->left) y->parent->left = x;
    else y->parent->right = x;
    x->right = y;
    y->parent = x;
}

void fix_violation(Node root, Node* z) {
    while (z->parent && z->parent->color == RED) {
        Node* gp = z->parent->parent;
        if (z->parent == gp->left) {
            Node* uncle = gp->right;
            if (uncle && uncle->color == RED) {
                z->parent->color = BLACK;
                uncle->color = BLACK;
                gp->color = RED;
                z = gp;
            } else {
                if (z == z->parent->right) {
                    z = z->parent;
                    rotate_left(root, z);
                }
                z->parent->color = BLACK;
                gp->color = RED;
                rotate_right(root, gp);
            }
        } else {
            Node* uncle = gp->left;
            if (uncle && uncle->color == RED) {
                z->parent->color = BLACK;
                uncle->color = BLACK;
                gp->color = RED;
                z = gp;
            } else {
                if (z == z->parent->left) {
                    z = z->parent;
                    rotate_right(root, z);
                }
                z->parent->color = BLACK;
                gp->color = RED;
                rotate_left(root, gp);
            }
        }
    }
    (*root)->color = BLACK;
}

Node* bst_insert(Node* root, Node* z) {
    if (!root) return z;
    if (z->key < root->key) {
        root->left = bst_insert(root->left, z);
        root->left->parent = root;
    } else if (z->key > root->key) {
        root->right = bst_insert(root->right, z);
        root->right->parent = root;
    }
    return root;
}

void insert(Node root, int key) {
    Node* z = new_node(key);
    *root = bst_insert(*root, z);
    fix_violation(root, z);
}

void inorder(Node* r) {
    if (!r) return;
    inorder(r->left);
    printf("%d(%c) ", r->key, r->color == RED ? 'R' : 'B');
    inorder(r->right);
}

int main(void) {
    Node* root = NULL;
    int keys[] = {10, 20, 30};
    for (int i = 0; i < 3; i++) insert(&root, keys[i]);
    inorder(root);
    printf("\n");
}
```

Python (Simplified)

```python
class Node:
    def __init__(self, key, color="R", parent=None):
        self.key = key
        self.color = color
        self.left = self.right = None
        self.parent = parent

def rotate_left(root, x):
    y = x.right
    x.right = y.left
    if y.left: y.left.parent = x
    y.parent = x.parent
    if not x.parent: root = y
    elif x == x.parent.left: x.parent.left = y
    else: x.parent.right = y
    y.left = x
    x.parent = y
    return root

def rotate_right(root, y):
    x = y.left
    y.left = x.right
    if x.right: x.right.parent = y
    x.parent = y.parent
    if not y.parent: root = x
    elif y == y.parent.left: y.parent.left = x
    else: y.parent.right = x
    x.right = y
    y.parent = x
    return root

def fix_violation(root, z):
    while z.parent and z.parent.color == "R":
        gp = z.parent.parent
        if z.parent == gp.left:
            uncle = gp.right
            if uncle and uncle.color == "R":
                z.parent.color = uncle.color = "B"
                gp.color = "R"
                z = gp
            else:
                if z == z.parent.right:
                    z = z.parent
                    root = rotate_left(root, z)
                z.parent.color = "B"
                gp.color = "R"
                root = rotate_right(root, gp)
        else:
            uncle = gp.left
            if uncle and uncle.color == "R":
                z.parent.color = uncle.color = "B"
                gp.color = "R"
                z = gp
            else:
                if z == z.parent.left:
                    z = z.parent
                    root = rotate_right(root, z)
                z.parent.color = "B"
                gp.color = "R"
                root = rotate_left(root, gp)
    root.color = "B"
    return root

def bst_insert(root, z):
    if not root: return z
    if z.key < root.key:
        root.left = bst_insert(root.left, z)
        root.left.parent = root
    elif z.key > root.key:
        root.right = bst_insert(root.right, z)
        root.right.parent = root
    return root

def insert(root, key):
    z = Node(key)
    root = bst_insert(root, z)
    return fix_violation(root, z)

def inorder(r):
    if not r: return
    inorder(r.left)
    print(f"{r.key}({r.color})", end=" ")
    inorder(r.right)

root = None
for k in [10,20,30]:
    root = insert(root, k)
inorder(root)
```

#### Why It Matters

- Fewer rotations than AVL
- Guarantees O(log n) for all operations
- Used in real systems: Linux kernel, Java `TreeMap`, C++ `map`
- Easy insertion logic using coloring + rotation

#### A Gentle Proof (Why It Works)

The black-height invariant ensures every path length is between `h` and `2h`.
Balancing via recolor + single rotation ensures logarithmic height.
Each insert requires at most 2 rotations, O(log n) traversal.

#### Try It Yourself

1. Insert [10, 20, 30] → RR rotation
2. Insert [30, 15, 10] → LL rotation
3. Insert [10, 15, 5] → recolor, no rotation
4. Draw colors, confirm invariants

#### Test Cases

| Sequence               | Rotations            | Root  | Black Height |
| ---------------------- | -------------------- | ----- | ------------ |
| [10,20,30]             | Left                 | 20(B) | 2            |
| [7,3,18,10,22,8,11,26] | Mixed                | 7(B)  | 3            |
| [1,2,3,4,5]            | Multiple recolor+rot | 2(B)  | 3            |

#### Complexity

| Operation | Time     | Space          |
| --------- | -------- | -------------- |
| Insert    | O(log n) | O(1) rotations |
| Search    | O(log n) | O(h)           |
| Delete    | O(log n) | O(1) rotations |

The Red-Black Tree paints order into chaos, each red spark balanced by a calm black shadow.

### 234 Red-Black Tree Delete

Deletion in a Red-Black Tree (RBT) is a delicate operation: remove a node, then restore balance by adjusting colors and rotations to maintain all five RBT invariants. While insertion fixes "too much red," deletion often fixes "too much black."

#### What Problem Are We Solving?

After deleting a node from an RBT, the black-height property may break (some paths lose a black node).
Our goal: restore all red-black invariants while keeping time complexity O(log n).

#### Red-Black Properties (Reminder)

1. Root is black.
2. Every node is red or black.
3. All NIL leaves are black.
4. Red nodes cannot have red children.
5. Every path from a node to NIL descendants has the same number of black nodes.

#### How Does It Work (Plain Language)?

1. Perform standard BST deletion.
2. Track if the removed node was black (this might cause "double black" issue).
3. Fix violations moving upward until the tree is balanced again.

| Case | Condition                                   | Fix                             |
| ---- | ------------------------------------------- | ------------------------------- |
| 1    | Node is red                                 | Simply remove (no rebalance)    |
| 2    | Node is black, child red                    | Replace and recolor child black |
| 3    | Node is black, child black ("double black") | Use sibling cases below         |

Double Black Fix Cases

| Case | Description                                    | Action                                                         |
| ---- | ---------------------------------------------- | -------------------------------------------------------------- |
| 1    | Sibling red                                    | Rotate and recolor to make sibling black                       |
| 2    | Sibling black, both children black             | Recolor sibling red, move double black up                      |
| 3    | Sibling black, near child red, far child black | Rotate sibling toward node, swap colors                        |
| 4    | Sibling black, far child red                   | Rotate parent, recolor sibling and parent, set far child black |

#### Example

Delete `30` from:

```
     20(B)
    /    \
 10(R)   30(R)
```

- 30 is red → remove directly
- No property violated

Delete `10`:

```
     20(B)
    /
   NIL
```

10 was red → simple remove → tree valid

Delete `20`:
Tree becomes empty → fine

#### Step-by-Step Example

Insert [10, 20, 30, 15, 25]
Delete 20

| Step | Action       | Violation               | Fix                                      |        |
| ---- | ------------ | ----------------------- | ---------------------------------------- | ------ |
| 1    | Delete 20    | double black at 25      | sibling 10 black, far child red → Case 4 | rotate |
| 2    | After rotate | all properties restored | -                                        |        |

#### Tiny Code (Simplified)

C (Conceptual)

```c
// This snippet omits full BST insertion for brevity.

typedef enum { RED, BLACK } Color;
typedef struct Node {
    int key;
    Color color;
    struct Node *left, *right, *parent;
} Node;

// Utility: get sibling
Node* sibling(Node* n) {
    if (!n->parent) return NULL;
    return n == n->parent->left ? n->parent->right : n->parent->left;
}

void fix_delete(Node root, Node* x) {
    while (x != *root && (!x || x->color == BLACK)) {
        Node* s = sibling(x);
        if (x == x->parent->left) {
            if (s->color == RED) {
                s->color = BLACK;
                x->parent->color = RED;
                rotate_left(root, x->parent);
                s = x->parent->right;
            }
            if ((!s->left || s->left->color == BLACK) &&
                (!s->right || s->right->color == BLACK)) {
                s->color = RED;
                x = x->parent;
            } else {
                if (!s->right || s->right->color == BLACK) {
                    if (s->left) s->left->color = BLACK;
                    s->color = RED;
                    rotate_right(root, s);
                    s = x->parent->right;
                }
                s->color = x->parent->color;
                x->parent->color = BLACK;
                if (s->right) s->right->color = BLACK;
                rotate_left(root, x->parent);
                x = *root;
            }
        } else {
            // mirror logic for right child
            if (s->color == RED) {
                s->color = BLACK;
                x->parent->color = RED;
                rotate_right(root, x->parent);
                s = x->parent->left;
            }
            if ((!s->left || s->left->color == BLACK) &&
                (!s->right || s->right->color == BLACK)) {
                s->color = RED;
                x = x->parent;
            } else {
                if (!s->left || s->left->color == BLACK) {
                    if (s->right) s->right->color = BLACK;
                    s->color = RED;
                    rotate_left(root, s);
                    s = x->parent->left;
                }
                s->color = x->parent->color;
                x->parent->color = BLACK;
                if (s->left) s->left->color = BLACK;
                rotate_right(root, x->parent);
                x = *root;
            }
        }
    }
    if (x) x->color = BLACK;
}
```

Python (Simplified Pseudocode)

```python
def fix_delete(root, x):
    while x != root and (not x or x.color == "B"):
        if x == x.parent.left:
            s = x.parent.right
            if s.color == "R":
                s.color, x.parent.color = "B", "R"
                root = rotate_left(root, x.parent)
                s = x.parent.right
            if all(c is None or c.color == "B" for c in [s.left, s.right]):
                s.color = "R"
                x = x.parent
            else:
                if not s.right or s.right.color == "B":
                    if s.left: s.left.color = "B"
                    s.color = "R"
                    root = rotate_right(root, s)
                    s = x.parent.right
                s.color, x.parent.color = x.parent.color, "B"
                if s.right: s.right.color = "B"
                root = rotate_left(root, x.parent)
                x = root
        else:
            # mirror logic
            ...
    if x: x.color = "B"
    return root
```

#### Why It Matters

- Preserves logarithmic height after deletion
- Used in core data structures (`std::map`, `TreeSet`)
- Demonstrates color logic as a soft balancing scheme
- Handles edge cases gracefully via double black fix

#### A Gentle Proof (Why It Works)

When a black node is removed, one path loses a black count.
The fix cases re-distribute black heights using rotations and recolors.
Each loop iteration moves double-black upward, O(log n) steps.

#### Try It Yourself

1. Build [10, 20, 30, 15, 25, 5]. Delete 10 (leaf).
2. Delete 30 (red leaf) → no fix.
3. Delete 20 (black node with one child) → recolor fix.
4. Visualize each case: sibling red, sibling black with red child.

#### Test Cases

| Insert Sequence        | Delete | Case Triggered | Fix              |
| ---------------------- | ------ | -------------- | ---------------- |
| [10,20,30]             | 20     | Case 4         | Rotate & recolor |
| [7,3,18,10,22,8,11,26] | 18     | Case 2         | Recolor sibling  |
| [10,5,1]               | 5      | Case 1         | Rotate parent    |

#### Complexity

| Operation | Time     | Space          |
| --------- | -------- | -------------- |
| Delete    | O(log n) | O(1) rotations |
| Insert    | O(log n) | O(1)           |
| Search    | O(log n) | O(h)           |

A Red-Black delete is like a careful tune-up, one color at a time, harmony restored across every path.

### 235 Splay Tree Access

A Splay Tree is a self-adjusting binary search tree that brings frequently accessed elements closer to the root through splaying, a series of tree rotations. Unlike AVL or Red-Black trees, it doesn't maintain strict balance but guarantees amortized O(log n) performance for access, insert, and delete.

#### What Problem Are We Solving?

In many workloads, some elements are accessed far more frequently than others. Ordinary BSTs give no advantage for "hot" keys, while balanced trees maintain shape but not recency.
A Splay Tree optimizes for temporal locality: recently accessed items move near the root, making repeated access faster.

#### How Does It Work (Plain Language)?

Whenever you access a node (via search, insert, or delete), perform a splay operation, repeatedly rotate the node toward the root according to its position and parent relationships.

The three rotation patterns:

| Case        | Structure                                            | Operation                                         |
| ----------- | ---------------------------------------------------- | ------------------------------------------------- |
| Zig     | Node is child of root                                | Single rotation                                   |
| Zig-Zig | Node and parent are both left or both right children | Double rotation (rotate parent, then grandparent) |
| Zig-Zag | Node and parent are opposite children                | Double rotation (rotate node twice upward)        |

After splaying, the accessed node becomes the root.

#### Example

Access sequence: 10, 20, 30

1. Insert 10 (root)
2. Insert 20 → 20 right of 10
3. Access 20 → Zig rotation → 20 becomes root
4. Insert 30 → 30 right of 20
5. Access 30 → Zig rotation → 30 root

Resulting tree (after all accesses):

```
   30
  /
20
/
10
```

Frequently used nodes rise to the top automatically.

#### Step-by-Step Example

Start with keys [5, 3, 8, 1, 4, 7, 9].
Access key 1.

| Step | Operation   | Case                | Rotation         | New Root |
| ---- | ----------- | ------------------- | ---------------- | -------- |
| 1    | Access 1    | Zig-Zig (left-left) | Rotate 3, then 5 | 1        |
| 2    | After splay | -                   | -                | 1        |

Final tree: 1 becomes root, path shortened for next access.

#### Tiny Code (Easy Versions)

C

```c
#include <stdio.h>
#include <stdlib.h>

typedef struct Node {
    int key;
    struct Node *left, *right;
} Node;

Node* new_node(int key) {
    Node* n = malloc(sizeof(Node));
    n->key = key;
    n->left = n->right = NULL;
    return n;
}

Node* rotate_right(Node* x) {
    Node* y = x->left;
    x->left = y->right;
    y->right = x;
    return y;
}

Node* rotate_left(Node* x) {
    Node* y = x->right;
    x->right = y->left;
    y->left = x;
    return y;
}

Node* splay(Node* root, int key) {
    if (!root || root->key == key) return root;

    // Key in left subtree
    if (key < root->key) {
        if (!root->left) return root;
        if (key < root->left->key) {
            root->left->left = splay(root->left->left, key);
            root = rotate_right(root);
        } else if (key > root->left->key) {
            root->left->right = splay(root->left->right, key);
            if (root->left->right)
                root->left = rotate_left(root->left);
        }
        return root->left ? rotate_right(root) : root;
    }
    // Key in right subtree
    else {
        if (!root->right) return root;
        if (key > root->right->key) {
            root->right->right = splay(root->right->right, key);
            root = rotate_left(root);
        } else if (key < root->right->key) {
            root->right->left = splay(root->right->left, key);
            if (root->right->left)
                root->right = rotate_right(root->right);
        }
        return root->right ? rotate_left(root) : root;
    }
}

Node* insert(Node* root, int key) {
    if (!root) return new_node(key);
    root = splay(root, key);
    if (root->key == key) return root;
    Node* n = new_node(key);
    if (key < root->key) {
        n->right = root;
        n->left = root->left;
        root->left = NULL;
    } else {
        n->left = root;
        n->right = root->right;
        root->right = NULL;
    }
    return n;
}

void inorder(Node* r) {
    if (!r) return;
    inorder(r->left);
    printf("%d ", r->key);
    inorder(r->right);
}

int main(void) {
    Node* root = NULL;
    int keys[] = {5, 3, 8, 1, 4, 7, 9};
    for (int i = 0; i < 7; i++) root = insert(root, keys[i]);
    root = splay(root, 1);
    inorder(root);
    printf("\n");
}
```

Python

```python
class Node:
    def __init__(self, key):
        self.key = key
        self.left = self.right = None

def rotate_right(x):
    y = x.left
    x.left = y.right
    y.right = x
    return y

def rotate_left(x):
    y = x.right
    x.right = y.left
    y.left = x
    return y

def splay(root, key):
    if not root or root.key == key:
        return root
    if key < root.key:
        if not root.left:
            return root
        if key < root.left.key:
            root.left.left = splay(root.left.left, key)
            root = rotate_right(root)
        elif key > root.left.key:
            root.left.right = splay(root.left.right, key)
            if root.left.right:
                root.left = rotate_left(root.left)
        return rotate_right(root) if root.left else root
    else:
        if not root.right:
            return root
        if key > root.right.key:
            root.right.right = splay(root.right.right, key)
            root = rotate_left(root)
        elif key < root.right.key:
            root.right.left = splay(root.right.left, key)
            if root.right.left:
                root.right = rotate_right(root.right)
        return rotate_left(root) if root.right else root

def insert(root, key):
    if not root:
        return Node(key)
    root = splay(root, key)
    if root.key == key:
        return root
    n = Node(key)
    if key < root.key:
        n.right = root
        n.left = root.left
        root.left = None
    else:
        n.left = root
        n.right = root.right
        root.right = None
    return n

def inorder(r):
    if not r: return
    inorder(r.left)
    print(r.key, end=" ")
    inorder(r.right)

root = None
for k in [5,3,8,1,4,7,9]:
    root = insert(root, k)
root = splay(root, 1)
inorder(root)
```

#### Why It Matters

- Amortized O(log n) performance
- Adapts dynamically to access patterns
- Ideal for caches, text editors, network routing tables
- Simple logic: no height or color tracking needed

#### A Gentle Proof (Why It Works)

Each splay operation may take O(h), but the amortized cost across multiple operations is O(log n).
Accessing frequently used elements keeps them near the root, improving future operations.

#### Try It Yourself

1. Build [5, 3, 8, 1, 4, 7, 9]
2. Access 1 → Observe Zig-Zig rotation
3. Access 8 → Observe Zig-Zag
4. Insert 10 → check rebalancing
5. Access 7 repeatedly → moves to root

#### Test Cases

| Sequence    | Access | Case    | Root After |
| ----------- | ------ | ------- | ---------- |
| [5,3,8,1,4] | 1      | Zig-Zig | 1          |
| [10,20,30]  | 30     | Zig     | 30         |
| [5,3,8,1,4] | 4      | Zig-Zag | 4          |

#### Complexity

| Operation | Time               | Space          |
| --------- | ------------------ | -------------- |
| Access    | Amortized O(log n) | O(h) recursion |
| Insert    | Amortized O(log n) | O(h)           |
| Delete    | Amortized O(log n) | O(h)           |

The Splay Tree is like a memory, the more you touch something, the closer it stays to you.

### 236 Treap Insert

A Treap (Tree + Heap) is a brilliant hybrid: it's a binary search tree by keys and a heap by priorities. Every node carries a key and a random priority. By combining ordering on keys with random heap priorities, the treap stays *balanced on average*, no strict rotations like AVL or color rules like Red-Black trees needed.

#### What Problem Are We Solving?

We want a simple balanced search tree with expected O(log n) performance,
but without maintaining height or color properties explicitly.
Treaps solve this by assigning random priorities, keeping the structure balanced *in expectation*.

#### How It Works (Plain Language)

Each node `(key, priority)` must satisfy:

- BST property: `key(left) < key < key(right)`
- Heap property: `priority(parent) < priority(children)` (min-heap or max-heap convention)

Insert like a normal BST by key, then fix heap property by rotations if the priority is violated.

| Step | Rule                                                   |
| ---- | ------------------------------------------------------ |
| 1    | Insert node by key (like BST)                          |
| 2    | Assign a random priority                               |
| 3    | While parent priority > node priority → rotate node up |

This randomization ensures expected logarithmic height.

#### Example

Insert sequence (key, priority):

| Step | Key      | Priority                        | Structure                  |
| ---- | -------- | ------------------------------- | -------------------------- |
| 1    | (50, 15) | root                            | 50                         |
| 2    | (30, 10) | smaller priority → rotate right | 30(root) → 50(right child) |
| 3    | (70, 25) | stays right of 50               | balanced                   |

Result (BST by key, min-heap by priority):

```
     (30,10)
         \
         (50,15)
             \
             (70,25)
```

#### Step-by-Step Example

Insert keys: [40, 20, 60, 10, 30, 50, 70]
Priorities: [80, 90, 70, 100, 85, 60, 75]

| Key | Priority | Rotation               | Result     |
| --- | -------- | ---------------------- | ---------- |
| 40  | 80       | root                   | 40         |
| 20  | 90       | none                   | left child |
| 60  | 70       | rotate left (heap fix) | 60 root    |
| 10  | 100      | none                   | leaf       |
| 30  | 85       | none                   | under 20   |
| 50  | 60       | rotate left            | 50 root    |
| 70  | 75       | none                   | leaf       |

#### Tiny Code (Easy Versions)

C

```c
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

typedef struct Node {
    int key, priority;
    struct Node *left, *right;
} Node;

Node* new_node(int key) {
    Node* n = malloc(sizeof(Node));
    n->key = key;
    n->priority = rand() % 100; // random priority
    n->left = n->right = NULL;
    return n;
}

Node* rotate_right(Node* y) {
    Node* x = y->left;
    y->left = x->right;
    x->right = y;
    return x;
}

Node* rotate_left(Node* x) {
    Node* y = x->right;
    x->right = y->left;
    y->left = x;
    return y;
}

Node* insert(Node* root, int key) {
    if (!root) return new_node(key);

    if (key < root->key) {
        root->left = insert(root->left, key);
        if (root->left->priority < root->priority)
            root = rotate_right(root);
    } else if (key > root->key) {
        root->right = insert(root->right, key);
        if (root->right->priority < root->priority)
            root = rotate_left(root);
    }
    return root;
}

void inorder(Node* root) {
    if (!root) return;
    inorder(root->left);
    printf("(%d,%d) ", root->key, root->priority);
    inorder(root->right);
}

int main(void) {
    srand(time(NULL));
    Node* root = NULL;
    int keys[] = {40, 20, 60, 10, 30, 50, 70};
    for (int i = 0; i < 7; i++)
        root = insert(root, keys[i]);
    inorder(root);
    printf("\n");
}
```

Python

```python
import random

class Node:
    def __init__(self, key):
        self.key = key
        self.priority = random.randint(1, 100)
        self.left = self.right = None

def rotate_right(y):
    x = y.left
    y.left = x.right
    x.right = y
    return x

def rotate_left(x):
    y = x.right
    x.right = y.left
    y.left = x
    return y

def insert(root, key):
    if not root:
        return Node(key)
    if key < root.key:
        root.left = insert(root.left, key)
        if root.left.priority < root.priority:
            root = rotate_right(root)
    elif key > root.key:
        root.right = insert(root.right, key)
        if root.right.priority < root.priority:
            root = rotate_left(root)
    return root

def inorder(root):
    if not root: return
    inorder(root.left)
    print(f"({root.key},{root.priority})", end=" ")
    inorder(root.right)

root = None
for k in [40, 20, 60, 10, 30, 50, 70]:
    root = insert(root, k)
inorder(root)
```

#### Why It Matters

- Simple implementation with expected balance
- Probabilistic alternative to AVL / Red-Black
- Great for randomized data structures and Cartesian tree applications
- No explicit height or color tracking

#### A Gentle Proof (Why It Works)

Random priorities imply random structure;
the probability of a node being high in the tree drops exponentially with depth.
Hence, expected height is O(log n) with high probability.

#### Try It Yourself

1. Insert keys [10, 20, 30, 40] with random priorities.
2. Observe random structure (not sorted by insertion).
3. Change priority generator → test skewness.
4. Visualize both BST and heap invariants.

#### Test Cases

| Keys        | Priorities | Resulting Root | Height   |
| ----------- | ---------- | -------------- | -------- |
| [10,20,30]  | [5,3,4]    | 20             | 2        |
| [5,2,8,1,3] | random     | varies         | ~log n   |
| [1..100]    | random     | balanced       | O(log n) |

#### Complexity

| Operation | Time              | Space          |
| --------- | ----------------- | -------------- |
| Insert    | O(log n) expected | O(h) recursion |
| Search    | O(log n) expected | O(h)           |
| Delete    | O(log n) expected | O(h)           |

A Treap dances to the rhythm of chance, balancing order with randomness to stay nimble and fair.

### 237 Treap Delete

Deleting from a Treap blends the best of both worlds: BST search for locating the node and heap-based rotations to remove it while keeping balance. Instead of rebalancing explicitly, we rotate the target node downward until it becomes a leaf, then remove it.

#### What Problem Are We Solving?

We want to delete a key while preserving both:

- BST property: keys in order
- Heap property: priorities follow min-/max-heap rule

Treaps handle this by rotating down the target node until it has at most one child, then removing it directly.

#### How It Works (Plain Language)

1. Search for the node by key (BST search).
2. Once found, rotate it downward until it has ≤ 1 child:

   * Rotate left if right child's priority < left child's priority
   * Rotate right otherwise
3. Remove it once it's a leaf or single-child node.

Because priorities are random, the tree remains balanced *in expectation*.

#### Example

Treap (key, priority):

```
      (40,20)
     /      \
 (30,10)   (50,25)
```

Delete 40:

- Compare children: (30,10) < (50,25) → rotate right on 40
- New root (30,10)
- 40 now right child → continue rotation if needed → eventually becomes leaf → delete

New structure maintains BST + heap properties.

#### Step-by-Step Example

Start with nodes:

| Key | Priority |
| --- | -------- |
| 40  | 50       |
| 20  | 60       |
| 60  | 70       |
| 10  | 90       |
| 30  | 80       |
| 50  | 65       |
| 70  | 75       |

Delete 40:

| Step | Node | Action                      | Rotation          | Root               |
| ---- | ---- | --------------------------- | ----------------- | ------------------ |
| 1    | 40   | Compare children priorities | Left child higher | Rotate right       |
| 2    | 30   | Now parent of 40            | 40 demoted        | Continue if needed |
| 3    | 40   | Becomes leaf                | Remove            | 30                 |

Final root: (30,80)
All properties hold.

#### Tiny Code (Easy Versions)

C

```c
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

typedef struct Node {
    int key, priority;
    struct Node *left, *right;
} Node;

Node* new_node(int key) {
    Node* n = malloc(sizeof(Node));
    n->key = key;
    n->priority = rand() % 100;
    n->left = n->right = NULL;
    return n;
}

Node* rotate_right(Node* y) {
    Node* x = y->left;
    y->left = x->right;
    x->right = y;
    return x;
}

Node* rotate_left(Node* x) {
    Node* y = x->right;
    x->right = y->left;
    y->left = x;
    return y;
}

Node* insert(Node* root, int key) {
    if (!root) return new_node(key);
    if (key < root->key) {
        root->left = insert(root->left, key);
        if (root->left->priority < root->priority)
            root = rotate_right(root);
    } else if (key > root->key) {
        root->right = insert(root->right, key);
        if (root->right->priority < root->priority)
            root = rotate_left(root);
    }
    return root;
}

Node* delete(Node* root, int key) {
    if (!root) return NULL;

    if (key < root->key)
        root->left = delete(root->left, key);
    else if (key > root->key)
        root->right = delete(root->right, key);
    else {
        // Found node to delete
        if (!root->left && !root->right) {
            free(root);
            return NULL;
        } else if (!root->left)
            root = rotate_left(root);
        else if (!root->right)
            root = rotate_right(root);
        else if (root->left->priority < root->right->priority)
            root = rotate_right(root);
        else
            root = rotate_left(root);

        root = delete(root, key);
    }
    return root;
}

void inorder(Node* root) {
    if (!root) return;
    inorder(root->left);
    printf("(%d,%d) ", root->key, root->priority);
    inorder(root->right);
}

int main(void) {
    srand(time(NULL));
    Node* root = NULL;
    int keys[] = {40, 20, 60, 10, 30, 50, 70};
    for (int i = 0; i < 7; i++)
        root = insert(root, keys[i]);

    printf("Before deletion:\n");
    inorder(root);
    printf("\n");

    root = delete(root, 40);

    printf("After deleting 40:\n");
    inorder(root);
    printf("\n");
}
```

Python

```python
import random

class Node:
    def __init__(self, key):
        self.key = key
        self.priority = random.randint(1, 100)
        self.left = self.right = None

def rotate_right(y):
    x = y.left
    y.left = x.right
    x.right = y
    return x

def rotate_left(x):
    y = x.right
    x.right = y.left
    y.left = x
    return y

def insert(root, key):
    if not root:
        return Node(key)
    if key < root.key:
        root.left = insert(root.left, key)
        if root.left.priority < root.priority:
            root = rotate_right(root)
    elif key > root.key:
        root.right = insert(root.right, key)
        if root.right.priority < root.priority:
            root = rotate_left(root)
    return root

def delete(root, key):
    if not root:
        return None
    if key < root.key:
        root.left = delete(root.left, key)
    elif key > root.key:
        root.right = delete(root.right, key)
    else:
        if not root.left and not root.right:
            return None
        elif not root.left:
            root = rotate_left(root)
        elif not root.right:
            root = rotate_right(root)
        elif root.left.priority < root.right.priority:
            root = rotate_right(root)
        else:
            root = rotate_left(root)
        root = delete(root, key)
    return root

def inorder(root):
    if not root: return
    inorder(root.left)
    print(f"({root.key},{root.priority})", end=" ")
    inorder(root.right)

root = None
for k in [40,20,60,10,30,50,70]:
    root = insert(root, k)
print("Before:", end=" ")
inorder(root)
print()
root = delete(root, 40)
print("After:", end=" ")
inorder(root)
```

#### Why It Matters

- No explicit rebalancing, rotations emerge from heap priorities
- Expected O(log n) deletion
- Natural, simple structure for randomized search trees
- Used in randomized algorithms, dynamic sets, and order-statistics

#### A Gentle Proof (Why It Works)

At each step, the node rotates toward a child with smaller priority, preserving heap property.
Expected height remains O(log n) due to independent random priorities.

#### Try It Yourself

1. Build treap with [10, 20, 30, 40, 50].
2. Delete 30 → observe rotations to leaf.
3. Delete 10 (root) → rotation chooses smaller-priority child.
4. Repeat deletions, verify BST + heap invariants hold.

#### Test Cases

| Keys                   | Delete | Rotations   | Balanced | Root After |
| ---------------------- | ------ | ----------- | -------- | ---------- |
| [40,20,60,10,30,50,70] | 40     | Right, Left | ✅        | 30         |
| [10,20,30]             | 20     | Left        | ✅        | 10         |
| [50,30,70,20,40,60,80] | 30     | Right       | ✅        | 50         |

#### Complexity

| Operation | Time              | Space          |
| --------- | ----------------- | -------------- |
| Delete    | O(log n) expected | O(h) recursion |
| Insert    | O(log n) expected | O(h)           |
| Search    | O(log n) expected | O(h)           |

The Treap deletes gracefully, rotating away until the target fades like a falling leaf, leaving balance behind.

### 238 Weight Balanced Tree

A Weight Balanced Tree (WBT) maintains balance not by height or color, but by subtree sizes (or "weights"). Each node keeps track of how many elements are in its left and right subtrees, and balance is enforced by requiring their weights to stay within a constant ratio.

#### What Problem Are We Solving?

In ordinary BSTs, imbalance can grow as keys are inserted in sorted order, degrading performance to O(n).
Weight Balanced Trees fix this by keeping subtree sizes within a safe range. They are particularly useful when you need order-statistics (like "find the k-th element") or split/merge operations that depend on sizes rather than heights.

#### How It Works (Plain Language)

Each node maintains a weight = total number of nodes in its subtree.
When inserting or deleting, you update weights and check the balance condition:

If

```
weight(left) ≤ α × weight(node)
weight(right) ≤ α × weight(node)
```

for some balance constant α (e.g., 0.7),
then the tree is considered balanced.

If not, perform rotations (like AVL) to restore balance.

| Operation  | Steps                                                                |
| ---------- | -------------------------------------------------------------------- |
| Insert | Insert like BST → Update weights → If ratio violated → Rotate        |
| Delete | Remove node → Update weights → If ratio violated → Rebuild or rotate |
| Search | BST search (weights guide decisions for rank queries)                |

Because weights reflect actual sizes, the tree ensures O(log n) expected height.

#### Example

Let's use α = 0.7

Insert [10, 20, 30, 40, 50]

| Step | Insert | Weight(left) | Weight(right) | Balanced? | Fix         |
| ---- | ------ | ------------ | ------------- | --------- | ----------- |
| 1    | 10     | 0            | 0             | ✅         | -           |
| 2    | 20     | 1            | 0             | ✅         | -           |
| 3    | 30     | 2            | 0             | ❌         | Rotate left |
| 4    | 40     | 3            | 0             | ❌         | Rotate left |
| 5    | 50     | 4            | 0             | ❌         | Rotate left |

Tree remains balanced by rotations once left-right ratio exceeds α.

#### Step-by-Step Example

Insert [1, 2, 3, 4, 5] with α = 0.7

| Step | Action   | Weights | Balanced? | Rotation    |
| ---- | -------- | ------- | --------- | ----------- |
| 1    | Insert 1 | (0,0)   | ✅         | -           |
| 2    | Insert 2 | (1,0)   | ✅         | -           |
| 3    | Insert 3 | (2,0)   | ❌         | Left Rotate |
| 4    | Insert 4 | (3,0)   | ❌         | Left Rotate |
| 5    | Insert 5 | (4,0)   | ❌         | Left Rotate |

Tree remains height-balanced based on weights.

#### Tiny Code (Easy Versions)

C (Simplified)

```c
#include <stdio.h>
#include <stdlib.h>

typedef struct Node {
    int key;
    int size; // weight = size of subtree
    struct Node *left, *right;
} Node;

int size(Node* n) { return n ? n->size : 0; }

Node* new_node(int key) {
    Node* n = malloc(sizeof(Node));
    n->key = key;
    n->size = 1;
    n->left = n->right = NULL;
    return n;
}

Node* rotate_left(Node* x) {
    Node* y = x->right;
    x->right = y->left;
    y->left = x;
    x->size = 1 + size(x->left) + size(x->right);
    y->size = 1 + size(y->left) + size(y->right);
    return y;
}

Node* rotate_right(Node* y) {
    Node* x = y->left;
    y->left = x->right;
    x->right = y;
    y->size = 1 + size(y->left) + size(y->right);
    x->size = 1 + size(x->left) + size(x->right);
    return x;
}

double alpha = 0.7;

int balanced(Node* n) {
    if (!n) return 1;
    int l = size(n->left), r = size(n->right);
    return l <= alpha * n->size && r <= alpha * n->size;
}

Node* insert(Node* root, int key) {
    if (!root) return new_node(key);
    if (key < root->key) root->left = insert(root->left, key);
    else if (key > root->key) root->right = insert(root->right, key);
    root->size = 1 + size(root->left) + size(root->right);
    if (!balanced(root)) {
        if (size(root->left) > size(root->right))
            root = rotate_right(root);
        else
            root = rotate_left(root);
    }
    return root;
}

void inorder(Node* n) {
    if (!n) return;
    inorder(n->left);
    printf("%d(%d) ", n->key, n->size);
    inorder(n->right);
}

int main(void) {
    Node* root = NULL;
    int keys[] = {10, 20, 30, 40, 50};
    for (int i = 0; i < 5; i++)
        root = insert(root, keys[i]);
    inorder(root);
    printf("\n");
}
```

Python (Simplified)

```python
class Node:
    def __init__(self, key):
        self.key = key
        self.size = 1
        self.left = self.right = None

def size(n): return n.size if n else 0

def update(n):
    if n:
        n.size = 1 + size(n.left) + size(n.right)

def rotate_left(x):
    y = x.right
    x.right = y.left
    y.left = x
    update(x)
    update(y)
    return y

def rotate_right(y):
    x = y.left
    y.left = x.right
    x.right = y
    update(y)
    update(x)
    return x

alpha = 0.7

def balanced(n):
    return not n or (size(n.left) <= alpha * n.size and size(n.right) <= alpha * n.size)

def insert(root, key):
    if not root: return Node(key)
    if key < root.key: root.left = insert(root.left, key)
    elif key > root.key: root.right = insert(root.right, key)
    update(root)
    if not balanced(root):
        if size(root.left) > size(root.right):
            root = rotate_right(root)
        else:
            root = rotate_left(root)
    return root

def inorder(n):
    if not n: return
    inorder(n.left)
    print(f"{n.key}({n.size})", end=" ")
    inorder(n.right)

root = None
for k in [10,20,30,40,50]:
    root = insert(root, k)
inorder(root)
```

#### Why It Matters

- Balances based on true subtree size, not height
- Excellent for order-statistics (`kth`, `rank`)
- Supports split, merge, and range queries naturally
- Deterministic balancing without randomness

#### A Gentle Proof (Why It Works)

Maintaining weight ratios guarantees logarithmic height.
If `α` < 1, each subtree has ≤ α × n nodes,
so height ≤ log₁/α(n) → O(log n).

#### Try It Yourself

1. Build [10, 20, 30, 40, 50] with α = 0.7.
2. Print sizes at each node.
3. Delete 20 → rebalance by rotation.
4. Try α = 0.6 and α = 0.8 → compare shapes.

#### Test Cases

| Sequence         | α   | Balanced? | Height   |
| ---------------- | --- | --------- | -------- |
| [10,20,30,40,50] | 0.7 | ✅         | log n    |
| [1..100]         | 0.7 | ✅         | O(log n) |
| [sorted 1..10]   | 0.6 | ✅         | ~4       |

#### Complexity

| Operation | Time     | Space |
| --------- | -------- | ----- |
| Insert    | O(log n) | O(h)  |
| Delete    | O(log n) | O(h)  |
| Search    | O(log n) | O(h)  |

A Weight Balanced Tree is like a tightrope walker, always adjusting its stance to keep perfect equilibrium, no matter how the sequence unfolds.

### 239 Scapegoat Tree Rebuild

A Scapegoat Tree is a binary search tree that maintains balance by occasionally rebuilding entire subtrees.  
Instead of performing rotations on every insertion, it monitors the depth of nodes.  
When a node’s depth exceeds the allowed limit, the algorithm locates a *scapegoat* ancestor whose subtree is too unbalanced, flattens that subtree into a sorted list, and rebuilds it into a perfectly balanced tree.

With a balance parameter $\alpha$ in $(0.5, 1)$, a Scapegoat Tree guarantees:
- $O(\log n)$ amortized time for insertions and deletions  
- $O(\log n)$ worst-case height


#### What Problem Are We Solving?

Standard BSTs can become skewed and degrade to O(n). Rotational trees like AVL and Red Black maintain strict local invariants but add per update overhead. Scapegoat trees choose a middle path

- Do nothing most of the time
- Occasionally rebuild an entire subtree when a global bound is exceeded

Goal
Keep height near $\log_{1/\alpha} n$ while keeping code simple and rotations out of the hot path.

#### How It Works (Plain Language)

Parameters and invariants

- Choose $\alpha \in (0.5, 1)$ such as $\alpha = \tfrac{2}{3}$
- Maintain `n` = current size and `n_max` = maximum size since the last global rebuild
- Height bound: if an insertion lands deeper than $\lfloor \log_{1/\alpha} n \rfloor$, the tree is too tall

Insert algorithm

1. Insert like a normal BST and track the path length `depth`.
2. If `depth` is within the allowed bound, stop.
3. Otherwise, walk up to find the scapegoat node `s` such that  
   $$
   \max\big(\text{size}(s.\text{left}),\ \text{size}(s.\text{right})\big) > \alpha \cdot \text{size}(s)
   $$
4. Rebuild the subtree rooted at `s` into a perfectly balanced BST in time linear in that subtree’s size.

Delete algorithm

- Perform a normal BST delete.  
- Decrement `n`.  
- If $n < \alpha \cdot n_{\text{max}}$, rebuild the entire tree and set $n_{\text{max}} = n$.

Why rebuilding works

- Rebuilding performs an inorder traversal to collect nodes into a sorted array, then constructs a balanced BST from that array.  
- Rebuilds are infrequent, and their costs are amortized over many cheap insertions and deletions, ensuring logarithmic time on average.


#### Example Walkthrough

Let $\alpha = \tfrac{2}{3}$.

Insert the keys `[10, 20, 30, 40, 50, 60]` in ascending order.

| Step | Action    | Depth vs bound        | Scapegoat found     | Rebuild              |
| ---- | --------- | --------------------- | ------------------- | -------------------- |
| 1    | insert 10 | depth 0 within        | none                | no                   |
| 2    | insert 20 | depth 1 within        | none                | no                   |
| 3    | insert 30 | depth 2 within        | none                | no                   |
| 4    | insert 40 | depth 3 exceeds bound | ancestor violates α | rebuild that subtree |
| 5    | insert 50 | likely within         | none                | no                   |
| 6    | insert 60 | if depth exceeds      | find ancestor       | rebuild subtree      |

The occasional rebuild produces a nearly perfect BST despite sorted input.

#### Tiny Code (Easy Versions)

Python (concise reference implementation)

```python
from math import log, floor

class Node:
    __slots__ = ("key", "left", "right", "size")
    def __init__(self, key):
        self.key = key
        self.left = None
        self.right = None
        self.size = 1

def size(x): return x.size if x else 0
def update(x): 
    if x: x.size = 1 + size(x.left) + size(x.right)

def flatten_inorder(x, arr):
    if not x: return
    flatten_inorder(x.left, arr)
    arr.append(x)
    flatten_inorder(x.right, arr)

def build_balanced(nodes, lo, hi):
    if lo >= hi: return None
    mid = (lo + hi) // 2
    root = nodes[mid]
    root.left = build_balanced(nodes, lo, mid)
    root.right = build_balanced(nodes, mid + 1, hi)
    update(root)
    return root

class ScapegoatTree:
    def __init__(self, alpha=2/3):
        self.alpha = alpha
        self.root = None
        self.n = 0
        self.n_max = 0

    def _log_alpha(self, n):
        # height bound floor(log_{1/alpha} n)
        if n <= 1: return 0
        return floor(log(n, 1 / self.alpha))

    def _find_and_rebuild(self, path):
        # path is list of nodes from root to inserted node
        for i in range(len(path) - 1, -1, -1):
            x = path[i]
            l, r = size(x.left), size(x.right)
            if max(l, r) > self.alpha * size(x):
                # rebuild subtree rooted at x and reattach to parent
                nodes = []
                flatten_inorder(x, nodes)
                new_sub = build_balanced(nodes, 0, len(nodes))
                if i == 0:
                    self.root = new_sub
                else:
                    p = path[i - 1]
                    if p.left is x: p.left = new_sub
                    else: p.right = new_sub
                    # fix sizes upward from parent
                    for j in range(i - 1, -1, -1):
                        update(path[j])
                return

    def insert(self, key):
        self.n += 1
        self.n_max = max(self.n_max, self.n)
        if not self.root:
            self.root = Node(key)
            return

        # standard BST insert with path tracking
        path = []
        cur = self.root
        while cur:
            path.append(cur)
            if key < cur.key:
                if cur.left: cur = cur.left
                else:
                    cur.left = Node(key)
                    path.append(cur.left)
                    break
            elif key > cur.key:
                if cur.right: cur = cur.right
                else:
                    cur.right = Node(key)
                    path.append(cur.right)
                    break
            else:
                # duplicate ignore
                self.n -= 1
                return
        # update sizes
        for node in reversed(path[:-1]):
            update(node)

        # depth check and possible rebuild
        if len(path) - 1 > self._log_alpha(self.n):
            self._find_and_rebuild(path)

    def _join_left_max(self, t):
        # remove and return max node from subtree t, plus the new subtree
        if not t.right:
            return t, t.left
        m, t.right = self._join_left_max(t.right)
        update(t)
        return m, t

    def delete(self, key):
        # standard BST delete
        def _del(x, key):
            if not x: return None, False
            if key < x.key:
                x.left, removed = _del(x.left, key)
            elif key > x.key:
                x.right, removed = _del(x.right, key)
            else:
                removed = True
                if not x.left: return x.right, True
                if not x.right: return x.left, True
                # replace with predecessor
                m, x.left = self._join_left_max(x.left)
                m.left, m.right = x.left, x.right
                x = m
            update(x)
            return x, removed

        self.root, removed = _del(self.root, key)
        if not removed: return
        self.n -= 1
        if self.n < self.alpha * self.n_max:
            # global rebuild
            nodes = []
            flatten_inorder(self.root, nodes)
            self.root = build_balanced(nodes, 0, len(nodes))
            self.n_max = self.n
```

C (outline of the rebuild idea)

```c
// Sketch only: show rebuild path
// 1) Do BST insert while recording path stack
// 2) If depth > bound, walk path upward to find scapegoat
// 3) Inorder-copy subtree nodes to an array
// 4) Recursively rebuild balanced subtree from that array
// 5) Reattach rebuilt subtree to parent and fix sizes up the path
```

#### Why It Matters

- Simple balancing approach with rare but powerful subtree rebuilds  
- Well suited for workloads where insertions arrive in bursts and frequent rotations are undesirable  
- Guarantees height $O(\log n)$ with a tunable constant controlled by $\alpha$  
- The inorder-based rebuild process makes ordered operations efficient and easy to implement


#### A Gentle Proof (Why It Works)

Let $\alpha \in (0.5, 1)$.

- Height bound: if height exceeds $\log_{1/\alpha} n$, a scapegoat exists since some ancestor must violate $\max\{|L|,\ |R|\} \le \alpha\,|T|$.
- Amortization: each node participates in $O(1)$ rebuilds with $\Theta(\text{size of its subtree})$ work, so the total cost over $m$ operations is $O(m \log n)$.
- Deletion rule: rebuilding when $n < \alpha\,n_{\max}$ prevents slack from accumulating.


#### Try It Yourself

1. Insert ascending keys to trigger rebuilds with $\alpha = \tfrac{2}{3}$
2. Delete many keys so that $n < \alpha\,n_{\max}$ and observe the global rebuild trigger
3. Experiment with different $\alpha$ values
4. Add an order statistic query using stored subtree sizes


#### Test Cases

| Operation        | Input                          | α    | Expected                                     |
| ---------------- | ------------------------------ | ---- | -------------------------------------------- |
| Insert ascending | [1..1000]                      | 0.66 | Height stays O(log n) with periodic rebuilds |
| Mixed ops        | random inserts and deletes     | 0.7  | Amortized O(log n) per op                    |
| Shrink check     | insert 1..200, delete 101..200 | 0.7  | Global rebuild when size drops below α n_max |
| Duplicate insert | insert 42 twice                | 0.66 | size unchanged after second insert           |

Edge Cases

- Rebuilding an empty or singleton subtree is a no-op  
- Duplicate keys are ignored or handled according to policy  
- Choose $\alpha > 0.5$ to guarantee the existence of a scapegoat


#### Complexity

| Operation       | Time                           | Space                          |
| --------------- | ------------------------------ | ------------------------------ |
| Search          | O(h) worst, O(log n) amortized | O(1) extra                     |
| Insert          | Amortized O(log n)             | O(1) extra plus rebuild buffer |
| Delete          | Amortized O(log n)             | O(1) extra plus rebuild buffer |
| Rebuild subtree | O(k) for k nodes in subtree    | O(k) temporary array           |

Scapegoat trees wait calmly until imbalance is undeniable, then rebuild decisively. The result is a BST that stays lean with minimal fuss.

### 240 AA Tree

An AA Tree is a simplified version of a red-black tree that maintains balance using a single level value per node (instead of color). It enforces balance by a small set of easy-to-code rules using two operations, `skew` and `split`. The simplicity of AA trees makes them a popular teaching and practical implementation choice when you want red-black performance without complex cases.

#### What Problem Are We Solving?

We want a balanced binary search tree with

- O(log n) time for insert/search/delete
- Simpler code than red-black or AVL
- Fewer rotation cases

AA trees achieve this by using levels (like black-heights in RBTs) to enforce structure, ensuring a balanced shape with one rotation per fix.

#### How It Works (Plain Language)

Each node has

- `key`: stored value
- `level`: like black height (root = 1)

AA-tree invariants:

1. Left child level < node level
2. Right child level ≤ node level
3. Right-right grandchild level < node level
4. Every leaf has level 1

To restore balance after insert/delete, apply skew and split operations:

| Operation | Rule                               | Action               |
| --------- | ---------------------------------- | -------------------- |
| Skew  | If left.level == node.level        | Rotate right         |
| Split | If right.right.level == node.level | Rotate left, level++ |

#### Example

Insert keys [10, 20, 30]

| Step | Tree         | Fix                                               |
| ---- | ------------ | ------------------------------------------------- |
| 1    | 10 (lvl 1)   | -                                                 |
| 2    | 10 → 20      | Skew none                                         |
| 3    | 10 → 20 → 30 | Right-right violation → Split → Rotate left on 10 |

Result:

```
    20 (2)
   /  \
 10(1) 30(1)
```

#### Step-by-Step Example

Insert [30, 20, 10, 25]

| Step | Insertion | Violation                     | Fix                 |          |
| ---- | --------- | ----------------------------- | ------------------- | -------- |
| 1    | 30        | none                          | -                   |          |
| 2    | 20        | left child level = node level | Skew → Rotate right |          |
| 3    | 10        | deeper left                   | Skew & Split        | 20 root  |
| 4    | 25        | insert right of 20            | Split if needed     | balanced |

#### Tiny Code (Easy Versions)

C

```c
#include <stdio.h>
#include <stdlib.h>

typedef struct Node {
    int key;
    int level;
    struct Node *left, *right;
} Node;

Node* new_node(int key) {
    Node* n = malloc(sizeof(Node));
    n->key = key;
    n->level = 1;
    n->left = n->right = NULL;
    return n;
}

Node* rotate_left(Node* x) {
    Node* y = x->right;
    x->right = y->left;
    y->left = x;
    return y;
}

Node* rotate_right(Node* y) {
    Node* x = y->left;
    y->left = x->right;
    x->right = y;
    return x;
}

Node* skew(Node* x) {
    if (x && x->left && x->left->level == x->level)
        x = rotate_right(x);
    return x;
}

Node* split(Node* x) {
    if (x && x->right && x->right->right && x->right->right->level == x->level) {
        x = rotate_left(x);
        x->level++;
    }
    return x;
}

Node* insert(Node* root, int key) {
    if (!root) return new_node(key);
    if (key < root->key)
        root->left = insert(root->left, key);
    else if (key > root->key)
        root->right = insert(root->right, key);
    root = skew(root);
    root = split(root);
    return root;
}

void inorder(Node* r) {
    if (!r) return;
    inorder(r->left);
    printf("(%d, %d) ", r->key, r->level);
    inorder(r->right);
}

int main(void) {
    Node* root = NULL;
    int keys[] = {10, 20, 30, 15, 25};
    for (int i = 0; i < 5; i++)
        root = insert(root, keys[i]);
    inorder(root);
    printf("\n");
}
```

Python

```python
class Node:
    def __init__(self, key):
        self.key = key
        self.level = 1
        self.left = None
        self.right = None

def rotate_left(x):
    y = x.right
    x.right = y.left
    y.left = x
    return y

def rotate_right(y):
    x = y.left
    y.left = x.right
    x.right = y
    return x

def skew(x):
    if x and x.left and x.left.level == x.level:
        x = rotate_right(x)
    return x

def split(x):
    if x and x.right and x.right.right and x.right.right.level == x.level:
        x = rotate_left(x)
        x.level += 1
    return x

def insert(root, key):
    if not root:
        return Node(key)
    if key < root.key:
        root.left = insert(root.left, key)
    elif key > root.key:
        root.right = insert(root.right, key)
    root = skew(root)
    root = split(root)
    return root

def inorder(r):
    if not r: return
    inorder(r.left)
    print(f"({r.key},{r.level})", end=" ")
    inorder(r.right)

root = None
for k in [10, 20, 30, 15, 25]:
    root = insert(root, k)
inorder(root)
```

#### Why It Matters

- Simpler than red-black
- Only two balancing functions (skew, split)
- Maintains O(log n) performance
- Great teaching and reference structure for self-balancing BSTs

#### A Gentle Proof (Why It Works)

AA-trees are a special case of red-black trees where

- Right-leaning red links represent level
- Skew fixes left-leaning reds
- Split fixes two consecutive rights

These rules ensure black-height invariants and logarithmic height.

#### Try It Yourself

1. Insert ascending keys → observe periodic right rotations
2. Draw tree after each step → check levels
3. Compare structure to equivalent red-black tree
4. Modify split condition → see imbalance appear

#### Test Cases

| Input Keys    | Expected Root | Height | Balanced? |
| ------------- | ------------- | ------ | --------- |
| [10,20,30]    | 20            | 2      | ✅         |
| [1..10]       | 4–5           | ~log n | ✅         |
| [30,20,10,25] | 20            | 3      | ✅         |

#### Complexity

| Operation | Time     | Space |
| --------- | -------- | ----- |
| Insert    | O(log n) | O(h)  |
| Search    | O(log n) | O(1)  |
| Delete    | O(log n) | O(h)  |

AA Trees blend red-black efficiency with AVL simplicity, balanced, elegant, and easy to reason about.

## Section 25. Segment Trees and Fenwick Trees 

### 241 Build Segment Tree

A segment tree is a binary tree over an array that stores range information for fast queries and updates. Building it means constructing the tree from the base array so each node summarizes a segment, like a sum, min, max, or gcd. The classic build time is O(n) and it enables O(log n) queries and updates.

#### What Problem Are We Solving?

We want to answer range questions quickly

- Sum on [l, r]
- Min or max on [l, r]
- Counts or bitwise ops on [l, r]

If we precompute a segment tree, each internal node stores the merge of its two children. Then queries and updates touch only O(log n) nodes.

Goal
Turn an array `A[0..n-1]` into a tree `T` where `T[v]` summarizes a segment `A[L..R]`.

#### How It Works (Plain Language)

Represent the tree in an array (1 indexed for simplicity)

- Node `v` covers segment `[L, R]`
- Left child `2v` covers `[L, mid]`
- Right child `2v+1` covers `[mid+1, R]`
- `T[v] = merge(T[2v], T[2v+1])` where `merge` is sum, min, max, etc.

Build Steps

1. If `L == R`, store the leaf value `A[L]` in `T[v]`.
2. Else split at `mid`, recursively build left and right, then `T[v] = merge(left, right)`.

Example Build (sum)

Array `A = [2, 1, 3, 4]`

| Node v | Segment [L,R] | Value |
| ------ | ------------- | ----- |
| 1      | [0,3]         | 10    |
| 2      | [0,1]         | 3     |
| 3      | [2,3]         | 7     |
| 4      | [0,0]         | 2     |
| 5      | [1,1]         | 1     |
| 6      | [2,2]         | 3     |
| 7      | [3,3]         | 4     |

`merge = sum`, so `T[1] = 3 + 7 = 10`, leaves store the original values.

Another Example Build (min)

Array `A = [5, 2, 6, 1]`

| Node v | Segment [L,R] | Value (min) |
| ------ | ------------- | ----------- |
| 1      | [0,3]         | 1           |
| 2      | [0,1]         | 2           |
| 3      | [2,3]         | 1           |
| 4      | [0,0]         | 5           |
| 5      | [1,1]         | 2           |
| 6      | [2,2]         | 6           |
| 7      | [3,3]         | 1           |

#### Tiny Code (Easy Versions)

C (iterative storage size 4n, recursive build for sum)

```c
#include <stdio.h>

#define MAXN 100000
int A[MAXN];
long long T[4*MAXN];

long long merge(long long a, long long b) { return a + b; }

void build(int v, int L, int R) {
    if (L == R) {
        T[v] = A[L];
        return;
    }
    int mid = (L + R) / 2;
    build(2*v, L, mid);
    build(2*v+1, mid+1, R);
    T[v] = merge(T[2*v], T[2*v+1]);
}

int main(void) {
    int n = 4;
    A[0]=2; A[1]=1; A[2]=3; A[3]=4;
    build(1, 0, n-1);
    for (int v = 1; v < 8; v++) printf("T[%d]=%lld\n", v, T[v]);
    return 0;
}
```

Python (sum segment tree, recursive build)

```python
def build(arr):
    n = len(arr)
    size = 1
    while size < n:
        size <<= 1
    T = [0] * (2 * size)

    # leaves
    for i in range(n):
        T[size + i] = arr[i]
    # internal nodes
    for v in range(size - 1, 0, -1):
        T[v] = T[2*v] + T[2*v + 1]
    return T, size  # T is 1..size*2-1, leaves start at index size

# Example
A = [2, 1, 3, 4]
T, base = build(A)
# T[1] holds sum of all, T[base+i] holds A[i]
print("Root sum:", T[1])
```

Notes

- The C version shows classic recursive top down build.
- The Python version shows an iterative bottom up build using a power of two base, also O(n).

#### Why It Matters

- Preprocessing time O(n) enables

  * Range queries in O(log n)
  * Point updates in O(log n)
- Choice of `merge` adapts the tree to many tasks

  * sum, min, max, gcd, bitwise and, custom structs

#### A Gentle Proof (Why It Works)

Each element appears in exactly one leaf and contributes to O(1) nodes per level. The tree has O(log n) levels. Total number of nodes built is at most 2n, so total work is O(n).

#### Try It Yourself

1. Change `merge` to `min` or `max` and rebuild.
2. Build then verify that `T[1]` equals `sum(A)`.
3. Print the tree level by level to visualize segments.
4. Extend the structure to support point update and range query.

#### Test Cases

| Array A   | Merge | Expected Root T[1] | Notes                        |
| --------- | ----- | ------------------ | ---------------------------- |
| [2,1,3,4] | sum   | 10                 | basic sum tree               |
| [5,2,6,1] | min   | 1                  | switch merge to min          |
| [7]       | sum   | 7                  | single element               |
| []        | sum   | 0 or no tree       | handle empty as special case |

#### Complexity

| Phase        | Time     | Space      |
| ------------ | -------- | ---------- |
| Build        | O(n)     | O(n)       |
| Query        | O(log n) | O(1) extra |
| Point update | O(log n) | O(1) extra |

Build once, answer fast forever. A segment tree turns an array into a range answering machine.

### 242 Range Sum Query

A Range Sum Query (RSQ) retrieves the sum of elements in a subarray `[L, R]` using a segment tree. Once the tree is built, each query runs in O(log n) time, by combining results from the minimal set of segments covering `[L, R]`.

#### What Problem Are We Solving?

Given an array `A[0..n-1]`, we want to answer queries like

```
sum(A[L..R])  
```

quickly, without recalculating from scratch each time.

Naive approach: O(R–L+1) per query
Segment tree approach: O(log n) per query after O(n) build.

#### How It Works (Plain Language)

We maintain a segment tree `T` built as before (`T[v]` = sum of segment).
To answer `query(v, L, R, qL, qR)`

1. If segment `[L, R]` is fully outside `[qL, qR]`, return 0.
2. If segment `[L, R]` is fully inside, return `T[v]`.
3. Otherwise, split at `mid` and combine results from left and right children.

So each query visits only O(log n) nodes, each exactly covering part of the query range.

#### Example

Array: `A = [2, 1, 3, 4, 5]`

Query: sum on [1,3] (1-based: elements 1..3 = 1+3+4=8)

| Node v | Segment [L,R] | Value | Relationship to [1,3] | Contribution |
| ------ | ------------- | ----- | --------------------- | ------------ |
| 1      | [0,4]         | 15    | overlaps              | recurse      |
| 2      | [0,2]         | 6     | overlaps              | recurse      |
| 3      | [3,4]         | 9     | overlaps              | recurse      |
| 4      | [0,1]         | 3     | partial               | recurse      |
| 5      | [2,2]         | 3     | inside                | +3           |
| 8      | [0,0]         | 2     | outside               | skip         |
| 9      | [1,1]         | 1     | inside                | +1           |
| 6      | [3,3]         | 4     | inside                | +4           |
| 7      | [4,4]         | 5     | outside               | skip         |

Sum = 1 + 3 + 4 = 8 ✅

#### Step-by-Step Table (for clarity)

| Step | Current Segment | Covered?       | Action   |
| ---- | --------------- | -------------- | -------- |
| 1    | [0,4]           | Overlaps [1,3] | Split    |
| 2    | [0,2]           | Overlaps [1,3] | Split    |
| 3    | [3,4]           | Overlaps [1,3] | Split    |
| 4    | [0,1]           | Overlaps       | Split    |
| 5    | [0,0]           | Outside        | Return 0 |
| 6    | [1,1]           | Inside         | Return 1 |
| 7    | [2,2]           | Inside         | Return 3 |
| 8    | [3,3]           | Inside         | Return 4 |
| 9    | [4,4]           | Outside        | Return 0 |

Total = 1+3+4 = 8

#### Tiny Code (Easy Versions)

C (Recursive RSQ)

```c
#include <stdio.h>

#define MAXN 100000
int A[MAXN];
long long T[4*MAXN];

long long merge(long long a, long long b) { return a + b; }

void build(int v, int L, int R) {
    if (L == R) {
        T[v] = A[L];
        return;
    }
    int mid = (L + R) / 2;
    build(2*v, L, mid);
    build(2*v+1, mid+1, R);
    T[v] = merge(T[2*v], T[2*v+1]);
}

long long query(int v, int L, int R, int qL, int qR) {
    if (qR < L || R < qL) return 0; // disjoint
    if (qL <= L && R <= qR) return T[v]; // fully covered
    int mid = (L + R) / 2;
    long long left = query(2*v, L, mid, qL, qR);
    long long right = query(2*v+1, mid+1, R, qL, qR);
    return merge(left, right);
}

int main(void) {
    int n = 5;
    int Avals[5] = {2,1,3,4,5};
    for (int i=0;i<n;i++) A[i]=Avals[i];
    build(1, 0, n-1);
    printf("Sum [1,3] = %lld\n", query(1,0,n-1,1,3)); // expect 8
}
```

Python (Iterative)

```python
def build(arr):
    n = len(arr)
    size = 1
    while size < n:
        size <<= 1
    T = [0]*(2*size)
    # leaves
    for i in range(n):
        T[size+i] = arr[i]
    # parents
    for v in range(size-1, 0, -1):
        T[v] = T[2*v] + T[2*v+1]
    return T, size

def query(T, base, l, r):
    l += base
    r += base
    res = 0
    while l <= r:
        if l % 2 == 1:
            res += T[l]
            l += 1
        if r % 2 == 0:
            res += T[r]
            r -= 1
        l //= 2
        r //= 2
    return res

A = [2,1,3,4,5]
T, base = build(A)
print("Sum [1,3] =", query(T, base, 1, 3))  # expect 8
```

#### Why It Matters

- Enables fast range queries on static or dynamic arrays.
- Foundation for many extensions:

  * Range min/max query (change merge)
  * Lazy propagation (for range updates)
  * 2D and persistent trees

#### A Gentle Proof (Why It Works)

Each level of recursion splits into disjoint segments.
At most 2 segments per level are added to the result.
Depth = O(log n), so total work = O(log n).

#### Try It Yourself

1. Query different [L, R] ranges after build.
2. Replace `merge` with `min()` or `max()` and verify results.
3. Combine queries to verify overlapping segments are counted once.
4. Add update operation to change elements and re-query.

#### Test Cases

| Array A     | Query [L,R] | Expected | Notes          |
| ----------- | ----------- | -------- | -------------- |
| [2,1,3,4,5] | [1,3]       | 8        | 1+3+4          |
| [5,5,5,5]   | [0,3]       | 20       | uniform        |
| [1,2,3,4,5] | [2,4]       | 12       | 3+4+5          |
| [7]         | [0,0]       | 7        | single element |

#### Complexity

| Operation | Time     | Space |
| --------- | -------- | ----- |
| Query     | O(log n) | O(1)  |
| Build     | O(n)     | O(n)  |
| Update    | O(log n) | O(1)  |

Segment trees let you ask *"what's in this range?"*, and get the answer fast, no matter how big the array.

### 243 Range Update (Lazy Propagation Technique)

A range update modifies all elements in a segment `[L, R]` efficiently. Without optimization, you'd touch every element, O(n). With lazy propagation, you defer work, storing pending updates in a separate array, achieving O(log n) time per update and query.

This pattern is essential when many updates overlap, like adding +5 to every element in [2, 6] repeatedly.

#### What Problem Are We Solving?

We want to efficiently support both:

1. Range updates: Add or set a value over `[L, R]`
2. Range queries: Get sum/min/max over `[L, R]`

Naive solution: O(R–L+1) per update.
Lazy propagation: O(log n) per update/query.

Example goal:

```
add +3 to A[2..5]
then query sum(A[0..7])
```

#### How It Works (Plain Language)

Each segment node `T[v]` stores summary (e.g. sum).
Each node `lazy[v]` stores pending update (not yet pushed to children).

When updating `[L, R]`:

- If node's segment is fully inside, apply update directly to `T[v]` and mark `lazy[v]` (no recursion).
- If partially overlapping, push pending updates down, then recurse.

When querying `[L, R]`:

- Push any pending updates first
- Combine child results as usual

This way, each node is updated at most once per path, giving O(log n).

#### Example

Array: `A = [1, 2, 3, 4, 5, 6, 7, 8]`
Build tree for sum.

Step 1: Range update `[2, 5]` += 3

| Node | Segment | Action              | lazy[]    | T[] sum   |
| ---- | ------- | ------------------- | --------- | --------- |
| 1    | [0,7]   | overlaps → recurse  | 0         | unchanged |
| 2    | [0,3]   | overlaps → recurse  | 0         | unchanged |
| 3    | [4,7]   | overlaps → recurse  | 0         | unchanged |
| 4    | [0,1]   | outside             | -         | -         |
| 5    | [2,3]   | inside → add +3×2=6 | lazy[5]=3 | T[5]+=6   |
| 6    | [4,5]   | inside → add +3×2=6 | lazy[6]=3 | T[6]+=6   |

Later queries automatically apply +3 to affected subranges.

#### Step-by-Step Example

Let's trace two operations:

1. `update(2,5,+3)`
2. `query(0,7)`

| Step   | Action                                | Result              |
| ------ | ------------------------------------- | ------------------- |
| Build  | sum = [1,2,3,4,5,6,7,8] → 36          | T[1]=36             |
| Update | mark lazy for segments fully in [2,5] | T[v]+=3×len         |
| Query  | propagate lazy before using T[v]      | sum = 36 + 3×4 = 48 |

Result after update: total = 48 ✅

#### Tiny Code (Easy Versions)

C (Recursive, sum tree with lazy add)

```c
#include <stdio.h>

#define MAXN 100000
int A[MAXN];
long long T[4*MAXN], lazy[4*MAXN];

long long merge(long long a, long long b) { return a + b; }

void build(int v, int L, int R) {
    if (L == R) { T[v] = A[L]; return; }
    int mid = (L + R) / 2;
    build(2*v, L, mid);
    build(2*v+1, mid+1, R);
    T[v] = merge(T[2*v], T[2*v+1]);
}

void push(int v, int L, int R) {
    if (lazy[v] != 0) {
        T[v] += lazy[v] * (R - L + 1);
        if (L != R) {
            lazy[2*v] += lazy[v];
            lazy[2*v+1] += lazy[v];
        }
        lazy[v] = 0;
    }
}

void update(int v, int L, int R, int qL, int qR, int val) {
    push(v, L, R);
    if (qR < L || R < qL) return;
    if (qL <= L && R <= qR) {
        lazy[v] += val;
        push(v, L, R);
        return;
    }
    int mid = (L + R) / 2;
    update(2*v, L, mid, qL, qR, val);
    update(2*v+1, mid+1, R, qL, qR, val);
    T[v] = merge(T[2*v], T[2*v+1]);
}

long long query(int v, int L, int R, int qL, int qR) {
    push(v, L, R);
    if (qR < L || R < qL) return 0;
    if (qL <= L && R <= qR) return T[v];
    int mid = (L + R) / 2;
    return merge(query(2*v, L, mid, qL, qR), query(2*v+1, mid+1, R, qL, qR));
}

int main(void) {
    int n = 8;
    int vals[] = {1,2,3,4,5,6,7,8};
    for (int i=0;i<n;i++) A[i]=vals[i];
    build(1,0,n-1);
    update(1,0,n-1,2,5,3);
    printf("Sum [0,7] = %lld\n", query(1,0,n-1,0,7)); // expect 48
}
```

Python (Iterative version)

```python
class LazySegTree:
    def __init__(self, arr):
        n = len(arr)
        size = 1
        while size < n: size <<= 1
        self.n = n; self.size = size
        self.T = [0]*(2*size)
        self.lazy = [0]*(2*size)
        for i in range(n): self.T[size+i] = arr[i]
        for i in range(size-1, 0, -1):
            self.T[i] = self.T[2*i] + self.T[2*i+1]
    
    def _apply(self, v, val, length):
        self.T[v] += val * length
        if v < self.size:
            self.lazy[v] += val

    def _push(self, v, length):
        if self.lazy[v]:
            self._apply(2*v, self.lazy[v], length//2)
            self._apply(2*v+1, self.lazy[v], length//2)
            self.lazy[v] = 0

    def update(self, l, r, val):
        def _upd(v, L, R):
            if r < L or R < l: return
            if l <= L and R <= r:
                self._apply(v, val, R-L+1)
                return
            self._push(v, R-L+1)
            mid = (L+R)//2
            _upd(2*v, L, mid)
            _upd(2*v+1, mid+1, R)
            self.T[v] = self.T[2*v] + self.T[2*v+1]
        _upd(1, 0, self.size-1)

    def query(self, l, r):
        def _qry(v, L, R):
            if r < L or R < l: return 0
            if l <= L and R <= r:
                return self.T[v]
            self._push(v, R-L+1)
            mid = (L+R)//2
            return _qry(2*v, L, mid) + _qry(2*v+1, mid+1, R)
        return _qry(1, 0, self.size-1)

A = [1,2,3,4,5,6,7,8]
st = LazySegTree(A)
st.update(2,5,3)
print("Sum [0,7] =", st.query(0,7))  # expect 48
```

#### Why It Matters

- Crucial for problems with many overlapping updates
- Used in range-add, range-assign, interval covering, and 2D extensions
- Foundation for Segment Tree Beats

#### A Gentle Proof (Why It Works)

Each update marks at most one node per level as "lazy."
When queried later, we "push" those updates downward once.
Each node's pending updates applied O(1) times → total cost O(log n).

#### Try It Yourself

1. Build `[1,2,3,4,5]` → update [1,3] += 2 → query sum [0,4] → expect 21
2. Update [0,4] += 1 → query [2,4] → expect 3+2+2+1+1 = 15
3. Combine multiple updates → verify cumulative results

#### Test Cases

| Operation         | Array             | Query              | Expected       |
| ----------------- | ----------------- | ------------------ | -------------- |
| update [2,5] += 3 | [1,2,3,4,5,6,7,8] | sum [0,7]          | 48             |
| update [0,3] += 2 | [5,5,5,5]         | sum [1,2]          | 14             |
| two updates       | [1,1,1,1,1]       | [1,3] +1, [2,4] +2 | [1,3] sum = 10 |

#### Complexity

| Operation    | Time     | Space |
| ------------ | -------- | ----- |
| Range update | O(log n) | O(n)  |
| Range query  | O(log n) | O(n)  |
| Build        | O(n)     | O(n)  |

Lazy propagation, work smarter, not harder. Apply only what's needed, when it's needed.

### 244 Point Update

A point update changes a single element in the array and updates all relevant segment tree nodes along the path to the root. This operation ensures the segment tree remains consistent for future range queries.

Unlike range updates (which mark many elements at once), point updates touch only O(log n) nodes, one per level.

#### What Problem Are We Solving?

Given a segment tree built over `A[0..n-1]`, we want to:

- Change one element: `A[pos] = new_value`
- Reflect the change in the segment tree so all range queries stay correct

Goal: Update efficiently, no full rebuild

Naive: rebuild tree → O(n)
Segment tree point update: O(log n)

#### How It Works (Plain Language)

Every tree node `T[v]` represents a segment `[L, R]`.
When `pos` lies within `[L, R]`, that node's value might need adjustment.

Algorithm (recursive):

1. If `L == R == pos`, assign `A[pos] = val`, set `T[v] = val`.
2. Else, find `mid`.

   * Recurse into left or right child depending on `pos`.
   * After child updates, recompute `T[v] = merge(T[2v], T[2v+1])`.

No lazy propagation needed, it's a direct path down the tree.

#### Example

Array: `A = [2, 1, 3, 4]`
Tree stores sum.

| Node | Range | Value |
| ---- | ----- | ----- |
| 1    | [0,3] | 10    |
| 2    | [0,1] | 3     |
| 3    | [2,3] | 7     |
| 4    | [0,0] | 2     |
| 5    | [1,1] | 1     |
| 6    | [2,2] | 3     |
| 7    | [3,3] | 4     |

Operation: `A[1] = 5`

Path: [1, 2, 5]

| Step   | Node | Old Value | New Value | Update      |
| ------ | ---- | --------- | --------- | ----------- |
| Leaf   | 5    | 1         | 5         | T[5]=5      |
| Parent | 2    | 3         | 7         | T[2]=2+5=7  |
| Root   | 1    | 10        | 14        | T[1]=7+7=14 |

New array: `[2, 5, 3, 4]`
New sum: 14 ✅

#### Step-by-Step Trace

```
update(v=1, L=0, R=3, pos=1, val=5)
  mid=1
  -> pos<=mid → left child (v=2)
    update(v=2, L=0, R=1)
      mid=0
      -> pos>mid → right child (v=5)
        update(v=5, L=1, R=1)
        T[5]=5
      T[2]=merge(T[4]=2, T[5]=5)=7
  T[1]=merge(T[2]=7, T[3]=7)=14
```

#### Tiny Code (Easy Versions)

C (Recursive, sum merge)

```c
#include <stdio.h>

#define MAXN 100000
int A[MAXN];
long long T[4*MAXN];

long long merge(long long a, long long b) { return a + b; }

void build(int v, int L, int R) {
    if (L == R) { T[v] = A[L]; return; }
    int mid = (L + R)/2;
    build(2*v, L, mid);
    build(2*v+1, mid+1, R);
    T[v] = merge(T[2*v], T[2*v+1]);
}

void point_update(int v, int L, int R, int pos, int val) {
    if (L == R) {
        T[v] = val;
        A[pos] = val;
        return;
    }
    int mid = (L + R)/2;
    if (pos <= mid) point_update(2*v, L, mid, pos, val);
    else point_update(2*v+1, mid+1, R, pos, val);
    T[v] = merge(T[2*v], T[2*v+1]);
}

long long query(int v, int L, int R, int qL, int qR) {
    if (qR < L || R < qL) return 0;
    if (qL <= L && R <= qR) return T[v];
    int mid = (L + R)/2;
    return merge(query(2*v, L, mid, qL, qR), query(2*v+1, mid+1, R, qL, qR));
}

int main(void) {
    int n = 4;
    int vals[] = {2,1,3,4};
    for (int i=0;i<n;i++) A[i]=vals[i];
    build(1,0,n-1);
    printf("Before: sum [0,3] = %lld\n", query(1,0,n-1,0,3)); // 10
    point_update(1,0,n-1,1,5);
    printf("After:  sum [0,3] = %lld\n", query(1,0,n-1,0,3)); // 14
}
```

Python (Iterative)

```python
def build(arr):
    n = len(arr)
    size = 1
    while size < n: size <<= 1
    T = [0]*(2*size)
    for i in range(n):
        T[size+i] = arr[i]
    for v in range(size-1, 0, -1):
        T[v] = T[2*v] + T[2*v+1]
    return T, size

def update(T, base, pos, val):
    v = base + pos
    T[v] = val
    v //= 2
    while v >= 1:
        T[v] = T[2*v] + T[2*v+1]
        v //= 2

def query(T, base, l, r):
    l += base; r += base
    res = 0
    while l <= r:
        if l%2==1: res += T[l]; l+=1
        if r%2==0: res += T[r]; r-=1
        l//=2; r//=2
    return res

A = [2,1,3,4]
T, base = build(A)
print("Before:", query(T, base, 0, 3))  # 10
update(T, base, 1, 5)
print("After:", query(T, base, 0, 3))   # 14
```

#### Why It Matters

- Foundation for dynamic data, quick local edits
- Used in segment trees, Fenwick trees, and BIT
- Great for applications like dynamic scoring, cumulative sums, real-time data updates

#### A Gentle Proof (Why It Works)

Each level has 1 node affected by position `pos`.
Tree height = O(log n).
Thus, exactly O(log n) nodes recomputed.

#### Try It Yourself

1. Build `[2,1,3,4]`, update `A[2]=10` → sum [0,3]=17
2. Replace merge with `min` → update element → test queries
3. Compare time with full rebuild for large n

#### Test Cases

| Input A   | Update | Query    | Expected |
| --------- | ------ | -------- | -------- |
| [2,1,3,4] | A[1]=5 | sum[0,3] | 14       |
| [5,5,5]   | A[2]=2 | sum[0,2] | 12       |
| [1]       | A[0]=9 | sum[0,0] | 9        |

#### Complexity

| Operation    | Time     | Space |
| ------------ | -------- | ----- |
| Point update | O(log n) | O(1)  |
| Query        | O(log n) | O(1)  |
| Build        | O(n)     | O(n)  |

A point update is like a ripple in a pond, a single change flows upward, keeping the whole structure in harmony.

### 245 Fenwick Tree Build

A Fenwick Tree, or Binary Indexed Tree (BIT), is a compact data structure for prefix queries and point updates. It's perfect for cumulative sums, frequencies, or any associative operation. Building it efficiently sets the stage for O(log n) queries and updates.

Unlike segment trees, a Fenwick Tree uses clever index arithmetic to represent overlapping ranges in a single array.

#### What Problem Are We Solving?

We want to precompute a structure to:

- Answer prefix queries: `sum(0..i)`
- Support updates: `A[i] += delta`

Naive prefix sum array:

- Query: O(1)
- Update: O(n)

Fenwick Tree:

- Query: O(log n)
- Update: O(log n)
- Build: O(n)

#### How It Works (Plain Language)

A Fenwick Tree uses the last set bit (LSB) of an index to determine segment length.

Each node `BIT[i]` stores sum of range:

```
(i - LSB(i) + 1) .. i
```

So:

- `BIT[1]` stores A[1]
- `BIT[2]` stores A[1..2]
- `BIT[3]` stores A[3]
- `BIT[4]` stores A[1..4]
- `BIT[5]` stores A[5]
- `BIT[6]` stores A[5..6]

Prefix sum = sum of these overlapping ranges.

#### Example

Array `A = [2, 1, 3, 4, 5]` (1-indexed for clarity)

| i | A[i] | LSB(i) | Range Stored | BIT[i] (sum) |
| - | ---- | ------ | ------------ | ------------ |
| 1 | 2    | 1      | [1]          | 2            |
| 2 | 1    | 2      | [1..2]       | 3            |
| 3 | 3    | 1      | [3]          | 3            |
| 4 | 4    | 4      | [1..4]       | 10           |
| 5 | 5    | 1      | [5]          | 5            |

#### Step-by-Step Build (O(n))

Iterate i from 1 to n:

1. `BIT[i] += A[i]`
2. Add `BIT[i]` to its parent: `BIT[i + LSB(i)] += BIT[i]`

| Step | i | LSB(i) | Update BIT[i+LSB(i)]             | Result BIT  |
| ---- | - | ------ | -------------------------------- | ----------- |
| 1    | 1 | 1      | BIT[2]+=2                        | [2,3,0,0,0] |
| 2    | 2 | 2      | BIT[4]+=3                        | [2,3,0,3,0] |
| 3    | 3 | 1      | BIT[4]+=3                        | [2,3,3,6,0] |
| 4    | 4 | 4      | BIT[8]+=6 (ignore, out of range) | [2,3,3,6,0] |
| 5    | 5 | 1      | BIT[6]+=5 (ignore, out of range) | [2,3,3,6,5] |

Built BIT: `[2, 3, 3, 6, 5]` ✅

#### Tiny Code (Easy Versions)

C (O(n) Build)

```c
#include <stdio.h>

#define MAXN 100005
int A[MAXN];
long long BIT[MAXN];
int n;

int lsb(int i) { return i & -i; }

void build() {
    for (int i = 1; i <= n; i++) {
        BIT[i] += A[i];
        int parent = i + lsb(i);
        if (parent <= n)
            BIT[parent] += BIT[i];
    }
}

long long prefix_sum(int i) {
    long long s = 0;
    while (i > 0) {
        s += BIT[i];
        i -= lsb(i);
    }
    return s;
}

int main(void) {
    n = 5;
    int vals[6] = {0, 2, 1, 3, 4, 5}; // 1-indexed
    for (int i=1;i<=n;i++) A[i]=vals[i];
    build();
    printf("Prefix sum [1..3] = %lld\n", prefix_sum(3)); // expect 6
}
```

Python (1-indexed)

```python
def build(A):
    n = len(A) - 1
    BIT = [0]*(n+1)
    for i in range(1, n+1):
        BIT[i] += A[i]
        parent = i + (i & -i)
        if parent <= n:
            BIT[parent] += BIT[i]
    return BIT

def prefix_sum(BIT, i):
    s = 0
    while i > 0:
        s += BIT[i]
        i -= (i & -i)
    return s

A = [0,2,1,3,4,5]  # 1-indexed
BIT = build(A)
print("BIT =", BIT[1:])
print("Sum[1..3] =", prefix_sum(BIT, 3))  # expect 6
```

#### Why It Matters

- Lightweight alternative to segment tree
- O(n) build, O(log n) query, O(log n) update
- Used in frequency tables, inversion count, prefix queries, cumulative histograms

#### A Gentle Proof (Why It Works)

Each index i contributes to at most log n BIT entries.
Every BIT[i] stores sum of a disjoint range defined by LSB.
Building with parent propagation ensures correct overlapping coverage.

#### Try It Yourself

1. Build BIT from [2, 1, 3, 4, 5] → query sum(3)=6
2. Update A[2]+=2 → prefix(3)=8
3. Compare with cumulative sum array for correctness

#### Test Cases

| A (1-indexed) | Query     | Expected |
| ------------- | --------- | -------- |
| [2,1,3,4,5]   | prefix(3) | 6        |
| [5,5,5,5]     | prefix(4) | 20       |
| [1]           | prefix(1) | 1        |

#### Complexity

| Operation | Time     | Space |
| --------- | -------- | ----- |
| Build     | O(n)     | O(n)  |
| Query     | O(log n) | O(1)  |
| Update    | O(log n) | O(1)  |

A Fenwick Tree is the art of doing less, storing just enough to make prefix sums lightning-fast.

### 246 Fenwick Update

A Fenwick Tree (Binary Indexed Tree) supports point updates, adjusting a single element and efficiently reflecting that change across all relevant cumulative sums. The update propagates upward through parent indices determined by the Least Significant Bit (LSB).

#### What Problem Are We Solving?

Given a built Fenwick Tree, we want to perform an operation like:

```
A[pos] += delta
```

and keep all prefix sums `sum(0..i)` consistent.

Naive approach: update every prefix → O(n)
Fenwick Tree: propagate through selected indices → O(log n)

#### How It Works (Plain Language)

In a Fenwick Tree, `BIT[i]` covers the range `(i - LSB(i) + 1) .. i`.
So when we update `A[pos]`, we must add delta to all `BIT[i]` where `i` includes `pos` in its range.

Rule:

```
for (i = pos; i <= n; i += LSB(i))
    BIT[i] += delta
```

- LSB(i) jumps to the next index covering `pos`
- Stops when index exceeds `n`

#### Example

Array `A = [2, 1, 3, 4, 5]` (1-indexed)
BIT built as `[2, 3, 3, 10, 5]`

Now perform update(2, +2) → A[2] = 3

| Step | i | LSB(i) | BIT[i] Change | New BIT      |
| ---- | - | ------ | ------------- | ------------ |
| 1    | 2 | 2      | BIT[2]+=2     | [2,5,3,10,5] |
| 2    | 4 | 4      | BIT[4]+=2     | [2,5,3,12,5] |
| 3    | 8 | stop   | -             | done         |

New prefix sums reflect updated array `[2,3,3,4,5]`

Check prefix(3) = 2+3+3=8 ✅

#### Step-by-Step Table

| Prefix | Old Sum | New Sum |
| ------ | ------- | ------- |
| 1      | 2       | 2       |
| 2      | 3       | 5       |
| 3      | 6       | 8       |
| 4      | 10      | 12      |
| 5      | 15      | 17      |

#### Tiny Code (Easy Versions)

C (Fenwick Update)

```c
#include <stdio.h>

#define MAXN 100005
long long BIT[MAXN];
int n;

int lsb(int i) { return i & -i; }

void update(int pos, int delta) {
    for (int i = pos; i <= n; i += lsb(i))
        BIT[i] += delta;
}

long long prefix_sum(int i) {
    long long s = 0;
    for (; i > 0; i -= lsb(i))
        s += BIT[i];
    return s;
}

int main(void) {
    n = 5;
    // Build from A = [2, 1, 3, 4, 5]
    BIT[1]=2; BIT[2]=3; BIT[3]=3; BIT[4]=10; BIT[5]=5;
    update(2, 2); // A[2]+=2
    printf("Sum [1..3] = %lld\n", prefix_sum(3)); // expect 8
}
```

Python (1-indexed)

```python
def update(BIT, n, pos, delta):
    while pos <= n:
        BIT[pos] += delta
        pos += (pos & -pos)

def prefix_sum(BIT, pos):
    s = 0
    while pos > 0:
        s += BIT[pos]
        pos -= (pos & -pos)
    return s

# Example
BIT = [0,2,3,3,10,5]  # built from [2,1,3,4,5]
update(BIT, 5, 2, 2)
print("Sum [1..3] =", prefix_sum(BIT, 3))  # expect 8
```

#### Why It Matters

- Enables real-time data updates with fast prefix queries
- Simpler and more space-efficient than segment trees for sums
- Core of many algorithms: inversion count, frequency accumulation, order statistics

#### A Gentle Proof (Why It Works)

Each `BIT[i]` covers a fixed range `(i - LSB(i) + 1 .. i)`.
If `pos` lies in that range, incrementing `BIT[i]` ensures correct future prefix sums.
Since each update moves by LSB(i), at most log₂(n) steps occur.

#### Try It Yourself

1. Build BIT from `[2,1,3,4,5]` → update(2,+2)
2. Query prefix(3) → expect 8
3. Update(5, +5) → prefix(5) = 22
4. Chain multiple updates → verify incremental sums

#### Test Cases

| A (1-indexed) | Update | Query     | Expected |
| ------------- | ------ | --------- | -------- |
| [2,1,3,4,5]   | (2,+2) | sum[1..3] | 8        |
| [5,5,5,5]     | (4,+1) | sum[1..4] | 21       |
| [1,2,3,4,5]   | (5,-2) | sum[1..5] | 13       |

#### Complexity

| Operation | Time     | Space |
| --------- | -------- | ----- |
| Update    | O(log n) | O(1)  |
| Query     | O(log n) | O(1)  |
| Build     | O(n)     | O(n)  |

Each update is a ripple that climbs the tree, keeping all prefix sums in perfect sync.

### 247 Fenwick Query

Once you've built or updated a Fenwick Tree (Binary Indexed Tree), you'll want to extract useful information, usually prefix sums. The query operation walks downward through the tree using index arithmetic, gathering partial sums along the way.

This simple yet powerful routine turns a linear prefix-sum scan into an elegant O(log n) solution.

#### What Problem Are We Solving?

We need to compute:

```
sum(1..i) = A[1] + A[2] + ... + A[i]
```

efficiently, after updates.

In a Fenwick Tree, each index holds the sum of a segment determined by its LSB (Least Significant Bit).
By moving downward (subtracting LSB each step), we collect disjoint ranges that together cover `[1..i]`.

#### How It Works (Plain Language)

Each `BIT[i]` stores sum of range `(i - LSB(i) + 1 .. i)`.
So to get the prefix sum, we combine all these segments by walking downward:

```
sum = 0
while i > 0:
    sum += BIT[i]
    i -= LSB(i)
```

Key Insight:

- Update: moves upward (`i += LSB(i)`)
- Query: moves downward (`i -= LSB(i)`)

Together, they mirror each other like yin and yang of cumulative logic.

#### Example

Suppose we have `A = [2, 3, 3, 4, 5]` (1-indexed),
Built BIT = [2, 5, 3, 12, 5]

Let's compute `prefix_sum(5)`

| Step | i | BIT[i] | LSB(i) | Accumulated Sum | Explanation          |
| ---- | - | ------ | ------ | --------------- | -------------------- |
| 1    | 5 | 5      | 1      | 5               | Add BIT[5] (A[5])    |
| 2    | 4 | 12     | 4      | 17              | Add BIT[4] (A[1..4]) |
| 3    | 0 | -      | -      | stop            | done                 |

✅ `prefix_sum(5) = 17` matches `2+3+3+4+5=17`

Now try `prefix_sum(3)`

| Step | i | BIT[i] | LSB(i) | Sum  |
| ---- | - | ------ | ------ | ---- |
| 1    | 3 | 3      | 1      | 3    |
| 2    | 2 | 5      | 2      | 8    |
| 3    | 0 | -      | -      | stop |

✅ `prefix_sum(3) = 8`

#### Tiny Code (Easy Versions)

C (Fenwick Query)

```c
#include <stdio.h>

#define MAXN 100005
long long BIT[MAXN];
int n;

int lsb(int i) { return i & -i; }

long long prefix_sum(int i) {
    long long s = 0;
    while (i > 0) {
        s += BIT[i];
        i -= lsb(i);
    }
    return s;
}

int main(void) {
    n = 5;
    long long B[6] = {0,2,5,3,12,5}; // built BIT
    for (int i=1;i<=n;i++) BIT[i] = B[i];
    printf("Prefix sum [1..3] = %lld\n", prefix_sum(3)); // expect 8
    printf("Prefix sum [1..5] = %lld\n", prefix_sum(5)); // expect 17
}
```

Python (1-indexed)

```python
def prefix_sum(BIT, i):
    s = 0
    while i > 0:
        s += BIT[i]
        i -= (i & -i)
    return s

BIT = [0,2,5,3,12,5]
print("sum[1..3] =", prefix_sum(BIT, 3))  # 8
print("sum[1..5] =", prefix_sum(BIT, 5))  # 17
```

#### Why It Matters

- Fast prefix sums after dynamic updates
- Essential for frequency tables, order statistics, inversion count
- Core to many competitive programming tricks (like "count less than k")

#### A Gentle Proof (Why It Works)

Each index `i` contributes to a fixed set of BIT nodes.
When querying, we collect all BIT segments that together form `[1..i]`.
The LSB ensures no overlap, each range is disjoint.
Total steps = number of bits set in `i` = O(log n).

#### Try It Yourself

1. Build BIT from `[2,3,3,4,5]`
2. Query prefix_sum(3) → expect 8
3. Update(2,+2), query(3) → expect 10
4. Query prefix_sum(5) → confirm correctness

#### Test Cases

| A (1-indexed) | Query  | Expected |
| ------------- | ------ | -------- |
| [2,3,3,4,5]   | sum(3) | 8        |
| [2,3,3,4,5]   | sum(5) | 17       |
| [1,2,3,4,5]   | sum(4) | 10       |

#### Complexity

| Operation | Time     | Space |
| --------- | -------- | ----- |
| Query     | O(log n) | O(1)  |
| Update    | O(log n) | O(1)  |
| Build     | O(n)     | O(n)  |

Fenwick query is the graceful descent, follow the bits down to gather every piece of the sum puzzle.

### 248 Segment Tree Merge

A Segment Tree can handle more than sums, it's a versatile structure that can merge results from two halves of a range using any associative operation (sum, min, max, gcd, etc.). The merge function is the heart of the segment tree: it tells us how to combine child nodes into a parent node.

#### What Problem Are We Solving?

We want to combine results from left and right child intervals into one parent result.
Without merge logic, the tree can't aggregate data or answer queries.

For example:

- For sum queries → merge(a, b) = a + b
- For min queries → merge(a, b) = min(a, b)
- For gcd queries → merge(a, b) = gcd(a, b)

So the merge function defines *what the tree means*.

#### How It Works (Plain Language)

Each node represents a segment `[L, R]`.
Its value is derived from its two children:

```
node = merge(left_child, right_child)
```

When you:

- Build: compute node from children recursively
- Query: merge partial overlaps
- Update: recompute affected nodes using merge

So merge is the unifying rule that glues the tree together.

#### Example (Sum Segment Tree)

Array `A = [2, 1, 3, 4]`

| Node  | Range  | Left | Right | Merge (sum) |
| ----- | ------ | ---- | ----- | ----------- |
| root  | [1..4] | 6    | 4     | 10          |
| left  | [1..2] | 2    | 1     | 3           |
| right | [3..4] | 3    | 4     | 7           |

Each parent = sum(left, right)

Tree structure:

```
          [1..4]=10
         /         \
   [1..2]=3       [3..4]=7
   /    \         /     \
$$1]=2 [2]=1   [3]=3   [4]=4
```

Merge rule: `merge(a, b) = a + b`

#### Example (Min Segment Tree)

Array `A = [5, 2, 7, 1]`
Merge rule: `min(a, b)`

| Node  | Range  | Left | Right | Merge (min) |
| ----- | ------ | ---- | ----- | ----------- |
| root  | [1..4] | 2    | 1     | 1           |
| left  | [1..2] | 5    | 2     | 2           |
| right | [3..4] | 7    | 1     | 1           |

Result: root stores 1, the global minimum.

#### Tiny Code (Easy Versions)

C (Sum Segment Tree Merge)

```c
#include <stdio.h>
#define MAXN 100005

int A[MAXN], tree[4*MAXN];
int n;

int merge(int left, int right) {
    return left + right;
}

void build(int node, int l, int r) {
    if (l == r) {
        tree[node] = A[l];
        return;
    }
    int mid = (l + r) / 2;
    build(node*2, l, mid);
    build(node*2+1, mid+1, r);
    tree[node] = merge(tree[node*2], tree[node*2+1]);
}

int query(int node, int l, int r, int ql, int qr) {
    if (qr < l || ql > r) return 0; // neutral element
    if (ql <= l && r <= qr) return tree[node];
    int mid = (l + r) / 2;
    int left = query(node*2, l, mid, ql, qr);
    int right = query(node*2+1, mid+1, r, ql, qr);
    return merge(left, right);
}

int main(void) {
    n = 4;
    int vals[5] = {0, 2, 1, 3, 4};
    for (int i=1; i<=n; i++) A[i] = vals[i];
    build(1,1,n);
    printf("Sum [1..4] = %d\n", query(1,1,n,1,4)); // 10
    printf("Sum [2..3] = %d\n", query(1,1,n,2,3)); // 4
}
```

Python

```python
def merge(a, b):
    return a + b  # define operation here

def build(arr, tree, node, l, r):
    if l == r:
        tree[node] = arr[l]
        return
    mid = (l + r) // 2
    build(arr, tree, 2*node, l, mid)
    build(arr, tree, 2*node+1, mid+1, r)
    tree[node] = merge(tree[2*node], tree[2*node+1])

def query(tree, node, l, r, ql, qr):
    if qr < l or ql > r:
        return 0  # neutral element for sum
    if ql <= l and r <= qr:
        return tree[node]
    mid = (l + r)//2
    left = query(tree, 2*node, l, mid, ql, qr)
    right = query(tree, 2*node+1, mid+1, r, ql, qr)
    return merge(left, right)

A = [0, 2, 1, 3, 4]
n = 4
tree = [0]*(4*n)
build(A, tree, 1, 1, n)
print("Sum[1..4] =", query(tree, 1, 1, n, 1, 4))  # 10
```

#### Why It Matters

- Merge is the "soul" of the segment tree, define merge, and you define the tree's purpose.
- Flexible across many tasks: sums, min/max, GCD, XOR, matrix multiplication.
- Unified pattern: build, query, update all rely on the same operation.

#### A Gentle Proof (Why It Works)

Segment tree works by divide and conquer.
If an operation is associative (like +, min, max, gcd), merging partial results yields the same answer as computing on the whole range.
So as long as `merge(a, b)` = `merge(b, a)` and associative, correctness follows.

#### Try It Yourself

1. Replace merge with `min(a,b)` → build min segment tree
2. Replace merge with `max(a,b)` → build max segment tree
3. Replace merge with `__gcd(a,b)` → build gcd tree
4. Test `query(2,3)` after changing merge rule

#### Test Cases

| A          | Operation | Query  | Expected |
| ---------- | --------- | ------ | -------- |
| [2,1,3,4]  | sum       | [1..4] | 10       |
| [5,2,7,1]  | min       | [1..4] | 1        |
| [3,6,9,12] | gcd       | [2..4] | 3        |

#### Complexity

| Operation | Time     | Space |
| --------- | -------- | ----- |
| Build     | O(n)     | O(n)  |
| Query     | O(log n) | O(n)  |
| Update    | O(log n) | O(n)  |

A merge function is the heartbeat of every segment tree, once you define it, your tree learns what "combine" means.

### 249 Persistent Segment Tree

A Persistent Segment Tree is a magical structure that lets you time-travel. Instead of overwriting nodes on update, it creates new versions while preserving old ones. Each update returns a new root, giving you full history and O(log n) access to every version.

#### What Problem Are We Solving?

We need a data structure that supports:

- Point updates without losing past state
- Range queries on any historical version

Use cases:

- Undo/rollback systems
- Versioned databases
- Offline queries ("What was sum[1..3] after the 2nd update?")

Each version is immutable, perfect for functional programming or auditability.

#### How It Works (Plain Language)

A persistent segment tree clones only nodes along the path from root to updated leaf.
All other nodes are shared.

For each update:

1. Create a new root.
2. Copy nodes along the path to the changed index.
3. Reuse unchanged subtrees.

Result: O(log n) memory per version, not O(n).

#### Example

Array `A = [1, 2, 3, 4]`

Version 0: built tree for `[1, 2, 3, 4]`

- sum = 10

Update A[2] = 5

- Create Version 1, copy path to A[2]
- New sum = 1 + 5 + 3 + 4 = 13

| Version | A         | Sum(1..4) | Sum(1..2) |
| ------- | --------- | --------- | --------- |
| 0       | [1,2,3,4] | 10        | 3         |
| 1       | [1,5,3,4] | 13        | 6         |

Both versions exist side by side.
Version 0 is unchanged. Version 1 reflects new value.

#### Visualization

```
Version 0: root0
          /       \
      [1..2]=3   [3..4]=7
      /    \      /     \
   [1]=1 [2]=2 [3]=3 [4]=4

Version 1: root1 (new)
          /       \
  [1..2]=6*       [3..4]=7 (shared)
    /     \
$$1]=1   [2]=5*
```

Asterisks mark newly created nodes.

#### Tiny Code (Easy Version)

C (Pointer-based, sum tree)

```c
#include <stdio.h>
#include <stdlib.h>

typedef struct Node {
    int val;
    struct Node *left, *right;
} Node;

Node* build(int arr[], int l, int r) {
    Node* node = malloc(sizeof(Node));
    if (l == r) {
        node->val = arr[l];
        node->left = node->right = NULL;
        return node;
    }
    int mid = (l + r) / 2;
    node->left = build(arr, l, mid);
    node->right = build(arr, mid+1, r);
    node->val = node->left->val + node->right->val;
    return node;
}

Node* update(Node* prev, int l, int r, int pos, int new_val) {
    Node* node = malloc(sizeof(Node));
    if (l == r) {
        node->val = new_val;
        node->left = node->right = NULL;
        return node;
    }
    int mid = (l + r) / 2;
    if (pos <= mid) {
        node->left = update(prev->left, l, mid, pos, new_val);
        node->right = prev->right;
    } else {
        node->left = prev->left;
        node->right = update(prev->right, mid+1, r, pos, new_val);
    }
    node->val = node->left->val + node->right->val;
    return node;
}

int query(Node* node, int l, int r, int ql, int qr) {
    if (qr < l || ql > r) return 0;
    if (ql <= l && r <= qr) return node->val;
    int mid = (l + r)/2;
    return query(node->left, l, mid, ql, qr)
         + query(node->right, mid+1, r, ql, qr);
}

int main(void) {
    int A[5] = {0,1,2,3,4}; // 1-indexed
    Node* root0 = build(A, 1, 4);
    Node* root1 = update(root0, 1, 4, 2, 5);
    printf("v0 sum[1..2]=%d\n", query(root0,1,4,1,2)); // 3
    printf("v1 sum[1..2]=%d\n", query(root1,1,4,1,2)); // 6
}
```

Python (Recursive, sum tree)

```python
class Node:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def build(arr, l, r):
    if l == r:
        return Node(arr[l])
    mid = (l + r) // 2
    left = build(arr, l, mid)
    right = build(arr, mid+1, r)
    return Node(left.val + right.val, left, right)

def update(prev, l, r, pos, val):
    if l == r:
        return Node(val)
    mid = (l + r) // 2
    if pos <= mid:
        return Node(prev.val - prev.left.val + val,
                    update(prev.left, l, mid, pos, val),
                    prev.right)
    else:
        return Node(prev.val - prev.right.val + val,
                    prev.left,
                    update(prev.right, mid+1, r, pos, val))

def query(node, l, r, ql, qr):
    if qr < l or ql > r: return 0
    if ql <= l and r <= qr: return node.val
    mid = (l + r)//2
    return query(node.left, l, mid, ql, qr) + query(node.right, mid+1, r, ql, qr)

A = [0,1,2,3,4]
root0 = build(A, 1, 4)
root1 = update(root0, 1, 4, 2, 5)
print("v0 sum[1..2] =", query(root0, 1, 4, 1, 2))  # 3
print("v1 sum[1..2] =", query(root1, 1, 4, 1, 2))  # 6
```

#### Why It Matters

- Immutable versions → ideal for undo systems, snapshots, persistent databases
- Saves memory (O(log n) per version)
- Each version is fully functional and independent

#### A Gentle Proof (Why It Works)

Each update affects O(log n) nodes. By copying only those nodes, we maintain O(log n) new memory.
Because all old nodes remain referenced, no data is lost.
Thus, each version is consistent and immutable.

#### Try It Yourself

1. Build version 0 with `[1,2,3,4]`
2. Update(2,5) → version 1
3. Query sum(1,4) in both versions → 10, 13
4. Create version 2 by update(3,1)
5. Query sum(1,3) in each version

#### Test Cases

| Version | A         | Query     | Expected |
| ------- | --------- | --------- | -------- |
| 0       | [1,2,3,4] | sum[1..2] | 3        |
| 1       | [1,5,3,4] | sum[1..2] | 6        |
| 2       | [1,5,1,4] | sum[1..3] | 7        |

#### Complexity

| Operation | Time     | Space (per version) |
| --------- | -------- | ------------------- |
| Build     | O(n)     | O(n)                |
| Update    | O(log n) | O(log n)            |
| Query     | O(log n) | O(1)                |

Each version of a persistent segment tree is a snapshot in time, perfect memory of every past, with no cost of forgetting.

### 250 2D Segment Tree

A 2D Segment Tree extends the classic segment tree into two dimensions, perfect for handling range queries over matrices, such as sums, minimums, or maximums across subrectangles.

It's the data structure that lets you ask:

> "What's the sum of elements in the rectangle (x1, y1) to (x2, y2)?"
> and still answer in O(log² n) time.

#### What Problem Are We Solving?

We want to perform two operations efficiently on a 2D grid:

1. Range Query: sum/min/max over a submatrix
2. Point Update: modify one element and reflect the change

Naive approach → O(n²) per query
2D Segment Tree → O(log² n) per query and update

#### How It Works (Plain Language)

Think of a 2D segment tree as a segment tree of segment trees.

1. Outer tree partitions rows.
2. Each node of the outer tree holds an inner segment tree for its rows.

Each node thus represents a rectangle in the matrix.

At build time:

- Combine children horizontally and vertically using a merge rule (like sum).

At query time:

- Combine answers from nodes overlapping the query rectangle.

#### Example

Matrix A (3×3):

|     | y=1 | y=2 | y=3 |
| --- | --- | --- | --- |
| x=1 | 2   | 1   | 3   |
| x=2 | 4   | 5   | 6   |
| x=3 | 7   | 8   | 9   |

Query rectangle `(1,1)` to `(2,2)`
→ 2 + 1 + 4 + 5 = 12

#### Key Idea

For each row range, build a column segment tree.
For each parent node (row range), merge children column trees:

```
tree[x][y] = merge(tree[2*x][y], tree[2*x+1][y])
```

#### Example Walkthrough (Sum Tree)

1. Build row segment tree
2. Inside each row node, build column segment tree
3. To query rectangle `(x1,y1)` to `(x2,y2)`:

   * Query over x-range → merge vertical results
   * Inside each x-node, query y-range → merge horizontal results

#### Tiny Code (Easy Version)

Python (Sum 2D Segment Tree)

```python
class SegmentTree2D:
    def __init__(self, mat):
        self.n = len(mat)
        self.m = len(mat[0])
        self.tree = [[0]*(4*self.m) for _ in range(4*self.n)]
        self.mat = mat
        self.build_x(1, 0, self.n-1)

    def merge(self, a, b):
        return a + b  # sum merge

    # build column tree for a fixed row range
    def build_y(self, nodex, lx, rx, nodey, ly, ry):
        if ly == ry:
            if lx == rx:
                self.tree[nodex][nodey] = self.mat[lx][ly]
            else:
                self.tree[nodex][nodey] = self.merge(
                    self.tree[2*nodex][nodey], self.tree[2*nodex+1][nodey]
                )
            return
        midy = (ly + ry)//2
        self.build_y(nodex, lx, rx, 2*nodey, ly, midy)
        self.build_y(nodex, lx, rx, 2*nodey+1, midy+1, ry)
        self.tree[nodex][nodey] = self.merge(
            self.tree[nodex][2*nodey], self.tree[nodex][2*nodey+1]
        )

    # build row tree
    def build_x(self, nodex, lx, rx):
        if lx != rx:
            midx = (lx + rx)//2
            self.build_x(2*nodex, lx, midx)
            self.build_x(2*nodex+1, midx+1, rx)
        self.build_y(nodex, lx, rx, 1, 0, self.m-1)

    def query_y(self, nodex, nodey, ly, ry, qly, qry):
        if qry < ly or qly > ry: return 0
        if qly <= ly and ry <= qry:
            return self.tree[nodex][nodey]
        midy = (ly + ry)//2
        return self.merge(
            self.query_y(nodex, 2*nodey, ly, midy, qly, qry),
            self.query_y(nodex, 2*nodey+1, midy+1, ry, qly, qry)
        )

    def query_x(self, nodex, lx, rx, qlx, qrx, qly, qry):
        if qrx < lx or qlx > rx: return 0
        if qlx <= lx and rx <= qrx:
            return self.query_y(nodex, 1, 0, self.m-1, qly, qry)
        midx = (lx + rx)//2
        return self.merge(
            self.query_x(2*nodex, lx, midx, qlx, qrx, qly, qry),
            self.query_x(2*nodex+1, midx+1, rx, qlx, qrx, qly, qry)
        )

    def query(self, x1, y1, x2, y2):
        return self.query_x(1, 0, self.n-1, x1, x2, y1, y2)

# Example
A = [
    [2, 1, 3],
    [4, 5, 6],
    [7, 8, 9]
$$
seg2d = SegmentTree2D(A)
print(seg2d.query(0, 0, 1, 1))  # expect 12
```

#### Why It Matters

- Supports 2D queries with log² complexity
- Works for sum, min, max, gcd, XOR
- A foundation for advanced 2D data structures (Fenwick 2D, KD-trees, etc.)

#### A Gentle Proof (Why It Works)

Segment tree correctness relies on associativity.
In 2D, we extend this property across both dimensions.
Each node represents a rectangular region; merging children yields correct aggregate for parent.

#### Try It Yourself

1. Build from 3×3 matrix
2. Query `(0,0)-(2,2)` → full sum
3. Query `(1,1)-(2,2)` → bottom-right 5+6+8+9=28
4. Modify A[1][2]=10, rebuild, recheck sum

#### Test Cases

| Matrix                    | Query       | Result |
| ------------------------- | ----------- | ------ |
| [[2,1,3],[4,5,6],[7,8,9]] | (0,0)-(1,1) | 12     |
| [[2,1,3],[4,5,6],[7,8,9]] | (1,1)-(2,2) | 28     |
| [[1,2],[3,4]]             | (0,0)-(1,1) | 10     |

#### Complexity

| Operation | Time               | Space  |
| --------- | ------------------ | ------ |
| Build     | O(n·m·log n·log m) | O(n·m) |
| Query     | O(log² n)          | O(1)   |
| Update    | O(log² n)          | O(1)   |

A 2D segment tree is your grid-wise superpower, merging rectangles like puzzle pieces, one log at a time.

## Section 26. Disjoint Set Union 

### 251 Make-Set

The Make-Set operation is the starting point of the Disjoint Set Union (DSU), also known as Union-Find, a fundamental data structure for managing partitions of elements into disjoint sets.

Each element begins in its own set, serving as its own parent. Later operations (`Find` and `Union`) will merge and track relationships among them efficiently.

#### What Problem Are We Solving?

We need a way to represent a collection of disjoint sets, groups that don't overlap, while supporting these operations:

1. Make-Set(x): create a new set containing only `x`
2. Find(x): find representative (leader) of x's set
3. Union(x, y): merge sets containing x and y

This structure is the backbone of many graph algorithms like Kruskal's MST, connected components, and clustering.

#### How It Works (Plain Language)

Initially, each element is its own parent, a self-loop.
We store two arrays (or maps):

- `parent[x]` → points to x's parent (initially itself)
- `rank[x]` or `size[x]` → helps balance unions later

So:

```
parent[x] = x  
rank[x] = 0
```

Each element is an isolated tree. Over time, unions connect trees.

#### Example

Initialize `n = 5` elements: `{1, 2, 3, 4, 5}`

| x | parent[x] | rank[x] | Meaning    |
| - | --------- | ------- | ---------- |
| 1 | 1         | 0       | own leader |
| 2 | 2         | 0       | own leader |
| 3 | 3         | 0       | own leader |
| 4 | 4         | 0       | own leader |
| 5 | 5         | 0       | own leader |

After all `Make-Set`, each element is in its own group:

```
{1}, {2}, {3}, {4}, {5}
```

#### Visualization

```
1   2   3   4   5
↑   ↑   ↑   ↑   ↑
|   |   |   |   |
self self self self self
```

Each node points to itself, five separate trees.

#### Tiny Code (Easy Versions)

C Implementation

```c
#include <stdio.h>

#define MAXN 1000

int parent[MAXN];
int rank_[MAXN];

void make_set(int v) {
    parent[v] = v;   // self parent
    rank_[v] = 0;    // initial rank
}

int main(void) {
    int n = 5;
    for (int i = 1; i <= n; i++)
        make_set(i);
    
    printf("Initial sets:\n");
    for (int i = 1; i <= n; i++)
        printf("Element %d: parent=%d, rank=%d\n", i, parent[i], rank_[i]);
    return 0;
}
```

Python Implementation

```python
def make_set(x, parent, rank):
    parent[x] = x
    rank[x] = 0

n = 5
parent = {}
rank = {}

for i in range(1, n+1):
    make_set(i, parent, rank)

print("Parent:", parent)
print("Rank:", rank)
# Parent: {1:1, 2:2, 3:3, 4:4, 5:5}
```

#### Why It Matters

- The foundation of Disjoint Set Union
- Enables efficient graph algorithms
- A building block for Union by Rank and Path Compression
- Every DSU begins with `Make-Set`

#### A Gentle Proof (Why It Works)

Each element begins as a singleton.
Since no pointers cross between elements, sets are disjoint.
Subsequent `Union` operations preserve this property by merging trees, never duplicating nodes.

Thus, `Make-Set` guarantees:

- Each new node is independent
- Parent pointers form valid forests

#### Try It Yourself

1. Initialize `{1..5}` with `Make-Set`
2. Print parent and rank arrays
3. Add `Union(1,2)` and check parent changes
4. Verify all parent[x] = x before unions

#### Test Cases

| Input | Operation      | Expected Output    |
| ----- | -------------- | ------------------ |
| n=3   | Make-Set(1..3) | parent = [1,2,3]   |
| n=5   | Make-Set(1..5) | rank = [0,0,0,0,0] |

#### Complexity

| Operation | Time                     | Space |
| --------- | ------------------------ | ----- |
| Make-Set  | O(1)                     | O(1)  |
| Find      | O(α(n)) with compression | O(1)  |
| Union     | O(α(n)) with rank        | O(1)  |

The `Make-Set` step is your first move in the DSU dance, simple, constant time, and crucial for what follows.

### 252 Find

The Find operation is the heart of the Disjoint Set Union (DSU), also known as Union-Find. It locates the representative (leader) of the set containing a given element. Every element in the same set shares the same leader, this is how DSU identifies which elements belong together.

To make lookups efficient, `Find` uses a clever optimization called Path Compression, which flattens the structure of the tree so future queries become nearly constant time.

#### What Problem Are We Solving?

Given an element `x`, we want to determine which set it belongs to.
Each set is represented by a root node (the leader).

We maintain a `parent[]` array such that:

- `parent[x] = x` if `x` is the root (leader)
- otherwise `parent[x] = parent of x`

The `Find(x)` operation recursively follows `parent[x]` pointers until it reaches the root.

#### How It Works (Plain Language)

Think of each set as a tree, where the root is the representative.

For example:

```
1 ← 2 ← 3    4 ← 5
```

means `{1,2,3}` is one set, `{4,5}` is another.
The Find(3) operation follows `3→2→1`, discovering 1 is the root.

Path Compression flattens the tree by pointing every node directly to the root, reducing depth and speeding up future finds.

After compression:

```
1 ← 2   1 ← 3
4 ← 5
```

Now `Find(3)` is O(1).

#### Example

Start:

| x | parent[x] | rank[x] |
| - | --------- | ------- |
| 1 | 1         | 1       |
| 2 | 1         | 0       |
| 3 | 2         | 0       |
| 4 | 4         | 0       |
| 5 | 4         | 0       |

Perform `Find(3)`:

1. 3 → parent[3] = 2
2. 2 → parent[2] = 1
3. 1 → root found

Path compression rewires `parent[3] = 1`

Result:

| x | parent[x] |
| - | --------- |
| 1 | 1         |
| 2 | 1         |
| 3 | 1         |
| 4 | 4         |
| 5 | 4         |

#### Visualization

Before compression:

```
1  
↑  
2  
↑  
3
```

After compression:

```
1  
↑ ↑  
2 3
```

Now all nodes point directly to root `1`.

#### Tiny Code (Easy Versions)

C Implementation

```c
#include <stdio.h>

#define MAXN 1000
int parent[MAXN];
int rank_[MAXN];

void make_set(int v) {
    parent[v] = v;
    rank_[v] = 0;
}

int find_set(int v) {
    if (v == parent[v])
        return v;
    // Path compression
    parent[v] = find_set(parent[v]);
    return parent[v];
}

int main(void) {
    int n = 5;
    for (int i = 1; i <= n; i++) make_set(i);
    parent[2] = 1;
    parent[3] = 2;
    printf("Find(3) before: %d\n", parent[3]);
    printf("Root of 3: %d\n", find_set(3));
    printf("Find(3) after: %d\n", parent[3]);
}
```

Python Implementation

```python
def make_set(x, parent, rank):
    parent[x] = x
    rank[x] = 0

def find_set(x, parent):
    if parent[x] != x:
        parent[x] = find_set(parent[x], parent)  # Path compression
    return parent[x]

n = 5
parent = {}
rank = {}
for i in range(1, n+1):
    make_set(i, parent, rank)

parent[2] = 1
parent[3] = 2

print("Find(3) before:", parent)
print("Root of 3:", find_set(3, parent))
print("Find(3) after:", parent)
```

#### Why It Matters

- Enables O(α(n)) almost-constant-time queries
- Fundamental for Union-Find efficiency
- Reduces tree height dramatically
- Used in graph algorithms (Kruskal, connected components, etc.)

#### A Gentle Proof (Why It Works)

Every set is a rooted tree.
Without compression, a sequence of unions could build a deep chain.
With path compression, each `Find` flattens paths, ensuring amortized constant time per operation (Ackermann-inverse time).

So after repeated operations, DSU trees become very shallow.

#### Try It Yourself

1. Initialize `{1..5}`
2. Set `parent[2]=1, parent[3]=2`
3. Call `Find(3)` and observe compression
4. Check `parent[3] == 1` after

#### Test Cases

| parent before | Find(x) | parent after |
| ------------- | ------- | ------------ |
| [1,1,2]       | Find(3) | [1,1,1]      |
| [1,2,3,4,5]   | Find(5) | [1,2,3,4,5]  |
| [1,1,1,1,1]   | Find(3) | [1,1,1,1,1]  |

#### Complexity

| Operation | Time (Amortized) | Space |
| --------- | ---------------- | ----- |
| Find      | O(α(n))          | O(1)  |
| Make-Set  | O(1)             | O(1)  |
| Union     | O(α(n))          | O(1)  |

The Find operation is your compass, it always leads you to the leader of your set, and with path compression, it gets faster every step.

### 253 Union

The Union operation is what ties the Disjoint Set Union (DSU) together. After initializing sets with `Make-Set` and locating leaders with `Find`, `Union` merges two disjoint sets into one, a cornerstone of dynamic connectivity.

To keep the trees shallow and efficient, we combine it with heuristics like Union by Rank or Union by Size.

#### What Problem Are We Solving?

We want to merge the sets containing two elements `a` and `b`.

If `a` and `b` belong to different sets, their leaders (`Find(a)` and `Find(b)`) are different.
The Union operation connects these leaders, ensuring both now share the same representative.

We must do it efficiently, no unnecessary tree height. That's where Union by Rank comes in.

#### How It Works (Plain Language)

1. Find roots of both elements:

   ```
   rootA = Find(a)
   rootB = Find(b)
   ```
2. If they're already equal → same set, no action.
3. Otherwise, attach the shorter tree under the taller one:

   * If `rank[rootA] < rank[rootB]`: `parent[rootA] = rootB`
   * Else if `rank[rootA] > rank[rootB]`: `parent[rootB] = rootA`
   * Else: `parent[rootB] = rootA`, and `rank[rootA]++`

This keeps the resulting forest balanced, maintaining O(α(n)) time for operations.

#### Example

Initial sets: `{1}, {2}, {3}, {4}`

| x | parent[x] | rank[x] |
| - | --------- | ------- |
| 1 | 1         | 0       |
| 2 | 2         | 0       |
| 3 | 3         | 0       |
| 4 | 4         | 0       |

Perform `Union(1,2)` → attach 2 under 1

```
parent[2] = 1  
rank[1] = 1
```

Now sets: `{1,2}, {3}, {4}`

Perform `Union(3,4)` → attach 4 under 3
Sets: `{1,2}, {3,4}`

Then `Union(2,3)` → merge leaders (1 and 3)
Attach lower rank under higher (both rank 1 → tie)
→ attach 3 under 1, rank[1] = 2

Final parent table:

| x | parent[x] | rank[x] |
| - | --------- | ------- |
| 1 | 1         | 2       |
| 2 | 1         | 0       |
| 3 | 1         | 1       |
| 4 | 3         | 0       |

Now all are connected under root 1. ✅

#### Visualization

Start:

```
1   2   3   4
```

Union(1,2):

```
  1
  |
  2
```

Union(3,4):

```
  3
  |
  4
```

Union(2,3):

```
    1
  / | \
 2  3  4
```

All connected under 1.

#### Tiny Code (Easy Versions)

C Implementation

```c
#include <stdio.h>

#define MAXN 1000
int parent[MAXN];
int rank_[MAXN];

void make_set(int v) {
    parent[v] = v;
    rank_[v] = 0;
}

int find_set(int v) {
    if (v == parent[v]) return v;
    return parent[v] = find_set(parent[v]); // path compression
}

void union_sets(int a, int b) {
    a = find_set(a);
    b = find_set(b);
    if (a != b) {
        if (rank_[a] < rank_[b])
            parent[a] = b;
        else if (rank_[a] > rank_[b])
            parent[b] = a;
        else {
            parent[b] = a;
            rank_[a]++;
        }
    }
}

int main(void) {
    for (int i=1;i<=4;i++) make_set(i);
    union_sets(1,2);
    union_sets(3,4);
    union_sets(2,3);
    for (int i=1;i<=4;i++)
        printf("Element %d: parent=%d, rank=%d\n", i, parent[i], rank_[i]);
}
```

Python Implementation

```python
def make_set(x, parent, rank):
    parent[x] = x
    rank[x] = 0

def find_set(x, parent):
    if parent[x] != x:
        parent[x] = find_set(parent[x], parent)
    return parent[x]

def union_sets(a, b, parent, rank):
    a = find_set(a, parent)
    b = find_set(b, parent)
    if a != b:
        if rank[a] < rank[b]:
            parent[a] = b
        elif rank[a] > rank[b]:
            parent[b] = a
        else:
            parent[b] = a
            rank[a] += 1

n = 4
parent = {}
rank = {}
for i in range(1, n+1): make_set(i, parent, rank)
union_sets(1,2,parent,rank)
union_sets(3,4,parent,rank)
union_sets(2,3,parent,rank)
print("Parent:", parent)
print("Rank:", rank)
```

#### Why It Matters

- Core operation in Union-Find
- Enables dynamic set merging efficiently
- Essential for graph algorithms:

  * Kruskal's Minimum Spanning Tree
  * Connected Components
  * Cycle Detection

Combining `Union by Rank` and `Path Compression` yields near-constant performance for huge datasets.

#### A Gentle Proof (Why It Works)

`Union` ensures disjointness:

- Only merges sets with different roots
- Maintains one leader per set
  `Union by Rank` keeps trees balanced → tree height ≤ log n
  With path compression, amortized cost per operation is α(n), effectively constant for all practical n.

#### Try It Yourself

1. Create 5 singleton sets
2. Union(1,2), Union(3,4), Union(2,3)
3. Verify all share same root
4. Inspect ranks before/after unions

#### Test Cases

| Operations     | Expected Sets   | Roots     |
| -------------- | --------------- | --------- |
| Make-Set(1..4) | {1},{2},{3},{4} | [1,2,3,4] |
| Union(1,2)     | {1,2}           | [1,1,3,4] |
| Union(3,4)     | {3,4}           | [1,1,3,3] |
| Union(2,3)     | {1,2,3,4}       | [1,1,1,1] |

#### Complexity

| Operation | Time (Amortized) | Space |
| --------- | ---------------- | ----- |
| Union     | O(α(n))          | O(1)  |
| Find      | O(α(n))          | O(1)  |
| Make-Set  | O(1)             | O(1)  |

`Union` is the handshake between sets, careful, balanced, and lightning-fast when combined with smart optimizations.

### 254 Union by Rank

Union by Rank is a balancing strategy used in the Disjoint Set Union (DSU) structure to keep trees shallow. When merging two sets, instead of arbitrarily attaching one root under another, we attach the shorter tree (lower rank) under the taller tree (higher rank).

This small trick, combined with Path Compression, gives DSU its legendary near-constant performance, almost O(1) per operation.

#### What Problem Are We Solving?

Without balancing, repeated `Union` operations might create tall chains like:

```
1 ← 2 ← 3 ← 4 ← 5
```

In the worst case, `Find(x)` becomes O(n).

Union by Rank prevents this by maintaining a rough measure of tree height.
Each `rank[root]` tracks the *approximate height* of its tree.

When merging:

- Attach the lower rank tree under the higher rank tree.
- If ranks are equal, choose one as parent and increase its rank by 1.

#### How It Works (Plain Language)

Each node `x` has:

- `parent[x]` → leader pointer
- `rank[x]` → estimate of tree height

When performing `Union(a, b)`:

1. `rootA = Find(a)`
2. `rootB = Find(b)`
3. If ranks differ, attach smaller under larger:

   ```
   if rank[rootA] < rank[rootB]:
       parent[rootA] = rootB
   else if rank[rootA] > rank[rootB]:
       parent[rootB] = rootA
   else:
       parent[rootB] = rootA
       rank[rootA] += 1
   ```

This ensures the tree grows only when necessary.

#### Example

Start with `{1}, {2}, {3}, {4}`
All have `rank = 0`

Perform `Union(1,2)` → same rank → attach 2 under 1, increment rank[1]=1

```
1
|
2
```

Perform `Union(3,4)` → same rank → attach 4 under 3, increment rank[3]=1

```
3
|
4
```

Now Union(1,3) → both roots rank=1 → tie → attach 3 under 1, rank[1]=2

```
   1(rank=2)
  / \
 2   3
     |
     4
```

Tree height stays small, well-balanced.

| Element | Parent | Rank |
| ------- | ------ | ---- |
| 1       | 1      | 2    |
| 2       | 1      | 0    |
| 3       | 1      | 1    |
| 4       | 3      | 0    |

#### Visualization

Before balancing:

```
1 ← 2 ← 3 ← 4
```

After union by rank:

```
    1
   / \
  2   3
      |
      4
```

Balanced and efficient ✅

#### Tiny Code (Easy Versions)

C Implementation

```c
#include <stdio.h>

#define MAXN 1000
int parent[MAXN];
int rank_[MAXN];

void make_set(int v) {
    parent[v] = v;
    rank_[v] = 0;
}

int find_set(int v) {
    if (v == parent[v]) return v;
    return parent[v] = find_set(parent[v]); // Path compression
}

void union_by_rank(int a, int b) {
    a = find_set(a);
    b = find_set(b);
    if (a != b) {
        if (rank_[a] < rank_[b]) parent[a] = b;
        else if (rank_[a] > rank_[b]) parent[b] = a;
        else {
            parent[b] = a;
            rank_[a]++;
        }
    }
}

int main(void) {
    for (int i=1;i<=4;i++) make_set(i);
    union_by_rank(1,2);
    union_by_rank(3,4);
    union_by_rank(2,3);
    for (int i=1;i<=4;i++)
        printf("Element %d: parent=%d rank=%d\n", i, parent[i], rank_[i]);
}
```

Python Implementation

```python
def make_set(x, parent, rank):
    parent[x] = x
    rank[x] = 0

def find_set(x, parent):
    if parent[x] != x:
        parent[x] = find_set(parent[x], parent)
    return parent[x]

def union_by_rank(a, b, parent, rank):
    a = find_set(a, parent)
    b = find_set(b, parent)
    if a != b:
        if rank[a] < rank[b]:
            parent[a] = b
        elif rank[a] > rank[b]:
            parent[b] = a
        else:
            parent[b] = a
            rank[a] += 1

n = 4
parent = {}
rank = {}
for i in range(1, n+1): make_set(i, parent, rank)
union_by_rank(1,2,parent,rank)
union_by_rank(3,4,parent,rank)
union_by_rank(2,3,parent,rank)
print("Parent:", parent)
print("Rank:", rank)
```

#### Why It Matters

- Keeps trees balanced and shallow
- Ensures amortized O(α(n)) performance
- Critical for massive-scale connectivity problems
- Used in Kruskal's MST, Union-Find with rollback, and network clustering

#### A Gentle Proof (Why It Works)

The rank grows only when two trees of equal height merge.
Thus, the height of any tree is bounded by log₂ n.
Combining this with Path Compression, the amortized complexity per operation becomes O(α(n)), where α(n) (inverse Ackermann function) < 5 for all practical n.

#### Try It Yourself

1. Create 8 singleton sets
2. Perform unions in sequence: (1,2), (3,4), (1,3), (5,6), (7,8), (5,7), (1,5)
3. Observe ranks and parent structure

#### Test Cases

| Operation  | Result (Parent Array) | Rank      |
| ---------- | --------------------- | --------- |
| Union(1,2) | [1,1,3,4]             | [1,0,0,0] |
| Union(3,4) | [1,1,3,3]             | [1,0,1,0] |
| Union(2,3) | [1,1,1,3]             | [2,0,1,0] |

#### Complexity

| Operation     | Time (Amortized) | Space |
| ------------- | ---------------- | ----- |
| Union by Rank | O(α(n))          | O(1)  |
| Find          | O(α(n))          | O(1)  |
| Make-Set      | O(1)             | O(1)  |

Union by Rank is the art of merging gracefully, always lifting smaller trees, keeping your forest light, flat, and fast.

### 255 Path Compression

Path Compression is the secret sauce that makes Disjoint Set Union (DSU) lightning fast. Every time you perform a `Find` operation, it *flattens* the structure by making each visited node point directly to the root. Over time, this transforms deep trees into almost flat structures, turning expensive lookups into near-constant time.

#### What Problem Are We Solving?

In a basic DSU, `Find(x)` follows parent pointers up the tree until it reaches the root. Without compression, frequent unions can form long chains:

```
1 ← 2 ← 3 ← 4 ← 5
```

A `Find(5)` would take 5 steps. Multiply this over many queries, and performance tanks.

Path Compression fixes this inefficiency by *rewiring* all nodes on the search path to the root directly, effectively flattening the tree.

#### How It Works (Plain Language)

Whenever we call `Find(x)`, we recursively find the root, then make every node along the way point directly to that root.

Pseudocode:

```
Find(x):
    if parent[x] != x:
        parent[x] = Find(parent[x])
    return parent[x]
```

Now, future lookups for `x` and its descendants become instant.

#### Example

Start with a chain:

```
1 ← 2 ← 3 ← 4 ← 5
```

Perform `Find(5)`:

1. `Find(5)` calls `Find(4)`
2. `Find(4)` calls `Find(3)`
3. `Find(3)` calls `Find(2)`
4. `Find(2)` calls `Find(1)` (root)
5. On the way back, each node gets updated:

   ```
   parent[5] = 1
   parent[4] = 1
   parent[3] = 1
   parent[2] = 1
   ```

After compression, structure becomes flat:

```
1
├── 2
├── 3
├── 4
└── 5
```

| Element | Parent Before | Parent After |
| ------- | ------------- | ------------ |
| 1       | 1             | 1            |
| 2       | 1             | 1            |
| 3       | 2             | 1            |
| 4       | 3             | 1            |
| 5       | 4             | 1            |

Next call `Find(5)` → single step.

#### Visualization

Before compression:

```
1 ← 2 ← 3 ← 4 ← 5
```

After compression:

```
1
├─2
├─3
├─4
└─5
```

#### Tiny Code (Easy Versions)

C Implementation

```c
#include <stdio.h>

#define MAXN 100
int parent[MAXN];

void make_set(int v) {
    parent[v] = v;
}

int find_set(int v) {
    if (v != parent[v])
        parent[v] = find_set(parent[v]); // Path compression
    return parent[v];
}

void union_sets(int a, int b) {
    a = find_set(a);
    b = find_set(b);
    if (a != b)
        parent[b] = a;
}

int main(void) {
    for (int i = 1; i <= 5; i++) make_set(i);
    union_sets(1, 2);
    union_sets(2, 3);
    union_sets(3, 4);
    union_sets(4, 5);
    find_set(5); // compress path
    for (int i = 1; i <= 5; i++)
        printf("Element %d → Parent %d\n", i, parent[i]);
}
```

Python Implementation

```python
def make_set(x, parent):
    parent[x] = x

def find_set(x, parent):
    if parent[x] != x:
        parent[x] = find_set(parent[x], parent)
    return parent[x]

def union_sets(a, b, parent):
    a = find_set(a, parent)
    b = find_set(b, parent)
    if a != b:
        parent[b] = a

parent = {}
for i in range(1, 6):
    make_set(i, parent)
union_sets(1, 2, parent)
union_sets(2, 3, parent)
union_sets(3, 4, parent)
union_sets(4, 5, parent)
find_set(5, parent)
print("Parent map after compression:", parent)
```

#### Why It Matters

- Speeds up Find drastically by flattening trees.
- Pairs beautifully with Union by Rank.
- Achieves amortized O(α(n)) performance.
- Essential in graph algorithms like Kruskal's MST, connectivity checks, and dynamic clustering.

#### A Gentle Proof (Why It Works)

Path Compression ensures each node's parent jumps directly to the root.
Each node's depth decreases exponentially with every `Find`.
After a few operations, trees become almost flat, and each subsequent `Find` becomes O(1).

Combining with Union by Rank:

> Every operation (Find or Union) becomes O(α(n)),
> where α(n) is the inverse Ackermann function (smaller than 5 for any practical input).

#### Try It Yourself

1. Create 6 singleton sets.
2. Perform unions: (1,2), (2,3), (3,4), (4,5), (5,6).
3. Call `Find(6)` and print parent map before and after.
4. Observe how the chain flattens.
5. Measure calls count difference before and after compression.

#### Test Cases

| Operation Sequence | Parent Map After            | Tree Depth |
| ------------------ | --------------------------- | ---------- |
| No Compression     | `{1:1, 2:1, 3:2, 4:3, 5:4}` | 4          |
| After Find(5)      | `{1:1, 2:1, 3:1, 4:1, 5:1}` | 1          |
| After Find(3)      | `{1:1, 2:1, 3:1, 4:1, 5:1}` | 1          |

#### Complexity

| Operation                  | Time (Amortized) | Space |
| -------------------------- | ---------------- | ----- |
| Find with Path Compression | O(α(n))          | O(1)  |
| Union (with rank)          | O(α(n))          | O(1)  |
| Make-Set                   | O(1)             | O(1)  |

Path Compression is the flattening spell of DSU, once cast, your sets become sleek, your lookups swift, and your unions unstoppable.

### 256 DSU with Rollback

DSU with Rollback extends the classic Disjoint Set Union to support *undoing* recent operations. This is vital in scenarios where you need to explore multiple states, like backtracking algorithms, dynamic connectivity queries, or offline problems where unions might need to be reversed.

Instead of destroying past states, this version *remembers* what changed and can roll back to a previous version in constant time.

#### What Problem Are We Solving?

Standard DSU operations (`Find`, `Union`) mutate the structure, parent pointers and ranks get updated, so you can't easily go back.

But what if you're exploring a search tree or processing offline queries where you need to:

- Add an edge (Union),
- Explore a path,
- Then revert to the previous structure?

A rollback DSU lets you undo changes, perfect for divide-and-conquer over time, Mo's algorithm on trees, and offline dynamic graphs.

#### How It Works (Plain Language)

The idea is simple:

- Every time you modify the DSU, record what you changed on a stack.
- When you need to revert, pop from the stack and undo the last operation.

Operations to track:

- When `parent[b]` changes, store `(b, old_parent)`
- When `rank[a]` changes, store `(a, old_rank)`

You never perform path compression, because it's not easily reversible. Instead, rely on union by rank for efficiency.

#### Example

Let's build sets `{1}, {2}, {3}, {4}`

Perform:

1. `Union(1,2)` → attach 2 under 1

   * Push `(2, parent=2, rank=None)`
2. `Union(3,4)` → attach 4 under 3

   * Push `(4, parent=4, rank=None)`
3. `Union(1,3)` → attach 3 under 1

   * Push `(3, parent=3, rank=None)`
   * Rank of 1 increases → push `(1, rank=0)`

Rollback once → undo last union:

- Restore `parent[3] = 3`
- Restore `rank[1] = 0`

Now DSU returns to state after step 2.

| Step             | Parent Map | Rank      | Stack                                         |
| ---------------- | ---------- | --------- | --------------------------------------------- |
| Init             | [1,2,3,4]  | [0,0,0,0] | []                                            |
| After Union(1,2) | [1,1,3,4]  | [1,0,0,0] | [(2,2,None)]                                  |
| After Union(3,4) | [1,1,3,3]  | [1,0,1,0] | [(2,2,None),(4,4,None)]                       |
| After Union(1,3) | [1,1,1,3]  | [2,0,1,0] | [(2,2,None),(4,4,None),(3,3,None),(1,None,0)] |
| After Rollback   | [1,1,3,3]  | [1,0,1,0] | [(2,2,None),(4,4,None)]                       |

#### Tiny Code (Easy Versions)

C Implementation (Conceptual)

```c
#include <stdio.h>

#define MAXN 1000
int parent[MAXN], rank_[MAXN];
typedef struct { int node, parent, rank_val, rank_changed; } Change;
Change stack[MAXN * 10];
int top = 0;

void make_set(int v) {
    parent[v] = v;
    rank_[v] = 0;
}

int find_set(int v) {
    while (v != parent[v]) v = parent[v];
    return v; // no path compression
}

void union_sets(int a, int b) {
    a = find_set(a);
    b = find_set(b);
    if (a == b) return;
    if (rank_[a] < rank_[b]) { int tmp = a; a = b; b = tmp; }
    stack[top++] = (Change){b, parent[b], 0, 0};
    parent[b] = a;
    if (rank_[a] == rank_[b]) {
        stack[top++] = (Change){a, 0, rank_[a], 1};
        rank_[a]++;
    }
}

void rollback() {
    if (top == 0) return;
    Change ch = stack[--top];
    if (ch.rank_changed) rank_[ch.node] = ch.rank_val;
    else parent[ch.node] = ch.parent;
}

int main() {
    for (int i=1;i<=4;i++) make_set(i);
    union_sets(1,2);
    union_sets(3,4);
    union_sets(1,3);
    printf("Before rollback: parent[3]=%d\n", parent[3]);
    rollback();
    printf("After rollback: parent[3]=%d\n", parent[3]);
}
```

Python Implementation

```python
class RollbackDSU:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0]*n
        self.stack = []

    def find(self, x):
        while x != self.parent[x]:
            x = self.parent[x]
        return x  # no path compression

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a == b:
            return False
        if self.rank[a] < self.rank[b]:
            a, b = b, a
        self.stack.append(('p', b, self.parent[b]))
        self.parent[b] = a
        if self.rank[a] == self.rank[b]:
            self.stack.append(('r', a, self.rank[a]))
            self.rank[a] += 1
        return True

    def rollback(self):
        if not self.stack:
            return
        typ, node, val = self.stack.pop()
        if typ == 'r':
            self.rank[node] = val
        else:
            self.parent[node] = val

dsu = RollbackDSU(5)
dsu.union(1,2)
dsu.union(3,4)
dsu.union(1,3)
print("Before rollback:", dsu.parent)
dsu.rollback()
print("After rollback:", dsu.parent)
```

#### Why It Matters

- Enables reversible union operations
- Perfect for offline dynamic connectivity
- Core in divide-and-conquer over time algorithms
- Used in Mo's algorithm on trees
- Helps in exploration backtracking (e.g., DSU on recursion)

#### A Gentle Proof (Why It Works)

Rollback DSU is efficient because:

- Each union modifies O(1) fields
- Each rollback reverts O(1) fields
- `Find` runs in O(log n) (no path compression)
  Thus, each operation is O(log n) or better, fully reversible.

#### Try It Yourself

1. Create 6 sets `{1}..{6}`
2. Perform unions: (1,2), (3,4), (2,3)
3. Rollback once, verify sets `{1,2}` and `{3,4}` remain separate
4. Rollback again, check individual sets restored
5. Print parent and rank at each step

#### Test Cases

| Step | Operation  | Parents   | Ranks     | Stack Size |
| ---- | ---------- | --------- | --------- | ---------- |
| 1    | Union(1,2) | [1,1,3,4] | [1,0,0,0] | 1          |
| 2    | Union(3,4) | [1,1,3,3] | [1,0,1,0] | 2          |
| 3    | Union(1,3) | [1,1,1,3] | [2,0,1,0] | 4          |
| 4    | Rollback   | [1,1,3,3] | [1,0,1,0] | 2          |

#### Complexity

| Operation | Time     | Space |
| --------- | -------- | ----- |
| Make-Set  | O(1)     | O(n)  |
| Union     | O(log n) | O(1)  |
| Rollback  | O(1)     | O(1)  |

DSU with Rollback is your time machine, merge, explore, undo. Perfect balance between persistence and performance.

### 257 DSU on Tree

DSU on Tree is a hybrid technique combining Disjoint Set Union (DSU) and Depth-First Search (DFS) to process subtree queries efficiently. It's often called the "Small-to-Large" merging technique, and it shines in problems where you need to answer queries like:

> "For each node, count something inside its subtree."

Instead of recomputing from scratch at every node, we use DSU logic to reuse computed data, merging smaller subtrees into larger ones for near-linear complexity.

#### What Problem Are We Solving?

Many tree problems ask for aggregated properties of subtrees:

- Number of distinct colors in a subtree
- Frequency of labels, values, or weights
- Subtree sums, counts, or modes

A naive DFS recomputes at every node → O(n²) time.

DSU on Tree avoids recomputation by merging information from child subtrees cleverly.

#### How It Works (Plain Language)

Think of each node's subtree as a bag of information (like a multiset of colors).
We process subtrees in DFS order, and at each step:

1. Process all *small children* first and discard their data after use.
2. Process the *heavy child* last and keep its data (reuse it).
3. Merge small subtrees into the large one incrementally.

This "small-to-large" merging ensures each element moves O(log n) times, leading to O(n log n) total complexity.

#### Example

Suppose we have a tree:

```
     1
   / | \
  2  3  4
 / \    \
5   6    7
```

Each node has a color:

```
Color = [1, 2, 2, 1, 3, 3, 2]
```

Goal: For every node, count distinct colors in its subtree.

Naive way: recompute every subtree from scratch, O(n²).
DSU-on-Tree way: reuse large child's color set, merge small ones.

| Node | Subtree         | Colors  | Distinct Count |
| ---- | --------------- | ------- | -------------- |
| 5    | [5]             | {3}     | 1              |
| 6    | [6]             | {3}     | 1              |
| 2    | [2,5,6]         | {2,3}   | 2              |
| 3    | [3]             | {2}     | 1              |
| 7    | [7]             | {2}     | 1              |
| 4    | [4,7]           | {1,2}   | 2              |
| 1    | [1,2,3,4,5,6,7] | {1,2,3} | 3              |

Each subtree merges small color sets into the big one only once → efficient.

#### Step-by-Step Idea

1. DFS to compute subtree sizes
2. Identify heavy child (largest subtree)
3. DFS again:

   * Process all *light children* (small subtrees), discarding results
   * Process *heavy child*, keep its results
   * Merge all light children's data into heavy child's data
4. Record answers for each node after merging

#### Tiny Code (Easy Versions)

C Implementation (Conceptual)

```c
#include <stdio.h>
#include <vector>
#include <set>

#define MAXN 100005
using namespace std;

vector<int> tree[MAXN];
int color[MAXN];
int subtree_size[MAXN];
int answer[MAXN];
int freq[MAXN];
int n;

void dfs_size(int u, int p) {
    subtree_size[u] = 1;
    for (int v : tree[u])
        if (v != p) {
            dfs_size(v, u);
            subtree_size[u] += subtree_size[v];
        }
}

void add_color(int u, int p, int val) {
    freq[color[u]] += val;
    for (int v : tree[u])
        if (v != p) add_color(v, u, val);
}

void dfs(int u, int p, bool keep) {
    int bigChild = -1, maxSize = -1;
    for (int v : tree[u])
        if (v != p && subtree_size[v] > maxSize)
            maxSize = subtree_size[v], bigChild = v;

    // Process small children
    for (int v : tree[u])
        if (v != p && v != bigChild)
            dfs(v, u, false);

    // Process big child
    if (bigChild != -1) dfs(bigChild, u, true);

    // Merge small children's info
    for (int v : tree[u])
        if (v != p && v != bigChild)
            add_color(v, u, +1);

    freq[color[u]]++;
    // Example query: count distinct colors
    answer[u] = 0;
    for (int i = 1; i <= n; i++)
        if (freq[i] > 0) answer[u]++;

    if (!keep) add_color(u, p, -1);
}

int main() {
    n = 7;
    // Build tree, set colors...
    // dfs_size(1,0); dfs(1,0,true);
}
```

Python Implementation (Simplified)

```python
from collections import defaultdict

def dfs_size(u, p, tree, size):
    size[u] = 1
    for v in tree[u]:
        if v != p:
            dfs_size(v, u, tree, size)
            size[u] += size[v]

def add_color(u, p, tree, color, freq, val):
    freq[color[u]] += val
    for v in tree[u]:
        if v != p:
            add_color(v, u, tree, color, freq, val)

def dfs(u, p, tree, size, color, freq, ans, keep):
    bigChild, maxSize = -1, -1
    for v in tree[u]:
        if v != p and size[v] > maxSize:
            maxSize, bigChild = size[v], v

    for v in tree[u]:
        if v != p and v != bigChild:
            dfs(v, u, tree, size, color, freq, ans, False)

    if bigChild != -1:
        dfs(bigChild, u, tree, size, color, freq, ans, True)

    for v in tree[u]:
        if v != p and v != bigChild:
            add_color(v, u, tree, color, freq, 1)

    freq[color[u]] += 1
    ans[u] = len([c for c in freq if freq[c] > 0])

    if not keep:
        add_color(u, p, tree, color, freq, -1)

# Example usage
n = 7
tree = {1:[2,3,4], 2:[1,5,6], 3:[1], 4:[1,7], 5:[2], 6:[2], 7:[4]}
color = {1:1,2:2,3:2,4:1,5:3,6:3,7:2}
size = {}
ans = {}
freq = defaultdict(int)
dfs_size(1,0,tree,size)
dfs(1,0,tree,size,color,freq,ans,True)
print(ans)
```

#### Why It Matters

- Handles subtree queries in O(n log n)
- Works well with static trees and offline queries
- Reuses computed results for heavy subtrees
- Foundation for Mo's algorithm on trees and color frequency problems

#### A Gentle Proof (Why It Works)

Each node's color (or element) is merged only O(log n) times:

- Each time, it moves from a smaller set to a larger one.
- Hence total merges = O(n log n).

The `keep` flag ensures we only retain big subtrees, discarding light ones to save memory and time.

#### Try It Yourself

1. Assign random colors to a tree of 8 nodes.
2. Use DSU on Tree to count distinct colors per subtree.
3. Compare with brute-force DFS result.
4. Verify both match, but DSU-on-Tree runs faster.

#### Test Cases

| Node | Subtree Colors | Distinct Count |
| ---- | -------------- | -------------- |
| 5    | {3}            | 1              |
| 6    | {3}            | 1              |
| 2    | {2,3}          | 2              |
| 3    | {2}            | 1              |
| 7    | {2}            | 1              |
| 4    | {1,2}          | 2              |
| 1    | {1,2,3}        | 3              |

#### Complexity

| Operation     | Time       | Space |
| ------------- | ---------- | ----- |
| DFS Traversal | O(n)       | O(n)  |
| DSU Merge     | O(n log n) | O(n)  |
| Overall       | O(n log n) | O(n)  |

DSU on Tree is your subtree superpower, merge smart, discard light, and conquer queries with elegance.

### 258 Kruskal's MST (Using DSU)

Kruskal's Algorithm is a classic greedy method to build a Minimum Spanning Tree (MST), a subset of edges connecting all vertices with the smallest total weight and no cycles.
It relies on Disjoint Set Union (DSU) to efficiently check whether adding an edge would form a cycle.

With Union-Find, Kruskal's MST becomes clean, fast, and conceptually elegant, building the tree edge by edge in sorted order.

#### What Problem Are We Solving?

Given a connected weighted graph with $n$ vertices and $m$ edges, we want to find a tree that:

- Connects all vertices (spanning)  
- Has no cycles (tree)  
- Minimizes the total edge weight

A naive approach would test every subset of edges, $O(2^m)$.  
Kruskal’s algorithm uses edge sorting and a Disjoint Set Union (DSU) structure to reduce this to $O(m \log m)$.


#### How It Works (Plain Language)

The algorithm follows three steps:

1. Sort all edges by weight in ascending order.  
2. Initialize each vertex as its own set (`Make-Set`).  
3. For each edge $(u, v, w)$ in order:  
   - If $\text{Find}(u) \ne \text{Find}(v)$, the vertices are in different components → add the edge to the MST and merge the sets using `Union`.  
   - Otherwise, skip the edge since it would form a cycle.

Repeat until the MST contains $n - 1$ edges.


#### Example

Graph:

| Edge | Weight |
| ---- | ------ |
| A–B  | 1      |
| B–C  | 4      |
| A–C  | 3      |
| C–D  | 2      |

Sort edges: (A–B, 1), (C–D, 2), (A–C, 3), (B–C, 4)

Step-by-step:

| Step | Edge    | Action       | MST Edges             | MST Weight | Parent Map         |
| ---- | ------- | ------------ | --------------------- | ---------- | ------------------ |
| 1    | (A,B,1) | Add          | [(A,B)]               | 1          | A→A, B→A           |
| 2    | (C,D,2) | Add          | [(A,B), (C,D)]        | 3          | C→C, D→C           |
| 3    | (A,C,3) | Add          | [(A,B), (C,D), (A,C)] | 6          | A→A, B→A, C→A, D→C |
| 4    | (B,C,4) | Skip (cycle) | –                     | –          | –                  |

✅ MST Weight = 6
✅ Edges = 3 = n − 1

#### Visualization

Before:

```
A --1-- B
|      /
3    4
|  /
C --2-- D
```

After:

```
A
| \
1  3
B   C
    |
    2
    D
```

#### Tiny Code (Easy Versions)

C Implementation

```c
#include <stdio.h>
#include <stdlib.h>

#define MAXN 100
#define MAXM 1000

typedef struct {
    int u, v, w;
} Edge;

int parent[MAXN], rank_[MAXN];
Edge edges[MAXM];

int cmp(const void *a, const void *b) {
    return ((Edge*)a)->w - ((Edge*)b)->w;
}

void make_set(int v) {
    parent[v] = v;
    rank_[v] = 0;
}

int find_set(int v) {
    if (v != parent[v]) parent[v] = find_set(parent[v]);
    return parent[v];
}

void union_sets(int a, int b) {
    a = find_set(a);
    b = find_set(b);
    if (a != b) {
        if (rank_[a] < rank_[b]) parent[a] = b;
        else if (rank_[a] > rank_[b]) parent[b] = a;
        else { parent[b] = a; rank_[a]++; }
    }
}

int main() {
    int n = 4, m = 4;
    edges[0] = (Edge){0,1,1};
    edges[1] = (Edge){1,2,4};
    edges[2] = (Edge){0,2,3};
    edges[3] = (Edge){2,3,2};
    qsort(edges, m, sizeof(Edge), cmp);

    for (int i = 0; i < n; i++) make_set(i);

    int total = 0;
    printf("Edges in MST:\n");
    for (int i = 0; i < m; i++) {
        int u = edges[i].u, v = edges[i].v, w = edges[i].w;
        if (find_set(u) != find_set(v)) {
            union_sets(u, v);
            total += w;
            printf("%d - %d (w=%d)\n", u, v, w);
        }
    }
    printf("Total Weight = %d\n", total);
}
```

Python Implementation

```python
def make_set(parent, rank, v):
    parent[v] = v
    rank[v] = 0

def find_set(parent, v):
    if parent[v] != v:
        parent[v] = find_set(parent, parent[v])
    return parent[v]

def union_sets(parent, rank, a, b):
    a, b = find_set(parent, a), find_set(parent, b)
    if a != b:
        if rank[a] < rank[b]:
            a, b = b, a
        parent[b] = a
        if rank[a] == rank[b]:
            rank[a] += 1

def kruskal(n, edges):
    parent, rank = {}, {}
    for i in range(n):
        make_set(parent, rank, i)
    mst, total = [], 0
    for u, v, w in sorted(edges, key=lambda e: e[2]):
        if find_set(parent, u) != find_set(parent, v):
            union_sets(parent, rank, u, v)
            mst.append((u, v, w))
            total += w
    return mst, total

edges = [(0,1,1),(1,2,4),(0,2,3),(2,3,2)]
mst, total = kruskal(4, edges)
print("MST:", mst)
print("Total Weight:", total)
```

#### Why It Matters

- Greedy and elegant: simple sorting + DSU logic
- Foundation for advanced topics:

  * Minimum spanning forests
  * Dynamic connectivity
  * MST variations (Maximum, Second-best, etc.)
- Works great with edge list input

#### A Gentle Proof (Why It Works)

By the Cut Property:
The smallest edge crossing any cut belongs to the MST.
Since Kruskal's always picks the smallest non-cycling edge, it constructs a valid MST.

Each union merges components without cycles → spanning tree in the end.

#### Try It Yourself

1. Build a graph with 5 nodes, random edges and weights
2. Sort edges, trace unions step-by-step
3. Draw MST
4. Compare with Prim's algorithm result, they'll match

#### Test Cases

| Graph                             | MST Edges        | Weight |
| --------------------------------- | ---------------- | ------ |
| Triangle (1–2:1, 2–3:2, 1–3:3)    | (1–2, 2–3)       | 3      |
| Square (4 edges, weights 1,2,3,4) | 3 smallest edges | 6      |

#### Complexity

| Step           | Time       |
| -------------- | ---------- |
| Sorting Edges  | O(m log m) |
| DSU Operations | O(m α(n))  |
| Total          | O(m log m) |

| Space | O(n + m) |

Kruskal's MST is the elegant handshake between greed and union, always connecting lightly, never circling back.

### 259 Connected Components (Using DSU)

Connected Components are groups of vertices where each node can reach any other through a sequence of edges. Using Disjoint Set Union (DSU), we can efficiently identify and label these components in graphs, even for massive datasets.

Instead of exploring each region via DFS or BFS, DSU builds the connectivity relationships incrementally, merging nodes as edges appear.

#### What Problem Are We Solving?

Given a graph (directed or undirected), we want to answer:

- How many connected components exist?
- Which vertices belong to the same component?
- Is there a path between `u` and `v`?

A naive approach (DFS for each node) runs in O(n + m) but may require recursion or adjacency traversal.
With DSU, we can process edge lists directly in nearly constant amortized time.

#### How It Works (Plain Language)

Each vertex starts in its own component.
For every edge `(u, v)`:

- If `Find(u) != Find(v)`, they are in different components → Union(u, v)
- Otherwise, skip (already connected)

After all edges are processed, all vertices sharing the same root belong to the same connected component.

#### Example

Graph:

```
1, 2     3, 4
      \   /
        5
```

Edges: (1–2), (2–5), (3–5), (3–4)

Step-by-step:

| Step | Edge  | Action | Components           |
| ---- | ----- | ------ | -------------------- |
| 1    | (1,2) | Union  | {1,2}, {3}, {4}, {5} |
| 2    | (2,5) | Union  | {1,2,5}, {3}, {4}    |
| 3    | (3,5) | Union  | {1,2,3,5}, {4}       |
| 4    | (3,4) | Union  | {1,2,3,4,5}          |

✅ All connected → 1 component

#### Visualization

Before:

```
1   2   3   4   5
```

After unions:

```
1—2—5—3—4
```

One large connected component.

#### Tiny Code (Easy Versions)

C Implementation

```c
#include <stdio.h>

#define MAXN 100
int parent[MAXN], rank_[MAXN];

void make_set(int v) {
    parent[v] = v;
    rank_[v] = 0;
}

int find_set(int v) {
    if (v != parent[v])
        parent[v] = find_set(parent[v]);
    return parent[v];
}

void union_sets(int a, int b) {
    a = find_set(a);
    b = find_set(b);
    if (a != b) {
        if (rank_[a] < rank_[b]) parent[a] = b;
        else if (rank_[a] > rank_[b]) parent[b] = a;
        else { parent[b] = a; rank_[a]++; }
    }
}

int main() {
    int n = 5;
    int edges[][2] = {{1,2},{2,5},{3,5},{3,4}};
    for (int i=1; i<=n; i++) make_set(i);
    for (int i=0; i<4; i++)
        union_sets(edges[i][0], edges[i][1]);
    
    int count = 0;
    for (int i=1; i<=n; i++)
        if (find_set(i) == i) count++;
    printf("Number of components: %d\n", count);
}
```

Python Implementation

```python
def make_set(parent, rank, v):
    parent[v] = v
    rank[v] = 0

def find_set(parent, v):
    if parent[v] != v:
        parent[v] = find_set(parent, parent[v])
    return parent[v]

def union_sets(parent, rank, a, b):
    a, b = find_set(parent, a), find_set(parent, b)
    if a != b:
        if rank[a] < rank[b]:
            a, b = b, a
        parent[b] = a
        if rank[a] == rank[b]:
            rank[a] += 1

def connected_components(n, edges):
    parent, rank = {}, {}
    for i in range(1, n+1):
        make_set(parent, rank, i)
    for u, v in edges:
        union_sets(parent, rank, u, v)
    roots = {find_set(parent, i) for i in parent}
    components = {}
    for i in range(1, n+1):
        root = find_set(parent, i)
        components.setdefault(root, []).append(i)
    return components

edges = [(1,2),(2,5),(3,5),(3,4)]
components = connected_components(5, edges)
print("Components:", components)
print("Count:", len(components))
```

Output:

```
Components: {1: [1, 2, 3, 4, 5]}
Count: 1
```

#### Why It Matters

- Quickly answers connectivity questions
- Works directly on edge list (no adjacency matrix needed)
- Forms the backbone of algorithms like Kruskal's MST
- Extensible to dynamic connectivity and offline queries

#### A Gentle Proof (Why It Works)

Union-Find forms a forest of trees, one per component.
Each union merges two trees if and only if there's an edge connecting them.
No cycles are introduced; final roots mark distinct connected components.

Each vertex ends up linked to exactly one representative.

#### Try It Yourself

1. Build a graph with 6 nodes and 2 disconnected clusters.
2. Run DSU unions across edges.
3. Count unique roots.
4. Print grouping `{root: [members]}`

#### Test Cases

| Graph                | Edges             | Components          |
| -------------------- | ----------------- | ------------------- |
| 1–2–3, 4–5           | (1,2),(2,3),(4,5) | {1,2,3}, {4,5}, {6} |
| Complete Graph (1–n) | all pairs         | 1                   |
| Empty Graph          | none              | n                   |

#### Complexity

| Operation       | Time (Amortized) | Space |
| --------------- | ---------------- | ----- |
| Make-Set        | O(1)             | O(n)  |
| Union           | O(α(n))          | O(1)  |
| Find            | O(α(n))          | O(1)  |
| Total (m edges) | O(m α(n))        | O(n)  |

Connected Components (DSU), a clean and scalable way to reveal the hidden clusters of any graph.

### 260 Offline Query DSU

Offline Query DSU is a clever twist on the standard Disjoint Set Union, used when you need to answer connectivity queries in a graph that changes over time, especially when edges are added or removed.

Instead of handling updates online (in real time), we collect all queries first, then process them in reverse, using DSU to efficiently track connections as we "undo" deletions or simulate the timeline backward.

#### What Problem Are We Solving?

We often face questions like:

- "After removing these edges, are nodes `u` and `v` still connected?"
- "If we add edges over time, when do `u` and `v` become connected?"

Online handling is hard because DSU doesn't support deletions directly.
The trick: reverse time, treat deletions as additions in reverse, and answer queries offline.

#### How It Works (Plain Language)

1. Record all events in the order they occur:

   * Edge additions or deletions
   * Connectivity queries

2. Reverse the timeline:

   * Process from the last event backward
   * Every "delete edge" becomes an "add edge"
   * Queries are answered in reverse order

3. Use DSU:

   * Each union merges components as edges appear (in reverse)
   * When processing a query, check if `Find(u) == Find(v)`

4. Finally, reverse the answers to match the original order.

#### Example

Imagine a graph:

```
1, 2, 3
```

Events (in time order):

1. Query(1,3)?
2. Remove edge (2,3)
3. Query(1,3)?

We can't handle removals easily online, so we reverse:

```
Reverse order:
1. Query(1,3)?
2. Add (2,3)
3. Query(1,3)?
```

Step-by-step (in reverse):

| Step | Operation  | Action                | Answer |
| ---- | ---------- | --------------------- | ------ |
| 1    | Query(1,3) | 1 and 3 not connected | No     |
| 2    | Add(2,3)   | Union(2,3)            | –      |
| 3    | Query(1,3) | 1–2–3 connected       | Yes    |

Reverse answers: [Yes, No]

✅ Final output:

- Query 1: Yes
- Query 2: No

#### Visualization

Forward time:

```
1—2—3
```

→ remove (2,3) →

```
1—2   3
```

Backward time:
Start with `1—2   3`
→ Add (2,3) → `1—2—3`

We rebuild connectivity over time by unioning edges in reverse.

#### Tiny Code (Easy Versions)

Python Implementation

```python
def make_set(parent, rank, v):
    parent[v] = v
    rank[v] = 0

def find_set(parent, v):
    if parent[v] != v:
        parent[v] = find_set(parent, parent[v])
    return parent[v]

def union_sets(parent, rank, a, b):
    a, b = find_set(parent, a), find_set(parent, b)
    if a != b:
        if rank[a] < rank[b]:
            a, b = b, a
        parent[b] = a
        if rank[a] == rank[b]:
            rank[a] += 1

# Example
n = 3
edges = {(1,2), (2,3)}
queries = [
    ("?", 1, 3),
    ("-", 2, 3),
    ("?", 1, 3)
$$

# Reverse events
events = list(reversed(queries))

parent, rank = {}, {}
for i in range(1, n+1):
    make_set(parent, rank, i)

active_edges = set(edges)
answers = []

for e in events:
    if e[0] == "?":
        _, u, v = e
        answers.append("YES" if find_set(parent, u) == find_set(parent, v) else "NO")
    elif e[0] == "-":
        _, u, v = e
        union_sets(parent, rank, u, v)

answers.reverse()
for ans in answers:
    print(ans)
```

Output:

```
YES
NO
```

#### Why It Matters

- Handles edge deletions without needing rollback
- Perfect for offline dynamic connectivity
- Used in problems like:

  * "Are u and v connected after k deletions?"
  * "What is the earliest time u and v become connected?"
- Core idea behind Dynamic Trees and Divide & Conquer over time

#### A Gentle Proof (Why It Works)

DSU is monotonic, it supports adding edges, not deleting them.
By reversing time, all deletions become additions.
Thus, we maintain valid connectivity information backward in time,
and can correctly answer queries that depend only on graph connectivity.

Reversing answers afterward restores their original sequence.

#### Try It Yourself

1. Create a graph with 5 nodes and edges (1–2, 2–3, 3–4, 4–5)
2. Remove (3–4), (2–3) sequentially
3. Ask connectivity between (1,5) after each removal
4. Reverse timeline, simulate with DSU

#### Test Cases

| Event Sequence               | Result    |
| ---------------------------- | --------- |
| [?, 1–3], [–, 2–3], [?, 1–3] | [YES, NO] |
| [–, 1–2], [?, 1–3]           | [NO]      |
| [?, 4–5] (no edge)           | [NO]      |

#### Complexity

| Operation                  | Time (Amortized) | Space  |
| -------------------------- | ---------------- | ------ |
| Make-Set                   | O(1)             | O(n)   |
| Union                      | O(α(n))          | O(1)   |
| Find                       | O(α(n))          | O(1)   |
| Total (Q queries, E edges) | O((Q+E) α(n))    | O(n+E) |

Offline Query DSU is your time-reversing tool, flip the story, add edges back, and reveal connectivity across history.

## Section 27. Probabilistic Data Structure 

### 261 Bloom Filter Insert

A Bloom Filter is a compact, probabilistic data structure used for membership testing, it can tell you if an element is definitely not present or possibly present, but never gives false negatives.

Insertion in a Bloom Filter is simple and elegant: hash the item with multiple hash functions and set the corresponding bits to `1` in a bit array.

#### What Problem Are We Solving?

You have a massive dataset, maybe millions or billions of keys, and you just want to ask:

> "Have I seen this before?"

A normal hash set would explode in memory.
A Bloom Filter gives a lightweight alternative:

- No false negatives (safe for skipping)
- Small memory footprint
- Fixed-size bit array

Used in systems like:

- Databases (caching, deduplication)
- Web crawlers (visited URLs)
- Distributed systems (HBase, Cassandra, Bigtable)

#### How It Works (Plain Language)

A Bloom Filter is just a bit array of length `m` (all zeroes initially), plus k independent hash functions.

To insert an element `x`:

1. Compute `k` hash values: `h1(x), h2(x), ..., hk(x)`
2. Map each hash to an index in `[0, m-1]`
3. Set all `bit[h_i(x)] = 1`

So every element lights up multiple bits. Later, to check membership, we look at those same bits, if any is 0, the item was never inserted.

#### Example

Let's build a Bloom Filter with:

- `m = 10` bits
- `k = 3` hash functions

Insert "cat":

```
h1(cat) = 2
h2(cat) = 5
h3(cat) = 7
```

Set bits 2, 5, and 7 to 1:

| Index | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
| ----- | - | - | - | - | - | - | - | - | - | - |
| Value | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 1 | 0 | 0 |

Insert "dog":

```
h1(dog) = 1
h2(dog) = 5
h3(dog) = 9
```

Now bit array:

| Index | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
| ----- | - | - | - | - | - | - | - | - | - | - |
| Value | 0 | 1 | 1 | 0 | 0 | 1 | 0 | 1 | 0 | 1 |

#### Visualization

Each insertion adds "footprints" in multiple spots:

```
Insert(x):
  for i in [1..k]:
     bit[ h_i(x) ] = 1
```

The overlap of bits allows huge compression, but leads to false positives when unrelated keys share bits.

#### Tiny Code (Easy Versions)

C Implementation (Conceptual)

```c
#include <stdio.h>
#include <string.h>

#define M 10
#define K 3

int bitset[M];

int hash1(int x) { return x % M; }
int hash2(int x) { return (x * 3 + 1) % M; }
int hash3(int x) { return (x * 7 + 5) % M; }

void insert(int x) {
    int h[K] = {hash1(x), hash2(x), hash3(x)};
    for (int i = 0; i < K; i++)
        bitset[h[i]] = 1;
}

void print_bits() {
    for (int i = 0; i < M; i++) printf("%d ", bitset[i]);
    printf("\n");
}

int main() {
    memset(bitset, 0, sizeof(bitset));
    insert(42);
    insert(23);
    print_bits();
}
```

Python Implementation

```python
m, k = 10, 3
bitset = [0] * m

def hash_functions(x):
    return [(hash(x) + i * i) % m for i in range(k)]

def insert(x):
    for h in hash_functions(x):
        bitset[h] = 1

def display():
    print("Bit array:", bitset)

insert("cat")
insert("dog")
display()
```

#### Why It Matters

- Extremely space-efficient
- No need to store actual data
- Ideal for membership filters, duplicate detection, and pre-checks before expensive lookups
- The backbone of approximate data structures

#### A Gentle Proof (Why It Works)

Each bit starts at 0.
Each insertion flips `k` bits to 1.
Query returns "maybe" if all `k` bits = 1, otherwise "no".
Thus:

- False negative: impossible (never unset a bit)
- False positive: possible, due to collisions

Probability of false positive ≈ ((1 - e^{-kn/m})^k)

Choosing (m) and (k) well balances accuracy vs. memory.

#### Try It Yourself

1. Choose `m = 20`, `k = 3`
2. Insert {"apple", "banana", "grape"}
3. Print bit array
4. Query for "mango" → likely "maybe" (false positive)

#### Test Cases

| Inserted Elements | Query | Result                               |
| ----------------- | ----- | ------------------------------------ |
| {cat, dog}        | cat   | maybe (true positive)                |
| {cat, dog}        | dog   | maybe (true positive)                |
| {cat, dog}        | fox   | maybe / no (false positive possible) |

#### Complexity

| Operation       | Time                | Space |
| --------------- | ------------------- | ----- |
| Insert          | O(k)                | O(m)  |
| Query           | O(k)                | O(m)  |
| False Positives | ≈ (1 - e^{-kn/m})^k | –     |

Bloom Filter Insert, write once, maybe forever. Compact, fast, and probabilistically powerful.

### 262 Bloom Filter Query

A Bloom Filter Query checks whether an element *might* be present in the set. It uses the same k hash functions and bit array as insertion, but instead of setting bits, it simply tests them.

The magic:

- If any bit is `0`, the element was never inserted (definitely not present).
- If all bits are `1`, the element is possibly present (maybe yes).

No false negatives, if the Bloom Filter says "no," it's always correct.

#### What Problem Are We Solving?

When handling huge datasets (web crawlers, caches, key-value stores), we need a fast and memory-efficient way to answer:

> "Have I seen this before?"

But full storage is expensive.
Bloom Filters let us skip expensive lookups by confidently ruling out items early.

Typical use cases:

- Databases: Avoid disk lookups for missing keys
- Web crawlers: Skip revisiting known URLs
- Networking: Cache membership checks

#### How It Works (Plain Language)

A Bloom Filter has:

- Bit array `bits[0..m-1]`
- `k` hash functions

To query element `x`:

1. Compute all hashes: `h1(x), h2(x), ..., hk(x)`
2. Check bits at those positions

   * If any `bit[h_i(x)] == 0`, return "No" (definitely not present)
   * If all are `1`, return "Maybe" (possible false positive)

The key rule: bits can only turn on, never off, so "no" answers are reliable.

#### Example

Let's use a filter of size `m = 10`, `k = 3`:

Bit array after inserting "cat" and "dog":

```
Index:  0 1 2 3 4 5 6 7 8 9
Bits:   0 1 1 0 0 1 0 1 0 1
```

Now query "cat":

```
h1(cat)=2, h2(cat)=5, h3(cat)=7
bits[2]=1, bits[5]=1, bits[7]=1 → maybe present ✅
```

Query "fox":

```
h1(fox)=3, h2(fox)=5, h3(fox)=8
bits[3]=0 → definitely not present ❌
```

#### Visualization

Query(x):

```
for i in 1..k:
  if bit[ h_i(x) ] == 0:
     return "NO"
return "MAYBE"
```

Bloom Filters say "maybe" for safety, but never lie with "no".

#### Tiny Code (Easy Versions)

C Implementation (Conceptual)

```c
#include <stdio.h>

#define M 10
#define K 3
int bitset[M];

int hash1(int x) { return x % M; }
int hash2(int x) { return (x * 3 + 1) % M; }
int hash3(int x) { return (x * 7 + 5) % M; }

int query(int x) {
    int h[K] = {hash1(x), hash2(x), hash3(x)};
    for (int i = 0; i < K; i++)
        if (bitset[h[i]] == 0)
            return 0; // definitely not
    return 1; // possibly yes
}

int main() {
    bitset[2] = bitset[5] = bitset[7] = 1; // insert "cat"
    printf("Query 42: %s\n", query(42) ? "maybe" : "no");
    printf("Query 23: %s\n", query(23) ? "maybe" : "no");
}
```

Python Implementation

```python
m, k = 10, 3
bitset = [0] * m

def hash_functions(x):
    return [(hash(x) + i * i) % m for i in range(k)]

def insert(x):
    for h in hash_functions(x):
        bitset[h] = 1

def query(x):
    for h in hash_functions(x):
        if bitset[h] == 0:
            return "NO"
    return "MAYBE"

insert("cat")
insert("dog")
print("Query cat:", query("cat"))
print("Query dog:", query("dog"))
print("Query fox:", query("fox"))
```

#### Why It Matters

- Constant-time membership test
- No false negatives, only rare false positives
- Great for pre-filtering before heavy lookups
- Common in distributed systems and caching layers

#### A Gentle Proof (Why It Works)

Each inserted element sets $k$ bits to 1 in the bit array.  
For a query, if all $k$ bits are 1, the element *might* be present (or collisions caused those bits).  
If any bit is 0, the element was definitely never inserted.

The false positive probability is:

$$
p = \left(1 - e^{-kn/m}\right)^k
$$

where:

- $m$: size of the bit array  
- $n$: number of inserted elements  
- $k$: number of hash functions  

Choose $m$ and $k$ to minimize $p$ for the target false positive rate.


#### Try It Yourself

1. Create a filter with `m=20`, `k=3`
2. Insert {"apple", "banana"}
3. Query {"apple", "grape"}
4. See how "grape" may return "maybe"

#### Test Cases

| Inserted Elements | Query    | Result |
| ----------------- | -------- | ------ |
| {cat, dog}        | cat      | maybe  |
| {cat, dog}        | dog      | maybe  |
| {cat, dog}        | fox      | no     |
| {}                | anything | no     |

#### Complexity

| Operation | Time | Space | False Negatives | False Positives |
| --------- | ---- | ----- | --------------- | --------------- |
| Query     | O(k) | O(m)  | None            | Possible        |

Bloom Filter Query, fast, memory-light, and trustable when it says "no."

### 263 Counting Bloom Filter

A Counting Bloom Filter (CBF) extends the classic Bloom Filter by allowing deletions.
Instead of a simple bit array, it uses an integer counter array, so each bit becomes a small counter tracking how many elements mapped to that position.

When inserting, increment the counters; when deleting, decrement them. If all required counters are greater than zero, the element is possibly present.

#### What Problem Are We Solving?

A regular Bloom Filter is *write-only*: you can insert, but not remove. Once a bit is set to `1`, it stays `1` forever.

But what if:

- You're tracking *active sessions*?
- You need to remove *expired cache keys*?
- You want to maintain a *sliding window* of data?

Then you need a Counting Bloom Filter, which supports safe deletions.

#### How It Works (Plain Language)

We replace the bit array with a counter array of size $m$.

For each element $x$:

Insert(x)  
- For each hash $h_i(x)$: increment `count[h_i(x)]++`

Query(x)  
- Check all `count[h_i(x)] > 0` → "maybe"  
- If any are `0`, → "no"

Delete(x)  
- For each hash $h_i(x)$: decrement `count[h_i(x)]--`  
- Ensure counters never become negative



#### Example

Let $m = 10$, $k = 3$

Initial state  
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

Insert "cat"  

$h_1(\text{cat}) = 2$  
$h_2(\text{cat}) = 5$  
$h_3(\text{cat}) = 7$  

→ increment $count[2]$, $count[5]$, $count[7]$  

Array after insertion  
[0, 0, 1, 0, 0, 1, 0, 1, 0, 0]

Insert "dog"  

$h_1(\text{dog}) = 1$  
$h_2(\text{dog}) = 5$  
$h_3(\text{dog}) = 9$  

→ increment $count[1]$, $count[5]$, $count[9]$  

Array after insertion  
[0, 1, 1, 0, 0, 2, 0, 1, 0, 1]

Delete "cat"  

$h_1(\text{cat}) = 2$  
$h_2(\text{cat}) = 5$  
$h_3(\text{cat}) = 7$  

→ decrement $count[2]$, $count[5]$, $count[7]$  

Array after deletion  
[0, 1, 0, 0, 0, 1, 0, 0, 0, 1]

After deletion, "cat" is removed while "dog" remains.



#### Visualization

Counters evolve with each insert/delete:

| Operation   | Index 1 | 2 | 5 | 7 | 9 |
| ----------- | ------- | - | - | - | - |
| Insert(cat) | –       | 1 | 1 | 1 | – |
| Insert(dog) | 1       | 1 | 2 | 1 | 1 |
| Delete(cat) | 1       | 0 | 1 | 0 | 1 |

#### Tiny Code (Easy Versions)

Python Implementation

```python
m, k = 10, 3
counts = [0] * m

def hash_functions(x):
    return [(hash(x) + i * i) % m for i in range(k)]

def insert(x):
    for h in hash_functions(x):
        counts[h] += 1

def query(x):
    return all(counts[h] > 0 for h in hash_functions(x))

def delete(x):
    for h in hash_functions(x):
        if counts[h] > 0:
            counts[h] -= 1

# Example
insert("cat")
insert("dog")
print("After insert:", counts)
delete("cat")
print("After delete(cat):", counts)
print("Query cat:", "Maybe" if query("cat") else "No")
print("Query dog:", "Maybe" if query("dog") else "No")
```

Output

```
After insert: [0,1,1,0,0,2,0,1,0,1]
After delete(cat): [0,1,0,0,0,1,0,0,0,1]
Query cat: No
Query dog: Maybe
```

#### Why It Matters

- Enables safe deletions without full reset
- Useful for cache invalidation, session tracking, streaming windows
- Memory-efficient alternative to dynamic hash sets

#### A Gentle Proof (Why It Works)

Each counter approximates how many items mapped to it.
Deletion only decrements counters, if multiple elements shared a hash, it stays ≥1.
Thus:

- No false negatives, unless you over-decrement (bug)
- False positives remain possible, as in classic Bloom Filters

Probability of false positive remains:
$$
p = \left(1 - e^{-kn/m}\right)^k
$$

#### Try It Yourself

1. Create a filter with `m=20, k=3`
2. Insert 5 words
3. Delete 2 of them
4. Query all 5, deleted ones should say no, others maybe

#### Test Cases

| Operation   | Array Snapshot        | Query Result          |
| ----------- | --------------------- | --------------------- |
| Insert(cat) | [0,0,1,0,0,1,0,1,0,0] | –                     |
| Insert(dog) | [0,1,1,0,0,2,0,1,0,1] | –                     |
| Delete(cat) | [0,1,0,0,0,1,0,0,0,1] | cat → No, dog → Maybe |

#### Complexity

| Operation | Time | Space |
| --------- | ---- | ----- |
| Insert    | O(k) | O(m)  |
| Query     | O(k) | O(m)  |
| Delete    | O(k) | O(m)  |

Counting Bloom Filter, flexible and reversible, keeping memory lean and deletions clean.

### 264 Cuckoo Filter

A Cuckoo Filter is a space-efficient alternative to Bloom filters that supports both insertions and deletions while maintaining low false positive rates. Instead of using a bit array, it stores small *fingerprints* of keys in hash buckets, using cuckoo hashing to resolve collisions.

It's like a smarter, tidier roommate, always making room by moving someone else when things get crowded.

#### What Problem Are We Solving?

Bloom filters are fast and compact, but they can't delete elements efficiently. Counting Bloom filters fix that, but they're more memory-hungry.

We want:

- Fast membership queries (`O(1)`)
- Insert + Delete support
- High load factor (~95%)
- Compact memory footprint

The Cuckoo Filter solves all three by using cuckoo hashing + small fingerprints.

#### How It Works (Plain Language)

Each element is represented by a short fingerprint (e.g., 8 bits).
Each fingerprint can be placed in two possible buckets, determined by two hash functions.

If a bucket is full, we *evict* an existing fingerprint and relocate it to its alternate bucket (cuckoo style).

Operations:

1. Insert(x)

   * Compute fingerprint `f = hash_fingerprint(x)`
   * Compute `i1 = hash(x) % m`
   * Compute `i2 = i1 ⊕ hash(f)` (alternate index)
   * Try to place `f` in either `i1` or `i2`
   * If both full, evict one fingerprint and relocate

2. Query(x)

   * Check if `f` is present in bucket `i1` or `i2`

3. Delete(x)

   * Remove `f` if found in either bucket

#### Example (Step-by-Step)

Assume:

- Buckets: `m = 4`
- Bucket size: `b = 2`
- Fingerprint size: 4 bits

Start (empty):

| Bucket 0 | Bucket 1 | Bucket 2 | Bucket 3 |
| -------- | -------- | -------- | -------- |
|          |          |          |          |

Insert A

```
f(A) = 1010  
i1 = 1, i2 = 1 ⊕ hash(1010) = 3  
→ Place in bucket 1
```

| B0 | B1   | B2 | B3 |
| -- | ---- | -- | -- |
|    | 1010 |    |    |

Insert B

```
f(B) = 0111  
i1 = 3, i2 = 3 ⊕ hash(0111) = 0  
→ Place in bucket 3
```

| B0 | B1   | B2 | B3   |
| -- | ---- | -- | ---- |
|    | 1010 |    | 0111 |

Insert C

```
f(C) = 0100  
i1 = 1, i2 = 1 ⊕ hash(0100) = 2  
→ Bucket 1 full? Move (cuckoo) if needed  
→ Place in bucket 2
```

| B0 | B1   | B2   | B3   |
| -- | ---- | ---- | ---- |
|    | 1010 | 0100 | 0111 |

Query C → found in bucket 2 ✅
Delete A → remove from bucket 1

#### Tiny Code (Easy Version)

Python Example

```python
import random

class CuckooFilter:
    def __init__(self, size=4, bucket_size=2, fingerprint_bits=4):
        self.size = size
        self.bucket_size = bucket_size
        self.buckets = [[] for _ in range(size)]
        self.mask = (1 << fingerprint_bits) - 1

    def _fingerprint(self, item):
        return hash(item) & self.mask

    def _alt_index(self, i, fp):
        return (i ^ hash(fp)) % self.size

    def insert(self, item, max_kicks=4):
        fp = self._fingerprint(item)
        i1 = hash(item) % self.size
        i2 = self._alt_index(i1, fp)
        for i in (i1, i2):
            if len(self.buckets[i]) < self.bucket_size:
                self.buckets[i].append(fp)
                return True
        # Cuckoo eviction
        i = random.choice([i1, i2])
        for _ in range(max_kicks):
            fp, self.buckets[i][0] = self.buckets[i][0], fp
            i = self._alt_index(i, fp)
            if len(self.buckets[i]) < self.bucket_size:
                self.buckets[i].append(fp)
                return True
        return False  # insert failed

    def contains(self, item):
        fp = self._fingerprint(item)
        i1 = hash(item) % self.size
        i2 = self._alt_index(i1, fp)
        return fp in self.buckets[i1] or fp in self.buckets[i2]

    def delete(self, item):
        fp = self._fingerprint(item)
        i1 = hash(item) % self.size
        i2 = self._alt_index(i1, fp)
        for i in (i1, i2):
            if fp in self.buckets[i]:
                self.buckets[i].remove(fp)
                return True
        return False
```

#### Why It Matters

- Supports deletions efficiently
- High load factor (~95%) before failure
- Smaller than Counting Bloom Filter for same error rate
- Practical for caches, membership checks, deduplication

#### A Gentle Proof (Why It Works)

Each element has two potential homes → high flexibility
Cuckoo eviction ensures the table remains compact
Short fingerprints preserve memory while keeping collisions low

False positive rate:
$$
p \approx \frac{2b}{2^f}
$$
where `b` is bucket size, `f` is fingerprint bits

#### Try It Yourself

1. Build a filter with 8 buckets, 2 slots each
2. Insert 5 words
3. Delete 1 word
4. Query all 5, deleted one should return "no"

#### Test Cases

| Operation | Bucket 0 | Bucket 1 | Bucket 2 | Bucket 3 | Result    |
| --------- | -------- | -------- | -------- | -------- | --------- |
| Insert(A) | –        | 1010     | –        | –        | OK        |
| Insert(B) | –        | 1010     | –        | 0111     | OK        |
| Insert(C) | –        | 1010     | 0100     | 0111     | OK        |
| Query(C)  | –        | 1010     | 0100     | 0111     | Maybe     |
| Delete(A) | –        | –        | 0100     | 0111     | Deleted ✅ |

#### Complexity

| Operation | Time           | Space        |
| --------- | -------------- | ------------ |
| Insert    | O(1) amortized | O(m × b × f) |
| Query     | O(1)           | O(m × b × f) |
| Delete    | O(1)           | O(m × b × f) |

Cuckoo Filter, a nimble, compact, and deletable membership structure built on playful eviction.

### 265 Count-Min Sketch

A Count-Min Sketch (CMS) is a compact data structure for estimating the frequency of elements in a data stream.
Instead of storing every item, it keeps a small 2D array of counters updated by multiple hash functions.
It never underestimates counts, but may slightly overestimate due to hash collisions.

Think of it as a memory-efficient radar, it doesn't see every car, but it knows roughly how many are on each lane.

#### What Problem Are We Solving?

In streaming or massive datasets, we can't store every key-value count exactly.
We want to track approximate frequencies with limited memory, supporting:

- Streaming updates: count items as they arrive
- Approximate queries: estimate item frequency
- Memory efficiency: sublinear space

Used in network monitoring, NLP (word counts), heavy hitter detection, and online analytics.

#### How It Works (Plain Language)

CMS uses a 2D array with `d` rows and `w` columns.
Each row has a different hash function.
Each item updates one counter per row, all at its hashed index.

Insert(x):
For each row `i`:

```
index = hash_i(x) % w
count[i][index] += 1
```

Query(x):
For each row `i`:

```
estimate = min(count[i][hash_i(x) % w])
```

We take the *minimum* across rows, hence "Count-Min".

#### Example (Step-by-Step)

Let `d = 3` hash functions, `w = 10` width.

Initialize table:

```
Row1: [0 0 0 0 0 0 0 0 0 0]
Row2: [0 0 0 0 0 0 0 0 0 0]
Row3: [0 0 0 0 0 0 0 0 0 0]
```

Insert "apple"

```
h1(apple)=2, h2(apple)=5, h3(apple)=9
→ increment positions (1,2), (2,5), (3,9)
```

Insert "banana"

```
h1(banana)=2, h2(banana)=3, h3(banana)=1
→ increment (1,2), (2,3), (3,1)
```

Now:

```
Row1: [0 0 2 0 0 0 0 0 0 0]
Row2: [0 0 0 1 0 1 0 0 0 0]
Row3: [0 1 0 0 0 0 0 0 0 1]
```

Query "apple":

```
min(count[1][2], count[2][5], count[3][9]) = min(2,1,1) = 1
```

Estimate frequency ≈ 1 (may be slightly high if collisions overlap).

#### Table Visualization

| Item   | h₁(x) | h₂(x) | h₃(x) | Estimated Count |
| ------ | ----- | ----- | ----- | --------------- |
| apple  | 2     | 5     | 9     | 1               |
| banana | 2     | 3     | 1     | 1               |

#### Tiny Code (Easy Version)

Python Example

```python
import mmh3

class CountMinSketch:
    def __init__(self, width=10, depth=3):
        self.width = width
        self.depth = depth
        self.table = [[0] * width for _ in range(depth)]
        self.seeds = [i * 17 for i in range(depth)]  # different hash seeds

    def _hash(self, item, seed):
        return mmh3.hash(str(item), seed) % self.width

    def add(self, item, count=1):
        for i, seed in enumerate(self.seeds):
            idx = self._hash(item, seed)
            self.table[i][idx] += count

    def query(self, item):
        estimates = []
        for i, seed in enumerate(self.seeds):
            idx = self._hash(item, seed)
            estimates.append(self.table[i][idx])
        return min(estimates)

# Example usage
cms = CountMinSketch(width=10, depth=3)
cms.add("apple")
cms.add("banana")
cms.add("apple")
print("apple:", cms.query("apple"))
print("banana:", cms.query("banana"))
```

Output

```
apple: 2
banana: 1
```

#### Why It Matters

- Compact: O(w × d) memory
- Fast: O(1) updates and queries
- Scalable: Works on unbounded data streams
- Deterministic upper bound: never underestimates

Used in:

- Word frequency estimation
- Network flow counting
- Clickstream analysis
- Approximate histograms

#### A Gentle Proof (Why It Works)

Each item is hashed to `d` positions.
Collisions can cause overestimation, never underestimation.
By taking the minimum, we get the best upper bound estimate.

Error bound:
$$
\text{error} \le \epsilon N, \quad \text{with probability } 1 - \delta
$$
Choose:
$$
w = \lceil e / \epsilon \rceil,\quad d = \lceil \ln(1/\delta) \rceil
$$

#### Try It Yourself

1. Create a CMS with `(w=20, d=4)`
2. Stream 1000 random items
3. Compare estimated vs. actual counts
4. Observe overestimation patterns

#### Test Cases

| Operation      | Action | Query Result |
| -------------- | ------ | ------------ |
| Insert(apple)  | +1     | –            |
| Insert(apple)  | +1     | –            |
| Insert(banana) | +1     | –            |
| Query(apple)   | –      | 2            |
| Query(banana)  | –      | 1            |

#### Complexity

| Operation | Time | Space    |
| --------- | ---- | -------- |
| Insert    | O(d) | O(w × d) |
| Query     | O(d) | O(w × d) |

Count-Min Sketch, lightweight, accurate-enough, and built for streams that never stop flowing.

### 266 HyperLogLog

A HyperLogLog (HLL) is a probabilistic data structure for cardinality estimation, that is, estimating the number of distinct elements in a massive dataset or stream using very little memory.

It doesn't remember *which* items you saw, only *how many distinct ones there likely were*. Think of it as a memory-efficient crowd counter, it doesn't know faces, but it knows how full the stadium is.

#### What Problem Are We Solving?

When processing large data streams (web analytics, logs, unique visitors, etc.), exact counting of distinct elements (using sets or hash tables) is too memory-heavy.

We want a solution that:

- Tracks distinct counts approximately
- Uses constant memory
- Supports mergeability (combine two sketches easily)

HyperLogLog delivers O(1) time per update and ~1.04/√m error rate with just kilobytes of memory.

#### How It Works (Plain Language)

Each incoming element is hashed to a large binary number.
HLL uses the position of the leftmost 1-bit in the hash to estimate how rare (and thus how many) elements exist.

It maintains an array of `m` *registers* (buckets). Each bucket stores the maximum leading zero count seen so far for its hash range.

Steps:

1. Hash element `x` to 64 bits: `h = hash(x)`
2. Bucket index: use first `p` bits of `h` to pick one of `m = 2^p` buckets
3. Rank: count leading zeros in the remaining bits + 1
4. Update: store `max(existing, rank)` in that bucket

Final count = harmonic mean of `2^rank` values, scaled by a bias-corrected constant.

#### Example (Step-by-Step)

Let `p = 2` → `m = 4` buckets.
Initialize registers: `[0,0,0,0]`

Insert "apple":

```
hash("apple") = 110010100…  
bucket = first 2 bits = 11 (3)
rank = position of first 1 after prefix = 2
→ bucket[3] = max(0, 2) = 2
```

Insert "banana":

```
hash("banana") = 01000100…  
bucket = 01 (1)
rank = 3
→ bucket[1] = 3
```

Insert "pear":

```
hash("pear") = 10010000…  
bucket = 10 (2)
rank = 4
→ bucket[2] = 4
```

Registers: `[0, 3, 4, 2]`

Estimate:
$$
E = \alpha_m \cdot m^2 / \sum 2^{-M[i]}
$$
where `α_m` is a bias-correction constant (≈0.673 for small m).

#### Visualization

| Bucket | Values Seen | Leading Zeros | Stored Rank |
| ------ | ----------- | ------------- | ----------- |
| 0      | none        | –             | 0           |
| 1      | banana      | 2             | 3           |
| 2      | pear        | 3             | 4           |
| 3      | apple       | 1             | 2           |

#### Tiny Code (Easy Version)

Python Implementation

```python
import mmh3
import math

class HyperLogLog:
    def __init__(self, p=4):
        self.p = p
        self.m = 1 << p
        self.registers = [0] * self.m
        self.alpha = 0.673 if self.m == 16 else 0.709 if self.m == 32 else 0.7213 / (1 + 1.079 / self.m)

    def _hash(self, x):
        return mmh3.hash(str(x), 42) & 0xffffffff

    def add(self, x):
        h = self._hash(x)
        idx = h >> (32 - self.p)
        w = (h << self.p) & 0xffffffff
        rank = self._rank(w, 32 - self.p)
        self.registers[idx] = max(self.registers[idx], rank)

    def _rank(self, w, bits):
        r = 1
        while w & (1 << (bits - 1)) == 0 and r <= bits:
            r += 1
            w <<= 1
        return r

    def count(self):
        Z = sum([2.0  -v for v in self.registers])
        E = self.alpha * self.m * self.m / Z
        return round(E)

# Example
hll = HyperLogLog(p=4)
for x in ["apple", "banana", "pear", "apple"]:
    hll.add(x)
print("Estimated distinct count:", hll.count())
```

Output

```
Estimated distinct count: 3
```

#### Why It Matters

- Tiny memory footprint (kilobytes for billions of elements)
- Mergeable: HLL(A∪B) = max(HLL(A), HLL(B)) bucket-wise
- Used in: Redis, Google BigQuery, Apache DataSketches, analytics systems

#### A Gentle Proof (Why It Works)

The position of the first 1-bit follows a geometric distribution, rare long zero streaks mean more unique elements.
By keeping the *max* observed rank per bucket, HLL captures global rarity efficiently.
Combining across buckets gives a harmonic mean that balances under/over-counting.

Error ≈ 1.04 / √m, so doubling `m` halves the error.

#### Try It Yourself

1. Create HLL with `p = 10` (`m = 1024`)
2. Add 1M random numbers
3. Compare HLL estimate with true count
4. Test merging two sketches

#### Test Cases

| Items | True Count | Estimated | Error |
| ----- | ---------- | --------- | ----- |
| 10    | 10         | 10        | 0%    |
| 1000  | 1000       | 995       | ~0.5% |
| 1e6   | 1,000,000  | 1,010,000 | ~1%   |

#### Complexity

| Operation | Time | Space |
| --------- | ---- | ----- |
| Add       | O(1) | O(m)  |
| Count     | O(m) | O(m)  |
| Merge     | O(m) | O(m)  |

HyperLogLog, counting the uncountable, one leading zero at a time.

### 267 Flajolet–Martin Algorithm

The Flajolet–Martin (FM) algorithm is one of the earliest and simplest approaches for probabilistic counting, estimating the number of distinct elements in a data stream using tiny memory.

It's the conceptual ancestor of HyperLogLog, showing the brilliant idea that *the position of the first 1-bit in a hash tells us something about rarity*.

#### What Problem Are We Solving?

When elements arrive in a stream too large to store exactly (web requests, IP addresses, words), we want to estimate the number of unique elements without storing them all.

A naive approach (hash set) is O(n) memory.
Flajolet–Martin achieves O(1) memory and O(1) updates.

We need:

- A streaming algorithm
- With constant space
- For distinct count estimation

#### How It Works (Plain Language)

Each element is hashed to a large binary number (uniformly random).  
We then find the position of the least significant 1-bit — the number of trailing zeros.  
A value with many trailing zeros is rare, which indicates a larger underlying population.

We track the maximum number of trailing zeros observed, denoted as `R`.  
The estimated number of distinct elements is:

$$
\hat{N} = \phi \times 2^{R}
$$

where $\phi \approx 0.77351$ is a correction constant.

Steps:

1. Initialize $R = 0$  
2. For each element $x$:  
   - $h = \text{hash}(x)$  
   - $r = \text{count\_trailing\_zeros}(h)$  
   - $R = \max(R, r)$  
3. Estimate distinct count as $\hat{N} \approx \phi \times 2^{R}$


#### Example (Step-by-Step)

Stream: `[apple, banana, apple, cherry, date]`

| Element | Hash (Binary) | Trailing Zeros | R |
| ------- | ------------- | -------------- | - |
| apple   | 10110         | 1              | 1 |
| banana  | 10000         | 4              | 4 |
| apple   | 10110         | 1              | 4 |
| cherry  | 11000         | 3              | 4 |
| date    | 01100         | 2              | 4 |

Final value: $R = 4$

Estimate:
$$
\hat{N} = 0.77351 \times 2^{4} = 0.77351 \times 16 \approx 12.38
$$

So the estimated number of distinct elements is about 12  
(overestimation due to small sample size).

In practice, multiple independent hash functions or registers are used,  
and their results are averaged to reduce variance and improve accuracy.


#### Visualization

| Hash  | Binary Form | Trailing Zeros | Meaning                                       |
| ----- | ----------- | -------------- | --------------------------------------------- |
| 10000 | 16          | 4              | very rare pattern → suggests large population |
| 11000 | 24          | 3              | rare-ish                                      |
| 10110 | 22          | 1              | common                                        |

The *longest* zero-run gives the scale of rarity.

#### Tiny Code (Easy Version)

Python Example

```python
import mmh3
import math

def trailing_zeros(x):
    if x == 0:
        return 32
    tz = 0
    while (x & 1) == 0:
        tz += 1
        x >>= 1
    return tz

def flajolet_martin(stream, seed=42):
    R = 0
    for x in stream:
        h = mmh3.hash(str(x), seed) & 0xffffffff
        r = trailing_zeros(h)
        R = max(R, r)
    phi = 0.77351
    return int(phi * (2  R))

# Example
stream = ["apple", "banana", "apple", "cherry", "date"]
print("Estimated distinct count:", flajolet_martin(stream))
```

Output

```
Estimated distinct count: 12
```

#### Why It Matters

- Foundational for modern streaming algorithms
- Inspired LogLog and HyperLogLog
- Memory-light: just a few integers
- Useful in approximate analytics, network telemetry, data warehouses

#### A Gentle Proof (Why It Works)

Each hash output is uniformly random.  
The probability of observing a value with $r$ trailing zeros is:

$$
P(r) = \frac{1}{2^{r+1}}
$$

If such a value appears, it suggests the stream size is roughly $2^{r}$.  
Therefore, the estimate $2^{R}$ naturally scales with the number of unique elements.

By using multiple independent estimators and averaging their results,  
the variance can be reduced from about 50% down to around 10%.


#### Try It Yourself

1. Generate 100 random numbers, feed to FM
2. Compare estimated vs. true count
3. Repeat 10 runs, compute average error
4. Try combining multiple estimators (median of means)

#### Test Cases

| Stream Size | True Distinct | R  | Estimate | Error |
| ----------- | ------------- | -- | -------- | ----- |
| 10          | 10            | 3  | 6        | -40%  |
| 100         | 100           | 7  | 99       | -1%   |
| 1000        | 1000          | 10 | 791      | -21%  |

Using multiple registers improves accuracy significantly.

#### Complexity

| Operation | Time | Space |
| --------- | ---- | ----- |
| Insert    | O(1) | O(1)  |
| Query     | O(1) | O(1)  |

Flajolet–Martin Algorithm, the original spark of probabilistic counting, turning randomness into estimation magic.

### 268 MinHash

A MinHash is a probabilistic algorithm for estimating set similarity, particularly the Jaccard similarity, without comparing all elements directly.
Instead of storing every element, it constructs a signature of compact hash values. The more overlap in signatures, the more similar the sets.

MinHash is foundational in large-scale similarity estimation, fast, memory-efficient, and mathematically elegant.

#### What Problem Are We Solving?

To compute the exact Jaccard similarity between two sets $A$ and $B$:

$$
J(A, B) = \frac{|A \cap B|}{|A \cup B|}
$$

This requires comparing all elements, which is infeasible for large datasets.

We need methods that provide:

- Compact sketches using small memory  
- Fast approximate comparisons  
- Adjustable accuracy through the number of hash functions  

MinHash meets these requirements by applying multiple random hash functions  
and recording the minimum hash value from each function as the set’s signature.


#### How It Works (Plain Language)

For each set, we apply several independent hash functions.
For each hash function, we record the minimum hash value among all elements.

Two sets that share many elements tend to share the same minimums across these hash functions.
Thus, the fraction of matching minimums estimates their Jaccard similarity.

Algorithm:

1. Choose $k$ hash functions $h_1, h_2, \ldots, h_k$.  
2. For each set $S$, compute its signature:

$$
\text{sig}(S)[i] = \min_{x \in S} h_i(x)
$$

3. Estimate the Jaccard similarity by:

$$
\widehat{J}(A, B) = \frac{1}{k} \sum_{i=1}^{k} \mathbf{1}\{\text{sig}(A)[i] = \text{sig}(B)[i]\}
$$

Each matching position between $\text{sig}(A)$ and $\text{sig}(B)$ corresponds  
to one agreeing hash function, and the fraction of matches approximates $J(A,B)$.


#### Example (Step by Step)

Let:

$$
A = {\text{apple}, \text{banana}, \text{cherry}}, \quad
B = {\text{banana}, \text{cherry}, \text{date}}
$$

Suppose we have 3 hash functions:

| Item   | $h_1$ | $h_2$ | $h_3$ |
| ------ | ----- | ----- | ----- |
| apple  | 5     | 1     | 7     |
| banana | 2     | 4     | 3     |
| cherry | 3     | 2     | 1     |
| date   | 4     | 3     | 2     |


Signature(A):
$$
\text{sig}(A) = [\min(5,2,3), \min(1,4,2), \min(7,3,1)] = [2,1,1]
$$

Signature(B):
$$
\text{sig}(B) = [\min(2,3,4), \min(4,2,3), \min(3,1,2)] = [2,2,1]
$$

Compare element-wise:  
Matches occur at positions 1 and 3 → $\tfrac{2}{3} \approx 0.67$

Actual Jaccard similarity:

$$
J(A, B) = \frac{|\{\text{banana}, \text{cherry}\}|}{|\{\text{apple}, \text{banana}, \text{cherry}, \text{date}\}|}
= \frac{2}{4} = 0.5
$$

The MinHash estimate of $0.67$ is reasonably close to the true value $0.5$,  
demonstrating that even a small number of hash functions can yield a good approximation.


#### Visualization

| Hash Function | $\text{sig}(A)$ | $\text{sig}(B)$ | Match |
| -------------- | --------------- | --------------- | ------ |
| $h_1$ | 2 | 2 | ✓ |
| $h_2$ | 1 | 2 | ✗ |
| $h_3$ | 1 | 1 | ✓ |
| Similarity | – | – | $(2/3 = 0.67)$ |


#### Tiny Code (Easy Version)

```python
import mmh3
import math

def minhash_signature(elements, num_hashes=5, seed=42):
    sig = [math.inf] * num_hashes
    for x in elements:
        for i in range(num_hashes):
            h = mmh3.hash(str(x), seed + i)
            if h < sig[i]:
                sig[i] = h
    return sig

def jaccard_minhash(sigA, sigB):
    matches = sum(1 for a, b in zip(sigA, sigB) if a == b)
    return matches / len(sigA)

# Example
A = {"apple", "banana", "cherry"}
B = {"banana", "cherry", "date"}

sigA = minhash_signature(A, 10)
sigB = minhash_signature(B, 10)
print("Approx similarity:", jaccard_minhash(sigA, sigB))
```

Output

```
Approx similarity: 0.6
```

#### Why It Matters

- Scalable similarity: enables fast comparison of very large sets  
- Compact representation: stores only $k$ integers per set  
- Composable: supports set unions using componentwise minimum  
- Common applications:
  - Document deduplication  
  - Web crawling and search indexing  
  - Recommendation systems  
  - Large-scale clustering


#### A Gentle Proof (Why It Works)

For a random permutation $h$:

$$
P[\min(h(A)) = \min(h(B))] = J(A, B)
$$

Each hash function behaves like a Bernoulli trial with success probability $J(A, B)$.  
The MinHash estimator is unbiased:

$$
E[\widehat{J}] = J(A, B)
$$

The variance decreases as $\tfrac{1}{k}$,  
so increasing the number of hash functions improves accuracy.


#### Try It Yourself

1. Choose two sets with partial overlap.  
2. Generate MinHash signatures using $k = 20$ hash functions.  
3. Compute both the estimated and true Jaccard similarities.  
4. Increase $k$ and observe how the estimated similarity converges toward the true value — larger $k$ reduces variance and improves accuracy.


#### Test Cases

| Sets                 | True $J(A,B)$ | $k$ | Estimated | Error |
| -------------------- | ------------- | --- | ---------- | ----- |
| $A=\{1,2,3\}, B=\{2,3,4\}$ | 0.5 | 10 | 0.6  | +0.1  |
| $A=\{1,2\}, B=\{1,2,3,4\}$ | 0.5 | 20 | 0.45 | -0.05 |
| $A=B$                | 1.0 | 10 | 1.0 | 0.0  |


#### Complexity

| Operation       | Time             | Space |
| --------------- | ---------------- | ----- |
| Build Signature | $O(n \times k)$  | $O(k)$ |
| Compare         | $O(k)$           | $O(k)$ |


MinHash turns set similarity into compact signatures —
a small sketch that captures the essence of large sets with statistical grace.

### 269 Reservoir Sampling

Reservoir Sampling is a classic algorithm for randomly sampling k elements from a stream of unknown or very large size, ensuring every element has an equal probability of being selected.

It's the perfect tool when you can't store everything, like catching a few fish from an endless river, one by one, without bias.

#### What Problem Are We Solving?

When data arrives as a stream too large to store in memory, we cannot know its total size in advance.  
Yet, we often need to maintain a uniform random sample of fixed size $k$.

A naive approach would store all items and then sample,  
but this becomes infeasible for large or unbounded data.

Reservoir Sampling provides a one-pass solution with these guarantees:

- Each item in the stream has equal probability $\tfrac{k}{n}$ of being included  
- Uses only $O(k)$ memory  
- Processes data in a single pass


#### How It Works (Plain Language)

We maintain a reservoir (array) of size $k$.  
As each new element arrives, we decide probabilistically whether it replaces one of the existing items.

Steps:

1. Fill the reservoir with the first $k$ elements.  
2. For each element at index $i$ (starting from $i = k + 1$):

   - Generate a random integer $j \in [1, i]$  
   - If $j \le k$, replace $\text{reservoir}[j]$ with the new element  

This ensures every element has an equal chance $\tfrac{k}{n}$ to remain.

#### Example (step by step)

Stream: [A, B, C, D, E]  
Goal: $k = 2$

1. Start with the first 2 → [A, B]  
2. $i = 3$, item = C  
   - Pick random $j \in [1, 3]$  
   - Suppose $j = 2$ → replace B → [A, C]  
3. $i = 4$, item = D  
   - Pick random $j \in [1, 4]$  
   - Suppose $j = 4$ → do nothing → [A, C]  
4. $i = 5$, item = E  
   - Pick random $j \in [1, 5]$  
   - Suppose $j = 1$ → replace A → [E, C]  

Final sample: [E, C]  
Each item A–E has equal probability to appear in the final reservoir.

### Mathematical Intuition

Each element $x_i$ at position $i$ has probability

$$
P(x_i \text{ in final sample}) = \frac{k}{i} \cdot \prod_{j=i+1}^{n} \left(1 - \frac{1}{j}\right) = \frac{k}{n}
$$

Thus, every item is equally likely to be chosen, ensuring perfect uniformity in the final sample.


#### Visualization

| Step | Item | Random $j$ | Action     | Reservoir |
| ---- | ---- | ------------ | ---------- | --------- |
| 1    | A    | –            | Add        | [A]       |
| 2    | B    | –            | Add        | [A, B]    |
| 3    | C    | 2            | Replace B  | [A, C]    |
| 4    | D    | 4            | No Replace | [A, C]    |
| 5    | E    | 1            | Replace A  | [E, C]    |

#### Tiny Code (Easy Version)

Python Implementation

```python
import random

def reservoir_sample(stream, k):
    reservoir = []
    for i, item in enumerate(stream, 1):
        if i <= k:
            reservoir.append(item)
        else:
            j = random.randint(1, i)
            if j <= k:
                reservoir[j - 1] = item
    return reservoir

# Example
stream = ["A", "B", "C", "D", "E"]
sample = reservoir_sample(stream, 2)
print("Reservoir sample:", sample)
```

Output (random):

```
Reservoir sample: ['E', 'C']
```

Each run produces a different uniform random sample.

#### Why It Matters

- Works on streaming data
- Needs only O(k) memory
- Provides uniform unbiased sampling
- Used in:

  * Big data analytics
  * Randomized algorithms
  * Online learning
  * Network monitoring

#### A Gentle Proof (Why It Works)

- First $k$ elements: probability $= 1$ to enter the reservoir initially.  
- Each new element at index $i$: probability $\tfrac{k}{i}$ to replace one of the existing items.  
- Earlier items may be replaced, but each remains with probability

$$
P(\text{survive}) = \prod_{j=i+1}^{n} \left(1 - \frac{1}{j}\right)
$$

Multiplying these terms gives the final inclusion probability $\tfrac{k}{n}$.

Uniformity of selection is guaranteed by induction.


#### Try It Yourself

1. Stream 10 numbers with $k = 3$.  
2. Run the algorithm multiple times — all 3-element subsets appear with roughly equal frequency.  
3. Increase $k$ and observe that the sample becomes more stable, with less variation between runs.


#### Test Cases

| Stream      | k  | Sample Size | Notes                |
| ----------- | -- | ----------- | -------------------- |
| [1,2,3,4,5] | 2  | 2           | Uniform random pairs |
| [A,B,C,D]   | 1  | 1           | Each 25% chance      |
| Range(1000) | 10 | 10          | Works in one pass    |

#### Complexity

| Operation | Time   | Space  |
| ---------- | ------ | ------ |
| Insert     | $O(1)$ | $O(k)$ |
| Query      | $O(1)$ | $O(k)$ |


Reservoir Sampling, elegantly fair, perfectly simple, and ready for infinite streams.

### 270 Skip Bloom Filter

A Skip Bloom Filter is a probabilistic data structure that extends the Bloom Filter to support range queries, determining whether any element exists within a given interval, not just checking for a single item.

It combines Bloom filters with hierarchical range segmentation, allowing approximate range lookups while keeping space usage compact.

#### What Problem Are We Solving?

A classic Bloom Filter answers only point queries:

$$
\text{"Is } x \text{ in the set?"}
$$

However, many real-world applications require range queries, such as:

- Databases: "Are there any keys between 10 and 20?"
- Time-series: "Were there any events during this time interval?"
- Networks: "Is any IP in this subnet?"

We need a space-efficient, stream-friendly, and probabilistic structure that can:

- Handle range membership checks,
- Maintain a low false-positive rate,
- Scale logarithmically with the universe size.

The Skip Bloom Filter solves this by layering Bloom filters over aligned ranges of increasing size.

#### How It Works (Plain Language)

A Skip Bloom Filter maintains multiple Bloom filters, each corresponding to a level that covers intervals (buckets) of sizes $2^0, 2^1, 2^2, \ldots$

Each element is inserted into all Bloom filters representing the ranges that contain it.  
When querying a range, the query is decomposed into aligned subranges that correspond to these levels, and each is checked in its respective filter.

Algorithm:

1. Divide the universe into intervals of size $2^\ell$ for each level $\ell$.  
2. Each level $\ell$ maintains a Bloom filter representing those buckets.  
3. To insert a key $x$: mark all buckets across levels that include $x$.  
4. To query a range $[a,b]$: decompose it into a set of disjoint aligned intervals and check the corresponding Bloom filters.


#### Example (Step by Step)

Suppose we store keys

$$
S = {3, 7, 14}
$$

in a universe ([0, 15]).
We build filters at levels with range sizes (1, 2, 4, 8):

| Level | Bucket Size | Buckets (Ranges)                  |
| ----- | ----------- | --------------------------------- |
| 0     | 1           | [0], [1], [2], ..., [15]          |
| 1     | 2           | [0–1], [2–3], [4–5], ..., [14–15] |
| 2     | 4           | [0–3], [4–7], [8–11], [12–15]     |
| 3     | 8           | [0–7], [8–15]                     |

Insert key 3:

- Level 0: [3]
- Level 1: [2–3]
- Level 2: [0–3]
- Level 3: [0–7]

Insert key 7:

- Level 0: [7]
- Level 1: [6–7]
- Level 2: [4–7]
- Level 3: [0–7]

Insert key 14:

- Level 0: [14]
- Level 1: [14–15]
- Level 2: [12–15]
- Level 3: [8–15]

Query range [2, 6]:

1. Decompose into aligned intervals: ([2–3], [4–5], [6])
2. Check filters:

   * [2–3] → hit
   * [4–5] → miss
   * [6] → miss

Result: possibly non-empty, since [2–3] contains 3.

#### Visualization

| Level | Bucket | Contains Key | Bloom Entry |
| ----- | ------ | ------------ | ----------- |
| 0     | [3]    | Yes          | 1           |
| 1     | [2–3]  | Yes          | 1           |
| 2     | [0–3]  | Yes          | 1           |
| 3     | [0–7]  | Yes          | 1           |

Each key is represented in multiple levels, enabling multi-scale range coverage.

#### Tiny Code (Simplified Python)

```python
import math, mmh3

class Bloom:
    def __init__(self, size=64, hash_count=3):
        self.size = size
        self.hash_count = hash_count
        self.bits = [0] * size

    def _hashes(self, key):
        return [mmh3.hash(str(key), i) % self.size for i in range(self.hash_count)]

    def add(self, key):
        for h in self._hashes(key):
            self.bits[h] = 1

    def query(self, key):
        return all(self.bits[h] for h in self._hashes(key))

class SkipBloom:
    def __init__(self, levels=4, size=64, hash_count=3):
        self.levels = [Bloom(size, hash_count) for _ in range(levels)]

    def add(self, key):
        level = 0
        while (1 << level) <= key:
            bucket = key // (1 << level)
            self.levels[level].add(bucket)
            level += 1

    def query_range(self, start, end):
        l = int(math.log2(end - start + 1))
        bucket = start // (1 << l)
        return self.levels[l].query(bucket)

# Example
sb = SkipBloom(levels=4)
for x in [3, 7, 14]:
    sb.add(x)

print("Query [2,6]:", sb.query_range(2,6))
```

Output:

```
Query [2,6]: True
```

#### Why It Matters

- Enables range queries in probabilistic manner
- Compact and hierarchical
- No false negatives (for properly configured filters)
- Widely applicable in:

  * Approximate database indexing
  * Network prefix search
  * Time-series event detection

#### A Gentle Proof (Why It Works)

Each inserted key participates in $O(\log U)$ Bloom filters, one per level.  
A range query $[a,b]$ is decomposed into $O(\log U)$ aligned subranges.

A Bloom filter with $m$ bits, $k$ hash functions, and $n$ inserted elements has false positive probability:

$$
p = \left(1 - e^{-kn/m}\right)^k
$$

For a Skip Bloom Filter, the total false positive rate is bounded by:

$$
P_{fp} \le O(\log U) \cdot p
$$

Each level guarantees no false negatives, since every range containing an element is marked.  
Thus, correctness is ensured, and only overestimation (false positives) can occur.

Space complexity  
Each level has $m$ bits, and there are $\log U$ levels:

$$
\text{Total space} = O(m \log U)
$$

Time complexity  
Each query checks $O(\log U)$ buckets, each requiring $O(k)$ time:

$$
T_{\text{query}} = O(k \log U)
$$


#### Try It Yourself

1. Insert keys $\{3, 7, 14\}$  
2. Query ranges $[2,6]$, $[8,12]$, $[0,15]$  
3. Compare true contents with the results  
4. Adjust parameters $m$, $k$, or the number of levels, and observe how the false positive rate changes

#### Test Cases

| Query Range | Result | True Contents |
| ------------ | ------- | ------------- |
| [2,6]        | True    | {3}           |
| [8,12]       | False   | ∅             |
| [12,15]      | True    | {14}          |

#### Complexity

| Operation | Time             | Space            |
| ---------- | ---------------- | ---------------- |
| Insert     | $O(\log U)$      | $O(m \log U)$    |
| Query      | $O(\log U)$      | $O(m \log U)$    |

A Skip Bloom Filter is a range-aware extension of standard Bloom filters.  
By combining hierarchical decomposition with standard hashing, it enables fast, memory-efficient, and approximate range queries across very large universes.


## Section 28. Skip Lists and B-Trees 

### 271 Skip List Insert

Skip Lists are probabilistic alternatives to balanced trees. They maintain multiple levels of sorted linked lists, where each level skips over more elements than the one below it. Insertion relies on randomization to achieve expected O(log n) search and update times, without strict rebalancing like AVL or Red-Black trees.

#### What Problem Are We Solving?

We want to store elements in sorted order and support:

- Fast search, insert, and delete operations.
- Simple structure and easy implementation.
- Expected logarithmic performance without complex rotations.

Balanced BSTs achieve $O(\log n)$ time but require intricate rotations.
Skip Lists solve this with *randomized promotion*: each inserted node is promoted to higher levels with decreasing probability, forming a tower.

#### How It Works (Plain Language)

Skip list insertion for value $x$

Steps
1. Start at the top level.
2. While the next node’s key < $x$, move right.
3. If you cannot move right, drop down one level.
4. Repeat until you reach level 0.
5. Insert the new node at its sorted position on level 0.
6. Randomly choose the node height $h$ (for example, flip a fair coin per level until tails, so $P(h \ge t) = 2^{-t}$).
7. For each level $1 \dots h$, link the new node into that level by repeating the right-then-drop search with the update pointers saved from the descent.


Notes
- Duplicate handling is policy dependent. Often you skip insertion if a node with key $x$ already exists.
- Typical height is $O(\log n)$, expected search and insert time is $O(\log n)$, space is $O(n)$.


#### Example Step by Step

Let's insert $x = 17$ into a skip list that currently contains: [ 5, 10, 15, 20, 25 ]

| Level | Nodes (before insertion) | Path Taken                         |
| ----- | ------------------------ | ---------------------------------- |
| 3     | 5 → 15 → 25              | Move from 5 to 15, then down       |
| 2     | 5 → 10 → 15 → 25         | Move from 5 to 10 to 15, then down |
| 1     | 5 → 10 → 15 → 20 → 25    | Move from 15 to 20, then down      |
| 0     | 5 → 10 → 15 → 20 → 25    | Insert after 15                    |

Suppose the random level for 17 is 2.
We insert 17 at level 0 and level 1.

| Level | Nodes (after insertion)    |
| ----- | -------------------------- |
| 3     | 5 → 15 → 25                |
| 2     | 5 → 10 → 15 → 25           |
| 1     | 5 → 10 → 15 → 17 → 20 → 25 |
| 0     | 5 → 10 → 15 → 17 → 20 → 25 |

#### Tiny Code (Simplified Python)

```python
import random

class Node:
    def __init__(self, key, level):
        self.key = key
        self.forward = [None] * (level + 1)

class SkipList:
    def __init__(self, max_level=4, p=0.5):
        self.max_level = max_level
        self.p = p
        self.header = Node(-1, max_level)
        self.level = 0

    def random_level(self):
        lvl = 0
        while random.random() < self.p and lvl < self.max_level:
            lvl += 1
        return lvl

    def insert(self, key):
        update = [None] * (self.max_level + 1)
        current = self.header

        # Move right and down
        for i in reversed(range(self.level + 1)):
            while current.forward[i] and current.forward[i].key < key:
                current = current.forward[i]
            update[i] = current

        current = current.forward[0]
        if current is None or current.key != key:
            lvl = self.random_level()
            if lvl > self.level:
                for i in range(self.level + 1, lvl + 1):
                    update[i] = self.header
                self.level = lvl
            new_node = Node(key, lvl)
            for i in range(lvl + 1):
                new_node.forward[i] = update[i].forward[i]
                update[i].forward[i] = new_node
```

#### Why It Matters

- Expected $O(\log n)$ time for search, insert, and delete  
- Simpler than AVL or Red-Black Trees  
- Probabilistic balancing avoids rigid rotations  
- Commonly used in databases and key-value stores such as LevelDB and Redis

#### A Gentle Proof (Why It Works)

Each node appears in level $i$ with probability $p^i$.  
The expected number of nodes per level is $n p^i$.  
The total number of levels is $O(\log_{1/p} n)$.

Expected search path length:

$$
E[\text{steps}] = \frac{1}{1 - p} \log_{1/p} n = O(\log n)
$$

Expected space usage:

$$
O(n) \text{ nodes} \times O\!\left(\frac{1}{1 - p}\right) \text{ pointers per node}
$$

Thus, Skip Lists achieve expected logarithmic performance and linear space.

#### Try It Yourself

1. Build a Skip List and insert $\{5, 10, 15, 20, 25\}$.  
2. Insert $17$ and trace which pointers are updated at each level.  
3. Experiment with $p = 0.25, 0.5, 0.75$.  
4. Observe how random heights influence the overall balance.


#### Test Cases

| Operation | Input | Expected Structure (Level 0) |
| --------- | ----- | ---------------------------- |
| Insert    | 10    | 10                           |
| Insert    | 5     | 5 → 10                       |
| Insert    | 15    | 5 → 10 → 15                  |
| Insert    | 17    | 5 → 10 → 15 → 17             |
| Search    | 15    | Found                        |
| Search    | 12    | Not Found                    |

#### Complexity

| Operation | Time (Expected) | Space   |
| ---------- | --------------- | ------- |
| Search     | $O(\log n)$     | $O(n)$  |
| Insert     | $O(\log n)$     | $O(n)$  |
| Delete     | $O(\log n)$     | $O(n)$  |


A Skip List is a simple yet powerful data structure.
With randomness as its balancing force, it achieves the elegance of trees and the flexibility of linked lists.

### 272 Skip List Delete

Deletion in a Skip List mirrors insertion: we traverse levels from top to bottom, keep track of predecessor nodes at each level, and unlink the target node across all levels it appears in.
The structure maintains probabilistic balance, so no rebalancing is needed, deletion is expected $O(\log n)$.

#### What Problem Are We Solving?

We want to remove an element efficiently from a sorted, probabilistically balanced structure.
Naive linked lists require $O(n)$ traversal; balanced BSTs need complex rotations.
A Skip List gives us a middle ground, simple pointer updates with expected logarithmic time.

#### How It Works (Plain Language)

Each node in a skip list can appear at multiple levels.
To delete a key $x$:

1. Start from the top level.
2. Move right while next node's key < $x$.
3. If next node's key == $x$, record current node in an `update` array.
4. Drop one level down and repeat.
5. Once you reach the bottom, remove all forward references to the node from the `update` array.
6. If the topmost level becomes empty, reduce list level.

#### Example Step by Step

Delete $x = 17$ from this skip list:

| Level | Nodes (Before)             |
| ----- | -------------------------- |
| 3     | 5 → 15 → 25                |
| 2     | 5 → 10 → 15 → 25           |
| 1     | 5 → 10 → 15 → 17 → 20 → 25 |
| 0     | 5 → 10 → 15 → 17 → 20 → 25 |

Traversal:

- Start at Level 3: 15 < 17 → move right → 25 > 17 → drop down
- Level 2: 15 < 17 → move right → 25 > 17 → drop down
- Level 1: 15 < 17 → move right → 17 found → record predecessor
- Level 0: 15 < 17 → move right → 17 found → record predecessor

Remove all forward pointers to 17 from recorded nodes.

| Level | Nodes (After)         |
| ----- | --------------------- |
| 3     | 5 → 15 → 25           |
| 2     | 5 → 10 → 15 → 25      |
| 1     | 5 → 10 → 15 → 20 → 25 |
| 0     | 5 → 10 → 15 → 20 → 25 |

#### Tiny Code (Simplified Python)

```python
import random

class Node:
    def __init__(self, key, level):
        self.key = key
        self.forward = [None] * (level + 1)

class SkipList:
    def __init__(self, max_level=4, p=0.5):
        self.max_level = max_level
        self.p = p
        self.header = Node(-1, max_level)
        self.level = 0

    def delete(self, key):
        update = [None] * (self.max_level + 1)
        current = self.header

        # Traverse from top to bottom
        for i in reversed(range(self.level + 1)):
            while current.forward[i] and current.forward[i].key < key:
                current = current.forward[i]
            update[i] = current

        current = current.forward[0]

        # Found the node
        if current and current.key == key:
            for i in range(self.level + 1):
                if update[i].forward[i] != current:
                    continue
                update[i].forward[i] = current.forward[i]
            # Reduce level if highest level empty
            while self.level > 0 and self.header.forward[self.level] is None:
                self.level -= 1
```

#### Why It Matters

- Symmetric to insertion
- No rotations or rebalancing
- Expected $O(\log n)$ performance
- Perfect for ordered maps, databases, key-value stores

#### A Gentle Proof (Why It Works)

Each level contains a fraction $p^i$ of nodes.
The expected number of levels traversed is $O(\log_{1/p} n)$.
At each level, we move horizontally $O(1)$ on average.

So expected cost:

$$
E[T_{\text{delete}}] = O(\log n)
$$

#### Try It Yourself

1. Insert ${5, 10, 15, 17, 20, 25}$.
2. Delete $17$.
3. Trace all pointer changes level by level.
4. Compare with AVL tree deletion complexity.

#### Test Cases

| Operation | Input         | Expected Level 0 Result |
| --------- | ------------- | ----------------------- |
| Insert    | 5,10,15,17,20 | 5 → 10 → 15 → 17 → 20   |
| Delete    | 17            | 5 → 10 → 15 → 20        |
| Delete    | 10            | 5 → 15 → 20             |
| Delete    | 5             | 15 → 20                 |

#### Complexity

| Operation | Time (Expected) | Space  |
| --------- | --------------- | ------ |
| Search    | $O(\log n)$     | $O(n)$ |
| Insert    | $O(\log n)$     | $O(n)$ |
| Delete    | $O(\log n)$     | $O(n)$ |

Skip List Deletion keeps elegance through simplicity, a clean pointer adjustment instead of tree surgery.

### 273 Skip List Search

Searching in a Skip List is a dance across levels, we move right until we can't, then down, repeating until we either find the key or conclude it doesn't exist.
Thanks to the randomized level structure, the expected time complexity is $O(\log n)$, just like balanced BSTs but with simpler pointer logic.

#### What Problem Are We Solving?

We need a fast search in a sorted collection that adapts gracefully to dynamic insertions and deletions.
Balanced trees guarantee $O(\log n)$ search but need rotations.
Skip Lists achieve the same expected time using randomization instead of strict balancing rules.

#### How It Works (Plain Language)

A skip list has multiple levels of linked lists.
Each level acts as a fast lane, skipping over multiple nodes.
To search for a key $x$:

1. Start at the top-left header node.
2. At each level, move right while `next.key < x`.
3. When `next.key ≥ x`, drop down one level.
4. Repeat until level 0.
5. If `current.forward[0].key == x`, found; else, not found.

The search path "zigzags" through levels, visiting roughly $\log n$ nodes on average.

#### Example Step by Step

Search for $x = 17$ in the following skip list:

| Level | Nodes                      |
| ----- | -------------------------- |
| 3     | 5 → 15 → 25                |
| 2     | 5 → 10 → 15 → 25           |
| 1     | 5 → 10 → 15 → 17 → 20 → 25 |
| 0     | 5 → 10 → 15 → 17 → 20 → 25 |

Traversal:

- Level 3: 5 → 15 → (next = 25 > 17) → drop down
- Level 2: 15 → (next = 25 > 17) → drop down
- Level 1: 15 → 17 found → success
- Level 0: confirm 17 exists

Path: 5 → 15 → (down) → 15 → (down) → 15 → 17

#### Visualization

Skip list search follows a staircase pattern:

```
Level 3:  5 --------> 15 -----↓
Level 2:  5 ----> 10 --> 15 --↓
Level 1:  5 -> 10 -> 15 -> 17 -> 20
Level 0:  5 -> 10 -> 15 -> 17 -> 20
```

Each "↓" means dropping a level when the next node is too large.

#### Tiny Code (Simplified Python)

```python
class SkipList:
    def __init__(self, max_level=4, p=0.5):
        self.max_level = max_level
        self.p = p
        self.header = Node(-1, max_level)
        self.level = 0

    def search(self, key):
        current = self.header
        # Traverse top-down
        for i in reversed(range(self.level + 1)):
            while current.forward[i] and current.forward[i].key < key:
                current = current.forward[i]
        current = current.forward[0]
        return current and current.key == key
```

#### Why It Matters

- Simple and efficient: expected $O(\log n)$ time
- Probabilistic balance: avoids tree rotations
- Foundation for ordered maps, indexes, and databases
- Search path length is logarithmic on average

#### A Gentle Proof (Why It Works)

Each level contains approximately a fraction $p$ of the nodes from the level below.
Expected number of levels: $O(\log_{1/p} n)$.

At each level, expected number of horizontal moves: $O(1)$.

So total expected search time:

$$
E[T_{\text{search}}] = O(\log n)
$$

#### Try It Yourself

1. Build a skip list with ${5, 10, 15, 17, 20, 25}$.
2. Search for $17$ and trace the path at each level.
3. Search for $13$, where do you stop?
4. Compare path length with a binary search tree of same size.

#### Test Cases

| Operation | Input | Expected Output | Path        |
| --------- | ----- | --------------- | ----------- |
| Search    | 17    | Found           | 5 → 15 → 17 |
| Search    | 10    | Found           | 5 → 10      |
| Search    | 13    | Not Found       | 5 → 10 → 15 |
| Search    | 5     | Found           | 5           |

#### Complexity

| Operation | Time (Expected) | Space  |
| --------- | --------------- | ------ |
| Search    | $O(\log n)$     | $O(n)$ |
| Insert    | $O(\log n)$     | $O(n)$ |
| Delete    | $O(\log n)$     | $O(n)$ |

Skip List Search shows how probabilistic structure yields deterministic-like efficiency, walking a staircase of randomness toward certainty.

### 274 B-Tree Insert

A B-Tree is a balanced search tree designed for external memory systems such as disks or SSDs. Unlike binary trees, each node can store multiple keys and multiple children, minimizing disk I/O by packing more data into a single node.
Insertion into a B-Tree preserves sorted order and balance by splitting full nodes as needed.

#### What Problem Are We Solving?

When data is too large to fit in memory, standard binary trees perform poorly because each node access may trigger a disk read.
We need a structure that:

- Reduces the number of I/O operations
- Keeps height small
- Maintains keys in sorted order
- Supports search, insert, and delete in $O(\log n)$

B-Trees solve this by storing many keys per node and balancing themselves through controlled splits.

#### How It Works (Plain Language)

Each B-Tree node can contain up to $2t - 1$ keys and $2t$ children, where $t$ is the minimum degree.

Insertion steps:

1. Start at root and traverse down like in binary search.
2. If a child node is full (2t − 1 keys), split it before descending.
3. Insert the new key into the appropriate non-full node.

Splitting a full node:

- Middle key moves up to parent
- Left and right halves become separate child nodes

This ensures every node stays within allowed size bounds, keeping height $O(\log_t n)$.

#### Example Step by Step

Let $t = 2$ (max 3 keys per node).
Insert keys in order: $[10, 20, 5, 6, 12, 30, 7, 17]$

Step 1: Insert 10 → Root = [10]
Step 2: Insert 20 → [10, 20]
Step 3: Insert 5 → [5, 10, 20]
Step 4: Insert 6 → Node full → Split

Split [5, 6, 10, 20]:

- Middle key 10 moves up
- Left child [5, 6], right child [20]
  Tree:

```
      [10]
     /    \
 [5, 6]   [20]
```

Step 5: Insert 12 → go right → [12, 20]
Step 6: Insert 30 → [12, 20, 30]
Step 7: Insert 7 → go left → [5, 6, 7] → full → split

- Middle 6 moves up

Tree now:

```
        [6, 10]
       /   |    \
 [5]  [7]  [12, 20, 30]
```

Step 8: Insert 17 → go to [12, 20, 30] → insert [12, 17, 20, 30] → split

- Middle 20 moves up

Final tree:

```
          [6, 10, 20]
         /   |   |   \
       [5] [7] [12,17] [30]
```

#### Visualization

B-Tree maintains sorted keys at each level and guarantees minimal height by splitting nodes during insertion.

```
Root: [6, 10, 20]
Children: [5], [7], [12, 17], [30]
```

#### Tiny Code (Simplified Python)

```python
class BTreeNode:
    def __init__(self, t, leaf=False):
        self.keys = []
        self.children = []
        self.leaf = leaf
        self.t = t

    def insert_non_full(self, key):
        i = len(self.keys) - 1
        if self.leaf:
            self.keys.append(key)
            self.keys.sort()
        else:
            while i >= 0 and key < self.keys[i]:
                i -= 1
            i += 1
            if len(self.children[i].keys) == 2 * self.t - 1:
                self.split_child(i)
                if key > self.keys[i]:
                    i += 1
            self.children[i].insert_non_full(key)

    def split_child(self, i):
        t = self.t
        y = self.children[i]
        z = BTreeNode(t, y.leaf)
        mid = y.keys[t - 1]
        z.keys = y.keys[t:]
        y.keys = y.keys[:t - 1]
        if not y.leaf:
            z.children = y.children[t:]
            y.children = y.children[:t]
        self.children.insert(i + 1, z)
        self.keys.insert(i, mid)

class BTree:
    def __init__(self, t):
        self.root = BTreeNode(t, True)
        self.t = t

    def insert(self, key):
        r = self.root
        if len(r.keys) == 2 * self.t - 1:
            s = BTreeNode(self.t)
            s.children.insert(0, r)
            s.split_child(0)
            i = 0
            if key > s.keys[0]:
                i += 1
            s.children[i].insert_non_full(key)
            self.root = s
        else:
            r.insert_non_full(key)
```

#### Why It Matters

- Disk-friendly: each node fits into one page
- Shallow height: $O(\log_t n)$ levels → few disk reads
- Deterministic balance: no randomness, always balanced
- Foundation of file systems, databases, indexes (e.g., NTFS, MySQL, PostgreSQL)

#### A Gentle Proof (Why It Works)

Each node has between $t-1$ and $2t-1$ keys (except root).

Each split increases height only when the root splits.

Thus, height $h$ satisfies:

$$
t^h \le n \le (2t)^h
$$

Taking logs:

$$
h = O(\log_t n)
$$

So insertions and searches take $O(t \cdot \log_t n)$, often simplified to $O(\log n)$ when $t$ is constant.

#### Try It Yourself

1. Build a B-Tree with $t=2$.
2. Insert $[10, 20, 5, 6, 12, 30, 7, 17]$.
3. Draw the tree after each insertion.
4. Observe when splits occur and which keys promote upward.

#### Test Cases

| Input Keys             | t | Final Root | Height |
| ---------------------- | - | ---------- | ------ |
| [10,20,5,6,12,30,7,17] | 2 | [6,10,20]  | 2      |
| [1,2,3,4,5,6,7,8,9]    | 2 | [4]        | 3      |

#### Complexity

| Operation | Time        | Space  |
| --------- | ----------- | ------ |
| Search    | $O(\log n)$ | $O(n)$ |
| Insert    | $O(\log n)$ | $O(n)$ |
| Delete    | $O(\log n)$ | $O(n)$ |

B-Tree insertion is the heartbeat of external memory algorithms, split, promote, balance, ensuring data stays close, shallow, and sorted.

### 275 B-Tree Delete

Deletion in a B-Tree is more intricate than insertion, we must carefully remove a key while preserving the B-Tree's balance properties. Every node must maintain at least $t - 1$ keys (except the root), so deletion may involve borrowing from siblings or merging nodes.

The goal is to maintain the B-Tree invariants:

- Keys sorted within each node
- Node key count between $t - 1$ and $2t - 1$
- Balanced height

#### What Problem Are We Solving?

We want to remove a key from a B-Tree without violating balance or occupancy constraints.
Unlike binary search trees, where we can simply replace or prune nodes, B-Trees must maintain minimum degree to ensure consistent height and I/O efficiency.

#### How It Works (Plain Language)

To delete a key $k$ from a B-Tree:

1. If $k$ is in a leaf node:

   * Simply remove it.

2. If $k$ is in an internal node:

   * Case A: If left child has ≥ $t$ keys → replace $k$ with predecessor.
   * Case B: Else if right child has ≥ $t$ keys → replace $k$ with successor.
   * Case C: Else both children have $t - 1$ keys → merge them and recurse.

3. If $k$ is not in the current node:

   * Move to the correct child.
   * Before descending, ensure the child has at least $t$ keys. If not,

     * Borrow from a sibling with ≥ $t$ keys, or
     * Merge with a sibling to guarantee occupancy.

This ensures no underflow occurs during traversal.

#### Example Step by Step

Let $t = 2$ (max 3 keys per node).
B-Tree before deletion:

```
          [6, 10, 20]
         /    |     |    \
       [5]  [7]  [12,17] [30]
```

#### Delete key 17

- 17 is in leaf [12, 17] → remove it directly

Result:

```
          [6, 10, 20]
         /    |     |    \
       [5]  [7]   [12]   [30]
```

#### Delete key 10

- 10 is in internal node
- Left child [7] has $t - 1 = 1$ key
- Right child [12] also has 1 key
  → Merge [7], 10, [12] → [7, 10, 12]

Tree becomes:

```
       [6, 20]
      /    |    \
    [5] [7,10,12] [30]
```

#### Delete key 6

- 6 in internal node
- Left [5] has 1 key, right [7,10,12] has ≥ 2 → borrow from right
- Replace 6 with successor 7

Tree after rebalancing:

```
       [7, 20]
      /    |    \
    [5,6] [10,12] [30]
```

#### Visualization

Every deletion keeps the tree balanced by ensuring all nodes (except root) stay ≥ $t - 1$ full.

```
Before:             After Deleting 10:
$$6,10,20]           [6,20]
 / | | \             / | \
$$5][7][12,17][30]   [5][7,10,12][30]
```

#### Tiny Code (Simplified Python)

```python
class BTreeNode:
    def __init__(self, t, leaf=False):
        self.t = t
        self.keys = []
        self.children = []
        self.leaf = leaf

    def find_key(self, k):
        for i, key in enumerate(self.keys):
            if key >= k:
                return i
        return len(self.keys)

class BTree:
    def __init__(self, t):
        self.root = BTreeNode(t, True)
        self.t = t

    def delete(self, node, k):
        t = self.t
        i = node.find_key(k)

        # Case 1: key in node
        if i < len(node.keys) and node.keys[i] == k:
            if node.leaf:
                node.keys.pop(i)
            else:
                if len(node.children[i].keys) >= t:
                    pred = self.get_predecessor(node, i)
                    node.keys[i] = pred
                    self.delete(node.children[i], pred)
                elif len(node.children[i+1].keys) >= t:
                    succ = self.get_successor(node, i)
                    node.keys[i] = succ
                    self.delete(node.children[i+1], succ)
                else:
                    self.merge(node, i)
                    self.delete(node.children[i], k)
        else:
            # Case 2: key not in node
            if node.leaf:
                return  # not found
            if len(node.children[i].keys) < t:
                self.fill(node, i)
            self.delete(node.children[i], k)
```

*(Helper methods `merge`, `fill`, `borrow_from_prev`, and `borrow_from_next` omitted for brevity)*

#### Why It Matters

- Maintains balanced height after deletions
- Prevents underflow in child nodes
- Ensures $O(\log n)$ complexity
- Used heavily in databases and filesystems where stable performance is critical

#### A Gentle Proof (Why It Works)

A B-Tree node always satisfies:

$$
t - 1 \le \text{keys per node} \le 2t - 1
$$

Merging or borrowing ensures all nodes remain within bounds.
The height $h$ satisfies:

$$
h \le \log_t n
$$

So deletions require visiting at most $O(\log_t n)$ nodes and performing constant-time merges/borrows per level.

Hence:

$$
T_{\text{delete}} = O(\log n)
$$

#### Try It Yourself

1. Build a B-Tree with $t = 2$.
2. Insert $[10, 20, 5, 6, 12, 30, 7, 17]$.
3. Delete keys in order: $17, 10, 6$.
4. Draw the tree after each deletion and observe merges/borrows.

#### Test Cases

| Input Keys             | Delete | Result (Level 0)    |
| ---------------------- | ------ | ------------------- |
| [5,6,7,10,12,17,20,30] | 17     | [5,6,7,10,12,20,30] |
| [5,6,7,10,12,20,30]    | 10     | [5,6,7,12,20,30]    |
| [5,6,7,12,20,30]       | 6      | [5,7,12,20,30]      |

#### Complexity

| Operation | Time        | Space  |
| --------- | ----------- | ------ |
| Search    | $O(\log n)$ | $O(n)$ |
| Insert    | $O(\log n)$ | $O(n)$ |
| Delete    | $O(\log n)$ | $O(n)$ |

B-Tree deletion is a surgical balancing act, merging, borrowing, and promoting keys just enough to keep the tree compact, shallow, and sorted.

### 276 B+ Tree Search

A B+ Tree is an extension of the B-Tree, optimized for range queries and sequential access. All actual data (records or values) reside in leaf nodes, which are linked together to form a sorted list. Internal nodes contain only keys that guide the search.

Searching in a B+ Tree follows the same principle as a B-Tree, top-down traversal based on key comparisons, but ends at the leaf level where the actual data is stored.

#### What Problem Are We Solving?

We need a disk-friendly search structure that:

- Keeps height small (few disk I/Os)
- Supports fast range scans
- Separates index keys (internal nodes) from records (leaves)

B+ Trees meet these needs with:

- High fan-out: many keys per node
- Linked leaves: for efficient sequential traversal
- Deterministic balance: height always $O(\log n)$

#### How It Works (Plain Language)

Each internal node acts as a router. Each leaf node contains keys + data pointers.

To search for key $k$:

1. Start at the root.
2. At each internal node, find the child whose key range contains $k$.
3. Follow that pointer to the next level.
4. Continue until reaching a leaf node.
5. Perform a linear scan within the leaf to find $k$.

If not found in the leaf, $k$ is not in the tree.

#### Example Step by Step

Let $t = 2$ (each node holds up to 3 keys).
B+ Tree:

```
          [10 | 20]
         /     |     \
   [1 5 8]  [12 15 18]  [22 25 30]
```

Search for 15:

- Root [10 | 20]: 15 > 10 and < 20 → follow middle pointer
- Node [12 15 18]: found 15

Search for 17:

- Root [10 | 20]: 17 > 10 and < 20 → middle pointer
- Node [12 15 18]: not found → not in tree

#### Visualization

```
         [10 | 20]
        /     |     \
 [1 5 8] [12 15 18] [22 25 30]
```

- Internal nodes guide the path
- Leaf nodes hold data (and link to next leaf)
- Search always ends at a leaf

#### Tiny Code (Simplified Python)

```python
class BPlusNode:
    def __init__(self, t, leaf=False):
        self.t = t
        self.leaf = leaf
        self.keys = []
        self.children = []
        self.next = None  # link to next leaf

class BPlusTree:
    def __init__(self, t):
        self.root = BPlusNode(t, True)
        self.t = t

    def search(self, node, key):
        if node.leaf:
            return key in node.keys
        i = 0
        while i < len(node.keys) and key >= node.keys[i]:
            i += 1
        return self.search(node.children[i], key)

    def find(self, key):
        return self.search(self.root, key)
```

#### Why It Matters

- Efficient disk I/O: high branching factor keeps height low
- All data in leaves: simplifies range queries
- Linked leaves: enable sequential traversal (sorted order)
- Used in: databases, filesystems, key-value stores (e.g., MySQL, InnoDB, NTFS)

#### A Gentle Proof (Why It Works)

Each internal node has between $t$ and $2t$ children.
Each leaf holds between $t - 1$ and $2t - 1$ keys.
Thus, height $h$ satisfies:

$$
t^h \le n \le (2t)^h
$$

Taking logs:

$$
h = O(\log_t n)
$$

So search time is:

$$
T_{\text{search}} = O(\log n)
$$

And since the final scan within the leaf is constant (small), total cost remains logarithmic.

#### Try It Yourself

1. Build a B+ Tree with $t = 2$ and insert keys $[1, 5, 8, 10, 12, 15, 18, 20, 22, 25, 30]$.
2. Search for 15, 17, and 8.
3. Trace your path from root → internal → leaf.
4. Observe that all searches end in leaves.

#### Test Cases

| Search Key | Expected Result | Path                 |
| ---------- | --------------- | -------------------- |
| 15         | Found           | Root → Middle → Leaf |
| 8          | Found           | Root → Left → Leaf   |
| 17         | Not Found       | Root → Middle → Leaf |
| 25         | Found           | Root → Right → Leaf  |

#### Complexity

| Operation   | Time            | Space  |
| ----------- | --------------- | ------ |
| Search      | $O(\log n)$     | $O(n)$ |
| Insert      | $O(\log n)$     | $O(n)$ |
| Range Query | $O(\log n + k)$ | $O(n)$ |

B+ Tree Search exemplifies I/O-aware design, every pointer followed is a disk page, every leaf scan is cache-friendly, and every key lives exactly where range queries want it.

### 278 B* Tree

A B* Tree is a refined version of the B-Tree, designed to achieve higher node occupancy and fewer splits. It enforces that each node (except root) must be at least two-thirds full, compared to the half-full guarantee in a standard B-Tree.

To achieve this, B* Trees use redistribution between siblings before splitting, which improves space utilization and I/O efficiency, making them ideal for database and file system indexes.

#### What Problem Are We Solving?

In a standard B-Tree, each node maintains at least $t - 1$ keys (50% occupancy). But frequent splits can cause fragmentation and wasted space.

We want to:

- Increase space efficiency (reduce empty slots)
- Defer splitting when possible
- Maintain balance and sorted order

B* Trees solve this by borrowing and redistributing keys between siblings before splitting, ensuring $\ge 2/3$ occupancy.

#### How It Works (Plain Language)

A B* Tree works like a B-Tree but with smarter split logic:

1. Insertion path:

   * Traverse top-down to find target leaf.
   * If the target node is full, check its sibling.

2. Redistribution step:

   * If sibling has room, redistribute keys between them and the parent key.

3. Double split step:

   * If both siblings are full, split both into three nodes (two full + one new node),
     redistributing keys evenly among them.

This ensures every node (except root) is at least 2/3 full, leading to better disk utilization.

#### Example Step by Step

Let $t = 2$ (max 3 keys per node).
Insert keys: $[5, 10, 15, 20, 25, 30, 35]$

Step 1–4: Build like B-Tree until root [10, 20].

```
        [10 | 20]
       /    |    \
   [5]   [15]   [25, 30, 35]
```

Now insert 40 → rightmost node [25, 30, 35] is full.

- Check sibling: left sibling [15] has room → redistribute keys
  Combine [15], [25, 30, 35], and parent key 20 → [15, 20, 25, 30, 35]
  Split into three nodes evenly: [15, 20], [25, 30], [35]
  Parent updated with new separator.

Result:

```
         [20 | 30]
        /     |     \
    [5,10,15] [20,25] [30,35,40]
```

Each node ≥ 2/3 full, no wasted space.

#### Visualization

```
          [20 | 30]
         /     |     \
 [5 10 15] [20 25] [30 35 40]
```

Redistribution ensures balance and density before splitting.

#### Tiny Code (Simplified Pseudocode)

```python
def insert_bstar(tree, key):
    node = find_leaf(tree.root, key)
    if node.full():
        sibling = node.get_sibling()
        if sibling and not sibling.full():
            redistribute(node, sibling, tree.parent(node))
        else:
            split_three(node, sibling, tree.parent(node))
    insert_into_leaf(node, key)
```

*(Actual implementation is more complex, involving parent updates and sibling pointers.)*

#### Why It Matters

- Better space utilization: nodes ≥ 66% full
- Fewer splits: more stable performance under heavy inserts
- Improved I/O locality: fewer disk blocks accessed
- Used in: database systems (IBM DB2), file systems (ReiserFS), B*-based caching structures

#### A Gentle Proof (Why It Works)

In B* Trees:

- Each non-root node contains $\ge \frac{2}{3} (2t - 1)$ keys.
- Height $h$ satisfies:

$$
\left( \frac{3}{2} t \right)^h \le n
$$

Taking logs:

$$
h = O(\log_t n)
$$

Thus, height remains logarithmic, but nodes pack more data per level.

Fewer levels → fewer I/Os → better performance.

#### Try It Yourself

1. Build a B* Tree with $t = 2$.
2. Insert keys: $[5, 10, 15, 20, 25, 30, 35, 40]$.
3. Watch how redistribution occurs before splits.
4. Compare with B-Tree splits for same sequence.

#### Test Cases

| Input Keys            | t | Result                   | Notes                       |
| --------------------- | - | ------------------------ | --------------------------- |
| [5,10,15,20,25]       | 2 | Root [15]                | Full utilization            |
| [5,10,15,20,25,30,35] | 2 | Root [20,30]             | Redistribution before split |
| [1..15]               | 2 | Balanced, 2/3 full nodes | High density                |

#### Complexity

| Operation | Time        | Space  | Occupancy |
| --------- | ----------- | ------ | --------- |
| Search    | $O(\log n)$ | $O(n)$ | $\ge 66%$ |
| Insert    | $O(\log n)$ | $O(n)$ | $\ge 66%$ |
| Delete    | $O(\log n)$ | $O(n)$ | $\ge 66%$ |

B* Trees take the elegance of B-Trees and push them closer to perfect, fewer splits, denser nodes, and smoother scaling for large datasets.

### 279 Adaptive Radix Tree

An Adaptive Radix Tree (ART) is a space-efficient, cache-friendly data structure that combines ideas from tries and radix trees. It dynamically adapts its node representation based on the number of children, optimizing both memory usage and lookup speed.

Unlike a fixed-size radix tree (which wastes space with sparse nodes), ART chooses a compact node type (like Node4, Node16, Node48, Node256) depending on occupancy, growing as needed.

#### What Problem Are We Solving?

Standard tries and radix trees are fast but memory-heavy.
If keys share long prefixes, many nodes hold only one child, wasting memory.

We want a structure that:

- Keeps O(L) lookup time (L = key length)
- Adapts node size to occupancy
- Minimizes pointer overhead
- Exploits cache locality

ART achieves this by dynamically switching node types as the number of children grows.

#### How It Works (Plain Language)

Each internal node in ART can be one of four types:

| Node Type | Capacity     | Description                      |
| --------- | ------------ | -------------------------------- |
| Node4     | 4 children   | smallest, uses linear search     |
| Node16    | 16 children  | small array, vectorized search   |
| Node48    | 48 children  | index map, stores child pointers |
| Node256   | 256 children | direct addressing by byte value  |

Keys are processed byte by byte, branching at each level.
When a node fills beyond its capacity, it upgrades to the next node type.

#### Example

Insert keys: `["A", "AB", "AC", "AD", "AE"]`

1. Start with root Node4 (can store 4 children).
2. After inserting "AE", Node4 exceeds capacity → upgrade to Node16.
3. Children remain in sorted order by key byte.

This adaptive upgrade keeps nodes dense and efficient.

#### Example Step by Step

| Step | Operation   | Node Type         | Keys Stored       | Note            |
| ---- | ----------- | ----------------- | ----------------- | --------------- |
| 1    | Insert "A"  | Node4             | A                 | Create root     |
| 2    | Insert "AB" | Node4             | A, AB             | Add branch      |
| 3    | Insert "AC" | Node4             | A, AB, AC         | Still under 4   |
| 4    | Insert "AD" | Node4             | A, AB, AC, AD     | Full            |
| 5    | Insert "AE" | Upgrade to Node16 | A, AB, AC, AD, AE | Adaptive growth |

#### Visualization

```
Root (Node16)
 ├── 'A' → Node
      ├── 'B' (Leaf)
      ├── 'C' (Leaf)
      ├── 'D' (Leaf)
      └── 'E' (Leaf)
```

Each node type adapts its layout for the best performance.

#### Tiny Code (Simplified Pseudocode)

```python
class Node:
    def __init__(self):
        self.children = {}

def insert_art(root, key):
    node = root
    for byte in key:
        if byte not in node.children:
            node.children[byte] = Node()
        node = node.children[byte]
    node.value = True
```

*(A real ART dynamically switches between Node4, Node16, Node48, Node256 representations.)*

#### Why It Matters

- Adaptive memory use, no wasted space for sparse nodes
- Cache-friendly, contiguous memory layout
- Fast lookups, vectorized search for Node16
- Used in modern databases (e.g., HyPer, Umbra, DuckDB)

#### A Gentle Proof (Why It Works)

Let $L$ = key length, $b$ = branching factor (max 256 per byte).
In a naive trie, each node allocates $O(b)$ slots, many unused.

In ART:

- Each node stores only actual children, so
  $$
  \text{space} \approx O(n + L)
  $$
- Lookup remains $O(L)$ since we traverse one node per byte.
- Space improves by factor proportional to sparsity.

Thus ART maintains trie-like performance with hash table-like compactness.

#### Try It Yourself

1. Insert `["dog", "dot", "door", "dorm"]`
2. Observe how Node4 → Node16 transitions happen
3. Count number of nodes, compare with naive trie
4. Measure memory usage and access speed

#### Test Cases

| Keys                         | Resulting Root Type | Notes                    |
| ---------------------------- | ------------------- | ------------------------ |
| `["a", "b"]`                 | Node4               | 2 children               |
| `["a", "b", "c", "d", "e"]`  | Node16              | Upgrade after 5th insert |
| `["aa", "ab", "ac"... "az"]` | Node48 or Node256   | Dense branching          |

#### Complexity

| Operation | Time   | Space      | Adaptive Behavior   |
| --------- | ------ | ---------- | ------------------- |
| Search    | $O(L)$ | $O(n + L)$ | Node grows/shrinks  |
| Insert    | $O(L)$ | $O(n)$     | Node type upgrade   |
| Delete    | $O(L)$ | $O(n)$     | Downgrade if sparse |

An Adaptive Radix Tree gives the best of both worlds: prefix compression of tries and space efficiency of hash maps, a modern weapon for high-performance indexing.

### 280 Trie Compression

A compressed trie (also called a radix tree or Patricia trie) is an optimized form of a trie where chains of single-child nodes are merged into a single edge. Instead of storing one character per node, each edge can hold an entire substring.

This reduces the height of the trie, minimizes memory usage, and accelerates searches, perfect for applications like prefix lookup, routing tables, and dictionary storage.

#### What Problem Are We Solving?

A naive trie wastes space when many nodes have only one child.

For example, inserting `["cat", "car", "dog"]` into a naive trie yields long, skinny paths:

```
c → a → t  
c → a → r  
d → o → g
```

We can compress those linear chains into edges labeled with substrings:

```
c → a → "t"  
c → a → "r"  
d → "og"
```

This saves memory and reduces traversal depth.

#### How It Works (Plain Language)

The key idea is path compression: whenever a node has a single child, merge them into one edge containing the combined substring.

| Step | Operation                           | Result                       |
| ---- | ----------------------------------- | ---------------------------- |
| 1    | Build a normal trie                 | One character per edge       |
| 2    | Traverse each path                  | If node has one child, merge |
| 3    | Replace chain with a substring edge | Fewer nodes, shorter height  |

Compressed tries store edge labels as substrings rather than single characters.

#### Example

Insert `["bear", "bell", "bid", "bull", "buy"]`

1. Start with naive trie.
2. Identify single-child paths.
3. Merge paths:

```
b
 ├── e → "ar"
 │    └── "ll"
 └── u → "ll"
      └── "y"
```

Each edge now carries a substring rather than a single letter.

#### Example Step by Step

| Step | Insert                      | Action                        | Result       |
| ---- | --------------------------- | ----------------------------- | ------------ |
| 1    | "bear"                      | Create path b-e-a-r           | 4 nodes      |
| 2    | "bell"                      | Shares prefix "be"            | Merge prefix |
| 3    | "bid"                       | New branch at "b"             | Add new edge |
| 4    | Compress single-child paths | Replace edges with substrings |              |

#### Tiny Code (Simplified Pseudocode)

```python
class Node:
    def __init__(self):
        self.children = {}
        self.is_end = False

def insert_trie_compressed(root, word):
    node = root
    i = 0
    while i < len(word):
        for edge, child in node.children.items():
            prefix_len = common_prefix(edge, word[i:])
            if prefix_len > 0:
                if prefix_len < len(edge):
                    # Split edge
                    remainder = edge[prefix_len:]
                    new_node = Node()
                    new_node.children[remainder] = child
                    node.children[word[i:i+prefix_len]] = new_node
                    del node.children[edge]
                node = node.children[word[i:i+prefix_len]]
                i += prefix_len
                break
        else:
            node.children[word[i:]] = Node()
            node.children[word[i:]].is_end = True
            break
    node.is_end = True
```

This simplified version merges edges whenever possible.

#### Why It Matters

- Saves memory by merging chains
- Faster search (fewer hops per lookup)
- Ideal for prefix-based queries
- Used in routing tables, autocomplete systems, and dictionaries

#### A Gentle Proof (Why It Works)

Let $n$ be the total length of all keys and $k$ the number of keys.

- A naive trie can have up to $O(n)$ nodes.
- A compressed trie has at most $k - 1$ internal nodes and $k$ leaves, since each branching point corresponds to a unique prefix shared by at least two keys.

Thus, compressed tries reduce both height and node count:

$$
O(n) \text{ nodes (naive)} \quad \to \quad O(k) \text{ nodes (compressed)}
$$

Search and insert remain $O(L)$, where $L$ is key length, but with fewer steps.

#### Try It Yourself

1. Insert `["car", "cat", "cart", "dog"]`
2. Draw both naive and compressed tries
3. Count number of nodes before and after compression
4. Verify edge labels as substrings

#### Test Cases

| Keys                     | Naive Trie Nodes | Compressed Trie Nodes |
| ------------------------ | ---------------- | --------------------- |
| `["a", "b"]`             | 3                | 3                     |
| `["apple", "app"]`       | 6                | 4                     |
| `["abc", "abd", "aef"]`  | 8                | 6                     |
| `["car", "cart", "cat"]` | 9                | 6                     |

#### Complexity

| Operation | Time   | Space  | Notes                       |
| --------- | ------ | ------ | --------------------------- |
| Search    | $O(L)$ | $O(k)$ | $L$ = key length            |
| Insert    | $O(L)$ | $O(k)$ | Split edges when needed     |
| Delete    | $O(L)$ | $O(k)$ | Merge edges if path shrinks |

A compressed trie elegantly blends trie structure and path compression, turning long chains into compact edges, a key step toward efficient prefix trees, routing tables, and tries for text indexing.

## Section 29. Persistent and Functional Data Structures 

### 281 Persistent Stack

A persistent stack is a versioned data structure that remembers all its past states. Instead of overwriting data, every `push` or `pop` operation creates a new version of the stack while keeping access to the old ones.

This concept is part of functional data structures, where immutability and version history are first-class citizens.

#### What Problem Are We Solving?

In traditional stacks, each operation mutates the structure, old versions are lost.

Persistent stacks solve this by allowing:

- Access to previous states at any time
- Undo or time-travel features
- Purely functional programs where data is never mutated

Used in compilers, backtracking systems, and functional programming languages.

#### How It Works (Plain Language)

A stack is a linked list:

- `push(x)` adds a new head node
- `pop()` returns the next node

For persistence, we never modify nodes, instead, each operation creates a new head pointing to existing tails.

| Version | Operation | Top Element | Structure   |
| ------- | --------- | ----------- | ----------- |
| v0      | empty     |,           | ∅           |
| v1      | push(10)  | 10          | 10 → ∅      |
| v2      | push(20)  | 20          | 20 → 10 → ∅ |
| v3      | pop()     | 10          | 10 → ∅      |

Each version reuses previous nodes, no data copying.

#### Example

Start with an empty stack `v0`:

1. `v1 = push(v0, 10)` → stack `[10]`
2. `v2 = push(v1, 20)` → stack `[20, 10]`
3. `v3 = pop(v2)` → returns 20, new stack `[10]`

Now we have three accessible versions:

```
v0: ∅  
v1: 10  
v2: 20 → 10  
v3: 10  
```

#### Tiny Code (Python)

```python
class Node:
    def __init__(self, value, next_node=None):
        self.value = value
        self.next = next_node

class PersistentStack:
    def __init__(self, top=None):
        self.top = top

    def push(self, value):
        # new node points to current top
        return PersistentStack(Node(value, self.top))

    def pop(self):
        if not self.top:
            return self, None
        return PersistentStack(self.top.next), self.top.value

    def peek(self):
        return None if not self.top else self.top.value

# Example
v0 = PersistentStack()
v1 = v0.push(10)
v2 = v1.push(20)
v3, popped = v2.pop()

print(v2.peek())  # 20
print(v3.peek())  # 10
```

This approach reuses nodes, creating new versions without mutation.

#### Tiny Code (C, Conceptual)

```c
#include <stdio.h>
#include <stdlib.h>

typedef struct Node {
    int value;
    struct Node* next;
} Node;

typedef struct {
    Node* top;
} Stack;

Stack* push(Stack* s, int value) {
    Node* node = malloc(sizeof(Node));
    node->value = value;
    node->next = s->top;
    Stack* new_stack = malloc(sizeof(Stack));
    new_stack->top = node;
    return new_stack;
}

Stack* pop(Stack* s, int* popped_value) {
    if (!s->top) return s;
    *popped_value = s->top->value;
    Stack* new_stack = malloc(sizeof(Stack));
    new_stack->top = s->top->next;
    return new_stack;
}
```

Every `push` or `pop` creates a new `Stack*` that points to the previous structure.

#### Why It Matters

- Immutability ensures data safety and concurrency-friendly design
- Versioning allows backtracking, undo, or branching computations
- Foundation for functional programming and persistent data stores

#### A Gentle Proof (Why It Works)

Each version of the stack shares unchanged nodes with previous versions.
Because `push` and `pop` only modify the head reference, older versions remain intact.

If $n$ is the total number of operations,

- Each new version adds $O(1)$ space
- Shared tails ensure total space = $O(n)$

Mathematically:

$$
S_n = S_{n-1} + O(1)
$$

and old versions never get overwritten, ensuring persistence.

#### Try It Yourself

1. Build a persistent stack with values [1, 2, 3]
2. Pop once, and confirm earlier versions still have their values
3. Compare with a mutable stack implementation
4. Visualize the shared linked nodes between versions

#### Test Cases

| Operation           | Result           | Notes                    |
| ------------------- | ---------------- | ------------------------ |
| `v1 = push(v0, 10)` | `[10]`           | new version              |
| `v2 = push(v1, 20)` | `[20, 10]`       | shares 10-node           |
| `v3, val = pop(v2)` | val = 20, `[10]` | old v2 intact            |
| `v1.peek()`         | `10`             | unaffected by later pops |

#### Complexity

| Operation          | Time   | Space  | Notes           |
| ------------------ | ------ | ------ | --------------- |
| Push               | $O(1)$ | $O(1)$ | new head        |
| Pop                | $O(1)$ | $O(1)$ | new top pointer |
| Access old version | $O(1)$ |,      | store reference |

A persistent stack elegantly combines immutability, sharing, and time-travel, a small but powerful step into the world of functional data structures.

### 282 Persistent Array

A persistent array is an immutable, versioned structure that allows access to all past states. Instead of overwriting elements, every update creates a new version that shares most of its structure with previous ones.

This makes it possible to "time travel", view or restore any earlier version in constant or logarithmic time, without copying the entire array.

#### What Problem Are We Solving?

A normal array is mutable, every `arr[i] = x` destroys the old value. If we want history, undo, or branching computation, this is unacceptable.

A persistent array keeps all versions:

| Version | Operation    | State                    |
| ------- | ------------ | ------------------------ |
| v0      | `[]`         | empty                    |
| v1      | `set(0, 10)` | `[10]`                   |
| v2      | `set(0, 20)` | `[20]` (v1 still `[10]`) |

Each version reuses unmodified parts of the array, avoiding full duplication.

#### How It Works (Plain Language)

A persistent array can be implemented using copy-on-write or tree-based structures.

1. Copy-on-Write (Small Arrays)

   * Create a new array copy only when an element changes.
   * Simple but $O(n)$ update cost.

2. Path Copying with Trees (Large Arrays)

   * Represent the array as a balanced binary tree (like a segment tree).
   * Each update copies only the path to the changed leaf.
   * Space per update = $O(\log n)$

So, each version points to a root node. When you modify index $i$, a new path is created down the tree, while untouched subtrees are shared.

#### Example

Let's build a persistent array of size 4.

#### Step 1: Initial version

```
v0 = [0, 0, 0, 0]
```

#### Step 2: Update index 2

```
v1 = set(v0, 2, 5)  → [0, 0, 5, 0]
```

#### Step 3: Update index 1

```
v2 = set(v1, 1, 9)  → [0, 9, 5, 0]
```

`v0`, `v1`, and `v2` all coexist independently.

#### Example Step-by-Step (Tree Representation)

Each node covers a range:

```
Root: [0..3]
  ├── Left [0..1]
  │     ├── [0] → 0
  │     └── [1] → 9
  └── Right [2..3]
        ├── [2] → 5
        └── [3] → 0
```

Updating index 1 copies only the path `[0..3] → [0..1] → [1]`, not the entire tree.

#### Tiny Code (Python, Tree-based)

```python
class Node:
    def __init__(self, left=None, right=None, value=0):
        self.left = left
        self.right = right
        self.value = value

def build(l, r):
    if l == r:
        return Node()
    m = (l + r) // 2
    return Node(build(l, m), build(m+1, r))

def update(node, l, r, idx, val):
    if l == r:
        return Node(value=val)
    m = (l + r) // 2
    if idx <= m:
        return Node(update(node.left, l, m, idx, val), node.right)
    else:
        return Node(node.left, update(node.right, m+1, r, idx, val))

def query(node, l, r, idx):
    if l == r:
        return node.value
    m = (l + r) // 2
    return query(node.left, l, m, idx) if idx <= m else query(node.right, m+1, r, idx)

# Example
n = 4
v0 = build(0, n-1)
v1 = update(v0, 0, n-1, 2, 5)
v2 = update(v1, 0, n-1, 1, 9)

print(query(v2, 0, n-1, 2))  # 5
print(query(v1, 0, n-1, 1))  # 0
```

Each update creates a new version root.

#### Why It Matters

- Time-travel debugging: retrieve old states
- Undo/redo systems in editors
- Branching computations in persistent algorithms
- Functional programming without mutation

#### A Gentle Proof (Why It Works)

Let $n$ be array size, $u$ number of updates.

Each update copies $O(\log n)$ nodes.
So total space:

$$
O(n + u \log n)
$$

Each query traverses one path $O(\log n)$.
No version ever invalidates another, all roots remain accessible.

Persistence holds because we never mutate existing nodes, only allocate new ones and reuse subtrees.

#### Try It Yourself

1. Build an array of size 8, all zeros.
2. Create v1 = set index 4 → 7
3. Create v2 = set index 2 → 9
4. Print values from v0, v1, v2
5. Confirm that old versions remain unchanged.

#### Test Cases

| Operation   | Input       | Output      | Notes       |
| ----------- | ----------- | ----------- | ----------- |
| build(4)    | `[0,0,0,0]` | v0          | base        |
| set(v0,2,5) |             | `[0,0,5,0]` | new version |
| set(v1,1,9) |             | `[0,9,5,0]` | v1 reused   |

#### Complexity

| Operation          | Time        | Space       | Notes          |
| ------------------ | ----------- | ----------- | -------------- |
| Build              | $O(n)$      | $O(n)$      | initial        |
| Update             | $O(\log n)$ | $O(\log n)$ | path copy      |
| Query              | $O(\log n)$ |,           | one path       |
| Access Old Version | $O(1)$      |,           | root reference |

A persistent array turns ephemeral memory into a versioned timeline, each change is a branch, every version eternal. Perfect for functional programming, debugging, and algorithmic history.

### 283 Persistent Segment Tree

A persistent segment tree is a versioned data structure that supports range queries and point updates, while keeping every past version accessible.

It's a powerful combination of segment trees and persistence, allowing you to query historical states, perform undo operations, and even compare past and present results efficiently.

#### What Problem Are We Solving?

A standard segment tree allows:

- Point updates: `arr[i] = x`
- Range queries: `sum(l, r)` or `min(l, r)`

But every update overwrites old values. A persistent segment tree solves this by creating a new version on each update, reusing unchanged nodes.

| Version | Operation    | State          |
| ------- | ------------ | -------------- |
| v0      | build        | `[1, 2, 3, 4]` |
| v1      | update(2, 5) | `[1, 2, 5, 4]` |
| v2      | update(1, 7) | `[1, 7, 5, 4]` |

Now you can query any version:

- `query(v0, 1, 3)` → old sum
- `query(v2, 1, 3)` → updated sum

#### How It Works (Plain Language)

A segment tree is a binary tree where each node stores an aggregate (sum, min, max) over a segment.

Persistence is achieved by path copying:

- When updating index `i`, only nodes on the path from root to leaf are replaced.
- All other nodes are shared between versions.

So each new version costs $O(\log n)$ nodes and space.

| Step | Operation    | Affected Nodes        |
| ---- | ------------ | --------------------- |
| 1    | Build        | $O(n)$ nodes          |
| 2    | Update(2, 5) | $O(\log n)$ new nodes |
| 3    | Update(1, 7) | $O(\log n)$ new nodes |

#### Example

Let initial array be `[1, 2, 3, 4]`

1. Build tree (v0)
2. v1 = update(v0, 2 → 5)
3. v2 = update(v1, 1 → 7)

Now:

- `query(v0, 1, 4) = 10`
- `query(v1, 1, 4) = 12`
- `query(v2, 1, 4) = 17`

All versions share most nodes, saving memory.

#### Example Step-by-Step

#### Update v0 → v1 at index 2

| Version | Tree Nodes Copied          | Shared           |
| ------- | -------------------------- | ---------------- |
| v1      | Path [root → left → right] | Others unchanged |

So v1 differs only along one path.

#### Tiny Code (Python)

```python
class Node:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def build(arr, l, r):
    if l == r:
        return Node(arr[l])
    m = (l + r) // 2
    left = build(arr, l, m)
    right = build(arr, m + 1, r)
    return Node(left.val + right.val, left, right)

def update(node, l, r, idx, val):
    if l == r:
        return Node(val)
    m = (l + r) // 2
    if idx <= m:
        left = update(node.left, l, m, idx, val)
        return Node(left.val + node.right.val, left, node.right)
    else:
        right = update(node.right, m + 1, r, idx, val)
        return Node(node.left.val + right.val, node.left, right)

def query(node, l, r, ql, qr):
    if qr < l or ql > r:
        return 0
    if ql <= l and r <= qr:
        return node.val
    m = (l + r) // 2
    return query(node.left, l, m, ql, qr) + query(node.right, m + 1, r, ql, qr)

# Example
arr = [1, 2, 3, 4]
v0 = build(arr, 0, 3)
v1 = update(v0, 0, 3, 2, 5)
v2 = update(v1, 0, 3, 1, 7)

print(query(v0, 0, 3, 0, 3))  # 10
print(query(v1, 0, 3, 0, 3))  # 12
print(query(v2, 0, 3, 0, 3))  # 17
```

#### Why It Matters

- Access any version instantly
- Enables time-travel queries
- Supports immutable analytics
- Used in offline queries, competitive programming, and functional databases

#### A Gentle Proof (Why It Works)

Each update only modifies $O(\log n)$ nodes.
All other subtrees are shared, so total space:

$$
S(u) = O(n + u \log n)
$$

Querying any version costs $O(\log n)$ since only one path is traversed.

Persistence holds since no nodes are mutated, only replaced.

#### Try It Yourself

1. Build `[1, 2, 3, 4]`
2. Update index 2 → 5 (v1)
3. Update index 1 → 7 (v2)
4. Query sum(1, 4) in v0, v1, v2
5. Verify shared subtrees via visualization

#### Test Cases

| Version | Operation   | Query(0,3) | Notes            |
| ------- | ----------- | ---------- | ---------------- |
| v0      | `[1,2,3,4]` | 10         | base             |
| v1      | set(2,5)    | 12         | changed one leaf |
| v2      | set(1,7)    | 17         | another update   |

#### Complexity

| Operation      | Time        | Space       | Notes     |
| -------------- | ----------- | ----------- | --------- |
| Build          | $O(n)$      | $O(n)$      | full tree |
| Update         | $O(\log n)$ | $O(\log n)$ | path copy |
| Query          | $O(\log n)$ |,           | one path  |
| Version Access | $O(1)$      |,           | via root  |

A persistent segment tree is your immutable oracle, each version a snapshot in time, forever queryable, forever intact.

### 284 Persistent Linked List

A persistent linked list is a versioned variant of the classic singly linked list, where every insertion or deletion produces a new version without destroying the old one.

Each version represents a distinct state of the list, and all versions coexist by sharing structure, unchanged nodes are reused, only the modified path is copied.

This technique is core to functional programming, undo systems, and immutable data structures.

#### What Problem Are We Solving?

A mutable linked list loses its history after every change.

With persistence, we preserve all past versions:

| Version | Operation      | List     |
| ------- | -------------- | -------- |
| v0      | empty          | ∅        |
| v1      | push_front(10) | [10]     |
| v2      | push_front(20) | [20, 10] |
| v3      | pop_front()    | [10]     |

Each version is a first-class citizen, you can traverse, query, or compare any version at any time.

#### How It Works (Plain Language)

Each node in a singly linked list has:

- `value`
- `next` pointer

For persistence, we never mutate nodes. Instead, operations return a new head:

- `push_front(x)`: create a new node `n = Node(x, old_head)`
- `pop_front()`: return `old_head.next` as new head

All old nodes remain intact and shared.

| Operation      | New Node        | Shared Structure      |
| -------------- | --------------- | --------------------- |
| push_front(20) | new head        | tail reused           |
| pop_front()    | new head (next) | old head still exists |

#### Example

#### Step-by-step versioning

1. `v0 = []`
2. `v1 = push_front(v0, 10)` → `[10]`
3. `v2 = push_front(v1, 20)` → `[20, 10]`
4. `v3 = pop_front(v2)` → `[10]`

Versions:

```
v0: ∅  
v1: 10  
v2: 20 → 10  
v3: 10  
```

All coexist and share structure.

#### Example (Graph View)

```
v0: ∅
v1: 10 → ∅
v2: 20 → 10 → ∅
v3: 10 → ∅
```

Notice: `v2.tail` is reused from `v1`.

#### Tiny Code (Python)

```python
class Node:
    def __init__(self, value, next_node=None):
        self.value = value
        self.next = next_node

class PersistentList:
    def __init__(self, head=None):
        self.head = head

    def push_front(self, value):
        new_node = Node(value, self.head)
        return PersistentList(new_node)

    def pop_front(self):
        if not self.head:
            return self, None
        return PersistentList(self.head.next), self.head.value

    def to_list(self):
        result, curr = [], self.head
        while curr:
            result.append(curr.value)
            curr = curr.next
        return result

# Example
v0 = PersistentList()
v1 = v0.push_front(10)
v2 = v1.push_front(20)
v3, popped = v2.pop_front()

print(v1.to_list())  # [10]
print(v2.to_list())  # [20, 10]
print(v3.to_list())  # [10]
```

#### Tiny Code (C, Conceptual)

```c
#include <stdio.h>
#include <stdlib.h>

typedef struct Node {
    int value;
    struct Node* next;
} Node;

typedef struct {
    Node* head;
} PList;

PList push_front(PList list, int value) {
    Node* new_node = malloc(sizeof(Node));
    new_node->value = value;
    new_node->next = list.head;
    PList new_list = { new_node };
    return new_list;
}

PList pop_front(PList list, int* popped) {
    if (!list.head) return list;
    *popped = list.head->value;
    PList new_list = { list.head->next };
    return new_list;
}
```

No mutations, only new nodes allocated.

#### Why It Matters

- Immutable, perfect for functional programs
- Undo / Time-travel, revisit old versions
- Safe concurrency, no data races
- Memory-efficient, tail sharing reuses old structure

#### A Gentle Proof (Why It Works)

Let $n$ be number of operations.
Each `push_front` or `pop_front` creates at most one new node.

Thus:

- Total space after $n$ ops: $O(n)$
- Time per operation: $O(1)$

Persistence is guaranteed since no node is modified in place.

All versions share the unchanged suffix:

$$
v_k = \text{Node}(x_k, v_{k-1})
$$

Hence, structure sharing is linear and safe.

#### Try It Yourself

1. Start with empty list
2. Push 3 → Push 2 → Push 1
3. Pop once
4. Print all versions
5. Observe how tails are shared

#### Test Cases

| Operation  | Input | Output   | Version |
| ---------- | ----- | -------- | ------- |
| push_front | 10    | [10]     | v1      |
| push_front | 20    | [20, 10] | v2      |
| pop_front  |,     | [10]     | v3      |
| to_list    | v1    | [10]     |,       |

#### Complexity

| Operation  | Time   | Space  | Notes               |
| ---------- | ------ | ------ | ------------------- |
| push_front | $O(1)$ | $O(1)$ | new head only       |
| pop_front  | $O(1)$ | $O(1)$ | reuse next          |
| access     | $O(n)$ |,      | same as linked list |

A persistent linked list is the simplest gateway to persistence, each operation is $O(1)$, each version immortal.
It's the backbone of functional stacks, queues, and immutable collections.

### 286 Finger Tree

A finger tree is a versatile, persistent data structure that provides amortized O(1) access to both ends (front and back) and O(log n) access or updates in the middle.

It's a functional, immutable sequence structure, a balanced tree augmented with *fingers* (fast access points) to its ends. Finger trees form the foundation for many persistent data types, such as queues, deques, priority sequences, and even rope-like text editors.

#### What Problem Are We Solving?

Immutable lists are fast at the front but slow at the back. Immutable arrays are the opposite. Deques with persistence are hard to maintain efficiently.

We want:

- $O(1)$ front access
- $O(1)$ back access
- $O(\log n)$ middle access
- Persistence and immutability

Finger trees achieve this by combining shallow digit buffers at the edges with balanced nodes in the middle.

#### How It Works (Plain Language)

A finger tree is built recursively:

```
FingerTree = Empty
           | Single(a)
           | Deep(prefix, deeper_tree, suffix)
```

- prefix and suffix: small arrays (digits) with 1–4 elements
- deeper_tree: recursively holds nodes of higher rank

The *fingers* (prefix/suffix) give constant-time access to both ends.
Insertions push elements into digits; when full, they roll into the deeper tree.

#### Example

Insert 1, 2, 3, 4, 5:

```
Deep [1,2] (Node [3,4]) [5]
```

You can:

- Push front → prepend into prefix
- Push back → append into suffix
- Access ends in O(1)
- Insert middle → recurse into deeper tree (O(log n))

Each operation returns a new version sharing unchanged subtrees.

#### Example State

| Operation     | Structure                | Notes           |
| ------------- | ------------------------ | --------------- |
| empty         | `Empty`                  | base            |
| push_front(1) | `Single(1)`              | one element     |
| push_front(2) | `Deep [2] Empty [1]`     | two ends        |
| push_back(3)  | `Deep [2] Empty [1,3]`   | add suffix      |
| push_back(4)  | `Deep [2,4] Empty [1,3]` | balanced growth |

Each version reuses most of its structure, ensuring persistence.

#### Tiny Code (Python – Conceptual)

This is a simplified model, not a complete implementation (real finger trees rely on more type-level machinery).

```python
class Empty:
    pass

class Single:
    def __init__(self, value):
        self.value = value

class Deep:
    def __init__(self, prefix, middle, suffix):
        self.prefix = prefix
        self.middle = middle
        self.suffix = suffix

def push_front(tree, x):
    if isinstance(tree, Empty):
        return Single(x)
    if isinstance(tree, Single):
        return Deep([x], Empty(), [tree.value])
    if len(tree.prefix) < 4:
        return Deep([x] + tree.prefix, tree.middle, tree.suffix)
    else:
        # roll prefix into deeper tree
        new_middle = push_front(tree.middle, tree.prefix)
        return Deep([x], new_middle, tree.suffix)

def to_list(tree):
    if isinstance(tree, Empty):
        return []
    if isinstance(tree, Single):
        return [tree.value]
    return tree.prefix + to_list(tree.middle) + tree.suffix
```

This captures the core recursive flavor, constant-time fingers, logarithmic recursion.

#### Why It Matters

- Generic framework for sequences
- Amortized O(1) insertion/removal at both ends
- O(log n) concatenation, split, or search
- Basis for:

  * Functional deques
  * Priority queues
  * Ordered sequences (like RRB-trees)
  * Incremental editors

#### A Gentle Proof (Why It Works)

Digits store up to 4 elements, guaranteeing bounded overhead.
Each recursive step reduces the size by a constant factor, ensuring depth = $O(\log n)$.

For each operation:

- Ends touched in $O(1)$
- Structural changes at depth $O(\log n)$

Thus, total cost:

$$
T(n) = O(\log n), \quad \text{amortized O(1) at ends}
$$

Persistence is ensured because all updates build new nodes without modifying existing ones.

#### Try It Yourself

1. Start with empty tree.
2. Push 1, 2, 3, 4.
3. Pop front, observe structure.
4. Push 5, inspect sharing between versions.
5. Convert each version to list, compare results.

#### Test Cases

| Operation     | Result    | Notes              |
| ------------- | --------- | ------------------ |
| push_front(1) | [1]       | base               |
| push_front(2) | [2, 1]    | prefix growth      |
| push_back(3)  | [2, 1, 3] | suffix add         |
| pop_front()   | [1, 3]    | remove from prefix |

#### Complexity

| Operation       | Time             | Space       | Notes            |
| --------------- | ---------------- | ----------- | ---------------- |
| push_front/back | $O(1)$ amortized | $O(1)$      | uses digits      |
| pop_front/back  | $O(1)$ amortized | $O(1)$      | constant fingers |
| random access   | $O(\log n)$      | $O(\log n)$ | recursion        |
| concat/split    | $O(\log n)$      | $O(\log n)$ | efficient split  |

A finger tree is the Swiss Army knife of persistent sequences, fast at both ends, balanced within, and beautifully immutable. It's the blueprint behind countless functional data structures.

### 287 Zipper Structure

A zipper is a powerful technique that makes immutable data structures behave like mutable ones. It provides a focus—a pointer-like position—inside a persistent structure (list, tree, etc.), allowing localized updates, navigation, and edits without mutation.

Think of it as a cursor in a purely functional world. Every movement or edit yields a new version, while sharing unmodified parts.

#### What Problem Are We Solving?

Immutable data structures can't be "modified in place." You can't just "move a cursor" or "replace an element" without reconstructing the entire structure.

A zipper solves this by maintaining:

1. The focused element, and
2. The context (what's on the left/right or above/below).

You can then move, update, or rebuild efficiently, reusing everything else.

#### How It Works (Plain Language)

A zipper separates a structure into:

- Focus: current element under attention
- Context: reversible description of the path you took

When you move the focus, you update the context.
When you change the focused element, you create a new node and rebuild from context.

For lists:

```
Zipper = (Left, Focus, Right)
```

For trees:

```
Zipper = (ParentContext, FocusNode)
```

You can think of it like a tape in a Turing machine—everything to the left and right is preserved.

#### Example (List Zipper)

We represent a list `[a, b, c, d]` with a cursor on `c`:

```
Left: [b, a]   Focus: c   Right: [d]
```

From here:

- `move_left` → focus = `b`
- `move_right` → focus = `d`
- `update(x)` → replace `c` with `x`

All in O(1), returning a new zipper version.

#### Example Operations

| Operation              | Result            | Description       |
| ---------------------- | ----------------- | ----------------- |
| `from_list([a,b,c,d])` | ([ ], a, [b,c,d]) | init at head      |
| `move_right`           | ([a], b, [c,d])   | shift focus right |
| `update('X')`          | ([a], X, [c,d])   | replace focus     |
| `to_list`              | [a,X,c,d]         | rebuild full list |

#### Tiny Code (Python – List Zipper)

```python
class Zipper:
    def __init__(self, left=None, focus=None, right=None):
        self.left = left or []
        self.focus = focus
        self.right = right or []

    @staticmethod
    def from_list(lst):
        if not lst: return Zipper([], None, [])
        return Zipper([], lst[0], lst[1:])

    def move_left(self):
        if not self.left: return self
        return Zipper(self.left[:-1], self.left[-1], [self.focus] + self.right)

    def move_right(self):
        if not self.right: return self
        return Zipper(self.left + [self.focus], self.right[0], self.right[1:])

    def update(self, value):
        return Zipper(self.left, value, self.right)

    def to_list(self):
        return self.left + [self.focus] + self.right

# Example
z = Zipper.from_list(['a', 'b', 'c', 'd'])
z1 = z.move_right().move_right()   # focus on 'c'
z2 = z1.update('X')
print(z2.to_list())  # ['a', 'b', 'X', 'd']
```

Each operation returns a new zipper, persistent editing made simple.

#### Example (Tree Zipper – Conceptual)

A tree zipper stores the path to the root as context:

```
Zipper = (ParentPath, FocusNode)
```

Each parent in the path remembers which side you came from, so you can rebuild upward after an edit.

For example, editing a leaf `L` creates a new `L'` and rebuilds only the nodes along the path, leaving other subtrees untouched.

#### Why It Matters

- Enables localized updates in immutable structures
- Used in functional editors, parsers, navigation systems
- Provides O(1) local movement, O(depth) rebuilds
- Core concept in Huet's zipper, a foundational idea in functional programming

#### A Gentle Proof (Why It Works)

Each movement or edit affects only the local context:

- Move left/right in a list → $O(1)$
- Move up/down in a tree → $O(1)$
- Rebuilding full structure → $O(\text{depth})$

No mutation occurs; each version reuses all untouched substructures.

Formally, if $S$ is the original structure and $f$ the focus,
$$
\text{zip}(S) = (\text{context}, f)
$$
and
$$
\text{unzip}(\text{context}, f) = S
$$
ensuring reversibility.

#### Try It Yourself

1. Create a zipper from `[1,2,3,4]`
2. Move focus to 3
3. Update focus to 99
4. Rebuild full list
5. Verify older zipper still has old value

#### Test Cases

| Step | Operation              | Result            | Notes       |
| ---- | ---------------------- | ----------------- | ----------- |
| 1    | `from_list([a,b,c,d])` | ([ ], a, [b,c,d]) | init        |
| 2    | `move_right()`         | ([a], b, [c,d])   | shift focus |
| 3    | `update('X')`          | ([a], X, [c,d])   | edit        |
| 4    | `to_list()`            | [a, X, c, d]      | rebuild     |

#### Complexity

| Operation          | Time   | Space  | Notes             |
| ------------------ | ------ | ------ | ----------------- |
| Move left/right    | $O(1)$ | $O(1)$ | shift focus       |
| Update             | $O(1)$ | $O(1)$ | local replacement |
| Rebuild            | $O(n)$ | $O(1)$ | when unzipping    |
| Access old version | $O(1)$ |,      | persistent        |

A zipper turns immutability into interactivity. With a zipper, you can move, focus, and edit, all without breaking persistence. It's the bridge between static structure and dynamic navigation.

### 289 Trie with Versioning

A Trie with Versioning is a persistent data structure that stores strings (or sequences) across multiple historical versions. Each new update—an insertion, deletion, or modification—creates a new version of the trie without mutating previous ones, using path copying for structural sharing.

This enables time-travel queries: you can look up keys as they existed at any point in history.

#### What Problem Are We Solving?

We want to maintain a versioned dictionary of strings or sequences, supporting:

- Fast prefix search ($O(\text{length})$)
- Efficient updates without mutation
- Access to past versions (e.g., snapshots, undo/redo, history)

A versioned trie achieves all three by copying only the path from the root to modified nodes while sharing all other subtrees.

Common use cases:

- Versioned symbol tables
- Historical dictionaries
- Autocomplete with rollback
- Persistent tries in functional languages

#### How It Works (Plain Language)

Each trie node contains:

- A mapping from character → child node
- A flag marking end of word

For persistence:

- When you insert or delete, copy only nodes on the affected path.
- Old nodes remain untouched and shared by old versions.

Thus, version $v_{k+1}$ differs from $v_k$ only along the modified path.

#### Tiny Code (Conceptual Python)

```python
class TrieNode:
    def __init__(self, children=None, is_end=False):
        self.children = dict(children or {})
        self.is_end = is_end

def insert(root, word):
    def _insert(node, i):
        node = TrieNode(node.children, node.is_end)
        if i == len(word):
            node.is_end = True
            return node
        ch = word[i]
        node.children[ch] = _insert(node.children.get(ch, TrieNode()), i + 1)
        return node
    return _insert(root, 0)

def search(root, word):
    node = root
    for ch in word:
        if ch not in node.children:
            return False
        node = node.children[ch]
    return node.is_end
```

Each call to `insert` returns a new root (new version), sharing all unmodified branches.

#### Example

| Version | Operation     | Trie Content            |
| ------- | ------------- | ----------------------- |
| v1      | insert("cat") | { "cat" }               |
| v2      | insert("car") | { "cat", "car" }        |
| v3      | insert("dog") | { "cat", "car", "dog" } |
| v4      | delete("car") | { "cat", "dog" }        |

Versions share nodes for prefix `"c"` between all earlier versions.

#### Why It Matters

- Immutable and Safe: No in-place mutation, perfect for functional systems
- Efficient Rollback: Access any prior version in $O(1)$ time
- Prefix Sharing: Saves memory through structural reuse
- Practical for History: Ideal for versioned dictionaries, IDEs, search indices

#### A Gentle Proof (Why It Works)

Each insertion copies one path of length $L$ (word length).
Total time complexity:
$$
T_{\text{insert}} = O(L)
$$

Each node on the new path shares all other unchanged subtrees.
If $N$ is total stored characters across all versions,
$$
\text{Space} = O(N)
$$

Each version root is a single pointer, enabling $O(1)$ access:
$$
\text{Version}_i = \text{Root}_i
$$

Old versions remain fully usable, as they never mutate.

#### Try It Yourself

1. Insert "cat", "car", "dog" into versioned trie
2. Delete "car" to form new version
3. Query prefix "ca" in all versions
4. Check that "car" exists only before deletion
5. Print shared node counts across versions

### Test Case

| Step | Operation     | Version | Exists in Version  | Result |
| ---- | ------------- | ------- | ------------------ | ------ |
| 1    | insert("cat") | v1      | cat                | True   |
| 2    | insert("car") | v2      | car, cat           | True   |
| 3    | insert("dog") | v3      | cat, car, dog      | True   |
| 4    | delete("car") | v4      | car (no), cat, dog | False  |

#### Complexity

| Operation          | Time   | Space  | Notes                    |
| ------------------ | ------ | ------ | ------------------------ |
| Insert             | $O(L)$ | $O(L)$ | Path-copying             |
| Delete             | $O(L)$ | $O(L)$ | Path-copying             |
| Lookup             | $O(L)$ | $O(1)$ | Follows shared structure |
| Access old version | $O(1)$ |,      | Version pointer access   |

A Trie with Versioning blends structural sharing with prefix indexing. Each version is a frozen snapshot—compact, queryable, and immutable—perfect for versioned word histories.

### 290 Persistent Union-Find

A Persistent Union-Find extends the classical Disjoint Set Union (DSU) structure to support *time-travel queries*. Instead of mutating the parent and rank arrays in place, each union operation produces a new version, enabling queries like:

- "Are $x$ and $y$ connected at version $v$?"
- "What did the set look like before the last merge?"

This structure is vital for dynamic connectivity problems where the history of unions matters.

#### What Problem Are We Solving?

Classical DSU supports `find` and `union` efficiently in near-constant time, but only for a single evolving state. Once you merge two sets, the old version is gone.

We need a versioned DSU that keeps all previous states intact, supporting:

- Undo/rollback of operations
- Queries over past connectivity
- Offline dynamic connectivity analysis

#### How It Works (Plain Language)

A Persistent Union-Find uses path copying (similar to persistent arrays) to maintain multiple versions:

- Each `union` creates a new version
- Only affected parent and rank entries are updated in a new structure
- All other nodes share structure with the previous version

There are two main designs:

1. Full persistence using path copying in functional style
2. Partial persistence using rollback stack (undo operations)

We focus here on *full persistence*.

#### Tiny Code (Conceptual Python)

```python
class PersistentDSU:
    def __init__(self, n):
        self.versions = []
        parent = list(range(n))
        rank = [0] * n
        self.versions.append((parent, rank))
    
    def find(self, parent, x):
        if parent[x] == x:
            return x
        return self.find(parent, parent[x])
    
    def union(self, ver, x, y):
        parent, rank = [*ver[0]], [*ver[1]]  # copy arrays
        rx, ry = self.find(parent, x), self.find(parent, y)
        if rx != ry:
            if rank[rx] < rank[ry]:
                parent[rx] = ry
            elif rank[rx] > rank[ry]:
                parent[ry] = rx
            else:
                parent[ry] = rx
                rank[rx] += 1
        self.versions.append((parent, rank))
        return len(self.versions) - 1  # new version index
    
    def connected(self, ver, x, y):
        parent, _ = self.versions[ver]
        return self.find(parent, x) == self.find(parent, y)
```

Each union returns a new version index. You can query `connected(version, a, b)` at any time.

#### Example

| Step | Operation            | Version | Connections            |
| ---- | -------------------- | ------- | ---------------------- |
| 1    | make-set(5)          | v0      | 0 1 2 3 4              |
| 2    | union(0,1)           | v1      | {0,1}, 2, 3, 4         |
| 3    | union(2,3)           | v2      | {0,1}, {2,3}, 4        |
| 4    | union(1,2)           | v3      | {0,1,2,3}, 4           |
| 5    | query connected(0,1) | v1      | True                   |
| 6    | query connected(1,3) | v1      | False (not yet merged) |

You can check connections at any version.

#### Why It Matters

- Time-travel queries across historical versions
- Non-destructive updates allow safe rollback
- Crucial for offline dynamic connectivity, e.g., edge insertions over time
- Simplifies debugging, simulation, and version tracking

#### A Gentle Proof (Why It Works)

Let $n$ be the number of elements and $q$ the number of versions.

Each version differs in only a small number of parent/rank entries.
If each `union` copies $O(\alpha(n))$ elements, total space after $q$ operations is:

$$
O(n + q \alpha(n))
$$

Each query operates on a fixed version in:

$$
O(\alpha(n))
$$

Path compression is often replaced with *partial compression* or omitted to ensure persistence (full path compression breaks immutability).

#### Try It Yourself

1. Initialize DSU with 5 elements
2. Perform unions step by step, saving each version
3. Query connectivity across multiple versions
4. Undo merges by reverting to older versions
5. Visualize parent tree evolution

#### Test Cases

| Version | Query          | Result |
| ------- | -------------- | ------ |
| v0      | connected(0,1) | False  |
| v1      | connected(0,1) | True   |
| v2      | connected(2,3) | True   |
| v3      | connected(1,3) | True   |
| v1      | connected(1,3) | False  |

#### Complexity

| Operation          | Time           | Space       | Notes                  |
| ------------------ | -------------- | ----------- | ---------------------- |
| Find               | $O(\alpha(n))$ | $O(1)$      | Per version            |
| Union              | $O(\alpha(n))$ | $O(n)$ copy | Copy-on-write path     |
| Connected          | $O(\alpha(n))$ | $O(1)$      | Versioned query        |
| Access old version | $O(1)$         |,           | Version pointer lookup |

A Persistent Union-Find is a historical map of connectivity. Each version captures a snapshot of relationships—immutable, queryable, and efficient—ideal for evolving graphs and rollback-capable algorithms.

## Section 30. Advanced Trees and Range Queries 

### 291 Sparse Table Build

A Sparse Table is a static data structure for answering idempotent range queries in O(1) time after O(n log n) preprocessing. It is perfect for Range Minimum/Maximum Query (RMQ), GCD, and any operation where combining overlapping answers is valid, such as `min`, `max`, `gcd`, `lcm` (with care), and bitwise `and/or`. It is not suitable for sum or other non-idempotent operations if you require O(1) queries.

#### What Problem Are We Solving?

Given an array `A[0..n-1]`, we want to answer queries like

- RMQ: minimum on interval `[L, R]`
- RMaxQ: maximum on interval `[L, R]`
  in O(1) time per query, with no updates.

#### How It Works (Plain Language)

Precompute answers for all ranges whose lengths are powers of two.
Let `st[k][i]` store the answer on the interval of length `2^k` starting at `i`, that is `[i, i + 2^k - 1]`.

Build recurrence:

- Base layer `k = 0`: intervals of length 1
  `st[0][i] = A[i]`
- Higher layers: combine two halves of length `2^{k-1}`
  `st[k][i] = op(st[k-1][i], st[k-1][i + 2^{k-1}])`

To answer a query on `[L, R]`, let `len = R - L + 1`, `k = floor(log2(len))`.
For idempotent operations like `min` or `max`, we can cover the range with two overlapping blocks:

- Block 1: `[L, L + 2^k - 1]`
- Block 2: `[R - 2^k + 1, R]`
  Then
  $$
  \text{ans} = \operatorname{op}\big(\text{st}[k][L],\ \text{st}[k][R - 2^k + 1]\big)
  $$

#### Example Step by Step

Array `A = [7, 2, 3, 0, 5, 10, 3, 12, 18]`, `op = min`.

1. Build `st[0]` (length 1):
   `st[0] = [7, 2, 3, 0, 5, 10, 3, 12, 18]`

2. Build `st[1]` (length 2):
   `st[1][i] = min(st[0][i], st[0][i+1])`
   `st[1] = [2, 2, 0, 0, 5, 3, 3, 12]`

3. Build `st[2]` (length 4):
   `st[2][i] = min(st[1][i], st[1][i+2])`
   `st[2] = [0, 0, 0, 0, 3, 3]`

4. Build `st[3]` (length 8):
   `st[3][i] = min(st[2][i], st[2][i+4])`
   `st[3] = [0, 0]`

Query example: RMQ on `[3, 8]`
`len = 6`, `k = floor(log2(6)) = 2`, `2^k = 4`

- Block 1: `[3, 6]` uses `st[2][3]`
- Block 2: `[5, 8]` uses `st[2][5]`
  Answer
  $$
  \min\big(\text{st}[2][3], \text{st}[2][5]\big) = \min(0, 3) = 0
  $$

#### Tiny Code (Python, RMQ with min)

```python
import math

def build_sparse_table(arr, op=min):
    n = len(arr)
    K = math.floor(math.log2(n)) + 1
    st = [[0] * n for _ in range(K)]
    for i in range(n):
        st[0][i] = arr[i]
    j = 1
    while (1 << j) <= n:
        step = 1 << (j - 1)
        for i in range(n - (1 << j) + 1):
            st[j][i] = op(st[j - 1][i], st[j - 1][i + step])
        j += 1
    # Precompute logs for O(1) queries
    lg = [0] * (n + 1)
    for i in range(2, n + 1):
        lg[i] = lg[i // 2] + 1
    return st, lg

def query(st, lg, L, R, op=min):
    length = R - L + 1
    k = lg[length]
    return op(st[k][L], st[k][R - (1 << k) + 1])

# Example
A = [7, 2, 3, 0, 5, 10, 3, 12, 18]
st, lg = build_sparse_table(A, op=min)
print(query(st, lg, 3, 8, op=min))  # 0
print(query(st, lg, 0, 2, op=min))  # 2
```

For `max`, just pass `op=max`. For `gcd`, pass `math.gcd`.

#### Why It Matters

- O(1) query time for static arrays
- O(n log n) preprocessing with simple transitions
- Excellent for RMQ style tasks, LCA via RMQ on Euler tours, and many competitive programming problems
- Cache friendly and implementation simple compared to segment trees when no updates are needed

#### A Gentle Proof (Why It Works)

The table stores answers for all intervals of length `2^k`. Any interval `[L, R]` can be covered by two overlapping power of two blocks of equal length `2^k`, where `k = floor(log2(R - L + 1))`.
For idempotent operations `op`, overlap does not affect correctness, so
$$
\text{op}\big([L, R]\big) = \text{op}\big([L, L + 2^k - 1],\ [R - 2^k + 1, R]\big)
$$
Both blocks are precomputed, so the query is constant time.

#### Try It Yourself

1. Build a table for `A = [5, 4, 3, 6, 1, 2]` with `op = min`.
2. Answer RMQ on `[1, 4]` and `[0, 5]`.
3. Swap `op` to `max` and recheck.
4. Use `op = gcd` and verify results on several ranges.

#### Test Cases

| Array                  | op  | Query  | Expected |
| ---------------------- | --- | ------ | -------- |
| [7, 2, 3, 0, 5, 10, 3] | min | [0, 2] | 2        |
| [7, 2, 3, 0, 5, 10, 3] | min | [3, 6] | 0        |
| [1, 5, 2, 4, 6, 1, 3]  | max | [2, 5] | 6        |
| [12, 18, 6, 9, 3]      | gcd | [1, 4] | 3        |

#### Complexity

| Phase      | Time          | Space         |
| ---------- | ------------- | ------------- |
| Preprocess | $O(n \log n)$ | $O(n \log n)$ |
| Query      | $O(1)$        |,             |

Note
Sparse Table supports static data. If you need updates, consider a segment tree or Fenwick tree. Sparse Table excels when the array is fixed and you need very fast queries.

### 292 Cartesian Tree

A Cartesian Tree is a binary tree built from an array such that:

1. In-order traversal of the tree reproduces the array, and
2. The tree satisfies the heap property with respect to the array values (min-heap or max-heap).

This structure elegantly bridges arrays and binary trees, and plays a key role in algorithms for Range Minimum Query (RMQ), Lowest Common Ancestor (LCA), and sequence decomposition.

#### What Problem Are We Solving?

We want to represent an array $A[0..n-1]$ as a tree that encodes range relationships. For RMQ, if the tree is a min-heap Cartesian Tree, then the LCA of nodes $i$ and $j$ corresponds to the index of the minimum element in the range $[i, j]$.

Thus, building a Cartesian Tree gives us an elegant path from RMQ to LCA in $O(1)$ after $O(n)$ preprocessing.

#### How It Works (Plain Language)

A Cartesian Tree is built recursively:

- The root is the smallest element (for min-heap) or largest (for max-heap).
- The left subtree is built from elements to the left of the root.
- The right subtree is built from elements to the right of the root.

A more efficient linear-time construction uses a stack:

1. Traverse the array from left to right.
2. Maintain a stack of nodes in increasing order.
3. For each new element, pop while the top is greater, then attach the new node as the right child of the last popped node or the left child of the current top.

#### Example

Let $A = [3, 2, 6, 1, 9]$

1. Start with empty stack
2. Insert 3 → stack = [3]
3. Insert 2 → pop 3 (since 3 > 2)

   * 2 becomes parent of 3
   * stack = [2]
4. Insert 6 → 6 > 2 → right child

   * stack = [2, 6]
5. Insert 1 → pop 6, pop 2 → 1 is new root

   * 2 becomes right child of 1
6. Insert 9 → right child of 6

Tree structure (min-heap):

```
       1
      / \
     2   9
    / \
   3   6
```

In-order traversal: `[3, 2, 6, 1, 9]`
Heap property: every parent is smaller than its children ✅

#### Tiny Code (Python, Min-Heap Cartesian Tree)

```python
class Node:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None

def build_cartesian_tree(arr):
    stack = []
    root = None
    for val in arr:
        node = Node(val)
        last = None
        while stack and stack[-1].val > val:
            last = stack.pop()
        node.left = last
        if stack:
            stack[-1].right = node
        else:
            root = node
        stack.append(node)
    return root

def inorder(node):
    return inorder(node.left) + [node.val] + inorder(node.right) if node else []
```

Example usage:

```python
A = [3, 2, 6, 1, 9]
root = build_cartesian_tree(A)
print(inorder(root))  # [3, 2, 6, 1, 9]
```

#### Why It Matters

- RMQ in O(1): RMQ becomes LCA in Cartesian Tree (after Euler Tour + Sparse Table).
- Monotonic Stack Connection: Linear construction mirrors the logic of stack-based range problems (Next Greater Element, Histogram).
- Divide-and-Conquer Decomposition: Represents array's recursive structure.
- Efficient Building: Linear time with stack.

#### A Gentle Proof (Why It Works)

Each element is pushed and popped at most once, so total operations = $O(n)$.
Heap property ensures RMQ correctness:

- In min-heap tree, root of any subtree is the minimum of that segment.
- Thus, LCA(i, j) gives the index of $\min(A[i..j])$.

So by reducing RMQ to LCA, we achieve:

$$
\text{RMQ}(i, j) = \text{index}( \text{LCA}(i, j) )
$$

#### Try It Yourself

1. Build a Cartesian Tree for $A = [4, 5, 2, 3, 1]$ (min-heap).
2. Verify in-order traversal equals original array.
3. Mark parents smaller than children.
4. Identify RMQ(1, 3) from the tree using LCA.

#### Test Cases

| Array           | Tree Type | Root | RMQ(1, 3) | Inorder Matches |
| --------------- | --------- | ---- | --------- | --------------- |
| [3, 2, 6, 1, 9] | min-heap  | 1    | 2         | ✅               |
| [5, 4, 3, 2, 1] | min-heap  | 1    | 2         | ✅               |
| [1, 2, 3, 4, 5] | min-heap  | 1    | 2         | ✅               |
| [2, 7, 5, 9]    | min-heap  | 2    | 5         | ✅               |

#### Complexity

| Operation      | Time          | Space         | Notes                      |
| -------------- | ------------- | ------------- | -------------------------- |
| Build          | $O(n)$        | $O(n)$        | Stack-based construction   |
| Query (RMQ)    | $O(1)$        |,             | After Euler + Sparse Table |
| LCA Preprocess | $O(n \log n)$ | $O(n \log n)$ | Sparse Table method        |

A Cartesian Tree weaves together order and hierarchy: in-order for sequence, heap for dominance, a silent bridge between arrays and trees.

### 293 Segment Tree Beats

Segment Tree Beats is an advanced variant of the classical segment tree that can handle non-trivial range queries and updates beyond sum, min, or max. It's designed for problems where the operation is *not linear* or *not invertible*, such as range chmin/chmax, range add with min tracking, or range second-min queries.

It "beats" the limitation of classical lazy propagation by storing extra state (like second minimum, second maximum) to decide when updates can stop early.

#### What Problem Are We Solving?

Standard segment trees can't efficiently handle complex updates like:

- "Set all $A[i]$ in $[L, R]$ to `min(A[i], x)`"
- "Set all $A[i]$ in $[L, R]$ to `max(A[i], x)`"

Because different elements in a segment may behave differently depending on their value relative to `x`.

Segment Tree Beats solves this by maintaining extra constraints in each node so we can "beat" recursion and skip branches early when conditions are met.

#### How It Works (Plain Language)

Each node stores not just an aggregate but *enough info to know when an operation can be fully applied*.

For Range Chmin (A[i] = min(A[i], x)):

- If `max <= x`: nothing changes
- If `second_max < x < max`: only elements with `max` are updated
- If `x < second_max`: recurse to children

Each node stores:

- `max` (maximum value in range)
- `second_max` (second largest value)
- `count_max` (number of times `max` appears)
- `sum` (sum over range)

This lets us decide update logic without touching all elements.

#### Example (Range Chmin)

Let $A = [4, 7, 6, 3]$

We apply `chmin(L=0, R=3, x=5)` → every element > 5 is set to 5.

Step:

1. Node `[0, 3]`: `max = 7`, `second_max = 6`
2. Since `x = 5 < second_max = 6`, recurse
3. Update left child `[0,1]`: has `7` → `7` becomes `5`
4. Update right `[2,3]`: `max = 6` → `6` becomes `5`

New array: `[4, 5, 5, 3]`

#### Tiny Code (Simplified Range Chmin)

```python
class Node:
    def __init__(self):
        self.sum = 0
        self.max = 0
        self.smax = -float('inf')
        self.cnt = 0
        self.l = None
        self.r = None

def merge(a, b):
    node = Node()
    node.sum = a.sum + b.sum
    if a.max == b.max:
        node.max = a.max
        node.smax = max(a.smax, b.smax)
        node.cnt = a.cnt + b.cnt
    elif a.max > b.max:
        node.max = a.max
        node.smax = max(a.smax, b.max)
        node.cnt = a.cnt
    else:
        node.max = b.max
        node.smax = max(a.max, b.smax)
        node.cnt = b.cnt
    return node

def push_chmin(node, x, length):
    if node.max <= x:
        return
    node.sum -= (node.max - x) * node.cnt
    node.max = x

def update_chmin(node, l, r, ql, qr, x):
    if r < ql or qr < l or node.max <= x:
        return
    if ql <= l and r <= qr and node.smax < x:
        push_chmin(node, x, r - l + 1)
        return
    m = (l + r) // 2
    update_chmin(node.l, l, m, ql, qr, x)
    update_chmin(node.r, m+1, r, ql, qr, x)
    new = merge(node.l, node.r)
    node.max, node.smax, node.cnt, node.sum = new.max, new.smax, new.cnt, new.sum
```

This is the essence: skip updates when possible, split when necessary.

#### Why It Matters

- Handles non-linear updates efficiently
- Preserves logarithmic complexity by reducing unnecessary recursion
- Used in many competitive programming RMQ-like challenges with range cap operations
- Generalizes segment tree to "hard" problems (range `min` caps, range `max` caps, conditional sums)

#### A Gentle Proof (Why It Works)

The trick: by storing max, second max, and count, we can stop descending if the operation affects only elements equal to max.
At most O(log n) nodes per update because:

- Each update lowers some max values
- Each element's value decreases logarithmically before stabilizing

Thus total complexity amortizes to:
$$
O((n + q) \log n)
$$

#### Try It Yourself

1. Build a Segment Tree Beats for $A = [4, 7, 6, 3]$.
2. Apply `chmin(0,3,5)` → verify `[4,5,5,3]`.
3. Apply `chmin(0,3,4)` → verify `[4,4,4,3]`.
4. Track `sum` after each operation.

#### Test Cases

| Array      | Operation    | Result    |
| ---------- | ------------ | --------- |
| [4,7,6,3]  | chmin(0,3,5) | [4,5,5,3] |
| [4,5,5,3]  | chmin(1,2,4) | [4,4,4,3] |
| [1,10,5,2] | chmin(0,3,6) | [1,6,5,2] |
| [5,5,5]    | chmin(0,2,4) | [4,4,4]   |

#### Complexity

| Operation   | Time (Amortized) | Space  | Notes                       |
| ----------- | ---------------- | ------ | --------------------------- |
| Build       | $O(n)$           | $O(n)$ | Same as normal segment tree |
| Range Chmin | $O(\log n)$      | $O(n)$ | Amortized over operations   |
| Query (Sum) | $O(\log n)$      |,      | Combine like usual          |

Segment Tree Beats lives at the intersection of elegance and power, retaining $O(\log n)$ intuition while tackling the kinds of operations classical segment trees can't touch.

### 294 Merge Sort Tree

A Merge Sort Tree is a segment tree where each node stores a sorted list of the elements in its range. It allows efficient queries that depend on order statistics, such as counting how many elements fall within a range or finding the $k$-th smallest element in a subarray.

It's called "Merge Sort Tree" because it is built exactly like merge sort: divide, conquer, and merge sorted halves.

#### What Problem Are We Solving?

Classical segment trees handle sum, min, or max, but not *value-based* queries. Merge Sort Trees enable operations like:

- Count how many numbers in $A[L..R]$ are $\le x$
- Count elements in a value range $[a,b]$
- Find the $k$-th smallest element in $A[L..R]$

These problems often arise in range frequency queries, inversions counting, and offline queries.

#### How It Works (Plain Language)

Each node of the tree covers a segment `[l, r]` of the array. Instead of storing a single number, it stores a sorted list of all elements in that segment.

Building:

1. If `l == r`, store `[A[l]]`.
2. Otherwise, recursively build left and right children.
3. Merge the two sorted lists to form this node's list.

Querying:
To count numbers $\le x$ in range `[L, R]`:

- Visit all segment tree nodes that fully or partially overlap `[L, R]`.
- In each node, binary search for position of `x` in the node's sorted list.

#### Example

Let $A = [2, 5, 1, 4, 3]$

#### Step 1: Build Tree

Each leaf stores one value:

| Node Range | Stored List |
| ---------- | ----------- |
| [0,0]      | [2]         |
| [1,1]      | [5]         |
| [2,2]      | [1]         |
| [3,3]      | [4]         |
| [4,4]      | [3]         |

Now merge:

| Node Range | Stored List |
| ---------- | ----------- |
| [0,1]      | [2,5]       |
| [2,3]      | [1,4]       |
| [2,4]      | [1,3,4]     |
| [0,4]      | [1,2,3,4,5] |

#### Step 2: Query Example

Count elements $\le 3$ in range `[1, 4]`.

We query nodes that cover `[1,4]`: `[1,1]`, `[2,3]`, `[4,4]`.

- `[1,1]` → list = `[5]` → count = 0
- `[2,3]` → list = `[1,4]` → count = 1
- `[4,4]` → list = `[3]` → count = 1

Total = 2 elements ≤ 3.

#### Tiny Code (Python, Count ≤ x)

```python
import bisect

class MergeSortTree:
    def __init__(self, arr):
        self.n = len(arr)
        self.tree = [[] for _ in range(4 * self.n)]
        self._build(arr, 1, 0, self.n - 1)

    def _build(self, arr, node, l, r):
        if l == r:
            self.tree[node] = [arr[l]]
            return
        m = (l + r) // 2
        self._build(arr, node * 2, l, m)
        self._build(arr, node * 2 + 1, m + 1, r)
        self.tree[node] = sorted(self.tree[node * 2] + self.tree[node * 2 + 1])

    def query_leq(self, node, l, r, ql, qr, x):
        if r < ql or qr < l:
            return 0
        if ql <= l and r <= qr:
            return bisect.bisect_right(self.tree[node], x)
        m = (l + r) // 2
        return (self.query_leq(node * 2, l, m, ql, qr, x) +
                self.query_leq(node * 2 + 1, m + 1, r, ql, qr, x))

# Example
A = [2, 5, 1, 4, 3]
mst = MergeSortTree(A)
print(mst.query_leq(1, 0, 4, 1, 4, 3))  # count ≤ 3 in [1,4] = 2
```

#### Why It Matters

- Enables order-based queries (≤, ≥, count, rank)
- Useful for offline range counting, kth-smallest, and inversion queries
- Combines divide-and-conquer sorting with range decomposition

#### A Gentle Proof (Why It Works)

Each level of the tree merges sorted lists from children.

- Each element appears in $O(\log n)$ nodes.
- Each merge is linear in subarray size.

Thus, total build time:
$$
O(n \log n)
$$

Each query visits $O(\log n)$ nodes, each with $O(\log n)$ binary search:
$$
O(\log^2 n)
$$

#### Try It Yourself

1. Build tree for $A = [5, 1, 4, 2, 3]$.
2. Query count ≤ 3 in `[0, 4]`.
3. Query count ≤ 2 in `[1, 3]`.
4. Implement query for "count between $[a,b]$" using two `query_leq` calls.

#### Test Cases

| Array       | Query     | Condition | Answer |
| ----------- | --------- | --------- | ------ |
| [2,5,1,4,3] | [1,4], ≤3 | count     | 2      |
| [2,5,1,4,3] | [0,2], ≤2 | count     | 2      |
| [1,2,3,4,5] | [2,4], ≤4 | count     | 3      |
| [5,4,3,2,1] | [0,4], ≤3 | count     | 3      |

#### Complexity

| Operation           | Time          | Space         |
| ------------------- | ------------- | ------------- |
| Build               | $O(n \log n)$ | $O(n \log n)$ |
| Query (≤ x)         | $O(\log^2 n)$ |,             |
| Query (range count) | $O(\log^2 n)$ |,             |

Merge Sort Trees elegantly bridge sorting and segmentation, empowering range queries that depend not on aggregates, but on the distribution of values themselves.

### 295 Wavelet Tree

A Wavelet Tree is a compact, indexable structure over a sequence that supports rank, select, and range queries by value in O(log \sigma) time with O(n log \sigma) space, where (\sigma) is the alphabet size. Think of it as a value-aware segment tree built on bitvectors that let you jump between levels using rank counts.

#### What Problem Are We Solving?

Given an array $A[1..n]$ over values in $[1..\sigma]$, define the following queries:

- $\text{rank}(x, r)$: number of occurrences of value $x$ in $A[1..r]$
- $\text{select}(x, k)$: position of the $k$-th occurrence of $x$ in $A$
- $\text{kth}(l, r, k)$: the $k$-th smallest value in the subarray $A[l..r]$
- $\text{range\_count}(l, r, a, b)$: number of values in $A[l..r]$ that lie in $[a, b]$


#### How It Works

1. Value partitioning by halves  
   Recursively partition the value domain $[v_{\min}, v_{\max}]$ into two halves at midpoint $m$.

   - Left child stores all elements $\le m$  
   - Right child stores all elements $> m$

2. Stable partition plus bitvector  
   At each node, keep the original order and record a bitvector $B$ of length equal to the number of elements that arrive at the node:

   - $B[i] = 0$ if the $i$-th element goes to the left child  
   - $B[i] = 1$ if it goes to the right child  

   Support fast ranks on $B$: $\mathrm{rank}_0(B,i)$ and $\mathrm{rank}_1(B,i)$.

3. Navigating queries

   - Position based descent: translate positions using prefix counts from $B$.  
     If a query interval at this node is $[l,r]$, then the corresponding interval in the left child is
     $[\mathrm{rank}_0(B,l-1)+1,\ \mathrm{rank}_0(B,r)]$ and in the right child is
     $[\mathrm{rank}_1(B,l-1)+1,\ \mathrm{rank}_1(B,r)]$.
   - Value based descent: choose left or right child by comparing the value with $m$.

Height is $O(\log \sigma)$. Each step uses $O(1)$ bitvector rank operations.


#### Example

Array: $A = [3, 1, 4, 1, 5, 9, 2, 6]$, values in $[1..9]$

Root split by midpoint $m = 5$

- Left child receives elements $\le 5$: $\{3, 1, 4, 1, 5, 2\}$
- Right child receives elements $> 5$: $\{9, 6\}$

Root bitvector marks routing left (0) or right (1), preserving order:

- $A$ by node: [3, 1, 4, 1, 5, 9, 2, 6]  
- $B$: [0, 0, 0, 0, 0, 1, 0, 1]

Left child split on $[1..5]$ by $m = 3$

- Left-left (values $\le 3$): positions where $B=0$ at root, then route by $m=3$  
  Sequence arriving: [3, 1, 4, 1, 5, 2]  
  Bitvector at this node $B_L$: [0, 0, 1, 0, 1, 0]  
  Children:
  - $\le 3$: [3, 1, 1, 2]
  - $> 3$: [4, 5]

- Right-right subtree of root contains [9, 6] (no further split shown)

Rank translation example at the root

For an interval $[l,r]$ at the root, the corresponding intervals are:
- Left child: $[\,\mathrm{rank}_0(B,l-1)+1,\ \mathrm{rank}_0(B,r)\,]$
- Right child: $[\,\mathrm{rank}_1(B,l-1)+1,\ \mathrm{rank}_1(B,r)\,]$

Example: take $[l,r]=[2,7]$ at the root with $B=[0,0,0,0,0,1,0,1]$
- $\mathrm{rank}_0(B,1)=1$, $\mathrm{rank}_0(B,7)=6$ → left interval $[2,6]$
- $\mathrm{rank}_1(B,1)=0$, $\mathrm{rank}_1(B,7)=1$ → right interval $[1,1]$

Height is $O(\log \sigma)$, each descent step uses $O(1)$ rank operations on the local bitvector.


### Core Operations

1. rank(x, r)  
   Walk top down. At a node with split value m:

   - If $x \le m$, set $r \leftarrow \mathrm{rank}_0(B, r)$ and go left  
   - Otherwise set $r \leftarrow \mathrm{rank}_1(B, r)$ and go right  

   When you reach the leaf for value $x$, the current $r$ is the answer.

2. select(x, k)  
   Start at the leaf for value $x$ with local index $k$.  
   Move upward to the root, inverting the position mapping at each parent:
   - If you came from the left child, set $k \leftarrow \mathrm{select\_pos\_0}(B, k)$  
   - If you came from the right child, set $k \leftarrow \mathrm{select\_pos\_1}(B, k)$  
   The final $k$ at the root is the global position of the $k$-th $x$.

3. kth(l, r, k)  
   At a node with split value $m$, let
   $$
   c \;=\; \mathrm{rank}_0(B, r)\;-\;\mathrm{rank}_0(B, l-1)
   $$
   which is the number of items routed left within $[l, r]$.

   - If $k \le c$, map the interval to the left child via
     $$
     l' \;=\; \mathrm{rank}_0(B, l-1) + 1,\quad
     r' \;=\; \mathrm{rank}_0(B, r)
     $$
     and recurse on $(l', r', k)$.
   - Otherwise go right with
     $$
     k \leftarrow k - c,\quad
     l' \;=\; \mathrm{rank}_1(B, l-1) + 1,\quad
     r' \;=\; \mathrm{rank}_1(B, r)
     $$
     and recurse on $(l', r', k)$.

4. range_count(l, r, a, b)  
   Recurse only into value intervals that intersect $[a, b]$.  
   At each visited node, map the position interval using $B$:
   - Left child interval
     $$
     [\,\mathrm{rank}_0(B, l-1)+1,\ \mathrm{rank}_0(B, r)\,]
     $$
   - Right child interval
     $$
     [\,\mathrm{rank}_1(B, l-1)+1,\ \mathrm{rank}_1(B, r)\,]
     $$
   Stop when a node’s value range is fully inside or outside $[a, b]$:
   - Fully inside: add its interval length  
   - Disjoint: add 0


#### Tiny Code Sketch in Python

This sketch shows the structure and kth query. A production version needs succinct rank structures for (B) to guarantee (O(1)) ranks.

```python
import bisect

class WaveletTree:
    def __init__(self, arr, lo=None, hi=None):
        self.lo = min(arr) if lo is None else lo
        self.hi = max(arr) if hi is None else hi
        self.b = []          # bitvector as 0-1 list
        self.pref = [0]      # prefix sums of ones for O(1) rank1
        if self.lo == self.hi or not arr:
            self.left = self.right = None
            return
        mid = (self.lo + self.hi) // 2
        left_part, right_part = [], []
        for x in arr:
            go_right = 1 if x > mid else 0
            self.b.append(go_right)
            self.pref.append(self.pref[-1] + go_right)
            if go_right:
                right_part.append(x)
            else:
                left_part.append(x)
        self.left = WaveletTree(left_part, self.lo, mid)
        self.right = WaveletTree(right_part, mid + 1, self.hi)

    def rank1(self, idx):  # ones in b[1..idx]
        return self.pref[idx]

    def rank0(self, idx):  # zeros in b[1..idx]
        return idx - self.rank1(idx)

    def kth(self, l, r, k):
        # 1-indexed positions
        if self.lo == self.hi:
            return self.lo
        mid = (self.lo + self.hi) // 2
        cnt_left = self.rank0(r) - self.rank0(l - 1)
        if k <= cnt_left:
            nl = self.rank0(l - 1) + 1
            nr = self.rank0(r)
            return self.left.kth(nl, nr, k)
        else:
            nl = self.rank1(l - 1) + 1
            nr = self.rank1(r)
            return self.right.kth(nl, nr, k - cnt_left)
```

Usage example:

```python
A = [3,1,4,1,5,9,2,6]
wt = WaveletTree(A)
print(wt.kth(1, 8, 3))  # 3rd smallest in the whole array
```

#### Why It Matters

- Combines value partitioning with positional stability, enabling order statistics on subranges
- Underpins succinct indexes, FM indexes, rank select dictionaries, and fast offline range queries
- Efficient when (\sigma) is moderate or compressible

#### A Gentle Proof of Bounds

The tree has height $O(\log \sigma)$ since each level halves the value domain. Each query descends one level and performs $O(1)$ rank operations on a bitvector. Therefore
$$
T_{\text{query}} = O(\log \sigma)
$$
Space stores one bit per element per level on average, so
$$
S = O(n \log \sigma)
$$
With compressed bitvectors supporting constant time rank and select, these bounds hold in practice.

#### Try It Yourself

1. Build a wavelet tree for $A = [2, 7, 1, 8, 2, 8, 1]$.
2. Compute $\text{kth}(2, 6, 2)$.
3. Compute $\text{range\_count}(2, 7, 2, 8)$.
4. Compare against a naive sort on $A[l..r]$.


#### Test Cases

| Array             | Query                | Answer |
| ----------------- | -------------------- | ------ |
| [3,1,4,1,5,9,2,6] | kth(1,8,3)           | 3      |
| [3,1,4,1,5,9,2,6] | range_count(3,7,2,5) | 3      |
| [1,1,1,1,1]       | rank(1,5)            | 5      |
| [5,4,3,2,1]       | kth(2,5,2)           | 3      |

#### Complexity

| Operation        | Time                | Space               |
| ---------------- | ------------------- | ------------------- |
| Build            | $O(n \log \sigma)$  | $O(n \log \sigma)$  |
| rank, select     | $O(\log \sigma)$    | –                   |
| kth, range_count | $O(\log \sigma)$    | –                   |


Wavelet trees are a sharp tool for order aware range queries. By weaving bitvectors with stable partitions, they deliver succinct, logarithmic time answers on top of the original sequence.

### 296 KD-Tree

A KD-Tree (k-dimensional tree) is a binary space partitioning data structure for organizing points in a k-dimensional space. It enables fast range searches, nearest neighbor queries, and spatial indexing, often used in geometry, graphics, and machine learning (like k-NN).

#### What Problem Are We Solving?

We need to store and query $n$ points in $k$-dimensional space such that:

- Nearest neighbor query: find the point closest to a query $q$.  
- Range query: find all points within a given region (rectangle or hypersphere).  
- k-NN query: find the $k$ closest points to $q$.

A naive search checks all $n$ points, which takes $O(n)$ time.  
A KD-Tree reduces this to $O(\log n)$ expected query time for balanced trees.


#### How It Works

#### 1. Recursive Partitioning by Dimension

At each level, the KD-Tree splits the set of points along one dimension, cycling through all dimensions.

If the current depth is $d$, use the axis $a = d \bmod k$:

- Sort points by their $a$-th coordinate.  
- Choose the median point as the root to maintain balance.  
- Left child: points with smaller $a$-th coordinate.  
- Right child: points with larger $a$-th coordinate.


#### 2. Search (Nearest Neighbor)

To find nearest neighbor of query $q$:

1. Descend tree following split planes (like BST).
2. Track current best (closest point found so far).
3. Backtrack if potential closer point exists across split plane (distance to plane < current best).

This ensures pruning of subtrees that cannot contain closer points.

#### Example

Suppose 2D points:
$$
P = { (2,3), (5,4), (9,6), (4,7), (8,1), (7,2) }
$$

Step 1: Root splits by (x)-axis (axis 0).
Sorted by (x): ((2,3), (4,7), (5,4), (7,2), (8,1), (9,6)).
Median = ((7,2)). Root = (7,2).

Step 2:

- Left subtree (points with (x < 7)) split by (y)-axis.
- Right subtree (points with (x > 7)) split by (y)-axis.

This creates alternating partitions by x and y, forming axis-aligned rectangles.

#### Tiny Code (Python)

```python
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
    median = len(points) // 2
    node = Node(points[median], axis)
    node.left = build_kdtree(points[:median], depth + 1)
    node.right = build_kdtree(points[median + 1:], depth + 1)
    return node
```

### How Nearest Neighbor Works

Given query (q), maintain best distance (d_{\text{best}}).
For each visited node:

- Compute distance (d = |q - p|)
- If (d < d_{\text{best}}), update best
- Check opposite branch only if (|q[a] - p[a]| < d_{\text{best}})

#### Example Table

| Step | Node Visited | Axis | Current Best | Distance to Plane | Search Next |
| ---- | ------------ | ---- | ------------ | ----------------- | ----------- |
| 1    | (7,2)        | x    | (7,2), 0.0   | 0.0               | Left        |
| 2    | (5,4)        | y    | (5,4), 2.8   | 2.0               | Left        |
| 3    | (2,3)        | x    | (5,4), 2.8   | 3.0               | Stop        |

#### Why It Matters

- Efficient spatial querying in multidimensional data.
- Common in k-NN classification, computer graphics, and robotics pathfinding.
- Basis for libraries like `scipy.spatial.KDTree`.

#### A Gentle Proof (Why It Works)

Each level splits points into halves, forming $O(\log n)$ height.
Each query visits a bounded number of nodes (dependent on dimension).
Expected nearest neighbor cost:

$$
T_{\text{query}} = O(\log n)
$$

Building sorts points at each level:

$$
T_{\text{build}} = O(n \log n)
$$

#### Try It Yourself

1. Build KD-Tree for 2D points ([(2,3),(5,4),(9,6),(4,7),(8,1),(7,2)]).
2. Query nearest neighbor of (q = (9,2)).
3. Trace visited nodes, prune subtrees when possible.

#### Test Cases

| Points                                | Query | Nearest | Expected Path       |
| ------------------------------------- | ----- | ------- | ------------------- |
| [(2,3),(5,4),(9,6),(4,7),(8,1),(7,2)] | (9,2) | (8,1)   | Root → Right → Leaf |
| [(1,1),(2,2),(3,3)]                   | (2,3) | (2,2)   | Root → Right        |
| [(0,0),(10,10)]                       | (5,5) | (10,10) | Root → Right        |

#### Complexity

| Operation        | Time                      | Space        |
| ---------------- | ------------------------- | ------------ |
| Build            | $O(n \log n)$             | $O(n)$       |
| Nearest Neighbor | $O(\log n)$ (expected)    | $O(\log n)$  |
| Range Query      | $O(n^{1 - 1/k} + m)$      | –            |


KD-Trees blend geometry with binary search, cutting space by dimensions to answer questions faster than brute force.

### 297 Range Tree

A Range Tree is a multi-level search structure for answering orthogonal range queries in multidimensional space, such as finding all points inside an axis-aligned rectangle. It extends 1D balanced search trees to higher dimensions using recursive trees on projections.

#### What Problem Are We Solving?

Given a set of $n$ points in $k$-dimensional space, we want to efficiently answer queries like:

> "List all points $(x, y)$ such that $x_1 \le x \le x_2$ and $y_1 \le y \le y_2$."

A naive scan is $O(n)$ per query. Range Trees reduce this to $O(\log^k n + m)$, where $m$ is the number of reported points.

#### How It Works

#### 1. 1D Case (Baseline)

A simple balanced BST (e.g. AVL) on $x$ coordinates supports range queries by traversing paths and collecting nodes in range.

#### 2. 2D Case (Extension)

- Build a primary tree on $x$ coordinates.
- At each node, store a secondary tree built on $y$ coordinates of the points in its subtree.

Each level recursively maintains sorted views along other axes.

#### 3. Range Query

1. Search primary tree for split node $s$, where paths to $x_1$ and $x_2$ diverge.
2. For nodes fully in $[x_1, x_2]$, query their associated $y$-tree for $[y_1, y_2]$.
3. Combine results.

#### Example

Given points:

$$
P = {(2,3), (4,7), (5,1), (7,2), (8,5)}
$$

Query: $[3,7] \times [1,5]$

- Primary tree on $x$: median $(5,1)$ as root.
- Secondary trees at each node on $y$.

Search path:

- Split node $(5,1)$ covers $x \in [3,7]$.
- Visit subtrees with $x \in [4,7]$.
- Query $y$ in $[1,5]$ inside secondary trees.

Returned points: $(5,1), (7,2), (4,7)$ → filter $y \le 5$ → $(5,1),(7,2)$.

#### Tiny Code (Python-like Pseudocode)

```python
class RangeTree:
    def __init__(self, points, depth=0):
        if not points: 
            self.node = None
            return
        axis = depth % 2
        points.sort(key=lambda p: p[axis])
        mid = len(points) // 2
        self.node = points[mid]
        self.left = RangeTree(points[:mid], depth + 1)
        self.right = RangeTree(points[mid + 1:], depth + 1)
        self.sorted_y = sorted(points, key=lambda p: p[1])
```

Query recursively:

- Filter nodes by $x$.
- Binary search on $y$-lists.

### Step-by-Step Table (2D Query)

| Step | Operation         | Axis | Condition       | Action              |
| ---- | ----------------- | ---- | --------------- | ------------------- |
| 1    | Split at (5,1)    | x    | $3 \le 5 \le 7$ | Recurse both sides  |
| 2    | Left (2,3),(4,7)  | x    | $x < 5$         | Visit (4,7) subtree |
| 3    | Right (7,2),(8,5) | x    | $x \le 7$       | Visit (7,2) subtree |
| 4    | Filter by $y$     | y    | $1 \le y \le 5$ | Keep (5,1),(7,2)    |

#### Why It Matters

Range Trees provide deterministic performance for multidimensional queries, unlike kd-trees (which may degrade). They're ideal for:

- Orthogonal range counting
- Database indexing
- Computational geometry problems

They are static structures, suited when data doesn't change often.

#### A Gentle Proof (Why It Works)

Each dimension adds a logarithmic factor.
In 2D, build time:

$$
T(n) = O(n \log n)
$$

Query visits $O(\log n)$ nodes in primary tree, each querying $O(\log n)$ secondary trees:

$$
Q(n) = O(\log^2 n + m)
$$

Space is $O(n \log n)$ due to secondary trees.

#### Try It Yourself

1. Build a 2D Range Tree for
   $P = {(1,2),(2,3),(3,4),(4,5),(5,6)}$
2. Query rectangle $[2,4] \times [3,5]$.
3. Trace visited nodes and verify output.

#### Test Cases

| Points                           | Query Rectangle | Result            |
| -------------------------------- | --------------- | ----------------- |
| [(2,3),(4,7),(5,1),(7,2),(8,5)]  | [3,7] × [1,5]   | (5,1),(7,2)       |
| [(1,2),(2,4),(3,6),(4,8),(5,10)] | [2,4] × [4,8]   | (2,4),(3,6),(4,8) |
| [(1,1),(2,2),(3,3)]              | [1,2] × [1,3]   | (1,1),(2,2)       |

#### Complexity

| Operation  | Time               | Space         |
| ---------- | ------------------ | ------------- |
| Build      | $O(n \log n)$      | $O(n \log n)$ |
| Query (2D) | $O(\log^2 n + m)$  |,             |
| Update     |, (rebuild needed) |,             |

Range Trees are precise geometric indexes: each axis divides the space, and nested trees give fast access to all points in any axis-aligned box.

### 298 Fenwick 2D Tree

A 2D Fenwick Tree (also known as 2D Binary Indexed Tree) extends the 1D Fenwick Tree to handle range queries and point updates on 2D grids such as matrices. It efficiently computes prefix sums and supports dynamic updates in $O(\log^2 n)$ time.

#### What Problem Are We Solving?

Given an $n \times m$ matrix $A$, we want to support two operations efficiently:

1. Update: Add a value $v$ to element $A[x][y]$.
2. Query: Compute the sum of all elements in submatrix $[1..x][1..y]$.

A naive approach takes $O(nm)$ per query. The 2D Fenwick Tree reduces both update and query to $O(\log n \cdot \log m)$.

#### How It Works

Each node $(i, j)$ in the tree stores the sum of a submatrix region determined by the binary representation of its indices:

$$
T[i][j] = \sum_{x = i - 2^{r_i} + 1}^{i} \sum_{y = j - 2^{r_j} + 1}^{j} A[x][y]
$$

where $r_i$ and $r_j$ denote the least significant bit (LSB) of $i$ and $j$.

#### Update Rule

When updating $(x, y)$ by $v$:

```python
for i in range(x, n+1, i & -i):
    for j in range(y, m+1, j & -j):
        tree[i][j] += v
```

#### Query Rule

To compute prefix sum $(1,1)$ to $(x,y)$:

```python
res = 0
for i in range(x, 0, -i & -i):
    for j in range(y, 0, -j & -j):
        res += tree[i][j]
return res
```

#### Range Query

Sum of submatrix $[(x_1,y_1),(x_2,y_2)]$:

$$
S = Q(x_2, y_2) - Q(x_1-1, y_2) - Q(x_2, y_1-1) + Q(x_1-1, y_1-1)
$$

#### Example

Given a $4 \times 4$ matrix:

| $x/y$ | 1 | 2 | 3 | 4 |
| ----- | - | - | - | - |
| 1     | 2 | 1 | 0 | 3 |
| 2     | 1 | 2 | 3 | 1 |
| 3     | 0 | 1 | 2 | 0 |
| 4     | 4 | 0 | 1 | 2 |


Build Fenwick 2D Tree, then query sum in submatrix $[2,2]$ to $[3,3]$.

Expected result:

$$
A[2][2] + A[2][3] + A[3][2] + A[3][3] = 2 + 3 + 1 + 2 = 8
$$

### Step-by-Step Update Example

Suppose we add $v = 5$ at $(2, 3)$:

| Step | $(i,j)$ Updated | Added Value | Reason                        |
| ---- | ---------------- | ----------- | ----------------------------- |
| 1    | $(2,3)$          | +5          | Base position                 |
| 2    | $(2,4)$          | +5          | Next by $j += j \& -j$        |
| 3    | $(4,3)$          | +5          | Next by $i += i \& -i$        |
| 4    | $(4,4)$          | +5          | Both indices propagate upward |


#### Tiny Code (Python-like Pseudocode)

```python
class Fenwick2D:
    def __init__(self, n, m):
        self.n, self.m = n, m
        self.tree = [[0]*(m+1) for _ in range(n+1)]

    def update(self, x, y, val):
        i = x
        while i <= self.n:
            j = y
            while j <= self.m:
                self.tree[i][j] += val
                j += j & -j
            i += i & -i

    def query(self, x, y):
        res = 0
        i = x
        while i > 0:
            j = y
            while j > 0:
                res += self.tree[i][j]
                j -= j & -j
            i -= i & -i
        return res

    def range_sum(self, x1, y1, x2, y2):
        return (self.query(x2, y2) - self.query(x1-1, y2)
                - self.query(x2, y1-1) + self.query(x1-1, y1-1))
```

#### Why It Matters

2D Fenwick Trees are lightweight, dynamic structures for prefix sums and submatrix queries. They're widely used in:

- Image processing (integral image updates)
- Grid-based dynamic programming
- Competitive programming for 2D range queries

They trade slightly higher code complexity for excellent update-query efficiency.

#### A Gentle Proof (Why It Works)

Each update/query operation performs $\log n$ steps on $x$ and $\log m$ on $y$:

$$
T(n,m) = O(\log n \cdot \log m)
$$

Each level adds contributions from subregions determined by LSB decomposition, ensuring every cell contributes exactly once to the query sum.

#### Try It Yourself

1. Initialize $4 \times 4$ Fenwick 2D.
2. Add $5$ to $(2,3)$.
3. Query sum $(1,1)$ to $(2,3)$.
4. Verify it matches manual computation.

#### Test Cases

| Matrix (Partial)              | Update    | Query Rectangle | Result |
| ----------------------------- | --------- | --------------- | ------ |
| $[ [2,1,0],[1,2,3],[0,1,2] ]$ | $(2,3)+5$ | $[1,1]$–$[2,3]$ | 14     |
| $[ [1,1,1],[1,1,1],[1,1,1] ]$ | $(3,3)+2$ | $[2,2]$–$[3,3]$ | 5      |
| $[ [4,0],[0,4] ]$             |,         | $[1,1]$–$[2,2]$ | 8      |

#### Complexity

| Operation | Time                     | Space   |
| --------- | ------------------------ | ------- |
| Update    | $O(\log n \cdot \log m)$ | $O(nm)$ |
| Query     | $O(\log n \cdot \log m)$ |,       |

The 2D Fenwick Tree is an elegant bridge between prefix sums and spatial queries, simple, powerful, and efficient for dynamic 2D grids.

### 299 Treap Split/Merge

A Treap Split/Merge algorithm allows you to divide and combine treaps (randomized balanced binary search trees) efficiently, using priority-based rotations and key-based ordering. This is the foundation for range operations like splits, merges, range updates, and segment queries on implicit treaps.

#### What Problem Are We Solving?

We often need to:

1. Split a treap into two parts:

   * All keys $\le k$ go to the left treap
   * All keys $> k$ go to the right treap

2. Merge two treaps $T_1$ and $T_2$ where all keys in $T_1 < T_2$

These operations enable efficient range queries, persistent edits, and order-statistics while maintaining balanced height.

#### How It Works (Plain Language)

Treaps combine two properties:

- BST Property: Left < Root < Right
- Heap Property: Node priority > children priorities

Split and merge rely on recursive descent guided by key and priority.

#### Split Operation

Split treap $T$ by key $k$:

- If $T.key \le k$, split $T.right$ into $(t2a, t2b)$ and set $T.right = t2a$
- Else, split $T.left$ into $(t1a, t1b)$ and set $T.left = t1b$

Return $(T.left, T.right)$

#### Merge Operation

Merge $T_1$ and $T_2$:

- If $T_1.priority > T_2.priority$, set $T_1.right = \text{merge}(T_1.right, T_2)$
- Else, set $T_2.left = \text{merge}(T_1, T_2.left)$

Return new root.

#### Example

Suppose we have treap with keys:
$$ [1, 2, 3, 4, 5, 6, 7] $$

#### Split by $k = 4$:

Left Treap: $[1, 2, 3, 4]$
Right Treap: $[5, 6, 7]$

Now, merge them back restores original order.

### Step-by-Step Split Example

| Step | Node Key  | Compare to $k=4$ | Action             |
| ---- | --------- | ---------------- | ------------------ |
| 1    | 4         | $\le$ 4          | Go right           |
| 2    | 5         | $>$ 4            | Split left subtree |
| 3    | 5.left=∅$ | return (null,5)  | Combine back       |

Result: left = $[1,2,3,4]$, right = $[5,6,7]$

#### Tiny Code (Python-like Pseudocode)

```python
import random

class Node:
    def __init__(self, key):
        self.key = key
        self.priority = random.random()
        self.left = None
        self.right = None

def split(root, key):
    if not root:
        return (None, None)
    if root.key <= key:
        left, right = split(root.right, key)
        root.right = left
        return (root, right)
    else:
        left, right = split(root.left, key)
        root.left = right
        return (left, root)

def merge(t1, t2):
    if not t1 or not t2:
        return t1 or t2
    if t1.priority > t2.priority:
        t1.right = merge(t1.right, t2)
        return t1
    else:
        t2.left = merge(t1, t2.left)
        return t2
```

#### Why It Matters

Treap split/merge unlocks flexible sequence manipulation and range-based operations:

- Range sum / min / max queries
- Insert or delete in $O(\log n)$
- Persistent or implicit treaps for lists
- Lazy propagation for intervals

It's a key building block in functional and competitive programming data structures.

#### A Gentle Proof (Why It Works)

Each split or merge operation traverses down the height of the treap. Since treaps are expected balanced, height is $O(\log n)$.

- Split correctness:
  Each recursive call preserves BST ordering.
- Merge correctness:
  Maintains heap property since highest priority becomes root.

Thus, both return valid treaps.

#### Try It Yourself

1. Build treap with keys $[1..7]$.
2. Split by $k=4$.
3. Print inorder traversals of both sub-treaps.
4. Merge back. Confirm structure matches original.

#### Test Cases

| Input Keys      | Split Key | Left Treap Keys | Right Treap Keys |
| --------------- | --------- | --------------- | ---------------- |
| [1,2,3,4,5,6,7] | 4         | [1,2,3,4]       | [5,6,7]          |
| [10,20,30,40]   | 25        | [10,20]         | [30,40]          |
| [5,10,15,20]    | 5         | [5]             | [10,15,20]       |

#### Complexity

| Operation | Time        | Space  |
| --------- | ----------- | ------ |
| Split     | $O(\log n)$ | $O(1)$ |
| Merge     | $O(\log n)$ | $O(1)$ |

Treap Split/Merge is the elegant heart of many dynamic set and sequence structures, one key, one random priority, two simple operations, infinite flexibility.

### 300 Mo's Algorithm on Tree

Mo's Algorithm on Tree is an extension of the classical Mo's algorithm used on arrays. It allows efficient processing of offline queries on trees, especially those involving subtrees or paths, by converting them into a linear order (Euler tour) and then applying a square root decomposition strategy.

#### What Problem Are We Solving?

When you need to answer multiple queries like:

- "How many distinct values are in the subtree of node $u$?"
- "What is the sum over the path from $u$ to $v$?"

A naive approach may require $O(n)$ traversal per query, leading to $O(nq)$ total complexity.
Mo's algorithm on trees reduces this to approximately $O((n + q)\sqrt{n})$, by reusing results from nearby queries.

#### How It Works (Plain Language)

1. Euler Tour Flattening
   Transform the tree into a linear array using an Euler tour. Each node's first appearance marks its position in the linearized sequence.

2. Query Transformation

   * For subtree queries, a subtree becomes a continuous range in the Euler array.
   * For path queries, break into two subranges and handle Least Common Ancestor (LCA) separately.

3. Mo's Ordering
   Sort queries by:

   * Block of left endpoint (using $\text{block} = \lfloor L / \sqrt{N} \rfloor$)
   * Right endpoint (ascending or alternating per block)

4. Add/Remove Function
   Maintain a frequency map or running result as the window moves.

#### Example

Given a tree:

```
1
├── 2
│   ├── 4
│   └── 5
└── 3
```

Euler Tour: `[1, 2, 4, 4, 5, 5, 2, 3, 3, 1]`

Subtree(2): Range covering `[2, 4, 4, 5, 5, 2]`

Each subtree query becomes a range query over the Euler array.
Mo's algorithm processes ranges efficiently in sorted order.

### Step-by-Step Example

| Step | Query (L,R) | Current Range | Add/Remove | Result  |
| ---- | ----------- | ------------- | ---------- | ------- |
| 1    | (2,6)       | [2,6]         | +4,+5      | Count=2 |
| 2    | (2,8)       | [2,8]         | +3         | Count=3 |
| 3    | (1,6)       | [1,6]         | -3,+1      | Count=3 |

Each query is answered by incremental adjustment, not recomputation.

#### Tiny Code (Python-like Pseudocode)

```python
import math

# Preprocessing
def euler_tour(u, p, g, order):
    order.append(u)
    for v in g[u]:
        if v != p:
            euler_tour(v, u, g, order)
            order.append(u)

# Mo's structure
class Query:
    def __init__(self, l, r, idx):
        self.l, self.r, self.idx = l, r, idx

def mo_on_tree(n, queries, order, value):
    block = int(math.sqrt(len(order)))
    queries.sort(key=lambda q: (q.l // block, q.r))

    freq = [0]*(n+1)
    answer = [0]*len(queries)
    cur = 0
    L, R = 0, -1

    def add(pos):
        nonlocal cur
        node = order[pos]
        freq[node] += 1
        if freq[node] == 1:
            cur += value[node]

    def remove(pos):
        nonlocal cur
        node = order[pos]
        freq[node] -= 1
        if freq[node] == 0:
            cur -= value[node]

    for q in queries:
        while L > q.l:
            L -= 1
            add(L)
        while R < q.r:
            R += 1
            add(R)
        while L < q.l:
            remove(L)
            L += 1
        while R > q.r:
            remove(R)
            R -= 1
        answer[q.idx] = cur
    return answer
```

#### Why It Matters

- Converts tree queries into range queries efficiently
- Reuses computation by sliding window technique
- Useful for frequency, sum, or distinct count queries
- Supports subtree queries, path queries (with LCA handling), and color-based queries

#### A Gentle Proof (Why It Works)

1. Euler Tour guarantees every subtree is a contiguous range.
2. Mo's algorithm ensures total number of add/remove operations is $O((n + q)\sqrt{n})$.
3. Combining them, each query is handled incrementally within logarithmic amortized cost.

Thus, offline complexity is sublinear per query.

#### Try It Yourself

1. Build Euler tour for tree of 7 nodes.
2. Write subtree queries for nodes $2,3,4$.
3. Sort queries by block order.
4. Implement add/remove logic to count distinct colors or sums.
5. Compare performance with naive DFS per query.

#### Test Cases

| Nodes               | Query      | Expected Result             |
| ------------------- | ---------- | --------------------------- |
| [1-2-3-4-5]         | Subtree(2) | Sum or count over nodes 2–5 |
| [1: {2,3}, 2:{4,5}] | Subtree(1) | All nodes                   |
| [1-2,1-3]           | Path(2,3)  | LCA=1 handled separately    |

#### Complexity

| Operation             | Time                 | Space  |
| --------------------- | -------------------- | ------ |
| Preprocessing (Euler) | $O(n)$               | $O(n)$ |
| Query Sorting         | $O(q \log q)$        | $O(q)$ |
| Processing            | $O((n + q)\sqrt{n})$ | $O(n)$ |

Mo's Algorithm on Tree is the elegant meeting point of graph traversal, offline range query, and amortized optimization, bringing sublinear query handling to complex hierarchical data.




